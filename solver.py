import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.autograd import Variable
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from utils.loss import CrossEntropy2d, loss_calc, lr_poly, adjust_learning_rate
import time
import datetime
from logger import Logger
from torchvision.utils import save_image

class Solver(object):
    def __init__(self, basemodel, netDImg, netDSem, netDFeat, netG, loader,
                 optBase, optDImg, optDSem, optDFeat, optG, config, args):
        self.args = args
        self.config = config
        self.gpu = args.gpu
        snapshot_dir = config['snapshot_dir']
        log_dir = config['log_dir']
        exp_name = args.exp_name
        self.snapshot_dir = os.path.join(snapshot_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)

        self.basemodel = basemodel
        self.netDImg = netDImg
        self.netDSem = netDSem
        self.netDFeat = netDFeat
        self.netG = netG
        self.loader = loader
        self.optBase = optBase
        self.optDImg = optDImg
        self.optDSem = optDSem
        self.optDFeat = optDFeat
        self.optG = optG
        self.loader_iter = iter(loader)
        self.num_domain = 1+len(config['data']['source'])
        self.num_source = self.num_domain-1
        self.batch_size = config['data']['batch_size']

        self.domain_label = torch.zeros(self.num_domain*self.batch_size, dtype=torch.float)
        for i in range(self.num_domain):
            self.domain_label[i*self.batch_size: (i+1)*self.batch_size] = i

        if config['train']['GAN'] == 'Vanilla':
            self.gan_loss= torch.nn.BCEWithLogitsLoss()
        elif config['train']['GAN'] == 'LS':
            self.GAN_loss = torch.nn.MSELoss()

        input_size = self.config['data']['input_size']
        w, h = map(int, input_size)
        self.input_size = (w, h)

        self.real_label = 0
        self.fake_label = 1

        self.base_lr = config['model']['train']['base_model']['lr']
        self.D_lr = config['model']['train']['netD']['lr']
        self.G_lr = config['model']['train']['netG']['lr']

        self.total_step = self.config['train']['num_steps']
        self.early_stop_step = self.config['train']['num_steps_stop']
        self.power = self.config['train']['lr_decay_power']

        self.log_loss = {}
        self.log_lr = {}
        self.log_step = 1000
        self.sample_step = 1000
        self.save_step = 5000
        self.logger = Logger(self.log_dir)

    def train(self):
        self.basemodel.train()
        self.netDFeat.train()
        self.netDImg.train()
        self.netDSem.train()
        self.netG.train()
        self.start_time = time.time()

        for i_iter in range(self.total_step):
            self._train_step(i_iter)

    def _adjust_lr_opts(self, i_iter):
        self.log_lr['base'] = adjust_learning_rate(self.optBase, self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['DImg'] = adjust_learning_rate(self.optDImg, self.D_lr, i_iter, self.total_step, self.power)
        self.log_lr['DSem'] = adjust_learning_rate(self.optDSem, self.D_lr, i_iter, self.total_step, self.power)
        self.log_lr['DFeat'] = adjust_learning_rate(self.optDFeat, self.D_lr, i_iter, self.total_step, self.power)
        self.log_lr['G'] = adjust_learning_rate(self.optG, self.G_lr, i_iter, self.total_step, self.power)

    def _zero_buffer(self):
        self.optBase.zero_grad()
        self.optDFeat.zero_grad()
        self.optDImg.zero_grad()
        self.optDSem.zero_grad()
        self.optG.zero_grad()

    def _update_opts(self):
        self.optBase.step()
        self.optDFeat.step()
        self.optDImg.step()
        self.optDSem.step()
        self.optG.step()

    def _denorm(self, data):
        mean=torch.tensor([0.485, 0.456, 0.406])
        std=torch.tensor([0.229, 0.224, 0.225])
        return 255.*(mean+(data*std))

    def _fixed_test_domain_label(self, num_sample):
        return torch.tensor(np.array([[i]*num_sample] for i in range(self.num_domain)))

    def _block_or_release_updateD(self, requires_grad=False):
        Ds = [self.netDFeat, self.netDImg, self.netDSem]
        for D in Ds:
            for param in D.parameters():
                param.requires_grad = requires_grad

    def _fake_domain_label(self, tensor):
        if type(tensor).__module__ == np.__name__:
            tensor = torch.tensor(tensor)
        ones = torch.ones_like(tensor, dtype=torch.float)
        for i in range(self.num_domain):
            ones[self.batch_size*i: self.batch_size*(i+1), i] = 0.
        return ones

    def _real_domain_label(self, tensor):
        if type(tensor).__module__ == np.__name__:
            tensor = torch.tensor(tensor)
        zeros = torch.zeros_like(tensor, dtype=torch.float)
        for i in range(self.num_domain):
            zeros[self.batch_size*i: self.batch_size*(i+1), i] = 1.
        return zeros

    def _interp(self, x):
        interp = nn.Upsample(size=(self.input_size[0], self.input_size[1]), mode='bilinear')
        return interp(x)

    def _aux_semantic_loss(self, aux_logit, label):
        aux_logit = aux_logit[ :self.num_source*self.batch_size]
        aux_logit_for_each_target_D = aux_logit[:, self.shuffled_domain_label] # i-th prediction
        return loss_calc(aux_logit_for_each_target_D, label, self.gpu)

    def _weight_losses(self, lambdas, aux_over_ths):
        if not aux_over_ths:
            for key in lambdas.keys():
                if 'aux' in key: lambdas[key]=0.

        loss = 0
        for k, weight in lambdas.items():
            k_loss = getattr(self, k)
            self.log_loss[k] = k_loss.item()
            loss += k_loss * weight
        return loss

    def _train_step(self, i_iter):
        self._adjust_lr_opts(i_iter)
        self._zero_buffer()

        """
        - Basemodel/Generator
            - BaseModel
                1. Classification Loss (For source only)
                2. AdvLoss
                    - Take a feature/predicts(mask), feed into FeatD and ImgD
            - (+ Generator)
                3. IdtLoss
                    - Feedforward with original domain label
                    - Should return itself (CycleGAN)
                4. FakeLoss
                    - Feedforward with shuffled domain label
                    - Fool D
                5. CycleLoss
                    - L1 Loss for cycle consistency
                6. DClsLoss
                    - Domain prediction loss
                7. SemanticLoss
                    - (Cycada, ICML 2018)

        - Discriminator
            - Detach all the needed losses
            - FeatD
                - (#Domain - 1) independent Adversarial losses
            - ImgD
                - Adv, DCls, AuxClf loss
            - SemD
                - (#Domain - 1) independent Adversarial losses
        """

        # -----------------------------
        # 1. Train Basemodel and netG
        # -----------------------------
        self._block_or_release_updateD(requires_grad=False)

        images, labels = next(self.loader_iter)
        images = Variable(images).cuda(self.gpu)
        labels = Variable(labels.long()).cuda(self.gpu)

        rand_idx = torch.randperm(self.domain_label.size(0))
        self.shuffled_domain_label = self.domain_label[rand_idx]

        """ Classification and Adversarial Loss (Basemodel) """
        feature, pred = self.basemodel(images)
        pred = self._interp(pred)
        self.bloss_Clf = loss_calc(pred[ :self.num_source*self.batch_size], labels, self.gpu)

        DFeatlogit = self.netDFeat(feature)
        DSemlogit = self.netDSem(pred)
        self.bloss_AdvFeat = self.gan_loss(nn.Sigmoid(DFeatlogit), self._fake_domain_label(DFeatlogit))
        self.bloss_AdvImg = self.gan_loss(nn.Sigmoid(DSemlogit), self._fake_domain_label(DSemlogit))

        """ Idt, Fake, Cycle, DCls, Semantic Loss (Generator) """
        idtfakeImg = self.netG(feature, self.domain_label)
        trsfakeImg = self.netG(feature, self.shuffled_domain_label)
        cycFeat, _ = self.basemodel(trsfakeImg)
        cycfakeImg = self.netG(cycFeat, self.domain_label)

        fake_logit, dcls_logit, aux_logit = self.netDImg(trsfakeImg)
        self.bGloss_fake = self.gan_loss(fake_logit,
                                    Variable(torch.FloatTensor(fake_logit.data.size()).fill_(self.real_label))
                                    .cuda(self.gpu))
        self.bGloss_idt = torch.mean(torch.abs(images - idtfakeImg))
        self.bGloss_cyc = torch.mean(torch.abs(images - cycfakeImg))
        self.bGloss_dcls = nn.CrossEntropyLoss(dcls_logit, self.shuffled_domain_label)
        self.bGloss_auxsem = self._aux_semantic_loss(aux_logit, labels)  # At this moment we don't care about target data

        bGloss = self._weight_losses(self.config['train']['lambda']['base_model_netG'],
                            self.bGloss_auxsem > self.config['train']['lambda']['aux_sem_ths'])
        bGloss.backward()


        # -----------------------------
        # 2. Train Discriminators
        # -----------------------------

        self._block_or_release_updateD(requires_grad=True)

        """ (FeatD, SemD) """

        # Train with original domain labels
        DFeatlogit = self.netDFeat(feature.detach())
        DSemlogit = self.netDSem(pred.detach())
        Dloss_AdvFeat = self.gan_loss(nn.Sigmoid(DFeatlogit), self._real_domain_label(DFeatlogit))
        Dloss_AdvSem = self.gan_loss(nn.Sigmoid(DSemlogit), self._real_domain_label(DSemlogit))
        Dloss_AdvFeat.backward()
        Dloss_AdvSem.backward()
        self.log_loss['Dloss_AdvFeat'] = Dloss_AdvFeat.item()
        self.log_loss['Dloss_AdvSem'] = Dloss_AdvSem.item()

        """ (ImgD) """
        fake_logit, _, _ = self.netDImg(trsfakeImg.detach())
        self.Dloss_fake = self.gan_loss(fake_logit,
                                        Variable(torch.FloatTensor(fake_logit.data.size()).fill_(self.fake_label))
                                        .cuda(self.gpu))
        self.Dloss_dcls = nn.CrossEntropyLoss(dcls_logit, self.shuffled_domain_label)
        real_logit, dcls_logit, aux_logit = self.netDImg(images)
        self.Dloss_real = self.gan_loss(real_logit,
                                        Variable(torch.FloatTensor(real_logit.data.size()).fill_(self.real_label))
                                        .cuda(self.gpu))
        self.Dloss_dcls = nn.CrossEntropyLoss(dcls_logit, self.domain_label)
        self.Dloss_auxsem = self._aux_semantic_loss(aux_logit, labels)  # At this moment we don't care about target data

        Dloss_AdvImg = self._weight_losses(self.config['train']['lambda']['netD'], aux_over_ths=True)
        Dloss_AdvImg.backward()


        # -----------------------------
        # 3. Update
        # -----------------------------

        self._update_opts()

        # -----------------------------------------------
        # -----------------------------------------------

        if (i_iter+1) % self.log_step == 0:
            et = time.time() - self.start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i_iter+1, self.num_iters)
            for tag, value in self.log_loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

            # TODO: Compute mIOU for target domain data

        if self.config['exp_setting']['use_tensorboard']:
            for tag, value in loss.items():
                self.logger.scalar_summary(tag, value, i_iter+1)

        if (i_iter+1) % self.sample_step == 0:
            with torch.no_grad():
                image_fixed = images[[i*self.batch_size for i in range(self.num_domain)]]
                image_fake_list = [image_fixed]
                for d_fixed in self._fixed_test_domain_label(num_sample=self.num_domain):
                    feature, _ = self.basemodel(image_fixed)
                    image_fake = self.netG(feature, d_fixed)
                    image_fake_list.append(image_fake)
                image_concat = torch.cat(image_fake_list, dim=3)
                sample_path = os.path.join(self.log_dir, '{}-images.jpg'.format(i_iter+1))
                save_image(self._denorm(image_concat.data.cpu()), sample_path, nrow=self.num_domain, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

        if (i_iter+1) % self.save_step == 0:
            print('taking snapshot ...')
            torch.save(self.basemodel.state_dict(), osp.join(self.snapshot_dir, 'basemodel_'+str(i_iter+1)+'.pth'))
            torch.save(self.netG.state_dict(), osp.join(self.snapshot_dir, 'netG_'+str(i_iter+1)+'.pth'))
            torch.save(self.netDImg.state_dict(), osp.join(self.snapshot_dir, 'netDImg_'+str(i_iter+1)+'.pth'))
            torch.save(self.netDFeat.state_dict(), osp.join(self.snapshot_dir, 'netDFeat_'+str(i_iter+1)+'.pth'))
            torch.save(self.netDSem.state_dict(), osp.join(self.snapshot_dir, 'netDSem_'+str(i_iter+1)+'.pth'))

        if (i_iter+1) >= self.config['train']['num_steps_stop']:
            break

