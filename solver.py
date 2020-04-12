import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
from utils.loss import CrossEntropy2d, loss_calc, lr_poly, adjust_learning_rate, seg_accuracy, per_class_iu
from utils.visualize import colorize_mask, save_prediction
import time
import datetime
from PIL import Image
from logger import Logger
from torchvision.utils import save_image
import horovod.torch as hvd
import math
import warnings
import time


class Solver(object):
    def __init__(self, base, basemodel, netDImg, netDFeat, netG, loader, TargetLoader,
                 base_lr, DImg_lr, DFeat_lr, G_lr,
                 optBase, optDImg, optDFeat, optG, config, args, gpu_map):
        self.args = args
        self.config = config
        self.gpu = args.gpu
        snapshot_dir = config['exp_setting']['snapshot_dir']
        log_dir = config['exp_setting']['log_dir']
        exp_name = args.exp_name
        self.snapshot_dir = os.path.join(snapshot_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)

        self.base = base
        self.basemodel = basemodel
        self.netDImg = netDImg
        self.netDFeat = netDFeat
        self.netG = netG
        self.loader = loader
        self.TargetLoader = TargetLoader
        self.optBase = optBase
        self.optDImg = optDImg
        self.optDFeat = optDFeat
        self.optG = optG
        self.gpu_map = gpu_map
        self.gpu0 = 'cuda:0'

        self.loader_iter = iter(loader)
        self.target_iter = iter(TargetLoader)
        self.num_domain = 1+len(config['data']['source'])
        self.num_source = self.num_domain-1
        self.batch_size = config['train']['batch_size']

        self.domain_label = torch.zeros(self.num_domain*self.batch_size, dtype=torch.long)
        for i in range(self.num_domain):
            self.domain_label[i*self.batch_size: (i+1)*self.batch_size] = i

        assert config['train']['GAN']['model'] == 'WGAN_GP'
        self.Dgan_loss = torch.nn.MSELoss()
        #self.Dgan_loss = torch.nn.BCEWithLogitsLoss()

        self.real_label = 0
        self.fake_label = 1

        self.base_lr = base_lr
        self.DImg_lr = DImg_lr
        self.DFeat_lr = DFeat_lr
        self.G_lr = G_lr
        self.num_classes = config['data']['num_classes']

        self.total_step = self.config['train']['num_steps']
        self.early_stop_step = self.config['train']['num_steps_stop']
        self.power = self.config['train']['lr_decay_power']

        self.log_loss = {}
        self.log_lr = {}
        self.log_step = 1000
        self.sample_step = 1000
        self.val_step = 1000
        self.save_step = 1000 #5000
        self.logger = Logger(self.log_dir)

        loss_lambda = {}
        for k in self.config['train']['lambda'].keys():
            coeff = self.config['train']['lambda'][k]
            loss_lambda[k] = {}

            for sub_k in coeff.keys():
                init = coeff[sub_k]['init']
                final = coeff[sub_k]['final']
                step = coeff[sub_k]['step']

                loss_lambda[k][sub_k] = {}
                loss_lambda[k][sub_k]['cur'] = init
                loss_lambda[k][sub_k]['inc'] = (final-init)/step
                loss_lambda[k][sub_k]['final'] = final
        self.loss_lambda = loss_lambda

    def train(self):
        # Broadcast parameters and optimizer state for every processes

        self.start_time = time.time()

        for i_iter in range(self.total_step):
            self.basemodel.train()
            self.netDFeat.train()
            self.netDImg.train()
            self.netG.train()
            self._train_step(i_iter)

            if (i_iter+1) % self.val_step == 0:
                self._validation(i_iter)

            # update lambda
            for k in self.loss_lambda.keys():
                for sub_k in self.loss_lambda[k].keys():
                    if self.loss_lambda[k][sub_k]['cur'] < self.loss_lambda[k][sub_k]['final']:
                        self.loss_lambda[k][sub_k]['cur'] += self.loss_lambda[k][sub_k]['inc']

            if (i_iter+1) >= self.config['train']['num_steps_stop']:
                break
                print('Training Finished')

    def _adjust_lr_opts(self, i_iter):
        self.log_lr['base'] = adjust_learning_rate(self.optBase, self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['DImg'] = adjust_learning_rate(self.optDImg, self.DImg_lr, i_iter, self.total_step, self.power)
        self.log_lr['DFeat'] = adjust_learning_rate(self.optDFeat, self.DFeat_lr, i_iter, self.total_step, self.power)
        self.log_lr['G'] = adjust_learning_rate(self.optG, self.G_lr, i_iter, self.total_step, self.power)

    def _broadcast_param_opt(self):
        hvd.broadcast_parameters(self.basemodel.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.netDFeat.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.netDImg.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.netG.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optBase, root_rank=0)
        hvd.broadcast_optimizer_state(self.optDFeat, root_rank=0)
        hvd.broadcast_optimizer_state(self.optDImg, root_rank=0)
        hvd.broadcast_optimizer_state(self.optG, root_rank=0)

    def _denorm(self, data):
        N, _, H, W = data.size()
        mean=torch.FloatTensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).repeat(N, 1, H, W)
        std=torch.FloatTensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).repeat(N, 1, H, W)
        return mean+data*std

    def _fixed_test_domain_label(self, num_sample):
        fixed_label = []
        for i in range(self.num_domain):
            fixed_label.append([i]*num_sample)

        fixed_label = torch.LongTensor(np.array(fixed_label))
        return fixed_label

    def _fake_domain_label(self, tensor, model):
        if type(tensor).__module__ == np.__name__:
            tensor = torch.tensor(tensor)
        ones = torch.ones_like(tensor, dtype=torch.float)
        for i in range(self.num_domain):
            ones[self.batch_size*i: self.batch_size*(i+1), i] = 0.
        return ones.to(self.gpu_map['netD{}'.format(model)])

    def _real_domain_label(self, tensor, model):
        if type(tensor).__module__ == np.__name__:
            tensor = torch.tensor(tensor)
        zeros = torch.zeros_like(tensor, dtype=torch.float)
        for i in range(self.num_domain):
            zeros[self.batch_size*i: self.batch_size*(i+1), i] = 1.
        return zeros.to(self.gpu_map['netD{}'.format(model)])

    def _gradient_penalty(self, real, fake, ld):
        alpha = torch.rand(real.size(0), 1, 1, 1).to(self.gpu_map['netDImg'])
        x_hat = (alpha * real.data + (1 - alpha) * fake.data).requires_grad_(True)
        out_src, _, _ = self.netDImg(x_hat)

        weight = torch.ones(out_src.size()).to(self.gpu_map['netDImg'])
        dydx = torch.autograd.grad(outputs=out_src,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return ld * torch.mean((dydx_l2norm-1)**2)

    def _save_prediction(self, path, pred, lb):
        pd_col = colorize_mask(pred)
        lb_col = colorize_mask(lb)
        save_prediction([pd_col, lb_col], path)

    def _aux_semantic_loss(self, aux_logit_5D, label):
        # aux_logit_5D: (num_source, batch, c, w, h)
        # aux_logit = aux_logit_5D.permute(1, 0, 2, 3, 4)
        aux_logit = aux_logit_5D.permute(1, 0, 2)
        N, S, C = aux_logit.size()
        aux_logit = aux_logit.view(N, S, C)

        aux_logit_source = aux_logit[ :self.num_source*self.batch_size]
        mask = self.domain_label[ :self.num_source*self.batch_size]
        aux_logit_for_each_target_D = torch.cat(
            [aux_logit_source[i, source_domain].unsqueeze(0) for i, source_domain in enumerate(mask)],
            dim=0)

        return nn.CrossEntropyLoss()(aux_logit_for_each_target_D, label)

    def _netVGG(self, concat1, concat2, concat3, concat4, feature, domain_label):
        concat1 = concat1.to(self.gpu_map['netG_2'])
        concat2 = concat2.to(self.gpu_map['netG_2'])
        concat3 = concat3.to(self.gpu_map['netG'])
        concat4 = concat4.to(self.gpu_map['netG'])
        feature = feature.to(self.gpu_map['netG'])
        domain_label = domain_label.to(self.gpu_map['netG'])
        return self.netG(concat1, concat2, concat3, concat4, feature, domain_label)

    def _backprop_weighted_losses(self, lambdas, aux_over_ths, retain_graph=False):
        if not aux_over_ths:
            for key in lambdas.keys():
                if 'aux' in key: lambdas[key]['cur']=0.

        loss = 0
        for k in lambdas.keys():
            k_loss = getattr(self, k)
            self.log_loss[k] = k_loss.item()
            weight = lambdas[k]['cur']
            loss += k_loss.to(self.gpu0) * weight
        loss.backward(retain_graph=retain_graph)

    def _train_step(self, i_iter):
        self._adjust_lr_opts(i_iter)

        self.optDFeat.zero_grad()
        self.optDImg.zero_grad()
        self.optBase.zero_grad()

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
        # 1. Load data
        # -----------------------------

        try:
            images, labels = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            images, labels = next(self.loader_iter)

        images = Variable(images)
        labels = Variable(labels.long())

        rand_idx = torch.randperm(self.domain_label.size(0))
        self.shuffled_domain_label = self.domain_label[rand_idx]

        images = images.to(self.gpu0)
        labels = labels.to(self.gpu0)
        self.domain_label = self.domain_label.to(self.gpu_map['netG'])
        self.shuffled_domain_label = self.shuffled_domain_label.to(self.gpu_map['netG'])

        # -----------------------------
        # 2. Feedforward Basemodel and netG
        # -----------------------------

        """ Classification and Adversarial Loss (Basemodel) """
        feature, pred = self.basemodel(images)
        if self.base == 'VGG':
            concat1, concat2, concat3, concat4, concat5, Dfeature = feature

        """ Idt, Fake, Cycle, DCls, Semantic Loss (Generator) """
        if self.base == 'ResNet':
            label_onehot = torch.cat([
                torch.eye(self.num_classes+1)[labels],
                torch.eye(self.num_classes+1)[-1].unsqueeze(0).repeat(self.batch_size, 1)], dim=0).to(self.gpu_map['netG'])
            trsfakeImg = self.netG(feature.to(self.gpu_map['netG']), self.shuffled_domain_label, label_onehot)
            cycFeat, _ = self.basemodel(trsfakeImg.to(self.gpu0))
            cycfakeImg = self.netG(cycFeat.to(self.gpu_map['netG']), self.domain_label, label_onehot)
        elif self.base == 'VGG':
            trsfakeImg = self._netVGG(concat1, concat2, concat3, concat4, concat5, self.shuffled_domain_label)
            cycFeat, _ = self.basemodel(trsfakeImg.to(self.gpu0))
            concat1, concat2, concat3, concat4, concat5, _ = cycFeat
            cycfakeImg = self._netVGG(concat1, concat2, concat3, concat4, concat5, self.domain_label)

        # -----------------------------
        # 3. Train Discriminators
        # -----------------------------
        for param in self.netDImg.parameters():
            param.requires_grad = True
        for param in self.netDFeat.parameters():
            param.requires_grad = True

        """FeatD"""
        # Train with original domain labels
        DFeatlogit = self.netDFeat(feature.detach().to(self.gpu_map['netDFeat']))
        Dloss_AdvFeat = self.Dgan_loss(DFeatlogit,
                                       self._real_domain_label(DFeatlogit, 'Feat'))
        Dloss_AdvFeat.backward()
        self.log_loss['Dloss_AdvFeat'] = Dloss_AdvFeat.item()

        """ImgD"""
        fake_logit, _, _ = self.netDImg(trsfakeImg.detach().to(self.gpu_map['netDImg']))
        self.Dloss_fake = torch.mean(fake_logit,
                                    Variable(torch.FloatTensor(fake_logit.data.size()).fill_(self.fake_label))
                                    .cuda(self.gpu_map['netDImg']))
        real_logit, dcls_logit, aux_logit = self.netDImg(images.to(self.gpu_map['netDImg']))
        self.Dloss_real = - torch.mean(real_logit,
                                        Variable(torch.FloatTensor(real_logit.data.size()).fill_(self.real_label))
                                        .cuda(self.gpu_map['netDImg']))
        self.Dloss_dcls = nn.CrossEntropyLoss()(dcls_logit, self.domain_label.to(self.gpu_map['netDImg']))
        self.Dloss_auxsem = self._aux_semantic_loss(aux_logit.to(self.gpu0), labels)

        self.Dloss_gp = self._gradient_penalty(real=images.to(self.gpu_map['netDImg']),
                                               fake=trsfakeImg.detach().to(self.gpu_map['netDImg']),
                                               ld=self.config['train']['GAN']['GP'])
        self.Dloss_fakereal = self.Dloss_fake + self.Dloss_real + self.Dloss_gp
        # At this moment we don't care about semantic loss of target data

        self._backprop_weighted_losses(self.loss_lambda['netD'],
                                       aux_over_ths=True,
                                       retain_graph=True)
        self.optDFeat.step()
        self.optDImg.step()
        # ----------------------------
        # 4. Train Basemodel
        # ----------------------------

        for param in self.netDImg.parameters():
            param.requires_grad = False
        for param in self.netDFeat.parameters():
            param.requires_grad = False

        self.bloss_Clf = nn.CrossEntropyLoss()(pred[ :self.num_source*self.batch_size].to(self.gpu0),
                                               labels)

        DFeatlogit = self.netDFeat(feature.to(self.gpu_map['netDFeat']))
        self.bloss_AdvFeat = self.Dgan_loss(DFeatlogit,
                                            self._fake_domain_label(DFeatlogit, 'Feat'))

        retain = True if (i_iter+1) % self.config['train']['GAN']['n_critic'] == 0 else False

        self._backprop_weighted_losses(self.loss_lambda['classification'],
                                       aux_over_ths=True,
                                       retain_graph=retain)
        self.optBase.step()

        # ----------------------------
        # 5. Train netG
        # ----------------------------


        if (i_iter+1) % self.config['train']['GAN']['n_critic'] == 0:
            self.optG.zero_grad()
            self.optBase.zero_grad()

            fake_logit, dcls_logit, aux_logit = self.netDImg(trsfakeImg.to(self.gpu_map['netDImg']))
            self.bGloss_fake = - torch.mean(fake_logit,
                                        Variable(torch.FloatTensor(fake_logit.data.size()).fill_(self.real_label))
                                        .cuda(self.gpu_map['netDImg']))

            self.bGloss_cyc = torch.mean(torch.abs(images - cycfakeImg.to(self.gpu0)))
            self.bGloss_dcls = nn.CrossEntropyLoss()(dcls_logit,
                                                    self.shuffled_domain_label.to(self.gpu_map['netDImg']))
            fake_c = self.loss_lambda['base_model_netG']['bGloss_fake']['cur']
            cyc_c = self.loss_lambda['base_model_netG']['bGloss_cyc']['cur']
            dcls_c = self.loss_lambda['base_model_netG']['bGloss_dcls']['cur']

            # Let Generator free from semantic loss
            self.bGloss = fake_c*self.bGloss_fake.to(self.gpu0) + cyc_c*self.bGloss_cyc.to(self.gpu0)
            self.bGloss.backward(retain_graph=True)
            self.optG.step()
            self.optBase.step()

            self.optG.zero_grad()
            self.Gloss = dcls_c*self.bGloss_dcls.to(self.gpu0)
            self.Gloss.backward(retain_graph=True)
            self.optG.step()

            # Feature extractor should be dependent to domain-related gradient
            self.optBase.zero_grad()
            aux_c = self.loss_lambda['base_model_netG']['bGloss_auxsem']['cur'] if self.Dloss_auxsem < self.config['train']['aux_sem_thres'] else 0.
            self.bloss_auxsem = aux_c*self._aux_semantic_loss(aux_logit.to(self.gpu0), labels)
            self.bloss_auxsem.backward(retain_graph=False)
            self.optBase.step()


        # -----------------------------------------------
        # -----------------------------------------------


        if (i_iter+1) % self.log_step == 0:
            et = time.time() - self.start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]\n".format(et, i_iter+1, self.early_stop_step)
            for tag, value in self.log_loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            histList = np.zeros((self.num_classes, self.num_classes))
            source_pd = pred.detach().data[:self.batch_size*self.num_source].max(1)[1].cpu().numpy()
            source_lb = labels.data.cpu().numpy()
            acc = np.mean(source_pd == source_lb)

            log += "\nAcc: {:.2f}".format(acc.item()*100)
            print(log)

        if self.config['exp_setting']['use_tensorboard']:
            if (i_iter+1) % self.log_step == 0:
                for tag, value in self.log_loss.items():
                    category, name = tag.split('_')[0], tag.split('_')[1]
                    self.logger.scalar_summary('{}/{}'.format(category, name), value, i_iter+1)
                for tag, value in self.log_lr.items():
                    self.logger.scalar_summary('lr/{}'.format(tag), value, i_iter+1)

        if (i_iter+1) % self.sample_step == 0:
            with torch.no_grad():
                image_fixed = images[[i*self.batch_size for i in range(self.num_domain)]]
                label_fixed = labels[[i*self.batch_size for i in range(self.num_domain-1)]]
                label_fixed_onehot = torch.cat([
                    torch.eye(self.num_classes+1)[label_fixed],
                    torch.eye(self.num_classes+1)[-1].unsqueeze(0)])

                image_fake_list = [image_fixed]
                for d_fixed in self._fixed_test_domain_label(num_sample=self.num_domain):
                    feature, _ = self.basemodel(image_fixed.to(self.gpu0))
                    if self.base == 'VGG':
                        concat1, concat2, concat3, concat4, concat5, _ = feature
                        image_fake = self._netVGG(concat1, concat2, concat3, concat4, concat5,
                                                        d_fixed.to(self.gpu_map['netG']))
                    else:
                        image_fake = self.netG(feature.to(self.gpu_map['netG']),
                                               d_fixed.to(self.gpu_map['netG']),
                                               label_fixed_onehot.to(self.gpu_map['netG']))
                    image_fake_list.append(image_fake)
                image_concat = torch.cat(image_fake_list, dim=3)
                sample_path = os.path.join(self.log_dir, '{}-FixTrsimages.jpg'.format(i_iter+1))
                save_image(self._denorm(image_concat.data.cpu()), sample_path, nrow=self.num_domain, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

        if (i_iter+1) % self.save_step == 0:
            print('taking snapshot ...')
            torch.save({
                'basemodel': self.basemodel.state_dict(),
                'netG': self.netG.state_dict(),
                'netDImg': self.netDImg.state_dict(),
                'netDFeat': self.netDFeat.state_dict(),
            }, osp.join(self.snapshot_dir, 'pretrain_'+str(i_iter+1)+'.pth'))

    def _validation(self, i_iter):
        accList = []
        histList = np.zeros((self.num_classes, self.num_classes))
        val_iter = 0

        for i in range(len(self.TargetLoader) // self.batch_size):
            with torch.no_grad():
                # To be modified
                target_images, target_labels = next(self.target_iter)
                val_iter += 1

                target_images = Variable(target_images.detach())
                target_labels = Variable(target_labels.long().detach())

                target_images = target_images.to(self.gpu0)
                target_labels = target_labels.to(self.gpu0)

                _, target_pred = self.basemodel(target_images)

                if torch.isnan(target_pred).any(): raise ValueError
                target_pred = target_pred.max(1)[1].data.cpu().numpy()
                target_labels = target_labels.data.cpu().numpy()
                acc = np.mean(target_pred == target_labels)
                accList.append(acc.item() * 100)

        info_str = 'Iteration {}: acc:{:0.2f}'.format(i_iter+1,
                                                      np.mean(accList))
        self.logger.scalar_summary('metrics/val_acc', np.mean(accList), i_iter+1)
        print(info_str)

