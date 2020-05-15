import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
from utils.loss import CrossEntropy2d, loss_calc, lr_poly, adjust_learning_rate, seg_accuracy, per_class_iu
import time
import datetime
from PIL import Image
from logger import Logger
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from Visualize import plot_embedding
import horovod.torch as hvd
import math
import warnings
import time
import pickle as pkl
from model.SVD import SVD_entropy


class Solver(object):
    def __init__(self, basemodel, C1, C2, netDFeat, loader, TargetLoader,
                 base_lr, DFeat_lr, task, num_domain, MCD,
                 optBase, optC1, optC2, optDFeat, config, args, gpu_map):
        self.args = args
        self.config = config
        self.gpu = args.gpu
        snapshot_dir = config['exp_setting']['snapshot_dir']
        log_dir = config['exp_setting']['log_dir']
        exp_name = args.exp_name
        self.snapshot_dir = os.path.join(snapshot_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)

        self.basemodel = basemodel
        self.C1 = C1
        self.C2 = C2
        self.netDFeat = netDFeat
        self.loader = loader
        self.TargetLoader = TargetLoader
        self.optBase = optBase
        self.optC1 = optC1
        self.optC2 = optC2
        self.optDFeat = optDFeat
        self.gpu_map = gpu_map
        self.gpu0 = 'cuda:0'

        self.loader_iter = iter(loader)
        self.target_iter = iter(TargetLoader)
        self.num_domain = num_domain
        self.num_source = self.num_domain-1
        self.MCD = MCD
        self.batch_size = config['train']['batch_size'][task]
        self.dataset = loader.dataset

        self.domain_label = torch.zeros(self.num_domain*self.batch_size, dtype=torch.long)
        for i in range(self.num_domain):
            self.domain_label[i*self.batch_size: (i+1)*self.batch_size] = i

        assert config['train']['GAN']['featAdv'] == 'Vanila' or \
            config['train']['GAN']['featAdv'] == 'LS' or \
            config['train']['GAN']['featAdv'] == 'WGAN_GP'
        self.featAdv_algorithm = config['train']['GAN']['featAdv']

        self.base_lr = base_lr
        self.DFeat_lr = DFeat_lr
        self.FeatAdv_coeff = config['train']['lambda']['base_model']['bloss_AdvFeat'][task]
        self.num_classes = config['data']['num_classes'][task]
        self.SVD_k = config['train']['SVD_k']
        self.SVD_ld = config['train']['SVD_ld']

        self.total_step = self.config['train']['num_steps']
        self.early_stop_step = self.config['train']['num_steps_stop']
        self.power = self.config['train']['lr_decay_power'][task]

        self.log_loss = {}
        self.log_loss['source_acc'] = []
        self.log_loss['source_loss'] = []
        self.log_loss['target_acc'] = []
        self.log_loss['SVD_entropy'] = {
            domain: [] for domain in self.dataset
        }
        self.log_loss['SVD_singular'] = {
            domain: [] for domain in self.dataset
        }
        self.log_loss['D_loss'] = []
        self.log_lr = {}
        self.log_step = 100
        self.val_step = 100
        self.tsne_step = 2000
        self.save_step = 1000 #5000
        self.logger = Logger(self.log_dir)

        self.tsne = TSNE(n_components=2, perplexity=20, init='pca', n_iter=3000)

    def train(self):
        # Broadcast parameters and optimizer state for every processes

        self.start_time = time.time()

        for i_iter in range(self.total_step):
            self.basemodel.train()
            self.C1.train()
            self.C2.train()
            self.netDFeat.train()

            #p = float(i_iter) / self.config['train']['num_steps_stop']
            #self.FeatAdv_coeff = 2. / (1. + np.exp(-6. * p)) - 1

            self._train_step(i_iter)

            if (i_iter+1) % self.val_step == 0:
                self.basemodel.eval()
                self.C1.eval()
                self.C2.eval()
                self._validation(i_iter)

            if (i_iter+1) % (10*self.val_step) == 0:
                with open(os.path.join(self.log_dir, '{}_log.pkl'.format(i_iter+1)), 'wb') as f:
                    pkl.dump(self.log_loss, f)

            if (i_iter+1) % self.tsne_step == 0:
                self._tsne(i_iter)
                self.basemodel.to(self.gpu_map['basemodel'])

            if (i_iter+1) >= self.config['train']['num_steps_stop']:
                break
                print('Training Finished')

    def _adjust_lr_opts(self, i_iter):
        self.log_lr['base'] = adjust_learning_rate(self.optBase, self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['C1'] = adjust_learning_rate(self.optC1, 10*self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['C2'] = adjust_learning_rate(self.optC2, 10*self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['DFeat'] = adjust_learning_rate(self.optDFeat, 10*self.DFeat_lr, i_iter, self.total_step, self.power)

    def _denorm(self, data):
        N, _, H, W = data.size()
        mean=torch.FloatTensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).repeat(N, 1, H, W)
        std=torch.FloatTensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).repeat(N, 1, H, W)
        return mean+data*std

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
        alpha = torch.rand(real.size(0), 1, 1, 1).to(self.gpu_map['netDFeat'])
        x_hat = (alpha * real.data + (1 - alpha) * fake.data).requires_grad_(True)
        out_src, _, _ = self.netDFeat(x_hat)

        weight = torch.ones(out_src.size()).to(self.gpu_map['netDFeat'])
        dydx = torch.autograd.grad(outputs=out_src,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return ld * torch.mean((dydx_l2norm-1)**2)

    def _D_hingeLoss(self, fake, real):
        loss = torch.mean(torch.relu(1. - real)) + torch.mean(torch.relu(1. + fake))
        return loss

    def _G_hingeLoss(self, fake):
        return - torch.mean(fake)

    def _discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def _maximum_classifier_discrepancy(self, images, labels):
        h, _ = self.basemodel(images)
        C1_feat = self.C1(h)
        C2_feat = self.C2(h)
        output_s1 = C1_feat[:self.batch_size*(self.num_domain-1)]
        output_s2 = C2_feat[:self.batch_size*(self.num_domain-1)]

        loss_s1 = nn.CrossEntropyLoss()(output_s1, labels)
        loss_s2 = nn.CrossEntropyLoss()(output_s2, labels)
        loss_s = loss_s1 + loss_s2

        output_t1 = C1_feat[self.batch_size*(self.num_domain-1):]
        output_t2 = C2_feat[self.batch_size*(self.num_domain-1):]
        loss_dis = self._discrepancy(output_t1, output_t2)
        return loss_s, loss_dis

    def _zero_grad(self):
        self.optDFeat.zero_grad()
        self.optBase.zero_grad()
        self.optC1.zero_grad()
        self.optC2.zero_grad()

    def _backprop_weighted_losses(self, lambdas, retain_graph=False):
        loss = 0
        for k in lambdas.keys():
            k_loss = getattr(self, k)
            self.log_loss[k] = k_loss.item()
            weight = lambdas[k]['cur']
            loss += k_loss.to(self.gpu0) * weight
        loss.backward(retain_graph=retain_graph)

    def _train_step(self, i_iter):
        self._adjust_lr_opts(i_iter)
        self._zero_grad()

        # -----------------------------
        # 1. Load data
        # -----------------------------

        try:
            images, labels = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            images, labels = next(self.loader_iter)

        images = Variable(images.to(torch.float))
        labels = Variable(labels.long())

        images = images.to(self.gpu0)
        labels = labels.to(self.gpu0)

        # -----------------------------
        # 2. Feedforward Basemodel
        # -----------------------------

        """ Classification and Adversarial Loss (Basemodel) """
        adv_feature, _ = self.basemodel(images)

        # -----------------------------
        # 3. Train Discriminators
        # -----------------------------
        for param in self.netDFeat.parameters():
            param.requires_grad = True

        # Train with original domain labels
        DFeatlogit = self.netDFeat(adv_feature.detach().to(self.gpu_map['netDFeat']))

        if self.featAdv_algorithm == 'Vanila':
            Dloss_AdvFeat = nn.BCEWithLogitsLoss()(DFeatlogit,
                                                  self._real_domain_label(DFeatlogit, 'Feat'))
        elif self.featAdv_algorithm == 'LS':
            Dloss_AdvFeat = nn.MSELoss()(DFeatlogit,
                                         self._real_domain_label(DFeatlogit, 'Feat'))

        Dloss_AdvFeat.backward()
        if (i_iter+1) % self.log_step == 0:
            self.log_loss['D_loss'].append(Dloss_AdvFeat.item())
        self.optDFeat.step()
        # ----------------------------
        # 4. Train Basemodel
        # ----------------------------

        for param in self.netDFeat.parameters():
            param.requires_grad = False

        # ----------------------------
        # Maximum Classifier Discrepancy
        # ----------------------------
        if self.MCD:
            loss_s, _ = self._maximum_classifier_discrepancy(images, labels)
            loss_s.backward()
            if (i_iter+1) % self.log_step == 0:
                self.log_loss['source_loss'].append(loss_s.item())
            self.optBase.step()
            self.optC1.step()
            self.optC2.step()
            self._zero_grad()

            loss_s, loss_dis = self._maximum_classifier_discrepancy(images, labels)
            loss = loss_s - loss_dis
            loss.backward()
            self.optC1.step()
            self.optC2.step()
            self._zero_grad()

            for i in range(4):
                _, loss_dis = self._maximum_classifier_discrepancy(images, labels)
                loss_dis.backward()
                self.optBase.step()
                self._zero_grad()
        else:
            h, _ = self.basemodel(images)
            C1_feat = self.C1(h)
            output_s1 = C1_feat[:self.batch_size*(self.num_domain-1)]
            loss_s1 = nn.CrossEntropyLoss()(output_s1, labels)
            loss_s1.backward()
            self.optBase.step()
            self.optC1.step()
            self._zero_grad()

        adv_feature, _ = self.basemodel(images)

        # ----------------------------
        # SVD Entropy regularization
        # ----------------------------
        SVD_en = Variable(torch.tensor(0.), requires_grad=True).to(self.gpu0)
        for d in range(self.num_domain):
            d_feature = adv_feature[d*self.batch_size: (d+1)*self.batch_size]
            en_transfer_d, en_discrim_d, singular_values = SVD_entropy(d_feature, self.SVD_k)
            total_en = en_transfer_d + en_discrim_d
            if (i_iter+1) % self.log_step == 0:
                self.log_loss['SVD_entropy'][self.dataset[d]].append(total_en.item())
                self.log_loss['SVD_singular'][self.dataset[d]].append(singular_values)
            SVD_en += self.SVD_ld * (en_transfer_d)
        SVD_en.backward()
        self.optBase.step()
        self._zero_grad()

        # ----------------------------

        adv_feature, _ = self.basemodel(images)
        DFeatlogit = self.netDFeat(adv_feature.to(self.gpu_map['netDFeat']))

        if self.featAdv_algorithm == 'Vanila':
            bloss_AdvFeat = nn.BCEWithLogitsLoss()(DFeatlogit,
                                                  self._fake_domain_label(DFeatlogit, 'Feat'))
        elif self.featAdv_algorithm == 'LS':
            bloss_AdvFeat = nn.MSELoss()(DFeatlogit,
                                         self._fake_domain_label(DFeatlogit, 'Feat'))
        bloss_AdvFeat *= self.FeatAdv_coeff
        bloss_AdvFeat.backward()
        self.optBase.step()
        # -----------------------------------------------
        # -----------------------------------------------

        if (i_iter+1) % self.log_step == 0:
            et = time.time() - self.start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]\n".format(et, i_iter+1, self.early_stop_step)
            h, _ = self.basemodel(images)
            pred = self.C1(h)
            source_pd = pred.detach().data[:self.batch_size*self.num_source].max(1)[1].cpu().numpy()
            source_lb = labels.data.cpu().numpy()
            acc = np.mean(source_pd == source_lb)
            if (i_iter+1) % self.log_step == 0:
                self.log_loss['source_acc'].append(acc.item())
            log += "\nAcc: {:.2f}".format(acc.item()*100)
            print(log)

        if (i_iter+1) % self.save_step == 0:
            print('taking snapshot ...')
            torch.save({
                'basemodel': self.basemodel.state_dict(),
                'netDFeat': self.netDFeat.state_dict(),
            }, osp.join(self.snapshot_dir, 'pretrain_'+str(i_iter+1)+'.pth'))

    def _validation(self, i_iter):
        val_iter = 0
        size = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0

        for i in range(len(self.TargetLoader) // self.batch_size):
            with torch.no_grad():
                # To be modified
                target_images, target_labels = next(self.target_iter)
                val_iter += 1

                target_images = Variable(target_images.to(torch.float).detach())
                target_labels = Variable(target_labels.long().detach())

                target_images = target_images.to(self.gpu0)
                target_labels = target_labels.to(self.gpu0)

                h, _ = self.basemodel(target_images)
                if torch.isnan(h).any(): raise ValueError

                output1 = self.C1(h)
                output2 = self.C2(h)
                output_ensemble = output1 + output2
                pred1 = output1.data.max(1)[1]
                pred2 = output2.data.max(1)[1]
                pred_ensemble = output_ensemble.data.max(1)[1]
                k = target_labels.data.size()[0]
                correct1 += pred1.eq(target_labels.data).cpu().sum()
                correct2 += pred2.eq(target_labels.data).cpu().sum()
                correct3 += pred_ensemble.eq(target_labels.data).cpu().sum()
                size += k

        acc1 = 100. * correct1 / size
        acc2 = 100. * correct2 / size
        acc3 = 100. * correct3 / size

        info_str = 'Iteration {}: acc1:{:0.2f} acc2:{:0.2f} acc_ensemble:{:0.2f}'.format(i_iter+1,
                                                                                         acc1, acc2, acc3)
        self.log_loss['target_acc'].append(acc3.item())
        print(info_str)

        with open(os.path.join(self.log_dir, 'val_result.txt'), 'a') as f:
            f.write(info_str+'\n')
            f.close()

    def _tsne(self, i_iter):
        # Plot t-SNE of hidden feature
        source_images1, source_labels1 = next(self.loader_iter)
        target_images1, target_labels1 = next(self.target_iter)
        tsne_images = torch.cat([source_images1[:self.batch_size*self.num_source],
                                 target_images1], dim=0).to(torch.float)
        tsne_labels = torch.cat([source_labels1[:self.batch_size*self.num_source].long(),
                                 target_labels1.long()], dim=0)
        tsne_domain = self.domain_label

        sample_path = os.path.join(self.log_dir, '{}-tSNE.jpg'.format(i_iter+1))
        _, h = self.basemodel.cpu()(tsne_images.cpu())
        tsne = self.tsne.fit_transform(h.data.cpu().numpy())
        plot_embedding(tsne, tsne_labels.data.cpu().numpy(),
                       tsne_domain.data.cpu().numpy(), sample_path)
        print('{} iter; saved t-SNE'.format(i_iter))

