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


class Solver(object):
    def __init__(self, basemodel, C1, C2, netDImg, netDFeat, netG, loader, TargetLoader,
                 base_lr, DImg_lr, DFeat_lr, G_lr,
                 optBase, optC1, optC2, optDImg, optDFeat, optG, config, args, gpu_map):
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
        self.netDImg = netDImg
        self.netDFeat = netDFeat
        self.netG = netG
        self.loader = loader
        self.TargetLoader = TargetLoader
        self.optBase = optBase
        self.optC1 = optC1
        self.optC2 = optC2
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

        assert config['train']['GAN']['main'] == 'WGAN_GP'

        if config['train']['GAN']['featAdv'] == 'Vanila':
            self.FeatAdv_loss = torch.nn.BCEWithLogitsLoss()
        elif config['train']['GAN']['featAdv'] == 'LS':
            self.FeatAdv_loss = torch.nn.MSELoss()

        self.real_label = 0
        self.fake_label = 1

        self.base_lr = base_lr
        self.DImg_lr = DImg_lr
        self.DFeat_lr = DFeat_lr
        self.G_lr = G_lr
        self.num_classes = config['data']['num_classes']
        self.task = self.config['data']['task']

        self.total_step = self.config['train']['num_steps']
        self.early_stop_step = self.config['train']['num_steps_stop']
        self.power = self.config['train']['lr_decay_power']
        self.partial = self.config['train']['partial'] # Whether G uses full skip-connection

        self.log_loss = {}
        self.log_lr = {}
        self.log_step = 100
        self.sample_step = 100
        self.val_step = 100
        self.tsne_step = 2000
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
        self.tsne = TSNE(n_components=2, perplexity=20, init='pca', n_iter=3000)

    def train(self):
        # Broadcast parameters and optimizer state for every processes

        self.start_time = time.time()

        for i_iter in range(self.total_step):
            self.basemodel.train()
            self.C1.train()
            self.C2.train()
            self.netDFeat.train()
            self.netDImg.train()
            self.netG.train()
            self._train_step(i_iter)

            if (i_iter+1) % self.val_step == 0:
                self.basemodel.eval()
                self.C1.eval()
                self.C2.eval()
                self._validation(i_iter)

            if (i_iter+1) % self.tsne_step == 0:
                self._tsne(i_iter)
                self.basemodel.to(self.gpu_map['basemodel'])

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
        self.log_lr['C1'] = adjust_learning_rate(self.optC1, self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['C2'] = adjust_learning_rate(self.optC2, self.base_lr, i_iter, self.total_step, self.power)
        self.log_lr['DImg'] = adjust_learning_rate(self.optDImg, self.DImg_lr, i_iter, self.total_step, self.power)
        self.log_lr['DFeat'] = adjust_learning_rate(self.optDFeat, self.DFeat_lr, i_iter, self.total_step, self.power)
        self.log_lr['G'] = adjust_learning_rate(self.optG, self.G_lr, i_iter, self.total_step, self.power)

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

    def _D_hingeLoss(self, fake, real):
        loss = torch.mean(torch.relu(1. - real)) + torch.mean(torch.relu(1. + fake))
        return loss

    def _G_hingeLoss(self, fake):
        return - torch.mean(fake)

    def _discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def _maximum_classifier_discrepancy(self, images, labels):
        _, h = self.basemodel(images)
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

    def _netG(self, domain_label, class_label, concat1, concat2, concat3, concat4, concat5):
        concat1 = concat1.to(self.gpu_map['netG_2'])
        concat2 = concat2.to(self.gpu_map['netG_2'])
        concat3 = concat3.to(self.gpu_map['netG'])
        concat4 = concat4.to(self.gpu_map['netG'])
        concat5 = concat5.to(self.gpu_map['netG'])
        domain_label = domain_label.to(self.gpu_map['netG'])
        class_label = class_label.to(self.gpu_map['netG'])
        return self.netG(domain_label, class_label, concat1, concat2, concat3, concat4, concat5)

    def _zero_grad(self):
        self.optDFeat.zero_grad()
        self.optDImg.zero_grad()
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
        self.domain_label = self.domain_label.to(self.gpu_map['netG'])

        # -----------------------------
        # 2. Feedforward Basemodel and netG
        # -----------------------------

        """ Classification and Adversarial Loss (Basemodel) """
        pix_feature, adv_feature = self.basemodel(images)

        """ Idt, Fake, Cycle, DCls, Semantic Loss (Generator) """
        label_onehot = torch.cat([
            torch.eye(self.num_classes+1)[labels],
            torch.eye(self.num_classes+1)[-1].unsqueeze(0).repeat(self.batch_size, 1)], dim=0).to(self.gpu_map['netG'])

        self.domain_label = self.domain_label.to(self.gpu_map['netG'])
        rand_idx = torch.randperm(self.domain_label.size(0))
        self.shuffled_domain_label = self.domain_label[rand_idx]

        label_onehot = label_onehot.to(self.gpu_map['netG'])
        pix_feature = pix_feature.to(self.gpu_map['netG'])
        trsfakeImg = self.netG(self.shuffled_domain_label, label_onehot, pix_feature)
        idtfakeImg = self.netG(self.domain_label, label_onehot, pix_feature)

        # -----------------------------
        # 3. Train Discriminators
        # -----------------------------
        for param in self.netDImg.parameters():
            param.requires_grad = True
        for param in self.netDFeat.parameters():
            param.requires_grad = True

        # Train with original domain labels
        DFeatlogit = self.netDFeat(adv_feature.detach().to(self.gpu_map['netDFeat']))
        Dloss_AdvFeat = self.FeatAdv_loss(DFeatlogit,
                                          self._real_domain_label(DFeatlogit, 'Feat'))
        Dloss_AdvFeat.backward()
        self.log_loss['Dloss_AdvFeat'] = Dloss_AdvFeat.item()
        self.optDFeat.step()

        fake_logit, _, _  = self.netDImg(trsfakeImg.detach().to(self.gpu_map['netDImg']))
        self.Dloss_fake = torch.mean(fake_logit)
        real_logit, dcls_logit, aux_logit = self.netDImg(images.to(self.gpu_map['netDImg']))
        self.Dloss_real = - torch.mean(real_logit)
        self.Dloss_dcls = nn.CrossEntropyLoss()(dcls_logit, self.domain_label.to(self.gpu_map['netDImg']))
        self.Dloss_auxsem = self._aux_semantic_loss(aux_logit.to(self.gpu0), labels)

        self.Dloss_gp = self._gradient_penalty(real=images.to(self.gpu_map['netDImg']),
                                               fake=trsfakeImg.detach().to(self.gpu_map['netDImg']),
                                               ld=self.config['train']['GAN']['GP'])
        self.Dloss_fakereal = self.Dloss_fake + self.Dloss_real + self.Dloss_gp
        # At this moment we don't care about semantic loss of target data

        self._backprop_weighted_losses(self.loss_lambda['netD'],
                                       retain_graph=True)
        self.optDImg.step()
        # ----------------------------
        # 4. Train Basemodel
        # ----------------------------

        for param in self.netDImg.parameters():
            param.requires_grad = False
        for param in self.netDFeat.parameters():
            param.requires_grad = False


        # ----------------------------
        # Maximum Classifier Discrepancy
        # ----------------------------
        loss_s, _ = self._maximum_classifier_discrepancy(images, labels)
        loss_s.backward()
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

        # -------------------------

        _, adv_feature = self.basemodel(images)
        DFeatlogit = self.netDFeat(adv_feature.to(self.gpu_map['netDFeat']))
        bloss_AdvFeat = self.FeatAdv_loss(DFeatlogit,
                                          self._fake_domain_label(DFeatlogit, 'Feat'))
        bloss_AdvFeat.backward()
        self.optBase.step()

        # ----------------------------
        # 5. Train netG
        # ----------------------------
        if (i_iter+1) % self.config['train']['GAN']['n_critic'] == 0:
            # 5-1. G
            self.optG.zero_grad()
            self.optBase.zero_grad()

            pix_feature, _ = self.basemodel(images)
            trsfakeImg = self.netG(self.shuffled_domain_label, label_onehot, pix_feature)
            pix_feature_, _ = self.basemodel(trsfakeImg.contiguous())
            cycfakeImg = self.netG(self.domain_label, label_onehot, pix_feature_)
            fake_logit, dcls_logit, aux_logit = self.netDImg(trsfakeImg.to(self.gpu_map['netDImg']))

            self.Gloss_fake = - torch.mean(fake_logit)
            self.Gloss_cyc = torch.mean(torch.abs(images - cycfakeImg.to(self.gpu0)))
            self.Gloss_auxsem = self._aux_semantic_loss(aux_logit.to(self.gpu0), labels)
            self.Gloss_dcls = nn.CrossEntropyLoss()(dcls_logit,
                                                    self.shuffled_domain_label.to(self.gpu_map['netDImg']))

            self._backprop_weighted_losses(self.loss_lambda['netG'])
            self.optBase.step()
            self.optG.step()

            """
            # 5-2. Domain Adversarial loss for Base only
            self.optBase.zero_grad()

            AdvDcls_fake_logit, _ = self.netDImg(trsfakeImg.to(self.gpu_map['netDImg']),
                                                 self.domain_label.to(self.gpu_map['netDImg']),
                                                 adv_training=True)
            self.bloss_fake = self._G_hingeLoss(AdvDcls_fake_logit)
            self.bloss_auxsem = self._aux_semantic_loss(aux_logit.to(self.gpu0), labels)
            self._backprop_weighted_losses(self.loss_lambda['base_only'],
                                           retain_graph=False)
            self.optBase.step()
            """
        # -----------------------------------------------
        # -----------------------------------------------

        if (i_iter+1) % self.log_step == 0:
            et = time.time() - self.start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]\n".format(et, i_iter+1, self.early_stop_step)
            for tag, value in self.log_loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            _, h = self.basemodel(images)
            pred = self.C1(h)
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
                self.basemodel.eval()
                self.netG.eval()

                image_fixed = images[[i*self.batch_size for i in range(self.num_domain)]]
                label_fixed = labels[[i*self.batch_size for i in range(self.num_domain-1)]]
                label_fixed_onehot = torch.cat([
                    torch.eye(self.num_classes+1)[label_fixed],
                    torch.eye(self.num_classes+1)[-1].unsqueeze(0)])

                image_fake_list = [image_fixed]
                domain_fixed = self._fixed_test_domain_label(self.num_domain)

                for d in domain_fixed:
                    pix_feature, adv_feature = self.basemodel(image_fixed.to(self.gpu0))
                    image_fake = self.netG(d.to(self.gpu_map['netG']),
                                           label_fixed_onehot.to(self.gpu_map['netG']),
                                           pix_feature.to(self.gpu_map['netG']))
                    image_fake_list.append(image_fake)

                image_concat = torch.cat(image_fake_list, dim=3)
                sample_path = os.path.join(self.log_dir, '{}-FixTrsimages.jpg'.format(i_iter+1))
                save_image(self._denorm(image_concat.data.cpu()), sample_path, nrow=self.num_domain, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))
                self.basemodel.train()
                self.netG.train()

        if (i_iter+1) % self.save_step == 0:
            print('taking snapshot ...')
            torch.save({
                'basemodel': self.basemodel.state_dict(),
                'netG': self.netG.state_dict(),
                'netDImg': self.netDImg.state_dict(),
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

                _, h = self.basemodel(target_images)
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
        self.logger.scalar_summary('metrics/val_acc', acc3, i_iter+1)
        print(info_str)

        with open(os.path.join(self.log_dir, 'val_result.txt'), 'a') as f:
            f.write(info_str+'\n')
            f.close()

    def _tsne(self, i_iter):
        # Plot t-SNE of hidden feature
        source_images1, source_labels1 = next(self.loader_iter)
        target_images1, target_labels1 = next(self.target_iter)
        tsne_images = torch.cat([source_images1[:self.batch_size*(self.num_domain-1)],
                                 target_images1], dim=0).to(torch.float)
        tsne_labels = torch.cat([source_labels1[:self.batch_size*(self.num_domain-1)],
                                 target_labels1], dim=0)
        tsne_domain = self.domain_label

        sample_path = os.path.join(self.log_dir, '{}-tSNE.jpg'.format(i_iter+1))
        _, h = self.basemodel.cpu()(tsne_images.cpu())
        tsne = self.tsne.fit_transform(h.data.cpu().numpy())
        plot_embedding(tsne, tsne_labels.data.cpu().numpy(),
                       tsne_domain.data.cpu().numpy(), sample_path)
        print('{} iter; saved t-SNE'.format(i_iter))

