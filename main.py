import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from solver import Solver
from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator, IMGDiscriminator
from model.generator import Generator
from utils.loss import CrossEntropy2d, loss_calc, lr_poly, adjust_learning_rate
from dataset.multiloader import MultiDomainLoader

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--yaml", type=str, default='config.yaml',
                        help="yaml pathway")
    parser.add_argument("--exp_name", type=str, default='GTA2City', required=True,
                        help="")
    return parser.parse_args()


def main(config, args):
    """Create the model and start the training."""

    cudnn.enabled = True
    cudnn.benchmark = True
    gpu = args.gpu

    num_classes = config['data']['num_classes']
    input_size = config['data']['input_size']
    cropped_size = config['data']['crop_size']
    dataset = config['data']['source'] + [config['data']['target']]
    num_workers = config['data']['num_workers']
    batch_size = config['data']['batch_size']
    num_domain = len(dataset)

    D_convdim = config['model']['netD']['conv_dim']
    D_repeat_num = config['model']['netD']['repeat_num']

    G_convdim = config['model']['netG']['conv_dim']
    G_repeat_num = config['model']['netG']['repeat_num']


    base_lr = config['model']['train']['base_model']['lr']
    base_momentum = config['model']['train']['base_model']['momentum']
    D_lr = config['model']['train']['netD']['lr']
    D_momentum = config['model']['train']['netD']['momentum']
    G_lr = config['model']['train']['netG']['lr']
    G_momentum = config['model']['train']['netG']['momentum']
    weight_decay = config['model']['train']['weight_decay']


    # ------------------------
    # 1. Create Model
    # ------------------------

    basemodel = DeeplabMulti(num_classes=num_classes)
    basemodel.train()
    basemodel.cuda(args.gpu)

    netDImg = IMGDiscriminator(cropped_size, D_convdim, repeat_num=D_repeat_num, semantic_mask=False, num_domain=num_domain, num_classes=num_classes)
    netDSem = IMGDiscriminator(cropped_size, D_convdim, repeat_num=D_repeat_num, semantic_mask=True, num_domain=num_domain, num_classes=num_classes)
    netDFeat = FCDiscriminator(num_domain=num_domain)

    netDImg.train()
    netDImg.cuda(args.gpu)
    netDSem.train()
    netDSem.cuda(args.gpu)
    netDFeat.train()
    netDFeat.cuda(args.gpu)

    netG = Generator(conv_dim=G_convdim, repeat_num=G_repeat_num, num_domain=num_domain)
    netG.train()
    netG.cuda(args.gpu)

    # ------------------------
    # 2. Create DataLoader
    # ------------------------

    w, h = map(int, input_size)
    input_size = (w, h)
    loader = MultiDomainLoader(dataset, '.', input_size, crop_size=cropped_size, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, half_crop=None)

    # ------------------------
    # 3. Create Optimizer and Solver
    # ------------------------

    optBase = optim.Adam(basemodel.optim_parameters(base_lr),
                        lr=base_lr, betas=(base_momentum, 0.99), weight_decay=weight_decay)

    optDImg = optim.Adam(netDImg.parameters(),
                        lr=D_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)
    optDSem = optim.Adam(netDSem.parameters(),
                        lr=D_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)
    optDFeat = optim.Adam(netDFeat.parameters(),
                        lr=D_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)

    optG = optim.Adam(netG.parameters(),
                      lr=G_lr, betas=(G_momentum, 0.99), weight_decay=weight_decay)

    solver = Solver(basemodel, netDImg, netDSem, netDFeat, netG, loader,
                    optBase, optDImg, optDSem, optDFeat, optG, config, args)

    # ------------------------
    # 4. Train
    # ------------------------

    solver.train()


if __name__ == '__main__':
    args = get_arguments()
    config = yaml.load(open(args.yaml, 'r'))

    snapshot_dir = config['snapshot_dir']
    log_dir = config['log_dir']
    exp_name = args.exp_name
    path_list = [os.path.join(snapshot_dir, exp_name), os.path.join(log_dir, exp_name)]

    for item in path_list:
        if not os.path.exists(item):
            os.makedirs(item)

    main(config, args)
