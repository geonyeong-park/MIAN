import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from shutil import copyfile

from solver import Solver
from model.deeplab_res import DeeplabRes
from model.deeplab_vgg import DeeplabVGG
from model.discriminator import IMGDiscriminator, SEMDiscriminator
from model.generator_res import GeneratorRes
from model.generator_vgg import GeneratorVGG
from dataset.multiloader import MultiDomainLoader
from utils.weight_init import weight_init

vgg16_path = './vgg16-00b39a1b-updated.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpu", type=int, nargs='+', default=None, required=True,
                        help="choose gpu device.")
    parser.add_argument("--yaml", type=str, default='config.yaml',
                        help="yaml pathway")
    parser.add_argument("--exp_name", type=str, default='GTA2City', required=True,
                        help="")
    parser.add_argument("--exp_detail", type=str, default=None, required=False,
                        help="")
    return parser.parse_args()


def main(config, args):
    """Create the model and start the training."""

    # -------------------------------
    # Setting Horovod

    gpus_tobe_used = ','.join([str(gpuNum) for gpuNum in args.gpu])
    print('gpus_tobe_used: {}'.format(gpus_tobe_used))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_tobe_used)

    # -------------------------------

    cudnn.enabled = True
    cudnn.benchmark = True
    gpu = args.gpu
    gpu_map = {
        'basemodel': 'cuda:0',
        'netDImg': 'cuda:0',
        'netDFeat': 'cuda:0',
        'netG': 'cuda:0',
        'netG_2': 'cuda:0',
        'all_order': gpu
    }

    num_classes = config['data']['num_classes']
    input_size = config['data']['input_size']
    cropped_size = config['data']['crop_size']
    dataset = config['data']['source'] + [config['data']['target']]
    num_workers = config['data']['num_workers']
    batch_size = config['train']['batch_size']
    num_domain = len(dataset)

    base = config['train']['base']
    assert base == 'VGG' or base == 'ResNet'
    base_lr = config['train']['base_model']['lr']
    base_momentum = config['train']['base_model']['momentum']
    D_lr = config['train']['netD']['lr']
    D_momentum = config['train']['netD']['momentum']
    G_lr = config['train']['netG']['lr']
    G_momentum = config['train']['netG']['momentum']
    weight_decay = config['train']['weight_decay']


    D_convdim_img = config['model']['netD']['conv_dim']['img']
    D_repeat_img = config['model']['netD']['repeat_num']['img']

    G_convdim = config['model']['netG']['conv_dim'][base]
    G_norm = config['model']['netG']['norm']
    G_repeat_num = config['model']['netG']['repeat_num']

    # ------------------------
    # 1. Create Model
    # ------------------------

    basemodel = DeeplabRes(num_classes=num_classes)
    basemodel.to(gpu_map['basemodel'])

    netDImg = IMGDiscriminator(image_size=cropped_size, conv_dim=D_convdim_img, repeat_num=D_repeat_img,
                               channel=3, num_domain=num_domain, num_classes=num_classes)
    netDFeat = SEMDiscriminator(conv_dim=512, repeat_num=2,
                                channel=2048, num_domain=num_domain, feat=True)

    netDImg.apply(weight_init)
    netDFeat.apply(weight_init)
    netDImg.to(gpu_map['netDImg'])
    netDFeat.to(gpu_map['netDFeat'])

    netG = GeneratorRes(num_filters=G_convdim, num_domain=num_domain, repeat_num=G_repeat_num,
                        norm=G_norm, gpu=gpu_map['netG'], gpu2=gpu_map['netG_2'], num_classes=num_classes+1)
    netG.apply(weight_init)

    # ------------------------
    # 2. Create DataLoader
    # ------------------------
    loader = MultiDomainLoader(dataset, '.', input_size, cropped_size, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, half_crop=None,
                               task='office')
    TargetLoader = loader.TargetLoader

    # ------------------------
    # 3. Create Optimizer and Solver
    # ------------------------

    optBase = optim.Adam(basemodel.optim_parameters(base_lr),
                         betas=(base_momentum, 0.99), weight_decay=weight_decay)

    DImg_lr = D_lr

    optDImg = optim.Adam(netDImg.parameters(),
                        lr=DImg_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)

    DFeat_lr = D_lr
    optDFeat = optim.Adam(netDFeat.parameters(),
                        lr=DFeat_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)

    optG = optim.Adam(netG.parameters(),
                      lr=G_lr, betas=(G_momentum, 0.99), weight_decay=weight_decay)

    solver = Solver(base, basemodel, netDImg, netDFeat, netG, loader, TargetLoader,
                    base_lr, DImg_lr, DFeat_lr, G_lr,
                    optBase, optDImg, optDFeat, optG, config, args, gpu_map)

    # ------------------------
    # 4. Train
    # ------------------------

    solver.train()


if __name__ == '__main__':
    args = get_arguments()
    config = yaml.load(open(args.yaml, 'r'))

    snapshot_dir = config['exp_setting']['snapshot_dir']
    log_dir = config['exp_setting']['log_dir']
    exp_name = args.exp_name
    path_list = [os.path.join(snapshot_dir, exp_name), os.path.join(log_dir, exp_name)]

    for item in path_list:
        if not os.path.exists(item):
            os.makedirs(item)

    if args.exp_detail is not None:
        print(args.exp_detail)
        with open(os.path.join(log_dir, exp_name, 'exp_detail.txt'), 'w') as f:
            f.write(args.exp_detail+'\n')
            f.close()
    copyfile(args.yaml, os.path.join(log_dir, exp_name, 'config.yaml'))

    main(config, args)
