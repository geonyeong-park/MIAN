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

from solver import Solver
from model.deeplab_res import DeeplabRes
from model.deeplab_vgg import DeeplabVGG
from model.discriminator import FCDiscriminator, IMGDiscriminator, SEMDiscriminator
from model.generator_res import GeneratorRes
from model.generator_vgg import GeneratorVGG
from dataset.multiloader import MultiDomainLoader
import horovod.torch as hvd

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
    return parser.parse_args()


def main(config, args):
    """Create the model and start the training."""

    # -------------------------------
    # Setting Horovod
    hvd.init()
    num_processes = hvd.size()
    print('Start process {}'.format(hvd.rank()))
    torch.set_num_threads(1)

    gpu_per_process = len(args.gpu) // num_processes
    device_map = {i: args.gpu[i*gpu_per_process: (i+1)*gpu_per_process] for i in range(num_processes)}
    gpus_tobe_used = device_map[hvd.rank()]
    gpus_tobe_used = ','.join([str(gpuNum) for gpuNum in gpus_tobe_used])
    print('gpus_tobe_used: {}'.format(gpus_tobe_used))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_tobe_used)

    # -------------------------------

    cudnn.enabled = True
    cudnn.benchmark = True
    gpu = args.gpu
    gpu_map = {
        'basemodel': 'cuda:0',
        'netDImg': 'cuda:2',
        'netDSem': 'cuda:2',
        'netDFeat': 'cuda:2',
        'netG': 'cuda:1',
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
    D_convdim_sem = config['model']['netD']['conv_dim']['sem']
    D_repeat_img = config['model']['netD']['repeat_num']['img']
    D_repeat_sem = config['model']['netD']['repeat_num']['sem']

    G_convdim = config['model']['netG']['conv_dim'][base]
    G_norm = config['model']['netG']['norm']
    G_repeat_num = config['model']['netG']['repeat_num']

    # ------------------------
    # 1. Create Model
    # ------------------------

    if base == 'ResNet':
        basemodel = DeeplabRes(num_classes=num_classes)
    elif base == 'VGG':
        basemodel = DeeplabVGG(num_classes=num_classes)
    basemodel.to(gpu_map['basemodel'])

    netDImg = IMGDiscriminator(image_size=cropped_size, conv_dim=D_convdim_img, repeat_num=D_repeat_img,
                               semantic_mask=False, channel=3, num_domain=num_domain, num_classes=num_classes)
    netDSem = SEMDiscriminator(conv_dim=D_convdim_sem, repeat_num=D_repeat_sem,
                               channel=num_classes, num_domain=num_domain)
    if base == 'ResNet':
        netDFeat = FCDiscriminator(num_features=2048, ndf=1024, num_domain=num_domain)
    elif base == 'VGG':
        netDFeat = FCDiscriminator(num_features=512, ndf=512, num_domain=num_domain)

    netDImg.to(gpu_map['netDImg'])
    netDSem.to(gpu_map['netDSem'])
    netDFeat.to(gpu_map['netDFeat'])

    if base == 'ResNet':
        netG = GeneratorRes(in_dim=2048, conv_dim=G_convdim, repeat_num=G_repeat_num,
                        num_domain=num_domain, norm=G_norm, gpu=gpu_map['netG'])
    elif base == 'VGG':
        netG = GeneratorVGG(num_filters=G_convdim, num_domain=num_domain,
                            norm=G_norm, gpu=gpu_map['netG'])

    netG.to(gpu_map['netG'])

    # ------------------------
    # 2. Create DataLoader
    # ------------------------
    w,h = map(int, input_size.split(','))
    input_size = (w,h)
    loader = MultiDomainLoader(dataset, '.', input_size, crop_size=cropped_size, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, half_crop=None,
                               num_processes=num_processes, rank=hvd.rank())
    TargetLoader = loader.TargetLoader

    # ------------------------
    # 3. Create Optimizer and Solver
    # ------------------------

    optBase_batches_per_allreduce = 8
    optBase = optim.Adam(basemodel.optim_parameters(base_lr),
                        lr=base_lr*num_processes*optBase_batches_per_allreduce,
                         betas=(base_momentum, 0.99), weight_decay=weight_decay)
    optBase = hvd.DistributedOptimizer(optBase,
                                       named_parameters=basemodel.named_parameters(),
                                       backward_passes_per_step=optBase_batches_per_allreduce)

    optDImg_batches_per_allreduce = 3
    optDImg = optim.Adam(netDImg.parameters(),
                        lr=D_lr*num_processes*optDImg_batches_per_allreduce,
                         betas=(D_momentum, 0.99), weight_decay=weight_decay)
    optDImg = hvd.DistributedOptimizer(optDImg,
                                       named_parameters=netDImg.named_parameters(),
                                       backward_passes_per_step=optDImg_batches_per_allreduce)

    optDSem = optim.Adam(netDSem.parameters(),
                        lr=D_lr*num_processes, betas=(D_momentum, 0.99), weight_decay=weight_decay)
    optDSem = hvd.DistributedOptimizer(optDSem,
                                       named_parameters=netDSem.named_parameters())

    optDFeat = optim.Adam(netDFeat.parameters(),
                        lr=D_lr*num_processes, betas=(D_momentum, 0.99), weight_decay=weight_decay)
    optDFeat = hvd.DistributedOptimizer(optDFeat,
                                       named_parameters=netDFeat.named_parameters())

    optG_batches_per_allreduce = 5
    optG = optim.Adam(netG.parameters(),
                      lr=G_lr*num_processes*optG_batches_per_allreduce,
                      betas=(G_momentum, 0.99), weight_decay=weight_decay)
    optG = hvd.DistributedOptimizer(optG,
                                    named_parameters=netG.named_parameters(),
                                    backward_passes_per_step=optG_batches_per_allreduce)

    solver = Solver(base, basemodel, netDImg, netDSem, netDFeat, netG, loader, TargetLoader,
                    optBase, optDImg, optDSem, optDFeat, optG, config, args, gpu_map)

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

    main(config, args)
