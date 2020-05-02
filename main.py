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
from model.deeplab_digit import DeepDigits
from model.deeplab_vgg import DeeplabVGG
from model.discriminator import FeatDiscriminator
from model.classifier import Predictor
from dataset.multiloader import MultiDomainLoader
from utils.weight_init import weight_init

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

    # Test arguments
    parser.add_argument("--advcoeff", type=float, default=None, required=False,
                        help="")
    parser.add_argument("--featAdv", type=str, default=None, required=False,
                        help="")
    parser.add_argument("--target", type=str, default=None, required=False,
                        help="")
    parser.add_argument("--task", type=str, default=None, required=False,
                        help="")
    parser.add_argument("--optimizer", type=str, default=None, required=False,
                        help="")
    parser.add_argument("--resume", type=str, default=None, required=False,
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
    # Setting Test arguments
    if args.advcoeff is not None:
        c = args.advcoeff
        print('advcoeff: ', c)
        config['train']['lambda']['base_model']['bloss_AdvFeat'] = c
    if args.featAdv is not None:
        p = args.featAdv
        print('featAdv: ', p)
        assert p == 'LS' or p == 'Vanila'
        config['train']['GAN']['featAdv'] = p
    if args.optimizer is not None:
        o = args.optimizer
        print('optimizer: ', o)
        assert o == 'Momentum' or o == 'Adam'
        config['train']['optimizer'] = o
    if args.target is not None:
        t = args.target
        print('target: ', t)
        config['data']['target'] = t
    if args.task is not None:
        t = args.task
        print('task: ', t)
        config['data']['task'] = t

    # -------------------------------

    cudnn.enabled = True
    cudnn.benchmark = True
    gpu = args.gpu
    gpu_map = {
        'basemodel': 'cuda:0',
        'C': 'cuda:0',
        'netDFeat': 'cuda:0',
        'all_order': gpu
    }

    task = config['data']['task']
    num_classes = config['data']['num_classes'][task]
    input_size = config['data']['input_size'][task]
    cropped_size = config['data']['crop_size'][task]
    dataset = config['data']['domain'][task]
    dataset.remove(config['data']['target'])
    dataset = dataset + [config['data']['target']]
    num_workers = config['data']['num_workers']
    batch_size = config['train']['batch_size'][task]
    num_domain = len(dataset)

    base_lr = config['train']['base_model']['lr']
    base_momentum = config['train']['base_model']['momentum']
    D_lr = config['train']['netD']['lr']
    D_momentum = config['train']['netD']['momentum']
    weight_decay = config['train']['weight_decay']

    # ------------------------
    # 1. Create Model
    # ------------------------
    if task == 'digits':
        basemodel = DeepDigits(num_classes=num_classes)
        prev_feature_size = 2048
    else:
        basemodel = DeeplabRes(num_classes=num_classes)
        prev_feature_size = 2048

    basemodel.to(gpu_map['basemodel'])

    c1 = Predictor(prev_feature_size=prev_feature_size, num_classes=num_classes).to(gpu_map['C'])
    c2 = Predictor(prev_feature_size=prev_feature_size, num_classes=num_classes).to(gpu_map['C'])

    netDFeat = FeatDiscriminator(channel=prev_feature_size, num_domain=num_domain)
    netDFeat.to(gpu_map['netDFeat'])

    c1.apply(weight_init)
    c2.apply(weight_init)
    netDFeat.apply(weight_init)

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        basemodel.load_state_dict(checkpoint['basemodel'])
        print('load {}'.format(args.resume))

    # ------------------------
    # 2. Create DataLoader
    # ------------------------
    loader = MultiDomainLoader(dataset, '.', input_size, cropped_size, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, half_crop=None,
                               task=task)
    TargetLoader = loader.TargetLoader

    # ------------------------
    # 3. Create Optimizer and Solver
    # ------------------------
    DFeat_lr = D_lr

    if config['train']['optimizer'] == 'Momentum':
        optBase = optim.SGD(basemodel.parameters(), lr=base_lr, momentum=base_momentum, weight_decay=weight_decay)
        optC1 = optim.SGD(c1.parameters(), lr=base_lr, momentum=base_momentum, weight_decay=weight_decay)
        optC2 = optim.SGD(c2.parameters(), lr=base_lr, momentum=base_momentum, weight_decay=weight_decay)
        optDFeat = optim.SGD(netDFeat.parameters(), lr=DFeat_lr, momentum=D_momentum, weight_decay=weight_decay)
    elif config['train']['optimizer'] == 'Adam':
        optBase = optim.Adam(basemodel.parameters(), lr=base_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)
        optC1 = optim.Adam(c1.parameters(), lr=base_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)
        optC2 = optim.Adam(c2.parameters(), lr=base_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)
        optDFeat = optim.Adam(netDFeat.parameters(), lr=DFeat_lr, betas=(D_momentum, 0.99), weight_decay=weight_decay)


    solver = Solver(basemodel, c1, c2, netDFeat, loader, TargetLoader,
                    base_lr, DFeat_lr, task, num_domain,
                    optBase, optC1, optC2, optDFeat, config, args, gpu_map)

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
