import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, model_urls, load_state_dict_from_url
from model.generator_block import ResidualBlock
import numpy as np


class DigitMulti(nn.Module):
    def __init__(self, num_classes):
        super(DigitMulti, self).__init__()

        self.enc = nn.Sequential(*[
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            ])

        self.res = nn.Sequential(*[
            ResidualBlock(256, 256, 5, 'BN'),
            ResidualBlock(256, 256, 5, 'BN'),
            ResidualBlock(256, 256, 5, 'BN'),
            ResidualBlock(256, 256, 5, 'BN'),
        ])

        self.compress = nn.Sequential(*[
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True)])

    def forward(self, x):
        h = self.enc(x)
        pix_feat = self.res(h)
        adv_feat = self.compress(pix_feat)

        return pix_feat, adv_feat.view(adv_feat.size(0), adv_feat.size(1))

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.enc)
        b.append(self.res)
        b.append(self.compress)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr}]

def DeepDigits(num_classes=21):
    model = DigitMulti(num_classes)
    return model

