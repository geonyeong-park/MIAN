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

        self.pool = nn.MaxPool2d(2,2)
        self.enc = nn.Sequential(*[
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            ])

        self.compress1 = nn.Sequential(*[
            nn.Linear(8192, 3072),
            nn.BatchNorm1d(3072, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()])
        self.compress2 = nn.Sequential(*[
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048, affine=True),
            nn.ReLU(inplace=True)])

    def forward(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        h = self.compress1(h)
        adv_feat = self.compress2(h)

        return adv_feat.view(adv_feat.size(0), adv_feat.size(1)), h

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

