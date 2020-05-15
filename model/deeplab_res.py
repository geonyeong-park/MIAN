import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, model_urls, load_state_dict_from_url
import numpy as np


class ResNetMulti(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMulti, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = nn.Sequential(*list(resnet.children())[:3]) # 64,112,112
        self.conv2 = nn.Sequential(*list(resnet.children())[3:5]) # 256,56,56
        self.conv3 = nn.Sequential(*list(resnet.children())[5]) # 512,28,28
        self.conv4 = nn.Sequential(*list(resnet.children())[6]) # 1024,14,14
        self.conv5 = nn.Sequential(*list(resnet.children())[7])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.Conv2d(2048, 256, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv5 = self.bottleneck(conv5)

        h = self.avgpool(conv5)
        h = torch.flatten(h, 1)

        return h, h

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.conv2)
        b.append(self.conv3)
        b.append(self.conv4)
        b.append(self.conv5)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.bottleneck)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                {'params': self.get_10x_lr_params_NOscale(), 'lr': 10*lr}]

def DeeplabRes(num_classes=21):
    model = ResNetMulti(num_classes)
    return model


