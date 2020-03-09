import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
import numpy as np
from model.deeplab_res import Classifier_Module


class VGGMulti(nn.Module):
    def __init__(self, num_classes=19):
        super(VGGMulti, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.vgg16(pretrained=True).features
        for i in [24, 26, 28]:
            self.encoder[i].dilation = (2,2)
            self.encoder[i].padding = (2,2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.expansion = nn.Sequential(*[
            nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        ])

        self.deeplab = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))  # conv5 is not pooled at this moment
        feature = [conv1, conv2, conv3, conv4, conv5]

        h = self.expansion(conv5)
        pred = self.deeplab(h)
        return feature, pred

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
        b.append(self.expansion)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.deeplab.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def DeeplabVGG(num_classes=21):
    model = VGGMulti(num_classes)
    return model

