import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
import numpy as np


class AlexMulti(nn.Module):
    def __init__(self, num_classes=19):
        super(AlexMulti, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        encoder = torchvision.models.alexnet(pretrained=True).features
        classifier = torchvision.models.alexnet(pretrained=True).classifier
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder[0],
                                   self.relu) # size: 55

        self.conv2 = nn.Sequential(encoder[3],
                                   self.relu) # size: 27

        self.conv3 = nn.Sequential(encoder[6],
                                   self.relu) # size: 13

        self.conv4 = nn.Sequential(encoder[8],
                                   self.relu) # size: 13

        self.conv5 = nn.Sequential(encoder[10],
                                   self.relu) # size: 13

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.compress = nn.Sequential(*list(classifier)[:-1])

        n_features = classifier[6].in_features
        self.predict = torch.nn.Linear(n_features, num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        h = self.pool(conv5)
        h = self.avgpool(h)
        h = torch.flatten(h, 1) # size: 225*6*6
        h = self.compress(h)
        feature = [conv1, conv2, conv3, conv4, conv5, h]

        pred = self.predict(h)
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
        b.append(self.compress)

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
        which does the classification
        """
        b = []
        b.append(self.predict.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': lr*10}]


def DeeplabAlex(num_classes=19):
    model = AlexMulti(num_classes)
    return model


