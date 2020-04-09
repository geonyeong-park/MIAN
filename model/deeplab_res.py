import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, model_urls, load_state_dict_from_url
from model.parallel_resnet import ParallelResNet101, PipelineParallelResNet101
import numpy as np

affine_par = True

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.requires_grad = False
        m.eval()

class ResNetMulti(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMulti, self).__init__()

        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        #self.resnet.apply(deactivate_batchnorm)
        self.clf = nn.Linear(2048, num_classes)

    def forward(self, x):
        feature = self.resnet(x)
        pred = self.clf(torch.mean(feature, dim=(2,3)))
        return feature, pred

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet)

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
        b.append(self.clf.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 0.1*lr},
                {'params': self.get_10x_lr_params(), 'lr': lr}]


def DeeplabRes(num_classes=21):
    model = ResNetMulti(num_classes)
    return model

