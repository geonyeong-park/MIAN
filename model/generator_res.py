import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock, UpsamplingBlock


class GeneratorRes(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, repeat_num=5, norm='CondIN', gpu=None, gpu2=None, num_classes=32):
        super(GeneratorRes, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2
        self.repeat_num = repeat_num
        self.encode = nn.Conv2d(2048+num_classes+num_domain, num_filters, kernel_size=3, stride=1, padding=1, bias=False).to(gpu)

        for i in range(repeat_num):
            setattr(self, 'RB{}'.format(i), ResidualBlock(num_filters, num_filters, num_domain, norm).to(gpu))

        filter = num_filters
        for i in range(5-1):
            setattr(self, 'US{}'.format(i), UpsamplingBlock(filter, filter//2, num_domain, norm).to(gpu))
            filter = filter // 2

        self.last_conv = UpsamplingBlock(filter, 3, num_domain, norm).to(gpu2)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def _tile_domain_code(self, feature, domain):
        w, h = feature.size()[-2], feature.size()[-1]
        domain_code = torch.eye(self.num_domain)[domain.long()].view(-1, self.num_domain, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, domain_code], dim=1).to(self.gpu)

    def forward(self, x, c, l):
        """c is not one-hot vector, l is one-hot vector for labels"""

        xlabel = self._tile_label_code(x, l)
        xlabel_domain = self._tile_domain_code(xlabel, c)
        h = self.encode(xlabel_domain)

        for i in range(self.repeat_num):
            h = getattr(self, 'RB{}'.format(i))(h, c)

        for i in range(5-1):
            h = getattr(self, 'US{}'.format(i))(h, c)

        h = h.to(self.gpu2)
        c = c.to(self.gpu2)

        return self.last_conv(h, c)

