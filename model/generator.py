import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock, UpsamplingBlock, DecoderBlock


class GeneratorDigits(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, num_classes=10, gpu=None, norm='CondIN', prev_feature_size=128):
        # ResNet size: 64, 256, 512, 1024, 2048
        # VGG size: 64, 128, 256, 512, 512
        super(GeneratorDigits, self).__init__()
        self.gpu = gpu
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm

        self.encode = nn.Conv2d(prev_feature_size+num_domain+num_classes+1, prev_feature_size, 1, 1)
        self.bn = nn.BatchNorm2d(prev_feature_size)
        self.relu = nn.ReLU()
        self.res1 = ResidualBlock(prev_feature_size, prev_feature_size, num_domain, norm).to(gpu)
        self.res2 = ResidualBlock(prev_feature_size, prev_feature_size, num_domain, norm).to(gpu)
        self.res3 = ResidualBlock(prev_feature_size, prev_feature_size, num_domain, norm).to(gpu)
        self.res4 = ResidualBlock(prev_feature_size, prev_feature_size, num_domain, norm).to(gpu)

        self.dec1 = UpsamplingBlock(prev_feature_size, prev_feature_size//2, num_domain, norm).to(gpu)
        self.dec2 = UpsamplingBlock(prev_feature_size//2, prev_feature_size//4, num_domain, norm).to(gpu)
        self.dec3 = UpsamplingBlock(prev_feature_size//4, 3, num_domain, norm).to(gpu)
        self.last_conv = nn.Conv2d(3, 3, 3, 1, 1).to(gpu)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes+1, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def _tile_domain_code(self, feature, domain):
        w, h = feature.size()[-2], feature.size()[-1]
        domain_code = torch.eye(self.num_domain)[domain.long()].view(-1, self.num_domain, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, domain_code], dim=1).to(self.gpu)

    def forward(self, c, l, h):
        if len(h.size()) < 4:
            h = h.view(h.size(0), h.size(1), 1, 1)

        h_l = self._tile_label_code(h, l)
        h_d_l = self._tile_domain_code(h_l, c)

        res = self.relu(self.bn(self.encode(h_d_l)))
        res = self.res1(res, c)
        res = self.res2(res, c)
        res = self.res3(res, c)
        res = self.res4(res, c)

        dec = res
        dec = self.dec1(dec, c)
        dec = self.dec2(dec, c)
        dec = self.dec3(dec, c)

        img = self.last_conv(dec)
        return img


