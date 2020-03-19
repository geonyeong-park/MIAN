import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_res import ResidualBlock, UpsamplingBlock, ConditionalBatchOrInstanceNorm2d


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_domain, norm, gpu, upsample=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.gpu = gpu
        self.num_domain = num_domain
        self.upsample = upsample

        self.conv = nn.Conv2d(in_channels+num_domain, middle_channels, kernel_size=3, stride=1, padding=1, bias=False).to(gpu) #Concat domain code everytime
        self.relu = nn.ReLU(inplace=True).to(gpu)
        self.block = ResidualBlock(middle_channels, middle_channels, num_domain, norm).to(gpu)

        if upsample:
            self.upsample = UpsamplingBlock(middle_channels, out_channels, num_domain, norm).to(gpu)
        else:
            assert middle_channels == out_channels

    def _tile_domain_code(self, feature, domain):
        w, h = feature.size()[-2], feature.size()[-1]
        domain_code = torch.eye(self.num_domain)[domain.long()].view(-1, self.num_domain, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, domain_code], dim=1).to(self.gpu)

    def forward(self, x, c):
        h = self._tile_domain_code(x, c)
        h = self.conv(h)
        h = self.relu(h)
        h = self.block(h, c)
        if self.upsample:
            return self.upsample(h, c)
        else:
            return h


class GeneratorVGG(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, norm='CondIN', gpu=None, gpu2=None):
        super(GeneratorVGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.num_domain = num_domain
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, num_domain, norm, gpu)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, num_domain, norm, gpu, upsample=False)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 4, num_filters * 4, num_domain, norm, gpu)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu2)
        self.dec1 = nn.Conv2d(64 + num_filters * 1, num_filters, kernel_size=3,padding=1).to(gpu2)
        self.relu = nn.ReLU(inplace=True).to(gpu2)
        self.last_conv = nn.Conv2d(num_filters, 3, kernel_size=1).to(gpu2)

    def forward(self, conv1, conv2, conv3, conv4, conv5, c):
        """c is not one-hot vector"""

        # conv5(Deepest hidden feature in VGG) is not pre-pooled. should be pooled at first
        center = self.center(self.pool(conv5), c)
        dec5 = self.dec5(torch.cat([center, conv5], 1), c)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1), c)
        dec4 = dec4.to(self.gpu2)
        c = c.to(self.gpu2)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1), c)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1), c)
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.last_conv(self.relu(dec1))

