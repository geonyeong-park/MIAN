import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import DecoderBlock, ResidualBlock

class GeneratorVGG(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, num_classes=32, norm='CondIN', gpu=None, gpu2=None):
        super(GeneratorVGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2

        self.center = DecoderBlock(512 + num_classes, num_filters * 8 * 2, num_filters * 8, num_domain, norm, gpu)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, num_domain, norm, gpu)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 4, num_filters * 4, num_domain, norm, gpu)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu2)
        self.dec1 = nn.Conv2d(64 + num_filters * 1, num_filters, kernel_size=3,padding=1).to(gpu2)
        self.relu = nn.ReLU(inplace=True).to(gpu2)
        self.last_conv = nn.Conv2d(num_filters, 3, kernel_size=1).to(gpu2)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def forward(self, conv1, conv2, conv3, conv4, conv5, c, l):
        """c is not one-hot vector"""

        # conv5(Deepest hidden feature in VGG) is not pre-pooled. should be pooled at first
        conv5_p = self.pool(conv5)
        conv5_p_l = self._tile_label_code(conv5_p, l)

        center = self.center(conv5_p_l, c)
        dec5 = self.dec5(torch.cat([center, conv5], 1), c)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1), c)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1), c)

        dec3 = dec3.to(self.gpu2)
        c = c.to(self.gpu2)

        dec2 = self.dec2(torch.cat([dec3, conv2], 1), c)
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.last_conv(self.relu(dec1))


"""
class GeneratorVGG(nn.Module):
    def __init__(self, num_filters=64, num_domain=3, norm='CondIN', gpu=None, gpu2=None):
        super(GeneratorVGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.num_domain = num_domain
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2

        self.center = DecoderBlock(512, 0, num_filters * 8, num_filters * 8, num_domain, norm, gpu, skip_connection=False)
        self.dec5 = DecoderBlock(num_filters * 8, 512, num_filters * 8, num_filters * 8, num_domain, norm, gpu, upsample=False)
        self.dec4 = DecoderBlock(num_filters * 8, 512, num_filters * 4, num_filters * 4, num_domain, norm, gpu)
        self.dec3 = DecoderBlock(num_filters * 4, 256, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
        self.dec2 = DecoderBlock(num_filters * 2, 128, num_filters * 1, num_filters * 1, num_domain, norm, gpu)
        self.skip_block = ResidualBlock(64, 64, num_domain, norm).to(gpu2)
        self.dec1 = nn.Conv2d(64 + num_filters * 1, num_filters, kernel_size=3, padding=1).to(gpu2)
        self.relu = nn.ReLU(inplace=True).to(gpu2)
        self.last_conv = nn.Conv2d(num_filters, 3, kernel_size=1).to(gpu2)

    def forward(self, conv1, conv2, conv3, conv4, conv5, c):
        # conv5(Deepest hidden feature in VGG) is not pre-pooled. should be pooled at first
        center = self.center(self.pool(conv5), c, prev_feat=None)
        dec5 = self.dec5(center, c, conv5)
        dec4 = self.dec4(dec5, c, conv4)
        dec3 = self.dec3(dec4, c, conv3)
        dec2 = self.dec2(dec3, c, conv2)

        dec2 = dec2.to(self.gpu2)
        c = c.to(self.gpu2)

        dec1 = self.dec1(torch.cat([dec2, self.skip_block(conv1, c)], 1))

        return self.last_conv(self.relu(dec1))
"""
