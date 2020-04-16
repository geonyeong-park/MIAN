import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock, UpsamplingBlock, DecoderBlock


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, num_classes=32, norm='CondIN', gpu=None, gpu2=None,
                 prev_feature_size=2048):
        # ResNet size: 64, 256, 512, 1024, 2048
        # VGG size: 64, 128, 256, 512, 512
        super(Generator, self).__init__()
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2

        self.dec6 = DecoderBlock(prev_feature_size+num_classes, num_filters * 16, num_filters * 16, num_domain, norm, gpu)
        self.dec5 = DecoderBlock(num_filters * 16, num_filters * 8, num_filters * 8, num_domain, norm, gpu)
        self.dec4 = DecoderBlock(num_filters * 8, num_filters * 4, num_filters * 4, num_domain, norm, gpu)
        self.dec3 = DecoderBlock(num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
        self.dec2 = DecoderBlock(num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu)
        self.dec1 = nn.ConvTranspose2d(num_filters * 1, 3, kernel_size=4, stride=2, padding=1).to(gpu)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def forward(self, c, l, h):
        if len(h.size()) < 4:
            h = h.view(h.size(0), h.size(1), 1, 1)

        h_l = self._tile_label_code(h, l)
        dec6 = self.dec6(h_l, c)
        dec5 = self.dec5(dec6, c)
        dec4 = self.dec4(dec5, c)
        dec3 = self.dec3(dec4, c)
        dec2 = self.dec2(dec3, c)
        img = self.dec1(dec2)

        return img




