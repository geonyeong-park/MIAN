import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock, UpsamplingBlock, DecoderBlock


class GeneratorVGG(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, num_classes=32, norm='CondIN', gpu=None, gpu2=None,
                 partial=False, prev_feature_size=[64, 256, 512, 1024, 2048]):
        # ResNet size: 64, 256, 512, 1024, 2048
        # VGG size: 64, 128, 256, 512, 512
        super(GeneratorVGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        size1, size2, size3, size4, size5 = prev_feature_size
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2

        self.center = DecoderBlock(size5 + num_classes, num_filters * 8 * 2, num_filters * 8, num_domain, norm, gpu)
        self.dec5 = DecoderBlock(size5 + num_filters * 8, num_filters * 8, num_filters * 8, num_domain, norm, gpu)
        self.dec4 = DecoderBlock(size4 + num_filters * 8, num_filters * 4, num_filters * 4, num_domain, norm, gpu)

        if not partial:
            self.dec3 = DecoderBlock(size3 + num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
            self.dec2 = DecoderBlock(size2 + num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu2)
            self.dec1 = nn.Conv2d(size1 + num_filters * 1, num_filters, kernel_size=3,padding=1).to(gpu2)
        else:
            self.dec3 = DecoderBlock(num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
            self.dec2 = DecoderBlock(num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu2)
            self.dec1 = nn.Conv2d(num_filters * 1, num_filters, kernel_size=3,padding=1).to(gpu2)

        self.relu = nn.ReLU(inplace=True).to(gpu2)
        self.last_conv = nn.Conv2d(num_filters, 3, kernel_size=1).to(gpu2)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def forward(self, c, l, conv1=None, conv2=None, conv3=None, conv4=None, conv5=None, partial=False):
        """c is not one-hot vector"""

        # conv5(Deepest hidden feature in VGG) is not pre-pooled. should be pooled at first
        conv5_p = self.pool(conv5)
        conv5_p_l = self._tile_label_code(conv5_p, l)

        center = self.center(conv5_p_l, c)
        dec5 = self.dec5(torch.cat([center, conv5], 1), c)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1), c)

        if not partial:
            assert conv4 is not None
            assert conv5 is not None

            dec3 = self.dec3(torch.cat([dec4, conv3], 1), c)
            dec3 = dec3.to(self.gpu2)
            c = c.to(self.gpu2)

            dec2 = self.dec2(torch.cat([dec3, conv2], 1), c)
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        else:
            dec3 = self.dec3(dec4)
            dec3 = dec3.to(self.gpu2, c)
            c = c.to(self.gpu2)

            dec2 = self.dec2(dec3, c)
            dec1 = self.dec1(dec2)

        return self.last_conv(self.relu(dec1))




