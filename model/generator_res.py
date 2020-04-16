import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock, UpsamplingBlock, DecoderBlock


class GeneratorRes(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, num_classes=32, norm='CondIN', gpu=None, gpu2=None,
                 partial=False, prev_feature_size=[64, 256, 512, 1024, 2048]):
        # ResNet size: 64, 256, 512, 1024, 2048
        # VGG size: 64, 128, 256, 512, 512
        super(GeneratorRes, self).__init__()
        size1, size2, size3, size4, size5 = prev_feature_size
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2
        self.partial = partial

        self.dec5 = DecoderBlock(size5 + num_classes, num_filters * 8 * 2, num_filters * 8, num_domain, norm, gpu)
        self.dec4 = DecoderBlock(size4 + num_filters * 8, num_filters * 8, num_filters * 8, num_domain, norm, gpu)

        if not partial:
            self.dec3 = DecoderBlock(size3 + num_filters * 8, num_filters * 4, num_filters * 4, num_domain, norm, gpu)
            self.dec2 = DecoderBlock(size2 + num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
            self.dec1 = DecoderBlock(size1 + num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu2)
        else:
            self.dec3 = DecoderBlock(num_filters * 8, num_filters * 4, num_filters * 4, num_domain, norm, gpu)
            self.dec2 = DecoderBlock(num_filters * 4, num_filters * 2, num_filters * 2, num_domain, norm, gpu)
            self.dec1 = DecoderBlock(num_filters * 2, num_filters * 1, num_filters * 1, num_domain, norm, gpu2)

        self.relu = nn.ReLU(inplace=True).to(gpu2)
        self.last_conv = nn.Conv2d(num_filters, 3, kernel_size=1).to(gpu2)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def forward(self, c, l, conv1=None, conv2=None, conv3=None, conv4=None, conv5=None):
        """c is not one-hot vector"""

        # conv5(Deepest hidden feature in VGG) is not pre-pooled. should be pooled at first
        conv5_l = self._tile_label_code(conv5, l)

        dec5 = self.dec5(conv5_l, c)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1), c)

        if not self.partial:
            assert conv1 is not None
            assert conv2 is not None
            assert conv3 is not None

            dec3 = self.dec3(torch.cat([dec4, conv3], 1), c)
            dec2 = self.dec2(torch.cat([dec3, conv2], 1), c)
            dec2 = dec2.to(self.gpu2)
            c = c.to(self.gpu2)
            dec1 = self.dec1(torch.cat([dec2, conv1], 1), c)
        else:
            dec3 = self.dec3(dec4, c)
            dec2 = self.dec2(dec3, c)
            dec2 = dec2.to(self.gpu2)
            c = c.to(self.gpu2)
            dec1 = self.dec1(dec2, c)

        img = self.last_conv(dec1)
        return img




