import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import DecoderBlock, ResidualBlock

class GeneratorAlex(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, num_classes=32, norm='CondIN', gpu=None, gpu2=None):
        super(GeneratorAlex, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2

        self.center = DecoderBlock(256+num_classes, num_filters*8, num_filters*4, num_domain, norm, gpu,
                                   upsample=True, up_kernel=8, up_stride=1, up_pad=0)
        self.dec5 = DecoderBlock(256+num_filters*4, num_filters*4, num_filters*4, num_domain, norm, gpu,
                                 upsample=False)
        self.dec4 = DecoderBlock(256+num_filters*4, num_filters*4, num_filters*4, num_domain, norm, gpu,
                                 upsample=False)
        self.dec3 = DecoderBlock(384+num_filters*4, num_filters*2, num_filters*2, num_domain, norm, gpu,
                                 upsample=True, up_kernel=3, up_stride=2, up_pad=0)
        self.dec2 = DecoderBlock(192+num_filters*2, num_filters*1, num_filters*1, num_domain, norm, gpu2,
                                 upsample=True, up_kernel=3, up_stride=2, up_pad=0)
        self.dec1 = DecoderBlock(64+num_filters*1, num_filters, num_filters, num_domain, norm, gpu2,
                                 upsample=True, up_kernel=5, up_stride=2, up_pad=1).to(gpu2)
        self.last_deconv = nn.ConvTranspose2d(num_filters, 3, kernel_size=4, stride=2, padding=0, bias=False).to(gpu2)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def forward(self, conv1, conv2, conv3, conv4, conv5, c, l):
        """c is not one-hot vector, l is one-hot vector for labels"""

        conv5_p = self.pool(conv5)
        conv5_p_l = self._tile_label_code(conv5_p, l)

        center = self.center(conv5_p_l, c)
        dec5 = self.dec5(torch.cat([center, conv5], 1), c)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1), c)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1), c)

        dec3 = dec3.to(self.gpu2)
        c = c.to(self.gpu2)

        dec2 = self.dec2(torch.cat([dec3, conv2], 1), c)
        dec1 = self.dec1(torch.cat([dec2, conv1], 1), c)

        return self.last_deconv(dec1)


