import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock


class IMGDiscriminator(nn.Module):
    def __init__(self, image_size=512, conv_dim=128, channel=3, repeat_num=5, num_domain=3, num_classes=19):
        super(IMGDiscriminator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        self.num_domain = num_domain

        assert channel == 3
        curr_dim = channel
        next_dim = conv_dim

        downsample = [
            spectral_norm(nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
        ]
        self.downsample = nn.Sequential(*downsample)

        self.conv_real_fake = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_domain_cls = nn.Conv2d(1024, num_domain, kernel_size=2, bias=False)

        aux_clf = []
        for i in range(num_domain-1):
            aux_clf.append(nn.Sequential(*[
                ResidualBlock(1024, 1024, num_domain, 'SN'),
                ResidualBlock(1024, 1024, num_domain, 'SN'),
                ResidualBlock(1024, 1024, num_domain, 'SN'),
                nn.Conv2d(1024, num_classes, kernel_size=2, stride=1, padding=0)]))

        self.aux_clf = nn.ModuleList(aux_clf)

    def forward(self, x):
        h = self.downsample(x)
        out_src = self.conv_real_fake(h)
        out_domain = self.conv_domain_cls(h)
        out_domain = out_domain.view(out_domain.size(0), out_domain.size(1))

        out_aux = torch.cat([clf(h).unsqueeze_(0) for clf in self.aux_clf], dim=0)

        return out_src, out_domain, out_aux.view(out_aux.size(0), out_aux.size(1), out_aux.size(2))

class ResDiscriminator(nn.Module):
    def __init__(self, channel=4096, num_domain=3):
        super(ResDiscriminator, self).__init__()

        self.conv_domain_cls_patch = nn.Sequential(*[
            nn.Linear(channel, channel//2),
            nn.ReLU(inplace=True),
            nn.Linear(channel//2, channel//4),
            nn.ReLU(inplace=True),
            nn.Linear(channel//4, num_domain)
        ])

    def forward(self, x):
        out_src = self.conv_domain_cls_patch(x)
        return out_src

class VGGDiscriminator(ResDiscriminator):
    def __init__(self, channel=4096, num_domain=3):
        super(VGGDiscriminator, self).__init__(channel, num_domain)

class AlexDiscriminator(ResDiscriminator):
    def __init__(self, channel=4096, num_domain=3):
        super(AlexDiscriminator, self).__init__(channel, num_domain)
