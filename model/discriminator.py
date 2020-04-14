import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_res import ResidualBlock


class IMGDiscriminator(nn.Module):
    def __init__(self, image_size=512, conv_dim=128, channel=3, repeat_num=5, num_domain=3, num_classes=19):
        super(IMGDiscriminator, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        assert channel == 3
        curr_dim = channel
        next_dim = conv_dim

        downsample = []
        for i in range(repeat_num):
            downsample.append(spectral_norm(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1)))
            downsample.append(nn.LeakyReLU(0.01))
            buffer_dim = next_dim
            curr_dim = next_dim
            next_dim = next_dim * 2 if not next_dim > 2000 else curr_dim

        downsample.append(spectral_norm(nn.Conv2d(curr_dim, 1024, kernel_size=3, stride=2, padding=0)))
        downsample.append(nn.LeakyReLU(0.01))
        self.downsample = nn.Sequential(*downsample)
        self.dropout = nn.Dropout(0.5)

        self.conv_real_fake = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_domain_cls = nn.Conv2d(1024, num_domain, kernel_size=3, padding=0, bias=False)

        aux_clf = []
        for i in range(num_domain-1):
            aux_clf.append(nn.Sequential(*[
                ResidualBlock(1024, 1024, num_domain=num_domain, norm='SN'),
                ResidualBlock(1024, 1024, num_domain=num_domain, norm='SN'),
                ResidualBlock(1024, 1024, num_domain=num_domain, norm='SN'),
                nn.Conv2d(1024, num_classes, kernel_size=3, padding=0, bias=False)]))
        self.aux_clf = nn.ModuleList(aux_clf)

    def forward(self, x):
        h = self.downsample(x)
        # h = self.dropout(h)

        out_src = self.conv_real_fake(h)
        out_domain = self.conv_domain_cls(h)
        out_aux = torch.cat([clf(h).unsqueeze_(0) for clf in self.aux_clf], dim=0)

        return out_src, out_domain.view(out_domain.size(0), out_domain.size(1)), out_aux.view(out_aux.size(0), out_aux.size(1), out_aux.size(2))


class ResDiscriminator(nn.Module):
    def __init__(self, conv_dim=1024, channel=2048, repeat_num=3, num_domain=3):
        super(ResDiscriminator, self).__init__()

        downsample = []
        next_dim = conv_dim
        curr_dim = channel

        for i in range(repeat_num):
            downsample.append(nn.Conv2d(curr_dim, next_dim, kernel_size=3, stride=2, padding=0))
            downsample.append(nn.LeakyReLU(0.01))
            curr_dim = next_dim
            next_dim = next_dim // 2 if not next_dim < 60 else next_dim

        self.dropout = nn.Dropout(0.5)
        self.downsample = nn.Sequential(*downsample)
        self.conv_domain_cls_patch = nn.Linear(curr_dim, num_domain)

    def forward(self, x):
        h = self.downsample(x)
        h = h.view(h.size()[0], h.size()[1])
        #h = self.dropout(h)
        out_src = self.conv_domain_cls_patch(h)
        return out_src


class VGGDiscriminator(nn.Module):
    def __init__(self, channel=4096, num_domain=3):
        super(VGGDiscriminator, self).__init__()

        self.conv_domain_cls_patch = nn.Sequential(*[
            nn.Linear(channel, channel//4),
            nn.ReLU(inplace=True),
            nn.Linear(channel//4, num_domain)])

    def forward(self, x):
        out_src = self.conv_domain_cls_patch(x)
        return out_src

class AlexDiscriminator(nn.Module):
    def __init__(self, channel=4096, num_domain=3):
        super(AlexDiscriminator, self).__init__()

        self.conv_domain_cls_patch = nn.Sequential(*[
            nn.Linear(channel, channel//4),
            nn.ReLU(inplace=True),
            nn.Linear(channel//4, num_domain)])

    def forward(self, x):
        out_src = self.conv_domain_cls_patch(x)
        return out_src
