import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_res import ResidualBlock
from model.deeplab_res import Classifier_Module


class IMGDiscriminator(nn.Module):
    def __init__(self, image_size=512, conv_dim=128, channel=3, repeat_num=7, num_domain=3, num_classes=19):
        super(IMGDiscriminator, self).__init__()

        init_block1 = [
            spectral_norm(nn.Conv2d(channel, conv_dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
            ResidualBlock(conv_dim, conv_dim, norm='SN'),
            ResidualBlock(conv_dim, conv_dim, norm='SN'),
        ]


        init_block2 = [
            spectral_norm(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
            ResidualBlock(conv_dim*2, conv_dim*2, norm='SN'),
            ResidualBlock(conv_dim*2, conv_dim*2, norm='SN'),
        ]

        init_block3 = [
            spectral_norm(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.01),
            ResidualBlock(conv_dim*4, conv_dim*4, norm='SN'),
            ResidualBlock(conv_dim*4, conv_dim*4, norm='SN'),
        ]

        curr_dim = conv_dim*4

        assert channel == 3
        aux_clf = []
        for i in range(num_domain-1):
            aux_clf.append(Classifier_Module(curr_dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes))
        self.aux_clf = nn.ModuleList(aux_clf)

        downsample = []
        for i in range(1, repeat_num-2):
            next_dim = curr_dim * 2 if not curr_dim > 1000 else curr_dim
            downsample.append(spectral_norm(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1)))
            downsample.append(nn.LeakyReLU(0.01))
            curr_dim = next_dim

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.init_block1 = nn.Sequential(*init_block1)
        self.init_block2 = nn.Sequential(*init_block2)
        self.init_block3 = nn.Sequential(*init_block3)
        self.downsample = nn.Sequential(*downsample)

        self.conv_real_fake = nn.Conv2d(next_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_domain_cls = nn.Conv2d(next_dim, num_domain, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.init_block1(x)
        h = self.init_block2(h)
        h = self.init_block3(h)

        out_aux = torch.cat([clf(h).unsqueeze_(0) for clf in self.aux_clf], dim=0)
        h = self.downsample(h)
        out_src = self.conv_real_fake(h)
        out_domain = self.conv_domain_cls(h)
        out_domain = torch.mean(out_domain.view(out_domain.size(0), out_domain.size(1), -1), dim=2)

        return out_src, out_domain.view(out_domain.size(0), out_domain.size(1)), out_aux


class SEMDiscriminator(nn.Module):
    def __init__(self, conv_dim=32, channel=3, repeat_num=7, num_domain=3, feat=False):
        super(SEMDiscriminator, self).__init__()

        downsample = []
        next_dim = conv_dim
        curr_dim = channel

        for i in range(repeat_num):
            downsample.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
            downsample.append(nn.LeakyReLU(0.01))
            #downsample.append(nn.Dropout(0.5))
            curr_dim = next_dim
            if not feat:
                next_dim = next_dim * 2 if not next_dim > 2000 else next_dim
            else:
                next_dim = next_dim // 2 if not next_dim < 60 else next_dim


        self.downsample = nn.Sequential(*downsample)
        self.conv_domain_cls_patch = nn.Conv2d(curr_dim, num_domain, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        h = self.downsample(x)
        out_src = self.conv_domain_cls_patch(h)
        return out_src

