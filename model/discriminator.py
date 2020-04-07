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
            curr_dim = next_dim
            next_dim = next_dim * 2 if not next_dim > 2000 else curr_dim

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.downsample = nn.Sequential(*downsample)
        self.dropout = nn.Dropout(0.5)

        self.conv_real_fake = nn.Conv2d(next_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_domain_cls = nn.Conv2d(next_dim, num_domain, kernel_size=kernel_size, bias=False)

        compress = []
        for i in range(kernel_size // 2):
            compress.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1)))
            compress.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim // 2
        self.compress = nn.Sequential(*compress)

        aux_clf = []
        for i in range(num_domain-1):
            aux_clf.append(nn.Sequential(*[nn.Linear(curr_dim, num_classes)]))

        self.aux_clf = nn.ModuleList(aux_clf)

    def forward(self, x):
        h = self.downsample(x)
        h = self.dropout(h)

        out_src = self.conv_real_fake(h)
        out_domain = self.conv_domain_cls(h)
        out_domain = torch.mean(out_domain.view(out_domain.size(0), out_domain.size(1), -1), dim=2)

        h = self.compress(h)
        h = h.view(h.size(0), h.size(1))

        out_aux = torch.cat([clf(h).unsqueeze_(0) for clf in self.aux_clf], dim=0)

        return out_src, out_domain.view(out_domain.size(0), out_domain.size(1)), out_aux


class SEMDiscriminator(nn.Module):
    def __init__(self, conv_dim=32, channel=3, repeat_num=7, num_domain=3, feat=False):
        super(SEMDiscriminator, self).__init__()

        downsample = []
        next_dim = conv_dim
        curr_dim = channel

        for i in range(repeat_num):
            downsample.append(nn.Linear(curr_dim, next_dim))
            downsample.append(nn.LeakyReLU(0.01))
            curr_dim = next_dim
            if not feat:
                next_dim = next_dim * 2 if not next_dim > 2000 else next_dim
            else:
                next_dim = next_dim // 2 if not next_dim < 60 else next_dim

        self.dropout = nn.Dropout(0.5)
        self.downsample = nn.Sequential(*downsample)
        self.conv_domain_cls_patch = nn.Linear(curr_dim, num_domain)

    def forward(self, x):
        if len(x.size()) == 4:
            x = torch.mean(x, dim=(2,3))
        h = self.downsample(x)
        #h = self.dropout(h)
        out_src = self.conv_domain_cls_patch(h)
        return out_src

