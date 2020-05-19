import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
from model.generator_block import ResidualBlock


class FeatDiscriminator(nn.Module):
    def __init__(self, channel=4096, num_domain=3):
        super(FeatDiscriminator, self).__init__()

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
