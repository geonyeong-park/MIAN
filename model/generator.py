import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np


class ConditionalBatchOrInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_domain, norm='CondIN'):
        super().__init__()
        self.num_features = num_features
        self.norm = norm
        assert norm == 'CondBN' or norm == 'CondIN'

        self.bn = nn.BatchNorm2d(num_features, affine=False) if norm == 'CondBN' else nn.InstanceNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_domain, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y.long()).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

class ResidualBlock(nn.Module):
    """Residual Block with conditional batch normalization."""
    def __init__(self, dim_in, dim_out, num_domain=3, norm='CondBN'):
        super(ResidualBlock, self).__init__()
        self.norm = norm

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.a1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)

        if norm == 'CondBN' or norm == 'CondIN':
            self.norm1 = ConditionalBatchOrInstanceNorm2d(dim_out, num_domain, norm)
            self.norm2 = ConditionalBatchOrInstanceNorm2d(dim_out, num_domain, norm)

        elif norm == 'SN':
            self.block = nn.Sequential(*[
                spectral_norm(self.conv1),
                self.a1,
                spectral_norm(self.conv2),
            ])

        else:
            raise ValueError

    def forward(self, input, cls=None):
        if self.norm == 'CondBN' or self.norm == 'CondIN':
            assert cls is not None
            x = self.conv1(input)
            x = self.a1(self.norm1(x, cls))
            x = self.norm2(self.conv2(x), cls)
        elif self.norm == 'SN':
            x = self.block(input)
        return input + x


class UpsamplingBlock(nn.Module):
    """Residual Block with conditional batch normalization."""
    def __init__(self, dim_in, num_domain, norm):
        super(UpsamplingBlock, self).__init__()
        assert norm == 'CondBN' or norm == 'CondIN'
        self.conv = nn.ConvTranspose2d(dim_in, dim_in // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn = ConditionalBatchOrInstanceNorm2d(dim_in // 2, num_domain, norm)
        self.a = nn.ReLU(inplace=True)

    def forward(self, input, cls):
        x = self.conv(input)
        x = self.cbn(x, cls)
        x = self.a(x)
        return x

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, in_dim=2048, conv_dim=1024, num_domain=3, repeat_num=3, norm='CondIN', gpu=None):
        super(Generator, self).__init__()

        curr_dim = conv_dim
        self.repeat_num_RB = repeat_num
        self.repeat_num_US = 5
        self.num_domain = num_domain
        self.norm = norm
        self.gpu = gpu

        # Encode layers
        self.encode_conv = nn.Conv2d(in_dim+num_domain, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.encode_conv2 = nn.Conv2d(curr_dim+num_domain, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # Bottleneck layers.
        for i in range(self.repeat_num_RB):
            setattr(self, 'RB{}'.format(i), ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, num_domain=num_domain, norm=norm))

        # Up-sampling layers.
        for i in range(self.repeat_num_US):
            setattr(self, 'US{}'.format(i), UpsamplingBlock(dim_in=curr_dim, num_domain=num_domain, norm=norm))
            curr_dim = curr_dim // 2

        self.last_conv = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        #layers.append(2*nn.Tanh())

    def _tile_domain_code(self, feature, domain):
        width = feature.size()[-1]
        domain_code = torch.eye(self.num_domain)[domain.long()].view(-1, self.num_domain, 1, 1).repeat(1, 1, width, width).to(self.gpu)
        return torch.cat([feature, domain_code], dim=1)

    def forward(self, x, c):
        """c is not one-hot vector"""
        h = self._tile_domain_code(x, c)
        h = self.encode_conv(h)
        h = self._tile_domain_code(h, c)
        h = self.encode_conv2(h)

        for i in range(self.repeat_num_RB):
            h = getattr(self, 'RB{}'.format(i))(h, c)

        for i in range(self.repeat_num_US):
            h = getattr(self, 'US{}'.format(i))(h, c)

        return self.last_conv(h)
