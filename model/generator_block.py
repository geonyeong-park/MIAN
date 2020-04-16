import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_domain, norm, gpu, upsample=True,
                 up_kernel=4, up_stride=2, up_pad=1):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.gpu = gpu
        self.num_domain = num_domain
        self.upsample = upsample

        self.conv = nn.Conv2d(in_channels+num_domain, middle_channels, kernel_size=3, stride=1, padding=1, bias=False).to(gpu) #Concat domain code everytime
        self.cn = ConditionalBatchOrInstanceNorm2d(middle_channels, num_domain, norm).to(gpu)
        self.relu = nn.ReLU(inplace=True).to(gpu)
        self.block = ResidualBlock(middle_channels, middle_channels, num_domain, norm).to(gpu)

        if upsample:
            self.upsample = UpsamplingBlock(middle_channels, out_channels, num_domain, norm,
                                            up_kernel, up_stride, up_pad).to(gpu)
        else:
            assert middle_channels == out_channels

    def _tile_domain_code(self, feature, domain):
        w, h = feature.size()[-2], feature.size()[-1]
        domain_code = torch.eye(self.num_domain)[domain.long()].view(-1, self.num_domain, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, domain_code], dim=1).to(self.gpu)

    def forward(self, x, c):
        h = self._tile_domain_code(x, c)
        h = self.conv(h)
        h = self.cn(h, c)
        h = self.relu(h)
        h = self.block(h, c)
        if self.upsample:
            img = self.upsample(h, c)
            return img
        else:
            return h


class ConditionalBatchOrInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_domain, norm='CondIN'):
        super().__init__()
        self.num_features = num_features
        self.norm = norm
        assert norm == 'CondBN' or norm == 'CondIN' or norm == 'BN' or norm == 'IN'

        if norm == 'CondBN' or norm == 'CondIN':
            self.bn = nn.BatchNorm2d(num_features, affine=False) if norm == 'CondBN' else nn.InstanceNorm2d(num_features, affine=False)
            self.embed = nn.Embedding(num_domain, num_features * 2)
            self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=True) if norm == 'BN' else nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x, y):
        out = self.bn(x)
        if self.norm == 'BN' or self.norm == 'IN':
            return out
        else:
            gamma, beta = self.embed(y.long()).chunk(2, 1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
            return out

class ResidualBlock(nn.Module):
    """Residual Block with conditional batch normalization."""
    def __init__(self, dim_in, dim_out, num_domain=3, norm='CondBN', kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.norm = norm

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.a1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        if norm == 'CondBN' or norm == 'CondIN' or norm == 'BN' or norm == 'IN':
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
        if self.norm == 'CondBN' or self.norm == 'CondIN' or self.norm == 'BN' or self.norm == 'IN':
            assert cls is not None
            x = self.conv1(input)
            x = self.a1(self.norm1(x, cls))
            x = self.norm2(self.conv2(x), cls)
        elif self.norm == 'SN':
            x = self.block(input)
        return input + x


class UpsamplingBlock(nn.Module):
    """Residual Block with conditional batch normalization."""
    def __init__(self, dim_in, dim_out, num_domain, norm, kernel=4, stride=2, pad=1):
        super(UpsamplingBlock, self).__init__()
        assert norm == 'CondBN' or norm == 'CondIN' or norm == 'BN' or norm == 'IN'
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.cbn = ConditionalBatchOrInstanceNorm2d(dim_out, num_domain, norm)
        self.a = nn.ReLU(inplace=True)

    def forward(self, input, cls):
        x = self.conv(input)
        x = self.cbn(x, cls)
        x = self.a(x)
        return x



"""

class DecoderBlock(nn.Module):
    def __init__(self, feat_channels, prev_channels, middle_channels, out_channels, num_domain, norm, gpu, skip_connection=True, upsample=True):
        super(DecoderBlock, self).__init__()
        self.gpu = gpu
        self.num_domain = num_domain
        self.upsample = upsample
        self.skip_connection = skip_connection

        if skip_connection:
            assert prev_channels != 0
            self.block = ResidualBlock(prev_channels, prev_channels, num_domain, norm).to(gpu)

        self.conv = nn.Conv2d(feat_channels+prev_channels, middle_channels, kernel_size=3, stride=1, padding=1, bias=False).to(gpu) #Concat domain code everytime
        self.cbn = ConditionalBatchOrInstanceNorm2d(middle_channels, num_domain, norm).to(gpu)
        self.relu = nn.ReLU(inplace=True).to(gpu)

        if upsample:
            self.upsample = UpsamplingBlock(middle_channels, out_channels, num_domain, norm).to(gpu)
        else:
            assert middle_channels == out_channels

    def forward(self, feature, c, prev_feat=None):
        if prev_feat is not None:
            assert self.skip_connection == True
            transformed_prev_skip = self.block(prev_feat, c)
            feature = torch.cat([feature, transformed_prev_skip], 1)

        h = self.conv(feature)
        h = self.cbn(h, c)
        h = self.relu(h)
        if self.upsample:
            return self.upsample(h, c)
        else:
            return h
"""



