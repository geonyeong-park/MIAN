import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_domain, norm, gpu, upsample=True):
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
            self.upsample = UpsamplingBlock(middle_channels, out_channels, num_domain, norm).to(gpu)
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
                self.a1
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
    def __init__(self, dim_in, dim_out, num_domain, norm):
        super(UpsamplingBlock, self).__init__()
        assert norm == 'CondBN' or norm == 'CondIN'
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn = ConditionalBatchOrInstanceNorm2d(dim_out, num_domain, norm)
        self.a = nn.ReLU(inplace=True)

    def forward(self, input, cls):
        x = self.conv(input)
        x = self.cbn(x, cls)
        x = self.a(x)
        return x

class GeneratorRes(nn.Module):
    """Generator network."""
    def __init__(self, num_filters=64, num_domain=3, repeat_num=5, norm='CondIN', gpu=None, gpu2=None, num_classes=32):
        super(GeneratorRes, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.num_domain = num_domain
        self.num_classes = num_classes
        self.norm = norm
        self.gpu = gpu
        self.gpu2 = gpu2
        self.repeat_num = repeat_num

        self.encode = nn.Conv2d(2048+num_classes+num_domain, num_filters, kernel_size=3, stride=1, padding=1, bias=False).to(gpu)

        for i in range(repeat_num):
            setattr(self, 'RB{}'.format(i), ResidualBlock(num_filters, num_filters, num_domain, norm).to(gpu))

        filter = num_filters
        for i in range(5-1):
            #setattr(self, 'US{}'.format(i), DecoderBlock(filter, filter//2, filter//2, num_domain, norm, gpu))
            setattr(self, 'US{}'.format(i), UpsamplingBlock(filter, filter//2, num_domain, norm).to(gpu))
            filter = filter // 2

        self.last_conv = UpsamplingBlock(filter, 3, num_domain, norm).to(gpu2)

    def _tile_label_code(self, feature, label):
        w, h = feature.size()[-2], feature.size()[-1]
        label_code = label.view(-1, self.num_classes, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, label_code], dim=1).to(self.gpu)

    def _tile_domain_code(self, feature, domain):
        w, h = feature.size()[-2], feature.size()[-1]
        domain_code = torch.eye(self.num_domain)[domain.long()].view(-1, self.num_domain, 1, 1).repeat(1, 1, w, h).to(self.gpu)
        return torch.cat([feature, domain_code], dim=1).to(self.gpu)

    def forward(self, x, c, l):
        """c is not one-hot vector, l is one-hot vector for labels"""

        xlabel = self._tile_label_code(x, l)
        xlabel_domain = self._tile_domain_code(xlabel, c)
        h = self.encode(xlabel_domain)

        for i in range(self.repeat_num):
            h = getattr(self, 'RB{}'.format(i))(h, c)

        for i in range(5-1):
            h = getattr(self, 'US{}'.format(i))(h, c)

        h = h.to(self.gpu2)
        c = c.to(self.gpu2)

        return self.last_conv(h, c)

