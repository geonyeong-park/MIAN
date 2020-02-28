import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_domain):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_domain, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
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

        if norm == 'CondBN':
            self.norm1 = ConditionalBatchNorm2d(dim_out, num_domain)
            self.norm2 = ConditionalBatchNorm2d(dim_out, num_domain)
        elif norm == 'SN':
            self.norm1 = nn.utils.spectral_norm()
            self.norm2 = nn.utils.spectral_norm()
        else:
            raise ValueError

    def forward(self, input, cls):
        x = self.conv1(input)
        x = self.norm1(x, cls) if self.norm == 'CondBN' else self.norm1(x)
        x = self.a1(x)
        x = self.conv2(x)
        x = self.norm2(x, cls) if self.norm == 'CondBN' else self.norm2(x)

        return input + x

class UpsamplingBlock(nn.Module):
    """Residual Block with conditional batch normalization."""
    def __init__(self, dim_in, num_domain):
        super(UpsamplingBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(dim_in, dim_in // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn = ConditionalBatchNorm2d(dim_in // 2, num_domain)
        self.a = nn.ReLU(inplace=True)

    def forward(self, input, cls):
        x = self.conv(input)
        x = self.cbn(x, cls)
        x = self.a(x)
        return x

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_domain=3, repeat_num=5):
        super(Generator, self).__init__()

        layers = []
        curr_dim = conv_dim

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, num_domain=num_domain))

        # Up-sampling layers.
        for i in range(5):
            layers.append(UpsamplingBlock(dim_in=curr_dim, num_domain=num_domain))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        #layers.append(2*nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        """c is not one-hot vector"""
        x = (x, c)
        return self.main(x)
