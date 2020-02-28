import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from generator import ResidualBlock
from deeplab_multi import Classifier_Module

class FCDiscriminator(nn.Module):
    def __init__(self, num_features=2048, ndf=1024, num_domain=3):
        """
        employed into hidden feature discrimination
        """
        super(FCDiscriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        compress = []
        curr_dim = ndf
        compress.append(nn.Conv2d(num_features, curr_dim, kernel_size=4, stride=2, padding=1))
        compress.append(self.leaky_relu)

        for i in range(2):
            compress.append(nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1))
            compress.append(self.leaky_relu)
            curr_dim = curr_dim // 2

        self.compress = nn.Sequential(*compress)
        self.classifier = nn.Linear(curr_dim, num_domain)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.compress(x)
        x = x.view(-1, x.size(1))
        x = self.classifier(x)
        return x

class IMGDiscriminator(nn.Module):
    def __init__(self, image_size=512, conv_dim=128, channel=3, repeat_num=7, semantic_mask=False, num_domain=3, num_classes=19):
        """
        if semantic_mask:
            employed into structured semantic_mask img (semantic mask) discrimination
            Do not return fake/real or label classification results but return domain classification
        else:
            employed into pixel img discrimination
            return fake/real, domain, label classification results
        Both fake/real, domain classification follows PatchGAN idea
        """
        super(IMGDiscriminator, self).__init__()
        self.semantic_mask = semantic_mask

        init_block1 = [
            nn.Conv2d(channel, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(),
            nn.LeakyReLU(0.01),
            ResidualBlock(conv_dim, conv_dim, norm='SN'),
            ResidualBlock(conv_dim, conv_dim, norm='SN'),
            ResidualBlock(conv_dim, conv_dim, norm='SN'),
        ]


        init_block2 = [
            nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(),
            nn.LeakyReLU(0.01),
            ResidualBlock(conv_dim*2, conv_dim*2, norm='SN'),
            ResidualBlock(conv_dim*2, conv_dim*2, norm='SN'),
            ResidualBlock(conv_dim*2, conv_dim*2, norm='SN'),
        ]

        init_block3 = [
            nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(),
            nn.LeakyReLU(0.01),
            ResidualBlock(conv_dim*4, conv_dim*4, norm='SN'),
            ResidualBlock(conv_dim*4, conv_dim*4, norm='SN'),
            ResidualBlock(conv_dim*4, conv_dim*4, norm='SN'),
        ]

        curr_dim = conv_dim*4

        if not self.semantic_mask:
            assert channel == 3
            self.aux_clf = []
            for i in range(num_domain-1):
                self.aux_clf.append(Classifier_Module(curr_dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes))

        downsample = []
        for i in range(1, repeat_num-2):
            next_dim = curr_dim * 2 if not curr_dim > 2000 else curr_dim
            downsample.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
            downsample.append(nn.utils.spectral_norm())
            downsample.append(nn.LeakyReLU(0.01))
            curr_dim = next_dim

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.init_block1 = nn.Sequential(*init_block1)
        self.init_block2 = nn.Sequential(*init_block2)
        self.init_block3 = nn.Sequential(*init_block3)
        self.downsample = downsample
        self.conv1 = nn.Conv2d(next_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(next_dim, num_domain, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.init_block1(x)
        h = self.init_block2(h)
        h = self.init_block3(h)

        if self.semantic_mask:
            h = self.downsample(h)
            out_domain = self.conv2(h)
            return out_domain.view(out_domain.size(0), out_domain.size(1))

        else:
            out_aux = torch.cat([clf(h).unsqueeze_(0) for clf in self.aux_clf], dim=0)
            h = self.downsample(h)
            out_src = self.conv1(h)
            out_domain = self.conv2(h)
            return out_src, out_domain.view(out_domain.size(0), out_domain.size(1)), out_aux

