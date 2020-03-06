import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, model_urls, load_state_dict_from_url
import os


class ParallelResNet101(ResNet):
    def __init__(self, gpus, num_classes, *args, **kwargs):
        super(ParallelResNet101, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        delattr(self, 'fc')

        self.pretrained_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
        self.gpu0 = gpus[0]
        self.gpu1 = gpus[1]

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2,
        ).to(self.gpu0)

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4
        ).to(self.gpu1)

    def forward(self, x):
        return self.seq2(self.seq1(x).to(self.gpu1))


class PipelineParallelResNet101(ParallelResNet101):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet101, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to(self.gpu1)
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(s_prev)

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to(self.gpu1)

        s_prev = self.seq2(s_prev)
        ret.append(s_prev)

        return torch.cat(ret)

