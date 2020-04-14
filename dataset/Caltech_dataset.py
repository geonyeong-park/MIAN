
import os
import os.path as osp
import numpy as np
import random
from glob import glob
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from dataset.transforms import to_tensor_raw
from dataset.Amazon_dataset import AmazonDataSet


class CaltechDataSet(AmazonDataSet):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(CaltechDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

