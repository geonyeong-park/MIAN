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
import cv2

def resize_img(data, size=32):
    tmp = []
    for img in data:
        tmp.append(cv2.resize(img, dsize=(size,size), interpolation=cv2.INTER_LINEAR))
    tmp = np.array(tmp)
    return tmp


class SVHNDataset(data.Dataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        self.root = root
        self.list_path = list_path
        self.resize = resize
        self.cropsize = cropsize
        self.img_pkl = os.path.join(root, '{}.pkl'.format(split))
        self.split = split

        with open(self.img_pkl, 'rb') as f:
            self.img = f['img']
            self.label = f['label']

        self.img = resize_img(self.img)
        if 'mnist' in self.__class__.__name__.lower() or 'usps' in self.__class__.__name__.lower():
            self.img = np.concatenate([self.img, self.img, self.img], axis=1)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        image = self.img[index]
        label = self.label[index]
        return image, label


class MNISTDataSet(SVHNDataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(MNISTDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

class MNISTMDataSet(SVHNDataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(MNISTMDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

class USPSDataSet(SVHNDataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(USPSDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

class SYNTHDataSet(SVHNDataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(SYNTHDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)
