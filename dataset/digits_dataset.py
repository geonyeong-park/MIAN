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
import pickle as pkl
import cv2

def resize_img(data, size=32, expand=False):
    tmp = []
    for img in data:
        tmp.append(cv2.resize(img.transpose(1,2,0), dsize=(size,size),
                              interpolation=cv2.INTER_LINEAR))
    tmp = np.array(tmp)
    tmp = tmp.transpose(0, 3, 1, 2)
    return tmp


class SVHNDataSet(data.Dataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        self.root = root
        self.list_path = list_path
        self.resize = resize
        self.cropsize = cropsize
        self.img_pkl = os.path.join(root, '{}.pkl'.format(split))
        self.split = split

        with open(self.img_pkl, 'rb') as f:
            files = pkl.load(f)
            self.img = files['img']
            self.label = files['label']

        if 'mnist' in self.__class__.__name__.lower() or 'usps' in self.__class__.__name__.lower():
            if self.__class__.__name__ == 'MNISTMDataSet': pass
            else:
                self.img = np.concatenate([self.img, self.img, self.img], axis=1)
        self.img = resize_img(self.img)
        self.img = ((self.img / 255.0) - np.array([0.485, 0.456, 0.406]).reshape(-1, 3, 1, 1)) / \
            (np.array([0.229,0.224,0.225]).reshape(-1, 3, 1, 1))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        image = self.img[index]
        label = self.label[index]
        return image, label


class MNISTDataSet(SVHNDataSet):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(MNISTDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

class MNISTMDataSet(SVHNDataSet):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(MNISTMDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

class USPSDataSet(SVHNDataSet):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(USPSDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)

class SYNTHDataSet(SVHNDataSet):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        super(SYNTHDataSet, self).__init__(root, list_path, base_transform, resize, cropsize, split)
