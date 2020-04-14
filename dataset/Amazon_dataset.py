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


class AmazonDataSet(data.Dataset):
    def __init__(self, root, list_path=None, base_transform=None, resize=300, cropsize=256, split='train'):
        self.root = root
        self.list_path = list_path
        self.resize = resize
        self.cropsize = cropsize
        self.img_folders = sorted(glob(self.root + '/*'))
        self.files = []
        self.split = split

        for i, folder in enumerate(self.img_folders):
            for img in glob(folder + '/*'):
                label = i
                self.files.append({
                    "img": img,
                    "label": i,
                })

        if split == 'train':
            assert resize >= cropsize
            if resize > cropsize:
                image_transform = [torchvision.transforms.Resize(self.resize, interpolation=Image.BICUBIC),
                                   torchvision.transforms.RandomCrop(self.cropsize)] + base_transform
            else:
                image_transform = [torchvision.transforms.Resize(self.cropsize, interpolation=Image.BICUBIC)] + base_transform

        elif split == 'val':
            image_transform = [torchvision.transforms.Resize(self.cropsize, interpolation=Image.BICUBIC)] + base_transform

        self.image_transform = torchvision.transforms.Compose(image_transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = datafiles["label"]
        image = self.image_transform(image)
        label = torch.from_numpy(np.array(label, np.int32, copy=False))
        return image, label

