import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from dataset.transforms import to_tensor_raw


class GTADataSet(data.Dataset):
    def __init__(self, root, list_path, base_transform=None, resize=(1024, 512), ignore_label=255):
        self.root = root
        self.list_path = list_path
        #self.resize = resize
        self.resize = (resize[1],resize[0])
        self.ignore_label = ignore_label
        self.img_ids = sorted([i_id.strip() for i_id in open(os.path.join(list_path, 'train_img.txt'))])
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.mean = np.array([104.00698793, 116.66876762, 122.67891434])
        image_transform = [torchvision.transforms.Resize(self.resize, interpolation=Image.BICUBIC)] + base_transform
        self.image_transform = torchvision.transforms.Compose(image_transform)

        label_transform = []
        label_transform.extend([torchvision.transforms.Resize(self.resize, interpolation=Image.NEAREST), to_tensor_raw])
        self.label_transform = torchvision.transforms.Compose(label_transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        # resize
        image = image.resize(self.resize, Image.BICUBIC)
        label = label.resize(self.resize, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(np.array(image.copy(), np.float32, copy=False))
        return image, to_tensor_raw(label_copy.copy())

"""
    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label_np = np.asarray(label)
        label_copy = self.ignore_label * np.ones_like(label_np)
        for k, v in self.id_to_trainid.items():
            label_copy[label_np == k] = v
        label = Image.fromarray(label_copy)
        name = datafiles["name"]

        image = self.image_transform(image)
        label = self.label_transform(label)

        return image, label
"""
