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
from GTA_dataset import GTADataSet


"""
dataset
ㄴCityscapes_list
    ㄴtrain_img.txt
    ㄴtrain_label.txt
    ㄴval_img.txt
    ㄴval_label.txt
ㄴGTA_list
    ㄴtrain_img.txt

data
ㄴCityscapes
    ㄴleftImg8bit
        ㄴtrain
            ㄴaugsbrug, ...
        ㄴval
            ㄴfrankfurt, ...
    ㄴgtCoarse  (work as train label)
        ㄴtrain
            ㄴaugsbrug, ...
        ㄴval
            ㄴfrankfurt, ...
    ㄴgtFine  (work as val label)
        ㄴtrain
            ㄴaugsbrug, ...
        ㄴval
            ㄴfrankfurt, ...
ㄴGTA (no val)
    ㄴimages
        ㄴ...
    ㄴlabels
        ㄴ...


"""

class CityscapesDataSet(GTADataSet):
    def __init__(self, root, list_path, transform=None, resize=(1024, 512), ignore_label=255, split='train'):
        super(CityscapesDataSet, self).__init__(root, list_path, transform, resize, ignore_label)
        self.files = []
        self.split = split
        self.img_ids = sorted([i_id.strip() for i_id in open(os.path.join(list_path, '{}_img.txt'.format(split)))])
        self.label_ids = sorted([l_id.strip() for l_id in open(os.path.join(list_path, '{}_label.txt'.format(split)))])

        label_root = 'gtCoarse' if self.split == 'train' else 'gtFine'
        for i, name in enumerate(self.img_ids):
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.split, name))
            label_file = osp.join(self.root, "%s/%s/%s" % (label_root, self.split, name))

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
