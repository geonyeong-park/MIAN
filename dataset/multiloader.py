import os.path

from PIL import Image
import torch.utils.data
import torchvision
import importlib
import numpy as np

from transforms import augment_collate

class MultiDomainLoader(object):
    def __init__(self, dataset, rootdir, resize, crop_size=None,
                 batch_size=1, shuffle=False, num_workers=2, half_crop=None):
        """
        dataset: list of domains, ['Cityscapes', 'GTA5', ...]
        rootdir: root for data folders
        dataset list (txt files)
        dataset
        ㄴCityscapes_list
            ㄴtrain_img.txt
        resize: new (w, h)
        crop_size: randomly crop data for augmentation
        batch_size: per domain
        """
        self.base_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ])
        self.dataset = dataset
        self.resize = resize
        self.crop_size = crop_size
        self.half_crop = half_crop
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        datadir = os.path.join(rootdir, 'data')
        txtdir = os.path.join(rootdir, 'dataset')

        self.source_dataset = []

        for source in self.dataset[:-1]:
            module = importlib.import_module('dataset.{}_dataset'.format(source))
            datadir_ = os.path.join(datadir, source)
            txtdir_ = os.path.join(txtdir, '{}_list'.format(source))

            source_ = getattr(module, '{}DataSet'.format(source))(datadir_, txtdir_,
                                                                  resize=self.resize,
                                                                  transform=self.base_transform)
            self.source_dataset.append(source_)

        target = self.dataset[-1]
        module = importlib.import_module('dataset.{}_dataset'.format(target))
        datadir_ = os.path.join(datadir, target)
        txtdir_ = os.path.join(txtdir, '{}_list'.format(target))
        target_ = getattr(module, '{}DataSet'.format(target))(datadir_, txtdir_,
                                                            resize=self.resize,
                                                            transform=self.base_transform)
        self.target_dataset = target_

        for i,d in enumerate(self.source_dataset):
            print('{}-th source / {}: length={}'.format(i+1, d, len(self.source_dataset[i])))
        print('target {}: length={}'.format(self.dataset[-1], len(self.target_dataset)))

        self.n = max([len(i) for i in self.source_dataset] + [len(self.target)]) # make sure you see all images
        self.num = 0
        self.set_loader()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, return_target_label=False):
        # target_test에서 return label하기

        img_label_list = [next(d) for d in self.loader_list]
        img = np.concatenate([i[0] for i in img_label_list], axis=0)

        if return_target_label:
            label = np.concatenate([i[1] for i in img_label_list[:-1]], axis=0)
        else:
            label = np.concatenate([i[1] for i in img_label_list], axis=0)

        self.num += 1
        return img, label

    def next_target_test(self):
        return self.next(return_target_label=True)

    def __len__(self):
        return min(len(self.source), len(self.target))

    def set_loader(self):
        self.loader_list = []
        self.dataset_list = self.source_dataset + [self.target_dataset]

        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        if self.crop_size is not None:
            collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
                    halfcrop=self.half_crop, flip=True)
        else:
            collate_fn=torch.utils.data.dataloader.default_collate

        for s in self.dataset_list:
            loader_src = torch.utils.data.DataLoader(s,
                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                    collate_fn=collate_fn, pin_memory=True)
            self.loader_list.append(iter(loader_src))
