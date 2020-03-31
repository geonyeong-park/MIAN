import os.path

from PIL import Image
import torch.utils.data
import torchvision
import importlib
import numpy as np

from dataset.transforms import augment_collate

class MultiDomainLoader(object):
    def __init__(self, dataset, rootdir, resize,
                 batch_size=1, shuffle=False, num_workers=2, half_crop=None,
                 task='segmentation'):
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
        self.base_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ]
        self.dataset = dataset
        self.resize = resize
        self.half_crop = half_crop
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.task = task

        datadir = os.path.join(rootdir, 'data')
        txtdir = os.path.join(rootdir, 'dataset')

        self.source_dataset = []

        for source in self.dataset[:-1]:
            module = importlib.import_module('dataset.{}_dataset'.format(source))
            datadir_ = os.path.join(datadir, source) if task == 'segmentation' else '{}/office/{}/images'.format(datadir, source.lower())
            txtdir_ = os.path.join(txtdir, '{}_list'.format(source)) if task == 'segmentation' else None

            source_ = getattr(module, '{}DataSet'.format(source))(datadir_, txtdir_,
                                                                  resize=self.resize,
                                                                  base_transform=self.base_transform)
            self.source_dataset.append(source_)

        target = self.dataset[-1]
        module = importlib.import_module('dataset.{}_dataset'.format(target))
        datadir_ = os.path.join(datadir, target) if task == 'segmentation' else '{}/office/{}/images'.format(datadir, target.lower())
        txtdir_ = os.path.join(txtdir, '{}_list'.format(target))
        target_ = getattr(module, '{}DataSet'.format(target))(datadir_, txtdir_,
                                                            resize=self.resize,
                                                            base_transform=self.base_transform)
        self.target_dataset = target_

        target_val = getattr(module, '{}DataSet'.format(target))(datadir_, txtdir_,
                                                            split='val',
                                                            resize=self.resize,
                                                            base_transform=self.base_transform)
        self.target_valid_dataset = target_val

        for i,d in enumerate(self.source_dataset):
            print('{}-th source / {}: length={}'.format(i+1, d, len(self.source_dataset[i])))
        print('target {}: length={}'.format(self.dataset[-1], len(self.target_dataset)))

        self.n = max([len(i) for i in self.source_dataset] + [len(self.target_dataset)]) # make sure you see all images
        self.num = 0
        self.set_loader()
        self.iter_list = [iter(l) for l in self.loader_list]
        self.TargetLoader = TargetDomainLoader(self.target_valid_dataset, self.set_loader(target=True))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, return_target_label=False):
        img_label_list = []
        for i, iterator in enumerate(self.iter_list):
            try:
                img_label_list.append(next(iterator))
            except StopIteration:
                new_iterator = iter(self.loader_list[i])
                self.iter_list[i] = new_iterator
                img_label_list.append(next(new_iterator))

        img = torch.cat([i[0] for i in img_label_list], axis=0)

        if return_target_label:
            label = torch.cat([i[1] for i in img_label_list], axis=0)
        else:
            label = torch.cat([i[1] for i in img_label_list[:-1]], axis=0)

        self.num += 1
        return img, label

    def next_target_test(self):
        return self.next(return_target_label=True)

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))

    def set_loader(self, target=False):
        loader_list = []
        self.dataset_list = self.source_dataset + [self.target_dataset]

        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        assert num_workers == 1

        collate_fn = lambda batch: augment_collate(batch, crop=None, halfcrop=None, flip=True)

        if target:
            collate_fn=torch.utils.data.dataloader.default_collate
            loader_tgt = torch.utils.data.DataLoader(self.target_valid_dataset,
                    batch_size=batch_size, num_workers=num_workers, drop_last=True,
                    collate_fn=collate_fn, pin_memory=True)
            return loader_tgt

        for s in self.dataset_list:
            loader_src = torch.utils.data.DataLoader(s,
                    batch_size=batch_size, num_workers=num_workers, drop_last=True,
                    collate_fn=collate_fn, pin_memory=True)
            loader_list.append((loader_src))
        self.loader_list = loader_list


class TargetDomainLoader(object):
    def __init__(self, targetset, loader):
        self.targetset = targetset
        self.loader = loader
        self.iterator = iter(loader)
        self.num = 0

    def __len__(self):
        return len(self.targetset)

    def __next__(self):
        try:
            img, label = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            print('Reset TargetDomainLoader')
            img, label = next(self.iterator)

        return img, label

    def __iter__(self):
        return self
