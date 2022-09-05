#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset


class Augmentation:
    def __init__(self, path, image_size, batch_size):
        self.path = path
        self.image_size = image_size
        self.batch_size = batch_size

    def resize(self):
        return datasets.ImageFolder(self.path, transform=T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    def hue(self):
        return datasets.ImageFolder(self.path, transform=T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.ColorJitter(hue=0.5),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    def mirror(self):
        return datasets.ImageFolder(self.path, transform=T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.RandomHorizontalFlip(p=1.0),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    def rotate(self):
        return datasets.ImageFolder(self.path, transform=T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.RandomRotation(degrees=5),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    def transform(self):
        dataset_list = [self.resize(), self.hue(), self.mirror(), self.rotate()]
        dataset = ConcatDataset(dataset_list)
        return DataLoader(dataset, self.batch_size, shuffle=True, num_workers=4, pin_memory=False)