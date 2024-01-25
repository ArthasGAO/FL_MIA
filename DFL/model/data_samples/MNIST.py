# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 06:03:02 2023

@author: yuanzhe
"""
import os, sys
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset, ConcatDataset
import random


# %%
class MNISTDataset(Dataset):
    mnist_train = None
    mnist_test = None

    def __init__(self, batch_size=64, val_percent=0.1):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.batch_size = batch_size
        self.val_percent = val_percent

        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        if MNISTDataset.mnist_train is None:
            MNISTDataset.mnist_train = MNIST(
                f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
                # ToTensor() transform converts the PIL Image
                # in the range [0,255] to [0,1]
            )

        if MNISTDataset.mnist_test is None:
            MNISTDataset.mnist_test = MNIST(
                f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
            )

        data_train, data_val = random_split(MNISTDataset.mnist_train, [round(len(MNISTDataset.mnist_train) * (1 - 0.1)),
                                                                       round(len(MNISTDataset.mnist_train) * 0.1)])

        self.train_set = data_train
        self.val_set = data_val
        self.test_set = MNISTDataset.mnist_test
