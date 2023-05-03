from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 


crop_size = 32
padding = 4

def prepare_cifar100_train_dataset(data_dir, dataset='cifar100', batch_size=128, 
                                    shuffle=True, num_workers=4, pin_memory=True):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = torchvision.datasets.ImageNet(root=data_dir, split='train', transform=train_transform)
    return train_dataset

def prepare_cifar100_test_dataset(data_dir, dataset='cifar100', batch_size=128, 
                                    shuffle=False, num_workers=4, pin_memory=True):
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    testset = torchvision.datasets.ImageNet(root=data_dir, split='val', transform=test_transform)
    return testset
