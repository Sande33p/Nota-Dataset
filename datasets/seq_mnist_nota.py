# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, store_masked_loaders_nota
from typing import Tuple
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from numpy.random import default_rng

from numpy.random import Generator, PCG64


class Nota(Dataset):

    def __init__(self, dataset, background, prob, notlabel=None):
        
        self.dataset = dataset
        self.prob = prob
        self.notlabel = notlabel
        self.background = background
        self.bsize = len(background)
        self.not_aug_transform = transforms.ToPILImage()

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        if np.random.random() < self.prob:
            img, targets = self.dataset[index]
        else:
            img, _ = self.background[np.random.randint(self.bsize)]
            targets = self.notlabel
        
        original_img = img

        if hasattr(self, 'logits'):
            return img, targets, original_img, self.logits[index]

        return img, targets, original_img

class BlockPermute1Channel:

    def __init__(self, size, blocksize):
        self.size = size
        self.blocksize = blocksize
        self.permutation = permuted_index(size, blocksize)

    def __call__(self, data):
        _, Nx, Ny = data.shape
        dflat = data.reshape([-1,Nx*Ny])
        dflat = dflat[:,torch.LongTensor(self.permutation)]
        return dflat.reshape([-1,Nx,Ny])


class BlockPermute3Channel:

    def __init__(self, size, blocksize):
        self.size = size
        self.blocksize = blocksize
        self.permutation = permuted_index(size, blocksize)

    def __call__(self, data):
        _, Nchan, Nx, Ny = data.shape
        dflat = data.reshape([-1, Nchan, Nx*Ny])
        dflat = dflat[:,:,torch.LongTensor(self.permutation)]
        return dflat.reshape([-1,Nchan, Nx, Ny])

def permuted_index(size, blocksize):
    N = size // blocksize
    d0 = np.arange(size*size)
    block = d0.reshape(N,
        blocksize,-1,blocksize).swapaxes(1,2).reshape(N*N,blocksize,blocksize)
    permuted = np.random.permutation(block)
    pfinal = permuted.reshape(-1,N,blocksize,blocksize).swapaxes(1,2).reshape(size*size)
    return pfinal




class SequentialMNIST_Nota(ContinualDataset):

    NAME = 'seq-mnist-nota'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK =  2 
    N_TASKS = 5 
    TRANSFORM = None
    

    def get_data_loaders(self):

        nota_prob = self.args.nota_prob

        print ("nota_prob",nota_prob)

        dset = datasets.MNIST('MNIST', train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))

        train_mask = np.logical_and(np.array(dset.targets) >= self.i,
        np.array(dset.targets) < self.i+self.N_CLASSES_PER_TASK)     

        dset.data = dset.data[train_mask]
        dset.targets = np.array(dset.targets)[train_mask]

        pdset = datasets.MNIST('MNIST', train=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                    BlockPermute1Channel(28,7)]))

        rng = Generator(PCG64(12345))
        train_mask_p = train_mask.copy()
        rng.shuffle(train_mask_p)

        pdset.data = pdset.data[train_mask_p]
        pdset.targets = np.array(pdset.targets)[train_mask_p]        

        train_dataset = Nota(dset, pdset, nota_prob, notlabel=10)

        inds1 = rng.choice(len(pdset.targets), int(len(pdset.targets)*nota_prob))
        inds2 = rng.choice(len(dset.targets), int(len(dset.targets)*nota_prob))
        
        train_dataset.data = np.concatenate((dset.data[inds2],pdset.data[inds1]),axis=0)
        train_dataset.targets = np.concatenate((dset.targets[inds2],pdset.targets[inds1]),axis=0)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:

            dset_te = datasets.MNIST('MNIST', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

            test_mask = np.logical_and(np.array(dset_te.targets) >= self.i,
            np.array(dset_te.targets) < self.i+self.N_CLASSES_PER_TASK)     

            dset_te.data = dset_te.data[test_mask]
            dset_te.targets = np.array(dset_te.targets)[test_mask]

            pdset_te = datasets.MNIST('MNIST', train=False,
                        transform=transforms.Compose([transforms.ToTensor(),
                        BlockPermute1Channel(28,7)]))

            rng = Generator(PCG64(12345))
            test_mask_p = test_mask.copy()
            rng.shuffle(test_mask_p)

            pdset_te.data = pdset_te.data[test_mask_p]
            pdset_te.targets = np.array(pdset_te.targets)[test_mask_p]        

            test_dataset = Nota(dset_te, pdset_te, nota_prob, notlabel=10)

        train, test = store_masked_loaders_nota(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        nota_prob = self.args.nota_prob

        dset = datasets.MNIST('MNIST', train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))

        train_mask = np.logical_and(np.array(dset.targets) >= self.i -
        self.N_CLASSES_PER_TASK, np.array(dset.targets) < self.i)     

        dset.data = dset.data[train_mask]
        dset.targets = np.array(dset.targets)[train_mask]

        pdset = datasets.MNIST('MNIST', train=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                    BlockPermute1Channel(28,7)]))

        rng = Generator(PCG64(12345))
        train_mask_p = train_mask.copy()
        rng.shuffle(train_mask_p)

        pdset.data = pdset.data[train_mask_p]
        pdset.targets = np.array(pdset.targets)[train_mask_p] 

        train_dataset = Nota(dset, pdset, nota_prob, notlabel=10)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        return train_loader

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, 1+SequentialMNIST_Nota.N_TASKS
                        * SequentialMNIST_Nota.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None
