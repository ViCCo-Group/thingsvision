#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['ImageDataset']

import os
import re
import torch

import numpy as np
import pandas as pd
import skimage.io as io

from torchvision import transforms as T
from typing import Tuple
from utils import get_digits

class ImageDataset(object):

    def __init__(
                 self,
                 PATH:str,
                 resize_dim:int,
                 crop_dim:int,
                 apply_transforms:bool,
                 things:bool,
                 k:int=12,
                 ):
        self.PATH = PATH
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        self.apply_transforms = apply_transforms
        self.things = things

        if self.things:
            self.objs = sorted([obj for obj in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, obj))])
            self.k = k #number of instances per object
            data_path = './data'
            concept_file = 'things_concepts.tsv'
            if not os.path.exists(os.path.join(data_path, concept_file)):
                os.mkdir(data_path)
                print(f'...Created PATH: {data_path}')
                print()
                raise FileNotFoundError(f'To extract activations for THINGS images concept file is required. Move {concept_file} to {data_path}.')

            self.concept_ids = pd.read_csv(os.path.join(data_path, concept_file), encoding='utf-8', sep='\t').uniqueID.tolist()
            assert len(self.objs) == len(self.concept_ids), 'Number of categories in dataset must be equal to the number of concept IDs (check img folder)'
            self.objs = self.objs if self.objs == self.concept_ids else self.concept_ids

            self.imgs = [img for obj in self.objs for i, img in enumerate(os.listdir(os.path.join(PATH, obj))) if img.endswith('.jpg') and i < self.k]

        else:
            self.objs = [obj for obj in os.listdir(PATH)]
            n_folders = sum(os.path.isdir(os.path.join(self.PATH, obj)) for obj in self.objs)
            if n_folders == len(self.objs):
                self.imgs = [img for obj in self.objs for img in os.listdir(os.path.join(PATH, obj)) if re.search(r'(.jpg|.png|.PNG)$', img)]
            else:
                self.imgs = [img for img in os.listdir(PATH) if re.search(r'(.jpg|.png|.PNG)$', img)]
                self.objs = list(map(lambda img: re.sub(r'(.jpg|.png|.PNG)$', '', img), self.imgs))

        self.obj_mapping = dict(enumerate(self.objs))

        if self.apply_transforms:
            self.transforms = self.compose_transforms(self.resize_dim, self.crop_dim)

    def __getitem__(self, idx:int) -> Tuple:
        if len(self.imgs) == len(self.objs):
            img_path = os.path.join(self.PATH, self.imgs[idx])
            img = io.imread(img_path)
            resized_img = self.transforms(img)
            target = torch.tensor([idx])
            return resized_img, target
        else:
            obj_path = os.path.join(self.PATH, self.objs[idx])
            imgs = np.array([img for img in os.listdir(obj_path) if re.search(r'(.jpg|.png|.PNG)$', img)])
            if self.things:
                imgs = self.sort_imgs(imgs)

            resized_imgs, targets = [], []
            for i, img in enumerate(imgs):
                if self.things:
                    if i < self.k:
                        img = io.imread(os.path.join(obj_path, img))

                        if self.apply_transforms:
                            resized_imgs.append(self.transforms(img))
                            targets.append(torch.tensor([idx]))
                        else:
                            resized_imgs.append(img)
                            targets.append(idx)
                else:
                    img = io.imread(os.path.join(obj_path, img))
                    if self.apply_transforms:
                        resized_imgs.append(self.transforms(img))
                        targets.append(torch.tensor([idx]))
                    else:
                        resized_imgs.append(img)
                        targets.append(idx)

            return resized_imgs, targets

    def __len__(self):
        return len(self.imgs)

    @property
    def idx2obj(self) -> dict:
        return self.obj_mapping

    @property
    def obj2idx(self) -> dict:
        return {obj: idx for idx, obj in self.obj_mapping.items()}


    def flatten_dataset(self, split_data:bool=False):
        #set split_data to True, if you want to obtain img matrix X and target vector y separately
        if split_data:
            return zip(*[(img, target) for idx in range(len(self.objs)) for img, target in zip(*self.__getitem__(idx))])
        #set split_data to False, if you want individual (x, y) tuples
        else:
            return [(img, target) for idx in range(len(self.objs)) for img, target in zip(*self.__getitem__(idx))]

    def sort_imgs(self, imgs:np.ndarray) -> np.ndarray:
        img_identifiers = list(map(get_digits, imgs))
        imgs_sorted = imgs[np.argsort(img_identifiers)]
        return imgs_sorted

    #define transformations to be applied to imgs (i.e., imgs must be resized and normalized for pretrained CV models)
    def compose_transforms(self, resize_dim:int, crop_dim:int):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        composition = T.Compose([T.ToPILImage(), T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
        return composition
