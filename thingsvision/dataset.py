#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['ImageDataset']

import os
import re
import torch

import numpy as np
import pandas as pd
import skimage.io as io
import thingsvision.vision as vision

from os.path import isdir as pisdir
from os.path import exists as pexists
from os.path import join as pjoin

from functools import cached_property
from PIL import Image
from torchvision import transforms as T
from typing import Tuple

def parse_img_name(img_name:str):
    return re.search(r'(.jpg|.png|.PNG)$', img_name)

def rm_suffix(img:str):
    return re.sub(r'(.jpg|.png|.PNG)$', '', img)

class ImageDataset(object):

    def __init__(
                 self,
                 PATH:str,
                 apply_transforms:bool,
                 things:bool,
                 things_behavior:bool,
                 resize_dim:int=256,
                 crop_dim:int=224,
                 k:int=12,
                 clip:bool=False,
                 transforms=None,
                 ):
        self.PATH = PATH
        self.apply_transforms = apply_transforms
        self.things = things
        self.things_behavior = things_behavior
        self.clip = clip

        if self.things:
            self.objs = sorted([obj for obj in os.listdir(PATH) if pisdir(pjoin(PATH, obj))])
            self.k = k #number of instances per object
            data_path = './data'
            concept_file = 'things_concepts.tsv'
            if not pexists(pjoin(data_path, concept_file)):
                os.mkdir(data_path)
                print(f'\n...Created PATH: {data_path}\n')
                raise FileNotFoundError(f'To extract activations for THINGS images concept file is required. Move {concept_file} to {data_path}.')

            self.concept_ids = pd.read_csv(pjoin(data_path, concept_file), encoding='utf-8', sep='\t').uniqueID.tolist()
            assert len(self.objs) == len(self.concept_ids), 'Number of categories in dataset must be equal to the number of concept IDs (check img folder)'
            self.objs = self.objs if self.objs == self.concept_ids else self.concept_ids

            self.imgs = [img for obj in self.objs for i, img in enumerate(os.listdir(pjoin(PATH, obj))) if parse_img_name(img) and i < self.k]

        else:
            if self.things_behavior:
                #sort objects according to item names in THINGS database
                self.objs = [pjoin(PATH, name + '.jpg').replace(PATH, '') for name in vision.load_item_names()]
            else:
                #sort objects in alphabetic order
                self.objs = sorted([obj for obj in os.listdir(PATH)])

            #check whether objs are paths or files
            n_folders = sum(pisdir(pjoin(self.PATH, obj)) for obj in self.objs)

            if n_folders == len(self.objs):
                self.imgs = [img for obj in self.objs for img in os.listdir(pjoin(PATH, obj)) if parse_img_name(img)]
            else:
                self.imgs = [img for img in self.objs if parse_img_name(img)]
                self.objs = list(map(rm_suffix, self.imgs))

        self.obj_mapping = dict(enumerate(self.objs))

        if self.apply_transforms:
            if not transforms and not self.clip:
                self.transforms = self.compose_transforms(resize_dim, crop_dim)
            else:
                assert self.clip, '\nCLIP models require a different image transformation than other torchvision models.\n'
                self.transforms = transforms

    def __getitem__(self, idx:int) -> Tuple:
        if len(self.imgs) == len(self.objs):
            img_path = pjoin(self.PATH, self.imgs[idx])
            img = Image.open(img_path) if self.clip else io.imread(img_path)
            resized_img = self.transforms(img)
            target = torch.tensor([idx])
            return resized_img, target
        else:
            obj_path = pjoin(self.PATH, self.objs[idx])
            imgs = np.array([img for img in os.listdir(obj_path) if parse_img_name(img)])
            if self.things:
                imgs = self.sort_imgs(imgs)

            resized_imgs, targets = [], []
            for i, img in enumerate(imgs):
                if self.things:
                    if i < self.k:
                        img = Image.open(pjoin(obj_path, img)) if self.clip else io.imread(pjoin(obj_path, img))

                        if self.apply_transforms:
                            resized_imgs.append(self.transforms(img))
                            targets.append(torch.tensor([idx]))
                        else:
                            resized_imgs.append(img)
                            targets.append(idx)
                else:
                    img = Image.open(pjoin(obj_path, img)) if self.clip else io.imread(pjoin(obj_path, img))
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

    @cached_property
    def obj2idx(self) -> dict:
        return {obj: idx for idx, obj in self.obj_mapping.items()}

    def flatten_dataset(self, split_data:bool=False):
        #set split_data to True, if you want to obtain img matrix X and target vector y separately
        if split_data:
            return zip(*[(img, target) for idx in range(len(self.objs)) for img, target in zip(*self.__getitem__(idx))])
        #set split_data to False, if you want individual (x, y) tuples
        else:
            return [(img, target) for idx in range(len(self.objs)) for img, target in zip(*self.__getitem__(idx))]

    #define transformations to be applied to imgs (i.e., imgs must be resized and normalized for pretrained CV models)
    def compose_transforms(self, resize_dim:int, crop_dim:int):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        composition = T.Compose([T.ToPILImage(), T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
        return composition

    def sort_imgs(self, imgs:np.ndarray) -> np.ndarray:
        img_identifiers = list(map(vision.get_digits, imgs))
        imgs_sorted = imgs[np.argsort(img_identifiers)]
        return imgs_sorted
