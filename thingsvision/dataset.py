#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['ImageDataset']

import os
import re
import torch

import numpy as np
import pandas as pd
import thingsvision.vision as vision

from os.path import join as pjoin

from PIL import Image
from torchvision import transforms as T
from typing import Tuple, List, Dict, Any

def parse_img_name(img_name:str) -> bool:
    return re.search(r'(.jpg|.jpeg|.png|.PNG|.tif|.tiff)$', img_name)

def rm_suffix(img:str) -> str:
    return re.sub(r'(.jpg|.jpeg|.png|.PNG|.tif|.tiff)$', '', img)

def instance_dataset(root:str, images:list) -> List[Tuple[str, int]]:
    instances = [os.path.join(root, img) for img in images]
    samples = tuple((instance, target) for target, instance in enumerate(instances))
    return samples

def class_dataset(PATH:str, cls_to_idx:Dict[str, int], things=None, add_ref_imgs=None) -> List[Tuple[str, int]]:
    samples = []
    for target_cls in sorted(cls_to_idx.keys()):
        cls_idx = cls_to_idx[target_cls]
        target_dir = os.path.join(PATH, target_cls)
        if os.path.isdir(target_dir):
            for root, _, files in sorted(os.walk(target_dir, followlinks=True)):
                if (things and add_ref_imgs):
                    first_img = files[0].rstrip('.jpg')
                    if not first_img.endswith('b'):
                        ref_img_path = get_ref_img(first_img)
                        item = ref_img_path, class_idx
                        instances.append(item)
                for k, file in enumerate(sorted(files)):
                    path = os.path.join(root, file)
                    if os.path.isfile(path) and parse_img_name(file):
                        item = path, cls_idx
                        samples.append(item)
    return samples

def get_ref_img(first_img:str, folder:str='./reference_images/') -> str:
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Directory for reference images not found. Move reference images to {folder}.')
    ref_images = [f.name for f in os.scandir(folder) if f.is_file() and parse_img_name(f)]
    for ref_img in ref_images:
        img_name = ref_img.rstrip('.jpg')
        if re.search(f'^{img_name}', first_img):
            return os.path.join(folder, img_name)

class ImageDataset(object):
    """
        :params:
        root (str) - parent directory from where to load images
        things (bool) - whether images are from the THINGS database
        things_behavior (bool) - whether images correspond to the THINGS images used in the behavioral experiments
        add_ref_imgs (bool) - whether to prepend references images (i.e., images used in behavioral experiments) to the *full* THINGS image dataset
        transforms (Any) - whether to apply a composition of image transformations

        :args:
        transforms (Any) - composition of image transformations
        samples (tuple) - tuple of image (PIL Image or torch.Tensor) and class index (np.ndarray or torch.Tensor) pairs
        idx_to_cls (dict) - index-to-class mapping
        cls_to_idx (dict) - class-to-index mapping
        targets (tuple) - tuple of class indices (stored as np.ndarray or torch.Tensor)
    """
    def __init__(
                self,
                root:str,
                things:bool,
                things_behavior:bool,
                add_ref_imgs:bool,
                transforms=None,
                ):
        self.root = root
        self.things = things
        self.things_behavior = things_behavior
        self.transforms = transforms

        classes, idx_to_cls, cls_to_idx, class_folders = self.find_classes_()
        self.class_dataset = class_folders

        if self.class_dataset:
            self.samples = class_dataset(self.root, cls_to_idx)
            self.classes = classes
        else:
            self.samples = instance_dataset(self.root, classes)
            self.classes = list(cls_to_idx.keys())

        images, targets = zip(*self.samples)

        self.idx_to_cls = idx_to_cls
        self.cls_to_idx = cls_to_idx
        self.targets = targets

    def find_classes_(self) -> Tuple[list, dict, dict]:
        classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        if classes:
            class_folders = True
            if self.things:
                data_path = './data'
                concept_file = 'things_concepts.tsv'
                if not os.path.exists(os.path.join(data_path, concept_file)):
                    os.mkdir(data_path)
                    print(f'\n...Created PATH: {data_path}\n')
                    raise FileNotFoundError(f'To extract features for THINGS images, concept file is required. Move {concept_file} to {data_path}.')

                concept_ids = pd.read_csv(pjoin(data_path, concept_file), encoding='utf-8', sep='\t').uniqueID.tolist()
                assert len(classes) == len(concept_ids), '\nNumber of categories in dataset must be equal to the number of concept IDs. Check img folder.\n'
                classes = classes if classes == concept_ids else concept_ids
            idx_to_cls = dict(enumerate(classes))
        else:
            class_folders = False
            if self.things_behavior:
                #sort objects according to item names in THINGS database
                classes = [''.join((name,'.jpg')) for name in vision.load_item_names()]
            else:
                classes = sorted([f.name for f in os.scandir(self.root) if f.is_file() and parse_img_name(f.name)])
            idx_to_cls = dict(enumerate(list(map(rm_suffix, classes))))
        cls_to_idx = {cls:idx for idx, cls in idx_to_cls.items()}
        return classes, idx_to_cls, cls_to_idx, class_folders

    def __getitem__(self, idx:int) -> Tuple[Any, Any]:
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
            target = torch.tensor([target])
        else:
            target = np.array([target])
        return img, target

    def __len__(self) -> int:
        return len(self.samples)
