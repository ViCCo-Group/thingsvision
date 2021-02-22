#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['ImageDataset']

import os
import re
import torch

import numpy as np
import pandas as pd

import thingsvision.vision as vision

from collections import defaultdict
from os.path import join as pjoin

from PIL import Image
from torchvision import transforms as T
from typing import Tuple, List, Dict, Iterator, Any

def get_classes(file_names:List[str]) -> Tuple[List[str], Dict[str, list]]:
    cls_to_files = defaultdict(list)
    classes = []
    for file in file_names:
        if re.search(r'\\', file):
            cls, f_name = file.split('\\')
        elif re.search(r'(/|//)', file):
            cls, f_name = file.split('/')
        else:
            continue
        if cls not in classes:
            classes.append(cls)
        cls_to_files[cls].append(f_name)
    return classes, cls_to_files

def parse_img_name(img_name:str) -> bool:
    return re.search(r'(.eps|.jpg|.jpeg|.png|.PNG|.tif|.tiff)$', img_name)

def rm_suffix(img:str) -> str:
    return re.sub(r'(.eps|.jpg|.jpeg|.png|.PNG|.tif|.tiff)$', '', img)

def instance_dataset(root:str, out_path:str, images:list) -> List[Tuple[str, int]]:
    instances = []
    with open(os.path.join(out_path, 'file_names.txt'), 'w') as f:
        for img in images:
            f.write(f'{img}\n')
            instances.append(os.path.join(root, img))
    samples = tuple((instance, target) for target, instance in enumerate(instances))
    return samples

def class_dataset(PATH:str, out_path:str, cls_to_idx:Dict[str, int], things:bool=None, add_ref_imgs:bool=None, cls_to_files:Dict[str, list]=None) -> List[Tuple[str, int]]:
    samples = []
    with open(os.path.join(out_path, 'file_names.txt'), 'w') as f:
        for target_cls in sorted(cls_to_idx.keys()):
            cls_idx = cls_to_idx[target_cls]
            target_dir = os.path.join(PATH, target_cls)
            if os.path.isdir(target_dir):
                if cls_to_files is None:
                    for root, _, files in sorted(os.walk(target_dir, followlinks=True)):
                        if (things and add_ref_imgs):
                            first_img = files[0].rstrip('.jpg')
                            if not first_img.endswith('b'):
                                ref_img_path = get_ref_img(first_img)
                                item = ref_img_path, class_idx
                                instances.append(item)
                        for k, file in enumerate(sorted(files)):
                            path = os.path.join(root, file)
                            f.write(f'{path}\n')
                            if os.path.isfile(path) and parse_img_name(file):
                                item = path, cls_idx
                                samples.append(item)
                else:
                    for f_name in cls_to_files[target_cls]:
                        path = os.path.join(target_dir, f_name)
                        f.write(f'{path}\n')
                        if os.path.isfile(path) and parse_img_name(f_name):
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
        out_path (str) - output directory where to save image features (to store .txt file with corresponding file_names)
        things (bool) - whether images are from the THINGS database
        things_behavior (bool) - whether images correspond to the THINGS images used in the behavioral experiments
        add_ref_imgs (bool) - whether to prepend references images (i.e., images used in behavioral experiments) to the *full* THINGS image dataset
        file_names (List[str]) - whether extracted features should be sorted according to a provided list of file names (following ['class/img_xy.png', ...] OR ['img_xy.png', ...])
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
                out_path:str,
                things:bool,
                things_behavior:bool,
                add_ref_imgs:bool,
                file_names:List[str]=None,
                transforms=None,
                ):
        self.root = root
        self.things = things
        self.things_behavior = things_behavior
        self.transforms = transforms
        self.file_names = file_names

        classes, idx_to_cls, cls_to_idx, class_folders = self.find_classes_()
        self.class_dataset = class_folders

        if self.class_dataset:
            if self.file_names:
                _, cls_to_files = get_classes(self.file_names)
                self.samples = class_dataset(self.root, out_path, cls_to_idx, self.things, add_ref_imgs, cls_to_files)
            else:
                self.samples = class_dataset(self.root, out_path, cls_to_idx, self.things, add_ref_imgs)
            self.classes = classes
        else:
            self.samples = instance_dataset(self.root, out_path, classes)
            self.classes = list(cls_to_idx.keys())

        images, targets = zip(*self.samples)

        self.idx_to_cls = idx_to_cls
        self.cls_to_idx = cls_to_idx
        self.targets = targets

    def find_classes_(self) -> Tuple[list, dict, dict]:
        classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        if classes:
            class_folders = True
            if self.file_names:
                classes, _ = get_classes(self.file_names)
            else:
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
                if self.file_names:
                    classes = [f for f in self.file_names if os.path.isfile(f) and parse_img_name(f)]
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
