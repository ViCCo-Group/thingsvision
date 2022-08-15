#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

import torch

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from os.path import join as pjoin

from PIL import Image
from typing import Tuple, List, Dict, Any


class ImageDataset(object):
    """Generic image dataset class

    Parameters
    ----------
    root : str
        Root directory. Directory from where to load images.
    out_path : str
        PATH where order of images features should be stored.
    imagenet_train : bool (optional)
        Whether ImageNet train set is used.
    imagenet_val : bool (optional)
        Whether ImageNet validation set is used.
    things : bool (optional)
        Whether THINGS database is used.
    things_behavior : bool (optional)
        Whether THINGS images used in behavioral experiments
        are used.
    add_ref_imgs : bool (optional)
        Whether the union of the THINGS database and those
        images that were used in behavioral experiments is used.
    file_names : List[str] (optional)
        List of file names. A list of file names that determines
        the order in which image features are extracted can optionally
        be passed.
    transforms : Any
        Composition of image transformations. Must be either a PyTorch composition
        or a Tensorflow Sequential model.

    Returns
    -------
    output : Dataset[Tuple[torch.Tensor, torch.Tensor]]
        Returns a generic image dataset of instance
        (i.e., image in matrix format) and target
        (i.e., class label) tensors.
    """

    def __init__(
        self,
        root: str,
        out_path: str,
        backend: str,
        imagenet_train: bool,
        imagenet_val: bool,
        things: bool,
        things_behavior: bool,
        add_ref_imgs: bool,
        file_names: List[str] = None,
        transforms=None,
    ):
        self.root = root
        self.backend = backend
        self.imagenet_train = imagenet_train
        self.imagenet_val = imagenet_val
        self.things = things
        self.things_behavior = things_behavior
        self.transforms = transforms
        self.file_names = file_names
        self.backend = backend

        classes, idx_to_cls, cls_to_idx, class_folders = self.find_classes_()
        self.class_dataset = class_folders

        if self.imagenet_val:
            img2cls, synsets = get_img2cls_mapping(self.root)
            self.classes = synsets

        if self.class_dataset:
            if self.file_names:
                _, cls_to_files = get_classes(self.file_names)
                self.samples = class_dataset(
                    self.root, out_path, cls_to_idx, self.imagenet_train, self.things, add_ref_imgs, cls_to_files)
            else:
                self.samples = class_dataset(
                    self.root, out_path, cls_to_idx, self.imagenet_train, self.things, add_ref_imgs)
            self.classes = classes
        else:
            self.samples = instance_dataset(
                self.root, out_path, classes, self.imagenet_val, img2cls if self.imagenet_val else None)
            if not self.imagenet_val:
                self.classes = list(cls_to_idx.keys())

        images, targets = zip(*self.samples)

        self.idx_to_cls = idx_to_cls
        self.cls_to_idx = cls_to_idx
        self.targets = targets
        self.images = images

    def find_classes_(self) -> Tuple[list, dict, dict]:
        children = sorted(
            [d.name for d in os.scandir(self.root) if d.is_dir()])
        if children:
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
                        raise FileNotFoundError(
                            f'To extract features for THINGS images, concept file is required. Move {concept_file} to {data_path}.')

                    concept_ids = pd.read_csv(pjoin(data_path, concept_file),
                                              encoding='utf-8', sep='\t').uniqueID.tolist()
                    assert len(children) == len(
                        concept_ids), '\nNumber of categories in dataset must be equal to the number of concept IDs. Check img folder.\n'
                    classes = children if children == concept_ids else concept_ids
                else:
                    classes = children
            if self.imagenet_train:
                cls_to_idx = get_syn2cls_mapping(self.root)
                idx_to_cls = {idx: cls for cls, idx in cls_to_idx.items()}
                classes = list(cls_to_idx.keys())
            else:
                idx_to_cls = dict(enumerate(classes))
        else:
            class_folders = False
            if self.things_behavior:
                # sort objects according to item names in THINGS database
                classes = [''.join((name, '.jpg'))
                           for name in load_item_names()]
            else:
                if self.file_names:
                    classes = list(filter(parse_img_name, self.file_names))
                else:
                    classes = sorted([f.name for f in os.scandir(self.root)
                                     if f.is_file() and parse_img_name(f.name)])
            idx_to_cls = dict(enumerate(list(map(rm_suffix, classes))))
        if not self.imagenet_train:
            cls_to_idx = {cls: idx for idx, cls in idx_to_cls.items()}
        return classes, idx_to_cls, cls_to_idx, class_folders

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img, target = self.transform_img_target(img, target)
            
        return img, target

    def transform_img_target(self, img, target):
        if self.transforms:
            if self.backend == 'pt':
                img = self.transforms(img)
                target = torch.tensor([target])
            elif self.backend == 'tf':
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = self.transforms(img)
                target = tf.convert_to_tensor(target)
        else:
            target = np.array([target])
        return img, target

    def __len__(self) -> int:
        return len(self.samples)


def get_syn2cls_mapping(root: str) -> Dict[str, int]:
    """Maps ImageNet classes (i.e., Wordnet synsets) to numeric labels."""
    try:
        with open(os.path.join(root, 'LOC_synset_mapping.txt'), 'r') as f:
            cls2syn = dict(
                enumerate(list(map(lambda x: x.split()[0], f.readlines()))))
    except FileNotFoundError:
        raise Exception(
            '\nCould not find LOC_synset_mapping.txt in root directory. Move file to root directory.\n')
    syn2cls = {syn: cls for cls, syn in cls2syn.items()}
    return syn2cls


def get_img2cls_mapping(root: str) -> Dict[str, int]:
    """Maps each image in the ImageNet subsample to its corresponding class."""
    split = root.split('/')[-1] if root.split('/')[-1] else root.split('/')[-2]
    try:
        solution = pd.read_csv(os.path.join(root, f'LOC_{split}_solution.csv'))
    except FileNotFoundError:
        raise Exception(
            f'\nCould not find LOC_{split}_solution.csv in root directory. Move file to root directory.\n')
    syn2cls = get_syn2cls_mapping(root)
    img2cls = {row[0] + '.JPEG': syn2cls[row[1].split()[0]]
               for idx, row in solution.iterrows()}
    img2cls = dict(sorted(img2cls.items(), key=lambda kv: kv[0]))
    return img2cls, list(syn2cls.keys())


def get_classes(file_names: List[str]) -> Tuple[List[str], Dict[str, list]]:
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


def parse_img_name(img_name: str) -> bool:
    return re.search(r'(.eps|.jpg|.JPG|.jpeg|.JPEG|.png|.PNG|.tif|.tiff)$', img_name)


def rm_suffix(img: str) -> str:
    return re.sub(r'(.eps|.jpg|.JPG|.jpeg|.JPEG|.png|.PNG|.tif|.tiff)$', '', img)


def instance_dataset(
    root: str,
    out_path: str,
    images: list,
    imagenet: bool = False,
    img2cls: dict = None,
) -> List[Tuple[str, int]]:
    """Creates a custom <instance> image dataset of image and class label pairs."""
    instances = []
    with open(os.path.join(out_path, 'file_names.txt'), 'w') as f:
        for img in images:
            f.write(f'{img}\n')
            instances.append(os.path.join(root, img))
    if imagenet:
        assert isinstance(
            img2cls, dict), '\nImage-to-cls mapping must be provided\n'
        samples = tuple(
            (instance, img2cls[instance.split('/')[-1]]) for instance in instances)
    else:
        samples = tuple((instance, target)
                        for target, instance in enumerate(instances))
    return samples


def class_dataset(
    PATH: str,
    out_path: str,
    cls_to_idx: Dict[str, int],
    imagenet: bool = None,
    things: bool = None,
    add_ref_imgs: bool = None,
    cls_to_files: Dict[str, List[str]] = None,
) -> List[Tuple[str, int]]:
    """Creates a custom <class> image dataset of image and class label pairs.

    Parameters
    ----------
        PATH : str
            Parent directory. Directory from where to load images.
        out_path : str
            PATH where order of images features should be stored.
        cls_to_idx : Dict[str, int]
            Dictionary of class to numeric label mapping.
        imagenet : bool (optional)
            Whether a subset of ImageNet is used.
        things : bool (optional)
            Whether the THINGS database is used.
        add_ref_imgs : bool (optional)
            Whether the union of the THINGS database and those
            images that were used in behavioral experiments is used.
        cls_to_files : Dict[str, List[str]] (optional)
            Dictionary that maps each class to a list of file names.
            For each class, the list of file names determines
            the order in which image features are extracted.

    Returns
    -------
    output : List[Tuple[str, int]]
        Returns a list of image and class label pairs.
    """
    samples = []
    with open(os.path.join(out_path, 'file_names.txt'), 'w') as f:
        classes = cls_to_idx.keys() if (things or imagenet) else sorted(cls_to_idx.keys())
        for target_cls in classes:
            cls_idx = cls_to_idx[target_cls]
            target_dir = os.path.join(PATH, target_cls)
            if os.path.isdir(target_dir):
                if cls_to_files is None:
                    for root, _, files in sorted(os.walk(target_dir, followlinks=True)):
                        if (things and add_ref_imgs):
                            first_img = sorted(files)[0]
                            if parse_img_name(first_img):
                                first_img = first_img.rstrip('.jpg')
                                if not first_img.endswith('b'):
                                    ref_img_path = get_ref_img(first_img)
                                    item = ref_img_path, cls_idx
                                    samples.append(item)
                                    f.write(f'{ref_img_path}\n')
                            else:
                                raise Exception(
                                    f'\nFound file that does not seem to be in the correct format: {first_img}.\nRemove file before proceeding with feature extraction.\n')
                        for file in sorted(files):
                            path = os.path.join(root, file)
                            if os.path.isfile(path) and parse_img_name(file):
                                item = path, cls_idx
                                samples.append(item)
                                f.write(f'{path}\n')
                else:
                    for f_name in cls_to_files[target_cls]:
                        path = os.path.join(target_dir, f_name)
                        if os.path.isfile(path) and parse_img_name(f_name):
                            item = path, cls_idx
                            samples.append(item)
                            f.write(f'{path}\n')
    return samples


def get_ref_img(
    first_img: str,
    folder: str = './reference_images/',
    suffix: str = '.jpg',
) -> str:
    """Create union of THINGS database and images used in behavioral experiments."""
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f'Directory for reference images not found. Move reference images to {folder}.')
    ref_images = [f.name for f in os.scandir(
        folder) if f.is_file() and parse_img_name(f.name)]
    for ref_img in ref_images:
        img_name = ref_img.rstrip(suffix)
        if re.search(f'^{img_name}', first_img):
            return os.path.join(folder, ref_img)

def load_item_names(folder: str = './data') -> np.ndarray:
    return pd.read_csv(pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values 
