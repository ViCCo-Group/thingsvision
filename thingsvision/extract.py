#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import re
import torch

import numpy as np

from os.path import join as pjoin
from dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING']=str(1)

def parseargs():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model_name', type=str,
        choices=['alexnet', 'resnet50', 'resnet101', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'clip-ViT', 'clip-RN'],
        help='PyTorch vision or CLIP model for which hidden unit activations / image features should be extracted')
    aa('--interactive', action='store_true',
        help='whether or not to interact with terminal, and select model part after looking at model architecture in terminal')
    aa('--module_name', type=str, default=None,
        help='if in non-interactive mode, then module name for which hidden unit actvations should be extracted must be provided')
    aa('--flatten_acts', action='store_true',
        help='whether or not to flatten activations at lower layers of the model (e.g., Convoluatonal layers) before storing them')
    aa('--center_acts', action='store_true',
        help='whether or not to center features (move their mean towards zero) after extraction')
    aa('--normalize_reps', action='store_true',
        help='whether or not to normalize object representations by their respective l2-norms')
    aa('--compress_acts', action='store_true',
        help='whether or not to transform features into lower-dimensional space via PCA after extraction')
    aa('--batch_size', type=int,
        help='define for how many images per mini-batch activations should be extracted')
    aa('--things', action='store_true',
        help='specify whether images are from the THINGS image database (all images)')
    aa('--things_behavior', action='store_true',
        help='specify whether images are from the THINGS image database (only images used for behavioral experiments)')
    aa('--pretrained', action='store_true',
        help='specify whether to download a pretrained torchvision or CLIP model. If not provided, model_path has to be specified.')
    aa('--fraction', type=float, default=None,
        help='specify fraction of dataset to be used, if you do not want to extract activations for all images')
    aa('--file_format', type=str, default='.txt',
        choices=['.npy', '.txt'],
        help='specify in what kind of file format activations should be stored')
    aa('--in_path', type=str, default='./images/',
        help='directory from where to load images')
    aa('--out_path', type=str, default='./activations/',
        help='directory where to save hidden unit activations')
    aa('--model_path', type=str, default=None,
        help='directory where to load torchvision model weights from')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def extract(
            model,
            model_name:str,
            module_name:str,
            batch_size:int,
            things:bool,
            things_behavior:bool,
            file_format:str,
            in_path:str,
            out_path:str,
            flatten_acts:bool,
            center_acts:bool,
            normalize_reps:bool,
            compress_acts:bool,
            device:torch.device,
            fraction=None,
            transforms=None,
            rnd_seed=None,
) -> None:
    #init dataset
    dataset = ImageDataset(
                            PATH=in_path,
                            apply_transforms=True,
                            things=things,
                            things_behavior=things_behavior,
                            clip=True if re.search(r'clip', model_name) else False,
                            transforms=transforms,
                            )
    #get idx2class and class2idx mappings (i.e., dictionaries)
    idx2obj = dataset.idx2obj
    obj2idx = dataset.obj2idx

    #if dataset consists of more images than classes, then apply flattening
    if len(dataset.imgs) > len(dataset.objs):
        dataset = dataset.flatten_dataset(split_data=False)

    n_samples = len(dataset)
    if isinstance(fraction, float):
        n_subset = int(n_samples * fraction)
        rndperm = torch.randperm(n_samples)[:n_subset]
        subset = Subset(dataset, rndperm)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False)
    else:
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not re.search(r'^clip', model_name):
        if re.search(r'ensemble$', module_name):
            ensembles = ['conv_ensemble', 'maxpool_ensemble', 'convpen_ensemble', 'maxpoolpen_ensemble']
            assert module_name in ensembles, f'\nIf aggregating filters across layers and subsequently concatenating activations, module name must be one of {ensembles}\n'
            if re.search(r'^conv', module_name):
                feature_extractor = nn.Conv2d
            else:
                feature_extractor = nn.MaxPool2d
        else:
            feature_extractor = None
    else:
        feature_extractor = None

    #perform feature extraction
    features, targets = extract_features(
                                         model=model,
                                         data_loader=dl,
                                         module_name=module_name,
                                         batch_size=batch_size,
                                         flatten_acts=flatten_acts,
                                         device=device,
                                         center_acts=center_acts,
                                         normalize_reps=normalize_reps,
                                         compress_acts=compress_acts,
                                         clip=True if re.search(r'^clip', model_name) else False,
                                         feature_extractor=feature_extractor,
                                         rnd_seed=rnd_seed,
                                         )

    print(f'\nFeatures successfully extracted for all {n_samples} images in the database. Now saving to {out_path}.\n')

    out_path = pjoin(out_path, model_name, module_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #save hidden unit actvations to disk (either as one single file or as several splits)
    if len(features.shape) == 2:
        try:
            store_activations(PATH=out_path, features=features, file_format=file_format)
        except MemoryError:
            #if you want activations to be splitted into more or fewer files, simply change number of splits
            n_splits = 10
            print(f'\n...Could not save activations as one single file due to memory problems.\n')
            print(f'\n...Now splitting activations along row axis into several batches.\n')
            split_activations(PATH=out_path, features=features, file_format=file_format, n_splits=n_splits)
            print(f'\n...Saved activations in {n_splits:02d} different files, enumerated in ascending order.\n')
    else:
        print(f'\n...Cannot save 4-way tensor in a single file.\n')
        print(f'\n...Now slicing tensor to store as a matrix.\n')
        tensor2slices(PATH=out_path, file_name='activations.txt', features=features)
        print(f'\n...Sliced tensor into separate parts, and saved resulting matrix as .txt file.\n')

    #save target vector to disk
    if re.search(r'npy', file_format):
        with open(pjoin(out_path, 'targets.npy'), 'wb') as f:
            np.save(f, targets)
    else:
        np.savetxt(pjoin(out_path, 'targets.txt'), targets)

if __name__ == '__main__':
    #parse all arguments
    args = parseargs()
    #set random seeds (important for reproducibility)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    #set device
    device = torch.device(args.device)

    #load pretrained torchvision or clip model
    model_name = args.model_name
    if re.search(r'^clip', model_name):
        model, transforms = get_model(model_name=model_name, pretrained=args.pretrained, model_path=args.model_path, device=args.device)
    else:
        model = get_model(model_name=model_name, pretrained=args.pretrained, model_path=args.model_path)
        transforms = None

    if args.interactive:
        if re.search(r'^clip', model_name):
            for l, (n, p) in enumerate(model.named_modules()):
                if l > 1:
                    if re.search(r'^visual', n):
                        print(n)
            print('visual')
        else:
            print(model)
        print(f'\nEnter part of the model for which you would like to extract features:\n')
        module_name = str(input())
        print()
    else:
        assert isinstance(args.module_name, str), 'in non-interactive mode, module name for which activations should be extracted must be provided'
        module_name = args.module_name

    #some variables to debug / resolve (potential) problems with CUDA
    if re.search(r'cuda', args.device):
        torch.cuda.manual_seed_all(args.rnd_seed)

    print(f'\nPyTorch CUDA version: {torch.version.cuda}\n')

    extract(
            model=model,
            model_name=model_name,
            module_name=module_name,
            batch_size=args.batch_size,
            things=args.things,
            things_behavior=args.things_behavior,
            file_format=args.file_format,
            in_path=args.in_path,
            out_path=args.out_path,
            flatten_acts=args.flatten_acts,
            center_acts=args.center_acts,
            normalize_reps=args.normalize_reps,
            compress_acts=args.compress_acts,
            transforms=transforms,
            device=device,
            fraction=args.fraction,
            rnd_seed=args.rnd_seed if args.compress_acts else None,
            )
