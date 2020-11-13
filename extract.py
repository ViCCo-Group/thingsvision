#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import re
import torch

import numpy as np

from dataset import ImageDataset
from torch.utils.data import DataLoader, Subset
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING']=str(1)

def parseargs():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('-m', '--model_name', type=str,
        choices=['alexnet', 'resnet50', 'resnet101', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'],
        help='PyTorch vision model for which hidden unit activations should be extracted')
    aa('-it', '--interactive', action='store_true',
        help='whether or not to interact with terminal, and choose model part after looking at model architecture in terminal')
    aa('-mn', '--module_name', type=str, default=None,
        help='if in non-interactive mode, then module name for which hidden unit actvations should be extracted must be provided')
    aa('-fl', '--flatten_acts', action='store_true',
        help='whether or not to flatten activations at lower layers of the model (e.g., Convoluatonal layers) before storing them')
    aa('-bs', '--batch_size', type=int,
        help='define for how many images per mini-batch activations should be extracted')
    aa('-t', '--things', action='store_true',
        help='specify whether images are from the THINGS images database')
    aa('-f', '--fraction', type=float, default=None,
        help='specify fraction of dataset to be used, if you do not want to extract activations for all images')
    aa('-ff', '--file_format', type=str, default='.txt',
        choices=['.npy', '.txt'],
        help='specify in what kind of file format activations should be stored')
    aa('-ip', '--in_path', type=str, default='./images/',
        help='directory from where to load images')
    aa('-op', '--out_path', type=str, default='./activations/',
        help='directory where to save hidden unit activations')
    aa('-mp', '--model_path', type=str, default=None,
        help='directory where to load torchvision model weights from')
    aa('-dv', '--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('-rs', '--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def extract(
            model,
            model_name:str,
            module_name:str,
            batch_size:int,
            things:bool,
            file_format:str,
            in_path:str,
            out_path:str,
            flatten_acts:bool,
            device:torch.device,
            fraction=None,
) -> None:
    #set variables important for image transformations
    resize_dim = 256
    crop_dim = 224

    dataset = ImageDataset(PATH=in_path, resize_dim=resize_dim, crop_dim=crop_dim, apply_transforms=True, things=things)
    #get idx2class and class2idx mappings (i.e., dictionaries)
    idx2obj = dataset.idx2obj
    obj2idx = dataset.obj2idx

    #if dataset consists of more images than classes, then apply flattening
    if len(dataset.imgs) > len(dataset.objs):
        dataset = dataset.flatten_dataset(split_data=False)

    if isinstance(fraction, float):
        n_samples = len(dataset)
        n_subset = int(n_samples * fraction)
        rndperm = torch.randperm(n_samples)[:n_subset]
        subset = Subset(dataset, rndperm)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False)
    else:
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #perform feature extraction
    features, targets = extract_features(
                                         model=model,
                                         data_loader=dl,
                                         module_name=module_name,
                                         batch_size=batch_size,
                                         flatten_acts=flatten_acts,
                                         device=device,
                                         standardize_acts=False,
                                         )
    out_path = os.path.join(out_path, model_name, module_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #save hidden unit actvations to disk (either as one single file or as several splits)
    if len(features.shape) == 2:
        try:
            store_activations(PATH=out_path, features=features, file_format=file_format)
        except MemoryError:
            #if you want activations to be splitted into more or fewer files, just change number of splits
            n_splits = 10
            print(f'...Could not save activations as one single file due to memory problems.')
            print(f'...Now splitting activations along row axis into several batches.')
            print()
            split_activations(PATH=out_path, features=features, file_format=file_format, n_splits=n_splits)
            print(f'...Saved activations in {n_splits:02d} different files, enumerated in ascending order.')
            print()
    else:
        print(f'...Cannot save 4-way tensor in a single file.')
        print(f'...Now slicing tensor to store as a matrix.')
        print()
        tensor2slices(PATH=out_path, file_name='activations.txt', features=features)
        print(f'...Sliced tensor into separate parts, and saved resulting matrix as .txt file.')
        print()

    #save target vector to disk
    if re.search(r'npy', file_format):
        with open(os.path.join(out_path, 'targets.npy'), 'wb') as f:
            np.save(f, targets)
    else:
        np.savetxt(os.path.join(out_path, 'targets.txt'), targets)

if __name__ == '__main__':
    #parse all arguments
    args = parseargs()
    #set random seeds (important for reproducibility)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    #set device
    device = torch.device(args.device)

    #load pretrained torchvision model
    model_name = args.model_name
    model = get_model(model_name, args.model_path)

    if args.interactive:
        print(model)
        print()
        print(f'Enter part of the model for which you would like to extract neural actvations:\n')
        module_name = str(input())
        print()
    else:
        assert isinstance(args.module_name, str), 'in non-interactive mode, module name for which activations should be extracted must be provided'
        module_name = args.module_name

    #some variables to debug / resolve (potential) problems with CUDA
    if re.search(r'cuda', args.device):
        torch.cuda.manual_seed_all(args.rnd_seed)

    print(f'PyTorch CUDA version: {torch.version.cuda}')
    print()

    extract(
            model=model,
            model_name=model_name,
            module_name=module_name,
            batch_size=args.batch_size,
            things=args.things,
            file_format=args.file_format,
            in_path=args.in_path,
            out_path=args.out_path,
            flatten_acts=args.flatten_acts,
            device=device,
            fraction=args.fraction,
            )
