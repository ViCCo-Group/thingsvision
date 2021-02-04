#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'load_dl',
            'load_model',
            'show_model',
            'center_features',
            'normalize_features',
            'compress_features',
            'enumerate_layers',
            'extract_features',
            'get_cls_mapping_imgnet',
            'get_digits',
            'get_model',
            'get_shape',
            'json2dict',
            'merge_activations',
            'parse_imagenet_classes',
            'parse_imagenet_synsets',
            'store_activations',
            'split_activations',
            'slices2tensor',
            'tensor2slices',
            'load_item_names',
            'save_features',
            'save_targets',
            ]

import os
import random
import re
import torch

import numpy as np
import pandas as pd
import thingsvision.clip as clip
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from os.path import join as pjoin
from skimage.transform import resize
from typing import Tuple, List, Iterator, Dict, Any
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from thingsvision.dataset import ImageDataset

def load_dl(
             root:str,
             batch_size:int,
             things=None,
             things_behavior=None,
             add_ref_imgs=None,
             transforms=None,
             fraction=None,
             ) -> Iterator:
    print(f'\n...Loading dataset into memory.')
    dataset = ImageDataset(
                            root=root,
                            things=things,
                            things_behavior=things_behavior,
                            add_ref_imgs=add_ref_imgs,
                            transforms=transforms,
                            )
    print(f'...Transforming dataset into PyTorch DataLoader.\n')
    n_samples = len(dataset)
    if isinstance(fraction, float):
        n_subset = int(n_samples * fraction)
        rndperm = torch.randperm(n_samples)[:n_subset]
        subset = Subset(dataset, rndperm)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False)
    else:
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dl

def load_model(model_name:str, pretrained:bool, device:torch.device, model_path:str=None):
    """load a pretrained *torchvision* or OpenAI CLIP model into memory"""
    if re.search(r'^clip', model_name):
        assert isinstance(device, str), '\nFor CLIP models, name of device (str) must be provided.\n'
        if re.search(r'ViT$', model_name):
            model, transforms = clip.load("ViT-B/32", device=device, model_path=model_path, jit=False)
        else:
            model, transforms = clip.load("RN50", device=device, model_path=model_path, jit=False)
        model.eval()
        return model, transforms

    if not model_path:
        assert pretrained, '\nTo download a torchvision model directly from network, pretrained must be set to True.\n'

    if re.search(r'bn$', model_name):
        if re.search(r'^vgg13', model_name):
            model = models.vgg13_bn(pretrained=pretrained)
        elif re.search(r'^vgg16', model_name):
            model = models.vgg16_bn(pretrained=pretrained)
        elif re.search(r'^vgg19', model_name):
            model = models.vgg19_bn(pretrained=pretrained)
    else:
        if re.search(r'^alex', model_name):
            model = models.alexnet(pretrained=pretrained)
        elif re.search(r'^resnet50', model_name):
            model = models.resnet50(pretrained=pretrained)
        elif re.search(r'^resnet101', model_name):
            model = models.resnet101(pretrained=pretrained)
        elif re.search(r'^vgg13', model_name):
            model = models.vgg13(pretrained=pretrained)
        elif re.search(r'^vgg16', model_name):
            model = models.vgg16(pretrained=pretrained)
        elif re.search(r'^vgg19', model_name):
            model = models.vgg19(pretrained=pretrained)

    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    #move pretrained model to current device and set it to evaluation mode
    model.to(device)
    model.eval()
    transforms = compose_transforms()
    return model, transforms

def show_model(model, model_name:str) -> str:
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
    return module_name

def get_activation(name):
    """store hidden unit activations at each layer of model"""
    def hook(model, input, output):
        try:
            activations[name] = output.detach()
        except AttributeError:
            activations[name] = output
    return hook

def register_hook(model):
    """register a forward hook to store activations"""
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name))
    return model

def center_features(X:np.ndarray) -> np.ndarray:
    """center activations to have zero mean"""
    try:
        X -= X.mean(axis=0)
        return X
    except:
        raise Exception('\nMake sure features are represented through a two-dimensional array\n')

def normalize_features(X:np.ndarray) -> np.ndarray:
    """normalize feature vectors by their l2-norm"""
    try:
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        return X
    except:
        raise Exception(f'\nMake sure features are represented through a two-dimensional array\n')

def enumerate_layers(model, feature_extractor) -> List[int]:
    layers = []
    k = 0
    for n, m in model.named_modules():
        if re.search(r'\d+$', n):
            if isinstance(m, feature_extractor):
                layers.append(k)
            k += 1
    return layers

def ensemble_featmaps(activations:dict, layers:list, pooling:str='max', alpha:float=3., beta:float=5.) -> torch.Tensor:
    """concatenate globally (max or average) pooled feature maps"""
    acts = [activations[''.join(('features.', str(l)))] for l in layers]
    func = torch.max if pooling == 'max' else torch.mean
    pooled_acts = [torch.tensor([list(map(func, featmaps)) for featmaps in acts_i]) for acts_i in acts]
    pooled_acts[-2] = pooled_acts[-2] * alpha #upweight second-to-last conv layer by 5.
    pooled_acts[-1] = pooled_acts[-1] * beta #upweight last conv layer by 10.
    stacked_acts = torch.cat(pooled_acts, dim=1)
    return stacked_acts

def compress_features(X:np.ndarray, rnd_seed:int, retained_var:float=.9) -> np.ndarray:
    from sklearn.decomposition import PCA
    assert isinstance(rnd_seed, int), '\nTo reproduce results, random state for PCA must be defined.\n'
    pca = PCA(n_components=retained_var, svd_solver='full', random_state=rnd_seed)
    transformed_feats = pca.fit_transform(X)
    return transformed_feats

def extract_features(
                    model:Any,
                    data_loader:Any,
                    module_name:str,
                    batch_size:int,
                    flatten_acts:bool,
                    device:torch.device,
                    clip:bool=False,
                    feature_extractor=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """extract hidden unit activations (at specified layer) for every image in the database"""
    if re.search(r'ensemble$', module_name):
        ensembles = ['conv_ensemble', 'maxpool_ensemble']
        assert module_name in ensembles, f'\nIf aggregating filters across layers and subsequently concatenating activations, module name must be one of {ensembles}\n'
        if re.search(r'^conv', module_name):
            feature_extractor = nn.Conv2d
        else:
            feature_extractor = nn.MaxPool2d
    #initialise dictionary to store hidden unit activations on the fly
    global activations
    activations = {}
    #register forward hook to store activations
    model = register_hook(model)
    features, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = (t.to(device) for t in batch)
            X, y = batch
            if clip:
                img_features = model.encode_image(X)
                if module_name == 'visual':
                    assert torch.unique(activations[module_name] == img_features).item(), '\nImage features should represent activations in last encoder layer.\n'
            else:
                out = model(X)
            if re.search(r'ensemble$', module_name):
                layers = enumerate_layers(model, feature_extractor)
                act = ensemble_featmaps(activations, layers, 'max')
            else:
                act = activations[module_name]
                if flatten_acts:
                    if clip:
                        if re.search(r'attn$', module_name):
                            act = act[0]
                        else:
                            if act.size(0) != batch_size:
                                act = act.permute(1, 0, 2)
                    act = act.view(act.size(0), -1)
            features.append(act.cpu())
            targets.extend(y.squeeze(-1).cpu())

    #stack each mini-batch of hidden activations to obtain an N x F matrix, and flatten targets to yield vector
    features = np.vstack(features)
    targets = np.asarray(targets).ravel()
    assert len(features) == len(targets)
    print(f'\nFeatures successfully extracted for all {len(features)} images in the database\n')
    return features, targets

def store_activations(PATH:str, features:np.ndarray, file_format:str) -> None:
    if re.search(r'npy', file_format):
        with open(pjoin(PATH, 'activations.npy'), 'wb') as f:
            np.save(f, features)
    else:
        np.savetxt(pjoin(PATH, 'activations.txt'), features)

def tensor2slices(PATH:str, file_name:str, features:np.ndarray) -> None:
    with open(pjoin(PATH, file_name), 'w') as outfile:
        outfile.write(f'# Array shape: {features.shape}\n')
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                #formatting string indicates that we are writing out
                #the values in left-justified columns 7 characters in width
                #with 5 decimal places
                np.savetxt(outfile, features[i, j, :, :], fmt='%-7.5f')
                outfile.write('# New slice\n')

def get_shape(PATH:str, file:str) -> tuple:
    with open(pjoin(PATH, file)) as f:
        line = f.readline().strip()
        line = re.sub(r'\D', ' ', line)
        line = line.split()
        shape = tuple(map(int, line))
    return shape

def get_digits(string:str) -> int:
    c = ""
    nonzero = False
    for i in string:
        if i.isdigit():
            if (int(i) == 0) and (not nonzero):
                continue
            else:
                c += i
                nonzero = True
    return int(c)

def slices2tensor(PATH:str, file:str) -> np.ndarray:
    slices = np.loadtxt(pjoin(PATH, file))
    shape = get_shape(PATH, file)
    tensor = slices.reshape(shape)
    return tensor

def split_activations(PATH:str, features:np.ndarray, file_format:str, n_splits:int) -> None:
    splits = np.linspace(0, len(features), n_splits, dtype=int)
    for i in range(1, len(splits)):
        feature_split = features[splits[i-1]:splits[i]]
        if re.search(r'npy', file_format):
            with open(pjoin(PATH, f'activations_{i:02d}.npy'), 'wb') as f:
                np.save(f, feature_split)
        else:
            np.savetxt(pjoin(PATH, f'activations_{i:02d}.txt'), feature_split)

def merge_activations(PATH:str) -> np.ndarray:
    activation_splits = np.array([act for act in os.listdir(PATH) if re.search(r'^act', act) and re.search(r'[0-9]+', act) and act.endswith('.txt')])
    enumerations = np.array([int(re.sub(r'\D', '', act)) for act in activation_splits])
    activation_splits = activation_splits[np.argsort(enumerations)]
    activations = np.vstack([np.loadtxt(pjoin(PATH, act)) for act in activation_splits])
    return activations

def parse_imagenet_synsets(file_name:str, folder:str='./data/'):
    def parse_str(str):
        return re.sub(r'[^a-zA-Z]', '', str).rstrip('n').lower()
    imagenet_synsets = []
    with open(pjoin(folder, file_name), 'r') as f:
        for i, l in enumerate(f):
            l = l.split('_')
            cls = '_'.join(list(map(parse_str, l)))
            imagenet_synsets.append(cls)
    return imagenet_synsets

def parse_imagenet_classes(file_name:str, folder:str='./data/'):
    imagenet_classes = []
    with open(pjoin(folder, file_name), 'r') as f:
        for i, l in enumerate(f):
            l = l.strip().split()
            cls = '_'.join(l[1:]).rstrip(',').strip("'").lower()
            cls = cls.split(',')
            cls = cls[0]
            imagenet_classes.append(cls)
    return imagenet_classes

def get_class_intersection(imagenet_classes:list, things_objects:list) -> set:
    return set(things_objects).intersection(set(imagenet_classes))

def get_cls_mapping_imgnet(PATH:str, filename:str, save_as_json:bool) -> dict:
    """store ImageNet classes in a idx2cls dictionary, and subsequently save as .json file"""
    if re.search(r'synset', filename):
        imagenet_classes = parse_imagenet_synsets(filename, PATH)
    else:
        imagenet_classes = parse_imagenet_classes(filename, PATH)
    idx2cls = dict(enumerate(imagenet_classes))
    if save_as_json:
        filename = 'imagenet_idx2class.json'
        with open(pjoin(PATH, filename), 'w') as f:
            json.dump(idx2cls, f)
    return idx2cls

def json2dict(PATH:str, filename:str) -> dict:
    with open(pjoin(PATH, filename), 'r') as f:
        idx2cls = dict(json.load(f))
    return idx2cls

def compose_transforms(resize_dim:int=256, crop_dim:int=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    composition = T.Compose([T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
    return composition

def load_item_names(folder:str='./data') -> np.ndarray:
    return pd.read_csv(pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values

def save_features(features:np.ndarray, out_path:str, file_format:str, n_splits:int=10) -> None:
    if not os.path.exists(out_path):
        print(f'\nOutput directory did not exist. Creating directories to save features...\n')
        os.makedirs(out_path)
    #save hidden unit actvations to disk (either as one single file or as several splits)
    if len(features.shape) == 2:
        try:
            store_activations(PATH=out_path, features=features, file_format=file_format)
        except MemoryError:
            print(f'\n...Could not save activations as one single file due to memory problems.')
            print(f'...Now splitting activations along row axis into several batches.')
            split_activations(PATH=out_path, features=features, file_format=file_format, n_splits=n_splits)
            print(f'...Saved activations in {n_splits:02d} different files, enumerated in ascending order.')
            print(f'If you want activations to be splitted into more or fewer files, simply change number of splits parameter.\n')
    else:
        print(f'\n...Cannot save 4-way tensor in a single file.')
        print(f'...Now slicing tensor to store as a matrix.')
        tensor2slices(PATH=out_path, file_name='activations.txt', features=features)
        print(f'\n...Sliced tensor into separate parts, and saved resulting matrix as .txt file.\n')

def save_targets(targets:np.ndarray, out_path:str, file_format:str) -> None:
    if not os.path.exists(out_path):
        print(f'\nOutput directory did not exist. Creating directories to save targets...\n')
        os.makedirs(out_path)
    if re.search(r'npy', file_format):
        with open(pjoin(out_path, 'targets.npy'), 'wb') as f:
            np.save(f, targets)
    else:
        np.savetxt(pjoin(out_path, 'targets.txt'), targets)
    print(f'\nTargets successfully saved to disk.\n')
