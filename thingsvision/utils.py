#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'center_activations',
            'compose_transforms',
            'extract_features',
            'get_cls_mapping_imgnet',
            'get_digits',
            'get_model',
            'get_shape',
            'json2dict',
            'sparsity',
            'merge_activations',
            'parse_imagenet_classes',
            'parse_imagenet_synsets',
            'split_activations',
            'register_hook',
            'store_activations',
            'slices2tensor',
            'tensor2slices',
            'load_item_names',
            ]

import clip
import os
import re
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from os.path import join as pjoin
from skimage.transform import resize
from typing import Tuple, List
from torchvision import transforms as T

def get_model(model_name:str, pretrained:bool, model_path:str=None, device=None):
    """load a pretrained Torchvision or CLIP model of choice into memory"""
    if re.search(r'^clip', model_name):
        assert isinstance(device, str), '\nFor CLIP models, name of device (str) has to be provided.\n'
        if re.search(r'ViT$', model_name):
            model, transforms = clip.load("ViT-B/32", device=device, model_path=model_path, jit=False)
        else:
            model, transforms = clip.load("RN50", device=device, model_path=model_path, jit=False)
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

    return model

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

def normalize_features(X:np.ndarray) -> np.ndarray:
    """normalize feature vectors by their l2-norm"""
    X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
    return X

def center_activations(X:np.ndarray) -> np.ndarray:
    """center activations to have zero mean"""
    X -= X.mean(axis=0)
    return X

def enumerate_layers(model, feature_extractor) -> List[int]:
    layers = []
    k = 0
    for n, m in model.named_modules():
        if re.search(r'\d+$', n):
            if isinstance(m, feature_extractor):
                layers.append(k)
            k += 1
    return layers

def ensemble_featmaps(activations:dict, layers:list, pooling:str='max', alpha:float=5., beta:float=10.) -> torch.Tensor:
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
    pca = PCA(n_components=retained_var, svd_solver='full', random_state=rnd_seed)
    transformed_feats = pca.fit_transform(X)
    return transformed_feats

def extract_features(
                     model,
                     data_loader,
                     module_name:str,
                     batch_size:int,
                     flatten_acts:bool,
                     device:torch.device,
                     center_acts:bool,
                     compress_acts:bool,
                     normalize_reps:bool,
                     clip:bool=False,
                     feature_extractor=None,
                     rnd_seed=None,
) -> Tuple[np.ndarray]:
    """extract hidden unit activations (at specified layer) for every image in database"""
    #initialise dictionary to store hidden unit activations on the fly
    global activations
    activations = {}
    #register forward hook to store activations
    model = register_hook(model)
    #move pretrained model to current device and set it to evaluation mode
    model.to(device)
    model.eval()
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
                if re.search(r'pen', module_name):
                    act = torch.cat((act, activations['classifier.3']), dim=-1)
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

    if center_acts:
        assert re.search(r'(^classifier|ensemble$|^visual)', module_name) or flatten_acts or clip, \
        '\nMake sure features are represented through a two-dimensional array\n'
        #center features to have zero mean (centered around the origin of the coordinate system)
        features = center_activations(features)
    if normalize_reps:
        assert re.search(r'(^classifier|ensemble$|^visual)', module_name) or flatten_acts or clip, \
        '\nMake sure features are represented through a two-dimensional array\n'
        #normalize object representations by their respective l2-norms
        features = normalize_features(features)
    if re.search(r'ensemble$', module_name) and compress_acts:
        #transform features into lower-dimensional space via PCA (retain 90% or 95% of the variance)
        assert isinstance(rnd_seed, int), '\nTo reproduce results, random state for PCA must be defined\n'
        features = compress_features(features, rnd_seed)

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

def sparsity(A:np.ndarray) -> float:
    return 1.0 - (A[A>0].size/A.size)

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
    composition = T.Compose([T.ToPILImage(), T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
    return composition

def load_item_names(folder:str='./data') -> np.ndarray:
    return pd.read_csv(pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values
