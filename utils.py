#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'compose_transforms',
            'extract_features',
            'get_cls_mapping_imgnet',
            'get_digits',
            'get_model',
            'get_shape',
            'json2dict',
            'matrix_sparseness',
            'merge_activations',
            'split_activations',
            'slices2tensor',
            'tensor2slices',
            ]

import os
import re
import torch

import numpy as np
import torch.nn as nn
import torchvision.models as models

from typing import Tuple
from torchvision import transforms as T

def get_model(model_name:str, model_path:str, pretrained:bool=False):
    """load a pretrained torchvision model of choice into memory"""
    if re.search(r'bn', model_name):
        if re.search(r'vgg13', model_name):
            model = models.vgg13_bn(pretrained=pretrained)
        elif re.search(r'vgg16', model_name):
            model = models.vgg16_bn(pretrained=pretrained)
        elif re.search(r'vgg19', model_name):
            model = models.vgg19_bn(pretrained=pretrained)
    else:
        if re.search(r'alex', model_name):
            model = models.alexnet(pretrained=pretrained)
        elif re.search(r'resnet50', model_name):
            model = models.resnet50(pretrained=pretrained)
        elif re.search(r'resnet101', model_name):
            model = models.resnet101(pretrained=pretrained)
        elif re.search(r'vgg13', model_name):
            model = models.vgg13(pretrained=pretrained)
        elif re.search(r'vgg16', model_name):
            model = models.vgg16(pretrained=pretrained)
        elif re.search(r'vgg19', model_name):
            model = models.vgg19(pretrained=pretrained)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def get_activation(name):
    """store hidden unit activations at each layer of model"""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def register_hook(model):
    """register a forward hook to store activations"""
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name))
    return model

def standardize_features(X:np.ndarray) -> np.ndarray:
    """standardize features to have zero mean and std of one"""
    std = X.std(axis=0, keepdims=True)
    mean = X.mean(axis=0, keepdims=True)
    X -= mean #center features
    X /= std #standardize features
    return X

def extract_features(
                     model,
                     data_loader,
                     module_name:str,
                     batch_size:int,
                     flatten_acts:bool,
                     device:torch.device,
                     standardize_acts:bool,
) -> Tuple[np.ndarray, np.ndarray]:
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
            out = model(X)
            act = activations[module_name]
            if flatten_acts:
                act = act.view(act.size(0), -1)
            features.append(act.cpu())
            targets.extend(y.squeeze(-1).cpu())

    #stack each mini-batch of hidden activations to obtain an N x F matrix, and flatten targets to yield vector
    features = np.vstack(features)
    targets = np.asarray(targets).ravel()
    assert len(features) == len(targets)
    if re.search(r'classifier', module_name) and standardize_acts:
        #standardize features iff they are extracted from classifier part of the model
        features = standardize_features(features)
    return features, targets

def store_activations(PATH:str, features:np.ndarray, file_format:str) -> None:
    if re.search(r'npy', file_format):
        with open(os.path.join(PATH, 'activations.npy'), 'wb') as f:
            np.save(f, features)
    else:
        np.savetxt(os.path.join(PATH, 'activations.txt'), features)

def tensor2slices(PATH:str, file_name:str, features:np.ndarray) -> None:
    with open(os.path.join(PATH, file_name), 'w') as outfile:
        outfile.write(f'# Array shape: {features.shape}\n')
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                #formatting string indicates that we are writing out
                #the values in left-justified columns 7 characters in width
                #with 5 decimal places
                np.savetxt(outfile, features[i, j, :, :], fmt='%-7.5f')
                outfile.write('# New slice\n')

def get_shape(PATH:str, file:str) -> tuple:
    with open(os.path.join(PATH, file)) as f:
        line = f.readline().strip()
        line = re.sub(r'\D', ' ', line)
        line = line.split()
        shape = tuple(map(int, line))
    return shape

def slices2tensor(PATH:str, file:str) -> np.ndarray:
    slices = np.loadtxt(os.path.join(PATH, file))
    shape = get_shape(PATH, file)
    tensor = slices.reshape(shape)
    return tensor

def split_activations(PATH:str, features:np.ndarray, file_format:str, n_splits:int) -> None:
    splits = np.linspace(0, len(features), n_splits, dtype=int)
    for i in range(1, len(splits)):
        feature_split = features[splits[i-1]:splits[i]]
        if re.search(r'npy', file_format):
            with open(os.path.join(PATH, f'activations_{i:02d}.npy'), 'wb') as f:
                np.save(f, feature_split)
        else:
            np.savetxt(os.path.join(PATH, f'activations_{i:02d}.txt'), feature_split)

def merge_activations(PATH:str) -> np.ndarray:
    activation_splits = np.array([act for act in os.listdir(PATH) if re.search(r'^act', act) and re.search(r'[0-9]+', act) and act.endswith('.txt')])
    enumerations = np.array([int(re.sub(r'\D', '', act)) for act in activation_splits])
    activation_splits = activation_splits[np.argsort(enumerations)]
    activations = np.vstack([np.loadtxt(os.path.join(PATH, act)) for act in activation_splits])
    return activations

def matrix_sparseness(A:np.ndarray) -> np.ndarray:
    """return average sparsity of input matrix"""
    def vector_energy(v:np.ndarray) -> np.ndarray:
        """this function rates the energy of a vector v with values in the closed interval [0, 1]"""
        x = np.sqrt(len(v)) - np.linalg.norm(v, ord=1) / np.linalg.norm(v, ord=2)
        y = 1 / (np.sqrt(len(v) - 1))
        s = x * y
        return s
    avg_sparsity = np.mean(list(map(lambda v: vector_energy(v), A)))
    return avg_sparsity

def get_cls_mapping_imgnet(PATH:str, filename:str, save_as_json:bool) -> dict:
    """load ImageNet classes as idx2cls dictionary, and subsequently save as .json file"""
    idx2cls = {}
    with open(os.path.join(PATH, filename), 'r') as f:
        for i, l in enumerate(f):
            if re.search(r'synset', filename):
                def parse_str(str):
                    return re.sub(r'[^a-zA-Z]', '', str).rstrip('n').capitalize()
                l = l.split('_')
                cls = ' '.join(list(map(parse_str, l)))
            else:
                l = l.strip().split()
                cls = ' '.join(l[1:]).rstrip(',').strip("'").capitalize()
                cls = cls.split(',')
                cls = cls[0]
            idx2cls[i] = cls

    if save_as_json:
        filename = 'imagenet_idx2class.json'
        with open(os.path.join(PATH, filename), 'w') as f:
            json.dump(idx2cls, f)

    return idx2cls

def json2dict(PATH:str, filename:str) -> dict:
    with open(os.path.join(PATH, filename), 'r') as f:
        idx2cls = dict(json.load(f))
    return idx2cls

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

def compose_transforms(resize_dim:int, crop_dim:int):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    composition = T.Compose([T.ToPILImage(), T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
    return composition
