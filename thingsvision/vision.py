#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import json
import os
import pickle
import re
import scipy
import scipy.io
import torch
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import thingsvision.cornet as cornet
import thingsvision.clip as clip

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import defaultdict
from numba import njit, prange
from os.path import join as pjoin
from scipy.stats import rankdata
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from thingsvision.model_class import Model
from thingsvision.dataset import ImageDataset
from thingsvision.dataloader import DataLoader
from torchvision import transforms as T
from typing import Tuple, List, Iterator, Dict, Any


FILE_FORMATS = ['hdf5', 'npy', 'mat', 'txt']



def load_dl(
    root: str,
    out_path: str,
    backend: str,
    batch_size: int,
    imagenet_train: bool = None,
    imagenet_val: bool = None,
    things: bool = None,
    things_behavior: bool = None,
    add_ref_imgs: bool = None,
    file_names: List[str] = None,
    transforms=None,
) -> Iterator:
    """Create a data loader for custom image dataset

    Parameters
    ----------
    root : str
        Root directory. Directory where images are stored.
    out_path : str
        PATH where order of images features should be stored.
    batch_size : int (optional)
        Number of samples (i.e., images) per mini-batch.
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
    output : Iterator
        Returns an iterator of image mini-batches.
        Each mini-batch consists of <batch_size> samples.
    """
    print(f'\n...Loading dataset into memory.')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f'...Creating output directory.')
    dataset = ImageDataset(
        root=root,
        out_path=out_path,
        backend=backend,
        imagenet_train=imagenet_train,
        imagenet_val=imagenet_val,
        things=things,
        things_behavior=things_behavior,
        add_ref_imgs=add_ref_imgs,
        file_names=file_names,
        transforms=transforms,
    )
    print(f'...Transforming dataset into {backend} DataLoader.\n')
    dl = DataLoader(dataset, batch_size=batch_size, backend=backend)
    return dl


def get_module_names(model, module: str) -> list:
    """Extract correct module names, if iterating over multiple modules is desired."""
    if model.backend == 'pt':
        module_names, _ = zip(*model.model.named_modules())
    else:
        module_names = list(map(lambda m: m.name, model.model.layers))    
    return list(filter(lambda n: re.search(f'{module}', n), module_names))


def extract_features_across_models_and_datasets(
    out_path: str,
    model_names: List[str],
    img_paths: List[str],
    module_names: List[str],
    clip: List[bool],
    pretrained: bool,
    batch_size: int,
    flatten_acts: bool,
    f_format: str = 'txt'
) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, model_name in enumerate(model_names):
        model = Model(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            model_path=None
            )
        transforms = model.get_transformations()
        for img_path in img_paths:
            PATH = os.path.join(out_path, img_path, model_name,
                                module_names[i], 'features')
            dl = load_dl(
                root=img_path,
                out_path=out_path,
                backend=model.backend,
                batch_size=batch_size,
                transforms=transforms,
                )
            features, _ = model.extract_features(
                data_loader=dl,
                module_name=module_names[i],
                flatten_acts=flatten_acts,
                clip=clip[i],
                )
            save_features(features, PATH, f_format)


def extract_features_across_models_datasets_and_modules(
    out_path: str,
    model_names: List[str],
    img_paths: List[str],
    module_names: List[str],
    clip: List[str],
    pretrained: bool,
    batch_size: int,
    flatten_acts: bool,
    f_format: str = 'txt'
) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, model_name in enumerate(model_names):
        model = Model(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            model_path=None
            )
        transforms = model.get_transformations()
        modules = get_module_names(model, module_names[i])
        for img_path in img_paths:
            for module_name in modules:
                PATH = os.path.join(out_path, img_path,
                                    model_name, module_name, 'features')
                dl = load_dl(
                    root=img_path,
                    out_path=out_path,
                    backend=model.backend,
                    batch_size=batch_size,
                    transforms=transforms,
                    )
                features, _ = model.extract_features(
                    data_loader=dl,
                    module_name=module_name,
                    flatten_acts=flatten_acts,
                    clip=clip[i],
                    )
                save_features(features, PATH, f_format)


def center_features(X: np.ndarray) -> np.ndarray:
    """Center features to have zero mean."""
    try:
        X -= X.mean(axis=0)
        return X
    except:
        raise Exception(
            '\nMake sure features are represented through a two-dimensional array\n')


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize feature vectors by their l2-norm."""
    try:
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        return X
    except:
        raise Exception(
            f'\nMake sure features are represented through a two-dimensional array\n')


def compress_features(X: np.ndarray, rnd_seed: int, retained_var: float = .9) -> np.ndarray:
    """Compress feature matrix with Principal Components Analysis (PCA)."""
    from sklearn.decomposition import PCA
    assert isinstance(
        rnd_seed, int), '\nTo reproduce results, random state for PCA must be defined.\n'
    pca = PCA(n_components=retained_var,
              svd_solver='full', random_state=rnd_seed)
    transformed_feats = pca.fit_transform(X)
    return transformed_feats


# ################################################################# #
# HELPER FUNCTIONS FOR SAVING, MERGING AND SLICING FEATURE MATRICES #
# ################################################################# #


def rm_suffix(img: str) -> str:
    return re.sub(r'(.eps|.jpg|.jpeg|.png|.PNG|.tif|.tiff)$', '', img)


def store_features(
    PATH: str,
    features: np.ndarray,
    file_format: str,
) -> None:
    """Save feature matrix to disk in pre-defined file format."""
    if not os.path.exists(PATH):
        print(f'...Output directory did not exist. Creating directories to save features.')
        os.makedirs(PATH)

    if file_format == 'npy':
        with open(pjoin(PATH, 'features.npy'), 'wb') as f:
            np.save(f, features)
    elif file_format == 'mat':
        try:
            with open(pjoin(PATH, 'file_names.txt'), 'r') as f:
                file_names = [rm_suffix(l.strip()) for l in f]
            features = {file_name: feature for file_name,
                        feature in zip(file_names, features)}
            scipy.io.savemat(pjoin(PATH, 'features.mat'), features)
        except FileNotFoundError:
            scipy.io.savemat(pjoin(PATH, 'features.mat'),
                             {'features': features})
    elif file_format == 'hdf5':
        h5f = h5py.File(pjoin(PATH, 'features.h5'), 'w')
        h5f.create_dataset('features', data=features)
        h5f.close()
    else:
        np.savetxt(pjoin(PATH, 'features.txt'), features)
    print(f'...Features successfully saved to disk.\n')


def split_features(
    PATH: str,
    features: np.ndarray,
    file_format: str,
    n_splits: int,
) -> None:
    """Split feature matrix into <n_splits> subsamples to counteract MemoryErrors."""
    if file_format == 'mat':
        try:
            with open(pjoin(PATH, 'file_names.txt'), 'r') as f:
                file_names = [rm_suffix(l.strip()) for l in f]
        except FileNotFoundError:
            file_names = None
    splits = np.linspace(0, len(features), n_splits, dtype=int)
    if file_format == 'hdf5':
        h5f = h5py.File(pjoin(PATH, 'features.h5'), 'w')

    for i in range(1, len(splits)):
        feature_split = features[splits[i - 1]:splits[i]]
        if file_format == 'npy':
            with open(pjoin(PATH, f'features_{i:02d}.npy'), 'wb') as f:
                np.save(f, feature_split)
        elif file_format == 'mat':
            if file_names:
                file_name_split = file_names[splits[i - 1]:splits[i]]
                new_features = {
                    file_name_split[i]: feature for i, feature in enumerate(feature_split)}
                scipy.io.savemat(
                    pjoin(PATH, f'features_{i:02d}.mat'), new_features)
            else:
                scipy.io.savemat(pjoin(PATH, f'features_{i:02d}.mat'), {
                                 'features': features})
        elif file_format == 'hdf5':
            h5f.create_dataset(f'features_{i:02d}', data=feature_split)
        else:
            np.savetxt(pjoin(PATH, f'features_{i:02d}.txt'), feature_split)

    if file_format == 'hdf5':
        h5f.close()


def merge_features(PATH: str, file_format: str) -> np.ndarray:
    if file_format == 'hdf5':
        with h5py.File(pjoin(PATH, 'features.h5'), 'r') as f:
            features = np.vstack([split[:] for split in f.values()])
    else:
        feature_splits = np.array([split for split in os.listdir(PATH) if split.endswith(
            file_format) and re.search(r'^(?=^features)(?=.*[0-9]+$).*$', split.rstrip('.' + file_format))])
        enumerations = np.array([int(re.sub(r'\D', '', feature))
                                for feature in feature_splits])
        feature_splits = feature_splits[np.argsort(enumerations)]
        if file_format == 'txt':
            features = np.vstack([np.loadtxt(pjoin(PATH, feature))
                                 for feature in feature_splits])
        elif file_format == 'mat':
            features = np.vstack([scipy.io.loadmat(pjoin(PATH, feature))['features']
                                 for feature in feature_splits])
        elif file_format == 'npy':
            features = np.vstack([np.load(pjoin(PATH, feature))
                                 for feature in feature_splits])
        else:
            raise Exception(
                '\nCan only process hdf5, npy, mat, or txt files.\n')
    return features


def parse_imagenet_synsets(PATH: str) -> List[str]:
    """Convert WN synsets into classes."""
    def parse_str(str):
        return re.sub(r'[^a-zA-Z]', '', str).rstrip('n').lower()
    imagenet_synsets = []
    with open(PATH, 'r') as f:
        for i, l in enumerate(f):
            l = l.split('_')
            cls = '_'.join(list(map(parse_str, l)))
            imagenet_synsets.append(cls)
    return imagenet_synsets


def parse_imagenet_classes(PATH: str) -> List[str]:
    """Disambiguate ImageNet classes."""
    imagenet_classes = []
    with open(PATH, 'r') as f:
        for i, l in enumerate(f):
            l = l.strip().split()
            cls = '_'.join(l[1:]).rstrip(',').strip("'").lower()
            cls = cls.split(',')
            cls = cls[0]
            imagenet_classes.append(cls)
    return imagenet_classes


def get_class_intersection(imagenet_classes: list, things_objects: list) -> set:
    """Return intersection of THINGS objects and ImageNet classes."""
    return set(things_objects).intersection(set(imagenet_classes))


def get_cls_mapping_imagenet(PATH: str, save_as_json: bool = False) -> dict:
    """Store ImageNet classes in an *index_to_class* dictionary, and subsequently save as .json file."""
    if re.search(r'synset', PATH.split('/')[-1]):
        imagenet_classes = parse_imagenet_synsets(PATH)
    else:
        imagenet_classes = parse_imagenet_classes(PATH)
    idx2cls = dict(enumerate(imagenet_classes))
    if save_as_json:
        filename = 'imagenet_idx2class.json'
        PATH = '/'.join(PATH.split('/')[:-1])
        with open(pjoin(PATH, filename), 'w') as f:
            json.dump(idx2cls, f)
    return idx2cls


def get_class_probabilities(
    probas: np.ndarray,
    out_path: str,
    cls_file: str,
    top_k: int,
    save_as_json: bool,
) -> Dict[str, Dict[str, float]]:
    """Compute probabilities per ImageNet class."""
    file_names = open(pjoin(out_path, 'file_names.txt'),
                      'r').read().splitlines()
    idx2cls = get_cls_mapping_imagenet(cls_file)
    class_probas = {}
    for i, (file, p_i) in enumerate(zip(file_names, probas)):
        sorted_predictions = np.argsort(-p_i)[:top_k]
        class_probas[file] = {idx2cls[pred]: float(
            p_i[pred]) for pred in sorted_predictions}
    if save_as_json:
        with open(pjoin(out_path, 'class_probabilities.json'), 'w') as f:
            json.dump(class_probas, f)
    return class_probas


def json2dict(PATH: str, filename: str) -> dict:
    with open(pjoin(PATH, filename), 'r') as f:
        idx2cls = dict(json.load(f))
    return idx2cls


def compose_transforms(resize_dim: int = 256, crop_dim: int = 224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])
    composition = T.Compose(
        [T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
    return composition


def load_item_names(folder: str = './data') -> np.ndarray:
    return pd.read_csv(pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values


def save_features(
    features: np.ndarray,
    out_path: str,
    file_format: str,
    n_splits: int = 10,
) -> None:
    """Save feature matrix in desired format to disk."""
    assert file_format in FILE_FORMATS, f'\nFile format must be one of {FILE_FORMATS}.\n'
    if not os.path.exists(out_path):
        print(
            f'\nOutput directory did not exist. Creating directories to save features...\n')
        os.makedirs(out_path)
    # save hidden unit actvations to disk (either as one single file or as several splits)
    if len(features.shape) > 2 and file_format == 'txt':
        print(f'\n...Cannot save 4-way tensor in a txt format.')
        print(f'...Change format to one of {FILE_FORMATS[:-1]}.\n')
    else:
        try:
            store_features(PATH=out_path, features=features,
                           file_format=file_format)
        except MemoryError:
            print(
                f'\n...Could not save features as one single file due to memory problems.')
            print(f'...Now splitting features along row axis into several batches.\n')
            split_features(PATH=out_path, features=features,
                           file_format=file_format, n_splits=n_splits)
            print(
                f'...Saved features in {n_splits:02d} different files, enumerated in ascending order.')
            print(f'If you want features to be splitted into more or fewer files, simply change number of splits parameter.\n')


def save_targets(
    targets: np.ndarray,
    PATH: str,
    file_format: str,
) -> None:
    """Save target vector to disk."""
    if not os.path.exists(PATH):
        print(
            f'\nOutput directory did not exist. Creating directories to save targets...\n')
        os.makedirs(PATH)

    if file_format == 'npy':
        with open(pjoin(PATH, 'targets.npy'), 'wb') as f:
            np.save(f, targets)
    elif file_format == 'mat':
        scipy.io.savemat(pjoin(PATH, 'targets.mat'), {'targets': targets})
    elif file_format == 'hdf5':
        h5f = h5py.File(pjoin(PATH, 'targets.h5'), 'w')
        h5f.create_dataset('targets', data=targets)
        h5f.close()
    else:
        np.savetxt(pjoin(PATH, 'targets.txt'), targets)
    print(f'...Targets successfully saved to disk.\n')

# ########################################### #
# HELPER FUNCTIONS FOR RSA & RDM COMPUTATIONS #
# ########################################### #


@njit(parallel=True, fastmath=True)
def squared_dists(F: np.ndarray) -> np.ndarray:
    """Compute squared l2-distances between feature representations in parallel."""
    N = F.shape[0]
    D = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            D[i, j] = np.linalg.norm(F[i] - F[j]) ** 2
    return D


def gaussian_kernel(F: np.ndarray) -> np.ndarray:
    """Compute dissimilarity matrix based on a Gaussian kernel."""
    D = squared_dists(F)
    return np.exp(-D / np.mean(D))


def correlation_matrix(F: np.ndarray, a_min: float = -1., a_max: float = 1.) -> np.ndarray:
    """Compute dissimilarity matrix based on correlation distance (on the matrix-level)."""
    F_c = F - F.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom).clip(min=a_min, max=a_max)
    return corr_mat


def cosine_matrix(F: np.ndarray, a_min: float = -1., a_max: float = 1.) -> np.ndarray:
    """Compute dissimilarity matrix based on cosine distance (on the matrix-level)."""
    num = F @ F.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    cos_mat = (num / denom).clip(min=a_min, max=a_max)
    return cos_mat


def compute_rdm(F: np.ndarray, method: str) -> np.ndarray:
    """Compute representational dissimilarity matrix based on some distance measure.

    Parameters
    ----------
    F : ndarray
        Input array. Feature matrix of size n x m,
        where n corresponds to the number of observations
        and m is the number of latent dimensions.
    method : str
        Distance metric (e.g., correlation, cosine).

    Returns
    -------
    output : ndarray
        Returns the representational dissimilarity matrix.
    """
    methods = ['correlation', 'cosine', 'euclidean', 'gaussian']
    assert method in methods, f'\nMethod to compute RDM must be one of {methods}.\n'
    if method == 'euclidean':
        rdm = squareform(pdist(F, method))
        return rdm
    else:
        if method == 'correlation':
            rsm = correlation_matrix(F)
        elif method == 'cosine':
            rsm = cosine_matrix(F)
        elif method == 'gaussian':
            rsm = gaussian_kernel(F)
    return 1 - rsm


def correlate_rdms(
    rdm_1: np.ndarray,
    rdm_2: np.ndarray,
    correlation: str = 'pearson',
) -> float:
    """Correlate the upper triangular parts of two distinct RDMs.

    Parameters
    ----------
    rdm_1 : ndarray
        First RDM.
    rdm_2 : ndarray
        Second RDM.
    correlation : str
        Correlation coefficient (e.g., Spearman, Pearson).

    Returns
    -------
    output : float
        Returns the correlation coefficient of the two RDMs.
    """
    triu_inds = np.triu_indices(len(rdm_1), k=1)
    corr_func = getattr(scipy.stats, ''.join((correlation, 'r')))
    rho = corr_func(rdm_1[triu_inds], rdm_2[triu_inds])[0]
    return rho


def plot_rdm(
    out_path: str,
    F: np.ndarray,
    method: str = 'correlation',
    format: str = '.png',
    colormap: str = 'cividis',
    show_plot: bool = False,
) -> None:
    """Compute and plot representational dissimilarity matrix based on some distance measure.

    Parameters
    ----------
    out_path : str
        Output directory. Directory where to store plots.
    F : ndarray
        Input array. Feature matrix of size n x m,
        where n corresponds to the number of observations
        and m is the number of latent dimensions.
    method : str
        Distance metric (e.g., correlation, cosine).
    format : str
        Image format in which to store visualized RDM.
    colormap : str
        Colormap for visualization of RDM.
    show_plot : bool
        Whether to show visualization of RDM after storing it to disk.

    Returns
    -------
    output : ndarray
        Returns the representational dissimilarity matrix.
    """
    rdm = compute_rdm(F, method)
    plt.figure(figsize=(10, 4), dpi=200)
    plt.imshow(rankdata(rdm).reshape(rdm.shape),
               cmap=getattr(plt.cm, colormap))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if not os.path.exists(out_path):
        print(f'\n...Output directory did not exists. Creating directories.\n')
        os.makedirs(out_path)
    plt.savefig(os.path.join(out_path, ''.join(('rdm', format))))
    if show_plot:
        plt.show()
    plt.close()


def get_features(
    root: str,
    out_path: str,
    model_names: List[str],
    module_names: List[str],
    clip: List[bool],
    pretrained: bool,
    batch_size: int,
    flatten_acts: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract features for a list of neural network models and corresponding modules.

    Parameters
    ----------
    root : str
        Root directory. Directory where images are stored.
    out_path : str
        PATH where order of images features should be stored.
        Files are alphabetically sorted and features are
        extracted accordingly.
    model_names : List[str]
        List of neural network models for which features
        should be extracted.
    module_names : List[str]
        List of neural network layers for which features
        should be extracted. Modules must correspond to
        models. This should be thought of as zipped lists.
    clip : List[bool]
        List of Booleans which indicates whether the
        corresponding model in the <model_names> list
        is a CLIP-based model or not (i.e., True if
        CLIP, else False)
    pretrained : bool
        Whether pretrained or randomly initialized models
        should be loaded into memory.
    batch_size : int
        Integer value that determines the number of images
        within a single mini-batch (i.e., subsample
        of the data).
    flatten_acts : bool
        Whether activation tensor (e.g., activations
        from an early layer of the neural network model)
        should be transformed into a feature vector.

    Returns
    -------
    output : Dict[str, Dict[str, np.ndarray]]
        Returns a dictionary of feature matrices
        corresponding to the selected models and layers.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_features = defaultdict(dict)
    for i, model_name in enumerate(model_names):
        model = Model(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            model_path=None
            )
        transforms = model.get_transformations()
        dl = load_dl(
                    root=root,
                    out_path=out_path,
                    backend=model.backend,
                    batch_size=batch_size,
                    transforms=transforms,
                    )
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_names[i],
            flatten_acts=flatten_acts,
            clip=clip[i],
            )
        model_features[model_name][module_names[i]] = features
    return model_features


def compare_models(
    root: str,
    out_path: str,
    model_names: List[str],
    module_names: List[str],
    pretrained: bool,
    batch_size: int,
    flatten_acts: bool,
    clip: List[bool],
    save_features: bool = True,
    dissimilarity: str = 'correlation',
    correlation: str = 'pearson',
) -> pd.DataFrame:
    """Compare object representations of different models against each other.

    Parameters
    ----------
    root : str
        Root directory. Directory from where to load images.
    out_path : str
        Output directory. Directory where to store features
        corresponding to each neural network model.
    model_names : List[str]
        List of neural network models whose object representations
        should be compared against.
    module_names : List[str]
        List of neural network layers for which features
        should be extracted. Modules must correspond to
        models. This should be thought of as zipped lists.
    pretrained : bool
        Whether pretrained or randomly initialized models
        should be loaded into memory.
    batch_size : int
        Integer value that determines the number of images
        within a single mini-batch (i.e., subsample
        of the data).
    flatten_acts : bool
        Whether activation tensor (e.g., activations
        from an early layer of the neural network model)
        should be transformed into a feature vector.
    clip : List[bool]
        List of Booleans which indicates whether the
        corresponding model in the <model_names> list
        is a CLIP-based model or not (i.e., True if
        CLIP, else False)
    save_features : bool
        Whether to save model features or solely compare
        their representations against each other
        without saving the features to disk.
    dissimilarity : str
        Distance metric to be used to compute RDMs
        corresponding to the model features.
    correlation : str
        Correlation coefficient (e.g., Spearman or Pearson)
        to be used when performing RSA.

    Returns
    -------
    output : pd.DataFrame
        Returns a correlation matrix whose rows and columns
        correspond to the names of the models in <model_names>.
        The cell elements are the correlation coefficients
        for each model combination. The dataframe can subsequently
        be converted into a heatmap with matplotlib or seaborn.
    """
    # extract features for each model and corresponding module
    model_features = get_features(
        root=root,
        out_path=out_path,
        model_names=model_names,
        module_names=module_names,
        clip=clip,
        pretrained=pretrained,
        batch_size=batch_size,
        flatten_acts=flatten_acts,
    )
    # save model features to disc
    if save_features:
        pickle_file_(model_features, out_path, 'features')
    # compare features of each model combination for N bootstraps
    corrs = pd.DataFrame(np.eye(len(model_names)), index=np.arange(
        len(model_names)), columns=model_names, dtype=float)
    model_combs = list(itertools.combinations(model_names, 2))
    for (model_i, model_j) in model_combs:
        module_i = module_names[model_names.index(model_i)]
        module_j = module_names[model_names.index(model_j)]
        features_i = model_features[model_i][module_i]
        features_j = model_features[model_j][module_j]
        rdm_i = compute_rdm(features_i, dissimilarity)
        rdm_j = compute_rdm(features_j, dissimilarity)
        corr = correlate_rdms(rdm_i, rdm_j, correlation)
        corrs.loc[model_names.index(model_i), model_j] = corr
        corrs.loc[model_names.index(model_j), model_i] = corr
    corrs['model_names'] = corrs.columns.to_list()
    corrs.set_index('model_names', inplace=True, drop=True)
    return corrs


def pickle_file_(file: dict, out_path: str, f_name: str) -> None:
    """Pickle any file."""
    with open(os.path.join(out_path, f_name + '.p'), 'wb') as f:
        pickle.dump(file, f)
