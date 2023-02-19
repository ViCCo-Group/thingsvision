import os
import re

import h5py
import numpy as np
import scipy
import scipy.io
import torch
import warnings

from typing import Union

Array = np.ndarray
Tensor = torch.Tensor 

FILE_FORMATS = ["hdf5", "npy", "mat", "pt", "txt"]
EXTENSIONS = r"(.eps|.jpg|.jpeg|.png|.PNG|.tif|.tiff)$"


def rm_suffix(img_name: str) -> str:
    return re.sub(EXTENSIONS, "", img_name)


def store_features(
    root: str,
    features: Union[Array,Tensor],
    file_format: str,
) -> None:
    """Save feature matrix to disk in pre-defined file format."""
    if not os.path.exists(root):
        print(
            "...Output directory did not exist. Creating directories to save features."
        )
        os.makedirs(root)

    if file_format == "npy":
        with open(os.path.join(root, "features.npy"), "wb") as f:
            np.save(f, features)
    elif file_format == "pt":
        torch.save(features, os.path.join(root, "features.pt"))
    elif file_format == "mat":
        try:
            with open(os.path.join(root, "file_names.txt"), "r") as f:
                file_names = [rm_suffix(l.strip()) for l in f]
            features = {
                file_name: feature for file_name, feature in zip(file_names, features)
            }
            scipy.io.savemat(os.path.join(root, "features.mat"), features)
        except FileNotFoundError:
            scipy.io.savemat(os.path.join(root, "features.mat"), {"features": features})
    elif file_format == "hdf5":
        h5f = h5py.File(os.path.join(root, "features.hdf5"), "w")
        h5f.create_dataset("features", data=features)
        h5f.close()
    else: # txt
        np.savetxt(os.path.join(root, "features.txt"), features)
    print("...Features successfully saved to disk.\n")


def split_features(
    root: str,
    features: Array,
    file_format: str,
    n_splits: int,
) -> None:
    """Split feature matrix into <n_splits> subsamples to counteract MemoryErrors."""
    if file_format == "mat":
        try:
            with open(os.path.join(root, "file_names.txt"), "r") as f:
                file_names = [rm_suffix(l.strip()) for l in f]
        except FileNotFoundError:
            file_names = None
    splits = np.linspace(0, len(features), n_splits, dtype=int)
    if file_format == "hdf5":
        h5f = h5py.File(os.path.join(root, "features.hdf5"), "w")

    for i in range(1, len(splits)):
        feature_split = features[splits[i - 1] : splits[i]]
        if file_format == "npy":
            with open(os.path.join(root, f"features_{i:02d}.npy"), "wb") as f:
                np.save(f, feature_split)
        elif file_format == "pt":
            torch.save(feature_split, os.path.join(root, f"features_{i:02d}.pt"))
        elif file_format == "mat":
            if file_names:
                file_name_split = file_names[splits[i - 1] : splits[i]]
                new_features = {
                    file_name_split[i]: feature
                    for i, feature in enumerate(feature_split)
                }
                scipy.io.savemat(
                    os.path.join(root, f"features_{i:02d}.mat"), new_features
                )
            else:
                scipy.io.savemat(
                    os.path.join(root, f"features_{i:02d}.mat"), {"features": features}
                )
        elif file_format == "hdf5":
            h5f.create_dataset(f"features_{i:02d}", data=feature_split)
        else:
            np.savetxt(os.path.join(root, f"features_{i:02d}.txt"), feature_split)

    if file_format == "hdf5":
        h5f.close()


def save_features(
    features: Union[Array, Tensor],
    out_path: str,
    file_format: str,
    n_splits: int = 10,
) -> None:
    """Save feature matrix in desired format to disk."""
    assert (
        file_format in FILE_FORMATS
    ), f"\nFile format must be one of {FILE_FORMATS}.\nChange output format accordingly.\n"
    if not os.path.exists(out_path):
        print(
            "\nOutput directory did not exist. Creating directories to save features...\n"
        )
        os.makedirs(out_path, exist_ok=True)
    if file_format == "pt":
        if not isinstance(features, torch.Tensor):
            warnings.warn(
                message=f"\nExpected features to be of type <torch.Tensor> but got {type(features)} instead.\nConverting features to type <torch.Tensor> now.\n",
                category=UserWarning,
                )
            features = torch.from_numpy(features)
    # save hidden unit actvations to disk (either as one single file or as several splits)
    if len(features.shape) > 2 and file_format == "txt":
        print("\n...Cannot save 4-way tensor in a txt format.")
        print(f"...Change format to one of {FILE_FORMATS[:-1]}.\n")
    else:
        try:
            store_features(root=out_path, features=features, file_format=file_format)
        except MemoryError:
            print(
                "\n...Could not save features as one single file due to memory problems."
            )
            print("...Now splitting features along row axis into several batches.\n")
            split_features(
                root=out_path,
                features=features,
                file_format=file_format,
                n_splits=n_splits,
            )
            print(
                f"...Saved features in {n_splits:02d} different files, enumerated in ascending order."
            )
            print(
                "If you want features to be splitted into more or fewer files, simply change number of splits parameter.\n"
            )


def merge_features(root: str, file_format: str) -> Array:
    if file_format == "hdf5":
        with h5py.file(os.path.join(root, "features.hdf5"), "r") as f:
            features = np.vstack([split[:] for split in f.values()])
    else:
        feature_splits = np.array(
            [
                split
                for split in os.listdir(root)
                if split.endswith(file_format)
                and re.search(
                    r"^(?=^features)(?=.*[0-9]+$).*$", split.rstrip("." + file_format)
                )
            ]
        )
        enumerations = np.array(
            [int(re.sub(r"\d", "", feature)) for feature in feature_splits]
        )
        feature_splits = feature_splits[np.argsort(enumerations)]

        def stack_features(feature_splits: Array) -> Array:
            if file_format == "txt":
                features = np.vstack(
                    [
                        np.loadtxt(os.path.join(root, feature))
                        for feature in feature_splits
                    ]
                )
            elif file_format == "mat":
                features = np.vstack(
                    [
                        scipy.io.loadmat(os.path.join(root, feature))["features"]
                        for feature in feature_splits
                    ]
                )
            elif file_format == "npy":
                features = np.vstack(
                    [np.load(os.path.join(root, feature)) for feature in feature_splits]
                )
            else:
                raise ValueError(
                    "\nCan only process .hdf5, .npy, .mat, or .txt files.\n"
                )
            return features

        features = stack_features(feature_splits)
    return features
