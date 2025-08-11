import abc
import os
import re
import warnings
from typing import Callable, Dict, Iterator, List, Optional, Union
from collections import defaultdict

import numpy as np
from torchtyping import TensorType
from tqdm.auto import tqdm

import torch

Array = np.ndarray


class BaseExtractor(metaclass=abc.ABCMeta):
    def __init__(self, device, preprocess) -> None:
        self.device = device
        self._check_device()
        self.preprocess = preprocess

    def show(self) -> None:
        warnings.warn(
            message="\nThe .show() method is deprecated and will be removed in future versions. Use .show_model() instead.\n",
            category=UserWarning,
        )
        self.show_model()

    def _check_device(self) -> None:
        """Check whether the selected device is available on the current compute node."""
        if self.device.startswith("cuda"):
            gpu_index = re.search(r"cuda:(\d+)", self.device)

            if not torch.cuda.is_available():
                warnings.warn(
                    "\nCUDA is not available on your system. Switching to device='cpu'.\n",
                    category=UserWarning,
                )
                self.device = "cpu"
            elif gpu_index and int(gpu_index.group(1)) >= torch.cuda.device_count():
                warnings.warn(
                    f"\nGPU index {gpu_index.group(1)} is out of range. "
                    f"Available GPUs: {torch.cuda.device_count()}. "
                    f"Switching to device='cuda:0'.\n",
                    category=UserWarning,
                )
                self.device = "cuda:0"

        print(f"\nUsing device: {self.device}\n")

    @abc.abstractmethod
    def show_model(self) -> None:
        """Show architecture."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_default_transformation(
        self,
        mean: List[float],
        std: List[float],
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Callable:
        raise NotImplementedError

    @abc.abstractmethod
    def get_module_names(self) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_batch(
        self,
        batch: Union[TensorType["b", "c", "h", "w"], Array],
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        flatten_acts: bool = False,
        output_type: str = "ndarray",
    ) -> Union[
        # This is the return type when 'module_names' is used
        Dict[
            str,
            Union[
                Union[
                    TensorType["n", "num_maps", "h_prime", "w_prime"],
                    TensorType["n", "t", "d"],
                    TensorType["n", "p"],
                    TensorType["n", "d"],
                ],
                Array,
            ],
        ],
        # This is the return type when 'module_name' is used (for backward compatibility)
        Union[
            TensorType["n", "num_maps", "h_prime", "w_prime"],
            TensorType["n", "t", "d"],
            TensorType["n", "p"],
            TensorType["n", "d"],
            Array,
        ],
    ]:
        """Extract the activations of a selected module for every image in a mini-batch.

        Parameters
        ----------
        batch : np.ndarray or torch.Tensor
            mini-batch of three-dimensional image tensors.
        module_name : str
            Name of the neural network layer for which features should be extracted.
        module_names : List[str]
            Names of the modules for which features should be extracted.
        flatten_acts : bool
            Whether the activation of a tensor should be flattened to a vector.
        output_type : str {"ndarray", "tensor"}
            Whether to return output features as torch.Tensor or np.ndarray.
            Available options are "ndarray" or "tensor".
        Returns
        -------
        output : np.ndarray or torch.Tensor
            Returns the feature matrix (e.g., $X \in \mathbb{R}^{B \times d}$ if penultimate or logits layer or flatten_acts = True).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _extract_batch(
        self,
        batch: Union[TensorType["b", "c", "h", "w"], Array],
        module_names: List[str],
        flatten_acts: bool = False,
    ) -> Dict[
        str,
        Union[
            Union[
                TensorType["n", "num_maps", "h_prime", "w_prime"],
                TensorType["n", "t", "d"],
                TensorType["n", "p"],
                TensorType["n", "d"],
            ],
            Array,
        ],
    ]:
        raise NotImplementedError

    def get_output_types(self) -> List[str]:
        """Return the list of available output types (for the feature matrix)."""
        return ["ndarray", "tensor"]

    def _module_and_output_check(
        self, module_names: List[str], output_type: str
    ) -> None:
        """Checks whether the provided module name and output type are valid."""
        valid_names = self.get_module_names()
        for module_name in module_names:
            if module_name not in valid_names:
                raise ValueError(
                    f"\n{module_name} is not a valid module name. Please choose a name from the following set of modules: {valid_names}\n"
                )
        assert (
            output_type in self.get_output_types()
        ), f"\nData type of output feature matrix must be set to one of the following available data types: {self.get_output_types()}\n"

    def _save_features(self, features, features_file, extension):
        if extension == "npy":
            np.save(features_file, features)
        elif extension == "pt":
            torch.save(features, features_file)
        else:
            raise ValueError(f"Invalid extension: {extension}")

    def extract_features(
        self,
        batches: Iterator[Union[TensorType["b", "c", "h", "w"], Array]],
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        flatten_acts: bool = False,
        output_type: Optional[str] = "ndarray",
        output_dir: Optional[str] = None,
        step_size: Optional[int] = None,
        file_name_suffix: str = "",
        save_in_one_file: bool = False,
    ) -> Union[
        # This is the return type when 'module_names' is used
        Dict[
            str,
            Union[
                Union[
                    TensorType["n", "num_maps", "h_prime", "w_prime"],
                    TensorType["n", "t", "d"],
                    TensorType["n", "p"],
                    TensorType["n", "d"],
                ],
                Array,
            ],
        ],
        # This is the return type when 'module_name' is used (for backward compatibility)
        Union[
            TensorType["n", "num_maps", "h_prime", "w_prime"],
            TensorType["n", "t", "d"],
            TensorType["n", "p"],
            TensorType["n", "d"],
            Array,
        ],
    ]:
        """Extract hidden unit activations (at specified layer) for every image in the database.

        Parameters
        ----------
        batches : Iterator
            Mini-batches. Iterator with equally sized
            mini-batches, where each element is a
            subsample of the full (image) dataset.
        module_name : str
            Layer name. Name of the neural network layer for
            which features should be extracted.
        module_names : List[str]
            Layer names. Names of neural network layers for
            which features should be extracted.
        flatten_acts : bool
            Whether activation tensor (e.g., activations
            from an early layer of the neural network model)
            should be transformed into a vector.
        output_type : str {"ndarray", "tensor"}
            Whether to return output features as torch.Tensor or np.ndarray.
            Use torch.Tensor if you don't save the features to disk after
            calling the exraction method (moving tensors to CPU is costly).
            Available options are "ndarray" or "tensor".
        output_dir : str, optional
            Path/to//output/directory. If defined, the extracted
            features will be iteratively (every step_size batches)
            stored to disk as NumPy files, freeing up memory space.
            Use this option if your dataset is too large or when extracting many features
            at once. The default is None, so that the features are kept
            in memory.
        step_size : int, optional
            Number of batches after which the extracted features
            are saved to disk. The default uses a heuristic so that
            extracted features should fit into 8GB of free memory.
            Only used if output_dir is defined.
        file_name_suffix: str
            Suffix to append to the output file names (e.g., "_train", "_val").
        save_in_one_file : bool
            If True, all features are saved in one file. If output_dir is defined,
            the features are saved in separate files for each module name. They are first
            saved in chunks of step_size batches, and then all features are concatenated
            and saved in one file.
        Returns
        -------
        output : np.ndarray or torch.Tensor
            Returns the feature matrix (e.g., $X \in \mathbb{R}^{n \times d}$ if penultimate or logits layer or flatten_acts = True).
        """
        if not bool(module_name) ^ bool(module_names):
            raise ValueError(
                "\nPlease provide either a single module name or a list of module names, but not both.\n"
            )
        if module_name is not None:
            single_module_call = True
            module_names = [module_name]
        else:
            single_module_call = False
        self._module_and_output_check(module_names, output_type)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            if not step_size:
                # if step size is not given, assume that features to every image consume 3MB of memory and that the user has at least 8GB of free RAM
                step_size = 8000 // (len(next(iter(batches))) * 3) + 1

        # create feature dict per module name
        features = defaultdict(list)
        feature_file_names = defaultdict(list)
        image_ct, last_image_ct = 0, 0
        for i, batch in tqdm(
            enumerate(batches, start=1), desc="Batch", total=len(batches)
        ):
            modules_features = self._extract_batch(
                batch=batch, module_names=module_names, flatten_acts=flatten_acts
            )

            image_ct += len(batch)
            del batch

            for module_name in module_names:
                features[module_name].append(modules_features[module_name])

                if output_dir and (i % step_size == 0 or i == len(batches)):
                    curr_output_dir = os.path.join(output_dir, module_name)
                    if not os.path.exists(curr_output_dir):
                        print(f"Creating output directory: {curr_output_dir}")
                        os.makedirs(curr_output_dir)

                    if self.get_backend() == "pt":
                        features_subset = torch.cat(features[module_name])
                        if output_type == "ndarray":
                            features_subset = self._to_numpy(features_subset)
                            file_extension = "npy"
                        else:
                            file_extension = "pt"
                    else:
                        features_subset = np.vstack(features[module_name])
                        file_extension = "npy"

                    features_subset_file = os.path.join(
                        curr_output_dir,
                        f"features{file_name_suffix}_{last_image_ct}-{image_ct}.{file_extension}",
                    )
                    self._save_features(
                        features_subset, features_subset_file, file_extension
                    )

                    # Note: we add full file paths to feature_file_names to be able to load the features later
                    feature_file_names[module_name].append(features_subset_file)
                    features[module_name] = []
                    last_image_ct = image_ct

        print(
            f"...Features successfully extracted for all {image_ct} images in the database."
        )
        if output_dir:
            if save_in_one_file:
                # load features per module name and concatenate them
                for module_name in module_names:
                    # load from files
                    features = []
                    for file in feature_file_names[module_name]:
                        if self.get_backend() == "pt" and output_type != "ndarray":
                            features.append(torch.load(file))
                        elif file.endswith(".npy"):
                            features.append(np.load(file))
                        else:
                            raise ValueError(
                                f"Invalid or unsupported file extension: {file}"
                            )

                    features_file = os.path.join(
                        output_dir, f"{module_name}/features{file_name_suffix}"
                    )
                    if output_type == "ndarray":
                        self._save_features(
                            np.concatenate(features), features_file + ".npy", "npy"
                        )
                    else:
                        self._save_features(
                            torch.cat(features), features_file + ".pt", "pt"
                        )
                    print(
                        f"...Features for module '{module_name}' were saved to {features_file}."
                    )
                    for file in feature_file_names[module_name]:
                        os.remove(file)

            print(f"...Features were saved to {output_dir}.")
            return None
        else:
            for module_name in module_names:
                if self.get_backend() == "pt":
                    features[module_name] = torch.cat(features[module_name])
                    if output_type == "ndarray":
                        features[module_name] = self._to_numpy(features[module_name])
                else:
                    features[module_name] = np.vstack(features[module_name])
                print(
                    f"...Features for module '{module_name}' have shape: {features[module_name].shape}"
                )

        if single_module_call:
            return features[module_name]
        return features

    @staticmethod
    def _to_numpy(
        features: Union[
            TensorType["n", "num_maps", "h_prime", "w_prime"],
            TensorType["n", "t", "d"],
            TensorType["n", "p"],
            TensorType["n", "d"],
        ],
    ) -> Array:
        """Move activations to CPU and convert torch.Tensor to np.ndarray."""
        return features.numpy()

    def get_transformations(
        self, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True
    ) -> Callable:
        """Load image transformations for a specific model. Image transformations depend on the backend."""
        if self.preprocess:
            return self.preprocess
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            composition = self.get_default_transformation(
                mean, std, resize_dim, crop_dim, apply_center_crop
            )
        return composition

    @abc.abstractmethod
    def get_backend(self) -> str:
        raise NotImplementedError
