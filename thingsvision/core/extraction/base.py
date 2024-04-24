import abc
import os
import re
import warnings
from typing import Callable, Iterator, List, Optional, Union

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
        module_name: str,
        flatten_acts: bool,
        output_type: str,
    ) -> Union[
        Union[
            TensorType["b", "num_maps", "h_prime", "w_prime"],
            TensorType["b", "t", "d"],
            TensorType["b", "p"],
            TensorType["b", "d"],
        ],
        Array,
    ]:
        """Extract the activations of a selected module for every image in a mini-batch.

        Parameters
        ----------
        batch : np.ndarray or torch.Tensor
            mini-batch of three-dimensional image tensors.
        module_name : str
            Name of the module for which features should be extraced.
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
        module_name: str,
        flatten_acts: bool,
    ) -> Union[
        Union[
            TensorType["b", "num_maps", "h_prime", "w_prime"],
            TensorType["b", "t", "d"],
            TensorType["b", "p"],
            TensorType["b", "d"],
        ],
        Array,
    ]:
        raise NotImplementedError

    def get_output_types(self) -> List[str]:
        """Return the list of available output types (for the feature matrix)."""
        return ["ndarray", "tensor"]

    def _module_and_output_check(self, module_name: str, output_type: str) -> None:
        """Checks whether the provided module name and output type are valid."""
        valid_names = self.get_module_names()
        if not module_name in valid_names:
            raise ValueError(
                f"\n{module_name} is not a valid module name. Please choose a name from the following set of modules: {valid_names}\n"
            )
        assert (
            output_type in self.get_output_types()
        ), f"\nData type of output feature matrix must be set to one of the following available data types: {self.get_output_types()}\n"

    def extract_features(
        self,
        batches: Iterator[Union[TensorType["b", "c", "h", "w"], Array]],
        module_name: str,
        flatten_acts: bool = False,
        output_type: Optional[str] = "ndarray",
        output_dir: Optional[str] = None,
        step_size: Optional[int] = None,
    ) -> Union[
        Union[
            TensorType["n", "num_maps", "h_prime", "w_prime"],
            TensorType["n", "t", "d"],
            TensorType["n", "p"],
            TensorType["n", "d"],
        ],
        Array,
    ]:
        """Extract hidden unit activations (at specified layer) for every image in the database.

        Parameters
        ----------
        batches : Iterator
            Mini-batches. Iterator with equally sized
            mini-batches, where each element is a
            subsample of the full (image) dataset.
        module_name : str
            Layer name. Name of neural network layer for
            which features should be extraced.
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

        Returns
        -------
        output : np.ndarray or torch.Tensor
            Returns the feature matrix (e.g., $X \in \mathbb{R}^{n \times d}$ if penultimate or logits layer or flatten_acts = True).
        """
        self._module_and_output_check(module_name, output_type)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            if not step_size:
                # if step size is not given, assume that features to every image consume 3MB of memory and that the user has at least 8GB of free RAM
                step_size = 8000 // (len(next(iter(batches))) * 3) + 1

        features = []
        image_ct, last_image_ct = 0, 0
        for i, batch in tqdm(
            enumerate(batches, start=1), desc="Batch", total=len(batches)
        ):
            features.append(
                self._extract_batch(
                    batch=batch, module_name=module_name, flatten_acts=flatten_acts
                )
            )

            image_ct += len(batch)
            del batch

            if output_dir and (i % step_size == 0 or i == len(batches)):
                if self.get_backend() == "pt":
                    features_subset = torch.cat(features)
                    if output_type == "ndarray":
                        features_subset = self._to_numpy(features_subset)
                        features_subset_file = os.path.join(
                            output_dir,
                            f"features_{last_image_ct}-{image_ct}.npy",
                        )
                        np.save(features_subset_file, features_subset)
                    else:  # output_type = tensor
                        features_subset_file = os.path.join(
                            output_dir,
                            f"features_{last_image_ct}-{image_ct}.pt",
                        )
                        torch.save(features_subset, features_subset_file)
                else:
                    features_subset_file = os.path.join(
                        output_dir, f"features_{last_image_ct}-{image_ct}.npy"
                    )
                    features_subset = np.vstack(features)
                    np.save(features_subset_file, features_subset)
                features = []
                last_image_ct = image_ct
        print(
            f"...Features successfully extracted for all {image_ct} images in the database."
        )
        if output_dir:
            print(f"...Features were saved to {output_dir}.")
            return None
        else:
            if self.get_backend() == "pt":
                features = torch.cat(features)
                if output_type == "ndarray":
                    features = self._to_numpy(features)
            else:
                features = np.vstack(features)
                print(f"...Features shape: {features.shape}")
        return features

    @staticmethod
    def _to_numpy(
        features: Union[
            TensorType["n", "num_maps", "h_prime", "w_prime"],
            TensorType["n", "t", "d"],
            TensorType["n", "p"],
            TensorType["n", "d"],
        ]
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
