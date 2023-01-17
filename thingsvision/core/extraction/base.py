import abc
import warnings
from typing import Any

import numpy as np

Array = np.ndarray


class BaseExtractor(metaclass=abc.ABCMeta):
    def __init__(self, device) -> None:
        self.device = device

    def show(self) -> None:
        warnings.warn(
            message="\nThe .show() method is deprecated and will be removed in future versions. Use .show_model() instead.\n",
            category=UserWarning,
        )
        self.show_model()

    @abc.abstractmethod
    def show_model(self) -> None:
        """Show model."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_transformations(self, **kwargs) -> Any:
        """Get default image transformations."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_default_transformation(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_module_names(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_features(self, **kwargs) -> Array:
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
        output_dir : str, optional
            Path/to//output/directory. If defined, the extracted
            features will be iteratively (every step_size batches)
            stored to disk as numpy files, freeing up memory space.
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
        output : np.ndarray
            Returns the feature matrix (e.g., X \in \mathbb{R}^{n \times p} if head or flatten_acts = True).
        """
        raise NotImplementedError
