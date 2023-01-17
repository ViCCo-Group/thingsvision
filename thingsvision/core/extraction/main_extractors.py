import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Union

import data
import numpy as np

from .base import BaseExtractor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress tensorflow warnings
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import layers
from torchtyping import TensorType
from torchvision import transforms as T
from tqdm import tqdm

Array = np.ndarray


class PyTorchExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Any = field(default_factory=lambda: {}),
        model: Any = None,
        preprocess: Any = None,
    ) -> None:
        super().__init__(device)
        self.model_names = model_name
        self.pretrained = pretrained
        self.model_path = model_path
        self.model_parameters = (model_parameters,)
        self.model = model
        self.preprocess = preprocess
        self.backend = field(init=False, default="pt")

        if not self.model:
            self.load_model()

    def get_activation(self, name: str) -> Callable:
        """Store copy of hidden unit activations at each layer of model."""

        def hook(model, input, output) -> None:
            # store copy of tensor rather than tensor itself
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            try:
                activations[name] = act.clone().detach()
            except AttributeError:
                activations[name] = act.clone()

        return hook

    def register_hook(self) -> None:
        """Register a forward hook to store activations."""
        for n, m in self.model.named_modules():
            if n == self.module_name:
                m.register_forward_hook(self.get_activation(n))

    def extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool,
        output_dir: str = None,
        step_size: int = None,
    ):
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
        self.model = self.model.to(self.device)
        # initialise an empty dict to store features for each mini-batch
        global activations
        activations = {}
        # register forward hook to store features
        self.register_hook()
        valid_names = self.get_module_names()
        if not module_name in valid_names:
            raise ValueError(
                f"\n{module_name} is not a valid module name. Please choose a name from the following set of modules: {valid_names}\n"
            )

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
                self._extract_features(
                    batch=batch, module_name=module_name, flatten_acts=flatten_acts
                )
            )

            image_ct += len(batch)

            if output_dir and (i % step_size == 0 or i == len(batches)):
                features_subset_file = os.path.join(
                    output_dir,
                    f"features_{last_image_ct}-{image_ct}.npy",
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
            features = np.vstack(features)
            print(f"...Features shape: {features.shape}")

        return features

    @torch.no_grad()
    def _extract_features(
        self,
        batch: TensorType["b", "c", "h", "w"],
        module_name: str,
        flatten_acts: bool,
    ) -> Union[
        TensorType["b", "num_maps", "h_prime", "w_prime"],
        TensorType["b", "t", "d"],
        TensorType["b", "p"],
        TensorType["b", "d"],
    ]:
        # move current batch to torch device
        batch = batch.to(self.device)
        _ = self.forward(batch)
        act = activations[module_name]
        if flatten_acts:
            if self.model_name.lower().startswith("clip"):
                act = self.flatten_acts(act, batch, module_name)
            else:
                act = self.flatten_acts(act)
        act = self._to_numpy(act)
        return act

    def forward(
        self, batch: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "num_cls"]:
        """Default forward pass."""
        return self.model(batch)

    @staticmethod
    def flatten_acts(
        act: Union[
            TensorType["b", "num_maps", "h_prime", "w_prime"], TensorType["b", "t", "d"]
        ]
    ) -> TensorType["b", "p"]:
        """Default flattening of activations."""
        return act.view(act.size(0), -1)

    @staticmethod
    def _to_numpy(
        act: Union[
            TensorType["b", "num_maps", "h_prime", "w_prime"],
            TensorType["b", "t", "d"],
            TensorType["b", "p"],
            TensorType["b", "d"],
        ]
    ) -> Array:
        """Move activation to CPU and convert torch.Tensor to np.ndarray."""
        return act.cpu().numpy()

    def show_model(self) -> str:
        return self.model

    def load_model(self) -> None:
        self.load_model_from_source()
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
            except FileNotFoundError:
                state_dict = torch.hub.load_state_dict_from_url(
                    self.model_path, map_location=self.device
                )
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_module_names(self) -> List[str]:
        module_names, _ = zip(*self.model.named_modules())
        module_names = list(filter(lambda n: len(n) > 0, module_names))
        return module_names

    def get_default_transformation(
        self,
        mean: List[float],
        std: List[float],
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Callable:
        normalize = T.Normalize(mean=mean, std=std)
        composes = [T.Resize(resize_dim)]
        if apply_center_crop:
            composes.append(T.CenterCrop(crop_dim))
        composes += [T.ToTensor(), normalize]
        composition = T.Compose(composes)

        return composition

    def get_backend(self) -> str:
        return "pt"


class TensorFlowExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Any = field(default_factory=lambda: {}),
        model: Any = None,
        preprocess: Any = None,
    ) -> None:
        super().__init__(device)
        self.model_names = model_name
        self.pretrained = pretrained
        self.model_path = model_path
        self.model_parameters = (model_parameters,)
        self.model = model
        self.preprocess = preprocess
        self.backend = field(init=False, default="tf")

        if not self.model:
            self.load_model()

    def extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool,
        output_dir: str = None,
        step_size: int = None,
    ):
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
        valid_names = self.get_module_names()
        if not module_name in valid_names:
            raise ValueError(
                f"\n{module_name} is not a valid module name. Please choose a name from the following set of modules: {valid_names}\n"
            )

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
                self._extract_features(
                    batch=batch, module_name=module_name, flatten_acts=flatten_acts
                )
            )

            image_ct += len(batch)

            if output_dir and (i % step_size == 0 or i == len(batches)):
                features_subset_file = os.path.join(
                    output_dir,
                    f"features_{last_image_ct}-{image_ct}.npy",
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
            features = np.vstack(features)
            print(f"...Features shape: {features.shape}")

        return features

    def _extract_features(
        self, batch: Array, module_name: str, flatten_acts: bool
    ) -> Array:
        layer_out = [self.model.get_layer(module_name).output]
        activation_model = keras.models.Model(
            inputs=self.model.input,
            outputs=layer_out,
        )
        activations = activation_model.predict(batch)
        if flatten_acts:
            activations = activations.reshape(activations.shape[0], -1)

        return activations

    def show_model(self) -> str:
        return self.model.summary()

    def load_model(self) -> None:
        self.load_model_from_source()
        if self.model_path:
            self.model.load_weights(self.model_path)
        self.model.trainable = False

    def get_module_names(self) -> List[str]:
        module_names = [l._name for l in self.model.submodules]
        return module_names

    def get_default_transformation(
        self,
        mean: List[float],
        std: List[float],
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Any:
        resize_dim = crop_dim
        composes = [layers.experimental.preprocessing.Resizing(resize_dim, resize_dim)]
        if apply_center_crop:
            pass
            # TODO: fix center crop problem with Keras
            # composes.append(layers.experimental.preprocessing.CenterCrop(crop_dim, crop_dim))

        composes += [
            layers.experimental.preprocessing.Normalization(
                mean=mean, variance=[std_ * std_ for std_ in std]
            )
        ]
        composition = tf.keras.Sequential(composes)

        return composition

    def get_backend(self) -> str:
        return "tf"
