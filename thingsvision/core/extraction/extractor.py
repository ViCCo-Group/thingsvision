import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as tensorflow_models
import timm
import torch
import torchvision
import torchvision.models as torchvision_models
from tensorflow import keras
from tensorflow.keras import layers
from torchvision import transforms as T
from tqdm import tqdm

import thingsvision.custom_models as custom_models
import thingsvision.custom_models.cornet as cornet

Tensor = torch.Tensor
Array = np.ndarray


@dataclass
class Extractor:
    model_name: str
    pretrained: bool
    device: str
    source: str
    model_path: str = None
    model_parameters: Any = field(
        default_factory=lambda: {}
    )
    """
    Parameters
    ----------
    model_name : str
        Model name. Name of model for which features should be extracted.
    pretrained : bool
        Whether to load a model with pretrained or
        randomly initialized weights.
    device : str
        Device. Whether model weights should be moved
        to CUDA or stay on the CPU.
    source: str
        Source of the model and its weights.
    model_path : str (optional)
        path/to/weights. If pretrained is set to False,
        model weights can be loaded from a path on the
        user's machine. This is useful when operating
        on a server without network access, or when
        features should be extracted for a model that
        was fine-tuned (or trained) on a custom image
        dataset.
    model_parameters: dict (optional)
        Further model parameters.
    """

    def __post_init__(self) -> None:
        self.preprocess = None
        self.load_model()

    @staticmethod
    def get_weights(model_name: str, suffix: str = "_weights") -> Any:
        weights_name = None
        for m in dir(torchvision.models):
            if m.lower() == model_name + suffix:
                weights_name = m
                break
        if not weights_name:
            raise ValueError(
                f"\nCould not find pretrained weights for {model_name} in <torchvision>. Choose a different model or change the source.\n"
            )
        weights = getattr(torchvision.models, f"{weights_name}").DEFAULT
        return weights

    def get_model_from_torchvision(self) -> Tuple[Any, str]:
        """Load a (pretrained) neural network model from <torchvision>."""
        self.backend = "pt"
        if hasattr(torchvision_models, self.model_name):
            model = getattr(torchvision_models, self.model_name)
            if self.pretrained:
                self.weights = self.get_weights(self.model_name)
            else:
                self.weights = None
            self.model = model(weights=self.weights)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in torchvision library.\nChoose a different model.\n"
            )

    def get_model_from_timm(self) -> None:
        """Load a (pretrained) neural network model from <timm>."""
        self.backend = "pt"
        if self.model_name in timm.list_models():
            self.model = timm.create_model(self.model_name, self.pretrained)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in timm library.\nChoose a different model.\n"
            )

    def get_model_from_custom(self) -> None:
        """Load a pretrained custom neural network model (e.g., clip, cornet)."""
        preprocess = None
        if self.model_name.startswith("cornet"):
            backend = "pt"
            try:
                model = getattr(cornet, f"cornet_{self.model_name[-1]}")
            except AttributeError:
                model = getattr(cornet, f"cornet_{self.model_name[-2:]}")
            model = model(
                pretrained=self.pretrained, map_location=torch.device(self.device)
            )
            model = model.module  # remove DataParallel
        elif hasattr(custom_models, self.model_name):
            custom_model = getattr(custom_models, self.model_name)
            custom_model = custom_model(self.device, self.model_parameters)
            model, preprocess = custom_model.create_model()
            backend = custom_model.get_backend()
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} among custom models.\nChoose a different model.\n"
            )
        self.model = model
        self.backend = backend
        self.preprocess = preprocess

    def get_model_from_keras(self) -> None:
        """Load a (pretrained) neural network model from TensorFlow/Keras."""
        self.backend = "tf"
        if hasattr(tensorflow_models, self.model_name):
            model = getattr(tensorflow_models, self.model_name)
            if self.pretrained:
                weights = "imagenet"
            elif self.model_path:
                weights = self.model_path
            else:
                weights = None
            self.model = model(weights=weights)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} among TensorFlow models.\n"
            )

    def load_model_from_source(self) -> None:
        """Load a (pretrained) neural network model from <source>."""
        try:
            getattr(self, f"get_model_from_{self.source}")()
        except AttributeError:
            raise ValueError(
                f"\nCannot load models from {self.source}.\nUse a different source for loading a pretrained model.\n"
            )
        if isinstance(self.model, type(None)):
            raise ValueError(
                f"\nCould not find {self.model_name} in {self.source}.\nCheck whether model name is correctly spelled or use a different model.\n"
            )

    def load_model(self) -> None:
        """Load a pretrained model from <source> into memory and move model to current device."""
        self.load_model_from_source()
        if self.backend == "pt":
            device = torch.device(self.device)
            if self.model_path:
                try:
                    state_dict = torch.load(self.model_path, map_location=device)
                except FileNotFoundError:
                    state_dict = torch.hub.load_state_dict_from_url(
                        self.model_path, map_location=device
                    )
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model = self.model.to(device)

    def show(self) -> str:
        warnings.warn(
            "\nThe .show() method is deprecated and will be removed in future versions. Use .show_model() instead.\n"
        )
        return self.show_model()

    def show_model(self) -> str:
        """Show architecture of model to select a specific module."""
        if self.backend == "pt":
            if re.search(r"^clip", self.model_name):
                for l, (n, p) in enumerate(self.model.named_modules()):
                    if l > 1:
                        if re.search(r"^visual", n):
                            print(n)
                print("visual")
            else:
                print(self.model)
        else:
            print(self.model.summary())
        print("\nEnter module name for which you would like to extract features:\n")
        module_name = str(input())
        print()
        return module_name

    def tf_extraction(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool,
    ) -> Array:
        """Main feature extraction function for TensorFlow/Keras models."""
        features = []
        for img in tqdm(batches, desc="Batch"):
            layer_out = [self.model.get_layer(module_name).output]
            activation_model = keras.models.Model(
                inputs=self.model.input,
                outputs=layer_out,
            )
            activations = activation_model.predict(img)
            if flatten_acts:
                activations = activations.reshape(activations.shape[0], -1)
            features.append(activations)
        features = np.vstack(features)
        return features

    @torch.no_grad()
    def pt_extraction(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool,
        clip: bool = False,
    ) -> Array:
        """Main feature extraction function for PyTorch models."""
        device = torch.device(self.device)
        # initialise an empty dict to store features for each mini-batch
        global activations
        activations = {}
        # register a forward hook to store features
        model = self.register_hook()
        features = []
        for img in tqdm(batches, desc="Batch"):
            img = img.to(device)
            if clip:
                img_features = model.encode_image(img)
                if module_name == "visual":
                    assert torch.unique(
                        activations[module_name] == img_features
                    ).item(), "\nFor CLIP, image features should represent activations in last encoder layer.\n"
            else:
                _ = model(img)
            act = activations[module_name]
            if flatten_acts:
                if clip:
                    if module_name.endswith("attn"):
                        if isinstance(act, tuple):
                            act = act[0]
                    else:
                        if act.size(0) != img.shape[0] and len(act.shape) == 3:
                            act = act.permute(1, 0, 2)
                act = act.view(act.size(0), -1)
            features.append(act.cpu().numpy())
        features = np.vstack(features)
        return features

    def extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool,
        clip: bool = False,
    ) -> Array:
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
        clip : bool (optional)
            Whether neural network model is a CNN-based
            torchvision or CLIP-based model. Since CLIP
            has a different training objective, feature
            extraction must be performed differently.
            For PyTorch only.
        Returns
        -------
        output : Array
            Returns the feature matrix (e.g., X \in \mathbb{R}^{n \times p} if head or flatten_acts = True).
        """
        if self.backend == "pt":
            features = self.pt_extraction(
                batches=batches,
                module_name=module_name,
                flatten_acts=flatten_acts,
                clip=clip,
            )
        else:
            features = self.tf_extraction(
                batches=batches,
                module_name=module_name,
                flatten_acts=flatten_acts,
            )
        print(
            f"...Features successfully extracted for all {len(features)} images in the database."
        )
        print(f"...Features shape: {features.shape}")
        return features

    def get_activation(self, name: str) -> Any:
        """Store copy of hidden unit activations at each layer of model."""

        def hook(model, input, output):
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

    def register_hook(self) -> Any:
        """Register a forward hook to store activations."""
        for n, m in self.model.named_modules():
            m.register_forward_hook(self.get_activation(n))
        return self.model

    def get_transformations(
        self, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True
    ) -> Any:
        """Load image transformations for a specific model. Image transformations depend on the backend."""
        if self.preprocess:
            return self.preprocess            
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if self.backend == "pt":
                if hasattr(self, "weights") and self.weights:
                    composition = self.weights.transforms()
                else:
                    normalize = T.Normalize(mean=mean, std=std)
                    composes = [T.Resize(resize_dim)]
                    if apply_center_crop:
                        composes.append(T.CenterCrop(crop_dim))
                    composes += [T.ToTensor(), normalize]
                    composition = T.Compose(composes)

            elif self.backend == "tf":
                resize_dim = crop_dim
                composes = [
                    layers.experimental.preprocessing.Resizing(resize_dim, resize_dim)
                ]
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
            else:
                raise ValueError(
                    "\nCannot identify backend.\nChange backend to PyTorch or TensorFlow.\n"
                )
        return composition
