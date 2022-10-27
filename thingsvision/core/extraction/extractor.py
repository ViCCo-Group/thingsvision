from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as tensorflow_models
import timm
import torch
import torchvision

from .base import BaseExtractor
from .mixin import PyTorchMixin, TensorFlowMixin

# neccessary to prevent gpu memory conflicts between torch and tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

Tensor = torch.Tensor
Array = np.ndarray


@dataclass(repr=True)
class TorchvisionExtractor(BaseExtractor, PyTorchMixin):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict = None,
        preprocess: Any = None,
    ) -> None:
        model_parameters = (
            model_parameters if model_parameters else {"weights": "DEFAULT"},
        )
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            model_path=model_path,
            model_parameters=model_parameters,
            preprocess=preprocess,
            device=device,
        )

    def get_weights(self, model_name: str, suffix: str = "_weights") -> Any:
        weights_name = None
        for m in dir(torchvision.models):
            if m.lower() == model_name + suffix:
                weights_name = m
                break
        if not weights_name:
            raise ValueError(
                f"\nCould not find pretrained weights for {model_name} in <torchvision>. Choose a different model or change the source.\n"
            )
        weights = getattr(
            getattr(torchvision.models, f"{weights_name}"),
            self.model_parameters[0]["weights"],
        )
        return weights

    def load_model_from_source(self) -> None:
        """Load a (pretrained) neural network model from <torchvision>."""
        if hasattr(torchvision.models, self.model_name):
            model = getattr(torchvision.models, self.model_name)
            if self.pretrained:
                self.weights = self.get_weights(self.model_name)
            else:
                self.weights = None
            self.model = model(weights=self.weights)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in torchvision library.\nChoose a different model.\n"
            )

    def get_default_transformation(
        self,
        mean,
        std,
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Any:
        if self.weights:
            transforms = self.weights.transforms()
        else:
            transforms = super().get_default_transformation(
                mean, std, resize_dim, crop_dim, apply_center_crop
            )

        return transforms


@dataclass(repr=True)
class TimmExtractor(BaseExtractor, PyTorchMixin):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict = None,
        preprocess: Any = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            model_path=model_path,
            model_parameters=model_parameters,
            preprocess=preprocess,
            device=device,
        )

    def load_model_from_source(self) -> None:
        """Load a (pretrained) neural network model from <timm>."""
        if self.model_name in timm.list_models():
            self.model = timm.create_model(self.model_name, self.pretrained)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in timm library.\nChoose a different model.\n"
            )


@dataclass(repr=True)
class KerasExtractor(BaseExtractor, TensorFlowMixin):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict = None,
        preprocess: Any = None,
    ) -> None:
        model_parameters = (
            model_parameters if model_parameters else {"weights": "imagenet"}
        )
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            model_path=model_path,
            model_parameters=model_parameters,
            preprocess=preprocess,
            device=device,
        )

    def load_model_from_source(self) -> None:
        """Load a (pretrained) neural network model from <keras>."""
        if hasattr(tensorflow_models, self.model_name):
            model = getattr(tensorflow_models, self.model_name)
            if self.pretrained:
                weights = self.model_parameters["weights"]
            elif self.model_path:
                weights = self.model_path
            else:
                weights = None
            self.model = model(weights=weights)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} among TensorFlow models.\n"
            )
