import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Tuple, List

import numpy as np
import tensorflow as tf
import torch
import torchvision
from tensorflow import keras
from tensorflow.keras import layers
from torchvision import transforms as T
from tqdm import tqdm

Tensor = torch.Tensor
Array = np.ndarray


@dataclass
class PyTorchMixin:
    backend: str = field(init=False, default="pt")

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

    @torch.no_grad()
    def _extract_features(self, batch, module_name: str, flatten_acts: bool) -> Array:
        device = torch.device(self.device)
        self.model = self.model.to(device)
        # initialise an empty dict to store features for each mini-batch
        global activations
        activations = {}
        # register forward hook to store features
        self.register_hook()

        batch = batch.to(device)
        _ = self.forward(batch)
        act = activations[module_name]
        if flatten_acts:
            act = self.flatten_acts(act).cpu().numpy()

        return act

    def forward(self, batch: Tensor) -> Tensor:
        """Default forward pass."""
        return self.model(batch)

    def flatten_acts(self, act: Tensor) -> Tensor:
        """Default flattening of activations."""
        return act.view(act.size(0), -1)

    def _show_model(self) -> str:
        return self.model

    def load_model(self) -> Any:
        self.load_model_from_source()
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

    def get_module_names(self) -> List[str]:
        module_names, _ = zip(*self.model.named_modules())
        module_names = list(filter(lambda n: len(n) > 0, module_names))
        return module_names

    def get_default_transformation(
        self,
        mean,
        std,
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Any:
        normalize = T.Normalize(mean=mean, std=std)
        composes = [T.Resize(resize_dim)]
        if apply_center_crop:
            composes.append(T.CenterCrop(crop_dim))
        composes += [T.ToTensor(), normalize]
        composition = T.Compose(composes)

        return composition

    def get_backend(self) -> str:
        return "pt"


@dataclass
class TensorFlowMixin:
    backend: str = field(init=False, default="tf")

    def _extract_features(self, batch, module_name: str, flatten_acts: bool) -> Array:
        layer_out = [self.model.get_layer(module_name).output]
        activation_model = keras.models.Model(
            inputs=self.model.input,
            outputs=layer_out,
        )
        activations = activation_model.predict(batch)
        if flatten_acts:
            activations = activations.reshape(activations.shape[0], -1)

        return activations

    def _show_model(self) -> str:
        return self.model.summary()

    def load_model(self) -> Any:
        self.load_model_from_source()
        if self.model_path:
            self.model.load_weights(self.model_path)
        self.model.trainable = False

    def get_module_names(self) -> List[str]:
        module_names = [l._name for l in self.model.submodules]
        return module_names

    def get_default_transformation(
        self,
        mean,
        std,
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
