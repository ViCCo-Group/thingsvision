import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .base import BaseExtractor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress tensorflow warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

Array = np.ndarray


class TensorFlowExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict[str, Union[str, bool, List[str]]] = None,
        model: Any = None,
        preprocess: Optional[Callable] = None,
    ) -> None:
        super().__init__(device, preprocess)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.model = model

        if not self.model:
            self.load_model()
        self.prepare_inference()

    def _extract_batch(
        self,
        batch: Array,
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        flatten_acts: bool = False,
    ) -> Array:
        assert (
            module_name ^ module_names
        ), "Please provide either a single module name or a list of module names for which features should be extracted.\n"
        if module_name:
            module_names = [module_name]
        layer_outs = [self.model.get_layer(name).output for name in module_names]
        activation_model = keras.models.Model(
            inputs=self.model.input,
            outputs=layer_outs,
        )
        activations = activation_model.predict(batch)
        if flatten_acts:
            activations = activations.reshape(activations.shape[0], -1)
        return activations

    def extract_batch(
        self,
        batch: Array,
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        flatten_acts: bool = False,
        output_type: str = "ndarray",
    ) -> Array:
        assert (
            module_name ^ module_names
        ), "Please provide either a single module name or a list of module names for which features should be extracted.\n"
        if module_name:
            module_names = [module_name]
        self.model = self.model.to(self.device)
        self.activations = {}
        if module_name:
            module_names = [module_name]
        self._module_and_output_check(module_names, output_type)
        activations = self._extract_batch(batch, module_names, flatten_acts)
        return activations

    def show_model(self) -> str:
        return self.model.summary()

    def load_model_from_source(self) -> None:
        raise NotImplementedError

    def load_model(self) -> None:
        self.load_model_from_source()
        if self.model_path:
            self.model.load_weights(self.model_path)

    def prepare_inference(self) -> None:
        self.model.trainable = False

    def get_module_names(self) -> List[str]:
        return [l._name for l in self.model.submodules]

    def get_default_transformation(
        self,
        mean: List[float],
        std: List[float],
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Any:
        resize_dim = crop_dim
        composes = [
            layers.experimental.preprocessing.Resizing(resize_dim, resize_dim),
            layers.experimental.preprocessing.Rescaling(1.0 / 255.0),
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

        return composition

    def get_backend(self) -> str:
        return "tf"
