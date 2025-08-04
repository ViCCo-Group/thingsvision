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
        module_names: Optional[List[str]],
        flatten_acts: bool,
    ) -> Dict[str, Array]:
        layer_outputs = [self.model.get_layer(name).output for name in module_names]
        activation_model = keras.models.Model(
            inputs=self.model.input,
            outputs=layer_outputs,
        )
        activations_list = activation_model.predict(batch)
        if len(module_names) == 1:
            activations_list = [activations_list]
        activations_dict = {
            name: act for name, act in zip(module_names, activations_list)
        }
        if flatten_acts:
            for name, act in activations_dict.items():
                activations_dict[name] = act.reshape(act.shape[0], -1)
        return activations_dict

    def extract_batch(
        self,
        batch: Array,
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        flatten_acts: bool = False,
        output_type: str = "ndarray",
    ) -> Union[Array, Dict[str, Array]]:
        if not bool(module_name) ^ bool(module_names):
            raise ValueError(
                "\nPlease provide either a single module name or a list of module names, but not both.\n"
            )
        if not module_names:
            module_names = [module_name]
        self._module_and_output_check(module_names, output_type)
        # Extract features from the specified module, tensorflow does not support multiple modules extraction
        activations = self._extract_batch(batch, module_names, flatten_acts)
        if module_name:
            return activations[module_name]
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
