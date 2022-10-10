import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Tuple, Dict

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
class BaseExtractor:
    model_name: str
    pretrained: bool
    device: str 
    model_path: str = None
    model_parameters: Any = field(default_factory=lambda: {})
    model: Any = None
    preprocess: Any = None

    def __post_init__(self):
        if not self.model:
            self.load_model()

    def show(self) -> str:
        warnings.warn(
            "\nThe .show() method is deprecated and will be removed in future versions. Use .show_model() instead.\n"
        )
        return self.show_model()

    def show_model(self) -> str:
        return self._show_model()

    def extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool
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
         Returns
        -------
        output : Array
            Returns the feature matrix (e.g., X \in \mathbb{R}^{n \times p} if head or flatten_acts = True).
        """
        features = self._extract_features(batches, module_name, flatten_acts)

        print(
            f"...Features successfully extracted for all {len(features)} images in the database."
        )
        print(f"...Features shape: {features.shape}")

        return features

    def get_transformations(
        self, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True
    ) -> Any:
        """Load image transformations for a specific model. Image transformations depend on the backend."""
        if self.preprocess:
            return self.preprocess
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            composition = self.get_default_transformation(mean, std, resize_dim, crop_dim, apply_center_crop)
        return composition


@dataclass
class PyTorchMixin:
    backend: str = 'pt'

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

    @torch.no_grad()
    def _extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool
    ):
        device = torch.device(self.device)
        # initialise an empty dict to store features for each mini-batch
        global activations
        activations = {}
        # register a forward hook to store features
        _ = self.register_hook()
        features = []
        for batch in tqdm(batches):
            batch = batch.to(device)
            with torch.no_grad():
                _ = self.forward(batch, module_name)
            act = activations[module_name]
            if flatten_acts:
                act = self.flatten_acts(act, batch, module_name)
            features.append(act.cpu().numpy())
        features = np.vstack(features)

        return features

    def forward(self, batch: Tensor, module_name: str) -> Tensor:
        """Default forward pass."""
        return self.model(batch)

    def flatten_acts(self, act: Tensor, img: Tensor, module_name: str) -> Tensor:
        """Default flatten of activations."""
        act = act.view(act.size(0), -1)

        return act

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

    def get_default_transformation(self, mean, std, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True) -> Any:
        normalize = T.Normalize(mean=mean, std=std)
        composes = [T.Resize(resize_dim)]
        if apply_center_crop:
            composes.append(T.CenterCrop(crop_dim))
        composes += [T.ToTensor(), normalize]
        composition = T.Compose(composes)

        return composition


@dataclass
class TensorFlowMixin:
    backend: str = 'tf'

    def _extract_features(
        self, 
        batches: Iterator,
        module_name: str,
        flatten_acts: bool
    ) -> Array:
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

    def _show_model(self) -> str:
        return self.model.summary()

    def load_model(self) -> Any:
        self.load_model_from_source()
        if self.model_path:
            self.model.load_weights(self.model_path)
        self.model.trainable = False

    def get_default_transformation(self, mean, std, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True) -> Any:
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

        return composition  



class TorchvisionExtractor(BaseExtractor, PyTorchMixin):
    def __init__(
        self,
        model_name: str,
        pretrained: bool, 
        device: str,
        model_path: str = None,
        model_parameters: Dict = None,
        preprocess: Any = None
    ):
        model_parameters = model_parameters if model_parameters else {
            "weights": "DEFAULT"
        }
        super().__init__(
            model_name=model_name, 
            pretrained=pretrained, 
            model_path=model_path, 
            model_parameters=model_parameters, 
            preprocess=preprocess,
            device=device
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
        weights = getattr(getattr(torchvision.models, f"{weights_name}"), self.model_parameters["weights"])
        return weights

    def load_model_from_source(self) -> Tuple[Any, str]:
        """Load a (pretrained) neural network model from <torchvision>."""
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

    def get_default_transformation(self, mean, std, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True) -> Any:
        if self.weights:
            transforms = self.weights.transforms()
        else:
            transforms = super().get_default_transformation(mean, std, resize_dim, crop_dim, apply_center_crop)

        return transforms


class TimmExtractor(BaseExtractor, PyTorchMixin):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict = None,
        preprocess: Any = None
    ):
        super().__init__(
            model_name=model_name, 
            pretrained=pretrained, 
            model_path=model_path, 
            model_parameters=model_parameters, 
            preprocess=preprocess,
            device=device
        )

    def load_model_from_source(self) -> Tuple[Any, str]:
        """Load a (pretrained) neural network model from <timm>."""
        if self.model_name in timm.list_models():
            self.model = timm.create_model(self.model_name, self.pretrained)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in timm library.\nChoose a different model.\n"
            )


class KerasExtractor(BaseExtractor, TensorFlowMixin):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict = None,
        preprocess: Any = None
    ):
        model_parameters = model_parameters if model_parameters else {
            "weights": "imagenet"
        }
        super().__init__(
            model_name=model_name, 
            pretrained=pretrained, 
            model_path=model_path, 
            model_parameters=model_parameters, 
            preprocess=preprocess,
            device=device
        )
        

    def load_model_from_source(self) -> Tuple[Any, str]:
        """Load a (pretrained) neural network model from <keras>."""
        if hasattr(tensorflow_models, self.model_name):
            model = getattr(tensorflow_models, self.model_name)
            if self.pretrained:
                weights = self.model_parameters['weights']
            elif self.model_path:
                weights = self.model_path
            else:
                weights = None
            self.model = model(weights=weights)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} among TensorFlow models.\n"
            )


def create_custom_extractor(
    model_name: str,
    pretrained: bool,
    device: str,
    model_path: str = None,
    model_parameters: Dict = None
) -> Any:
    """Create a custom extractor from a pretrained model."""
    if model_name.startswith("cornet"):
        backend = "pt"
        try:
            model = getattr(cornet, f"cornet_{model_name[-1]}")
        except AttributeError:
            model = getattr(cornet, f"cornet_{model_name[-2:]}")
        model = model(
            pretrained=pretrained, map_location=torch.device(device)
        )
        model = model.module  # remove DataParallel
    elif hasattr(custom_models, model_name):
        custom_model = getattr(custom_models, model_name)
        custom_model = custom_model(device, model_parameters)
        model, preprocess = custom_model.create_model()
        backend = custom_model.get_backend()
    else:
        raise ValueError(
            f"\nCould not find {model_name} among custom models.\nChoose a different model.\n"
        )

    backend_mixin = PyTorchMixin if backend == "pt" else TensorFlowMixin

    class CustomExtractor(BaseExtractor, backend_mixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    #TODO: this should probably be defined in the custom model itself
    if model_name.lower().startswith('clip'):  
        def show_model(self):
            for l, (n, p) in enumerate(self.model.named_modules()):
                if l > 1:
                    if n.startswith("visual"):
                        print(n)
            print("visual")

        def forward(self, batch: Tensor, module_name: str) -> Tensor: 
            img_features = model.encode_image(batch)
            if module_name == "visual":
                assert torch.unique(
                    activations[module_name] == img_features
                ).item(), "\nFor CLIP, image features should represent activations in last encoder layer.\n"
            
            return img_features

        def flatten_acts(self, act: Tensor, img: Tensor, module_name: str) -> Tensor:
            if module_name.endswith("attn"):
                if isinstance(act, tuple):
                    act = act[0]
            else:
                if act.size(0) != img.shape[0] and len(act.shape) == 3:
                    act = act.permute(1, 0, 2)
            act = act.view(act.size(0), -1)
            return act

        CustomExtractor.show_model = show_model
        CustomExtractor.forward = forward
        CustomExtractor.flatten_acts = flatten_acts

    custom_extractor = CustomExtractor(
        model_name=model_name, 
        pretrained=pretrained,
        device=device, 
        model_path=model_path, 
        model=model, 
        preprocess=preprocess
    )

    return custom_extractor



def create_model_extractor(
    model: Any,
    device: str, 
    preprocess: Any = None,
    backend: str = "pt",
    forward_fn: Callable = None,
):
    """
    Creates a class for extracting activations from a given model (PyTorch or TensorFlow).

    Parameters:
    -----------
    model: Any
        The model from which activations will be extracted.
    device: str
        The device on which the model is loaded.
    preprocess: Any
        The preprocessing function to be applied to the input images (default: None).
    backend: str
        The backend of the model. Either "pt" for PyTorch or "tf" for TensorFlow.
    forward_fn: Callable
        In case your model requires more complicated forward passes than simply using model(img),
        you can pass a custom forward function here. The function must have the following signature:
        
        forward_fn(self, img, module_name) -> activations
        
        and calls to the model have to be made on the self.model attribute.

    Returns:
    --------
    extractor: Any
        The custom extractor class.
    """
    backend_mixin = PyTorchMixin if backend == "pt" else TensorFlowMixin

    class ModelExtractor(BaseExtractor, backend_mixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    if forward_fn:
        ModelExtractor.forward = forward_fn

    model_extractor = ModelExtractor(
        model_name="custom", 
        model_path=None, 
        device=device, 
        model=model, 
        preprocess=preprocess
    )

    return model_extractor


def get_extractor(
    model_name: str,
    pretrained: bool,
    device: str,
    source: str,
    model_path: str = None,
    model_parameters: Dict = None,
) -> Any:

    model_args = {
        "model_name": model_name,
        "model_path": model_path,
        "device": device,
        "pretrained": pretrained,
        "model_parameters": model_parameters,
    }

    """Get a model extractor from <source> library."""
    if source == "torchvision":
        return TorchvisionExtractor(**model_args)
    elif source == "timm":
        return TimmExtractor(**model_args)
    elif source == "keras":
        return KerasExtractor(**model_args)
    elif source == "custom":
        return create_custom_extractor(**model_args)
    else:
        raise ValueError(
            f"\nCould not find {source} library.\nChoose a different source.\n"
        )


def get_extractor_from_model(
    model: Any,
    device: str,
    preprocess: Any = None,
    backend: str = "pt",
    forward_fn: Callable = None,
) -> Any:
    """Get a model extractor from a model."""
    return create_model_extractor(model, device, preprocess, backend, forward_fn)