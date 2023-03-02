from typing import Any, Callable, Dict, Union

import numpy as np
import thingsvision.custom_models as custom_models
import thingsvision.custom_models.cornet as cornet
from torchtyping import TensorType

import torch
import warnings

from .extractors import (
    KerasExtractor,
    SSLExtractor,
    TimmExtractor,
    TorchvisionExtractor,
)
from .tensorflow import TensorFlowExtractor
from .torch import PyTorchExtractor

Array = np.ndarray
AxisError = np.AxisError


def create_custom_extractor(
    model_name: str,
    pretrained: bool,
    device: str,
    model_path: str = None,
    model_parameters: Dict[str, str] = None,
) -> Any:
    """Create a custom extractor from a pretrained model."""
    if model_name.startswith("cornet"):
        backend = "pt"
        try:
            model = getattr(cornet, f"cornet_{model_name[-1]}")
        except AttributeError:
            model = getattr(cornet, f"cornet_{model_name[-2:]}")
        model = model(pretrained=pretrained, map_location=torch.device(device))
        model = model.module  # remove DataParallel
        preprocess = None
    elif hasattr(custom_models, model_name):
        model_parameters = model_parameters if model_parameters else {}
        custom_model = getattr(custom_models, model_name)
        custom_model = custom_model(device, model_parameters)
        model, preprocess = custom_model.create_model()
        backend = custom_model.get_backend()
    else:
        raise ValueError(
            f"\nCould not find {model_name} among custom models.\nChoose a different model.\n"
        )

    Extractor = PyTorchExtractor if backend == "pt" else TensorFlowExtractor

    class CustomExtractor(Extractor):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    # TODO: this should probably be defined in the custom model itself
    if model_name.lower().startswith("clip"):

        def show_model(self):
            for l, (n, p) in enumerate(self.model.named_modules()):
                if l > 1:
                    if n.startswith("visual"):
                        print(n)
            print("visual")

        @staticmethod
        def forward(batch: TensorType["b", "c", "h", "w"]) -> TensorType["b", "d"]:
            img_features = model.encode_image(batch)
            return img_features

        @staticmethod
        def flatten_acts(
            act: Union[
                TensorType["b", "num_maps", "h_prime", "w_prime"],
                TensorType["b", "t", "d"],
            ],
            batch: TensorType["b", "c", "h", "w"],
            module_name: str,
        ) -> TensorType["b", "p"]:
            if module_name.endswith("attn"):
                if isinstance(act, tuple):
                    act = act[0]
            else:
                if act.size(0) != batch.shape[0] and len(act.shape) == 3:
                    act = act.permute(1, 0, 2)
            act = act.view(act.size(0), -1)
            return act

        CustomExtractor.show_model = show_model
        CustomExtractor.forward = forward
        CustomExtractor.flatten_acts = flatten_acts

    if model_name == "OpenCLIP":

        def forward(
            self, batch: TensorType["b", "c", "h", "w"]
        ) -> TensorType["b", "d"]:
            return self.model(batch, text=None)

        CustomExtractor.forward = forward

    custom_extractor = CustomExtractor(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        model_path=model_path,
        model=model,
        preprocess=preprocess,
    )

    return custom_extractor


def create_model_extractor(
    model: Any,
    device: str,
    preprocess: Any = None,
    backend: str = "pt",
    forward_fn: Callable = None,
    flatten_fn: Callable = None,
) -> Any:
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
        In case your model requires more complicated forward passes than the default,
        for example if the forward pass requires multiple inputs, you can pass a custom
        forward function here, which should take a batch of images as input. Only required
        for PyTorch models.

        Default:
            def forward(
                self, batch: TensorType["b", "c", "h", "w"]
            ) -> TensorType["b", "num_cls"]:
                return self.model(batch)

    flatten_fn: Callable
        In case your model requires more complicated flattening of activations than
        the default, you can pass a custom flattening function here. Only required for
        PyTorch models.

        Default:
            def flatten_acts(
                act: Union[
                    TensorType["b", "num_maps", "h_prime", "w_prime"], TensorType["b", "t", "d"]
                ]
            ) -> TensorType["b", "p"]:
                return act.view(act.size(0), -1)

    Returns:
    --------
    extractor: Any
        The custom extractor class for your model.
    """
    Extractor = PyTorchExtractor if backend == "pt" else TensorFlowExtractor

    class ModelExtractor(Extractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    if forward_fn:
        ModelExtractor.forward = forward_fn

    if flatten_fn:
        ModelExtractor.flatten_acts = flatten_fn

    model_extractor = ModelExtractor(
        model_name="custom",
        model_path=None,
        device=device,
        model=model,
        preprocess=preprocess,
        pretrained=True,
    )

    return model_extractor


def get_extractor(
    model_name: str,
    pretrained: bool,
    device: str,
    source: str,
    model_path: str = None,
    model_parameters: Dict[str, str] = None,
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
    elif source == "ssl":
        return SSLExtractor(**model_args)
    elif source == "vissl":
        warnings.warn(
            message='\nThe source "vissl" is deprecated. Use the source "ssl" instead.\n',
            category=DeprecationWarning,
            stacklevel=2,
        )
        return SSLExtractor(**model_args)
    else:
        raise ValueError(
            f"\nCould not find source: {source}.\nChoose a different source.\n"
        )


def get_extractor_from_model(
    model: Any,
    device: str,
    backend: str,
    preprocess: Any = None,
    forward_fn: Callable = None,
    flatten_fn: Callable = None,
) -> Any:
    """Get a model extractor from a model."""
    return create_model_extractor(
        model, device, preprocess, backend, forward_fn, flatten_fn
    )


def center_features(X: Array) -> Array:
    """Center features to have zero mean."""
    X -= X.mean(axis=0)
    return X


def normalize_features(X: Array) -> Array:
    """Normalize feature vectors by their l2-norm."""
    try:
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
    except AxisError:
        raise Exception(
            "\nMake sure that features are represented as an n-dimensional NumPy array\n"
        )
    return X
