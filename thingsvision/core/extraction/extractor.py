import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as tensorflow_models
import timm
import torch
import torchvision

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

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


@dataclass(repr=True)
class SSLExtractor(BaseExtractor, PyTorchMixin):
    ENV_TORCH_HOME = "TORCH_HOME"
    ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
    DEFAULT_CACHE_DIR = "~/.cache"
    MODELS = {
        "simclr-rn50": {
            "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
            "arch": "resnet50",
            "type": "vissl",
        },
        "mocov2-rn50": {
            "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch",
            "arch": "resnet50",
            "type": "vissl",
        },
        "jigsaw-rn50": {
            "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch",
            "arch": "resnet50",
            "type": "vissl",
        },
        "rotnet-rn50": {
            "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch",
            "arch": "resnet50",
            "type": "vissl",
        },
        "swav-rn50": {
            "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch",
            "arch": "resnet50",
            "type": "vissl",
        },
        "pirl-rn50": {
            "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch",
            "arch": "resnet50",
            "type": "vissl",
        },
        "barlowtwins-rn50": {
            "repository": "facebookresearch/barlowtwins:main",
            "arch": "resnet50",
            "type": "hub",
        },
        "vicreg-rn50": {
            "repository": "facebookresearch/vicreg:main",
            "arch": "resnet50",
            "type": "hub",
        },
    }

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

    def _download_and_save_model(self, model_url: str, output_model_filepath: str):
        """
        Downloads the model in vissl format, converts it to torchvision format and
        saves it under output_model_filepath.
        """
        model = load_state_dict_from_url(model_url, map_location=torch.device("cpu"))

        # get the model trunk to rename
        if "classy_state_dict" in model.keys():
            model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in model.keys():
            model_trunk = model["model_state_dict"]
        else:
            model_trunk = model

        converted_model = self._replace_module_prefix(model_trunk, "_feature_blocks.")
        torch.save(converted_model, output_model_filepath)
        return converted_model

    def _replace_module_prefix(
        self, state_dict: Dict[str, Any], prefix: str, replace_with: str = ""
    ):
        """
        Remove prefixes in a state_dict needed when loading models that are not VISSL
        trained models.
        Specify the prefix in the keys that should be removed.
        """
        state_dict = {
            (
                key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key
            ): val
            for (key, val) in state_dict.items()
        }
        return state_dict

    def _get_torch_home(self):
        """
        Gets the torch home folder used as a cache directory for the vissl models.
        """
        torch_home = os.path.expanduser(
            os.getenv(
                SSLExtractor.ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(
                        SSLExtractor.ENV_XDG_CACHE_HOME, SSLExtractor.DEFAULT_CACHE_DIR
                    ),
                    "torch",
                ),
            )
        )
        return torch_home

    def load_model_from_source(self) -> None:
        """
        Load a (pretrained) neural network model from vissl. Downloads the model when it is not available.
        Otherwise, loads it from the cache directory.
        """
        if self.model_name in SSLExtractor.MODELS:
            model_config = SSLExtractor.MODELS[self.model_name]
            if model_config["type"] == "vissl":
                cache_dir = os.path.join(self._get_torch_home(), "vissl")
                model_filepath = os.path.join(cache_dir, self.model_name + ".torch")
                if not os.path.exists(model_filepath):
                    os.makedirs(cache_dir, exist_ok=True)
                    model_state_dict = self._download_and_save_model(
                        model_url=model_config["url"],
                        output_model_filepath=model_filepath,
                    )
                else:
                    model_state_dict = torch.load(
                        model_filepath, map_location=torch.device("cpu")
                    )
                self.model = getattr(torchvision.models, model_config["arch"])()
                self.model.fc = torch.nn.Identity()
                self.model.load_state_dict(model_state_dict, strict=True)
            elif model_config["type"] == "hub":
                self.model = torch.hub.load(
                    model_config["repository"], model_config["arch"]
                )
                self.model.fc = torch.nn.Identity()
            else:
                raise ValueError(f"\nUnknown model type.\n")
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in the SSLExtractor.\n"
            )
