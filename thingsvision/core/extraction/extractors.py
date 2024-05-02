import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import timm
import torchvision

import tensorflow as tf
import tensorflow.keras.applications as tensorflow_models
import torch

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from thingsvision.utils.checkpointing import get_torch_home
from thingsvision.utils.models.dino import vit_base, vit_small, vit_tiny
from thingsvision.utils.models.mae import (
    interpolate_pos_embed,
    vit_base_patch16,
    vit_huge_patch14,
    vit_large_patch16,
)

from .tensorflow import TensorFlowExtractor
from .torch import PyTorchExtractor

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


class TorchvisionExtractor(PyTorchExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict[str, Union[str, bool, List[str]]] = None,
        preprocess: Optional[Callable] = None,
    ) -> None:
        model_parameters = (
            model_parameters if model_parameters else {"weights": "DEFAULT"}
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
            self.model_parameters["weights"],
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


class TimmExtractor(PyTorchExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict[str, Union[str, bool, List[str]]] = None,
        preprocess: Optional[Callable] = None,
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
        if self.model_name.split(".")[0] in timm.list_models():
            self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in timm library.\nChoose a different model.\n"
            )


class KerasExtractor(TensorFlowExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict[str, Union[str, bool, List[str]]] = None,
        preprocess: Optional[Callable] = None,
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


class SSLExtractor(PyTorchExtractor):
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
        "dino-vit-small-p16": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_vits16",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        },
        "dino-vit-small-p8": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_vits8",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
        },
        "dino-vit-base-p16": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_vitb16",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        },
        "dino-vit-base-p8": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_vitb8",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        },
        "dino-xcit-small-12-p16": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_xcit_small_12_p16",
            "type": "hub",
        },
        "dino-xcit-small-12-p8": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_xcit_small_12_p8",
            "type": "hub",
        },
        "dino-xcit-medium-24-p16": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_xcit_medium_24_p16",
            "type": "hub",
        },
        "dino-xcit-medium-24-p8": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_xcit_medium_24_p8",
            "type": "hub",
        },
        "dino-rn50": {
            "repository": "facebookresearch/dino:main",
            "arch": "dino_resnet50",
            "type": "hub",
        },
        "dinov2-vit-small-p14": {
            "repository": "facebookresearch/dinov2",
            "arch": "dinov2_vits14",
            "type": "hub",
        },
        "dinov2-vit-base-p14": {
            "repository": "facebookresearch/dinov2",
            "arch": "dinov2_vitb14",
            "type": "hub",
        },
        "dinov2-vit-large-p14": {
            "repository": "facebookresearch/dinov2",
            "arch": "dinov2_vitl14",
            "type": "hub",
        },
        "dinov2-vit-giant-p14": {
            "repository": "facebookresearch/dinov2",
            "arch": "dinov2_vitg14",
            "type": "hub",
        },
        "mae-vit-base-p16": {
            "repository": "facebookresearch/mae",
            "arch": "mae_vit_base_patch16",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        },
        "mae-vit-large-p16": {
            "repository": "facebookresearch/mae",
            "arch": "mae_vit_large_patch16",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        },
        "mae-vit-huge-p14": {
            "repository": "facebookresearch/mae",
            "arch": "mae_vit_huge_patch14",
            "type": "hub",
            "checkpoint_url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        },
    }

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict[str, Union[str, bool, List[str]]] = None,
        preprocess: Optional[Callable] = None,
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

    def load_model_from_source(self) -> None:
        """
        Load a (pretrained) neural network model from vissl. Downloads the model when it is not available.
        Otherwise, loads it from the cache directory.
        """
        if self.model_name in SSLExtractor.MODELS:
            model_config = SSLExtractor.MODELS[self.model_name]
            if model_config["type"] == "vissl":
                cache_dir = os.path.join(get_torch_home(), "vissl")
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
                if model_config["arch"] == "resnet50":
                    self.model.fc = torch.nn.Identity()
                self.model.load_state_dict(model_state_dict, strict=True)
            elif model_config["type"] == "hub":
                if self.model_name.startswith("dino-vit"):
                    if self.model_name == "dino-vit-tiny-p8":
                        model = vit_tiny(patch_size=8)
                    elif self.model_name == "dino-vit-tiny-p16":
                        model = vit_tiny(patch_size=16)
                    elif self.model_name == "dino-vit-small-p8":
                        model = vit_small(patch_size=8)
                    elif self.model_name == "dino-vit-small-p16":
                        model = vit_small(patch_size=16)
                    elif self.model_name == "dino-vit-base-p8":
                        model = vit_base(patch_size=8)
                    elif self.model_name == "dino-vit-base-p16":
                        model = vit_base(patch_size=16)
                    else:
                        raise ValueError(f"\n{self.model_name} is not available.\n")
                    state_dict = torch.hub.load_state_dict_from_url(
                        model_config["checkpoint_url"]
                    )
                    model.load_state_dict(state_dict, strict=True)
                    self.model = model
                elif self.model_name.startswith("mae"):
                    if self.model_name == "mae-vit-base-p16":
                        model = vit_base_patch16(num_classes=0, drop_rate=0.0)
                    elif self.model_name == "mae-vit-large-p16":
                        model = vit_large_patch16(num_classes=0, drop_rate=0.0)
                    elif self.model_name == "mae-vit-huge-p14":
                        model = vit_huge_patch14(num_classes=0, drop_rate=0.0)
                    else:
                        raise ValueError(f"\n{self.model_name} is not available.\n")
                    state_dict = torch.hub.load_state_dict_from_url(
                        model_config["checkpoint_url"]
                    )
                    checkpoint_model = state_dict["model"]
                    # interpolate position embedding
                    interpolate_pos_embed(model, checkpoint_model)
                    model.load_state_dict(checkpoint_model, strict=False)
                    self.model = model
                else:
                    self.model = torch.hub.load(
                        model_config["repository"], model_config["arch"]
                    )
                    if model_config["arch"] == "resnet50":
                        self.model.fc = torch.nn.Identity()
            else:
                type = model_config["type"]
                raise ValueError(f"\nUnknown model type: {type}.\n")
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} in the SSLExtractor.\nUse a different model.\n"
            )
