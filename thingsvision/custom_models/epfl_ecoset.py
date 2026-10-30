"""
Ecoset-trained models released with Gokce & Schrimpf (2025), "Scaling Laws for
Task-Optimized Models of the Primate Visual Ventral Stream", ICML 2025.
https://github.com/epflneuroailab/scaling-primate-vvs

Architectures, checkpoint layout and preprocessing follow that repository
(MIT License, Copyright (c) 2024 EPFL NeuroAI Lab). Weights are downloaded from
EPFL's bucket and are not redistributed here. Ecoset: Mehrer et al. (2021), PNAS.
"""

import os
import pickle
import types
from functools import partial
from typing import Any

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .cornet.cornet_s import CORnet_S
from .custom import Custom

NUM_CLASSES = 565
BASE_URL = "https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints"

# Final checkpoint of every full-ecoset run, pinned from
# scaling_primate_vvs/artifacts/model_checkpoints.csv. Seed 0 of ResNet and
# EfficientNet has no "_seed-0" suffix, so ids are copied rather than built.
CHECKPOINTS = {
    ("alexnet", 0): "alexnet_ecoset_full_seed-0/ep100.pt",
    ("alexnet", 1): "alexnet_ecoset_full_seed-1/ep100.pt",
    ("alexnet", 2): "alexnet_ecoset_full_seed-2/ep100.pt",
    ("cornet_s", 0): "cornet_s_ecoset_full_seed-0/ep100.pt",
    ("cornet_s", 1): "cornet_s_ecoset_full_seed-1/ep100.pt",
    ("cornet_s", 2): "cornet_s_ecoset_full_seed-2/ep100.pt",
    ("resnet18", 0): "resnet18_ecoset_full/ep100.pt",
    ("resnet18", 1): "resnet18_ecoset_full_seed-1/ep100.pt",
    ("resnet18", 2): "resnet18_ecoset_full_seed-2/ep100.pt",
    ("resnet34", 0): "resnet34_ecoset_full/ep100.pt",
    ("resnet34", 1): "resnet34_ecoset_full_seed-1/ep100.pt",
    ("resnet34", 2): "resnet34_ecoset_full_seed-2/ep100.pt",
    ("resnet50", 0): "resnet50_ecoset_full/ep100.pt",
    ("resnet50", 1): "resnet50_ecoset_full_seed-1/ep100.pt",
    ("resnet50", 2): "resnet50_ecoset_full_seed-2/ep100.pt",
    ("resnet101", 0): "resnet101_ecoset_full/ep100.pt",
    ("resnet101", 1): "resnet101_ecoset_full_seed-1/ep100.pt",
    ("resnet101", 2): "resnet101_ecoset_full_seed-2/ep100.pt",
    ("resnet152", 0): "resnet152_ecoset_full/ep100.pt",
    ("resnet152", 1): "resnet152_ecoset_full_seed-1/ep100.pt",
    ("resnet152", 2): "resnet152_ecoset_full_seed-2/ep100.pt",
    ("efficientnet_b0", 0): "efficientnet_b0_ecoset_full/ep100.pt",
    ("efficientnet_b0", 1): "efficientnet_b0_ecoset_full_seed-1/ep100.pt",
    ("efficientnet_b0", 2): "efficientnet_b0_ecoset_full_seed-2/ep100.pt",
    ("efficientnet_b1", 0): "efficientnet_b1_ecoset_full/ep100.pt",
    ("efficientnet_b1", 1): "efficientnet_b1_ecoset_full_seed-1/ep100.pt",
    ("efficientnet_b1", 2): "efficientnet_b1_ecoset_full_seed-2/ep100.pt",
    ("efficientnet_b2", 0): "efficientnet_b2_ecoset_full/ep100.pt",
    ("efficientnet_b2", 1): "efficientnet_b2_ecoset_full_seed-1/ep100.pt",
    ("efficientnet_b2", 2): "efficientnet_b2_ecoset_full_seed-2/ep100.pt",
    ("convnext_tiny", 0): "convnext_tiny_ecoset_full_seed-0/ep300.pt",
    ("convnext_tiny", 1): "convnext_tiny_ecoset_full_seed-1/ep300.pt",
    ("convnext_tiny", 2): "convnext_tiny_ecoset_full_seed-2/ep300.pt",
    ("convnext_small", 0): "convnext_small_ecoset_full_seed-0/ep300.pt",
    ("convnext_small", 1): "convnext_small_ecoset_full_seed-1/ep300.pt",
    ("convnext_small", 2): "convnext_small_ecoset_full_seed-2/ep300.pt",
    ("convnext_base", 0): "convnext_base_ecoset_full_seed-0/ep300.pt",
    ("convnext_base", 1): "convnext_base_ecoset_full_seed-1/ep300.pt",
    ("convnext_base", 2): "convnext_base_ecoset_full_seed-2/ep300.pt",
    ("convnext_large", 0): "convnext_large_ecoset_full_seed-0/ep300.pt",
    ("convnext_large", 1): "convnext_large_ecoset_full_seed-1/ep300.pt",
    ("convnext_large", 2): "convnext_large_ecoset_full_seed-2/ep300.pt",
    ("deit_tiny", 0): "deit_tiny_ecoset_full_seed-0/ep300.pt",
    ("deit_tiny", 1): "deit_tiny_ecoset_full_seed-1/ep300.pt",
    ("deit_tiny", 2): "deit_tiny_ecoset_full_seed-2/ep300.pt",
    ("deit_small", 0): "deit_small_ecoset_full_seed-0/ep300.pt",
    ("deit_small", 1): "deit_small_ecoset_full_seed-1/ep300.pt",
    ("deit_small", 2): "deit_small_ecoset_full_seed-2/ep300.pt",
    ("deit_base", 0): "deit_base_ecoset_full_seed-0/ep300.pt",
    ("deit_base", 1): "deit_base_ecoset_full_seed-1/ep300.pt",
    ("deit_base", 2): "deit_base_ecoset_full_seed-2/ep300.pt",
    ("deit_large", 0): "deit_large_ecoset_full_seed-0/ep300.pt",
    ("deit_large", 1): "deit_large_ecoset_full_seed-1/ep300.pt",
    ("deit_large", 2): "deit_large_ecoset_full_seed-2/ep300.pt",
}

VARIANTS = list(dict.fromkeys(variant for variant, _ in CHECKPOINTS))
SEEDS = [0, 1, 2]

# DeiT III (timm's deit3_*_patch16_224): layer scale, no position embedding on the class token.
DEIT_CONFIGS = {
    "deit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "deit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "deit_base": dict(embed_dim=768, depth=12, num_heads=12),
    "deit_large": dict(embed_dim=1024, depth=24, num_heads=16),
}

# (resize, crop) from EPFL's training configs; every other variant uses (256, 224).
IMAGE_SIZES = {
    "efficientnet_b1": (255, 240),
    "efficientnet_b2": (288, 288),
}

# The checkpoints are full MosaicML composer checkpoints, so besides tensors they
# hold rng state and timestamps. torch 2.0 cannot extend the weights_only
# allowlist, hence this restricted unpickler. These are all the globals the files use.
SAFE_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch.torch_version", "TorchVersion"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("_codecs", "encode"),
    ("datetime", "timedelta"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in SAFE_GLOBALS:
            raise pickle.UnpicklingError(
                f"\nRefusing to load '{module}.{name}' from an EPFL ecoset checkpoint.\n"
            )
        return super().find_class(module, name)


# torch.load uses Unpickler for zip checkpoints and load() for legacy ones, so restrict both
_restricted_pickle = types.ModuleType("_restricted_pickle")
_restricted_pickle.Unpickler = _RestrictedUnpickler
_restricted_pickle.load = lambda file, **kwargs: _RestrictedUnpickler(
    file, **kwargs
).load()


def _build_model(variant: str) -> nn.Module:
    if variant == "cornet_s":
        model = CORnet_S()
        model.decoder.linear = nn.Linear(512, NUM_CLASSES)
        return model
    if variant in DEIT_CONFIGS:
        return VisionTransformer(
            patch_size=16,
            no_embed_class=True,
            init_values=1e-6,
            num_classes=NUM_CLASSES,
            **DEIT_CONFIGS[variant],
        )
    return getattr(models, variant)(weights=None, num_classes=NUM_CLASSES)


def _load_epfl_state_dict(checkpoint: str, variant: str) -> dict[str, torch.Tensor]:
    model_id, file_name = checkpoint.split("/")
    # every run's file is called ep100.pt or ep300.pt, so the cache name needs the model id
    path = os.path.join(torch.hub.get_dir(), "checkpoints", f"{model_id}_{file_name}")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(f"{BASE_URL}/{checkpoint}", path)
    # load on CPU: ConvNeXt and DeiT files hold two copies of the weights (raw and EMA),
    # and the extractor moves the model to the device afterwards
    ckpt = torch.load(
        path, map_location="cpu", pickle_module=_restricted_pickle, weights_only=False
    )
    # EPFL evaluates ConvNeXt and DeiT with their EMA weights
    key = (
        "model_ema_state_dict" if variant.startswith(("convnext", "deit")) else "model"
    )
    return {k.removeprefix("module."): v for k, v in ckpt["state"][key].items()}


def _get_preprocessing(resize_dim: int, crop_dim: int) -> T.Compose:
    # Reproduces EPFL's validation transform (albumentations): the image is squashed
    # to a square with OpenCV bilinear, which does not antialias (their `interpolation`
    # setting never takes effect), and the crop offset is floored. T.CenterCrop rounds
    # instead, which would shift EfficientNet-B1 by a pixel.
    offset = (resize_dim - crop_dim) // 2
    return T.Compose(
        [
            T.PILToTensor(),
            T.Resize(
                (resize_dim, resize_dim),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=False,
            ),
            partial(TF.crop, top=offset, left=offset, height=crop_dim, width=crop_dim),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class EPFL_ecoset(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.variant = parameters.get("variant")
        self.seed = parameters.get("seed", 0)
        self.check_available_variants_and_seeds()

    def check_available_variants_and_seeds(self) -> None:
        if (self.variant, self.seed) not in CHECKPOINTS:
            raise ValueError(
                f"\nCould not find EPFL ecoset model with variant={self.variant!r} and seed={self.seed!r}."
                f"\nAvailable variants: {VARIANTS}\nAvailable seeds: {SEEDS}\n"
            )

    def create_model(self) -> Any:
        model = _build_model(self.variant)
        state_dict = _load_epfl_state_dict(
            CHECKPOINTS[(self.variant, self.seed)], self.variant
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        resize_dim, crop_dim = IMAGE_SIZES.get(self.variant, (256, 224))
        return model, _get_preprocessing(resize_dim, crop_dim)
