# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.models import vision_transformer


class VisionTransformer(vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

    def forward_features(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Orig. vision_transformer.VisionTransformer.forward_features:
        https://github.com/huggingface/pytorch-image-models/blob/v1.0.19/timm/models/vision_transformer.py
        Add cls_token to after patch_embed and before patch_drop.
        """
        x = super().forward_features(x, attn_mask)
        outcome = x[
            :, 0
        ]  # TODO: Not clear if we should not return all tokens or just the cls token
        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
