from typing import Any
import torch

from .custom import Custom
from functools import partial
from thingsvision.utils.models.sam.image_encoder import ImageEncoderViT
from torchvision import transforms as T

MODEL_CONFIG = {
    "vit_b": {
        "weights": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "config": dict(
            embed_dim=768,
            depth=12,
            num_heads=12,
            global_attn_indexes=[2, 5, 8, 11]
        )
    },
    "vit_l": {
        "weights": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "config": dict(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            global_attn_indexes=[5, 11, 17, 23]
        )
    },
    "vit_h": {
        "weights": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "config": dict(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            global_attn_indexes=[7, 15, 23, 31]
        )
    }
}


def _get_preprocessing(resize_dim=1024, crop_dim=1024):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose(([T.Resize(resize_dim),
                       T.CenterCrop(crop_dim),
                       T.ToTensor(), normalize]))


class SegmentAnything(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.variant = parameters.get("variant", "vit_h")

    def check_available_variants_and_datasets(self):
        if not self.variant in MODEL_CONFIG.keys():
            raise ValueError(
                f"\nCould not find Segment Anything model with variant: {self.variant}.\n"
            )

    def create_model(self) -> Any:
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        params = MODEL_CONFIG[self.variant]['config']
        model = ImageEncoderViT(
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            window_size=14,
            out_chans=prompt_embed_dim,
            **params
        )
        model.eval()
        return model, _get_preprocessing()
