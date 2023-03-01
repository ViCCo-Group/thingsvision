from harmonization.models import (
    load_ViT_B16,
    load_ResNet50,
    load_VGG16,
    load_EfficientNetB0,
    load_tiny_ConvNeXT,
    # load_tiny_MaxViT,
    load_LeViT_small,
)
from typing import Any
from .custom import Custom


class Harmonization(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"
        self.variant = parameters.get("variant", "ViT_B16")

    def check_available_variants(self):
        variants = [
            "ViT_B16",
            "ResNet50",
            "VGG16",
            "EfficientNetB0",
            "tiny_ConvNeXT",
            # "tiny_MaxViT",
            "LeViT_small",
        ]

        if self.variant not in variants:
            raise ValueError(f"\nVariant must be one of {variants}")

    def create_model(self) -> Any:
        self.check_available_variants()
        variant_function_dict = {
            "ViT_B16": load_ViT_B16,
            "ResNet50": load_ResNet50,
            "VGG16": load_VGG16,
            "EfficientNetB0": load_EfficientNetB0,
            "tiny_ConvNeXT": load_tiny_ConvNeXT,
            # "tiny_MaxViT": load_tiny_MaxViT(),
            "LeViT_small": load_LeViT_small,
        }
        model = variant_function_dict[self.variant]()
        return model, None
