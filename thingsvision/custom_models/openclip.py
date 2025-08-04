from functools import partial
from typing import Any

import open_clip

from .custom import Custom


class OpenCLIP(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.dataset = parameters.get("dataset", "laion400m_e32")
        self.variant = parameters.get("variant", "ViT-B-32-quickgelu")
        self.vision_cfg = parameters.get("vision_cfg", None)

    def check_available_variants_and_datasets(self):
        found = False
        for variant, dataset in open_clip.list_pretrained():
            if variant == self.variant and dataset == self.dataset:
                found = True
                break
        if not found:
            raise ValueError(
                f"\nCould not find an OpenCLIP model with variant: {self.variant} and dataset: {self.dataset}.\n"
            )

    def create_model(self) -> Any:
        self.check_available_variants_and_datasets()
        kwargs = {
            "model_name": self.variant,
            "pretrained": self.dataset,
        }
        if self.vision_cfg:
            kwargs["vision_cfg"] = self.vision_cfg

        model, _, preprocess = open_clip.create_model_and_transforms(**kwargs)
        return model, preprocess
