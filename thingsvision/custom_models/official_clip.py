from typing import Any

import clip as official_clip

from thingsvision.custom_models.custom import Custom


class clip(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.variant = parameters.get("variant", "ViT-B/32")

    def create_model(self) -> Any:
        model, preprocess = official_clip.load(self.variant, device=self.device)
        return model, preprocess
