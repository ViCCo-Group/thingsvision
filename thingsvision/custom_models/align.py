from typing import Any
from .custom import Custom
from transformers import AlignModel, AutoProcessor


class Kakaobrain_Align(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = AlignModel.from_pretrained("kakaobrain/align-base")
        processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        def preprocess_fn(images):
            out = processor(images=images, return_tensors="pt")
            return out['pixel_values'].squeeze(0)

        return model.vision_model, preprocess_fn
