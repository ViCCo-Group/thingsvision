import torch
from typing import Any
from .custom import Custom


class Swav(Custom):
    def __init__(self, device, parameters) -> None:
        super(Swav, self).__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        return model, None
