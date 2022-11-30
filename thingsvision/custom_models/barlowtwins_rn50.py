import torch
from typing import Any
from .custom import Custom


class BarlowTwins(Custom):
    def __init__(self, device, parameters) -> None:
        super(BarlowTwins, self).__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
        return model, None
