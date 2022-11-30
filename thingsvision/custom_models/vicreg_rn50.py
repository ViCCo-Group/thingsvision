import torch
from typing import Any
from .custom import Custom


class Vicreg(Custom):
    def __init__(self, device, parameters) -> None:
        super(Vicreg, self).__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
        return model, None
