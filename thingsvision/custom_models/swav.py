import torch

from .custom import Custom


class Swav(Custom):
    def __init__(self, device) -> None:
        super(Swav, self).__init__(device)
        self.backend = "pt"

    def create_model(self):
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        return model
