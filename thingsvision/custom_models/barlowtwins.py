import torch

from .custom import Custom


class BarlowTwins(Custom):
    def __init__(self, device) -> None:
        super(BarlowTwins, self).__init__(device)
        self.backend = "pt"

    def create_model(self):
        model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
        return model
