import torch
import torchvision.models as models

from typing import Any
from .custom import Custom


class VGG16_ecoset(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = models.vgg16(weights=None, num_classes=565)
        path_to_weights = "https://osf.io/z5uf3/download"
        state_dict = torch.hub.load_state_dict_from_url(
            path_to_weights, map_location=self.device, file_name="VGG16bn_ecoset"
        )
        model.load_state_dict(state_dict)
        return model, None
