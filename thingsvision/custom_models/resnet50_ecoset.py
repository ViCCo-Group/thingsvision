from typing import Any

import torch
import torchvision.models as models

from .custom import Custom


class Resnet50_ecoset(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = models.resnet50(weights=None, num_classes=565)
        path_to_weights = "https://osf.io/gd9kn/download"
        state_dict = torch.hub.load_state_dict_from_url(
            path_to_weights, map_location=self.device, file_name="Resnet50_ecoset"
        )
        model.load_state_dict(state_dict)
        return model, None
