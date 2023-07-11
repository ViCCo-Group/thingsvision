from typing import Any

import torch
import torchvision.models as models

from .custom import Custom


class Alexnet_ecoset(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = models.alexnet(weights=None, num_classes=565)
        path_to_weights = "https://osf.io/t6h3c/download"
        state_dict = torch.hub.load_state_dict_from_url(
            path_to_weights, map_location=self.device, file_name="Alexnet_ecoset"
        )
        model.load_state_dict(state_dict)
        return model, None
