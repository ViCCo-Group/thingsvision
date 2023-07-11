from typing import Any

import torch
import torchvision.models as models

from .custom import Custom


class AlexNet_SalObjSub(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self) -> Any:
        model = models.alexnet(weights=None, num_classes=565)
        path_to_weights = "https://osf.io/download/sd3xj/"
        state_dict = torch.hub.load_state_dict_from_url(
            path_to_weights, map_location=self.device, file_name="AlexNet_SalObjSub"
        )
        model.load_state_dict(state_dict)
        return model, None
