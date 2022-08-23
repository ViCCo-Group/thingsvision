import torch
import torchvision.models as torchvision_models

from .custom import Custom


class Resnet50_ecoset(Custom):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.backend = "pt"

    def create_model(self):
        model = torchvision_models.resnet50(pretrained=False, num_classes=565)
        path_to_weights = "https://osf.io/abfq4/download"
        state_dict = torch.hub.load_state_dict_from_url(
            path_to_weights, map_location=self.device, file_name="Resnet50_ecoset"
        )
        model.load_state_dict(state_dict)
        return model
