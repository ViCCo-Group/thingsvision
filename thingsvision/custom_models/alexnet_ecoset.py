from thingsvision.custom_models.custom import Custom
import torchvision.models as torchvision_models
import tensorflow.keras.applications as tensorflow_models
import torch

class Alexnet_ecoset(Custom):
    def __init__(self, device, backend) -> None:
        super().__init__(device, backend)

    def create_model(self):
        if self.backend == 'pt':
            model = torchvision_models.alexnet(pretrained=False, num_classes=565)
            path_to_weights = 'https://osf.io/xtdf4/download'
            state_dict = torch.hub.load_state_dict_from_url(path_to_weights, map_location=self.device, file_name='Alexnet_ecoset')
            model.load_state_dict(state_dict)
            return model