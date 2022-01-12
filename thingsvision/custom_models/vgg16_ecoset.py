from thingsvision.custom_models.custom import Custom
import torchvision.models as torchvision_models
import torch

class VGG16_ecoset(Custom):
    def __init__(self, device, backend) -> None:
        super().__init__(device, backend)

    def create_model(self):
        if self.backend == 'pt':
            model = torchvision_models.vgg16_bn(pretrained=False, num_classes=565)
            path_to_weights = 'https://osf.io/fe7s5/download'
            state_dict = torch.hub.load_state_dict_from_url(path_to_weights, map_location=self.device, file_name='VGG16_ecoset')
            model.load_state_dict(state_dict)
            return model