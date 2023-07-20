from thingsvision.custom_models.custom import Custom
from dreamsim import dreamsim
import torchvision.models as torchvision_models
from torchvision import transforms
import torch
import torch.nn as nn


class DreamSimModel(nn.Module):
    def __init__(self, model_type, device):
        """
        Wrapper class that instantiates the DreamSim model and calls the embed function.
        :param model_type: either clip_vitb32 or open_clip_vitb32
        """
        super().__init__()
        if model_type not in ["clip_vitb32", "open_clip_vitb32"]:
            raise ValueError(f"Model type {model_type} not supported")

        self.model_type = model_type
        self.checkpoint_path = f"./dreamsim_models/dreamsim_{self.model_type}"
        self.model, _ = dreamsim(pretrained=True, dreamsim_type=model_type, normalize_embeds=False)
        self.device = device

    def forward(self, im):
        return self.model.embed(im)


class DreamSim(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = 'pt'
        self.img_size = 224
        self.model_type = parameters.get('model_type', 'open_clip_vitb32')
        self.device = device

        t = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        def preprocess(pil_img):
            pil_img = pil_img.convert('RGB')
            return t(pil_img)

        self.preprocess = preprocess

    def create_model(self):
        return DreamSimModel(self.model_type, self.device), self.preprocess