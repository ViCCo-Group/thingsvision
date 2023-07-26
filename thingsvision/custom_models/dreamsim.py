from typing import Callable, Tuple
import os
import torch
import torch.nn as nn
from dreamsim import dreamsim
from torchvision import transforms

from thingsvision.custom_models.custom import Custom
from thingsvision.utils.checkpointing import get_torch_home

Tensor = torch.Tensor


class DreamSimModel(nn.Module):
    def __init__(self, model_type, device) -> None:
        """
        Wrapper class that instantiates the DreamSim model and calls the embed function.
        :param model_type: either clip_vitb32 or open_clip_vitb32
        """
        super().__init__()
        if model_type not in ["clip_vitb32", "open_clip_vitb32"]:
            raise ValueError(f"Model type {model_type} not supported")

        self.model_type = model_type
        self.device = device
        model_dir = os.path.join(get_torch_home(), 'dreamsim')
        self.model, _ = dreamsim(
            pretrained=True, dreamsim_type=model_type, normalize_embeds=False,
            device=device, cache_dir=model_dir
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model.embed(x)


class DreamSim(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.img_size = 224
        self.variant = parameters.get("variant", "open_clip_vitb32")
        self.device = device

        t = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size, self.img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )

        def preprocess(pil_img) -> Callable:
            pil_img = pil_img.convert("RGB")
            return t(pil_img)

        self.preprocess = preprocess

    def create_model(self) -> Tuple[nn.Module, Callable]:
        return DreamSimModel(self.variant, self.device), self.preprocess

