from torchvision import transforms as T

from typing import Any
from thingsvision.custom_models.custom import Custom
from .clip_utils import load

class clip(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.variant = parameters.get('variant', 'RN50')
        self.model_path = parameters.get('model_path')
        self.pretrained = parameters.get('pretrained')

    def get_preprocess(self):
        composes = [
            T.Resize(self.clip_n_px, interpolation=T.InterpolationMode.BICUBIC)
        ]

        #if apply_center_crop:
        composes.append(T.CenterCrop(self.clip_n_px))

        composes += [
                lambda image: image.convert("RGB"),
                T.ToTensor(),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
        ]

        return T.Compose(composes)

    def create_model(self) -> Any:
        model, self.clip_n_px = load(
                self.variant,
                device=self.device,
                model_path=self.model_path,
                pretrained=self.pretrained,
                jit=False,
        )

        return model, self.get_preprocess()

        
