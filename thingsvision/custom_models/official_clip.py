
from torchvision import transforms as T

from typing import Any
from thingsvision.custom_models.custom import Custom
import clip as official_clip

class clip(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.variant = parameters.get('variant', 'ViT-B/32')

    def create_model(self) -> Any:
        model, preprocess = official_clip.load(self.variant, device=self.device)
        return model, preprocess
        

        
