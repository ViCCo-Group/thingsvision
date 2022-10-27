---
layout: default
title: Using Custom Models
nav_order: 3
---


## Adding custom models

If you want to use your own model and/or want to make it public, you just need to implement a class inheriting from the `custom_models/custom.py:Custom` class and implement the `create_model` method.
There you can build/download the model and its weights. The constructors expects a `device` (str) and a `kwargs` (dict) where you can put model parameters. The `backend` attribute needs to be set to either `pt` (PyTorch) or `tf` (Tensorflow). The `create_model` method needs to return the model and an optional preprocessing method. If no preprocessing is set, the ImageNet default preprocessing is used. Afterwards you can put the file in the `custom_models` directory and create a pull request to include the model in the official GitHub repository.

```python
from thingsvision.custom_models.custom import Custom
import torchvision.models as torchvision_models
import torch

class VGG16_ecoset(Custom):
    def __init__(self, device, **kwargs) -> None:
        super().__init__(device)
        self.backend = 'pt'
        self.preprocess = None

    def create_model(self):
          model = torchvision_models.vgg16(pretrained=False, num_classes=565)
          path_to_weights = 'https://osf.io/fe7s5/download'
          state_dict = torch.hub.load_state_dict_from_url(path_to_weights, map_location=self.device)
          model.load_state_dict(state_dict)
          return model, self.preprocess
```


## Extract features from custom models

If you want to use a custom model from the `custom_models` directory, you need to use their class name (e.g., `VGG16_ecoset`) as the model name. 

```python
from thingsvision import Extractor
model_name = 'VGG16_ecoset'
source = 'custom'
extractor = Extractor(
  model_name=model_name, 
  pretrained=True, 
  model_path=None, 
  device=device, 
  source=source,
)
```
