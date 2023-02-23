---
layout: default
title: Using your own models
nav_order: 5
---

# Using your own models

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

## Use custom models with the `get_extractor_from_model` function
Alternatively, you can use the `get_extractor_from_model()` helper function to directly create an extractor for any PyTorch or TensorFlow model, without the need for creating a custom class. Simply pass the model, the device, the backend (either `pt` or `tf`) and optionally a preprocessing function to the `get_extractor_from_model` function. The function will return an extractor that can be used with the `extract` method, just like the built-in extractors.

```python
from thingsvision import get_extractor_from_model
from torchvision.models import alexnet, AlexNet_Weights
import torch

# initialize a model of your choice, here we use AlexNet from torchvision 
# and load the ImageNet weights
model_weights = AlexNet_Weights.DEFAULT
model = alexnet(model_weights)

# you can also pass a custom preprocessing function that is applied to every 
# image before extraction
transforms = model_weights.transforms()

# provide the backend of the model (either 'pt' or 'tf')
backend = 'pt'

# set the device to 'cuda' if you have a GPU available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor_from_model(
  model=model, 
  device=device,
  transforms=transforms,
  backend=backend
)
```

### Custom forward and flatten functions for PyTorch models
The created extractor will pass image batches to the model's forward function (`self.model(batch)`) and optionally flatten the activations (`act.view(act.size(0), -1)`). Your models (e.g., a CLIP variant) forward function may have a different signature or activations in your model require a different way of flattening. In that case, you can provide custom forward and flatten functions to the `get_extractor_from_model` function to enable the extractor to work with your model. These will be used instead of the default ones. You can pass them as `forward_fn` and `flatten_fn` arguments to the `get_extractor_from_model` function.

```python
from thingsvision import get_extractor_from_model
import torch.nn as nn

class ComplexForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # model definition 
        # ...

    def forward(self, x, y):
        # forward function with custom signature
        # ...
        return x

def custom_forward_fn(self, batch):
    # custom forward function that passes the batch to the model's forward function
    return self.model(batch, y=None)

model = ComplexForwardModel()
backend = 'pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor_from_model(
  model=model, 
  device=device,
  transforms=transforms,
  backend=backend,
  forward_fn=custom_forward_fn
)
```

Note that the custom forward function needs to have the following signature: `forward_fn(self, batch)`, as it replaces the default `forward_fn` function of the extractor. The `self` argument is the extractor instance and the `batch` argument is the batch of images that is passed to the model's forward function. 