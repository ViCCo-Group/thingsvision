---
layout: default
title: AlexNet with PyTorch
parent: Examples
nav_order: 1
---


### Example call for AlexNet with PyTorch:

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'alexnet'
source = 'torchvision'
batch_size = 64
class_names = None  # optional list of class names for class dataset
file_names = None # optional list of file names according to which features should be sorted

device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  pretrained=True,
  model_path=None, 
  device=device, 
  source=source,
)
extractor.show_model()

AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

# enter part of the model for which you would like to extract features
module_name = "features.10"

dataset = ImageDataset(
  root=root,
  out_path='path/to/features',
  backend=extractor.backend,
  transforms=extractor.get_transformations(),
  class_names=class_names,
  file_names=file_names,
)
batches = DataLoader(
  dataset=dataset,
  batch_size=batch_size, 
  backend=extractor.backend
)
features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True,
  clip=False,
)
save_features(features, out_path='path/to/features', file_format='npy')
```