---
layout: default
title: CORnet with Pytorch
parent: Examples
nav_order: 3
---


### Example call for [CORnet](https://github.com/dicarlolab/CORnet) with PyTorch:

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'cornet-s'
source = 'custom'
batch_size = 64
class_names = None  # optional list of class names for class dataset
file_names = None # optional list of file names according to which features should be sorted

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize extractor module
extractor = get_extractor(
  model_name=model_name,
  pretrained=True,
  model_path=None,
  device=device,
  source=source,
)
extractor.show_model()

Sequential(
  (V1): Sequential(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (nonlin1): ReLU(inplace=True)
    (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (nonlin2): ReLU(inplace=True)
    (output): Identity()
  )
  ...
  (decoder): Sequential(
    (avgpool): AdaptiveAvgPool2d(output_size=1)
    (flatten): Flatten()
    (linear): Linear(in_features=512, out_features=1000, bias=True)
    (output): Identity()
  )
)

# enter part of the model for which you would like to extract features (e.g., penultimate layer)
module_name = "decoder.flatten"

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
  flatten_acts=False,
  clip=False,
)
save_features(features, out_path='path/to/features', file_format='npy')
```