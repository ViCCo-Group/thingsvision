---
layout: default
title: VGG16 with Tensorflow
parent: Examples
nav_order: 4
---


### Example call for VGG16 with TensorFlow:

```python
import torch
from thingsvision import Extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'VGG16'
module_name = 'block1_conv1'
source = 'keras' # TensorFlow backend
batch_size = 64
class_names = None  # optional list of class names for class dataset
file_names = None # optional list of file names according to which features should be sorted

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize extractor module
extractor = Extractor(
  model_name=model_name,
  pretrained=True,
  model_path=None,
  device=device,
  source=source,
)
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
  backend=extractor.backend,
)
features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=False,
  clip=False,
)
save_features(features, out_path='path/to/features', file_format='npy')
```
