---
layout: default
title: CLIP/OpenCLIP with Pytorch
parent: Examples
nav_order: 2
---

### Example call for [CLIP](https://github.com/openai/CLIP) with PyTorch:
Note, that the vision model has to be defined in the `model_parameters` dictionary with the `variant` key. You can either use `ViT-B/32` or `RN50`.

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.core.extraction import center_features

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'clip'
module_name = 'visual'
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
  model_parameters={'variant': 'ViT-B/32'},
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
  clip=True,
)
features = center_features(features)
save_features(features, out_path='path/to/features', file_format='npy')
```


### Example call for [Open CLIP](https://github.com/mlfoundations/open_clip) with PyTorch:

Note that the vision model and the dataset that was used for training CLIP have to be defined in the `model_parameters` dictionary `variant` and `dataset` keys. Possible values can be found in the [Open CLIP](https://github.com/mlfoundations/open_clip) pretrained models list.

```python
import torch
from thingsvision import Extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.core.extraction import center_features

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'OpenCLIP'
module_name = 'visual'
source = 'custom'
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
  model_parameters={'variant': 'ViT-H-14', 'dataset': 'laion2b_s32b_b79k'},
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
  clip=True,
)
features = center_features(features)
save_features(features, out_path='path/to/features', file_format='npy')
```
