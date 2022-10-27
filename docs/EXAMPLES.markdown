---
layout: page
title: Examples
permalink: /examples/
---


{% highlight python %}
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
{% endhighlight %}

### Example call for [CLIP](https://github.com/openai/CLIP) with PyTorch:
Note, that the vision model has to be defined in the `model_parameters` dictionary with the `variant` key. You can either use `ViT-B/32` or `RN50`.


{% highlight python %}

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

{% endhighlight %}

### Example call for [Open CLIP](https://github.com/mlfoundations/open_clip) with PyTorch:

Note that the vision model and the dataset that was used for training CLIP have to be defined in the `model_parameters` dictionary `variant` and `dataset` keys. Possible values can be found in the [Open CLIP](https://github.com/mlfoundations/open_clip) pretrained models list.

{% highlight python %}
import torch
from thingsvision import get_extractor
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
extractor = get_extractor(
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
{% endhighlight %}

### Example call for [CORnet](https://github.com/dicarlolab/CORnet) with PyTorch:

{% highlight python %}
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
{% endhighlight %}

### Example call for VGG16 with TensorFlow:

{% highlight python %}
import torch
from thingsvision import get_extractor
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
extractor = get_extractor(
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

{% endhighlight %}
