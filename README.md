<div align="center">
    <a href="https://github.com/ViCCo-Group/THINGSvision/actions/workflows/tests.yml" rel="nofollow">
        <img src="https://github.com/ViCCo-Group/THINGSvision/actions/workflows/tests.yml/badge.svg" alt="Tests" />
    </a>
    <a href="https://github.com/ViCCo-Group/THINGSvision/actions/workflows/coverage.yml" rel="nofollow">
        <img src="https://codecov.io/gh/ViCCo-Group/THINGSvision/branch/master/graph/badge.svg" alt="Code Coverage" />
    </a>
    <a href="https://pypi.org/project/thingsvision/" rel="nofollow">
        <img src="https://img.shields.io/pypi/v/thingsvision" alt="PyPI" />
    </a>
    <a href="https://www.python.org/" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg" alt="Python version" />
    </a>
    <a href="https://github.com/ViCCo-Group/THINGSvision/blob/master/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/pypi/l/thingsvision" alt="License" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
    <a href="https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/pytorch.ipynb" rel="nofollow">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
    </a>
</div>


## Model collection

Features can be extracted for all models in [torchvision](https://pytorch.org/vision/0.8/models.html), [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications), [timm](https://github.com/rwightman/pytorch-image-models), custom models (VGG-16, Resnet50, Inception_v3 and Alexnet) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118), each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions and both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`).<br> 


Note that you have to use the respective model name (`str`). For example, if you want to use VGG16 from torchvision, use `vgg16` as the model name and if you want to use VGG16 from TensorFlow/Keras, use the model name `VGG16`. You can further specify the model source by setting the `source` parameter (e.g., `timm`, `torchvision`, `keras`).<br>


For the correct abbreviations of [torchvision](https://pytorch.org/vision/0.8/models.html) models have a look [here](https://github.com/pytorch/vision/tree/master/torchvision/models). For the correct abbreviations of [CORnet](https://github.com/dicarlolab/CORnet) models look [here](https://github.com/dicarlolab/CORnet/tree/master/cornet). To separate the string `cornet` from its variant (e.g., `s`, `z`) use a hyphen instead of an underscore (e.g., `cornet-s`, `cornet-z`).<br>

PyTorch examples:  `alexnet`, `resnet18`, `resnet50`, `resnet101`, `vit_b_16`, `vit_b_32`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`, `cornet-s`, `clip-ViT`

## Environment Setup

We recommend to create a new `conda environment` with Python version 3.7, 3.8, or 3.9 before using `thingsvision`. Check out the `environment.yml` file in `envs`, if you want to create a `conda environment` via `yml`. Activate the `environment` and run the following `pip` command in your terminal.

```bash
$ pip install --upgrade thingsvision
```

You have to download files from the parent folder of this repository, if you want to extract network activations for [THINGS](https://osf.io/jum2f/). Simply download the shell script `get_files.sh` from this repo and execute it as follows (the shell script will automatically do file downloading and moving for you):

```bash
$ wget https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (Linux)
$ curl -O https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (macOS)
$ bash get_files.sh
```

## Google Colab

Alternatively, you can use Google Colab to play around with `thingsvision` by uploading your image data to Google Drive.
You can find the jupyter notebook using `PyTorch` [here](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/pytorch.ipynb) and the `TensorFlow` example [here](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/tensorflow.ipynb).

## IMPORTANT NOTES:

1. There exist four different sources from which neural network models and their (pretrained) weights can be downloaded. You can define the source of a model using the `source` argument. Possible sources are [`torchvision`](https://pytorch.org/vision/stable/models.html), [`keras`](https://keras.io/api/applications/), [`timm`](https://github.com/rwightman/pytorch-image-models), and `custom` (e.g., `source = torchvision`).

2. If you happen to use the [THINGS image database](https://osf.io/jum2f/), make sure to correctly `unzip` all zip files (sorted from A-Z), and have all `object` directories stored in the parent directory `./images/` (e.g., `./images/object_xy/`) as well as the `things_concepts.tsv` file stored in the `./data/` folder. `bash get_files.sh` does the latter for you. Images, however, must be downloaded from the [THINGS database](https://osf.io/jum2f/) `Main` subfolder.  **The download is around 5GB**.

*   Go to <https://osf.io/jum2f/files/>
*   Select `Main` folder and click on "Download as zip" button (top right).
*   Unzip contained `object_images_*.zip` file using the password (check the
    `description.txt` file for details). For example:

    ``` {.bash}
    for fn in object_images_*.zip; do unzip -P the_password $fn; done
    ```

3. Features can be extracted for every layer for all `timm`, `torchvision`, `TensorFlow`, `CORnet` and `CLIP`/`OpenCLIP` models.

4. The script automatically extracts features for the specified `model` and `module`.

5. If you happen to extract hidden unit activations for many images, it is possible to run into `MemoryErrors`. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 

## Feature extraction

### Extract features for a specific layer/module of a state-of-the-art `torchvision`, `timm`, `TensorFlow`, `CORnet`, or `CLIP` model

The following examples demonstrate how to load a model with PyTorch or TensorFlow/Keras and how to subsequently extract features. 
Please keep in mind that the model names as well as the layer names depend on the backend you want to use. If you use PyTorch, you will need to use these [model names](https://pytorch.org/vision/stable/models.html). If you use Tensorflow, you will need to use these [model names](https://keras.io/api/applications/). You can find the layer names by using `extractor.show_model()`.

### Example call for AlexNet with PyTorch:

```python
import torch
from thingsvision import Extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'alexnet'
source = 'torchvision'
batch_size = 64
class_names = None  # optional list of class names for class dataset
file_names = None # optional list of file names according to which features should be sorted

device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = Extractor(
  model_name=model_name,
  pretrained=True,
  model_path=None, 
  device=device, 
  source=source,
)
module_name = extractor.show_model()

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

#Enter part of the model for which you would like to extract features:

(e.g., "features.10")

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

### Example call for [CLIP](https://github.com/openai/CLIP) with PyTorch:
Note, that the vision model has to be defined in the `model_parameters` dictionary with the `variant` key. You can either use `ViT-B/32` or `RN50`.

```python
import torch
from thingsvision import Extractor
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
extractor = Extractor(
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

### Example call for [CORnet](https://github.com/dicarlolab/CORnet) with PyTorch:

```python
import torch
from thingsvision import Extractor
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
extractor = Extractor(
  model_name=model_name,
  pretrained=True,
  model_path=None,
  device=device,
  source=source,
)
module_name = extractor.show_model()

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
(e.g., "decoder.flatten")

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

### Optional Center Cropping

Center cropping is used by default but can be deactivated by turning off the `apply_center_crop` argument of the `get_transformations` method.

```python
root = 'path/to/images'
apply_center_crop = False
dataset = ImageDataset(
  root=root,
  out_path='path/to/features',
  backend=extractor.backend,
  transforms=model.get_transformations(apply_center_crop=apply_center_crop),
  class_names=class_names,
  file_names=file_names,
)
```

## Using HDF5 datasets (e.g. NSD stimuli)

You can also extract features for images stored in HDF5 dataset. For this you can simply replace `ImageDataset` with `HDF5Dataset`, providing the path to the HDF5 file as `hdf5_fp` and the name of the dataset containing the images as `img_ds_key`. 

Optionally, you can specify which images to extract features for by providing a list of indices as `img_indices`, otherwise features for all images will be extracted. 

The following example demonstrates how to extract features for images corresponding to the NSD stimuli dataset shown to subject 1:

```python
from thingsvision.utils.data import HDF5Dataset

# get indices of all 10000 images shown to first subject
img_indices = np.unique(
    experiment['subjectim'][:, experiment['masterordering'][0] - 1][0]
)

dataset = HDF5Dataset(
    hdf5_fp="<path_to_nsd>/nsddata_stimuli/stimuli/nsd_stimuli.hdf5",
    img_ds_key="imgBrick",
    transforms=extractor.get_transformations(),
    backend=extractor.backend,
    img_indices=img_indices
)
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

## Representational Similarity Analysis (RSA) 

Compare representational (dis-)similarity matrices (RDMs) corresponding to model features and human representations (e.g., fMRI recordings).

```python
from thingsvision.core.rsa import compute_rdm, correlate_rdms

rdm_dnn = compute_rdm(features, method='correlation')
corr_coeff = correlate_rdms(rdm_dnn, rdm_human, correlation='pearson')
```

## Centered Kernel Alignment (CKA)

Perform CKA to compare image features of two different model architectures for the same layer, or two different layers of the same architecture.

```python
from thingsvision.core.cka import CKA

m = # number of images (e.g., features_i.shape[0])
kernel = 'linear'
cka = CKA(m=m, kernel=kernel)
rho = cka.compare(X=features_i, Y=features_j)
```

## Citation

If you use this GitHub repository (or any modules associated with it), we would grately appreciate to cite our [paper](https://www.frontiersin.org/articles/10.3389/fninf.2021.679838/full) as follows:

```latex
@article{Muttenthaler_2021,
	author = {Muttenthaler, Lukas and Hebart, Martin N.},
	title = {THINGSvision: A Python Toolbox for Streamlining the Extraction of Activations From Deep Neural Networks},
	journal ={Frontiers in Neuroinformatics},
	volume = {15},
	pages = {45},
	year = {2021},
	url = {https://www.frontiersin.org/article/10.3389/fninf.2021.679838},
	doi = {10.3389/fninf.2021.679838},
	issn = {1662-5196},
}
```
