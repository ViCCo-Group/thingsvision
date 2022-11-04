<a name="readme-top"></a>
<div align="center">
    <a href="https://github.com/ViCCo-Group/thingsvision/actions/workflows/tests.yml" rel="nofollow">
        <img src="https://github.com/ViCCo-Group/thingsvision/actions/workflows/tests.yml/badge.svg" alt="Tests" />
    </a>
    <a href="https://github.com/ViCCo-Group/thingsvision/actions/workflows/coverage.yml" rel="nofollow">
        <img src="https://codecov.io/gh/ViCCo-Group/thingsvision/branch/master/graph/badge.svg" alt="Code Coverage" />
    </a>
    <a href="https://pypi.org/project/thingsvision/" rel="nofollow">
        <img src="https://img.shields.io/pypi/v/thingsvision" alt="PyPI" />
    </a>
    <a href="https://www.python.org/" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg" alt="Python version" />
    </a>
    <a href="https://github.com/ViCCo-Group/thingsvision/blob/master/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/pypi/l/thingsvision" alt="License" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
    <a href="https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/pytorch.ipynb" rel="nofollow">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
    </a>
</div>


<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
  * [Model collection](#file_cabinet-model-collection)
  * [Functionality](#mechanical_arm-functionality)
- [Getting Started](#running-getting-started)
- [Basic usage](#computer-basic-usage)
- [Contributing](#wave-how-to-contribute)
- [License](#warning-license)
- [Citation](#page_with_curl-citation)
- [Acknowledgements](#gem-acknowledgements)


<!-- About the Project -->
## :star2: About the Project
`thingsvision` is a Python package that let's you easily extract image representations from many state-of-the-art neural networks for computer vision. In a nutshell, you feed `thingsvision` with a bunch of images and tell it which neural network you are interested in. `thingsvision` will then give you the  representation of the indicated neural network for each image so that you will end up with one feature vector per image. You can use these feature vectors for further analyses. We use the word `features` for short when we mean "image representation".
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Model collection -->
### :file_cabinet: Model collection
Neural networks come from different sources. With `thingsvision`, you can extract image representations of all models from:
- [torchvision](https://pytorch.org/vision/0.8/models.html)
- [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [timm](https://github.com/rwightman/pytorch-image-models)
- some custom models (VGG-16, Resnet50, Inception_v3 and Alexnet) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118)
- each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions
- both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`).<br> 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Functionality -->
### :mechanical_arm: Functionality
With `thingsvision`, you can:
- extract features for any imageset from many popular networks.
- extract features for any imageset from your custom networks.
- extract features for the [THINGS image database](https://osf.io/jum2f/).
- optionally turn off the standard center cropping performed by many networks before extracting features.
- use HDF5 datasets.
- conduct basic Representational Similarity Analysis (RSA) after feature extraction.
- perform Centered Kernel Alignment (CKA) to compare image features across model-module combinations.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Getting Started -->
## :running: Getting Started

#### Working locally.
First, create a new `conda environment` with Python version 3.7, 3.8, or 3.9, e.g. by using `conda` and the [`environment.yml` file](https://github.com/ViCCo-Group/thingsvision/blob/master/envs/environment.yml). Then, activate the environment and simply install `thingsvision` via running the following `pip` command in your terminal.

#### Google Colab.
Alternatively, you can use Google Colab to play around with `thingsvision` by uploading your image data to Google Drive.
You can find the jupyter notebook using `PyTorch` [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/pytorch.ipynb) and the `TensorFlow` example [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/tensorflow.ipynb).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Basic usage -->
## :computer: Basic usage
The basic usage of `thingsvision` is straightforward. There are just a handful of variables you have to declare to begin: <br>
- `root` is the path to the directory that holds the images you want to extract the features for.<br>
- `source` indicates from which source you want to use a network  (e.g., `torchvision`, `keras`, `timm`, `custom`.).<br>
- `model_name` denotes the specific network architecture you want to use. Sometimes, the same network architecture is available from different sources, though. Therefore, within `thingsvision`, you have to use the source-specific model name when declaring the model. For torchvision's abbreviations, look [here](https://github.com/pytorch/vision/tree/master/torchvision/models). For CORnet's abbreviations, look [here](https://github.com/dicarlolab/CORnet/tree/master/cornet). To separate the string `cornet` from its variant (e.g., `s`, `z`) use a hyphen instead of an underscore (e.g., `cornet-s`).
- `batch_size` indicates how many images in parallel `thingsvision` shall process. The higher, the more RAM of your machine is used (and it might appear unresponsive). The lower, the longer it takes to extract all features. A good default value is 64.
- `class_names` is an optional list of class names for class dataset.
- `file_names` is an optional list of file names according to which features should be sorted.
- `module_name` denotes for which specific module of your chosen network architecture you want to extract features. You can see all module names of your chosen network by executing `extractor.show_model()` (see example below).


### Example call for AlexNet with PyTorch:
The following examples demonstrate how to load the model AlexNet with PyTorch and how to subsequently extract features. 

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

root='path/to/root/img/directory' # (e.g., './images/)
model_name = 'alexnet'
source = 'torchvision'
batch_size = 64
class_names = None
file_names = None

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

_For more examples and explanations of additional functionality like how to optionally turn off center cropping, how to use HDF5 datasets, how to perform RSA or CKA, or how to easily extract features for the [THINGS image database](https://osf.io/jum2f/), please refer to the [Documentation](https://vicco-group.github.io/thingsvision/)._
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Contributing -->
## :wave: How to contribute
If you come across problems or have suggestions please submit an issue!
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- License -->
## :warning: License
This GitHub repository is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Citation -->
## :page_with_curl: Citation
If you use this GitHub repository (or any modules associated with it), please cite our [paper](https://www.frontiersin.org/articles/10.3389/fninf.2021.679838/full) as follows:

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
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Acknowledgements -->
## :gem: Acknowledgements
- mention useful resources and libraries.
- mention collaborators.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

