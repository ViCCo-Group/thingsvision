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
  * [Functionality](#mechanical_arm-functionality)
  * [Model collection](#file_cabinet-model-collection)
- [Getting Started](#running-getting-started)
  * [Setting up your environment](#computer-setting-up-your-environment)
  * [Basic usage](#mag-basic-usage)
- [Contributing](#wave-how-to-contribute)
- [License](#warning-license)
- [Citation](#page_with_curl-citation)
- [Contributions](#gem-contributions)


<!-- About the Project -->
## :star2: About the Project
`thingsvision` is a Python package that let's you easily extract image representations from many state-of-the-art neural networks for computer vision. In a nutshell, you feed `thingsvision` with a directory of images and tell it which neural network you are interested in. `thingsvision` will then give you the  representation of the indicated neural network for each image so that you will end up with one feature vector per image. You can use these feature vectors for further analyses. We use the word `features` for short when we mean "image representation".

:rotating_light: Note: some function calls mentioned in the [paper](https://www.frontiersin.org/articles/10.3389/fninf.2021.679838/full) have been deprecated. To use this package successfully, exclusively follow this `README` and the [Documentation](https://vicco-group.github.io/thingsvision/). :rotating_light:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Functionality -->
### :mechanical_arm: Functionality
With `thingsvision`, you can:
- extract features for any imageset from many popular networks.
- extract features for any imageset from your custom networks.
- extract features for the [THINGS image database](https://osf.io/jum2f/).
- optionally turn off the standard center cropping performed by many networks before extracting features.
- extract features from HDF5 datasets directly (e.g. NSD stimuli)
- conduct basic Representational Similarity Analysis (RSA) after feature extraction.
- perform Centered Kernel Alignment (CKA) to compare image features across model-module combinations.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Model collection -->
### :file_cabinet: Model collection
Neural networks come from different sources. With `thingsvision`, you can extract image representations of all models from:
- [torchvision](https://pytorch.org/vision/0.8/models.html)
- [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [vissl](https://github.com/facebookresearch/vissl)
- some custom models (VGG-16, Resnet50, Inception_v3 and Alexnet) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118)
- each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions
- both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`).<br> 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Getting Started -->
## :running: Getting Started

<!-- Setting up your environment -->
### :computer: Setting up your environment
#### Working locally.
First, create a new `conda environment` with Python version 3.7, 3.8, or 3.9, e.g. by using `conda` and the [`environment.yml` file](https://github.com/ViCCo-Group/thingsvision/blob/master/envs/environment.yml) like so

```bash
$ conda env create --prefix /path/to/conda/envs/thingsvision --file envs/environment.yml
$ conda activate thingsvision
```

Then, activate the environment and simply install `thingsvision` via running the following `pip` command in your terminal.

```bash
$ pip install --upgrade thingsvision
```

#### Google Colab.
Alternatively, you can use Google Colab to play around with `thingsvision` by uploading your image data to Google Drive (via directory mounting).
You can find the jupyter notebook using `PyTorch` [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/pytorch.ipynb) and the `TensorFlow` example [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/tensorflow.ipynb).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Basic usage -->
### :mag: Basic usage
`thingsvision` was designed to make extracting features as easy as possible. Start by importing all the necessary components and instantiating a thingsvision extractor. Here we're using `AlexNet` from the `torchvision` library as the model to extract features from and also load the model to GPU for faster inference:

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

model_name = 'alexnet'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device, 
)
```

Next, create the Dataset and Dataloader for your images. Here, we have all our images in a single directory `root`, which can also contain subfolders (e.g. for individual classes), so we're using the `ImageDataset` class. 

```python
root='path/to/root/img/directory' # (e.g., './images/)
batch_size = 32

dataset = ImageDataset(
  root=root,
  out_path='path/to/features',
  backend=extractor.get_backend(),
  transforms=extractor.get_transformations(),
)

batches = DataLoader(
  dataset=dataset,
  batch_size=batch_size, 
  backend=extractor.get_backend()
)
```

Now all that is left is to extract the image features and store them to disk! Here we're extracting features from the last convolutional layer of AlexNet (`features.10`), but if you don't know which modules are available for a given model, just call `extractor.show_model()` to print all modules.

```python
module_name = 'features.10'

features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True,  # flatten 2D feature maps from convolutional layer 
)

save_features(features, out_path='path/to/features', file_format='npy')
```

_For more examples and explanations of additional functionality like how to optionally turn off center cropping, how to use HDF5 datasets (e.g. NSD stimuli), how to perform RSA or CKA, or how to easily extract features for the [THINGS image database](https://osf.io/jum2f/), please refer to the [Documentation](https://vicco-group.github.io/thingsvision/)._
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
If you use this GitHub repository (or any modules associated with it), please cite our [paper](https://www.frontiersin.org/articles/10.3389/fninf.2021.679838/full) for the initial version of `thingsvision` as follows:

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


<!-- Contributions -->
## :gem: Contributions
Here, we list all the people that have made and still make this project possible. We only list contributors that have made substantial contributions without which core functionalities would not work and who are still actively involved. Contributors are listed in descending order according to their quality $\cup$ quantity of contributions.


- Conception / Design: [Lukas Muttenthaler](https://lukasmut.github.io/), [Martin N. Hebart](http://martin-hebart.de/), [Johannes Roth](https://jroth.space/), [Philipp Kaniuth](https://www.cbs.mpg.de/person/kaniuth/1483114)
- Software Dev.: [Lukas Muttenthaler](https://lukasmut.github.io/), [Johannes Roth](https://jroth.space/), [Hannes Hansen](https://github.com/hahahannes), [Jonas Dippel](https://github.com/jonasd4), [Alex Murphy](https://github.com/Alxmrphi), [Roman Leipe](https://github.com/RLeipe)
- Docs / Website: [Roman Leipe](https://github.com/RLeipe), [Philipp Kaniuth](https://www.cbs.mpg.de/person/kaniuth/1483114), [Lukas Muttenthaler](https://lukasmut.github.io/), [Johannes Roth](https://jroth.space/)
- README: [Philipp Kaniuth](https://www.cbs.mpg.de/person/kaniuth/1483114), [Lukas Muttenthaler](https://lukasmut.github.io/), [Johannes Roth](https://jroth.space/), [Oliver Contier](https://olivercontier.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

