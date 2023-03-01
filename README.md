<a name="readme-top"></a>
<div align="center">
    <a href="https://github.com/ViCCo-Group/thingsvision/actions/workflows/tests.yml" rel="nofollow">
        <img src="https://github.com/ViCCo-Group/thingsvision/actions/workflows/tests.yml/badge.svg" alt="Tests" />
    </a>
    <a href="https://github.com/ViCCo-Group/thingsvision/actions/workflows/coverage.yml" rel="nofollow">
        <img src="https://codecov.io/gh/ViCCo-Group/thingsvision/branch/master/graph/badge.svg" alt="Code Coverage" />
    </a>
    <a href="https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d" rel="nofollow">
        <img src="https://img.shields.io/badge/maintenance-yes-brightgreen.svg" alt="Maintenance" />
    </a>
    <a href="https://pypi.org/project/thingsvision/" rel="nofollow">
        <img src="https://img.shields.io/pypi/v/thingsvision" alt="PyPI" />
    </a>
    <a href="https://pepy.tech/project/thingsvision">
        <img alt="Pepy" src="https://pepy.tech/badge/thingsvision">
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

:rotating_light: Note: some function calls mentioned in the [paper](https://www.frontiersin.org/articles/10.3389/fninf.2021.679838/full) have been deprecated. To use this package successfully, exclusively follow this `README` and the [documentation](https://vicco-group.github.io/thingsvision/). :rotating_light:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Functionality -->
### :mechanical_arm: Functionality
With `thingsvision`, you can:
- extract features for any imageset from many popular networks.
- extract features for any imageset from your custom networks.
- extract features for >26,000 images from the [THINGS image database](https://osf.io/jum2f/).
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
- `ssl` (Self-Supervised Learning Models)
  - `simclr-rn50`, `mocov2-rn50`, `jigsaw-rn50`, `rotnet-rn50`, `swav-rn50`, `pirl-rn50` (retrieved from [vissl](https://github.com/facebookresearch/vissl))
  - `barlowtwins-rn50`, `vicreg-rn50`, `dino-vit{s/b}{8/16}`, `dino-xcit-{small/medium}-{12/24}-p{8/16}`, `dino-rn50` (retrieved from [torch.hub](https://pytorch.org/hub/))
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- both original [CLIP](https://github.com/openai/CLIP) variants (`ViT-B/32` and `RN50`)
- a few custom models (Alexnet, VGG-16, Resnet50, and Inception_v3) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118) rather than ImageNet  and one Alexnet pretrained on ImageNet and fine-tuned on [SalObjSub](https://cs-people.bu.edu/jmzhang/sos.html)
- each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions
- [Harmonization](https://arxiv.org/abs/2211.04533) models from the [official repo](https://github.com/serre-lab/harmonization). The default variant is `ViT_B16`. However, the following encoders are additionally available: `ResNet50`, `VGG16`, `EfficientNetB0`, `tiny_ConvNeXT`, `tiny_MaxViT`, `LeViT_small`<br> 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Getting Started -->
## :running: Getting Started

<!-- Setting up your environment -->
### :computer: Setting up your environment
#### Working locally.
First, create a new `conda environment` with Python version 3.8, 3.9, or 3.10 e.g. by using `conda`:

```bash
$ conda create -n thingsvision python=3.9
$ conda activate thingsvision
```

Then, activate the environment and simply install `thingsvision` via running the following `pip` command in your terminal.

```bash
$ pip install --upgrade thingsvision
$ pip install git+https://github.com/openai/CLIP.git
$ pip install git+https://github.com/serre-lab/Harmonization.git
```

#### Google Colab.
Alternatively, you can use Google Colab to play around with `thingsvision` by uploading your image data to Google Drive (via directory mounting).
You can find the jupyter notebook using `PyTorch` [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/pytorch.ipynb) and the `TensorFlow` example [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/tensorflow.ipynb).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Basic usage -->
### :mag: Basic usage

#### Command Line Interface (CLI)

`thingsvision` was designed to simplify feature extraction. If you have some folder of images (e.g., `./images`) and want to extract features for each of these images without opening a Jupyter Notebook instance or writing a Python script, it's probably easiest to use our CLI. The interface includes two options,

- `thingsvision show-model`
- `thingsvision extract-features`

Example calls might look as follows:

```bash
thingsvision show-model --model-name "alexnet" --source "torchvision"
thingsvision extract_features --image-root "./data" --model-name "alexnet" --module-name "features.10" --batch-size 32 --device "cuda" --source "torchvision" --file-format "npy" --out-path "./features"
```

See `thingsvision show-model -h` and `thingsvision extract-features -h` for a list of all possible arguments. Note that the CLI provides just the basic extraction functionalities but is probably enough for most users that don't want to dive too deep into various models and modules. If you need more fine-grained control over the extraction itself, we recommend to use the python package directly and write your own Python script.

#### Python commands

To do this start by importing all the necessary components and instantiating a `thingsvision` extractor. Here we're using `AlexNet` from the `torchvision` library as the model to extract features from and also load the model to GPU for faster inference,

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
    pretrained=True
)
```

As a next step, create both dataset and dataloader for your images. We assume that all of your images are in a single `root` directory which can contain subfolders (e.g., for individual classes). Therefore, we leverage the `ImageDataset` class. 

```python
root='path/to/root/img/directory' # (e.g., './images/)
batch_size = 32

dataset = ImageDataset(
    root=root,
    out_path='path/to/features',
    backend=extractor.get_backend(), # backend framework of model
    transforms=extractor.get_transformations(resize_dim=256, crop_dim=224) # set input dimensionality to whatever is required for your pretrained model
)

batches = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    backend=extractor.get_backend() # backend framework of model
)
```

Now all that is left is to extract the image features and store them to disk! Here we're extracting features from the last convolutional layer of AlexNet (`features.10`), but if you don't know which modules are available for a given model, just call `extractor.show_model()` to print all modules.

```python
module_name = 'features.10'

features = extractor.extract_features(
    batches=batches,
    module_name=module_name,
    flatten_acts=True, # flatten 2D feature maps from convolutional layer
    output_type="ndarray", # or "tensor" (only applicable to PyTorch models)
)

save_features(features, out_path='path/to/features', file_format='npy')
```

_For more examples on the many models available in `thingsvision` and explanations of additional functionality like how to optionally turn off center cropping, how to use HDF5 datasets (e.g. NSD stimuli), how to perform RSA or CKA, or how to easily extract features for the [THINGS image database](https://osf.io/jum2f/), please refer to the [Documentation](https://vicco-group.github.io/thingsvision/)._
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
This library is based on the groundwork laid by [Lukas Muttenthaler](https://lukasmut.github.io/) and [Martin N. Hebart](http://martin-hebart.de/), who are both still actively involved, but has been extended and refined into its current form with the help of our many contributors,

- [Alex Murphy](https://github.com/Alxmrphi) (software dev.)
- [Florian Mahner](https://www.cbs.mpg.de/person/mahner/1483114) (software dev.)
- [Hannes Hansen](https://github.com/hahahannes) (software dev.)
- [Johannes Roth](https://jroth.space/) (software dev., design, docs)
- [Jonas Dippel](https://github.com/jonasd4) (software dev.)
- [Lukas Muttenthaler](https://lukasmut.github.io/) (software dev., design, docs, general responsibility)
- [Martin N. Hebart](http://martin-hebart.de/) (design)
- [Oliver Contier](https://olivercontier.com/) (docs)
- [Philipp Kaniuth](https://www.cbs.mpg.de/person/kaniuth/1483114) (design, docs)
- [Roman Leipe](https://github.com/RLeipe) (sofware dev., docs),

sorted alphabetically. 

This is a joint open-source project between the Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig, and the Machine Learning Group at Technische Universtität Berlin. Correspondence and requests for contributing should be adressed to [Lukas Muttenthaler](https://lukasmut.github.io/). Feel free to contact us if you want to become a contributor or have any suggestions/feedback. For the latter, you could also just post an issue or engange in discussions. We'll try to respond as fast as we can.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

