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
        <img src="https://img.shields.io/pypi/dm/thingsvision" alt="downloads">
    </a>
    <a href="https://www.python.org/" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg" alt="Python version" />
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
`thingsvision` is a Python package for extracting (image) representations from many state-of-the-art computer vision models. Essentially, you provide `thingsvision` with a directory of images and specify the neural network you're interested in. Subsequently, `thingsvision` returns the representation of the selected neural network for each image, resulting in one feature map (vector or matrix, depending on the layer) per image. These features, used interchangeably with _image representations_, can then be used for further analyses.

:rotating_light: NOTE: some function calls mentioned in the original [paper](https://www.frontiersin.org/articles/10.3389/fninf.2021.679838/full) have been deprecated. To use this package successfully, exclusively follow this `README` and the [documentation](https://vicco-group.github.io/thingsvision/)! :rotating_light:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Functionality -->
### :mechanical_arm: Functionality
With `thingsvision`, you can:
- extract features for any imageset from many popular networks.
- extract features for any imageset from your custom networks.
- extract features for >26,000 images from the [THINGS image database](https://osf.io/jum2f/).
- [align](https://vicco-group.github.io/thingsvision/Alignment.html) the extracted features with human object perception (e.g., using [gLocal](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)).
- extract features from [HDF5 datasets](https://vicco-group.github.io/thingsvision/LoadingYourData.html#using-the-hdf5dataset-class) directly (e.g., [NSD stimuli](https://naturalscenesdataset.org/))
- conduct basic [Representational Similarity Analysis (RSA)](https://vicco-group.github.io/thingsvision/RSA.html#representational-similarity-analysis-rsa) after feature extraction.
- perform efficient [Centered Kernel Alignment (CKA)](https://vicco-group.github.io/thingsvision/RSA.html#centered-kernel-alignment-cka) to compare image features across model-module combinations.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Model collection -->
### :file_cabinet: Model collection
Neural networks come from different sources. With `thingsvision`, you can extract image representations of all models from:
- [torchvision](https://pytorch.org/vision/0.8/models.html)
- [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [timm](https://github.com/rwightman/pytorch-image-models)
- `ssl` (self-supervised learning models)
  - `simclr-rn50`, `mocov2-rn50`, `barlowtwins-rn50`, `pirl-rn50`
  - `jigsaw-rn50`, `rotnet-rn50`, `swav-rn50`, `vicreg-rn50`
  - `dino-rn50`, `dino-xcit-{small/medium}-{12/24}-p{8/16}`
  - `dino-vit-{tiny/small/base}-p{8/16}`
  - `dinov2-vit-{small/base/large/giant}-p14`
  - `mae-vit-{base/large}-p16`, `mae-vit-huge-p14`<br>
- [OpenCLIP](https://github.com/mlfoundations/open_clip) models (CLIP trained on LAION-{400M/2B/5B})
- [CLIP](https://github.com/openai/CLIP) models (CLIP trained on WiT)
- a few custom models (Alexnet, VGG-16, Resnet50, and Inception_v3) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118) rather than ImageNet and one Alexnet model pretrained on ImageNet and fine-tuned on [SalObjSub](https://cs-people.bu.edu/jmzhang/sos.html)<br>
- each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions (recurrent vision models)
- [Harmonization](https://arxiv.org/abs/2211.04533) models (see [Harmonization repo](https://github.com/serre-lab/harmonization)). The default variant is `ViT_B16`. Other available models are `ResNet50`, `VGG16`, `EfficientNetB0`, `tiny_ConvNeXT`, `tiny_MaxViT`, and `LeViT_small`<br> 
- [DreamSim](https://dreamsim-nights.github.io/) models  (see [DreamSim repo](https://github.com/ssundaram21/dreamsim)). The default variant is `open_clip_vitb32`. Other available models are `clip_vitb32`, `dino_vitb16`, and an `ensemble`. See the [docs](https://vicco-group.github.io/thingsvision/AvailableModels.html#dreamsim) for more information

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Getting Started -->
## :running: Getting Started

<!-- Setting up your environment -->
### :computer: Setting up your environment
#### Working locally
First, create a new `conda environment` with Python version 3.8, 3.9, or 3.10 e.g. by using `conda`:

```bash
$ conda create -n thingsvision python=3.9
$ conda activate thingsvision
```

Then, activate the environment and simply install `thingsvision` via running the following `pip` command in your terminal.

```bash
$ pip install --upgrade thingsvision
$ pip install git+https://github.com/openai/CLIP.git
```

If you want to extract features for [harmonized models](https://vicco-group.github.io/thingsvision/AvailableModels.html#harmonization) from the [Harmonization repo](https://github.com/serre-lab/harmonization), you have to additionally run the following `pip` command in your `thingsvision` environment (FYI: as of now, this seems to be working smoothly on Ubuntu only but not on macOS),

```bash
$ pip install git+https://github.com/serre-lab/Harmonization.git
$ pip install keras-cv-attention-models>=1.3.5
```

If you want to extract features for [DreamSim](https://dreamsim-nights.github.io/) from the [DreamSim repo](https://github.com/ssundaram21/dreamsim), you have to additionally run the following `pip` command in your `thingsvision` environment,

```bash
$ pip install dreamsim==0.1.2
```

See the [docs](https://vicco-group.github.io/thingsvision/AvailableModels.html#dreamsim) for which `DreamSim` models are available in `thingsvision`.

#### Google Colab
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
thingsvision extract-features --image-root "./data" --model-name "alexnet" --module-name "features.10" --batch-size 32 --device "cuda" --source "torchvision" --file-format "npy" --out-path "./features"
```

See `thingsvision show-model -h` and `thingsvision extract-features -h` for a list of all possible arguments. Note that the CLI provides just the basic extraction functionalities but is probably enough for most users that don't want to dive too deep into various models and modules. If you need more fine-grained control over the extraction itself, we recommend to use the python package directly and write your own Python script.

#### Python commands

To do this start by importing all the necessary components and instantiating a `thingsvision` extractor. Here we're using `CLIP` from the official clip repo as the model to extract features from and also load the model to GPU for faster inference,

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

model_name = 'clip'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {
    'variant': 'ViT-L/14'
}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters,
)
```

As a next step, create both dataset and dataloader for your images. We assume that all of your images are in a single `root` directory which can contain subfolders (e.g., for individual classes). Therefore, we leverage the `ImageDataset` class. 

```python
root='path/to/your/image/directory' # (e.g., './images/)
batch_size = 32

dataset = ImageDataset(
    root=root,
    out_path='path/to/features',
    backend=extractor.get_backend(), # backend framework of model
    transforms=extractor.get_transformations(resize_dim=256, crop_dim=224) # set the input dimensionality to whichever values are required for your pretrained model
)

batches = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    backend=extractor.get_backend() # backend framework of model
)
```

Now all that is left is to extract the image features and store them on disk! Here we're extracting features from the image encoder module of CLIP (`visual`), but if you don't know which modules are available for a given model, just call `extractor.show_model()` to print all the modules.

```python
module_name = 'visual'

features = extractor.extract_features(
    batches=batches,
    module_name=module_name,
    flatten_acts=True,
    output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
)

save_features(features, out_path='path/to/features', file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"
```

#### Feature extraction with custom data pipeline

##### PyTorch

```python
module_name = 'visual'

# your custom dataset and dataloader classes come here (for example, a PyTorch data loader)
my_dataset = ...
my_dataloader = ...

with extractor.batch_extraction(module_name, output_type="tensor") as e: 
  for batch in my_dataloader:
    ... # whatever preprocessing you want to add to the batch
    feature_batch = e.extract_batch(
      batch=batch,
      flatten_acts=True, # flatten 2D feature maps from an early convolutional or attention layer
      )
    ... # whatever post-processing you want to add to the extracted features
```

##### TensorFlow / Keras

```python
module_name = 'visual'

# your custom dataset and dataloader classes come here (for example, TFRecords files)
my_dataset = ...
my_dataloader = ...

for batch in my_dataloader:
  ... # whatever preprocessing you want to add to the batch
  feature_batch = extractor.extract_batch(
    batch=batch,
    module_name=module_name,
    flatten_acts=True, # flatten 2D feature maps from an early convolutional or attention layer
    )
  ... # whatever post-processing you want to add to the extracted features
```

#### Human alignment

*Human alignment*: If you want to align the extracted features with human object similarity according to the approach introduced in *[Improving neural network representations using human similiarty judgments](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)* you can optionally `align` the extracted features using the following method:

```python
aligned_features = extractor.align(
    features=features,
    module_name=module_name,
    alignment_type="gLocal",
)
```

For more information about the available alignment types and aligned models see the [docs](https://vicco-group.github.io/thingsvision/Alignment.html). 


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

This is a joint open-source project between the [Max Planck Institute for Human Cognitive and Brain Sciences](https://www.cbs.mpg.de/en), Leipzig, and the [Machine Learning Group](https://web.ml.tu-berlin.de/) at Technische Universtität Berlin. Correspondence and requests for contributing should be adressed to [Lukas Muttenthaler](https://lukasmut.github.io/). Feel free to contact us if you want to become a contributor or have any suggestions/feedback. For the latter, you could also just post an issue or engange in discussions. We'll try to respond as fast as we can.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

