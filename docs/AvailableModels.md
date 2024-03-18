---
title: Available models and sources (+ examples)
nav_order: 4
---

# Available models and sources

`thingsvision` currently supports many models from several different sources, which represent different places or other libraries from which the model architectures or weights can come from. You can find more information about which models are available in which source and notes on their usage on this page.

## `torchvision`
`thingsvision` supports all models from the `torchvision.models` module. You can find a list of all available models [here](https://pytorch.org/vision/stable/models.html). 

Example:
```python
import torch
from thingsvision import get_extractor

model_name = 'alexnet'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {'weights': 'DEFAULT'}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters,
)
```

Model names are case-sensitive and must be spelled exactly as they are in the `torchvision` documentation (e.g., `alexnet`, `resnet18`, `vgg16`, ...).

If you use `pretrained=True`, the model will by default be pretrained on ImageNet, otherwise it is initialized randomly. For some models, `torchvision` provides multiple weight initializations, in which case you can pass the name of the weights in the `model_parameters` argument, e.g. if you want to get the extractor for a `RegNet Y 32GF` model, pretrained using SWAG and finetuned on ImageNet, you would do the following:

```python
import torch
from thingsvision import get_extractor

model_name = 'regnet_y_32gf'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {'weights': 'IMAGENET1K_SWAG_LINEAR_V1'}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters,
)
```

For a list of all available weights, please refer to the [torchvision documentation](https://pytorch.org/vision/stable/models.html).

## `timm`
`thingsvision` supports all models from the `timm` module. You can find a list of all available models [here](https://rwightman.github.io/pytorch-image-models/models/).

Example:
```python
import torch
from thingsvision import get_extractor

model_name = 'tf_efficientnet_b0'
source = 'timm'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

Model names are case-sensitive and must be spelled exactly as they are in the `timm` documentation (e.g., `tf_efficientnet_b0`, `densenet121`, `mixnet_l`, ...). 

If you use `pretrained=True`, the model will be pretrained according to the model documentation, otherwise it is initialized randomly. 

## `ssl`
`thingsvision` provides various Self-supervised learning models that are loaded from the [VISSL](https://vissl.readthedocs.io/en/v0.1.5/) library or the Torch Hub.
* SimCLR (`simclr-rn50`)
* MoCov V2 (`mocov2-rn50`), 
* Jigsaw (`jigsaw-rn50`), 
* RotNet (`rotnet-rn50`)
* SwAV (`swav-rn50`) 
* PIRL (`pirl-rn50`) 
* BarlowTwins (`barlowtwins-rn50`)
* VicReg (`vicreg-rn50`)
* DINO (`dino-rn50`)

All models have the ResNet50 architecture and are pretrained on ImageNet-1K. 
Here, the model name describes the pre-training method, instead of the model architecture.

DINO models are available in ViT (Vision Transformer) and XCiT (Cross-Covariance Image Transformer) variants. For ViT models trained using DINO, the following models are available: `dino-vit-small-p8`, `dino-vit-small-p16`, `dino-vit-base-p8`, `dino-vit-base-p16`, where the trailing number describes the image patch resolution in the ViT (i.e. either 8x8 or 16x16). For the XCiT models, we have `dino-xcit-small-12-p16`, `dino-xcit-small-12-p8`, `dino-xcit-medium-24-p16`, `dino-xcit-medium-24-p8`, where the penultimate number represents model depth (12 = small, 24 = medium).


Example SimCLR:

```python
import torch
from thingsvision import get_extractor

model_name = 'simclr-rn50'
source = 'ssl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

Example DINO:

```python
import torch
from thingsvision import get_extractor

model_name = 'dino-vit-base-p16'
source = 'ssl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_paramters = {"extract_cls_token": True} # extract features only for the [cls] token of DINO

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters,
)
```


## `keras`
`thingsvision` supports all models from the `keras.applications` module. You can find a list of all available models [here](https://keras.io/api/applications/).

Example:
```python
import torch
from thingsvision import get_extractor

model_name = 'VGG16'
source = 'keras'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

Model names are case-sensitive and must be spelled exactly as they are in the `keras.applications` documentation (e.g., `VGG16`, `ResNet50`, `InceptionV3`, ...).

If you use `pretrained=True`, the model will be pretrained on ImageNet, otherwise it is initialized randomly.

## `custom` 

In addition, we provide several custom models - that are not available in other sources -, in the `custom` source. These models are:

### CORnet
We provide all CORnet models from [this paper](https://proceedings.neurips.cc/paper/2019/file/7813d1590d28a7dd372ad54b5d29d033-Paper.pdf). Available model names are:

- `cornet-s`
- `cornet-r`
- `cornet-rt`
- `cornet-z`

Example:
```python
import torch
from thingsvision import get_extractor

model_name = 'cornet-s'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

### Models trained on Ecoset

We provide models trained on the [Ecoset](https://www.kietzmannlab.org/ecoset/) dataset, which contains 1.5m images from 565 categories selected to be both frequent in linguistic use and rated as concrete by human observers. Available `model_name`s are:

- `Alexnet_ecoset`
- `Resnet50_ecoset`
- `VGG16_ecoset`
- `Inception_ecoset`

Example:

```python
import torch
from thingsvision import get_extractor

model_name = 'Alexnet_ecoset'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

### Official CLIP and OpenCLIP

We provide [CLIP](https://arxiv.org/abs/2103.00020) models from the official CLIP repo and from [OpenCLIP](https://github.com/mlfoundations/open_clip). Available `model_name`'s are:

- `clip`
- `OpenClip`

Both provide multiple model architectures and, in the case of OpenCLIP also different training datasets, which can both be specified using the `model_parameters` argument. For example, if you want to get a `ViT-B/32` model from the official CLIP repo (trained on WIT), you would do the following:

```python
import torch
from thingsvision import get_extractor

model_name = 'clip'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {
    'variant': 'ViT-B/32'
}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters
)
```

`ViT-B/32` is the default model architecture, so you can also leave out the `model_parameters` argument. For a list of all available architectures and datasets, please refer to the [CLIP repo](https://github.com/openai/CLIP/blob/main/clip/clip.py).

In the case of `OpenCLIP`, you can also specify the dataset used for training for most models, e.g. if you want to get a `ViT-B/32` model trained on the `LAION-400M` dataset, you would do the following:

```python
import torch
from thingsvision import get_extractor

model_name = 'OpenCLIP'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {
    'variant': 'ViT-B/32',
    'dataset': 'laion400m_e32'
}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters
)
```

For a list of all available architectures and datasets, please refer to the [OpenCLIP repo](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py).

### Harmonization

If you want to extract features for [harmonized models](https://vicco-group.github.io/thingsvision/AvailableModels.html#harmonization) from the [Harmonization repo](https://github.com/serre-lab/harmonization), you have to run the following `pip` command in your `thingsvision` environment (FYI: as of now, this seems to be working smoothly only on Ubuntu but not on macOS),

```bash
$ pip install git+https://github.com/serre-lab/Harmonization.git
$ pip install keras-cv-attention-models>=1.3.5
```

The following models from [here](https://github.com/serre-lab/harmonization) are available for feature extraction:

- `ViT_B16`
- `ResNet50`
- `VGG16`
- `EfficientNetB0`
- `tiny_ConvNeXT`
- `tiny_MaxViT`
- `LeViT_small`


Example:

```python
import torch
from thingsvision import get_extractor

model_name = 'Harmonization'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {
    'variant': 'ViT_B16'
}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters
)
```


### DreamSim
In `thingsvision` you can extract representations from [DreamSim](https://dreamsim-nights.github.io/). See the official [DreamSim repo](https://github.com/ssundaram21/dreamsim) for more information. To extract features, install the `dreamsim` package with the following `pip` command (ideally, into your `thingsvision` environment):

```bash
 $ pip install dreamsim==0.1.2
```

The base model name is:
- `DreamSim`

We provide four `DreamSim` models: `clip_vitb32`, `open_clip_vitb32`, `dino_vitb16`, and a DreamSim `ensemble`. Specify this using the `model_parameters` argument. For instance, to get the OpenCLIP variant of DreamSim you want to do the following:

```python
import torch
from thingsvision import get_extractor

model_name = 'DreamSim'
module_name = 'model.mlp'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {
    'variant': 'open_clip_vitb32'
}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters
)
```

To load the CLIP ViT-B/32 version of DreamSim, pass `'clip_vitb32'` to the `variant` parameter instead. Caution (!): for the DreamSim `ensemble` features can only be extracted from the `model.mlp` module and not for the `model` block. We are currently working on a version that allows feature extraction from the `model` block. Please be patient until then.
