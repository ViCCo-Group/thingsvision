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
model_name = 'alexnet'
source = 'torchvision'
device = 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

Model names are case-sensitive and must be spelled exactly as they are in the `torchvision` documentation (e.g., `alexnet`, `resnet18`, `vgg16`, ...).

If you use `pretrained=True`, the model will by default be pretrained on ImageNet, otherwise it is initialized randomly. For some models, `torchvision` provides multiple weight initializations, in which case you can pass the name of the weights in the `model_parameters` argument, e.g. if you want to get the extractor for a `RegNet Y 32GF` model, pretrained using SWAG and finetuned on ImageNet, you would do the following:

```python
model_name = 'regnet_y_32gf'
source = 'torchvision'
device = 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters={'weights': 'IMAGENET1K_SWAG_LINEAR_V1'}
)
```

For a list of all available weights, please refer to the [torchvision documentation](https://pytorch.org/vision/stable/models.html).

## `timm`
`thingsvision` supports all models from the `timm` module. You can find a list of all available models [here](https://rwightman.github.io/pytorch-image-models/models/).

Example:
```python
model_name = 'tf_efficientnet_b0'
source = 'timm'
device = 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

Model names are case-sensitive and must be spelled exactly as they are in the `timm` documentation (e.g., `tf_efficientnet_b0`, `densenet121`, `mixnet_l`, ...). 

If you use `pretrained=True`, the model will be pretrained according to the model documentation, otherwise it is initialized randomly. 

## `vissl`
For the [VISSL](https://vissl.readthedocs.io/en/v0.1.5/) library, `thingsvision` supports ResNet50, pretrained using SimCLR (`simclr-rn50`), MoCov V2 (`mocov2-rn50`), Jigsaw (`jigsaw-rn50`), RotNet (`rotnet-rn50`), SwAV (`swav-rn50`) or PIRL (`pirl-rn50`), and finetuned on ImageNet. Here, the model name describes the pre-training method, instead of the model architecture.

Example:
```python
model_name = 'simclr-rn50'
source = 'vissl'
device = 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```


## `keras`
`thingsvision` supports all models from the `keras.applications` module. You can find a list of all available models [here](https://keras.io/api/applications/).

Example:
```python
model_name = 'VGG16'
source = 'keras'
device = 'cpu'

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

We also provide several custom models, which were not available in other sources at the time of writing, in the source `custom`. These models are:

### CORnet
We provide all CORnet models from [this paper](https://proceedings.neurips.cc/paper/2019/file/7813d1590d28a7dd372ad54b5d29d033-Paper.pdf). Available model names are:

- `cornet_s`
- `cornet_r`
- `cornet_rt`
- `cornet_z`

Example:
```python
model_name = 'cornet_s'
source = 'custom'
device = 'cpu'

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
model_name = 'Alexnet_ecoset'
source = 'custom'
device = 'cpu'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)
```

### Official CLIP and OpenCLIP

We provide models trained using [CLIP](https://arxiv.org/abs/2103.00020), both from the official repository and from [OpenCLIP](https://github.com/mlfoundations/open_clip). Available `model_name`s are:
- `clip`
- `OpenClip`

Both provide multiple model architectures and, in the case of OpenCLIP also multiple training datasets, which can be specified using the `model_parameters` argument. For example, if you want to get a `ViT-B/32` model from official CLIP, you would do the following:

```python
model_name = 'clip'
source = 'custom'
device = 'cpu'
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

In the case of OpenCLIP, you can also specify the dataset used for training for most models, e.g. if you want to get a `ViT-B/32` model trained on the `LAION-400M` dataset, you would do the following:

```python
model_name = 'openclip'
source = 'custom'
device = 'cpu'
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
