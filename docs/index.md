---
layout: default
title: Home
nav_order: 1
description: "THINGSVision: Extracting Features from Deep Neural Networks"
permalink: /
---

# this is a header.




## Model collection

Features can be extracted for all models in [torchvision](https://pytorch.org/vision/0.8/models.html), [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications), [timm](https://github.com/rwightman/pytorch-image-models), custom models (VGG-16, Resnet50, Inception_v3 and Alexnet) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118), each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions and both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`).<br> 


Note that you have to use the respective model name (`str`). For example, if you want to use VGG16 from torchvision, use `vgg16` as the model name and if you want to use VGG16 from TensorFlow/Keras, use the model name `VGG16`. You can further specify the model source by setting the `source` parameter (e.g., `timm`, `torchvision`, `keras`).<br>


For the correct abbreviations of [torchvision](https://pytorch.org/vision/0.8/models.html) models have a look [here](https://github.com/pytorch/vision/tree/master/torchvision/models). For the correct abbreviations of [CORnet](https://github.com/dicarlolab/CORnet) models look [here](https://github.com/dicarlolab/CORnet/tree/master/cornet). To separate the string `cornet` from its variant (e.g., `s`, `z`) use a hyphen instead of an underscore (e.g., `cornet-s`, `cornet-z`).<br>

PyTorch examples:  `alexnet`, `resnet18`, `resnet50`, `resnet101`, `vit_b_16`, `vit_b_32`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`, `cornet-s`, `clip-ViT`



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
