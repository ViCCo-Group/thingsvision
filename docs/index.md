---
layout: default
title: Home
nav_order: 1
description: "THINGSVision: Extracting Features from Deep Neural Networks"
permalink: /
---

# THINGSVision - Extracting Features from Deep Neural Networks.

THINGSVision is a python toolbox to quickly and easily extract and analyze image representations from state-of-the-art neural networks for computer vision. 

To get started, please check out the [Getting Started](GettingStarted.md) page, as well as our list of [Examples](Examples.md), or try our Colab notebooks for [Pytorch](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/pytorch.ipynb) and [Tensorflow](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/tensorflow.ipynb) .


## Model collection

Features can be extracted for all models in [torchvision](https://pytorch.org/vision/0.8/models.html), [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications), [timm](https://github.com/rwightman/pytorch-image-models), custom models (VGG-16, Resnet50, Inception_v3 and Alexnet) trained on [Ecoset](https://www.pnas.org/doi/10.1073/pnas.2011417118), each of the many [CORnet](https://github.com/dicarlolab/CORnet) versions and both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`).<br> 


Note that you have to use the respective model name (`str`). For example, if you want to use VGG16 from torchvision, use `vgg16` as the model name and if you want to use VGG16 from TensorFlow/Keras, use the model name `VGG16`. You can further specify the model source by setting the `source` parameter (e.g., `timm`, `torchvision`, `keras`).<br>


For the correct abbreviations of [torchvision](https://pytorch.org/vision/0.8/models.html) models have a look [here](https://github.com/pytorch/vision/tree/master/torchvision/models). For the correct abbreviations of [CORnet](https://github.com/dicarlolab/CORnet) models look [here](https://github.com/dicarlolab/CORnet/tree/master/cornet). To separate the string `cornet` from its variant (e.g., `s`, `z`) use a hyphen instead of an underscore (e.g., `cornet-s`, `cornet-z`).<br>

PyTorch examples:  `alexnet`, `resnet18`, `resnet50`, `resnet101`, `vit_b_16`, `vit_b_32`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`, `cornet-s`, `clip-ViT`


## IMPORTANT NOTES:

1. There exist four different sources from which neural network models and their (pretrained) weights can be downloaded. You can define the source of a model using the `source` argument. Possible sources are [`torchvision`](https://pytorch.org/vision/stable/models.html), [`keras`](https://keras.io/api/applications/), [`timm`](https://github.com/rwightman/pytorch-image-models), and `custom` (e.g., `source = torchvision`).

2. If you happen to use the [THINGS image database](https://osf.io/jum2f/), make sure to correctly `unzip` all zip files (sorted from A-Z), and have all `object` directories stored in the parent directory `./images/` (e.g., `./images/object_xy/`) as well as the `things_concepts.tsv` file stored in the `./data/` folder. `bash get_files.sh` does the latter for you. Images, however, must be downloaded from the [THINGS database](https://osf.io/jum2f/) `Main` subfolder.  **The download is around 5GB**.

    *   Go to <https://osf.io/jum2f/files/>
    *   Select `Main` folder and click on "Download as zip" button (top right).
    *   Unzip contained `object_images_*.zip` file using the password (check the `description.txt` file for details). For example:

        ``` {.bash}
        for fn in object_images_*.zip; do unzip -P the_password $fn; done
        ```

3. Features can be extracted for every layer for all `timm`, `torchvision`, `TensorFlow`, `CORnet` and `CLIP`/`OpenCLIP` models.

4. The script automatically extracts features for the specified `model` and `module`.

5. If you happen to extract hidden unit activations for many images, it is possible to run into `MemoryErrors`. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 



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
