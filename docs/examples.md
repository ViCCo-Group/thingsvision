---
title: Examples
has_children: true
nav_order: 3
---

## Feature extraction

### Extract features for a specific layer/module of a state-of-the-art `torchvision`, `timm`, `TensorFlow`, `CORnet`, or `CLIP` model

The following examples demonstrate how to load a model with PyTorch or TensorFlow/Keras and how to subsequently extract features. 
Please keep in mind that the model names as well as the layer names depend on the backend you want to use. If you use PyTorch, you will need to use these [model names](https://pytorch.org/vision/stable/models.html). If you use Tensorflow, you will need to use these [model names](https://keras.io/api/applications/). You can find the layer names by using `extractor.show_model()`.