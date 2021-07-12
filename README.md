## Model collection

Features can be extracted for all models in [torchvision](https://pytorch.org/vision/0.8/models.html), each of the [CORnet](https://github.com/dicarlolab/CORnet) versions and both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`). For the correct abbreviations of [torchvision](https://pytorch.org/vision/0.8/models.html) models have a look [here](https://github.com/pytorch/vision/tree/master/torchvision/models). For the correct abbreviations of [CORnet](https://github.com/dicarlolab/CORnet) models look [here](https://github.com/dicarlolab/CORnet/tree/master/cornet). To separate the string `cornet` from its variant (e.g., `s`, `z`) use a hyphen instead of an underscore (e.g., `cornet-s`, `cornet-z`).<br>

Examples:  `alexnet`, `resnet50`, `resnet101`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`, `cornet-s`, `clip-ViT`

## Environment Setup

Make sure you have the latest Python version (>= 3.7) and [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/). Note that [PyTorch 1.7.1](https://pytorch.org/) requires CUDA 10.2 or above, if you want to extract network activations on a GPU. However, the code runs already pretty fast on a strong CPU (Intel i7 or i9). Run the following `pip` command in your terminal. 

``` bash
$ pip install thingsvision
```

You have to download files from the parent repository (i.e., this repo), if you want to extract network activations for [THINGS](https://osf.io/jum2f/). Simply download the shell script `get_files.sh` from this repo and execute it as follows (the shell script will do file downloading and moving for you):

``` bash
$ wget https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (Linux)
$ curl -O https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (macOS)
$ bash get_files.sh
```

Execute the following lines to have the latest `PyTorch` and `CUDA` versions available (not necessary, but perhaps desirable):

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine (e.g., 10.2) or `cpuonly` when installing on a machine without a GPU.

## IMPORTANT NOTES:

1. Image data will automatically be converted into a ready-to-use dataset class, and subsequently wrapped with a `PyTorch` mini-batch dataloader to make neural activation extraction more efficient.

2. If you happen to use the [THINGS image database](https://osf.io/jum2f/), make sure to correctly `unzip` all zip files (sorted from A-Z), and have all `object` directories stored in the parent directory `./images/` (e.g., `./images/object_xy/`) as well as the `things_concepts.tsv` file stored in the `./data/` folder. `bash get_files.sh` does the latter for you. Images, however, must be downloaded from the [THINGS database](https://osf.io/jum2f/) `Main` subfolder.  **The download is around 5GB**.

*   Go to <https://osf.io/jum2f/files/>
*   Select `Main` folder and click on "Download as zip" button (top right).
*   Unzip contained `object_images_*.zip` file using the password (check the
    `description.txt` file for details). For example:

    ``` {.bash}
    for fn in object_images_*.zip; do unzip -P the_password $fn; done
    ```

3. Features can be extracted at every layer for all `torchvision`, `CORnet` and `CLIP` models.

4. If you happen to be interested in an ensemble of `feature maps`, as introduced in this recent [COLING 2020 paper](https://www.aclweb.org/anthology/2020.coling-main.173/), you can simply extract an ensemble of `conv` or `max-pool` layers. The ensemble can additionally be concatenated with the activations of the penultimate layer, and subsequently transformed into a lower-dimensional space with `PCA` to reduce noise and only keep those dimensions that account for most of the variance. 

5. The script automatically extracts features for the specified `model` and `layer` and stores them together with the `targets` in `out_path` (see above).

6. If you happen to extract hidden unit activations for many images, it is possible to run into `MemoryErrors`. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 

## Extract features at specific layer of a state-of-the-art `torchvision`, `CORnet` or `CLIP` model 

### Example call for AlexNet:

```python
import torch
import thingsvision.vision as vision

model_name = 'alexnet'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, transforms = vision.load_model(model_name, pretrained=True, model_path=None, device=device)
module_name = vision.show_model(model, model_name)

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

dl = vision.load_dl(root='./images/', out_path=f'./{model_name}/{module_name}/features', batch_size=64, transforms=transforms)
features, targets, probas = vision.extract_features(model, dl, module_name, batch_size=64, flatten_acts=True, device=device, return_probabilities=True)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
```

### Example call for [CLIP](https://github.com/openai/CLIP):

```python
import torch
import thingsvision.vision as vision

model_name = 'clip-ViT'
module_name = 'visual'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, transforms = vision.load_model(model_name, pretrained=True, model_path=None, device=device)
dl = vision.load_dl(root='./images/', out_path=f'./{model_name}/{module_name}/features', batch_size=64, transforms=transforms)
features, targets = vision.extract_features(model, dl, module_name, batch_size=64, flatten_acts=False, device=device, clip=True, return_probabilities=False)

features = vision.center_features(features)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
vision.save_targets(targets, f'./{model_name}/{module_name}/targets', 'npy')
```

### Example call for [CORnet](https://github.com/dicarlolab/CORnet)

```python
import torch
import thingsvision.vision as vision

model_name = 'cornet-s'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, transforms = vision.load_model(model_name, pretrained=True, model_path=None, device=device)
module_name = vision.show_model(model, model_name)

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
  (V2): CORblock_S(
    (conv_input): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (skip): Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    (norm_skip): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (nonlin1): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (nonlin2): ReLU(inplace=True)
    (conv3): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (nonlin3): ReLU(inplace=True)
    (output): Identity()
    (norm1_0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm1_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (V4): CORblock_S(
    (conv_input): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (skip): Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    (norm_skip): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (nonlin1): ReLU(inplace=True)
    (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (nonlin2): ReLU(inplace=True)
    (conv3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (nonlin3): ReLU(inplace=True)
    (output): Identity()
    (norm1_0): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_0): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm1_1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm1_2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm1_3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (IT): CORblock_S(
    (conv_input): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (skip): Conv2d(512, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    (norm_skip): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (nonlin1): ReLU(inplace=True)
    (conv2): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (nonlin2): ReLU(inplace=True)
    (conv3): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (nonlin3): ReLU(inplace=True)
    (output): Identity()
    (norm1_0): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_0): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm1_1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2_1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (decoder): Sequential(
    (avgpool): AdaptiveAvgPool2d(output_size=1)
    (flatten): Flatten()
    (linear): Linear(in_features=512, out_features=1000, bias=True)
    (output): Identity()
  )
)

#Enter part of the model for which you would like to extract features:

(e.g., "decoder.flatten")

dl = vision.load_dl(root='./images/', out_path=f'./{model_name}/{module_name}/features', batch_size=64, transforms=transforms)
features, targets = vision.extract_features(model, dl, module_name, batch_size=64, flatten_acts=False, device=device, return_probabilities=False)

features = vision.center_features(features)
features = vision.normalize_features(features)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
```

## ImageNet class predictions

Would you like to know the probabilities corresponding to the `top k` predicted ImageNet classes for each of your images? Simply set the `return_probabilities` argument to `True` and use the `get_class_probabilities` helper (the function works for both `synset` and `class` files). Note that this is, unfortunately, not (yet) possible for `CLIP` models due to their multi-modality and different training objectives. You are required to use the same `out_path` throughout which is why `out_path` must correspond to the path that was used in `vision.load_dl`.

```python

features, targets, probas = vision.extract_features(model, dl, module_name, batch_size, flatten_acts=False, device=device, return_probabilities=True)
class_probas = vision.get_class_probabilities(probas=probas, out_path=out_path, cls_file='./data/imagenet1000_classes.txt', top_k=5, save_as_json=True)
```

## Model comparison

To compare object representations extracted from specifid models and layers against each other, for a `List[str]` of models and layers a user can perform the following operation,


```python
clip_list = [n.startswith('clip') for n in model_names]

correlations = vision.compare_models(
                                     root=root,
                                     out_path=out_path,
                                     model_names=model_names,
                                     module_names=module_names,
                                     pretrained=True,
                                     batch_size=batch_size,
                                     flatten_acts=True,
                                     clip=clip_list,
                                     save_features=True,
                                     dissimilarity='correlation',
                                     correlation='pearson',
                                    )
```

The function returns a correlation matrix in the form of a `Pandas` dataframe whose rows and columns correspond to the names of the models in `model_names`. The cell elements are the correlation coefficients for each model combination. The dataframe can subsequently be converted into a heatmap with `matplotlib` or `seaborn`. We will release a clear and concise documentary as soon as possible. Until then, we recommend to look at Section 3.2.3 in the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.03.11.434979v1.full).

## OpenAI's CLIP models

### CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

## Citation

If you use this GitHub repository (or any modules associated with it), we would grately appreciate to cite our [preprint](https://www.biorxiv.org/content/10.1101/2021.03.11.434979v1.full) as follows:

```latex
@article{Muttenthaler_2021,
	author = {Muttenthaler, Lukas and Hebart, Martin N.},
	title = {THINGSvision: a Python toolbox for streamlining the extraction of activations from deep neural networks},
	elocation-id = {2021.03.11.434979},
	year = {2021},
	doi = {10.1101/2021.03.11.434979},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/03/12/2021.03.11.434979},
	eprint = {https://www.biorxiv.org/content/early/2021/03/12/2021.03.11.434979.full.pdf},
	journal = {bioRxiv}
}
```
