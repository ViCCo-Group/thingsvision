[![Unittests](https://github.com/ViCCo-Group/THINGSvision/actions/workflows/tests.yml/badge.svg)](https://github.com/ViCCo-Group/THINGSvision/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/ViCCo-Group/THINGSvision/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ViCCo-Group/THINGSvision/actions/workflows/python-publish.yml)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Code Coverage](https://codecov.io/gh/ViCCo-Group/THINGSvision/branch/master/graph/badge.svg)](https://github.com/ViCCo-Group/THINGSvision/actions/workflows/coverage.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/pytorch.ipynb)




## Model collection

Features can be extracted for all models in [torchvision](https://pytorch.org/vision/0.8/models.html), [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications), each of the [CORnet](https://github.com/dicarlolab/CORnet) versions and both [CLIP](https://github.com/openai/CLIP) variants (`clip-ViT` and `clip-RN`). For the correct abbreviations of [torchvision](https://pytorch.org/vision/0.8/models.html) models have a look [here](https://github.com/pytorch/vision/tree/master/torchvision/models). For the correct abbreviations of [CORnet](https://github.com/dicarlolab/CORnet) models look [here](https://github.com/dicarlolab/CORnet/tree/master/cornet). To separate the string `cornet` from its variant (e.g., `s`, `z`) use a hyphen instead of an underscore (e.g., `cornet-s`, `cornet-z`).<br>

Examples:  `alexnet`, `resnet18`, `resnet50`, `resnet101`, `vit_b_16`, `vit_b_32`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`, `cornet-s`, `clip-ViT`

## Environment Setup

We recommend to create a new `conda environment` with Python version 3.7 or 3.8 (no tests for 3.9 yet) before using `thingsvision`. Check out the `environment.yml` file in `envs`, if you want to create a `conda environment` via `yml`. Activate the `environment` and run the following `pip` command in your terminal. 

```bash
$ pip install --upgrade thingsvision
```

You have to download files from the parent repository (i.e., this repo), if you want to extract network activations for [THINGS](https://osf.io/jum2f/). Simply download the shell script `get_files.sh` from this repo and execute it as follows (the shell script will do file downloading and moving for you):

```bash
$ wget https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (Linux)
$ curl -O https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (macOS)
$ bash get_files.sh
```

Execute the following lines to have the latest `PyTorch` and `CUDA` versions available (not necessary, but perhaps desirable):

```bash
$ conda install pytorch torchvision torchaudio -c pytorch
```

## Google Colab

Alternatively, you can use Google Colab to play around with `thingsvision` by uploading your image data to Google Drive.
You can find the jupyter notebook using `PyTorch` [here](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/pytorch.ipynb) and the `TensorFlow` example [here](https://colab.research.google.com/github/ViCCo-Group/THINGSvision/blob/master/doc/tensorflow.ipynb).

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

3. Features can be extracted at every layer for all `torchvision`, `TensorFlow`, `CORnet` and `CLIP` models.

4. If you are interested in an ensemble of `feature maps`, as introduced in this recent [COLING 2020 paper](https://www.aclweb.org/anthology/2020.coling-main.173/), you can simply extract an ensemble of `conv` or `max-pool` layers. The ensemble can additionally be concatenated with the activations of the penultimate layer, and subsequently be mapped into a lower-dimensional space with `PCA` to reduce noise, and only keep those dimensions that account for most of the variance in the data. 

5. The script automatically extracts features for the specified `model` and `layer`.

6. If you happen to extract hidden unit activations for many images, it is possible to run into `MemoryErrors`. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 

## Extract features at specific layer of a state-of-the-art `torchvision`, `TensorFlow`, `CORnet`, or `CLIP` model 

The following examples demonstrate how to load a model with PyTorch or TensorFlow into memory, and how to subsequently extract features. 
Please keep in mind, that the model names as well as the layer names depend on the backend. If you use PyTorch, you will need to use these [model names](https://pytorch.org/vision/stable/models.html). If you use Tensorflow, you will need to use these [model names](https://keras.io/api/applications/). You can find the layer names by using `model.show()`. 


### Example call for AlexNet with PyTorch:

```python
import torch
import thingsvision.vision as vision

from thingsvision.model_class import Model

model_name = 'alexnet'
backend = 'pt'
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)
module_name = model.show()

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

dl = vision.load_dl(
	root='./images/',
	out_path=f'./{model_name}/{module_name}/features',
	batch_size=batch_size,
	transforms=model.get_transformations(),
	backend=backend,
	)
features, targets, probas = model.extract_features(
				data_loader=dl,
				module_name=module_name,
				flatten_acts=True,
				clip=False,
				return_probabilities=True,
				)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
```

### Example call for [CLIP](https://github.com/openai/CLIP) with PyTorch:

```python
import torch
import thingsvision.vision as vision

from thingsvision.model_class import Model

model_name = 'clip-ViT'
module_name = 'visual'
backend = 'pt'
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)
dl = vision.load_dl(
	root='./images/',
	out_path=f'./{model_name}/{module_name}/features',
	batch_size=batch_size,
	transforms=model.get_transformations(),
	backend=backend,
	)
features, targets = model.extract_features(
			data_loader=dl,
			module_name=module_name,
			flatten_acts=False,
			clip=True,
			)
features = vision.center_features(features)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
vision.save_targets(targets, f'./{model_name}/{module_name}/targets', 'npy')
```

### Example call for [ViT](https://arxiv.org/pdf/2010.11929.pdf) with PyTorch:

```python
import torch
import thingsvision.vision as vision

from thingsvision.model_class import Model

model_name = 'vit_b_16'
backend = 'pt'
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)
module_name = model.show()

VisionTransformer(
  (conv_proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  (encoder): Encoder(
    (dropout): Dropout(p=0.0, inplace=False)
    (layers): Sequential(
      (encoder_layer_0): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_1): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_2): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_3): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_4): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_5): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_6): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_7): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_8): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_9): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_10): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
      (encoder_layer_11): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU()
          (dropout_1): Dropout(p=0.0, inplace=False)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout_2): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
  (heads): Sequential(
    (head): Linear(in_features=768, out_features=1000, bias=True)
  )
)

#Enter part of the model for which you would like to extract features:

(e.g., "encoder.layers.encoder_layer_11.mlp.linear_2")

dl = vision.load_dl(
	root='./images/',
	out_path=f'./{model_name}/{module_name}/features',
	batch_size=batch_size,
	transforms=model.get_transformations(),
	backend=backend,
	)
	
features, targets, probas = model.extract_features(
                data_loader=dl,
                module_name=module_name,
                flatten_acts=False,
                clip=False,
                return_probabilities=True,
)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
vision.save_targets(targets, f'./{model_name}/{module_name}/targets', 'npy')
```

### Example call for [CORnet](https://github.com/dicarlolab/CORnet) with PyTorch:

```python
import torch
import thingsvision.vision as vision

from thingsvision.model_class import Model

model_name = 'cornet-s'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64

model = Model(model_name, pretrained=True, model_path=None, device=device)
module_name = model.show()

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

dl = vision.load_dl(
	root='./images/',
	out_path=f'./{model_name}/{module_name}/features',
	batch_size=batch_size,
	transforms=model.get_transformations(),
	backend=backend,
	)
features, targets = model.extract_features(
			data_loader=dl,
			module_name=module_name,
			flatten_acts=False,
			clip=False,
			)

features = vision.center_features(features)
features = vision.normalize_features(features)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
```

### Example call for VGG16 with TensorFlow:

```python
import tensorflow as tf 
import thingsvision.vision as vision
from thingsvision.model_class import Model

model_name = 'VGG16'
backend = 'tf'
module_name = 'block1_conv1'
batch_size = 64

device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)

dl = vision.load_dl(
	root='./images/',
	out_path=f'./{model_name}/{module_name}/features',
	batch_size=batch_size,
	transforms=model.get_transformations(),
	backend=backend,
	)
features, targets, probas = model.extract_features(
				data_loader=dl,
				module_name=module_name,
				flatten_acts=True,
				clip=False,
				return_probabilities=True,
				)

vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
```

### Optional Center Cropping

Center cropping is used by default but can be deactivated by turning off the `apply_center_crop` argument of the `get_transformations` method.

```python
apply_center_crop = False
dl = vision.load_dl(
		root='./images/',
		out_path=f'./{model_name}/{module_name}/features',
		batch_size=batch_size,
		transforms=model.get_transformations(apply_center_crop=apply_center_crop),
		backend=backend,
		)
```

## Extract features from custom models

If you want to use a custom model from the `custom_models` directory, you need to use the class name e.g. `VGG16_ecoset` as model name. The script will use the PyTorch or Tensorflow implementation depending on the `backend` value.

```python
from thingsvision.model_class import Model
model_name = 'VGG16_ecoset'
model = Model(model_name, pretrained=True, model_path=None, device=device, backend=backend)
```

## Representational Similarity Analysis (RSA) 

Comparison between representational (dis-)similarity matrices corresponding to model features and human representations (e.g., fMRI recordings) respectively.

```python
rdm_dnn = vision.compute_rdm(features, method='correlation')
corr_coeff = vision.correlate_rdms(rdm_dnn, rdm_human, correlation='pearson')
```

## ImageNet class predictions

Would you like to know the probabilities corresponding to the `top k` predicted ImageNet classes for each of your images? Simply set the `return_probabilities` argument to `True` and use the `get_class_probabilities` helper (the function works for both `synset` and `class` files). Note that this is, unfortunately, not (yet) possible for `CLIP` models due to their multi-modality and different training objectives. You are required to use the same `out_path` throughout which is why `out_path` must correspond to the path that was used in `vision.load_dl`. Save the ImageNet class file (e.g., `imagenet1000_classes.txt`) in your cwd in a subfolder called `data`. You can download the class file [here](https://github.com/ViCCo-Group/THINGSvision/tree/master/thingsvision/data).

```python
features, targets, probas = model.extract_features(
				data_loader=dl,
				module_name=module_name,
				flatten_acts=False,
				clip=False,
				return_probabilities=True,
				)
# return top k class probabilities and save dict as json file
class_probas = vision.get_class_probabilities(probas=probas, out_path=out_path, cls_file='./data/imagenet1000_classes.txt', top_k=5, save_as_json=True)
# save features to disk
vision.save_features(features, f'./{model_name}/{module_name}/features', 'npy')
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
                                     backend='pt',
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

## Adding custom models

If you want to use your own model and/or want to make it public, you just need to implement a class inheriting from the `custom_models/custom.py:Custom` class and implement the `create_model` method.
There you can build/download the model and its weights. The constructors expects a `device` (str) and a `backend` (str).
Afterwards you can put the file in the `custom_models` directory and create a pull request to include the model in the official GitHub repository.

```python
from thingsvision.custom_models.custom import Custom
import torchvision.models as torchvision_models
import torch

class VGG16_ecoset(Custom):
    def __init__(self, device, backend) -> None:
        super().__init__(device, backend)

    def create_model(self):
        if self.backend == 'pt':
            model = torchvision_models.vgg16_bn(pretrained=False, num_classes=565)
            path_to_weights = 'https://osf.io/fe7s5/download'
            state_dict = torch.hub.load_state_dict_from_url(path_to_weights, map_location=self.device)
            model.load_state_dict(state_dict)
            return model
```

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
