---
title: Getting Started
nav_order: 2
---
# Getting started

## Setting up your environment

### Working locally
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

### Google Colab
Alternatively, you can use Google Colab to play around with `thingsvision` by uploading your image data to Google Drive (via directory mounting).
You can find the jupyter notebook using `PyTorch` [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/pytorch.ipynb) and the `TensorFlow` example [here](https://colab.research.google.com/github/ViCCo-Group/thingsvision/blob/master/notebooks/tensorflow.ipynb).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Basic usage -->
## Basic usage

### Command Line Interface (CLI)

`thingsvision` was designed to simplify feature extraction. If you have some folder of images (e.g., `./images`) and want to extract features for each of these images without opening a Jupyter Notebook instance or writing a Python script, it's probably easiest to use our CLI. The interface includes two options,

- `thingsvision show-model`
- `thingsvision extract-features`

Example calls might look as follows:

```bash
thingsvision show-model --model-name "alexnet" --source "torchvision"
thingsvision extract_features --image-root "./data" --model-name "alexnet" --module-name "features.10" --batch-size 32 --device "cuda" --source "torchvision" --file-format "npy" --out-path "./features"
```

See `thingsvision show-model -h` and `thingsvision extract-features -h` for a list of all possible arguments. Note that the CLI provides just the basic extraction functionalities but is probably enough for most users that don't want to dive too deep into various models and modules. 

### Python commands for custom script or notebook

If you need more fine-grained control over the extraction itself, we recommend to use the python package directly and write your own Python script. To do this start by importing all the necessary components and instantiating a `thingsvision` extractor. Here we're using a `CLIP` model as the model to extract features from. In addition, we move the model to GPU for faster inference,

```python
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

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
  model_parameters=model_parameters,
)
```

As a next step, create both a dataset and a dataloader for your images. Here, we assume that all of your images are stored in a single `root` directory which can contain subfolders (e.g., for individual classes as in ImageNet). Therefore, we leverage the thingsvision `ImageDataset` class. 

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

Now all that is left is to extract the image features and store them to disk! We're extracting features from the image encoder of CLIP (`visual`).

```python
module_name = 'visual'

features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True, # flatten 2D feature maps from an early convolutional or attention layer
  output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP is one!)
)

save_features(features, out_path='path/to/features', file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"
```

### Extraction with custom data pipeline

#### PyTorch

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

#### TensorFlow / Keras

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

### Showing available modules of a model
If you don't know which modules exist in your model, you can use the `show_model` method to print a summary of the model architecture. For example, if you want to see which modules exist in AlexNet (using the extractor from above), you can run the following:

```python
extractor.show_model()

# Output:
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
```

The module names you have to use in the extractor depend on the model you're using. For example, the first convolutional layer in AlexNet is called `features.0` and the last convolutional layer is called `features.10`. The last fully connected layer is called `classifier.6`.

In comparison, the `.show_model()` output for ResNet50 looks like this:
```python
extractor.show_model()

# Output:
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size [...]
```
so the first convolutional layer is called `conv1` and the last convolutional layer is called `layer4.2.conv3`. The last fully connected layer would be called `fc`.
