---
title: Getting Started
nav_order: 2
---


## Environment Setup

We recommend to create a new `conda environment` with Python version 3.7, 3.8, 3.9 or 3.10 before using `thingsvision`. 
Please use the `environment.yml` file in the `envs` subfolder like so:
    
```bash
$ conda env create --prefix /path/to/conda/envs/thingsvision --file envs/environment.yml
$ conda activate thingsvision
```

Then run the following `pip` command in your terminal,

```bash
$ pip install --upgrade thingsvision
```

You have to download files from the parent folder of this repository, if you want to extract features for [THINGS](https://osf.io/jum2f/). Simply download the shell script `get_files.sh` from this repo and execute it as follows (the shell script will automatically do file downloading and moving for you):

```bash
$ wget https://raw.githubusercontent.com/ViCCo-Group/thingsvision/master/get_files.sh (Linux)
$ curl -O https://raw.githubusercontent.com/ViCCo-Group/thingsvision/master/get_files.sh (macOS)
$ bash get_files.sh
```

## Command Line Interface

`thingsvision` was designed to make extracting features as easy as possible. If you have a folder of images `./data` and simply want to extract their features, its easiest use our available command-line interface. The interface includes two options `thingsvision show-model` and `thingsvision extract-features`. Example calls might be:

```bash
thingsvision show-model --model-name "alexnet" --source "torchvision"
thingsvision extract_features --image-root "./data" --model-name "alexnet" --module-name "features.10" --batch-size 32 --device "cuda" --source "torchvision" --file-format "npy" --out-path "./features"
```

See `thingsvision show-model -h` and `thingsvision extract-features -h` for a list of all optional arguments. The command-line interface generally provides just the basic extraction functionality, but is probably enough for most users. If you need more fine-grained control over the extraction itself, use the python package directly. To do this, we provide examples for a range of models and libraries [here](../docs/examples.md).
