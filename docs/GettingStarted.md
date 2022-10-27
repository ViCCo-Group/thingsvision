---
title: Getting Started
nav_order: 2
---


## Environment Setup

We recommend to create a new `conda environment` with Python version 3.7, 3.8, or 3.9 before using `thingsvision`. 
To use the prepared `environment.yml` file, use:
    
```bash
$ conda env create -f environment.yml
$ conda activate thingsvision
```


Then run the following `pip` command in your terminal.

```bash
$ pip install --upgrade thingsvision
```

You have to download files from the parent folder of this repository, if you want to extract network activations for [THINGS](https://osf.io/jum2f/). Simply download the shell script `get_files.sh` from this repo and execute it as follows (the shell script will automatically do file downloading and moving for you):

```bash
$ wget https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (Linux)
$ curl -O https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh (macOS)
$ bash get_files.sh
```

