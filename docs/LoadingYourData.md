---
title: Loading your data
nav_order: 3
---

# Loading your data

To use `thingsvision` you need to load your data into a `Dataset` object. This can be done in two ways:

## Using the ImageDataset class
For images simply stored in a folder, you can use the `ImageDataset` class. The most basic usage is to simply provide the path to the folder containing the images. The images will be loaded automatically and the labels will be inferred from the folder structure. For example, if you have the following folder structure:

```
root
├── class1
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
└── class2
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```

Then the `ImageDataset` will automatically load all images and assign the labels `class1` and `class2` to the images in the respective folders. Example usage:

```python
from thingsvision.utils.data import ImageDataset

# load extractor beforehand
extractor = ...

dataset = ImageDataset(
    root='path/to/root/img/directory' # (e.g., './root/')
    out_path='path/to/features',
    backend=extractor.get_backend(),
    transforms=extractor.get_transformations()
)
```

To later assign images to extracted features, the matching is written into `file_names.txt` in the `out_path` directory. This file contains the image file names in the order of the extracted features. 

Note that an image transformation and extractor backend has to be provided to the `ImageDataset` class. This is because the images are loaded in the `ImageDataset` class and then passed to the extractor. The extractor expects the images to be preprocessed in a certain way, which is why the transformation is provided by the extractor, but you can also provide your own transformation.

## Using the HDF5Dataset class
Some image datasets (notably the NSD stimuli) are stored in HDF5 files. To load such a dataset, you can use the `HDF5Dataset` class. The most basic usage is to simply provide the path to the HDF5 file and the key of the image dataset in the HDF5 file. Example usage:

```python
from thingsvision.utils.data import HDF5Dataset

# load extractor beforehand
extractor = ...

dataset = HDF5Dataset(
    hdf5_fp='path/to/hdf5/file' # (e.g., './nsd_stimuli.hdf5')
    img_ds_key='imgBrick',
    out_path='path/to/features',
    backend=extractor.get_backend(),
    transforms=extractor.get_transformations()
)
```

Features are extracted in the order of the images in the HDF5 file. 

Note, that you can also provide `img_indices` to the `HDF5Dataset` class. This is a list of indices of the images in the HDF5 file that should be loaded. This can be useful if you want to load only a subset of the images in the HDF5 file.
