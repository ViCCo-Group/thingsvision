---
layout: default
title: Low memory options
nav_order: 6
---

# Low memory options

There are several reasons why you could run into memory issues while using `thingsvision`:

## Running out of GPU memory

When extracting features on GPU, the default batch size might be too large for your GPU. Try reducing the batch size by setting the `batch_size` parameter in the `DataLoader` to a smaller value.

Alternatively, you can also run the extraction on CPU. This will be slower, but you can use a larger batch size. To do so, set the `device` parameter in the `get_extractor` function to `'cpu'`.

## Running out of RAM

As all features are stored in RAM while the extraction is running, you might run out of RAM if you extract features for a large number of images. To avoid this, you can instead write them directly to disk by setting the `output_dir` parameter in the `extract_features` function to a directory of your choice. This will write the features to disk as they are extracted, freeing up RAM. The `step_size` parameter can be used to specify how many batches are extracted before the features are written to disk. For the default, we set it so that it uses about 8GB of RAM. 

Usage example:
```python
# get extractor and dataloader 
extractor = ...
batches = ...

output_dir = '/path/to/output/directory'
extractor.extract_features(
    batches=batches,
    module_name=...,
    flatten_acts=True,
    output_dir=output_dir
) # returns None if output_dir is set
```

## Running into `MemoryError` while storing features to disk
If you happen to extract activations for many images, which do fit into RAM, it is still possible to run into `MemoryErrors` when saving the extracted features to disk. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 
