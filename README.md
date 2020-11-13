#### Environment Setup

1. The code uses Python 3.8,  [Pytorch 1.6.0](https://pytorch.org/) (Note that PyTorch 1.6.0 requires CUDA 10.2, if you want to run code on a GPU)
2. Install PyTorch and `torchvision`: `pip install pytorch` and `pip install torchvision` or `conda install pytorch torchvision -c pytorch` and `conda install -c pytorch torchvision` (the latter way is recommended if you use Anaconda)
3. Install Python dependencies: `pip install -r requirements.txt` (only necessary, if you don't use Anaconda)

#### Extract hidden unit activations at specific layer of a state-of-the-art torchvision model 

```
  python extract.py
  
 --model_name (str) (PyTorch vision model for which neural activations should be extracted)
 --interactive (bool) (whether or not to interact with terminal, and choose model part after looking at model architecture in terminal)
 --module_name (str) (if in non-interactive mode, then module name for which hidden unit activations should be extracted must be provided)
 --flatten_acts (bool) (whether or not to flatten activations at lower layers of the model (e.g., convoluatonal layers) before saving them)
 --batch_size (int) (neural activations will be extracted for a batch of image samples; set number of images per mini-batch)
 --things (bool) (specify whether images are from the THINGS images database or not; if they are make sure to first load images from the THINGS image database into in_path)
 --fraction (float) (specify fraction of dataset to be used, if you do not want to extract neural activations for *all* images)
 --file_format (str) (whether to store activations as .txt or .npy files; note that the latter is more memory efficient but requires NumPy)
 --in_path (str) (directory from where to load images)
 --out_path (str) (directory where neural activations should be saved)
 --model_path (str) (directory where to load torchvision model weights from; weights won't be compute on the fly and must be stored on disk)
 --device (str) (CPU or CUDA)
 --rnd_seed (int) (random seed)
```

Here is an example call for `interactive` mode:

```
python extract.py --model_name alexnet --interactive --flatten_acts --batch_size 32 --things --file_format .txt --in_path ./images/ --out_path ./activations/ --device cuda:1 --rnd_seed 42
```


After you've called `extract.py` and all arguments have been parsed, you will see your torchvision model of choice printed like that:

```
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

Then you will be prompted to interact with the terminal, and subsequently enter the part of the model for which you'd like to extract hidden unit activations.
Note that each `torchvision` model (e.g., AlexNet) is split into different parts (e.g., `features`, `avgpool`, `classifier`) which are all enumerated separately.
Hence, you have to specify the part of the model like `features.11`, `avgpool`, or `classifier.4`. The rest will be done automatically for you.

Here is an example call for `non-interactive` mode (useful for `bash` scripts on a `job-system` such as `Slurm`):

```
python extract.py --model_name alexnet --module_name classifier.4 --batch_size 32 --things --in_path ./images/ --out_path ./activations/ --device cuda:1 --rnd_seed 42
```

#### IMPORTANT NOTES:

1. Image data will automatically be converted into a ready-to-use dataset class, and subsequently wrapped with a `PyTorch` mini-batch dataloader to make neural activation extraction more efficient.

2. If you happen to use the [THINGS image database](https://osf.io/jum2f/), make sure to correctly `unzip` all zip files, and have all `object` directories stored in the parent directory `./images/` (e.g., `./images/object_xy/`) as well as the `things_concept.tsv` file stored in the `./data/` folder. The script will automatically check, whether you have done the latter correctly. 

3. In case you would like to use your own images or a different dataset make sure that all images are `.jpg`, `.png`, or `.PNG` files. Image files must be saved either in `in_path` (e.g., `./images/image_xy.jpg`), or in subfolders of `in_path` (e.g., `./images/class_xy/image_xy.jpg`) in case images correspond to different classes where `n` images are stored for each of the `k` classes (such as in ImageNet or THINGS). You don't need to tell the script in which of the two ways your images are stored. You just need to pass `in_path`. However, images have to be stored in one way or the other.

4. Hidden unit activations can be extracted at each layer for both `features` and `classifier` for the following `torchvision` models: `alexnet`, `resnet50`, `resnet101`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`

5. The script automatically extracts hidden unit activations for the specified `model` and `layer` and stores them together with the `targets` in `out_path` (see above).

6. Since 4-way tensors cannot be easily saved to disk, they must be sliced into different parts to be efficiently stored as a matrix. The helper function `tensor2slices` will slice any 4-way tensor (activations extraced from `features.##`) automatically for you, and will save it as a matrix in a file called `activations.txt`. To merge the slices back into the original shape (i.e., 4-way tensor) simply call `slices2tensor` which takes `out_path` and `file_name` (see above) as input arguments (e.g., `tensor = slices2tensor(PATH, file)`).

7. If you happen to extract hidden unit activations for many images, it is possible to run into `MemoryErrors`. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 

