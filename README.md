## Environment Setup

1. Make sure you have the latest Python version (>= 3.8) and [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/). Note that [PyTorch 1.7.1](https://pytorch.org/) requires CUDA 10.2 or above, if you want to extract features on a GPU. However, the code runs pretty fast on a strong CPU (Intel i7 or i9). 

2. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install -r requirements.txt
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine (e.g., 10.2) or `cpuonly` when installing on a machine without a GPU.


## Extract features at specific layer of a state-of-the-art torchvision or CLIP model 

```
  python extract.py
  
 --model_name (str) (PyTorch vision model for which neural activations should be extracted)
 --interactive (bool) (whether or not to interact with terminal, and choose model part after looking at model architecture in terminal)
 --module_name (str) (if in non-interactive mode, then module name for which hidden unit activations should be extracted must be provided)
 --flatten_acts (bool) (whether or not to flatten features at lower layers of the model (e.g., convoluatonal layers) before saving them)
 --center_acts (bool) (whether or not to center features (move their mean towards zero))
 --normalize_reps (bool) (whether or not to normalize object representations by their l2-norms)
 --compress_acts (bool) (whether or not to transform features into lower-dimensional space via PCA (only relevant for feature ensembles))
 --batch_size (int) (neural activations will be extracted for a batch of image samples; set number of images per mini-batch)
 --things (bool) (specify whether images are from the THINGS images database or not; if they are make sure to first load images from the THINGS image database into in_path)
 --pretrained (bool) (specify whether to download a pretrained torchvision or CLIP model from network; if not provided, model_path has to be specified)
 --fraction (float) (specify fraction of dataset to be used, if you do not want to extract neural activations for *all* images)
 --file_format (str) (whether to store activations as .txt or .npy files; note that the latter is both more memory and time efficient but requires NumPy)
 --in_path (str) (directory from where to load images)
 --out_path (str) (directory where features should be saved)
 --model_path (str) (directory where to load torchvision model weights from)
 --device (str) (CPU or CUDA)
 --rnd_seed (int) (random seed)
```

Here is an example call for `interactive` mode:

```
python extract.py --model_name alexnet --interactive --flatten_acts --pretrained --batch_size 64 --things --file_format .txt --in_path ./images/ --out_path ./activations/ --device cuda --rnd_seed 42
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
python extract.py --model_name alexnet --module_name classifier.4 --pretrained --batch_size 32 --things --in_path ./images/ --out_path ./activations/ --device cuda --rnd_seed 42
```

## IMPORTANT NOTES:

1. Image data will automatically be converted into a ready-to-use dataset class, and subsequently wrapped with a `PyTorch` mini-batch dataloader to make neural activation extraction more efficient.

2. If you happen to use the [THINGS image database](https://osf.io/jum2f/), make sure to correctly `unzip` all zip files, and have all `object` directories stored in the parent directory `./images/` (e.g., `./images/object_xy/`) as well as the `things_concept.tsv` file stored in the `./data/` folder. The script will automatically check, whether you have done the latter correctly. 

3. In case you would like to use your own images or a different dataset make sure that all images are `.jpg`, `.png`, or `.PNG` files. Image files must be saved either in `in_path` (e.g., `./images/image_xy.jpg`), or in subfolders of `in_path` (e.g., `./images/class_xy/image_xy.jpg`) in case images correspond to different classes where `n` images are stored for each of the `k` classes (such as in ImageNet or THINGS). You don't need to tell the script in which of the two ways your images are stored. You just need to pass `in_path`. However, images have to be stored in one way or the other.

4. Features can be extracted at every layer for both `features` and `classifier` for the following `torchvision` models: `alexnet`, `resnet50`, `resnet101`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`, and additionally for OpenAi's `CLIP` models `RN50` and `ViT-32`.

5. If you happen to be interested in an ensemble of `feature maps`, as introduced in this recent [COLING 2020 paper](https://www.aclweb.org/anthology/2020.coling-main.173/), you can simply extract an ensemble of `conv` or `max-pool` layers. The ensemble can additionally be concatenated with the activations of the penultimate layer, and subsequently transformed into a lower-dimensional space with `PCA` to reduce noise and only keep those dimensions that account for most of the variance. 

6. The script automatically extracts features for the specified `model` and `layer` and stores them together with the `targets` in `out_path` (see above).

7. Since 4-way tensors cannot be easily saved to disk, they must be sliced into different parts to be efficiently stored as a matrix. The helper function `tensor2slices` will slice any 4-way tensor (activations extraced from `features.##`) automatically for you, and will save it as a matrix in a file called `activations.txt`. To merge the slices back into the original shape (i.e., 4-way tensor) simply call `slices2tensor` which takes `out_path` and `file_name` (see above) as input arguments (e.g., `tensor = slices2tensor(PATH, file)`).

8. If you happen to extract hidden unit activations for many images, it is possible to run into `MemoryErrors`. To circumvent such problems, a helper function called `split_activations` will split the activation matrix into several batches, and stores them in separate files. For now, the split parameter is set to `10`. Hence, the function will split the activation matrix into `10` files. This parameter can, however, easily be modified in case you need more (or fewer) splits. To merge the separate activation batches back into a single activation matrix, just call `merge_activations` when loading the activations (e.g., `activations = merge_activations(PATH)`). 

## OpenAI's CLIP models (read carefully)


### CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.


## API

The CLIP module `clip` provides the following methods:

#### `clip.available_models()`

Returns the name(s) of the available CLIP models.

#### `clip.load(name, device=..., jit=True)`

Returns the model and the TorchVision transform needed by the model, specified by the model name returned by `clip.available_models()`. It will download the model as necessary. The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU.

When `jit` is `False`, a non-JIT version of the model will be loaded.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a LongTensor containing tokenized sequences of given text input(s). This can be used as the input to the model

---

The model returned by `clip.load()` supports the following methods:

#### `model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.

#### `model.encode_text(text: Tensor)`

Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

#### `model(image: Tensor, text: Tensor)`

Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.
