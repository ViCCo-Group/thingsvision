import os
from typing import Any, List, Tuple

import imageio
import numpy as np
import skimage
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader, ImageDataset
from torch.utils.data import Subset

DATA_PATH = "./data"
TEST_PATH = "./test_images"
OUT_PATH = "./test"

MODEL_AND_MODULE_NAMES = {
    # Torchvision models
    "vgg16": {
        "model_name": "vgg16",
        "modules": ["features.23", "classifier.3"],
        "pretrained": True,
        "source": "torchvision",
    },
    "vgg19_bn": {
        "model_name": "vgg19_bn",
        "modules": ["features.23", "classifier.3"],
        "pretrained": False,
        "source": "torchvision",
    },
    "vit_b_16": {
        "model_name": "vit_b_16",
        "modules": ["encoder.ln"],
        "pretrained": True,
        "source": "torchvision",
        "kwargs": {"token_extraction": "cls_token", "weights": "DEFAULT"},
    },
    # Hardcoded models
    "cornet_r": {
        "model_name": "cornet_r",
        "modules": ["decoder.flatten"],
        "pretrained": True,
        "source": "custom",
    },
    "cornet_rt": {
        "model_name": "cornet_rt",
        "modules": ["decoder.flatten"],
        "pretrained": True,
        "source": "custom",
    },
    "cornet_s": {
        "model_name": "cornet_s",
        "modules": ["decoder.flatten"],
        "pretrained": False,
        "source": "custom",
    },
    "cornet_z": {
        "model_name": "cornet_z",
        "modules": ["decoder.flatten"],
        "pretrained": True,
        "source": "custom",
    },
    # Custom models
    "VGG16_ecoset": {
        "model_name": "VGG16_ecoset",
        "modules": ["classifier.3"],
        "pretrained": True,
        "source": "custom",
    },
    "clip_rn50": {
        "model_name": "clip",
        "modules": ["visual"],
        "pretrained": True,
        "source": "custom",
        "clip": True,
        "kwargs": {"variant": "RN50"},
    },
    "OpenCLIP_vitl14": {
        "model_name": "OpenCLIP",
        "modules": ["visual"],
        "pretrained": True,
        "source": "custom",
        "clip": True,
        "kwargs": {"variant": "ViT-L-14", "dataset": "laion400m_e32"},
    },
    # Timm models
    # "vit_base_patch16_224": {
    #    "model_name": "vit_base_patch16_224",
    #    "modules": ["encoder.ln"],
    #    "pretrained": True,
    #    "source": "timm",
    #    "kwargs": {"token_extraction": "avg_pool"},
    # },
    # "mixnet_l": {
    #    "model_name": "mixnet_l",
    #    "modules": ["conv_head"],
    #    "pretrained": True,
    #    "source": "timm",
    # },
    # Keras models
    "VGG16_keras": {
        "model_name": "VGG16",
        "modules": ["block1_conv1", "flatten"],
        "pretrained": True,
        "source": "keras",
    },
    # Vissl models
    "simclr-rn50": {
        "model_name": "simclr-rn50",
        "modules": ["avgpool"],
        "pretrained": True,
        "source": "ssl",
    },
    "mocov2-rn50": {
        "model_name": "mocov2-rn50",
        "modules": ["avgpool"],
        "pretrained": True,
        "source": "ssl",
    },
    "jigsaw-rn50": {
        "model_name": "jigsaw-rn50",
        "modules": ["avgpool"],
        "pretrained": True,
        "source": "ssl",
    },
    "swav-rn50": {
        "model_name": "swav-rn50",
        "modules": ["avgpool"],
        "pretrained": True,
        "source": "ssl",
    },
    "barlowtwins-rn50": {
        "model_name": "barlowtwins-rn50",
        "modules": ["avgpool"],
        "pretrained": True,
        "source": "ssl",
    },
    "dino-vit-tiny-p8": {
        "model_name": "dino-vit-small-p8",
        "modules": ["norm"],
        "pretrained": True,
        "source": "ssl",
        "kwargs": {"token_extraction": "cls_token+avg_pool"},
    },
    "dino-vit-small-p8": {
        "model_name": "dino-vit-small-p8",
        "modules": ["norm"],
        "pretrained": True,
        "source": "ssl",
        "kwargs": {"token_extraction": "avg_pool"},
    },
    "dino-vit-base-p8": {
        "model_name": "dino-vit-base-p8",
        "modules": ["norm"],
        "pretrained": True,
        "source": "ssl",
        "kwargs": {"token_extraction": "cls_token"},
    },
    "dinov2-vit-small-p14": {
        "model_name": "dinov2-vit-small-p14",
        "modules": ["norm"],
        "pretrained": True,
        "source": "ssl",
        "kwargs": {"token_extraction": "cls_token+avg_pool"},
    },
    "dinov2-vit-base-p14": {
        "model_name": "dinov2-vit-base-p14",
        "modules": ["norm"],
        "pretrained": True,
        "source": "ssl",
        "kwargs": {"extract_cls_token": True},
    },
    "mae-vit-base-p16": {
        "model_name": "mae-vit-base-p16",
        "modules": ["norm", "fc_norm"],
        "pretrained": True,
        "source": "ssl",
        "kwargs": {"token_extraction": "avg_pool"},
    },
    # Additional models
    "Harmonization_visual_ResNet50": {
        "model_name": "Harmonization",
        "modules": ["avg_pool"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "ResNet50"},
    },
    "Harmonization_head_ViT_B16": {
        "model_name": "Harmonization",
        "modules": ["head"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "ViT_B16"},
    },
    "DreamSim_mlp_clip_vitb32": {
        "model_name": "DreamSim",
        "modules": ["model.mlp"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "clip_vitb32"},
    },
    "DreamSim_mlp_dino_vitb16": {
        "model_name": "DreamSim",
        "modules": ["model.mlp"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "dino_vitb16"},
    },
    "SegmentAnything_vit_b": {
        "model_name": "SegmentAnything",
        "modules": ["flatten"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "vit_b"},
        # model input size is large, therefore reduce test case on cpu
        "batch_size": 1,
        "num_samples": 1
    },
    "kakobrain_align_model": {
        "model_name": "Kakaobrain_Align",
        "modules": ["pooler"],
        "pretrained": True,
        "source": "custom",
    },
}

ALIGNED_MODELS = {
    "alexnet": "classifier.4",
    "vgg16": "classifier.3",
    "resnet18": "avgpool",
    "resnet50": "avgpool",
    "clip": "visual",
    "OpenCLIP": "visual",
    "dinov2-vit-base-p14": "norm",
    "dinov2-vit-large-p14": "norm",
    "dino-vit-base-p16": "norm",
    "dino-vit-base-p8": "norm",
}

FILE_FORMATS = ["hdf5", "npy", "mat", "pt", "txt"]
DISTANCES = ["correlation", "cosine", "euclidean", "gaussian"]

BATCH_SIZE = 16
NUM_OBJECTS = 1854
# we want to iterate over two batches to exhaustively test mini-batching
NUM_SAMPLES = int(BATCH_SIZE * 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tf_model = Sequential()
tf_model.add(Dense(2, input_dim=1, activation="relu", use_bias=False, name="relu"))
weights = np.array([[[1, 1]]])
tf_model.get_layer("relu").set_weights(weights)
tf_model.add(Dense(2, input_dim=2, activation="relu", use_bias=False, name="relu2"))
weights = np.array([[[1, 1], [1, 1]]])
tf_model.get_layer("relu2").set_weights(weights)


class NN(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(NN, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_size, 1, bias=False)
        # exchange weight value with 1.
        self.linear.weight = nn.Parameter(torch.tensor([[1.0], [1.0]]))
        self.linear2.weight = nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        act = self.relu(x)
        # print(act)
        y = self.linear2(act)
        act = self.relu2(y)
        # print(y)
        return y


pt_model = NN(1, 2)


class ComplexForwardNN(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(ComplexForwardNN, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_size, 1, bias=False)
        # exchange weight value with 1.
        self.linear.weight = nn.Parameter(torch.tensor([[1.0], [1.0]]))
        self.linear2.weight = nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.relu2 = nn.ReLU()

    def forward(self, x, y):
        x = self.linear(x)
        act = self.relu(x)
        # print(act)
        y = self.linear2(act)
        act = self.relu2(y)
        # print(y)
        return y


class SimpleDataset(object):
    def __init__(self, values: List[int], backend: str):
        self.values = values
        self.backend = backend

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        value = self.values[idx]

        if self.backend == "pt":
            value = torch.tensor([float(value)])
        elif self.backend == "tf":
            value = tf.convert_to_tensor([float(value)])

        return value

    def __len__(self) -> int:
        return len(self.values)


def iterate_through_all_model_combinations():
    for model_config in MODEL_AND_MODULE_NAMES.values():
        model_name = model_config["model_name"]
        pretrained = model_config["pretrained"]
        source = model_config["source"]
        kwargs = model_config.get("kwargs", {})

        # we can set batch size and num_samples to reduce load for large models requiring a large image size
        batch_size = model_config.get("batch_size", BATCH_SIZE)
        num_samples = model_config.get("num_samples", NUM_SAMPLES)

        extractor, dataset, batches = create_extractor_and_dataloader(
            model_name, pretrained, source, kwargs, batch_size, num_samples
        )

        modules = model_config["modules"]
        yield extractor, dataset, batches, modules, model_name


def create_extractor_and_dataloader(
        model_name: str, pretrained: bool, source: str, kwargs: dict = {},
        batch_size: int = BATCH_SIZE, num_samples: int = NUM_SAMPLES
):
    """Iterate through models and create model, dataset and data loader."""
    extractor = get_extractor(
        model_name=model_name,
        pretrained=pretrained,
        device=DEVICE,
        source=source,
        model_parameters=kwargs,
    )
    dataset = ImageDataset(
        root=TEST_PATH,
        out_path=OUT_PATH,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )
    if num_samples > NUM_SAMPLES:
        raise ValueError("\nNumber of samples in test case cannot be larger than the default value, only smaller.\n")
    elif num_samples < NUM_SAMPLES:
        dataset = Subset(dataset, np.arange(num_samples))

    batches = DataLoader(
        dataset,
        batch_size=batch_size,
        backend=extractor.get_backend(),
    )
    return extractor, dataset, batches


def create_test_images(n_samples: int = NUM_SAMPLES) -> None:
    """Create an artificial image dataset to be used for performing tests."""
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    if not os.path.exists(TEST_PATH):
        test_img_1 = skimage.data.hubble_deep_field()
        test_img_2 = skimage.data.coffee()
        test_imgs = list(map(lambda x: x / x.max(), [test_img_1, test_img_2]))

        classes = ["hubble", "coffee"]
        for cls in classes:
            PATH = os.path.join(TEST_PATH, cls)
            if not os.path.exists(PATH):
                os.makedirs(PATH)

        for i in range(n_samples):
            if i > n_samples // 2:
                test_img = np.copy(test_imgs[0])
                cls = classes[0]
            else:
                test_img = np.copy(test_imgs[1])
                cls = classes[1]
            H, W, C = test_img.shape
            # add random Gaussian noise to test image
            noisy_img = test_img + np.random.randn(H, W, C)
            noisy_img = noisy_img.astype(np.uint8)
            imageio.imsave(
                os.path.join(TEST_PATH, cls, f"test_img_{i + 1:03d}.png"), noisy_img
            )
        print("\n...Successfully created image dataset for testing.\n")
