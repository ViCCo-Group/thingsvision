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
from thingsvision.utils.data import DataLoader, ImageDataset
from thingsvision import get_extractor

DATA_PATH = "./data"
TEST_PATH = "./test_images"
OUT_PATH = "./test"

SSL_RN50_DEFAULT_CONFIG = {
    "modules": ["avgpool"],
    "pretrained": True,
    "source": "ssl",
}

MODEL_AND_MODULE_NAMES = {
    # Torchvision models
    "vgg16": {
        "modules": ["features.23", "classifier.3"],
        "pretrained": True,
        "source": "torchvision",
    },
    "vgg19_bn": {
        "modules": ["features.23", "classifier.3"],
        "pretrained": False,
        "source": "torchvision",
    },
    # Hardcoded models
    "cornet_r": {
        "modules": ["decoder.flatten"],
        "pretrained": True,
        "source": "custom",
    },
    "cornet_rt": {
        "modules": ["decoder.flatten"],
        "pretrained": False,
        "source": "custom",
    },
    "cornet_s": {
        "modules": ["decoder.flatten"],
        "pretrained": False,
        "source": "custom",
    },
    "cornet_z": {
        "modules": ["decoder.flatten"],
        "pretrained": True,
        "source": "custom",
    },
    # Custom models
    "VGG16_ecoset": {
        "modules": ["classifier.3"],
        "pretrained": True,
        "source": "custom",
    },
    "clip": {
        "modules": ["visual"],
        "pretrained": True,
        "source": "custom",
        "clip": True,
        "kwargs": {"variant": "ViT-B/32"},
    },
    "clip": {
        "modules": ["visual"],
        "pretrained": True,
        "source": "custom",
        "clip": True,
        "kwargs": {"variant": "RN50"},
    },
    "OpenCLIP": {
        "modules": ["visual"],
        "pretrained": True,
        "source": "custom",
        "clip": True,
        "kwargs": {"variant": "ViT-B-32", "dataset": "openai"},
    },
    # Timm models
    "mixnet_l": {"modules": ["conv_head"], "pretrained": True, "source": "timm"},
    "gluon_inception_v3": {
        "modules": ["Mixed_6d"],
        "pretrained": False,
        "source": "timm",
    },
    # Keras models
    "VGG16": {
        "modules": ["block1_conv1", "flatten"],
        "pretrained": True,
        "source": "keras",
    },
    "VGG19": {
        "modules": ["block1_conv1", "flatten"],
        "pretrained": False,
        "source": "keras",
    },
    # Vissl models
    'simclr-rn50': SSL_RN50_DEFAULT_CONFIG,
    'mocov2-rn50': SSL_RN50_DEFAULT_CONFIG,
    'jigsaw-rn50': SSL_RN50_DEFAULT_CONFIG,
    'rotnet-rn50': SSL_RN50_DEFAULT_CONFIG,
    'swav-rn50': SSL_RN50_DEFAULT_CONFIG,
    'pirl-rn50': SSL_RN50_DEFAULT_CONFIG,
    'barlowtwins-rn50': SSL_RN50_DEFAULT_CONFIG,
    'vicreg-rn50': SSL_RN50_DEFAULT_CONFIG,
    'dino-rn50' : SSL_RN50_DEFAULT_CONFIG,
    # Harmonization models
    "Harmonization": {
        "modules": ["visual"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "ResNet50"},
    },
    "Harmonization": {
        "modules": ["fc2"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "VGG16"},
    },
     "Harmonization": {
        "modules": ["head"],
        "pretrained": True,
        "source": "custom",
        "kwargs": {"variant": "ViT_B16"},
    }
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
        target = 0
        value = self.values[idx]

        if self.backend == "pt":
            value = torch.tensor([float(value)])
        elif self.backend == "tf":
            value = tf.convert_to_tensor([float(value)])

        return value

    def __len__(self) -> int:
        return len(self.values)


def iterate_through_all_model_combinations():
    for model_name in MODEL_AND_MODULE_NAMES:
        pretrained = MODEL_AND_MODULE_NAMES[model_name]["pretrained"]
        source = MODEL_AND_MODULE_NAMES[model_name]["source"]
        kwargs = MODEL_AND_MODULE_NAMES[model_name].get("kwargs", {})
        extractor, dataset, batches = create_extractor_and_dataloader(
            model_name, pretrained, source, kwargs
        )

        modules = MODEL_AND_MODULE_NAMES[model_name]["modules"]
        clip = MODEL_AND_MODULE_NAMES[model_name].get("clip", False)
        yield extractor, dataset, batches, modules, model_name, clip


def create_extractor_and_dataloader(
    model_name: str, pretrained: bool, source: str, kwargs: dict = {}
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
    batches = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
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
                os.path.join(TEST_PATH, cls, f"test_img_{i+1:03d}.png"), noisy_img
            )
        print("\n...Successfully created image dataset for testing.\n")
