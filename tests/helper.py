import os
import re
from typing import Tuple, List, Any

import thingsvision.vision 
from thingsvision.dataloader import DataLoader
from thingsvision.dataset import ImageDataset
from thingsvision.model_class import Model

import numpy as np

import skimage
import imageio

import torch 
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

DATA_PATH = './data'
TEST_PATH = './test_images'
OUT_PATH = './test'

PT_MODEL_AND_MODULE_NAMES = {
    # Torchvision models
    'vgg16_bn': {
        'modules': ['features.23', 'classifier.3'],
        'pretrained': True
    },
    'vgg19_bn': {
        'modules': ['features.23', 'classifier.3'],
        'pretrained': False
    },

    # Hardcoded models
    'cornet_r': {
        'modules': ['decoder.flatten'],
        'pretrained': True
    },
    'cornet_rt': {
        'modules': ['decoder.flatten'],
        'pretrained': False
    },
    'cornet_s': {
        'modules': ['decoder.flatten'],
        'pretrained': False
    },
    'cornet_z': {
        'modules': ['decoder.flatten'],
        'pretrained': False
    },
    'clip-ViT': {
        'modules': ['visual'],
        'pretrained': True
    },
    'clip-RN': {
        'modules': ['visual'],
        'pretrained': False
    },

    # Custom models
    'VGG16bn_ecoset': {
        'modules': ['classifier.3'],
        'pretrained': True
    },

    # Timm models
    'mixnet_l': {
        'modules': ['conv_head'],
        'pretrained': True
    },
    'gluon_inception_v3': {
        'modules': ['Mixed_6d'],
        'pretrained': False
    }
}

TF_MODEL_AND_MODULES_NAMES = {
    # Keras models
    'VGG16': {
        'modules': ['block1_conv1', 'flatten'],
        'pretrained': True
    },
    'VGG19': {
        'modules': ['block1_conv1', 'flatten'],
        'pretrained': False
    }
}

BACKENDS = ['tf', 'pt']

FILE_FORMATS = ['hdf5', 'npy', 'mat', 'txt']
DISTANCES = ['correlation', 'cosine', 'euclidean', 'gaussian']

BATCH_SIZE = 16
NUM_OBJECTS = 1854
# we want to iterate over two batches to exhaustively test mini-batching
NUM_SAMPLES = int(BATCH_SIZE * 2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


tf_model = Sequential()
tf_model.add(Dense(2, input_dim=1, activation='relu',
             use_bias=False, name='relu'))
weights = np.array([[[1,1]]])
tf_model.get_layer('relu').set_weights(weights)
tf_model.add(Dense(2, input_dim=2, activation='relu',
             use_bias=False, name='relu2'))
weights = np.array([[[1,1], [1,1]]])
tf_model.get_layer('relu2').set_weights(weights)


class NN(nn.Module):

    def __init__(self, in_size: int, out_size: int):
        super(NN, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_size, 1, bias=False)
        # exchange weight value with 1.
        self.linear.weight = nn.Parameter(torch.tensor([[1.], [1.]]))
        self.linear2.weight = nn.Parameter(torch.tensor([[1., 1.], [1., 1.]]))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        act = self.relu(x)
        print(act)
        y = self.linear2(act)
        act = self.relu2(y)
        print(y)
        return y


pt_model = NN(1, 2)


class SimpleDataset(object):

    def __init__(self, values: List[int], backend: str):
        self.values = values
        self.backend = backend

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        target = 0
        value = self.values[idx]

        if self.backend == 'pt':
            value = torch.tensor([float(value)])
            target = torch.tensor([target])
        elif self.backend == 'tf':
            value = tf.convert_to_tensor([float(value)])
            target = tf.convert_to_tensor([target])

        return value, target

    def __len__(self) -> int:
        return len(self.values)

def iterate_through_all_model_combinations():
    for backend in BACKENDS:
        MODEL_AND_MODULE_NAMES = None
        if backend == 'pt':
            MODEL_AND_MODULE_NAMES = PT_MODEL_AND_MODULE_NAMES
        elif backend == 'tf':
            MODEL_AND_MODULE_NAMES = TF_MODEL_AND_MODULES_NAMES

        for model_name in MODEL_AND_MODULE_NAMES:
            pretrained = MODEL_AND_MODULE_NAMES[model_name]['pretrained']
            model, dataset, dl = create_model_and_dl(model_name, backend, pretrained)

            modules =  MODEL_AND_MODULE_NAMES[model_name]['modules']
            yield model, dataset, dl, modules, model_name


def create_model_and_dl(model_name, backend, pretrained=False):
    """Iterate through all backends and models and create model, dataset and data loader."""
    model = Model(
        model_name=model_name,
        pretrained=pretrained,
        device=DEVICE,
        backend=backend
    )

    dataset = ImageDataset(
        root=TEST_PATH,
        out_path=OUT_PATH,
        backend=backend,
        imagenet_train=None,
        imagenet_val=None,
        things=None,
        things_behavior=None,
        add_ref_imgs=None,
        transforms=model.get_transformations()
    )
    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        backend=backend,
    )
    return model, dataset, dl

def create_test_images(n_samples: int = NUM_SAMPLES) -> None:
    """Create an artificial image dataset to be used for performing tests."""
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    if not os.path.exists(TEST_PATH):
        test_img_1 = skimage.data.hubble_deep_field()
        test_img_2 = skimage.data.coffee()
        test_imgs = list(map(lambda x: x / x.max(), [test_img_1, test_img_2]))

        classes = ['hubble', 'coffee']
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
            imageio.imsave(os.path.join(
                TEST_PATH, cls, f'test_img_{i+1:03d}.png'), noisy_img)
        print('\n...Successfully created image dataset for testing.\n')

def download_item_names():
    if not os.path.isfile(os.path.join(DATA_PATH, 'item_names.tsv')):
        try:
            os.system(
                "curl -O https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/get_files.sh")
            os.system("bash get_files.sh")
        except:
            raise RuntimeError(
                "Download the THINGS item names <tsv> file to run test;\n"
                "See README.md for details."
            )
