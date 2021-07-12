#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imageio
import os
import re
import skimage
import shutil
from typing import Tuple, List, Dict, Iterator, Any
import unittest

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import torch
import tensorflow as tf

import thingsvision.vision as vision
from thingsvision.model_class import Model
from thingsvision.dataset import ImageDataset
from thingsvision.dataloader import DataLoader

DATA_PATH = './data'
TEST_PATH = './test_images'
OUT_PATH = './test'

PT_MODEL_AND_MODULE_NAMES = {
    'vgg16_bn': ['features.23', 'classifier.3'],
    'vgg19_bn': ['features.23', 'classifier.3'],
    'cornet_r': ['decoder.flatten'],
    'cornet_rt': ['decoder.flatten'],
    'cornet_s': ['decoder.flatten'], 
    'cornet_z': ['decoder.flatten'],
    'clip-ViT': ['visual'],
    'clip-RN': ['visual']
}

TF_MODEL_AND_MODULES_NAMES = {
    'VGG16': ['block1_conv1', 'flatten'],
    'VGG19': ['block1_conv1', 'flatten']
}

CLIP = [True if re.search(r'^clip', model_name)
        else False for model_name in PT_MODEL_AND_MODULE_NAMES]

FILE_FORMATS = ['hdf5', 'npy', 'mat', 'txt']
DISTANCES = ['correlation', 'cosine', 'euclidean', 'gaussian']

BATCH_SIZE = 16
NUM_OBJECTS = 1854
# we want to iterate over two batches to exhaustively test mini-batching
NUM_SAMPLES = int(BATCH_SIZE * 2)
DEVICE = 'cpu'
BACKENDS = ['pt', 'tf']

class SimpleDataset():
    def __init__(self, values, backend):
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


class ModelLoadingTestCase(unittest.TestCase):

    def test_mode_and_device(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = create_model_and_dl(model_name, 'pt')
        self.assertTrue(hasattr(model.model, DEVICE))
        self.assertFalse(model.model.training)

def iterate_through_all_model_combinations():
    for backend in BACKENDS:
        MODEL_AND_MODULE_NAMES = None
        if backend == 'pt':
            MODEL_AND_MODULE_NAMES = PT_MODEL_AND_MODULE_NAMES
        elif backend == 'tf':
            MODEL_AND_MODULE_NAMES = TF_MODEL_AND_MODULES_NAMES

        for model_name in MODEL_AND_MODULE_NAMES:
            model, dataset, dl = create_model_and_dl(model_name, backend)
            yield model, dataset, dl, MODEL_AND_MODULE_NAMES[model_name], model_name

def create_model_and_dl(model_name, backend):
    """Iterate through all backends and models and create model, dataset and data loader."""
    model = Model(
                model_name=model_name,
                pretrained=True,
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

class ExtractionTestCase(unittest.TestCase):

    def test_extraction_pretrained_modells(self):
        """Tests basic feature extraction pipeline."""
        for model, dataset, dl, module_names, model_name in iterate_through_all_model_combinations():
            clip = re.search(r'^clip', model_name)

            for module_name in module_names:
                self.assertEqual(len(dataset), len(dl) * BATCH_SIZE)

                features, targets = model.extract_features(
                    data_loader=dl,
                    module_name=module_name,
                    batch_size=BATCH_SIZE,
                    flatten_acts=False,
                    device=DEVICE,
                    clip=clip
                )

                self.assertTrue(isinstance(features, np.ndarray))
                self.assertTrue(isinstance(targets, np.ndarray))
                self.assertEqual(features.shape[0], len(dataset))
                self.assertEqual(len(targets), features.shape[0])

                if module_name.startswith('classifier'):
                    self.assertEqual(features.shape[1],
                                    model.model.classifier[int(module_name[-1])].out_features)

    def test_extraction_custom_model(self):
        layer_name = 'relu'
        values = [2, -10]

        backend = 'tf'
        tf_dataset = SimpleDataset(values, backend)

        tf_dl = DataLoader(
                    tf_dataset,
                    batch_size=1,
                    backend=backend,
        )

        tf_model = Sequential()
        tf_model.add(Dense(1, input_dim=1, activation='relu', use_bias=False, name='relu'))
        weights = np.array([[[1]]])
        tf_model.get_layer('relu').set_weights(weights)
        model = Model('VGG16', pretrained=False, device='cpu', backend=backend)
        model.model = tf_model
        tf_features, _ = model.extract_features(tf_dl, layer_name, batch_size=BATCH_SIZE, flatten_acts=False, device=DEVICE)
        expected_features = np.array([[2], [0]])
        assert_allclose(tf_features, expected_features)

    def test_postprocessing(self):
        """Test different postprocessing methods (e.g., centering, normalization, compression)."""
        model_name = 'vgg16_bn'
        model, dataset, dl = create_model_and_dl(model_name, 'pt')
        module_name = PT_MODEL_AND_MODULE_NAMES[model_name][0]
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=BATCH_SIZE,
            flatten_acts=False,
            device=DEVICE,
        )
        flattened_features = features.reshape(NUM_SAMPLES, -1)
        centred_features = vision.center_features(flattened_features)
        normalized_features = vision.normalize_features(flattened_features)
        transformed_features = vision.compress_features(
            flattened_features, rnd_seed=42, retained_var=.9)

        self.assertTrue(centred_features.mean(axis=0).sum() < 1e-3)
        self.assertEqual(np.linalg.norm(normalized_features, axis=1).sum(),
                         np.ones(features.shape[0]).sum())
        self.assertTrue(
            transformed_features.shape[1] < flattened_features.shape[1])

    def test_storing(self):
        """Test storing possibilities."""
        model_name = 'vgg16_bn'
        model, dataset, dl = create_model_and_dl(model_name, 'pt')
        module_name = PT_MODEL_AND_MODULE_NAMES[model_name][0]
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=BATCH_SIZE,
            flatten_acts=False,
            device=DEVICE,
        )
        for format in FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            vision.save_features(
                features=features,
                out_path=OUT_PATH,
                file_format=format,
            )

    def test_same_extraction_tf_torch(self):
        layer_name = 'relu'
        values = [2, -10]

        backend = 'tf'
        tf_dataset = SimpleDataset(values, backend)

        tf_dl = DataLoader(
                    tf_dataset,
                    batch_size=1,
                    backend=backend,
        )

        tf_model = Sequential()
        tf_model.add(Dense(1, input_dim=1, activation='relu', use_bias=False, name='relu'))
        weights = np.array([[[1]]])
        tf_model.get_layer('relu').set_weights(weights)
        model = Model('VGG16', pretrained=False, device='cpu', backend=backend)
        model.model = tf_model
        tf_features, tf_targets = model.extract_features(tf_dl, layer_name, batch_size=BATCH_SIZE, flatten_acts=False, device=DEVICE)

        backend = 'pt'
        pt_dataset = SimpleDataset(values, backend)

        pt_dl = DataLoader(
                    pt_dataset,
                    batch_size=1,
                    backend=backend,
        )

        class NeuralNetwork(torch.nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.linear = torch.nn.Linear(1,1, bias=False)
                    self.relu = torch.nn.ReLU()
                    

                def forward(self, x):
                    print('input %s' % x)
                    with torch.no_grad():
                        self.linear.weight = torch.nn.Parameter(torch.tensor([1.]))
                    print('weight %s' % self.linear.weight)

                    x = self.linear(x)
                    act = self.relu(x)
                    print('act %s' % act)
                    return act

        pt_model = NeuralNetwork()
            
        model = Model('vgg16', pretrained=False, device='cpu', backend=backend)
        model.model = pt_model
        pt_features, pt_target = model.extract_features(pt_dl, layer_name, batch_size=BATCH_SIZE, flatten_acts=False, device=DEVICE)
        assert_allclose(tf_features, pt_features)



class RDMTestCase(unittest.TestCase):

    def test_rdms(self):
        """Test different distance metrics on which RDMs are based."""

        features = np.load(os.path.join(OUT_PATH, 'features.npy'))
        features = features.reshape(NUM_SAMPLES, -1)
  
        rdms = []
        for distance in DISTANCES:
            rdm = vision.compute_rdm(features, distance)
            self.assertEqual(len(rdm.shape), 2)
            self.assertEqual(features.shape[0], rdm.shape[0], rdm.shape[1])
            vision.plot_rdm(OUT_PATH, features, distance)
            rdms.append(rdm)

        for rdm_i, rdm_j in zip(rdms[:-1], rdms[1:]):
            corr = vision.correlate_rdms(rdm_i, rdm_j)
            self.assertTrue(isinstance(corr, float))
            self.assertTrue(corr > float(-1))
            self.assertTrue(corr < float(1))


class ComparisonTestCase(unittest.TestCase):

    def test_comparison(self):
        backend = 'pt'
        compare_model_names = ['vgg16_bn', 'vgg19_bn']
        compare_module_names = ['features.23', 'classifier.3']
        corr_mat = vision.compare_models(
            root=TEST_PATH,
            out_path=OUT_PATH,
            model_names=compare_model_names,
            module_names=compare_module_names,
            pretrained=True,
            batch_size=BATCH_SIZE,
            backend=backend,
            flatten_acts=True,
            clip=CLIP,
            save_features=False,
        )
        self.assertTrue(isinstance(corr_mat, pd.DataFrame))
        self.assertEqual(corr_mat.shape, (len(compare_model_names), len(compare_module_names)))


class LoadItemsTestCase(unittest.TestCase):

    def test_item_name_loading(self):
        item_names = vision.load_item_names(DATA_PATH)
        self.assertTrue(isinstance(item_names, np.ndarray))
        self.assertEqual(len(item_names), NUM_OBJECTS)


class FileNamesTestCase(unittest.TestCase):

    def test_filenames(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = create_model_and_dl(model_name, 'pt')
        module_name = PT_MODEL_AND_MODULE_NAMES[model_name][0]
        file_names = open(os.path.join(
            OUT_PATH, 'file_names.txt'), 'r').read().split()
        img_files = []
        for root, _, files in os.walk(TEST_PATH):
            for f in files:
                if f.endswith('png'):
                    img_files.append(os.path.join(root, f))
        self.assertEqual(sorted(file_names), sorted(img_files))


def create_test_images(n_samples: int) -> None:
    """Create an artificial image dataset to be used for performing tests."""
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
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


if __name__ == '__main__':
    create_test_images(NUM_SAMPLES)
    unittest.main()
    shutil.rmtree(TEST_PATH)
    shutil.rmtree(OUT_PATH)
