#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imageio
import os
import re
import skimage
import shutil
import unittest

import numpy as np
import pandas as pd
import thingsvision.vision as vision

from thingsvision.dataset import ImageDataset
from torch.utils.data import DataLoader

DATA_PATH = './data'
TEST_PATH = './test_images'
OUT_PATH = './test'

MODEL_NAMES = ['vgg16_bn', 'vgg19_bn', 'cornet_r', 'cornet_rt', 'cornet_s', 'cornet_z', 'clip-ViT', 'clip-RN']
MODULE_NAMES = ['features.23', 'classifier.3', 'decoder.flatten', 'decoder.flatten', 'decoder.flatten', 'decoder.flatten', 'visual', 'visual']
CLIP = [True if re.search(r'^clip', model_name) else False for model_name in MODEL_NAMES]

FILE_FORMATS = ['hdf5', 'npy', 'mat', 'txt']
DISTANCES = ['correlation', 'cosine', 'euclidean', 'gaussian']

BATCH_SIZE = 32
NUM_OBJECTS = 1854
DEVICE = 'cpu'

if not os.path.isfile(os.path.join(DATA_PATH, 'item_names.tsv')):
    raise RuntimeError(
        "Download the THINGS item names <tsv> file to run test;\n"
        "See README.md for details."
    )

class ModelLoadingTestCase(unittest.TestCase):

    def test_mode_and_device(self):
        """Tests whether model is on DEVICE and in evaluation (opposed to training) mode."""
        for model_name in MODEL_NAMES:
            model, _ = vision.load_model(
                                         model_name=model_name,
                                         pretrained=True,
                                         device=DEVICE,
                                         )
            self.assertTrue(hasattr(model, DEVICE))
            self.assertFalse(model.training)

class ExtractionTestCase(unittest.TestCase):

    def test_extraction(self):
        """Tests basic feature extraction pipeline."""
        model, transforms = vision.load_model(
                                                model_name=MODEL_NAMES[0],
                                                pretrained=True,
                                                device=DEVICE,
        )
        dataset = ImageDataset(
                                root=TEST_PATH,
                                out_path=OUT_PATH,
                                transforms=transforms,
                                imagenet_train=None,
                                imagenet_val=None,
                                things=None,
                                things_behavior=None,
                                add_ref_imgs=None,
        )
        dl = DataLoader(
                        dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
        )

        self.assertEqual(len(dataset), len(dl)*BATCH_SIZE)

        global features
        features, targets = vision.extract_features(
                                                    model=model,
                                                    data_loader=dl,
                                                    module_name=MODULE_NAMES[0],
                                                    batch_size=BATCH_SIZE,
                                                    flatten_acts=False,
                                                    device=DEVICE,
        )

        self.assertTrue(isinstance(features, np.ndarray))
        self.assertTrue(isinstance(targets, np.ndarray))
        self.assertEqual(features.shape[0], len(dataset))
        self.assertEqual(len(targets), features.shape[0])

        if MODULE_NAMES[0].startswith('classifier'):
            self.assertEqual(features.shape[1], model.classifier[int(MODULE_NAMES[0][-1])].out_features)


    def test_postprocessing(self):
        """Test different postprocessing methods (e.g., centering, normalization, compression)."""
        flattened_features = features.reshape(BATCH_SIZE, -1)
        centred_features = vision.center_features(flattened_features)
        normalized_features = vision.normalize_features(flattened_features)
        transformed_features = vision.compress_features(flattened_features, rnd_seed=42, retained_var=.9)

        self.assertTrue(centred_features.mean(axis=0).sum() < 1e-5)
        self.assertEqual(np.linalg.norm(normalized_features, axis=1).sum(), np.ones(features.shape[0]).sum())
        self.assertTrue(transformed_features.shape[1] < flattened_features.shape[1])


    def test_storing(self):
        """Test storing possibilities."""
        for format in FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            vision.save_features(
                                features=features,
                                out_path=OUT_PATH,
                                file_format=format,
            )

class RDMTestCase(unittest.TestCase):

    def test_rdms(self):
        """Test different distance metrics on which RDMs are based."""

        features = np.load(os.path.join(OUT_PATH, 'features.npy'))
        features = features.reshape(BATCH_SIZE, -1)

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

        corr_mat = vision.compare_models(
                                        root=TEST_PATH,
                                        out_path=OUT_PATH,
                                        model_names=MODEL_NAMES,
                                        module_names=MODULE_NAMES,
                                        pretrained=True,
                                        batch_size=BATCH_SIZE,
                                        flatten_acts=True,
                                        clip=CLIP,
                                        save_features=False,
        )
        self.assertTrue(isinstance(corr_mat, pd.DataFrame))
        self.assertEqual(corr_mat.shape, (len(MODEL_NAMES), len(MODULE_NAMES)))

class LoadItemsTestCase(unittest.TestCase):

    def test_item_name_loading(self):
        item_names = vision.load_item_names(DATA_PATH)
        self.assertTrue(isinstance(item_names, np.ndarray))
        self.assertEqual(len(item_names), NUM_OBJECTS)


def create_test_images(n_samples: int) -> None:
    """Create image dataset to be used for performing tests."""
    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)
    test_img_1 = skimage.data.hubble_deep_field()
    test_img_2 = skimage.data.coffee()
    test_imgs = list(map(lambda x: x / x.max(), [test_img_1, test_img_2]))
    for i in range(n_samples):
        if (i + 1) % (n_samples // 2) == 0:
            test_img = np.copy(test_imgs[0])
        else:
            test_img = np.copy(test_imgs[1])
        H, W, C = test_img.shape
        # add random Gaussian noise to test image
        noisy_img = test_img + np.random.randn(H, W, C)
        noisy_img = noisy_img.astype(np.uint8)
        imageio.imsave(os.path.join(TEST_PATH, f'test_img_{i+1:03d}.png'), noisy_img)
    print('\n...Successfully created image dataset for testing.\n')


if __name__ == '__main__':
    create_test_images(BATCH_SIZE)
    unittest.main()
    shutil.rmtree(TEST_PATH)
