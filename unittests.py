#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imageio
import os
import skimage
import shutil
import unittest

import numpy as np
import pandas as pd
import thingsvision.vision as vision

from thingsvision.dataset import ImageDataset
from torch.utils.data import DataLoader

DEVICE = 'cpu'
TEST_PATH = './test_images'
OUT_PATH = './test'
MODEL_NAMES = ['vgg16_bn', 'vgg19_bn']
MODULE_NAMES = ['features.23', 'classifier.3']
FILE_FORMATS = ['hdf5', 'npy', 'mat', 'txt']
BATCH_SIZE = 32

class ModelLoadingTestCase(unittest.TestCase):

    def test_mode_and_device(self):
        """Tests whether model is on DEVICE and in evaluation (opposed to training) mode."""
        model, _ = vision.load_model(
                                     model_name=MODEL_NAMES[0],
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

        features, targets = vision.extract_features(
                                                    model=model,
                                                    data_loader=dl,
                                                    module_name=MODULE_NAMES[0],
                                                    batch_size=BATCH_SIZE,
                                                    flatten_acts=False,
                                                    device=DEVICE,
        )

        for format in FILE_FORMATS:
            # tests whether features can be saved in any of the formats
            vision.save_features(
                                features=features,
                                out_path=OUT_PATH,
                                file_format=format,
            )
        self.assertEqual(len(dataset), len(dl)*batch_size)
        self.assertEqual(features.shape[0], len(dataset))
        self.assertEqual(len(targets), features.shape[0])

        if MODULE_NAMES[0].startswith('classifier'):
            self.assertEqual(features.shape[1], model.classifier[int(MODULE_NAMES[0][-1])].out_features)

        self.assertTrue(isinstance(features, np.ndarray))
        self.assertTrue(isinstance(targets, np.ndarray))


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
                                        clip=[False, False],
                                        save_features=False,
        )
        self.assertTrue(isinstance(corr_mat, pd.DataFrame))
        self.assertEqual(corr_mat.shape, (len(MODEL_NAMES), len(MODULE_NAMES)))


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
