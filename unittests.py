#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import pandas as pd
import thingsvision.vision as vision

from thingsvision.dataset import ImageDataset
from torch.utils.data import DataLoader

DEVICE = 'cpu'
IN_PATH = './images92'
OUT_PATH = './test'
MODEL_NAMES = ['vgg16_bn', 'vgg19_bn']
MODULE_NAMES = ['features.23', 'classifier.3']
FILE_FORMATS = ['.npy', '.txt']
BATCH_SIZE = 32

class ModelLoadingTestCase(unittest.TestCase):

    def test_mode_and_device(self):
        model, _ = vision.load_model(
                                     model_name=MODEL_NAMES[0],
                                     pretrained=True,
                                     device=DEVICE,
                                     )
        self.assertTrue(hasattr(model, DEVICE))
        self.assertFalse(model.training)

class ExtractionTestCase(unittest.TestCase):

    def test_extraction(self):
        model, transforms = vision.load_model(
                                                model_name=MODEL_NAMES[0],
                                                pretrained=True,
                                                device=DEVICE,
        )
        dataset = ImageDataset(
                                root=IN_PATH,
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
            vision.save_features(
                                features=features,
                                out_path=OUT_PATH,
                                file_format=format,
            )
        self.assertEqual(features.shape[0], len(dataset))
        self.assertEqual(len(targets), features.shape[0])

        if MODULE_NAMES[0].startswith('classifier'):
            self.assertEqual(features.shape[1], model.classifier[int(MODULE_NAMES[0][-1])].out_features)

        self.assertTrue(isinstance(features, np.ndarray))
        self.assertTrue(isinstance(targets, np.ndarray))

class SlicingTestCase(unittest.TestCase):

    def test_slicing(self):
        features_npy = np.load(os.path.join(OUT_PATH, 'features.npy'))
        features_txt = vision.slices2tensor(OUT_PATH, 'features.txt')

        self.assertEqual(features_npy.shape, features_txt.shape)

class ComparisonTestCase(unittest.TestCase):

    def test_comparison(self):

        corr_mat = vision.compare_models(
                                        root=IN_PATH,
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

if __name__ == '__main__':
    unittest.main()
