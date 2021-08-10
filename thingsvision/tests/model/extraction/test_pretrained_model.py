import os
import re
import unittest

import thingsvision.tests.helper as helper 
import thingsvision.vision as vision
from thingsvision.dataloader import DataLoader
from thingsvision.model_class import Model

import numpy as np 

class ExtractionPretrainedTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_extraction_pretrained_modells(self):
        """Tests basic feature extraction pipeline."""
        for model, dataset, dl, module_names, model_name in helper.iterate_through_all_model_combinations():
            clip = re.search(r'^clip', model_name)
            self.assertEqual(len(dataset), len(dl) * helper.BATCH_SIZE)
            amount_objects = len(dataset)

            for module_name in module_names:
                features, targets = model.extract_features(
                        data_loader=dl,
                        module_name=module_name,
                        batch_size=helper.BATCH_SIZE,
                        flatten_acts=False,
                        clip=clip,
                        return_probabilities=False
                )

                self.assertTrue(isinstance(features, np.ndarray))
                self.assertTrue(isinstance(targets, np.ndarray))
                self.assertEqual(features.shape[0], amount_objects)
                self.assertEqual(len(targets), amount_objects)

                if module_name.startswith('classifier'):
                    self.assertEqual(features.shape[1],
                                     model.model.classifier[int(module_name[-1])].out_features)

                if not clip:
                    features, targets, probs = model.extract_features(
                        data_loader=dl,
                        module_name=module_name,
                        batch_size=helper.BATCH_SIZE,
                        flatten_acts=False,
                        clip=clip,
                        return_probabilities=True
                    )

                    self.assertTrue(isinstance(features, np.ndarray))
                    self.assertTrue(isinstance(targets, np.ndarray))
                    self.assertEqual(features.shape[0], amount_objects)
                    self.assertEqual(len(targets), amount_objects)
                    self.assertEqual(len(probs), amount_objects)

    def test_extraction_across_models(self):
        backend = 'pt'
        model_names = ['vgg16_bn', 'vgg19_bn']
        module_names = ['features.23', 'features.23']
        img_paths = [helper.TEST_PATH, helper.TEST_PATH]
        vision.extract_features_across_models_and_datasets(helper.OUT_PATH, model_names, img_paths, module_names, [False, False], False, 1, backend, False)
        path1 = os.path.join(helper.OUT_PATH, helper.TEST_PATH, model_names[0],
                                module_names[0], 'features')
        self.assertTrue(os.path.exists(path1))
        path2 = os.path.join(helper.OUT_PATH, helper.TEST_PATH, model_names[1],
                                module_names[1], 'features')
        self.assertTrue(os.path.exists(path2))

    def test_extraction_across_models_and_modules(self):
        backend = 'pt'
        model_names = ['vgg16_bn', 'alexnet']
        module_names = ['features.22', 'features.1']
        img_paths = [helper.TEST_PATH, helper.TEST_PATH]
        vision.extract_features_across_models_datasets_and_modules(helper.OUT_PATH, model_names, img_paths, module_names, [False, False], False, 1, backend, False)
        path1 = os.path.join(helper.OUT_PATH, helper.TEST_PATH, model_names[0],
                                module_names[0], 'features')
        self.assertTrue(os.path.exists(path1))
        path2 = os.path.join(helper.OUT_PATH, helper.TEST_PATH, model_names[1],
                                module_names[1], 'features')
        self.assertTrue(os.path.exists(path2))
        
       