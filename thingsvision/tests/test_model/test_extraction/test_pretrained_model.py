import re
import unittest

import helper
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

            for module_name in module_names:
                self.assertEqual(len(dataset), len(dl) * helper.BATCH_SIZE)

                features, targets = model.extract_features(
                    data_loader=dl,
                    module_name=module_name,
                    batch_size=helper.BATCH_SIZE,
                    flatten_acts=False,
                    device=helper.DEVICE,
                    clip=clip
                )

                self.assertTrue(isinstance(features, np.ndarray))
                self.assertTrue(isinstance(targets, np.ndarray))
                self.assertEqual(features.shape[0], len(dataset))
                self.assertEqual(len(targets), features.shape[0])

                if module_name.startswith('classifier'):
                    self.assertEqual(features.shape[1],
                                     model.model.classifier[int(module_name[-1])].out_features)
