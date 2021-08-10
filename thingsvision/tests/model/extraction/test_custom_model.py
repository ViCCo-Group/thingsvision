import re
import unittest

import thingsvision.tests.helper as helper 
from thingsvision.dataloader import DataLoader
from thingsvision.model_class import Model

import numpy as np 

class ExtractionTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_custom_model(self):
        layer_name = 'relu'
        values = [2, -10]
        backends = [['pt', helper.pt_model, 'vgg16'], ['tf', helper.tf_model, 'VGG16']]
        batch_size = 1
        for backend, custom_model, vgg_model in backends:
            dataset = helper.SimpleDataset(values, backend)
            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                backend=backend,
            )
            model = Model(vgg_model, pretrained=False,
                        device=helper.DEVICE, backend=backend)

            model.model = custom_model
            expected_features = np.array([[2], [0]])
            expected_targets = np.array([0, 0])

            features, targets = model.extract_features(dl, 
                                                    layer_name, 
                                                    batch_size=batch_size, 
                                                    flatten_acts=False, 
                                                    return_probabilities=False)
            np.testing.assert_allclose(features, expected_features)
            np.testing.assert_allclose(targets, expected_targets)


            expected_probs = np.array([[1], [1]])
            features, targets, probs = model.extract_features(dl, 
                                                            layer_name, 
                                                            batch_size=batch_size, 
                                                            flatten_acts=False, 
                                                            return_probabilities=True)
            np.testing.assert_allclose(features, expected_features)
            np.testing.assert_allclose(targets, expected_targets)
            np.testing.assert_allclose(probs, expected_probs)


    

    