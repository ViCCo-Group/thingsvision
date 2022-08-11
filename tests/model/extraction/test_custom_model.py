import re
import unittest
from tests import helper  
from thingsvision.dataloader import DataLoader
from thingsvision.model_class import Model

import numpy as np 

class ExtractionCustomModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_extract_features(self):
        layer_name = 'relu'
        values = [2, -10]
        backends = [['pt', helper.pt_model, 'vgg16'], ['tf', helper.tf_model, 'VGG16']]
        batch_size = 1
        for backend, custom_model, vgg_model in backends:
            model = Model(vgg_model, pretrained=False,
                        device=helper.DEVICE)
            model.backend = backend
            model.model = custom_model

            dataset = helper.SimpleDataset(values, backend)
            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                backend=backend,
            )
            
            expected_features = np.array([[2, 2], [0, 0]])
            expected_targets = np.array([0, 0])

            features, targets = model.extract_features(
                data_loader=dl, 
                module_name=layer_name, 
                flatten_acts=False, 
                return_probabilities=False)
            np.testing.assert_allclose(features, expected_features)
            np.testing.assert_allclose(targets, expected_targets)


            expected_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
            features, targets, probs = model.extract_features(
                data_loader=dl, 
                module_name=layer_name,
                flatten_acts=False,
                return_probabilities=True)
            np.testing.assert_allclose(features, expected_features)
            np.testing.assert_allclose(targets, expected_targets)
            np.testing.assert_allclose(probs, expected_probs)

    def test_extraction_batches(self):
        values = [1] * 10
        backend = 'pt'
        dataset = helper.SimpleDataset(values, backend)
        model = Model('vgg16', pretrained=False, device=helper.DEVICE)
        model.backend = backend
        model.model = helper.pt_model

        # no batch remainders -> 5 batches with 2 examples
        # batch remainders -> 3 batches with 3 examples and 1 batch with 1 remainder

        for batch_size in [2, 3]:
            dl = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    backend=backend,
            )
            features, targets = model.extract_features(
                data_loader=dl, 
                module_name='relu',
                flatten_acts=False, 
                return_probabilities=False)
            self.assertEqual(features.shape[0], len(dataset))
            self.assertEqual(targets.shape[0], len(dataset))
       
        
