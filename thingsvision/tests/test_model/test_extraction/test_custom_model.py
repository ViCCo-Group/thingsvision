import re
import unittest

import helper
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
        for backend, custom_model, vgg_model in backends:
            dataset = helper.SimpleDataset(values, backend)
            dl = DataLoader(
                dataset,
                batch_size=1,
                backend=backend,
            )
            model = Model(vgg_model, pretrained=False, device=helper.DEVICE, backend=backend)

            model.model = custom_model
            features, _ = model.extract_features(
                dl, layer_name, batch_size=helper.BATCH_SIZE, flatten_acts=False, device=helper.DEVICE)

            expected_features = np.array([[2], [0]])
            np.testing.assert_allclose(features, expected_features)


    

    