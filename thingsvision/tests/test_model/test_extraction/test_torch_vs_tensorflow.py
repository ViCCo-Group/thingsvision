import unittest

import helper
from thingsvision.model_class import Model
from thingsvision.dataloader import DataLoader

import numpy as np 

class ExtractionPTvsTFTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_custom_torch_vs_tf_extraction(self):
        layer_name = 'relu'
        values = [2, -10]
        backend = 'tf'
        tf_dataset = helper.SimpleDataset(values, backend)
        tf_dl = DataLoader(
            tf_dataset,
            batch_size=1,
            backend=backend,
        )

        model = Model('VGG16', pretrained=False,
                      device=helper.DEVICE, backend=backend)
        model.model = helper.tf_model
        tf_features, _ = model.extract_features(
            tf_dl, layer_name, batch_size=helper.BATCH_SIZE, flatten_acts=False, device=helper.DEVICE)

        backend = 'pt'
        pt_dataset = helper.SimpleDataset(values, backend)
        pt_dl = DataLoader(
            pt_dataset,
            batch_size=1,
            backend=backend,
        )
        model = Model('vgg16', pretrained=False,
                      device=helper.DEVICE, backend=backend)
        model.model = helper.pt_model
        pt_features, _ = model.extract_features(
            pt_dl, layer_name, batch_size=helper.BATCH_SIZE, flatten_acts=False, device=helper.DEVICE)
        np.testing.assert_allclose(tf_features, pt_features)

        expected_features = np.array([[2], [0]])
        np.testing.assert_allclose(pt_features, expected_features)
