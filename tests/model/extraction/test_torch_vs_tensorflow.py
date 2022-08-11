import unittest

import tests.helper as helper 
from thingsvision.model_class import Model
from thingsvision.dataloader import DataLoader

import numpy as np 

class ExtractionPTvsTFTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_custom_torch_vs_tf_extraction(self):
        values = [2, -10]

        tf_backend = 'tf'
        tf_dataset = helper.SimpleDataset(values, tf_backend)
        tf_dl = DataLoader(
            tf_dataset,
            batch_size=1,
            backend=tf_backend,
        )
        tf_model = Model('VGG16', pretrained=False,
                      device=helper.DEVICE)
        tf_model.model = helper.tf_model
        tf_model.backend = tf_backend

        torch_backend = 'pt'
        pt_dataset = helper.SimpleDataset(values, torch_backend)
        pt_dl = DataLoader(
            pt_dataset,
            batch_size=1,
            backend=torch_backend,
        )
        pt_model = Model('vgg16', pretrained=False,
                      device=helper.DEVICE)
        pt_model.model = helper.pt_model
        pt_model.backend = torch_backend

        layer_name = 'relu'
        tf_features, _ = tf_model.extract_features(
            data_loader=tf_dl,
            module_name=layer_name,
            flatten_acts=False,
            )
        pt_features, _ = pt_model.extract_features(
            data_loader=pt_dl,
            module_name=layer_name,
            flatten_acts=False,
            )
        expected_features = np.array([[2, 2], [0, 0]])
        np.testing.assert_allclose(pt_features, expected_features)
        np.testing.assert_allclose(tf_features, expected_features)

        layer_name = 'relu2'
        tf_features, _ = tf_model.extract_features(
            data_loader=tf_dl,
            module_name=layer_name,
            flatten_acts=False,
            )
        pt_features, _ = pt_model.extract_features(
            data_loader=pt_dl,
            module_name=layer_name,
            flatten_acts=False,
            )
        expected_features = np.array([[4, 4], [0, 0]])
        np.testing.assert_allclose(pt_features, expected_features)
        np.testing.assert_allclose(tf_features, expected_features)
