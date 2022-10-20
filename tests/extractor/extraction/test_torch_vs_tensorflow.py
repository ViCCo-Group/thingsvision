import unittest

import numpy as np
import tests.helper as helper
from thingsvision.utils.data import DataLoader
import thingsvision.core.extraction.helpers as core_helpers



class ExtractionPTvsTFTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_custom_torch_vs_tf_extraction(self):
        values = [2, -10]

        tf_backend = "tf"
        tf_source = "keras"
        tf_dataset = helper.SimpleDataset(values, tf_backend)
        tf_dl = DataLoader(
            dataset=tf_dataset,
            batch_size=1,
            backend=tf_backend,
        )
        tf_model = core_helpers.get_extractor(
            "VGG16", pretrained=False, device=helper.DEVICE, source=tf_source
        )
        tf_model.model = helper.tf_model
        tf_model.backend = tf_backend

        pt_backend = "pt"
        pt_source = "torchvision"
        pt_dataset = helper.SimpleDataset(values, pt_backend)
        pt_dl = DataLoader(
            dataset=pt_dataset,
            batch_size=1,
            backend=pt_backend,
        )
        pt_model = core_helpers.get_extractor(
            "vgg16", pretrained=False, device=helper.DEVICE, source=pt_source
        )
        pt_model.model = helper.pt_model
        pt_model.backend = pt_backend

        layer_name = "relu"
        tf_features = tf_model.extract_features(
            batches=tf_dl,
            module_name=layer_name,
            flatten_acts=False,
        )
        pt_features = pt_model.extract_features(
            batches=pt_dl,
            module_name=layer_name,
            flatten_acts=False,
        )
        expected_features = np.array([[2, 2], [0, 0]])
        np.testing.assert_allclose(pt_features, expected_features)
        np.testing.assert_allclose(tf_features, expected_features)

        layer_name = "relu2"
        tf_features = tf_model.extract_features(
            batches=tf_dl,
            module_name=layer_name,
            flatten_acts=False,
        )
        pt_features = pt_model.extract_features(
            batches=pt_dl,
            module_name=layer_name,
            flatten_acts=False,
        )
        expected_features = np.array([[4, 4], [0, 0]])
        np.testing.assert_allclose(pt_features, expected_features)
        np.testing.assert_allclose(tf_features, expected_features)
