import unittest

import numpy as np
import torch

import tests.helper as helper
import thingsvision.core.extraction.helpers as core_helpers
from thingsvision.utils.data import DataLoader


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
        expected_features_tf = np.array([[2.0, 2.0], [0, 0.0]])
        expected_features_pt = torch.tensor([[2.0, 2.0], [0.0, 0.0]])

        for i, batch in enumerate(tf_dl):
            tf_features = tf_model.extract_batch(
                batch=batch,
                module_name=layer_name,
                flatten_acts=False,
            )
            expected_features = expected_features_tf[i][None, :]
            np.testing.assert_allclose(tf_features, expected_features)

        with pt_model.batch_extraction(layer_name, output_type="tensor") as e:
            for i, batch in enumerate(pt_dl):
                pt_features = e.extract_batch(
                    batch=batch,
                    flatten_acts=False,
                )
                expected_features = expected_features_pt[i][None, :]
                np.testing.assert_allclose(pt_features, expected_features)

        layer_name = "relu2"
        expected_features = np.array([[4.0, 4.0], [0.0, 0.0]])
        for i, batch in enumerate(tf_dl):
            tf_features = tf_model.extract_batch(
                batch=batch,
                module_name=layer_name,
                flatten_acts=False,
            )
            np.testing.assert_allclose(tf_features, expected_features[i][None, :])

        with pt_model.batch_extraction(layer_name, output_type="ndarray") as e:
            for i, batch in enumerate(pt_dl):
                pt_features = e.extract_batch(
                    batch=batch,
                    flatten_acts=False,
                )
                np.testing.assert_allclose(pt_features, expected_features[i][None, :])
