import unittest

import numpy as np
from tests import helper
from thingsvision.utils.data import DataLoader
from thingsvision.core.extraction.helpers import get_extractor


class ExtractionCustomModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_extract_features(self):
        layer_name = "relu"
        values = [2, -10]
        backends = [
            ["pt", helper.pt_model, "vgg16", "torchvision"],
            ["tf", helper.tf_model, "VGG16", "keras"],
        ]
        expected_features = np.array([[2, 2], [0, 0]])
        batch_size = 1
        for backend, custom_model, vgg_model, source in backends:
            extractor = get_extractor(
                vgg_model, pretrained=False, device=helper.DEVICE, source=source
            )
            extractor.backend = backend
            extractor.model = custom_model
            if backend == "pt":
                extractor.model = extractor.model.to(helper.DEVICE)
            dataset = helper.SimpleDataset(values, backend)
            batches = DataLoader(
                dataset,
                batch_size=batch_size,
                backend=backend,
            )
            features = extractor.extract_features(
                batches=batches,
                module_name=layer_name,
                flatten_acts=False,
            )
            np.testing.assert_allclose(features, expected_features)

    def test_extraction_batches(self):
        values = [1] * 10
        backend = "pt"
        dataset = helper.SimpleDataset(values, backend)
        extractor = get_extractor(
            "vgg16", pretrained=False, device=helper.DEVICE, source="torchvision"
        )
        extractor.backend = backend
        extractor.model = helper.pt_model
        if backend == "pt":
            extractor.model = extractor.model.to(helper.DEVICE)
        # no batch remainders -> 5 batches with 2 examples
        # batch remainders -> 3 batches with 3 examples and 1 batch with 1 remainder

        for batch_size in [2, 3]:
            batches = DataLoader(
                dataset,
                batch_size=batch_size,
                backend=backend,
            )
            features = extractor.extract_features(
                batches=batches,
                module_name="relu",
                flatten_acts=False,
            )
            self.assertEqual(features.shape[0], len(dataset))
