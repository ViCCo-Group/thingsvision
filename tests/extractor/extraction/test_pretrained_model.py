import re
import unittest

import numpy as np
import tests.helper as helper

Array = np.ndarray


class ExtractionPretrainedTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_extraction_pretrained_models(self):
        """Tests basic feature extraction pipeline."""
        for (
            extractor,
            dataset,
            batches,
            module_names,
            model_name,
            clip
        ) in helper.iterate_through_all_model_combinations():
            self.assertEqual(len(dataset), len(batches) * helper.BATCH_SIZE)
            num_objects = len(dataset)

            for module_name in module_names:
                features = extractor.extract_features(
                    batches=batches,
                    module_name=module_name,
                    flatten_acts=False,
                )

                self.assertTrue(isinstance(features, Array))
                self.assertEqual(features.shape[0], num_objects)

                if module_name.startswith("classifier"):
                    self.assertEqual(
                        features.shape[1],
                        extractor.model.classifier[int(module_name[-1])].out_features,
                    )

                if not clip:
                    features = extractor.extract_features(
                        batches=batches,
                        module_name=module_name,
                        flatten_acts=False,
                    )

                    self.assertTrue(isinstance(features, Array))
                    self.assertEqual(features.shape[0], num_objects)
