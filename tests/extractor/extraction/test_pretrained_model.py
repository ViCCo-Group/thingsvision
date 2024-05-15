import shutil
import unittest
import os

import numpy as np
import tests.helper as helper
from thingsvision.utils.checkpointing import get_torch_home

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
            model_name
        ) in helper.iterate_through_all_model_combinations():
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

                if extractor.get_backend() == "pt":
                    if model_name in helper.ALIGNED_MODELS:
                        if module_name == helper.ALIGNED_MODELS[model_name]:
                            print(f"\nAligning representations extracted from layer: {module_name} of model: {model_name}")
                            aligned_features = extractor.align(features=features, module_name=module_name)
                            print(f"Successfully aligned the representation space of model: {model_name}\n")
                            self.assertTrue(isinstance(aligned_features, Array))
                            self.assertEqual(aligned_features.shape, features.shape)

            # cleanup downloaded torch models
            torch_home = get_torch_home()
            if os.path.exists(torch_home):
                shutil.rmtree(torch_home)
                os.makedirs(torch_home, exist_ok=True)
