import unittest

import helper
import thingsvision.vision as vision
import numpy as np 

class PostExtractionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_postprocessing(self):
        """Test different postprocessing methods (e.g., centering, normalization, compression)."""
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        module_name = helper.PT_MODEL_AND_MODULE_NAMES[model_name][0]
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=helper.BATCH_SIZE,
            flatten_acts=False,
            device=helper.DEVICE,
        )
        flattened_features = features.reshape(helper.NUM_SAMPLES, -1)
        centred_features = vision.center_features(flattened_features)
        normalized_features = vision.normalize_features(flattened_features)
        transformed_features = vision.compress_features(
            flattened_features, rnd_seed=42, retained_var=.9)

        self.assertTrue(centred_features.mean(axis=0).sum() < 1e-3)
        self.assertEqual(np.linalg.norm(normalized_features, axis=1).sum(),
                         np.ones(features.shape[0]).sum())
        self.assertTrue(
            transformed_features.shape[1] < flattened_features.shape[1])

    def test_storing(self):
        """Test storing possibilities."""
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        module_name = helper.PT_MODEL_AND_MODULE_NAMES[model_name][0]
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=helper.BATCH_SIZE,
            flatten_acts=False,
            device=helper.DEVICE,
        )
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            vision.save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )