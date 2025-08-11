import os
import shutil
import unittest

import numpy as np
import torch

import tests.helper as helper
from thingsvision.core.extraction import center_features, normalize_features
from thingsvision.utils.storing import save_features, split_features


class FeaturesTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def setUp(self):
        shutil.rmtree(helper.OUT_PATH)
        os.makedirs(helper.OUT_PATH)

    def get_2D_features(self):
        model_name = "vgg16_bn"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name=model_name, pretrained=False, source="torchvision"
        )
        module_name = "classifier.3"
        features = extractor.extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=False,
        )
        return features

    def get_4D_features(self):
        model_name = "vgg16_bn"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name=model_name, pretrained=False, source="torchvision"
        )
        module_name = "features.23"
        features = extractor.extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=False,
        )
        return features

    def get_multi_features(self):
        model_name = "vgg16_bn"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name=model_name, pretrained=False, source="torchvision"
        )
        module_names = ["features.23", "classifier.3"]
        features = extractor.extract_features(
            batches=batches,
            module_names=module_names,
            flatten_acts=False,
        )
        return features

    def test_postprocessing(self):
        """Test different postprocessing methods (e.g., centering, normalization, compression)."""
        features = self.get_2D_features()
        flattened_features = features.reshape(helper.NUM_SAMPLES, -1)
        centred_features = center_features(flattened_features)
        normalized_features = normalize_features(flattened_features)
        self.assertEqual(
            np.linalg.norm(normalized_features, axis=1).sum(),
            np.ones(features.shape[0]).sum(),
        )

    def check_file_exists(self, file_name, format, txt_should_exists=True):
        if format == "hdf5":
            file_name = "features"
        path = os.path.join(helper.OUT_PATH, f"{file_name}.{format}")
        if format == "txt" and not txt_should_exists:
            self.assertTrue(not os.path.exists(path))
        else:
            self.assertTrue(os.path.exists(path))

    def test_storing_2d(self):
        """Test storing possibilities."""
        features = self.get_2D_features()
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as two-dimensional tensor
            save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )

            self.check_file_exists("features", format)

    def test_storing_4d(self):
        features = self.get_4D_features()
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )

            self.check_file_exists("features", format, False)

    def test_storing_multi(self):
        features = self.get_multi_features()
        for _, feature in features.items():
            for format in set(helper.FILE_FORMATS) - set(["txt"]):
                # tests whether features can be saved in any of the formats except txt
                save_features(
                    features=feature,
                    out_path=helper.OUT_PATH,
                    file_format=format,
                )
                self.check_file_exists(f"features", format, False)

    def test_extract_multi(self):
        features = self.get_multi_features()
        row_counts = [feature.shape[0] for feature in features.values()]
        self.assertTrue(
            all(count == row_counts[0] for count in row_counts),
            "Not all features have the same number of rows!",
        )

    def test_splitting_2d(self):
        n_splits = 3
        features = self.get_2D_features()
        for format in helper.FILE_FORMATS:
            if format == "pt":
                features = torch.from_numpy(features)
            split_features(
                features=features,
                root=helper.OUT_PATH,
                file_format=format,
                n_splits=n_splits,
            )

            for i in range(1, n_splits):
                self.check_file_exists(f"features_{i:02d}", format)

    def test_splitting_4d(self):
        n_splits = 3
        features = self.get_4D_features()
        for format in set(helper.FILE_FORMATS) - set(["txt"]):
            if format == "pt":
                features = torch.from_numpy(features)
            split_features(
                features=features,
                root=helper.OUT_PATH,
                file_format=format,
                n_splits=n_splits,
            )

            for i in range(1, n_splits):
                self.check_file_exists(f"features_{i:02d}", format, False)

        with self.assertRaises(Exception):
            split_features(
                features=features,
                root=helper.OUT_PATH,
                file_format="txt",
                n_splits=n_splits,
            )

    def test_splitting_multi(self):
        n_splits = 3
        features = self.get_multi_features()
        for format in set(helper.FILE_FORMATS) - set(["txt"]):
            for _, feature in features.items():
                if format == "pt":
                    feature = torch.from_numpy(feature)
                split_features(
                    features=feature,
                    root=helper.OUT_PATH,
                    file_format=format,
                    n_splits=n_splits,
                )

                for i in range(1, n_splits):
                    self.check_file_exists(f"features_{i:02d}", format, False)

        with self.assertRaises(Exception):
            for _, feature in features.items():
                split_features(
                    features=feature,
                    root=helper.OUT_PATH,
                    file_format="txt",
                    n_splits=n_splits,
                )
