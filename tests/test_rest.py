import os
import unittest

import numpy as np
from thingsvision.core.rsa import compute_rdm, correlate_rdms, plot_rdm
from thingsvision.core.cka import CKA
from thingsvision.utils.storing import save_features

import tests.helper as helper

class RSATestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        model_name = "vgg16"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="torchvision"
        )
        module_name = helper.MODEL_AND_MODULE_NAMES[model_name]["modules"][0]
        features = extractor.extract_features(
            batches=batches, module_name=module_name, flatten_acts=False
        )
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )

    def test_rdms(self):
        """Test different distance metrics for RDM computation."""

        features = np.load(os.path.join(helper.OUT_PATH, "features.npy"))
        features = features.reshape(helper.NUM_SAMPLES, -1)

        rdms = []
        for distance in helper.DISTANCES:
            rdm = compute_rdm(features, distance)
            self.assertEqual(len(rdm.shape), 2)
            self.assertEqual(features.shape[0], rdm.shape[0], rdm.shape[1])
            plot_rdm(helper.OUT_PATH, features, distance)
            rdms.append(rdm)

        for rdm_i, rdm_j in zip(rdms[:-1], rdms[1:]):
            corr = correlate_rdms(rdm_i, rdm_j)
            self.assertTrue(isinstance(corr, float))
            self.assertTrue(corr > float(-1) and corr < float(1))



class CKATestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_cka(self):
        """Test CKA for two different models."""
        model_name_i = "vgg16"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name_i, pretrained=False, source="torchvision"
        )
        module_name_i = helper.MODEL_AND_MODULE_NAMES[model_name_i]["modules"][1]
        features_i = extractor.extract_features(
            batches=batches, module_name=module_name_i, flatten_acts=False
        )
        model_name_j = "vgg19_bn"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name_j, pretrained=False, source="torchvision"
        )
        module_name = helper.MODEL_AND_MODULE_NAMES[model_name_j]["modules"][1]
        features_j = extractor.extract_features(
            batches=batches, module_name=module_name, flatten_acts=False
        )
        self.assertEqual(features_i.shape, features_j.shape)
        m = features_i.shape[0]
        for kernel in ['linear', 'rbf']:
            cka = CKA(m=m, kernel=kernel, sigma=0.5 if kernel == 'rbf' else None)
            rho = cka.compare(features_i, features_j)
            self.assertTrue(isinstance(rho, float))
            self.assertTrue(rho > float(-1) and rho < float(1))


class FileNamesTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_filenames(self):
        with open(os.path.join(helper.OUT_PATH, "file_names.txt"), "r") as f:
            file_names = f.read().split()
        img_files = []
        for root, _, files in os.walk(helper.TEST_PATH):
            for f in files:
                if f.endswith("png"):
                    img_files.append(os.path.join(root, f))
        self.assertEqual(sorted(file_names), sorted(img_files))
