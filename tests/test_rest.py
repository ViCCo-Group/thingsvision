import os
import unittest
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

import tests.helper as helper
from thingsvision.core.cka import get_cka
from thingsvision.core.rsa import compute_rdm, correlate_rdms, plot_rdm
from thingsvision.utils.storing import save_features
from thingsvision.core.extraction.torch import ImageOnlyDataloaderModifier


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
            self.assertTrue(isinstance(corr, (float, np.floating)))
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
            batches=batches,
            module_name=module_name_i,
            flatten_acts=False,
            output_type="ndarray",
        )
        model_name_j = "vgg19_bn"
        extractor, _, batches = helper.create_extractor_and_dataloader(
            model_name_j, pretrained=False, source="torchvision"
        )
        module_name = helper.MODEL_AND_MODULE_NAMES[model_name_j]["modules"][1]
        features_j = extractor.extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=False,
            output_type="ndarray",
        )
        self.assertEqual(features_i.shape, features_j.shape)
        m = features_i.shape[0]
        for backend in ["numpy", "torch"]:
            device = "cpu" if backend == "torch" else None
            for kernel in ["linear", "rbf"]:
                sigma = 1.0 if kernel == "rbf" else None
                for debiased in [True, False]:
                    cka = get_cka(
                        backend=backend,
                        m=m,
                        unbiased=debiased,
                        kernel=kernel,
                        sigma=sigma,
                        device=device,
                    )
                    rho = cka.compare(features_i, features_j)
                    if backend == "torch":
                        rho = rho.item()
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


class TestImageOnlyDataloaderModifier(unittest.TestCase):

    def test_context_manager_with_tuple_format_dataloader(self):
        """
        Test 1: Test the context manager with a dataloader that returns (image, label) tuples.
        Should replace collate function and extract only images.
        """
        dataset = helper.MockImageDataset(size=4)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        modifier = ImageOnlyDataloaderModifier(dataloader)

        original_collate_fn = dataloader.collate_fn

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with modifier as modified_dataloader:
                self.assertEqual(len(w), 1)
                self.assertIn(
                    "The dataloader is not in the correct format", str(w[0].message)
                )

                self.assertNotEqual(modified_dataloader.collate_fn, original_collate_fn)
                self.assertEqual(
                    modified_dataloader.collate_fn, modifier.new_collate_fn
                )
                self.assertTrue(modifier.should_replace)

                batch = next(iter(modified_dataloader))
                self.assertIsInstance(batch, torch.Tensor)
                self.assertEqual(batch.shape, (2, 3, 32, 32))

        self.assertEqual(dataloader.collate_fn, original_collate_fn)
        self.assertEqual(modifier.original_collate_fn, original_collate_fn)

    def test_context_manager_with_image_only_dataloader(self):
        """
        Test 2: Test the context manager with a dataloader that already returns only images.
        Should NOT replace collate function since format is already correct.
        """

        dataset = helper.MockImageOnlyDataset(size=4)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        modifier = ImageOnlyDataloaderModifier(dataloader)

        original_collate_fn = dataloader.collate_fn

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with modifier as modified_dataloader:
                assert len(w) == 0
                assert modified_dataloader.collate_fn == original_collate_fn
                assert modifier.should_replace is False
                assert modifier.original_collate_fn is None

                batch = next(iter(modified_dataloader))
                assert isinstance(batch, torch.Tensor)
                assert batch.shape == (2, 3, 32, 32)

        assert dataloader.collate_fn == original_collate_fn

    def test_images_only_collate_function(self):
        """
        Test 3: Test the static _images_only_collate function directly.
        Verify it correctly extracts images from tuples.
        """
        mock_batch = [
            (torch.randn(3, 32, 32), torch.tensor(0)),
            (torch.randn(3, 32, 32), torch.tensor(1)),
            (torch.randn(3, 32, 32), torch.tensor(2)),
        ]

        result = ImageOnlyDataloaderModifier._images_only_collate(mock_batch)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3, 32, 32)

        for i, (original_image, _) in enumerate(mock_batch):
            torch.testing.assert_close(result[i], original_image)

    def test_check_dataloader_format_method(self):
        """
        Bonus Test: Test the _check_dataloader_format method directly.
        """
        dataset_tuple = helper.MockImageDataset(size=2)
        dataloader_tuple = DataLoader(dataset_tuple, batch_size=1)
        modifier_tuple = ImageOnlyDataloaderModifier(dataloader_tuple)
        assert modifier_tuple._check_dataloader_format() is True

        dataset_image = helper.MockImageOnlyDataset(size=2)
        dataloader_image = DataLoader(dataset_image, batch_size=1)
        modifier_image = ImageOnlyDataloaderModifier(dataloader_image)
        assert modifier_image._check_dataloader_format() is False
