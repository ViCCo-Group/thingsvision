import unittest

import tensorflow as tf
import tests.helper as helper
from torchvision import transforms as T


class ModelTransformationsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_transformations_clip(self):
        model_name = "clip-RN"
        extractor, _, _ = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="custom"
        )
        transforms = extractor.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))

        with self.assertRaisesRegex(
            Exception,
            "You need to use PyTorch as backend if you want to use a CLIP model.",
        ):
            extractor.backend = "tf"
            transforms = extractor.get_transformations()

    def test_transformations_cnn(self):
        model_name = "vgg16_bn"
        extractor, _, _ = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="torchvision"
        )
        transforms = extractor.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))

        model_name = "VGG16"
        extractor, _, _ = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="keras"
        )
        transforms = extractor.get_transformations()
        self.assertTrue(isinstance(transforms, tf.keras.Sequential))
