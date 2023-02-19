import unittest

import tensorflow as tf
import tests.helper as helper
from torchvision import transforms as T


class ModelTransformationsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

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

    def test_transformations_transformer(self):
        model_name = "dino-xcit-medium-24-p16"
        extractor, _, _ = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="ssl"
        )
        transforms = extractor.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))
        
        model_name = "dino-vit-small-p8"
        extractor, _, _ = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="ssl"
        )
        transforms = extractor.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))
