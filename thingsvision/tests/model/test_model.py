import unittest

import thingsvision.tests.helper as helper 

import numpy as np 
import tensorflow as tf 
from torchvision import transforms as T

class ModelLoadingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_mode_and_device(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        self.assertTrue(hasattr(model.model, helper.DEVICE))
        self.assertFalse(model.model.training)

    def test_transformations_clip(self):
        model_name = 'clip-RN'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        transforms = model.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))

        with self.assertRaisesRegex(Exception, "You need to use Tensorflow 'tf' as backend if you want to use the CLIP model."):
            model.backend = 'tf'
            transforms = model.get_transformations()

    def test_transformations_cnn(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        transforms = model.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))

        model_name = 'VGG16'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'tf')
        transforms = model.get_transformations()
        self.assertTrue(isinstance(transforms, tf.keras.Sequential))

