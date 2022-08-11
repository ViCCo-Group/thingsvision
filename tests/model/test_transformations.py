import unittest

import tests.helper as helper 

import numpy as np 
import tensorflow as tf 
from torchvision import transforms as T
from thingsvision.model_class import Model 

class ModelTransformationsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_transformations_clip(self):
        model_name = 'clip-RN'
        model, dataset, dl = helper.create_model_and_dl(model_name)
        transforms = model.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))

        with self.assertRaisesRegex(Exception, "You need to use PyTorch 'pt' as backend if you want to use the CLIP model."):
            model.backend = 'tf'
            transforms = model.get_transformations()

    def test_transformations_cnn(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name)
        transforms = model.get_transformations()
        self.assertTrue(isinstance(transforms, T.Compose))

        model_name = 'VGG16'
        model, dataset, dl = helper.create_model_and_dl(model_name)
        transforms = model.get_transformations()
        self.assertTrue(isinstance(transforms, tf.keras.Sequential))