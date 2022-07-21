import unittest

import tests.helper as helper 

import numpy as np 
import tensorflow as tf 
from torchvision import transforms as T
from thingsvision.model_class import Model 

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

        with self.assertRaisesRegex(Exception, "You need to use PyTorch 'pt' as backend if you want to use the CLIP model."):
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

    def test_load_custom_user_model(self):
        model_name = 'VGG16bn_ecoset'
        model = Model(model_name, False, 'cpu')
        self.assertEqual(model.model.__class__.__name__, 'VGG')

        model_name = 'Resnet50_ecoset'
        model = Model(model_name, False, 'cpu')
        self.assertEqual(model.model.__class__.__name__, 'ResNet')

        model_name = 'Alexnet_ecoset'
        model = Model(model_name, False, 'cpu')
        print(model.__class__.__name__)
        self.assertEqual(model.model.__class__.__name__, 'AlexNet')

    def test_load_timm_models(self):
        model_name = 'mixnet_l'
        model = Model(model_name, False, 'cpu')
        self.assertEqual(model.model.__class__.__name__, 'EfficientNet')

