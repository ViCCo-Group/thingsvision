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

    def check_model_loading(self, model_name, expected_class_name, source=None):
        model = Model(model_name, False, 'cpu', source=source)
        self.assertEqual(model.model.__class__.__name__, expected_class_name)

    def check_unknown_model_loading(self, model_name, expected_exception, source=None):
        with self.assertRaises(Exception) as e:
            model = Model(model_name, False, 'cpu', source=source)
            self.assertEqual(e.exception, expected_exception)

    def test_mode_and_device(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name)
        self.assertTrue(hasattr(model.model, helper.DEVICE))
        self.assertFalse(model.model.training)

    def test_load_model_without_source(self):
        model_name = 'vgg16' # PyTorch
        self.check_model_loading(model_name, 'VGG')

        model_name = 'VGG16' # Tensorflow
        self.check_model_loading(model_name, 'Functional')

        model_name = 'random'
        self.check_unknown_model_loading(model_name, f'Model {model_name} not found in all sources')

    def test_load_custom_user_model(self):
        source = 'custom'

        model_name = 'VGG16bn_ecoset'
        self.check_model_loading(model_name, 'VGG', source)

        model_name = 'Resnet50_ecoset'
        self.check_model_loading(model_name, 'ResNet', source)

        model_name = 'Alexnet_ecoset'
        self.check_model_loading(model_name, 'AlexNet', source)

        model_name = 'random'
        self.check_unknown_model_loading(model_name, f'Model {model_name} not found in {source}')

    def test_load_timm_models(self):
        model_name = 'mixnet_l'
        source='timm'
        self.check_model_loading(model_name, 'EfficientNet', source)

        model_name = 'random'
        self.check_unknown_model_loading(model_name, f'Model {model_name} not found in {source}')
        
    def test_load_torchvision_models(self):
        model_name = 'vgg16'
        source='torchvision'
        self.check_model_loading(model_name, 'VGG', source)

        model_name = 'random'
        self.check_unknown_model_loading(model_name, f'Model {model_name} not found in {source}')

    def test_load_keras_models(self):
        source = 'keras'
        model_name = 'VGG16'
        self.check_model_loading(model_name, 'Functional', source)

        model_name = 'random'
        self.check_unknown_model_loading(model_name, f'Model {model_name} not found in {source}')


