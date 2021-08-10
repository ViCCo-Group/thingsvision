import unittest

import helper

import numpy as np 

class ModelLoadingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        
    def test_mode_and_device(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        self.assertTrue(hasattr(model.model, helper.DEVICE))
        self.assertFalse(model.model.training)
