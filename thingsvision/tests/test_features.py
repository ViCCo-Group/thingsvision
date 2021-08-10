import os
import shutil
import unittest

import thingsvision.tests.helper as helper
import thingsvision.vision as vision
import numpy as np 

class FeaturesTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def setUp(self):
        shutil.rmtree(helper.OUT_PATH)
        os.makedirs(helper.OUT_PATH)
        
    def get_2D_features(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        module_name ='classifier.3'
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=helper.BATCH_SIZE,
            flatten_acts=False
        )
        return features
 
    def get_4D_features(self):
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        module_name ='features.23'
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=helper.BATCH_SIZE,
            flatten_acts=False
        )
        return features

    def test_postprocessing(self):
        """Test different postprocessing methods (e.g., centering, normalization, compression)."""
        features = self.get_2D_features()
        flattened_features = features.reshape(helper.NUM_SAMPLES, -1)
        centred_features = vision.center_features(flattened_features)
        normalized_features = vision.normalize_features(flattened_features)
        transformed_features = vision.compress_features(
            flattened_features, rnd_seed=42, retained_var=.9)

        self.assertTrue(centred_features.mean(axis=0).sum() < 1e-3)
        self.assertEqual(np.linalg.norm(normalized_features, axis=1).sum(),
                         np.ones(features.shape[0]).sum())
        self.assertTrue(
            transformed_features.shape[1] < flattened_features.shape[1])

    def check_file_exists(self, file_name, format, txt_should_exists=True):
        if format == 'hdf5':
            format = 'h5'
            file_name = 'features'

        path = os.path.join(helper.OUT_PATH, f'{file_name}.{format}')
        if format == 'txt' and not txt_should_exists:
            self.assertTrue(not os.path.exists(path))
        else:
            self.assertTrue(os.path.exists(path))

    def test_storing_2d(self):
        """Test storing possibilities."""
        features = self.get_2D_features()
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as two-dimensional tensor
            vision.save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )
            
            self.check_file_exists('features', format)

    def test_storing_4d(self):
        features = self.get_4D_features()
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            vision.save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )
            
            self.check_file_exists('features', format, False)

    def test_splitting_2d(self):
        n_splits = 3
        features = self.get_2D_features()
        for format in helper.FILE_FORMATS:
            vision.split_features(
                        features=features,
                        PATH=helper.OUT_PATH,
                        file_format=format,
                        n_splits=n_splits
            )
            
            for i in range(1, n_splits):
                self.check_file_exists(f'features_0{i}', format)

    def test_splitting_4d(self):
        n_splits = 3
        features = self.get_4D_features()
        for format in set(helper.FILE_FORMATS) - set(['txt']):
            vision.split_features(
                        features=features,
                        PATH=helper.OUT_PATH,
                        file_format=format,
                        n_splits=n_splits
            )
            
            for i in range(1, n_splits):
                self.check_file_exists(f'features_0{i}', format, False)

        with self.assertRaises(Exception):
            vision.split_features(
                        features=features,
                        PATH=helper.OUT_PATH,
                        file_format='txt',
                        n_splits=n_splits
            )
