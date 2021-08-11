import os 
import unittest

import thingsvision.vision as vision 
import thingsvision.tests.helper as helper 

import numpy as np
import pandas as pd 

class RDMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()
        model_name = 'vgg16_bn'
        model, dataset, dl = helper.create_model_and_dl(model_name, 'pt')
        module_name = helper.PT_MODEL_AND_MODULE_NAMES[model_name][0]
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            batch_size=helper.BATCH_SIZE,
            flatten_acts=False
        )
        for format in helper.FILE_FORMATS:
            # tests whether features can be saved in any of the formats as four-dimensional tensor
            vision.save_features(
                features=features,
                out_path=helper.OUT_PATH,
                file_format=format,
            )


    def test_rdms(self):
        """Test different distance metrics on which RDMs are based."""

        features = np.load(os.path.join(helper.OUT_PATH, 'features.npy'))
        features = features.reshape(helper.NUM_SAMPLES, -1)

        rdms = []
        for distance in helper.DISTANCES:
            rdm = vision.compute_rdm(features, distance)
            self.assertEqual(len(rdm.shape), 2)
            self.assertEqual(features.shape[0], rdm.shape[0], rdm.shape[1])
            vision.plot_rdm(helper.OUT_PATH, features, distance)
            rdms.append(rdm)

        for rdm_i, rdm_j in zip(rdms[:-1], rdms[1:]):
            corr = vision.correlate_rdms(rdm_i, rdm_j)
            self.assertTrue(isinstance(corr, float))
            self.assertTrue(corr > float(-1))
            self.assertTrue(corr < float(1))


class ComparisonTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_comparison(self):
        backend = 'pt'
        compare_model_names = ['vgg16_bn', 'vgg19_bn']
        compare_module_names = ['features.23', 'classifier.3']
        corr_mat = vision.compare_models(
            root=helper.TEST_PATH,
            out_path=helper.OUT_PATH,
            model_names=compare_model_names,
            module_names=compare_module_names,
            pretrained=True,
            batch_size=helper.BATCH_SIZE,
            backend=backend,
            flatten_acts=True,
            clip=[False, False],
            save_features=False,
        )
        self.assertTrue(isinstance(corr_mat, pd.DataFrame))
        self.assertEqual(corr_mat.shape, (len(
            compare_model_names), len(compare_module_names)))


class LoadItemsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.download_item_names()

    def test_item_name_loading(self):
        item_names = vision.load_item_names(helper.DATA_PATH)
        self.assertTrue(isinstance(item_names, np.ndarray))
        self.assertEqual(len(item_names), helper.NUM_OBJECTS)


class FileNamesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_filenames(self):
        import os
        print(os.getcwd())
        file_names = open(os.path.join(helper.OUT_PATH, 'file_names.txt'), 'r').read().split()
        img_files = []
        for root, _, files in os.walk(helper.TEST_PATH):
            for f in files:
                if f.endswith('png'):
                    img_files.append(os.path.join(root, f))
        self.assertEqual(sorted(file_names), sorted(img_files))


