import unittest

import tests.helper as helper
import thingsvision.core.extraction.helpers as core_helpers


class ModelLoadingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def check_model_loading(self, model_name, expected_class_name, source=None):
        extractor = core_helpers.get_extractor(model_name, pretrained=False, device="cpu", source=source)
        self.assertEqual(extractor.model.__class__.__name__, expected_class_name)

    def check_unknown_model_loading(self, model_name, expected_exception, source=None):
        with self.assertRaises(Exception) as e:
            _ = core_helpers.get_extractor(model_name, pretrained=False, device="cpu", source=source)
            self.assertEqual(e.exception, expected_exception)

    def test_mode_and_device(self):
        model_name = "vgg16"
        extractor, _, _ = helper.create_extractor_and_dataloader(
            model_name, pretrained=False, source="torchvision"
        )
        self.assertTrue(hasattr(extractor.model, helper.DEVICE))
        self.assertFalse(extractor.model.training)

    def test_load_model_without_source(self):
        model_name = "vgg16"  # PyTorch
        source = "torchvision"
        self.check_model_loading(
            model_name=model_name, expected_class_name="VGG", source=source
        )

        model_name = "VGG16"  # Tensorflow
        source = "keras"
        self.check_model_loading(
            model_name=model_name, expected_class_name="Functional", source=source
        )

        model_name = "random"
        source = "torchvision"
        self.check_unknown_model_loading(
            model_name=model_name,
            expected_exception=f"Could not find {model_name} in {source}.\nCheck whether model name is correctly spelled or use a different model.\n",
            source=source,
        )

    def test_load_custom_user_model(self):
        source = "custom"
        model_name = "VGG16_ecoset"
        self.check_model_loading(
            model_name=model_name, expected_class_name="VGG", source=source
        )

        model_name = "Resnet50_ecoset"
        self.check_model_loading(
            model_name=model_name, expected_class_name="ResNet", source=source
        )

        model_name = "Alexnet_ecoset"
        self.check_model_loading(
            model_name=model_name, expected_class_name="AlexNet", source=source
        )

        model_name = "Inception_ecoset"
        self.check_model_loading(
            model_name=model_name, expected_class_name="Inception3", source=source
        )

        model_name = "random"
        self.check_unknown_model_loading(
            model_name=model_name,
            expected_exception=f"Could not find {model_name} in {source}.\nCheck whether model name is correctly spelled or use a different model.\n",
            source=source,
        )

    def test_load_timm_models(self):
        source = "timm"
        model_name = "mixnet_l"
        self.check_model_loading(
            model_name=model_name, expected_class_name="EfficientNet", source=source
        )

        model_name = "random"
        self.check_unknown_model_loading(
            model_name=model_name,
            expected_exception=f"Could not find {model_name} in {source}.\nCheck whether model name is correctly spelled or use a different model.\n",
            source=source,
        )

    def test_load_torchvision_models(self):
        source = "torchvision"
        model_name = "vgg16"
        self.check_model_loading(
            model_name=model_name, expected_class_name="VGG", source=source
        )

        model_name = "random"
        self.check_unknown_model_loading(
            model_name=model_name,
            expected_exception=f"Could not find {model_name} in {source}.\nCheck whether model name is correctly spelled or use a different model.\n",
            source=source,
        )

    def test_load_keras_models(self):
        source = "keras"
        model_name = "VGG16"
        self.check_model_loading(
            model_name=model_name, expected_class_name="Functional", source=source
        )

        model_name = "random"
        self.check_unknown_model_loading(
            model_name=model_name,
            expected_exception=f"Could not find {model_name} in {source}.\nCheck whether model name is correctly spelled or use a different model.\n",
            source=source,
        )
