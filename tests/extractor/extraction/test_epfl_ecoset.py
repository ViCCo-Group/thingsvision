import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from tests import helper
from thingsvision.core.extraction.helpers import get_extractor
from thingsvision.custom_models.epfl_ecoset import (
    CHECKPOINTS,
    EPFL_ecoset,
    _restricted_pickle,
)

# The full set of checkpoints is ~22 GB, so it only runs on request.
TEST_ALL = os.environ.get("THINGSVISION_TEST_ALL_EPFL_ECOSET") == "1"


class Exec:
    """Unpickling this calls exec, as a malicious checkpoint would."""

    def __reduce__(self):
        return (exec, ("raise SystemExit('unpickler ran code')",))


class EPFLEcosetTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper.create_test_images()

    def test_extract_features(self):
        extractor, dataset, batches = helper.create_extractor_and_dataloader(
            "EPFL_ecoset",
            pretrained=True,
            source="custom",
            kwargs={"variant": "resnet18", "seed": 0},
        )
        features = extractor.extract_features(
            batches=batches, module_name="avgpool", flatten_acts=True
        )
        self.assertEqual(features.shape, (len(dataset), 512))
        self.assertTrue(np.isfinite(features).all())

    def test_unknown_variant_or_seed(self):
        for parameters in [
            {"variant": "resnet200"},
            {"variant": "resnet18", "seed": 3},
            {},
        ]:
            with self.assertRaisesRegex(ValueError, "Available variants"):
                get_extractor(
                    model_name="EPFL_ecoset",
                    source="custom",
                    device="cpu",
                    pretrained=True,
                    model_parameters=parameters,
                )

    def test_unpickler_refuses_code(self):
        payload = {"x": Exec()}
        with tempfile.TemporaryDirectory() as tmp:
            zip_path, legacy_path = (
                os.path.join(tmp, "zip.pt"),
                os.path.join(tmp, "legacy.pt"),
            )
            torch.save(payload, zip_path)
            with open(legacy_path, "wb") as f:
                pickle.dump(payload, f)
            for path in (zip_path, legacy_path):
                with self.assertRaises(pickle.UnpicklingError):
                    torch.load(
                        path, pickle_module=_restricted_pickle, weights_only=False
                    )

    @unittest.skipUnless(
        TEST_ALL, "set THINGSVISION_TEST_ALL_EPFL_ECOSET=1 to load all checkpoints"
    )
    def test_load_all_checkpoints(self):
        for variant, seed in CHECKPOINTS:
            with self.subTest(variant=variant, seed=seed):
                model, _ = EPFL_ecoset(
                    "cpu", {"variant": variant, "seed": seed}
                ).create_model()
                self.assertFalse(model.training)
