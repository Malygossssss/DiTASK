import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from yacs.config import CfgNode as CN

import run_ditask_flops as flops


def build_config(image_size=224, mtl=True):
    cfg = CN()
    cfg.SEED = 7
    cfg.MTL = bool(mtl)
    cfg.EVAL_MODE = False
    cfg.DATA = CN()
    cfg.DATA.IMG_SIZE = image_size
    cfg.MODEL = CN()
    cfg.MODEL.NAME = "ditask-test"
    cfg.MODEL.RESUME = ""
    cfg.TASKS = ["depth", "edge"]
    cfg.freeze()
    return cfg


class FlopsSummaryHelpersTest(unittest.TestCase):
    def test_get_output_dir_defaults_to_checkpoint_directory(self):
        args = SimpleNamespace(
            output_dir=None,
            checkpoint=os.path.join("some", "nested", "model.pth"),
        )

        output_dir = flops.get_output_dir(args)

        self.assertTrue(output_dir.endswith(os.path.join("some", "nested", "standalone_flops")))

    def test_normalize_image_size_accepts_scalar_and_pair(self):
        self.assertEqual(flops.normalize_image_size(build_config(image_size=448)), (448, 448))
        self.assertEqual(flops.normalize_image_size(build_config(image_size=(480, 640))), (480, 640))
        with self.assertRaises(ValueError):
            flops.normalize_image_size(build_config(image_size=(1, 2, 3)))

    def test_compute_flops_summary_converts_ptflops_output(self):
        model = mock.Mock()

        with mock.patch(
            "run_ditask_flops.get_model_complexity_info",
            return_value=(1.5e9, 2.25e6),
        ) as mocked_ptflops:
            summary = flops.compute_flops_summary(model, (3, 224, 224), print_per_layer_stat=True, verbose=True)

        model.eval.assert_called_once()
        mocked_ptflops.assert_called_once_with(
            model,
            (3, 224, 224),
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True,
        )
        self.assertEqual(summary["input_shape"], [3, 224, 224])
        self.assertEqual(summary["macs"], 1.5e9)
        self.assertEqual(summary["gmacs"], 1.5)
        self.assertEqual(summary["gflops"], 3.0)
        self.assertEqual(summary["parameter_count"], 2250000)
        self.assertEqual(summary["parameter_millions"], 2.25)

    def test_load_model_for_flops_uses_shared_loading_path(self):
        fake_backbone = object()
        fake_model = mock.Mock()
        logger = mock.Mock()

        with mock.patch(
            "run_ditask_flops.build_model",
            return_value=fake_backbone,
        ) as mocked_build_model, mock.patch(
            "run_ditask_flops.build_mtl_model",
            return_value=fake_model,
        ) as mocked_build_mtl_model, mock.patch(
            "run_ditask_flops.load_checkpoint"
        ) as mocked_load:
            fake_model.to.return_value = fake_model
            result = flops.load_model_for_flops(
                build_config(),
                "checkpoint.pth",
                torch.device("cpu"),
                logger,
            )

        self.assertIs(result, fake_model)
        mocked_build_model.assert_called_once()
        mocked_build_mtl_model.assert_called_once_with(fake_backbone, mock.ANY)
        mocked_load.assert_called_once()
        load_config = mocked_load.call_args[0][0]
        self.assertEqual(load_config.MODEL.RESUME, "checkpoint.pth")
        self.assertTrue(load_config.EVAL_MODE)

    def test_build_summary_payload_contains_required_fields(self):
        args = SimpleNamespace(checkpoint="checkpoint.pth")
        flops_summary = {
            "input_shape": [3, 224, 224],
            "macs": 1.5e9,
            "gmacs": 1.5,
            "gflops": 3.0,
            "parameter_count": 2250000,
            "parameter_millions": 2.25,
        }

        payload = flops.build_summary_payload(args, build_config(), flops_summary)

        self.assertEqual(payload["checkpoint"], os.path.abspath("checkpoint.pth"))
        self.assertEqual(payload["model_name"], "ditask-test")
        self.assertEqual(payload["tasks"], ["depth", "edge"])
        self.assertEqual(payload["input_shape"], [3, 224, 224])
        self.assertEqual(payload["notes"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
