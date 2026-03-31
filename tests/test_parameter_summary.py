import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from yacs.config import CfgNode as CN

import run_ditask_parameter_summary as parameter_summary


def build_config(mtl=True):
    cfg = CN()
    cfg.SEED = 7
    cfg.MTL = bool(mtl)
    cfg.EVAL_MODE = False
    cfg.MODEL = CN()
    cfg.MODEL.NAME = "ditask-test"
    cfg.MODEL.RESUME = ""
    cfg.TASKS = ["depth", "edge"]
    cfg.freeze()
    return cfg


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 3)
        self.decoder = torch.nn.Linear(3, 2, bias=False)
        self.decoder.weight.requires_grad_(False)
        self.register_buffer("running_scale", torch.ones(5))


class ParameterSummaryHelpersTest(unittest.TestCase):
    def test_get_output_dir_defaults_to_checkpoint_directory(self):
        args = SimpleNamespace(
            output_dir=None,
            checkpoint=os.path.join("some", "nested", "model.pth"),
        )

        output_dir = parameter_summary.get_output_dir(args)

        self.assertTrue(output_dir.endswith(os.path.join("some", "nested", "standalone_parameter_summary")))

    def test_summarize_named_parameters_counts_total_and_trainable(self):
        model = TinyModel()

        summary = parameter_summary.summarize_named_parameters(model)

        self.assertEqual(summary["total_parameter_count"], 21)
        self.assertEqual(summary["trainable_parameter_count"], 15)
        self.assertEqual(summary["module_parameter_summary"][0]["name"], "encoder")

    def test_summarize_named_buffers_counts_registered_buffers(self):
        model = TinyModel()

        summary = parameter_summary.summarize_named_buffers(model)

        self.assertEqual(summary["total_buffer_tensor_count"], 1)
        self.assertEqual(summary["total_buffer_element_count"], 5)

    def test_load_model_for_summary_uses_shared_loading_path(self):
        fake_backbone = object()
        fake_model = mock.Mock()
        logger = mock.Mock()

        with mock.patch(
            "run_ditask_parameter_summary.build_model",
            return_value=fake_backbone,
        ) as mocked_build_model, mock.patch(
            "run_ditask_parameter_summary.build_mtl_model",
            return_value=fake_model,
        ) as mocked_build_mtl_model, mock.patch(
            "run_ditask_parameter_summary.load_checkpoint"
        ) as mocked_load:
            fake_model.to.return_value = fake_model
            result = parameter_summary.load_model_for_summary(
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

        payload = parameter_summary.build_summary_payload(args, build_config(), TinyModel())

        self.assertEqual(payload["checkpoint"], os.path.abspath("checkpoint.pth"))
        self.assertEqual(payload["model_name"], "ditask-test")
        self.assertEqual(payload["tasks"], ["depth", "edge"])
        self.assertIn("total_parameter_count", payload)
        self.assertIn("buffer_mb", payload)


if __name__ == "__main__":
    unittest.main()
