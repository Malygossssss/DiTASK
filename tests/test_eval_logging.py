import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.eval_depth import DepthMeter
from evaluation.eval_edge import EdgeMeter
from evaluation.eval_human_parts import HumanPartsMeter
from evaluation.eval_normals import NormalsMeter
from evaluation.eval_sal import SaliencyMeter
from evaluation.eval_semseg import SemsegMeter
from logger import bind_eval_logger, create_logger


def _flush_handlers(logger):
    for handler in logger.handlers:
        handler.flush()


def _build_updated_meters():
    semseg = SemsegMeter("NYUD", config=None)
    semseg.update(
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.zeros((1, 2, 2), dtype=torch.long),
    )

    normals = NormalsMeter()
    normals_pred = torch.tensor(
        [[
            [[255.0, 127.5, 127.5], [255.0, 127.5, 127.5]],
            [[255.0, 127.5, 127.5], [255.0, 127.5, 127.5]],
        ]],
        dtype=torch.float32,
    )
    normals_gt = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    normals_gt[:, 0, :, :] = 1.0
    normals.update(normals_pred, normals_gt)

    depth = DepthMeter()
    depth.update(
        torch.ones((1, 2, 2), dtype=torch.float32),
        torch.ones((1, 2, 2), dtype=torch.float32),
    )

    sal = SaliencyMeter()
    sal.update(
        torch.full((2, 2, 2), 255.0, dtype=torch.float32),
        torch.ones((2, 2, 2), dtype=torch.float32),
    )

    edge = EdgeMeter(pos_weight=0.95)
    edge.update(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 2, 2), dtype=torch.float32),
    )

    human_parts = HumanPartsMeter("PASCALContext")
    human_parts.update(
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.zeros((1, 2, 2), dtype=torch.long),
    )

    return {
        "semseg": semseg,
        "normals": normals,
        "depth": depth,
        "sal": sal,
        "edge": edge,
        "human_parts": human_parts,
    }


class EvalLoggingTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.addCleanup(bind_eval_logger, None)
        self.log_dir = Path(self.tmp_dir.name)
        self.logger = create_logger(
            output_dir=str(self.log_dir),
            dist_rank=0,
            name=f"eval_logging_{id(self)}",
        )
        bind_eval_logger(self.logger)
        self.log_path = self.log_dir / "log_rank0.txt"

    def test_all_task_meters_write_to_run_log(self):
        meters = _build_updated_meters()

        with patch("builtins.print") as mock_print:
            for meter in meters.values():
                meter.get_score(verbose=True)

        _flush_handlers(self.logger)
        log_text = self.log_path.read_text(encoding="utf-8")

        self.assertIn("Semantic Segmentation mIoU", log_text)
        self.assertIn("Results for Surface Normal Estimation", log_text)
        self.assertIn("Results for depth prediction", log_text)
        self.assertIn("Results for Saliency Estimation", log_text)
        self.assertIn("Edge Detection Evaluation", log_text)
        self.assertIn("Human Parts mIoU", log_text)
        mock_print.assert_not_called()

    def test_verbose_false_suppresses_eval_logging(self):
        meters = _build_updated_meters()

        with patch("builtins.print") as mock_print:
            for meter in meters.values():
                meter.get_score(verbose=False)

        _flush_handlers(self.logger)
        log_text = self.log_path.read_text(encoding="utf-8")

        self.assertEqual(log_text, "")
        mock_print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
