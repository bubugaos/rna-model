import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from src.config import ExperimentConfig
from src.model.model import RNADynamicModel
from src.training.losses import RNALMCriterion
from src.training.train import _mlm_debug_stats, build_rna_lm_criterion
from scripts.run_multiseed import _do_one_run


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "training" / "train.py"


class TestTrainLossContract(unittest.TestCase):
    def test_mlm_debug_stats_supports_wrapped_model(self):
        torch.manual_seed(5)
        model = RNADynamicModel(
            input_vocab_size=7,
            output_vocab_size=7,
            embed_dim=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            beta=0.5,
            num_classes=3,
            dropout=0.0,
        )
        model.eval()

        x = torch.randint(0, 4, (2, 8))
        _bppm = torch.rand(2, 8, 8)
        _bppm = (_bppm + _bppm.transpose(1, 2)) / 2
        row_sum = _bppm.sum(dim=-1)
        p_safe = _bppm.clamp(min=1e-8)
        entropy = -(p_safe * p_safe.log()).sum(dim=-1) / 8.0
        S = _bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        mask = torch.ones(2, 8)
        mask_pos = torch.zeros(2, 8, dtype=torch.bool)
        mask_pos[:, 0] = True
        mlm_labels = torch.randint(0, 7, (2, 8), dtype=torch.long)

        with torch.no_grad():
            output = model(x, row_sum=row_sum, entropy=entropy, cross_pair_sum=cross_pair_sum, mask=mask)

        stats = _mlm_debug_stats(output, mask, mask_pos, mlm_labels)

        self.assertEqual(stats["mlm_n"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(stats["mlm_acc"])).item())

    def test_build_rna_lm_criterion_ignores_removed_loss_fields(self):
        config = ExperimentConfig()
        criterion = build_rna_lm_criterion(config)

        self.assertIsInstance(criterion, RNALMCriterion)
        self.assertEqual(criterion.vocab_size, config.model.output_vocab_size)
        self.assertEqual(criterion.lambda_recon, config.loss.lambda_recon)
        self.assertEqual(criterion.lambda_mlm, config.loss.lambda_mlm)
        self.assertEqual(criterion.lambda_mdl, config.loss.lambda_mdl)

    def test_train_smoke_runs_with_default_loss_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(REPO_ROOT) if not python_path else os.pathsep.join([str(REPO_ROOT), python_path])
            env["MPLBACKEND"] = "Agg"

            result = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--smoke",
                    "--device",
                    "cpu",
                    "--num_epochs",
                    "1",
                    "--batch_size",
                    "2",
                    "--max_steps",
                    "1",
                    "--exp_name",
                    "task6_smoke",
                    "--exp_dir",
                    tmp_dir,
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

            if result.returncode != 0:
                self.fail(f"train.py smoke failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            self.assertTrue((Path(tmp_dir) / "config.json").exists())
            self.assertTrue((Path(tmp_dir) / "history.json").exists())
            self.assertTrue((Path(tmp_dir) / "run_info.json").exists())

            with open(Path(tmp_dir) / "history.json", "r", encoding="utf-8") as handle:
                history = json.load(handle)

            self.assertEqual(len(history["train_total"]), 1)
            self.assertEqual(len(history["val_total"]), 1)

    def test_mdl_loss_grad_flows(self):
        """Verify MDL loss produces gradients w.r.t. boundary_probs."""
        torch.manual_seed(7)
        criterion = RNALMCriterion(lambda_mdl=0.5, mdl_cost_base=0.3)
        criterion.set_mdl_weight(1.0)

        B, L, C, D = 2, 16, 4, 64
        boundary_probs = torch.rand(B, L - 1, requires_grad=True)
        # Make all chunks identical → gain ≈ 0 → cost > gain → loss > 0
        base_chunk = torch.randn(1, D)
        smoothed_chunks = base_chunk.expand(B, C, D).clone()
        segment_ids = torch.zeros(B, L, dtype=torch.long)
        segment_ids[:, 4:8] = 1
        segment_ids[:, 8:12] = 2
        segment_ids[:, 12:] = 3
        mask = torch.ones(B, L)
        bppm = torch.rand(B, L, L)
        bppm = (bppm + bppm.transpose(1, 2)) / 2
        bppm_row_sum = bppm.sum(dim=-1)

        mdl_loss = criterion._compute_mdl_loss(
            boundary_probs, smoothed_chunks, segment_ids, mask, bppm_row_sum
        )
        self.assertTrue(mdl_loss.isfinite().item())
        self.assertGreater(mdl_loss.item(), 0.0)

        grad = torch.autograd.grad(mdl_loss, boundary_probs, retain_graph=True)[0]
        self.assertIsNotNone(grad)
        self.assertTrue(grad.abs().sum().item() > 0.0)

    def test_train_accepts_no_chunk_without_removed_struct_guards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(REPO_ROOT) if not python_path else os.pathsep.join([str(REPO_ROOT), python_path])
            env["MPLBACKEND"] = "Agg"

            result = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--smoke",
                    "--device",
                    "cpu",
                    "--num_epochs",
                    "1",
                    "--batch_size",
                    "2",
                    "--max_steps",
                    "1",
                    "--exp_name",
                    "task7_no_chunk_guard",
                    "--exp_dir",
                    tmp_dir,
                    "--no_chunk",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

            if result.returncode != 0:
                self.fail(f"train.py no_chunk smoke failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    def test_run_multiseed_forwards_skip_boundary_to_evaluate_only(self):
        args = Namespace(
            base_name="unit",
            base_dir=str(REPO_ROOT),
            chunk_size=3,
            bppm_mode=None,
            cache_dir=None,
            skip_boundary=True,
            smoke=True,
            pretrain_epochs=None,
            pretrain_max_steps=None,
            finetune_epochs=None,
            finetune_max_steps=None,
            eval_max_steps=None,
            num_workers=None,
            pin_memory=False,
            persistent_workers=False,
            prefetch_workers=None,
        )
        calls = []

        def fake_run(cmd, log_path=None):
            calls.append(list(cmd))
            return {"ok": True, "exit_code": 0, "log_path": log_path, "stdout_tail": "", "stderr_tail": ""}

        with patch("scripts.run_multiseed._run", side_effect=fake_run):
            record = _do_one_run("dynamic", 42, args)

        self.assertTrue(record["phases"]["evaluate"]["ok"])
        self.assertEqual(len(calls), 3)
        self.assertNotIn("--skip_boundary", calls[0])
        self.assertNotIn("--skip_boundary", calls[1])
        self.assertIn("--skip_boundary", calls[2])


if __name__ == "__main__":
    unittest.main()
