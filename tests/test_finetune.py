"""Tests for reflex.finetune — the v0.3 MVP surface.

v0.3 scope: SmolVLA LoRA wrapper over lerobot-train + auto-export. These
tests don't actually run lerobot-train (requires GPU + HF dataset
downloads); they mock the subprocess and verify orchestration logic.

Goal: fine-tuning-pipeline (weight 5).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reflex.finetune import FinetuneConfig, FinetuneResult, run_finetune
from reflex.finetune.run import (
    _build_lerobot_command,
    _locate_checkpoint,
    _validate_config,
)


class TestConfigValidation:
    def test_minimal_valid_config(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        assert _validate_config(cfg) == []

    def test_missing_base_rejected(self, tmp_path):
        cfg = FinetuneConfig(base="", dataset="lerobot/libero", output=tmp_path)
        errs = _validate_config(cfg)
        assert any("base is required" in e for e in errs)

    def test_missing_dataset_rejected(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base", dataset="", output=tmp_path
        )
        assert any("dataset is required" in e for e in _validate_config(cfg))

    def test_zero_steps_rejected(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            num_steps=0,
        )
        assert any("num_steps" in e for e in _validate_config(cfg))

    def test_full_mode_rejected_in_v03(self, tmp_path):
        """v0.3 supports LoRA only. --mode full should fail cleanly."""
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            mode="full",
        )
        errs = _validate_config(cfg)
        assert any("v0.3 only supports --mode lora" in e for e in errs)

    def test_non_lerobot_backend_rejected(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            backend="openpi",
        )
        assert any("v0.3 only supports --backend lerobot" in e for e in _validate_config(cfg))

    def test_bad_precision_rejected(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            precision="fp8",
        )
        assert any("precision" in e for e in _validate_config(cfg))


class TestLerobotCommandBuild:
    def test_basic_command_shape(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            num_steps=5000,
            batch_size=16,
            learning_rate=2e-4,
            lora_rank=32,
            precision="bf16",
        )
        cmd = _build_lerobot_command(cfg)
        assert cmd[0] == "lerobot-train"
        # Sanity: all the important knobs flow through as draccus args.
        joined = " ".join(cmd)
        assert "--policy.path=lerobot/smolvla_base" in joined
        assert "--dataset.repo_id=lerobot/libero" in joined
        assert "--steps=5000" in joined
        assert "--batch_size=16" in joined
        assert "--policy.optimizer_lr=0.0002" in joined
        assert "--policy.use_lora=true" in joined
        assert "--policy.lora_rank=32" in joined
        assert "--precision=bf16" in joined

    def test_extra_args_pass_through(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            extra_lerobot_args={"policy.freeze_vision_encoder": "true"},
        )
        cmd = _build_lerobot_command(cfg)
        assert "--policy.freeze_vision_encoder=true" in cmd


class TestCheckpointLocation:
    def test_missing_dir_returns_none(self, tmp_path):
        assert _locate_checkpoint(tmp_path) is None

    def test_picks_highest_numeric_step(self, tmp_path):
        # Simulate lerobot's checkpoint layout
        for step in (500, 1000, 1500, 2000):
            d = tmp_path / "checkpoints" / str(step) / "pretrained_model"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"\x00")
        ckpt = _locate_checkpoint(tmp_path)
        assert ckpt is not None
        assert ckpt.parent.name == "2000"

    def test_falls_back_to_mtime_on_nonnumeric(self, tmp_path):
        (tmp_path / "checkpoints" / "last").mkdir(parents=True)
        ckpt = _locate_checkpoint(tmp_path)
        # Should still return something (the one dir present).
        assert ckpt is not None


class TestRunFinetuneOrchestration:
    """Mocks subprocess + export to verify the orchestration flow.
    No real training happens — we're testing wiring, not correctness."""

    def _setup_fake_checkpoint(self, output_dir: Path, step: int = 1000) -> Path:
        d = output_dir / "checkpoints" / str(step) / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"\x00")
        return d

    def test_config_failure_aborts(self, tmp_path):
        """Invalid config → aborted status, no subprocess launched."""
        cfg = FinetuneConfig(
            base="", dataset="lerobot/libero", output=tmp_path,  # missing base
        )
        with patch("reflex.finetune.run._run_lerobot_training") as mock_train:
            result = run_finetune(cfg)
        assert result.status == "aborted"
        mock_train.assert_not_called()
        assert "base is required" in (result.error or "")

    def test_training_failure_surfaces_rc(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        with patch("reflex.finetune.run._run_lerobot_training", return_value=42):
            result = run_finetune(cfg)
        assert result.status == "training_failed"
        assert "exited with code 42" in (result.error or "")

    def test_successful_training_plus_export(self, tmp_path):
        """Training succeeds (rc=0), checkpoint exists, export mock
        returns ONNX path → FinetuneResult has status=ok + onnx_path."""
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            num_steps=1000,
        )

        def _fake_train(cfg, log_path, **kwargs):
            # Simulate a successful training run that wrote a checkpoint.
            ckpt = self._setup_fake_checkpoint(cfg.output, step=1000)
            return 0

        with patch("reflex.finetune.run._run_lerobot_training", side_effect=_fake_train), \
             patch(
                 "reflex.finetune.run._auto_export",
                 return_value=(tmp_path / "export" / "model.onnx", None),
             ):
            result = run_finetune(cfg)

        assert result.status == "ok"
        assert result.final_checkpoint_path is not None
        assert result.final_checkpoint_path.name == "pretrained_model"
        assert result.onnx_path is not None
        assert result.training_log_path is not None

    def test_successful_training_but_no_checkpoint_fails(self, tmp_path):
        """If lerobot-train exits 0 but produces no checkpoint, that's a
        bug we want to surface — not report success."""
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        with patch("reflex.finetune.run._run_lerobot_training", return_value=0):
            result = run_finetune(cfg)
        assert result.status == "training_failed"
        assert "no checkpoint found" in (result.error or "")

    def test_export_failure_flagged(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )

        def _fake_train(cfg, log_path, **kwargs):
            self._setup_fake_checkpoint(cfg.output)
            return 0

        with patch("reflex.finetune.run._run_lerobot_training", side_effect=_fake_train), \
             patch(
                 "reflex.finetune.run._auto_export",
                 return_value=(None, "torch.onnx.export raised: OOM"),
             ):
            result = run_finetune(cfg)
        assert result.status == "export_failed"
        assert "OOM" in (result.error or "")

    def test_skip_export_flag(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            skip_export=True,
        )

        def _fake_train(cfg, log_path, **kwargs):
            self._setup_fake_checkpoint(cfg.output)
            return 0

        auto_export_mock = MagicMock()
        with patch("reflex.finetune.run._run_lerobot_training", side_effect=_fake_train), \
             patch("reflex.finetune.run._auto_export", auto_export_mock):
            result = run_finetune(cfg)

        assert result.status == "ok"
        assert result.onnx_path is None
        auto_export_mock.assert_not_called()


class TestCliWiring:
    """The `reflex finetune` subcommand should be registered on the main
    typer app and showable via --help without a functional lerobot install."""

    def test_finetune_help_works(self):
        from typer.testing import CliRunner
        from reflex.cli import app as cli_app
        runner = CliRunner()
        result = runner.invoke(cli_app, ["finetune", "--help"])
        assert result.exit_code == 0
        assert "base" in result.stdout.lower()
        assert "dataset" in result.stdout.lower()
        assert "output" in result.stdout.lower()
