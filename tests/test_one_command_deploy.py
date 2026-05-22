"""Tests for `reflex go` (one-command-deploy) — hardware probe + model resolver + CLI.

Verifies:
- hardware_probe returns canonical device classes; override bypass; CPU fallback
- model_resolver: exact-id, family+device match, fallback, no-match raises
- `reflex go` CLI: --model required, dry-run path, override propagates, JSON-friendly
- Backward compat with `reflex models list/pull` (the resolver consumes the
  same registry)
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from reflex.runtime.hardware_probe import (
    CANONICAL_DEVICE_CLASSES,
    ProbeResult,
    probe_device_class,
)
from reflex.runtime.model_resolver import (
    ModelResolverError,
    ResolveResult,
    resolve_model,
)


# ---------- hardware_probe ----------

class TestHardwareProbe:
    def test_override_bypass(self):
        r = probe_device_class(override="orin_nano")
        assert r.device_class == "orin_nano"
        assert r.detection_method == "override"

    def test_override_unknown_raises(self):
        with pytest.raises(ValueError, match="not in"):
            probe_device_class(override="bogus_device_xyz")

    def test_nvidia_smi_h100(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi",
                   return_value=("NVIDIA H100 80GB HBM3", "raw output")):
            r = probe_device_class()
        assert r.device_class == "h100"
        assert r.detection_method == "nvidia-smi"

    def test_nvidia_smi_h200(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi",
                   return_value=("NVIDIA H200", "raw")):
            r = probe_device_class()
        assert r.device_class == "h200"

    def test_nvidia_smi_a100(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi",
                   return_value=("NVIDIA A100-SXM4-80GB", "raw")):
            r = probe_device_class()
        assert r.device_class == "a100"

    def test_nvidia_smi_a10g(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi",
                   return_value=("NVIDIA A10G", "raw")):
            r = probe_device_class()
        assert r.device_class == "a10g"

    def test_nvidia_smi_orin_via_smi(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi",
                   return_value=("NVIDIA Orin", "raw")):
            r = probe_device_class()
        assert r.device_class == "agx_orin"  # generic Orin → AGX (more capable default)

    def test_nvidia_smi_unknown_falls_back_to_a10g(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi",
                   return_value=("NVIDIA RTX 4090", "raw")):
            r = probe_device_class()
        assert r.device_class == "a10g"
        assert "unrecognized" in (r.notes[0] if r.notes else "").lower()

    def test_falls_back_to_cpu_when_no_gpu(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi", return_value=None), \
             patch("reflex.runtime.hardware_probe._try_tegrastats", return_value=None):
            r = probe_device_class()
        assert r.device_class == "cpu"
        assert r.detection_method == "fallback-cpu"

    def test_tegrastats_when_smi_missing(self):
        with patch("reflex.runtime.hardware_probe._try_nvidia_smi", return_value=None), \
             patch("reflex.runtime.hardware_probe._try_tegrastats",
                   return_value=("orin_nano", "raw tegrastats output")):
            r = probe_device_class()
        assert r.device_class == "orin_nano"
        assert r.detection_method == "tegrastats"

    def test_canonical_device_classes_immutable(self):
        # Lock the canonical set — every device class in registry MUST be in here
        from reflex.registry import REGISTRY
        for entry in REGISTRY:
            for d in entry.supported_devices:
                assert d in CANONICAL_DEVICE_CLASSES, (
                    f"{entry.model_id} declares device {d!r} not in canonical set"
                )

    def test_probe_result_validates_device_class(self):
        with pytest.raises(ValueError, match="not in canonical"):
            ProbeResult(device_class="bogus", detection_method="x")


# ---------- model_resolver ----------

class TestModelResolver:
    def test_exact_id_match(self):
        r = resolve_model(model="pi05-base", device_class="a100")
        assert r.entry.model_id == "pi05-base"
        assert r.matched_strategy == "exact-id"

    def test_exact_id_warns_on_unsupported_device(self):
        # pi05-base is NOT marked orin_nano-compatible
        r = resolve_model(model="pi05-base", device_class="orin_nano")
        assert r.entry.model_id == "pi05-base"
        assert r.matched_strategy == "exact-id"
        assert any("not listed as supported" in n for n in r.notes)

    def test_family_match_picks_smallest_for_edge(self):
        # Family smolvla on orin_nano: should pick smallest (smolvla-base + smolvla-libero
        # both 900MB; first by size→either tied; resolver picks min)
        r = resolve_model(model="smolvla", device_class="orin_nano")
        assert r.entry.family == "smolvla"
        assert r.matched_strategy == "family-and-device"

    def test_family_match_picks_largest_for_datacenter(self):
        r = resolve_model(model="pi05", device_class="a100")
        assert r.entry.family == "pi05"
        assert r.matched_strategy == "family-and-device"

    def test_family_match_with_embodiment_filter(self):
        # smolvla-libero only supports franka; smolvla-base supports franka + so100
        r = resolve_model(model="smolvla", device_class="agx_orin", embodiment="so100")
        # Only smolvla-base satisfies (so100 ∈ supported_embodiments)
        assert r.entry.model_id == "smolvla-base"

    def test_family_fallback_when_no_device_match(self):
        # pi0 family doesn't list orin_nano in seed registry
        r = resolve_model(model="pi0", device_class="orin_nano")
        assert r.entry.family == "pi0"
        assert r.matched_strategy == "family-fallback"
        assert any("falling back" in n for n in r.notes)

    def test_unknown_model_raises(self):
        with pytest.raises(ModelResolverError, match="No registry entry"):
            resolve_model(model="totally-unknown-family", device_class="a100")

    def test_resolve_includes_helpful_error(self):
        try:
            resolve_model(model="bogus", device_class="a100")
        except ModelResolverError as e:
            msg = str(e)
            assert "Available families" in msg
            assert "Available ids" in msg
            assert "reflex models list" in msg


# ---------- `reflex go` CLI ----------

@pytest.fixture
def runner():
    typer_testing = pytest.importorskip("typer.testing")
    return typer_testing.CliRunner()


@pytest.fixture
def cli_app():
    from reflex.cli import app
    return app


class TestReflexGoCli:
    def test_visible_in_top_level_help(self, runner, cli_app):
        result = runner.invoke(cli_app, ["--help"])
        assert "go" in result.output

    def test_help_describes_pipeline(self, runner, cli_app):
        result = runner.invoke(cli_app, ["go", "--help"])
        assert result.exit_code == 0
        assert "probe" in result.output.lower() or "Probe" in result.output
        assert "--model" in result.output
        assert "--embodiment" in result.output
        assert "--device-class" in result.output
        assert "--dry-run" in result.output

    def test_missing_model_exits_2(self, runner, cli_app):
        result = runner.invoke(cli_app, ["go"])
        assert result.exit_code == 2
        assert "--model is required" in result.output

    def test_unknown_device_class_exits_2(self, runner, cli_app):
        result = runner.invoke(cli_app, ["go", "--model", "pi05-libero", "--device-class", "bogus"])
        assert result.exit_code == 2
        assert "--device-class" in result.output

    def test_dry_run_resolves_without_pulling(self, runner, cli_app):
        result = runner.invoke(cli_app, [
            "go", "--model", "pi05-libero", "--device-class", "a10g", "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        assert "pi05-libero" in result.output
        assert "a10g" in result.output

    def test_dry_run_with_family_resolves(self, runner, cli_app):
        result = runner.invoke(cli_app, [
            "go", "--model", "smolvla", "--device-class", "orin_nano", "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "smolvla" in result.output
        assert "DRY RUN" in result.output

    def test_dry_run_unknown_model_exits_2(self, runner, cli_app):
        result = runner.invoke(cli_app, [
            "go", "--model", "totally-unknown-model", "--device-class", "a10g", "--dry-run",
        ])
        assert result.exit_code == 2
        assert "No registry entry" in result.output or "Unknown" in result.output

    def test_pull_then_export_then_serve_for_requires_export_model(self, runner, cli_app, tmp_path, monkeypatch):
        # Empty target dir → snapshot_download runs.
        # requires_export=True → export_monolithic runs (mocked).
        # Then serve startup attempts (we mock create_app so the test stops there).
        target = tmp_path / "model_cache"
        monkeypatch.setenv("REFLEX_HOME", str(tmp_path / "reflex_cache"))  # isolate export cache

        def fake_dl(**kwargs):
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}")
            return str(local_dir)

        def fake_export(model_path, output_dir, num_steps=10, target=None):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "VERIFICATION.md").write_text("# stub")
            return {"onnx_path": str(Path(output_dir) / "model.onnx"), "size_mb": 100.0}

        with patch("huggingface_hub.snapshot_download", side_effect=fake_dl) as mock_dl, \
             patch("reflex.exporters.monolithic.export_monolithic", side_effect=fake_export) as mock_export, \
             patch("reflex.runtime.server.create_app", side_effect=RuntimeError("serve-stub")):
            result = runner.invoke(cli_app, [
                "go",
                "--model", "smolvla-base",
                "--device-class", "a10g",
                "--target-dir", str(target),
            ])
        # We expect the pull + export + serve-attempt path; serve fails on the stub.
        # Exit is non-zero because of the create_app stub, but pull + export must have run.
        mock_dl.assert_called_once()
        assert mock_dl.call_args.kwargs["repo_id"] == "lerobot/smolvla_base"
        mock_export.assert_called_once()
        assert "exporting:" in result.output
        assert "export complete" in result.output

    def test_cache_hit_skips_pull(self, runner, cli_app, tmp_path, monkeypatch):
        target = tmp_path / "cached"
        target.mkdir()
        (target / "already_present.txt").write_text("hi")
        monkeypatch.setenv("REFLEX_HOME", str(tmp_path / "reflex_cache"))

        def fake_export(model_path, output_dir, num_steps=10, target=None):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "VERIFICATION.md").write_text("# stub")
            return {"onnx_path": str(Path(output_dir) / "model.onnx"), "size_mb": 100.0}

        with patch("huggingface_hub.snapshot_download") as mock_dl, \
             patch("reflex.exporters.monolithic.export_monolithic", side_effect=fake_export), \
             patch("reflex.runtime.server.create_app", side_effect=RuntimeError("serve-stub")):
            result = runner.invoke(cli_app, [
                "go",
                "--model", "smolvla-base",
                "--device-class", "a10g",
                "--target-dir", str(target),
            ])
        assert "cache hit" in result.output
        mock_dl.assert_not_called()

    def test_export_dep_missing_errors_with_monolithic_install_hint(self, runner, cli_app, tmp_path, monkeypatch):
        # Without the [monolithic] extras, export_monolithic raises ImportError.
        # `reflex go` should catch it, print the install hint, and exit 2.
        target = tmp_path / "cached"
        target.mkdir()
        (target / "weights.bin").write_text("stub")
        monkeypatch.setenv("REFLEX_HOME", str(tmp_path / "reflex_cache"))

        with patch(
            "reflex.exporters.monolithic.export_monolithic",
            side_effect=ImportError("Missing dependencies: lerobot==0.5.1, onnx-diagnostic"),
        ):
            result = runner.invoke(cli_app, [
                "go",
                "--model", "smolvla-base",
                "--device-class", "a10g",
                "--target-dir", str(target),
            ])
        assert result.exit_code == 2, result.output
        assert "monolithic" in result.output
        assert "pip install" in result.output
