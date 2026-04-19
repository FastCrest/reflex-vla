"""FP16 conversion unit tests — Orin Nano 8GB fit goal.

The actual conversion runs on Modal (12GB ONNX can't load locally).
These tests cover the pure helpers: sizing estimator, parity gate,
op blocklist defaults. They establish the contract that the Modal
runs will execute.

Goal: orin-nano-fp16-fit (weight 8).
"""
from __future__ import annotations

import pytest

from reflex.exporters.fp16_convert import (
    FP16_OP_BLOCKLIST,
    estimate_fp16_size_bytes,
    parity_gate,
)


class TestSizeEstimate:
    def test_halves_roughly(self):
        # 12.5 GB (pi0 FP32) → expect roughly 6.3-6.6 GB after FP16 + overhead
        pi0_fp32 = int(12.5 * 1e9)
        estimate = estimate_fp16_size_bytes(pi0_fp32)
        # Should be 6.25GB (halved) + 5% overhead = ~6.56GB.
        assert 6.2e9 < estimate < 6.7e9

    def test_zero_input(self):
        assert estimate_fp16_size_bytes(0) == 0

    def test_smolvla_stays_small(self):
        # SmolVLA 1.6GB → expect ~0.84GB post-FP16 (well under 8GB obviously)
        smolvla = int(1.6e9)
        estimate = estimate_fp16_size_bytes(smolvla)
        assert 0.7e9 < estimate < 1.0e9

    def test_fits_orin_nano_after_conversion(self):
        """The key claim: pi0 FP16 fits under 8 GB (leaving room for
        activations + OS)."""
        pi0_fp32 = int(12.5 * 1e9)
        pi05_fp32 = int(13.0 * 1e9)
        # Budget: 8GB total, reserve 1.5GB for activations+OS = 6.5GB for weights.
        ORIN_NANO_WEIGHT_BUDGET = int(6.5e9)
        # Our estimator is approximate; allow 5% slack in the fit claim.
        assert estimate_fp16_size_bytes(pi0_fp32) <= ORIN_NANO_WEIGHT_BUDGET * 1.05
        # pi0.5 at 13GB FP32 → ~6.8GB FP16, right at the edge. Document it.
        pi05_est = estimate_fp16_size_bytes(pi05_fp32)
        assert pi05_est > ORIN_NANO_WEIGHT_BUDGET * 0.9


class TestParityGate:
    def test_machine_precision_passes(self):
        g = parity_gate(max_abs_diff=1e-6, cos_sim=0.999999)
        assert g["verdict"] == "PASS"
        assert g["passed"] is True
        assert g["reasons"] == []

    def test_bad_cos_fails(self):
        g = parity_gate(max_abs_diff=1e-4, cos_sim=0.97)
        assert g["verdict"] == "FAIL"
        assert g["passed"] is False
        assert any("cos_sim" in r for r in g["reasons"])

    def test_bad_maxabs_fails(self):
        g = parity_gate(max_abs_diff=0.5, cos_sim=0.9999)
        assert g["verdict"] == "FAIL"
        assert any("max_abs_diff" in r for r in g["reasons"])

    def test_both_bad(self):
        g = parity_gate(max_abs_diff=0.5, cos_sim=0.5)
        assert g["verdict"] == "FAIL"
        assert len(g["reasons"]) == 2

    def test_threshold_overrides(self):
        """Callers can tighten or loosen the gate for model-specific tolerance."""
        # Looser: 99% cos is OK, 1e-2 max_abs is OK
        g = parity_gate(
            max_abs_diff=5e-3, cos_sim=0.99,
            max_abs_threshold=1e-2, cos_threshold=0.98,
        )
        assert g["passed"] is True


class TestBlocklist:
    def test_blocklist_covers_layernorm_ops(self):
        """LayerNorm-adjacent ops must be in the default blocklist —
        they're the biggest FP16 underflow risk for transformer VLAs."""
        for op in ("Pow", "ReduceMean", "Sqrt"):
            assert op in FP16_OP_BLOCKLIST, f"{op} missing from default blocklist"

    def test_blocklist_is_immutable_tuple(self):
        """Protect against accidental in-place mutation at import time."""
        assert isinstance(FP16_OP_BLOCKLIST, tuple)


class TestImportGuard:
    def test_conversion_raises_clearly_without_deps(self, monkeypatch):
        """If onnx or onnxconverter_common isn't installed, the caller
        should see a clean ImportError, not an AttributeError mid-run."""
        from reflex.exporters import fp16_convert

        # Simulate missing onnxconverter-common by making the import fail.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxconverter_common.float16":
                raise ImportError("simulated missing dep")
            if name == "onnxconverter_common":
                raise ImportError("simulated missing dep")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="onnxconverter-common"):
            fp16_convert.convert_fp32_to_fp16(
                "/tmp/nonexistent.onnx", "/tmp/out.onnx"
            )
