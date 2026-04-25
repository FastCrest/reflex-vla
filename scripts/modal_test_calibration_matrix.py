"""Modal cross-hardware calibration matrix smoke (auto-calibration Day 7).

Per ADR/plan `2026-04-25-auto-calibration-architecture` Day 7: validates
the calibration substrate on real Modal hardware. Phase 1 minimum: probe
HardwareFingerprint + run measure_latency_profile against a synthetic
predict callable on A10G. Asserts outputs are in expected ranges.

Larger-matrix expansion (A10G + A100 × franka + so100 + ur5) is the
follow-up for the full Day 7/9 cost-table refresh; this script ships
the cheapest meaningful entry first per CLAUDE.md "Maximum tests, not
minimum viable".

Usage:
    modal run scripts/modal_test_calibration_matrix.py

Cost: ~$0.30-$0.50 on A10G (image build cached after first run).
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("reflex-calibration-matrix")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUILD_BUST = _build_bust()


# Light image -- just CUDA + reflex-vla. No LIBERO, no lerobot, no full
# inference stack. Calibration substrate is pure-Python after CUDA probe.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy",
        "torch",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20",
        "psutil",
        "pyyaml",
    )
    .run_commands(
        f'echo "build_bust={_BUILD_BUST}"',
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/rylinjames/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,  # 10 min cap -- calibration substrate is fast
    secrets=[_hf_secret()],
)
def probe_calibration_a10g() -> dict:
    """Probe HardwareFingerprint + measure_latency_profile on Modal A10G.

    Returns a dict with the probed fingerprint + measurement quality so
    the local entrypoint can render a structured RESULT block + assert.
    """
    import time
    import numpy as np

    print("[calibration] probing HardwareFingerprint...")
    from reflex.runtime.calibration import (
        HardwareFingerprint,
        measure_latency_profile,
    )

    fp = HardwareFingerprint.current()
    print(f"[calibration]   gpu_name: {fp.gpu_name!r}")
    print(f"[calibration]   gpu_uuid: {fp.gpu_uuid!r}")
    print(f"[calibration]   driver: {fp.driver_major}.{fp.driver_minor}")
    print(f"[calibration]   cuda: {fp.cuda_major}.{fp.cuda_minor}")
    print(f"[calibration]   ram_gb: {fp.ram_gb}")
    print(f"[calibration]   reflex_version: {fp.reflex_version}")

    print("[calibration] running measure_latency_profile against a synthetic predict...")

    # Synthetic predict: 1ms-ish busy work that exercises the timing
    # harness without needing a real model loaded. Validates the
    # measurement pipeline (warmup -> N-iter loop -> outlier trim ->
    # median/p99/quality_score).
    def _stub_predict():
        # ~0.5-2ms of numpy work; enough to be measurable without
        # being trivially nondeterministic.
        a = np.random.rand(128, 128).astype(np.float32)
        b = np.random.rand(128, 128).astype(np.float32)
        c = a @ b
        return c.sum()

    quality = measure_latency_profile(
        _stub_predict,
        n_iters=100,
        warmup_iters=10,
    )
    print(f"[calibration]   median_ms: {quality.median_ms:.4f}")
    print(f"[calibration]   p99_ms: {quality.p99_ms:.4f}")
    print(f"[calibration]   quality_score: {quality.quality_score:.4f}")
    print(f"[calibration]   n_outliers_dropped: {quality.n_outliers_dropped}")

    # Assertions: invariants the substrate guarantees.
    assert quality.median_ms > 0, "median_ms must be > 0"
    assert quality.p99_ms >= quality.median_ms, "p99_ms must be >= median_ms"
    assert 0.0 <= quality.quality_score <= 1.0, (
        f"quality_score must be in [0, 1], got {quality.quality_score}"
    )
    assert quality.n_outliers_dropped >= 0, "n_outliers_dropped must be >= 0"

    return {
        "status": "ok",
        "tier": "A10G",
        "fingerprint": {
            "gpu_name": fp.gpu_name,
            "gpu_uuid": fp.gpu_uuid,
            "driver_major": fp.driver_major,
            "driver_minor": fp.driver_minor,
            "cuda_major": fp.cuda_major,
            "cuda_minor": fp.cuda_minor,
            "ram_gb": fp.ram_gb,
            "reflex_version": fp.reflex_version,
        },
        "quality": {
            "median_ms": quality.median_ms,
            "p99_ms": quality.p99_ms,
            "quality_score": quality.quality_score,
            "warmup_iters": quality.warmup_iters,
            "measurement_iters": quality.measurement_iters,
            "n_outliers_dropped": quality.n_outliers_dropped,
        },
    }


@app.local_entrypoint()
def main():
    """Local entry: invoke the A10G probe + render the RESULT block."""
    print("=" * 70)
    print("Auto-calibration matrix smoke -- Phase 1 minimum (A10G only)")
    print("=" * 70)
    r = probe_calibration_a10g.remote()
    print("\n=== RESULT ===")
    if r.get("status") == "fail":
        print(f"  status: FAIL")
        print(f"  reason: {r.get('reason', '(no reason)')}")
        return
    print(f"  status: {r['status']}")
    print(f"  tier:   {r['tier']}")
    fp = r["fingerprint"]
    print(f"  gpu:    {fp['gpu_name']!r} (uuid={fp['gpu_uuid'][:16]!r}...)")
    print(f"  driver: {fp['driver_major']}.{fp['driver_minor']}")
    print(f"  cuda:   {fp['cuda_major']}.{fp['cuda_minor']}")
    print(f"  ram:    {fp['ram_gb']}GB")
    print(f"  reflex: {fp['reflex_version']}")
    q = r["quality"]
    print(f"  median: {q['median_ms']:.4f} ms")
    print(f"  p99:    {q['p99_ms']:.4f} ms")
    print(f"  quality_score: {q['quality_score']:.4f}")
    print(f"  outliers_dropped: {q['n_outliers_dropped']}/{q['measurement_iters'] + q['n_outliers_dropped']}")
