#!/usr/bin/env python3
"""Day-0 spike: verify two ORT InferenceSessions can coexist in one process.

Feature: policy-versioning (Phase 1 Track B #2).
ADR: 01_decisions/2026-04-25-policy-versioning-architecture.md
Research: features/01_serve/subfeatures/_ecosystem/policy-versioning/
          policy-versioning_research.md  (Lens 3 riskiest assumption)

The core assumption under test: two `Pi05DecomposedInference` instances
(or, at minimum, two ORT `InferenceSession` pairs loaded from the same
export) can coexist in one process without:

- CUDA context / stream / allocator corruption
- ORT logging singleton interleaving that breaks either session
- Divergent outputs vs. a single-instance baseline

This CPU-first variant runs against the export dir's vlm_prefix.onnx and
expert_denoise.onnx with random synthetic inputs. Deterministic seeds +
CPU EP isolate any ORT-level state leak from GPU-level concerns. A GPU
re-run (under CUDAExecutionProvider) is the actual gate for policy-
versioning ship; this script shipping green on CPU is necessary-but-not-
sufficient.

Usage:
    python scripts/spike_dual_instance_ort.py --export-dir <path> [--gpu]

PASS:
    - Both instances load cleanly
    - N alternating predicts produce bitwise-identical outputs to a
      single-instance baseline (within 1e-5 absolute on CPU; 5e-4 on GPU
      due to non-deterministic cuDNN conv algorithms)
    - No ORT log interleaving or segfault

FAIL + contingency:
    - Output divergence → refactor policy-versioning to spawn each policy
      in its own subprocess (heavier ~50ms IPC; correctness-safe).
    - CUDA deadlock / OOM at construction → same refactor.
    - ORT log interleaving that breaks state → disable ORT global logger
      (graph_optimization_level=0 as a debug stopgap) and retry.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def make_random_feed(session: ort.InferenceSession, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    feed = {}
    for inp in session.get_inputs():
        shape = [1 if (isinstance(d, str) or d is None) else int(d) for d in inp.shape]
        dtype = _ORT_TO_NP.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, 2, size=shape, dtype=dtype)
        else:
            arr = rng.standard_normal(shape).astype(dtype)
        feed[inp.name] = arr
    return feed


def build_session(model_path: Path, providers: list[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def max_abs_diff(a_outputs: list, b_outputs: list) -> float:
    """Element-wise max-abs-diff across all output tensors."""
    if len(a_outputs) != len(b_outputs):
        raise ValueError(f"output count mismatch: {len(a_outputs)} vs {len(b_outputs)}")
    worst = 0.0
    for a, b in zip(a_outputs, b_outputs):
        a_f = np.asarray(a, dtype=np.float32)
        b_f = np.asarray(b, dtype=np.float32)
        if a_f.shape != b_f.shape:
            raise ValueError(f"output shape mismatch: {a_f.shape} vs {b_f.shape}")
        diff = float(np.max(np.abs(a_f - b_f)))
        worst = max(worst, diff)
    return worst


def run_alternating(
    session_a: ort.InferenceSession,
    session_b: ort.InferenceSession,
    session_baseline: ort.InferenceSession,
    n_iters: int,
    seed_start: int,
    tol: float,
) -> dict:
    """Run n_iters alternating predicts on A and B; compare each against a
    fresh baseline session's prediction with the same feed."""
    max_diff_a = 0.0
    max_diff_b = 0.0
    t0 = time.perf_counter()
    for i in range(n_iters):
        seed = seed_start + i
        # Alternate A / B / A / B ...
        active, label, diff_ref = (
            (session_a, "a", None) if i % 2 == 0 else (session_b, "b", None)
        )
        feed = make_random_feed(active, seed=seed)
        outs = active.run(None, feed)

        # Baseline: run the SAME feed on a fresh session instance. If outputs
        # differ, then either A or B has drifted because of cross-session
        # state leak. (Feed is identical, so ORT kernels must be identical.)
        base_outs = session_baseline.run(None, feed)

        d = max_abs_diff(outs, base_outs)
        if label == "a":
            max_diff_a = max(max_diff_a, d)
        else:
            max_diff_b = max(max_diff_b, d)
    elapsed = time.perf_counter() - t0
    return {
        "n_iters": n_iters,
        "max_diff_a_vs_baseline": max_diff_a,
        "max_diff_b_vs_baseline": max_diff_b,
        "tolerance": tol,
        "pass": (max_diff_a <= tol and max_diff_b <= tol),
        "elapsed_s": round(elapsed, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--gpu", action="store_true", help="run under CUDA EP (default CPU)")
    ap.add_argument("--n-iters", type=int, default=100, help="alternating-predict iters")
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--tol", type=float, default=None,
                    help="max abs diff tolerance; default CPU=1e-5, GPU=5e-4")
    args = ap.parse_args()

    export = args.export_dir
    if not export.exists():
        print(f"FAIL: export dir not found: {export}")
        return 2

    vlm_path = export / "vlm_prefix.onnx"
    exp_path = export / "expert_denoise.onnx"
    if not vlm_path.exists() or not exp_path.exists():
        print(f"FAIL: expected decomposed export; missing vlm_prefix.onnx or "
              f"expert_denoise.onnx in {export}")
        return 2

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.gpu else ["CPUExecutionProvider"])
    tol = args.tol if args.tol is not None else (5e-4 if args.gpu else 1e-5)

    print(f"[spike_dual_instance_ort] providers={providers} tol={tol}")
    print(f"[spike_dual_instance_ort] loading 2 × (vlm_prefix + expert_denoise)...")

    t0 = time.perf_counter()
    session_a_vlm = build_session(vlm_path, providers)
    session_a_exp = build_session(exp_path, providers)
    session_b_vlm = build_session(vlm_path, providers)
    session_b_exp = build_session(exp_path, providers)
    baseline_vlm = build_session(vlm_path, providers)
    baseline_exp = build_session(exp_path, providers)
    load_s = time.perf_counter() - t0

    print(f"[spike_dual_instance_ort] loaded in {load_s:.2f}s "
          f"(6 sessions: a+b+baseline × vlm+exp)")

    # Run alternating on vlm_prefix and on expert_denoise independently — both
    # are load-bearing. The actual end-to-end pipeline chains them, but for
    # the state-isolation question, isolating each session type is sharper.
    print(f"\n[spike_dual_instance_ort] vlm_prefix — {args.n_iters} alternating iters")
    vlm_result = run_alternating(
        session_a_vlm, session_b_vlm, baseline_vlm,
        args.n_iters, args.seed_start, tol,
    )
    print(f"  result: {vlm_result}")

    print(f"\n[spike_dual_instance_ort] expert_denoise — {args.n_iters} alternating iters")
    exp_result = run_alternating(
        session_a_exp, session_b_exp, baseline_exp,
        args.n_iters, args.seed_start + 1_000_000, tol,
    )
    print(f"  result: {exp_result}")

    print("\n=== summary ===")
    overall_pass = vlm_result["pass"] and exp_result["pass"]
    print(f"vlm_prefix      PASS: {vlm_result['pass']}  max_diff_a={vlm_result['max_diff_a_vs_baseline']:.2e}  max_diff_b={vlm_result['max_diff_b_vs_baseline']:.2e}")
    print(f"expert_denoise  PASS: {exp_result['pass']}  max_diff_a={exp_result['max_diff_a_vs_baseline']:.2e}  max_diff_b={exp_result['max_diff_b_vs_baseline']:.2e}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        print("\nContingency per ADR 2026-04-25-policy-versioning-architecture:")
        print("  Refactor to subprocess IPC per policy (~50ms overhead, correctness-safe).")
        return 1
    print("\nPolicy-versioning Day-0 gate OPEN. Proceed to Day 1 (policy_router.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
