#!/usr/bin/env python3
"""Day-0 spike: verify ORT CUDA graphs capture works on pi0.5 decomposed export.

Usage:
    python scripts/spike_cuda_graphs_ort.py --export-dir <path>

PASS criteria:
- Both vlm_prefix.onnx and expert_denoise.onnx capture cleanly on first run
- Second run replays successfully
- Outputs match eager-mode outputs (atol=1e-6 default)

FAIL criteria + contingency:
- Capture exception -> ORT rejects an op in our export. Contingency A
  (re-export workaround per cuda-graphs_research.md Lens 7).
- Replay divergence -> ORT path not viable. Contingency B (kill for
  Phase 1; revisit torch.cuda.graph in Phase 2).

Reference: features/01_serve/subfeatures/_perf_compound/cuda-graphs/cuda-graphs_plan.md
ADR:       01_decisions/2026-04-24-cuda-graphs-architecture.md
"""
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


def make_random_feed(session: ort.InferenceSession, seed: int = 0) -> dict:
    """Build a feed dict of synthetic inputs matching the session's input shapes.

    Any dynamic dim (None or symbolic) defaults to 1. Decomposed pi0.5 ships
    static-shape exports (per ADR 2026-04-21), so dynamic dims should not appear.
    """
    rng = np.random.default_rng(seed)
    feed = {}
    for inp in session.get_inputs():
        shape = [1 if (isinstance(d, str) or d is None) else int(d) for d in inp.shape]
        dtype = _ORT_TO_NP.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            feed[inp.name] = (rng.integers(0, 2, size=shape) > 0)
        else:
            feed[inp.name] = rng.integers(0, 100, size=shape, dtype=dtype)
    return feed


def assert_close(label: str, a: list, b: list, atol: float) -> None:
    for i, (xa, xb) in enumerate(zip(a, b)):
        np.testing.assert_allclose(
            xa, xb, atol=atol, err_msg=f"{label} output[{i}] diverged"
        )


def spike_one(model_path: Path, atol: float, name: str) -> dict:
    print(f"\n=== Spike: {name} ({model_path.name}) ===")
    if not model_path.exists():
        print(f"  SKIP: {model_path} does not exist")
        return {"passed": False, "reason": "file_missing"}

    # Build the eager session to introspect input shapes
    eager_providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    eager = ort.InferenceSession(str(model_path), providers=eager_providers)
    # Hard gate: must actually be on CUDA, not silently fallen back to CPU
    active_providers = eager.get_providers()
    if "CUDAExecutionProvider" not in active_providers:
        print(f"  FAIL: CUDAExecutionProvider not active. Got providers={active_providers}")
        print(f"         Check nvidia-smi, cuDNN, cuBLAS lib availability on the host.")
        return {"passed": False, "reason": "cuda_ep_not_loaded",
                "active_providers": active_providers}
    feed = make_random_feed(eager, seed=0)
    print(f"  Inputs: {[(i.name, list(i.shape), i.type) for i in eager.get_inputs()]}")
    print(f"  Active providers (eager): {active_providers}")

    # Eager forward (baseline output + timing)
    print("  Running eager...", flush=True)
    t0 = time.perf_counter()
    eager_out = eager.run(None, feed)
    eager_ms = (time.perf_counter() - t0) * 1000
    print(f"  Eager: {eager_ms:.1f} ms")

    # CUDA graph capture (first call) + replay (second call)
    # Minimal provider options — 2026-04-24 diagnostic spike confirmed that
    # cudnn_conv_algo_search="HEURISTIC" + arena_extend_strategy="kSameAsRequested"
    # selected non-deterministic conv algorithms that caused output divergence
    # vs eager (max abs err 0.008, 79% mismatch at atol=1e-6). With minimal
    # flags, vlm_prefix captured at perfect parity (cos=1.0, abs_max=0) on A100.
    # Keep this list minimal unless a new diagnostic proves otherwise.
    print("  Running with enable_cuda_graph=1 (minimal provider options)...", flush=True)
    cg_providers = [
        ("CUDAExecutionProvider", {"enable_cuda_graph": "1"}),
        "CPUExecutionProvider",
    ]
    try:
        cg_session = ort.InferenceSession(str(model_path), providers=cg_providers)
        cg_active = cg_session.get_providers()
        if "CUDAExecutionProvider" not in cg_active:
            print(f"  FAIL: CUDAExecutionProvider not active on cuda-graph session. Got {cg_active}")
            return {"passed": False, "reason": "cuda_ep_not_loaded_cg",
                    "active_providers": cg_active}
        print(f"  Active providers (cuda-graph): {cg_active}")
        t0 = time.perf_counter()
        cg_capture = cg_session.run(None, feed)
        capture_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cg_replay = cg_session.run(None, feed)
        replay_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"  FAIL: capture/replay raised {type(e).__name__}: {e}")
        return {
            "passed": False,
            "reason": f"capture_exception:{type(e).__name__}",
            "exc_msg": str(e),
        }

    print(f"  Capture: {capture_ms:.1f} ms (includes graph build)")
    print(f"  Replay:  {replay_ms:.1f} ms")

    # Parity: capture output matches eager bit-similarly
    try:
        assert_close(f"{name} capture vs eager", eager_out, cg_capture, atol=atol)
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return {"passed": False, "reason": "capture_vs_eager_diverged"}

    # Idempotence: replay output matches capture output (should be exact)
    try:
        assert_close(f"{name} replay vs capture", cg_capture, cg_replay, atol=1e-9)
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return {"passed": False, "reason": "replay_vs_capture_diverged"}

    speedup = eager_ms / replay_ms if replay_ms > 0 else float("inf")
    print(f"  PASS: replay {speedup:.2f}x faster than eager")
    return {
        "passed": True,
        "eager_ms": eager_ms,
        "capture_ms": capture_ms,
        "replay_ms": replay_ms,
        "speedup": speedup,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", type=Path, required=True,
                   help="Directory containing vlm_prefix.onnx + expert_denoise.onnx")
    p.add_argument("--atol", type=float, default=1e-6,
                   help="Absolute tolerance for capture-vs-eager parity (default 1e-6)")
    args = p.parse_args()

    print("=" * 60)
    print("Day-0 ORT CUDA Graphs verification spike")
    print(f"Export dir: {args.export_dir}")
    print(f"Tolerance:  atol={args.atol}")
    print("=" * 60)

    results: dict[str, dict] = {}
    for fname, label in (("vlm_prefix.onnx", "vlm_prefix"),
                         ("expert_denoise.onnx", "expert_denoise")):
        results[label] = spike_one(args.export_dir / fname, atol=args.atol, name=label)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for label, r in results.items():
        status = "PASS" if r["passed"] else f"FAIL ({r['reason']})"
        print(f"  {label}: {status}")
        if r["passed"]:
            print(f"    eager={r['eager_ms']:.1f}ms  "
                  f"capture={r['capture_ms']:.1f}ms  "
                  f"replay={r['replay_ms']:.1f}ms  "
                  f"speedup={r['speedup']:.2f}x")
        else:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: PASS - proceed with cuda-graphs Day 1+ implementation")
        print()
        print("Next: write 03_experiments/2026-04-24-cuda-graphs-ort-spike.md")
        print("      noting the result + speedup numbers.")
        return 0
    print("OVERALL: FAIL")
    print()
    print("Contingency triage (per cuda-graphs_research.md Lens 7):")
    print("  capture_exception:*       -> Contingency A (re-export workaround, +3 days)")
    print("  capture_vs_eager_diverged -> ORT capture corrupts forward; investigate the op set")
    print("  replay_vs_capture_diverged -> ORT replay non-deterministic; Contingency B")
    print("                                (kill cuda-graphs for Phase 1; revisit torch path Phase 2)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
