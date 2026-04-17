#!/usr/bin/env python3
"""Trajectory smoke test for GOALS.yaml regression gate.

Quick mode (--quick): generates synthetic observation frames, runs them
through the exported ONNX sessions directly (no server needed), and
verifies output shapes, bounded actions, and smoothness. Runs in <30s.

Full mode: exports SmolVLA, starts reflex serve, downloads dataset
episodes, and replays them through /act with L2 error comparison.

Usage:
    python scripts/sim_smoke_test.py --quick       # <30s, GOALS.yaml gate
    python scripts/sim_smoke_test.py               # full local replay
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Pass/fail thresholds
L2_THRESHOLD = 2.0
QUICK_MAX_ACTION_MAGNITUDE = 50.0  # actions should be bounded
QUICK_SMOOTHNESS_THRESHOLD = 5.0  # consecutive action L2 diff


def quick_mode(export_dir: str | None = None) -> bool:
    """Run quick synthetic smoke test via ONNX sessions directly.

    Generates 3 synthetic observation frames, runs them through the
    exported ONNX expert_stack.onnx, and verifies:
    1. Output shape is correct (chunk_size x action_dim)
    2. Actions are bounded (no NaN/Inf, magnitude < threshold)
    3. Actions are smooth (consecutive frames produce similar outputs)

    Returns True if all checks pass.
    """
    print("=" * 60)
    print("QUICK SMOKE TEST (synthetic frames, ONNX direct)")
    print("=" * 60)

    passed = True
    results = []

    def log(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}: {name} -- {detail}")
        results.append({"check": name, "pass": ok, "detail": detail})
        if not ok:
            passed = False

    # Find export directory
    if export_dir is None:
        # Look for common export locations
        candidates = [
            Path("./reflex_export"),
            Path("/tmp/reflex_export"),
            Path("/tmp/reflex_traj_export"),
        ]
        for c in candidates:
            if (c / "expert_stack.onnx").exists():
                export_dir = str(c)
                break

    if export_dir is None or not Path(export_dir).exists():
        # Try to export first
        print("  No export directory found. Running reflex export...")
        export_dir = "/tmp/reflex_quick_export"
        r = subprocess.run(
            [
                sys.executable, "-m", "reflex.cli", "export",
                "lerobot/smolvla_base",
                "--target", "desktop",
                "--output", export_dir,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            # Try via the reflex command
            r = subprocess.run(
                [
                    "reflex", "export", "lerobot/smolvla_base",
                    "--target", "desktop",
                    "--output", export_dir,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        if r.returncode != 0:
            log("export", False, f"Export failed: {(r.stdout + r.stderr)[-300:]}")
            return False
        log("export", True, f"Exported to {export_dir}")

    export_path = Path(export_dir)
    onnx_path = export_path / "expert_stack.onnx"
    config_path = export_path / "reflex_config.json"

    if not onnx_path.exists():
        log("onnx_exists", False, f"{onnx_path} not found")
        return False
    log("onnx_exists", True, f"Found {onnx_path}")

    # Load config
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    expert_meta = config.get("expert", {})
    action_dim = expert_meta.get("action_dim", 32)
    chunk_size = config.get("action_chunk_size", 50)
    expert_hidden = expert_meta.get("expert_hidden", 720)
    vlm_kv_dim = config.get("vlm_kv_dim", 960)
    vlm_prefix_seq_len = config.get("vlm_prefix_seq_len", 50)

    log("config", True, f"action_dim={action_dim}, chunk={chunk_size}")

    # Load ONNX session
    try:
        import onnxruntime as ort
    except ImportError:
        log("onnxruntime", False, "onnxruntime not installed")
        return False

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        log("onnx_load", True, f"inputs={input_names}, outputs={output_names}")
    except Exception as e:
        log("onnx_load", False, str(e)[:200])
        return False

    # Check if expert expects vlm_kv
    has_vlm_kv = "vlm_kv" in input_names

    # Generate 3 synthetic frames and run denoising
    num_frames = 3
    all_actions = []
    np.random.seed(42)

    for frame_i in range(num_frames):
        print(f"\n  Frame {frame_i + 1}/{num_frames}:")

        # Simulate denoising loop: start from noise, run 10 Euler steps
        noisy_actions = np.random.randn(1, chunk_size, action_dim).astype(np.float32)
        position_ids = np.arange(chunk_size, dtype=np.int64).reshape(1, -1)

        num_steps = 10
        dt = -1.0 / num_steps

        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = np.array([t], dtype=np.float32)
            feed = {
                "noisy_actions": noisy_actions,
                "timestep": timestep,
                "position_ids": position_ids,
            }
            if has_vlm_kv:
                # Use zeros (dummy conditioning, same as server v0.1 fallback)
                vlm_kv = np.zeros(
                    (1, vlm_prefix_seq_len, vlm_kv_dim), dtype=np.float32
                )
                feed["vlm_kv"] = vlm_kv

            try:
                velocity = session.run(None, feed)[0]
                noisy_actions = noisy_actions + velocity * dt
            except Exception as e:
                log(f"denoise_frame{frame_i}", False, f"step {step}: {e}")
                return False

        actions = noisy_actions[0]  # [chunk_size, action_dim]
        all_actions.append(actions)

        # Check 1: Output shape
        expected_shape = (chunk_size, action_dim)
        shape_ok = actions.shape == expected_shape
        log(
            f"shape_frame{frame_i}",
            shape_ok,
            f"got {actions.shape}, expected {expected_shape}",
        )

        # Check 2: No NaN/Inf
        finite_ok = bool(np.all(np.isfinite(actions)))
        log(f"finite_frame{frame_i}", finite_ok, f"all finite: {finite_ok}")

        # Check 3: Bounded magnitude
        max_mag = float(np.max(np.abs(actions)))
        bounded_ok = max_mag < QUICK_MAX_ACTION_MAGNITUDE
        log(
            f"bounded_frame{frame_i}",
            bounded_ok,
            f"max |action|={max_mag:.4f} (threshold={QUICK_MAX_ACTION_MAGNITUDE})",
        )

    # Check 4: Smoothness between consecutive frames
    # With identical dummy conditioning, outputs should be similar
    print()
    for i in range(len(all_actions) - 1):
        diff = float(np.mean(np.linalg.norm(all_actions[i] - all_actions[i + 1], axis=-1)))
        smooth_ok = diff < QUICK_SMOOTHNESS_THRESHOLD
        log(
            f"smoothness_{i}_{i+1}",
            smooth_ok,
            f"mean L2 diff={diff:.4f} (threshold={QUICK_SMOOTHNESS_THRESHOLD})",
        )

    # Summary
    print("\n" + "-" * 44)
    total = len(results)
    pass_count = sum(1 for r in results if r["pass"])
    print(f"Quick smoke test: {pass_count}/{total} checks passed")
    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

    return passed


def full_mode(export_dir: str | None = None) -> bool:
    """Full trajectory replay: export, serve, download data, compute L2 error.

    Same logic as modal_trajectory_replay.py but runs locally.
    Returns True if mean L2 < threshold.
    """
    import threading

    try:
        import httpx
    except ImportError:
        print("FAIL: httpx not installed. pip install httpx")
        return False

    try:
        from PIL import Image
    except ImportError:
        print("FAIL: Pillow not installed. pip install Pillow")
        return False

    print("=" * 60)
    print("FULL TRAJECTORY REPLAY (local)")
    print("=" * 60)

    # Step 1: Export
    if export_dir is None:
        export_dir = "/tmp/reflex_full_export"

    export_path = Path(export_dir)
    if not (export_path / "expert_stack.onnx").exists():
        print("\n=== Step 1: Export SmolVLA ===")
        r = subprocess.run(
            [
                "reflex", "export", "lerobot/smolvla_base",
                "--target", "desktop",
                "--output", export_dir,
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            print(f"FAIL: Export failed\n{(r.stdout + r.stderr)[-500:]}")
            return False
        print(f"  Exported to {export_dir}")
    else:
        print(f"\n=== Step 1: Using existing export at {export_dir} ===")

    # Step 2: Start server
    print("\n=== Step 2: Start reflex serve ===")
    serve_port = 8399
    serve_proc = subprocess.Popen(
        [
            "reflex", "serve", export_dir,
            "--port", str(serve_port),
            "--device", "cpu",
            "--no-strict-providers",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_lines: list[str] = []

    def _drain():
        for line in serve_proc.stdout:
            log_lines.append(line.rstrip())

    threading.Thread(target=_drain, daemon=True).start()

    server_ready = False
    for attempt in range(60):
        time.sleep(2)
        try:
            resp = httpx.get(f"http://127.0.0.1:{serve_port}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                server_ready = True
                break
        except Exception:
            pass

    if not server_ready:
        print("FAIL: Server did not start")
        for line in log_lines[-20:]:
            print(f"  [server] {line}")
        serve_proc.terminate()
        return False
    print(f"  Server ready after {(attempt + 1) * 2}s")

    # Step 3: Download dataset
    print("\n=== Step 3: Download dataset ===")
    try:
        from datasets import load_dataset

        dataset = None
        for candidate in ["lerobot/xarm_lift_medium", "lerobot/pusht"]:
            try:
                dataset = load_dataset(candidate, split="train", streaming=True)
                print(f"  Loaded {candidate}")
                break
            except Exception as e:
                print(f"  {candidate} failed: {e}")
        if dataset is None:
            print("FAIL: No dataset available")
            serve_proc.terminate()
            return False
    except ImportError:
        print("FAIL: datasets library not installed. pip install datasets")
        serve_proc.terminate()
        return False

    # Step 4: Collect episodes
    print("\n=== Step 4: Collect episodes ===")
    episodes: dict[int, dict] = {}
    frames_seen = 0
    max_frames = 300
    num_episodes = 5

    for sample in dataset:
        if frames_seen >= max_frames:
            break
        ep_idx = sample.get("episode_index", 0)
        if len(episodes) >= num_episodes and ep_idx not in episodes:
            continue
        if ep_idx not in episodes:
            episodes[ep_idx] = {"frames": [], "actions": []}

        img = None
        for img_key in [
            "observation.images.top",
            "observation.image",
            "observation.images.front",
            "image",
        ]:
            if img_key in sample and sample[img_key] is not None:
                img = sample[img_key]
                break
        if img is None:
            for k, v in sample.items():
                if "image" in k.lower() and v is not None:
                    img = v
                    break

        action = sample.get("action", None)
        if action is None:
            continue
        if hasattr(action, "tolist"):
            action = action.tolist()
        elif isinstance(action, (list, tuple)):
            action = list(action)
        else:
            action = [float(action)]

        episodes[ep_idx]["frames"].append(img)
        episodes[ep_idx]["actions"].append(action)
        frames_seen += 1

    print(f"  {len(episodes)} episodes, {frames_seen} frames")

    # Step 5: Replay
    print("\n=== Step 5: Replay episodes ===")
    all_l2 = []
    ep_results = []
    instruction = "pick up the object"

    for ep_idx in sorted(episodes.keys())[:num_episodes]:
        ep = episodes[ep_idx]
        ep_l2 = []
        num_frames = len(ep["frames"])
        step_indices = list(range(0, num_frames, max(1, num_frames // 10)))[:10]

        for si in step_indices:
            raw_img = ep["frames"][si]
            expert_action = np.array(ep["actions"][si], dtype=np.float32)

            if isinstance(raw_img, Image.Image):
                pil_img = raw_img
            elif isinstance(raw_img, np.ndarray):
                pil_img = Image.fromarray(raw_img)
            elif isinstance(raw_img, dict) and "bytes" in raw_img:
                pil_img = Image.open(io.BytesIO(raw_img["bytes"]))
            else:
                continue

            buf = io.BytesIO()
            pil_img.convert("RGB").resize((224, 224)).save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            try:
                resp = httpx.post(
                    f"http://127.0.0.1:{serve_port}/act",
                    json={"image": img_b64, "instruction": instruction},
                    timeout=30,
                )
                result = resp.json()
                if "error" in result:
                    continue
                pred = np.array(result["actions"][0], dtype=np.float32)
                min_dim = min(len(pred), len(expert_action))
                l2 = float(np.linalg.norm(pred[:min_dim] - expert_action[:min_dim]))
                ep_l2.append(l2)
            except Exception as e:
                print(f"    /act error ep={ep_idx} step={si}: {e}")

        if ep_l2:
            em = float(np.mean(ep_l2))
            ex = float(np.max(ep_l2))
            ep_results.append((ep_idx, len(ep_l2), em, ex))
            all_l2.extend(ep_l2)
            print(f"  Episode {ep_idx}: {len(ep_l2)} steps, mean_L2={em:.4f}, max_L2={ex:.4f}")

    # Cleanup server
    serve_proc.terminate()
    serve_proc.wait(timeout=10)

    # Report
    print("\n" + "=" * 60)
    print("TRAJECTORY REPLAY RESULTS")
    print("=" * 60)

    if not all_l2:
        print("No frames replayed")
        return False

    mean_l2 = float(np.mean(all_l2))
    max_l2 = float(np.max(all_l2))

    print(f"\n{'Episode':<12} {'Steps':<8} {'Mean L2':<12} {'Max L2':<12}")
    print("-" * 44)
    for ep_idx, steps, em, ex in ep_results:
        print(f"{ep_idx:<12} {steps:<8} {em:<12.4f} {ex:<12.4f}")
    print("-" * 44)
    print(f"{'TOTAL':<12} {len(all_l2):<8} {mean_l2:<12.4f} {max_l2:<12.4f}")

    print(f"\nThreshold: mean L2 < {L2_THRESHOLD}")
    if mean_l2 < L2_THRESHOLD:
        print("RESULT: PASS")
        return True
    else:
        print("RESULT: FAIL")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory smoke test for Reflex VLA (GOALS.yaml regression gate)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: synthetic frames through ONNX directly (<30s)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Path to an existing reflex export directory",
    )
    args = parser.parse_args()

    if args.quick:
        ok = quick_mode(export_dir=args.export_dir)
    else:
        ok = full_mode(export_dir=args.export_dir)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
