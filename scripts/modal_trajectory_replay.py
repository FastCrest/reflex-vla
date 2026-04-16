"""Trajectory replay smoke test for Reflex VLA on Modal.

Downloads episodes from a LeRobot dataset on HuggingFace, feeds each
observation frame through a reflex serve endpoint, and compares predicted
actions to expert ground truth via L2 error.

Usage:
    modal run scripts/modal_trajectory_replay.py
"""

import json
import sys
import time

import modal

app = modal.App("reflex-trajectory-replay")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "safetensors",
        "huggingface_hub",
        "transformers>=4.51",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "httpx>=0.24.0",
        "typer",
        "rich",
        "pyyaml",
        "datasets",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e .")
)

# Pass/fail threshold
L2_THRESHOLD = 2.0
NUM_EPISODES = 5


@app.function(gpu="A10G", image=image, timeout=1800, scaledown_window=60)
def run_trajectory_replay():
    """Full trajectory replay: export -> serve -> replay episodes -> report."""
    import base64
    import io
    import subprocess
    import threading

    import httpx
    import numpy as np
    from PIL import Image

    results = {
        "steps": [],
        "episodes": [],
        "pass": False,
        "mean_l2": None,
        "max_l2": None,
    }

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL" if status == "fail" else "INFO"
        print(f"{tag}: {name} -- {detail}")

    # ── Step 1: Export SmolVLA via `reflex export` ──────────────────
    print("\n=== Step 1: Export SmolVLA ===")
    export_dir = "/tmp/reflex_traj_export"
    t0 = time.time()
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
    elapsed = time.time() - t0
    if r.returncode == 0:
        log("export", "pass", f"SmolVLA exported in {elapsed:.1f}s")
    else:
        log("export", "fail", (r.stdout + r.stderr)[-500:])
        results["pass"] = False
        return results

    # Verify export files
    import os
    from pathlib import Path

    export_path = Path(export_dir)
    files = list(export_path.iterdir())
    log("export_files", "pass", f"{len(files)} files: {[f.name for f in files]}")

    # ── Step 2: Start reflex serve in background ────────────────────
    print("\n=== Step 2: Start reflex serve ===")
    serve_port = 8321
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

    # Collect server logs in background thread
    server_log_lines = []

    def _drain_stdout():
        for line in serve_proc.stdout:
            server_log_lines.append(line.rstrip())

    log_thread = threading.Thread(target=_drain_stdout, daemon=True)
    log_thread.start()

    # Wait for server to be ready (poll /health)
    server_ready = False
    for attempt in range(60):
        time.sleep(2)
        try:
            resp = httpx.get(f"http://127.0.0.1:{serve_port}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                if health.get("model_loaded"):
                    server_ready = True
                    break
        except Exception:
            pass

    if server_ready:
        log("serve_ready", "pass", f"Server ready after {(attempt + 1) * 2}s")
    else:
        # Dump last server logs for debugging
        for line in server_log_lines[-30:]:
            print(f"  [server] {line}")
        log("serve_ready", "fail", "Server did not become ready in 120s")
        serve_proc.terminate()
        results["pass"] = False
        return results

    # ── Step 3: Download dataset episodes ───────────────────────────
    print("\n=== Step 3: Download dataset episodes ===")
    try:
        from datasets import load_dataset

        # lerobot/xarm_lift_medium is small, has image observations + actions.
        # Try it first, fall back to pusht.
        dataset = None
        dataset_name = None
        for candidate in [
            "lerobot/xarm_lift_medium",
            "lerobot/pusht",
        ]:
            try:
                print(f"  Trying {candidate}...")
                dataset = load_dataset(candidate, split="train", streaming=True)
                dataset_name = candidate
                break
            except Exception as e:
                print(f"  {candidate} failed: {e}")
                continue

        if dataset is None:
            log("dataset", "fail", "Could not load any candidate dataset")
            serve_proc.terminate()
            results["pass"] = False
            return results

        log("dataset", "pass", f"Loaded {dataset_name} (streaming)")
    except Exception as e:
        log("dataset", "fail", str(e))
        serve_proc.terminate()
        results["pass"] = False
        return results

    # ── Step 4: Group into episodes and replay ──────────────────────
    print("\n=== Step 4: Replay episodes ===")

    # Collect frames grouped by episode
    # LeRobot datasets have "episode_index", "observation.image" (or similar),
    # and "action" columns.
    episodes = {}
    frames_seen = 0
    max_frames = 500  # cap total frames to keep runtime reasonable

    try:
        for sample in dataset:
            if frames_seen >= max_frames:
                break

            ep_idx = sample.get("episode_index", 0)
            if len(episodes) >= NUM_EPISODES and ep_idx not in episodes:
                # Already have enough episodes, skip new ones
                continue

            if ep_idx not in episodes:
                episodes[ep_idx] = {"frames": [], "actions": []}

            # Extract image -- LeRobot uses PIL images in "observation.images.top"
            # or "observation.image" depending on dataset version
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
                # Try to find any key with "image" in the name
                for k, v in sample.items():
                    if "image" in k.lower() and v is not None:
                        img = v
                        break

            # Extract action
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

        log("collect", "pass", f"{len(episodes)} episodes, {frames_seen} total frames")
    except Exception as e:
        log("collect", "fail", str(e)[:300])
        serve_proc.terminate()
        results["pass"] = False
        return results

    if len(episodes) == 0:
        log("replay", "fail", "No episodes collected")
        serve_proc.terminate()
        results["pass"] = False
        return results

    # ── Step 5: Replay each episode through /act ────────────────────
    print("\n=== Step 5: Compute L2 errors ===")
    all_l2_errors = []
    instruction = "pick up the object"  # generic instruction

    for ep_idx in sorted(episodes.keys())[:NUM_EPISODES]:
        ep = episodes[ep_idx]
        ep_l2_errors = []
        num_frames = len(ep["frames"])
        # Sample up to 10 frames per episode to keep it fast
        step_indices = list(range(0, num_frames, max(1, num_frames // 10)))[:10]

        for step_i in step_indices:
            img = ep["frames"][step_i]
            expert_action = np.array(ep["actions"][step_i], dtype=np.float32)

            # Encode image to base64
            if isinstance(img, Image.Image):
                pil_img = img
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img)
            elif isinstance(img, dict) and "bytes" in img:
                pil_img = Image.open(io.BytesIO(img["bytes"]))
            else:
                # Skip frames we can't decode
                continue

            buf = io.BytesIO()
            pil_img.convert("RGB").resize((224, 224)).save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # POST /act
            try:
                resp = httpx.post(
                    f"http://127.0.0.1:{serve_port}/act",
                    json={
                        "image": img_b64,
                        "instruction": instruction,
                        "state": [0.0] * 6,
                    },
                    timeout=30,
                )
                result = resp.json()

                if "error" in result:
                    print(f"  /act error at ep={ep_idx} step={step_i}: {result['error']}")
                    continue

                pred_actions = np.array(result["actions"], dtype=np.float32)
                # Compare first action in chunk to expert action
                # Expert action dim may differ from model output dim; truncate to min
                pred_first = pred_actions[0]
                min_dim = min(len(pred_first), len(expert_action))
                l2 = float(np.linalg.norm(pred_first[:min_dim] - expert_action[:min_dim]))
                ep_l2_errors.append(l2)
            except Exception as e:
                print(f"  /act failed at ep={ep_idx} step={step_i}: {e}")
                continue

        if ep_l2_errors:
            ep_mean = float(np.mean(ep_l2_errors))
            ep_max = float(np.max(ep_l2_errors))
            results["episodes"].append({
                "episode_index": int(ep_idx),
                "num_steps": len(ep_l2_errors),
                "mean_l2": round(ep_mean, 4),
                "max_l2": round(ep_max, 4),
            })
            all_l2_errors.extend(ep_l2_errors)
            print(
                f"  Episode {ep_idx}: {len(ep_l2_errors)} steps, "
                f"mean_L2={ep_mean:.4f}, max_L2={ep_max:.4f}"
            )

    # ── Step 6: Report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAJECTORY REPLAY RESULTS")
    print("=" * 60)

    if all_l2_errors:
        mean_l2 = float(np.mean(all_l2_errors))
        max_l2 = float(np.max(all_l2_errors))
        results["mean_l2"] = round(mean_l2, 4)
        results["max_l2"] = round(max_l2, 4)

        print(f"\n{'Episode':<12} {'Steps':<8} {'Mean L2':<12} {'Max L2':<12}")
        print("-" * 44)
        for ep in results["episodes"]:
            print(
                f"{ep['episode_index']:<12} {ep['num_steps']:<8} "
                f"{ep['mean_l2']:<12.4f} {ep['max_l2']:<12.4f}"
            )
        print("-" * 44)
        print(
            f"{'TOTAL':<12} {len(all_l2_errors):<8} "
            f"{mean_l2:<12.4f} {max_l2:<12.4f}"
        )

        print(f"\nThreshold: mean L2 < {L2_THRESHOLD}")
        if mean_l2 < L2_THRESHOLD:
            print("RESULT: PASS")
            results["pass"] = True
            log("trajectory_replay", "pass", f"mean_L2={mean_l2:.4f} < {L2_THRESHOLD}")
        else:
            print("RESULT: FAIL")
            results["pass"] = False
            log(
                "trajectory_replay",
                "fail",
                f"mean_L2={mean_l2:.4f} >= {L2_THRESHOLD}",
            )
    else:
        print("No L2 errors computed -- no frames were successfully replayed")
        results["pass"] = False
        log("trajectory_replay", "fail", "No frames replayed")

    # Cleanup
    serve_proc.terminate()
    serve_proc.wait(timeout=10)

    return results


@app.local_entrypoint()
def main():
    """Run trajectory replay and report results."""
    print("Starting trajectory replay smoke test on Modal (A10G)...")
    result = run_trajectory_replay.remote()

    print("\n" + "=" * 60)
    print("LOCAL SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    if result.get("pass"):
        print("\nSMOKE TEST PASSED")
        sys.exit(0)
    else:
        print("\nSMOKE TEST FAILED")
        sys.exit(1)
