"""LIBERO-10 eval — port of OpenPI's battle-tested rollout.

This is a rewrite (2026-04-19). Prior iterations (9 runs) got 0% because our
script missed critical pieces from the reference implementation. This version
is a line-by-line port of:

    openpi/examples/libero/main.py (battle-tested OpenPI LIBERO rollout)

cloned at `/Users/romirjain/Desktop/building projects/openpi/`, adapted for
in-process SmolVLAPolicy (OpenPI uses a WebsocketClient to a separate model
server; we call policy.select_action directly).

10 fixes applied vs prior version:
  1. max_steps=520 for libero_10 (was 300 — truncated episodes)
  2. Use OffScreenRenderEnv directly (was lerobot's gym-wrapped LiberoEnv —
     obs was nested under 'pixels' with missing state)
  3. Init state rotation: env.set_init_state(initial_states[episode_idx])
     per episode (fixes lerobot#2375 trap where every episode hits init_0)
  4. num_steps_wait=10 zero-action settling at episode start (objects are
     still falling)
  5. Correct _quat2axisangle (magnitude-preserving, copied from robosuite)
  6. Resize images to 224×224 with pad before policy inference
  7. replan_steps=5 action-plan deque (was: select_action every env step)
  8. Env resolution 256×256 (was: lerobot's 360×360 default)
  9. env.seed(7) for reproducibility
 10. 180° H+W image flip via numpy slicing (matches OpenPI exactly)

Usage:
    modal run scripts/modal_libero_lerobot_native.py --tasks 0 --num-episodes 1
    modal run scripts/modal_libero_lerobot_native.py --tasks all --num-episodes 5
"""
import os
import subprocess
import modal

app = modal.App("reflex-libero-lerobot-native")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


# Image: Python 3.12 + lerobot 0.5.1 + LIBERO + MuJoCo + robosuite 1.4.1.
# mujoco==3.3.2 per lerobot#2258 (older versions render different colors).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "build-essential",
        "libosmesa6", "libosmesa6-dev",
        "clang",
    )
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "mujoco==3.3.2",  # pinned per lerobot#2258
        "robosuite==1.4.1",
        "h5py",
        "bddl==1.0.1",
        "future",
        "robomimic",
        "hydra-core>=1.1",
        "easydict",
        "einops",
        "opencv-python-headless",
        "gym",
        "gymnasium",
        "lerobot==0.5.1",
        "num2words",
        "imageio",  # replay video save (optional)
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .env({
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
    })
    .run_commands("mkdir -p /tmp/libero_data")
)


# ─── Constants matching OpenPI reference ─────────────────────────────
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    secrets=[_hf_secret()],
)
def run_ported_libero(
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    num_episodes: int = 1,
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
):
    """Port of openpi/examples/libero/main.py rolled out end-to-end.

    Returns per-task success summary. Uses OffScreenRenderEnv directly
    (NOT lerobot's gym-wrapped LiberoEnv) for OpenPI-style obs access.
    """
    import collections
    import math
    import time
    import traceback
    import numpy as np
    import torch

    # ─── Load policy (in-process, not via websocket) ─────────────────
    print(f"[ported] Loading {model_id}...")
    t0 = time.time()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
    )
    from huggingface_hub import snapshot_download

    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval().to("cuda").to(torch.float32)
    repo_dir = snapshot_download(model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    print(f"[ported] Policy + preprocessor loaded in {time.time()-t0:.1f}s")

    # ─── Set up LIBERO suite ─────────────────────────────────────────
    np.random.seed(seed)
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
    print(f"[ported] suite={task_suite_name}, num_tasks={num_tasks_in_suite}, "
          f"max_steps={max_steps}")

    # Helper — _quat2axisangle, verbatim from OpenPI (robosuite formula).
    def _quat2axisangle(quat):
        if quat[3] > 1.0: quat[3] = 1.0
        elif quat[3] < -1.0: quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    # Helper — resize with pad (matches openpi_client.image_tools.resize_with_pad).
    # Pads to square first, then resizes. Preserves aspect ratio.
    def _resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image
        h, w = img.shape[:2]
        # Pad to square with zeros
        if h > w:
            pad = (h - w) // 2
            img = np.pad(img, [(0, 0), (pad, h - w - pad), (0, 0)], mode="constant")
        elif w > h:
            pad = (w - h) // 2
            img = np.pad(img, [(pad, w - h - pad), (0, 0), (0, 0)], mode="constant")
        # Resize
        pil = Image.fromarray(img)
        pil = pil.resize((size, size), Image.BILINEAR)
        return np.asarray(pil)

    def _build_env(task):
        task_bddl_file = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)
        return env

    def _build_batch(obs, task_description):
        """Build batch in SmolVLA's expected format, mirroring OpenPI obs."""
        # Images — 180° flip per LIBERO convention, resize to 224 with pad
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = _resize_with_pad(img, resize_size)
        wrist_img = _resize_with_pad(wrist_img, resize_size)

        def _to_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            return t.permute(2, 0, 1).unsqueeze(0).to("cuda")

        # State: eef_pos(3) + axis-angle(3) + gripper_qpos(2) = 8D
        state = np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]).astype(np.float32)

        batch = {
            "observation.images.image": _to_tensor(img),
            "observation.images.image2": _to_tensor(wrist_img),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to("cuda"),
            "task": [task_description],
        }
        return batch

    # ─── Results struct ──────────────────────────────────────────────
    results = {
        "model": model_id,
        "harness": "openpi-port-lerobot-native",
        "suite": task_suite_name,
        "num_episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "resize_size": resize_size,
        "replan_steps": replan_steps,
        "num_steps_wait": num_steps_wait,
        "per_task": [],
        "total_success": 0,
        "total_eps": 0,
        "errors": [],
    }

    tasks_to_run = task_indices if task_indices is not None else list(range(num_tasks_in_suite))
    print(f"[ported] Running tasks: {tasks_to_run}")

    # ─── Main loop: tasks × episodes ─────────────────────────────────
    for task_idx in tasks_to_run:
        task = task_suite.get_task(task_idx)
        task_description = task.language
        print(f"\n[ported] TASK {task_idx}: {task_description!r}")
        initial_states = task_suite.get_task_init_states(task_idx)
        print(f"[ported] {len(initial_states)} init states available")

        env = _build_env(task)
        task_start = time.time()
        task_result = {
            "task_idx": task_idx,
            "task_description": task_description,
            "episodes": [],
            "success": 0,
            "total": 0,
        }

        for ep in range(num_episodes):
            try:
                env.reset()
                # CRITICAL: rotate init state per episode (fixes #2375)
                init_idx = ep % len(initial_states)
                obs = env.set_init_state(initial_states[init_idx])
                policy.reset()
                action_plan = collections.deque()
                t = 0
                done = False

                while t < max_steps + num_steps_wait:
                    try:
                        # num_steps_wait: let objects settle
                        if t < num_steps_wait:
                            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Dump obs schema on first real step
                        if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                            obs_info = {
                                k: (obs[k].shape if hasattr(obs[k], "shape") else type(obs[k]).__name__)
                                for k in sorted(obs.keys())
                                if any(x in k.lower() for x in ["image", "eef", "gripper", "joint"])
                            }
                            print(f"[debug] obs keys: {obs_info}")

                        if not action_plan:
                            batch = _build_batch(obs, task_description)
                            batch_pp = preprocessor(batch)
                            batch_pp = {
                                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                                for k, v in batch_pp.items()
                            }
                            if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                                print(f"[debug] batch_pp keys: {sorted(batch_pp.keys())}")
                            with torch.no_grad():
                                chunk = policy.predict_action_chunk(batch_pp)
                                # chunk: (1, chunk_size, action_dim) → squeeze batch
                                chunk_np = chunk[0].cpu().numpy()
                            # Trim to 7-dim LIBERO action
                            chunk_np = chunk_np[:, :7]
                            action_plan.extend(chunk_np[:replan_steps])
                            if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                                print(f"[debug] first action: {chunk_np[0]}")

                        action = action_plan.popleft()
                        obs, _, done, info = env.step(action.tolist())
                        if done:
                            task_result["success"] += 1
                            results["total_success"] += 1
                            break
                        t += 1
                    except Exception as e:
                        err_tb = traceback.format_exc()
                        print(f"  step error: {e}")
                        print(err_tb[-800:])
                        results["errors"].append({
                            "task": task_idx, "ep": ep,
                            "error": str(e), "tb": err_tb[-400:],
                        })
                        break

                task_result["episodes"].append({
                    "ep": ep, "init_idx": init_idx, "steps": t, "success": done,
                })
                task_result["total"] += 1
                results["total_eps"] += 1
                print(f"  ep {ep} (init_idx={init_idx}): "
                      f"{'SUCCESS' if done else 'fail'} at {t} steps "
                      f"({time.time()-task_start:.1f}s total)")
            except Exception as e:
                err_tb = traceback.format_exc()
                print(f"  episode error: {e}")
                print(err_tb[-1000:])
                results["errors"].append({
                    "task": task_idx, "ep": ep,
                    "error": str(e), "tb": err_tb[-400:],
                })
                task_result["total"] += 1
                results["total_eps"] += 1

        results["per_task"].append(task_result)
        print(f"[ported] task {task_idx} done: "
              f"{task_result['success']}/{task_result['total']}")
        try:
            env.close()
        except Exception:
            pass

    success_rate = (
        100.0 * results["total_success"] / results["total_eps"]
        if results["total_eps"] else 0.0
    )
    results["success_rate_pct"] = round(success_rate, 1)
    print(f"\n====== {task_suite_name} (OpenPI-ported) ======")
    print(f"  Model: {model_id}")
    print(f"  Success: {results['total_success']}/{results['total_eps']} "
          f"= {success_rate:.1f}%")
    return results


@app.local_entrypoint()
def main(num_episodes: int = 1, tasks: str = "0", suite: str = "libero_10"):
    """
    --num-episodes N: episodes per task (OpenPI default: 50)
    --tasks "0"       single task
    --tasks "0,1,2"   multiple
    --tasks "all"     all tasks in suite
    --suite libero_10|libero_spatial|libero_object|libero_goal|libero_90
    """
    if tasks == "all":
        task_list = None  # run all
    else:
        task_list = [int(t) for t in tasks.split(",")]
    print(f"Running OpenPI-port LIBERO {suite}: tasks={task_list or 'all'}, "
          f"{num_episodes} eps each")
    r = run_ported_libero.remote(
        num_episodes=num_episodes,
        task_suite_name=suite,
        task_indices=task_list,
    )
    print("\n=== RESULT ===")
    print(f"  success_rate: {r.get('success_rate_pct', '?')}%")
    print(f"  total: {r['total_success']}/{r['total_eps']}")
    print(f"  errors: {len(r.get('errors', []))}")
    for task in r.get("per_task", []):
        print(f"  task {task['task_idx']}: "
              f"{task['success']}/{task['total']} — "
              f"{task['task_description'][:60]}")
