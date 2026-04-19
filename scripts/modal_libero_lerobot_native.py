"""LIBERO-10 eval using lerobot's OWN harness (not vla-eval).

After 3 vla-eval runs hit 0% with every config permutation (community +
canonical checkpoint × flip on/off × camera keys camera1/2/3 vs image/image2),
adapter-vs-raw-policy parity came back bit-exact (cos=+1.000000, max_abs=0).
This means the wrapping pipeline is clean — the 0% is a vla-eval harness
integration issue, not a model or adapter bug.

This script bypasses vla-eval entirely. It uses:
  - lerobot.envs.libero.LiberoEnv (lerobot's own LIBERO wrapper)
  - lerobot.processor.env_processor.LiberoProcessorStep (the flip + state
    concat + normalization step lerobot uses internally)
  - lerobot.policies.smolvla.SmolVLAPolicy.select_action (the deque-backed
    per-step re-query the paper uses with n_action_steps=1)

Goal: run the canonical HuggingFaceVLA/smolvla_libero checkpoint against
libero_10 for N rollouts. If we see non-zero success (ideally ~71% per the
paper), that CONFIRMS our model + loading is correct and the gap was at
the vla-eval integration layer.

Usage:
    modal run scripts/modal_libero_lerobot_native.py
    modal run scripts/modal_libero_lerobot_native.py --num-episodes 10
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


# Same image as the customer dogfood: Python 3.12 + lerobot==0.5.1 + all
# LIBERO / MuJoCo / robosuite deps. lerobot 0.5.1 requires Python 3.12.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "build-essential",
        "libosmesa6", "libosmesa6-dev",  # software MuJoCo rendering
        "clang",  # evdev C build in lerobot deps
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
        "mujoco>=3.0",
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[_hf_secret()],
)
def run_lerobot_native_libero(
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    num_episodes: int = 1,
    max_steps: int = 300,
    task_indices: list[int] | None = None,
):
    """Run LIBERO-10 using lerobot's native env + SmolVLAPolicy.select_action.

    Returns a per-task success summary.
    """
    import time
    import traceback
    import numpy as np
    import torch

    results = {
        "model": model_id,
        "harness": "lerobot-native",
        "num_episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "per_task": [],
        "total_success": 0,
        "total_eps": 0,
        "errors": [],
    }

    print(f"[lerobot-native] Loading {model_id}...")
    t0 = time.time()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval().to("cuda").to(torch.float32)
    # Paper uses n_action_steps=1 — re-query every env step.
    # predict_action_chunk still produces chunk_size actions; select_action()
    # only pops 1 per call then triggers a fresh run once queue drains.
    try:
        policy.config.n_action_steps = 1
    except Exception:
        pass
    print(f"[lerobot-native] Policy loaded in {time.time()-t0:.1f}s")

    # Load lerobot's env + processor
    print(f"[lerobot-native] Importing lerobot env + processors...")
    from lerobot.envs.libero import LiberoEnv
    from lerobot.processor.env_processor import LiberoProcessorStep
    import inspect as _inspect

    # Introspect LiberoEnv.__init__ so we don't guess kwarg names
    sig = _inspect.signature(LiberoEnv.__init__)
    print(f"[lerobot-native] LiberoEnv.__init__ params: {list(sig.parameters)}")

    # Build the processor step (handles image flip + state concat)
    processor = LiberoProcessorStep()

    tasks_to_run = task_indices if task_indices is not None else list(range(10))
    print(f"[lerobot-native] Running tasks: {tasks_to_run}")

    def _build_env(task_idx: int):
        """Try several parameter naming conventions for LiberoEnv."""
        attempts = [
            lambda: LiberoEnv(task_id=task_idx, task_suite="libero_10"),
            lambda: LiberoEnv(task_id=task_idx, task_suite_name="libero_10"),
            lambda: LiberoEnv(task_idx, "libero_10"),
            lambda: LiberoEnv("libero_10", task_idx),
        ]
        last_err = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as e:
                last_err = e
                continue
        raise last_err

    for task_idx in tasks_to_run:
        task_start = time.time()
        task_result = {"task_idx": task_idx, "episodes": [], "success": 0, "total": 0}
        env = _build_env(task_idx)
        print(f"[lerobot-native] LiberoEnv built for task {task_idx}")
        for ep in range(num_episodes):
            try:
                obs, _ = env.reset()
                policy.reset()
                steps = 0
                done = False
                while not done and steps < max_steps:
                    # Apply lerobot's processor step (image flip + state concat)
                    processed = processor.process_observation(obs)
                    # select_action re-queries every step when n_action_steps=1
                    with torch.no_grad():
                        action = policy.select_action(processed)
                    # Convert to numpy for env.step
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    # env.step returns (obs, reward, terminated, truncated, info)
                    step_out = env.step(action)
                    if len(step_out) == 5:
                        obs, reward, terminated, truncated, info = step_out
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_out  # older gym api
                    steps += 1
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                task_result["episodes"].append({
                    "ep": ep, "steps": steps, "success": success,
                })
                task_result["total"] += 1
                if success:
                    task_result["success"] += 1
                    results["total_success"] += 1
                results["total_eps"] += 1
                print(f"  task {task_idx} ep {ep}: "
                      f"{'SUCCESS' if success else 'fail'} at {steps} steps "
                      f"({time.time()-task_start:.1f}s)")
            except Exception as e:
                tb = traceback.format_exc()
                err = f"task {task_idx} ep {ep}: {type(e).__name__}: {e}"
                print(f"  ERROR — {err}")
                print(tb[-1500:])
                results["errors"].append({"task": task_idx, "ep": ep,
                                          "error": str(e), "traceback": tb[-500:]})
                task_result["episodes"].append({"ep": ep, "error": str(e)})
                task_result["total"] += 1
                results["total_eps"] += 1
        results["per_task"].append(task_result)
        print(f"task {task_idx} done: {task_result['success']}/{task_result['total']}")
        try:
            env.close()
        except Exception:
            pass

    success_rate = (
        100.0 * results["total_success"] / results["total_eps"]
        if results["total_eps"] else 0.0
    )
    results["success_rate_pct"] = round(success_rate, 1)
    print(f"\n====== LIBERO-10 (lerobot-native) ======")
    print(f"  Model: {model_id}")
    print(f"  Success: {results['total_success']}/{results['total_eps']} "
          f"= {success_rate:.1f}%")
    return results


@app.local_entrypoint()
def main(num_episodes: int = 1, tasks: str = "0"):
    """
    --num-episodes N: episodes per task (default 1)
    --tasks "0"       single task (fast)
    --tasks "0,1,2"   multiple tasks
    --tasks "all"     all 10 LIBERO-10 tasks
    """
    if tasks == "all":
        task_list = list(range(10))
    else:
        task_list = [int(t) for t in tasks.split(",")]
    print(f"Running lerobot-native LIBERO-10: tasks={task_list}, "
          f"{num_episodes} eps each")
    r = run_lerobot_native_libero.remote(
        num_episodes=num_episodes,
        task_indices=task_list,
    )
    print("\n=== RESULT ===")
    print(f"  success_rate: {r.get('success_rate_pct', '?')}%")
    print(f"  total: {r['total_success']}/{r['total_eps']}")
    print(f"  errors: {len(r.get('errors', []))}")
    for task in r.get("per_task", []):
        print(f"  task {task['task_idx']}: "
              f"{task['success']}/{task['total']}")
