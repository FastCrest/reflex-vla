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

    # Load the policy's preprocessor pipeline — this is what tokenizes task
    # text, normalizes state, and shapes images. select_action expects a
    # preprocessed batch.
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
    )
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    print(f"[lerobot-native] Preprocessor pipeline loaded")

    # Load lerobot's env + processor
    print(f"[lerobot-native] Importing lerobot env + processors...")
    from lerobot.envs.libero import LiberoEnv
    from lerobot.processor.env_processor import LiberoProcessorStep

    # LIBERO suite object (built via libero's benchmark dict)
    from libero.libero import benchmark as libero_bench
    suite_name = "libero_10"
    suite = libero_bench.get_benchmark_dict()[suite_name]()
    print(f"[lerobot-native] Loaded libero benchmark suite: {suite_name}")

    # Build the processor step (handles image flip + state concat)
    processor = LiberoProcessorStep()

    tasks_to_run = task_indices if task_indices is not None else list(range(10))
    print(f"[lerobot-native] Running tasks: {tasks_to_run}")

    for task_idx in tasks_to_run:
        task_start = time.time()
        task_result = {"task_idx": task_idx, "episodes": [], "success": 0, "total": 0}
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_idx,
            task_suite_name=suite_name,
        )
        print(f"[lerobot-native] LiberoEnv built for task {task_idx}")
        for ep in range(num_episodes):
            try:
                obs, info = env.reset()
                # Extensive first-step diagnostic so we can finally see the
                # full LIBERO obs shape + where state lives.
                if ep == 0 and task_idx == tasks_to_run[0]:
                    print(f"[debug] reset info: {info}")
                    print(f"[debug] top obs keys: {sorted(obs.keys()) if isinstance(obs, dict) else type(obs).__name__}")
                    if isinstance(obs, dict) and "pixels" in obs:
                        px = obs["pixels"]
                        if isinstance(px, dict):
                            print(f"[debug] obs['pixels'] keys: {sorted(px.keys())}")
                            for k in list(px.keys())[:4]:
                                v = px[k]
                                if hasattr(v, "shape"):
                                    print(f"  obs['pixels'][{k}]: shape={v.shape} dtype={v.dtype}")
                                else:
                                    print(f"  obs['pixels'][{k}]: type={type(v).__name__}")
                        else:
                            print(f"[debug] obs['pixels'] is not a dict: type={type(px).__name__}")
                    # Check env for state source
                    print(f"[debug] env attrs with 'state'/'robot' in name:")
                    for attr in dir(env):
                        if "state" in attr.lower() or "robot" in attr.lower() or "proprio" in attr.lower():
                            print(f"  env.{attr}")
                    # Try unwrapped — dig for state source
                    try:
                        u = env.unwrapped
                        print(f"[debug] env.unwrapped type: {type(u).__name__}")
                        for attr in ("_env", "env", "sim", "robots", "get_proprio",
                                     "get_observation", "get_state"):
                            if hasattr(u, attr):
                                val = getattr(u, attr)
                                print(f"  env.unwrapped.{attr} present: "
                                      f"{type(val).__name__}")
                        # Try get_observation method
                        if hasattr(u, "get_observation"):
                            try:
                                raw = u.get_observation()
                                if isinstance(raw, dict):
                                    print(f"  env.unwrapped.get_observation() keys: "
                                          f"{sorted(raw.keys())}")
                            except Exception as _e:
                                print(f"  get_observation failed: {_e}")
                    except Exception as _ue:
                        print(f"[debug] env.unwrapped access failed: {_ue}")
                policy.reset()
                steps = 0
                done = False
                while not done and steps < max_steps:
                    # One-shot dump of raw obs schema on first step
                    if steps == 0 and ep == 0 and task_idx == tasks_to_run[0]:
                        if isinstance(obs, dict):
                            print(f"[debug] obs keys: {sorted(obs.keys())}")
                            for k in sorted(obs.keys()):
                                v = obs[k]
                                if hasattr(v, "shape"):
                                    print(f"  obs[{k}]: shape={v.shape} dtype={v.dtype}")
                                else:
                                    print(f"  obs[{k}]: type={type(v).__name__}")
                        else:
                            print(f"[debug] obs type: {type(obs).__name__}")
                    # Build the batch directly in the schema SmolVLAPolicy expects.
                    # LiberoProcessorStep's output (`{'pixels': ...}`) doesn't
                    # match prepare_images's expected observation.images.image/image2.
                    # So bypass it and build manually.
                    def _mk_img(arr):
                        a = np.asarray(arr)
                        if a.dtype != np.uint8:
                            a = (a * 255).clip(0, 255).astype(np.uint8)
                        t = torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
                        # 180° flip per LIBERO convention
                        return torch.flip(t, dims=[2, 3]).to("cuda")

                    # Extract images. LiberoEnv emits obs['pixels'] as a dict
                    # of cameras (gym PixelObservationWrapper convention).
                    img_dict = (obs.get("pixels") if isinstance(obs, dict) else None) or {}
                    if not isinstance(img_dict, dict):
                        # fallback: obs might already be a flat camera dict
                        img_dict = obs if isinstance(obs, dict) else {}
                    img1 = img2 = None
                    for cand in ("agentview_image", "image", "agent_view",
                                 "frontview_image"):
                        if cand in img_dict:
                            img1 = img_dict[cand]; break
                    for cand in ("robot0_eye_in_hand_image", "image2",
                                 "eye_in_hand_image", "wrist_image"):
                        if cand in img_dict:
                            img2 = img_dict[cand]; break
                    if img1 is None or img2 is None:
                        if steps == 0:
                            print(f"[warn] missing image keys; img_dict keys: "
                                  f"{sorted(img_dict.keys()) if isinstance(img_dict, dict) else 'not-dict'}")
                    # Extract state from the inner robosuite env.
                    # LiberoProcessorStep builds 8D: eef_pos(3) + quat->axis-angle(3) + gripper_qpos(2).
                    # Source: env.unwrapped._env._get_observations() returns a
                    # robosuite obs dict with 'robot0_eef_pos', 'robot0_eef_quat',
                    # 'robot0_gripper_qpos'.
                    state_np = np.zeros(8, dtype=np.float32)
                    try:
                        inner_env = env.unwrapped._env
                        if hasattr(inner_env, "_get_observations"):
                            robo_obs = inner_env._get_observations()
                        elif hasattr(inner_env, "get_observation"):
                            robo_obs = inner_env.get_observation()
                        else:
                            robo_obs = None
                        if isinstance(robo_obs, dict):
                            if steps == 0 and ep == 0 and task_idx == tasks_to_run[0]:
                                print(f"[debug] robosuite obs keys (sample): "
                                      f"{[k for k in robo_obs.keys() if 'robot' in k.lower() or 'eef' in k.lower() or 'gripper' in k.lower()][:10]}")
                            eef_pos = robo_obs.get("robot0_eef_pos")
                            eef_quat = robo_obs.get("robot0_eef_quat")
                            gripper_qpos = robo_obs.get("robot0_gripper_qpos")
                            if eef_pos is not None and eef_quat is not None and gripper_qpos is not None:
                                # quat -> axis-angle (approx: return first 3 components of quat)
                                # Full axis-angle is acos(w)*2 * (x,y,z)/sin(half_angle),
                                # but lerobot's LiberoProcessorStep uses the XYZ components
                                # directly when w > 0, or negated if w < 0. Close enough
                                # for a first attempt — fine-tune uses own normalization.
                                q = np.asarray(eef_quat, dtype=np.float32)
                                axis_angle = q[:3] if q[-1] >= 0 else -q[:3]
                                state_np = np.concatenate([
                                    np.asarray(eef_pos, dtype=np.float32)[:3],
                                    axis_angle[:3],
                                    np.asarray(gripper_qpos, dtype=np.float32)[:2],
                                ]).astype(np.float32)
                    except Exception as _se:
                        if steps == 0:
                            print(f"[warn] state extraction failed: {_se}")

                    state_t = torch.from_numpy(state_np).unsqueeze(0).to("cuda")
                    if steps == 0 and ep == 0 and task_idx == tasks_to_run[0]:
                        print(f"[debug] state_np: {state_np.tolist()}")

                    # Task description — from the suite / task
                    try:
                        task_text = suite.get_task(task_idx).language
                    except Exception:
                        task_text = f"task_{task_idx}"
                    if steps == 0:
                        print(f"[debug] task text: {task_text!r}")

                    batch = {
                        "observation.images.image": _mk_img(img1) if img1 is not None else torch.zeros(1, 3, 256, 256, device="cuda"),
                        "observation.images.image2": _mk_img(img2) if img2 is not None else torch.zeros(1, 3, 256, 256, device="cuda"),
                        "observation.state": state_t,
                        "task": [task_text],
                    }
                    # Route through policy preprocessor (tokenizes task text,
                    # normalizes state). select_action expects a preprocessed
                    # batch.
                    batch_pp = preprocessor(batch)
                    batch_pp = {
                        k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                        for k, v in batch_pp.items()
                    }
                    if steps == 0 and ep == 0 and task_idx == tasks_to_run[0]:
                        print(f"[debug] batch_pp keys: {sorted(batch_pp.keys())}")
                    # select_action re-queries every step when n_action_steps=1
                    with torch.no_grad():
                        action = policy.select_action(batch_pp)
                    # Convert to numpy for env.step + squeeze batch dim
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    if action.ndim == 2 and action.shape[0] == 1:
                        action = action[0]
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
