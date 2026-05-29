"""Lift #5 L3 side-by-side — native lerobot vs Triton on the SAME proven loop.

Quality-parity gate for the `--fast-kernels` Triton path. Runs BOTH arms through
`reflex.eval.libero_rollout.run_libero_rollout` — the exact primitive the proven
native eval (`modal_libero_lerobot_native.py`, 80%+ on libero_10 task 0 at N=20)
uses — so the ONLY thing that differs between arms is the inference backend:

  ARM A (native): use_native=True  → policy.select_action (proven path)
  ARM B (triton): inference=TritonLIBEROAdapter, use_native=False

Identical preprocessing (cv2 INTER_AREA resize + 180° flip), identical seed,
identical task set, identical centrally-generated noise, identical bool(done)
success criterion. This REPLACES the earlier bespoke loop whose PIL-BILINEAR
resize + seed-42 + hard-task-set [3,4,6] produced a spurious 0/9 "baseline"
(see 03_experiments/2026-05-24-lift5-l3-sbs-baseline-zero.md). There was never
a proven native run at that config to compare against.

Usage:
    modal profile activate novarepmarketing
    # cheap baseline diagnostic — native arm only, confirm baseline > 0:
    modal run scripts/modal_fast_kernels_l3_side_by_side.py --arms native \
        --task-indices 0 --num-episodes 2
    # full paired parity gate:
    modal run scripts/modal_fast_kernels_l3_side_by_side.py --arms both \
        --task-indices 0,1,2 --num-episodes 10
"""
import os
import subprocess

import modal

app = modal.App("reflex-fast-kernels-l3-side-by-side")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "lift/5-day1-2-vendor-triton-kernels"


_HEAD = _repo_head_sha()


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "ninja-build", "clang", "build-essential",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "libosmesa6", "libosmesa6-dev",
        "gnupg", "wget",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb"
        " && apt-get update"
        " && apt-get install -y cuda-toolkit-12-4 --no-install-recommends"
        " && rm cuda-keyring_1.1-1_all.deb",
    )
    .pip_install(
        "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "triton>=3.1", "ninja",
        "mujoco==3.3.2", "robosuite==1.4.1",
        "h5py", "bddl==1.0.1", "future", "robomimic",
        "hydra-core>=1.1", "easydict", "einops",
        "opencv-python-headless", "gym", "gymnasium",
        "lerobot==0.5.1", "num2words", "imageio",
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
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
    image=image, gpu="A100-40GB", timeout=7200,
    secrets=[_hf_secret()],
)
def run_side_by_side(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    num_episodes: int = 10,
    seed: int = 7,
    arms: str = "both",  # "native" | "triton" | "both"
) -> dict:
    """Native vs Triton on the shared proven rollout loop. See module docstring."""
    import time

    import torch

    # PyTorch 2.6+ defaults torch.load to weights_only=True; LIBERO init-state
    # pickles need weights_only=False. (run_libero_rollout patches this too, but
    # the policy load below happens first.)
    _orig_torch_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_load

    if task_indices is None:
        # Anchor on task 0 (proven ~80%+ native at N=20) + 1,2 for breadth.
        task_indices = [0, 1, 2]

    # PyTorch Inductor autotuner blocks the GPU 30+s per matmul shape on first
    # call → Modal kills the task for "failed to respond to cancellation".
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"[sbs] suite={task_suite_name} tasks={task_indices} "
        f"N={num_episodes} seed={seed} arms={arms}",
        flush=True,
    )
    print(f"[sbs] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    t_total = time.time()

    from reflex.eval.libero_rollout import run_libero_rollout

    # ── Load policy (fp32 cuda — native baseline quality + shared preprocessing)
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id).to(dtype=torch.float32).to("cuda")
    policy.eval()
    print(f"[sbs] [{time.time()-t0:.1f}s] PI05Policy loaded (cuda fp32)", flush=True)

    # ── Pre/post processors (shared by both arms via the rollout) ─────
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(repo_id=model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print("[sbs] pre/post processors loaded", flush=True)

    common = dict(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        replan_steps=5,
        num_steps_wait=10,
    )

    native = None
    triton = None

    # ── ARM A: native (proven select_action path) ────────────────────
    if arms in ("native", "both"):
        print(f"\n[sbs] {'='*60}", flush=True)
        print("[sbs] ARM A: native lerobot (select_action, fp32)", flush=True)
        print(f"[sbs] {'='*60}", flush=True)
        native = run_libero_rollout(
            inference=None, use_native=True, label="NATIVE", **common,
        )

    # ── ARM B: Triton fast-kernels ───────────────────────────────────
    if arms in ("triton", "both"):
        print(f"\n[sbs] {'='*60}", flush=True)
        print("[sbs] ARM B: Triton fast-kernels (Pi05FastKernelsInference, bf16)", flush=True)
        print(f"[sbs] {'='*60}", flush=True)
        from reflex.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter
        adapter = TritonLIBEROAdapter.from_policy(policy, capture=True)
        triton = run_libero_rollout(
            inference=adapter, use_native=False, label="TRITON", **common,
        )

    # ── Compare ───────────────────────────────────────────────────────
    out: dict = {
        "native": native,
        "triton": triton,
        "task_indices": task_indices,
        "num_episodes": num_episodes,
        "seed": seed,
        "arms": arms,
    }
    print(f"\n[sbs] {'='*60}", flush=True)
    if native is not None:
        nr = native.get("success_rate_pct", 0.0)
        out["native_rate_pct"] = nr
        print(f"[sbs] NATIVE:  {native['total_success']}/{native['total_eps']} ({nr:.1f}%)", flush=True)
    if triton is not None:
        tr = triton.get("success_rate_pct", 0.0)
        out["triton_rate_pct"] = tr
        print(f"[sbs] TRITON:  {triton['total_success']}/{triton['total_eps']} ({tr:.1f}%)", flush=True)
    if native is not None and triton is not None:
        delta = out["triton_rate_pct"] - out["native_rate_pct"]
        out["delta_pp"] = delta
        print(
            f"[sbs] Delta:   {delta:+.1f}pp  "
            f"(kill-trigger: native−triton regression > 5pp)",
            flush=True,
        )
    print(f"[sbs] Total time: {time.time()-t_total:.1f}s", flush=True)
    print(f"[sbs] {'='*60}", flush=True)
    return out


@app.local_entrypoint()
def main(
    suite: str = "libero_10",
    task_indices: str = "",
    num_episodes: int = 10,
    seed: int = 7,
    arms: str = "both",
):
    print("=" * 70)
    print("Lift #5 L3 side-by-side: native lerobot vs Triton (proven rollout loop)")
    print("=" * 70)
    idx = [int(x) for x in task_indices.split(",") if x.strip()] or None
    result = run_side_by_side.remote(
        task_suite_name=suite,
        task_indices=idx,
        num_episodes=num_episodes,
        seed=seed,
        arms=arms,
    )
    print("\n" + "=" * 70)
    n = result.get("native")
    tr = result.get("triton")
    if n is not None:
        print(f"NATIVE: {n['total_success']}/{n['total_eps']} ({result.get('native_rate_pct', 0.0):.1f}%)")
    if tr is not None:
        print(f"TRITON: {tr['total_success']}/{tr['total_eps']} ({result.get('triton_rate_pct', 0.0):.1f}%)")
    if "delta_pp" in result:
        print(f"DELTA:  {result['delta_pp']:+.1f}pp")
    print("=" * 70)
