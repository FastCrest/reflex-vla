"""Modal: build + bench FlashRT public Python version on L40S.

Validates LiangSu8899/FlashRT's 18× JAX baseline claim on Modal-available
Ada-class hardware (L40S = SM89, same arch as RTX 4090). Initial attempt
on A100 (SM80) failed: FlashRT's Pi0.5 RTX frontend uses
torch.float8_e4m3fn which requires SM89+ tensor cores. L40S is the
closest Modal-available match to FlashRT's published 4090 numbers.

Establishes reflex-vla's reference before Liang's pure-C++ pi05 binary
arrives ~2026-05-06 for the `--runtime flashrt` integration spike.

Per 2026-04-29 Discord exchange + competitor profile
`reflex_context/02_research/competitors/flashrt.md`.

Usage:
    modal profile activate suranjana-jain
    HF_TOKEN=<token> modal run scripts/modal_flashrt_bench.py

Cost: ~$5-8 ($3 image build first time / cached afterward, $2-3 GPU run).
Wall: ~25-40 min first run (cmake + make on CUTLASS + checkpoint download
+ FP8 calibration + benchmark), ~5-10 min subsequent runs (cached).
"""
from __future__ import annotations

import os
import modal

app = modal.App("flashrt-bench")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Locked to v5's exact build-bust value so v6+ hit the cached FA2 build
# (60 min compile worth ~$4 on L40S). Bump to a new string when you want
# to force a fresh FlashRT git clone + rebuild (e.g., new upstream HEAD).
_BUILD_BUST = "1777491975"


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


# Persistent volume to cache the cloned FlashRT + built kernels + HF cache
# across runs. First run takes the build hit; subsequent runs reuse.
hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
flashrt_cache = modal.Volume.from_name("flashrt-cache", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
FLASHRT_DIR = "/opt/flashrt"


# Per FlashRT INSTALL.md prerequisites:
# - CUDA 12.4+ devel image (cmake needs nvcc, not just CUDA runtime)
# - GCC 11+ (C++17), CMake 3.24+
# - SM80+ GPU (A100 = SM80, fits)
# - Python 3.10/3.11/3.12 with the SAME interpreter for cmake AND import
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "build-essential",
        "wget",
        "ninja-build",
        # NOTE: cmake from apt is 3.22.1 on Ubuntu 22.04 jammy, but FlashRT's
        # CMakeLists.txt requires 3.24+. We install cmake via pip below so
        # /usr/local/bin/cmake (3.x latest) wins on PATH.
    )
    .pip_install(
        # cmake>=3.24 required by FlashRT's CMakeLists.txt.
        # The pip-installed cmake binary lives in /usr/local/bin, which is
        # before /usr/bin in our PATH env, so it shadows the apt cmake (if any).
        "cmake>=3.24,<4",
        # FlashRT's torch frontend requires torch + safetensors + numpy.
        # Pinning torch to match CUDA 12 + SM80.
        "torch==2.5.1",
        "safetensors>=0.4.0",
        "numpy<2.0",
        "pybind11>=2.12",
        "huggingface_hub>=0.20",
        "transformers>=4.40,<5.4",
        "Pillow",
    )
    .run_commands(
        # Verify versions: cmake (need 3.24+ from pip), gcc (need 11+ from apt).
        "which cmake && cmake --version && gcc --version",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
        # CMake needs to find the right CUDA + compiler.
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
    .run_commands(
        # Bust cache for the actual build step so we re-pull FlashRT HEAD
        # if needed but keep the image base layers cached.
        f'echo "build_bust={_BUILD_BUST}"',
        # Clone FlashRT + CUTLASS 4.4.2 (pinned per their INSTALL.md §4),
        # then cmake + make + editable pip install. This is the one-time
        # ~10-15 min compile step.
        f"mkdir -p {FLASHRT_DIR} && cd {FLASHRT_DIR} && "
        "git clone --depth 1 https://github.com/LiangSu8899/FlashRT.git . && "
        "git clone --depth 1 --branch v4.4.2 "
        "  https://github.com/NVIDIA/cutlass.git third_party/cutlass && "
        "mkdir -p build && cd build && "
        # SM80 = A100 (no FP8), SM86 = A10/A10G (no FP8), SM89 = 4090/L40S (FP8 native),
        # SM110 = Thor, SM120 = 5090. FlashRT's pi0.5 frontend requires SM89+
        # for FP8 tensor cores (torch.float8_e4m3fn). Building only SM89 to
        # match L40S target.
        "cmake .. -GNinja "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  -DCMAKE_CUDA_ARCHITECTURES=89 "
        "  -DENABLE_FA2=ON "
        # FA2_ARCH_NATIVE_ONLY=ON skips sm_80 + compute_120 PTX AOT path.
        # Required because our CUDA 12.5 image's nvcc doesn't support
        # compute_120 (Blackwell needs CUDA 12.8+). Documented at
        # FlashRT/CMakeLists.txt:93.
        "  -DFA2_ARCH_NATIVE_ONLY=ON && "
        "ninja -j$(nproc) && "
        f"cd {FLASHRT_DIR} && pip install -e '.[torch]'",
        gpu="L40S",  # cmake reads $CUDA_ARCH_LIST + needs nvcc; build on GPU
    )
    # Late-stage deps that aren't in FlashRT's pyproject extras.
    # Kept in a separate pip_install AFTER the run_commands so adding
    # them doesn't invalidate the expensive FA2 build cache.
    # ml_dtypes is required by flash_vla.models.pi05.pipeline_rtx (line 45);
    # surfaced via v5's GPU_FAIL.
    .pip_install("ml_dtypes")
)


@app.function(
    image=image,
    gpu="L40S",  # SM89 — closest Modal-available match to FlashRT's published 4090 numbers
    volumes={HF_CACHE: hf_cache, FLASHRT_DIR + "_cache": flashrt_cache},
    secrets=[_hf_secret()],
    timeout=2400,  # 40 min hard cap
)
def bench(
    checkpoint_id: str = "lerobot/pi05_libero_finetuned_v044",
    benchmark_iters: int = 20,
    warmup_iters: int = 50,
    num_views: int = 2,
    prompt: str = "pick up the red block and place it in the tray",
):
    """Run FlashRT's bundled quickstart benchmark + compare to reflex-vla
    cuda-graphs A/B numbers.

    Returns dict with measured latency stats + verification info.
    """
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    print("=" * 60)
    print("FlashRT bench on L40S (SM89)")
    print(f"checkpoint: {checkpoint_id}")
    print(f"benchmark_iters: {benchmark_iters}, warmup: {warmup_iters}")
    print(f"num_views: {num_views}")
    print("=" * 60)

    # ---- 1. Verify FlashRT is built + importable ----
    # Public surface = flash_vla.load_model + flash_vla.VLAModel only;
    # compiled .so kernels are loaded internally by frontends.
    print("\n[1/4] verifying flash_vla import...", flush=True)
    try:
        import flash_vla
        print(f"  flash_vla.__version__ = {getattr(flash_vla, '__version__', '(no __version__)')}")
        assert hasattr(flash_vla, "load_model"), "flash_vla.load_model missing"
        assert hasattr(flash_vla, "VLAModel"), "flash_vla.VLAModel missing"
        print(f"  flash_vla.load_model + VLAModel exposed OK")
    except Exception as exc:
        return {"status": "fail", "stage": "import", "error": repr(exc)}

    # ---- 2. Download checkpoint via HF cache ----
    print(f"\n[2/4] downloading {checkpoint_id}...", flush=True)
    from huggingface_hub import snapshot_download
    try:
        ckpt_path = snapshot_download(
            repo_id=checkpoint_id,
            cache_dir=HF_CACHE,
        )
        print(f"  checkpoint at: {ckpt_path}")
        # FlashRT expects a directory containing the safetensors weights;
        # explore the structure to find the right subdirectory.
        for entry in Path(ckpt_path).iterdir():
            print(f"  - {entry.name}")
    except Exception as exc:
        return {"status": "fail", "stage": "checkpoint_download", "error": repr(exc)}

    # ---- 3. Run quickstart benchmark ----
    print(f"\n[3/4] running examples/quickstart.py --benchmark {benchmark_iters}...", flush=True)
    cmd = [
        sys.executable,
        f"{FLASHRT_DIR}/examples/quickstart.py",
        "--checkpoint", str(ckpt_path),
        "--framework", "torch",
        "--num_views", str(num_views),
        "--prompt", prompt,
        "--benchmark", str(benchmark_iters),
        "--warmup", str(warmup_iters),
        "--autotune", "3",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=1800,
    )
    wall_s = time.perf_counter() - t0
    print(f"\n--- quickstart stdout ---")
    print(proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout)
    if proc.returncode != 0:
        print(f"\n--- quickstart stderr ---")
        print(proc.stderr[-2000:])
        return {
            "status": "fail",
            "stage": "benchmark",
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "wall_s": wall_s,
        }
    print(f"\n  wall: {wall_s:.1f}s, exit 0")

    # ---- 4. Summarize ----
    print(f"\n[4/4] summary")
    print(f"  reflex-vla cuda-graphs A/B (A100, pi0.5 num_steps=10):")
    print(f"    OFF: 270.85 ms / chunk")
    print(f"    ON:  207.74 ms / chunk (1.30x speedup)")
    print(f"  See reflex_context/03_experiments/2026-04-29-cuda-graphs-ab-modal-a100.md")
    print(f"  (note: A100 ≠ this L40S run — separate L40S A/B needed for true apples-to-apples)")
    print(f"")
    print(f"  FlashRT published numbers (Ada-class):")
    print(f"    Pi0.5 RTX 4090: published reference (Liang's repo)")
    print(f"    Pi0.5 RTX 5090: 17.58 ms (57 Hz)")
    print(f"    Pi0.5 Thor:     44 ms / 39.78 ms NVFP4")
    print(f"    L40S: not published, this run is the first reference")

    return {
        "status": "ok",
        "checkpoint": checkpoint_id,
        "wall_s": wall_s,
        "stdout_tail": proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout,
    }


@app.local_entrypoint()
def main(
    checkpoint_id: str = "lerobot/pi05_libero_finetuned_v044",
    benchmark_iters: int = 20,
    warmup_iters: int = 50,
):
    """Local entrypoint."""
    print("FlashRT bench → A100-80GB → quickstart --benchmark")
    print(f"  checkpoint: {checkpoint_id}")
    print(f"  iters: {benchmark_iters} (warmup: {warmup_iters})")
    print()
    result = bench.remote(
        checkpoint_id=checkpoint_id,
        benchmark_iters=benchmark_iters,
        warmup_iters=warmup_iters,
    )
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  status: {result.get('status')}")
    if result.get("status") == "ok":
        print(f"  wall:   {result.get('wall_s', '?'):.1f}s")
        print()
        print("Stdout tail (last ~4000 chars):")
        print(result.get("stdout_tail", "(none)"))
    else:
        print(f"  stage:  {result.get('stage')}")
        print(f"  error:  {result.get('error', result.get('stderr_tail', '(none)'))}")
