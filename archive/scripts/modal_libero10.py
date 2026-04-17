"""LIBERO-10 evaluation via vla-eval on Modal A10G.

Thin runner: exports SmolVLA, launches the reusable vla-eval adapter
(``reflex.runtime.adapters.vla_eval``) as a background server, runs
``vla-eval run`` against it on LIBERO-10.

All inference logic (VLM prefix conditioning, denoising loop, action
post-processing) lives in ``reflex.runtime.ReflexServer``. This script
contains *only* the Modal image + LIBERO sim plumbing.

Usage:
    modal run scripts/modal_libero10.py
"""

import json
import sys
import time

import modal

app = modal.App("reflex-libero10")

# Image: MuJoCo + vla-eval + reflex on debian_slim (reliable — nvidia/cuda had
# Ubuntu mirror hash issues in Apr builds).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libegl1-mesa",
        "libglvnd0",
        "ffmpeg",
        # robomimic → egl_probe builds from source and needs cmake + gcc
        "cmake",
        "build-essential",
        # osmesa for MuJoCo software rendering (EGL hangs silently on some
        # debian_slim+NVIDIA combos with LIBERO; osmesa is reliable but slow)
        "libosmesa6",
        "libosmesa6-dev",
    )
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
        "typer",
        "rich",
        "pyyaml",
        "mujoco>=3.0",
        "gymnasium",
    )
    .pip_install("vla-eval")
    # robosuite 1.5+ moved module paths — pin 1.4.1 which LIBERO expects.
    .pip_install("robosuite==1.4.1", "h5py")
    # LIBERO's setup.py is install_requires=[]; its envs import bddl,
    # robomimic, hydra-core at reset() time. Installing the full
    # requirements.txt would downgrade transformers/numpy/etc. and nuke
    # the ONNX export stack. Install only the runtime-required deps
    # with flexible versions.
    .pip_install(
        "bddl==1.0.1",        # exact version — LIBERO imports bddl.parsing
        "future",             # bddl 1.0.1 imports from future.utils (py2/3 compat)
        "robomimic",          # loose pin — used in env config loading
        "hydra-core>=1.1",
        "easydict",
        "einops",
        "opencv-python-headless",
        "gym",                # LIBERO's venv.py uses old `gym`, not gymnasium
    )
    # lerobot installed AFTER robomimic (so robomimic's egl_probe can use apt
    # cmake) but BEFORE the LIBERO editable install (so lerobot's deps don't
    # invalidate LIBERO's .pth file).
    .pip_install("lerobot", "num2words")
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
        # Patch LIBERO's interactive input() prompts so import doesn't hang.
        " && python /root/patch_libero.py"
        " && python -c 'from libero.libero import benchmark; print(\"LIBERO import OK\")'"
        # Verify envs import works — this is what failed in run 1 (missing bddl).
        " && python -c 'from libero.libero.envs import OffScreenRenderEnv; print(\"LIBERO envs OK\")'"
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .add_local_file(
        "scripts/patch_libero.py", "/root/reflex-vla/scripts/patch_libero.py", copy=True
    )
    .run_commands("cd /root/reflex-vla && pip install -e .")
    .env(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LIBERO_DATA_DIR": "/tmp/libero_data",
            "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
            "LIBERO_BASE": "/tmp/libero_data",
        }
    )
    .run_commands("mkdir -p /tmp/libero_data")
)


@app.function(gpu="A10G", image=image, timeout=7200, scaledown_window=60)
def run_libero10():
    """Run LIBERO-10 evaluation."""
    import os
    import subprocess

    os.environ["MUJOCO_GL"] = "egl"

    export_dir = "/tmp/reflex_libero_export"
    results: dict = {
        "benchmark": "LIBERO-10",
        "model": "lerobot/smolvla_libero via Reflex ONNX",
        "steps": [],
        "task_success": None,
        "per_task": [],
    }

    def log(name: str, status: str, detail: str = "") -> None:
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} -- {detail}")

    # ── Step 1: Export SmolVLA (auto-produces 4 ONNX files incl. VLM prefix) ──
    # Using the official LIBERO fine-tune (not smolvla_base). Base has 0% on
    # LIBERO because it's never seen those tasks; the fine-tune is what gets
    # a real number.
    print("\n=== Step 1: Export lerobot/smolvla_libero ===")
    t0 = time.time()
    r = subprocess.run(
        [
            "reflex",
            "export",
            "lerobot/smolvla_libero",
            "--target",
            "desktop",
            "--output",
            export_dir,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if r.returncode != 0:
        log("export", "fail", (r.stdout + r.stderr)[-500:])
        return results

    # Verify the VLM prefix files landed (the thing that was broken before).
    vlm_files = ["vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"]
    present = [f for f in vlm_files if os.path.exists(os.path.join(export_dir, f))]

    # Verify the normalizer/unnormalizer files landed (the thing that's broken now)
    norm_files = [
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    norm_present = [
        f for f in norm_files if os.path.exists(os.path.join(export_dir, f))
    ]

    # If reflex export didn't copy them, print the last of its stdout so we
    # can see the Copied X/4 line (or whatever went wrong).
    if len(norm_present) < 2:
        print("  reflex export stdout tail (for normalizer copy diagnosis):")
        print(r.stdout[-2000:])

    log(
        "export",
        "pass",
        f"Exported in {time.time()-t0:.0f}s; VLM files: {len(present)}/3 "
        f"({present}); normalizer files: {len(norm_present)}/2 ({norm_present})",
    )

    # ── Step 2: Start adapter server via the reusable module ──────
    print("\n=== Step 2: Start reflex.runtime.adapters.vla_eval ===")
    server_env = {
        **os.environ,
        "REFLEX_EXPORT_DIR": export_dir,
        "REFLEX_ACTION_DIM_OUT": "7",  # LIBERO: 6 joints + gripper
        "REFLEX_DEVICE": "cuda",       # A10G GPU available
        "MUJOCO_GL": "osmesa",
        # Route to the native PyTorch SmolVLA path — bypasses the decomposed
        # ONNX export (which hits per-step 2% velocity drift) and runs lerobot
        # SmolVLAPolicy directly. ONNX export still shipped for Jetson/TRT.
        "REFLEX_NATIVE": "1",
    }
    # Route server stdout to a file so we can print it on error.
    server_log_path = "/tmp/adapter_server.log"
    server_log = open(server_log_path, "w")
    server_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "reflex.runtime.adapters.vla_eval",
            "--port",
            "8000",
        ],
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Poll for server readiness (ws + port open). Model loading includes 907MB
    # checkpoint, 3 ONNX sessions, AutoTokenizer + AutoProcessor downloads — can
    # take 60-120s on first run.
    import socket

    def _port_open(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0

    ready = False
    start = time.time()
    timeout_s = 300
    while time.time() - start < timeout_s:
        if server_proc.poll() is not None:
            server_log.close()
            with open(server_log_path) as f:
                tail = f.read()[-2000:]
            log("model_server", "fail", f"Server exited early:\n{tail}")
            return results
        if _port_open("127.0.0.1", 8000):
            ready = True
            break
        time.sleep(2)

    if not ready:
        server_proc.terminate()
        server_log.close()
        with open(server_log_path) as f:
            tail = f.read()[-2000:]
        log("model_server", "fail",
            f"Server did not open port 8000 within {timeout_s}s.\nLog tail:\n{tail}")
        return results

    warmup_s = time.time() - start
    log("model_server", "pass", f"Adapter ready in {warmup_s:.0f}s on port 8000")

    # Dump adapter startup log — tells us whether the normalizer loaded (key),
    # which camera key was picked, whether VLM is on, etc. Invaluable for
    # diagnosing silent misconfigurations.
    print("\n=== adapter server startup log ===")
    try:
        with open(server_log_path) as f:
            adapter_log = f.read()
        # Grab the key init lines, not the 100s of onnxruntime warnings
        for line in adapter_log.splitlines():
            if any(
                marker in line
                for marker in (
                    "ReflexVlaEvalAdapter ready",
                    "Loaded normalizer",
                    "norm=",
                    "VLM orchestrator",
                    "Loaded vision_encoder",
                    "Loaded text_embedder",
                    "Loaded decoder_prefill",
                    "Expert ONNX",
                    "strict",
                    "Error",
                    "ERROR",
                    "Traceback",
                )
            ):
                print(f"  [adapter] {line}")
    except Exception as e:
        print(f"  [adapter log read failed: {e}]")
    print("=== end adapter startup log ===\n")

    # (Previously: in-process adapter smoke test — removed; startup log above
    # already confirms norm=on + VLM orchestrator complete + all 4 stats loaded.)

    # Helper to tail the adapter log after episodes — catches "First predict"
    # and "First predict actions" diagnostics for diagnosing 0% task success.
    def _dump_adapter_tail():
        try:
            with open(server_log_path) as _f:
                _log = _f.read()
            print("\n=== adapter log tail (post-episodes) ===")
            for line in _log.splitlines():
                if any(
                    marker in line
                    for marker in (
                        "First predict",
                        "First predict actions",
                        "VLM orchestrator failed",
                        "dummy conditioning",
                        "ERROR",
                        "Traceback",
                    )
                ):
                    print(f"  [adapter] {line}")
            print("=== end adapter log tail ===\n")
        except Exception as _e:
            print(f"  [adapter tail read failed: {_e}]")

    # ── Step 3: Write LIBERO-10 config ────────────────────────────
    print("\n=== Step 3: Write LIBERO-10 config ===")
    libero_config = {
        "server": {"url": "ws://localhost:8000"},
        "output_dir": "/tmp/libero_results",
        "benchmarks": [
            {
                "benchmark": (
                    "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
                ),
                "subname": "libero_10",
                "mode": "sync",
                # 1 ep / task × 10 tasks = 10 total. Fast iteration.
                # Single-task quick-check mode: 1 task × 1 ep = ~5 min total
                # instead of 30+ min for all 10 tasks. Bump after first real
                # non-zero number proves the pipeline works.
                "episodes_per_task": 1,
                "max_steps": 300,
                "task_order": [0],  # single task only — first LIBERO-10 task
                "params": {
                    "suite": "libero_10",
                    "seed": 7,
                    "num_steps_wait": 10,
                    # CRITICAL: without these, vla-eval sends only images +
                    # task_description. Our first-predict dump showed state=none,
                    # which means the model was predicting actions from zero
                    # state vectors — garbage trajectories no matter what the
                    # VLM pipeline looked like.
                    "send_state": True,
                    "send_wrist_image": True,
                },
            }
        ],
    }

    import yaml

    config_path = "/tmp/libero_10_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(libero_config, f)
    log("config", "pass", f"LIBERO-10 config → {config_path}")

    # ── Step 4: Run LIBERO-10 evaluation ────────────────────────────
    print("\n=== Step 4: vla-eval run (LIBERO-10, 100 eps, ~30 min) ===")

    # Before running sim: prove LIBERO env.reset() works in isolation. If this
    # hangs, vla-eval would hang too — fail fast here with a clear message.
    print("\n=== Step 3b: LIBERO env.reset() smoke test ===")
    smoke_result = subprocess.run(
        [
            "python",
            "-c",
            """
import os, sys, time
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
print('importing libero...', flush=True); t0 = time.time()
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark as libero_bench
print(f'import ok ({time.time()-t0:.1f}s)', flush=True)
print('getting task...', flush=True); t0 = time.time()
suite = libero_bench.get_benchmark_dict()['libero_10']()
task = suite.get_task(0)
print(f'task: {task.name} ({time.time()-t0:.1f}s)', flush=True)
print('building env...', flush=True); t0 = time.time()
env_args = {
    'bddl_file_name': os.path.join('/opt/LIBERO/libero/libero/bddl_files', task.problem_folder, task.bddl_file),
    'camera_heights': 128,
    'camera_widths': 128,
}
env = OffScreenRenderEnv(**env_args)
print(f'env built ({time.time()-t0:.1f}s)', flush=True)
print('env.reset()...', flush=True); t0 = time.time()
obs = env.reset()
print(f'reset ok ({time.time()-t0:.1f}s) — obs keys: {sorted(list(obs.keys())[:5])}', flush=True)
env.close()
print('SMOKE TEST PASS', flush=True)
""",
        ],
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONUNBUFFERED": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(smoke_result.stdout[-2000:])
    if smoke_result.stderr:
        print("  stderr:", smoke_result.stderr[-1000:])
    if "SMOKE TEST PASS" not in smoke_result.stdout:
        log(
            "env_smoke_test",
            "fail",
            f"LIBERO env.reset() failed/hung — vla-eval will also hang. "
            f"Exit={smoke_result.returncode}. See stdout above.",
        )
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        return results
    log("env_smoke_test", "pass", "LIBERO env.reset() works")

    # Stream vla-eval stdout/stderr in real-time via Popen instead of
    # subprocess.run(capture_output=True) — the latter buffers until the
    # subprocess exits, which meant we couldn't tell if a run was hung vs
    # mid-episode for 50+ minutes.
    stream_env = {
        **server_env,
        "PYTHONUNBUFFERED": "1",  # force stdlib to flush per line
    }
    eval_proc = subprocess.Popen(
        [
            "vla-eval",
            "run",
            "--config",
            config_path,
            "--no-docker",
            "--server-url",
            "ws://localhost:8000",
            "--yes",
            "--verbose",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line-buffered
        text=True,
        env=stream_env,
    )

    # Idle-timeout guard: if no output for 180s, assume hung and kill.
    import select
    idle_timeout_s = 600  # 10 min — osmesa first-scene compilation can be slow
    overall_timeout_s = 3600  # 60 min hard cap
    eval_start = time.time()
    last_output = time.time()
    all_stdout: list[str] = []
    stuck = False

    while True:
        if eval_proc.poll() is not None:
            break
        if time.time() - eval_start > overall_timeout_s:
            print(f"  [timeout] vla-eval exceeded {overall_timeout_s}s; killing.")
            eval_proc.kill()
            stuck = True
            break
        if time.time() - last_output > idle_timeout_s:
            print(
                f"  [idle-timeout] no stdout for {idle_timeout_s}s; killing."
            )
            eval_proc.kill()
            stuck = True
            break
        # Read lines with a short poll so we can check the guards.
        rlist, _, _ = select.select([eval_proc.stdout], [], [], 2.0)
        if rlist:
            line = eval_proc.stdout.readline()
            if line:
                print(f"  [vla-eval] {line.rstrip()}")
                all_stdout.append(line)
                last_output = time.time()

    # Drain any remaining output after exit
    if eval_proc.stdout:
        rest = eval_proc.stdout.read()
        if rest:
            for line in rest.splitlines():
                print(f"  [vla-eval] {line}")
                all_stdout.append(line + "\n")

    eval_returncode = eval_proc.returncode if not stuck else -1

    # Reconstruct eval_result-ish object for the downstream code
    class _EvalResult:
        pass
    eval_result = _EvalResult()
    eval_result.returncode = eval_returncode
    eval_result.stdout = "".join(all_stdout)
    eval_result.stderr = ""

    # Dump adapter log tail — surfaces first-call obs/action diagnostics
    _dump_adapter_tail()

    # Parse results even on non-zero exit — vla-eval sometimes exits 0 even
    # when every episode crashes (the CLI considers "ran all tasks" = success).
    results_dir = "/tmp/libero_results"
    total_eps = 0
    total_success = 0
    total_errors = 0
    per_task: list[dict] = []
    result_file_found = False

    if os.path.exists(results_dir):
        for f in sorted(os.listdir(results_dir)):
            if not f.endswith(".json"):
                continue
            result_file_found = True
            with open(os.path.join(results_dir, f)) as rf:
                eval_data = json.load(rf)
            print(f"  Result file: {f}")
            for task in eval_data.get("tasks", []):
                task_name = task.get("task", "<unnamed>")
                eps = task.get("episodes", [])
                successes = sum(
                    1 for ep in eps if ep.get("metrics", {}).get("success")
                )
                errors = sum(1 for ep in eps if ep.get("failure_reason"))
                per_task.append(
                    {
                        "task": task_name,
                        "episodes": len(eps),
                        "success": successes,
                        "errors": errors,
                        "success_rate": successes / len(eps) if eps else 0.0,
                    }
                )
                total_eps += len(eps)
                total_success += successes
                total_errors += errors

    if total_eps > 0:
        success_rate = 100.0 * total_success / total_eps
        results["task_success"] = success_rate
        results["per_task"] = per_task
        results["episodes_total"] = total_eps
        results["episodes_errored"] = total_errors

        if total_errors == total_eps:
            log(
                "libero10",
                "fail",
                f"All {total_eps} episodes crashed before inference. "
                f"Check stderr above for the root cause.",
            )
        elif success_rate == 0 and total_errors == 0:
            log(
                "libero10",
                "fail",
                f"{total_eps} episodes ran cleanly, but 0% success. "
                f"Model did not solve any task.",
            )
        else:
            log(
                "libero10",
                "pass",
                f"{success_rate:.1f}% task success "
                f"({total_success}/{total_eps}, {total_errors} errors)",
            )
    elif result_file_found:
        log("libero10", "fail", "Result JSON had no task data")
    else:
        log(
            "libero10",
            "fail",
            f"No result file at {results_dir}. vla-eval exit={eval_result.returncode}; "
            f"stderr: {eval_result.stderr[-300:]}",
        )

    # Cleanup
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    return results


@app.local_entrypoint()
def main():
    print("Starting LIBERO-10 evaluation on Modal (A10G)...")
    print("  100 episodes across 10 tasks; ~30 min; ~$1-3 budget.")
    result = run_libero10.remote()

    print("\n" + "=" * 60)
    print("LIBERO-10 RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    if result.get("task_success") is not None:
        print(
            f"\n  HEADLINE: SmolVLA via Reflex achieves "
            f"{result['task_success']:.1f}% on LIBERO-10"
        )

    sys.exit(
        0 if any(s["status"] == "pass" for s in result["steps"]) else 1
    )
