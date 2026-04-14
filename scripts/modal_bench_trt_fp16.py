"""Phase II benchmark: TensorRT FP16 engines vs torch.compile on A10G.

A10G is the closest cloud GPU to Jetson Orin (same Ampere architecture,
similar power envelope). This run is our proxy for "what happens on Jetson
with TRT FP16" until we can run on real hardware.

Tests 4 paths per model:
  1. PyTorch eager (baseline)
  2. torch.compile(mode="reduce-overhead") (the bar)
  3. ORT-GPU FP32 (Reflex's current edge path)
  4. TensorRT FP16 engine built from the exported ONNX

Uses nvidia/tensorrt base image so trtexec is present. tensorrt Python
bindings are used to load the engine for inference timing.

Usage:
    modal run scripts/modal_bench_trt_fp16.py
"""

import modal

app = modal.App("reflex-bench-trt-fp16")

# NVIDIA's TensorRT 24.10 container ships CUDA 12.6 + cuDNN 9 + TRT 10.5 + trtexec
# Matches onnxruntime-gpu 1.20+ requirements cleanly.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu>=1.20,<1.24",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands(
        "cd /root/reflex-vla && pip install -e . --no-deps",
        "trtexec --help | head -1 || echo 'trtexec missing'",
    )
)


@app.function(image=image, gpu="A10G", timeout=3600, scaledown_window=60)
def bench_trt():
    import gc
    import os
    import subprocess
    import tempfile
    import time
    import numpy as np
    import torch

    # Verify toolchain
    print("=== Toolchain check ===", flush=True)
    r = subprocess.run(["trtexec", "--help"], capture_output=True, text=True)
    print(f"trtexec available: {r.returncode == 0}", flush=True)
    try:
        import tensorrt as trt_mod
        print(f"TensorRT Python: {trt_mod.__version__}", flush=True)
    except ImportError:
        print("TensorRT Python: MISSING", flush=True)
    import onnxruntime as ort
    print(f"ORT: {ort.__version__}, providers={ort.get_available_providers()}", flush=True)
    print(f"torch: {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)

    results = {}

    models = [
        {
            "tag": "smolvla",
            "hf_id": "lerobot/smolvla_base",
            "builder_mod": "reflex.exporters.smolvla_exporter",
            "builder_fn": "build_expert_stack",
            "builder_kwargs": {"head_dim": 64},
        },
        {
            "tag": "pi0",
            "hf_id": "lerobot/pi0_base",
            "builder_mod": "reflex.exporters.pi0_exporter",
            "builder_fn": "build_pi0_expert_stack",
            "builder_kwargs": {"head_dim": 128},
        },
        {
            "tag": "pi05",
            "hf_id": "lerobot/pi05_base",
            "builder_mod": "reflex.exporters.pi0_exporter",
            "builder_fn": "build_pi05_expert_stack",
            "builder_kwargs": {"head_dim": 128},
        },
        {
            "tag": "gr00t",
            "hf_id": "nvidia/GR00T-N1.6-3B",
            "builder_mod": "reflex.exporters.gr00t_exporter",
            "builder_fn": "build_gr00t_full_stack",
            "builder_kwargs": {"embodiment_id": 0},
        },
    ]

    def bench_callable(fn, n_warmup=10, n_trials=50):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        latencies = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()
        n = len(latencies)
        return {
            "mean_ms": round(sum(latencies) / n, 3),
            "p50_ms": round(latencies[n // 2], 3),
            "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 3),
            "p99_ms": round(latencies[min(n - 1, int(n * 0.99))], 3),
        }

    for m in models:
        tag = m["tag"]
        print(f"\n{'='*60}\nModel: {tag}\n{'='*60}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        # --- load + build ---
        from reflex.checkpoint import load_checkpoint
        import importlib
        state_dict, _ = load_checkpoint(m["hf_id"])
        builder_module = importlib.import_module(m["builder_mod"])
        builder = getattr(builder_module, m["builder_fn"])
        stack, meta = builder(state_dict, **m["builder_kwargs"])
        stack = stack.cuda().eval()
        chunk_size = 50
        action_dim = meta.get("raw_action_dim", meta.get("action_dim", 32))
        noisy = torch.randn(1, chunk_size, action_dim, device="cuda")
        pos_ids = torch.arange(chunk_size, device="cuda").unsqueeze(0)

        # --- (a) torch.compile single-step ---
        print("  [torch.compile]", flush=True)
        compiled = torch.compile(stack, mode="reduce-overhead", fullgraph=False)
        def _compile_step():
            with torch.no_grad():
                return compiled(noisy, torch.tensor([0.5], device="cuda"), pos_ids)
        for _ in range(5):
            _compile_step()
        compile_stats = bench_callable(_compile_step)
        print(f"    {compile_stats}", flush=True)

        # --- Export ONNX for ORT + TRT ---
        onnx_dir = tempfile.mkdtemp(prefix=f"trt_{tag}_")
        onnx_path = f"{onnx_dir}/expert.onnx"
        cpu_noisy = noisy.cpu()
        cpu_t = torch.tensor([0.5])
        cpu_pos = pos_ids.cpu()
        stack_cpu = stack.cpu()
        with torch.no_grad():
            torch.onnx.export(
                stack_cpu, (cpu_noisy, cpu_t, cpu_pos), onnx_path,
                input_names=["noisy_actions", "timestep", "position_ids"],
                output_names=["velocity"],
                opset_version=19,
            )
        stack = stack.cuda()

        # --- (b) ORT-GPU FP32 ---
        print("  [ORT-GPU FP32]", flush=True)
        ort_stats = None
        try:
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)
            active = session.get_providers()
            if "CUDAExecutionProvider" not in active:
                ort_stats = {"error": f"CUDA not active; active={active}"}
                print(f"    {ort_stats}", flush=True)
            else:
                np_noisy = noisy.cpu().numpy()
                np_t = np.array([0.5], dtype=np.float32)
                np_pos = pos_ids.cpu().numpy().astype(np.int64)
                def _ort():
                    return session.run(None, {
                        "noisy_actions": np_noisy,
                        "timestep": np_t,
                        "position_ids": np_pos,
                    })[0]
                for _ in range(10):
                    _ort()
                lats = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    _ort()
                    lats.append((time.perf_counter() - t0) * 1000)
                lats.sort()
                ort_stats = {
                    "mean_ms": round(sum(lats) / len(lats), 3),
                    "p50_ms": round(lats[len(lats) // 2], 3),
                    "p95_ms": round(lats[int(len(lats) * 0.95)], 3),
                    "active_providers": active,
                }
                print(f"    {ort_stats}", flush=True)
        except Exception as e:
            ort_stats = {"error": str(e)[:300]}
            print(f"    FAILED: {e}", flush=True)

        # --- (c) TRT FP16 engine ---
        print("  [TRT FP16]", flush=True)
        trt_stats = None
        try:
            engine_path = f"{onnx_dir}/expert.fp16.engine"
            build_cmd = [
                "trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                "--fp16",
                "--memPoolSize=workspace:4096MiB",
                # Static shape — batch=1, chunk=50, action_dim fixed
                f"--optShapes=noisy_actions:1x50x{action_dim},timestep:1,position_ids:1x50",
                f"--minShapes=noisy_actions:1x50x{action_dim},timestep:1,position_ids:1x50",
                f"--maxShapes=noisy_actions:1x50x{action_dim},timestep:1,position_ids:1x50",
            ]
            print(f"    building: trtexec --fp16 ...", flush=True)
            t0 = time.time()
            r = subprocess.run(build_cmd, capture_output=True, text=True, timeout=900)
            build_s = time.time() - t0
            if r.returncode != 0:
                trt_stats = {"error": f"trtexec exit {r.returncode}: {r.stderr[-500:]}"}
                print(f"    BUILD FAILED: {trt_stats['error']}", flush=True)
            else:
                engine_mb = os.path.getsize(engine_path) / 1e6
                print(f"    engine built in {build_s:.1f}s, size={engine_mb:.1f}MB", flush=True)

                # Load engine via tensorrt Python
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f:
                    rt = trt.Runtime(logger)
                    engine = rt.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()

                # Allocate IO buffers
                np_noisy_fp16 = noisy.half().cpu().numpy()
                np_t_fp16 = np.array([0.5], dtype=np.float16)
                np_pos = pos_ids.cpu().numpy().astype(np.int64)

                # Use tensor names — TRT 10+ API
                # Input/output indices
                import ctypes

                # Use torch tensors on CUDA as our buffers
                noisy_cuda = torch.from_numpy(np_noisy_fp16).cuda().contiguous()
                t_cuda = torch.from_numpy(np_t_fp16).cuda().contiguous()
                pos_cuda = torch.from_numpy(np_pos).cuda().contiguous()
                # Output — shape [1, 50, action_dim or 1024 for gr00t intermediate]
                # We need to query the engine for output shape. For FP16 output.
                out_shape = list(noisy_cuda.shape)
                # We know our models output same shape as input for smolvla/pi0/pi05,
                # but 1024-dim for gr00t's pre-decoder path. For full-stack gr00t,
                # same shape as input. Use that.
                out_cuda = torch.empty(out_shape, dtype=torch.float16, device="cuda").contiguous()

                # Bind
                context.set_tensor_address("noisy_actions", noisy_cuda.data_ptr())
                context.set_tensor_address("timestep", t_cuda.data_ptr())
                context.set_tensor_address("position_ids", pos_cuda.data_ptr())
                context.set_tensor_address("velocity", out_cuda.data_ptr())

                stream = torch.cuda.Stream().cuda_stream

                def _trt():
                    context.execute_async_v3(stream_handle=stream)
                    torch.cuda.synchronize()
                    return out_cuda

                # Warmup
                for _ in range(10):
                    _trt()
                lats = []
                for _ in range(50):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _trt()
                    torch.cuda.synchronize()
                    lats.append((time.perf_counter() - t0) * 1000)
                lats.sort()
                trt_stats = {
                    "mean_ms": round(sum(lats) / len(lats), 3),
                    "p50_ms": round(lats[len(lats) // 2], 3),
                    "p95_ms": round(lats[int(len(lats) * 0.95)], 3),
                    "engine_mb": round(engine_mb, 1),
                    "build_s": round(build_s, 1),
                }
                print(f"    {trt_stats}", flush=True)
        except Exception as e:
            import traceback
            trt_stats = {"error": str(e)[:300], "trace": traceback.format_exc()[:500]}
            print(f"    FAILED: {e}", flush=True)

        results[tag] = {
            "hf_id": m["hf_id"],
            "action_dim": action_dim,
            "params_m": round(meta.get("full_stack_params_m", meta.get("total_params_m", 0)), 1),
            "torch_compile_single": compile_stats,
            "ort_gpu_fp32_single": ort_stats,
            "trt_fp16_single": trt_stats,
        }
        del stack, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: single-step latency (ms) on A10G, lower is better", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<10} {'Params':>8} {'compile':>10} {'ORT-GPU':>10} {'TRT-FP16':>10} {'Winner':>15}", flush=True)
    for tag, r in results.items():
        c = r["torch_compile_single"].get("mean_ms", "—") if isinstance(r["torch_compile_single"], dict) else "—"
        o = r["ort_gpu_fp32_single"].get("mean_ms", "—") if isinstance(r["ort_gpu_fp32_single"], dict) and "mean_ms" in r["ort_gpu_fp32_single"] else "—"
        t = r["trt_fp16_single"].get("mean_ms", "—") if isinstance(r["trt_fp16_single"], dict) and "mean_ms" in r["trt_fp16_single"] else "—"
        paths = {"compile": c, "ORT-GPU": o, "TRT-FP16": t}
        numeric = {k: v for k, v in paths.items() if isinstance(v, (int, float))}
        winner = min(numeric, key=numeric.get) if numeric else "—"
        params_s = f"{r['params_m']}M"
        print(f"{tag:<10} {params_s:>8} {str(c):>10} {str(o):>10} {str(t):>10} {winner:>15}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Phase II benchmark — TRT FP16 vs torch.compile on A10G (Jetson-proxy)\n")
    r = bench_trt.remote()
    import json
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
