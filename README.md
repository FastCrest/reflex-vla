# Reflex

**The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot.

Cross-framework ONNX export, edge-first serving, composable runtime wedges (safety, adaptive denoising, cloud-edge split, pre-flight validation). One CLI, seven verbs.

**Want a deeper walkthrough?** See [docs/getting_started.md](docs/getting_started.md) for a 30-min guide covering safety configs, fleet-mode batching, deadline enforcement, and common troubleshooting.

## Quickstart — 3 commands from zero to actions

```bash
# 1. Install (v0.1 — install from GitHub until we publish to PyPI)
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
# Or CPU-only: pip install 'reflex-vla[serve] @ git+https://github.com/rylinjames/reflex-vla'
# Note: GPU install requires the FULL cuDNN 9 system library (incl. libcudnn_adv.so.9),
# not just the pip wheel. Easiest path is NVIDIA's container:
#   docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3
#   (then pip install reflex-vla[serve,gpu] inside)
# `reflex serve` errors loudly if cuDNN can't load — no silent CPU fallback.

# 2. Export any supported VLA to ONNX (auto-detects model type)
reflex export lerobot/pi0_base --target orin-nano --output ./p0

# 3. Serve it — POST /act to get 50-step action chunks
reflex serve ./p0 --port 8000
```

In another terminal:

```bash
curl -X POST http://localhost:8000/act -H 'content-type: application/json' \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

```json
{
  "actions": [[...], [...], ...],   // 50 × action_dim chunk
  "num_actions": 50,
  "latency_ms": 47.3,
  "denoising_steps": 10,
  "inference_mode": "onnx_gpu"
}
```

That's it. `reflex` auto-detects whether you gave it SmolVLA / pi0 / pi0.5 / GR00T and dispatches to the right exporter. No framework-specific flags.

## Composable wedges

Every wedge is a flag on `reflex serve`:

```bash
reflex serve ./p0 \
  --safety-config ./robot_limits.json \   # joint-limit clamping + EU AI Act audit log
  --adaptive-steps \                       # stop denoise loop early when velocity converges
  --deadline-ms 33 \                       # return last-known-good action if over budget
  --cloud-fallback http://cloud:8000      # edge-first with cloud backup
```

The response JSON surfaces telemetry from each enabled wedge so you can see what's actually happening (`safety_violations`, `deadline_exceeded`, `adaptive_enabled`, etc.).

## Supported VLA models

| Model | HF ID | Params | Export status |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | 450M | ONNX + validated (max_diff=3.3e-06) |
| pi0 | `lerobot/pi0_base` | 3.5B | ONNX + validated (max_diff=6.0e-08) |
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX + AdaRMSNorm (max_diff=2.5e-06) |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + DiT/AdaLN (max_diff=3.8e-06) |
| OpenVLA | `openvla/openvla-7b` | 7.5B | `optimum-cli export onnx` + `reflex.postprocess.openvla.decode_actions` |

`reflex models` lists current support at any time. OpenVLA is a vanilla Llama-2-7B VLM — there's no custom action expert to reconstruct, so we defer to the standard HuggingFace export path and ship only the bin-to-continuous postprocess helper.

## Hardware targets

| Target | Hardware | Memory | Precision |
|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | fp16 |
| `orin` | Jetson Orin | 32 GB | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | fp16 |
| `thor` | Jetson Thor | 128 GB | fp8 |
| `desktop` | RTX / A100 | 40 GB | fp16 |

All 4 flow-matching VLAs fit on a `$500` Orin Nano 8GB in FP16 with 2× overhead (verified empirically) — no need to jump to Thor for most deployments.

`reflex targets` lists current profiles.

## The 7 wedges (+ 1 planned)

```
reflex export   # checkpoint → ONNX + TensorRT
reflex serve    # HTTP inference server, composable wedges
reflex guard    # URDF-derived safety limits + EU AI Act logging
reflex turbo    # adaptive denoising (stops early on convergence)
reflex split    # cloud-edge orchestration with fallback modes
reflex adapt    # cross-embodiment action-space mapping
reflex check    # 5 pre-deployment checks (loadable, size, structure, dtype, nan_inf)
reflex distill  # [planned] flow-matching step distillation (10 → 2)
```

Each wedge works standalone for scripting, and every wedge that belongs in the inference path is composable through `reflex serve` flags.

## What Reflex is and isn't

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export (4 VLA families covered), composable runtime (serve + safety + turbo + split), Jetson-first.

**Isn't:** a training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Reflex's moat is the deployment toolchain: cross-framework ONNX, TensorRT FP16 engines that beat `torch.compile` on cloud GPU by 2.6-3.3× *and* run on Jetson, deterministic deploy graph, and the wedge composition for production robot deployments.

## Performance — Reflex TRT FP16 vs PyTorch alternatives

Per-denoising-step latency on Modal A10G (the closest cloud GPU to Jetson Orin's Ampere architecture). Lower is better:

| Model | Params | PyTorch `torch.compile` | ONNX Runtime GPU FP32 | **Reflex TRT FP16** | Speedup |
|---|---|---|---|---|---|
| SmolVLA | 99.8M | 3.06ms | 3.26ms | **0.95ms** | **3.2×** |
| pi0 | 314.6M | 6.23ms | 5.53ms | **1.94ms** | **3.2×** |
| pi0.5 | 426.9M | 7.34ms | 7.37ms | **2.24ms** | **3.3×** |
| GR00T N1.6 | 1091.7M | 14.61ms | 14.45ms | **5.59ms** | **2.6×** |

Reflex's TensorRT FP16 path beats `torch.compile` by 2.6-3.3× on cloud GPU, and the same ONNX → TRT pipeline is what runs on Jetson — there is no "cloud version" vs "edge version" of the model.

Per-chunk (10 denoising steps):

| Model | TRT FP16 wall-clock | Effective Hz |
|---|---|---|
| SmolVLA | 9.5 ms | 105 Hz |
| pi0 | 19.4 ms | 52 Hz |
| pi0.5 | 22.4 ms | 45 Hz |
| GR00T N1.6 | 55.9 ms | 18 Hz |

All four sit comfortably above the 20-30 Hz needed for real-time robot control on A10G. Real Jetson Orin Nano numbers landing in a follow-up.

### Multi-robot batching (`reflex serve --max-batch N`)

pi0 on A10G, 32 concurrent /act requests:

| `--max-batch` | Throughput | Wall-clock to serve 32 reqs | Speedup |
|---|---|---|---|
| 1 (baseline, serial) | 17.1 qps | 1.87s | 1.00× |
| 4 | 40.0 qps | 0.80s | **2.34×** |
| 8 | 46.6 qps | 0.69s | **2.73×** |
| 16 | 49.3 qps | 0.65s | **2.88×** |

Continuous batching on the HTTP layer — each `/act` request enters an asyncio queue, the server flushes the queue every `--batch-timeout-ms` (default 5ms) into one batched ONNX inference. Throughput scales 2.3-2.9× at batch sizes 4-16, and per-request latency *drops* because requests no longer serialize through the model.

Methodology + raw data: [vla_to_hardware_roadmap/phase_1_vla_software/deployment_export](https://github.com/rylinjames/vla_to_hardware_roadmap/blob/main/phase_1_vla_software/deployment_export/build_candidates.md). Reproduce via `modal run scripts/modal_bench_trt_fp16.py`.

## License

Apache 2.0

## Status

v0.1 — active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.2.
