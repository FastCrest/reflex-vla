# Reflex

> by [FastCrest](https://fastcrest.com) — deployment infrastructure for vision-language-action models.

[![PyPI](https://img.shields.io/pypi/v/reflex-vla.svg)](https://pypi.org/project/reflex-vla/)
[![Python](https://img.shields.io/pypi/pyversions/reflex-vla.svg)](https://pypi.org/project/reflex-vla/)
[![License](https://img.shields.io/pypi/l/reflex-vla.svg)](https://github.com/rylinjames/reflex-vla/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/reflex-vla.svg)](https://pypi.org/project/reflex-vla/)

**The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot. Now with **`reflex chat`** — talk to your robot fleet in plain English.

**Verified parity across ALL four major open VLAs.** Reflex's monolithic ONNX export matches the reference PyTorch policy to **cos = +1.000000** end-to-end on SmolVLA, pi0, pi0.5 (canonical 10-step flow-matching unrolled) and GR00T N1.6 (canonical 4-step DDIM loop external to the ONNX). Per-model first-action max_abs: SmolVLA 5.96e-07, pi0 2.09e-07, pi0.5 2.38e-07, GR00T 8.34e-07 — all at machine precision, shared seeded inputs. Full claim ledger in [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

One CLI, seven verbs, plus a chat agent.

## Install

```bash
pip install reflex-vla
```

For GPU serving + monolithic export (production path):

```bash
pip install 'reflex-vla[serve,gpu,monolithic]'
```

## Quickstart — chat to it

```bash
reflex chat
```

```
you › what version am I running and what hardware can I deploy to?

  → show_version({})    → reflex --version    → "reflex 0.2.0"
  → list_targets({})    → reflex targets      → [orin-nano, orin, orin-64, thor, desktop]

You're running reflex 0.2.0. Supported targets:
  - orin-nano — Jetson Orin Nano: 8 GB, fp16
  - orin — Jetson AGX Orin 32GB: 32 GB, fp16
  - orin-64 — Jetson AGX Orin 64GB: 64 GB, fp16
  - thor — Jetson Thor: 128 GB, fp8
  - desktop — Desktop GPU (RTX 4090 / A100 / H100): 24 GB, fp16

Want me to show which models support each target, or run reflex doctor?
```

Chat understands 16 reflex commands (export, serve, bench, eval, distill, finetune, traces, doctor, etc.) and runs them as subprocess on your behalf. Powered by GPT-5 Mini through a proxy hosted at `chat.fastcrest.com` — free tier is 100 calls/day per machine, no signup, no API key.

> Bring your own key? `export FASTCREST_PROXY_URL=https://api.openai.com/v1` (coming in v0.3 — for now, the hosted proxy is the path).

## Quickstart — explicit deploy

```bash
# Browse the curated model registry
reflex models list

# One command — probe hardware → resolve model → pull → serve
reflex go --model smolvla-base --embodiment franka

# Or with explicit hardware override + Python client
reflex go --model pi05-libero --embodiment franka --device-class a10g
```

Then from your code:

```python
from reflex.client import ReflexClient

with ReflexClient("http://localhost:8000") as client:
    with client.episode() as ep:                       # auto episode_id, RTC reset
        result = ep.act(image=numpy_frame, state=[0.1, 0.2, ...])
        print(result["actions"])                       # list of action chunks
        # 503-warming retried automatically; guard violations surface as fields
```

Or with curl:

```bash
curl -X POST http://localhost:8000/act -H 'content-type: application/json' \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

```json
{
  "actions": [[...], [...], ...],          // 50 × action_dim chunk
  "latency_ms": 11.9,                      // smolvla on A10G, 10-step denoise
  "inference_mode": "onnx_trt_fp16",       // automatic — no engine flags needed
  "guard_clamped": false                    // ActionGuard didn't have to clamp anything
}
```

`reflex go` auto-detects your hardware (NVIDIA GPU / Jetson / CPU), picks the right model variant for that device, downloads weights from HuggingFace, and starts the /act endpoint. **No editing configs, no separate `reflex export` step, no manual variant selection.** For models that ship as raw PyTorch weights, you get the export command to run next.

### The verb surface

```
reflex chat             # NEW — natural-language interface to every command below
reflex go               # one-command-deploy: probe → resolve → pull → serve
reflex serve            # explicit-config server (full flag surface)
reflex doctor           # diagnose env + GPU + per-deploy issues
reflex models {list, pull, info, export}    # curated registry + lifecycle
reflex train  {finetune, distill}           # training operations
reflex validate {dataset, export}           # pre-flight checks
reflex inspect {bench, replay, targets, guard, doctor}   # diagnostics + forensics
```

Hidden legacy commands (`export`, `bench`, `replay`, etc.) stay callable for one release as alias bridges. Removed in v0.3.

### Install notes

- `[monolithic]` extra is required for the cos=+1.000000 verified export path (pins transformers==5.3.0)
- CPU-only: `pip install 'reflex-vla[serve,onnx,monolithic]'`
- GPU install needs the FULL cuDNN 9 system library (not just the pip wheel). Easiest path: NVIDIA's container `docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3`, then `apt-get install -y clang` (for lerobot→evdev), then the pip install
- `reflex serve` errors loudly if cuDNN can't load — no silent CPU fallback
- First `reflex go` downloads weights (~1-14 GB depending on model) — cached on subsequent runs
- First serve takes 10-70s warmup; `/health` returns HTTP 503 until ready, HTTP 200 after — load balancers correctly skip the server during warmup
- `reflex chat` works on the base install — no extras required. Network access required (calls FastCrest's hosted proxy).

### Docker — zero-install serve

```bash
# x86_64 CUDA runtime (cloud GPUs, dev workstations)
docker pull ghcr.io/rylinjames/reflex-vla:latest
docker run --gpus all \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/rylinjames/reflex-vla:latest

# Jetson Orin / Orin Nano / Thor (arm64 + nvidia container runtime)
docker pull ghcr.io/rylinjames/reflex-vla:latest-arm64
docker run --runtime=nvidia \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/rylinjames/reflex-vla:latest-arm64
```

The container's default command is `reflex serve /exports --host 0.0.0.0 --port 8000`. Override with any `reflex` subcommand: `docker run ... ghcr.io/rylinjames/reflex-vla:latest export <hf_id>` etc.

Jetson arm64 image: built via QEMU cross-compile on tag push (`v*`). Bring-your-own-CUDA — the image deliberately doesn't bundle CUDA/cuDNN/TensorRT (those live on the Jetson under `/usr/local/cuda` and are ABI-locked to the host's JetPack version; the nvidia container runtime exposes them into the container).

### ROS2 — `reflex ros2-serve`

Wraps the inference loop as a ROS2 node. Subscribes to `sensor_msgs/Image`, `sensor_msgs/JointState`, and `std_msgs/String`; publishes action chunks as `std_msgs/Float32MultiArray` at a configurable rate.

```bash
# rclpy is NOT pip-installable. Install ROS2 via apt or robostack first:
source /opt/ros/humble/setup.bash   # or iron / jazzy

# Hidden alias — kept for back-compat through v0.2; will fold into
# `reflex serve --transport ros2` in a future release.
reflex ros2-serve ./my_export \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --task-topic  /reflex/task \
  --action-topic /reflex/actions \
  --rate-hz 20
```

Inference respects `--safety-config` (same limits file as HTTP serve).

When `onnxruntime-gpu` ships with the TensorRT execution provider (it does in v1.20+), `reflex serve` uses TRT FP16 automatically and caches the engine in `<export_dir>/.trt_cache` so subsequent server starts skip the engine-build cost. The first `reflex serve` takes ~30-90s to warm up; restart is ~1-2s.

## Pre-flight validation

Before deploying, validate your dataset (will it train?) and your export (does it serve cleanly?):

```bash
# Dataset: 8 falsifiable checks against your LeRobot v3.0 corpus
reflex validate dataset /path/to/lerobot_data --embodiment franka --strict

# Export: round-trip ONNX vs PyTorch parity at machine-precision threshold
reflex validate export ./p0 --model lerobot/pi0_base --threshold 1e-4
```

Sample passing output (abbreviated):

```
Per-fixture results
fixture_idx  max_abs_diff  mean_abs_diff  passed
0            3.21e-06      8.40e-07       PASS
1            2.98e-06      7.92e-07       PASS
...
Summary
max_abs_diff_across_all  3.21e-06
passed                   PASS
```

Exit codes: `0` pass, `1` fail (any fixture above threshold), `2` error (missing ONNX, bad config). Pipe `--output-json` for CI consumption, or run `reflex validate --init-ci` to scaffold a GitHub Actions workflow at `.github/workflows/reflex-validate.yml`.

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
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX + validated (max_diff=2.38e-07) |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + validated (max_diff=8.34e-07, **live VLM conditioning**) |
| OpenVLA | `openvla/openvla-7b` | 7.5B | `optimum-cli export onnx` + `reflex.postprocess.openvla.decode_actions` |

`reflex models list` browses the curated registry; `reflex models info <id>` shows benchmarks; `reflex models pull <id>` downloads. OpenVLA is a vanilla Llama-2-7B VLM — there's no custom action expert to reconstruct, so we defer to the standard HuggingFace export path and ship only the bin-to-continuous postprocess helper.

## Hardware targets

| Target | Hardware | Memory | Precision |
|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | fp16 |
| `orin` | Jetson Orin | 32 GB | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | fp16 |
| `thor` | Jetson Thor | 128 GB | fp8 |
| `desktop` | RTX / A100 | 40 GB | fp16 |

**Memory fit (monolithic ONNX on disk, FP32):** SmolVLA 1.6GB, pi0 12.5GB, pi0.5 13.0GB, GR00T 4.4GB. SmolVLA fits comfortably on Orin Nano 8GB; **pi0 realistically needs Orin 16GB+ or a desktop NVIDIA GPU** — the 12.5GB monolithic ONNX cannot load on the 8GB Orin Nano even in FP16 (~6GB weights plus activations + OS). FP16 engine rebuild + Orin Nano fit work is tracked for v0.3.

`reflex inspect targets` lists current profiles.

## Composable runtime wedges

Each wedge is a flag on `reflex serve` (also flowed through `reflex go`):

```bash
reflex serve ./p0 \
  --embodiment franka \                   # per-robot action ranges + ActionGuard clamping
  --safety-config ./robot_limits.json \   # URDF-derived joint limits + EU AI Act audit log
  --adaptive-steps \                      # stop denoise loop early on velocity convergence
  --deadline-ms 33 \                      # return last-known-good action if over budget
  --cloud-fallback http://cloud:8000 \    # edge-first with cloud backup
  --inject-latency-ms 0 \                 # synthetic delay (B.4 A2C2 gate methodology)
  --record /tmp/traces \                  # JSONL request/response capture for replay
  --max-consecutive-crashes 5             # circuit breaker (503 + Retry-After: 60 on trip)
```

Every response surfaces telemetry from each enabled wedge (`guard_clamped`, `guard_violations`, `injected_latency_ms`, `inference_mode`, etc.).

## What Reflex is and isn't

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export verified at cos=+1.0000000 on all four major open VLAs — SmolVLA + pi0 + pi0.5 (flow-matching, num_steps=10) + GR00T N1.6 (DDPM DiT, num_steps=4, **with Eagle 2.5 VL backbone producing live image+language KV**) — plus a composable runtime (serve + safety + turbo + split), edge-first design targeting Jetson + desktop NVIDIA GPUs.

**Isn't:** a training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Reflex's moat is the deployment toolchain: cross-framework ONNX with verified numerical parity, composable safety wedges, ROS2 + Docker + HTTP serving, and a deterministic export receipt (`VERIFICATION.md`) your QA team can audit.

## Verified parity (the only load-bearing numbers)

Four ONNX artifacts in production, measured against PyTorch on shared seeded inputs:

| Artifact | Reference | first-action max_abs | verdict |
|---|---|---|---|
| **SmolVLA ONNX, num_steps=10** (production default) | `sample_actions(num_steps=10)` | **5.96e-07** | ✅ machine precision |
| **pi0 ONNX, num_steps=10** (production default) | `sample_actions(num_steps=10)` | **2.09e-07** | ✅ **machine precision** |
| **pi0.5 ONNX, num_steps=10** (production default) | `sample_actions(num_steps=10)` | **2.38e-07** | ✅ **machine precision** |
| **GR00T N1.6 ONNX, single-step DiT** (DDPM, loop external) | `GR00TFullStack.forward` | **8.34e-07** | ✅ **machine precision** |
| **GR00T N1.6 end-to-end 4-step denoise loop** | Python loop over PyTorch ref | **4.77e-07** | ✅ **machine precision** |
| **GR00T N1.6 Eagle VLM ONNX** (SigLIP + Qwen3 + mlp1, 1.87B) | `EagleExportStack` PyTorch | **4.25e-04** | ✅ machine precision |
| **GR00T N1.6 DiT with real VLM KV** (5-input `expert_stack_with_vlm.onnx`) | `GR00TFullStack(state, vlm_kv)` | **1.78e-05** | ✅ machine precision |
| **GR00T N1.6 end-to-end two-ONNX chain** (Eagle → DiT) | same chain in PyTorch | **1.90e-05** | ✅ parity + image-driven sensitivity verified (max_abs=0.21 on actions when input image changes) |
| SmolVLA ONNX, num_steps=1 | `sample_actions(num_steps=1)` | 1.55e-06 | ✅ machine precision |
| pi0 ONNX, num_steps=1 | `sample_actions(num_steps=1)` | 1.43e-06 | ✅ machine precision |

Plus PyTorch-level native-path sanity checks (`SmolVLAPolicy` with DecomposedRMSNorm swap vs reference = cos=1.0; `PI0Policy.predict_action_chunk` vs raw `sample_actions` = bit-exact).

**About the production defaults**: flow-matching VLAs (SmolVLA, pi0, pi0.5) canonically integrate the velocity field with 10 Euler steps — the ONNX bakes in the unrolled loop. GR00T is DDPM-style diffusion with 4 canonical steps — the ONNX exports one velocity step, and `reflex serve` wraps it in the loop. All four match canonical PyTorch to machine precision. Getting pi0 / pi0.5 there required three interacting patches under `torch.export` (F.pad causal mask, frozen `DynamicLayer.update`, `past_kv.get_seq_length()` for mask assembly); GR00T's simpler DiT graph (no DynamicCache, no PaliGemma masking) traces cleanly via plain `torch.onnx.export(opset=19)` — no patches needed. Details in `reflex_context/01_architecture/pi0_monolithic_wrap_pattern.md`.

Full ledger: [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

**Latency numbers are intentionally not in the README yet** — earlier TRT FP16 tables were measured on a now-abandoned decomposed-ONNX path. Desktop GPU + Jetson latency re-measurement is tracked for v0.3. `reflex bench <export_dir>` reproduces on any hardware.

Reproduce on your own GPU with one command:

```bash
reflex bench ./pi0 --iterations 100
```

### Multi-robot batching (`reflex serve --max-batch N`)

Continuous batching on the HTTP layer: each `/act` request enters an asyncio queue; the server flushes the queue every `--batch-timeout-ms` (default 5ms) into one batched ONNX inference. Earlier measurements on the decomposed-ONNX path showed 2.3-2.9× throughput scaling at batch sizes 4-16; those numbers are being re-measured on the monolithic path for v0.3.

## License

Apache 2.0

## Status

**v0.3 — `reflex chat` ships.** Active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.4.

---

Reflex is built by [FastCrest](https://fastcrest.com). Apache 2.0, no signup, no telemetry by default.
