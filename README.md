# Reflex

**The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot.

**Verified parity across ALL four major open VLAs.** Reflex's monolithic ONNX export matches the reference PyTorch policy to **cos = +1.000000** end-to-end on SmolVLA, pi0, pi0.5 (canonical 10-step flow-matching unrolled) and GR00T N1.6 (canonical 4-step DDIM loop external to the ONNX). Per-model first-action max_abs: SmolVLA 5.96e-07, pi0 2.09e-07, pi0.5 2.38e-07, GR00T 8.34e-07 — all at machine precision, shared seeded inputs. Reproducers: `modal run scripts/modal_{smolvla,pi0,pi05,gr00t}_monolithic_export.py --parity`. The full claim ledger (verified / unverified / unmeasured) lives in [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

Cross-framework ONNX export, edge-first serving, composable runtime wedges (safety, adaptive denoising, cloud-edge split, pre-flight validation). One CLI, seven verbs.

**Want a deeper walkthrough?** See [docs/getting_started.md](docs/getting_started.md) for a 30-min guide covering safety configs, fleet-mode batching, deadline enforcement, and common troubleshooting.

**Something not working?** Run `reflex doctor` first — diagnoses install + GPU issues in one screen.

## Quickstart — 3 commands from zero to actions

```bash
# 1. Install (v0.1 — install from GitHub until we publish to PyPI)
# `[monolithic]` is required for the cos=+1.000000 verified export path (it pins transformers==5.3.0).
pip install 'reflex-vla[serve,gpu,monolithic] @ git+https://github.com/rylinjames/reflex-vla'
# Or CPU-only: pip install 'reflex-vla[serve,onnx,monolithic] @ git+https://github.com/rylinjames/reflex-vla'
# Note: GPU install requires the FULL cuDNN 9 system library (incl. libcudnn_adv.so.9),
# not just the pip wheel. Easiest path is NVIDIA's container:
#   docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3
#   apt-get install -y clang   # lerobot → evdev needs clang to build
#   pip install 'reflex-vla[serve,gpu,monolithic] @ git+...'
# `reflex serve` errors loudly if cuDNN can't load — no silent CPU fallback.
# Easier alternative: use our published Docker image (see below).

# 2. Export any supported VLA to ONNX (auto-detects model type)
# SmolVLA fits on Orin Nano 8GB; pi0 (~12GB monolithic) needs Orin 16GB+ or desktop GPU.
# Note: first export is 5-15 min (SmolVLA ~10min, pi0 ~7min on A100). Subsequent
# `reflex serve` calls reuse the cached artifact and warm up in 10-70s.
reflex export lerobot/smolvla_base --target desktop --output ./smol

# 3. Serve it — POST /act to get 50-step action chunks
reflex serve ./smol --port 8000
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
  "latency_ms": 11.9,                // smolvla on A10G, full 10-step denoise
  "denoising_steps": 10,
  "inference_mode": "onnx_trt_fp16"  // automatic — no engine flags needed
}
```

That's it. `reflex` auto-detects whether you gave it SmolVLA / pi0 / pi0.5 / GR00T and dispatches to the right exporter. No framework-specific flags.

### Docker — zero-install serve

```bash
# Pull the published image (x86_64 CUDA runtime)
docker pull ghcr.io/rylinjames/reflex-vla:latest

# Mount your exports dir at /exports, expose port 8000
docker run --gpus all \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/rylinjames/reflex-vla:latest
```

The container's default command is `reflex serve /exports --host 0.0.0.0 --port 8000`. Override with any `reflex` subcommand: `docker run ... ghcr.io/rylinjames/reflex-vla:latest export <hf_id>` etc. Jetson arm64 images land in v0.3 (contact us if you need one sooner).

### ROS2 — `reflex ros2-serve`

Wraps the inference loop as a ROS2 node. Subscribes to `sensor_msgs/Image`, `sensor_msgs/JointState`, and `std_msgs/String`; publishes action chunks as `std_msgs/Float32MultiArray` at a configurable rate.

```bash
# rclpy is NOT pip-installable. Install ROS2 via apt or robostack first:
source /opt/ros/humble/setup.bash   # or iron / jazzy

reflex ros2-serve ./my_export \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --task-topic  /reflex/task \
  --action-topic /reflex/actions \
  --rate-hz 20
```

Inference respects `--safety-config` (same limits file as HTTP serve).

When `onnxruntime-gpu` ships with the TensorRT execution provider (it does in v1.20+), `reflex serve` uses TRT FP16 automatically and caches the engine in `<export_dir>/.trt_cache` so subsequent server starts skip the engine-build cost. The first `reflex serve` takes ~30-90s to warm up; restart is ~1-2s.

## Validation — round-trip ONNX vs PyTorch parity

After exporting, run `reflex validate` to confirm the ONNX graph matches the PyTorch reference within numerical tolerance:

```bash
# After exporting, validate that ONNX parity holds vs the reference:
reflex validate ./p0 --model lerobot/pi0_base --threshold 1e-4
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
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + validated (max_diff=8.34e-07) |
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

**Memory fit (monolithic ONNX on disk, FP32):** SmolVLA 1.6GB, pi0 12.5GB, pi0.5 13.0GB, GR00T 4.4GB. SmolVLA fits comfortably on Orin Nano 8GB; **pi0 realistically needs Orin 16GB+ or a desktop NVIDIA GPU** — the 12.5GB monolithic ONNX cannot load on the 8GB Orin Nano even in FP16 (~6GB weights plus activations + OS). FP16 engine rebuild + Orin Nano fit work is tracked for v0.3.

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

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export verified at cos=+1.0000000 on all four major open VLAs — SmolVLA + pi0 + pi0.5 (flow-matching, num_steps=10) + GR00T N1.6 (DDPM DiT, num_steps=4) — plus a composable runtime (serve + safety + turbo + split), edge-first design targeting Jetson + desktop NVIDIA GPUs.

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

v0.1 — active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.2.
