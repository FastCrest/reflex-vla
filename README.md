# Reflex

Deploy any VLA (Vision-Language-Action) model to any edge hardware. One command.

```bash
pip install -e '.[dev,serve]'
reflex export lerobot/smolvla_base --target orin-nano
reflex serve ./reflex_export/ --port 8000
curl -X POST http://localhost:8000/act -d '{"instruction":"pick up the cup","state":[...]}'
```

## Supported VLA models

| Model | HF ID | Params | Status |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | 450M | ONNX + validated (max_diff=3.8e-06) |
| pi0 | `lerobot/pi0_base` | 3.5B | ONNX + validated (max_diff=3.7e-08) |
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX + AdaRMSNorm (max_diff=2.4e-06) |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + DiT/AdaLN (max_diff=2.2e-05) |
| OpenVLA | `openvla/openvla-7b` | 7.5B | use `optimum-cli export onnx` + our postprocess helper |

OpenVLA is a vanilla Llama-2-7B VLM whose "action head" is
`argmax(lm_logits[:, -7:])` + 256-bin lookup — no custom expert to
reconstruct, so `optimum-onnx` handles the model and
`reflex.postprocess.openvla.decode_actions` handles the
bin-to-continuous conversion.

Run `reflex models` to see current support.

## Hardware targets

| Target | Name | Memory | FP8 | Precision |
|---|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | no | fp16 |
| `orin` | Jetson Orin | 32 GB | no | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | no | fp16 |
| `thor` | Jetson Thor | 128 GB | yes | fp8 |
| `desktop` | RTX/A100 | 40 GB | yes | fp16 |

Run `reflex targets` to see hardware options.

## 7-wedge product

```
reflex export   # checkpoint → ONNX + TensorRT
reflex serve    # HTTP inference server (POST /act)
reflex guard    # safety constraints (joint limits, EU AI Act logging)
reflex turbo    # denoising loop optimization (adaptive step count)
reflex split    # cloud-edge orchestration (fallback on latency spikes)
reflex adapt    # cross-embodiment action mapping
reflex check    # pre-deployment validation
```

## Verification

Full E2E (export → serve → POST /act) tested on Modal A100, ONNX Runtime CPU provider, 10 denoising steps:

| Model | Expert params | Export | ONNX max_diff | Mean latency | Hz |
|---|---|---|---|---|---|
| SmolVLA | 100M | 50s | 3.3e-06 | 417ms | 2.4 |
| pi0 | 314M | 103s | 5.2e-08 | 968ms | 1.0 |
| pi0.5 | 427M | 106s | 2.2e-06 | 1036ms | 1.0 |
| GR00T N1.6 | 1091M | 111s | 3.5e-05 | export only | — |

GR00T's expert emits velocity in a different dim (1024) than its input
tokens (1536), so the simple SmolVLA-style denoising loop in `reflex
serve` can't run it directly. A GR00T-aware serve path needs the
embodiment-specific `action_decoder` (planned for v0.3). Expert ONNX
export already works and validates end-to-end.

GPU provider and TensorRT builds (on Jetson) expected to push
latencies 5-10x faster. Benchmark latencies shown are CPU-provider
ONNX Runtime on A100 — not the target deployment.
