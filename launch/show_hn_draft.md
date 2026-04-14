# Draft: Show HN: Reflex — deploy any VLA to a robot

## Title (~80 chars max)

**Show HN: Reflex – ONNX/TensorRT export for VLA models, runs on Jetson**

## Body (text post)

Hi HN —

I built Reflex because the path from "we have a trained Vision-Language-Action model" to "it runs on a real robot" is brutal. Every VLA team writes their own export pipeline and most of them break.

Reflex covers the four current flow-matching VLAs (SmolVLA from HuggingFace, pi0 + pi0.5 from Physical Intelligence via lerobot, and NVIDIA's GR00T N1.6) with one CLI:

```bash
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
reflex export lerobot/pi0_base --target orin-nano --output ./p0
reflex serve ./p0 --port 8000
```

Then `POST /act` returns 50-step action chunks at 18-105 Hz on A10G depending on model size. Same ONNX → TRT pipeline runs on Jetson, so there's no separate "cloud version" and "edge version" of your model.

**Performance** (Modal A10G, single denoising step):

| Model | torch.compile | ONNX Runtime GPU | TensorRT FP16 | Speedup |
|---|---|---|---|---|
| SmolVLA 100M | 3.06 ms | 3.26 ms | **0.95 ms** | 3.2× over compile |
| pi0 314M | 6.23 ms | 5.53 ms | **1.94 ms** | 3.2× |
| pi0.5 427M | 7.34 ms | 7.37 ms | **2.24 ms** | 3.3× |
| GR00T 1.09B | 14.61 ms | 14.45 ms | **5.59 ms** | 2.6× |

I went into this thinking the moat was edge-only because torch.compile was crushing my early benchmarks. Turned out my onnxruntime-gpu was silently falling back to CPU due to a CUDA 12-vs-13 library mismatch. Once that was fixed, TRT FP16 wins by 2.6-3.3× across the board.

There's also a continuous-batching server mode (`--max-batch N`) for fleet operators serving many robots through one GPU. With pi0 on A10G + 32 concurrent requests: 2.34× throughput at batch=4, 2.88× at batch=16.

Honest about what's not done: the current ONNX export covers the action-expert denoising loop with random VLM conditioning. Wiring the VLM prefix encoder into the same graph is the next big milestone (~weeks of work, depending on tokenizer + image processor decomposition).

Repo: https://github.com/rylinjames/reflex-vla
Roadmap with full benchmark methodology: https://github.com/rylinjames/vla_to_hardware_roadmap

Looking for testers, especially anyone with a Jetson Orin or a real robot to point this at. Apache 2.0, single maintainer, will respond to issues fast.

---

## Tone notes

- Avoid hype words ("revolutionary", "game-changing", etc.)
- Lead with the install + benchmark numbers, not the architecture
- Disclose what's not done before commenters ask
- Avoid "we" — single maintainer
- Don't oversell speedup ratios; let readers compute
