# Draft post for huggingface/lerobot#3146

> Issue title (paraphrased): "Add ONNX export and TensorRT support for VLA models so they can be deployed to edge hardware (Jetson, etc.)"

---

## Draft v1 — to be posted as a comment on issue #3146

Hi all — picking this up from the outside.

I've spent the last few weeks building a standalone export + serve toolchain for flow-matching VLAs. It's open source under Apache 2.0 and currently covers four families:

| Model | HF ID | Status |
|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | ✅ ONNX export, validated max_diff 3.3e-06 |
| pi0 | `lerobot/pi0_base` | ✅ ONNX, max_diff 6.0e-08 |
| pi0.5 | `lerobot/pi05_base` | ✅ ONNX + AdaRMSNorm decomposition, max_diff 2.5e-06 |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | ✅ ONNX + DiT/AdaLN + per-embodiment encoder/decoder, max_diff 3.8e-06 |

Repo: https://github.com/rylinjames/reflex-vla

### Performance

Modal A10G benchmark (closest cloud GPU to Jetson Orin Ampere). Per single denoising step in ms:

| Model | Params | `torch.compile` | ORT-GPU FP32 | **TensorRT FP16** | TRT speedup |
|---|---|---|---|---|---|
| SmolVLA | 99.8M | 3.06 | 3.26 | **0.95** | 3.2× |
| pi0 | 314.6M | 6.23 | 5.53 | **1.94** | 3.2× |
| pi0.5 | 426.9M | 7.34 | 7.37 | **2.24** | 3.3× |
| GR00T N1.6 | 1091.7M | 14.61 | 14.45 | **5.59** | 2.6× |

Per chunk (10 denoising steps), the TRT FP16 numbers translate to **18-105 Hz on A10G** depending on model size — comfortably above the 20-30 Hz typically wanted for real-time robot control. Same ONNX → TRT pipeline runs on Jetson, no separate cloud vs edge model variant.

For fleet operators: `reflex serve --max-batch N` does continuous HTTP batching. With pi0 on A10G and 32 concurrent /act requests, throughput scales 2.34× at batch=4, 2.73× at batch=8, 2.88× at batch=16. Per-request latency drops too because requests no longer serialize through the model.

Methodology + raw data + reproducer: https://github.com/rylinjames/vla_to_hardware_roadmap/blob/main/phase_1_vla_software/deployment_export/build_candidates.md

### How to try it

```bash
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
reflex export lerobot/pi0_base --target orin-nano --output ./p0
reflex serve ./p0 --port 8000
```

```bash
curl -X POST http://localhost:8000/act -H 'content-type: application/json' \
  -d '{"instruction":"reach", "state":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}'
```

### What's in scope vs not

**In scope for v0.1:** the action-expert (denoising loop) export + a FastAPI serve layer with composable wedges (`--safety-config`, `--adaptive-steps`, `--cloud-fallback`, `--deadline-ms`, `--max-batch`).

**Not in scope yet:** VLM prefix encoding inside the exported graph — the current ONNX runs the action expert with random conditioning. Adding the VLM prefix is the obvious next milestone but it's a much bigger lift (needs SigLIP/PaliGemma/Qwen3 encoder decomposition); doing that right is what's holding up the 0.2 release.

### Asks

1. **Testers welcome.** If you're deploying a VLA to a Jetson or any edge box, install and tell me what breaks. Open an issue on the reflex-vla repo, I respond fast.
2. **Looking for someone with a Jetson Orin Nano dev kit** to run a 30-min benchmark on real hardware and confirm the cloud-A10G → Jetson latency ratio holds.
3. **Architectural feedback** on how this should sit relative to LeRobot's own deployment story — happy to upstream any subset that fits, or stay separate, whichever the LeRobot maintainers prefer.

Honest disclaimer: this is alpha, single maintainer, no funding. If it works for you, great; if it doesn't, please tell me how it broke so I can fix it.

— @rylinjames
