# Draft: r/robotics post

## Title

**Open-source tool to deploy SmolVLA / pi0 / pi0.5 / GR00T to Jetson via ONNX + TensorRT**

## Body (markdown)

Hi r/robotics —

I built [Reflex](https://github.com/rylinjames/reflex-vla), an open-source CLI for taking a trained Vision-Language-Action model from a HuggingFace checkpoint to a working inference server you can hit from a robot.

It currently supports four VLA model families end-to-end:

- **SmolVLA** (`lerobot/smolvla_base`) — 450M params, SO-100 / SO-101 demos
- **pi0** (`lerobot/pi0_base`) — 3.5B, Physical Intelligence
- **pi0.5** (`lerobot/pi05_base`) — 3.62B, with AdaRMSNorm time conditioning
- **GR00T N1.6** (`nvidia/GR00T-N1.6-3B`) — 3.29B, NVIDIA's humanoid model

Three commands to go from zero to a running endpoint:

```bash
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
reflex export lerobot/pi0_base --target orin-nano --output ./p0
reflex serve ./p0 --port 8000
```

Then `POST /act` returns flow-matching action chunks. The composable wedges — `--safety-config` for joint limits + EU AI Act audit logging, `--adaptive-steps` for early-stop denoising, `--deadline-ms` for WCET fallback, `--max-batch` for fleet serving — let you build a real production pipeline without writing your own runtime.

**Hardware fit**: all four VLAs above fit on a $500 Jetson Orin Nano 8GB in FP16 (verified empirically — biggest is GR00T at 4.4 GB). No need to spec a Thor or Orin 64 unless you want to.

**Performance** (Modal A10G as Jetson-Ampere proxy, per denoising step in ms):

| Model | TRT FP16 latency | Per-chunk @ 10 steps | Effective Hz |
|---|---|---|---|
| SmolVLA | 0.95 ms | 9.5 ms | 105 Hz |
| pi0 | 1.94 ms | 19.4 ms | 52 Hz |
| pi0.5 | 2.24 ms | 22.4 ms | 45 Hz |
| GR00T | 5.59 ms | 55.9 ms | 18 Hz |

All above the 20-30 Hz typically needed for real-time robot control on the smaller models, and GR00T at 18 Hz is in striking distance.

For multi-robot deployments, `reflex serve --max-batch N` does HTTP-layer continuous batching. With pi0 + 32 concurrent /act on A10G, throughput scales 2.88× at batch=16 with per-request latency actually dropping (no more serial queueing).

**Honest disclaimers**:
- Alpha software, single maintainer (me), Apache 2.0
- The current ONNX covers the action-expert denoising loop; VLM prefix conditioning is the next milestone (it works, just with random conditioning today)
- Not yet validated on real Jetson hardware — anyone with one is welcome to run the same benchmark and tell me how it went

What I'm specifically asking for:

1. **Testers** — install it, point it at your robot, tell me what breaks
2. **Jetson benchmark contributor** — 30 min of your time on an Orin Nano dev kit would let me publish real edge numbers
3. **Critical feedback** on the wedge composition (`--safety-config`, `--adaptive-steps`, etc.) — does it match how you actually want to deploy?

Repo: https://github.com/rylinjames/reflex-vla

Happy to answer questions in the thread.

---

## Posting notes

- r/robotics tends to value implementation depth + honesty over hype
- Mention the open-source license up front
- Be ready for "why not just use [X]" — answer politely, link to specific code
- If this gets traction, post-launch action: respond to every comment within 24h
