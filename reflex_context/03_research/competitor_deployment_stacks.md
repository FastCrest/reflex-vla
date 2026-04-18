# VLA Deployment Stacks — Comprehensive Competitor Enumeration (2025-2026)

Compiled 2026-04-16. This file enumerates **every** tool, project, and company positioning around VLA inference + deployment in the 2025-2026 window — not just the headline names. For each: what they ship, how they package, what benchmarks they cite, target audience, license, their approach to the LIBERO-vs-modern-lerobot dependency puzzle, and a competitive tier rating.

Companion to `competitor_landscape.md` (the synthesis). This file is the **enumeration**.

Tier conventions:
- **Tier 1** — Direct competitor (does the same thing, same segment)
- **Tier 2** — Adjacent (could become a competitor, or they are a partner/integration target)
- **Tier 3** — Different wedge (informs decisions but not head-to-head)
- **Tier 0** — Defunct or not relevant to the deployment wedge

---

## 1. LeRobot (HuggingFace)

### What they ship
- Open-source VLA training + policy-evaluation framework. Repo: `huggingface/lerobot`. v0.5.0 released March 2026 with 200+ merged PRs; LeRobot paper accepted to ICLR 2026.
- Built-in policies: ACT, Diffusion, SmolVLA (`lerobot/smolvla_base`, `lerobot/smolvla_libero`), pi0 (`lerobot/pi0_base`), pi0.5 (`lerobot/pi05_base`), X-VLA (native since 2026-03: `lerobot/xvla-base`, `lerobot/xvla-libero`, `lerobot/xvla-folding`, `lerobot/xvla-agibot-world`, `lerobot/xvla-google-robot`).
- RTC real-time chunking (`lerobot.policies.rtc`, v0.5) — adopted the Black/Galliker/Levine 2025 paper.
- Unified training script. Single `lerobot eval` entry point; `huggingface-cli upload` for checkpoint sharing.

### Packaging approach
- **Python / pip / `uv`.** `pip install lerobot` on PyPI. Dependencies resolve through uv or pip.
- LIBERO needs Linux. MuJoCo / OpenGL dependency issues recurring on macOS and in Docker containers; `MUJOCO_GL=glx` is the documented workaround.
- `lerobot#2501` is the canonical uv-broke-with-libero bug (hf-libero==0.1.3 distutils unsafe uninstall). Unresolved as of 2026-04.
- **No Docker image for deployment.** Users bring-your-own compose files (several community forks at `lerobot_mujoco_sim`, `lerobot-gpu` etc.).
- **No ONNX/TRT pipeline.** Tracked gaps: `lerobot#3146` (ONNX/TRT ask, 0 comments/5 reactions, untouched since March 2026), `lerobot#1899` (smolvla ONNX), `lerobot#2061` (torch.compile), `lerobot#819` (torch/torchvision vs Jetson wheel matrix — closed "not planned"), `lerobot#2356` (AsyncInference broken).

### Benchmarks they publish
- LIBERO 130+ tasks now officially supported (per `@LeRobotHF` tweet 2025-09-24). Dataset preprocessed to LeRobot format.
- SmolVLA reproduction numbers contested: `lerobot#2354` and `lerobot#3287` report users unable to reproduce paper numbers on LIBERO — config & hyperparam mismatch.
- Integration with NVIDIA Isaac Lab-Arena as evaluation backend (2026-03).

### License + audience
- Apache 2.0. Audience: researchers, robotics students, hobbyists, early-stage startups. HuggingFace-style "lower the barrier to entry."

### Their solution to the LIBERO/lerobot packaging puzzle
- **None shipped.** They depend on `hf-libero`; users hit the uv conflict and either drop to raw pip or fork. Isaac Lab-Arena integration is their long-term escape hatch (GPU-parallel eval without LIBERO's MuJoCo stack).

### Tier: **1** (direct competitor on the "deploy any VLA" wedge; they *could* ship `lerobot deploy` tomorrow and fold Reflex's headline pitch; probability medium since LeRobot staff are training-focused). Main allies: `@jashshah999` (wrote #3146, 19 PRs to LeRobot), `@imstevenpmwork`, `@fracapuano`, `@not-heavychevy`.

---

## 2. OpenPI (Physical Intelligence)

### What they ship
- `Physical-Intelligence/openpi` — **10.9k GitHub stars as of Feb 2026.**
- Three model architectures: π₀ (flow matching), π₀-FAST (FAST action tokenizer, autoregressive through Gemma), π₀.₅ (knowledge-insulation upgrade, released Sept 2025).
- JAX/Flax + PyTorch (PyTorch support added Sept 2025).
- Training + limited deployment infra: `serve_policy.py` launches a policy server with WebSocket protocol. `WebsocketClientPolicy(host, port)` on the client side.
- Dataset loaders for LeRobot format and RLDS.

### Packaging approach
- `uv` everywhere. Dockerfile exists but deployment is "roll your own."
- Examples: `examples/droid/`, `examples/libero/`, `examples/aloha_sim/`. Each example has its own `uv` environment.
- **No official Jetson image.** NVIDIA Jetson AI Lab published an *external* tutorial (jetson-ai-lab.com/tutorials/openpi_on_thor/) that walks through ONNX export + TensorRT NVFP4 quantization on Thor — community-contributed, not Physical Intelligence first-party.

### Benchmarks they publish
- LIBERO (pi0, pi0.5). DROID. ALOHA simple tasks ("1-20 hours of data enough to tune pi0 to a new task").
- No latency/Hz numbers published by PI itself. Third-party numbers circulated via Jetson AI Lab and `@allenzren/open-pi-zero` re-implementation.

### License + audience
- Apache 2.0 weights. Research-grade, but well-funded ($11B valuation; Levine, Hausman, Finn).
- Audience: VLA researchers; early-stage robotics startups; Trossen users (integration announced 2025).

### Their solution to the LIBERO/lerobot packaging puzzle
- **Dodges it entirely** by running LIBERO in its own `examples/libero/` uv env separate from the training code. Did not unify.

### Open issues that are Reflex's wedge signal
- `openpi#826` — pi0.5 Jetson Thor key mismatch (pi0.5 literally won't load on Thor out of the box).
- `openpi#386` — pi0 on Jetson Orin errors.
- `openpi#872, #740, #714, #580, #449, #591` — cross-embodiment adaptation, 6+ zero-response issues.
- `openpi#579` — WebSocket InvalidMessage. Server is flaky.

### Tier: **1** (direct competitor on the "deploy pi0/pi0.5 to edge" cut; primary existential risk is if PI ships `pi-serve` as a first-party Jetson runtime with pi-1.5 — high probability within 18 months).

---

## 3. OpenVLA (Stanford / UC Berkeley)

### What they ship
- `openvla/openvla` — 7B parameter VLA. SigLIP + DINOv2 + Llama-2 + projector + 256-bin discrete action head.
- Trained on 970k OXE trajectories.
- **A lightweight REST-API serve script** for remote inference (the official "deploy" surface).
- `Stanford-ILIAD/openvla-mini` for smaller variants.
- `openvla-oft` (Moo Jin Kim et al., 2025) — parallel decoding + action chunking + L1 regression — **26× speedup from ~5 Hz to ~130 Hz on LIBERO; 97.1% success (up from 76.5%). Fits on a single A100 or RTX 4090 16GB+.**

### Packaging approach
- `dusty-nv/openvla` is the go-to for Jetson — NVIDIA's dustynv container approach (l4t-text-generation base).
- No ONNX/TRT out of the box. `optimum-cli export onnx` handles the Llama-2 + SigLIP + DINOv2 + projector automatically; only the 256-bin lookup needs custom postprocess (what Reflex ships).

### Benchmarks they publish
- LIBERO 4-suite (spatial, object, goal, long): OpenVLA-OFT+ hits 97.1% average.
- OpenVLA base: ~3 FPS on Jetson AGX Orin even at INT4. This is the well-cited "edge too slow" datapoint.

### License + audience
- MIT (model + code). Audience: academia (>1000 citations per `State of Robotics 2026`), researchers, hobbyists.

### Their solution to the LIBERO/lerobot packaging puzzle
- Uses LIBERO directly (Linux-only, their own container). No uv/lerobot adjacency.

### Tier: **2** (model + bare-bones REST serve; not a deployment toolchain; potential partner / supported-model target).

---

## 4. NVIDIA Isaac GR00T + GR00T N1.6/N1.7/N2 + TensorRT Edge-LLM

### What they ship
- **`NVIDIA/Isaac-GR00T`** — open model repo. GR00T N1.6-3B uses internal Cosmos-Reason-2B backbone. N1.7 in early access with commercial licensing. **N2 targeted end of 2026, claims to more-than-double new-task success.**
- **NVIDIA/TensorRT-Edge-LLM** — open-source C++ SDK (Apache 2.0) purpose-built for LLM/VLM inference on Jetson Thor, DRIVE AGX Thor, and Jetson T4000. Features: EAGLE-3 speculative decoding, NVFP4 quantization, chunked prefill, FP8/INT4/NVFP4 precision. Partners at CES 2026 (Bosch, ThunderSoft, MediaTek). **This is the single biggest 2026-new entrant on Reflex's terrain.**
- **Isaac Sim 5.0 + Isaac Lab-Arena** — open-source policy evaluation framework. Pre-alpha released 2026. GPU-parallel eval reduces day-to-day eval to under an hour. Integrated into LeRobot's EnvHub.
- **Cosmos Reason 2** — open reasoning VLM, top of Physical Reasoning Leaderboard on HF, 1M+ downloads.
- **Lightwheel-RoboCasa-Tasks, Lightwheel-LIBERO-Tasks** — 250+ Isaac Lab-Arena tasks.
- **JetPack 7.1** (2026): TensorRT Edge-LLM bundled for Jetson T4000/T5000 (1200 FP4 TFLOPs, 64 GB memory).

### Packaging approach
- Jetson container registry (NGC). NVIDIA JetPack SDK. Docker-first.
- `jetson-ai-lab.com` tutorials for GR00T, Cosmos, pi0.5. Tutorial format: Dockerfile → ONNX export script → `trtexec` engine build → Python inference loop.
- **GR00T published TensorRT numbers (GR00T-N1.6-3B, 4 denoising steps):**
  - RTX 5090: 31ms end-to-end, 32.1 Hz.
  - H100: 36ms, 27.9 Hz.
  - RTX 4090: 43ms, 23.3 Hz.

### Benchmarks they publish
- Lightwheel-LIBERO, Lightwheel-RoboCasa. Isaac Lab Eval Tasks (`isaac-sim/IsaacLabEvalTasks` — specific to GR00T N1 benchmarking).
- GR00T N1 and N1.6 task-success numbers against LIBERO-tasks migrated to Isaac Lab-Arena.
- `gr00t#517` — GR00T TRT export work-in-progress, signals they know the gap exists.

### License + audience
- Apache 2.0 for GR00T weights and TensorRT-Edge-LLM. JetPack SDK proprietary but freely distributed.
- Audience: enterprise integrators, Tier-1 robotics OEMs (Seeed Studio is an Elite NVIDIA Partner; AGIBOT, 1X, Skild AI all use NVIDIA Blackwell / NVENventures-backed).

### Their solution to the LIBERO/lerobot packaging puzzle
- **Migrated LIBERO tasks into Isaac Lab-Arena.** No more MuJoCo/uv fight — the tasks run natively in GPU-parallel Isaac Sim. This is the systemic escape hatch.

### Tier: **1** (TensorRT Edge-LLM is direct competition for Reflex's TRT wedge. Isaac Lab-Arena directly competes with benchmark-delivery. The GR00T N2 launch could collapse the multi-VLA framing by making GR00T the default humanoid policy).

---

## 5. Dexbotic / Dexmal (DM0 + realtime-vla)

### What they ship
- `dexmal/dexbotic` — open-source VLA toolbox in PyTorch. One-stop research platform. Release DM0 (Feb 10, 2026) — "Embodied-Native VLA towards Physical AI." Arxiv 2602.14974.
- `Dexmal/realtime-vla` — runs VLA at **30 Hz frame rate and 480 Hz trajectory frequency**.
- Dexdata format to unify embodiments. Modular VLA framework (vision encoder + LLM + action expert).
- Pretrained DexboticVLM as a foundation model.
- Partnership with RLinf (2026-02-10) for VLA + RL.
- RoboChallenge — online service evaluating models on real machines.
- Internal data: 52 manipulation tasks, 8 single-arm real-world robots (UR5, Franka, Unitree Z1, Realman GEN72, Realman 75-6F, UMI, ARX5, WidowX).

### Packaging approach
- Python, PyTorch. Modular "vision encoder + LLM + action expert" drop-in.
- Docs show DM0 + RLinf for RL fine-tuning.
- No ONNX/TRT emphasis.

### Benchmarks they publish
- LIBERO, ManiSkill, RoboTwin (via RLinf-VLA integration — 20-85% improvements over baselines).

### License + audience
- Open-source (check exact on repo — appears Apache-2.0-compatible). Audience: Chinese robotics research community; adopted by AgiBot, X Square Robot, PsiBot, Moore Threads, D-Robotics.

### LIBERO/lerobot packaging
- Dexbotic doesn't ride the LeRobot stack. Separate toolbox. Their format (Dexdata) is a LeRobot alternative.

### Tier: **2** (adjacent — their toolbox positioning is more "research training platform" than deployment, though realtime-vla is a genuine serving component).

---

## 6. X-VLA (ICLR 2026) — `2toinf/X-VLA`

### What they ship
- 0.9B-param "Soft-Prompted Transformer" VLA. Embodiment-specific learnable embeddings gate a unified backbone.
- Native LeRobot support since March 2026; checkpoints live at `lerobot/xvla-*`.
- **Server-Client architecture** — separates model environment from simulation/robot dependencies. Avoids package conflicts. Supports distributed inference across GPUs/SLURM/edge devices. This is a deliberate LIBERO/lerobot packaging answer.
- Pretraining: 290K episodes from Droid, Robomind, Agibot; 7 platforms, 5 robot arm types.

### Packaging approach
- Model-side Python env + WebSocket predict_action() interface. Same pattern as starVLA, vla-eval, OpenPI.
- Not currently shipping ONNX.

### Benchmarks they publish
- 6 simulation environments + 3 real robots. LIBERO, Google Robot, AGIbot-World, folding tasks.

### License + audience
- Apache-2.0-like (per the `2toinf/X-VLA` ICLR 2026 release).
- Researchers; LeRobot users who want cross-embodiment.

### LIBERO/lerobot packaging
- The WebSocket server-client pattern is their answer. **Same as allenai/vla-evaluation-harness, starVLA.** This pattern is becoming the standard.

### Tier: **2** (model + serving pattern; not a deployment CLI; potential supported-model target — and *confirms* that the WebSocket+msgpack pattern is the emerging cross-framework standard).

---

## 7. LiteVLA-Edge (arXiv 2603.03380)

### What they ship
- Deployment-oriented VLA pipeline for fully on-device Jetson Orin-class inference.
- FP32 SFT → post-training 4-bit GGUF quantization → llama.cpp GPU-accelerated runtime.
- **150.5ms end-to-end, ~6.6 Hz on Jetson AGX Orin (40W).**

### Packaging approach
- GGUF format. llama.cpp runtime. Assumes ROS2 pipeline integration.

### Benchmarks they publish
- Own paper's Hz-per-task numbers. Comparisons to OpenVLA, EdgeVLA, EfficientVLA on "Reasoning-to-Hz" ratio.

### License + audience
- Open-source (check repo for exact terms). Research-first.

### LIBERO/lerobot packaging
- Not LeRobot-based; uses LIBERO tasks as a benchmark dataset via their own scripts.

### Tier: **1** (direct competitor on Jetson-edge wedge; 6.6 Hz vs Reflex's targeted 20-30 Hz is slower but they lead on ROS2 story).

---

## 8. EdgeVLA (arXiv 2507.14049)

### What they ship
- Hierarchical VLA separating semantic reasoning from high-frequency visuomotor tokens.
- 10-15 Hz on Jetson, at cost of shallower multi-step reasoning.

### Packaging approach
- Research paper + code; no dedicated deployment CLI.

### Benchmarks they publish
- Own paper's Hz-per-task numbers on Jetson.

### License + audience
- Research.

### Tier: **3** (informs architecture decisions; not a deployment-toolchain competitor).

---

## 9. EfficientVLA (arXiv 2506.10100)

### What they ship
- Training-free acceleration for diffusion-based VLAs: language-layer pruning + visual-token selection + iterative diffusion head caching.
- Reports speedup vs baseline with SIMPLER benchmark.

### Packaging approach
- Research inference framework; relies on TensorRT engines (acknowledged as lacking cross-platform flexibility compared to GGUF).

### License + audience
- Research.

### Tier: **3** (techniques to adopt, not a product).

---

## 10. NanoVLA (arXiv 2510.25122)

### What they ship
- Nano-sized VLA for edge: vision-language decoupling (late fusion), long-short action chunking, dynamic routing backbone.
- **52× faster inference vs previous SOTA VLA on edge; 98% less parameters; same-or-better accuracy.**

### Packaging approach
- Research model; no dedicated CLI.

### Tier: **3** (inference-efficiency technique; supported-model target).

---

## 11. GigaBrain-0 / GigaBrain-0-Small / GigaBrain-0.5M*

### What they ship
- `GigaBrain-0` — World-model-powered VLA foundation model (arXiv 2510.19430).
- `GigaBrain-0-Small` — Jetson AGX Orin edge variant. **840 GFLOPs, 0.13s inference latency, 80% task success.**
- `GigaBrain-0.5M*` — world-model-based RL training (arXiv 2602.12099).
- GigaBrain Challenge at CVPR 2026.

### Packaging approach
- Research release; sim+real benchmarks published.

### License + audience
- Research. Adoption likely through GigaBrain Challenge.

### Tier: **2** (adjacent — supported-model target, potential partnership).

---

## 12. RDT-1B / RDT2-FM (Tsinghua / `thu-ml/RoboticsDiffusionTransformer`)

### What they ship
- RDT-1B: 1B-param Diffusion Transformer, pretrained on 1M+ multi-robot episodes (46 datasets). Bimanual focus.
- RDT2-FM: zero-shot deployment on unseen embodiments for simple open-vocab tasks (released 2025-09 on HF).
- Unified action space across manipulators.

### Packaging approach
- PyTorch training code + HF model card. No bundled ONNX/TRT.

### License + audience
- Research (check exact). Audience: bimanual manipulation research.

### Tier: **2** (model target; Reflex could support RDT as a new VLA family).

---

## 13. Octo (Octo Model Team, UCB et al.)

### What they ship
- `octo-models/octo` — 27M (Small) / 93M (Base) transformer-based diffusion policy.
- Pretrained on 800k OXE episodes.
- Simple inference API: `Octo.predict_actions()`.

### Packaging approach
- JAX-based. Hobbyist-friendly but shows its age (2024 vintage).

### Benchmarks they publish
- Own paper; 9 real-robot setups across 4 institutions.

### License + audience
- MIT. Research.

### LIBERO/lerobot packaging
- Pre-dates the current LIBERO/lerobot pain; lives in its own JAX ecosystem.

### Tier: **0** (superseded by SmolVLA/pi0/OpenVLA; mostly of historical interest).

---

## 14. Open-Pi-Zero (`allenzren/open-pi-zero`)

### What they ship
- Third-party re-implementation of pi0 from scratch, MIT-style open. Instantiation of MoE + block-attention (JointModel, Mixture classes).
- Trains on fractal or bridge datasets.

### Packaging approach
- Research repo; PyTorch-based.

### License + audience
- Apache 2.0. Researchers who want a cleaner fork of pi0.

### Tier: **0** (alternate impl of pi0; superseded by Physical Intelligence's official `openpi`).

---

## 15. StarVLA (`starVLA/starVLA`, arXiv 2604.05014)

### What they ship
- "Lego-like" modular codebase for VLA development. Unified multi-benchmark co-training: LIBERO, SimplerEnv, RoboTwin 2.0, RoboCasa-GR1, BEHAVIOR-1K, VLA-Arena.
- **Lightweight WebSocket Server-Client evaluation abstraction** — model side only exposes `predict_action()`. No code modification to deploy to a real Franka.
- LeRobot dataset v3.0 + DeepSpeed ZeRO-3 support.
- Opened by HKUST; described as drastically reducing reproduction cost.

### Packaging approach
- Python; modular plug-and-play components (model / data / trainer / config / eval / deploy, each independently debuggable).
- Franka real-robot example shipped 2026-03-19.

### LIBERO/lerobot packaging
- Uses the WebSocket client-server pattern. **Same pattern as allenai/vla-evaluation-harness and X-VLA.** Consistently decouples model env from sim env.

### Tier: **2** (VLA development platform, not a CLI-first deployment tool; good partnership target).

---

## 16. allenai/vla-evaluation-harness (arXiv 2603.13966)

### What they ship
- One framework to evaluate any VLA on any simulation benchmark (LIBERO / SimplerEnv / ManiSkill / CALVIN + others).
- **Benchmarks run inside Docker. Model servers are standalone `uv` scripts with PEP 723 inline deps.** `vla-eval serve` launches via `uv run` in an isolated env.
- WebSocket + msgpack wire format.

### Packaging approach
- **This is the gold-standard answer to the LIBERO/lerobot packaging puzzle.** Docker-per-benchmark isolation, `uv`-per-model isolation. Cross-eval matrix fills itself.

### Benchmarks they publish
- Harness is the benchmark. Paper by Choi, Lee, Park, Kim, Krishna, Fox, Yu (2026).

### License + audience
- Apache 2.0. AllenAI-backed. Audience: academic leaderboards; any team wanting reproducible VLA eval.

### Tier: **2** (consumed, not competed with. ADR `2026-04-14-wrap-not-rebuild-vla-eval.md` confirms this — `reflex.runtime.adapters.vla_eval.ReflexVlaEvalAdapter` implements their `PredictModelServer` interface).

---

## 17. OpenVLA-OFT (`moojink/openvla-oft`)

### What they ship
- Optimized Fine-Tuning recipe for OpenVLA. Parallel decoding + action chunking + continuous action representation + L1 regression.
- **26× faster action generation (5 Hz → 130 Hz), 3× lower latency, 97.1% LIBERO success (up from 76.5%).** With 25-timestep chunks, OFT+ is 43× faster than base.
- Fits on a single A100 / RTX 4090 16GB+. Enables real-time robot control without cloud.

### Packaging approach
- Research repo; extends OpenVLA training/inference code. No dedicated CLI.

### Tier: **3** (optimization technique; Reflex could reference these numbers for OpenVLA engine benchmarking).

---

## 18. UnifoLM-VLA-0 (Unitree, 2026-01-29)

### What they ship
- Unitree's open-source VLA for humanoid whole-body manipulation.
- **98.7% on LIBERO — ahead of OpenVLA, InternVLA, pi0.** The highest published LIBERO score in 2026-Q1.
- UnifoLM-WBT-Dataset — largest open humanoid whole-body teleoperation dataset released 2026-03-05 on HF.

### Packaging approach
- Model weights on HF; inference through Unitree's G1-D end-to-end humanoid platform.

### License + audience
- Open-source (check repo). Audience: humanoid researchers, Unitree customers.

### Tier: **2** (model target; Reflex could support as a VLA family if demand materializes).

---

## 19. Figure AI (Helix / Helix 02)

### What they ship
- Helix: "System 1 / System 2" dual-process VLA for humanoid upper body.
- **200 Hz upper-body control; zero-shot manipulation of thousands of unseen objects; fully on-board embedded low-power GPUs.**
- Figure 02 robot shipped with Helix; BotQ factory at 12,000 units/year target; 90-minute build cadence.
- Figure 03 in development.

### Packaging approach
- Vertically integrated. Closed. Own robot + own model + own deployment.

### License + audience
- Proprietary. End customers.

### LIBERO/lerobot packaging
- N/A. Closed stack.

### Tier: **3** (won't sell model; can't be "Windows of robotics"; non-overlapping segment).

---

## 20. 1X Technologies (Neo / World Model)

### What they ship
- 1X World Model — physics-based cognitive core; video+prompt conditioning.
- Neo humanoid: $20k pre-order, $499/mo subscription; U.S. deliveries 2026; 10k-unit deal with EQT portfolio companies 2026-2030.
- Trained on NVIDIA Blackwell HGX B200 GPUs.

### Packaging approach
- Vertically integrated. Closed.

### License + audience
- Proprietary. Consumers + EQT industrial.

### Tier: **3** (closed stack).

---

## 21. Skild AI (Skild Brain)

### What they ship
- Omni-bodied "Skild Brain" foundation model for quadrupeds, humanoids, tabletop arms, mobile manipulators.
- **$1.4B Series C (SoftBank, NVentures, Bezos, Samsung, LG, Schneider, Salesforce). $14B+ valuation.** Closed Jan 2026.
- Trained on human video + physics sim. "1000× more data points than competitors" claim.
- Partnerships with ABB Robotics, Universal Robots, NVIDIA.

### Packaging approach
- Closed. Platform-as-a-service pitch: "we build the brain, someone else builds the body."

### License + audience
- Proprietary. Robot OEMs.

### Tier: **2** (future customer — if Skild needs a deployment partner for OEM integrations, Reflex could slot in).

---

## 22. Physical Intelligence (company, not repo)

### What they ship
- pi0, pi0.5 weights (Apache 2.0). Next release likely pi-1.5 or pi-2.
- $11B valuation. No deployment runtime shipped.

### Tier: **1** (existential risk — if PI ships `pi-serve` with pi-1.5 on Jetson Thor, Reflex's pi0-segment pitch evaporates).

---

## 23. Intrinsic (now Google, Flowstate)

### What they ship
- Flowstate — web-based dev env + simulation engine. "Skills" as composable robotic behaviors.
- Integrating with Gemini Robotics + DeepMind post-Feb-2026 absorption. Foxconn JV targeting U.S. manufacturing 2026.

### Packaging approach
- Web-first / cloud-first. Not edge.

### License + audience
- Proprietary. Industrial manufacturers.

### Tier: **3** (different layer — "skill libraries for industrial" not "deploy any VLA to edge").

---

## 24. Gemini Robotics + Gemini Robotics On-Device (DeepMind)

### What they ship
- Gemini Robotics — VLA built on Gemini 2.0, 20 Hz high-frequency motor control.
- Gemini Robotics-ER 1.6 — reasoning-focused model; ISO 13849 PLd-certified safety.
- **Gemini Robotics On-Device** — fully offline VLA for bi-arm robots, SDK + MuJoCo sim hooks, trusted-tester program.
- Boston Dynamics partnership.

### Packaging approach
- Closed SDK. Trusted-tester-only. Gemini-ecosystem gated.

### License + audience
- Proprietary. Enterprise + academic partners.

### Tier: **3** (different philosophy — Google wants the VLA layer itself, not the serve layer).

---

## 25. Covariant RFM-1 (absorbed by Amazon)

### What they ship
- 8B param robotics foundation model. Text/image/video/action/measurement autoregressive.
- Founders (Abbeel, Chen, Duan) + team moved to Amazon Fulfillment Tech & Robotics 2024-Q3; Amazon licensed the foundation model.
- Limited public deployment data.

### Tier: **0** (effectively defunct as a standalone deploy-tool vendor; now inside Amazon fulfillment).

---

## 26. AGIBOT (Genie Studio)

### What they ship
- AGIBOT A2 / X2 / G2 humanoids. 5,000 shipped as of CES 2026.
- **Genie Studio Agent** — deployment automation; torque-controlled grasping, high-freq navigation, VLA-driven decision-making encapsulated as "reusable nodes."
- Robot-as-a-Service (RaaS) leasing.

### Tier: **3** (integrated RaaS + humanoid hardware; not a "deploy any VLA" tool).

---

## 27. Xiaomi Robotics-0

### What they ship
- 4.7B-param open-source VLA with async execution decoupled (rollout-while-planner-thinks).
- Not a toolchain — a model family.

### Tier: **2** (model target; validates the RTC/async-chunking pattern).

---

## 28. RLinf / RLinf-VLA / RLinf-Co / RLinf-USER

### What they ship
- `RLinf/RLinf` — production-grade RL infrastructure for embodied AI. Adopted by AgiBot, X Square Robot, PsiBot, Dexmal, Moore Threads, D-Robotics.
- PyPI-installable since 2026-02.
- RLinf-VLA framework for RL fine-tuning of VLAs. 20-85% improvements on LIBERO, ManiSkill, RoboTwin.
- RLinf-USER for real-world online policy learning.
- Wan World Model RL support.

### Packaging approach
- PyPI. Python. Library-mode, not CLI-first.

### License + audience
- Open-source. Research + Chinese robotics companies.

### Tier: **3** (RL fine-tuning, not deployment; partnership target).

---

## 29. HBVLA / QuantVLA / QVLA / EaqVLA / Shallow-π / Legato

### What they ship (aggregated)
- **HBVLA (2602.13710)** — 1-bit post-training quantization for VLAs.
- **QuantVLA (2602.20309)** — scale-calibrated PTQ with attention temperature matching, output head balancing.
- **QVLA (2602.03782)** — channel-wise bit allocation prioritizing action-space sensitivity.
- **EaqVLA (2505.21567)** — encoding-aligned mixed-precision quantization.
- **Shallow-π (2601.20262)** — knowledge distillation for flow-based VLAs; depth reduction; targets Jetson Orin/Thor + humanoids.
- **Legato (2602.12978)** — training-time continuation for action chunking flow policies.

### Tier: **3** (quantization/distillation techniques to adopt in Reflex engine builds, not deploy-tool competitors).

---

## 30. StreamingVLA (arXiv 2603.28565, 2026-03-30)

### What they ship
- Streaming VLA with action flow matching + adaptive early observation. Asynchronous parallelization across vision-language-action stages.
- Latency speedup + execution fluency improvements.

### Tier: **3** (technique; informs Reflex serve architecture).

---

## 31. Running VLAs at Real-time Speed (arXiv 2510.26742)

### What they ship
- Multi-view VLA framework. 100% success rate on grasping tasks, high frame rates, streaming inference framework using pi0 policy.

### Tier: **3** (technique).

---

# Inference Runtimes (not deployment, but relevant)

## 32. ONNX Runtime

### What they ship
- Cross-framework inference runtime (Microsoft-led). Version 1.24.4 in 2026.
- Python 3.14 support. Strong Jetson support via ORT-GenAI for LLM-side primitives.

### Packaging
- Wheels per JetPack version. Matrix complexity is a known pain (cf. lerobot#819 torch/torchvision incompat).

### License
- MIT.

### Tier: **3** (Reflex builds on ONNX Runtime; it's an ingredient, not a competitor).

---

## 33. NVIDIA TensorRT / TensorRT-LLM / TensorRT Edge-LLM

### What they ship
- TensorRT — the canonical NVIDIA engine compiler. NVFP4, FP8, INT8, INT4.
- TensorRT-LLM — LLM-optimized.
- **TensorRT Edge-LLM (2026-new)** — C++ SDK for LLM/VLM inference on Jetson Thor, DRIVE AGX Thor, Jetson T4000. EAGLE-3 spec decoding, NVFP4, chunked prefill. **Open source, Apache 2.0. Direct answer to "run pi0-class VLM-heavy models on Jetson."**

### License
- Apache 2.0 for Edge-LLM; TensorRT proper has a mixed license.

### Tier: **1** for TensorRT Edge-LLM (most dangerous entrant of 2026; same wedge); **3** for TensorRT itself (ingredient).

---

## 34. NVIDIA Triton / Dynamo-Triton

### What they ship
- Multi-model, multi-framework inference server. TensorRT + PyTorch + ONNX + OpenVINO + Python + RAPIDS FIL.
- Model Ensembles for pipelined pre/post-processing.
- Cloud + datacenter + edge + embedded.

### License
- BSD-3.

### Tier: **3** (enterprise-grade serving layer; Reflex could expose a Triton backend as a deployment target).

---

## 35. TorchServe / PyTorch serve

### What they ship
- AWS+Meta joint. Serves PyTorch models. Aging; vLLM / SGLang ecosystems have eaten TorchServe's LLM mindshare.

### License
- Apache 2.0.

### Tier: **3** (adjacent; falling in relevance).

---

## 36. KServe (Kubeflow)

### What they ship
- Kubernetes-native serverless model serving. Framework-agnostic. Integrates with Ray Serve.

### License
- Apache 2.0.

### Tier: **3** (enterprise MLOps infra; not robotics-specific).

---

## 37. Ray Serve (Anyscale)

### What they ship
- Distributed serving on Ray. Dynamic scaling; composite pipelines.

### License
- Apache 2.0.

### Tier: **3** (infra layer).

---

## 38. BentoML / Truss (Baseten)

### What they ship
- BentoML — OSS serving framework. Truss (Baseten's OSS, underlies their platform). Not robotics-specific.

### License
- Apache 2.0.

### Tier: **3** (general serving; potential integration target via Truss).

---

## 39. MLflow (Databricks)

### What they ship
- Experiment tracking + model registry + deployment. >55% of production ML teams.

### License
- Apache 2.0.

### Tier: **3** (MLOps plumbing; Reflex could emit MLflow-compatible artifacts).

---

## 40. vLLM

### What they ship
- High-throughput LLM inference engine from UC Berkeley. 12,500 tok/s on H100.
- Multi-framework / multi-GPU support.

### License
- Apache 2.0.

### Tier: **3** (LLM-centric; relevant for VLM prefix inside VLAs; Reflex could split the VLM prefix to a vLLM backend).

---

## 41. SGLang

### What they ship
- ~16,200 tok/s on H100 (29% ahead of vLLM). RadixAttention for shared prefixes. Strong multimodal support.

### License
- Apache 2.0.

### Tier: **3** (same as vLLM — backend option).

---

## 42. LMDeploy

### What they ship
- ~16,200 tok/s on H100. Different tradeoffs than vLLM/SGLang.

### Tier: **3** (backend option).

---

## 43. Roboflow Inference

### What they ship
- Computer-vision inference for edge (YOLO26, RF-DETR at ICLR 2026).
- Docker containers with TensorRT per hardware target (NVIDIA, OAK, CPU).
- Zero-shot pose estimation for robotics (YOLO26-Pose + Workflows).

### License
- Apache 2.0.

### Tier: **3** (CV, not VLA — but packaging model is a good reference: Docker + TRT + a workflow builder layer).

---

## 44. Baseten / Modal / Replicate / Runpod / Cerebrium

### What they ship
- Managed serverless-GPU platforms. Cold start vs cost tradeoffs. Python-decorator-driven (Modal) or config-driven (Baseten).

### License
- Proprietary (platforms). BentoML/Truss open-source underneath for Baseten.

### Tier: **3** (where Reflex users might host their TRT engines; integration surface).

---

# Hardware-native stacks

## 45. NVIDIA Isaac ROS / Isaac Sim / Jetson

### What they ship
- **Isaac ROS** — accelerated ROS2 packages for navigation, perception. Prebuilt Debian packages on x86 and Jetson.
- **Isaac Sim 5.0 / Isaac Lab-Arena** — simulation + eval.
- **Jetson AGX Thor** — 2070 FP4 TFLOPS, 128 GB, 40-130 W. 7.5× AI perf vs Orin.
- **Jetson T4000 / T5000** — JetPack 7.1. 1200 FP4 TFLOPs.

### Packaging
- NGC containers. JetPack bundles.

### License
- JetPack proprietary; Isaac ROS open-source BSD-3.

### Tier: **3** (Reflex deploys *to* Jetson; partnership surface).

---

## 46. ROS 2 Jazzy / Kilted / franka_ros2

### What they ship
- ROS 2 Jazzy (LTS) / Kilted distributions.
- `franka_ros2 v3.0.0+` — official Franka Research 3 support on Jazzy. Cartesian impedance controllers (PR #51), GELLO teleoperation integration.

### License
- BSD-3 / Apache 2.0.

### Tier: **3** (integration target — `reflex serve` emits ROS2-compatible actions).

---

## 47. PyTorch Robot Controller / libfranka

### What they ship
- `libfranka` — C++ realtime control library for Franka FCI.
- No first-party "PyTorch robot controller" product; research wrappers (`justagist/franka_ros_interface`, `franka_ros_TUD`).

### License
- Apache 2.0 / BSD-3.

### Tier: **3** (actuation layer; out of scope for Reflex directly).

---

# Adjacent — observability, benchmarks, CV platforms

## 48. Foxglove

### What they ship
- Observability + multimodal-data platform for robotics.
- **$40M Series B (Nov 2025, Bessemer-led).** Customers: NVIDIA, Amazon, Anduril, Wayve, Dexterity.
- Foxglove 2.0 — Agent + Cloud; unified data platform for Physical AI.

### License
- Mozilla Public License 2.0 / proprietary cloud.

### Tier: **2** (adjacent; partnership target — Foxglove visualizes what Reflex serves).

---

## 49. VLA-Perf (arXiv 2602.18397, Feb 2026)

### What they ship
- Analytical performance model (roofline-based) for VLA inference across hardware.
- Claims "first VLA inference benchmark."
- Reports: B100 11.7 Hz @ 1K timesteps but 1.2 Hz @ 10K. Jetson Thor 19.0 Hz steady. A100 → B100 ranges 61.7-314.4 Hz (datacenter).
- Phrase "memory-bound on Thor" worth stealing.

### License
- Research paper + code (check).

### Tier: **2** (benchmark authority framing — Reflex should publish numbers on this leaderboard or be ready when they extend to deployment-timing metrics).

---

## 50. VLAgents (arXiv 2601.11250)

### What they ship
- Framework-survey paper. Mostly taxonomy, not tooling.

### Tier: **3** (citation fodder).

---

## 51. awesome-efficient-vla (`guanweifan/awesome-efficient-vla`)

### What they ship
- Curated roadmap to the Efficient VLA landscape. Living repo.

### Tier: **3** (reading list; follow for 2026 arxiv drops).

---

## 52. awesome-physical-ai (`keon/awesome-physical-ai`)

### What they ship
- Academic papers + resources — VLAs, world models, embodied AI.

### Tier: **3** (reading list).

---

## 53. Efficient VLA Survey (arXiv 2510.17111)

### What they ship
- Systematic review: model architecture, perception features, action generation, training/inference strategies.

### Tier: **3** (reference).

---

## 54. NXP + HF blog: "Bringing Robotics AI to Embedded Platforms"

### What they ship
- SmolVLA ONNX FP32 baseline 6.15s latency on NXP embedded platforms. Fine-tuning + on-device optimization.

### License
- Vendor-specific.

### Tier: **3** (NXP ecosystem vendor; non-overlapping with Jetson-first).

---

## 55. Compute/grants (Inference.net, Crusoe, Trossen)

### What they ship
- Inference.net grants $10k, ~50% approval. Crusoe compute. Trossen hardware integrations with OpenPI + LeRobot.

### Tier: **3** (plumbing / capital; non-competitors).

---

# Summary table — competitor tiering by wedge

| Project | Surface | Tier | Why |
| --- | --- | --- | --- |
| LeRobot (HF) | Training + eval framework | 1 | Can ship `lerobot deploy` any day |
| OpenPI (Physical Intelligence) | Model + WebSocket serve | 1 | Existential risk (pi-serve) |
| NVIDIA TensorRT Edge-LLM | OSS C++ edge SDK | 1 | New 2026 entrant; same wedge |
| NVIDIA GR00T / Isaac Lab-Arena | Model + eval platform | 1 | Fold VLA mindshare into NV ecosystem |
| LiteVLA-Edge | Jetson-edge deployment pipeline | 1 | Direct Jetson competitor (6.6 Hz GGUF) |
| Physical Intelligence (co) | Foundation model lab | 1 | Could ship pi-serve |
| OpenVLA / OpenVLA-OFT | Model + REST serve | 2 | Partner/integration target |
| Dexbotic / Dexmal | PyTorch VLA toolbox | 2 | Research-first; partner target |
| X-VLA (`2toinf/X-VLA`) | Model + server-client | 2 | Validates WebSocket pattern |
| starVLA | Modular codebase | 2 | Research platform, partner |
| allenai/vla-evaluation-harness | Eval harness (Docker + uv) | 2 | We wrap, not compete |
| UnifoLM-VLA-0 (Unitree) | Model | 2 | Supported-model target |
| RDT-1B / RDT2-FM | Model | 2 | Supported-model target |
| Xiaomi Robotics-0 | Model | 2 | Validates async pattern |
| GigaBrain-0 / -Small | Model + edge variant | 2 | Supported-model target |
| Skild AI | Platform foundation model | 2 | Future customer |
| Foxglove | Observability | 2 | Partnership target |
| VLA-Perf | Benchmark authority | 2 | Publish numbers to their leaderboard |
| EdgeVLA / EfficientVLA / NanoVLA | Efficiency techniques | 3 | Adopt techniques, not competitor |
| Quantization papers (Quant/HB/Q/Eaq-VLA) | PTQ research | 3 | Adopt |
| Distillation (Shallow-π, Legato) | KD research | 3 | Adopt |
| StreamingVLA / RT Flow | Inference architecture | 3 | Adopt |
| ONNX Runtime | Cross-framework runtime | 3 | Ingredient |
| TensorRT | NV engine compiler | 3 | Ingredient |
| Triton / Dynamo-Triton | Multi-model server | 3 | Deployment-target |
| TorchServe / KServe / Ray Serve | Enterprise serving | 3 | Enterprise plumbing |
| BentoML / Truss | OSS serving | 3 | Integration |
| MLflow | MLOps tracking | 3 | Artifact compat |
| vLLM / SGLang / LMDeploy | LLM inference | 3 | VLM-prefix backend |
| Roboflow Inference | CV edge serving | 3 | Packaging reference |
| Baseten / Modal / Replicate / Runpod | Serverless GPU | 3 | Host for Reflex engines |
| NVIDIA Isaac ROS / Jetson | Hardware-native | 3 | Deploy target |
| ROS 2 Jazzy/Kilted + franka_ros2 | Middleware | 3 | Integration |
| libfranka / PyTorch controllers | Actuation | 3 | Out-of-scope |
| Figure Helix / Figure 02 | Vertically-integrated humanoid | 3 | Closed silo |
| 1X Neo / World Model | Vertically-integrated humanoid | 3 | Closed silo |
| Intrinsic/Flowstate (Google) | Industrial skills platform | 3 | Different layer |
| Gemini Robotics + On-Device | Closed VLA SDK | 3 | Different philosophy |
| AGIBOT Genie Studio | Humanoid RaaS | 3 | Different layer |
| Covariant RFM-1 | Absorbed by Amazon | 0 | Defunct as vendor |
| Octo | 2024 diffusion policy | 0 | Superseded |
| Open-Pi-Zero | 3rd-party pi0 reimpl | 0 | Superseded by `openpi` |
| VLAgents | Framework survey | 3 | Citation |
| awesome-efficient-vla / -physical-ai / Efficient VLA Survey | Reading lists | 3 | Reference |
| NXP HF blog | Embedded variant | 3 | Non-Jetson |
| RLinf / RLinf-VLA | RL fine-tuning | 3 | Partnership target |

---

# Consolidated LIBERO-vs-modern-lerobot solutions observed

Every serious 2026 project has converged on the same answer to "lerobot-v0.5 won't install alongside LIBERO's MuJoCo pin":

1. **Docker-per-benchmark isolation + uv-per-model isolation** — allenai/vla-evaluation-harness is the reference impl. Each benchmark gets its own Docker image; each model is a `uv run`-launched PEP 723 script. Model and benchmark communicate over WebSocket+msgpack.
2. **WebSocket server-client split** — starVLA, X-VLA, OpenPI, allenai, Dexmal, eventually everyone. Predict_action() over the wire. No single Python env has to satisfy both dependency trees.
3. **Migrate LIBERO tasks to Isaac Lab-Arena** — NVIDIA's systemic bet. MuJoCo → PhysX, CPU-bound → GPU-parallel. Lightwheel-LIBERO-Tasks already done. Eval time days → under an hour.
4. **Pin uv-incompatible LIBERO deps in a separate `examples/libero/` env** — OpenPI's pragmatic fix. LeRobot #2501 is still unfixed upstream.

**Reflex's ADR-2026-04-14 (wrap-not-rebuild-vla-eval) picks path 1+2** by implementing `ReflexVlaEvalAdapter : PredictModelServer`. This is the consensus-correct choice.

---

# Differentiation table — Reflex vs each Tier-1 competitor

| Tier-1 competitor | What they do | What Reflex does differently |
| --- | --- | --- |
| **LeRobot (HF)** | Training + eval; no ONNX/TRT; no serve composability; no wedge composition | Reflex ships export → TRT → serve with guard + split + RTC. `reflex export` is the command LeRobot hasn't shipped. Cross-VLA-family (SmolVLA + pi0 + pi0.5 + OpenVLA + GR00T, 4 families + 1). Drop a PR in `lerobot#3146` on launch. |
| **OpenPI (Physical Intelligence)** | Research checkpoints only; WebSocket policy server is rough; no first-party Jetson runtime; 6+ cross-embodiment issues with zero responses | Reflex = the bridge. Multi-VLA by construction (own the positioning before PI ships pi-serve). Published numbers: pi0 23.6 ms / 42 Hz on A10G, pi0.5 27.1 ms / 37 Hz. Apache bundle + guard + split. |
| **TensorRT Edge-LLM (NVIDIA)** | C++ SDK, LLM/VLM-centric, Jetson-Thor-native, NVFP4 + EAGLE-3 + chunked prefill | Reflex is Python-first, VLA-first (not LLM-adjacent), wraps TRT Edge-LLM as a *backend*. Cross-HW (A10G, H100, RTX 4090, Jetson Orin, Jetson Thor). Positions TRT Edge-LLM as an ingredient, not a replacement. |
| **NVIDIA GR00T / Isaac Lab-Arena** | Closed-ish ecosystem (though models Apache-2.0); Isaac Lab-Arena is eval-focused; GR00T bundled with Jetson SDK | Reflex already exports GR00T N1.6 (commit ff9fc3a wrapped DiT expert for raw-in/raw-out denoise loop). Cross-framework + cross-hardware before NVIDIA ships a bundled serving runtime (Watch signal: any GR00T release note mentioning "Serve"). |
| **LiteVLA-Edge** | GGUF + llama.cpp, Jetson AGX Orin, ROS2 pipeline, 6.6 Hz | Reflex uses TRT FP16 default (pro: FP8, INT4), targets 20-30 Hz. Weaker ROS2 story at launch; stronger on cross-VLA-family and cloud-GPU parity. |
| **Physical Intelligence (company)** | Could ship `pi-serve` with pi-1.5 in 2026 | Reflex must be multi-VLA and own the deployment-tool positioning by the time PI ships. Phase 2 bundle with Seeed / Trossen / integrators limits PI's gravity. |

---

# "What would reflex look like differentiated from each Tier 1 competitor?"

- **vs LeRobot:** Reflex is `lerobot export` + `lerobot serve`, cross-VLA-family, before they ship it. Own the `reflex` name and the 4-families story.
- **vs OpenPI:** Reflex is "the bridge" — the serve layer PI chose not to build. Multi-model positioning.
- **vs TensorRT Edge-LLM:** Reflex is Python-first VLA CLI that *uses* TRT Edge-LLM as a backend, not a C++ SDK. Narrower wedge, higher ergonomics.
- **vs NVIDIA GR00T / Isaac Lab-Arena:** Reflex is vendor-neutral (A10G, RTX, Jetson Orin, Jetson Thor) and family-neutral (SmolVLA, pi0, pi0.5, OpenVLA, GR00T). Isaac Lab-Arena locks you to NVIDIA's eval stack; Reflex wraps vla-eval for the same benchmark without the Isaac Sim dependency.
- **vs LiteVLA-Edge:** Reflex wins on cross-HW parity and VLA coverage; loses on ROS2 story out-of-box. Plumbing ROS2 is a Phase-2 differentiator.
- **vs Physical Intelligence:** Reflex wins on multi-VLA before PI ships pi-serve. After PI ships, Reflex's pi0 segment commoditizes — so Reflex must have GR00T + SmolVLA + OpenVLA revenue before that day.

---

# 2026 velocity signals (for GTM calibration)

- `openpi` 10.9k stars (Feb 2026).
- LeRobot v0.5.0 — 200+ PRs, 50+ new contributors, ICLR 2026 paper.
- GR00T / Cosmos Reason — 1M+ downloads on Cosmos Reason; top of Physical Reasoning Leaderboard.
- OpenVLA — 1000+ citations in 12 months ("State of Robotics 2026" SVRC).
- Foxglove $40M Series B (Nov 2025).
- Skild AI $1.4B Series C, $14B valuation (Jan 2026).
- Dexbotic + RLinf partnership (Feb 2026).
- Figure AI 90-minute humanoid build cadence.
- UnifoLM-VLA-0 published LIBERO 98.7% (Jan 2026), highest in category.
- Q1 2026 — "at least 11 commercial deployments using VLA as primary policy backbone."
- Quantized VLAs hitting 10-25 Hz on consumer GPUs.
- ICRA 2026 VLA Pipelines workshop (icra2026vlapipeline.github.io) + ~10k hours of real-world robot data competition.

---

# Key arxiv IDs / HF paper pages (2025-2026) worth monitoring

- `2602.18397` — VLA-Perf (benchmark authority framing)
- `2602.20309` — QuantVLA (PTQ with attention temp matching)
- `2602.03782` — QVLA (channel-wise bit allocation)
- `2602.13710` — HBVLA (1-bit PTQ)
- `2601.20262` — Shallow-π (KD for flow VLAs, Jetson Orin/Thor)
- `2602.12978` — Legato (training-time RT chunking)
- `2603.28565` — StreamingVLA (async parallelization)
- `2512.05964` — Training-Time Action Conditioning for RT Chunking (Black/Ren/Equi/Levine, Dec 2025)
- `2506.07339` — RTC (Black/Galliker/Levine, Jun 2025) — the RTC foundation
- `2510.19430` — GigaBrain-0
- `2510.25122` — NanoVLA (52× speedup)
- `2510.26742` — Running VLAs at Real-Time Speed (Dexmal)
- `2510.17111` — Efficient VLA Survey
- `2604.05014` — StarVLA (Lego-like)
- `2603.13966` — vla-eval (allenai harness)
- `2510.06710` — RLinf-VLA
- `2602.12628` — RLinf-Co (sim-real co-training)
- `2602.10377` — Hardware Co-Design Scaling Laws (on-device LLMs for physical AI)
- `2512.20276` — ActionFlow (pipelined action accel on edge)
- `2503.02310` — Parallel Decoding for VLA
- `2502.19645` — OpenVLA-OFT

---

# Files linked / related

- `reflex_context/03_research/competitor_landscape.md` — synthesis file (`reflex` vs top-5 competitors, phases of competition)
- `reflex_context/02_research/competitors/physical_intelligence.md`
- `reflex_context/02_research/competitors/nvidia_groot.md`
- `reflex_context/02_research/competitors/lerobot.md`
- `reflex_context/02_research/competitors/vlagents.md`
- `reflex_context/02_research/competitors/allenai_vla_eval.md`
- `reflex_context/03_research/hardware_targets.md`
- `reflex_context/03_research/papers_referenced.md`
- `reflex_context/03_research/vla_eval_integration.md`
- `reflex_context/03_research/direct_torch_export_viability.md`
- ADR `2026-04-14-wrap-not-rebuild-vla-eval.md`
