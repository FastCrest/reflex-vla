# VLA Benchmark Landscape (April 2026)

Comprehensive enumeration of every VLA / robot-manipulation benchmark a reflex-vla user might want to run. Scoped to the deployment-quote question: **"where do we get the task-success numbers to put on the landing page?"**

Source mix: vla-eval harness docs + leaderboard (AllenAI, v0.1.0 April 5 2026), arXiv 2024-2026, Moritz Reuss's ICLR 2026 state-of-VLA survey (164 submissions analyzed), and primary repo README/install docs as of April 2026.

**Context assumed from `vla_eval_integration.md`:** reflex already has `ReflexVlaEvalAdapter` which plugs into vla-eval's WebSocket+msgpack `PredictModelServer`. Anything vla-eval supports "for free" is a tier-1 candidate by construction. LIBERO install on Modal is already a solved problem (bddl 1.0.1, robosuite 1.4.1, gym, hydra-core, easydict, einops, num2words; `MUJOCO_GL=osmesa`; LIBERO stdin wizard patched; `setup.py install_requires=[]` trap avoided).

---

## Headline finding

- **90% of all VLA papers at ICLR 2026 (n=164 submissions) cite LIBERO, CALVIN, or SIMPLER.** LIBERO is saturating (95-98% on standard suites), SimplerEnv has huge cross-paper variance (40-99%), CALVIN is more differentiating.
- **vla-eval already supports 14 benchmarks** (LIBERO, LIBERO-Pro, LIBERO-Mem, CALVIN, SimplerEnv, ManiSkill2, RLBench, RoboCasa, RoboCerebra, RoboTwin, MIKASA-Robo, Kinetix, VLABench, RoboMME). BEHAVIOR-1K and FurnitureBench are "coming soon" (v0.1.0 ships April 5 2026; 227 stars).
- **Real-world**: RoboArena (DROID-based, crowdsourced, 7 institutions), RoboChallenge Table30 (Dexmal + Huggingface), Isaac Lab-Arena (NVIDIA, 40x parallel speedup).
- **Known SmolVLA numbers exist only for LIBERO** (87.3% avg across 4 suites, per arXiv 2506.01844). Everything else = "run it yourself."
- **Install complexity heat-map (from most painful to least):**
  1. LIBERO (stdin wizard + robosuite pin + bddl pypi drift + osmesa) — reflex has already eaten this cost
  2. RoboCasa / RoboCasa365 (35.6 GB docker, 10 GB assets, mujoco-dll windows bugs)
  3. SimplerEnv (Vulkan, numpy 1.24.4 pin, TF 2.15.0, Python 3.10-3.11)
  4. CALVIN (Python 3.8 only, pyhash/setuptools conflict)
  5. ManiSkill3 (beta; sapien 3.0.0.b1; Python >=3.9; gymnasium 0.29.1 pin)
  6. RLBench (CoppeliaSim 4.1 binary install, PyRep fork)
  7. VLABench / RoboTwin (containerized via vla-eval; direct install less tested)

Consequence for reflex: **lean on vla-eval as the primary plumbing**. Any benchmark vla-eval ships a Docker image for is a ~day of adapter work. Benchmarks outside vla-eval's list are multi-day to weeks.

---

## Benchmark: LIBERO

### Description

- 4 suites × 10 tasks = 40 tasks: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (a.k.a. LIBERO-Long).
- Robot: single-arm Franka Panda in robosuite. 7-dim action = 6 joints + gripper. RGB 256×256 per camera (agentview + wrist if `send_wrist_image=True`). State = eef_pos(3) + axis_angle(3) + gripper_qpos(2).
- Sim engine: MuJoCo via robosuite 1.4.1. BDDL task definitions.
- Publication: Lifelong-Robot-Learning, NeurIPS 2023. Now de facto industry standard.

### Install story

**This is the minefield reflex has fought.** Current canonical path in `scripts/modal_libero10.py`:

```
debian_slim + torch + onnx + vla-eval
+ git clone Lifelong-Robot-Learning/LIBERO && pip install -e .
+ pip install bddl==1.0.1 robosuite==1.4.1 gym hydra-core easydict einops num2words
+ MUJOCO_GL=osmesa LIBERO_DATA_DIR=... LIBERO_BASE=... (skip stdin wizard)
+ python scripts/patch_libero.py  (regex-patch LIBERO __init__.py's input() calls)
+ nuke .pyc caches before first import
```

Known gotchas:
- `LIBERO/setup.py install_requires=[]` → requirements.txt never installs. If you trust requirements.txt it downgrades transformers/numpy and breaks ONNX stack.
- `robosuite==1.5+` removed `SingleArmEnv` → import errors. Must pin 1.4.1.
- `bddl==1.0.1` is on pypi but not in install_requires. LIBERO imports it at `env.reset()` time, not package-import time, so errors surface late.
- LIBERO `__init__.py` calls `input()` 3 times for a "data dir wizard" that hangs in Modal containers. Patched via regex in `scripts/patch_libero.py`.
- `MUJOCO_GL=osmesa` over `egl` — EGL hangs silently on Modal's debian_slim + NVIDIA.
- `num2words` is a transitive of `SmolVLMProcessor` — easy to miss.
- Huggingface has an official fork at `huggingface/lerobot-libero` (16 stars, Apr 2026) that bundles the simplifications. Uses Python 3.8.13, PyTorch 1.11.0 + CUDA 11.3.
- `pip install -e ".[libero]"` also works inside lerobot 0.5+.

### Adoption signal

- **Overwhelming #1 VLA benchmark by citations.** Moritz Reuss analysis: appeared in ~90% of ICLR 2026 VLA submissions (combined with CALVIN + SimplerEnv).
- Every SmolVLA, pi0, pi0.5, OpenVLA, GR00T N1.5, xVLA, Dream-VLA, VLA-0 paper reports LIBERO numbers.
- AllenAI vla-eval leaderboard uses it as the canonical quick-eval benchmark (2000 episodes in ~18 min on H100 with sharding).
- HuggingFace runs its own LIBERO leaderboard Space.

### SmolVLA numbers (if published)

From arXiv 2506.01844 (SmolVLA paper, v1 June 2025):
- **LIBERO-Spatial: 90%**
- **LIBERO-Object: 96%**
- **LIBERO-Goal: 92%**
- **LIBERO-Long (LIBERO-10): 71%**
- **Average: 87.3%** (0.45B params; outperforms Octo 75.1%, OpenVLA 76.5%, pi0 86.0% at 3.3B).

### Compatibility notes

- **lerobot / SmolVLA: native.** `lerobot/smolvla_libero` checkpoint exists on HF Hub, trained on 3 cameras (`camera1/2/3`). `reflex export lerobot/smolvla_libero --target desktop` is the canonical smoke test.
- **pi0 / pi0.5: fine-tune required.** PI0FAST scores reported as low on LIBERO (issue lerobot#955); pi0.5 base gets 96%+ after libero-specific post-training (issue lerobot#2533). Physical-Intelligence/openpi issue #711 discusses norm_stats pitfalls in pi0.5 LIBERO lora finetune.
- **OpenVLA: LoRA r=32 per-suite fine-tune.** OpenVLA paper v2 Appendix E. Not zero-shot.
- **GR00T N1.6: not a native LIBERO target** — trained on YAM / G1 / Agibot. Possible cross-embodiment eval but not zero-shot SOTA.
- **vla-eval support: YES.** Docker image `libero` (6.0 GB). Reflex already wraps via `ReflexVlaEvalAdapter`.

### Cost to benchmark

- ~3 min per episode at 150 steps on A10G. 10 tasks × 1 ep = 30 min smoke. 10 tasks × 10 eps = 5 hours (A10G).
- With vla-eval parallel sharding: 2,000 LIBERO episodes in ~18 min on single H100.
- Disk: 10 GB for datasets in RLDS format.

### vla-eval support: YES (docker `libero` 6.0 GB)

### Tier: 1 (reason: widely adopted, SmolVLA numbers published, install pain already paid, vla-eval wraps it natively; the reflex "LIBERO-10 0% task success" problem is a numerical-export bug, not a benchmark-integration bug)

---

## Benchmark: LIBERO-Pro

### Description

- Extension of LIBERO with **perturbations on 4 axes**: object attributes, initial positions, task instructions, environments.
- Same 40 tasks × perturbation variants.
- Purpose: expose LIBERO's memorization issue. Models with 98% standard-LIBERO collapse to 0% under LIBERO-Pro.

### Install story

- vla-eval ships docker image `libero-pro` (6.2 GB). One click via the harness.
- Direct install: `Zxy-MLlab/LIBERO-PRO` (arXiv 2510.03827, Oct 2025). Same underlying robosuite/bddl stack as LIBERO.

### Adoption signal

- **Emerging** — Oct 2025 arXiv; appeared in ICLR 2026 submissions.
- vla-eval's inclusion is the strongest adoption vector.
- Hard counter to "LIBERO is solved" claim.

### SmolVLA numbers

- **Not published for SmolVLA specifically** as of April 2026.
- Paper reports OpenVLA, pi0, pi0.5, UniVLA all crash from ~98% standard LIBERO to **0.0%** under the fully perturbed setting.

### Compatibility notes

- Same interface as LIBERO. Any model that works on LIBERO works here (the whole point is to prove generalization).
- vla-eval support: YES.

### Cost to benchmark

- Episode count grows with perturbation variants. Expect 2-4× LIBERO cost.

### Tier: 2 (reason: emerging benchmark, strong methodological contribution, but SmolVLA numbers would have to be generated by us — great if we want the "first to publish SmolVLA-Pro score" GTM beat)

---

## Benchmark: LIBERO-Mem / LIBERO-Plus

### Description

- Two robustness extensions of LIBERO. LIBERO-Plus (arXiv 2510.13626, Oct 2025) focuses on in-depth robustness analysis.
- vla-eval ships `libero-mem` docker image (11.3 GB) — significantly larger than base LIBERO, suggesting more scenes/assets.

### Install story

- vla-eval only. Direct install from `sylvestf/LIBERO-plus` github.

### Adoption signal

- **Niche/emerging.** Referenced in vla-eval's "17 benchmark" count.
- No SmolVLA numbers published.

### Tier: 3 (reason: experimental, no established numbers, only use if we want to claim robustness-test coverage)

---

## Benchmark: SimplerEnv

### Description

- 8 core manipulation tasks on two robots:
  - Google Robot: `pick_coke_can`, `pick_object`, `move_near`, `open_drawer`, `close_drawer`, `place_in_closed_drawer`
  - WidowX + Bridge: `spoon_on_towel`, `carrot_on_plate`
  - Plus variants (pose variations, drawer positions)
- Sim engine: SAPIEN via ManiSkill2 (real2sim fork) or ManiSkill3 (10-15x faster GPU-parallel).
- Purpose: evaluate real-world-trained policies in matched simulation (RT-1, RT-1-X, Octo).
- CoRL 2024 paper (arXiv 2405.xxxxx), now extended by BridgeV2 50-task variant (object/language/vision-language subcategories).

### Install story

- `conda create -n simplerenv python=3.10` → clone with submodules → `pip install numpy==1.24.4` → `pip install -e ManiSkill2_real2sim` → `pip install -e .`
- CUDA >= 11.8, <13. TensorFlow 2.15.0 required for RT-1 checkpoint.
- Known issues: Vulkan `enumeratePhysicalDevices ErrorInitializationFailed` on headless GPU nodes; segfault in `env.step()` inside some Docker setups; camera `take_picture` occasionally freezes.
- Python 3.13 fails at ManiSkill install — downgrade to 3.10 required.

### Adoption signal

- **1,000 GitHub stars** (`simpler-env/SimplerEnv`).
- 185 forks, 28 open issues (April 2026).
- Part of the "big 3" for VLA evaluation (LIBERO + SimplerEnv + CALVIN = 90% of ICLR 2026 papers).
- AllenAI has its own fork (`allenai/SimplerEnv`) aligned with vla-eval.

### SmolVLA numbers

- **Not published as of April 2026.** SmolVLA paper evaluates on LIBERO only; no SimplerEnv numbers.
- Bayesian-VLA (Oct 2025) reports 66.5% SoTA on SimplerEnv. Dream-VLA reports 50%+ gains.
- Opportunity: first to publish a SmolVLA SimplerEnv number (another "reflex ships the numbers" GTM angle).

### Compatibility notes

- **lerobot / SmolVLA: manual integration.** SimplerEnv's policy interface expects numpy arrays; needs an adapter in `src/reflex/eval/simpler.py` (which is already scaffolded but stubbed).
- **pi0 / pi0.5: known scores (50%+ gains claim from Dream-VLA comparison).**
- **OpenVLA: native** — DelinQu/SimplerEnv-OpenVLA fork exists. Primary real2sim harness for OpenVLA in the literature.
- **GR00T: not a native target** (wrong embodiment).
- **vla-eval support: YES.** Docker image `simpler` (4.9 GB). Per-paper comparison remains difficult because of high variance across configs.

### Cost to benchmark

- Per-episode is fast (~30s-2min depending on task).
- 8 tasks × 100 evals = ~1-3 hours on a single A100.
- GPU required for rendering.

### Tier: 1 (reason: massive adoption as real2sim harness; vla-eval wraps it; first-published SmolVLA SimplerEnv number is a GTM win; already scaffolded in reflex at `src/reflex/eval/simpler.py`)

---

## Benchmark: ManiSkill3

### Description

- SAPIEN-based GPU-parallelized robotics simulator.
- 20+ tasks across manipulation (pick-place, peg insertion, chair-assembly, pour water, etc.) + locomotion.
- Integrates SimplerEnv Bridge environments for GPU parallel.
- RSS 2025 paper (arXiv 2410.00425). Documentation claims "10-1000x faster" than other platforms, 30,000+ FPS.

### Install story

- `pip install mani-skill` (currently v3 in beta).
- Python >= 3.9. Dependencies: `numpy>=1.22,<2.0.0`, `sapien==3.0.0.b1`, `gymnasium==0.29.1`, `h5py`.
- Known: Python 3.13 incompat (ManiSkill issue #646). SAPIEN camera occasional freezes (SAPIEN issue #171).

### Adoption signal

- **Medium-high.** Used in SimplerEnv (via ManiSkill2_real2sim). Primary benchmark for the "GPU-parallel manipulation" research thread.
- Widely cited in 2025 papers; ICLR 2026 mentions it as an integration target rather than primary benchmark.

### SmolVLA numbers

- **Not published directly.** SimplerEnv numbers on ManiSkill3 backend = transitive.

### Compatibility notes

- **lerobot/SmolVLA: manual.** Scaffolded at `src/reflex/eval/maniskill.py`.
- **OpenVLA: native via RDT-1B, RT-X, Octo integrations in ManiSkill docs.**
- **GR00T: via ManiSkill's VLA integrations (unofficial).**
- **vla-eval support: YES (ManiSkill2).** Docker image `maniskill2` (9.8 GB). ManiSkill3 integration path less clear — vla-eval currently ships ManiSkill2 specifically.

### Cost to benchmark

- GPU-parallelized — cheap per-episode (30,000+ FPS claim for state-only envs).
- Rendering-heavy eval is slower but still GPU-dominated.

### Tier: 2 (reason: strong tech, but under-adopted as standalone VLA benchmark; most people run it via SimplerEnv or ManiSkill2 backend. Pick up only if the GPU-parallel story becomes important for reflex's runtime comparisons)

---

## Benchmark: RLBench

### Description

- 100+ hand-designed manipulation tasks on Franka Panda. RGB + depth + segmentation + proprioception.
- Sim engine: CoppeliaSim 4.1.0 via PyRep.
- Range: reaching, door opening, multi-stage (open oven + place tray).
- 1909.12271 (arXiv, 2019). Stepjam / James Kyle. Legacy but still used.

### Install story

- Two-phase: install CoppeliaSim 4.1.0 (binary download from CoppeliaRobotics) → install PyRep (wrapper) → install RLBench (pip).
- Python 3.8-3.10 works best. PyRep install is the hard part — requires specific Qt5 / LD_LIBRARY_PATH setup.
- Known: many install issues on headless Linux; works best on Ubuntu 20.04 with X server.

### Adoption signal

- **Medium-high.** ~1.4k GitHub stars (legacy benchmark).
- Used by BridgeVLA (88.2%), InternVLA, GeneralVLA (10/14 tasks SOTA).
- Less favored in 2026 — Moritz Reuss ICLR analysis doesn't mention it in top-3.

### SmolVLA numbers

- **Not published.** SmolVLA has no RLBench eval.
- GeneralVLA (arXiv 2602.04315) reports SOTA on 10/14 tasks. BridgeVLA 88.2%.

### Compatibility notes

- **lerobot/SmolVLA: manual integration.** Not scaffolded yet in reflex.
- **OpenVLA / pi0 / GR00T: manual.**
- **vla-eval support: YES.** Docker image `rlbench` (4.7 GB). CoppeliaSim licensing headaches means Docker is strongly preferred.

### Cost to benchmark

- Slower than LIBERO (CoppeliaSim not GPU-accelerated at the same level).
- ~5-10 min per episode depending on task complexity. Full suite = many hours.

### Tier: 2 (reason: established, Docker-wrapped by vla-eval, but SmolVLA numbers missing and install painful outside Docker. Include if we want "legacy benchmark coverage" on the leaderboard)

---

## Benchmark: Meta-World (MT50 / ML45 / ML10)

### Description

- 50 distinct robotic manipulation tasks on a simulated Sawyer arm in MuJoCo.
- Multi-task (MT1/MT10/MT50) and meta-learning (ML1/ML10/ML45) splits.
- Meta-World+ (arXiv 2505.11289) standardizes and introduces MT25/ML25 for ~50% cost reduction vs MT50/ML45.
- Farama-Foundation/Metaworld (actively maintained as of 2026).

### Install story

- `pip install metaworld` works on Python 3.8-3.11, Linux/macOS.
- LeRobot has a ready-to-use `lerobot/metaworld_mt50` dataset on HF Hub.
- Known: `AssertionError` with render modes = Gymnasium version mismatch.

### Adoption signal

- **Medium.** Canonical in RL research since 2019. Less common in VLA evals (VLAs tend to prefer language-conditioned benchmarks like LIBERO).
- 1.1k+ GitHub stars.

### SmolVLA numbers

- **Not published.** Meta-World is typically evaluated by RL algorithms (SAC, PPO, TD-MPC2) rather than VLAs.

### Compatibility notes

- **lerobot: partial.** Meta-World MT50 dataset on HF Hub exists for supervised training; native eval inside lerobot multi-eval suite.
- **VLA compatibility: Meta-World actions are 4-dim (Δx, Δy, Δz, gripper)** — not 7-dim; action-dim mapping required.
- **vla-eval support: NO** (not in the 17 supported list).

### Cost to benchmark

- Fast MuJoCo-based. ~30s-1min per episode.
- 50 tasks × 50 evals = manageable on CPU.

### Tier: 3 (reason: VLA-unfriendly action dim, not in vla-eval, no SmolVLA numbers published. Useful for RL-baseline comparisons but not for the deployment landing-page story)

---

## Benchmark: RoboArena (real-world)

### Description

- **Real-world distributed eval** on DROID robots (Franka Panda + ZED cameras + Robotiq gripper).
- Crowdsourced pairwise comparisons ("Chatbot Arena for robotics") across 7 institutions.
- 612 pairwise comparisons between 7 generalist policies in initial launch (RSS 2026 paper arXiv 2506.18123).
- Hosted at https://robo-arena.github.io/

### Install story

- Not an install-and-run benchmark. You submit your policy to the arena and evaluators at participating institutions (Stanford, Berkeley, TRI, etc.) run pairwise comparisons.
- Requires access to a DROID robot or submitting for remote eval.

### Adoption signal

- **Strong and growing.** Explicitly positioned as the "real-world leaderboard."
- Tony Lee / Karl Pertsch / Percy Liang advising. Initial 7 universities.
- Compatible with any DROID-trained generalist policy.

### SmolVLA numbers

- **Not on the arena leaderboard** as of April 2026. SmolVLA is not DROID-trained so scoring requires fine-tune first.

### Compatibility notes

- **lerobot / SmolVLA: pre-requires DROID fine-tuning.** `lerobot/smolvla_droid` doesn't exist publicly.
- **pi0 / pi0.5 / OpenVLA: native** (DROID-trained by default).
- **GR00T: via DROID subset.**
- **vla-eval support: NO.** RoboArena is not a sim benchmark; vla-eval is sim-only.

### Cost to benchmark

- **No local GPU cost.** Submit and wait.
- Wall-clock: weeks (depends on arena throughput).

### Tier: 2 (reason: strong real-world credibility signal, but SmolVLA doesn't target DROID and the submission process is a multi-week loop. Pick up as GTM move once we have a DROID-trained reflex baseline)

---

## Benchmark: RobotArena-Infinity

### Description

- **Real-to-sim translation benchmark.** Converts real robot videos into simulated digital twins via vision-language models + 2D-to-3D generation + differentiable rendering.
- Dual scoring: VLM-guided automated + crowdsourced human preference.
- arXiv 2510.23571, submitted Oct 2025, v2 Mar 2026.

### Install story

- Experimental. Relies on custom reconstruction pipeline. Not pip-installable.

### Adoption signal

- **Emerging.** Nascent as of April 2026.

### Tier: 3 (reason: experimental, no clear hook for reflex's deployment pitch)

---

## Benchmark: CALVIN

### Description

- Long-horizon language-conditioned continuous control.
- Task splits: D, ABC, ABCD (A/B/C/D are different environments — ABC is the hard generalization test: train on A+B+C, eval on D).
- Evaluates chains of 5 sequential tasks (can you do 1/2/3/4/5 in a row?).
- Primary sim engine: PyBullet with EGL rendering.
- Mees/Calvin, arXiv 2112.03227. NeurIPS 2022.

### Install story

- Clone with submodules → Python 3.8 conda env → `sh install.sh` → `sh download_data.sh`.
- PyBullet is the core dependency. Setuptools < 58 required if pyhash install fails.
- Known: EGL device IDs don't match CUDA device IDs on some systems. Multi-GPU training OOMs due to PyBullet GPU rendering.
- **Python 3.8 ONLY** — older than most current stacks.

### Adoption signal

- **Top-3 VLA benchmark.** Part of the 90% (LIBERO + CALVIN + SimplerEnv) citation share.
- 886 GitHub stars, 115 forks.
- Extended to L-CALVIN (10-step sequences) by Long-VLA (arXiv 2508.19958).
- X-VLA has a canonical CALVIN evaluation recipe (`2toinf/X-VLA/evaluation/calvin`).
- FLOWER (ICLR 2026) achieves 4.5+ on ABC — considered "very good" per Moritz Reuss.

### SmolVLA numbers

- **Not published for SmolVLA specifically.**
- Competitive models report 4.0-4.5 on ABC.

### Compatibility notes

- **lerobot / SmolVLA: manual — CALVIN is not in the lerobot multi-eval list.** Adapter work needed.
- **pi0: adapted by X-VLA and others. Not native.**
- **OpenVLA: known integrations.**
- **vla-eval support: YES.** Docker image `calvin` (9.6 GB). Strong integration — solves the Python 3.8 pain.

### Cost to benchmark

- Longer episodes than LIBERO (chains of 5 tasks).
- ~30-60s per subtask → 2.5-5 min per 5-task chain.
- Full ABC eval = 1000s of chains = many GPU-hours.

### Tier: 1 (reason: top-3 citation count, vla-eval wraps it, Python 3.8 hell is avoided via Docker. First-to-publish SmolVLA CALVIN number is a strong GTM angle and technical win)

---

## Benchmark: RoboCasa / RoboCasa365

### Description

- Large-scale kitchen simulation with 2,500+ kitchen scenes, 3,200+ 3D objects.
- RoboCasa365: 365 tasks (60 distinct kitchen activities + 300 composite long-horizon tasks). Published ICLR 2026 (arXiv 2603.04356).
- 600+ hours of human demos + 1,600+ hours synthetic.
- Sim engine: MuJoCo via robosuite (same family as LIBERO).

### Install story

- `git clone github.com/robocasa/robocasa && pip install -e .` → `python -m robocasa.scripts.setup_macros` → `python -m robocasa.scripts.download_kitchen_assets` (~10 GB).
- Uses robosuite master branch. Can conflict with LIBERO's robosuite 1.4.1 pin.
- Known: Windows `mujoco.dll not found`; EGL→WGL swap required; headless-Ubuntu MuJoCo import issues; Numba/NumPy compat occasionally requires `numba==0.56.4`.

### Adoption signal

- **Rising fast.** ICLR 2026 publication.
- Supports Diffusion Policy, pi0, GR00T natively in its benchmark recipes.
- Yuke Zhu (UT Austin) lab — not actually Diego lab as hinted in the brief; brief had a memory error.

### SmolVLA numbers

- **Not published.** SmolVLA targets LIBERO; RoboCasa is not in its eval battery.
- RoboCasa's reference baselines are pi0, Diffusion Policy, GR00T.

### Compatibility notes

- **lerobot / SmolVLA: manual.** Robosuite-compatible so the LIBERO adapter could be reused with config changes.
- **pi0, GR00T: native** — RoboCasa ships benchmark configs.
- **vla-eval support: YES.** Docker image `robocasa` (**35.6 GB** — largest image in the harness). The asset-download cost is moved into the image.

### Cost to benchmark

- Long-horizon tasks (500+ timesteps vs LIBERO's 120).
- Expect 5-10× LIBERO wall-clock for comparable task counts.
- GPU required for rendering.

### Tier: 2 (reason: very strong emerging benchmark, vla-eval wraps it, aligns with pi0 / GR00T which reflex already supports. Cost of running is high — 35.6 GB docker. Good Phase-2 target; LIBERO first)

---

## Benchmark: RoboCerebra

### Description

- Long-horizon manipulation targeting System 2 reasoning. Hierarchical VLM-planner + VLA-controller architecture.
- 1,000 human-annotated trajectories across 100 task variants, up to 3,000 simulation steps per task.
- Tasks: prepare drinks, tidy groceries, etc. Household focus.
- arXiv 2506.06677 (Oct 2025).

### Install story

- Experimental. Primary path via vla-eval's `robocerebra` docker image (6.4 GB).

### Adoption signal

- **Emerging.** Dec 2025 release. Included in vla-eval's v0.1.0 (April 2026).

### SmolVLA numbers

- Not published. The benchmark's baselines are VLMs (GPT-4o, Qwen2.5-VL, LLaVA-Next-Video) as planners, not SmolVLA.

### Compatibility notes

- **Requires a planning-capable VLM layer** — pure SmolVLA / pi0 don't fit cleanly.
- **vla-eval support: YES.** Docker image `robocerebra` (6.4 GB).

### Tier: 3 (reason: reasoning-stacked benchmark, not a clean SmolVLA target, too specialized for deployment landing page)

---

## Benchmark: RoboTwin 2.0

### Description

- Scalable data generator + benchmark for **bimanual dual-arm manipulation**.
- 50 dual-arm tasks across 5 robot embodiments.
- Object library RoboTwin-OD: 731 instances, 147 categories.
- 5-axis domain randomization (clutter, textures, lighting, table height, language).
- arXiv 2506.18088 (RSS 2026).

### Install story

- `robotwin-platform.github.io`. Python + Sapien + mujoco combo. Direct install non-trivial.
- vla-eval ships `robotwin` docker (28.6 GB).

### Adoption signal

- **Medium-strong.** CVPR 2025 MEIS Workshop ran a RoboTwin Dual-Arm Challenge.
- Tests: ACT, DP, RDT, Pi0, DP3 as baselines.

### SmolVLA numbers

- **Not published.** SmolVLA is single-arm.

### Compatibility notes

- **lerobot / SmolVLA: poor fit** (SmolVLA is single-arm).
- **pi0: native benchmark baseline.**
- **ACT, DP, RDT, DP3: native.**
- **vla-eval support: YES** (28.6 GB docker).

### Cost to benchmark

- Heavy — bimanual tasks + domain randomization + 50 tasks.

### Tier: 3 (reason: single-arm SmolVLA doesn't belong; pi0 benchmarks exist but real competition is ACT/DP. Not a reflex priority unless we claim bimanual support)

---

## Benchmark: VLABench

### Description

- 100 task categories × strong randomization × 2,000+ objects.
- Evaluates: (1) common sense + world knowledge, (2) mesh/texture, (3) semantic instructions, (4) spatial understanding, (5) physics rules, (6) reasoning.
- Long-horizon (500+ timesteps vs 120 for primitive tasks).
- ICCV 2025 (arXiv 2412.18194). OpenMOSS group.

### Install story

- Direct: `OpenMOSS/VLABench` github. Non-trivial.
- vla-eval ships `vlabench` docker (17.7 GB).

### Adoption signal

- **Medium.** ICCV 2025 publication. Currently the most ambitious "general capability" VLA benchmark.

### SmolVLA numbers

- Not published. Benchmark claims "both current SOTA VLAs and VLM-workflow approaches face challenges" — weak baselines across the board.

### Compatibility notes

- **vla-eval support: YES.**
- Tests VLAs, VLM/LLM workflows, and VLMs in three separate eval streams.

### Tier: 3 (reason: research-oriented stress test; SmolVLA-scale models aren't expected to excel. Useful if we want to claim "tested on hardest benchmark")

---

## Benchmark: MIKASA-Robo

### Description

- 32 tasks × 12 groups evaluating **memory-intensive tabletop manipulation**.
- Taxonomy: object, spatial, sequential, capacity memory.
- Paper: arXiv 2502.10550 (CognitiveAISystems lab).

### Install story

- `pip install -e .` from `CognitiveAISystems/MIKASA-Robo`.
- vla-eval docker `mikasa-robo` (10.1 GB).

### Adoption signal

- **Niche.** New (Feb 2025). Specialized on memory.

### SmolVLA numbers

- Reported on some subset (ShellGameTouch, InterceptMedium, RememberColor3/5/9). Numbers suggest VLA memory is weak across the board — not a differentiating benchmark.

### Compatibility notes

- **vla-eval support: YES.**

### Tier: 3 (reason: specialized memory benchmark, not a mainstream VLA eval)

---

## Benchmark: Kinetix

### Description

- Included in vla-eval as docker image `kinetix` (10.0 GB). Minimal public documentation.
- Possibly related to Kinetix physics engine or kinetix-inspired manipulation tasks.

### Adoption signal

- **Unknown / niche.**

### Tier: 3 (insufficient signal; include only if vla-eval-driven "free eval" is worth one line)

---

## Benchmark: RoboMME

### Description

- "Benchmarking and Understanding Memory for Robotic Generalist Policies" (arXiv 2603.04639, March 2026).
- vla-eval docker `robomme` (17.0 GB).

### Adoption signal

- **Emerging (March 2026).**

### Tier: 3 (very new, no established SmolVLA numbers)

---

## Benchmark: ARNOLD

### Description

- Language-grounded manipulation with 8 tasks across 40 objects and 20 scenes.
- Built on NVIDIA Isaac Sim; photo-realistic RGB + 5 camera views.
- 10k expert demos.
- ICCV 2023 (arnold-benchmark.github.io).

### Install story

- Requires Isaac Sim (multi-GB NVIDIA install).
- Nontrivial outside NVIDIA-curated setups.

### Adoption signal

- **Low-medium.** Older (2023). Eclipsed by RoboCasa and VLABench in 2026.

### Tier: 3 (Isaac Sim dep too heavy, adoption waning)

---

## Benchmark: BEHAVIOR-1K

### Description

- 1,000 everyday household activities in ecological virtual environments (houses, offices, restaurants).
- BEHAVIOR Robot Suite extends with whole-body manipulation (arXiv 2503.05652).
- Stanford (Fei-Fei Li lab).

### Install story

- Uses OmniGibson (Isaac Sim under the hood). Multi-step install, heavy.
- vla-eval lists BEHAVIOR-1K as "coming soon" (not yet in v0.1.0).

### Adoption signal

- **Medium.** Strong academic prestige (Stanford, BEHAVIOR name).

### SmolVLA numbers

- Not published.

### Compatibility notes

- **vla-eval support: COMING SOON.**
- Whole-body manipulation is outside SmolVLA's manipulation focus.

### Tier: 2 (reason: pending vla-eval support; if BEHAVIOR-1K ships in vla-eval, it becomes a free benchmark for us. Monitor)

---

## Benchmark: FurnitureBench

### Description

- Real-world furniture assembly benchmark (IKEA-style).
- Long-horizon, dexterous.
- 200+ hours pre-collected demos, 5000+ trajectories, 3D-printable furniture models.
- Published RSS 2023 extended in IJRR 2025.

### Install story

- Real-world hardware setup. Sim version in OmniGibson.
- vla-eval lists as "coming soon."

### Adoption signal

- **Medium.** Established benchmark but real-world assembly is niche.

### SmolVLA numbers

- Not published.

### Tier: 3 (real-world assembly is too niche for SmolVLA's positioning; skip unless a user specifically asks)

---

## Benchmark: RoboChallenge / Table30

### Description

- **Real-robot** evaluation platform. Initial benchmark Table30 = 30 tasks around a fixed table.
- Cross-embodiment: single-arm and dual-arm setups.
- Dexmal + HuggingFace partnership. Launched Oct 2025 (arXiv 2510.17950).
- As of Jan 2026: Spirit AI's Spirit v1.5 ranked #1.

### Install story

- Submit your model for remote real-robot evaluation. No install.

### Adoption signal

- **Rising.** Large Chinese industry push. HF backing.

### SmolVLA numbers

- Not on the leaderboard as of April 2026.

### Compatibility notes

- **vla-eval support: NO** (real-world only).
- Cross-embodiment — potentially good for GR00T and pi0.

### Tier: 2 (reason: real-world credibility signal with industry adoption; GTM angle for reflex "our exported model hits RoboChallenge"; no SmolVLA numbers yet → first-published opportunity)

---

## Benchmark: Isaac Lab-Arena (NVIDIA)

### Description

- NVIDIA's open-source framework for large-scale policy eval in Isaac Lab (GPU-accelerated).
- Co-developed with Lightwheel. Released 2026.
- 40× parallel speedup vs sequential Isaac Lab eval (34.9 hours → 0.76 hours).
- Used internally by NVIDIA to benchmark GR00T N1.6.

### Install story

- Requires Isaac Lab (Isaac Sim under the hood). NVIDIA GPU required.
- `developer.nvidia.com/isaac/lab-arena` for the SDK.

### Adoption signal

- **Rising fast (NVIDIA backing).** Hugginface LeRobot integration announced Feb 2026.
- Three benchmarks already refreshed on Isaac Lab-Arena (Lightwheel announcement Jan 2026).

### SmolVLA numbers

- Not published. NVIDIA's reference model is GR00T N1.6.

### Compatibility notes

- **GR00T: native.**
- **lerobot / SmolVLA: via the HF LeRobot Environment Hub integration (Feb 2026 blog).**
- **vla-eval support: NO** (parallel strategy).

### Cost to benchmark

- Fast GPU-parallel — 40× faster than sequential Isaac Lab.
- Requires NVIDIA GPU.

### Tier: 2 (reason: aligns with reflex's GR00T + Jetson positioning; strong NVIDIA co-marketing opportunity if we run GR00T on Isaac Lab-Arena; SmolVLA not native target)

---

## Benchmark: RoboVerse

### Description

- **Unified platform** across 8+ physics engines (Isaac Lab/Gym, MuJoCo, SAPIEN, PyBullet, Genesis, cuRobo, PyRep, Blender).
- 1,000+ tasks, 10M+ transitions.
- Evaluates pi0, pi0-FAST, GR00T N1.5. RSS 2025.

### Install story

- Via MetaSim infrastructure. Docker is the realistic path.

### Adoption signal

- **Emerging / high potential.** Ambitious unification play.

### SmolVLA numbers

- Not in their reported baselines.

### Compatibility notes

- **vla-eval support: NO.**
- Covers pi0 / GR00T natively.

### Tier: 3 (reason: high-ambition but still consolidating; not yet a canonical citation)

---

## Benchmark: VLA-Arena (PKU-Alignment)

### Description

- Open-source benchmark for systematic VLA eval.
- Tasks: Safety, Distractor, Extrapolation, Long Horizon, LIBERO.
- Evaluates OpenVLA, UniVLA, pi0, **SmolVLA** (natively!) via standardized SR + cumulative cost metrics.
- arXiv 2512.22539 (Dec 2025).

### Install story

- `PKU-Alignment/VLA-Arena` github. Python-based.

### Adoption signal

- **Emerging (Dec 2025).** Safety and extrapolation focus differentiates.

### SmolVLA numbers

- **Published natively on the VLA-Arena leaderboard.** Exact numbers need to be pulled from the leaderboard JSON.

### Compatibility notes

- **Native SmolVLA support** — the most SmolVLA-aware benchmark on this list after LIBERO.
- **vla-eval support: NO (separate leaderboard).**

### Tier: 2 (reason: SmolVLA has published numbers, unique safety+extrapolation angle, GTM win for "reflex runs the certified SmolVLA")

---

## Benchmark: VLA-Perf (inference / latency, not task-success)

### Description

- **Not a task-success benchmark.** Analytical model for VLA inference latency across hardware × architecture combos.
- arXiv 2602.18397 (Feb 2026). Authors at UIUC / NVIDIA.

### Compatibility notes

- Measures latency and throughput, not success rate.
- **Relevant for reflex's serving pitch** — reflex serves SmolVLA at 86 Hz, pi0 at 42 Hz on A10G. VLA-Perf validates the inference-performance framing.

### Tier: 2 for differentiation (latency numbers; different axis from task success)

---

## Benchmark: DROID (dataset + eval)

### Description

- 76k demos / 350 hours on Franka Panda across 564 scenes, 86 tasks, 52 buildings.
- Collected by 50 collectors across 13 institutions.
- RSS 2024 (arXiv 2403.12945).

### Install story

- Dataset available via `droid-dataset.github.io` (CC-BY 4.0).
- Not a "benchmark" per se — it's training data. Eval is downstream (e.g., RoboArena uses DROID-trained policies).

### Adoption signal

- **Very high** as a dataset. Every large-scale generalist policy trains on DROID.

### SmolVLA numbers

- SmolVLA is not DROID-trained; no numbers published.

### Tier: 1-as-dataset, not-a-benchmark. Skip for the "claim task-success" goal.

---

## Benchmark: Open X-Embodiment + RT-1 / RT-2 evaluation suite

### Description

- 1M+ trajectories across 22 robot embodiments pooled from 60 datasets at 34 labs.
- RT-1-X (Transformer) and RT-2-X (VLM-based) trained on OXE.
- arXiv 2310.08864 (CoRL 2023, updated 2024).

### Install story

- Dataset, not a benchmark.
- Eval done via downstream benchmarks (SimplerEnv wraps RT-1 / RT-1-X as baselines).

### Adoption signal

- **Foundational dataset.** Moral weight for cross-embodiment claims.

### Tier: 1-as-dataset, skip-as-benchmark. The practical eval is SimplerEnv (tier 1).

---

## Benchmark: ALOHA / Mobile ALOHA

### Description

- Low-cost bimanual teleoperation hardware + imitation learning tasks.
- ALOHA 2 (arXiv 2405.02292) improves hardware; MuJoCo model released.
- Real-world only (unless you use the ALOHA 2 MuJoCo model).

### Install story

- Real-world hardware or the ALOHA 2 MuJoCo sim.

### Adoption signal

- **Hardware-heavy.** Used widely for bimanual papers (pi0, etc.).

### SmolVLA numbers

- Not applicable — SmolVLA is single-arm.

### Tier: 3 (bimanual, hardware-centric; not for reflex deployment story)

---

## Benchmark: TDMPC2 benchmark / DeepMind Control Suite

### Description

- 80-task multi-task (MT80) and 30-task (MT30) sets across DMControl + MetaWorld + Maniskill.
- TD-MPC2 reference results (arXiv 2310.16828).

### Install story

- `nicklashansen/tdmpc2` github. dm_control + mujoco required.

### Adoption signal

- **RL-focused**, not VLA.

### SmolVLA numbers

- Not applicable.

### Tier: 3 (RL/continuous-control focus, wrong benchmark family for VLA deployment)

---

## Benchmark: OpenVLA-Eval

### Description

- Not a distinct benchmark — OpenVLA paper's eval suite spans LIBERO (all 4 suites, LoRA r=32 per-suite), SimplerEnv (via DelinQu fork), BridgeV2 real-robot.

### Compatibility notes

- Functionally = LIBERO ∪ SimplerEnv ∪ BridgeV2 evaluated in the OpenVLA config.

### Tier: Functionally covered by tier-1 LIBERO + SimplerEnv.

---

## Benchmark: DexArt / DexMimicGen

### Description

- Dexterous manipulation with articulated objects. CVPR 2023.
- DexMimicGen: 21k synthetic trajectories for bimanual dexterous tasks. Used as multi-task benchmark.

### Adoption signal

- **Specialized dexterous / bimanual focus.** Not central to VLA.

### Tier: 3 (dexterous hand focus, wrong embodiment for SmolVLA)

---

## Benchmark: CortexBench

### Description

- 17 tasks across locomotion, navigation, dexterous, mobile manipulation.
- Assembles 7 sub-benchmarks.
- Meta EAI VC-1 paper (arXiv 2303.18240, 2023).

### Adoption signal

- **Representation-focused**, not current for VLA manipulation.

### Tier: 3 (visual-representation eval, not a VLA task-success benchmark)

---

## Benchmark: Habitat / Habitat 2.0

### Description

- Photorealistic RGBD sim. Embodied navigation + rearrangement.
- Facebook/Meta Habitat-Lab.

### Adoption signal

- **Navigation-first**, partial manipulation support.

### Compatibility notes

- Not a VLA manipulation benchmark per se.

### Tier: 3 (navigation-dominated; wrong benchmark family)

---

## Summary ranking table (all benchmarks)

| Benchmark | Tier | vla-eval | SmolVLA numbers | Install Ease | Adoption | Strategic value for reflex |
|---|---|---|---|---|---|---|
| **LIBERO** (4 suites) | 1 | YES (6.0 GB) | **Yes: 87.3% avg** | Hard (reflex already paid cost) | Overwhelming #1 | Canonical. Ship this first |
| **CALVIN** | 1 | YES (9.6 GB) | No (opportunity) | Hard (Python 3.8) — easy via Docker | Top-3 | First-to-publish SmolVLA CALVIN = GTM win |
| **SimplerEnv** | 1 | YES (4.9 GB) | No (opportunity) | Medium (Vulkan, TF, numpy pin) | Top-3 | Real2sim credibility; scaffolded in reflex |
| **LIBERO-Pro** | 2 | YES (6.2 GB) | No (opportunity) | Free via Docker | Emerging | Robustness angle |
| **ManiSkill3 / ManiSkill2** | 2 | YES (9.8 GB) | Indirect (via SimplerEnv) | Medium | Medium-High | GPU-parallel, good runtime benchmark |
| **RLBench** | 2 | YES (4.7 GB) | No | Hard outside Docker | Medium-High | Legacy coverage |
| **RoboCasa / 365** | 2 | YES (35.6 GB) | No | Hard | Rising | Aligns with pi0, GR00T; large image |
| **VLA-Arena** (PKU) | 2 | NO | **Yes: published on leaderboard** | Medium | Emerging | SmolVLA native + safety angle |
| **RoboArena** (real-world) | 2 | NO (real) | No | N/A (submit) | Rising | "Chatbot Arena for robots," Stanford |
| **RoboChallenge Table30** | 2 | NO (real) | No | N/A (submit) | Rising (China) | Real-world industry |
| **Isaac Lab-Arena** | 2 | NO (NVIDIA) | No | Needs Isaac | Rising | Aligns w/ GR00T + Jetson |
| **BEHAVIOR-1K** | 2 | COMING | No | Heavy (Isaac) | Medium | Watch list |
| **VLA-Perf** | 2 | N/A | Latency, not success | N/A (analytical) | Emerging | Validates reflex's serving pitch |
| **LIBERO-Mem / Plus** | 3 | YES (11.3 GB) | No | Docker | Niche | Robustness, redundant w LIBERO-Pro |
| **Meta-World (MT50)** | 3 | NO | No | Easy | Medium | RL-oriented, wrong action dim |
| **RoboCerebra** | 3 | YES (6.4 GB) | No | Docker | Emerging | Needs planner VLM; not pure VLA |
| **RoboTwin 2.0** | 3 | YES (28.6 GB) | No | Hard | Medium-Strong | Bimanual, not SmolVLA target |
| **VLABench** | 3 | YES (17.7 GB) | No | Hard | Medium | Hardest benchmark; stress test |
| **MIKASA-Robo** | 3 | YES (10.1 GB) | No | Docker | Niche | Memory-specialized |
| **Kinetix** | 3 | YES (10.0 GB) | No | Docker | Unknown | Free via vla-eval |
| **RoboMME** | 3 | YES (17.0 GB) | No | Docker | Emerging | Free via vla-eval |
| **ARNOLD** | 3 | NO | No | Heavy (Isaac) | Waning | Older (ICCV 2023) |
| **FurnitureBench** | 3 | COMING | No | Heavy | Medium | Real-world assembly |
| **DexArt / DexMimicGen** | 3 | NO | No | Medium | Niche | Dexterous hands |
| **RobotArena-Infinity** | 3 | NO | No | Experimental | Emerging | Too new |
| **RoboVerse** | 3 | NO | No | Hard | Emerging | Cross-sim unification |
| **ALOHA / Mobile ALOHA** | 3 | NO | No | Hardware | Medium | Bimanual hardware |
| **DexArt** | 3 | NO | No | Medium | Niche | Dexterous |
| **TDMPC2 bench / DMControl** | 3 | NO | No | Easy | Medium (RL) | RL-focused |
| **CortexBench** | 3 | NO | No | Medium | Niche | Representation eval |
| **Habitat** | 3 | NO | No | Medium | Medium (nav) | Navigation-focused |
| **DROID dataset** | 1-as-data | N/A | N/A | N/A | Very high | Training data, not benchmark |
| **Open X-Embodiment** | 1-as-data | N/A | N/A | N/A | Foundational | Training data, not benchmark |
| **OpenVLA-Eval** | 1-covered | via LIBERO+Simpler | Via sub-benchmarks | Covered | Covered | Already in tier 1 |
| **AllenAI vla-eval harness** | METATIER | N/A | Aggregates | Easy | Adopted | **The plumbing itself** |

---

## Recommended sequence for reflex

**Phase 1 (ship this month):**
1. LIBERO 4 suites via existing `ReflexVlaEvalAdapter`. Fix the 0% numerical-export problem (not a benchmark problem). Publish SmolVLA LIBERO matching the paper's 87.3%.
2. SimplerEnv 8 tasks. Plugin exists at `src/reflex/eval/simpler.py` stub. **First published SmolVLA SimplerEnv number** is a clean GTM beat.
3. CALVIN ABC via vla-eval docker. **First published SmolVLA CALVIN number.**

**Phase 2 (1-2 months out):**
4. LIBERO-Pro robustness numbers (0% collapse narrative — publish SmolVLA's robustness honestly).
5. VLA-Arena SmolVLA leaderboard submission (they natively support SmolVLA).
6. ManiSkill3 via GPU parallel (for the "how fast does reflex run in ManiSkill3" inference benchmark).
7. RoboCasa365 pi0 eval (pi0 is their native baseline, and reflex serves pi0 at 42 Hz).

**Phase 3 (real-world credibility, 3+ months out):**
8. RoboArena submission (requires DROID fine-tune first).
9. RoboChallenge Table30 submission.
10. Isaac Lab-Arena GR00T run (NVIDIA co-marketing).

**Skip entirely:**
- Meta-World, TDMPC2 bench, CortexBench, Habitat — wrong benchmark family.
- ARNOLD, DexArt — adoption waning or wrong embodiment.
- FurnitureBench, DexMimicGen, RoboTwin — bimanual/hardware focus.
- Kinetix, RoboMME, MIKASA-Robo — niche, no strategic value for the deployment landing page.

---

## Cross-reference: what vla-eval gives us for free (April 2026 v0.1.0)

Any benchmark that ships as a Docker image in vla-eval is ~1 day of adapter-config work:

```
libero, libero-pro, libero-mem, calvin, simpler, maniskill2, rlbench,
robocasa, robocerebra, robotwin, mikasa-robo, kinetix, vlabench, robomme
(coming: behavior-1k, furniturebench)
```

That's **14 benchmarks for free** if reflex's adapter works end-to-end. The LIBERO 0% issue (a numerical-export bug, see `direct_torch_export_viability.md`) is the gating problem — once fixed, the benchmark floodgates open.

---

## Open questions for the team

1. Do we publish **bad** SmolVLA LIBERO-Pro numbers (expected ~0%) as a "we're honest about robustness" signal, or wait for better?
2. Do we submit to RoboArena with a DROID-finetuned SmolVLA-DROID (doesn't exist yet), or do we wait?
3. Is the VLA-Arena PKU leaderboard credibility-neutral (China-affiliated) — does that affect GTM?
4. Should reflex ship `reflex eval <export_dir> <benchmark>` CLI *first* (from `vla_eval_integration.md` aspirational section), or ship LIBERO numbers first and CLI later?

---

## Sources (grounding)

- AllenAI vla-eval harness & leaderboard (arXiv 2603.13966, github.com/allenai/vla-evaluation-harness, allenai.github.io/vla-evaluation-harness/leaderboard)
- Moritz Reuss, "State of VLA Research at ICLR 2026" (mbreuss.github.io) — n=164 submissions, 90% LIBERO+CALVIN+SIMPLER citation share
- SmolVLA paper (arXiv 2506.01844) — LIBERO numbers 87.3% avg
- LIBERO-Pro (arXiv 2510.03827), LIBERO-Plus (arXiv 2510.13626), LIBERO-Para (arXiv 2603.28301)
- SimplerEnv (simpler-env.github.io, CoRL 2024)
- ManiSkill3 (arXiv 2410.00425)
- CALVIN (arXiv 2112.03227, mees/calvin github)
- RoboCasa365 (arXiv 2603.04356, ICLR 2026)
- RoboCerebra (arXiv 2506.06677), RoboTwin 2.0 (arXiv 2506.18088)
- VLABench (arXiv 2412.18194, ICCV 2025), VLA-Arena (arXiv 2512.22539)
- MIKASA-Robo (arXiv 2502.10550), RoboMME (arXiv 2603.04639)
- RoboArena (arXiv 2506.18123, robo-arena.github.io)
- RobotArena-Infinity (arXiv 2510.23571)
- RoboChallenge Table30 (arXiv 2510.17950)
- VLA-Perf (arXiv 2602.18397)
- Isaac Lab-Arena (developer.nvidia.com/isaac/lab-arena; NVIDIA + Huggingface LeRobot blog Feb 2026)
- Open X-Embodiment (arXiv 2310.08864), DROID (arXiv 2403.12945)
- RLBench (arXiv 1909.12271, stepjam/RLBench)
- Meta-World (Farama-Foundation/Metaworld), Meta-World+ (arXiv 2505.11289)
- ARNOLD (arnold-benchmark.github.io, ICCV 2023), BEHAVIOR-1K (behavior-robot-suite.github.io, arXiv 2503.05652)
- DexArt (arXiv 2305.05706), DexMimicGen, CortexBench (arXiv 2303.18240)
- ALOHA 2 (arXiv 2405.02292), FurnitureBench (IJRR 2025)
- RoboVerse (arXiv 2504.18904, RSS 2025)
- Reflex internal: `reflex_context/03_research/vla_eval_integration.md`, `competitor_landscape.md`, `scripts/modal_libero10.py`, `src/reflex/eval/{libero,simpler,maniskill}.py`

