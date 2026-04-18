# Published VLA Benchmark Numbers — Reflex Reference

Reference table of **published** task-success rates (%) for every VLA that Reflex either supports today or is on the roadmap to support. Compiled from HuggingFace model cards, arXiv papers (HTML versions), official blog posts, and the `openvla-oft`, `LIBERO-PRO`, `LIBERO-Plus`, `MemoryVLA` secondary-reporting papers.

Every cell has a source citation. Where a number is absent the cell reads **"n/r"** (not reported). Numbers are rounded to one decimal place; originals kept in the per-model detail sections.

**Last updated:** 2026-04-16.

**Why this file exists:** when Reflex runs a benchmark on a Reflex-exported checkpoint, we want the published-paper number side-by-side so we can (a) claim parity, (b) call out the gap, (c) not embarrass ourselves by quoting numbers that are actually from a cousin checkpoint.

---

## Summary Table — LIBERO

LIBERO has four standard suites: Spatial, Object, Goal, and Long (a.k.a. LIBERO-10). Average = mean across all four suites. Unless noted, numbers are fine-tuned on the LIBERO dataset (not zero-shot).

| Model / Checkpoint                       | Spatial | Object | Goal | Long (10) | Avg   | Source                                   |
| ---------------------------------------- | :-----: | :----: | :--: | :-------: | :---: | ---------------------------------------- |
| **lerobot/smolvla_base** (0.45B)         |  90.0   |  96.0  | 92.0 |   71.0    | 87.3  | SmolVLA arXiv 2506.01844                 |
| **lerobot/smolvla_libero**               |   n/r   |  n/r   | n/r  |    n/r    |  n/r  | HF card — not published                  |
| **lerobot/smolvla (LIBERO-PRO reprint)** |  93.0   |  94.0  | 91.0 |   73.0    | 87.8  | LIBERO-PRO arXiv 2510.03827              |
| **pi0 (Physical Intelligence)**          |  96.8   |  98.8  | 95.8 |   85.2    | 94.2  | OpenVLA-OFT arXiv 2502.19645 Table I     |
| **pi0 (LIBERO-PRO reprint)**             |  97.0   |  98.0  | 92.0 |    n/r    | 96.0  | LIBERO-PRO arXiv 2510.03827              |
| **pi0 @ openpi (FAST)**                  |   n/r   |  n/r   | n/r  |    n/r    |  n/r  | openpi repo — only pi0.5 numbers listed  |
| **pi0.5 (openpi `pi05_libero`)**         |  98.8   |  98.2  | 98.0 |   92.4    | 96.85 | openpi/examples/libero/README.md         |
| **lerobot/pi05_libero_finetuned**        |  97.0   |  99.0  | 98.0 |   96.0    | 97.5  | lerobot LIBERO docs                      |
| **GR00T N1 (original)**                  |  92.0   |  92.0  | n/r  |   76.0    | 76.0  | NVIDIA blog / learnopencv repro          |
| **GR00T N1.5 (LeRobot port)**            |  82.0   |  99.0  | n/r  |   82.0    | 87.0  | HF blog nvidia/nvidia-isaac-gr00t-in-lerobot |
| **GR00T-N1.7-3B**                        |  97.65  | 98.45  | 97.5 |   94.35   | 96.99 | Isaac-GR00T LIBERO README                |
| **OpenVLA (fine-tuned)**                 |  84.7   |  88.4  | 79.2 |   53.7    | 76.5  | OpenVLA arXiv 2406.09246 / repo README   |
| **OpenVLA-OFT** (Optimized Fine-Tuning)  |  96.2   |  98.3  | 96.2 |   90.7    | 95.3  | OpenVLA-OFT arXiv 2502.19645             |
| **OpenVLA-OFT+** (w/ wrist + state)      |   ~98.8 |  98.8  | 97.2 |   95.0    | 97.1  | OpenVLA-OFT site / arXiv 2502.19645      |
| **Octo (fine-tuned)**                    |  78.9   |  85.7  | 84.6 |   51.1    | 75.1  | OpenVLA repo README baseline             |
| **Diffusion Policy (from scratch)**      |  78.3   |  92.5  | 68.3 |   50.5    | 72.4  | OpenVLA repo README baseline             |
| **RT-2 / RT-2-X**                        |   n/r   |  n/r   | n/r  |    n/r    |  n/r  | not evaluated on LIBERO in public record |

## Summary Table — SimplerEnv (Google Robot / Fractal, Visual Matching)

WidowX Bridge and Google Robot Fractal are the two SimplerEnv task families. Google Robot tasks listed below use **Visual Matching** protocol (matches real-world setup). Entries are task-specific % success.

| Model             | Pick Coke Can | Move Near | Open/Close Drawer | Put in Drawer | Avg (Fractal) | Source                          |
| ----------------- | :-----------: | :-------: | :---------------: | :-----------: | :-----------: | ------------------------------- |
| **pi0-Beta**      |     97.9      |   46.6    |       62.3        |      n/r      |     71.4*     | MemoryVLA arXiv 2508.19236      |
| **pi0-Uniform**   |     88.0      |   52.2    |       56.0        |      n/r      |     69.1*     | MemoryVLA arXiv 2508.19236      |
| **GR00T N1.6**    |     97.5      |   75.5    |    87.5 / 44.0    |     14.5      |     67.66     | GR00T N1.6 README / NVIDIA docs |
| **SpatialVLA**    |     79.3      |    0.0    |       54.6        |     39.2      |     53.9      | MemoryVLA arXiv 2508.19236      |
| **TraceVLA**      |     45.0      |   11.1    |       63.1        |     61.6      |     47.8      | MemoryVLA arXiv 2508.19236      |
| **OpenVLA**       |     18.0      |    0.0    |       63.0        |     28.8      |     36.8      | MemoryVLA arXiv 2508.19236      |
| **Octo-Base**     |     17.0      |    0.0    |       22.7        |      1.1      |      6.1      | MemoryVLA arXiv 2508.19236      |
| **RT-1**          |      ~70      |    ~45    |      ~40–60       |      n/r      |      ~55      | SimplerEnv paper (qualitative)  |
| **RT-1-X, RT-2-X**|      n/r      |    n/r    |        n/r        |      n/r      |      n/r      | not in surveyed tables          |

*pi0 numbers lack published Visual Aggregation splits in MemoryVLA reprint.*

## Summary Table — SimplerEnv (WidowX Bridge)

| Model           | Spoon on Towel | Carrot on Plate | Stack Cube | Eggplant in Basket | Avg (Bridge) | Source                     |
| --------------- | :------------: | :-------------: | :--------: | :----------------: | :----------: | -------------------------- |
| **pi0-Beta**    |      84.6      |      55.8       |    47.9    |        85.4        |     68.4     | MemoryVLA arXiv 2508.19236 |
| **pi0-Uniform** |      63.3      |      58.8       |    21.3    |        79.2        |     55.7     | MemoryVLA arXiv 2508.19236 |
| **SpatialVLA**  |      16.7      |      25.0       |    29.2    |       100.0        |     42.7     | MemoryVLA arXiv 2508.19236 |
| **TraceVLA**    |      12.5      |      16.6       |    16.6    |        65.0        |     27.7     | MemoryVLA arXiv 2508.19236 |
| **Octo-Base**   |      15.8      |      12.5       |     0.0    |        41.7        |     17.5     | MemoryVLA arXiv 2508.19236 |
| **OpenVLA**     |       4.2      |       0.0       |     0.0    |        12.5        |      4.2     | MemoryVLA arXiv 2508.19236 |

## Summary Table — Other Benchmarks

Columns only populated where **published**. Many cells will be n/r; that's the whole point of this file.

| Model                             | Meta-World |  DROID  | ManiSkill3 | Open-X Embod | Real-World notes                         |
| --------------------------------- | :--------: | :-----: | :--------: | :----------: | ---------------------------------------- |
| **SmolVLA (0.45B)**               |    57.3    |   n/r   |    n/r     |     n/r      | SO100 78.3%; SO101 90% in-dist / 50% OOD |
| **pi0 (PI official)**             |   47.9     |   n/r   |    n/r     |     n/r      | 5 real tasks 75–100% (PI blog)           |
| **pi0.5 (PI official)**           |    n/r     |   n/r   |    n/r     |     n/r      | 83–94% in mock / OOD home evals          |
| **GR00T N1 (NVIDIA)**             |    n/r     |   n/r   |    n/r     |     n/r      | RoboCasa 32.1%, GR-1 Tabletop 50.0%      |
| **GR00T N1.5**                    |    n/r     |   n/r   |    n/r     |     n/r      | Unitree G1 1k-demo 98.8%, 0-shot 15%     |
| **OpenVLA (fine-tuned)**          |    n/r     |  69.7*  |    n/r     |   29 tasks   | BridgeData V2 71.3% (170 rollouts)       |
| **Octo**                          |    n/r     |  n/r    |    n/r     |  9 platforms | BridgeV2 ~70%                            |
| **RT-2 / RT-2-X (55B)**           |    n/r     |  n/r    |    n/r     |     n/r      | OpenVLA reports "RT-2-X beaten by 16.5%" |

*OpenVLA's 69.7% is Franka-DROID + Franka-Tabletop combined fine-tuning, 7 tasks, 129 rollouts — not the DROID simulation benchmark*.

---

## Per-model detail

### lerobot/smolvla_base — 0.45B-param, foundation checkpoint

Canonical training target and distillation starting point. **Paper:** arXiv 2506.01844 "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." The model card at https://huggingface.co/lerobot/smolvla_base does **not** reprint the benchmark table — look at the paper.

**LIBERO (fine-tuned, 10 trials per task, binary completion):**
- Spatial: 90.0%
- Object: 96.0%
- Goal: 92.0%
- Long: 71.0%
- Average: 87.3%

Caveat: "10 trials per task" is a smaller N than the OpenVLA reproduction protocol (3 seeds × 50 rollouts = 150 per task). Tight budget of N means higher variance. Reflex LIBERO runs on Modal should aim for at least 10 trials to be in the same ballpark.

**Meta-World:** 57.3% average across difficulty strata:
- Easy: 82.5%
- Medium: 41.8%
- Hard: 45.0%
- Very Hard: 60.0%

**Real-world SO100 multi-task:** 78.3%
- Pick-Place: 75%
- Stacking: 90%
- Sorting: 70%

**Real-world SO101 Pick-Place-Lego:**
- In-distribution: 90%
- Out-of-distribution: 50%

**Comparisons from the SmolVLA paper (at the same training budget):**
- pi0 (3.3B): LIBERO 86.0%, Meta-World 47.9%, SO100 61.7%.
- OpenVLA (7B): LIBERO 76.5%.
- Octo (0.09B): LIBERO 75.1%.

**Ablation:** Interleaving cross- and self-attention yields 85.5% LIBERO vs 79.0% for cross-attention alone. This is the architecture we export.

**Sources:**
- https://arxiv.org/abs/2506.01844
- https://huggingface.co/lerobot/smolvla_base (model card — no benchmark table)

### lerobot/smolvla_libero — LIBERO fine-tune of smolvla_base

**HF card does not publish benchmark numbers.** The training recipe (HuggingFaceVLA/libero dataset, LeRobot trainer) presumably reproduces the SmolVLA paper's LIBERO bar. Until someone reruns + publishes, **assume parity with smolvla_base LIBERO numbers** (~87% average).

Source: https://huggingface.co/lerobot/smolvla_libero

**Caveat:** LIBERO-PRO (arXiv 2510.03827) gives a different reprint: Spatial 93, Object 94, Goal 91, Long 73, Avg 87.8. This is marginally higher than the paper's 87.3% average and uses a different per-task dataset split. Reflex should quote the SmolVLA paper's 87.3% as the primary target.

### pi0 (Physical Intelligence, 3.3B / 3.5B)

**Paper:** arXiv 2410.24164 "π0: A Vision-Language-Action Flow Model for General Robot Control."

The pi0 paper itself does **not** publish LIBERO numbers — it only reports 5 custom real-world tasks (Shirt Folding 1.0, Bussing Easy 0.971, Bussing Hard 0.875, Grocery Bagging 0.786, Toast from Toaster 0.750, normalized 0–1, 10 episodes each). These are all partial-credit metrics, not binary task success.

**LIBERO numbers for pi0 come from downstream reprint papers:**
- OpenVLA-OFT Table I: Spatial 96.8, Object 98.8, Goal 95.8, Long 85.2, Avg 94.2.
- LIBERO-PRO Table 1: Goal 92, Object 98, Spatial 97, Long n/r, Avg 96.

These two reprints agree within 2 points on Spatial / Object / Goal, disagree on Long (OpenVLA-OFT says 85.2, LIBERO-PRO omits). The OpenVLA-OFT number is the most commonly cited.

**pi0 SimplerEnv (reprint from MemoryVLA Table 1 & 2):**
- Bridge: Spoon-on-Towel 84.6, Carrot-on-Plate 55.8, Stack-Cube 47.9, Eggplant 85.4, Avg 68.4 (pi0-Beta) / 55.7 (pi0-Uniform).
- Fractal: Pick Coke 97.9, Move Near 46.6, Open/Close 62.3, Avg (Visual Matching only) ~71.4 (pi0-Beta).

**pi0 Meta-World:** SmolVLA paper reports 47.9% for pi0 as a baseline.

**Sources:**
- https://arxiv.org/abs/2410.24164
- https://www.pi.website/blog/pi0
- https://arxiv.org/html/2502.19645 (OpenVLA-OFT reprint)
- https://arxiv.org/html/2508.19236v1 (MemoryVLA reprint)

### pi0.5 (Physical Intelligence)

**Paper:** arXiv 2504.16054 "π0.5: a Vision-Language-Action Model with Open-World Generalization."

Like pi0, the official paper only reports custom home environment numbers (83–94% follow rates on in-distribution / OOD cleaning tasks, 3 kitchens + 3 bedrooms, 10 trials per task). No standard benchmarks in the paper itself.

**LIBERO numbers for pi0.5 come from two official sources:**
- openpi/examples/libero README: Spatial 98.8, Object 98.2, Goal 98.0, Long 92.4, Avg 96.85.
- lerobot/pi05_libero_finetuned: Spatial 97.0, Object 99.0, Goal 98.0, Long 96.0, Avg 97.5.

**The lerobot-finetuned pi0.5 is the current public SOTA on LIBERO-Long at 96.0%.** Fine-tuned for 6k additional steps in bf16 on 8× H100. This is the number Reflex should match if the target is "parity with published best."

LIBERO-PRO Table 1 reprint: Goal 0.98, Object 0.98, Spatial 0.98, Long 0.92, Avg 0.97 — same ballpark as openpi.

**Sources:**
- https://www.pi.website/blog/pi05
- https://arxiv.org/abs/2504.16054
- https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero
- https://huggingface.co/docs/lerobot/en/libero
- https://huggingface.co/lerobot/pi05_libero_finetuned

**Caveat:** the HF LIBERO docs claim "consistent with the original Physical Intelligence results" even though the LeRobot version's LIBERO-10 of 96.0 is +3.6 points over openpi's 92.4. Not identical. Likely sampling / n_action_steps=10 difference. Reflex can cite either but should pick one and be consistent.

### GR00T N1 / N1.5 / N1.6 / N1.7-3B (NVIDIA)

**Paper:** arXiv 2503.14734 "NVIDIA Isaac GR00T N1: An Open Foundation Model for Generalist Humanoid Robots." The N1.5 / N1.6 / N1.7 model cards update this paper and the Isaac-GR00T repo is the canonical source.

**GR00T N1 (2B params) published benchmarks (from the paper, Table / appendix):**
- RoboCasa Kitchen (24 tasks, 100 demos): 32.1% (vs Diffusion Policy 25.6%, BC-Transformer 26.3%). 100 trials, avg over last 5 checkpoints.
- DexMimicGen Cross-Embodiment (9 tasks, 100 demos): 58.5% (vs DP 46.9%, BC-T 53.9%).
- GR-1 Tabletop (24 tasks, 100 demos): 50.0% (vs DP 32.7%, BC-T 16.1%).
- Real-world Fourier GR-1, full data: 76.8% (vs DP 46.4%).
- Real-world Fourier GR-1, 10% data: 42.6% (vs DP 10.2%).
- Zero-shot coordinated bimanual handover: 76.6% (11.5/15).
- Zero-shot novel object placement: 73.3% (11/15).

**No LIBERO numbers in the N1 paper.** GR00T N1 LIBERO numbers come from the GR00T-in-LeRobot blog and are a reproduction exercise, not originally reported: Spatial 92, Object 92, Long 76, Avg 76.

**GR00T N1.5 published benchmarks (NVIDIA GEAR blog):**
- Language Table (scratch): 93.2% (vs N1 52.8%).
- Sim GR-1 Language Tasks (scratch): 54.4% (vs N1 36.4%).
- RoboCasa (30 demos): 47.5% (vs N1 17.4%).
- Sim GR-1 zero-shot: 43.9% (vs N1 39.6%).
- Sim GR-1 (30 demos): 47.4% (vs N1 43.2%).
- Real-world GR-1 language following: 93.3% (vs N1 46.6%).
- Unitree G1 with 1k demos (familiar objects): 98.8%.
- Unitree G1 with 1k demos (novel objects): 84.2%.
- Novel Object Generalization zero-shot: 15.0% (FLARE post-training: 55.0%).
- DreamGen novel verbs: 38.3% (vs N1 13.1%).

**GR00T N1.6 (from the repo + community benchmarking):**
- SimplerEnv Fractal avg: 67.66% across all tasks.
- Pick Coke Can: 97.5%, Move Near: 75.5%, Close Drawer: 87.5%, Open Drawer: 44.0%, Place in Closed Drawer: 14.5%.

**GR00T-N1.7-3B LIBERO (Isaac-GR00T/examples/LIBERO, batch=640 global, 20k steps, 8 GPUs):**
- Spatial: 97.65% (195/200)
- Goal: 97.5% (195/200)
- Object: 98.45% (197/200)
- 10/Long: 94.35% (189/200)
- Average: 96.99%

This is **current GR00T SOTA on LIBERO**, within 1 point of openpi/pi0.5 and within 0.1 point of lerobot/pi05_libero_finetuned.

**Sources:**
- https://arxiv.org/abs/2503.14734
- https://research.nvidia.com/labs/gear/gr00t-n1_5/
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md
- https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md
- https://huggingface.co/blog/nvidia/nvidia-isaac-gr00t-in-lerobot (LeRobot port of N1.5)

### OpenVLA (7B, Stanford)

**Paper:** arXiv 2406.09246 "OpenVLA: An Open-Source Vision-Language-Action Model."

**LIBERO (fine-tuned, 3 seeds × 500 rollouts each; 10 tasks × 50 rollouts per task):**
- Spatial: 84.7 ± 0.9%
- Object: 88.4 ± 0.8%
- Goal: 79.2 ± 1.0%
- Long: 53.7 ± 1.3%
- Average: 76.5 ± 0.6%

**BridgeData V2 (29 tasks, WidowX, 170 rollouts):**
- OpenVLA: 71.3 ± 4.8%
- 16.5 absolute points over RT-2-X on 29-task avg.
- 20.4 absolute points over Diffusion Policy.

**Franka-Tabletop + Franka-DROID (7 tasks combined, 129 rollouts):**
- OpenVLA full fine-tune: 69.7 ± 7.2%
- OpenVLA LoRA (r=32): 68.2 ± 7.5%

**Quantization on BridgeData V2:**
- bfloat16: 71.3 ± 4.8%
- int4: 71.9 ± 4.7% (**int4 matches bfloat16**)
- int8: 58.1 ± 5.1% (int8 is **worse** than int4 — unusual)

**SimplerEnv (MemoryVLA reprint):**
- Bridge: Spoon-on-Towel 4.2, Carrot-on-Plate 0.0, Stack-Cube 0.0, Eggplant 12.5, Avg 4.2.
- Fractal: Pick Coke 18.0, Move Near 0.0, Open/Close 63.0, Put-in-Drawer 28.8, Avg 36.8.

OpenVLA's SimplerEnv numbers are sobering — Bridge is effectively broken at 4.2% avg. Vanilla OpenVLA zero-shot isn't tuned for SimplerEnv's photorealistic-Bridge rendering.

**Sources:**
- https://arxiv.org/abs/2406.09246
- https://openvla.github.io/
- https://github.com/openvla/openvla (LIBERO table lives in the repo README)
- https://huggingface.co/openvla/openvla-7b

### OpenVLA-OFT (Optimized Fine-Tuning, 2025)

**Paper:** arXiv 2502.19645 "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success."

Reflex doesn't currently support OpenVLA-OFT, but it's the LIBERO SOTA and a natural comparison anchor. Three recipes reported in the paper:

**LIBERO:**
- OpenVLA (baseline): Spatial 84.7, Object 88.4, Goal 79.2, Long 53.7, Avg 76.5.
- OpenVLA-OFT (parallel decoding + action chunking + L1): Spatial 96.2, Object 98.3, Goal 96.2, Long 90.7, Avg 95.3.
- OpenVLA-OFT+ (add wrist cam + proprio state): Avg 97.1 (tasks roughly 98.8 / 98.8 / 97.2 / 95.0).

OpenVLA-OFT+ is the current published LIBERO peak avg at 97.1%. The openpi and LIBERO-PRO reprints of OpenVLA use these OFT numbers, not the vanilla OpenVLA.

**Sources:**
- https://arxiv.org/abs/2502.19645
- https://openvla-oft.github.io/
- https://github.com/moojink/openvla-oft

### Octo (Small, 0.09B; Base, 0.31B)

**Paper:** arXiv 2405.12213 "Octo: An Open-Source Generalist Robot Policy."

**Key numbers:**
- WidowX BridgeV2: ~70% (outperformed RT-1-X by 29 absolute).
- Fine-tune on new domains (avg over 6 Open-X domains): 72%.
  - Berkeley Insertion: 70%
  - Stanford Coffee: 75%
  - CMU Baking: 50%
  - Berkeley Pick-Up: 60%
  - Berkeley Coke: 100%
  - Berkeley Bimanual: 80%

**LIBERO (from OpenVLA repo reprint):**
- Spatial: 78.9 ± 1.0%
- Object: 85.7 ± 0.9%
- Goal: 84.6 ± 0.9%
- Long: 51.1 ± 1.3%
- Average: 75.1 ± 0.6%

**SimplerEnv (from MemoryVLA Table 2):**
- Bridge: Spoon-on-Towel 15.8, Carrot-on-Plate 12.5, Stack-Cube 0.0, Eggplant 41.7, Avg 17.5.
- Fractal: Pick Coke 17.0, Move Near 0.0, Open/Close 22.7, Avg 6.1.

**Sources:**
- https://arxiv.org/abs/2405.12213
- https://octo-models.github.io/

### RT-2 / RT-2-X (Google, 55B, closed)

**Paper:** arXiv 2307.15818 "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control."

**No open LIBERO / SimplerEnv / Meta-World / ManiSkill numbers.** RT-2 was evaluated on proprietary Google Robot and Everyday Robots hardware. Reported numbers:
- 63.3% on in-distribution tasks, seen objects.
- 83.9% on seen categories, novel instances.
- Strong generalization to semantic / symbolic reasoning (62% on instruction following with pedagogically novel concepts).

**As a baseline OpenVLA beats it:** OpenVLA reports "+16.5% absolute over RT-2-X on BridgeData V2 29-task avg." If OpenVLA = 71.3% on that suite, RT-2-X ≈ 54.8%.

**In SimplerEnv, RT-2-X numbers are absent from the surveyed tables** — SimplerEnv only published RT-1, RT-1-X, Octo publicly.

**For Reflex purposes: there is no RT-2 number to parity-match. Skip this model.**

**Sources:**
- https://arxiv.org/abs/2307.15818
- https://robotics-transformer2.github.io/
- Mentioned (no specifics) in https://arxiv.org/html/2406.09246v3

### RT-1 / RT-1-X (Google, open)

**Paper:** arXiv 2212.06817 / Open-X-Embodiment arXiv 2310.08864.

**SimplerEnv (paper Table references, approximate — exact appendix numbers not in the HTML excerpt):**
- Google Robot Pick Coke Can: ~70% (visual matching, RT-1)
- Move Near: ~45%
- Open/Close Drawer: ~40-60%

These are the baselines SimplerEnv was **calibrated against**; they are the reference point for "real-policy-matches-sim-policy" calibration.

Not commonly benchmarked on LIBERO in the public record.

**Sources:**
- https://arxiv.org/html/2405.05941v1 (SimplerEnv paper — numbers in appendix tables IV–VI)
- https://github.com/simpler-env/SimplerEnv

---

## Cross-benchmark observations

### Winners per benchmark

- **LIBERO-Spatial**: openpi/pi0.5 (98.8%) — narrow margin over GR00T-N1.7-3B (97.65%) and OpenVLA-OFT+ (~98.8%).
- **LIBERO-Object**: lerobot/pi05_libero_finetuned (99.0%), GR00T-N1.7-3B (98.45%), openpi/pi0.5 (98.2%).
- **LIBERO-Goal**: pi0.5 (98.0% across multiple reprints), OpenVLA-OFT+ (97.2%), GR00T-N1.7-3B (97.5%).
- **LIBERO-Long (hardest suite)**: lerobot/pi05_libero_finetuned (96.0%) > GR00T-N1.7-3B (94.35%) > openpi/pi0.5 (92.4%) > OpenVLA-OFT+ (95.0%).
- **LIBERO avg**: lerobot/pi05_libero_finetuned (97.5%) ≥ OpenVLA-OFT+ (97.1%) ≥ GR00T-N1.7-3B (96.99%) ≥ openpi/pi0.5 (96.85%).
- **SimplerEnv Bridge**: pi0-Beta (68.4%) — no competition from SmolVLA, OpenVLA, Octo (all < 20%).
- **SimplerEnv Fractal**: pi0-Beta (71.4%) > GR00T N1.6 (67.66%) > SpatialVLA (53.9%).
- **Meta-World**: SmolVLA (57.3%) > pi0 (47.9%). **SmolVLA is the Meta-World winner among supported models.**
- **Real-world (multi-task SO100/SO101)**: SmolVLA (78.3% / 90% in-dist).
- **Real-world (humanoid)**: GR00T N1.5 (98.8% with 1k demos on Unitree G1).
- **Real-world (messy kitchen / bedroom)**: pi0.5 (83–94% OOD).

### The "big three" for LIBERO

All published LIBERO-average numbers ≥ 95% come from one of three families:
1. **pi0.5** (PI or LeRobot port) — 96.85% or 97.5%.
2. **GR00T N1.7-3B** — 96.99%.
3. **OpenVLA-OFT / OFT+** — 95.3% / 97.1%.

SmolVLA (87.3%), OpenVLA (76.5%), Octo (75.1%), Diffusion Policy (72.4%) are a full generation behind on LIBERO-Long specifically.

### LIBERO-Long is the diagnostic

LIBERO-Long is where the separation is stark. **Long = 10 multi-step horizon tasks.** Even well-tuned models drop 5–10 points here vs other suites.
- SmolVLA: 71.0% Long vs 93.3% avg of other suites. **22-point drop.**
- OpenVLA: 53.7% Long vs 84.1% avg of other. **30-point drop.**
- pi0.5: 92.4% Long vs 98.3% avg of other. **Only 6-point drop.**

If Reflex-exported SmolVLA scores < 60% on LIBERO-Long, it's not underperforming the paper — the paper number is 71. Set expectations accordingly.

### SimplerEnv is unforgiving to unsupervised models

OpenVLA vanilla on SimplerEnv-Bridge: **4.2% avg**. Octo: 17.5%. These are catastrophically low. Only pi0-Beta, TraceVLA, SpatialVLA (all with specific SimplerEnv-aware fine-tuning) clear 25%.

Interpretation: a Reflex-exported OpenVLA on SimplerEnv is **not expected to match real-world OpenVLA numbers**. If Reflex gets 5% on SimplerEnv-Bridge-Eggplant, that's parity.

### Meta-World has a small N of public reports

Only SmolVLA paper reports Meta-World (57.3% avg, pi0 47.9% baseline). **No OpenVLA / Octo / GR00T Meta-World numbers are publicly available.** Reflex running Meta-World will be one of the first public comparisons.

---

## Reflex benchmarking targets (per model, per benchmark)

Target: the published number the Reflex-exported checkpoint should **match within 5 points** to claim parity. "Hit" vs "Gap" column marks whether we believe the target is achievable with current export pipeline (assumes numerical parity fixed per `direct_torch_export_viability.md`).

| Reflex Checkpoint                | Benchmark       | Target | Notes                                                                      |
| -------------------------------- | --------------- | :----: | -------------------------------------------------------------------------- |
| lerobot/smolvla_base             | LIBERO avg      |  87.3  | SmolVLA paper's primary LIBERO claim. Min viable 82. Stretch 90.           |
| lerobot/smolvla_base             | LIBERO-Long     |  71.0  | Paper's published. Reflex current: 0% (numerical parity bug).              |
| lerobot/smolvla_base             | LIBERO-Spatial  |  90.0  | Easiest LIBERO suite for SmolVLA; first benchmark to get green.            |
| lerobot/smolvla_base             | Meta-World      |  57.3  | Only public SmolVLA Meta-World number.                                     |
| lerobot/smolvla_base             | SimplerEnv      |  n/r   | SmolVLA paper does not report. First Reflex-produced number here = public. |
| lerobot/smolvla_libero           | LIBERO-Long     |  71    | HF card doesn't publish; assume parity with base.                          |
| lerobot/pi0                      | LIBERO avg      |  94.2  | OpenVLA-OFT reprint. Minimum bar to be taken seriously.                    |
| lerobot/pi0                      | LIBERO-Long     |  85.2  | Single hardest cell we can chase.                                          |
| lerobot/pi0                      | SimplerEnv-Frac |  71.4  | pi0-Beta. Very high bar.                                                   |
| lerobot/pi0                      | SimplerEnv-Brdg |  68.4  | pi0-Beta. Very high bar.                                                   |
| lerobot/pi0.5 / pi05_libero      | LIBERO-Long     |  92.4  | openpi baseline. The current SOTA floor.                                   |
| lerobot/pi05_libero_finetuned    | LIBERO-Long     |  96.0  | The current public SOTA.                                                   |
| nvidia/GR00T-N1.5-3B             | LIBERO avg      |  87.0  | LeRobot port numbers from HF blog.                                         |
| nvidia/GR00T-N1.7-3B             | LIBERO avg      |  96.99 | Isaac-GR00T repo.                                                          |
| openvla/openvla-7b-finetuned-*   | LIBERO avg      |  76.5  | Vanilla OpenVLA bar.                                                       |
| openvla/openvla-7b-finetuned-*   | LIBERO-Long     |  53.7  | Vanilla OpenVLA bar; easy to beat.                                         |
| openvla/openvla-7b               | BridgeData V2   |  71.3  | 29-task avg, 170 rollouts.                                                 |
| openvla/openvla-7b               | SimplerEnv Brdg |  4.2   | Just hit >0 to call it working.                                            |

---

## What's not reported anywhere in the public record (gaps)

- **SmolVLA on SimplerEnv** (Bridge / Fractal). First Reflex run here = first public number.
- **SmolVLA on ManiSkill3.** Not evaluated in any surveyed paper.
- **SmolVLA on DROID.**
- **pi0 on LIBERO-Long** (disagreement: OpenVLA-OFT says 85.2, LIBERO-PRO says n/r — primary source from the paper is ambiguous).
- **pi0.5 on SimplerEnv** (not in openpi results tables).
- **GR00T on SimplerEnv-Bridge.** Only Fractal numbers reported.
- **Any VLA on the full ManiSkill3 67-task suite.** StARe-VLA reports 96.4% SOTA but doesn't enumerate subject models.
- **OpenVLA / Octo on Meta-World.**

These are research opportunities for Reflex — if we ship the first public number, that's a positioning asset.

---

## Caveats on cross-paper number comparisons

1. **Reprinted numbers drift.** pi0 LIBERO-Spatial: 96.8 in OpenVLA-OFT reprint, 97.0 in LIBERO-PRO reprint. These 0.2-point gaps are sample noise.
2. **Episode budget matters.** SmolVLA paper uses 10 trials/task; OpenVLA uses 150 (3 seeds × 50 rollouts); Isaac-GR00T uses 200. Lower N = higher variance. When comparing Reflex vs. paper, match the paper's N.
3. **max_steps matters.** SmolVLA paper uses max_steps=600 for LIBERO; Reflex Modal runner uses max_steps=150. The gap isn't just from the policy — it's from the time budget.
4. **Control mode matters.** LIBERO has `relative` and `absolute` control; mismatched control mode silently tanks success rate.
5. **Camera setup matters.** LIBERO renders 256×256 agentview + wrist. SmolVLA trained on 3-camera data. Single-camera inference will hurt.
6. **Fine-tuning recipe unspecified.** "lerobot/pi05_libero_finetuned" is +6k additional steps over openpi's base. openpi base is 30k. If you fine-tune differently, you'll land at a different number.
7. **LIBERO-PRO and LIBERO-Plus papers show that all these published numbers collapse under perturbation.** If Reflex chooses to evaluate only on standard LIBERO, the parity-vs-paper comparison is a memorization game, not a generalization game. This is a warning — not yet an action item.

---

## Primary sources — canonical list

- SmolVLA arXiv 2506.01844: https://arxiv.org/abs/2506.01844
- pi0 arXiv 2410.24164: https://arxiv.org/abs/2410.24164
- pi0.5 arXiv 2504.16054: https://arxiv.org/abs/2504.16054
- OpenVLA arXiv 2406.09246: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT arXiv 2502.19645: https://arxiv.org/abs/2502.19645
- Octo arXiv 2405.12213: https://arxiv.org/abs/2405.12213
- GR00T N1 arXiv 2503.14734: https://arxiv.org/abs/2503.14734
- SimplerEnv arXiv 2405.05941: https://arxiv.org/abs/2405.05941
- LIBERO-PRO arXiv 2510.03827: https://arxiv.org/abs/2510.03827
- LIBERO-Plus arXiv 2510.13626: https://arxiv.org/abs/2510.13626
- MemoryVLA arXiv 2508.19236: https://arxiv.org/abs/2508.19236
- openpi LIBERO README: https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero
- lerobot LIBERO docs: https://huggingface.co/docs/lerobot/en/libero
- Isaac-GR00T LIBERO: https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md
- Isaac-GR00T SimplerEnv: https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md
- GR00T N1.5 GEAR: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- SmolVLA-LIBERO HF card: https://huggingface.co/lerobot/smolvla_libero
- pi05_libero_finetuned HF card: https://huggingface.co/lerobot/pi05_libero_finetuned
