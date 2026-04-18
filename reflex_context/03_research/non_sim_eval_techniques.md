# Non-Simulation VLA Evaluation Techniques (2025-2026)

A comprehensive enumeration of every non-sim technique used to benchmark a Vision-Language-Action (VLA) model, with publishability and feasibility-as-Reflex-top-line-metric ranked.

**TL;DR for the Reflex decision:** Sim-based task-success (LIBERO, SimplerEnv) is the *only* headline metric the VLA field currently treats as publishable evidence that "the policy works." Every non-sim metric is either (a) a supplementary training signal that the community has repeatedly shown does NOT correlate with task success, or (b) a research proposal that hasn't yet been accepted as a replacement. A "99.9% action cosine similarity vs. lerobot expert on LIBERO-10 held-out frames" claim is defensible as a **correctness / parity** signal for export-pipeline fidelity (reflex's actual product), but it is **not publishable as a VLA-quality claim** — reviewers will ask for sim success. The right Reflex framing is not "our model is good" but "our inference preserves the upstream model's behavior to within N decimals," which is export-tool-correctness, not policy-quality.

Sources for this TL;DR: Barannikov 2024 (HF blog) and the LeRobot GH issue #2853 show validation loss and task success have **no correlation** (PushT: +134% val loss yet +55% success; Transfer Cube: -2% val loss and +17% success). SimplerEnv authors Pearson/MMRV frame their whole work around bridging offline-to-real. vla-eval's authors explicitly state "supported metrics are limited to task success rate." SmolVLA paper reports *only* success rate, no action-level offline metric. pi0 and OpenVLA papers report only task success or action-token accuracy. See end-of-document citations.

---

## 1. Trajectory replay action matching (L2 / cosine distance vs. expert)

### Definition
Given dataset D = {(o_i, s_i, a_i*)}_{i=1..N} of expert (observation, state, action) triples and a learned policy π_θ, compute:
- **Action MSE**: (1/N) Σ || π_θ(o_i, s_i) − a_i* ||_2^2
- **Action L1**: (1/N) Σ || π_θ(o_i, s_i) − a_i* ||_1
- **Action cosine**: (1/N) Σ (π_θ(o_i, s_i) · a_i*) / (||π_θ|| ||a_i*||)
- **Normalized variants** (NAMSE): min-max normalized per dimension so a 6D arm and a 14D bimanual comparable across datasets.

### Who uses it
- **Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks** (Sikka et al., arXiv 2411.05821) — uses AMSE, NAMSE, "Completion Rate" (final action matching) across OpenX/GPT-4o/OpenVLA/JAT. This is the most explicit published benchmark in the action-matching family.
- **NVIDIA Isaac GR00T** — official `open_loop_eval.py` reports MSE between predicted and ground-truth actions as the recommended offline signal. No threshold published; NVIDIA notes 5-6% run-to-run variance from non-deterministic augmentations.
- **OpenVLA-OFT** (arXiv 2502.19645) — L1 regression loss is the reported training objective and is monitored; "continuous L1 variants improve task success by 5% absolute over discrete action tokens."
- **LeRobot** — does NOT natively compute validation loss (GH issue #2853); users roll their own.
- Informal: nearly every imitation-learning codebase logs some form of per-step BC loss.

### Compute cost
- Trivial. O(dataset size) forward passes. ~5-30 GPU-minutes per 10k-frame LIBERO validation split on A10G.
- Can be streamed from a LeRobot dataset without any sim stack.

### Data requirements
- A held-out split of (o, s, a*) triples. LIBERO-10 has ~50 demos/task × 10 tasks × ~200 frames = ~100k frames. Freely available on HF (`lerobot/libero_10`).
- DROID validation split ~90k episodes (public).
- Bridge V2, RT-X, OpenX — all public, all compatible.

### Publishability
- **Not publishable as a VLA-quality headline.** The 2024 HF-blog evidence from Barannikov et al. — *"no correlation between MSE and success rate … The validation losses are more than twice as high after 50K training steps than after 20K, while success rate improves by 50%"* — circulates widely and has been absorbed into the community's prior: reviewers do not trust action-MSE as evidence of policy quality.
- **Publishable as a parity / correctness claim.** "Our exported model's action output matches the source policy to within 1e-4 on N validation frames" is a legitimate engineering claim. This is how quantization papers, ONNX-export papers, and distillation papers (pi-Flow, OneDP) frame their numbers.
- Community "action token accuracy >95%" as an OpenVLA training watermark is reported in the OpenVLA paper but framed as a training-progress signal, not a benchmark claim.

### Tier: 2 — supplementary
- Tier 1 only as a *parity* metric (export fidelity vs. upstream). As a VLA quality claim, demoted to Tier 2.

---

## 2. Behavior cloning MSE on held-out split

### Definition
The textbook supervised-learning generalization metric: L_val = E_{(o,s,a*) ~ D_val} [L(π_θ(o,s), a*)] with L = MSE / L1 / NLL.

### Who uses it
- Diffusion Policy (Chi et al., RSS 2023) reports validation loss but the paper explicitly uses task success as the headline.
- ACT (Zhao et al., RSS 2023) — L1 reconstruction loss reported during training.
- GR00T open-loop eval — same.
- LeRobot — doesn't track by default.

### Compute cost
- Same as #1. Near-zero.

### Data requirements
- Held-out split of the same dataset. Any LeRobot dataset or OpenX TFDS.

### Publishability
- **Not publishable alone.** Too close to "training loss." Reviewers ask "OK, but does it succeed?"
- See Barannikov 2024: val loss sometimes *anti-correlated* with task success due to distributional shift at rollout time (compounding errors).

### Tier: 2 — supplementary (same caveats as #1)

---

## 3. Action chunk matching (multi-step horizon L1/L2)

### Definition
For a chunk of length H (typically 50 for ACT / 16 for π0 / 4-10 for flow-matching VLAs):
- **Chunk MSE**: (1/N) Σ (1/H) Σ_{h=1..H} || π_θ(o_i, s_i)[h] − a_{i+h}* ||_2^2
- Variant: per-horizon weighting (later-horizon actions expected to have higher error due to open-loop drift).

### Who uses it
- ACT paper (Zhao et al.) — reports chunk reconstruction L1.
- "Reinforcement Learning with Action Chunking" (arXiv 2507.07969) — uses chunk L1 as primary training signal; does NOT claim chunk-L1 is a headline benchmark.
- ALOE (arXiv 2602.12691) — uses action-level (chunk-level) off-policy evaluation with temporal-difference bootstrapping, explicitly framing this as a research-grade OPE technique.

### Compute cost
- ~H× the cost of per-step matching. Still minutes on A10G for LIBERO-scale.

### Data requirements
- Same as #1 but needs access to next-H-step ground-truth actions from the dataset, not just single-step.

### Publishability
- **Not publishable as a headline.** Still shares the "no correlation with task success" critique from Barannikov.
- Publishable as a training-diagnostics supplementary table.

### Tier: 2 — supplementary

---

## 4. Semantic task decomposition (sub-goal / subtask scoring)

### Definition
Decompose an episode into K named sub-goals (e.g., "approach object", "grasp", "lift", "transport", "place"). For each sub-goal k, output a binary success ŷ_k. Report per-subgoal accuracy A_k and all-subgoals accuracy A_task.

### Who uses it
- **StepEval** (Score the Steps, Not Just the Goal, arXiv 2509.19524) — proposes per-subgoal VLM-judged evaluation; *no correlation-with-task-success data provided; proposal only*.
- **RoboEval** (arXiv 2507.00435) — "stage-wise success indicators" as outcome metrics. Stage-wise scoring shows ~60% correlation with final task success across its bimanual tasks.
- Physical Intelligence (π₀ paper) — fine-grained scoring for real-world tasks: "0.5 for grasp, 0.5 for placement" for pick-and-place.

### Compute cost
- Low if using rule-based sub-goal detection (contact sensor, gripper state transitions).
- High if using VLM judge (GPT-4o @ ~$0.01/frame × hundreds of frames/episode × hundreds of episodes = $100s-$1000s).

### Data requirements
- Annotated sub-goal timestamps OR a VLM judge that can infer them from video.
- Recorded policy rollouts. *This is the catch for non-sim settings: sub-goal evaluation still requires a rollout to annotate, so you either need a real-world setup OR a sim.*

### Publishability
- **Publishable as a supplementary diagnostic.** RoboEval and StepEval are the papers establishing this category in 2025.
- Not yet a headline metric — still paired with binary success in published results.

### Tier: 2 — supplementary (and only useful if you have rollouts, which is the question we're trying to avoid)

---

## 5. Inverse dynamics consistency

### Definition
Given consecutive states (s_t, s_{t+1}), an inverse dynamics model f_inv predicts the action a_t that would drive s_t → s_{t+1}. For a VLA π_θ, compare π_θ(o_t, s_t) to f_inv(s_t, s_{t+1}). Measure L1/L2 residual.

### Who uses it
- **Predictive Inverse Dynamics Models** (Tian et al., arXiv 2412.15109) — PIDM uses inverse dynamics as a scalable training backbone, not as a VLA evaluation metric.
- **World Action Verifier** (arXiv 2604.01985) — enforces *forward-inverse* cycle consistency for world-model verification, adjacent but not a VLA benchmark.

### Compute cost
- Requires a pretrained inverse-dynamics model (adds 1-10 GPU-hours training overhead) or a closed-form analytical one (if kinematics is fully known).
- Then O(N) forward passes at eval.

### Data requirements
- Consecutive-state pairs. Any trajectory dataset provides these.

### Publishability
- **Not publishable** as a standalone VLA quality metric. Niche. Risks circularity — if f_inv is itself imperfect, comparing to π_θ just measures model-vs-model divergence.

### Tier: 3 — experimental / niche

---

## 6. Trajectory coverage / mode coverage

### Definition
Does the policy's rollout distribution cover the modes of the expert distribution? Measure via:
- **Signature-kernel entropy** (FAKTUAL, arXiv 2603.11634): Shannon or von Neumann entropy on signature-kernel eigenvalues.
- **KL / Wasserstein** between embedded policy actions and expert actions in a shared latent.
- **Precision-Recall for distributions** (adapted from GAN eval).

### Who uses it
- **FAKTUAL / Diversity You Can Actually Measure** (arXiv 2603.11634) — as a *data curation* metric, not a policy benchmark.
- Some offline-RL papers (Bellman-Wasserstein Distance, arXiv 2507.10843) use Wasserstein between policy/behavior.

### Compute cost
- Moderate. Kernel computations are O(N^2) naively; signature kernels add further cost.
- 1-10 GPU-hours for 10k-trajectory LIBERO-scale datasets.

### Data requirements
- Dataset of expert trajectories + generated policy trajectories.
- If you're not rolling out, you can't measure policy trajectory distribution — so this is fundamentally limited for non-sim settings.

### Publishability
- **Not publishable.** No established venue paper has made this the headline claim.

### Tier: 3 — experimental

---

## 7. Language-conditioned action distribution clustering

### Definition
For each instruction i, embed the policy's predicted actions and the expert's actions. Cluster by instruction. Measure whether the policy's cluster centroids align with the expert's — i.e., does the model distinguish instructions at the action level?

### Who uses it
- No published VLA-eval paper uses this as a primary metric.
- Related: CALVIN / VLABench (arXiv 2412.18194) test instruction-following via *task success* under instruction variations, not via clustering.

### Compute cost
- Low — embedding + clustering on a few thousand predictions.

### Data requirements
- A dataset with multiple instructions per scene, or instruction-paraphrased versions.

### Publishability
- **Not publishable as a headline.** At best, a supplementary analysis plot.

### Tier: 3 — experimental

---

## 8. Static image classification (vision encoder probe)

### Definition
Freeze the VLA's vision backbone. Train a linear probe on SigLIP / CLIP / ImageNet / COCO classification. Report top-1 accuracy.

### Who uses it
- **SigLIP 2** (arXiv 2502.14786) — linear probe benchmarks on ImageNet, COCO, segmentation/depth as standard vision-encoder evaluation.
- Rarely done for VLAs specifically. OpenVLA, π₀, SmolVLA do not publish vision-encoder probe scores.

### Compute cost
- Linear probe training: ~30 min on a single GPU. Near-zero for evaluation.

### Data requirements
- Any classification dataset (ImageNet).

### Publishability
- **Publishable as a supplementary encoder-quality claim** but irrelevant to action-quality. Reviewers will ask why this predicts robot task success (it doesn't).

### Tier: 3 — niche (relevant only for vision-encoder ablations)

---

## 9. Retrieval-based scoring (nearest-neighbor trajectories)

### Definition
Embed the policy's predicted trajectory and expert trajectories in a shared space. Report top-k retrieval accuracy — does the predicted trajectory retrieve the ground-truth expert trajectory for the same instruction?

### Who uses it
- **ExpReS-VLA** (arXiv 2511.06202) — uses cosine-similarity retrieval (top-5 cosine 0.91 vs. 0.53 random) but as part of the *method*, not as a standalone benchmark.
- Instant Policy (arXiv 2411.12633) — in-context imitation, retrieval adjacent.

### Compute cost
- Low. Embedding + k-NN over a few thousand trajectories.

### Data requirements
- Labeled trajectories (expert + predictions).

### Publishability
- **Not publishable as a VLA benchmark.** Currently framed as a training/method component.

### Tier: 3 — niche

---

## 10. Gripper-state accuracy (binary)

### Definition
For each frame, treat gripper open/close as a binary classification. Report accuracy, precision, recall, F1, or Brier score.

### Who uses it
- **Confidence Calibration in VLAs** (Zollo et al., arXiv 2507.17383) — Brier score and NLL across OpenVLA / MolmoAct / UniVLA / NORA on LIBERO. All models show monotonic relationship between task error and Brier/NLL — this is *some* offline-to-success correlation evidence, but only for the calibration dimension.
- OpenVLA paper — gripper action as a single discrete token, monitored but not the headline.

### Compute cost
- Trivial.

### Data requirements
- Labeled gripper states per frame. Available in every robot dataset.

### Publishability
- **Publishable as a supplementary per-dimension breakdown** — common in OpenVLA-family papers. Not a headline.
- Sensitive to class imbalance (gripper is often closed/open for long stretches).

### Tier: 2 — supplementary

---

## 11. Pose reachability / workspace validity

### Definition
For each predicted action, check whether the resulting pose is reachable given the robot's kinematic constraints. Report % of frames where prediction stays within workspace.

### Who uses it
- No published VLA paper uses this as a benchmark.
- Grasp-reachability literature (e.g., `dynamic_grasping` at Columbia) uses it for grasp pre-filtering, not policy evaluation.

### Compute cost
- Trivial per-frame IK check. ~0.1s/frame × 10k frames = minutes.

### Data requirements
- Robot URDF + IK solver. No dataset required beyond the policy's predictions.

### Publishability
- **Not publishable as a headline.** Useful as a safety / sanity check diagnostic.

### Tier: 3 — niche diagnostic

---

## 12. Human preference ranking

### Definition
Show pairs of rollout videos (policy A vs. policy B) to human evaluators. Collect pairwise preferences. Fit Bradley-Terry / Elo ratings.

### Who uses it
- **RoboArena** (Atreya et al., arXiv 2506.18123) — distributed real-world pairwise evaluations across 7 institutions, 4,284 episodes, 612 pairwise comparisons. Uses *extended Bradley-Terry with task-aware parameters* (not plain Elo). This is the state-of-the-art 2025 benchmark for real-world generalist VLA evaluation.
- **Chatbot Arena / GenAI Arena** — the philosophical prior.

### Compute cost
- Human time is the bottleneck. Crowd-sourcing makes it parallelizable but still expensive: ~$5-50 per pairwise comparison.
- Compute per trial: one rollout (which requires either real hardware or sim — so not actually *non-sim* unless you have a real robot).

### Data requirements
- Rollouts of the policies to be compared. *Fundamentally requires rollout, so not a pure-offline method.*
- Human evaluators.

### Publishability
- **Publishable as a headline for multi-policy comparison** (RoboArena pattern). Excellent for a leaderboard or a "we beat N baselines" claim.
- Not cheap. Requires a hardware setup OR an existing video corpus.

### Tier: 1 — widely accepted (but expensive and needs rollouts)

---

## 13. LLM / VLM judge (GPT-4V watching videos)

### Definition
Record rollout videos. Prompt a VLM (GPT-4o, Gemini-2.0, Claude 3.5, Qwen3-VL) to: (a) classify binary success, (b) score sub-goals, (c) rate trajectory quality.

### Who uses it
- **WorldEval** (arXiv 2505.19017) — Gemini-2.0 as success classifier over world-model-generated videos. Pearson r=0.942 with real-world success (strong).
- **StepEval** (arXiv 2509.19524) — GPT-4o / GLM-4.5V as subgoal judge.
- **RoboArena Infinity / RobotArena∞** (arXiv 2510.23571) — VLM judge over real-to-sim generated rollouts.
- Caveat: **Score the Steps** notes "VLMs can produce incorrect judgments, particularly for fine-grained gripper-object interactions in wrist-view images or scenes with severe occlusion" and "VLM suffered from severe hallucination."

### Compute cost
- Inference-only: $0.01–$0.05 per frame for frontier VLMs × 100s of frames × 100s of episodes = $100s-$1000s per benchmark.
- Plus a video corpus, which requires rollouts.

### Data requirements
- Video rollouts (not offline-friendly unless someone else recorded them).
- VLM API access or self-hosted VLM.

### Publishability
- **Emerging as publishable** — WorldEval, StepEval, AutoEval all peer-reviewed in 2025-2026. But reviewers still ask for correlation-with-real data.
- VLM hallucination is a well-known caveat.

### Tier: 2 — supplementary / emerging-headline (requires rollouts)

---

## 14. Embedding cosine on generated trajectories

### Definition
Embed rollout trajectories (sequence of actions + images) into a learned space; compare policy and expert via cosine similarity of embeddings. Analogous to FID for images.

### Who uses it
- **FID** (Fréchet Inception Distance) repurposed by WorldEval for policy-video comparison. WorldEval identifies FID as "a lightweight quantitative metric for rapid comparative policy evaluation."
- Diffusion-policy literature uses it intermittently.

### Compute cost
- Embedding cost dominates (~seconds/trajectory on a pretrained video encoder).

### Data requirements
- Rollout videos (so again, not pure offline).
- A pretrained video/trajectory encoder.

### Publishability
- **Publishable as supplementary** (FID has a precedent).
- Not accepted as a standalone VLA-quality claim.

### Tier: 2 — supplementary

---

## 15. Action-token accuracy (OpenVLA / RT-2 discrete-token pattern)

### Definition
For models that discretize actions into tokens (RT-2, OpenVLA): report per-token top-1 accuracy on the held-out set. Usually bucket into 256 bins per dim.

### Who uses it
- **OpenVLA** (Kim et al., arXiv 2406.09246) — reports action-token accuracy; cites 95% as the "ready to evaluate on a robot" watermark.
- **RT-2** (Brohan et al.) — ditto.
- **FAST** (Black et al., arXiv 2501.09747) — token-prediction accuracy reported in a compressed action-token scheme.

### Compute cost
- Trivial.

### Data requirements
- Held-out dataset with matching action tokenizer.
- Only applies to discrete-token models. *Does not apply to SmolVLA, π0, π0.5, or GR00T*, which are all continuous-action / flow-matching.

### Publishability
- **Widely used as a training diagnostic** in the discrete-token VLA literature. Still not treated as a replacement for task success.

### Tier: 2 — supplementary (and inapplicable to Reflex's main model zoo)

---

## 16. Calibration metrics (ECE, Brier, NLL) on action confidence

### Definition
For a model that outputs a distribution over actions (Gaussian, categorical, etc):
- **ECE**: bucket predictions by confidence; measure |accuracy − confidence| per bucket.
- **Brier score**: Σ (p_i − y_i)^2.
- **NLL**: − log p(y | x).

### Who uses it
- **Confidence Calibration in VLAs** (Zollo et al., arXiv 2507.17383) — across OpenVLA / MolmoAct / UniVLA / NORA on LIBERO task suites; shows monotonic relationship between task error and Brier/NLL.
- **Shifting Uncertainty to Critical Moments** (arXiv 2603.18342) — uses calibration-adjacent metrics.

### Compute cost
- Trivial if the model outputs distributions.

### Data requirements
- Held-out dataset + a probabilistic model (SmolVLA's flow-matching head is not obviously a calibration surface; diffusion policies need tweaking).

### Publishability
- **Publishable as a legitimate headline in the "trust / calibration" subfield** (Zollo 2025 is published).
- Not a replacement for task success, but *the one non-sim metric with published correlation to task success*.

### Tier: 2 — supplementary, but trending toward 1 in the calibration subfield

---

## 17. Parity / export-correctness metrics (max_diff, round-trip)

### Definition
For a deployment pipeline (exporter, quantizer, distiller): compare the output of the transformed model vs. the source model on a fixed input batch. Report:
- **max_diff**: max |π_θ_exported(x) − π_θ_source(x)|
- **mean_diff / L1 / L2**
- **max relative diff**
- **cosine similarity** between output vectors

### Who uses it
- Reflex itself (per `project_reflex_vla.md`: "SmolVLA max_diff 6e-08 to 3e-06 after ONNX export").
- PaliGemma2 ONNX community.
- Every quantization paper (GGUF, GPTQ, AWQ, etc.).
- veRL log-prob divergence study referenced in the Reflex corpus.

### Compute cost
- Trivial. One forward pass per sample.

### Data requirements
- A small (~100-1000 frame) validation set. Can be a mock or a real dataset.

### Publishability
- **Publishable as an engineering / systems claim.** This is what Reflex actually ships evidence for. "Our exported model matches the source to max_diff < 1e-4" is defensible at an MLSys / SysML venue.
- NOT publishable as a policy-quality claim.

### Tier: **1 — cheap, widely accepted, and honest for Reflex's wedge**

---

## 18. FID on rollout videos

### Definition
Fréchet Inception Distance between policy-rollout video frames and expert-rollout video frames, using a pretrained image feature extractor.

### Who uses it
- **WorldEval** (arXiv 2505.19017) uses FID as a "lightweight" metric on generated vs. real videos.
- Image-generation literature prior.

### Compute cost
- Low (if you have rollouts). ~1 GPU-minute per 100 video frames.

### Data requirements
- Rollout videos. *Not offline-friendly.*

### Publishability
- Supplementary.

### Tier: 2 — supplementary

---

## 19. MMRV (Mean Maximum Rank Violation) against a reference benchmark

### Definition
Given multiple policies ranked by offline metric X and a "reference" ranking (from real hardware), MMRV measures the worst-case rank-flip between them. Lower is better. Paired with Pearson correlation.

### Who uses it
- **SimplerEnv** (Li et al., arXiv 2405.05941) — proposes MMRV + Pearson as the dual metric for evaluating sim-to-real eval pipelines.
- **WorldEval** — MMRV 0.044 across 5 tasks.
- **REALM** (arXiv 2512.19562) — uses MMRV+Pearson as eval-pipeline quality.
- **Sim2Val / Luo et al.** (arXiv 2506.20553) — cross-platform MMRV.

### Compute cost
- Requires running a reference evaluation (real hardware or a trusted simulator) for N policies. Offline metric itself is cheap.

### Data requirements
- Multiple policy checkpoints + reference success rates.

### Publishability
- **Publishable as a meta-metric for evaluation-pipeline quality.** This is the go-to tool when you're arguing "my sim / world-model / offline metric tracks real success."

### Tier: 1 (meta-metric) — essential for reflex if reflex wants to argue its non-sim eval is trustworthy.

---

## 20. Robometer / learned reward model scoring

### Definition
Train a reward model on pairwise human preferences or demonstration-quality labels. At eval time, score policy rollouts with the learned reward.

### Who uses it
- **Robometer** (arXiv 2603.02115) — trajectory-comparison reward models, scales general-purpose reward functions via pairwise comparisons.
- HALO (arXiv 2508.01539) — Plackett-Luce preference reward for robot navigation.

### Compute cost
- Training the reward model: ~10-100 GPU-hours.
- Scoring: fast.

### Data requirements
- Pairwise or ranking preferences (human-collected).
- Rollouts to score.

### Publishability
- **Publishable as a research contribution in its own right** (Robometer is), but not as a pure VLA-quality benchmark yet.

### Tier: 3 — experimental

---

## 21. Cycle consistency via a world model (forward-inverse)

### Definition
Given a world model W and the policy π, check: does π produce actions that W predicts drive a scene forward consistently with goals/sub-goals? WAV (arXiv 2604.01985) formalizes this.

### Who uses it
- **World Action Verifier (WAV)** — forward-inverse asymmetry.
- **Ctrl-World** (arXiv 2510.10125) — policy-in-the-loop rollouts inside a world model.
- **1X World Model** (Evaluating Bits, not Atoms) — uses a world-model sim for policy eval.

### Compute cost
- **High**: world-model training is tens to hundreds of GPU-hours. Inference ~50 diffusion steps per evaluation.

### Data requirements
- A pretrained world model (WAN 2.1 14B for WorldEval, custom for others).

### Publishability
- **Emerging-headline**: WorldEval achieved r=0.942 with real-world success, strong signal.
- But — is this actually "non-sim"? It replaces physics sim with a learned video sim, which the paper positions as superior. Still effectively a simulator.

### Tier: 2 — emerging, but not "non-sim" in the spirit of the ask (just a different sim).

---

## 22. Self-consistency / temporal consistency under stochastic forward passes

### Definition
For a stochastic policy (diffusion, flow-matching), run the same (o, s) through the model N times and measure variance of the predicted action. High variance at an unambiguous state = under-trained / poorly calibrated.

### Who uses it
- Diffusion-Policy literature reports but doesn't publish as a headline.
- Some calibration papers (arXiv 2603.18342) use this for uncertainty estimation.

### Compute cost
- N× forward passes per eval point.

### Data requirements
- Just the held-out split.

### Publishability
- Supplementary / niche.

### Tier: 3 — experimental

---

## 23. Latency / throughput / Hz

### Definition
Measure wall-clock inference time per action, per chunk, per batch. Report p50, p99.

### Who uses it
- **VLA-Perf** (arXiv 2602.18397) — explicit inference-benchmark paper.
- **Dexmal realtime-vla** (arXiv 2510.26742) — coins the "3-5 FPS vs. 20-30 Hz needed" framing.
- Every deployment paper.

### Compute cost
- Trivial.

### Data requirements
- Trivial.

### Publishability
- **Publishable as a systems / deployment claim.** Standard.
- Not a "policy quality" metric.

### Tier: **1 — standard systems metric, always publishable alongside task success**

---

## 24. Offline policy evaluation via importance sampling / Q-learning (OPE)

### Definition
Classical RL offline-policy-evaluation: importance sampling, per-step Q-value estimation, doubly-robust estimators. Produce a scalar estimate of expected return without running the policy.

### Who uses it
- **ALOE** (arXiv 2602.12691) — action-level OPE via chunking-based temporal-difference bootstrapping for VLA post-training.
- Offline RL literature (Wasserstein regularization, Bellman-Wasserstein distance, etc.).
- Historical: Precup, Jiang, Dudík OPE lineage.

### Compute cost
- Moderate. Requires training a Q function (10s of GPU-hours).

### Data requirements
- A reward-labeled dataset (the hard part — most VLA datasets have binary success at end only, not dense reward).

### Publishability
- **Publishable in the offline-RL subfield.** ALOE is at arXiv (2026 submission). Not yet mainstream as a VLA benchmark.

### Tier: 3 — experimental / niche

---

## 25. Robustness under perturbation (LIBERO-PRO / LIBERO-Plus / Eva-VLA)

### Definition
Apply controlled perturbations (object poses, lighting, adversarial patches, instruction paraphrase) and measure success-rate degradation.

### Who uses it
- **LIBERO-PRO** (arXiv 2510.03827) — exposes the memorization flaw; OpenVLA / π0 / π0.5 drop from >90% to ~0% under mild perturbations.
- **LIBERO-Plus** (arXiv 2510.13626).
- **Eva-VLA** (arXiv 2509.18953) — continuous-optimization over physical perturbations; OpenVLA fails >60% under all variation categories.

### Compute cost
- Same as the base benchmark (still requires a sim).

### Data requirements
- The sim + perturbation parameterization.

### Publishability
- **Publishable and rising fast.** LIBERO-PRO specifically argues the baseline LIBERO number is nearly meaningless. But it still needs sim.

### Tier: 1 (for the robustness subfield) — but NOT non-sim.

---

## 26. Dataset-centric diagnostic metrics (dataset-quality as a policy-quality proxy)

### Definition
Metrics that describe the dataset used (entropy, coverage, curation quality). The claim: a model trained on a better dataset is a better model.

### Who uses it
- **FAKTUAL** (arXiv 2603.11634) — signature-kernel entropy for dataset curation.
- **Diversity You Can Actually Measure** — same paper.

### Compute cost
- Moderate.

### Data requirements
- Dataset, no policy needed.

### Publishability
- Only as a dataset-quality claim, not a policy-quality claim.

### Tier: 3 — niche (not a policy benchmark at all)

---

## Summary Table

| # | Technique | Tier | Cost | Needs rollouts? | Publishable as headline? | Notes |
|---|-----------|-----:|-----:|----------------:|:------------------------:|-------|
| 17 | Parity / export-correctness (max_diff, round-trip) | **1** | trivial | no | yes (MLSys/DevEx) | Reflex's actual product signal |
| 23 | Latency / throughput / Hz | **1** | trivial | no | yes (systems) | Always reported alongside success |
| 19 | MMRV + Pearson vs. reference | **1** (meta) | moderate | partially | yes (as meta-metric) | Required if arguing offline eval is trustworthy |
| 12 | Human preference ranking (RoboArena) | **1** | high $ | yes | yes | State-of-art real-world eval 2025 |
| 25 | Robustness under perturbation (LIBERO-PRO) | **1** (sub) | sim required | yes | yes | Not non-sim |
| 1 | Action MSE / L1 / cosine | 2 | trivial | no | no (as quality) / yes (as parity) | Demoted post-Barannikov |
| 2 | BC validation loss | 2 | trivial | no | no | Same as #1 |
| 3 | Action chunk matching | 2 | low | no | no | Same family |
| 4 | Sub-goal / task-decomposition scoring | 2 | low-high | yes | partially (RoboEval, StepEval) | Emerging |
| 10 | Gripper-state accuracy | 2 | trivial | no | supplementary | Never a headline |
| 13 | VLM / LLM judge | 2 | $$ | yes | emerging | WorldEval r=0.942 |
| 14 | FID on rollouts | 2 | low | yes | supplementary | |
| 15 | Action-token accuracy | 2 | trivial | no | supplementary (discrete only) | N/A for SmolVLA/π0 |
| 16 | Calibration (ECE, Brier, NLL) | 2 | trivial | no | yes (calibration subfield) | Only non-sim metric with *published* monotonic link to success (Zollo 2025) |
| 18 | FID | 2 | low | yes | supplementary | Variant of 14 |
| 21 | World-model eval (WorldEval) | 2 | high | no (sim-adjacent) | emerging | Effectively a neural sim |
| 5 | Inverse dynamics consistency | 3 | moderate | no | no | Circular-risk |
| 6 | Trajectory coverage / mode coverage | 3 | moderate | yes | no | Niche |
| 7 | Language-conditioned clustering | 3 | low | no | no | Exploratory |
| 8 | Vision-encoder linear probe | 3 | low | no | only for encoder-quality | Not action quality |
| 9 | Retrieval scoring | 3 | low | no | method-only | |
| 11 | Workspace / reachability validity | 3 | trivial | no | no | Safety diagnostic |
| 20 | Learned reward (Robometer) | 3 | high | yes | research claim only | |
| 22 | Stochastic-forward-pass self-consistency | 3 | low | no | no | |
| 24 | OPE (importance sampling / Q-learning) | 3 | moderate | no | niche | ALOE |
| 26 | Dataset-centric diagnostics | 3 | low | no | not a policy metric | FAKTUAL |
| 0 | "val loss == quality" | **0** | trivial | no | **deprecated** | Barannikov 2024 broke this |

---

## Answer to the key question

> **Is "99.9% action cosine similarity with lerobot's expert policy over LIBERO-10 held-out frames" publishable / credible?**

**Short answer: No — not as a VLA-quality claim. Yes — as a parity / export-correctness claim, which is what Reflex's product actually sells.**

The Barannikov/LeRobot finding that validation loss is *uncorrelated* (and sometimes anti-correlated) with task success is now community-accepted prior — it circulates via the HF blog, the LeRobot GH issues, and implicitly in every paper (SmolVLA, π0, OpenVLA, GR00T) that refuses to publish anything but task success. A reviewer seeing "99.9% action cosine" as the headline claim will either reject outright ("show me LIBERO success") or ask the follow-up "and how well does this track task success on a held-out sim?" — at which point you've reintroduced the sim stack you were trying to avoid.

What you *can* publish:

1. **Export / parity framing (MLSys style):** "Reflex's export preserves action output within max_diff 3e-6 and mean-cosine-similarity 0.9999 across 50k LIBERO-10 held-out frames, with a 3.5× speedup, on three VLA families." This is true, defensible, useful. Reflex's actual corpus (max_diff 6e-08 to 3e-06) is already at this bar.
2. **Calibration framing (after Zollo 2025):** "Reflex's exported models retain 99% of the source policy's calibration (ECE, Brier) across LIBERO-held-out." This is a Tier-2 metric with *published* monotonic link to task success.
3. **Latency + export-quality pair:** "3-5 FPS → 20 Hz at max_diff < 1e-4" is the clean Reflex story.

What you cannot honestly publish:

- "Our VLA achieves 99.9% action accuracy" as evidence "our VLA is good."
- A LIBERO-task-success-style ranking based on action-matching alone.

## Recommended "Reflex non-sim benchmark suite" (3 metrics, cheap + defensible)

Ship all three together as reflex's "offline quality suite":

1. **Parity: `reflex validate --parity`** — report `max_diff`, `mean_cosine_similarity`, `p99_abs_diff` of exported-vs-source policy across a held-out LIBERO-10 / DROID validation split (5k-50k frames). Tier 1, engineering-honest, already shipping. This is the claim Reflex can defend at any venue.

2. **Calibration delta: `reflex validate --calibration`** — ECE, Brier, NLL delta between exported and source models. Cites Zollo et al. 2507.17383. Demonstrates "exported model has the same trust profile as source." Tier 2 but monotonically related to task success (the published link).

3. **Latency / throughput: `reflex bench --latency`** — p50 / p99 wall-clock on target hardware (CPU, A10G, Orin, Thor). Cites VLA-Perf 2602.18397 and Dexmal 2510.26742. Always publishable.

None of these three requires a sim. All three are cheap (total ~30 GPU-minutes for the full LIBERO-10 held-out split + calibration + latency bench). All three are honest about what they measure. None of them claims the policy is *good* — they claim Reflex doesn't break the policy, which is exactly Reflex's wedge.

You still want ONE real task-success number to land any VLA paper — that's the 0% LIBERO-10 problem being hunted separately. But the *parity suite above* is publishable TODAY, at an MLSys / DevEx venue (not at RSS/CoRL as a policy paper), and it cleanly separates "Reflex preserves fidelity" (shipped, defensible) from "our policy is good" (not Reflex's claim).

---

## Citations

- Barannikov, M. (2024). *Is using a validation set useful for end-to-end learning in robotics?* Hugging Face Blog. https://huggingface.co/blog/m1b/validation-loss-robotics
- Kim, M. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv 2406.09246.
- Kim, M. et al. (2025). *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OFT).* arXiv 2502.19645.
- Shukor, M. et al. (2025). *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.* arXiv 2506.01844.
- Black, K. et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv 2410.24164.
- Sikka, H. et al. (2024). *Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks.* arXiv 2411.05821 (AMSE, NAMSE, Completion Rate).
- Zhao, Y. et al. (2025). *RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation.* arXiv 2507.00435. (11 behavioral metrics including Joint/Cartesian Path Length, Joint/Cartesian Jerk, Height Discrepancy, Velocity Divergence, etc. Behavioral metrics correlate with success in 59.4% of task-metric pairs.)
- Atreya, P. et al. (2025). *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies.* arXiv 2506.18123. (4,284 episodes, 612 pairwise comparisons, extended Bradley-Terry.)
- Zollo, T. et al. (2025). *Confidence Calibration in Vision-Language-Action Models.* arXiv 2507.17383. (Brier, NLL, ECE across OpenVLA/MolmoAct/UniVLA/NORA; monotonic with task error.)
- Wu, Y. et al. (2025). *WorldEval: World Model as Real-World Robot Policies Evaluator.* arXiv 2505.19017. (Pearson r=0.942 with real success; 50-diffusion-step FID eval.)
- Park, K. et al. (2025). *AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World.* arXiv 2503.24278 (CoRL 2025).
- Li, X. et al. (2024). *SimplerEnv / Evaluating Real-World Robot Manipulation Policies in Simulation.* arXiv 2405.05941 (CoRL 2024). (MMRV + Pearson framework.)
- Zheng, X. et al. (2025). *LIBERO-PRO: Towards Robust and Fair Evaluation of VLA Models Beyond Memorization.* arXiv 2510.03827. (OpenVLA/π0/π0.5 drop from 90%+ to ~0% under mild perturbations.)
- Liu, H. et al. (2025). *Eva-VLA: Evaluating VLA Models' Robustness Under Real-World Physical Variations.* arXiv 2509.18953. (>60% failure on OpenVLA under continuous-optimization perturbations.)
- Tian, X. et al. (2024). *Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation.* arXiv 2412.15109.
- Yan, Q. et al. (2026). *ALOE: Action-Level Off-Policy Evaluation for VLA Post-Training.* arXiv 2602.12691.
- Chen, Z. et al. (2026). *World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry.* arXiv 2604.01985.
- Qin, S. et al. (2025). *Score the Steps, Not Just the Goal (StepEval): VLM-Based Subgoal Evaluation for Robotic Manipulation.* arXiv 2509.19524.
- Park, Y. et al. (2025). *Reinforcement Learning with Action Chunking.* arXiv 2507.07969.
- Tschannen, M. et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.* arXiv 2502.14786.
- Wei, J. et al. (2025). *How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf.* arXiv 2602.18397.
- Lin, Y. et al. (2025). *Running VLAs at Real-time Speed (Dexmal).* arXiv 2510.26742.
- Atreya, R. (2025). *AllenAI vla-eval: A Unified Evaluation Harness for VLA Models.* arXiv 2603.13966. (13 sim benchmarks; supported metrics "limited to task success rate.")
- NVIDIA (2026). *Isaac GR00T N1.6.* https://github.com/NVIDIA/Isaac-GR00T. (open-loop eval: MSE action-matching, 5-6% run-to-run variance.)
- Hugging Face LeRobot GH issue #2853: *Question about loss validation and VLA evaluation in LeRobot.* https://github.com/huggingface/lerobot/issues/2853. (LeRobot does not natively track validation loss; consistent val loss coexists with systematic inference failure.)
- Ando, K. et al. (2025). *Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets (FAKTUAL).* arXiv 2603.11634.
- Luo, Y. et al. (2025). *Sim2Val: Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation.* arXiv 2506.20553.
- Pan, Z. et al. (2025). *REALM: A Real-to-Sim Validated Benchmark for Generalization in Robotic Manipulation.* arXiv 2512.19562.
- Ma, Y. et al. (2025). *Trustworthy Evaluation of Robotic Manipulation: A New Benchmark and AutoEval Methods.* arXiv 2601.18723. (99.6% policy-vs-teleoperation video discrimination.)
- Zhang, Y. et al. (2025). *ExpReS-VLA: Specializing VLA Models Through Experience Replay and Retrieval.* arXiv 2511.06202.
- 1X Tech (2024). *1X World Model: Evaluating Bits, not Atoms.* https://www.1x.tech/1x-world-model.pdf.
- Reflex internal: `vla_eval_integration.md`, `papers_referenced.md`, veRL log-prob divergence study reference.
