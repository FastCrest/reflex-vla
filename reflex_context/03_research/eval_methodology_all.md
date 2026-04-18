# VLA Evaluation Methodologies — Comprehensive Enumeration (2025-2026)

Every evaluation methodology surfaced in VLA literature through April 2026. For each: formal definition, origin, compute cost, adoption, publishability, and tier ranking by signal-per-hour-of-engineering. Ends with a recommended default battery for a new VLA deployment project (Reflex).

Scope note: methodologies are grouped into (1) sim task-success family, (2) offline / dataset metrics, (3) runtime / systems metrics, (4) generalization & robustness axes, (5) real-robot methodologies, (6) learned-evaluator / VLM-judge methodologies, (7) embodied / motion-quality metrics. Within each group, tier ranks are explicit.

---

## 1. Sim Task Success Rate (SR) — binary per-episode

### Definition
For a given suite of N tasks × K episodes, Success Rate = (# episodes that reach success condition) / (N × K). Success condition is env-defined (e.g., LIBERO: object at target, BDDL predicate satisfied; SimplerEnv: pick-up-and-place completion). Aggregate is a scalar percentage per task suite, often further averaged across suites.

### Origin paper / lab
- LIBERO (Liu, Zhu et al., NeurIPS 2023) crystallized the four-task-suite matrix (Spatial / Object / Goal / Long) as the dominant BC benchmark.
- CALVIN (Mees et al., RAL 2022) for language-conditioned long-horizon chains.
- SimplerEnv (Li et al., CoRL 2024) introduced two-sub-protocol structure: visual-matching (minimize real↔sim appearance gap) and variant-aggregation (lighting/distractor/camera-pose variants). Simpler is the de-facto sim-for-real proxy.
- Meta-World, RoboCasa (Nasiriany et al., 2024), DexMimicGen — parallel sim benchmarks, same SR metric.

### Compute cost
Per episode: ~30s–5min of sim wall-clock depending on suite. LIBERO-10 @ 150 steps ≈ 3 min/episode on A10G; LIBERO full (4 suites × 10 tasks × 50 eps) ≈ dozens of hours. SimplerEnv similar. CALVIN 5-step chains are slower. **Cost is dominated by sim step cost × episode-count, not inference**. Reflex LIBERO-10 × 1 ep ≈ 30 min; × 10 eps ≈ 5 hours on A10G (per `vla_eval_integration.md` § "episodes_per_task").

### Who reports it
Every VLA paper. SmolVLA paper (arXiv 2506.01844): LIBERO 87.3% @ 0.45B, 88.75% @ 2.25B; Meta-World 57.3%/68.24%. OpenVLA-OFT (openvla-oft.github.io): 97.1% LIBERO avg. Dream-VLA: 60.5% SimplerEnv. pi0, pi0-FAST, GR00T N1/N1.5 all report LIBERO + SimplerEnv + their chosen real-robot. RT-2 used 6,000 trial counts. GR00T N1.5 reports 98.8% on Unitree G1 fruit-placement post-train.

### Publishability
**Yes — the unambiguous headline metric.** No VLA paper publishes without it. LIBERO + SimplerEnv aggregate is the current de-facto pair.

### Tier: 1 (already-established, cheap-ish, reflex has wired it)
The vla-eval harness + ReflexVlaEvalAdapter produces exactly this number. The Reflex LIBERO-10 pipeline is plumbed end-to-end; only remaining work is the numerical-correctness fix on the expert (per `direct_torch_export_viability.md`). Zero incremental engineering once numbers land.

---

## 2. Partial-Progress / Subgoal Success Rate (Subgoal-SR)

### Definition
For trajectories that fail the terminal success condition, decompose task into subgoals (grasp → lift → transport → release) and report per-subgoal binary success. Aggregate is a vector of subgoal SRs, optionally weighted into a scalar.

### Origin paper / lab
- BEHAVIOR-1K / BEHAVIOR Challenge 2025 (Stanford): q-score = weighted combination of full-success + partial-subgoal credit.
- Score-the-Steps-Not-Just-the-Goal (StepEval, arXiv 2509.19524): per-subgoal reporting via VLM judge from recorded video; uses cost-aware VLM calls.
- RoboEval (arXiv 2507.00435): tiered stage decomposition (push / grasp / hold / rotate / lift).

### Compute cost
Same episode collection as SR but adds either (a) BDDL / sim-predicate evaluation at key steps (free, on top of sim), or (b) a VLM judge pass over recorded frames (~$0.01–$0.10 per episode via hosted VLM; ~15s per episode on a local 7B VLM). For Reflex: free if sim supplies subgoal predicates (LIBERO does for some tasks via BDDL), cheap if VLM-judged offline.

### Who reports it
BEHAVIOR Challenge winning submission (q-score 0.26 Robot-Learning-Collective 2025). RoboEval. VLABench long-horizon split. Emerging standard in 2026 critique literature ("success-rate-alone hides compound-failure modes"); StepEval argues it should become routine.

### Publishability
**Yes, and increasingly expected for long-horizon claims.** Pure SR is critiqued as opaque ("which subtask broke?"). Reporting subgoal-SR differentiates serious evaluation from benchmark-gaming.

### Tier: 1 (cheap if sim supplies predicates; Reflex already on LIBERO-10 long-horizon)
LIBERO-10 tasks are explicitly long-horizon; the BDDL predicates are already encoded. Extracting subgoal-level traces from vla-eval output is a low-effort add. Publishable as a differentiator.

---

## 3. Action-Chunk Cosine Similarity / Trajectory Replay L2

### Definition
Given a held-out dataset of expert (image, state, language) → action trajectories, run the policy open-loop (teacher-forced observations) and compare predicted action chunks against ground-truth chunks. Report per-step cosine similarity, per-step L2, per-chunk RMSE, or full-trajectory MSE. Variants: cos(expert, student) at each denoise step (flow-matching policies), per-dim MSE, per-joint L2.

### Origin paper / lab
- Standard behavior cloning diagnostic from imitation-learning literature (Pomerleau, Bojarski et al.) — predates VLAs.
- Adapted for action-chunking in ACT (Zhao et al., 2023) and Diffusion Policy (Chi et al., 2023).
- Adaptive-chunking literature (arXiv 2510.12392): uses chunk-to-chunk cosine to decide whether to re-plan.
- Reshaping-action-error (arXiv 2602.04228): critiques MSE's Gaussian assumption, proposes MEE (minimum error entropy).

### Compute cost
**Dramatically cheaper than sim SR**: no environment stepping. Per-episode it's just N forward passes on the policy. Typical: 100–1000 trajectories × 50–100 steps × 1 forward pass ≈ minutes on a single GPU. Dataset sits in memory. Reflex already does this implicitly via `reflex validate` (round-trip parity @ 1e-4 cos=1.0 against LeRobot).

### Who reports it
Not typically a headline number, but *always* in the appendix. LeRobot's `validation/` dir reports MSE per checkpoint. Diffusion Policy papers always show reconstruction MSE. Critical caveat from HuggingFace/m1b validation-loss-robotics blog (2024): **MSE does not correlate with success rate** past a certain training phase — validation MSE can increase 40K→60K steps while SR improves. This is why papers lead with SR, not MSE.

### Publishability
**As a correctness/parity claim: yes.** As a headline of model *quality*: no. Reflex's cos=1.0 vs LeRobot is publishable as an *export-fidelity* claim ("Reflex export preserves 1.0 cosine parity with the original PyTorch policy"); it is NOT a claim about model task performance. The distinction is load-bearing: never mix them in marketing copy.

### Tier: 1 (for export parity / reflex-specific correctness headline)
This is Reflex's cheapest defensible win. Ship cos=1.0 / MSE<1e-4 as the *export-correctness* banner. It separates us from competitors whose ONNX export silently drifts. But Tier 3 for claiming model-quality — don't position it as a headline for that.

---

## 4. Behavior-Cloning Action MSE on Held-Out Test Split

### Definition
Strict formal cousin of #3: on a fixed held-out dataset split (typically 10–20% of an Open-X / LIBERO / DROID dataset), compute Σ (a_pred − a_gt)² / N averaged across all steps. Typically reported per-action-dim and aggregated.

### Origin paper / lab
- Classical BC evaluation. See Octo (arXiv 2405.12213) for a canonical contemporary report.
- Every LeRobot policy checkpoint card includes this.

### Compute cost
Same as #3. Offline. Single-digit minutes per model on a dataset snapshot.

### Who reports it
Octo, Diffusion Policy, ACT, most LeRobot-hosted VLAs (under validation loss). Rarely the headline for VLA papers — kept in appendices because of the divergence-from-SR issue cited above.

### Publishability
**Appendix-grade, not headline-grade.** Reviewers expect it as a sanity check. Leading with it is seen as "hiding behind easy metrics."

### Tier: 2 (cheap, but not publishable as headline; keep as internal sanity check)
Reflex should track it per-checkpoint to catch training-pipeline regressions. Do not put it on a marketing page.

---

## 5. Language-Conditioned Success — Task-Description Coverage

### Definition
Holding other conditions fixed, measure SR across (a) paraphrased instructions ("pick up the red block" / "grab the crimson cube") and (b) held-out (novel) instruction wordings. The delta between seen-phrasing SR and novel-phrasing SR is the language-generalization score.

### Origin paper / lab
- RT-2 (Brohan et al., 2023): formalized novel-instruction evaluation at scale (6,000 trials including unseen commands).
- CALVIN (Mees et al.): chains of 5 distinct language directives; average-chain-length metric.
- LIBERO-Para (arXiv 2603.28301): specifically diagnostic for paraphrase robustness.
- STAR-Gen (arXiv 2503.01238): language perturbations are one of three primary axes.

### Compute cost
Extra sim rollouts with modified instructions. If you have a base LIBERO run, language variants are ~2× the cost (re-run with new instructions). LIBERO-Plus language split is ~same cost per dimension as vanilla LIBERO. LIBERO-Para is ~1000-2000 extra episodes per model.

### Who reports it
RT-2, pi0 "in the wild" report, OpenVLA, GR00T. Rising expectation as of 2026. LIBERO-Plus finding: *most VLAs ignore language almost entirely* — models are insensitive to language variations. This finding is itself a publishable empirical result.

### Publishability
**Yes, especially as a second-order differentiator.** "Works with paraphrases" is a current VLA weakness and therefore a publishable delta if you beat baselines.

### Tier: 2 (moderately expensive; Reflex doesn't need it for v0.2 headline but will for generalization claims)
If Reflex only claims export-fidelity + LIBERO SR, skip. If Reflex ever claims generalization, mandatory.

---

## 6. Latency-to-First-Action (TTFA) / Action-Generation Latency

### Definition
Wall-clock time from receiving a new observation to emitting the first action (or first action-chunk). For chunked policies: time to produce the chunk. Broken down by: vision encoder latency, VLM prefill latency, action-expert denoise latency, tokenizer/detokenizer latency. Often reported as median + p95 + p99.

### Origin paper / lab
- VLA-Perf (arXiv 2602.18397): "first VLA inference benchmark," introduced the VLA-specific latency decomposition (vision compute-bound vs action memory-bound). Coined "memory-bound on Thor" phrase.
- Characterizing-VLA (arXiv 2603.02271): quantified "action generation = 75% of latency" across VLA families.
- Dexmal Real-Time VLA (arXiv 2510.26742): popularized the 3-5 FPS vs 20-30 Hz gap framing.
- FASTER (arXiv 2603.19199), BLURR (arXiv 2512.11769), VOTE (arXiv 2507.05116): all report Hz.

### Compute cost
Trivial: a timer around a forward pass. Per-model per-hardware-target you need maybe 100 warmup + 1000 timed calls for a stable median. Minutes. Can be done in parallel with other evals.

### Who reports it
pi0 (~10 Hz for 3B params at single-arm config), OpenVLA (~5 Hz @ 7B; ~3 FPS on Jetson AGX Orin INT4), SmolVLA paper reports async 2× speedup, BLURR-pi0 claims 50-60 Hz, VOTE 46 Hz w/ chunk=16, Xiaomi Robotics-0 async. Core number in every VLA launch post.

### Publishability
**Unambiguously yes.** Latency/Hz is the *other* headline number besides SR. Reviewers and customers both ask for it. VLA-Perf normalized it.

### Tier: 1 (trivial to measure, publishable, reflex already has a serve path)
Run it on desktop (cos=1.0 checkpoint) and report p50/p95/p99 per model × hardware. Do this for every supported exports in `reflex bench`.

---

## 7. Streaming Latency (p50 / p95 / p99) and Chunk-to-Chunk Gap

### Definition
Distinct from #6: measures end-to-end request latency when the policy is serving continuous observations (inter-observation time, chunk-refresh gap). p99 exposes tail-latency spikes that cause the robot to stall. Also: time-between-chunks (how long the robot waits for the next action batch in async mode).

### Origin paper / lab
- Real-Time Chunking / RTC (arXiv 2506.07339, pi / Hsu): defined the overlap-schedule; chunk request at 70% consumption.
- Async Robot Inference (HuggingFace blog 2025): canonical write-up of decoupled predict-vs-execute.
- General LLM serving literature (p50/p95/p99) — adopted wholesale.

### Compute cost
Requires a serving harness (Reflex has one: FastAPI /act + msgpack). Run a 1000-request load test. Minutes to hours depending on sample size. Per-hardware-target.

### Who reports it
pi RTC blog, Xiaomi Robotics-0 async demo, Async Robot Inference post. Becoming standard in deployment-focused papers. Rare in vanilla VLA model papers (which focus on SR + Hz).

### Publishability
**Yes for deployment/infra papers.** Reflex's positioning makes this the most discriminating infra metric — "serving p99 < X ms" is a direct customer-readable claim.

### Tier: 1 (cheap, differentiating, reflex-native)
Must-report for a deployment CLI. This is where Reflex out-differentiates model-only releases.

---

## 8. Throughput / Inference Hz

### Definition
Sustained actions-per-second (or chunks-per-second) over a steady-state rollout. Usually quoted as Hz. Related but distinct from #6: TTFA is single-shot, throughput is steady-state with chunks + KV-cache reuse.

### Origin paper / lab
- Same as #6 (VLA-Perf, Dexmal, FASTER, BLURR, VOTE).
- 10 Hz considered acceptable, 100 Hz high-performance (VLA-Perf, deepsense.ai 100g-device writeup).

### Compute cost
Same as #6; if anything cheaper because you only need a stable-state average.

### Who reports it
Everyone. OpenVLA 5 Hz, pi0 10-50 Hz, BLURR 50-60 Hz, VOTE 46 Hz, Xiaomi Robotics-0 (async). SmolVLA async 2× over sync.

### Publishability
**Yes.** Twin of #6. Both together are the standard perf plot.

### Tier: 1 (same as #6; reflex-wins-by-default if export-pipeline is correct)

---

## 9. Memory Footprint / Peak VRAM / Model Size

### Definition
Two numbers: (a) on-disk model size at shipping precision (FP16 / INT8 / INT4), (b) peak VRAM during steady-state inference (model weights + KV cache + activations + framework overhead). Sometimes + CPU RAM if VLM prefill is offloaded.

### Origin paper / lab
- Standard ML-systems metric. VLA-Perf, Efficient-VLA Survey (arXiv 2510.24795) surface it per model.
- Reflex-Context "deepsense.ai 100g device" writeup frames it as a hardware-envelope question.

### Compute cost
Nearly free. `torch.cuda.max_memory_allocated()` around the rollout. Report once per model × target.

### Who reports it
OpenVLA (7B → 16 GB VRAM min), SmolVLA (0.45B → fits on Jetson Orin Nano 8GB), pi0 (3B → Jetson AGX Orin class). Every edge-deployment paper.

### Publishability
**Yes, as a deployment claim.** "Fits in 8GB VRAM at INT4" is table-stakes for edge stories.

### Tier: 1 (trivial, reflex should report per export format)

---

## 10. Real-Robot Success Rate (Counted Trials + Video Rubric)

### Definition
Deploy on a physical robot for N trials per task. Human (or VLM) scores each trial as success/failure per a pre-registered rubric. Report SR with confidence intervals (Wilson / bootstrap). Record all trials on video.

### Origin paper / lab
- RT-2 (6,000 trials), pi0 in-the-wild (Penn PAL Lab, Pi0-Experiment-in-the-Wild site; three trials per task for consistency), OpenVLA-OFT real SO100/SO101 tasks, GR00T N1/N1.5 GR-1 humanoid, SmolVLA paper (SO100/SO101: 78.3%).

### Compute cost
**Dominant cost of VLA evaluation.** Robot time + human scorer time. Per trial: 1–3 minutes robot, often 30+ seconds scoring. 50-trials-per-task × 10 tasks × 4 models = 2000 trials = days of dedicated robot time. This is why papers frequently do only 3–10 trials/task and report wide CIs.

### Who reports it
Every serious VLA paper. SmolVLA (78.3% SO100), pi0 in-the-wild, GR00T (humanoid G1 98.8% post-train), RT-2 (6k trials), OpenVLA-OFT, X-VLA cross-embodiment.

### Publishability
**Yes, gold-standard.** But "no real-robot numbers" is a known reviewer flag.

### Tier: 2 (expensive, but the headline that matters to customers)
Reflex does not yet own a physical robot pipeline. Partner / use an AutoEval endpoint (below) instead of building in-house.

---

## 11. Distribution-Shift / OOD Robustness (Perturbation Suites)

### Definition
Controlled perturbations along named axes: camera viewpoint, lighting, background, object pose, distractor objects, initial robot state, language paraphrase, sensor noise. For each perturbation-level L, measure SR. Report SR curves across L.

### Origin paper / lab
- LIBERO-Plus (arXiv 2510.13626, 2026): 7 perturbation factors × 21 sub-dimensions × 5 difficulty levels. 10,030 tasks total.
- LIBERO-PRO (arXiv 2510.03827): 4 orthogonal perturbation types.
- Eva-VLA (arXiv 2509.18953): CMA-ES optimization over continuous perturbation spaces; reports *worst-case* failure rates. OpenVLA >90% failure on LIBERO-Long under this stress test.
- VLATest (FSE 2025): fuzzing framework for lighting + camera + object variations.
- VLA-Risk (OpenReview, 2026): physical robustness benchmark.

### Compute cost
Expensive: LIBERO-Plus is 10× the vanilla LIBERO ep count. But it's still sim, so parallelizable. LIBERO-Plus per model ≈ 5-10× vanilla LIBERO cost.

### Who reports it
OpenVLA-OFT under LIBERO-Plus (after 20K-traj training, 79.6% overall — 37.2pp camera-view improvement). LIBERO-PRO. Increasingly every "robustness-focused" paper. Baseline OpenVLA drops from 76.5% → 1.1% under camera viewpoint shift — widely cited number.

### Publishability
**Yes, rising expectation.** 2026 reviewers now push back on SR-only claims. LIBERO-Plus is likely to become the new must-cite.

### Tier: 2 (valuable, but expensive; Reflex should wait until v0.3)
v0.2 ships LIBERO + latency. v0.3 targets LIBERO-Plus.

---

## 12. Generalization Taxonomy — STAR-Gen (Visual / Semantic / Behavioral)

### Definition
Structured 13-axis framework (stargen-taxonomy.github.io). Every evaluation is a perturbation relative to a "base task" classified along Visual / Semantic / Behavioral axes, plus intersections (Visual+Behavioral, Semantic+Behavioral, Visual+Semantic+Behavioral). 55 task variations × 13 axes = 885-evaluation canonical grid.

### Origin paper / lab
STAR-Gen (Gao et al., arXiv 2503.01238): "A Taxonomy for Evaluating Generalist Robot Manipulation Policies." Published at CoRL 2025.

### Compute cost
Per axis ~2-5× vanilla LIBERO or Bridge-V2 evaluation. Full 885-eval grid on Bridge V2 / ALOHA 2 = multi-day sim run.

### Who reports it
STAR-Gen itself. X-VLA (arXiv 2510.10274) uses STAR-Gen's cross-embodiment axis. ET-VLA (arXiv 2511.01224). Still adoption-gating — not yet universal but clearly the direction of travel in 2026.

### Publishability
**Yes, for any paper claiming "generalist" capability.** For a deployment CLI like Reflex, overkill.

### Tier: 3 (overkill for reflex today; revisit if we ever claim generalization)

---

## 13. Cross-Embodiment Transfer Evaluation

### Definition
Train (or pretrain) on embodiment A, evaluate on embodiments B/C/D without further training — or with K-shot adaptation. Report SR on each embodiment; sometimes embodiment-normalized aggregate.

### Origin paper / lab
- Open X-Embodiment (arXiv 2310.08864): 22-embodiment dataset; RT-X/RT-2-X showed 3× emergent-skill improvement via cross-embodiment pretraining.
- X-VLA (arXiv 2510.10274, ICLR 2026): soft-prompted transformer; 7 platforms × 5 arm types.
- ET-VLA (arXiv 2511.01224): Synthetic Continued Pretraining + Embodied Graph-of-Thought.
- Embodiment Scaling Laws (arXiv 2505.05753): argues scale-beats-adapters; directly influenced Reflex's deprecation of `reflex adapt`.

### Compute cost
Very high: requires multiple real-robot setups or carefully curated sim-embodiment-bench.

### Who reports it
X-VLA, ET-VLA, RT-X, GR00T (cross-humanoid post-train), OpenVLA-OFT (few-shot).

### Publishability
Yes — model-capability-focused headline. Less relevant for deployment tooling.

### Tier: 3 (not a reflex wedge; too expensive to self-run; cite when model supports it)

---

## 14. Long-Horizon Multi-Step Success — Chain Length / q-score

### Definition
For multi-step task chains, report (a) average-chain-length (CALVIN), (b) q-score with partial-subgoal credit (BEHAVIOR-1K), (c) LIBERO-Long SR. All variations on "how far through a long task do you get before failing."

### Origin paper / lab
- CALVIN (RAL 2022): 5-step chains; 34 subtasks; 1000 unique chain orderings. Average-chain-length is the canonical metric.
- BEHAVIOR-1K / BEHAVIOR Challenge 2025: q-score — a weighted combination of full-success and partial-subgoal credit. Winning submission achieved q-score ~0.26.
- LIBERO-10 subset: the 10 long-horizon tasks of LIBERO.
- RoboCerebra (arXiv 2506.06677): 6× longer sequences, denser subtask annotations.
- LoLA (arXiv 2512.20166).

### Compute cost
Same order as sim SR but ep length 3-10× longer.

### Who reports it
CALVIN papers, BEHAVIOR Challenge papers, anyone targeting multi-step. Dream-VLA: 95.0% LIBERO-Long.

### Publishability
**Yes, differentiator for long-horizon claims.**

### Tier: 2 (LIBERO-Long is sub-set of what Reflex ships, so this falls out for free once LIBERO-10 works)

---

## 15. Safety Incidents / Joint-Limit Violations / Collision Count

### Definition
Count of (a) joint-position violations (JOINT_SLP), (b) joint-velocity violations (JOINT_SLS), (c) self-collisions, (d) environment collisions, (e) end-effector-force exceedances. Report per-episode and per-trial.

### Origin paper / lab
- Industrial robotics: ISO 10218 + ISO/TS 15066 (pre-VLA, but the terminology and limits come from here).
- Doosan dart-studio docs: joint-limit violation codes are canonical.
- RoboEval (arXiv 2507.00435): reports collision count + slip count + jerk as first-class metrics alongside SR.
- VLA-Risk (OpenReview 2026).

### Compute cost
Free if sim exposes collision + joint-limit signals (LIBERO / ManiSkill / RoboEval all do). Just instrument the rollout.

### Who reports it
RoboEval, VLA-Risk, safety-focused deployment papers. Rare in model-capability papers.

### Publishability
**Yes, for deployment / safety-critical papers.** Not a core VLA-quality metric but a credibility signal for "ready-for-real-deployment" claims.

### Tier: 2 (cheap; Reflex should surface it in its eval JSON)

---

## 16. Energy per Task / Inference

### Definition
Joules per episode or per successful-task. Measured via GPU RAPL + CPU RAPL + optional robot-wattage. Report (a) train-time energy, (b) inference energy per episode, (c) energy per successful task (sensitivity to SR).

### Origin paper / lab
- Neuro-Symbolic-Outperforms-VLA on Structured Long-Horizon (arXiv 2602.19260): dramatic "100× energy reduction" claim; reframed VLA-energy as a first-class concern.
- Industrial robotics power-consumption metrics (arXiv 2508.06295): robot-specific framework.
- Generic LLM-energy survey (ScienceDirect 2025): methodology borrowed.

### Compute cost
Requires GPU RAPL logs (Weights & Biases GPU-power, CPU-RAPL). Free if you instrument. Modest extra-engineering to set up the measurement harness.

### Who reports it
Neuro-Symbolic-VLA comparison paper. Emerging in green-AI literature. Not yet universal in VLA papers.

### Publishability
**Yes, for edge/mobile deployment claims.** "Jetson Orin Nano: 2.3 J per action" is a crisp customer-readable number.

### Tier: 2 (easy to add; strong differentiator for edge pitch, not needed for v0.2)

---

## 17. Human A/B Tests / Pairwise Policy Comparison

### Definition
Pairs of policies compared on the same observation / task by humans; humans pick the better of two. Aggregate via Elo / Bradley-Terry or win-rate.

### Origin paper / lab
- RoboArena (arXiv 2506.18123): distributed real-robot pairwise evaluation across 7 institutions, 600+ pairwise episodes on DROID. Core finding: crowd-sourced pairwise ranks policies more accurately than centralized eval.
- RobotArena ∞ (arXiv 2510.23571): sim version with VLM scoring + crowd-sourced preferences.

### Compute cost
Very high if human-in-loop. Crowd-sourcing reduces cost-per-comparison but adds quality-control overhead. A single 7-policy tournament is 7-choose-2 × N trials each = hundreds of hours.

### Who reports it
RoboArena, RobotArena-∞. Increasingly the aspirational gold-standard.

### Publishability
**Yes, top-tier signal.** "Ranked #1 on RoboArena" is a strong claim.

### Tier: 3 (out of scope for Reflex today; submit to RoboArena once we have a public-facing deployment)

---

## 18. Learned-Evaluator / World-Model-Based Policy Eval

### Definition
Train (or use off-the-shelf) a world-model or VLM to predict success. Policy is rolled out "in" the world model; the world-model's predicted success is the score. Or: VLM watches recorded rollout and scores rubric.

### Origin paper / lab
- WorldEval (arXiv 2505.19017): Policy2Vec video-generation-based world simulator; claims strong correlation with real-world rankings.
- WorldGym (arXiv 2506.00613): world model as env for evaluation.
- Can-VLMs-Judge-Action-Quality (arXiv 2604.08294): empirical; found VLMs marginally above chance, with two systematic biases.
- StepEval (arXiv 2509.19524): VLM subgoal judge.
- Robometer (arXiv 2603.02115): general-purpose robotic reward model from trajectory comparisons. +14% rank-correlation, +32% suboptimal-vs-successful discrimination.

### Compute cost
World-model eval: expensive to train the world model, cheap to run. VLM-as-judge: $0.01-$0.10 per episode via hosted VLM, or ~15s on a 7B local model.

### Who reports it
WorldEval, Robometer, StepEval. Rising but not yet universal.

### Publishability
**Yes with caveats** — reviewers want calibration against ground-truth (correlation with real / sim SR) shown. Action Quality Assessment paper shows naive VLM judging is unreliable.

### Tier: 3 (too much infra for Reflex to own; cite when we need scalable eval)

---

## 19. AutoEval / 24-7 Autonomous Real-Robot Evaluation

### Definition
Submit policy → queued onto a physical robot → auto-reset + auto-success-detection via trained classifiers → SR returned with videos. No human in the loop.

### Origin paper / lab
AutoEval (Zhou et al., arXiv 2503.24278, CoRL 2025). Public access: two WidowX stations, four Bridge-V2 tasks. Compared 6 generalist policies (OpenVLA, MiniVLA, +4) to human-run ground truth with close correspondence.

### Compute cost
Zero engineering on Reflex's side — submit through their dashboard. Wall-clock: per-queue depth; they advertise 24/7.

### Who reports it
AutoEval paper itself (6-policy comparison). Cited as infrastructure-milestone, adoption still emerging.

### Publishability
**Yes** — real-robot numbers without owning a robot fleet. Strong credibility signal.

### Tier: 2 (low Reflex-effort; high-leverage for real-robot claim; recommended post-v0.2)

---

## 20. RoboEval Motion-Quality Metrics (Jerk / Path Length / Slip)

### Definition
Beyond binary SR, compute:
- Joint Path Length: total angular joint distance.
- Cartesian Path Length: 3D distance traveled by end-effectors.
- Jerk: third derivative of position — motion smoothness.
- Height Discrepancy: arm-coordination failure.
- Velocity Divergence: arm-coordination metric.
- Slip Count: unintended object drops.

### Origin paper / lab
RoboEval (arXiv 2507.00435): introduced as first-class metrics for bimanual manipulation. Finding: "behavioral metrics correlate with success in >50% of task-metric pairs and remain informative even when binary SR saturates."

### Compute cost
Free if sim supplies the traces; single-digit % of overall sim-eval time.

### Who reports it
RoboEval itself. Embodied-Efficiency literature (arXiv 2603.19131) argues this should be standard.

### Publishability
**Yes, increasingly expected for deployment-realism claims.** Useful when SR saturates.

### Tier: 2 (mid-term add; Reflex benefits from reporting it to differentiate from SR-only submissions)

---

## 21. Embodied-Efficiency — Task-Completion Time / Trajectory Smoothness

### Definition
System-level metrics for robotic actuation:
- Task-completion wall-clock (time-to-success).
- Trajectory smoothness (spectral arc length, or inverse jerk).
- Cumulative joint rotation.
- Motion energy.
- End-effector path length.

### Origin paper / lab
From-Inference-Efficiency-to-Embodied-Efficiency (arXiv 2603.19131): *"Prevailing notion of efficiency (params/FLOPs/tokens/sec) does not reflect actual robotic-platform performance. Methods that reduce computation under conventional metrics often increase end-to-end execution cost or degrade motion quality despite maintaining task SR."*

### Compute cost
Same traces you collected for #20; free-with-instrumentation.

### Who reports it
Embodied-Efficiency survey. Select 2026 papers. Early-adopter metric.

### Publishability
**Yes, differentiator** — especially for "we serve fast" claims where pure Hz can be misleading.

### Tier: 2 (fits Reflex's deployment thesis; consider v0.3)

---

## 22. FLOPs / MACs / Parameters / Activation-Token Count

### Definition
Computational-complexity accounting. Params, total FLOPs per action (or per action-chunk), MACs, peak activation-token count. Often accompanied by model-flops-utilization (MFU).

### Origin paper / lab
Classical deep-learning accounting. VLA-specific treatment in VLA-Perf. Most explicit in Efficient-VLA survey (arXiv 2510.24795).

### Compute cost
Free: one-shot analysis via profilers (DeepSpeed flops-profiler, torch.profiler).

### Who reports it
VLA-Perf, Efficient-VLA-survey, DeeR-VLA (NeurIPS 2024 dynamic inference). Most papers tout params; only infra-focused papers quote FLOPs explicitly.

### Publishability
Yes, in infra / efficiency discussions. Appendix-grade for pure capability papers.

### Tier: 2 (cheap-to-add; use when framing vs competitors)

---

## 23. Mean Maximum Rank Violation (MMRV) / Sim-Real Correlation

### Definition
Given paired sim-and-real results for the same set of policies, compute rank correlation (Pearson or MMRV). MMRV: the maximum amount by which a policy's real-world rank exceeds its sim-world rank. Lower-is-better. Used to validate whether a sim benchmark is predictive of real performance.

### Origin paper / lab
SimplerEnv (CoRL 2024) popularized the framing; MMRV is explicitly defined there.

### Compute cost
Requires a parallel sim + real evaluation of several policies (N ≥ 4 for a meaningful rank). Secondary-analysis metric, built on top of #1 and #10.

### Who reports it
SimplerEnv, any paper proposing a new sim benchmark, WorldEval (as one of its correlation claims).

### Publishability
**Yes, for benchmark papers.** For a policy paper, rarely relevant.

### Tier: 3 (niche; Reflex is not in the benchmark-design business)

---

## 24. Reward Model / Trajectory-Comparison Scoring (Robometer)

### Definition
Use a learned general-purpose reward model (Robometer) to score each rollout. Report reward-rank correlation with ground truth and suboptimal-vs-successful discrimination.

### Origin paper / lab
Robometer (arXiv 2603.02115, March 2026): RBM-1M dataset; video-language-conditioned dense rewards. +14% rank correlation vs SOTA baselines, +32% relative improvement in suboptimal/successful discrimination.

### Compute cost
Per-rollout: one VLM-scale forward pass. Moderate.

### Who reports it
Robometer itself; still establishing adoption.

### Publishability
Yes in infra / RL-evaluation context. Less in vanilla VLA.

### Tier: 3 (emerging; revisit in v0.3+)

---

## 25. Chain-Length Stability / Adaptive-Chunking Signal

### Definition
For flow-matching / diffusion / chunked policies: cosine similarity between successive chunks. Stable cosine = policy is "confident"; divergent cosine = policy should re-plan. The metric doubles as a diagnostic for open-loop stability.

### Origin paper / lab
Self-Guidance-and-Adaptive-Chunking (arXiv 2510.12392).

### Compute cost
Free: cosine of two vectors per step.

### Who reports it
Adaptive-Chunking paper. Niche.

### Publishability
Diagnostic — rarely a headline. Useful in latency-vs-quality trade-off discussions.

### Tier: 3 (niche diagnostic; report as supporting plot not headline)

---

## 26. Export-Fidelity / Round-Trip Parity (Reflex-native)

### Definition
For a deployment CLI: cosine similarity (or L∞ / MSE) between pre-export (reference PyTorch) and post-export (ONNX / TensorRT / GGUF) action predictions on a fixed input grid. Target: cos ≥ 0.9999, L∞ < 1e-4. Captures numerical correctness of the export pipeline independent of model quality.

### Origin paper / lab
Not a formal academic metric. Canonical framing in `reflex validate` (Reflex's CLI; threshold tightened from 0.02 → 1e-4 per CHANGELOG BREAKING). Parallel to veRL log-prob divergence study for LLMs (training-precision ≠ serving-precision causes silent drift).

### Compute cost
Free: one pair of forward passes per random batch, no env.

### Who reports it
Reflex. ETARS notebook explicitly validates cos-sim post-export.

### Publishability
**Yes, as the Reflex-native wedge.** Reflex cos=1.0 vs LeRobot reference is the correctness claim that precedes any sim-SR claim.

### Tier: 1 (free, defensible, reflex's signature metric)

---

## 27. Automated Subgoal Scoring via VLM (StepEval)

### Definition
VLM watches recorded trajectory video; scores subgoal completion per frame or per window; aggregates to per-subgoal SR.

### Origin paper / lab
StepEval / Score-the-Steps-Not-Just-the-Goal (arXiv 2509.19524).

### Compute cost
~15s of VLM inference per episode for a 7B local VLM; ~$0.01-$0.10 per ep via hosted VLM.

### Who reports it
StepEval itself; growing.

### Publishability
Yes, when sim doesn't supply subgoal predicates.

### Tier: 3 (overkill for Reflex; LIBERO supplies predicates for free)

---

## 28. Test-Time Interactive Evaluation (VLABench interactive protocol)

### Definition
Instead of static N-episodes, evaluator proposes new tasks online based on prior failures (curriculum / adversarial). Reports SR on dynamically-generated hardest-failure tasks.

### Origin paper / lab
VLABench (ICCV 2025, arXiv 2412.18194): 100 task categories × 2000+ objects; interactive eval for VLM-in-the-loop policies; non-interactive for vanilla VLAs.

### Compute cost
Higher variance than static eval; similar order of magnitude but less reproducible.

### Who reports it
VLABench itself. Adoption emerging.

### Publishability
Yes with reviewer-skepticism-overhead (reproducibility concerns).

### Tier: 3 (adoption still rare; skip for reflex v0.2/v0.3)

---

## 29. Multi-Modal Alignment Score (VLA-FEB — CMAS / FEI)

### Definition
- CMAS (Cross-Modal Alignment Score): measure of fusion quality between vision and language features.
- FEI (Fusion Energy Index): efficiency of cross-modal fusion.
Both defined in VLA-FEB benchmark; probe feature representations rather than task success.

### Origin paper / lab
VLA-FEB (cited in multimodal-fusion review, 2025).

### Compute cost
Feature-extraction pass per input; moderate.

### Who reports it
VLA-FEB itself; still new.

### Publishability
Niche — architecturally-focused audiences only.

### Tier: 3

---

## 30. Risk / Adversarial Perturbation — VLA-Risk, Sensor Attacks

### Definition
Bound the failure rate under adversarial / worst-case perturbations (vs nominal distribution shift in #11). CMA-ES (Eva-VLA) and sensor-attack studies both quantify worst-case-not-average-case.

### Origin paper / lab
- Eva-VLA (arXiv 2509.18953): CMA-ES adversarial search.
- Sensor-Attack-VLA (ACM Workshop 2025): sensor-input perturbations.
- VLA-Risk (OpenReview 2026).

### Compute cost
Dramatically expensive: CMA-ES iterations over the perturbation space.

### Who reports it
Safety-critical deployment papers.

### Publishability
Yes for safety papers.

### Tier: 3 (out of scope for reflex)

---

# Bottom summary table

Legend: SR = Success Rate, cos = cosine similarity, p99 = 99th-percentile latency, ~  = approximate cost.

| # | Methodology | Compute Cost | Publishable Headline? | Tier | Who Reports |
|---|-------------|--------------|-----------------------|------|-------------|
| 1 | Sim Task Success (LIBERO/SimplerEnv/CALVIN SR) | High ~hrs-days | **YES — default** | 1 | All VLAs |
| 2 | Subgoal / q-score partial credit | Med ~hrs | **YES — long-horizon** | 1 | BEHAVIOR, RoboEval |
| 3 | Action-Chunk Cos / L2 vs expert | Low ~min | Export-parity YES, quality NO | 1 | Reflex, Diffusion-Policy |
| 4 | BC MSE on held-out split | Low ~min | Appendix only | 2 | Octo, LeRobot |
| 5 | Language-Conditioned SR (paraphrase / novel instr.) | Med ~hrs | YES if generalization claim | 2 | RT-2, CALVIN, LIBERO-Para |
| 6 | Latency / TTFA / action-gen latency | Trivial ~min | **YES — default** | 1 | VLA-Perf, pi0, SmolVLA |
| 7 | Streaming p50/p95/p99 + chunk gap | Trivial ~min | **YES — infra differentiator** | 1 | RTC, Async-Robot, pi |
| 8 | Throughput / Hz | Trivial ~min | **YES — default** | 1 | All infra papers |
| 9 | Memory / VRAM / model size | Trivial | YES — edge claim | 1 | Edge papers |
| 10 | Real-Robot SR with video rubric | **Very High days** | **YES — gold-standard** | 2 | RT-2, pi0, GR00T, SmolVLA |
| 11 | Distribution-Shift / OOD (LIBERO-Plus/PRO) | High ~hrs-day | YES — rising | 2 | OpenVLA-OFT, LIBERO-Plus |
| 12 | STAR-Gen 13-axis taxonomy | Very High | YES — capability | 3 | STAR-Gen, X-VLA |
| 13 | Cross-Embodiment Transfer | Very High | YES — capability | 3 | X-VLA, RT-X, GR00T |
| 14 | Long-Horizon Chain Length / q-score | Med-High | YES | 2 | CALVIN, BEHAVIOR |
| 15 | Safety Incidents / Joint-Limit / Collision | Trivial | YES — deployment | 2 | RoboEval, VLA-Risk |
| 16 | Energy per Task | Low ~min | YES — edge claim | 2 | Neuro-Symbolic-VLA |
| 17 | Human A/B / Pairwise (RoboArena) | Very High | **YES — tournament** | 3 | RoboArena |
| 18 | World-Model / VLM-Judge Eval | Med | Yes w/ calibration | 3 | WorldEval, StepEval |
| 19 | AutoEval 24/7 autonomous real-robot | **Zero-reflex effort** | **YES** | 2 | AutoEval (CoRL 2025) |
| 20 | RoboEval motion quality (jerk / slip / path-len) | Trivial | YES — deployment | 2 | RoboEval |
| 21 | Embodied-Efficiency (completion time, smoothness) | Trivial | YES | 2 | Embodied-Efficiency survey |
| 22 | FLOPs / MACs / Params | Trivial | Yes — appendix | 2 | VLA-Perf, Efficient-VLA |
| 23 | MMRV / Sim-Real Correlation | Med | YES benchmark papers | 3 | SimplerEnv, WorldEval |
| 24 | Robometer reward-model scoring | Med | Yes | 3 | Robometer |
| 25 | Chain-length stability / chunk-cos | Trivial | Niche diagnostic | 3 | Adaptive-Chunking |
| 26 | Export Fidelity (cos=1.0 ref parity) | Trivial | **YES — reflex-native** | 1 | Reflex |
| 27 | StepEval subgoal VLM score | Low | Yes when no sim predicates | 3 | StepEval |
| 28 | VLABench interactive protocol | Med | Yes emerging | 3 | VLABench |
| 29 | VLA-FEB CMAS / FEI alignment | Med | Niche | 3 | VLA-FEB |
| 30 | Adversarial Risk (Eva-VLA / VLA-Risk) | Very High | Yes — safety | 3 | Eva-VLA, VLA-Risk |

### Tier 0 (deprecated / never caught on)
Single-number MSE-as-headline (fails to correlate with SR past ~40K training steps — HuggingFace/m1b 2024 blog; reviewer push-back explicit from 2024 onwards). FLOPs-only efficiency (criticized by Embodied-Efficiency survey 2026). Vanilla LIBERO SR reported in isolation without latency is now borderline-Tier-0 as of ICLR 2026 reviewer expectations.

---

# Recommended default battery for a new VLA deployment project (Reflex)

Four metrics. One for each of: correctness, capability, throughput, tail-latency. All Tier 1. All reflex can deliver today with the pipeline we have.

### 1. Export Fidelity — cos ≥ 0.9999 / L∞ < 1e-4 vs reference PyTorch

Why: this is the *precondition*. Without it, no other number is trustworthy. Reflex validate already computes it. Frame as "Reflex export preserves bit-near-identical outputs (cos=1.0 against LeRobot reference)." — the differentiator vs competitors whose ONNX quietly drifts.

Reference: Reflex's own CHANGELOG BREAKING (0.02 → 1e-4 threshold), veRL log-prob divergence precedent.

### 2. LIBERO-10 Sim Success Rate + SimplerEnv Visual-Matching SR

Why: these two together are the 2025-2026 de-facto capability headline. LIBERO-10 for long-horizon (10 tasks × 50 eps is the canonical protocol). SimplerEnv visual-matching as sim-for-real proxy. Report per-task SR + average. Both Reflex already wraps via vla-eval.

Reference: LIBERO (Liu et al., NeurIPS 2023), SimplerEnv (Li et al., CoRL 2024). OpenVLA-OFT 97.1%, SmolVLA 87.3%, Dream-VLA 60.5% SimplerEnv — these are the numbers to contextualize ours against.

### 3. Inference Hz + Memory Footprint per Hardware Target

Why: Reflex's wedge is edge deployment. One-line-per-target (Desktop / Jetson Orin Nano / Orin AGX / Thor): { Hz, peak VRAM MB, model size MB }. This is what a robotics customer reads first. SmolVLA paper's 2× async speedup framing is the template.

Reference: VLA-Perf (arXiv 2602.18397), Dexmal 3-5 FPS vs 20-30 Hz framing (arXiv 2510.26742), Characterizing-VLA "action gen 75% latency" (arXiv 2603.02271).

### 4. Streaming p50 / p95 / p99 Latency

Why: differentiates Reflex from model-only releases. Run the `reflex serve` harness under a 1000-request load test. A p99 under a clear budget (e.g., <100ms for single-arm) is the most customer-readable infra claim we can make. Gets the RTC chunk-overlap story into numbers.

Reference: Real-Time Chunking (arXiv 2506.07339), Async Robot Inference (HuggingFace blog 2025).

---

### What to include as **supporting plots** (Tier 2, below the fold)

If we have bandwidth for v0.2.1 or v0.3:
- LIBERO-Plus robustness curves (camera-viewpoint, lighting, distractor perturbations) — catches the "brittle under OOD" critique.
- Subgoal-SR decomposition on LIBERO-10 (free from BDDL predicates; distinguishes us from SR-only competitors).
- Energy-per-successful-task on Jetson (1-line claim: "N Joules / successful pick-place").
- RoboEval-style motion-quality metrics (jerk, path length) for a "motion-quality-aware deployment" story.

### What to explicitly **defer** (Tier 3, not v0.2/v0.3)

- STAR-Gen full 885-eval grid (overkill; revisit only if we claim generalist).
- Cross-embodiment SR (we ship export, we don't train — not our claim).
- Human-tournament RoboArena ranking (submit once a public deployment exists).
- Adversarial Eva-VLA CMA-ES stress tests (safety-paper scope only).

---

# Appendix: Key origin-paper pointers for citations

- **LIBERO**: Liu, Zhu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023.
- **SimplerEnv**: Li et al., *Evaluating Real-World Robot Manipulation Policies in Simulation*, CoRL 2024.
- **CALVIN**: Mees et al., *CALVIN: A Benchmark for Language-Conditioned Policy Learning*, RAL 2022 (arXiv 2112.03227).
- **Open X-Embodiment**: RT-X team, arXiv 2310.08864.
- **RT-2**: Brohan et al., arXiv 2307.15818.
- **OpenVLA / OpenVLA-OFT**: openvla-oft.github.io + arXiv.
- **SmolVLA**: arXiv 2506.01844 (LIBERO 87.3%, SO100 78.3%, async 2×).
- **pi0 / pi0-FAST**: pi.website + Penn-PAL-Lab wild-study.
- **GR00T N1 / N1.5**: NVIDIA, arXiv 2503.14734.
- **STAR-Gen**: Gao et al., arXiv 2503.01238 (CoRL 2025).
- **BEHAVIOR-1K**: behavior.stanford.edu, 2025 Challenge.
- **LIBERO-Plus / LIBERO-PRO**: arXiv 2510.13626, 2510.03827.
- **Eva-VLA**: arXiv 2509.18953.
- **VLA-Perf**: arXiv 2602.18397.
- **Characterizing-VLA**: arXiv 2603.02271 ("75% latency = action generation").
- **Dexmal Real-Time VLA**: arXiv 2510.26742.
- **RTC Real-Time Chunking**: arXiv 2506.07339.
- **vla-eval**: arXiv 2603.13966 (AllenAI harness).
- **AutoEval**: arXiv 2503.24278 (CoRL 2025).
- **RoboArena (real)**: arXiv 2506.18123.
- **RobotArena ∞ (sim)**: arXiv 2510.23571.
- **WorldEval**: arXiv 2505.19017.
- **RoboEval**: arXiv 2507.00435.
- **StepEval**: arXiv 2509.19524.
- **Embodied-Efficiency**: arXiv 2603.19131.
- **Robometer**: arXiv 2603.02115.
- **VLABench**: arXiv 2412.18194 (ICCV 2025).
- **Can-VLMs-Judge-Action-Quality**: arXiv 2604.08294.
- **X-VLA**: arXiv 2510.10274 (ICLR 2026).
- **ET-VLA**: arXiv 2511.01224.
- **Embodiment Scaling Laws**: arXiv 2505.05753.
- **veRL log-prob divergence**: upstream of Reflex; foundational for export-parity story.
