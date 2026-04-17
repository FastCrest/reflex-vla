# Reflex VLA — Knowledge Base

Internal knowledge preservation for the Reflex VLA project. This folder is the single entry point for "what do we know, what did we decide, what did we try, what broke, and why" — designed to survive Claude session boundaries and serve as ground truth when a new session needs to ramp up quickly.

**This folder is NOT shipped to users.** It's the engineering / strategic scratchpad. User-facing docs live in `reflex-vla/README.md`, `reflex-vla/docs/getting_started.md`, and `reflex-vla/launch/`.

## What this folder is

- A **context cache** distilled from 13 session files (~1.12M tokens in the marquee session alone), 113 git commits, 32 Modal scripts, 6 prior transcripts, 11 currently-running Modal apps, and `GOALS.yaml`.
- **Decisions** with their provenance (why this, why not the alternative).
- **Bugs** found, with the exact symptoms, root cause, fix, and where the fix lives in code.
- **Research** notes on papers and competitor landscape.
- **Experiments** with setup, results, and caveats.
- **Sessions** — chronological narrative of debugging marathons and aha moments.
- **Iteration lessons** — workflow-level "always do X because Y bit us."

## How to use it

1. **Start here** (`reflex_context/README.md`) to orient — pick the question you're asking.
2. **Use the Quick-reference index** below to jump to the right file.
3. When in doubt, cross-links between docs are generous; follow them.
4. When you close a session, update the **Maintenance** section at the bottom.

## Directory tree

```
reflex_context/
├── README.md                                    ← you are here
├── mvp_queue.md                                 — MVP ship list (11 goals), deferred-to-v0.3 list, pricing + outreach
├── measured_numbers.md                          — verified / unverified / unmeasured (single source of truth for claims)
├── _raw/                                        — verbatim session / git / script extracts (source of truth)
│   ├── sessions_md.md
│   ├── modal_scripts.md
│   ├── git_history.md
│   ├── current_session.md
│   ├── prior_sessions.md
│   └── modal_apps_and_pm_docs.md
│
├── 00_vision/
│   ├── INDEX.md                                 — table of contents for 00_vision
│   ├── north_star.md                            — "Deploy any VLA model to any edge hardware. One command."
│   ├── positioning.md                           — "the deployment layer for VLAs"; not training, not cloud inference
│   └── moat.md                                  — cross-framework ONNX + Jetson-first + deterministic graph
│
├── 01_architecture/
│   ├── smolvla_forward_pass.md                  — canonical: how the real SmolVLA computes actions (VLM prefix + expert + flow-match)
│   ├── onnx_export_decisions.md                 — why we decompose RMSNorm+RoPE, what's tradable, opset 19/23 gap
│   ├── reflex_server_stack.md                   — pipeline diagram: ORT / TRT / CUDA-graph / batched / wedge composition
│   └── gr00t_full_stack.md                      — GR00T's action_encoder + DiT + action_decoder wrap
│
├── 02_bugs_fixed/
│   ├── smolvla_pipeline_bugs.md                 — the 12 bugs table (state_proj, 5D pixels, RoPE base, send_state, normalizer, text_det, ...)
│   ├── libero_integration.md                    — bddl / gym / osmesa / robosuite pin / patch_libero.py saga
│   ├── modal_gotchas.md                         — ORT silent CPU fallback, cuDNN gap, subprocess.PIPE deadlock, preemption
│   └── pydantic_forward_ref.md                  — FastAPI HealthResponse module-scope fix
│
├── 03_research/
│   ├── direct_torch_export_viability.md         — can `torch.onnx.export` replace our decomposition? TRT opset 23 gap analysis
│   ├── pi_flow_vs_dmpo.md                       — distillation recipe: pi-Flow (ICLR'26) vs DMPO (arXiv 2601.20701 / 1770 Hz one-step)
│   ├── checkpoint_formats.md                    — smolvla_base vs smolvla_libero layouts, state_dict prefixes
│   ├── vla_eval_schema.md                       — obs schema, camera naming, states vs controller_states
│   └── hardware_alternatives.md                 — CloudJetson.com, used Orin Nano ($200-240), NVIDIA Inception reality
│
├── 04_iteration_lessons/
│   ├── local_vs_modal.md                        — local iteration ~100× cheaper, do this first next time
│   ├── diagnostic_ladder.md                     — stage diff → single-layer → composition; the bisection strategy
│   ├── cost_log.md                              — ~$8-15 Modal per hunting session, $200-500 for distill
│   └── subprocess_buffering.md                  — line-by-line stream > capture_output; file > PIPE
│
├── 05_sessions/
│   ├── 2026-04-10_marquee_session.md            — path selection + wedge design + full week of LIBERO hunt
│   ├── 2026-04-14_gpu_postmortem.md             — silent CPU fallback → TRT FP16 flip (2.6-3.3×)
│   ├── 2026-04-14_batching_phase3.md            — 2.88× throughput + TRT static-shape 34s rebuild
│   ├── 2026-04-16_vlm_prefix_real.md            — 4-file VLM split, GQA spike, decoder prefill
│   ├── 2026-04-16_libero_integration.md         — 18-commit LIBERO install death march
│   └── 2026-04-17_libero_correctness_hunt.md    — stage-diff + pytorch-vs-onnx correctness probes
│
├── 06_experiments/
│   ├── latency_benchmarks.md                    — TRT FP16 per-model Hz, per-hardware extrapolations, memory fit
│   ├── adaptive_denoising_validation.md         — Phase IV per-model verdicts (pi0 only)
│   ├── batching_validation.md                   — Phase III 2.88× throughput + TRT-EP auto-disable
│   ├── pytorch_vs_onnx_cos_sim_timeline.md      — chronology: 0.28 → 0.498 → 0.305 → 0.08 → -0.27 → -0.24
│   └── stage_diff_snapshot.md                   — per-stage cos_sim after all fixes
│
└── [not yet created]
    ├── 07_product/                              — PRDs, roadmap (currently lives in _raw/modal_apps_and_pm_docs.md)
    ├── 08_launch/                               — launch drafts + sequencing (currently lives in reflex-vla/launch/)
    ├── 09_inbox/                                — unprocessed notes
    └── 10_archive/                              — retired docs
```

Note: `01_architecture`, `02_bugs_fixed`, `03_research`, `04_iteration_lessons`, `05_sessions` above are the target layout. Some files may be in-progress; `_raw/` has the verbatim source material until the distilled versions are complete.

## Quick-reference index

**Product / strategy**
- What's the one-liner? → `00_vision/north_star.md`
- What's the moat? → `00_vision/moat.md`
- Why did we pivot from Datadog-for-inference to VLA? → `00_vision/positioning.md` and `_raw/sessions_md.md` § Path Alt-VLA
- What's the pricing plan? → `_raw/modal_apps_and_pm_docs.md` § Pricing ladder

**Architecture**
- How does SmolVLA forward pass work? → `01_architecture/smolvla_forward_pass.md`
- Does pi0 need DecomposedRMSNorm swap? → `01_architecture/pi0_rmsnorm_already_decomposed.md` (answer: no, PiGemmaRMSNorm is already elementwise)
- Why do we decompose RMSNorm and RoPE? → `01_architecture/onnx_export_decisions.md`
- What's the serve pipeline? → `01_architecture/reflex_server_stack.md`
- How does GR00T's full-stack wrap work? → `01_architecture/gr00t_full_stack.md`

**Bugs**
- What bugs should I check on a fresh SmolVLA decomposed export? → `02_bugs_fixed/smolvla_pipeline_bugs.md`
- Why did the Apr-17 LIBERO run fail? → `05_sessions/2026-04-17_libero_correctness_hunt.md` + `02_bugs_fixed/smolvla_pipeline_bugs.md`
- Modal image won't run my ORT-GPU code? → `02_bugs_fixed/modal_gotchas.md` (answer: CUDA 12 vs 13, cuDNN 9, pin `torch==2.5.1 + onnxruntime-gpu==1.20.1 + nvidia-cudnn-cu12==9.*`; use `nvcr.io/nvidia/tensorrt:24.10-py3` base)
- LIBERO silently hangs on `env.reset`? → `02_bugs_fixed/libero_integration.md` (answer: subprocess stdout capture buffers; switch to osmesa; `patch_libero.py` for `input()` calls; robosuite==1.4.1 pin)
- FastAPI crashes on Pydantic ForwardRef? → `02_bugs_fixed/pydantic_forward_ref.md`

**Research**
- Can I just use `torch.onnx.export` directly? → `03_research/direct_torch_export_viability.md` (answer: yes for attention/RoPE/GQA; no for RMSNorm on Jetson TRT until opset 23 support lands in the ONNX parser)
- What's our distillation recipe? → `03_research/pi_flow_vs_dmpo.md` (pi-Flow → DMPO as of Apr-16)
- How do I get cheap Jetson access? → `03_research/hardware_alternatives.md` (CloudJetson.com ~$5/session, used Orin Nano Super $200-240)
- **What code can I import for pi0-onnx-parity?** → `03_research/pi0_onnx_importable_sources.md` (15+ sources tiered by license risk; ~25–35 day savings possible)
- What's the LeRobot issue we're closing? → `_raw/modal_apps_and_pm_docs.md` § LeRobot #3146

**Iteration / workflow**
- Is local or Modal faster for iteration? → `04_iteration_lessons/local_vs_modal.md` (answer: **local is ~100× cheaper** for diff/bisect work; use Modal for final integration tests only)
- How should I bisect a correctness bug? → `04_iteration_lessons/diagnostic_ladder.md` (stage diff → single layer → composition)
- How much does a hunting session cost? → `04_iteration_lessons/cost_log.md` (~$8-15 Modal per session, $200-500 for distill)
- Why does `subprocess.run(capture_output=True)` look like it hangs? → `04_iteration_lessons/subprocess_buffering.md` (answer: use a file for stdout, not PIPE; stream line-by-line)

**Product / strategy (operational)**
- **What ships in MVP, what defers to v0.3?** → `mvp_queue.md`
- **Who are the first 3 customers and pricing?** → `mvp_queue.md` § Pricing + customer outreach

**Benchmarks**
- **What can we actually claim?** → `measured_numbers.md` (verified / unverified / unmeasured — cite only from Verified)
- What's the latency per model? → `06_experiments/latency_benchmarks.md`
- Does adaptive denoising work? → `06_experiments/adaptive_denoising_validation.md` (answer: only on pi0; 58% savings; never triggers on SmolVLA, drifts on pi0.5/GR00T)
- Does batching work? → `06_experiments/batching_validation.md` (answer: 2.88× on pi0 at batch=16; TRT EP auto-disabled when `max_batch > 1` because static-shape ONNX rebuilds engine per call = 34s)
- How did cos_sim evolve as we fixed bugs? → `06_experiments/pytorch_vs_onnx_cos_sim_timeline.md` (0.28 → 0.498 → 0.305 → 0.08 → -0.27 → -0.24)
- Per-stage diff snapshot? → `06_experiments/stage_diff_snapshot.md` (components all at 1.0000 / 0.999 / 0.9117; expert velocity 0.977 per step; final -0.24)

**Sessions**
- The marquee session that defined the project? → `05_sessions/2026-04-10_marquee_session.md`
- The GPU post-mortem that flipped the narrative? → `05_sessions/2026-04-14_gpu_postmortem.md`
- The LIBERO install death march? → `05_sessions/2026-04-16_libero_integration.md`
- The latest correctness hunt? → `05_sessions/2026-04-17_libero_correctness_hunt.md`

**Raw source material**
- Verbatim session extracts → `_raw/sessions_md.md`
- All 32 Modal scripts with gotchas → `_raw/modal_scripts.md`
- Git history with per-commit learnings → `_raw/git_history.md`
- Current (Apr-10) session mined → `_raw/current_session.md`
- Prior sessions discovered → `_raw/prior_sessions.md`
- Modal apps + PM docs (README/CHANGELOG/launch drafts/GOALS.yaml) → `_raw/modal_apps_and_pm_docs.md`

## Conventions

- **Each file has a header, setup/params/results/caveats structure where relevant.**
- **File naming**: `topic_snake_case.md`. No timestamps in filenames unless the file is time-bound (session logs in `05_sessions/` use `YYYY-MM-DD_topic.md`).
- **Cross-links**: use relative paths from `reflex_context/`, e.g., `03_research/pi_flow_vs_dmpo.md`. Avoid absolute paths except in `_raw/` source citations.
- **Numbers**: always cite source (git SHA, Modal run ID, or session line ref). Any number without a source is a smell.
- **ADRs**: live in `01_decisions/` (TBD naming — currently embedded in `_raw/sessions_md.md` and `git_history.md`). Reference them as `ADR: 2026-04-14-strict-provider-no-silent-cpu-fallback.md` etc.
- **Length**: each file 80-250 lines. Longer → split into subtopics. Shorter → merge into a parent topic.
- **Do not duplicate.** If information lives in another doc, link to it. Dedup on write.

## Maintenance

**At the end of every session that touches reflex-vla:**

1. **Read the latest session transcript** and extract any new substantive finding. If it fits an existing file, edit. If it's new, create (or add to `_raw/` first as a staging area).
2. **Update the cost log** (`04_iteration_lessons/cost_log.md`) with Modal spend for the session.
3. **Append to the relevant session log** in `05_sessions/YYYY-MM-DD_topic.md` with the chronological narrative — what was tried, what worked, what broke.
4. **Refresh benchmarks** if you re-ran any bench script. `06_experiments/latency_benchmarks.md` is the single table.
5. **If you fixed a bug**, add to `02_bugs_fixed/` with symptoms / root cause / fix / code location.
6. **If you made a strategic decision**, add an ADR. The ADR filename format is `YYYY-MM-DD-decision-slug.md`.
7. **Update this README's Quick-reference index** if a new question now has a good answer.
8. **Check for stale cross-links** (files renamed / merged without redirecting).
9. **Archive** anything that's now wrong (e.g., old benchmark numbers superseded) into `10_archive/` with a one-line note on why.

**At the end of every week:**
- Review `05_sessions/` and fold repeat patterns up into `04_iteration_lessons/`.
- Review `02_bugs_fixed/` for patterns that should become `03_research/` entries.
- Check `GOALS.yaml` vs this folder — goals should cite the relevant context doc.

**When starting a new Claude session:**
- Open this README first.
- Jump to the Quick-reference index for the specific question.
- If none of the listed answers fit, scan `_raw/` for verbatim material.

## Known open threads (as of 2026-04-17)

- **LIBERO-10 task success = 0%** despite 12 fixes — expert velocity cos=0.977 per step compounds to -0.24 final. Remaining candidate: cross-attention composition inside the expert stack. Path forward: copy `lerobot`'s `SmolVLAPolicy.sample_actions` + `forward_cross_attn_layer` verbatim, swap only `RMSNorm → DecomposedRMSNorm`. See `06_experiments/stage_diff_snapshot.md`.
- **`layer_0_v cos=0.9117`** is the single reproducible structural discrepancy in VLM prefill. Tracked as task #25.
- **Jetson-native benchmarks** not yet captured. High-weight goal. CloudJetson.com + eBay refurb are the likely paths.
- **FP16 TRT vs FP16 torch.compile** apples-to-apples not yet done; published 2.6-3.3× headline compares FP16 vs FP32.
- **Fine-tuned SmolVLA VLM layers not preserved** (v0.3 item) — current export uses BASE SmolVLM2-500M.
- **`reflex distill` training loop** scaffolded but not operational (v0.2.1 target).
- **Per-model adaptive denoising thresholds** deferred to v0.2.

## Attribution

Compiled from session transcripts, git history, Modal script mining, and PM docs. Sources are cited in each file. The marquee source for most of this knowledge is session `ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl` (~1.12M tokens) — when in doubt, that's where the verbatim claim lives.

Last updated: 2026-04-17.
