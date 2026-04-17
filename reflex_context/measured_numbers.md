# Measured Numbers

Single source of truth for what reflex-vla can credibly claim today. Three buckets: **verified** (reproducible script + commit), **unverified** (ran but don't trust), **unmeasured** (vapor).

Any number not in this file is not a claim. Any number in this file without a reproducer is a bug in this file — fix or move it.

Last updated: 2026-04-17.

---

## Verified

Reproducible on a fresh clone with the linked script at the linked commit. These are the only numbers safe to put in pitch decks, README, tweets.

| Date | Metric | Value | Reproducer | Commit |
|---|---|---|---|---|
| 2026-04-17 | Native path (lerobot `SmolVLAPolicy` + `DecomposedRMSNorm` swap) vs PyTorch reference, end-to-end first-action | **cos = 1.0000, L2 = 0.000** | `pip install 'reflex-vla[native]' && REFLEX_NATIVE=1 python scripts/local_full_diff.py` | `9de9f64` (re-verified post-rebalance; also holds on `0616265`) |
| 2026-04-17 | Vision encoder (SigLIP) vs PyTorch | cos = 1.0000, L2 = 2.2e-03, max_abs = 1.07e-04 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | Text embedder vs PyTorch | cos = 1.0000 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | State projection vs PyTorch | cos = 1.0000 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | Per-layer vlm_k (decomposed path) | cos = 1.0000 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | Single self-attn layer (layer 0, isolated weight copy) | cos = 1.0000 to 1e-5 | `scripts/local_single_layer_diff.py` | `0616265` |
| 2026-04-17 | Optimum-exported tiny Gemma ONNX vs PyTorch (`text-generation-with-past`) | cos = +1.00000000, max_abs_diff = 7.15e-07, KV diff = 1.79e-07 | `scripts/local_tiny_gemma_sanity.py` (per-layer K/V exposed as `present.N.key/value` outputs) | `85c24c0` |
| 2026-04-17 | **pi0's actual 2.51B Gemma backbone** (extracted from pi0_base) exported via Optimum ONNX vs PyTorch | **cos = +0.99999994**, logits max_abs_diff = 3.86e-05, present.17.key (deepest layer) max_diff = 3.02e-05 | Pi0's `paligemma_with_expert.paligemma.model.language_model.*` state-dict subset loaded into `GemmaForCausalLM`, exported via `optimum-cli export onnx --task text-generation-with-past`, 10GB on-disk, 9.3GB external-data ONNX | `f67a012` (Day 2 empirical) |

**Headline claim:** "Reflex's native export path matches the reference PyTorch policy to cos = 1.0000 end-to-end on SmolVLA." This is the only number load-bearing for the product pitch today.

---

## Unverified (ran, don't trust)

Numbers that came out of an actual run but aren't stable, aren't reproducible, or are known to be from a buggy code path. Do not cite externally.

| Date | Metric | Value | Why unverified |
|---|---|---|---|
| 2026-04-17 | LIBERO-10 success rate, decomposed ONNX path | 0% | Path had 12 known bugs at time of run; composition issue unresolved (layer_0_v cos = 0.9117 outlier, per-step velocity cos = 0.977 compounds to -0.24 over 10 Euler steps) |
| 2026-04-17 | LIBERO-10 success rate, native path | never completed | Modal container never finished — dep conflict between LIBERO-2023 stack and lerobot-2026 stack |
| 2026-04-13 | TRT FP16 latency headlines (2.6–3.3× FP32) | archived | Apples-to-oranges: FP16 TRT vs FP32 torch, not FP16 vs FP16; also never re-verified on native path |
| 2026-04-17 | Per-layer vlm_v cos (decomposed path, layers 1–15) | 0.91–1.00 | layer_0_v = 0.9117 is the outlier; unresolved in decomposed path. Native path bypasses this. |
| 2026-04-17 | Per-step velocity (decomposed expert) | cos = 0.977 per step | Compounds catastrophically across 10 flow-matching steps → final cos = -0.24. Native path supersedes. |

---

## Unmeasured (vapor)

Claims we'd want to make but have zero evidence for. If someone asks for one of these in a customer call or pitch, the honest answer is "we haven't measured it yet."

- **Jetson Orin Nano latency** — ms/step, Hz, memory. Never run.
- **Jetson Orin (32GB / 64GB) latency** — never run.
- **Jetson Thor (FP8) latency** — never run; no hardware access.
- **Desktop A10G latency on native path** — the benchmarks we have are from the decomposed path, pre-native pivot. Need re-run.
- **Calibration (ECE / Brier / NLL)** on any benchmark — never computed. This is the *one* non-sim metric with a monotonic link to real-world task success (Zollo 2025), and we have nothing.
- **Memory footprint** — VRAM per export stage on any target. Never measured.
- **Task success on any benchmark** (LIBERO, SimplerEnv, ManiSkill, real robot) — never completed.
- **pi0 / pi0.5 / GR00T native path parity** — only SmolVLA has a native path today. Decomposed paths for these were exported but never cos-compared.
- **Cold-start time** — first-action latency after load. Not measured.

---

## Methodology notes

**Shared noise.** All cos comparisons feed *the same noise tensor* to both PyTorch and ONNX paths. Flow matching is deterministic given noise, so a non-zero L2 with shared noise is a real discrepancy, not RNG. Without shared noise, the comparison is meaningless (see `04_iteration_lessons/shared_noise_discipline.md`).

**Cos_sim threshold.** cos = 0.977 per-step is *catastrophic* after 10 integration steps (observed: final cos = -0.24). Per-step parity must be > 0.999 to survive composition. This is why the decomposed path died and the native path is the production target.

**Reproducer discipline.** Every row in Verified has: script path, env flags, commit SHA. If you can't paste the command and get the number back, the number doesn't belong in Verified.

---

## Update protocol

When a new number gets measured:
1. If reproducible → add a row to **Verified** with reproducer + commit.
2. If ran but suspect → add to **Unverified** with the reason.
3. If superseded → move to Unverified with `archived` status, link to the replacement.
4. If targeted but not yet run → add to **Unmeasured** as a tracked gap.

When citing a number externally (README, tweet, pitch, blog): pull it from **Verified** only. Nothing else.
