# 2026-04-19 (afternoon/evening) — LIBERO task success + GR00T VLM step 2

**Session theme:** continuation of the morning's "all four VLAs at cos=1.0" work (see `2026-04-19_all_four_vlas.md`). Afternoon focus: close the gap from "cos=1.0 parity" to "real task success" + start closing the GR00T VLM zero-stub gap.

Two big wins:
1. **LIBERO-10 first real task-success: 10/25 = 40%** via OpenPI-ported harness. Root cause of prior 0% was missing `policy_postprocessor` (action unnormalizer), not the 5 hypotheses from earlier research rounds.
2. **GR00T Step 2 port VERIFIED** — state_encoder + vlm_kv plumbing through `GR00TFullStack`, both conditioning paths confirmed LIVE via Modal sanity. Also discovered a bonus gap: prior GR00T export was missing `action_head.state_encoder` (54.6M params), not just VLM conditioning.

---

## Customer dogfood → 8 bug fixes → v0.2.1

Early afternoon: first-time-user dogfood exercise on fresh Modal container. Pretended to have only seen README; typed commands verbatim; recorded friction. Uncovered 8 customer-facing bugs:

1. README `reflex export` command defaulted to DECOMPOSED (abandoned) path instead of monolithic (cos=1.0 verified)
2. `[monolithic]` extras install conflict with base `transformers<5.0`
3. NVIDIA TRT container missing `clang` (evdev C build fails)
4. SmolVLA tokenizer silently returned zeros (no pad_token set)
5. `/act` response missing `denoising_steps` field that README promises
6. VERIFICATION.md `Target: unknown` despite `--target desktop`
7. No README warning for 10+ min first-export time
8. Modal image cache served stale code silently between iterations (SHA-pin fix)

All 8 fixed + verified end-to-end. Tagged **v0.2.1** — GHCR image auto-publish triggered.

Details: `reflex_context/06_experiments/customer_first_run_transcript.md` (full arc v1→v6) + `02_bugs_fixed/modal_deployment_gotchas.md` (+3 new Modal gotchas).

---

## LIBERO-10 arc — 10 iterations, 40% final

Entered the afternoon with LIBERO = 0% from our April measurements (decomposed path, 12 reimpl bugs) + an implicit assumption that "cos=1.0 parity implies task success." Exited with a measured 40%.

**Research rounds during iteration** (to be clear about what went wrong):
- Round 1 hypothesis: adapter `vlm=off` silences language conditioning. **WRONG** — was a false log signal (checked wrong attribute on wrong server class).
- Round 2 hypothesis (5 deltas via web research): flip, camera keys, n_action_steps, state format, checkpoint version. Implementing all 5 still produced 0%.
- Pivoted to **OpenPI port** (`scripts/modal_libero_monolithic.py` → `scripts/modal_libero_lerobot_native.py`): line-by-line port of `openpi/examples/libero/main.py`. Still 0% at run 1 of the port.
- **Run 4 discovery**: missing `policy_postprocessor.json` unnormalizer. Bit-exact parity test HAD it; LIBERO script didn't. After 5 hours of iteration. First success: **1/3 on task 0 at N=3**.

**N=25 sample** (5 tasks × 5 episodes with init-state rotation):
- Task 0 (alphabet-soup+tomato): 3/5 = 60%
- Task 1 (cream-cheese+butter): 3/5 = 60%
- Task 2 (stove+moka): 3/5 = 60%
- Task 3 (bowl-in-drawer): 1/5 = 20%
- Task 4 (mug-on-plate): 0/5 = 0%
- **Overall: 10/25 = 40.0%**

Solidly in the lerobot community's 43-51% reported range. Paper claims 71% but community hasn't reproduced that publicly.

Full writeup: `06_experiments/task_success_results.md`.

**Meta-lessons captured:**
1. Always compare your script against a bit-exact parity test that's known to work. If one has the postprocessor and the other doesn't, five hours of iteration gets wasted.
2. "Reasonable-looking" action magnitudes can be misleading — normalized (zero-mean, ~0.3-0.5 std) and unnormalized (cm-scale) can live in similar numerical ranges but be semantically unrelated.
3. Research-round agents produce plausibly-wrong hypotheses. Disconfirming (bit-exact parity test) is often cheaper than validating five deltas sequentially.

---

## GR00T Eagle VLM — Steps 1+2 done

After LIBERO, pivoted to closing the GR00T zero-stub VLM gap. Research showed the 5-step plan (`01_architecture/gr00t_eagle_vlm_export_plan.md`):

1. Vendor Eagle source — ✅ DONE (4 files, 1575 lines copied from lerobot, peft made optional, FA2 → eager default)
2. Port state_encoder + extend GR00TFullStack — ✅ DONE, sanity VERIFIED
3. Lerobot parity test — 🟡 running at time of this write
4. Modal ONNX export — pending Step 3
5. End-to-end chain test — pending Step 4

**Key discoveries during Step 1c** (state-dict key dump, `01_architecture/gr00t_n16_state_dict_analysis.md`):
- `eagle_linear` (2048→1536 projection) ABSENT in N1.6 — feeds raw Qwen2 into DiT's `vlln`
- State dict prefix is `backbone.model.*` (not `backbone.eagle_model.*` as research guessed)
- **NEW gap discovered: `action_head.state_encoder`** (54.6M params, 32-embodiment) was missing from prior reflex exports. Our current GR00T was missing STATE conditioning too, not just VLM.
- `future_tokens` ABSENT in N1.6 — DiT sequence is simpler: `sa_embs = cat(state_token, action_tokens)` (no learnable prefix)
- DiT substructure: timestep_encoder, 32 transformer_blocks, proj_out_1 (final AdaLN), proj_out_2 (velocity)

**Step 2 verification** (`scripts/modal_gr00t_state_encoder_sanity.py`):
- state_encoder loads from N1.6 ✅
- `GR00TFullStack.forward(noisy, t, pos, state, vlm_kv)` runs ✅
- **VLM conditioning dominant**: ratio 2.85 (changing vlm_kv from zeros to random = 47x larger effect than changing state). Expected — vision + language drive decisions, proprio state secondary.
- Back-compat (state=None) works ✅

---

## GOALS.yaml backlog refreshed

Completed and marked:
- `task-success-benchmark` → DONE (10/25 = 40% N=25)
- `libero-adapter-lerobot-conformance` → DONE (via postprocessor fix)
- `vla-eval-adapter-vlm-on` → SUPERSEDED (wrong hypothesis)
- `libero-preprocessing-audit` → SOLVED (postprocessor was the actual delta)

Added to improvement backlog:
- `gr00t-eagle-vlm-steps-3-5` (weight 9) — complete the Eagle VLM port
- `libero-task34-gap-audit` (weight 8) — investigate why tasks 3-4 underperform (20%/0% vs 60% avg)
- `libero-via-monolithic-onnx-task-success` (weight 9) — verify ONNX path hits same 40%
- `libero-n50-paper-standard-sample` (weight 5) — OpenPI standard N=50 rollouts
- `simpler-env-task-success` (weight 6) — 2nd benchmark beyond LIBERO

---

## Commits today (afternoon arc)

Rough selection from ~30 commits since `2026-04-19_all_four_vlas.md` closed:

- `a9f7a24` — captured morning dogfood arc in reflex_context
- `3bba419` — launch verification gates narrative doc
- `1e34adb` — 6 monetization-blocker goals added
- `c8a6929` — customer dogfood transcript + Modal gotchas
- `a8fd0c4` / `b8a7916` / `51592e5` — 3 customer-dogfood fixes
- `95ef679` / `6858838` — 4 more dogfood fixes + Modal SHA-pin cache fix
- `23f0c90` — first LIBERO 0/10 finding documented
- `fc069c0` → `32e814d` → `f4101eb` → many iterations — LIBERO harness debugging
- `ddec588` — **POSTPROCESSOR FIX** (the breakthrough)
- `7b7ba24` — LIBERO breakthrough committed (1/3)
- `d98658c` — GR00T state-dict analysis
- `16d6d9c` — GR00T step 2 port (state_encoder)
- `cb99865` / `f1d1510` — Step 2 sanity script + VERIFIED
- `c6374dc` — **LIBERO N=25 FINAL: 10/25 = 40%**
- `890adcd` — GOALS.yaml refresh with improvement backlog

Plus `v0.2.1` tag pushed to GHCR.

---

## What's next (post-parity)

Decision tree waiting on the Step 3 parity test:
- If **cos=+1.000000, max_abs<1e-4** (bit-exact): proceed directly to Step 4 (Modal ONNX export of `eagle_vlm.onnx` + `expert_stack_with_vlm.onnx`)
- If **cos ≥ 0.99 but not bit-exact**: small numerical drift, investigate but could still ship
- If **cos < 0.99**: real divergence, debug the delta in Step 2 port before continuing

After GR00T Steps 3-5:
- `libero-task34-gap-audit` to investigate the 20%/0% weakness on hard tasks
- `libero-via-monolithic-onnx-task-success` to verify customer-facing ONNX path hits 40%
- `libero-n50-paper-standard-sample` for the definitive public number

---

## Related docs

- `reflex_context/05_sessions/2026-04-19_all_four_vlas.md` — the morning arc (cos=1.0 on 4 VLAs)
- `reflex_context/06_experiments/customer_first_run_transcript.md` — dogfood exercise
- `reflex_context/06_experiments/task_success_results.md` — LIBERO arc + N=25 final
- `reflex_context/01_architecture/gr00t_eagle_vlm_export_plan.md` — 5-step plan with Steps 1+2 marked DONE
- `reflex_context/01_architecture/gr00t_n16_state_dict_analysis.md` — key-dump findings
- `reflex_context/02_bugs_fixed/modal_deployment_gotchas.md` — +3 new Modal gotchas
- `GOALS.yaml` — current_focus refreshed post-N=25
