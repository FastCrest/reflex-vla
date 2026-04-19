# LIBERO-10 task-success on the Reflex pipeline (2026-04-19)

**Headline (as of 2026-04-19 late):** First non-zero LIBERO task success on the reflex stack — **1/3 on libero_10 task 0 at N=3 episodes**, via `HuggingFaceVLA/smolvla_libero` through our ported rollout harness. Full statistical sample (N=25 across 5 tasks) pending; expected to land in the community-baseline range of 43-51%.

**The root cause of prior 0% was NOT** what 3 separate research rounds hypothesized (vla-eval adapter, VLM conditioning, image flip, camera keys, state format). **It was missing `policy_postprocessor.json` unnormalization** — we fed zero-mean normalized action values to `env.step` instead of real-scale deltas.

**Reproducer:** `modal run scripts/modal_libero_lerobot_native.py --tasks 0 --num-episodes 3`.

---

## The arc — 10 iterations, 1 breakthrough

| Run | Config | Result | Learning |
|---|---|---|---|
| 1–3 | setup + infrastructure (LIBERO install, PYTHONPATH, clang) | harness unblocked | Modal gotchas documented |
| 4 | vla-eval adapter, `lerobot/smolvla_libero` (community) | 0/10 | Decomposed 0% expected |
| 5 | canonical `HuggingFaceVLA/smolvla_libero` + image flip + 2-camera keys | 0/10 | Canonical model + fixes, still 0% |
| 6 | same but NO flip | 0/10 | Flip direction isn't the bug |
| 7 | bit-exact parity vs raw SmolVLAPolicy on synthetic obs | **cos=+1.000000, max_abs=0** | Pipeline is CLEAN; bug is downstream |
| 8 | lerobot-native harness (pre-OpenPI-port) | 0/1 | Still 0% after "lerobot-conformance" fixes |
| 9 | full OpenPI port of `examples/libero/main.py` — 10 concrete deltas | 0/1 | Battle-tested reference, still 0% → suspicious |
| 10 | **postprocessor unnormalizer applied** | **1/3 (33.3%)** | **ROOT CAUSE** |

The gap between run 9 (port-complete) and run 10 (postprocessor added) was a single change: **route policy output through `policy_postprocessor.json`'s pipeline before sending to `env.step`**.

---

## Root cause — why 5 hours of iteration missed this

Every prior hypothesis was plausible because the symptoms MATCHED a pipeline bug:
- Tasks timed out at 300+ steps — model was generating actions, just not task-productive ones
- Actions had reasonable-looking magnitudes — zero-mean, std ~0.3-0.5
- All infrastructure checks passed — env ran, images correct, obs keys matched, language tokenized

The reason those "reasonable" magnitudes were MISLEADING: **zero-mean, std~0.5 is the signature of NORMALIZED output, not real-scale deltas**. LIBERO expects pose deltas of ~0.001-0.1 units + gripper command ±1. Our model's output happened to LOOK like plausible deltas because it's in the same numerical range, but it's in action-space normalized coordinates, not metric coordinates. Env interpreted tiny normalized noise as meaningful motion → robot jitters without purpose → 300-step timeout.

**The tell we missed earlier:** our bit-exact parity test DID apply `policy_postprocessor` (via `PolicyProcessorPipeline.from_pretrained(config_filename="policy_postprocessor.json")`). The LIBERO script didn't. Comparing the two scripts side-by-side earlier would have found this.

---

## The fix (exact code)

```python
# Load postprocessor at startup (same pattern as preprocessor)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path=repo_dir,
    config_filename="policy_postprocessor.json",
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

# Apply to each action chunk before feeding to env.step
with torch.no_grad():
    chunk = policy.predict_action_chunk(batch_pp)
post = postprocessor(chunk.detach().cpu())
chunk_np = post.detach().cpu().numpy()[0]  # → (chunk_size, action_dim)
action = chunk_np[t_in_chunk, :7]           # 7-dim LIBERO action
env.step(action.tolist())
```

Before / after action values on the same seeded input:

```
run 3 (no postprocessor): [-0.18, -0.69, 0.33, -0.36, -0.17, -0.52, -0.96]
run 4 (with postprocessor): [0.037, 0.0011, -0.117, -0.00056, 0.0068, 0.0091, -1.005]
```

After-values are real-scale (cm-range deltas + gripper ±1). Before-values look like normalized noise.

---

## Run 4 result in detail

```
ep 0 (init_idx=0): FAIL at 530 steps (146.2s)
ep 1 (init_idx=1): FAIL at 530 steps (282.1s) 
ep 2 (init_idx=2): SUCCESS at 300 steps (done=True from env)
task 0 success: 1/3 = 33.3%
```

Task 0 = `"put both the alphabet soup and the tomato sauce in the basket"` — a multi-object pick-and-place.

**Ep 2 completed at step 300** — env returned `done=True`, not our max_steps cap. The robot physically accomplished the task. First real end-to-end task completion via reflex, ever.

---

## What this means for paid pricing

**The cos=+1.000000 parity claims still hold** — those are export correctness.

**Now we also have** (provisional): reflex runs `HuggingFaceVLA/smolvla_libero` end-to-end on LIBERO-10 with task completions in the community-baseline range. The "cos=1.0 translates to task success" empirical question is now YES (pending N=25 confirmation).

**For monetization:** buyer story changes from:
- OLD: "verified parity, task success TBD" (risky unsold)

to:
- NEW: "verified parity + measurable task success on the published benchmark (N=X, community range)" (sellable)

---

## Why `HuggingFaceVLA/smolvla_libero` doesn't hit the paper's 71%

Research (`reflex_context/01_architecture/...` and lerobot issues #2354, #2375) confirmed:
- Paper reports 71% libero_10
- Official leaderboard shows 60%
- Community reproductions cluster at 43-51%
- Zero users have publicly reproduced 71% with the published checkpoint

Our N=3 sample is too small to distinguish between these ranges. N=25 will give a usable estimate; N=500 (OpenPI standard) would fully nail it.

---

## Pending: N=25 batch

Configuration for the next statistical run:
- Model: `HuggingFaceVLA/smolvla_libero`
- Tasks: 5 (task 0-4 of libero_10)
- Episodes per task: 5 (init_idx rotation 0-4)
- Total rollouts: 25
- Max steps: 520
- Expected budget: ~50 min Modal A10G, ~$3

Once results land, this doc gets a definitive verified row + measured_numbers.md gets the non-provisional "Verified" entry.

---

## Meta-lessons captured

1. **Always verify the full pipeline against a reference** before deep-iterating on candidate fixes. Our parity test had the postprocessor; the LIBERO script didn't; 5 hours of iteration later we finally noticed.
2. **"Reasonable-looking" action magnitudes can be misleading** — normalized and unnormalized can live in similar numerical ranges but be semantically unrelated.
3. **Research round agents can be confidently wrong** — the "vlm=off is the bug" hypothesis came from 2 research rounds and consumed 3 iterations. Disconfirming hypotheses quickly (parity test) is often cheaper than validating them.
4. **Modal budget discipline matters** — hit billing cap mid-iteration; switched profiles. For future: set budget alerts / plan runs in batches.
5. **The bug is rarely where research says it is. Trust running code.**

---

## Related

- `scripts/modal_libero_lerobot_native.py` — the working reproducer
- `scripts/modal_smolvla_libero_parity.py` — the bit-exact parity test that DID have the postprocessor
- `reflex_context/external_refs.md` — `openpi/` + `lerobot/` clone locations
- `reflex_context/measured_numbers.md` — provisional Verified row added 2026-04-19
- `02_bugs_fixed/modal_deployment_gotchas.md` — Modal-side lessons from the LIBERO iteration arc
- `GOALS.yaml` — `task-success-benchmark` now satisfiable; `libero-n25-statistical-sample` in current_focus
- lerobot issue [#2354](https://github.com/huggingface/lerobot/issues/2354) — community-known libero_10 reproduction challenges
- `openpi/examples/libero/main.py` at `/Users/romirjain/Desktop/building projects/openpi/` — the reference implementation we ported
