# External reference clones (2026-04-19)

We cloned two external repos OUTSIDE this project's directory (as siblings, not inside `reflex_context/`) so that future Claude sessions can `Read`/`Grep` them without bloating our own repo or tracking external git history.

## Clone locations

Both at `/Users/romirjain/Desktop/building projects/` (sibling to `reflex-vla/`):

| Repo | Path | Purpose |
|---|---|---|
| `Physical-Intelligence/openpi` (depth 1) | `/Users/romirjain/Desktop/building projects/openpi/` | Battle-tested LIBERO rollout reference |
| `huggingface/lerobot` (depth 1) | `/Users/romirjain/Desktop/building projects/lerobot/` | Official SmolVLA source + LIBERO env + processors + eval script |

Total size: ~17 MB (openpi=2.4MB, lerobot=15MB).

## Key files (the ones we're porting from)

**openpi** (scope: the 219-line battle-tested LIBERO rollout that actually runs):
- `openpi/examples/libero/main.py` — 219 lines. The end-to-end rollout loop: init_state rotation, 180° flip, `num_steps_wait=10` settling, `replan_steps=5` chunk management, `_quat2axisangle`. Copy this wholesale as the template.

**lerobot** (scope: exact source for the official SmolVLA/LIBERO integration):
- `lerobot/src/lerobot/envs/libero.py` — 489 lines. `LiberoEnv` class, `_format_raw_obs`, per-suite `TASK_SUITE_MAX_STEPS` (libero_10 = 520, not 300), init_state handling.
- `lerobot/src/lerobot/processor/env_processor.py` — 228 lines. `LiberoProcessorStep` with `_quat2axisangle` (the CORRECT formula, not the wrong one the earlier research agent suggested) + 180° image flip applied inside `_process_observation`.
- `lerobot/src/lerobot/scripts/lerobot_eval.py` — 844 lines. The official eval loop showing the full processor pipeline order: `preprocess_observation` → task injection → `env_preprocessor` (LiberoProcessorStep) → policy `preprocessor` → `select_action` → `postprocessor`.
- `lerobot/src/lerobot/policies/smolvla/processor_smolvla.py` — `make_smolvla_pre_post_processors` defines the policy preprocessor steps (rename → to_batch → add newline → tokenize → device → normalize).

## Why NOT to put these inside `reflex_context/`

- `reflex_context/` is for distilled project knowledge (session logs, bugs, architecture notes, measured numbers). External repos would bloat it from ~2 MB of markdown to ~19 MB, and mix our knowledge with vendor code.
- Git-tracking external code is an anti-pattern (submodules / vendoring is the "proper" solution but overkill here).
- Siblings at the parent dir are out of the way but accessible. Future sessions find this note, check paths exist, and `Read` the files directly.

## Refresh protocol

These are `--depth 1` shallow clones. To update later:

```bash
cd "/Users/romirjain/Desktop/building projects/openpi" && git pull --ff-only origin main
cd "/Users/romirjain/Desktop/building projects/lerobot" && git pull --ff-only origin main
```

If either repo gets deleted, re-clone via:

```bash
cd "/Users/romirjain/Desktop/building projects/"
git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git
git clone --depth 1 https://github.com/huggingface/lerobot.git
```

## Cross-links

- `06_experiments/task_success_results.md` — the 0% LIBERO finding that motivated pulling in these references
- `02_bugs_fixed/modal_deployment_gotchas.md` — Modal-side lessons from the LIBERO iteration arc
- The pending OpenPI port lives at `scripts/modal_libero_monolithic.py` (current) and will be replaced by a port-from-openpi next commit
