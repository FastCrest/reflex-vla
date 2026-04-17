# archive/

Code and scripts retired from the active reflex-vla codebase but preserved for provenance. Not shipped to users, not imported from `src/reflex/`, not maintained.

Resurrect at your own risk — the rest of the codebase has evolved since these were archived.

## Contents

### 2026-04-17 — LIBERO sim benchmarking

Archived because reflex's product wedge is **deployment parity + latency**, not sim benchmarking. LIBERO task-success numbers:
- Don't measure what customers buy (they care about parity vs their PyTorch model and latency on their Jetson).
- Required ~1–2 weeks of dep-conflict yak-shaving per quarter (LIBERO-2023 stack vs lerobot-2026 stack).
- Produced zero verified task-success numbers despite 6+ hours of Modal debugging across ~5 runs.

| File | Was at | Purpose |
|---|---|---|
| `scripts/modal_libero10.py` | `scripts/modal_libero10.py` | 602-line Modal A10G runner: LIBERO+robosuite+bddl+mujoco+lerobot install, patch LIBERO's `input()` prompts, run vla-eval harness |
| `scripts/modal_sim_test.py` | `scripts/modal_sim_test.py` | Earlier Modal sim smoke test |
| `scripts/sim_smoke_test.py` | `scripts/sim_smoke_test.py` | Local MuJoCo smoke test |
| `scripts/patch_libero.py` | `scripts/patch_libero.py` | Regex patch for LIBERO's interactive `input()` calls that hang in non-TTY subprocess contexts |
| `src/reflex/eval/libero.py` | `src/reflex/eval/libero.py` | v0.2 stub that dispatched `reflex bench --benchmark libero_10` to the Modal script |

## References

- Decision: `reflex_context/measured_numbers.md` — the "what can we credibly claim" ledger that drove this archive.
- Related: `reflex_context/02_bugs_fixed/libero_integration.md` — the bugs found during the install death march (kept active for future reference).
- Related: `reflex_context/05_sessions/2026-04-17_libero_correctness_hunt.md` — the session narrative.
