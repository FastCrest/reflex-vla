# 2026-04-20 — Fine-tuning pipeline v0.3 MVP ships

## Headline

**`reflex finetune` v0.3 MVP is shipped and working end-to-end.** First successful fine-tune → LoRA merge → ONNX chain landed at ~10:41 UTC on Modal (run `ap-459jwMLiJK0tif6neIKn7h`). Measured row in `reflex_context/measured_numbers.md` as `2026-04-20-finetune-e2e`.

## What shipped

### Research + architecture (pre-code)

5 markdown docs in `reflex_context/01_architecture/` totaling ~13k words:
1. `finetune_competitive_research.md` — 13 tools surveyed, top 10 customer pain points, 3 differentiation candidates (parity-gated / calibration-first / orthogonalized platform), contrarian takes including "NVIDIA ships one-click GR00T in a future Isaac release"
2. `finetune_sota_research.md` — per-model LoRA configs with citations (LoRA-SP arxiv 2603.07404 → VLA intrinsic rank is 4-16× LLM), LeRobotDataset v3 as de-facto winner (54.6% adoption), backend split (lerobot + openpi-JAX), VLM2VLA contrarian finding re action-as-language
3. `finetune_roadmap.md` — v0.3 / v0.5 / v1.0 horizons with explicit kill-gates, customer forks with quantified thresholds, "never sell GPU-minutes" monetization
4. `finetune_architecture.md` — 3 proposals evaluated; Proposal 2 (in-process trainer registry) picked with file-level module plan, 11 open product calls with defaults
5. `finetune_SYNTHESIS.md` — one-page decision doc tying the above four together; future sessions should read this first

Generated via 4 subagents run in phases (competitive + SOTA in parallel, then roadmap + architecture sequentially, consuming Phase 1 outputs).

### Code

**New package**: `src/reflex/finetune/` (replacing the single-file stub path):
- `__init__.py` — public API (`run_finetune`, `FinetuneConfig`, `FinetuneResult`)
- `config.py` — dataclasses
- `run.py` — orchestration: validate → invoke lerobot-train → locate checkpoint → merge LoRA → auto-export
- `cli.py` — typer command wired into `src/reflex/cli.py` at app registration time

**CLI surface**:
```bash
reflex finetune \
    --base lerobot/smolvla_base \
    --dataset lerobot/pusht \
    --output ./my_export \
    --steps 5000
```

**Modal scaffold**: `scripts/modal_reflex_finetune.py` — A10G default, writes to `pi0-onnx-outputs` volume so downstream serve / parity can pick up the ONNX without re-copying 1.5 GB over the network.

**Tests**: 22 unit tests in `tests/test_finetune.py` covering config validation, CLI command-build, checkpoint location, full orchestration via mocked subprocess + export, `policy.type` inference from base-model id.

## The 10-iteration Modal debug trail

Getting lerobot-train + LoRA adapter + reflex monolithic export aligned took 10 iterations. Each iteration revealed one more environmental / schema quirk. Log verbatim for future-session retrieval:

| # | Failure | Root cause | Fix |
|---|---|---|---|
| v1 | `unrecognized argument --precision=bf16` | lerobot 0.5.1 has no top-level precision flag | Dropped the arg; precision lives in policy config |
| v2 | `--policy.path` rejected | Flag doesn't exist in lerobot 0.5.1 schema | Changed to `--policy.pretrained_model_path` |
| v3 | `policy: Expected dict with 'type' key` | draccus requires `policy.type` to select the PreTrainedConfig subclass | Added `_infer_policy_type(base)` → `smolvla` / `pi0` / `pi05` / `gr00t_n1_5` |
| v4 | `pretrained_model_path not valid for SmolVLAConfig` | Field name wrong | Corrected to `--policy.pretrained_path` |
| v5 | `Output directory already exists` | lerobot wants to own output_dir creation | Reflex root now owns `cfg.output/`, lerobot gets `cfg.output/training/` (fresh) |
| v6 | `policy.repo_id argument missing` | lerobot validates hub repo_id even when not pushing | Pass `--policy.repo_id=local/<name>` + `--policy.push_to_hub=false` |
| v7 | `libavutil.so.56 not found` (torchcodec) | Modal debian_slim missing ffmpeg runtime libs | `apt_install("ffmpeg", "libavutil-dev", ...)` |
| v8 | TRAINED OK (200 steps, loss 0.688, 36s); auto-export failed | reflex-vla installed without `[monolithic]` extra | `pip install 'reflex-vla[monolithic] @ git+...'` |
| v9 | Training OK; export failed on `No such file: model.safetensors` | LoRA checkpoint writes `adapter_model.safetensors` only (no base weights) + `_locate_checkpoint` picked the `last` symlink | New `_merge_lora_adapter()` helper uses peft to merge adapter into base, copies processors, returns merged dir; `_locate_checkpoint` now prefers numeric step dirs over `last` |
| **v10** | **STATUS OK** | — | End-to-end: 200 training steps → merge_and_unload → `export_smolvla_monolithic` → `model.onnx` (15.5 MB graph + 1.5 GB .data) + `VERIFICATION.md` auto-seeded |

## v10 measured result (landed)

```
status: ok
output_dir: /onnx_out/finetune_smolvla_pusht_v10
training_steps_completed: 200
final_checkpoint_path: .../checkpoints/000200/pretrained_model
onnx_path: .../export/model.onnx       (15.5 MB + 1.5 GB .data)
verification_md_path: .../export/VERIFICATION.md
training_log_path: .../training_log.jsonl
error: None
```

Training loss at step 200: 0.688, gradient norm 0.796, lr 5e-05, ~5.5 step/s on A10G. Wall time ~12 min total (image build + training + merge + export).

## What v0.3 MVP does NOT do (intentional, deferred to v0.5)

Per the synthesis doc, v0.3 is SmolVLA LoRA ONLY:
- No pre-flight schema validator (catches top-10 pain points like the shape-mismatch-6-vs-7 issue before GPU time)
- No parity-gate (cos=+1.0 check at checkpoint saves)
- No calibration-first eval (ECE / Brier / NLL on held-out data as stopping signal)
- No pi0 / pi0.5 / GR00T support (raises `NotImplementedError` with a pointer to `reflex finetune` v0.5 in v0.6+)
- No openpi-JAX backend
- No custom optimizers / auxiliary losses / pluggable action heads

All are architecturally hooked for (plugin registries already defined in `finetune_architecture.md` Section D); v0.5 wires them.

## Honest caveats captured in measured_numbers row

- **Parity vs pre-fine-tune SmolVLA not yet measured** — pusht is single-cam single-task with 200 steps, so loss 0.688 doesn't tell us anything about task transfer. Only the structural soundness of the chain is proven.
- **Modal image build overhead** — first run after Dockerfile change costs ~5 min of apt+pip. Steady-state finetune is ~1-2 min per 200 steps.
- **Backend locked to lerobot-train via subprocess** — if lerobot 0.6 renames flags we re-hit the v1-v6 debug cycle. Thin orchestrator approach (competitive research "don't rewrite gradient math") keeps us honest but exposes us to upstream churn.

## What commits landed this session (chronological)

Most relevant:
- `fd8d47b` — 5 research docs + SYNTHESIS committed pre-code
- `b17e56d` — v0.3 MVP code (finetune package + CLI + 19 tests + Modal scaffold + GOALS.yaml done marker)
- `60ac16c` / `010dcd2` / `1dabd80` / `bb24177` / `2d9ce53` / `315f4d2` / `3ffc06a` / `c4911d9` / `bc44f13` — nine iteration fixes
- `f19fd0c` — measured_numbers row for v10 end-to-end success

Total LOC delta (finetune feature only): ~1100 lines added across 8 files.

## What's next

Per the roadmap doc `finetune_roadmap.md`:

- **v0.5 (3 months)**: pre-flight validator, parity-gate, calibration-first eval, pi0 LoRA support, Pro tier monetization gate
- **v1.0 (conditional, 6-12 months)**: pluggable action heads, multi-backend (lerobot + openpi-JAX), orthogonalized platform matrix — with three kill-gates (cell-count <10, zero external PRs, NVIDIA ships one-click GR00T)

Immediate polish items (not part of v0.3 goal but nice to land):
- Real parity measurement: fine-tune vs pre-fine-tune cos sim on shared seeded inputs (would be first "fine-tune preserves cos=+1.0" number)
- A dataset that has 2+ cams so the 3rd-cam-padding code path gets exercised (pusht is single-cam)
- Move off Modal iteration for this feature — once CLI schema is stable, fine-tune can run locally on H100 / A100 for more granular testing

## Related

- `reflex_context/01_architecture/finetune_SYNTHESIS.md` — the one-page product decision
- `reflex_context/01_architecture/finetune_architecture.md` — the canonical design, file-level module plan
- `reflex_context/measured_numbers.md` — line `2026-04-20-finetune-e2e`
- `GOALS.yaml` — `fine-tuning-pipeline` marked `status: done`
- `scripts/modal_reflex_finetune.py` — the Modal reproducer
- `src/reflex/finetune/` — the actual package
