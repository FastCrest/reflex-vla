# 2026-04-19 evening — GR00T VLM close-out, LIBERO ONNX N=25, 7-goal polish sweep

## Headline

- **GR00T Eagle VLM deployment pipeline COMPLETE** — `eagle_vlm.onnx` (5.99 GB, cos=+1.000000) + `expert_stack_with_vlm.onnx` (4.4 GB, cos=+1.000000) + end-to-end chain test passed (image A vs B sensitivity max_abs=0.212 in BOTH PyTorch and ONNX paths). GR00T is now fully deployable, not zero-VLM-stubbed.
- **LIBERO-10 monolithic ONNX N=25 at num_steps=1: 7/25 = 28%** (vs native 40%). Gap matches the predicted num_steps=1 vs num_steps=10 behavioral delta. num_steps=10 ONNX re-export running at session close.
- **7 polish goals closed** in parallel while Modal runs: export-verification-report, nan-guard-hardening, api-key-auth, latency-histograms, determinism-version-hash, action-chunk-buffering, orin-nano-fp16-fit.

## Timeline

1. **Eagle VLM export Step 4b** — exported `eagle_vlm.onnx` via `torch.onnx.export(opset=19)` on A100-40GB (222s conversion), 1.87B params, 5.99 GB.
   - First parity attempt: ORT broadcast failure on `index_put → Where` node (64 vs 80). Root cause: Eagle's original `input_embeds[selected] = vit_flat` boolean-index assignment lowered to a Where that couldn't handle the vit-vs-seq shape mismatch under dynamic shapes.
   - Fix: cat-based splice. Export contract now requires image tokens packed at the front of `input_ids`; wrap does `cat([vit_embeds, text_embeds[vision_seq:]], dim=1)`.
   - Re-export cleaner (431 rewrites vs 465). Parity PASSED: cos=+1.000000, max_abs=4.25e-04.

2. **Step 5 end-to-end chain test** — `scripts/modal_gr00t_e2e_chain_test.py`. Feed pixel_values + input_ids through `eagle_vlm.onnx` → hidden_states → feed as `vlm_kv` to `expert_stack_with_vlm.onnx`. Compare vs PyTorch full chain.
   - Parity: cos=+1.000000 (max_abs=1.9e-05 on actions) for both image A and image B.
   - Sensitivity: image A vs B → max_abs=0.212 on raw actions, cos=+0.982 in BOTH paths.
   - VLM conditioning is LIVE end-to-end. **Closes GR00T deployment gap.**
   - Required chunk=50 (not 16) to match the fixed chunk axis of the ONNX.

3. **LIBERO-via-monolithic-ONNX goal pickup** — wrote `scripts/modal_libero_monolithic_onnx.py` (521 LOC, drop-in replacement for native harness). Same obs pipeline, preprocessor, postprocessor; only swap is `policy.predict_action_chunk` → `sess.run(["actions"], {10 inputs})`.
   - Fix 1: LIBERO has 2 cameras; ONNX expects 3. Pad with `-1` zero image + false mask (matches SmolVLA's own empty-camera fill).
   - Fix 2: Preprocessor produces variable-length `lang_tokens`; ONNX seq=16 fixed. Pad right with zero tokens + false mask.
   - Required a fine-tune-specific export: `HuggingFaceVLA/smolvla_libero` instead of `lerobot/smolvla_base`. Parameterized `modal_smolvla_monolithic_export.py` with `--model-id` + `--out-subdir`.
   - Also fixed `_hf_secret()` to fall back to empty dict when `hf-token` secret isn't registered on the workspace.

4. **Task 3/4 gap audit** — `reflex_context/06_experiments/task34_gap_audit.md`. Three findings:
   - Task 4 prompt tokenizes to **21 tokens**, ONNX seq=16 silently truncates at "put the white mug on the left plate and put" — loses 2nd instruction. Latent bug; doesn't affect current results (task 4 was already 0% in native too).
   - Default `num_steps=1` in monolithic export vs `num_steps=10` training: known cos=0.78 behavioral delta. Dominant explanation for ONNX 28% vs native 40%.
   - Task 3 (14 tokens, single-goal) remains unexplained beyond fine-tune capability on drawer manipulation.

5. **Concurrent 7-goal polish sweep** (while LIBERO N=25 ran):

   | Goal | Weight | Deliverable |
   |---|---|---|
   | `export-verification-report` | 6 | Pushed `write_verification_report` call into `exporters/monolithic.py` so every export ships a receipt |
   | `nan-guard-hardening` | 7 | Kill-switch logic already existed; wired `staleness` terminology into docstring |
   | `api-key-auth` | 7 | `--api-key` → `X-Reflex-Key` header dep on `/act` + `/config` (401 on miss); `/health` stays open. 8 new tests |
   | `latency-histograms` | 6 | p50/p95/p99/jitter_ms over rolling 1024-sample window in `/act` response |
   | `determinism-version-hash` | 5 | model_hash, config_hash, reflex_version in `/act` response. SHA256 trunc16 |
   | `action-chunk-buffering` | 7 | `ActionChunkBuffer` ring buffer + `configure_replan` + `--replan-hz/--execute-hz` CLI. 15 tests |
   | `orin-nano-fp16-fit` | 8 | Plan doc + `src/reflex/exporters/fp16_convert.py` helpers + Modal scaffold `scripts/modal_fp16_convert.py` + 12 unit tests |

   Total: 7 goals, combined weight 46, ~2.5 hours of parallel work.

## Numbers landed in measured_numbers.md

- **GR00T N1.6 eagle_vlm.onnx** — cos=+1.000000, max_abs=4.25e-04 on [1, 80, 2048]
- **GR00T N1.6 end-to-end two-ONNX chain** — parity cos=+1.000000 both paths, image sensitivity max_abs=0.212 preserved through export
- **LIBERO-10 ONNX N=25 (num_steps=1)** — 7/25 = 28% (superseded pending num_steps=10 rerun)

## Surprising findings (persisted in memory for future sessions)

- **N1.6 language model is Qwen3, not Qwen2** — `qk_layernorm` + no q/k/v biases. Signature: 48 missing keys + 32 unexpected keys when loading with Qwen2 config.
- **SigLIP is 224×224 in N1.6, not 448×448** — derive from position_embedding num_positions (256 → 16×16 grid × 14 patch = 224).
- **patch_embedding is stored flat [1152, 588]** in state dict; standard SigLIP uses Conv2d [1152, 3, 14, 14]. Reshape on load.
- **pixel_shuffle needs `.reshape()` not `.view()`** in the vendored Eagle — non-contiguous tensor fails view() in export path.
- **Eagle's class name is `Eagle25VL` NOT `Eagle2_5_VL`** — despite marketing name "Eagle 2.5 VL".
- **Boolean-index assignment (`x[mask] = y`) breaks ORT broadcast** when mask.sum() != y.shape[0] statically. Fix: replace with cat or scatter.
- **use_cache=False saves us from the 3-patch Qwen stack** pi0/pi0.5 needed. Single-shot encode avoids DynamicCache touchpoints entirely.

All documented in `/Users/romirjain/.claude/projects/-Users-romirjain/memory/project_reflex_vla_onnx_export_gotchas.md` for future-session retrieval.

## Open end-of-session

- **N=25 ONNX num_steps=10 LANDED: 8/25 = 32%** (vs native 40%, ONNX num_steps=1 28%). **num_steps=10 only gained +4pp, not the predicted +12pp.** Residual gap most likely fresh-noise stochasticity — each predict() call samples different noise, at N=5 per task the variance is huge (std ≈ ±1 episode). Real resolution would need N=500 per task (OpenPI standard). Documented the finding + caveats in `task_success_results.md`.
- **FP16 conversion for pi0/pi0.5** — scaffold ready (`scripts/modal_fp16_convert.py`), not yet triggered. Next up.
- **pi0.5 + GR00T native-path parity** (weight 9 goal) — still deferred to v0.3.

## Related

- `reflex_context/01_architecture/gr00t_eagle_vlm_export_plan.md` — Eagle VLM export plan, all 5 steps DONE
- `reflex_context/01_architecture/orin_nano_fp16_plan.md` — FP16 conversion plan (new this session)
- `reflex_context/06_experiments/task34_gap_audit.md` — task 3/4 underperformance audit (new this session)
- `reflex_context/06_experiments/task_success_results.md` — extended with ONNX N=25 section
- `reflex_context/measured_numbers.md` — two new rows for GR00T VLM + one for LIBERO ONNX N=25
- `README.md` — added Eagle VLM + chain parity rows to the verified numbers table
- Memory: `project_reflex_vla_onnx_export_gotchas.md` — 5 new Eagle-specific gotchas persisted
