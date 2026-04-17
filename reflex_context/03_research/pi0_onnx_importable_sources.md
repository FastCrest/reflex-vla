# pi0-onnx-parity: importable code sources + action plan

**Compiled:** 2026-04-17
**Goal:** end-to-end pi0 ONNX export with cos ≥ 0.999 vs reference PyTorch `policy.predict_action_chunk` on shared noise. Tracks GOALS.yaml `pi0-onnx-parity` (weight 9, MVP).

**Key finding:** multiple production-grade pi0 ONNX+TRT implementations already exist in the open-source ecosystem. Revised estimate: **1.5–3 weeks** (vs. 3–5 weeks writing from scratch).

This doc replaces the "write from scratch" plan. Next session should consult this first before writing new exporter code.

---

## Tier 1 — drop-in / near drop-in replacements (license-gated)

These sources have components that **REPLACE** code in our plan, not just augment. Biggest time saves. Each has a license risk that must be resolved before commercial import.

### 1. Jetson AI Lab `openpi_on_thor` — the silver bullet

- **URL:** https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- **What's in the bundle:**
  - `pytorch_to_onnx.py` — pi0.5 → ONNX conversion
  - `patches/apply_gemma_fixes.py` — PaliGemma/Gemma quirks patched (gold)
  - `build_engine.sh` — trtexec invocation for pi0.5
  - `trt_model_forward.py` + `trt_torch.py` — TRT engine inference wrappers
  - `calibration_data.py` — INT8/FP8 calibration harness
  - `pi05_inference.py` — full end-to-end inference script
  - `serve_policy.py` — server wrapper
  - `thor.Dockerfile` — deployment container
- **Proven results:** cos > 0.99, 1.71× speedup over PyTorch
- **Precision:** FP8 + NVFP4 on Thor
- **License:** NOT STATED — blocker for commercial reflex-vla import. **Must contact NVIDIA DevRel for clarification.**
- **If permissive:** saves ~15–20 engineer-days.
- **Integration risk:** MEDIUM
  - Targeted at Thor (FP8-capable SM 10.0); Orin Nano is SM 8.7 and rejects FP8 Q/DQ at runtime
  - Must port FP8 → FP16 for Orin target
  - Scripts use pi0.5 (AdaRMSNorm variant); we want plain pi0
  - Gemma patches are still applicable — extract `apply_gemma_fixes.py` regardless

### 2. `Tacoin/openpi-pi0.5-libero-onnx` — MIT-licensed parity oracle

- **URL:** https://huggingface.co/Tacoin/openpi-pi0.5-libero-onnx
- **What it gives us:** pre-exported ONNX of pi0.5-LIBERO as a unified graph. Includes ORT+TRT EP integration snippets.
- **Variants:** W8A16, FP16, W4A4 (W4A4 is experimental quantization)
- **License:** MIT (✓ safe for commercial)
- **Use case:** **parity oracle**. Run our decomposed graph and this monolithic graph at matching shared noise; diff outputs. Any discrepancy is a bug in our pipeline, since Tacoin's is validated.
- **Saves:** 3–5 days on reference-generation infrastructure.
- **Integration risk:** LOW
  - Note: pi0.5 ≠ pi0. The models are related but have different action heads. Can only cross-check against our pi0.5 work (post-MVP).
  - For pi0 (MVP target), use as architectural reference only.

### 3. `NVIDIA/Isaac-GR00T/scripts/deployment/` — pattern source

- **URL:** https://github.com/NVIDIA/Isaac-GR00T/tree/main/scripts/deployment
- **What's there:**
  - `build_trt_pipeline.py`
  - `export_onnx_n1d7.py`
  - `trt_model_forward.py`, `trt_torch.py`
  - `verify_n1d7_trt.py` — **verification harness we can fork**
  - `benchmark_inference.py`
- **Gold patterns:**
  - `DiTInputCapture`, `LLMInputCapture` — hooks that record real inputs during a forward pass, then replay as traced inputs during ONNX export (avoids dummy-input shape mismatches)
  - `_apply_rotary_real` — complex-free RoPE implementation that exports cleanly (ours uses complex numbers via sin/cos; this is an alternative)
  - `_simple_causal_mask` — simple causal mask that exports cleanly
  - `_make_onnx_vision_attention_forward` — chunked attention wrapper for large attention matrices
- **License:** UNCLEAR — check repo root LICENSE before copying. If NVIDIA's "Isaac License" or similar, may be restrictive.
- **Saves:** 7–10 days (verify infra, rotary patterns, dynamic-axes for vision attention).
- **Integration risk:** MEDIUM — tightly coupled to GR00T's N1.7 architecture but extraction is mechanical.

### 4. HuggingFace Optimum PaliGemma2 ONNX — 4-file split reference

- **URL:** https://github.com/huggingface/optimum-onnx
- **Reference export:** `onnx-community/paligemma2-3b-pt-224` (on HF Hub)
- **What it gives us:** the exact 4-file decomposition our plan targets:
  - `embed_tokens.onnx` — text token embeddings
  - `vision_encoder.onnx` — SigLIP
  - `decoder_model_merged.onnx` — prefill + decode unified
  - `decoder_with_past.onnx` — decode-only (with KV cache input)
- **License:** Apache-2.0 (✓ safe for commercial)
- **Saves:** 4–6 days on decomposition plumbing.
- **Integration risk:** LOW for SigLIP + Gemma-decoder components. MEDIUM for the "image-text-to-text" task integration.
- **Known gap:** Optimum does NOT officially support the `image-text-to-text` task for PaliGemma. Open issue: https://discuss.huggingface.co/t/paligemma2-onnx-export-keyerror-unknown-task-image-text-to-text/139726
  - Workaround: `@register_for_onnx` custom-config pattern. We write a small PaliGemmaOnnxConfig subclass (~100 lines) — still much less than writing the whole exporter.

---

## Tier 2 — confirmed-safe library code + tooling (Apache-2.0 / MIT)

These are unambiguously safe to import. Use liberally.

### 5. `lerobot/src/lerobot/policies/pi0/modeling_pi0.py`

- **URL:** https://github.com/huggingface/lerobot
- **License:** Apache-2.0
- **What to reference:**
  - `PI0Policy.predict_action_chunk` — reference forward
  - `PI0Pytorch.sample_actions` — flow-matching loop
  - `PI0Pytorch.embed_prefix` + `embed_suffix` — how tokens become embeddings
  - `PI0Pytorch.denoise_step` — single expert forward
  - `PaliGemmaWithExpertModel` — dual-expert wrapper (our direct target)
- **Confirms:** the right seams exist for ONNX extraction. Already in our venv.

### 6. NVIDIA TensorRT-LLM `examples/models/core/gemma`

- **URL:** https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma
- **License:** Apache-2.0
- **What's there:** `convert_checkpoint.py` handles HF/JAX/Torch/Keras checkpoints → TRT-LLM. Quantization recipes: FP8, NVFP4, INT4-AWQ, SmoothQuant, INT8 KV-cache.
- **Saves:** 2–3 days on Gemma quantization recipes (relevant for v0.3 distill-dmpo).
- **Integration risk:** LOW for weight conversion; HIGH if we need TRT-LLM runtime (heavy dep — avoid unless v0.3+).
- **Gap:** no PaliGemma branch — Gemma decoder only.

### 7. `Dexmal/realtime-vla` (arxiv 2510.26742)

- **URL:** https://github.com/Dexmal/realtime-vla
- **License:** MIT
- **What it gives:** pi0/pi0.5 at 30 Hz via CUDA-graph + fused Triton kernels. Includes:
  - `convert_from_jax.py` + `convert_from_jax_pi05.py` — JAX → pickle helpers (validated ingest)
  - Reference 24-GEMM decomposition of pi0 inference (performance upper-bound)
- **Saves:** 1–2 days if we need JAX ingest (unlikely since we use lerobot's PyTorch port, but nice to have).
- **Not an ONNX path.** Useful as performance benchmark reference.

### 8. `allenzren/open-pi-zero`

- **URL:** https://github.com/allenzren/open-pi-zero
- **License:** MIT
- **What it gives:** second PyTorch reference implementation for pi0 on PaliGemma 2B. Block-causal attention + flow-matching patterns.
- **Integration risk:** MEDIUM — not exact architectural match; use for pattern reference only.

### 9. `ainekko/smolvla_base_onnx`

- **URL:** https://huggingface.co/ainekko/smolvla_base_onnx
- **License:** Apache-2.0
- **What it gives:** 9-file decomposition of SmolVLA matching exactly our `vlm_prefix_exporter.py` split (vision, text, expert_prefill, expert_decode, state_projector, time_in/out, action_in/out). **Public validation that our approach is correct.**
- **Use case:** sanity cross-check our SmolVLA export. Pattern-transfers to pi0.

### 10. `onnx-community/siglip2-so400m-patch14-384-ONNX`

- **URL:** https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX
- **License:** Apache-2.0
- **What it gives:** pre-exported 1152-dim SigLIP2 so400m (matching pi0's vision dim)
- **Integration risk:** MEDIUM — verify pi0's SigLIP variant has matching weights (patch size 14 vs pi0's need-to-check).
- **If compatible:** saves 1 day on vision encoder export.

### 11. `transformers.StaticCache` + ONNX dynamic-cache docs

- **URLs:**
  - https://huggingface.co/docs/transformers/kv_cache
  - PyTorch issue #136582 (DynamicCache exportability discussion)
  - `sdpython/onnx-diagnostic` (dynamic-shape guessing)
- **License:** Apache-2.0 (transformers)
- **What it gives:** `StaticCache` is the ONNX-exportable path (pre-allocated, fixed-size). Avoids DynamicCache Python-object-serialization issues.
- **Saves:** 1–2 days of trial-and-error on prefix-KV I/O shape.

### 12. `aifoundry-org/ETARS`

- **URL:** https://github.com/aifoundry-org/ETARS
- **License:** NOT STATED (check before using)
- **What it gives:** ONNXRuntime inference backend for SmolVLA. Good runtime-layer reference.
- **Saves:** 0.5–1 day if license permits.

### 13. `onnx-community/paligemma2-3b-*` ONNX exports

- **URLs:** `paligemma2-3b-pt-224`, `paligemma2-3b-ft-docci-448`, `paligemma2-3b-pt-448/896` on HF Hub
- **License:** per base model (Gemma + PaliGemma terms — not fully open for all commercial uses). Check Gemma terms before commercial reflex-vla ship.
- **Use case:** reference diffing (component-by-component parity check).
- **Saves:** 1–2 days if licensing clears.

---

## Tier 3 — knowledge / patterns only (no direct code reuse)

### 14. `lucidrains/pi-zero-pytorch`
Simplified pi0 implementation. MIT. Not architecturally exact. SKIP for code; reference for concepts.

### 15. NVIDIA Isaac GR00T N1.6/N1.7 model cards + deployment blog
- **URL:** https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow
- **Critical takeaway:** FP8-on-Orin is NOT supported (Q/DQ rejected at runtime). Must use FP16 for Orin target. This contradicts our MVP assumption that we can copy Thor's FP8+NVFP4 recipe.

### 16. `D-Robotics/rdk_LeRobot_tools`
ACT-only, not pi0. SKIP.

### 17. Papers with code
- `2507.14049` EdgeVLA — concepts only, no exports
- `2510.23511` Dexbotic — concepts only, no exports

---

## Critical risks surfaced by research

These are MVP-relevant gotchas that weren't on my radar before this research:

1. **Orin rejects FP8 Q/DQ at runtime** — Thor's FP8+NVFP4 pipeline WON'T port 1:1 to Orin Nano. **Must use FP16 on Orin target.** Per NVIDIA GR00T team's own deployment guide.

2. **Optimum does NOT officially support image-text-to-text for PaliGemma** — confirmed open issue. We WILL need to write a custom `OnnxConfig` subclass. No way around it.

3. **Gemma3 has `torch.onnx.export` vmap failure** — PyTorch issue #160761. **Stay on Gemma2.** PaliGemma2 uses Gemma2 which works; confirmed safe.

4. **PI's open ONNX weights are pi0.5-native** — most community exports target pi0.5 (AdaRMSNorm version). Our MVP wants plain pi0 (simpler). Trade-off:
   - pi0 is architecturally simpler (no AdaRMSNorm) — easier for us to hand-write
   - No drop-in ONNX reference — we're the trailblazer for pi0 ONNX
   - Tacoin's pi0.5 reference is close but not a direct oracle

5. **License unknowns block commercial adoption of Tier-1 silver bullets.** The `openpi_on_thor` bundle and Isaac-GR00T deployment scripts are the biggest time savers but both have unclear license terms. Resolve first before heavy integration.

---

## Revised timeline estimate

**With Tier-1 imports (all permissive):** ~1.5–2 weeks
**With Tier-2 only (if Tier-1 blocked):** ~2.5–3 weeks
**Writing from scratch (pre-research baseline):** 3–5 weeks

**Mess-up probability:**
- Best case (all tiers): ~35%
- Middle case (Tier-2 only): ~40%
- Worst case (bugs hunt dominates): ~50%

Decrease vs. pre-research baseline (~60%) driven by:
- `apply_gemma_fixes.py` — patches already made and proven
- `onnx-community/paligemma2-3b-pt-224` — known-good 4-file decomposition
- Tacoin MIT ONNX — parity oracle available for pi0.5 cross-checks
- `_apply_rotary_real` — ONNX-clean RoPE pattern validated in GR00T
- `StaticCache` — documented ONNX-exportable cache type

---

## 7-day action plan

| Day | Task | Dependency |
|---|---|---|
| 0 (today) | Pull `Tacoin/openpi-pi0.5-libero-onnx` (MIT) to `./_oracles/pi05_libero_onnx/`. Verify loads. | HF Hub access |
| 0 (parallel) | Email NVIDIA DevRel requesting license clarity on `openpi_on_thor` + `Isaac-GR00T/scripts/deployment/` for commercial reflex-vla use. | NVIDIA contact |
| 1 | Run `optimum-cli export onnx --model google/paligemma-3b-pt-224 ./_optimum_sanity/` to confirm baseline PaliGemma export (will fail on task but should produce components). | optimum-onnx installed |
| 2–3 | Fork `src/reflex/exporters/vlm_prefix_exporter.py` → `src/reflex/exporters/pi0_prefix_exporter.py`. Swap SmolVLM2→PaliGemma2, SmolLM2→Gemma2, update state-dict prefixes. Use Optimum's 4-component split as structural guide. | Day 1 sanity passes |
| 4–6 | Integrate patterns: Isaac-GR00T's `_apply_rotary_real`, `_simple_causal_mask`, `DiTInputCapture` hooks. Extract `apply_gemma_fixes.py` lessons (patch-level bug fixes). License-clean rewrite if needed. | Day 1 NVIDIA response (ideal) |
| 7 | First end-to-end parity diff: export pi0 full pipeline, run with shared noise, compare vs real PyTorch pi0. Expect cos=0.X (SmolVLA precedent). Record result. | Days 2–6 complete |
| 8+ | Bug hunt using established diagnostic ladder (stage-diff → single-layer → full). Document each bug in `reflex_context/02_bugs_fixed/pi0_pipeline_bugs.md`. | Day 7 baseline diff |

---

## Sources index

- Jetson AI Lab OpenPi on Thor: https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- Tacoin/openpi-pi0.5-libero-onnx: https://huggingface.co/Tacoin/openpi-pi0.5-libero-onnx
- NVIDIA Isaac-GR00T deployment: https://github.com/NVIDIA/Isaac-GR00T/tree/main/scripts/deployment
- HuggingFace optimum-onnx: https://github.com/huggingface/optimum-onnx
- onnx-community/paligemma2-3b-pt-224: https://huggingface.co/onnx-community/paligemma2-3b-pt-224
- lerobot pi0 source: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py
- NVIDIA TensorRT-LLM Gemma: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma
- Dexmal/realtime-vla: https://github.com/Dexmal/realtime-vla
- allenzren/open-pi-zero: https://github.com/allenzren/open-pi-zero
- ainekko/smolvla_base_onnx: https://huggingface.co/ainekko/smolvla_base_onnx
- onnx-community/siglip2-so400m-patch14-384-ONNX: https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX
- PaliGemma2 ONNX issue: https://discuss.huggingface.co/t/paligemma2-onnx-export-keyerror-unknown-task-image-text-to-text/139726
- GR00T N1.6 blog (FP8/Orin constraint): https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow
- transformers KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- aifoundry-org/ETARS: https://github.com/aifoundry-org/ETARS
