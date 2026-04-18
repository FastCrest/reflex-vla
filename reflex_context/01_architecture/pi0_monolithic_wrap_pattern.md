# pi0 monolithic ONNX wrap pattern (Path C)

**Discovered:** 2026-04-18 via deep research (agent synthesis of openpi_on_thor, Tacoin, Dexmal).

**Why this doc exists:** our first two attempts at pi0 ONNX export (custom ExpertStack + concat prefix_kv; Optimum with past_kv as tensor inputs) both hit fundamental blockers. Every production pi0/pi0.5 ONNX in the open-source world uses the same third pattern — monolithic wrap. Writing this down so next session doesn't re-discover.

## The pattern

Wrap pi0's `sample_actions` in a thin `nn.Module` whose `forward(images, img_masks, lang_tokens, lang_masks, state, noise)` calls the full policy forward end-to-end. `past_key_values` stays as a Python variable **inside** the traced graph — it never becomes an ONNX input. The 10-step Euler flow-matching loop is **unrolled** at trace time.

```python
class Pi0MonolithicWrapper(nn.Module):
    def __init__(self, pi0_model, num_steps=10):
        super().__init__()
        self.model = pi0_model
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        return self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=self.num_steps,
        )

torch.onnx.export(
    wrapper, dummy_inputs, onnx_path,
    dynamo=False,           # MANDATORY — dynamo broken for Gemma in 2026
    opset_version=19,       # NOT 23; TRT parser compat
    do_constant_folding=True,
    input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
    output_names=["actions"],
)
```

## Why this works

1. **`dynamo=False` legacy tracer** traces Python-level control flow (loops, conditionals, cache ops) into tensor ops. DynamicCache.update → `torch.cat(past_k, new_k)` nodes baked into the graph.

2. **past_kv never crosses the ONNX boundary**, so we never trip the 3 unfixed PyTorch bugs:
   - pytorch#160761 — Gemma3 vmap in masking_utils
   - pytorch#170172 — dynamo batch=1 hardcode
   - pytorch#172903 — DynamicCache "NameError: 'L' not defined"

3. **Denoise loop unrolls** at `num_steps=10` at trace time — inspecting the graph shows 10× expert forward calls with growing attention matrices.

4. **Uses pi0's own tested forward code.** We inherit every correctness property (normalizer, attention masks, position IDs, state projection, block-causal pattern). No re-implementation bugs.

## Required Gemma source patches

Per openpi_on_thor's `apply_gemma_fixes.py`, two patches are needed (reimplement, don't copy — Thor bundle has no license):

### 1. `GemmaRMSNorm.extra_repr` hasattr guard

Issue: during trace, `extra_repr` may be called on a partially-initialized module → `AttributeError`. 

Fix: guard with `hasattr(self, 'weight')` before accessing.

### 2. `GemmaAttention.forward` explicit dim reshape

Issue: `hidden_shape = (*input_shape, -1, self.head_dim)` uses `-1` which makes that dim dynamic → breaks TRT FP4 block quantization.

Fix: replace `-1` with explicit `num_attention_heads` (for q_proj reshape) or `num_key_value_heads` (for k/v_proj reshape).

Both patches are monkey-patches; no source edits needed to transformers.

## Pros

- **Single ONNX file** — simpler deployment, fewer engine builds on Jetson
- **Proven working** — Tacoin, openpi_on_thor, Dexmal all produce pi0/pi0.5 ONNX this way
- **Inherits pi0's correctness** — no re-implementation bugs from Path A
- **No DynamicCache battles** — past_kv lives in Python vars, not as ONNX inputs
- **Low dev time** — ~50-100 lines of wrapper code + 2 patches

## Cons

- **~14GB monolithic ONNX** — single file with external data. Fine for production but debugging is harder (can't diff per-stage)
- **Fixed num_steps at export time** — changing denoise steps requires re-export. Acceptable for pi0 (10 steps canonical)
- **Larger TRT engine build time** — one big graph vs 5 smaller ones
- **License care** — Thor bundle has no license file. Pattern is public knowledge; implementation must be clean-room

## Trade-off vs our 5-file decomposition

Our existing decomposition (vision_encoder, multi_modal_projector, text_embedder, decoder_prefill, expert_stack) is still valuable:
- Per-stage parity diffing (we used this heavily today)
- Runtime flexibility (can replace individual stages without re-exporting)
- Smaller per-file sizes for debugging

Keep both. Use monolithic for **deployment** (production MVP), use 5-file for **debugging + research**.

## What the ONNX graph looks like (from openpi_on_thor inspection)

Single graph with ~10× unrolled denoise steps. Attention concat ops for past_kv extension baked in as `Concat` nodes. ~14GB total with external data (Gemma-2b backbone weights are the bulk).

## Current ship status (2026-04-18)

**num_steps=1 monolithic ONNX: VERIFIED at cos=+1.0000000 vs PyTorch.** Reproducer: `scripts/modal_pi0_monolithic_export.py --parity`. Artifact: `/onnx_out/monolithic/model.onnx` in Modal Volume `pi0-onnx-outputs` (~12.5GB).

**num_steps=10 monolithic: BLOCKED** on `RuntimeError: expand: attempting to expand a dimension of length 835 -> 886!` raised by onnx-diagnostic's `patched__maybe_broadcast` during torch.export shape tracing. 835 is prefix_len + state=1; 886 is prefix_len + state + chunk. Some expand op inside the unrolled 10-step loop can't reconcile the pre-suffix vs post-suffix shape. Unresolved as of this write.

**Runtime mechanics (IMPORTANT — correcting an earlier mistake):** The num_steps=1 monolithic ONNX is NOT a "single denoise step you can host-loop." It's a full `sample_actions(num_steps=1)` call — computes prefix + runs 1 big Euler step with dt=-1.0 + returns x_final. Calling it N times from Python gives N identical outputs (deterministic). To actually achieve num_steps=10 semantics, you need either:

1. **Re-export at num_steps=10 directly** (blocked on the 835→886 bug above).
2. **Per-step ONNX** (denoise_step as standalone, takes x_t + timestep + past_kv as inputs). Past_kv-as-ONNX-input fails torch.export's DynamicCache tracer (see `03_research/pi0_onnx_importable_sources.md` critical risks #7-9).
3. **Multiple monolithic exports at varying num_steps** (each ~7 min on Modal, each bakes a fixed num_steps). Viable for a small N set (1, 5, 10, 20).

**Current customer-facing ship: num_steps=1.** Lower quality than pi0's default but numerically verified. For v0.3 we'll land at least one of the three upgrade paths above.

## Follow-up goals (tracked)

- **pi0-onnx-parity-multistep** — land num_steps=10 (or dynamic N) parity. Either fix the 835→886 expand bug OR export per-step + solve DynamicCache tracer issue OR bake multiple fixed-N artifacts.
- **Distribution** — publish the 12.5GB ONNX to HF Hub once customers exist (not before — versioned artifact with a customer is better than one without).

## Related

- `reflex_context/03_research/pi0_onnx_importable_sources.md` — Tier-1 source analysis (Thor, Tacoin, GR00T)
- `reflex_context/03_research/pi0_empirical_derisk_findings.md` — the Apr-17 component-level Optimum exports
- `src/reflex/exporters/pi0_prefix_exporter.py` — 5-file decomposition (still valuable for debug)
- `scripts/export_pi0_monolithic.py` — the Path C export script (this pattern)
