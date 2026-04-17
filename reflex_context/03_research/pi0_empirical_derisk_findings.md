# pi0-onnx-parity: empirical de-risk findings

**Session:** 2026-04-17 (Day 1 of 2-day plan)
**Status:** Day 1 partial — blocked on HF auth for full PaliGemma test, but toolchain signal captured.

## Key finding — Optimum support map is better than research indicated

Before downloading anything, we confirmed via `TasksManager.get_supported_tasks_for_model_type`:

| Component | Optimum support | Implication for pi0 |
|---|---|---|
| **Gemma** | ✅ `text-generation-with-past` (KV cache output built-in) | Pi0 backbone prefix-KV extraction = 1 CLI command |
| **Gemma2** | ✅ same tasks | Future-proofs pi0 migrations |
| **SigLIP** | ✅ `feature-extraction`, `zero-shot-image-classification` | Pi0 vision encoder = 1 CLI command |
| BERT / Llama | ✅ supported (reference) | Sanity check |
| **PaliGemma / PaliGemma2** | ❌ NOT supported | Confirms research — need custom glue |
| **SmolVLM** | ❌ NOT supported | Explains why our Apr-17 SmolVLA path hand-rolled everything |

## Strategic implication

Pi0 export = SigLIP + Gemma + pi0 glue. Two of three components are Optimum-native.

**Revised pi0-onnx-parity work breakdown:**

| Component | Path | Estimated LoC | Time |
|---|---|---|---|
| SigLIP vision encoder | `optimum-cli export onnx --task feature-extraction google/siglip-so400m-patch14-224` | ~0 (CLI) | 0.5 day |
| Gemma language (prefix-KV) | `optimum-cli export onnx --task text-generation-with-past google/gemma-2b` then apply pi0 fine-tuned weights | ~100 (weight overlay) | 1 day |
| Expert stack (18 layers, GQA, RoPE) | Hand-written; fork existing `pi0_exporter.py` + our SmolVLA `ExpertStack` | ~300–500 | 3–5 days |
| Flow matching loop glue | Host Python (per Isaac-GR00T pattern); N trt engine calls | ~50 | 0.5 day |
| Composition: vision + KV + expert + loop | New `pi0_prefix_exporter.py` + `pi0_runtime/server.py` | ~300 | 2–3 days |
| Parity verification + bug hunt | Polygraphy diff ladder + our `local_*_diff.py` port | — | 3–5 days |
| **Total** | | **~750–950 LoC of our custom code** | **~10–15 engineer-days (2–3 weeks)** |

**Previous estimate:** 2–3 weeks (matches). **Confidence:** significantly higher now because the biggest question (how to handle PaliGemma monolith) is resolved into "handle it as two Optimum-native components + glue."

## Revised mess-up probability: 25–30%

Down from 25–35% based on:
- **Optimum covers 2/3 of the major components** — confirmed via dry-run TasksManager query
- **KV extraction via CLI flag** — eliminates the riskiest custom code (prefix-KV Python-object serialization)
- **We already have SigLIP export patterns** from our SmolVLA work — pattern transfer direct
- Tooling installed and working (Polygraphy, graphsurgeon, onnx-diagnostic, onnxsim)

Residual risk:
- HF auth blocker (solvable) — need HF_TOKEN or `hf auth login`
- Custom glue between Optimum-exported components + our expert stack + flow-matching loop — this is where bugs will live
- Orin TRT 10.3 numerical bugs (~5% residual, can't de-risk without real hardware)

## Day 1 tooling install — completed

```
✓ polygraphy 0.49.26
✓ onnx-graphsurgeon 0.6.1
✓ onnx-diagnostic 0.9.2
✓ onnxsim 0.6.2
✓ optimum 2.1.0 + optimum-onnx 0.1.0 + transformers 4.57.6 (downgraded from 5.5.4 automatically)
```

Side effect: transformers downgrade from 5.5.4 → 4.57.6. lerobot still imports OK. Verified.

## Day 1 blockers

### Blocker 1: HF rate limit on Tacoin download

Attempted: `hf download Tacoin/openpi-pi0.5-libero-onnx --local-dir /tmp/oracle_pi05_libero_onnx`

Error: `HfHubHTTPError: 429 Client Error: Too Many Requests. We had to rate limit your IP (14.195.187.228). To continue using our service, create a HF account or login to your existing account, and make sure you pass a HF_TOKEN if you're using the API.`

**Resolution required:** user sets `HF_TOKEN` or runs `hf auth login`.

### Blocker 2: same rate limit blocks PaliGemma download

Attempted: `optimum-cli export onnx --model google/paligemma-3b-pt-224 ...`

Error: `OSError: We couldn't connect to 'https://huggingface.co' to load the files`.

**Resolution required:** same as Blocker 1.

### Partial workaround attempted: Optimum on cached SmolVLM2

Optimum does NOT support `smolvlm` model type (confirmed via empirical export attempt). Cannot use SmolVLM2 as a proxy sanity check for Optimum toolchain.

**But:** we learned Optimum DOES support `gemma` and `siglip` as separate architectures. That's the unlock.

## Day 2 LATE — pi0's actual Gemma backbone at 2.51B scale PASSED

After the xet-bridge download of vanilla Gemma-2b kept stalling, we bypassed HF entirely by:
1. Extracting pi0's Gemma backbone state-dict subset (prefix `paligemma_with_expert.paligemma.model.language_model.*`) from the already-cached 14GB `pi0_base` safetensors
2. Building a `GemmaForCausalLM` with GemmaConfig(hidden=2048, 18 layers, 8 q-heads, 1 kv-head, head_dim=256)
3. Loading pi0's 163 backbone keys into the model (only `embed_tokens.weight` missing — random-init, doesn't affect parity test)
4. Saving as a standard HF model dir at `/tmp/pi0_gemma_backbone` (10GB on disk)
5. Running `optimum-cli export onnx --model /tmp/pi0_gemma_backbone --task text-generation-with-past`

**Result on real-scale (2.51B params):**

```
Export: succeeded, 9.3GB external-data ONNX + 837K .onnx model structure file
       (external-data format auto-used — bypasses 2GB protobuf limit)

Parity:
  logits max_abs_diff:  3.86e-05  (26× below 1e-3 target)
  logits mean_abs_diff: 5.13e-06
  logits cos_sim:       +0.99999994
  present.0.key  diff:  1.24e-05  (first layer)
  present.17.key diff:  3.02e-05  (deepest layer — precision DOES accumulate but bounded)

VERDICT: PASS
```

**What this validates at real scale:**

1. ✅ Optimum exports the FULL Gemma-2b (2.51B params) cleanly — not just tiny-scale
2. ✅ External-data format (>2GB protobuf limit) works automatically
3. ✅ All 18 layers of K/V cache exposed as named outputs
4. ✅ Pi0's fine-tuned Gemma weights load cleanly into standard HF `GemmaForCausalLM` — architecture matches exactly
5. ✅ Precision accumulates across layers but stays well under our 1e-3 max_abs target
6. ✅ cos = 0.99999994 — exceeds 0.9999 target with comfortable headroom

**Mess-up probability after real-scale validation: ~10-15%** (was 20-25%).

The single biggest unknown is resolved. Remaining risk:
- Custom pi0 glue code (expert stack + flow matching + composition) — new bugs live here
- SigLIP export — untested (xet-bridge still blocked; can extract from pi0_base like we did for Gemma)
- Orin TRT 10.3 numerical bugs (~5%, untestable without hardware)

## Day 2 — tiny-Gemma Optimum sanity PASSED

Ran `scripts/local_tiny_gemma_sanity.py` to verify the Optimum + Gemma + KV-extraction path without waiting on the slow xet-bridge download of the real Gemma-2b weights.

**Config:** 2-layer Gemma, hidden=128, GQA 4/2 heads, head_dim=32, vocab=1000, 0.4M params total.

**Export command:** `optimum-cli export onnx --model <local_pt> --task text-generation-with-past --framework pt <out>`

**Result:**

```
ONNX inputs:
  input_ids: [batch_size, sequence_length]
  attention_mask: [batch_size, past_sequence_length + sequence_length]
  past_key_values.0.key: [batch_size, 2, past_sequence_length, 32]
  past_key_values.0.value: [batch_size, 2, past_sequence_length, 32]
  past_key_values.1.key: [batch_size, 2, past_sequence_length, 32]
  past_key_values.1.value: [batch_size, 2, past_sequence_length, 32]

ONNX outputs:
  logits: [batch_size, sequence_length, 1000]
  present.0.key: [batch_size, 2, past_sequence_length + sequence_length, 32]
  present.0.value: [batch_size, 2, past_sequence_length + sequence_length, 32]
  present.1.key: [...]
  present.1.value: [...]

PARITY:
  logits max_abs_diff:  7.15e-07
  logits mean_abs_diff: 3.21e-08
  logits cos_sim:       +1.00000000
  present.0.key diff:   1.79e-07
  VERDICT: PASS
```

**What this proves (big deal):**

1. **Optimum's Gemma → ONNX path is bit-exact** — cos=+1.00000000, max_abs_diff at float32 precision limits (1e-7 range).
2. **The exported graph exposes per-layer K/V as named outputs** — `present.0.key`, `present.1.value`, etc. This is EXACTLY the pattern pi0 needs for prefix-KV extraction.
3. **GQA is handled correctly** — 2 kv_heads + 4 q_heads + head_dim=32 all preserved.
4. **Dynamic axes work** — `past_sequence_length + sequence_length` properly parameterized (supports variable prompt lengths for pi0's dynamic-length prefix).
5. **The `text-generation-with-past` task flag does the right thing** — no custom OnnxConfig subclass needed for Gemma.

**What this does NOT prove (residual risk):**

- Won't catch Gemma-2b-scale issues (sqrt(hidden) constant-folding in FP32 vs BF16) until the real weights arrive
- Doesn't test pi0's fine-tuned Gemma (different weight statistics from plain Gemma base)
- Doesn't test the expert stack or flow-matching glue — those are still our custom code
- Doesn't cover SigLIP (download still blocked by xet-bridge; pivot needed)

**Revised mess-up probability after Day 2 empirical:**

- Was: 25–30%
- Now: **~20–25%**

The single biggest unknown (Optimum correctness) is resolved. Remaining risk is in pi0-specific integration code + SigLIP path + Gemma-2b scale parity.

## Gemma-2b full-scale download status

- **In progress** ~731MB / ~5GB at time of this write (2026-04-17 early eve)
- xet-bridge intermittently throttles but does resume
- Will enable full-scale Optimum Gemma-2b export test once complete

## SigLIP download — blocked

- `google/siglip-so400m-patch14-224` (3.3GB): xet-bridge stalls after ~30MB. Connection reset by peer.
- `google/siglip-base-patch16-224` (400MB): stalls at 0 bytes on incomplete file.
- **Pivot option:** pi0_base's cached state_dict contains a PaliGemma SigLIP subset. We can extract those weights and bypass HF download for SigLIP entirely.
- **Alternative:** resume later when xet-bridge cooperates (works intermittently).

## Next concrete actions

1. Wait for Gemma-2b download to complete (~30-60 min more based on observed rate).
2. Run full `optimum-cli export onnx --model google/gemma-2b --task text-generation-with-past` — verify scale parity.
3. If Gemma-2b parity holds → commit to pi0-onnx-parity implementation at 1.5–2 week estimate.
4. Defer SigLIP to next session; in the meantime, extract SigLIP weights from pi0_base cache.

## Original Day 2 plan (kept for reference)

### User: unblock HF auth

```bash
# Option 1: set token
export HF_TOKEN=hf_xxx

# Option 2: interactive login
hf auth login
```

### Day 2: proceed with Optimum-per-component tests

Once HF auth works:

1. **SigLIP sanity**: `optimum-cli export onnx --model google/siglip-so400m-patch14-224 --task feature-extraction /tmp/siglip_sanity/`
   - Verifies Optimum SigLIP path works end-to-end
   - Size check: should be ~400MB for so400m; watch for 2GB protobuf limit
   - Polygraphy-diff vs `transformers.SiglipVisionModel` PyTorch output

2. **Gemma backbone sanity**: `optimum-cli export onnx --model google/gemma-2b --task text-generation-with-past /tmp/gemma_sanity/`
   - Verifies Gemma ONNX with KV outputs works
   - Inspect `decoder_model_merged.onnx` and `decoder_with_past.onnx` structure
   - Polygraphy-diff vs `transformers.GemmaModel` PyTorch output on a fixed prompt

3. **Apply pi0 fine-tuned weights to Gemma ONNX**
   - Extract Gemma subset from pi0_base state_dict (`paligemma_with_expert.paligemma.language_model.*`)
   - Overlay onto Optimum's Gemma ONNX via onnx-graphsurgeon weight replacement
   - Verify replaced-weight Gemma ONNX still produces sensible outputs

4. **Verdict + revised plan update**
   - If SigLIP + Gemma both clean with pi0 weights overlaid → commit to 2-week pi0-onnx-parity
   - If Gemma weight overlay fails → fallback: export pi0's Gemma subset directly via our own `torch.onnx.export` pipeline

## Output artifacts expected

By end of Day 2:
- `/tmp/siglip_sanity/siglip.onnx` + parity diff result
- `/tmp/gemma_sanity/decoder_*.onnx` + parity diff result
- `/tmp/oracle_pi05_libero_onnx/` (Tacoin — for pi0.5 future work)
- Updated `reflex_context/mvp_queue.md` with empirical ETA
- Draft `src/reflex/exporters/pi0_prefix_exporter.py` skeleton (new file)
