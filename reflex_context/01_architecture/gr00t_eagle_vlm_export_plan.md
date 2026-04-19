# GR00T Eagle VLM ONNX export plan (2026-04-19)

**Goal:** close the last gap in GR00T support — export the Eagle VLM backbone so real images + language conditioning drive the DiT, instead of a zero-stubbed VLM-KV tensor. This is goal `gr00t-vlm-conditioning` (weight 9) in GOALS.yaml, the last monetization blocker for GR00T customers.

**Current state:** `scripts/modal_gr00t_monolithic_export.py` exports a DiT-only ONNX at `cos=+1.000000` parity with PyTorch. But the VLM-KV input is a zero tensor. Deployed, the model ignores the image + prompt and produces noise-driven actions. Not shippable for real multi-modal control.

---

## Eagle 2.5 VL — what we're exporting

Eagle 2.5 VL is a SigLIP + Qwen2 composite VLM. GR00T uses a fine-tuned copy bundled inside `nvidia/GR00T-N1.6-3B`.

| Component | Model | Hidden dim | Purpose |
|---|---|---|---|
| Vision | `SiglipVisionModel` (SigLIP-so400m-patch14-384, ~400M) | 1152 | Image → patch tokens |
| Pixel shuffle | 0.5× downsample | — | 1024 patch tokens → 256 image tokens/tile |
| Connector | `mlp1`: LayerNorm + 2-layer MLP | 2048 | Image features → Qwen2 hidden space |
| Text | `Qwen2ForCausalLM` (0.5B) | 2048 | Process image tokens + text, output contextualized hidden states |

**Output fed to DiT:** `[B, T, 2048]` where T is vision tokens + text tokens. Our DiT's `vlln` is `LayerNorm(2048)` — confirms N1.6 consumes raw Qwen2 hidden (skipping the `eagle_linear` 2048→1536 projection that N1.5 had).

**State-dict keys** (inside `nvidia/GR00T-N1.6-3B`):
- `backbone.eagle_model.vision_model.*` — SigLIP
- `backbone.eagle_model.language_model.*` — Qwen2
- `backbone.eagle_model.mlp1.*` — connector
- (`backbone.eagle_linear.*` — absent in N1.6 per hypothesis; verify by dumping keys)

**Entry point** (in lerobot source, `groot_n1.py:135-146`):

```python
eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
eagle_features = eagle_output.hidden_states[self.select_layer]
return eagle_features, eagle_input["attention_mask"]
```

Single-shot encode, no autoregressive loop, no runtime KV cache. **Good for ONNX** — avoids the DynamicCache surgery we needed for pi0/pi0.5.

---

## Torch-export patches required

Qwen2's decoder-only attention uses the same stack that bit us on pi0/pi0.5:

1. **F.pad causal mask** instead of `torch.cat` (FakeTensor loses suffix dim)
2. **Frozen DynamicLayer.update** during trace (Qwen2 calls `past_kv.update()` unconditionally)
3. **`past_kv.get_seq_length()`** for mask K dim, not `input_ids.shape[-1]`

Plus a fourth patch specific to Eagle:
- **Force `_attn_implementation="eager"`** at load time. Eagle's config hardcodes `flash_attention_2`. ORT/TRT can't consume FA2 ops.

Reference stack lives in `scripts/modal_pi0_monolithic_export.py` lines 200–310. Port these wholesale to the Eagle export script.

**SigLIP needs no patches** — bidirectional attention, no causal mask, no cache.

---

## Integration with existing DiT ONNX

Good news: `GR00TExpertStack.forward` (in `src/reflex/exporters/gr00t_exporter.py:208-241`) already accepts `vlm_kv: Optional[Tensor]`. The existing export just passes `None`. Surgical add to `GR00TFullStack`:

```python
# Current (zero-stubbed):
def forward(self, noisy_actions, timestep, position_ids):
    ...
    velocity_tokens = self.dit(tokens, timestep, position_ids, vlm_kv=None)

# New:
def forward(self, noisy_actions, timestep, position_ids, vlm_kv):
    ...
    velocity_tokens = self.dit(tokens, timestep, position_ids, vlm_kv=vlm_kv)
```

Keep the existing zero-VLM ONNX for backward compat (naming it `model.onnx` currently). The new export writes **two ONNXes**:
- `eagle_vlm.onnx` — vision + text → KV features `[B, T, 2048]`
- `expert_stack_with_vlm.onnx` — DiT taking `noisy_actions + timestep + position_ids + vlm_kv` → `velocity`

Runtime chains them: Eagle runs once per observation (images + prompt), DiT runs 4 times per action chunk (DDPM denoise loop) reusing the same KV.

---

## 5-step implementation plan (1–2 days)

### Step 1 — vendor Eagle source (~30 min)
Copy `lerobot/src/lerobot/policies/groot/eagle2_hg_model/*` into `src/reflex/exporters/eagle_vendor/`. Strip `peft`/`LoraConfig` imports (not needed for export). Force `_attn_implementation="eager"` in config at load time.

### Step 2 — build `EagleExportStack(nn.Module)` (~4 hours)
Mirrors `EagleBackbone.forward_eagle`. Signature:

```python
class EagleExportStack(nn.Module):
    def forward(self, pixel_values, input_ids, attention_mask, image_flags):
        # SigLIP encode → [B, N_patches, 1152]
        # mlp1 → [B, N_patches, 2048]
        # Qwen2 decoder (single-pass, output_hidden_states=True) → [B, T, 2048]
        return hidden_states[self.select_layer], attention_mask
```

Load weights from state_dict with `backbone.eagle_model.*` prefix. Apply the 3-patch stack + eager-attn at build time.

### Step 3 — local parity test (~2 hours)
`scripts/local_eagle_parity.py`: one image + "pick up the apple" → compare `EagleExportStack` features vs `GR00TN15.from_pretrained().backbone.forward_eagle()`. Target `cos=+1.000000, max_abs<1e-4`. Same seeded-noise discipline as pi0 parity tests.

### Step 4 — Modal export (~3 hours)
Extend `scripts/modal_gr00t_monolithic_export.py` with `export_gr00t_vlm_modal()`:
1. Build `EagleExportStack`, export via `torch.onnx.export(opset=19)` → `eagle_vlm.onnx`
2. Build `GR00TFullStack` with `vlm_kv` input plumbed, export → `expert_stack_with_vlm.onnx`
3. Parity test chains the two ONNXes and compares against raw PyTorch

### Step 5 — end-to-end test (~2 hours)
Real image + "pick up the red cup" through the chain → actions. Flip the image; actions change (currently with zero-KV they don't). Then update `measured_numbers.md` + launch drafts with "GR00T now has real VLM conditioning, cos=+1.000000 verified."

---

## Known land mines

- **N1.5 vs N1.6 difference.** Research agent couldn't verify N1.6's config directly. N1.5 has `eagle_linear` (2048→1536); N1.6 hypothesized to skip it. **Verify first** by dumping state_dict keys. If `backbone.eagle_linear.weight` exists, include it; if absent, skip.
- **Standalone `nvidia/Eagle2.5-8B` ≠ GR00T's Eagle.** GR00T's is a fine-tune. Export from the GR00T checkpoint, not from the standalone release.
- **Flash Attention 2 hardcoded.** Eagle's `configuration_eagle2_5_vl.py:52` has `_attn_implementation = "flash_attention_2"` baked in. Must override to "eager" before calling `from_pretrained`, or ONNX export breaks on FA2 ops.

---

## Expected ONNX sizes

Rough estimates (post-fp32):
- `eagle_vlm.onnx`: ~2–3 GB (SigLIP ~400M + Qwen2 ~0.5B + connector ~5M = ~1B params × 4 bytes + embeddings)
- `expert_stack_with_vlm.onnx`: ~4.4 GB (same as current DiT-only export + one more input)

Combined: ~7 GB. Fits on Orin 16GB+, not Orin Nano 8GB (same limitation as pi0/pi0.5).

---

## Cross-references

- `src/reflex/exporters/gr00t_exporter.py` lines 208-241 — where `vlm_kv` plumbs through `GR00TExpertStack`
- `src/reflex/exporters/gr00t_exporter.py` lines 480-700 — `GR00TFullStack` (needs new arg)
- `scripts/modal_gr00t_monolithic_export.py` — extend this with `export_gr00t_vlm_modal`
- `scripts/modal_pi0_monolithic_export.py` lines 200–310 — the 3-patch stack reference
- `/Users/romirjain/Desktop/building projects/lerobot/src/lerobot/policies/groot/groot_n1.py` — `EagleBackbone` source to mirror
- `/Users/romirjain/Desktop/building projects/lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py` — Eagle model definition
- `GOALS.yaml` — `gr00t-vlm-conditioning` (weight 9, in current_focus)
- `reflex_context/external_refs.md` — clone location of lerobot
