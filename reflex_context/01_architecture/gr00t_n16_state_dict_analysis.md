# GR00T N1.6 state-dict analysis (2026-04-19)

**Purpose.** Before porting Eagle VLM + state_encoder into reflex's GR00T export, empirically map the actual state dict of `nvidia/GR00T-N1.6-3B`. Answers key land-mine questions from the Eagle VLM export plan and surfaces a newly-discovered gap (state conditioning is also missing in our current export).

**Diagnostic reproducer:** `modal run scripts/modal_gr00t_keys_dump.py`.

**Background:** The Eagle VLM export plan (`gr00t_eagle_vlm_export_plan.md`) identified three land mines. The research agent couldn't verify them directly from web sources. This script pulls the safetensors and reads the key schema locally on Modal.

---

## Top-level prefix breakdown

1106 keys total across the state dict:

| Prefix | Key count | Params |
|---|---|---|
| `action_head.action_decoder` | 4 | 37.8M |
| `action_head.action_encoder` | 6 | 232.9M |
| `action_head.model` | 456 | 1091.7M |
| `action_head.position_embedding` | 1 | 1.6M |
| **`action_head.state_encoder`** | **4** | **54.6M** |
| `action_head.vlln` | 2 | 0.0M |
| **`backbone.model`** | **633** | **1868.0M** |

Action head total: ~1.4B. Backbone (Eagle VLM) total: ~1.9B. Full checkpoint: ~3.3B (matches model card's 3.29B).

---

## Finding 1 — `eagle_linear` is ABSENT ✅

The hypothesis from the Eagle VLM export plan was that N1.6 skips the 2048→1536 projection that N1.5 has. **Confirmed:**

```
ABSENT — no eagle_linear keys in state dict.
→ N1.6 SKIPS the projection. DiT's vlln=LayerNorm(2048) consumes Qwen2 hidden directly.
```

Port implication: **do not add `eagle_linear`** to `EagleExportStack`. Feed raw Qwen2 hidden (2048-dim) directly into DiT's `vlln = LayerNorm(2048)`.

---

## Finding 2 — Backbone prefix is `backbone.model.*` (not `backbone.eagle_model.*`)

Research agent guessed `backbone.eagle_model.vision_model.*` etc. Actual prefix is `backbone.model.*` — the `eagle_model` attribute is renamed to just `model` in the fine-tuned checkpoint.

Sub-structure under `backbone.model.*` (633 keys total) — verified by the deeper v2 dump:

| Sub-prefix | Purpose |
|---|---|
| `backbone.model.language_model.*` | Qwen2 decoder (~0.5B, layers + embeds + lm_head tied) |
| `backbone.model.vision_model.*` | SigLIP-so400m-patch14-384 vision tower (~400M) |
| `backbone.model.mlp1.*` | 2-layer MLP connector (vision → Qwen2 hidden space) |

Key shapes (sampled):
- `backbone.model.language_model.lm_head.weight: (151680, 2048)` — vocab 151680, hidden 2048
- `backbone.model.language_model.model.embed_tokens.weight: (151680, 2048)` — tied with lm_head
- `backbone.model.language_model.model.layers.0.mlp.down_proj.weight: (2048, 6144)` — Qwen2 MLP inner dim 6144
- `backbone.model.language_model.model.layers.0.input_layernorm.weight: (2048,)` — LayerNorm at 2048

Port implication: **state-dict key prefix for Eagle vendoring is `backbone.model.*`**, not `backbone.eagle_model.*`. Adjust the weight-loading code accordingly.

---

## Finding 3 🔴 — `action_head.state_encoder` EXISTS (newly discovered gap)

Our current `scripts/modal_gr00t_monolithic_export.py` exports `action_encoder → DiT → action_decoder`. It **does not include** `state_encoder`, because our current understanding of the architecture didn't include state conditioning on the action head.

The state dict has state_encoder weights:

```
action_head.state_encoder.layer1.W: shape=(32, 128, 1024)   # 32 embodiments × state_dim=128 × hidden=1024
action_head.state_encoder.layer1.b: shape=(32, 1024)
action_head.state_encoder.layer2.W: shape=(32, 1024, 1536)  # project to DiT hidden
action_head.state_encoder.layer2.b: shape=(32, 1536)
```

Total: 54.6M params. 32 per-embodiment layers (matches the 32-embodiment-encoder/decoder pattern we already handle). Project state (128-dim) → hidden (1024) → DiT hidden (1536).

**Implication:** our current GR00T export is missing BOTH state and VLM conditioning. Actions are generated purely from noise + timestep. This matches why the single-step parity test (against `GR00TFullStack.forward(noisy_actions, timestep, position_ids)`) passes at cos=1.0 — both sides have the same zero state + zero VLM. But the model as deployed generates pose-and-task-irrelevant actions.

Port implication: the full integration needs **three** encoders plumbed into the DiT:
1. Action encoder (noisy_actions → action tokens) — already ported
2. **State encoder (robot_state → state token) — NEW, needs port**
3. VLM encoder (images + text → KV features) — next step

---

## Finding 4 — SigLIP vision tower at `backbone.model.vision_model.*` ✅

Confirmed by the v2 dump:

| Sub-prefix | Key count | Params |
|---|---|---|
| `backbone.model.language_model.*` | 179 | 1426.7M |
| `backbone.model.mlp1.*` | 6 | 13.6M |
| `backbone.model.vision_model.*` | 448 | 427.7M |

SigLIP vision tower is 427.7M params, 448 keys — consistent with SigLIP-so400m-patch14-384.

**`mlp1` connector structure** (full dump):
```
backbone.model.mlp1.0.bias:   (4608,)       BF16  # LayerNorm (4608-dim)
backbone.model.mlp1.0.weight: (4608,)       BF16
backbone.model.mlp1.1.bias:   (2048,)       BF16  # Linear(4608 → 2048)
backbone.model.mlp1.1.weight: (2048, 4608)  BF16
backbone.model.mlp1.3.bias:   (2048,)       BF16  # Linear(2048 → 2048)
backbone.model.mlp1.3.weight: (2048, 2048)  BF16
```

Structure = `nn.Sequential(LayerNorm(4608), Linear(4608, 2048), GELU(), Linear(2048, 2048))` — matches the vendored Eagle modeling code's `mlp_connector_layers=2` path.

**Why 4608?** SigLIP hidden 1152 × pixel_shuffle(0.5)²× = 1152 × 4 = 4608. Pixel shuffle concatenates 2×2 patches into a single 4×-hidden token before feeding the connector.

---

## Finding 5 — Qwen2 is "big Qwen2" (`151680` vocab, `2048` hidden)

Matches Qwen2-0.5B sized backbone. Vocab 151680 = standard Qwen2 tokenizer. Hidden 2048 = matches `vlln=LayerNorm(2048)` on the DiT side.

Per-layer MLP intermediate dim is 6144 (3× hidden) — consistent with Qwen2 SwiGLU pattern.

Port implication: tokenizer used should be `Qwen/Qwen2-0.5B` (or whatever Qwen2 checkpoint the Eagle config points to). The `Qwen/Qwen2.5-0.5B` tokenizer or HuggingFaceVLA-tuned variant may also apply.

---

## Finding 6 — DiT action head has:
- `action_head.model.timestep_encoder.timestep_embedder.linear_1: (1536, 256)` — sinusoidal 256 → hidden 1536
- `action_head.model.transformer_blocks.0.attn1.to_k.bias: (1536,)` — cross-attn to key projection, output hidden 1536
- `action_head.position_embedding.weight: (1024, 1536)` — 1024 positions × hidden 1536
- `action_head.vlln.weight/bias: (2048,)` — LayerNorm at 2048 (consumes Qwen2 hidden directly, pre-mapping to KV)

DiT hidden dim = 1536. VLM KV input dim = 2048 → 1536 via `attn1.to_k`.

---

## Revised Eagle VLM export plan (diff vs the original plan doc)

The original plan stands at 5 steps. Amendments based on this analysis:

### Step 1c amendments
- ✅ `eagle_linear` absent — confirmed, plan doc was already conservative on this
- 🔴 **Add state_encoder port** as a prerequisite step 0.5 (or fold into step 2)
- 🟡 State-dict key prefix is `backbone.model.*` not `backbone.eagle_model.*` — adjust weight-loading code

### Step 2 additions
- Port `action_head.state_encoder` alongside the Eagle `EagleExportStack`
- Update `GR00TFullStack.forward` signature to accept BOTH `state` AND `vlm_kv`:
  ```python
  def forward(self, noisy_actions, timestep, position_ids, state, vlm_kv):
      ...
  ```

### Updated plumbing diagram
```
images + text  →  EagleExportStack  →  vlm_kv (B, T, 2048)
robot_state    →  StateEncoderStack →  state_token (B, 1, 1536)
noisy_actions  →  action_encoder    →  action_tokens (B, 50, 1536)
timestep                            →  time_emb (B, 1536)

[state_token || action_tokens] with positional embed → DiT transformer blocks (cross-attn to vlm_kv) → action_decoder → velocity
```

---

## Numbers for sizing

| Component | Params | FP32 size | FP16 size |
|---|---|---|---|
| Eagle VLM (backbone.model.*) | 1.87B | 7.5 GB | 3.7 GB |
| DiT (action_head.model) | 1.09B | 4.4 GB | 2.2 GB |
| Action encoder/decoder (32 emb) | 270M | 1.1 GB | 0.5 GB |
| State encoder (32 emb) | 55M | 0.2 GB | 0.1 GB |
| **Total FP32** | **3.29B** | **~13.2 GB** | **~6.5 GB** |

For runtime: Eagle VLM ONNX (~7.5 GB FP32) runs once per observation, fed into DiT ONNX (~5.7 GB FP32 for DiT + encoders) × 4 DDPM steps. Combined disk: ~13 GB.

**Fits on:** Orin 16GB+, desktop GPU. **Does not fit on:** Orin Nano 8GB (same limitation as pi0/pi0.5 — all Jetson Nano 8GB support pending v0.3 FP16 engine rebuild).

---

## Verified (all by this key dump — two iterations v1 + v2)

- ✅ `eagle_linear` absent — N1.6 feeds raw Qwen2 2048-dim into DiT
- ✅ State dict prefix is `backbone.model.*` (not `backbone.eagle_model.*`)
- ✅ `action_head.state_encoder` exists: 4 keys, 54.6M params, 32-embodiment (128→1024→1536)
- ✅ `backbone.model.vision_model.*`: 448 keys, 427.7M params (SigLIP-so400m)
- ✅ `backbone.model.language_model.*`: 179 keys, 1426.7M params (Qwen2)
- ✅ `backbone.model.mlp1.*`: 6 keys, 13.6M params — Sequential(LayerNorm(4608) → Linear(4608→2048) → GELU → Linear(2048→2048))
- ✅ DiT hidden = 1536, VLM output = 2048, state hidden = 1024, pixel-shuffled vision = 4608

Nothing pending. Full tree confirmed.

---

## Cross-references

- `reflex_context/01_architecture/gr00t_eagle_vlm_export_plan.md` — the 5-step implementation plan
- `reflex_context/01_architecture/gr00t_ddpm_dit_vs_flow_matching.md` — why GR00T's architecture differs from pi0
- `scripts/modal_gr00t_keys_dump.py` — the diagnostic reproducer
- `src/reflex/exporters/gr00t_exporter.py` — current GR00T exporter (will be extended with state_encoder + vlm_kv)
- `src/reflex/exporters/eagle_vendor/` — vendored Eagle source (step 1 done, peft made optional, eager attn default)
- `GOALS.yaml` — `gr00t-vlm-conditioning` (weight 9, current_focus)
