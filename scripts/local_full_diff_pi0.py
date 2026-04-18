"""End-to-end pi0 parity: PyTorch pi0 policy vs ONNX backbone + ONNX expert.

Uses the proven ONNX exports:
  - backbone (decoder_prefill): HF GemmaForCausalLM via Optimum
  - expert: HF GemmaForCausalLM (gemma-300m) via Optimum

Reuses PyTorch pi0's own embed_prefix + embed_suffix for inputs (coupled
to lerobot; production-decoupled in v0.3). This isolates the parity
question to the ONNX computation itself.

Target: cos >= 0.999 vs PyTorch pi0 predict_action_chunk on shared noise.
"""
import sys
import types

for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

import numpy as np
import torch


def _patch_pi0_for_transformers_457():
    from lerobot.policies.pi0 import modeling_pi0

    def patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        if hasattr(image_outputs, "pooler_output"):
            features = image_outputs.pooler_output
        else:
            features = image_outputs
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    modeling_pi0.PaliGemmaWithExpertModel.embed_image = patched_embed_image


def _patch_create_causal_mask_kwarg():
    from transformers import masking_utils

    original = masking_utils.create_causal_mask

    def shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    masking_utils.create_causal_mask = shim
    from lerobot.policies import pi_gemma
    if hasattr(pi_gemma, "create_causal_mask"):
        pi_gemma.create_causal_mask = shim


_patch_pi0_for_transformers_457()
_patch_create_causal_mask_kwarg()


EXPORT_DIR = "/tmp/pi0_full_prefix_export"
EXPERT_ONNX = "/tmp/pi0_expert_onnx/model.onnx"
BACKBONE_ONNX = f"{EXPORT_DIR}/decoder_prefill/model.onnx"


def run_pytorch_reference(policy, batch, noise):
    """Run pi0 PyTorch with shared noise. Returns full actions [B, chunk, action_dim]."""
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise))
    return actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)


def run_onnx_hybrid(policy, batch_pp, noise, num_steps=10):
    """Run ONNX backbone + ONNX expert, reusing pi0's embed + output projections."""
    import onnxruntime as ort
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch  # noqa

    sess_bb = ort.InferenceSession(BACKBONE_ONNX, providers=["CPUExecutionProvider"])
    sess_ex = ort.InferenceSession(EXPERT_ONNX, providers=["CPUExecutionProvider"])

    # Use PyTorch pi0 to compute the prefix embeddings + suffix embeddings
    # (couples to lerobot; decoupled in v0.3)
    # _preprocess_images and prepare_state live on PI0Policy (outer), not PI0Pytorch (model)
    images, img_masks = policy._preprocess_images(batch_pp)
    lang_tokens = batch_pp["observation.language.tokens"]
    lang_masks = batch_pp["observation.language.attention_mask"]
    state = policy.prepare_state(batch_pp)
    model = policy.model

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )

    # --- ONNX backbone: prefix_embs -> prefix_kv (per-layer) ---
    pe = prefix_embs.cpu().numpy().astype(np.float32)
    pm = prefix_pad_masks.cpu().numpy().astype(np.int64)
    bb_out = sess_bb.run(None, {"inputs_embeds": pe, "attention_mask": pm})
    # bb_out[0] = last_hidden_state; bb_out[1+] = present.N.key/value
    num_layers = policy.config.num_expert_layers if hasattr(policy.config, "num_expert_layers") else 18
    prefix_kv = []
    for i in range(num_layers):
        prefix_kv.append((bb_out[1 + 2 * i], bb_out[2 + 2 * i]))
    # Shape of each: [B, nkv, prefix_len, head_dim]

    # --- Flow matching loop ---
    B, chunk, action_dim = noise.shape
    x_t = torch.from_numpy(noise).to(torch.float32)
    dt = -1.0 / num_steps

    prefix_len = prefix_embs.shape[1]
    for step in range(num_steps):
        t = torch.tensor([1.0 + step * dt], dtype=torch.float32)

        with torch.no_grad():
            suffix_embs, suffix_pad_masks, _, _ = model.embed_suffix(state, x_t, t)

        # Attention mask: all valid (no padding)
        suffix_len = suffix_embs.shape[1]
        full_mask = np.ones((B, prefix_len + suffix_len), dtype=np.int64)

        # Feed expert ONNX with inputs_embeds=suffix, past_key_values=prefix_kv
        ex_inputs = {
            "inputs_embeds": suffix_embs.cpu().numpy().astype(np.float32),
            "attention_mask": full_mask,
        }
        for i in range(num_layers):
            ex_inputs[f"past_key_values.{i}.key"] = prefix_kv[i][0]
            ex_inputs[f"past_key_values.{i}.value"] = prefix_kv[i][1]

        ex_out = sess_ex.run(None, ex_inputs)
        last_hidden = ex_out[0]  # [B, suffix_len, hidden]
        # Take last chunk tokens, apply action_out_proj
        chunk_hidden = torch.from_numpy(last_hidden[:, -chunk:, :]).to(torch.float32)
        with torch.no_grad():
            velocity = model.action_out_proj(chunk_hidden).cpu().numpy()

        x_t = x_t + dt * torch.from_numpy(velocity)

    return x_t.numpy()


def main():
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download

    print("Loading PyTorch pi0_base...")
    policy = PI0Policy.from_pretrained("lerobot/pi0_base").eval().to(dtype=torch.float32).to("cpu")

    rng = np.random.RandomState(42)
    img = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)
    task = "pick up the red bowl"

    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": [task],
    }
    repo = snapshot_download("lerobot/pi0_base")
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch_raw)

    chunk = policy.config.chunk_size
    action_dim = policy.config.max_action_dim
    noise = np.random.RandomState(99).randn(1, chunk, action_dim).astype(np.float32)

    print("\n--- PyTorch pi0 reference ---")
    pt_actions = run_pytorch_reference(policy, batch_pp, noise)
    print(f"  shape: {pt_actions.shape}, first: {pt_actions[0, 0, :5]}")

    print("\n--- ONNX hybrid (backbone + expert ONNX, PyTorch embed_suffix) ---")
    onnx_actions = run_onnx_hybrid(policy, batch_pp, noise)
    print(f"  shape: {onnx_actions.shape}, first: {onnx_actions[0, 0, :5]}")

    # Compare first action
    pt0 = pt_actions[0, 0]
    on0 = onnx_actions[0, 0]
    diff = pt0 - on0
    max_abs = float(np.abs(diff).max())
    l2 = float(np.linalg.norm(diff))
    cos = float(np.dot(pt0, on0) / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-8))

    print(f"\n====== FIRST-ACTION PARITY ======")
    print(f"  max_abs = {max_abs:.4e}")
    print(f"  L2      = {l2:.4e}")
    print(f"  cos     = {cos:+.6f}")

    full_cos = float(
        np.dot(pt_actions.flatten(), onnx_actions.flatten())
        / (np.linalg.norm(pt_actions) * np.linalg.norm(onnx_actions) + 1e-8)
    )
    full_max = float(np.abs(pt_actions - onnx_actions).max())
    print(f"\n  full chunk: max_abs = {full_max:.4e}, cos = {full_cos:+.6f}")

    passed = cos >= 0.999 and max_abs < 0.1
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
