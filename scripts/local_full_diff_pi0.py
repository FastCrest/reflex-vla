"""End-to-end pi0-onnx-parity: shared-noise PyTorch vs Pi0OnnxServer.

Loads real PI0Policy + our composed Pi0OnnxServer (5 ONNX files), runs both
with identical (seeded) inputs + shared noise, compares first action
via cos, max_abs, L2. Target: cos >= 0.999.

Prerequisites:
  - pi0_base cached in ~/.cache/huggingface (14GB)
  - Full pi0 ONNX bundle at /tmp/pi0_full_prefix_export/ (from
    scripts/run_export_pi0_prefix.py or export_pi0_prefix())

Expected first run: likely cos < 0.999 due to composition bugs. Use
diagnostic ladder (stage-diff, single-layer) to localize. This script
is the entry point; bug-hunt iterations modify it to print per-stage.
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

EXPORT_DIR = "/tmp/pi0_full_prefix_export"


def build_reference_inputs():
    """Build a deterministic test batch for pi0."""
    rng = np.random.RandomState(42)
    img = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    # PaliGemma expects [-1, 1] normalization
    img_t = img_t * 2.0 - 1.0
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)
    return img_t, img, state, "pick up the red bowl"


def run_pytorch_reference(policy, img_t, state, task, noise):
    """Run real PI0Policy with shared noise. Returns [B, chunk, action_dim]."""
    # Build batch matching pi0's expected format
    batch = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": [task],
    }
    # Preprocess via lerobot's pipeline
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
    )
    from huggingface_hub import snapshot_download
    repo = snapshot_download("lerobot/pi0_base")
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch)

    with torch.no_grad():
        actions = policy.predict_action_chunk(batch_pp, noise=torch.from_numpy(noise))
    return actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)


def run_onnx_pipeline(img, task, state, noise, num_steps=10):
    """Run Pi0OnnxServer with same inputs (after matching preprocessing)."""
    from reflex.runtime.pi0_onnx_server import Pi0OnnxServer

    srv = Pi0OnnxServer(EXPORT_DIR)
    srv.load()

    # Match pi0's preprocessing:
    #   - Image: [-1, 1] normalize to pixel_values [B, 3, 224, 224]
    #   - Text: tokenize via PaliGemmaTokenizer (PaliGemma = SentencePiece)
    img_arr = img.astype(np.float32) / 255.0
    img_arr = img_arr * 2.0 - 1.0
    pixel_values = img_arr.transpose(2, 0, 1)[None, :]  # [1, 3, 224, 224]

    # Tokenize text — PaliGemma uses Gemma's SentencePiece
    # For the de-risk, use simple seeded tokenization
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    text_inputs = tok(task, return_tensors="np", padding=False)
    input_ids = text_inputs["input_ids"].astype(np.int64)

    result = srv.predict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        state=state.numpy() if hasattr(state, "numpy") else state,
        noise=noise,
        num_steps=num_steps,
    )
    return result["actions"]


def main():
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    print("Loading PyTorch reference (pi0_base)...")
    policy = PI0Policy.from_pretrained("lerobot/pi0_base")
    policy.eval().to(dtype=torch.float32).to("cpu")

    img_t, img_np, state, task = build_reference_inputs()

    chunk = policy.config.chunk_size
    action_dim = policy.config.max_action_dim
    noise = np.random.RandomState(99).randn(1, chunk, action_dim).astype(np.float32)

    print(f"Shared noise: shape={noise.shape}, seed=99")
    print(f"PyTorch pi0 reference run ({chunk} step chunk)...")
    pt_actions = run_pytorch_reference(policy, img_t, state, task, noise)
    print(f"  pt actions shape: {pt_actions.shape}")

    print("\nONNX composed pipeline run...")
    onnx_actions = run_onnx_pipeline(img_np, task, state, noise)
    print(f"  onnx actions shape: {onnx_actions.shape}")

    # Compare first action (most interpretable), all dims
    pt_first = pt_actions[0, 0]
    onnx_first = onnx_actions[0, 0]
    diff = pt_first - onnx_first
    max_abs = float(np.abs(diff).max())
    l2 = float(np.linalg.norm(diff))
    denom = np.linalg.norm(pt_first) * np.linalg.norm(onnx_first) + 1e-8
    cos = float(np.dot(pt_first, onnx_first) / denom)

    print(f"\n====== FIRST-ACTION PARITY ======")
    print(f"  pt  : {np.round(pt_first[:7], 3)}")
    print(f"  onnx: {np.round(onnx_first[:7], 3)}")
    print(f"  max_abs = {max_abs:.4f}")
    print(f"  L2      = {l2:.4f}")
    print(f"  cos     = {cos:+.6f}")

    # Full chunk cos for robustness
    full_diff = pt_actions.flatten() - onnx_actions.flatten()
    full_max = float(np.abs(full_diff).max())
    full_cos = float(
        np.dot(pt_actions.flatten(), onnx_actions.flatten())
        / (np.linalg.norm(pt_actions) * np.linalg.norm(onnx_actions) + 1e-8)
    )
    print(f"\n  full chunk max_abs = {full_max:.4f}, cos = {full_cos:+.6f}")

    passed = cos >= 0.999 and max_abs < 0.1
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL — bug hunt next'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
