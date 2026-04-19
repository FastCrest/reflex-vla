"""Step 3 parity — compare our GR00TFullStack to lerobot's action head.

Goal: verify that our extended `GR00TFullStack` (with state + vlm_kv
plumbed) produces the same single-step velocity output as lerobot's
`FlowMatchingActionHead` on shared seeded inputs.

Approach:
  1. Load nvidia/GR00T-N1.6-3B state_dict.
  2. Build OUR `GR00TFullStack` (embodiment 0).
  3. Instantiate lerobot's `GR00TN15` from_pretrained (handles all init
     quirks). Extract just the action_head.
  4. Generate synthetic inputs:
       - noisy_actions (1, 50, 128) fp32
       - timestep (1,) fp32
       - state (1, 128) fp32
       - vlm_kv (1, seq_kv, 2048) random fp32
  5. Path A (ours): GR00TFullStack.forward(noisy, t, pos, state, vlm_kv)
  6. Path B (lerobot): mirror the relevant portion of
     FlowMatchingActionHead.get_action for a SINGLE step
       (skip the Euler loop, skip future_tokens because N1.6's
        checkpoint has no future_tokens weights — verified via
        state-dict dump).
  7. Compare actions: cos + max_abs on the action chunk.

Target: cos=+1.000000, max_abs<1e-4 (like our pi0/pi0.5 bit-exact
parity runs). If we don't hit it, the delta shows where our port
diverges from lerobot.

Usage:
    modal run scripts/modal_gr00t_vlm_parity.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-gr00t-vlm-parity")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "clang")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "typer",
        "rich",
        "lerobot==0.5.1",
        "num2words",
    )
    .run_commands(
        f"pip install 'reflex-vla @ git+https://github.com/rylinjames/reflex-vla@{_HEAD}'",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[_hf_secret()],
)
def run_parity(model_id: str = "nvidia/GR00T-N1.6-3B"):
    import time
    import numpy as np
    import torch

    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t_exporter import build_gr00t_full_stack

    print(f"[parity] Loading {model_id} via reflex path...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    ours, meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
    ours = ours.float().eval().to("cuda")
    print(f"[parity] Ours built in {time.time()-t0:.1f}s, "
          f"has_state_encoder={meta.get('has_state_encoder')}")

    # ─── Seeded synthetic inputs ─────────────────────────────────
    torch.manual_seed(42)
    B = 1
    chunk = 50
    raw_action_dim = meta["raw_action_dim"]      # 128
    raw_state_dim = meta["raw_state_dim"]        # 128
    vlm_kv_dim = 2048
    vlm_seq_len = 256

    noisy_actions = torch.randn(B, chunk, raw_action_dim, device="cuda")
    timestep = torch.tensor([0.5], device="cuda")
    state = torch.randn(B, raw_state_dim, device="cuda")
    vlm_kv = torch.randn(B, vlm_seq_len, vlm_kv_dim, device="cuda")
    position_ids = torch.arange(chunk + 1, device="cuda").unsqueeze(0)

    # ─── Path A: our GR00TFullStack ─────────────────────────────
    print(f"\n[parity] Path A (our extended GR00TFullStack)")
    with torch.no_grad():
        actions_a = ours(noisy_actions, timestep, position_ids,
                           state=state, vlm_kv=vlm_kv)
    print(f"         shape={tuple(actions_a.shape)}, "
          f"mean={actions_a.mean().item():+.4f} std={actions_a.std().item():.4f}")
    print(f"         first_action = {actions_a[0, 0, :7].cpu().numpy().tolist()}")

    # ─── Path B: lerobot's action head (reference) ───────────────
    print(f"\n[parity] Path B (lerobot's FlowMatchingActionHead reference)")
    t0 = time.time()
    # lerobot 0.5.1 names the class with lowercase 'm': FlowmatchingActionHead.
    # (Not FlowMatchingActionHead.) Importing it directly isn't strictly
    # required since we load via GrootPolicy.from_pretrained below — the
    # class is present on the loaded policy regardless.
    try:
        from lerobot.policies.groot.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )
    except ImportError as e:
        print(f"[parity] Flowmatching import note: {e}")
        # Non-fatal — we only need GrootPolicy.from_pretrained which brings
        # in the correct class transparently.

    # Instantiate from config. N1.6 doesn't bundle action_head_config.json
    # directly — we need to construct the FlowMatchingActionHeadConfig.
    # For a pragmatic reference, we'll load lerobot's GR00TN15 (the full
    # wrapper) via from_pretrained which handles all init internally.
    # lerobot 0.5.1 exposes GR00TN15 (not GrootPolicy). Verified via
    # modal_gr00t_lerobot_probe.py. GR00TN15 is the top-level
    # HF-compatible class with .from_pretrained support.
    try:
        from lerobot.policies.groot.groot_n1 import GR00TN15
        policy = GR00TN15.from_pretrained(model_id)
        policy.eval().to("cuda").to(torch.float32)
        print(f"[parity] Lerobot GR00TN15 loaded in {time.time()-t0:.1f}s")
        action_head = policy.action_head
    except Exception as e:
        import traceback
        print(f"[parity] GR00TN15 load failed ({type(e).__name__}: {e})")
        print(traceback.format_exc()[-1200:])
        print(f"[parity] Cannot do full reference parity — reporting Path A only")
        return {
            "status": "partial",
            "path_a_first_action": actions_a[0, 0].cpu().numpy().tolist(),
            "path_a_mean_abs": float(actions_a.abs().mean()),
            "reason": f"lerobot reference load failed: {type(e).__name__}: {str(e)[:200]}",
        }

    # Single-step reference forward, mirroring lerobot's get_action body
    # but fed externally-provided vl_embs + skipping denoise loop.
    # Reference implementation replicates lines 356-400 of
    # flow_matching_action_head.py (get_action method).
    embodiment_id = torch.tensor([0], device="cuda", dtype=torch.long)

    with torch.no_grad():
        state_features = action_head.state_encoder(state, embodiment_id)  # [B, 1, 1536]
        print(f"[parity] lerobot state_features shape: {tuple(state_features.shape)}")

        # Replicate the denoise-step forward (single iteration)
        num_timestep_buckets = action_head.num_timestep_buckets
        t_discretized = torch.tensor([int(0.5 * num_timestep_buckets)], device="cuda")
        action_features = action_head.action_encoder(
            noisy_actions, t_discretized, embodiment_id,
        )  # [B, chunk, 1536]
        print(f"[parity] lerobot action_features shape: {tuple(action_features.shape)}")

        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device="cuda")
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # CRITICAL: check whether action_head has future_tokens — N1.6's
        # checkpoint has NO future_tokens in its state dict (verified via
        # modal_gr00t_keys_dump.py). If lerobot's code has it, handle both.
        has_future_tokens = (
            hasattr(action_head, "future_tokens")
            and action_head.future_tokens.weight.abs().sum() > 0
        )
        print(f"[parity] lerobot action_head.future_tokens present + nonzero: "
              f"{has_future_tokens}")

        if has_future_tokens:
            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
                B, -1, -1,
            )
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            ft_len = future_tokens.shape[1]
            print(f"[parity] lerobot sa_embs includes {ft_len} future_tokens")
        else:
            sa_embs = torch.cat((state_features, action_features), dim=1)
            ft_len = 0
            print(f"[parity] lerobot sa_embs: state + actions only (no future_tokens)")

        # Feed to DiT. Lerobot uses encoder_hidden_states=vl_embs.
        vl_attn_mask = torch.ones(B, vlm_seq_len, device="cuda", dtype=torch.bool)
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vlm_kv,  # feed the same vlm_kv we fed Path A
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred_b = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity_b = pred_b[:, -chunk:]  # take last action_horizon tokens

    print(f"\n[parity] Path B actions:")
    print(f"         shape={tuple(pred_velocity_b.shape)}, "
          f"mean={pred_velocity_b.mean().item():+.4f} "
          f"std={pred_velocity_b.std().item():.4f}")
    print(f"         first_action = {pred_velocity_b[0, 0, :7].cpu().numpy().tolist()}")

    # ─── Compare ─────────────────────────────────────────────────
    a = actions_a.reshape(-1)
    b = pred_velocity_b.reshape(-1)
    max_abs = float((a - b).abs().max())
    cos = float(
        (a * b).sum() / (a.norm() * b.norm() + 1e-12)
    )
    print(f"\n====== PARITY RESULT ======")
    print(f"  cos:        {cos:+.6f}")
    print(f"  max_abs:    {max_abs:.4e}")
    print(f"  mean_abs:   {float(a.abs().mean()):.4e}")
    if cos >= 0.9999 and max_abs < 1e-4:
        verdict = "PASS (bit-exact)"
    elif cos >= 0.99:
        verdict = "CLOSE but not bit-exact"
    else:
        verdict = f"FAIL — real divergence (cos={cos:+.4f})"
    print(f"  VERDICT: {verdict}")

    return {
        "status": "ok",
        "cos": cos,
        "max_abs": max_abs,
        "has_future_tokens": has_future_tokens,
        "ft_len": ft_len,
        "path_a_first": actions_a[0, 0, :5].cpu().numpy().tolist(),
        "path_b_first": pred_velocity_b[0, 0, :5].cpu().numpy().tolist(),
        "verdict": verdict,
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("GR00T extended GR00TFullStack vs lerobot action_head parity")
    print("=" * 60)
    r = run_parity.remote()
    print(f"\n=== RESULT ===")
    for k, v in r.items():
        if isinstance(v, list):
            print(f"  {k}: {[round(x, 4) for x in v]}")
        else:
            print(f"  {k}: {v}")
