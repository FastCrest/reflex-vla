"""GR00T N1.6 export pipeline.

Exports the DiT action expert of nvidia/GR00T-N1.6-3B. Differs from
SmolVLA/pi0/pi0.5 in:

- 32 DiT transformer blocks with diffusers-style attention naming
  (`attn1.to_q/to_k/to_v/to_out.0`, bias enabled)
- 32-head MHA (no GQA), head_dim=48, hidden=1536
- Plain GELU-approx MLP feed-forward (ff.net.0.proj: Linear(hidden, inner) + GELU;
  ff.net.2: Linear(inner, hidden)). GR00T does NOT use GEGLU despite early notes.
- AdaLN (2-chunk: scale + shift), NOT AdaRMSNorm like pi0.5 (which has 3 chunks)
- Alternating cross-attn (even blocks) / self-attn (odd blocks). Cross-attn
  blocks consume VLM KV at 2048-dim.
- Final AdaLN + linear output head (1536 → 1024 action tokens)
- Per-embodiment state/action encoders (32 embodiments); we pin to
  embodiment_id 0 at export time.

For v0.1 we export the expert without VLM conditioning — the VLM KV is
a zero placeholder (same convention as SmolVLA/pi0 exports). This gives
valid actions when prompted with a dummy instruction but real control
needs VLM conditioning, which is a follow-up.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from reflex.config import ExportConfig, get_hardware_profile
from reflex.checkpoint import load_checkpoint
from reflex.exporters.onnx_export import export_module_to_onnx, optimize_onnx
from reflex.exporters.trt_build import build_engine, check_trtexec

logger = logging.getLogger(__name__)


GR00T_BLOCK_PREFIX = "action_head.model.transformer_blocks."
GR00T_META_KEYS = {
    "timestep_linear_1_w": "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight",
    "timestep_linear_1_b": "action_head.model.timestep_encoder.timestep_embedder.linear_1.bias",
    "timestep_linear_2_w": "action_head.model.timestep_encoder.timestep_embedder.linear_2.weight",
    "timestep_linear_2_b": "action_head.model.timestep_encoder.timestep_embedder.linear_2.bias",
    "proj_out_1_w": "action_head.model.proj_out_1.weight",
    "proj_out_1_b": "action_head.model.proj_out_1.bias",
    "proj_out_2_w": "action_head.model.proj_out_2.weight",
    "proj_out_2_b": "action_head.model.proj_out_2.bias",
    "pos_embed": "action_head.position_embedding.weight",
    "vlln_w": "action_head.vlln.weight",
    "vlln_b": "action_head.vlln.bias",
}


def _sinusoidal_timestep(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """Sinusoidal embedding used inside diffusers TimestepEmbedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)


class GR00TDiTBlock(nn.Module):
    """One DiT block: AdaLN → (cross-attn or self-attn) → residual →
    LayerNorm(non-affine) → GEGLU-FF → residual.

    Cross-attn blocks accept VLM KV at a possibly-different `kv_in` dim;
    self-attn blocks always use kv_in == hidden.
    """

    def __init__(self, hidden: int, num_heads: int, head_dim: int, ff_inner: int,
                 kv_in: int, is_cross: bool):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.is_cross = is_cross

        # AdaLN: time embedding → scale, shift (2 chunks, NOT 3 like pi0.5)
        self.norm1_linear = nn.Linear(hidden, 2 * hidden, bias=True)

        # Attention (diffusers naming)
        self.to_q = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.to_k = nn.Linear(kv_in, num_heads * head_dim, bias=True)
        self.to_v = nn.Linear(kv_in, num_heads * head_dim, bias=True)
        self.to_out_0 = nn.Linear(num_heads * head_dim, hidden, bias=True)

        # GELU-approx MLP: ff.net.0.proj → gelu_approx → ff.net.2. NOT GEGLU.
        self.ff_net_0_proj = nn.Linear(hidden, ff_inner, bias=True)
        self.ff_net_2 = nn.Linear(ff_inner, hidden, bias=True)

    def forward(self, x: torch.Tensor, temb: torch.Tensor,
                encoder_kv: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape

        # AdaLN pre-attn modulation
        scale_shift = self.norm1_linear(F.silu(temb))  # [b, 2*hidden]
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        # Non-affine LayerNorm (elementwise_affine=False)
        x_n = F.layer_norm(x, (self.hidden,))
        x_n = (1 + scale) * x_n + shift

        # Attention: Q always from action tokens; K/V from encoder (cross) or x (self)
        kv_src = encoder_kv if (self.is_cross and encoder_kv is not None) else x_n
        q = self.to_q(x_n)
        k = self.to_k(kv_src)
        v = self.to_v(kv_src)

        # Reshape for multi-head
        seq_kv = kv_src.shape[1]
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1
        )
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
        x = x + self.to_out_0(attn_out)

        # Post-attn non-affine LayerNorm + GELU-approx MLP
        x_n = F.layer_norm(x, (self.hidden,))
        h = F.gelu(self.ff_net_0_proj(x_n), approximate="tanh")
        ff_out = self.ff_net_2(h)
        return x + ff_out


class GR00TExpertStack(nn.Module):
    """Full GR00T DiT action expert for ONNX export.

    Inputs (for a single denoising step):
        noisy_action_tokens: [b, chunk, hidden]   — pre-encoded action tokens
        timestep: [b]                              — scalar in [0, 1]
        position_ids: [b, chunk]
        vlm_kv (optional): [b, seq_kv, vlm_kv_dim] — for cross-attn blocks

    Output:
        velocity: [b, chunk, 1024]  — pre-decoder action-token velocities

    NOTE: the exporter wraps this to accept raw noisy actions (action_dim) and
    emit velocities in the same space via the per-embodiment encoder/decoder.
    """

    def __init__(self, blocks: list[GR00TDiTBlock], hidden: int,
                 pos_embed_weight: torch.Tensor,
                 timestep_mlp_weights: dict,
                 proj_out_weights: dict,
                 vlln_weights: dict,
                 vlm_kv_dim: int = 2048,
                 output_dim: int = 1024):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.hidden = hidden
        self.vlm_kv_dim = vlm_kv_dim

        # Position embedding as a lookup table
        self.register_buffer("pos_embed", pos_embed_weight.clone())

        # Timestep MLP (256-dim sinusoidal → linear_1 → silu → linear_2)
        self.sinusoidal_dim = timestep_mlp_weights["in_dim"]
        self.timestep_linear_1 = nn.Linear(
            timestep_mlp_weights["in_dim"], timestep_mlp_weights["mid_dim"], bias=True,
        )
        self.timestep_linear_2 = nn.Linear(
            timestep_mlp_weights["mid_dim"], hidden, bias=True,
        )
        self.timestep_linear_1.weight = nn.Parameter(timestep_mlp_weights["l1_w"])
        self.timestep_linear_1.bias = nn.Parameter(timestep_mlp_weights["l1_b"])
        self.timestep_linear_2.weight = nn.Parameter(timestep_mlp_weights["l2_w"])
        self.timestep_linear_2.bias = nn.Parameter(timestep_mlp_weights["l2_b"])

        # Final AdaLN (proj_out_1: Linear(hidden, 2*hidden) for scale+shift)
        self.proj_out_1 = nn.Linear(hidden, 2 * hidden, bias=True)
        self.proj_out_1.weight = nn.Parameter(proj_out_weights["p1_w"])
        self.proj_out_1.bias = nn.Parameter(proj_out_weights["p1_b"])
        # proj_out_2: Linear(hidden, output_dim) for velocity
        self.proj_out_2 = nn.Linear(hidden, output_dim, bias=True)
        self.proj_out_2.weight = nn.Parameter(proj_out_weights["p2_w"])
        self.proj_out_2.bias = nn.Parameter(proj_out_weights["p2_b"])

        # VLLN: LayerNorm(vlm_kv_dim) on backbone features before cross-attn consumes them
        self.vlln = nn.LayerNorm(vlm_kv_dim)
        self.vlln.weight = nn.Parameter(vlln_weights["w"])
        self.vlln.bias = nn.Parameter(vlln_weights["b"])

    def forward(self, action_tokens: torch.Tensor, timestep: torch.Tensor,
                position_ids: torch.Tensor,
                vlm_kv: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = action_tokens.shape

        # Add position embeddings
        pos = self.pos_embed[position_ids]  # [b, s, hidden]
        x = action_tokens + pos

        # Timestep embedding: sinusoidal → linear_1 → silu → linear_2
        t_sin = _sinusoidal_timestep(timestep, self.sinusoidal_dim)
        temb = self.timestep_linear_2(F.silu(self.timestep_linear_1(t_sin)))

        # VLM features go through VLLN (scale/shift normalization) before cross-attn consumes them
        vlm_kv_normed = None
        if vlm_kv is not None:
            vlm_kv_normed = self.vlln(vlm_kv)
        else:
            # Use a length-1 zero placeholder (valid for export without VLM conditioning)
            vlm_kv_normed = torch.zeros(b, 1, self.vlm_kv_dim, device=x.device, dtype=x.dtype)

        # 32 DiT blocks: alternating cross (even) / self (odd)
        for i, block in enumerate(self.blocks):
            x = block(x, temb, vlm_kv_normed if block.is_cross else None)

        # Final AdaLN
        scale_shift = self.proj_out_1(F.silu(temb))
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        x = (1 + scale) * F.layer_norm(x, (self.hidden,)) + shift

        # Output projection
        return self.proj_out_2(x)


def build_gr00t_expert_stack(
    state_dict: dict[str, torch.Tensor],
    embodiment_id: int = 0,
) -> tuple[GR00TExpertStack, dict]:
    """Build GR00T N1.6 DiT expert stack from state_dict.

    Args:
        state_dict: full GR00T checkpoint tensors
        embodiment_id: which embodiment's state/action encoders to use (0..31)

    Returns (stack, metadata).
    """
    # 1. Find DiT blocks
    layer_indices = set()
    for k in state_dict.keys():
        if not k.startswith(GR00T_BLOCK_PREFIX):
            continue
        rest = k[len(GR00T_BLOCK_PREFIX):]
        parts = rest.split(".")
        if parts and parts[0].isdigit():
            layer_indices.add(int(parts[0]))
    if not layer_indices:
        raise ValueError(f"No GR00T DiT blocks found under {GR00T_BLOCK_PREFIX}")
    num_layers = max(layer_indices) + 1

    # 2. Infer shapes from block 0
    q_w = state_dict[f"{GR00T_BLOCK_PREFIX}0.attn1.to_q.weight"]
    hidden = q_w.shape[1]  # input dim of to_q = hidden
    num_q = q_w.shape[0]   # output = num_heads * head_dim

    # Block 0 is a cross-attn block (even idx); its to_k kv_in reveals vlm_kv_dim
    k_w_0 = state_dict[f"{GR00T_BLOCK_PREFIX}0.attn1.to_k.weight"]
    vlm_kv_dim = k_w_0.shape[1]  # for cross-attn block

    # Block 1 is self-attn (odd idx); kv_in == hidden
    k_w_1 = state_dict[f"{GR00T_BLOCK_PREFIX}1.attn1.to_k.weight"]
    self_kv_in = k_w_1.shape[1]

    # Infer num_heads by factoring num_q; head_dim=48 is the convention from config
    head_dim = 48
    num_heads = num_q // head_dim
    if num_heads * head_dim != num_q:
        # Try other common head_dims
        for hd in (64, 32, 96, 128):
            if num_q % hd == 0:
                head_dim, num_heads = hd, num_q // hd
                break

    # FF inner from ff.net.0.proj.weight (shape [inner, hidden]) — plain MLP, not GEGLU
    ff_w = state_dict[f"{GR00T_BLOCK_PREFIX}0.ff.net.0.proj.weight"]
    ff_inner = ff_w.shape[0]

    logger.info(
        "GR00T DiT: %d blocks, hidden=%d, heads=%d × hd=%d, ff_inner=%d, vlm_kv_dim=%d",
        num_layers, hidden, num_heads, head_dim, ff_inner, vlm_kv_dim,
    )

    # 3. Build blocks with proper kv_in per block
    blocks = []
    for i in range(num_layers):
        is_cross = (i % 2 == 0)  # even = cross-attn
        kv_in = vlm_kv_dim if is_cross else hidden
        block = GR00TDiTBlock(hidden, num_heads, head_dim, ff_inner, kv_in, is_cross)
        prefix = f"{GR00T_BLOCK_PREFIX}{i}"
        block_sd = {
            "norm1_linear.weight": state_dict[f"{prefix}.norm1.linear.weight"],
            "norm1_linear.bias": state_dict[f"{prefix}.norm1.linear.bias"],
            "to_q.weight": state_dict[f"{prefix}.attn1.to_q.weight"],
            "to_q.bias": state_dict[f"{prefix}.attn1.to_q.bias"],
            "to_k.weight": state_dict[f"{prefix}.attn1.to_k.weight"],
            "to_k.bias": state_dict[f"{prefix}.attn1.to_k.bias"],
            "to_v.weight": state_dict[f"{prefix}.attn1.to_v.weight"],
            "to_v.bias": state_dict[f"{prefix}.attn1.to_v.bias"],
            "to_out_0.weight": state_dict[f"{prefix}.attn1.to_out.0.weight"],
            "to_out_0.bias": state_dict[f"{prefix}.attn1.to_out.0.bias"],
            "ff_net_0_proj.weight": state_dict[f"{prefix}.ff.net.0.proj.weight"],
            "ff_net_0_proj.bias": state_dict[f"{prefix}.ff.net.0.proj.bias"],
            "ff_net_2.weight": state_dict[f"{prefix}.ff.net.2.weight"],
            "ff_net_2.bias": state_dict[f"{prefix}.ff.net.2.bias"],
        }
        block.load_state_dict(block_sd, strict=True)
        blocks.append(block)

    # 4. Timestep MLP
    l1_w = state_dict[GR00T_META_KEYS["timestep_linear_1_w"]]
    sin_dim = l1_w.shape[1]  # input to linear_1 is sinusoidal dim
    mid_dim = l1_w.shape[0]

    timestep_mlp = {
        "in_dim": sin_dim,
        "mid_dim": mid_dim,
        "l1_w": l1_w,
        "l1_b": state_dict[GR00T_META_KEYS["timestep_linear_1_b"]],
        "l2_w": state_dict[GR00T_META_KEYS["timestep_linear_2_w"]],
        "l2_b": state_dict[GR00T_META_KEYS["timestep_linear_2_b"]],
    }

    proj_out_weights = {
        "p1_w": state_dict[GR00T_META_KEYS["proj_out_1_w"]],
        "p1_b": state_dict[GR00T_META_KEYS["proj_out_1_b"]],
        "p2_w": state_dict[GR00T_META_KEYS["proj_out_2_w"]],
        "p2_b": state_dict[GR00T_META_KEYS["proj_out_2_b"]],
    }

    vlln_weights = {
        "w": state_dict[GR00T_META_KEYS["vlln_w"]],
        "b": state_dict[GR00T_META_KEYS["vlln_b"]],
    }

    output_dim = proj_out_weights["p2_w"].shape[0]  # usually 1024 pre-decoder

    stack = GR00TExpertStack(
        blocks=blocks,
        hidden=hidden,
        pos_embed_weight=state_dict[GR00T_META_KEYS["pos_embed"]],
        timestep_mlp_weights=timestep_mlp,
        proj_out_weights=proj_out_weights,
        vlln_weights=vlln_weights,
        vlm_kv_dim=vlm_kv_dim,
        output_dim=output_dim,
    )
    # GR00T weights are stored in bf16. Cast the whole stack to fp32 for the ONNX
    # export path — TRT/ORT can do fp16 at engine-build time; fp32 is the portable
    # source of truth.
    stack = stack.float()
    stack.eval()

    # GR00T action horizon from config is 50; pos_embed is oversized (1024) to
    # accommodate other tokens. Use 50 as the real chunk_size for export dummies.
    max_pos = state_dict[GR00T_META_KEYS["pos_embed"]].shape[0]
    chunk_size = 50

    metadata = {
        "num_layers": num_layers,
        "hidden": hidden,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "ff_inner": ff_inner,
        "vlm_kv_dim": vlm_kv_dim,
        "output_dim": output_dim,
        "chunk_size": chunk_size,
        "max_pos_seq_len": max_pos,
        "embodiment_id": embodiment_id,
        "total_params_m": sum(p.numel() for p in stack.parameters()) / 1e6,
    }
    return stack, metadata


def export_gr00t(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Full GR00T N1.6 expert export pipeline.

    Exports the DiT expert (32 blocks) with zero VLM-KV placeholder.
    Output is the [b, chunk, output_dim] action-token velocity — consumers
    need to run the per-embodiment action_decoder downstream to recover
    actions in the native DoF space.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = get_hardware_profile(config.target)
    result = {"status": "ok", "files": {}, "metadata": {}}

    if state_dict is None:
        logger.info("Loading GR00T checkpoint: %s", config.model_id)
        state_dict, _ = load_checkpoint(config.model_id)
    total_params = sum(v.numel() for v in state_dict.values())
    logger.info("Loaded %d tensors, %.1fM params", len(state_dict), total_params / 1e6)

    logger.info("Building GR00T expert stack...")
    expert_stack, meta = build_gr00t_expert_stack(state_dict, embodiment_id=0)
    result["metadata"]["expert"] = meta

    chunk_size = meta["chunk_size"]
    hidden = meta["hidden"]
    dummy_action_tokens = torch.randn(1, chunk_size, hidden)
    dummy_time = torch.tensor([0.5])
    dummy_pos = torch.arange(chunk_size).unsqueeze(0)

    expert_onnx = output_dir / "expert_stack.onnx"
    export_module_to_onnx(
        expert_stack,
        (dummy_action_tokens, dummy_time, dummy_pos),
        expert_onnx,
        input_names=["noisy_actions", "timestep", "position_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
        },
        opset_version=config.opset,
    )
    optimize_onnx(expert_onnx)
    result["files"]["expert_onnx"] = str(expert_onnx)

    if config.validate:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(expert_onnx))
            ort_out = sess.run(None, {
                "noisy_actions": dummy_action_tokens.numpy(),
                "timestep": dummy_time.numpy(),
                "position_ids": dummy_pos.numpy().astype(np.int64),
            })[0]
            torch_out = expert_stack(dummy_action_tokens, dummy_time, dummy_pos).detach().numpy()
            max_diff = float(np.abs(ort_out - torch_out).max())
            result["metadata"]["onnx_validation"] = {"max_diff": max_diff, "passed": max_diff < 0.01}
            logger.info("ONNX validation: max_diff=%.2e (%s)", max_diff, "PASS" if max_diff < 0.01 else "FAIL")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")

    if check_trtexec():
        expert_trt = output_dir / "expert_stack.trt"
        try:
            build_engine(expert_onnx, expert_trt, hardware)
            result["files"]["expert_trt"] = str(expert_trt)
        except RuntimeError as e:
            logger.warning("TRT build failed: %s", e)

    # For serve-path compatibility: reflex serve samples noise of shape
    # [b, chunk, action_dim] and pipes it to the ONNX. GR00T's expert accepts
    # action-token input (pre-encoded, `hidden`-dim) rather than raw actions,
    # so we surface `action_dim=hidden` at top-level for the server to allocate.
    # The `output_dim` field records the real post-expert dim (1024) for
    # downstream action_decoder usage.
    meta_with_action_dim = dict(meta)
    meta_with_action_dim["action_dim"] = hidden  # server uses this to shape input
    export_config = {
        "model_id": config.model_id,
        "model_type": "gr00t",
        "target": config.target,
        "precision": config.precision,
        "opset": config.opset,
        "num_denoising_steps": 4,  # GR00T config says 4 inference steps
        "action_chunk_size": chunk_size,
        "action_dim": hidden,  # action-token dim (not native DoF)
        "hidden": hidden,
        "output_dim": meta["output_dim"],
        "note": "expert accepts action tokens (hidden-dim), emits velocity tokens (output_dim). "
                "action_decoder (per-embodiment) needed downstream to recover native actions.",
        "hardware": {
            "name": hardware.name,
            "memory_gb": hardware.memory_gb,
            "fp8": hardware.fp8_support,
            "precision": hardware.trt_precision,
        },
        "expert": meta_with_action_dim,
    }
    config_path = output_dir / "reflex_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)
    return result
