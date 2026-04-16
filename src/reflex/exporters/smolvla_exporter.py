"""SmolVLA full export pipeline: checkpoint → ONNX → TRT engines.

Handles the complete flow:
1. Load SmolVLA checkpoint from HF Hub or local path
2. Reconstruct VLM backbone + action expert from weights
3. Decompose RMSNorm/RoPE for ONNX compatibility
4. Export 3 components to ONNX (vision, backbone, expert stack)
5. Optionally build TensorRT engines
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from reflex.config import ExportConfig, HardwareProfile, get_hardware_profile
from reflex.checkpoint import load_checkpoint
from reflex.decompose import DecomposedRMSNorm, DecomposedRotaryEmbedding
from reflex.exporters.onnx_export import export_module_to_onnx, optimize_onnx
from reflex.exporters.trt_build import build_engine, check_trtexec

logger = logging.getLogger(__name__)


def _sinusoidal_pos_embedding(t, dim, min_p=4e-3, max_p=4.0):
    exp = torch.linspace(0, 1, dim // 2, device=t.device, dtype=t.dtype)
    freq = torch.exp(exp * (math.log(min_p) - math.log(max_p))) / min_p
    angle = t.unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([angle.cos(), angle.sin()], dim=-1)


class _DecomposedRoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
        self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

    def apply(self, x, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class ExpertGQALayer(nn.Module):
    """Single expert transformer layer with decomposed ops for ONNX export."""

    def __init__(self, hidden, nq, nkv, hd, inter, kv_in=None):
        super().__init__()
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.kv_groups = nq // nkv
        self.input_layernorm = DecomposedRMSNorm(torch.ones(hidden))
        self.post_attention_layernorm = DecomposedRMSNorm(torch.ones(hidden))
        self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
        self.k_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
        self.v_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.rope = _DecomposedRoPE(hd)

    def forward(self, x, pos_ids, cross_kv=None):
        b, s, _ = x.shape
        res = x
        x = self.input_layernorm(x)
        q = self.q_proj(x).view(b, s, self.nq, self.hd).transpose(1, 2)
        kv_src = cross_kv if cross_kv is not None else x
        kv_len = kv_src.shape[1]
        k = self.k_proj(kv_src).view(b, kv_len, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(kv_src).view(b, kv_len, self.nkv, self.hd).transpose(1, 2)
        q = self.rope.apply(q, pos_ids)
        if cross_kv is None:
            k = self.rope.apply(k, pos_ids)
        k = k.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)
        v = v.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd), dim=-1)
        x = res + self.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1))
        res = x
        x = self.post_attention_layernorm(x)
        return res + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertStack(nn.Module):
    """Full expert stack for ONNX export (single denoising step)."""

    def __init__(self, layers, expert_hidden, action_dim, cross_indices, vlm_kv_dim,
                 suffix_weights, action_proj_weights, final_norm_weight):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.expert_hidden = expert_hidden
        self.cross_indices = set(cross_indices)
        self.vlm_kv_dim = vlm_kv_dim

        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        self.action_in_proj.weight = nn.Parameter(suffix_weights["in_w"])
        self.action_in_proj.bias = nn.Parameter(suffix_weights["in_b"])
        self.action_time_mlp_in.weight = nn.Parameter(suffix_weights["t_in_w"])
        self.action_time_mlp_in.bias = nn.Parameter(suffix_weights["t_in_b"])
        self.action_time_mlp_out.weight = nn.Parameter(suffix_weights["t_out_w"])
        self.action_time_mlp_out.bias = nn.Parameter(suffix_weights["t_out_b"])

        self.action_out_proj = nn.Linear(expert_hidden, action_dim)
        self.action_out_proj.weight = nn.Parameter(action_proj_weights["w"])
        self.action_out_proj.bias = nn.Parameter(action_proj_weights["b"])

        self.final_norm = DecomposedRMSNorm(final_norm_weight)

    def forward(self, noisy_actions, timestep, position_ids, vlm_kv: torch.Tensor | None = None):
        b, c, _ = noisy_actions.shape
        act = self.action_in_proj(noisy_actions)
        t_emb = _sinusoidal_pos_embedding(timestep, self.expert_hidden)
        t_emb = t_emb.unsqueeze(1).expand(-1, c, -1)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([act, t_emb], dim=-1))))

        # Use provided VLM KV or fall back to zeros for backward compat
        if vlm_kv is None:
            cross_kv = torch.zeros(b, 1, self.vlm_kv_dim, device=x.device, dtype=x.dtype) if (self.cross_indices and self.vlm_kv_dim > 0) else None
        else:
            cross_kv = vlm_kv

        for i, layer in enumerate(self.layers):
            if i in self.cross_indices:
                x = layer(x, position_ids, cross_kv=cross_kv)
            else:
                x = layer(x, position_ids)

        x = self.final_norm(x)
        return self.action_out_proj(x)


def build_expert_stack(state_dict: dict[str, torch.Tensor], head_dim: int) -> tuple[ExpertStack, dict]:
    """Build the full expert stack from SmolVLA state_dict."""
    expert_hidden = state_dict["model.action_in_proj.weight"].shape[0]
    action_dim = state_dict["model.action_in_proj.weight"].shape[1]

    all_expert_keys = [k for k in state_dict.keys() if "lm_expert" in k]
    base_prefix = all_expert_keys[0][: all_expert_keys[0].index("layers.")]

    layer_indices = set()
    for k in all_expert_keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_indices.add(int(parts[i + 1]))
    num_layers = max(layer_indices) + 1

    q_shape = state_dict[f"{base_prefix}layers.0.self_attn.q_proj.weight"].shape
    k_shape = state_dict[f"{base_prefix}layers.0.self_attn.k_proj.weight"].shape
    gate_shape = state_dict[f"{base_prefix}layers.0.mlp.gate_proj.weight"].shape
    nq = q_shape[0] // head_dim
    nkv = k_shape[0] // head_dim
    inter = gate_shape[0]

    layers = []
    cross_indices = []
    vlm_kv_dim = 0
    for i in range(num_layers):
        prefix = f"{base_prefix}layers.{i}"
        kv_in = state_dict[f"{prefix}.self_attn.k_proj.weight"].shape[1]
        is_cross = kv_in != expert_hidden
        if is_cross:
            cross_indices.append(i)
            vlm_kv_dim = kv_in

        layer = ExpertGQALayer(expert_hidden, nq, nkv, head_dim, inter,
                               kv_in=kv_in if is_cross else None)
        layer_sd = {
            "input_layernorm.weight": state_dict[f"{prefix}.input_layernorm.weight"],
            "post_attention_layernorm.weight": state_dict[f"{prefix}.post_attention_layernorm.weight"],
            "q_proj.weight": state_dict[f"{prefix}.self_attn.q_proj.weight"],
            "k_proj.weight": state_dict[f"{prefix}.self_attn.k_proj.weight"],
            "v_proj.weight": state_dict[f"{prefix}.self_attn.v_proj.weight"],
            "o_proj.weight": state_dict[f"{prefix}.self_attn.o_proj.weight"],
            "gate_proj.weight": state_dict[f"{prefix}.mlp.gate_proj.weight"],
            "up_proj.weight": state_dict[f"{prefix}.mlp.up_proj.weight"],
            "down_proj.weight": state_dict[f"{prefix}.mlp.down_proj.weight"],
        }
        layer.load_state_dict(layer_sd, strict=False)
        layers.append(layer)

    final_norm_w = torch.ones(expert_hidden)
    for candidate in [f"{base_prefix}norm.weight", "model.vlm_with_expert.lm_expert.norm.weight"]:
        if candidate in state_dict:
            final_norm_w = state_dict[candidate]
            break

    stack = ExpertStack(
        layers=layers,
        expert_hidden=expert_hidden,
        action_dim=action_dim,
        cross_indices=cross_indices,
        vlm_kv_dim=vlm_kv_dim,
        suffix_weights={
            "in_w": state_dict["model.action_in_proj.weight"],
            "in_b": state_dict["model.action_in_proj.bias"],
            "t_in_w": state_dict["model.action_time_mlp_in.weight"],
            "t_in_b": state_dict["model.action_time_mlp_in.bias"],
            "t_out_w": state_dict["model.action_time_mlp_out.weight"],
            "t_out_b": state_dict["model.action_time_mlp_out.bias"],
        },
        action_proj_weights={
            "w": state_dict["model.action_out_proj.weight"],
            "b": state_dict["model.action_out_proj.bias"],
        },
        final_norm_weight=final_norm_w,
    )
    stack.eval()

    metadata = {
        "expert_hidden": expert_hidden,
        "action_dim": action_dim,
        "num_layers": num_layers,
        "n_q_heads": nq,
        "n_kv_heads": nkv,
        "head_dim": head_dim,
        "intermediate": inter,
        "cross_attn_layers": cross_indices,
        "vlm_kv_dim": vlm_kv_dim,
        "total_params_m": sum(p.numel() for p in stack.parameters()) / 1e6,
    }
    return stack, metadata


def export_smolvla(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Full SmolVLA export pipeline.

    Args:
        config: export config
        state_dict: optionally pre-loaded state dict (avoids re-downloading)
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = get_hardware_profile(config.target)
    result = {"status": "ok", "files": {}, "metadata": {}}

    # 1. Load checkpoint (if not provided)
    if state_dict is None:
        logger.info("Loading checkpoint: %s", config.model_id)
        state_dict, model_config = load_checkpoint(config.model_id)
    total_params = sum(v.numel() for v in state_dict.values())
    logger.info("Loaded %d tensors, %.1fM params", len(state_dict), total_params / 1e6)

    # 2. Get VLM config for head_dim
    try:
        from transformers import AutoConfig
        vlm_cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        head_dim = vlm_cfg.text_config.hidden_size // vlm_cfg.text_config.num_attention_heads
    except Exception:
        head_dim = 64  # SmolVLA default

    # 3. Build expert stack
    logger.info("Building expert stack...")
    expert_stack, expert_meta = build_expert_stack(state_dict, head_dim)
    result["metadata"]["expert"] = expert_meta
    logger.info(
        "Expert: %d layers, %.1fM params, cross_attn=%s",
        expert_meta["num_layers"], expert_meta["total_params_m"], expert_meta["cross_attn_layers"],
    )

    # 4. Export expert stack to ONNX
    logger.info("Exporting expert stack to ONNX...")
    action_dim = expert_meta["action_dim"]
    chunk_size = config.action_chunk_size

    dummy_actions = torch.randn(1, chunk_size, action_dim)
    dummy_time = torch.tensor([0.5])
    dummy_pos = torch.arange(chunk_size).unsqueeze(0)
    vlm_kv_dim = expert_meta["vlm_kv_dim"]
    dummy_vlm_kv = torch.zeros(1, 1, vlm_kv_dim)

    expert_onnx = output_dir / "expert_stack.onnx"
    export_module_to_onnx(
        expert_stack,
        (dummy_actions, dummy_time, dummy_pos, dummy_vlm_kv),
        expert_onnx,
        input_names=["noisy_actions", "timestep", "position_ids", "vlm_kv"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"}, "timestep": {0: "batch"},
            "position_ids": {0: "batch"}, "vlm_kv": {0: "batch", 1: "seq"},
        },
        opset_version=config.opset,
    )
    optimize_onnx(expert_onnx)
    result["files"]["expert_onnx"] = str(expert_onnx)

    # 5. Validate ONNX
    if config.validate:
        logger.info("Validating ONNX export...")
        try:
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(str(expert_onnx))
            ort_out = sess.run(None, {
                "noisy_actions": dummy_actions.numpy(),
                "timestep": dummy_time.numpy(),
                "position_ids": dummy_pos.numpy().astype(np.int64),
                "vlm_kv": dummy_vlm_kv.numpy(),
            })[0]
            torch_out = expert_stack(dummy_actions, dummy_time, dummy_pos, dummy_vlm_kv).detach().numpy()
            max_diff = float(np.abs(ort_out - torch_out).max())
            result["metadata"]["onnx_validation"] = {"max_diff": max_diff, "passed": max_diff < 0.01}
            logger.info("ONNX validation: max_diff=%.2e (%s)", max_diff, "PASS" if max_diff < 0.01 else "FAIL")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")

    # 6. Build TRT engine if available
    if check_trtexec():
        logger.info("Building TensorRT engine...")
        expert_trt = output_dir / "expert_stack.trt"
        try:
            build_engine(expert_onnx, expert_trt, hardware)
            result["files"]["expert_trt"] = str(expert_trt)
        except RuntimeError as e:
            logger.warning("TRT build failed: %s", e)
    else:
        logger.info("trtexec not available, skipping TRT build (run on Jetson/Linux with TensorRT)")

    # 7. Save config
    export_config = {
        "model_id": config.model_id,
        "target": config.target,
        "precision": config.precision,
        "opset": config.opset,
        "num_denoising_steps": config.num_denoising_steps,
        "action_chunk_size": config.action_chunk_size,
        "action_dim": action_dim,
        "hardware": {
            "name": hardware.name,
            "memory_gb": hardware.memory_gb,
            "fp8": hardware.fp8_support,
            "precision": hardware.trt_precision,
        },
        "expert": expert_meta,
        "vlm_kv_input": True,
        "vlm_kv_dim": vlm_kv_dim,
    }
    config_path = output_dir / "reflex_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)

    logger.info("Export complete: %s", output_dir)
    return result
