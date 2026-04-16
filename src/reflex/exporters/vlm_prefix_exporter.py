"""VLM prefix encoder export for SmolVLA.

Extracts the VLM backbone (SigLIP vision encoder + SmolLM2 language model)
from a SmolVLA checkpoint and exports it as a standalone ONNX graph.

The output ``prefix_kv`` tensor feeds the expert's cross-attention layers.

Approach chosen: **C (stub)**
    Reconstructing the full SigLIP + SmolLM2 from raw state_dict tensors is
    prohibitively complex — HuggingFace VLMs have deeply nested configs,
    multi-modal merging layers, and vision-language connectors that don't
    map 1:1 to simple nn.Module reconstruction. Approach A (AutoModel) would
    work but requires downloading ~350 MB at export time and tight coupling
    to a specific ``transformers`` version.

    This v1 stub generates a *learned linear projection* of the correct shape,
    proving the ONNX I/O contract (image + instruction_ids → prefix_kv) and
    unblocking the server wiring (Issue 3). Numerical correctness of the VLM
    itself is deferred to v2 when we wire the real HuggingFace forward pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from reflex.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

# SmolVLA architecture constants
DEFAULT_VLM_KV_DIM = 512
DEFAULT_IMAGE_SIZE = 384  # SigLIP-SO400M patch size used by SmolVLM2-500M
DEFAULT_INSTRUCTION_SEQ_LEN = 32
DEFAULT_PREFIX_SEQ_LEN = 50  # output sequence length for prefix_kv


class VLMPrefixEncoder(nn.Module):
    """Wraps the VLM backbone (SigLIP + SmolLM2) for prefix generation.

    v1 (stub): A small learned network that maps image + instruction tokens
    to a ``prefix_kv`` tensor of shape ``[batch, prefix_seq, vlm_kv_dim]``.

    The stub preserves the exact I/O contract so downstream consumers
    (expert cross-attention, ONNX runtime, server predict loop) work
    identically once the real VLM forward pass is wired in v2.

    TODO: Wire real SigLIP → SmolLM2 forward pass (v2).
    """

    def __init__(
        self,
        vlm_state_dict: dict[str, torch.Tensor] | None = None,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        vlm_kv_dim: int = DEFAULT_VLM_KV_DIM,
        prefix_seq_len: int = DEFAULT_PREFIX_SEQ_LEN,
        instruction_seq_len: int = DEFAULT_INSTRUCTION_SEQ_LEN,
    ):
        super().__init__()
        self.image_size = image_size
        self.vlm_kv_dim = vlm_kv_dim
        self.prefix_seq_len = prefix_seq_len
        self.instruction_seq_len = instruction_seq_len

        # --- Stub layers (replace with real VLM in v2) ---
        # Image path: flatten a small spatial pool → project
        self.image_pool = nn.AdaptiveAvgPool2d((4, 4))  # → [B, 3, 4, 4]
        self.image_proj = nn.Linear(3 * 4 * 4, vlm_kv_dim)

        # Instruction path: tiny embedding → mean pool → project
        self.tok_embed = nn.Embedding(50257, 64)  # small vocab embed
        self.tok_proj = nn.Linear(64, vlm_kv_dim)

        # Fuse image + text → prefix_kv sequence
        self.fuse_proj = nn.Linear(vlm_kv_dim, prefix_seq_len * vlm_kv_dim)

    def forward(
        self,
        image: torch.Tensor,
        instruction_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate VLM prefix for expert cross-attention.

        Args:
            image: ``[batch, image_size, image_size, 3]`` float32 (HWC)
            instruction_ids: ``[batch, seq]`` int64 token IDs

        Returns:
            prefix_kv: ``[batch, prefix_seq_len, vlm_kv_dim]`` float32
        """
        b = image.shape[0]

        # Image branch: HWC → CHW → pool → flatten → project
        img = image.permute(0, 3, 1, 2)  # [B, 3, H, W]
        img = self.image_pool(img)  # [B, 3, 4, 4]
        img_flat = img.reshape(b, -1)  # [B, 48]
        img_emb = self.image_proj(img_flat)  # [B, vlm_kv_dim]

        # Text branch: embed → mean pool → project
        tok = self.tok_embed(instruction_ids)  # [B, seq, 64]
        tok_mean = tok.mean(dim=1)  # [B, 64]
        tok_emb = self.tok_proj(tok_mean)  # [B, vlm_kv_dim]

        # Fuse and reshape to prefix sequence
        fused = img_emb + tok_emb  # [B, vlm_kv_dim]
        prefix = self.fuse_proj(fused)  # [B, prefix_seq_len * vlm_kv_dim]
        prefix_kv = prefix.reshape(b, self.prefix_seq_len, self.vlm_kv_dim)

        return prefix_kv


def _detect_vlm_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Detect VLM architecture parameters from checkpoint keys.

    Looks at cross-attention key projection shapes to infer vlm_kv_dim,
    and counts VLM-prefixed tensors to confirm VLM presence.
    """
    vlm_keys = [k for k in state_dict if "vlm_with_expert.vlm." in k or k.startswith("model.vlm_with_expert.vlm.")]
    expert_keys = [k for k in state_dict if "lm_expert" in k]

    vlm_kv_dim = DEFAULT_VLM_KV_DIM
    # Infer vlm_kv_dim from any cross-attention k_proj whose input != expert hidden
    for k in expert_keys:
        if "self_attn.k_proj.weight" in k:
            shape = state_dict[k].shape
            # Cross-attn layers have kv_in == vlm_kv_dim (≠ expert_hidden)
            expert_hidden_candidates = [
                state_dict[ek].shape[0]
                for ek in state_dict
                if ek.endswith("action_in_proj.weight")
            ]
            if expert_hidden_candidates:
                expert_hidden = expert_hidden_candidates[0]
                if shape[1] != expert_hidden:
                    vlm_kv_dim = shape[1]
                    break

    return {
        "vlm_kv_dim": vlm_kv_dim,
        "vlm_key_count": len(vlm_keys),
        "expert_key_count": len(expert_keys),
    }


def export_vlm_prefix(
    checkpoint_path_or_id: str,
    output_dir: str | Path,
    opset: int = 19,
) -> Path:
    """Export VLM prefix encoder as ONNX.

    Loads a SmolVLA checkpoint, extracts the VLM-side weights, builds a
    ``VLMPrefixEncoder``, exports to ONNX, and validates numerically.

    Args:
        checkpoint_path_or_id: Local path or HuggingFace Hub model ID.
        output_dir: Directory for output files.
        opset: ONNX opset version (default 19).

    Returns:
        Path to the exported ``vlm_prefix.onnx`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load checkpoint
    logger.info("Loading checkpoint: %s", checkpoint_path_or_id)
    state_dict, model_config = load_checkpoint(checkpoint_path_or_id)
    total_params = sum(v.numel() for v in state_dict.values())
    logger.info("Loaded %d tensors, %.1fM params total", len(state_dict), total_params / 1e6)

    # 2. Detect VLM config from checkpoint structure
    vlm_info = _detect_vlm_config(state_dict)
    vlm_kv_dim = vlm_info["vlm_kv_dim"]
    logger.info(
        "VLM config: vlm_kv_dim=%d, vlm_keys=%d, expert_keys=%d",
        vlm_kv_dim, vlm_info["vlm_key_count"], vlm_info["expert_key_count"],
    )

    # 3. Determine image size from model config or default
    image_size = DEFAULT_IMAGE_SIZE
    if isinstance(model_config, dict):
        # SmolVLM2 stores image size in vision_config
        vision_cfg = model_config.get("vision_config", {})
        if "image_size" in vision_cfg:
            image_size = vision_cfg["image_size"]
    logger.info("Using image_size=%d", image_size)

    # 4. Build VLMPrefixEncoder (v1 stub — does not use real VLM weights)
    encoder = VLMPrefixEncoder(
        vlm_state_dict=None,  # stub ignores weights for now
        image_size=image_size,
        vlm_kv_dim=vlm_kv_dim,
    )
    encoder.eval()
    logger.info(
        "Built VLMPrefixEncoder (stub): %.3fM params",
        sum(p.numel() for p in encoder.parameters()) / 1e6,
    )

    # 5. Create dummy inputs
    dummy_image = torch.randn(1, image_size, image_size, 3)
    dummy_ids = torch.randint(0, 1000, (1, DEFAULT_INSTRUCTION_SEQ_LEN), dtype=torch.int64)

    # 6. Export to ONNX
    onnx_path = output_dir / "vlm_prefix.onnx"
    logger.info("Exporting ONNX to %s (opset %d)...", onnx_path, opset)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_image, dummy_ids),
            str(onnx_path),
            input_names=["image", "instruction_ids"],
            output_names=["prefix_kv"],
            dynamic_axes={
                "image": {0: "batch"},
                "instruction_ids": {0: "batch", 1: "seq"},
                "prefix_kv": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)

    # 7. Validate: PyTorch vs ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            torch_out = encoder(dummy_image, dummy_ids).numpy()
        ort_out = sess.run(None, {
            "image": dummy_image.numpy(),
            "instruction_ids": dummy_ids.numpy(),
        })[0]
        max_diff = float(np.abs(torch_out - ort_out).max())
        passed = max_diff < 1e-4
        logger.info("ONNX validation: max_diff=%.2e (%s)", max_diff, "PASS" if passed else "FAIL")
        if not passed:
            logger.warning(
                "ONNX numerical mismatch: max_diff=%.2e exceeds 1e-4 threshold", max_diff
            )
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX validation")

    # 8. Write / update reflex_config.json
    config_path = output_dir / "reflex_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    config["vlm_image_size"] = [image_size, image_size]
    config["vlm_kv_dim"] = vlm_kv_dim
    config["vlm_prefix_onnx"] = "vlm_prefix.onnx"
    config["vlm_prefix_seq_len"] = DEFAULT_PREFIX_SEQ_LEN
    config["export_version"] = "0.2"

    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Updated config: %s", config_path)

    return onnx_path
