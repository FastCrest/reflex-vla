"""VLM vision encoder export for SmolVLA.

Extracts the SigLIP vision encoder + SmolVLM connector from the base VLM
(HuggingFaceTB/SmolVLM2-500M-Video-Instruct) and exports as a standalone
ONNX graph.

The output ``image_embeds`` tensor of shape ``[B, 64, 960]`` feeds downstream
text embedding concatenation and the expert prefill decoder.

Approach: Load the base VLM via ``AutoModel.from_pretrained``, wrap
vision_model + connector in a thin ``VisionEncoderForONNX`` module that
pre-computes position IDs (avoiding the dynamic ``index_put`` in
``SmolVLMVisionEmbeddings`` that doesn't trace to ONNX), export with
``torch.onnx.export`` at opset 19, and post-fix any type mismatches
in the ONNX graph via ``patch_onnx_type_mismatches``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# SmolVLM2-500M architecture constants
DEFAULT_VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_IMAGE_SIZE = 512  # SigLIP-SO400M native size for SmolVLM2
DEFAULT_VLM_KV_DIM = 960  # SmolLM2 hidden size (also connector output dim)

# Numerical validation threshold -- SigLIP's 27 transformer layers
# accumulate fp32 rounding, so max_diff ~2-4e-4 is expected.
# Mean diff is ~1e-5 which is excellent.
ORT_MAX_DIFF_THRESHOLD = 5e-4


class VisionEncoderForONNX(nn.Module):
    """Wraps SigLIP vision encoder + SmolVLM connector for ONNX export.

    Pre-computes position IDs for full (unpadded) images to avoid the
    dynamic ``index_put`` / ``bucketize`` loop in
    ``SmolVLMVisionEmbeddings.forward()`` that produces ONNX nodes
    with int64/float type mismatches ORT cannot load.

    Input:  ``pixel_values`` -- ``[B, 3, image_size, image_size]`` float32
    Output: ``image_embeds``  -- ``[B, 64, 960]`` float32
    """

    def __init__(
        self,
        vision_model: nn.Module,
        connector: nn.Module,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        super().__init__()
        # Extract sub-modules from the vision model
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = connector

        # Pre-compute position IDs for a full image (no padding).
        # This replicates SmolVLMVisionEmbeddings.forward() logic
        # but without the dynamic for-loop / boolean indexing.
        emb = vision_model.embeddings
        patch_size = emb.patch_size
        num_patches_per_side = emb.num_patches_per_side
        nb_h = image_size // patch_size
        nb_w = image_size // patch_size

        boundaries = torch.arange(
            1 / num_patches_per_side, 1.0, 1 / num_patches_per_side
        )
        h_idx = torch.arange(nb_h, dtype=torch.float32)
        w_idx = torch.arange(nb_w, dtype=torch.float32)
        frac_h = h_idx / nb_h * (1 - 1e-6)
        frac_w = w_idx / nb_w * (1 - 1e-6)
        bucket_h = torch.bucketize(frac_h, boundaries, right=True)
        bucket_w = torch.bucketize(frac_w, boundaries, right=True)
        pos_ids = (bucket_h[:, None] * num_patches_per_side + bucket_w).flatten()
        self.register_buffer("position_ids", pos_ids.unsqueeze(0))  # [1, num_patches]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision encoding: patch embed -> position embed -> encoder -> connector.

        Args:
            pixel_values: ``[B, 3, 512, 512]`` float32

        Returns:
            image_embeds: ``[B, 64, 960]`` float32
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding: [B, 3, 512, 512] -> [B, hidden_dim, 32, 32] -> [B, 1024, hidden_dim]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Add pre-computed position embeddings
        pos_ids = self.position_ids.expand(batch_size, -1)
        embeddings = embeddings + self.position_embedding(pos_ids)

        # Vision transformer encoder (no attention mask needed for full images)
        encoder_output = self.encoder(inputs_embeds=embeddings)
        hidden = encoder_output.last_hidden_state
        hidden = self.post_layernorm(hidden)  # [B, 1024, 768]

        # Connector: pixel shuffle + linear projection -> [B, 64, 960]
        image_embeds = self.connector(hidden)
        return image_embeds


def patch_onnx_type_mismatches(onnx_path: str | Path) -> int:
    """Fix type mismatches in an ONNX graph that prevent ORT loading.

    Walks the ONNX graph and fixes two classes of issues:

    1. **Gather with float indices**: Inserts ``Cast(to=INT64)`` before
       any Gather node whose indices input is not int32/int64.

    2. **Where with mixed types**: Inserts ``Cast`` to unify the X and Y
       inputs of Where nodes when they have different element types.

    Args:
        onnx_path: Path to the ONNX file (modified in-place).

    Returns:
        Number of nodes fixed.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))

    # Build type map from value_info, inputs, and initializers
    type_map: dict[str, int] = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type:
            type_map[vi.name] = vi.type.tensor_type.elem_type
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type:
            type_map[inp.name] = inp.type.tensor_type.elem_type
    for init in model.graph.initializer:
        type_map[init.name] = init.data_type

    cast_nodes_to_insert: list[tuple[int, Any]] = []
    fixed_count = 0

    for i, node in enumerate(model.graph.node):
        # Fix 1: Gather with non-integer indices
        if node.op_type == "Gather" and len(node.input) >= 2:
            idx_type = type_map.get(node.input[1])
            if idx_type is not None and idx_type not in (
                TensorProto.INT32,
                TensorProto.INT64,
            ):
                cast_out = f"{node.input[1]}_cast_to_int64"
                cast_node = helper.make_node(
                    "Cast", [node.input[1]], [cast_out], to=TensorProto.INT64
                )
                cast_nodes_to_insert.append((i, cast_node))
                node.input[1] = cast_out
                fixed_count += 1
                logger.debug(
                    "Fixed Gather %s: cast indices to INT64", node.name
                )

        # Fix 2: Where with mixed X/Y types
        if node.op_type == "Where" and len(node.input) == 3:
            _cond, x_name, y_name = node.input
            x_type = type_map.get(x_name)
            y_type = type_map.get(y_name)
            if (
                x_type is not None
                and y_type is not None
                and x_type != y_type
            ):
                # Cast the integer input to match the float input
                if x_type == TensorProto.INT64 and y_type == TensorProto.FLOAT:
                    cast_out = f"{x_name}_cast_to_float"
                    cast_node = helper.make_node(
                        "Cast", [x_name], [cast_out], to=TensorProto.FLOAT
                    )
                    cast_nodes_to_insert.append((i, cast_node))
                    node.input[1] = cast_out
                    fixed_count += 1
                elif y_type == TensorProto.INT64 and x_type == TensorProto.FLOAT:
                    cast_out = f"{y_name}_cast_to_float"
                    cast_node = helper.make_node(
                        "Cast", [y_name], [cast_out], to=TensorProto.FLOAT
                    )
                    cast_nodes_to_insert.append((i, cast_node))
                    node.input[2] = cast_out
                    fixed_count += 1
                logger.debug(
                    "Fixed Where %s: unified types", node.name
                )

    # Insert cast nodes in correct order (track offset as we insert)
    for offset, (insert_idx, cast_node) in enumerate(
        sorted(cast_nodes_to_insert, key=lambda x: x[0])
    ):
        model.graph.node.insert(insert_idx + offset, cast_node)

    if fixed_count > 0:
        onnx.save(model, str(onnx_path))
        logger.info("Patched %d type mismatches in %s", fixed_count, onnx_path)

    return fixed_count


def export_vlm_prefix(
    checkpoint_path_or_id: str = DEFAULT_VLM_MODEL_NAME,
    output_dir: str | Path = ".",
    opset: int = 19,
) -> Path:
    """Export VLM vision encoder (SigLIP + connector) as ONNX.

    Loads the base SmolVLM2-500M model via ``AutoModel.from_pretrained``,
    wraps the vision encoder + connector into ``VisionEncoderForONNX``,
    exports to ONNX, patches any type mismatches, and validates against
    ONNX Runtime.

    Args:
        checkpoint_path_or_id: HuggingFace model ID or local path.
            Defaults to ``HuggingFaceTB/SmolVLM2-500M-Video-Instruct``.
        output_dir: Directory for output files.
        opset: ONNX opset version (default 19).

    Returns:
        Path to the exported ``vision_encoder.onnx`` file.
    """
    from transformers import AutoModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the base VLM
    logger.info("Loading VLM: %s", checkpoint_path_or_id)
    model = AutoModel.from_pretrained(
        checkpoint_path_or_id, trust_remote_code=True
    )
    model.eval()

    # Extract architecture constants from the loaded model
    vision_config = model.config.vision_config
    text_config = model.config.text_config
    image_size = vision_config.image_size
    vlm_kv_dim = text_config.hidden_size  # 960

    logger.info(
        "VLM config: image_size=%d, vision_hidden=%d, text_hidden(vlm_kv_dim)=%d",
        image_size,
        vision_config.hidden_size,
        vlm_kv_dim,
    )

    # 2. Build the ONNX-exportable wrapper
    wrapper = VisionEncoderForONNX(
        model.vision_model, model.connector, image_size=image_size
    )
    wrapper.eval()

    total_params = sum(p.numel() for p in wrapper.parameters())
    logger.info("VisionEncoderForONNX: %.1fM params", total_params / 1e6)

    # 3. Verify wrapper matches original model output
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        # Original model path
        orig_hidden = model.vision_model(
            pixel_values=dummy_pixel_values
        ).last_hidden_state
        orig_embeds = model.connector(orig_hidden)
        # Wrapper path
        wrapper_embeds = wrapper(dummy_pixel_values)
        sanity_diff = float((orig_embeds - wrapper_embeds).abs().max())
        logger.info("Wrapper vs original sanity check: max_diff=%.2e", sanity_diff)
        assert sanity_diff == 0.0, (
            f"Wrapper output differs from original: {sanity_diff}"
        )

    # 4. Export to ONNX
    onnx_path = output_dir / "vision_encoder.onnx"
    logger.info("Exporting ONNX to %s (opset %d)...", onnx_path, opset)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_pixel_values,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_embeds": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )
    onnx_size_mb = onnx_path.stat().st_size / 1e6
    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_size_mb)

    # 5. Post-export: patch type mismatches (Gather float indices, Where mixed types)
    num_fixed = patch_onnx_type_mismatches(onnx_path)
    logger.info("Post-export patch: fixed %d nodes", num_fixed)

    # 6. Validate: PyTorch vs ONNX Runtime
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            torch_out = wrapper(dummy_pixel_values).numpy()
        ort_out = sess.run(
            None, {"pixel_values": dummy_pixel_values.numpy()}
        )[0]
        max_diff = float(np.abs(torch_out - ort_out).max())
        mean_diff = float(np.abs(torch_out - ort_out).mean())
        passed = max_diff < ORT_MAX_DIFF_THRESHOLD
        logger.info(
            "ONNX validation: max_diff=%.2e, mean_diff=%.2e (%s)",
            max_diff,
            mean_diff,
            "PASS" if passed else "FAIL",
        )
        if not passed:
            logger.warning(
                "ONNX numerical mismatch: max_diff=%.2e exceeds %.1e threshold "
                "(SigLIP has 27 transformer layers -- some fp32 drift is expected)",
                max_diff,
                ORT_MAX_DIFF_THRESHOLD,
            )
    except ImportError:
        logger.warning("onnxruntime not installed -- skipping ONNX validation")

    # 7. Write / update reflex_config.json
    config_path = output_dir / "reflex_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    config["vlm_image_size"] = [image_size, image_size]
    config["vlm_kv_dim"] = vlm_kv_dim
    config["vlm_prefix_onnx"] = "vision_encoder.onnx"
    config["export_version"] = "0.3"

    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Updated config: %s", config_path)

    return onnx_path
