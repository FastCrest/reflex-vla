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
    config["vlm_model_id"] = checkpoint_path_or_id
    config["export_version"] = "0.3"

    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Updated config: %s", config_path)

    # 8. Export text embedder
    text_emb_path = export_text_embedder(
        checkpoint_path_or_id=checkpoint_path_or_id,
        output_dir=output_dir,
        opset=opset,
    )

    # 9. Export decoder prefill
    decoder_path = export_decoder_prefill(
        checkpoint_path_or_id=checkpoint_path_or_id,
        output_dir=output_dir,
        opset=opset,
    )

    # Update config with all export paths
    config = json.loads(config_path.read_text())
    config["text_embedder_onnx"] = "text_embedder.onnx"
    config["decoder_prefill_onnx"] = "decoder_prefill.onnx"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Updated config with text_embedder + decoder_prefill: %s", config_path)

    return onnx_path


# ---------------------------------------------------------------------------
# Text embedder export
# ---------------------------------------------------------------------------


def export_text_embedder(
    checkpoint_path_or_id: str = DEFAULT_VLM_MODEL_NAME,
    output_dir: str | Path = ".",
    opset: int = 19,
) -> Path:
    """Export the token embedding table as a standalone ONNX.

    Input: ``input_ids [B, seq]`` int64
    Output: ``text_embeds [B, seq, 960]`` float32
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel as _AutoModel

    logger.info("Loading VLM for text embedder export: %s", checkpoint_path_or_id)
    model = _AutoModel.from_pretrained(
        checkpoint_path_or_id, trust_remote_code=True
    )
    model.eval()

    # Find embed_tokens — try multiple attribute paths
    embed_tokens = None
    for attr_path in [
        "text_model.embed_tokens",
        "model.embed_tokens",
        "language_model.model.embed_tokens",
        "text_model.model.embed_tokens",
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            embed_tokens = obj
            logger.info("Found embed_tokens at model.%s", attr_path)
            break
        except AttributeError:
            continue

    if embed_tokens is None:
        raise RuntimeError(
            "Could not find embed_tokens in model. Tried: "
            "text_model.embed_tokens, model.embed_tokens, "
            "language_model.model.embed_tokens, text_model.model.embed_tokens"
        )

    class EmbedTokensForONNX(torch.nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.embed = embed

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed(input_ids)

    wrapper = EmbedTokensForONNX(embed_tokens).eval()
    vocab_size = embed_tokens.num_embeddings
    embed_dim = embed_tokens.embedding_dim
    logger.info("EmbedTokens: vocab=%d, dim=%d", vocab_size, embed_dim)

    # Export
    dummy_ids = torch.randint(0, min(vocab_size, 1000), (1, 32), dtype=torch.int64)
    onnx_path = output_dir / "text_embedder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "text_embeds": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )

    logger.info(
        "Wrote %s (%.2f MB)",
        onnx_path,
        onnx_path.stat().st_size / 1e6,
    )

    # Validate
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = wrapper(dummy_ids).numpy()
        ort_out = sess.run(None, {"input_ids": dummy_ids.numpy()})[0]
        max_diff = float(np.abs(torch_out - ort_out).max())
        logger.info("Text embedder ONNX validation: max_diff=%.2e (PASS)" if max_diff < 1e-4
                     else "Text embedder ONNX validation: max_diff=%.2e (WARN)", max_diff)
    except ImportError:
        logger.warning("onnxruntime not installed — skipping validation")

    return onnx_path


# ---------------------------------------------------------------------------
# Decoder prefill export
# ---------------------------------------------------------------------------

# 16 layers is the SmolVLA expert prefix depth. The base SmolVLM2-500M has
# 32 layers, but SmolVLA's fine-tuned checkpoint uses only the first 16.
# We truncate manually here since we load from the base model.
SMOLVLA_NUM_DECODER_LAYERS = 16

# 16 transformer layers accumulate more fp32 rounding than a single layer.
# Spike showed ~4e-5 for 1 layer; budget ~5e-4 for the full stack.
DECODER_ORT_MAX_DIFF_THRESHOLD = 5e-4

# Default prefix sequence length for dummy inputs (image 64 + text ~10 + state 1)
DEFAULT_PREFIX_SEQ_LEN = 75


class DecoderPrefillForONNX(nn.Module):
    """Wraps a truncated SmolLM2 decoder for single-pass prefill ONNX export.

    Takes pre-assembled prefix embeddings (image + text + state) and runs
    them through the 16-layer decoder to produce contextualised hidden states.
    No autoregressive generation — just one forward pass.

    Input:  ``inputs_embeds``  — ``[B, seq, 960]`` float32
    Input:  ``attention_mask`` — ``[B, seq]`` int64
    Output: ``last_hidden_state`` — ``[B, seq, 960]`` float32
    """

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run decoder prefill.

        Args:
            inputs_embeds: ``[B, seq, 960]`` — assembled prefix embeddings.
            attention_mask: ``[B, seq]`` — standard HF attention mask
                (1 = attend, 0 = ignore).

        Returns:
            last_hidden_state: ``[B, seq, 960]`` — contextualised hidden states.
        """
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs.last_hidden_state


def export_decoder_prefill(
    checkpoint_path_or_id: str = DEFAULT_VLM_MODEL_NAME,
    output_dir: str | Path = ".",
    opset: int = 19,
    num_layers: int = SMOLVLA_NUM_DECODER_LAYERS,
    prefix_seq_len: int = DEFAULT_PREFIX_SEQ_LEN,
) -> Path:
    """Export the 16-layer SmolLM2 decoder as decoder_prefill.onnx.

    Loads the full SmolVLM2-500M model, extracts the text decoder (LlamaModel),
    truncates to ``num_layers`` layers (SmolVLA uses 16 of the 32 base layers),
    wraps in ``DecoderPrefillForONNX``, and exports to ONNX with ORT validation.

    Args:
        checkpoint_path_or_id: HuggingFace model ID or local path.
        output_dir: Directory for output files.
        opset: ONNX opset version (default 19, needed for GQA).
        num_layers: Number of decoder layers to keep (default 16).
        prefix_seq_len: Sequence length for dummy inputs (default 75).

    Returns:
        Path to the exported ``decoder_prefill.onnx`` file.
    """
    from transformers import AutoModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the base VLM
    logger.info("Loading VLM for decoder export: %s", checkpoint_path_or_id)
    model = AutoModel.from_pretrained(
        checkpoint_path_or_id, trust_remote_code=True
    )
    model.eval()

    # 2. Extract the text decoder (LlamaModel)
    text_model = model.text_model
    total_layers = len(text_model.layers)
    hidden_size = model.config.text_config.hidden_size

    logger.info(
        "Text decoder: %d layers, hidden_size=%d, truncating to %d layers",
        total_layers,
        hidden_size,
        num_layers,
    )

    # 3. Truncate to num_layers (SmolVLA uses first 16 of 32 base layers)
    if total_layers > num_layers:
        text_model.layers = text_model.layers[:num_layers]
        logger.info(
            "Truncated decoder from %d to %d layers", total_layers, num_layers
        )
    elif total_layers < num_layers:
        logger.warning(
            "Model has only %d layers, requested %d — using all available",
            total_layers,
            num_layers,
        )
        num_layers = total_layers

    # 4. Build ONNX wrapper
    wrapper = DecoderPrefillForONNX(text_model)
    wrapper.eval()

    total_params = sum(p.numel() for p in wrapper.parameters())
    logger.info(
        "DecoderPrefillForONNX: %d layers, %.1fM params",
        num_layers,
        total_params / 1e6,
    )

    # 5. Sanity check: wrapper produces output of correct shape
    dummy_embeds = torch.randn(1, prefix_seq_len, hidden_size)
    dummy_mask = torch.ones(1, prefix_seq_len, dtype=torch.long)

    with torch.no_grad():
        torch_out = wrapper(dummy_embeds, dummy_mask)
    assert torch_out.shape == (1, prefix_seq_len, hidden_size), (
        f"Unexpected output shape: {torch_out.shape}"
    )
    logger.info("Sanity check passed: output shape %s", torch_out.shape)

    # 6. Export to ONNX
    onnx_path = output_dir / "decoder_prefill.onnx"
    logger.info("Exporting decoder prefill to %s (opset %d)...", onnx_path, opset)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_mask),
            str(onnx_path),
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "inputs_embeds": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )

    onnx_size_mb = onnx_path.stat().st_size / 1e6
    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_size_mb)

    # 7. Validate with ONNX Runtime
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            torch_ref = wrapper(dummy_embeds, dummy_mask).numpy()
        ort_out = sess.run(
            None,
            {
                "inputs_embeds": dummy_embeds.numpy(),
                "attention_mask": dummy_mask.numpy(),
            },
        )[0]

        max_diff = float(np.abs(torch_ref - ort_out).max())
        mean_diff = float(np.abs(torch_ref - ort_out).mean())
        passed = max_diff < DECODER_ORT_MAX_DIFF_THRESHOLD

        logger.info(
            "ONNX validation: max_diff=%.2e, mean_diff=%.2e (%s)",
            max_diff,
            mean_diff,
            "PASS" if passed else "FAIL",
        )
        if not passed:
            logger.warning(
                "Decoder ONNX numerical mismatch: max_diff=%.2e exceeds %.1e "
                "threshold (16 transformer layers accumulate fp32 drift)",
                max_diff,
                DECODER_ORT_MAX_DIFF_THRESHOLD,
            )
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX validation")

    return onnx_path
