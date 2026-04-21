"""Decomposed pi0.5 ONNX export — vlm_prefix.onnx + expert_denoise.onnx.

Design doc: `reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md`.

## What this module does

Exports pi0.5 (optionally a SnapFlow-distilled student) as TWO ONNX
graphs instead of one:

1. ``vlm_prefix.onnx`` — takes the vision + language inputs, returns a
   flat tuple of 36 ``past_k_i`` / ``past_v_i`` tensors (18 paligemma
   layers × 2) plus ``prefix_pad_masks``. Runs the full VLM forward
   pass once per observation.

2. ``expert_denoise.onnx`` — takes the 36 past_kv tensors +
   ``prefix_pad_masks`` + ``state`` (pi05 has no state input actually —
   see note below) + ``noise``, runs the action-expert denoising Euler
   loop (1 step for distilled students with ``target_time=1``, 10 steps
   for the teacher), returns the action chunk.

Serve layer (see ``reflex.runtime.pi05_decomposed_server``) hashes the
VLM inputs per-call and reuses the last ``past_kv`` output when the
observation hasn't meaningfully changed — the 3–4× deployment speedup
described in the design doc.

## Structural notes

- pi0.5 has no ``state`` input at the Policy level; state is tokenized
  into the language prompt upstream. The expert-denoise wrapper here
  still has no ``state`` input.
- paligemma layer count: 18. kv_heads=1, head_dim=256. Each K or V
  tensor is ``(B, 1, seq_len, 256)``.
- seq_len is dynamic per observation (tokenized-language length +
  vision patches) but bounded by the preprocessor's config.
- The expert stack's attention reads the paligemma past_kv as
  "past tokens" — this is the cross-attention pattern that makes
  caching work.

## Dependencies

Same as ``monolithic`` extra: transformers==5.3.0, lerobot==0.5.1,
onnx-diagnostic>=0.9. ``apply_export_patches`` from ``monolithic.py``
is invoked here to get the shared denoise-step + cache-freeze patches.

## What this module does NOT do

- Does NOT implement the serve-layer cache (that's
  ``reflex.runtime.pi05_decomposed_server``).
- Does NOT implement the perceptual-hash obs-matcher (same server).
- Does NOT handle SmolVLA — that already has a decomposed path in
  ``reflex.runtime.vlm_orchestrator``.
- Does NOT handle pi0 (the non-.5 variant). pi0 decomposed export is
  a separate future goal (pi0 has state_proj, different suffix shape).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# pi0.5 constants — pulled from get_gemma_config("gemma_2b") in
# lerobot/policies/pi05/modeling_pi05.py line 322-330.
PI05_PALIGEMMA_LAYERS: int = 18  # pi05 paligemma has 18 transformer layers
PI05_KV_HEADS: int = 1
PI05_HEAD_DIM: int = 256


def export_pi05_decomposed(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 1,
    target: str = "desktop",
    student_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Export pi0.5 as ``vlm_prefix.onnx`` + ``expert_denoise.onnx``.

    Args:
        model_id: HF repo id for the pi0.5 base/variant (e.g.
            ``"lerobot/pi05_libero_finetuned_v044"``). Ignored if
            ``student_checkpoint`` is provided.
        output_dir: where to write the two ONNX files + reflex_config.json.
        num_steps: denoising steps baked into expert_denoise.onnx.
            1 for distilled students (target_time=1 path); 10 for the
            canonical teacher.
        target: target hardware profile; passed through to reflex_config.
        student_checkpoint: optional path to a SnapFlow-distilled
            checkpoint dir. When set, loads via ``load_snapflow_student``
            and enables the ``target_time_embed_mlp`` path. Must use
            ``num_steps=1`` in this mode.

    Returns dict with paths + byte sizes + sanity metadata.
    """
    _require_decomposed_deps()

    import torch
    import torch.nn as nn
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    from reflex.exporters.monolithic import (
        apply_export_patches,
        _force_eager_attn,
        _fix_onnx_where_dtype_mismatches,
        _apply_pi05_denoise_step_patch,
    )

    apply_export_patches()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy (student path handles target_time_embed_mlp weights;
    # base path is a plain PI05Policy).
    t0 = time.time()
    if student_checkpoint is not None:
        if num_steps != 1:
            raise ValueError(
                f"student_checkpoint requires num_steps=1 (SnapFlow "
                f"distilled students use a single 1-NFE denoise call); "
                f"got num_steps={num_steps}"
            )
        from reflex.distill.snapflow_pi0_model import load_snapflow_student
        logger.info("[decomposed] Loading SnapFlow student from %s", student_checkpoint)
        policy = load_snapflow_student(student_checkpoint)
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        logger.info("[decomposed] Loading %s", model_id)
        # Apply the same torch.compile suppression as monolithic export
        # — LIBERO-finetuned pi0.5 configs set compile_model=True.
        _orig_compile = torch.compile
        torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))
        try:
            policy = PI05Policy.from_pretrained(model_id)
        finally:
            torch.compile = _orig_compile

    policy.eval().to("cpu").to(torch.float32)
    _gc_disable = getattr(policy.model, "gradient_checkpointing_disable", None)
    if callable(_gc_disable):
        _gc_disable()
    _force_eager_attn(policy.model)
    _apply_pi05_denoise_step_patch()  # reuses monolithic's F.pad mask fix
    logger.info("[decomposed] Loaded in %.1fs", time.time() - t0)

    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    # ---- Export 1: vlm_prefix.onnx -----------------------------------
    prefix_wrapper = Pi05PrefixWrapper(policy.model).eval()

    prefix_dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
    )

    # 36 past_k_i / past_v_i outputs + 1 prefix_pad_masks.
    past_kv_names = []
    for layer_idx in range(PI05_PALIGEMMA_LAYERS):
        past_kv_names.append(f"past_k_{layer_idx}")
        past_kv_names.append(f"past_v_{layer_idx}")
    prefix_output_names = past_kv_names + ["prefix_pad_masks"]

    prefix_path = output_dir / "vlm_prefix.onnx"
    logger.info("[decomposed] Exporting prefix → %s", prefix_path)
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep_prefix = torch.export.export(
            prefix_wrapper, tuple(prefix_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[decomposed] prefix torch.export: %.1fs", time.time() - t0)

    t0 = time.time()
    torch.onnx.export(
        ep_prefix, tuple(prefix_dummy.values()), str(prefix_path),
        input_names=list(prefix_dummy.keys()),
        output_names=prefix_output_names,
        opset_version=19,
    )
    logger.info("[decomposed] prefix ONNX conversion: %.1fs", time.time() - t0)

    prefix_fixes = _fix_onnx_where_dtype_mismatches(prefix_path)
    logger.info("[decomposed] prefix Cast fixes: %d", prefix_fixes)

    # ---- Export 2: expert_denoise.onnx --------------------------------
    # Free the prefix wrapper before building the expert — on A100-80GB
    # we OOM'd with both loaded + a second prefix forward for dummy
    # inputs. Generate expert dummies from known static shapes instead.
    import gc
    del prefix_wrapper, ep_prefix
    gc.collect()

    # pi05 prefix_seq_len = 3 vision views × 256 patches + 16 language
    # tokens + small slack = 1024 (confirmed via earlier smoke runs at
    # prefix_len=1018 for pi05_libero_finetuned_v044 dummy inputs). Use
    # 1024 as a safe round number; the ONNX graph is shape-specialized
    # at this fixed length regardless.
    prefix_seq_len = 1024
    past_kv_shape = (B, PI05_KV_HEADS, prefix_seq_len, PI05_HEAD_DIM)
    past_kv_dummies = [
        torch.randn(past_kv_shape, dtype=torch.float32)
        for _ in range(PI05_PALIGEMMA_LAYERS * 2)
    ]
    prefix_pad_masks_dummy = torch.ones(B, prefix_seq_len, dtype=torch.bool)

    expert_wrapper = Pi05ExpertWrapper(policy.model, num_steps).eval()

    expert_dummy = {}
    for idx, t in enumerate(past_kv_dummies):
        expert_dummy[past_kv_names[idx]] = t
    expert_dummy["prefix_pad_masks"] = prefix_pad_masks_dummy
    expert_dummy["noise"] = torch.randn(B, chunk, action_dim, dtype=torch.float32)

    expert_path = output_dir / "expert_denoise.onnx"
    logger.info("[decomposed] Exporting expert (num_steps=%d) → %s", num_steps, expert_path)
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep_expert = torch.export.export(
            expert_wrapper, tuple(expert_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[decomposed] expert torch.export: %.1fs", time.time() - t0)

    t0 = time.time()
    torch.onnx.export(
        ep_expert, tuple(expert_dummy.values()), str(expert_path),
        input_names=list(expert_dummy.keys()),
        output_names=["actions"],
        opset_version=19,
    )
    logger.info("[decomposed] expert ONNX conversion: %.1fs", time.time() - t0)

    expert_fixes = _fix_onnx_where_dtype_mismatches(expert_path)
    logger.info("[decomposed] expert Cast fixes: %d", expert_fixes)

    # ---- Write reflex_config.json + VERIFICATION stub -----------------
    reflex_cfg = {
        "model_id": model_id if student_checkpoint is None else str(student_checkpoint),
        "model_type": "pi05_decomposed_student" if student_checkpoint else "pi05_decomposed",
        "target": target,
        "num_denoising_steps": num_steps,
        "chunk_size": chunk,
        "action_chunk_size": chunk,
        "action_dim": action_dim,
        "opset": 19,
        "export_kind": "decomposed",
        "decomposed": {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "paligemma_layers": PI05_PALIGEMMA_LAYERS,
            "kv_heads": PI05_KV_HEADS,
            "head_dim": PI05_HEAD_DIM,
            "past_kv_tensor_names": past_kv_names,
        },
    }
    (output_dir / "reflex_config.json").write_text(json.dumps(reflex_cfg, indent=2))
    try:
        from reflex.verification_report import write_verification_report
        write_verification_report(output_dir, parity=None)
    except Exception:
        pass

    size_prefix = prefix_path.stat().st_size / 1e6
    size_expert = expert_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data"))
    external_mb = sum(f.stat().st_size for f in data_files) / 1e6

    return {
        "status": "ok",
        "vlm_prefix_onnx": str(prefix_path),
        "expert_denoise_onnx": str(expert_path),
        "vlm_prefix_mb": size_prefix,
        "expert_denoise_mb": size_expert,
        "external_data_mb": external_mb,
        "total_mb": size_prefix + size_expert + external_mb,
        "num_steps": num_steps,
        "paligemma_layers": PI05_PALIGEMMA_LAYERS,
    }


class Pi05PrefixWrapper:
    """VLM prefix wrapper for pi0.5. Runs the paligemma forward pass
    and returns a flat tuple of (past_k_0, past_v_0, ..., past_k_17,
    past_v_17, prefix_pad_masks).

    Defined as a lazy class built on first instantiation so we don't
    force lerobot import at module load. See ``_build_prefix_class``.
    """
    def __new__(cls, pi05_model):
        impl = _build_prefix_class()
        return impl(pi05_model)


class Pi05ExpertWrapper:
    """Expert denoising wrapper for pi0.5. Takes 36 flat past_kv tensors
    + prefix_pad_masks + noise, runs the Euler loop (num_steps iterations;
    1 for SnapFlow students with target_time=1), returns actions.
    """
    def __new__(cls, pi05_model, num_steps):
        impl = _build_expert_class()
        return impl(pi05_model, num_steps)


_PREFIX_CLASS: Any = None
_EXPERT_CLASS: Any = None


def _build_prefix_class():
    global _PREFIX_CLASS
    if _PREFIX_CLASS is not None:
        return _PREFIX_CLASS

    import torch
    import torch.nn as nn
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    class _Pi05PrefixWrapper(nn.Module):
        def __init__(self, pi05_model):
            super().__init__()
            self.model = pi05_model

        def forward(
            self,
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
        ):
            images = [img_base, img_wrist_l, img_wrist_r]
            img_masks = [mask_base, mask_wrist_l, mask_wrist_r]

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d)
            self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            # Flatten DynamicCache → tuple of tensors. transformers 5.3
            # DynamicCache has a `.layers[i]` list of `DynamicLayer`
            # objects, each exposing `.keys` and `.values` tensors. The
            # onnx-diagnostic torch_export_patches context may strip
            # to_legacy_cache off the class, so we pull the tensors
            # directly from the per-layer objects.
            flat: list = []
            for layer_idx in range(PI05_PALIGEMMA_LAYERS):
                layer = past_key_values.layers[layer_idx]
                flat.append(layer.keys)
                flat.append(layer.values)
            flat.append(prefix_pad_masks)
            return tuple(flat)

    _PREFIX_CLASS = _Pi05PrefixWrapper
    return _PREFIX_CLASS


def _build_expert_class():
    global _EXPERT_CLASS
    if _EXPERT_CLASS is not None:
        return _EXPERT_CLASS

    import torch
    import torch.nn as nn

    # Build the list of past_k_i / past_v_i kwarg names that the
    # wrapper's forward will accept. Needed so torch.export can trace
    # the signature.
    kv_param_names = []
    for layer_idx in range(PI05_PALIGEMMA_LAYERS):
        kv_param_names.append(f"past_k_{layer_idx}")
        kv_param_names.append(f"past_v_{layer_idx}")

    class _Pi05ExpertWrapper(nn.Module):
        """Runs the Euler denoise loop using a past_kv reconstructed
        from flat tensor inputs. ``num_steps`` is baked in at init.
        """

        def __init__(self, pi05_model, num_steps):
            super().__init__()
            self.model = pi05_model
            self.n_steps = num_steps
            # SnapFlow student path: if the model has target_time_embed_mlp,
            # we pass target_time=1 to denoise_step. Otherwise plain teacher.
            self._is_snapflow = hasattr(pi05_model, "target_time_embed_mlp")

        def forward(self, *args):
            # args layout: 36 past_kv tensors (k_0,v_0,k_1,v_1,...,
            # k_17,v_17) + prefix_pad_masks + noise.
            past_flat = args[:PI05_PALIGEMMA_LAYERS * 2]
            prefix_pad_masks = args[PI05_PALIGEMMA_LAYERS * 2]
            noise = args[PI05_PALIGEMMA_LAYERS * 2 + 1]

            # Reconstruct a proper DynamicCache by populating per-layer
            # via .update() — transformers 5.3 removed from_legacy_cache
            # as a classmethod on DynamicCache. update() appends to an
            # empty cache layer so the first call with a given layer_idx
            # initializes that layer's K/V. pi_gemma forward needs a
            # real DynamicCache (isinstance check) so we can't pass a
            # shim.
            from transformers.cache_utils import DynamicCache
            past_kv = DynamicCache()
            for i in range(PI05_PALIGEMMA_LAYERS):
                past_kv.update(
                    key_states=past_flat[2 * i],
                    value_states=past_flat[2 * i + 1],
                    layer_idx=i,
                    cache_kwargs=None,
                )

            action_dtype = self.model.action_in_proj.weight.dtype
            if noise.dtype != action_dtype:
                noise = noise.to(action_dtype)

            dt = -1.0 / self.n_steps
            x_t = noise
            for step in range(self.n_steps):
                time_val = 1.0 + step * dt
                time_tensor = torch.full(
                    (x_t.shape[0],), time_val,
                    dtype=torch.float32, device=x_t.device,
                )
                if self._is_snapflow:
                    target_time_tensor = torch.ones_like(time_tensor)
                    v_t = self.model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_kv,
                        x_t=x_t,
                        timestep=time_tensor,
                        target_time=target_time_tensor,
                    )
                else:
                    v_t = self.model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_kv,
                        x_t=x_t,
                        timestep=time_tensor,
                    )
                x_t = x_t + dt * v_t

            return x_t.to(noise.dtype)

    _EXPERT_CLASS = _Pi05ExpertWrapper
    return _EXPERT_CLASS


class _FlatCache:
    """Minimal DynamicCache-shaped shim that wraps a flat tuple of
    (K_0, V_0, K_1, V_1, ..., K_{N-1}, V_{N-1}) tensors. Exposes
    ``.key_cache[i]``, ``.value_cache[i]``, and ``get_seq_length()``
    — the only attributes pi_gemma's forward path reads from past_kv
    under our denoise_step patch.
    """
    def __init__(self, flat: tuple, num_layers: int):
        self.key_cache = [flat[2 * i] for i in range(num_layers)]
        self.value_cache = [flat[2 * i + 1] for i in range(num_layers)]
        self.is_initialized = True

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.key_cache:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])

    def __len__(self) -> int:
        return len(self.key_cache)


def _require_decomposed_deps() -> None:
    """Decomposed export shares the monolithic ``[monolithic]`` extra."""
    from reflex.exporters.monolithic import _require_monolithic_deps
    _require_monolithic_deps()


__all__ = [
    "PI05_PALIGEMMA_LAYERS",
    "PI05_KV_HEADS",
    "PI05_HEAD_DIM",
    "export_pi05_decomposed",
]
