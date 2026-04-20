"""SnapFlowPI0Pytorch — target_time-aware PI0Pytorch for SnapFlow distillation.

Replaces three v0.3 monkey-patches with proper method overrides:

  1. ``embed_suffix(state, x_t, t, target_time=None)`` — when ``target_time``
     is given, a zero-init learnable MLP maps ``sinusoidal(target_time)``
     to a correction vector added to the time channel before the
     action_time MLP. When ``target_time`` is None, behaves exactly like
     the parent (teacher inference path is unaffected).
  2. ``denoise_step`` does NOT ``copy.deepcopy(past_key_values)`` — the
     downstream ``paligemma_with_expert.forward`` is called with
     ``use_cache=False`` so past_kv is read-only; deepcopy is strictly
     defensive overhead (and breaks on graph-attached tensors during
     training — pytorch#103001).
  3. ``denoise_step`` does NOT force-cast ``suffix_out`` to fp32 — uses
     ``self.action_out_proj.weight.dtype`` instead, so bf16 training
     stays bf16 end-to-end.

## How to activate

    from reflex.distill.snapflow_pi0_model import enable_snapflow
    enable_snapflow(student_policy.model)   # mutates in place

After the call:
  - ``student.model.__class__`` is the dynamically-built SnapFlowPI0Pytorch
    subclass of the installed lerobot ``PI0Pytorch``.
  - ``student.model.target_time_embed_mlp`` is registered as a submodule
    (zero-init), so ``student.parameters()`` includes its weights for
    the optimizer to update.
  - ``student.model.embed_suffix`` and ``.denoise_step`` accept the new
    ``target_time`` kwarg with default None (= backward compat).

The student's weights are NOT moved or copied; this is a pure behavioral
change on the same instance.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cached subclass — built once per process at first enable_snapflow() call,
# which is when we can safely import lerobot.
_SNAPFLOW_CLASS: Any = None


def enable_snapflow(pi0_model: Any) -> None:
    """Mutate a PI0Pytorch instance in place so it supports target_time.

    See module docstring for the three behavioral changes this enables.

    Args:
      pi0_model: an instance of ``lerobot.policies.pi0.modeling_pi0.PI0Pytorch``
        (typically ``student_policy.model`` after PI0Policy.from_pretrained).

    Raises:
      TypeError: if ``pi0_model`` isn't a PI0Pytorch instance.
    """
    import torch
    import torch.nn as nn
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch

    if not isinstance(pi0_model, PI0Pytorch):
        raise TypeError(
            f"enable_snapflow expects PI0Pytorch; got {type(pi0_model).__name__}"
        )

    dim = pi0_model.action_in_proj.out_features
    hidden_dim = max(dim, 256)

    target_time_mlp = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, dim),
    )
    with torch.no_grad():
        target_time_mlp[-1].weight.zero_()
        target_time_mlp[-1].bias.zero_()

    ref_param = next(pi0_model.parameters())
    target_time_mlp = target_time_mlp.to(
        dtype=ref_param.dtype, device=ref_param.device,
    )

    pi0_model.add_module("target_time_embed_mlp", target_time_mlp)
    pi0_model.__class__ = _resolve_snapflow_class()
    logger.info(
        "[snapflow] enabled target_time on PI0Pytorch (dim=%d, hidden=%d, init=zero)",
        dim, hidden_dim,
    )


def _resolve_snapflow_class() -> type:
    """Build (or return cached) SnapFlowPI0Pytorch subclass of lerobot's
    PI0Pytorch. Lazy so importing this module doesn't force lerobot import.
    """
    global _SNAPFLOW_CLASS
    if _SNAPFLOW_CLASS is not None:
        return _SNAPFLOW_CLASS

    import torch
    from lerobot.policies.pi0.modeling_pi0 import (
        PI0Pytorch,
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
    )

    class SnapFlowPI0Pytorch(PI0Pytorch):
        """PI0Pytorch + target_time embed_suffix + deepcopy-free denoise_step.

        Not constructed directly — produced by enable_snapflow() via
        __class__-swap on a loaded PI0Pytorch instance.
        """

        def embed_suffix(
            self,
            state,
            noisy_actions,
            timestep,
            target_time=None,
        ):
            """Parent embed_suffix + optional zero-init target_time contribution.

            When target_time is None, bit-exact with parent. When given a
            (B,) tensor, adds ``target_time_embed_mlp(sinusoidal(target_time))``
            to the time embedding before it's fused with the action embedding.
            """
            import torch.nn.functional as F

            embs = []
            pad_masks = []
            att_masks = []

            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]

            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=timestep.device,
            )
            time_emb = time_emb.type(dtype=timestep.dtype)

            if target_time is not None:
                tt = target_time
                if tt.ndim == 0:
                    tt = tt.expand(bsize)
                tt_emb = create_sinusoidal_pos_embedding(
                    tt,
                    self.action_in_proj.out_features,
                    min_period=self.config.min_period,
                    max_period=self.config.max_period,
                    device=tt.device,
                )
                tt_emb = tt_emb.type(dtype=time_emb.dtype)
                time_emb = time_emb + self.target_time_embed_mlp(tt_emb)

            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None

            embs.append(action_time_emb)
            bsize2, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(
                bsize2, action_time_dim, dtype=torch.bool, device=timestep.device,
            )
            pad_masks.append(action_time_mask)

            att_masks += [1] + ([0] * (self.config.chunk_size - 1))

            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
            att_masks = att_masks[None, :].expand(bsize2, len(att_masks))

            return embs, pad_masks, att_masks, adarms_cond

        def denoise_step(
            self,
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
            target_time=None,
        ):
            """Parent denoise_step minus the deepcopy and minus the fp32 cast.

            - Skips ``copy.deepcopy(past_key_values)``: safe because the
              downstream forward uses ``use_cache=False`` (read-only pass).
            - Skips the hardcoded ``suffix_out.to(fp32)``: casts to
              ``self.action_out_proj.weight.dtype`` instead, so bf16 training
              stays bf16.
            - Passes ``target_time`` through to ``embed_suffix``.
            """
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, timestep, target_time=target_time,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2,
            )

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        @torch.no_grad()
        def sample_actions_1step(
            self,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=None,
        ):
            """SnapFlow 1-NFE inference: single denoise_step at target_time=1.

            After SnapFlow distillation, the ``target_time_embed_mlp`` has
            learned to produce the 2-step-Euler shortcut velocity at
            target_time=1. This method runs exactly one denoise_step with
            time=1 (pure noise) and target_time=1 (request one-shot
            generation), then does one Euler step (dt=-1) to produce the
            action chunk.

            Analog of PI0Pytorch.sample_actions with num_steps=1 + the
            target_time kwarg threaded through. Parent prefix-cache setup
            is replicated verbatim; only the integration loop shrinks.
            """
            bsize = state.shape[0]
            device = state.device

            if noise is None:
                actions_shape = (
                    bsize, self.config.chunk_size, self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            time = torch.ones(bsize, dtype=torch.float32, device=device)
            # Single denoise call at time=1, target_time=1. Inputs match the
            # dtype of action_in_proj so embed_suffix doesn't hit a cast.
            action_dtype = self.action_in_proj.weight.dtype
            x_t = noise.to(action_dtype)
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                target_time=time,
            )
            # Euler step: x_{k+1} = x_k + dt * v_t, with dt=-1 and x_k=noise.
            # v_t from a well-distilled student approximates noise - action.
            return (x_t - v_t).to(noise.dtype)

    _SNAPFLOW_CLASS = SnapFlowPI0Pytorch
    return SnapFlowPI0Pytorch


def load_snapflow_student(checkpoint_path: Any) -> Any:
    """Load a SnapFlow-distilled student checkpoint from a reflex-saved dir.

    The backend's ``_save_student_checkpoint`` dumps the student via
    ``PI05Policy.save_pretrained(...)``, which includes the
    ``target_time_embed_mlp.*`` keys. Plain ``PI05Policy.from_pretrained``
    may reject those keys (strict=True default).

    This loader:

      1. Reads ``model.safetensors``, splits off the
         ``model.target_time_embed_mlp.*`` keys into a separate dict.
      2. Writes the base-only safetensors (+ copies sibling config files)
         into a temp dir, loads via ``PI05Policy.from_pretrained``
         (or PI0 if that's the teacher family).
      3. Calls ``enable_snapflow(policy.model)`` to attach the target_time
         machinery + an empty target_time_embed_mlp submodule.
      4. Loads the extracted MLP weights into that submodule.

    Callers then typically do ``.eval().to(device).to(dtype)`` as usual,
    and invoke ``policy.model.sample_actions_1step(...)`` for 1-NFE
    inference or ``policy.model.sample_actions(...)`` for multi-step.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SnapFlow student checkpoint not found at {path}. Expected a "
            f"reflex-saved pretrained_model/ dir with model.safetensors + "
            f"config.json."
        )

    sf_path = path / "model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(f"missing model.safetensors in {path}")

    full_state = load_file(str(sf_path))
    mlp_prefix = "model.target_time_embed_mlp."
    base_state = {k: v for k, v in full_state.items() if not k.startswith(mlp_prefix)}
    mlp_state = {
        k[len(mlp_prefix):]: v
        for k, v in full_state.items() if k.startswith(mlp_prefix)
    }

    policy_cls = _dispatch_policy_class(path)

    with tempfile.TemporaryDirectory(prefix="snapflow_load_") as td:
        td_path = Path(td)
        for f in path.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy(f, td_path / f.name)
        save_file(base_state, str(td_path / "model.safetensors"))
        policy = policy_cls.from_pretrained(str(td_path))

    enable_snapflow(policy.model)

    if not mlp_state:
        logger.warning(
            "[load_snapflow_student] no target_time_embed_mlp weights in %s; "
            "loaded as zero-init student (behaves like teacher at target_time=1). "
            "If this is a trained distill output, the checkpoint is corrupt.",
            path,
        )
        return policy

    missing, unexpected = policy.model.target_time_embed_mlp.load_state_dict(
        mlp_state, strict=True,
    )
    if missing:
        logger.warning("[load_snapflow_student] missing mlp keys: %s", missing)
    if unexpected:
        logger.warning("[load_snapflow_student] unexpected mlp keys: %s", unexpected)
    logger.info("[load_snapflow_student] loaded student from %s", path)
    return policy


def _dispatch_policy_class(checkpoint_path: Any):
    """Pick PI0Policy vs PI05Policy from checkpoint config."""
    import json
    from pathlib import Path

    cfg_path = Path(checkpoint_path) / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        ptype = cfg.get("type") or cfg.get("_reflex_distill_teacher_type", "")
        if ptype == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            return PI05Policy
        if ptype == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            return PI0Policy
    # Fallback: try pi0.5 first (current distill target), then pi0.
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy
    except ImportError:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy


__all__ = ["enable_snapflow", "load_snapflow_student"]
