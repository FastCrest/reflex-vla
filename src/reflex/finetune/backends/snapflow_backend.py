"""SnapFlow backend — in-process 1-step self-distillation trainer.

Implements `Backend.fit()` using the math in `reflex.distill.snapflow`
and the frozen teacher loader in `reflex.distill.teacher_loader`.

## What this backend does

1. Load teacher policy (frozen, eval mode) via teacher_loader.
2. deepcopy() teacher → student. Install the zero-init target_time
   embedding into the student's velocity head (model surgery via
   the policy-specific adapter in `_velocity_adapters`).
3. Build a LeRobotDataset dataloader matching the teacher's expected
   obs_kwargs (image, state, language).
4. Training loop: for each batch, sample t + noise, compute SnapFlow
   loss (flow_matching + consistency), backprop through student only.
5. Save checkpoint every `checkpoint_every` steps via the teacher's
   native `save_pretrained` (same format `_auto_export` consumes).
6. Fire the lifecycle hooks (`on_start`, `on_step`, `on_checkpoint`,
   `on_end`) at the documented points.

## What this backend does NOT do

- No LoRA/adapter logic — SnapFlow trains FULL weights (the student
  is a full-weight copy, no PEFT merge step needed).
- No ONNX export (postprocess.finalize chain owns that).
- No LIBERO eval (libero_drop_gate hook owns that, fired by finalize).
- No VLM prefix caching (current v0.3 recomputes per step for
  simplicity; v0.5 can cache when profiling shows it's the bottleneck).

## The velocity-adapter split

The SnapFlow math is paradigm-agnostic, but actually EXTRACTING the
velocity function from a lerobot PI0Policy / PI05Policy depends on
per-policy internals (expert-stack attribute names, prefix KV cache
construction, etc.). So this module splits:

  - `SnapFlowBackend.fit()` — generic loop (optimizer, dataloader,
    loss step, checkpoint, hooks). Takes velocity_fn callables.
  - `_build_velocity_adapters(teacher, student, policy_type)` —
    policy-specific. For pi0/pi05, wraps the PaliGemma-VLM +
    expert-stack. Lazy-imported per-policy-type so CI can test
    the generic loop without lerobot installed.

v0.3: pi0 + pi05 only. SmolVLA is v0.3.1 pending velocity-convergence
validation; GR00T is v0.5+ (different denoising paradigm — DDPM).
"""
from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from reflex.finetune.backends.base import (
    Backend,
    CheckpointResult,
    TrainerContext,
)

logger = logging.getLogger(__name__)


# Default mix coefficient for the SnapFlow consistency term when the
# FinetuneConfig doesn't override via extra_lerobot_args. Paper uses 1.0.
DEFAULT_CONSISTENCY_ALPHA: float = 1.0

# Save a student checkpoint every N steps (same convention as
# lerobot-train). Override via cfg.extra_lerobot_args["checkpoint_every"].
DEFAULT_CHECKPOINT_EVERY: int = 1_000


class SnapFlowBackend:
    """In-process SnapFlow distillation trainer."""

    def fit(self, ctx: TrainerContext) -> CheckpointResult:
        cfg = ctx.config
        if not cfg.teacher_export:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=(
                    "SnapFlowBackend requires cfg.teacher_export to be set. "
                    "Pass --teacher-export <path-or-hf-id> pointing at a "
                    "reflex-exported dir (merged PyTorch checkpoint)."
                ),
            )

        # Lazy-import torch + SnapFlow math so a CI run that never touches
        # distill doesn't pay the import cost.
        import torch
        from torch.optim import AdamW

        from reflex.distill.snapflow import snapflow_loss_step
        from reflex.distill.teacher_loader import load_teacher

        # ---- 1. Load teacher, build student --------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = cfg.precision if cfg.precision in ("bf16", "fp32") else "bf16"
        logger.info("[snapflow] loading teacher from %s on %s", cfg.teacher_export, device)
        loaded = load_teacher(cfg.teacher_export, device=device, dtype=dtype)
        teacher = loaded.policy
        policy_type = loaded.policy_type

        logger.info("[snapflow] building student (deepcopy of teacher)")
        student = copy.deepcopy(teacher)
        # Student is TRAINABLE — teacher_loader froze it, undo that for the copy.
        for p in student.parameters():
            p.requires_grad = True
        student.train()

        # ---- 2. Install velocity adapters + target_time embedding ----------
        teacher_velocity_fn, student_velocity_fn = _build_velocity_adapters(
            teacher=teacher,
            student=student,
            policy_type=policy_type,
        )

        # ---- 3. Build dataloader + preprocessor ---------------------------
        try:
            loader = _build_dataloader(cfg, policy_type=policy_type)
        except Exception as e:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=f"dataloader construction failed: {type(e).__name__}: {e}",
            )
        try:
            preprocessor = _build_preprocessor(
                teacher_path=loaded.checkpoint_dir,
                image_key_map=cfg.extra_lerobot_args.get("image_key_map"),
                device=device,
            )
        except Exception as e:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=f"preprocessor construction failed: {type(e).__name__}: {e}",
            )

        # ---- 4. Optimizer + checkpoint dir --------------------------------
        opt = AdamW(
            (p for p in student.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )
        checkpoint_root = Path(cfg.output) / "training" / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_every = int(
            cfg.extra_lerobot_args.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY)
        )
        consistency_alpha = float(
            cfg.extra_lerobot_args.get("consistency_alpha", DEFAULT_CONSISTENCY_ALPHA)
        )

        # ---- 5. Fire on_start ---------------------------------------------
        ctx.hooks.run("on_start", ctx, config=cfg, policy_type=policy_type)

        # ---- 6. Training loop ---------------------------------------------
        step = 0
        last_ckpt: Path | None = None
        loss_history: list[dict[str, float]] = []
        log_handle = open(ctx.training_log_path, "a", encoding="utf-8")
        try:
            import torch as _torch
            target_dtype = _torch.bfloat16 if dtype == "bf16" else _torch.float32
            max_action_dim = getattr(teacher.config, "max_action_dim", None)
            chunk_size = getattr(teacher.config, "chunk_size", None)
            for step, batch in enumerate(loader, start=1):
                # Apply lerobot's preprocessor pipeline: rename_map (image keys),
                # tokenizer (task -> language_tokens), device transfer, normalize.
                batch = preprocessor(batch)
                action, noise, t, obs_kwargs = _prepare_batch(
                    batch,
                    device=device,
                    compute_dtype=target_dtype,
                    max_action_dim=max_action_dim,
                    chunk_size=chunk_size,
                )

                opt.zero_grad()
                loss, snap = snapflow_loss_step(
                    student_velocity_fn,
                    teacher_velocity_fn,
                    action=action,
                    noise=noise,
                    t=t,
                    obs_kwargs=obs_kwargs,
                    consistency_alpha=consistency_alpha,
                )
                loss.backward()
                opt.step()

                loss_history.append(asdict(snap) | {"step": step})
                log_handle.write(
                    json.dumps({"step": step, **asdict(snap), "lr": cfg.learning_rate}) + "\n"
                )
                log_handle.flush()

                ctx.hooks.run(
                    "on_step",
                    ctx,
                    step=step,
                    loss=snap.total,
                    lr=cfg.learning_rate,
                    flow_matching=snap.flow_matching,
                    consistency=snap.consistency,
                )

                if step % checkpoint_every == 0 or step == cfg.num_steps:
                    last_ckpt = _save_student_checkpoint(
                        student, checkpoint_root, step, teacher_config=loaded.config,
                    )
                    ctx.hooks.run(
                        "on_checkpoint", ctx, step=step, ckpt_path=last_ckpt,
                    )

                if step >= cfg.num_steps:
                    break
        finally:
            log_handle.close()

        # ---- 7. Ensure we have a final checkpoint -------------------------
        if last_ckpt is None:
            last_ckpt = _save_student_checkpoint(
                student, checkpoint_root, step or 0, teacher_config=loaded.config,
            )

        # ---- 8. Provenance stamp ------------------------------------------
        _write_provenance(
            last_ckpt,
            teacher_dir=loaded.checkpoint_dir,
            policy_type=policy_type,
            steps=step,
            consistency_alpha=consistency_alpha,
        )

        # ---- 9. Fire on_end + return --------------------------------------
        ctx.hooks.run(
            "on_end", ctx, status="ok", steps_completed=step,
        )
        return CheckpointResult(
            final_checkpoint_path=last_ckpt,
            training_steps_completed=step,
            status="ok",
            intermediate_metrics={
                "policy_type": policy_type,
                "final_loss": loss_history[-1]["total"] if loss_history else None,
                "final_fm_loss": loss_history[-1]["flow_matching"] if loss_history else None,
                "final_consistency_loss": loss_history[-1]["consistency"] if loss_history else None,
                "consistency_alpha": consistency_alpha,
                "teacher_export": str(loaded.checkpoint_dir),
            },
        )


# ---------------------------------------------------------------------------
# Policy-specific velocity adapters
# ---------------------------------------------------------------------------

def _build_velocity_adapters(
    teacher: Any,
    student: Any,
    policy_type: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return `(teacher_velocity_fn, student_velocity_fn)` for policy_type.

    Each returned callable has signature:
        teacher_velocity_fn(x, t, **obs_kwargs) -> velocity tensor
        student_velocity_fn(x, t, target_time=None, **obs_kwargs) -> velocity tensor

    The student's `target_time` routes through the zero-init embedding
    installed by `_install_target_time_embedding`.

    Raises NotImplementedError for policy_types we haven't wired yet
    (SmolVLA v0.3.1; GR00T v0.5+).
    """
    if policy_type in ("pi0", "pi05"):
        return _build_pi_family_adapters(teacher, student)
    raise NotImplementedError(
        f"No velocity adapter for policy_type={policy_type!r}. "
        f"pi0/pi05 are supported in v0.3; SmolVLA in v0.3.1; "
        f"GR00T in v0.5+. See "
        f"reflex_context/01_architecture/distill_SYNTHESIS.md."
    )


def _build_pi_family_adapters(
    teacher: Any,
    student: Any,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Velocity adapters for pi0 / pi05.

    Both pi0 and pi05 wrap a PaliGemma VLM + a Gemma action expert.
    Velocity extraction mirrors lerobot's internal sampling loop
    (modeling_pi0.py:854-886):

      1. `policy._preprocess_images(batch)` → `(images, img_masks)`
      2. `policy.prepare_state(batch)` → state
      3. `policy.model.embed_prefix(...)` → prefix_embs + masks
      4. `policy.model.paligemma_with_expert.forward(use_cache=True)` →
         `past_key_values` (VLM prefix cache)
      5. Per-step: `policy.model.denoise_step(state, prefix_pad_masks,
         past_key_values, x_t, timestep)` → velocity

    We recompute the VLM prefix per velocity call for simplicity. The
    Euler-loop in lerobot caches the prefix once per action query, but
    during training each step uses a fresh batch so there's nothing to
    cache across steps anyway.

    ## target_time is NOT yet wired

    SnapFlow's exact paper trick — a zero-init `target_time` embedding
    added to the time embedding inside `embed_suffix` — requires
    subclassing `PI0Pytorch` to override `embed_suffix` with a
    target_time-aware version. v0.3 ships with a SIMPLER variant:

      - Flow-matching loss: student.denoise_step at random t vs true velocity (WORKS)
      - Consistency loss: student.denoise_step at t vs teacher's 2-step
        Euler shortcut from x_t at t (WORKS)

    The student is a full-weight copy of the teacher, trained to match
    the teacher's shortcut velocity at all timesteps. This trains the
    student to do 2-3 step sampling, not exact 1-step. Full 1-step via
    the target_time embedding is v0.3.1 (tracked in
    reflex_context/01_architecture/distill_SYNTHESIS.md §"Deferred").
    """
    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    def _build_prefix_cache(policy, obs_kwargs):
        """Compute (past_kv, prefix_pad_masks, state) from a lerobot batch.

        obs_kwargs is the batch dict minus 'action' — the flattened
        observation/language/state that LeRobotDataset produces AFTER
        lerobot's preprocessor pipeline has run (so language is already
        tokenized into OBS_LANGUAGE_TOKENS + OBS_LANGUAGE_ATTENTION_MASK).
        """
        m = policy.model
        images, img_masks = policy._preprocess_images(obs_kwargs)
        lang_tokens = obs_kwargs[OBS_LANGUAGE_TOKENS]
        lang_masks = obs_kwargs[OBS_LANGUAGE_ATTENTION_MASK]
        state = policy.prepare_state(obs_kwargs)
        # Defensive re-pad: prepare_state SHOULD pad state to max_state_dim,
        # but observed in smoke runs that some preprocessor configs leave
        # the state untouched. Force-pad here so state_proj always sees
        # the expected max_state_dim.
        max_state_dim = getattr(policy.config, "max_state_dim", None)
        if max_state_dim is not None and state.shape[-1] < max_state_dim:
            state = F.pad(state, (0, max_state_dim - state.shape[-1]))
        # lerobot auto-casts state to fp32 if state_proj weights are fp32
        # (modeling_pi0.py:693) but doesn't handle the bf16-weights + fp32-state
        # direction. Cast state to match state_proj weight dtype here.
        state_dtype = m.state_proj.weight.dtype
        if state.dtype != state_dtype:
            state = state.to(state_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        # Mirror lerobot's sample_actions trick: force eager attention
        # before the prefix forward so bf16 models don't hit the SDPA
        # dtype-mismatch-on-bias error (modeling_pi0.py:844).
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        _, past_kv = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_kv, prefix_pad_masks, state

    def _run_denoise_step(policy, x, t, obs_kwargs):
        past_kv, prefix_pad_masks, state = _build_prefix_cache(policy, obs_kwargs)
        # Action (x_t) should already be padded to max_action_dim in
        # _prepare_batch so the whole flow-matching chain stays at the
        # model's expected dim. Cast to action_in_proj dtype for safety.
        action_dtype = policy.model.action_in_proj.weight.dtype
        if x.dtype != action_dtype:
            x = x.to(action_dtype)
        return policy.model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_kv,
            x_t=x,
            timestep=t,
        )

    def teacher_velocity_fn(x, t, **obs_kwargs):
        with torch.no_grad():
            return _run_denoise_step(teacher, x, t, obs_kwargs)

    def student_velocity_fn(x, t, target_time=None, **obs_kwargs):
        # target_time is accepted for API compatibility with snapflow_loss_step
        # but is currently IGNORED (v0.3 runs SnapFlow-lite per docstring).
        # When v0.3.1 lands the embed_suffix surgery, route target_time
        # into the student's target_time_embed here.
        return _run_denoise_step(student, x, t, obs_kwargs)

    return teacher_velocity_fn, student_velocity_fn


def _install_target_time_embedding(student: Any) -> None:
    """[v0.3.1 scaffold] Install the zero-init target_time embedding.

    NOT CALLED by the v0.3 SnapFlow loop — kept here as the planned
    attach point for the full paper-exact SnapFlow variant.

    When v0.3.1 lands, this function:
      1. Subclasses PI0Pytorch → SnapFlowPI0Pytorch with an overridden
         `embed_suffix(state, x_t, timestep, target_time=None)` that
         adds `target_time_embed(target_time)` to the sinusoidal time
         embedding before the action_time MLP.
      2. Initializes target_time_embed's output layer to zero so the
         student starts identical to the teacher.
      3. Routes target_time through _run_denoise_step in
         `_build_pi_family_adapters` instead of ignoring it.

    See architecture doc §C.1 + §C.3 + distill_SYNTHESIS.md "Deferred".
    """
    import torch.nn as nn

    from reflex.distill.snapflow import ZeroInitTargetTimeEmbedding

    # Try to infer embedding_dim from the model config. pi0/pi05 both
    # have an action_expert with hidden_size.
    emb_dim = _infer_time_embedding_dim(student)
    embed = ZeroInitTargetTimeEmbedding(embedding_dim=emb_dim)
    # Attach mlp as a submodule so optimizer.step() updates it.
    if hasattr(student, "target_time_embed"):
        logger.warning(
            "[snapflow] student already has target_time_embed; overwriting"
        )
    # `embed` is a vanilla Python class holding an nn.Sequential; register
    # the inner module so it's tracked by student.parameters().
    if isinstance(student, nn.Module):
        student.add_module("target_time_embed_mlp", embed.mlp)
    student.target_time_embed = embed


def _infer_time_embedding_dim(model: Any) -> int:
    """Best-effort: infer the time-embedding hidden size from the model.

    pi0/pi05 typically have an `action_expert` or `expert` attribute with
    a hidden_size in its config. We fall back to 1024 (pi0-base default)
    if we can't find it, rather than crashing — the zero-init means the
    student behaves as teacher-identical regardless, so a wrong dim only
    matters once training has progressed (when the bug will be obvious).
    """
    for attr_path in (
        ("config", "expert_hidden_size"),
        ("config", "action_expert_hidden_size"),
        ("model", "config", "expert_hidden_size"),
        ("action_expert", "config", "hidden_size"),
    ):
        cur = model
        try:
            for name in attr_path:
                cur = getattr(cur, name)
            if isinstance(cur, int) and cur > 0:
                return cur
        except AttributeError:
            continue
    logger.warning(
        "[snapflow] could not infer time-embedding dim from model; "
        "defaulting to 1024 (pi0-base convention)",
    )
    return 1024


# ---------------------------------------------------------------------------
# Dataloader + batch prep
# ---------------------------------------------------------------------------

def _build_preprocessor(
    *,
    teacher_path: Path,
    image_key_map: dict | None,
    device: str,
):
    """Load the teacher's preprocessor pipeline with optional rename overrides.

    lerobot stores per-model preprocessor configs at
    `policy_preprocessor.json` alongside the weights. The pipeline does:
      1. rename_observations_processor  — maps dataset image keys to
         what the model expects (empty for `lerobot/pi0_base`; users
         must supply via image_key_map for real datasets)
      2. to_batch_processor             — tensor-dict normalization
      3. pi0_new_line_processor         — appends newline for language
      4. tokenizer_processor            — task string -> language_tokens
      5. device_processor               — moves to cuda
      6. normalizer_processor           — state/action normalization

    We override (1) with the caller's image_key_map and (5) with
    the run's device. Other steps are left as-shipped.
    """
    from lerobot.processor.pipeline import DataProcessorPipeline

    overrides: dict = {}
    if image_key_map:
        overrides["rename_observations_processor"] = {"rename_map": dict(image_key_map)}
    overrides.setdefault("device_processor", {})["device"] = device

    return DataProcessorPipeline.from_pretrained(
        str(teacher_path),
        config_filename="policy_preprocessor.json",
        overrides=overrides,
    )


def _build_dataloader(cfg, *, policy_type: str):
    """Build a LeRobotDataset dataloader for the distillation run.

    Lazy-imports lerobot.datasets so CI without lerobot installed can
    still test the rest of the loop.
    """
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(cfg.dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _prepare_batch(
    batch: dict,
    *,
    device: str,
    compute_dtype=None,
    max_action_dim: int | None = None,
    chunk_size: int | None = None,
) -> tuple:
    """Unpack a preprocessed LeRobotDataset batch into (action, noise, t, obs_kwargs).

    The batch has already been through the lerobot preprocessor pipeline,
    so tensors are on-device and language is tokenized. We split the
    action ground-truth out and leave everything else in obs_kwargs for
    the velocity fn.

    When `compute_dtype` is set (e.g. torch.bfloat16), action + noise
    are cast so the flow-matching chain runs in the model's dtype and
    avoids Linear-layer dtype mismatches downstream.

    When `max_action_dim` is set, the action is right-padded with zeros
    to that width so the flow-matching chain stays at the model's
    expected action dim (pi0/pi05 expect 32 even if the dataset is 7-dim).
    """
    import torch
    import torch.nn.functional as F

    action = batch["action"]
    if action.device.type != device.split(":")[0]:
        action = action.to(device)
    if compute_dtype is not None and action.dtype != compute_dtype:
        action = action.to(compute_dtype)
    # pi0/pi05 training expects (B, chunk_size, action_dim) — if the dataset
    # returns (B, action_dim) single-step (no delta_timestamps configured),
    # tile across chunk_size so the flow-matching chain has a time axis.
    # This is a smoke-test shortcut; production distill should configure
    # delta_timestamps on LeRobotDataset for a real action chunk.
    if chunk_size is not None and action.ndim == 2:
        action = action.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()
    if max_action_dim is not None and action.shape[-1] < max_action_dim:
        action = F.pad(action, (0, max_action_dim - action.shape[-1]))
    batch_size = action.shape[0]
    noise = torch.randn_like(action)
    t = torch.rand(batch_size, device=device)
    if compute_dtype is not None:
        t = t.to(compute_dtype)

    obs_kwargs = {k: v for k, v in batch.items() if k != "action"}
    return action, noise, t, obs_kwargs


# ---------------------------------------------------------------------------
# Checkpoint save + provenance
# ---------------------------------------------------------------------------

def _save_student_checkpoint(
    student: Any,
    checkpoint_root: Path,
    step: int,
    *,
    teacher_config: dict,
) -> Path:
    """Save the student under the standard lerobot pretrained_model/
    layout so `_auto_export` can consume it unchanged.

    Directory: <root>/<step:08d>/pretrained_model/
      ├── config.json           (inherited from teacher + distill tag)
      ├── model.safetensors     (student's full weights)
      └── distill_provenance.json  (added by _write_provenance)
    """
    ckpt_dir = checkpoint_root / f"{step:08d}" / "pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained writes config.json + model.safetensors atomically.
    save_fn = getattr(student, "save_pretrained", None)
    if save_fn is None:
        raise AttributeError(
            f"student policy has no save_pretrained — SnapFlow expects a "
            f"HuggingFace-style model. Got {type(student).__name__}."
        )
    save_fn(str(ckpt_dir))

    # Stamp the config with a distill marker so downstream consumers
    # (export, VERIFICATION.md) know this came from SnapFlow.
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
        cfg["_reflex_distill_method"] = "snapflow"
        cfg["_reflex_distill_teacher_type"] = teacher_config.get("type", "unknown")
        with config_path.open("w") as f:
            json.dump(cfg, f, indent=2)

    logger.info("[snapflow] checkpoint saved: %s", ckpt_dir)
    return ckpt_dir


def _write_provenance(
    ckpt_dir: Path,
    *,
    teacher_dir: Path,
    policy_type: str,
    steps: int,
    consistency_alpha: float,
) -> None:
    """Write a distill_provenance.json next to the checkpoint so
    VERIFICATION.md can reference where the teacher came from.

    Fields:
      - teacher_export: absolute path
      - policy_type: pi0 | pi05 | ...
      - steps: training steps completed
      - consistency_alpha: mix coef used
      - method: "snapflow"
    """
    prov = {
        "method": "snapflow",
        "policy_type": policy_type,
        "teacher_export": str(teacher_dir),
        "steps": steps,
        "consistency_alpha": consistency_alpha,
        "paper": "arxiv.org/abs/2604.05656",
    }
    (ckpt_dir / "distill_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )


__all__ = ["SnapFlowBackend"]
