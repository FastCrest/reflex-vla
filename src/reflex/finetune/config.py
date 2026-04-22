"""Configs + result shapes for reflex finetune.

Decoupled from run.py so tests + external callers can construct a
FinetuneConfig without importing the full orchestration path (which
would pull lerobot + HF + ORT).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FinetuneConfig:
    """Input to run_finetune(). Flat dataclass; maps ~1:1 to CLI flags.

    v0.3 exposes a narrow slice of what the full architecture doc lists.
    Extra fields are added here as backends / preflight / action-head
    registries land in v0.5+.
    """

    base: str
    """HF model id of the base checkpoint (e.g. lerobot/smolvla_base)."""

    dataset: str
    """HF dataset id to fine-tune on (e.g. lerobot/libero)."""

    output: Path
    """Output directory. Will contain model.onnx + VERIFICATION.md after
    successful run."""

    num_steps: int = 20_000
    """Total training steps. For SmolVLA LoRA, 2-20k is typical depending
    on dataset size."""

    batch_size: int = 8
    learning_rate: float = 1e-4
    mode: str = "lora"
    """One of: lora | lora-cross-embodiment | full. v0.3 supports lora."""

    lora_rank: int = 32
    """Default r=32 — VLA intrinsic rank is 4-16x LLM's (LoRA-SP).
    lerobot default is r=16; we override to avoid the norm-stats
    mismatch that drops task success 96%→1% (competitive research #6)."""

    precision: str = "bf16"
    """bf16 by default (SOTA research A4). Options: bf16 | fp32."""

    seed: int = 42
    backend: str = "lerobot"
    """v0.3 supports only lerobot. openpi + hf_transformers in v0.5+."""

    skip_export: bool = False
    """v0.3 auto-exports on success. Set True to skip (e.g. for a quick
    training sanity check that doesn't need ONNX)."""

    target: str = "desktop"
    """Hardware profile for the auto-export step."""

    extra_lerobot_args: dict[str, Any] = field(default_factory=dict)
    """Escape hatch: pass through extra args to lerobot-train. Used for
    features not yet first-class in FinetuneConfig."""

    dry_run: bool = False
    """If True, run preflight checks and exit. No GPU time spent.
    Useful for validating config before committing to a multi-hour run."""

    skip_preflight: bool = False
    """If True, skip preflight validation. Escape hatch for local-dataset
    or gated-repo flows where preflight can't resolve the schema.
    Only set if you know what you're doing."""

    phase: str = "train"
    """One of: 'train' | 'distill'.
    - 'train' (default, v0.3): fine-tune via lerobot-train subprocess
    - 'distill' (v0.3+): SnapFlow distillation. Requires teacher_export.
    Routed to backends via reflex.finetune.backends.resolve_backend()."""

    teacher_export: str | None = None
    """For phase='distill' only: path/HF-id of the teacher's reflex-
    export dir (merged PyTorch checkpoint). None for phase='train'."""

    distillation_method: str = "snapflow"
    """For phase='distill' only: which distillation trainer to use.
    v0.3 supports 'snapflow' only. 'consistency' (DDPM for GR00T)
    lands in v0.5+."""

    variant: str = "default"
    """Architecture variant of the student. v0.5 adds 'state_out' for
    pi0.5 students: strips state from the language prompt and adds an
    explicit state_proj layer that consumes proprio state. Unlocks the
    prefix KV cache in production (lang becomes stable per episode).
    Values:
      - 'default': v0.3.1 SnapFlow student (state-in-lang via the
        default pi0.5 preprocessor + no state_proj layer).
      - 'state_out': pi0.5 only. Applies enable_snapflow_state_out
        to the student + swaps the preprocessor step.
    See reflex_vault 01_architecture/distill_state_out_pi05_design.md."""

    def __post_init__(self) -> None:
        self.output = Path(self.output)


@dataclass
class FinetuneResult:
    """Output of run_finetune()."""

    status: str
    """One of: ok | training_failed | export_failed | aborted."""

    output_dir: Path
    training_steps_completed: int = 0
    final_checkpoint_path: Path | None = None

    onnx_path: Path | None = None
    """If skip_export=False and status=="ok", path to the exported ONNX."""

    verification_md_path: Path | None = None
    """Path to VERIFICATION.md written by the auto-export chain."""

    training_log_path: Path | None = None
    """Path to training_log.jsonl (stdout capture of the training run)."""

    error: str | None = None
    """Populated when status != "ok"."""

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        for attr in (
            "final_checkpoint_path",
            "onnx_path",
            "verification_md_path",
            "training_log_path",
        ):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, Path(v))


__all__ = ["FinetuneConfig", "FinetuneResult"]
