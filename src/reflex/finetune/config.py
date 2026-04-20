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
