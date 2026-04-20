"""Reflex fine-tuning pipeline.

Thin orchestrator over lerobot-train (and eventually openpi-JAX) that:
  1. Validates dataset + checkpoint compatibility before GPU time
  2. Invokes the chosen training backend
  3. Auto-runs reflex export on the resulting checkpoint
  4. Attaches a VERIFICATION.md receipt with parity + calibration

v0.3 scope: SmolVLA LoRA only via subprocess-lerobot-train + auto-export.
Everything else (pi0, parity-gate, calibration-first eval, pluggable
action heads, openpi-JAX backend) is v0.5+.

Design doc: reflex_context/01_architecture/finetune_SYNTHESIS.md
Architecture: reflex_context/01_architecture/finetune_architecture.md
"""
from __future__ import annotations

from reflex.finetune.config import FinetuneConfig, FinetuneResult
from reflex.finetune.run import run_finetune

__all__ = [
    "FinetuneConfig",
    "FinetuneResult",
    "run_finetune",
]
