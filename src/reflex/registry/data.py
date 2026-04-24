"""Curated registry rows. Edit this file (NOT models.py) to add/remove entries.

PR convention when adding a row:
1. Verify the model loads + serves with `reflex serve <export>` end-to-end
2. Run the bundled parity test against an existing model in the same family
3. Pin `hf_revision` to a specific commit sha (not HEAD) for reproducibility
4. Add benchmark numbers from the matching `03_experiments/*.md` if measured
5. Set `requires_export=False` ONLY if the HF repo contains
   reflex_config.json + the .onnx files (a Reflex-ready export, not raw weights)

Initial registry seeded 2026-04-24 with public LeRobot-distributed weights —
all require `reflex export` after `pull` because they ship as raw PyTorch
checkpoints. Reflex-pre-exported entries (under our own HF org) added in a
follow-up once the export-and-upload pipeline ships.
"""
from __future__ import annotations

from reflex.registry.models import ModelBenchmark, ModelEntry

REGISTRY: tuple[ModelEntry, ...] = (
    ModelEntry(
        model_id="pi0-base",
        hf_repo="lerobot/pi0_base",
        family="pi0",
        action_dim=32,
        size_mb=14000,  # ~14 GB BF16
        supported_embodiments=("franka", "so100", "ur5"),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=110.0, p99_ms=180.0, vram_mb=14000,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="pi0 base from Physical Intelligence (LeRobot mirror). 7B params, BF16. "
                    "Run `reflex export pi0-base` after pull to produce ONNX.",
        license="apache-2.0",
        hf_revision=None,  # use HEAD until we pin
    ),
    ModelEntry(
        model_id="pi05-base",
        hf_repo="lerobot/pi05_base",
        family="pi05",
        action_dim=32,
        size_mb=14000,
        supported_embodiments=("franka", "so100", "ur5"),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=98.0, p99_ms=160.0, vram_mb=14000,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="pi0.5 base — Physical Intelligence's improved pi0 (state-out variant). "
                    "Use this as the teacher for SnapFlow distillation.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="pi05-libero",
        hf_repo="lerobot/pi05_libero_finetuned_v044",
        family="pi05",
        action_dim=7,  # Franka — 7 joints
        size_mb=14000,
        supported_embodiments=("franka",),
        supported_devices=("a10g", "a100", "h100"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=98.0, p99_ms=160.0, vram_mb=14000,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="pi0.5 finetuned on LIBERO benchmark (Franka 7-DoF). "
                    "Reflex's reference SnapFlow teacher — 28/30 task-success at 10 NFE.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="smolvla-base",
        hf_repo="lerobot/smolvla_base",
        family="smolvla",
        action_dim=7,
        size_mb=900,  # SmolVLA is small (~900MB FP32)
        supported_embodiments=("franka", "so100"),
        supported_devices=("orin_nano", "agx_orin", "a10g", "a100", "h100"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=22.0, p99_ms=45.0, vram_mb=900,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="SmolVLA — HuggingFace's small VLA. Edge-friendly (~900MB). "
                    "Best starting point for Jetson Orin Nano deployments.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="smolvla-libero",
        hf_repo="lerobot/smolvla_libero",
        family="smolvla",
        action_dim=7,
        size_mb=900,
        supported_embodiments=("franka",),
        supported_devices=("orin_nano", "agx_orin", "a10g", "a100"),
        benchmarks=(),  # not measured yet
        requires_export=True,
        description="SmolVLA finetuned on LIBERO. Smaller-footprint alternative to pi0.5-libero "
                    "for edge-first deployments.",
        license="apache-2.0",
        hf_revision=None,
    ),
)
