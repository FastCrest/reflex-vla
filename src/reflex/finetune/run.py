"""The run_finetune() orchestration.

v0.3 flow:
  1. Validate config (basic, not full preflight — that's v0.5)
  2. Invoke the chosen backend's fit() — currently lerobot only
  3. Locate the final checkpoint in <output>/checkpoints/
  4. Auto-invoke reflex export on it unless skip_export=True
  5. Return a FinetuneResult with paths + status

v0.5 will add: preflight schema + memory + norm-stats check, parity-gate
at each checkpoint save, calibration-on-holdout, pluggable action-head
registry. See architecture doc Section D for the full target shape.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from reflex.finetune.config import FinetuneConfig, FinetuneResult

logger = logging.getLogger(__name__)


def _validate_config(cfg: FinetuneConfig) -> list[str]:
    """Cheap up-front checks. Returns list of error messages (empty = OK).

    Real preflight (checkpoint loading, dataset schema validation,
    memory budget) is v0.5; this just catches obvious misconfigurations
    before we spin up any training process.
    """
    errs: list[str] = []
    if not cfg.base:
        errs.append("base is required (e.g. lerobot/smolvla_base)")
    if not cfg.dataset:
        errs.append("dataset is required (e.g. lerobot/libero)")
    if cfg.num_steps <= 0:
        errs.append(f"num_steps must be > 0; got {cfg.num_steps}")
    if cfg.batch_size <= 0:
        errs.append(f"batch_size must be > 0; got {cfg.batch_size}")
    if cfg.mode not in ("lora", "lora-cross-embodiment", "full"):
        errs.append(
            f"mode must be one of lora|lora-cross-embodiment|full; got {cfg.mode!r}"
        )
    if cfg.mode != "lora":
        # v0.3 only supports LoRA. Flag explicitly instead of silently
        # proceeding (the subprocess would fail later with a confusing error).
        errs.append(
            f"v0.3 only supports --mode lora; {cfg.mode!r} lands in v0.5+"
        )
    if cfg.backend != "lerobot":
        errs.append(
            f"v0.3 only supports --backend lerobot; {cfg.backend!r} lands in v0.5+"
        )
    if cfg.precision not in ("bf16", "fp32"):
        errs.append(
            f"precision must be bf16 or fp32; got {cfg.precision!r}"
        )
    return errs


def _build_lerobot_command(cfg: FinetuneConfig) -> list[str]:
    """Construct the lerobot-train invocation.

    lerobot-train accepts draccus-style CLI args (dotted keys). We
    translate our flat FinetuneConfig into the equivalent. Extra knobs
    not yet first-class in FinetuneConfig come through extra_lerobot_args.

    Schema here is best-effort for lerobot 0.5.1; the generated command
    is surfaced in the training log so customers can reproduce manually
    if lerobot's CLI shifts upstream.
    """
    cmd = [
        "lerobot-train",
        f"--policy.path={cfg.base}",
        f"--dataset.repo_id={cfg.dataset}",
        f"--output_dir={cfg.output}",
        f"--steps={cfg.num_steps}",
        f"--batch_size={cfg.batch_size}",
        f"--policy.optimizer_lr={cfg.learning_rate}",
        f"--seed={cfg.seed}",
        # LoRA-specific knobs — lerobot 0.4.4+ supports these via PEFT.
        f"--policy.use_lora=true",
        f"--policy.lora_rank={cfg.lora_rank}",
        f"--precision={cfg.precision}",
    ]
    for k, v in cfg.extra_lerobot_args.items():
        cmd.append(f"--{k}={v}")
    return cmd


def _run_lerobot_training(
    cfg: FinetuneConfig,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> int:
    """Invoke lerobot-train via subprocess. Streams stdout to log file
    and the root logger. Returns the subprocess exit code.
    """
    cmd = _build_lerobot_command(cfg)
    logger.info("[finetune] exec: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"# reflex finetune — lerobot-train invocation\n")
        log.write(f"# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            # Mirror to our logger at INFO — customers see progress.
            logger.info(line.rstrip())
        proc.wait()
        return proc.returncode


def _locate_checkpoint(output_dir: Path) -> Path | None:
    """Find the final lerobot checkpoint under the output dir.

    lerobot's train script writes checkpoints under
    <output>/checkpoints/<step>/pretrained_model/. We pick the highest-
    numbered step.
    """
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    step_dirs = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not step_dirs:
        return None
    # Try numeric sort first (lerobot uses step-number dirs). Fall back
    # to mtime if names aren't numeric.
    try:
        step_dirs.sort(key=lambda p: int(p.name))
    except ValueError:
        step_dirs.sort(key=lambda p: p.stat().st_mtime)
    final = step_dirs[-1] / "pretrained_model"
    if final.exists():
        return final
    return step_dirs[-1]


def _auto_export(checkpoint: Path, cfg: FinetuneConfig) -> tuple[Path | None, str | None]:
    """Run reflex's existing monolithic export on the fine-tuned checkpoint.

    Returns (onnx_path, error). On success, error is None.
    """
    try:
        from reflex.exporters.monolithic import export_monolithic
    except ImportError as exc:
        return None, f"reflex.exporters.monolithic import failed: {exc}"

    try:
        result = export_monolithic(
            model_id=str(checkpoint),
            output_dir=cfg.output / "export",
            target=cfg.target,
        )
        onnx_path = result.get("onnx_path") if isinstance(result, dict) else None
        if onnx_path is None:
            return None, "export_monolithic returned no onnx_path"
        return Path(onnx_path), None
    except Exception as exc:
        return None, f"export_monolithic raised: {type(exc).__name__}: {exc}"


def run_finetune(cfg: FinetuneConfig) -> FinetuneResult:
    """Run a fine-tune end-to-end: train → auto-export → receipt.

    v0.3 flow: SmolVLA LoRA via subprocess-lerobot-train, then reflex
    export on the final checkpoint. No parity gate, no calibration, no
    pre-flight — those land in v0.5. A run_finetune() call that reaches
    cfg.num_steps + exports ONNX is a SUCCESS; anything else is an
    error with actionable details in FinetuneResult.error.
    """
    cfg.output.mkdir(parents=True, exist_ok=True)
    training_log = cfg.output / "training_log.jsonl"

    errs = _validate_config(cfg)
    if errs:
        return FinetuneResult(
            status="aborted",
            output_dir=cfg.output,
            error="config validation failed:\n  " + "\n  ".join(errs),
        )

    logger.info("[finetune] start: base=%s dataset=%s output=%s steps=%d",
                cfg.base, cfg.dataset, cfg.output, cfg.num_steps)
    t0 = time.time()

    rc = _run_lerobot_training(cfg, training_log)
    elapsed = time.time() - t0
    logger.info("[finetune] training exit_code=%d elapsed=%.1fs", rc, elapsed)

    if rc != 0:
        return FinetuneResult(
            status="training_failed",
            output_dir=cfg.output,
            training_log_path=training_log,
            error=f"lerobot-train exited with code {rc}; see {training_log}",
        )

    checkpoint = _locate_checkpoint(cfg.output)
    if checkpoint is None:
        return FinetuneResult(
            status="training_failed",
            output_dir=cfg.output,
            training_log_path=training_log,
            error=f"no checkpoint found under {cfg.output / 'checkpoints'}; "
                  f"training reported success but produced no output",
        )

    result = FinetuneResult(
        status="ok",
        output_dir=cfg.output,
        training_steps_completed=cfg.num_steps,
        final_checkpoint_path=checkpoint,
        training_log_path=training_log,
    )

    if cfg.skip_export:
        logger.info("[finetune] skip_export=True; done (no ONNX)")
        return result

    logger.info("[finetune] auto-exporting checkpoint %s", checkpoint)
    onnx_path, export_err = _auto_export(checkpoint, cfg)
    if export_err:
        result.status = "export_failed"
        result.error = export_err
        return result

    result.onnx_path = onnx_path
    # If the export wrote a VERIFICATION.md, surface its path.
    v_md = cfg.output / "export" / "VERIFICATION.md"
    if v_md.exists():
        result.verification_md_path = v_md
    logger.info("[finetune] DONE: onnx=%s", onnx_path)
    return result


__all__ = ["run_finetune"]
