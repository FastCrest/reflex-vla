"""FP16 conversion helpers for monolithic ONNX exports.

Halves on-disk size. Targeted at the pi0 / pi0.5 Orin Nano 8GB fit
problem — pi0 FP32 is 12.5GB, won't load; pi0 FP16 should be ~6.3GB.

Strategy: use `onnxconverter_common.float16.convert_float_to_float16`
with an op blocklist for ops known to underflow at FP16
(LayerNorm-adjacent: Pow, ReduceMean, Sqrt), and `keep_io_types=True`
so downstream clients don't need to change their input tensors.

Pure ONNX → ONNX conversion; no retraining, no TensorRT dependency.
Post-conversion, the FP16 ONNX goes through the same parity harness
as the FP32 export. See reflex_context/01_architecture/orin_nano_fp16_plan.md
for the full plan.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Ops that commonly underflow at FP16 in flow-matching / transformer VLAs.
# Keeping these in FP32 is the standard mitigation — the cost is ~1-2% of
# total weights still at FP32, which is worth it for the correctness win.
FP16_OP_BLOCKLIST: tuple[str, ...] = (
    "Pow",           # x^2 / sqrt(x) in LayerNorm variance calc
    "ReduceMean",    # LayerNorm mean
    "Sqrt",          # LayerNorm denominator
    # Add empirically as we discover them:
    # "Softmax",     # if attention scores destabilize
    # "LayerNormalization",  # full op keep-in-FP32
)


def estimate_fp16_size_bytes(fp32_total_bytes: int) -> int:
    """Rough estimate of post-conversion size.

    Weights are roughly halved (fp32 → fp16). Graph structure + a few
    per-op FP32 residuals (the blocklist) add ~5% overhead on top of
    the halved weight footprint. This gets us within ~10% of the actual
    post-conversion size — good enough for pre-flight fit checks.

    Real conversion size is reported by onnx.save() once the run
    completes.
    """
    halved = fp32_total_bytes // 2
    overhead = halved // 20  # 5% slop for metadata + blocklist residuals
    return halved + overhead


def parity_gate(
    max_abs_diff: float,
    cos_sim: float,
    *,
    max_abs_threshold: float = 5e-3,
    cos_threshold: float = 0.999,
) -> dict[str, Any]:
    """Apply the FP16-vs-FP32 parity gate.

    Returns a verdict dict compatible with VERIFICATION.md seeding.
    PASS requires cos > cos_threshold AND max_abs < max_abs_threshold.
    Either failure flips to FAIL with a human-readable reason.
    """
    cos_ok = cos_sim >= cos_threshold
    maxabs_ok = max_abs_diff <= max_abs_threshold
    passed = cos_ok and maxabs_ok

    reasons: list[str] = []
    if not cos_ok:
        reasons.append(
            f"cos_sim {cos_sim:.6f} below threshold {cos_threshold}"
        )
    if not maxabs_ok:
        reasons.append(
            f"max_abs_diff {max_abs_diff:.2e} above threshold "
            f"{max_abs_threshold:.0e}"
        )

    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "cos_threshold": cos_threshold,
        "max_abs_threshold": max_abs_threshold,
        "reasons": reasons,
    }


def convert_fp32_to_fp16(
    fp32_onnx_path: str | Path,
    fp16_onnx_path: str | Path,
    *,
    keep_io_types: bool = True,
    op_block_list: tuple[str, ...] | None = None,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
) -> dict[str, Any]:
    """Convert an ONNX model on disk from FP32 weights to FP16.

    Args:
        fp32_onnx_path: input ONNX (with external data if >2GB).
        fp16_onnx_path: output ONNX path; external data goes
            alongside with the same stem + `.bin`.
        keep_io_types: if True (default), the graph's input/output
            tensors stay FP32 — callers don't have to change their
            numpy dtype. Internal weights + activations go FP16.
        op_block_list: ops kept in FP32 to avoid underflow.
            Defaults to FP16_OP_BLOCKLIST.
        min_positive_val / max_finite_val: clamps for FP16 range
            (smaller/larger values are saturated to avoid NaN/Inf).

    Returns: a summary dict — input/output paths, byte sizes, reduction ratio.
    """
    try:
        import onnx
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError as e:
        raise ImportError(
            "FP16 conversion requires `onnx` and `onnxconverter-common`. "
            "Install with: pip install onnx onnxconverter-common"
        ) from e

    src = Path(fp32_onnx_path)
    dst = Path(fp16_onnx_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Protobuf's 2GB-per-message limit bites both `ByteSize()` AND
    # `SerializeToString()`. Detect oversized models by checking the
    # on-disk size of external data BEFORE loading — if weights alone
    # exceed ~1.8GB, skip internal shape inference (which would
    # serialize the loaded proto).
    on_disk_bytes = _size_with_external(src)
    disable_shape_infer = on_disk_bytes > 1_800_000_000

    logger.info("[fp16] Loading %s (%.2f GB on disk)...",
                src, on_disk_bytes / 1e9)
    model_fp32 = onnx.load(str(src), load_external_data=True)

    blocklist = list(op_block_list if op_block_list is not None else FP16_OP_BLOCKLIST)

    logger.info(
        "[fp16] Converting (keep_io=%s, blocklist=%s, disable_shape_infer=%s)...",
        keep_io_types, blocklist, disable_shape_infer,
    )
    # The smolvla libero monolithic is 2.2GB on disk — triggers disable_shape_infer
    # (we don't want to hit the 2GB protobuf serialization limit during
    # onnx.shape_inference). pi0 (12.5GB) and pi0.5 (13GB) will also use this path.
    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=keep_io_types,
        op_block_list=blocklist,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        disable_shape_infer=disable_shape_infer,
    )

    logger.info("[fp16] Saving %s...", dst)
    onnx.save(
        model_fp16,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{dst.stem}.bin",
        size_threshold=1024,
    )

    # Size accounting
    fp32_bytes = _size_with_external(src)
    fp16_bytes = _size_with_external(dst)
    reduction = 1.0 - (fp16_bytes / fp32_bytes) if fp32_bytes else 0.0

    logger.info(
        "[fp16] %s -> %s: %.1f GB -> %.1f GB (%.1f%% reduction)",
        src.name, dst.name,
        fp32_bytes / 1e9, fp16_bytes / 1e9, reduction * 100,
    )

    return {
        "src_path": str(src),
        "dst_path": str(dst),
        "src_bytes": fp32_bytes,
        "dst_bytes": fp16_bytes,
        "reduction_ratio": reduction,
        "op_block_list": blocklist,
    }


def _size_with_external(onnx_path: Path) -> int:
    """Size of the graph file + any .bin/.data siblings."""
    total = onnx_path.stat().st_size
    for pat in ("*.bin", "*.data"):
        for p in onnx_path.parent.glob(pat):
            if p.stem == onnx_path.stem or p.stem.startswith(onnx_path.stem):
                total += p.stat().st_size
    return total


__all__ = [
    "FP16_OP_BLOCKLIST",
    "convert_fp32_to_fp16",
    "estimate_fp16_size_bytes",
    "parity_gate",
]
