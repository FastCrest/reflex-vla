# Orin Nano FP16 fit plan (2026-04-19)

## Why this matters

pi0 and pi0.5 don't fit on Orin Nano 8GB at FP32 — the monolithic ONNX files are 12.5GB and 13GB respectively. SmolVLA (1.6GB) and GR00T (4.4GB) already fit.

This is a product-pitch gap: the README says "deploy any VLA to any Jetson" but the most-deployed VLA family (pi) literally can't load on the most-deployed Jetson ($499 Orin Nano).

FP16 halves memory (weights go from 4 bytes to 2). Expected post-conversion sizes:
- pi0 FP16: ~6.3 GB on disk
- pi0.5 FP16: ~6.5 GB

Plus runtime activations (~1-2 GB) + OS (~1 GB). **Estimated fit: tight but plausible at 8 GB.**

## Conversion approach

Two viable paths:

### Option A: onnxconverter-common float16
```python
from onnxconverter_common.float16 import convert_float_to_float16
model_fp32 = onnx.load("pi0/model.onnx", load_external_data=True)
model_fp16 = convert_float_to_float16(
    model_fp32,
    keep_io_types=True,   # keep inputs/outputs FP32 — avoids downstream client changes
    min_positive_val=1e-7,
    max_finite_val=1e4,
    op_block_list=["Pow", "ReduceMean", "Sqrt"],  # LayerNorm-adjacent; FP16 often underflows
)
onnx.save(model_fp16, "pi0_fp16/model.onnx", save_as_external_data=True)
```

Pros: one-shot, no retraining, deterministic output.
Cons: naive blanket cast can underflow on softmax, layernorm, small-value ops. Need op blocklist + min_positive_val clamping.

### Option B: TensorRT builder with --fp16
```bash
trtexec --onnx=pi0/model.onnx --fp16 --saveEngine=pi0_fp16.plan \
        --minShapes=... --optShapes=... --maxShapes=...
```

Pros: runtime picks per-op FP16 based on calibration; usually higher parity.
Cons: requires TensorRT. Jetson-side compile. Orin Nano-specific SM 8.7 build. Adds a deployment tool dependency.

**Decision**: start with Option A (pure ONNX → ONNX conversion, Modal-runnable). Validate parity. If cos < 0.99 on any model, fall back to Option B with op calibration.

## Parity gates

Per model:
- **Gate 1 (correctness)**: first-action cos sim > 0.999 vs FP32, max_abs < 5e-3. Re-use `scripts/modal_{pi0,pi05}_monolithic_export.py --parity` with a FP16-loaded ONNX.
- **Gate 2 (task success)**: if we have a rollout harness for the model (LIBERO for smolvla_libero, no sim for pi0 today), task success must be within ±5pp of FP32 on N=25. Gate deferred for pi0 until we have a pi0 rollout harness.
- **Gate 3 (fit)**: on a 8GB Jetson Orin Nano (or emulated memory limit), ORT session loads successfully AND runs one forward pass. Needs real hardware OR a Modal job with `--gpu A10G` and `CUDA_VISIBLE_MEMORY_LIMIT=8192` hack.

## Modal run plan

Two new Modal scripts:

1. `scripts/modal_fp16_convert.py` — entrypoint `--model-id pi0_monolithic --out-subdir pi0_monolithic_fp16`. Loads the FP32 ONNX from the `pi0-onnx-outputs` volume, runs `convert_float_to_float16`, writes back to the volume under a new subdir. Measures size reduction. ~5-10 min per model on CPU (conversion doesn't need GPU).

2. `scripts/modal_pi0_fp16_parity.py` — load the FP16 ONNX with ORT, run shared seeded inputs, compare against the PyTorch reference. Reuses `PI0Pytorch.sample_actions(num_steps=10)` from the existing parity harness. Expected output row for `measured_numbers.md`:
   > pi0 FP16 vs PyTorch num_steps=10: first-action cos=?.?????, max_abs=?.??e-?? — verdict PASS/FAIL

## Test coverage

`tests/test_orin_nano_fp16_fit.py` — unit tests for:
- Sizing: compute expected FP16 size from FP32 (weights/2 + 2KB overhead per external_data file). Assert the estimate is plausible for pi0 (12.5GB → ~6.3GB).
- Op blocklist: given a synthetic onnx proto, `_with_fp16_safe_ops()` produces the expected list (Pow, ReduceMean, Sqrt, plus any we learn need it empirically).
- Parity threshold plumbing: given `max_abs=1e-3`, the gate returns PASS; at `max_abs=1.0`, FAIL.

These are hermetic — no model load, no Modal. The real parity runs happen on Modal and populate `measured_numbers.md`.

## Rollout

1. ✅ Land conversion scaffold + tests (this doc + files).
2. ✅ Trigger Modal FP16 conversion for pi0 (~10 min compute). **Result: 12.52 GB → 6.26 GB (-50.0%), 248s on Modal CPU.**
3. ✅ Cast-insertion post-pass unblocks ORT direct-load (`fix_fp16_dtype_mismatches`). Walks graph, detects mixed FP32/FP16 operands on Mul/MatMul/Add/Sub/Div/Pow/Gemm/Concat/Where, inserts `Cast(to=FP16)` on the FP32 operand. For pi0: 2 Cast nodes inserted. For smolvla_libero: 484 Cast nodes.
4. ✅ FP16 parity run on smolvla_libero: **cos=+0.999994, max_abs=7.82e-03, PASS** at 1e-2 threshold. First end-to-end FP16 ONNX that ORT loads AND preserves behavior within expected FP16 precision.
5. ✅ Add measured row to `measured_numbers.md` (two rows: conversion + parity).
6. (pending) Update README's "Memory fit" table with FP16 numbers + add pi0 Orin Nano as supported.
7. (Optional) Repeat 2-4 for pi0.5 (13 GB FP32 → est. ~6.6 GB FP16).
8. (Pending — hardware dependent) Jetson-side real-device fit test + TRT engine FP16 build.

## Learnings from the conversion pass (2026-04-19)

- **Don't use `ByteSize()` to gate logic on model size** — ByteSize() itself serializes the proto and hits the 2GB limit. Use on-disk size via `_size_with_external()`.
- **External data rewrite requires explicit unlink of old siblings** — `onnx.save(save_as_external_data=True)` will leave stale `.bin` files alongside the new one if the destination was previously written. Need to `unlink()` first.
- **`keep_io_types=True` breaks on oversized graphs** — Cast-node wiring gets Mul/Add operand dtypes wrong. For >1.8GB we flip to `keep_io_types=False` (end-to-end FP16; callers cast inputs/outputs).
- **Op blocklist causes ORT load failures** — blocklisted ops (Pow/ReduceMean/Sqrt) stay FP32, but downstream MatMul/Mul receives FP16 weights → operand dtype mismatch. `convert_float_to_float16` doesn't insert Cast nodes at blocklist boundaries. Empty blocklist is the safer default.
- **`infer_shapes_path` rehydrates weights to FP32** — calling it post-save undoes the FP16 conversion silently (size went from 1.13 GB back to 2.24 GB). Use value_info-strip instead.
- **TRT is the Jetson deployment target anyway** — the ORT-FP16 load failure isn't a blocker for production; TRT engine build handles FP16 conversion independently. But ORT-FP16 direct-load would be a nice developer-ergonomic win.

## Risk / failure modes

- **FP16 activations underflow** on long flow-matching loops (num_steps=10 integrates velocity 10x; small errors compound). If cos drops below 0.99, the op blocklist probably needs `Softmax` or `LayerNorm`-adjacent ops added.
- **ORT provider mismatch**: some ops downcasted to FP16 may not have FP16 kernels on CUDAExecutionProvider. ORT will fall back to FP32 transparently; only cost is memory (defeats the point). `onnxruntime-gpu` 1.20+ has good FP16 coverage for transformer ops.
- **ExternalData rewrite**: the FP16 conversion rewrites the big .bin file — need to save with `save_as_external_data=True` to keep the ONNX graph tiny and weights in one blob. Without this, ONNX files grow to 2GB limit.

## Related

- `reflex_context/measured_numbers.md` — line 153: "pi0 realistically needs Orin 16GB+ or a desktop NVIDIA GPU" (the claim this goal removes)
- `scripts/modal_pi0_monolithic_export.py` — FP32 source export
- `scripts/modal_pi05_monolithic_export.py` — same for pi0.5
- `GOALS.yaml`: `orin-nano-fp16-fit` (weight 8)
