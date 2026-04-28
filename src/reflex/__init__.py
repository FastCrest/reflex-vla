"""Reflex — Deploy any VLA model to any edge hardware. One command."""

__version__ = "0.7.0"

# Heavy submodules (validate_roundtrip pulls in torch) are lazy-loaded so that
# `reflex --version`, `reflex --help`, `reflex chat`, etc. don't pay the
# 700ms+ torch-import cost on every invocation. Importers like
# `from reflex import ValidateRoundTrip` still work — the __getattr__ hook
# imports on first access.
__all__ = [
    "__version__",
    "ValidateRoundTrip",
    "SUPPORTED_MODEL_TYPES",
    "UNSUPPORTED_MODEL_MESSAGE",
    "load_fixtures",
]


# ─── ORT-TRT EP first-class support (v0.7) ──────────────────────────────────
# ORT-TRT EP needs libnvinfer.so.10 (from the `tensorrt` pip pkg) + CUDA libs
# (libcublas, libcudnn) loadable at session-create time. The pip-installed
# nvidia/tensorrt libs live under site-packages but Linux's dynamic loader
# doesn't know to look there. Without LD_LIBRARY_PATH set, ORT-TRT EP fails
# to load and ORT silently falls back to CUDA EP — losing the 5.55× perf win
# measured 2026-04-29 (Modal A10G, SmolVLA monolithic).
#
# We auto-prepend the right paths at import time. Idempotent (won't re-add if
# already present). No-op on macOS/Windows or when the paths don't exist.
# Opt out via REFLEX_NO_LD_LIBRARY_PATH_PATCH=1.
#
# Per ADR 2026-04-29-ort-trt-ep-first-class-support.md.
def _patch_ld_library_path() -> None:
    """Prepend pip-installed nvidia/tensorrt lib dirs to LD_LIBRARY_PATH.

    Runs at module load (NOT lazy) so it takes effect before any other
    import triggers CUDA / ORT initialization. Returns silently on macOS,
    Windows, or when no paths exist.
    """
    import os
    import sys

    if os.environ.get("REFLEX_NO_LD_LIBRARY_PATH_PATCH"):
        return
    if sys.platform not in ("linux", "linux2"):
        return

    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"

    # Candidate lib dirs from pip-installed nvidia/* + tensorrt pkgs.
    # Ordered so libnvinfer (TRT EP's runtime dep) is found first.
    candidates = []
    for base in (sys.prefix, "/usr/local"):
        candidates.extend([
            f"{base}/lib/{py_lib}/site-packages/tensorrt_libs",
            f"{base}/lib/{py_lib}/site-packages/tensorrt",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cudnn/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cublas/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cuda_runtime/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cuda_nvrtc/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/nccl/lib",
        ])

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(os.pathsep) if p]

    to_prepend = []
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if path in existing_parts or path in to_prepend:
            continue  # idempotent: don't re-add
        to_prepend.append(path)

    if not to_prepend:
        return  # nothing to add — silent no-op

    new_value = os.pathsep.join(to_prepend + existing_parts)
    os.environ["LD_LIBRARY_PATH"] = new_value


_patch_ld_library_path()


def __getattr__(name: str):
    if name in {"ValidateRoundTrip", "SUPPORTED_MODEL_TYPES", "UNSUPPORTED_MODEL_MESSAGE"}:
        from reflex.validate_roundtrip import (
            SUPPORTED_MODEL_TYPES,
            UNSUPPORTED_MODEL_MESSAGE,
            ValidateRoundTrip,
        )
        return {
            "ValidateRoundTrip": ValidateRoundTrip,
            "SUPPORTED_MODEL_TYPES": SUPPORTED_MODEL_TYPES,
            "UNSUPPORTED_MODEL_MESSAGE": UNSUPPORTED_MODEL_MESSAGE,
        }[name]
    if name == "load_fixtures":
        from reflex.fixtures import load_fixtures
        return load_fixtures
    raise AttributeError(f"module 'reflex' has no attribute {name!r}")
