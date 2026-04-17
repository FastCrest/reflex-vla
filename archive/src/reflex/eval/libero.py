"""LIBERO task-success evaluation plugin.

Requires: pip install 'reflex-vla[eval]' (adds vla-eval + MuJoCo + robosuite).

The implementation delegates to AllenAI's vla-evaluation-harness. We spin up
a model server adapter that wraps the Reflex ONNX pipeline, then invoke
vla-eval's LIBERO benchmark against it.

For the Modal-hosted version, see scripts/modal_libero10.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["run_libero"]

SUITE_MAP = {
    "libero_10": "libero_10",
    "libero_long": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}


def run_libero(
    *,
    suite: str = "libero_10",
    export_dir: str,
    episodes_per_task: int = 10,
    device: str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run LIBERO evaluation via vla-eval.

    v0.2 status: this is a stub that imports the vla-eval framework and reports
    whether the deps are available. Full eval execution still runs via the
    Modal script at scripts/modal_libero10.py, which handles the LIBERO
    installation quirks (interactive setup wizard, robosuite version pinning,
    EGL rendering) that haven't yet been packaged into the base install.
    """
    try:
        import vla_eval  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LIBERO eval requires the eval extra:\n"
            "  pip install 'reflex-vla[eval]'\n"
            "Or use the ready-to-run Modal script: modal run scripts/modal_libero10.py"
        ) from exc

    suite_name = SUITE_MAP.get(suite, "libero_10")

    # Verify the export has the files we need
    expert_onnx = Path(export_dir) / "expert_stack.onnx"
    if not expert_onnx.exists():
        raise FileNotFoundError(
            f"expert_stack.onnx not found in {export_dir}. Run `reflex export` first."
        )

    # TODO(v0.3): inline LIBERO execution. For now point at the Modal script.
    return {
        "benchmark": f"libero:{suite_name}",
        "status": "not_implemented_local",
        "message": (
            f"Local LIBERO eval not yet shipped (v0.3). "
            f"Use: modal run scripts/modal_libero10.py "
            f"(pass episodes_per_task via the config in that script)."
        ),
        "success_rate": None,
        "episodes_completed": 0,
        "per_task": [],
        "export_dir": str(export_dir),
        "episodes_per_task_requested": episodes_per_task,
    }
