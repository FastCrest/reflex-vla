"""OpenAI-style tool schemas for the Reflex chat agent.

Each tool maps to a `reflex <subcommand>` invocation. Schemas are kept tight: only
the parameters the LLM is likely to fill correctly. Power users use the CLI directly.
"""

from __future__ import annotations

from typing import Any

# Hardware target choices reused across multiple tools.
_TARGETS = ["orin-nano", "orin", "orin-64", "thor", "desktop"]
_PRECISIONS = ["fp16", "fp8", "int8"]


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                **parameters,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "export_model",
        "Export a HuggingFace VLA model (pi0, pi0.5, SmolVLA, GR00T) to ONNX for a target hardware tier. Returns the export directory path.",
        {
            "properties": {
                "model": {"type": "string", "description": "HF model ID or local checkpoint path, e.g. 'lerobot/smolvla_base'"},
                "target": {"type": "string", "enum": _TARGETS, "description": "Hardware tier"},
                "output": {"type": "string", "description": "Output directory (default ./reflex_export)"},
                "precision": {"type": "string", "enum": _PRECISIONS, "description": "Numeric precision"},
                "decomposed": {"type": "boolean", "description": "Use 5-stage decomposed export instead of monolithic. Default false."},
            },
            "required": ["model", "target"],
        },
    ),
    _tool(
        "serve_model",
        "Start the Reflex inference server for a previously exported model. Returns the server URL.",
        {
            "properties": {
                "export_dir": {"type": "string", "description": "Path to the export directory produced by export_model"},
                "port": {"type": "integer", "description": "HTTP port (default 8000)"},
                "host": {"type": "string", "description": "Bind host (default 0.0.0.0)"},
            },
            "required": ["export_dir"],
        },
    ),
    _tool(
        "benchmark",
        "Measure latency/throughput of an exported model on the local machine.",
        {
            "properties": {
                "export_dir": {"type": "string"},
                "iterations": {"type": "integer", "description": "Number of /act calls (default 100)"},
                "batch_size": {"type": "integer", "description": "Concurrent requests (default 1)"},
            },
            "required": ["export_dir"],
        },
    ),
    _tool(
        "evaluate",
        "Run a LIBERO benchmark suite against an exported model. Slow (minutes-hours).",
        {
            "properties": {
                "export_dir": {"type": "string"},
                "suite": {"type": "string", "enum": ["libero_object", "libero_spatial", "libero_goal", "libero_10"]},
                "num_episodes": {"type": "integer", "description": "Episodes per task (default 50)"},
            },
            "required": ["export_dir", "suite"],
        },
    ),
    _tool(
        "list_models",
        "List models available locally (already pulled or exported).",
        {"properties": {}},
    ),
    _tool(
        "pull_model",
        "Download a model checkpoint from HuggingFace into the local registry.",
        {
            "properties": {
                "model": {"type": "string", "description": "HF model ID"},
            },
            "required": ["model"],
        },
    ),
    _tool(
        "model_info",
        "Show metadata + supported targets for a single model in the registry.",
        {
            "properties": {
                "model": {"type": "string"},
            },
            "required": ["model"],
        },
    ),
    _tool(
        "list_targets",
        "List supported hardware targets and their compute/memory budgets.",
        {"properties": {}},
    ),
    _tool(
        "doctor",
        "Run diagnostic checks (ONNX runtime, CUDA, GPU memory, registry health).",
        {"properties": {}},
    ),
    _tool(
        "distill",
        "Distill a teacher VLA into a faster student via SnapFlow (training-free is target; takes hours).",
        {
            "properties": {
                "teacher": {"type": "string", "description": "HF model ID of the teacher"},
                "student_steps": {"type": "integer", "description": "Target denoise steps for student (1, 2, 4)"},
                "output": {"type": "string", "description": "Output checkpoint path"},
            },
            "required": ["teacher", "student_steps"],
        },
    ),
    _tool(
        "finetune",
        "Fine-tune a VLA on a LeRobot dataset (LoRA by default).",
        {
            "properties": {
                "model": {"type": "string"},
                "dataset": {"type": "string", "description": "HF dataset ID or local path"},
                "output": {"type": "string"},
                "lora": {"type": "boolean", "description": "Use LoRA (default true)"},
            },
            "required": ["model", "dataset"],
        },
    ),
    _tool(
        "list_traces",
        "List recent /act traces from the local trace archive (record-replay).",
        {
            "properties": {
                "since": {"type": "string", "description": "Time window, e.g. '7d', '24h', '1h'"},
                "task": {"type": "string", "description": "Filter by task name"},
                "status": {"type": "string", "enum": ["success", "failed", "any"]},
                "limit": {"type": "integer", "description": "Max rows (default 50)"},
            },
        },
    ),
    _tool(
        "replay_trace",
        "Replay a single trace through a model + show diff vs original predictions.",
        {
            "properties": {
                "episode_id": {"type": "string"},
                "export_dir": {"type": "string"},
            },
            "required": ["episode_id", "export_dir"],
        },
    ),
    _tool(
        "show_status",
        "Show running Reflex serve instances + their health.",
        {"properties": {}},
    ),
    _tool(
        "show_config",
        "Show the active Reflex configuration (paths, defaults, registry root).",
        {"properties": {}},
    ),
    _tool(
        "show_version",
        "Show the installed Reflex version + build info.",
        {"properties": {}},
    ),
]


def by_name() -> dict[str, dict[str, Any]]:
    """Return tools indexed by function name."""
    return {t["function"]["name"]: t for t in TOOLS}
