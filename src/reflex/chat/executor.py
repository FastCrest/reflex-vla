"""Translate LLM tool calls into local Reflex CLI invocations.

We shell out to `reflex <subcommand>` (the same binary the user installed) so the
chat loop never re-implements logic that already lives in the CLI. Subprocess output
goes back to the LLM as the tool result.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from typing import Any

# Map tool name → callable that returns argv (list[str]) for `reflex <subcommand>`.
# Each builder validates required args and ignores unknown keys.

OutputCap = 8000  # truncate stdout so we don't blow context


def _flag(args: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(f"--{key}")
        return
    args.extend([f"--{key}", str(value)])


def _build_export(p: dict[str, Any]) -> list[str]:
    args = ["export", str(p["model"]), "--target", str(p["target"])]
    _flag(args, "output", p.get("output"))
    _flag(args, "precision", p.get("precision"))
    if p.get("decomposed") is True:
        args.append("--decomposed")
    return args


def _build_serve(p: dict[str, Any]) -> list[str]:
    args = ["serve", str(p["export_dir"])]
    _flag(args, "port", p.get("port"))
    _flag(args, "host", p.get("host"))
    return args


def _build_bench(p: dict[str, Any]) -> list[str]:
    args = ["bench", str(p["export_dir"])]
    _flag(args, "iterations", p.get("iterations"))
    _flag(args, "batch-size", p.get("batch_size"))
    return args


def _build_eval(p: dict[str, Any]) -> list[str]:
    args = ["eval", str(p["export_dir"]), "--suite", str(p["suite"])]
    _flag(args, "num-episodes", p.get("num_episodes"))
    return args


def _build_pull(p: dict[str, Any]) -> list[str]:
    return ["models", "pull", str(p["model"])]


def _build_model_info(p: dict[str, Any]) -> list[str]:
    return ["models", "info", str(p["model"])]


def _build_distill(p: dict[str, Any]) -> list[str]:
    args = ["distill", str(p["teacher"]), "--student-steps", str(p["student_steps"])]
    _flag(args, "output", p.get("output"))
    return args


def _build_finetune(p: dict[str, Any]) -> list[str]:
    args = ["finetune", str(p["model"]), str(p["dataset"])]
    _flag(args, "output", p.get("output"))
    if p.get("lora") is False:
        args.append("--no-lora")
    return args


def _build_traces(p: dict[str, Any]) -> list[str]:
    args = ["traces", "query"]
    _flag(args, "since", p.get("since"))
    _flag(args, "task", p.get("task"))
    if p.get("status") and p["status"] != "any":
        _flag(args, "status", p["status"])
    _flag(args, "limit", p.get("limit"))
    return args


def _build_replay(p: dict[str, Any]) -> list[str]:
    return ["replay", str(p["episode_id"]), "--export-dir", str(p["export_dir"])]


# Builders that take no args — just static argv.
_STATIC = {
    "list_models": ["models", "list"],
    "list_targets": ["targets"],
    "doctor": ["doctor"],
    "show_status": ["status"],
    "show_config": ["config", "show"],
    "show_version": ["--version"],
}

_BUILDERS = {
    "export_model": _build_export,
    "serve_model": _build_serve,
    "benchmark": _build_bench,
    "evaluate": _build_eval,
    "pull_model": _build_pull,
    "model_info": _build_model_info,
    "distill": _build_distill,
    "finetune": _build_finetune,
    "list_traces": _build_traces,
    "replay_trace": _build_replay,
}


def _argv_for(name: str, params: dict[str, Any]) -> list[str]:
    if name in _STATIC:
        return list(_STATIC[name])
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"unknown tool: {name}")
    return builder(params)


def execute(name: str, params: dict[str, Any], reflex_bin: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Run a tool. Returns dict with stdout, stderr, exit_code, command, dry_run."""
    binary = reflex_bin or shutil.which("reflex") or "reflex"
    argv = [binary] + _argv_for(name, params)
    cmd_str = " ".join(shlex.quote(a) for a in argv)

    if dry_run:
        return {"command": cmd_str, "dry_run": True, "stdout": "", "stderr": "", "exit_code": 0}

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"command": cmd_str, "stdout": "", "stderr": "timeout after 600s", "exit_code": 124}
    except FileNotFoundError as e:
        return {"command": cmd_str, "stdout": "", "stderr": str(e), "exit_code": 127}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if len(stdout) > OutputCap:
        stdout = stdout[:OutputCap] + "\n... [truncated]"
    if len(stderr) > OutputCap:
        stderr = stderr[:OutputCap] + "\n... [truncated]"

    return {
        "command": cmd_str,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": proc.returncode,
    }


def format_tool_result(name: str, result: dict[str, Any]) -> str:
    """Compact tool result for LLM consumption."""
    parts = [f"$ {result['command']}"]
    if result.get("dry_run"):
        parts.append("(dry-run, not executed)")
        return "\n".join(parts)
    parts.append(f"exit_code={result['exit_code']}")
    if result["stdout"]:
        parts.append(f"--- stdout ---\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"--- stderr ---\n{result['stderr']}")
    return "\n".join(parts)
