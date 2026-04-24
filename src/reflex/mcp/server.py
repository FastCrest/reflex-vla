"""FastMCP server factory bound to a live ReflexServer.

Exposes 4 tools + 1 resource to MCP-compatible agents:

- tool: `act(instruction, image_b64, state, episode_id?)` → action chunk +
  policy_version + inference_ms
- tool: `health()` → {state, model_version, uptime_seconds, cuda_graphs_active}
- tool: `models_list()` → [{id, hf_id, size_gb_fp16, hardware_fit}, ...]
- tool: `validate_dataset(dataset_path)` → {summary, checks: [...]}
- resource: `metrics://prometheus` → current Prometheus exposition text

Usage:

    from reflex.mcp import create_mcp_server

    mcp = create_mcp_server(reflex_server=my_reflex_server)
    mcp.run(transport="stdio")  # or transport="streamable-http", host=..., port=...

Per the verb-noun CLI ADR (2026-04-24), MCP is additive to the existing REST
API — both transports share the same `ReflexServer` inference engine; the MCP
tools forward to the same methods the `/act` / `/healthz` / `/metrics` HTTP
handlers call.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from reflex.runtime.server import ReflexServer

logger = logging.getLogger(__name__)


_MCP_INSTRUCTIONS = """Reflex is a Vision-Language-Action (VLA) policy server for
robotics. This MCP surface exposes Reflex's per-chunk action prediction as agent-
callable tools. Policies are pre-trained (pi0 / pi0.5 / SmolVLA) and served via
ONNX Runtime. Action chunks are semi-fixed-shape (not token-by-token); callers
provide (instruction, image, state) and receive a chunk of actions to actuate at
the robot's control rate.

Available tools:
- act: predict one action chunk from the current observation
- health: server state (ready / warming / degraded / etc.)
- models_list: available pre-built models with hardware fit
- validate_dataset: pre-flight check a LeRobot v3.0 training dataset

Available resources:
- metrics://prometheus: current Prometheus metrics in text exposition format

Safety note: tool `act` returns actions but does NOT actuate them. The caller is
responsible for sending the action chunk to the robot's actuation controller
(SO-ARM / Trossen / ROS2 via the ros2-mcp-bridge feature, also planned).
"""


def create_mcp_server(
    reflex_server: "ReflexServer",
    *,
    name: str = "reflex",
) -> "FastMCP":
    """Build an MCP server bound to a live ReflexServer.

    Args:
        reflex_server: running `ReflexServer` (Pi05DecomposedInference-backed,
            Pi0OnnxServer, or SmolVLAOnnxServer). All tools forward to its
            public methods.
        name: MCP server name; defaults to "reflex" — appears in catalog
            listings.

    Returns:
        FastMCP instance ready to run:
        - `mcp.run(transport="stdio")` for Claude Desktop / Cursor integration
        - `mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)`
          for HTTP-based MCP clients

    Raises:
        ImportError: if `fastmcp` is not installed. Install via
            `pip install reflex-vla[mcp]` or `pip install fastmcp`.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "fastmcp not installed. Install via `pip install reflex-vla[mcp]` "
            "or `pip install fastmcp`."
        ) from exc

    mcp = FastMCP(name, instructions=_MCP_INSTRUCTIONS)
    _startup_ts = time.time()

    @mcp.tool()
    async def act(
        instruction: str,
        image_b64: str,
        state: list[float],
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Predict one action chunk for the current observation.

        Args:
            instruction: natural-language task description (e.g. "pick up the red block").
            image_b64: base64-encoded RGB image from the robot's primary camera.
            state: current proprioceptive state vector (joint positions + gripper).
                Dimensionality must match the loaded model's expected action_dim.
            episode_id: optional client-provided episode id for RTC continuity
                + record-replay tagging. Passing a new episode_id across requests
                triggers RTC's boundary reset; reusing the same id across requests
                within an episode lets RTC carry over chunk guidance.

        Returns:
            On success: {actions: [[float]], policy_version: str, inference_ms: float}.
                `actions` is a list of action vectors (chunk of e.g. 50 × action_dim).
                The caller actuates these sequentially at the robot's control rate.
            On failure: {error: {kind: str, message: str, remediation: str}}.
        """
        try:
            t0 = time.perf_counter()
            result = await reflex_server.predict_from_base64_async(
                image_b64=image_b64,
                instruction=instruction,
                state=state,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            logger.error("mcp.act error: %s: %s", type(exc).__name__, exc)
            return {
                "error": {
                    "kind": type(exc).__name__,
                    "message": str(exc),
                    "remediation": (
                        "Inspect server logs for the full traceback. Verify "
                        "(image_b64, instruction, state) match the loaded model's "
                        "expected shapes via the health tool or `reflex doctor`."
                    ),
                }
            }

        if isinstance(result, dict) and "error" in result:
            return {"error": {"kind": "DecodeError", "message": result["error"],
                              "remediation": "Check image_b64 is a valid base64-encoded "
                                             "RGB image (PNG or JPEG)."}}

        return {
            "actions": result.get("actions") if isinstance(result, dict) else None,
            "policy_version": getattr(reflex_server, "export_dir", "unknown"),
            "inference_ms": round(elapsed_ms, 2),
        }

    @mcp.tool()
    async def health() -> dict[str, Any]:
        """Server health and readiness.

        Returns:
            {state: str, model_version: str, uptime_seconds: float,
             cuda_graphs_active: bool | None}

            `state` is one of:
            - "initializing" — process just started
            - "loading" — model weights being loaded into memory
            - "warming" — first forward pass running (also triggers cuda-graph
              capture if --cuda-graphs was set)
            - "ready" — accepting /act requests
            - "warmup_failed" — init failed; server won't accept requests
            - "degraded" — consecutive crashes exceeded circuit-breaker threshold
        """
        state = getattr(reflex_server, "health_state", "initializing")
        cuda_graphs = getattr(reflex_server, "_cuda_graphs_enabled", None)
        return {
            "state": state,
            "model_version": str(getattr(reflex_server, "export_dir", "unknown")),
            "uptime_seconds": round(time.time() - _startup_ts, 2),
            "cuda_graphs_active": cuda_graphs,
        }

    @mcp.tool()
    async def models_list() -> list[dict[str, Any]]:
        """List pre-built models in the Reflex registry.

        Returns:
            List of model entries with id, hf_id, family, device fit, and
            published benchmarks. Curated set; each entry verified against
            Reflex parity tests.
        """
        try:
            from reflex.registry import filter_models
            entries = filter_models()
        except ImportError:
            return [{"error": {"kind": "ImportError",
                               "message": "reflex.registry unavailable",
                               "remediation": "reinstall reflex with the default extras."}}]

        return [
            {
                "model_id": e.model_id,
                "hf_repo": e.hf_repo,
                "family": e.family,
                "action_dim": e.action_dim,
                "size_mb": e.size_mb,
                "supported_embodiments": list(e.supported_embodiments),
                "supported_devices": list(e.supported_devices),
                "license": e.license,
                "description": e.description,
            }
            for e in entries
        ]

    @mcp.tool()
    async def validate_dataset(dataset_path: str) -> dict[str, Any]:
        """Pre-flight check a LeRobot v3.0 training dataset.

        Runs the 8 falsifiable checks from `reflex validate-dataset`:
        schema completeness, shape consistency, action-finite, embodiment match,
        episode count, timing monotonicity, etc.

        Args:
            dataset_path: filesystem path to the LeRobot dataset root.

        Returns:
            {schema_version, summary: {pass, fail, warn, skip}, decision, checks: [...]}
            Decision is one of "proceed" | "warn" | "block".
        """
        try:
            from reflex.validation import (
                DatasetContext,
                format_json,
                overall_decision,
                run_all_checks,
            )
            from pathlib import Path
            import json
        except ImportError:
            return {"error": {"kind": "ImportError",
                              "message": "reflex.validation unavailable",
                              "remediation": "reinstall reflex with the default extras."}}

        try:
            root = Path(dataset_path)
            if not root.exists():
                return {"error": {
                    "kind": "FileNotFoundError",
                    "message": f"Dataset path does not exist: {dataset_path}",
                    "remediation": "Check that dataset_path exists and is a LeRobot v3.0 dataset.",
                }}
            ctx = DatasetContext(root=root)
            results = run_all_checks(ctx)
            decision = overall_decision(results)
            payload_text = format_json(results)
            return json.loads(payload_text) | {
                "decision": decision.value if hasattr(decision, "value") else str(decision),
            }
        except Exception as exc:
            logger.error("mcp.validate_dataset error: %s: %s", type(exc).__name__, exc)
            return {"error": {"kind": type(exc).__name__,
                              "message": str(exc),
                              "remediation": "Run `reflex validate-dataset <path>` from the CLI for full diagnostics."}}

    @mcp.resource("metrics://prometheus")
    async def prometheus_metrics() -> str:
        """Current Prometheus metrics in text exposition format.

        Same content as the `/metrics` HTTP endpoint. Agents can scrape this
        to monitor latency, cache hit rate, cuda-graph capture status, and
        SLO violations without needing a separate HTTP port.
        """
        try:
            from reflex.observability.prometheus import render_metrics
        except ImportError:
            return "# reflex.observability.prometheus unavailable\n"
        return render_metrics().decode("utf-8")

    return mcp
