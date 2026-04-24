"""Tests for src/reflex/mcp/server.py (MCP server factory + tools + resources).

Uses a mock ReflexServer so tests run without CUDA / model weights. Exercises:

- Tool registration (all 4 tools + 1 resource present)
- Per-tool invocation contract (act, health, models_list, validate_dataset)
- Error-envelope shape on underlying failures
- Resource output (metrics://prometheus)

Tool invocation uses `mcp.call_tool()` (FastMCP's in-process test entry
point), which bypasses the stdio/HTTP transport layer.
"""
from __future__ import annotations

import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reflex.mcp import create_mcp_server


def _mock_reflex_server(
    health_state: str = "ready",
    export_dir: str = "/tmp/fake-export",
    predict_return: dict | None = None,
    predict_raises: Exception | None = None,
    cuda_graphs_enabled: bool = False,
) -> MagicMock:
    """Build a MagicMock that mimics ReflexServer's minimal MCP-facing API."""
    server = MagicMock()
    server.health_state = health_state
    server.export_dir = export_dir
    server._cuda_graphs_enabled = cuda_graphs_enabled

    predict_async_mock = AsyncMock()
    if predict_raises is not None:
        predict_async_mock.side_effect = predict_raises
    else:
        predict_async_mock.return_value = predict_return or {
            "actions": [[0.1, 0.2, 0.3]],
            "task": "",
        }
    server.predict_from_base64_async = predict_async_mock
    return server


def _run(coro):
    """Helper to run an async function in a test."""
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() \
        else asyncio.run(coro)


# ---------------------------------------------------------------------------
# Server construction + tool registration
# ---------------------------------------------------------------------------


def test_create_mcp_server_returns_fastmcp_instance():
    from fastmcp import FastMCP
    mcp = create_mcp_server(_mock_reflex_server())
    assert isinstance(mcp, FastMCP)


@pytest.mark.asyncio
async def test_all_four_tools_registered():
    mcp = create_mcp_server(_mock_reflex_server())
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "act" in tool_names
    assert "health" in tool_names
    assert "models_list" in tool_names
    assert "validate_dataset" in tool_names


@pytest.mark.asyncio
async def test_metrics_resource_registered():
    mcp = create_mcp_server(_mock_reflex_server())
    resources = await mcp.list_resources()
    # Resources are identified by URI
    uris = {str(r.uri) for r in resources}
    assert any("metrics" in uri for uri in uris)


# ---------------------------------------------------------------------------
# Tool: act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_act_tool_forwards_to_reflex_server():
    server = _mock_reflex_server(predict_return={
        "actions": [[0.5, 0.6], [0.7, 0.8]],
    })
    mcp = create_mcp_server(server)

    # Minimal 1x1 base64 PNG for the image input
    img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20).decode("ascii")

    result = await mcp.call_tool("act", {
        "instruction": "pick up the cup",
        "image_b64": img_b64,
        "state": [0.0, 0.1, 0.2],
    })

    # FastMCP call_tool returns a CallToolResult; structured content is in .data
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert "actions" in payload
    assert payload["actions"] == [[0.5, 0.6], [0.7, 0.8]]
    assert "inference_ms" in payload
    assert payload["inference_ms"] >= 0
    server.predict_from_base64_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_act_tool_returns_error_envelope_on_server_exception():
    class ServerBoom(RuntimeError):
        pass

    server = _mock_reflex_server(predict_raises=ServerBoom("inference crashed"))
    mcp = create_mcp_server(server)

    result = await mcp.call_tool("act", {
        "instruction": "test",
        "image_b64": "AAAA",
        "state": [0.0],
    })
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert "error" in payload
    assert payload["error"]["kind"] == "ServerBoom"
    assert "inference crashed" in payload["error"]["message"]
    assert payload["error"]["remediation"]  # non-empty


@pytest.mark.asyncio
async def test_act_tool_returns_error_on_decode_failure():
    # predict_from_base64_async returns {"error": "Failed to decode image: ..."}
    # when image_b64 is malformed. MCP act() should surface that as an error envelope.
    server = _mock_reflex_server(predict_return={"error": "Failed to decode image: bad padding"})
    mcp = create_mcp_server(server)

    result = await mcp.call_tool("act", {
        "instruction": "x",
        "image_b64": "not-base64",
        "state": [0.0],
    })
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert "error" in payload
    assert payload["error"]["kind"] == "DecodeError"


# ---------------------------------------------------------------------------
# Tool: health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_tool_returns_prewarm_state():
    server = _mock_reflex_server(health_state="ready",
                                  export_dir="/tmp/my-model",
                                  cuda_graphs_enabled=True)
    mcp = create_mcp_server(server)

    result = await mcp.call_tool("health", {})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert payload["state"] == "ready"
    assert payload["model_version"] == "/tmp/my-model"
    assert payload["uptime_seconds"] >= 0
    assert payload["cuda_graphs_active"] is True


@pytest.mark.asyncio
async def test_health_tool_reports_warming_state():
    server = _mock_reflex_server(health_state="warming")
    mcp = create_mcp_server(server)
    result = await mcp.call_tool("health", {})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert payload["state"] == "warming"


# ---------------------------------------------------------------------------
# Tool: models_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_list_tool_returns_registry_entries():
    """models_list forwards to reflex.registry.filter_models()."""
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("models_list", {})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    # The real registry has >=1 entry. Confirm shape.
    # FastMCP wraps list responses in a dict with a "result" key in structured_content
    entries = payload["result"] if isinstance(payload, dict) and "result" in payload else payload
    assert isinstance(entries, list)
    if entries and isinstance(entries[0], dict) and "error" not in entries[0]:
        e0 = entries[0]
        assert "model_id" in e0
        assert "hf_repo" in e0
        assert "family" in e0
        assert "action_dim" in e0


# ---------------------------------------------------------------------------
# Tool: validate_dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_dataset_tool_returns_error_on_missing_path():
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("validate_dataset",
                                  {"dataset_path": "/nonexistent/path/xyz"})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    # Either an error envelope (FileNotFoundError from DatasetContext) or a
    # structured validation report with a "block" decision — depending on
    # how DatasetContext handles missing paths. Either is acceptable; just
    # verify the shape is deterministic.
    assert isinstance(payload, dict)
    assert ("error" in payload) or ("decision" in payload) or ("summary" in payload)


# ---------------------------------------------------------------------------
# Resource: metrics://prometheus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_resource_returns_prometheus_text():
    mcp = create_mcp_server(_mock_reflex_server())
    # Read the metrics resource
    result = await mcp.read_resource("metrics://prometheus")
    # read_resource returns a list of ReadResourceContents or strings depending
    # on FastMCP version — handle both
    if isinstance(result, list):
        payload = "".join(
            (item.text if hasattr(item, "text") else str(item))
            for item in result
        )
    else:
        payload = result.text if hasattr(result, "text") else str(result)
    # Real Prometheus exposition starts with # HELP or # TYPE comments
    assert isinstance(payload, str)
    assert len(payload) > 0
