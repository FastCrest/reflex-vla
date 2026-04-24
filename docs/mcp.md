# MCP integration

Reflex exposes a [Model Context Protocol](https://spec.modelcontextprotocol.io/) server so MCP-compatible agents (Claude Desktop, Cursor, custom) can call a VLA policy as a tool. Additive to the HTTP API — both share the same inference engine.

## Install

```bash
pip install reflex-vla[mcp]
```

Pulls [`fastmcp`](https://github.com/jlowin/fastmcp) >= 3.0 alongside the core dependencies.

## Quick start: Claude Desktop / Cursor integration

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on Linux/Windows:

```json
{
  "mcpServers": {
    "reflex": {
      "command": "reflex",
      "args": [
        "serve",
        "/path/to/your/exported/model/",
        "--mcp",
        "--mcp-transport", "stdio",
        "--embodiment", "franka"
      ]
    }
  }
}
```

Claude Desktop spawns Reflex as a subprocess; stdio transport means no ports, no firewall, no auth. Cursor's MCP config is analogous.

Restart Claude Desktop. `reflex` now appears in the tool picker. Call it like any MCP tool:

```
/reflex act
  instruction: "pick up the red block"
  image_b64: "<base64 PNG from robot camera>"
  state: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  episode_id: "ep-2026-04-24-001"
```

## HTTP transport (for networked agents)

```bash
reflex serve ./my-export/ \
  --mcp \
  --mcp-transport http \
  --mcp-port 8001 \
  --port 8000 \
  --embodiment franka
```

Both MCP (streamable-http on `127.0.0.1:8001`) AND the REST API (`0.0.0.0:8000`) run concurrently. The same `ReflexServer` backs both.

For production HTTP deployment, front MCP with a reverse proxy that handles TLS + auth. MCP doesn't ship its own TLS layer.

## Available tools

| Tool | Inputs | Output |
|---|---|---|
| `act` | `instruction`, `image_b64`, `state`, `episode_id?` | `{actions: [[float]], policy_version, inference_ms}` or `{error: ...}` |
| `health` | — | `{state, model_version, uptime_seconds, cuda_graphs_active}` |
| `models_list` | — | `[{model_id, hf_repo, family, action_dim, size_mb, supported_embodiments, supported_devices, license, description}]` |
| `validate_dataset` | `dataset_path` | Validation report with `pass`/`warn`/`block` decision + per-check details |

## Available resources

| URI | Content |
|---|---|
| `metrics://prometheus` | Current Prometheus metrics in text exposition format (same as the `/metrics` HTTP endpoint) |

## Safety

The `act` tool returns action chunks but does NOT actuate them. Callers are responsible for sending actions to the robot's actuation controller (SO-ARM / Trossen / ROS2). Reflex's `act` is pure inference.

Safety features that DO run inside `act`:

- `ActionGuard` from the loaded embodiment config (joint-limit clamping, velocity caps, torque caps)
- Per-request circuit breaker (`--max-consecutive-crashes`)
- Optional audit log (`--record <dir>`)

Shadow actions, A/B policy routing, and dataset validation run via explicit tools (the caller decides when to invoke).

## Troubleshooting

**"fastmcp not installed"**
```bash
pip install reflex-vla[mcp]
```

**Claude Desktop doesn't list Reflex as a tool**
Verify the `claude_desktop_config.json` path. On macOS, quit + relaunch Claude Desktop fully (cmd-Q, not just close the window).

**"Could not find ReflexServer on the app state"**
This shouldn't happen in released versions — file a bug at github.com/rylinjames/reflex-vla/issues.

**stdio mode blocks the terminal**
By design. stdio owns stdin/stdout for MCP's bidirectional framing. For interactive dev, use `--mcp-transport http` on a separate port.

**MCP + FastAPI on same port**
Not supported — the two transports use different protocols. `--port` is FastAPI; `--mcp-port` is MCP HTTP; they must differ.

## Feature spec

- `features/01_serve/subfeatures/_dx_gaps/mcp-server/mcp-server.md`
- `features/01_serve/subfeatures/_dx_gaps/mcp-server/mcp-server_plan.md`

Pattern source: [InferScope](https://github.com/rylinjames/easyinference) (sibling project at `EasyInference-main/products/inferscope/`). Reflex's MCP server lifts InferScope's FastMCP tool/resource pattern with VLA-specific tool semantics (`act` instead of `chat/completions`).
