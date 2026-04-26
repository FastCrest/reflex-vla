"""Plain-console chat UI (REPL). Used as fallback when Textual isn't installed."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from reflex.chat.backends import ChatBackend, ProxyError, RateLimitError
from reflex.chat.loop import LoopState

console = Console()


def _on_event(evt: dict[str, Any]) -> None:
    kind = evt["kind"]
    if kind == "thinking":
        console.print("[dim]thinking…[/dim]")
    elif kind == "tool_start":
        args_preview = evt.get("args") or {}
        console.print(f"[cyan]→ {evt['name']}[/cyan] [dim]{args_preview}[/dim]")
    elif kind == "tool_end":
        result = evt.get("result") or {}
        ec = result.get("exit_code")
        color = "green" if ec == 0 else "red"
        console.print(f"[{color}]exit_code={ec}[/{color}] [dim]{result.get('command','')}[/dim]")
    elif kind == "final":
        pass  # printed by run_repl


def run_repl(proxy_url: str | None = None, dry_run: bool = False) -> None:
    backend = ChatBackend(proxy_url=proxy_url) if proxy_url else ChatBackend()
    try:
        h = backend.health()
        console.print(f"[dim]connected: {backend.proxy_url} (model={h.get('model','?')})[/dim]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]warning: health check failed ({e}); will try requests anyway[/yellow]")

    state = LoopState(backend=backend, on_event=_on_event, dry_run=dry_run)
    state.reset()
    console.print("[bold]reflex chat[/bold] — Ctrl+C or 'exit' to quit\n")

    while True:
        try:
            user = console.input("[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            return
        if user.lower() == "/reset":
            state.reset()
            console.print("[dim]conversation cleared[/dim]")
            continue

        try:
            reply = state.send(user)
        except RateLimitError as e:
            console.print(f"[red]rate limit:[/red] {e}")
            continue
        except ProxyError as e:
            console.print(f"[red]proxy error:[/red] {e}")
            continue

        console.print()
        console.print(Panel(Markdown(reply or "_(empty reply)_"), border_style="green"))
        console.print()
