"""Plain-console chat UI (REPL). Streams tokens live as they arrive."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from reflex.chat.backends import ChatBackend, ProxyError, RateLimitError
from reflex.chat.loop import LoopState

console = Console()


@dataclass
class _UIState:
    """Mutable per-turn UI state. Tracks whether we've started a streaming reply
    so we can print a leading separator on the first token, and a trailing
    newline when the turn finishes."""

    streaming_reply: bool = False
    _stream_started: bool = False

    def on_token(self, text: str) -> None:
        if not self._stream_started:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._stream_started = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_stream(self) -> None:
        if self._stream_started:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
        self._stream_started = False
        self.streaming_reply = False


def _make_event_handler(ui: _UIState):
    def handler(evt: dict[str, Any]) -> None:
        kind = evt["kind"]
        if kind == "turn_start":
            # New round-trip; reset stream tracking.
            ui._stream_started = False
        elif kind == "token":
            ui.on_token(evt["text"])
        elif kind == "tool_start":
            # If we were mid-stream (model wrote some thinking text before the
            # tool call), close that out cleanly first.
            if ui._stream_started:
                ui.end_stream()
            args_preview = evt.get("args") or {}
            console.print(f"[cyan]→ {evt['name']}[/cyan] [dim]{args_preview}[/dim]")
        elif kind == "tool_end":
            result = evt.get("result") or {}
            ec = result.get("exit_code")
            color = "green" if ec == 0 else "red"
            console.print(f"  [{color}]exit_code={ec}[/{color}] [dim]{result.get('command','')}[/dim]")
        elif kind == "final":
            ui.end_stream()
    return handler


def run_repl(proxy_url: str | None = None, dry_run: bool = False, no_stream: bool = False) -> None:
    backend = ChatBackend(proxy_url=proxy_url) if proxy_url else ChatBackend()
    try:
        h = backend.health()
        console.print(f"[dim]connected: {backend.proxy_url} (model={h.get('model','?')})[/dim]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]warning: health check failed ({e}); will try requests anyway[/yellow]")

    ui = _UIState()
    state = LoopState(
        backend=backend,
        on_event=_make_event_handler(ui),
        dry_run=dry_run,
        streaming=not no_stream,
    )
    state.reset()
    console.print("[bold]reflex chat[/bold] — Ctrl+C or 'exit' to quit. /reset to clear.\n")

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
            ui.end_stream()
            console.print(f"[red]rate limit:[/red] {e}")
            continue
        except ProxyError as e:
            ui.end_stream()
            console.print(f"[red]proxy error:[/red] {e}")
            continue

        # Streaming path already printed the reply token-by-token; non-streaming
        # path returns the full reply here, so render it once.
        if no_stream:
            console.print()
            console.print(reply or "_(empty reply)_")
            console.print()
