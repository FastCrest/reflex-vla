"""Agent loop: send messages, run tool calls, loop until LLM stops calling tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from reflex.chat.backends import ChatBackend
from reflex.chat.executor import execute, format_tool_result
from reflex.chat.schema import TOOLS

SYSTEM_PROMPT = """You are the Reflex assistant. Reflex is a CLI that exports, serves, and benchmarks vision-language-action (VLA) models on edge hardware.

You have tools that wrap the `reflex` CLI. Use them to act on the user's behalf instead of describing commands. Pick the smallest tool that answers the question. Don't ask for confirmation before read-only tools (list_models, doctor, list_traces). For destructive or long-running tools (export_model, distill, finetune, evaluate), confirm intent first if the user's request is ambiguous about scope.

When a tool returns a non-zero exit code, read its stderr, explain what went wrong in one sentence, and suggest a concrete next action. Don't fabricate tool output."""


@dataclass
class LoopState:
    backend: ChatBackend
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_calls: int = 16
    on_event: Callable[[dict[str, Any]], None] | None = None
    dry_run: bool = False

    def emit(self, kind: str, **payload: Any) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, **payload})

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, user_text: str) -> str:
        if not self.messages:
            self.reset()
        self.messages.append({"role": "user", "content": user_text})
        return self._run_loop()

    def _run_loop(self) -> str:
        for _ in range(self.max_tool_calls):
            self.emit("thinking")
            resp = self.backend.chat(self.messages, tools=TOOLS, tool_choice="auto")
            choice = resp["choices"][0]
            msg = choice["message"]
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = msg.get("content") or ""
                self.emit("final", content=content)
                return content

            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.emit("tool_start", name=name, args=args)
                result = execute(name, args, dry_run=self.dry_run)
                self.emit("tool_end", name=name, result=result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": format_tool_result(name, result),
                })

        # Hit the cap. Ask LLM to wrap up without more tool calls.
        self.messages.append({"role": "user", "content": "[system] tool-call cap reached; summarize results and stop calling tools."})
        resp = self.backend.chat(self.messages, tools=None)
        content = resp["choices"][0]["message"].get("content") or ""
        self.emit("final", content=content)
        return content
