"""Agent loop: send messages, run tool calls, loop until LLM stops calling tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from reflex.chat.backends import ChatBackend, assemble_stream
from reflex.chat.executor import execute, format_tool_result
from reflex.chat.schema import TOOLS

# Hallucination guard (#13): the model previously paraphrased version numbers from
# tool output (cited "torch 2.10.0" when the actual was 2.11.0). The verbatim rule
# below is the cheap mitigation — the LLM has the right data in context, it just
# needs explicit instruction to copy specific values rather than summarize them.
SYSTEM_PROMPT = """You are the Reflex assistant. Reflex is a CLI that exports, serves, and benchmarks vision-language-action (VLA) models on edge hardware.

You have tools that wrap the `reflex` CLI. Use them to act on the user's behalf instead of describing commands. Pick the smallest tool that answers the question. Don't ask for confirmation before read-only tools (list_models, doctor, list_traces). For destructive or long-running tools (export_model, distill, finetune, evaluate), confirm intent first if the user's request is ambiguous about scope.

When a tool returns a non-zero exit code, read its stderr, explain what went wrong in one sentence, and suggest a concrete next action. Don't fabricate tool output.

CRITICAL — verbatim values: when reporting specific values from tool output (version numbers, file paths, IDs, sizes, error messages, model names, latency numbers), copy them exactly as they appear. Do not paraphrase, round, or "fix" them. If a tool says `torch 2.11.0`, write `torch 2.11.0` — never `torch 2.10` or `torch 2.11`. If you didn't run a tool that returned the value, say "I don't have that information" instead of guessing."""


@dataclass
class LoopState:
    backend: ChatBackend
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_calls: int = 16
    on_event: Callable[[dict[str, Any]], None] | None = None
    dry_run: bool = False
    streaming: bool = True  # set False for tests that need deterministic single-shot

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

    def _one_turn(self, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """One LLM round-trip. Returns the assistant message dict.

        When streaming=True, emits per-token `token` events so the UI can render
        live. Tool-call deltas don't emit tokens — they accumulate silently and
        the assembled tool calls fire `tool_start`/`tool_end` from the caller.
        """
        if self.streaming:
            chunks = self.backend.chat_stream(self.messages, tools=tools, tool_choice="auto" if tools else "none")
            return assemble_stream(
                chunks,
                on_token=lambda t: self.emit("token", text=t),
            )
        resp = self.backend.chat(self.messages, tools=tools, tool_choice="auto" if tools else "none")
        return resp["choices"][0]["message"]

    def _run_loop(self) -> str:
        for _ in range(self.max_tool_calls):
            self.emit("turn_start")
            msg = self._one_turn(tools=TOOLS)
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
        msg = self._one_turn(tools=None)
        content = msg.get("content") or ""
        self.emit("final", content=content)
        return content
