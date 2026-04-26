"""HTTP backend that talks to the FastCrest proxy (Cloudflare Worker)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_PROXY_URL = "https://chat.fastcrest.com"


@dataclass
class ChatBackend:
    """Stateless wrapper around POST /chat on the FastCrest proxy."""

    proxy_url: str = field(default_factory=lambda: os.environ.get("FASTCREST_PROXY_URL", DEFAULT_PROXY_URL))
    client_id: str = field(default_factory=lambda: os.environ.get("FASTCREST_CLIENT_ID", "reflex-cli"))
    timeout_s: float = 60.0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, "client_id": self.client_id}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{self.proxy_url}/chat", json=body)
        if r.status_code == 429:
            raise RateLimitError(r.json().get("message", "rate limit"))
        if r.status_code >= 400:
            raise ProxyError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.proxy_url}/health")
        r.raise_for_status()
        return r.json()


class ProxyError(RuntimeError):
    pass


class RateLimitError(RuntimeError):
    pass
