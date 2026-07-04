"""LLM backend seam: one `complete` method, three implementations.

Every response carries its `kind` (api | replay | agent) and a computed
`gate_eligible`; gate evidence requires kind==api end to end, and cached
replays or agent sessions can never satisfy that — the property is derived,
never asserted by a caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PINNED_DETECTOR_MODEL = "claude-sonnet-4-6"
PINNED_VERIFIER_MODEL = "gpt-4.1-2025-04-14"
MAX_TOKENS = 8192


class BackendError(Exception):
    """Response could not be parsed as JSON after one retry, was truncated, or the API call failed."""


class Backend(Protocol):
    provider: str
    model: str
    kind: str  # "api" | "replay" | "agent"

    @property
    def gate_eligible(self) -> bool: ...

    def complete(self, system: str, user: str, schema: dict | None = None) -> dict: ...


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _with_schema(user: str, schema: dict | None) -> str:
    if schema is None:
        return user
    return f"{user}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}"


def _call_with_retry(request_fn, system: str, user: str) -> dict:
    text = request_fn(system, user)
    try:
        return json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        retry_user = f"{user}\n\n{exc}\nReturn only valid JSON."
        text = request_fn(system, retry_user)
        try:
            return json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc2:
            raise BackendError(f"could not parse response as JSON: {exc2}") from exc2


@dataclass
class AnthropicBackend:
    model: str = PINNED_DETECTOR_MODEL
    provider: str = field(default="anthropic", init=False)
    kind: str = field(default="api", init=False)

    @property
    def gate_eligible(self) -> bool:
        return True

    def _request(self, system: str, user: str) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        body = json.dumps({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendError(f"anthropic request failed: {exc}") from exc
        if payload.get("stop_reason") == "max_tokens":
            raise BackendError("anthropic response truncated at max_tokens")
        try:
            return payload["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise BackendError(f"anthropic response envelope malformed: {exc}") from exc

    def complete(self, system: str, user: str, schema: dict | None = None) -> dict:
        return _call_with_retry(self._request, system, _with_schema(user, schema))


@dataclass
class OpenAIBackend:
    model: str = PINNED_VERIFIER_MODEL
    provider: str = field(default="openai", init=False)
    kind: str = field(default="api", init=False)

    @property
    def gate_eligible(self) -> bool:
        return True

    def _request(self, system: str, user: str) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendError(f"openai request failed: {exc}") from exc
        try:
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                raise BackendError("openai response truncated at max_tokens")
            return choice["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise BackendError(f"openai response envelope malformed: {exc}") from exc

    def complete(self, system: str, user: str, schema: dict | None = None) -> dict:
        return _call_with_retry(self._request, system, _with_schema(user, schema))


@dataclass
class AgentBackend:
    provider: str = field(default="agent", init=False)
    model: str = field(default="session", init=False)
    kind: str = field(default="agent", init=False)

    @property
    def gate_eligible(self) -> bool:
        return False

    def _request(self, system: str, user: str) -> str:
        result = subprocess.run(["claude", "-p", system + user], capture_output=True, text=True)
        return result.stdout

    def complete(self, system: str, user: str, schema: dict | None = None) -> dict:
        return _call_with_retry(self._request, system, _with_schema(user, schema))


@dataclass
class ReplayBackend:
    dir: Path
    inner: Backend | None = None

    def __post_init__(self):
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kind = "replay"
        self.served_from_disk = 0
        self.provider = self.inner.provider if self.inner is not None else ""
        self.model = self.inner.model if self.inner is not None else ""

    @property
    def gate_eligible(self) -> bool:
        return self.inner is not None and self.inner.gate_eligible and self.served_from_disk == 0

    def complete(self, system: str, user: str, schema: dict | None = None) -> dict:
        key = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:16]
        path = self.dir / f"{key}.json"
        if path.exists():
            envelope = json.loads(path.read_text())
            self.served_from_disk += 1
            self.provider = envelope["provider"]
            self.model = envelope["model"]
            return envelope["response"]
        if self.inner is None:
            raise KeyError(key)
        response = self.inner.complete(system, user, schema)
        envelope = {
            "provider": self.inner.provider,
            "model": self.inner.model,
            "kind": self.inner.kind,
            "response": response,
        }
        path.write_text(json.dumps(envelope))
        self.provider = envelope["provider"]
        self.model = envelope["model"]
        return response
