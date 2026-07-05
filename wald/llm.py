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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PINNED_DETECTOR_MODEL = "claude-sonnet-4-6"
PINNED_VERIFIER_MODEL = "gpt-4.1-2025-04-14"
MAX_TOKENS = 8192
HTTP_TIMEOUT = 120
RETRY_AFTER_DEFAULT = 20.0


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


def _post_json(req: urllib.request.Request) -> dict:
    """POST returning parsed JSON. Retry once on 429/5xx after Retry-After
    (or ~20s); any other transport failure becomes a BackendError."""
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if attempt == 0 and retryable:
                ra = exc.headers.get("Retry-After") if exc.headers else None
                time.sleep(float(ra) if ra and ra.strip().isdigit() else RETRY_AFTER_DEFAULT)
                continue
            raise BackendError(f"request failed: http {exc.code}: {exc}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendError(f"request failed: {exc}") from exc


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
    usage: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}, init=False
    )

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
        payload = _post_json(req)
        u = payload.get("usage") or {}
        self.usage["input_tokens"] += u.get("input_tokens", 0)
        self.usage["output_tokens"] += u.get("output_tokens", 0)
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
    usage: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}, init=False
    )

    @property
    def gate_eligible(self) -> bool:
        return True

    def _request(self, system: str, user: str) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = json.dumps({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
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
        payload = _post_json(req)
        u = payload.get("usage") or {}
        self.usage["input_tokens"] += u.get("prompt_tokens", 0)
        self.usage["output_tokens"] += u.get("completion_tokens", 0)
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
        try:
            result = subprocess.run(
                ["claude", "-p", system + user],
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"agent session timed out: {exc}") from exc
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
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(envelope))
        os.replace(tmp, path)
        self.provider = envelope["provider"]
        self.model = envelope["model"]
        return response
