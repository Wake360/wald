import hashlib
import json
import urllib.error

import pytest

from wald import llm


class StubBackend:
    """Protocol-conforming backend with canned responses; no network."""

    provider = "stub-provider"
    model = "stub-model-1"
    kind = "api"

    def __init__(self, response, gate_eligible=True):
        self._response = response
        self._gate_eligible = gate_eligible
        self.calls = 0

    @property
    def gate_eligible(self):
        return self._gate_eligible

    def complete(self, system, user, schema=None):
        self.calls += 1
        return self._response


class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_record_then_replay_round_trip(tmp_path):
    inner = StubBackend({"ok": True})
    recorder = llm.ReplayBackend(tmp_path, inner=inner)
    first = recorder.complete("sys", "user")
    assert first == {"ok": True}
    assert inner.calls == 1
    assert recorder.served_from_disk == 0

    replayer = llm.ReplayBackend(tmp_path, inner=inner)
    second = replayer.complete("sys", "user")
    assert second == {"ok": True}
    assert inner.calls == 1  # not called again: served from disk
    assert replayer.served_from_disk == 1
    assert replayer.provider == inner.provider
    assert replayer.model == inner.model


def test_replay_hit_flips_gate_eligible_false(tmp_path):
    inner = StubBackend({"ok": True})
    backend = llm.ReplayBackend(tmp_path, inner=inner)
    backend.complete("sys", "user")  # miss: fresh from inner
    assert backend.gate_eligible is True

    backend.complete("sys", "user")  # hit: served from disk
    assert backend.served_from_disk == 1
    assert backend.gate_eligible is False


def test_miss_without_inner_raises_keyerror_with_hash(tmp_path):
    backend = llm.ReplayBackend(tmp_path)
    with pytest.raises(KeyError) as excinfo:
        backend.complete("sys", "user")
    expected_key = hashlib.sha256(("sys" + "\x00" + "user").encode()).hexdigest()[:16]
    assert excinfo.value.args[0] == expected_key


def test_api_backend_raises_runtime_error_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = llm.AnthropicBackend()  # constructor never raises
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        backend.complete("sys", "user")


def test_fence_stripping(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse({"content": [{"text": '```json\n{"a": 1}\n```'}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    assert backend.complete("sys", "user") == {"a": 1}


def test_backend_error_after_retry_exhaustion(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        return FakeResponse({"content": [{"text": "not json"}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")
    assert len(calls) == 2  # initial attempt + one retry


def test_anthropic_uses_raised_max_tokens_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    bodies = []

    def fake_urlopen(req, *a, **kw):
        bodies.append(json.loads(req.data))
        return FakeResponse({"content": [{"text": "{}"}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    backend.complete("sys", "user")
    assert bodies[0]["max_tokens"] == 8192


def test_anthropic_truncated_response_raises_backend_error_without_parsing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        return FakeResponse({"stop_reason": "max_tokens", "content": [{"text": "{cut off"}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    with pytest.raises(llm.BackendError, match="truncat"):
        backend.complete("sys", "user")
    assert len(calls) == 1  # truncation fails closed, no retry attempted


def test_anthropic_network_error_wrapped_in_backend_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")


def test_anthropic_malformed_envelope_wrapped_in_backend_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")


def test_openai_backend_raises_runtime_error_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = llm.OpenAIBackend()  # constructor never raises
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        backend.complete("sys", "user")


def test_openai_request_construction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def fake_urlopen(req, *a, **kw):
        requests.append(req)
        return FakeResponse({"choices": [{"message": {"content": '{"ok": 1}'}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    result = backend.complete("sys", "user")

    assert result == {"ok": 1}
    req = requests[0]
    assert req.full_url == "https://api.openai.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer test-key"
    body = json.loads(req.data)
    assert body["model"] == llm.PINNED_VERIFIER_MODEL
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_openai_fence_stripping(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse({"choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    assert backend.complete("sys", "user") == {"a": 1}


def test_openai_truncated_response_raises_backend_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        return FakeResponse(
            {"choices": [{"finish_reason": "length", "message": {"content": "{cut off"}}]}
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    with pytest.raises(llm.BackendError, match="truncat"):
        backend.complete("sys", "user")
    assert len(calls) == 1


def test_openai_network_error_wrapped_in_backend_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")


def test_openai_malformed_envelope_wrapped_in_backend_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")
