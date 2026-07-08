import email.message
import hashlib
import json
import urllib.error
from pathlib import Path

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
    calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")
    assert len(calls) == 2  # transport failure retried once before giving up


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
    calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")
    assert len(calls) == 2  # transport failure retried once before giving up


def test_openai_malformed_envelope_wrapped_in_backend_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    with pytest.raises(llm.BackendError):
        backend.complete("sys", "user")


def test_openai_sends_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies = []

    def fake_urlopen(req, *a, **kw):
        bodies.append(json.loads(req.data))
        return FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    llm.OpenAIBackend().complete("sys", "user")
    assert bodies[0]["max_tokens"] == 8192


def test_retry_on_429_honors_retry_after_then_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    hdrs = email.message.Message()
    hdrs["Retry-After"] = "3"
    calls = []
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", hdrs, None)
        return FakeResponse({"content": [{"text": '{"ok": 1}'}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert llm.AnthropicBackend().complete("sys", "user") == {"ok": 1}
    assert len(calls) == 2
    assert slept == [3.0]


def test_retry_exhausted_on_persistent_5xx_raises_backend_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        raise urllib.error.HTTPError("http://x", 529, "Overloaded", None, None)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(llm.BackendError):
        llm.OpenAIBackend().complete("sys", "user")
    assert len(calls) == 2  # one retry, then give up


def test_anthropic_accumulates_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse(
            {"content": [{"text": "{}"}], "usage": {"input_tokens": 10, "output_tokens": 4}}
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.AnthropicBackend()
    backend.complete("sys", "user")
    backend.complete("sys", "user")
    assert backend.usage == {"input_tokens": 20, "output_tokens": 8}


def test_openai_accumulates_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_urlopen(req, *a, **kw):
        return FakeResponse(
            {"choices": [{"message": {"content": "{}"}}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    backend = llm.OpenAIBackend()
    backend.complete("sys", "user")
    assert backend.usage == {"input_tokens": 7, "output_tokens": 2}


def test_transport_error_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.URLError("connection refused")
        return FakeResponse({"content": [{"text": '{"ok": 1}'}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert llm.AnthropicBackend().complete("sys", "user") == {"ok": 1}
    assert len(calls) == 2
    assert slept == [llm.TRANSPORT_RETRY_PAUSE]  # ~5s pause before the single retry


def test_timeout_error_retries_once(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fake_urlopen(req, *a, **kw):
        calls.append(req)
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return FakeResponse({"choices": [{"message": {"content": '{"ok": 2}'}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert llm.OpenAIBackend().complete("sys", "user") == {"ok": 2}
    assert len(calls) == 2


class FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_agent_backend_surfaces_stderr_on_nonzero_exit(monkeypatch):
    stderr = "fatal: " + "x" * 1000

    def fake_run(cmd, *a, **kw):
        return FakeCompleted(2, stdout="", stderr=stderr)

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(llm.BackendError) as excinfo:
        llm.AgentBackend().complete("sys", "user")
    msg = str(excinfo.value)
    assert "exit 2" in msg
    assert stderr[:500] in msg  # first 500 stderr chars surfaced
    assert stderr not in msg    # but the full stderr is truncated


def test_agent_backend_passes_prompt_via_stdin_not_argv(monkeypatch):
    captured = {}
    envelope = json.dumps({
        "type": "result", "subtype": "success", "is_error": False, "result": '{"ok": 1}',
    })

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        return FakeCompleted(0, stdout=envelope)

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    result = llm.AgentBackend().complete("SYSTEM", "USERPROMPT")
    assert result == {"ok": 1}
    assert captured["cmd"][:3] == ["claude", "-p", "--output-format"]
    assert "--system-prompt" in captured["cmd"]
    assert "--strict-mcp-config" in captured["cmd"]
    # prompt kept out of argv: SYSTEM only appears as the --system-prompt value
    assert captured["cmd"][captured["cmd"].index("--system-prompt") + 1] == "SYSTEM"
    assert "USERPROMPT" not in captured["cmd"]
    assert captured["input"] == "USERPROMPT"


def test_agent_backend_raises_on_error_envelope(monkeypatch):
    envelope = json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True})

    def fake_run(cmd, *a, **kw):
        return FakeCompleted(0, stdout=envelope)

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(llm.BackendError, match="agent session turn failed"):
        llm.AgentBackend().complete("sys", "user")


def test_codex_backend_reads_answer_from_output_file(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text('{"verdict": "supported", "reason": "x"}')
        return FakeCompleted(0)

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    result = llm.CodexBackend().complete("SYSTEM", "USERPROMPT")
    assert result == {"verdict": "supported", "reason": "x"}
    assert captured["cmd"][:2] == ["codex", "exec"]
    assert "-a" not in captured["cmd"]
    assert "--ask-for-approval" not in captured["cmd"]
    assert captured["cmd"][-1] == "-"  # prompt read from stdin, not argv
    assert "SYSTEM" in captured["input"] and "USERPROMPT" in captured["input"]


def test_codex_backend_surfaces_stderr_on_nonzero_exit(monkeypatch):
    stderr = "fatal: " + "x" * 1000

    def fake_run(cmd, *a, **kw):
        return FakeCompleted(2, stdout="", stderr=stderr)

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(llm.BackendError) as excinfo:
        llm.CodexBackend().complete("sys", "user")
    msg = str(excinfo.value)
    assert "exit 2" in msg
    assert stderr[:500] in msg
    assert stderr not in msg


def test_replay_cache_write_is_atomic(tmp_path):
    inner = StubBackend({"ok": True})
    llm.ReplayBackend(tmp_path, inner=inner).complete("sys", "user")
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".json"  # no leftover .tmp
    assert json.loads(files[0].read_text())["response"] == {"ok": True}
