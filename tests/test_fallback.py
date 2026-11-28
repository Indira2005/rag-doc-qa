import asyncio

import groq
import httpx
import pytest

from rag import config, generate

HITS = [{"source": "a.pdf", "page": 1, "section": "", "text": "x", "score": 1.0}]


def rate_limit():
    request = httpx.Request("POST", "http://test")
    return groq.RateLimitError("limit", response=httpx.Response(429, request=request), body=None)


def fake(calls, fail_after=None):
    async def _stream(provider, messages, max_retries=2):
        calls.append(provider)
        if provider == "groq":
            if fail_after:
                yield "partial"
            raise rate_limit()
        yield "backup answer"
    return _stream


def run(**kwargs):
    async def collect():
        return [t async for t in generate.stream_answer("q", HITS, **kwargs)]
    return asyncio.run(collect())


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "LLM_FALLBACK", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test")


def test_uses_backup_when_groq_is_rate_limited(on, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "_stream", fake(calls))
    assert run() == ["backup answer"] and calls == ["groq", "gemini"]


def test_backup_is_off_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "LLM_FALLBACK", "")
    monkeypatch.setattr(generate, "_stream", fake(calls))
    with pytest.raises(groq.RateLimitError):
        run()
    assert calls == ["groq"]


def test_no_mixing_after_reply_started(on, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "_stream", fake(calls, fail_after=True))
    with pytest.raises(groq.RateLimitError):
        run()
    assert calls == ["groq"]


def test_benchmark_provider_never_falls_back(on, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "_stream", fake(calls))
    with pytest.raises(groq.RateLimitError):
        run(provider="groq")
    assert calls == ["groq"]
