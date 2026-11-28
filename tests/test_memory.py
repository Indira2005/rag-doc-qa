import asyncio

from rag import generate


def run(question, history, reply, monkeypatch):
    seen = []

    async def fake(provider, messages, max_retries=2):
        seen.append(messages)
        if isinstance(reply, Exception):
            raise reply
        yield reply

    monkeypatch.setattr(generate, "_stream", fake)
    return asyncio.run(generate.standalone_question(question, history)), seen


HISTORY = [{"role": "user", "content": "How many attention heads?"},
           {"role": "assistant", "content": "Eight."}]


def test_first_question_makes_no_extra_call(monkeypatch):
    assert run("hello", [], "x", monkeypatch) == ("hello", [])


def test_follow_up_is_rewritten_using_history(monkeypatch):
    out, seen = run("why that many?", HISTORY, " Why eight attention heads? ", monkeypatch)
    assert out == "Why eight attention heads?"
    assert "How many attention heads?" in seen[0][1]["content"]


def test_failed_or_empty_rewrite_keeps_original(monkeypatch):
    assert run("why?", HISTORY, RuntimeError(), monkeypatch)[0] == "why?"
    assert run("why?", HISTORY, "", monkeypatch)[0] == "why?"
