import json

import pytest
from fastapi.testclient import TestClient

from rag import history


def test_history_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DIR", tmp_path)
    chat = "a" * 32
    assert history.load(chat) == []
    history.save(chat, [{"role": "user", "content": "hi"}])
    assert history.load(chat)[0]["content"] == "hi"
    history.clear(chat)
    assert history.load(chat) == []


@pytest.mark.parametrize("bad", ["../x", "abc", "A" * 32, "a" * 33])
def test_history_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        history.load(bad)
