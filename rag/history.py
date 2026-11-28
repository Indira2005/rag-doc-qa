"""Saved chat messages: one small JSON file per chat, kept in the data volume."""
import json
import re
from pathlib import Path

DIR = Path("data/history")


def _path(chat_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", chat_id):  # also blocks paths such as ../
        raise ValueError("Invalid chat id")
    return DIR / f"{chat_id}.json"


def load(chat_id: str) -> list:
    path = _path(chat_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save(chat_id: str, messages: list) -> None:
    path = _path(chat_id)
    DIR.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(messages), encoding="utf-8")
    temp.replace(path)  # never leave a half-written file


def clear(chat_id: str) -> None:
    _path(chat_id).unlink(missing_ok=True)
