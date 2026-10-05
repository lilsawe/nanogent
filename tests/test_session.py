"""Tests for session persistence."""

from __future__ import annotations

from nanogent.session import SessionStore


def messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_save_and_load(tmp_path):
    store = SessionStore(directory=tmp_path)
    store.save("work", messages(), model="deepseek-chat")

    loaded = store.load("work")
    assert loaded is not None
    assert loaded[-1]["content"] == "hello"


def test_load_missing_returns_none(tmp_path):
    assert SessionStore(directory=tmp_path).load("nope") is None


def test_latest_name(tmp_path):
    store = SessionStore(directory=tmp_path)
    store.save("first", messages())
    store.save("second", messages())
    assert store.latest_name() in {"first", "second"}


def test_list_reports_metadata(tmp_path):
    store = SessionStore(directory=tmp_path)
    store.save("work", messages(), model="deepseek-chat")

    items = store.list()
    assert len(items) == 1
    assert items[0]["name"] == "work"
    assert items[0]["message_count"] == 3


def test_unsafe_name_is_sanitized(tmp_path):
    store = SessionStore(directory=tmp_path)
    path = store.save("../../etc/passwd", messages())
    assert path.parent == tmp_path
