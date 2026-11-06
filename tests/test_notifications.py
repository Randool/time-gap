"""Notification turns must not erase human conversation gaps."""
import json
import sqlite3

import pytest

from test_time_gap import tg, _add_msg


@pytest.fixture
def clock(tg, monkeypatch):
    now = 1_800_000_000.0
    monkeypatch.setattr(tg.time, "time", lambda: now)
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, current: 0)
    monkeypatch.setattr(tg, "_plugin_config", lambda: {
        "exclude_user_prefixes": ["[background-event "]
    })
    return now


def seed(tg, now):
    _add_msg(tg._db, "s", "user", now - 6 * 3600 - 60)
    _add_msg(tg._db, "s", "assistant", now - 6 * 3600)


@pytest.mark.parametrize("persist_current", [False, True])
def test_notification_flood_preserves_gap(tg, clock, persist_current):
    seed(tg, clock)
    for minutes in (300, 240, 180, 120, 60, 12):
        _add_msg(tg._db, "s", "user", clock - minutes * 60 - 10,
                 "[background-event id] reminder")
        _add_msg(tg._db, "s", "assistant", clock - minutes * 60)
    if persist_current:
        _add_msg(tg._db, "s", "user", clock)
    result = tg._pre_llm_call(session_id="s", user_message="hello")
    assert "about 6 hours" in result["context"]
    assert "new calendar day" not in result["context"]


def test_assistant_completion_not_user_start_is_anchor(tg, clock):
    _add_msg(tg._db, "s", "user", clock - 6 * 3600)
    _add_msg(tg._db, "s", "assistant", clock - 60)
    assert tg._pre_llm_call(session_id="s", user_message="thanks") is None


def test_normal_reply_after_notification_resets_gap(tg, clock):
    seed(tg, clock)
    _add_msg(tg._db, "s", "user", clock - 180, "[background-event id]")
    _add_msg(tg._db, "s", "assistant", clock - 160)
    _add_msg(tg._db, "s", "user", clock - 120)
    _add_msg(tg._db, "s", "assistant", clock - 60)
    assert tg._pre_llm_call(session_id="s", user_message="hello") is None


def test_notification_only_history_is_not_a_human_exchange(tg, clock):
    _add_msg(tg._db, "s", "user", clock - 6 * 3600, "[background-event id]")
    _add_msg(tg._db, "s", "assistant", clock - 5 * 3600)
    assert tg._pre_llm_call(session_id="s", user_message="hello") is None


@pytest.mark.parametrize("content", [
    "[background-event id] task",
    [{"type": "text", "text": "[background-event id] task"}],
    json.dumps([{"type": "text", "text": "[background-event id] task"}]),
])
def test_current_notification_is_silent(tg, clock, content):
    seed(tg, clock)
    assert tg._pre_llm_call(session_id="s", user_message=content) is None


@pytest.mark.parametrize("kind,metadata", [
    ("notification", None),
    ("hidden", '{"notification_category":"diagnostic"}'),
    (None, '{"notification_category":"reminder"}'),
])
def test_metadata_filters_entire_turn_including_unmarked_tool_calls(tg, clock, kind, metadata):
    seed(tg, clock)
    with sqlite3.connect(tg._db) as conn:
        conn.execute("ALTER TABLE messages ADD COLUMN display_kind TEXT")
        conn.execute("ALTER TABLE messages ADD COLUMN display_metadata TEXT")
        conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                     ("s", "user", "arbitrary event text", clock - 120, kind, metadata))
    _add_msg(tg._db, "s", "assistant", clock - 100)  # tool call
    _add_msg(tg._db, "s", "tool", clock - 80)
    _add_msg(tg._db, "s", "assistant", clock - 60)  # final response
    assert "about 6 hours" in tg._pre_llm_call(session_id="s")["context"]


def test_midnight_rollover_survives_background_turn(tg, clock, monkeypatch):
    _add_msg(tg._db, "s", "user", clock - 600)
    _add_msg(tg._db, "s", "assistant", clock - 590)
    _add_msg(tg._db, "s", "user", clock - 120, "[background-event id]")
    _add_msg(tg._db, "s", "assistant", clock - 60)
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, now: int(prev < clock - 300))
    assert "new calendar day" in tg._pre_llm_call(session_id="s")["context"]


def test_hidden_alone_does_not_mean_notification(tg, clock):
    assert not tg._notification("hello", "hidden", "invalid json")
    assert not tg._notification("hello", "hidden", {})
    assert not tg._notification("mention [background-event id] in prose")


@pytest.mark.parametrize("prefixes", [None, "[", ["", None, 42]])
def test_invalid_prefix_configuration_does_not_exclude_humans(tg, clock, monkeypatch, prefixes):
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"exclude_user_prefixes": prefixes})
    assert not tg._notification("hello")


def test_equal_timestamps_keep_turn_boundaries(tg, clock):
    seed(tg, clock)
    _add_msg(tg._db, "s", "user", clock - 60, "[background-event id]")
    _add_msg(tg._db, "s", "assistant", clock - 60)
    _add_msg(tg._db, "s", "user", clock - 60)
    assert "about 6 hours" in tg._pre_llm_call(session_id="s")["context"]
