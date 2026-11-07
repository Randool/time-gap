"""Hermes marks gateway notification turns with display_kind='internal_notification'.

time-gap must recognise that marker so a background delegation completion does
not look like a fresh human conversation turn.
"""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from test_time_gap import _PLUGIN_DIR


@pytest.fixture
def tg(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "time_gap_internal_notification", _PLUGIN_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
        "content TEXT, timestamp REAL, display_kind TEXT, display_metadata TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(module, "_db_path", lambda: db)
    monkeypatch.setattr(module, "_plugin_config", lambda: {
        "exclude_user_prefixes": [],
    })
    return module


def _add(tg, session_id, role, ts, content="x", display_kind=None):
    conn = sqlite3.connect(str(tg._db_path()))
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, display_kind) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, ts, display_kind),
    )
    conn.commit()
    conn.close()


def test_internal_notification_is_recognised(tg):
    assert tg._notification(
        "[ASYNC DELEGATION BATCH COMPLETE] done", "internal_notification"
    ) is True
    assert tg._notification(
        "[IMPORTANT: Background process] done", "internal_notification"
    ) is True


def test_plain_notification_still_recognised(tg):
    assert tg._notification("anything", "notification") is True


def test_ordinary_message_not_recognised(tg):
    assert tg._notification("刚刷到消息说上游本体更新了", None, None) is False


def test_internal_notification_turn_does_not_reset_clock(tg):
    """A delegation completion at the end of a session must not look new."""
    base = 1_800_000_000.0
    _add(tg, "s", "user", base)
    _add(tg, "s", "assistant", base + 5)
    _add(tg, "s", "user", base + 10,
         "[ASYNC DELEGATION BATCH COMPLETE — deleg_x] done", "internal_notification")
    _add(tg, "s", "assistant", base + 15)

    assert tg._last_exchange_ts("s") == base + 5


def test_internal_notification_without_display_kind_still_caught_by_prefix(tg):
    """Hermes sometimes persists the marker only in the content prefix."""
    base = 1_800_000_000.0
    _add(tg, "s", "user", base)
    _add(tg, "s", "assistant", base + 5)
    _add(tg, "s", "user", base + 10, "[ASYNC DELEGATION BATCH COMPLETE — deleg_x] done")
    _add(tg, "s", "assistant", base + 15)

    tg._plugin_config = lambda: {
        "exclude_user_prefixes": ["[ASYNC DELEGATION BATCH COMPLETE"],
    }
    assert tg._last_exchange_ts("s") == base + 5


def test_real_human_turn_still_advances_clock(tg):
    base = 1_800_000_000.0
    _add(tg, "s", "user", base)
    _add(tg, "s", "assistant", base + 5)
    _add(tg, "s", "user", base + 10, "[ASYNC DELEGATION BATCH COMPLETE] done",
         "internal_notification")
    _add(tg, "s", "assistant", base + 15)
    _add(tg, "s", "user", base + 100, "刚才那些是啥")
    _add(tg, "s", "assistant", base + 105)

    assert tg._last_exchange_ts("s") == base + 105
