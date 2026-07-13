"""Unit tests for the time-gap Hermes plugin.

Run:
    python -m pytest plugins/time-gap/tests/test_time_gap.py -q
"""

import importlib.util
import sqlite3
import sys
import time
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_DAY_S = 24 * 3600


def _make_state_db(path: Path) -> None:
    """Create a minimal state.db with just the messages columns we query."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE messages (session_id TEXT, role TEXT, "
        "content TEXT, timestamp REAL)"
    )
    conn.commit()
    conn.close()


def _add_msg(path: Path, session_id: str, role: str, ts: float) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (session_id, role, "x", ts),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def tg(monkeypatch, tmp_path):
    """Fresh module instance pointed at a temp state.db, with a clean config."""
    spec = importlib.util.spec_from_file_location(
        "time_gap_under_test", _PLUGIN_DIR / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    db = tmp_path / "state.db"
    _make_state_db(db)
    mod._db = db  # stash for tests to write into
    monkeypatch.setattr(mod, "_db_path", lambda: db)
    monkeypatch.setattr(mod, "_plugin_config", lambda: {})
    return mod


def test_no_prior_assistant_no_injection(tg):
    """Brand-new session (no assistant row yet) → nothing injected."""
    # Only a current user row exists; there is no prior exchange.
    _add_msg(tg._db, "s1", "user", time.time())
    assert tg._pre_llm_call(session_id="s1") is None


def test_missing_db_no_injection(tg, monkeypatch):
    """A non-existent state.db must not raise, just returns None."""
    monkeypatch.setattr(tg, "_db_path", lambda: tg._db.parent / "nope.db")
    assert tg._pre_llm_call(session_id="s1") is None


def test_recent_exchange_no_injection(tg):
    """Last assistant reply 5 min ago (< 30 min default) → no injection."""
    _add_msg(tg._db, "s2", "assistant", time.time() - 5 * 60)
    assert tg._pre_llm_call(session_id="s2") is None


def test_resume_after_gap_injects(tg):
    """Last assistant reply 2 days ago → inject a day-tier note."""
    _add_msg(tg._db, "s3", "assistant", time.time() - 2 * _DAY_S)
    result = tg._pre_llm_call(session_id="s3")
    assert result is not None
    assert "time-gap" in result["context"]
    assert "days" in result["context"]


def test_uses_last_assistant_not_user(tg):
    """A recent user row must not reset the clock — we key off assistant time.

    Simulates 'resume old session and type a new message': the current user row
    is 'now', but the last assistant reply is 2 days old → gap still fires.
    """
    now = time.time()
    _add_msg(tg._db, "s4", "assistant", now - 2 * _DAY_S)
    _add_msg(tg._db, "s4", "user", now)  # current turn, already persisted
    result = tg._pre_llm_call(session_id="s4")
    assert result is not None
    assert "days" in result["context"]


def test_picks_most_recent_assistant(tg):
    """With several assistant rows, the newest one wins."""
    now = time.time()
    _add_msg(tg._db, "s5", "assistant", now - 5 * _DAY_S)
    _add_msg(tg._db, "s5", "assistant", now - 10 * 60)  # 10 min ago → under threshold
    assert tg._pre_llm_call(session_id="s5") is None


def test_session_isolation(tg):
    """A gap in one session does not leak into another."""
    now = time.time()
    _add_msg(tg._db, "old", "assistant", now - 3 * _DAY_S)
    _add_msg(tg._db, "fresh", "assistant", now - 60)
    assert tg._pre_llm_call(session_id="fresh") is None
    assert tg._pre_llm_call(session_id="old") is not None


def test_graded_phrasing(tg):
    """Phrasing coarsens with the tier and rounds to the nearest unit."""
    assert "minutes" in tg._humanize(40 * 60)
    assert "hours" in tg._humanize(5 * 3600)
    assert tg._humanize(25 * 3600) == "about a day"
    assert tg._humanize(int(1.9 * 24 * 3600)) == "about 2 days"
    assert tg._humanize(3 * 24 * 3600) == "about 3 days"


def test_platform_exclusion(tg, monkeypatch):
    """cron platform is excluded by default → never injects even with a big gap."""
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"exclude_platforms": ["cron"]})
    _add_msg(tg._db, "s6", "assistant", time.time() - 5 * _DAY_S)
    assert tg._pre_llm_call(session_id="s6", platform="cron") is None


def test_disabled_config(tg, monkeypatch):
    """enabled: false disables the plugin entirely."""
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"enabled": False})
    _add_msg(tg._db, "s7", "assistant", time.time() - 5 * _DAY_S)
    assert tg._pre_llm_call(session_id="s7") is None


def test_custom_threshold(tg, monkeypatch):
    """A small custom threshold makes even a short gap fire."""
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"min_gap_seconds": 10})
    _add_msg(tg._db, "s8", "assistant", time.time() - 20)
    assert tg._pre_llm_call(session_id="s8") is not None


def test_empty_session_id(tg):
    """Empty session id → None, no query attempted."""
    assert tg._pre_llm_call(session_id="") is None


def test_reads_uncheckpointed_wal_commit(tg):
    """Regression: the plugin must see committed-but-uncheckpointed WAL rows.

    Hermes opens state.db in WAL mode. A read-only connection reads the tail of
    the WAL, so a committed assistant row that has not yet been checkpointed
    into the main db file MUST still be visible — otherwise a resume right after
    the prior turn could miss the gap. This pins the connection contract
    (?mode=ro over a WAL db) against accidental changes.
    """
    w = sqlite3.connect(str(tg._db))
    w.execute("PRAGMA journal_mode=WAL")
    w.execute("PRAGMA wal_autocheckpoint=0")  # nothing gets checkpointed
    w.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)",
        ("wal", "assistant", "x", time.time() - 3 * _DAY_S),
    )
    w.commit()  # committed, but data lives only in state.db-wal
    try:
        result = tg._pre_llm_call(session_id="wal")
        assert result is not None
        assert "days" in result["context"]
    finally:
        w.close()


def test_log_silent_by_default(tg, monkeypatch, tmp_path):
    """No debug flag → nothing is written, and no log file is created."""
    logfile = tmp_path / "debug.log"
    monkeypatch.setattr(tg, "_LOG_PATH", logfile)
    monkeypatch.setattr(tg, "_plugin_config", lambda: {})  # no 'debug' key
    _add_msg(tg._db, "s-quiet", "assistant", time.time() - 3 * _DAY_S)
    tg._pre_llm_call(session_id="s-quiet")
    assert not logfile.exists()


def test_log_writes_when_debug_enabled(tg, monkeypatch, tmp_path):
    """debug: true → an INJECT line is written to the plugin-local log file."""
    logfile = tmp_path / "debug.log"
    monkeypatch.setattr(tg, "_LOG_PATH", logfile)
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"debug": True})
    _add_msg(tg._db, "s-dbg", "assistant", time.time() - 3 * _DAY_S)
    tg._pre_llm_call(session_id="s-dbg")
    assert logfile.exists()
    body = logfile.read_text()
    assert "INJECT" in body and "s-dbg" in body


def test_log_never_raises_on_bad_path(tg, monkeypatch):
    """A broken log path must be swallowed, never propagated into a turn."""
    monkeypatch.setattr(tg, "_LOG_PATH", Path("/nonexistent-dir-xyz/debug.log"))
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"debug": True})
    tg._log("boom")  # must not raise


# --- date-rollover (past-midnight) trigger --------------------------------


def test_days_crossed_same_day_is_zero(tg):
    """Two instants on the same calendar day → 0 boundaries crossed."""
    from datetime import datetime

    base = datetime(2026, 7, 13, 10, 0).timestamp()
    later = datetime(2026, 7, 13, 23, 0).timestamp()
    assert tg._days_crossed(base, later) == 0


def test_days_crossed_over_midnight(tg):
    """23:55 → 00:05 next day is a tiny gap but one calendar boundary."""
    from datetime import datetime

    before = datetime(2026, 7, 13, 23, 55).timestamp()
    after = datetime(2026, 7, 14, 0, 5).timestamp()
    assert tg._days_crossed(before, after) == 1


def test_midnight_crossing_injects_despite_small_gap(tg, monkeypatch):
    """A sub-threshold gap still fires when the calendar date rolled over."""
    # Force a rollover regardless of the real clock/timezone.
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, now: 1)
    _add_msg(tg._db, "s-mid", "assistant", time.time() - 120)  # 2 min ago
    result = tg._pre_llm_call(session_id="s-mid")
    assert result is not None
    assert "new calendar day" in result["context"]


def test_same_day_small_gap_still_silent(tg, monkeypatch):
    """No rollover + sub-threshold gap → still silent (regression guard)."""
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, now: 0)
    _add_msg(tg._db, "s-same", "assistant", time.time() - 120)
    assert tg._pre_llm_call(session_id="s-same") is None


def test_large_gap_over_midnight_mentions_both(tg, monkeypatch):
    """A big gap that also crosses days mentions elapsed time AND the new day."""
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, now: 1)
    _add_msg(tg._db, "s-both", "assistant", time.time() - 2 * 3600)  # 2h, one day
    result = tg._pre_llm_call(session_id="s-both")
    assert result is not None
    assert "hours" in result["context"] and "new calendar day" in result["context"]


def test_multiday_gap_omits_redundant_rollover(tg, monkeypatch):
    """A multi-day gap already implies a date change → no rollover clause."""
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, now: 3)
    _add_msg(tg._db, "s-multi", "assistant", time.time() - 3 * _DAY_S)
    result = tg._pre_llm_call(session_id="s-multi")
    assert result is not None
    assert "days" in result["context"]
    assert "new calendar day" not in result["context"]


def test_date_change_trigger_can_be_disabled(tg, monkeypatch):
    """notify_date_change: false → rollover alone does not fire."""
    monkeypatch.setattr(tg, "_plugin_config", lambda: {"notify_date_change": False})
    monkeypatch.setattr(tg, "_days_crossed", lambda prev, now: 1)
    _add_msg(tg._db, "s-off", "assistant", time.time() - 120)  # small gap only
    assert tg._pre_llm_call(session_id="s-off") is None
