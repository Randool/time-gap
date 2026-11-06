"""Time-gap awareness for Hermes.

Appends an implicit "elapsed time" note to the current user turn *only* when the
gap since the last exchange crosses a threshold — silence is the default, so a
continuous conversation is never interrupted.

Via the ``pre_llm_call`` hook the note rides on the user message at API-call
time only (never persisted, never touches the cached system prompt). The gap is
read from ``state.db`` (read-only) using the last persisted *assistant* message
in a non-notification turn — the moment the prior exchange completed, which
naturally excludes the current user row. No in-process state; fails open on errors.

Set ``plugins.entries.time-gap.debug: true`` in config.yaml to append decisions
to ``debug.log`` next to this file.
"""

import json
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_MINUTE, _HOUR, _DAY = 60, 3600, 86400
_DEFAULT_MIN_GAP_SECONDS = 120 * _MINUTE
_DEFAULT_EXCLUDE_PLATFORMS = ("cron",)
_LOG_PATH = Path(__file__).resolve().parent / "debug.log"

# config.yaml doesn't change mid-run and load_config does file I/O — resolve once.
_config_cache: Optional[Dict[str, Any]] = None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _pre_llm_call)


def _log(msg: str) -> None:
    if not _plugin_config().get("debug"):
        return
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass  # a broken log path must never break a turn


def _plugin_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache is None:
        try:
            from hermes_cli.config import load_config

            entries = ((load_config() or {}).get("plugins") or {}).get("entries") or {}
            _config_cache = entries.get("time-gap") or {}
        except Exception:
            _config_cache = {}
    return _config_cache


def _min_gap_seconds(cfg: Dict[str, Any]) -> float:
    try:
        val = float(cfg.get("min_gap_seconds", _DEFAULT_MIN_GAP_SECONDS))
        return val if val > 0 else _DEFAULT_MIN_GAP_SECONDS
    except (TypeError, ValueError):
        return _DEFAULT_MIN_GAP_SECONDS


def _excluded(platform: str, cfg: Dict[str, Any]) -> bool:
    try:
        raw = cfg.get("exclude_platforms", _DEFAULT_EXCLUDE_PLATFORMS)
        return (platform or "").strip().lower() in {str(p).strip().lower() for p in raw}
    except TypeError:
        return (platform or "").strip().lower() in _DEFAULT_EXCLUDE_PLATFORMS


def _db_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state.db"


def _notification(content: Any, display_kind: Any = None,
                  display_metadata: Any = None) -> bool:
    """Recognize explicit notification markers, without guessing from prose."""
    if display_kind == "notification":
        return True
    if isinstance(display_metadata, str):
        try:
            display_metadata = json.loads(display_metadata)
        except (ValueError, TypeError):
            display_metadata = None
    if isinstance(display_metadata, dict) and display_metadata.get("notification_category"):
        return True
    prefixes = _plugin_config().get("exclude_user_prefixes", [])
    if not isinstance(prefixes, (list, tuple)) or not prefixes:
        return False
    if isinstance(content, str):
        # Persisted multimodal messages may be JSON-encoded content blocks.
        try:
            decoded = json.loads(content)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            content = decoded
    if isinstance(content, list):
        content = "\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
            and isinstance(part.get("text"), str)
        )
    return isinstance(content, str) and any(
        isinstance(prefix, str) and bool(prefix) and content.startswith(prefix)
        for prefix in prefixes
    )


def _last_exchange_ts(session_id: str) -> Optional[float]:
    """Last assistant timestamp outside notification turns, or None.

    Walk backwards in persisted message order. A user row owns all assistant
    rows up to the next user row, including tool-call messages. Do not filter
    hidden/inactive rows out of this stream: they still delimit turns.
    """
    path = _db_path()
    if not session_id or not path.exists():
        return None
    conn = None
    try:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=1.0
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        kind = "display_kind" if "display_kind" in columns else "NULL"
        metadata = "display_metadata" if "display_metadata" in columns else "NULL"
        rows = conn.execute(
            f"SELECT role, content, timestamp, {kind}, {metadata} FROM messages "
            "WHERE session_id = ? AND role IN ('user', 'assistant') "
            "ORDER BY rowid DESC",
            (session_id,),
        )
        candidate = None
        for role, content, ts, display_kind, display_metadata in rows:
            marked = _notification(content if role == "user" else None,
                                   display_kind, display_metadata)
            if role == "assistant":
                if not marked and isinstance(ts, (int, float)) and ts > 0:
                    candidate = max(candidate or ts, ts)
            elif marked:
                candidate = None
            elif candidate is not None:
                return candidate
        # Old/imported histories can contain assistant rows without a user row.
        return candidate
    except Exception:
        _log(f"DB read failed:\n{traceback.format_exc()}")
        return None
    finally:
        if conn is not None:
            conn.close()


def _local_date(epoch: float):
    """Calendar date at ``epoch`` in the user's configured timezone.

    Reuses Hermes' own timezone resolution so "midnight" matches the user's wall
    clock, not UTC. Falls back to server-local time if the helper is unavailable.
    """
    try:
        from hermes_time import get_timezone

        return datetime.fromtimestamp(epoch, get_timezone()).date()
    except Exception:
        return datetime.fromtimestamp(epoch).astimezone().date()


def _days_crossed(prev: float, now: float) -> int:
    """Number of calendar-day boundaries between prev and now (0 if same day)."""
    return (_local_date(now) - _local_date(prev)).days


def _humanize(gap: float) -> str:
    """Coarse phrasing, rounded to the nearest tier unit."""
    if gap >= _DAY:
        days = round(gap / _DAY)
        return "about a day" if days <= 1 else f"about {days} days"
    if gap >= _HOUR:
        hours = round(gap / _HOUR)
        if hours >= 24:  # 23.5h+ rounds up — call it a day, not "24 hours".
            return "about a day"
        return "about an hour" if hours <= 1 else f"about {hours} hours"
    minutes = round(gap / _MINUTE)
    return "about a minute" if minutes <= 1 else f"about {minutes} minutes"


def _build_context(gap: float, days_crossed: int, now: float) -> str:
    """Compose the note from whichever signals fired"""
    head = f"[time-gap: {_humanize(gap)} have passed since the previous exchange"
    if days_crossed >= 1:
        today = _local_date(now)
        weekday = today.strftime('%A')
        today_iso = today.isoformat()
        if gap < _DAY:
            head += f", crossing into a new calendar day (today is {weekday}, {today_iso})]"
        else:
            head += f" (today is {weekday}, {today_iso})]"
    else:
        head += "]"
    return (
        f"{head} (Your own sense of time — never quote or mention it. Just factor it in: "
        "earlier context or the current date may be stale; re-ground if needed.)"
    )


def _pre_llm_call(*, session_id: str = "", platform: str = "",
                  user_message: Any = None, **_: Any):
    """Return time-gap context when a gap or a date rollover crosses, else None."""
    try:
        cfg = _plugin_config()
        if cfg.get("enabled", True) is False or _excluded(platform, cfg):
            return None
        if _notification(user_message):
            return None
        prev = _last_exchange_ts(session_id)
        if prev is None:
            return None
        now = time.time()
        elapsed = now - prev
        days_crossed = (
            0
            if cfg.get("notify_date_change", True) is False
            else _days_crossed(prev, now)
        )
        # Two independent triggers: a large enough gap, OR a calendar rollover.
        if elapsed < _min_gap_seconds(cfg) and days_crossed < 1:
            return None
        context = _build_context(elapsed, days_crossed, now)
        _log(
            f"INJECT session={session_id[:16]} elapsed={int(elapsed)}s days={days_crossed}"
        )
        return {"context": context}
    except Exception:
        _log(f"error:\n{traceback.format_exc()}")
        return None
