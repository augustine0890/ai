"""Shared structured logger for the dds project.

Each run gets a unique session_id so all events from one execution are
grouped together. The JSONL file is kept to MAX_LINES (500) by trimming
the oldest entries once the file exceeds TRIM_THRESHOLD (550).

LLM-optimised entry shape
─────────────────────────
{
  "ts":      "2026-03-21T10:30:00Z",   // ISO-8601 UTC timestamp
  "seq":     42,                        // monotonic counter within the session
  "session": "a1b2c3d4",               // groups all events from one run
  "level":   "INFO",                   // INFO | WARN | ERROR
  "module":  "dds.worker",             // which file emitted the event
  "event":   "video_download_start",   // snake_case event name
  "data":    { ... },                  // all extra keyword args
  "error":   "Token expired ..."       // present only when level == ERROR
}

Levels are inferred automatically from the event name / presence of an
"error" kwarg:
  • ERROR  – event contains "error", or an "error=" kwarg is passed
  • WARN   – event contains "skip", "missing", "expired", or "warn"
  • INFO   – everything else
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_LINES: int = 500        # target line-count after a trim
TRIM_THRESHOLD: int = 550   # only rewrite the file once we exceed this

# ── Session state ──────────────────────────────────────────────────────────────
_session_id: str = uuid.uuid4().hex[:8]
_seq: int = 0
_trace_path: Path = Path(__file__).parent / "trace.jsonl"

# ── Internal helpers ───────────────────────────────────────────────────────────

def _infer_level(event: str, has_error: bool) -> str:
    if has_error or "error" in event:
        return "ERROR"
    if any(kw in event for kw in ("skip", "missing", "expired", "warn", "fail")):
        return "WARN"
    return "INFO"


def _trim_if_needed() -> None:
    """Keep only the most recent MAX_LINES entries."""
    try:
        text = _trace_path.read_text(encoding="utf-8")
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) > TRIM_THRESHOLD:
            trimmed = "\n".join(lines[-MAX_LINES:]) + "\n"
            _trace_path.write_text(trimmed, encoding="utf-8")
    except Exception:
        pass


# ── Public API ─────────────────────────────────────────────────────────────────

def log_event(module: str, event: str, **fields: Any) -> None:
    """Emit one structured log entry to stdout and trace.jsonl."""
    global _seq
    _seq += 1

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    # Promote "error" to a top-level field so LLMs can filter easily
    error_value = fields.pop("error", None)
    if error_value is not None:
        error_value = str(error_value)

    level = _infer_level(event, error_value is not None)

    # Coerce Path / Exception → str so json.dumps never raises
    data: dict[str, Any] = {
        k: str(v) if isinstance(v, (Path, Exception)) else v
        for k, v in fields.items()
    }

    entry: dict[str, Any] = {
        "ts":      now,
        "seq":     _seq,
        "session": _session_id,
        "level":   level,
        "module":  module,
        "event":   event,
        "data":    data,
    }
    if error_value is not None:
        entry["error"] = error_value

    # ── Console ────────────────────────────────────────────────────────────────
    level_tag = f"[{level}]".ljust(7)   # [INFO]  / [WARN]  / [ERROR]
    parts = [f"[{now}]", level_tag, f"[{module}]", f"session={_session_id}", f"seq={_seq}", f"event={event}"]
    for k, v in data.items():
        parts.append(f"{k}={v!r}")
    if error_value is not None:
        parts.append(f"error={error_value!r}")
    print(" ".join(parts))

    # ── JSONL file ─────────────────────────────────────────────────────────────
    try:
        with _trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        _trim_if_needed()
    except Exception:
        pass
