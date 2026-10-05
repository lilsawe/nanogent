"""Lightweight JSONL tracing for agent runs."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preview(value: Any, limit: int = 500) -> str:
    """Return a compact string preview that is safe to store in trace logs."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


class TraceRecorder:
    """
    In-memory + optional JSONL trace recorder.

    Each event is a dictionary with a run id, timestamp, event name, and
    event-specific payload. This makes traces easy to inspect, diff, and feed
    into small evaluation scripts.
    """

    def __init__(self, path: str | Path | None = None, run_id: str | None = None):
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        item = {
            "run_id": self.run_id,
            "ts": _utc_now(),
            "event": event,
            **payload,
        }
        self.events.append(item)

        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return item

    def timer(self) -> float:
        return time.perf_counter()

    def elapsed_ms(self, start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
