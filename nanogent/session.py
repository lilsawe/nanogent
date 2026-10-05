"""Session persistence: save and resume agent conversations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SESSION_DIR = Path(os.path.expanduser("~/.nanogent/sessions"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Stores conversation history as JSON files under ~/.nanogent/sessions."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else DEFAULT_SESSION_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_.") or "default"
        return self.directory / f"{safe}.json"

    def save(self, name: str, messages: list[dict[str, Any]], model: str | None = None) -> Path:
        payload = {
            "name": name,
            "saved_at": _utc_now(),
            "model": model,
            "message_count": len(messages),
            "messages": messages,
        }
        path = self._path(name)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, name: str) -> list[dict[str, Any]] | None:
        path = self._path(name)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        messages = payload.get("messages")
        return messages if isinstance(messages, list) else None

    def latest_name(self) -> str | None:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0].stem if files else None

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items.append({
                "name": payload.get("name", path.stem),
                "saved_at": payload.get("saved_at"),
                "message_count": payload.get("message_count"),
                "model": payload.get("model"),
            })
        return items
