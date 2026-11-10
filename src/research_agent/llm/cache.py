"""SQLite cache of API responses keyed by the SHA-256 of the canonical request, so reruns cost nothing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ..canonical import canonical_json


def request_key(payload: dict) -> str:
    """SHA-256 of the request as canonical JSON: sorted keys, no whitespace, UTF-8 without escapes."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    """Request key -> response JSON. Pass ":memory:" for a throwaway cache."""

    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT NOT NULL, "
                        "response TEXT NOT NULL, created TEXT NOT NULL)")
        self.db.commit()

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, response: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                        (key, model, json.dumps(response, ensure_ascii=False), datetime.now(UTC).isoformat()))
        self.db.commit()

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self.db.close()
