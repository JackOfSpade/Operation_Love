"""Storage interface + local SQLite backend.

Two backends implement the same ``Store`` surface (see bigquery_store.py for the
cloud one). The orchestrator loads labels once at startup, keeps them in memory
for fast inference, and appends new rows through the store — so the hot path
never blocks on per-row I/O regardless of backend.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..costing import Usage


@runtime_checkable
class Store(Protocol):
    def load_labels(self) -> list[tuple[bool, list[float]]]: ...
    def record_profile(self, run_id: str, app: str, profile_id: str, liked: bool,
                       source: str = "manual", photos: list[bytes] | None = None,
                       photo_count: int = 0) -> bool: ...
    def add_label(self, run_id: str, app: str, liked: bool, embedding: list[float],
                  source: str = "manual", photo_count: int = 0,
                  profile_id: str = "") -> None: ...
    def record_decision(self, run_id: str, app: str, decision: str, score: float) -> None: ...
    def record_opener(self, run_id: str, app: str, model: str, opener: str, referenced: str) -> None: ...
    def record_spend(self, run_id: str, model: str, usage: Usage, cost: float) -> None: ...
    def label_count(self) -> int: ...
    def count_today(self, app: str) -> int: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    liked INTEGER, source TEXT, embedding TEXT, photo_count INTEGER
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    decision TEXT, score REAL
);
CREATE TABLE IF NOT EXISTS openers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    model TEXT, opener TEXT, referenced TEXT
);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, created_at REAL, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
    cache_write_tokens INTEGER, cost_usd REAL
);
"""


class SQLiteStore:
    """Local, offline, zero-dependency backend. Good default / fallback."""

    def __init__(self, db_file: str | Path):
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: safe to share across worker threads.
        self.con = sqlite3.connect(str(db_file), check_same_thread=False)
        self.con.executescript(_SCHEMA)
        self.con.commit()
        self._lock = threading.Lock()

    def load_labels(self) -> list[tuple[bool, list[float]]]:
        with self._lock:
            rows = self.con.execute("SELECT liked, embedding FROM labels").fetchall()
        return [(bool(liked), json.loads(emb)) for liked, emb in rows]

    def record_profile(self, run_id, app, profile_id, liked, source="manual",
                       photos=None, photo_count=0) -> bool:
        # SQLite is the offline, labels-only fallback: it doesn't archive images,
        # so there's nothing that can fail here — always "recorded".
        return True

    def add_label(self, run_id, app, liked, embedding, source="manual", photo_count=0,
                  profile_id="", **_):
        with self._lock:
            self.con.execute(
                "INSERT INTO labels (run_id, app, created_at, liked, source, embedding,"
                " photo_count) VALUES (?,?,?,?,?,?,?)",
                (run_id, app, time.time(), int(liked), source, json.dumps(embedding), photo_count),
            )
            self.con.commit()

    def record_decision(self, run_id, app, decision, score):
        with self._lock:
            self.con.execute(
                "INSERT INTO decisions (run_id, app, created_at, decision, score) VALUES (?,?,?,?,?)",
                (run_id, app, time.time(), decision, score),
            )
            self.con.commit()

    def record_opener(self, run_id, app, model, opener, referenced):
        with self._lock:
            self.con.execute(
                "INSERT INTO openers (run_id, app, created_at, model, opener, referenced) VALUES (?,?,?,?,?,?)",
                (run_id, app, time.time(), model, opener, referenced),
            )
            self.con.commit()

    def record_spend(self, run_id, model, usage: Usage, cost):
        with self._lock:
            self.con.execute(
                "INSERT INTO spend (run_id, created_at, model, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, time.time(), model, usage.input_tokens, usage.output_tokens,
                 usage.cache_read_input_tokens, usage.cache_creation_input_tokens, cost),
            )
            self.con.commit()

    def label_count(self) -> int:
        with self._lock:
            return self.con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    def count_today(self, app: str) -> int:
        import datetime
        start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        with self._lock:
            return self.con.execute(
                "SELECT COUNT(*) FROM decisions WHERE app=? AND created_at>=?", (app, start)
            ).fetchone()[0]

    def flush(self) -> None:
        with self._lock:
            self.con.commit()

    def close(self) -> None:
        with self._lock:
            self.con.close()
