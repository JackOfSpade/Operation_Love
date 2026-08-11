"""Local SQLite storage backend.

Two backends implement the same ``Store`` surface defined in ranker/__init__.py
(see bigquery_store.py for the cloud one). The supervisor loads labels once at
startup, keeps them in memory for fast inference, and appends new rows through
the store — so the hot path never blocks on per-row I/O regardless of backend.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import threading
import time
from pathlib import Path

from ..costing import Usage


def local_midnight_epoch() -> float:
    """Epoch seconds for local midnight — the start of "today" in the host's own
    timezone. The daily anti-ban limits (limits.max_per_day, budget.day_budget_usd)
    are human-scale, tied to the owner's own day, not a UTC reporting boundary, so
    BOTH store backends derive "today" from this one function and can't drift apart
    again (see BigQueryStore.count_today / spend_today)."""
    return datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    liked INTEGER, source TEXT, embedding TEXT, photo_count INTEGER, profile_id TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    decision TEXT, score REAL, source TEXT
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
        try:
            self.con.execute("ALTER TABLE decisions ADD COLUMN source TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        try:
            self.con.execute("ALTER TABLE labels ADD COLUMN profile_id TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        self.con.commit()
        self._lock = threading.Lock()

    def load_labels(self) -> list[tuple[bool, list[float]]]:
        with self._lock:
            rows = self.con.execute("SELECT liked, embedding FROM labels").fetchall()
        return [(bool(liked), json.loads(emb)) for liked, emb in rows]

    def load_labels_ordered(self) -> list[tuple[bool, list[float]]]:
        """Labels in swipe order (created_at asc) for the quality-trajectory chart."""
        with self._lock:
            rows = self.con.execute(
                "SELECT liked, embedding FROM labels ORDER BY created_at, id").fetchall()
        return [(bool(liked), json.loads(emb)) for liked, emb in rows]

    def record_profile(self, run_id, app, profile_id, liked, source="manual",
                       photos=None, photo_count=0, capture_truncated: bool = False) -> bool:
        # SQLite is the offline, labels-only fallback: it doesn't archive images,
        # so there's nothing that can fail here — always "recorded". capture_truncated
        # is accepted-and-ignored for the same reason: there's no profiles manifest
        # row here to hang it off of (see BigQueryStore.record_profile, the system of
        # record, for where "was this label made from an incomplete profile read?"
        # is actually kept queryable).
        return True

    def add_label(self, run_id, app, liked, embedding, source="manual", photo_count=0,
                  profile_id="", **_):
        with self._lock:
            self.con.execute(
                "INSERT INTO labels (run_id, app, created_at, liked, source, embedding,"
                " photo_count, profile_id) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, app, time.time(), int(liked), source,
                 json.dumps(embedding, allow_nan=False),
                 photo_count, profile_id),
            )
            self.con.commit()

    def record_decision(self, run_id, app, decision, score, source="auto"):
        with self._lock:
            self.con.execute(
                "INSERT INTO decisions (run_id, app, created_at, decision, score, source) VALUES (?,?,?,?,?,?)",
                (run_id, app, time.time(), decision, score, source),
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

    def count_today(self, app: str) -> int:
        """Auto-mode swipes recorded today (LOCAL day, i.e. since local midnight)."""
        start = local_midnight_epoch()
        with self._lock:
            return self.con.execute(
                "SELECT COUNT(*) FROM decisions WHERE app=? AND created_at>=? AND source='auto'",
                (app, start),
            ).fetchone()[0]

    def spend_today(self) -> float:
        """Sum of cost_usd recorded today (LOCAL day, i.e. since local midnight)."""
        start = local_midnight_epoch()
        with self._lock:
            row = self.con.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spend WHERE created_at >= ?",
                (start,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def flush(self) -> None:
        with self._lock:
            self.con.commit()

    def close(self) -> None:
        with self._lock:
            self.con.close()
