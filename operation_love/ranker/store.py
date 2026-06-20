"""SQLite persistence: swipe labels (training data), decisions, openers, spend.

Replaces the old `targeted_index.txt` state file. Every swipe is a free label,
so the personal ranker improves over time; the spend table gives exact per-run
and lifetime opener cost.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..costing import Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    app         TEXT,
    captured_at REAL,
    bio         TEXT,
    prompts     TEXT,        -- JSON [[question, answer], ...]
    photo_count INTEGER
);
CREATE TABLE IF NOT EXISTS labels (        -- training labels = your swipes
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id  INTEGER,
    liked       INTEGER,      -- 1 like, 0 dislike
    source      TEXT,         -- manual | ranker
    embedding   BLOB,         -- float32 feature vector
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id  INTEGER,
    run_id      TEXT,
    decision    TEXT,         -- like | dislike | no_face
    score       REAL,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS openers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id  INTEGER,
    run_id      TEXT,
    model       TEXT,
    opener      TEXT,
    referenced  TEXT,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS spend (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT,
    created_at          REAL,
    model               TEXT,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_write_tokens  INTEGER,
    cost_usd            REAL
);
"""


class Store:
    def __init__(self, db_file: str | Path):
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(db_file))
        self.con.executescript(_SCHEMA)
        self.con.commit()

    # --- writes ---------------------------------------------------------
    def add_label(self, profile_id: int, liked: bool, source: str, embedding: bytes) -> None:
        self.con.execute(
            "INSERT INTO labels (profile_id, liked, source, embedding, created_at)"
            " VALUES (?,?,?,?,?)",
            (profile_id, int(liked), source, embedding, time.time()),
        )
        self.con.commit()

    def record_spend(self, run_id: str, model: str, usage: Usage, cost: float) -> None:
        self.con.execute(
            "INSERT INTO spend (run_id, created_at, model, input_tokens, output_tokens,"
            " cache_read_tokens, cache_write_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id, time.time(), model,
                usage.input_tokens, usage.output_tokens,
                usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
                cost,
            ),
        )
        self.con.commit()

    # --- reads ----------------------------------------------------------
    def label_count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    def run_spend(self, run_id: str) -> float:
        row = self.con.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM spend WHERE run_id = ?", (run_id,)
        ).fetchone()
        return float(row[0])

    def lifetime_spend(self) -> float:
        return float(self.con.execute("SELECT COALESCE(SUM(cost_usd),0) FROM spend").fetchone()[0])

    def close(self) -> None:
        self.con.close()
