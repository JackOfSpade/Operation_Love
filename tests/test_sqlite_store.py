"""SQLiteStore source-tagging and daily-limit behavior."""
import json
import sqlite3
import time

from operation_love.ranker import Store
from operation_love.ranker.store import SQLiteStore


def test_sqlite_store_conforms_to_store_protocol(tmp_path):
    s = SQLiteStore(tmp_path / "store.db")
    try:
        assert isinstance(s, Store)
    finally:
        s.close()


def test_sqlite_count_today_counts_only_auto_decisions(tmp_path):
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE decisions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "decision TEXT, score REAL)"
    )
    con.execute(
        "INSERT INTO decisions (run_id, app, created_at, decision, score) VALUES (?,?,?,?,?)",
        ("legacy", "bumble", time.time(), "like", 1.0),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        store.record_decision("r", "bumble", "like", 0.9, source="manual")
        store.record_decision("r", "bumble", "dislike", 0.1, source="auto")
        store.record_decision("r", "hinge", "like", 0.8, source="auto")

        assert store.count_today("bumble") == 1
        rows = store.con.execute(
            "SELECT decision, source FROM decisions WHERE app='bumble' ORDER BY id"
        ).fetchall()
        assert rows == [("like", None), ("like", "manual"), ("dislike", "auto")]
    finally:
        store.close()


def test_sqlite_add_label_persists_profile_id(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.add_label("r", "bumble", True, [0.1, 0.2], profile_id="profile-1")

        row = store.con.execute("SELECT profile_id FROM labels").fetchone()
        assert row == ("profile-1",)
    finally:
        store.close()


def test_sqlite_migrates_legacy_labels_profile_id_column(tmp_path):
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE labels ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "liked INTEGER, source TEXT, embedding TEXT, photo_count INTEGER)"
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(labels)").fetchall()]
        assert "profile_id" in cols

        store.add_label("r", "bumble", False, [0.3], profile_id="profile-legacy")
        row = store.con.execute("SELECT profile_id FROM labels").fetchone()
        assert row == ("profile-legacy",)
    finally:
        store.close()


def test_sqlite_load_labels_ordered_sorts_by_created_at(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        # Insert the later created_at row first so insertion order != chronological order.
        store.con.execute(
            "INSERT INTO labels (run_id, app, created_at, liked, source, embedding,"
            " photo_count, profile_id) VALUES (?,?,?,?,?,?,?,?)",
            ("r", "bumble", 200.0, 1, "manual", json.dumps([0.2]), 0, "later"),
        )
        store.con.execute(
            "INSERT INTO labels (run_id, app, created_at, liked, source, embedding,"
            " photo_count, profile_id) VALUES (?,?,?,?,?,?,?,?)",
            ("r", "bumble", 100.0, 0, "manual", json.dumps([0.1]), 0, "earlier"),
        )
        store.con.commit()

        assert store.load_labels_ordered() == [(False, [0.1]), (True, [0.2])]
    finally:
        store.close()



def test_sqlite_add_label_rejects_nan_embedding(tmp_path):
    """add_label() must raise (not silently store) NaN/Inf values in embeddings."""
    import math
    store = SQLiteStore(tmp_path / "store.db")
    try:
        try:
            store.add_label("r", "bumble", True, [0.1, float("nan"), 0.3])
            raise AssertionError("expected ValueError on NaN embedding")
        except ValueError as e:
            # json.dumps(..., allow_nan=False) is what actually rejects it (store.py),
            # and its real message is "Out of range float values are not JSON
            # compliant" — it never literally says "nan"/"allow_nan".
            assert "not json compliant" in str(e).lower()
    finally:
        store.close()
