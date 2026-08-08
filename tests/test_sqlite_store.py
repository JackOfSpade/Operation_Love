"""SQLiteStore source-tagging and daily-limit behavior."""
import json
import sqlite3
import time

from operation_love.ranker.store import SQLiteStore


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
    store = SQLiteStore(tmp_path / "store.db")
    try:
        try:
            store.add_label("r", "bumble", True, [0.1, float("nan"), 0.3])
            raise AssertionError("expected ValueError on NaN embedding")
        except ValueError as e:
            assert "nan" in str(e).lower() or "allow_nan" in str(e).lower()
    finally:
        store.close()


def test_sqlite_spend_today_sums_only_local_today(tmp_path):
    """spend_today() gates budget.day_budget_usd -- it must sum only rows since local
    midnight, excluding anything from a prior local day."""
    from operation_love.costing import Usage
    store = SQLiteStore(tmp_path / "store.db")
    try:
        usage = Usage(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                      cache_creation_input_tokens=0)
        store.record_spend("r", "claude-x", usage, 0.05)
        store.record_spend("r", "claude-x", usage, 0.02)
        # Backdate one row well outside any local day (25h) so it must be excluded.
        store.con.execute(
            "UPDATE spend SET created_at = ? WHERE id = (SELECT MIN(id) FROM spend)",
            (time.time() - 90_000,),
        )
        store.con.commit()

        assert abs(store.spend_today() - 0.02) < 1e-9
    finally:
        store.close()


def test_stats_show_uses_read_only_store(tmp_path, monkeypatch):
    """stats.show() only reads; it must ask make_store() for ensure=False so it doesn't
    run BigQuery table DDL / bucket IAM patching just to print a readout (hub.py and
    tools/eval_aggregation.py already pass ensure=False for their read-only paths)."""
    from operation_love import config as cfg_mod
    from operation_love import stats

    cfg = cfg_mod.Config(
        enabled_apps=["bumble"], mode="observe", apps={}, limits={},
        data_dir=tmp_path, db_file=tmp_path / "s.db",
        ranker=cfg_mod.RankerCfg(), quality_filter=cfg_mod.QualityCfg(),
        opener=cfg_mod.OpenerCfg(enabled=False),
        budget=cfg_mod.BudgetCfg(pricing={}),
        pacing=cfg_mod.PacingCfg(),
        storage=cfg_mod.StorageCfg(backend="sqlite"),
    )
    monkeypatch.setattr(cfg_mod, "load", lambda path: cfg)

    seen = {}
    real_store = SQLiteStore(tmp_path / "s.db")

    def fake_make_store(passed_cfg, ensure=True):
        seen["ensure"] = ensure
        return real_store

    monkeypatch.setattr(stats, "make_store", fake_make_store)

    stats.show("unused.yaml")

    assert seen["ensure"] is False
