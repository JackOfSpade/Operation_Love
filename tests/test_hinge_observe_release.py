"""Release-gate tests: production OBSERVE evidence unlocks AUTO, nothing weaker does."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from operation_love import config as config_mod
from tools import hinge_observe_release as release


def _calibration():
    return {
        "schema_version": 3, "device": "pixel", "hinge_version_name": "9.134.0",
        "frame_size_px": [1080, 2400], "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v1", "identity_match_max_dist": 1.0,
        "inline_item_max_dist": 2.0, "calibrated_at": "2026-08-13T00:00:00Z",
        "identity_band": [0.0, 0.0, 1.0, 0.1], "content_band": [0.1, 0.9],
    }


def _cfg(calibration, *, mode="observe"):
    return SimpleNamespace(enabled_apps=["hinge"], mode=mode, db_file=Path("store.db"),
                           storage=SimpleNamespace(backend="sqlite", bigquery={}),
                           apps={"hinge": {"mode": mode, "targeting_calibration": calibration}})


def _production_inputs(tmp_path: Path, run_id="run-1"):
    debug = tmp_path / "data" / "hinge_debug" / run_id
    debug.mkdir(parents=True)
    rows = [
        {"ts": "2026-08-13T00:00:00", "action": "observe_release_run_binding",
         "run_id": run_id, "app": "hinge"},
        {"action": "observe_decision", "decision": "pass"},
        {"action": "observe_release_hub_pre_tap_published"},
        {"action": "observe_like_anchor"},
        {"action": "observe_release_post_tap_item_verified"},
        {"action": "observe_waiting", "reason": "like_sending"},
        {"action": "observe_decision", "decision": "like"},
        {"action": "observe_release_refusal_or_paywall_logged"},
    ]
    (debug / "actions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    db = tmp_path / "store.db"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE labels (id INTEGER, run_id TEXT, app TEXT, created_at REAL, liked INTEGER, source TEXT);
            CREATE TABLE decisions (id INTEGER, run_id TEXT, app TEXT, created_at REAL, decision TEXT, source TEXT);
        """)
        con.execute("INSERT INTO labels VALUES (1, ?, 'hinge', 1, 0, 'manual')", (run_id,))
        con.execute("INSERT INTO labels VALUES (2, ?, 'hinge', 2, 1, 'manual')", (run_id,))
        con.execute("INSERT INTO decisions VALUES (1, ?, 'hinge', 1, 'dislike', 'manual')", (run_id,))
        con.execute("INSERT INTO decisions VALUES (2, ?, 'hinge', 2, 'like', 'manual')", (run_id,))
        con.execute("CREATE TABLE openers (id INTEGER, run_id TEXT, app TEXT)")
        con.execute("INSERT INTO openers VALUES (1, ?, 'hinge')", (run_id,))
        con.execute("CREATE TABLE opener_rejections (id INTEGER, run_id TEXT, app TEXT)")
        con.execute("CREATE TABLE spend (id INTEGER, run_id TEXT)")
    return debug, db


def test_verifier_emits_exact_release_mapping_that_config_accepts(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    calibration = _calibration()
    cfg = _cfg(calibration)
    debug, db = _production_inputs(tmp_path)

    cfg.db_file = db
    artifact, paste = release.verify(cfg=cfg, run_dir=debug, run_id="run-1",
                                     out_dir=tmp_path / "ops" / "release")

    assert artifact["completed"] is True
    assert (tmp_path / paste["verification_file"]).is_file()
    cfg.mode = "auto"
    cfg.apps["hinge"]["mode"] = "auto"
    cfg.apps["hinge"]["observe_release_evidence"] = paste
    config_mod._validate_hinge_auto_release_evidence(cfg)


def test_auto_release_gate_refuses_missing_or_tampered_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(_calibration(), mode="auto")
    with pytest.raises(ValueError, match="AUTO is blocked"):
        config_mod._validate_hinge_auto_release_evidence(cfg)

    debug, db = _production_inputs(tmp_path)
    observe_cfg = _cfg(_calibration())
    observe_cfg.db_file = db
    _artifact, paste = release.verify(cfg=observe_cfg, run_dir=debug, run_id="run-1",
                                      out_dir=tmp_path / "ops" / "release")
    cfg.apps["hinge"]["observe_release_evidence"] = paste
    artifact_path = tmp_path / paste["verification_file"]
    artifact_path.write_text("{}")
    with pytest.raises(ValueError, match="verification_sha256"):
        config_mod._validate_hinge_auto_release_evidence(cfg)


def test_verifier_refuses_without_a_persisted_pass_or_like_half_of_the_cycle(tmp_path: Path):
    calibration = _calibration()
    debug, db = _production_inputs(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM labels WHERE liked=0")
    cfg = _cfg(calibration)
    cfg.db_file = db
    with pytest.raises(release.ReleaseEvidenceRefused, match="complete manual Hinge pass\\+like cycle"):
        release.verify(cfg=cfg, run_dir=debug, run_id="run-1", out_dir=tmp_path / "release")


@pytest.mark.parametrize("table, where", [
    ("labels", "liked=1"),
    ("decisions", "decision='dislike'"),
    ("decisions", "decision='like'"),
])
def test_verifier_requires_every_outcome_specific_persistence_record(tmp_path: Path, table: str, where: str):
    """Broad manual totals must not let duplicated likes impersonate a pass+like cycle."""
    debug, db = _production_inputs(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute(f"DELETE FROM {table} WHERE {where}")
    cfg = _cfg(_calibration())
    cfg.db_file = db
    with pytest.raises(release.ReleaseEvidenceRefused, match="complete manual Hinge pass\\+like cycle"):
        release.verify(cfg=cfg, run_dir=debug, run_id="run-1", out_dir=tmp_path / "release")


def test_verifier_requires_successful_hinge_opener_not_rejection_or_spend(tmp_path: Path):
    debug, db = _production_inputs(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM openers")
        con.execute("INSERT INTO opener_rejections VALUES (1, 'run-1', 'hinge')")
        con.execute("INSERT INTO spend VALUES (1, 'run-1')")
    cfg = _cfg(_calibration())
    cfg.db_file = db
    with pytest.raises(release.ReleaseEvidenceRefused, match="successful persisted Hinge opener"):
        release.verify(cfg=cfg, run_dir=debug, run_id="run-1", out_dir=tmp_path / "release")


def test_verifier_refuses_unrelated_debug_dir_or_worker_binding(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    debug, db = _production_inputs(tmp_path)
    cfg = _cfg(_calibration())
    cfg.db_file = db
    with pytest.raises(release.ReleaseEvidenceRefused, match="directory name"):
        release.verify(cfg=cfg, run_dir=debug, run_id="other-run", out_dir=tmp_path / "release-a")

    rows = [json.loads(line) for line in (debug / "actions.jsonl").read_text().splitlines()]
    rows[0]["run_id"] = "other-run"
    (debug / "actions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(release.ReleaseEvidenceRefused, match="first production Worker run binding"):
        release.verify(cfg=cfg, run_dir=debug, run_id="run-1", out_dir=tmp_path / "release-b")


def test_verifier_does_not_require_a_naturally_encountered_blocked_path(monkeypatch, tmp_path: Path):
    """A release run must not induce a paywall/refusal just to unlock AUTO."""
    monkeypatch.chdir(tmp_path)
    calibration = _calibration()
    debug, db = _production_inputs(tmp_path)
    rows = [json.loads(line) for line in (debug / "actions.jsonl").read_text().splitlines()]
    (debug / "actions.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in rows
        if row["action"] != "observe_release_refusal_or_paywall_logged"))
    cfg = _cfg(calibration)
    cfg.db_file = db
    artifact, _paste = release.verify(
        cfg=cfg, run_dir=debug, run_id="run-1", out_dir=tmp_path / "release")
    assert artifact["completed"] is True


def test_verifier_accepts_a_non_sqlite_active_store_via_aggregate_release_api(monkeypatch, tmp_path: Path):
    """The verifier consumes the Store contract, so BigQuery needs no SQLite shadow copy."""
    monkeypatch.chdir(tmp_path)
    debug, _db = _production_inputs(tmp_path)
    cfg = _cfg(_calibration())
    cfg.storage = SimpleNamespace(backend="bigquery", bigquery={})

    class BigQueryShapedStore:
        called = None

        def observe_release_persistence_summary(self, run_id, app):
            self.called = (run_id, app)
            return {"manual_pass_labels": 1, "manual_like_labels": 1,
                    "manual_pass_decisions": 1, "manual_like_decisions": 1,
                    "successful_hinge_openers": 1}

    store = BigQueryShapedStore()
    _artifact, paste = release.verify(cfg=cfg, run_dir=debug, run_id="run-1",
                                      out_dir=tmp_path / "release", store=store)
    assert store.called == ("run-1", "hinge")
    assert paste["verification_file"].endswith("hinge_observe_release.json")


def test_historical_observe_log_lacks_new_runtime_release_facts_and_is_refused():
    """Old logs prove actual `like_sheet`/`like_sending` spelling but cannot unlock AUTO."""
    run_dir = Path("data/hinge_debug/run_20260811_011416")
    if not run_dir.is_dir():  # local debug evidence is intentionally not a CI fixture
        pytest.skip("private historical debug capture is unavailable")
    with pytest.raises(release.ReleaseEvidenceRefused, match="directory name|first production Worker run binding"):
        release._verify_debug_run(run_dir, run_dir.name)
