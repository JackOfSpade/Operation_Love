from __future__ import annotations

import copy
import json

import pytest

from operation_love.ranker.retractions import RetractionRefused
from operation_love.ranker.opener_retractions import make_plan, verify_plan
from operation_love.ranker.retractions import canonical_sha
from operation_love.ranker.store import SQLiteStore
from tools import advisory_opener_cleanup


def test_legacy_712_gate_pins_the_full_precision_cloud_snapshot():
    assert advisory_opener_cleanup._LEGACY_712_OPENER == {
        "created_at": "2026-08-10T21:40:52.136804+00:00", "model": "gemini-3.6-flash",
        "opener_fingerprint": "04839c778f19c74cd1158afec4fc76a32ae41d34abbe8ffb4bb91c3161d2180b",
    }


def _actions(tmp_path, *, decision=False):
    path = tmp_path / "data" / "run-no-decision" / "actions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": "now", "action": "observe_release_run_binding", "run_id": "run-no-decision", "app": "hinge"}]
    if decision:
        rows.append({"ts": "later", "action": "observe_decision", "decision": "like"})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _snapshot() -> dict:
    return {
        "run_id": "run", "app": "hinge", "retractions": [],
        "preference_counts": {
            "profiles": 0, "profile_photos": 0, "labels": 0, "decisions": 0,
        },
        "effective_counts": {
            "like_labels": 0, "like_decisions": 0,
            "pass_labels": 0, "pass_decisions": 0,
        },
        "openers": [{
            "created_at": "2026-08-10T21:40:52.136804+00:00",
            "model": "gemini-x", "opener_fingerprint": "a" * 64,
        }],
    }


def _direct_plan(rows=None) -> dict:
    return make_plan(
        rows=_snapshot() if rows is None else rows,
        run_id="run", app="hinge", reason="unacted advisory draft",
        evidence_ref="data/run/actions.jsonl;actions_sha256=" + "b" * 64,
        evidence_metadata={"debug_actions_sha256": "b" * 64},
    )


@pytest.mark.parametrize("value", [False, 0.0, "0", -1])
def test_plan_refuses_coerced_or_negative_evidence_counts(value):
    rows = _snapshot()
    rows["effective_counts"]["like_labels"] = value
    with pytest.raises(RetractionRefused, match="nonnegative integers"):
        _direct_plan(rows)


def test_plan_refuses_missing_count_keys_and_non_mapping_snapshot():
    rows = _snapshot()
    del rows["preference_counts"]["labels"]
    with pytest.raises(RetractionRefused, match="exactly"):
        _direct_plan(rows)
    with pytest.raises(RetractionRefused, match="store snapshot is malformed"):
        make_plan(
            rows=None, run_id="run", app="hinge", reason="bad snapshot",
            evidence_ref="debug#1", evidence_metadata={},
        )


def test_verify_plan_rejects_rehashed_malformed_content_and_extra_fields():
    plan = _direct_plan()
    malformed = copy.deepcopy(plan)
    malformed["effective_counts"]["like_labels"] = False
    malformed["plan_sha256"] = canonical_sha({
        key: value for key, value in malformed.items() if key != "plan_sha256"})
    with pytest.raises(RetractionRefused, match="nonnegative integers"):
        verify_plan(malformed)

    extra = copy.deepcopy(plan)
    extra["approved"] = True
    extra["plan_sha256"] = canonical_sha({
        key: value for key, value in extra.items() if key != "plan_sha256"})
    with pytest.raises(RetractionRefused, match="invalid advisory"):
        verify_plan(extra)


def test_plan_refuses_non_hex_fingerprints_and_unparseable_timestamps():
    bad_digest = _snapshot()
    bad_digest["openers"][0]["opener_fingerprint"] = "g" * 64
    with pytest.raises(RetractionRefused, match="fingerprint"):
        _direct_plan(bad_digest)

    bad_timestamp = _snapshot()
    bad_timestamp["openers"][0]["created_at"] = "not-a-timestamp"
    with pytest.raises(RetractionRefused, match="created_at"):
        _direct_plan(bad_timestamp)


def test_dry_run_and_apply_tombstone_only_eager_openers_and_preserve_spend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("run-no-decision", "hinge", "gemini-x", "private opener", "detail")
        store.record_spend("run-no-decision", "gemini-x", type("U", (), {
            "input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0})(), 0.01)
        plan = advisory_opener_cleanup.dry_run(
            store=store, run_id="run-no-decision", app="hinge", reason="advisory draft was never acted",
            debug_actions=_actions(tmp_path))
        rendered = json.dumps(plan)
        assert "private opener" not in rendered
        assert plan["preference_counts"] == {"profiles": 0, "profile_photos": 0, "labels": 0, "decisions": 0}
        confirmation = advisory_opener_cleanup.CONFIRM_PREFIX + plan["plan_sha256"]
        assert advisory_opener_cleanup.apply(store=store, document=plan, confirmation=confirmation) is True
        assert advisory_opener_cleanup.apply(store=store, document=plan, confirmation=confirmation) is False
        assert store.con.execute("SELECT COUNT(*) FROM openers").fetchone()[0] == 1
        assert store.con.execute("SELECT COUNT(*) FROM spend").fetchone()[0] == 1
        assert store.ai_observe_release_persistence_summary("run-no-decision", "hinge", "external_ai_review")["successful_hinge_openers"] == 0
    finally:
        store.close()


def test_cleanup_refuses_observed_or_persisted_like(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("run-no-decision", "hinge", "gemini-x", "private opener", "detail")
        with pytest.raises(RetractionRefused, match="observed Like"):
            advisory_opener_cleanup.dry_run(store=store, run_id="run-no-decision", app="hinge",
                                             reason="bad eager row", debug_actions=_actions(tmp_path, decision=True))
        store.record_decision("run-no-decision", "hinge", "like", 0.5, source="auto")
        with pytest.raises(RetractionRefused, match="effective Like evidence"):
            advisory_opener_cleanup.dry_run(store=store, run_id="run-no-decision", app="hinge",
                                             reason="bad eager row", debug_actions=_actions(tmp_path))
    finally:
        store.close()


def test_cleanup_preserves_pass_only_history_but_refuses_any_effective_like(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("run-no-decision", "hinge", "gemini-x", "eager draft", "detail")
        store.add_label("run-no-decision", "hinge", False, [0.1], source="manual", profile_id="pass-card")
        store.record_decision("run-no-decision", "hinge", "dislike", 0.0, source="manual")
        plan = advisory_opener_cleanup.dry_run(
            store=store, run_id="run-no-decision", app="hinge", reason="unacted eager draft",
            debug_actions=_actions(tmp_path))
        assert plan["preference_counts"]["labels"] == 1
        assert plan["effective_counts"] == {
            "like_labels": 0, "like_decisions": 0, "pass_labels": 1, "pass_decisions": 1,
        }
    finally:
        store.close()


def test_two_row_cleanup_is_exact_idempotent_and_resumes_after_interruption(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("run-no-decision", "hinge", "gemini-x", "first draft", "detail")
        store.record_opener("run-no-decision", "hinge", "gemini-y", "second draft", "detail")
        plan = advisory_opener_cleanup.dry_run(
            store=store, run_id="run-no-decision", app="hinge", reason="unacted eager drafts",
            debug_actions=_actions(tmp_path))
        confirmation = advisory_opener_cleanup.CONFIRM_PREFIX + plan["plan_sha256"]

        original = store.append_opener_retraction
        calls = 0
        def fail_after_first(row):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected interruption")
            return original(row)
        store.append_opener_retraction = fail_after_first
        with pytest.raises(RuntimeError, match="interruption"):
            advisory_opener_cleanup.apply(store=store, document=plan, confirmation=confirmation)
        store.append_opener_retraction = original

        assert advisory_opener_cleanup.apply(store=store, document=plan, confirmation=confirmation) is True
        assert advisory_opener_cleanup.apply(store=store, document=plan, confirmation=confirmation) is False
        rows = store.con.execute(
            "SELECT correction_id,model,opener_fingerprint FROM opener_retractions ORDER BY opener_created_at"
        ).fetchall()
        assert len(rows) == 2 and {row[0] for row in rows} == {plan["correction_id"]}
    finally:
        store.close()
