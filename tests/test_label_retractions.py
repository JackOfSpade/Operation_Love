from __future__ import annotations

import copy
import json
import math

import pytest

from operation_love.ranker.retractions import RetractionRefused, make_plan, retraction_row
from operation_love.ranker.store import SQLiteStore
from tools import label_retraction


def _rows(*, current_order=True, linked=True):
    label_time = "2026-08-14T08:04:59+00:00" if current_order else "2026-08-14T08:04:58+00:00"
    decision_time = "2026-08-14T08:04:58+00:00" if current_order else "2026-08-14T08:04:59+00:00"
    decision = {"created_at": decision_time, "decision": "dislike", "score": 0.0}
    if linked:
        decision["profile_id"] = "Malaika"
    return {"run_id": "run", "app": "hinge", "source": "external_ai_review",
            "profiles": [{"profile_id": "Malaika"}], "retractions": [],
            "labels": [{"profile_id": "Malaika", "created_at": label_time, "liked": False}],
            "decisions": [decision]}


class _Store:
    def __init__(self, rows):
        self.rows = rows
        self.writes = []
    def retraction_run_rows(self, *_):
        return copy.deepcopy(self.rows)
    def append_label_retraction(self, row):
        if any(x["correction_id"] == row["correction_id"] for x in self.writes):
            return False
        self.writes.append(row)
        self.rows["retractions"].append({key: row[key] for key in (
            "correction_id", "profile_id", "label_created_at", "decision_created_at", "decision_fingerprint")})
        return True


def _plan(rows=None):
    rows = _rows() if rows is None else rows
    return make_plan(rows=rows, run_id="run", app="hinge", source="external_ai_review",
                     profile_id="Malaika", reason="external controller moved same profile",
                     evidence_ref="data/hinge_debug/a01fbcd1e9a0/actions.jsonl#225")


def test_plan_binds_current_worker_decision_before_label_by_profile_lineage():
    document = _plan()
    assert document["label_ordinal"] == 0
    assert document["decision_created_at"] == "2026-08-14T08:04:58+00:00"
    assert retraction_row(document)["correction_id"] == document["correction_id"]


def test_linked_plan_is_not_shifted_by_an_unlabelled_decision():
    rows = _rows()
    rows["decisions"].append({
        "profile_id": "archive-failed-profile",
        "created_at": "2026-08-14T08:05:01+00:00",
        "decision": "like",
        "score": 1.0,
    })
    document = _plan(rows)
    assert document["decision_created_at"] == "2026-08-14T08:04:58+00:00"


def test_plan_supports_legacy_label_before_unlinked_decision_ordering():
    document = _plan(_rows(current_order=False, linked=False))
    assert document["label_created_at"] == "2026-08-14T08:04:58+00:00"
    assert document["decision_created_at"] == "2026-08-14T08:04:59+00:00"


def test_plan_supports_unlinked_current_order_only_when_timeline_is_causal():
    rows = _rows(linked=False)
    rows["profiles"].append({"profile_id": "Bea"})
    rows["labels"] = [
        {"profile_id": "Malaika", "created_at": 11.0, "liked": False},
        {"profile_id": "Bea", "created_at": 21.0, "liked": True},
    ]
    rows["decisions"] = [
        {"created_at": 10.0, "decision": "dislike", "score": 0.0},
        {"created_at": 20.0, "decision": "like", "score": 1.0},
    ]
    document = _plan(rows)
    assert document["decision_created_at"] == 10.0


def test_plan_refuses_ambiguous_legacy_sequence():
    rows = _rows()
    rows["decisions"].append({"profile_id": "other",
                              "created_at": "2026-08-14T08:04:58+00:00",
                              "decision": "dislike", "score": 0})
    with pytest.raises(RetractionRefused, match="strict unique"):
        _plan(rows)


def test_plan_refuses_duplicate_or_missing_linked_target_decision():
    duplicate = _rows()
    duplicate["decisions"].append({
        "profile_id": "Malaika", "created_at": "2026-08-14T08:04:58.5+00:00",
        "decision": "dislike", "score": 0.0,
    })
    with pytest.raises(RetractionRefused, match="multiple linked decisions"):
        _plan(duplicate)

    missing = _rows()
    missing["decisions"][0]["profile_id"] = "someone-else"
    with pytest.raises(RetractionRefused, match="no linked decision"):
        _plan(missing)


def test_plan_refuses_missing_or_overlapping_legacy_rows():
    missing = _rows(linked=False)
    missing["decisions"] = []
    with pytest.raises(RetractionRefused, match="cardinality differs"):
        _plan(missing)

    overlapping = _rows(linked=False)
    overlapping["profiles"].append({"profile_id": "Bea"})
    overlapping["labels"] = [
        {"profile_id": "Malaika", "created_at": 30.0, "liked": False},
        {"profile_id": "Bea", "created_at": 40.0, "liked": True},
    ]
    overlapping["decisions"] = [
        {"created_at": 10.0, "decision": "dislike", "score": 0.0},
        {"created_at": 20.0, "decision": "like", "score": 1.0},
    ]
    with pytest.raises(RetractionRefused, match="adjacent causal sequence"):
        _plan(overlapping)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("profiles", [None], "profile rows are malformed"),
    ("retractions", ["not-a-row"], "retraction rows are malformed"),
    ("labels", [None], "label rows are malformed"),
    ("decisions", [None], "decision rows are malformed"),
    ("labels", None, "label rows are malformed"),
])
def test_plan_refuses_malformed_snapshot_row_shapes(field, value, message):
    rows = _rows()
    rows[field] = value

    with pytest.raises(RetractionRefused, match=message):
        _plan(rows)


def test_plan_refuses_non_mapping_snapshot_with_documented_error():
    with pytest.raises(RetractionRefused, match="store snapshot is malformed"):
        make_plan(
            rows=None, run_id="run", app="hinge", source="external_ai_review",
            profile_id="Malaika", reason="false pass", evidence_ref="debug#225")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_plan_refuses_noncanonical_json_in_snapshot_or_evidence_metadata(value):
    rows = _rows()
    rows["decisions"][0]["score"] = value
    with pytest.raises(RetractionRefused, match="canonical JSON"):
        _plan(rows)

    with pytest.raises(RetractionRefused, match="canonical JSON"):
        make_plan(
            rows=_rows(), run_id="run", app="hinge", source="external_ai_review",
            profile_id="Malaika", reason="false pass", evidence_ref="debug#225",
            evidence_metadata={"score": value},
        )


def test_apply_requires_exact_confirmation_and_unchanged_rows_and_is_idempotent():
    store = _Store(_rows())
    document = _plan()
    with pytest.raises(RetractionRefused, match="exact confirmation"):
        label_retraction.apply(store=store, document=document, confirmation="no")
    confirmation = label_retraction.CONFIRM_PREFIX + document["plan_sha256"]
    assert label_retraction.apply(store=store, document=document, confirmation=confirmation) is True
    # A retry has identical meaning and does not create a second tombstone.
    assert label_retraction.apply(store=store, document=document, confirmation=confirmation) is False
    changed = _Store(_rows())
    changed.rows["decisions"][0]["score"] = 0.5
    with pytest.raises(RetractionRefused, match="changed since planning"):
        label_retraction.apply(store=changed, document=document, confirmation=confirmation)


def test_sqlite_end_to_end_apply_retracts_training_label_and_its_paired_decision(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.con.execute(
            "INSERT INTO decisions (run_id,app,created_at,decision,score,source,profile_id) "
            "VALUES (?,?,?,?,?,?,?)",
            ("run", "hinge", 10.0, "dislike", 0.0, "external_ai_review", "Malaika"))
        store.con.execute("INSERT INTO labels VALUES (NULL,?,?,?,?,?,?,?,?)",
                          ("run", "hinge", 11.0, 0, "external_ai_review", "[0.1]", 0, "Malaika"))
        store.con.commit()
        document = label_retraction.plan(store=store, run_id="run", app="hinge", source="external_ai_review",
                                          profile_id="Malaika", reason="false pass", evidence_ref="debug#225")
        confirmation = label_retraction.CONFIRM_PREFIX + document["plan_sha256"]
        assert label_retraction.apply(store=store, document=document, confirmation=confirmation)
        assert store.load_labels() == []
        assert label_retraction.apply(store=store, document=document, confirmation=confirmation) is False
    finally:
        store.close()


def test_sqlite_retraction_uses_exact_real_timestamp_not_lossy_microsecond_iso(tmp_path):
    """A SQLite REAL can carry bits an ISO timestamp omits; equality must still retract it."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        # Deliberately not microsecond-aligned.  The historical ISO conversion rounded this
        # and left the label training-visible after a successful-looking append.
        decision_time = 1_786_710_199.5451021
        label_time = decision_time + 0.1234567
        store.con.execute(
            "INSERT INTO decisions (run_id,app,created_at,decision,score,source,profile_id) "
            "VALUES (?,?,?,?,?,?,?)",
            ("run", "hinge", decision_time, "dislike", 0.0, "external_ai_review", "Malaika"))
        store.con.execute("INSERT INTO labels VALUES (NULL,?,?,?,?,?,?,?,?)",
                          ("run", "hinge", label_time, 0, "external_ai_review", "[0.1]", 0, "Malaika"))
        store.con.commit()
        document = label_retraction.plan(store=store, run_id="run", app="hinge", source="external_ai_review",
                                          profile_id="Malaika", reason="false pass", evidence_ref="debug#225")
        assert "." in document["label_created_at"]
        assert label_retraction.apply(
            store=store, document=document,
            confirmation=label_retraction.CONFIRM_PREFIX + document["plan_sha256"],
        )
        assert store.load_labels() == []
    finally:
        store.close()


def test_debug_store_sequence_refuses_ordinal_shift_from_dropped_or_inverted_row():
    rows = _rows()
    label_retraction._require_exact_debug_store_sequence(rows, ["pass"])
    with pytest.raises(RetractionRefused, match="sequences differ"):
        label_retraction._require_exact_debug_store_sequence(rows, ["pass", "pass"])
    rows["labels"][0]["liked"] = True
    with pytest.raises(RetractionRefused, match="sequences differ"):
        label_retraction._require_exact_debug_store_sequence(rows, ["pass"])


def test_sqlite_store_refuses_competing_tombstone_for_same_bound_pair(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.con.execute("INSERT INTO labels VALUES (NULL,?,?,?,?,?,?,?,?)",
                          ("run", "hinge", 10.0, 0, "external_ai_review", "[0.1]", 0, "Malaika"))
        store.con.execute(
            "INSERT INTO decisions (run_id,app,created_at,decision,score,source,profile_id) "
            "VALUES (?,?,?,?,?,?,?)",
            ("run", "hinge", 11.0, "dislike", 0.0, "external_ai_review", ""))
        store.con.commit()
        document = label_retraction.plan(store=store, run_id="run", app="hinge", source="external_ai_review",
                                          profile_id="Malaika", reason="false pass", evidence_ref="debug#225")
        row = retraction_row(document)
        assert store.append_label_retraction(row)
        competing = dict(row, correction_id="different")
        with pytest.raises(RetractionRefused, match="different retraction"):
            store.append_label_retraction(competing)
    finally:
        store.close()


def test_debug_target_requires_local_non_symlinked_exact_worker_run_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    actions = tmp_path / "data" / "run-1" / "actions.jsonl"
    actions.parent.mkdir(parents=True)
    actions.write_text("\n".join(json.dumps(row) for row in [
        {"ts": "now", "action": "observe_release_run_binding", "run_id": "run-1", "app": "hinge"},
        {"ts": "now", "action": "observe_decision", "decision": "pass"},
    ]) + "\n")
    ordinal, outcomes, reference, _ = label_retraction._debug_target(
        actions, 2, run_id="run-1", app="hinge")
    assert (ordinal, outcomes) == (0, ["pass"])
    assert reference.startswith("data/run-1/actions.jsonl#row=2;")

    with pytest.raises(RetractionRefused, match="run/app binding"):
        label_retraction._debug_target(actions, 2, run_id="run-1", app="bumble")
    alternate = tmp_path / "outside.jsonl"
    alternate.write_text(actions.read_text())
    (tmp_path / "data" / "run-2").mkdir()
    (tmp_path / "data" / "run-2" / "actions.jsonl").symlink_to(alternate)
    with pytest.raises(RetractionRefused, match="symlinked"):
        label_retraction._debug_target(tmp_path / "data" / "run-2" / "actions.jsonl", 2,
                                        run_id="run-2", app="hinge")


@pytest.mark.parametrize("intruder", ["[]", "3", '"x"', "null"])
def test_debug_target_refuses_a_non_dict_row_rather_than_crashing(tmp_path, monkeypatch, intruder):
    """A correction is bound to a DECISION ORDINAL counted across every row in this file.

    `[]`, `3`, `"x"` and `null` are all valid JSON that `json.loads` accepts, and only the
    binding row and the target row were isinstance-checked, so `.get` on any other one raised a
    bare AttributeError that `main`'s handler does not catch: a traceback instead of the tool's
    REFUSED contract, from a tool whose whole job is auditable evidence.
    """
    monkeypatch.chdir(tmp_path)
    actions = tmp_path / "data" / "run-1" / "actions.jsonl"
    actions.parent.mkdir(parents=True)
    actions.write_text("\n".join([
        json.dumps({"ts": "now", "action": "observe_release_run_binding",
                    "run_id": "run-1", "app": "hinge"}),
        intruder,
        json.dumps({"ts": "now", "action": "observe_decision", "decision": "pass"}),
    ]) + "\n")
    with pytest.raises(RetractionRefused, match="malformed"):
        label_retraction._debug_target(actions, 3, run_id="run-1", app="hinge")
