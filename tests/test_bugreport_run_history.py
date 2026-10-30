"""Run-wide completion evidence must survive a later successful capture. Offline."""

import json
from types import SimpleNamespace

import pytest

from operation_love import bugreport, config


RUN_ID = "history-run"


def _report(tmp_path, monkeypatch, records):
    debug_root = tmp_path / "debug"
    run = debug_root / RUN_ID
    run.mkdir(parents=True)
    (run / "actions.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    cfg = SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"debug_log": True, "debug_dir": str(debug_root)}},
    )
    monkeypatch.setattr(config, "load", lambda _path: cfg)
    monkeypatch.setattr(bugreport, "recent_logs", lambda _limit: [])
    status = {
        "run_id": RUN_ID,
        "phase": "stopped",
        "running": False,
        "apps": {"hinge": {"state": "stopped"}},
    }
    hub = SimpleNamespace(snapshot=lambda: {"running": False, "status": status})
    return (
        bugreport._completion_capture_facts(status, "unused.yaml"),
        bugreport._run_completion_assessment_md(hub, "unused.yaml"),
    )


def _refused(name="Sam", **fields):
    return {
        "action": "capture",
        "profile_name": name,
        "items": 0,
        "item_coverage": None,
        "items_unavailable": "the item index could not be built",
        **fields,
    }


def test_earlier_refusals_and_terminal_stop_survive_a_successful_latest_capture(
    tmp_path, monkeypatch
):
    records = []
    for name in ("Taylor", "Nicole", "Ellery"):
        records.extend(
            [
                {"action": "item_index_refused", "reason": "index unavailable"},
                _refused(name),
            ]
        )
    records.extend(
        [
            {"action": "capture", "profile_name": "Madi", "items": 5},
            {"action": "capture_aborted", "profile_name": "Juliana", "frames": 10},
            {
                "action": "capture_iteration_timing",
                "exit_reason": "stopped_during_settle",
            },
        ]
    )

    facts, md = _report(tmp_path, monkeypatch, records)

    assert facts["items_unavailable"] is None
    assert facts["targeting_refusals"] == 3  # capture + refusal rows count only once
    assert facts["capture_aborted_count"] == 1
    assert "Outcome: COMPLETED SAFELY, WITH LIMITATIONS" in md
    assert "targeting was refused in 3 other capture(s) during this run" in md
    assert "requested Stop abandoned 1 in-progress profile read(s)" in md
    assert "latest capture" not in md


@pytest.mark.parametrize("earlier_count", [0, 2])
def test_latest_refusal_is_not_counted_again_in_history(
    tmp_path, monkeypatch, earlier_count
):
    facts, md = _report(
        tmp_path,
        monkeypatch,
        [_refused("Earlier")] * earlier_count + [_refused("Latest")],
    )

    assert facts["targeting_refusals"] == earlier_count + 1
    assert "the latest capture for profile `Latest` produced no numbered items" in md
    if earlier_count:
        assert f"targeting was refused in {earlier_count} other capture(s)" in md
    else:
        assert "other capture(s)" not in md


@pytest.mark.parametrize("stop_token", ["stop", "navigation_cancelled"])
def test_history_does_not_label_a_stop_interruption_a_targeting_fault(
    tmp_path, monkeypatch, stop_token
):
    facts, md = _report(
        tmp_path,
        monkeypatch,
        [
            {
                "action": "still_photo_dwell_walk_candidate",
                "outcome": stop_token,
            },
            _refused(),
            {"action": "capture", "items": 3},
        ],
    )

    assert facts.get("targeting_refusals", 0) == 0
    assert "targeting was refused" not in md


def test_history_does_not_count_a_capture_without_an_opener_consumer(
    tmp_path, monkeypatch
):
    facts, md = _report(
        tmp_path,
        monkeypatch,
        [
            _refused(items_unavailable_kind="no_opener_consumer"),
            {"action": "capture", "items": 3},
        ],
    )

    assert facts.get("targeting_refusals", 0) == 0
    assert "Outcome: COMPLETED CLEANLY" in md


@pytest.mark.parametrize("boundary", ["capture", "capture_aborted"])
def test_stop_evidence_cannot_cross_a_previous_capture_boundary(
    tmp_path, monkeypatch, boundary
):
    facts, md = _report(
        tmp_path,
        monkeypatch,
        [
            {"action": "still_photo_dwell_walk_candidate", "outcome": "stop"},
            {"action": boundary},
            _refused(),
        ],
    )

    assert facts["targeting_refusals"] == 1
    assert facts["items_unavailable_after_stop"] is False
    assert "the latest capture for profile `Sam` produced no numbered items" in md


def test_unrecognised_capture_end_does_not_claim_requested_stop(tmp_path, monkeypatch):
    facts, md = _report(
        tmp_path,
        monkeypatch,
        [{"action": "capture", "items": 3}, {"action": "capture_failed"}],
    )

    assert facts.get("capture_aborted_count", 0) == 0
    assert "requested Stop" not in md
