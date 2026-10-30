"""A saved shutdown must retain the device failure that caused it. Offline."""

from types import SimpleNamespace

import pytest

from operation_love import bugreport


RUN_ID = "device-failure-run"
CAUSE = "persistent UHID session is not open (or has already died)"
STOP_LINE = f"14:08:14 {CAUSE}; Stopping run so buffered data can be saved."
SAVE_LINE = f"14:08:14 Run {RUN_ID}: ✅ all data saved to bigquery [decisions=3]"


def _assessment(monkeypatch, *, logs=(), stop_kind=None, stop_reason=None):
    monkeypatch.setattr(bugreport, "recent_logs", lambda _limit: list(logs))
    monkeypatch.setattr(bugreport, "_completion_capture_facts", lambda *_: {})
    status = {
        "run_id": RUN_ID,
        "phase": "stopped",
        "running": False,
        "apps": {
            "hinge": {
                "state": "stopped",
                "stop_kind": stop_kind,
                "stop_reason": stop_reason,
            }
        },
    }
    hub = SimpleNamespace(snapshot=lambda: {"running": False, "status": status})
    return bugreport._run_completion_assessment_md(hub, "unused.yaml")


def test_driver_closed_status_survives_a_successful_save_without_process_logs(
    monkeypatch,
):
    md = _assessment(
        monkeypatch,
        stop_kind="driver_closed",
        stop_reason=f"DriverClosed: {CAUSE}",
    )

    assert "Outcome: COMPLETED WITH ERRORS" in md
    assert "all workers exited and the store flush succeeded" in md
    assert "driver-session failure interrupted the run" in md
    assert f"DriverClosed: {CAUSE}" in md
    assert "safety or deck condition" not in md


def test_historical_log_only_driver_failure_is_not_hidden_by_provider_fallback(
    monkeypatch,
):
    md = _assessment(
        monkeypatch,
        logs=[
            f"14:03:12 Run {RUN_ID}: Gemini opener: trying the next configured model.",
            STOP_LINE,
            "14:08:14 Supervisor: stopping — waiting up to 105s for 1 worker(s)",
            SAVE_LINE,
        ],
    )

    assert "Outcome: COMPLETED WITH ERRORS" in md
    assert CAUSE in md
    assert "1 provider failure(s) recovered" in md


@pytest.mark.parametrize(
    "logs",
    [
        [
            STOP_LINE,
            "14:08:14 Run another-run: ✅ all data saved to bigquery",
            SAVE_LINE,
        ],
        [STOP_LINE, "14:08:14 Operation Love — Darwin arm64", SAVE_LINE],
        [STOP_LINE],
        [STOP_LINE, f"14:08:14 Run {RUN_ID}: Gemini opener: trying the next model"],
        [
            f"14:08:14 Run another-run: hinge: {CAUSE}; "
            "Stopping run so buffered data can be saved.",
            SAVE_LINE,
        ],
    ],
)
def test_historical_failure_needs_evidence_linking_it_to_the_assessed_run(
    monkeypatch, logs
):
    md = _assessment(monkeypatch, logs=logs)

    assert "Outcome: COMPLETED CLEANLY" in md
    assert "driver-session failure" not in md


def test_current_tagged_failure_does_not_depend_on_a_save_log_line(monkeypatch):
    md = _assessment(
        monkeypatch,
        logs=[
            f"14:08:14 Run {RUN_ID}: hinge: {CAUSE}; "
            "Stopping run so buffered data can be saved."
        ],
    )

    assert "Outcome: COMPLETED WITH ERRORS" in md
    assert CAUSE in md


def test_status_and_log_failure_are_not_reported_twice(monkeypatch):
    md = _assessment(
        monkeypatch,
        logs=[STOP_LINE, SAVE_LINE],
        stop_kind="driver_closed",
        stop_reason=f"DriverClosed: {CAUSE}",
    )

    assert md.count(CAUSE) == 1


def test_unidentified_run_cannot_claim_an_untagged_historical_failure():
    assert bugreport._driver_closed_log_failures([STOP_LINE, SAVE_LINE], None) == []
