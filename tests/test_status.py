"""RunStatus live-status bus + the drivers' render_status overlay hook.

Pure offline: the status object is plain Python; the Bumble overlay is exercised
with a fake page that records the evaluate() call (the real HUD render is visual,
seen live in the browser). The Hinge driver inherits the base no-op.
"""
import math
import threading

import pytest

from operation_love.status import RunStatus
from operation_love.drivers.hinge import HingeDriver


class _Cfg:
    apps = {"bumble": {}, "hinge": {}}


def _mk(**kw):
    return RunStatus("run123", ["bumble", "hinge"], min_labels=40, mode="training", **kw)


# --- RunStatus -------------------------------------------------------------
def test_initial_snapshot():
    snap = _mk(labels=5).snapshot()
    assert snap["run_id"] == "run123"
    assert snap["labels"] == 5 and snap["min_labels"] == 40
    assert snap["labels_needed"] == 35
    assert snap["ranker_ready"] is False
    assert set(snap["apps"]) == {"bumble", "hinge"}
    assert snap["apps"]["bumble"]["state"] == "starting"


def test_record_swipe_and_inc_labels():
    s = _mk()
    s.record_swipe("bumble", "like", 0.0)
    s.record_swipe("bumble", "pass")
    s.inc_labels(2)
    snap = s.snapshot()
    assert snap["apps"]["bumble"]["swipes_run"] == 2
    assert snap["apps"]["bumble"]["last_decision"] == "pass"
    assert snap["labels"] == 2


@pytest.mark.parametrize(("field", "value"), [
    ("min_labels", True),
    ("min_labels", 1.9),
    ("min_labels", -1),
    ("labels", True),
    ("labels", 1.9),
    ("labels", -1),
    ("ranker_ready", 1),
])
def test_constructor_rejects_lossy_or_invalid_counter_state(field, value):
    kwargs = {"min_labels": 40, "mode": "training", field: value}
    with pytest.raises(ValueError, match=field):
        RunStatus("run123", ["hinge"], **kwargs)


@pytest.mark.parametrize(("args", "kwargs", "message"), [
    (("", ["hinge"]), {"min_labels": 1, "mode": "training"}, "run_id"),
    ((" run ", ["hinge"]), {"min_labels": 1, "mode": "training"}, "run_id"),
    (("run", ["hinge"]), {"min_labels": 1, "mode": "mixed"}, "mode"),
    (("run", "hinge"), {"min_labels": 1, "mode": "training"}, "apps"),
    (("run", (app for app in ["hinge"])),
     {"min_labels": 1, "mode": "training"}, "apps"),
    (("run", ["hinge", "hinge"]), {"min_labels": 1, "mode": "training"}, "duplicate"),
    (("run", [" hinge "]), {"min_labels": 1, "mode": "training"}, "app names"),
])
def test_constructor_rejects_malformed_run_identity_and_apps(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        RunStatus(*args, **kwargs)


@pytest.mark.parametrize("budget_cap", [
    True, -1, math.nan, math.inf, pytest.param(10 ** 10_000, id="huge_int"), "5",
])
def test_constructor_rejects_invalid_budget_cap(budget_cap):
    with pytest.raises(ValueError, match="budget_cap"):
        RunStatus(
            "run123", ["hinge"], min_labels=1, mode="training", budget_cap=budget_cap,
        )


@pytest.mark.parametrize("increment", [True, -1, 1.5, "1"])
def test_inc_labels_rejects_invalid_increments_without_corrupting_state(increment):
    status = _mk(labels=3)
    with pytest.raises(ValueError, match="increment"):
        status.inc_labels(increment)
    assert status.snapshot()["labels"] == 3


def test_labels_needed_floors_at_zero():
    assert _mk(labels=50).snapshot()["labels_needed"] == 0


def test_set_global():
    s = _mk()
    s.set_global(ranker_ready=True, budget_spent=1.2345, running=False)
    snap = s.snapshot()
    assert snap["ranker_ready"] is True
    assert snap["budget_spent"] == 1.2345
    assert snap["running"] is False


@pytest.mark.parametrize(("field", "value"), [
    ("labels", -1), ("labels", True), ("openers", 1.5),
    ("budget_spent", math.nan), ("budget_cap", math.inf),
    ("ranker_ready", 1), ("running", "false"), ("phase", " "),
])
def test_set_global_rejects_invalid_snapshot_values(field, value):
    status = _mk()
    with pytest.raises(ValueError, match=field):
        status.set_global(**{field: value})
    assert status.snapshot()[field] != value


@pytest.mark.parametrize("score", [True, -0.1, math.nan, math.inf, "0.5"])
def test_record_swipe_rejects_invalid_score_without_mutating_status(score):
    status = _mk()
    with pytest.raises(ValueError, match="score"):
        status.record_swipe("bumble", "like", score)
    assert status.snapshot()["apps"]["bumble"]["swipes_run"] == 0


def test_status_updates_reject_unknown_fields_atomically():
    status = _mk()
    with pytest.raises(ValueError, match="unknown AppStatus"):
        status.set_app("hinge", state="acting", statte="typo")
    assert status.snapshot()["apps"]["hinge"]["state"] == "starting"
    assert not hasattr(status._apps["hinge"], "statte")

    with pytest.raises(ValueError, match="unknown RunStatus"):
        status.set_global(phase="live", phaze="typo")
    assert status.snapshot()["phase"] == "starting"
    assert not hasattr(status, "phaze")


@pytest.mark.parametrize(("field", "value"), [
    ("last_score", math.nan), ("last_score", True), ("swipes_run", -1),
    ("swipes_run", 1.5), ("state", ["acting"]), ("mode", "observe"),
    ("last_decision", "maybe"), ("stop_kind", "unknown"), ("error", 1),
])
def test_set_app_rejects_invalid_snapshot_values_atomically(field, value):
    status = _mk()
    with pytest.raises(ValueError, match=field):
        status.set_app("hinge", **{field: value})
    app = status.snapshot()["apps"]["hinge"]
    assert app["state"] == "starting"


def test_stopping_defaults_false_and_is_settable_and_serialized():
    # `stopping` means "stop_event is set and workers are being given time to notice" --
    # strictly between running and stopped (see the field's own docstring in status.py).
    # supervisor.py's shutdown `finally` is the only writer; pin the plain mechanics here
    # so a future dataclass/field-list refactor can't silently drop it from snapshot().
    s = _mk()
    assert s.snapshot()["stopping"] is False
    s.set_global(stopping=True)
    assert s.snapshot()["stopping"] is True
    s.set_global(stopping=False)
    assert s.snapshot()["stopping"] is False


def test_set_app_autocreates_unknown_app():
    s = _mk()
    s.set_app("newapp", state="scoring")
    assert s.snapshot()["apps"]["newapp"]["state"] == "scoring"


@pytest.mark.parametrize("mode", ["observe", "auto_testing", "mixed", ""])
def test_constructor_rejects_retired_or_unknown_modes(mode):
    with pytest.raises(ValueError, match="mode"):
        RunStatus("run123", ["hinge"], min_labels=1, mode=mode)


def test_stop_reason_defaults_to_none_and_survives_serialization():
    # WS-opener-reason: an opener-exhaustion stop must be distinguishable from a plain
    # operator Stop in the SAME snapshot dict the hub's /api/status endpoint serves --
    # asdict(AppStatus) is the only path there (see status.snapshot()/app_view()), so
    # pinning it here catches any future field-list drift that would silently drop it.
    s = _mk()
    assert s.snapshot()["apps"]["bumble"]["stop_reason"] is None   # untouched app

    s.set_app("bumble", state="stopped", stop_reason="run budget reached")
    snap = s.snapshot()
    assert snap["apps"]["bumble"]["state"] == "stopped"
    assert snap["apps"]["bumble"]["stop_reason"] == "run budget reached"

    # app_view() (the per-app overlay/hub read) goes through the same asdict() call.
    view = s.app_view("bumble")
    assert view["app"]["stop_reason"] == "run budget reached"


def test_targeting_calibration_stop_kind_is_valid_and_serialized():
    s = _mk()
    s.set_app("hinge", state="stopped", stop_reason="fresh calibration required",
              stop_kind="targeting_calibration")
    app = s.app_view("hinge")["app"]
    assert app["stop_kind"] == "targeting_calibration"
    assert app["stop_reason"] == "fresh calibration required"


def test_app_view_includes_app_slice_and_global():
    s = _mk(labels=7)
    s.record_swipe("bumble", "like", 0.91)
    view = s.app_view("bumble")
    assert view["app"]["last_decision"] == "like"
    assert view["app"]["last_score"] == 0.91
    assert view["labels"] == 7                # global slice present too


def test_app_view_unknown_app_is_none():
    assert _mk().app_view("nope")["app"] is None


def test_concurrent_updates_are_consistent():
    s = _mk()

    def worker():
        for _ in range(200):
            s.record_swipe("bumble", "like", 0.5)
            s.inc_labels(1)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    snap = s.snapshot()
    assert snap["apps"]["bumble"]["swipes_run"] == 800
    assert snap["labels"] == 800


def test_hinge_render_status_is_noop():
    HingeDriver(_Cfg()).render_status({"any": "thing"})  # base no-op; no page to inject

def test_hinge_render_busy_is_noop():
    HingeDriver(_Cfg()).render_busy("x")     # base no-op
