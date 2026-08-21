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
    return RunStatus("run123", ["bumble", "hinge"], min_labels=40, mode="observe", **kw)


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
    kwargs = {"min_labels": 40, "mode": "observe", field: value}
    with pytest.raises(ValueError, match=field):
        RunStatus("run123", ["hinge"], **kwargs)


@pytest.mark.parametrize(("args", "kwargs", "message"), [
    (("", ["hinge"]), {"min_labels": 1, "mode": "observe"}, "run_id"),
    ((" run ", ["hinge"]), {"min_labels": 1, "mode": "observe"}, "run_id"),
    (("run", ["hinge"]), {"min_labels": 1, "mode": "mixed"}, "mode"),
    (("run", "hinge"), {"min_labels": 1, "mode": "observe"}, "apps"),
    (("run", (app for app in ["hinge"])),
     {"min_labels": 1, "mode": "observe"}, "apps"),
    (("run", ["hinge", "hinge"]), {"min_labels": 1, "mode": "observe"}, "duplicate"),
    (("run", [" hinge "]), {"min_labels": 1, "mode": "observe"}, "app names"),
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
            "run123", ["hinge"], min_labels=1, mode="observe", budget_cap=budget_cap,
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


def test_observe_opener_suggestion_is_published_in_app_snapshot():
    s = _mk()
    s.set_app("hinge", state="waiting_for_send", opener_suggestion="A curious question?")
    app = s.snapshot()["apps"]["hinge"]
    assert app["state"] == "waiting_for_send"
    assert app["opener_suggestion"] == "A curious question?"

    # A dismissal/decision/new card clears it with its state transition, even when a worker
    # need not remember to include a redundant opener_suggestion=None field.
    s.set_app("hinge", state="waiting")
    app = s.snapshot()["apps"]["hinge"]
    assert app["state"] == "waiting" and app["opener_suggestion"] is None

    s.set_app("hinge", state="waiting_for_send", opener_suggestion="Another opener")
    s.record_swipe("hinge", "like")
    assert s.snapshot()["apps"]["hinge"]["opener_suggestion"] is None


def test_suggesting_state_clears_a_stale_opener_suggestion():
    """A WAIT-style state transition must clear any stale opener_suggestion left over from a
    PRIOR card automatically, or a re-render between the flip and the real suggestion landing
    could momentarily show the previous card's text next to the new one.

    'suggesting' is the case pinned here because it is the one that has NO producer left (doc
    5.9's inversion generates on its own thread, before the human acts, so there is nothing to
    block on); pinning the branch keeps it honest for the older snapshots that can still carry
    it, rather than letting it rot quietly."""
    s = _mk()
    s.set_app("hinge", state="waiting_for_send", opener_suggestion="stale suggestion")
    s.set_app("hinge", state="suggesting")
    app = s.snapshot()["apps"]["hinge"]
    assert app["state"] == "suggesting" and app["opener_suggestion"] is None


def test_blocked_terminal_state_clears_stale_opener_guidance():
    status = _mk()
    status.set_app(
        "hinge", state="waiting_for_send", opener_suggestion="stale suggestion",
        opener_item=3, opener_pending=True,
    )
    status.set_app("hinge", state="blocked", stop_kind="deck_blocked")
    app = status.snapshot()["apps"]["hinge"]
    assert app["state"] == "blocked"
    assert app["opener_suggestion"] is None
    assert app["opener_item"] is None
    assert app["opener_pending"] is False


def test_a_state_transition_clears_every_opener_field_together():
    """They describe ONE suggestion for one card between them -- its text, what it claims to be
    about, which item it names, that item's description, why there is no text, and whether one
    is still coming -- so any of them outliving the rest is the exact lie the auto-clear exists
    to prevent. Since doc 5.9's inversion a survivor is worse than a stale caption: "like item
    3" pointing at a profile that is no longer on screen is an instruction, not a note.

    Asserted against status._OPENER_FIELDS itself rather than a re-typed list, so adding a field
    to the set without teaching the net about it cannot pass."""
    from operation_love.status import _OPENER_FIELDS

    s = _mk()
    s.set_app("hinge", state="waiting_for_send", opener_suggestion="about her dog",
              opener_referenced="her dog", opener_item=3,
              opener_item_description="the ridgeline photo", opener_warning="stale warning",
              opener_pending=True)

    s.set_app("hinge", state="capturing")     # a transition that does NOT name opener_suggestion
    app = s.snapshot()["apps"]["hinge"]
    assert {name: app[name] for name in _OPENER_FIELDS} == _OPENER_FIELDS


def test_set_app_does_not_autoclear_the_other_opener_fields_when_suggestion_is_named():
    """RunStatus.set_app's own safety net -- auto-clearing the whole opener set on a normal
    state transition -- fires ONLY when opener_suggestion is ABSENT from the update dict (see
    set_app's own comment). Any caller that explicitly passes opener_suggestion, even to clear
    it, opts itself OUT of that safety net and must therefore name the rest too, or they survive
    untouched exactly as pinned here. worker.py has two call sites that do this (the observe
    loop's per-card reset and its `finally`); an adversarial review found both had been passing
    opener_suggestion alone, leaving the others stale. This test pins the RunStatus mechanism
    responsible for that gap, so the reason callers must name all of them lives in a test, not
    just a comment -- and status.cleared_opener_fields() is what those callers splat so the list
    cannot drift."""
    s = _mk()
    s.set_app("hinge", state="waiting_for_send", opener_suggestion="about her dog",
              opener_referenced="her dog", opener_item=3)

    # A state transition ("waiting") that WOULD normally auto-clear the whole set -- except this
    # call explicitly supplies opener_suggestion, so set_app must leave the others exactly as
    # they were; only what the caller literally set changes.
    s.set_app("hinge", state="waiting", opener_suggestion=None)
    app = s.snapshot()["apps"]["hinge"]
    assert app["state"] == "waiting"
    assert app["opener_suggestion"] is None          # explicitly cleared by the caller
    assert app["opener_referenced"] == "her dog"      # NOT auto-cleared -- caller opted out
    assert app["opener_item"] == 3                    # NOT auto-cleared -- caller opted out


def test_cleared_opener_fields_is_the_whole_set_and_a_fresh_dict_each_call():
    """The helper the two opting-out callers splat. It must cover the set completely (a caller
    that splats it is trusting it to) and must never hand out the module's own dict, which a
    caller adding a key to its update would otherwise mutate for every future call."""
    from operation_love.status import AppStatus, _OPENER_FIELDS, cleared_opener_fields

    first = cleared_opener_fields()
    assert first == _OPENER_FIELDS and first is not _OPENER_FIELDS
    first["opener_suggestion"] = "mutated"
    assert cleared_opener_fields()["opener_suggestion"] is None
    # Every name in the set is a real AppStatus field, or splatting it would raise nothing and
    # silently write an attribute the hub never reads.
    blank = AppStatus(app="hinge")
    for name in _OPENER_FIELDS:
        assert hasattr(blank, name)


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
