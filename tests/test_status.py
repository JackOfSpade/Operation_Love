"""RunStatus live-status bus + the drivers' render_status overlay hook.

Pure offline: the status object is plain Python; the Bumble overlay is exercised
with a fake page that records the evaluate() call (the real HUD render is visual,
seen live in the browser). The Hinge driver inherits the base no-op.
"""
import threading

from operation_love.status import RunStatus
from operation_love.drivers.bumble import BumbleDriver, _BUSY_JS, _OVERLAY_JS
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


def test_labels_needed_floors_at_zero():
    assert _mk(labels=50).snapshot()["labels_needed"] == 0


def test_set_global():
    s = _mk()
    s.set_global(ranker_ready=True, budget_spent=1.2345, running=False)
    snap = s.snapshot()
    assert snap["ranker_ready"] is True
    assert snap["budget_spent"] == 1.2345
    assert snap["running"] is False


def test_set_app_autocreates_unknown_app():
    s = _mk()
    s.set_app("newapp", state="scoring")
    assert s.snapshot()["apps"]["newapp"]["state"] == "scoring"


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


# --- render_status overlay hook -------------------------------------------
def test_bumble_render_status_paints_overlay():
    calls = []

    class P:
        def evaluate(self, script, arg=None):
            calls.append((script, arg))

    drv = BumbleDriver(_Cfg())
    drv.page = P()
    drv.render_status({"app": {"state": "waiting"}, "labels": 3})
    assert calls and calls[0][0] is _OVERLAY_JS         # injects the HUD script
    assert calls[0][1]["labels"] == 3                   # with the snapshot


def test_bumble_render_status_no_page_is_noop():
    BumbleDriver(_Cfg()).render_status({"x": 1})        # page is None -> must not raise


def test_bumble_render_status_swallows_errors():
    class P:
        def evaluate(self, script, arg=None):
            raise RuntimeError("page navigated")

    drv = BumbleDriver(_Cfg())
    drv.page = P()
    drv.render_status({"x": 1})                         # must not propagate


def test_hinge_render_status_is_noop():
    HingeDriver(_Cfg()).render_status({"any": "thing"})  # base no-op; no page to inject


def test_bumble_render_busy_shows_and_hides():
    calls = []

    class P:
        def evaluate(self, script, arg=None):
            calls.append((script, arg))

    drv = BumbleDriver(_Cfg())
    drv.page = P()
    drv.render_busy("processing…")          # show
    drv.render_busy(None)                    # hide
    assert all(c[0] is _BUSY_JS for c in calls)
    assert calls[0][1] == "processing…" and calls[1][1] is None


def test_bumble_render_busy_no_page_is_noop():
    BumbleDriver(_Cfg()).render_busy("x")    # page is None -> must not raise


def test_hinge_render_busy_is_noop():
    HingeDriver(_Cfg()).render_busy("x")     # base no-op


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
