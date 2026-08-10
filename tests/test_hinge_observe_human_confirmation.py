"""Regression coverage for Hinge's human-confirmed observe flow.

Observe is passive.  In particular, a Hinge heart only opens the app's compose
sheet: it is *not* a completed LIKE until the owner manually sends it.  The
worker may suggest an opener in the hub while that sheet is open, but must never
touch the heart, X, text field, or Send Like control itself.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading

import pytest

from operation_love.drivers import hinge
from operation_love.drivers.base import DatingAppDriver
from operation_love.drivers.hinge import HingeDriver
from operation_love.opener.service import OpenerPick
from operation_love.perception.capture import Profile
from operation_love.status import RunStatus
from operation_love.worker import Worker


class _Cfg:
    apps = {"hinge": {"serial": "pixel", "halt_on_error": False}}


class _Adb:
    """Minimal ADB double which makes any observe-mode action evident."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0
        self.taps = []
        self.swipes = []
        self.texts = []

    def screen_size(self):
        return (1080, 2400)

    def screencap(self):
        value = self.frames[min(self.i, len(self.frames) - 1)]
        if self.i < len(self.frames) - 1:
            self.i += 1
        return value

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, *args, **kwargs):
        self.swipes.append((args, kwargs))

    def text(self, value):
        self.texts.append(value)


def _hinge(adb):
    driver = HingeDriver(_Cfg())
    driver._adb = adb
    driver._touch = adb
    return driver


def _sheet_frame():
    """A real, template-detectable Hinge Send Like sheet frame.

    The intent callback is deliberately gated on seeing Hinge's own confirmation
    glyph, not merely any bottom-half animation.  Supplying the glyph here keeps
    this regression on that safe production path.
    """
    import cv2
    import numpy as np

    canvas = np.full((2400, 1080), 120, dtype=np.uint8)
    template = hinge._load_template("hinge_send_like.png")
    assert template is not None
    height, width = template.shape
    x, y = 540, 1300
    canvas[y - height // 2:y - height // 2 + height,
           x - width // 2:x - width // 2 + width] = template
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def _deck_frame(*, invert_heart: bool = False):
    """A synthetic, template-detectable Hinge swipe deck (heart plus pass X)."""
    import cv2
    import numpy as np

    canvas = np.full((2400, 1080), 120, dtype=np.uint8)
    for name, (x, y) in (("hinge_heart.png", (930, 1600)),
                         ("hinge_pass_x.png", (130, 2030))):
        template = hinge._load_template(name)
        assert template is not None
        if invert_heart and name == "hinge_heart.png":
            template = np.bitwise_not(template)
        height, width = template.shape
        canvas[y - height // 2:y - height // 2 + height,
               x - width // 2:x - width // 2 + width] = template
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def test_hinge_observe_deck_ready_accepts_live_inverted_heart_without_broadening_actions():
    """The passive ready guard supports Hinge's white-outline heart on a dark circle."""
    driver = _hinge(_Adb([b"unused"]))

    assert driver._observe_deck_ready(_deck_frame(invert_heart=True)) is True


class _ScriptedDiff:
    def __init__(self, *pairs):
        self.pairs = list(pairs)

    def __call__(self, _before, _after):
        return self.pairs.pop(0) if self.pairs else (0.0, 0.0)


def test_hinge_observe_reports_sheet_intent_before_manual_send_without_touching_app(monkeypatch):
    """Heart -> sheet must publish intent, then only a real advance becomes LIKE.

    The fake frames model a human tapping the heart, seeing the compose sheet,
    entering the suggested text manually, and tapping Hinge's Send Like.  The
    observe driver may inspect frames and invoke the callback, but may not emit
    any input command itself.
    """
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    next_card = _deck_frame()
    adb = _Adb([b"card", _sheet_frame(), next_card])
    callbacks = []

    result = _hinge(adb).wait_for_decision(
        timeout=1.0, on_like_intent=lambda active: callbacks.append(active))

    assert result is True
    assert callbacks == [True, False]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_hinge_observe_waits_through_closed_sending_state_until_ready_deck(monkeypatch):
    """A closed sheet's processing screen is not a completed LIKE or a capture target."""
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    sheet = _sheet_frame()
    next_card = _deck_frame()
    # ``sending`` has no Send Like glyph and is deliberately a full-screen visual change.
    # The observer must keep polling it until both ordinary deck controls are stable.
    adb = _Adb([b"card", sheet, b"sending", next_card])
    callbacks = []

    assert _hinge(adb).wait_for_decision(
        timeout=1.0, on_like_intent=lambda active: callbacks.append(active)) is True
    assert callbacks == [True, False]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_hinge_observe_cancelled_sheet_clears_intent_and_keeps_waiting(monkeypatch):
    """Dismissing the compose sheet is neither a pass nor a persisted like."""
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0)))
    adb = _Adb([b"card", _sheet_frame(), b"card", b"card"])
    callbacks = []
    stop_checks = 0

    def stop_after_cancel():
        nonlocal stop_checks
        stop_checks += 1
        # First outer poll and the sheet-resolution poll run; stop only after
        # the driver has observed that the sheet returned to the same card.
        # _await_live_frame also polls once before the outer loop starts, so
        # stopping on the fourth check lets sheet resolution clear intent first.
        return stop_checks >= 4

    result = _hinge(adb).wait_for_decision(
        timeout=None, should_stop=stop_after_cancel,
        on_like_intent=lambda active: callbacks.append(active))

    assert result is None
    assert callbacks == [True, False]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_hinge_observe_bottom_animation_without_send_like_never_surfaces_opener(monkeypatch):
    """A bottom-only change is not enough evidence to spend opener budget."""
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0)))
    adb = _Adb([b"card", b"bottom-animation", b"card", b"card"])
    driver = _hinge(adb)
    monkeypatch.setattr(driver, "_observe_like_sheet_visible", lambda _frame: False)
    callbacks = []
    checks = 0

    def stop_after_candidate_is_cancelled():
        nonlocal checks
        checks += 1
        return checks >= 4

    assert driver.wait_for_decision(
        timeout=None, should_stop=stop_after_candidate_is_cancelled,
        on_like_intent=lambda active: callbacks.append(active),
    ) is None
    assert callbacks == []
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_hinge_observe_waits_through_keyboard_motion_until_send_like_closes(monkeypatch):
    """Typing may move the card behind a still-open sheet; it is not a sent like."""
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hinge, "_split_diff",
        _ScriptedDiff((2.0, 50.0), (55.0, 80.0), (55.0, 80.0), (55.0, 80.0)),
    )
    adb = _Adb([b"card", b"sheet", b"typing", b"next-card"])
    driver = _hinge(adb)
    monkeypatch.setattr(driver, "_observe_like_sheet_visible",
                        lambda frame: frame in {b"sheet", b"typing"})
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda frame: frame == b"next-card")
    callbacks = []

    assert driver.wait_for_decision(timeout=1.0,
                                    on_like_intent=lambda active: callbacks.append(active)) is True
    assert callbacks == [True, False]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


class _Store:
    def __init__(self):
        self.profiles = []
        self.labels = []
        self.decisions = []

    def record_profile(self, run_id, app, profile_id, liked, **kwargs):
        self.profiles.append((run_id, app, profile_id, liked, kwargs))
        return True

    def add_label(self, run_id, app, liked, embedding, **kwargs):
        self.labels.append((run_id, app, liked, embedding, kwargs))

    def record_decision(self, run_id, app, decision, score, **kwargs):
        self.decisions.append((run_id, app, decision, score, kwargs))

    def load_labels(self):
        return []

    def flush(self):
        pass

    def close(self):
        pass


class _Decider:
    def embed(self, _profile):
        return [0.2, 0.8]

    def retrain(self, _store):
        return True


class _OpenerService:
    stop_requested = False

    def __init__(self):
        self.calls = []
        self.advisory_seen = []   # records advisory= from every call -- see change A's tests

    def maybe_opener(self, run_id, app, profile, *, should_stop=None, advisory=False):
        self.calls.append((run_id, app, profile))
        self.advisory_seen.append(advisory)
        return OpenerPick("Your trail photo looks like a great weekend plan.", index=0)


class _Pacing:
    swipe_delay_s = 0.0


class _HumanHingeDriver(DatingAppDriver):
    """A passive driver with a scripted owner outcome for one captured card."""

    accepts_opener = True
    supports_observe_like_intent = True

    def __init__(self, outcome, store, status):
        self.outcome = outcome
        self.store = store
        self.status = status
        self.profile = Profile(photos=[b"photo"], meta={"app": "hinge"})
        self.done = False
        self.closed = False
        self.action_calls = []
        self.before_confirmation = None
        self.before_pass_advance = None
        self.after_dismiss = None
        self.intent_state = []
        self.capture_calls = 0
        self.wait_calls = 0

    def open_session(self):
        pass

    def out_of_profiles(self):
        if self.outcome == "dismiss":
            # A cancellation must make the worker capture/wait on the same card
            # again, rather than quietly turning it into a pass.
            return self.wait_calls >= 2
        return self.done

    def current_profile(self):
        self.capture_calls += 1
        return self.profile

    def next_profile(self):
        return self.current_profile()

    # These are deliberately tripwires: observe must never ask a driver to act,
    # type, or send on the owner's behalf.
    def like(self, *args, **kwargs):
        self.action_calls.append(("like", args, kwargs))
        raise AssertionError("observe invoked driver.like()")

    def dislike(self):
        self.action_calls.append(("dislike", (), {}))
        raise AssertionError("observe invoked driver.dislike()")

    def pass_(self):
        self.action_calls.append(("pass_", (), {}))
        raise AssertionError("observe invoked driver.pass_()")

    def type(self, value):
        self.action_calls.append(("type", (value,), {}))
        raise AssertionError("observe typed an opener")

    def send(self):
        self.action_calls.append(("send", (), {}))
        raise AssertionError("observe sent a like")

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        assert timeout is None
        self.wait_calls += 1
        if self.outcome == "like":
            assert on_like_intent is not None, "worker must listen for Hinge's sheet-open stage"
            on_like_intent(True)
            app = self.status.app_view("hinge")["app"]
            self.before_confirmation = {
                "profiles": list(self.store.profiles),
                "labels": list(self.store.labels),
                "decisions": list(self.store.decisions),
                "opener_suggestion": app.get("opener_suggestion"),
                "state": app["state"],
            }
            on_like_intent(False)
            self.done = True
            return True
        if self.outcome == "dismiss":
            assert on_like_intent is not None
            if self.wait_calls == 1:
                on_like_intent(True)
                on_like_intent(False)
                app = self.status.app_view("hinge")["app"]
                self.after_dismiss = {
                    "opener_suggestion": app.get("opener_suggestion"),
                    "state": app["state"],
                }
            return None
        if self.outcome == "pass":
            self.before_pass_advance = {
                "profiles": list(self.store.profiles),
                "labels": list(self.store.labels),
                "decisions": list(self.store.decisions),
            }
            self.done = True                 # represents the owner's X and card advance
            return False
        raise AssertionError(f"unknown outcome {self.outcome}")

    def render_busy(self, message=None):
        pass

    def close(self):
        self.closed = True


def _run_one_observe(outcome):
    store = _Store()
    status = RunStatus("run", ["hinge"], min_labels=1, mode="observe")
    driver = _HumanHingeDriver(outcome, store, status)
    opener = _OpenerService()
    Worker("hinge", driver, _Decider(), opener, store, "run", _Pacing(),
           threading.Event(), mode="observe", status=status).run()
    return driver, store, opener, status


def test_observe_hinge_suggests_opener_before_human_send_then_persists_like():
    driver, store, opener, _status = _run_one_observe("like")

    assert opener.calls and len(opener.calls) == 1
    assert driver.before_confirmation == {
        "profiles": [], "labels": [], "decisions": [],
        "opener_suggestion": "Your trail photo looks like a great weekend plan.",
        "state": "waiting_for_send",
    }
    assert [row[2] for row in store.labels] == [True]
    assert [row[2] for row in store.decisions] == ["like"]
    assert driver.action_calls == []
    assert driver.closed


def test_observe_hinge_dismissed_sheet_does_not_persist_or_call_actions():
    driver, store, opener, _status = _run_one_observe("dismiss")

    assert len(opener.calls) == 1              # suggestion may be prepared, never sent
    assert store.profiles == []
    assert store.labels == []
    assert store.decisions == []
    assert driver.action_calls == []
    assert driver.capture_calls == 2            # dismissal returns to the same profile's decision loop
    assert driver.after_dismiss == {"opener_suggestion": None, "state": "waiting"}
    assert driver.closed


def test_observe_hinge_persists_pass_only_after_human_x_advances_profile():
    driver, store, opener, _status = _run_one_observe("pass")

    assert driver.before_pass_advance == {"profiles": [], "labels": [], "decisions": []}
    assert opener.calls == []
    assert [row[2] for row in store.labels] == [False]
    assert [row[2] for row in store.decisions] == ["dislike"]
    assert driver.action_calls == []
    assert driver.closed


def _extract_js_function(source: str, name: str) -> str:
    match = re.search(rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*{{", source)
    assert match, f"function {name} not found"
    start = match.end() - 1
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():i + 1]
    raise AssertionError(f"function {name} is not balanced")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to execute hub JavaScript")
def test_hub_replaces_swipe_instruction_with_manual_hinge_send_instruction_when_opener_pending():
    """The only operator instruction must describe the real manual next step."""
    from operation_love.hub import _PAGE

    script = "\n".join((
        "let banner = {style:{display:''}, innerHTML:''};",
        "function $(selector){ return selector === '#swipebanner' ? banner : null; }",
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "selectObserveApps"),
        _extract_js_function(_PAGE, "renderSwipe"),
        "renderSwipe({running:true,status:{apps:{hinge:{app:'hinge',mode:'observe',state:'waiting_for_send',opener_suggestion:'Try the taco place in your photo?'}}}});",
        "console.log(JSON.stringify(banner));",
    ))
    run = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True,
                         timeout=10, check=True)
    banner = json.loads(run.stdout)

    assert "Try the taco place in your photo?" in banner["innerHTML"]
    assert "Send Like" in banner["innerHTML"]
    assert "SWIPE hinge now" not in banner["innerHTML"]
