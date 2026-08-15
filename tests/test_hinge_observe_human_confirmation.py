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
import time

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


def _pin_identity_advance(driver, monkeypatch, next_card):
    """Give a synthetic sheet->next-card flow the same two-frame positive identity evidence
    production observe mode requires before it may persist a LIKE."""
    import numpy as np

    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    driver._identity_sig = old_sig
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band",
                        lambda frame, _rect: new_sig if frame == next_card else old_sig)


def _sheet_frame():
    """A structurally valid, template-detectable Hinge inline-composer frame.

    The intent callback is deliberately gated on the whole composer: Hinge's own
    confirmation glyph inside a filled CTA, with the wide outlined comment input
    immediately above it.  Painting every independent proof keeps this regression
    on the same fail-closed production path as Hinge 9.134.0 rather than allowing
    a copied ``Send Like`` phrase to count as human intent.
    """
    import cv2
    import numpy as np

    canvas = np.full((2400, 1080), 249, dtype=np.uint8)
    comment = (95, 1597, 985, 1775)
    send = (390, 1807, 985, 1916)
    canvas[comment[1]:comment[1] + 2, comment[0]:comment[2]] = 222
    canvas[comment[3] - 2:comment[3], comment[0]:comment[2]] = 222
    canvas[send[1]:send[3], send[0]:send[2]] = 228
    template = hinge._load_template("hinge_send_like.png")
    assert template is not None
    height, width = template.shape
    x, y = 695, 1856
    canvas[y - height // 2:y - height // 2 + height,
           x - width // 2:x - width // 2 + width] = template
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def _deck_frame(*, invert_heart: bool = False):
    """A synthetic, template-detectable Hinge swipe deck (heart plus pass X).

    The heart glyph is HINGE_SPEC.templates["like"] (hinge_like_button.png), the real
    per-card like button (a white heart in a filled black circle) -- not hinge_heart.png,
    which is a DIFFERENT control (the "Which do we have in common" widget's outline heart;
    see hinge.py's templates dict comment). hinge_like_button.png is cropped directly from a
    live frame, so it is already at the live render's polarity and invert_heart now models a
    glyph that never actually occurs -- kept only so
    test_hinge_observe_deck_ready_rejects_inverted_heart below can pin that it is deliberately
    no longer accepted (see that test's docstring)."""
    import cv2
    import numpy as np

    canvas = np.full((2400, 1080), 120, dtype=np.uint8)
    like_glyph = hinge.HINGE_SPEC.templates["like"]
    for name, (x, y) in ((like_glyph, (930, 1600)),
                         ("hinge_pass_x.png", (130, 2030))):
        template = hinge._load_template(name)
        assert template is not None
        if invert_heart and name == like_glyph:
            template = np.bitwise_not(template)
        height, width = template.shape
        canvas[y - height // 2:y - height // 2 + height,
               x - width // 2:x - width // 2 + width] = template
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def test_hinge_observe_deck_ready_rejects_inverted_heart():
    """SUPERSEDES test_hinge_observe_deck_ready_accepts_live_inverted_heart_without_broadening_actions:
    that test encoded a workaround for the OLD "like" template (hinge_heart.png), which was
    the wrong polarity versus the live render, so _observe_glyph_visible tried both polarities
    to catch it. hinge_like_button.png (the current "like" template) is cropped directly from
    a live frame and is ALREADY at the live polarity (measured 0.815..1.000 correlation,
    uninverted, across 115 real frames -- see _LIKE_MATCH_THRESHOLD in hinge.py). Inverting it
    reproduces Hinge's OUTLINE heart -- the "Which do we have in common" widget's control --
    almost exactly, so _observe_glyph_visible now deliberately skips the inverted check for
    role == "like": accepting an inverted match here would quietly resurrect, in this
    perception-only path, the exact false positive the like-button template swap exists to
    fix. This pins the new, narrower contract."""
    driver = _hinge(_Adb([b"unused"]))

    assert driver._observe_deck_ready(_deck_frame(invert_heart=True)) is False


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

    Deliberately ``timeout=None`` plus a should_stop() poll BUDGET rather than a
    real wall-clock timeout. Every poll here runs genuine cv2 template matching
    against full 1080x2400 frames, and the scripted ``_split_diff`` sequence below
    needs an exact, fixed NUMBER of polls to exhaust (3 "changed" pairs, then the
    scripted default "unchanged") before the driver concludes the like sent. With
    a real timeout, that fixed amount of real CPU work races the wall clock: this
    test measured ~1.06-1.3s of actual matching against a `timeout=1.0`, so it
    already ran past its own deadline and only "passed" because the deadline is
    checked between polls, not against total elapsed time -- a machine/run just
    slightly slower (a colder cache, a busier CI runner, a different cv2 build)
    tips the LAST deadline check over 1.0s one poll early and the same, correct
    driver behavior reads back as `None` instead of `True`. That is exactly what
    happened in CI (this test has never passed there). The budget below removes
    the clock from the equation while still catching a genuine regression (the
    driver failing to ever resolve) as a `None` result instead of hanging CI.
    """
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    sheet = _sheet_frame()
    next_card = _deck_frame()
    adb = _Adb([b"card", sheet, next_card])
    callbacks = []
    polls = 0

    def stop_after_budget():
        nonlocal polls
        polls += 1
        # The success path needs exactly 6 should_stop() checks (1 in
        # _await_live_frame for `base`, 1 in the outer poll that sees the sheet,
        # then 4 in _await_like_resolved to exhaust the 3 scripted diffs). 200 is
        # a generous ceiling that only trips if the driver regresses to never
        # resolving.
        return polls > 200

    driver = _hinge(adb)
    _pin_identity_advance(driver, monkeypatch, next_card)
    result = driver.wait_for_decision(
        timeout=None, should_stop=stop_after_budget,
        on_like_intent=lambda active, anchor: callbacks.append((active, anchor)))

    assert result is True
    # The True notification carries the actual on-screen sheet frame as its anchor (the
    # picture the driver used to know a like was in progress); the clearing False
    # notification carries no anchor -- there is nothing left on screen to anchor once the
    # sheet has closed.
    assert callbacks == [(True, sheet), (False, None)]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_hinge_observe_waits_through_closed_sending_state_until_ready_deck(monkeypatch):
    """A closed sheet's processing screen is not a completed LIKE or a capture target.

    Same fix as the intent-publishing regression above and for the same reason:
    this loop does real cv2 matching on full-size frames, so a real wall-clock
    timeout races that (machine-speed-dependent) work against the clock instead
    of testing the driver's actual logic. Bound the polls instead.
    """
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    sheet = _sheet_frame()
    next_card = _deck_frame()
    # ``sending`` has no Send Like glyph and is deliberately a full-screen visual change.
    # The observer must keep polling it until both ordinary deck controls are stable.
    adb = _Adb([b"card", sheet, b"sending", next_card])
    callbacks = []
    polls = 0

    def stop_after_budget():
        nonlocal polls
        polls += 1
        # The success path needs 4 should_stop() checks; 200 is a generous
        # ceiling that only trips on a genuine "never resolves" regression.
        return polls > 200

    driver = _hinge(adb)
    _pin_identity_advance(driver, monkeypatch, next_card)
    assert driver.wait_for_decision(
        timeout=None, should_stop=stop_after_budget,
        on_like_intent=lambda active, anchor: callbacks.append((active, anchor))) is True
    assert callbacks == [(True, sheet), (False, None)]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_hinge_observe_cancelled_sheet_clears_intent_and_keeps_waiting(monkeypatch):
    """Dismissing the compose sheet is neither a pass nor a persisted like."""
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0)))
    sheet = _sheet_frame()
    adb = _Adb([b"card", sheet, b"card", b"card"])
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
        on_like_intent=lambda active, anchor: callbacks.append((active, anchor)))

    assert result is None
    assert callbacks == [(True, sheet), (False, None)]
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
        on_like_intent=lambda active, anchor: callbacks.append((active, anchor)),
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
    _pin_identity_advance(driver, monkeypatch, b"next-card")
    monkeypatch.setattr(driver, "_observe_like_sheet_visible",
                        lambda frame: frame in {b"sheet", b"typing"})
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda frame: frame == b"next-card")
    callbacks = []

    assert driver.wait_for_decision(
        timeout=1.0,
        on_like_intent=lambda active, anchor: callbacks.append((active, anchor))) is True
    # The True notification's anchor is the sheet frame that proved the sheet was up.
    assert callbacks == [(True, b"sheet"), (False, None)]
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


def test_notify_observe_like_intent_passes_frame_as_anchor_and_none_on_clear():
    """Direct unit coverage of the anchor contract every wait_for_decision call site relies
    on: the active=True notification carries the EXACT on-screen frame it was given (the
    picture of the item the human's own tap opened a comment sheet for, which is what observe's
    anchored suggestion is grounded in -- and, since the auto-mode repair hatch was removed on
    2026-08-12, the only consumer of the anchor left anywhere), and the clearing active=False
    notification always carries anchor=None -- there is nothing left on screen to anchor a
    picture of once the sheet has closed."""
    drv = _hinge(_Adb([b"unused"]))
    calls = []

    drv._notify_observe_like_intent(
        lambda active, anchor: calls.append((active, anchor)), True, b"the-real-on-screen-frame")
    drv._notify_observe_like_intent(
        lambda active, anchor: calls.append((active, anchor)), False)

    assert calls == [(True, b"the-real-on-screen-frame"), (False, None)]


def test_notify_observe_like_intent_callback_exception_is_printed_not_swallowed(capsys):
    """A callback that raises must not vanish silently -- see _notify_observe_like_intent's
    own docstring: the OLD bare `except Exception: pass` meant a stale callback signature
    (an ordinary TypeError after some future refactor) made suggestions disappear forever
    with zero evidence anything was ever wrong, nothing printed, nothing to grep for. The
    failure must stay non-fatal (this method must not raise out to its caller -- observation
    has to keep running with no suggestion rather than stop) but it must be VISIBLE."""
    drv = _hinge(_Adb([b"unused"]))

    def broken_callback(active, anchor):
        raise ValueError("suggestion renderer exploded")

    drv._notify_observe_like_intent(broken_callback, True, b"frame")   # must not raise

    printed = capsys.readouterr().out
    assert "ValueError" in printed
    assert "suggestion renderer exploded" in printed


def test_hinge_observe_broken_callback_does_not_break_observation_but_failure_is_printed(monkeypatch, capsys):
    """End-to-end version of the unit test above: a callback that raises on every call must
    not stop wait_for_decision from resolving the human's actual like/pass -- the run must
    continue observing normally -- but each failure must still be printed to stdout, so a
    broken suggestion hook is loud instead of just quietly producing no suggestions ever
    again."""
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    sheet = _sheet_frame()
    next_card = _deck_frame()
    adb = _Adb([b"card", sheet, next_card])
    polls = 0

    def stop_after_budget():
        nonlocal polls
        polls += 1
        return polls > 200

    def broken_callback(active, anchor):
        raise ValueError("suggestion renderer exploded")

    driver = _hinge(adb)
    _pin_identity_advance(driver, monkeypatch, next_card)
    result = driver.wait_for_decision(
        timeout=None, should_stop=stop_after_budget, on_like_intent=broken_callback)

    assert result is True             # observation still resolves the real decision
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []
    printed = capsys.readouterr().out
    assert printed.count("ValueError") >= 1
    assert "suggestion renderer exploded" in printed


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
    # Declared, because worker.py reads it with `getattr(..., "disabled", True)` -- a service
    # that does not say is treated as switched off, which is the safe default (never spend on a
    # service nobody vouched for) and which a fake has to opt out of explicitly.
    disabled = False
    last_skip_reason = None

    def __init__(self):
        self.calls = []
        self.advisory_seen = []   # records advisory= from every call -- see change A's tests
        self.anchor_seen = []     # records anchor= from every call -- always None since doc 5.9
        self.items_seen = []      # records items= -- the numbered crops BOTH modes now send
        # worker.py's on_like_intent now forwards UNCONDITIONALLY (see its own docstring: a
        # client/service that cannot accept this kwarg must fail LOUDLY, not have it silently
        # dropped). A fake missing this parameter entirely used to raise a TypeError right at
        # the call boundary -- before self.calls.append() ever ran -- so opener.calls stayed
        # empty and the caller never learned why.

    def maybe_opener(self, run_id, app, profile, *, anchor=None, items=None,
                     should_stop=None, advisory=False):
        self.calls.append((run_id, app, profile))
        self.advisory_seen.append(advisory)
        self.anchor_seen.append(anchor)
        self.items_seen.append(items)
        # A REAL item number (ops/OPENER-REDESIGN.md 5.9): the inversion's whole output is
        # "like item N plus this text", and index=0 (ITEM_INDEX_ABSENT) is what observe now
        # refuses to show, so a fake returning it would exercise the warning path by accident.
        return OpenerPick("Your trail photo looks like a great weekend plan.", index=2,
                          item_description="the trail photo")


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
        # An ENUMERATED capture, because doc 5.9's observe now sends the numbered crops auto
        # sends and refuses to suggest anything without them.
        self.profile = Profile(photos=[b"photo"], meta={"app": "hinge"}, name="Ada",
                               items=(b"item-1", b"item-2", b"item-3"),
                               item_context=(b"vitals",))
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

    def _await_suggestion(self, timeout=10.0):
        """The human looks at the hub before tapping.

        Doc 5.9 generates on its own thread so READY can be published immediately, so a fake that
        tapped the instant wait_for_decision was entered would be racing it. `opener_pending`
        going False is the worker's own "this card's suggestion has settled" signal."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app = self.status.app_view("hinge")["app"]
            if app is not None and not app.get("opener_pending"):
                return
            time.sleep(0.002)

    def observe_item_mismatch(self, sheet, model_item_index):
        """The human opened the item the suggestion named. Doc 5.9's guard, satisfied."""
        return ""

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        assert timeout is None
        self.wait_calls += 1
        self._await_suggestion()
        if self.outcome == "like":
            assert on_like_intent is not None, "worker must listen for Hinge's sheet-open stage"
            on_like_intent(True, b"the-open-comment-sheet")
            app = self.status.app_view("hinge")["app"]
            self.before_confirmation = {
                "profiles": list(self.store.profiles),
                "labels": list(self.store.labels),
                "decisions": list(self.store.decisions),
                "opener_suggestion": app.get("opener_suggestion"),
                "opener_item": app.get("opener_item"),
                "state": app["state"],
            }
            on_like_intent(False, None)
            self.done = True
            return True
        if self.outcome == "dismiss":
            assert on_like_intent is not None
            if self.wait_calls == 1:
                on_like_intent(True, b"the-open-comment-sheet")
                on_like_intent(False, None)
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
    # Doc 5.9's inversion: the suggestion is on the hub before the heart is tapped, so what is
    # still live once the sheet is open is the SAME suggestion plus the item number it names --
    # not one generated in response to the tap.
    assert driver.before_confirmation == {
        "profiles": [], "labels": [], "decisions": [],
        "opener_suggestion": "Your trail photo looks like a great weekend plan.",
        "opener_item": 2,
        "state": "waiting_for_send",
    }
    assert [row[2] for row in store.labels] == [True]
    assert [row[2] for row in store.decisions] == ["like"]
    assert driver.action_calls == []
    assert driver.closed
    # The request shape, which is the property that makes observe a canary for auto: the
    # numbered crops, and no anchor at all (doc 5.9 retired the anchored shape's last caller).
    assert opener.anchor_seen == [None]
    assert [i.items for i in opener.items_seen] == [(b"item-1", b"item-2", b"item-3")]


def test_observe_hinge_dismissed_sheet_does_not_persist_or_call_actions():
    driver, store, opener, _status = _run_one_observe("dismiss")

    # One per CAPTURE, and a dismissal recaptures -- doc 5.9 asks before the human acts, so it
    # has to ask again for the re-read, which the loop cannot tell from a new card. Suggestions
    # may be prepared; none is ever sent.
    assert len(opener.calls) == 2
    assert store.profiles == []
    assert store.labels == []
    assert store.decisions == []
    assert driver.action_calls == []
    assert driver.capture_calls == 2            # dismissal returns to the same profile's decision loop
    # Backing out of the sheet returns the hub to the pre-tap INSTRUCTION rather than clearing
    # it: the card has not changed, so "like item 2, and here is the text" is still the advice,
    # and the human may go and open item 2 next. What a dismiss drops is only the evidence about
    # what they had opened.
    assert driver.after_dismiss == {
        "opener_suggestion": "Your trail photo looks like a great weekend plan.",
        "state": "waiting",
    }
    assert driver.closed


def test_observe_hinge_persists_pass_only_after_human_x_advances_profile():
    driver, store, opener, _status = _run_one_observe("pass")

    assert driver.before_pass_advance == {"profiles": [], "labels": [], "decisions": []}
    # ONE call, for a profile the human then PASSED. Doc 5.9's inversion has to ask before it
    # knows what the human will do, so observe now spends an opener call on every card rather
    # than only on hearted ones -- the honest cost of generating before the tap, recorded here
    # rather than left to be discovered as a quota surprise.
    assert len(opener.calls) == 1
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
