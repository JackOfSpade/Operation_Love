"""Regression coverage for Hinge's passive sheet and decision detector.

These lower-level driver tests ensure a human-opened compose sheet is recognized
without the detector issuing touches, text, or Send Like actions itself.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from operation_love.drivers import hinge
from operation_love.drivers.hinge import HingeDriver


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


def test_hinge_observe_recognizes_priority_like_with_geometry_but_keeps_typing_strict(monkeypatch):
    """The passive observer supports Hinge's ``Send Priority Like`` CTA variant only.

    The reduced template score represents the live variant, where inserting ``Priority`` into
    the label drops the old literal ``Send Like`` template from the typing threshold.  The
    fallback remains safe because ``locate_inline_composer`` must still prove the CTA and input
    geometry, and its result is intentionally discarded rather than offered to auto mode.
    """
    from operation_love.drivers.like_composer import ComposerSurface, Rect

    driver = _hinge(_Adb([b"unused"]))
    checked = []
    surface = ComposerSurface("hinge_inline_v1", Rect(95, 1124, 985, 1302),
                              Rect(390, 1334, 985, 1443), (630, 1383))
    # Both stand-ins accept `image=` (2026-08-23 perf pass: `_locate_observed_inline_composer`
    # now decodes the frame once and threads it into both calls below to skip a second
    # cv2.imdecode of the same bytes -- see that method's own docstring) and ignore it, since
    # this test is about which THRESHOLD the fallback call uses, not about the shared decode.
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame, **_kw: None)

    def low_confidence_surface(frame, template, *, threshold, **_kw):
        checked.append((frame, template, threshold))
        assert threshold == 0.68
        return surface

    monkeypatch.setattr(hinge, "locate_inline_composer", low_confidence_surface)

    assert driver._observe_like_sheet_visible(b"priority-composer") is True
    assert driver._observe_like_sheet_detection == "priority_variant"
    assert checked == [(b"priority-composer", driver._template("confirm"), 0.68)]


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
    # The first True carries the frame that proved the sheet was up.  While the composer remains
    # visible, a later keyboard-settled frame refreshes that anchor so a provisional item read can
    # correct itself; it is still one open intent, cleared exactly once after the sheet closes.
    assert callbacks == [(True, b"sheet"), (True, b"typing"), (False, None)]
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
def test_hub_training_checkpoint_shows_the_typed_opener_with_like_and_dislike():
    """Training replaces the retired on-phone Observe cue with one Hub decision surface."""
    from operation_love.hub import _PAGE

    script = "\n".join((
        "let panel = {style:{display:''}, innerHTML:''};",
        "let layout = {classList:{toggle(){}}};",
        "function $(selector){ return selector === '#trainingpanel' ? panel : null; }",
        "const document = {querySelector(){ return layout; }};",
        "let _trainingCheckpoint = null; let _trainingRequest = 0;",
        "let _trainingActionBusy = false; let _trainingBusyKey = ''; let _trainingBusyRequest = 0;",
        "let _trainingImageKey = ''; let _trainingImageIndex = 0;",
        "const _trainingIdempotency = new Map();",
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
        "renderTrainingCheckpoint({run_id:'r1',app:'hinge',profile_token:'p1',approval_token:'a1',"
        "image_data_url:'data:image/png;base64,AA==',opener:'Try the taco place in your photo?',"
        "item:2,item_description:'taco photo',pending:true,phase:'waiting_training_decision',action:'ready'});",
        "console.log(JSON.stringify(panel));",
    ))
    run = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True,
                         timeout=10, check=True)
    panel = json.loads(run.stdout)

    assert "Try the taco place in your photo?" in panel["innerHTML"]
    assert ">Like</button>" in panel["innerHTML"]
    assert ">Dislike</button>" in panel["innerHTML"]
    assert "training data" in panel["innerHTML"]
    assert "Send Like" not in panel["innerHTML"]
    assert "swipe" not in panel["innerHTML"].lower()
