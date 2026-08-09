"""Hinge driver — offline tests for the HOST-SIDE ADB rewrite.

No device: a FakeAdb scripts screencap frames and records taps/swipes/text, so we
exercise capture (humanized read + dedup), the like/dislike/like-with-comment tap
sequences, and wait_for_decision's like/pass/none logic. Non-PNG fake frames make
the driver's frame-diff fall back to exact-bytes (any change = whole-frame change),
which is enough for pass/none/stop/timeout; the region-based LIKE path is driven by
monkeypatching _split_diff with scripted (top, bottom) deltas. Real coordinates and
diff thresholds are confirmed live on a finished profile (see hinge.py header).
"""
import random

import pytest

from operation_love.drivers import hinge
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.hinge import HingeDriver
from operation_love.interaction import AutoSessionPolicy


class _Cfg:
    apps = {"hinge": {"serial": "pixel"}}


def _png(value=0, size=(24, 24)):
    """A decodable solid-grey PNG (so _downsample succeeds), for tests that exercise the
    real decode path rather than the undecodable-bytes fallback."""
    from io import BytesIO

    import numpy as np
    from PIL import Image
    arr = np.full((size[1], size[0]), value, dtype="uint8")
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _action_frame(heart_xy=(937, 1600), x_xy=(125, 2035), confirm_xy=(540, 1300)):
    """A decodable frame with the like-heart, pass-X, and comment-sheet 'confirm' glyphs
    pasted at known spots, so the driver's vision locator finds them — exercises the real
    action path, not the fallback. The confirm glyph is what lets _await_sheet_open() (the
    gate that confirms the comment sheet actually opened before the driver taps the FIXED
    comment_box / send_like coordinates) succeed from a single scripted frame."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(1)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in (
        ("hinge_heart.png", heart_xy),
        ("hinge_pass_x.png", x_xy),
        ("hinge_send_like.png", confirm_xy),
    ):
        t = hinge._load_template(name)
        th, tw = t.shape
        canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


class FakeAdb:
    def __init__(self, frames, advance_on_screencap=False):
        self.frames = list(frames) or [b""]
        self.i = 0
        self.advance_on_screencap = advance_on_screencap
        self.taps = []
        self.swipes = 0
        self.scrolls = 0
        self.texts = []

    def screen_size(self):
        return (1080, 2400)

    def devices(self):
        return ["pixel"]

    def shell(self, command="", **_):
        return ""

    def screencap(self):
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        if self.advance_on_screencap and self.i < len(self.frames) - 1:
            self.i += 1
        return frame

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, *_a, **_k):
        self.swipes += 1

    def scroll_up(self, *_a, **_k):
        self.scrolls += 1
        if self.i < len(self.frames) - 1:
            self.i += 1

    def text(self, s):
        self.texts.append(s)


def _drv(adb, **cfg):
    # halt_on_error defaults False here so tap-focused tests don't trigger progress-verification
    # (which screencaps after an action); tests that want it pass halt_on_error=True explicitly.
    class C:
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False, **cfg}}
    d = HingeDriver(C())
    d._adb = adb
    d._touch = adb            # touches route through the (fake) transport
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


# --- capture: humanized read + frame-repeat dedup -----------------------
def test_capture_stops_when_scroll_repeats():
    adb = FakeAdb([b"a", b"b", b"b", b"c"])     # frame repeats after the scroll to "b"
    drv = _drv(adb)
    profile = drv._capture_current()
    assert profile.photos == [b"a", b"b"]
    assert profile.meta["app"] == "hinge"
    assert profile.meta["capture_frames"] == 2
    assert profile.meta["read_scrolls"] == 2
    assert profile.meta["read_dwell_s_total"] > 0
    assert adb.scrolls == 2


def test_capture_respects_scroll_capture_limit():
    adb = FakeAdb([b"a", b"b", b"c", b"d"])
    drv = _drv(adb, scroll_captures=2)
    profile = drv._capture_current()
    assert profile.photos == [b"a", b"b"]


def test_auto_policy_varies_read_geometry_dwell_and_only_raises_capture_ceiling(monkeypatch):
    class Policy:
        def __init__(self):
            self.scroll_calls = []
            self.dwell_calls = []

        def sample_read_scroll(self, base_frac, depth):
            self.scroll_calls.append((base_frac, depth))
            return base_frac + depth * 0.005, 0.40 + depth * 0.01

        def read_dwell(self, base_s, depth, complexity_hint):
            self.dwell_calls.append((base_s, depth, complexity_hint))
            return 0.2 + depth * 0.01

    # The policy path may add zero, one, or two captures; pin the upper case and prove it is
    # upward-only from the configured safety baseline rather than replacing it with a new cap.
    monkeypatch.setattr(hinge.random, "randint", lambda _a, _b: 2)
    adb = FakeAdb([f"f{i}".encode() for i in range(10)])
    drv = _drv(adb, scroll_captures=8)
    policy = Policy()
    drv._auto_policy = policy

    profile = drv._capture_current()

    assert len(profile.photos) == 10
    assert drv._profile_capture_limit == 10
    assert policy.scroll_calls == [(0.55, i) for i in range(9)]
    assert [depth for _base, depth, _hint in policy.dwell_calls] == list(range(9))
    assert drv._capture_scroll_ledger == [
        (0.55 + i * 0.005, 0.40 + i * 0.01) for i in range(9)
    ]


def test_real_auto_session_policy_drives_capture_geometry_dwell_and_metadata(monkeypatch):
    adb = FakeAdb([f"frame-{i}".encode() for i in range(12)])
    drv = _drv(adb, scroll_captures=8)
    policy = AutoSessionPolicy(rng=random.Random(19), local_hour=lambda: 15)
    drv.set_auto_session_policy(policy)
    slept = []
    monkeypatch.setattr(hinge.time, "sleep", lambda seconds: slept.append(seconds))

    profile = drv._capture_current()

    assert 8 <= len(profile.photos) <= 10
    assert len(drv._capture_scroll_ledger) == len(profile.photos) - 1
    assert len({round(frac, 4) for frac, _lane in drv._capture_scroll_ledger}) > 1
    assert len({round(lane, 4) for _frac, lane in drv._capture_scroll_ledger}) > 1
    assert len({round(seconds, 4) for seconds in slept}) > 1
    assert profile.meta["app"] == "hinge"
    assert profile.meta["capture_frames"] == len(profile.photos)
    assert profile.meta["read_scrolls"] == len(drv._capture_scroll_ledger)
    assert profile.meta["read_dwell_s_total"] == pytest.approx(sum(slept))


def test_malformed_auto_policy_falls_back_to_existing_read_behavior():
    class BrokenPolicy:
        def sample_read_scroll(self, _base, _depth):
            return float("nan"), 99

        def read_dwell(self, _base, _depth, _hint):
            raise RuntimeError("sampling failed")

    class RecordingAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"x"])
            self.scroll_args = []

        def scroll_up(self, distance_frac=0.55, x_frac=0.5):
            self.scroll_args.append((distance_frac, x_frac))
            super().scroll_up(distance_frac, x_frac)

    adb = RecordingAdb()
    drv = _drv(adb)
    drv._auto_policy = BrokenPolicy()

    drv._scroll_down_one()

    assert adb.scroll_args == [(0.55, 0.5)]
    assert drv._capture_scroll_ledger == [(0.55, 0.5)]


# --- actions: NORMAL like only, opener as a comment ---------------------
def test_dislike_taps_the_vision_located_pass_x():
    adb = FakeAdb([_action_frame()])
    _drv(adb).dislike()
    assert adb.taps == [(125, 2035)]             # X located by vision (glyph), not a fixed fraction


def test_like_without_opener_taps_heart_then_send():
    adb = FakeAdb([_action_frame()])             # constant frame -> _scroll_to_top stops at once
    _drv(adb).like()
    heart = (937, 1600)                          # heart located by vision on the first photo
    send = (int(0.643 * 1080), int(0.576 * 2400))   # "Send Like" — calibrated fixed coord
    assert adb.taps == [heart, send]
    assert adb.texts == []                       # no comment typed


def test_like_with_opener_types_comment_between_heart_and_send():
    adb = FakeAdb([_action_frame()])
    _drv(adb).like("loved your stargazing prompt")
    heart = (937, 1600)                          # vision-located
    box = (int(0.500 * 1080), int(0.529 * 2400))
    send = (int(0.643 * 1080), int(0.576 * 2400))
    assert adb.taps == [heart, box, send]
    assert adb.texts == ["loved your stargazing prompt"]


# --- observe: wait_for_decision via frame deltas ------------------------
def test_pass_detected_on_card_advance():
    adb = FakeAdb([b"a", b"b"], advance_on_screencap=True)
    assert _drv(adb).wait_for_decision(timeout=5.0) is False


def test_waits_until_a_change_then_pass():
    adb = FakeAdb([b"a", b"a", b"b"], advance_on_screencap=True)
    assert _drv(adb).wait_for_decision(timeout=None) is False


def test_timeout_returns_none():
    adb = FakeAdb([b"a"])                         # never changes
    assert _drv(adb).wait_for_decision(timeout=0.05) is None


def test_stop_interrupts_and_returns_none():
    adb = FakeAdb([b"a", b"a", b"a"])
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 1

    assert _drv(adb).wait_for_decision(timeout=None, should_stop=should_stop) is None
    assert calls["n"] == 2


class _ScriptedDiff:
    """Replaces hinge._split_diff with a scripted (top, bottom) per call."""
    def __init__(self, *pairs):
        self.pairs = list(pairs)

    def __call__(self, _a, _b):
        return self.pairs.pop(0) if self.pairs else (0.0, 0.0)   # exhausted -> "no change"


def test_like_detected_sheet_then_advance(monkeypatch):
    # bottom-only change (sheet up), then a whole-frame change (advanced) -> like sent
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"sheet", b"b"])
    assert _drv(adb).wait_for_decision(timeout=5.0) is True


def test_cancelled_like_then_pass(monkeypatch):
    # sheet up, reverts to the SAME card (cancelled), then a real advance (pass)
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"sheet", b"a", b"c"])
    assert _drv(adb).wait_for_decision(timeout=5.0) is False


# --- session lifecycle + device loss -----------------------------------
def test_open_session_requires_the_configured_device(monkeypatch):
    monkeypatch.setattr(hinge, "Adb", lambda *a, **k: FakeAdbNoDevice())
    with pytest.raises(DriverClosed):
        HingeDriver(_Cfg()).open_session()


class FakeAdbNoDevice(FakeAdb):
    def __init__(self):
        super().__init__([b"x"])

    def devices(self):
        return []                                # nothing connected


def test_device_loss_propagates_as_driver_closed():
    class Lost(FakeAdb):
        def screencap(self):
            raise DriverClosed("ADB device was disconnected")

    drv = _drv(Lost([b"x"]))
    with pytest.raises(DriverClosed):
        drv.current_profile()
    with pytest.raises(DriverClosed):
        drv.wait_for_decision(timeout=5.0)


def test_real_bug_is_not_masked_as_driver_closed():
    class Buggy(FakeAdb):
        def screencap(self):
            raise ValueError("unexpected UI state")

    with pytest.raises(ValueError):
        _drv(Buggy([b"x"])).current_profile()


def test_open_session_raises_driver_closed_when_deps_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "PIL", None)
    drv = HingeDriver(_Cfg())
    with pytest.raises(DriverClosed) as exc:
        drv.open_session()
    assert "requires PIL and numpy" in str(exc.value)


def test_wait_for_decision_ignores_scroll(monkeypatch):
    adb = FakeAdb([b"a", b"b", b"c"], advance_on_screencap=True)
    drv = _drv(adb)

    # Pre-populate the captured signatures with downsamples of "a" and "b"
    # (use 24x24 arrays of ones to simulate valid downsampled images)
    import numpy as np
    drv._current_sigs = [np.ones((24, 24)), np.ones((24, 24)) * 2]

    # Mock _downsample to return our simulated signatures
    def mock_downsample(frame, size=24):
        if frame == b"a":
            return drv._current_sigs[0]
        if frame == b"b":
            return drv._current_sigs[1]
        return np.ones((24, 24)) * 99  # "c" is far away

    monkeypatch.setattr(hinge, "_downsample", mock_downsample)
    monkeypatch.setattr(hinge, "_split_diff", lambda x, y: (0.0, 0.0) if x == y else (15.0, 15.0))

    # wait_for_decision should ignore the transition to "b" (scroll) and only return on "c" (pass -> False)
    assert drv.wait_for_decision(timeout=5.0) is False
    assert adb.i == 2


def test_capture_returns_none_on_static_screen():
    # A DECODABLE but unchanging screen on the first scroll = Out of Profiles / Loading -> None.
    frame = _png(120)
    adb = FakeAdb([frame, frame])
    drv = _drv(adb)
    drv._observe_ready = True

    assert drv._capture_current() is None


def test_capture_raises_on_identical_undecodable_when_observe_ready():
    # Audit fix: a wedged device returning the SAME undecodable bytes (e.g. empty screencap on
    # exit 0) must NOT masquerade as a static 'Out of Profiles' screen (return None -> the
    # observe worker livelocks). With deps validated, it hard-fails so the worker stops clean.
    adb = FakeAdb([b"x", b"x"])          # identical, undecodable -> not a real static screen
    drv = _drv(adb)
    drv._observe_ready = True
    with pytest.raises(DriverClosed):
        drv._capture_current()


def test_like_sheet_not_misread_as_scroll(monkeypatch):
    # C1 regression: a real LIKE = like-sheet (top calm, bottom busy) -> advance to a
    # genuinely NEW profile. Sheet detected before the scroll check, AND the advance frame
    # must NOT match the current profile's sigs (so it counts as a new card, not a scroll).
    import numpy as np
    sig = np.ones((24, 24))
    new_profile = np.ones((24, 24)) * 99
    monkeypatch.setattr(hinge, "_downsample",
                        lambda frame, size=24: new_profile if frame == b"b" else sig)
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))  # sheet, then advance confirmed
    adb = FakeAdb([b"a", b"sheet", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._current_sigs = [sig]            # advance frame b"b" doesn't match -> genuine new profile
    assert drv.wait_for_decision(timeout=5.0) is True   # LIKE


def test_uniform_top_scroll_not_misread_as_like(monkeypatch):
    # #6 regression: a SCROLL on a uniform-top profile (top calm, bottom busy) hits the
    # like-branch, but the 'advanced' frame still matches the current profile's sigs, so it
    # must be recognized as a scroll -> NOT reported as a LIKE.
    import numpy as np
    sig = np.ones((24, 24))
    monkeypatch.setattr(hinge, "_downsample", lambda *a, **k: sig)   # every frame = the SAME profile
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"s1", b"s2"])
    drv = _drv(adb)
    drv._current_sigs = [sig]
    assert drv.wait_for_decision(timeout=0.2) is not True   # not a false LIKE


def test_capture_refuses_degraded_when_decode_fails():
    # H1: in a real run (open_session validated deps), undecodable frames must hard-fail
    # rather than silently disable scroll-detection (which would mislabel scrolls as PASS).
    adb = FakeAdb([b"a", b"b"])          # non-PNG bytes -> _downsample returns None
    drv = _drv(adb)
    drv._observe_ready = True            # simulate open_session having validated PIL/numpy
    with pytest.raises(DriverClosed):
        drv._capture_current()


def test_scroll_to_top_ceiling_derived_from_capture_scrolls_not_a_magic_four(monkeypatch):
    # bug 1 fix: a fixed cap of 4 could NOT undo a capture that scrolled further than that
    # (scroll_captures defaults to 8, i.e. up to 7 read-scrolls) -- it left the phone parked
    # mid-profile, so the next heart/pass tap template-matched a photo from the WRONG card.
    # The ceiling must track how far the just-completed capture actually scrolled, not a guess.
    frames = [b"f0", b"f1", b"f2", b"f3", b"f4", b"f5", b"f6", b"f7"]   # 8 distinct -> 7 scrolls
    adb = FakeAdb(frames)
    drv = _drv(adb, scroll_captures=8)
    drv._capture_current()
    assert adb.scrolls == 7                          # sanity: capture scrolled down 7 times

    adb.swipes = 0
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)  # never settles -> ceiling governs
    drv._scroll_to_top()
    assert adb.swipes == 7               # old code capped this at a magic 4 -> would fail here


def test_scroll_to_top_still_bounded_against_a_stuck_screen(monkeypatch):
    # the (now-correct) ceiling is still a hard cap, not an infinite loop, when a screen never
    # settles (e.g. an animated/video card whose frames always differ).
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)  # always "still moving"
    adb = FakeAdb([b"x"])
    drv = _drv(adb, scroll_captures=8)
    drv._capture_scrolls = 3             # simulate a capture that scrolled 3 times
    drv._scroll_to_top()
    assert adb.swipes == 3               # stops at the ceiling, doesn't spin forever


class PositionTrackingAdb(FakeAdb):
    """FakeAdb that also tracks cumulative scroll DISPLACEMENT as a scalar profile position
    (0 == top, growing as the profile is read further down) instead of just counting
    scroll_up/swipe calls. A pure call-COUNT assertion (as in the tests above) can't see a
    DISTANCE mismatch between a forward read-scroll and its undo-swipe -- it happily passes
    even when the undo systematically travels less (or more) than the scroll it's meant to
    cancel, which is exactly the bug a fixed-drag undo-swipe had: same swipe COUNT as
    read-scrolls, but each swipe covered less ground, so the net position never reached zero.
    """
    def __init__(self, frames, **kw):
        super().__init__(frames, **kw)
        self.position = 0.0

    def scroll_up(self, distance_frac=0.55, x_frac=0.5):
        super().scroll_up(distance_frac=distance_frac, x_frac=x_frac)
        self.position += distance_frac    # forward read-scroll: further down the profile

    def swipe(self, x1, y1, x2, y2, duration_ms=450):
        super().swipe(x1, y1, x2, y2, duration_ms=duration_ms)
        h = self.screen_size()[1]
        self.position += (y1 - y2) / h    # scroll_to_top's undo gesture: top-of-screen -> bottom-of-screen


def test_scroll_to_top_returns_to_true_top_not_just_matching_swipe_count(monkeypatch):
    # A swipe-COUNT match (test_scroll_to_top_ceiling_derived_from_capture_scrolls_not_a_magic_four
    # above) is not enough: the undo-swipe must also cover the SAME DISTANCE as the read-scroll it
    # cancels. A fixed 0.45h undo-swipe against a 0.55h default read-scroll left the profile
    # scrolled down by ~18% per scroll even with a correct swipe count -- up to ~0.7 screen-heights
    # short of the top on a long profile -- so the next heart/pass tap could land on the wrong
    # photo. Track actual displacement (not just call counts) so an under/over-correction is visible.
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)   # never settle early -> full ceiling
    frames = [f"f{i}".encode() for i in range(8)]     # 8 distinct frames -> 7 read-scrolls
    adb = PositionTrackingAdb(frames)
    drv = _drv(adb, scroll_captures=8, read_scroll_frac=0.55)

    drv._capture_current()
    assert adb.position == pytest.approx(7 * 0.55)     # capture read-scrolled forward 7 times

    drv._scroll_to_top()
    assert adb.position <= 1e-9, (
        f"scroll-to-top left the profile at position {adb.position:.3f} screens down from the "
        "top -- a like/opener can land on the wrong photo"
    )


def test_auto_policy_undo_uses_ledger_but_not_a_one_for_one_reverse_replay(monkeypatch):
    class Policy:
        pass

    # Six forward strokes total three screen-heights.  Pin the independent undo sampler at
    # 1.2x the mean forward distance; screenshot settling therefore reaches the real top in
    # five strokes, proving this is not a six-item reverse replay of the ledger.
    monkeypatch.setattr(
        hinge.random, "uniform", lambda low, _high: 1.20 if low >= 1.0 else 0.44)
    adb = PositionTrackingAdb([b"x"])
    drv = _drv(adb, scroll_captures=8, read_scroll_frac=0.55)
    drv._auto_policy = Policy()
    drv._capture_scroll_ledger = [
        (0.46, 0.41), (0.50, 0.57), (0.54, 0.45),
        (0.48, 0.60), (0.52, 0.39), (0.50, 0.53),
    ]
    drv._capture_scrolls = len(drv._capture_scroll_ledger)
    adb.position = sum(frac for frac, _lane in drv._capture_scroll_ledger)
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, _a, _b: adb.position > 1e-9)

    drv._scroll_to_top()

    assert adb.swipes == 5
    assert adb.position <= 1e-9
    assert drv._capture_scroll_ledger == [] and drv._capture_scrolls == 0


def test_auto_policy_undo_has_a_hard_ceiling_on_a_never_settling_screen(monkeypatch):
    class Policy:
        pass

    monkeypatch.setattr(HingeDriver, "_changed", lambda self, _a, _b: True)
    adb = FakeAdb([b"animated"])
    drv = _drv(adb)
    drv._auto_policy = Policy()
    drv._capture_scroll_ledger = [(0.5, 0.5)] * 4
    drv._capture_scrolls = 4

    drv._scroll_to_top()

    assert adb.swipes == 7                 # four recorded strokes + three safety attempts


def test_auto_policy_undo_varies_distance_and_lane_across_profiles(monkeypatch):
    class Policy:
        pass

    gestures = []
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, _a, _b: False)
    drv = _drv(FakeAdb([b"stable"]))
    drv._auto_policy = Policy()
    monkeypatch.setattr(drv, "_swipe", lambda *args: gestures.append(args))

    for _ in range(40):
        drv._capture_scroll_ledger = [(0.46, 0.43), (0.58, 0.56)]
        drv._capture_scrolls = 2
        drv._scroll_to_top()

    assert len({args[0] for args in gestures}) > 15             # x lane varies
    assert len({args[3] - args[1] for args in gestures}) > 15   # reverse distance varies


def test_current_profile_returns_to_top_after_observe_capture(monkeypatch):
    # bug 2 fix: current_profile() is the ONLY observe capture path, and worker.py prints
    # "READY - swipe this profile" immediately after it returns -- so leaving the phone
    # scrolled to the bottom (all _capture_current did before this fix) has the operator
    # swipe a view that isn't the top of the profile they just read.
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)  # force full ceiling
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)

    drv.current_profile()

    assert adb.scrolls == 2              # capture read-scrolled twice (3 distinct frames)
    assert adb.swipes == 2               # _scroll_to_top swiped all the way back up before returning


def test_next_profile_does_not_scroll_to_top_itself():
    # auto path: like() already calls _scroll_to_top() before acting, so next_profile() must
    # NOT also scroll back up -- that would be a redundant extra scroll (bug 2 fix note).
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)
    drv.next_profile()
    assert adb.swipes == 0


# --- Phase 4: touch transport selection (UHID preferred, adb input fallback) -----
class _HidAdb(FakeAdb):
    """A FakeAdb whose `test -e /system/bin/hid` probe says hid IS present."""
    def shell(self, command="", **_):
        super().shell(command)
        return "yes" if "system/bin/hid" in command else ""


def test_make_touch_prefers_uhid_when_hid_present():
    from operation_love.drivers.uhid import UhidTouch
    drv = HingeDriver(_Cfg())
    drv._adb = _HidAdb([b"x"])
    assert isinstance(drv._make_touch(), UhidTouch)


def test_make_touch_refuses_to_downgrade_when_uhid_unavailable():
    # UHID is the genuine transport (real kernel TOOL_TYPE_FINGER events, a variable
    # pressure ramp); adb's `input motionevent` cannot vary pressure at all. touch_backend
    # defaults to "auto", and "auto" now REFUSES rather than silently trading down to the
    # weaker transport -- a silent downgrade could run for an unknown length of time with
    # no error, on the account we actually care about. Only an explicit
    # touch_backend="adb" (see test_touch_backend_adb_forces_input_transport) accepts it.
    adb = FakeAdb([b"x"])                 # shell returns "" -> no hid -> UhidUnavailable
    drv = HingeDriver(_Cfg())
    drv._adb = adb
    with pytest.raises(DriverClosed):
        drv._make_touch()


def test_touch_backend_adb_forces_input_transport():
    adb = _HidAdb([b"x"])                 # hid present, but config forces adb
    drv = HingeDriver(type("C", (), {"apps": {"hinge": {"touch_backend": "adb"}}}))
    drv._adb = adb
    assert drv._make_touch() is adb


def test_touch_backend_uhid_required_raises_when_unavailable():
    adb = FakeAdb([b"x"])                 # no hid
    drv = HingeDriver(type("C", (), {"apps": {"hinge": {"touch_backend": "uhid"}}}))
    drv._adb = adb
    with pytest.raises(DriverClosed):
        drv._make_touch()


def test_make_touch_refuses_to_downgrade_when_probe_raises_adb_error():
    # A laggy/erroring `test -e hid` probe still means "UHID unavailable" -- it must NOT
    # escape into a silent adb-input downgrade either. The operator needs to see the failure
    # (device/link issue) and fix it, or explicitly opt into touch_backend: adb, rather than
    # the run quietly using the weaker, constant-pressure transport indefinitely.
    from operation_love.drivers.adb import AdbError

    class ErrAdb(FakeAdb):
        def shell(self, command="", **_):
            if "system/bin/hid" in command:
                raise AdbError(["adb", "shell", command], "probe failed")
            return ""

    adb = ErrAdb([b"x"])
    drv = HingeDriver(_Cfg())
    drv._adb = adb
    with pytest.raises(DriverClosed):
        drv._make_touch()


def _modal_frame():
    """A decodable frame containing the 'Send Like anyway' modal text (the Rose upsell)."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(2)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    t = hinge._load_template("hinge_send_like_anyway.png")
    th, tw = t.shape
    cx, cy = 420, 2197
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def test_handle_rose_upsell_taps_send_like_anyway_when_modal_present():
    # Rose upsell appears when a Rose is available; we send the NORMAL like, never a Rose.
    adb = FakeAdb([_modal_frame()])
    drv = _drv(adb)
    assert drv._handle_rose_upsell() is True
    assert adb.taps == [(420, 2197)]          # "Send Like anyway", not the Rose button above it


def test_handle_rose_upsell_noop_when_no_modal():
    adb = FakeAdb([_action_frame()])          # no Rose modal present
    drv = _drv(adb)
    assert drv._handle_rose_upsell() is False
    assert adb.taps == []


def _like_flow_frame_with_modal(heart_xy=(937, 1600), confirm_xy=(540, 1300), modal_xy=(420, 2197)):
    """A decodable frame carrying the heart glyph (so the heart tap is vision-located), the
    comment-sheet 'confirm' glyph (so _await_sheet_open's gate finds the sheet actually up),
    AND the Rose-upsell 'Send Like anyway' glyph (so the same scripted frame also drives the
    post-send upsell dismissal) -- lets a single frame exercise the FULL like() path end to
    end, unlike the old vision-miss-then-fallback shortcut this replaces."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(9)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in (
        ("hinge_heart.png", heart_xy),
        ("hinge_send_like.png", confirm_xy),
        ("hinge_send_like_anyway.png", modal_xy),
    ):
        t = hinge._load_template(name)
        th, tw = t.shape
        canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def test_like_dismisses_rose_upsell_modal():
    # full like() path: heart -> sheet confirmed open -> Send Like -> "Send Like anyway"
    # (Rose modal auto-dismissed). The heart is now vision-located for real (no fallback
    # left to lean on), so the frame must actually carry the glyph.
    adb = FakeAdb([_like_flow_frame_with_modal()])
    _drv(adb).like()
    assert (420, 2197) in adb.taps            # the normal-like confirmation was tapped


# --- opener item targeting (like the RIGHT photo, not always the first) -
def test_locate_target_heart_navigates_to_referenced_item(monkeypatch):
    import numpy as np
    sig0, sig1 = np.zeros((24, 24)), np.ones((24, 24)) * 50
    adb = FakeAdb([b"f0", b"f1"])
    drv = _drv(adb)
    drv._current_sigs = [sig0, sig1]
    monkeypatch.setattr(hinge, "_downsample", lambda f, size=24: sig0 if f == b"f0" else sig1)
    monkeypatch.setattr(hinge, "_match_glyph",
                        lambda frame, t, side="right", **k: [(937, 1500)] if frame == b"f1" else [])
    assert drv._locate_target_heart(1) == (937, 1500)   # scrolled to frame 1, tapped its heart


def test_locate_target_heart_index0_uses_topmost():
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)
    drv._current_sigs = [object(), object()]
    assert drv._locate_target_heart(0) == (937, 1600)   # index 0 -> topmost (first photo), no nav


def test_capture_keeps_sigs_index_aligned_with_photos():
    # an undecodable middle frame: it lands in photos AND in _current_sigs (as None) at the SAME
    # index, so opener referenced_index (into photos) and the sig lookup never desync.
    adb = FakeAdb([_png(10), b"BADFRAME", _png(20), _png(20)])   # f3 dups f2 -> stop
    drv = _drv(adb)
    prof = drv._capture_current()
    assert len(drv._current_sigs) == len(prof.photos)
    assert drv._current_sigs[1] is None                  # the undecodable frame's aligned slot


def test_locate_target_heart_falls_back_on_none_sig():
    drv = _drv(FakeAdb([_action_frame()]))
    drv._current_sigs = [object(), None, object()]       # target index 1 is undecodable
    assert drv._locate_target_heart(1) == (937, 1600)    # falls back to the first photo's heart


# --- HINGE-05: fallback must be recorded in the debug log, and the search must not sweep
# the whole deck when the target is never found --------------------------------------------
def test_locate_target_heart_logs_fallback_when_target_not_found(monkeypatch, tmp_path):
    from operation_love.drivers.debuglog import HingeDebugLog
    import numpy as np
    never_matches = np.ones((24, 24)) * 250
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: np.zeros((24, 24)))
    # The final fallback is _await_button("like"), which now vision-locates for real (no
    # blind-coordinate fallback left) -- the scripted frame must actually carry the heart
    # glyph, or the "fallback" this test is pinning would itself raise UnlocatedControlError.
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="r")
    drv._current_sigs = [np.zeros((24, 24)), never_matches]

    drv._locate_target_heart(1)   # target sig never matches any captured frame -> falls back

    import json
    recs = [json.loads(ln) for ln in (tmp_path / "r" / "actions.jsonl").read_text().splitlines()]
    fallback_recs = [r for r in recs if r["action"] == "locate_target_heart"]
    assert len(fallback_recs) == 1
    assert fallback_recs[0]["outcome"] == "fallback"
    assert fallback_recs[0]["reason"] == "target_frame_not_found"


def test_locate_target_heart_bounds_search_when_target_not_found(monkeypatch):
    # Old code swept the FULL scroll_captures+1 depth (9 scrolls, for the default
    # scroll_captures=8) before giving up on a target it never matched -- a target already
    # scrolled past effectively costs a full wasted sweep. The search must be capped closer to
    # item_index (how far down from the top the target should be).
    import numpy as np
    never_matches = np.ones((24, 24)) * 250
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: np.zeros((24, 24)))
    # As above: the final fallback vision-locates for real, so the frame needs the heart glyph.
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._current_sigs = [np.zeros((24, 24)), never_matches]

    drv._locate_target_heart(1)

    assert adb.scrolls <= 4   # old code: 9 (scroll_captures + 1); capped near item_index + 3 here


def test_verify_like_landed_raises_when_sheet_or_modal_open(monkeypatch):
    from operation_love.drivers.hinge import HingeActionError
    drv = _drv(FakeAdb([b"f"]), halt_on_error=True)
    monkeypatch.setattr(hinge, "_match_glyph", lambda *a, **k: [(1, 1)])   # sheet/modal still up
    with pytest.raises(HingeActionError):
        drv._verify_like_landed(b"before")


def test_verify_like_landed_ok_when_closed_and_advanced(monkeypatch):
    drv = _drv(FakeAdb([b"after"]), halt_on_error=True)
    monkeypatch.setattr(hinge, "_match_glyph", lambda *a, **k: [])         # no sheet/modal
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (50.0, 50.0))   # advanced
    drv._verify_like_landed(b"before")                                     # no raise


def test_verify_like_landed_raises_when_screen_unchanged(monkeypatch):
    from operation_love.drivers.hinge import HingeActionError
    drv = _drv(FakeAdb([b"same"]), halt_on_error=True)
    monkeypatch.setattr(hinge, "_match_glyph", lambda *a, **k: [])         # no sheet/modal
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0))     # never advanced
    with pytest.raises(HingeActionError):
        drv._verify_like_landed(b"before")


def test_verify_like_landed_dismisses_late_animating_rose_modal(monkeypatch):
    # HINGE-07: the Rose modal can animate in AFTER _handle_rose_upsell's own poll window
    # already gave up (it wasn't visible yet at that point). _verify_like_landed must tolerate
    # that -- dismiss it itself and keep checking -- instead of treating a still-open modal as a
    # dead run and halting a perfectly-sent like.
    calls = {"n": 0}

    def fake_match(frame, template, side="any", threshold=0.6):
        calls["n"] += 1
        # iter1: sheet check (gone), modal check (not yet animated in)
        # iter2: sheet check (gone), modal check (now up -> gets dismissed)
        # iter3: sheet check (gone), modal check (gone, dismissed)
        if calls["n"] == 4:
            return [(5, 5)]
        return []

    monkeypatch.setattr(hinge, "_match_glyph", fake_match)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))

    adb = FakeAdb([b"before", b"before", b"after"], advance_on_screencap=True)
    drv = _drv(adb, halt_on_error=True)

    drv._verify_like_landed(b"before")            # must NOT raise
    assert drv._touch.taps == [(5, 5)]             # dismissed the late modal itself, never a Rose


def test_snap_retries_once_past_transient_failure():
    from operation_love.drivers.adb import AdbError

    class Flaky(FakeAdb):
        def __init__(self, frames):
            super().__init__(frames)
            self.n = 0

        def screencap(self):
            self.n += 1
            if self.n == 1:
                raise AdbError(["adb"], "transient timeout")
            return super().screencap()

    drv = _drv(Flaky([b"ok"]), halt_on_error=True)
    assert drv._snap() == b"ok"                           # retried past the transient AdbError


def test_snap_propagates_driver_closed():
    class Lost(FakeAdb):
        def screencap(self):
            raise DriverClosed("device gone")

    drv = _drv(Lost([b"x"]), halt_on_error=True)
    with pytest.raises(DriverClosed):
        drv._snap()


def test_snap_raises_after_two_failures_when_halt_on_error():
    from operation_love.drivers.adb import AdbError
    from operation_love.drivers.hinge import HingeActionError

    class Wedged(FakeAdb):
        def screencap(self):
            raise AdbError(["adb"], "device offline")

    drv = _drv(Wedged([b"x"]), halt_on_error=True)
    # Two consecutive screencap failures must halt (not silently return None and let
    # _verify_progress/_verify_like_landed treat the missing "before" frame as success).
    with pytest.raises(HingeActionError):
        drv._snap()


def test_snap_returns_none_after_two_failures_when_halt_on_error_false():
    from operation_love.drivers.adb import AdbError

    class Wedged(FakeAdb):
        def screencap(self):
            raise AdbError(["adb"], "device offline")

    # debug_log=True keeps _snap() from short-circuiting; halt_on_error=False means a
    # flaky screencap here is debug-only best-effort and must not halt the run.
    drv = _drv(Wedged([b"x"]), halt_on_error=False, debug_log=True)
    drv._dbg = object()   # any non-None sentinel; _snap only checks "is None"
    assert drv._snap() is None


def test_retry_until_default_bool_predicate_matches_existing_callers():
    calls = iter([None, (0, 0)])   # a falsy-looking-but-later-truthy sequence: None then a tuple
    assert hinge._retry_until(lambda: next(calls), tries=3, delay_s=0) == (0, 0)


def test_retry_until_custom_predicate_accepts_a_falsy_found_result():
    # A hypothetical check_fn whose "found" result can legitimately be falsy (e.g. 0)
    # would be silently retried away by the default `bool` predicate; an explicit
    # is_found predicate is exactly the escape hatch that prevents that.
    calls = iter([0])
    result = hinge._retry_until(lambda: next(calls), tries=3, delay_s=0,
                                is_found=lambda r: r is not None)
    assert result == 0


def test_retry_until_default_predicate_would_have_missed_a_falsy_found_result():
    # Documents the exact footgun the is_found parameter exists to close: with the
    # default `bool` predicate, a legitimately-found-but-falsy result (0) reads as
    # "not found" and retries exhaust to None instead of returning it.
    calls = iter([0, 0, 0])
    assert hinge._retry_until(lambda: next(calls), tries=3, delay_s=0) is None


# --- debug logging + halt-on-unexpected --------------------------------
def test_verify_progress_raises_when_screen_unchanged():
    from operation_love.drivers.hinge import HingeActionError
    drv = _drv(FakeAdb([b"same"]), halt_on_error=True)
    with pytest.raises(HingeActionError):
        drv._verify_progress(b"same", "dislike")          # before == after -> stuck -> raise


def test_verify_progress_ok_when_screen_changes():
    drv = _drv(FakeAdb([b"after"]), halt_on_error=True)
    drv._verify_progress(b"before", "dislike")            # before != after -> changed -> no raise


def test_verify_progress_skipped_when_halt_disabled():
    drv = _drv(FakeAdb([b"same"]))                        # halt_on_error False -> never raises
    drv._verify_progress(b"same", "dislike")


def test_snapshot_failure_writes_error_record(tmp_path):
    from operation_love.drivers.debuglog import HingeDebugLog
    drv = _drv(FakeAdb([b"frame"]))
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="r")
    drv.snapshot_failure(ValueError("boom"))
    import json
    rec = json.loads((tmp_path / "r" / "actions.jsonl").read_text().strip())
    assert rec["action"] == "unexpected" and "boom" in rec["error"]


def test_dislike_writes_debug_action_record(tmp_path):
    from operation_love.drivers.debuglog import HingeDebugLog
    drv = _drv(FakeAdb([_action_frame()]))
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="r")
    drv.dislike()
    import json
    recs = [json.loads(ln) for ln in (tmp_path / "r" / "actions.jsonl").read_text().splitlines()]
    assert any(r["action"] == "dislike" for r in recs)


def test_actions_route_through_touch_transport_not_adb():
    # touches must go to self.touch; screencaps (incl. vision button-location) stay on self.adb.
    adb = FakeAdb([_action_frame()])
    touch = FakeAdb([b"x"])
    drv = HingeDriver(_Cfg())
    drv._adb, drv._touch = adb, touch
    drv.halt_on_error = False                                # routing test: skip progress-verify
    drv.dislike()
    assert touch.taps == [(125, 2035)] and adb.taps == []   # X found via adb screencap, tapped via touch
    drv._scroll_to_top()
    assert touch.swipes >= 1 and adb.swipes == 0   # swipes via the touch transport


def test_await_button_refuses_when_vision_fails(monkeypatch):
    """When vision can't locate the glyph after all retries, _await_button() must REFUSE
    (UnlocatedControlError) rather than fall back to the calibrated fixed coordinate. That
    fallback used to exist on the theory that a degraded action beats no action; it doesn't
    here, because (a) the stale coordinate can now be occupied by a paid control (Hinge's
    Rose, Bumble's SuperSwipe) or something irreversible, and (b) a persistent vision miss is
    usually a missing OpenCV install, not a transient one -- silently tapping blind forever
    is worse than refusing and surfacing the real problem."""
    adb = FakeAdb([_png()])    # non-action frame: vision finds nothing
    drv = _drv(adb)
    # Force every vision attempt to return None
    monkeypatch.setattr(drv, "_locate_button", lambda _which: None)

    with pytest.raises(hinge.UnlocatedControlError):
        drv._await_button("like", tries=2)
    with pytest.raises(hinge.UnlocatedControlError):
        drv._await_button("pass", tries=2)
    assert adb.taps == []      # neither refusal issued a blind tap at the fixed coordinate



# --- blank-frame guard (screen off / keyguard) -------------------------------
# `adb exec-out screencap` SUCCEEDS on a sleeping/locked device and returns solid
# black. Two black frames read as "nothing changed" (silent stall); a good frame
# followed by a black one reads as a huge delta (phantom "pass"). The guard fails
# loudly on ACTING paths and merely keeps waiting on PASSIVE observe loops.

def _sparse_bright_png():
    """mean<=2 but LARGE spread. Deliberately tuned so ONLY the uniformity (ptp)
    clause can reject it — the mean clause alone would call this blank. Delete the
    ptp check and this fixture starts reading as a dead screen."""
    from io import BytesIO

    import numpy as np
    from PIL import Image
    arr = np.zeros((24, 24), dtype=np.uint8)
    arr[0, 0] = arr[5, 5] = 200
    arr[11, 11] = 180                       # -> mean 1.01, ptp 200
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _noisy_png(size=(24, 24), lo=40, hi=200):
    """A decodable frame with real spread — stands in for rendered UI content."""
    from io import BytesIO

    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)
    arr = rng.integers(lo, hi, size=(size[1], size[0]), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


class _CountingAdb(FakeAdb):
    """Counts screencaps so retry CARDINALITY is pinned, not just the outcome."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.caps = 0

    def screencap(self):
        self.caps += 1
        return super().screencap()


def test_is_blank_frame_detects_solid_black():
    assert hinge._is_blank_frame(_png(0)) is True


def test_is_blank_frame_needs_uniformity_not_just_darkness():
    """The ptp clause must be load-bearing: a frame dark enough to pass the mean
    test but full of edges (text on black) is real content, not a dead screen."""
    frame = _sparse_bright_png()
    import numpy as np
    from io import BytesIO
    from PIL import Image
    arr = np.asarray(Image.open(BytesIO(frame)).convert("L").resize((24, 24)), dtype="int16")
    assert np.mean(arr) <= hinge._BLANK_MEAN_MAX, "fixture must pass the MEAN clause..."
    assert np.ptp(arr) > hinge._BLANK_SPREAD_MAX, "...so that only PTP can reject it"
    assert hinge._is_blank_frame(frame) is False


def test_is_blank_frame_tolerance_band_is_not_exactly_zero():
    """Thresholds are 2.0, not 0.0, so a real panel that reads 1/255 instead of a
    perfect 0 still counts as blank. Setting the constants to 0.0 breaks this."""
    assert hinge._BLANK_MEAN_MAX > 0.0 and hinge._BLANK_SPREAD_MAX > 0.0
    assert hinge._is_blank_frame(_png(1)) is True


def test_is_blank_frame_allows_uniform_but_not_black():
    assert hinge._is_blank_frame(_png(10)) is False
    assert hinge._is_blank_frame(_png(255)) is False


def test_is_blank_frame_false_when_undecodable():
    """No decode -> we cannot tell, so we must not claim 'blank' and halt a healthy run."""
    assert hinge._is_blank_frame(b"NOTAPNG") is False


def test_is_blank_frame_ignores_a_monkeypatched_downsample(monkeypatch):
    """Regression: the guard must NOT ride on _downsample. That helper serves the
    dedup/diff layer and is freely stubbed by tests — sharing it let a sentinel
    signature masquerade as a dead screen (and let a patch disable the guard)."""
    import numpy as np
    monkeypatch.setattr(hinge, "_downsample", lambda *a, **k: np.ones((24, 24)))
    assert hinge._is_blank_frame(_noisy_png()) is False   # real content stays visible
    assert hinge._is_blank_frame(_png(0)) is True         # real black stays blank


# --- acting paths: raise -----------------------------------------------------
def test_screencap_raises_on_persistently_blank_screen():
    adb = FakeAdb([_png(0), _png(0)], advance_on_screencap=True)
    with pytest.raises(hinge.HingeActionError) as ei:
        _drv(adb)._screencap()
    assert "blank" in str(ei.value).lower()


def test_screencap_raises_even_when_halt_on_error_is_false():
    """Deliberate departure from _snap's gating: you cannot tap blind on a burner,
    so a blank screen on an ACTING path is fatal regardless of halt_on_error."""
    adb = FakeAdb([_png(0)])
    with pytest.raises(hinge.HingeActionError):
        _drv(adb, halt_on_error=False)._screencap()


def test_screencap_retries_exactly_once_before_raising():
    """Cardinality, not just outcome — a 6-try version would still 'pass' otherwise."""
    adb = _CountingAdb([_png(0)])
    with pytest.raises(hinge.HingeActionError):
        _drv(adb)._screencap()
    assert adb.caps == 2, f"expected 2 captures (one retry), got {adb.caps}"


def test_screencap_settles_between_the_two_attempts(monkeypatch):
    """The retry must actually WAIT. The autouse _no_sleep fixture would otherwise
    hide a tight busy-loop retry, so spy on sleep instead of relying on wall time."""
    slept = []
    monkeypatch.setattr(hinge.time, "sleep", lambda s: slept.append(s))
    with pytest.raises(hinge.HingeActionError):
        _drv(FakeAdb([_png(0)]))._screencap()
    assert slept and any(s > 0 for s in slept), f"no settle between attempts: {slept}"


def test_screencap_recovers_from_a_single_blank_frame():
    good = _noisy_png()
    adb = FakeAdb([_png(0), good], advance_on_screencap=True)
    assert _drv(adb)._screencap() == good


def test_screencap_passes_normal_frames_through_unchanged():
    good = _noisy_png()
    assert _drv(FakeAdb([good]))._screencap() == good


def test_snap_does_not_bury_or_swallow_the_blank_diagnosis():
    """_snap wraps captures in `except Exception`. The blank error must survive that
    verbatim — not be re-wrapped as 'screencap failed twice in a row', and not be
    swallowed to None when halt_on_error is False."""
    adb = FakeAdb([_png(0)])
    drv = _drv(adb, halt_on_error=False)
    drv._dbg = object()                      # force _snap past its early-out
    with pytest.raises(hinge.HingeActionError) as ei:
        drv._snap()
    assert "blank" in str(ei.value).lower()
    assert "twice in a row" not in str(ei.value)


# --- passive observe paths: keep waiting, never halt -------------------------
def test_wait_for_decision_does_not_raise_when_screen_sleeps():
    """THE regression this guard originally introduced: observe waits on a human
    with no touches of its own, so the screen sleeping mid-wait is ordinary. It must
    keep watching and time out benignly, NOT halt the run (which, via the shared
    stop_event, would also kill the concurrent Bumble worker)."""
    adb = FakeAdb([_png(0)])                 # asleep for the whole wait
    assert _drv(adb).wait_for_decision(timeout=0.05) is None


def test_wait_for_decision_resumes_after_the_screen_comes_back():
    """Asleep, then the owner wakes it and swipes -> the decision is still detected."""
    adb = FakeAdb([_png(0), _noisy_png(lo=10, hi=60), _noisy_png(lo=150, hi=250)],
                  advance_on_screencap=True)
    assert _drv(adb).wait_for_decision(timeout=5.0) is False   # advanced -> pass


def test_wait_for_decision_never_scores_a_blank_frame_as_a_pass():
    """A good baseline followed by black is a HUGE delta; diffing it would report a
    phantom 'pass' the human never made. Skipping the frame is what prevents that."""
    adb = FakeAdb([_noisy_png(), _png(0)], advance_on_screencap=True)
    assert _drv(adb).wait_for_decision(timeout=0.05) is None    # not False


def test_handle_rose_upsell_degrades_instead_of_raising_post_tap():
    """It runs AFTER the Send Like tap. Raising here would abort before
    worker.record_decision(), losing the record of a like that really went out."""
    assert _drv(FakeAdb([_png(0)]))._handle_rose_upsell(tries=1) is False


def test_blank_reason_reports_real_device_state():
    """The message must carry the actual diagnosis — that is the whole point of it."""
    class _Stateful(FakeAdb):
        def shell(self, command="", **_):
            if "power" in command:
                return "mWakefulness=Dozing\n"
            if "trust" in command:
                return "deviceLocked=1\n"
            return ""
    reason = _drv(_Stateful([_png(0)]))._blank_reason()
    assert "Dozing" in reason and "deviceLocked=1" in reason


def test_blank_reason_never_raises_even_if_shell_fails():
    class _Boom(FakeAdb):
        def shell(self, command="", **_):
            raise RuntimeError("device gone")
    assert isinstance(_drv(_Boom([_png(0)]))._blank_reason(), str)
