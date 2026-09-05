"""Hinge driver — offline tests for the HOST-SIDE ADB rewrite.

No device: a FakeAdb scripts screencap frames and records taps/swipes/text, so we
exercise capture (humanized read + dedup), the like/dislike/like-with-comment tap
sequences, and wait_for_decision's like/pass/none logic. Non-PNG fake frames make
the driver's frame-diff fall back to exact-bytes (any change = whole-frame change),
which is enough for pass/none/stop/timeout; the region-based LIKE path is driven by
monkeypatching _split_diff with scripted (top, bottom) deltas. Real coordinates and
diff thresholds are confirmed live on a finished profile (see hinge.py header).
"""
import hashlib
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from operation_love.drivers import hinge, scroll_top
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.hinge import HingeDriver
from operation_love.drivers.touchwatch import Gesture
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


_INLINE_COMMENT = (95, 1597, 985, 1775)
_INLINE_SEND = (390, 1807, 985, 1916)
_INLINE_CONFIRM = (695, 1856)


def _paint_inline_composer(canvas):
    """Paint the independently required input, CTA, and shipped confirmation glyph."""
    x0, y0, x1, y1 = _INLINE_COMMENT
    canvas[y0:y0 + 2, x0:x1] = 222
    canvas[y1 - 2:y1, x0:x1] = 222
    x0, y0, x1, y1 = _INLINE_SEND
    canvas[y0:y1, x0:x1] = 228
    glyph = hinge._load_template("hinge_send_like.png")
    height, width = glyph.shape
    x, y = _INLINE_CONFIRM
    canvas[y - height // 2:y - height // 2 + height,
           x - width // 2:x - width // 2 + width] = glyph


def _action_frame(heart_xy=(937, 1600), x_xy=(125, 2035)):
    """A deck frame plus a structurally valid inline composer.

    The Hinge inline layout is not proven by copied ``Send Like`` text: these fixtures paint the
    measured input outline, filled CTA, and glyph together so the real detector returns safe
    coordinates. Heart/pass glyphs remain independent deck evidence.

    The heart glyph is HINGE_SPEC.templates["like"] (hinge_like_button.png), not a hardcoded
    filename — it must always be whatever the real spec currently wires to the "like" role, or
    this fixture silently stops representing a real deck the moment that role's asset changes
    again (as it did away from hinge_heart.png, the "Which do we have in common" outline
    heart — see the templates dict comment on HINGE_SPEC in hinge.py)."""
    import cv2
    import numpy as np
    canvas = np.full((2400, 1080), 249, dtype=np.uint8)
    _paint_inline_composer(canvas)
    for name, (cx, cy) in (
        (hinge.HINGE_SPEC.templates["like"], heart_xy),
        ("hinge_pass_x.png", x_xy),
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

    def foreground_package(self):
        return "co.hinge.app"

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
def test_capture_stops_when_scroll_repeats(monkeypatch):
    # Pin the read's one randomized pause (2026-08-24 dwell-shape change -- see
    # _READ_PAUSE_COUNTS in hinge.py) to iteration 0, which this short read is guaranteed to
    # reach, so read_dwell_s_total > 0 below is a deterministic assertion rather than one that
    # only holds on the ~85% of unseeded draws that happen to land a pause somewhere this
    # 2-iteration read actually gets to.
    monkeypatch.setattr(HingeDriver, "_plan_read_pause_iterations", lambda self, _n: {0})
    adb = FakeAdb([b"a", b"b", b"b", b"c"])     # frame repeats after the scroll to "b"
    drv = _drv(adb)
    profile = drv._capture_current()
    assert profile.photos == [b"a", b"b"]
    assert profile.meta["app"] == "hinge"
    assert profile.meta["capture_frames"] == 2
    assert profile.meta["read_scrolls"] == 2
    assert profile.meta["read_dwell_s_total"] > 0
    assert adb.scrolls == 2
    assert profile.meta["capture_truncated"] is False    # the repeat proves the true bottom was seen
    assert drv._current_capture_truncated is False


def test_system_ui_foreground_blocks_capture_before_a_scroll_top_verdict(monkeypatch):
    """The Android notification shade is not a scrolled Hinge card.

    Regression for the 2026-08-19 run: quick settings covered Hinge, but its dark system header
    was compared with the filter-chip band and reported as ``confirmed_not_top``. The focused
    window is independent evidence, so no capture or recovery swipe may start.
    """
    class SystemUiAdb(FakeAdb):
        def foreground_package(self):
            return "com.android.systemui"

    adb = SystemUiAdb([b"quick-settings"])
    drv = _drv(adb)
    monkeypatch.setattr(drv, "out_of_profiles", lambda: False)
    drv._session_top_done = True

    assert drv.current_profile() is None
    assert adb.scrolls == adb.swipes == 0
    assert "quick-settings" in (drv.blocked_reason() or "")
    assert "no Hinge profile" in drv.blocked_reason()


def test_system_ui_mid_wait_pauses_then_resyncs_without_stopping_or_input():
    """A shade opened while READY is a reversible Observe interruption, not a stuck deck.

    Regression for the 2026-08-21 run: Hinge was unchanged through 01:38:41, System UI owned
    the next frame at 01:38:48, and the generic unknown-screen watchdog stopped 94.7s later.
    Observe must wait without touching anything, then recapture when Hinge returns.
    """
    class SwitchingForegroundAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"hinge", b"system-ui", b"hinge-returned"],
                             advance_on_screencap=True)
            self.packages = iter([
                "co.hinge.app",            # initial base handoff
                "com.android.systemui",    # changed frame proves the interruption
                "com.android.systemui",    # shade remains open for one pause poll
                "co.hinge.app",            # owner/system closes it
            ])

        def foreground_package(self):
            return next(self.packages, "co.hinge.app")

    adb = SwitchingForegroundAdb()
    drv = _drv(adb)
    drv._dbg = _FakeDbg()

    assert drv.wait_for_decision(timeout=5.0) is None
    assert adb.taps == [] and adb.swipes == 0 and adb.scrolls == 0 and adb.texts == []
    assert drv.blocked_reason() is None
    names = [name for name, _fields in drv._dbg.calls]
    assert names == ["observe_foreground_paused", "observe_foreground_resumed"]
    resumed = drv._dbg.calls[1][1]
    assert resumed["result"] == "recapture_without_decision"


def test_unknown_screen_watchdog_claims_only_that_operation_love_did_not_inject_input(
        monkeypatch, capsys):
    """A watchdog timeout cannot know whether the owner touched the phone while waiting."""
    drv = _drv(FakeAdb([b"unknown-screen"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_recognized = 1.0
    drv._observe_stuck_budget_s = 1.0
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(drv, "_deck_blocked_reason", lambda _frame: None)

    reason = drv._observe_stuck_bail(b"unknown-screen")
    output = capsys.readouterr().out

    assert reason is not None
    assert "Operation Love has not injected input" in reason
    assert "Operation Love has not injected input" in output
    assert "phone is untouched" not in reason
    assert "phone has not been touched" not in output


def test_completed_driver_inputs_are_audited_at_every_gesture_choke_point():
    adb = FakeAdb([b"frame"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()

    drv._tap(500, 1000)
    drv._swipe(500, 1000, 500, 1500, duration_ms=321)
    drv._scroll(0.2, 0.4)
    drv._text("hello")

    rows = [fields for name, fields in drv._dbg.calls if name == "device_input"]
    assert [row["kind"] for row in rows] == ["tap", "swipe", "scroll", "text"]
    assert rows[0]["start"] == rows[0]["end"] == [500, 1000]
    assert rows[1]["start"] == [500, 1000]
    assert rows[1]["end"] == [500, 1500]
    assert rows[1]["duration_ms"] == 321
    assert rows[2]["direction"] == "forward"
    assert rows[2]["distance_frac"] == 0.2
    assert rows[3]["chars"] == 5
    assert adb.texts == ["hello"]
    assert all(row["session_mode"] == "observe" for row in rows)
    assert all(row["transport"] == "FakeAdb" for row in rows)


def test_training_inputs_are_not_mislabeled_as_autonomous():
    adb = FakeAdb([b"frame"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv.set_training_decision(lambda _frame, _evidence: "dislike")

    drv._tap(500, 1000)

    row = next(fields for name, fields in drv._dbg.calls if name == "device_input")
    assert row["session_mode"] == "training"


def test_failed_text_transport_is_not_audited_as_completed_input():
    class FailingTextAdb(FakeAdb):
        def text(self, s):
            raise RuntimeError("text transport failed")

    drv = _drv(FailingTextAdb([b"frame"]))
    drv._dbg = _FakeDbg()

    with pytest.raises(RuntimeError, match="text transport failed"):
        drv._text("hello")

    assert not [fields for name, fields in drv._dbg.calls if name == "device_input"]


def test_downward_swipe_cannot_start_in_android_notification_shade_zone():
    adb = FakeAdb([b"frame"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()

    with pytest.raises(hinge.HingeActionError, match="notification shade"):
        drv._swipe(540, 100, 540, 1800)

    assert adb.swipes == 0
    assert not [fields for name, fields in drv._dbg.calls if name == "device_input"]

    # The calibrated 0.78-screen Hinge rewind starts at 0.11h and remains legal.
    drv._swipe(540, 264, 540, 2136)
    assert adb.swipes == 1


def test_capture_truncated_flag_set_when_ceiling_reached_without_a_natural_stop():
    # 8 distinct frames, scroll_captures=8 -> the loop exhausts its full range() WITHOUT ever
    # seeing the repeated frame that means "reached the true bottom". _current_sigs then does
    # not cover this profile's whole scrollable range, and wait_for_decision's
    # _vertical_shift_match can only recognize a manual scroll into territory it has actually
    # seen -- see the for/else in _capture_current.
    frames = [b"f0", b"f1", b"f2", b"f3", b"f4", b"f5", b"f6", b"f7"]
    adb = FakeAdb(frames)
    drv = _drv(adb, scroll_captures=8)

    profile = drv._capture_current()

    assert len(profile.photos) == 8
    assert profile.meta["capture_truncated"] is True
    assert drv._current_capture_truncated is True


def test_current_profile_starting_on_visible_like_sheet_returns_none_without_input():
    """A pending manual Send Like sheet is never profile input for read-scroll capture."""
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)

    assert drv.current_profile() is None
    assert adb.scrolls == 0
    assert adb.swipes == 0
    assert adb.taps == []
    assert adb.texts == []


def test_capture_respects_scroll_capture_limit():
    adb = FakeAdb([b"a", b"b", b"c", b"d"])
    drv = _drv(adb, scroll_captures=2)
    profile = drv._capture_current()
    assert profile.photos == [b"a", b"b"]


def test_capture_carries_one_contradictory_frame_to_index_rebuild(monkeypatch):
    """A single marked fallback retains enumeration cadence; the indexer remains the gate."""
    adb = FakeAdb([b"a", b"b", b"b"])
    drv = _drv(adb, scroll_captures=4)
    calls = []
    indexed = []

    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top", lambda: "")

    def plan(_frame, _lane, *, allow_segmentation_failure_fallback=False):
        calls.append(allow_segmentation_failure_fallback)
        return SimpleNamespace(
            frac=0.4, x_frac=0.5, step_px=240, sized_against_px=1027,
            basis=(hinge.COVERAGE_STEP_SEGMENTATION_FALLBACK if len(calls) == 1 else "measured"),
            spacing=SimpleNamespace(measured=False, px=None),
            reason="synthetic contradictory frame")

    monkeypatch.setattr(drv, "_plan_enumeration_step", plan)
    monkeypatch.setattr(drv, "_index_captured_items",
                        lambda photos, should_stop=None: indexed.append(list(photos)) or "")

    profile = drv._capture_current()

    assert profile is not None
    # The clean frame itself is planned while the fallback is still permitted; its measured
    # result closes the window for every later frame (pinned by the disjoint-run test below).
    assert calls == [True, True]
    assert indexed == [[b"a", b"b"]]


def test_capture_carries_two_adjacent_contradictory_frames_to_index_rebuild(monkeypatch):
    adb = FakeAdb([b"a", b"b", b"c", b"c"])
    drv = _drv(adb, scroll_captures=4)
    calls = []
    indexed = []

    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top", lambda: "")

    def plan(_frame, _lane, *, allow_segmentation_failure_fallback=False):
        calls.append(allow_segmentation_failure_fallback)
        return SimpleNamespace(
            frac=0.4, x_frac=0.5, step_px=240, sized_against_px=1027,
            basis=(hinge.COVERAGE_STEP_SEGMENTATION_FALLBACK if len(calls) <= 2 else "measured"),
            spacing=SimpleNamespace(measured=False, px=None), reason="synthetic contradiction")

    monkeypatch.setattr(drv, "_plan_enumeration_step", plan)
    monkeypatch.setattr(drv, "_index_captured_items",
                        lambda photos, should_stop=None: indexed.append(list(photos)) or "")

    assert drv._capture_current() is not None
    assert calls == [True, True, True]
    assert indexed == [[b"a", b"b", b"c"]]


def test_capture_carries_four_adjacent_contradictory_frames_to_index_rebuild(monkeypatch):
    """The fourth bounded failure reaches the indexer's measured-bridge gate."""
    adb = FakeAdb([b"a", b"b", b"c", b"d", b"e", b"e"])
    drv = _drv(adb, scroll_captures=5)
    calls = []
    indexed = []

    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top", lambda: "")

    def plan(_frame, _lane, *, allow_segmentation_failure_fallback=False):
        calls.append(allow_segmentation_failure_fallback)
        return SimpleNamespace(
            frac=0.4, x_frac=0.5, step_px=240, sized_against_px=1027,
            basis=(hinge.COVERAGE_STEP_SEGMENTATION_FALLBACK if len(calls) <= 4 else "measured"),
            spacing=SimpleNamespace(measured=False, px=None), reason="synthetic contradiction")

    monkeypatch.setattr(drv, "_plan_enumeration_step", plan)
    monkeypatch.setattr(drv, "_index_captured_items",
                        lambda photos, should_stop=None: indexed.append(list(photos)) or "")

    assert drv._capture_current() is not None
    assert calls == [True, True, True, True, False]
    assert indexed == [[b"a", b"b", b"c", b"d", b"e"]]


def test_capture_refuses_a_fifth_contradictory_frame(monkeypatch):
    """The expanded recovery stays a bounded bridge, never an unmeasured read mode."""
    adb = FakeAdb([b"a", b"b", b"c", b"d", b"e", b"f", b"f"])
    drv = _drv(adb, scroll_captures=6)
    calls = []
    indexed = []

    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top", lambda: "")

    def plan(_frame, _lane, *, allow_segmentation_failure_fallback=False):
        calls.append(allow_segmentation_failure_fallback)
        if not allow_segmentation_failure_fallback:
            raise hinge.ScrollStepError("synthetic fifth contradiction")
        return SimpleNamespace(
            frac=0.4, x_frac=0.5, step_px=240, sized_against_px=1027,
            basis=hinge.COVERAGE_STEP_SEGMENTATION_FALLBACK,
            spacing=SimpleNamespace(measured=False, px=None), reason="synthetic contradiction")

    monkeypatch.setattr(drv, "_plan_enumeration_step", plan)
    monkeypatch.setattr(drv, "_index_captured_items",
                        lambda photos, should_stop=None: indexed.append(list(photos)) or "")

    assert drv._capture_current() is not None
    assert calls == [True, True, True, True, False]
    assert indexed == []


def test_capture_refuses_a_second_disjoint_contradictory_run(monkeypatch):
    """The indexer can omit one contiguous window, never two separated bad-frame runs."""
    adb = FakeAdb([b"a", b"b", b"c", b"c"])
    drv = _drv(adb, scroll_captures=4)
    calls = []
    indexed = []

    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top", lambda: "")

    def plan(_frame, _lane, *, allow_segmentation_failure_fallback=False):
        calls.append(allow_segmentation_failure_fallback)
        if len(calls) == 3:
            assert not allow_segmentation_failure_fallback
            raise hinge.ScrollStepError("synthetic disjoint contradiction")
        return SimpleNamespace(
            frac=0.4, x_frac=0.5, step_px=240, sized_against_px=1027,
            basis=(hinge.COVERAGE_STEP_SEGMENTATION_FALLBACK if len(calls) == 1
                   else "measured"),
            spacing=SimpleNamespace(measured=False, px=None),
            reason="synthetic contradiction")

    monkeypatch.setattr(drv, "_plan_enumeration_step", plan)
    monkeypatch.setattr(drv, "_index_captured_items",
                        lambda photos, should_stop=None: indexed.append(list(photos)) or "")

    profile = drv._capture_current()

    assert profile is not None
    assert calls == [True, True, False]
    assert indexed == []
    assert "frame 2 of this profile" in profile.items_unavailable
    assert "synthetic disjoint contradiction" in profile.items_unavailable


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
    # 2026-08-24: the per-frame dwell this test used to pin down (one dwell + one settle per
    # scroll, in strict alternation) no longer exists -- see _READ_PAUSE_COUNTS in hinge.py and
    # ops/ANTI-BOT-RESEARCH.md's dated addendum. Pin the read's one randomized pause to "none
    # drawn" so this test keeps proving what it always meant to prove -- the REAL production
    # policy's scroll geometry (frac/lane) varies frame to frame and threads correctly into
    # metadata -- without also having to hardcode a pause position the sleep-call bookkeeping
    # below would otherwise have to guess at. The pause draw itself is covered by its own
    # dedicated tests just below (test_read_pause_count_and_position_are_drawn_not_fixed and
    # test_read_pause_duration_is_drawn_and_fully_credited_when_it_fires).
    monkeypatch.setattr(HingeDriver, "_plan_read_pause_iterations", lambda self, _n: set())
    adb = FakeAdb([f"frame-{i}".encode() for i in range(12)])
    drv = _drv(adb, scroll_captures=8)
    policy = AutoSessionPolicy(rng=random.Random(19), local_hour=lambda: 15)
    drv.set_auto_session_policy(policy)
    slept = []
    monkeypatch.setattr(hinge.time, "sleep", lambda seconds: slept.append(seconds))
    sleep_requests = []
    real_interruptible_sleep = drv._interruptible_sleep

    def _record_sleep(seconds, should_stop=None):
        sleep_requests.append(seconds)
        return real_interruptible_sleep(seconds, should_stop)

    monkeypatch.setattr(drv, "_interruptible_sleep", _record_sleep)

    profile = drv._capture_current()

    assert 8 <= len(profile.photos) <= 10
    assert len(drv._capture_scroll_ledger) == len(profile.photos) - 1
    assert len({round(frac, 4) for frac, _lane in drv._capture_scroll_ledger}) > 1
    assert len({round(lane, 4) for _frac, lane in drv._capture_scroll_ledger}) > 1
    assert len({round(seconds, 4) for seconds in slept}) > 1
    # With no pause drawn, every _interruptible_sleep call is the post-scroll UI settle -- one
    # per read-scroll, no read-dwell call at all. The settle represents animation latency rather
    # than time spent reading, so it must not inflate read_dwell_s_total.
    assert len(sleep_requests) == len(drv._capture_scroll_ledger)
    assert all(seconds > 0 for seconds in sleep_requests)
    assert profile.meta["app"] == "hinge"
    assert profile.meta["capture_frames"] == len(profile.photos)
    assert profile.meta["read_scrolls"] == len(drv._capture_scroll_ledger)
    # No pause was drawn for this read, so it must claim none of Signals behaviour #1's dwell
    # credit -- same "an abandoned/undrawn read never claims dwell it didn't spend" rule the old
    # per-frame site already enforced for a Stop landing mid-dwell.
    assert profile.meta["read_dwell_s_total"] == 0.0


def test_read_pause_count_and_position_are_drawn_not_fixed():
    """`_plan_read_pause_iterations` decides, once per read, how many of its iterations (if any)
    get the profile's one attention pause and which ones -- see `_READ_PAUSE_COUNTS` in hinge.py
    and ops/ANTI-BOT-RESEARCH.md's 2026-08-24 addendum for why a FIXED position, or a hardcoded
    "always exactly one" count, would just be the old per-frame metronome wearing a smaller
    disguise. Both have to actually vary across reads, not merely be capable of it.
    """
    random.seed(4)
    drv = _drv(FakeAdb([b"x"]))
    eligible = 10
    draws = [drv._plan_read_pause_iterations(eligible) for _ in range(500)]

    # The count is not hardcoded to exactly one: over 500 draws at 15%/75%/10% weights, both a
    # zero-pause and a one-pause read are certain to appear.
    counts = {len(draw) for draw in draws}
    assert counts >= {0, 1}, f"expected at least 0- and 1-pause reads among 500 draws, got {counts}"
    # Every returned position is a real candidate iteration, and a read never plans more pauses
    # than there are iterations to place them in.
    assert all(0 <= i < eligible for draw in draws for i in draw)
    assert all(len(draw) <= eligible for draw in draws)

    single_pause_positions = {next(iter(draw)) for draw in draws if len(draw) == 1}
    assert len(single_pause_positions) > 1, (
        "the pause position must vary across reads, not settle on one constant iteration")


def test_read_pause_duration_is_drawn_and_fully_credited_when_it_fires(monkeypatch):
    """End-to-end analogue of the old per-frame dwell-accounting test, for the new single-pause
    site: when the read's one pause DOES land on an iteration, its duration is a genuine draw
    (not a hardcoded literal), `_interruptible_sleep` is actually asked to sleep that exact
    amount, and `read_dwell_s_total` is credited with exactly what was slept -- nothing more,
    nothing less. Position is pinned (to iteration 0) so this test is about the DURATION draw
    specifically; position variety has its own test just above.
    """
    monkeypatch.setattr(HingeDriver, "_plan_read_pause_iterations", lambda self, _n: {0})

    def _one_run(seed):
        random.seed(seed)
        adb = FakeAdb([b"a", b"b", b"c"])
        drv = _drv(adb, scroll_captures=3, dwell_s=1.1)
        sleep_requests = []
        real_interruptible_sleep = drv._interruptible_sleep

        def _record(seconds, should_stop=None):
            sleep_requests.append(seconds)
            return real_interruptible_sleep(seconds, should_stop)

        monkeypatch.setattr(drv, "_interruptible_sleep", _record)
        profile = drv._capture_current()
        return profile, sleep_requests

    profile_a, sleeps_a = _one_run(10)
    profile_b, sleeps_b = _one_run(11)

    # Iteration 0: pause then settle. Iteration 1: settle only (pause_iterations == {0}).
    assert len(sleeps_a) == 3 and len(sleeps_b) == 3
    pause_a, pause_b = sleeps_a[0], sleeps_b[0]
    assert pause_a > 0 and pause_b > 0
    assert pause_a != pause_b, "the pause duration must be drawn, not a fixed constant"
    assert profile_a.meta["read_dwell_s_total"] == pytest.approx(pause_a)
    assert profile_b.meta["read_dwell_s_total"] == pytest.approx(pause_b)


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
    send = _INLINE_CONFIRM                          # vision-located inline Send Like glyph
    assert adb.taps == [heart, send]
    assert adb.texts == []                       # no comment typed


def test_like_with_capture_order_opener_is_refused_before_heart_and_send():
    adb = FakeAdb([_action_frame()])
    with pytest.raises(hinge.HingeTargetingError, match="model item index"):
        _drv(adb).like("loved your stargazing prompt", 0)
    assert adb.taps == [] and adb.texts == []


# --- observe: wait_for_decision via frame deltas ------------------------
def test_pass_detected_on_card_advance(monkeypatch):
    # PASS requires a positive different identity on both the candidate and its settle frame,
    # in addition to deck readiness. Raw fake frames have no such pixels by themselves, so give
    # this focused positive-path test explicit old/new identity bands.
    import numpy as np
    adb = FakeAdb([b"a", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"b" else old_sig)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")
    assert drv.wait_for_decision(timeout=5.0) is False


def test_waits_until_a_change_then_pass(monkeypatch):
    import numpy as np
    adb = FakeAdb([b"a", b"a", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"b" else old_sig)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)  # deck-ready evidence -- see above
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")
    assert drv.wait_for_decision(timeout=None) is False


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
    # A structurally observed sheet, then a stable identity-proven new deck -> like sent.
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"sheet", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"b" else old_sig)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"b")
    assert drv.wait_for_decision(timeout=5.0) is True


def test_like_writes_its_own_decision_record(monkeypatch):
    """A LIKE used to write NO decision record at all: observe_decision hardcoded
    decision="pass", and the two `return True` sites logged nothing. observe_like_anchor fires
    on INTENT (a sheet was spotted), not on resolution, so an actions.jsonl reader could not
    tell a sent like from a sheet the human opened and backed out of. Without a resolution
    record, one profile's capture can be followed by an anchor and another profile's capture,
    with nothing on disk proving the first profile was liked."""
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"sheet", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._identity_name = "profile_a"
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"b" else old_sig)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"b")

    assert drv.wait_for_decision(timeout=5.0) is True

    decisions = [f for name, f in drv._dbg.calls if name == "observe_decision"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "like"
    assert decisions[0]["profile_name"] == "profile_a"
    # Layer 3 measures distance to the PASS control, so running it on a like would confidently
    # report "resync" for a tap that correctly hit the heart. An honest absence beats a wrong answer.
    assert decisions[0]["gesture"] == "not_checked"
    assert decisions[0]["sheet_seen"] is True


def test_like_decision_is_evidenced_by_the_composer_not_a_stale_pre_tap_frame(monkeypatch):
    """The frame filed with a LIKE must be one that actually showed the composer.

    _await_like_resolved holds `base` frozen for the whole wait, on purpose -- it is the
    dismissal comparand. Logging that same `base` as the decision's evidence made the record
    show whatever the screen held when the wait STARTED. In the audited 2026-08-14 run that
    was a scrolled profile from five minutes before the heart was even tapped, so the one
    frame stored to prove a like showed a screen the human had already left.
    """
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"sheet1", b"sheet2", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._identity_name = "profile_a"
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"b" else old_sig)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame.startswith(b"sheet"))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"b")

    assert drv.wait_for_decision(timeout=5.0) is True

    evidence = dict(drv._dbg.befores)["observe_decision"]
    assert evidence == b"sheet2"        # the LAST composer state -- what the human sent
    assert evidence != b"a"             # not the frozen pre-tap anchor
    # The resolved composer must not survive into the next wait on another profile.
    assert drv._observe_like_evidence is None


def test_dismissed_like_sheet_is_recorded_but_never_as_a_decision(monkeypatch):
    import numpy as np
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"sheet", b"a", b"c"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"c" else old_sig)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    # This test's deliberately tiny deadline is about composer resolution, not cold-loading
    # the unrelated paywall template on a fresh xdist worker.
    monkeypatch.setattr(drv, "_deck_blocked_reason", lambda _frame: None)
    # A real, matchable Send Like glyph while the sheet is up, gone once it is dismissed --
    # this is what separates a genuine dismissal from the sibling test below. Counted rather
    # than keyed on frame bytes because FakeAdb replays one frame until a scroll advances it.
    polls = {"n": 0}

    def _sheet_visible(_frame):
        polls["n"] += 1
        return polls["n"] <= 2                   # up for the first two polls, then dismissed

    monkeypatch.setattr(drv, "_observe_like_sheet_visible", _sheet_visible)

    assert drv.wait_for_decision(timeout=0.2) is None        # no stable positive new identity

    names = [name for name, _f in drv._dbg.calls]
    assert "observe_like_dismissed" in names                 # the backed-out sheet left a trace...
    decisions = [f for name, f in drv._dbg.calls if name == "observe_decision"]
    assert decisions == []                                      # ...and no label was guessed


def test_bottom_delta_with_no_sheet_is_not_logged_as_a_dismissed_like(monkeypatch):
    """A bottom-only change is not proof of a like sheet: a snackbar ("Your like was sent"), a
    toast, or a keyboard dismissal produces the same delta. Recording those as a dismissed like
    sheet would assert something nobody ever observed, in the log you read specifically to find
    out what the human did."""
    import numpy as np
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"snackbar", b"a", b"c"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"c" else old_sig)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: False)   # never a sheet

    assert drv.wait_for_decision(timeout=0.2) is None

    names = [name for name, _f in drv._dbg.calls]
    assert "observe_bottom_delta" in names
    assert "observe_like_dismissed" not in names
    assert not [name for name, _f in drv._dbg.calls if name == "observe_decision"]


def test_unobserved_bottom_delta_on_an_identity_proven_card_keeps_waiting(monkeypatch):
    """Exact 2026-08-15 regression: the owner scrolled up and down to READ a profile, pressed
    nothing, and the run abandoned the card anyway -- with the name at the top of the phone
    unchanged the whole time.

    Mechanism, which is also the 2026-08-14 Hayley setup right up to the last step: a
    read-scroll makes a bottom-only delta, which enters _await_like_resolved speculatively, and
    the manual scroll offset then matches none of the coarse capture-time downsamples. The
    sticky header still names the captured profile, and 'same' is authoritative for the outer
    loop on every poll -- so it must be authoritative HERE too, where the only two outcomes are
    "keep waiting" and "throw the profile away". Hayley's invariant is unchanged (never a LIKE);
    what is added is that it must not resync either.
    """
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (0.0, 0.0)))
    adb = FakeAdb([b"base", b"scroll", b"processing", b"steady"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    import numpy as np
    identity = np.full((16, 64), 10, dtype="int16")
    captured_content = np.full((24, 24), 10, dtype="int16")
    scrolled_content = np.full((24, 24), 200, dtype="int16")
    drv._identity_sig = identity
    drv._identity_top_sig = np.full((16, 64), 150, dtype="int16")
    drv._current_sigs = [captured_content]
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect: identity)
    monkeypatch.setattr(hinge, "_downsample", lambda *_a, **_k: scrolled_content)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"steady")

    # The two halves of the mechanism, asserted directly so a future change to either one
    # cannot quietly make this test pass for the wrong reason.
    assert drv._is_current_profile_frame(b"steady") is True                       # header: same card
    assert drv._is_current_profile_frame(b"steady", require_content=True) is False  # photos: no match

    sent, notified = drv._await_like_resolved(b"base", None, lambda: False,
                                              intent_notified=False)

    assert sent is False            # "no like happened" -> the outer loop keeps this card
    assert notified is False
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_resync"]
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_unobserved_bottom_delta_still_resyncs_when_identity_cannot_name_the_card(monkeypatch):
    """The other side of the same rule, and the reason it is not simply "always keep waiting".

    When the anchor CANNOT say this is the captured profile, the card really may have advanced
    underneath us with no sheet ever observed. Continuing to wait there would leave the loop
    watching a card whose _current_sigs/profile are stale and file the human's NEXT decision
    against the person they already left. So that stays an unlabeled resync -- never a LIKE --
    and the record now carries the identity verdicts that made the call.
    """
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (0.0, 0.0)))
    adb = FakeAdb([b"base", b"scroll", b"steady", b"steady"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    import numpy as np
    identity = np.full((16, 64), 10, dtype="int16")
    other = np.full((16, 64), 250, dtype="int16")
    captured_content = np.full((24, 24), 10, dtype="int16")
    scrolled_content = np.full((24, 24), 200, dtype="int16")
    drv._identity_sig = identity
    drv._identity_top_sig = np.full((16, 64), 150, dtype="int16")
    drv._current_sigs = [captured_content]
    # A DIFFERENT profile's header on every frame the resolver sees.
    monkeypatch.setattr(hinge, "_band",
                        lambda frame, _rect: identity if frame == b"base" else other)
    monkeypatch.setattr(hinge, "_downsample", lambda *_a, **_k: scrolled_content)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda _frame: True)

    sent, notified = drv._await_like_resolved(b"base", None, lambda: False,
                                              intent_notified=False)

    assert sent is None             # unlabeled resync, never a LIKE
    assert notified is False
    resyncs = [fields for name, fields in drv._dbg.calls if name == "observe_resync"]
    assert resyncs == [{"reason": "like_candidate_without_observed_sheet",
                        "sheet_seen": False, "profile_name": None,
                        "identity": "new", "confirm_identity": "new",
                        "current": False, "deck_ready": True}]
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_observed_sheet_keeps_the_strict_content_rule_for_a_same_named_next_card(monkeypatch):
    """The asymmetry must not leak the other way. Once a composer HAS been observed, a wrong
    "still the current profile" silently drops a like the human really sent, so that state
    still demands the header AND the photos agree before calling a frame current."""
    import numpy as np
    adb = FakeAdb([b"steady", b"steady"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    identity = np.full((16, 64), 10, dtype="int16")
    drv._identity_sig = identity
    drv._identity_top_sig = np.full((16, 64), 150, dtype="int16")
    drv._current_sigs = [np.full((24, 24), 10, dtype="int16")]
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect: identity)     # same first name
    monkeypatch.setattr(hinge, "_downsample",
                        lambda *_a, **_k: np.full((24, 24), 200, dtype="int16"))
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda _frame: True)

    sent, notified = drv._await_like_resolved(b"base", None, lambda: False,
                                              intent_notified=True)

    # Not a dismissal (which is what require_content=False would have concluded here), and not
    # a fabricated LIKE either: identity cannot prove a NEW deck, so it resyncs.
    assert sent is None
    assert notified is True
    resyncs = [fields for name, fields in drv._dbg.calls if name == "observe_resync"]
    assert [f["reason"] for f in resyncs] == ["like_send_identity_unproven"]


def test_observed_sheet_treats_a_verified_content_scroll_as_a_dismissal(monkeypatch):
    """A review scroll after opening a composer stays on the same profile.

    The same-name guard remains strict: this succeeds only because the current card's content
    also matches a captured frame at a measured vertical offset.
    """
    import numpy as np

    adb = FakeAdb([b"scrolled", b"scrolled"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    identity = np.full((16, 64), 10, dtype="int16")
    drv._identity_sig = identity
    drv._identity_top_sig = np.full((16, 64), 150, dtype="int16")

    r0, r1 = hinge._content_rows(drv.content_band, 24)
    content = np.random.default_rng(91).integers(
        0, 255, size=(r1 - r0, 24)).astype("int16")
    captured = np.full((24, 24), -50, dtype="int16")
    captured[r0:r1] = content
    scrolled = np.full((24, 24), -50, dtype="int16")
    shifted = np.full_like(content, -50)
    shifted[4:] = content[:-4]
    scrolled[r0:r1] = shifted
    drv._current_sigs = [captured]

    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect: identity)
    monkeypatch.setattr(hinge, "_downsample", lambda *_a, **_k: scrolled)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda _frame: True)

    assert drv._is_current_profile_frame(b"scrolled", require_content=True)
    sent, notified = drv._await_like_resolved(b"base", None, lambda: False,
                                              intent_notified=True)

    assert sent is False
    assert notified is True
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_resync"]


def test_observed_sheet_needs_identity_proven_new_deck_when_anchor_exists(monkeypatch):
    """A real composer does not license a label if an available identity anchor still says
    the deck is the old profile.  This includes the same-first-name/content-ambiguous case;
    resyncing is safer than assigning a LIKE to the profile held before the sheet opened.
    """
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"base", b"sheet", b"steady"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect: old_sig)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    monkeypatch.setattr(drv, "_is_current_profile_frame", lambda *_a, **_k: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"steady")

    assert drv.wait_for_decision(timeout=5.0) is None

    resyncs = [fields for name, fields in drv._dbg.calls if name == "observe_resync"]
    assert resyncs == [{"reason": "like_send_identity_unproven", "sheet_seen": True,
                        "profile_name": None, "identity": "same",
                        "confirm_identity": "same", "current": False,
                        "deck_ready": True}]
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_observed_sheet_without_identity_anchor_resyncs_never_likes(monkeypatch):
    """A real composer proves the human opened it, but not that they sent it.  Without a
    captured identity anchor a dismissal followed by a manual scroll is indistinguishable from
    a new card, so the result must stay unlabeled.
    """
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"base", b"sheet", b"steady"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    monkeypatch.setattr(drv, "_is_current_profile_frame", lambda *_a, **_k: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"steady")

    assert drv.wait_for_decision(timeout=5.0) is None

    resyncs = [fields for name, fields in drv._dbg.calls if name == "observe_resync"]
    assert resyncs == [{"reason": "like_send_identity_unavailable", "sheet_seen": True,
                        "profile_name": None, "identity": "unknown",
                        "confirm_identity": "unknown", "current": False,
                        "deck_ready": True}]
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_like_sheet_wait_emits_a_heartbeat_instead_of_going_silent(monkeypatch):
    """_await_like_resolved was the ONE observe wait with no heartbeat -- and the longest
    (3m44s of total console + actions.jsonl silence in the audited run, while the human typed
    a comment). Observe mode passes deadline=None, so a wedged sheet would have spun there
    forever showing the operator nothing at all."""
    clock = [1_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    drv = _drv(FakeAdb([b"sheet"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_notice = 0.0
    drv._observe_last_reason = None
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: True)

    def _stop_after_a_while():
        clock[0] += 20.0                    # each poll advances past the like-wait interval
        return clock[0] > 1_100.0

    drv._await_like_resolved(b"base", None, _stop_after_a_while)

    waiting = [f for name, f in drv._dbg.calls if name == "observe_waiting"]
    assert waiting and all(f["reason"] == "like_sheet" for f in waiting)


def test_one_negative_composer_poll_does_not_close_a_still_open_sheet(monkeypatch):
    """A missed glyph must not briefly turn an open human draft into `like_sending`.

    The live 2026-08-15 trace read like_sheet -> like_sending -> like_sheet while the same
    composer stayed open.  The resolver now confirms the first negative read; its next frame is
    again a valid sheet, so the callback remains active and only an open-sheet heartbeat occurs.
    """
    adb = FakeAdb([b"negative", b"sheet"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    callbacks = []
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        # Keep one post-screencap Stop re-check false so the sheet heartbeat is emitted
        # before the next poll sees the requested stop.
        return calls["n"] > 2

    sent, seen = drv._await_like_resolved(
        b"base", None, should_stop,
        on_like_intent=lambda active, anchor=None: callbacks.append((active, anchor)),
        intent_notified=True)

    assert sent is None and seen is True
    assert callbacks == [(True, b"sheet")]
    waiting = [fields for name, fields in drv._dbg.calls if name == "observe_waiting"]
    assert waiting and all(fields["reason"] == "like_sheet" for fields in waiting)


def test_like_sheet_wait_suppresses_heartbeat_when_stop_lands_after_its_screencap(monkeypatch):
    """The composer resolver has the same post-screencap Stop race as the ordinary wait.
    It must exit without writing a stale `like_sheet` heartbeat."""
    drv = _drv(FakeAdb([b"sheet"], advance_on_screencap=True))
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: True)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 1                 # after the resolver's screencap

    sent, seen = drv._await_like_resolved(b"base", None, should_stop, intent_notified=True)

    assert sent is None and seen is True
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_waiting"]


def test_two_negative_composer_pairs_do_not_spend_a_human_draft_as_sending(monkeypatch):
    """Transient detector failures must not turn an edited sheet into app work.

    The first negative pair is the old confirmation guard; this covers the next failure too,
    before the focused composer becomes detectable again. No like_sending heartbeat may appear
    and the recognized-sheet budget must be refreshed while we are still uncertain.
    """
    adb = FakeAdb(
        [b"negative-1", b"negative-2", b"negative-3", b"negative-4", b"sheet"],
        advance_on_screencap=True,
    )
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 3

    sent, seen = drv._await_like_resolved(
        b"base", None, should_stop, intent_notified=True)

    assert sent is None and seen is True
    waiting = [fields for name, fields in drv._dbg.calls if name == "observe_waiting"]
    assert waiting and all(fields["reason"] == "like_sheet" for fields in waiting)
    assert any(fields.get("composer_detection") == "unconfirmed" for fields in waiting)


def test_many_negative_composer_pairs_remain_unconfirmed_until_positive_closure(monkeypatch):
    """No arbitrary retry count may turn an edited draft into app-side sending.

    The 2026-08-18 device trace lost the strict composer detector for three
    confirmation pairs, then rediscovered the open sheet.  `like_sending` has
    a finite watchdog, so emitting it from those negative reads can abandon a
    human who is still composing.
    """
    adb = FakeAdb(
        [b"negative-1", b"negative-2", b"negative-3", b"negative-4",
         b"negative-5", b"negative-6", b"sheet"],
        advance_on_screencap=True,
    )
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 4

    sent, seen = drv._await_like_resolved(
        b"base", None, should_stop, intent_notified=True)

    assert sent is None and seen is True
    waiting = [fields for name, fields in drv._dbg.calls if name == "observe_waiting"]
    assert waiting and all(fields["reason"] == "like_sheet" for fields in waiting)
    assert any(fields.get("composer_detection") == "unconfirmed" for fields in waiting)


def test_focused_partial_composer_is_not_a_shared_composer_proof():
    """A selection overlay may hide the top input border while the keyboard remains open."""
    import cv2
    import numpy as np

    canvas = np.full((2400, 1080), 250, dtype=np.uint8)
    # The visible lower input outline, CTA, literal glyph, and Android keyboard reproduce the
    # read-only fallback's evidence. Deliberately omit the top input outline: strict location
    # must still refuse geometry for typing.
    canvas[1299:1302, 110:970] = 238
    canvas[1334:1443, 390:985] = 228
    glyph = hinge._load_template("hinge_send_like.png")
    gh, gw = glyph.shape
    canvas[1383 - gh // 2:1383 - gh // 2 + gh,
           695 - gw // 2:695 - gw // 2 + gw] = glyph
    canvas[1560:, :] = 43
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok

    drv = _drv(FakeAdb([]))
    frame = encoded.tobytes()
    assert drv._locate_inline_composer(frame) is None
    assert drv._focused_draft_composer_visible(frame)
    assert not drv._observe_like_sheet_visible(frame)
    assert drv._observe_like_sheet_detection == "not_visible"


def test_focused_partial_only_defers_a_prior_strict_like_intent(monkeypatch):
    """Weak focused-draft evidence can keep a known human draft open, never create intent."""
    adb = FakeAdb([b"strict-miss", b"focused-partial"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_focused_draft_composer_visible", lambda _frame: True)
    callbacks = []
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        # Focused-draft evidence also writes a heartbeat only after re-checking Stop.
        return calls["n"] > 2

    sent, seen = drv._await_like_resolved(
        b"base", None, should_stop,
        on_like_intent=lambda active, anchor=None: callbacks.append((active, anchor)),
        intent_notified=True,
    )

    assert sent is None and seen is True
    assert callbacks == []                         # no new/refreshed intent from weak evidence
    waiting = [fields for name, fields in drv._dbg.calls if name == "observe_waiting"]
    assert waiting and waiting[-1]["composer_detection"] == "focused_partial"


def test_focused_partial_never_originates_intent_or_a_like_outcome(monkeypatch):
    """A candidate with no strict composer proof remains unlabeled, even if fallback matches."""
    adb = FakeAdb([b"candidate"])
    drv = _drv(adb)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    partial_calls = []
    monkeypatch.setattr(drv, "_focused_draft_composer_visible",
                        lambda frame: partial_calls.append(frame) or True)
    callbacks = []
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 1

    result = drv.wait_for_decision(
        timeout=None, should_stop=should_stop,
        on_like_intent=lambda active, anchor=None: callbacks.append((active, anchor)),
    )

    assert result is None
    assert partial_calls == []                     # fallback cannot originate the resolver
    assert callbacks == []


def test_focused_partial_does_not_block_the_strict_only_session_top_path(monkeypatch):
    """Auto/session paths must never treat the observe-only fallback as an open sheet."""
    drv = _drv(FakeAdb([b"partial"]))
    monkeypatch.setattr(drv, "_focused_draft_composer_visible", lambda _frame: True)
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    calls = []
    monkeypatch.setattr(drv, "_scroll_to_top", lambda _stop: calls.append("scroll") or True)

    drv._ensure_session_top()

    assert calls == ["scroll"]
    assert drv._session_top_done is True


def test_waiting_notice_always_fires_when_the_reason_changes(monkeypatch):
    """Rate-limiting by CHANNEL rather than by reason hid the informative event: a
    no_change -> not_deck_ready transition is precisely the moment the screen started moving,
    and it vanished entirely if it happened within 15s of the previous notice."""
    clock = [1_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    drv = _drv(FakeAdb([b"x"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_notice = 0.0
    drv._observe_last_reason = None

    drv._note_observe_waiting("no_change")
    clock[0] += hinge._OBSERVE_NOTICE_FLOOR_S     # past the floor, far inside the 15s interval
    drv._note_observe_waiting("not_deck_ready")
    clock[0] += 1.0
    drv._note_observe_waiting("not_deck_ready")   # a repeat IS still suppressed

    reasons = [f["reason"] for name, f in drv._dbg.calls if name == "observe_waiting"]
    assert reasons == ["no_change", "not_deck_ready"]


def test_waiting_notice_floor_stops_a_flapping_screen_becoming_a_firehose(monkeypatch):
    """The reason-change bypass must not be unbounded: two branches disagreeing on a flapping
    screen (a sheet glyph matching on one poll and not the next) would otherwise alternate
    reasons every _OBSERVE_POLL_S and emit ~3 console lines a second."""
    clock = [1_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    drv = _drv(FakeAdb([b"x"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_notice = 0.0
    drv._observe_last_reason = None

    for i in range(20):                          # 20 polls at the real 0.35s cadence = 7s
        drv._note_observe_waiting("no_change" if i % 2 else "not_deck_ready")
        clock[0] += hinge._OBSERVE_POLL_S

    records = [f for name, f in drv._dbg.calls if name == "observe_waiting"]
    assert len(records) <= 1 + 7.0 / hinge._OBSERVE_NOTICE_FLOOR_S


def test_waiting_notice_logs_the_frame_the_verdict_came_from(monkeypatch):
    """The saved image used to be a SECOND screencap taken 1-2s after the verdict, so when the
    human tapped in that gap the picture showed the next state while the record beside it said
    'nothing has moved' -- misleading in the one artifact you open to diagnose a stall."""
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 1_000.0)
    adb = FakeAdb([b"fresh-screencap-that-must-not-be-used"])
    drv = _drv(adb)
    seen = {}

    class _FrameDbg(_FakeDbg):
        def action(self, name, *, before=None, after=None, **fields):
            seen["before"], seen["after"] = before, after
            super().action(name, before=before, after=after, **fields)

    drv._dbg = _FrameDbg()
    drv._observe_last_notice = 0.0
    drv._observe_last_reason = None

    drv._note_observe_waiting("no_change", b"the-frame-the-verdict-used")

    assert seen["before"] == b"the-frame-the-verdict-used"
    assert seen["after"] is None              # and no extra ADB round-trip was made for it


def test_cancelled_like_then_pass(monkeypatch):
    # Sheet up then reverts to the same card. Without a stable positive new identity afterward,
    # the conservative observer must not guess either a LIKE or a PASS.
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"sheet", b"a", b"c"])
    drv = _drv(adb)
    # Deck-ready evidence for "c" -- see test_pass_detected_on_card_advance's comment: the
    # new rule requires positive proof of a settled deck-ready card (these raw fake frames
    # never satisfy the real glyph template match) before it will conclude PASS at all.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    assert drv.wait_for_decision(timeout=0.2) is None


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
    monkeypatch.setattr("operation_love.platforms.unavailable_reason", lambda *_a, **_k: None)
    # This is an internal legacy passive-observer dependency-order test.  It intentionally
    # avoids the autonomous/training readiness gate so the missing-PIL failure remains the
    # first lifecycle error it exercises.
    drv = HingeDriver(type("ObserveCfg", (), {"mode": "observe", "apps": _Cfg.apps})())
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
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"c" else old_sig)

    # Mock _downsample to return our simulated signatures
    def mock_downsample(frame, size=24):
        if frame == b"a":
            return drv._current_sigs[0]
        if frame == b"b":
            return drv._current_sigs[1]
        return np.ones((24, 24)) * 99  # "c" is far away

    monkeypatch.setattr(hinge, "_downsample", mock_downsample)
    monkeypatch.setattr(hinge, "_split_diff", lambda x, y: (0.0, 0.0) if x == y else (15.0, 15.0))
    # Deck-ready evidence for "c" -- see test_pass_detected_on_card_advance's comment: these
    # raw fake frames can't satisfy the real glyph template match, and the new rule requires
    # positive proof of a settled deck-ready card before concluding PASS.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")

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
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"b" else old_sig)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: frame == b"sheet")
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"b")
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


# --- _vertical_shift_match: bug report regression -------------------------
# Reported: the owner scrolled a profile up and down to read it -- no pass/like tap at all --
# and the run silently recorded a PASS. Root cause: wait_for_decision's scroll-vs-pass check
# only recognized a scroll that happened to land back on one of _capture_current's own fixed
# read-scroll stops (up to 8 of them). A human's manual scroll distance almost never matches
# the bot's own stride, so it landed BETWEEN two stops and read as "whole card changed".
def test_vertical_shift_match_recognizes_a_scroll_that_lands_between_two_capture_stops():
    import numpy as np
    # Row i's value is i, so any vertical shift is unambiguous and exactly checkable.
    seen = (np.arange(24, dtype="int16").reshape(24, 1) * np.ones((1, 24), dtype="int16"))
    cur = np.full((24, 24), 250, dtype="int16")
    cur[:18] = seen[6:24]           # human scrolled 6 rows further than this capture stop
    # _vertical_shift_match now returns (matched, shift, overlap_rows), not a bare bool -- a
    # (False, ...) 3-tuple is still TRUTHY, so `assert hinge._vertical_shift_match(...)` alone
    # would silently stop being a real assertion; index [0] for the boolean this test cares
    # about (the exact shift/overlap_rows values are covered by their own dedicated test below,
    # where the fixture data is built to make them unambiguous -- this gradient data is not,
    # see that test's docstring).
    assert hinge._vertical_shift_match(cur, seen, threshold=9.0)[0]


def test_vertical_shift_match_rejects_a_genuinely_different_profile():
    import numpy as np
    seen = (np.arange(24, dtype="int16").reshape(24, 1) * np.ones((1, 24), dtype="int16"))
    cur = np.full((24, 24), 220, dtype="int16")   # nowhere near `seen`'s 0..23 range at any offset
    assert hinge._vertical_shift_match(cur, seen, threshold=9.0)[0] is False


def test_wait_for_decision_ignores_a_scroll_between_capture_stops_but_still_catches_a_real_pass(
        monkeypatch):
    """Integration regression for the bug report: a scroll landing between two of
    _capture_current's stops must NOT be recorded as a pass, but a genuinely different profile
    (no vertical alignment at any offset) still must be."""
    import numpy as np
    seen0 = np.arange(24, dtype="int16").reshape(24, 1) * np.ones((1, 24), dtype="int16")
    mid = np.full((24, 24), 250, dtype="int16")
    mid[:18] = seen0[6:24]                       # a manual scroll 6 rows past the captured stop
    new_profile = np.full((24, 24), 220, dtype="int16")   # a real X-tap advance

    def mock_downsample(frame, size=24):
        return {b"a": seen0, b"mid": mid, b"new": new_profile}[frame]

    monkeypatch.setattr(hinge, "_downsample", mock_downsample)
    monkeypatch.setattr(hinge, "_split_diff", lambda x, y: (0.0, 0.0) if x == y else (15.0, 15.0))
    adb = FakeAdb([b"a", b"mid", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._current_sigs = [seen0]
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"new" else old_sig)
    # Deck-ready evidence for "new" -- see test_pass_detected_on_card_advance's comment: the
    # new rule requires positive proof of a settled deck-ready card (these raw fake frames
    # never satisfy the real glyph template match) before it will conclude PASS at all.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")

    assert drv.wait_for_decision(timeout=5.0) is False   # only "new" is a genuine pass
    assert adb.i == 2                                    # both "mid" and "new" were observed


class _FakeDbg:
    def __init__(self):
        self.calls = []
        self.befores = []          # (name, before) -- the frame each record is evidenced by
        self.afters = []           # (name, after) -- only rare retained pair diagnostics use it

    def action(self, name, *, before=None, after=None, **fields):
        self.calls.append((name, fields))
        self.befores.append((name, before))
        self.afters.append((name, after))


def test_reviewed_observe_pass_emits_a_verified_frame_bound_decision(monkeypatch):
    drv = _drv(FakeAdb([b"before", b"after"]), halt_on_error=True)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_snap", lambda: b"before")
    monkeypatch.setattr(drv, "_deliver_decision", lambda _kind: (12, 34))
    monkeypatch.setattr(drv, "_interruptible_sleep", lambda *_a, **_k: True)
    verified = []
    monkeypatch.setattr(drv, "_verify_progress", lambda before, kind: verified.append((before, kind)))

    drv.observe_pass()

    assert verified == [(b"before", "dislike")]
    decisions = [fields for name, fields in drv._dbg.calls if name == "observe_decision"]
    assert decisions == [{"decision": "pass", "reviewed": True, "x": [12, 34],
                          "gesture": drv.spec.decide_gesture}]


def test_reviewed_capture_keeps_its_real_index_anchor_while_manual_capture_unwinds(monkeypatch):
    """The bridge starts where the index was measured; manual Observe remains top-facing."""
    profile = object()

    reviewed = _drv(FakeAdb([b"unused"]))
    reviewed_top_calls = []
    monkeypatch.setattr(reviewed, "_ensure_session_top",
                        lambda _stop=None: reviewed_top_calls.append("top") or True)
    monkeypatch.setattr(reviewed, "_capture_current", lambda _stop=None: profile)
    monkeypatch.setattr(reviewed, "_scroll_to_top",
                        lambda *_a, **_k: pytest.fail("reviewed capture must not rewind"))
    assert reviewed.current_profile_reviewed() is profile
    assert reviewed_top_calls == ["top"]

    manual = _drv(FakeAdb([b"unused"]))
    manual._session_top_done = True
    calls = []
    monkeypatch.setattr(manual, "_capture_current", lambda _stop=None: profile)
    monkeypatch.setattr(manual, "_scroll_to_top",
                        lambda *_a, **_k: calls.append("unwind") or True)
    assert manual.current_profile() is profile
    assert calls == ["unwind"]


def test_reviewed_send_does_not_fabricate_sending_for_ready_next_card(monkeypatch):
    drv = _drv(FakeAdb([b"ready-next"]))
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_is_current_profile_frame", lambda *_a, **_k: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda _frame: True)

    assert not drv._record_reviewed_like_sending_if_observed(b"before")
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_waiting"]


def test_reviewed_open_commits_anchor_and_verification_fact_inside_its_lease(monkeypatch):
    drv = _drv(FakeAdb([b"sheet"]))
    drv._dbg = _FakeDbg()
    payload = object()
    monkeypatch.setattr(drv, "_verifiable_payload", lambda _item: payload)
    monkeypatch.setattr(drv, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(drv, "_navigate_to_model_item", lambda _item, **_k: (12, 34))
    monkeypatch.setattr(drv, "_snap", lambda: b"before")
    monkeypatch.setattr(drv, "_tap", lambda *_point: None)
    monkeypatch.setattr(drv, "_interruptible_sleep", lambda *_a, **_k: True)
    monkeypatch.setattr(drv, "_await_sheet_open", lambda: None)
    monkeypatch.setattr(drv, "_verify_sheet_shows", lambda *_a, **_k: None)

    drv.observe_open_targeted_like(3)

    names = [name for name, _fields in drv._dbg.calls]
    assert names.index("observe_like_anchor") < names.index("observe_release_post_tap_item_verified")
    assert names.index("observe_release_post_tap_item_verified") < names.index("observe_reviewed_open")


def test_observe_release_facts_are_transport_free_exact_debug_rows(tmp_path):
    """Worker/provider release facts must be append-only, never a hidden screencap.

    This uses the production DebugLog rather than a loose mock so the assertion also catches
    the original ``_dbg_action(name)`` regression: _dbg_action requires its positional
    ``before`` argument and therefore raises before it can append any row.
    """
    import json

    class NoCaptureAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"unused"])
            self.capture_calls = 0

        def screencap(self):
            self.capture_calls += 1
            raise AssertionError("observe_release_fact must not use ADB screencap")

    from operation_love.drivers.debuglog import HingeDebugLog

    adb = NoCaptureAdb()
    drv = _drv(adb)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="release-facts")

    for fact in ("hub_pre_tap_published", "post_tap_item_verified",
                 "refusal_or_paywall_logged"):
        drv.observe_release_fact(fact)

    rows = [json.loads(line) for line in
            (tmp_path / "release-facts" / "actions.jsonl").read_text().splitlines()]
    assert [row["action"] for row in rows] == [
        "observe_release_hub_pre_tap_published",
        "observe_release_post_tap_item_verified",
        "observe_release_refusal_or_paywall_logged",
    ]
    assert all(set(row) == {"ts", "action"} for row in rows)
    assert adb.capture_calls == 0


def test_observe_release_fact_without_debug_log_remains_transport_free():
    """Debug-disabled production runs keep the same harmless no-op semantics."""
    class NoCaptureAdb(FakeAdb):
        def screencap(self):
            raise AssertionError("observe_release_fact must not use ADB screencap")

    drv = _drv(NoCaptureAdb([b"unused"]))
    assert drv._dbg is None
    drv.observe_release_fact("hub_pre_tap_published")


def test_wait_for_decision_records_pass_diagnostics_in_the_debug_log(monkeypatch):
    """The bug report that motivated this fix had NO actions.jsonl entry explaining why a PASS
    was recorded -- just a bare 'Got PASS' in the console log. A corroborated PASS verdict must
    leave a paper trail for the next report to point at."""
    import numpy as np
    adb = FakeAdb([b"a", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._current_sigs = [np.ones((24, 24))]
    drv._current_capture_truncated = False
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"new" else old_sig)
    monkeypatch.setattr(
        hinge, "_downsample",
        lambda frame, size=24: np.ones((24, 24)) if frame == b"a" else np.full((24, 24), 220.0))
    monkeypatch.setattr(hinge, "_split_diff", lambda x, y: (0.0, 0.0) if x == y else (15.0, 15.0))
    # Deck-ready evidence for "new" -- see test_pass_detected_on_card_advance's comment.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")

    assert drv.wait_for_decision(timeout=5.0) is False

    decisions = [fields for name, fields in drv._dbg.calls if name == "observe_decision"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "pass"
    assert decisions[0]["shift_matched"] is False
    assert decisions[0]["capture_truncated"] is False
    assert decisions[0]["top"] == 15.0 and decisions[0]["bot"] == 15.0
    assert decisions[0]["identity"] == "new"
    assert decisions[0]["confirm_identity"] == "new"
    assert decisions[0]["confirm_identity_dist"] is not None
    assert decisions[0]["gesture"] == "pass"


def test_observe_scroll_shift_match_record_carries_sig_index_shift_overlap_rows_and_safe_ocr_metadata(
        monkeypatch):
    """Observability fix motivated directly by the reported incident: the exact record type
    that was written while wait_for_decision was actually looking at a DIFFERENT woman's
    profile (reason=shift_match) used to say only "matched a stored signature", with no way
    to see which one, at what offset, or how much of the band even overlapped. These fields
    are what turn that into "matched stored frame 0 at a 5-row shift with only 13 of 18 band
    rows overlapping".

    Random content at a known offset (not a smooth gradient) is used so the shift/overlap
    this test asserts are unambiguous -- see
    test_vertical_shift_match_reports_which_shift_matched_and_how_much_overlapped's docstring
    for why a gradient fixture can't pin an exact expected shift.
    """
    import numpy as np
    rng = np.random.default_rng(11)
    r0, r1 = hinge._content_rows(hinge.HINGE_SPEC.content_band, 24)
    band_h = r1 - r0
    content = rng.integers(0, 255, size=(band_h, 24)).astype("int16")
    seen0 = np.full((24, 24), -50, dtype="int16")          # -50: never coincidentally close to
    seen0[r0:r1] = content                                 # any 0..254 random content value

    shift = 5
    mid_content = np.full((band_h, 24), -50, dtype="int16")
    mid_content[shift:] = content[:band_h - shift]         # mid = seen's content shifted DOWN
    mid = np.full((24, 24), -50, dtype="int16")
    mid[r0:r1] = mid_content

    adb = FakeAdb([b"a", b"mid"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._current_sigs = [seen0]     # index 0 -- this IS the stored signature that matches

    def mock_downsample(frame, size=24):
        return {b"a": seen0, b"mid": mid}[frame]

    monkeypatch.setattr(hinge, "_downsample", mock_downsample)
    monkeypatch.setattr(hinge, "_split_diff", lambda x, y: (0.0, 0.0) if x == y else (15.0, 15.0))

    assert drv.wait_for_decision(timeout=0.2) is None     # times out -- correctly read as a scroll

    scrolls = [fields for name, fields in drv._dbg.calls if name == "observe_scroll"]
    assert scrolls, "expected at least one observe_scroll debug record"
    rec = scrolls[0]
    assert rec["reason"] == "shift_match"
    assert rec["sig_index"] == 0
    assert rec["shift"] == shift
    assert rec["overlap_rows"] == band_h - shift
    assert rec["band_rows"] == band_h
    assert "name_read" not in rec     # raw OCR transcript must not enter action records
    assert rec["ocr_attempts"] == []  # identity never reached "top" here (raw bytes -> "unknown")


# --- observe: identity-anchor + gesture-corroboration redesign ------------------------
# Tests 1-8, 12-13 of the redesign spec's section 4 (tests 9-10 -- TouchWatcher line
# parsing and getevent -p device selection -- already live in tests/test_touchwatch.py;
# test 11 -- AndroidAppSpec validation for identity_band/observe_touch_watch -- already
# lives in tests/test_android_spec.py; neither is duplicated here).

def test_identity_match_keeps_waiting_despite_an_unmatched_manual_scroll(monkeypatch):
    """THE reported bug, reproduced directly (the owner hit this twice). A profile is
    captured -- its sticky per-profile identity-band signature is known -- then the human
    scrolls BY HAND to a position wait_for_decision has never captured a stop for, so the
    OLD scroll-vs-pass discriminator ("changed and unrecognised by any capture-stop sig ->
    PASS") would have fired. But the identity band still reads as the SAME profile's
    header. THIS TEST WOULD HAVE FAILED against the pre-fix code: pre-fix, this exact
    scenario recorded a silent PASS with no tap of any kind, so wait_for_decision would
    have returned False here instead of continuing to wait. Post-fix, LAYER 1 (identity)
    runs FIRST and is authoritative, so it never gets there -- this asserts the wait keeps
    going and times out to None, never False.
    """
    import numpy as np
    identity_sig = np.full((16, 64), 30, dtype="int16")
    adb = FakeAdb([b"base", b"scrolled"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_sig = identity_sig                              # the captured profile's own header
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")  # scroll-top chrome, unrelated
    # This scroll position was never one of _capture_current's own stops -- neither the
    # exact-signature match nor _vertical_shift_match would recognise it (both key off
    # _current_sigs/_downsample, never reached here: identity 'same' short-circuits first).
    drv._current_sigs = [np.zeros((24, 24), dtype="int16")]
    monkeypatch.setattr(hinge, "_band",
                        lambda frame, rect: identity_sig if frame == b"scrolled" else drv._identity_top_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (50.0, 50.0))   # a big "whole card changed" delta

    assert drv.wait_for_decision(timeout=0.05) is None
    assert adb.taps == [] and adb.swipes == 0             # never touched the device either


def test_external_current_profile_or_scroll_refuses_during_observe_wait():
    """A hybrid controller must not read-scroll the phone underneath a Worker wait.

    This is the production failure behind the false Malaika PASS: a second controller called
    current_profile(), whose normal trailing _scroll_to_top changed the first driver's identity
    anchor mid-wait. The shared per-device lease refuses both the public path and the private
    unwind escape hatch before either can take a screencap or issue a swipe.
    """
    worker_adb = FakeAdb([b"worker"])
    controller_adb = FakeAdb([b"controller"])
    worker = _drv(worker_adb)
    controller = _drv(controller_adb)

    with worker._observe_input_lease("wait_for_decision"):
        with pytest.raises(hinge.HingeActionError, match="already owned"):
            controller.current_profile()
        with pytest.raises(hinge.HingeActionError, match="already owned"):
            controller._scroll_to_top()

    assert controller_adb.i == 0
    assert controller_adb.swipes == 0


def test_observe_input_lease_key_is_a_stable_non_identifying_hash_of_app_and_serial():
    """Pins the on-disk NAME `_observe_input_lease_key()` computes -- the counterweight to
    tests/conftest.py's session-scoped `_machine_global_state_is_never_the_operators` fixture,
    which redirects `tempfile.tempdir` for the whole suite so that two xdist workers using
    fixed fake serials can no longer genuinely contend over a real file via `fcntl.flock`
    (root cause A of the machine-global-state incident: two worker PROCESSES computing the
    identical lock path under the real system temp directory and actually locking each other
    out). That fixture only moves WHERE the file lives (by redirecting what
    `tempfile.gettempdir()` returns); it never touches what HingeDriver puts in the file's
    NAME, so this test calls the real method directly. `_observe_input_lease_key()` only
    formats a path string -- it opens nothing and acquires no lock either way, so this needs
    no lock and touches no real filesystem path.

    The contract that makes the shared lease actually work is the name: it must be
    `operation-love-observe-<24 lowercase hex chars>.lock`, where the hex digest is
    `sha256(f"{app}:{serial}")` truncated to 24 characters. That makes the name STABLE (the
    same app/serial always resolve to the same file, so a hub run and a separately launched
    controller sharing one real device agree on one lease to contend over), NON-IDENTIFYING
    (a hash, never the raw serial, appears in the name), and PER APP/DEVICE PAIR (two
    different phones, or one phone running two different apps, must never collide on the
    same lease file).
    """
    import hashlib
    import re
    from pathlib import Path

    class _SerialCfg:
        apps = {"hinge": {"serial": "33111JEHN04475"}}

    drv_a = HingeDriver(_SerialCfg())
    drv_b = HingeDriver(_SerialCfg())          # a second instance, identical inputs

    key_a = drv_a._observe_input_lease_key()
    key_b = drv_b._observe_input_lease_key()
    assert key_a == key_b, "same app/serial must resolve to the same lease file every time"

    name = Path(key_a).name
    match = re.fullmatch(r"operation-love-observe-([0-9a-f]{24})\.lock", name)
    assert match is not None, f"unexpected lease filename shape: {name!r}"

    expected_digest = hashlib.sha256(
        f"{drv_a.spec.app}:{drv_a.serial}".encode()).hexdigest()[:24]
    assert match.group(1) == expected_digest
    assert drv_a.serial not in name    # non-identifying: the raw serial must not leak into it

    class _OtherSerialCfg:
        apps = {"hinge": {"serial": "some-other-pixel"}}

    drv_c = HingeDriver(_OtherSerialCfg())
    assert drv_c._observe_input_lease_key() != key_a, "different serials must not share a lease"


def test_identity_new_profile_plus_deck_ready_and_settle_confirms_a_pass(monkeypatch):
    """The positive-path complement to the test above: proving the identity anchor also
    lets a REAL advance through, not just refuses to mistake a scroll for one. The identity
    band reads as a DIFFERENT profile's header (not merely 'unrecognised content'), the
    next deck is confirmed ready, a settle re-capture reconfirms both, and the pass-control
    gesture independently corroborates it -- wait_for_decision returns False (PASS)."""
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    adb = FakeAdb([b"base", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if frame == b"new" else old_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")

    assert drv.wait_for_decision(timeout=5.0) is False


def test_repeated_tight_name_cannot_override_weak_pixel_same_without_canonical_header_proof(
        monkeypatch):
    """A tight/local candidate is diagnostic only, not canonical-header advance proof."""
    import numpy as np

    adb = FakeAdb([b"allison", b"brittany"], advance_on_screencap=True)
    drv = _drv(adb, observe_touch_watch=False)
    drv._dbg = _FakeDbg()
    drv._identity_name = "Allison"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: drv._identity_sig)
    monkeypatch.setattr(
        hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "Brittany" if psm == "7" else None)

    assert drv.wait_for_decision(timeout=5.0) is None
    decisions = [fields for name, fields in drv._dbg.calls if name == "observe_decision"]
    assert not decisions


def test_two_different_tight_ocr_candidates_do_not_create_a_no_touch_pass(monkeypatch):
    """Two unrelated OCR guesses are not a repeated next-profile identity."""
    import itertools
    import numpy as np

    adb = FakeAdb([b"allison", b"candidate"], advance_on_screencap=True)
    drv = _drv(adb, observe_touch_watch=False)
    drv._dbg = _FakeDbg()
    drv._identity_name = "Allison"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: drv._identity_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (50.0, 50.0))
    monkeypatch.setattr(drv, "_changed", lambda a, b: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    names = itertools.cycle(("Brittany", "Brenda"))
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": next(names) if psm == "7" else None)

    assert drv.wait_for_decision(timeout=0.01) is None
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_same_profile_reflow_with_pixel_new_and_no_gesture_resyncs_without_a_pass(monkeypatch):
    """Exact 2026-08-16 Julia regression.

    Hinge moved the filter-chip/name/card layout while the same profile remained visible. The
    thin pixel identity band consequently read ``new`` on two settled deck-ready frames, while
    the broad card-header OCR returned garbage and touch watching was unavailable. Those facts
    may prove that pixels moved, but they do not prove a human pressed X: resync and record
    nothing instead of manufacturing a PASS.
    """
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    reflowed_sig = np.full((16, 64), 250, dtype="int16")
    adb = FakeAdb([b"base", b"same-julia-reflow"], advance_on_screencap=True)
    drv = _drv(adb, observe_name_ocr=False, observe_touch_watch=False)
    drv._dbg = _FakeDbg()
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(
        hinge, "_band",
        lambda frame, rect: reflowed_sig if frame == b"same-julia-reflow" else old_sig,
    )
    monkeypatch.setattr(
        hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)

    assert drv.wait_for_decision(timeout=5.0) is None
    resyncs = [fields for name, fields in drv._dbg.calls if name == "observe_resync"]
    assert len(resyncs) == 1
    assert resyncs[0]["reason"] == "pass_identity_name_unconfirmed"
    assert resyncs[0]["identity"] == resyncs[0]["confirm_identity"] == "new"
    assert resyncs[0]["gesture"] == "no_data"
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_identity_same_beats_a_large_top_delta_across_repeated_scrolls_then_catches_the_real_pass(
        monkeypatch):
    """identity 'same' beats _split_diff's top delta on its OWN terms, explicitly: every
    frame here -- including the two manual-scroll ones -- reports a huge top-half delta,
    exactly the signal the pre-fix rule read as 'unrecognised card -> PASS'. Only the frame
    whose identity band genuinely reads as a NEW profile ends the wait. Also proves the
    identity anchor doesn't get the observer stuck: repeated 'same' verdicts still let a
    later, genuine advance conclude normally."""
    import numpy as np
    id_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    top_sig = np.full((16, 64), 200, dtype="int16")
    band_by_frame = {b"base": id_sig, b"scr1": id_sig, b"scr2": id_sig, b"new": new_sig}

    adb = FakeAdb([b"base", b"scr1", b"scr2", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_sig = id_sig
    drv._identity_top_sig = top_sig
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: band_by_frame[frame])
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")

    assert drv.wait_for_decision(timeout=5.0) is False
    assert adb.i == 3                # every frame observed: two same-identity scrolls, then the pass


def test_vertical_shift_match_restricted_to_content_rows_ignores_fixed_chrome():
    """Regression for the shipped-but-broken whole-frame _vertical_shift_match (see its own
    docstring's measured table): on the real device the status bar / sticky header / bottom
    nav do NOT translate when content scrolls, so searching the WHOLE frame is dominated by
    that fixed chrome and never matches. Build two frames with IDENTICAL chrome rows (so
    only the chrome's PRESENCE, not its content, is under test) and content that differs
    only by a vertical shift: the whole-frame search must MISS, the content-rows-restricted
    search must HIT. A pair of genuinely different profiles (same chrome, unrelated content
    at every offset) must still be rejected either way."""
    import numpy as np
    size = 24
    r0, r1 = hinge._content_rows(hinge.HINGE_SPEC.content_band, size)
    chrome_val = 200                # identical, fixed chrome in every frame -- never translates
    content = np.arange(size, dtype="int16").reshape(size, 1) * np.ones((1, size), dtype="int16")

    seen = content.copy()
    seen[:r0] = chrome_val
    seen[r1:] = chrome_val

    shift = 6                       # a manual scroll landing between two capture stops
    cur = np.full((size, size), chrome_val, dtype="int16")
    cur[r0:r1 - shift] = seen[r0 + shift:r1]

    # [0] everywhere below: _vertical_shift_match now returns (matched, shift, overlap_rows),
    # and a (False, ...) 3-tuple is still truthy -- see the module-level regression test above
    # for why indexing is required rather than asserting the tuple itself.
    assert hinge._vertical_shift_match(cur, seen, threshold=9.0, rows=None)[0] is False
    assert hinge._vertical_shift_match(cur, seen, threshold=9.0, rows=(r0, r1))[0]

    different = np.full((size, size), 240, dtype="int16")   # a genuinely different profile
    different[:r0] = chrome_val
    different[r1:] = chrome_val
    assert hinge._vertical_shift_match(different, seen, threshold=9.0, rows=None)[0] is False
    assert hinge._vertical_shift_match(different, seen, threshold=9.0, rows=(r0, r1))[0] is False


def test_vertical_shift_match_requires_at_least_half_the_content_band():
    """Regression for the completed Marina -> Sara Training Like.

    The new Sara card coincidentally matched four rows of a captured Marina frame at shift
    +14.  Four of eighteen content rows are not enough evidence to overrule a real deck
    advance.  The weakest previously measured genuine same-profile match was exactly 9/18,
    so preserve that boundary while rejecting the incident's 4/18 collision.
    """
    import numpy as np

    rng = np.random.default_rng(17)
    size = 24
    rows = hinge._content_rows(hinge.HINGE_SPEC.content_band, size)
    r0, r1 = rows
    seen = rng.integers(0, 255, size=(size, size)).astype("int16")

    four_row_collision = rng.integers(0, 255, size=(size, size)).astype("int16")
    four_row_collision[r0 + 14:r1] = seen[r0:r1 - 14]
    matched, _shift, overlap = hinge._vertical_shift_match(
        four_row_collision, seen, threshold=0.1, rows=rows, min_overlap_rows=9)
    assert matched is False
    assert overlap >= 9

    half_band_scroll = rng.integers(0, 255, size=(size, size)).astype("int16")
    half_band_scroll[r0 + 9:r1] = seen[r0:r1 - 9]
    matched, shift, overlap = hinge._vertical_shift_match(
        half_band_scroll, seen, threshold=0.1, rows=rows, min_overlap_rows=9)
    assert matched is True
    assert shift == 9
    assert overlap == 9


def test_current_profile_check_rejects_a_four_row_collision(monkeypatch):
    """The stricter overlap policy is wired to the consequential post-Like caller."""
    import numpy as np

    drv = _drv(FakeAdb([b"sara"]))
    rng = np.random.default_rng(17)
    size = 24
    r0, r1 = hinge._content_rows(drv.content_band, size)
    marina_sig = rng.integers(0, 255, size=(size, size)).astype("int16")
    sara_sig = rng.integers(0, 255, size=(size, size)).astype("int16")
    sara_sig[r0 + 14:r1] = marina_sig[r0:r1 - 14]
    drv._current_sigs = [marina_sig]
    monkeypatch.setattr(drv, "_identity_of", lambda _frame: ("same", 42.85))
    monkeypatch.setattr(hinge, "_downsample", lambda _frame: sara_sig)
    diagnostics = {}

    assert drv._is_current_profile_frame(
        b"sara", require_content=True, diagnostics=diagnostics) is False
    assert diagnostics["current_content_shift_matched"] is False
    assert diagnostics["current_profile_result"] is False


def test_vertical_shift_match_reports_which_shift_matched_and_how_much_overlapped():
    """New return shape (observability fix): _vertical_shift_match used to return a bare
    bool, which is exactly why the reported incident's observe_scroll record said only
    "matched a stored signature" with no way to see WHICH one, at what offset, or how much of
    the band even overlapped. Random (not smoothly-graded) content is used deliberately here,
    unlike the gradient fixtures above: a smooth np.arange ramp is close to SEVERAL shifts at
    once (small shift errors produce small, still-under-threshold diffs), which makes the
    exact returned shift depend on scan order rather than being a clean, single fact to
    assert. An exact copy embedded in random noise at ONE known offset has no such ambiguity:
    every other offset compares unrelated random values and lands far over threshold."""
    import numpy as np
    rng = np.random.default_rng(3)
    size = 24
    seen = rng.integers(0, 255, size=(size, size)).astype("int16")
    cur = np.zeros((size, size), dtype="int16")
    shift = 5
    cur[shift:] = seen[:size - shift]      # cur's content is `seen` shifted DOWN by 5 rows

    matched, reported_shift, overlap_rows = hinge._vertical_shift_match(cur, seen, threshold=9.0)

    assert matched is True
    assert reported_shift == shift
    assert overlap_rows == size - shift

    # The no-match path also reports its BEST (lowest-diff) attempt, purely as a diagnostic --
    # unrelated random content should not clear the threshold at any offset.
    unrelated = rng.integers(0, 255, size=(size, size)).astype("int16")
    matched2, _shift2, overlap2 = hinge._vertical_shift_match(unrelated, seen, threshold=9.0)
    assert matched2 is False
    assert overlap2 > 0        # still a real (best-effort) overlap count, not a placeholder


# --- observe: card-header OCR resolves scroll-top, and vetoes transient pixel-new ------------
# At scroll-top identity_band contains profile-independent filter chips. The card-header OCR
# crop can resolve a different name there, but that path must reproduce on a second frame before
# PASS; when pixels already say ``new``, a nonmatch never strengthens that weaker evidence.

def _top_state_drv(monkeypatch, *, stored_name="Zorva", **cfg):
    """A driver whose pixel identity verdict is pinned to exactly 'top' -- the SAME per-call
    setup every test below needs, so only the OCR stub (and the stored name, for the
    near-miss test) differs between callers."""
    import numpy as np
    drv = _drv(FakeAdb([b"frame"]), **cfg)
    drv._identity_name = stored_name
    drv._identity_top_sig = np.zeros((16, 64), dtype="int16")   # matches whatever `_band` returns below
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: np.zeros((16, 64), dtype="int16"))
    return drv


def test_identity_top_name_ocr_resolves_top_to_new_when_a_different_name_is_read(monkeypatch):
    """At measured scroll-top geometry, a clean different name resolves the generic top
    signature to a candidate ``new``; wait_for_decision separately requires it twice."""
    drv = _top_state_drv(monkeypatch)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "qelix" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "new"


def test_identity_top_name_ocr_resolves_top_to_same_when_the_stored_name_is_read(monkeypatch):
    """Complement of the above: reading the SAME stored name resolves 'top' to 'same' instead
    of leaving the wait loop stuck on an inconclusive verdict."""
    drv = _top_state_drv(monkeypatch)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "Zorva" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "same"


def test_identity_top_name_ocr_near_miss_reads_stay_same_the_false_pass_guard(monkeypatch):
    """MEASURED calibration for _NAME_MATCH_RATIO: 'Zorva' vs plausible OCR misreads 'Zorba'
    and 'Zorna' scores 0.80 -- both MUST stay 'same', because a false 'new'
    here records a PASS the human never made and corrupts the taste model, while a false
    'same' only ever costs a missed wait (see _NAME_MATCH_RATIO's own module comment for the
    full asymmetry argument). Exercise both measured misreads."""
    for misread in ("Zorba", "Zorna"):
        drv = _top_state_drv(monkeypatch)
        monkeypatch.setattr(drv, "_ocr_band",
                            lambda frame, rect, psm="7", _r=misread: _r if psm == "6" else None)

        state, _dist = drv._identity_of(b"frame")

        assert state == "same", f"OCR misread {misread!r} of the stored name must stay 'same'"


def test_identity_top_name_ocr_distinguishes_sara_from_marina(monkeypatch):
    """Regression: a clean repeated Sara header is not an OCR variant of Marina.

    Their SequenceMatcher score is exactly 0.60, which the former inclusive 0.60 fuzzy-name
    boundary accepted as ``same``. That swallowed a completed Training Like after Hinge had
    already advanced to Sara. The measured Zorva/Zorba and Zorva/Zorna OCR positives remain
    safely above the recalibrated boundary at 0.80.
    """
    drv = _top_state_drv(monkeypatch, stored_name="Marina")
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: "Sara" if psm == "6" else None)

    state, _dist = drv._identity_of(b"sara-after-like")

    assert state == "new"
    assert drv._identity_top_name_verdict == "new"
    assert drv._identity_name_candidate == "Sara"


def test_passive_identity_keeps_sofia_sophia_fuzzy_match_conservative(monkeypatch):
    """A close spelling conflict alone cannot manufacture a passive decision.

    Sofia/Sophia scores 0.727, below the measured 0.80 Zorva OCR near-misses but above the
    shared conservative fuzzy boundary. Training may resolve this only with its additional
    post-action content and stability proof; `_identity_of` alone must keep saying same.
    """
    drv = _top_state_drv(monkeypatch, stored_name="Sophia")
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: "Sofia" if psm == "6" else None)

    state, _dist = drv._identity_of(b"sofia-after-like")

    assert state == "same"
    assert drv._identity_top_name_verdict == "same"
    assert drv._identity_name_candidate is None
    assert drv._identity_top_name_read == "Sofia"
    assert drv._identity_top_name_read_source == "top_card_header"


def test_identity_top_name_ocr_all_chrome_words_stays_top_inconclusive(monkeypatch):
    """A read that caught nothing but Hinge's own chrome (the name itself went unread -- cut
    off, misrecognised, whatever) must leave the verdict at 'top' (inconclusive), never be
    promoted to a false 'new' -- see _TOP_NAME_CHROME_WORDS's own module comment for exactly
    this failure mode."""
    drv = _top_state_drv(monkeypatch)
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: (
            "Signals Active today" if psm == "6" else None))

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"


def test_identity_top_name_ocr_garbage_with_multiple_candidates_stays_inconclusive(monkeypatch):
    """Exact OCR side of the 2026-08-16 Julia regression: a broad/misaligned crop can include
    photo texture and hallucinate many word-like tokens. None is positive evidence of one clean
    different profile name, so the generic scroll-top verdict must remain inconclusive."""
    drv = _top_state_drv(monkeypatch, stored_name="Julia")
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: (
            "Neh eae Bk hySey spate Batya Marina SENS Cie" if psm == "6" else None
        ),
    )

    state, _dist = drv._identity_of(b"same-julia-reflow")

    assert state == "top"
    assert drv._identity_top_name_verdict is None


def test_identity_top_name_ocr_uses_the_name_line_not_pronouns_or_activity(monkeypatch):
    """Regression for the 2026-08-16 Sammy -> Jen missed X report.

    Tesseract correctly returned ``Jen`` on line 1 and Hinge's ``she/her Active now`` metadata
    on line 2.  Treating the whole block as an unordered bag of candidate words made the clean
    name ambiguous, after which the loose content matcher swallowed the real card advance as a
    scroll.  The measured header geometry gives line 1 a distinct meaning: it is the name.
    """
    drv = _top_state_drv(monkeypatch, stored_name="Sammy")
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7": "Jen\nshe her Active now" if psm == "6" else None,
    )

    state, _dist = drv._identity_of(b"jen-after-x")

    assert state == "new"
    assert drv._identity_top_name_verdict == "new"


def test_identity_top_name_pronoun_cannot_fuzzy_match_a_short_stored_name(monkeypatch):
    """The Ery -> Roisin incident: ``her`` scored as a fuzzy match for stored ``Ery``.

    Pronouns are Hinge header metadata, not alternate spellings of the profile name. The clean
    first-line Roisin read must remain a new-name candidate for the caller's repeated-frame gate.
    """
    drv = _top_state_drv(monkeypatch, stored_name="Ery")
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7": (
            "Roisin\nshe her\nMe in the wild" if psm == "6" else None),
    )

    state, _dist = drv._identity_of(b"roisin-after-x")

    assert state == "new"
    assert drv._identity_top_name_verdict == "new"
    assert drv._identity_name_candidate == "Roisin"


def test_tight_name_ocr_overrules_a_weak_pixel_same_for_a_different_profile(monkeypatch):
    """Regression for the Allison -> Brittany missed X incident.

    The two sticky headers measured only 6.81 apart because their shared white chrome dominated
    the thin pixel band, below the 9.0 ``same`` threshold. The same tight band nevertheless OCR'd
    ``Brittany`` cleanly. That positive different-name evidence must remain eligible for the
    caller's settled two-frame proof instead of adopting Brittany as Allison's new baseline.
    """
    import numpy as np

    drv = _drv(FakeAdb([b"brittany-scrolled"]))
    drv._identity_name = "Allison"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: drv._identity_sig)
    psms = []

    def ocr(frame, rect, psm="7"):
        psms.append(psm)
        return "Brittany" if psm == "7" else None

    monkeypatch.setattr(drv, "_ocr_band", ocr)

    state, distance = drv._identity_of(b"brittany-scrolled")

    assert state == "same"                      # OCR is only a candidate, never a one-frame verdict
    assert distance == 0.0                       # the pixel layer alone said an exact "same"
    assert drv._identity_top_name_verdict == "new"
    assert drv._identity_name_candidate == "Brittany"
    assert psms == ["7"]                         # the clean tight read needs no broad photo OCR


def test_tight_name_ocr_near_match_preserves_pixel_same(monkeypatch):
    """A normal OCR spelling error must retain the conservative same-profile verdict."""
    import numpy as np

    drv = _drv(FakeAdb([b"allison"]))
    drv._identity_name = "Allison"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: drv._identity_sig)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "Alllson" if psm == "7" else None)

    assert drv._identity_of(b"allison")[0] == "same"
    assert drv._identity_name_candidate is None


def test_identity_top_name_ocr_vetoes_a_pixel_new_when_the_stored_name_is_read(monkeypatch):
    """The card header closes the incident's exact hole: a transition can make the thin
    identity band look pixel-``new``, but reading the captured name is positive same-card proof.
    A nonmatch remains inert; only this matching direction may override the pixels."""
    import numpy as np
    psms_seen = []

    drv = _drv(FakeAdb([b"frame"]))
    drv._identity_name = "Zorva"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")           # id_sig set
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    # band far from BOTH id_sig and top_sig -> pixel verdict is 'new' outright
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: np.full((16, 64), 99, dtype="int16"))
    def matching_ocr(frame, rect, psm="7"):
        psms_seen.append(psm)
        return "Zorva" if psm == "6" else None

    monkeypatch.setattr(drv, "_ocr_band", matching_ocr)

    state, _dist = drv._identity_of(b"frame")

    assert state == "same"
    assert "6" in psms_seen


def test_identity_top_name_ocr_nonmatch_does_not_strengthen_pixel_new(monkeypatch):
    """The additional OCR read for a pixel-``new`` transition has only one safe direction:
    a stored-name match vetoes it. A nonmatch leaves the pixel verdict untouched rather than
    declaring a second, OCR-derived reason to label a PASS."""
    import numpy as np
    drv = _drv(FakeAdb([b"frame"]))
    drv._identity_name = "Zorva"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: np.full((16, 64), 99, dtype="int16"))
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7", **_kwargs: (
                            "qelix" if psm == "6" else None))

    state, _dist = drv._identity_of(b"frame")

    assert state == "new"
    assert drv._identity_top_name_verdict is None


def test_canonical_top_with_changed_filter_chips_uses_name_proof(monkeypatch):
    """Dynamic top chrome must not hide a real next card from the repeated-name gate.

    A Signals chip can make the next card's raw filter row differ from the capture-local top
    signature. Only the canonical top detector licenses interpreting the broad header OCR as a
    name; the ordinary non-top pixel-new case remains veto-only in the regression above.
    """
    import numpy as np

    drv = _drv(FakeAdb([b"kate-at-top"]))
    drv._identity_name = "Mackinley MJ"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    # Far from BOTH stored bands, reproducing the incident's provisional pixel-new verdict.
    monkeypatch.setattr(
        hinge, "_band", lambda frame, rect: np.full((16, 64), 99, dtype="int16"))
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda frame, **_kw: SimpleNamespace(confirmed=True))
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7": (
            "Kate\nshe her\nLet me introduce you" if psm == "6" else None))

    state, _distance = drv._identity_of(b"kate-at-top")

    assert state == "new"
    assert drv._identity_top_name_verdict == "new"
    assert drv._identity_name_candidate == "Kate"


def test_identity_top_name_ocr_skipped_when_observe_name_ocr_is_off(monkeypatch):
    """observe_name_ocr=False must disable this layer exactly like it disables the existing
    identity_band OCR corroboration above it -- no OCR call of any kind, not even a wasted
    one whose result is then discarded."""
    calls = []
    drv = _top_state_drv(monkeypatch, observe_name_ocr=False)
    monkeypatch.setattr(drv, "_ocr_band", lambda frame, rect, psm="7": calls.append(psm))

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"
    assert calls == []


def test_identity_top_name_ocr_skipped_when_no_stored_name(monkeypatch):
    """No captured self._identity_name (this profile's capture never revealed the sticky
    header) means there is nothing to compare a read against -- must not run, not even to
    compare against an empty/None stored name."""
    import numpy as np
    calls = []
    drv = _drv(FakeAdb([b"frame"]))
    drv._identity_name = None
    drv._identity_top_sig = np.zeros((16, 64), dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: np.zeros((16, 64), dtype="int16"))
    monkeypatch.setattr(drv, "_ocr_band", lambda frame, rect, psm="7": calls.append(psm))

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"
    assert calls == []


def test_identity_top_name_ocr_skipped_when_band_not_declared(monkeypatch):
    """identity_top_name_band=None (its own default -- an app never measured for one) must
    behave exactly as it did before this feature existed: 'top' stays inconclusive."""
    calls = []
    drv = _top_state_drv(monkeypatch)
    drv.identity_top_name_band = None
    monkeypatch.setattr(drv, "_ocr_band", lambda frame, rect, psm="7": calls.append(psm))

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"
    assert "6" not in calls


def test_scroll_top_pass_swallowed_by_content_match_is_now_correctly_a_pass_via_name_ocr(
        monkeypatch):
    """A repeatedly read different name at scroll-top gives the real-advance path a positive
    ``new`` identity on both frames, before Layer 2 can swallow a lookalike new card as scroll."""
    import numpy as np
    chrome_sig = np.full((16, 64), 200, dtype="int16")
    seen_ds = np.ones((24, 24), dtype="int16") * 50
    cur_ds = np.ones((24, 24), dtype="int16") * 51   # would content-match seen_ds if Layer 2 ran

    adb = FakeAdb([b"base", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Zorva"
    drv._identity_sig = None              # never revealed Zorva's own sticky header this capture
    drv._identity_top_sig = chrome_sig    # the scroll-top chrome IS recognised -> pixel verdict 'top'
    drv._current_sigs = [seen_ds]
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: chrome_sig)
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: cur_ds)
    monkeypatch.setattr(hinge, "confirm_scroll_top",
                        lambda *_a, **_kw: SimpleNamespace(confirmed=True))
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "qelix" if psm == "6" else None)

    assert drv.wait_for_decision(timeout=5.0) is False


# --- observe: Layer 1b hole 1 (HIGH) -- a truncated OCR read is a false 'new' --------------
# Adversarial review of the Layer 1b fix above found a genuine false-PASS hole: _NAME_MATCH_
# RATIO was calibrated only against SUBSTITUTION-style misreads (a letter swapped for another),
# never against TRUNCATION (a partial crop at the band edge, tight kerning -- a common
# tesseract failure), which scores BELOW the ratio bar for longer names and would have been
# misread as a DIFFERENT person -- i.e. exactly the false 'new' this whole layer exists to
# prevent. _name_token_matches' prefix test (hinge.py, module-level, next to
# _NAME_MATCH_RATIO) is the fix. Every case below is MEASURED (see that function's own
# docstring for the exact numbers) and must stay 'same' -- the owner rule in force is that a
# false 'same' only ever costs a wait, never a wrong label.

def test_identity_top_name_ocr_truncated_read_of_a_longer_name_stays_same(monkeypatch):
    """False-PASS guard: a truncated READ this poll ('Zo' for stored 'Zorva', 'Kat'/'Ka' for
    stored 'Katherine') scores 0.57/0.50/0.36 against _NAME_MATCH_RATIO=0.6 -- below the bar,
    so without the prefix test this would have resolved 'top' straight to a false 'new' and
    recorded a PASS the human never made. A truncation is by definition a prefix of the name
    it came from, so the prefix test must catch every one of these."""
    for stored_name, read_name in (("Zorva", "Zo"), ("Katherine", "Kat"), ("Katherine", "Ka")):
        drv = _top_state_drv(monkeypatch, stored_name=stored_name)
        monkeypatch.setattr(
            drv, "_ocr_band",
            lambda frame, rect, psm="7", _r=read_name: _r if psm == "6" else None)

        state, _dist = drv._identity_of(b"frame")

        assert state == "same", (
            f"truncated read {read_name!r} of stored {stored_name!r} must stay 'same' "
            f"(false PASS guard)")


def test_identity_top_name_ocr_stored_name_truncated_by_a_bad_capture_stays_same(monkeypatch):
    """The REVERSE direction of the same hole: a BAD CAPTURE-TIME read is what got stored as
    self._identity_name ('Zo' for a profile actually named 'Zorva', 'Sam' for 'Samantha') --
    not this poll's OCR read, which is the FULL correct name. Every later full-name read must
    still resolve 'same' against that truncated stored value, or the one bad capture poisons
    the whole rest of the profile with spurious 'new' verdicts."""
    for stored_name, read_name in (("Zo", "Zorva"), ("Sam", "Samantha")):
        drv = _top_state_drv(monkeypatch, stored_name=stored_name)
        monkeypatch.setattr(
            drv, "_ocr_band",
            lambda frame, rect, psm="7", _r=read_name: _r if psm == "6" else None)

        state, _dist = drv._identity_of(b"frame")

        assert state == "same", (
            f"full read {read_name!r} against a truncated stored name {stored_name!r} must "
            f"stay 'same' (a bad capture-time read must not poison the whole profile)")


def test_identity_top_name_ocr_genuine_different_name_still_resolves_new(monkeypatch):
    """The top-name path stays able to distinguish a real top-of-next-card from generic
    scroll-top chrome; the two-frame PASS gate is what makes that candidate safe."""
    drv = _top_state_drv(monkeypatch, stored_name="Katherine")
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "Michelle" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "new"


def test_identity_top_name_ocr_short_candidate_token_never_becomes_new(monkeypatch):
    """Fix 1b: the 'new'-candidate selection requires >= 3 alphabetic characters. A 1-2
    character token is noise, never a usable name, and must never by itself be the basis for
    recording a PASS -- even when it is not one of Hinge's own chrome words and does not
    fuzzy/prefix-match the stored name closely enough to resolve 'same' either. This must stay
    the inconclusive 'top', not be promoted to a false 'new'."""
    drv = _top_state_drv(monkeypatch, stored_name="Katherine")
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7", **_kwargs: (
                            "Xy" if psm == "6" else None))

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"


@pytest.mark.parametrize(
    "stored_name, expected_state, expected_candidate",
    [("Aisha", "new", "S"), ("S", "same", None), ("Samantha", "same", None)],
)
def test_identity_top_name_ocr_accepts_only_structured_single_letter_signals_name(
        monkeypatch, stored_name, expected_state, expected_candidate):
    """The Aisha -> S incident: Hinge itself binds the initial to the profile-name slot.

    The exact Signals banner may recover the displayed one-letter name, while the same read for
    a captured S (or a conservatively prefix-matching Samantha) must veto an advance.
    """
    drv = _top_state_drv(monkeypatch, stored_name=stored_name)
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: (
            "S shows thoughtful signals" if psm == "6" else None))

    state, _dist = drv._identity_of(b"frame")

    assert state == expected_state
    assert drv._identity_name_candidate == expected_candidate


@pytest.mark.parametrize(
    "stored_name, expected_state, expected_candidate",
    [("Francesca", "new", "Ri"), ("Ri", "same", None), ("Rina", "same", None)],
)
def test_identity_top_name_ocr_accepts_only_structured_two_letter_signals_name(
        monkeypatch, stored_name, expected_state, expected_candidate):
    """The run-24179f2e77e0 exception is equally structural for a two-letter name.

    A different stored name may produce the candidate, while an exact or prefix-compatible
    captured name must retain the conservative same-profile veto.
    """
    drv = _top_state_drv(monkeypatch, stored_name=stored_name)
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: (
            "Ri shows thoughtful signals" if psm == "6" else None))

    state, _dist = drv._identity_of(b"frame")

    assert state == expected_state
    assert drv._identity_name_candidate == expected_candidate


@pytest.mark.parametrize(
    "ocr_text", ["S", "S Active today", "X shows signals",
                 "Ri", "Ri Active today", "Ri shows signals"])
def test_identity_top_name_ocr_unstructured_short_name_stays_inconclusive(
        monkeypatch, ocr_text):
    """A short name has no authority unless Hinge's complete fixed banner binds it."""
    drv = _top_state_drv(monkeypatch, stored_name="Aisha")
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: ocr_text if psm == "6" else None)

    assert drv._identity_of(b"frame")[0] == "top"
    assert drv._identity_name_candidate is None


# --- observe: PASS needs stable positive identity-new evidence on BOTH frames ----------------
def test_transient_pixel_new_followed_by_scroll_top_never_produces_a_pass(monkeypatch):
    """Exact regression for f192396916e8.

    A manual scroll can put the thin sticky-header band in an in-between rendering which is
    neither the captured header nor the filter-chip signature. The old `!= "same"` confirm
    gate accepted that one-poll pixel ``new`` plus a settled ``top`` as PASS. Neither frame
    positively proves a different profile twice, so this must remain an unlabeled wait.
    """
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    top_sig = np.full((16, 64), 200, dtype="int16")
    transition_sig = np.full((16, 64), 100, dtype="int16")
    adb = FakeAdb([b"base", b"transition", b"top"], advance_on_screencap=True)
    drv = _drv(adb, observe_name_ocr=False)
    drv._identity_sig = old_sig
    drv._identity_top_sig = top_sig
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(
        hinge, "_band",
        lambda frame, rect: {b"base": old_sig, b"transition": transition_sig,
                             b"top": top_sig}[frame])
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_changed", lambda a, b: False)

    assert drv.wait_for_decision(timeout=0.2) is None
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_name_derived_new_reproduced_on_confirm_frame_produces_a_pass(monkeypatch):
    """The genuine scroll-top advance path remains valid when the independently captured
    confirm frame also resolves a different card-header name to ``new``."""
    import numpy as np
    chrome_sig = np.full((16, 64), 200, dtype="int16")
    adb = FakeAdb([b"base", b"new", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Zorva"
    drv._identity_sig = None
    drv._identity_top_sig = chrome_sig
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: chrome_sig)
    monkeypatch.setattr(hinge, "confirm_scroll_top",
                        lambda *_a, **_kw: SimpleNamespace(confirmed=True))
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7", **_kwargs: (
                            "qelix\nprivate OCR transcript" if psm == "6" else None))

    assert drv.wait_for_decision(timeout=5.0) is False
    decision = next(fields for name, fields in drv._dbg.calls if name == "observe_decision")
    serialized = repr(decision)
    assert "identity_name_read" not in decision
    assert "confirm_identity_name_read" not in decision
    assert "private OCR transcript" not in serialized
    assert "qelix" not in serialized.casefold()
    assert decision["identity_name_candidate_sha256"] == hashlib.sha256(
        b"qelix").hexdigest()
    assert decision["identity_ocr_attempts"][0]["token_count"] == 0
    assert all(set(attempt) == {
        "recipe", "digest", "verdict", "candidate_sha256", "token_count",
    }
               for attempt in decision["identity_ocr_attempts"])


def test_observe_name_proof_refuses_hybrid_primary_and_fallback_reads(monkeypatch):
    """Matching text from different OCR geometries is not a repeated proof."""
    import numpy as np

    chrome_sig = np.full((16, 64), 200, dtype="int16")
    adb = FakeAdb([b"base", b"primary", b"fallback"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Zorva"
    drv._identity_top_sig = chrome_sig
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(hinge, "_band", lambda *_a: chrome_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(drv, "_changed", lambda _a, _b: False)
    monkeypatch.setattr(hinge, "confirm_scroll_top",
                        lambda *_a, **_kw: SimpleNamespace(confirmed=True))
    monkeypatch.setattr(drv, "_note_observe_waiting",
                        lambda reason, *_a, **_kw: reason == "not_settled")

    def ocr(frame, rect, *, psm="7", **_kwargs):
        if psm != "6":
            return None
        if frame == b"primary":
            return "Qelix" if tuple(rect) == drv.identity_top_name_band else None
        if frame == b"fallback":
            return ("pronouns Active now" if tuple(rect) == drv.identity_top_name_band
                    else "Qelix")
        return None

    monkeypatch.setattr(drv, "_ocr_band", ocr)

    assert drv.wait_for_decision(timeout=5.0) is None
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


def test_observe_name_proof_refuses_local_top_when_canonical_top_is_pinned(monkeypatch):
    """A capture-local top signature cannot authorize either frame of an Observe PASS."""
    import numpy as np

    chrome_sig = np.full((16, 64), 200, dtype="int16")
    adb = FakeAdb([b"base", b"new", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Zorva"
    drv._identity_top_sig = chrome_sig
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(hinge, "_band", lambda *_a: chrome_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(drv, "_changed", lambda _a, _b: False)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda _frame, _rect, *, psm="7", **_kw: "Qelix" if psm == "6" else None)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(confirmed=False, state="pinned", reason="pinned row"))
    monkeypatch.setattr(drv, "_note_observe_waiting",
                        lambda reason, *_a, **_kw: reason == "not_settled")

    assert drv.wait_for_decision(timeout=5.0) is None
    assert not [name for name, _fields in drv._dbg.calls if name == "observe_decision"]


# --- observe: Layer 1b hole 3 (informational) -- bound _ocr_band's tesseract cost with a cache
# MEASURED on the Pixel 7a 2026-08-10: ~103ms per _ocr_band call at psm 7 (the sticky
# identity_band), ~110ms at psm 6 (the card-header identity_top_name_band). Both bands can be
# OCR'd on the SAME poll (the identity_band OCR corroboration plus Layer 1b), against an
# _OBSERVE_POLL_S of 0.35s -- up to ~220ms of subprocess time inside a 350ms poll. These tests
# exercise _ocr_band's OWN memo cache directly (stubbing subprocess.run/shutil.which, not the
# driver-level _ocr_band monkeypatch every other test in this file uses, since that would
# bypass the very code path being tested).

def test_ocr_band_caches_a_repeated_identical_band_without_rerunning_tesseract(monkeypatch):
    """The common repeat this cache targets: wait_for_decision's own settle/confirm frame 0.5s
    later is usually the identical screen, and _is_current_profile_frame re-enters _identity_of
    (and so this method) on frames already seen earlier in the same poll. Identical
    (rect, psm, frame bytes) must not spawn a second tesseract subprocess."""
    from types import SimpleNamespace
    drv = _drv(FakeAdb([b"x"]))
    calls = []
    monkeypatch.setattr(hinge.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        hinge.subprocess, "run",
        lambda *a, **k: (calls.append(1), SimpleNamespace(stdout=b"Zorva", returncode=0))[1])
    frame = _png(value=10)
    rect = (0.0, 0.0, 1.0, 1.0)

    first = drv._ocr_band(frame, rect, psm="6")
    second = drv._ocr_band(frame, rect, psm="6")

    assert first == "Zorva"
    assert second == "Zorva"
    assert len(calls) == 1, f"expected exactly 1 tesseract invocation, got {len(calls)}"


def test_ocr_band_native_retry_has_its_own_cache_entry_and_skips_resize(monkeypatch):
    """The native card-header retry must not reuse the failed 3x OCR result."""
    from io import BytesIO
    from types import SimpleNamespace

    from PIL import Image

    drv = _drv(FakeAdb([b"x"]))
    sizes = []
    monkeypatch.setattr(hinge.shutil, "which", lambda name: "/usr/bin/tesseract")

    def run(*_args, **kwargs):
        sizes.append(Image.open(BytesIO(kwargs["input"])).size)
        return SimpleNamespace(stdout=b"Christina", returncode=0)

    monkeypatch.setattr(hinge.subprocess, "run", run)
    frame = _png(value=10)
    rect = (0.0, 0.0, 1.0, 1.0)

    assert drv._ocr_band(frame, rect, psm="6") == "Christina"
    assert drv._ocr_band(frame, rect, psm="6", upscale=1) == "Christina"
    assert sizes[0] == (sizes[1][0] * 3, sizes[1][1] * 3)
    assert len(sizes) == 2


def test_scroll_top_name_retries_native_after_garbled_upscale(monkeypatch):
    """Replay the report's OCR shape without weakening the repeated-name proof.

    Hinge 10.1.0's photo texture made the 3x psm-6 read look like several bogus lines while
    native resolution read the plainly visible next-profile name.  The native result may
    supply the candidate, but the caller still has to reproduce it on a settled frame.
    """
    import numpy as np

    drv = _drv(FakeAdb([b"x"]))
    top_sig = np.full((16, 64), 200, dtype="int16")
    drv._identity_name = "Kassie"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = top_sig
    calls = []
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect: top_sig)

    def ocr(_frame, _rect, *, psm="7", upscale=3, **_kwargs):
        calls.append((psm, upscale))
        if psm == "7":
            return None
        return "id\nJ pA\nx\n4h a" if upscale == 3 else "Christina"

    monkeypatch.setattr(drv, "_ocr_band", ocr)

    assert drv._identity_of(b"NEXT_PROFILE")[0] == "new"
    assert drv._identity_top_name_read == "Christina"
    assert drv._identity_top_name_verdict == "new"
    assert drv._identity_name_candidate == "Christina"
    assert ("6", 3) in calls and ("6", 1) in calls


def test_scroll_top_name_native_retry_keeps_noise_inconclusive(monkeypatch):
    """Two OCR recipes returning photo noise still cannot manufacture a new profile."""
    import numpy as np

    drv = _drv(FakeAdb([b"x"]))
    top_sig = np.full((16, 64), 200, dtype="int16")
    drv._identity_name = "Kassie"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")
    drv._identity_top_sig = top_sig
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect: top_sig)
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda _frame, _rect, *, psm="7", upscale=3, **_kwargs: (
            None if psm == "7" else "photo texture has many words"),
    )

    assert drv._identity_of(b"AMBIGUOUS_PROFILE")[0] == "top"
    assert drv._identity_top_name_verdict is None
    assert drv._identity_name_candidate is None


def test_scroll_top_name_compact_fallback_recovers_lara_after_broad_prompt_read(monkeypatch):
    """The Lara incident: both broad recipes read only metadata/prompt, while the contained
    compact fallback returns the actual first name.  Its provenance must remain distinct."""
    drv = _top_state_drv(monkeypatch, stored_name="Emily")
    calls = []

    def ocr(_frame, rect, *, psm="7", upscale=3, **_kwargs):
        calls.append((tuple(rect), psm, upscale))
        if psm == "7":
            return None
        if tuple(rect) == drv.identity_top_name_band:
            return "she her hers Active now\nAs seen on my Mom's fridge"
        return "Lara\nshe her hers Active now"

    monkeypatch.setattr(drv, "_ocr_band", ocr)

    assert drv._identity_of(b"LARA_TOP")[0] == "new"
    assert drv._identity_name_candidate == "Lara"
    assert drv._identity_name_candidate_source == "top_card_header_fallback"
    assert calls[-1] == (drv.identity_top_name_fallback_band, "6", 3)
    attempts = drv._identity_ocr_attempts
    assert [a["recipe"] for a in attempts] == [
        "identity_band_psm7_3x", "top_card_header_psm6_3x",
        "top_card_header_psm6_native", "top_card_header_fallback_psm6_3x",
    ]
    assert all("text" not in attempt and "text_sha256" in attempt for attempt in attempts)


@pytest.mark.parametrize("fallback_text, expected_state", [
    ("Emily", "same"),
    ("Lara Michelle", "top"),
    ("she her hers Active now", "top"),
])
def test_scroll_top_name_compact_fallback_keeps_stored_multiple_and_prompt_reads_closed(
        monkeypatch, fallback_text, expected_state):
    """The compact retry preserves the existing stored-name veto and exact-one parser."""
    drv = _top_state_drv(monkeypatch, stored_name="Emily")

    def ocr(_frame, rect, *, psm="7", upscale=3, **_kwargs):
        if psm == "7":
            return None
        if tuple(rect) == drv.identity_top_name_band:
            return "she her hers Active now"
        return fallback_text

    monkeypatch.setattr(drv, "_ocr_band", ocr)
    assert drv._identity_of(b"TOP")[0] == expected_state
    assert (drv._identity_name_candidate is None) == (expected_state != "new")


def test_ocr_band_preserves_lines_and_separates_punctuation(monkeypatch):
    """The exact raw OCR shape from the missed Jen pass must retain its name-line boundary.

    Slashes and the verification badge are separators, not characters to delete: deleting
    them used to weld ``she/her`` into another plausible name token, while flattening the
    newline discarded the strongest evidence that ``Jen`` was the actual profile name.
    """
    from types import SimpleNamespace
    drv = _drv(FakeAdb([b"x"]))
    monkeypatch.setattr(hinge.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        hinge.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            stdout=b"Jen &\nshe/her Active now\n", returncode=0),
    )

    assert drv._ocr_band(_png(value=10), (0.0, 0.0, 1.0, 1.0), psm="6") == (
        "Jen\nshe her Active now"
    )


def test_ocr_band_cache_miss_on_a_different_band_still_invokes_tesseract(monkeypatch):
    """Complement of the above: caching must never SUPPRESS a genuinely different read -- a
    different frame (a different band crop) is a cache miss and must still spawn tesseract, so
    a miss behaves exactly as it did before this cache existed."""
    from types import SimpleNamespace
    drv = _drv(FakeAdb([b"x"]))
    calls = []
    monkeypatch.setattr(hinge.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        hinge.subprocess, "run",
        lambda *a, **k: (calls.append(1), SimpleNamespace(stdout=b"Zorva", returncode=0))[1])
    rect = (0.0, 0.0, 1.0, 1.0)

    drv._ocr_band(_png(value=10), rect, psm="6")
    drv._ocr_band(_png(value=20), rect, psm="6")   # different frame bytes -> different band

    assert len(calls) == 2


def test_ocr_band_cache_resets_per_profile(monkeypatch):
    """The cache is keyed on frame bytes, not profile identity -- see _ocr_band_cache's own
    __init__ comment for why a coincidental byte-identical crop from the NEXT profile could
    otherwise return a stale answer. _capture_current must reset it exactly where it resets the
    other per-profile identity state (_identity_top_sig / _identity_sig / _identity_name)."""
    from types import SimpleNamespace
    drv = _drv(FakeAdb([b"x"]))
    calls = []
    monkeypatch.setattr(hinge.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        hinge.subprocess, "run",
        lambda *a, **k: (calls.append(1), SimpleNamespace(stdout=b"Zorva", returncode=0))[1])
    frame = _png(value=10)
    rect = (0.0, 0.0, 1.0, 1.0)

    drv._ocr_band(frame, rect, psm="6")
    assert len(calls) == 1

    drv._capture_current()          # the per-profile reset point

    drv._ocr_band(frame, rect, psm="6")
    assert len(calls) == 2, "cache must not survive across _capture_current (a new profile)"


# --- observe: rate-limited "still watching" operator notice (2026-08-10 incident) ----------
# The incident this exists for was TOTALLY silent: the owner pressed X, the misclassification
# above kept the wait going, and nothing appeared in the console, the hub, or actions.jsonl
# for the whole time before Stop was pressed by hand. This does not fix the misclassification
# (the tests above do) -- it exists so the NEXT time anything stalls, for this reason or a new
# one nobody has hit yet, there is something to look at instead of silence.

def test_note_observe_waiting_prints_and_records_but_is_rate_limited(monkeypatch):
    """Direct test of the rate limiter: a burst of calls (as wait_for_decision's poll loop
    makes, every _OBSERVE_POLL_S) must produce only ONE print + ONE observe_waiting debug
    record until _OBSERVE_WAIT_NOTICE_S has genuinely passed, at which point it fires again
    with whatever the CURRENT reason is (not a stale one from the first firing)."""
    clock = [1_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    drv = _drv(FakeAdb([b"x"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_notice = 0.0   # long enough ago that the very first call is already due

    drv._note_observe_waiting("no_change")
    clock[0] += 1.0                  # 1s later -- well under the 15s interval
    drv._note_observe_waiting("no_change")
    clock[0] += 1.0
    drv._note_observe_waiting("no_change")

    records = [f for name, f in drv._dbg.calls if name == "observe_waiting"]
    assert len(records) == 1
    assert records[0]["reason"] == "no_change"

    clock[0] += hinge.AndroidDriver._OBSERVE_WAIT_NOTICE_S   # now enough time has passed
    drv._note_observe_waiting("not_settled")

    records = [f for name, f in drv._dbg.calls if name == "observe_waiting"]
    assert len(records) == 2
    assert records[1]["reason"] == "not_settled"    # repeats with the CURRENT reason


def test_note_observe_waiting_reuses_identical_before_evidence_without_recapturing(
        monkeypatch, tmp_path):
    """The production heartbeat records its already-held frame; it never takes another shot.

    A human can deliberate on an unchanged card for minutes.  The heartbeat must retain each
    timestamp/reason row, but byte-identical evidence must resolve to one DebugLog PNG even
    though the real call path supplies it as ``before=`` (not the generic logger test's
    ``after=``).  A fresh screencap here would both add device latency and break the claim that
    the saved frame is the one the wait verdict actually used.
    """
    import json

    class NoFreshCaptureAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"unused"])
            self.capture_calls = 0

        def screencap(self):
            self.capture_calls += 1
            raise AssertionError("observe_waiting must log the supplied frame, not recapture")

    from operation_love.drivers.debuglog import HingeDebugLog

    clock = [1_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    adb = NoFreshCaptureAdb()
    drv = _drv(adb)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="wait-before-dedupe")
    drv._observe_last_notice = 0.0
    frame = b"same-frame-from-the-poll-that-produced-no-change"

    for _ in range(3):
        assert drv._note_observe_waiting("no_change", frame) is False
        clock[0] += hinge.AndroidDriver._OBSERVE_WAIT_NOTICE_S

    run_dir = tmp_path / "wait-before-dedupe"
    rows = [json.loads(line) for line in (run_dir / "actions.jsonl").read_text().splitlines()]
    assert [row["action"] for row in rows] == ["observe_waiting"] * 3
    assert [row["reason"] for row in rows] == ["no_change"] * 3
    assert len({row["before"] for row in rows}) == 1
    assert len(list(run_dir.glob("*.png"))) == 1
    assert adb.capture_calls == 0


def test_note_observe_waiting_prints_a_plain_actionable_status_line(monkeypatch, capsys):
    """The operator-facing half of the fix: plain wording naming the reason, and something
    actionable (press Stop) -- not just a debug-log line only a developer would ever read."""
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 1_000.0)
    drv = _drv(FakeAdb([b"x"]))
    drv._observe_last_notice = 0.0

    drv._note_observe_waiting("not_deck_ready")

    out = capsys.readouterr().out
    assert "not_deck_ready" in out
    assert "Stop" in out


def test_note_observe_waiting_suppresses_a_heartbeat_after_stop(monkeypatch, capsys):
    """A screencap can finish just after Hub Stop lands.  That stale frame must not append a
    fresh "still watching" record, or a completed run misleadingly reads as a live wait."""
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 1_000.0)
    stopped = True
    drv = _drv(FakeAdb([b"x"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_notice = 0.0

    assert drv._note_observe_waiting(
        "no_change", b"stale-frame", should_stop=lambda: stopped) is True
    assert drv._dbg.calls == []
    assert capsys.readouterr().out == ""


def test_like_candidate_wait_is_truthful_and_uses_the_normal_notice_cadence(monkeypatch, capsys):
    """No composer has been seen in this state, so it must not inherit either the wording or
    slower human-compose/send cadence of `like_sheet` / `like_sending`."""
    clock = [1_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    drv = _drv(FakeAdb([b"x"]))
    drv._dbg = _FakeDbg()
    drv._observe_last_notice = 0.0
    drv._observe_last_reason = None

    drv._note_observe_waiting("like_candidate")
    clock[0] += hinge.AndroidDriver._OBSERVE_WAIT_NOTICE_S - 1.0
    drv._note_observe_waiting("like_candidate")

    records = [fields for name, fields in drv._dbg.calls if name == "observe_waiting"]
    assert [record["reason"] for record in records] == ["like_candidate"]
    out = capsys.readouterr().out
    assert "possible like" in out
    assert "no Send Like sheet has been observed" in out
    assert "sheet closed" not in out

    clock[0] += 1.0
    drv._note_observe_waiting("like_candidate")
    records = [fields for name, fields in drv._dbg.calls if name == "observe_waiting"]
    assert [record["reason"] for record in records] == ["like_candidate", "like_candidate"]


def test_wait_for_decision_resets_the_waiting_rate_limiter_at_the_start_of_every_call():
    """self._observe_last_notice is reset unconditionally at the top of every wait_for_decision
    call (one call == one profile's wait) -- this is what keeps the rate limiter from ever
    going permanently quiet across profiles: a notice suppressed near the end of one profile's
    wait must not suppress the FIRST notice of the very next one."""
    drv = _drv(FakeAdb([b"a"]))
    drv._observe_last_notice = -10_000.0     # simulate ancient leftover state from a prior call

    assert drv.wait_for_decision(timeout=0.05) is None

    assert drv._observe_last_notice > 0      # reset to a real, current monotonic() value


# --- observe: gesture corroboration (layer 3, read-only touch-stream watcher) ----------
class _FakeWatcher:
    """Minimal stand-in for touchwatch.TouchWatcher: wait_for_decision's gesture
    corroboration (_observe_gesture_verdict) only ever reads `alive`, `event_count`, and
    `gestures_since()` off the real watcher -- never its Popen/thread internals -- so a
    fake exposing just those three is a faithful substitute."""
    def __init__(self, gestures=(), *, alive=True, event_count=1):
        self._gestures = list(gestures)
        self.alive = alive
        self.event_count = event_count

    def gestures_since(self, _t):
        return list(self._gestures)


def _tap_gesture(x, y):
    return Gesture(t_down=0.0, t_up=0.0, down=(x, y), up=(x, y), travel_px=0.0)


def _drag_gesture(x0, y0, x1, y1):
    return Gesture(t_down=0.0, t_up=0.0, down=(x0, y0), up=(x1, y1),
                   travel_px=math.hypot(x1 - x0, y1 - y0))


def _proven_advance_drv(monkeypatch, watcher, frames=(b"base", b"new")):
    """A driver already primed for the SAME identity-and-deck-ready-proven card advance
    every gesture-corroboration test below needs -- only the watcher (and, for the health-
    exception test's repeat-call check, the frame pair) differs between callers. Mirrors
    test_identity_new_profile_plus_deck_ready_and_settle_confirms_a_pass's setup so these
    tests exercise LAYER 3 in isolation, not layers 1/2 again. `_band` keys off the
    SUBSTRING 'new' (not an exact frame match) so a driver reused across a second advance
    with a fresh frame pair keeps classifying correctly."""
    import numpy as np
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    adb = FakeAdb(list(frames), advance_on_screencap=True)
    # observe_touch_watch ships False (Android 17 withholds the touch stream -- see
    # HINGE_SPEC's comment). These tests exercise layer 3 itself, so they opt in.
    drv = _drv(adb, observe_touch_watch=True)
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    drv._touch_watcher = watcher
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: new_sig if b"new" in frame else old_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    return drv, adb


def test_gesture_verdict_drag_only_resyncs_without_recording_a_decision(monkeypatch):
    """A card advance whose ONLY gesture since the profile became ready was a drag -- no
    tap at all -- must NOT be corroborated as a pass: a tap-gesture app's card cannot
    advance from a drag (see _observe_gesture_verdict's docstring). wait_for_decision
    returns None (worker.py already treats that as "recapture, record nothing"), and the
    debug log gets an observe_resync record, never an observe_decision one."""
    watcher = _FakeWatcher([_drag_gesture(500, 1000, 500, 1800)], event_count=4)
    drv, _adb = _proven_advance_drv(monkeypatch, watcher)
    drv._dbg = _FakeDbg()

    assert drv.wait_for_decision(timeout=5.0) is None

    names = [name for name, _fields in drv._dbg.calls]
    assert names == ["observe_resync"]


def test_gesture_verdict_tap_in_ignore_zone_resyncs(monkeypatch):
    """A card advance whose corroborating tap landed in an observe_ignore_zones rect (here:
    the rewind-arrow / '...' overflow zone, top-right) is a known non-decision -- resync,
    not a pass, even though the card genuinely changed and the deck is genuinely ready."""
    watcher = _FakeWatcher([_tap_gesture(900, 150)], event_count=3)   # inside (0.75,0.03,1.00,0.115)
    drv, _adb = _proven_advance_drv(monkeypatch, watcher)

    assert drv.wait_for_decision(timeout=5.0) is None


def test_gesture_verdict_tap_on_pass_x_confirms_the_pass(monkeypatch):
    """A card advance whose corroborating tap landed ON the located pass-X is exactly the
    affirmative evidence layer 3 exists to supply -- wait_for_decision returns False."""
    watcher = _FakeWatcher([_tap_gesture(125, 2035)], event_count=2)
    drv, _adb = _proven_advance_drv(monkeypatch, watcher)
    monkeypatch.setattr(drv, "_observe_locate_pass_x", lambda frame: (125, 2035))

    assert drv.wait_for_decision(timeout=5.0) is False


def test_touch_watcher_health_exception_requires_name_proof_and_warns_once(monkeypatch, capsys):
    """event_count == 0 for the whole run is proof the STREAM itself isn't delivering
    anything on this device (wrong node selected, a permissions change mid-run, ...) -- not
    "the human genuinely never touched the screen". It therefore cannot veto an independently
    name-proven advance, but pixel identity alone cannot manufacture one either. These fixtures
    have no name proof, so both resync without a label, and the health warning appears exactly
    once across both waits on the same driver."""
    watcher = _FakeWatcher([], alive=True, event_count=0)
    drv, _adb = _proven_advance_drv(monkeypatch, watcher, frames=(b"base1", b"new1"))

    assert drv.wait_for_decision(timeout=5.0) is None

    drv._adb = FakeAdb([b"base2", b"new2"], advance_on_screencap=True)
    drv._touch = drv._adb
    assert drv.wait_for_decision(timeout=5.0) is None

    out = capsys.readouterr().out
    assert out.count("touch watcher has seen no events this run") == 1


def test_observe_open_session_raises_when_its_touch_watcher_cant_start(monkeypatch):
    """observe_touch_watch=True (opt-in; it ships False because the target device's
    platform withholds the stream) must fail LOUDLY -- the same "explicit
    operator decision, never a silent downgrade" contract as touch_backend -- when the
    device's touch event stream can't be attached, rather than silently narrowing observe
    mode's PASS proof back to repeated name + identity + deck readiness. The raised DriverClosed
    must name the config key an operator can set to accept that narrower proof on purpose.

    This was a dead test until 2026-09-02 (FINDING 2): the name lacked the `test_` prefix
    (a leftover from the ea6756e8 rename to supervised training) so pytest never collected it.
    Collecting it exposed a second bug in the fixture itself, not just its name: it called
    `drv.set_auto_session_policy(None)`, which unconditionally sets `_auto_session = True` (see
    that method's own docstring) -- but the watcher block this test exists to exercise is gated
    `if self.observe_touch_watch and not self._auto_session`, OBSERVE SESSIONS ONLY, so that
    call silently skipped the whole branch and TouchWatcher.start() was never even reached.
    Confirmed by running the pre-rename body verbatim: it did not raise. Dropped here so
    `_auto_session` stays at its observe-mode default of False, per fixtures-can-miss-the-
    branch-they-name -- collecting a test is not the same as it reaching the code it names.

    It now also proves FINDING 1's leak fix: by the time TouchWatcher.start() fails,
    open_session() has already called `_make_touch()` a few lines above and registered a real
    UHID virtual touchscreen on the device (faked here as `_FakeTouch`, so this stays a device-
    free test). Before the fix, the guard that runs `self.close()` on any open_session()
    failure only wrapped the debug-log setup AFTER this block, so the DriverClosed raised here
    escaped uncaught and the registered touch transport (and the ADB link) leaked -- the same
    failure class that already leaked 7 virtual touchscreens once from a different trigger."""
    class _OpenAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"x"])

        def devices(self):
            return ["pixel"]

    class _DeadWatcher:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            raise hinge.TouchWatchUnavailable("no ABS_MT_POSITION_X/Y device")

    class _FakeTouch:
        """Stands in for UhidTouch so this test can prove the registered transport gets
        released on the open_session() failure path below, without touching a real
        /dev/hidg node."""
        instances = []

        def __init__(self, adb):
            self.adb = adb
            self.closed = False
            _FakeTouch.instances.append(self)

        def open(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(hinge, "Adb", lambda *a, **k: _OpenAdb())
    monkeypatch.setattr(hinge, "TouchWatcher", _DeadWatcher)
    monkeypatch.setattr(hinge, "UhidTouch", _FakeTouch)   # touch_backend defaults to "auto"
    monkeypatch.setattr("operation_love.platforms.unavailable_reason", lambda *_a, **_k: None)
    cfg = type("C", (), {"mode": "training", "apps": {
        "hinge": {"serial": "pixel", "observe_touch_watch": True}}})
    drv = HingeDriver(cfg)
    # No set_auto_session_policy() call: this simulates an OBSERVE session, and
    # _auto_session must stay at its constructed-False default for the gate above to
    # actually let TouchWatcher.start() run (see the docstring's fixture-bug note).

    with pytest.raises(DriverClosed) as exc:
        drv.open_session()
    assert "observe_touch_watch" in str(exc.value)

    # FINDING 1: the virtual touchscreen _make_touch() registered above must not survive a
    # failed open_session() -- close() must have run and released both the touch transport
    # and the ADB link, not just left them dangling for the hub to leak until it exits.
    assert len(_FakeTouch.instances) == 1
    assert _FakeTouch.instances[0].closed is True
    assert drv._touch is None
    assert drv._adb is None


def test_open_session_always_reads_the_targeting_binding_live(monkeypatch):
    """`_refresh_targeting_calibration_binding` memoizes its `dumpsys package` + `screen_size`
    read for one item-index lifetime (STAGE B2 FINDING 2). That memo is only safe under the
    property that the FIRST read of a session is live, and `close()` does not clear the flag --
    so a driver object that opened a second session would otherwise carry the first session's
    build/frame binding into it and could license a targeted like against a Hinge build that
    is no longer installed. Nothing in production reopens a driver today; this pins the
    invariant so the memo stays safe if something ever does."""
    launched = []

    class _OpenAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"x"])
            self.dumpsys_package_calls = 0

        def devices(self):
            return ["pixel"]

        def shell(self, command="", **_):
            if command.startswith("dumpsys package"):
                self.dumpsys_package_calls += 1
                return "    versionName=10.1.0\n"
            if command.startswith("monkey "):
                launched.append(command)
            return ""

    adbs = []

    def _make_adb(*_a, **_k):
        adbs.append(_OpenAdb())
        return adbs[-1]

    class _FakeTouch:
        def __init__(self, adb):
            self.adb = adb

        def open(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(hinge, "Adb", _make_adb)
    monkeypatch.setattr(hinge, "UhidTouch", _FakeTouch)
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("operation_love.platforms.unavailable_reason", lambda *_a, **_k: None)
    calibration = {
        "schema_version": 3, "hinge_version_name": "10.1.0", "frame_size_px": [1080, 2400],
        "composer_layout_id": "hinge_inline_v1", "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 2.0, "inline_item_max_dist": 10.0, "device": "pixel",
        "calibrated_at": "2026-08-12",
        "identity_band": list(hinge.HINGE_SPEC.identity_band),
        "content_band": list(hinge.HINGE_SPEC.content_band),
    }
    cfg = type("C", (), {"mode": "training", "apps": {"hinge": {
        "serial": "pixel", "targeting_calibration": calibration}}})
    drv = HingeDriver(cfg)
    assert drv.targeting_calibration is not None, "fixture must reach the binding read at all"

    drv.open_session()
    assert adbs[0].dumpsys_package_calls == 1
    # Simulate the memo being latched by the session's work, then close and reopen.
    drv._targeting_binding_fetched = True
    drv.close()
    drv.open_session()
    assert adbs[-1].dumpsys_package_calls == 1, (
        "the second session reused the first session's memoized build/frame binding instead "
        "of reading the device it is actually driving")
    assert len(launched) == 2


def test_observe_mode_never_taps_swipes_or_types_across_the_new_decision_paths(monkeypatch):
    """Passivity contract, across layer 1 (identity), layer 2 (content /
    _vertical_shift_match -- exercised transitively by the identity tests above, which
    reuse the same wait_for_decision path), and layer 3 (gesture corroboration) alike:
    NOTHING on any of these new observe-mode paths issues a tap, a swipe, or types text. A
    regression here would mean the redesign accidentally turned a passive OBSERVER into an
    ACTOR on the burner's own deck."""
    import numpy as np

    # (a) identity 'same' -> keeps waiting.
    id_sig = np.full((16, 64), 30, dtype="int16")
    adb = FakeAdb([b"base", b"scrolled"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_sig = id_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: id_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (50.0, 50.0))
    assert drv.wait_for_decision(timeout=0.05) is None
    assert adb.taps == [] and adb.swipes == 0 and adb.texts == []

    # (b) a real advance corroborated by a tap on the pass-X -- still only READS the touch
    # stream, never writes to the device.
    watcher = _FakeWatcher([_tap_gesture(125, 2035)], event_count=2)
    drv, adb = _proven_advance_drv(monkeypatch, watcher, frames=(b"base7", b"new7"))
    monkeypatch.setattr(drv, "_observe_locate_pass_x", lambda frame: (125, 2035))
    assert drv.wait_for_decision(timeout=5.0) is False
    assert adb.taps == [] and adb.swipes == 0 and adb.texts == []

    # (c) a real advance whose only gesture was a drag -> resync -- still nothing tapped,
    # swiped, or typed on the device itself.
    watcher = _FakeWatcher([_drag_gesture(500, 1000, 500, 1800)], event_count=4)
    drv, adb = _proven_advance_drv(monkeypatch, watcher, frames=(b"base5", b"new5"))
    assert drv.wait_for_decision(timeout=5.0) is None
    assert adb.taps == [] and adb.swipes == 0 and adb.texts == []


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


@pytest.mark.parametrize("top_result", (
    SimpleNamespace(confirmed=False, state="confirmed_not_top", reason="sticky header visible"),
    hinge.ScrollTopError("identity band unreadable"),
), ids=("refuted", "unreadable"))
def test_hinge_rewind_does_not_accept_no_motion_without_an_affirmative_top_verdict(
        monkeypatch, top_result):
    """A swallowed reverse swipe mid-card is not proof that the card reached its top."""
    adb = FakeAdb([b"mid-card"])
    drv = _drv(adb, scroll_captures=2)
    drv._capture_scroll_ledger = [(0.55, 0.5)] * 2
    drv._capture_scrolls = 2
    monkeypatch.setattr(drv, "_changed", lambda *_a: False)

    def top_verdict(*_a, **_k):
        if isinstance(top_result, Exception):
            raise top_result
        return top_result

    monkeypatch.setattr(hinge, "confirm_scroll_top", top_verdict)

    assert drv._scroll_to_top() is False
    assert adb.swipes == 2                    # bounded retries still happen
    assert drv._capture_scrolls == 2          # failure cannot erase recovery evidence
    assert drv._capture_scroll_ledger == [(0.55, 0.5)] * 2


def test_failed_hinge_rewind_records_its_final_filter_chip_verdict(monkeypatch):
    """A bounded failure must say what the top detector saw, not imply ongoing motion."""
    adb = FakeAdb([b"settled-top-with-new-chrome"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._capture_scrolls = 1
    verdict = SimpleNamespace(
        confirmed=False,
        state="cannot_tell",
        distance=8.859375,
        alignment_offset_px=1,
    )
    monkeypatch.setattr(hinge, "confirm_scroll_top", lambda *_a, **_k: verdict)

    assert drv._scroll_to_top() is False

    recovery = next(fields for name, fields in drv._dbg.calls
                    if name == "capture_entry_recovery_spent")
    assert recovery["scroll_top_state"] == "cannot_tell"
    assert recovery["scroll_top_distance"] == pytest.approx(8.859375)
    assert recovery["scroll_top_alignment_offset_px"] == 1
    assert "scroll_top_error" not in recovery


def test_failed_hinge_rewind_retains_bounded_attempt_facts_and_endpoint_frames(monkeypatch):
    """Per-attempt facts stay scalar; only the bounded rewind's endpoints retain frames."""
    adb = FakeAdb([b"same-scrolled-card"])
    drv = _drv(adb, scroll_captures=2, read_scroll_frac=0.55, rewind_scroll_frac=0.78)
    drv._dbg = _FakeDbg()
    drv._capture_scrolls = 2
    verdict = SimpleNamespace(
        confirmed=False, state="confirmed_not_top", distance=9.640625,
        alignment_offset_px=12,
    )
    monkeypatch.setattr(hinge, "confirm_scroll_top", lambda *_a, **_k: verdict)
    monkeypatch.setattr(drv, "_changed", lambda *_a: False)

    assert drv._scroll_to_top() is False
    recovery = next(fields for name, fields in drv._dbg.calls
                    if name == "capture_entry_recovery_spent")
    assert recovery["scroll_top_state"] == "confirmed_not_top"
    assert recovery["scroll_top_initial"]["scroll_top_state"] == "confirmed_not_top"
    attempts = recovery["scroll_top_attempts"]
    assert len(attempts) == 2
    assert [(row["start"], row["end"], row["duration_ms"], row["frame_changed"])
            for row in attempts] == [
                ([540, 540], [540, 1860], 450, False),
                ([540, 540], [540, 1860], 450, False),
            ]
    assert all(row["detector"]["scroll_top_distance"] == pytest.approx(9.640625)
               for row in attempts)
    assert not any("before" in row or "after" in row for row in attempts)
    assert recovery["keep_before"] is True and recovery["keep_after"] is True
    assert ("capture_entry_recovery_spent", b"same-scrolled-card") in drv._dbg.befores
    assert ("capture_entry_recovery_spent", b"same-scrolled-card") in drv._dbg.afters


def test_saved_rewind_failure_frame_is_refuted_while_prior_same_card_top_confirms():
    """Replay the real session-start failure without embedding a private profile frame in git.

    The saved failure is the decisive control for the geometry change: it still has the profile
    header in ``identity_band`` after twelve old long-flick attempts, whereas an earlier capture
    of the same card has Hinge's filter chips at that exact crop. Both files are local debug
    evidence and this test intentionally skips in CI when they are not present.
    """
    root = Path(__file__).resolve().parent.parent
    failure = root / "data/hinge_debug/de89edfe05e1/00001_capture_entry_refused_before.png"
    confirmed_top = root / "data/hinge_debug/4efcff0d78ca/00315_capture_before.png"
    if not failure.is_file() or not confirmed_top.is_file():
        pytest.skip("private local Hinge rewind evidence is unavailable")

    failed = scroll_top.confirm_scroll_top(
        failure.read_bytes(), identity_band=hinge.HINGE_SPEC.identity_band)
    top = scroll_top.confirm_scroll_top(
        confirmed_top.read_bytes(), identity_band=hinge.HINGE_SPEC.identity_band)

    assert failed.state == scroll_top.SCROLL_TOP_REFUTED
    assert failed.distance == pytest.approx(9.640625)
    assert failed.alignment_offset_px == 12
    assert top.confirmed is True


def test_session_top_failure_prints_filter_chip_detector_fact_not_motion(monkeypatch, capsys):
    """A settled but uncalibrated top must not be reported as a screen that kept moving."""
    drv = _drv(FakeAdb([b"settled-top-with-new-chrome"]))
    drv._dbg = _FakeDbg()
    verdict = SimpleNamespace(
        confirmed=False,
        state="cannot_tell",
        distance=8.859375,
        alignment_offset_px=1,
    )
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(hinge, "confirm_scroll_top", lambda *_a, **_k: verdict)

    assert drv._ensure_session_top() is False

    out = capsys.readouterr().out
    assert "final filter-chip verdict was `cannot_tell`" in out
    assert "distance 8.859" in out and "alignment offset +1px" in out
    assert "never stopped moving" not in out
    session = next(fields for name, fields in drv._dbg.calls if name == "session_top_unconfirmed")
    assert session["scroll_top_state"] == "cannot_tell"
    assert session["scroll_top_distance"] == pytest.approx(8.859375)
    assert session["scroll_top_alignment_offset_px"] == 1


def test_failed_hinge_rewind_records_filter_chip_decode_error(monkeypatch):
    """An undecodable final band is retained as the alternative detector diagnostic."""
    drv = _drv(FakeAdb([b"unreadable-band"]))
    drv._dbg = _FakeDbg()
    drv._capture_scrolls = 1

    def unreadable(*_args, **_kwargs):
        raise hinge.ScrollTopError("identity band unreadable")

    monkeypatch.setattr(hinge, "confirm_scroll_top", unreadable)
    assert drv._scroll_to_top() is False

    recovery = next(fields for name, fields in drv._dbg.calls
                    if name == "capture_entry_recovery_spent")
    assert recovery["scroll_top_error"] == "identity band unreadable"
    assert "scroll_top_state" not in recovery


def test_generic_rewind_keeps_legacy_no_motion_top_confirmation(monkeypatch):
    """No-band drivers retain the old fallback because they have no affirmative top signal."""
    adb = FakeAdb([b"generic-card"])
    drv = _drv(adb)
    drv.identity_band = None
    drv._capture_scroll_ledger = [(0.55, 0.5)] * 2
    drv._capture_scrolls = 2
    monkeypatch.setattr(drv, "_changed", lambda *_a: False)
    monkeypatch.setattr(hinge, "confirm_scroll_top",
                        lambda *_a, **_k: pytest.fail("generic rewind must not request a band"))

    assert drv._scroll_to_top() is True
    assert adb.swipes == 1
    assert drv._capture_scrolls == 0 and drv._capture_scroll_ledger == []


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


def test_scroll_to_top_caps_configured_rewind_to_safe_central_geometry(monkeypatch):
    """A config value cannot put a Hinge recovery release in the fixed bottom navigation."""
    adb = FakeAdb([b"before", b"after"])
    drv = _drv(adb, read_scroll_frac=0.55, rewind_scroll_frac=0.78)
    drv._capture_scroll_ledger = [(0.55, 0.5)] * 6
    drv._capture_scrolls = len(drv._capture_scroll_ledger)
    gestures = []
    monkeypatch.setattr(drv, "_swipe", lambda *args, **kwargs: gestures.append((args, kwargs)))
    monkeypatch.setattr(
        hinge, "confirm_scroll_top", lambda *_a, **_k: SimpleNamespace(confirmed=True))
    monkeypatch.setattr(
        drv, "_changed", lambda *_a, **_k: pytest.fail("confirmed top must stop immediately"))

    assert drv._scroll_to_top() is True
    # 0.78 used to be 263 -> 2136 @160ms, ending in the fixed Hinge nav.  The cap keeps the
    # reverse stroke on the ordinary read corridor and uses _swipe's standard 450ms timing.
    assert gestures == [((540, 540, 540, 1860), {})]
    assert drv._capture_scroll_ledger == [] and drv._capture_scrolls == 0


def test_auto_policy_hinge_undo_stays_in_the_central_read_corridor(monkeypatch):
    class Policy:
        pass

    # Six forward strokes total three screen-heights. Hinge deliberately does NOT replay them
    # with a long randomized flick: every recovery stroke uses the proven central 0.55h corridor.
    adb = PositionTrackingAdb([b"x"])
    drv = _drv(adb, scroll_captures=8, read_scroll_frac=0.55)
    # This is a transport-distance test rather than a Hinge identity-band test.  Generic
    # no-band callers retain no-motion as their only available top proof.
    drv.identity_band = None
    drv._auto_policy = Policy()
    drv._capture_scroll_ledger = [
        (0.46, 0.41), (0.50, 0.57), (0.54, 0.45),
        (0.48, 0.60), (0.52, 0.39), (0.50, 0.53),
    ]
    drv._capture_scrolls = len(drv._capture_scroll_ledger)
    adb.position = sum(frac for frac, _lane in drv._capture_scroll_ledger)
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, _a, _b: adb.position > 1e-9)

    drv._scroll_to_top()

    assert adb.swipes == 6
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


def test_auto_policy_hinge_undo_cannot_vary_out_of_the_central_safe_geometry(monkeypatch):
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

    assert set(gestures) == {(540, 540, 540, 1860)}


def test_current_profile_returns_to_top_after_observe_capture(monkeypatch):
    # bug 2 fix: current_profile() is the ONLY observe capture path, and worker.py prints
    # "READY - swipe this profile" immediately after it returns -- so leaving the phone
    # scrolled to the bottom (all _capture_current did before this fix) has the operator
    # swipe a view that isn't the top of the profile they just read.
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)  # force full ceiling
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)
    drv._session_top_done = True         # isolate the per-capture unwind from the once-per-session one

    drv.current_profile()

    assert adb.scrolls == 2              # capture read-scrolled twice (3 distinct frames)
    assert adb.swipes == 2               # _scroll_to_top swiped all the way back up before returning


def test_manual_capture_is_not_published_when_trailing_rewind_fails(monkeypatch):
    """READY must never describe a card that the driver could not restore to its top."""
    drv = _drv(FakeAdb([b"card"]))
    drv._session_top_done = True
    marker = object()
    monkeypatch.setattr(drv, "_capture_current", lambda *_a, **_k: marker)
    monkeypatch.setattr(drv, "_scroll_to_top", lambda *_a, **_k: False)

    assert drv.current_profile() is None
    assert "could not be returned to a confirmed scroll top" in drv._current_items_unavailable
    # This is Observe (the default _auto_session is False): the terminal refusal must still be
    # visible to Worker rather than being retried as an unexplained capture None.
    assert "could not be returned to a confirmed scroll top" in (drv.blocked_reason() or "")


def test_manual_capture_keeps_completed_profile_at_a_stop_boundary(monkeypatch):
    """Worker observes Stop before READY; direct callers retain the completed stop boundary."""
    drv = _drv(FakeAdb([b"card"]))
    drv._session_top_done = True
    marker = object()
    monkeypatch.setattr(drv, "_capture_current", lambda *_a, **_k: marker)
    monkeypatch.setattr(drv, "_scroll_to_top", lambda *_a, **_k: False)

    assert drv.current_profile(should_stop=lambda: True) is marker


def test_actionable_manual_capture_rewinds_and_reproves_before_any_profile_read(monkeypatch):
    """A user-scrolled READY card must not first be read as an unnumbered profile."""
    drv = _drv(FakeAdb([b"card"]))
    drv._session_top_done = True
    events = []
    profile = object()
    top_answers = iter(["confirmed_not_top", ""])
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: events.append("top") or next(top_answers))
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: events.append("rewind") or True)
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: events.append("capture") or profile)

    assert drv.current_profile() is profile
    assert events == ["top", "rewind", "top", "capture", "rewind"]


def test_actionable_capture_refuses_to_publish_when_rewind_cannot_reprove_top(monkeypatch):
    """No ranker/identity Profile escapes when the post-rewind entry proof is still absent."""
    drv = _drv(FakeAdb([b"card"]))
    drv._session_top_done = True
    captured = []
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: "the card is not confirmed to be at its scroll top")
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_scroll_to_top", lambda *_a, **_k: False)
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: captured.append(True) or object())

    assert drv.current_profile() is None
    assert captured == []
    assert "capture entry could not be proven" in drv._current_items_unavailable


def test_unchanged_refused_observe_entry_spends_only_one_bounded_rewind(monkeypatch):
    """Repeated worker recaptures must not turn one stuck card into endless autonomous input."""
    drv = _drv(FakeAdb([b"same-scrolled-card"]))
    drv._session_top_done = True
    rewinds = []
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: "confirmed_not_top: sticky profile header")
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: rewinds.append("rewind") or False)

    assert drv.current_profile() is None
    assert drv.current_profile() is None
    assert rewinds == ["rewind"]


def test_animated_refused_observe_entry_still_spends_only_one_bounded_rewind(monkeypatch):
    """Frame churn is diagnostic noise, never permission for a second autonomous recovery."""
    class AnimatedAdb(FakeAdb):
        def __init__(self):
            super().__init__([b"seed"])
            self.n = 0

        def screencap(self):
            self.n += 1
            return f"animated-{self.n}".encode()

    drv = _drv(AnimatedAdb())
    drv._session_top_done = True
    rewinds = []
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: "confirmed_not_top: animated sticky header")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: rewinds.append("rewind") or False)

    assert drv.current_profile() is None
    assert drv.current_profile() is None
    assert rewinds == ["rewind"]


def test_ambiguous_in_package_surface_never_rewinds_even_when_top_would_be_refuted(monkeypatch):
    """Hinge ownership alone cannot turn an unknown in-app page into a scroll target."""
    drv = _drv(FakeAdb([b"hinge-settings-or-modal"]))
    drv._session_top_done = True
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: None)
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: "confirmed_not_top: unrelated identity band")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: pytest.fail("ambiguous in-package surface must be no-input"))

    assert drv.current_profile() is None


def test_session_start_ambiguous_in_package_surface_never_spends_its_first_rewind(monkeypatch):
    """The session helper is guarded too; prep must not be the first no-input boundary."""
    drv = _drv(FakeAdb([b"hinge-settings-or-modal"]))
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: None)
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: pytest.fail("session rewind must require profile evidence"))

    assert drv.current_profile() is None
    assert drv._session_top_done is False


def test_failed_session_rewind_does_not_trigger_a_second_capture_entry_rewind(monkeypatch):
    """One public capture call gets one bounded session recovery attempt, never two ceilings."""
    drv = _drv(FakeAdb([b"deck"]))
    calls = []
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: calls.append("session-rewind") or False)
    monkeypatch.setattr(drv, "_prepare_actionable_capture_entry",
                        lambda *_a, **_k: pytest.fail("must not immediately retry a failed session rewind"))

    assert drv.current_profile() is None
    assert calls == ["session-rewind"]


def test_reviewed_capture_does_not_retry_a_failed_session_rewind(monkeypatch):
    """Reviewed's unconditional session helper must respect the same spent recovery latch."""
    drv = _drv(FakeAdb([b"deck"]))
    rewinds = []
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: rewinds.append("rewind") or False)
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: pytest.fail("failed recovery must not capture"))

    assert drv.current_profile_reviewed() is None
    assert drv.current_profile_reviewed() is None
    assert rewinds == ["rewind"]


def test_failed_manual_trailing_rewind_blocks_the_next_recapture(monkeypatch):
    """A READY rejection is also a spent recovery attempt, not a new swipe budget."""
    drv = _drv(FakeAdb([b"card"]))
    drv._session_top_done = True
    calls = []
    profile = SimpleNamespace(photos=[b"captured"])
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: calls.append("capture") or profile)
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: calls.append("trailing-rewind") or False)

    assert drv.current_profile() is None
    assert drv.current_profile() is None
    assert calls == ["capture", "trailing-rewind"]


def test_failed_split_recovery_blocks_an_immediate_second_rewind(monkeypatch):
    """A split's recovery failure cannot hand the next capture another autonomous budget."""
    drv = _drv(FakeAdb([b"advanced-card"]))
    drv._session_top_done = True
    drv._current_capture_split = True
    rewinds = []
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: rewinds.append("split-rewind") or False)
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: pytest.fail("spent split recovery must block recapture"))

    drv._recover_capture_split()
    assert drv.next_profile() is None
    assert rewinds == ["split-rewind"]


def test_actionable_capture_does_not_rewind_under_an_open_like_sheet(monkeypatch):
    """The repeated per-card entry gate keeps the existing compose-sheet no-input promise."""
    drv = _drv(FakeAdb([b"sheet"]))
    drv._session_top_done = True
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top", lambda: "top is unreadable")
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: True)
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: pytest.fail("must not rewind under a like sheet"))

    assert drv.current_profile() is None


def test_actionable_auto_capture_uses_the_same_pre_read_top_gate(monkeypatch):
    """The absolute item-index contract is mode-independent, not an Observe-only patch."""
    drv = _drv(FakeAdb([b"card"]))
    drv._session_top_done = True
    drv._current_capture_split = False
    events = []
    top_answers = iter(["confirmed_not_top", ""])
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: events.append("top") or next(top_answers))
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: events.append("rewind") or True)
    marker = object()
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: events.append("capture") or marker)

    assert drv.next_profile() is marker
    assert events == ["top", "rewind", "top", "capture"]


def test_auto_capture_entry_refusal_latches_a_worker_visible_block_reason(monkeypatch):
    """AUTO's `None` is a specific graceful block, not an unexplained end of the deck."""
    drv = _drv(FakeAdb([b"scrolled-card"]))
    drv._session_top_done = True
    drv._auto_session = True
    drv._current_capture_split = False
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: "confirmed_not_top: sticky profile header")
    monkeypatch.setattr(drv, "_scroll_to_top", lambda *_a, **_k: False)

    assert drv.next_profile() is None
    assert "capture entry could not be proven" in (drv.blocked_reason() or "")


def test_reviewed_actionable_capture_rewinds_before_building_its_item_anchor(monkeypatch):
    """The reviewed bridge retains only an index anchored after the fresh top proof."""
    drv = _drv(FakeAdb([b"card"]))
    events = []
    top_answers = iter(["confirmed_not_top", ""])
    monkeypatch.setattr(drv, "_ensure_session_top",
                        lambda *_a, **_k: events.append("session-rewind") or True)
    monkeypatch.setattr(drv, "_item_enumeration_blocker", lambda: "")
    monkeypatch.setattr(drv, "_confirm_enumeration_top",
                        lambda: events.append("top") or next(top_answers))
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda _frame: False)
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")
    monkeypatch.setattr(drv, "_scroll_to_top",
                        lambda *_a, **_k: events.append("rewind") or True)
    marker = object()
    monkeypatch.setattr(drv, "_capture_current",
                        lambda *_a, **_k: events.append("capture") or marker)

    assert drv.current_profile_reviewed() is marker
    assert events == ["session-rewind", "top", "rewind", "top", "capture"]


def test_next_profile_does_not_scroll_to_top_itself():
    # auto path: like() already calls _scroll_to_top() before acting, so next_profile() must
    # NOT also scroll back up -- that would be a redundant extra scroll (bug 2 fix note).
    # The ONLY unwind on this path is _ensure_session_top's one-shot pass, which is a different
    # thing entirely: it runs once per session, to establish where the card actually is (see
    # its docstring), not once per profile.
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)
    # Fake bytes have no Hinge filter-chip band; isolate the legacy generic contract this test
    # is asserting rather than asking the Hinge-specific affirmative-top path to decode them.
    drv.identity_band = None

    drv.next_profile()
    assert adb.swipes == 1               # the one-shot session-top pass, settled immediately

    swipes_after_first = adb.swipes
    drv.next_profile()
    assert adb.swipes == swipes_after_first   # and never again for the rest of the session


# --- session-start scroll-top invariant ------------------------------------------
# A Stop that abandons a read leaves the card scrolled (see the interruptible-capture tests
# above), and open_session only foregrounds the app -- so without this pass, the NEXT run's
# first capture starts mid-card and seeds the identity anchor from that person's sticky header
# instead of the app's scroll-top chrome, silently killing layer 1 for the first card.

def test_first_capture_of_a_session_confirms_the_card_is_at_the_top():
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)
    drv.identity_band = None
    assert drv._session_top_done is False

    drv.current_profile()

    assert drv._session_top_done is True


def test_session_top_pass_never_touches_an_open_like_sheet():
    """Swiping under a sheet the operator is composing in would drag it out from under them.
    _capture_current already refuses to read or scroll on a sheet; this pass runs BEFORE that
    check, so it has to make the same promise itself."""
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)

    assert drv.current_profile() is None
    assert adb.swipes == 0 and adb.scrolls == 0 and adb.taps == []
    # ...and it must NOT count as done: nothing was restored, so the capture after the sheet
    # closes has to try again.
    assert drv._session_top_done is False


def test_session_top_pass_says_so_when_it_cannot_confirm_the_top(monkeypatch, capsys):
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)   # never settles
    drv = _drv(FakeAdb([b"a"]), scroll_captures=3)
    monkeypatch.setattr(drv, "_capture_entry_profile_evidence", lambda _frame: "deck_controls")

    drv._ensure_session_top()

    assert drv._session_top_done is True          # one attempt per session, even on failure
    assert "could not confirm" in capsys.readouterr().out


def test_identity_never_manufactures_new_from_the_scroll_top_chrome_alone(monkeypatch):
    """With no per-profile header locked (_identity_sig is None), the scroll-top chrome can
    only positively prove 'top'. Answering 'new' for anything else invents a verdict from
    nothing -- and 'new' is the one verdict that SKIPS layer 2, so it goes straight to the
    deck-ready/settle check and can record a PASS for a human who only scrolled."""
    drv = _drv(FakeAdb([b"x"]))
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: frame)
    monkeypatch.setattr(hinge, "_band_dist", lambda a, b: 0.0 if a == b else 99.0)
    drv._identity_top_sig = b"chrome"
    drv._identity_sig = None

    assert drv._identity_of(b"chrome")[0] == "top"        # positively the scroll-top chrome
    assert drv._identity_of(b"anything-else")[0] == "unknown"   # NOT "new"


# --- Stop honoured DURING a profile read (the ~85s stop-deaf window) -------------
# Reported by the owner: "when I hit stop, it doesn't stop while it's reading a profile, it
# completes the read (by scrolling a bunch) then stops." Before this, worker.py's only stop
# check between one profile and the next ran AFTER current_profile() returned, so the whole
# read (12 screencaps + 11 read-scrolls) plus the whole scroll-back-to-top ran regardless.

def test_capture_abandons_the_read_as_soon_as_stop_is_requested(monkeypatch):
    # dwell_s=0 keeps the dwell instant while still exercising the in-dwell stop check.
    # 2026-08-24: the per-frame dwell was replaced with ONE randomized per-profile pause (see
    # _READ_PAUSE_COUNTS in hinge.py and ops/ANTI-BOT-RESEARCH.md's dated addendum), so which
    # iteration -- if any -- takes the pause is no longer "every one of them", it is a draw. Pin
    # that draw to iteration 1 so the stop-during-pause check this test exists for is exercised
    # deterministically: iteration 1's pause is where the read must notice `_should_stop` BEFORE
    # issuing iteration 1's scroll, exactly like iteration 1's dwell always did pre-2026-08-24.
    monkeypatch.setattr(HingeDriver, "_plan_read_pause_iterations", lambda self, _n: {1})
    adb = FakeAdb([b"a", b"b", b"c", b"d", b"e"])
    drv = _drv(adb, scroll_captures=5, dwell_s=0)

    # Stop lands once two frames have been read: the read must end there, not run to the ceiling.
    def _should_stop():
        return len(drv._current_sigs) >= 2

    assert drv._capture_current(_should_stop) is None      # partial read is discarded, never returned
    assert adb.scrolls == 1                                # only the one scroll already issued
    assert drv._current_sigs and len(drv._current_sigs) == 2   # it really did get two frames in


def test_capture_without_a_stop_callable_is_byte_for_byte_the_old_behaviour():
    # should_stop=None must reproduce the pre-fix path exactly -- tools/hinge_inspect.py and
    # every existing test double call the capture with no stop argument at all.
    adb = FakeAdb([b"a", b"b", b"b", b"c"])
    drv = _drv(adb)
    profile = drv._capture_current()
    assert profile.photos == [b"a", b"b"]
    assert adb.scrolls == 2


def test_stop_during_capture_leaves_the_scroll_ledger_intact_for_a_later_unwind(monkeypatch):
    # The ledger is the driver's only record of how far down the card actually is. An abandoned
    # read must not clear it, or a later _scroll_to_top (from a restarted session, like(), or
    # _locate_target_heart's fallback) would think it was already at the top and tap the wrong item.
    # Pin the read's one randomized pause to iteration 1, same reasoning as
    # test_capture_abandons_the_read_as_soon_as_stop_is_requested just above.
    monkeypatch.setattr(HingeDriver, "_plan_read_pause_iterations", lambda self, _n: {1})
    adb = FakeAdb([b"a", b"b", b"c", b"d"])
    drv = _drv(adb, scroll_captures=4, dwell_s=0)

    assert drv._capture_current(lambda: len(drv._current_sigs) >= 2) is None
    assert drv._capture_scroll_ledger                       # outstanding scrolls still recorded
    assert drv._capture_scrolls == len(drv._capture_scroll_ledger) == 1


def test_stop_during_scroll_to_top_stops_swiping_without_claiming_it_reached_the_top(monkeypatch):
    monkeypatch.setattr(HingeDriver, "_changed", lambda self, a, b: True)   # never settles on its own
    monkeypatch.setattr(hinge, "human_delay", lambda seconds, **_k: 0.0)    # instant settle waits
    drv = _drv(FakeAdb([b"stable"]))
    drv._capture_scroll_ledger = [(0.55, 0.5)] * 6
    drv._capture_scrolls = 6
    swipes = []
    monkeypatch.setattr(drv, "_swipe", lambda *args: swipes.append(args))

    drv._scroll_to_top(lambda: len(swipes) >= 2)

    assert len(swipes) == 2                                 # stopped early instead of all 6
    assert drv._capture_scroll_ledger                       # and did NOT claim to be back at the top
    assert drv._capture_scrolls == 6


def test_current_profile_still_unwinds_fully_when_no_stop_is_requested():
    # The stop plumbing must not short-circuit the normal path: a never-firing should_stop has
    # to behave exactly like the no-argument call.
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)
    drv.identity_band = None
    drv.current_profile(should_stop=lambda: False)
    assert adb.scrolls == 2
    assert adb.swipes >= 1                                  # scroll_to_top still ran


def test_android_driver_declares_interruptible_capture_and_the_abc_defaults_to_off():
    # worker.py only passes should_stop to a driver that advertises the capability, so the flag
    # must be True exactly where the code honours it -- a flag that lies is worse than no flag.
    from operation_love.drivers.base import DatingAppDriver
    assert DatingAppDriver.supports_interruptible_capture is False
    assert HingeDriver.supports_interruptible_capture is True


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


# --- Phase 4b: touch_backend="uhid_persistent" (2026-08-24) -----------------------------
class _MinimalFakePersistentPopen:
    """Just enough of subprocess.Popen for PersistentUhidTouch.open() to succeed with no
    real device or real `adb` involved: a live (never-dying) process whose stdin accepts
    every write. The write/report content itself is already pinned by
    tests/test_uhid.py's own PersistentUhidTouch suite -- this fake only needs to prove
    _make_touch() picked the RIGHT CLASS, not re-prove what that class does internally."""

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)

    def write(self, s):
        pass

    def flush(self):
        pass

    def close(self):
        pass

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0

    @property
    def stdin(self):
        return self

    @property
    def stdout(self):
        return self

    @property
    def stderr(self):
        return self


def test_make_touch_selects_persistent_uhid_when_backend_is_uhid_persistent(monkeypatch):
    from operation_love.drivers import uhid as uhid_mod
    from operation_love.drivers.uhid import PersistentUhidTouch, UhidTouch

    monkeypatch.setattr(uhid_mod.subprocess, "Popen", _MinimalFakePersistentPopen)
    monkeypatch.setattr(uhid_mod.time, "sleep", lambda *_a, **_k: None)

    adb = _HidAdb([b"x"])
    adb.adb_path = "adb"    # PersistentUhidTouch.open() reads these directly off self.adb,
    adb.serial = "pixel"    # the same convention Adb._run already uses (see adb.py)
    drv = HingeDriver(type("C", (), {"apps": {"hinge": {"touch_backend": "uhid_persistent"}}}))
    drv._adb = adb

    touch = drv._make_touch()

    assert isinstance(touch, PersistentUhidTouch)
    assert not isinstance(touch, UhidTouch)   # a genuinely different class, not a subclass


def test_touch_backend_uhid_persistent_required_raises_when_unavailable(monkeypatch):
    from operation_love.drivers import uhid as uhid_mod

    def _must_not_spawn(*_a, **_k):
        raise AssertionError("no real adb process should be spawned when hid is absent")

    monkeypatch.setattr(uhid_mod.subprocess, "Popen", _must_not_spawn)
    adb = FakeAdb([b"x"])                 # no hid -> UhidUnavailable, same probe as `uhid`
    drv = HingeDriver(type("C", (), {"apps": {"hinge": {"touch_backend": "uhid_persistent"}}}))
    drv._adb = adb
    with pytest.raises(DriverClosed):     # fails loudly -- no silent fallback to UhidTouch/adb
        drv._make_touch()


def test_touch_backend_auto_and_uhid_never_select_persistent_uhid(monkeypatch):
    """The behavior-preservation half of this wiring: `auto` (the default) and the explicit
    `uhid` value must keep constructing plain UhidTouch, byte-for-byte as before this class
    existed -- only the literal string "uhid_persistent" may ever select the new class."""
    from operation_love.drivers import uhid as uhid_mod
    from operation_love.drivers.uhid import PersistentUhidTouch, UhidTouch

    monkeypatch.setattr(uhid_mod.subprocess, "Popen", _MinimalFakePersistentPopen)

    for backend_cfg in ({"serial": "pixel"}, {"serial": "pixel", "touch_backend": "uhid"}):
        adb = _HidAdb([b"x"])
        drv = HingeDriver(type("C", (), {"apps": {"hinge": backend_cfg}}))
        drv._adb = adb
        touch = drv._make_touch()
        assert isinstance(touch, UhidTouch)
        assert not isinstance(touch, PersistentUhidTouch)


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


def _like_flow_frame_with_modal(heart_xy=(937, 1600), modal_xy=(420, 2197)):
    """A decodable frame carrying the heart glyph (so the heart tap is vision-located), the
    structurally confirmed inline composer (so its detector finds safe controls),
    AND the Rose-upsell 'Send Like anyway' glyph (so the same scripted frame also drives the
    post-send upsell dismissal) -- lets a single frame exercise the FULL like() path end to
    end, unlike the old vision-miss-then-fallback shortcut this replaces."""
    import cv2
    import numpy as np
    canvas = np.full((2400, 1080), 249, dtype=np.uint8)
    _paint_inline_composer(canvas)
    for name, (cx, cy) in (
        (hinge.HINGE_SPEC.templates["like"], heart_xy),
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


# --- opener item targeting: reach the RIGHT item, or STOP -----------------------------------
# THE RULE THESE PIN (ops/OPENER-REDESIGN.md 5.6, standing owner rule): never substitute a
# different item. Until 2026-08-12 every route that could not reach the item the opener was
# written about fell back to the topmost heart on screen and reported the miss as
# on_target=False, for the caller to repair the TEXT against. Repairing the text does not undo
# spending the LIKE on an item the model never chose, so every one of those routes is now a stop.
# The tests below are the same cases as before, inverted: what used to assert a fallback point
# plus a False flag now asserts a HingeTargetingError and an untouched screen.
def test_locate_target_heart_navigates_to_referenced_item(monkeypatch):
    """The happy path: the referenced frame is found, its heart is visible, and the method
    returns that heart's point. There is no second return value any more -- this method either
    answers with the chosen item's heart or raises, so there is no state left for an
    "is this actually the right one" flag to describe."""
    import numpy as np
    sig0, sig1 = np.zeros((24, 24)), np.ones((24, 24)) * 50
    adb = FakeAdb([b"f0", b"f1"])
    drv = _drv(adb)
    drv._current_sigs = [sig0, sig1]
    monkeypatch.setattr(hinge, "_downsample", lambda f, size=24: sig0 if f == b"f0" else sig1)
    monkeypatch.setattr(hinge, "_match_glyph",
                        lambda frame, t, side="right", **k: [(937, 1500)] if frame == b"f1" else [])
    # scrolled to frame 1 and took its heart
    assert drv._locate_target_heart(1) == (937, 1500)


def test_locate_target_heart_index0_uses_topmost():
    """index 0 -> topmost (first photo), no navigation needed. This is a FAST PATH, not a
    fallback: _like_comment_sheet scrolls to the top immediately before calling this, so the
    topmost heart on screen IS item 0's -- the item asked for and the item found are the same
    item, which is exactly what distinguishes it from the removed first-photo fallback."""
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)
    drv._current_sigs = [object(), object()]
    assert drv._locate_target_heart(0) == (937, 1600)


def test_locate_target_heart_none_taps_the_topmost_heart_because_nothing_was_chosen():
    """None means NOBODY SAID WHICH ITEM, and it is still not the same input as 0. It is legal
    only where there is no opener to misplace (Hinge with opener.enabled=false sends a plain
    like), and there it takes the topmost heart because something has to open the sheet. That is
    not a substitution: no item was ever chosen, so there is no chosen item to substitute FOR.
    The combination that WOULD be a substitution -- an opener plus a None index -- is refused one
    layer up, before any gesture; see the like() test for it below."""
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)
    drv._current_sigs = [object(), object()]
    assert drv._locate_target_heart(None) == (937, 1600)


def test_locate_target_heart_none_is_logged_as_its_own_reason(tmp_path):
    """"we were never told which item" and "we aimed at item N and could not find it" are
    different bugs in different layers, so a debug log must not render them the same. Recorded
    even though there ARE sigs to search -- unlike every other branch, this one is not a search
    that failed, and unlike every other branch it is not a stop either."""
    from operation_love.drivers.debuglog import HingeDebugLog
    import json
    drv = _drv(FakeAdb([_action_frame()]))
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="r")
    drv._current_sigs = [object(), object()]

    assert drv._locate_target_heart(None) == (937, 1600)

    recs = [json.loads(ln) for ln in (tmp_path / "r" / "actions.jsonl").read_text().splitlines()]
    entries = [r for r in recs if r["action"] == "locate_target_heart"]
    assert len(entries) == 1
    assert entries[0]["outcome"] == "no_item_named"      # not "stop": nothing was aimed at
    assert entries[0]["reason"] == "no_target_index"
    assert entries[0]["item_index"] is None


def test_capture_keeps_sigs_index_aligned_with_photos():
    # an undecodable middle frame: it lands in photos AND in _current_sigs (as None) at the SAME
    # index, so the driver's own capture-order index and the sig lookup never desync. (That
    # capture-order space is no longer what the opener answers in -- since 2026-08-12 the model
    # returns a 1-based index over numbered ITEMS, ops/OPENER-REDESIGN.md 5.1/5.7 -- but the
    # alignment pinned here is a property of the driver alone and holds either way.)
    adb = FakeAdb([_png(10), b"BADFRAME", _png(20), _png(20)])   # f3 dups f2 -> stop
    drv = _drv(adb)
    prof = drv._capture_current()
    assert len(drv._current_sigs) == len(prof.photos)
    assert drv._current_sigs[1] is None                  # the undecodable frame's aligned slot


def test_locate_target_heart_stops_rather_than_substitute_on_an_undecodable_sig():
    """An undecodable target signature used to fall back to the first photo's heart and report
    on_target=False. It now stops: the opener was written about item 1, and item 0 is a different
    item, which is the one thing that is never an option."""
    drv = _drv(FakeAdb([_action_frame()]))
    drv._current_sigs = [object(), None, object()]       # target index 1 is undecodable
    with pytest.raises(hinge.HingeTargetingError, match="could not be decoded"):
        drv._locate_target_heart(1)


def test_locate_target_heart_stops_rather_than_substitute_on_an_out_of_range_index():
    """item_index >= len(_current_sigs): the opener referenced an item outside the captured
    profile entirely. Also a stop, for the same reason -- there is no item here to land on, and
    landing on a different one is not a degraded success."""
    drv = _drv(FakeAdb([_action_frame()]))
    drv._current_sigs = [object(), object()]              # only indices 0..1 exist
    with pytest.raises(hinge.HingeTargetingError, match="outside the 2 frame"):
        drv._locate_target_heart(5)


def test_locate_target_heart_stops_rather_than_substitute_on_a_negative_index():
    """A negative index is invalid input, NOT the index-0 fast path, and the two must not
    collapse: treating -1 as "the first item" is precisely how an unusable index would once again
    resolve into a confident tap on card 1."""
    drv = _drv(FakeAdb([_action_frame()]))
    drv._current_sigs = [object(), object()]
    with pytest.raises(hinge.HingeTargetingError, match="not a valid capture-order index"):
        drv._locate_target_heart(-1)


def test_locate_target_heart_stops_when_no_frames_were_captured_at_all():
    """No _current_sigs means the profile on screen was never captured by this driver instance,
    so there is nothing to navigate back to. Old behaviour: tap the first heart anyway."""
    drv = _drv(FakeAdb([_action_frame()]))
    drv._current_sigs = []
    with pytest.raises(hinge.HingeTargetingError, match="no captured frames"):
        drv._locate_target_heart(3)


def test_locate_target_heart_retries_the_same_item_before_it_stops(monkeypatch):
    """RETRYING THE SAME ITEM IS ALLOWED AND IS TRIED FIRST -- a shaky hand is not a wrong
    decision. The search is run _TARGET_HEART_ATTEMPTS times, each from its own re-established
    scroll top, and only then does the run stop. Pinned by the number of forward read-scrolls:
    one attempt's worth would be at most `min(limit + 1, item_index + 3)`, and two attempts is
    what actually happens."""
    import numpy as np
    never_matches = np.ones((24, 24)) * 250
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: np.zeros((24, 24)))
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._current_sigs = [np.zeros((24, 24)), never_matches]

    with pytest.raises(hinge.HingeTargetingError):
        drv._locate_target_heart(1)

    per_attempt = min(drv._profile_capture_limit + 1, 1 + 3)
    assert adb.scrolls == per_attempt * hinge._TARGET_HEART_ATTEMPTS
    assert adb.swipes >= 1        # the retry re-established the top before searching again
    assert adb.taps == []         # locating never taps, and a stop least of all


def test_locate_target_heart_returns_the_heart_when_the_retry_finds_it(monkeypatch):
    """The other half of the retry rule: a target the FIRST search misses and the second finds is
    a success, not a stop. This is the case the retry exists for (HINGE-05's measured cause: a
    real-device scroll can over/undershoot the intended frame), and it must not cost a run."""
    import numpy as np
    target = np.ones((24, 24)) * 50
    seen = {"n": 0}

    def _ds(frame, size=24):
        seen["n"] += 1
        # Attempt 1 reads `per_attempt` frames and matches none of them; attempt 2's very first
        # frame is the target, which is what a corrected over/undershoot looks like.
        return target if seen["n"] > 4 else np.zeros((24, 24))

    monkeypatch.setattr(hinge, "_downsample", _ds)
    monkeypatch.setattr(hinge, "_match_glyph", lambda frame, t, side="right", **k: [(937, 1500)])
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._current_sigs = [np.zeros((24, 24)), target]

    assert drv._locate_target_heart(1) == (937, 1500)
    assert hinge._TARGET_HEART_ATTEMPTS >= 2      # or the retry above never ran at all


def test_locate_target_heart_records_the_stop_when_the_target_is_never_found(monkeypatch, tmp_path):
    """HINGE-05's debug record survives the change from fallback to stop, and says which of the
    two search failures it was: the frame never matched, versus it matched and carried no heart.
    Every attempt is recorded, and only the last one is an outcome of "stop"."""
    from operation_love.drivers.debuglog import HingeDebugLog
    import json
    import numpy as np
    never_matches = np.ones((24, 24)) * 250
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: np.zeros((24, 24)))
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="r")
    drv._current_sigs = [np.zeros((24, 24)), never_matches]

    with pytest.raises(hinge.HingeTargetingError, match="none of the frames read back matched"):
        drv._locate_target_heart(1)

    recs = [json.loads(ln) for ln in (tmp_path / "r" / "actions.jsonl").read_text().splitlines()]
    entries = [r for r in recs if r["action"] == "locate_target_heart"]
    assert len(entries) == hinge._TARGET_HEART_ATTEMPTS
    assert [e["outcome"] for e in entries] == ["retry", "stop"]
    assert {e["reason"] for e in entries} == {"target_frame_not_found"}


def test_locate_target_heart_records_the_stop_when_the_matched_frame_has_no_heart(monkeypatch, tmp_path):
    """The other search failure: the frame IS the one the target was captured on, but no like
    glyph is visible on it. Distinguished in both the debug record and the raised message, because
    "we never found the card" and "we found it and it has no heart" are different bugs."""
    from operation_love.drivers.debuglog import HingeDebugLog
    import json
    import numpy as np
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: np.zeros((24, 24)))
    monkeypatch.setattr(hinge, "_match_glyph", lambda *a, **k: [])     # nothing on the frame
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="r")
    drv._current_sigs = [np.zeros((24, 24)), np.zeros((24, 24))]       # target matches at once

    with pytest.raises(hinge.HingeTargetingError, match="no like heart was visible"):
        drv._locate_target_heart(1)

    recs = [json.loads(ln) for ln in (tmp_path / "r" / "actions.jsonl").read_text().splitlines()]
    entries = [r for r in recs if r["action"] == "locate_target_heart"]
    assert {e["reason"] for e in entries} == {"heart_not_visible_on_matched_frame"}


def test_locate_target_heart_bounds_search_when_target_not_found(monkeypatch):
    # Old code swept the FULL scroll_captures+1 depth (9 scrolls, for the default
    # scroll_captures=8) before giving up on a target it never matched -- a target already
    # scrolled past effectively costs a full wasted sweep. Each attempt's search must stay capped
    # close to item_index (how far down from the top the target should be), so the bound holds
    # per attempt and the retry multiplies a small number rather than a full sweep.
    import numpy as np
    never_matches = np.ones((24, 24)) * 250
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: np.zeros((24, 24)))
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb, scroll_captures=8)
    drv._current_sigs = [np.zeros((24, 24)), never_matches]

    with pytest.raises(hinge.HingeTargetingError):
        drv._locate_target_heart(1)

    # old code: 9 (scroll_captures + 1) for ONE attempt; capped near item_index + 3 per attempt
    assert adb.scrolls <= 4 * hinge._TARGET_HEART_ATTEMPTS


def test_a_targeting_stop_is_the_driver_agnostic_type_the_worker_catches():
    """The exception has to satisfy two callers at once. worker.py is app-agnostic and catches
    base.ItemTargetingError (it must not import a Hinge symbol to recognise the one failure class
    with a specific non-error stop to render); everything that has always caught this driver's
    action failures by type still sees a HingeActionError. It also carries intended/actual/stage
    as FIELDS, so the worker's stop record is read off the exception rather than parsed back out
    of its prose."""
    from operation_love.drivers.base import ItemTargetingError
    from operation_love.drivers.hinge import HingeActionError
    drv = _drv(FakeAdb([_action_frame()]))
    drv._current_sigs = [object(), object()]

    with pytest.raises(hinge.HingeTargetingError) as exc:
        drv._locate_target_heart(5)

    assert isinstance(exc.value, ItemTargetingError)
    assert isinstance(exc.value, HingeActionError)
    assert exc.value.stage == "navigate"
    assert exc.value.intended == 5
    assert exc.value.actual is None          # we never got far enough to see what we would hit
    assert exc.value.index_space == "capture_order"


# --- like(): never substitutes, and there is no repair callback left to substitute WITH ------
def test_like_rejects_a_capture_order_opener_before_targeting():
    """Only a calibrated model-item number can now license text on a comment sheet."""
    adb = FakeAdb([_action_frame()])

    with pytest.raises(hinge.HingeTargetingError, match="model item index"):
        _drv(adb).like(opener="original opener about the first photo", item_index=0)

    assert adb.texts == [] and adb.taps == []


def test_like_stops_and_sends_nothing_when_targeting_cannot_reach_the_item():
    """THE CORE REGRESSION TEST, inverted from what it used to be. It used to assert that a
    targeting miss called an anchored_opener callback and typed its replacement under whatever
    item the fallback tap had landed on. That IS the substitution the owner rule forbids -- the
    like still went to an item the model never chose, and only the wording was made to agree with
    it afterwards. Now: no tap, no text, no Send, and the run stops."""
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)
    drv._current_sigs = [object(), object()]        # item 5 is outside the capture

    with pytest.raises(hinge.HingeTargetingError, match="model item index"):
        drv.like(opener="original opener about a different item", item_index=5)

    assert adb.taps == []       # the heart was never tapped, so no sheet and no like
    assert adb.texts == []


def test_like_refuses_an_opener_with_no_item_index_at_all_before_touching_the_screen():
    """An opener with no item index is "we have real text and no idea which photo or prompt it
    would hang under". It used to tap the first heart and route the text through the repair hatch;
    it is now refused before the scroll, so the screen is exactly as the caller left it -- no
    gesture of any kind, not just no tap. The message names the cause rather than rendering it as
    "item None"."""
    adb = FakeAdb([_action_frame()])

    with pytest.raises(hinge.HingeTargetingError, match="model item index"):
        _drv(adb).like(opener="original opener about an unidentified item")

    assert adb.taps == [] and adb.texts == []
    assert adb.scrolls == 0 and adb.swipes == 0     # not even the scroll-to-top happened


def test_like_with_no_opener_and_no_item_still_likes():
    """The openers-disabled path (opener.enabled=false in config), which must keep working
    exactly as if openers had never existed: a plain like, no comment, no item chosen and
    therefore nothing that could be substituted for one."""
    adb = FakeAdb([_action_frame()])

    _drv(adb).like()

    assert adb.texts == []
    assert (937, 1600) in adb.taps                  # the heart was tapped


def test_like_no_longer_accepts_an_anchored_opener_callback():
    """The repair hatch is REMOVED, not merely unused: a caller still passing one must fail
    loudly at the call rather than have its callback silently ignored, which would look exactly
    like a repair that was never needed. Pins the parameter's absence from the whole chain
    (base.Driver.like -> hinge.like -> _like_comment_sheet)."""
    adb = FakeAdb([_action_frame()])

    with pytest.raises(TypeError):
        _drv(adb).like(opener="x", item_index=0, anchored_opener=lambda anchor: "repaired")
@pytest.mark.parametrize(
    ("sheet_visible", "modal_visible", "message"),
    [
        (True, False, "like did not complete — the like composer is still open"),
        (False, True, "like did not complete — the like upsell modal is still open"),
        (True, True,
         "like did not complete — the like composer and upsell modal are still open"),
    ],
)
def test_verify_like_landed_names_the_open_blocker(
        monkeypatch, sheet_visible, modal_visible, message):
    """The terminal diagnostic names the concrete post-send UI still blocking progress."""
    from operation_love.drivers.hinge import HingeActionError
    drv = _drv(FakeAdb([b"f"]), halt_on_error=True)
    monkeypatch.setattr(drv, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible",
                        lambda _frame: sheet_visible)
    monkeypatch.setattr(hinge, "_match_glyph",
                        lambda *a, **k: [(1, 1)] if modal_visible else [])

    with pytest.raises(HingeActionError) as exc:
        drv._verify_like_landed(b"before")

    assert str(exc.value) == message


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
    calls = {"modal": 0}

    # Give the matcher logical role names instead of depending on its global invocation count.
    # `_verify_like_landed` deliberately probes paywall before the sheet/modal glyphs now; that
    # extra read-only probe must not change when the Rose itself arrives.
    drv = _drv(FakeAdb([b"before", b"before", b"after"], advance_on_screencap=True),
               halt_on_error=True)
    monkeypatch.setattr(drv, "_template", lambda role: role)

    def fake_match(frame, template, side="any", threshold=0.6):
        if template != "upsell_dismiss":             # paywall + sheet are absent throughout
            return []
        calls["modal"] += 1
        # Iteration 1: not yet animated; iteration 2: up and dismissed; iteration 3: gone.
        if calls["modal"] == 2:
            return [(5, 5)]
        return []

    monkeypatch.setattr(hinge, "_match_glyph", fake_match)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))

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


def test_screencap_retry_still_enforces_v2_calibrated_frame_size():
    """A blank first frame cannot bypass the geometry check on the recovered frame."""
    calibration = {
        "schema_version": 3,
        "device": "pixel",
        "hinge_version_name": "9.134.0",
        "frame_size_px": [1080, 2400],
        "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 2.0,
        "inline_item_max_dist": 10.0,
        "calibrated_at": "synthetic",
        "identity_band": list(hinge.HINGE_SPEC.identity_band),
        "content_band": list(hinge.HINGE_SPEC.content_band),
    }
    adb = FakeAdb(
        [_png(0), _noisy_png(size=(720, 1600))], advance_on_screencap=True)
    with pytest.raises(hinge.HingeActionError, match="does not match schema-v3"):
        _drv(adb, targeting_calibration=calibration)._screencap()


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


def test_wait_for_decision_resumes_after_the_screen_comes_back(monkeypatch):
    """Asleep, then the owner wakes it and swipes -> the decision is still detected."""
    import numpy as np
    asleep = _png(0)
    woken = _noisy_png(lo=10, hi=60)
    advanced = _noisy_png(lo=150, hi=250)
    adb = FakeAdb([asleep, woken, advanced],
                  advance_on_screencap=True)
    drv = _drv(adb)
    old_sig = np.full((16, 64), 10, dtype="int16")
    new_sig = np.full((16, 64), 250, dtype="int16")
    drv._identity_sig = old_sig
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    monkeypatch.setattr(hinge, "_band",
                        lambda frame, rect: new_sig if frame == advanced else old_sig)
    # Deck-ready evidence for the final frame -- see test_pass_detected_on_card_advance's
    # comment: these noise frames carry no real glyph for _observe_deck_ready to match.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda frame: "pass")
    assert drv.wait_for_decision(timeout=5.0) is False   # advanced -> pass


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


# --- OCR is asymmetric: it may only ever say "same", never "new" --------------------------
def _identity_drv(monkeypatch, adb=None, **cfg):
    """Driver with a captured profile identity in place and OCR forced on, so
    _identity_of's OCR branch is exercised without needing a real tesseract."""
    import numpy as np
    drv = _drv(adb or FakeAdb([b"a"]), observe_name_ocr=True, **cfg)
    drv._identity_sig = np.full((16, 64), 10, dtype="int16")     # captured profile's header
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")  # scroll-top chrome
    drv._identity_name = "qelix"
    return drv


def test_ocr_cannot_turn_a_scroll_top_frame_into_a_new_profile(monkeypatch):
    """REGRESSION: at scroll-top the identity band shows the app's filter-chips row, which
    OCRs to something like "Signals ( Agev ) Height v" -- a non-empty string that matches no
    name. Letting a name MISMATCH force 'new' flipped that correct 'top' verdict straight to
    'new', i.e. manufactured the exact "this is a different profile" claim a PASS label is
    built on, out of chrome the human never touched.

    This is specifically about identity_band's OWN OCR corroboration (the `psm` default,
    i.e. NOT "6") -- Layer 1b's separate card-header read (psm="6") is a different band with
    its own dedicated tests above; stub it to return nothing here so it stays out of the way
    of what THIS test is actually exercising.
    """
    drv = _identity_drv(monkeypatch)
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: drv._identity_top_sig)
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7", **_kwargs: (
            None if psm == "6" else "Signals ( Agev ) Height v"))

    assert drv._identity_of(b"scroll-top-frame")[0] == "top"


def test_ocr_cannot_turn_a_pixel_match_into_a_new_profile(monkeypatch):
    """A garbled tesseract read on a frame the pixel band already proved is the SAME profile
    must not be able to override it -- a false 'new' writes a wrong training label, a false
    'same' costs at most a missed pass."""
    drv = _identity_drv(monkeypatch)
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: drv._identity_sig)
    monkeypatch.setattr(drv, "_ocr_band", lambda frame, rect: "jessi<a")   # garbled

    assert drv._identity_of(b"same-profile")[0] == "same"


def test_ocr_can_still_rescue_a_position_shifted_header_as_same(monkeypatch):
    """The whole reason OCR is wired in at all: a header that moved a few px (a banner
    appearing) fails the position-sensitive pixel compare, but the NAME is still the name."""
    import numpy as np
    drv = _identity_drv(monkeypatch)
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: np.full((16, 64), 120, dtype="int16"))
    monkeypatch.setattr(drv, "_ocr_band", lambda frame, rect: "Qelix")   # case-insensitive

    assert drv._identity_of(b"shifted-header")[0] == "same"


# --- tap slop and control radius are different numbers -----------------------------------
def test_tap_slop_is_much_tighter_than_the_control_radius():
    """REGRESSION: these were the same number (12% of screen height, ~288px on this panel).
    Reused as tap slop, that classifies an ordinary ~200px read-scroll as a "tap", which then
    only has to land within the equally generous radius of the pass-X to be corroborated as a
    deliberate PASS. A scroll must never be able to become a decision."""
    drv = _drv(FakeAdb([b"a"]))
    slop = drv._observe_tap_slop_px()
    radius = drv._observe_tap_radius_px()

    assert slop < radius / 3            # not merely different -- a different order of size
    assert slop < 200.0                 # a 200px flick is a scroll, and must not be a tap
    assert radius >= 200.0              # but the button itself still gets a generous target


def test_like_vs_scroll_uses_the_identity_anchor_instead_of_lookalike_photos(monkeypatch):
    """REGRESSION (audit-flagged, previously deferred): _is_current_profile_frame was
    full-frame min-over-all-captured-sigs, so a NEW card whose first photo merely looked like
    one of the current profile's frames (centred face, light background -- i.e. most dating
    photos) collided and read as a scroll. That drops a real LIKE and desyncs the loop: the
    NEXT decision gets attributed to this profile's photos. The identity anchor answers "same
    card?" without depending on how alike two people's photos are, so it wins outright."""
    import numpy as np
    drv = _drv(FakeAdb([b"a"]), observe_name_ocr=False)
    drv._identity_sig = np.full((16, 64), 10, dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    drv._identity_name = None
    # A colliding photo signature: full-frame matching alone would call this the same profile.
    drv._current_sigs = [np.full((24, 24), 50, dtype="int16")]
    monkeypatch.setattr(hinge, "_downsample", lambda f: np.full((24, 24), 50, dtype="int16"))
    # ...but the sticky header says it is a DIFFERENT person.
    monkeypatch.setattr(hinge, "_band", lambda f, rect: np.full((16, 64), 120, dtype="int16"))

    assert drv._is_current_profile_frame(b"new-card-that-looks-alike") is False


def test_like_vs_scroll_still_falls_back_to_photos_when_the_header_is_not_visible(monkeypatch):
    """A 'top'/'unknown' identity verdict must not become a silent "it's a new card": at
    scroll-top the header simply isn't showing, and the original full-frame comparison is
    still the best available answer -- no worse than the behaviour this replaced."""
    import numpy as np
    drv = _drv(FakeAdb([b"a"]), observe_name_ocr=False)
    top = np.full((16, 64), 200, dtype="int16")
    drv._identity_sig = np.full((16, 64), 10, dtype="int16")
    drv._identity_top_sig = top
    drv._identity_name = None
    drv._current_sigs = [np.full((24, 24), 50, dtype="int16")]
    monkeypatch.setattr(hinge, "_downsample", lambda f: np.full((24, 24), 50, dtype="int16"))
    monkeypatch.setattr(hinge, "_band", lambda f, rect: top)     # -> 'top', header not visible

    assert drv._is_current_profile_frame(b"scrolled-back-to-top") is True


def test_current_profile_diagnostics_keep_original_signature_index_after_none_slots(monkeypatch):
    """A diagnostic index must point at the capture deque, not its filtered position."""
    import numpy as np

    drv = _drv(FakeAdb([b"a"]), observe_name_ocr=False)
    target = np.full((24, 24), 50, dtype="int16")
    drv._current_sigs = [None, np.zeros((24, 24), dtype="int16"), target]
    monkeypatch.setattr(drv, "_identity_of", lambda _frame: ("top", None))
    monkeypatch.setattr(hinge, "_downsample", lambda _frame: target)

    diagnostics = {}
    assert drv._is_current_profile_frame(b"matching-third-capture", diagnostics=diagnostics)
    assert diagnostics["current_content_exact_min_index"] == 2


# --- review findings: identity precedence, like-flow corroboration, capture split ---------
def _idrv(monkeypatch, adb=None, **cfg):
    """Driver with an identity anchor already locked to a captured profile."""
    import numpy as np
    drv = _drv(adb or FakeAdb([b"a"]), observe_name_ocr=False, **cfg)
    drv._identity_sig = np.full((16, 64), 10, dtype="int16")
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    drv._identity_name = None
    return drv


def test_a_content_collision_cannot_override_an_authoritative_identity_new(monkeypatch):
    """REGRESSION: wait_for_decision's own comment promised layers 2/3 could only CORROBORATE
    an identity verdict, but the code only short-circuited on 'same' -- Layer 2's photo match
    still ran after identity said 'new'. Dating first photos are alike (centred face, light
    background), so a genuine advance could collide with the PREVIOUS profile's captured frame
    and be swallowed as 'just a scroll': the human's decision on the card they actually left
    is never recorded, and their NEXT one gets attributed to the old profile's photos."""
    import numpy as np
    drv = _idrv(monkeypatch, FakeAdb([b"base", b"newcard"], advance_on_screencap=True))
    drv._current_sigs = [np.full((24, 24), 50, dtype="int16")]
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (50.0, 50.0))
    # The new card's photos coincidentally match the OLD profile's stored signature...
    monkeypatch.setattr(hinge, "_downsample", lambda f: np.full((24, 24), 50, dtype="int16"))
    # ...but its sticky header is a different person's -> identity 'new'.
    monkeypatch.setattr(hinge, "_band", lambda f, rect: np.full((16, 64), 120, dtype="int16"))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda f: True)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda f: False)
    monkeypatch.setattr(drv, "_changed", lambda a, b: False)
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda f: "pass")

    assert drv.wait_for_decision(timeout=5.0) is False   # the advance is NOT swallowed


def test_like_flow_requires_content_as_well_as_identity_before_calling_a_like_cancelled(monkeypatch):
    """REGRESSION: the identity band holds only a FIRST NAME, so two adjacent profiles sharing
    one render an identical header. At _await_like_resolved's call site a wrong 'same' does not
    merely defer -- it reports a like the human really SENT as a dismissal, dropping it. So
    that caller now needs the header AND the photos to agree."""
    import numpy as np
    drv = _idrv(monkeypatch)
    monkeypatch.setattr(hinge, "_band", lambda f, rect: drv._identity_sig)   # collided header
    drv._current_sigs = [np.full((24, 24), 10, dtype="int16")]
    monkeypatch.setattr(hinge, "_downsample", lambda f: np.full((24, 24), 200, dtype="int16"))

    # identity alone still says "current profile" for the cheap caller...
    assert drv._is_current_profile_frame(b"nextcard") is True
    # ...but the like flow, where a wrong answer discards a real decision, refuses.
    assert drv._is_current_profile_frame(b"nextcard", require_content=True) is False


def test_two_pass_taps_in_one_wait_resync_rather_than_guess_the_profile(monkeypatch):
    """A human does not press pass twice on one card. A second pass-control tap inside a single
    wait is direct evidence an earlier decision was swallowed (the same-first-name blind spot),
    so the loop must resync instead of attaching the ambiguity to whatever profile it holds."""
    from operation_love.drivers.touchwatch import Gesture

    drv = _idrv(monkeypatch)

    class _W:
        alive = True
        event_count = 40
        def gestures_since(self, t):
            return [Gesture(t_down=1.0, t_up=1.1, down=(130, 2035), up=(130, 2035), travel_px=1.0),
                    Gesture(t_down=2.0, t_up=2.1, down=(128, 2030), up=(128, 2030), travel_px=1.0)]

    drv._touch_watcher = _W()
    drv.observe_touch_watch = True
    drv._observe_since = 0.0
    monkeypatch.setattr(drv, "_observe_locate_pass_x", lambda f: (125, 2035))

    assert drv._observe_gesture_verdict(b"frame") == "resync"


def test_one_pass_tap_in_a_wait_is_still_a_pass(monkeypatch):
    """The guard above must not break the ordinary case it sits next to."""
    from operation_love.drivers.touchwatch import Gesture

    drv = _idrv(monkeypatch)

    class _W:
        alive = True
        event_count = 40
        def gestures_since(self, t):
            return [Gesture(t_down=1.0, t_up=1.4, down=(540, 1800), up=(540, 900), travel_px=900.0),
                    Gesture(t_down=2.0, t_up=2.1, down=(130, 2035), up=(130, 2035), travel_px=1.0)]

    drv._touch_watcher = _W()
    drv.observe_touch_watch = True
    drv._observe_since = 0.0
    monkeypatch.setattr(drv, "_observe_locate_pass_x", lambda f: (125, 2035))

    assert drv._observe_gesture_verdict(b"frame") == "pass"


def test_a_deck_advance_mid_capture_discards_the_mixed_profile(monkeypatch):
    """REGRESSION: _capture_current stopped only on a repeat frame or the ceiling, so if the
    deck advanced while the bot was still reading (Hinge shows no on-screen busy overlay, and
    the human's finger is a concurrent input stream), the frames after the advance were
    appended as more of the SAME person. worker.py would then mean-pool two different faces
    into one embedding and store it under one label."""
    import numpy as np
    # Identity band per frame: f0 is the scroll-top chrome, f1/f2 are THIS profile's sticky
    # header, f3 is a different person's. Content is tracked separately and must differ per
    # frame -- _frame_sig hashes the DOWNSAMPLE, so equal content would read as "reached the
    # bottom" and stop the loop before it ever saw f3.
    bands = {b"f0": 200, b"f1": 10, b"f2": 10, b"f3": 120}
    content = {b"f0": 200, b"f1": 90, b"f2": 91, b"f3": 120}
    adb = FakeAdb([b"f0", b"f1", b"f2", b"f3"])
    drv = _drv(adb, observe_name_ocr=False, scroll_captures=6)
    # Observe enumerates since doc 5.9's inversion, and its scroll-top gate reads this same band
    # through the driver's one decode -- with a THIRD argument (the grid). The 2-argument lambda
    # this fixture used to install raised a TypeError out of that gate rather than being refused
    # by it. `*rest` keeps the fake honest for both callers; what this test is about is untouched.
    monkeypatch.setattr(hinge, "_band",
                        lambda f, rect, *rest: np.full((16, 64), bands[f], dtype="int16"))
    monkeypatch.setattr(hinge, "_downsample",
                        lambda f: np.full((24, 24), content[f], dtype="int16"))
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda f: False)
    monkeypatch.setattr(drv, "_scroll_down_one", lambda *a, **k: adb.scroll_up())

    assert drv._capture_current() is None      # discarded, not returned as one mixed Profile
