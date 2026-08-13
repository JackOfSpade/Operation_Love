"""Hinge driver — offline tests for the HOST-SIDE ADB rewrite.

No device: a FakeAdb scripts screencap frames and records taps/swipes/text, so we
exercise capture (humanized read + dedup), the like/dislike/like-with-comment tap
sequences, and wait_for_decision's like/pass/none logic. Non-PNG fake frames make
the driver's frame-diff fall back to exact-bytes (any change = whole-frame change),
which is enough for pass/none/stop/timeout; the region-based LIKE path is driven by
monkeypatching _split_diff with scripted (top, bottom) deltas. Real coordinates and
diff thresholds are confirmed live on a finished profile (see hinge.py header).
"""
import math
import random

import pytest

from operation_love.drivers import hinge
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


def _action_frame(heart_xy=(937, 1600), x_xy=(125, 2035), confirm_xy=(540, 1300)):
    """A decodable frame with the like-heart, pass-X, and comment-sheet 'confirm' glyphs
    pasted at known spots, so the driver's vision locator finds them — exercises the real
    action path, not the fallback. The confirm glyph is what lets _await_sheet_open() (the
    gate that confirms the comment sheet actually opened before the driver taps the FIXED
    comment_box / send_like coordinates) succeed from a single scripted frame.

    The heart glyph is HINGE_SPEC.templates["like"] (hinge_like_button.png), not a hardcoded
    filename — it must always be whatever the real spec currently wires to the "like" role, or
    this fixture silently stops representing a real deck the moment that role's asset changes
    again (as it did away from hinge_heart.png, the "Which do we have in common" outline
    heart — see the templates dict comment on HINGE_SPEC in hinge.py)."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(1)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in (
        (hinge.HINGE_SPEC.templates["like"], heart_xy),
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
    assert profile.meta["capture_truncated"] is False    # the repeat proves the true bottom was seen
    assert drv._current_capture_truncated is False


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
    # item_index=0 -- "the opener is about the first captured frame", the shape the live pipeline
    # produces for model item 1 in profile-photos space. An opener with NO item index at all is
    # refused before any gesture since 2026-08-12 (never attach real text to whichever heart
    # happens to be topmost); that refusal has its own test below.
    _drv(adb).like("loved your stargazing prompt", 0)
    heart = (937, 1600)                          # vision-located
    box = (int(0.500 * 1080), int(0.529 * 2400))
    send = (int(0.643 * 1080), int(0.576 * 2400))
    assert adb.taps == [heart, box, send]
    assert adb.texts == ["loved your stargazing prompt"]


# --- observe: wait_for_decision via frame deltas ------------------------
def test_pass_detected_on_card_advance(monkeypatch):
    # The new rule requires POSITIVE proof of a different, deck-ready, settled card (not
    # just "changed and unrecognised") before concluding PASS -- see the redesign notes on
    # wait_for_decision. These raw, non-PNG fake frames can never satisfy the real glyph
    # template match _observe_deck_ready performs, so the test supplies that evidence
    # directly, the same way test_like_detected_sheet_then_advance already does for the
    # like-sheet path above.
    adb = FakeAdb([b"a", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    assert drv.wait_for_decision(timeout=5.0) is False


def test_waits_until_a_change_then_pass(monkeypatch):
    adb = FakeAdb([b"a", b"a", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)  # deck-ready evidence -- see above
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
    # bottom-only change (sheet up), then a stable ready deck -> like sent
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"sheet", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"b")
    assert drv.wait_for_decision(timeout=5.0) is True


def test_like_writes_its_own_decision_record(monkeypatch):
    """A LIKE used to write NO decision record at all: observe_decision hardcoded
    decision="pass", and the two `return True` sites logged nothing. observe_like_anchor fires
    on INTENT (a sheet was spotted), not on resolution, so an actions.jsonl reader could not
    tell a sent like from a sheet the human opened and backed out of. A real run
    (2026-08-10) shows exactly that hole: capture(jessica) -> observe_waiting ->
    observe_like_anchor -> capture(Victoria), with nothing on disk proving jessica was liked."""
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (50.0, 0.0), (50.0, 0.0)))
    adb = FakeAdb([b"a", b"sheet", b"b"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._identity_name = "jessica"
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: frame == b"b")

    assert drv.wait_for_decision(timeout=5.0) is True

    decisions = [f for name, f in drv._dbg.calls if name == "observe_decision"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "like"
    assert decisions[0]["profile_name"] == "jessica"
    # Layer 3 measures distance to the PASS control, so running it on a like would confidently
    # report "resync" for a tap that correctly hit the heart. An honest absence beats a wrong answer.
    assert decisions[0]["gesture"] == "not_checked"


def test_dismissed_like_sheet_is_recorded_but_never_as_a_decision(monkeypatch):
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"sheet", b"a", b"c"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    # A real, matchable Send Like glyph while the sheet is up, gone once it is dismissed --
    # this is what separates a genuine dismissal from the sibling test below. Counted rather
    # than keyed on frame bytes because FakeAdb replays one frame until a scroll advances it.
    polls = {"n": 0}

    def _sheet_visible(_frame):
        polls["n"] += 1
        return polls["n"] <= 2                   # up for the first two polls, then dismissed

    monkeypatch.setattr(drv, "_observe_like_sheet_visible", _sheet_visible)

    assert drv.wait_for_decision(timeout=5.0) is False       # the later advance is the pass

    names = [name for name, _f in drv._dbg.calls]
    assert "observe_like_dismissed" in names                 # the backed-out sheet left a trace...
    decisions = [f for name, f in drv._dbg.calls if name == "observe_decision"]
    assert [d["decision"] for d in decisions] == ["pass"]     # ...but was never counted as one


def test_bottom_delta_with_no_sheet_is_not_logged_as_a_dismissed_like(monkeypatch):
    """A bottom-only change is not proof of a like sheet: a snackbar ("Your like was sent"), a
    toast, or a keyboard dismissal produces the same delta. Recording those as a dismissed like
    sheet would assert something nobody ever observed, in the log you read specifically to find
    out what the human did."""
    monkeypatch.setattr(hinge, "_split_diff",
                        _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"snackbar", b"a", b"c"])
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: False)   # never a sheet

    assert drv.wait_for_decision(timeout=5.0) is False

    names = [name for name, _f in drv._dbg.calls]
    assert "observe_bottom_delta" in names
    assert "observe_like_dismissed" not in names


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
    # sheet up, reverts to the SAME card (cancelled), then a real advance (pass)
    monkeypatch.setattr(hinge, "_split_diff", _ScriptedDiff((2.0, 50.0), (2.0, 2.0), (2.0, 2.0), (50.0, 50.0)))
    adb = FakeAdb([b"a", b"sheet", b"a", b"c"])
    drv = _drv(adb)
    # Deck-ready evidence for "c" -- see test_pass_detected_on_card_advance's comment: the
    # new rule requires positive proof of a settled deck-ready card (these raw fake frames
    # never satisfy the real glyph template match) before it will conclude PASS at all.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    assert drv.wait_for_decision(timeout=5.0) is False


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
    # Deck-ready evidence for "c" -- see test_pass_detected_on_card_advance's comment: these
    # raw fake frames can't satisfy the real glyph template match, and the new rule requires
    # positive proof of a settled deck-ready card before concluding PASS.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)

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
    # Deck-ready evidence for "new" -- see test_pass_detected_on_card_advance's comment: the
    # new rule requires positive proof of a settled deck-ready card (these raw fake frames
    # never satisfy the real glyph template match) before it will conclude PASS at all.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)

    assert drv.wait_for_decision(timeout=5.0) is False   # only "new" is a genuine pass
    assert adb.i == 2                                    # both "mid" and "new" were observed


class _FakeDbg:
    def __init__(self):
        self.calls = []

    def action(self, name, *, before=None, after=None, **fields):
        self.calls.append((name, fields))


def test_wait_for_decision_records_pass_diagnostics_in_the_debug_log(monkeypatch):
    """The bug report that motivated this fix had NO actions.jsonl entry explaining why a PASS
    was recorded -- just a bare 'Got PASS' in the console log. A silent pixel-only PASS verdict
    must leave a paper trail for the next report to point at."""
    import numpy as np
    adb = FakeAdb([b"a", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._dbg = _FakeDbg()
    drv._current_sigs = [np.ones((24, 24))]
    drv._current_capture_truncated = False
    monkeypatch.setattr(
        hinge, "_downsample",
        lambda frame, size=24: np.ones((24, 24)) if frame == b"a" else np.full((24, 24), 220.0))
    monkeypatch.setattr(hinge, "_split_diff", lambda x, y: (0.0, 0.0) if x == y else (15.0, 15.0))
    # Deck-ready evidence for "new" -- see test_pass_detected_on_card_advance's comment.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)

    assert drv.wait_for_decision(timeout=5.0) is False

    decisions = [fields for name, fields in drv._dbg.calls if name == "observe_decision"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "pass"
    assert decisions[0]["shift_matched"] is False
    assert decisions[0]["capture_truncated"] is False
    assert decisions[0]["top"] == 15.0 and decisions[0]["bot"] == 15.0


def test_observe_scroll_shift_match_record_carries_sig_index_shift_overlap_rows_and_name_read(
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
    assert rec["name_read"] is None    # identity never reached "top" here (raw bytes -> "unknown")


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


def test_identity_new_profile_plus_deck_ready_and_settle_confirms_a_pass(monkeypatch):
    """The positive-path complement to the test above: proving the identity anchor also
    lets a REAL advance through, not just refuses to mistake a scroll for one. The identity
    band reads as a DIFFERENT profile's header (not merely 'unrecognised content'), the
    next deck is confirmed ready, and a settle re-capture reconfirms both -- wait_for_decision
    returns False (PASS)."""
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

    assert drv.wait_for_decision(timeout=5.0) is False


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


# --- observe: resolving a "top" identity verdict by OCR'ing the card header (2026-08-10) ---
# Reported incident: at scroll-top, identity_band shows Hinge's own profile-independent
# filter-chips row, not a name, so _identity_of could only ever say "top" -- and the decision
# fell through to a spurious content/shift match against a DIFFERENT woman's profile (Alina's
# pass was recorded as a scroll of her own card, when the deck had actually already advanced
# to jessica). identity_top_name_band + _identity_of's "Layer 1b" block is the fix: it OCRs
# the card header itself (which DOES carry the name even at scroll-top) to resolve "top" into
# "same" or "new" directly. `_ocr_band` is stubbed throughout (never requires a real
# tesseract binary) with a fake that returns text keyed on the `psm` argument, so these tests
# also double as plumbing checks that Layer 1b really does request psm="6" for the new band.

def _top_state_drv(monkeypatch, *, stored_name="Alina", **cfg):
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
    """The positive fix: a clean read of a DIFFERENT name at scroll-top must resolve the
    otherwise-inconclusive 'top' verdict straight to 'new', without ever consulting the
    content/shift-match layers Layer 2 would otherwise fall through to."""
    drv = _top_state_drv(monkeypatch)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "jessica" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "new"


def test_identity_top_name_ocr_resolves_top_to_same_when_the_stored_name_is_read(monkeypatch):
    """Complement of the above: reading the SAME stored name resolves 'top' to 'same' instead
    of leaving the wait loop stuck on an inconclusive verdict."""
    drv = _top_state_drv(monkeypatch)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "Alina" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "same"


def test_identity_top_name_ocr_near_miss_reads_stay_same_the_false_pass_guard(monkeypatch):
    """MEASURED calibration for _NAME_MATCH_RATIO: 'Alina' vs a plausible OCR misread 'Alma'
    scores 0.67, and vs 'Aiina' scores 0.80 -- both MUST stay 'same', because a false 'new'
    here records a PASS the human never made and corrupts the taste model, while a false
    'same' only ever costs a missed wait (see _NAME_MATCH_RATIO's own module comment for the
    full asymmetry argument). Exercise both measured misreads."""
    for misread in ("Alma", "Aiina"):
        drv = _top_state_drv(monkeypatch)
        monkeypatch.setattr(drv, "_ocr_band",
                            lambda frame, rect, psm="7", _r=misread: _r if psm == "6" else None)

        state, _dist = drv._identity_of(b"frame")

        assert state == "same", f"OCR misread {misread!r} of the stored name must stay 'same'"


def test_identity_top_name_ocr_all_chrome_words_stays_top_inconclusive(monkeypatch):
    """A read that caught nothing but Hinge's own chrome (the name itself went unread -- cut
    off, misrecognised, whatever) must leave the verdict at 'top' (inconclusive), never be
    promoted to a false 'new' -- see _TOP_NAME_CHROME_WORDS's own module comment for exactly
    this failure mode."""
    drv = _top_state_drv(monkeypatch)
    monkeypatch.setattr(
        drv, "_ocr_band",
        lambda frame, rect, psm="7": "Signals Active today" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"


def test_identity_top_name_ocr_never_runs_when_the_verdict_is_not_top(monkeypatch):
    """Gated on state == 'top' specifically: on a SCROLLED frame the same crop is photo
    content and OCRs to garbage (measured), so this must never be consulted for 'new',
    'same', or 'unknown' -- only the pixel/content layers above it decide those."""
    import numpy as np
    psms_seen = []

    def fake_ocr(frame, rect, psm="7"):
        psms_seen.append(psm)
        return None

    drv = _drv(FakeAdb([b"frame"]))
    drv._identity_name = "Alina"
    drv._identity_sig = np.zeros((16, 64), dtype="int16")           # id_sig set
    drv._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    # band far from BOTH id_sig and top_sig -> pixel verdict is 'new' outright
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: np.full((16, 64), 99, dtype="int16"))
    monkeypatch.setattr(drv, "_ocr_band", fake_ocr)

    state, _dist = drv._identity_of(b"frame")

    assert state == "new"
    assert "6" not in psms_seen     # the card-header band (psm="6") was never consulted


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
    """THE regression test for the reported incident, named so that is obvious. At
    scroll-top, identity_band alone can only ever say 'top' (Hinge's own chrome renders
    there, not a name) -- and the previous profile's captured signatures can coincidentally
    content-match the NEXT profile's first frame closely enough that Layer 2 would swallow a
    genuine pass as "just a scroll" (dating first-photos are alike: centred face, light
    background -- see wait_for_decision's own Layer 2 comment on this weak cross-profile
    signal). This is EXACTLY what happened on 2026-08-10: a pass from Alina to jessica was
    logged as observe_scroll reason=shift_match.

    Layer 1b's card-header OCR is what breaks the tie: it reads "jessica" against the stored
    "Alina", resolves the 'top' verdict to 'new' BEFORE Layer 2 ever runs (Layer 2 is skipped
    outright once identity says 'new' -- see wait_for_decision's own comment on why), and
    wait_for_decision now correctly returns False (PASS) instead of looping forever as an
    unrecorded scroll. Without the fix (state staying 'top'), the scripted content match below
    WOULD have been taken as a scroll -- that's the whole point of choosing signatures 1 apart
    on a 0..255 scale, well under change_threshold=9.0.
    """
    import numpy as np
    chrome_sig = np.full((16, 64), 200, dtype="int16")
    seen_ds = np.ones((24, 24), dtype="int16") * 50
    cur_ds = np.ones((24, 24), dtype="int16") * 51   # would content-match seen_ds if Layer 2 ran

    adb = FakeAdb([b"base", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Alina"
    drv._identity_sig = None              # never revealed Alina's own sticky header this capture
    drv._identity_top_sig = chrome_sig    # the scroll-top chrome IS recognised -> pixel verdict 'top'
    drv._current_sigs = [seen_ds]
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: chrome_sig)
    monkeypatch.setattr(hinge, "_downsample", lambda frame, size=24: cur_ds)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "jessica" if psm == "6" else None)

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
    """False-PASS guard: a truncated READ this poll ('Al' for stored 'Alina', 'Kat'/'Ka' for
    stored 'Katherine') scores 0.57/0.50/0.36 against _NAME_MATCH_RATIO=0.6 -- below the bar,
    so without the prefix test this would have resolved 'top' straight to a false 'new' and
    recorded a PASS the human never made. A truncation is by definition a prefix of the name
    it came from, so the prefix test must catch every one of these."""
    for stored_name, read_name in (("Alina", "Al"), ("Katherine", "Kat"), ("Katherine", "Ka")):
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
    self._identity_name ('Al' for a profile actually named 'Alina', 'Sam' for 'Samantha') --
    not this poll's OCR read, which is the FULL correct name. Every later full-name read must
    still resolve 'same' against that truncated stored value, or the one bad capture poisons
    the whole rest of the profile with spurious 'new' verdicts."""
    for stored_name, read_name in (("Al", "Alina"), ("Sam", "Samantha")):
        drv = _top_state_drv(monkeypatch, stored_name=stored_name)
        monkeypatch.setattr(
            drv, "_ocr_band",
            lambda frame, rect, psm="7", _r=read_name: _r if psm == "6" else None)

        state, _dist = drv._identity_of(b"frame")

        assert state == "same", (
            f"full read {read_name!r} against a truncated stored name {stored_name!r} must "
            f"stay 'same' (a bad capture-time read must not poison the whole profile)")


def test_identity_top_name_ocr_genuine_different_name_still_resolves_new(monkeypatch):
    """The prefix-test fix above must not neuter the original bug fix: a GENUINELY different
    name (no prefix relationship either direction, ratio well below the bar) must still
    resolve 'top' to 'new'. Alina-vs-jessica (the actual reported incident) is already covered
    by test_identity_top_name_ocr_resolves_top_to_new_when_a_different_name_is_read above;
    this is the second MEASURED genuine-difference pair (Katherine/Michelle, ratio 0.35)."""
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
                        lambda frame, rect, psm="7": "Xy" if psm == "6" else None)

    state, _dist = drv._identity_of(b"frame")

    assert state == "top"


# --- observe: Layer 1b hole 1c -- a name-derived 'new' must reproduce before it is acted on --
# A pixel-verified 'new' is near-unambiguous (measured separation ~0 vs ~18 -- see _identity_of
# module comments). A name-derived 'new' (Layer 1b's OCR of the card header) is a noisier
# signal being given identical trust by the plain `confirm_identity_state != "same"` check
# alone. wait_for_decision now additionally requires, ONLY when the first frame's 'new' came
# from Layer 1b (self._identity_top_name_verdict == "new"), that the independent confirm-frame
# read 0.5s later ALSO says 'new' -- a mere 'top' is no longer good enough to corroborate it.

def test_name_derived_new_reproduced_on_confirm_frame_produces_a_pass(monkeypatch):
    """Positive case: the confirm frame -- captured independently, 0.5s later -- ALSO resolves
    'new' via the same Layer 1b OCR path. That is the cheapest available corroboration and
    wait_for_decision correctly reports PASS (the negative case, not reproduced, is the test
    right below)."""
    import numpy as np
    chrome_sig = np.full((16, 64), 200, dtype="int16")
    adb = FakeAdb([b"base", b"new", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Alina"
    drv._identity_sig = None              # never revealed Alina's own sticky header this capture
    drv._identity_top_sig = chrome_sig    # the scroll-top chrome IS recognised -> pixel verdict 'top'
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: chrome_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    # Constant regardless of which frame/call this is -- a genuinely reproducing read.
    monkeypatch.setattr(drv, "_ocr_band",
                        lambda frame, rect, psm="7": "jessica" if psm == "6" else None)

    assert drv.wait_for_decision(timeout=5.0) is False


def test_name_derived_new_not_reproduced_on_confirm_frame_does_not_produce_a_pass(monkeypatch):
    """Negative case: the confirm-frame OCR read fails to reproduce 'new' (comes back
    inconclusive 'top' instead, e.g. a one-off garbled tesseract read) on every poll.
    wait_for_decision must never resolve this to a PASS -- it keeps watching and times out
    instead, exactly like any other unresolved wait. (A SYSTEMATIC misread -- e.g. a crop
    boundary that is consistently off, not a one-off garble -- WOULD reproduce identically and
    this check would not catch it; that failure mode is what the prefix rule in
    _name_token_matches covers instead, see the hole-1 tests above.)"""
    import numpy as np
    chrome_sig = np.full((16, 64), 200, dtype="int16")
    adb = FakeAdb([b"base", b"new", b"new"], advance_on_screencap=True)
    drv = _drv(adb)
    drv._identity_name = "Alina"
    drv._identity_sig = None
    drv._identity_top_sig = chrome_sig
    monkeypatch.setattr(hinge, "_band", lambda frame, rect: chrome_sig)
    monkeypatch.setattr(hinge, "_split_diff", lambda a, b: (0.0, 0.0) if a == b else (50.0, 50.0))
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
    calls = {"n": 0}

    def flaky_ocr(frame, rect, psm="7"):
        if psm != "6":
            return None
        calls["n"] += 1
        # Odd calls are each poll's FIRST _identity_of (the 'cur' check); even calls are the
        # settle/confirm re-check 0.5s later. Naming the new profile on odd calls only -- never
        # reproducing it on the even ones -- is exactly a misread that does not survive a
        # second independent read, on every single poll (not just the first), so this can never
        # accidentally pass by outlasting a finite scripted sequence.
        return "jessica" if calls["n"] % 2 == 1 else None

    monkeypatch.setattr(drv, "_ocr_band", flaky_ocr)

    assert drv.wait_for_decision(timeout=0.2) is None


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
        lambda *a, **k: (calls.append(1), SimpleNamespace(stdout=b"Alina", returncode=0))[1])
    frame = _png(value=10)
    rect = (0.0, 0.0, 1.0, 1.0)

    first = drv._ocr_band(frame, rect, psm="6")
    second = drv._ocr_band(frame, rect, psm="6")

    assert first == "Alina"
    assert second == "Alina"
    assert len(calls) == 1, f"expected exactly 1 tesseract invocation, got {len(calls)}"


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
        lambda *a, **k: (calls.append(1), SimpleNamespace(stdout=b"Alina", returncode=0))[1])
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
        lambda *a, **k: (calls.append(1), SimpleNamespace(stdout=b"Alina", returncode=0))[1])
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


def test_touch_watcher_health_exception_falls_back_to_pass_and_warns_once(monkeypatch, capsys):
    """event_count == 0 for the whole run is proof the STREAM itself isn't delivering
    anything on this device (wrong node selected, a permissions change mid-run, ...) -- not
    "the human genuinely never touched the screen". A broken sensor must never veto a real,
    identity-and-deck-ready-proven advance. Falls back to PASS, and warns exactly once even
    across a SECOND, independent advance on the same driver -- the warned flag lives on the
    driver instance, not the call."""
    watcher = _FakeWatcher([], alive=True, event_count=0)
    drv, _adb = _proven_advance_drv(monkeypatch, watcher, frames=(b"base1", b"new1"))

    assert drv.wait_for_decision(timeout=5.0) is False

    drv._adb = FakeAdb([b"base2", b"new2"], advance_on_screencap=True)
    drv._touch = drv._adb
    assert drv.wait_for_decision(timeout=5.0) is False

    out = capsys.readouterr().out
    assert out.count("touch watcher has seen no events this run") == 1


def test_open_session_raises_driver_closed_naming_config_key_when_watcher_cant_start(monkeypatch):
    """observe_touch_watch=True (opt-in; it ships False because the target device's
    platform withholds the stream) must fail LOUDLY -- the same "explicit
    operator decision, never a silent downgrade" contract as touch_backend -- when the
    device's touch event stream can't be attached, rather than silently narrowing observe
    mode's PASS proof back to identity+deck-ready alone. The raised DriverClosed must name
    the config key an operator can set to accept that narrower proof on purpose."""
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

    monkeypatch.setattr(hinge, "Adb", lambda *a, **k: _OpenAdb())
    monkeypatch.setattr(hinge, "TouchWatcher", _DeadWatcher)
    cfg = type("C", (), {"apps": {"hinge": {"serial": "pixel", "touch_backend": "adb",
                                            "observe_touch_watch": True}}})
    drv = HingeDriver(cfg)

    with pytest.raises(DriverClosed) as exc:
        drv.open_session()
    assert "observe_touch_watch" in str(exc.value)


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
    drv._session_top_done = True         # isolate the per-capture unwind from the once-per-session one

    drv.current_profile()

    assert adb.scrolls == 2              # capture read-scrolled twice (3 distinct frames)
    assert adb.swipes == 2               # _scroll_to_top swiped all the way back up before returning


def test_next_profile_does_not_scroll_to_top_itself():
    # auto path: like() already calls _scroll_to_top() before acting, so next_profile() must
    # NOT also scroll back up -- that would be a redundant extra scroll (bug 2 fix note).
    # The ONLY unwind on this path is _ensure_session_top's one-shot pass, which is a different
    # thing entirely: it runs once per session, to establish where the card actually is (see
    # its docstring), not once per profile.
    adb = FakeAdb([b"a", b"b", b"c"])
    drv = _drv(adb, scroll_captures=3)

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

def test_capture_abandons_the_read_as_soon_as_stop_is_requested():
    # dwell_s=0 keeps the dwell instant while still exercising the in-dwell stop check (the
    # dwell is where a Stop most often lands in production: 1.1s humanized, x11 per profile).
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


def test_stop_during_capture_leaves_the_scroll_ledger_intact_for_a_later_unwind():
    # The ledger is the driver's only record of how far down the card actually is. An abandoned
    # read must not clear it, or a later _scroll_to_top (from a restarted session, like(), or
    # _locate_target_heart's fallback) would think it was already at the top and tap the wrong item.
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
        (hinge.HINGE_SPEC.templates["like"], heart_xy),
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
def test_like_types_the_original_opener_when_targeting_reaches_the_item():
    """Targeting lands on the exact item the opener was written about (index 0, the fast path),
    so the ORIGINAL opener text is what reaches the phone. Nothing re-asks, nothing rewrites."""
    adb = FakeAdb([_action_frame()])

    _drv(adb).like(opener="original opener about the first photo", item_index=0)

    assert adb.texts == ["original opener about the first photo"]


def test_like_stops_and_sends_nothing_when_targeting_cannot_reach_the_item():
    """THE CORE REGRESSION TEST, inverted from what it used to be. It used to assert that a
    targeting miss called an anchored_opener callback and typed its replacement under whatever
    item the fallback tap had landed on. That IS the substitution the owner rule forbids -- the
    like still went to an item the model never chose, and only the wording was made to agree with
    it afterwards. Now: no tap, no text, no Send, and the run stops."""
    adb = FakeAdb([_action_frame()])
    drv = _drv(adb)
    drv._current_sigs = [object(), object()]        # item 5 is outside the capture

    with pytest.raises(hinge.HingeTargetingError):
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

    with pytest.raises(hinge.HingeTargetingError, match="no item index was supplied at all"):
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


def test_wait_for_decision_resumes_after_the_screen_comes_back(monkeypatch):
    """Asleep, then the owner wakes it and swipes -> the decision is still detected."""
    adb = FakeAdb([_png(0), _noisy_png(lo=10, hi=60), _noisy_png(lo=150, hi=250)],
                  advance_on_screencap=True)
    drv = _drv(adb)
    # Deck-ready evidence for the final frame -- see test_pass_detected_on_card_advance's
    # comment: these noise frames carry no real glyph for _observe_deck_ready to match.
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: True)
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
    drv._identity_name = "jessica"
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
        lambda frame, rect, psm="7": None if psm == "6" else "Signals ( Agev ) Height v")

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
    monkeypatch.setattr(drv, "_ocr_band", lambda frame, rect: "Jessica")   # case-insensitive

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
    monkeypatch.setattr(drv, "_observe_gesture_verdict", lambda f: "no_data")

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
