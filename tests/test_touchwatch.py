"""touchwatch.py — offline tests for the read-only touch-stream observer.

No device: canned `getevent -lt` / `getevent -p` transcripts are fed straight through
the parser (TouchWatcher._feed_line) or through select_touch_device(), and
subprocess.run/Popen are monkeypatched with fakes for the two tests that exercise
start()/close() end to end. Coordinates in the -lt fixtures are chosen so the hex
decode is easy to verify by eye (0x1f4 = 500, 0x320 = 800, 0x258 = 600, 0x12c = 300).

Tests #9 and #10 below are exactly the ones numbered in the redesign spec §4; the rest
are supplementary coverage for the rest of this new module's public surface (start/close
lifecycle, the TouchWatchUnavailable failure modes, the gesture deque cap, and the
"must never raise into the reader thread" contract).
"""
import subprocess

import pytest

from operation_love.drivers import touchwatch

# Liveness bound, not a performance bound: it exists only so a genuine hang fails this test
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`), which measured a
# ~15x slowdown (0.33s idle vs 5.06s under load) on tests/test_concurrency.py's positive
# liveness waits of the same shape. Nothing about the property under test (did the reader
# thread reach EOF and exit?) depends on the exact number, so widening it loses nothing.
_LIVENESS_TIMEOUT_S = 15.0


# --- fixtures --------------------------------------------------------------------------

# One tap (down=up=(500, 800)) and one drag ((500, 800) -> (500, 300) via an intermediate
# (500, 600)), on event3, in the exact `getevent -lt` line shape from the design spec:
# `[   12345.678901] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    000001f4`.
_LT_TRANSCRIPT = """\
[   12345.001000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   00000001
[   12345.001000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    000001f4
[   12345.001000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000320
[   12345.001000] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN
[   12345.001000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   12345.041000] /dev/input/event3: EV_KEY       BTN_TOUCH            UP
[   12345.041000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff
[   12345.041000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   12346.001000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   00000002
[   12346.001000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    000001f4
[   12346.001000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000320
[   12346.001000] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN
[   12346.001000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   12346.051000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000258
[   12346.051000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   12346.101000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    0000012c
[   12346.101000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   12346.151000] /dev/input/event3: EV_KEY       BTN_TOUCH            UP
[   12346.151000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff
[   12346.151000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
"""

# Pixel 7a shape (ops/ANTI-BOT-RESEARCH.md, 2026-08-10 §1): goodix_ts0 at event3,
# INPUT_PROP_DIRECT, ABS_MT_POSITION_X max 1079 / ABS_MT_POSITION_Y max 2399, BTN_TOUCH
# (KEY 014a) declared. Prepended with our OWN og_touch_* UHID digitizer (uhid.py) at
# event1 -- SAME axes, printed FIRST -- so picking event3 anyway proves the exclusion is
# load-bearing, not just an accident of ordering; and a gpio-keys block at event2 with no
# ABS axes at all, so a device with zero qualifying axes is skipped too.
_GETEVENT_P_TRANSCRIPT = """\
add device 1: /dev/input/event1
  bus:      0018
  vendor    0001
  product   0001
  version   0100
  name:     "og_touch_a1b2c3d4"
  location: "builtin"
  id:       "0000"
  version:  1.0.1
  events:
    KEY (0001): 014a
    ABS (0003): 0030  : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
                0035  : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                0036  : value 0, min 0, max 2399, fuzz 0, flat 0, resolution 0
                0039  : value 0, min 0, max 65535, fuzz 0, flat 0, resolution 0
  input props:
    INPUT_PROP_DIRECT
add device 2: /dev/input/event2
  bus:      0019
  vendor    0000
  product   0000
  version   0000
  name:     "gpio-keys"
  location: ""
  id:       "0000"
  version:  1.0.1
  events:
    KEY (0001): 0072  0073  009e
  input props:
    <none>
add device 3: /dev/input/event3
  bus:      0018
  vendor    0000
  product   0000
  version   0000
  name:     "goodix_ts0"
  location: ""
  id:       "0000"
  version:  1.0.1
  events:
    KEY (0001): 014a
    ABS (0003): 0030  : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
                0035  : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                0036  : value 0, min 0, max 2399, fuzz 0, flat 0, resolution 0
                0039  : value 0, min 0, max 65535, fuzz 0, flat 0, resolution 0
                003a  : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
  input props:
    INPUT_PROP_DIRECT
"""

# Same shape, but with the touchscreen block removed entirely -- nothing anywhere
# declares both ABS_MT_POSITION_X and ABS_MT_POSITION_Y.
_NO_TOUCH_TRANSCRIPT = """\
add device 1: /dev/input/event0
  name:     "gpio-keys"
  events:
    KEY (0001): 0072  0073  009e
  input props:
    <none>
"""


def _tap_lines(x: int, y: int, t: float, tracking_id: int) -> list[str]:
    """A minimal synthetic DOWN/UP tap block on event3, same line shape as _LT_TRANSCRIPT,
    for tests that need several distinct taps rather than one hand-written transcript."""
    xh, yh, idh = format(x, "08x"), format(y, "08x"), format(tracking_id, "08x")
    return [
        f"[   {t:.6f}] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   {idh}",
        f"[   {t:.6f}] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    {xh}",
        f"[   {t:.6f}] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    {yh}",
        f"[   {t:.6f}] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN",
        f"[   {t:.6f}] /dev/input/event3: EV_SYN       SYN_REPORT           00000000",
        f"[   {t + 0.02:.6f}] /dev/input/event3: EV_KEY       BTN_TOUCH            UP",
        f"[   {t + 0.02:.6f}] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff",
        f"[   {t + 0.02:.6f}] /dev/input/event3: EV_SYN       SYN_REPORT           00000000",
    ]


class _FakeStdout:
    """Mimics Popen.stdout well enough to drive TouchWatcher._read_loop: an iterable of
    text lines that exhausts (EOF), exactly like a real getevent process being killed or
    disconnecting -- which is what lets the reader thread's `alive = False` transition be
    exercised without a real device."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


class _FakePopen:
    """Stands in for subprocess.Popen: records the argv it was launched with and streams
    _LT_TRANSCRIPT's lines through .stdout, same as the real `adb shell getevent -lt`
    process would over its pipe."""

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.stdout = _FakeStdout(_LT_TRANSCRIPT.splitlines(keepends=True))
        self.stderr = _FakeStdout([])
        self.terminate_called = False
        self.wait_called = False

    def terminate(self):
        self.terminate_called = True

    def wait(self, timeout=None):
        self.wait_called = True
        return 0

    def kill(self):
        pass


def _fake_run_ok(argv, capture_output=True, timeout=None, check=None):
    assert check is False
    return subprocess.CompletedProcess(argv, 0, stdout=_GETEVENT_P_TRANSCRIPT.encode(), stderr=b"")


# --- #9: getevent -lt parsing -----------------------------------------------------------
def test_feed_line_parses_lt_transcript_into_tap_and_drag_gestures():
    """Spec §4 test #9: feed a canned `getevent -lt` transcript through the parser and
    assert the gestures (one tap, one drag) come out right, including hex decoding."""
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in _LT_TRANSCRIPT.splitlines():
        w._feed_line(line)

    gestures = w.gestures_since(0)
    assert len(gestures) == 2

    tap, drag = gestures
    assert tap.down == (500, 800)      # 0x1f4 -> 500, 0x320 -> 800
    assert tap.up == (500, 800)
    assert tap.travel_px == 0.0
    assert tap.is_tap(12.0)

    assert drag.down == (500, 800)
    assert drag.up == (500, 300)       # 0x12c -> 300
    assert drag.travel_px == 500.0     # farthest point reached from the down point
    assert not drag.is_tap(12.0)

    # Every line in the transcript matches the getevent -lt shape (including the plain
    # EV_SYN/tracking-id lines), so event_count -- the health signal -- counts all of them,
    # not just the ones that moved a gesture along.
    assert w.event_count == len(_LT_TRANSCRIPT.splitlines())


def test_feed_line_parses_numeric_getevent_codes_when_toolbox_ignores_l_flag():
    """Some Android builds keep ``-t`` timestamps but print numeric event/code fields."""
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in [
        "[   1.000000] /dev/input/event3: 0003 0035 000001f4",
        "[   1.000000] /dev/input/event3: 0003 0036 00000320",
        "[   1.000000] /dev/input/event3: 0001 014a 00000001",
        "[   1.050000] /dev/input/event3: 0003 0035 00000384",
        "[   1.050000] /dev/input/event3: 0001 014a 00000000",
    ]:
        w._feed_line(line)

    gestures = w.gestures_since(0)
    assert len(gestures) == 1
    assert gestures[0].down == (500, 800)
    assert gestures[0].up == (900, 800)


def test_feed_line_ignores_unmatched_and_malformed_lines_without_raising():
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    w._feed_line("")                                     # blank
    w._feed_line("this is not a getevent line at all")   # doesn't match the shape at all
    w._feed_line(
        "[   1.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    zzzzzzzz"
    )   # matches the shape, but the value isn't valid hex

    assert w.event_count == 1              # only the shape-matching line counts
    assert w.gestures_since(0) == []


def test_ends_gesture_via_tracking_id_reset_when_btn_touch_up_is_absent():
    # Devices that never emit BTN_TOUCH UP signal release purely via the multitouch
    # tracking ID resetting to -1 (ffffffff). BTN_TOUCH DOWN still opens the gesture
    # (goodix_ts0 does declare BTN_TOUCH -- see module docstring); this exercises the
    # secondary close path in isolation.
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in [
        "[   1.000000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   00000001",
        "[   1.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    000001f4",
        "[   1.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000320",
        "[   1.000000] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN",
        "[   1.020000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff",
    ]:
        w._feed_line(line)

    gestures = w.gestures_since(0)
    assert len(gestures) == 1
    assert gestures[0].down == (500, 800)
    assert gestures[0].is_tap(1.0)


def test_gesture_deque_is_capped():
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400), gesture_cap=3)
    for i in range(5):
        for line in _tap_lines(100 + i, 200, t=1.0 + i, tracking_id=i + 1):
            w._feed_line(line)

    gestures = w.gestures_since(0)
    assert len(gestures) == 3
    # oldest two taps (x=100, x=101) were evicted; the last three survive, in order
    assert [g.down[0] for g in gestures] == [102, 103, 104]


def test_gestures_since_filters_by_host_monotonic_down_time():
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in _tap_lines(500, 800, t=1.0, tracking_id=1):
        w._feed_line(line)
    cutoff = w.gestures_since(0)[0].t_up + 1000.0   # comfortably after this gesture ended
    for line in _tap_lines(600, 800, t=2.0, tracking_id=2):
        w._feed_line(line)

    assert len(w.gestures_since(0)) == 2
    assert w.gestures_since(cutoff) == []   # the first (and only pre-cutoff) tap is excluded


# --- #10: getevent -p device selection ---------------------------------------------------
def test_select_touch_device_picks_pixel7a_touchscreen_and_skips_uhid():
    """Spec §4 test #10: a canned `getevent -p` transcript (the REAL Pixel 7a shape,
    spec §1) picks /dev/input/event3 and skips an og_touch_* entry."""
    dev_path, name, x_max, y_max = touchwatch.select_touch_device(_GETEVENT_P_TRANSCRIPT)

    assert dev_path == "/dev/input/event3"
    assert name == "goodix_ts0"
    assert x_max == 1079
    assert y_max == 2399


def test_select_touch_device_uhid_exclusion_is_load_bearing():
    # Without the og_touch_ exclusion, the SAME transcript would pick the UHID device
    # instead -- it's printed first and declares the identical axes. Proves the skip
    # actually changes the outcome rather than being coincidentally irrelevant.
    dev_path, name, _, _ = touchwatch.select_touch_device(
        _GETEVENT_P_TRANSCRIPT, exclude_name_prefixes=(),
    )
    assert dev_path == "/dev/input/event1"
    assert name == "og_touch_a1b2c3d4"


def test_select_touch_device_raises_when_nothing_qualifies():
    with pytest.raises(touchwatch.TouchWatchUnavailable):
        touchwatch.select_touch_device(_NO_TOUCH_TRANSCRIPT)


# --- Gesture.is_tap --------------------------------------------------------------------
def test_gesture_is_tap_boundary_is_inclusive():
    g = touchwatch.Gesture(t_down=0.0, t_up=0.1, down=(0, 0), up=(10, 0), travel_px=10.0)
    assert g.is_tap(10.0)          # exactly at the slop -> still a tap
    assert not g.is_tap(9.999)     # a hair under -> a drag


# --- start()/close() lifecycle, via a fake Popen (no real device) -----------------------
def test_start_streams_getevent_and_close_is_safe_after_eof(monkeypatch):
    monkeypatch.setattr(touchwatch.subprocess, "run", _fake_run_ok)
    monkeypatch.setattr(touchwatch.subprocess, "Popen", _FakePopen)

    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    w.start()
    w._thread.join(timeout=_LIVENESS_TIMEOUT_S)   # the fake stream is finite: the reader hits EOF quickly

    assert w.device_path == "/dev/input/event3"
    assert w.device_name == "goodix_ts0"
    assert w.alive is False       # EOF -> subprocess "death" -> alive flips off on its own
    assert w.event_count == len(_LT_TRANSCRIPT.splitlines())
    gestures = w.gestures_since(0)
    assert len(gestures) == 2
    assert gestures[0].is_tap(5.0)
    assert not gestures[1].is_tap(5.0)

    w.close()   # must not raise even though the process already finished on its own


def test_start_is_idempotent(monkeypatch):
    monkeypatch.setattr(touchwatch.subprocess, "run", _fake_run_ok)
    monkeypatch.setattr(touchwatch.subprocess, "Popen", _FakePopen)

    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    w.start()
    first_proc = w._proc
    w.start()   # a second call must not launch a second getevent process
    assert w._proc is first_proc
    w.close()


def test_close_without_start_does_not_raise():
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    w.close()
    assert w.alive is False


def test_start_raises_touchwatch_unavailable_when_probe_process_cannot_start(monkeypatch):
    def _boom(argv, capture_output=True, timeout=None, check=None):
        assert check is False
        raise FileNotFoundError("adb not on PATH")

    monkeypatch.setattr(touchwatch.subprocess, "run", _boom)
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))

    with pytest.raises(touchwatch.TouchWatchUnavailable):
        w.start()


def test_start_raises_touchwatch_unavailable_when_probe_exits_nonzero(monkeypatch):
    def _fail_run(argv, capture_output=True, timeout=None, check=None):
        assert check is False
        return subprocess.CompletedProcess(
            argv, 1, stdout=b"", stderr=b"error: no devices/emulators found",
        )

    monkeypatch.setattr(touchwatch.subprocess, "run", _fail_run)
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))

    with pytest.raises(touchwatch.TouchWatchUnavailable):
        w.start()


def test_start_raises_touchwatch_unavailable_when_no_touch_device_present(monkeypatch):
    def _no_touch_run(argv, capture_output=True, timeout=None, check=None):
        assert check is False
        return subprocess.CompletedProcess(argv, 0, stdout=_NO_TOUCH_TRANSCRIPT.encode(), stderr=b"")

    monkeypatch.setattr(touchwatch.subprocess, "run", _no_touch_run)
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))

    with pytest.raises(touchwatch.TouchWatchUnavailable):
        w.start()


def test_start_raises_touchwatch_unavailable_when_popen_cannot_start(monkeypatch):
    def _boom_popen(argv, **kwargs):
        raise OSError("no such file or directory")

    monkeypatch.setattr(touchwatch.subprocess, "run", _fake_run_ok)
    monkeypatch.setattr(touchwatch.subprocess, "Popen", _boom_popen)
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))

    with pytest.raises(touchwatch.TouchWatchUnavailable):
        w.start()


# --- hard rule: this module is READ-ONLY -------------------------------------------------
def test_touchwatcher_exposes_no_device_write_surface():
    # This is an observer, never an actor -- it must never grow a tap/swipe/text/write
    # method that would make it look, from the driver side, like a second touch injector.
    forbidden = {"tap", "swipe", "text", "write_file", "scroll_up", "click"}
    assert not (forbidden & set(dir(touchwatch.TouchWatcher)))


# --- down-point anchoring when BTN_TOUCH DOWN precedes the first coordinate ---------------
# goodix_ts0 emits ABS_MT_POSITION_X/Y BEFORE BTN_TOUCH inside one SYN frame, which is what
# the fixtures above use. The opposite ordering is equally legal, and getting it wrong is
# SILENT: the gesture anchors on whatever coordinate happened to be left over -- the previous
# gesture's release point, or a fabricated (0, 0) for the very first touch of a session --
# so a stationary tap reads as a long drag and stops counting as a tap at all. Since a tap on
# the pass-X is the only affirmative PASS evidence hinge.py's gesture corroboration accepts,
# that turns every real pass into a resync.
_DOWN_BEFORE_COORDS = """\
[   50000.001000] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN
[   50000.001000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    000001f4
[   50000.001000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000320
[   50000.001000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   50000.041000] /dev/input/event3: EV_KEY       BTN_TOUCH            UP
[   50000.041000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
"""


def test_first_gesture_of_a_session_anchors_on_the_real_down_point_not_zero_zero():
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in _DOWN_BEFORE_COORDS.splitlines():
        w._feed_line(line)

    (g,) = w.gestures_since(0.0)
    assert g.down == (500, 800)          # NOT (0, 0)
    assert g.up == (500, 800)
    assert g.travel_px == 0.0
    assert g.is_tap(50.0)                # a stationary press is still a tap


def test_down_before_coords_does_not_anchor_on_the_previous_gestures_release_point():
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in _LT_TRANSCRIPT.splitlines():     # leaves _x/_y at the drag's release point
        w._feed_line(line)
    before = len(w.gestures_since(0.0))
    for line in _DOWN_BEFORE_COORDS.splitlines():
        w._feed_line(line)

    fresh = w.gestures_since(0.0)[before:]
    assert len(fresh) == 1
    assert fresh[0].down == (500, 800)   # this gesture's own first coordinate, not the last one's
    assert fresh[0].travel_px == 0.0


def test_an_up_with_no_coordinate_at_all_emits_nothing_and_does_not_poison_the_next_gesture():
    # A DOWN/UP pair that never reported a position corroborates nothing, so it must not be
    # emitted -- but it must also not leave the "adopt the next coordinate as my down point"
    # flag set, or the FOLLOWING gesture would silently inherit it.
    w = touchwatch.TouchWatcher("adb", "pixel", (1080, 2400))
    for line in ("[ 1.0] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN",
                 "[ 1.1] /dev/input/event3: EV_KEY       BTN_TOUCH            UP"):
        w._feed_line(line)
    assert w.gestures_since(0.0) == []

    for line in _DOWN_BEFORE_COORDS.splitlines():
        w._feed_line(line)
    (g,) = w.gestures_since(0.0)
    assert g.down == (500, 800)
