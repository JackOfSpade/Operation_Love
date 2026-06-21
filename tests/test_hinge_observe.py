"""Hinge observe-mode decision detection — offline, with a fake uiautomator2 device.

No emulator: a FakeDevice scripts a sequence of UI 'frames' (which resource-ids
are present + the visible prompt texts) that the driver polls, so we exercise
wait_for_decision()'s like/pass/none logic. The real tap signals are confirmed
live (see ops/RUNBOOK.md). Each poll cycle advances one frame, driven by the
patched time.sleep inside the driver.
"""
from operation_love.drivers import hinge
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.hinge import HingeDriver, _is_device_lost_error


class _Cfg:
    apps = {"hinge": {}}


IDS = HingeDriver(_Cfg()).ids
SEND, COMMENT, EMPTY = IDS["send_like"], IDS["comment_box"], IDS["empty"]


class FakeEl:
    def __init__(self, text):
        self._t = text

    def get_text(self):
        return self._t


class FakeSel:
    """One resource-id query against the current frame."""
    def __init__(self, rid, frame):
        self._rid = rid
        self._frame = frame

    @property
    def exists(self):
        return self._rid in self._frame.get("ids", ())

    def __iter__(self):
        return iter(FakeEl(t) for t in self._frame.get("prompts", ()))

    def click(self):
        pass


class FakeDevice:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def advance(self):
        if self.i < len(self.frames) - 1:
            self.i += 1

    def __call__(self, resourceId=None, **kw):
        return FakeSel(resourceId, self.frames[self.i])


class FakeCaptureDevice:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.swipes = 0

    def screenshot(self, format="raw"):
        assert format == "raw"
        return self.frames[self.i].get("shot", b"")

    def swipe_ext(self, *_args, **_kwargs):
        self.swipes += 1
        if self.i < len(self.frames) - 1:
            self.i += 1

    def __call__(self, resourceId=None, **kw):
        return FakeSel(resourceId, self.frames[self.i])


def _frame(prompts=(), ids=(), shot=b""):
    return {"prompts": tuple(prompts), "ids": tuple(ids), "shot": shot}


def _capture(frames, scroll_captures=5):
    class CaptureCfg:
        apps = {"hinge": {"scroll_captures": scroll_captures}}

    drv = HingeDriver(CaptureCfg())
    dev = FakeCaptureDevice(frames)
    drv.d = dev
    orig_sleep = hinge.time.sleep
    hinge.time.sleep = lambda *_: None
    try:
        return drv._capture_current(), dev
    finally:
        hinge.time.sleep = orig_sleep


def _run(frames, timeout=50.0):
    """Build a driver on a FakeDevice, advancing one frame per poll (sleep)."""
    drv = HingeDriver(_Cfg())
    dev = FakeDevice(frames)
    drv.d = dev
    orig_sleep = hinge.time.sleep
    hinge.time.sleep = lambda *_: dev.advance()
    try:
        return drv.wait_for_decision(timeout=timeout)
    finally:
        hinge.time.sleep = orig_sleep


def test_pass_detected():
    # card advances to a new profile with no like sheet -> pass
    assert _run([_frame(["A"]), _frame(["B"])]) is False


def test_like_detected():
    # like sheet opens, then closes with the deck advanced -> like sent
    frames = [_frame(["A"]), _frame(["A"]),
              _frame(["A"], ids=[SEND]),    # send-like sheet open
              _frame(["B"])]                # sent: closed + advanced
    assert _run(frames) is True


def test_like_via_comment_box():
    # the comment box id alone also signals an in-progress like
    frames = [_frame(["A"]), _frame(["A"], ids=[COMMENT]), _frame(["B"])]
    assert _run(frames) is True


def test_cancelled_like_then_pass():
    # sheet opens then closes on the SAME card (cancelled) -> ignored; a later
    # real advance is the pass
    frames = [_frame(["A"]), _frame(["A"], ids=[SEND]),
              _frame(["A"]),               # cancelled: closed, same card
              _frame(["C"])]               # now a real pass
    assert _run(frames) is False


def test_deck_empty_returns_none():
    assert _run([_frame(ids=[EMPTY])]) is None


def test_timeout_returns_none():
    # never decides, deck never empties
    assert _run([_frame(["A"])], timeout=0.2) is None


# --- device-loss -> DriverClosed (clean stop on emulator/ADB disconnect) -----

class DeviceError(Exception):
    """Stand-in named exactly like uiautomator2.exceptions.DeviceError so the
    detector's by-name match fires without importing uiautomator2."""


class _LostDevice:
    """A device that raises `exc` on every interaction (emulator/ADB gone)."""
    def __init__(self, exc):
        self._exc = exc

    def screenshot(self, format="raw"):
        raise self._exc

    def swipe_ext(self, *_args, **_kwargs):
        raise self._exc

    def __call__(self, resourceId=None, **_kw):
        raise self._exc


def _lost_driver(exc):
    drv = HingeDriver(_Cfg())
    drv.d = _LostDevice(exc)
    return drv


def _expect_driver_closed(fn):
    try:
        fn()
    except DriverClosed:
        return
    raise AssertionError("expected DriverClosed")


def test_device_lost_detector_matches_disconnects_but_not_app_bugs():
    # by exception type name (no uiautomator2 import needed to classify)
    assert _is_device_lost_error(DeviceError("co.hinge.app gone"))
    assert _is_device_lost_error(ConnectionResetError("Connection reset by peer"))
    # by message fragment on a plain error
    assert _is_device_lost_error(RuntimeError("device offline"))
    assert _is_device_lost_error(OSError("cannot connect to 127.0.0.1:7912"))
    assert _is_device_lost_error(RuntimeError("uiautomator is not running anymore"))
    # a genuine app/logic bug must NOT be mistaken for a disconnect
    assert not _is_device_lost_error(ValueError("prompt_answer text was unexpectedly None"))
    assert not _is_device_lost_error(KeyError("send_like"))


def test_out_of_profiles_disconnect_raises_driver_closed():
    drv = _lost_driver(DeviceError("device 'emulator-5554' not found"))
    _expect_driver_closed(drv.out_of_profiles)


def test_current_profile_disconnect_raises_driver_closed():
    drv = _lost_driver(ConnectionError("Connection refused"))
    _expect_driver_closed(drv.current_profile)


def test_wait_for_decision_disconnect_raises_driver_closed():
    drv = _lost_driver(DeviceError("atx-agent gateway gone"))
    _expect_driver_closed(lambda: drv.wait_for_decision(timeout=5.0))


def test_non_device_error_is_not_masked_as_driver_closed():
    # a real bug must propagate as itself so it isn't swallowed by the clean-stop path
    drv = _lost_driver(ValueError("unexpected UI state"))
    try:
        drv.out_of_profiles()
    except DriverClosed:
        raise AssertionError("masked a real error as DriverClosed")
    except ValueError:
        pass


def test_capture_current_stops_when_scroll_repeats():
    profile, dev = _capture([
        _frame(["A"], shot=b"shot-a"),
        _frame(["B"], shot=b"shot-b"),
        _frame(["B"], shot=b"shot-b"),
        _frame(["C"], shot=b"shot-c"),
    ])

    assert profile.photos == [b"shot-a", b"shot-b"]
    assert profile.prompts == [("", "A"), ("", "B")]
    assert dev.swipes == 2


def test_capture_current_respects_scroll_capture_limit():
    profile, _ = _capture([
        _frame(["A"], shot=b"shot-a"),
        _frame(["B"], shot=b"shot-b"),
        _frame(["C"], shot=b"shot-c"),
    ], scroll_captures=2)

    assert profile.photos == [b"shot-a", b"shot-b"]
    assert profile.prompts == [("", "A"), ("", "B")]


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
