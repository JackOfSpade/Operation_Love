"""Foreground ownership is rechecked at Android's final device-input boundary.

These tests call the private transport helpers directly on purpose: calibration and diagnostic
tools use the same helpers, so a public-driver entry check alone cannot protect those paths.
"""
from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from operation_love.drivers import hinge as hinge_module
from operation_love.drivers.adb import parse_foreground_package
from operation_love.drivers.android.bumble import BUMBLE_SPEC, BumbleAndroidDriver
from operation_love.drivers.base import DeckBlockedError, DriverClosed
from operation_love.drivers.hinge import AndroidDriver, HINGE_SPEC, HingeActionError


_DriverT = TypeVar("_DriverT", bound=AndroidDriver)


class _ForegroundAdb:
    def __init__(self, packages: list[str | None]):
        self.packages = packages
        self.foreground_calls = 0
        self.screencaps = 0
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int]] = []
        self.scrolls = 0
        self.texts: list[str] = []

    def screen_size(self):
        return (1080, 2400)

    def foreground_package(self):
        index = min(self.foreground_calls, len(self.packages) - 1)
        self.foreground_calls += 1
        return self.packages[index]

    def screencap(self):
        self.screencaps += 1
        return b"nonblank-frame"

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, **_kwargs):
        self.swipes.append((x1, y1, x2, y2))

    def scroll_up(self, *_args, **_kwargs):
        self.scrolls += 1

    def text(self, value):
        self.texts.append(value)


def _config(spec):
    class Cfg:
        apps = {spec.app: {"halt_on_error": False}}

    return Cfg()


def _configured_driver(driver: _DriverT, adb: _ForegroundAdb) -> _DriverT:
    """Attach one fake transport to both input paths used by Android drivers."""
    driver._adb = adb
    driver._touch = adb
    return driver


def _driver(adb: _ForegroundAdb, spec=HINGE_SPEC) -> AndroidDriver:
    return _configured_driver(AndroidDriver(_config(spec), spec), adb)


def test_malformed_configured_package_is_rejected_before_any_adb_command():
    class Config:
        apps = {"hinge": {"package": "co.hinge.app; id >/tmp/operation-love-injected"}}

    with pytest.raises(DriverClosed, match="apps.hinge.package.*dotted Android identifier"):
        AndroidDriver(Config(), HINGE_SPEC)


_GESTURES: tuple[
    tuple[str, Callable[[AndroidDriver], None]], ...
] = (
    ("tap", lambda driver: driver._tap(540, 1200)),
    ("swipe", lambda driver: driver._swipe(540, 1700, 540, 700)),
    ("scroll", lambda driver: driver._scroll(0.2, 0.5)),
    ("text", lambda driver: driver._text("hello")),
)


@pytest.mark.parametrize(("name", "gesture"), _GESTURES)
def test_direct_gesture_refuses_a_proven_foreign_foreground_without_transport_input(
        name, gesture):
    adb = _ForegroundAdb(["com.android.systemui"])
    driver = _driver(adb)

    with pytest.raises(HingeActionError, match="no input was sent") as caught:
        gesture(driver)

    assert isinstance(caught.value, DeckBlockedError)
    assert adb.foreground_calls == 1, f"{name} did not make one fresh ownership probe"
    assert adb.taps == []
    assert adb.swipes == []
    assert adb.scrolls == 0
    assert adb.texts == []
    assert "quick-settings shade" in (driver._blocked_reason or "")


@pytest.mark.parametrize(("name", "gesture"), _GESTURES)
def test_direct_gesture_preserves_visual_fallback_when_foreground_is_unknown(name, gesture):
    adb = _ForegroundAdb([None])
    driver = _driver(adb)

    gesture(driver)

    assert adb.foreground_calls == 1, f"{name} did not make one fresh ownership probe"
    assert len(adb.taps) + len(adb.swipes) + adb.scrolls + len(adb.texts) == 1
    assert driver._blocked_reason is None


def test_foreground_refusal_names_the_configured_android_app():
    adb = _ForegroundAdb(["com.android.systemui"])
    driver = _driver(adb, BUMBLE_SPEC)

    with pytest.raises(DeckBlockedError) as caught:
        driver._tap(540, 1200)

    assert "covering Bumble" in str(caught.value)
    assert "Hinge" not in str(caught.value)
    assert adb.taps == []


def test_device_loss_during_final_ownership_probe_stops_before_transport_input():
    class LostAdb(_ForegroundAdb):
        def foreground_package(self):
            self.foreground_calls += 1
            raise DriverClosed("ADB device was disconnected")

    adb = LostAdb([None])
    driver = _driver(adb)

    with pytest.raises(DriverClosed, match="disconnected"):
        driver._tap(540, 1200)

    assert adb.foreground_calls == 1
    assert adb.taps == []


def test_capture_rechecks_foreground_after_initial_frame_before_identity_exists():
    # The entry probe sees Hinge. System UI takes focus while the first frame is captured. The
    # old conditional skipped this check until a sticky identity signature existed, allowing a
    # read scroll on the foreign surface during the initial frames.
    adb = _ForegroundAdb([HINGE_SPEC.package, "com.android.systemui"])
    driver = _driver(adb)
    driver._item_enumeration_blocker = lambda: "test does not need item enumeration"

    assert driver._identity_sig is None
    assert driver._capture_current() is None

    assert adb.foreground_calls == 2
    assert adb.screencaps == 1
    assert adb.taps == [] and adb.swipes == [] and adb.scrolls == 0
    assert "quick-settings shade" in (driver._blocked_reason or "")


def _bumble_driver(adb: _ForegroundAdb) -> BumbleAndroidDriver:
    return _configured_driver(BumbleAndroidDriver(_config(BUMBLE_SPEC)), adb)


def test_bumble_capture_refuses_foreign_foreground_before_screencap():
    adb = _ForegroundAdb(["com.android.systemui"])
    driver = _bumble_driver(adb)

    assert driver.next_profile() is None

    assert adb.foreground_calls == 1
    assert adb.screencaps == 0
    assert "covering Bumble" in (driver._blocked_reason or "")


def test_bumble_capture_rechecks_foreground_after_screencap_before_returning_profile():
    adb = _ForegroundAdb([BUMBLE_SPEC.package, "com.android.systemui"])
    driver = _bumble_driver(adb)

    assert driver.next_profile() is None

    assert adb.foreground_calls == 2
    assert adb.screencaps == 1
    assert "covering Bumble" in (driver._blocked_reason or "")


def test_bumble_capture_returns_frame_when_foreground_remains_owned():
    adb = _ForegroundAdb([BUMBLE_SPEC.package, BUMBLE_SPEC.package])
    driver = _bumble_driver(adb)

    profile = driver.next_profile()

    assert profile is not None and profile.photos == [b"nonblank-frame"]
    assert adb.foreground_calls == 2
    assert adb.screencaps == 1


def test_android_driver_has_no_transport_input_bypass_outside_guarded_choke_points():
    """Pin the repository-wide direct-call audit so a future helper cannot bypass ownership."""
    tree = ast.parse(Path(hinge_module.__file__).read_text(encoding="utf-8"))
    direct_calls: set[tuple[str, str, str]] = set()

    class InputCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.functions: list[str] = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node):
            function = node.func
            owner = function.value if isinstance(function, ast.Attribute) else None
            if (isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name) and owner.value.id == "self"
                    and owner.attr in {"adb", "touch", "_adb", "_touch"}
                    and function.attr in {"tap", "swipe", "scroll_up", "text"}):
                direct_calls.add((self.functions[-1], owner.attr, function.attr))
            self.generic_visit(node)

    InputCallVisitor().visit(tree)

    assert direct_calls == {
        ("_tap", "touch", "tap"),
        ("_swipe", "touch", "swipe"),
        ("_scroll", "touch", "scroll_up"),
        ("_text", "adb", "text"),
    }


@pytest.mark.parametrize("current_first", [True, False])
def test_current_focused_window_wins_over_stale_focused_app_regardless_of_line_order(
        current_first):
    current = (
        "mCurrentFocus=Window{123 u0 "
        "com.android.systemui/.shade.NotificationShadeWindowView}")
    activity = "mFocusedApp=ActivityRecord{abc u0 co.hinge.app/.MainActivity t42}"
    lines = [current, activity] if current_first else [activity, current]

    assert parse_foreground_package("\n".join(lines)) == "com.android.systemui"
