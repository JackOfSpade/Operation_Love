"""Bumble observe-mode decision detection — offline, with a fake Playwright page.

No browser: a FakePage scripts what window.__oplove_decision reads back, so we
exercise wait_for_decision()'s polling/return logic (the DOM click listener is
exercised live by tools/bumble_inspect.py).
"""
from contextlib import contextmanager
import tempfile

import operation_love.drivers.bumble as bumble
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.bumble import BumbleDriver


class _Cfg:
    apps = {"bumble": {}}


class FakePage:
    def __init__(self, reads, empty=False):
        self.reads = list(reads)
        self.empty = empty
        self.installed = False
        self.cleared = 0

    def evaluate(self, script, arg=None):
        if "__oplove_obs" in script:          # the install snippet
            self.installed = True
            # like/pass/superlike selectors passed through (superswipe -> 'like')
            assert arg == [d.selectors["like"], d.selectors["pass"], d.selectors["superlike"]]
            return None
        if "return v" not in script:          # the pre-poll clear -> no-op, doesn't consume a read
            self.cleared += 1
            return None
        return self.reads.pop(0) if self.reads else None   # the read snippet

    def query_selector(self, sel):            # used by out_of_profiles()
        return object() if self.empty else None


class FakeClosedPage(FakePage):
    def evaluate(self, script, arg=None):
        raise RuntimeError("Target page, context or browser has been closed")


class FakePhotoElement:
    def __init__(self, name, box, loaded_after=1):
        self.name = name
        self._box = box
        self.loaded_after = loaded_after
        self.load_checks = 0

    def bounding_box(self):
        return self._box

    def evaluate(self, script):
        assert script == bumble._PHOTO_LOADED_JS
        self.load_checks += 1
        return self.load_checks >= self.loaded_after

    def screenshot(self):
        return self.name.encode()


class FakeClosedPhotoElement(FakePhotoElement):
    def __init__(self, close_at):
        super().__init__("closed", {"x": 420, "y": 170, "width": 500, "height": 680})
        self.close_at = close_at

    def bounding_box(self):
        if self.close_at == "box":
            raise RuntimeError("Target page, context or browser has been closed")
        return super().bounding_box()

    def screenshot(self):
        if self.close_at == "screenshot":
            raise RuntimeError("Target page, context or browser has been closed")
        return super().screenshot()


class FakePhotoPage:
    viewport_size = {"width": 1280, "height": 900}

    def __init__(self, elements, advance=False):
        if elements and isinstance(elements[0], list):
            self.frames = elements
        else:
            self.frames = [elements]
        self.i = 0
        self.advance = advance
        self.mouse = self

    def query_selector_all(self, sel):
        assert sel == d.selectors["photo"]
        return self.frames[self.i]

    def click(self, *_):
        if not self.advance:
            raise RuntimeError("no album")
        if self.i < len(self.frames) - 1:
            self.i += 1


class FakeClosedAlbumAdvancePage(FakePhotoPage):
    def click(self, *_):
        raise RuntimeError("Target page, context or browser has been closed")


class FakeClosedActionPage:
    def query_selector(self, *_):
        raise RuntimeError("Browser has been closed")

    def click(self, *_):
        raise RuntimeError("Browser has been closed")


class _RecordingMouse:
    def __init__(self):
        self.moves = []
        self.clicks = []

    def move(self, x, y):
        self.moves.append((x, y))

    def click(self, x, y):
        self.clicks.append((x, y))


class FakeActionElement:
    def __init__(self, box):
        self._box = box

    def bounding_box(self):
        return self._box


class FakeActionPage:
    """Exercises the human-cursor click path: query_selector -> element with a box,
    plus a real mouse exposing move + click."""
    viewport_size = {"width": 1280, "height": 900}

    def __init__(self, box):
        self._box = box
        self.mouse = _RecordingMouse()
        self.plain_clicks = []

    def query_selector(self, sel):
        return FakeActionElement(self._box)

    def click(self, sel):
        self.plain_clicks.append(sel)


class FakeNoElementPage:
    """No element found -> human-click must fall back to a plain page.click."""
    def __init__(self):
        self.plain_clicks = []

    def query_selector(self, sel):
        return None

    def click(self, sel):
        self.plain_clicks.append(sel)


class FakeStartupPage:
    def __init__(self):
        self.default_timeout = None
        self.goto_calls = []
        self.clicks = []

    def set_default_timeout(self, timeout):
        self.default_timeout = timeout

    def goto(self, url, wait_until=None):
        self.goto_calls.append((url, wait_until))

    def click(self, selector, **kwargs):
        self.clicks.append((selector, kwargs))
        raise RuntimeError("not found")


class FakeContext:
    def __init__(self, page):
        self.pages = [page]
        self.closed = False

    def new_page(self):
        return self.pages[0]

    def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, ctx):
        self.ctx = ctx
        self.launch_kwargs = None

    def launch_persistent_context(self, **kwargs):
        self.launch_kwargs = kwargs
        return self.ctx


class FakePlaywright:
    def __init__(self, chromium):
        self.chromium = chromium
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakePlaywrightManager:
    def __init__(self, pw):
        self.pw = pw

    def start(self):
        return self.pw


class FakeLoadingPhotoPage(FakePhotoPage):
    def query_selector_all(self, sel):
        assert sel == d.selectors["photo"]
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        if self.i < len(self.frames) - 1:
            self.i += 1
        return frame


def _driver(page):
    drv = BumbleDriver(_Cfg())
    drv.page = page
    return drv


@contextmanager
def _fast_capture_waits(timeout=0.2, poll=0.0, settle=0.0):
    old = (bumble._PHOTO_READY_TIMEOUT_S, bumble._PHOTO_READY_POLL_S, bumble._PHOTO_READY_SETTLE_S)
    bumble._PHOTO_READY_TIMEOUT_S = timeout
    bumble._PHOTO_READY_POLL_S = poll
    bumble._PHOTO_READY_SETTLE_S = settle
    try:
        yield
    finally:
        (bumble._PHOTO_READY_TIMEOUT_S, bumble._PHOTO_READY_POLL_S,
         bumble._PHOTO_READY_SETTLE_S) = old


d = BumbleDriver(_Cfg())   # module-level for the selector assertion inside FakePage


def test_like_detected():
    drv = _driver(FakePage(reads=["like"]))
    assert drv.wait_for_decision(timeout=5) is True
    assert drv.page.installed


def test_pass_detected():
    drv = _driver(FakePage(reads=["pass"]))
    assert drv.wait_for_decision(timeout=5) is False


def test_none_then_like():
    drv = _driver(FakePage(reads=[None, None, "like"]))
    assert drv.wait_for_decision(timeout=5) is True


def test_no_timeout_waits_until_like(monkeypatch):
    monkeypatch.setattr(bumble.time, "sleep", lambda *_: None)
    drv = _driver(FakePage(reads=[None, None, "like"]))
    assert drv.wait_for_decision(timeout=None) is True


def test_wait_clears_stale_decision_before_polling():
    # A swipe captured during the previous card's embed (overlays off) must be
    # discarded before we wait on the next card, not mis-attributed to it.
    drv = _driver(FakePage(reads=["like"]))
    assert drv.wait_for_decision(timeout=5) is True
    assert drv.page.cleared == 1            # the pre-poll clear ran exactly once


def test_deck_empty_returns_none():
    drv = _driver(FakePage(reads=[None], empty=True))
    assert drv.wait_for_decision(timeout=5) is None


def test_timeout_returns_none():
    drv = _driver(FakePage(reads=[]))            # never any decision, deck not empty
    assert drv.wait_for_decision(timeout=0.2) is None


def test_browser_close_raises_driver_closed():
    drv = _driver(FakeClosedPage(reads=[]))
    try:
        drv.wait_for_decision(timeout=5)
    except DriverClosed:
        pass
    else:
        raise AssertionError("expected DriverClosed")


def test_capture_photos_excludes_sidebar_and_small_images():
    page = FakePhotoPage([
        FakePhotoElement("sidebar_avatar", {"x": 58, "y": 438, "width": 72, "height": 72}),
        FakePhotoElement("left_sidebar_large", {"x": 60, "y": 180, "width": 250, "height": 250}),
        FakePhotoElement("spotify_thumbnail", {"x": 870, "y": 520, "width": 44, "height": 44}),
        FakePhotoElement("profile_photo", {"x": 420, "y": 170, "width": 500, "height": 680}),
    ])
    drv = _driver(page)

    assert drv._capture_photos() == [b"profile_photo"]


def test_capture_photos_collects_distinct_album_steps():
    frames = [
        [FakePhotoElement("profile_1", {"x": 420, "y": 170, "width": 500, "height": 680})],
        [FakePhotoElement("profile_2", {"x": 420, "y": 170, "width": 500, "height": 680})],
        [FakePhotoElement("profile_1", {"x": 420, "y": 170, "width": 500, "height": 680})],
    ]
    page = FakePhotoPage(frames, advance=True)
    drv = _driver(page)

    assert drv._capture_photos() == [b"profile_1", b"profile_2"]


def test_capture_photos_waits_for_album_elements_before_screenshotting():
    photo = FakePhotoElement("loaded_profile", {"x": 420, "y": 170, "width": 500, "height": 680})
    page = FakeLoadingPhotoPage([[], [], [photo]])
    drv = _driver(page)
    with _fast_capture_waits():
        assert drv._capture_photos() == [b"loaded_profile"]


def test_capture_photos_waits_for_image_load_before_screenshotting():
    photo = FakePhotoElement(
        "loaded_late", {"x": 420, "y": 170, "width": 500, "height": 680}, loaded_after=3
    )
    drv = _driver(FakePhotoPage([photo]))
    with _fast_capture_waits():
        assert drv._capture_photos() == [b"loaded_late"]
    assert photo.load_checks >= 3


def test_capture_photos_waits_for_all_album_images_before_screenshotting():
    ready = FakePhotoElement("ready", {"x": 420, "y": 170, "width": 500, "height": 680})
    late = FakePhotoElement(
        "late", {"x": 420, "y": 170, "width": 500, "height": 680}, loaded_after=3
    )
    drv = _driver(FakePhotoPage([ready, late]))
    with _fast_capture_waits():
        assert drv._capture_photos() == [b"ready", b"late"]
    assert late.load_checks >= 3


def test_capture_photos_times_out_when_image_never_reports_loaded():
    photo = FakePhotoElement(
        "eventual_capture", {"x": 420, "y": 170, "width": 500, "height": 680}, loaded_after=1_000_000
    )
    drv = _driver(FakePhotoPage([photo]))
    with _fast_capture_waits(timeout=0.03, poll=0.01):
        assert drv._capture_photos() == [b"eventual_capture"]
    assert 1 < photo.load_checks < 1_000_000


def test_capture_photos_browser_close_raises_driver_closed():
    drv = _driver(FakePhotoPage([FakeClosedPhotoElement("screenshot")]))

    try:
        drv._capture_photos()
    except DriverClosed:
        pass
    else:
        raise AssertionError("expected DriverClosed")


def test_advance_photo_album_browser_close_raises_driver_closed():
    photo = FakePhotoElement("profile", {"x": 420, "y": 170, "width": 500, "height": 680})
    drv = _driver(FakeClosedAlbumAdvancePage([photo]))

    try:
        drv._advance_photo_album()
    except DriverClosed:
        pass
    else:
        raise AssertionError("expected DriverClosed")


def test_like_browser_close_raises_driver_closed():
    drv = _driver(FakeClosedActionPage())

    try:
        drv.like()
    except DriverClosed:
        pass
    else:
        raise AssertionError("expected DriverClosed")


def test_like_uses_human_cursor_path_inside_button():
    box = {"x": 1000, "y": 820, "width": 56, "height": 56}
    drv = _driver(FakeActionPage(box))
    drv.like()
    page = drv.page
    assert page.mouse.clicks, "expected a real mouse click (human path)"
    cx, cy = page.mouse.clicks[-1]
    assert box["x"] <= cx <= box["x"] + box["width"]      # landed inside the button
    assert box["y"] <= cy <= box["y"] + box["height"]
    assert page.mouse.moves, "expected curved cursor movement, not a teleport"
    assert not page.plain_clicks                          # used the mouse path, not the fallback


def test_dislike_falls_back_to_plain_click_without_box():
    drv = _driver(FakeNoElementPage())
    drv.dislike()
    assert drv.page.plain_clicks == [drv.selectors["pass"]]


def test_like_falls_back_to_plain_click_when_box_center_off_screen():
    # Valid box, but its center is far outside the ~1280x900 viewport -> raw-coord
    # clicking there isn't safe, so we must use page.click (its own actionability +
    # scroll-into-view), not a mouse.click at the off-screen point.
    box = {"x": 4000, "y": 4000, "width": 56, "height": 56}
    # FakeActionElement has no scroll_into_view_if_needed, so the box stays off-screen.
    assert not hasattr(FakeActionElement(box), "scroll_into_view_if_needed")
    drv = _driver(FakeActionPage(box))
    drv.like()
    page = drv.page
    assert page.plain_clicks == [drv.selectors["like"]]   # plain-click fallback used
    assert not page.mouse.clicks                          # no raw mouse click off-screen


def test_open_session_denies_native_permission_prompts():
    page = FakeStartupPage()
    ctx = FakeContext(page)
    chromium = FakeChromium(ctx)
    pw = FakePlaywright(chromium)

    def fake_sync_playwright():
        return FakePlaywrightManager(pw)

    old_import = BumbleDriver._import_playwright
    BumbleDriver._import_playwright = staticmethod(lambda: (fake_sync_playwright, "fake"))
    try:
        with tempfile.TemporaryDirectory() as user_dir:
            cfg = type("Cfg", (), {"apps": {"bumble": {"user_data_dir": user_dir}}})()
            drv = BumbleDriver(cfg)
            try:
                drv.open_session()
                args = chromium.launch_kwargs["args"]
                assert "--disable-blink-features=AutomationControlled" in args
                assert "--deny-permission-prompts" in args
                assert page.goto_calls == [(drv.url, "domcontentloaded")]
            finally:
                drv.close()
    finally:
        BumbleDriver._import_playwright = old_import


def test_dismiss_startup_interstitials_is_nonfatal_when_selectors_fail():
    page = FakeStartupPage()
    drv = _driver(page)
    drv._dismiss_startup_interstitials()
    assert len(page.clicks) == len(bumble._STARTUP_INTERSTITIALS)


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
