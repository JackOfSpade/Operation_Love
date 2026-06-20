"""Bumble observe-mode decision detection — offline, with a fake Playwright page.

No browser: a FakePage scripts what window.__oplove_decision reads back, so we
exercise wait_for_decision()'s polling/return logic (the DOM click listener is
exercised live by tools/bumble_inspect.py).
"""
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.bumble import BumbleDriver


class _Cfg:
    apps = {"bumble": {}}


class FakePage:
    def __init__(self, reads, empty=False):
        self.reads = list(reads)
        self.empty = empty
        self.installed = False

    def evaluate(self, script, arg=None):
        if "__oplove_obs" in script:          # the install snippet
            self.installed = True
            # like/pass/superlike selectors passed through (superswipe -> 'like')
            assert arg == [d.selectors["like"], d.selectors["pass"], d.selectors["superlike"]]
            return None
        return self.reads.pop(0) if self.reads else None   # the read snippet

    def query_selector(self, sel):            # used by out_of_profiles()
        return object() if self.empty else None


class FakeClosedPage(FakePage):
    def evaluate(self, script, arg=None):
        raise RuntimeError("Target page, context or browser has been closed")


class FakePhotoElement:
    def __init__(self, name, box):
        self.name = name
        self._box = box

    def bounding_box(self):
        return self._box

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


class FakeClosedActionPage:
    def click(self, *_):
        raise RuntimeError("Browser has been closed")


def _driver(page):
    drv = BumbleDriver(_Cfg())
    drv.page = page
    return drv


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


def test_capture_photos_browser_close_raises_driver_closed():
    drv = _driver(FakePhotoPage([FakeClosedPhotoElement("screenshot")]))

    try:
        drv._capture_photos()
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
