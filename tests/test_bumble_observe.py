"""Bumble observe-mode decision detection — offline, with a fake Playwright page.

No browser: a FakePage scripts what window.__oplove_decision reads back, so we
exercise wait_for_decision()'s polling/return logic (the DOM click listener is
exercised live by tools/bumble_inspect.py).
"""
from contextlib import contextmanager
import tempfile

import pytest

# Imports from the new split location (operation_love/drivers/web/), not the
# operation_love.drivers.bumble compat shim: several tests below monkeypatch
# module-level globals (e.g. bumble._PHOTO_READY_TIMEOUT_S, bumble.time.sleep) by
# reassigning attributes on this exact module object, and that only affects the
# driver's real behavior if `bumble` here IS the module the driver code actually
# reads those names from at call time.
import operation_love.drivers.web.bumble_web as bumble
from operation_love import platforms
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.web.bumble_web import BumbleWebDriver as BumbleDriver
from operation_love.drivers.web.playwright_base import HumanInputUnavailable


class _Cfg:
    # `apps.bumble` is the ANDROID block now; web settings live under `apps.bumble_web`.
    apps = {"bumble_web": {}}


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
        if script == bumble._CARD_PHOTO_IDS_JS:   # _card_fingerprint's photo-identity read;
            return []                              # no card-identity signal in the base fake
        if "return v" not in script:          # the pre-poll clear -> no-op, doesn't consume a read
            self.cleared += 1
            return None
        return self.reads.pop(0) if self.reads else None   # the read snippet

    def query_selector(self, sel):            # used by out_of_profiles()
        return object() if self.empty else None


class FakeClosedPage(FakePage):
    def evaluate(self, script, arg=None):
        raise RuntimeError("Target page, context or browser has been closed")


class _FakeBioElement:
    def __init__(self, text):
        self._text = text

    def inner_text(self):
        return self._text


class FakeCardChangePage(FakePage):
    """Like FakePage (drives wait_for_decision's decision-read loop), but also backs
    _card_fingerprint() with a bio + photo-identity set that can change mid-poll --
    simulating the deck advancing via a manual swipe GESTURE or keyboard shortcut (not
    a like/pass button click), which the injected click listener can't observe.
    bios/photo_ids are read together, one pair per _card_fingerprint() call (indexed
    by fp_calls, then clamped to the last entry). photo_ids mirrors the real driver's
    _CARD_PHOTO_IDS_JS return shape: a list of per-photo identity strings, read via
    evaluate() (not query_selector_all -- see BUMBLE-7)."""
    def __init__(self, reads, bios, photo_ids):
        super().__init__(reads)
        self.bios = list(bios)
        self.photo_ids = list(photo_ids)
        self.fp_calls = 0

    def query_selector(self, sel):
        if sel == d.selectors["bio"]:
            idx = min(self.fp_calls, len(self.bios) - 1)
            return _FakeBioElement(self.bios[idx])
        return None                       # "empty" selector -> deck not empty

    def evaluate(self, script, arg=None):
        if script == bumble._CARD_PHOTO_IDS_JS:
            idx = min(self.fp_calls, len(self.photo_ids) - 1)
            self.fp_calls += 1            # advances the (bio, photo-ids) pair together
            return list(self.photo_ids[idx])
        return super().evaluate(script, arg)


class FakeSwipePage:
    """Drives like()/dislike() end to end -- the human-cursor click AND the real
    _verify_swipe_landed check -- against a scripted sequence of _card_fingerprint()
    snapshots (bio + per-photo identity strings, one entry per call, clamped to the
    last once exhausted).

    Unlike FakeActionPage/FakeNoElementPage (used by the click-path tests above),
    this fake implements BOTH query_selector_all (the pre-fix, count-only signal)
    AND evaluate (the current, identity-based signal from _CARD_PHOTO_IDS_JS), driven
    by the SAME fp_calls counter/photo_ids list -- so the identical fake can exercise
    either fingerprint implementation, which is what lets BUMBLE-7's regression test
    below prove the old scheme collides while the new one doesn't (see the docstring
    on test_like_tells_apart_distinct_profiles_with_same_bio_and_photo_count)."""
    viewport_size = {"width": 1280, "height": 900}

    def __init__(self, bios, photo_ids, box=None):
        self.bios = list(bios)
        self.photo_ids = list(photo_ids)
        self.fp_calls = 0
        self.mouse = _RecordingMouse()
        self.plain_clicks = []
        self._box = box or {"x": 1000, "y": 820, "width": 56, "height": 56}

    def _ids_at(self, idx):
        return self.photo_ids[min(idx, len(self.photo_ids) - 1)]

    def query_selector(self, sel):
        if sel == d.selectors["bio"]:
            return _FakeBioElement(self.bios[min(self.fp_calls, len(self.bios) - 1)])
        if sel in (d.selectors["like"], d.selectors["pass"]):
            return FakeActionElement(self._box)
        return None                       # "empty" selector -> deck not empty

    def query_selector_all(self, sel):
        assert sel == d.selectors["photo"]
        ids = self._ids_at(self.fp_calls)
        self.fp_calls += 1                # pre-fix path: count only, advances the pair
        return [object()] * len(ids)

    def evaluate(self, script, arg=None):
        assert script == bumble._CARD_PHOTO_IDS_JS
        ids = self._ids_at(self.fp_calls)
        self.fp_calls += 1                # current path: real identities, advances the pair
        return list(ids)

    def click(self, sel):
        self.plain_clicks.append(sel)


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
    """No banners present: every selector lookup misses.

    `clicks` records teleport page.click() calls, which startup cleanup must no longer
    make — it goes through the human cursor path like every other click now.
    """

    def __init__(self):
        self.default_timeout = None
        self.goto_calls = []
        self.clicks = []
        self.lookups = []

    def set_default_timeout(self, timeout):
        self.default_timeout = timeout

    def goto(self, url, wait_until=None):
        self.goto_calls.append((url, wait_until))

    def query_selector(self, selector):
        self.lookups.append(selector)
        return None                      # banner not on the page

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


class RecordingOverlayPage(FakePhotoPage):
    """Like FakePhotoPage, but also simulates oplove-hud/oplove-busy visibility and
    records every evaluate() call (plus, via RecordingElement, every screenshot())
    into one ordered list -- so a test can assert the overlay-hide happens BEFORE
    any el.screenshot() and is restored afterwards, not just that it happened."""
    def __init__(self, elements, advance=False):
        super().__init__(elements, advance=advance)
        self.events = []
        self.hud_visible = True
        self.busy_visible = False

    def evaluate(self, script, arg=None):
        if script == bumble._OVERLAY_JS:
            self.hud_visible = True
            self.events.append("show_hud")
        elif script == bumble._BUSY_JS:
            self.busy_visible = bool(arg)
            self.events.append("show_busy" if arg else "hide_busy")
        else:                                  # _hide_oplove_overlays' inline hide script
            self.hud_visible = False
            self.busy_visible = False
            self.events.append("hide_overlays")
        return None


class RecordingElement(FakePhotoElement):
    def __init__(self, name, box, events):
        super().__init__(name, box)
        self._events = events

    def screenshot(self):
        self._events.append(f"screenshot:{self.name}")
        return super().screenshot()


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


def test_card_change_without_click_returns_none_not_misattributed():
    # Regression for BUMBLE-2: the deck can advance without the like/pass listener
    # firing (a swipe gesture, a keyboard shortcut, an app-driven advance). Without a
    # card-identity signal, wait_for_decision has no way to notice -- so the 'like'
    # read on the 3rd poll (a real click, but on the profile that REPLACED the one we
    # were waiting on) would be returned as True and mis-attributed to the old card, a
    # silently mislabelled training example. The fingerprint (bio + photo identities)
    # changes on the 2nd poll, before that 'like' is ever read, and must short-circuit
    # to None instead.
    page = FakeCardChangePage(
        reads=[None, None, "like"],
        bios=["profile A bio", "profile A bio", "profile B bio"],
        photo_ids=[["p1", "p2", "p3", "p4"], ["p1", "p2", "p3", "p4"], ["q1", "q2"]],
    )
    drv = _driver(page)
    assert drv.wait_for_decision(timeout=5) is None


def test_stable_card_still_detects_like_through_fingerprint_checks():
    # The new fingerprint check must not itself cause false positives: a card that
    # hasn't changed should still let a like through once the listener fires.
    page = FakeCardChangePage(
        reads=[None, None, "like"],
        bios=["profile A bio"],
        photo_ids=[["p1", "p2", "p3", "p4"]],
    )
    drv = _driver(page)
    assert drv.wait_for_decision(timeout=5) is True


def test_wait_for_profile_album_ready_returns_nothing():
    # Regression for BUMBLE-6: the raw/filtered counts used to be returned here, but
    # the sole caller (_capture_photos) always recomputes both itself on the very next
    # line, so the return value was dead. Locks in the pure-wait contract.
    photo = FakePhotoElement("profile_photo", {"x": 420, "y": 170, "width": 500, "height": 680})
    drv = _driver(FakePhotoPage([photo]))
    with _fast_capture_waits():
        assert drv._wait_for_profile_album_ready() is None


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


def test_capture_photos_hides_overlays_before_screenshotting():
    # Regression for the no_face outage: our own HUD/busy overlays must be hidden
    # BEFORE el.screenshot() captures a profile photo, or they get baked into the
    # image, ArcFace finds zero faces, and embed_profile silently returns None for
    # every profile (see bumble.py's _capture_photos / _hide_oplove_overlays).
    page = RecordingOverlayPage([])
    photo = RecordingElement(
        "profile_photo", {"x": 420, "y": 170, "width": 500, "height": 680}, page.events
    )
    page.frames = [[photo]]
    drv = _driver(page)
    drv.inpage_overlays = True             # so the hide/show JS actually runs (off by default)

    with _fast_capture_waits():
        assert drv._capture_photos() == [b"profile_photo"]

    assert page.events == ["hide_overlays", "screenshot:profile_photo"]
    assert page.hud_visible is False and page.busy_visible is False


def test_render_status_after_capture_restores_hidden_overlay():
    # The hide is only supposed to be temporary: the worker's normal per-loop
    # render_status() call afterwards must show the HUD again. If the hide were
    # never undone (or happened again after the restore), the HUD would stay dark
    # for the rest of the run.
    page = RecordingOverlayPage([])
    photo = RecordingElement(
        "profile_photo", {"x": 420, "y": 170, "width": 500, "height": 680}, page.events
    )
    page.frames = [[photo]]
    drv = _driver(page)
    drv.inpage_overlays = True

    with _fast_capture_waits():
        drv._capture_photos()
    assert page.hud_visible is False       # still hidden right after capture

    drv.render_status({"mode": "observe"})
    assert page.events[-1] == "show_hud"
    assert page.hud_visible is True        # restored on the worker's next render


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


# These two previously asserted a fallback to page.click(). That fallback is GONE: it
# dispatches at the element with no cursor travel, and ops/ANTI-BOT-RESEARCH.md §1 lists
# cursor path ("Bezier vs zero-time teleport") as a HIGH-confidence behavioural detection
# vector. Silently swapping the human path for a teleport bought one extra swipe in
# exchange for an invisible, open-ended increase in detectability. Refusing is correct:
# a control that is still off-screen AFTER scroll-into-view means the page is not in the
# state we think it is, which is a reason to stop and look, not to click harder.
def test_dislike_refuses_rather_than_teleport_clicking_without_a_box():
    drv = _driver(FakeNoElementPage())
    with pytest.raises(HumanInputUnavailable, match="no bounding box"):
        drv.dislike()
    assert drv.page.plain_clicks == [], "no teleport click may be issued"


def test_like_refuses_rather_than_teleport_clicking_an_off_screen_control():
    box = {"x": 4000, "y": 4000, "width": 56, "height": 56}
    # FakeActionElement has no scroll_into_view_if_needed, so the box stays off-screen.
    assert not hasattr(FakeActionElement(box), "scroll_into_view_if_needed")
    drv = _driver(FakeActionPage(box))
    with pytest.raises(HumanInputUnavailable, match="outside the viewport"):
        drv.like()
    page = drv.page
    assert page.plain_clicks == [], "no teleport click may be issued"
    assert not page.mouse.clicks, "and no raw mouse click at the off-screen point either"


# --- BUMBLE-7: _verify_swipe_landed actually executes against a real fingerprint ---
# FakeActionPage/FakeNoElementPage (used by the click-path tests above) don't implement
# evaluate() or query_selector_all(), so _card_fingerprint() raises internally, is
# swallowed by its own broad `except Exception: return None`, and _verify_swipe_landed
# silently no-ops (before is None) -- the verification path added alongside
# _card_fingerprint is never actually reached by those tests. FakeSwipePage backs both
# query_selector_all (the pre-fix count signal) and evaluate (the current identity
# signal), so the checks below drive the real check, not a short-circuit.

def test_like_verifies_swipe_landed_via_changing_fingerprint():
    # (a) A genuinely landed swipe: bio + photo identities differ between the
    # pre-click and post-click snapshot, so _verify_swipe_landed sees a real change
    # and like() returns cleanly on the first re-check.
    page = FakeSwipePage(
        bios=["profile A bio", "profile B bio"],
        photo_ids=[
            ["https://cdn.bumble.example/a1.jpg", "https://cdn.bumble.example/a2.jpg"],
            ["https://cdn.bumble.example/b1.jpg", "https://cdn.bumble.example/b2.jpg"],
        ],
    )
    drv = _driver(page)
    drv.like()                       # must not raise
    assert page.fp_calls == 2        # before-click snapshot + one landed re-check


def test_dislike_raises_when_fingerprint_never_changes(monkeypatch):
    # (b) An unlanded swipe: the fingerprint reads identical before and after every
    # re-check (e.g. the click was covered by a modal, or hit a card mid-animation),
    # so _verify_swipe_landed must still raise BumbleActionError -- proving the check
    # actually fires now, not just that it stays quiet on a genuine change.
    monkeypatch.setattr(bumble.time, "sleep", lambda *_: None)   # skip the real settle delay
    page = FakeSwipePage(
        bios=["profile A bio"],
        photo_ids=[["https://cdn.bumble.example/a1.jpg", "https://cdn.bumble.example/a2.jpg"]],
    )
    drv = _driver(page)
    try:
        drv.dislike()
    except bumble.BumbleActionError:
        pass
    else:
        raise AssertionError("expected BumbleActionError")


def test_like_tells_apart_distinct_profiles_with_same_bio_and_photo_count():
    # (c) Regression for BUMBLE-7: two DIFFERENT real profiles sharing a blank bio and
    # the same photo COUNT -- entirely plausible on Bumble, since the "About" bio is
    # optional and frequently blank, and photo counts cluster tightly at the app's max
    # -- must NOT look identical to the verifier. Both snapshots here have bio="" and
    # 3 photos: exactly the shape that collided under the pre-fix
    # (out_of_profiles, bio, photo_count) fingerprint. The underlying photo URLs
    # differ, so the identity-based fingerprint tells them apart and like() returns
    # cleanly instead of raising BumbleActionError on a swipe that genuinely landed.
    #
    # This test is proven to catch the regression by temporarily reverting
    # _card_fingerprint to the pre-fix (out_of_profiles, bio, photo_count) tuple: with
    # that implementation the test fails (BumbleActionError, unlanded-swipe false
    # positive) because FakeSwipePage's query_selector_all-backed count is 3 in both
    # snapshots, colliding exactly like two real Bumble profiles would.
    page = FakeSwipePage(
        bios=["", ""],
        photo_ids=[
            ["https://cdn.bumble.example/photoA1.jpg",
             "https://cdn.bumble.example/photoA2.jpg",
             "https://cdn.bumble.example/photoA3.jpg"],
            ["https://cdn.bumble.example/photoB1.jpg",
             "https://cdn.bumble.example/photoB2.jpg",
             "https://cdn.bumble.example/photoB3.jpg"],
        ],
    )
    drv = _driver(page)
    drv.like()                       # must not raise -- distinct photo identities prove the change


def test_open_session_denies_native_permission_prompts(monkeypatch):
    # This test is about launch-hardening mechanics (the anti-automation args,
    # dropping --enable-automation), not about the platform-availability guard
    # PlaywrightDriver.open_session() now runs first (see test_web_base.py for
    # that) -- Bumble web has no live target, so the guard would otherwise stop
    # this test before it ever reaches the fake Playwright launch it's exercising.
    monkeypatch.setattr(platforms, "unavailable_reason", lambda app: None)

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
            cfg = type("Cfg", (), {"apps": {"bumble_web": {"user_data_dir": user_dir}}})()
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
    drv._dismiss_startup_interstitials()          # must not raise
    # Every banner is still ATTEMPTED — one absent selector must not abort the rest.
    assert page.lookups == [sel for _label, sel in bumble._STARTUP_INTERSTITIALS]


def test_dismiss_startup_interstitials_never_teleport_clicks():
    # Startup cleanup used to call page.click() directly, which dispatches with no cursor
    # travel. A session whose first interactions are teleports and whose later ones are
    # Bezier paths is arguably more distinctive than one that is consistently either, and
    # consistency costs nothing here — so this path uses the human cursor too. A banner we
    # cannot click humanly is skipped, never clicked worse.
    page = FakeStartupPage()
    drv = _driver(page)
    drv._dismiss_startup_interstitials()
    assert page.clicks == [], "startup cleanup must not fall back to a teleport click"
