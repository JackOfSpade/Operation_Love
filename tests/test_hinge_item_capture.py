"""Item enumeration inside the profile read (ops/OPENER-REDESIGN.md 5.2/5.3/5.5).

The layer under test is `hinge._capture_current`: the read that already scrolls and screencaps
now also confirms the scroll top, sizes every step against the card in front of it, folds the
frames into an item index, cuts the numbered crops, and carries them out on the Profile.

Every frame here is SYNTHESISED, never a real screencap. The calibration captures the modules
under this one were measured against are real people's dating profiles and are gitignored
(ops/calibration/, .gitignore:26); only geometry and counts from them appear anywhere in this
repo. So the fixtures build a tall scrollable WORLD from first principles -- the same
construction tests/test_item_index.py and tests/test_item_nav.py use -- and a fake ADB serves
1080x2400 windows of it, moving by exactly what the measured transport model says the requested
`frac` delivers. Nothing here approximates the vision: the segmenter really finds the glyphs,
the shift estimator really measures the pixels, and the assertions can name the item count, the
heart ordinals and the delivered step distance rather than approximating any of them.

The ONE piece of real calibration data in the fixtures is `_SCROLL_TOP_BAND_FINGERPRINT` -- the
shipped constant, which is Hinge's own filter-chips chrome and is already in the repo. Painting
it into the top-of-page frames is what lets the real scroll-top gate run against synthetic
frames with its real reference.

Every positive is paired with a negative: for each thing the enumeration produces there is a
test that it is REFUSED, by name and with a reason, rather than degraded into raw frames.
"""
import dataclasses
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import (
    hinge, item_identity, item_index, scroll_step, scroll_top, segment)
from operation_love.drivers.hinge import HingeDriver
from operation_love.drivers.debuglog import HingeDebugLog

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 11
_CONTENT_BAND = hinge.HINGE_SPEC.content_band          # (0.125, 0.875) -> rows 300..2100
_IDENTITY_BAND = hinge.HINGE_SPEC.identity_band        # (0.10, 0.048, 0.80, 0.094)
_BAND0, _BAND1 = 300, 2100
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - _CARD_X0
_GUTTER = max(segment._GUTTER_PX)          # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                     # doc 5.10: "~20-23px"
_HEART_CX = 937                            # bottom-right, as test_segment.py stamps it
_HEART_ABOVE_BOTTOM = 90
_PAGE_TOP, _PAGE_BOTTOM = 254, 243         # doc 5.10's background gradient

# THE PAGE, in world rows. Four likeable cards and, between the second and third, the heartless
# 215px vitals block doc 5.10 tracked across 7 consecutive frames. Heights are from doc 5.10's
# card table; the heart-to-heart pitches this produces (813, 1321, 873) all clear the ~608px
# floor below which `plan_scroll_step` refuses to enumerate at all.
_PAGE_TOP_GAP = 400
_LAYOUT = (("card", 900), ("card", 760), ("context", 215), ("card", 1000), ("card", 820))
# The world ends 493px below the last card, so the frame the scroll CLAMPS at still shows that
# card's bottom corner with more page background under it than any gutter -- which is what
# `reached_end` is read off, i.e. this world has a real bottom rather than an infinite tail.
_WORLD_H = 4800


def _layout_rows(layout=_LAYOUT, top_gap=_PAGE_TOP_GAP):
    rows, y = [], top_gap
    for kind, height in layout:
        rows.append((kind, y, y + height))
        y += height + _GUTTER
    return rows


_ROWS = _layout_rows()
_CARDS = [r for r in _ROWS if r[0] == "card"]
_HEART_PAGE_Y = [y1 - _HEART_ABOVE_BOTTOM for _kind, _y0, y1 in _CARDS]
_MAX_SCROLL = _WORLD_H - _H
assert _ROWS[-1][2] + _GUTTER < _MAX_SCROLL + _BAND1, (
    "the last card's bottom must still be inside the analysed band once the scroll clamps, or "
    "the read can never observe the end of the profile")

_TEMPLATE = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"


def _page_column(height=_WORLD_H):
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, height).round().astype(np.uint8)


def _build_world(*, layout=_LAYOUT, top_gap=_PAGE_TOP_GAP):
    rng = np.random.default_rng(_SEED)
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    for kind, y0, y1 in _layout_rows(layout, top_gap):
        world[y0:y1, _CARD_X0:_CARD_X1] = rng.integers(
            60, 200, size=(y1 - y0, _CARD_X1 - _CARD_X0), dtype=np.uint8)
        for i in range(_CORNER_RADIUS_PX):             # carve the four corner arcs back to page
            dy = _CORNER_RADIUS_PX - i
            inset = int(math.ceil(_CORNER_RADIUS_PX
                                  - math.sqrt(max(0.0, _CORNER_RADIUS_PX ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for y in (y0 + i, y1 - 1 - i):
                world[y, _CARD_X0:_CARD_X0 + inset] = col[y]
                world[y, _CARD_X1 - inset:_CARD_X1] = col[y]
        if kind != "card":
            continue
        th, tw = _TEMPLATE.shape
        cy = y1 - _HEART_ABOVE_BOTTOM
        world[cy - th // 2: cy - th // 2 + th,
              _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = _TEMPLATE
    return world


# Static chrome outside the content band, textured but IDENTICAL on every frame -- the real
# device's status bar, floating buttons and bottom nav, none of which translate.
_CHROME_TOP = np.random.default_rng(3).integers(0, 60, size=(_BAND0, _W), dtype=np.uint8)
_CHROME_BOTTOM = np.random.default_rng(4).integers(0, 60, size=(_H - _BAND1, _W), dtype=np.uint8)


def _identity_rect(rect=_IDENTITY_BAND):
    """The exact pixel rect `hinge._band` will cut for `rect`, mirroring its own rounding."""
    x0, y0, x1, y1 = rect
    return round(y0 * _H), round(y1 * _H), round(x0 * _W), round(x1 * _W)


def _chips_band(rows, cols):
    """Hinge's filter-chips strip, reconstructed from the SHIPPED fingerprint constant.

    `scroll_top`'s reference is 64 grey levels on a 16x4 grid over this rect, so painting those
    cells back as blocks and letting `hinge._band` downsample them again reproduces the real
    reference through the real decode. It is app chrome, not profile content -- the constant's
    own comment records it as byte-identical across two different people, which is why one array
    serves every profile and why nothing about anyone is in it.
    """
    cells = np.array(scroll_top._SCROLL_TOP_BAND_FINGERPRINT, dtype="uint8").reshape(4, 16)
    return np.asarray(Image.fromarray(cells, mode="L").resize((cols, rows), Image.NEAREST))


# The sticky per-profile header that covers that strip the moment the card is scrolled at all.
# Flat mid-grey: 113 grey levels from the chips row, i.e. unambiguously REFUTED.
_HEADER_VALUE = 128
# A DIFFERENT person's sticky header. Far enough from _HEADER_VALUE to clear the driver's own
# change_threshold, which is what makes it read as "the deck advanced mid-read".
_OTHER_HEADER_VALUE = 30

_TARGETING_CALIBRATION = {
    "schema_version": 3,
    "hinge_version_name": "9.134.0",
    "frame_size_px": [1080, 2400],
    "composer_layout_id": "hinge_inline_v1",
    "item_selection_policy_id": "hinge_photos_only_v1",
    "identity_match_max_dist": 2.0,
    "inline_item_max_dist": 10.0,
    "device": "pixel",
    "calibrated_at": "2026-08-12",
    "identity_band": list(hinge.HINGE_SPEC.identity_band),
    "content_band": list(hinge.HINGE_SPEC.content_band),
}

_WORLD = None
_FRAMES: dict[tuple, bytes] = {}


def _world():
    global _WORLD
    if _WORLD is None:
        _WORLD = _build_world()
    return _WORLD


def _frame(scroll: int, *, at_top: bool | None = None, header=_HEADER_VALUE) -> bytes:
    """The 1080x2400 window of the world at `scroll`, as PNG.

    Content at world row `w` lands on frame row `w - scroll`, so `_frame(s)` then `_frame(s + d)`
    is a forward scroll of exactly `d`. `at_top` defaults to `scroll == 0` and decides which of
    the two strips is painted into the identity band, which is the only signal the scroll-top
    gate reads.
    """
    top = (scroll == 0) if at_top is None else at_top
    key = (scroll, top, header)
    if key not in _FRAMES:
        gray = _world()[scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        r0, r1, c0, c1 = _identity_rect()
        gray[r0:r1, c0:c1] = _chips_band(r1 - r0, c1 - c0) if top else header
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAMES[key] = buf.tobytes()
    return _FRAMES[key]


def _foreign_frame(scroll: int, *, at_top: bool | None = None,
                   header=_OTHER_HEADER_VALUE) -> bytes:
    """A different profile: same app chrome/geometry, unrelated scrolling content."""
    gray = cv2.imdecode(np.frombuffer(_frame(scroll, at_top=at_top, header=header), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    gray[_BAND0:_BAND1] = 255 - gray[_BAND0:_BAND1]
    # A different profile still has Hinge's selectable heart controls. Re-stamp the fixed app
    # glyph after changing the card pixels so the recovery/index assertions remain realistic.
    th, tw = _TEMPLATE.shape
    for page_y in _HEART_PAGE_Y:
        cy = page_y - scroll
        y0 = cy - th // 2
        x0 = _HEART_CX - tw // 2
        if _BAND0 <= y0 and y0 + th <= _BAND1:
            gray[y0:y0 + th, x0:x0 + tw] = _TEMPLATE
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    return buf.tobytes()


class WorldAdb:
    """A phone made of arithmetic: it serves windows of the world and moves by exactly what the
    transport model says a `frac` delivers, including its measured 21px touch slop.

    Deliberately has no `tap`/`text` beyond recording: this file exercises the READ, and a test
    that accidentally acted would show up as an unexpected recorded tap.
    """

    def __init__(self, *, start=0, at_top=None, header=_HEADER_VALUE,
                 header_after=None, frozen=False):
        self.scroll = start
        self._at_top = at_top
        self._header = header
        self._header_after = header_after   # (n_scrolls, value): the deck advances mid-read
        self._frozen = frozen               # a screen that never moves, whatever we ask of it
        self.scrolls = 0
        self.gestures: list[tuple[float, float]] = []
        self.taps = []
        self.texts = []

    def screen_size(self):
        return (_W, _H)

    def devices(self):
        return ["pixel"]

    def shell(self, command="", **_):
        if command.startswith("dumpsys package "):
            return "versionName=9.134.0\n"
        return ""

    def screencap(self):
        header = self._header
        if self._header_after is not None and self.scrolls >= self._header_after[0]:
            header = self._header_after[1]
        at_top = self._at_top if self._at_top is not None else (self.scroll == 0)
        if self._header_after is not None and self.scrolls >= self._header_after[0]:
            return _foreign_frame(self.scroll, at_top=at_top, header=header)
        return _frame(self.scroll, at_top=at_top, header=header)

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, *_a, **_k):
        pass

    def scroll_up(self, frac, x_frac=0.5):
        self.scrolls += 1
        self.gestures.append((frac, x_frac))
        if self._frozen:
            return
        self.scroll = max(0, min(_MAX_SCROLL, self.scroll + scroll_step.step_px_for_frac(frac, _H)))

    def text(self, s):
        self.texts.append(s)


def _drv(adb, *, auto=True, openers=True, targeting_calibration=True, **cfg):
    """A HingeDriver over the fake world, with the two session hooks a Worker installs.

    `auto` stands in for Worker._auto_loop's set_auto_session_policy (policy None keeps the read
    deterministic: no capture-limit jitter, production dwell/lane fallbacks). It no longer
    decides whether the read ENUMERATES -- doc 5.9's inversion made observe an enumerating mode
    too, so `openers` is what that turns on now, standing in for the set_opener_enabled call BOTH
    loops make. `openers=False` is therefore the way to ask for an ordinary, non-enumerating
    read, in either mode."""
    app_cfg = {"serial": "pixel", "halt_on_error": False, **cfg}
    if targeting_calibration:
        app_cfg["targeting_calibration"] = _TARGETING_CALIBRATION

    class C:
        apps = {"hinge": app_cfg}
    d = HingeDriver(C())
    d._adb = adb
    d._touch = adb            # touches route through the (fake) transport
    if auto:
        d.set_auto_session_policy(None)
    d.set_opener_enabled(openers)
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch, request):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    # This synthetic world's grayscale noise cards carry geometry, not semantic photo/prompt
    # evidence.  The real classifier has its own pixel tests and the crop layer has explicit
    # photo-only numbering tests; keep these capture/navigation tests focused on their declared
    # geometry by labelling every synthetic selectable card a photo.
    monkeypatch.setattr(hinge, "unnumber_unless_confident_photo", lambda _crop: None)
    # Video screening has its own exact-template/ROI tests below. Every synthetic noise card here
    # is a successfully screened still by default.
    if request.node.name != "test_video_mute_template_is_a_near_perfect_app_ui_match":
        monkeypatch.setattr(
            HingeDriver, "_match_video_mute",
            staticmethod(lambda _frame, _rect: (True, 0.0)))
    # The sticky-header OCR is a subprocess call to tesseract; it is not what this file tests,
    # and stubbing it also lets the name assertions below be exact.
    monkeypatch.setattr(HingeDriver, "_ocr_band", lambda self, *a, **k: "Ada")


@pytest.fixture(autouse=True)
def _deterministic_draws():
    """`plan_scroll_step` draws its step from the module-level `random`. Seeding makes a failure
    reproducible; nothing below depends on a particular draw, only on the window it comes from."""
    random.seed(_SEED)


# =====================================================================================
# The world is what the test says it is
# =====================================================================================

def test_the_fixture_page_is_the_page_the_assertions_below_assume():
    """Not an assertion about the driver -- a guard on the fixture. Everything below reads its
    ground truth off these numbers, so if the world stops matching the layout table the failures
    would look like capture bugs."""
    scrolls = tuple(260 * i for i in range(10))
    assert scrolls[-1] <= _MAX_SCROLL
    index = item_index.build_item_index(
        [_frame(s) for s in scrolls], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)

    assert index.usable, index.failures
    assert index.at_scroll_top and index.reached_end
    assert [b.kind for b in index.blocks] == [
        item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE, item_index.ITEM_CONTEXT,
        item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE]
    assert index.translation == (1, 2, 3, 4)


def test_the_scroll_top_frames_confirm_and_the_scrolled_ones_refute():
    """The other fixture guard, and the one that makes every "top" assertion below mean
    something: the SHIPPED fingerprint, through the SHIPPED decode, on these synthetic frames."""
    assert scroll_top.confirm_scroll_top(_frame(0), identity_band=_IDENTITY_BAND).confirmed
    assert scroll_top.confirm_scroll_top(_frame(600), identity_band=_IDENTITY_BAND).refuted


# =====================================================================================
# THE HEADLINE: an auto read enumerates the profile
# =====================================================================================

def test_auto_capture_enumerates_the_profile_into_numbered_crops():
    """Doc 5.2/5.3 end to end. The read produces one crop per heart-bearing item in model order,
    the heartless vitals block as unnumbered context, her name as text, and a truncation flag
    that is False because the capture demonstrably started at a confirmed top and reached the
    end. `items_unavailable` is empty: exactly one of it and `items` is ever set."""
    adb = WorldAdb()
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile is not None
    assert len(profile.items) == len(_CARDS) == 4
    assert len(profile.item_context) == 1
    assert profile.name == "Ada"
    assert profile.items_truncated is False
    assert profile.items_unavailable == ""


def test_hinge_numbers_only_policy_approved_photos_and_preserves_prompt_heart(monkeypatch):
    seen = 0

    def photo_only(_crop):
        nonlocal seen
        seen += 1
        return "synthetic written prompt" if seen == 2 else None

    monkeypatch.setattr(hinge, "unnumber_unless_confident_photo", photo_only)
    drv = _drv(WorldAdb())
    profile = drv._capture_current()

    assert len(profile.items) == 3
    assert drv._current_item_payload.translation == (1, 3, 4)
    prompt = next(c for c in drv._current_item_payload.context if c.heart_ordinal == 2)
    assert prompt.number is None and prompt.sent and "written prompt" in prompt.reason
    assert all(isinstance(crop, bytes) and crop for crop in profile.items + profile.item_context)


def test_video_mute_exclusion_keeps_heart_space_but_removes_model_choice(monkeypatch):
    """A video is never numbered, while later photos retain their original heart ordinals."""
    monkeypatch.setattr(
        HingeDriver, "_video_selection_exclusions",
        lambda self, frames, index: {
            2: "video_mute_v1: upper-left mute control matched"})
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert profile is not None and len(profile.items) == 3
    assert drv._current_item_index.heart_count == 4
    assert drv._current_item_payload.translation == (1, 3, 4)
    video = next(crop for crop in drv._current_item_payload.excluded
                 if crop.heart_ordinal == 2)
    assert video.number is None and video.image is None and not video.sent
    assert "video_mute_v1" in video.reason


def test_all_video_profile_fails_before_opener_or_targeting(monkeypatch):
    monkeypatch.setattr(
        HingeDriver, "_video_selection_exclusions",
        lambda self, frames, index: {
            block.heart_ordinal: "video_mute_v1: mute control matched"
            for block in index.selectable})
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert profile is not None and profile.items == ()
    assert "no policy-approved selectable item" in profile.items_unavailable
    assert drv._current_item_index is None and drv._current_item_payload is None


def test_video_screen_reads_only_card_local_upper_left_sightings(monkeypatch):
    """The Android mute status icon and card like-heart are never inside the searched ROI."""
    frames = [_frame(260 * i) for i in range(10)]
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)
    seen: list[tuple[int, int, int, int]] = []

    def screen(_frame_bytes, rect):
        seen.append(rect)
        return True, (1.0 if len(seen) == 2 else 0.0)

    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(screen))
    exclusions = _drv(WorldAdb())._video_selection_exclusions(frames, index)

    assert exclusions and any("mute control" in reason for reason in exclusions.values())
    # Manifest diagnostics retain only bounded geometry plus a scalar matcher score, never OCR
    # text or image data from the prospective video card.
    assert any("screen outcomes: f" in reason and "mute_matched@[" in reason
               and "score=1.0" in reason for reason in exclusions.values())
    assert all("ocr" not in reason.lower() and "pixel" not in reason.lower()
               for reason in exclusions.values())
    assert seen
    assert all(x0 >= 53 and x1 <= 1027 and y0 >= round(_H * _CONTENT_BAND[0])
               for x0, y0, x1, _y1 in seen)


def test_video_screen_matcher_failure_fails_closed_with_bounded_roi_diagnostics(monkeypatch):
    """An unreadable mute ROI is an exclusion, with no video content leaking into the manifest."""
    frames = [_frame(260 * i) for i in range(10)]
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)
    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(
        lambda _frame_bytes, _rect: (False, None)))

    exclusions = _drv(WorldAdb())._video_selection_exclusions(frames, index)

    assert exclusions
    for reason in exclusions.values():
        assert "not targetable because upper-left mute-control screening failed" in reason
        assert "screen outcomes: f" in reason
        assert "matcher_failed@[" in reason
        assert "ocr" not in reason.lower() and "pixel" not in reason.lower()
        assert "b'" not in reason


def test_clipped_mute_roi_does_not_overrule_a_later_successful_no_mute_screen(monkeypatch):
    """A 9px card sliver cannot be mislabeled matcher failure and exclude a real photo."""
    block = SimpleNamespace(
        heart_ordinal=1, x0=_CARD_X0, x1=_CARD_X1, height=974,
        observations=(
            SimpleNamespace(top_observed=True, frame_index=0, frame_y0=2091, frame_y1=2100),
            SimpleNamespace(top_observed=True, frame_index=1, frame_y0=700, frame_y1=1674),
        ))
    index = SimpleNamespace(selectable=(block,))
    calls = []

    def no_mute(_frame, rect):
        calls.append(rect)
        return True, 0.1

    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(no_mute))
    exclusions = _drv(WorldAdb())._video_selection_exclusions(
        [_frame(0), _frame(0)], index)

    assert exclusions == {}
    assert calls == [(_CARD_X0, 700, _CARD_X0 + round(0.22 * 974), 836)]


def test_only_clipped_mute_rois_still_fail_closed_without_calling_matcher(monkeypatch):
    """Skipping an impossible ROI is not permission when no complete screen exists at all."""
    block = SimpleNamespace(
        heart_ordinal=1, x0=_CARD_X0, x1=_CARD_X1, height=974,
        observations=(
            SimpleNamespace(top_observed=True, frame_index=0, frame_y0=2094, frame_y1=2100),
        ))
    index = SimpleNamespace(selectable=(block,))
    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(
        lambda *_args: pytest.fail("an undersized ROI must be skipped before OpenCV")))

    exclusions = _drv(WorldAdb())._video_selection_exclusions([_frame(0)], index)

    reason = exclusions[1]
    assert "no source frame exposed the 42x42" in reason
    assert "insufficient_visible_roi" in reason


def test_video_mute_template_is_a_near_perfect_app_ui_match():
    """The embedded template contains only the stable glyph/black disk, not video pixels."""
    template = cv2.imdecode(
        np.frombuffer(hinge.base64.b64decode(hinge._VIDEO_MUTE_TEMPLATE_B64), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE)
    assert template.shape == (hinge._VIDEO_MUTE_TEMPLATE_SIDE_PX,) * 2
    frame = np.full((_H, _W), 173, dtype=np.uint8)
    y0, x0 = 660, 106
    frame[y0:y0 + template.shape[0], x0:x0 + template.shape[1]] = template
    ok, encoded = cv2.imencode(".png", frame)
    assert ok

    screened, score = HingeDriver._match_video_mute(
        encoded.tobytes(), (_CARD_X0, 607, 268, 812))

    assert screened and score == pytest.approx(1.0, abs=1e-6)
    assert score >= hinge._VIDEO_MUTE_MATCH_THRESHOLD

    class MarkerReader:
        content_band = _CONTENT_BAND
        _match_video_mute = staticmethod(HingeDriver._match_video_mute)

    assert HingeDriver._video_mute_frame_markers(
        MarkerReader(), [encoded.tobytes()]) == (True,)

    # The identical UI pixels in Android's top bar or Hinge's right-side control lane do not
    # grant animation-repair authority because both sit outside the frozen marker search ROI.
    outside = np.full((_H, _W), 173, dtype=np.uint8)
    outside[40:40 + template.shape[0], 106:106 + template.shape[1]] = template
    outside[660:660 + template.shape[0], 900:900 + template.shape[1]] = template
    ok, outside_encoded = cv2.imencode(".png", outside)
    assert ok
    assert HingeDriver._video_mute_frame_markers(
        MarkerReader(), [outside_encoded.tobytes()]) == (False,)


def test_the_driver_keeps_the_index_and_the_translation_table_the_profile_does_not():
    """Doc 5.3: "Index space belongs to the driver." The model's dense 1..N and the heart
    ordinals behind it stay on the driver, where navigation and doc 5.6's verification will read
    them; the Profile carries only the images the model is shown."""
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert drv._current_item_index is not None and drv._current_item_index.usable
    assert drv._current_item_index.at_scroll_top is True
    payload = drv._current_item_payload
    assert payload is not None and payload.usable
    assert payload.translation == (1, 2, 3, 4)      # model item k -> heart ordinal
    assert [c.signature is not None for c in payload.items] == [True] * 4
    assert drv._current_items_unavailable == ""
    # The Profile is the model's view and nothing more: no index, no signatures, no ordinals.
    assert not hasattr(profile, "translation")
    assert profile.items == tuple(c.image for c in payload.items)
    assert profile.item_context == tuple(c.image for c in payload.context)


def test_capture_debug_manifest_preserves_page_order_and_model_to_heart_mapping():
    """A report must explain which physical crop became model item N without storing more
    profile imagery.  In particular, dense numbering after a policy demotion must never make a
    later photo look like it was the first card on the page.
    """
    class Debug:
        def __init__(self):
            self.calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    drv = _drv(WorldAdb())
    debug = Debug()
    drv._dbg = debug

    profile = drv._capture_current()

    assert profile is not None
    capture = next(fields for name, fields in debug.calls if name == "capture")
    assert capture["item_translation"] == [1, 2, 3, 4]
    numbered = [row for row in capture["item_manifest"] if row["model_item"] is not None]
    assert [row["model_item"] for row in numbered] == [1, 2, 3, 4]
    assert [row["heart_ordinal"] for row in numbered] == [1, 2, 3, 4]
    assert [row["page_rows"] for row in numbered] == sorted(
        (row["page_rows"] for row in numbered), key=lambda rows: rows[0])
    assert all(row["source_frame_index"] is not None for row in numbered)
    assert all(len(row["crop_sha256"]) == 16 for row in numbered)
    assert all(row["selection_evidence"]["classifier_id"] == "hinge_crop_type_v2"
               for row in numbered)
    assert all(row["selection_evidence"]["classification"] in
               {"photo", "written", "unknown"} for row in numbered)


def test_capture_debug_manifest_translates_repaired_local_frame_to_original_source_frame():
    """An isolated-frame index repair removes one local frame. Manifest provenance must name
    the original capture frame, not silently report the compacted index position."""
    crop = SimpleNamespace(
        kind="item", number=1, heart_ordinal=3, frame_index=1,
        page_y0=600, page_y1=900, width=974, height=300,
        image=b"numbered crop", reason="item 1")
    payload = SimpleNamespace(crops=(crop,))
    index = SimpleNamespace(source_frame_indices=(0, 2, 3))

    manifest = HingeDriver._item_payload_debug_manifest(payload, index)

    assert manifest[0]["source_frame_index"] == 2
    assert manifest[0]["model_item"] == 1
    assert manifest[0]["heart_ordinal"] == 3


def test_the_crops_are_crops_and_not_the_frames_they_came_from():
    """Doc 5.7's "Not sent: full screenshots, scroll frames". A crop that was really a frame
    would carry the duplication bias doc 5.2 removes -- and would make image k stop being item
    k -- so this pins that the images on the Profile are neither the frames nor as tall as one."""
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    frames = set(profile.photos)
    for crop in profile.items + profile.item_context:
        assert crop not in frames
        assert cv2.imdecode(np.frombuffer(crop, dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE).shape[0] < _H


# =====================================================================================
# The scroll: sized from the card in front of us, still through the humanized path
# =====================================================================================

def test_the_enumeration_step_is_sized_from_local_spacing_never_the_config_cadence():
    """Doc 5.10.1's ratio rule, on the gestures actually issued. Production's `read_scroll_frac`
    (0.55, 1299px) is ~1.6x this page's smallest 813px heart spacing, i.e. deep inside the
    aliasing band where "the same heart moved" and "the next heart arrived" are geometrically
    indistinguishable. Every gesture this read makes must instead land inside the window the
    plan drew from: at or above the driver's own smallest legal read-scroll, and at or below
    0.36 of the local spacing."""
    adb = WorldAdb()
    drv = _drv(adb)

    drv._capture_current()

    assert adb.gestures, "the read must have scrolled"
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    ceiling_px = int(scroll_step._STEP_RATIO_MAX * min(
        b - a for a, b in zip(_HEART_PAGE_Y, _HEART_PAGE_Y[1:])))
    for frac, _lane in adb.gestures:
        assert frac != drv.read_scroll_frac
        assert floor_px <= scroll_step.step_px_for_frac(frac, _H) <= ceiling_px
    # The distance is DRAWN, not a constant: a fixed step is the bot signature the owner rule
    # forbids, and the whole point of drawing over pixels rather than clamping a ratio is that
    # the realised distances do not collapse onto one value.
    assert len({round(frac, 6) for frac, _lane in adb.gestures}) > 1


def test_every_enumeration_gesture_goes_through_the_humanized_scroll_ledger():
    """HUMANIZED INPUT ONLY. The plan produces a number; the only thing that can spend it is
    `_scroll_down_one`, so the ledger (which `_scroll_to_top` unwinds by) must account for every
    gesture the transport saw, and both arguments must be passed -- handing over the frac alone
    makes that method re-sample BOTH from the behaviour policy and silently issue production's
    cadence instead."""
    adb = WorldAdb()
    drv = _drv(adb)

    drv._capture_current()

    assert len(drv._capture_scroll_ledger) == adb.scrolls == len(adb.gestures)
    assert drv._capture_scroll_ledger == adb.gestures
    assert drv._capture_scrolls == len(adb.gestures)
    assert all(0.10 <= lane <= 0.90 for _frac, lane in adb.gestures)


def test_the_capture_ceiling_is_raised_for_the_enumeration_read_only():
    """Doc 5.6's unraised-ceiling blocker. The closed loop needs ~37-43 frames for one profile
    against `scroll_captures`' 12, and the two are different jobs -- so the enumeration read
    gets its own ceiling and the ordinary read keeps the configured one, untouched."""
    enumerating = _drv(WorldAdb())
    enumerating._capture_current()
    assert enumerating._profile_capture_limit == hinge._ENUMERATION_CAPTURE_LIMIT

    ordinary = _drv(WorldAdb(), openers=False)
    ordinary._capture_current()
    assert ordinary._profile_capture_limit == ordinary.scroll_captures
    assert ordinary.scroll_captures == hinge.HINGE_SPEC.scroll_captures


def test_the_enumeration_ceiling_is_derived_from_the_measured_page_and_cadence():
    """`_ENUMERATION_CAPTURE_LIMIT`'s DERIVATION, pinned so the constant cannot drift back to
    "the number the probe happened to run at" (doc 5.5's bottom-up workflow, 2026-08-12).

    It is not a free choice and it is not `_MAX_STEP_PX`-based arithmetic: the loop draws its
    step from a window whose ends are the ratio rule applied to the SMALLEST spacing seen on the
    profile so far, so the realised cadence on both original calibration profiles is ~233..262px
    and not the 363px ceiling. A later profile with optional video content moved at least
    11,773px in 47 gestures and was still not at bottom, invalidating the old 48-frame bound.
    The replacement covers 13,797px even if every gesture lands on the safe floor."""
    worst_page_px, worst_measured_frames = 10027, 44
    realised_step_px = worst_page_px / (worst_measured_frames - 1)
    assert 233 <= realised_step_px <= 262, "the measured cadence the derivation rests on"
    assert hinge._ENUMERATION_CAPTURE_LIMIT == 64

    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    reported_lower_bound_px = 11773
    floor_coverage_px = (hinge._ENUMERATION_CAPTURE_LIMIT - 1) * floor_px
    assert floor_coverage_px == 13797
    assert floor_coverage_px >= math.ceil(reported_lower_bound_px * 1.17)


def test_an_enumeration_read_that_runs_out_of_ceiling_says_so_out_loud(capsys, monkeypatch):
    """"Fail loud rather than truncate silently", doc 5.11's own ask, and the half of it that was
    genuinely missing. Truncation was already recorded — `ItemIndex.truncated` reaches the model
    on `Profile.items_truncated`, per doc 5.7 — but ONLY the model was told, so an operator
    watching a run had no way to know the numbered list stopped part way down a profile.

    Deliberately NOT a hard stop: a profile longer than the derived ceiling is not an error, the
    owner's stop-condition rule scopes stops to an unrecognized screen or an error, and the item
    the model picks is inside the enumerated region either way so navigation is unaffected. What
    it costs is Connect material from the tail, which is a quality cost.

    The ceiling is squeezed to 4 frames rather than the world being made longer: what is under
    test is the loop's own out-of-ceiling branch, not the fixture's page."""
    monkeypatch.setattr(hinge, "_ENUMERATION_CAPTURE_LIMIT", 4)
    drv = _drv(WorldAdb())
    monkeypatch.setattr(drv, "_capture_limit_for_profile", lambda base=None: base or 4)

    profile = drv._capture_current()

    assert profile is not None
    printed = capsys.readouterr().out
    assert "longer than the enumeration read could cover" in printed
    assert "4 frame(s)" in printed and "derived ceiling of 4" in printed

    # The ORDINARY read hitting `scroll_captures` is the pre-existing `capture_truncated` case
    # and must stay quiet here: this notice is about the derived ceiling, not about every short
    # read, and an operator who sees it on every profile will stop reading it.
    ordinary = _drv(WorldAdb(), openers=False)
    monkeypatch.setattr(ordinary, "_capture_limit_for_profile", lambda base=None: 4)
    ordinary._capture_current()
    assert "longer than the enumeration read could cover" not in capsys.readouterr().out


def test_ranker_frames_from_enumeration_is_pure_and_evenly_spaced():
    """Unit coverage for `_ranker_frames_from_enumeration` itself (BUG 3's fix), independent of
    the whole synthetic-world/segmentation stack the end-to-end test below needs to exercise it
    through a real capture. Plain bytes stand in for frames; the function never decodes them."""
    frames = [str(i).encode() for i in range(20)]

    # Fewer frames than the target, or exactly the target: returned untouched, in order.
    assert hinge._ranker_frames_from_enumeration(frames[:5], 8) == frames[:5]
    assert hinge._ranker_frames_from_enumeration(frames[:8], 8) == frames[:8]

    # More frames than the target: thinned to exactly the target, first and last frame kept,
    # strictly increasing (never reordered, never repeated).
    kept = hinge._ranker_frames_from_enumeration(frames, 8)
    assert len(kept) == 8
    assert kept[0] == frames[0]
    assert kept[-1] == frames[-1]
    indices = [frames.index(f) for f in kept]
    assert indices == sorted(set(indices))

    # Edge cases named in the docstring: never invents a frame, and degrades to "everything" on
    # a degenerate target rather than raising.
    assert hinge._ranker_frames_from_enumeration(frames, 1) == [frames[0]]
    assert hinge._ranker_frames_from_enumeration(frames, 0) == frames
    assert hinge._ranker_frames_from_enumeration([], 8) == []


def test_the_ranker_still_sees_scroll_captures_frames_not_the_enumeration_ceiling(monkeypatch):
    """Audit fix, "BUG 3" (2026-08-12). `_ENUMERATION_CAPTURE_LIMIT` (64) raises the CEILING for
    a read that also builds an item index -- the index needs the finer, closed-loop cadence, and
    this test's own fixture already needs more than `scroll_captures` (8, HINGE_SPEC's default)
    raw frames to cover the page at that cadence (see the assertion on `served` below). But
    nothing about the item index needed the RANKER (decider.decide, worker.py's auto loop) to
    see four times as many frames of the same profile: aggregation-design.md's ArcFace pools by
    a raw MEAN over every detected face, so a card sampled more often would silently carry more
    weight. `Profile.photos` must keep matching what the ranker saw before Part B raised the
    ceiling -- `scroll_captures` frames, evenly spread across the whole read rather than
    clustered at the top -- even though the enumeration read itself goes on to read many more
    frames than that to build the index."""
    adb = WorldAdb()
    served: list[bytes] = []
    real_screencap = adb.screencap
    monkeypatch.setattr(adb, "screencap", lambda: served.__iadd__([real_screencap()]) and served[-1])
    drv = _drv(adb)

    profile = drv._capture_current()

    # The fixture must actually exercise thinning, or this test proves nothing.
    assert len(served) > drv.scroll_captures
    assert drv._current_item_index is not None and drv._current_item_index.usable

    assert len(profile.photos) == drv.scroll_captures
    assert set(profile.photos) <= set(served), "every ranker frame must be a genuine capture"
    # The item index itself must be unaffected: it was built from the FULL capture, before any
    # thinning, so its item/context counts are the ones test_auto_capture_enumerates_the_profile_
    # into_numbered_crops pins regardless of how the ranker's copy was resampled afterward.
    assert len(profile.items) == len(_CARDS) == 4
    assert len(profile.item_context) == 1
    # Coverage, not concentration: the ranker still sees the top and the bottom of the profile,
    # not just the head start the finer enumeration cadence would otherwise give it. The very
    # last SERVED frame is the repeated one that told the loop it had reached the bottom (see
    # _capture_current's `if sig in seen: ... break`), which is deliberately never kept in
    # `photos` at all -- so "covers to the bottom" means within one frame of the end, not
    # literally the last thing screencapped.
    positions = sorted(served.index(frame) for frame in profile.photos)
    assert positions[0] == 0
    assert positions[-1] >= len(served) - 2
    assert positions == sorted(set(positions)), "no frame handed to the ranker twice"


# =====================================================================================
# Every refusal is a stated reason, never a fallback to raw frames
# =====================================================================================

def test_observe_now_enumerates_exactly_as_auto_does():
    """DOC 5.9's INVERSION, at the capture layer. This test is the inverse of the one it replaces.

    Until 2026-08-12 `_item_enumeration_blocker`'s FIRST condition was `_auto_session`, so an
    observe read produced no numbered items at all and said so. That was a policy choice made for
    wall clock (~40 frames instead of 12, on the mode a human waits through) and it was defensible
    only while observe had nothing to do with a payload. Observe now runs auto's whole opener
    pipeline as its canary, so the payload is exactly what it needs -- and an observe read that
    kept generating from raw scroll frames while auto generated from crops would be the
    request-shape divergence that makes testing observe tell the owner nothing about auto.

    Mode is deliberately not an input to enumeration any more: the SAME driver, with only the
    session-policy hook flipped, must produce the SAME numbered list, because that is what
    "observe issues the same request as auto" means one layer down."""
    observing = _drv(WorldAdb(), auto=False)
    auto = _drv(WorldAdb())

    observed = observing._capture_current()
    automatic = auto._capture_current()

    assert observing._item_enumeration_blocker() == ""
    assert observed.items_unavailable == ""
    assert observed.items and observed.items == automatic.items
    assert observed.item_context == automatic.item_context
    assert observing._current_item_payload is not None
    assert observing._current_item_index is not None
    # The enumeration cadence, not the configured read fraction -- the step now follows the card
    # in front of it in observe too (doc 5.10.1's ratio rule).
    assert {frac for frac, _lane in observing._adb.gestures} != {observing.read_scroll_frac}
    assert observing._profile_capture_limit == hinge._ENUMERATION_CAPTURE_LIMIT


def test_openers_disabled_does_not_enumerate_and_says_so_rather_than_going_quiet():
    """Audit fix, "BUG 2" (2026-08-12). Item enumeration exists solely to let the model pick an
    item for an opener -- with opener.enabled: false there is no consumer for a numbered item
    list at all, so reading a profile at the ~40-frame enumeration cadence for it would be pure
    waste, and worse, it would risk stopping a run that was only ever going to send bare likes
    over the ABSENCE of a payload nobody asked for. Not a capability limit (Hinge can always
    attach a comment; accepts_opener stays True) but a policy one.

    Since doc 5.9's inversion this is the ONLY session-level policy gate left -- observe
    enumerates now -- so it also carries the observe case, which is why the test below asserts it
    in both modes rather than only in the auto one it was written for."""
    adb = WorldAdb()
    drv = _drv(adb, openers=False)  # opener.enabled: false, mirroring worker.py's session-start hook

    profile = drv._capture_current()

    assert profile.photos, "the read itself must still produce frames for the ranker"
    assert profile.items == () and profile.item_context == ()
    assert "disabled" in profile.items_unavailable
    assert drv._current_item_payload is None and drv._current_item_index is None
    # Never planned a single enumeration step: the read ran at the ordinary, non-enumerating
    # cadence and ceiling, exactly like observe's.
    assert {frac for frac, _lane in adb.gestures} == {drv.read_scroll_frac}
    assert drv._profile_capture_limit == drv.scroll_captures

    # And the same in OBSERVE, which is the half this gate inherited from doc 5.9's inversion:
    # with no auto session policy at all, openers-off must still be what decides, not the mode.
    observing = _drv(WorldAdb(), auto=False, openers=False)
    observing._capture_current()
    assert "disabled" in observing._current_items_unavailable
    assert observing._profile_capture_limit == observing.scroll_captures


@pytest.mark.parametrize("auto", [False, True], ids=("observe", "auto"))
def test_missing_targeting_calibration_skips_unusable_enumeration_in_every_mode(
        monkeypatch, auto):
    """An enabled provider is not a consumer when targeted text can never be licensed.

    The reported run spent 33 fine-cadence frames enumerating eight items, then the observe
    readiness gate rejected every suggestion because this calibration was absent. The capture
    must stay at the ordinary ceiling and state why no item payload was built.  The blocker is
    intentionally mode-independent: AUTO must also fail before the expensive top confirmation
    or index builder, leaving the worker's existing ``items_unavailable`` hard-stop to withhold
    a targeted like rather than discovering the missing calibration after it has read the card.
    """
    adb = WorldAdb()
    drv = _drv(adb, auto=auto, targeting_calibration=False)

    def enumeration_must_not_start(*_args, **_kwargs):
        raise AssertionError("missing targeting calibration must block before enumeration")

    # Gesture cadence alone would show that enumeration *usually* did not happen.  These two
    # tripwires pin the more important guarantee: neither its scroll-top round trip nor its
    # item-index construction may run at all once the configuration has already ruled out a
    # usable model-item action.
    monkeypatch.setattr(drv, "_confirm_enumeration_top", enumeration_must_not_start)
    monkeypatch.setattr(drv, "_index_captured_items", enumeration_must_not_start)

    profile = drv._capture_current()

    assert profile.photos
    assert profile.items == () and profile.item_context == ()
    assert "targeting_calibration" in profile.items_unavailable
    assert drv._profile_capture_limit == drv.scroll_captures
    assert {frac for frac, _lane in adb.gestures} == {drv.read_scroll_frac}


def test_openers_enabled_by_default_preserves_every_other_enumeration_test():
    """set_opener_enabled defaults to True (a driver nobody calls it on -- an older caller, a
    test double -- keeps enumerating exactly as it did before this hook existed), and this is the
    one assertion that would catch a flipped default silently breaking every other test in this
    file rather than only this one.

    The default matters MORE since doc 5.9's inversion, because this hook became the only
    session-level gate on enumeration: a standalone caller that wants the ordinary 12-frame read
    now has to say so, which is why tools/hinge_bot_scroll_probe.py calls set_opener_enabled(False)
    explicitly instead of getting it from never having been an auto session."""
    class C:
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False,
                          "targeting_calibration": _TARGETING_CALIBRATION}}
    drv = HingeDriver(C())          # deliberately NOT through _drv: nobody calls the hook here
    drv._adb = drv._touch = WorldAdb()
    assert drv._openers_enabled is True
    assert drv._item_enumeration_blocker() == ""


def test_a_read_that_does_not_start_at_a_confirmed_top_refuses_to_enumerate():
    """Doc 5.5: "treat failure to confirm as a hard stop". Counting items from an unconfirmed
    top gives a systematic off-by-N in every heart ordinal, and the geometry cannot contradict
    it reliably (roughly 1 scroll position in 20 looks innocent), so the affirmative
    filter-chips signal is a gate and not a formality. The frames are still returned -- the
    ranker's need for them is untouched by the opener's need for an index."""
    adb = WorldAdb(start=900, at_top=False)
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile.photos, "the read itself must still produce frames for the ranker"
    assert profile.items == ()
    assert "scroll top" in profile.items_unavailable
    assert drv._current_item_payload is None
    # And it never planned a single enumeration step: the read ran at the ordinary cadence.
    assert {frac for frac, _lane in adb.gestures} == {drv.read_scroll_frac}


def test_an_index_that_contradicts_itself_is_reported_and_never_degrades_to_raw_frames():
    """Doc 5.3's hard gate, inherited rather than re-derived: an index whose numbering cannot be
    trusted produces NO item list. The temptation this pins against is sending `profile.photos`
    instead, which would silently reintroduce exactly the frame/item ambiguity doc 5.2 exists to
    remove -- so the answer is a sentence naming the contradiction, and the worker stops on it."""
    drv = _drv(WorldAdb())
    real = hinge.build_item_index

    def _unusable(*a, **k):
        index = real(*a, **k)
        return dataclasses.replace(index, failures=("frame 3 contradicts itself",))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "build_item_index", _unusable)
        profile = drv._capture_current()

    assert profile.items == () and profile.item_context == ()
    assert "frame 3 contradicts itself" in profile.items_unavailable
    assert profile.photos, "the frames are still the ranker's"
    assert drv._current_item_index is None and drv._current_item_payload is None


def test_item_index_refusal_logs_the_broken_pair_and_compact_shift_ledger(tmp_path):
    """A broken correspondence chain used to lose the only two frames that explain it.  The
    refusal record saves that exact pair, every realised delta (with unknown kept as null), and
    per-refusal strip evidence; it remains an ordinary hard refusal, never a usable prefix.
    """
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-refusal")
    real = hinge.build_item_index

    def _broken_pair(*a, **k):
        index = real(*a, **k)
        # The production failure shape: four strips see the same shift while nine animated /
        # changed regions are silent.  The refusal record must preserve that distinction.
        strips = tuple(
            dataclasses.replace(strip, state=("matched" if i < 4 else "weak"))
            for i, strip in enumerate(index.shifts[0].strips))
        assert len(strips) == 13
        first = dataclasses.replace(
            index.shifts[0], delta_px=None, status="no_consensus", consensus_px=209,
            confidence=0.5, agreeing=4, dissenting=0, eligible=4,
            reason="the animation left no coordinate-space consensus", strips=strips)
        return dataclasses.replace(
            index, shifts=(first, *index.shifts[1:]),
            offsets=(0, *([None] * (len(index.offsets) - 1))),
            failures=("frames 0 and 1 could not be put in one coordinate space",))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "build_item_index", _broken_pair)
        profile = drv._capture_current()

    assert profile.items == ()
    records = [json.loads(line) for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    refusal = next(rec for rec in records if rec["action"] == "item_index_refused")
    assert refusal["failing_pair"] == [0, 1]
    assert refusal["steps_px"][0] is None
    assert refusal["refused_pairs"] == [{
        "pair": [0, 1], "status": "no_consensus", "consensus_px": 209,
        "confidence": 0.5, "agreeing": 4, "dissenting": 0, "eligible": 4,
        "strip_states": {"matched": 4, "weak": 9},
        "reason": "the animation left no coordinate-space consensus",
    }]
    assert (drv._dbg.dir / refusal["before"]).read_bytes() == profile.photos[0]
    assert (drv._dbg.dir / refusal["after"]).read_bytes() == profile.photos[1]


def test_item_index_refusal_diagnostics_cannot_break_the_live_refusal():
    """DebugLog promises best-effort operation; retain that safety property even if a future
    logger implementation (or a test double) raises before it gets to DebugLog's own guard."""
    drv = _drv(WorldAdb())

    class _ExplodingLog:
        def action(self, *args, **kwargs):
            raise OSError("debug disk unavailable")

    drv._dbg = _ExplodingLog()
    assert drv._item_index_refused([b"frame 0", b"frame 1"], "index refused") == "index refused"


def test_item_index_refusal_saves_at_most_eight_source_mapped_frames_and_bounded_sidecar(tmp_path):
    """The actual cited capture frames, not just a pair thumbnail, make a later refusal
    diagnosable; the dossier stays bounded even when a malformed reason cites every frame."""
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-dossier")

    class _Edge:
        observed = True

    class _Block:
        y0, y1, complete = 20, 60, True
        top, bottom = _Edge(), _Edge()
        hearts = ((900, 42),)

    class _Seg:
        blocks = (_Block(),)

    class _Index:
        frames = (_Seg(),) * 11
        source_frame_indices = tuple(range(11))
        offsets = tuple(i * 100 for i in range(11))
        shifts = ()
        failures = ()

    photos = [f"frame-{i}".encode() for i in range(11)]
    reason = "; ".join(f"frame {i} contributed contradictory geometry" for i in range(11))
    assert drv._item_index_refused(photos, reason, _Index()) == reason

    records = [json.loads(line) for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    refusal = next(rec for rec in records if rec["action"] == "item_index_refused")
    assert len(refusal["evidence_frames"]) == 8
    assert all("_frame_" in name for name in refusal["evidence_frames"])
    runtime = refusal["item_index_runtime"]
    assert runtime["algorithm_id"] == item_index.ITEM_INDEX_ALGORITHM_ID
    assert runtime["module_path"].endswith("operation_love/drivers/item_index.py")
    assert len(runtime["indexer_code_sha256"]) == 64
    assert len(runtime["splitter_code_sha256"]) == 64
    sidecar = json.loads((drv._dbg.dir / refusal["evidence_sidecar"]).read_text())
    assert sidecar["schema_version"] == 6
    assert sidecar["runtime"] == runtime
    assert len(sidecar["frames"]) == 8
    assert len(sidecar["all_frame_geometry"]) == 11
    assert sidecar["pair_evidence"] == []
    frame = sidecar["frames"][0]
    assert {"local_frame_index", "source_frame_index", "offset_px", "animation_marker",
            "video_mute_markers", "blocks", "background_runs"} <= set(frame)
    assert frame["animation_marker"] is None
    assert frame["video_mute_markers"] == []
    assert frame["blocks"][0]["frame_rows"] == [20, 60]
    assert frame["blocks"][0]["page_rows"] == [20, 60]
    assert frame["blocks"][0]["complete"] is True
    assert frame["blocks"][0]["top_observed"] is True
    assert frame["blocks"][0]["bottom_observed"] is True
    assert frame["blocks"][0]["hearts"] == {
        "frame_rows": [[900, 42]], "page_rows": [[900, 42]],
    }


def test_item_index_refusal_sidecar_keeps_measured_pair_strips_and_lookahead(tmp_path):
    """A downstream measured contradiction remains diagnosable after an earlier refusal."""
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-pair-ledger")
    photos = [_frame(260 * i) for i in range(5)]
    index = item_index.build_item_index(
        photos, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)
    assert index.usable, index.failures

    refused = dataclasses.replace(
        index.shifts[1], delta_px=None, consensus_px=None, status="no_consensus",
        reason="synthetic refusal whose measured look-ahead must survive")
    diagnostic_index = dataclasses.replace(
        index, shifts=(index.shifts[0], refused, *index.shifts[2:]))
    drv._item_index_refused(
        photos, "frames 1 and 2 could not be put in one coordinate space", diagnostic_index)

    actions = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    action = next(record for record in actions if record["action"] == "item_index_refused")
    # One anchor before the failed pair and two look-ahead frames after its right endpoint.
    assert sorted(int(Path(name).stem.rsplit("_", 1)[1]) for name in action["evidence_frames"]) \
        == [0, 1, 2, 3, 4]
    sidecar = json.loads((drv._dbg.dir / action["evidence_sidecar"]).read_text())
    assert sidecar["schema_version"] == 6
    assert len(sidecar["pair_evidence"]) == 4
    measured_lookahead = sidecar["pair_evidence"][2]
    assert measured_lookahead["status"] == "measured"
    assert isinstance(measured_lookahead["delta_px"], int)
    assert measured_lookahead["strips"]
    assert {"frame_rows", "state", "delta_px", "score", "runner_up", "stddev", "search"} \
        <= set(measured_lookahead["strips"][0])
    assert measured_lookahead["structural_landmarks"]
    assert {"before", "after"} == set(measured_lookahead["observed_gutters"])
    assert "local_proposals" in measured_lookahead
    assert "v12_video_track_delta_px" in measured_lookahead
    assert measured_lookahead["v12_video_track_delta_px"] is None
    refused_pair = sidecar["pair_evidence"][1]
    assert any(proposal["kind"] == "exact_multi"
               for proposal in refused_pair["local_proposals"])


def test_item_index_runtime_provenance_hashes_loaded_indexer_not_worktree_source(
        monkeypatch, tmp_path):
    """An imported indexer is what a refusal ran, not whatever its file contains now."""
    baseline = hinge._item_index_runtime_provenance()
    assert len(baseline["indexer_code_sha256"]) == 64

    # A source path/content change alone cannot alter code already loaded into this process.
    source_copy = tmp_path / "item_index.py"
    source_copy.write_text("before import-time code")
    monkeypatch.setattr(item_index, "__file__", str(source_copy))
    from_source_path = hinge._item_index_runtime_provenance()
    source_copy.write_text("edited after import; executable code is still unchanged")
    after_worktree_edit = hinge._item_index_runtime_provenance()
    assert from_source_path["indexer_code_sha256"] == baseline["indexer_code_sha256"]
    assert after_worktree_edit["indexer_code_sha256"] == baseline["indexer_code_sha256"]

    # In contrast, a live helper replacement changes behavior and must be visible.  `_assemble`
    # is an explicit member of the indexer fingerprint, so no source reload is needed to prove it.
    def changed_assemble(*_args, **_kwargs):
        return (), (), ()

    monkeypatch.setattr(item_index, "_assemble", changed_assemble)
    helper_changed = hinge._item_index_runtime_provenance()
    assert helper_changed["indexer_code_sha256"] != baseline["indexer_code_sha256"]

    # The v9 five-pair boundary helper is an explicit decision stage too.  It may be replaced in
    # a live process independently of `build_item_index`, so provenance must name it directly.
    with monkeypatch.context() as patched:
        patched.setattr(item_index, "_exact_multi_strip_shift", changed_assemble)
        assert hinge._item_index_runtime_provenance()["indexer_code_sha256"] \
            != helper_changed["indexer_code_sha256"]

    # Loaded calibration globals are part of the same contract, rather than an invisible
    # behavioural input outside the code-object digest.
    monkeypatch.setattr(item_index, "_MAX_STEP_PX", item_index._MAX_STEP_PX + 1)
    calibration_changed = hinge._item_index_runtime_provenance()
    assert calibration_changed["indexer_code_sha256"] != helper_changed["indexer_code_sha256"]


def test_item_index_runtime_provenance_reaches_the_helpers_below_its_leaf_dependencies(
        monkeypatch):
    """Naming the stages explicitly stopped one call-frame short of the real deciders.

    `segment_frame`, `estimate_shift` and `capture_profile_identity` are thin orchestrators. The
    row classifier, the strip matcher and the scroll-top confirmer beneath them decide as much of
    a refusal as anything in `item_index`, and hashing only the three entry points left a process
    running a changed row classifier reporting a byte-identical fingerprint — the same blind spot
    the digest exists to close, moved one level down.
    """
    from operation_love.drivers import frameshift

    baseline = hinge._item_index_runtime_provenance()["indexer_code_sha256"]

    def changed(*_args, **_kwargs):
        raise AssertionError("never called; only its code object is fingerprinted")

    # One helper per cross-module dependency, each private to a module `item_index` only ever
    # reaches THROUGH its named leaf.
    for module, name in ((segment, "_classify_rows"),
                         (frameshift, "_resolve"),
                         (item_identity, "confirm_scroll_top")):
        assert hasattr(module, name), f"{module.__name__}.{name} moved; pick its replacement"
        with monkeypatch.context() as patched:
            patched.setattr(module, name, changed)
            assert hinge._item_index_runtime_provenance()["indexer_code_sha256"] != baseline, (
                f"a changed {module.__name__}.{name} must not report an unchanged indexer")
    assert hinge._item_index_runtime_provenance()["indexer_code_sha256"] == baseline

    # An unused import must still not make the fingerprint drift: the walk follows what the
    # loaded code actually reads, rather than sweeping each module.
    monkeypatch.setattr(item_index, "_an_unused_helper", changed, raising=False)
    assert hinge._item_index_runtime_provenance()["indexer_code_sha256"] == baseline


def test_item_index_runtime_provenance_survives_a_self_referential_global(monkeypatch):
    """Diagnostics must degrade to a value, never to an absent dict on a live refusal.

    The whole body is wrapped in a blanket ``except``, so an unbounded walk into a container that
    contains itself would not raise — it would quietly return all-None provenance for a refusal
    that most needs to say which code produced it.
    """
    baseline = hinge._item_index_runtime_provenance()
    assert baseline["indexer_code_sha256"] is not None

    # `_MAX_STEP_PX` is read inside a function BODY, so it reaches the globals walk. Prove that
    # here rather than assuming it: a constant that only supplies a keyword default would NOT
    # reach it (those are captured off the loaded function's `__defaults__`, where a post-import
    # rebinding correctly changes nothing), and this test would then pass without exercising
    # anything at all.
    with monkeypatch.context() as patched:
        patched.setattr(item_index, "_MAX_STEP_PX", item_index._MAX_STEP_PX + 1)
        assert (hinge._item_index_runtime_provenance()["indexer_code_sha256"]
                != baseline["indexer_code_sha256"]), "pick a global the walk actually reads"

    cyclic: dict = {"frames": 3}
    cyclic["self"] = cyclic
    monkeypatch.setattr(item_index, "_MAX_STEP_PX", cyclic)

    provenance = hinge._item_index_runtime_provenance()
    assert provenance["indexer_code_sha256"] is not None
    assert provenance["algorithm_id"] == item_index.ITEM_INDEX_ALGORITHM_ID


def test_item_index_refusal_filesystem_evidence_failure_cannot_break_the_live_refusal(
        monkeypatch, tmp_path):
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-dossier-disk-failure")
    monkeypatch.setattr(Path, "write_bytes", lambda *_a, **_k: (_ for _ in ()).throw(OSError("full")))
    assert drv._item_index_refused([b"first", b"last"], "frame 0 is unusable") == "frame 0 is unusable"


def test_item_index_repair_notes_log_original_capture_frame_after_omission_recovery():
    drv = _drv(WorldAdb())

    class _Debug:
        calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    class _Index:
        frames = (object(), object(), object())
        source_frame_indices = (0, 2, 3)
        notes = ("frame 1's sighting at page rows 600..900 spans a proven boundary",)
        video_mute_markers = (item_index.VideoMuteMarker(
            frame_index=1, x=106, y=876, score=1.0),)
        repair_provenance = (item_index.ItemIndexRepair(
            pair_index=1, path="v12_mute_card_track",
            raw_status="no_consensus", raw_delta_px=None,
            effective_status="measured", effective_delta_px=234,
            marker_frames=(1,)),)

    debug = _Debug()
    drv._dbg = debug
    photos = [b"zero", b"one", b"two", b"three"]
    drv._record_item_index_notes(photos, _Index())

    assert debug.calls == [("item_index_repaired", {
        "before": b"two",
        "notes": ["source frame 2 (index frame 1)'s sighting at page rows 600..900 spans a proven boundary"],
        "note_frames": [{"local_frame_index": 1, "source_frame_index": 2}],
        "source_frame_indices": [0, 2, 3],
        "item_index_runtime": hinge._item_index_runtime_provenance(),
        "repairs": [{
            "path": "v12_mute_card_track",
            "local_pair": [1, 2], "source_pair": [2, 3],
            "raw": {"status": "no_consensus", "delta_px": None},
            "effective": {"status": "measured", "delta_px": 234},
            "mute_markers": [{
                "local_frame_index": 1, "source_frame_index": 2,
                "x": 106, "y": 876, "score": 1.0,
            }],
        }],
    })]


@pytest.mark.parametrize(
    ("source_indices", "bridge", "expected_omitted", "expected_before", "expected_after"),
    [((0, 2, 3), (0, 2), [1], b"zero", b"two"),
     ((0, 1, 3), (1, 3), [2], b"one", b"three")],
    ids=("omit_first_failed_frame", "omit_second_failed_frame"))
def test_item_index_recovery_log_uses_the_actual_omitted_side_and_keeps_failure_evidence(
        source_indices, bridge, expected_omitted, expected_before, expected_after):
    """Either side of a failed pair can be the bad intermediate frame, not just its second."""
    drv = _drv(WorldAdb())

    class _Debug:
        calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    class _FailedShift:
        status = "no_consensus"
        reason = "two exact witnesses, below quorum"
        agreeing = 2
        dissenting = 1
        eligible = 2

    class _RecoveredIndex:
        source_frame_indices = source_indices
        recovered_from_pair = (1, 2)
        recovery_bridge = bridge
        recovery_failed_shift = _FailedShift()
        recovery_reason = "fresh bridge measured"

    debug = _Debug()
    drv._dbg = debug
    photos = [b"zero", b"one", b"two", b"three"]
    drv._record_item_index_recovery(photos, _RecoveredIndex())

    assert debug.calls == [("item_index_recovered", {
        "before": expected_before, "after": expected_after,
        "anchor": photos[expected_omitted[0]], "recovered_from_pair": [1, 2],
        "omitted_frame_indices": expected_omitted, "recovery_bridge": list(bridge),
        "original_status": "no_consensus", "original_reason": "two exact witnesses, below quorum",
        "original_agreeing": 2, "original_dissenting": 1, "original_eligible": 2,
        "recovery_reason": "fresh bridge measured",
    })]


def test_two_pair_recovery_log_retains_both_refusals_and_the_omitted_frame():
    """A transient can poison both neighbours; the dossier must not collapse that to one."""
    drv = _drv(WorldAdb())

    class _Debug:
        calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    class _FailedShift:
        def __init__(self, reason, agreeing):
            self.status = "no_consensus"
            self.reason = reason
            self.agreeing = agreeing
            self.dissenting = 0
            self.eligible = agreeing + 1

    first = _FailedShift("left side of transient", 2)
    second = _FailedShift("right side of transient", 2)

    class _RecoveredIndex:
        source_frame_indices = (0, 1, 3)
        recovered_from_pair = (1, 2)
        recovered_from_pairs = ((1, 2), (2, 3))
        recovery_bridge = (1, 3)
        recovery_failed_shift = first
        recovery_failed_shifts = (first, second)
        recovery_reason = "fresh bridge measured"

    debug = _Debug()
    drv._dbg = debug
    photos = [b"zero", b"one", b"transient", b"three"]
    drv._record_item_index_recovery(photos, _RecoveredIndex())

    assert len(debug.calls) == 1
    name, fields = debug.calls[0]
    assert name == "item_index_recovered"
    assert fields["before"] == b"one" and fields["after"] == b"three"
    assert fields["anchor"] == b"transient"
    assert fields["recovered_from_pairs"] == [[1, 2], [2, 3]]
    assert [failure["reason"] for failure in fields["original_failures"]] == [
        "left side of transient", "right side of transient"]


def test_recovered_index_crops_only_the_exact_rebuilt_frame_sequence(monkeypatch):
    """An omitted frame must not offset every later crop onto the wrong source image."""
    drv = _drv(WorldAdb())
    photos = [b"zero", b"one", b"two", b"three"]
    seen = []

    class _Index:
        usable = True
        frames = (object(), object(), object())
        source_frame_indices = (0, 2, 3)
        identity = type("Identity", (), {"known": True, "reason": "ok"})()
        recovered_from_pair = None
        recovery_bridge = None

    payload = type("Payload", (), {"usable": True})()
    monkeypatch.setattr(hinge, "build_item_index", lambda *_args, **_kwargs: _Index())

    def crop_only_indexed_frames(frames, index, **_kwargs):
        seen.append((list(frames), index))
        return payload

    monkeypatch.setattr(hinge, "build_item_payload", crop_only_indexed_frames)

    assert drv._index_captured_items(photos) == ""
    assert seen == [([b"zero", b"two", b"three"], drv._current_item_index)]
    assert drv._current_item_payload is payload


def test_the_index_carries_this_profiles_identity_and_navigation_can_check_it():
    """Doc 5.7's carried-forward requirement 1, at the point the table is built. The index knows
    WHOSE profile it describes, fingerprinted from the frames it was built from through the
    driver's own sticky-header primitive, and the fingerprint really is the header this world
    painted rather than anything about the cards."""
    adb = WorldAdb()
    drv = _drv(adb)

    drv._capture_current()

    identity = drv._current_item_index.identity
    assert identity.known and identity.band == tuple(drv.identity_band)
    assert identity.frame_index >= 1                   # frame 0 is the filter-chips row
    assert identity.fingerprint == scroll_top.band_fingerprint(
        _frame(600), identity_band=_IDENTITY_BAND, grid=item_identity._IDENTITY_GRID)

    # It is the driver's OWN anchor, not a second scheme: the same array `_identity_sig` holds.
    assert list(identity.fingerprint) == [int(v) for v in drv._identity_sig.reshape(-1)]

    # And it answers the question navigation asks: this person yes, the other person no.
    assert item_identity.compare_profile_identity(
        _frame(600), identity, identity_band=drv.identity_band).matched
    assert item_identity.compare_profile_identity(
        _frame(600, header=_OTHER_HEADER_VALUE), identity,
        identity_band=drv.identity_band).mismatched


def test_a_capture_with_no_identity_is_refused_before_a_billed_opener_call():
    """An index that cannot say whose profile it describes is one `item_nav` will refuse at its
    entry gate, so producing crops from it would buy an opener for a card that can never be
    targeted. Refused here instead, on the same terms as every other enumeration failure: a
    sentence on the Profile, the ranker's frames untouched, and worker.py's stop before the model
    is asked anything.

    The world here keeps drawing the filter-chips row at every scroll offset, so the page really
    is read and really does index -- and the one strip that says who is on it never shows a
    person. That is a capture in which nothing distinguishes this profile from any other."""
    adb = WorldAdb(at_top=True)
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile is not None and profile.photos, "the frames are still the ranker's"
    assert profile.items == () and profile.item_context == ()
    assert "identity" in profile.items_unavailable
    assert "never appeared" in profile.items_unavailable
    assert drv._current_item_index is None and drv._current_item_payload is None

    # The control: the identical read with the header drawn is enumerated normally, so what
    # refused above is the identity and not the world.
    assert _drv(WorldAdb())._capture_current().items


def test_a_dependency_that_raises_becomes_the_same_stated_reason(tmp_path):
    """The three leaf modules raise rather than answer when they cannot look at all
    (`SegmentationError` on a frame that will not decode, `ShiftEstimationError` when no page
    space spans the capture, `ItemIndexError`/`ItemCropError` on a capture that cannot be
    indexed, including a missing cv2/numpy). None of them may take down a read whose frames the
    ranker still wants, and none of them may pass silently."""
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-exception")

    def _boom(*a, **k):
        raise segment.SegmentationError("frame 0 could not be decoded")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "build_item_index", _boom)
        profile = drv._capture_current()

    assert profile.items == ()
    assert "SegmentationError" in profile.items_unavailable
    assert "frame 0 could not be decoded" in profile.items_unavailable
    assert profile.photos
    records = [json.loads(line) for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    refusal = next(rec for rec in records if rec["action"] == "item_index_refused")
    assert refusal["steps_px"] == [] and refusal["refused_pairs"] == []
    assert "failing_pair" not in refusal
    assert (drv._dbg.dir / refusal["before"]).read_bytes() == profile.photos[0]
    assert (drv._dbg.dir / refusal["after"]).read_bytes() == profile.photos[-1]


def test_a_scroll_that_cannot_be_sized_ends_the_enumeration_without_ending_the_read():
    """`plan_scroll_step` raises rather than substituting a larger step when the local card
    spacing is too small for any permitted gesture to respect the ratio rule. That must end the
    ENUMERATION (no crops, a stated reason) while the read finishes normally -- the frames have
    a second consumer that has no index in them to be wrong about."""
    adb = WorldAdb()
    drv = _drv(adb)

    def _refuse(*a, **k):
        raise scroll_step.ScrollStepError("the local card spacing is 400px")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_scroll_step", _refuse)
        profile = drv._capture_current()

    assert profile.items == ()
    assert "the local card spacing is 400px" in profile.items_unavailable
    assert len(profile.photos) > 1, "the read must have carried on to the bottom"
    assert {frac for frac, _lane in adb.gestures} == {drv.read_scroll_frac}


# =====================================================================================
# Invalidation: the table lives exactly one profile (doc 5.3)
# =====================================================================================

def test_the_deck_advancing_mid_read_leaves_no_table_behind():
    """Doc 5.3 names this path explicitly. A capture that spans two people is discarded (the
    frames would mean-pool two faces into one label), and the item state must go with it -- an
    index from the previous profile surviving into the recapture is the stale table a wrong like
    is built from, and at model index 1 nothing before the tap catches it."""
    adb = WorldAdb(header_after=(2, _OTHER_HEADER_VALUE))
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile is None, "a capture spanning two cards is discarded, as it always was"
    assert drv._current_capture_split is True
    assert drv._current_item_index is None and drv._current_item_payload is None
    assert "deck advanced" in drv._current_items_unavailable


def test_post_scroll_settle_keeps_a_header_transition_from_splitting_one_profile(monkeypatch):
    """A capture frame taken while Hinge animates its sticky header is not a new card.

    Before the post-scroll settle, the temporary value became `_identity_sig` and the stable
    header on the next frame falsely hit capture_split.  The settle must happen before that
    next frame is read, without softening the actual boundary rule.
    """
    class TransitionAdb(WorldAdb):
        def __init__(self):
            super().__init__()
            self.settled = False
            self._transition_seen = False

        def screencap(self):
            if self.scrolls == 0:
                return _frame(0)
            if self.scrolls == 1 and not self.settled and not self._transition_seen:
                self._transition_seen = True
                return _frame(self.scroll, header=60)  # neither chips nor stable header
            return _frame(self.scroll, header=_HEADER_VALUE)

    adb = TransitionAdb()
    drv = _drv(adb, auto=False, openers=False, scroll_captures=3)

    def _sleep(_seconds, _should_stop=None):
        # The ordinary dwell happens before the first scroll.  The new settle is the first
        # wait after it, and is what lets this fake UI finish its header transition.
        if adb.scrolls:
            adb.settled = True
        return True

    monkeypatch.setattr(drv, "_interruptible_sleep", _sleep)
    profile = drv._capture_current()

    assert profile is not None
    assert drv._current_capture_split is False
    assert np.all(drv._identity_sig == _HEADER_VALUE)


def test_transient_anchor_then_stable_header_on_same_profile_does_not_split():
    """Regression for the 2026-08-13 Shuman false capture_split.

    The transition survives the settle and becomes the first identity candidate. On the next
    read frame the real sticky header appears, but the scrolling content still aligns with the
    previous frame and proves this is one card. The candidate is replaced, then confirmed.
    """
    class SlowTransitionAdb(WorldAdb):
        def screencap(self):
            if self.scrolls == 0:
                return _frame(0)
            if self.scrolls == 1:
                return _frame(self.scroll, header=60)
            return _frame(self.scroll, header=_HEADER_VALUE)

    class Debug:
        def __init__(self):
            self.calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    adb = SlowTransitionAdb()
    drv = _drv(adb, auto=False, openers=False, scroll_captures=5)
    dbg = Debug()
    drv._dbg = dbg

    profile = drv._capture_current()

    assert profile is not None
    assert drv._current_capture_split is False
    assert drv._identity_anchor_confirmed is True
    assert np.all(drv._identity_sig == _HEADER_VALUE)
    replacement = next(fields for name, fields in dbg.calls
                       if name == "identity_anchor_replaced")
    assert replacement["old_anchor_frame_index"] == 1
    assert replacement["new_anchor_frame_index"] == 2
    assert replacement["content_overlap_rows"] > 0


def test_real_header_change_still_splits_before_the_foreign_frame_is_appended():
    """Settling is not a relaxation: a stable foreign header remains a hard boundary."""
    adb = WorldAdb(header_after=(2, _OTHER_HEADER_VALUE))
    drv = _drv(adb)

    assert drv._capture_current() is None
    assert drv._current_capture_split is True
    assert len(drv._current_sigs) == 2  # the foreign trigger was rejected before append
    assert np.all(drv._identity_sig == _HEADER_VALUE)


def test_stop_during_post_scroll_settle_abandons_without_another_screencap(monkeypatch):
    """The added settle has the same hard Stop boundary as the read dwell."""
    class CountingAdb(WorldAdb):
        def __init__(self):
            super().__init__()
            self.captures = 0

        def screencap(self):
            self.captures += 1
            return super().screencap()

    adb = CountingAdb()
    drv = _drv(adb, auto=False, openers=False, scroll_captures=3)
    calls = 0

    def _sleep(_seconds, _should_stop=None):
        nonlocal calls
        calls += 1
        return calls == 1  # read dwell completes; post-scroll settle is interrupted

    monkeypatch.setattr(drv, "_interruptible_sleep", _sleep)
    assert drv._capture_current() is None
    assert adb.scrolls == 1
    assert adb.captures == 1
    assert len(drv._capture_scroll_ledger) == 1


def test_capture_split_debug_record_keeps_the_trigger_frame_and_distances():
    """A split record must preserve the rejected frame, not only capture frame zero."""
    class Debug:
        def __init__(self):
            self.calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    drv = _drv(WorldAdb(header_after=(2, _OTHER_HEADER_VALUE)))
    dbg = Debug()
    drv._dbg = dbg

    assert drv._capture_current() is None
    name, fields = next(call for call in dbg.calls if call[0] == "capture_split")
    assert name == "capture_split"
    assert fields["before"] != fields["after"]
    assert fields["anchor"] != fields["after"]
    assert fields["trigger_frame_index"] == fields["captured_frames"] == 2
    assert fields["read_scrolls"] == 2
    assert fields["identity_dist"] >= drv.change_threshold
    assert fields["top_dist"] >= drv.change_threshold
    assert fields["scroll_top_state"] == scroll_top.SCROLL_TOP_REFUTED
    assert fields["scroll_top_distance"] >= drv.change_threshold
    assert fields["identity_anchor_frame_index"] == 1
    assert fields["identity_anchor_confirmed"] is False
    assert fields["content_match"] is False


def test_current_profile_rewinds_the_new_card_after_a_mid_read_deck_advance():
    """The public observe capture path must not recapture the advanced card mid-scroll.

    A split is intentionally returned as ``None`` so the worker discards the mixed profile and
    recaptures.  That recapture still has to start from Hinge's chips row: otherwise the strict
    scroll-top gate rightly refuses enumeration as ``confirmed_not_top`` and the card falls back
    to a truncated, unnumbered capture (the 2026-08-13 incident).
    """
    class RewindingWorldAdb(WorldAdb):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reverse_swipes = 0

        def swipe(self, _x1, y1, _x2, y2, **_kwargs):
            # `_scroll_to_top` is the only test path that swipes down.  Mirror the fake
            # transport's forward model so its settle screenshot can establish a real top.
            if y2 > y1:
                self.reverse_swipes += 1
                self.scroll = max(
                    0, self.scroll - scroll_step.step_px_for_frac((y2 - y1) / _H, _H))

    adb = RewindingWorldAdb(header_after=(2, _OTHER_HEADER_VALUE))
    drv = _drv(adb)
    drv._session_top_done = True       # isolate the split recovery from the once-per-run pass

    assert drv.current_profile() is None
    assert drv._current_capture_split is True
    assert adb.reverse_swipes > 0
    assert adb.scroll == 0

    profile = drv.current_profile()

    assert profile is not None
    assert profile.items
    assert profile.items_unavailable == ""
    assert profile.items_truncated is False


def test_next_profile_rewinds_only_a_split_then_enumerates_the_new_card():
    """Auto leaves a successful capture at the bottom for bottom-up navigation, but a split
    has no usable index to navigate.  Its immediate retry must therefore restore the new card
    to Hinge's chips row first, just like observe's retry.
    """
    class RewindingWorldAdb(WorldAdb):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reverse_swipes = 0

        def swipe(self, _x1, y1, _x2, y2, **_kwargs):
            if y2 > y1:
                self.reverse_swipes += 1
                self.scroll = max(
                    0, self.scroll - scroll_step.step_px_for_frac((y2 - y1) / _H, _H))

    adb = RewindingWorldAdb(header_after=(2, _OTHER_HEADER_VALUE))
    drv = _drv(adb)
    drv._session_top_done = True       # isolate recovery from the once-per-session top pass

    assert drv.next_profile() is None
    assert drv._current_capture_split is True
    assert adb.reverse_swipes > 0
    assert adb.scroll == 0

    profile = drv.next_profile()

    assert profile is not None
    assert profile.items
    assert profile.items_unavailable == ""
    # Unlike `current_profile`, the successful auto read is deliberately left at bottom for
    # counting navigation.  The recovery was split-only, not a hidden auto-path unwind.
    assert adb.scroll > 0


def test_a_fresh_read_drops_the_previous_profiles_table_before_it_starts():
    """The reset that sits alongside `_current_sigs`: whatever is on screen now belongs to
    someone else, so the table describing the last person must not survive into this read even
    for the duration of it."""
    drv = _drv(WorldAdb())
    drv._capture_current()
    assert drv._current_item_payload is not None

    seen = {}
    real = hinge.build_item_index

    def _peek(*a, **k):
        seen["payload_during_read"] = drv._current_item_payload
        seen["reason_during_read"] = drv._current_items_unavailable
        return real(*a, **k)

    drv._adb = drv._touch = WorldAdb()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "build_item_index", _peek)
        drv._capture_current()

    assert seen["payload_during_read"] is None
    assert "not finished" in seen["reason_during_read"]


@pytest.mark.parametrize("action", ["like", "dislike"])
def test_an_action_that_advances_the_deck_drops_the_table_even_when_it_raises(action):
    """The other place the profile on screen changes. Invalidated in a `finally` deliberately:
    a failed action is when a stale table would survive longest, because nobody then knows where
    the deck is."""
    drv = _drv(WorldAdb())
    drv._capture_current()
    assert drv._current_item_payload is not None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HingeDriver, "_like_comment_sheet",
                   lambda *a, **k: (_ for _ in ()).throw(hinge.HingeActionError("boom")))
        mp.setattr(HingeDriver, "_deliver_decision",
                   lambda *a, **k: (_ for _ in ()).throw(hinge.HingeActionError("boom")))
        with pytest.raises(hinge.HingeActionError):
            drv.like("hi", 0) if action == "like" else drv.dislike()

    assert drv._current_item_index is None and drv._current_item_payload is None
    assert "deck advanced" in drv._current_items_unavailable


def test_a_stopped_read_is_not_reported_as_an_enumerated_one():
    """A Stop mid-read abandons the capture and returns None. The table must not be left holding
    a half-profile's items, which would then be navigated by on the next run."""
    drv = _drv(WorldAdb())
    drv._capture_current()
    assert drv._current_item_payload is not None

    calls = {"n": 0}

    def _stop():
        calls["n"] += 1
        return calls["n"] > 2

    drv._adb = drv._touch = WorldAdb()
    assert drv._capture_current(_stop) is None
    assert drv._current_item_index is None and drv._current_item_payload is None


# =====================================================================================
# The bottom of a profile that is still ANIMATING (live 2026-08-16, profile "Grace")
# =====================================================================================

_VIDEO_TEXTURE = cv2.GaussianBlur(
    np.random.default_rng(_SEED).integers(
        0, 256, (4000, _CARD_X1 - _CARD_X0), dtype=np.uint8), (9, 9), 0)
# The video lives INSIDE the last card, in PAGE coordinates, clear of both gutters and of the
# heart glyph near that card's bottom -- a video is part of a card, so it scrolls with the card
# and only its own content moves once the page has stopped. Painting it at fixed FRAME rows
# instead would smear it across a card boundary, weld two cards into one block and make the
# segmenter refuse for a reason that has nothing to do with what these tests are about.
_VIDEO_PAGE_ROWS = (_CARDS[-1][1] + 30, _CARDS[-1][2] - _HEART_ABOVE_BOTTOM - 80)
# At the clamped scroll this lands at frame rows 1117..1737, inside the analysed band.
_VIDEO_ROWS = (_VIDEO_PAGE_ROWS[0] - _MAX_SCROLL, _VIDEO_PAGE_ROWS[1] - _MAX_SCROLL)


def _animated(frame: bytes, tick: int, *, rows: tuple[int, int] = _VIDEO_ROWS,
              parallax: int = 14) -> bytes:
    """`frame` with one autoplaying video card painted over `rows`.

    Two properties, and the tests below need both.

    It REPAINTS, so no two frames of a motionless page are ever byte-identical -- which is the
    whole mechanism of the incident, since `_frame_sig` compares a whole-frame 24x24 downsample
    for EXACT equality and one animating region therefore keeps every frame "new" forever.

    And its motion is NOT RIGID: rows nearer the bottom of the patch travel further per tick, the
    way anything with depth in it does. That is what makes the estimator's strips over a video
    land on a SPREAD of wrong offsets rather than one shared wrong offset, and the distinction is
    load-bearing rather than decorative -- a patch that translated rigidly would be genuinely
    indistinguishable from a page scroll by pixel evidence alone, and `_exact_cluster_shift` is
    supposed to refuse that case, not rescue it. Live, the five video strips reported +73..+81
    against the page's unanimous +145; this fixture reproduces that shape.
    """
    gray = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_GRAYSCALE)
    r0, r1 = rows
    height = r1 - r0
    # Wrapped rather than clamped, so the patch keeps animating for any tick count a future
    # test might reach.  A clamp would quietly stop the video and hand the byte-identical
    # signal back the bottom detection these tests exist to take away from it.
    span = len(_VIDEO_TEXTURE) - height
    for y in range(height):
        start = (1000 - tick * (61 + round(parallax * y / height))) % span
        gray[r0 + y, _CARD_X0:_CARD_X1] = _VIDEO_TEXTURE[start + y]
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    return buf.tobytes()


class _VideoWorldAdb(WorldAdb):
    """The same arithmetic phone, plus a video card that repaints on every screencap."""

    def __init__(self, *, video_page_rows=_VIDEO_PAGE_ROWS, **kw):
        super().__init__(**kw)
        self._video_page_rows = video_page_rows
        self._tick = 0

    def screencap(self):
        self._tick += 1
        # Page rows -> this frame's rows, clipped to the part of the card actually on screen.
        p0, p1 = self._video_page_rows
        r0, r1 = max(_BAND0, p0 - self.scroll), min(_BAND1, p1 - self.scroll)
        frame = super().screencap()
        return frame if r1 - r0 < 64 else _animated(frame, self._tick, rows=(r0, r1))


def test_a_repainting_video_defeats_the_byte_identical_bottom_signal():
    """The premise of the incident, pinned on its own so the two tests below cannot both pass
    for the wrong reason. A motionless page under an autoplaying video produces frames that are
    all DIFFERENT to the byte, so `seen` can never recognise the bottom."""
    still = _frame(_MAX_SCROLL)
    sigs = {hinge._frame_sig(_animated(still, tick)) for tick in range(6)}

    assert hinge._frame_sig(still) == hinge._frame_sig(still)   # the check itself works
    assert len(sigs) == 6, "the animated frames must be mutually distinct, or nothing is proven"


def test_a_static_pair_is_the_bottom_only_on_an_affirmative_zero_measurement():
    """The predicate that replaces the byte test. It must say "bottom" for a page that did not
    move under a repainting video, and must NOT say it for a page that did move -- and its
    default on any uncertainty is False, because a wrong True truncates a real profile."""
    still = _frame(_MAX_SCROLL)
    a, b = _animated(still, 1), _animated(still, 2)

    assert hinge._static_pair_is_the_bottom(a, b, _CONTENT_BAND)
    assert not hinge._static_pair_is_the_bottom(
        a, _animated(_frame(_MAX_SCROLL - 240), 3), _CONTENT_BAND)
    # "I cannot tell" is never "bottom": undecodable bytes must not end a read.
    assert not hinge._static_pair_is_the_bottom(b"not a png", b"nor this", _CONTENT_BAND)


def test_the_read_stops_at_a_bottom_it_cannot_see_by_bytes_and_does_not_claim_truncation():
    """The incident itself. Grace's profile reached its bottom at frame 37 with a video still
    playing; the byte-identical signal never fired, so the read ran its full 64-frame ceiling,
    issued 26 futile swipes at a page that could not move, and then reported the profile as
    LONGER than the read could cover -- into the operator's banner and BigQuery's
    `capture_truncated` column. Every part of that is the opposite of what happened.

    Two consecutive MEASURED 0px pairs now end the read, so the swipes stop, and because the
    loop leaves by `break` rather than exhausting its `range`, the truncation branch does not
    run: the capture is correctly recorded as having reached the profile's end.
    """
    adb = _VideoWorldAdb()
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile is not None
    assert adb.scroll == _MAX_SCROLL, "the fixture must actually reach its own bottom"
    assert adb.scrolls < hinge._ENUMERATION_CAPTURE_LIMIT - 1, (
        "the read kept swiping at a page that could not move")
    assert not drv._current_capture_truncated
    assert "capture_truncated" not in (profile.meta or {}) or not profile.meta["capture_truncated"]


def test_one_swallowed_gesture_alone_never_ends_a_read():
    """The safety direction of `_STATIC_PAIRS_FOR_BOTTOM`. A single read-scroll that fails to
    move the page -- swallowed by an animation, lost by the input transport -- is not the bottom,
    and ending the read on it would silently truncate a real profile: the model would then pick
    an item from a list that stops part way down. One stall costs one extra frame and the read
    carries on to the real bottom.

    The stall is placed where the video card is already on screen, which is the only place this
    rule is what decides anything: with no animation the two frames either side of a swallowed
    gesture are byte-identical, and the far older `seen` test ends the read there on its own.
    """
    adb = _VideoWorldAdb()
    drv = _drv(adb)
    real_scroll_up, stalled = adb.scroll_up, {"done": False}

    def visible_video_px():
        p0, p1 = _VIDEO_PAGE_ROWS
        return min(_BAND1, p1 - adb.scroll) - max(_BAND0, p0 - adb.scroll)

    def scroll_up(frac, x_frac=0.5):
        # One gesture goes nowhere, once enough of the animating card is on screen that the
        # two frames either side of it cannot be byte-identical.
        if visible_video_px() >= 400 and not stalled["done"]:
            stalled["done"] = True
            adb.scrolls += 1
            adb.gestures.append((frac, x_frac))
            return
        real_scroll_up(frac, x_frac)

    adb.scroll_up = scroll_up

    assert drv._capture_current() is not None
    assert stalled["done"], "the fixture never actually stalled a gesture"
    assert adb.scroll == _MAX_SCROLL, "one stall must not have ended the read early"


def test_the_dossier_keeps_the_failing_pairs_own_frames_when_the_cap_has_to_drop_some(tmp_path):
    """Live 2026-08-16 (Grace): six refused pairs merged into a 25-frame candidate pool, the
    eight-image cap's even stride walked LIST POSITIONS with no idea which of them formed a pair,
    and the bundle shipped frames 35 and 38 -- the neighbours of failing pair (36, 37) -- while
    both frames of the pair itself were dropped. The one refusal the dossier is named after was
    the one thing it could not be used to replay.

    Breadth is still spent on the rest of the pool; it just no longer outranks the evidence."""
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-failing-pair")

    class _Edge:
        def __init__(self, y):
            self.y, self.observed = y, True

    class _Block:
        y0, y1, complete = 20, 60, True
        top, bottom = _Edge(20), _Edge(60)
        hearts = ((900, 42),)

    class _Seg:
        blocks = (_Block(),)
        card_x = (_CARD_X0, _CARD_X1)
        band = (_BAND0, _BAND1)
        failures = ()

    refused_at = (36, 43, 46, 53, 57, 60)      # the incident's own six refused pairs

    class _Shift:
        def __init__(self, delta_px):
            self.delta_px = delta_px
            self.status = "measured" if delta_px is not None else "no_consensus"
            self.consensus_px = delta_px
            self.confidence = 1.0 if delta_px is not None else 0.0
            self.agreeing = 9 if delta_px is not None else 0
            self.dissenting = 0 if delta_px is not None else 10
            self.eligible = 9 if delta_px is not None else 11
            self.reason = "synthetic"
            self.strips = ()

    class _Index:
        frames = (_Seg(),) * 64
        source_frame_indices = tuple(range(64))
        offsets = tuple(i * 100 for i in range(64))
        shifts = tuple(_Shift(None if i in refused_at else 240) for i in range(63))
        failures = ()

    photos = [f"frame-{i}".encode() for i in range(64)]
    reason = ("frames 36 and 37 could not be put in one coordinate space: no_consensus"
              + "".join(f"; frames {a} and {a + 1} likewise" for a in (43, 46, 53, 57, 60)))

    assert drv._item_index_refused(photos, reason, _Index()) == reason

    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    refusal = next(rec for rec in records if rec["action"] == "item_index_refused")
    saved = refusal["evidence_frames"]

    assert refusal["failing_pair"] == [36, 37]
    assert len(saved) == 8, "the cap still binds"
    assert "item_index_refused_" in saved[0]
    numbers = sorted(int(name.rsplit("_frame_", 1)[1].split(".")[0]) for name in saved)
    assert 36 in numbers and 37 in numbers, (
        f"the failing pair's own frames were dropped again: {numbers}")
    assert numbers[0] == 35 and numbers[-1] == 63, "breadth still reaches both endpoints"
    for name in saved:
        assert (drv._dbg.dir / name).exists()
