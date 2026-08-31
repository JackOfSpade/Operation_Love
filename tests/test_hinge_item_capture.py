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
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from operation_love import targeting_policy as tp
from operation_love.drivers import (
    hinge, item_crops, item_identity, item_index, item_nav, scroll_step, scroll_top, segment)
from operation_love.drivers.hinge import HingeDriver
from operation_love.drivers.debuglog import HingeDebugLog
from operation_love.perception.capture import Profile

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
    "item_selection_policy_id": "hinge_photos_only_v2",
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
    monkeypatch.setattr(
        hinge, "unnumber_without_still_photo_evidence", lambda _evidence: None)
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
    end. `items_unavailable` and `items_unnumbered` are both empty: `items` is the live one of
    the three (see perception.capture.Profile's docstring for all three states)."""
    adb = WorldAdb()
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile is not None
    assert len(profile.items) == len(_CARDS) == 4
    assert len(profile.item_context) == 1
    assert profile.name == "Ada"
    assert profile.items_truncated is False
    assert profile.items_unavailable == ""
    assert profile.items_unnumbered == ""


def test_hinge_numbers_only_policy_approved_photos_and_preserves_prompt_heart(monkeypatch):
    seen = 0

    def photo_only(_crop):
        nonlocal seen
        seen += 1
        return "synthetic written prompt" if seen == 2 else None

    monkeypatch.setattr(hinge, "unnumber_unless_confident_photo", photo_only)
    monkeypatch.setattr(
        hinge, "confident_photo_heart_ordinals",
        lambda _frames, _index, **_kw: (1, 3, 4))
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
    assert drv.model_item_media_ordinal(2) == 3
    video = next(crop for crop in drv._current_item_payload.excluded
                 if crop.heart_ordinal == 2)
    assert video.number is None and video.image is None and not video.sent
    assert "video_mute_v1" in video.reason


def test_profile_items_unnumbered_defaults_to_empty():
    """The plain dataclass default, independent of any driver: a `Profile` nobody populated has
    produced a numbered list (vacuously, via an empty `items`) and has nothing to explain about
    a captured-but-empty enumeration, so `items_unnumbered` is "" exactly like `items_unavailable`
    is not the live reason by default."""
    profile = Profile()
    assert profile.items_unnumbered == ""
    assert profile.items_unavailable_kind == ""
    assert profile.items == ()


def test_all_video_profile_numbers_nothing_but_does_not_stop_the_run(monkeypatch):
    """found+fixed 2026-08-22 (ops/STILL-PHOTO-DISCRIMINATOR.md 5d): a profile whose every card is
    excluded still finishes enumeration -- the index and crops are both sound, nothing survived
    policy, and that is `Profile.items_unnumbered`, never `items_unavailable`. The OLD assertion
    here (`items_unavailable` set, `_current_item_index is None`) encoded exactly the bug this
    fix removes: worker.py's auto loop hard-stops on `items_unavailable`, and a profile of videos
    is a normal outcome that must never do that."""
    monkeypatch.setattr(
        HingeDriver, "_video_selection_exclusions",
        lambda self, frames, index: {
            block.heart_ordinal: "video_mute_v1: mute control matched"
            for block in index.selectable})
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert profile is not None and profile.items == ()
    assert profile.items_unavailable == ""
    assert "video_mute_v1" in profile.items_unnumbered
    assert "selectable card" in profile.items_unnumbered
    # The index and payload are real and current -- enumeration succeeded -- not invalidated the
    # way a genuine refusal (`_item_index_refused`) would leave them.
    assert drv._current_item_index is not None and drv._current_item_payload is not None
    assert drv._current_item_payload.item_count == 0


def test_driver_refuses_auto_hidden_video_risk_when_still_photo_evidence_is_absent(
        monkeypatch):
    """Clean mute screens alone never authorize a production numbered payload.

    Every candidate is still refused (the gate's per-item decision is unchanged), but as of
    2026-08-22 that no longer reads as a capture failure: enumeration ran fine, so this is
    `items_unnumbered`, not `items_unavailable` -- see the sibling test above for the full
    reasoning."""
    monkeypatch.setattr(
        hinge, "unnumber_without_still_photo_evidence",
        item_crops.unnumber_without_still_photo_evidence)
    monkeypatch.setattr(
        HingeDriver, "_match_video_mute",
        staticmethod(lambda _frame, _rect: (True, 0.0)))
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert profile is not None and profile.items == ()
    assert profile.items_unavailable == ""
    assert profile.items_unnumbered != ""
    assert drv._current_item_payload is not None


@pytest.fixture
def installed_still_photo_bound():
    """Numbering readiness exactly as config.validate() installs it, dropped again at teardown.

    Readiness is process-global (ops/STILL-PHOTO-DISCRIMINATOR.md section 5), so it is installed
    through the real API rather than monkeypatched, and always torn down.
    """
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL,
        human_ground_truth=True, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=3, max_video_exact_run_s=0.2, artifact_sha256="b" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1"))
    yield
    tp._reset_installed_still_photo_bound_for_tests()


def test_an_unlicensed_read_takes_no_dwell_burst_at_all(monkeypatch):
    """With no verified bound the payload can number nothing whatever the dwell says, so a burst
    would spend seconds of screen time to change no outcome.  This read must be byte-identical
    to the pre-dwell driver: no extra screencap, no extra sleep, no dwell record."""
    bursts = []
    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst",
                        lambda self: bursts.append(1) or ([], 0.0))
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert bursts == []
    assert profile is not None and len(profile.items) == 4
    assert drv._current_item_payload.translation == (1, 2, 3, 4)


def test_a_licensed_read_dwells_persists_every_frame_and_threads_the_evidence(
        monkeypatch, tmp_path, installed_still_photo_bound):
    """The other half: once a bound is installed the driver holds the screen, screencaps it
    several times with NO input, keeps every frame and its digest, and hands the per-card verdict
    to the payload builder as the `still_photo_dwell` mapping."""
    monkeypatch.setattr(hinge, "unnumber_without_still_photo_evidence",
                        item_crops.unnumber_without_still_photo_evidence)
    seen = {}

    def dwell_spy(index, frames, dwell_frames, **kw):
        seen["burst"] = list(dwell_frames)
        return item_crops.still_photo_dwell_evidence(index, frames, dwell_frames, **kw)

    real_payload = hinge.build_item_payload

    def payload_spy(*args, **kw):
        seen["threaded"] = kw.get("still_photo_dwell")
        return real_payload(*args, **kw)

    monkeypatch.setattr(hinge, "still_photo_dwell_evidence", dwell_spy)
    monkeypatch.setattr(hinge, "build_item_payload", payload_spy)
    # This synthetic world's flat painted cards are not fixtures for the production pixel
    # content classifier. Keep this test scoped to dwell persistence by declaring every indexed
    # heart eligible for the new cheap candidate pre-pass.
    monkeypatch.setattr(
        hinge, "confident_photo_heart_ordinals",
        lambda _frames, index, **_kw: tuple(index.translation))
    adb = WorldAdb()
    # still_photo_dwell_candidates=1: this test is about the ONE free card's dwell burst
    # (2026-08-23's K-candidate walk is untested here on purpose -- see the dedicated section
    # near the end of this file), and WorldAdb's `swipe` is a deliberate no-op the walk's
    # navigation hops cannot climb with.
    drv = _drv(adb, still_photo_dwell_candidates=1)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="still-photo-dwell")
    # Snapshot the transport either side of the burst: "no input" is the load-bearing claim, and
    # a dwell that scrolled would be measuring a moving card while calling it still.
    real_burst = HingeDriver._still_photo_dwell_burst

    def guarded_burst(self, should_stop=None):
        before = (adb.scrolls, len(adb.gestures), len(adb.taps))
        result = real_burst(self, should_stop)
        seen["input_during_dwell"] = (adb.scrolls, len(adb.gestures), len(adb.taps)) != before
        return result

    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst", guarded_burst)

    drv._capture_current()

    burst = seen["burst"]
    assert hinge._STILL_PHOTO_DWELL_FRAMES[0] <= len(burst) <= hinge._STILL_PHOTO_DWELL_FRAMES[1]
    assert adb.taps == [] and adb.texts == [], "a dwell issues no input of any kind"
    assert seen["input_during_dwell"] is False
    # The evidence really reached the gate, keyed by heart ordinal, with C2 and C3 both PASSING
    # on a motionless synthetic screen -- which is what a still photo looks like.
    threaded = seen["threaded"]
    assert threaded, "the dwell verdict must reach build_item_payload"
    for entry in threaded.values():
        assert entry.dwell_exact is True
        assert entry.mute_screens_complete is True
        assert entry.dwell_span_s > 0
        assert len(entry.dwell_frame_sha256s) == len(burst) + 1  # the anchor leads the burst
    # A still photo produces N identical frames, which is exactly the shape the debug log's
    # per-label dedup collapses. Every observation survives in JSONL, while the identical bytes
    # occupy one PNG rather than consuming N slots in the bounded screenshot ring.
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    frame_records = [r for r in records if "dwell_frame_index" in r]
    assert len(frame_records) == len(burst)
    assert {record["action"] for record in frame_records} == {"still_photo_dwell_frame"}
    assert len({record["before"] for record in frame_records}) == 1
    for position, record in enumerate(frame_records):
        assert record["dwell_frame_index"] == position
        assert (drv._dbg.dir / record["before"]).read_bytes() == burst[position]
        assert record["sha256"] == hashlib.sha256(burst[position]).hexdigest()
    summary = next(r for r in records if r["action"] == "still_photo_dwell")
    assert summary["dwell_frames"] == len(burst)
    assert summary["dwell_span_s"] > 0
    assert summary["cards"], "the summary names what the dwell concluded per card"


def test_dwell_debug_frames_dedupe_identical_bytes_but_keep_changed_frames(tmp_path):
    """The driver's stable dwell action label is the wiring DebugLog's content dedupe needs.

    Every observation remains a separate ordered JSONL row. Only byte-identical image payloads
    share a file; a changed frame still receives its own PNG and path.
    """
    drv = _drv(WorldAdb(), still_photo_dwell_candidates=1)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="still-photo-dwell-dedupe")

    drv._record_still_photo_dwell([b"STILL", b"STILL", b"CHANGED"], 1.0, {})

    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    frames = [record for record in records if record["action"] == "still_photo_dwell_frame"]
    assert [record["dwell_frame_index"] for record in frames] == [0, 1, 2]
    assert frames[0]["before"] == frames[1]["before"]
    assert frames[2]["before"] != frames[0]["before"]
    assert len(list(drv._dbg.dir.glob("*.png"))) == 2
    assert (drv._dbg.dir / frames[0]["before"]).read_bytes() == b"STILL"
    assert (drv._dbg.dir / frames[2]["before"]).read_bytes() == b"CHANGED"


@pytest.fixture
def accepted_still_photo_assumption():
    """The OTHER readiness channel: the owner's accepted centred-autoplay assumption, which
    carries no measured worst-case exact run at all."""
    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION,
        accepted_at="2026-08-21", device="synthetic-pixel", hinge_version_name="10.0.1",
        rationale="owner accepted the centred-autoplay assumption instead of the held-out "
                  "false-accept measurement"))
    yield
    tp._reset_installed_still_photo_bound_for_tests()


def test_the_dwell_window_is_anchored_on_whichever_licence_is_installed(
        monkeypatch, installed_still_photo_bound):
    """The measured channel multiplies the artifact's worst held-out exact run by the safety
    factor. Nothing here is a constant chosen in the driver."""
    windows = []
    monkeypatch.setattr(hinge, "human_cooldown", lambda seconds: windows.append(seconds) or 0.0)
    drv = _drv(WorldAdb())

    drv._still_photo_dwell_burst()

    bound = tp.installed_still_photo_bound()
    assert windows == [max(hinge._STILL_PHOTO_DWELL_MIN_WINDOW_S,
                           tp.STILL_PHOTO_DWELL_WINDOW_SAFETY_FACTOR
                           * bound.max_video_exact_run_s)]


def test_an_assumption_licence_dwells_on_its_named_default_instead_of_crashing(
        monkeypatch, accepted_still_photo_assumption):
    """REGRESSION. The burst used to read `installed_still_photo_bound().max_video_exact_run_s`
    behind a guard that only asked whether numbering was licensed AT ALL. Under the assumption
    channel that guard passes while the measured bound is None, so the live run raised
    AttributeError here. The assumption has no measurement to multiply, so the window comes from
    a named accepted default -- and it is never shorter than the measured path's."""
    windows = []
    real_cooldown = hinge.human_cooldown
    monkeypatch.setattr(hinge, "human_cooldown",
                        lambda seconds: windows.append(seconds) or real_cooldown(seconds))
    drv = _drv(WorldAdb())

    burst, span_s = drv._still_photo_dwell_burst()

    assert windows == [hinge._STILL_PHOTO_DWELL_ASSUMED_WINDOW_S]
    assert hinge._STILL_PHOTO_DWELL_ASSUMED_WINDOW_S >= hinge._STILL_PHOTO_DWELL_MIN_WINDOW_S
    assert hinge._STILL_PHOTO_DWELL_FRAMES[0] <= len(burst) <= hinge._STILL_PHOTO_DWELL_FRAMES[1]
    assert span_s >= 0


def test_slow_screencaps_do_not_get_added_to_every_dwell_gap(
        monkeypatch, accepted_still_photo_assumption):
    """A nominal six-second burst must not add every ADB screencap to every gap."""
    clock = {"now": 0.0}
    capture_s = 0.8
    count = 10
    window_s = hinge._STILL_PHOTO_DWELL_ASSUMED_WINDOW_S
    drv = _drv(WorldAdb())

    monkeypatch.setattr(hinge, "human_cooldown", lambda _seconds: window_s)
    monkeypatch.setattr(hinge.random, "randint", lambda *_args: count)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock["now"])

    def slow_screencap(*_args, **_kwargs):
        clock["now"] += capture_s
        return b"frame"

    def advance(seconds, _should_stop=None):
        clock["now"] += seconds
        return True

    monkeypatch.setattr(drv, "_screencap", slow_screencap)
    monkeypatch.setattr(drv, "_interruptible_sleep", advance)

    burst, span_s = drv._still_photo_dwell_burst()

    assert len(burst) == count
    assert span_s >= window_s
    assert span_s < window_s + 3 * capture_s


def test_an_assumption_licence_produces_real_dwell_evidence_end_to_end(
        accepted_still_photo_assumption):
    """The same regression from the other side: the licensed path must actually run and produce
    per-card evidence, not merely avoid an exception."""
    index, frames = _centred_capture()
    drv = _drv(ProbeWorldAdb(start=_CENTRED_SCROLLS[-1]))

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert evidence[_CENTRED_ORDINAL].dwell_exact is True
    assert evidence[_CENTRED_ORDINAL].reattach_probe_ran is True


def test_no_licence_means_no_burst_at_all_rather_than_a_missing_bound():
    """The third state, unchanged: with nothing installed the burst is empty, which every caller
    already treats as "no dwell happened" and refuses on."""
    assert tp.installed_still_photo_licence() is None

    assert _drv(WorldAdb())._still_photo_dwell_burst() == ([], 0.0)


def test_a_blank_screen_ends_the_burst_and_yields_no_dwell_evidence(monkeypatch):
    """A screen that cannot be read has not been dwelled on.  Returning the frames it did get
    would be a SHORTER dwell still claiming to be one, so the burst comes back empty and the
    gate refuses for the reason it should: no dwell."""
    drv = _drv(WorldAdb())
    monkeypatch.setattr(HingeDriver, "_screencap", lambda self, **_kw: None)
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL,
        human_ground_truth=True, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=0, max_video_exact_run_s=0.1, artifact_sha256="c" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1"))
    try:
        assert drv._still_photo_dwell_burst() == ([], 0.0)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()


def test_an_empty_dwell_burst_numbers_nothing_but_still_finishes_enumeration(
        monkeypatch, installed_still_photo_bound):
    """End to end: a licensed read whose dwell produced nothing numbers nothing, and says so --
    as `items_unnumbered`, not `items_unavailable` (found+fixed 2026-08-22; see the sibling
    all-video tests above for the full reasoning). An empty dwell burst is exactly the "never
    observed" case, so every selectable card's own reason is `EXCLUSION_NEVER_DWELLED`."""
    monkeypatch.setattr(hinge, "unnumber_without_still_photo_evidence",
                        item_crops.unnumber_without_still_photo_evidence)
    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst",
                        lambda self, should_stop=None: ([], 0.0))
    drv = _drv(WorldAdb())

    profile = drv._capture_current()

    assert profile is not None and profile.items == ()
    assert profile.items_unavailable == ""
    assert "configured bounded dwell walk did not cover" in profile.items_unnumbered
    assert drv._current_item_payload is not None


# =====================================================================================
# The re-attach probe: the residual a single centred dwell cannot see
# =====================================================================================

# Scroll positions whose LAST frame puts one card dead-centre in the content band, which is the
# only situation the probe is ever spent on. Card 3 lands on page rows 2434..3434, so at 1734 it
# occupies frame rows 700..1700 -- centre 1200, exactly the band's own centre.
_CENTRED_SCROLLS = (0, 260, 520, 780, 1040, 1300, 1560, 1734)
_CENTRED_ORDINAL = 3


class ProbeWorldAdb(WorldAdb):
    """WorldAdb that also honours the REVERSE read-scroll, which the probe's exit leg is.

    The base fake only models `scroll_up` (the forward stroke); a reverse stroke arrives as a
    plain downward `swipe`, exactly as `_scroll(..., reverse=True)` delivers it. Mirroring the
    same transport model in both directions is what lets the round trip be measured rather than
    assumed -- and lets this file assert that it really nets to zero.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reverse_swipes = 0

    def swipe(self, _x1, y1, _x2, y2, **_kwargs):
        if y2 <= y1:
            return
        self.reverse_swipes += 1
        if self._frozen:      # "a screen that never moves, whatever we ask of it" -- either way
            return
        self.scroll = max(0, self.scroll - scroll_step.step_px_for_frac((y2 - y1) / _H, _H))


class RestartingVideoWorldAdb(ProbeWorldAdb):
    """A card that holds byte-exact until Hinge re-attaches it, then starts playing.

    This is the whole residual: stalled, buffering, unloaded and ended-non-looping video all emit
    nothing while the first burst watches. The repainted pixel sits inside the centred card's
    rect and changes on every read once the card has left the autoplay band and come back.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plays = 0

    def screencap(self):
        frame = super().screencap()
        if not (self.reverse_swipes and self.scrolls):
            return frame
        self._plays += 1
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_GRAYSCALE).copy()
        image[1200, 500] = (self._plays * 29) % 255
        ok, buf = cv2.imencode(".png", image)
        assert ok
        return buf.tobytes()


class ScrollTopWorldAdb(ProbeWorldAdb):
    """A page parked at Hinge's TOP STOP: a backward stroke rubber-bands, a forward one moves.

    Where a profile's FIRST photo sits -- calibration's depth-1 target, and production's item 1
    -- and the live failure of 2026-08-22. The reverse stroke IS delivered; Hinge simply has
    nothing above the top to bring into view, so the page comes back byte-identical and the
    shift estimator MEASURES +0px rather than refusing. Once a forward stroke has taken the page
    off that stop, backward travel works again exactly as it does on the device, which is why
    the stop is modelled as a FLOOR and not as a swallowed direction.

    The floor sits at the starting scroll rather than at world row 0 because only a CENTRED card
    is ever probed and this world's first card is not centred at row 0. Nothing the driver reads
    can tell the two apart: it sees a backward stroke that was delivered and moved the page by
    nothing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._top_stop = self.scroll

    def swipe(self, _x1, y1, _x2, y2, **_kwargs):
        if y2 <= y1:
            return
        self.reverse_swipes += 1
        self.scroll = max(self._top_stop,
                          self.scroll - scroll_step.step_px_for_frac((y2 - y1) / _H, _H))


def _centred_capture():
    """The index and frames whose anchor holds one card inside the autoplay trigger zone."""
    frames = [_frame(s) for s in _CENTRED_SCROLLS]
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)
    assert index.usable, index.failures
    return index, frames


def test_the_centred_fixture_really_centres_a_card():
    """A guard on the fixture, not the driver: every probe assertion below is meaningless if the
    anchor frame does not actually hold a card in Hinge's autoplay trigger zone."""
    index, frames = _centred_capture()
    rects = item_crops.dwell_card_rects(index, len(frames) - 1)

    assert set(rects) == {_CENTRED_ORDINAL}
    assert item_crops.card_center_offset_frac(
        rects[_CENTRED_ORDINAL], frame_height=_H, content_band=_CONTENT_BAND) == 0.0


def test_a_centred_byte_exact_card_is_probed_and_the_page_is_left_where_it_was(
        installed_still_photo_bound):
    """The headline. A first burst that comes back byte-exact does NOT buy an acceptance: the
    card is taken out of the autoplay band and brought back, and only a second byte-exact,
    screened, centred burst after that re-entry can be numbered."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    # still_photo_dwell_candidates=1: this test's gesture-count assertions below are about the
    # ONE free card's re-attach probe specifically (2026-08-23's K-candidate walk, which would
    # otherwise also climb to hearts 2 and 1 here, has its own dedicated section near the end of
    # this file).
    drv = _drv(adb, still_photo_dwell_candidates=1)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    entry = evidence[_CENTRED_ORDINAL]
    assert (entry.dwell_exact, entry.centered) == (True, True)
    assert entry.reattach_probe_ran is True
    assert entry.reattach_dwell_exact is True
    assert entry.reattach_mute_screens_complete is True
    assert entry.reattach_centered is True
    assert entry.reattach_dwell_span_s > 0
    assert len(entry.reattach_dwell_frame_sha256s) >= 2
    # Out once, back once, and the page ends exactly where the read left it -- which is what
    # keeps `_index_captured_items`'s entry anchor, and bottom-up navigation, unaffected.
    assert adb.reverse_swipes == 1 and adb.scrolls == 1
    assert adb.scroll == _CENTRED_SCROLLS[-1]
    assert adb.taps == [] and adb.texts == []


def test_the_probe_uses_only_the_drivers_guarded_humanized_read_scrolls(
        tmp_path, installed_still_photo_bound):
    """Owner rule: best humanized interaction or fail loudly. Every gesture the probe issues has
    to arrive through the ledger-keeping, forbidden-zone-guarded read-scroll primitives, never a
    raw coordinate and never the transport directly."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    # still_photo_dwell_candidates=1: the exact input trace asserted below is the ONE free
    # card's probe; the K-candidate walk (2026-08-23) is exercised separately near the end of
    # this file.
    drv = _drv(adb, still_photo_dwell_candidates=1)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="reattach-probe")

    drv._still_photo_dwell(frames, index)

    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    inputs = [r for r in records if r["action"] == "device_input"]
    assert [(r["kind"], r["source"]) for r in inputs] == [
        ("swipe", "_scroll"),              # the reverse exit, through _scroll's guarded stroke
        ("scroll", "_scroll_down_one"),    # the forward return, through the read-scroll ledger
    ]
    for record in inputs:
        assert record["transport"] == type(drv.touch).__name__
    # Both distances live inside the read-scroll envelope every other gesture here lives in.
    forward = next(r for r in inputs if r["kind"] == "scroll")
    assert hinge._READ_SCROLL_FRAC_MIN <= forward["distance_frac"] <= hinge._READ_SCROLL_FRAC_MAX
    # And the probe's frames are kept, under their own labels, beside the first burst's.
    probe_frames = [r for r in records if r["action"].startswith("still_photo_reattach_")]
    assert len(probe_frames) >= 3
    for position, record in enumerate(probe_frames):
        assert record["reattach_frame_index"] == position
        assert record["sha256"] == hashlib.sha256(
            (drv._dbg.dir / record["before"]).read_bytes()).hexdigest()
    summary = next(r for r in records if r["action"] == "still_photo_dwell")
    assert summary["reattach_probe_ran"] is True
    assert summary["reattach_page_shift_px"] == 0
    assert summary["reattach_span_s"] > 0


def test_media_that_only_starts_on_re_attach_is_caught_by_the_second_burst(
        installed_still_photo_bound):
    """The defect the probe exists for, end to end: a card that was NOT PLAYING while the first
    burst watched it holds byte-exact and looks exactly like a photograph."""
    index, frames = _centred_capture()
    adb = RestartingVideoWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    entry = evidence[_CENTRED_ORDINAL]
    assert entry.dwell_exact is True, "the first burst saw a perfectly still card"
    assert entry.reattach_probe_ran is True
    assert entry.reattach_dwell_exact is False
    assert item_crops.unnumber_without_still_photo_evidence(
        item_crops.still_photo_evidence_from_drift(0.0, (1, 2), entry)) is not None


def test_an_off_centre_capture_is_never_charged_a_probe(installed_still_photo_bound):
    """No card could reach the probe rung, so no gesture is spent proving it. The cards demote at
    the centring rung above, exactly as they did before the probe existed."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    # still_photo_dwell_candidates=1: "spend nothing at all" below is about the PROBE
    # specifically: the K-candidate walk (2026-08-23) would otherwise still climb to hearts 2
    # and 1 for its OWN reason (covering more of the profile), which is unrelated to this
    # test's off-centre-demotion claim and is exercised separately near the end of this file.
    drv = _drv(adb, still_photo_dwell_candidates=1)
    # The same capture, with the one centred card demoted to off-centre before the probe is
    # considered: the driver must then spend nothing at all.
    real = hinge.still_photo_dwell_evidence

    def off_centre(*args, **kwargs):
        return {ordinal: dataclasses.replace(entry, centered=False, center_offset_frac=0.42)
                for ordinal, entry in real(*args, **kwargs).items()}

    original, hinge.still_photo_dwell_evidence = hinge.still_photo_dwell_evidence, off_centre
    try:
        evidence, _anchor = drv._still_photo_dwell(frames, index)
    finally:
        hinge.still_photo_dwell_evidence = original

    assert evidence[_CENTRED_ORDINAL].reattach_probe_ran is None
    assert (adb.reverse_swipes, adb.scrolls) == (0, 0)
    reason = item_crops.unnumber_without_still_photo_evidence(
        item_crops.still_photo_evidence_from_drift(0.0, (1, 2), evidence[_CENTRED_ORDINAL]))
    assert reason is not None and "autoplay trigger zone" in reason


def test_an_unlicensed_build_never_moves_the_screen_for_a_probe():
    """The probe is the only still-photo path that spends real gestures, so it re-asserts the
    licence itself rather than trusting the caller."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)

    assert drv._still_photo_reattach_probe(frames[-1], (53, 700, 1027, 1700)) is None
    assert (adb.reverse_swipes, adb.scrolls, adb.taps) == (0, 0, [])


def test_a_probe_whose_page_will_not_move_refuses_rather_than_claiming_a_re_attach(
        monkeypatch, installed_still_photo_bound):
    """Fail closed: a page that did not move detached nothing, so nothing can re-attach, and
    calling the re-entry a re-attach anyway would manufacture the observation the rung exists to
    earn."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1], frozen=True)
    drv = _drv(adb)
    monkeypatch.setattr(HingeDriver, "_measured_page_shift",
                        lambda self, _before, _after: 0)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert evidence[_CENTRED_ORDINAL].reattach_probe_ran is None


def test_a_probe_that_cannot_measure_where_the_page_went_refuses(
        monkeypatch, installed_still_photo_bound):
    """`estimate_shift` is the one comparator here that says "I cannot tell"; the probe never
    falls back to the distance it asked for, because the rect has to follow the page."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)
    monkeypatch.setattr(HingeDriver, "_measured_page_shift",
                        lambda self, _before, _after: None)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert evidence[_CENTRED_ORDINAL].reattach_probe_ran is None


def test_a_card_parked_at_the_scroll_top_exits_forward_when_backward_rubber_bands(
        tmp_path, installed_still_photo_bound):
    """The live failure of 2026-08-22, pinned. The exit stroke is BACKWARD, which is right for
    the deep-parked cards production mostly probes and structurally impossible at the scroll
    top: a profile's first photo has nothing above it, so the stroke rubber-bands, the shift
    comes back a MEASURED +0px, and the card is still inside the autoplay trigger zone. Refusing
    there made every depth-1 target -- which is calibration's whole target set, and production's
    item 1 -- permanently unprovable, so a measured clamp buys ONE retry the other way and the
    card leaves through the top instead. The page still comes back to where the read left it."""
    index, frames = _centred_capture()
    adb = ScrollTopWorldAdb(start=_CENTRED_SCROLLS[-1])
    # still_photo_dwell_candidates=1: the exact input trace asserted below is the ONE free
    # card's clamped probe; the K-candidate walk (2026-08-23) is exercised separately near the
    # end of this file.
    drv = _drv(adb, still_photo_dwell_candidates=1)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="reattach-scroll-top")

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    entry = evidence[_CENTRED_ORDINAL]
    assert entry.reattach_probe_ran is True
    assert entry.reattach_dwell_exact is True
    assert entry.reattach_mute_screens_complete is True
    assert entry.reattach_centered is True
    assert len(entry.reattach_dwell_frame_sha256s) >= 2
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    inputs = [r for r in records if r["action"] == "device_input"]
    # Backward first, forward second, and the return leg after that: the clamped case costs
    # exactly one stroke more than the unclamped one, and every stroke is still a guarded
    # humanized read-scroll.
    assert [(r["kind"], r["source"]) for r in inputs] == [
        ("swipe", "_scroll"),              # the backward exit, swallowed by the top stop
        ("scroll", "_scroll_down_one"),    # the retry the MEASURED clamp bought: out via the top
        ("swipe", "_scroll"),              # the return, back down to where the read was
    ]
    for record in inputs:
        assert record["transport"] == type(drv.touch).__name__
    forward = next(r for r in inputs if r["kind"] == "scroll")
    assert hinge._READ_SCROLL_FRAC_MIN <= forward["distance_frac"] <= hinge._READ_SCROLL_FRAC_MAX
    summary = next(r for r in records if r["action"] == "still_photo_dwell")
    assert summary["reattach_probe_ran"] is True
    assert summary["reattach_page_shift_px"] == 0
    assert adb.scroll == _CENTRED_SCROLLS[-1]


def test_a_probe_clamped_in_both_directions_refuses_as_it_always_did(
        monkeypatch, installed_still_photo_bound):
    """The retry widens the way OUT, never the standard of proof. A page that will not move
    either way detached nothing, so nothing can re-attach: the probe spends its two measured
    strokes, takes no second burst at all, and refuses exactly as it did before the retry
    existed. Nothing about the refusal is monkeypatched into place: the estimator really
    measures +0px across two byte-identical frames, which is the evidence a rubber-banding
    device produces, and the one patch here only COUNTS bursts."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1], frozen=True)
    drv = _drv(adb)
    rect = item_crops.dwell_card_rects(index, len(frames) - 1)[_CENTRED_ORDINAL]
    bursts = []
    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst",
                        lambda self: bursts.append(1) or ([], 0.0))

    assert drv._still_photo_reattach_probe(frames[-1], rect) is None

    assert (adb.reverse_swipes, adb.scrolls) == (1, 1), "one stroke each way, and not one more"
    assert bursts == [], "a card that never left the zone is charged no second burst"
    assert adb.scroll == _CENTRED_SCROLLS[-1] and adb.taps == []


# =====================================================================================
# STOP after the read loop (found+fixed 2026-08-23): the dwell burst and the re-attach probe
# are not cheap -- a burst alone can hold the screen for several real seconds, and the probe
# spends real scroll gestures on top of that -- and until this fix nothing polled `should_stop`
# for any of it once the read loop itself had finished. See `_capture_current`'s "STOP, PART 2"
# docstring paragraph for the full incident and `_still_photo_reattach_probe`'s STOP paragraph
# for the return-leg decision pinned below.
# =====================================================================================

def test_a_stop_before_the_dwell_starts_takes_no_burst_at_all(
        monkeypatch, installed_still_photo_bound):
    """Pins the fix's headline claim: a Stop noticed the instant the read loop ends -- before
    `_still_photo_dwell` has spent even the opening screencap of its first burst -- must not
    call the burst at all, not merely call it and have it come back empty."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)
    bursts = []
    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst",
                        lambda self, should_stop=None: bursts.append(1) or ([], 0.0))

    evidence, _anchor = drv._still_photo_dwell(frames, index, should_stop=lambda: True)

    assert bursts == [], "the burst must never be called, not just called and cut short"
    assert evidence == {}
    assert (adb.reverse_swipes, adb.scrolls, adb.taps) == (0, 0, [])


def test_a_stop_mid_burst_returns_no_dwell_and_never_a_shorter_completed_one(
        monkeypatch, installed_still_photo_bound):
    """A Stop that lands partway through the burst's own sampled window must come back as the
    SAME empty `([], 0.0)` a blank frame already returns -- never a shorter list of real frames
    that still claims to be a completed dwell. Every caller trusts a non-empty burst to mean
    exactly that, so a truncated one reported as real would be a silent correctness bug, not
    merely a slower stop."""
    drv = _drv(WorldAdb())
    calls = {"n": 0}
    real_screencap = HingeDriver._screencap

    def counting_screencap(self, **kw):
        calls["n"] += 1
        return real_screencap(self, **kw)

    monkeypatch.setattr(HingeDriver, "_screencap", counting_screencap)

    burst, span_s = drv._still_photo_dwell_burst(should_stop=lambda: calls["n"] >= 2)

    assert (burst, span_s) == ([], 0.0)
    assert calls["n"] == 2, "stopped mid-window, not after finishing a shorter one"
    assert calls["n"] < hinge._STILL_PHOTO_DWELL_FRAMES[0]


def test_a_stop_before_any_probe_gesture_returns_none_and_moves_nothing(
        installed_still_photo_bound):
    """The probe's own "check before each gesture" contract: a Stop that is already true before
    the first exit stroke costs one poll, not one more humanized swipe the operator did not ask
    for, and never a partial `ReattachProbe`."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)
    rect = item_crops.dwell_card_rects(index, len(frames) - 1)[_CENTRED_ORDINAL]

    assert drv._still_photo_reattach_probe(frames[-1], rect, should_stop=lambda: True) is None
    assert (adb.reverse_swipes, adb.scrolls, adb.taps) == (0, 0, [])


def test_a_stop_mid_probe_reaches_the_ladder_as_no_reattach_evidence(
        installed_still_photo_bound):
    """End to end through `_still_photo_dwell`: a card that earned the probe rung (byte-exact,
    centred first burst) but whose probe was cut short by a Stop must reach the evidence ladder
    with `reattach_probe_ran` unset -- exactly the same refusal shape a probe that failed to
    measure its own displacement already produces -- never a `reattach_dwell_exact=True` built
    from frames the probe never actually took."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)

    def should_stop():
        return adb.reverse_swipes >= 1     # true only once the exit stroke has fired

    evidence, _anchor = drv._still_photo_dwell(frames, index, should_stop)

    entry = evidence[_CENTRED_ORDINAL]
    assert (entry.dwell_exact, entry.centered) == (True, True)   # earned the probe rung
    assert entry.reattach_probe_ran is None    # ...but the probe itself never completed
    reason = item_crops.unnumber_without_still_photo_evidence(
        item_crops.still_photo_evidence_from_drift(0.0, (1, 2), entry))
    assert reason is not None, "no partial probe evidence may reach the ladder as a pass"


def test_a_stop_after_the_exit_still_completes_the_return_leg_before_bailing(
        monkeypatch, installed_still_photo_bound):
    """Pins the return-leg design decision (see `_still_photo_reattach_probe`'s STOP
    docstring paragraph): once the exit stroke has displaced the page, a Stop does not truncate
    the walk back. The alternative -- bailing immediately -- would leave the phone scrolled to
    an arbitrary, unmeasured offset for the rest of the session, which is worse for the operator
    watching the screen (and for the "leaves the page where it found it" contract bottom-up
    navigation relies on) than paying for the few remaining, already-bounded
    (`_REATTACH_RETURN_STEPS_SPAN`) read-scrolls it takes to walk back. Only the expensive,
    gesture-free second burst that would follow is skipped."""
    index, frames = _centred_capture()
    adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
    drv = _drv(adb)
    rect = item_crops.dwell_card_rects(index, len(frames) - 1)[_CENTRED_ORDINAL]
    bursts = []
    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst",
                        lambda self, should_stop=None: bursts.append(1) or ([], 0.0))

    def should_stop():
        return adb.reverse_swipes >= 1

    result = drv._still_photo_reattach_probe(frames[-1], rect, should_stop)

    assert result is None
    assert bursts == [], "the second burst is the expensive part and must not be spent"
    assert (adb.reverse_swipes, adb.scrolls) == (1, 1), "the return leg still ran to completion"
    assert adb.scroll == _CENTRED_SCROLLS[-1], "the page is restored despite the stop"


def test_the_dwell_and_probe_are_unchanged_by_a_should_stop_that_never_fires(
        installed_still_photo_bound):
    """`should_stop` defaults to None throughout this chain; threading an always-False callable
    instead must reach the exact same verdict -- a full C2/C3 dwell plus a completed re-attach
    probe -- as the existing `should_stop=None` callers already get. Compares the OUTCOME and
    the physical trace (gesture counts, final scroll position) rather than wall-clock-derived
    timing fields, which are not expected to match to the microsecond across two separate runs."""
    index, frames = _centred_capture()

    for stop in (None, lambda: False):
        adb = ProbeWorldAdb(start=_CENTRED_SCROLLS[-1])
        # still_photo_dwell_candidates=1: the gesture counts asserted below are the ONE free
        # card's probe; the K-candidate walk (2026-08-23) is exercised separately near the end
        # of this file.
        drv = _drv(adb, still_photo_dwell_candidates=1)

        evidence, _anchor = drv._still_photo_dwell(frames, index, stop)

        entry = evidence[_CENTRED_ORDINAL]
        assert (entry.dwell_exact, entry.centered) == (True, True)
        assert entry.reattach_probe_ran is True
        assert entry.reattach_dwell_exact is True
        assert entry.reattach_mute_screens_complete is True
        assert entry.reattach_centered is True
        assert (adb.reverse_swipes, adb.scrolls) == (1, 1)
        assert adb.scroll == _CENTRED_SCROLLS[-1]


def test_the_whole_capture_path_is_unchanged_by_a_should_stop_that_never_fires():
    """The same guarantee end to end, on the ordinary (unlicensed) enumeration path every other
    test in this file already exercises with `should_stop=None`: threading an always-False
    callable through `_capture_current` instead must be a complete no-op."""
    for stop in (None, lambda: False):
        drv = _drv(WorldAdb())

        profile = drv._capture_current(stop)

        assert profile is not None
        assert len(profile.items) == len(_CARDS) == 4
        assert profile.items_unavailable == ""
        assert profile.items_unnumbered == ""
        assert profile.name == "Ada"


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


@pytest.mark.parametrize(
    ("screen_result", "expected"),
    [
        ((True, 0.0), None),
        ((True, 1.0), "selected media is a video"),
        ((False, None), "screening could not be completed"),
    ],
    ids=["mute-absent", "mute-visible", "matcher-failed"],
)
def test_exact_target_frame_video_screen_fails_closed(monkeypatch, screen_result, expected):
    monkeypatch.setattr(
        HingeDriver, "_match_video_mute",
        staticmethod(lambda _frame, _rect: screen_result))
    block = SimpleNamespace(x0=_CARD_X0, x1=_CARD_X1, y0=700, y1=1674)

    reason = _drv(WorldAdb())._target_frame_video_screen_reason(_frame(0), block)

    if expected is None:
        assert reason is None
    else:
        assert expected in reason


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


# --- the positioned-marker route (found+fixed 2026-08-23) ---------------------------------------
# `video_mute_screen_reason`'s ROI is anchored to the BLOCK's own top and assumes the mute glyph
# sits within the top 14% of the block's OWN height. That is true for a plain photo/video card
# (glyph ~53px below a block top that IS the media top) but false for a compound card -- a short
# prompt caption drawn above the media inside ONE rounded block (segment.py:243-252 deliberately
# never splits the two) -- where the glyph is really ~188px below the block top: measured live,
# block height ~1109px (~135px caption + 974px media), glyph offset 188px, well outside the
# 0..14% (0..155px) window the screen searches. The numbers below are exactly those measured
# ones, not round test fixtures, so a regression here reproduces the real geometry.
def test_prompt_caption_card_is_excluded_by_the_located_marker_the_screen_cannot_see(monkeypatch):
    """A compound prompt-then-media card the block-relative screen legitimately clears (it
    searched the wrong sub-window and correctly found nothing there) is still excluded, because
    `_video_mute_marker_rows`'s unbounded, block-agnostic search located the real glyph inside
    this same card."""
    block_height = 1109
    glyph_dy = 188
    block = SimpleNamespace(
        heart_ordinal=5, x0=_CARD_X0, x1=_CARD_X1, height=block_height,
        observations=(
            SimpleNamespace(top_observed=True, frame_index=0,
                            frame_y0=1600, frame_y1=1600 + block_height),
        ))
    index = SimpleNamespace(
        selectable=(block,),
        video_mute_markers=(item_index.VideoMuteMarker(
            frame_index=0, x=_CARD_X0 + 12, y=1600 + glyph_dy, score=0.991234),))
    # The block-relative screen, asked about ITS OWN (wrong) sub-window, legitimately finds
    # nothing there -- this is what "the screen cannot see it" looks like from inside the code,
    # not a matcher failure.
    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(
        lambda _frame, _rect: (True, 0.0)))

    exclusions = _drv(WorldAdb())._video_selection_exclusions([_frame(0)], index)

    assert exclusions.keys() == {5}
    reason = exclusions[5]
    # Load-bearing prefix (`model_item_media_ordinal` prefix-matches it as affirmative video
    # evidence, same as the screen route) plus a distinguishable detail an operator/debug log
    # can use to tell the two routes apart.
    assert reason.startswith("video_mute_v1: upper-left")
    assert "positioned marker track" in reason
    assert "188" in reason  # the card-local offset actually located, for an operator to audit


def test_plain_card_is_still_excluded_by_the_existing_screen_route_unchanged(monkeypatch):
    """The ordinary case (glyph inside the screen's own window) is untouched by this fix: the
    screen route still fires, and its own wording is not replaced even though a marker is ALSO
    supplied at the same position -- the two routes may agree, but the screen's reason is never
    overwritten by the marker's."""
    block_height = 974
    glyph_dy = 53
    block = SimpleNamespace(
        heart_ordinal=2, x0=_CARD_X0, x1=_CARD_X1, height=block_height,
        observations=(
            SimpleNamespace(top_observed=True, frame_index=0,
                            frame_y0=700, frame_y1=700 + block_height),
        ))
    index = SimpleNamespace(
        selectable=(block,),
        video_mute_markers=(item_index.VideoMuteMarker(
            frame_index=0, x=_CARD_X0 + 12, y=700 + glyph_dy, score=0.995),))
    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(
        lambda _frame, _rect: (True, 1.0)))

    exclusions = _drv(WorldAdb())._video_selection_exclusions([_frame(0)], index)

    assert exclusions.keys() == {2}
    reason = exclusions[2]
    assert reason.startswith(
        "video_mute_v1: upper-left Hinge mute control matched in card source frame(s)")
    assert "positioned marker track" not in reason


def test_marker_route_only_ever_adds_never_clears_a_card_the_screen_left_clean(monkeypatch):
    """A card with a clean screen and no located marker at all must still be numberable -- the
    second route may only ADD an exclusion, never invent one from nothing."""
    block = SimpleNamespace(
        heart_ordinal=3, x0=_CARD_X0, x1=_CARD_X1, height=974,
        observations=(
            SimpleNamespace(top_observed=True, frame_index=0, frame_y0=700, frame_y1=1674),
        ))
    index = SimpleNamespace(selectable=(block,), video_mute_markers=())
    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(
        lambda _frame, _rect: (True, 0.0)))

    exclusions = _drv(WorldAdb())._video_selection_exclusions([_frame(0)], index)

    assert exclusions == {}


def test_marker_outside_this_cards_own_rows_does_not_exclude_it(monkeypatch):
    """Containment matters: a positioned marker that belongs to page chrome, a gutter, or a
    DIFFERENT card must never be read as evidence about this one."""
    block = SimpleNamespace(
        heart_ordinal=4, x0=_CARD_X0, x1=_CARD_X1, height=974,
        observations=(
            SimpleNamespace(top_observed=True, frame_index=0, frame_y0=700, frame_y1=1674),
        ))
    index = SimpleNamespace(
        selectable=(block,),
        # One row below this card's own last observed row -- belongs to whatever is next, not
        # to this block.
        video_mute_markers=(item_index.VideoMuteMarker(
            frame_index=0, x=_CARD_X0 + 12, y=1674, score=0.99),))
    monkeypatch.setattr(HingeDriver, "_match_video_mute", staticmethod(
        lambda _frame, _rect: (True, 0.0)))

    exclusions = _drv(WorldAdb())._video_selection_exclusions([_frame(0)], index)

    assert exclusions == {}


def test_video_mute_screen_reason_return_contract_is_unchanged_by_this_fix():
    """Pins the deliberate non-fix: `video_mute_screen_reason`'s ROI arithmetic is left exactly
    as measured (see its HONESTY LIMIT paragraph). Re-anchoring the window to a media top is not
    attempted because nothing in this pipeline records where inside a block the media begins
    (`segment.py` never splits a prompt caption from its media), and widening the window was
    already rejected for a different, unrelated reason (it would start catching the per-card
    like heart and the Android status bar). So on the exact measured prompt-caption geometry --
    block height 1109, real glyph 188px down -- the screen given only that block's own rect still
    legitimately finds nothing in its own window and returns None: the honest answer for the
    information THIS function has, not a guarantee no control is really on the card. Closing that
    gap is `_video_selection_exclusions`'s job now (see the marker-route tests above), not this
    function's."""
    block_height = 1109
    glyph_dy = 188
    rect = (_CARD_X0, 0, _CARD_X1, block_height)

    def match_only_at_true_offset(_frame, roi):
        _x0, y0, _x1, y1 = roi
        return True, (1.0 if y0 <= glyph_dy < y1 else 0.0)

    reason = hinge.video_mute_screen_reason(_frame(0), rect, match=match_only_at_true_offset)

    assert reason is None


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
    assert capture["item_coverage"] == {
        "still_photo_dwell_candidate_limit": 3,
        "photo_candidate_page_hearts": [1, 2, 3, 4],
        "dwell_covered_page_hearts": [],
        "heart_bearing_cards": 4,
        "numbered_page_hearts": [1, 2, 3, 4],
        "no_dwell_coverage_page_hearts": [],
        "content_non_photo_page_hearts": [],
        "still_photo_refused_page_hearts": [],
        "other_unnumbered_page_hearts": [],
    }
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


def test_capture_coverage_diagnostics_separates_photo_policy_from_dwell_coverage():
    """The compact capture-row summary must not make an unobserved photo look like a prompt,
    or a prompt look like a still-photo failure.  The full manifest has the prose detail; this
    aggregate makes the reported item count and long dwell cost immediately interpretable."""
    payload = SimpleNamespace(crops=(
        SimpleNamespace(kind=item_crops.CROP_ITEM, heart_ordinal=8, reason="item 1"),
        SimpleNamespace(kind=item_crops.CROP_CONTEXT, heart_ordinal=1,
                        reason=item_crops.EXCLUSION_NEVER_DWELLED),
        SimpleNamespace(kind=item_crops.CROP_CONTEXT, heart_ordinal=2,
                        reason="photo_only: crop classified as written"),
        SimpleNamespace(kind=item_crops.CROP_CONTEXT, heart_ordinal=3,
                        reason="photo_only: re-attach probe did not complete"),
        SimpleNamespace(kind=item_crops.CROP_CONTEXT, heart_ordinal=4, reason="ordinary context"),
    ))

    assert HingeDriver._item_coverage_diagnostics(
            payload, 3, photo_candidate_hearts=(1, 3, 8),
            dwell_covered_hearts=(3, 8)) == {
        "still_photo_dwell_candidate_limit": 3,
        "photo_candidate_page_hearts": [1, 3, 8],
        "dwell_covered_page_hearts": [3, 8],
        "heart_bearing_cards": 5,
        "numbered_page_hearts": [8],
        "no_dwell_coverage_page_hearts": [1],
        "content_non_photo_page_hearts": [2],
        "still_photo_refused_page_hearts": [3],
        "other_unnumbered_page_hearts": [4],
    }


def test_model_context_preserves_every_unnumbered_crop_for_supporting_connections():
    """The wire view retains unnumbered photos/prompts as well as heartless profile context.

    The opener instruction requires the selected numbered item to stay the clear main subject;
    it does not need to discard a useful supporting image to achieve that.
    """
    payload = SimpleNamespace(context=(
        SimpleNamespace(heart_ordinal=1, image=b"unmeasured bridge photo"),
        SimpleNamespace(heart_ordinal=None, image=b"heartless vitals"),
        SimpleNamespace(heart_ordinal=3, image=b"written prompt supporting context"),
        SimpleNamespace(heart_ordinal=None, image=None),
    ))

    assert HingeDriver._model_item_context(payload) == (
        b"unmeasured bridge photo", b"heartless vitals", b"written prompt supporting context")

# =====================================================================================
# What sized every enumeration scroll: the `coverage` aggregate on the capture row
#
# `plan_coverage_step` labels every step it plans with a `basis`, and until 2026-08-28 not one
# of those four tokens had ever reached actions.jsonl -- so "how often does the blind fallback
# fire?" could only be answered by re-segmenting the whole on-disk PNG archive. These tests pin
# the aggregate that makes it a query, and in particular the REFUSAL bucket: that branch becomes
# `Profile.items_unavailable`, which in Training and in AUTO-with-openers stops the run.
# =====================================================================================

_COVERAGE_BASES = (
    scroll_step.COVERAGE_STEP_OPEN,
    scroll_step.COVERAGE_STEP_THROTTLED,
    scroll_step.COVERAGE_STEP_FALLBACK,
    scroll_step.COVERAGE_STEP_SEGMENTATION_FALLBACK,
)


class _RecordingLog:
    """The in-memory debug log the capture-row tests above already use, named once here because
    this section reads several different rows of the same run."""

    def __init__(self):
        self.calls = []

    def action(self, name, **fields):
        self.calls.append((name, fields))

    def row(self, name):
        return next(fields for called, fields in self.calls if called == name)

    def keys_of(self, name):
        return {key for called, fields in self.calls if called == name for key in fields}


def _median_low(values):
    """The same low median the ledger reports: an actually observed value, never a half pixel."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def test_the_capture_row_says_what_sized_every_enumeration_scroll():
    """One row per capture, never one per frame: the counter must agree with what the planner
    really returned, and each span must be that quantity's own min/low-median/max over exactly
    the steps that were planned."""
    drv = _drv(WorldAdb())
    dbg = _RecordingLog()
    drv._dbg = dbg
    planned = []
    real_plan = hinge.plan_coverage_step

    def spy(*a, **k):
        step = real_plan(*a, **k)
        planned.append(step)
        return step

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_coverage_step", spy)
        profile = drv._capture_current()

    assert profile is not None and planned, "the fixture must really enumerate"
    coverage = dbg.row("capture")["coverage"]
    assert coverage["steps"] == len(planned)
    assert coverage["basis"] == {
        basis: sum(1 for step in planned if step.basis == basis) for basis in _COVERAGE_BASES}
    assert sum(coverage["basis"].values()) == coverage["steps"]
    for field in ("cap_px", "step_px", "trust_ceiling_px"):
        observed = [getattr(step, field) for step in planned]
        assert coverage[field] == {"n": len(observed), "min": min(observed),
                                   "median": _median_low(observed), "max": max(observed)}
    # `depth_px` is None on every step no open trailing card constrained, so this bucket's own
    # `n` is the answer to "how often does bound 2 have anything to say?" -- the measurement the
    # archive pass had to re-derive by re-segmenting 3696 PNGs.
    depths = [step.depth_px for step in planned if step.depth_px is not None]
    assert coverage["depth_px"] == (
        None if not depths else {"n": len(depths), "min": min(depths),
                                 "median": _median_low(depths), "max": max(depths)})
    lows = [step.window_px[0] for step in planned]
    highs = [step.window_px[1] for step in planned]
    assert coverage["window_px"]["low"]["min"] == min(lows)
    assert coverage["window_px"]["high"]["max"] == max(highs)
    assert coverage["window_px"]["single_valued"] == sum(
        1 for step in planned if step.window_px[0] == step.window_px[1])
    # One viewport sample per segmented frame, i.e. per planning attempt.
    assert coverage["viewport_offset_px"]["n"] == len(planned)
    assert coverage["refusals"] == []


def test_all_four_coverage_bases_are_counted_even_when_only_one_ever_fires():
    """A basis that never fired must read as a MEASURED zero, not as an absent key -- the exact
    ambiguity that made the archive grep unanswerable (zero hits proves nothing when nothing was
    ever written). All four are reachable in the counter, whether or not this corpus reaches
    them."""
    ledger = hinge._EnumerationCoverageLedger()
    for basis in _COVERAGE_BASES:
        ledger.note_step(SimpleNamespace(
            basis=basis, cap_px=540, step_px=300, trust_ceiling_px=540,
            depth_px=None, window_px=(280, 320)))
    assert ledger.summary()["basis"] == {basis: 1 for basis in _COVERAGE_BASES}

    only_open = hinge._EnumerationCoverageLedger()
    only_open.note_step(SimpleNamespace(
        basis=scroll_step.COVERAGE_STEP_OPEN, cap_px=540, step_px=300,
        trust_ceiling_px=540, depth_px=None, window_px=(280, 320)))
    assert only_open.summary()["basis"] == {
        scroll_step.COVERAGE_STEP_OPEN: 1,
        scroll_step.COVERAGE_STEP_THROTTLED: 0,
        scroll_step.COVERAGE_STEP_FALLBACK: 0,
        scroll_step.COVERAGE_STEP_SEGMENTATION_FALLBACK: 0,
    }


def test_a_step_without_the_geometry_fields_still_counts_its_basis():
    """Doubles standing in for a `CoverageStep` (tests/test_hinge_observe.py) carry only some of
    its fields. A missing measurement must simply not be sampled -- never an AttributeError on a
    live capture's planning path, and never a lost basis count."""
    ledger = hinge._EnumerationCoverageLedger()
    ledger.note_step(SimpleNamespace(basis=scroll_step.COVERAGE_STEP_OPEN, frac=0.4, step_px=240))
    summary = ledger.summary()
    assert summary["basis"][scroll_step.COVERAGE_STEP_OPEN] == 1
    assert summary["steps"] == 1
    assert summary["cap_px"] is None and summary["trust_ceiling_px"] is None
    assert summary["step_px"] == {"n": 1, "min": 240, "median": 240, "max": 240}


@pytest.mark.parametrize("error", [scroll_step.ScrollStepError, segment.SegmentationError],
                         ids=("scroll_step", "segmentation"))
def test_the_refusal_that_stops_the_run_records_its_frame_and_exception_type(error):
    """THE BUCKET THAT MATTERS. This refusal propagates through `_invalidate_item_index` to
    `Profile.items_unavailable` and then, in Training and in AUTO with openers enabled, to
    `stop_event.set(); break` -- it stops the run. The frame index and the exception TYPE make
    that attributable by query instead of by pattern-matching a prose sentence."""
    drv = _drv(WorldAdb())
    dbg = _RecordingLog()
    drv._dbg = dbg

    def _refuse(*a, **k):
        raise error("the local card spacing is 400px")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_coverage_step", _refuse)
        profile = drv._capture_current()

    assert profile is not None and profile.items == ()
    coverage = dbg.row("capture")["coverage"]
    assert coverage["refusals"] == [{"frame_index": 0, "error": error.__name__}]
    # The same frame the refusal SENTENCE names, so the two can be reconciled.
    assert "frame 0 of this profile" in profile.items_unavailable
    assert error.__name__ in profile.items_unavailable
    assert coverage["steps"] == 0, "nothing was ever planned on this read"
    # The refused frame was segmented before the plan was attempted, so its geometry survives --
    # that frame is the interesting one.
    assert coverage["viewport_offset_px"]["n"] == 1


def test_terminal_enumeration_planning_refusal_keeps_the_exact_frame_for_debug(tmp_path):
    """The operator gets the rejected frame, not merely its prose reason in the capture row."""
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="enumeration-planning-refusal")

    def _refuse(*_a, **_k):
        raise scroll_step.ScrollStepError("synthetic unbound selectable heart")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_coverage_step", _refuse)
        profile = drv._capture_current()

    records = [json.loads(line) for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    refusal = next(record for record in records
                   if record["action"] == "enumeration_planning_refused")
    assert refusal["frame_index"] == 0
    assert refusal["error"] == "ScrollStepError"
    assert refusal["reason"] == "synthetic unbound selectable heart"
    assert (drv._dbg.dir / refusal["before"]).read_bytes() == profile.photos[0]


def test_the_coverage_row_survives_a_capture_that_produced_no_item_payload():
    """`item_coverage` is None whenever the payload is, which is exactly the refusal case this
    aggregate exists to explain. A top-level key on the capture row survives it."""
    drv = _drv(WorldAdb())
    dbg = _RecordingLog()
    drv._dbg = dbg

    def _refuse(*a, **k):
        raise scroll_step.ScrollStepError("the local card spacing is 400px")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_coverage_step", _refuse)
        profile = drv._capture_current()

    capture = dbg.row("capture")
    assert profile.items == () and drv._current_item_payload is None
    assert capture["item_coverage"] is None and capture["item_manifest"] == []
    assert capture["items_unavailable"]
    assert isinstance(capture["coverage"], dict)
    assert capture["coverage"]["refusals"][0]["error"] == "ScrollStepError"


def test_the_coverage_aggregate_is_one_nested_dict_and_never_splatted_into_the_row():
    """A basis name is a token this driver does not own. Splatting the aggregate would let a
    future one collide with a key already on the row it is written to; nesting makes that
    unreachable rather than merely unlikely."""
    drv = _drv(WorldAdb())
    dbg = _RecordingLog()
    drv._dbg = dbg

    assert drv._capture_current() is not None
    capture = dbg.row("capture")
    assert isinstance(capture["coverage"], dict)
    # Nothing the aggregate carries leaked out into the row itself.
    assert not set(capture) & (set(_COVERAGE_BASES) | set(capture["coverage"]))
    # The claim the nesting is defensive about, verified against the keys the SAME run really
    # wrote: today no basis name collides with a timing bucket, so nesting costs nothing now and
    # is the only thing that keeps a future one from being a silent overwrite.
    timing_keys = dbg.keys_of("capture_iteration_timing") | dbg.keys_of("capture_timing_summary")
    assert timing_keys and not timing_keys & set(_COVERAGE_BASES)


def test_an_ordinary_read_that_plans_no_coverage_step_carries_no_aggregate():
    """`openers=False` is an ordinary, non-enumerating read. It sizes no coverage step, so its
    row must not grow a set of empty buckets -- the log is long-lived and this aggregate is
    deliberately one row per capture, not one per frame."""
    drv = _drv(WorldAdb(), openers=False)
    dbg = _RecordingLog()
    drv._dbg = dbg

    profile = drv._capture_current()

    assert profile is not None and profile.items == ()
    assert dbg.row("capture")["coverage"] is None


def test_the_honest_viewport_offset_is_measured_from_the_pinned_chrome_not_the_band():
    """The number that would let a future pass decide the honest-viewport question on evidence.
    `_scrolling_viewport_top` reports the first band row the PAGE can occupy; on the 2026-08-28
    incident capture that was 517 against a band starting at 300, i.e. 217px of licence the band
    row alone would have claimed. It is invisible in every log today, and takes many distinct
    values across the archive, so a bare "was it ever non-zero" flag would not carry it."""
    def _pinned(clip_row, run_px):
        return SimpleNamespace(
            band=(_BAND0, _BAND1),
            blocks=(SimpleNamespace(kind=segment.BLOCK_UNANCHORED, y1=clip_row,
                                    bottom=SimpleNamespace(run_px=run_px)),))

    incident = _pinned(500, 17)                  # content top 517 against band row 300
    plain = SimpleNamespace(band=(_BAND0, _BAND1), blocks=())
    assert scroll_step._scrolling_viewport_top(incident) - incident.band[0] == 217
    assert scroll_step._scrolling_viewport_top(plain) - plain.band[0] == 0

    ledger = hinge._EnumerationCoverageLedger()
    for segmentation in (incident, plain, _pinned(390, 10), _pinned(600, 20)):
        ledger.note_frame(segmentation)
    # Four distinct offsets, so the median is pinned as the LOW one: an offset the frames really
    # showed (100), never the 158.5 a mean-of-the-middle-two would invent for a pixel count.
    assert ledger.summary()["viewport_offset_px"] == {
        "n": 4, "min": 0, "median": 100, "max": 320, "nonzero": 3}


def test_a_coverage_ledger_that_cannot_summarise_itself_never_breaks_the_capture():
    """Best-effort, exactly like the neighbouring debug writes: this aggregate is evaluated as an
    argument to the capture row's own `action` call, which is not itself guarded -- so a raising
    summary would abort a live capture that had already read the whole profile."""
    poisoned = hinge._EnumerationCoverageLedger()
    poisoned.refusals = object()                 # not iterable: the summary cannot be built
    assert poisoned.summary() is None

    # End to end, through the ledger the capture builds for itself (it replaces any ledger
    # planted on the driver beforehand, which is the point of that reset).
    class _PoisonedLedger(hinge._EnumerationCoverageLedger):
        def __init__(self):
            super().__init__()
            self.refusals = object()

    drv = _drv(WorldAdb())
    dbg = _RecordingLog()
    drv._dbg = dbg

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "_EnumerationCoverageLedger", _PoisonedLedger)
        profile = drv._capture_current()

    assert profile is not None and profile.items != (), "the read itself is untouched"
    assert dbg.row("capture")["coverage"] is None


def test_bugreport_still_parses_a_capture_row_carrying_the_coverage_aggregate(tmp_path):
    """bugreport locates rows by action name and reads only the keys it knows, so no reader
    change is owed. Pinned against the real writer rather than a hand-built row, because the
    aggregate has to survive `json.dumps` on the way in."""
    from operation_love import bugreport      # local: no other test in this file needs it

    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="coverage-aggregate")

    assert drv._capture_current() is not None
    lines = (drv._dbg.dir / "actions.jsonl").read_text().splitlines()
    capture = next(rec for rec in bugreport._action_records(lines)
                   if rec.get("action") == "capture")
    assert isinstance(capture["coverage"]["basis"], dict)
    assert set(capture["coverage"]["basis"]) == set(_COVERAGE_BASES)
    assert "read " in bugreport._latest_completed_capture_timing_md(lines)
    assert bugreport._item_manifest_summary_md(lines)


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

def test_the_enumeration_step_is_sized_from_page_geometry_never_the_config_cadence():
    """`scroll_step.plan_coverage_step` (2026-08-24), on the gestures actually issued. Production's
    `read_scroll_frac` (0.55, 1299px) is past `frameshift.estimate_shift`'s 900px trust window
    outright — a pair at that cadence is refused before any card-spacing question is even asked.
    Every gesture this read makes must instead land inside the coverage-aimed window: at or above
    the driver's own smallest legal read-scroll, and at or below the 540px trust-window ceiling
    (`_ENUM_TRUST_CEILING_BAND_FRAC` of this device's 1800px analysed band) — never above it, even
    when a still-open card's own depth would have permitted less."""
    adb = WorldAdb()
    drv = _drv(adb)

    drv._capture_current()

    assert adb.gestures, "the read must have scrolled"
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    ceiling_px = round(scroll_step._ENUM_TRUST_CEILING_BAND_FRAC * (_BAND1 - _BAND0))
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

    # Force the smallest legal step every gesture, decoupling this test from whatever cadence
    # plan_coverage_step's real page-geometry throttle happens to choose on this fixture's world
    # (2026-08-24: the coverage-aimed rule reads this same WorldAdb page in far fewer frames than
    # the old ratio rule did, since it is not bound to ~1/3 of the local card spacing). The
    # property under test is the THINNING (BUG 3's fix) firing when enumeration reads MORE frames
    # than `scroll_captures`, not any particular step size, so forcing the driver's own smallest
    # sanctioned read-scroll is the most direct way to guarantee that precondition here.
    real_plan = HingeDriver._plan_enumeration_step

    def forced_floor_step(self, frame, x_frac, **kwargs):
        step = real_plan(self, frame, x_frac, **kwargs)
        floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
        return dataclasses.replace(
            step, frac=hinge._READ_SCROLL_FRAC_MIN, step_px=floor_px,
            window_px=(floor_px, floor_px))

    monkeypatch.setattr(HingeDriver, "_plan_enumeration_step", forced_floor_step)

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
    assert profile.items_unavailable_kind == "targeting_calibration"
    assert drv._profile_capture_limit == drv.scroll_captures
    assert {frac for frac, _lane in adb.gestures} == {drv.read_scroll_frac}


def test_rejected_live_targeting_calibration_is_typed_before_enumeration(monkeypatch):
    """The real session probe latches version-only drift before enumeration can start."""
    class LiveVersionAdb(WorldAdb):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

        def shell(self, command="", **kwargs):
            if command.startswith("dumpsys package "):
                return "versionName=10.1.0\n"
            return super().shell(command, **kwargs)

        def scroll_up(self, frac, x_frac=0.5, *, _timing=None):
            return super().scroll_up(frac, x_frac)

    class C:
        apps = {"hinge": {
            "serial": "pixel", "halt_on_error": False,
            "targeting_calibration": {
                **_TARGETING_CALIBRATION, "hinge_version_name": "10.0.1",
            },
        }}

    monkeypatch.setattr(hinge, "Adb", LiveVersionAdb)
    monkeypatch.setattr("operation_love.platforms.unavailable_reason",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(HingeDriver, "_require_vision", lambda _self: None)
    monkeypatch.setattr(HingeDriver, "_make_touch", lambda self: self._adb)
    drv = HingeDriver(C())
    drv.set_opener_enabled(True)
    drv.open_session()

    assert drv.targeting_calibration is None
    assert "'10.1.0'/(1080, 2400)" in drv._targeting_calibration_unavailable
    assert "'10.0.1'/(1080, 2400)" in drv._targeting_calibration_unavailable

    def enumeration_must_not_start(*_args, **_kwargs):
        raise AssertionError("rejected targeting calibration must block before enumeration")

    monkeypatch.setattr(drv, "_confirm_enumeration_top", enumeration_must_not_start)
    monkeypatch.setattr(drv, "_index_captured_items", enumeration_must_not_start)

    profile = drv._capture_current()

    assert profile is not None and profile.photos
    assert "10.1.0" in profile.items_unavailable
    assert profile.items_unavailable_kind == "targeting_calibration"
    assert drv._profile_capture_limit == drv.scroll_captures
    drv.close()


def test_runtime_targeting_calibration_drift_blocks_the_next_enumeration_before_reading(
        monkeypatch):
    """A long-running session rechecks its binding before spending another fine-cadence read."""
    class RuntimeDriftAdb(WorldAdb):
        version = "9.134.0"

        def shell(self, command="", **kwargs):
            if command.startswith("dumpsys package "):
                return f"versionName={self.version}\n"
            return super().shell(command, **kwargs)

    adb = RuntimeDriftAdb()
    drv = _drv(adb)
    assert drv._item_enumeration_blocker() == ""
    adb.version = "10.1.0"

    def enumeration_must_not_start(*_args, **_kwargs):
        raise AssertionError("runtime calibration drift must block before enumeration")

    monkeypatch.setattr(drv, "_confirm_enumeration_top", enumeration_must_not_start)
    monkeypatch.setattr(drv, "_index_captured_items", enumeration_must_not_start)

    profile = drv._capture_current()

    assert profile is not None and profile.photos
    assert profile.items == () and profile.item_context == ()
    assert "10.1.0" in profile.items_unavailable
    assert profile.items_unavailable_kind == "targeting_calibration"
    assert drv.targeting_calibration is None
    assert drv._profile_capture_limit == drv.scroll_captures


def test_a_superseded_v1_selection_policy_gets_its_own_parse_refusal():
    """The driver repeats config's policy pin at the gesture boundary, and names the fix.

    v1 was measured against a selection contract that no longer exists (C1-C4 replaced it, see
    ops/STILL-PHOTO-DISCRIMINATOR.md section 3), so retyping the id over an old mapping would be
    exactly the wrong repair. An UNKNOWN id keeps the generic "unsupported" wording.
    """
    superseded, reason = hinge._parse_targeting_calibration(
        {**_TARGETING_CALIBRATION, "item_selection_policy_id": "hinge_photos_only_v1"},
        "pixel", identity_band=hinge.HINGE_SPEC.identity_band,
        content_band=hinge.HINGE_SPEC.content_band)

    assert superseded is None
    assert "hinge_photos_only_v1 is superseded by hinge_photos_only_v2" in reason
    assert "recalibrate under the current policy" in reason

    unknown, unknown_reason = hinge._parse_targeting_calibration(
        {**_TARGETING_CALIBRATION, "item_selection_policy_id": "hinge_written_only_v9"},
        "pixel", identity_band=hinge.HINGE_SPEC.identity_band,
        content_band=hinge.HINGE_SPEC.content_band)

    assert unknown is None
    assert "must be the supported 'hinge_photos_only_v2' policy" in unknown_reason

    parsed, no_reason = hinge._parse_targeting_calibration(
        _TARGETING_CALIBRATION, "pixel", identity_band=hinge.HINGE_SPEC.identity_band,
        content_band=hinge.HINGE_SPEC.content_band)

    assert no_reason is None and parsed is not None


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
    assert profile.items_unavailable_kind == ""
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
    assert profile.items_unavailable_kind == ""
    # `items_unnumbered` is the DIFFERENT field for "enumeration finished and numbered nothing"
    # (found+fixed 2026-08-22); a genuine index failure never reaches that state, so it must stay
    # empty here even though `items` is also empty -- the two reasons are never both live.
    assert profile.items_unnumbered == ""
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
    assert sidecar["schema_version"] == 7
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


def test_item_index_refusal_sidecar_carries_the_unanchored_strip_evidence_bugreport_reads(tmp_path):
    """The screen-fixed proof must survive into the sidecar, under the key the report reads.

    `_screen_fixed_islands` decides whether a leading strip is chrome pinned to the SCREEN from
    two things: identical frame rows at spread page offsets, and an identical pixel digest. The
    first is recoverable from any sidecar; the second exists only if `segment.Block.content_digest`
    is written out. Without it a replay can see that a strip held one frame position and still not
    conclude anything, which is precisely the diagnosis the 2026-08-28 incident needed and the
    report could not produce.

    This is a CROSS-MODULE contract and that is why it is pinned here: `hinge.py` writes the key
    and `bugreport.py` reads it, they share no import, and a rename on either side would leave both
    files individually correct and the diagnosis silently blank. The `reason` is carried alongside
    because it holds the measured widest-span-against-card-width figure that the per-frame rule
    turns on, which has only 6px of margin and must be visible in a report before it drifts.
    """
    from operation_love import bugreport

    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-unanchored")

    class _Unobserved:
        observed = False
        kind = segment.EDGE_UNANCHORED_ISLAND
        run_px = 106

    class _Strip:
        y0, y1, complete = 368, 411, False
        top, bottom = _Unobserved(), _Unobserved()
        hearts = ()
        kind = segment.BLOCK_UNANCHORED
        content_digest = "d" * 64
        reason = "a leading strip this frame cannot place: ... 968px of 974px ..."

    class _Seg:
        blocks = (_Strip(),)

    class _Index:
        frames = (_Seg(), _Seg())
        source_frame_indices = (0, 1)
        offsets = (0, 900)
        shifts = ()
        failures = ()

    drv._item_index_refused([b"frame 0", b"frame 1"], "index refused", _Index())
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    refusal = next(rec for rec in records if rec["action"] == "item_index_refused")
    sidecar = json.loads((drv._dbg.dir / refusal["evidence_sidecar"]).read_text())

    block = sidecar["all_frame_geometry"][0]["blocks"][0]
    assert block["kind"] == segment.BLOCK_UNANCHORED
    assert block["content_digest"] == "d" * 64
    assert "968px of 974px" in block["unanchored_reason"]

    # THE SEAM, closed end to end rather than by comparing key spellings: feed the sidecar this
    # writer just produced to the reader that consumes it, and require the reader to reach its
    # PROVEN verdict. Two frames, the same 43px rows, the same digest, 900px of scroll between
    # them — which is exactly what `_screen_fixed_islands` calls screen-fixed. If either side
    # renames the key, or stops writing the digest, this drops to an undecidable verdict and goes
    # red; asserting the strings matched would not have caught the digest going missing.
    assert block["kind"] == bugreport._SIDECAR_UNANCHORED_BLOCK_KIND
    rendered = bugreport._item_index_geometry_md(
        (drv._dbg.dir / "actions.jsonl").read_text().splitlines(), drv._dbg.dir)
    assert "PROVEN screen-fixed" in rendered, rendered
    assert "UNDECIDABLE" not in rendered, rendered


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
    expected_max_step_px = item_index._pitch_relative_max_step(index.frames)
    assert expected_max_step_px > item_index._MAX_STEP_PX
    seen_max_steps: dict[str, list[int]] = {}

    def record_bound(name, helper):
        def wrapped(*args, **kwargs):
            seen_max_steps.setdefault(name, []).append(kwargs["max_step_px"])
            return helper(*args, **kwargs)
        return wrapped

    def unavailable_track_notes(*_args, **_kwargs):
        raise RuntimeError("synthetic optional video-track diagnostic failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "_video_track_deltas", unavailable_track_notes)
        for name in ("_layout_repaired_shift", "_edge_only_two_strip_shift",
                     "_structural_tail_shift", "_exact_multi_strip_shift",
                     "_measured_layout_bridge", "_structural_landmarks"):
            mp.setattr(hinge, name, record_bound(name, getattr(hinge, name)))
        drv._item_index_refused(
            photos, "frames 1 and 2 could not be put in one coordinate space", diagnostic_index)

    actions = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    action = next(record for record in actions if record["action"] == "item_index_refused")
    # One anchor before the failed pair and two look-ahead frames after its right endpoint.
    assert sorted(int(Path(name).stem.rsplit("_", 1)[1]) for name in action["evidence_frames"]) \
        == [0, 1, 2, 3, 4]
    sidecar = json.loads((drv._dbg.dir / action["evidence_sidecar"]).read_text())
    assert sidecar["schema_version"] == 7
    assert len(sidecar["pair_evidence"]) == 4
    measured_lookahead = sidecar["pair_evidence"][2]
    assert measured_lookahead["status"] == "measured"
    assert isinstance(measured_lookahead["delta_px"], int)
    assert measured_lookahead["strips"]
    assert {"frame_rows", "state", "delta_px", "score", "runner_up", "stddev", "search"} \
        <= set(measured_lookahead["strips"][0])
    assert measured_lookahead["max_step_px"] == expected_max_step_px
    assert all(seen_max_steps[name] and set(seen_max_steps[name]) == {expected_max_step_px}
               for name in seen_max_steps)
    assert {"_layout_repaired_shift", "_edge_only_two_strip_shift", "_structural_tail_shift",
            "_exact_multi_strip_shift", "_measured_layout_bridge", "_structural_landmarks"} \
        == set(seen_max_steps)
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
    #
    # NOT `_MAX_STEP_PX` (2026-08-24): that constant stopped being a global any reachable function
    # body reads live the day `_structural_landmarks` and its family took a `max_step_px`
    # PARAMETER instead (see item_index.py's "PITCH-RELATIVE LANDMARK-PAIRING BOUND" section) --
    # production now always threads a profile-derived value in explicitly, and `_MAX_STEP_PX`
    # survives only as that parameter's bare-test DEFAULT, captured once into each function's own
    # `__defaults__` at import time. A default is already part of the code hash below (see
    # `encoded_defaults`), but rebinding the MODULE-LEVEL name afterwards changes nothing a
    # function reads, which is a correct reflection of production reality, not a gap: production
    # never falls back to that default. `_AGREEMENT_TOLERANCE_PX` is `_matched_delta_clusters`'s
    # own live-read global (reached one level below the seed list, exactly what
    # `test_item_index_runtime_provenance_reaches_the_helpers_below_its_leaf_dependencies` checks
    # for) and keeps this test's original point intact.
    monkeypatch.setattr(
        item_index, "_AGREEMENT_TOLERANCE_PX", item_index._AGREEMENT_TOLERANCE_PX + 1)
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

    # `_AGREEMENT_TOLERANCE_PX` is read inside a function BODY (`_matched_delta_clusters`), so it
    # reaches the globals walk. Prove that here rather than assuming it: a constant that only
    # supplies a keyword default would NOT reach it (those are captured off the loaded function's
    # `__defaults__`, where a post-import rebinding correctly changes nothing), and this test
    # would then pass without exercising anything at all. `_MAX_STEP_PX` was this test's original
    # example and stopped being one 2026-08-24, when `_structural_landmarks` and its family moved
    # to a `max_step_px` PARAMETER (item_index.py's "PITCH-RELATIVE LANDMARK-PAIRING BOUND"); see
    # `test_item_index_runtime_provenance_hashes_loaded_indexer_not_worktree_source`'s own comment
    # on the same swap for the full reasoning.
    with monkeypatch.context() as patched:
        patched.setattr(
            item_index, "_AGREEMENT_TOLERANCE_PX", item_index._AGREEMENT_TOLERANCE_PX + 1)
        assert (hinge._item_index_runtime_provenance()["indexer_code_sha256"]
                != baseline["indexer_code_sha256"]), "pick a global the walk actually reads"

    cyclic: dict = {"frames": 3}
    cyclic["self"] = cyclic
    monkeypatch.setattr(item_index, "_AGREEMENT_TOLERANCE_PX", cyclic)

    provenance = hinge._item_index_runtime_provenance()
    assert provenance["indexer_code_sha256"] is not None
    assert provenance["algorithm_id"] == item_index.ITEM_INDEX_ALGORITHM_ID


def test_item_index_refusal_filesystem_evidence_failure_cannot_break_the_live_refusal(
        monkeypatch, tmp_path):
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="item-index-dossier-disk-failure")
    monkeypatch.setattr(
        hinge, "atomic_write_private_bytes",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("full")),
    )
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


@pytest.mark.parametrize("source_index", [True, False])
def test_recovered_index_rejects_boolean_source_frame_provenance(monkeypatch, source_index):
    """Booleans are integers in Python but never valid source-frame ordinals.

    Treating True as one or False as zero would silently crop a different source image than the
    recovered index claims. Refuse before any payload/crop builder can consume that mapping.
    """
    drv = _drv(WorldAdb())
    photos = [b"zero", b"one"]

    class _Index:
        usable = True
        frames = (object(),)
        source_frame_indices = (source_index,)
        offsets = (0,)
        identity = type("Identity", (), {"known": True, "reason": "ok"})()
        recovered_from_pair = None
        recovery_bridge = None

    monkeypatch.setattr(hinge, "build_item_index", lambda *_args, **_kwargs: _Index())
    monkeypatch.setattr(
        hinge, "build_item_payload",
        lambda *_args, **_kwargs: pytest.fail("invalid source provenance reached payload building"),
    )

    refusal = drv._index_captured_items(photos)

    assert "invalid frame provenance" in refusal
    assert drv._current_item_index is None
    assert drv._current_item_payload is None


@pytest.mark.parametrize("source_index", [True, False])
def test_refusal_provenance_normalizer_never_maps_boolean_frame_indices(source_index):
    """The refusal/debug path shares the main crop path's literal-index invariant."""
    index = SimpleNamespace(
        frames=(object(),), source_frame_indices=(source_index,))

    assert HingeDriver._item_index_source_indices(index, [b"zero", b"one"]) == (0,)


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
    person. That is a capture in which nothing distinguishes this profile from any other.

    WHICH REASON IT GIVES CHANGED ON 2026-08-28, and the change is the point. This fixture draws
    the chips row at EVERY offset, so its identity band is pinned to the screen -- exactly the
    Hinge 10.1.0 shape. The old message blamed the capture ("never scrolled far enough to reveal
    the header"), which would send an operator to re-run a capture forever against a check that
    cannot answer on this build. Now that `build_item_index` hands the measured page offsets to
    `capture_profile_identity`, the two causes are told apart from evidence and this one is named
    correctly. The refusal itself -- no items, no crops, no billed opener call, the ranker's
    frames untouched -- is unchanged, and that is what the rest of this test pins."""
    adb = WorldAdb(at_top=True)
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile is not None and profile.photos, "the frames are still the ranker's"
    assert profile.items == () and profile.item_context == ()
    assert "identity" in profile.items_unavailable
    assert "PINNED to the screen" in profile.items_unavailable
    assert "never scrolled far enough" not in profile.items_unavailable, (
        "a pinned band is not a capture that under-scrolled; naming the wrong cause here is the "
        "defect this assertion exists to keep fixed")
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
    """`plan_coverage_step` raises rather than substituting a larger step when even the smallest
    permitted gesture would risk scrolling an open card's top row out of the band. That must end
    the ENUMERATION (no crops, a stated reason) while the read finishes normally -- the frames
    have a second consumer that has no index in them to be wrong about."""
    adb = WorldAdb()
    drv = _drv(adb)

    def _refuse(*a, **k):
        raise scroll_step.ScrollStepError("the local card spacing is 400px")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_coverage_step", _refuse)
        profile = drv._capture_current()

    assert profile.items == ()
    assert "the local card spacing is 400px" in profile.items_unavailable
    assert profile.items_unavailable_kind == ""
    assert len(profile.photos) > 1, "the read must have carried on to the bottom"
    assert {frac for frac, _lane in adb.gestures} == {drv.read_scroll_frac}


def test_training_stops_the_read_immediately_when_enumeration_becomes_impossible():
    """Training cannot score or label a profile without a verifiable targeted opener. Once the
    planner refuses, later ordinary-read gestures have no consumer and must not touch the device;
    AUTO's ranker-only continuation remains covered by the control immediately above."""
    adb = WorldAdb()
    drv = _drv(adb)
    drv.begin_training_session()

    def _refuse(*a, **k):
        raise scroll_step.ScrollStepError("the local card spacing is 400px")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "plan_coverage_step", _refuse)
        profile = drv._capture_current()

    assert profile is not None and len(profile.photos) == 1
    assert profile.items == ()
    assert "the local card spacing is 400px" in profile.items_unavailable
    assert adb.gestures == [], "no discarded ordinary-read gesture should follow the refusal"


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

    def _sleep(_seconds, _should_stop=None):
        # 2026-08-24: the read no longer sleeps a dwell on EVERY iteration (see
        # _READ_PAUSE_COUNTS in hinge.py), so "call 1 is the dwell, call 2 is the settle" is no
        # longer a safe assumption -- this profile's one randomized pause might land on
        # iteration 0, some other iteration, or not be drawn at all. What is still guaranteed,
        # by the read loop's own ordering, is that the pause (if any) for an iteration always
        # runs BEFORE that iteration's scroll, and the settle always runs AFTER it -- so the
        # first call seen once a scroll has actually happened is unambiguously a settle.
        # Interrupting exactly that one call reproduces "read dwell (if any) completes; the
        # post-scroll settle is interrupted" regardless of which iteration drew the pause.
        return adb.scrolls == 0

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


# =====================================================================================
# The K-candidate walk (2026-08-23): a still-photo dwell used to cover exactly ONE card per
# capture -- whichever one happened to have a complete sighting in the read's own last frame,
# i.e. wherever the read stopped -- because that card is free (the phone is already parked on
# it). `_still_photo_dwell_candidate_walk` spends up to `still_photo_dwell_candidates - 1` MORE
# real `item_nav.navigate_to_item` hops to cover more of the profile per capture, each one
# proved by the identical two-burst evidence and returned to the entry before the next.
#
# `_full_read_capture` reuses this file's own `_LAYOUT` (four selectable cards) read all the way
# to the bottom, so its last frame completely sights only card 4 -- the free candidate -- leaving
# cards 3, 2 and 1 reachable only by a climb. `ProbeWorldAdb` (not the plain `WorldAdb` every
# other section of this file uses) is required throughout: navigation's ascending walk issues
# REVERSE strokes (`_scroll_up_one`), which only `ProbeWorldAdb.swipe` actually moves the world
# for -- the base fixture's `swipe` is a deliberate no-op (see its own docstring).
# =====================================================================================

_FULL_READ_SCROLLS = tuple(260 * i for i in range(10))


def _full_read_capture():
    """The index and frames for a read that reaches the bottom of the profile. See the guard
    test immediately below for why its last frame frees only card 4."""
    frames = [_frame(s) for s in _FULL_READ_SCROLLS]
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)
    assert index.usable, index.failures
    assert index.at_scroll_top
    assert index.translation == (1, 2, 3, 4)
    return index, frames


def test_the_full_read_fixture_leaves_only_the_last_card_free():
    """A guard on the fixture, not the driver: every walk assertion below depends on ordinal 4
    being the only one `dwell_card_rects` can resolve from the read's own frames, so cards 3, 2
    and 1 can only ever get evidence from a real navigation hop."""
    index, frames = _full_read_capture()
    assert set(item_crops.dwell_card_rects(index, len(frames) - 1)) == {4}


def test_default_still_photo_dwell_candidates_is_three():
    """The config default this feature ships behind (config.py validates
    `apps.hinge.still_photo_dwell_candidates` the same way it validates `scroll_captures`)."""
    assert _drv(WorldAdb()).still_photo_dwell_candidates == 3


def test_one_candidate_reproduces_the_pre_walk_driver_byte_for_byte(installed_still_photo_bound):
    """K=1 is the floor `_still_photo_dwell_candidate_walk` must be a complete no-op at: exactly
    the one free card's evidence, no extra screencap, no extra scroll gesture of any kind."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=1)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4}
    assert (adb.scrolls, adb.reverse_swipes) == (0, 0)
    assert adb.scroll == _FULL_READ_SCROLLS[-1]
    assert adb.taps == []


def test_candidate_walk_keeps_a_final_hop_beyond_the_return_envelope(
        monkeypatch, tmp_path, installed_still_photo_bound):
    """The final optional hop does not pay a return envelope it does not need.

    Its terminal anchor is the signed composition of the same measured navigation/correction/
    probe legs the ordinary return would consume.  This recovers the nearest otherwise-skipped
    candidate without weakening the two-burst video gate or making an intermediate hop start
    from an unverified page position.
    """
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=2)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="return-budget-skip")
    _required, return_cap_px, _leg_cap_px = drv._still_photo_dwell_walk_return_budget(index, 3)
    far_top = int(index.offsets[-1]) + _BAND0 - return_cap_px - 1
    far_blocks = tuple(
        dataclasses.replace(block, page_y0=far_top)
        if block.heart_ordinal == 3 else block
        for block in index.blocks)
    far_index = dataclasses.replace(index, blocks=far_blocks)
    # The synthetic page fixture begins at row zero, while this deliberately far *logical*
    # climb extends beyond it.  The terminal frame only stands for an already-measured result;
    # its bytes are not used by this branch, so keep the fixture in its representable range.
    target = _stub_target(_frame(0), (700, 1700), climbed_px=return_cap_px + 1)
    navigation_calls = []
    monkeypatch.setattr(
        hinge, "navigate_to_item",
        lambda *_args, **_kwargs: navigation_calls.append(1) or target)
    correction = hinge._CenteringCorrection(total_px=0, frame=target.frame)
    monkeypatch.setattr(
        HingeDriver, "_still_photo_dwell_over_navigated_target",
        lambda *_args, **_kwargs: (None, None, correction))
    returns = []
    monkeypatch.setattr(
        HingeDriver, "_still_photo_dwell_walk_return_to_entry",
        lambda *_args, **_kwargs: returns.append(1))

    evidence, anchor = drv._still_photo_dwell_candidate_walk(
        {4: object()}, frames=frames, index=far_index, mute_screen=lambda _f, _r: True,
        eligible_heart_ordinals={3},
        entry_anchor=hinge._MeasuredItemAnchor(frames[-1], 0))

    assert evidence.keys() == {4}
    assert anchor == hinge._MeasuredItemAnchor(
        target.frame, -(return_cap_px + 1))
    assert navigation_calls == [1]
    assert returns == [], "the final hop must not make a redundant return"
    assert (adb.scrolls, adb.reverse_swipes) == (0, 0)
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    skipped = [record for record in records
               if record["action"] == "still_photo_dwell_walk_candidate"]
    assert [row["outcome"] for row in skipped] == ["parked_unproved"]


def test_candidate_walk_still_skips_an_intermediate_hop_beyond_the_return_envelope(
        monkeypatch, installed_still_photo_bound):
    """Only the final hop can retain its terminal anchor.

    A later hop would need the enumeration entry again, so the fixed, vetted return envelope
    remains an early refusal here.  This is the guard against accidentally turning the terminal
    coverage fix into an unbounded multi-hop walk.
    """
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)
    _required, return_cap_px, _leg_cap_px = drv._still_photo_dwell_walk_return_budget(index, 3)
    far_top = int(index.offsets[-1]) + _BAND0 - return_cap_px - 1
    far_blocks = tuple(
        dataclasses.replace(block, page_y0=far_top)
        if block.heart_ordinal in {3, 2} else block
        for block in index.blocks)
    far_index = dataclasses.replace(index, blocks=far_blocks)
    navigation_calls = []
    monkeypatch.setattr(
        hinge, "navigate_to_item",
        lambda *_args, **_kwargs: navigation_calls.append(1))
    stop_checks = {"count": 0}

    def stop_before_second_candidate():
        stop_checks["count"] += 1
        return stop_checks["count"] > 1

    evidence, anchor = drv._still_photo_dwell_candidate_walk(
        {4: object()}, frames=frames, index=far_index, mute_screen=lambda _f, _r: True,
        eligible_heart_ordinals={3, 2}, should_stop=stop_before_second_candidate,
        entry_anchor=hinge._MeasuredItemAnchor(frames[-1], 0))

    assert evidence.keys() == {4}
    assert anchor == hinge._MeasuredItemAnchor(frames[-1], 0)
    assert navigation_calls == []
    assert (adb.scrolls, adb.reverse_swipes) == (0, 0)


def test_candidate_walk_still_navigates_a_hop_inside_the_vetted_return_envelope(
        monkeypatch, installed_still_photo_bound):
    """The pre-skip constrains only hops that cannot fit the established return protocol."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=2)
    calls = []
    real_navigate = hinge.navigate_to_item

    def counted_navigate(*args, **kwargs):
        calls.append(kwargs.get("entry_reference"))
        return real_navigate(*args, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", counted_navigate)
    evidence, anchor = drv._still_photo_dwell_candidate_walk(
        {4: object()}, frames=frames, index=index, mute_screen=lambda _f, _r: True,
        eligible_heart_ordinals={3},
        entry_anchor=hinge._MeasuredItemAnchor(frames[-1], 0))

    assert set(evidence) == {3, 4}
    assert len(calls) == 1
    assert anchor is not None
    assert adb.reverse_swipes > 0


def _overshot_navigation_recovery(frame: bytes, page_shift_px: int):
    """A live-shaped measured refusal payload for the candidate-walk cleanup tests."""
    return item_nav.NavigationRecovery(
        frame=frame, page_shift_px=page_shift_px, step_index=5, frame_index=6,
        planned_step_px=243, achieved_step_px=514, bound_px=363, frac=0.11,
        window_px=(219, 363), basis=scroll_step.STEP_MEASURED, spacing_px=1027,
        sized_against_px=1027, measurement_delta_px=-514,
        measurement_status="measured", measurement_confidence=1.0,
        measurement_agreeing=9, measurement_dissenting=0, measurement_eligible=9,
        violation="moved 514px past its 363px aliasing bound")


def test_measured_overshot_candidate_returns_to_a_nonzero_entry_residual_and_keeps_evidence(
        monkeypatch, tmp_path, installed_still_photo_bound):
    """An overlarge measured step is rejected and returned from, and the walk goes ON.

    This mirrors the Rachel incident: another card was already proved and a candidate overshot
    after a real upward walk.  The refused ordinal is still refused -- never retried, never
    substituted -- but once the return to the entry is MEASURED AND VERIFIED the loop's own
    precondition is back, so the candidates after it are no longer forfeited.

    Abandoning them was a real cost, found on a live run 2026-08-27: a refusal on the FIRST
    candidate ended a walk with 1 of 6 photo candidates dwelled, and the one-item payload that
    produced dropped doc 5.6's like-sheet verification from its half-nearest-neighbour separation
    proof onto the far weaker held-out one-item ceiling (measured on that profile: a 37.506 bound
    with a second item, against 7.000 with one).

    Here `navigate_to_item` overshoots for EVERY candidate, so the walk is stopped by the
    `_STILL_PHOTO_WALK_RETURNED_REFUSAL_BUDGET` rather than by any one refusal -- which is the
    other half of the contract: continuing is bounded, because each of these has already spent a
    climb and a measured return leg.
    """
    index, frames = _full_read_capture()
    entry_scroll = _FULL_READ_SCROLLS[-1] + 20
    terminal_scroll = entry_scroll - 260
    adb = ProbeWorldAdb(start=terminal_scroll)
    drv = _drv(adb, still_photo_dwell_candidates=3)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="overshot-walk-return")
    entry_anchor = hinge._MeasuredItemAnchor(_frame(entry_scroll), 20)
    earlier_evidence = {4: object()}
    recovery = _overshot_navigation_recovery(_frame(terminal_scroll), -260)
    calls = []

    def overshot(_driver, _index, model_index, **_kwargs):
        calls.append(model_index)
        raise item_nav.ItemNavigationError(
            item_nav.NAV_SCROLL_OVERSHOT, "synthetic measured overshoot",
            frame=recovery.frame, frame_index=recovery.frame_index, recovery=recovery)

    monkeypatch.setattr(hinge, "navigate_to_item", overshot)
    evidence, restored = drv._still_photo_dwell_candidate_walk(
        earlier_evidence, frames=frames, index=index, mute_screen=lambda _f, _r: True,
        entry_anchor=entry_anchor)

    assert evidence == earlier_evidence, "a refused candidate never adds evidence"
    # Two refusals are absorbed and the third spends the budget, so the walk tries three
    # candidates rather than dying on the first.
    assert calls == [3, 2, 1]
    assert restored is not None
    # THE INVARIANT THAT MATTERS UNDER REPETITION: the anchor never lies about where the phone
    # is. Not asserted equal to the ENTRY's frame/shift, because this test deliberately feeds the
    # same fixed `recovery` to every refusal -- after the first restore that recovery is claiming
    # a 260px displacement the phone no longer has, so each further return leg walks a distance
    # the world does not owe and lands a little short. That is a property of the stub, not of the
    # driver; what the driver owes is that the anchor it hands back DESCRIBES the resulting
    # position, and it does. The exact single-refusal residual is pinned in
    # `test_one_returned_refusal_is_absorbed_and_a_later_candidate_is_still_proved`, where the
    # recovery is coherent and the number means something.
    assert restored.frame == _frame(adb.scroll)
    drift_bound = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert abs(adb.scroll - entry_scroll) < drift_bound, (
        "each return still had to land inside the ordinary entry gate to be accepted at all")

    records = [json.loads(line) for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    rows = [r for r in records if r["action"] == "still_photo_dwell_walk_candidate"]
    assert [r["outcome"] for r in rows] == ["navigation_refused_returned"] * 3
    assert [r["navigation_refusal"]["walk"]["outcome"] for r in rows] == [
        "continued", "continued", "abandoned_budget_spent"]
    assert [r["navigation_refusal"]["walk"]["returned_refusals_spent"] for r in rows] == [1, 2, 3]
    row = rows[0]
    assert row["navigation_refusal"].pop("walk") == {
        "outcome": "continued", "returned_refusals_spent": 1,
        "returned_refusal_budget": hinge._STILL_PHOTO_WALK_RETURNED_REFUSAL_BUDGET}
    assert row["navigation_refusal"] == {
        "schema_version": 1,
        "code": item_nav.NAV_SCROLL_OVERSHOT,
        "frame_index": 6,
        "terminal_shift_px": -260,
        "return_outcome": "restored",
        "restored_page_shift_px": 20,
        "planned": {"step_px": 243, "bound_px": 363, "frac": 0.11,
                    "window_px": [219, 363], "basis": scroll_step.STEP_MEASURED,
                    "spacing_px": 1027, "sized_against_px": 1027},
        "achieved": {"climb_px": 514, "overshoot_px": 151,
                     "measurement_delta_px": -514, "measurement_status": "measured",
                     "measurement_confidence": 1.0, "measurement_agreeing": 9,
                     "measurement_dissenting": 0, "measurement_eligible": 9},
    }


def test_one_returned_refusal_is_absorbed_and_a_later_candidate_is_still_proved(
        monkeypatch, installed_still_photo_bound):
    """THE COVERAGE THIS FIX EXISTS FOR, and the reason it is worth the extra hop.

    One candidate overshoots and is returned from; every later candidate navigates normally. The
    walk must go on and actually PROVE one, because that second proved card is what gives doc
    5.6's like-sheet verification a neighbour to derive a separation bound from. On 2026-08-27
    the abandon cost exactly this: 1 of 6 photo candidates dwelled, a one-item payload, and a
    verification refused against the weakest ceiling the module has.

    The refused ordinal itself is still never dwelt -- NEVER SUBSTITUTE is unchanged; only the
    candidates AFTER it are recovered.
    """
    index, frames = _full_read_capture()
    entry_scroll = _FULL_READ_SCROLLS[-1]
    adb = ProbeWorldAdb(start=entry_scroll)
    drv = _drv(adb, still_photo_dwell_candidates=3)
    entry_anchor = hinge._MeasuredItemAnchor(_frame(entry_scroll), 0)
    real_navigate = item_nav.navigate_to_item
    calls = []

    def overshoot_once(driver, walk_index, model_index, **kwargs):
        calls.append(model_index)
        if len(calls) == 1:
            # A COHERENT recovery: the phone really is where this says it is, so the return leg
            # owes exactly this displacement and the residual below means something.
            frame = driver._screencap()
            shift = drv._measured_page_shift(entry_anchor.frame, frame)
            raise item_nav.ItemNavigationError(
                item_nav.NAV_SCROLL_OVERSHOT, "synthetic measured overshoot", frame=frame,
                frame_index=6, recovery=_overshot_navigation_recovery(frame, shift or 0))
        return real_navigate(driver, walk_index, model_index, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", overshoot_once)
    evidence, anchor = drv._still_photo_dwell_candidate_walk(
        {4: object()}, frames=frames, index=index, mute_screen=lambda _f, _r: True,
        entry_anchor=entry_anchor)

    assert len(calls) > 1, "the walk did not stop at the refused candidate"
    assert calls[0] == 3
    assert 3 not in evidence, "the refused ordinal is never dwelt, retried or substituted"
    assert set(evidence) - {4}, "a later candidate was actually proved after the refusal"
    assert anchor is not None


def test_a_returned_refusal_whose_return_is_unverified_still_abandons_the_walk(
        monkeypatch, installed_still_photo_bound):
    """The other half of the contract: continuing is licensed by the RETURN, not by the refusal.

    An unverified return leaves the phone somewhere this method cannot account for, so the walk
    must stop exactly as it always did -- and hand back no anchor, so the capture's own
    `refused_dwell_anchor` gate fires rather than a later navigation trusting a stale one.
    """
    index, frames = _full_read_capture()
    entry_scroll = _FULL_READ_SCROLLS[-1]
    adb = ProbeWorldAdb(start=entry_scroll)
    drv = _drv(adb, still_photo_dwell_candidates=3)
    recovery = _overshot_navigation_recovery(_frame(entry_scroll - 260), -260)
    calls = []

    def overshot(_driver, _index, model_index, **_kwargs):
        calls.append(model_index)
        raise item_nav.ItemNavigationError(
            item_nav.NAV_SCROLL_OVERSHOT, "synthetic measured overshoot",
            frame=recovery.frame, frame_index=recovery.frame_index, recovery=recovery)

    monkeypatch.setattr(hinge, "navigate_to_item", overshot)
    monkeypatch.setattr(drv, "_return_to_entry_from_measured_position",
                        lambda *_a, **_k: None)
    evidence, anchor = drv._still_photo_dwell_candidate_walk(
        {4: object()}, frames=frames, index=index, mute_screen=lambda _f, _r: True,
        entry_anchor=hinge._MeasuredItemAnchor(_frame(entry_scroll), 0))

    assert set(evidence) == {4}
    assert anchor is None
    assert calls == [3], "an unverified return abandons on the first refusal, as it always did"


def test_candidate_walk_without_a_measured_recovery_still_refuses_the_capture_anchor(
        monkeypatch, installed_still_photo_bound):
    """Legacy/unknown errors cannot be converted into an assumed zero-position cleanup."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)
    calls = []

    def unknown(_driver, _index, model_index, **_kwargs):
        calls.append(model_index)
        raise item_nav.ItemNavigationError("legacy_unknown", "no terminal measurement")

    monkeypatch.setattr(hinge, "navigate_to_item", unknown)
    evidence, anchor = drv._still_photo_dwell_candidate_walk(
        {4: object()}, frames=frames, index=index, mute_screen=lambda _f, _r: True,
        entry_anchor=hinge._MeasuredItemAnchor(frames[-1], 0))

    assert set(evidence) == {4}
    assert anchor is None
    assert calls == [3]


def test_three_candidates_produce_evidence_for_three_distinct_hearts(installed_still_photo_bound):
    """The headline: K=3 walks two real navigation hops beyond the free card, bottom-most first
    (heart 3, then heart 2), and both come back proved."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)

    evidence, anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4, 3, 2}
    for ordinal, dwell in evidence.items():
        assert dwell.dwell_exact is True, ordinal
    # The first hop returned before the second began.  The final hop retains its measured
    # terminal frame instead of paying a redundant return, and that anchor must describe the
    # real screen exactly so `_index_captured_items` can rebase the original item table there.
    assert anchor is not None
    assert anchor.frame == _frame(adb.scroll)
    assert anchor.page_shift_px == adb.scroll - _FULL_READ_SCROLLS[-1]


def test_measured_dwell_walk_rebases_the_index_to_its_terminal_frame_for_targeting(
        monkeypatch, installed_still_photo_bound):
    """A dwell/reattach walk can finish a short, measured distance beyond the read's old
    terminal frame. Its terminal frame, not the stale read frame, must become navigation's zero
    point while model-item numbering remains in the original heart space."""
    index, frames = _full_read_capture()
    old_terminal = _FULL_READ_SCROLLS[-1]
    # The synthetic world clamps at 2400 and this read ends at 2340, so keep both measured legs
    # inside the world while still finishing at a non-zero signed residual beyond the old entry.
    walk_frames = (frames[-1], _frame(old_terminal + 60), _frame(old_terminal + 20))
    adb = ProbeWorldAdb(start=old_terminal + 20)
    drv = _drv(adb, still_photo_dwell_candidates=3)

    leg_shifts = [drv._measured_page_shift(before, after)
                  for before, after in zip(walk_frames, walk_frames[1:], strict=False)]
    assert leg_shifts == [60, -40]
    assert all(shift is not None for shift in leg_shifts)
    terminal_shift = sum(leg_shifts)

    def measured_terminal_walk(self, seen_frames, live_index, should_stop=None, **_kwargs):
        assert seen_frames[-1] == frames[-1]
        assert live_index.translation == index.translation
        return {}, hinge._MeasuredItemAnchor(walk_frames[-1], terminal_shift)

    monkeypatch.setattr(HingeDriver, "_still_photo_dwell", measured_terminal_walk)

    assert drv._index_captured_items(frames) == ""
    assert drv._current_item_anchor == walk_frames[-1]
    assert drv._current_item_index.translation == index.translation == (1, 2, 3, 4)
    assert drv._current_item_payload.translation == (1, 2, 3, 4)
    assert drv._current_item_index.offsets == tuple(
        offset + terminal_shift for offset in index.offsets)

    seen = {}
    real_navigate = hinge.navigate_to_item

    def check_current_anchor(driver, live_index, navigation_index, **kwargs):
        seen.update(index=live_index, navigation_index=navigation_index,
                    entry_reference=kwargs["entry_reference"])
        return real_navigate(driver, live_index, navigation_index, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", check_current_anchor)
    point = drv._navigate_to_model_item(1)

    assert point[0] > 0 and point[1] > 0
    assert seen == {
        "index": drv._current_item_index,
        "navigation_index": 1,
        "entry_reference": walk_frames[-1],
    }
    assert adb.taps == [], "locating an opener target must never tap by itself"


def test_candidate_walk_return_residual_keeps_direct_opener_count_in_index_space(
        monkeypatch, installed_still_photo_bound):
    """An accepted, nonzero candidate-walk return residual is signed page movement, not debt.

    The walk climbed 300px from the read terminal frame, then its one return stroke delivered
    only 280px.  That leaves the returned frame 20px ABOVE the old entry in page space, which
    is inside the one-gesture drift bound and therefore accepted.  Installing that returned
    anchor exactly as `_index_captured_items` does must still let the selected opener target
    navigate through item-nav's real count/index cross-check.
    """
    index, frames = _full_read_capture()
    old_terminal = _FULL_READ_SCROLLS[-1]
    climbed_px = 300
    residual_px = 20
    target_scroll = old_terminal - climbed_px
    adb = ProbeWorldAdb(start=target_scroll)
    drv = _drv(adb, still_photo_dwell_candidates=2)
    target = _stub_target(_frame(target_scroll), (400, 1200), climbed_px=climbed_px)
    correction = hinge._CenteringCorrection(total_px=0, frame=target.frame)

    def underdeliver_return(frac, _x_frac):
        # The return helper asks for the 300px debt but the device delivers 200px.  Its next
        # screenshot is therefore a genuine, accepted -100px residual from the old entry.
        adb.scroll += scroll_step.step_px_for_frac(frac, _H) - residual_px

    monkeypatch.setattr(drv, "_scroll_down_one", underdeliver_return)
    returned = drv._still_photo_dwell_walk_return_to_entry(
        target, None, hinge._MeasuredItemAnchor(frames[-1], 0), correction)

    assert returned is not None
    assert returned.frame == _frame(old_terminal - residual_px)
    assert returned.page_shift_px == -residual_px

    anchored_index = drv._rebase_item_index_for_anchor(index, returned.page_shift_px)
    assert anchored_index is not None
    drv._current_item_index = anchored_index
    drv._current_item_anchor = returned.frame
    drv._current_item_payload = item_crops.build_item_payload(frames, index)

    point = drv._navigate_to_model_item(2)

    assert point[0] > 0 and point[1] > 0
    assert adb.taps == [], "locating the selected opener target must not tap"


def test_unmeasured_dwell_walk_rejects_the_capture_before_targeting_state(
        monkeypatch, installed_still_photo_bound):
    """An unknown dwell/reattach leg cannot become an assumed zero offset. The capture is
    refused and `_navigate_to_model_item` must stop before calling item navigation."""
    _index, frames = _full_read_capture()
    drv = _drv(ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1]), still_photo_dwell_candidates=3)

    monkeypatch.setattr(HingeDriver, "_still_photo_dwell",
                        lambda self, *_args, **_kwargs: ({}, None))
    navigation_calls = []
    monkeypatch.setattr(
        hinge, "navigate_to_item",
        lambda *_args, **_kwargs: navigation_calls.append(1))

    refusal = drv._index_captured_items(frames)

    assert "complete measured chain" in refusal
    assert drv._current_item_index is None
    assert drv._current_item_payload is None
    assert drv._current_item_anchor is None
    with pytest.raises(hinge.HingeTargetingError, match="holds no item index"):
        drv._navigate_to_model_item(1)
    assert navigation_calls == []


def test_the_number_of_candidates_never_exceeds_k(installed_still_photo_bound):
    """K=2 on a profile with four selectable cards: only ONE extra candidate -- the one nearest
    the entry -- is ever attempted, never all three that are structurally available."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=2)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4, 3}
    assert 2 not in evidence and 1 not in evidence


def test_every_walk_gesture_goes_through_the_ledgered_scroll_helpers(installed_still_photo_bound):
    """Owner rule: best humanized interaction or fail loudly. Asserted on the driver's own scroll
    ledger against the fake transport's own gesture counters -- not on a comment -- because a
    gesture that bypassed `_scroll_down_one`/`_scroll_up_one` (a raw `adb.scroll_up`/`swipe` call)
    would move the fake phone without the ledger ever growing to match."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)

    drv._still_photo_dwell(frames, index)

    assert len(drv._capture_scroll_ledger) == adb.scrolls + adb.reverse_swipes
    assert adb.scrolls + adb.reverse_swipes > 0, "the walk must have actually moved the phone"


def test_a_navigation_refusal_on_the_second_candidate_abandons_the_walk(
        monkeypatch, installed_still_photo_bound):
    """Candidate 1 (heart 3) is proved and kept; candidate 2 (heart 2) refuses; candidate 3
    (heart 1) is NEVER attempted and never appears in the evidence -- no substitution, no retry
    of a different ordinal for the one that was asked for."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=4)
    real_navigate = hinge.navigate_to_item
    calls: list[int] = []

    def flaky_navigate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise hinge.ItemNavigationError("navigation_refused_for_test", "synthetic refusal")
        return real_navigate(*args, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", flaky_navigate)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4, 3}
    assert 2 not in evidence and 1 not in evidence
    assert len(calls) == 2, "candidate 3 (heart 1) must never be attempted"


def test_a_card_below_the_entry_is_stepped_over_not_treated_as_a_failed_walk(
        monkeypatch, installed_still_photo_bound):
    """found 2026-08-23, before the walk had ever run on a device: the bottom-most uncovered card
    is very often one the read left cut off BELOW the analysed band, and an ascending-only
    navigator refuses exactly that with NAV_ITEM_BELOW_ENTRY -- raised on the first frame, `if not
    steps`, with ZERO gestures spent. Abandoning the whole walk over it meant the single most
    likely first-candidate outcome silently reduced K back to 1. Nothing moved, so the walk steps
    over that card, keeps its budget, and the two hops that DO run are each judged on their own
    merits.

    Candidate 1 -- this fixture's own FIRST photo -- is one of those two hops, and its outcome
    changed once centring correction shipped (still 2026-08-23): it climbs close enough to the
    true top of this fixture's scrollable world that the reverse correction an above-centre park
    needs has nowhere left to reveal, rubber-bands at 0px, and the card is still outside Hinge's
    autoplay zone once the correction budget runs out. That is a genuine, structural refusal --
    `STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC` is never relaxed to paper over it, see
    `_still_photo_dwell_over_navigated_target`'s own CENTERING CORRECTION paragraph -- not a walk
    failure: it costs no burst, candidate 2 above it is unaffected and still fully proved, and the
    earlier skip still cost nothing."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)
    real_navigate = hinge.navigate_to_item
    asked: list[int] = []

    def below_entry_for_the_first_candidate(driver, idx, model_index, **kwargs):
        asked.append(model_index)
        if len(asked) == 1:
            raise hinge.ItemNavigationError(
                hinge.NAV_ITEM_BELOW_ENTRY, "synthetic: below the analysed band")
        return real_navigate(driver, idx, model_index, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", below_entry_for_the_first_candidate)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    # The skipped card (heart 3) contributes nothing, but the budget was NOT spent on it: two
    # hops still ran. Candidate 2 is proved; candidate 1 hits the scroll-top centring clamp
    # described above and goes unmeasured -- a per-candidate refusal, not a walk abandon (the
    # walk did not stop there; it simply ran out of candidates to try).
    assert 3 not in evidence
    assert set(evidence) == {4, 2}
    assert 1 not in evidence
    assert len(asked) == 3, "the skip must not consume one of the K hops"


def test_an_unreachable_run_of_cards_is_bounded_and_never_scans_the_whole_index(
        monkeypatch, installed_still_photo_bound):
    """The complement: a skip is cheap but not free -- navigate_to_item still pays a screencap, a
    segmentation and an identity compare before it can say "below the entry" -- so the walk is
    allowed only `_STILL_PHOTO_WALK_SKIP_SLACK` attempts beyond its hop budget rather than
    scanning every card on the profile at that price for no evidence."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=2)      # remaining == 1 hop
    asked: list[int] = []

    def always_below_entry(driver, idx, model_index, **kwargs):
        asked.append(model_index)
        raise hinge.ItemNavigationError(
            hinge.NAV_ITEM_BELOW_ENTRY, "synthetic: below the analysed band")

    monkeypatch.setattr(hinge, "navigate_to_item", always_below_entry)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4}, "no candidate was reachable, so only the free card survives"
    assert len(asked) <= 1 + hinge._STILL_PHOTO_WALK_SKIP_SLACK
    assert adb.scrolls == 0 and adb.reverse_swipes == 0, "a skip must move nothing at all"


def test_an_unverified_return_abandons_the_walk_and_stops_further_candidates(
        monkeypatch, installed_still_photo_bound):
    """Candidate 1 (heart 3) is proved; ITS OWN return to the entry cannot be verified; the walk
    stops there -- candidate 2 (heart 2), nearer the entry than the one that failed and the card
    that would ordinarily be tried next, is never attempted at all."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=4)
    navigate_calls: list[int] = []
    real_navigate = hinge.navigate_to_item

    def counting_navigate(*args, **kwargs):
        navigate_calls.append(1)
        return real_navigate(*args, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", counting_navigate)
    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_walk_return_to_entry",
                        lambda self, target, probe, entry_reference, correction: None)

    evidence, anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4, 3}
    assert 2 not in evidence and 1 not in evidence
    assert anchor is None, "an unverified return must not be represented as a zero-shift anchor"
    assert len(navigate_calls) == 1, "no candidate may be attempted after an unverified return"


def test_should_stop_between_candidates_stops_the_walk_with_no_partial_candidate(
        monkeypatch, installed_still_photo_bound):
    """`should_stop` becomes true only once candidate 1 (heart 3) has FULLY completed -- proof
    and a verified return -- so its own evidence is untouched by the stop; candidate 2 (heart 2)
    is then never attempted at all (caught at the walk's own top-of-loop check, before a single
    navigate_to_item call)."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=4)
    state = {"returns": 0}
    real_return = HingeDriver._still_photo_dwell_walk_return_to_entry

    def counting_return(self, target, probe, entry_reference, correction):
        ok = real_return(self, target, probe, entry_reference, correction)
        state["returns"] += 1
        return ok

    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_walk_return_to_entry", counting_return)
    navigate_calls: list[int] = []
    real_navigate = hinge.navigate_to_item

    def counting_navigate(*args, **kwargs):
        navigate_calls.append(1)
        return real_navigate(*args, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", counting_navigate)

    evidence, _anchor = drv._still_photo_dwell(frames, index, lambda: state["returns"] >= 1)

    assert set(evidence) == {4, 3}
    assert 2 not in evidence and 1 not in evidence
    assert len(navigate_calls) == 1, "no candidate after the first must ever be attempted"


def test_should_stop_mid_proof_yields_no_partial_candidate(
        monkeypatch, installed_still_photo_bound):
    """The stricter form of the same rule: `should_stop` fires WHILE candidate 2 (heart 2)'s own
    two-burst proof is running -- after navigation already parked it, so the phone genuinely
    moved -- not merely between candidates. That candidate must still get NO evidence entry at
    all: a `StillPhotoDwell` built from an interrupted burst would misrepresent an observation
    that was never completed, exactly as an interrupted burst on the free card already does
    (`_still_photo_dwell_burst`'s own STOP contract). The walk must not attempt a third
    candidate afterwards either."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=4)
    state = {"navigations": 0}
    real_navigate = hinge.navigate_to_item

    def counting_navigate(*args, **kwargs):
        target = real_navigate(*args, **kwargs)
        state["navigations"] += 1
        return target

    monkeypatch.setattr(hinge, "navigate_to_item", counting_navigate)

    evidence, _anchor = drv._still_photo_dwell(frames, index, lambda: state["navigations"] >= 2)

    assert set(evidence) == {4, 3}
    assert 2 not in evidence and 1 not in evidence
    assert state["navigations"] == 2, "candidate 2's navigation DID run, just not its proof"


def test_no_targeting_calibration_means_the_walk_never_runs(installed_still_photo_bound):
    """With no `apps.hinge.targeting_calibration` installed, `navigate_to_item` has no
    `identity_match_max_dist` to navigate with, so the walk must not run at all -- and the base
    (K=1) path is completely unaffected: same evidence, zero extra device gestures."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, targeting_calibration=False, still_photo_dwell_candidates=4)

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    assert set(evidence) == {4}
    assert (adb.scrolls, adb.reverse_swipes) == (0, 0)
    assert adb.scroll == _FULL_READ_SCROLLS[-1]


def test_the_walk_records_one_debug_row_per_candidate_it_attempted(
        tmp_path, installed_still_photo_bound):
    """"Record each candidate's outcome" (parked/proved/refused/return-unverified), following
    the surrounding `self._dbg.action(...)` style every other still-photo debug record uses."""
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="walk-candidates")

    evidence, _anchor = drv._still_photo_dwell(frames, index)

    records = [json.loads(line)
              for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    walk_rows = [r for r in records if r["action"] == "still_photo_dwell_walk_candidate"]
    assert [r["heart_ordinal"] for r in walk_rows] == [3, 2]
    assert all(r["outcome"] == "proved" for r in walk_rows)
    assert {r["heart_ordinal"]: r["dwell_exact"] for r in walk_rows} == {
        ordinal: evidence[ordinal].dwell_exact for ordinal in (3, 2)}
    dwell_rows = [r for r in records if r["action"] == "still_photo_dwell"]
    assert [r["heart_ordinals"] for r in dwell_rows] == [[4], [3], [2]]
    frame_rows = [r for r in records
                  if "dwell_frame_index" in r]
    assert {tuple(r["heart_ordinals"]) for r in frame_rows} == {(4,), (3,), (2,)}
    timing_rows = [r for r in records
                   if r["action"] == "still_photo_dwell_walk_candidate_timing"]
    assert [r["heart_ordinal"] for r in timing_rows] == [3, 2]
    assert all(r["outcome"] == "proved" for r in timing_rows)
    for row in timing_rows:
        assert all(row[key] >= 0 for key in (
            "navigation_s", "proof_s", "return_s", "unattributed_s"))
        assert (row["navigation_s"] + row["proof_s"] + row["return_s"]
                + row["unattributed_s"]
                == pytest.approx(row["candidate_wall_s"], abs=3e-6))


def test_candidate_walk_can_filter_out_written_cards_without_spending_its_budget(
        monkeypatch, installed_still_photo_bound):
    """The production content pre-pass supplies only confidently photographic heart ordinals.

    A written prompt near the read's end used to consume a full navigation, two dwell bursts and
    a re-attach probe before the payload's already-existing content gate discarded it.  Filtering
    the walk's pool must skip that ordinal and spend the same K budget on the next eligible card;
    it does not weaken the later still-media proof for any accepted card.
    """
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=3)
    asked: list[int] = []
    real_navigate = hinge.navigate_to_item

    def recording_navigate(*args, **kwargs):
        asked.append(args[2])
        return real_navigate(*args, **kwargs)

    monkeypatch.setattr(hinge, "navigate_to_item", recording_navigate)
    evidence, _anchor = drv._still_photo_dwell(
        frames, index, eligible_heart_ordinals={4, 2, 1})

    assert 3 not in evidence, "heart 3 represents the content-preflight rejection"
    assert set(evidence) == {4, 2}
    assert asked == [2, 1], (
        "heart 3 must be skipped without consuming a hop, and the remaining eligible slot "
        "must still be attempted")


def test_ineligible_parked_card_spends_no_base_dwell_at_k_one(
        monkeypatch, installed_still_photo_bound):
    """A prompt at the free parked position is not worth a burst the final gate discards.

    K=1 still means no navigation walk, so filtering that prompt produces no evidence and no
    input; importantly, it also produces no passive dwell cost.
    """
    index, frames = _full_read_capture()
    adb = ProbeWorldAdb(start=_FULL_READ_SCROLLS[-1])
    drv = _drv(adb, still_photo_dwell_candidates=1)
    bursts: list[int] = []
    monkeypatch.setattr(
        HingeDriver, "_still_photo_dwell_burst",
        lambda self, should_stop=None: bursts.append(1) or ([], 0.0))

    evidence, _anchor = drv._still_photo_dwell(
        frames, index, eligible_heart_ordinals={1, 2, 3})

    assert evidence == {} and bursts == []
    assert adb.scrolls == 0 and adb.reverse_swipes == 0


# =====================================================================================
# Centring correction after navigate_to_item parks a card off-zone (2026-08-23). Verified live on
# this repo's own harness that `item_nav.navigate_to_item` has no centring objective and parks a
# card -0.187..-0.287 against the +-0.150 `STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC` gate -- its
# ascending walk stops the instant the target block is merely `complete`, with the card's top
# edge freshly cleared and the rest of it still below. `tools/hinge_calibrate.py`'s own
# `_apply_bounded_centering_correction` is the proven fix `_still_photo_dwell_over_navigated_
# target` now mirrors.
#
# `SimpleNamespace` stands in for `item_nav.ItemTarget` / the index block below because the
# method under test only ever reads `.frame`/`.block_frame_rows` off the former and `.x0`/`.x1`
# off the latter -- a synthetic rect lets these tests place the "card" wherever a scenario needs
# without spending a real navigation hop to get there.
# =====================================================================================

def _stub_target(frame: bytes, rows: tuple[int, int], **extra):
    return SimpleNamespace(frame=frame, block_frame_rows=rows, **extra)


_STUB_BLOCK = SimpleNamespace(x0=_CARD_X0, x1=_CARD_X1)


def test_a_card_parked_out_of_zone_is_corrected_and_then_dwelt(installed_still_photo_bound):
    """The headline case this fix exists for: a card parked at -0.222 (past the +-0.150 gate, in
    the same direction this repo measured live: -0.187, -0.232, -0.287) is corrected INTO the
    zone before the two-burst proof runs, rather than being dwelt anyway and refused downstream
    for a guaranteed reason."""
    adb = ProbeWorldAdb(start=1200)
    drv = _drv(adb)
    frame = _frame(1200)
    rows = (400, 1200)                          # centre 800 vs band centre 1200 -> offset -0.222
    rect = (_CARD_X0, rows[0], _CARD_X1, rows[1])
    initial_offset = item_crops.card_center_offset_frac(
        rect, frame_height=_H, content_band=_CONTENT_BAND)
    assert initial_offset < -item_crops.STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC, (
        "fixture guard: this rect must actually start out of zone")
    target = _stub_target(frame, rows)

    dwell, probe, correction = drv._still_photo_dwell_over_navigated_target(
        target, _STUB_BLOCK, heart_ordinal=1, mute_screen=lambda _f, _r: True)

    assert dwell is not None, "a correctable card must still be dwelt, not abandoned"
    assert dwell.centered is True
    assert dwell.center_offset_frac == 0.0
    assert correction.total_px != 0, "the correction must have actually issued a gesture"
    assert adb.reverse_swipes > 0, "an above-centre card corrects with a REVERSE read-scroll"


def test_a_card_parked_above_centre_is_corrected_with_a_reverse_scroll(
        monkeypatch, installed_still_photo_bound):
    """Direction pin. `card_center_offset_frac`'s own docstring: NEGATIVE means the card sits
    ABOVE the content centre. Bringing it toward centre needs content to move DOWN the screen,
    which is the REVERSE read-scroll (`_scroll_up_one`) -- never the forward one. Get this
    backwards and the correction pushes the card FURTHER from centre instead of into it, exactly
    the class of bug this repo has hit live before (the re-attach probe's own exit stroke, once
    hardcoded backwards, found on a device). Asserted on the measured offset actually SHRINKING
    AND on which scroll counter moved -- either alone could pass by coincidence, an inverted
    direction fails both."""
    # Isolated from the re-attach probe on purpose: a centred, byte-exact burst earns one too,
    # and the probe's own exit leg is ALSO allowed to go backward first (see
    # `_still_photo_reattach_probe`'s docstring), which would blur which stroke this test is
    # pinning the direction of.
    monkeypatch.setattr(HingeDriver, "_still_photo_reattach_candidate",
                        staticmethod(lambda evidence: None))
    adb = ProbeWorldAdb(start=1200)
    drv = _drv(adb)
    frame = _frame(1200)
    rows = (400, 1200)                          # centre 800 vs band centre 1200 -> offset -0.222
    rect = (_CARD_X0, rows[0], _CARD_X1, rows[1])
    initial_offset = item_crops.card_center_offset_frac(
        rect, frame_height=_H, content_band=_CONTENT_BAND)
    assert initial_offset < 0, "fixture guard: this rect must start ABOVE centre"
    target = _stub_target(frame, rows)

    dwell, probe, correction = drv._still_photo_dwell_over_navigated_target(
        target, _STUB_BLOCK, heart_ordinal=1, mute_screen=lambda _f, _r: True)

    assert probe is None, "isolated above -- only the correction may have moved anything"
    assert dwell is not None and dwell.centered is True
    assert abs(dwell.center_offset_frac) < abs(initial_offset), "the offset must have shrunk"
    assert (adb.scrolls, adb.reverse_swipes) == (0, 1), (
        "an above-centre card must correct with exactly one REVERSE read-scroll, never a "
        "forward one")


def test_a_card_parked_below_centre_is_corrected_with_a_forward_scroll(
        monkeypatch, installed_still_photo_bound):
    """The other half of the same pin: POSITIVE means the card sits BELOW centre, and a FORWARD
    read-scroll (`_scroll_down_one`) is what brings it up -- the mirror image of the case above,
    proved the same way so an implementation that only gets ONE direction right cannot pass
    both."""
    monkeypatch.setattr(HingeDriver, "_still_photo_reattach_candidate",
                        staticmethod(lambda evidence: None))
    adb = ProbeWorldAdb(start=1200)
    drv = _drv(adb)
    frame = _frame(1200)
    rows = (1600, 2000)                         # centre 1800 vs band centre 1200 -> offset +0.333
    rect = (_CARD_X0, rows[0], _CARD_X1, rows[1])
    initial_offset = item_crops.card_center_offset_frac(
        rect, frame_height=_H, content_band=_CONTENT_BAND)
    assert initial_offset > 0, "fixture guard: this rect must start BELOW centre"
    target = _stub_target(frame, rows)

    dwell, probe, correction = drv._still_photo_dwell_over_navigated_target(
        target, _STUB_BLOCK, heart_ordinal=1, mute_screen=lambda _f, _r: True)

    assert probe is None, "isolated above -- only the correction may have moved anything"
    assert dwell is not None and dwell.centered is True
    assert abs(dwell.center_offset_frac) < abs(initial_offset), "the offset must have shrunk"
    assert (adb.scrolls, adb.reverse_swipes) == (1, 0), (
        "a below-centre card must correct with exactly one FORWARD read-scroll, never a "
        "reverse one")


def test_a_card_that_cannot_be_centred_within_budget_costs_no_burst(
        monkeypatch, installed_still_photo_bound):
    """The correction budget is bounded on purpose (see `_STILL_PHOTO_WALK_CENTERING_BUDGET`'s
    own comment): a card whose park is too far out of zone to fix within it must go UNMEASURED,
    never dwelt anyway, because the centring rung would refuse it regardless and a burst spent on
    it would buy nothing. Forcing the budget to 0 exercises this deterministically on ANY
    out-of-zone card, without depending on exactly how far one corrective stroke reaches."""
    monkeypatch.setattr(hinge, "_STILL_PHOTO_WALK_CENTERING_BUDGET", 0)
    burst_calls: list[int] = []

    def spy_burst(self, should_stop=None):
        burst_calls.append(1)
        return [], 0.0

    monkeypatch.setattr(HingeDriver, "_still_photo_dwell_burst", spy_burst)
    adb = ProbeWorldAdb(start=1200)
    drv = _drv(adb)
    frame = _frame(1200)
    rows = (400, 1200)
    target = _stub_target(frame, rows)

    dwell, probe, correction = drv._still_photo_dwell_over_navigated_target(
        target, _STUB_BLOCK, heart_ordinal=1, mute_screen=lambda _f, _r: True)

    assert dwell is None and probe is None
    assert correction.total_px == 0 and correction.frame is frame
    assert burst_calls == [], "a candidate that cannot be centred must cost no burst"
    assert (adb.scrolls, adb.reverse_swipes) == (0, 0), "budget exhausted before any gesture"


def test_an_unmeasurable_correction_refuses_the_whole_candidate(
        monkeypatch, installed_still_photo_bound):
    """`_measured_page_shift` returning None must be treated exactly like every other caller in
    this file treats it: 'this pass cannot claim to know where the card is now', never an
    assumption that the requested scroll landed where it was asked to
    (`_measured_page_shift`'s own contract). The whole candidate is refused -- dwell, probe AND
    correction all `None` -- so the caller can never be tempted to walk back a distance nobody
    actually measured."""
    monkeypatch.setattr(HingeDriver, "_measured_page_shift", lambda self, before, after: None)
    adb = ProbeWorldAdb(start=1200)
    drv = _drv(adb)
    frame = _frame(1200)
    rows = (400, 1200)
    target = _stub_target(frame, rows)

    result = drv._still_photo_dwell_over_navigated_target(
        target, _STUB_BLOCK, heart_ordinal=1, mute_screen=lambda _f, _r: True)

    assert result == (None, None, None)


def test_return_to_entry_accounts_for_a_corrective_scroll(
        monkeypatch, installed_still_photo_bound):
    """`_still_photo_dwell_walk_return_to_entry` now composes THREE measured displacements: the
    navigation climb, the centring correction, and (when one ran) the re-attach probe's own
    residual. Isolated here with a real correction and NO probe (monkeypatched off, same as the
    direction-pin tests above) so a wrong sign in the composition shows up as a verification
    failure rather than as a smaller residual that might still happen to clear the drift bound by
    coincidence."""
    monkeypatch.setattr(HingeDriver, "_still_photo_reattach_candidate",
                        staticmethod(lambda evidence: None))
    entry_reference = _frame(1200)
    entry_anchor = hinge._MeasuredItemAnchor(entry_reference, 0)
    adb = ProbeWorldAdb(start=1200)
    drv = _drv(adb)
    # Simulate a navigation hop that climbed 300px up from the entry (scroll 1200 -> 900) and
    # parked its card 400px above the content band's centre -- the same shape `navigate_to_item`
    # itself produces, without spending a real navigation hop's own gestures to get there.
    adb.scroll = 900
    target = _stub_target(_frame(900), (400, 1200), climbed_px=300)

    dwell, probe, correction = drv._still_photo_dwell_over_navigated_target(
        target, _STUB_BLOCK, heart_ordinal=1, mute_screen=lambda _f, _r: True)
    assert probe is None
    assert dwell is not None and dwell.centered is True
    assert correction.total_px != 0, "fixture guard: the correction must have actually run"

    returned = drv._still_photo_dwell_walk_return_to_entry(
        target, probe, entry_anchor, correction)
    assert returned is not None
    assert returned.page_shift_px == adb.scroll - 1200
    assert returned.frame == _frame(adb.scroll)
    drift_bound = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert abs(adb.scroll - 1200) < drift_bound


# =====================================================================================
# Return-chain trace: a 2026-08-30 live candidate climbed 4243px, completed its dwell/probe,
# then only reported ``return_unverified``.  The return helper is deliberately fail-closed; these
# tests pin both that decision and its diagnostic-only terminal record without teaching the trace
# to affect any measurement or retry.
# =====================================================================================

def _return_chain_records(drv):
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    return [record for record in records
            if record.get("action") == "still_photo_dwell_return_chain"]


def test_long_measured_return_chain_records_a_verified_terminal_anchor(monkeypatch, tmp_path):
    """A 4243px candidate return needs seven capped, individually measured strokes.

    This is intentionally a direct test of the return helper rather than the compact four-card
    walk fixture: it proves the attempt budget is derived from the debt and a long clean chain
    does not regress into the live run's generic refusal.
    """
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="long-return-chain")
    scrolls = []
    frames = iter([f"return-{i}".encode() for i in range(1, 8)])
    measurements = iter([630] * 6 + [463, 0])  # seven legs, then entry-frame corroboration
    monkeypatch.setattr(drv, "_sample_read_step", lambda *_a: (0.0, 0.0, 0.5))
    monkeypatch.setattr(drv, "_scroll_down_one", lambda frac, x: scrolls.append((frac, x)))
    monkeypatch.setattr(drv, "_screencap", lambda **_kw: next(frames))
    monkeypatch.setattr(drv, "_measured_page_shift",
                        lambda _before, _after: next(measurements))

    returned = drv._return_to_entry_from_measured_position(
        hinge._MeasuredItemAnchor(b"entry", 19), frame=b"target", terminal_shift_px=-4243)

    assert returned == hinge._MeasuredItemAnchor(b"return-7", 19)
    assert len(scrolls) == 7
    records = _return_chain_records(drv)
    assert len(records) == 1
    record = records[0]
    assert {key: record[key] for key in (
        "action", "outcome", "reason", "attempts", "initial_terminal_shift_px",
        "terminal_shift_px", "direct_shift_px", "drift_bound_px")} == {
            "action": "still_photo_dwell_return_chain", "outcome": "returned",
            "reason": "direct_remeasure_matched", "attempts": 7,
            "initial_terminal_shift_px": -4243, "terminal_shift_px": 0,
            "direct_shift_px": 0,
            "drift_bound_px": scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H),
        }


def test_return_chain_blank_frame_refuses_and_records_the_specific_leg(monkeypatch, tmp_path):
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="blank-return-chain")
    scrolls = []
    monkeypatch.setattr(drv, "_sample_read_step", lambda *_a: (0.0, 0.0, 0.5))
    monkeypatch.setattr(drv, "_scroll_down_one", lambda frac, x: scrolls.append((frac, x)))
    monkeypatch.setattr(drv, "_screencap", lambda **_kw: None)
    measured = []
    monkeypatch.setattr(drv, "_measured_page_shift",
                        lambda *_a: measured.append(1))

    assert drv._return_to_entry_from_measured_position(
        hinge._MeasuredItemAnchor(b"entry", 0), frame=b"target", terminal_shift_px=-300) is None
    assert len(scrolls) == 1 and measured == []
    record = _return_chain_records(drv)
    assert len(record) == 1
    assert record[0]["outcome"] == "refused"
    assert record[0]["reason"] == "blank_return_frame"
    assert record[0]["attempts"] == 1


def test_return_chain_unmeasurable_leg_refuses_without_direct_fallback(monkeypatch, tmp_path):
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="unmeasurable-return-chain")
    monkeypatch.setattr(drv, "_sample_read_step", lambda *_a: (0.0, 0.0, 0.5))
    monkeypatch.setattr(drv, "_scroll_down_one", lambda *_a: None)
    monkeypatch.setattr(drv, "_screencap", lambda **_kw: b"leg-1")
    calls = []

    def unmeasurable(*_args):
        calls.append(1)
        return None

    monkeypatch.setattr(drv, "_measured_page_shift", unmeasurable)

    assert drv._return_to_entry_from_measured_position(
        hinge._MeasuredItemAnchor(b"entry", 0), frame=b"target", terminal_shift_px=-300) is None
    assert len(calls) == 1, "no entry-frame comparison may replace a missing chained leg"
    record = _return_chain_records(drv)
    assert len(record) == 1
    assert record[0]["reason"] == "return_leg_unmeasurable"
    assert record[0]["terminal_shift_px"] == -300
    assert record[0]["requested_step_px"] == 300
    assert record[0]["kept_before"] == record[0]["before"]
    assert record[0]["kept_after"] == record[0]["after"]
    assert (drv._dbg.dir / record[0]["before"]).exists()
    assert (drv._dbg.dir / record[0]["after"]).exists()


def test_return_chain_exhausted_terminal_drift_refuses_without_direct_fallback(
        monkeypatch, tmp_path):
    """A chain that remains one minimum scroll away is not close enough to call returned."""
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="drifting-return-chain")
    drift_bound = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    monkeypatch.setattr(drv, "_sample_read_step", lambda *_a: (0.0, 0.0, 0.5))
    monkeypatch.setattr(drv, "_scroll_down_one", lambda *_a: None)
    frames = iter([b"leg-1", b"leg-2", b"leg-3", b"leg-4"])
    monkeypatch.setattr(drv, "_screencap", lambda **_kw: next(frames))
    calls = []
    monkeypatch.setattr(drv, "_measured_page_shift",
                        lambda *_a: calls.append(1) or 0)

    assert drv._return_to_entry_from_measured_position(
        hinge._MeasuredItemAnchor(b"entry", 0), frame=b"target",
        terminal_shift_px=-drift_bound) is None
    assert len(calls) == 4, "the direct comparison cannot rescue an out-of-gate residual"
    record = _return_chain_records(drv)
    assert len(record) == 1
    assert record[0]["reason"] == "terminal_drift_exceeds_bound"
    assert record[0]["attempts"] == 4
    assert record[0]["terminal_shift_px"] == -drift_bound


def test_return_chain_direct_remeasure_disagreement_refuses(monkeypatch, tmp_path):
    drv = _drv(WorldAdb())
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="mismatched-return-chain")
    monkeypatch.setattr(drv, "_sample_read_step", lambda *_a: (0.0, 0.0, 0.5))
    monkeypatch.setattr(drv, "_scroll_down_one", lambda *_a: None)
    monkeypatch.setattr(drv, "_screencap", lambda **_kw: b"leg-1")
    # First value closes the local chain.  The next is the independently attempted direct
    # comparison against entry and contradicts it by more than one minimum legal read step.
    shifts = iter([300, 300])
    monkeypatch.setattr(drv, "_measured_page_shift", lambda *_a: next(shifts))

    assert drv._return_to_entry_from_measured_position(
        hinge._MeasuredItemAnchor(b"entry", 0), frame=b"target", terminal_shift_px=-300) is None
    record = _return_chain_records(drv)
    assert len(record) == 1
    assert record[0]["reason"] == "direct_remeasure_disagrees"
    assert record[0]["terminal_shift_px"] == 0
    assert record[0]["direct_shift_px"] == 300


# =====================================================================================
# The capture timing ledger (found+fixed 2026-08-23), wired into a REAL read over this file's
# synthetic world. tests/test_hinge_capture_timing.py covers the ledger's own arithmetic with a
# fully controlled fake clock; these three prove it is actually wired into the real read loop
# and the real fold end to end, using the exact production driver every other test in this file
# already exercises -- not a stand-in.
# =====================================================================================

def test_capture_timing_ledger_records_one_row_per_iteration_a_summary_and_a_fold_row(tmp_path):
    from tools import hinge_capture_timing as hct

    adb = WorldAdb()
    drv = _drv(adb)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="capture-timing")

    profile = drv._capture_current()

    assert profile is not None and profile.items_unavailable == ""
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]
    summary_records = [r for r in records if r["action"] == "capture_timing_summary"]
    fold_records = [r for r in records if r["action"] == "capture_fold_timing"]

    # One iteration row per loop pass: every APPENDED frame took one, plus at most one more for
    # whichever detector (a repeated/static-bottom frame) ended the read without appending its
    # own trigger frame -- see _capture_current's repeated-frame/bottom-detect branches, both of
    # which `break` (and so still reach this ledger's `finally`) rather than appending. One
    # summary row for the whole read, and exactly one fold row -- this profile enumerates
    # cleanly, so _index_captured_items runs exactly once ("if enumerating: ... =
    # self._index_captured_items").
    capture_frames = profile.meta["capture_frames"]
    assert capture_frames <= len(iteration_records) <= capture_frames + 1
    assert len(summary_records) == 1
    assert len(fold_records) == 1

    # THE ARITHMETIC IDENTITY, over real (if tiny, since hinge.time.sleep is patched to a no-op
    # by the autouse _no_sleep fixture) wall-clock measurements: every named bucket a record
    # reports plus its own unattributed_s must reconstruct that record's own total exactly, per
    # _capture_current's and _index_captured_items' TIMING LEDGER paragraphs.
    iter_attr = hct.summarize_iteration_timing(iteration_records)
    for record in iteration_records:
        named = sum(v for k, v in record.items()
                    if k not in hct._ITERATION_META_KEYS and isinstance(v, (int, float)))
        assert named == pytest.approx(record["iter_wall_s"], abs=1e-4)
    summary = summary_records[0]
    assert summary["iterations"] == len(iteration_records)
    assert summary["iter_wall_s_total"] == pytest.approx(
        sum(r["iter_wall_s"] for r in iteration_records), abs=1e-4)
    assert summary["unattributed_s_total"] == pytest.approx(
        iter_attr.unattributed_s_total, abs=1e-4)

    fold_attr = hct.summarize_fold_timing(fold_records)
    assert fold_attr.records == 1
    fold = fold_records[0]
    assert fold["outcome"] == "usable"
    named_fold = sum(v for k, v in fold.items()
                     if k not in hct._FOLD_META_KEYS and isinstance(v, (int, float)))
    assert named_fold == pytest.approx(fold["fold_wall_s"], abs=1e-4)
    # And the reader built on the real production writer's own output agrees with it.
    assert iter_attr.records == len(iteration_records)


def test_capture_timing_ledger_is_silent_with_no_debug_log():
    """`drv._dbg` defaults to None, exactly like every other test above that never sets it --
    this one exists to say explicitly that the timing ledger is part of why that is safe: no
    `time.monotonic()` stamp, no actions.jsonl row, and (the only thing there is to assert) no
    exception anywhere along the way."""
    adb = WorldAdb()
    drv = _drv(adb)
    assert drv._dbg is None

    profile = drv._capture_current()

    assert profile is not None
    assert profile.items_unavailable == ""


def test_gesture_timing_ledger_records_one_row_per_gesture_joined_to_its_iteration(tmp_path):
    """One level down from the iteration ledger above (found+fixed 2026-08-23, the day the
    iteration ledger's own "gesture_s" bucket turned out to be 53.6% of an entire read with
    nothing inside it named): `_capture_current`'s loop now calls
    `_scroll_down_one(frac, x_frac, _iteration=i)`, and this proves that call is actually wired
    to emit one `capture_gesture_timing` row per gesture -- joined back to the iteration it
    belongs to via `frame_index` -- over the SAME real read this file's other capture-timing
    test exercises. tests/test_hinge_capture_timing.py covers `_emit_gesture_timing`'s own
    arithmetic with a fully controlled fake clock; this is the "actually wired into the real
    read loop" half, using the exact production driver every other test in this file already
    exercises, not a stand-in."""
    from tools import hinge_capture_timing as hct

    adb = WorldAdb()
    drv = _drv(adb)
    drv._dbg = HingeDebugLog(str(tmp_path), run_id="gesture-timing")

    profile = drv._capture_current()

    assert profile is not None and profile.items_unavailable == ""
    records = [json.loads(line)
               for line in (drv._dbg.dir / "actions.jsonl").read_text().splitlines()]
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]
    gesture_records = [r for r in records if r["action"] == "capture_gesture_timing"]

    assert gesture_records, "a multi-frame read must issue at least one read-scroll gesture"
    assert all(r["direction"] == "down" for r in gesture_records)
    iteration_frames = {r["frame_index"] for r in iteration_records}
    assert {r["frame_index"] for r in gesture_records} <= iteration_frames, (
        "every gesture this loop issues belongs to a real iteration of the SAME read -- a "
        "frame_index pointing outside that set would mean the join key is wrong, not just "
        "unused")

    # THE ARITHMETIC IDENTITY, same as the iteration ledger's own test above: every named bucket
    # plus this gesture's own unattributed_s must reconstruct gesture_wall_s exactly.
    for record in gesture_records:
        named = sum(v for k, v in record.items()
                    if k not in hct._GESTURE_META_KEYS and isinstance(v, (int, float)))
        assert named == pytest.approx(record["gesture_wall_s"], abs=1e-4)
    gesture_attr = hct.summarize_gesture_timing(gesture_records)
    assert gesture_attr.records == len(gesture_records)

    # WorldAdb (this file's synthetic world) is a duck-typed test double, not the real
    # UhidTouch/Adb transport -- AndroidDriver._touch_supports_timing() correctly recognises
    # that and never passes it `_timing`, so only `_scroll`'s OWN buckets (the geometry guard
    # and the per-gesture foreground recheck, both real hinge.py-level costs) are named here;
    # the transport's own cost is honestly unattributed rather than invented for a transport
    # that was never asked to report it.
    assert "screen_size_s" in gesture_attr.buckets
    assert "zone_check_s" in gesture_attr.buckets
    assert "foreground_reassert_s" in gesture_attr.buckets
    assert not any(key.startswith(("uhid_", "adb_")) for key in gesture_attr.buckets)


def test_gesture_timing_ledger_is_silent_with_no_debug_log():
    """Same guarantee as the iteration ledger's own no-debug-log test, one level down: with
    `drv._dbg` at its default None, `_scroll_down_one` never calls `time.monotonic()` for the
    gesture ledger and never touches `self._emit_gesture_timing` in a way that could raise."""
    adb = WorldAdb()
    drv = _drv(adb)
    assert drv._dbg is None

    profile = drv._capture_current()

    assert profile is not None
    assert profile.items_unavailable == ""


# =====================================================================================
# THE GATE'S OWN PREMISE CAN FAIL (Hinge 10.1.0, 2026-08-28). Doc 5.5's scroll-top check reads
# ONE strip of ONE frame and calls a match to Hinge's filter-chips row "we are at the top". The
# 10.1.0 "expanded" per-profile header pins that row to the SCREEN, so on the live capture
# `item_index_refused_3fa8b8db66fd` the gate returned `confirmed_top` at distance 0.000 on six
# frames at page offsets 6180..8485 -- thousands of px deep. `scroll_top.band_pinned_evidence`
# is the two-frames-at-different-offsets test that detects it; these tests are about the driver
# SUPPLYING that evidence, because until it does the gate answers yes unconditionally on a live
# run no matter what the module can prove.
#
# `WorldAdb(at_top=True)` is that app state, reconstructed: the shipped chips constant painted
# into the identity band at EVERY scroll offset. Every negative control here is the ordinary
# `WorldAdb()`, whose band becomes a person's sticky header the moment the card scrolls -- the
# state this gate reads correctly and must go on reading correctly.
# =====================================================================================

def _pinned_driver(**cfg):
    """A driver that has READ one profile on a chips-row-pinned build, so its evidence is latched.

    The latch comes from a real `_capture_current()` over the pinned world rather than from a
    hand-set attribute: the whole question this section exists to answer is whether the driver
    can OBTAIN the evidence during an ordinary read, and assigning it directly would assume the
    answer.
    """
    adb = WorldAdb(at_top=True)
    drv = _drv(adb, **cfg)
    drv._capture_current()
    return drv, adb


def test_an_enumeration_read_latches_this_builds_pinned_chips_row_from_its_own_frames():
    """The read already holds both halves the proof needs, and neither is measured twice.

    `band_pinned_evidence` requires frames at two or more DIFFERENT, MEASURED page offsets -- a
    second screencap at the same position proves nothing, because a pinned strip and page content
    that has not scrolled yet predict identical pixels. The enumeration read is the one place in
    this driver that already has both: the frames it kept, and the offsets `build_item_index`
    chained through `frameshift.estimate_shift` for them. So the driver reads what was measured
    instead of re-measuring it -- no extra screencap, no extra gesture, and no second opinion on
    a question this repo has answered once.
    """
    adb = WorldAdb(at_top=True)
    drv = _drv(adb)

    # BEFORE the read, on the very screen the incident was measured on: the gate is not merely
    # weak, it is affirmatively wrong -- an unqualified "" (confirmed) at the BOTTOM of a card.
    adb.scroll = _MAX_SCROLL
    assert drv._pinned_band_evidence() is None
    assert drv._confirm_enumeration_top() == ""
    adb.scroll = 0

    drv._capture_current()

    evidence = drv._pinned_band_evidence()
    assert evidence is not None and evidence.pinned
    # The four conditions the helper actually imposed, asserted as the evidence rather than
    # taken on trust: more than one distinct offset, a span clearing the band's own height (so
    # the page rows those sightings would have had to show are disjoint), and one single band.
    assert len(evidence.offsets) >= 2
    assert evidence.offset_span >= evidence.band_height_px
    assert evidence.distinct_bands == 1
    assert evidence.max_chips_distance <= scroll_top._CONFIRM_MAX_DIST


def test_the_latched_evidence_turns_the_gates_affirmative_yes_into_check_unavailable():
    """The point of the wiring: the same frame, the same module, a different answer."""
    drv, adb = _pinned_driver()
    adb.scroll = _MAX_SCROLL
    frame = adb.screencap()

    # The RAW gate still says `confirmed_top` on this exact frame -- so the downgrade below is
    # the evidence's doing, not a different frame's or a different threshold's.
    assert scroll_top.confirm_scroll_top(frame, identity_band=_IDENTITY_BAND).confirmed
    downgraded = scroll_top.confirm_scroll_top(
        frame, identity_band=_IDENTITY_BAND, pinned_evidence=drv._pinned_band_evidence())

    assert downgraded.state == scroll_top.SCROLL_TOP_UNAVAILABLE
    assert drv._confirm_enumeration_top() != ""


def test_the_next_read_on_a_pinned_build_refuses_enumeration_and_the_stop_line_carries_why():
    """END TO END, which is the whole point: the operator-visible outcome of the wiring.

    Capture one is the read that MEASURES the build (its own gate ran before anything was
    proven, exactly as it must -- the evidence does not exist until the frames do). Capture two
    is every capture after it: the gate now has the proof, refuses to raise the enumeration
    ceiling, and `Profile.items_unavailable` -- which worker.py quotes verbatim into the hub's
    stop line -- carries the recalibration sentence rather than an instruction to scroll.
    """
    drv, adb = _pinned_driver()
    assert drv._pinned_band_evidence().pinned
    adb.scroll = 0                        # the next card, served from its own top

    profile = drv._capture_current()

    assert profile is not None
    assert profile.items == () and profile.item_context == ()
    assert scroll_top.SCROLL_TOP_UNAVAILABLE in profile.items_unavailable
    assert "recalibrated" in profile.items_unavailable
    assert "not confirmed to be at its scroll top" not in profile.items_unavailable
    # The read stays at the ordinary cadence: the expensive enumeration ceiling is not spent on
    # a profile whose numbering could never be trusted.
    assert drv._profile_capture_limit == drv.scroll_captures
    assert profile.photos, "the read itself still produces frames for the ranker"


def test_a_dead_signal_stop_line_names_recalibration_where_a_refused_one_names_a_scroll():
    """GUIDANCE MUST DERIVE FROM ITS PRECONDITION (this repo's standing rule, 2026-08-22).

    "the card is not confirmed to be at its scroll top" is an INSTRUCTION: it sends whoever reads
    the hub's stop line to scroll the phone back up. On a build that pins the chips row that
    instruction is not merely unhelpful, it is unfollowable -- they would scroll to a genuine top
    and be refused again, identically, because the strip carries no scroll information at all.
    So the two states get two sentences, and this test drives both out of ONE driver whose only
    difference between them is which strip the screen is showing.
    """
    drv, adb = _pinned_driver()
    adb.scroll = _MAX_SCROLL

    adb._at_top = True                      # the pinned chips row: the check cannot answer
    unavailable = drv._confirm_enumeration_top()
    adb._at_top = False                     # this person's sticky header: positively not at top
    refuted = drv._confirm_enumeration_top()

    assert scroll_top.SCROLL_TOP_UNAVAILABLE in unavailable
    assert "cannot answer" in unavailable
    assert "recalibrated" in unavailable
    assert "Scrolling the card will not change this answer" in unavailable
    assert "not confirmed to be at its scroll top" not in unavailable

    assert scroll_top.SCROLL_TOP_REFUTED in refuted
    assert "not confirmed to be at its scroll top" in refuted
    assert "recalibrat" not in refuted


def test_with_nothing_proven_the_enumeration_gate_is_word_for_word_what_it_was():
    """The status-quo half, and the one that has to hold on every build but the broken one.

    An ordinary read cannot prove pinning -- the band becomes this person's sticky header the
    moment the card scrolls, which fails `band_pinned_evidence`'s "every sighting reads as the
    chips row" half -- so nothing latches and every gate keeps its exact previous behaviour. The
    expected strings here are built from an UNEVIDENCED `confirm_scroll_top` call, so this
    asserts identity with the module's own answer rather than with a copy of the wording.
    """
    adb = WorldAdb()
    drv = _drv(adb)

    profile = drv._capture_current()

    assert profile.items_unavailable == ""          # the read itself is untouched
    assert drv._pinned_band_evidence() is None
    for scroll in (0, 1300, _MAX_SCROLL):
        adb.scroll = scroll
        raw = scroll_top.confirm_scroll_top(adb.screencap(), identity_band=_IDENTITY_BAND)
        expected = "" if raw.confirmed else (
            f"the card is not confirmed to be at its scroll top ({raw.state}): {raw.reason}")
        assert drv._confirm_enumeration_top() == expected


def test_a_collapsed_header_capture_cannot_prove_itself_pinned_and_never_clears_a_proof():
    """Both halves of the latch's lifetime rule, against the same frames.

    A sticky per-profile header is pinned TOO -- measured 0.000 across 143 corpus frames of one
    profile -- so a helper that stopped at "the band held constant" would prove every ordinary
    capture "pinned" and disable the gate everywhere. It does not (first assertion). And because
    a collapsed-header capture can never prove pinning, allowing one to CLEAR a proof would let
    the dead gate come back to life on the very next card and resume affirming a top it has been
    shown it cannot see (second assertion): absence of proof is not proof of absence.
    """
    ordinary_photos = [_frame(s) for s in _FULL_READ_SCROLLS]
    index = item_index.build_item_index(
        ordinary_photos, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)

    fresh = _drv(WorldAdb())
    fresh._latch_scroll_top_pinning(ordinary_photos, index)
    assert fresh._pinned_band_evidence() is None

    proven, _adb = _pinned_driver()
    proven._latch_scroll_top_pinning(ordinary_photos, index)
    assert proven._pinned_band_evidence().pinned


def test_pinning_observer_treats_bad_provenance_or_helper_failures_as_no_measurement(
        monkeypatch):
    """This is a diagnostic for later reads, never a new way to abort this capture.

    Repaired/legacy index doubles can carry malformed source provenance, and an image-library
    failure inside the optional band comparison is similarly not evidence that the current
    index is wrong.  Both must leave the signal unproven rather than escaping the observer.
    """
    photos = [_frame(scroll) for scroll in _FULL_READ_SCROLLS[:2]]
    drv = _drv(WorldAdb())
    malformed = SimpleNamespace(offsets=(0, 300), source_frame_indices=("0", 1))

    drv._latch_scroll_top_pinning(photos, malformed)
    assert drv._pinned_band_evidence() is None

    class BrokenOffsets:
        source_frame_indices = (0, 1)

        @property
        def offsets(self):
            raise RuntimeError("repaired index offsets are unavailable")

    drv._latch_scroll_top_pinning(photos, BrokenOffsets())
    assert drv._pinned_band_evidence() is None

    well_formed = SimpleNamespace(offsets=(0, 300), source_frame_indices=(0, 1))
    monkeypatch.setattr(
        hinge, "band_pinned_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decoder unavailable")))

    drv._latch_scroll_top_pinning(photos, well_formed)
    assert drv._pinned_band_evidence() is None


def test_a_package_update_drops_a_pinning_finding_measured_on_the_previous_build():
    """The finding is about an app BUILD, so it must not outlive the build it was taken on.

    Bound to the same version/geometry pair `_refresh_targeting_calibration_binding` already
    latches from the device (read here, never re-asked of the phone) plus the identity band it
    was measured through. A mid-session update is a new subject, and this one was never measured.
    """
    class VersionAdb(WorldAdb):
        version = "9.134.0"

        def shell(self, command="", **kwargs):
            if command.startswith("dumpsys package "):
                return f"versionName={self.version}\n"
            return super().shell(command, **kwargs)

    adb = VersionAdb(at_top=True)
    drv = _drv(adb)
    drv._capture_current()
    assert drv._pinned_band_evidence().pinned

    adb.version = "10.2.0"
    drv._refresh_targeting_calibration_binding()

    assert drv._pinned_band_evidence() is None
    assert drv._scroll_top_pinned is None            # dropped, not merely hidden from readers


def test_the_bounded_rewind_stops_reporting_a_settled_top_it_cannot_see():
    """Site 2 of the wiring, and the one whose old answer was the most expensive.

    `WorldAdb.swipe` is a no-op, so nothing this loop does can move the card. On a pinned build
    the rewind nevertheless used to break out on its first look with `settled=True`, hand back
    "the card is at its filter-chip top", and re-license capture entry. With the evidence it
    cannot confirm, so it spends its bounded budget, returns False, and its recovery-spent
    diagnostic records `check_unavailable` -- which is what tells an operator the signal is dead
    rather than that the screen was stuck.
    """
    control = _drv(WorldAdb(at_top=True))
    assert control._scroll_to_top_unlocked() is True
    assert control._last_scroll_top_diagnostic["scroll_top_state"] == scroll_top.SCROLL_TOP_CONFIRMED

    drv, _adb = _pinned_driver()
    assert drv._scroll_to_top_unlocked() is False
    assert (drv._last_scroll_top_diagnostic["scroll_top_state"]
            == scroll_top.SCROLL_TOP_UNAVAILABLE)
    assert drv._capture_entry_recovery_spent
    assert scroll_top.SCROLL_TOP_UNAVAILABLE in drv._last_scroll_top_detector_summary()


def test_the_capture_entry_evidence_and_the_recovery_latch_stop_affirming_a_pinned_row():
    """Site 3: the two `.confirmed`-only readers. Both already fail SAFE under the fourth state
    (it is not `.confirmed`), but both were still vacuous until the evidence reached them -- one
    licenses a rewind gesture on an unproven surface, the other resets `_capture_scrolls` and
    re-licenses autonomous recovery on an unproven top."""
    chips = _frame(_MAX_SCROLL, at_top=True)

    control = _drv(WorldAdb(at_top=True))
    control._capture_scrolls = 4
    assert control._capture_entry_profile_evidence(chips) is not None
    assert control._recovery_spent_top_confirmed(chips) is True
    assert control._capture_scrolls == 0

    drv, _adb = _pinned_driver()
    drv._capture_scrolls = 4
    drv._current_sigs = []                       # no captured-profile corroboration either
    assert drv._capture_entry_profile_evidence(chips) is None
    assert drv._recovery_spent_top_confirmed(chips) is False
    assert drv._capture_scrolls == 4             # the counter was NOT reset by a dead signal


def test_a_pinned_row_no_longer_upgrades_a_new_identity_verdict_into_a_proven_top():
    """Site 2b: the canonical detector that licenses Layer 1b's CARD-HEADER OCR.

    ``top`` is not a free verdict -- it is what lets `_identity_of` OCR the card header, a crop
    whose geometry is measured FROM the scroll top. At a mid-profile offset the same rect holds
    photo texture, and turning photo texture into a name is the 2026-08-16 Julia failure. So on
    a build proven unable to tell top from mid-profile, the upgrade is withheld and the pixel
    verdict stands, exactly as it already does for an unreadable or refuted frame.

    Both drivers get the same two capture-local signatures (each far from the chips row, so the
    pixel verdict is ``new`` and the canonical detector is the only thing that could change it)
    and the same frame; the ONLY difference is the latch.
    """
    chips = _frame(1300, at_top=True)
    control = _drv(WorldAdb(at_top=True))
    drv, _adb = _pinned_driver()
    for d in (control, drv):
        d._identity_sig = hinge._band(_frame(600), _IDENTITY_BAND)
        d._identity_top_sig = hinge._band(
            _frame(600, header=_OTHER_HEADER_VALUE), _IDENTITY_BAND)
        d._identity_name = None                  # isolate the pixel + canonical layers from OCR

    assert control._identity_of(chips)[0] == "top"
    assert drv._identity_of(chips)[0] == "new"


def test_the_pinning_measurement_is_stamped_on_the_folds_own_timing_ledger():
    """It reads every kept frame's identity band, so it is a NAMED cost rather than a hidden one
    -- the same rule every other bucket in `_index_captured_items` follows, and the reason
    `unattributed_s` stays meaningful instead of quietly absorbing a new per-frame scan."""
    class Debug:
        def __init__(self):
            self.calls = []

        def action(self, name, **fields):
            self.calls.append((name, fields))

    drv = _drv(WorldAdb())
    drv._dbg = Debug()

    drv._capture_current()

    _name, fold = next(c for c in drv._dbg.calls if c[0] == "capture_fold_timing")
    assert "scroll_top_pinning_s" in fold
