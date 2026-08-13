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
import math
import random

import cv2
import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import (
    hinge, item_identity, item_index, scroll_step, scroll_top, segment)
from operation_love.drivers.hinge import HingeDriver

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
        return ""

    def screencap(self):
        header = self._header
        if self._header_after is not None and self.scrolls >= self._header_after[0]:
            header = self._header_after[1]
        at_top = self._at_top if self._at_top is not None else (self.scroll == 0)
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


def _drv(adb, *, auto=True, openers=True, **cfg):
    """A HingeDriver over the fake world, with the two session hooks a Worker installs.

    `auto` stands in for Worker._auto_loop's set_auto_session_policy (policy None keeps the read
    deterministic: no capture-limit jitter, production dwell/lane fallbacks). It no longer
    decides whether the read ENUMERATES -- doc 5.9's inversion made observe an enumerating mode
    too, so `openers` is what that turns on now, standing in for the set_opener_enabled call BOTH
    loops make. `openers=False` is therefore the way to ask for an ordinary, non-enumerating
    read, in either mode."""
    class C:
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False, **cfg}}
    d = HingeDriver(C())
    d._adb = adb
    d._touch = adb            # touches route through the (fake) transport
    if auto:
        d.set_auto_session_policy(None)
    d.set_opener_enabled(openers)
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
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
    assert all(isinstance(crop, bytes) and crop for crop in profile.items + profile.item_context)


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
    profile so far, so the realised cadence on both calibration profiles is ~233..262px and not
    the 363px ceiling. 44 frames for the worst measured page, plus a 10% margin for a page longer
    than either calibration profile, is 48 — and the absolute worst case (every draw landing on
    the gesture floor) is 47, which the same number covers."""
    worst_page_px, worst_measured_frames = 10027, 44
    realised_step_px = worst_page_px / (worst_measured_frames - 1)
    assert 233 <= realised_step_px <= 262, "the measured cadence the derivation rests on"
    assert hinge._ENUMERATION_CAPTURE_LIMIT == math.floor(worst_measured_frames * 1.1)

    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert hinge._ENUMERATION_CAPTURE_LIMIT >= math.ceil(worst_page_px / floor_px) + 1


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
    """Audit fix, "BUG 3" (2026-08-12). `_ENUMERATION_CAPTURE_LIMIT` (48) raises the CEILING for
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
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False}}
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


def test_a_dependency_that_raises_becomes_the_same_stated_reason():
    """The three leaf modules raise rather than answer when they cannot look at all
    (`SegmentationError` on a frame that will not decode, `ShiftEstimationError` when no page
    space spans the capture, `ItemIndexError`/`ItemCropError` on a capture that cannot be
    indexed, including a missing cv2/numpy). None of them may take down a read whose frames the
    ranker still wants, and none of them may pass silently."""
    drv = _drv(WorldAdb())

    def _boom(*a, **k):
        raise segment.SegmentationError("frame 0 could not be decoded")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hinge, "build_item_index", _boom)
        profile = drv._capture_current()

    assert profile.items == ()
    assert "SegmentationError" in profile.items_unavailable
    assert "frame 0 could not be decoded" in profile.items_unavailable
    assert profile.photos


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
    assert fields["trigger_frame_index"] == fields["captured_frames"] == 2
    assert fields["read_scrolls"] == 2
    assert fields["identity_dist"] >= drv.change_threshold
    assert fields["top_dist"] >= drv.change_threshold
    assert fields["scroll_top_state"] == scroll_top.SCROLL_TOP_REFUTED
    assert fields["scroll_top_distance"] >= drv.change_threshold


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
