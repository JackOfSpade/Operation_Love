"""Counting navigation (operation_love/drivers/item_nav.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures the modules
under this one were measured against are real people's dating profiles and are gitignored
(ops/calibration/, .gitignore:26); only geometry and counts from them appear anywhere in this
repo. So the fixtures build a tall scrollable WORLD from first principles — the same construction
tests/test_item_index.py uses, because the two files are testing two halves of one mechanism —
and a fake driver serves 1080x2400 windows of it, moving by exactly what the measured transport
model says the requested `frac` delivers.

That makes this a closed loop with no device in it and no approximation anywhere: the ground
truth is a table at the top of this file, the shift estimator really measures the pixels, the
segmenter really finds the glyphs, and the assertions can name the item, its page row, its heart
ordinal and the tap coordinate rather than approximating any of them.

The ONE piece of real calibration data in the fixtures is `_SCROLL_TOP_BAND_FINGERPRINT` — the
shipped constant, which is Hinge's own filter-chips chrome and is already in the repo. Painting
it into the top-of-page frames is what lets the real scroll-top gate run against synthetic
frames with its real reference, instead of the test handing it a reference of its own and proving
nothing about the shipped one.

Following the house pattern, every positive is paired with a negative plus a control that proves
WHICH mechanism did the refusing, and the two mechanisms with arithmetic worth isolating
(`_no_heart_was_missed`, `_count_disagrees`) are driven directly as well as end to end.

BOTTOM-UP (2026-08-12). Navigation no longer rewinds to the scroll top and walks forward; it
walks UP from where the profile read left the card, anchored on one measured shift against the
read's own last frame. Two consequences for this file, both deliberate:

  * the fake driver's `_scroll_to_top` RAISES unconditionally, and `_scroll_down_one` raises
    unless a test explicitly opts in (`allow_entry_scroll=True`) AND has not already used its one
    call. Both are kept as tripwires rather than deleted precisely so that reintroducing the
    rewind fails every test in this file immediately and by name, instead of quietly working and
    costing ~100 gestures a profile; the opt-in exists only for blocker 12's scroll-top entry
    recovery (2026-08-22), which spends exactly ONE forward gesture and is tested below;
  * `_no_heart_was_missed` and `_count_disagrees` each carry BOTH sign conventions behind an
    `ascending` flag, and both branches are driven directly here. An inverted comparison in the
    descending half would be invisible end to end (production only walks up), and an inverted
    comparison in the ascending half would simply never fire, which is the failure mode the two
    of them exist to prevent — so neither is tested only through the other.
"""
import dataclasses
import math
import random
from itertools import pairwise

import cv2
import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import hinge, item_index, item_nav, scroll_step, scroll_top, segment

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 17
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
_WORLD_H = 6000

# THE PAGE, in world rows. Four likeable cards and, between the second and third, the heartless
# 215px vitals block doc 5.10 tracked across 7 consecutive frames. Heights are from doc 5.10's
# card table; every heart-to-heart pitch this produces (813, 1321, 873) clears the ~608px floor
# below which `plan_scroll_step` refuses to enumerate at all.
_PAGE_TOP_GAP = 400
_LAYOUT = (("card", 900), ("card", 760), ("context", 215), ("card", 1000), ("card", 820))


def _layout_rows(layout=_LAYOUT, top_gap=_PAGE_TOP_GAP):
    rows, y = [], top_gap
    for kind, height in layout:
        rows.append((kind, y, y + height))
        y += height + _GUTTER
    return rows


_ROWS = _layout_rows()
_CARDS = [r for r in _ROWS if r[0] == "card"]
_HEART_PAGE_Y = [y1 - _HEART_ABOVE_BOTTOM for _kind, _y0, y1 in _CARDS]
assert _ROWS[-1][2] + 400 < _WORLD_H, "the world must have page background below the last card"

_TEMPLATE = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"


def _page_column(height=_WORLD_H):
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, height).round().astype(np.uint8)


def _build_world(*, layout=_LAYOUT, top_gap=_PAGE_TOP_GAP, hearts=True, extra_heart_on=None):
    """The tall page. `hearts=False` paints the same cards with no like glyph on any of them —
    the synthetic form of "the glyph matcher did not find it this frame". `extra_heart_on` stamps
    a SECOND glyph on that card, which is the two-hearts-in-one-block segmentation failure."""
    rng = np.random.default_rng(_SEED)
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    card_no = 0
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
        card_no += 1
        th, tw = _TEMPLATE.shape
        stamps = []
        if hearts:
            stamps.append(y1 - _HEART_ABOVE_BOTTOM)
        if extra_heart_on == card_no:
            # Deep enough inside the card that it is still comfortably within the analysed band
            # after the first gesture, whatever distance the closed loop drew for it. At y0 + 200
            # the glyph's own top row crossed above `content_band` for the larger draws, so
            # whether the frame contradicted itself depended on the step size — a fixture
            # coincidence, not a property of the code under test.
            stamps.append(y0 + 420)
        for cy in stamps:
            world[cy - th // 2: cy - th // 2 + th,
                  _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = _TEMPLATE
    return world


# Static chrome outside the content band, textured but IDENTICAL on every frame — the real
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
    reference through the real decode. [measured here: 1.609 grey levels, comfortably inside the
    shipped 3.0 confirm bound and 7x under the 9.0 refute bound.] It is app chrome, not profile
    content — the constant's own comment records it as byte-identical across two different
    people, which is why one array serves every profile and why nothing about anyone is in it.
    """
    cells = np.array(scroll_top._SCROLL_TOP_BAND_FINGERPRINT, dtype="uint8").reshape(4, 16)
    return np.asarray(Image.fromarray(cells, mode="L").resize((cols, rows), Image.NEAREST))


# The sticky per-profile header that covers that strip the moment the card is scrolled at all.
# Flat mid-grey: 113 grey levels from the chips row, i.e. unambiguously REFUTED.
_HEADER_VALUE = 128
# A DIFFERENT person's sticky header, and the whole of what makes a "foreign profile" foreign
# here. 98 grey levels from the one above, against the 9.0 the identity gate matches inside —
# the real device measured 0.00 between three frames of one profile's header and 17.95 between a
# header and the chips row, so any two distinct headers are far outside the bound.
_OTHER_HEADER_VALUE = 30

# The scroll positions the reference enumeration read stopped at, and therefore where a
# navigation is ENTERED from: the LAST of them, which is the bottom of the read. Not a detail of
# the fixture and not a free choice — bottom-up navigation anchors on one measured shift against
# the read's own last frame, so "where the read ended" and "what the anchor is measured against"
# are one fact, and the driver double and `_reference_index` must agree on it or the fixture is
# testing something the driver can never be in.
#
# It is also scrolled, with the sticky header showing, which is what the identity gate needs: at
# a scroll top that same strip is Hinge's own chrome, byte-identical across two different people,
# so identity is not readable there at all (item_identity's docstring). Under the rewind that was
# a precondition a caller had to remember; here it is where the read leaves the card anyway.
_READ_SCROLLS = tuple(280 * i for i in range(11))
_ENTRY_SCROLL = _READ_SCROLLS[-1]                          # 2800

_WORLDS: dict[tuple, np.ndarray] = {}
_FRAMES: dict[tuple, bytes] = {}


def _world(**kw):
    key = repr(sorted(kw.items(), key=lambda kv: kv[0]))
    if key not in _WORLDS:
        _WORLDS[key] = _build_world(**kw)
    return _WORLDS[key]


def _frame(scroll: int, *, at_top: bool | None = None, identity_rect=_IDENTITY_BAND,
           header: int = _HEADER_VALUE, **world_kw) -> bytes:
    """The 1080x2400 window of the world at `scroll`, as PNG.

    Content at world row `w` lands on frame row `w - scroll`, so `_frame(s)` then `_frame(s + d)`
    is a forward scroll of exactly `d`. `at_top` defaults to `scroll == 0` and decides which of
    the two strips is painted into `identity_rect`: Hinge's profile-independent filter chips at
    the top, or `header` — this person's sticky header — at any other offset. Those two are the
    only signals the scroll-top gate and the identity gate read.
    """
    top = (scroll == 0) if at_top is None else at_top
    key = (scroll, top, tuple(identity_rect), header, repr(sorted(world_kw.items())))
    if key not in _FRAMES:
        world = _world(**world_kw)
        gray = world[scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        r0, r1, c0, c1 = _identity_rect(identity_rect)
        gray[r0:r1, c0:c1] = (_chips_band(r1 - r0, c1 - c0) if top else header)
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAMES[key] = buf.tobytes()
    return _FRAMES[key]


# =====================================================================================
# The fake driver: exactly the surface item_nav's docstring lists, and nothing else
# =====================================================================================

class RewindAttempted(AssertionError):
    """The double was asked to rewind. Bottom-up navigation must never do that again."""


class FakeDriver:
    """A phone made of arithmetic.

    `_scroll_up_one` moves the world by `scroll_step.step_px_for_frac(frac, _H)` — the module's
    own measured transport model, including its 21px touch slop — so the delta the shift
    estimator then measures off the pixels is the delta the plan asked for, with the sign
    flipped, because an upward gesture moves content DOWN. Nothing here approximates the device:
    what is faked is the phone, not the vision.

    `_scroll_to_top` RAISES unconditionally: it is the rewind, kept as a live tripwire rather than
    deleted, so a regression that reintroduces it fails every test in this file immediately and
    by name instead of quietly working and costing ~100 gestures a profile.

    `_scroll_down_one` raises too, UNLESS a test opts in with `allow_entry_scroll=True` — and even
    then only ONCE. That opt-in exists solely for blocker 12's scroll-top entry recovery
    (2026-08-22, `navigate_to_item`'s own module docstring, "THE FIFTH THING"): the one bounded
    forward read-scroll it may take to leave a card's scroll top. A second call raises exactly as
    the unconditional form always did, so a regression that turns that single rescue step into a
    rewind-by-another-name still fails loudly.

    Deliberately has NO tap method. `navigate_to_item`'s contract ends at the coordinate, so a
    test that accidentally tapped would fail with AttributeError rather than pass quietly.
    """

    def __init__(self, *, content_band=_CONTENT_BAND, identity_band=_IDENTITY_BAND,
                 fixed_step_px=None, hide_hearts_between=None,
                 extra_heart_after=None, extra_heart_card=1,
                 identity_rect=_IDENTITY_BAND, scroll_captures=8,
                 start_scroll=_ENTRY_SCROLL, header=_HEADER_VALUE, world_kw=None,
                 allow_entry_scroll=False, entry_scroll_unmeasurable=False,
                 pinned_evidence=None):
        self.content_band = tuple(content_band)
        self.identity_band = None if identity_band is None else tuple(identity_band)
        self.scroll_captures = scroll_captures
        self._capture_scrolls = 0
        self.scroll = start_scroll
        self.gestures: list[tuple[float, float]] = []
        self.captures = 0
        self._fixed_step_px = fixed_step_px
        self._hide_hearts_between = hide_hearts_between
        self._extra_heart_after = extra_heart_after
        self._extra_heart_card = extra_heart_card
        self._identity_rect = identity_rect
        self._header = header
        self._world_kw = dict(world_kw or {})     # a DIFFERENT page: see `_foreign_driver`
        self._allow_entry_scroll = allow_entry_scroll
        self._entry_scroll_unmeasurable = entry_scroll_unmeasurable
        self.entry_scrolls: list[tuple[float, float]] = []    # blocker 12's ONE rescue gesture,
                                                               # kept apart from `gestures` (the
                                                               # ascending walk's own) on purpose
        self._scramble_next_capture = False
        if pinned_evidence is not None:
            # ONLY defined when a test asks for it. Every other test in this file therefore
            # drives a driver with NO `_pinned_band_evidence` at all, which is both the legacy
            # driver and the `getattr` fallback in one -- so "no evidence available means
            # exactly today's behaviour" is asserted by the whole file rather than by one test.
            self._pinned_band_evidence = lambda: pinned_evidence

    # --- the surface item_nav uses -------------------------------------------------
    def _template(self, role):
        assert role == "like", role
        return _TEMPLATE

    def _screencap(self):
        self.captures += 1
        if self._scramble_next_capture:
            # Blocker 12's "unmeasurable entry step" case: the frame right after the recovery
            # scroll shares nothing with the one before it, exactly as
            # `test_a_screen_that_cannot_be_joined_to_the_index_is_a_hard_stop` uses an unrelated
            # frame to force `estimate_shift` to refuse rather than guess.
            self._scramble_next_capture = False
            unrelated = np.random.default_rng(1000 + self.captures).integers(
                0, 255, size=(_H, _W), dtype=np.uint8)
            ok, buf = cv2.imencode(".png", unrelated)
            assert ok
            return buf.tobytes()
        kw = {}
        lo_hi = self._hide_hearts_between
        if lo_hi is not None and lo_hi[0] <= self.scroll <= lo_hi[1]:
            kw["hearts"] = False
        if self._extra_heart_after is not None and len(self.gestures) >= self._extra_heart_after:
            kw["extra_heart_on"] = self._extra_heart_card
        return _frame(self.scroll, at_top=self.scroll == 0, identity_rect=self._identity_rect,
                      header=self._header, **self._world_kw, **kw)

    def _scroll_to_top(self, should_stop=None):
        raise RewindAttempted(
            "navigation rewound to the scroll top. Doc 5.5's bottom-up design walks UP from "
            "where the read left the card; the rewind cost ~51 gestures to arrive somewhere the "
            "read had already been, and replaced a measured anchor with a replayed one")

    def _scroll_down_one(self, frac=None, x_frac=None):
        if not self._allow_entry_scroll or self.entry_scrolls:
            raise RewindAttempted(
                "navigation scrolled FORWARD. Walking up is the whole design; a forward gesture "
                "here means either the third pass came back, or blocker 12's one bounded "
                "scroll-top entry recovery (2026-08-22) ran more than once")
        assert frac is not None and x_frac is not None
        self.entry_scrolls.append((frac, x_frac))
        if self._entry_scroll_unmeasurable:
            self._scramble_next_capture = True
            return
        moved = (self._fixed_step_px if self._fixed_step_px is not None
                 else scroll_step.step_px_for_frac(frac, _H))
        self.scroll = max(0, min(_WORLD_H - _H, self.scroll + moved))

    def _scroll_up_one(self, frac, x_frac):
        # Positional-or-keyword with NO defaults, exactly as the driver's own method is: the
        # `_scroll_down_one(frac)` trap (re-sampling both from the behaviour policy when either
        # is None, and so silently issuing production's cadence) is unreachable when there is
        # nowhere to put a None.
        assert frac is not None and x_frac is not None
        self.gestures.append((frac, x_frac))
        moved = (self._fixed_step_px if self._fixed_step_px is not None
                 else scroll_step.step_px_for_frac(frac, _H))
        self.scroll = max(0, min(_WORLD_H - _H, self.scroll - moved))

    def _sample_read_step(self, depth, complexity_hint):
        # (dwell, frac, x_frac). The frac is production's read cadence — the one item_nav must
        # discard — and the tests below assert it never reaches a gesture.
        return 0.0, hinge.HINGE_SPEC.read_scroll_frac, 0.5


_POLICY_FRAC = hinge.HINGE_SPEC.read_scroll_frac
_CACHE: dict[str, object] = {}


def _reference_index(*, identity_band=_IDENTITY_BAND, identity_rect=_IDENTITY_BAND,
                     header=_HEADER_VALUE):
    """The index a real enumeration pass would have produced for this world: a clean read from
    the top past the last card, at a cadence the closed loop would itself have planned.

    It carries the profile's IDENTITY as well as its items, fingerprinted from these same frames
    — frame 0 confirms the chips row and frame 1 shows `header` over it, which is the pair of
    observations `capture_profile_identity` requires before it will trust the rect at all."""
    key = (tuple(identity_band or ()), tuple(identity_rect), header)
    if key not in _CACHE:
        assert _READ_SCROLLS[-1] + _H <= _WORLD_H
        _CACHE[key] = item_index.build_item_index(
            [_frame(s, identity_rect=identity_rect, header=header) for s in _READ_SCROLLS],
            content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
            identity_band=identity_band)
    return _CACHE[key]


def _entry_reference(*, identity_rect=_IDENTITY_BAND, header=_HEADER_VALUE):
    """The LAST frame `_reference_index` was built from — what `hinge` keeps as
    `_current_item_anchor` and what the entry anchor is measured against.

    Deliberately built from the DEFAULT world even when the driver serves a different one: that
    IS the stale-payload shape, and it is the case the identity gate exists for."""
    return _frame(_READ_SCROLLS[-1], identity_rect=identity_rect, header=header)


@pytest.fixture(autouse=True)
def _deterministic_draws():
    """`plan_scroll_step` draws its ratio from the module-level `random`. Seeding makes a failure
    reproducible; nothing below depends on a particular draw, only on the window it comes from."""
    random.seed(_SEED)


def _navigate(driver, model_index, index=None, reference=None, **kw):
    return item_nav.navigate_to_item(
        driver, index or _reference_index(), model_index,
        entry_reference=_entry_reference() if reference is None else reference,
        identity_match_max_dist=kw.pop("identity_match_max_dist", 2.0), **kw)


def test_navigation_requires_an_explicit_identity_calibration_bound():
    """This leaf returns a heart a caller can tap, so it must not retain a permissive default."""
    with pytest.raises(TypeError, match="identity_match_max_dist"):
        item_nav.navigate_to_item(
            FakeDriver(), _reference_index(), 1, entry_reference=_entry_reference())


@pytest.mark.parametrize("bound", [math.nan, math.inf, -math.inf, True, 0, -1, 2.565, 3.0])
def test_invalid_identity_calibration_refuses_before_the_first_screencap_or_gesture(bound):
    driver = FakeDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, identity_match_max_dist=bound)
    assert exc.value.code == item_nav.NAV_CALIBRATION_INVALID
    assert driver.captures == 0 and driver.gestures == []


# =====================================================================================
# The world is what the test says it is
# =====================================================================================

def test_the_reference_index_is_the_page_that_was_painted():
    """Not an assertion about item_nav — a guard on the fixture. Everything below reads its
    ground truth off these numbers, so if the world stops matching the layout table the failures
    would look like navigation bugs."""
    index = _reference_index()

    assert index.usable, index.failures
    assert index.at_scroll_top and index.reached_end
    assert [b.kind for b in index.blocks] == [
        item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE, item_index.ITEM_CONTEXT,
        item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE]
    assert index.translation == (1, 2, 3, 4)
    assert [item_nav.index_hearts(index)[k][1] for k in (1, 2, 3, 4)] == _HEART_PAGE_Y


def test_the_scroll_top_frames_confirm_and_the_scrolled_ones_refute():
    """The other fixture guard, and the one that makes every "top" assertion below mean
    something: the SHIPPED fingerprint, through the SHIPPED decode, on these synthetic frames."""
    top = scroll_top.confirm_scroll_top(_frame(0), identity_band=_IDENTITY_BAND)
    assert top.confirmed and top.distance <= scroll_top._CONFIRM_MAX_DIST

    scrolled = scroll_top.confirm_scroll_top(_frame(600), identity_band=_IDENTITY_BAND)
    assert scrolled.refuted and scrolled.distance >= scroll_top._REFUTE_MIN_DIST


# =====================================================================================
# THE HEADLINE: every item, landed on exactly
# =====================================================================================

def test_every_item_is_navigated_to_and_lands_on_its_own_heart():
    """Doc 5.5 end to end, bottom-up. For each of the profile's four items in turn: enter where
    the read left the card, walk UP counting hearts in reverse, and land with that item's heart
    on screen — the RIGHT heart, at the page row the index recorded, on a card of the height the
    index recorded.

    Exact, not approximate: the world is a table, so the tap coordinate translated back into page
    space must equal the heart's painted row to the pixel."""
    index = _reference_index()
    hearts = item_nav.index_hearts(index)

    for model_index in (1, 2, 3, 4):
        driver = FakeDriver()
        target = _navigate(driver, model_index)

        assert target.model_index == model_index
        assert target.heart_ordinal == model_index          # this page's translation is 1:1
        assert target.point[1] + target.page_offset == _HEART_PAGE_Y[model_index - 1]
        assert target.point[0] == hearts[model_index][0]
        assert target.agreement_px == 0

        # The card under it is on screen and whole, which is the precondition for tapping it.
        block = index.block_for(model_index)
        y0, y1 = target.block_frame_rows
        assert _BAND0 <= y0 and y1 <= _BAND1
        assert y1 - y0 == block.height
        assert target.block_page_rows == (block.page_y0, block.page_y1)
        assert target.hearts_counted >= 1
        # Walking up NEVER goes down, and the page offset only shrinks.
        assert all(est.delta_px < 0 for est in target.shifts)
        assert target.page_offset <= target.entry_offset


def test_the_rewind_is_gone_and_the_cost_is_the_walk_up_and_nothing_else():
    """THE POINT OF THE REDESIGN, as arithmetic rather than as prose.

    Under the rewind every navigation paid ~51 undo swipes to reach the scroll top and then
    walked forward again, so the CHEAPEST item on the page (the last one, which the read is
    already showing) was also the most expensive thing the design could do. Bottom-up inverts
    that: the last item costs nothing at all, and every other item costs only the distance
    between it and where the read ended.

    Three properties, all of them checked against the fixture's own geometry rather than against
    a recorded number: the last item takes ZERO gestures, cost rises monotonically the further up
    the page the item is, and the worst item on the page still costs less than the rewind alone
    would have."""
    costs = {}
    for model_index in (1, 2, 3, 4):
        driver = FakeDriver()
        target = _navigate(driver, model_index)
        assert len(driver.gestures) == len(target.steps)
        costs[model_index] = len(target.steps)

    assert costs[4] == 0, "the read already left the card on the last item"
    assert costs[1] > costs[2] > costs[3] >= costs[4]

    # Two bounds on the worst item, both read off the fixture rather than recorded. The walk up
    # cannot exceed the number of minimum-size gestures that spans the ascent; and it costs no
    # more than the READ's own walk down over the same page, which is the whole saving stated
    # exactly — the rewind paid that same walk PLUS an undo of every gesture the read made.
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert costs[1] <= math.ceil(_ENTRY_SCROLL / floor_px)
    assert costs[1] <= len(_READ_SCROLLS)


def test_the_returned_point_really_is_a_like_glyph_in_the_returned_frame():
    """The coordinate is only worth anything if the glyph is under it. Re-run the driver's own
    matcher over the landing frame, with no help from the index, and the point must be one of the
    hearts it finds."""
    target = _navigate(FakeDriver(), 3)
    found = hinge._match_glyph(target.frame, _TEMPLATE, side="right",
                               threshold=hinge._LIKE_MATCH_THRESHOLD, y_band=_CONTENT_BAND)

    assert target.point in found


def test_navigation_never_taps_and_leaves_the_phone_where_it_landed():
    """Doc 5.6: this function's contract ends at the coordinate. The fake driver has no tap
    method at all, so a tap would be an AttributeError rather than a silent extra gesture; what
    is asserted here is the positive half — the only device verbs used are the two sanctioned
    ones, and the phone is left showing the frame the point refers to."""
    driver = FakeDriver()
    target = _navigate(driver, 2)

    assert not hasattr(driver, "_tap")
    assert driver.scroll == target.page_offset            # still exactly where it landed
    assert driver._screencap() == target.frame
    # One capture per frame, and the ENTRY frame is both the identity gate's and the anchor's, so
    # there is no separate gate capture to pay for — that is one fewer ADB round-trip than the
    # rewind spent, before counting the swipes it does not make.
    assert driver.captures == len(target.steps) + 2       # +1 entry, +1 the re-read above


def test_every_gesture_goes_through_the_humanized_path_with_the_planned_distance():
    """The humanized-input rule and `scroll_step`'s load-bearing trap in one test.

    Every reverse gesture is `_scroll_up_one`, which goes through the driver's own `_scroll`
    (forbidden-zone guard, shared column jitter, humanized kinematics) and takes BOTH arguments
    with no defaults — which is what makes `_scroll_down_one`'s "re-sample both from the policy
    if either is None" trap unreachable from here. The policy's own frac is discarded, its LANE
    is kept, and every distance stays inside the driver's sanctioned read-scroll window and under
    doc 5.10.1's aliasing ratio."""
    driver = FakeDriver()
    target = _navigate(driver, 1)

    assert driver.gestures and len(driver.gestures) == len(target.steps)
    for (frac, x_frac), step in zip(driver.gestures, target.steps, strict=True):
        assert (frac, x_frac) == (step.frac, step.x_frac)
        assert frac != _POLICY_FRAC                       # the policy's frac was DISCARDED
        assert hinge._READ_SCROLL_FRAC_MIN <= frac <= hinge._READ_SCROLL_FRAC_MAX
        assert x_frac == 0.5                              # ...while its LANE was kept
        assert step.ratio is None or step.ratio <= 0.36


def test_a_navigation_that_rewinds_fails_loudly_rather_than_working_expensively():
    """The tripwire, asserted as a tripwire. The double's `_scroll_to_top` and `_scroll_down_one`
    raise, so this test is what says WHY they raise rather than leaving it to a comment: a
    reintroduced rewind is not a wrong answer, it is a right answer that costs ~100 gestures a
    profile and replaces a measured anchor with a replayed one, which is exactly the kind of
    regression that survives a test suite."""
    driver = FakeDriver()
    with pytest.raises(RewindAttempted, match="bottom-up"):
        driver._scroll_to_top()
    with pytest.raises(RewindAttempted, match="Walking up"):
        driver._scroll_down_one(0.16, 0.5)

    # ...and a real navigation over the same double completes, so what is being pinned is that
    # nothing on the path calls them rather than that the double is broken.
    assert _navigate(FakeDriver(), 1).heart_ordinal == 1


def test_the_crosscheck_bound_sums_this_pass_and_the_indexs_own_extent_slack(monkeypatch):
    """`_CROSSCHECK_TOLERANCE_PX` is the SUM of two chains' slack, and only one chain is ours.

    The other belongs to the index in hand. `item_index`'s frame-omission recovery deliberately
    folds its blocks at `_RECOVERY_EXTENT_TOLERANCE_PX` (9) rather than the default 8, so a fixed
    `2 * _EXTENT_TOLERANCE_PX` bound is one pixel short for exactly the indexes that were hardest
    to build: worst case 9 + 8 = 17 against a 16px bound, which hard-stops a navigation whose two
    passes actually agreed. Read the slack off the index instead of assuming it.
    """
    seen: list[int] = []
    real = item_nav._count_disagrees

    def capture(*args, **kwargs):
        seen.append(kwargs["tolerance"])
        return real(*args, **kwargs)

    monkeypatch.setattr(item_nav, "_count_disagrees", capture)

    assert _navigate(FakeDriver(), 1).heart_ordinal == 1
    assert seen and set(seen) == {2 * item_index._EXTENT_TOLERANCE_PX}

    seen.clear()
    recovered = dataclasses.replace(
        _reference_index(), extent_tolerance_px=item_index._RECOVERY_EXTENT_TOLERANCE_PX)
    assert _navigate(FakeDriver(), 1, index=recovered).heart_ordinal == 1
    assert seen and set(seen) == {
        item_index._RECOVERY_EXTENT_TOLERANCE_PX + item_index._EXTENT_TOLERANCE_PX}

    # An explicit argument still wins, for a calibration pass varying one bound at a time.
    seen.clear()
    _navigate(FakeDriver(), 1, index=recovered, crosscheck_tolerance_px=21)
    assert seen and set(seen) == {21}


def test_the_step_is_sized_against_this_profiles_own_measured_spacing():
    """Doc 5.10.1's rule, at the joint where navigation spends it: the first gesture is already
    sized against the smallest heart pitch the INDEX measured over the whole page, not discovered
    one card at a time, because the enumeration pass has already met every card."""
    index = _reference_index()
    smallest = item_nav._min_heart_pitch(item_nav.index_hearts(index))
    assert smallest == min(b - a for a, b in pairwise(_HEART_PAGE_Y))

    target = _navigate(FakeDriver(), 1)
    for step in target.steps:
        assert step.sized_against_px <= smallest
        assert step.step_px <= int(0.36 * smallest)


def test_the_gate_reads_the_drivers_band_and_not_the_specs():
    """Same rule `segment_frame` states for `content_band`: an operator's config.yaml override
    must be honoured, or it is silently ignored. Paint the strips at a MOVED rect and declare
    that rect on the driver, and the whole navigation runs; leave the driver on the spec's rect
    while the paint is elsewhere, and the identity gate refuses — a rect that never shows the
    chips row yields no fingerprint at all, which is `capture_profile_identity`'s deliberate
    refusal to trust a rect that might be pointing at static chrome."""
    moved = (0.10, 0.020, 0.80, 0.066)                    # same size, 67px further UP
    assert moved != _IDENTITY_BAND                        # ...and still clear of content_band

    honoured = FakeDriver(identity_band=moved, identity_rect=moved)
    index = _reference_index(identity_band=moved, identity_rect=moved)
    assert index.identity.known and index.identity.band == moved
    target = _navigate(honoured, 1, index=index,
                       reference=_entry_reference(identity_rect=moved))
    assert target.identity.matched

    ignored = FakeDriver(identity_band=_IDENTITY_BAND, identity_rect=moved)
    blind = _reference_index(identity_band=_IDENTITY_BAND, identity_rect=moved)
    assert not blind.identity.known                       # nothing ever showed the chips row
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(ignored, 1, index=blind, reference=_entry_reference(identity_rect=moved))
    assert exc.value.code == item_nav.NAV_IDENTITY_UNCONFIRMED
    assert ignored.gestures == []


# =====================================================================================
# Refusal: the index, before a finger moves
# =====================================================================================

def test_an_unusable_index_is_refused_before_the_phone_is_touched():
    """Doc 5.3: "treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate". The control is that NOTHING happened — no swipe, no capture, no gesture."""
    broken = item_index.build_item_index(
        [_frame(0), _frame(280), _frame(280 + 1400)], content_band=_CONTENT_BAND,
        like_template=_TEMPLATE, like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=_IDENTITY_BAND)
    assert not broken.usable

    driver = FakeDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, index=broken)

    assert exc.value.code == item_nav.NAV_INDEX_UNUSABLE
    assert (driver.captures, driver.gestures) == (0, [])


def test_a_relative_index_is_refused_before_the_phone_is_touched():
    """BLOCKER 3. With `at_scroll_top=False` the heart ordinals are relative to whatever was in
    view, so counting to one of them from a genuine top lands a fixed number of cards away from
    the item that was chosen. The index is otherwise perfectly usable, which is the whole
    danger — nothing else on it says the numbering is not the profile's own."""
    relative = item_index.build_item_index(
        [_frame(280 * i) for i in range(2, 10)], content_band=_CONTENT_BAND,
        like_template=_TEMPLATE, like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=False,
        identity_band=_IDENTITY_BAND)
    assert relative.usable and relative.selectable and not relative.at_scroll_top

    driver = FakeDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, index=relative)

    assert exc.value.code == item_nav.NAV_INDEX_RELATIVE
    assert (driver.captures, driver.gestures) == (0, [])

    # The enforcement is in two places, and this is the other one: the accessor a navigator
    # calls refuses to answer at all, so the refusal does not depend on remembering to check.
    with pytest.raises(item_index.ItemIndexError, match="at_scroll_top=False"):
        relative.heart_ordinal_for(1)


def test_an_item_the_model_was_never_offered_is_refused_and_never_substituted():
    """Doc 5.6's standing owner rule: never substitute a different item. An out-of-range number
    is refused rather than clamped to the nearest real one."""
    driver = FakeDriver()
    for bad in (0, -1, len(_reference_index().selectable) + 1):
        with pytest.raises(item_nav.ItemNavigationError) as exc:
            _navigate(driver, bad)
        assert exc.value.code == item_nav.NAV_ORDINAL_OUT_OF_RANGE
    assert driver.gestures == []


# =====================================================================================
# Refusal: the entry anchor (what replaced the top gate)
# =====================================================================================

def test_the_entry_anchor_is_measured_against_the_reads_own_last_frame_and_is_zero():
    """Bottom-up's zero point. `hinge._capture_current` leaves the screen showing its own last
    KEPT frame in both of its terminating paths, so the shift between the reference and the entry
    frame is exactly 0px — not approximately, and not a tolerance being satisfied. Everything
    below it is a page row in the INDEX's own coordinates, which is why there is no second origin
    to reconcile and no residual for the cross-check to absorb."""
    driver = FakeDriver()
    target = _navigate(driver, 3)

    assert target.anchor.ok and target.anchor.delta_px == 0
    assert target.entry_offset == _ENTRY_SCROLL
    assert target.offsets[0] == target.entry_offset
    assert target.offsets[-1] == target.page_offset
    # And it really is the index's space: the landing block's page rows are the index's own.
    block = _reference_index().block_for(3)
    assert target.block_page_rows == (block.page_y0, block.page_y1)


def test_a_screen_that_cannot_be_joined_to_the_index_is_a_hard_stop():
    """The refusal that replaced "the top was never confirmed". Handed a reference frame from a
    page this screen has nothing in common with, `estimate_shift` returns no delta at all, and
    there is no page row to count from. Assuming a distance is what fabricated doc 5.10's phantom
    item, so nothing is assumed.

    This is also the shape a foreign profile takes on REAL frames, which the synthetic
    cross-profile fixtures below cannot reproduce: two different people's cards do not correlate,
    so a stale payload fails here as well as at the identity gate."""
    unrelated = np.random.default_rng(11).integers(0, 255, size=(_H, _W), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", unrelated)
    assert ok

    driver = FakeDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, reference=buf.tobytes())

    assert exc.value.code == item_nav.NAV_ANCHOR_UNMEASURED
    assert exc.value.anchor is not None and exc.value.anchor.delta_px is None
    assert driver.gestures == []                  # refused before it moved the phone


def test_a_card_that_moved_between_the_read_and_the_like_is_a_hard_stop():
    """The drift bound, and it is a physical claim rather than noise slack: the read leaves the
    card exactly where its last kept frame was, so a drift at or beyond the SMALLEST gesture this
    driver can make cannot have come from rendering or a settle — something scrolled the profile
    in between.

    The driver here starts one full gesture-floor above where the read ended, which is what a
    finger on the phone during the ~90s opener call looks like."""
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    moved = FakeDriver(start_scroll=_ENTRY_SCROLL - floor_px)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(moved, 3)

    assert exc.value.code == item_nav.NAV_ANCHOR_UNMEASURED
    assert exc.value.anchor.delta_px == -floor_px
    assert "moved" in str(exc.value)
    assert moved.gestures == []

    # The control, one pixel inside the bound: a drift smaller than any gesture we could have
    # made is honoured as a measurement and the navigation completes from the shifted origin.
    nudged = FakeDriver(start_scroll=_ENTRY_SCROLL - (floor_px - 1))
    target = _navigate(nudged, 3)
    assert target.anchor.delta_px == -(floor_px - 1)
    assert target.entry_offset == _ENTRY_SCROLL - (floor_px - 1)
    assert target.heart_ordinal == 3


# =====================================================================================
# Refusal: the scroll
# =====================================================================================

def test_a_scroll_that_moves_nothing_is_a_stalled_loop_and_stops():
    """`step_overshoot`'s other half, through the sign flip. Ascending, the magnitude handed to
    it is `-delta`, so "the profile did not move" now also covers "it moved the WRONG WAY" — both
    are `<= 0` and both are the same stop, because a loop that is not climbing would re-count the
    same frame forever."""
    driver = FakeDriver(fixed_step_px=0)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_SCROLL_STALLED
    assert len(driver.gestures) == 1              # stopped at the first one, not after a sweep
    assert exc.value.frame is not None            # ...and kept the frame for the operator
    recovery = exc.value.recovery
    assert recovery is not None
    assert recovery.achieved_step_px == 0
    assert recovery.page_shift_px == 0
    assert recovery.planned_step_px > 0 and recovery.bound_px > 0


def test_a_gesture_that_goes_the_WRONG_WAY_is_the_same_stall_and_not_progress():
    """The sign flip's own failure mode, and the one an inverted comparison would let through:
    a "reverse" gesture that actually moved the content FORWARD. Under the descending convention
    a positive delta is progress; under this one it is the opposite of progress, and the two must
    not be collapsed. A negative `fixed_step_px` makes the double scroll the wrong way while
    every other part of the loop behaves normally."""
    driver = FakeDriver(fixed_step_px=-260)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_SCROLL_STALLED
    assert "did not move" in str(exc.value)
    assert "+260px" in str(exc.value)             # the measured delta, sign and all
    assert len(driver.gestures) == 1


def test_a_scroll_past_the_plans_own_aliasing_bound_stops():
    """Doc 5.10.1's failure, caught at the gesture that caused it rather than at the wrong item
    it would have produced. Aliasing does not care which way the finger went, so the ratio rule
    binds a reverse gesture exactly as it binds a forward one. 500px is inside the shift
    estimator's trust window — so this is a MEASURED over-large step and not a broken
    correspondence — and past the ~292px bound the plan was built to respect."""
    driver = FakeDriver(fixed_step_px=500)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_SCROLL_OVERSHOT
    assert "aliasing bound" in str(exc.value)
    assert len(driver.gestures) == 1
    recovery = exc.value.recovery
    assert recovery is not None
    assert recovery.frame is exc.value.frame
    assert recovery.achieved_step_px == 500
    assert recovery.page_shift_px == -500
    assert recovery.achieved_step_px > recovery.bound_px
    assert "aliasing bound" in recovery.violation


def test_frames_that_cannot_be_put_in_one_coordinate_space_stop_the_count():
    """The phantom-item guard, inherited from `frameshift`: a shift beyond the trust window
    returns no delta at all, so there is no page space to count in. 1400px clears the 900px
    window — it is refused as unmeasurable rather than reported as an over-large step, which is
    a different stop and must not be collapsed into it."""
    driver = FakeDriver(fixed_step_px=1400)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_CHAIN_BROKEN
    assert len(driver.gestures) == 1


# =====================================================================================
# Refusal: the count
# =====================================================================================

def test_a_heart_that_was_on_screen_and_not_counted_stops_the_run():
    """The count's silent failure mode, made loud, in the ascending direction. An UPWARD scroll
    admits new content at the band's TOP edge, so a heart first seen in frame i must have been
    ABOVE frame i-1's band. Here the glyph is invisible across the window in which the second
    heart enters from above, so it is first seen several frames later — inside a band that
    already covered it — and the run stops instead of shifting every ordinal counted past it.

    The control is that the identical driver WITHOUT the blanking navigates to the same item."""
    entered_at = _HEART_PAGE_Y[1] - _BAND0                 # the scroll at which heart 2 enters
    blinded = FakeDriver(hide_hearts_between=(entered_at - 700, entered_at))
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(blinded, 1)

    assert exc.value.code == item_nav.NAV_HEART_MISSED
    assert "was on screen and not counted" in str(exc.value)

    assert _navigate(FakeDriver(), 1).heart_ordinal == 1


def test_the_missed_heart_rule_directly_in_BOTH_directions():
    """The same rule as arithmetic, with no dependence on what a correlation does, and BOTH sign
    conventions side by side — which is the point, because an inverted comparison in either half
    would simply never fire and would therefore pass an end-to-end test.

    Descending, new content arrives at the band's BOTTOM: a heart first seen in frame i must have
    been BELOW frame i-1's band. Ascending, it arrives at the TOP: it must have been ABOVE it.
    Frame 0 is exempt in both, because it has no predecessor."""
    def obs(frame_index, page_y0, page_y1, hearts):
        return item_index.BlockObservation(
            frame_index=frame_index, page_y0=page_y0, page_y1=page_y1, frame_y0=page_y0,
            frame_y1=page_y1, kind=segment.BLOCK_SELECTABLE, complete=True,
            top_observed=True, bottom_observed=True,
            hearts=tuple((_HEART_CX, y) for y in hearts))

    band, tol = (_BAND0, _BAND1), item_index._EXTENT_TOLERANCE_PX

    # DESCENDING. Frame 1's window is 300px FURTHER DOWN the page.
    down = [0, 300]
    clean = [obs(0, 900, 1900, [1810]), obs(1, 2200, 3000, [2910])]
    assert item_nav._no_heart_was_missed(
        [(_HEART_CX, 1810), (_HEART_CX, 2910)], clean, down, band, tol,
        ascending=False) is None
    missed = [obs(0, 900, 1900, [1810]), obs(1, 1000, 2000, [1950])]
    reason = item_nav._no_heart_was_missed(
        [(_HEART_CX, 1810), (_HEART_CX, 1950)], missed, down, band, tol, ascending=False)
    assert reason is not None and "not counted" in reason

    # ASCENDING. Frame 1's window is 300px FURTHER UP, so `offsets` decreases and the legitimate
    # new heart is the one ABOVE frame 0's band top (page row 0 + 300).
    up = [1000, 700]
    clean_up = [obs(0, 1400, 2400, [2310]), obs(1, 950, 1300, [1210])]
    assert item_nav._no_heart_was_missed(
        [(_HEART_CX, 1210), (_HEART_CX, 2310)], clean_up, up, band, tol,
        ascending=True) is None
    # The failure: a heart first seen in frame 1 at a row frame 0's band already covered
    # (page row 1500, which is inside frame 0's 1300..3100 window).
    missed_up = [obs(0, 1400, 2400, [2310]), obs(1, 1400, 1600, [1500])]
    reason_up = item_nav._no_heart_was_missed(
        [(_HEART_CX, 1500), (_HEART_CX, 2310)], missed_up, up, band, tol, ascending=True)
    assert reason_up is not None and "not counted" in reason_up

    # AND THE CROSS-CHECK THAT CATCHES AN INVERTED COMPARISON: each direction's LEGITIMATE
    # arrival is the other direction's failure, because a heart that arrived from below cannot
    # also have arrived from above. A branch that read the wrong edge would therefore turn one of
    # these two silent, which no end-to-end test could show.
    assert item_nav._no_heart_was_missed(
        [(_HEART_CX, 1810), (_HEART_CX, 2910)], clean, down, band, tol,
        ascending=True) is not None
    assert item_nav._no_heart_was_missed(
        [(_HEART_CX, 1210), (_HEART_CX, 2310)], clean_up, up, band, tol,
        ascending=False) is not None


def test_a_count_that_disagrees_with_the_index_stops_rather_than_choosing():
    """Two measurements of one page. The world navigated has its second card 130px SHORTER and
    its last card 130px TALLER than the world the index was built from — chosen so that the
    bottom-most heart lands exactly where the index says it does, and the anchor is therefore
    perfect. Everything above it is 130px out, which is what the RELATIVE half of the cross-check
    is for. Neither measurement can be preferred, so the run stops rather than tapping "the
    nearest heart"."""
    skewed = (("card", 900), ("card", 630), ("context", 215), ("card", 1000), ("card", 950))
    driver = FakeDriver(world_kw={"layout": skewed})

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, reference=_frame(_ENTRY_SCROLL, layout=skewed))

    assert exc.value.code == item_nav.NAV_COUNT_DISAGREES
    assert "from the anchor heart 4" in str(exc.value)
    assert driver.gestures == []                  # caught on the very first frame


def test_the_landing_card_itself_must_be_the_card_the_index_recorded():
    """The other half of the cross-check, and a different question from "is this the k-th
    heart": is the k-th heart on a card laid out like the one the index calls item N. What it
    catches is the right ordinal on a differently-SIZED card, which is a miscount; what it does
    NOT catch is a foreign profile, whose cards are stereotyped — see
    `test_the_landing_check_cannot_tell_a_foreign_profiles_card_from_this_ones`.

    All three comparisons are origin-free, so none of them depends on the two passes sharing a
    page origin, which they do not."""
    index = _reference_index()
    block = index.block_for(1)
    heart = item_nav.index_hearts(index)[1]

    class Fake:
        def __init__(self, y0, y1, hx, hy):
            self.y0, self.y1, self.hx, self.hy = y0, y1, hx, hy
        height = property(lambda self: self.y1 - self.y0)

    def check(y0, y1, hx, hy):
        return item_nav._confirm_landing(
            heart=(hx, hy), block=Fake(y0, y1, hx, hy), index_block=block, index_heart=heart,
            extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX,
            crosscheck_tolerance_px=item_nav._CROSSCHECK_TOLERANCE_PX)

    # The same card, seen at a different page origin: 900 rows further down, identical geometry.
    inset = heart[1] - block.page_y0
    assert check(900, 900 + block.height, heart[0], 900 + inset) is None

    tall, moved = 900 + block.height + 20, 900 + inset + 20
    assert "px tall in this frame" in check(900, tall, heart[0], 900 + inset)
    assert "below its card's top edge" in check(900, 900 + block.height, heart[0], moved)
    assert "no horizontal component" in check(900, 900 + block.height, heart[0] + 40, 900 + inset)


_HEARTS = {1: (_HEART_CX, 1000), 2: (_HEART_CX, 1900), 3: (_HEART_CX, 3000)}


def _disagrees(clusters, *, ascending, hearts=None, heart_count=3, tolerance=None):
    return item_nav._count_disagrees(
        clusters, _HEARTS if hearts is None else hearts, heart_count=heart_count,
        tolerance=item_nav._CROSSCHECK_TOLERANCE_PX if tolerance is None else tolerance,
        ascending=ascending)


def test_the_crosscheck_compares_distances_and_not_page_rows_in_BOTH_directions():
    """Why the comparison survives a page-origin difference at all. Shifting every counted heart
    by a constant must change nothing, while moving ONE of them relative to the others must be
    caught — and that has to hold whichever end the anchor is at, because the ordinal arithmetic
    is shared between the two branches and only the anchor differs.

    Descending, the constant is the top gate's residual (two confirmed tops may sit (0, 202]
    apart). Ascending, this pass shares the index's page space through one measured shift, so the
    tolerated constant is much smaller — but the arithmetic being exercised is identical, which
    is the property worth pinning."""
    tol = item_nav._CROSSCHECK_TOLERANCE_PX
    for ascending, offset in ((False, 201), (True, tol)):
        shifted = [(_HEART_CX, y + offset) for y in (1000, 1900, 3000)]
        assert _disagrees(shifted, ascending=ascending) == (None, 0, {
            1: (_HEART_CX, 1000 + offset), 2: (_HEART_CX, 1900 + offset),
            3: (_HEART_CX, 3000 + offset)})

        displaced = [(_HEART_CX, y) for y in (1000, 1900 + tol + 1, 3000)]
        reason, worst, counted = _disagrees(displaced, ascending=ascending)
        assert reason is not None and worst == tol + 1 and counted == {}

        # And a count that has found MORE hearts than the profile has is refused on sight.
        too_many = [(_HEART_CX, y) for y in (1000, 1900, 3000, 4000)]
        reason, _worst, _counted = _disagrees(too_many, ascending=ascending)
        assert reason is not None and "not there" in reason


def test_every_ordinal_counted_is_checked_and_not_only_those_at_the_target():
    """Checking every ordinal the count has reached — rather than stopping at the target — costs
    nothing and is what turns the anchor's own trivially-zero comparison from unguarded into
    guarded whenever the page offers a second heart. Asserted in both directions, since which
    ordinal is "free" differs: descending it is heart 1, ascending it is the bottom-most one."""
    tol = item_nav._CROSSCHECK_TOLERANCE_PX

    # DESCENDING: heart 1 is the anchor, so only heart 2 can contradict anything.
    reason, worst, _ = _disagrees(
        [(_HEART_CX, 1000), (_HEART_CX, 1900 + tol + 1)], ascending=False)
    assert reason is not None and "heart 2 sits" in reason and worst == tol + 1

    # ASCENDING: heart 3 is the anchor (bottom-most), so heart 2 is where a displacement shows.
    reason, worst, _ = _disagrees(
        [(_HEART_CX, 1900 + tol + 1), (_HEART_CX, 3000)], ascending=True)
    assert reason is not None and "heart 2 sits" in reason and worst == tol + 1

    # The controls: the same pairs where the index says they are, which must stay silent.
    assert _disagrees([(_HEART_CX, 1000), (_HEART_CX, 1900)], ascending=False)[0] is None
    assert _disagrees([(_HEART_CX, 1900), (_HEART_CX, 3000)], ascending=True)[0] is None


def test_the_ascending_anchor_is_the_bottom_most_heart_and_its_ordinal_is_measured():
    """The ascending branch's one direction-dependent decision, isolated. Ascending, the anchor
    is the LAST cluster and its ordinal is not known a priori — the bottom of a profile is not
    guaranteed to show the last card — so it is matched to the index by nearest page row and the
    ordinals run DOWNWARD from it.

    That is exactly the descending branch's mirror, and the third case below is what makes it a
    mirror rather than a coincidence: the same three clusters, read from the other end, produce
    different ordinals and therefore a different verdict."""
    # Two hearts that are really ordinals 2 and 3: ascending gets that right by measurement.
    reason, worst, counted = _disagrees(
        [(_HEART_CX, 1900), (_HEART_CX, 3000)], ascending=True)
    assert (reason, worst) == (None, 0)
    assert counted == {2: (_HEART_CX, 1900), 3: (_HEART_CX, 3000)}

    # ...where the DESCENDING branch would call the same pair ordinals 1 and 2 and refuse, since
    # its anchor claim is "the first heart I saw is heart 1".
    reason, _worst, _counted = _disagrees(
        [(_HEART_CX, 1900), (_HEART_CX, 3000)], ascending=False)
    assert reason is not None and "not the first heart of the profile" in reason

    # A bottom-most heart that matches NO index heart within the entry anchor's residual is a
    # stop rather than a nearest-neighbour guess.
    bound = item_nav._ENTRY_ANCHOR_RESIDUAL_PX + item_nav._CROSSCHECK_TOLERANCE_PX
    assert _disagrees([(_HEART_CX, 3000 + bound - 1)], ascending=True)[0] is None
    reason, _worst, _counted = _disagrees([(_HEART_CX, 3000 + bound)], ascending=True)
    assert reason is not None and "px apart against" in reason

    # ...and counting more hearts above the anchor than the index has ordinals for is a stop,
    # which is the ascending form of "the index holds no heart with ordinal k".
    reason, _worst, _counted = _disagrees(
        [(_HEART_CX, -1000), (_HEART_CX, 1000), (_HEART_CX, 1900), (_HEART_CX, 3000)],
        ascending=True, heart_count=4)
    assert reason is not None and "no ordinal 0 for" in reason


def test_the_entry_anchor_residual_cannot_reach_a_neighbouring_heart():
    """The bound's safety argument, as an assertion rather than a comment. A mis-assignment needs
    the WRONG index heart to be nearer the anchor than the right one, which takes half a card
    pitch; the smallest heart-bearing spacing measured anywhere in the corpus is 738px, so the
    bound has 23x of room. The module asserts this at import; this is the same fact where a
    reader will look for it."""
    bound = item_nav._ENTRY_ANCHOR_RESIDUAL_PX + item_nav._CROSSCHECK_TOLERANCE_PX
    assert bound * 2 < scroll_step._FALLBACK_SPACING_PX // 2


def test_the_descending_anchor_is_still_checked_against_the_top_gates_residual():
    """The descending branch's own anchor rule, kept and kept tested even though production walks
    the other way — an untested mirror is one that can quietly stop being a mirror.

    Everything is measured from that pass's FIRST heart, so a pass whose first heart is really
    the profile's second one agrees with itself perfectly. The bound is the top gate's own
    residual: two confirmed tops may sit (0, 202] apart, far below one card pitch, so an origin
    may differ by that much while the wrong heart cannot be that close. It is EXCLUSIVE, and that
    is a measurement rather than a style choice: 202 is the first offset the gate was measured to
    REFUTE, so two frames it CONFIRMS differ by strictly less."""
    tol = item_nav._CROSSCHECK_TOLERANCE_PX
    bound = item_nav._TOP_ORIGIN_RESIDUAL_PX + tol

    inside = [(_HEART_CX, y + bound - 1) for y in (1000, 1900)]
    assert _disagrees(inside, ascending=False)[:2] == (None, 0)

    # AT the bound, one pixel past it, and — the case that matters — a pass that started counting
    # at heart 2. The first of those three is the one a strict `>` used to admit, and it is not
    # hypothetical: the only real cross-profile pair in the calibration corpus sits exactly 218px
    # apart against exactly this 218px bound.
    assert bound == 218
    for clusters in ([(_HEART_CX, 1000 + bound)],
                     [(_HEART_CX, 1000 + bound + 1)],
                     [(_HEART_CX, 1900), (_HEART_CX, 3000)]):
        reason, _worst, _counted = _disagrees(clusters, ascending=False)
        assert reason is not None
        assert "not the first heart of the profile" in reason or "different profile" in reason


# =====================================================================================
# A STALE OR FOREIGN INDEX: the identity gate, and what geometry alone could never do
#
# Doc 5.3's named reliability failure is a translation table surviving a deck advance. A
# validation pass drove profile A's real index over profile B's real frames and found that at
# model index 1 nothing refused, because Hinge's cards are stereotyped: both profiles' item 1
# measured 974px tall with the heart 885px down at x=938. The fixtures below reproduce that shape
# synthetically — same layout, different header height above card 1, which is the one thing the
# corpus measured as profile-dependent (the background gap above item 1 is 554..697 on one
# profile and 410..479 on the other) — and, separately, the thing that actually differs between
# two people: the sticky header itself.
# =====================================================================================

def _foreign_driver(shift, *, header=_OTHER_HEADER_VALUE):
    """A driver serving a DIFFERENT profile: the same card layout, its first card `shift` px
    further down the page, and a different person's sticky header.

    `header` is exposed so a test can hold identity CONSTANT while varying geometry, which is the
    only way to isolate what the geometric checks can and cannot do. Two real people never share
    a header — that is the measured premise the identity gate rests on — so `header=_HEADER_VALUE`
    is a control, never a scenario."""
    return FakeDriver(header=header, world_kw={"top_gap": _PAGE_TOP_GAP + shift})


def test_the_landing_check_cannot_tell_a_foreign_profiles_card_from_this_ones():
    """The claim `_confirm_landing`'s docstring used to make, pinned as FALSE so it cannot be
    written again. Its three comparisons — card height, heart inset, heart x — are all identical
    between two ordinary Hinge profiles, so it is not the guard against a stale translation
    table; it is the guard against the right ordinal on a differently-SIZED card."""
    index = _reference_index()
    block, heart = index.block_for(1), item_nav.index_hearts(index)[1]
    shift = 218

    class Fake:
        def __init__(self, y0, y1):
            self.y0, self.y1 = y0, y1
        height = property(lambda self: self.y1 - self.y0)

    foreign = Fake(block.page_y0 + shift, block.page_y1 + shift)
    assert item_nav._confirm_landing(
        heart=(heart[0], heart[1] + shift), block=foreign, index_block=block, index_heart=heart,
        extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX,
        crosscheck_tolerance_px=item_nav._CROSSCHECK_TOLERANCE_PX) is None


def test_a_uniformly_translated_foreign_page_is_caught_by_the_entry_drift_and_not_by_geometry():
    """WHAT BOTTOM-UP CHANGED ABOUT THE GEOMETRIC BACKSTOP, stated as a test rather than left for
    someone to discover.

    Under the rewind both passes anchored on their own confirmed scroll top, so a foreign profile
    whose first card sat 218px lower showed up as an anchor GAP. Bottom-up has no second origin:
    the entry shift is measured, so a page that is a uniform translation of the indexed one is
    absorbed by that measurement and every relative heart pitch then agrees exactly. The
    geometric anchor no longer catches this case, and pretending otherwise would be worse than
    losing it.

    What catches it instead, in order:
      * on REAL frames, the entry anchor itself — two different people's cards do not correlate,
        so `estimate_shift` returns no delta at all (the NAV_ANCHOR_UNMEASURED test above). This
        fixture cannot reproduce that, because its "foreign" page is byte-identical content at a
        different offset, which is the worst case rather than the likely one;
      * the entry DRIFT bound, once the translation exceeds the smallest gesture this driver can
        make — 219px on the calibrated transport — because at that point the page has demonstrably
        moved further than anything we could have done to it;
      * and the identity gate, which is the one that actually answers the question, catches BOTH
        halves below, and is the test that follows this one.

    Identity is held CONSTANT here (the same header on both), which is physically impossible for
    two real people and is exactly why it is the right control: it takes the identity gate out of
    the picture so what is left is the geometry on its own."""
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert floor_px == 219

    caught = _foreign_driver(floor_px, header=_HEADER_VALUE)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(caught, 1)
    assert exc.value.code == item_nav.NAV_ANCHOR_UNMEASURED
    assert exc.value.anchor.delta_px == -floor_px
    assert caught.gestures == []                  # refused before it moved the phone

    # AND THE RESIDUAL, stated rather than left implicit: one pixel under that, geometry has
    # nothing to say. The rewind's anchor DID catch this case (218px against its own 218px
    # bound); bottom-up's measured entry shift absorbs it, and the identity gate is what is left.
    missed = _foreign_driver(floor_px - 1, header=_HEADER_VALUE)
    target = _navigate(missed, 1)
    assert target.heart_ordinal == 1 and target.agreement_px == 0


def test_a_nearer_foreign_profile_is_caught_by_the_identity_gate_before_any_gesture():
    """THE HOLE THAT WAS DOCUMENTED HERE, now closed, and the test rewritten rather than deleted
    so the closure is pinned by the same case that used to prove the gap.

    A foreign profile whose page sits within one gesture-floor of this one's is indistinguishable
    to every GEOMETRIC comparison this module can make: the entry shift ABSORBS the offset, so
    every relative heart pitch then agrees exactly rather than merely closely, and the landing
    card is stereotyped. The second half of this test is that fact, and bottom-up makes it
    sharper rather than softer.

    What refuses is the identity gate, on the one part of the screen that says who is on it, and
    it refuses at ENTRY: no gesture at all, one screencap. Doc 5.7's carried-forward
    requirement 1."""
    driver = _foreign_driver(100)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_IDENTITY_MISMATCH
    assert (driver.captures, driver.gestures) == (1, [])
    assert exc.value.frame is not None             # the screen is kept for the operator

    # The control, and the reason the gate had to exist: with identity held constant — which two
    # real people cannot do — the very same frames navigate to a confident tap target for the
    # WRONG person's card, at the ordinal asked for, agreeing with itself to 0px.
    geometry_only = _foreign_driver(100, header=_HEADER_VALUE)
    target = _navigate(geometry_only, 1)
    assert target.heart_ordinal == 1
    assert target.anchor.delta_px == -100
    assert target.point[1] + target.page_offset == _HEART_PAGE_Y[0]
    assert target.agreement_px == 0


def test_every_model_index_refuses_against_a_foreign_profile_in_both_directions():
    """The matrix, synthetically: TWO profiles, each one's index driven over the other's frames,
    for every model index the payload could ever carry. Not one of them may return a target.

    The offline validation runs this same shape over the real calibration captures (9 items each
    way, 18 navigations, all refused). This is the version that lives in the repo, because those
    frames are real people's profiles and never leave `ops/calibration/`."""
    mine = _reference_index(header=_HEADER_VALUE)
    theirs = _reference_index(header=_OTHER_HEADER_VALUE)
    assert mine.identity.known and theirs.identity.known
    assert mine.identity.fingerprint != theirs.identity.fingerprint

    for index, driver_header in ((mine, _OTHER_HEADER_VALUE), (theirs, _HEADER_VALUE)):
        for model_index in range(1, len(index.selectable) + 1):
            driver = FakeDriver(header=driver_header)
            with pytest.raises(item_nav.ItemNavigationError) as exc:
                _navigate(driver, model_index, index=index)
            assert exc.value.code == item_nav.NAV_IDENTITY_MISMATCH
            assert driver.gestures == []

    # The positive control, on the same two indexes: each one navigates fine against its OWN
    # profile, so what refuses above is the identity and not the fixture.
    for index, header in ((mine, _HEADER_VALUE), (theirs, _OTHER_HEADER_VALUE)):
        for model_index in range(1, len(index.selectable) + 1):
            target = _navigate(FakeDriver(header=header), model_index, index=index)
            assert target.heart_ordinal == model_index
            assert target.identity.matched


def _short_profile_index(*, header=_HEADER_VALUE):
    """A synthetic stand-in for blocker 12's SHORT profile: a valid, identity-known index whose
    own last read frame is treated as page offset 0 rather than `_ENTRY_SCROLL`.

    Reuses `_reference_index()`'s real geometry and identity outright and overrides only
    `offsets` — every heart's PAGE row is an ABSOLUTE quantity fixed at build time
    (`test_the_reference_index_is_the_page_that_was_painted` pins it), so replacing the one field
    `navigate_to_item` actually reads off `index.offsets` (`offsets[-1]`, the entry reference
    offset) is enough to make "the read ended at the top" true of this index without hand-building
    a second world. `dataclasses.replace` on a frozen dataclass is the same tool
    `test_the_crosscheck_bound_sums_this_pass_and_the_indexs_own_extent_slack` already uses to vary
    one field of a built index."""
    return dataclasses.replace(_reference_index(header=header), offsets=(0,))


def test_entering_at_a_scroll_top_recovers_with_one_bounded_scroll_and_reaches_the_target():
    """BLOCKER 12 (2026-08-22). A SHORT profile — one photo plus a video, nothing below the fold
    — ends its read exactly where it began, at the card's scroll top: the identity band is
    Hinge's own filter-chips row there, byte-identical across two different people in the
    calibration corpus, so identity was previously not visible at ALL and every such profile was
    unreachable no matter which item the model chose.

    One bounded, guarded forward read-scroll is what reveals the header — sized to the SMALLEST
    legal read-scroll rather than production's own read cadence, because a full-size step risks
    landing beyond `estimate_shift`'s own trust window (measured: a 0.55 `read_scroll_frac` step
    moves ~1299px on this calibrated 2400px screen, past the 900px window) while the floor (219px)
    sits safely inside it. `navigate_to_item` takes that one step — through the driver's own
    humanized `_scroll_down_one`, exactly once — before refusing, folds its OWN measured
    displacement into the entry offset, and only then resumes the ordinary ascending count."""
    index = _short_profile_index()
    driver = FakeDriver(start_scroll=0, allow_entry_scroll=True)
    assert scroll_top.confirm_scroll_top(driver._screencap(),
                                         identity_band=_IDENTITY_BAND).confirmed

    target = _navigate(driver, 1, index=index, reference=_frame(0))

    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert len(driver.entry_scrolls) == 1              # exactly one rescue gesture, never a retry
    # The SMALLEST legal read-scroll, on the ascending walk's own precedent of discarding the
    # policy's frac and keeping only its LANE — the policy's frac here is production's read
    # cadence, which is far too large a single jump to stay inside the trust window above.
    assert driver.entry_scrolls[0] == (hinge._READ_SCROLL_FRAC_MIN, 0.5)
    # reference_offset (0, from the replaced index) + the first-leg anchor (0, entry_reference and
    # the pre-scroll frame are both `_frame(0)`) + this ONE step's own measured displacement.
    assert target.anchor.delta_px == 0
    assert target.entry_offset == floor_px
    assert target.identity.matched
    assert target.heart_ordinal == 1
    assert target.point[1] + target.page_offset == _HEART_PAGE_Y[0]


def test_entering_at_a_scroll_top_still_refuses_a_different_profile_after_the_step():
    """The negative half of the old "entering at a scroll top is an immediate refusal" test, kept
    rather than lost when that test was split (one-line justification: blocker 12, 2026-08-22,
    made a scroll-top entry a RECOVERABLE state rather than a terminal one — see the positive test
    above). The step must still fail a genuinely DIFFERENT profile, because doc 5.6's owner rule
    is that we never substitute one item's card for another's.

    The header the one rescue scroll reveals here belongs to `_OTHER_HEADER_VALUE`, not the
    index's own `_HEADER_VALUE`, so identity still cannot be confirmed afterwards — and the
    refusal carries the ORIGINAL "cannot tell, at scroll top" wording (never rewritten into a
    fresh "mismatch" diagnosis) plus a note that the step was already tried."""
    index = _short_profile_index()
    driver = FakeDriver(start_scroll=0, allow_entry_scroll=True, header=_OTHER_HEADER_VALUE)
    assert scroll_top.confirm_scroll_top(driver._screencap(),
                                         identity_band=_IDENTITY_BAND).confirmed

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, index=index, reference=_frame(0))

    assert exc.value.code == item_nav.NAV_IDENTITY_UNCONFIRMED
    assert "scroll top" in str(exc.value) and "cannot tell" in str(exc.value)
    assert "already" in str(exc.value)                 # the appended recovery note
    assert len(driver.entry_scrolls) == 1               # exactly one attempt, never a second
    assert driver.gestures == []                        # the ascending walk was never entered


def _pinned_evidence(offsets=(0, 900)):
    """The REAL `band_pinned_evidence` answer for a build that pins the chips row to the screen.

    Never hand-constructed: `_frame(s, at_top=True)` paints the shipped filter-chips constant at
    every scroll offset, which is exactly what Hinge 10.1.0's expanded per-profile header does
    (measured `confirmed_top` at distance 0.000 on six frames spanning 2305px of proven scroll),
    and the helper is left to decide for itself whether that proves anything. Building a
    `PinnedBandEvidence(pinned=True, ...)` by hand here would assert the conclusion instead of
    measuring it, and would keep passing if the helper's own conditions were deleted.
    """
    evidence = scroll_top.band_pinned_evidence(
        [_frame(s, at_top=True) for s in offsets],
        identity_band=_IDENTITY_BAND, page_offsets=offsets)
    assert evidence.pinned, evidence.reason        # the fixture must really be a pinned build
    return evidence


def test_the_scroll_top_entry_recovery_declines_to_fire_on_a_build_that_pins_the_chips_row():
    """Blocker 12's rescue step needs a PREMISE, and a pinned build never establishes it.

    The branch reads "the entry frame affirmatively confirms as the chips row, so we are at this
    card's scroll top and one bounded step will reveal the sticky header". On Hinge 10.1.0's
    expanded per-profile header state that row is pinned to the SCREEN, so the raw gate confirms
    on every frame of every offset -- the branch would fire on every entry, spend a real gesture
    on a card that was never at its top, and reveal nothing, because there is no header below to
    reveal. `check_unavailable` is not `.confirmed`, so it declines; the original refusal stands
    with its own wording intact and NO recovery note, because no recovery was attempted.

    `allow_entry_scroll=True` is deliberate: the double would happily permit the one gesture, so
    a gesture not happening is the code declining rather than the fixture forbidding it. The
    identity here is the same `_OTHER_HEADER_VALUE` mismatch the test above uses, so the two
    differ in exactly one thing -- whether the build was proven to pin the row."""
    index = _short_profile_index()
    driver = FakeDriver(start_scroll=0, allow_entry_scroll=True, header=_OTHER_HEADER_VALUE,
                        pinned_evidence=_pinned_evidence())
    # The RAW gate still confirms this very frame: the downgrade is the evidence's doing, not a
    # different frame's. Without that control the test could pass for the wrong reason.
    assert scroll_top.confirm_scroll_top(driver._screencap(),
                                         identity_band=_IDENTITY_BAND).confirmed

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, index=index, reference=_frame(0))

    assert exc.value.code == item_nav.NAV_IDENTITY_UNCONFIRMED
    assert driver.entry_scrolls == []                   # the rescue gesture was never spent
    assert driver.gestures == []                        # nor was the ascending walk entered
    assert "already" not in str(exc.value)              # no recovery note: none was attempted


def test_an_unproven_pinning_leaves_the_entry_recovery_exactly_as_it_was():
    """The negative control for the test above, and the one that keeps it honest.

    `band_pinned_evidence` over frames at ONE offset cannot prove pinning -- a screen-pinned
    strip and page content that simply had not scrolled yet predict identical pixels there -- so
    this drives the SAME code path with a `pinned=False` object and must recover normally. That
    separates "the recovery declined because the evidence proved something" from "the recovery
    declined because a `pinned_evidence` argument was present at all", which is the mistake that
    would silently disable blocker 12 on every build."""
    unproven = scroll_top.band_pinned_evidence(
        [_frame(0, at_top=True)], identity_band=_IDENTITY_BAND, page_offsets=(0,))
    assert unproven.pinned is False

    index = _short_profile_index()
    driver = FakeDriver(start_scroll=0, allow_entry_scroll=True, pinned_evidence=unproven)

    target = _navigate(driver, 1, index=index, reference=_frame(0))

    assert len(driver.entry_scrolls) == 1
    assert target.heart_ordinal == 1
    assert target.identity.matched


def test_a_scroll_top_entry_step_that_cannot_be_measured_is_a_named_refusal():
    """The recovery step is MEASURED like every other leg in this module (doc 5.10's rule,
    restated for blocker 12's one added gesture): if the shift it produced cannot be put in one
    coordinate space, guessing the displacement is exactly how doc 5.10's phantom item was
    fabricated in the first place, so this names the failure rather than assuming the scroll
    landed where it was aimed."""
    index = _short_profile_index()
    driver = FakeDriver(start_scroll=0, allow_entry_scroll=True, entry_scroll_unmeasurable=True)

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, index=index, reference=_frame(0))

    assert exc.value.code == item_nav.NAV_ENTRY_STEP_UNMEASURED
    assert "could not be measured" in str(exc.value)
    assert len(driver.entry_scrolls) == 1
    assert driver.gestures == []


def test_entering_already_scrolled_takes_no_extra_entry_step():
    """The recovery gate is silent unless the entry frame really is a scroll top: the ordinary
    case (the read already left the card scrolled) must cost nothing new, or the rescue step
    becomes a fixed extra gesture on every navigation rather than the rare recovery it is meant
    to be. `FakeDriver()`'s default `start_scroll` is `_ENTRY_SCROLL`, not 0, so this is every
    other test in this file as much as it is its own."""
    driver = FakeDriver()
    target = _navigate(driver, 4)

    assert driver.entry_scrolls == []
    assert driver.captures == 1                        # the entry capture alone, no rescue one
    assert target.heart_ordinal == 4


def test_an_index_that_cannot_say_whose_profile_it_is_cannot_be_navigated_with():
    """The structural half. An index carries its identity as a FIELD, `build_item_index` requires
    the band that produces it, and an index whose identity is UNKNOWN is refused exactly as a
    foreign one is — so "navigate with an index nobody checked" is not a state that can be
    reached by forgetting a call, only by an index that says outright it does not know."""
    anonymous = _reference_index(identity_band=None)
    assert anonymous.usable and not anonymous.identity.known

    driver = FakeDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, index=anonymous)

    assert exc.value.code == item_nav.NAV_IDENTITY_UNCONFIRMED
    assert "no identity fingerprint" in str(exc.value)
    assert driver.gestures == []

    # ...and the same refusal from the other side: a driver that declares no band at all.
    blind = FakeDriver(identity_band=None)
    with pytest.raises(item_nav.ItemNavigationError) as blind_exc:
        _navigate(blind, 1)
    assert blind_exc.value.code == item_nav.NAV_IDENTITY_UNCONFIRMED
    assert blind.gestures == []


def test_a_band_that_moved_since_the_index_was_built_is_a_refusal_not_a_distance():
    """Two different crops of a screen have no distance between them that measures anything, so
    an operator who edits `apps.hinge.identity_band` between the read and the tap gets a stop
    rather than a comparison. `IdentityError` is "could not look", and navigation routes it to
    the same code "cannot tell" gets, keeping the diagnosis in the message."""
    moved = (0.10, 0.020, 0.80, 0.066)
    driver = FakeDriver(identity_band=moved, identity_rect=_IDENTITY_BAND)

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_IDENTITY_UNCONFIRMED
    assert "IdentityError" in str(exc.value)
    assert driver.gestures == []


def test_a_frame_whose_segmentation_contradicts_itself_stops_the_count():
    """`segment.py` calls two hearts in one block a segmentation FAILURE, not a choice. A page
    that cannot be segmented cannot be counted, and the count stops on the frame it happened
    on rather than picking one of the two glyphs."""
    driver = FakeDriver(extra_heart_after=1, extra_heart_card=4)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_FRAME_CONTRADICTS
    assert exc.value.frame_index == 1             # the frame after the first gesture


# =====================================================================================
# Refusal: never reaching the item
# =====================================================================================

def test_an_item_that_never_becomes_fully_visible_is_refused():
    """A tap needs the card bounded end to end — the extent is what the post-tap check compares
    against, and a card whose edges were never seen has no extent. Here the driver's own
    `content_band` is narrower than the card, so the item's heart crosses the screen and off the
    BOTTOM (walking up, content moves down) without the card ever fitting. The run stops rather
    than tapping a heart whose card it never saw whole.

    The control is the same navigation with the shipped band, which lands."""
    narrow = FakeDriver(content_band=(0.30, 0.70))         # 960 rows, against a 1000px card
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(narrow, 3)

    assert exc.value.code == item_nav.NAV_ITEM_NOT_FULLY_VISIBLE
    assert _navigate(FakeDriver(), 3).heart_ordinal == 3


def test_a_lower_edge_background_run_is_seated_before_the_ascending_count(monkeypatch):
    """The live failure: the index projects the whole target card into the band, but the frame
    cannot prove its LOWER edge because a 90px background run could be a blank card tail. A
    reverse gesture makes that uncertainty worse; one measured forward seating step gets a fresh
    lower-edge observation before any count is committed.

    The world normally draws a trustworthy gutter here, so mutate only the FIRST real
    segmentation into the exact ``background_run`` evidence the production screenshot supplied.
    The post-step segmentation is deliberately real and must bound the target before it can land.
    """
    index = _reference_index()
    target_block = index.block_for(3)
    # Project card 3 to frame rows 1010..2010: fully inside the 300..2100 band, as in the live
    # report. Its heart is consequently visible, and a planned legal forward step still leaves
    # both indexed edges in-band.
    entry_scroll = target_block.page_y1 - 2010
    positioned = dataclasses.replace(index, offsets=(entry_scroll,))
    driver = FakeDriver(start_scroll=entry_scroll, allow_entry_scroll=True)

    real_segment = item_nav.segment_frame
    calls = 0

    def unresolved_lower_edge(frame, **kwargs):
        nonlocal calls
        seg = real_segment(frame, **kwargs)
        calls += 1
        if calls != 1:
            return seg
        blocks = []
        for block in seg.blocks:
            if not any(abs(heart[1] - (target_block.heart[1] - entry_scroll))
                       <= item_index._EXTENT_TOLERANCE_PX for heart in block.hearts):
                blocks.append(block)
                continue
            bottom = dataclasses.replace(block.bottom, observed=False, kind="background_run",
                                         run_px=90)
            blocks.append(dataclasses.replace(
                block, bottom=bottom,
                reason="top edge observed; 90px lower background run is not a trusted card end"))
        return dataclasses.replace(seg, blocks=tuple(blocks))

    monkeypatch.setattr(item_nav, "segment_frame", unresolved_lower_edge)

    target = _navigate(driver, 3, index=positioned, reference=_frame(entry_scroll))

    assert len(driver.entry_scrolls) == 1
    assert driver.entry_scrolls[0][0] != _POLICY_FRAC  # the closed loop, not policy, sized it
    assert driver.gestures == []                       # no reverse count was spent before seating
    assert target.steps == ()
    assert target.heart_ordinal == 3
    assert target.block_frame_rows[0] >= _BAND0
    assert target.block_frame_rows[1] <= _BAND1
    assert target.entry_offset > entry_scroll          # the positioning displacement was measured
    assert calls >= 2                                  # the fresh frame supplied the trusted edge


def test_an_unmeasurable_lower_edge_positioning_step_is_a_named_hard_stop():
    index = _reference_index()
    target_block = index.block_for(3)
    entry_scroll = target_block.page_y1 - (_BAND1 + 50)
    positioned = dataclasses.replace(index, offsets=(entry_scroll,))
    driver = FakeDriver(start_scroll=entry_scroll, allow_entry_scroll=True,
                        entry_scroll_unmeasurable=True)

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 3, index=positioned, reference=_frame(entry_scroll))

    assert exc.value.code == item_nav.NAV_ENTRY_POSITION_UNRESOLVED
    assert "could not be measured" in str(exc.value)
    assert len(driver.entry_scrolls) == 1
    assert driver.gestures == []


def test_scroll_top_identity_recovery_never_spends_a_second_forward_positioning_step(monkeypatch):
    """The two entry recoveries share one forward-gesture budget. A short-profile scroll-top
    recovery has already spent it before the post-scroll frame can reveal a lower-edge uncertainty;
    that uncertainty must stop rather than turn into an unbounded two-step forward walk."""
    index = _short_profile_index()
    driver = FakeDriver(start_scroll=0, allow_entry_scroll=True)
    target_block = index.block_for(2)
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    real_segment = item_nav.segment_frame
    calls = 0

    def unresolved_lower_edge_after_identity_recovery(frame, **kwargs):
        nonlocal calls
        seg = real_segment(frame, **kwargs)
        calls += 1
        if calls != 1:
            return seg
        target_y = target_block.heart[1] - floor_px
        blocks = []
        for block in seg.blocks:
            if not any(abs(heart[1] - target_y) <= item_index._EXTENT_TOLERANCE_PX
                       for heart in block.hearts):
                blocks.append(block)
                continue
            bottom = dataclasses.replace(block.bottom, observed=False, kind="background_run",
                                         run_px=90)
            blocks.append(dataclasses.replace(block, bottom=bottom))
        return dataclasses.replace(seg, blocks=tuple(blocks))

    monkeypatch.setattr(item_nav, "segment_frame", unresolved_lower_edge_after_identity_recovery)

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 2, index=index, reference=_frame(0))

    assert exc.value.code == item_nav.NAV_ENTRY_POSITION_UNRESOLVED
    assert "second forward" in str(exc.value)
    assert len(driver.entry_scrolls) == 1
    assert driver.gestures == []


def test_a_top_edge_partial_never_uses_lower_edge_forward_positioning():
    """The direction guard is asymmetric. A card entering through the TOP needs the ordinary
    upward walk (content down) to reveal that edge; a forward positioning step would hide it
    farther above the band. The FakeDriver's default forward-scroll tripwire makes the distinction
    executable rather than documentary."""
    index = _reference_index()
    target_block = index.block_for(3)
    # Card 3 spans frame rows 250..1250: its heart is visible, top is outside the band, and one
    # ordinary ascending gesture exposes the top without ever calling `_scroll_down_one`.
    entry_scroll = target_block.page_y0 - 250
    positioned = dataclasses.replace(index, offsets=(entry_scroll,))
    driver = FakeDriver(start_scroll=entry_scroll)

    target = _navigate(driver, 3, index=positioned, reference=_frame(entry_scroll))

    assert driver.entry_scrolls == []
    assert len(driver.gestures) == 1
    assert target.heart_ordinal == 3


def test_a_lower_edge_clipped_card_taller_than_the_band_never_gets_a_forward_recovery():
    index = _reference_index()
    target_block = index.block_for(3)
    narrow_band = (0.20, 0.60)  # 960px, deliberately shorter than target card 3's 1000px extent
    narrow_y1 = round(narrow_band[1] * _H)
    entry_scroll = target_block.page_y1 - (narrow_y1 + 50)
    positioned = dataclasses.replace(index, offsets=(entry_scroll,))
    driver = FakeDriver(content_band=narrow_band, start_scroll=entry_scroll,
                        allow_entry_scroll=True)

    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 3, index=positioned, reference=_frame(entry_scroll))

    assert exc.value.code == item_nav.NAV_ITEM_NOT_FULLY_VISIBLE
    assert driver.entry_scrolls == []                  # never forward-scroll a card that cannot fit


def test_an_item_below_where_the_read_ended_is_refused_and_named_as_such():
    """The ascending mirror of "it scrolled off the top", and it gets its OWN code because the
    diagnosis differs. Walking up can only move the target further down, so a target already
    below the band on the ENTRY frame means the index and the screen disagree about where the
    bottom of this page is — not that a gesture over-delivered later.

    Reproduced by narrowing the driver's `content_band` after the index was built, which is a
    real operator-config scenario and is the cleanest way to make the analysed band END above a
    heart the index says is inside it. The entry anchor itself is fine (the frames are identical,
    so the shift is 0), which is the point: the refusal is about the TARGET's position, not about
    the origin."""
    index = _reference_index()
    assert item_nav.index_hearts(index)[4][1] == _HEART_PAGE_Y[3]

    # Rows 300..1320, so the entry frame's band ends at page row 4120 — 97px above heart 4.
    narrow = FakeDriver(content_band=(0.125, 0.55))
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(narrow, 4, index=index, selected_model_item_index=3)

    assert exc.value.code == item_nav.NAV_ITEM_BELOW_ENTRY
    assert "BELOW the analysed band" in str(exc.value)
    assert "full-index selectable item 4, selected by model item 3" in str(exc.value)
    assert narrow.gestures == []                  # named on the entry frame, before any gesture

    # The control: the shipped band, same index, same driver position — item 4 is on screen and
    # complete already, so the same navigation returns with no gestures at all.
    assert _navigate(FakeDriver(), 4).scrolls == 0


def test_the_frame_budget_bounds_the_loop_and_is_derived_from_the_ascent():
    """The budget is "how many minimum-size steps could bring this card's TOP edge into the
    band", counted UP from where the read ended — not `scroll_captures` (8, which sizes the
    profile read). Squeezed to two frames it refuses; at its derived value the same navigation
    has room to spare.

    And the item the read is already showing gets a budget of exactly `_FRAME_BUDGET_SLACK`,
    because its ascent is zero. That is the sign flip in the budget, and it is the case the
    descending version could never produce."""
    driver = FakeDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        _navigate(driver, 1, max_frames=2)

    assert exc.value.code == item_nav.NAV_BUDGET_EXHAUSTED
    assert len(driver.gestures) == 1              # two FRAMES is one gesture

    index = _reference_index()
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    derived = item_nav._frame_budget(_ENTRY_SCROLL, index.block_for(1).page_y0, _BAND0, _H)
    assert derived == math.ceil(
        (_ENTRY_SCROLL + _BAND0 - index.block_for(1).page_y0) / floor_px) + 4
    assert derived > driver.scroll_captures
    assert len(_navigate(FakeDriver(), 1).steps) + 1 < derived

    # The last item is already on screen: nothing to climb, so nothing but the slack.
    assert item_nav._frame_budget(
        _ENTRY_SCROLL, index.block_for(4).page_y0, _BAND0, _H) == item_nav._FRAME_BUDGET_SLACK


# =====================================================================================
# The evidence the contract promises
# =====================================================================================

def test_the_target_carries_the_case_for_itself():
    """Doc 5.5's contract ends at "here is where item N's heart is, and here is why I believe
    it". The why has to be on the result, because the next layer's verification and any stop
    record are both written from it rather than from a re-capture."""
    driver = FakeDriver()
    target = _navigate(driver, 1)

    assert target.anchor.ok and target.entry_offset == _ENTRY_SCROLL
    assert target.identity.matched and target.identity.distance == 0.0
    assert len(target.steps) == len(target.shifts) == target.scrolls
    assert len(target.offsets) == len(target.steps) + 1
    assert target.offsets[0] == target.entry_offset
    assert target.offsets[-1] == target.page_offset
    assert all(s.delta_px is not None and s.delta_px < 0 for s in target.shifts)
    assert target.climbed_px == target.entry_offset - target.page_offset > 0
    assert target.frame_index == len(target.steps)
    assert "up from where the profile read ended" in target.reason
    assert "agrees with the index" in target.reason

    # The landing frame is carried, not re-captured: doc 5.6's post-tap check needs the BEFORE
    # picture of the card it is about to open, and a second screencap is a different frame.
    y0, y1 = target.block_frame_rows
    decoded = cv2.imdecode(np.frombuffer(target.frame, np.uint8), cv2.IMREAD_GRAYSCALE)
    painted = _world()[target.block_page_rows[0]:target.block_page_rows[1],
                       _CARD_X0:_CARD_X1]
    assert np.array_equal(decoded[y0:y1, _CARD_X0:_CARD_X1], painted)


def test_target_reason_distinguishes_model_payload_item_from_full_index_position():
    """Payload exclusions can make the model's item number differ from the navigation index."""
    target = _navigate(FakeDriver(), 4, selected_model_item_index=3)

    assert ("page heart 4 is full-index selectable item 4, selected by model item 3"
            in target.reason)
    assert "heart 4 is model item 4" not in target.reason


def test_stop_cancels_before_the_entry_capture_or_any_gesture():
    """Cancellation is a normal action stop and must not even take the first screencap."""
    driver = FakeDriver()
    from operation_love.drivers.base import ActionCancelled
    with pytest.raises(ActionCancelled):
        _navigate(driver, 1, should_stop=lambda: True)
    assert driver.captures == 0
    assert driver.gestures == []


def test_stop_cancels_before_the_next_navigation_gesture():
    """A stop observed after entry processing prevents the first upward swipe."""
    driver = FakeDriver()
    polls = iter((False, True))
    from operation_love.drivers.base import ActionCancelled
    with pytest.raises(ActionCancelled):
        _navigate(driver, 1, should_stop=lambda: next(polls))
    assert driver.captures == 1
    assert driver.gestures == []
