"""Segmentation of a Hinge frame into blocks (operation_love/drivers/segment.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures that
segment.py's constants were measured against are real people's dating profiles and are
gitignored (ops/calibration/, .gitignore:26); only geometry and counts from them appear
anywhere in this repo. So the fixtures rebuild the measured LAYOUT from first principles —
a gradient page background, rounded-rect cards inset 53px per side, 53px gutters — and stamp
the genuine shipped like glyph (HINGE_SPEC.templates["like"]) at known positions. Ground truth
is therefore known by construction, and the assertions can be exact rather than approximate.

Following tests/test_hinge_vision.py's house pattern: textured card interiors (a flat fill
gives cv2.TM_CCOEFF_NORMED spurious perfect scores), the glyph referenced through
HINGE_SPEC.templates rather than by filename, thresholds referenced through
hinge._LIKE_MATCH_THRESHOLD rather than by literal, and every positive paired with a negative
plus a control that proves WHICH mechanism did the excluding.

One fixture detail is load-bearing enough to state up front: `_Frame.card` draws a TRUE rounded
rect, whose corner arc closes exactly `radius` rows in from the edge row. That is the shape the
real captures measure and the shape segment.py's list-boundary handling keys on, so a fixture
that only approximated it would quietly stop testing the thing these tests are about.
"""
import math
import sys

import cv2
import numpy as np
import pytest

from operation_love.drivers import hinge, segment

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 7
_CONTENT_BAND = hinge.HINGE_SPEC.content_band          # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - segment._CARD_MARGIN_PX                # 1027
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px", and inside
assert (min(segment._CARD_CORNER_PX) <= _CORNER_RADIUS_PX <= max(segment._CARD_CORNER_PX)), (
    "the default fixture card must be a card segment.py would accept the corner of")
_HEART_CX = 937                                        # bottom-right, well past _match_glyph's
                                                       # 0.55*width side cutoff (594)

# The page background is a GRADIENT (doc 5.10: ~RGB(255,254,253) at y=300 down to ~(243,243,243)
# at y=2100), which is the whole reason segment.py reads its reference off the page margins per
# row instead of holding one global constant. The fixtures reproduce it so the tests exercise
# that, and test_gradient_defeats_a_single_global_background asserts the fixture really does.
_PAGE_TOP, _PAGE_BOTTOM = 254, 243


def _page_column():
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, _H).round().astype(np.uint8)


class _Frame:
    """A synthetic Hinge profile frame, built by painting onto a gradient page background."""

    def __init__(self):
        self.col = _page_column()
        self.gray = np.repeat(self.col[:, None], _W, axis=1)
        self.rng = np.random.default_rng(_SEED)
        self.template = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
        assert self.template is not None, "shipped like glyph must load"

    def card(self, y0, y1, *, heart_y=None, radius=_CORNER_RADIUS_PX, x0=_CARD_X0, x1=_CARD_X1):
        """A textured rounded-rect card occupying rows [y0, y1). Optionally stamps its heart.

        The corner arc is carved with `ceil`, not `round`, and that is load-bearing rather than
        taste. On a true rounded rect the outermost row is inset by exactly `radius` on each side
        and the inset reaches zero exactly `radius` rows in, so the span the edge row starts from
        and the rows it takes to reach full width agree on one number — which is precisely the
        invariant segment.py's corner test keys on, and precisely what the real captures measure
        (219 of 219 gutter-proven card edges). Rounding the arc instead would close it ~4 rows
        early and make the fixture a shape Hinge never draws.
        """
        self.gray[y0:y1, x0:x1] = self.rng.integers(
            60, 200, size=(y1 - y0, x1 - x0), dtype=np.uint8)
        for i in range(radius):                        # carve the four corner arcs back to page
            dy = radius - i
            inset = int(math.ceil(radius - math.sqrt(max(0.0, radius ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for y in (y0 + i, y1 - 1 - i):
                self.gray[y, x0:x0 + inset] = self.col[y]
                self.gray[y, x1 - inset:x1] = self.col[y]
        if heart_y is not None:
            self.heart(heart_y)
        return self

    def heart(self, cy, cx=_HEART_CX):
        """Stamp the real shipped like glyph centred at (cx, cy)."""
        th, tw = self.template.shape
        self.gray[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = self.template
        return self

    def fill(self, y0, y1, x0, x1, value):
        """An arbitrary painted rectangle — bright uniform photo regions, narrow chrome, etc."""
        self.gray[y0:y1, x0:x1] = value
        return self

    def page(self, y0, y1, x0=0, x1=_W):
        """Repaint a rectangle back to exact page background (a bright uniform photo region
        whose colour happens to match the page — doc 5.4's amendment-one hazard)."""
        self.gray[y0:y1, x0:x1] = self.col[y0:y1, None]
        return self

    def png(self):
        ok, buf = cv2.imencode(".png", self.gray)
        assert ok
        return buf.tobytes()

    def segment(self, content_band=_CONTENT_BAND, **kw):
        return segment.segment_frame(
            self.png(), content_band=content_band, like_template=self.template,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, **kw)


def _kinds(result):
    return [b.kind for b in result.blocks]


def _extents(result):
    return [(b.y0, b.y1) for b in result.blocks]


def _gutters(result):
    return [(r.y0, r.y1) for r in result.runs if r.kind == segment.RUN_GUTTER]


def _card_edge_runs(result):
    return [(r.y0, r.y1) for r in result.runs if r.kind == segment.RUN_CARD_EDGE]


# =====================================================================================
# The clean case
# =====================================================================================

def test_clean_multi_card_frame_segments_correctly():
    """Four cards, three canonical gutters. The two interior cards are bounded by gutters on
    BOTH sides so their extents are fully observed; the outer two are sliced mid-height by the
    content band's edges and must come back PARTIAL rather than pretending to know where they
    end. Note the outer cards START outside the band — a card whose own corner happens to sit on
    the band's first row is a different case, covered by the corner tests below."""
    f = _Frame()
    f.card(200, 700, heart_y=640)                       # sliced by the band's top edge
    f.card(700 + _GUTTER, 1150, heart_y=1090)           # 753..1150, complete, one heart
    f.card(1150 + _GUTTER, 1418)                        # 1203..1418, complete, NO heart
    f.card(1418 + _GUTTER, 2300, heart_y=2000)          # sliced by the band's bottom edge
    r = f.segment()

    assert r.ok, r.failures
    assert _kinds(r) == [segment.BLOCK_PARTIAL, segment.BLOCK_SELECTABLE,
                         segment.BLOCK_CONTEXT, segment.BLOCK_PARTIAL]
    assert _extents(r) == [(_BAND0, 700), (753, 1150), (1203, 1418), (1471, _BAND1)]
    assert _gutters(r) == [(700, 753), (1150, 1203), (1418, 1471)]
    assert all(g.height == _GUTTER for g in r.runs if g.kind == segment.RUN_GUTTER)
    assert r.unassigned_hearts == ()

    selectable = r.selectable
    assert len(selectable) == 1                          # PARTIAL blocks are never selectable,
    assert selectable[0].y0 == 753                       # even when they do carry a heart
    assert selectable[0].heart == (_HEART_CX, 1090)
    assert selectable[0].complete

    context = r.blocks[2]
    assert context.hearts == () and context.heart is None
    assert context.complete and "no like heart" in context.reason
    assert (context.x0, context.x1) == (_CARD_X0, _CARD_X1)

    # The two outer blocks say WHY they are partial, and which edge was the problem.
    assert r.blocks[0].top.kind == segment.EDGE_BAND_EDGE and not r.blocks[0].top.observed
    assert r.blocks[0].bottom.kind == segment.EDGE_GUTTER and r.blocks[0].bottom.observed
    assert r.blocks[-1].bottom.kind == segment.EDGE_BAND_EDGE
    assert "true extent unknown from this frame" in r.blocks[0].reason


def test_gradient_defeats_a_single_global_background():
    """A gutter at the top of the content band and one at the bottom are both found, with the
    same tolerance, even though the page background differs between them by more than that
    tolerance. That difference is the control: it is exactly what a single global background
    constant could not have absorbed (doc 5.10, amendment one)."""
    f = _Frame()
    f.card(_BAND0, 420, heart_y=380)
    f.card(420 + _GUTTER, 1900, heart_y=1840)
    f.card(1900 + _GUTTER, _BAND1, heart_y=2050)

    top_bg, bottom_bg = int(f.gray[430, 500]), int(f.gray[1960, 500])
    assert abs(top_bg - bottom_bg) > segment._BACKGROUND_TOLERANCE, (
        "fixture is not actually a gradient, so this test proves nothing")

    r = f.segment()
    assert r.ok, r.failures
    assert _gutters(r) == [(420, 473), (1900, 1953)]     # both ends of the band, one tolerance


# =====================================================================================
# Doc 5.4 amendment one: colour alone is not sufficient — a 192-row false span
# =====================================================================================

def test_bright_uniform_span_inside_a_card_does_not_split_it():
    """The measured hazard: 192 contiguous rows inside ONE card whose colour matches the page
    background exactly (snow, sky, a white shirt, a prompt card's blank middle). It must not
    become a block boundary, because splitting a card renumbers every item below it."""
    f = _Frame()
    f.card(_BAND0, 700, heart_y=640)
    f.card(753, 1800, heart_y=1740)
    f.page(1000, 1192, _CARD_X0, _CARD_X1)               # 192 rows, exact page background
    f.card(1853, _BAND1, heart_y=2050)
    r = f.segment()

    assert r.ok, r.failures
    assert _extents(r) == [(_BAND0, 700), (753, 1800), (1853, _BAND1)]
    block = r.blocks[1]
    assert block.kind == segment.BLOCK_SELECTABLE and block.height == 1047

    # The control: the false span WAS seen, and it was the LENGTH gate that rejected it — not
    # the colour test (which it passed) and not the span test (which it also passed).
    false_run = [run for run in r.runs if (run.y0, run.y1) == (1000, 1192)]
    assert len(false_run) == 1
    assert false_run[0].kind == segment.RUN_TOO_LONG
    assert false_run[0].widest_intruder_px == 0          # genuinely indistinguishable by colour


def test_a_gutter_sized_background_span_does_split():
    """The negative control for the test above: same fixture, same colour, only the LENGTH
    changed from 192 rows to the canonical 53. Now it splits — which proves the previous test's
    single block came from the length gate and not from some other accident of the fixture."""
    f = _Frame()
    f.card(_BAND0, 700, heart_y=640)
    f.card(753, 1800, heart_y=1740)
    f.page(1000, 1000 + _GUTTER, _CARD_X0, _CARD_X1)
    f.card(1853, _BAND1, heart_y=2050)
    r = f.segment()

    assert _extents(r) == [(_BAND0, 700), (753, 1000), (1053, 1800), (1853, _BAND1)]
    assert [run.kind for run in r.runs if run.y0 == 1000] == [segment.RUN_GUTTER]
    assert r.blocks[1].kind == segment.BLOCK_CONTEXT      # the fragment above the false gutter
    assert r.blocks[2].kind == segment.BLOCK_SELECTABLE   # the fragment carrying the heart


# =====================================================================================
# Doc 5.4 amendment two: a narrow element must not hide a gutter
# =====================================================================================

def test_narrow_element_in_a_gutter_does_not_hide_it():
    """The measured miss: a real gutter was lost to an "any pixel differs from background" test
    because a narrow element spanned only x=30..216 across it (Hinge's floating pass-X, which
    overlaps the gutter by design). The row test keys on the horizontal SPAN of non-background
    pixels, so a ~190px intruder leaves the gutter perfectly visible."""
    f = _Frame()
    f.card(_BAND0, 700, heart_y=640)
    f.card(753, 1600, heart_y=1540)
    f.fill(700, 753, 30, 217, 40)                        # dark, narrow, spans the whole gutter
    f.card(1653, _BAND1, heart_y=2050)
    r = f.segment()

    assert r.ok, r.failures
    assert (700, 753) in _gutters(r)
    assert _extents(r) == [(_BAND0, 700), (753, 1600), (1653, _BAND1)]
    intruded = [run for run in r.runs if (run.y0, run.y1) == (700, 753)][0]
    assert intruded.widest_intruder_px > 0, "the intruder must actually be in the fixture"
    assert intruded.widest_intruder_px < segment._CARD_ROW_MIN_SPAN_FRAC * (_CARD_X1 - _CARD_X0)


def test_a_full_width_element_in_a_gutter_does_hide_it():
    """The control for the test above: widen the same intruder from ~190px to the card's full
    width and the gutter correctly disappears, merging the two cards into one block that then
    reports itself as a two-heart segmentation FAILURE. That is the mechanism the previous test
    depends on — span, not "some pixel differs" — proven by making it fire."""
    f = _Frame()
    f.card(_BAND0, 700, heart_y=640)
    f.card(753, 1600, heart_y=1540)
    f.fill(700, 753, _CARD_X0, _CARD_X1, 40)
    f.card(1653, _BAND1, heart_y=2050)
    r = f.segment()

    assert (700, 753) not in _gutters(r)
    assert not r.ok
    assert r.blocks[0].kind == segment.BLOCK_AMBIGUOUS


# =====================================================================================
# Doc 5.3: hearts decide the class
# =====================================================================================

def test_heartless_block_is_context_and_the_same_block_with_a_heart_is_selectable():
    """The vitals block (doc 5.10: ~215px tall, carries no heart, tracked across 7 consecutive
    real frames) versus a photo/prompt card. Identical geometry, so the ONLY thing separating
    context from selectable is the heart."""
    for heart_y, expected in ((None, segment.BLOCK_CONTEXT), (1360, segment.BLOCK_SELECTABLE)):
        f = _Frame()
        f.card(_BAND0, 1150, heart_y=1090)
        f.card(1203, 1418, heart_y=heart_y)              # 215px, exactly the measured vitals size
        f.card(1471, _BAND1, heart_y=2050)
        r = f.segment()

        assert r.ok, r.failures
        block = r.blocks[1]
        assert block.height == 215
        assert block.kind == expected, block.reason
        assert (block.heart is None) == (heart_y is None)


def test_two_hearts_in_one_block_reports_a_failure_instead_of_guessing():
    """A block cannot carry two hearts — every likeable Hinge item has exactly one — so this is
    a statement that a gutter was missed. It must surface as a failure, and `heart` must stay
    None: returning hearts[0] here is precisely the guess that would silently comment on the
    wrong item."""
    f = _Frame()
    f.card(_BAND0, 700, heart_y=640)
    f.card(753, 1800, heart_y=900).heart(1740)           # two hearts, one block
    f.card(1853, _BAND1, heart_y=2050)
    r = f.segment()

    assert not r.ok
    block = r.blocks[1]
    assert block.kind == segment.BLOCK_AMBIGUOUS
    assert len(block.hearts) == 2 and block.heart is None
    assert block not in r.selectable
    assert any("ambiguous" in msg for msg in r.failures)
    assert "2 like hearts inside one block" in block.reason


def test_the_same_two_hearts_split_by_a_gutter_are_two_selectable_blocks():
    """Control for the test above: the hearts are not the problem, the missing boundary is.
    Put a canonical gutter between the identical pair and both become ordinary items."""
    f = _Frame()
    f.card(200, 700, heart_y=640)                        # sliced by the band edges, so the two
    f.card(753, 1150, heart_y=900)                       # interior cards are the only complete
    f.card(1203, 1800, heart_y=1740)                     # ones and the count below is exact
    f.card(1853, 2300, heart_y=2050)
    r = f.segment()

    assert r.ok, r.failures
    assert [b.kind for b in r.blocks[1:3]] == [segment.BLOCK_SELECTABLE,
                                               segment.BLOCK_SELECTABLE]
    assert len(r.selectable) == 2


def test_a_heart_below_the_last_card_row_is_still_assigned_to_its_block():
    """A card whose bottom is blank page-coloured except for the heart itself (a short prompt
    answer) puts the heart below the last row that clears the span test. Assigning hearts by
    SEGMENT rather than by the trimmed extent keeps it attached, and the extent is widened to
    cover the glyph so a crop of the block still contains it. Dropping it would silently demote
    a selectable card to a context block, which is a wrong item list, not a cosmetic error.

    The block's bottom must stay UNOBSERVED even though the card's real bottom corner is right
    there at row 1299: once the heart has pushed the reported extent past it, the corner no
    longer describes the edge being reported, and calling it observed would hand back a 1445 that
    was measured at 1300."""
    f = _Frame()
    f.card(200, 700, heart_y=640)
    f.card(753, 1300)                                    # dense card content stops at 1300
    f.heart(1400)                                        # ...but its heart sits 100 rows lower
    r = f.segment()

    assert r.unassigned_hearts == ()
    block = r.blocks[1]
    assert block.hearts == ((_HEART_CX, 1400),)
    assert block.y1 >= 1400 + f.template.shape[0] // 2, "crop must contain the whole glyph"
    assert block.y1 > 1300, "fixture must actually exercise the widening"
    assert block.kind == segment.BLOCK_PARTIAL           # its bottom still runs off the band
    assert block.bottom.kind == segment.EDGE_BACKGROUND_RUN
    assert not block.bottom.observed and block.bottom.corner_px is None
    assert block.bottom.run_px > 0                       # background beyond it, length unknown


# =====================================================================================
# Doc 5.4, failure class three: the list boundaries have chrome, not page background.
#
# There is no gutter above item 1 or below item N, so these are the cases the interior gutter
# rule cannot reach. They are settled by the card's OWN rounded corner, which is why none of
# these tests passes a boundary row in: an earlier revision took `list_top_y`/`list_bottom_y`
# as parameters and the calibration corpus proved no constant can serve two profiles.
# =====================================================================================

def _scroll_top_frame(*, header_y1=446, name_y0=497, name_y1=561, card_y0=700):
    """Scroll-top, rebuilt from the measured layout: a header strip spanning the card width, a
    gap, a wide-but-not-full-width name row, a longer gap, then the first card (974px), a
    canonical gutter, and the start of the second.

    The default 51-row header gap is the important part. Chrome whitespace is under no obligation
    to avoid the canonical gutter window, and here it does not — so this fixture is the hostile
    case where the gutter-length gate alone cannot tell chrome apart from layout. (On the real
    captures the equivalent gap measured 60 and 69 rows.) Nothing about the first card's recovery
    may depend on that gap being rejected.
    """
    f = _Frame()
    f.fill(_BAND0, header_y1, _CARD_X0, _CARD_X1, 90)    # header strip, card width, no corners
    f.fill(name_y0, name_y1, 100, 800, 70)               # name row: wide, but not full width
    f.card(card_y0, card_y0 + 974, heart_y=card_y0 + 914)
    f.card(card_y0 + 974 + _GUTTER, 2300, heart_y=card_y0 + 1200)
    return f


def test_the_first_card_of_the_list_is_bounded_by_its_own_corner():
    """Item 1 has no gutter above it, ever. Its top edge is recovered from the card's own rounded
    corner: the chrome gap above it is not gutter-length, but a card starts on the row directly
    below it, and that is what makes the run a boundary. No caller declaration, no constant."""
    r = _scroll_top_frame().segment()

    assert r.ok, r.failures
    first = [b for b in r.blocks if b.y0 == 700]
    assert len(first) == 1, _extents(r)
    first = first[0]
    assert (first.y0, first.y1) == (700, 1674) and first.height == 974
    assert first.kind == segment.BLOCK_SELECTABLE and first.complete
    assert first.top.kind == segment.EDGE_CARD_CORNER and first.top.observed
    assert first.top.corner_px == _CORNER_RADIUS_PX      # the radius it was measured at
    assert first.bottom.kind == segment.EDGE_GUTTER

    # The mechanism, spelled out: the 139-row chrome gap cut the block, and it cut it because of
    # the corner below it and not because of its length.
    assert (561, 700) in _card_edge_runs(r)
    assert (446, 497) in _gutters(r), "fixture must keep the hostile gutter-length chrome gap"

    # And the chrome itself never enters the model's view: neither strip is selectable, and
    # neither is a CONTEXT block either, because neither of their ends is gutter- or
    # corner-bounded.
    chrome = [b for b in r.blocks if b.y1 <= 700]
    assert chrome and all(b.kind == segment.BLOCK_PARTIAL for b in chrome), _kinds(r)


def test_two_profiles_with_different_headers_both_recover_item_one():
    """THE regression this redesign exists for. The header above item 1 is profile-dependent:
    measured, the two calibration profiles put item 1's top at row 697 and row 479, so a declared
    boundary would have to fall inside 554..696 on one and 410..478 on the other — DISJOINT
    windows. The fixture reproduces that disjointness, and both profiles must come back exact
    from the SAME call with no per-profile input."""
    tall = _scroll_top_frame(header_y1=446, name_y0=497, name_y1=561, card_y0=700)
    short = _scroll_top_frame(header_y1=330, name_y0=345, name_y1=409, card_y0=480)

    # The control: no single row lies in both chrome gaps, so no constant could have served both.
    assert set(range(561, 700)).isdisjoint(range(409, 480))

    for f, top in ((tall, 700), (short, 480)):
        r = f.segment()
        assert r.ok, r.failures
        card = [b for b in r.blocks if b.kind == segment.BLOCK_SELECTABLE and b.height == 974]
        assert len(card) == 1, _extents(r)
        assert (card[0].y0, card[0].y1) == (top, top + 974)
        assert card[0].top.kind == segment.EDGE_CARD_CORNER


def test_a_chrome_row_is_not_mistaken_for_a_card_top():
    """The negative control for the two tests above. Same fixture, but the row below the long
    chrome gap is the name strip rather than a card — wide, flat-topped, no corner arc. The run
    must NOT cut, and the block below it must NOT claim an observed top edge. Measured, this is
    a wide margin and not a near miss: Hinge's real header rows imply corner radii of 68.5px and
    up against an accepted window of 18..25."""
    f = _Frame()
    f.fill(_BAND0, 446, _CARD_X0, _CARD_X1, 90)          # header strip
    f.fill(600, 664, 100, 800, 70)                       # name row below a 154-row gap
    f.card(753, 1727, heart_y=1667)                      # a real card, 974px
    r = f.segment()

    assert _card_edge_runs(r) == [(664, 753)]            # only the CARD's gap cut, not the name's
    assert (446, 600) not in _card_edge_runs(r)
    merged = [b for b in r.blocks if b.y0 == _BAND0]
    assert len(merged) == 1 and merged[0].y1 == 664      # header + name stayed one block
    assert merged[0].kind == segment.BLOCK_PARTIAL
    assert merged[0].bottom.kind == segment.EDGE_BACKGROUND_RUN


def test_the_last_card_of_the_list_is_bounded_by_its_own_corner():
    """Item N has no gutter below it — just page background to the dark bottom nav. Its bottom
    corner, with that background beyond it, is the evidence. This is what makes the last item
    selectable at all; before the corner test it was permanently PARTIAL."""
    f = _Frame()
    f.card(200, 700, heart_y=640)
    f.card(753, 1700, heart_y=1640)                      # the last card; page background below
    r = f.segment()

    last = r.blocks[-1]
    assert (last.y0, last.y1) == (753, 1700)
    assert last.kind == segment.BLOCK_SELECTABLE and last.complete
    assert last.bottom.kind == segment.EDGE_CARD_CORNER and last.bottom.observed
    assert last.bottom.corner_px == _CORNER_RADIUS_PX
    assert last.bottom.run_px == _BAND1 - 1700           # page background beyond it, reported


def test_a_card_sliced_by_the_band_edge_is_still_partial():
    """The negative control for the test above, and the reason a caller-declared boundary was
    unsafe: a card whose content simply runs off the analysed band has background nowhere near
    it, no corner to measure, and must stay PARTIAL. Same fixture as the last-card test with the
    card extended past the band."""
    f = _Frame()
    f.card(200, 700, heart_y=640)
    f.card(753, 2300, heart_y=1640)                      # runs off the bottom of the band
    r = f.segment()

    last = r.blocks[-1]
    assert last.y1 == _BAND1
    assert last.kind == segment.BLOCK_PARTIAL
    assert last.bottom.kind == segment.EDGE_BAND_EDGE
    assert not last.bottom.observed and last.bottom.corner_px is None


def test_a_gutter_clipped_by_the_band_edge_is_recovered_from_the_corner():
    """A band edge landing inside a gutter used to cost the whole block: the run is CLIPPED, so
    its length is only a lower bound and cannot be tested against the gutter window. The card's
    own bottom corner settles it anyway, which is a straight coverage gain on ordinary
    mid-scroll frames — not just at the two list boundaries."""
    f = _Frame()
    f.card(200, 700, heart_y=640)
    f.card(753, 2080, heart_y=2020)                      # ends 20 rows above the band edge
    r = f.segment()

    clipped = [run for run in r.runs if run.kind == segment.RUN_CLIPPED]
    assert [(run.y0, run.y1) for run in clipped] == [(2080, _BAND1)]
    last = r.blocks[-1]
    assert (last.y0, last.y1) == (753, 2080)
    assert last.kind == segment.BLOCK_SELECTABLE and last.complete
    assert last.bottom.kind == segment.EDGE_CARD_CORNER


# =====================================================================================
# The corner test's two safety properties. Both are the difference between "trusted edge"
# and "confidently wrong extent", so both get an explicit control.
# =====================================================================================

def test_a_rounded_element_inside_a_card_cannot_forge_a_card_edge():
    """Safety property one: the arc must reach the FULL card width. Only the card itself spans
    x=53..1026 — anything drawn inside one is inset by the card's own padding — so this is what
    stops a card-shaped element inside a photo from cutting a card in two and renumbering every
    item below it.

    The element here is built to defeat every OTHER part of the test: its top row's span is
    exactly the 930px a real 22px corner would start from, and its own arc closes monotonically.
    It is 24px too narrow, and that alone must sink it."""
    f = _Frame()
    f.card(200, 700, heart_y=640)
    f.card(753, 1800, heart_y=1740)
    f.page(1000, 1100, _CARD_X0, _CARD_X1)               # 100-row gap: too long to be a gutter
    f.page(1100, 1500, _CARD_X0, _CARD_X1)
    f.card(1100, 1500, x0=65, x1=1015, radius=10)        # 950px wide: span starts at 930...
    r = f.segment()

    assert r.ok, r.failures
    assert _card_edge_runs(r) == []
    assert (753, 1800) in _extents(r), "the card must not have been split"
    faked = [run for run in r.runs if (run.y0, run.y1) == (1000, 1100)]
    assert len(faked) == 1 and faked[0].kind == segment.RUN_TOO_LONG


def test_a_corner_is_rejected_when_the_two_radius_estimates_disagree():
    """Safety property two: it is the AGREEMENT between the edge row's span and the ramp length
    that identifies a corner, not the span alone. The chamfer below starts at exactly the 930px
    a 22px corner starts from and then jumps to full width on the very next row, so the two
    estimates read 22 and 1 — no cut, and the 300-row gap above it stays a merged block.

    The second half is the control that proves it was the ramp and not the fixture: replace the
    chamfer with a genuine 22px arc, change nothing else, and the identical gap now cuts."""
    def frame(top_shape):
        f = _Frame()
        f.card(200, 700, heart_y=640)                     # sliced by the band's top edge
        top_shape(f)                                      # ...300 rows of page background, then:
        return f

    def chamfer(f):
        f.fill(1000, 1001, _CARD_X0 + 22, _CARD_X1 - 22, 40)   # one row at span 930...
        f.card(1001, 1700, radius=0)                           # ...then full width immediately

    r = frame(chamfer).segment()
    assert _card_edge_runs(r) == []
    assert [run.kind for run in r.runs if run.y0 == 700] == [segment.RUN_TOO_LONG]
    assert _extents(r) == [(_BAND0, 1700)], "the chamfer must not have cut the block"
    assert r.blocks[0].kind == segment.BLOCK_PARTIAL

    r2 = frame(lambda f: f.card(1000, 1700, radius=_CORNER_RADIUS_PX)).segment()
    assert _card_edge_runs(r2) == [(700, 1000)]
    assert _extents(r2) == [(_BAND0, 700), (1000, 1700)]
    assert r2.blocks[1].top.kind == segment.EDGE_CARD_CORNER


def test_a_corner_outside_the_measured_radius_window_is_rejected():
    """The window is Hinge's measured card radius (18..25px on the calibrated device), not "any
    rounded shape". A 40px-radius rounded rect is a perfectly self-consistent arc — both
    estimates agree on 40 — and is still not a Hinge card, so it must not bound a block."""
    f = _Frame()
    f.card(200, 700, heart_y=640)
    f.card(1000, 1700, heart_y=1640, radius=40)
    r = f.segment()

    assert 40 > max(segment._CARD_CORNER_PX), "fixture must sit outside the accepted window"
    assert _card_edge_runs(r) == []
    assert _extents(r) == [(_BAND0, 1700)]               # merged, because nothing cut
    assert not r.ok                                      # ...and loudly: two hearts, one block
    assert r.blocks[0].kind == segment.BLOCK_AMBIGUOUS


# =====================================================================================
# Fail loud, and the limits of what "loud" can mean from one frame
# =====================================================================================

def test_a_frame_with_no_cards_is_not_a_failure_and_not_a_screen_verdict():
    """Segmentation is not screen recognition. Fed something that is not a profile at all — here
    a full-width sheet with no card geometry, standing in for the out-of-likes paywall — it must
    return an honest "I could not bound anything here" rather than inventing blocks OR raising.
    `ok` stays True, which is exactly why `ok` must never be read as "this is a profile": on the
    real captures 15 of 148 genuine profile frames also complete no block. The screen check is a
    separate, earlier decision."""
    f = _Frame()
    f.fill(_BAND0, _BAND1, 0, _W, 120)                   # edge to edge: no page margin, no cards
    r = f.segment()

    assert r.ok and r.failures == ()
    assert r.hearts == () and r.unassigned_hearts == ()
    assert r.selectable == ()
    assert all(b.kind == segment.BLOCK_PARTIAL for b in r.blocks)


def test_undecodable_frame_raises_rather_than_returning_no_blocks():
    with pytest.raises(segment.SegmentationError, match="did not decode"):
        segment.segment_frame(b"not an image", content_band=_CONTENT_BAND,
                              like_template=hinge._load_template(
                                  hinge.HINGE_SPEC.templates["like"]),
                              like_threshold=hinge._LIKE_MATCH_THRESHOLD)


def test_missing_like_template_raises_rather_than_calling_every_card_context():
    """_match_glyph returns [] for a None template. Passing that through would relabel every
    selectable card on the frame as a heartless CONTEXT block — a silently emptied item list."""
    f = _Frame()
    f.card(_BAND0, 1150, heart_y=1090)
    with pytest.raises(segment.SegmentationError, match="no like-glyph template"):
        segment.segment_frame(f.png(), content_band=_CONTENT_BAND, like_template=None,
                              like_threshold=hinge._LIKE_MATCH_THRESHOLD)


def test_degenerate_band_raises():
    f = _Frame()
    f.card(_BAND0, 1150, heart_y=1090)
    with pytest.raises(segment.SegmentationError, match="analysed band is empty"):
        f.segment(content_band=(0.5, 0.5))


def test_missing_cv2_raises_instead_of_degrading(monkeypatch):
    """Every other vision helper in the driver falls back to a fixed coordinate when cv2 is
    absent. Segmentation must not: the thing it would have to invent is which items exist."""
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(segment.SegmentationError, match="opencv-python"):
        segment.segment_frame(b"", content_band=_CONTENT_BAND, like_template=object(),
                              like_threshold=hinge._LIKE_MATCH_THRESHOLD)


def test_hitting_the_matcher_hit_cap_is_a_failure():
    """_match_glyph's non-max-suppression loop runs a fixed 12 iterations, so at 12 hits the
    true heart count is unknown. An index built on an unknown count is a guess."""
    f = _Frame()
    f.card(_BAND0, _BAND1)
    for i in range(13):
        f.heart(_BAND0 + 60 + i * 130)
    r = f.segment()

    assert len(r.hearts) == segment._MATCH_GLYPH_HIT_CAP
    assert not r.ok
    assert any("non-max-suppression cap" in msg for msg in r.failures)
