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
from pathlib import Path

import cv2
import numpy as np
import pytest

from operation_love.drivers import hinge, item_index, segment

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

    def round_bottom_corners(self, y0, y1, *, radius=_CORNER_RADIUS_PX, x0=_CARD_X0, x1=_CARD_X1):
        """Carve only the BOTTOM two corners of an already-painted card rect, leaving its top
        edge flat full-width. Same arc math as `card()` above (see its docstring for why `ceil`
        is load-bearing), applied to one edge only, so a fixture can give a card a real rounded
        corner on the edge under test (segment.py's own corner rule, `_corner_radius`) without
        also creating one at the edge that a test needs to stay a plain, non-corner boundary."""
        for i in range(radius):
            dy = radius - i
            inset = int(math.ceil(radius - math.sqrt(max(0.0, radius ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            y = y1 - 1 - i
            self.gray[y, x0:x0 + inset] = self.col[y]
            self.gray[y, x1 - inset:x1] = self.col[y]
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


def _low_contrast_scroll_top(*, card_y1=1491, heart_y=1402, bottom_gutter=True):
    """The sanitized geometry of the reported first-photo failure, with no profile pixels."""
    f = _Frame()
    f.fill(368, 411, 100, 900, 40)                    # name/header content
    f.card(517, card_y1, heart_y=heart_y)
    # Keep one pale edge indistinguishable from the page after the real corner. The initial
    # bilateral span still implies a 22px radius and expands like a corner, but never reaches
    # full card width, which is exactly why the ordinary strict detector must decline it.
    for y in range(517, min(550, card_y1)):
        f.page(y, y + 1, _CARD_X1 - _CORNER_RADIUS_PX, _CARD_X1)
    if bottom_gutter:
        f.card(card_y1 + _GUTTER, _BAND1 + 100, heart_y=2000)
    return f


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


def test_caption_seam_that_is_card_white_does_not_split_one_compound_photo_card():
    """Regression for the live `Selfie #503` capture.

    Hinge put a 135px caption panel and its photo inside one rounded card, with a 47px blank
    white seam between them.  Forty-seven pixels is inside the accepted gutter window, so the
    old length-only rule split the caption into a heartless context block and the media into a
    second card.  The seam is card white, however, not the page background beside it.  That
    affirmative colour evidence must keep the outer rounded card whole; a genuine gutter of
    the same layout still cuts normally immediately below it.
    """
    f = _Frame()
    f.card(200, 379, heart_y=340)                       # top-clipped previous card
    f.card(432, 1541, heart_y=1452)                    # caption + media, one outer card
    seam_y0, seam_y1 = 567, 614                        # exact live 47px internal seam
    card_white = np.minimum(
        f.col[seam_y0:seam_y1].astype(np.int16) + 2, 255).astype(np.uint8)
    f.gray[seam_y0:seam_y1, _CARD_X0:_CARD_X1] = card_white[:, None]
    f.card(1594, 2300, heart_y=2050)                   # true 53px gutter above

    r = f.segment()

    assert r.ok, r.failures
    assert _extents(r) == [(_BAND0, 379), (432, 1541), (1594, _BAND1)]
    compound = r.blocks[1]
    assert compound.kind == segment.BLOCK_SELECTABLE
    assert compound.complete and compound.hearts == ((_HEART_CX, 1452),)
    assert (seam_y0, seam_y1) not in _gutters(r)
    seam = next(run for run in r.runs if (run.y0, run.y1) == (seam_y0, seam_y1))
    assert seam.kind == segment.RUN_CARD_SURFACE
    assert seam.median_level_delta > segment._GUTTER_BACKGROUND_LEVEL_TOLERANCE
    assert (1541, 1594) in _gutters(r), "the real page-background gutter must still cut"

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


def test_heart_anchored_tall_media_gap_splits_without_widening_generic_gutters():
    """Rebecca's 98..101px photo gap ends exactly one normal heart inset below item 1."""
    f = _Frame()
    f.card(200, 700, heart_y=611, radius=0)             # heart is 89px above the tall gap
    f.card(801, 1400, heart_y=1311, radius=0)           # 101px page-coloured media gap
    f.card(1453, _BAND1 + 100, heart_y=2000, radius=0)  # ordinary 53px control gutter

    r = f.segment()

    assert r.ok, r.failures
    assert _extents(r) == [(_BAND0, 700), (801, 1400), (1453, _BAND1)]
    tall_gap = next(run for run in r.runs if (run.y0, run.y1) == (700, 801))
    assert tall_gap.kind == segment.RUN_HEART_ANCHORED_MEDIA_GUTTER
    assert r.blocks[0].bottom.kind == segment.EDGE_HEART_ANCHORED_MEDIA_GUTTER
    assert r.blocks[1].top.kind == segment.EDGE_HEART_ANCHORED_MEDIA_GUTTER
    assert r.blocks[1].kind == segment.BLOCK_SELECTABLE


def test_two_hearts_prove_a_low_contrast_near_gutter_is_a_card_boundary():
    """A 60px page-coloured gap must not merge a prompt and its following photo.

    It remains too long for the ordinary 47..58px gutter rule.  The split is licensed only
    because independently matched hearts prove Hinge items on both sides of it.
    """
    f = _Frame()
    f.card(200, 799, heart_y=710, radius=0)
    f.card(859, 1826, heart_y=1737, radius=0)          # 60px low-contrast boundary
    f.card(1879, _BAND1 + 100, heart_y=2000, radius=0)

    r = f.segment()

    assert r.ok, r.failures
    near_gap = next(run for run in r.runs if (run.y0, run.y1) == (799, 859))
    assert near_gap.kind == segment.RUN_HEART_SEPARATED_NEAR_GUTTER
    assert r.blocks[0].bottom.kind == segment.EDGE_HEART_SEPARATED_NEAR_GUTTER
    assert r.blocks[1].top.kind == segment.EDGE_HEART_SEPARATED_NEAR_GUTTER
    assert r.blocks[1].kind == segment.BLOCK_SELECTABLE

    # The same slightly long blank span without a lower heart is not enough to invent a card.
    unproven = _Frame()
    unproven.card(200, 799, heart_y=710, radius=0)
    unproven.card(859, 1826, radius=0)
    unproven.card(1879, _BAND1 + 100, heart_y=2000, radius=0)
    unproven_result = unproven.segment()
    assert next(run for run in unproven_result.runs if (run.y0, run.y1) == (799, 859)).kind \
        == segment.RUN_TOO_LONG


def test_heart_separated_near_gutter_refuses_past_an_unrelated_card_edge():
    """Regression for the two-pass split in `segment_frame`.

    Layout, top to bottom: card A (with its heart), a 61px candidate near-gutter gap, card B
    (heartless, flat top, a genuinely ROUNDED bottom corner), a 65px gap that only that rounded
    corner explains, card C (with its own heart), then an ordinary canonical gutter.

    Card B's rounded bottom proves ITS OWN gap is a real card boundary (RUN_CARD_EDGE) -- and
    that boundary sits between the 61px candidate gap and card C's heart. `card_edge_run`'s own
    assertion below is the load-bearing sanity check: without a real intervening boundary this
    test would not distinguish the bug from the fix at all. Card D exists only so the 53px
    canonical gutter below card C is a genuine enclosed run (RUN_GUTTER) rather than an
    open-ended one that runs off the band's own bottom edge (RUN_CLIPPED, never a `boundary_kind`
    either way) -- without it `following` finds nothing at all regardless of the bug.

    Before the two-pass split, `_heart_separated_near_gutter`'s forward `following` search ran
    inside the SAME loop that assigns RUN_CARD_EDGE, so at the moment the 61px gap was tested,
    the run between card B and card C still read as its pre-reclassification RUN_TOO_LONG and
    was invisible to `following`. The search then skipped straight past the real boundary to the
    canonical gutter below card C, treated card C's heart as proof, and split the 61px gap --
    even though the heartless card B in between never exposed a heart of its own. Reverting the
    segment_frame split (recombining the two loops into one and dropping the pass-1/pass-2
    comments) reproduces exactly that: the assertions below turn red because the 61px gap is
    reclassified to RUN_HEART_SEPARATED_NEAR_GUTTER and _extents(r) splits card A from card B.
    """
    f = _Frame()
    f.card(_BAND0, 700, heart_y=610, radius=0)          # card A: heart 90px above the 61px gap
    f.card(761, 900, radius=0)                          # card B: heartless, flat top
    f.round_bottom_corners(761, 900, radius=_CORNER_RADIUS_PX)  # ...but a real rounded bottom
    f.card(965, 1500, heart_y=1411, radius=0)            # card C: its own heart, square shape
    # 1500..1553 is left as untouched page background: an exact 53px canonical gutter.
    f.card(1553, 2000, radius=0)                        # card D: closes the gutter run; unused

    r = f.segment()

    assert r.ok, r.failures
    near_gap = next(run for run in r.runs if (run.y0, run.y1) == (700, 761))
    card_edge_run = next(run for run in r.runs if (run.y0, run.y1) == (900, 965))
    # Sanity check first: card B's rounded bottom really is an independently confirmed boundary,
    # and it really does sit between the candidate gap and card C's heart -- otherwise this test
    # would not be exercising the bug at all.
    assert card_edge_run.kind == segment.RUN_CARD_EDGE
    # The bug: the candidate gap must NOT be promoted just because a heart exists somewhere
    # further down the profile, past a real boundary that has no heart of its own on either side.
    assert near_gap.kind == segment.RUN_TOO_LONG
    # And card A must therefore stay merged with card B rather than being cut at the 61px gap.
    assert (700, 761) not in _gutters(r)
    assert (_BAND0, 761) not in _extents(r) and (_BAND0, 700) not in _extents(r)
    assert any(y0 == _BAND0 and y1 >= 900 for y0, y1 in _extents(r))


def test_a_rounded_card_bottom_proves_an_overlong_page_gap_is_a_boundary():
    """Regression for Laura frame 5: a 65px page gap followed a rounded heartless details card,
    while the pale photo below did not expose a readable top corner or heart yet.

    The upper corner proves that card ended, so the regions must be split. It does *not* prove
    where the lower card starts: that edge remains unobserved and therefore cannot become a
    crop or numbered item from this frame alone.
    """
    f = _Frame()
    f.card(500, 1200)                                  # rounded, heartless details card
    f.card(1265, 2300, radius=0)                       # 65px gap; square/pale-media shape

    r = f.segment()

    assert r.ok, r.failures
    assert _extents(r) == [(500, 1200), (1265, _BAND1)]
    gap = next(run for run in r.runs if (run.y0, run.y1) == (1200, 1265))
    assert gap.kind == segment.RUN_CARD_EDGE
    assert r.blocks[0].kind == segment.BLOCK_CONTEXT
    assert r.blocks[0].bottom.kind == segment.EDGE_CARD_CORNER
    assert r.blocks[0].bottom.observed
    assert r.blocks[1].kind == segment.BLOCK_PARTIAL
    assert r.blocks[1].top.kind == segment.EDGE_BACKGROUND_RUN
    assert not r.blocks[1].top.observed


def test_an_overlong_page_gap_without_a_corner_or_heart_stays_unsplit():
    """The new symmetry is corner evidence, not a generic widening of the gutter window."""
    f = _Frame()
    f.card(500, 1200, radius=0)
    f.card(1265, 2300, radius=0)

    r = f.segment()

    gap = next(run for run in r.runs if (run.y0, run.y1) == (1200, 1265))
    assert gap.kind == segment.RUN_TOO_LONG
    assert _extents(r) == [(500, _BAND1)]


def test_tall_media_gap_without_a_heart_at_the_known_bottom_inset_stays_unsplit():
    """A generic 101px blank span remains conservative; it is not a relaxed gutter window."""
    f = _Frame()
    f.card(200, 700, heart_y=560, radius=0)             # 140px from the gap, outside the guard
    f.card(801, 1400, heart_y=1311, radius=0)
    f.card(1453, _BAND1 + 100, heart_y=2000, radius=0)

    r = f.segment()

    assert not r.ok
    assert any("2 like hearts inside one block" in failure for failure in r.failures)
    tall_gap = next(run for run in r.runs if (run.y0, run.y1) == (700, 801))
    assert tall_gap.kind == segment.RUN_TOO_LONG


def test_strict_gutter_prefix_recovers_low_contrast_leading_card_surface():
    """A true gutter may be followed by a card-white media top that misses the span test.

    The 53 rows of strict page background, the adjacent non-page residual, the following
    textured card, and the first card's measured heart inset are all necessary.  The recovered
    card must start directly after the gutter, not at its first dark/textured row.
    """
    f = _Frame()
    f.card(200, 700, heart_y=611, radius=0)
    f.card(753, 1500, heart_y=1411, radius=0)
    f.fill(753, 874, _CARD_X0, _CARD_X1,
           np.minimum(f.col[753:874].astype(np.int16) + 4, 255).astype(np.uint8)[:, None])
    f.card(1553, _BAND1 + 100, heart_y=2000, radius=0)

    r = f.segment()

    assert r.ok, r.failures
    assert (700, 753) in _gutters(r)
    assert _extents(r) == [(_BAND0, 700), (753, 1500), (1553, _BAND1)]
    assert r.blocks[1].kind == segment.BLOCK_SELECTABLE
    assert r.blocks[1].top.kind == segment.EDGE_GUTTER


def test_strict_gutter_suffix_recovers_low_contrast_trailing_card_surface():
    """The mirror case keeps a heart in its card rather than inside a 174px cut."""
    f = _Frame()
    f.card(200, 1200, heart_y=1111, radius=0)
    f.fill(1079, 1200, _CARD_X0, _CARD_X1,
           np.minimum(f.col[1079:1200].astype(np.int16) + 4, 255).astype(np.uint8)[:, None])
    f.heart(1111)
    f.card(1253, _BAND1 + 100, heart_y=2000, radius=0)

    r = f.segment()

    assert r.ok, r.failures
    assert (1200, 1253) in _gutters(r)
    assert _extents(r) == [(_BAND0, 1200), (1253, _BAND1)]
    assert r.blocks[0].hearts == ((_HEART_CX, 1111),)
    assert r.blocks[0].bottom.kind == segment.EDGE_GUTTER


def test_strict_gutter_rescue_requires_the_measured_heart_bottom_inset():
    """The same prefix shape without its anchoring heart stays a loud merged-card failure."""
    f = _Frame()
    f.card(200, 700, heart_y=540, radius=0)             # 160px above the gutter, not 60..120
    f.card(753, 1500, heart_y=1411, radius=0)
    f.fill(753, 874, _CARD_X0, _CARD_X1,
           np.minimum(f.col[753:874].astype(np.int16) + 4, 255).astype(np.uint8)[:, None])
    f.card(1553, _BAND1 + 100, heart_y=2000, radius=0)

    r = f.segment()

    assert not r.ok
    assert (700, 753) not in _gutters(r)
    assert any("2 like hearts inside one block" in failure for failure in r.failures)


def test_strict_page_span_inside_a_long_run_is_not_a_low_contrast_card_rescue():
    """Only an edge-aligned strict gutter can licence the rescue; an interior one cannot."""
    f = _Frame()
    f.card(200, 700, heart_y=611, radius=0)
    f.card(700, 1500, heart_y=1411, radius=0)
    low = np.minimum(f.col[700:720].astype(np.int16) + 4, 255).astype(np.uint8)
    f.fill(700, 720, _CARD_X0, _CARD_X1, low[:, None])
    f.page(720, 773, _CARD_X0, _CARD_X1)                # strict 53px page span, but internal
    low = np.minimum(f.col[773:894].astype(np.int16) + 4, 255).astype(np.uint8)
    f.fill(773, 894, _CARD_X0, _CARD_X1, low[:, None])

    r = f.segment()

    assert not r.ok
    internal = next(run for run in r.runs if (run.y0, run.y1) == (700, 894))
    assert internal.kind == segment.RUN_TOO_LONG
    assert (720, 773) not in _gutters(r)


def test_allison_low_contrast_gutter_frames_replay_when_available():
    """Regression replay of the saved diagnostics; CI may not carry private screenshots."""
    root = Path(__file__).resolve().parents[1]
    frames = [root / "data/hinge_debug/5d79a2d45bfa"
              / f"{number:05d}_enumeration_segmentation_fallback_before.png"
              for number in (164, 165, 166, 167)]
    if not all(frame.exists() for frame in frames):
        pytest.skip("the Allison diagnostic screenshots are not on this machine")

    template = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
    assert template is not None
    for frame in frames:
        result = segment.segment_frame(
            frame.read_bytes(), content_band=_CONTENT_BAND, like_template=template,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD)
        assert result.ok, (frame.name, result.failures)
        assert result.unassigned_hearts == ()


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

def test_confirmed_scroll_top_recovers_a_square_first_photo_with_one_pale_corner():
    """Regression: header chrome must not swallow a real first photo and shift every number.

    The ordinary-path baseline CHANGED on 2026-08-28 and this test used to pin the old one. It
    asserted the chrome and the photo came back merged as one `(368, 1491)` block — which is the
    swallowing this test is named after, tolerated then because only the opt-in below could undo
    it. Hinge 10.1.0 made that merge unaffordable: it pins the header to the SCREEN, so the
    merged block's top is a fabricated page row that lands inside the card above and refuses the
    whole index (`_unanchored_leading_island_rows`). segment.py now splits the strip off on the
    ordinary path too, so the photo is no longer swallowed.

    The opt-in still does real work, and that is what the second half pins: without it the
    photo's top is merely UNOBSERVED, so the card is `BLOCK_PARTIAL` and cannot be cropped,
    counted or verified. The recovery is what turns that into a bounded, selectable item 1.
    """
    f = _low_contrast_scroll_top()
    ordinary = f.segment()
    assert (368, 1491) not in _extents(ordinary)
    assert (517, 1491) in _extents(ordinary)
    assert ordinary.blocks[0].kind == segment.BLOCK_UNANCHORED
    unrecovered_photo = next(block for block in ordinary.blocks
                             if (block.y0, block.y1) == (517, 1491))
    assert unrecovered_photo.kind == segment.BLOCK_PARTIAL
    assert not unrecovered_photo.complete
    assert unrecovered_photo.top.kind == segment.EDGE_UNANCHORED_ISLAND
    assert not unrecovered_photo.top.observed

    recovered = f.segment(recover_leading_low_contrast_media=True)

    assert recovered.ok, recovered.failures
    assert (517, 1491) in _extents(recovered)
    first_photo = next(block for block in recovered.blocks
                       if (block.y0, block.y1) == (517, 1491))
    assert first_photo.kind == segment.BLOCK_SELECTABLE and first_photo.complete
    assert first_photo.top.kind == segment.EDGE_SCROLL_TOP_MEDIA
    assert first_photo.heart == (_HEART_CX, 1402)
    assert (411, 517) in [
        (run.y0, run.y1) for run in recovered.runs
        if run.kind == segment.RUN_SCROLL_TOP_MEDIA]

    # Production does not call the opt-in directly: the item indexer grants it only to frame 0
    # when its caller has already confirmed scroll-top. Pin that wiring and the recovered heart
    # ordinal, which is what prevents every later photo number shifting down by one.
    index = item_index.build_item_index(
        [f.png()], content_band=_CONTENT_BAND, like_template=f.template,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
        identity_band=None)
    assert index.usable, index.failures
    assert index.translation == (1,)
    assert index.selectable[0].page_y0 == 517


@pytest.mark.parametrize("change", ["non_square", "missing_heart", "missing_gutter"])
def test_scroll_top_media_recovery_refuses_when_any_independent_guard_is_missing(change):
    """The opt-in path is a conjunction, never a generic licence to split long blank spans."""
    if change == "non_square":
        frame = _low_contrast_scroll_top(card_y1=1450, heart_y=1361)
    elif change == "missing_heart":
        frame = _low_contrast_scroll_top(heart_y=None)
    else:
        frame = _low_contrast_scroll_top(bottom_gutter=False)

    result = frame.segment(recover_leading_low_contrast_media=True)

    assert not any(run.kind == segment.RUN_SCROLL_TOP_MEDIA for run in result.runs)
    assert not any(block.top.kind == segment.EDGE_SCROLL_TOP_MEDIA for block in result.blocks)


def test_low_contrast_recovery_is_never_enabled_for_an_ordinary_frame():
    """The same pixels without the caller's confirmed-top authority stay conservatively unbounded.

    The leading strip is `BLOCK_UNANCHORED` rather than `BLOCK_PARTIAL` since 2026-08-28: both
    say "this frame cannot bound it", and the newer one additionally says WHY — a strip with page
    background on both sides may be page content or screen-pinned chrome, and one frame cannot
    tell. What matters here is unchanged: with no confirmed top, nothing below is upgraded to a
    bounded, selectable item.
    """
    result = _low_contrast_scroll_top().segment()
    assert not any(run.kind == segment.RUN_SCROLL_TOP_MEDIA for run in result.runs)
    assert result.blocks[0].kind == segment.BLOCK_UNANCHORED
    assert not any(block.top.kind == segment.EDGE_SCROLL_TOP_MEDIA for block in result.blocks)
    assert not any(block.complete for block in result.blocks
                   if (block.y0, block.y1) == (517, 1491))


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
        # Square bottom so this test isolates the candidate corner BELOW the long gap. A
        # valid rounded upper bottom now independently proves the gap is a boundary.
        f.card(200, 700, heart_y=640, radius=0)           # sliced by the band's top edge
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
    # Square bottom so only the out-of-range candidate BELOW the long gap is under test.
    f.card(200, 700, heart_y=640, radius=0)
    f.card(1000, 1700, heart_y=1640, radius=40)
    r = f.segment()

    assert 40 > max(segment._CARD_CORNER_PX), "fixture must sit outside the accepted window"
    assert _card_edge_runs(r) == []
    assert _extents(r) == [(_BAND0, 1700)]               # merged, because nothing cut
    assert not r.ok                                      # ...and loudly: two hearts, one block
    assert r.blocks[0].kind == segment.BLOCK_AMBIGUOUS


# =====================================================================================
# Hinge 10.1.0 pins the per-profile header TO THE SCREEN, inside the analysed band
#
# [measured on the incident capture `data/hinge_debug/8fb11094ef4d`, 2026-08-28] The header —
# filter chips, the name, a verified badge, a back arrow, an overflow menu and a
# pronoun/activity sub-row — stopped scrolling with the content. Frame rows 300..516 are
# pixel-identical (max absolute difference 0) across all six evidence frames of that run, while
# rows 517+ differ by a mean of 59..143 grey levels. The band therefore opens on 68 rows of page
# background, then a 43px header strip at frame rows 368..410, then 106 more rows of page
# background (411..516), then the scrolling content, clipped at row 517.
#
# 106px is not gutter-length, so every ordinary rule absorbed that run and merged the strip into
# the clipped card below — one block whose reported top is 149px above any real content. A
# block's page position is `frame_row + scroll_offset`, which is fabricated for a strip that
# never moved: on 2026-08-28 that fabricated top landed inside the card ABOVE, bridged two cards
# into one fold group and hard-refused a whole item index.
#
# segment.py answers with `BLOCK_UNANCHORED` — "this frame cannot place this strip" — and
# declines to say whether it is chrome, because chrome-ness is a CROSS-FRAME property. These
# tests pin the split, the four things that must NOT split, and the precedence between the split
# and the card-corner rescue that already existed.
# =====================================================================================

_STRIP_Y0, _STRIP_Y1 = 368, 411          # the pinned header strip: frame rows 368..410, 43px
_CONTENT_CLIP_Y = 517                    # first row of scrolling content; the run above is 106px
_HEADER_WIDEST_SPAN_PX = 968             # of the card's 974 — the whole rule's 6px of margin
_CARD_WIDTH_PX = _CARD_X1 - _CARD_X0     # 974


def _widest_span(f, y0, y1):
    """The widest non-background span over rows [y0, y1), measured the way segment.py measures.

    A fixture-side re-derivation on purpose: these tests turn on 968 versus 974 out of 974, six
    pixels, and a fixture that merely INTENDED a span would be the third instance of this repo's
    standing hazard — a test that never reaches the branch it is named for.
    """
    band = f.gray[y0:y1, _CARD_X0:_CARD_X1].astype(np.int32)
    nonbg = np.abs(band - f.col[y0:y1].astype(np.int32)[:, None]) > segment._BACKGROUND_TOLERANCE
    widest = 0
    for row in nonbg:
        hit = np.flatnonzero(row)
        widest = max(widest, int(hit[-1] - hit[0]) + 1 if hit.size else 0)
    return widest


def _pinned_header_frame(*, strip_span=_HEADER_WIDEST_SPAN_PX, card_radius=0, card_y1=1491,
                         heart_y=1402, strip_heart_y=None, run_heart_y=None):
    """The measured 10.1.0 mid-scroll shape: background, pinned strip, background, clipped card.

    The strip is drawn `strip_span` px wide and CENTRED in the card band, so the fixture's own
    span is the parameter under test rather than an accident of where the chrome glyphs landed.
    The card below defaults to `radius=0`: mid-scroll its top corner is off-screen above, so its
    first visible row is already full width — which is why the corner rescue cannot bound it and
    why the 106px run was absorbed rather than cutting.
    """
    f = _Frame()
    x0 = _CARD_X0 + (_CARD_WIDTH_PX - strip_span) // 2
    f.fill(_STRIP_Y0, _STRIP_Y1, x0, x0 + strip_span, 40)
    f.card(_CONTENT_CLIP_Y, card_y1, heart_y=heart_y, radius=card_radius)
    if strip_heart_y is not None:
        f.heart(strip_heart_y)
    if run_heart_y is not None:
        f.heart(run_heart_y)
    return f


def test_a_screen_pinned_header_strip_is_split_off_and_reported_as_unplaceable():
    """THE 2026-08-28 regression. The pinned strip must not merge into the card below it.

    Pins the whole shape of the answer, because each part of it is load-bearing somewhere else:
    the strip comes back as its own block at exactly its own rows (368..411) so none of the
    149px above the content clip line is ever attributed to real content; BOTH of its edges are UNOBSERVED,
    because the 106px run is the reason the two were separated and is evidence that neither of
    them ENDED; and it carries a `content_digest`, which is the only thing a cross-frame caller
    can use to settle what one frame cannot — whether these rows are pinned to the screen or are
    page content the band happened to slice this way.

    The card below must start at the content clip line 517, not at 368. If it starts at 368 the
    index layer places its top 149px too high, which in page space lands inside the card above,
    welds two cards into one fold group and refuses the profile — the measured incident.
    """
    r = _pinned_header_frame().segment()

    assert r.ok, r.failures
    assert _extents(r) == [(_STRIP_Y0, _STRIP_Y1), (_CONTENT_CLIP_Y, 1491)]
    assert _kinds(r) == [segment.BLOCK_UNANCHORED, segment.BLOCK_PARTIAL]

    strip = r.blocks[0]
    assert (strip.y0, strip.y1) == (368, 411) and strip.height == 43
    assert not strip.top.observed
    assert strip.bottom.kind == segment.EDGE_UNANCHORED_ISLAND
    assert not strip.bottom.observed
    assert strip.bottom.run_px == 106                    # 411..517, the measured run
    assert strip.content_digest is not None
    assert f"{_HEADER_WIDEST_SPAN_PX}px of {_CARD_WIDTH_PX}px" in strip.reason

    # The cut itself is a new CATEGORY of run, not a fifth flavour of gutter: it separates
    # without bounding, which is exactly what both unobserved edges above say.
    island_runs = [(run.y0, run.y1) for run in r.runs
                   if run.kind == segment.RUN_UNANCHORED_ISLAND]
    assert island_runs == [(411, _CONTENT_CLIP_Y)]
    assert (411, _CONTENT_CLIP_Y) not in _gutters(r)
    assert (411, _CONTENT_CLIP_Y) not in _card_edge_runs(r)

    card = r.blocks[1]
    assert card.y0 == _CONTENT_CLIP_Y, "the card must start at real content, not at the strip"
    assert card.top.kind == segment.EDGE_UNANCHORED_ISLAND and not card.top.observed
    assert not card.complete                             # unplaceable above => not croppable
    assert card.hearts == ((_HEART_CX, 1402),)

    # The digest is the UNANCHORED block's alone. Anywhere else it would be an invitation to
    # compare block content across frames on a path whose decode equality is not guaranteed.
    assert [b.content_digest for b in r.blocks if b.kind != segment.BLOCK_UNANCHORED] == [None]


def test_a_leading_strip_reaching_the_full_card_width_is_a_card_slice_and_never_splits():
    """Clause (c), the load-bearing per-frame discriminator, with its measured 6px of margin.

    Only the card itself reaches x=53..1026 — anything drawn INSIDE a card is inset by the
    card's own padding — so "no row reaches the full card width" is what separates pinned chrome
    from a band-sliced piece of a real card. [measured on 15 frames of 3 profiles, one app
    version: the pinned header's widest row spans 968 of 974 card px on the incident profile and
    961 on two others; a real card slice reaches exactly 974 on 94.9% of its rows.]

    Six pixels is the weakest number in the rule, so both halves are measured off the fixture's
    own pixels rather than intended, and the split half asserts the span segment.py REPORTED —
    if Hinge ever pushes the back arrow out to the card edge, that number in a bug report is the
    warning that the clause has silently disarmed.

    If this test loses, a card sliced by the band's top edge is torn off its own top rows and
    handed to the caller as unplaceable chrome, which deletes a real item from the index.
    """
    narrow = _pinned_header_frame(strip_span=_HEADER_WIDEST_SPAN_PX)
    full = _pinned_header_frame(strip_span=_CARD_WIDTH_PX)
    assert _widest_span(narrow, _STRIP_Y0, _STRIP_Y1) == 968
    assert _widest_span(full, _STRIP_Y0, _STRIP_Y1) == 974 == _CARD_WIDTH_PX

    split = narrow.segment()
    assert split.blocks[0].kind == segment.BLOCK_UNANCHORED
    assert "968px of 974px" in split.blocks[0].reason

    merged = full.segment()
    assert merged.ok, merged.failures
    assert _extents(merged) == [(_STRIP_Y0, 1491)], (
        "a full-width slice must come back exactly as it did before the 10.1.0 fix: one block")
    assert _kinds(merged) == [segment.BLOCK_PARTIAL]
    assert merged.blocks[0].content_digest is None
    assert merged.blocks[0].top.kind == segment.EDGE_BACKGROUND_RUN
    assert merged.blocks[0].bottom.kind == segment.EDGE_BACKGROUND_RUN
    assert not any(run.kind == segment.RUN_UNANCHORED_ISLAND for run in merged.runs)
    assert [run.kind for run in merged.runs if run.y0 == 411] == [segment.RUN_TOO_LONG]


def test_a_leading_card_with_its_own_corners_is_a_card_and_never_an_unplaceable_strip():
    """Clauses (c) and (d): a whole small card at the top of the band keeps its identity.

    The shape is the one clause (c) alone would admit if a card ever presented rows narrower
    than the full width — a leading island that is really a card, readable rounded corners at
    both ends. It must stay an ordinary, fully-bounded block: its top corner is its own evidence
    (`EDGE_CARD_CORNER`, 22px, the measured 18..25 window) and a canonical 53px gutter bounds its
    bottom, so the index may count and crop it. Calling it unplaceable would withhold a real
    Hinge item from the item table.

    MUTATION RECORD (2026-08-28). Removing clause (d) alone leaves this test GREEN, and so does
    removing clause (c) alone; only removing BOTH turns it red. That is not slack in the test,
    it is the geometry: `_corner_radius` reports a corner only when the arc reaches the FULL card
    width, so any island with a readable corner necessarily contains a full-width row and clause
    (c) has already refused it. Clause (d) is therefore unreachable while (c) stands, and is
    exactly what its docstring calls it — the independent second gate, the thing that still says
    no if (c) is ever loosened. Do not "simplify" it away on the grounds that it never fires.

    The leading card is deliberately HEARTLESS (a `BLOCK_CONTEXT`, complete on both edges) so
    that clause (e) is not silently doing the work here. Give it a heart and every one of the
    mutations above stays green, because a heart inside the candidate rows refuses the split on
    its own — and the test would then prove nothing about corners at all.
    """
    f = _Frame()
    f.card(_STRIP_Y0, 560, radius=_CORNER_RADIUS_PX)     # a whole 192px card, both corners on
    f.card(560 + _GUTTER, 1587, heart_y=1498, radius=_CORNER_RADIUS_PX)
    r = f.segment()

    # The control: the band opens on page background, so clause (a)'s precondition HOLDS and the
    # detector really was consulted about this frame. Without this the test could pass for the
    # same uninteresting reason test 4 below passes.
    assert r.runs[0].kind == segment.RUN_CLIPPED and r.runs[0].y0 == _BAND0

    assert r.ok, r.failures
    unplaceable = [(block.y0, block.y1) for block in r.blocks
                   if block.kind == segment.BLOCK_UNANCHORED]
    assert unplaceable == [], f"a whole card was reported as unplaceable chrome: {unplaceable}"
    assert not any(run.kind == segment.RUN_UNANCHORED_ISLAND for run in r.runs)

    leading = r.blocks[0]
    assert (leading.y0, leading.y1) == (_STRIP_Y0, 560)
    assert leading.kind == segment.BLOCK_CONTEXT and leading.complete
    assert leading.top.kind == segment.EDGE_CARD_CORNER and leading.top.observed
    assert leading.top.corner_px == _CORNER_RADIUS_PX
    assert leading.bottom.kind == segment.EDGE_GUTTER and leading.bottom.observed
    assert leading.content_digest is None


def test_a_page_coloured_span_inside_a_band_clipped_card_never_becomes_an_unplaceable_strip():
    """Clause (a): the rule may only ever look at a band that OPENS in page background.

    This is doc 5.4 amendment one's frame — 11.5% of rows sampled inside confirmed cards match
    the page background, and one measured span ran 192 contiguous rows — carrying the one extra
    property the new rule cares about: the card's top is clipped by the band, so the band opens
    on CARD rows. [corpus: 140 of the 157 frames saved in the incident run open that way and are
    never considered at all.] The card must stay whole. Splitting it here would renumber every
    item below it, and hand the caller a fabricated "unplaceable" strip made of real profile
    content that it must then withhold.

    The fixture is built so clause (a) is the ONLY thing standing between this frame and a
    split, and the three controls below prove that rather than assume it: the interior rows are
    ragged-right text (944px of 974, so clause (c) would pass), the second blank span is 106px
    and page-coloured (so a mutant has a `RUN_TOO_LONG` to cut on), and no heart lies in the
    candidate rows (so clause (e) would pass). A fixture rejected by three clauses at once would
    prove nothing about the one it is named for.
    """
    f = _Frame()
    f.card(200, 1800, heart_y=1740)                      # top clipped by the band at 300
    f.page(200, 1800, _CARD_X1 - 30, _CARD_X1)           # ragged-right text: never full width
    f.page(700, 892, _CARD_X0, _CARD_X1)                 # the measured 192-row blank interior
    f.page(1000, 1106, _CARD_X0, _CARD_X1)               # a second blank span, 106px
    r = f.segment()

    assert r.ok, r.failures
    assert _extents(r) == [(300, 1800)], "the card must stay whole"
    assert _kinds(r) == [segment.BLOCK_PARTIAL]
    assert not any(run.kind == segment.RUN_UNANCHORED_ISLAND for run in r.runs)
    assert r.blocks[0].content_digest is None

    # The controls, in the order of the clauses they stand for. The first is the precondition
    # under test; the other three are the clauses that must NOT be the reason this frame is safe.
    assert r.runs[0].y0 == 700 and r.runs[0].kind == segment.RUN_TOO_LONG, (
        "the band must open on CARD rows, or clause (a) is not what rejected this frame")
    assert [run.kind for run in r.runs if run.y0 == 1000] == [segment.RUN_TOO_LONG], (
        "the second blank span must be a cuttable RUN_TOO_LONG, or a mutant has nothing to cut")
    assert _widest_span(f, 892, 1000) == 944 < _CARD_WIDTH_PX      # clause (c) would pass
    assert not any(892 <= y < 1106 for _, y in r.hearts)           # clause (e) would pass


@pytest.mark.parametrize("place, merged_extent", [
    (dict(strip_heart_y=389), (345, 1491)),
    (dict(run_heart_y=460), (_STRIP_Y0, 1491)),
])
def test_a_heart_in_the_leading_strip_or_its_run_declines_the_split(place, merged_extent):
    """Clause (e): a like heart anywhere in the candidate means this rule does not fire.

    A heart is a likeable Hinge item, and this rule would rather report the old merged block
    than risk touching one. [corpus: genuine card hearts measured at y 570..1890 over 115 real
    frames, so on real pixels this never fires — it is a fail-closed guard, not a case.]

    It also forecloses a failure mode the split would CREATE. The cut consumes the whole 106px
    run, so a heart inside that run belongs to no segment afterwards and comes back as an
    `unassigned_hearts` hard refusal: "a selectable item exists that segmentation cannot bound".
    Turning a frame we could read into a refusal is a worse answer than the merge.

    The two merged extents differ by 23 rows for a reason worth stating: the like glyph is 88px
    tall and the strip is 43, so a heart centred at 389 pushes the block's top out to 389-44=345
    so the crop would contain the glyph that made it selectable. Nothing about the strip moved.
    """
    r = _pinned_header_frame(heart_y=None, **place).segment()

    assert len(r.hearts) == 1
    assert _extents(r) == [merged_extent], "the split must have been declined"
    assert _kinds(r) == [segment.BLOCK_PARTIAL]
    assert not any(run.kind == segment.RUN_UNANCHORED_ISLAND for run in r.runs)
    assert r.blocks[0].content_digest is None

    assert r.unassigned_hearts == ()
    assert not [msg for msg in r.failures if "fell outside every block" in msg]
    assert r.ok, r.failures


def test_a_card_corner_below_the_run_still_outranks_the_island_cut():
    """ORDERING, and it is load-bearing rather than decorative.

    Same 43px strip and same 106px run as the incident frame, but the card below shows its own
    rounded corner — the scroll-top shape, where the corner rescue has bounded item 1 since the
    `list_top_y` declaration was removed. The island `elif` sits LAST in the cut chain precisely
    so it can never pre-empt that: a run that a card's own corner explains is a real boundary
    (`RUN_CARD_EDGE`), and item 1 keeps an OBSERVED `EDGE_CARD_CORNER` top, which is what makes
    it complete, countable and croppable. Let the island cut win instead and item 1's top
    silently degrades to unobserved on every scroll-top frame — the profile's first item stops
    being selectable and the index loses it.

    The strip above is still `BLOCK_UNANCHORED`, because the block label is keyed on the
    island's EXTENT and not on the cut. That is deliberate: these frames place a fabricated page
    position for a screen-pinned strip exactly like the merged case does, so both shapes have to
    reach the caller with the same warning. Note its bottom edge is an ordinary
    `EDGE_BACKGROUND_RUN` here — the run belongs to the card below it now.
    """
    r = _pinned_header_frame(card_radius=_CORNER_RADIUS_PX).segment()

    assert r.ok, r.failures
    assert _extents(r) == [(_STRIP_Y0, _STRIP_Y1), (_CONTENT_CLIP_Y, 1491)]
    assert _card_edge_runs(r) == [(411, _CONTENT_CLIP_Y)]
    assert not any(run.kind == segment.RUN_UNANCHORED_ISLAND for run in r.runs)

    card = r.blocks[1]
    assert card.top.kind == segment.EDGE_CARD_CORNER and card.top.observed
    assert card.top.corner_px == _CORNER_RADIUS_PX
    assert card.kind == segment.BLOCK_SELECTABLE and card.complete

    strip = r.blocks[0]
    assert strip.kind == segment.BLOCK_UNANCHORED and strip.content_digest is not None
    assert strip.bottom.kind == segment.EDGE_BACKGROUND_RUN and not strip.bottom.observed


def test_the_unanchored_content_digest_is_taken_from_the_strips_decoded_pixels():
    """`content_digest` is the cross-frame caller's ONLY evidence, so it must be content.

    A caller holding several frames of one scroll settles "is this strip pinned to the screen"
    by comparing this digest across them — [measured: frame rows 300..516 were pixel-identical
    across all six evidence frames of run 8fb11094ef4d, while rows 517+ differed by a mean of
    59..143 grey levels]. Two properties make that usable and both are pinned here: the same
    frame bytes always yield the same digest, and a strip whose PIXELS differ yields a different
    one. A digest of the extent, of the block kind, or of anything else the two frames share
    would compare equal on every pair and answer "pinned" for everything.
    """
    first = _pinned_header_frame().segment()
    second = _pinned_header_frame().segment()
    assert first.frame_digest == second.frame_digest
    assert first.blocks[0].content_digest == second.blocks[0].content_digest

    # One grey level, inside the strip, invisible to every geometric test: same rows, same
    # spans, same blocks, same runs — and a different digest.
    nudged = _pinned_header_frame()
    nudged.gray[_STRIP_Y0:_STRIP_Y1, 500:600] += 1
    other = nudged.segment()
    assert _extents(other) == _extents(first) and _kinds(other) == _kinds(first)
    assert _widest_span(nudged, _STRIP_Y0, _STRIP_Y1) == _HEADER_WIDEST_SPAN_PX
    assert other.blocks[0].content_digest != first.blocks[0].content_digest


def test_the_content_digest_may_only_be_compared_along_this_one_decode_path():
    """The contract, not a value: compare digests only from THIS function on THIS decode path.

    `cv2.IMREAD_GRAYSCALE` greys an sRGB-tagged device PNG differently from a `cv2.imencode`
    round-trip of the same image. That is measured, not theoretical: on 2026-08-27 it made a
    correct Hinge like sheet score 10.283 against a 7.00 ceiling and cost this project a wrong
    measurement. It also cannot be reproduced here — cv2 writes no sRGB chunk, so no synthetic
    fixture can carry one, which is exactly why this test pins the CONTRACT instead of asserting
    that two paths agree or that they differ.

    What the fixture can show is the shape of the trap. The two frames below hold identical
    pixels in two different PNG encodings, so their FILE digests differ while their decoded rows
    do not. `content_digest` is therefore not the file's fingerprint and must never be compared
    against one, against a stored constant, or against a digest some other tool computed from
    the same screencap: the only guaranteed-comparable digests are the ones `segment_frame`
    itself produced from bytes it decoded.
    """
    direct = _pinned_header_frame().segment()
    png = _pinned_header_frame().png()
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    ok, buf = cv2.imencode(".png", decoded, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    reencoded = buf.tobytes()
    assert reencoded != png, (
        "fixture must really be two different files, or it stands in for nothing")

    other = segment.segment_frame(
        reencoded, content_band=_CONTENT_BAND,
        like_template=hinge._load_template(hinge.HINGE_SPEC.templates["like"]),
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)

    assert other.frame_digest != direct.frame_digest      # different files...
    assert _extents(other) == _extents(direct)            # ...identical pixels
    assert other.blocks[0].content_digest is not None
    assert direct.blocks[0].content_digest != direct.frame_digest, (
        "the content digest is the STRIP's pixels, never the frame's bytes — substituting one "
        "for the other is the mistake this field's docstring forbids")


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


def test_band_rows_agrees_with_the_driver_it_mirrors():
    """`segment._band_rows` is a deliberate byte-for-byte duplicate of `hinge._content_rows`
    (see `_band_rows`'s own docstring for why it is copied rather than imported: segment.py must
    stay importable BEFORE hinge.py, which is the importer). `bugreport._content_band_rows` is
    the repo's OTHER copy of this same arithmetic, and it is protected from drifting by
    tests/test_bugreport.py::test_content_band_rows_agrees_with_the_driver_it_mirrors -- this is
    the matching test for THIS copy, modelled on that one and run over the same band/size matrix,
    so a change to the clamp/rounding rule in one copy and not the other fails loudly here."""
    for band in ((0.125, 0.875), (0.0, 1.0), (0.3, 0.31), (0.9, 0.1)):
        for size in (2400, 1920, 2):
            assert segment._band_rows(band, size) == hinge._content_rows(band, size)


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
