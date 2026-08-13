"""The driver-owned item index (operation_love/drivers/item_index.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures that
segment.py's and frameshift.py's constants were measured against are real people's dating
profiles and are gitignored (ops/calibration/, .gitignore:26); only geometry and counts from them
appear anywhere in this repo. So the fixtures build a tall scrollable WORLD from first principles
— gradient page background, rounded-rect cards inset 53px per side, canonical 53px gutters, the
genuine shipped like glyph stamped bottom-right on the likeable ones and a heartless 215px vitals
block among them — and cut 1080x2400 windows out of it at offsets the test chooses.

That makes the ground truth exact and known BY CONSTRUCTION: the page layout is a table at the
top of this file, so the assertions can name the items, their page rows, their heart ordinals and
their model indices rather than approximating any of it.

Two mechanisms are tested at two levels on purpose, following the house pattern that
tests/test_frameshift.py uses for `_resolve`:

  * end to end, through `build_item_index`, where the shift estimator and the segmenter are real
    and the only input is pixels;
  * and directly on `_assemble`, which is pure arithmetic over `BlockObservation` records. That
    is where the numbering, the disagreement rules and the fabricated-item guard can be driven to
    the exact contradiction they exist to catch, with no dependence on what a correlation happens
    to do.

Every positive is paired with a negative plus a control that proves WHICH mechanism did the
excluding.
"""
import math
import subprocess
import sys

import cv2
import numpy as np
import pytest

from operation_love.drivers import frameshift, hinge, item_index, segment

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 13
_CONTENT_BAND = hinge.HINGE_SPEC.content_band          # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_BAND_H = _BAND1 - _BAND0
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - segment._CARD_MARGIN_PX                # 1027
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px"
_HEART_CX = 937                                        # bottom-right, as test_segment.py stamps it
_HEART_ABOVE_BOTTOM = 90                               # segment.py: the heart sits ~89px above a
                                                       # complete card's bottom edge
_PAGE_TOP, _PAGE_BOTTOM = 254, 243                     # doc 5.10's background gradient
_WORLD_H = 5400

# The step doc 5.10.1 measured at 100% frame-to-frame heart tracking with zero phantoms
# (read_scroll_frac 0.16 -> a rock-steady 363px/step).
_STEP = 363

# THE PAGE, in world rows. Five blocks: four likeable cards and, between the second and third,
# the heartless 215px vitals block doc 5.10 tracked across 7 consecutive frames. Heights are from
# doc 5.10's card table (nothing here exceeds the 1114px tallest card observed end to end).
_PAGE_TOP_GAP = 400                        # page background above card 1, where Hinge's header is
_LAYOUT = (("card", 900), ("card", 760), ("context", 215), ("card", 1000), ("card", 820))


def _layout_rows():
    """`[(kind, y0, y1), ...]` in world rows, gutter-separated exactly as the layout draws."""
    rows, y = [], _PAGE_TOP_GAP
    for kind, height in _LAYOUT:
        rows.append((kind, y, y + height))
        y += height + _GUTTER
    return rows


_ROWS = _layout_rows()
_CARD1, _CARD2, _VITALS, _CARD3, _CARD4 = _ROWS
assert _CARD4[2] + 400 < _WORLD_H, "the world must have page background below the last card"


def _page_column(height=_WORLD_H):
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, height).round().astype(np.uint8)


_TEMPLATE = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"


def _build_world(*, short_card2=0):
    """The tall page. `short_card2` shortens the second card by that many rows, moving its bottom
    edge (and its heart) up while leaving every other block exactly where it was — the ONE way to
    make two frames disagree about one card's height without also moving everything below it."""
    rng = np.random.default_rng(_SEED)
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    for kind, y0, y1 in _ROWS:
        if (kind, y0) == (_CARD2[0], _CARD2[1]):
            y1 -= short_card2
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
        if kind == "card":
            th, tw = _TEMPLATE.shape
            cy = y1 - _HEART_ABOVE_BOTTOM
            world[cy - th // 2: cy - th // 2 + th,
                  _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = _TEMPLATE
    return world


_WORLDS = {0: _build_world()}
# Static chrome outside the content band, textured but IDENTICAL on every frame — the real
# device's status bar, sticky header, floating buttons and bottom nav, none of which translate.
_CHROME_TOP = np.random.default_rng(3).integers(0, 60, size=(_BAND0, _W), dtype=np.uint8)
_CHROME_BOTTOM = np.random.default_rng(4).integers(0, 60, size=(_H - _BAND1, _W), dtype=np.uint8)
_FRAMES: dict[tuple[int, int], bytes] = {}


def _frame(scroll: int, *, short_card2: int = 0) -> bytes:
    """The 1080x2400 window of the world at `scroll`, as PNG. Content at world row `w` lands on
    frame row `w - scroll`, so `_frame(s)` then `_frame(s + d)` is a forward scroll of `d`."""
    key = (scroll, short_card2)
    if key not in _FRAMES:
        if short_card2 not in _WORLDS:
            _WORLDS[short_card2] = _build_world(short_card2=short_card2)
        gray = _WORLDS[short_card2][scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAMES[key] = buf.tobytes()
    return _FRAMES[key]


def _index(scrolls, *, at_scroll_top, **kw):
    return item_index.build_item_index(
        [_frame(s) for s in scrolls], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=at_scroll_top,
        identity_band=kw.pop("identity_band", None), **kw)


# Scroll offsets: a full read from the top of the profile to past its last card, at the cadence
# doc 5.10.1 validated. Frame 0 sits at world row 0 so its band opens on the page background
# above card 1; the last frame clears card 4's bottom by 334 rows.
_FULL_SCROLL = tuple(_STEP * i for i in range(8))
assert _FULL_SCROLL[-1] + _H <= _WORLD_H
_CACHE: dict[str, object] = {}


def _full():
    """The clean whole-profile index, built once — eight frames means seven real shift estimates
    and eight real segmentations, which is worth about four seconds and is not worth repeating."""
    if "full" not in _CACHE:
        _CACHE["full"] = _index(_FULL_SCROLL, at_scroll_top=True)
    return _CACHE["full"]


def _kinds(index):
    return [b.kind for b in index.blocks]


# =====================================================================================
# Observation-level helpers, for driving `_assemble` directly
# =====================================================================================

def _obs(frame_index, page_y0, page_y1, *, complete=True, hearts=(), kind=None):
    """One hand-written sighting. `hearts` are page rows; x is fixed because a list scroll has
    no horizontal component."""
    if kind is None:
        kind = (segment.BLOCK_SELECTABLE if hearts else segment.BLOCK_CONTEXT) if complete \
            else segment.BLOCK_PARTIAL
    return item_index.BlockObservation(
        frame_index=frame_index, page_y0=page_y0, page_y1=page_y1,
        frame_y0=page_y0, frame_y1=page_y1, kind=kind, complete=complete,
        top_observed=complete, bottom_observed=complete,
        hearts=tuple((_HEART_CX, y) for y in hearts))


def _assemble(observations, *, at_scroll_top=True, **kw):
    kw.setdefault("card_x", (_CARD_X0, _CARD_X1))
    kw.setdefault("extent_tolerance_px", item_index._EXTENT_TOLERANCE_PX)
    kw.setdefault("min_item_gap_px", item_index._MIN_ITEM_GAP_PX)
    kw.setdefault("band_y0", _BAND0)
    return item_index._assemble(list(observations), at_scroll_top=at_scroll_top, **kw)


# =====================================================================================
# The clean scroll: N items recovered exactly, once, with both index spaces
# =====================================================================================

def test_a_clean_scroll_recovers_exactly_the_items_that_were_planted():
    """THE headline. Eight frames, five physical blocks, each of them seen in three to five
    frames — and the index must contain five blocks at the world rows they were painted at, not
    the twenty-odd sightings they were seen as. Exact rows, not approximate: page coordinates are
    frame rows plus a measured shift, and the shift here is a slice offset."""
    index = _full()

    assert index.usable, index.failures
    assert len(index.blocks) == len(_ROWS)
    assert [(b.page_y0, b.page_y1) for b in index.blocks] == [(y0, y1) for _, y0, y1 in _ROWS]
    assert _kinds(index) == [item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_CONTEXT, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_SELECTABLE]

    # Every block really was seen several times, so the deduplication is doing work rather than
    # the capture happening to show each card once.
    assert all(len(b.frames) >= 3 for b in index.blocks), [b.frames for b in index.blocks]
    assert sum(len(b.observations) for b in index.blocks) > 3 * len(index.blocks)


def test_the_two_index_spaces_and_the_translation_between_them():
    """Doc 5.3: the model gets a dense 1..N over the selectable blocks, navigation counts hearts,
    and the driver keeps the private table between them. Here the vitals block sits third on the
    page and consumes NEITHER number, so model item 3 is the page's third heart but the page's
    FOURTH block."""
    index = _full()

    assert [b.model_index for b in index.blocks] == [1, 2, None, 3, 4]
    assert [b.heart_ordinal for b in index.blocks] == [1, 2, None, 3, 4]
    assert index.translation == (1, 2, 3, 4)
    assert index.heart_count == 4

    third = index.block_for(3)
    assert (third.page_y0, third.page_y1) == (_CARD3[1], _CARD3[2])
    assert index.blocks.index(third) == 3          # fourth block on the page, third model item
    assert index.heart_ordinal_for(3) == 3


def test_each_block_carries_the_frames_it_was_seen_in_and_a_frame_to_crop_it_from():
    """Requirement of doc 5.6's crop-and-verify path: a block is only croppable from a frame that
    saw ALL of it, and the index has to say which frame that is in that frame's own rows."""
    index = _full()
    card1 = index.block_for(1)

    assert card1.frames == tuple(range(len(card1.frames)))       # seen from frame 0 onward
    assert card1.complete and card1.croppable
    for sighting in card1.croppable:
        scroll = _FULL_SCROLL[sighting.frame_index]
        assert sighting.frame_y0 == _CARD1[1] - scroll
        assert sighting.frame_y1 == _CARD1[2] - scroll
        # ...and slicing those rows out of that very frame really is the card.
        frame = cv2.imdecode(np.frombuffer(_frame(scroll), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(frame[sighting.frame_y0:sighting.frame_y1, _CARD_X0:_CARD_X1],
                              _WORLDS[0][_CARD1[1]:_CARD1[2], _CARD_X0:_CARD_X1])

    # The negative: a sighting that did NOT see both edges is not offered as a crop source.
    assert any(not o.complete for o in card1.observations)
    assert all(o.complete for o in card1.croppable)


def test_the_heart_of_each_selectable_block_is_where_it_was_stamped():
    """The heart's page position is what a counting navigation eventually taps, so it is folded
    across frames like the extent is, and must come back at the world row it was painted at."""
    index = _full()
    for block, (_, _, y1) in zip(index.selectable, [r for r in _ROWS if r[0] == "card"]):
        assert block.heart is not None
        x, page_y = block.heart
        assert x == pytest.approx(_HEART_CX, abs=2)
        assert page_y == pytest.approx(y1 - _HEART_ABOVE_BOTTOM, abs=2)


def test_completeness_is_reported_and_a_short_capture_is_reported_as_truncated():
    """Requirement four: the index says what it does NOT establish. The full scroll starts at a
    confirmed top and clears the last card, so nothing is partial and nothing is truncated; the
    control is the same machinery on a prefix of the same frames, which is honest about both."""
    full, prefix = _full(), _index(_FULL_SCROLL[:3], at_scroll_top=True)

    assert full.at_scroll_top and full.reached_end and not full.truncated
    assert full.partial == () and full.complete
    assert full.tail_gap_px is not None and full.tail_gap_px > item_index._END_TAIL_GAP_PX

    assert prefix.usable, prefix.failures
    assert not prefix.reached_end and prefix.truncated and not prefix.complete
    assert prefix.partial, "the card the capture stopped inside is a partial block, not a gap"
    assert all(b.kind != item_index.ITEM_PARTIAL for b in prefix.selectable)


def test_the_same_profile_at_a_different_cadence_gives_the_same_items():
    """The invariant that says the item list belongs to the PROFILE and not to the capture. Two
    reads of the same page — one at doc 5.10.1's 363px step, one at double that, still inside
    frameshift's 900px trust window — must agree on every block, every extent and both index
    spaces. A capture-dependent answer is a fabricated or a dropped item by another name."""
    fine = _full()
    coarse = _index(tuple(2 * _STEP * i for i in range(5)), at_scroll_top=True)

    assert coarse.usable, coarse.failures
    assert len(coarse.frames) < len(fine.frames)          # genuinely fewer looks at the page
    assert [(b.page_y0, b.page_y1, b.kind, b.heart_ordinal, b.model_index) for b in coarse.blocks] \
        == [(b.page_y0, b.page_y1, b.kind, b.heart_ordinal, b.model_index) for b in fine.blocks]
    assert coarse.translation == fine.translation
    assert coarse.reached_end and not coarse.truncated

    # The blocks are also strictly ordered and non-overlapping in page space, which is what makes
    # "the k-th heart" a well-defined thing to count down to.
    rows = [(b.page_y0, b.page_y1) for b in coarse.blocks]
    assert rows == sorted(rows)
    assert all(a[1] < b[0] for a, b in zip(rows, rows[1:]))


def test_page_coordinates_come_from_the_measured_shift_and_not_from_an_assumed_step():
    """The offsets are the accumulated shift estimates. They must equal the scroll offsets the
    fixture cut the frames at — exactly — because everything the index says about position is
    downstream of them."""
    index = _full()

    assert index.offsets == _FULL_SCROLL
    assert [s.delta_px for s in index.shifts] == [_STEP] * (len(_FULL_SCROLL) - 1)
    assert index.page_span == (_BAND0, _FULL_SCROLL[-1] + _BAND1)


# =====================================================================================
# Fail loud: an unmeasurable shift must produce NO items, never an extra one
# =====================================================================================

def test_a_saturated_pair_makes_the_index_unusable_instead_of_fabricating_an_item():
    """THE regression this module exists for. Doc 5.10 measured a chained tracker recovering 9
    real items and fabricating a spurious 10th out of one large-jump tracking failure. Here the
    third frame is 1300px on from the second — past frameshift's 900px trust window — so the
    shift comes back as a refusal carrying a magnitude, and the index must contain NOTHING
    rather than a best guess at what moved."""
    index = _index((0, _STEP, _STEP + 1300), at_scroll_top=True)

    assert not index.usable
    assert index.blocks == ()
    assert index.selectable == () and index.translation == () and index.heart_count == 0

    # The refusal is the shift estimator's, quoted with the pair that produced it...
    assert index.shifts[-1].status == frameshift.SHIFT_BEYOND_WINDOW
    assert index.shifts[-1].delta_px is None and index.shifts[-1].consensus_px == 1300
    assert index.offsets == (0, _STEP, None)
    assert any("frames 1 and 2" in f and "beyond_window" in f for f in index.failures), \
        index.failures
    # ...and the measured magnitude is quoted as a magnitude, explicitly not as a shift.
    assert any("1300" in f and "may not be used as a shift" in f for f in index.failures)


def test_the_same_frames_index_cleanly_once_the_shift_is_measurable():
    """The control for the test above: nothing about those three frames is unreadable, the
    middle jump is only untrusted. Widen frameshift's window past it and the identical pixels
    produce the same five blocks — which proves the empty index came from the refusal and not
    from an inability to segment or to fold."""
    frames = (0, _STEP, _STEP + 1300)
    refused = _index(frames, at_scroll_top=True)
    allowed = _index(frames, at_scroll_top=True, trust_window_px=1500)

    assert not refused.usable and refused.blocks == ()
    assert allowed.usable, allowed.failures
    assert len(allowed.blocks) == len(_ROWS)
    assert [(b.page_y0, b.page_y1) for b in allowed.blocks[:4]] == \
        [(y0, y1) for _, y0, y1 in _ROWS[:4]]
    # Card 4 is the one this shorter capture stops inside, above its heart, so it is honestly
    # PARTIAL with no heart known — tolerated only because nothing heart-bearing is below it,
    # and reported through `partial` and `truncated` rather than through a failure.
    assert allowed.blocks[4].kind == item_index.ITEM_PARTIAL
    assert allowed.blocks[4].hearts == () and allowed.blocks[4].heart_ordinal is None
    assert allowed.translation == (1, 2, 3)
    assert allowed.heart_count == 3
    assert allowed.truncated and allowed.partial == (allowed.blocks[4],)


def test_unrelated_frames_are_refused_rather_than_indexed():
    """Low confidence, not saturation: two frames that share no content give the estimator
    nothing to agree on. Same outcome, different reason, and still not one item."""
    rng = np.random.default_rng(97)
    junk = cv2.imencode(".png", rng.integers(0, 255, size=(_H, _W), dtype=np.uint8))[1].tobytes()
    index = item_index.build_item_index(
        [_frame(0), junk], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable and index.blocks == ()
    assert index.offsets == (0, None)
    assert any("could not be put in one coordinate space" in f for f in index.failures)


def test_the_accessors_refuse_to_answer_from_an_unusable_index():
    """Doc 5.3: "treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate". A table that contradicts itself is a missing table, so the lookups raise rather
    than return None for a caller to interpret however it likes."""
    index = _index((0, _STEP, _STEP + 1300), at_scroll_top=True)

    with pytest.raises(item_index.ItemIndexError, match="unusable"):
        index.block_for(1)
    with pytest.raises(item_index.ItemIndexError, match="unusable"):
        index.heart_ordinal_for(1)

    # ...and on a perfectly good index, an index the model was never offered is still refused,
    # rather than clamped to the nearest item (doc 5.6: never substitute a different item).
    good = _full()
    assert good.block_for(4) is good.selectable[3]
    for bad in (0, -1, len(good.selectable) + 1):
        with pytest.raises(item_index.ItemIndexError, match="outside 1.."):
            good.block_for(bad)


def test_an_empty_capture_raises_rather_than_returning_an_empty_index():
    """"We captured nothing" and "this profile has no items" must never be the same value."""
    with pytest.raises(item_index.ItemIndexError, match="no frames"):
        item_index.build_item_index(
            [], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)


# =====================================================================================
# A context block occupies a page position without consuming a model index
# =====================================================================================

def test_a_context_block_takes_a_page_position_but_neither_number():
    """Doc 5.3's two tiers, at the level of the numbering. The vitals block is a real block with
    a real extent that the model may read, and it must not shift the model's 1..N — while the
    heart-bearing block below it keeps its place in BOTH sequences."""
    index = _full()
    vitals = index.blocks[2]

    assert vitals.kind == item_index.ITEM_CONTEXT
    assert (vitals.page_y0, vitals.page_y1) == (_VITALS[1], _VITALS[2])
    assert vitals.height == 215                    # doc 5.10's measured vitals block
    assert vitals.hearts == () and vitals.heart is None
    assert vitals.model_index is None and vitals.heart_ordinal is None
    assert vitals in index.context and vitals not in index.selectable

    # The block after it carries the next value of BOTH counters: a context block consumes
    # neither, so nothing below it is renumbered by its presence.
    assert index.blocks[3].model_index == 3 and index.blocks[3].heart_ordinal == 3


def test_a_heart_on_an_unbounded_block_still_consumes_a_heart_ordinal():
    """The other half of "index space belongs to the driver, selectability is policy", and the
    reason the translation table is not the identity function. A card the capture never bounded
    end to end cannot be cropped, so it is not offered to the model — but its heart is on the
    page, a counting navigation will tick it off, and dropping it from the ordinals would put
    every tap below it one card out.

    Built by starting the capture one step INTO the profile, so card 1's top corner is never
    seen, and stopping before card 4's bottom."""
    index = _index(_FULL_SCROLL[1:7], at_scroll_top=False)
    assert index.usable, index.failures

    assert _kinds(index) == [item_index.ITEM_PARTIAL, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_CONTEXT, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_PARTIAL]
    assert [b.heart_ordinal for b in index.blocks] == [1, 2, None, 3, 4]
    assert [b.model_index for b in index.blocks] == [None, 1, None, 2, None]

    # The table is therefore NOT the identity: the model's item 1 is the page's SECOND heart.
    # (Read off `translation`, which describes the capture. `heart_ordinal_for` refuses on this
    # index because at_scroll_top is False — see the test directly below.)
    assert index.translation == (2, 3)
    assert index.heart_count == 4                  # every heart counted, selectable or not
    assert len(index.selectable) == 2

    # ...and the two partial blocks are reported as the incompleteness they are.
    assert len(index.partial) == 2 and index.truncated and not index.complete


def test_heart_ordinal_for_refuses_a_relative_index_while_translation_still_describes_it():
    """Doc 5.3's addendum 2026-08-12, third blocker: with `at_scroll_top=False` the ordinals are
    RELATIVE — this very capture's model item 1 is really the profile's item 2 — and the accessor
    a counting navigation calls used to answer with a confident int anyway. It now raises.

    The negative that proves WHICH mechanism refuses: the same index is otherwise perfectly
    usable, `block_for` still answers (crops and the model's list are unaffected by where the
    capture started), and `translation` still describes the capture. Only the ordinal lookup —
    the number a tap is counted towards — is withheld."""
    relative = _index(_FULL_SCROLL[1:7], at_scroll_top=False)
    assert relative.usable and not relative.at_scroll_top

    with pytest.raises(item_index.ItemIndexError, match="at_scroll_top=False"):
        relative.heart_ordinal_for(1)

    # Everything else on the same index keeps working, so this is a targeted refusal rather
    # than the whole result being condemned.
    assert relative.block_for(1) is relative.selectable[0]
    assert relative.translation == (2, 3)

    # ...and the identical capture asserted at a confirmed top answers normally, which is what
    # makes the refusal about the assertion and not about this profile.
    absolute = _full()
    assert absolute.at_scroll_top and absolute.heart_ordinal_for(1) == 1


# =====================================================================================
# Disagreement is an error, never an average
# =====================================================================================

def test_two_frames_that_disagree_about_one_cards_height_report_it():
    """Requirement three, end to end. Both frames bound card 2 end to end — the first in the
    ordinary world, the second in one where that card is 120px shorter and nothing else moved —
    so two frames measure different heights at the same page position. Exactly one of them is
    right; the index must say so rather than pick or blend, and both heights must appear in the
    complaint."""
    short = 120
    frames = [_frame(_STEP), _frame(2 * _STEP, short_card2=short)]
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=False, identity_band=None)

    height = _CARD2[2] - _CARD2[1]
    assert index.shifts[0].delta_px == _STEP, index.shifts[0].reason
    assert not index.usable
    conflicts = [f for f in index.failures if "disagree about the block" in f]
    assert conflicts, index.failures
    assert f"{height}px against {height - short}px" in conflicts[0] \
        or f"{height - short}px against {height}px" in conflicts[0], conflicts[0]

    # The control: the same two scroll positions in one consistent world are clean, so the
    # failure came from the disagreement and not from the two-frame capture.
    assert _index((_STEP, 2 * _STEP), at_scroll_top=False).usable


def test_conflicting_extents_are_reported_at_the_observation_level_too():
    """The same rule driven directly, with no correlation involved: two sightings that both claim
    to have bounded the block at one page position, at heights 121px apart. The resolved extent
    must be one a frame actually SAW — never the mean, which is the number that would then be
    cropped and stored as the post-tap verification reference."""
    blocks, failures = _assemble([
        _obs(0, 700, 1674, hearts=(1584,)),        # 974px
        _obs(1, 700, 1553, hearts=(1584,)),        # 853px
    ])

    assert len(blocks) == 1                        # one page position, not two items
    assert any("disagree about the block" in f for f in failures), failures
    assert any("averaging them would produce an extent neither frame saw" in f for f in failures)
    assert (blocks[0].page_y0, blocks[0].page_y1) in [(700, 1674), (700, 1553)]
    assert blocks[0].height != (974 + 853) // 2

    # The control: the same two sightings agreeing to within the chain tolerance are ONE block
    # with no complaint at all, so it is the size of the disagreement doing the work.
    agreeing, ok = _assemble([_obs(0, 700, 1674, hearts=(1584,)),
                              _obs(1, 702, 1676, hearts=(1586,))])
    assert ok == () and len(agreeing) == 1
    assert agreeing[0].kind == item_index.ITEM_SELECTABLE


def test_two_hearts_at_one_page_position_are_a_failure_and_not_a_choice():
    """A block with two hearts is a statement that a gutter between two cards was missed. It must
    surface, it must not be resolved by taking the first heart, and the count must still advance
    by two so nothing below it is renumbered."""
    blocks, failures = _assemble([
        _obs(0, 700, 2700, hearts=(1600, 2610)),
        _obs(1, 2800, 3600, hearts=(3510,)),
    ])

    assert blocks[0].kind == item_index.ITEM_AMBIGUOUS
    assert len(blocks[0].hearts) == 2
    assert blocks[0].heart is None                 # never hearts[0]
    assert blocks[0].model_index is None
    assert any("ambiguous" in f for f in failures), failures

    # The block below is still the THIRD heart on the page, because both of the ambiguous
    # block's hearts occupied an ordinal.
    assert blocks[0].heart_ordinal == 1 and blocks[1].heart_ordinal == 3


def test_a_fragment_reaching_past_a_bounded_card_is_reported():
    """A sighting that overruns a card whose both edges were observed is a merged block or a
    mis-tracked frame — a fragment cannot be bigger than the thing it is a fragment of."""
    blocks, failures = _assemble([
        _obs(0, 700, 1674, hearts=(1584,)),
        _obs(1, 700, 2400, complete=False),
    ])

    assert len(blocks) == 1
    assert any("cannot reach past the card that contains it" in f for f in failures), failures


# =====================================================================================
# The fabricated-item guard
# =====================================================================================

def test_two_fragments_of_one_card_that_never_overlapped_are_reported_not_counted_twice():
    """The one way folding by overlap could still invent an item: a card whose sightings never
    share a row. Two blocks closer than the smallest gutter the layout draws contradict the
    layout, and counting them as two is exactly doc 5.10's phantom."""
    gap = 20
    blocks, failures = _assemble([_obs(0, 700, 1200, complete=False, hearts=(1150,)),
                                  _obs(1, 1200 + gap, 1700, complete=False, hearts=(1650,))])

    assert any(f"only {gap}px apart" in f for f in failures), failures
    assert any("would fabricate an item" in f for f in failures)

    # The control: push them a canonical gutter apart and they ARE two items, silently.
    _, ok = _assemble([_obs(0, 700, 1200, complete=False, hearts=(1150,)),
                       _obs(1, 1200 + _GUTTER, 1700, complete=False, hearts=(1650,))])
    assert ok == ()
    assert item_index._MIN_ITEM_GAP_PX == 47       # min(52,53) - 5, from segment.py's own window
    assert gap + item_index._EXTENT_TOLERANCE_PX < item_index._MIN_ITEM_GAP_PX <= _GUTTER


def test_the_same_card_seen_in_many_frames_is_one_block():
    """Deduplication by page position, driven directly: six overlapping sightings of one card,
    from six frames, are one block that names all six."""
    blocks, failures = _assemble(
        [_obs(i, 700 + 40 * i, 1674, complete=(i == 0), hearts=(1584,)) for i in range(6)])

    assert failures == () and len(blocks) == 1
    assert blocks[0].frames == (0, 1, 2, 3, 4, 5)
    assert (blocks[0].page_y0, blocks[0].page_y1) == (700, 1674)   # the one bounded sighting
    assert blocks[0].kind == item_index.ITEM_SELECTABLE
    assert len(blocks[0].hearts) == 1


# =====================================================================================
# A heart that may be hiding, and the one place it is tolerated
# =====================================================================================

def test_an_unbounded_heartless_block_above_a_heart_is_a_failure():
    """A heartless block nobody ever bounded may be hiding a heart in the rows that were never
    in band, and a hidden heart shifts every ordinal below it. Where there IS something below,
    that is a hard stop; the control is the same block with nothing heart-bearing under it,
    which costs only coverage and is reported through `partial` instead."""
    dangerous, failures = _assemble(
        [_obs(0, 300, 900, complete=False), _obs(1, 1000, 1900, hearts=(1810,))],
        at_scroll_top=False)
    assert any("a heart may sit in the rows that were never inside" in f for f in failures)

    harmless, ok = _assemble(
        [_obs(0, 300, 900, hearts=(810,)), _obs(1, 1000, 1900, complete=False)],
        at_scroll_top=False)
    assert ok == ()
    assert harmless[1].kind == item_index.ITEM_PARTIAL
    assert dangerous[0].kind == item_index.ITEM_PARTIAL


def test_a_confirmed_scroll_top_excuses_hinges_header_and_only_that():
    """segment.py's docstring hands this job to its caller: at scroll-top Hinge's filter-chips
    header and name row sit above item 1 and came back PARTIAL on every scroll-top frame in the
    corpus. With the top affirmatively confirmed (doc 5.5) the leading block is chrome, outside
    both index spaces; without that confirmation the identical observations are an item that
    might be hiding heart 1.

    The header starts 34 rows below the band's own first row, which is the smaller of the two
    clearances the corpus measured (34px and 68px). That is not decoration: a header flush ON
    that row is the false-scroll-top shape, and the test below drives it."""
    leading = [_obs(0, _BAND0 + 34, 480, complete=False), _obs(0, 700, 1674, hearts=(1584,)),
               _obs(1, 1727, 2500, hearts=(2410,))]

    confirmed, ok = _assemble(leading, at_scroll_top=True)
    assert ok == ()
    assert confirmed[0].kind == item_index.ITEM_LEADING_CHROME
    assert confirmed[0].heart_ordinal is None and confirmed[0].model_index is None
    assert [b.model_index for b in confirmed] == [None, 1, 2]
    assert [b.heart_ordinal for b in confirmed] == [None, 1, 2]

    unconfirmed, failures = _assemble(leading, at_scroll_top=False)
    assert unconfirmed[0].kind == item_index.ITEM_PARTIAL
    assert any("a heart may sit in the rows that were never inside" in f for f in failures)

    # ...and the exception is narrow: a leading block that ever showed a heart, or was ever
    # bounded end to end, is an item and stays one even at a confirmed scroll-top.
    with_heart, _ = _assemble(
        [_obs(0, _BAND0 + 34, 480, complete=False, hearts=(400,))] + leading[1:],
        at_scroll_top=True)
    assert with_heart[0].kind == item_index.ITEM_PARTIAL and with_heart[0].heart_ordinal == 1
    bounded, _ = _assemble([_obs(0, _BAND0 + 34, 480)] + leading[1:], at_scroll_top=True)
    assert bounded[0].kind == item_index.ITEM_CONTEXT


def test_a_topmost_block_flush_against_the_band_edge_refuses_the_scroll_top_claim():
    """THE FALSE-SCROLL-TOP REGRESSION. A validation pass asserted `at_scroll_top=True` about a
    capture that began three frames down a real profile and got back a usable index with eight
    items, translation 1..8, `truncated` False and ZERO failures — whose model item 1 was really
    the profile's item 2. Every completeness property corroborated the lie, and the leading-chrome
    relabelling actively swallowed the one piece of evidence against it by calling the sliced
    fragment "chrome".

    The evidence is that a genuine scroll top has page background above its topmost block
    (measured 34px and 68px on the two calibration profiles) while a band edge slicing a card
    does not. So the identical observations, moved up by 34 rows to sit ON the band's first row,
    must now refuse — and must NOT be relabelled chrome on the way."""
    sliced = [_obs(0, _BAND0, 480 - 34, complete=False), _obs(0, 700, 1674, hearts=(1584,)),
              _obs(1, 1727, 2500, hearts=(2410,))]

    blocks, failures = _assemble(sliced, at_scroll_top=True)

    assert any("at_scroll_top was asserted" in f for f in failures), failures
    assert any("begins on the analysed band's own first row" in f for f in failures)
    assert blocks[0].kind == item_index.ITEM_PARTIAL, "the evidence must not be relabelled chrome"

    # The control that proves WHICH mechanism refused: one row of background above the same
    # fragment is all the evidence the check asks for, and the claim stands again.
    clear, ok = _assemble(
        [_obs(0, _BAND0 + 1, 480 - 34, complete=False)] + sliced[1:], at_scroll_top=True)
    assert ok == ()
    assert clear[0].kind == item_index.ITEM_LEADING_CHROME

    # And the check is scoped to the claim: with no scroll-top asserted, flush content is just
    # the top of a window and the ordinals are relative, which is what `at_scroll_top=False` is.
    relative, failures = _assemble(sliced, at_scroll_top=False)
    assert not any("at_scroll_top was asserted" in f for f in failures)


# =====================================================================================
# Per-frame segmentation failures are carried forward, not swallowed
# =====================================================================================

def test_a_frame_that_contradicted_itself_makes_the_index_unusable():
    """segment.py's own failures are the index-corrupting kind — a two-heart block, a heart no
    block contains. They must reach `ItemIndex.failures` with the frame number attached, because
    a page that has to be self-consistent cannot be folded out of a frame that is not."""
    world = _WORLDS[0].copy()
    # A second heart on card 1, inside the same block: a gutter would have to be missed for this
    # to happen on the phone, and that is precisely what makes it a failure.
    th, tw = _TEMPLATE.shape
    cy = _CARD1[1] + 200
    world[cy - th // 2: cy - th // 2 + th, _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = \
        _TEMPLATE
    gray = world[0:_H].copy()
    gray[:_BAND0] = _CHROME_TOP
    gray[_BAND1:] = _CHROME_BOTTOM
    two_hearts = cv2.imencode(".png", gray)[1].tobytes()

    index = item_index.build_item_index(
        [two_hearts, _frame(_STEP)], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable
    assert any(f.startswith("frame 0:") and "ambiguous" in f for f in index.failures), \
        index.failures
    assert not index.frames[0].ok and index.frames[1].ok


# =====================================================================================
# Module properties
# =====================================================================================

def test_the_module_is_a_leaf_and_does_not_pull_in_the_driver():
    """item_index.py must stay importable without hinge.py — hinge.py is the eventual IMPORTER,
    and segment.py's one call back into it is deliberately deferred to call time. Checked in a
    fresh interpreter, since this test session has hinge.py loaded already."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import operation_love.drivers.item_index as m; "
         "print('operation_love.drivers.hinge' in sys.modules); "
         "print(m.build_item_index.__name__)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "build_item_index"], out.stdout + out.stderr


def test_the_gutter_geometry_is_shared_with_segment_and_not_re_declared():
    """The minimum gap between two blocks IS segment.py's gutter window, or the fabricated-item
    guard would be measuring against a constant free to drift from the one that cut the blocks."""
    assert item_index._MIN_ITEM_GAP_PX == min(segment._GUTTER_PX) - segment._GUTTER_TOLERANCE_PX
    assert item_index._END_TAIL_GAP_PX == max(segment._GUTTER_PX) + segment._GUTTER_TOLERANCE_PX
    assert item_index._EXTENT_TOLERANCE_PX * 2 < item_index._MIN_ITEM_GAP_PX


def test_usable_is_exactly_the_absence_of_failures():
    """`usable` must not drift into meaning anything else — it is the one-line form of "did
    anything contradict anything", on every index this file builds."""
    for index in (_full(), _index((0, _STEP, _STEP + 1300), at_scroll_top=True)):
        assert index.usable is (index.failures == ())
