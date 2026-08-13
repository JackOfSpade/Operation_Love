"""Two-tier crop assembly (operation_love/drivers/item_crops.py).

Every frame here is SYNTHESISED, never a real screencap, for the same reason test_item_index.py
and test_segment.py synthesise theirs: the calibration captures the constants were measured
against are real people's dating profiles and are gitignored (ops/calibration/, .gitignore:26),
so only geometry and counts from them appear anywhere in this repo.

The fixture builds a tall scrollable WORLD from first principles — gradient page background,
rounded-rect cards inset 53px per side, canonical 53px gutters, the genuine shipped like glyph on
the likeable ones, a heartless vitals block among them, and a strip of header chrome above card 1
— and cuts 1080x2400 windows out of it at offsets the test chooses. Ground truth is therefore a
table at the top of this file, and the assertions can name the items, their page rows, their
numbers and the exact pixels each crop must contain.

The negative that matters most is checked pixel by pixel: an item that was never bounded end to
end must produce NO image at all, not a crop of the part that happened to be visible.
"""
import math

import cv2
import numpy as np
import pytest

from operation_love.drivers import hinge, item_crops, item_identity, item_index, segment

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 21
_CONTENT_BAND = hinge.HINGE_SPEC.content_band          # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_BAND_H = _BAND1 - _BAND0
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - segment._CARD_MARGIN_PX                # 1027
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px"
_HEART_CX = 937                                        # bottom-right, as test_segment.py stamps it
_HEART_ABOVE_BOTTOM = 90                               # segment.py: ~89px above a card's bottom
_PAGE_TOP, _PAGE_BOTTOM = 254, 243                     # doc 5.10's background gradient
_WORLD_H = 6400

# The step doc 5.10.1 measured at 100% frame-to-frame heart tracking with zero phantoms.
_STEP = 363

# THE PAGE, in world rows. A strip of Hinge's header chrome, then four likeable cards with the
# heartless vitals block between the second and third. The 60-row gap under the chrome is the
# corpus's own measurement of the whitespace between the header strip and the card below it, and
# is deliberately outside the 47..58 gutter window so the chrome is cut as a card EDGE, not a
# gutter. Heights are from doc 5.10's card table (nothing exceeds the 1114px tallest card).
#
# The chrome STARTS 68 rows below the band's first row, which is what the corpus measured on both
# calibration profiles (68px on one, 34px on the other) and what `item_index._scroll_top_evidence`
# reads as evidence that this really is a scroll top. An earlier revision of this fixture ran the
# chrome flush into the band edge, which is the shape a card SLICED by the band edge has — i.e.
# the false-scroll-top geometry — and is now a refusal.
_CHROME = (_BAND0 + 68, _BAND0 + 158)
_PAGE_TOP_GAP = 1200
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
assert _CHROME[1] + 47 < _PAGE_TOP_GAP, "the chrome gap must be wider than the gutter window"

_TEMPLATE = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"


def _page_column(height=_WORLD_H):
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, height).round().astype(np.uint8)


def _paint_card(world, col, y0, y1, *, heart):
    rng = np.random.default_rng(_SEED + y0)
    world[y0:y1, _CARD_X0:_CARD_X1] = rng.integers(
        60, 200, size=(y1 - y0, _CARD_X1 - _CARD_X0), dtype=np.uint8)
    for i in range(_CORNER_RADIUS_PX):                 # carve the four corner arcs back to page
        dy = _CORNER_RADIUS_PX - i
        inset = int(math.ceil(_CORNER_RADIUS_PX
                              - math.sqrt(max(0.0, _CORNER_RADIUS_PX ** 2 - dy ** 2))))
        if inset <= 0:
            continue
        for y in (y0 + i, y1 - 1 - i):
            world[y, _CARD_X0:_CARD_X0 + inset] = col[y]
            world[y, _CARD_X1 - inset:_CARD_X1] = col[y]
    if heart:
        th, tw = _TEMPLATE.shape
        cy = y1 - _HEART_ABOVE_BOTTOM
        world[cy - th // 2: cy - th // 2 + th, _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = \
            _TEMPLATE


def _build_world():
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    # Header chrome: full card width so it reads as content, but square-cornered and heartless,
    # which is what makes it a PARTIAL block the index can only class as chrome.
    world[_CHROME[0]:_CHROME[1], _CARD_X0:_CARD_X1] = np.random.default_rng(7).integers(
        20, 90, size=(_CHROME[1] - _CHROME[0], _CARD_X1 - _CARD_X0), dtype=np.uint8)
    for kind, y0, y1 in _ROWS:
        _paint_card(world, col, y0, y1, heart=(kind == "card"))
    return world


_WORLD = _build_world()
_CHROME_TOP = np.random.default_rng(3).integers(0, 60, size=(_BAND0, _W), dtype=np.uint8)
_CHROME_BOTTOM = np.random.default_rng(4).integers(0, 60, size=(_H - _BAND1, _W), dtype=np.uint8)
_FRAMES: dict[int, bytes] = {}


def _frame(scroll: int) -> bytes:
    """The 1080x2400 window of the world at `scroll`, as PNG. Content at world row `w` lands on
    frame row `w - scroll`, so `_frame(s)` then `_frame(s + d)` is a forward scroll of `d`."""
    if scroll not in _FRAMES:
        gray = _WORLD[scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAMES[scroll] = buf.tobytes()
    return _FRAMES[scroll]


# Eleven frames at the calibrated cadence: enough that EVERY card is bounded end to end in at
# least two of them, which is what makes "choose the frame" and the signature re-match real
# questions rather than forced ones.
_FULL_SCROLL = tuple(_STEP * i for i in range(11))
assert _FULL_SCROLL[-1] + _H <= _WORLD_H
_CACHE: dict[str, object] = {}


def _index(scrolls, *, at_scroll_top=True):
    return item_index.build_item_index(
        [_frame(s) for s in scrolls], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=at_scroll_top,
        identity_band=None)


def _full():
    """The clean whole-profile index and its frames, built once — eleven real segmentations and
    ten real shift estimates are worth a few seconds and are not worth repeating."""
    if "full" not in _CACHE:
        _CACHE["full"] = (_index(_FULL_SCROLL), [_frame(s) for s in _FULL_SCROLL])
    return _CACHE["full"]


def _payload(**kw):
    index, frames = _full()
    return item_crops.build_item_payload(frames, index, **kw)


def _decode(image: bytes):
    """A crop's own pixels, as the single greyscale channel the world was painted in."""
    return cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)[:, :, 0]


# The animated card, for the drift tests: the SECOND card, chosen because the fixture shows it
# whole in three frames (3, 4 and 5), so the same page rows really are re-observed. It is the
# model's item 2 — item 1 is the first card, and the heartless vitals block takes no number.
_ANIMATED_CARD = _CARD2
_ANIMATED_ITEM = 2
# The repainted region is the card's interior, clear of the corner arcs and of the heart. Edges
# untouched on purpose: a card that redraws its video does not move its own boundary, and leaving
# the boundary alone is what keeps the INDEX and the chosen sighting identical between the two
# captures, so the only thing that differs is the stored reference.
_ANIMATED_INSET = _CORNER_RADIUS_PX + 8
# The first frame that redraws. It is deliberately AFTER the frame `_choose_sighting` picks for
# this card (frame 4, the one with the most clearance from the band edges), so the stored crop is
# the card as it was and its distance to the other items is the undisturbed one. That is the real
# shape of the failure: the reference is fine and the card has stopped matching it.
_ANIMATED_FROM = 5


def _animated_capture(*, delta: int):
    """The standard eleven frames, but with `_ANIMATED_CARD`'s interior `delta` grey levels
    brighter from frame `_ANIMATED_FROM` on: one card redrawing itself mid-scroll.

    A uniform offset rather than fresh noise, so the drift it produces is `delta` scaled by the
    repainted fraction of the crop and a test can put it either side of the measured nearest-item
    distance on purpose, instead of hoping a reroll lands there.
    """
    moved = _WORLD.copy()
    y0, y1 = _ANIMATED_CARD[1] + _ANIMATED_INSET, _ANIMATED_CARD[2] - _ANIMATED_INSET
    x0, x1 = _CARD_X0 + _ANIMATED_INSET, _CARD_X1 - _ANIMATED_INSET
    moved[y0:y1, x0:x1] = np.clip(
        moved[y0:y1, x0:x1].astype(np.int16) + delta, 0, 255).astype(np.uint8)

    frames = []
    for i, scroll in enumerate(_FULL_SCROLL):
        gray = (moved if i >= _ANIMATED_FROM else _WORLD)[scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        frames.append(buf.tobytes())

    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)
    return frames, index, item_crops.build_item_payload(frames, index)


# =====================================================================================
# The headline: one crop per item, numbered, and nothing else in the request
# =====================================================================================

def test_a_clean_capture_yields_one_numbered_crop_per_selectable_item():
    """Doc 5.2: "image 3 in the request IS item 3". Four planted cards become items 1..4, the
    heartless vitals block becomes an unnumbered context crop, and Hinge's header chrome is
    accounted for without being sent."""
    index, _ = _full()
    assert index.usable, index.failures
    payload = _payload()

    assert payload.usable, payload.failures
    assert [c.kind for c in payload.crops] == [
        item_crops.CROP_CHROME, item_crops.CROP_ITEM, item_crops.CROP_ITEM,
        item_crops.CROP_CONTEXT, item_crops.CROP_ITEM, item_crops.CROP_ITEM]
    assert [c.number for c in payload.items] == [1, 2, 3, 4]
    assert payload.item_count == 4 and payload.context_count == 1
    assert not payload.excluded and not payload.uncroppable
    assert not payload.truncated and payload.at_scroll_top

    # An entry for EVERY indexed block, so nothing is dropped without a record of it.
    assert len(payload.crops) == len(index.blocks)


def test_the_model_numbering_and_the_private_heart_table():
    """Doc 5.3: the model sees a dense 1..N, navigation counts hearts, and the payload holds the
    table between them. The vitals block sits third on the page and consumes neither number."""
    index, _ = _full()
    payload = _payload()

    assert payload.translation == (1, 2, 3, 4)
    assert payload.translation == index.translation      # nothing excluded, so they agree
    assert [c.heart_ordinal for c in payload.crops] == [None, 1, 2, None, 3, 4]
    assert payload.item(3).heart_ordinal == 3
    assert payload.item(3).number == 3


def test_each_item_crop_is_exactly_the_card_that_was_planted():
    """The crop is not approximately the card: it is the card's own rows and columns out of one
    frame, which is also what makes it usable as doc 5.6's verification reference."""
    payload = _payload()
    planted = [r for r in _ROWS if r[0] == "card"]

    for crop, (_, y0, y1) in zip(payload.items, planted):
        assert (crop.page_y0, crop.page_y1) == (y0, y1)
        assert (crop.x0, crop.x1) == (_CARD_X0, _CARD_X1)
        assert (crop.width, crop.height) == (_CARD_X1 - _CARD_X0, y1 - y0)
        assert np.array_equal(_decode(crop.image), _WORLD[y0:y1, _CARD_X0:_CARD_X1])

    context = payload.context[0]
    assert (context.page_y0, context.page_y1) == (_VITALS[1], _VITALS[2])
    assert np.array_equal(_decode(context.image),
                          _WORLD[_VITALS[1]:_VITALS[2], _CARD_X0:_CARD_X1])


def test_the_request_is_crops_only_each_item_exactly_once():
    """Doc 5.2's duplication bias: "A card straddling a scroll seam appears in two or three
    frames... repetition reads as salience". Every card here IS seen in several frames, and the
    payload still contains it once. And doc 5.7's "Not sent: full screenshots, scroll frames"."""
    index, frames = _full()
    payload = _payload()

    assert all(len(b.frames) >= 2 for b in index.selectable), [b.frames for b in index.selectable]
    assert len(payload.images) == payload.item_count + payload.context_count
    assert len(set(payload.images)) == len(payload.images)
    assert payload.images == tuple(c.image for c in payload.items + payload.context)
    assert not set(payload.images) & set(frames)


def test_the_chrome_block_is_accounted_for_but_never_cropped():
    """item_index classes Hinge's header as leading chrome at a confirmed scroll-top. It is
    outside both index spaces, so it carries no number and no image — but it is still in `crops`,
    because a payload that silently forgot a block could not be audited."""
    payload = _payload()
    chrome = payload.crops[0]

    assert chrome.kind == item_crops.CROP_CHROME
    assert chrome.image is None and chrome.signature is None and not chrome.sent
    assert chrome.number is None and chrome.heart_ordinal is None
    assert chrome.frame_index is None


# =====================================================================================
# Rule one: a crop is never a fragment
# =====================================================================================

def test_an_item_never_bounded_end_to_end_is_reported_and_not_cropped():
    """A capture that stops mid-card. The last block is real, it has a page position, and it has
    no honest image — so it is a CROP_UNCROPPABLE entry with a reason, not a crop of the part
    that happened to be on screen."""
    scrolls = _FULL_SCROLL[:3]
    index = _index(scrolls)
    assert index.usable, index.failures
    assert index.partial, "this prefix must leave a block unbounded for the test to mean anything"

    payload = item_crops.build_item_payload([_frame(s) for s in scrolls], index)

    assert payload.usable, payload.failures
    assert payload.truncated
    assert len(payload.uncroppable) == len(index.partial)
    for crop in payload.uncroppable:
        assert crop.image is None and crop.signature is None and not crop.sent
        assert crop.number is None
        assert "fragment" in crop.reason
    # The control: it is excluded for being unbounded, not for being uninteresting — the items
    # that WERE bounded in the same capture are all cropped.
    assert payload.item_count == len(index.selectable)
    assert all(c.image is not None for c in payload.items)


def test_a_partial_item_keeps_its_heart_ordinal_even_though_it_is_not_sent():
    """Doc 5.3's index trap in its mildest form: navigation counts hearts, so a heart-bearing
    block that could not be cropped still occupies its ordinal and nothing below it renumbers."""
    observations = [
        _obs(0, 400, 1300, hearts=(1200,)),
        _obs(0, 1400, 1900, complete=False, hearts=(1800,)),   # never bounded, but has a heart
        _obs(0, 1960, 2340, hearts=(2250,)),
    ]
    blocks, failures = item_index._assemble(
        observations, at_scroll_top=False, card_x=(_CARD_X0, _CARD_X1),
        extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX,
        min_item_gap_px=item_index._MIN_ITEM_GAP_PX, band_y0=_BAND0)
    assert not failures
    index = _fake_index(blocks)

    payload = item_crops.build_item_payload([_frame(0)], index)

    assert [c.kind for c in payload.crops] == [
        item_crops.CROP_ITEM, item_crops.CROP_UNCROPPABLE, item_crops.CROP_ITEM]
    assert [c.number for c in payload.crops] == [1, None, 2]
    assert [c.heart_ordinal for c in payload.crops] == [1, 2, 3]
    # The model's item 2 is the page's THIRD heart: the uncroppable one in between kept its own.
    assert payload.translation == (1, 3)


# =====================================================================================
# Rule two: the over-tall case is detected, not stitched around
# =====================================================================================

def test_a_block_taller_than_the_band_fails_loud_instead_of_being_stitched():
    """Doc 5.10 closed the viewport question on "the tallest card fully observed is 1114px
    against an 1800px content band" and added the caveat this is: "a very long prompt answer
    could still exceed it, so the crop path should detect the case rather than assume it away".
    No single frame can ever contain such a block, and stitching two is not implemented."""
    tall = _BAND_H + 1
    blocks, failures = item_index._assemble(
        [_obs(0, 500, 1400, hearts=(1300,)),
         _obs(0, 1500, 1500 + tall, complete=False, hearts=(1600,))],
        at_scroll_top=False, card_x=(_CARD_X0, _CARD_X1),
        extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX,
        min_item_gap_px=item_index._MIN_ITEM_GAP_PX, band_y0=_BAND0)
    assert not failures
    payload = item_crops.build_item_payload([_frame(0)], _fake_index(blocks))

    assert not payload.usable
    assert len(payload.failures) == 1
    assert f"{tall}px tall" in payload.failures[0]
    assert f"{_BAND_H}px analysed band" in payload.failures[0]
    assert "stitching" in payload.failures[0]

    # The control: one pixel shorter and the same block is merely uncroppable, with no failure.
    blocks, _ = item_index._assemble(
        [_obs(0, 500, 1400, hearts=(1300,)),
         _obs(0, 1500, 1500 + _BAND_H, complete=False, hearts=(1600,))],
        at_scroll_top=False, card_x=(_CARD_X0, _CARD_X1),
        extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX,
        min_item_gap_px=item_index._MIN_ITEM_GAP_PX, band_y0=_BAND0)
    assert item_crops.build_item_payload([_frame(0)], _fake_index(blocks)).usable


def test_no_item_of_a_real_capture_comes_close_to_the_band():
    """The positive side of the same measurement: on a layout built to doc 5.10's card table,
    every crop fits inside the analysed band with room to spare, which is why single frames
    suffice and no stitching exists to be tempted by."""
    payload = _payload()
    assert all(c.height < _BAND_H for c in payload.items + payload.context)
    assert max(c.height for c in payload.items + payload.context) / _BAND_H < 0.75


# =====================================================================================
# Exclusion: a mechanism, with no detector attached
# =====================================================================================

def test_exclusion_withholds_the_image_entirely():
    """Doc 2.4 excludes the endorsement section "entirely... because models reference what they
    are shown". So there must be no image at all, not an image the prompt asks the model to
    ignore."""
    index, frames = _full()
    payload = item_crops.build_item_payload(
        frames, index, exclude=item_crops.exclude_page_rows([(_VITALS[1], _VITALS[2])]))

    assert payload.usable, payload.failures
    assert payload.context_count == 0
    assert len(payload.excluded) == 1
    dropped = payload.excluded[0]
    assert dropped.image is None and dropped.signature is None and not dropped.sent
    assert dropped.number is None
    assert item_crops.EXCLUSION_ENDORSEMENT in dropped.reason
    assert f"{_VITALS[1]}..{_VITALS[2]}" in dropped.reason
    # ...and the item numbering is untouched, because a context block never held a number.
    assert payload.translation == (1, 2, 3, 4)
    assert len(payload.images) == 4


def test_excluding_a_heart_bearing_block_renumbers_the_model_and_not_navigation():
    """Doc 5.3's index trap, stated exactly: "if an excluded block turns out to HAVE a heart, the
    driver still counts it, because navigation counts hearts". The model's list closes up; the
    heart ordinals do not."""
    index, frames = _full()
    payload = item_crops.build_item_payload(
        frames, index, exclude=item_crops.exclude_page_rows([(_CARD2[1], _CARD2[2])]))

    assert payload.usable, payload.failures
    assert [c.number for c in payload.items] == [1, 2, 3]
    assert payload.translation == (1, 3, 4)              # heart 2 is gone from the model's list
    assert payload.translation != index.translation      # ...so THIS is the table to navigate by
    assert payload.excluded[0].heart_ordinal == 2        # ...and its own ordinal is untouched
    assert index.heart_ordinal_for(2) == 2               # the index still knows it as heart 2


def test_exclude_page_rows_only_touches_the_rows_it_was_given():
    """The mechanism is positional and takes the answer from the caller — there is deliberately
    no detector here for a block type neither calibration capture contained."""
    predicate = item_crops.exclude_page_rows([(_VITALS[1], _VITALS[2])])
    index, _ = _full()
    hit = [b for b in index.blocks if predicate(b) is not None]

    assert len(hit) == 1 and (hit[0].page_y0, hit[0].page_y1) == (_VITALS[1], _VITALS[2])
    # A span one row short of the block touches nothing.
    assert all(item_crops.exclude_page_rows([(_VITALS[1] - 10, _VITALS[1])])(b) is None
               for b in index.blocks)
    assert "endorsement" == item_crops.EXCLUSION_ENDORSEMENT


def test_excluding_everything_selectable_is_a_failure_not_an_empty_request():
    """A payload the model cannot answer is not a payload."""
    index, frames = _full()
    payload = item_crops.build_item_payload(
        frames, index, exclude=item_crops.exclude_page_rows([(0, 10 ** 6)]))

    assert not payload.usable
    assert any("nothing for the model to choose between" in f for f in payload.failures)
    with pytest.raises(item_crops.ItemCropError, match="unusable"):
        payload.item(1)


# =====================================================================================
# The crop signature (doc 5.6's verification reference)
# =====================================================================================

def test_a_signature_matches_the_same_item_seen_in_another_frame_and_no_other():
    """The whole point of storing one: later, a screenshot of a sheet is signed the same way and
    compared. The same card from a DIFFERENT frame must be nearest to its own stored signature by
    a wide margin, and `signature_of` is the function both sides go through."""
    index, frames = _full()
    payload = _payload()

    for number, block in enumerate(index.selectable, start=1):
        crop = payload.item(number)
        elsewhere = [o for o in block.croppable if o.frame_index != crop.frame_index]
        assert elsewhere, "the fixture must show every card whole in more than one frame"
        seen_again = item_crops.signature_of(
            frames[elsewhere[0].frame_index], y0=elsewhere[0].frame_y0,
            y1=elsewhere[0].frame_y1, x0=block.x0, x1=block.x1)

        own = crop.signature.distance(seen_again)
        others = [payload.item(k).signature.distance(seen_again)
                  for k in range(1, payload.item_count + 1) if k != number]
        assert own == 0.0                       # same pixels, same decode, same grid
        assert min(others) > 1.0, (number, own, others)

    assert payload.signature_grid == item_crops._SIGNATURE_GRID
    assert len(payload.signature_for(1).cells) == 32 * 32


def test_signature_geometry_and_digest_are_carried_with_the_cells():
    """A signature normalises scale away, so the crop's real extent is carried beside it — a
    215px context block and a 1000px photo card are not the same item however their cells
    compare."""
    payload = _payload()
    for crop in payload.items + payload.context:
        assert (crop.signature.width, crop.signature.height) == (crop.width, crop.height)
        assert crop.signature.grid == item_crops._SIGNATURE_GRID
    digests = {c.signature.digest for c in payload.items + payload.context}
    assert len(digests) == payload.item_count + payload.context_count
    assert all(len(d) == 64 for d in digests)


def test_signature_of_refuses_a_rect_that_is_not_inside_the_frame():
    """Clipping a rect to fit would return the signature of a different region under the name of
    the one that was asked for."""
    with pytest.raises(item_crops.ItemCropError, match="does not fit inside"):
        item_crops.signature_of(_frame(0), y0=0, y1=_H + 1, x0=_CARD_X0, x1=_CARD_X1)
    with pytest.raises(item_crops.ItemCropError, match="does not fit inside"):
        item_crops.signature_of(_frame(0), y0=100, y1=100, x0=_CARD_X0, x1=_CARD_X1)
    with pytest.raises(item_crops.ItemCropError, match="did not decode"):
        item_crops.signature_of(b"not an image", y0=0, y1=10, x0=0, x1=10)


def test_distance_refuses_two_different_grids():
    """Two signatures at different grids are not two measurements of the same thing."""
    coarse = item_crops.signature_of(_frame(0), y0=400, y1=900, x0=_CARD_X0, x1=_CARD_X1,
                                     grid=(8, 8))
    fine = item_crops.signature_of(_frame(0), y0=400, y1=900, x0=_CARD_X0, x1=_CARD_X1)
    assert coarse.distance(coarse) == 0.0
    with pytest.raises(item_crops.ItemCropError, match="not a distance"):
        fine.distance(coarse)


# =====================================================================================
# Whether a stored crop can serve as doc 5.6's reference at all (the animated-card class)
# =====================================================================================

def test_a_static_capture_measures_no_drift_and_every_item_stays_separable():
    """The baseline the two negatives below are read against. Nothing in this world moves, so
    every item re-observed at its own page rows in another frame must be EXACTLY itself, and the
    comparison `separable` makes — own drift against distance to the nearest other item — must
    come out True on all of them with the numbers on the record."""
    payload = _payload()

    for crop in payload.items:
        assert crop.signature_drift == 0.0, (crop.number, crop.signature_drift)
        assert crop.drift_frames, "the fixture must re-observe every item at least once"
        assert crop.separable is True
    assert payload.max_signature_drift == 0.0
    assert payload.min_item_separation > 1.0
    assert payload.unseparable_items == () and payload.undetermined_items == ()


def test_drift_is_measured_by_resampling_the_rect_in_every_frame_that_contains_it():
    """THE MECHANISM, and it is the correction: an earlier revision of this module's docstring
    measured re-observation only over the frames where SEGMENTATION returned a byte-identical
    rect, and reported max 0.02 against a 4.66 nearest-item distance. That sample systematically
    excludes moving cards, because a moving card's segmented edge wobbles by a pixel and drops
    out of it. So the frames sampled are chosen by rect CONTAINMENT — every frame whose analysed
    band holds the crop's own page rows — and segmentation is not consulted for any of them."""
    index, _ = _full()
    payload = _payload()

    for crop in payload.items + payload.context:
        contained = tuple(
            i for i, offset in enumerate(index.offsets)
            if i != crop.frame_index
            and index.frames[i].band[0] <= crop.page_y0 - offset
            and crop.page_y1 - offset <= index.frames[i].band[1])
        assert crop.drift_frames == contained, crop.number
        # ...and a frame that only PARTLY holds the rect is skipped rather than clipped: a
        # clipped rect is a different region, so its distance would not be a drift.
        assert all(index.frames[i].band[0] <= crop.page_y0 - index.offsets[i]
                   for i in crop.drift_frames)


def test_a_card_whose_pixels_move_is_reported_unseparable_rather_than_thresholded():
    """THE ANIMATED-CARD REGRESSION. Doc 5.4 named animated cards as needing "a tolerance band on
    the signature comparison, or explicit detection and a different path", and the corpus ruled
    the tolerance band out: one profile re-observes an animated card 25.0 grey levels from itself
    against a 6.27 distance to the nearest DIFFERENT item, so no fixed threshold can accept the
    right answer there and reject the wrong one. This is the detection.

    Here one card's interior is repainted between frames, exactly as a video card redraws, while
    its edges, its heart and every other block stay where they were — so the index is unchanged
    and only the stored reference is compromised. The item must come back with its drift measured
    and `separable` False, and the eight other items must be untouched: failing the whole payload
    would throw away good references to punish one card."""
    frames, index, payload = _animated_capture(delta=60)
    assert index.usable, index.failures
    assert payload.usable and payload.item_count == _payload().item_count

    moved = payload.item(_ANIMATED_ITEM)
    assert moved.frame_index < _ANIMATED_FROM, "the stored crop must predate the redraw"
    assert moved.signature_drift > moved.nearest_item_distance
    assert moved.separable is False
    assert payload.unseparable_items == (_ANIMATED_ITEM,)
    assert payload.max_signature_drift == moved.signature_drift

    for crop in payload.items:
        if crop.number != _ANIMATED_ITEM:
            assert crop.signature_drift == 0.0 and crop.separable is True


def test_a_small_change_is_a_comparison_and_not_a_threshold():
    """The control that proves WHICH mechanism failed the card above: `separable` compares two
    measured numbers, so the same card moving by less than its distance to its nearest neighbour
    is still separable — with a non-zero drift on the record either way. No constant is shipped
    and none could be: the corpus's two profiles put the nearest-item distance at 4.66 and 6.27
    and the worst drift at 2.66 and 25.0, so the same threshold cannot serve both."""
    _, _, payload = _animated_capture(delta=1)

    moved = payload.item(_ANIMATED_ITEM)
    assert 0.0 < moved.signature_drift < moved.nearest_item_distance
    assert moved.separable is True
    assert payload.unseparable_items == ()


def test_an_item_nothing_re_observed_is_undetermined_and_never_reported_stable():
    """None is not a soft False. A capture that never revisits a scroll position measures nothing
    about its own references, and reporting silence as stability is the substitution doc 5.6
    forbids — so the item is `undetermined`, `separable` is None, and it is kept apart from the
    items that were measured and failed."""
    index = _index((0,))
    payload = item_crops.build_item_payload([_frame(0)], index)

    lonely = [c for c in payload.items if not c.drift_frames]
    assert lonely, "a one-frame capture can re-observe nothing"
    for crop in lonely:
        assert crop.signature_drift is None
        assert crop.separable is None
        assert crop.number in payload.undetermined_items
        assert crop.number not in payload.unseparable_items
    assert payload.max_signature_drift is None, "no evidence is not zero drift"


# =====================================================================================
# Frame choice, and the hard stops
# =====================================================================================

def test_the_frame_each_item_is_cropped_from_is_chosen_and_deterministic():
    """Only a sighting that saw both edges may be cropped, and among those the one with the most
    clearance from the analysed band's edges wins, ties going to the lowest frame index. Two runs
    over the same capture therefore produce byte-identical crops, which matters because the crop
    is stored as a verification reference."""
    index, frames = _full()
    payload = _payload()

    for number, block in enumerate(index.selectable, start=1):
        crop = payload.item(number)
        chosen = [o for o in block.croppable if o.frame_index == crop.frame_index]
        assert chosen, "the crop must come from a sighting that observed both edges"

        def clearance(obs):
            r0, r1 = index.frames[obs.frame_index].band
            return min(obs.frame_y0 - r0, r1 - obs.frame_y1)

        best = max(clearance(o) for o in block.croppable)
        assert clearance(chosen[0]) == best
        assert crop.frame_index == min(o.frame_index for o in block.croppable
                                       if clearance(o) == best)

    again = item_crops.build_item_payload(frames, index)
    assert again.images == payload.images


def test_an_unusable_index_is_a_hard_stop_before_any_crop_is_made():
    """Doc 5.3: "treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate". The gate is inherited from ItemIndex.usable rather than re-derived."""
    scrolls = (0, _STEP * 4)                   # a step far past the trust window
    index = _index(scrolls)
    assert not index.usable and index.blocks == ()

    with pytest.raises(item_crops.ItemCropError, match="refusing to crop an unusable item index"):
        item_crops.build_item_payload([_frame(s) for s in scrolls], index)


def test_frames_that_are_not_the_indexed_frames_are_refused():
    """The index's page coordinates mean nothing against another capture, and cropping them at
    those coordinates would produce crops of the wrong profile at exactly the right rows.

    THE REGRESSION IS THE THIRD CASE. The guard used to be the frame COUNT plus each decoded
    frame's size, and a validation pass drove straight through it: every Hinge screencap on this
    device is 1080x2400, so handing the crop pass the same frames in reverse order produced a
    usable payload with ten images, ZERO failures, and signatures 18.8 to 156.5 away from the
    right ones — against the 4.66 that separates the two most alike genuine items. The size test
    could never fire on a real mix-up. A per-frame digest can only fire on one."""
    index, frames = _full()

    with pytest.raises(item_crops.ItemCropError, match="frame\\(s\\) given for an index"):
        item_crops.build_item_payload(frames[:-1], index)

    ok, small = cv2.imencode(".png", np.zeros((_H // 2, _W // 2), np.uint8))
    assert ok
    wrong = list(frames)
    wrong[_payload().item(1).frame_index] = small.tobytes()
    with pytest.raises(item_crops.ItemCropError, match="not the frame the index was built from"):
        item_crops.build_item_payload(wrong, index)

    # Same count, same 1080x2400 size, same profile, same bytes — only the order differs, and
    # every rect in the index now names different content.
    with pytest.raises(item_crops.ItemCropError, match="not the frame the index was built from"):
        item_crops.build_item_payload(list(reversed(frames)), index)

    # ...and it fires even when the swapped frame is one no item is cropped from, because the
    # payload is a claim about the whole capture and not only about the frames it happened to
    # slice.
    cropped_from = {c.frame_index for c in _payload().crops if c.frame_index is not None}
    unused = next(i for i in range(len(frames)) if i not in cropped_from)
    untouched = list(frames)
    untouched[unused] = _frame(_FULL_SCROLL[-1] + 1)
    with pytest.raises(item_crops.ItemCropError, match=f"frame {unused} is not the frame"):
        item_crops.build_item_payload(untouched, index)


def test_a_page_class_this_module_has_no_policy_for_is_refused():
    """Selectability is policy (doc 5.3), so a block class nobody assigned a tier to must stop the
    build rather than be guessed into one. Unreachable from a usable index today — an ambiguous
    block always carries its own failure — which is exactly why it is asserted rather than
    assumed."""
    blocks, _ = item_index._assemble(
        [_obs(0, 400, 1300, hearts=(1000, 1200))],          # two hearts: ITEM_AMBIGUOUS
        at_scroll_top=False, card_x=(_CARD_X0, _CARD_X1),
        extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX,
        min_item_gap_px=item_index._MIN_ITEM_GAP_PX, band_y0=_BAND0)
    assert [b.kind for b in blocks] == [item_index.ITEM_AMBIGUOUS]

    with pytest.raises(item_crops.ItemCropError, match="no crop policy for"):
        item_crops.build_item_payload([_frame(0)], _fake_index(blocks))


def test_item_refuses_a_number_the_model_was_never_offered():
    """Doc 5.6: never substitute a different item. Out of range raises rather than clamping."""
    payload = _payload()
    assert payload.item(payload.item_count).number == payload.item_count
    for bad in (0, -1, payload.item_count + 1):
        with pytest.raises(item_crops.ItemCropError, match="outside 1.."):
            payload.item(bad)
        with pytest.raises(item_crops.ItemCropError, match="outside 1.."):
            payload.signature_for(bad)


# =====================================================================================
# Helpers for driving the payload off hand-written blocks
# =====================================================================================

def _obs(frame_index, page_y0, page_y1, *, complete=True, hearts=()):
    """One hand-written sighting, in a frame whose rows equal its page rows."""
    kind = (segment.BLOCK_SELECTABLE if hearts else segment.BLOCK_CONTEXT) if complete \
        else segment.BLOCK_PARTIAL
    return item_index.BlockObservation(
        frame_index=frame_index, page_y0=page_y0, page_y1=page_y1,
        frame_y0=page_y0, frame_y1=page_y1, kind=kind, complete=complete,
        top_observed=complete, bottom_observed=complete,
        hearts=tuple((_HEART_CX, y) for y in hearts))


def _fake_index(blocks):
    """An `ItemIndex` over hand-written blocks, with one frame's real segmentation behind it so
    the band, the frame size and the crop path are all the shipped ones."""
    segmentation = segment.segment_frame(
        _frame(0), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    return item_index.ItemIndex(
        blocks=tuple(blocks), frames=(segmentation,), shifts=(), offsets=(0,),
        page_span=segmentation.band, at_scroll_top=False, reached_end=False, tail_gap_px=None,
        failures=(), identity=item_identity.capture_profile_identity((), identity_band=None))
