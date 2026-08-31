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
import hashlib
import math

import cv2
import numpy as np
import dataclasses

import pytest

from operation_love import targeting_policy as tp
from operation_love.drivers import (
    hinge, item_crops, item_identity, item_index, item_type_preflight, segment)

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

    for crop, (_, y0, y1) in zip(payload.items, planted, strict=True):
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

def test_photo_only_policy_demotes_a_heart_bearing_prompt_without_renumbering_hearts():
    """A written card may keep Hinge's like heart, but it must not become a model choice.

    The crop layer is where the choice list is made, so the policy receives the complete encoded
    crop and demotes the second likeable card to readable, *unnumbered* context.  It must not
    erase that card's heart from the navigation space: model item 2 is now page heart 3.
    """
    index, frames = _full()
    seen: list[bytes] = []

    def photo_only(crop: bytes) -> str | None:
        seen.append(crop)
        return "written prompt cards are context, never model items" if len(seen) == 2 else None

    payload = item_crops.build_item_payload(frames, index, unnumber=photo_only)

    assert payload.usable, payload.failures
    assert len(seen) == 4                         # complete, heart-bearing cards only
    assert [c.kind for c in payload.crops] == [
        item_crops.CROP_CHROME, item_crops.CROP_ITEM, item_crops.CROP_CONTEXT,
        item_crops.CROP_CONTEXT, item_crops.CROP_ITEM, item_crops.CROP_ITEM]
    assert [c.number for c in payload.items] == [1, 2, 3]
    assert [(c.page_y0, c.page_y1) for c in payload.items] == [
        _CARD1[1:], _CARD3[1:], _CARD4[1:]]

    prompt = next(c for c in payload.context if (c.page_y0, c.page_y1) == _CARD2[1:])
    assert prompt.number is None and prompt.heart_ordinal == 2
    assert prompt.image == seen[1] and prompt.sent
    assert "written prompt" in prompt.reason
    # `translation`, not ItemIndex.translation, is what the counting-navigation caller must
    # spend after the policy has removed a heart-bearing card from the model's dense list.
    assert index.translation == (1, 2, 3, 4)
    assert payload.translation == (1, 3, 4)
    assert payload.item(2).heart_ordinal == 3


def test_photo_only_policy_refuses_a_capture_with_no_numbered_photographs():
    """Unknown is not permission to offer a possible prompt to the model.

    If every likeable crop is written or cannot confidently be called a photograph, they remain
    available as unnumbered context but the payload is unusable: no caller can manufacture a
    model item number and attach an opener to a prompt card.
    """
    index, frames = _full()
    payload = item_crops.build_item_payload(
        frames, index, unnumber=lambda _crop: "not confidently a photograph")

    assert not payload.usable
    assert payload.item_count == 0 and payload.translation == ()
    prompt_cards = [c for c in payload.context if c.heart_ordinal is not None]
    assert [c.heart_ordinal for c in prompt_cards] == [1, 2, 3, 4]
    assert all(c.number is None and c.sent for c in prompt_cards)
    assert all("not confidently a photograph" in c.reason for c in prompt_cards)
    assert any("nothing for the model to choose between" in failure for failure in payload.failures)
    with pytest.raises(item_crops.ItemCropError, match="unusable"):
        payload.item(1)


def test_hinge_photo_policy_ignores_geometry_and_requires_photo_evidence():
    rng = np.random.default_rng(991)
    square_photo = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    landscape_photo = rng.integers(0, 256, size=(180, 256, 3), dtype=np.uint8)
    # A featureless crop is deliberately UNKNOWN, not PHOTO. Ambiguity must not manufacture a
    # photo ordinal; it remains readable context instead.
    portrait_unknown = np.full((256, 180, 3), (130, 105, 80), dtype=np.uint8)
    written_square = np.full((256, 256, 3), 246, dtype=np.uint8)
    for y in (70, 130):
        cv2.putText(written_square, "PROMPT", (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2, cv2.LINE_AA)

    def encoded(image):
        ok, value = cv2.imencode(".png", image)
        assert ok
        return value.tobytes()

    assert item_crops.unnumber_unless_confident_photo(encoded(square_photo)) is None
    assert item_crops.unnumber_unless_confident_photo(encoded(landscape_photo)) is None
    assert (item_type_preflight.classify_crop(encoded(portrait_unknown))
            == item_type_preflight.UNKNOWN)
    assert "photo_only" in item_crops.unnumber_unless_confident_photo(encoded(portrait_unknown))
    assert "photo_only" in item_crops.unnumber_unless_confident_photo(encoded(written_square))


def test_hinge_photo_policy_does_not_number_unknown_cards(monkeypatch):
    """Classifier ambiguity is readable context, never an invented photo position.

    The synthetic profile's four heart-bearing cards are 974px wide but 900, 760, 1000, and
    820px tall. Force an inconclusive result for all four; geometry and a presumed item count may
    not override the content classifier.
    """
    monkeypatch.setattr(item_type_preflight, "classify_crop",
                        lambda _image: item_type_preflight.UNKNOWN)
    index, frames = _full()
    payload = item_crops.build_item_payload(
        frames, index, unnumber=item_crops.unnumber_unless_confident_photo)

    assert not payload.usable
    assert payload.item_count == 0 and payload.translation == ()
    assert [crop.heart_ordinal for crop in payload.context if crop.heart_ordinal] == [1, 2, 3, 4]
    assert all("classified as unknown" in crop.reason for crop in payload.context
               if crop.heart_ordinal)


def test_confident_photo_candidate_prepass_matches_content_policy_and_exclusions(monkeypatch):
    """Dwell candidates are the confidently photographic selectable crops only.

    This pre-pass intentionally runs before the expensive still-media proof, but it must use the
    same PHOTO/WRITTEN/UNKNOWN decision and the same video/block exclusion as the final payload.
    It is only a cost filter: later code still requires every returned heart to pass dwell.
    """
    classes = iter([
        item_type_preflight.PHOTO, item_type_preflight.WRITTEN,
        item_type_preflight.UNKNOWN, item_type_preflight.PHOTO,
    ])
    monkeypatch.setattr(item_type_preflight, "classify_crop", lambda _image: next(classes))
    index, frames = _full()

    ordinals = item_crops.confident_photo_heart_ordinals(
        frames, index, exclude=lambda block: "video" if block.heart_ordinal == 4 else None)

    assert ordinals == (1,)


def test_unknown_prompt_cannot_shift_the_last_photo_from_six_to_seven(monkeypatch):
    """Regression for the reported dog-photo ordinal, derived from classes rather than a cap."""
    classes = iter([
        item_type_preflight.PHOTO, item_type_preflight.WRITTEN,
        item_type_preflight.PHOTO, item_type_preflight.PHOTO,
        item_type_preflight.WRITTEN, item_type_preflight.PHOTO,
        item_type_preflight.UNKNOWN, item_type_preflight.PHOTO,
        item_type_preflight.PHOTO,
    ])
    monkeypatch.setattr(item_type_preflight, "classify_crop", lambda _image: next(classes))

    numbered_hearts = tuple(
        heart for heart in range(1, 10)
        if item_crops.unnumber_unless_confident_photo(b"crop") is None)

    assert numbered_hearts == (1, 3, 4, 6, 8, 9)
    assert numbered_hearts.index(9) + 1 == 6


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


def _evidence(drift, frames=(2,), *, dwell=None, regime=None):
    """The gate's argument, built the way every production caller builds it.

    `regime` defaults to None on purpose, matching every caller that does not declare where its
    drift came from: those objects must keep taking the STRICTER parked ceiling, so the default
    here is also the assertion that the parked bound was not loosened for anybody.
    """
    return item_crops.still_photo_evidence_from_drift(drift, frames, dwell, drift_regime=regime)


def _full_dwell(*, exact=True, span_s=6.0, screened=True, digests=2, centered=True,
                offset=0.02, probe_ran=True, probe_exact=True, probe_screened=True,
                probe_digests=2, probe_centered=True, probe_offset=0.02, probe_span_s=6.0):
    """A dwell that passes every rung, so one field at a time can be spoiled.

    `centered` is the autoplay precondition (2026-08-21): Hinge plays a video only near the
    centre of the screen, so byte-exactness measured off-centre proves nothing. It defaults True
    here so the OTHER rungs stay individually spoilable; its own rung is exercised below.

    The `probe_*` half is the RE-ATTACH probe's second burst, taken after the card was scrolled
    out of the autoplay band and back. It defaults to passing for the same reason: each of its
    rungs is spoiled one at a time below.
    """
    return item_crops.StillPhotoDwell(
        dwell_frame_sha256s=tuple(f"{i:064x}" for i in range(digests)),
        dwell_exact=exact, dwell_span_s=span_s, mute_screens_complete=screened,
        centered=centered, center_offset_frac=offset,
        reattach_probe_ran=probe_ran,
        reattach_dwell_frame_sha256s=tuple(f"{i:064x}" for i in range(probe_digests)),
        reattach_dwell_exact=probe_exact, reattach_dwell_span_s=probe_span_s,
        reattach_mute_screens_complete=probe_screened, reattach_centered=probe_centered,
        reattach_center_offset_frac=probe_offset)


@pytest.mark.parametrize("centered, offset", [(None, None), (False, 0.42), (1, 0.02)])
def test_the_dwell_rung_refuses_a_card_that_was_not_in_the_autoplay_trigger_zone(
        centered, offset, installed_still_photo_bound):
    """Hinge autoplays only near the centre, so an off-centre still run is not evidence.

    `1` is included deliberately: a truthy placeholder must not buy the observation, exactly as
    every other three-valued field on this object is compared with `is True`.
    """
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(0.01, dwell=_full_dwell(centered=centered, offset=offset)))
    assert reason is not None
    assert "autoplay trigger zone" in reason
    assert "never near enough to the centre" in reason


def test_a_centred_card_passes_the_autoplay_rung_and_reaches_the_later_ones(
        installed_still_photo_bound):
    """The rung is a precondition on byte-exactness, not an extra veto on everything after it."""
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(0.01, dwell=_full_dwell(centered=True, screened=False)))
    assert reason is not None and "mute-control screening" in reason


def test_still_photo_dwell_evidence_measures_centring_from_the_content_band():
    """The producer computes `centered` itself; a caller cannot assert it into existence."""
    index, frames = _full()
    without = item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1], frames[-1]], dwell_span_s=6.0,
        mute_screen=lambda _frame, _rect: True)
    assert without, "the fixture must produce at least one dwell entry"
    # No content_band -> the observation was never made -> fail closed, never a silent pass.
    assert all(entry.centered is None and entry.center_offset_frac is None
               for entry in without.values())

    with_band = item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1], frames[-1]], dwell_span_s=6.0,
        mute_screen=lambda _frame, _rect: True, content_band=(0.0, 1.0))
    assert with_band and set(with_band) == set(without)
    for entry in with_band.values():
        assert isinstance(entry.center_offset_frac, float)
        assert entry.centered is (abs(entry.center_offset_frac)
                                  <= item_crops.STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC)


def _fold_reattach(dwell, index, frames, *, burst=None, shift=0,
                   mute_screen=lambda _frame, _rect: True):
    """Fold one probe's second burst in, the way the driver and the replay both do."""
    probe = item_crops.ReattachProbe(
        anchor=frames[-1], frames=tuple(frames[-1:] if burst is None else burst),
        span_s=6.0, page_shift_px=shift)
    return item_crops.still_photo_reattach_evidence(
        dwell, index, frame_count=len(frames), probe=probe, mute_screen=mute_screen,
        content_band=(0.0, 1.0))


def _first_burst(index, frames, mute_screen=lambda _frame, _rect: True):
    return item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1], frames[-1]], dwell_span_s=6.0, mute_screen=mute_screen,
        content_band=(0.0, 1.0))


def _repaint(frame: bytes, rect) -> bytes:
    """The same frame with one pixel inside `rect` inverted: motion, at zero tolerance."""
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR).copy()
    row = (rect[1] + rect[3]) // 2
    col = (rect[0] + rect[2]) // 2
    image[row, col, 0] = 255 - image[row, col, 0]
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_the_reattach_producer_records_a_second_burst_that_agrees_with_the_first():
    """A probe that ran, came back, and saw nothing move: every second-burst leg is measured."""
    index, frames = _full()
    folded = _fold_reattach(_first_burst(index, frames), index, frames,
                            burst=[frames[-1], frames[-1]])

    assert folded, "the fixture must produce at least one dwell entry"
    for entry in folded.values():
        assert entry.reattach_probe_ran is True
        assert entry.reattach_dwell_exact is True
        assert entry.reattach_mute_screens_complete is True
        assert entry.reattach_dwell_span_s == 6.0
        assert len(entry.reattach_dwell_frame_sha256s) == 3   # the probe anchor leads its burst
        assert entry.reattach_centered is (abs(entry.reattach_center_offset_frac)
                                           <= item_crops.STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC)


def test_motion_in_the_second_burst_alone_is_still_motion():
    """The whole point of looking twice: media that only starts once Hinge re-attaches it."""
    index, frames = _full()
    rect = item_crops.dwell_card_rects(index, len(frames) - 1)
    ordinal = sorted(rect)[0]
    first = _first_burst(index, frames)
    assert first[ordinal].dwell_exact is True

    folded = _fold_reattach(first, index, frames,
                            burst=[frames[-1], _repaint(frames[-1], rect[ordinal])])

    assert folded[ordinal].dwell_exact is True, "the FIRST burst still saw nothing move"
    assert folded[ordinal].reattach_dwell_exact is False


def test_a_mute_control_that_renders_only_on_re_entry_fails_the_second_screen():
    """C3 on the second burst, for the control Hinge redraws when it re-attaches media."""
    index, frames = _full()
    folded = _fold_reattach(_first_burst(index, frames), index, frames,
                            burst=[frames[-1], frames[-1]],
                            mute_screen=lambda _frame, _rect: False)

    assert all(entry.reattach_mute_screens_complete is False for entry in folded.values())


def test_a_card_the_probe_pushed_off_the_screen_made_no_second_observation():
    """Missing, not failed: an unmeasurable card refuses at the probe rung and is never read as
    motion, which would name it a video on the strength of nothing."""
    index, frames = _full()
    folded = _fold_reattach(_first_burst(index, frames), index, frames,
                            burst=[frames[-1], frames[-1]], shift=100_000)

    assert folded
    for entry in folded.values():
        assert entry.reattach_probe_ran is False
        assert entry.reattach_dwell_exact is None
        assert entry.reattach_dwell_frame_sha256s == ()


def test_the_second_burst_follows_the_measured_page_shift_rather_than_the_card():
    """The rect is TRANSLATED by what the probe measured, so the second burst looks at the card
    the first one measured instead of at whatever now occupies those rows.  Nothing here
    re-identifies a card, which is what keeps the probe from substituting one."""
    index, frames = _full()
    rects = item_crops.dwell_card_rects(index, len(frames) - 1)
    ordinal = sorted(rects)[0]
    shift = 40
    moved_rect = (rects[ordinal][0], rects[ordinal][1] - shift,
                  rects[ordinal][2], rects[ordinal][3] - shift)
    seen: list[tuple[int, int, int, int]] = []

    def record(_frame, rect):
        seen.append(rect)
        return True

    _fold_reattach(_first_burst(index, frames), index, frames,
                   burst=[frames[-1], frames[-1]], shift=shift, mute_screen=record)

    assert moved_rect in seen
    assert rects[ordinal] not in seen


def test_the_centering_geometry_is_defined_once_and_measured_from_the_screen_centre():
    """The zone, on the CALIBRATED band, where the aim point and the band centre coincide.

    Renamed 2026-08-28 with `content_band_center_row`: the offsets below are still
    denominated in the band's height, but the row they are measured FROM is the screen's
    centre. The test below this one is the one that can tell those two apart.
    """
    band = (0.125, 0.875)
    height = 2400
    centre_row = item_crops.content_band_center_row(height, band)
    assert centre_row == pytest.approx(1200.0)
    # A card centred on the band's centre has zero offset and is inside the zone.
    dead_centre = (53, int(centre_row) - 487, 1027, int(centre_row) + 487)
    assert item_crops.card_center_offset_frac(
        dead_centre, frame_height=height, content_band=band) == pytest.approx(0.0, abs=1e-3)
    assert item_crops.card_is_centered(dead_centre, frame_height=height, content_band=band)
    # A card near the bottom of the band is outside it, and the sign says which way to scroll.
    low = (53, 1900, 1027, 2100)
    assert item_crops.card_center_offset_frac(
        low, frame_height=height, content_band=band) > item_crops.STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC
    assert not item_crops.card_is_centered(low, frame_height=height, content_band=band)
    high = (53, 300, 1027, 500)
    assert item_crops.card_center_offset_frac(
        high, frame_height=height, content_band=band) < 0


def test_the_autoplay_aim_point_is_the_screen_centre_and_never_moves_with_the_content_band():
    """The aim point is a property of the PHONE; `content_band` is our own analysis window.

    Hinge autoplays a card only at SCREEN centre (config.yaml
    `still_photo_assumption_acceptance.rationale`, the owner-accepted UNMEASURED assumption the
    whole still-photo ladder rests on). `content_band` is which rows we let the segmenter read,
    and Hinge 10.1.0 drawing a pinned profile header inside it is exactly the sort of thing that
    makes someone narrow it.

    `content_band_center_row` derived its row from the BAND until 2026-08-28, which was a no-op
    only because the calibrated `(0.125, 0.875)` is symmetric about 0.5. This test exists so that
    coincidence can never be load-bearing again: narrowing the band to (0.2154, 0.875) to clear
    that header would have re-aimed the row to 1308.5 and dwelt every candidate 108.5px below the
    only place playback is claimed to start, while `centered` still read True everywhere
    downstream. Mutation-checked against exactly that reversion.
    """
    height = _H
    assert item_crops.content_band_center_row(height, _CONTENT_BAND) == pytest.approx(height / 2)

    # Move the band anywhere, symmetric or not: the aim point does not follow it.
    for moved in ((0.2154, 0.875), (0.125, 0.875), (0.0, 1.0), (0.30, 0.70), (0.10, 0.60)):
        assert item_crops.content_band_center_row(height, moved) == pytest.approx(
            height / 2), f"band {moved} moved the autoplay aim point off screen centre"

    # The worked example, spelled out: the band-derived reading really is a DIFFERENT row, so the
    # assertions above are not agreeing with the old formula by luck.
    pinned_header_band = (0.2154, 0.875)
    band_derived = (pinned_header_band[0] + pinned_header_band[1]) / 2 * height
    assert band_derived == pytest.approx(1308.48)
    assert abs(band_derived - height / 2) == pytest.approx(108.48)

    # ...and the whole offset chain follows the screen, not the band: a card sitting exactly on
    # screen centre reads ZERO offset under a band whose own centre is 108px away from it.
    dead_centre = (_CARD_X0, height // 2 - 400, _CARD_X1, height // 2 + 400)
    assert item_crops.card_center_offset_frac(
        dead_centre, frame_height=height, content_band=pinned_header_band) == pytest.approx(0.0)
    assert item_crops.card_is_centered(
        dead_centre, frame_height=height, content_band=pinned_header_band)

    # The DENOMINATOR, by contrast, stays the band's height on purpose (see
    # `card_center_offset_frac`): `STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC` is a tolerance chosen in
    # band units, and it is the unit every recorded `center_offset_frac` and every `offset *
    # band_px` corrective scroll is written in. 180px below centre on the calibrated 1800px band
    # is 0.100, never 180/2400 = 0.075.
    below = (_CARD_X0, height // 2 + 180 - 400, _CARD_X1, height // 2 + 180 + 400)
    assert item_crops.card_center_offset_frac(
        below, frame_height=height, content_band=_CONTENT_BAND) == pytest.approx(0.100)


@pytest.mark.parametrize("drift", [True, -0.01, float("nan"), float("inf")])
def test_still_photo_gate_rejects_a_corrupt_reobservation_measurement(drift):
    """A BROKEN measurement is not a MISSING one, and only the missing case falls through.

    `None` is deliberately absent from this list: an absent drift is ignorance (a capture that
    never re-observed the crop's rect), which the rungs below answer for.  A bool, a non-finite
    value or a negative distance is a measurement that went wrong, and a wrong measurement can
    never buy the pass-through the missing one gets.
    """
    reason = item_crops.unnumber_without_still_photo_evidence(_evidence(drift, (1,)))

    assert reason is not None
    assert "auto-hidden video cannot be ruled out" in reason


def test_still_photo_gate_rejects_anything_that_is_not_an_evidence_object():
    """A caller that still passes a bare drift number has not been ported, and a number is not
    evidence.  Refusing rather than duck-typing is what stops the old two-argument reading --
    "drift is low, therefore still" -- from surviving in some unported corner."""
    for stale in (0.0, (0.0, (2,)), None, object()):
        reason = item_crops.unnumber_without_still_photo_evidence(stale)
        assert reason is not None
        assert "auto-hidden video cannot be ruled out" in reason


def test_still_photo_gate_uses_drift_only_to_reject_and_never_as_positive_proof(
        installed_still_photo_bound):
    """Drift may only ever REJECT, and the assertions have to reach the drift rung to show it.

    The licence is what makes this test about drift at all.  Without one, every call here is
    answered by the policy blocker two rungs down and "it refused" would be true no matter what
    the drift rung did -- the assertion would pass while testing nothing it names.  With a
    licence installed the blocker is out of the way and each shape below is answered by the rung
    it is aimed at.
    """
    ceiling = item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT

    # A drift measured over ZERO frames is a broken measurement, not a missing one.
    corrupt = item_crops.unnumber_without_still_photo_evidence(_evidence(0.0, ()))
    assert corrupt is not None and "auto-hidden video cannot be ruled out" in corrupt
    # The one thing drift is allowed to say on its own, said by name.
    over = item_crops.unnumber_without_still_photo_evidence(_evidence(ceiling + 1e-6))
    assert over is not None
    assert (f"exceeds the measured static-photo ceiling {ceiling:.6g}") in over
    assert "animated or video media is not targetable" in over
    # And what it may never say: a drift UNDER the ceiling is not proof of anything.  It passes
    # its own rung and hands the question straight to the dwell, which never ran at all here
    # (no `dwell=` given) -- the NEVER OBSERVED wording, not the measured-and-refused one below.
    silent = item_crops.unnumber_without_still_photo_evidence(_evidence(ceiling))
    assert silent is not None and "never observed" in silent
    # Same with a perfect drift beside a dwell that failed: the refusal is the dwell's, and no
    # amount of clean re-observation buys it off.
    spoiled = item_crops.unnumber_without_still_photo_evidence(
        _evidence(0.0, dwell=_full_dwell(exact=False)))
    assert spoiled is not None and "no un-interacted dwell proved" in spoiled


@pytest.fixture
def installed_still_photo_bound():
    """Numbering readiness exactly as config.validate() installs it, dropped again at teardown.

    Readiness is process-global (ops/STILL-PHOTO-DISCRIMINATOR.md section 5), so it is installed
    through the real API rather than monkeypatched, and always torn down.
    """
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL,
        human_ground_truth=True, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=3, max_video_exact_run_s=1.5, artifact_sha256="a" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1"))
    yield
    tp._reset_installed_still_photo_bound_for_tests()


def test_drift_alone_is_never_positive_proof_even_with_a_bound_installed(
        installed_still_photo_bound):
    """The heart of the design: a bound licenses the RULE, never an individual card.  With a
    verified bound installed and nothing but a clean drift measurement, the gate still refuses --
    naming the dwell it never took, not the policy blocker it no longer has."""
    ceiling = item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT

    reason = item_crops.unnumber_without_still_photo_evidence(_evidence(ceiling))

    assert reason is not None
    assert reason == item_crops.EXCLUSION_NEVER_DWELLED


def test_the_still_photo_gate_stops_refusing_only_once_a_verified_bound_is_installed(
        installed_still_photo_bound):
    """The policy blocker is the middle check, so the rejectors above it are unaffected by it."""
    ceiling = item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT

    assert item_crops.unnumber_without_still_photo_evidence(
        _evidence(ceiling, dwell=_full_dwell())) is None
    # Readiness licenses numbering; it never excuses missing or failing re-observation evidence.
    drifted = item_crops.unnumber_without_still_photo_evidence(
        _evidence(ceiling + 1e-6, dwell=_full_dwell()))
    assert drifted is not None and "animated or video media is not targetable" in drifted
    # What readiness does NOT excuse is a FAILING dwell; a MISSING drift is a different thing and
    # is covered by `test_an_unmeasurable_drift_falls_through_to_the_rungs_that_carry_the_proof`.
    assert item_crops.unnumber_without_still_photo_evidence(
        _evidence(ceiling, dwell=_full_dwell(exact=False))) is not None


def test_an_unmeasurable_drift_falls_through_to_the_rungs_that_carry_the_proof(
        installed_still_photo_bound):
    """The depth-1 case, and the reason this rung stopped refusing on silence.

    Drift is measured by resampling a crop's page rows in the OTHER frames of the capture, so
    for the FIRST card of a top-down read there is nothing to measure against -- no other
    frame's analysed band contains those rows -- and `_signature_drift` returns `(None, ())` by
    construction.  Refusing on that made every depth-1 target permanently unnumberable while
    proving nothing: a complete dwell plus a complete re-attach burst is a strictly stronger
    independent re-observation than a drift number ever was.

    `(None, ())` is the ONLY shape that falls through, and it is the exact pair the producers
    emit for that case.  A drift without its frames, or frames without their drift, is an
    inconsistent object rather than a silent one and is refused by
    `test_a_corrupt_drift_is_refused_rather_than_treated_as_unmeasurable`.

    Unchanged by the per-regime ceilings, and asserted across all three: silence is judged
    before any ceiling is chosen, so a measurement nobody made cannot exceed a number nobody
    compared it against, whichever regime the producer declares.
    """
    for regime in (None, item_crops.DRIFT_REGIME_PARKED, item_crops.DRIFT_REGIME_READ_SCROLL):
        assert item_crops.unnumber_without_still_photo_evidence(
            item_crops.still_photo_evidence_from_drift(
                None, (), _full_dwell(), drift_regime=regime)) is None, regime


def test_a_measured_drift_over_the_ceiling_still_refuses_however_perfect_the_dwell(
        installed_still_photo_bound):
    """A card that was OBSERVED to move is disqualified, and no amount of later stillness undoes
    it: byte-exactness measured after the fact cannot un-see the motion the read already saw."""
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT + 1e-6, dwell=_full_dwell()))

    assert reason is not None
    assert (f"exceeds the measured static-photo ceiling "
            f"{item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT:.6g}") in reason
    assert "animated or video media is not targetable" in reason


def test_a_read_scroll_drift_over_the_parked_ceiling_is_numbered_now_that_it_is_judged_as_one(
        installed_still_photo_bound):
    """THE CATEGORY ERROR THIS SPLIT FIXES.  `_signature_drift` compares a crop against frames
    taken at OTHER scroll positions, where the card was re-rasterised at a different sub-pixel
    offset, so what it measures is resampling difference rather than motion.  Judging it at the
    parked ceiling -- which was measured where rect error is zero and anything reported IS
    motion -- refused real photographs with the words "animated or video media", and an offline
    replay of ops/calibration/ put 25 of 46 measurable real read-scroll drifts above 0.24
    (median 0.285, max 27.2).

    So a read-scroll drift in the band BETWEEN the two ceilings, with otherwise complete parked
    dwell evidence, is now accepted where it was previously refused.  The band is not empty and
    is not narrow: it is where the median real photograph lives.
    """
    parked = item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT
    read_scroll = item_crops._READ_SCROLL_MAX_SIGNATURE_DRIFT
    assert read_scroll > parked, "the split is pointless unless the read-scroll band exists"

    for drift in (parked + 1e-6, 0.285, 2.15, 27.22, read_scroll):
        assert item_crops.unnumber_without_still_photo_evidence(
            _evidence(drift, dwell=_full_dwell(),
                      regime=item_crops.DRIFT_REGIME_READ_SCROLL)) is None, drift
        # The pin: the SAME number, declared parked, is still a moving card.
        assert item_crops.unnumber_without_still_photo_evidence(
            _evidence(drift, dwell=_full_dwell(),
                      regime=item_crops.DRIFT_REGIME_PARKED)) is not None, drift


def test_a_read_scroll_drift_above_the_read_scroll_ceiling_is_still_refused_as_animated(
        installed_still_photo_bound):
    """Widening the band is not removing the rung.  A card whose pixels turned into DIFFERENT
    pixels between enumeration frames is still rejected, with the same operator-facing sentence,
    and the reason names the read-scroll ceiling it actually failed rather than a number nobody
    compared it against."""
    ceiling = item_crops._READ_SCROLL_MAX_SIGNATURE_DRIFT

    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(ceiling + 1e-6, dwell=_full_dwell(),
                  regime=item_crops.DRIFT_REGIME_READ_SCROLL))

    assert reason is not None
    assert f"exceeds the measured static-photo ceiling {ceiling:.6g}" in reason
    assert item_crops.DRIFT_REGIME_READ_SCROLL in reason
    assert "animated or video media is not targetable" in reason


def test_the_parked_ceiling_is_not_loosened_by_the_read_scroll_ceiling_existing(
        installed_still_photo_bound):
    """THE PIN.  `_verified_still_photo_proof` -- the sole licence for touching a heart --
    measures drift over its own PARKED re-observations, where rect error is zero by construction
    and any distance at all is motion in the media.  That population still answers to 0.24, so
    adding a second constant for a second regime must be provably invisible to it: the same
    drift that a read-scroll object now passes on still refuses when it is parked."""
    parked = item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT
    assert parked == 0.24, "the parked ceiling is measured evidence, not a tunable"

    assert item_crops.unnumber_without_still_photo_evidence(
        _evidence(parked, dwell=_full_dwell(),
                  regime=item_crops.DRIFT_REGIME_PARKED)) is None

    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(parked + 1e-6, dwell=_full_dwell(),
                  regime=item_crops.DRIFT_REGIME_PARKED))
    assert reason is not None
    assert f"exceeds the measured static-photo ceiling {parked:.6g}" in reason
    assert item_crops.DRIFT_REGIME_PARKED in reason
    assert "animated or video media is not targetable" in reason


@pytest.mark.parametrize("regime", [None, "", "read scroll", "READ_SCROLL", "parked_burst", 7],
                         ids=["undeclared", "empty", "spaced", "cased", "invented", "not-a-string"])
def test_an_undeclared_or_unrecognised_drift_regime_takes_the_stricter_parked_ceiling(
        installed_still_photo_bound, regime):
    """Unknown provenance must never win the looser bound.  A producer that did not say where
    its drift came from -- and a producer that said something this module has not been taught,
    including a near-miss spelling of the real name -- has not earned the resampling allowance,
    so both fall to 0.24.  Written as the failing direction on purpose: the safe default is only
    a safe default if getting it wrong REFUSES."""
    between = (item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT
               + item_crops._READ_SCROLL_MAX_SIGNATURE_DRIFT) / 2

    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(between, dwell=_full_dwell(), regime=regime))

    assert reason is not None
    assert (f"exceeds the measured static-photo ceiling "
            f"{item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT:.6g}") in reason
    assert "animated or video media is not targetable" in reason


@pytest.mark.parametrize(
    "regime",
    [None, "parked", "read_scroll"],
    ids=["undeclared", "parked", "read-scroll"])
@pytest.mark.parametrize(
    "drift, drift_frames",
    [(float("nan"), (1,)), (float("inf"), (1,)), (-0.01, (1,)), (True, (1,)),
     (0.0, ()), (None, (2,))],
    ids=["nan", "inf", "negative", "bool", "drift-without-frames", "frames-without-drift"])
def test_a_corrupt_drift_is_refused_rather_than_treated_as_unmeasurable(
        installed_still_photo_bound, drift, drift_frames, regime):
    """The line the pass-through must not cross.  A measurement that came back broken is not the
    same claim as a measurement nobody made, and only the second one may fall through -- reading
    a NaN as "nothing to compare against" is how a fail-closed gate quietly becomes fail-open.

    The last two cases are broken in SHAPE rather than in value.  A drift measured over an EMPTY
    frame list is a distance computed against nothing, and a frame list with no drift behind it
    is a re-observation that produced no measurement; neither is the `(None, ())` silence the
    rung forgives.  No producer in tree emits either shape (`_signature_drift` and the calibrate
    tool's `_parked_signature_drift` both return `(None, ())` or `(float, non-empty)`), which is
    exactly why the gate has to say so itself: since the pass-through shipped, nothing else in
    the ladder reads `drift_frames` at all.

    Swept across every regime because the corrupt rung sits ABOVE the ceiling choice and must
    stay there: a NaN is not a number any ceiling can be compared against, so declaring a
    read-scroll provenance must not turn a broken measurement into a passing one.
    """
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(drift, drift_frames, dwell=_full_dwell(), regime=regime))

    assert reason is not None
    assert "auto-hidden video cannot be ruled out" in reason


def test_without_a_licence_an_unmeasurable_drift_still_stops_at_the_policy_rung():
    """Unlicensed behaviour is UNCHANGED by the pass-through: with no artifact installed the
    blocker answers before any dwell rung is consulted, so a build with no discriminator cannot
    number a card merely because its drift could not be measured."""
    reason = item_crops.unnumber_without_still_photo_evidence(
        item_crops.still_photo_evidence_from_drift(None, (), _full_dwell()))

    assert reason is not None
    assert "positive still-photo discriminator unavailable" in reason


@pytest.mark.parametrize(
    ("dwell", "expected"),
    [
        (_full_dwell(exact=None), "no un-interacted dwell proved"),
        (_full_dwell(exact=False), "no un-interacted dwell proved"),
        (_full_dwell(exact="yes"), "no un-interacted dwell proved"),
        (_full_dwell(digests=1), "no un-interacted dwell proved"),
        (_full_dwell(screened=None), "mute-control screening did not complete"),
        (_full_dwell(screened=False), "mute-control screening did not complete"),
        (_full_dwell(screened=1), "mute-control screening did not complete"),
        (_full_dwell(span_s=None), "dwell window is missing or non-positive"),
        (_full_dwell(span_s=0.0), "dwell window is missing or non-positive"),
        (_full_dwell(span_s=-1.0), "dwell window is missing or non-positive"),
        (_full_dwell(span_s=float("nan")), "dwell window is missing or non-positive"),
        (_full_dwell(span_s=True), "dwell window is missing or non-positive"),
    ],
    ids=["exact-none", "exact-false", "exact-truthy", "one-digest", "screen-none",
         "screen-false", "screen-truthy", "span-none", "span-zero", "span-negative",
         "span-nan", "span-bool"],
)
def test_every_dwell_rung_of_the_evidence_ladder_is_reachable_and_says_which_one_failed(
        installed_still_photo_bound, dwell, expected):
    """One spoiled field at a time, with everything else passing, so each rung is proven to be
    load-bearing rather than shadowed by an earlier one.  `is not True` matters here: a truthy
    placeholder must not be able to claim an observation nothing made."""
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT, dwell=dwell))

    assert reason is not None and expected in reason
    assert reason.startswith(item_crops.EXCLUSION_NON_PHOTO + ":")


def test_never_dwelled_reads_differently_from_dwelled_and_refused(installed_still_photo_bound):
    """found+fixed 2026-08-22 (ops/STILL-PHOTO-DISCRIMINATOR.md 5d): "we never looked at this
    card" (a structural coverage gap that happens on nearly every real profile -- production
    dwells once, at wherever the read stopped) must not read as the same sentence as "we looked
    and it moved" (an actual measurement). Both still refuse -- neither is weakened by one
    micron -- but only a dwell with BOTH `dwell_exact is None` AND no frames behind it is the
    never-observed case; `dwell_exact is None` with frames present, or a measured `False`, keep
    the original wording because something really was observed."""
    ceiling = item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT

    never_dwelled = item_crops.unnumber_without_still_photo_evidence(_evidence(ceiling))
    assert never_dwelled == item_crops.EXCLUSION_NEVER_DWELLED

    measured_but_inconclusive = item_crops.unnumber_without_still_photo_evidence(
        _evidence(ceiling, dwell=_full_dwell(exact=None)))
    assert measured_but_inconclusive is not None
    assert measured_but_inconclusive != item_crops.EXCLUSION_NEVER_DWELLED
    assert "no un-interacted dwell proved" in measured_but_inconclusive

    measured_and_moved = item_crops.unnumber_without_still_photo_evidence(
        _evidence(ceiling, dwell=_full_dwell(exact=False)))
    assert measured_and_moved is not None
    assert measured_and_moved != item_crops.EXCLUSION_NEVER_DWELLED
    assert "no un-interacted dwell proved" in measured_and_moved

    # Both refuse -- the EXCLUSION_NON_PHOTO prefix and the gate's decision are unchanged, only
    # the wording differs.
    for reason in (never_dwelled, measured_but_inconclusive, measured_and_moved):
        assert reason.startswith(item_crops.EXCLUSION_NON_PHOTO)


@pytest.mark.parametrize(
    ("dwell", "expected"),
    [
        (_full_dwell(probe_ran=None), "no re-attach probe scrolled this card"),
        (_full_dwell(probe_ran=False), "no re-attach probe scrolled this card"),
        (_full_dwell(probe_ran=1), "no re-attach probe scrolled this card"),
        (_full_dwell(probe_exact=None), "second dwell burst"),
        (_full_dwell(probe_exact=False), "second dwell burst"),
        (_full_dwell(probe_exact="yes"), "second dwell burst"),
        (_full_dwell(probe_digests=1), "second dwell burst"),
        (_full_dwell(probe_centered=None), "did not put this card back inside"),
        (_full_dwell(probe_centered=False, probe_offset=0.42), "did not put this card back"),
        (_full_dwell(probe_centered=1), "did not put this card back inside"),
        (_full_dwell(probe_screened=None), "every re-attach burst frame"),
        (_full_dwell(probe_screened=False), "every re-attach burst frame"),
        (_full_dwell(probe_screened=1), "every re-attach burst frame"),
        (_full_dwell(probe_span_s=None), "re-attach dwell window is missing"),
        (_full_dwell(probe_span_s=0.0), "re-attach dwell window is missing"),
        (_full_dwell(probe_span_s=float("nan")), "re-attach dwell window is missing"),
        (_full_dwell(probe_span_s=True), "re-attach dwell window is missing"),
    ],
    ids=["probe-none", "probe-false", "probe-truthy", "exact-none", "exact-false",
         "exact-truthy", "one-digest", "centred-none", "centred-false", "centred-truthy",
         "screen-none", "screen-false", "screen-truthy", "span-none", "span-zero", "span-nan",
         "span-bool"],
)
def test_the_reattach_rung_refuses_the_stalled_video_the_dwell_cannot_see(
        installed_still_photo_bound, dwell, expected):
    """The residual rung (g) exists for.  A centred byte-exact burst proves Hinge was ASKED to
    play the card and that nothing moved; it cannot prove the media ANSWERED, so a video that
    was stalled, buffering, unloaded or already ended holds byte-exact for as long as anyone
    watches.  Every leg of the second burst is spoiled one at a time, with the first burst
    passing, so each is proven load-bearing rather than shadowed."""
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT, dwell=dwell))

    assert reason is not None and expected in reason
    assert reason.startswith(item_crops.EXCLUSION_NON_PHOTO + ":")


def test_the_missing_probe_names_the_stalled_video_residual_it_exists_to_catch():
    """The wording is the contract: an operator reading this refusal has to learn WHY a card
    that never moved is still not a photograph."""
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL,
        human_ground_truth=True, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=3, max_video_exact_run_s=1.5, artifact_sha256="e" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1"))
    try:
        reason = item_crops.unnumber_without_still_photo_evidence(
            _evidence(0.01, dwell=_full_dwell(probe_ran=None)))
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "autoplay band and back into it" in reason
    for residual in ("stalled", "buffering", "unloaded", "already ended"):
        assert residual in reason


def test_the_reattach_rung_is_the_last_word_and_never_speaks_over_an_earlier_one(
        installed_still_photo_bound):
    """Order is contract: the probe costs two real gestures, so it answers only once every
    cheaper, always-available rung has passed."""
    for spoiled, expected in (
            (dict(exact=False), "no un-interacted dwell proved"),
            (dict(centered=False), "autoplay trigger zone"),
            (dict(screened=False), "mute-control screening did not complete"),
            (dict(span_s=0.0), "dwell window is missing or non-positive"),
    ):
        reason = item_crops.unnumber_without_still_photo_evidence(
            _evidence(item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT,
                      dwell=_full_dwell(probe_ran=None, **spoiled)))
        assert reason is not None and expected in reason
        assert "re-attach" not in reason


def test_the_ladder_answers_with_the_cheapest_failing_rung_first(installed_still_photo_bound):
    """Order is part of the contract: an animated card is reported as animated whatever the
    dwell did, so the operator-facing vocabulary does not change with the bound."""
    animated = item_crops.unnumber_without_still_photo_evidence(
        _evidence(item_crops._STILL_PHOTO_MAX_SIGNATURE_DRIFT + 1e-6,
                  dwell=_full_dwell(exact=False)))

    assert animated is not None and "animated or video media is not targetable" in animated


def test_the_policy_blocker_outranks_the_dwell_rungs_when_no_bound_is_installed():
    """With no artifact the operator's fix is the campaign, not the dwell, so the blocker is
    what the reason names -- and rungs (d)-(f) stay unreachable, exactly as designed."""
    reason = item_crops.unnumber_without_still_photo_evidence(
        _evidence(0.0, dwell=_full_dwell(exact=False)))

    assert reason is not None
    assert "positive still-photo discriminator unavailable" in reason


def test_payload_disables_targeting_when_no_positive_still_photo_proof_exists():
    """Neither stable nor moving photographic pixels can prove still media."""
    frames, index, _ = _animated_capture(delta=60)

    payload = item_crops.build_item_payload(
        frames, index,
        unnumber=lambda _image: None,
        unnumber_without_evidence=item_crops.unnumber_without_still_photo_evidence)

    assert not payload.usable
    assert payload.translation == ()
    demoted = next(crop for crop in payload.context if crop.heart_ordinal == 2)
    assert demoted.number is None
    assert "animated or video media is not targetable" in demoted.reason
    assert any("positive still-photo discriminator unavailable" in crop.reason
               for crop in payload.context if crop.heart_ordinal != 2)


# =====================================================================================
# C2: the dwell, which is the only POSITIVE observation the discriminator makes
# =====================================================================================

def test_dwell_exactness_is_byte_equality_and_one_pixel_breaks_it():
    """Zero tolerance over the rect, and only over the rect.  The one-pixel case is the point:
    an autoplaying video's strips match CONFIDENTLY WRONG rather than weakly, so any score-based
    test goes bimodal on exactly the population this has to catch."""
    rect = (100, 200, 400, 700)
    still = _frame(0)
    moved = np.frombuffer(cv2.imdecode(np.frombuffer(still, np.uint8), cv2.IMREAD_COLOR),
                          np.uint8).reshape(_H, _W, 3).copy()
    moved[300, 200, 0] = 255 - moved[300, 200, 0]
    ok, buf = cv2.imencode(".png", moved)
    assert ok

    assert item_crops.dwell_exact_over_rect([still, still, still], rect) is True
    assert item_crops.dwell_exact_over_rect([still, buf.tobytes()], rect) is False
    # The changed pixel is INSIDE the rect above and outside this one, so the same pair of
    # frames answers differently -- the rect is doing the work, not a whole-frame compare.
    assert item_crops.dwell_exact_over_rect([still, buf.tobytes()], (500, 900, 900, 1400)) is True


def test_a_dwell_that_never_happened_is_not_an_exact_dwell():
    """Fewer than two frames means zero consecutive pairs, and a vacuous "all pairs matched" is
    exactly the unearned True this whole design exists to prevent."""
    rect = (100, 200, 400, 700)

    assert item_crops.dwell_exact_over_rect([], rect) is False
    assert item_crops.dwell_exact_over_rect([_frame(0)], rect) is False
    # Undecodable bytes are a failed observation, never a passing one.
    assert item_crops.dwell_exact_over_rect([b"not-a-png", b"not-a-png"], rect) is False
    # A rect off the end of the frame is likewise unobserved, not equal-by-emptiness.
    assert item_crops.dwell_exact_over_rect([_frame(0), _frame(0)], (0, 0, _W + 1, 10)) is False


@pytest.mark.parametrize("rect", [(10, 10, 10, 20), (10, 10, 20, 10), (-1, 0, 20, 20),
                                  (0, 0, 20), "rect", (0.0, 0.0, 20.0, 20.0)])
def test_a_structurally_invalid_dwell_rect_is_a_caller_bug_and_raises(rect):
    """A degenerate rect is not an observation that failed; it is a caller that never had one.
    Returning False would let it read as a measured refusal in a log."""
    with pytest.raises(item_crops.ItemCropError, match="dwell rect"):
        item_crops.dwell_exact_over_rect([_frame(0), _frame(0)], rect)


def test_dwell_evidence_covers_only_the_cards_the_anchor_frame_bounded():
    """A card that scrolled out of view before the read ended cannot be held still and looked
    at, so it gets NO entry -- rather than an entry measured on the last rect it happened to
    occupy, which would be a dwell of the wrong pixels."""
    index, frames = _full()
    dwell = item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1], frames[-1]], dwell_span_s=6.0,
        mute_screen=lambda _frame, _rect: True)

    anchor_cards = {block.heart_ordinal for block in index.selectable
                    if any(obs.frame_index == len(frames) - 1 for obs in block.croppable)}
    assert set(dwell) == anchor_cards and anchor_cards
    for entry in dwell.values():
        assert entry.dwell_exact is True
        assert entry.mute_screens_complete is True
        assert entry.dwell_span_s == 6.0
        # The anchor frame is chained in front of the burst, so its digest leads the list.
        assert entry.dwell_frame_sha256s[0] == hashlib.sha256(frames[-1]).hexdigest()
        assert len(entry.dwell_frame_sha256s) == 3


def test_dwell_evidence_without_a_mute_screen_leaves_that_leg_unmeasured():
    """None, never False and never True: C3 either ran on every frame or it did not run."""
    index, frames = _full()

    dwell = item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1]], dwell_span_s=6.0)

    assert dwell and all(e.mute_screens_complete is None for e in dwell.values())


def test_no_dwell_frames_means_no_dwell_evidence_at_all():
    index, frames = _full()

    assert item_crops.still_photo_dwell_evidence(
        index, frames, [], dwell_span_s=6.0) == {}


def test_a_bound_plus_a_complete_dwell_is_what_finally_numbers_a_photo(
        installed_still_photo_bound):
    """The whole C1-C4 conjunction, end to end through the payload builder: with a verified
    bound installed AND a byte-exact, fully screened dwell of every candidate, numbering is
    allowed.  Nothing short of that has ever produced a number in this test file."""
    index, frames = _full()
    dwell = item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1], frames[-1]], dwell_span_s=6.0,
        mute_screen=lambda _frame, _rect: True)
    # This test is about the C1-C4 conjunction, not about where the synthetic cards happen to sit
    # on a synthetic screen. The autoplay-centring precondition is measured from real geometry in
    # `test_still_photo_dwell_evidence_measures_centring_from_the_content_band` below, and its
    # refusal path has its own parametrised test; here it is satisfied explicitly so the rest of
    # the ladder stays the subject.
    dwell = {ordinal: dataclasses.replace(entry, centered=True, center_offset_frac=0.0)
             for ordinal, entry in dwell.items()}
    # Rung (g): the re-attach probe's second burst, folded in by the shared producer from a probe
    # whose MEASURED page displacement is zero -- the card came back exactly where it was.
    dwell = _fold_reattach(dwell, index, frames)
    dwell = {ordinal: dataclasses.replace(entry, reattach_centered=True,
                                          reattach_center_offset_frac=0.0)
             for ordinal, entry in dwell.items()}

    payload = item_crops.build_item_payload(
        frames, index, unnumber=lambda _image: None,
        unnumber_without_evidence=item_crops.unnumber_without_still_photo_evidence,
        still_photo_dwell=dwell)

    numbered = {crop.heart_ordinal for crop in payload.items}
    assert numbered, "a complete dwell under a verified bound must number something"
    # And every card the dwell could NOT cover stays unnumbered, with the NEVER OBSERVED wording
    # (found+fixed 2026-08-22) rather than the measured-and-refused one: nothing dwelled these
    # cards at all, so they were never judged.
    for crop in payload.context:
        if crop.heart_ordinal is not None and crop.heart_ordinal not in dwell:
            assert crop.reason == item_crops.EXCLUSION_NEVER_DWELLED


def test_the_payload_hook_receives_one_evidence_object_carrying_both_halves(
        installed_still_photo_bound):
    """The hook contract itself: drift comes from the frames the builder holds, the dwell from
    the mapping the caller supplies, and they arrive together so no caller can grade one alone.

    And the builder must DECLARE its drift's provenance.  Its number came from `_signature_drift`
    resampling the crop's page rows in frames taken at other scroll positions, so it is a
    read-scroll measurement and is only judged correctly if it says so -- an object that stayed
    silent would take the parked ceiling and reproduce the refusals this split removed.  The
    wiring is asserted here rather than inferred, because nothing downstream can recover the
    regime from the numbers.
    """
    index, frames = _full()
    dwell = item_crops.still_photo_dwell_evidence(
        index, frames, [frames[-1], frames[-1]], dwell_span_s=6.0,
        mute_screen=lambda _frame, _rect: True)
    seen = []

    item_crops.build_item_payload(
        frames, index, unnumber=lambda _image: None,
        unnumber_without_evidence=lambda evidence: seen.append(evidence) or "photo_only: stop",
        still_photo_dwell=dwell)

    assert seen and all(isinstance(e, item_crops.StillPhotoEvidence) for e in seen)
    assert any(e.dwell_exact is True and e.dwell_span_s == 6.0 for e in seen)
    assert all(e.drift_frames == () or e.signature_drift is not None for e in seen)
    assert all(e.drift_regime == item_crops.DRIFT_REGIME_READ_SCROLL for e in seen)


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


def test_a_crop_carries_a_greyscale_reference_that_reproduces_its_signature_exactly():
    """The verification reference is a THIRD artefact, cut from the same rows as the other two.

    `image` is a colour re-encode for the model and cannot double as doc 5.6's reference: greying
    it means `IMREAD_GRAYSCALE` on a PNG this codebase wrote, while the sheet it will be compared
    against is `IMREAD_GRAYSCALE` on the PNG the DEVICE wrote. Those are the two paths this
    module's docstring calls a measured trap, and on 2026-08-27 they put a live comparison in
    different units and refused a correct card at 10.283 against a 7.00 ceiling.

    So the reference is carried out of `_crop_image` directly, as a lossless 8-bit greyscale PNG:
    it must decode back with no colour conversion anywhere in the path, which makes reproducing
    the stored signature exact rather than merely close.
    """
    payload = _payload()
    for crop in payload.items + payload.context:
        assert crop.verify_image, crop.reason
        decoded = cv2.imdecode(np.frombuffer(crop.verify_image, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert decoded.ndim == 2
        colour = cv2.imdecode(np.frombuffer(crop.image, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == colour.shape[:2]
        if crop.signature is not None:
            reproduced = item_crops._signature_from_gray(
                decoded, grid=payload.signature_grid, cv2=cv2, np=np)
            assert crop.signature.distance(reproduced) == 0.0, crop.reason


def test_a_block_with_no_image_carries_no_verification_reference_either():
    """`verify_image` follows `image` exactly: withheld blocks have neither."""
    payload = _payload()
    for crop in payload.excluded + payload.uncroppable:
        assert crop.image is None and crop.verify_image is None, crop.reason
