"""Post-tap sheet verification (operation_love/drivers/item_verify.py, ops/OPENER-REDESIGN.md 5.6).

Every frame here is SYNTHESISED, never a real screencap, for the same reason test_item_crops.py
and test_item_index.py synthesise theirs: the captures the constants were measured against are
real people's dating profiles and are gitignored (`ops/calibration/` and `data/`), so only
geometry and grey levels from them appear anywhere in this repo.

The fixture builds the same scrollable WORLD those files build — gradient page background,
rounded-rect cards inset 53px per side, canonical gutters, the shipped like glyph on the likeable
ones — indexes it and crops it with the real `build_item_payload`, and then RE-RENDERS a chosen
card the way the real comment sheet was measured to render it: uniformly scaled to a 890px-wide
preview indented to column 95, anchored at the card's BOTTOM when the card is taller than the
sheet's content region. So the sheet a test hands the verifier is a genuine re-render of a genuine
crop, and the distances the assertions rest on are produced by the shipped comparison rather than
stipulated.

Every positive is paired with its negative: for each thing that verifies there is a test that the
eight other items do NOT, that a wrongly anchored render does not, and that an item whose own
crop cannot serve as a reference is refused BEFORE anything is tapped.
"""
import dataclasses
import math
from unittest import mock
import zlib

import cv2
import numpy as np
import pytest

from operation_love.drivers import hinge, item_crops, item_index, item_verify, segment
from operation_love.drivers.like_composer import ComposerSurface, Rect

_W, _H = 1080, 2400                                    # the calibrated Pixel 7a screencap size
_SEED = 31
_CONTENT_BAND = hinge.HINGE_SPEC.content_band          # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - _CARD_X0                               # 1027
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px"
_HEART_CX = 937
_HEART_ABOVE_BOTTOM = 90
_PAGE_TOP, _PAGE_BOTTOM = 254, 243                     # doc 5.10's background gradient
_STEP = 363                                            # doc 5.10.1's measured clean cadence
_WORLD_H = 6400

# THE PAGE. Four likeable cards with the heartless vitals block between the second and third, so
# the model's items are 1..4 while the hearts are 1,2,3,4 and the context block takes no number.
# The last card is 1109px tall, which is TALLER than the sheet's 856px content region once
# scaled, so it is the one that exercises the bottom-anchored window rather than a whole-card
# render.
_PAGE_TOP_GAP = 1200
_LAYOUT = (("card", 900), ("card", 760), ("context", 215), ("card", 1000), ("card", 1109))

# --- the comment sheet, as measured on the six real like screens ----------------------
# Numbers only; see item_verify's module docstring for the table they come from.
_SHEET_BG = 250
_SHEET_PREVIEW_X0 = 95
_SHEET_PREVIEW_W = 890
_SHEET_PREVIEW_Y0 = 236
_SHEET_PREVIEW_MAX_H = 856

_TEMPLATE = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"


def _layout_rows():
    rows, y = [], _PAGE_TOP_GAP
    for kind, height in _LAYOUT:
        rows.append((kind, y, y + height))
        y += height + _GUTTER
    return rows


_ROWS = _layout_rows()


def _page_column(height=_WORLD_H):
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, height).round().astype(np.uint8)


def _build_world():
    rng = np.random.default_rng(_SEED)
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    for position, (_kind, y0, y1) in enumerate(_ROWS):
        # A different mean per card, on top of the noise, so the cards sit far apart in signature
        # space the way real photographs do rather than being five draws from one distribution.
        height, width = y1 - y0, _CARD_X1 - _CARD_X0
        rows, cols = np.mgrid[0:height, 0:width]
        # A distinct low-frequency pattern per card, so the five of them sit far apart in
        # signature space the way five photographs do rather than being five draws from one
        # distribution -- plus a VERTICAL RAMP, so each card's top half and bottom half are
        # different pictures. The ramp is what makes the bottom-anchored window rule testable at
        # all: on a card with no vertical structure the top window and the bottom window reduce
        # to the same signature, so a wrongly anchored render would verify and prove nothing.
        pattern = (55 * np.sin(2 * np.pi * rows / (70 + 25 * position))
                   * np.cos(2 * np.pi * cols / (110 + 40 * position)))
        card = (100 + pattern
                + np.linspace(-30, 30, height)[:, None]
                + rng.integers(-25, 26, size=(height, width)))
        world[y0:y1, _CARD_X0:_CARD_X1] = np.clip(card, 0, 210).astype(np.uint8)
        for i in range(_CORNER_RADIUS_PX):             # carve the corner arcs back to page
            dy = _CORNER_RADIUS_PX - i
            inset = int(math.ceil(_CORNER_RADIUS_PX
                                  - math.sqrt(max(0.0, _CORNER_RADIUS_PX ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for y in (y0 + i, y1 - 1 - i):
                world[y, _CARD_X0:_CARD_X0 + inset] = col[y]
                world[y, _CARD_X1 - inset:_CARD_X1] = col[y]
        if _kind != "card":
            continue
        th, tw = _TEMPLATE.shape
        cy = y1 - _HEART_ABOVE_BOTTOM
        world[cy - th // 2: cy - th // 2 + th,
              _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = _TEMPLATE
    return world


_WORLD = _build_world()
_CHROME_TOP = np.random.default_rng(3).integers(0, 60, size=(_BAND0, _W), dtype=np.uint8)
_CHROME_BOTTOM = np.random.default_rng(4).integers(0, 60, size=(_H - _BAND1, _W), dtype=np.uint8)
_FRAMES: dict[int, bytes] = {}


def _frame(scroll: int) -> bytes:
    if scroll not in _FRAMES:
        gray = _WORLD[scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAMES[scroll] = buf.tobytes()
    return _FRAMES[scroll]


_SCROLLS = tuple(_STEP * i for i in range(12))
assert _SCROLLS[-1] + _H <= _WORLD_H
_CACHE: dict[str, object] = {}


def _payload():
    """The real index and the real crops, built once — twelve segmentations are not free."""
    if "payload" not in _CACHE:
        frames = [_frame(s) for s in _SCROLLS]
        index = item_index.build_item_index(
            frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)
        assert index.usable, index.failures
        _CACHE["payload"] = item_crops.build_item_payload(frames, index)
    return _CACHE["payload"]


def _recut(crop, card):
    """A crop re-cut from a modified `card` array, all three artefacts together.

    `dataclasses.replace(crop, image=...)` alone is not enough and has not been since the
    verification reference stopped being derived from `image`: a crop carries the model's colour
    image, the greyscale reference doc 5.6 windows, and the stored signature, and the real
    `item_crops._crop_image` cuts all three from the same rows of the same frame. A fixture that
    repaints one of them and leaves the other two describing the previous pixels is not a
    modified card, it is an inconsistent one -- which `item_verify._check_reference_provenance`
    now refuses outright, as it should.

    The greyscale side is decoded FROM the encoded colour card rather than converted from the
    array in memory, because that is what the real path does: the reference is
    `cv2.IMREAD_GRAYSCALE` of the frame, never `cvtColor` of a colour decode of it.
    """
    ok, colour = cv2.imencode(".png", card)
    assert ok
    image = colour.tobytes()
    grey = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    ok, encoded_grey = cv2.imencode(".png", grey)
    assert ok
    verify_image = encoded_grey.tobytes()
    return dataclasses.replace(
        crop, image=image, verify_image=verify_image,
        signature=item_crops.signature_of(
            verify_image, y0=0, y1=grey.shape[0], x0=0, x1=grey.shape[1],
            grid=_payload().signature_grid))


def paint_sheet(crop_image: bytes, *, preview_w=_SHEET_PREVIEW_W, x0=_SHEET_PREVIEW_X0,
                y0=_SHEET_PREVIEW_Y0, max_h=_SHEET_PREVIEW_MAX_H, anchor="bottom") -> bytes:
    """A comment sheet rendering `crop_image` the way the real one was measured to render a card.

    Uniform width scale, and when the scaled card is taller than the sheet's content region the
    part shown is the card's BOTTOM. `anchor="top"` is the deliberate negative: the same render
    with the wrong end kept, which doc 5.6 must refuse rather than accept.
    """
    card = cv2.imdecode(np.frombuffer(crop_image, np.uint8), cv2.IMREAD_COLOR)
    h, w = card.shape[:2]
    scale = preview_w / w
    ph = int(round(h * scale))
    if ph > max_h:
        keep = int(round(max_h / scale))
        card = card[h - keep:h] if anchor == "bottom" else card[:keep]
        ph = max_h
    small = cv2.resize(card, (preview_w, ph), interpolation=cv2.INTER_AREA)
    canvas = np.full((_H, _W, 3), _SHEET_BG, dtype=np.uint8)
    canvas[y0:y0 + ph, x0:x0 + preview_w] = small
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def _sheet_for(number: int, **kw) -> bytes:
    return paint_sheet(_payload().item(number).image, **kw)


def _inline_surface() -> ComposerSurface:
    preview = item_verify.locate_sheet_preview(_sheet_for(1))
    comment = Rect(preview.x0, preview.y1 + 20, preview.x1, preview.y1 + 198)
    send = Rect(390, comment.y1 + 15, 985, comment.y1 + 124)
    return ComposerSurface("hinge_inline_v1", comment, send, (695, send.y0 + 50))


def _paint_inline_reframe(crop_image: bytes, *, start: int = 37, rows: int = 933,
                          preview_height: int = _SHEET_PREVIEW_MAX_H,
                          y0: int = _SHEET_PREVIEW_Y0) -> bytes:
    """Synthetic 9.134 selected-photo composer, including its removed card-heart lane.

    The real Malaika regression is a complete square source card rendered as a 933-row interior
    window above the inline controls (not the legacy bottom window).  Its profile-card heart is
    in the source's lower-right lane; the selected preview intentionally has ordinary image
    pixels there.  This fixture carries the same geometry without embedding a real profile.
    """
    card = cv2.imdecode(np.frombuffer(crop_image, np.uint8), cv2.IMREAD_COLOR)
    source_x1 = card.shape[1] - round(card.shape[1] * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    assert 0 <= start and start + rows <= card.shape[0]
    preview = np.full((preview_height, _SHEET_PREVIEW_W, 3), 145, dtype=np.uint8)
    # The left photo area is exactly the full-width, bounded source window verifier is allowed to
    # search.  The right side represents the layout's selected-photo control-free lane and is
    # excluded symmetrically by the inline verifier.
    left_w = _SHEET_PREVIEW_W - round(
        _SHEET_PREVIEW_W * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    preview[:, :left_w] = cv2.resize(card[start:start + rows, :source_x1],
                                     (left_w, preview_height),
                                     interpolation=cv2.INTER_AREA)
    canvas = np.full((_H, _W, 3), _SHEET_BG, dtype=np.uint8)
    canvas[y0:y0 + preview_height,
           _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = preview
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def _with_inline_suggestion_shelf(frame: bytes, *, filled: bool = False,
                                  pills: int = 2) -> bytes:
    """Paint only the measured 10.1.0 outlined prompt-shelf geometry, never its text."""
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    rectangles = [(185, 1003, 711, 1092), (737, 1003, 1070, 1092)]
    for x0, y0, x1, y1 in rectangles[:pills]:
        cv2.rectangle(image, (x0, y0), (x1, y1), (220, 220, 220),
                      thickness=-1 if filled else 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _right_silent_inline_sheet(crop_image: bytes, *, ink_x0: int = _SHEET_PREVIEW_X0,
                               ink_x1: int = 927, gap_row: int | None = None) -> bytes:
    """A full card whose right rail is visually indistinguishable from page background.

    This is private-free geometry for the 2026-08-31 Hinge 10.1.0 selected-card halt: its
    detected median ink was x=96..927 against the comment field x=95..985.  The card still has
    the full 890px layout width, but only the left attached ink is measurable after the profile
    heart disappears.  `gap_row` creates an unsupported horizontal break for a hard-boundary
    control without changing the composer rectangles.
    """
    frame = paint_sheet(crop_image)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H,
          _SHEET_PREVIEW_X0:ink_x0] = _SHEET_BG
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H,
          ink_x1:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = _SHEET_BG
    if gap_row is not None:
        image[gap_row:gap_row + item_verify._INLINE_PREVIEW_MAX_INTERNAL_GAP_PX + 1,
              ink_x0:ink_x1] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _inline_surface_for(frame: bytes) -> ComposerSurface:
    preview = item_verify.locate_sheet_preview(frame)
    comment = Rect(preview.x0, preview.y1 + 20, preview.x1, preview.y1 + 198)
    send = Rect(390, comment.y1 + 15, 985, comment.y1 + 124)
    return ComposerSurface("hinge_inline_v1", comment, send, (695, send.y0 + 50))


def _review_scrolled_inline_frame(*, selected: int, upper: int):
    """Put an unrelated profile card above a still-open, lower inline composer.

    This is the synthetic geometry of the 2026-08-25 Training refusal: during the review wait,
    scrolling exposed an x=53 profile card above the selected x=95 preview.  The composer itself
    remained open and actionable below the selected photo.
    """
    selected_y0 = 800
    frame = paint_sheet(
        _payload().item(selected).image, y0=selected_y0, max_h=700)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    source = cv2.imdecode(
        np.frombuffer(_payload().item(upper).image, np.uint8), cv2.IMREAD_COLOR)
    upper_y0, upper_height = 236, 328
    image[upper_y0:upper_y0 + upper_height, _CARD_X0:_CARD_X1] = cv2.resize(
        source, (_CARD_X1 - _CARD_X0, upper_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    selected_height = min(
        700, round(_payload().item(selected).signature.height
                   * _SHEET_PREVIEW_W / _payload().item(selected).signature.width))
    comment = Rect(
        _SHEET_PREVIEW_X0, selected_y0 + selected_height + 20,
        _SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W,
        selected_y0 + selected_height + 198)
    send = Rect(390, comment.y1 + 15, 985, comment.y1 + 124)
    return encoded.tobytes(), ComposerSurface(
        "hinge_inline_v1", comment, send, (695, send.y0 + 50))


def _fragment_legacy_wide_runs(frame: bytes) -> bytes:
    """Make sparse bright source rows invisible to the strict 870px row-span probe.

    This is the synthetic form of Tega's bright-sky image: every row still carries a 745px image
    run, but one 145px interval never reaches 870px, so the legacy locator has no 300px run.
    The compact composer fallback may use the 745px evidence only after the controls bind it.
    """
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    for y in range(_SHEET_PREVIEW_Y0 + 145, _SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, 145):
        image[y:y + 1, 840:985] = _SHEET_BG
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


# The Hinge 10.0.1 composer refused live on 2026-08-22, as numbers. Frame sha256 29175f91...,
# 1080x2400, `ops/calibration/targeting_20260822T_cal-a/refused_composer_p1_item1.png`, which is
# gitignored and stays that way: only its geometry is written down here.
_V1001_PREVIEW_ROWS = 856                 # the selected photo really spans rows 236..1092
_V1001_FRAGMENT_ROWS = 620                # ...but the strict 870px floor stopped at row 856
_V1001_NARROWEST_SPAN = 849               # the block's narrowest row, 21px under the 870 floor
_V1001_COMMENT = Rect(95, 1124, 985, 1302)    # comment_rect, as `locate_inline_composer` read it
_V1001_SEND = Rect(390, 1334, 985, 1443)      # send_rect, likewise
_V1001_PHOTO_TO_FIELD_GAP = 32            # 1124 - 1092, the real card -> field gap


def _spend_the_wide_run_headroom(frame: bytes, *, first_row: int, step: int = 145) -> bytes:
    """Return 21px of the preview's right edge to page background on a few interior rows.

    `_PREVIEW_MIN_WIDTH_PX` is 870 against a preview measured at 890, i.e. 20px of headroom, and
    real photo content spends it: on the 10.0.1 frame above, the bright car window in the
    selected photo reaches the right edge, so 29 of the 856 block rows carried only 849..868px of
    non-background span and the strict floor cut one contiguous image into the runs 236..856,
    857..862, 875..883, 893..902, 903..1088.  Unlike `_fragment_legacy_wide_runs` (Tega, whose
    fragments left NO 300-row run at all, so the locator refused outright) the first fragment
    here is 620 rows tall, so the locator succeeds and silently reports a fragment as the whole
    preview.  That is the strictly worse failure and the one this reproduces: same edge, same
    handful of pixels, none of the profile.
    """
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    assert _SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W - 964 == 21
    for y in range(first_row, _SHEET_PREVIEW_Y0 + _V1001_PREVIEW_ROWS, step):
        image[y:y + 1, 964:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = _SHEET_BG
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def _leading_pale_inline_reframe(*, erased_display_px: int = 229):
    """Build the 2026-08-27 failure shape without retaining its private photograph.

    The live selected photo's first 477 displayed rows blended into the page on the left while
    retaining its right edge.  Only the final 412px high-contrast run satisfied the compact
    locator, which made a complete 890px preview look 53.7% cropped.  Modify the synthetic source
    crop itself before rendering so the located frame and its stored reference still describe
    exactly the same pixels; this tests boundary recovery rather than tolerance to altered
    content.
    """
    payload = _payload()
    selected = payload.item(4)
    card = cv2.imdecode(np.frombuffer(selected.image, np.uint8), cv2.IMREAD_COLOR)
    start = 37
    source_rows = 933
    pale_display_rows = 477
    source_x1 = card.shape[1] - round(
        card.shape[1] * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    display_x1 = _SHEET_PREVIEW_W - round(
        _SHEET_PREVIEW_W * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    erased_source_px = math.ceil(erased_display_px * source_x1 / display_x1)
    pale_source_rows = math.ceil(
        pale_display_rows * source_rows / _SHEET_PREVIEW_MAX_H)
    card[start:start + pale_source_rows, :erased_source_px] = _SHEET_BG
    selected = _recut(selected, card)
    payload = dataclasses.replace(
        payload,
        crops=tuple(selected if crop.number == selected.number else crop
                    for crop in payload.crops))
    return payload, _paint_inline_reframe(selected.image)


def test_inline_item_verification_requires_selected_card_above_the_detected_controls():
    frame = _sheet_for(1)
    assert item_verify.verify_sheet_item(
        frame, _payload(), 1, composer_surface=_inline_surface()).matched

    unrelated = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1800, 985, 1978), Rect(390, 2000, 985, 2109),
        (695, 2050))
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify.verify_sheet_item(
            frame, _payload(), 1, composer_surface=unrelated)


def test_malaika_style_inline_reframe_uses_a_bounded_one_item_cap_without_loosening_legacy_or_wrong_cards():
    """Regression for held-out Malaika: a correct selected photo was 6.703, above modal 3.900.

    The companion controls are deliberate: an arbitrary post-tap patch, a foreign photo, and a
    prompt-style card must not enter the new bounded origin sweep, and legacy callers continue to
    use their original bottom-anchor/3.90 fallback.  The source card is tall enough that exactly
    the bounded 933-row full-width reframe is required.
    """
    payload = _payload()
    selected = payload.item(4)  # 1109px, so 933px leaves 176px (16%) hidden.
    frame = _paint_inline_reframe(selected.image)
    surface = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1112, 985, 1290), Rect(390, 1305, 985, 1414),
        (695, 1355))
    # With other indexed choices present, the ordinary unique-nearest proof remains the gate.
    closed_set = item_verify.verify_sheet_item(frame, payload, 4, composer_surface=surface)
    assert closed_set.matched, closed_set.reason
    assert closed_set.nearest_index == 4
    assert closed_set.comparisons[3].nearest_other is not None
    only = dataclasses.replace(payload, crops=(dataclasses.replace(selected, number=1),))

    inline = item_verify.verify_sheet_item(frame, only, 1, composer_surface=surface)
    assert inline.matched, inline.reason
    assert inline.distance < item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST
    assert inline.bound == item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST
    assert "inline composer" in inline.reason
    assert not item_verify.verify_sheet_item(
        frame, only, 1, composer_surface=surface,
        absolute_max_dist=inline.distance / 2).matched
    # The very same selected-card pixels are not evidence of the historical modal geometry.
    assert not item_verify.verify_sheet_item(frame, only, 1).matched

    # A different photo remains far from the selected preview even though the full bounded origin
    # sweep is available; it cannot exploit the reframe search to become a false positive.
    foreign_photo = dataclasses.replace(
        payload, crops=(dataclasses.replace(payload.item(3), number=1),))
    wrong_photo = item_verify.verify_sheet_item(frame, foreign_photo, 1, composer_surface=surface)
    assert not wrong_photo.matched
    assert wrong_photo.distance > item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST

    # The text/prompt-shaped corpus is an independent negative control.  It is purposefully
    # outside the photo-only model path, but verifier safety must not depend on classification.
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    wrong_prompt = item_verify.verify_sheet_item(frame, prompt_only, 1, composer_surface=surface)
    assert not wrong_prompt.matched
    assert wrong_prompt.distance > item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST


def test_alex_style_inline_reframe_allows_the_measured_28_percent_crop_but_not_more():
    """A 731px sheet preview maps to 800 source rows after the fixed control-lane crop.

    The live Alex frame kept 800/1109 rows of the selected portrait (27.86% hidden).  The
    previous 22% cap made the intended item unmeasurable and then let a merely reachable item
    be reported as the one the human opened.  Thirty percent is a bounded renderer envelope,
    not a general patch search: every candidate remains a full-width contiguous window, and
    the existing multi-item separation proof and absolute ceiling still decide acceptance.
    """
    payload = _payload()
    selected = payload.item(4)  # 1109px tall in this fixture, matching Alex's selected crop.
    frame = _paint_inline_reframe(selected.image, start=109, rows=800, preview_height=731)
    surface = _inline_surface_for(frame)

    # Production additionally supplies this calibrated foreign-card ceiling; the real geometry
    # must pass it too, not only the payload-relative proof.
    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.nearest_index == 4

    foreign = dataclasses.replace(payload, crops=(dataclasses.replace(payload.item(3), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, foreign, 1, composer_surface=surface).matched
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    assert not item_verify.verify_sheet_item(
        frame, prompt_only, 1, composer_surface=surface).matched

    # A 707px preview derives a 774-row source window: 335/1109 (30.2%) hidden, beyond the
    # calibrated renderer envelope.
    too_cropped = _paint_inline_reframe(selected.image, start=167, rows=774, preview_height=707)
    refusal = item_verify.verify_sheet_item(
        too_cropped, payload, 4, composer_surface=_inline_surface_for(too_cropped))
    mine = next(c for c in refusal.comparisons if c.number == 4)
    assert mine.distance is None
    assert "above the 30% reframe limit" in mine.reason


def test_tega_style_bright_photo_uses_compact_locator_only_when_composer_binds_it():
    """A fragmented wide-run must remain a legacy refusal, but verify under real controls.

    This pins Tega's exact failure class without putting a real profile image in the repository:
    bright rows split the 870px legacy runs while retaining 745px of photo on every row.  A wrong
    photo and prompt-style crop remain outside the composer-only fallback cap.
    """
    payload = _payload()
    selected = payload.item(4)
    frame = _fragment_legacy_wide_runs(_paint_inline_reframe(selected.image))
    with pytest.raises(item_verify.SheetVerificationError, match="no comment-sheet item preview"):
        item_verify.locate_sheet_preview(frame)
    surface = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1112, 985, 1290), Rect(390, 1305, 985, 1414),
        (695, 1355))
    only = dataclasses.replace(payload, crops=(dataclasses.replace(selected, number=1),))
    verdict = item_verify.verify_sheet_item(frame, only, 1, composer_surface=surface)
    assert verdict.matched, verdict.reason
    assert "compact fallback" in verdict.preview.reason
    assert verdict.preview == item_verify.SheetPreview(
        _SHEET_PREVIEW_Y0, _SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, 95, 985,
        verdict.preview.reason)

    wrong_photo = dataclasses.replace(
        payload, crops=(dataclasses.replace(payload.item(3), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, wrong_photo, 1, composer_surface=surface).matched
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    assert not item_verify.verify_sheet_item(
        frame, prompt_only, 1, composer_surface=surface).matched


def test_scrolled_training_review_uses_the_preview_bound_to_the_live_composer():
    """An upper profile card must neither mask nor impersonate the selected preview."""
    payload = _payload()
    frame, surface = _review_scrolled_inline_frame(selected=2, upper=1)

    # The unchanged legacy locator sees only the upper profile card and refuses it.  Supplying
    # the independently detected composer selects the lower, adjacent preview instead.
    with pytest.raises(item_verify.SheetVerificationError, match="profile screen"):
        item_verify.locate_sheet_preview(frame)
    verdict = item_verify.verify_sheet_item(
        frame, payload, 2, composer_surface=surface)
    assert verdict.matched, verdict.reason
    assert verdict.nearest_index == 2
    assert verdict.preview.y0 >= 800
    assert "composer-bound" in verdict.preview.reason

    # Adversarial control: the upper card is now the requested item, while the preview actually
    # attached to the composer is item 3.  A global scan would confirm the wrong upper content;
    # the bound scan must inspect the lower preview and refuse item 2.
    wrong_frame, wrong_surface = _review_scrolled_inline_frame(selected=3, upper=2)
    wrong = item_verify.verify_sheet_item(
        wrong_frame, payload, 2, composer_surface=wrong_surface)
    assert not wrong.matched
    assert wrong.nearest_index == 3
    assert wrong.preview.y0 >= 800


def test_hinge_10_0_1_bright_photo_edge_does_not_truncate_the_selected_preview_above_the_composer():
    """The 2026-08-22 live refusal was a measurement fault, not a composer fault.

    Blocker 13: an ordinary, correct Hinge 10.0.1 composer -- selected photo, comment field
    directly under it, rose pill and Send Priority Like under that -- was refused post-tap with
    "the selected-card preview is not immediately above ... inline comment field and Send Like
    CTA".  Measured offline on the refusing frame (sha256 29175f91..., 1080x2400):

        composer      comment_rect (95,1124)-(985,1302), send_rect (390,1334)-(985,1443)
        photo         rows 236..1092, columns 95..985 -- 856 rows, 87.9% of the 974px stored
                      crop, so not remotely the 30% reframe limit either
        gap           1124 - 1092 = 32px
        located       rows 236..856 (620 rows), because 29 of the block's rows measure
                      849..868px of non-background span against the strict 870px floor
        the check     gap 1124 - 856 = 268px against max(40, round(620 * 0.25)) = 155px

    Both of the check's numbers were right about the fragment and neither was about the photo.
    The bottom edge is the thing that was wrong, and it was wrong for the CONTENT comparison too:
    `_compare_item` derives its source window from `preview.height`, so a 620px preview asks the
    stored crop for 620/856 of the rows the sheet is actually showing.
    """
    payload = _payload()
    selected = payload.item(4)
    frame = _spend_the_wide_run_headroom(
        _paint_inline_reframe(selected.image),
        first_row=_SHEET_PREVIEW_Y0 + _V1001_FRAGMENT_ROWS)
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    assert _V1001_COMMENT.y0 - (_SHEET_PREVIEW_Y0 + _V1001_PREVIEW_ROWS) == _V1001_PHOTO_TO_FIELD_GAP
    # Both the real 849px row and this fixture's 869px one clear the composer-bound floor while
    # failing the absolute one, which is the whole regime this defect lives in.
    assert (round(_V1001_COMMENT.width * item_verify._INLINE_COMPACT_MIN_WIDTH_FRACTION)
            <= _V1001_NARROWEST_SPAN < item_verify._PREVIEW_MIN_WIDTH_PX)

    # The public locator is unchanged and still reports the fragment: this fix does not widen it.
    assert item_verify.locate_sheet_preview(frame).height == _V1001_FRAGMENT_ROWS

    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.nearest_index == 4
    assert verdict.preview.y0 == _SHEET_PREVIEW_Y0
    assert verdict.preview.height == _V1001_PREVIEW_ROWS
    assert "bottom edge carried" in verdict.preview.reason

    # The corrected bottom edge is a better measurement, not a wider door.
    foreign = dataclasses.replace(payload, crops=(dataclasses.replace(payload.item(3), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, foreign, 1, composer_surface=surface).matched
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    assert not item_verify.verify_sheet_item(
        frame, prompt_only, 1, composer_surface=surface).matched


def test_hinge_10_1_suggestion_shelf_proves_the_short_selected_photo_regime():
    """The 2026-09-01 halt was a new photo -> prompt shelf -> field topology.

    Its exact measured geometry was preview=(95,421)-(985,971), comment y=1124, hence a 153px
    gap. The selected square card showed 602/974 source rows (38.2%); this synthetic equivalent
    uses 602/1000 (39.8%) so both the shelf-bound gap and the separate 40% reframe cap are needed.
    """
    payload = _payload()
    selected = payload.item(3)
    frame = _paint_inline_reframe(
        selected.image, start=398, rows=602, preview_height=550, y0=421)
    surface = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1124, 985, 1302),
        Rect(390, 1334, 985, 1443), (695, 1390))
    only = dataclasses.replace(
        payload, crops=(dataclasses.replace(selected, number=1),))

    # A blind 153px allowance would let blank separation opt into the broader window search.
    # The old direct topology remains a refusal until the outlined shelf is visible.
    with pytest.raises(item_verify.SheetVerificationError, match="no bounded outlined"):
        item_verify.verify_sheet_item(
            frame, only, 1, composer_surface=surface, absolute_max_dist=10.0)

    shelf_frame = _with_inline_suggestion_shelf(frame)
    verdict = item_verify.verify_sheet_item(
        shelf_frame, only, 1, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.preview == item_verify.SheetPreview(
        421, 971, 95, 985, verdict.preview.reason)
    assert "suggestion shelf" in verdict.preview.reason
    assert "reframe limit 40%" in verdict.preview.reason
    assert verdict.comparisons[0].window_px == 602

    # The new structure changes neither item identity nor the absolute content ceiling.
    foreign = dataclasses.replace(
        payload, crops=(dataclasses.replace(payload.item(4), number=1),))
    assert not item_verify.verify_sheet_item(
        shelf_frame, foreign, 1, composer_surface=surface,
        absolute_max_dist=10.0).matched


@pytest.mark.parametrize("filled,pills", [(True, 2), (False, 1)])
def test_short_preview_cannot_use_a_filled_bar_or_one_outline_as_a_suggestion_shelf(
        filled, pills):
    selected = _payload().item(3)
    frame = _paint_inline_reframe(
        selected.image, start=398, rows=602, preview_height=550, y0=421)
    frame = _with_inline_suggestion_shelf(frame, filled=filled, pills=pills)
    surface = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1124, 985, 1302),
        Rect(390, 1334, 985, 1443), (695, 1390))
    only = dataclasses.replace(
        _payload(), crops=(dataclasses.replace(selected, number=1),))

    with pytest.raises(item_verify.SheetVerificationError, match="no bounded outlined"):
        item_verify.verify_sheet_item(frame, only, 1, composer_surface=surface)


def test_left_attached_card_with_a_right_silent_rail_keeps_existing_signature_gates():
    """The 2026-08-31 one-sided Hinge 10.1.0 halt, reproduced without profile data.

    The live selected card's median visible ink was x=96..927, while its proven comment field
    was x=95..985.  The full card rect is still the field's 890px column; its final 58px are
    simply page-coloured after Hinge removes the profile heart.  The existing inline comparison
    already excludes x=825..985 (18%), so x=927 reaches 102px beyond every compared column.
    """
    payload = _payload()
    selected = payload.item(4)
    frame = _right_silent_inline_sheet(selected.image, ink_x0=96, ink_x1=927)
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    cv2_mod, np_mod = item_verify._require_vision()

    # The public full-bleed detector deliberately remains strict. Only the independently proven
    # composer enables this recovery, and its returned rect is the full card while its ink record
    # keeps the asymmetry auditable.
    with pytest.raises(item_verify.SheetVerificationError):
        item_verify.locate_sheet_preview(frame)
    preview = item_verify._locate_inline_composer_preview(
        frame, surface, cv2=cv2_mod, np=np_mod)
    lane = round(_V1001_COMMENT.width * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    assert (preview.x0, preview.x1) == (_V1001_COMMENT.x0, _V1001_COMMENT.x1)
    assert preview.ink_bounds == (96, 927)
    assert preview.ink_x1 - (_V1001_COMMENT.x1 - lane) == 102
    assert "left-edge-attached" in preview.reason

    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.nearest_index == 4

    # Geometry recovery does not supply identity: the existing complete-payload proof and frozen
    # absolute ceiling still refuse foreign and prompt-shaped content.
    wrong_photo = dataclasses.replace(
        payload, crops=(dataclasses.replace(payload.item(3), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, wrong_photo, 1, composer_surface=surface, absolute_max_dist=10.0).matched
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    assert not item_verify.verify_sheet_item(
        frame, prompt_only, 1, composer_surface=surface, absolute_max_dist=10.0).matched


def test_left_attached_right_silent_regime_requires_the_existing_compared_columns_and_topology():
    """No mirror, truncated card, distant field, or unsupported internal gap may use the regime."""
    selected = _payload().item(4)
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    cv2_mod, np_mod = item_verify._require_vision()

    # The last signature column is x=824 when the control-free lane starts at x=825: one pixel
    # short is still refusal, even though the old 694px compact width floor would be satisfied.
    before_lane = _right_silent_inline_sheet(selected.image, ink_x0=96, ink_x1=824)
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify._locate_inline_composer_preview(
            before_lane, surface, cv2=cv2_mod, np=np_mod)

    # A 28px left displacement is just outside the existing 27px edge slack.  There is
    # intentionally no right-attached/mirrored counterpart because the signature excludes only
    # the right lane.
    shifted_left = _right_silent_inline_sheet(selected.image, ink_x0=123, ink_x1=927)
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify._locate_inline_composer_preview(
            shifted_left, surface, cv2=cv2_mod, np=np_mod)

    distant = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1400, 985, 1578), Rect(390, 1610, 985, 1719), (695, 1660))
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify._locate_inline_composer_preview(
            _right_silent_inline_sheet(selected.image, ink_x0=96, ink_x1=927), distant,
            cv2=cv2_mod, np=np_mod)

    # Four page-coloured rows exceed the separately calibrated three-row bridge. The lower
    # fragment is intentionally under 300px and the upper one is too distant from the field, so
    # neither can manufacture a selected-card candidate.
    broken = _right_silent_inline_sheet(
        selected.image, ink_x0=96, ink_x1=927, gap_row=800)
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify._locate_inline_composer_preview(
            broken, surface, cv2=cv2_mod, np=np_mod)


# The 2026-08-28 pillarbox halt, as numbers. Live Pixel 7a, Hinge 10.1.0, profile `Yvonne`
# page heart 7, run `data/hinge_debug/f6a15d162e68` (gitignored; only geometry here).
_PILLARBOX_CARD_PAD = 209          # card background either side of a 556px photo in a 974px card
_PILLARBOX_INK_W = 508             # ...which the sheet renders as 508px at the 0.9137 scale
_PILLARBOX_INK_X0 = 286            # 95 + round(209 * 890/974)
_PILLARBOX_INK_X1 = 794            # 985 - 191, symmetric to the pixel
_PILLARBOX_SHEET_PAD = 191


def _pillarboxed_sheet(number: int = 4, *, pad: int = _PILLARBOX_CARD_PAD):
    """A PROMPT-PHOTO card whose photograph is portrait, rendered the way the sheet renders it.

    Every other fixture here is a full-bleed card, where the photograph runs edge to edge, so the
    preview's ink and the sheet's content column coincide and "the preview spans the column" looks
    like a layout invariant. It is a property of the CARD's content, and this is the card that
    says so: Hinge fits a portrait photo to the card's height and centres it, leaving `pad` px of
    card background on each side. Painting that background to the page's own grey is not a cheat,
    it is the measured condition -- card white and sheet white are the same colour, so the row
    probe can only ever see the photograph, never the card it sits on.

    Returns the payload whose item `number` really is this card, the rendered sheet, and the
    composer surface. The geometry that comes out is the live one to the pixel: a 1109px card at
    the corpus 890/974 scale, bottom-anchored into the 856px content region, ink at columns
    286..794 with 191px of card either side, and a 32px gap to the comment field.
    """
    payload = _payload()
    crop = payload.item(number)
    card = cv2.imdecode(np.frombuffer(crop.image, np.uint8), cv2.IMREAD_COLOR)
    card[:, :pad] = _SHEET_BG
    card[:, card.shape[1] - pad:] = _SHEET_BG
    recut = _recut(crop, card)
    payload = dataclasses.replace(payload, crops=tuple(
        recut if c.number == number and c.kind == item_crops.CROP_ITEM else c
        for c in payload.crops))
    frame = paint_sheet(recut.image)
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    return payload, frame, surface


def test_a_portrait_photo_centred_in_its_card_is_still_the_selected_preview():
    """Blocker: an ordinary, correct composer refused post-tap because the photo was portrait.

    The 2026-08-28 live halt, measured on the refusing frame:

        composer   comment_rect (95,1124)-(985,1302) -- content column 890px wide
        card       974px wide, 1109 tall, its photograph 556x932 centred with 209px of card
                   background either side
        rendered   the whole card into the column at 890/974 = 0.9137, so the photo's INK is
                   508px at columns 286..794 -- 191px of card left, 191px right
        the check  508 >= round(890*0.78) = 694 ? no. |286-95| <= 27 ? no. |794-985| <= 27 ? no.
                   -> "no candidate at least 694px wide and 300px tall is aligned with it"

    All three tests were right about the ink and none of them was about the card. The rect the
    comparison needs is the CARD's, because the stored crop is the whole card; the ink box is a
    fact about the photograph inside it.
    """
    payload, frame, surface = _pillarboxed_sheet()
    cv2_mod, np_mod = item_verify._require_vision()

    # The flush regime genuinely cannot see this preview -- without which this test would pass
    # through the old path and prove nothing about the new one.
    assert _PILLARBOX_INK_W < round(
        _V1001_COMMENT.width * item_verify._INLINE_COMPACT_MIN_WIDTH_FRACTION)
    with pytest.raises(item_verify.SheetVerificationError):
        item_verify.locate_sheet_preview(frame)

    preview = item_verify._locate_inline_composer_preview(
        frame, surface, cv2=cv2_mod, np=np_mod)
    assert preview.pillarboxed
    assert preview.ink_bounds == (_PILLARBOX_INK_X0, _PILLARBOX_INK_X1)
    assert preview.ink_width == _PILLARBOX_INK_W
    # THE RECT IS THE CARD'S COLUMNS, NOT THE INK'S. This is the assertion the fix exists for:
    # `_compare_item` scales the stored 974px crop against `preview.width`, so an ink-box rect
    # would ask a 1109px card for round(837 * 974/508) = 1604 rows and refuse one stage later.
    assert (preview.x0, preview.x1) == (_V1001_COMMENT.x0, _V1001_COMMENT.x1)
    assert preview.y0 == _SHEET_PREVIEW_Y0
    assert preview.height == _SHEET_PREVIEW_MAX_H
    assert _V1001_COMMENT.y0 - preview.y1 == _V1001_PHOTO_TO_FIELD_GAP
    assert (preview.ink_x0 - preview.x0) == (preview.x1 - preview.ink_x1) == _PILLARBOX_SHEET_PAD

    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.nearest_index == 4

    # ...and it is a better measurement, not a wider door: the other stored items, a card from a
    # different profile, and the same card rendered top-anchored all still fail.
    for other in (1, 2, 3):
        assert not item_verify.verify_sheet_item(
            frame, payload, other, composer_surface=surface,
            absolute_max_dist=10.0).matched, f"item {other} verified against item 4's sheet"
    foreign = dataclasses.replace(
        _lookalike_payload(), crops=(dataclasses.replace(
            _lookalike_payload().item(2), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, foreign, 1, composer_surface=surface).matched


def test_a_pillarboxed_comparison_applies_its_ceilings_in_the_units_they_were_fitted_in():
    """A pillarboxed rect is blank on both sides, so every distance through it contracts.

    Roughly 30% of the compared rect is card-white in the sheet AND card-white in every candidate
    crop, so those cells contribute nothing and the whole scale shrinks. The RELATIVE bound does
    not care -- `0.5 x nearest_other` is a ratio of two quantities that contract together, which
    is exactly why a multi-item run looks safe and hides this. The two ABSOLUTE ceilings do care:
    they are frozen numbers fitted on full-bleed renders, and `inline_item_max_dist` is clamped to
    0.0001 below the known 14.91 foreign-card collision on purpose. Left uncorrected, making this
    regime reachable would have widened both of them by the contraction factor -- the one guard
    that catches content the payload does not contain at all, quietly loosened.

    Scaling by the fraction of the compared rect that carries ink puts them back in their own
    units. It can only ever LOWER a ceiling, so it cannot create a false accept.
    """
    payload, frame, surface = _pillarboxed_sheet()
    one_item = dataclasses.replace(
        payload, crops=(dataclasses.replace(payload.item(4), number=1),))
    verdict = item_verify.verify_sheet_item(
        frame, one_item, 1, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason

    # the ink fraction of the compared rect: the content column less the excluded heart lane
    lane = round(_V1001_COMMENT.width * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    compared = _V1001_COMMENT.width - lane
    expected = item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST * (_PILLARBOX_INK_W / compared)
    assert verdict.bound == pytest.approx(expected, rel=1e-9)
    assert verdict.bound < item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST

    # ...and a FULL-BLEED sheet keeps the constant exactly, so nothing that predates the
    # pillarbox regime moves.
    full = _payload()
    flush_frame = _paint_inline_reframe(full.item(4).image)
    flush_surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    flush = item_verify.verify_sheet_item(
        flush_frame, dataclasses.replace(
            full, crops=(dataclasses.replace(full.item(4), number=1),)),
        1, composer_surface=flush_surface, absolute_max_dist=10.0)
    assert flush.matched, flush.reason
    assert flush.bound == item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST


def test_a_pillarboxed_card_scrolled_above_the_composer_does_not_displace_the_preview():
    """The one shape that is contained AND centred without being the selected preview.

    An ordinary full-bleed profile card is refused by containment -- it spans x=53..1027 against a
    field at 95..985 and overhangs both ends. A PILLARBOXED card does not: measured against the
    content column, the very card this regime was written for sits at 262..818, which is 167px
    inside on the left and 167px inside on the right, i.e. contained and centred to the pixel. So
    for this one shape containment and centring decide nothing, and what is left holding the line
    is adjacency to the comment field -- exactly the situation a Training reviewer creates by
    scrolling while deciding.

    [swept 2026-08-28: no frame in the corpus has both a pillarboxed preview and a second
    pillarboxed block above it, so nothing measured exercises this. It is realizable and it is
    the fix's thinnest margin, so it is constructed here rather than left to a live run.]
    """
    payload = _payload()
    crop = payload.item(4)
    card = cv2.imdecode(np.frombuffer(crop.image, np.uint8), cv2.IMREAD_COLOR)
    card[:, :_PILLARBOX_CARD_PAD] = _SHEET_BG
    card[:, card.shape[1] - _PILLARBOX_CARD_PAD:] = _SHEET_BG
    recut = _recut(crop, card)
    payload = dataclasses.replace(payload, crops=tuple(
        recut if c.number == 4 and c.kind == item_crops.CROP_ITEM else c for c in payload.crops))

    # The sheet's real content height, not a smaller one: at 700 this 1109px card would
    # hide 343px = 30.9% of itself and be refused by the reframe limit rather than by
    # anything this test is about.
    selected_y0, selected_h = 800, _SHEET_PREVIEW_MAX_H
    frame = paint_sheet(recut.image, y0=selected_y0, max_h=selected_h)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    # A second pillarboxed card, scrolled into view above the still-open composer. Its ink is the
    # same photo geometry at the card's own scale: 262..818 of the 53..1027 card.
    upper_y0, upper_y1 = 236, 640
    image[upper_y0:upper_y1, _CARD_X0 + _PILLARBOX_CARD_PAD:_CARD_X1 - _PILLARBOX_CARD_PAD] = 40
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frame = encoded.tobytes()
    comment = Rect(95, selected_y0 + selected_h + 20, 985, selected_y0 + selected_h + 198)
    surface = ComposerSurface(
        "hinge_inline_v1", comment, Rect(390, comment.y1 + 15, 985, comment.y1 + 124),
        (695, comment.y1 + 65))

    # The upper card really is contained and really is centred -- this test is worthless if it
    # is refused by the tests that catch an ordinary card, so pin that it is not.
    upper_left = (_CARD_X0 + _PILLARBOX_CARD_PAD) - comment.x0
    upper_right = comment.x1 - (_CARD_X1 - _PILLARBOX_CARD_PAD)
    assert upper_left == upper_right > 0
    assert upper_y1 - upper_y0 >= item_verify._PREVIEW_MIN_HEIGHT_PX

    cv2_mod, np_mod = item_verify._require_vision()
    preview = item_verify._locate_inline_composer_preview(
        frame, surface, cv2=cv2_mod, np=np_mod)
    assert preview.pillarboxed
    assert preview.y0 == selected_y0, (
        f"the locator took the scrolled card at {preview.y0} instead of the selected preview "
        f"at {selected_y0}")
    assert preview.y1 > upper_y1
    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason

    # ISOLATING THE GAP BOUND FROM THE TIEBREAK. Two independent things protect this frame -- the
    # bound refuses a distant candidate outright, and `min(candidates, key=gap)` prefers the
    # nearest one -- and with both present the tiebreak alone is enough, so the assertions above
    # pass even with the bound removed. [confirmed by mutation 2026-08-28: deleting the bound
    # left this test green.] The bound is therefore pinned on its own: the scrolled card with
    # NOTHING adjacent to the field must be refused rather than promoted to the preview.
    alone = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    alone[selected_y0:selected_y0 + selected_h, :] = _SHEET_BG
    ok, encoded_alone = cv2.imencode(".png", alone)
    assert ok
    with pytest.raises(item_verify.SheetVerificationError) as refusal:
        item_verify._locate_inline_composer_preview(
            encoded_alone.tobytes(), surface, cv2=cv2_mod, np=np_mod)
    assert "field gap" in str(refusal.value)


def test_a_pillarboxed_run_that_spills_past_the_content_column_is_still_refused():
    """Containment does the work the two edge-alignment tests used to do, and does it as hard.

    The pillarbox regime drops "both edges sit ON the comment field's" for "the ink sits INSIDE
    them, centred". An ordinary profile card is the thing that must not survive the swap: it
    spans x=53..1027 against a field at 95..985, so it spills 42px past BOTH ends -- the same
    42px the module docstring's margin test has always turned on, now measured as containment
    rather than as alignment.
    """
    payload, frame, surface = _pillarboxed_sheet()
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    # Replace the pillarboxed preview with a full-width x=53 card at the same rows: the only tall
    # run on the frame now spills past the column on both sides.
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, :] = _SHEET_BG
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, _CARD_X0:_CARD_X1] = 40
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    cv2_mod, np_mod = item_verify._require_vision()
    assert _CARD_X0 < _V1001_COMMENT.x0 and _CARD_X1 > _V1001_COMMENT.x1
    with pytest.raises(item_verify.SheetVerificationError) as refusal:
        item_verify._locate_inline_composer_preview(
            encoded.tobytes(), surface, cv2=cv2_mod, np=np_mod)
    message = str(refusal.value)
    # over the field's columns on BOTH sides, by the same 42px the margin test has always used
    assert f"left edge {_CARD_X0 - _V1001_COMMENT.x0:+d}px" in message
    assert f"right edge {_CARD_X1 - _V1001_COMMENT.x1:+d}px" in message
    assert "pillarboxed inside it" not in message, (
        "a card that overhangs the column is the opposite of a pillarboxed one")


def test_an_off_centre_contained_run_is_refused_by_the_centring_test():
    """Contained is not sufficient on its own; the card centres its photo and so must we.

    A tall block that sits inside the content column but hard against one side of it is not a
    pillarboxed card render -- the card fills the column, so its photo's two margins are equal by
    construction. Accepting an off-centre block would be accepting a rect whose ink says the
    scale is one thing and whose columns say it is another.
    """
    payload, frame, surface = _pillarboxed_sheet()
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, :] = _SHEET_BG
    shifted_x1 = _V1001_COMMENT.x0 + _PILLARBOX_INK_W
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H,
          _V1001_COMMENT.x0:shifted_x1] = 40
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    cv2_mod, np_mod = item_verify._require_vision()
    with pytest.raises(item_verify.SheetVerificationError) as refusal:
        item_verify._locate_inline_composer_preview(
            encoded.tobytes(), surface, cv2=cv2_mod, np=np_mod)
    message = str(refusal.value)
    # contained, tall enough, adjacent enough -- and refused purely on the asymmetry
    assert f"left edge {0:+d}px" in message
    assert f"right edge {shifted_x1 - _V1001_COMMENT.x1:+d}px" in message
    assert "pillarboxed inside it" not in message, (
        "an off-centre block must not be described as a pillarboxed card render")


def test_the_refusal_names_the_runs_it_measured_and_the_test_each_one_failed():
    """A refusal that says only "nothing was aligned" cannot be told from "nothing was there".

    The 2026-08-28 report carried exactly one sentence -- "no candidate at least 694px wide and
    300px tall is aligned with it" -- and diagnosing it needed the frame re-measured by hand. The
    near-miss IS the diagnosis, so the runs that were rejected, their geometry, and which test
    each failed are part of the refusal rather than of a later investigation.
    """
    payload, frame, surface = _pillarboxed_sheet()
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, :] = _SHEET_BG
    image[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H, _CARD_X0:_CARD_X1] = 40
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    cv2_mod, np_mod = item_verify._require_vision()
    with pytest.raises(item_verify.SheetVerificationError) as refusal:
        item_verify._locate_inline_composer_preview(
            encoded.tobytes(), surface, cv2=cv2_mod, np=np_mod)
    message = str(refusal.value)
    # the field it measured against, and the block it actually saw, with its real geometry
    assert f"columns {_V1001_COMMENT.x0}..{_V1001_COMMENT.x1}" in message
    assert f"rows {_SHEET_PREVIEW_Y0}..{_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H}" in message
    assert f"columns {_CARD_X0}..{_CARD_X1}" in message
    # signed against the field's own edges, so "over on both sides" reads at a glance
    assert f"left edge {_CARD_X0 - _V1001_COMMENT.x0:+d}px" in message
    assert f"right edge {_CARD_X1 - _V1001_COMMENT.x1:+d}px" in message
    # and what it PASSED, without which a pillarbox reads as an ordinary mislocated block
    assert "passes:" in message and f"height {_SHEET_PREVIEW_MAX_H} rows" in message


def _pale_edge_card_above(*, pale: str, gap_rows: int = 3):
    """A foreign profile card above the preview whose LEADING COLUMNS read as page background.

    The existing `_review_scrolled_inline_frame` paints its upper card opaque across the full
    x=53..1027 card width, whose span fails both alignment tests by 42px against 27px of slack --
    so it never reaches the upward walk at all and proves nothing about it. This is the case that
    does: with one edge pale enough to be background, the reported span STARTS (or ends) exactly
    on the preview's own column, passes the alignment test, and runs 42px past the opposite edge.
    """
    payload = _payload()
    frame = _paint_inline_reframe(payload.item(4).image)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    top, bottom = 110, _SHEET_PREVIEW_Y0 - gap_rows
    image[top:bottom, _CARD_X0:_CARD_X1] = 40                     # an unrelated dark card
    if pale == "left":
        image[top:bottom, _CARD_X0:_SHEET_PREVIEW_X0] = _SHEET_BG
    else:
        image[top:bottom, _SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W:_CARD_X1] = _SHEET_BG
    image[bottom:_SHEET_PREVIEW_Y0, :] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return payload, encoded.tobytes()


@pytest.mark.parametrize("pale", ["left", "right"])
def test_a_foreign_card_with_one_pale_edge_cannot_weld_onto_the_preview(pale):
    """The upward walk must not annex a DIFFERENT block, in either polarity.

    Alignment alone does not say a row belongs to this photo -- it says one of its edges does. A
    card whose leading columns blend into the page reports a span that starts on the preview's own
    x0 and still runs 42px past its x1, and before the containment check every test above passed
    it. [measured 2026-08-27 on the real 00097 composer frame: a pale-left card at rows 110..232
    welded a 236..1092 preview into 110..1092 and turned a `verify_match` into a
    `verify_mismatch`.] A welded preview is a taller window than the sheet is really showing,
    which mis-scales every distance measured through it.
    """
    payload, frame = _pale_edge_card_above(pale=pale)
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    seed = item_verify.locate_sheet_preview(frame)
    cv2_mod, np_mod = item_verify._require_vision()
    extended = item_verify._extend_inline_preview_to_block_edges(
        frame, seed, surface, cv2=cv2_mod, np=np_mod)

    assert extended.y0 >= seed.y0, (
        f"the walk annexed the foreign card above: {seed.y0} -> {extended.y0}")
    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason


def test_a_seam_in_each_direction_does_not_discard_either_direction_s_carry():
    """The seam budget is judged PER EDGE; summing the two directions was a regression.

    Only one direction existed before the walk became bidirectional, so testing the SUM against a
    single-direction budget meant a legal seam below plus a bridged row above pushed
    `bridged_runs` to 2, threw the whole extension away, and produced the same false-refusal class
    the walk exists to prevent. Each direction is now held to exactly the budget the one direction
    was held to -- no direction gets a wider seam than before.

    The frame carries ONE bridged run in each direction (1 row up, the measured 3 down): four
    bridged rows over two runs in total, which the old summed test rejected outright. What still
    stops the walk reaching a different block is unchanged and is checked next door -- the cap on
    CONSECUTIVE unsupported rows, and the column containment in `row_support`.
    """
    payload, frame = _leading_pale_inline_reframe()
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    x0, x1 = _SHEET_PREVIEW_X0, _SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W
    seed = item_verify.locate_sheet_preview(frame)
    image[seed.y0 - 40:seed.y0 - 39, x0:x1] = _SHEET_BG     # one bridged row, ABOVE the seed
    image[seed.y1 - 5:seed.y1 - 2, x0:x1] = _SHEET_BG       # the measured 3-row seam, BELOW it
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frame = encoded.tobytes()
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))

    cv2_mod, np_mod = item_verify._require_vision()
    reseed = item_verify.locate_sheet_preview(frame)
    extended = item_verify._extend_inline_preview_to_block_edges(
        frame, reseed, surface, cv2=cv2_mod, np=np_mod)

    assert extended.y0 < reseed.y0, "the upward carry was discarded by the other edge's seam"
    assert extended.y1 > reseed.y1, "the downward carry was discarded by the other edge's seam"
    assert (extended.y0, extended.y1) == (_SHEET_PREVIEW_Y0, _SHEET_PREVIEW_Y0 + _V1001_PREVIEW_ROWS)
    assert "bridging 4 measured internal background rows" in extended.reason


def test_inline_preview_bridges_only_the_measured_three_row_internal_interruption():
    """Hinge 10.1.0 can paint three page-coloured rows inside one selected photo."""
    payload = _payload()
    selected = payload.item(4)
    frame = _paint_inline_reframe(selected.image)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    gap_y = _SHEET_PREVIEW_Y0 + 620
    image[gap_y:gap_y + 3, _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frame = encoded.tobytes()
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))

    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.preview.height == _V1001_PREVIEW_ROWS
    assert "bridging 3 measured internal background rows" in verdict.preview.reason

    # One row beyond the measured seam remains a real topology boundary and cannot be crossed.
    image[gap_y:gap_y + 4, _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify.verify_sheet_item(
            encoded.tobytes(), payload, 4, composer_surface=surface, absolute_max_dist=10.0)


def test_inline_preview_does_not_combine_separate_short_internal_interruptions():
    """Only the one observed three-row seam is calibrated for composer-only extension."""
    payload = _payload()
    selected = payload.item(4)
    frame = _paint_inline_reframe(selected.image)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    first_gap_y = _SHEET_PREVIEW_Y0 + 620
    second_gap_y = first_gap_y + 80
    for gap_y in (first_gap_y, second_gap_y):
        image[gap_y:gap_y + 1,
              _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))

    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify.verify_sheet_item(
            encoded.tobytes(), payload, 4, composer_surface=surface, absolute_max_dist=10.0)


def test_inline_preview_keeps_a_measured_right_edge_through_left_sky_blending():
    """Hinge 10.1.0 can make a long pale photo edge indistinguishable from its page.

    Lauren's refused frame kept x=985 on every affected row, but pale sky erased the left 229px
    at the narrowest point.  The strict locator consequently reported rows 236..628 instead of
    the real 236..1092 selected photo and manufactured a 496px gap to a correctly detected
    composer.  This fixture records only those measured pixels, never the private photo.
    """
    payload = _payload()
    selected = payload.item(4)
    frame = _paint_inline_reframe(selected.image)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    sky_y0 = _SHEET_PREVIEW_Y0 + 392
    sky_rows = 73
    erased_left = 229
    image[sky_y0:sky_y0 + sky_rows,
          _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + erased_left] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frame = encoded.tobytes()
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))

    # The public detector remains strict; only a separately proven composer can join the runs.
    assert item_verify.locate_sheet_preview(frame).height == 392
    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.preview.height == _V1001_PREVIEW_ROWS
    assert "retaining one aligned edge across 73 rows" in verdict.preview.reason

    # Recovering the measured boundary changes only which pixels are compared.  The ordinary
    # relative/absolute content gates must still reject a different photo and a prompt card.
    foreign = dataclasses.replace(payload, crops=(dataclasses.replace(payload.item(3), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, foreign, 1, composer_surface=surface, absolute_max_dist=10.0).matched
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    assert not item_verify.verify_sheet_item(
        frame, prompt_only, 1, composer_surface=surface, absolute_max_dist=10.0).matched

    # One pixel below the calibrated 74% content floor is not continuation evidence.  A later
    # wide run cannot pull the preview across it merely because the right edge still lines up.
    image[sky_y0:sky_y0 + sky_rows,
          _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + erased_left + 3] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify.verify_sheet_item(
            encoded.tobytes(), payload, 4, composer_surface=surface, absolute_max_dist=10.0)


def test_inline_preview_recovers_pale_rows_above_the_only_compact_seed():
    """Regression for Lauren's post-keyboard false 53.7% reframe refusal.

    The full photo remains on screen, but its pale leading region is below the compact 78% row
    floor.  The composer-bound locator therefore seeds on the lower 379 rows and extent recovery
    must carry the top edge upward under the narrower, one-edge-attached 74% rule.  The calibrated
    30% source-reframe limit remains untouched and still judges the recovered full preview.
    """
    payload, frame = _leading_pale_inline_reframe()
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))

    seed = item_verify.locate_sheet_preview(frame)
    assert seed.y0 == _SHEET_PREVIEW_Y0 + 477
    assert seed.y1 == _SHEET_PREVIEW_Y0 + _V1001_PREVIEW_ROWS
    verdict = item_verify.verify_sheet_item(
        frame, payload, 4, composer_surface=surface, absolute_max_dist=10.0)
    assert verdict.matched, verdict.reason
    assert verdict.nearest_index == 4
    assert verdict.preview.y0 == _SHEET_PREVIEW_Y0
    assert verdict.preview.height == _V1001_PREVIEW_ROWS
    assert "top edge carried" in verdict.preview.reason
    assert "retaining one aligned edge across 477 rows" in verdict.preview.reason

    # Edge recovery changes geometry only. The unique-nearest and absolute content gates still
    # refuse both a different photograph and the independent prompt-card family.
    foreign = dataclasses.replace(
        payload, crops=(dataclasses.replace(_payload().item(3), number=1),))
    assert not item_verify.verify_sheet_item(
        frame, foreign, 1, composer_surface=surface, absolute_max_dist=10.0).matched
    prompt = dataclasses.replace(_lookalike_payload().item(2), number=1)
    prompt_only = dataclasses.replace(_lookalike_payload(), crops=(prompt,))
    assert not item_verify.verify_sheet_item(
        frame, prompt_only, 1, composer_surface=surface, absolute_max_dist=10.0).matched


def test_inline_preview_top_edge_recovery_keeps_its_measured_boundaries():
    """A weaker edge or four page-coloured rows cannot join an upper block to the seed."""
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))

    # 232 erased pixels leave 658/890 visible, one pixel below the rounded 74% floor. The lower
    # seed remains real and adjacent to the composer, but it truthfully cannot confirm the item.
    too_narrow_payload, too_narrow = _leading_pale_inline_reframe(erased_display_px=232)
    refusal = item_verify.verify_sheet_item(
        too_narrow, too_narrow_payload, 4, composer_surface=surface)
    intended = next(c for c in refusal.comparisons if c.number == 4)
    assert not refusal.matched and intended.distance is None
    assert "above the 30% reframe limit" in intended.reason

    # Even otherwise-supported upper rows cannot be reached through four blank rows. This is a
    # real topology boundary, not the measured at-most-three-row interruption inside one photo.
    payload, frame = _leading_pale_inline_reframe()
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    seed_y0 = _SHEET_PREVIEW_Y0 + 477
    image[seed_y0 - 4:seed_y0, _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = _SHEET_BG
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    refusal = item_verify.verify_sheet_item(
        encoded.tobytes(), payload, 4, composer_surface=surface)
    intended = next(c for c in refusal.comparisons if c.number == 4)
    assert not refusal.matched and intended.distance is None
    assert "above the 30% reframe limit" in intended.reason


def test_inline_preview_bottom_edge_cannot_walk_to_a_composer_the_photo_does_not_reach():
    """The safety property the topology check exists for, re-proved against the corrected edge.

    Carrying the preview's bottom edge down would be worthless if it could reach any composer
    that happened to be on the frame: the check is what stops a heart being verified against a
    sheet that is not the inline composer for the selected card.  Three structures that are
    genuinely wrong must still be refused -- a composer nowhere near the photo, a photo that
    stops with page background between it and the field (a partially rendered view), and a
    surface with no CTA at all.
    """
    payload = _payload()
    selected = payload.item(4)
    frame = _spend_the_wide_run_headroom(
        _paint_inline_reframe(selected.image),
        first_row=_SHEET_PREVIEW_Y0 + _V1001_FRAGMENT_ROWS)

    # 1. The photo ends at row 1092; this composer starts 708px lower, and no walk over image
    #    rows can cross the page background between them.
    distant = ComposerSurface(
        "hinge_inline_v1", Rect(95, 1800, 985, 1978), Rect(390, 2000, 985, 2109), (695, 2050))
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify.verify_sheet_item(frame, payload, 4, composer_surface=distant)

    # 2. The composer is where 10.0.1 really puts it, but only 500 rows of photo were rendered,
    #    so rows 736..1124 are page background. The edge stops at 736 and the gap is 388px.
    partial = _paint_inline_reframe(selected.image, preview_height=500)
    assert item_verify.locate_sheet_preview(partial).y1 == _SHEET_PREVIEW_Y0 + 500
    surface = ComposerSurface("hinge_inline_v1", _V1001_COMMENT, _V1001_SEND, (695, 1390))
    with pytest.raises(item_verify.SheetVerificationError, match="not immediately above"):
        item_verify.verify_sheet_item(partial, payload, 4, composer_surface=surface)

    # 3. No CTA: an unsendable half-detected surface is not a composer, before geometry is read.
    with pytest.raises(item_verify.SheetVerificationError, match="no comment/send rectangles"):
        item_verify.verify_sheet_item(
            frame, payload, 4, composer_surface=dataclasses.replace(surface, send_rect=None))


# =====================================================================================
# A SECOND PAGE: the shape that produced a measured false accept
#
# On the calibration corpus (2026-08-12) one profile's look-alike prompt cards differed in
# HEIGHT: one of them was just tall enough to be the window the sheet renders and its two nearest
# neighbours were just too short. The short ones were dropped out of the neighbour set the accept
# bound is derived from, that bound grew ~4x, and a card belonging to a DIFFERENT PROFILE landed
# inside it — 10 accepts in 540 comparisons, which the driver turned into a sent like.
#
# This page is that shape, painted from first principles. `_LOOKALIKE_WINDOW_FLOOR` is the height
# the split is arranged around: a full-height sheet renders 937px of card and sweeps +-19, so a
# card under 918px cannot be what is on screen and used to fall out of the reference set with it.
# =====================================================================================

_LOOKALIKE_SEED = 100
_LOOKALIKE_WINDOW_FLOOR = 918
# (height, family, wobble). One photo-ish card far away in signature space, then a text-card
# family: the SURVIVOR above the floor and two near-twins below it.
_LOOKALIKE_LAYOUT = ((1109, "photo", 1.0), (940, "text", 0.0), (860, "text", 0.9),
                     (880, "text", 1.4))
# A card of the same text family that is on NO page this payload describes: 960px, so the sheet
# renders it as a full-height window, and far enough from the survivor to sit outside the bound
# the look-alikes impose (12.78 measured) while staying inside the 16.99 the survivor would have
# had if they were pruned away. That gap is the defect, and it is what this fixture exists to pin.
_LOOKALIKE_FOREIGN = (960, "text", 8.0)


def _lookalike_card(height: int, family: str, wobble: float, seed: int):
    """One card of a family of near-identical prompt cards, pushed `wobble` away from the family.

    Text cards and photographs are deliberately different shapes here for the reason doc 5.8
    gives: a text card is a near-uniform background with sparse structure, so a page's text cards
    sit close together in signature space while a photograph sits far from all of them.
    """
    rng = np.random.default_rng(seed)
    width = _CARD_X1 - _CARD_X0
    rows, cols = np.mgrid[0:height, 0:width]
    if family == "photo":
        base = 60 * np.sin(2 * np.pi * rows / 55) * np.cos(2 * np.pi * cols / 70) + 30
    else:
        base = 34 * np.sin(2 * np.pi * rows / 120) * np.cos(2 * np.pi * cols / 260)
    card = 120 + base + wobble * rng.integers(-30, 31, size=(height, width))
    return np.clip(card, 0, 210).astype(np.uint8)


def _lookalike_world():
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    y = _PAGE_TOP_GAP
    for i, (height, family, wobble) in enumerate(_LOOKALIKE_LAYOUT):
        y0, y1 = y, y + height
        world[y0:y1, _CARD_X0:_CARD_X1] = _lookalike_card(height, family, wobble,
                                                          _LOOKALIKE_SEED + i)
        for k in range(_CORNER_RADIUS_PX):
            dy = _CORNER_RADIUS_PX - k
            inset = int(math.ceil(_CORNER_RADIUS_PX
                                  - math.sqrt(max(0.0, _CORNER_RADIUS_PX ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for row in (y0 + k, y1 - 1 - k):
                world[row, _CARD_X0:_CARD_X0 + inset] = col[row]
                world[row, _CARD_X1 - inset:_CARD_X1] = col[row]
        th, tw = _TEMPLATE.shape
        cy = y1 - _HEART_ABOVE_BOTTOM
        world[cy - th // 2: cy - th // 2 + th,
              _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = _TEMPLATE
        y = y1 + _GUTTER
    return world


def _lookalike_payload():
    """The real index and the real crops over the look-alike page, built once."""
    if "lookalike" not in _CACHE:
        world = _lookalike_world()
        frames = []
        for scroll in _SCROLLS:
            gray = world[scroll:scroll + _H].copy()
            gray[:_BAND0] = _CHROME_TOP
            gray[_BAND1:] = _CHROME_BOTTOM
            ok, buf = cv2.imencode(".png", gray)
            assert ok
            frames.append(buf.tobytes())
        index = item_index.build_item_index(
            frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)
        assert index.usable, index.failures
        _CACHE["lookalike"] = item_crops.build_item_payload(frames, index)
    return _CACHE["lookalike"]


def _foreign_sheet() -> bytes:
    """A card this payload does not contain, rendered as an open sheet. Never indexed, never
    cropped, never numbered — the whole point is that it is not in the list being asked about."""
    height, family, wobble = _LOOKALIKE_FOREIGN
    gray = _lookalike_card(height, family, wobble, _LOOKALIKE_SEED + 900)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert ok
    return paint_sheet(buf.tobytes())


# =====================================================================================
# The fixture is the page the assertions below assume
# =====================================================================================

def test_the_fixture_page_is_the_page_the_assertions_below_assume():
    """A guard on the world, not on the module: everything below reads its ground truth off
    these numbers, so a drifted fixture would look like a verification bug."""
    payload = _payload()
    assert payload.usable, payload.failures
    assert payload.item_count == 4 and payload.context_count == 1
    assert payload.translation == (1, 2, 3, 4)
    # The last card is taller than the sheet's content region once scaled, so it is rendered as a
    # bottom-anchored WINDOW rather than whole. That is what makes the window rule testable here.
    tall = payload.item(4)
    assert tall.signature.height * _SHEET_PREVIEW_W / tall.signature.width > _SHEET_PREVIEW_MAX_H
    assert payload.item(1).signature.height * _SHEET_PREVIEW_W / payload.item(1).signature.width \
        <= _SHEET_PREVIEW_MAX_H, "at least one card must fit whole, as the control"


# =====================================================================================
# Locating the preview
# =====================================================================================

def test_the_locator_finds_the_preview_at_the_measured_sheet_geometry():
    preview = item_verify.locate_sheet_preview(_sheet_for(1))
    assert preview.x0 == _SHEET_PREVIEW_X0
    assert preview.width == _SHEET_PREVIEW_W
    # Within the card's own corner radius of the painted top edge: the first rows of a rounded
    # rect are narrower than the full card, so the width test picks them up a few rows down. The
    # scale sweep is what absorbs the couple of rows that costs (see the sweep tests below).
    assert 0 <= preview.y0 - _SHEET_PREVIEW_Y0 <= _CORNER_RADIUS_PX
    assert preview.height > item_verify._PREVIEW_MIN_HEIGHT_PX


def test_the_locator_refuses_a_profile_screen_because_a_card_sits_at_the_wrong_margin():
    """The test that is not redundant with width and height. A profile card is 974px wide and up
    to 1114 tall, so it passes both; what it does not do is sit at the sheet's 95px indent. Without
    this the verifier would happily compare a screen with no sheet on it against the stored crops
    of the very profile it is showing, and confirm an item nobody tapped."""
    with pytest.raises(item_verify.SheetVerificationError) as exc:
        item_verify.locate_sheet_preview(_frame(_STEP * 3))
    assert "profile screen rather than an open sheet" in str(exc.value)
    assert str(_CARD_X0) in str(exc.value)


def test_the_locator_refuses_a_frame_with_nothing_wide_enough_on_it():
    flat = np.full((_H, _W), _SHEET_BG, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", flat)
    assert ok
    with pytest.raises(item_verify.SheetVerificationError) as exc:
        item_verify.locate_sheet_preview(buf.tobytes())
    assert "no comment-sheet item preview" in str(exc.value)


def test_the_locator_refuses_bytes_that_are_not_an_image():
    with pytest.raises(item_verify.SheetVerificationError) as exc:
        item_verify.locate_sheet_preview(b"not a png")
    assert "did not decode" in str(exc.value)


# =====================================================================================
# THE HEADLINE: every item verifies against its own sheet, and against no other
# =====================================================================================

def test_every_item_verifies_on_its_own_sheet_and_every_other_number_is_refused():
    """Doc 5.6's whole contract, as an NxN matrix over the real crop machinery: for each item's
    re-rendered sheet, its own number MATCHES and every other number does not."""
    payload = _payload()
    numbers = [c.number for c in payload.items]
    accepted_wrong = []
    diagonal, off_diagonal = [], []
    for shown in numbers:
        sheet = _sheet_for(shown)
        for asked in numbers:
            verdict = item_verify.verify_sheet_item(sheet, payload, asked)
            mine = next(c for c in verdict.comparisons if c.number == asked)
            (diagonal if asked == shown else off_diagonal).append(mine.distance)
            if asked == shown:
                assert verdict.matched, verdict.reason
                assert verdict.nearest_index == shown
            elif verdict.matched:
                accepted_wrong.append((shown, asked))
    assert accepted_wrong == []
    assert max(diagonal) < min(d for d in off_diagonal if d is not None), (
        f"diagonal {max(diagonal):.3f} must sit well below the nearest off-diagonal "
        f"{min(d for d in off_diagonal if d is not None):.3f}")


def test_a_mismatch_names_the_intended_item_and_the_actual_one():
    payload = _payload()
    verdict = item_verify.verify_sheet_item(_sheet_for(2), payload, 3)
    assert verdict.state == item_verify.VERIFY_MISMATCH
    assert verdict.nearest_index == 2
    assert "showing model item 2" in verdict.reason
    assert "item 3 the opener was written about" in verdict.reason
    assert "Nothing is sent" in verdict.reason


def test_a_tall_card_is_matched_against_the_bottom_of_its_crop_and_not_the_top():
    """The measured render rule and its direction. A card taller than the sheet's content region
    is shown BOTTOM first, so the reference has to be the crop's last rows: on the real captures
    the bottom window sits 1.011 and 0.533 grey levels from the sheet where the whole card sits
    9.296 and 18.836 and the TOP window 20.108 and 31.661.

    What is asserted is the RATIO, not a refusal, and the difference is worth stating. Two windows
    of one card overlap by ~80%, so how far apart they are is a property of how much vertical
    structure that particular card has -- a lot on a photograph (20x and 59x on the two real
    ones), less on a flat one. Verification does not depend on the wrong window being refused; it
    depends on the RIGHT window being used, which is what this pins.
    """
    payload = _payload()
    right_way = item_verify.verify_sheet_item(_sheet_for(4), payload, 4)
    assert right_way.matched, right_way.reason
    mine = next(c for c in right_way.comparisons if c.number == 4)
    assert mine.window_px < mine.crop_px, "this fixture card must not fit the sheet whole"
    wrong_end = item_verify.verify_sheet_item(_sheet_for(4, anchor="top"), payload, 4)
    theirs = next(c for c in wrong_end.comparisons if c.number == 4)
    assert theirs.distance > 3 * mine.distance


def test_the_scale_tolerance_absorbs_a_measurement_error_in_the_preview_width():
    """A percent of error in the located preview width becomes a percent of error in the derived
    window, which on an 800-row window is eight rows — the exact size of the miss measured on the
    tallest real card. The sweep is what turns that from a 4.6 grey-level penalty into 0.9."""
    payload = _payload()
    narrower = _sheet_for(4, preview_w=_SHEET_PREVIEW_W - 8)
    assert item_verify.verify_sheet_item(narrower, payload, 4).matched


def test_the_sweep_does_not_pull_a_wrong_item_into_range():
    """The other half of the tolerance question doc 5.6 asks to be quantified: a band wide enough
    for the render is only safe if it is nowhere near wide enough for a different item. Measured on
    the real captures the sweep moves a wrong card by at most 0.9 grey levels; here every wrong
    item stays outside its bound by a wide margin."""
    payload = _payload()
    verdict = item_verify.verify_sheet_item(_sheet_for(1), payload, 1)
    for comparison in verdict.comparisons:
        if comparison.number == 1 or comparison.distance is None:
            continue
        assert comparison.distance > 2 * comparison.bound


# =====================================================================================
# The animated card: explicit detection BEFORE the tap, never a wider tolerance
# =====================================================================================

def _payload_with_drift(number: int, drift: float):
    """The real payload with one item's measured re-observation drift replaced.

    Substituted rather than simulated because what is under test is the DECISION doc 5.6 makes
    about a drift value, not `item_crops`' ability to measure one — that has its own tests, and
    the numbers here (25.009 against a 47.861 nearest neighbour) are a real profile's.
    """
    payload = _payload()
    crops = tuple(dataclasses.replace(c, signature_drift=drift) if c.number == number else c
                  for c in payload.crops)
    return dataclasses.replace(payload, crops=crops)


def test_an_item_whose_own_crop_drifts_too_far_is_refused_before_anything_is_tapped():
    """Doc 5.4's animated-card class, handled by explicit detection rather than a tolerance band.
    The real profile's animated card drifts 25.009 grey levels against a 47.861 distance to its
    nearest neighbour, so a band wide enough to accept it is wide enough to accept a different
    item — and on the other calibration profile the two most alike items are 4.661 apart."""
    payload = _payload()
    nearest = payload.item(2).nearest_item_distance
    blocked = _payload_with_drift(2, nearest)          # drift equal to the whole neighbour gap
    reason = item_verify.verification_blocker(blocked, 2)
    assert reason
    assert "cannot be verified on the like sheet" in reason
    assert "nothing is tapped" in reason
    # ...and only that item. The other three are untouched evidence and stay verifiable.
    assert [n for n in (1, 3, 4) if item_verify.verification_blocker(blocked, n)] == []


def test_a_still_card_is_not_blocked_and_the_measured_headroom_is_real():
    payload = _payload()
    for number in (c.number for c in payload.items):
        assert item_verify.verification_blocker(payload, number) == ""


def test_the_post_tap_check_reports_unverifiable_rather_than_matching_when_the_bound_is_too_tight():
    """The same test re-made in the space the comparison actually happened in. It exists because
    `verification_blocker` uses the payload's coarser stored numbers, so an item can slip past it
    and still have no discriminating power once the windows are cut."""
    payload = _payload()
    blocked = _payload_with_drift(1, payload.item(1).nearest_item_distance * 4)
    verdict = item_verify.verify_sheet_item(_sheet_for(1), blocked, 1)
    assert verdict.state == item_verify.VERIFY_UNVERIFIABLE
    assert not verdict.matched
    assert "cannot be told from its neighbours" in verdict.reason


def test_verification_blocker_raises_rather_than_answering_for_an_item_that_does_not_exist():
    with pytest.raises(item_verify.SheetVerificationError):
        item_verify.verification_blocker(_payload(), 99)


def test_verify_raises_rather_than_answering_for_an_item_that_does_not_exist():
    with pytest.raises(item_verify.SheetVerificationError):
        item_verify.verify_sheet_item(_sheet_for(1), _payload(), 0)


# =====================================================================================
# The API's own guard rails
# =====================================================================================

def test_a_verdict_has_no_truth_value():
    """`if verify_sheet_item(...):` would read every outcome as a match, including the one that
    means the question could not be answered. Same rule as ScrollTopVerdict and IdentityVerdict."""
    verdict = item_verify.verify_sheet_item(_sheet_for(1), _payload(), 1)
    with pytest.raises(TypeError) as exc:
        bool(verdict)
    assert "test .matched" in str(exc.value)


def test_the_verdict_carries_the_evidence_a_stop_record_needs():
    verdict = item_verify.verify_sheet_item(_sheet_for(3), _payload(), 3)
    assert verdict.matched
    assert verdict.grid == item_verify._VERIFY_GRID
    assert verdict.preview.width == _SHEET_PREVIEW_W
    assert len(verdict.comparisons) == _payload().item_count
    mine = next(c for c in verdict.comparisons if c.number == 3)
    assert mine.distance is not None and mine.bound is not None
    assert mine.nearest_other is not None
    assert mine.distance < mine.bound <= mine.nearest_other


def test_with_no_other_item_to_bound_against_the_bound_comes_off_the_reproduction_noise():
    """A degenerate list, and the one case where the halving has nothing to halve. The same
    fraction then scales the expected noise UP instead, so there is still exactly one constant --
    and the bound it produces (2 x 1.85) sits 14x under the >=51.7 a different card measured."""
    payload = _payload()
    only = dataclasses.replace(payload, crops=(payload.item(1),))
    assert only.item_count == 1
    verdict = item_verify.verify_sheet_item(_sheet_for(1), only, 1)
    mine = next(c for c in verdict.comparisons if c.number == 1)
    assert mine.nearest_other is None
    assert mine.bound == pytest.approx(
        item_verify._SHEET_RENDER_DRIFT / item_verify._SEPARATION_FRACTION)
    assert verdict.matched, verdict.reason
    # ...and a different card on the sheet is still refused, which is what makes it a check.
    assert not item_verify.verify_sheet_item(_sheet_for(3), only, 1).matched


# =====================================================================================
# THE NEIGHBOUR SET: every numbered item bounds every other one, whatever the sheet's height
# =====================================================================================

def test_the_lookalike_page_is_the_page_the_assertions_below_assume():
    """A guard on the second world. Everything below reads its ground truth off these numbers,
    and the whole construction turns on ONE height comparison — the survivor above the window
    floor, its two near-twins below it."""
    payload = _lookalike_payload()
    assert payload.usable, payload.failures
    assert payload.translation == (1, 2, 3, 4)
    heights = {c.number: c.height for c in payload.items}
    assert heights[2] >= _LOOKALIKE_WINDOW_FLOOR, "item 2 must survive a full-height window"
    assert heights[3] < _LOOKALIKE_WINDOW_FLOOR and heights[4] < _LOOKALIKE_WINDOW_FLOOR, (
        "items 3 and 4 must be too short to be a full-height window — that is the pruning this "
        "fixture exists to exercise")
    # ...and they really are item 2's nearest neighbours, which is what makes their removal
    # change the answer rather than merely change a number.
    assert payload.item(2).nearest_item_distance < payload.item(1).nearest_item_distance


def test_an_item_too_short_to_be_the_sheet_still_bounds_every_other_item():
    """THE ROOT CAUSE OF A MEASURED FALSE ACCEPT (doc 5.6's 2026-08-12 addendum).

    An item shorter than the window the sheet is rendering cannot be what is on screen, so it has
    no `distance` — that part was always right. What was wrong is that it also lost its
    REFERENCE, and the accept bound is derived from how far an item sits from the nearest OTHER
    reference. A tall sheet therefore pruned a profile's short cards out of every surviving
    item's bound, and the bound grew by exactly as much as those cards were near.

    The neighbour set is a property of the payload. It must not depend on how tall the sheet in
    front of us happens to be.
    """
    payload = _lookalike_payload()
    verdict = item_verify.verify_sheet_item(_foreign_sheet(), payload, 2)
    by_number = {c.number: c for c in verdict.comparisons}
    assert [n for n, c in by_number.items() if c.distance is None] == [3, 4]
    for number in (3, 4):
        assert by_number[number].nearest_other is not None, (
            f"item {number} is too short to BE the sheet, but it is still a stored item this "
            f"profile can be confused with and it must still bound the others")
    # ...and item 2's own bound is the tightest of ALL three others, not of the tall ones only.
    pairwise = []
    for other in (1, 3, 4):
        two = dataclasses.replace(payload, crops=(payload.item(2), payload.item(other)))
        alone = item_verify.verify_sheet_item(_foreign_sheet(), two, 2)
        pairwise.append(next(c for c in alone.comparisons if c.number == 2).nearest_other)
    assert min(pairwise) == pytest.approx(by_number[2].nearest_other)


def test_a_sheet_nothing_could_be_measured_against_says_so_instead_of_naming_item_none():
    """The operator reads this sentence verbatim on the hub and in the console.

    The 2026-08-15 observe run printed "The nearest stored item is None" — the bare Python value,
    which reads as an item NAMED None rather than as "no item in this payload could be compared
    with the sheet at all". Both facts are refusals, but only one of them is true here, and a
    refusal the operator misreads is the failure this project spends its refusals to avoid.
    """
    payload = _lookalike_payload()
    # Both survivors are too short to be what this sheet renders, so nothing is measurable.
    unmeasurable = dataclasses.replace(payload, crops=(
        dataclasses.replace(payload.item(3), number=1),
        dataclasses.replace(payload.item(4), number=2)))
    verdict = item_verify.verify_sheet_item(_foreign_sheet(), unmeasurable, 1)

    assert verdict.state == item_verify.VERIFY_MISMATCH
    assert verdict.nearest_index is None
    assert [c.distance for c in verdict.comparisons] == [None, None]
    assert verdict.reason.endswith("No stored item could be measured against this sheet either")
    assert "is None" not in verdict.reason

    # ...and with a measurable neighbour the sentence still names it, which is the other fact.
    named = item_verify.verify_sheet_item(_foreign_sheet(), payload, 3)
    assert named.state == item_verify.VERIFY_MISMATCH
    assert named.nearest_index is not None
    assert named.reason.endswith(f"The nearest stored item is {named.nearest_index}")


def test_a_card_that_is_not_in_the_payload_at_all_is_refused_by_every_number():
    """The failure this closes, end to end: a sheet showing a card from ANOTHER PROFILE.

    Measured on the real captures before the fix: a stale payload plus a foreign card returned
    VERIFY_MATCH and the driver typed the opener and sent the like, 10 times in 540 comparisons.
    Here the foreign card sits 12.78 grey levels from the survivor against a 9.12 bound.
    """
    payload = _lookalike_payload()
    sheet = _foreign_sheet()
    accepted = [n for n in payload.translation
                if item_verify.verify_sheet_item(sheet, payload, n).matched]
    assert accepted == []


def test_removing_the_short_lookalikes_is_what_used_to_let_the_foreign_card_through():
    """The mechanism, isolated: the SAME sheet and the SAME item, accepted when the look-alikes
    are absent from the list and refused when they are present. The height filter used to remove
    them for us, which is why this was reachable without anybody assembling a partial payload.

    The accept on the two-item list is not a bug being asserted as correct — it is doc 5.6's rule
    working exactly as specified on a profile whose only other item is far away. A relative bound
    cannot do better; see item_verify's "THIS IS A CLOSED-SET TEST" for what that leaves open and
    why an absolute ceiling is a calibration task rather than a constant picked here.
    """
    payload = _lookalike_payload()
    sheet = _foreign_sheet()
    assert not item_verify.verify_sheet_item(sheet, payload, 2).matched
    without = dataclasses.replace(payload, crops=(payload.item(1), payload.item(2)))
    assert item_verify.verify_sheet_item(sheet, without, 2).matched


def test_the_accept_bound_is_half_the_distance_to_the_nearest_other_item():
    """The 0.5 is a proof rather than a tuning: `CropSignature.distance` is a scaled L1 norm, so
    a sheet within half an item's neighbour distance of it is, by the triangle inequality, nearer
    to that item than to any other. This pins the relationship, not the number's provenance."""
    verdict = item_verify.verify_sheet_item(_sheet_for(2), _payload(), 2)
    mine = next(c for c in verdict.comparisons if c.number == 2)
    assert mine.bound == pytest.approx(item_verify._SEPARATION_FRACTION * mine.nearest_other)


def test_an_absolute_sheet_ceiling_is_required_in_addition_to_relative_separation():
    """A foreign/unknown card can be nearest and relatively close; calibration supplies the cap."""
    sheet, payload = _sheet_for(2), _payload()
    relative_only = item_verify.verify_sheet_item(sheet, payload, 2)
    assert relative_only.matched
    capped = item_verify.verify_sheet_item(
        sheet, payload, 2, absolute_max_dist=relative_only.distance / 2)
    assert not capped.matched
    assert "absolute" in capped.reason


@pytest.mark.parametrize("ceiling", [math.nan, math.inf, -math.inf, 0, -1, 14.91, 20])
def test_an_invalid_absolute_sheet_ceiling_is_a_verification_error_not_a_disabled_guard(ceiling):
    """NaN/Inf and unsafe values used to make the final ``>=`` check silently false."""
    with pytest.raises(item_verify.SheetVerificationError, match="absolute_max_dist"):
        item_verify.verify_sheet_item(_sheet_for(2), _payload(), 2,
                                      absolute_max_dist=ceiling)


# --- the verification reference's grey space (2026-08-27 live false refusal) --------------


def test_the_verification_reference_is_windowed_not_the_colour_image():
    """Regression for the live halt: a correct sheet refused at 10.283 against a 7.00 ceiling.

    `_decode_crop` used to grey-decode `crop.image` and argue that was safe because
    `item_crops.signature_of` makes the same `cv2.IMREAD_GRAYSCALE` call. Calling one decoder on
    two differently encoded PNGs is not one grey space: the sheet is the RGBA/sRGB frame the
    device wrote, while `image` is a plain RGB PNG this codebase re-encoded, and libpng's
    rgb-to-gray does not reproduce it. The distances that came out were in neither side's units.

    Painting the colour image and the greyscale reference with DIFFERENT content is how the test
    proves which one is being measured — the real pair always describes the same pixels, so a
    version that read `image` would score the sheet against content the reference does not have.
    """
    crop = _payload().item(1)
    reference = cv2.imdecode(np.frombuffer(crop.verify_image, np.uint8), cv2.IMREAD_GRAYSCALE)
    decoy = np.zeros_like(cv2.imdecode(np.frombuffer(crop.image, np.uint8), cv2.IMREAD_COLOR))
    ok, encoded_decoy = cv2.imencode(".png", decoy)
    assert ok
    # Signature stays with the reference, which is what `_check_reference_provenance` asserts;
    # only the model's colour image is replaced.
    swapped = dataclasses.replace(crop, image=encoded_decoy.tobytes())
    payload = dataclasses.replace(
        _payload(),
        crops=tuple(swapped if c.number == 1 else c for c in _payload().crops))

    verdict = item_verify.verify_sheet_item(paint_sheet(crop.image), payload, 1)
    assert verdict.matched, verdict.reason
    assert reference.ndim == 2


def test_a_crop_without_a_greyscale_reference_is_refused_rather_than_measured():
    """Fail closed, never fall back to `image`: that fallback IS the bug, and it looks like a
    number rather than like a failure."""
    payload = _payload()
    stripped = dataclasses.replace(payload.item(1), verify_image=None)
    payload = dataclasses.replace(
        payload, crops=tuple(stripped if c.number == 1 else c for c in payload.crops))
    with pytest.raises(item_verify.SheetVerificationError,
                       match="no greyscale verification reference"):
        item_verify.verify_sheet_item(_sheet_for(1), payload, 1)


def _with_srgb_chunk(png: bytes) -> bytes:
    """The same PNG with an `sRGB` chunk, which is the whole of what the live fault turned on.

    An Android screencap carries one; `cv2.imencode` writes none. `cv2.IMREAD_GRAYSCALE` is
    sRGB-aware, so the presence of this 1-byte chunk changes the grey it produces from identical
    RGB samples. No painted fixture in this repo can reproduce that on its own -- every synthetic
    frame here is written by `cv2.imencode` and therefore has no chunk on either side of the
    comparison, which is precisely why the whole corpus passed while a live run refused a correct
    card. So the regression test writes the chunk itself.
    """
    length = int.from_bytes(png[8:12], "big")
    end_of_ihdr = 8 + 12 + length
    payload = b"sRGB" + bytes([0])                       # rendering intent 0: perceptual
    chunk = (len(payload[4:])).to_bytes(4, "big") + payload + (
        zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
    return png[:end_of_ihdr] + chunk + png[end_of_ihdr:]


def test_a_reference_in_the_wrong_grey_space_is_refused_before_any_distance_is_believed():
    """The guard for the fault class itself, reproduced in the shape it actually shipped in.

    The stored signature is made from the frame the device wrote, which carries an `sRGB` chunk;
    the reference here is the same rows without it, which is exactly what re-encoding the crop in
    colour produced. The two then describe the same pixels in two different greys, every distance
    below is in the wrong units, and a wrongly-scaled distance is indistinguishable from an
    honest one by inspection. So it must be refused, not measured.
    """
    payload = _payload()
    crop = payload.item(1)
    grey = cv2.imdecode(np.frombuffer(crop.verify_image, np.uint8), cv2.IMREAD_GRAYSCALE)
    colour = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    # Saturate the colour channels apart: an sRGB-aware and an sRGB-blind conversion agree
    # exactly on neutral pixels, and the live fault was on a dark, heavily saturated photo.
    colour[:, :, 0] = np.clip(colour[:, :, 0].astype(int) + 60, 0, 255).astype(np.uint8)
    colour[:, :, 2] = np.clip(colour[:, :, 2].astype(int) - 60, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", colour)
    assert ok
    tagged, untagged = _with_srgb_chunk(encoded.tobytes()), encoded.tobytes()
    signed = item_crops.signature_of(
        tagged, y0=0, y1=grey.shape[0], x0=0, x1=grey.shape[1], grid=payload.signature_grid)
    # The premise: one ancillary chunk, and only that, moves the decode.
    assert not np.array_equal(
        cv2.imdecode(np.frombuffer(tagged, np.uint8), cv2.IMREAD_GRAYSCALE),
        cv2.imdecode(np.frombuffer(untagged, np.uint8), cv2.IMREAD_GRAYSCALE))

    mismatched = dataclasses.replace(crop, verify_image=untagged, signature=signed)
    payload = dataclasses.replace(
        payload, crops=tuple(mismatched if c.number == 1 else c for c in payload.crops))
    with pytest.raises(item_verify.SheetVerificationError,
                       match="does not reproduce the signature stored for the same rows"):
        item_verify.verify_sheet_item(_sheet_for(1), payload, 1)


def test_every_real_crop_reproduces_its_stored_signature_from_its_reference():
    """The invariant the guard exists to protect, asserted on the whole real payload.

    `build_item_payload` cuts image, reference and signature from the same rows of the same
    frame, so the reference must reduce back to the signature EXACTLY — the reference is a
    lossless 8-bit greyscale PNG, so this is 0.0 rather than merely small.
    """
    payload = _payload()
    for crop in payload.items:
        gray = cv2.imdecode(np.frombuffer(crop.verify_image, np.uint8), cv2.IMREAD_GRAYSCALE)
        reproduced = item_crops._signature_from_gray(
            gray, grid=payload.signature_grid, cv2=cv2, np=np)
        assert crop.signature.distance(reproduced) == 0.0, crop.number


def test_the_one_item_ceiling_stays_between_the_measured_correct_and_foreign_populations():
    """Pin the 2026-08-28 re-fit against the corpus it was measured on.

    The previous 7.00 was fitted on two renders whose numbers were decode bias rather than render
    penalty (see `_INLINE_COMPOSER_ONE_ITEM_MAX_DIST`'s own comment and `_decode_crop`). The
    replacement is measured over 148 real like-sheet verifications and 218 cross-profile pairs,
    and this test exists so a later edit cannot drift it back into either population without the
    failure being obvious.

    Numbers are the measured extremes, not re-derived here: re-deriving them would need the
    gitignored device corpus, and the whole point is that they came from real frames.
    """
    worst_correct_observed = 1.380          # max over 148 correct sheets, corrected grey space
    nearest_foreign_observed = 40.823       # min over 218 cross-profile pairs
    ceiling = item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST

    assert ceiling > worst_correct_observed, (
        "the ceiling would refuse a correct sheet the corpus actually produced")
    assert ceiling >= 2.0 * worst_correct_observed, (
        "less than 2x over the worst measured correct reading leaves no room for locator error")
    assert ceiling < nearest_foreign_observed, "the ceiling would accept a real foreign card"
    # It must also stay under the absolute foreign-acceptance clamp, which is a different guard.
    assert ceiling < item_verify._SHEET_FALSE_MATCH_DISTANCE


def test_lowering_the_one_item_ceiling_can_only_turn_accepts_into_refusals():
    """The safety argument for re-fitting it at all, checked rather than asserted in prose.

    The constant feeds exactly one comparison, so a smaller value is monotone: it can never
    convert a refusal into an acceptance. Verified by running the same sheet at a ceiling above
    and below its own distance.
    """
    payload = _payload()
    frame = _sheet_for(1)
    surface = _inline_surface()
    one_item = dataclasses.replace(
        payload, crops=tuple(c for c in payload.crops if c.number == 1))

    verdict = item_verify.verify_sheet_item(frame, one_item, 1, composer_surface=surface)
    assert verdict.matched, verdict.reason
    assert verdict.bound == item_verify._INLINE_COMPOSER_ONE_ITEM_MAX_DIST

    tightened = verdict.distance / 2 if verdict.distance else 0.001
    with mock.patch.object(item_verify, "_INLINE_COMPOSER_ONE_ITEM_MAX_DIST", tightened):
        refused = item_verify.verify_sheet_item(frame, one_item, 1, composer_surface=surface)
    assert not refused.matched, "a tighter ceiling must refuse what it used to accept"
    assert refused.state == item_verify.VERIFY_MISMATCH
