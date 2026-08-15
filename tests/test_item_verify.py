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


def _paint_inline_reframe(crop_image: bytes, *, start: int = 37, rows: int = 933) -> bytes:
    """Synthetic 9.134 selected-photo composer, including its removed card-heart lane.

    The real Malaika regression is a complete square source card rendered as a 933-row interior
    window above the inline controls (not the legacy bottom window).  Its profile-card heart is
    in the source's lower-right lane; the selected preview intentionally has ordinary image
    pixels there.  This fixture carries the same geometry without embedding a real profile.
    """
    card = cv2.imdecode(np.frombuffer(crop_image, np.uint8), cv2.IMREAD_COLOR)
    source_x1 = card.shape[1] - round(card.shape[1] * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    assert 0 <= start and start + rows <= card.shape[0]
    preview = np.full((_SHEET_PREVIEW_MAX_H, _SHEET_PREVIEW_W, 3), 145, dtype=np.uint8)
    # The left photo area is exactly the full-width, bounded source window verifier is allowed to
    # search.  The right side represents the layout's selected-photo control-free lane and is
    # excluded symmetrically by the inline verifier.
    left_w = _SHEET_PREVIEW_W - round(
        _SHEET_PREVIEW_W * item_verify._INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
    preview[:, :left_w] = cv2.resize(card[start:start + rows, :source_x1],
                                     (left_w, _SHEET_PREVIEW_MAX_H),
                                     interpolation=cv2.INTER_AREA)
    canvas = np.full((_H, _W, 3), _SHEET_BG, dtype=np.uint8)
    canvas[_SHEET_PREVIEW_Y0:_SHEET_PREVIEW_Y0 + _SHEET_PREVIEW_MAX_H,
           _SHEET_PREVIEW_X0:_SHEET_PREVIEW_X0 + _SHEET_PREVIEW_W] = preview
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def _inline_surface_for(frame: bytes) -> ComposerSurface:
    preview = item_verify.locate_sheet_preview(frame)
    comment = Rect(preview.x0, preview.y1 + 20, preview.x1, preview.y1 + 198)
    send = Rect(390, comment.y1 + 15, 985, comment.y1 + 124)
    return ComposerSurface("hinge_inline_v1", comment, send, (695, send.y0 + 50))


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
    assert "Nothing was typed" in verdict.reason


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
