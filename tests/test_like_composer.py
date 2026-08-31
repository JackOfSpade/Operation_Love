"""Synthetic, fail-closed coverage for Hinge 9.134.0's inline like composer.

The production measurements are geometry only.  These canvases contain no profile data: they
paint the measured input and CTA around the shipped ``Send Like`` glyph, then remove one piece
of evidence at a time.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from operation_love.drivers import hinge
from operation_love.drivers.like_composer import (
    ComposerDetectionError, Rect, locate_inline_composer)


_W, _H = 1080, 2400
_BG = 249
_COMMENT = Rect(95, 1597, 985, 1775)
_SEND = Rect(390, 1807, 985, 1916)
_CONFIRM_CENTER = (695, 1856)
# Keyboard-open Pixel 7a layout observed on the current phone.  Only measured geometry is kept
# here; the source screenshot remains private and is never read by this test.
_KEYBOARD_COMMENT = Rect(95, 1123, 985, 1301)
_KEYBOARD_SEND = Rect(390, 1333, 985, 1442)
_KEYBOARD_CONFIRM_CENTER = (695, 1383)
_TEMPLATE = hinge._load_template("hinge_send_like.png")
assert _TEMPLATE is not None, "the shipped Send Like glyph is part of this detector's contract"


def _png(gray: np.ndarray, *, color: bool) -> bytes:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if color else gray
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _paint_glyph(canvas: np.ndarray, *, center=_CONFIRM_CENTER) -> None:
    height, width = _TEMPLATE.shape
    x, y = center
    canvas[y - height // 2:y - height // 2 + height,
           x - width // 2:x - width // 2 + width] = _TEMPLATE


def _scaled_rect(rect: Rect, width: int, height: int) -> Rect:
    return Rect(round(rect.x0 / _W * width), round(rect.y0 / _H * height),
                round(rect.x1 / _W * width), round(rect.y1 / _H * height))


def _inline_composer(*, input_present=True, cta_present=True, glyph_center=_CONFIRM_CENTER,
                     color=False, width=_W, height=_H) -> bytes:
    canvas = np.full((height, width), _BG, dtype=np.uint8)
    comment = _scaled_rect(_COMMENT, width, height)
    send = _scaled_rect(_SEND, width, height)
    if input_present:
        # Long, dark top/bottom lines are an outlined input, not a large card-shaped region.
        canvas[comment.y0:comment.y0 + 2, comment.x0:comment.x1] = 222
        canvas[comment.y1 - 2:comment.y1, comment.x0:comment.x1] = 222
    if cta_present:
        canvas[send.y0:send.y1, send.x0:send.x1] = 228
    _paint_glyph(canvas, center=glyph_center)
    return _png(canvas, color=color)


def _keyboard_open_composer(*, color=False) -> bytes:
    """Public synthetic layout for the keyboard-open current-phone state."""
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    canvas[_KEYBOARD_COMMENT.y0:_KEYBOARD_COMMENT.y0 + 2,
           _KEYBOARD_COMMENT.x0:_KEYBOARD_COMMENT.x1] = 222
    canvas[_KEYBOARD_COMMENT.y1 - 2:_KEYBOARD_COMMENT.y1,
           _KEYBOARD_COMMENT.x0:_KEYBOARD_COMMENT.x1] = 222
    canvas[_KEYBOARD_SEND.y0:_KEYBOARD_SEND.y1, _KEYBOARD_SEND.x0:_KEYBOARD_SEND.x1] = 228
    _paint_glyph(canvas, center=_KEYBOARD_CONFIRM_CENTER)
    return _png(canvas, color=color)


def _scrolled_priority_like_composer() -> bytes:
    """The fully visible post-review layout from the reported live error frame.

    A human inspected the selected item during AUTO review and returned to the
    still-open Send Priority Like sheet.  The sheet's CTA was completely
    visible, but at these measured coordinates rather than the keyboard-open
    geometry.  This contains only control geometry and the shipped CTA glyph.
    """
    comment = Rect(95, 1616, 985, 1794)
    send = Rect(390, 1826, 985, 1935)
    confirm = (690, 1882)
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    canvas[comment.y0:comment.y0 + 2, comment.x0:comment.x1] = 222
    canvas[comment.y1 - 2:comment.y1, comment.x0:comment.x1] = 222
    canvas[send.y0:send.y1, send.x0:send.x1] = 228
    _paint_glyph(canvas, center=confirm)
    return _png(canvas, color=False)


def _keyboard_open_composer_with_selected_photo_edge() -> bytes:
    """Synthetic keyboard layout with a wide selected-photo edge above the input.

    This mirrors only the measured geometry of a captured composer: a photo's
    lower edge is another wide dark run in the comment-input search band.  No
    profile pixels or identifying content are retained.
    """
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    # The selected photo ends just above the outlined comment field.
    canvas[1046:1092, 95:985] = 190
    canvas[_KEYBOARD_COMMENT.y0:_KEYBOARD_COMMENT.y0 + 2,
           _KEYBOARD_COMMENT.x0:_KEYBOARD_COMMENT.x1] = 222
    canvas[_KEYBOARD_COMMENT.y1 - 2:_KEYBOARD_COMMENT.y1,
           _KEYBOARD_COMMENT.x0:_KEYBOARD_COMMENT.x1] = 222
    canvas[_KEYBOARD_SEND.y0:_KEYBOARD_SEND.y1, _KEYBOARD_SEND.x0:_KEYBOARD_SEND.x1] = 228
    _paint_glyph(canvas, center=_KEYBOARD_CONFIRM_CENTER)
    return _png(canvas, color=False)


def _keyboard_open_composer_with_text_selection_handle() -> bytes:
    """Synthetic keyboard layout with Android's text-selection handle in the control gap.

    Measured on the Pixel 7a while the owner edited a generated opener before
    sending it: tapping into the comment field draws a ~56px teardrop that
    tapers across the whole gap, touching the input's lower border and the CTA
    below it.  Only that geometry is reproduced; no profile pixels are kept.
    """
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    canvas[_KEYBOARD_COMMENT.y0:_KEYBOARD_COMMENT.y0 + 2,
           _KEYBOARD_COMMENT.x0:_KEYBOARD_COMMENT.x1] = 222
    canvas[_KEYBOARD_COMMENT.y1 - 2:_KEYBOARD_COMMENT.y1,
           _KEYBOARD_COMMENT.x0:_KEYBOARD_COMMENT.x1] = 222
    canvas[_KEYBOARD_SEND.y0:_KEYBOARD_SEND.y1, _KEYBOARD_SEND.x0:_KEYBOARD_SEND.x1] = 228
    for offset, y in enumerate(range(_KEYBOARD_COMMENT.y1 - 1, _KEYBOARD_SEND.y0 + 1)):
        half = round(28 - offset * 0.5)
        canvas[y, 502 - half:502 + half] = 60
    _paint_glyph(canvas, center=_KEYBOARD_CONFIRM_CENTER)
    return _png(canvas, color=False)


def test_text_selection_handle_bridging_the_controls_is_still_an_open_composer():
    """Editing the opener must never read as a composer that closed itself."""
    surface = locate_inline_composer(
        _keyboard_open_composer_with_text_selection_handle(), _TEMPLATE)

    assert surface.comment_rect == _KEYBOARD_COMMENT
    assert surface.send_rect == _KEYBOARD_SEND
    assert surface.confirm_point == _KEYBOARD_CONFIRM_CENTER


def test_text_selection_handle_does_not_move_any_composer_geometry():
    """The handle is a transient overlay, so tap targets must not shift under it."""
    edited = locate_inline_composer(
        _keyboard_open_composer_with_text_selection_handle(), _TEMPLATE)
    untouched = locate_inline_composer(_keyboard_open_composer(), _TEMPLATE)

    assert edited == untouched


@pytest.mark.parametrize("color", [False, True], ids=["grayscale", "color"])
def test_locates_measured_inline_composer_geometry_and_glyph_center(color):
    surface = locate_inline_composer(_inline_composer(color=color), _TEMPLATE)

    assert surface.layout_id == "hinge_inline_v1"
    assert surface.comment_rect == _COMMENT
    assert surface.send_rect == _SEND
    assert surface.comment_rect.center == (540, 1686)
    assert surface.send_rect.center == (687, 1861)
    assert surface.confirm_point == _CONFIRM_CENTER
    assert surface.send_rect.contains(*surface.confirm_point)


def test_rejects_send_like_glyph_without_its_composer_surfaces():
    frame = np.full((_H, _W), _BG, dtype=np.uint8)
    _paint_glyph(frame)

    with pytest.raises(ComposerDetectionError, match="CTA is not visibly filled"):
        locate_inline_composer(_png(frame, color=False), _TEMPLATE)


def test_rejects_filled_cta_when_the_wide_comment_input_is_missing():
    with pytest.raises(ComposerDetectionError, match="comment input was not found"):
        locate_inline_composer(_inline_composer(input_present=False), _TEMPLATE)


def test_rejects_a_matching_glyph_outside_the_measured_cta_region():
    with pytest.raises(ComposerDetectionError, match="outside the inline composer's measured CTA"):
        locate_inline_composer(_inline_composer(glyph_center=(695, 1200)), _TEMPLATE)


def test_rejects_an_ordinary_wide_profile_card_even_at_the_old_preview_margin():
    """A card can have the old locator's width/height/margin yet is not a composer."""
    card = np.full((_H, _W), _BG, dtype=np.uint8)
    card[236:1400, 95:985] = 190

    with pytest.raises(ComposerDetectionError, match="no Send Like confirmation glyph"):
        locate_inline_composer(_png(card, color=True), _TEMPLATE)


def test_rejects_undecodable_frame_bytes():
    with pytest.raises(ComposerDetectionError, match="did not decode"):
        locate_inline_composer(b"not a PNG", _TEMPLATE)


@pytest.mark.parametrize("image", [
    np.empty((0, _W), dtype=np.uint8),
    np.empty((_H, 0), dtype=np.uint8),
    np.empty((_H, _W, 3), dtype=np.uint8),
])
def test_reused_image_must_be_a_nonempty_grayscale_buffer(image):
    with pytest.raises(ComposerDetectionError, match="non-empty grayscale"):
        locate_inline_composer(_inline_composer(), _TEMPLATE, image=image)


def test_boolean_composer_threshold_is_not_accepted_as_a_number():
    with pytest.raises(ValueError, match="finite correlation"):
        locate_inline_composer(_inline_composer(), _TEMPLATE, threshold=True)


def test_rejects_missing_confirmation_even_when_input_and_cta_geometry_are_present():
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    canvas[_COMMENT.y0:_COMMENT.y0 + 2, _COMMENT.x0:_COMMENT.x1] = 222
    canvas[_COMMENT.y1 - 2:_COMMENT.y1, _COMMENT.x0:_COMMENT.x1] = 222
    canvas[_SEND.y0:_SEND.y1, _SEND.x0:_SEND.x1] = 228

    with pytest.raises(ComposerDetectionError, match="no Send Like confirmation glyph"):
        locate_inline_composer(_png(canvas, color=False), _TEMPLATE)


def test_rejects_a_false_confirm_on_a_plain_card_without_input_or_filled_cta():
    card = np.full((_H, _W), _BG, dtype=np.uint8)
    card[236:1400, 95:985] = 190
    _paint_glyph(card)  # a copied phrase is not permission to use composer coordinates

    with pytest.raises(ComposerDetectionError, match="CTA is not visibly filled"):
        locate_inline_composer(_png(card, color=False), _TEMPLATE)


@pytest.mark.parametrize("which", ["input", "cta"], ids=["narrow_input", "narrow_cta"])
def test_rejects_narrow_surface_that_cannot_be_the_measured_composer(which):
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    if which == "input":
        canvas[_COMMENT.y0:_COMMENT.y0 + 2, 400:650] = 222
        canvas[_COMMENT.y1 - 2:_COMMENT.y1, 400:650] = 222
        canvas[_SEND.y0:_SEND.y1, _SEND.x0:_SEND.x1] = 228
        message = "comment input was not found"
    else:
        canvas[_COMMENT.y0:_COMMENT.y0 + 2, _COMMENT.x0:_COMMENT.x1] = 222
        canvas[_COMMENT.y1 - 2:_COMMENT.y1, _COMMENT.x0:_COMMENT.x1] = 222
        canvas[_SEND.y0:_SEND.y1, 700:850] = 228
        message = "CTA is not visibly filled"
    _paint_glyph(canvas)

    with pytest.raises(ComposerDetectionError, match=message):
        locate_inline_composer(_png(canvas, color=False), _TEMPLATE)


def test_rejects_input_and_cta_when_their_vertical_topology_is_reversed():
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    _paint_glyph(canvas)
    # Both shapes exist, but the input below the CTA is not the measured composer topology.
    canvas[1807:1809, 95:985] = 222
    canvas[1914:1916, 95:985] = 222
    canvas[1597:1775, 390:985] = 228

    with pytest.raises(ComposerDetectionError, match="CTA is not visibly filled"):
        locate_inline_composer(_png(canvas, color=True), _TEMPLATE)


def test_rejects_two_equally_plausible_send_like_glyphs_in_the_cta():
    """A detector cannot choose which copied confirmation text licenses input."""
    canvas = np.full((_H, _W), _BG, dtype=np.uint8)
    canvas[_COMMENT.y0:_COMMENT.y0 + 2, _COMMENT.x0:_COMMENT.x1] = 222
    canvas[_COMMENT.y1 - 2:_COMMENT.y1, _COMMENT.x0:_COMMENT.x1] = 222
    canvas[_SEND.y0:_SEND.y1, _SEND.x0:_SEND.x1] = 228
    _paint_glyph(canvas, center=(520, 1856))
    _paint_glyph(canvas, center=(860, 1856))

    with pytest.raises(ComposerDetectionError, match="ambiguous"):
        locate_inline_composer(_png(canvas, color=False), _TEMPLATE)


def test_resolution_relative_layout_scales_with_the_frame():
    width, height = 1440, 3200
    confirm = (round(_CONFIRM_CENTER[0] / _W * width), round(_CONFIRM_CENTER[1] / _H * height))
    surface = locate_inline_composer(
        _inline_composer(width=width, height=height, glyph_center=confirm), _TEMPLATE)

    assert surface.comment_rect == _scaled_rect(_COMMENT, width, height)
    assert surface.send_rect == _scaled_rect(_SEND, width, height)
    assert surface.confirm_point == confirm


@pytest.mark.parametrize("color", [False, True], ids=["keyboard_grayscale", "keyboard_color"])
def test_keyboard_open_current_phone_layout_returns_its_shifted_actual_geometry(color):
    surface = locate_inline_composer(_keyboard_open_composer(color=color), _TEMPLATE)

    assert surface.comment_rect == _KEYBOARD_COMMENT
    assert surface.send_rect == _KEYBOARD_SEND
    assert surface.confirm_point == _KEYBOARD_CONFIRM_CENTER
    assert surface.send_rect.contains(*surface.confirm_point)


def test_keyboard_open_composer_ignores_a_selected_photo_edge_above_its_input():
    """Only adjacent wide-run groups can be the input's top and bottom borders."""
    surface = locate_inline_composer(_keyboard_open_composer_with_selected_photo_edge(), _TEMPLATE)

    assert surface.comment_rect == _KEYBOARD_COMMENT
    assert surface.send_rect == _KEYBOARD_SEND


def test_fully_visible_priority_like_after_human_review_returns_its_shifted_geometry():
    """Scroll review must not leave callers with the pre-review Send coordinates."""
    surface = locate_inline_composer(_scrolled_priority_like_composer(), _TEMPLATE)

    assert surface.comment_rect == Rect(95, 1616, 985, 1794)
    assert surface.send_rect == Rect(390, 1826, 985, 1935)
    assert surface.confirm_point == (690, 1882)
    assert surface.send_rect.contains(*surface.confirm_point)
