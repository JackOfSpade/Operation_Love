"""Synthetic bitmap coverage for doc 5.8's deliberately conservative coarse check."""
import cv2
import numpy as np

from operation_love.drivers.item_type_preflight import (
    INCONCLUSIVE,
    MATCH,
    MISMATCH,
    PHOTO,
    UNKNOWN,
    WRITTEN,
    classify_description,
    crop_type_evidence,
    preflight_item_type,
)


def _png(array) -> bytes:
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return bytes(encoded)


def _written_card() -> bytes:
    card = np.full((180, 260, 3), 238, dtype=np.uint8)
    cv2.putText(card, "A PROMPT", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2)
    cv2.putText(card, "with an answer", (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1)
    return _png(card)


def _photo_like_crop() -> bytes:
    # A deterministic high-variation, high-edge bitmap -- it is intentionally not a real person.
    rng = np.random.default_rng(7)
    return _png(rng.integers(0, 256, (180, 260, 3), dtype=np.uint8))


def _mixed_prompt_card() -> bytes:
    """Written text in a large calm panel over a deliberately busy synthetic background."""
    rng = np.random.default_rng(17)
    card = rng.integers(0, 256, (180, 260, 3), dtype=np.uint8)
    cv2.rectangle(card, (10, 30), (250, 150), (245, 245, 245), -1)
    cv2.putText(card, "A PROMPT", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return _png(card)


def test_description_only_uses_unambiguous_canonical_type_words():
    assert classify_description("a mountain photo") == PHOTO
    assert classify_description("the written prompt answer") == WRITTEN
    assert classify_description("a photo of her prompt") == UNKNOWN
    assert classify_description("a great card") == UNKNOWN


def test_confident_cross_class_synthetic_bitmaps_are_mismatches():
    photo_vs_written = preflight_item_type("the prompt answer", _photo_like_crop())
    written_vs_photo = preflight_item_type("a photo of her hiking", _written_card())

    assert photo_vs_written.state == MISMATCH
    assert photo_vs_written.description_type == WRITTEN
    assert photo_vs_written.crop_type == PHOTO
    assert written_vs_photo.state == MISMATCH
    assert written_vs_photo.description_type == PHOTO
    assert written_vs_photo.crop_type == WRITTEN


def test_same_type_and_ambiguous_evidence_do_not_stop():
    assert preflight_item_type("a photo", _photo_like_crop()).state == MATCH
    assert preflight_item_type("a prompt", _written_card()).state == MATCH
    # A uniform bitmap is deliberately not called either class; ambiguity passes.
    flat = _png(np.full((180, 260, 3), 140, dtype=np.uint8))
    result = preflight_item_type("a photo", flat)
    assert result.state == INCONCLUSIVE
    assert result.crop_type == UNKNOWN


def test_written_panel_over_a_busy_background_is_inconclusive_not_a_photo_mismatch():
    """Mixed composition is not enough evidence to stop AUTO on a claimed written prompt."""
    result = preflight_item_type("a written prompt", _mixed_prompt_card())

    assert result.state == INCONCLUSIVE
    assert result.crop_type == UNKNOWN


def test_rectangular_captioned_low_detail_photo_is_affirmatively_photo():
    """A title band, calm sky and non-square geometry must not turn a real photo into UNKNOWN."""
    height, width = 300, 260
    card = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(card, "TAKE ME BACK", (15, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 2, cv2.LINE_AA)
    y, x = np.mgrid[0:height - 50, 0:width]
    card[50:, :, 0] = np.clip(160 + y * 0.15 + 20 * np.sin(x / 10), 0, 255)
    card[50:, :, 1] = np.clip(120 + y * 0.25 + 18 * np.sin(x / 14), 0, 255)
    card[50:, :, 2] = np.clip(80 + y * 0.45 + 15 * np.sin(x / 18), 0, 255)
    cv2.circle(card, (190, 195), 35, (70, 55, 40), -1)

    evidence = crop_type_evidence(_png(card))

    assert evidence["large_uniform_panel"] is True
    assert evidence["dominant_background"] < 0.45
    assert evidence["classification"] == PHOTO


def test_high_contrast_hinge_prompt_with_heart_control_is_still_written():
    """Large serif-like text and the black heart broke the old global-std-only prompt gate.

    The card deliberately exceeds that old 52-level bound.  The v2 verdict comes from the
    stronger dominant white background plus affirmative multi-row glyph layout, not from making
    every low-variation rectangle a prompt.
    """
    card = np.full((220, 320, 3), 246, dtype=np.uint8)
    cv2.putText(card, "UNUSUAL SKILLS", (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(card, "I BUY STOCKS", (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(card, "HIGH SELL LOW", (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.circle(card, (290, 187), 24, (12, 12, 12), -1)

    evidence = crop_type_evidence(_png(card))

    assert evidence["colour_std"] > 52
    assert evidence["dominant_background"] > 0.84
    assert evidence["text_layout"] is True
    assert evidence["classification"] == WRITTEN


def test_quiet_wall_photo_shape_without_text_rows_remains_unknown():
    """The prompt recall fix must not turn a person against a calm wall into written context."""
    card = np.full((240, 180, 3), (225, 220, 210), dtype=np.uint8)
    cv2.ellipse(card, (90, 112), (30, 72), 0, 0, 360, (85, 65, 50), -1)
    cv2.circle(card, (90, 40), 22, (105, 80, 60), -1)

    evidence = crop_type_evidence(_png(card))

    assert evidence["text_layout"] is False
    assert evidence["classification"] == UNKNOWN
