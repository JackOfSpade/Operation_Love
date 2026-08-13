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
