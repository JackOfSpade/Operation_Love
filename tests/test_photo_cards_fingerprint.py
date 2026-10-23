"""Pinned fingerprint of the whole-photo extraction (`whole_photo_v1`).

Every stored `photo_features` row, live or backfilled, was cut by `extract_photo_cards`. If
`drivers/segment.py`, a like template, the like-match threshold or this module's own code
changes what the extractor returns, every new live row moves away from the backfilled ones
under the same feature_version, and no other test notices. These tests pin it.

When one fails because you changed extraction on purpose: bump
`WHOLE_PHOTO_FEATURE_VERSION` in operation_love/vision/photo_cards.py, update the pins below,
re-run `python -m tools.backfill_photo_features` (every row of the old version is then
missing) and re-run `python -m tools.compare_feature_sets`.

The fingerprint only catches changes that affect these synthetic fixtures (tests/test_photo_cards.py
layouts: both like glyphs, the pass-X overlay with a static and a video card, a white-band
studio portrait beside a dense prompt card, and a chain break); a change that leaves all of them
byte-identical can still move real profiles.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from operation_love.drivers import hinge
from operation_love.vision import photo_cards as pc
from tests import test_photo_cards as layout

_BUMP = (
    "whole-photo extraction changed: bump WHOLE_PHOTO_FEATURE_VERSION, update this pin and "
    "re-run python -m tools.backfill_photo_features (see this module's docstring)"
)

_TEMPLATE_SHA256 = {
    "hinge_like_button.png": "23e89ebc6e8bc525d71b73111d7e55a42d2c905234623343637b87d0c9ce5957",
    "hinge_like_button_10_8.png": "4a3a351653fb6f35983f6a477dfa5291ecd9acb86db854fd23c17dff96ddf8a5",
}
_EXTRACTION_SHA256 = "55121089c982cd264787df81d7bd12360825749101dfce08a1078df5b2475593"


def _studio_band_profile():
    """A photo, a portrait against a white wall (white share ~0.62, kept by the contiguous
    content cue), another photo and the dense prompt card (white ~0.59, neither cue)."""
    rng = np.random.default_rng(5)
    person = layout._photo(rng, heart=None)
    studio = np.full_like(person, layout._SURFACE)
    studio[150:900, 250:724] = person[150:900, 250:724]
    glyph = hinge._load_template(layout._PRE_10_8)
    th, tw = glyph.shape
    cy, cx = layout._PHOTO - 70, 884
    studio[cy - th // 2 : cy - th // 2 + th, cx - tw // 2 : cx - tw // 2 + tw] = glyph[
        :, :, None
    ]
    cards = layout._layout(
        [
            (layout._photo(rng), None),
            (studio, None),
            (layout._photo(rng), None),
            (layout._text_card(rng, line=30, every=60), None),
        ],
        start=500,
    )
    return [layout._render(cards, o, k) for k, o in enumerate((0, 700, 1400, 2100))]


def _digest(results) -> str:
    h = hashlib.sha256()
    for r in results:
        h.update(
            repr((r.template, r.components, r.shifts, r.merges, r.band_cards)).encode()
        )
        for c in r.cards:
            h.update(c.pixels.tobytes())
            h.update(
                repr(
                    (
                        c.order,
                        c.height,
                        c.page_y0,
                        c.stitched,
                        c.video_like,
                        c.x_overlay_rows,
                        c.button_boxes,
                    )
                ).encode()
            )
    return h.hexdigest()


def _extractions(monkeypatch):
    _, pre = layout._profile(heart=layout._PRE_10_8)
    _, new = layout._profile(heart=layout._GLYPH_10_8)
    results = [
        pc.extract_photo_cards(pre),
        pc.extract_photo_cards(new),
        pc.extract_photo_cards(layout._overlay(False)[1]),
        pc.extract_photo_cards(layout._overlay(True)[1]),
        pc.extract_photo_cards(_studio_band_profile()),
    ]
    # A chain break: the frame 1 -> 2 alignment refused, so one card is seen in two components
    # and merged by its exact pixels.
    real_align = pc._align
    calls = []

    def align(a, b, *rest):
        calls.append(None)
        return None if len(calls) == 2 else real_align(a, b, *rest)

    monkeypatch.setattr(pc, "_align", align)
    results.append(pc.extract_photo_cards(pre))
    return results


def test_the_like_templates_are_the_pinned_files():
    for filename, expected in _TEMPLATE_SHA256.items():
        actual = hashlib.sha256((hinge._ASSETS / filename).read_bytes()).hexdigest()
        assert actual == expected, f"{filename}: {_BUMP}"
    assert {f: d for _n, f, d in pc._LIKE_TEMPLATES} == _TEMPLATE_SHA256, _BUMP


def test_the_like_match_threshold_is_pinned():
    # photo_cards pins its own copy, so the driver's _LIKE_MATCH_THRESHOLD may be recalibrated
    # without moving the stored features; this copy must not move without a version bump.
    assert pc._LIKE_MATCH_THRESHOLD == 0.75, _BUMP


def test_the_extraction_of_the_fixtures_is_pinned(monkeypatch):
    results = _extractions(monkeypatch)
    # the fixtures still reach the paths they are here for
    assert [r.template for r in results[:2]] == ["pre_10_8", "10_8"]
    assert any(c.video_like for c in results[3].cards)
    assert pc.BAND_CONTIGUOUS in results[4].band_cards
    assert results[5].components == 2 and results[5].merges == ("pixel",)
    assert all(c.button_boxes for r in results for c in r.cards)
    assert _digest(results) == _EXTRACTION_SHA256, _BUMP


@pytest.mark.parametrize("glyph", [layout._PRE_10_8, layout._GLYPH_10_8])
def test_the_fixtures_decode_to_the_same_pixels_whichever_encoder_wrote_them(glyph):
    """The digest covers decoded pixels and geometry, never frame bytes: a different PNG
    encoder (another cv2 build in CI) must not fail the pin."""
    _, frames = layout._profile(heart=glyph)
    import cv2

    again = []
    for frame in frames:
        rgb = layout._rgb(frame)
        ok, buf = cv2.imencode(
            ".png",
            np.ascontiguousarray(rgb[:, :, ::-1]),
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        assert ok
        again.append(buf.tobytes())
    assert again != frames  # really different bytes
    assert _digest([pc.extract_photo_cards(again)]) == _digest(
        [pc.extract_photo_cards(frames)]
    )
