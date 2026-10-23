"""Review findings on whole-photo extraction correctness and train/serve parity.

Every frame is SYNTHESISED with the layout helpers of tests/test_photo_cards.py (real
screencaps are strangers' dating profiles and never enter the repo). Each test pins one defect
found in review; it fails against the code it was written against.
"""

from __future__ import annotations

import numpy as np

from operation_love.vision import photo_cards as pc
from tests import test_photo_cards as layout


def _studio_portrait(rng):
    """A photo slot holding a portrait against a white studio wall: card-white from edge to
    edge, the person in the middle ~49% of the columns and ~77% of the rows. White share
    ~0.62, so it sits in the 0.5..0.8 band where build2.py asked CLIP's zero-shot prompts
    ("a photo of a person" vs "a text card ...") and the port asks the edge-media test."""
    person = layout._photo(rng, heart=None)
    pixels = np.full_like(person, layout._SURFACE)
    pixels[150:900, 250:724] = person[150:900, 250:724]
    glyph = layout.hinge._load_template(layout._PRE_10_8)
    th, tw = glyph.shape
    cy, cx = layout._PHOTO - 70, 884  # the like heart, where the real one sits
    pixels[cy - th // 2 : cy - th // 2 + th, cx - tw // 2 : cx - tw // 2 + tw] = glyph[
        :, :, None
    ]
    return pixels


def test_a_white_backdrop_portrait_is_kept_as_a_photo_card():
    """A complete 974-row photo card is dropped as 'text' because a white backdrop reaches the
    card edge: the edge-media probe cannot tell a white studio wall from a prompt card's white
    padding. The experiment's CLIP zero-shot rule would have scored it as a photo of a person.
    The whole-photo vector then silently pools five of the profile's six photos."""
    rng = np.random.default_rng(5)
    studio = _studio_portrait(rng)
    assert 0.5 <= layout._white_share(studio) < 0.8  # the premise: the undecided band
    cards = layout._layout(
        [
            (layout._photo(rng), None),
            (studio, None),
            (layout._photo(rng), None),
            (layout._text_card(rng), None),
        ],
        start=500,
    )
    frames = [layout._render(cards, o, k) for k, o in enumerate((0, 700, 1400, 2100))]
    result = pc.extract_photo_cards(frames)
    assert result.components == 1  # alignment is not the problem
    assert [c.page_y0 for c in result.cards] == [cards[0].y0, cards[1].y0, cards[2].y0]
    got = result.cards[1]
    assert got.complete and got.height == layout._PHOTO
    # Kept by the contiguous-content cue: edge media alone still calls it text.
    assert got.media_row_share < pc._MIN_MEDIA_ROWS
    assert got.content_run_share >= pc._CONTENT_RUN_MIN
    assert result.band_cards == (pc.BAND_CONTIGUOUS,)
    # Compared outside the painted like button (WP-PARITY-2), which is CLIP's mean colour.
    layout._assert_pixels(got, studio)


def test_card_pixels_do_not_depend_on_which_hinge_build_drew_the_like_button():
    """The same six photos captured on either side of Hinge 10.8's like-glyph change must give
    the same card pixels: the like button is app chrome overlaid on the photo, and the contract
    says cards carry "the photo media only (caption/chrome excluded)". Today the button is
    kept, so the whole-photo vector encodes the app build (the offline verify measured
    "predict Hinge-10.8 template from W: AUC 0.993"); serving is all 10.8+ while ~70% of the
    labels the cutoff is calibrated on are pre-10.8, and the next glyph change moves every
    live vector with no version bump."""
    _, frames_pre = layout._profile(heart=layout._PRE_10_8)
    _, frames_new = layout._profile(heart=layout._GLYPH_10_8)
    pre = pc.extract_photo_cards(frames_pre)
    new = pc.extract_photo_cards(frames_new)
    assert (pre.template, new.template) == ("pre_10_8", "10_8")
    assert [c.page_y0 for c in pre.cards] == [c.page_y0 for c in new.cards]
    for a, b in zip(pre.cards, new.cards, strict=True):
        assert (
            a.button_boxes == b.button_boxes
        )  # the mask never depends on the template
        differing = np.argwhere((a.pixels != b.pixels).any(-1))
        assert differing.size == 0, (
            f"card {a.order}: {len(differing)} pixels differ, all inside the like button "
            f"(rows {differing[:, 0].min()}..{differing[:, 0].max()}, "
            f"cols {differing[:, 1].min()}..{differing[:, 1].max()})"
        )
