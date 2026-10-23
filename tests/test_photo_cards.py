"""Whole-photo card extraction (operation_love/vision/photo_cards.py).

Every frame here is SYNTHESISED. Real screencaps are strangers' dating profiles and never enter
the repo, so the fixtures rebuild Hinge's measured layout from first principles, as
tests/test_segment.py does: a gradient page background, rounded-rect cards inset 53px with
53px gutters, the shipped like glyph stamped on photo cards, and Hinge's floating pass-X painted
at its fixed screen position on every frame. A profile is a tall PAGE of cards; each frame is
the content band's view of that page at one scroll offset, so ground truth (which card, which
pixels, which frame showed what) is known by construction and the assertions are exact.
"""

from __future__ import annotations

import hashlib
import io
import math

import cv2
import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import hinge, segment
from operation_love.vision import photo_cards as pc

_W, _H = 1080, 2400
_BAND0, _BAND1 = 300, 2100  # HINGE_SPEC.content_band on a 2400-row frame
_X0, _X1 = segment._CARD_MARGIN_PX, _W - segment._CARD_MARGIN_PX  # 53, 1027
_CW = _X1 - _X0  # 974
_GUTTER = max(segment._GUTTER_PX)  # 53
_RADIUS = 22
_PHOTO = 974
# Card white (every channel >= 232) yet at least 9 grey levels off the page gradient, so the
# row-projection segmenter sees a card surface where a real card's 255 would sometimes read as
# page background near the top of the screen (segment.py's amendment-one hazard).
_SURFACE = 234
_PAGE_TOP, _PAGE_BOTTOM = 254, 243
_PRE_10_8 = "hinge_like_button.png"
_GLYPH_10_8 = "hinge_like_button_10_8.png"
_X_CENTRE, _X_RADIUS = (120, 2030), 75  # the floating pass-X, screen-fixed
_INTERIOR = slice(_RADIUS, _CW - _RADIUS)  # card columns clear of the corner arcs

_BG = np.linspace(_PAGE_TOP, _PAGE_BOTTOM, _H).round().astype(np.uint8)
_INSET = [
    int(math.ceil(_RADIUS - math.sqrt(max(0.0, _RADIUS**2 - (_RADIUS - i) ** 2))))
    for i in range(_RADIUS)
]


class _Card:
    """One card on the synthetic page. A video card draws fresh pixels on every frame."""

    def __init__(self, y0, pixels, *, video_seed=None):
        self.y0, self.pixels, self.video_seed = y0, pixels, video_seed

    @property
    def y1(self):
        return self.y0 + len(self.pixels)

    def rows(self, frame_index):
        if self.video_seed is None:
            return self.pixels
        rng = np.random.default_rng((self.video_seed, frame_index))
        return rng.integers(30, 220, size=self.pixels.shape, dtype=np.uint8)


def _photo(rng, *, heart=_PRE_10_8, height=_PHOTO):
    pixels = rng.integers(30, 220, size=(height, _CW, 3), dtype=np.uint8)
    if heart is not None:
        glyph = hinge._load_template(heart)
        th, tw = glyph.shape
        cy, cx = height - 70, 884  # frame x 937, the measured bottom-right heart column
        pixels[cy - th // 2 : cy - th // 2 + th, cx - tw // 2 : cx - tw // 2 + tw] = (
            glyph[:, :, None]
        )
    return pixels


def _bright_photo(rng, *, heart=_PRE_10_8):
    """A photo whose middle is one bright card-white field (~65% white): edge to edge media."""
    pixels = _photo(rng, heart=heart)
    pixels[170:800] = _SURFACE
    return pixels


def _text_card(rng, *, height=600, line=14, every=100):
    """A prompt/vitals-like card: white surface, dark text lines inside 48px white padding."""
    pixels = np.full((height, _CW, 3), _SURFACE, np.uint8)
    for y in range(40, height - 40 - line, every):
        pixels[y : y + line, 48 : _CW - 48] = rng.integers(
            20, 120, size=(line, _CW - 96, 3), dtype=np.uint8
        )
    return pixels


def _layout(parts, start=400):
    cards, y = [], start
    for pixels, video_seed in parts:
        cards.append(_Card(y, pixels, video_seed=video_seed))
        y += len(pixels) + _GUTTER
    return cards


def _render(cards, offset, frame_index):
    """The PNG of the frame at scroll `offset`: frame row y shows page row y + offset."""
    img = np.repeat(np.repeat(_BG[:, None], _W, 1)[:, :, None], 3, 2).copy()
    img[:280] = 200  # header chrome above the content band
    img[2120:] = 40  # bottom nav below it
    for card in cards:
        rows = card.rows(frame_index)
        lo, hi = max(_BAND0, card.y0 - offset), min(_BAND1, card.y1 - offset)
        if hi <= lo:
            continue
        img[lo:hi, _X0:_X1] = rows[lo + offset - card.y0 : hi + offset - card.y0]
        for i, inset in enumerate(_INSET):
            for y in (card.y0 + i - offset, card.y1 - 1 - i - offset):
                if lo <= y < hi and inset > 0:
                    img[y, _X0 : _X0 + inset] = _BG[y]
                    img[y, _X1 - inset : _X1] = _BG[y]
    yy, xx = np.ogrid[:_H, :_W]
    dy, dx = yy - _X_CENTRE[1], xx - _X_CENTRE[0]
    disk = dy**2 + dx**2 <= _X_RADIUS**2
    img[disk] = 255
    img[disk & ((np.abs(dy - dx) <= 4) | (np.abs(dy + dx) <= 4))] = 60
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[:, :, ::-1]))
    assert ok
    return buf.tobytes()


def _rgb(frame):
    with Image.open(io.BytesIO(frame)) as image:
        return np.array(image.convert("RGB"))


def _white_share(pixels):
    return float((pixels.min(-1) >= 232).mean())


def _button_mask(card):
    """The card's painted like-button pixels (bool), from its reported boxes."""
    mask = np.zeros(card.pixels.shape[:2], bool)
    for a0, a1, b0, b1 in card.button_boxes:
        mask[a0:a1, b0:b1] = True
    return mask


def _expected_bottom_boxes(height):
    """The fixed like-button box of a card whose media bottom edge is its last row."""
    rows_above, rows_below = pc._BUTTON_ROWS_ABOVE_BOTTOM
    return ((height - rows_above, height - rows_below, *pc._BUTTON_COLS),)


def _assert_pixels(got, want):
    """`got`'s pixels equal `want` on the card columns clear of the corner arcs, except on the
    painted like button, which is exactly CLIP's mean colour."""
    mask = _button_mask(got)
    assert (got.pixels[mask] == pc.CLIP_MEAN_RGB).all()
    same = got.pixels[:, _INTERIOR] == want[:, _INTERIOR]
    assert same.all(-1)[~mask[:, _INTERIOR]].all()


# Scenario PROFILE: photo, prompt card, photo (never wholly on any one frame), dense prompt
# card, bright photo, photo, prompt card. Six frames, 900..1000 rows of scroll apart.
_PROFILE_OFFSETS = (0, 900, 1800, 2700, 3700, 4300)


def _profile(heart=_PRE_10_8):
    rng = np.random.default_rng(20261023)
    cards = _layout(
        [
            (_photo(rng, heart=heart), None),  # P0 page 400..1374
            (_text_card(rng), None),  # T0 1427..2027, white ~0.87
            (_photo(rng, heart=heart), None),  # P1 2080..3054, split across frames
            (_text_card(rng, line=30, every=60), None),  # T1 3107..3707, white ~0.59
            (_bright_photo(rng, heart=heart), None),  # P2 3760..4734, white ~0.65
            (_photo(rng, heart=heart), None),  # P3 4787..5761
            (_text_card(rng, height=500), None),  # T2 5814..6314
        ]
    )
    frames = [_render(cards, o, k) for k, o in enumerate(_PROFILE_OFFSETS)]
    return cards, frames


@pytest.fixture(scope="module")
def profile():
    cards, frames = _profile()
    return cards, frames, pc.extract_photo_cards(frames)


# Scenario OVERLAY: photo A, photo-or-video B, photo C, prompt card. On frame 1, B is wholly
# visible but its bottom 61 rows sit under the floating pass-X; frame 2 shows those rows clear
# while missing B's top; frame 0 shows B's top 573 rows.
_OVERLAY_OFFSETS = (0, 500, 1300, 2200)


def _overlay(video):
    rng = np.random.default_rng(7)
    cards = _layout(
        [
            (_photo(rng, heart=None), None),  # A 500..1474
            (_photo(rng, heart=None), 99 if video else None),  # B 1527..2501
            (_photo(rng, heart=None), None),  # C 2554..3528
            (_text_card(rng), None),
        ],
        start=500,
    )
    frames = [_render(cards, o, k) for k, o in enumerate(_OVERLAY_OFFSETS)]
    return cards, frames


# =============================================================================================
# Constants and the digest
# =============================================================================================


def test_feature_constants():
    assert pc.WHOLE_PHOTO_FEATURE_VERSION == "whole_photo_v1"
    assert pc.WHOLE_PHOTO_DIMS == 768


def test_pinned_content_band_matches_the_hinge_spec_it_was_measured_on():
    assert pc._CONTENT_BAND == hinge.HINGE_SPEC.content_band


def test_frames_sha256_is_the_sha_of_the_concatenated_per_frame_hex_digests():
    frames = [b"first frame", b"second", b""]
    expected = hashlib.sha256(
        (
            hashlib.sha256(b"first frame").hexdigest()
            + hashlib.sha256(b"second").hexdigest()
            + hashlib.sha256(b"").hexdigest()
        ).encode("ascii")
    ).hexdigest()
    assert pc.frames_sha256(frames) == expected
    assert pc.frames_sha256(list(reversed(frames))) != expected  # order is part of it
    assert pc.frames_sha256([]) == hashlib.sha256(b"").hexdigest()


def test_result_carries_the_frames_digest(profile):
    _, frames, result = profile
    assert result.frames_sha256 == pc.frames_sha256(frames)
    assert result.frame_count == len(frames)


def test_no_frames_is_zero_cards_not_an_error():
    result = pc.extract_photo_cards([])
    assert result.cards == () and len(result) == 0
    assert result.frames_sha256 == hashlib.sha256(b"").hexdigest()


# =============================================================================================
# Cards: order, dedup, stitching, classification
# =============================================================================================


def test_unique_photo_cards_in_page_order(profile):
    cards, _, result = profile
    photos = [cards[i] for i in (0, 2, 4, 5)]  # P0, P1, P2 (bright), P3
    assert len(result) == 4
    assert [c.order for c in result.cards] == [0, 1, 2, 3]
    assert [c.page_y0 for c in result.cards] == [p.y0 for p in photos]
    assert result.components == 1 and result.merges == ()
    for got, want in zip(result.cards, photos, strict=True):
        assert got.height == _PHOTO and got.complete and got.completeness == 1.0
        assert got.pixels.shape == (_PHOTO, _CW, 3) and got.pixels.dtype == np.uint8
        assert got.button_boxes == _expected_bottom_boxes(_PHOTO)
        _assert_pixels(got, want.pixels)
        assert got.x_overlay_rows == 0 and not got.video_like


def test_a_card_seen_in_two_overlapping_frames_is_kept_once(profile):
    cards, _, result = profile
    p0 = result.cards[0]
    # P0 is whole on frame 0 and its bottom 174 rows are on frame 1: two instances, one card
    assert p0.instances == 2 and p0.first_frame == 0
    assert p0.stitched is False
    assert sum(1 for c in result.cards if c.page_y0 == cards[0].y0) == 1


def test_a_card_no_frame_shows_whole_is_stitched_pixel_exact(profile):
    cards, frames, result = profile
    p1_card = cards[2]
    for offset in _PROFILE_OFFSETS:  # the premise: never wholly inside one content band
        assert not (p1_card.y0 - offset >= _BAND0 and p1_card.y1 - offset <= _BAND1)
    p1 = result.cards[1]
    assert p1.stitched and p1.complete and p1.height == _PHOTO
    _assert_pixels(p1, p1_card.pixels)


def test_prompt_and_vitals_cards_are_excluded(profile):
    cards, _, result = profile
    kept = {c.page_y0 for c in result.cards}
    for text in (cards[1], cards[3], cards[6]):
        assert text.y0 not in kept
    # the dense prompt card is in the band the white share alone cannot decide, and its white
    # side padding is what excludes it
    dense = cards[3].pixels
    assert 0.5 <= _white_share(dense) < 0.8
    assert pc._media_row_share(dense, np.zeros(len(dense), bool), np) == 0.0
    # ... and its ink lines are separated by blank rows, so no long content run either
    assert pc._content_run_share(dense, np) < pc._CONTENT_RUN_MIN


def test_band_cards_record_how_each_cue_decided(profile):
    """Every reconstructed card in the 0.5..0.8 white band, kept or not, in page order: the
    dense prompt card T1 (neither cue) and the bright photo P2 (edge media only)."""
    _, _, result = profile
    assert result.band_cards == (pc.BAND_TEXT, pc.BAND_EDGE_MEDIA)


# =============================================================================================
# The like button
# =============================================================================================


def test_the_fixed_button_box_covers_both_builds_measured_disc():
    """Measured on real frames: heart centre at card column 885 / 886 (frame x 938 / 939) and
    89 / 90 rows above the card's bottom edge, disc plus halo radius ~72."""
    rows_above, rows_below = pc._BUTTON_ROWS_ABOVE_BOTTOM
    c0, c1 = pc._BUTTON_COLS
    for cx, inset in ((885, 89), (886, 90)):
        cy = _PHOTO - inset
        assert _PHOTO - rows_above <= cy - 72 and cy + 72 < _PHOTO - rows_below
        assert c0 <= cx - 72 and cx + 72 < c1
    # and the synthetic glyph these fixtures stamp (88x88 at (height - 70, 884))
    assert (
        _PHOTO - rows_above <= _PHOTO - 70 - 44
        and _PHOTO - 70 + 44 <= _PHOTO - rows_below
    )
    assert c0 <= 884 - 44 and 884 + 44 <= c1


def test_without_a_known_bottom_edge_each_heart_gets_a_centred_square():
    half = pc._BUTTON_HALF
    assert pc._button_boxes(600, _CW, 1000, None, [(885, 300)]) == (
        (300 - half, 300 + half, 885 - half, 885 + half),
    )
    # clipped to the canvas; a heart wholly outside it paints nothing
    assert pc._button_boxes(600, _CW, 1000, None, [(885, 590)]) == (
        (590 - half, 600, 885 - half, 885 + half),
    )
    assert pc._button_boxes(600, _CW, 1000, None, [(885, 700)]) == ()
    # a known bottom edge places the fixed box and ignores where any template matched
    assert pc._button_boxes(974, _CW, 0, 974, [(886, 884)]) == _expected_bottom_boxes(
        974
    )
    assert pc._button_boxes(974, _CW, 0, 974, []) == _expected_bottom_boxes(974)


def test_the_painted_button_is_excluded_from_every_cue():
    """A prompt card's white surface with the button painted at its bottom-right: counted, the
    fill would read as edge media on ~30% of the rows and turn the card into a photo."""
    canvas = np.full((600, _CW, 3), _SURFACE, np.uint8)
    boxes = pc._button_boxes(600, _CW, 0, 600, [])
    keep = np.ones(canvas.shape[:2], bool)
    for a0, a1, b0, b1 in boxes:
        keep[a0:a1, b0:b1] = False
    canvas[~keep] = pc.CLIP_MEAN_RGB
    from_x = np.zeros(len(canvas), bool)
    assert pc._media_row_share(canvas, from_x, np) >= pc._MIN_MEDIA_ROWS  # the hazard
    assert pc._media_row_share(canvas, from_x, np, keep) == 0.0
    assert pc._content_run_share(canvas, np, keep) == 0.0
    a = _noise(11)
    assert pc._colour_similarity(canvas, a, np, keep, None) == pytest.approx(
        pc._colour_similarity(canvas[keep][None], a, np)
    )


def test_a_template_that_is_not_the_pinned_file_raises(profile, monkeypatch):
    _, frames, _ = profile
    name, filename, _digest = pc._LIKE_TEMPLATES[1]
    monkeypatch.setattr(
        pc,
        "_LIKE_TEMPLATES",
        (pc._LIKE_TEMPLATES[0], (name, filename, "0" * 64)),
    )
    with pytest.raises(pc.PhotoCardsError, match="WHOLE_PHOTO_FEATURE_VERSION"):
        pc.extract_photo_cards(frames[:1])


def test_no_photo_cards_is_a_photo_cards_error():
    assert issubclass(pc.NoPhotoCards, pc.PhotoCardsError)


def test_a_bright_photo_in_the_undecided_band_is_kept_by_its_edge_to_edge_media(
    profile,
):
    _, _, result = profile
    bright = result.cards[2]
    assert 0.5 <= bright.white_share < 0.8
    assert bright.media_row_share >= pc._MIN_MEDIA_ROWS


def test_extraction_is_deterministic(profile):
    _, frames, result = profile
    again = pc.extract_photo_cards(tuple(frames))
    assert [c.pixels.tobytes() for c in again.cards] == [
        c.pixels.tobytes() for c in result.cards
    ]
    strip = [{k: v for k, v in vars(c).items() if k != "pixels"} for c in again.cards]
    assert strip == [
        {k: v for k, v in vars(c).items() if k != "pixels"} for c in result.cards
    ]
    assert (again.shifts, again.template, again.frames_sha256) == (
        result.shifts,
        result.template,
        result.frames_sha256,
    )


def test_card_pixels_are_read_only(profile):
    _, _, result = profile
    with pytest.raises(ValueError):
        result.cards[0].pixels[0, 0, 0] = 1


def test_like_glyph_is_chosen_from_the_frames_not_told(profile):
    """Frames from either side of Hinge 10.8's heart change give the same cards, and the
    extractor names the glyph it found without being told the build."""
    _, _, pre = profile
    _, frames_10_8 = _profile(heart=_GLYPH_10_8)
    new = pc.extract_photo_cards(frames_10_8)
    assert (pre.template, new.template) == ("pre_10_8", "10_8")
    assert [c.page_y0 for c in new.cards] == [c.page_y0 for c in pre.cards]
    assert [c.page_y0 for c in new.cards] == [400, 2080, 3760, 4787]


# =============================================================================================
# The floating pass-X and video cards
# =============================================================================================


def test_rows_under_the_floating_pass_x_are_taken_from_a_frame_that_shows_them_clear():
    cards, frames = _overlay(video=False)
    result = pc.extract_photo_cards(frames)
    assert result.components == 1
    b = next(c for c in result.cards if c.page_y0 == cards[1].y0)
    # frame 1 shows B whole but with its bottom rows under the X; frame 2 shows them clear
    assert b.stitched and b.x_overlay_rows == 0
    _assert_pixels(b, cards[1].pixels)


def test_without_a_clear_view_the_overlay_rows_are_counted_not_hidden():
    cards, frames = _overlay(video=False)
    result = pc.extract_photo_cards(frames[:2])  # no frame shows B's bottom rows clear
    b = next(c for c in result.cards if c.page_y0 == cards[1].y0)
    assert b.x_overlay_rows == 2501 - (_OVERLAY_OFFSETS[1] + pc._X_Y0)  # 61
    assert not b.stitched


def test_a_video_card_is_never_stitched():
    """Same geometry as the overlay test, but B's pixels change on every frame. Stitching would
    splice two different video frames into one fabricated image, so B is exactly its single
    most complete instance (frame 1), X rows and all."""
    cards, frames = _overlay(video=True)
    result = pc.extract_photo_cards(frames)
    assert result.components == 1  # the video rescue aligned the frames across it
    assert len(result) == 3
    b = next(c for c in result.cards if c.page_y0 == cards[1].y0)
    assert b.video_like and not b.stitched
    assert b.x_overlay_rows == 61 and b.instances == 3
    frame1 = _rgb(frames[1])
    y0 = cards[1].y0 - _OVERLAY_OFFSETS[1]
    expected = frame1[y0 : y0 + _PHOTO, _X0:_X1]
    assert b.button_boxes == _expected_bottom_boxes(
        _PHOTO
    )  # a video card is masked too
    _assert_pixels(b, expected)
    # and the static neighbours are unaffected
    assert not any(c.video_like for c in result.cards if c is not b)


# =============================================================================================
# Cross-component merges
# =============================================================================================


def test_a_chain_break_keeps_one_card_seen_on_both_sides_by_its_exact_pixels(
    profile, monkeypatch
):
    """Refuse the frame 1 -> 2 alignment: P1 is then seen partially in two components with an
    unknown relative offset. Their 900 shared rows are pixel-exact, so it is one card."""
    cards, frames, baseline = profile
    real_align = pc._align
    calls = []

    def align(a, b, *rest):
        calls.append(None)
        return None if len(calls) == 2 else real_align(a, b, *rest)

    monkeypatch.setattr(pc, "_align", align)
    result = pc.extract_photo_cards(frames)
    assert result.components == 2 and result.shifts[1] is None
    assert result.merges == ("pixel",)
    assert len(result) == len(baseline) == 4
    p1 = result.cards[1]
    assert p1.first_frame == 1  # first appearance survives the merge
    assert p1.component == 1 and not p1.complete  # the larger partial view is kept
    want = cards[2].pixels[_PHOTO - p1.height :]
    assert p1.button_boxes == _expected_bottom_boxes(
        p1.height
    )  # its bottom edge was seen
    _assert_pixels(p1, want)


def _card(serial, component, r0, canvas, **kw):
    fields = dict(
        serial=serial,
        component=component,
        r0=r0,
        height=len(canvas),
        canvas=canvas,
        top=None,
        bot=None,
        caption=None,
        white=0.0,
        media=1.0,
        first_frame=serial,
        first_y=0,
        n_inst=1,
        stitched=False,
        video_like=False,
        x_rows=0,
        static_known=False,
        complete=False,
        is_photo=True,
    )
    fields.update(kw)
    return pc._Card(**fields)


def _noise(seed, height=600, lo=30, hi=220, channel_bias=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    img = rng.integers(lo, hi, size=(height, _CW, 3), dtype=np.int16)
    return np.clip(img + np.array(channel_bias), 0, 255).astype(np.uint8)


def _distinct_static(n, component):
    """n complete, static, mutually dissimilar cards (separate hue ranges)."""
    out = []
    for i in range(n):
        bias = [(-60, 0, 60), (60, -60, 0), (0, 60, -60), (60, 60, -60)][i % 4]
        out.append(
            _card(
                100 + 10 * component + i,
                component,
                1000 * i,
                _noise(1000 + 10 * component + i, lo=60, hi=120, channel_bias=bias),
                static_known=True,
                complete=True,
                top=0,
                bot=974,
            )
        )
    return out


def test_six_slot_rule_merges_a_video_seen_on_both_sides_of_a_break():
    """Seven photo cards survive the pixel test because one video card straddles a chain break
    (two DIFFERENT video frames, no exact rows). The pair at the boundary is merged."""
    import cv2 as _cv2

    left = _distinct_static(3, 0)
    right = _distinct_static(2, 1)
    video_end = _card(
        1, 0, 9000, _noise(1), video_like=True
    )  # last card of component 0
    video_start = _card(
        2, 1, -50, _noise(2), video_like=True
    )  # first card of component 1
    kept = [*left, video_end, video_start, *right]
    merges = []
    out = pc._merge_across_components(kept, merges, _cv2, np)
    assert merges == ["hinge6_boundary"]
    assert len(out) == 6
    survivors = {id(c) for c in out}
    assert (id(video_end) in survivors) != (id(video_start) in survivors)
    assert all(id(c) in survivors for c in left + right)


def test_six_slot_rule_prefers_the_pair_straddling_the_break_over_a_closer_match():
    """The break is usually a video that could not be aligned, so the boundary pair wins even
    against a more similar pair elsewhere (both unproven, so both are eligible)."""
    import cv2 as _cv2

    left = _distinct_static(2, 0)
    right = _distinct_static(2, 1)
    # partial, not last of component 0 / not first of component 1, alike only to each other
    look_alike_0 = _card(3, 0, 1500, _noise(3, lo=150, hi=250))
    look_alike_1 = _card(4, 1, 500, _noise(4, lo=150, hi=250))
    video_end = _card(1, 0, 9000, _noise(1), video_like=True)
    video_start = _card(2, 1, -50, _noise(2, lo=60), video_like=True)
    boundary_similarity = pc._colour_similarity(
        video_end.canvas, video_start.canvas, np
    )
    other_similarity = pc._colour_similarity(
        look_alike_0.canvas, look_alike_1.canvas, np
    )
    assert pc._BOUNDARY_SIMILARITY < boundary_similarity < pc._ANY_PAIR_SIMILARITY
    assert other_similarity > max(boundary_similarity, pc._ANY_PAIR_SIMILARITY)
    kept = [*left, look_alike_0, video_end, video_start, look_alike_1, *right]
    merges = []
    out = pc._merge_across_components(kept, merges, _cv2, np)
    assert merges == [
        "hinge6_boundary",
        "hinge6",
    ]  # 8 -> 7 at the boundary, then 7 -> 6
    assert len(out) == 6
    survivors = {id(c) for c in out}
    assert (id(video_end) in survivors) != (id(video_start) in survivors)
    assert (id(look_alike_0) in survivors) != (id(look_alike_1) in survivors)


def test_six_slot_rule_never_merges_cards_the_pixels_proved_distinct():
    import cv2 as _cv2

    kept = _distinct_static(4, 0) + _distinct_static(3, 1)
    merges = []
    out = pc._merge_across_components(list(kept), merges, _cv2, np)
    assert merges == [] and len(out) == 7


def test_pixel_rule_keeps_the_larger_view_and_the_earliest_appearance():
    import cv2 as _cv2

    full = _noise(5, height=974)
    top_view = _card(1, 0, 3000, full[:600].copy(), first_frame=1, first_y=900)
    bottom_view = _card(2, 1, 300, full[200:].copy(), first_frame=4, first_y=300)
    merges = []
    out = pc._merge_across_components([top_view, bottom_view], merges, _cv2, np)
    assert merges == ["pixel"]
    assert len(out) == 1 and out[0] is bottom_view
    assert (bottom_view.first_frame, bottom_view.first_y) == (1, 900)
    assert bottom_view.n_inst == 2


# =============================================================================================
# Failing loud
# =============================================================================================


def test_undecodable_frame_raises():
    with pytest.raises(pc.PhotoCardsError, match="frame 0 did not decode"):
        pc.extract_photo_cards([b"not a png"])


def test_non_bytes_frame_raises():
    with pytest.raises(pc.PhotoCardsError, match="frame 1 is str"):
        pc.extract_photo_cards([b"x", "not bytes"])


def test_a_frame_of_another_size_raises_instead_of_measuring_with_the_wrong_ruler():
    ok, buf = cv2.imencode(".png", np.full((1200, 540, 3), 128, np.uint8))
    assert ok
    with pytest.raises(pc.PhotoCardsError, match="540x1200"):
        pc.extract_photo_cards([buf.tobytes()])


def test_a_missing_like_template_raises(profile, monkeypatch):
    _, frames, _ = profile
    monkeypatch.setattr(hinge, "_load_template", lambda name: None)
    with pytest.raises(pc.PhotoCardsError, match="did not load"):
        pc.extract_photo_cards(frames[:1])
