"""Embedder.embed_whole_photos (operation_love/vision/embed.py): the whole-photo feature.

The CLIP model and its preprocess are FAKES that record what they were handed, so these tests
pin the contract (pad every card to a square with CLIP's mean colour, run the embedder's OWN
model and preprocess, L2 each card, GeM-pool with no cosine dedup, L2 the pool) without loading
weights. The cards come either from a stubbed extractor (exact control over their pixels) or
from the real extractor on tests/test_photo_cards.py's synthetic frames; no real screencap is
ever used.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest
import torch
from PIL import Image

from operation_love.vision import embed, photo_cards
from operation_love.vision.embed import (
    CLIP_MEAN_RGB,
    Embedder,
    WholePhotoEmbedding,
    dedup_by_cosine,
    gem_pool,
    l2_normalize,
    pad_to_square,
)
from operation_love.vision.photo_cards import (
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
    PhotoCard,
    PhotoCards,
)


class _FakeClip:
    """Records every encode_image batch; returns preset (unnormalised) rows in call order."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.batches = []
        self.lock = threading.Lock()

    def encode_image(self, batch):
        with self.lock:
            self.batches.append(batch)
            start = sum(len(b) for b in self.batches[:-1]) % max(1, len(self.rows))
            out = [self.rows[(start + i) % len(self.rows)] for i in range(len(batch))]
        return torch.tensor(out, dtype=torch.float32)


class _FakePreprocess:
    """Records the PIL image it was given; returns a tiny tensor tagged with the call index."""

    def __init__(self):
        self.images = []
        self.lock = threading.Lock()

    def __call__(self, image):
        with self.lock:
            self.images.append(image)
            index = len(self.images)
        return torch.full((3, 2, 2), float(index))


def _embedder(rows):
    emb = Embedder()
    emb._arc = object()  # non-None: short-circuits _ensure()'s lazy model load
    emb._device = "cpu"
    emb._clip = _FakeClip(rows)
    emb._clip_preprocess = _FakePreprocess()
    return emb


def _row(seed):
    return np.random.default_rng(seed).normal(size=WHOLE_PHOTO_DIMS).tolist()


def _unit(vec):
    return l2_normalize([float(x) for x in vec])


def _cards(shapes, digest="cd" * 32, band_cards=()):
    cards = []
    for order, (h, w) in enumerate(shapes):
        pixels = np.random.default_rng(order).integers(
            0, 256, (h, w, 3), dtype=np.uint8
        )
        pixels.setflags(write=False)
        cards.append(
            PhotoCard(
                order=order,
                pixels=pixels,
                height=h,
                completeness=min(1.0, h / 974),
                complete=h >= 970,
                stitched=False,
                video_like=False,
                first_frame=order,
                component=0,
                page_y0=0,
                top_observed=True,
                bottom_observed=True,
                caption_rows=0,
                x_overlay_rows=0,
                white_share=0.0,
                media_row_share=1.0,
                content_run_share=1.0,
                button_boxes=(),
                instances=1,
            )
        )
    return PhotoCards(
        cards=tuple(cards),
        frames_sha256=digest,
        frame_count=len(cards),
        template="10_8",
        components=1,
        shifts=(),
        merges=(),
        band_cards=band_cards,
    )


@pytest.fixture
def stub_cards(monkeypatch):
    """Replace the extractor inside embed.py; records the frames it was called with."""
    calls = []

    def install(result):
        def fake(frames):
            calls.append(list(frames))
            return result

        monkeypatch.setattr(embed, "extract_photo_cards", fake)
        return calls

    return install


# =============================================================================================
# pad_to_square
# =============================================================================================


def test_clip_mean_rgb_is_openai_clips_normalisation_mean():
    constants = pytest.importorskip("open_clip.constants")
    assert CLIP_MEAN_RGB == tuple(round(255 * m) for m in constants.OPENAI_DATASET_MEAN)


@pytest.mark.parametrize(("h", "w"), [(600, 974), (974, 974), (974, 300), (1, 5)])
def test_pad_to_square_centres_the_card_and_fills_with_the_clip_mean(h, w):
    card = np.random.default_rng(h * w).integers(0, 256, (h, w, 3), dtype=np.uint8)
    out = pad_to_square(card)
    side = max(h, w)
    assert out.shape == (side, side, 3) and out.dtype == np.uint8
    oy, ox = (side - h) // 2, (side - w) // 2
    assert (out[oy : oy + h, ox : ox + w] == card).all()  # nothing cropped or resampled
    mask = np.ones((side, side), bool)
    mask[oy : oy + h, ox : ox + w] = False
    assert (out[mask] == np.array(CLIP_MEAN_RGB, np.uint8)).all()


# =============================================================================================
# embed_whole_photos
# =============================================================================================


def test_uses_the_embedders_own_clip_on_padded_cards_and_gem_pools_them(stub_cards):
    shapes = [(974, 974), (600, 974), (974, 974)]
    calls = stub_cards(_cards(shapes, digest="ef" * 32))
    rows = [_row(i) for i in range(3)]
    emb = _embedder(rows)
    frames = [b"frame-%d" % i for i in range(12)]

    result = emb.embed_whole_photos(frames)

    assert calls == [frames]  # ALL frames, in order, no filter
    images = emb._clip_preprocess.images
    assert len(images) == 3
    cards = _cards(shapes).cards
    for image, card in zip(images, cards, strict=True):
        assert isinstance(image, Image.Image)
        assert image.size == (974, 974)
        assert (np.array(image) == pad_to_square(card.pixels)).all()
    (batch,) = emb._clip.batches
    assert tuple(batch.shape) == (
        3,
        3,
        2,
        2,
    )  # one stacked batch of the preprocess outputs
    assert [float(batch[i, 0, 0, 0]) for i in range(3)] == [1.0, 2.0, 3.0]
    expected = l2_normalize(gem_pool([_unit(r) for r in rows]))
    assert isinstance(result, WholePhotoEmbedding)
    assert result.card_count == 3
    assert result.frames_sha256 == "ef" * 32
    assert result.feature_version == WHOLE_PHOTO_FEATURE_VERSION
    assert len(result.vector) == WHOLE_PHOTO_DIMS
    assert result.vector == pytest.approx(expected, abs=1e-6)
    assert math.isclose(math.sqrt(sum(x * x for x in result.vector)), 1.0, abs_tol=1e-9)


def test_cards_are_not_cosine_deduplicated(stub_cards):
    """The cards are already unique photos; two that CLIP finds alike both count."""
    stub_cards(_cards([(974, 974)] * 3))
    a = _row(1)
    near_a = [x + 0.01 * y for x, y in zip(a, _row(2), strict=True)]
    rows = [a, near_a, _row(3)]
    units = [_unit(r) for r in rows]
    assert len(dedup_by_cosine(units)) == 2  # the premise: dedup WOULD drop one
    result = _embedder(rows).embed_whole_photos([b"f"])
    assert result.card_count == 3
    assert result.vector == pytest.approx(l2_normalize(gem_pool(units)), abs=1e-6)
    assert result.vector != pytest.approx(
        l2_normalize(gem_pool(dedup_by_cosine(units))), abs=1e-4
    )


@pytest.mark.parametrize("frames", [[b"a", b"b"], []], ids=["two-frames", "no-frames"])
def test_zero_cards_raise_no_photo_cards_and_never_call_clip(
    stub_cards, frames, capsys
):
    """A Hinge profile always carries photos: zero cards is an extraction failure (RT-2), so
    it raises rather than returning a vector-less embedding anything could store or score."""
    stub_cards(_cards([], digest="00" * 32))
    emb = _embedder([_row(0)])
    with pytest.raises(photo_cards.NoPhotoCards) as raised:
        emb.embed_whole_photos(frames)
    assert isinstance(raised.value, photo_cards.PhotoCardsError)
    assert (f"{len(frames)} frame(s)" if frames else "no frames") in str(raised.value)
    assert emb._clip.batches == [] and emb._clip_preprocess.images == []
    assert "Whole-photo embedding error: NoPhotoCards" in capsys.readouterr().out


def test_the_real_extractor_on_frames_with_no_card_raises_no_photo_cards():
    import cv2

    ok, buf = cv2.imencode(".png", np.full((2400, 1080, 3), 245, np.uint8))
    assert ok
    with pytest.raises(photo_cards.NoPhotoCards, match="12 frame"):
        _embedder([_row(0)]).embed_whole_photos([buf.tobytes()] * 12)


def test_the_band_verdicts_travel_with_the_embedding(stub_cards):
    verdicts = (photo_cards.BAND_TEXT, photo_cards.BAND_CONTIGUOUS)
    stub_cards(_cards([(974, 974)], band_cards=verdicts))
    result = _embedder([_row(1)]).embed_whole_photos([b"f"])
    assert result.band_cards == verdicts


def test_a_single_card_is_its_own_l2_vector(stub_cards):
    stub_cards(_cards([(974, 974)]))
    row = _row(9)
    result = _embedder([row]).embed_whole_photos([b"f"])
    assert result.card_count == 1
    assert result.vector == pytest.approx(_unit(row), abs=1e-6)


def test_more_cards_than_one_batch_are_all_embedded(stub_cards):
    n = embed._WHOLE_PHOTO_BATCH + 1
    stub_cards(_cards([(974, 974)] * n))
    rows = [_row(i) for i in range(n)]
    emb = _embedder(rows)
    result = emb.embed_whole_photos([b"f"])
    assert [len(b) for b in emb._clip.batches] == [embed._WHOLE_PHOTO_BATCH, 1]
    assert result.card_count == n
    assert result.vector == pytest.approx(
        l2_normalize(gem_pool([_unit(r) for r in rows])), abs=1e-6
    )


def test_a_model_error_is_raised_never_a_partial_vector(stub_cards, capsys):
    stub_cards(_cards([(974, 974)] * 2))
    emb = _embedder([_row(0)])

    def broken(batch):
        raise RuntimeError("CLIP exploded")

    emb._clip.encode_image = broken
    with pytest.raises(RuntimeError, match="CLIP exploded"):
        emb.embed_whole_photos([b"f"])
    assert (
        "Whole-photo embedding error: RuntimeError: CLIP exploded"
        in capsys.readouterr().out
    )


def test_unreadable_frames_raise_through_the_real_extractor():
    with pytest.raises(photo_cards.PhotoCardsError):
        _embedder([_row(0)]).embed_whole_photos([b"not a png"])


@pytest.mark.parametrize(
    "bad_row",
    [[float("nan")] + [0.1] * (WHOLE_PHOTO_DIMS - 1), [0.1] * (WHOLE_PHOTO_DIMS - 1)],
    ids=["non_finite", "wrong_width"],
)
def test_a_non_finite_or_wrong_width_card_vector_raises(stub_cards, bad_row):
    stub_cards(_cards([(974, 974)]))
    with pytest.raises(ValueError, match="expected 768 finite floats"):
        _embedder([bad_row]).embed_whole_photos([b"f"])


def test_a_broken_model_fails_loud_before_extraction(stub_cards):
    stub_cards(_cards([]))
    emb = Embedder()

    def broken_ensure():
        raise OSError("weights missing")

    emb._ensure = broken_ensure
    with pytest.raises(OSError, match="weights missing"):
        emb.embed_whole_photos([])


def test_concurrent_calls_give_identical_vectors(stub_cards):
    stub_cards(_cards([(974, 974)] * 4))
    rows = [_row(i) for i in range(4)]
    emb = _embedder(rows)
    results, errors = [], []

    def run():
        try:
            results.append(emb.embed_whole_photos([b"f"]).vector)
        except Exception as exc:  # noqa: BLE001 — surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(results) == 8
    assert all(r == results[0] for r in results)


def test_end_to_end_on_synthetic_frames_embeds_every_extracted_card():
    from tests.test_photo_cards import _profile

    _, frames = _profile()
    cards = photo_cards.extract_photo_cards(frames)
    rows = [_row(i) for i in range(len(cards))]
    emb = _embedder(rows)
    result = emb.embed_whole_photos(frames)
    assert result.card_count == len(cards) == 4
    assert result.frames_sha256 == photo_cards.frames_sha256(frames)
    for image, card in zip(emb._clip_preprocess.images, cards.cards, strict=True):
        assert (np.array(image) == pad_to_square(card.pixels)).all()
    assert result.vector == pytest.approx(
        l2_normalize(gem_pool([_unit(r) for r in rows])), abs=1e-6
    )
