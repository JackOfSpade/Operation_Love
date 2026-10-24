"""Whole-photo cards: the unique photo/video media of one Hinge profile, cut from its frames.

The owner decides on the PHOTOS, all of them. The ranker's v1 vector only ever sees a
face-centred crop of each full-screen frame, so a profile's faceless photos, its settings and
the photos' composition are invisible to it, and a photo seen in three frames counts three
times. This module recovers what the owner actually looked at: every unique photo card of the
profile, once, as the card's own media pixels with the caption and app chrome cut away.
`Embedder.embed_whole_photos` (vision/embed.py) then embeds and pools those cards.

It is a pure, deterministic function of the frame BYTES: no model, no device, no network, no
global state. The only file it reads is the shipped like-glyph template pair (the segmentation
needs a heart template; see "Which like glyph" below), checked against pinned digests. Train/serve
parity rests on this: live capture and the backfill call this same function on the same stored
bytes (`Profile.photos` == `profile_photos` by `photo_index` == `auto_pass_photos`), so the
feature cannot depend on anything but those bytes and this code. Everything that shapes the
output and is not this module's own code is pinned here or fingerprinted by a test
(tests/test_photo_cards_fingerprint.py): the content band, the like-match threshold and the two
template files are pinned; `drivers.segment` is covered by the fingerprint. A change to any of
them changes stored vectors, so it needs a WHOLE_PHOTO_FEATURE_VERSION bump and a re-backfill.

ZERO CARDS. `extract_photo_cards` answers with the cards it found, possibly none. A Hinge
profile always carries photos (a video counts as a card), so zero cards from a non-empty frame
list means the extractor did not recognise the capture; `Embedder.embed_whole_photos` raises
`NoPhotoCards` for it rather than embedding nothing, and nothing ever stores a zero-card row.

Ported from the offline experiment that measured the feature (scratch `build2.py`, `lib.py`,
`lib2.py`; 502 labels, identity-grouped CV AUC 0.725 -> 0.749 with this vector added).

PIPELINE
--------
1. Decode every frame twice, exactly as the experiment did: greyscale with
   `cv2.IMREAD_GRAYSCALE` for segmentation and scroll alignment, RGB with PIL for the card
   pixels (the same decoder the embedder uses). Two decodes of one PNG are NOT interchangeable
   (an sRGB-tagged device PNG greys differently through cv2 than through an RGB->grey
   conversion), so each measurement keeps the decode it was calibrated on.
2. Segment each frame into card blocks with `drivers.segment.segment_frame`.
3. Chain consecutive frames by their vertical scroll shift into page space: textured-strip
   template matching verified at full resolution (tier 1), an exhaustive row-signature search
   accepted only when pixel-exact and unambiguous (tier 2), a +-4px tie-break toward the shift
   with the most pixel-exact textured rows, and, for a pair both tiers refuse, the video rescue
   (tier 3): an autoplaying video changes its pixels between frames, so the overlap's MEDIAN
   row disagrees although the scroll is known exactly; the rescue accepts a shift on a large
   set of rows that are pixel-exact outside the floating pass-X, unambiguous against every
   other shift, and confirmed by coincident observed card edges. An unaligned pair starts a new
   chain component whose page offset relative to the previous one is unknown.
4. Group block instances that overlap by more than 30 page rows within one component into
   one card; reconstruct each card's photo region from its observed edges (cutting a white
   caption strip above the 974-row media square); stitch the region from the instance that
   shows most of it first, rows ABOVE Hinge's floating pass-X before rows under it, so the
   overlay is never pasted into the card when another frame shows the same rows clear.
   Instances that disagree by more than 3 grey levels on the X-free columns are a video (or
   animation) and are never stitched: the card is its single most complete instance. The like
   button is then painted over (see "The like button" below).
5. Keep photo cards only (see "Photo or text" below) with at least 500 media rows.
6. Merge one card seen in two chain components (whose relative offset is unknown): first on
   >= 60 pixel-exact textured rows at one vertical offset (decisive for static media), then,
   while more than Hinge's six photo slots survive, the most similar cross-component pair,
   preferring the pair straddling the break, never a pair the pixels proved distinct.
7. Return the survivors in order of first appearance (frame, then row).

WHICH LIKE GLYPH (build-agnostic)
---------------------------------
`segment_frame` needs the like-heart template, and Hinge 10.8.0 changed the glyph. The caller
never says which build captured the frames: both shipped templates are tried on the first three
frames and the one that finds more hearts segments the whole profile (one profile is one
build), which is exactly how the experiment chose per profile. On a tie the pre-10.8 template is
used (the experiment's order); a tie can only mean both found the same hearts or none, and with
no hearts at all the two segmentations are identical.

THE LIKE BUTTON (masked, build-independent)
-------------------------------------------
Hinge overlays its like button (a black disc with a white heart, whose glyph Hinge 10.8.0
changed) on the bottom-right of every card. Left in, it would put the app BUILD into the vector,
so every card canvas has a fixed region painted with `CLIP_MEAN_RGB` (which CLIP's Normalize
maps to ~0) before anything measures it, and that region is excluded from every pixel statistic
below. Measured on ~400 local calibration frames of both builds (2026-10-23): the heart centre
sits at frame x 938 (pre-10.8) / 939 (10.8), 89 / 90 rows above the card's bottom edge on every
card, and the dark disc ends at radius 61..64 with a faint halo to ~72. The region is card
columns 809..961 and the rows from 166 to 14 above the media's bottom edge: the disc plus a halo
margin for either build, wherever that bottom edge is known in page space (observed, or the top
edge plus the caption plus the 974-row square). It never depends on which template matched.
Only when the bottom edge is unknown is a 152px square centred on each heart `segment_frame`
found in the card painted instead; the two builds' heart centres differ by a pixel, so such a
card's pixels can differ between builds by that pixel, and a button no template matched (one cut
by the content band's edge) stays in.
`extract_photo_cards(frames, paint_like_button=False)` is for DISPLAY only (the hub's review
card shows the owner the photos as they looked): every card is still measured, classified and
merged on the painted canvas, so it returns exactly the default call's cards, with the button
left in their pixels. Nothing embedded or stored may come from it.

PHOTO OR TEXT (no model)
------------------------
The experiment classified a card as a photo when its near-white share (every channel >= 232)
was below 0.5, never when it was 0.8 or more, and asked a CLIP zero-shot prompt set in between.
This module keeps both white-share cuts and replaces the CLIP question with two pixel cues; a
card in the 0.5..0.8 band is a photo when EITHER holds:
* edge media: photo and video media fill the card edge to edge, while prompt, voice, poll and
  vitals cards draw on a white surface with white padding at both sides. A row of the card
  counts as MEDIA when a 24px strip just inside the card's right edge, or the same strip inside
  its left edge on a row not taken from under the pass-X (which overlaps the left edge), is
  mostly not card white. At least 25% of the card's rows must be media rows.
* one contiguous region: a photo against a white backdrop (a studio wall, a bright room) is
  card white at both edges, so edge media alone drops it as text. Its content is still one
  unbroken region, while a text card's ink comes in line bands separated by blank rows. A row
  CARRIES CONTENT when at least 3% of its pixels are not card white; the longest run of such
  rows must cover at least half of the card's rows.
`PhotoCards.band_cards` records how the two cues decided every card in the band, so the
backfill can count on the real corpus how far this departs from the CLIP rule. Residual: a text
card whose ink forms one unbroken band over half its height would be kept (no such Hinge card is
known).

SIMPLIFICATIONS AND CHANGES RELATIVE TO build2.py (every one)
------------------------------------------------------------
* No model anywhere: the CLIP zero-shot photo/text classifier is replaced by the two pixel cues
  above (it only ever decided cards with white share in [0.5, 0.8)).
* The like button is masked on every card canvas (build2 kept it; its offline verify predicted
  the Hinge-10.8 template from the whole-photo vector at AUC 0.993). White share, edge media,
  content rows and the colour histogram all ignore the masked pixels.
* The like-match threshold (0.75) and the template files are pinned, not read live from the
  driver: the threshold is a constant here, the files are checked against their sha256.
* The cross-component pixel test runs for every cross-component pair; the experiment ran it
  only when the pair's whole-card CLIP cosine exceeded 0.5, which was a speed gate (two crops of
  one image with >= 60 pixel-exact rows are never CLIP-dissimilar).
* The six-slot merge ranks and gates pairs by a colour-histogram intersection (512-bin joint
  RGB, in [0, 1]) instead of the whole-card CLIP cosine, with the experiment's two gates kept
  numerically (> 0.5 for the pair straddling a chain break, > 0.75 otherwise). This rule fires
  only when more than six photo cards survive every pixel test, and it is UNCALIBRATED against
  the corpus: the histogram is a substitute for the cosine, not a measurement of it.
* Ties are broken deterministically and platform-independently: candidate shifts are tried in
  order of support, then first appearance (the experiment iterated a Python set); every
  `argsort` is stable (the experiment used numpy's default introsort, whose tie order is an
  implementation detail).
* A frame that cannot be decoded or segmented raises `PhotoCardsError` (the experiment skipped
  a frame `segment_frame` raised on; an undecodable frame failed its whole profile anyway at the
  RGB decode). Frames must all be 1080x2400, the device geometry every pixel constant here was
  measured on; anything else raises rather than measuring with the wrong ruler.
* Dropped as diagnostics or as inputs to arms that lost: per-card face detection and ArcFace,
  the frame-level v1 replica, the zero-shot probabilities, CLIP duplicate groups, edge-spread
  and shift-tier bookkeeping.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # numpy is the `ml`/`hinge` extra; this module must import without it
    import numpy as np

# Single source of truth for the whole-photo feature's identity and width. The store, the
# loader and the decider import these; nothing else may spell them.
WHOLE_PHOTO_FEATURE_VERSION = "whole_photo_v1"
WHOLE_PHOTO_DIMS = 768  # CLIP ViT-L-14 image embedding width

# --- geometry, measured on the 1080x2400 Pixel 7a by the experiment -------------------------
_FRAME_SIZE = (1080, 2400)  # (width, height) every constant below assumes
# HINGE_SPEC.content_band when the corpus was captured. Pinned here rather than read from the
# spec: it is part of this feature version's definition, and a later spec change must not
# silently move the stored backfill and the live vector apart.
_CONTENT_BAND = (0.125, 0.875)
_PHOTO_H = 974  # a Hinge photo card's media is a 973x974 square
_MIN_INSTANCE_ROWS = 120  # ignore block instances shorter than this
_MERGE_OVERLAP_ROWS = 30  # same component + page overlap above this => same card
_MIN_KEEP_ROWS = (
    500  # keep a photo card only if at least this many media rows were seen
)
_VIDEO_MAD = (
    3.0  # instances of one card disagreeing by more than this => video/animation
)
_STATIC_MAD = 1.0  # instances agreeing within this => static media, known to be still
_WHITE_MIN = 232  # "card white": every channel >= this
_PHOTO_WHITE_MAX = 0.5  # below this white share a card is a photo outright
_TEXT_WHITE_MIN = 0.8  # at or above it a card is never a photo
_EDGE_STRIP = (8, 32)  # card columns, measured from each side, of the edge-media probe
_EDGE_WHITE_MAX = 0.5  # a probe strip less white than this shows media on that row
_MIN_MEDIA_ROWS = 0.25  # media-row share that makes a 0.5..0.8-white card a photo
_CAPTION_MIN_ROWS, _CAPTION_MAX_ROWS = 90, 300  # caption strips measured 135 / 189 rows
# Hinge's floating pass-X: a fixed SCREEN position, measured from the disagreement of 312
# exactly aligned frame pairs at y ~1955..2100 (band end), x ~40..200, padded.
_X_Y0, _X_X1 = 1940, 220
_XFREE_C0 = _X_X1 - 53  # first card column right of the X (cards start at x=53)
_HINGE_PHOTO_SLOTS = 6  # every Hinge profile carries exactly six photo/video slots
_PIXEL_MERGE_ROWS = 60  # pixel-exact textured rows that prove two canvases one card
_DISTINCT_MAX_ROWS = 20  # fewer exact rows than this between static cards => distinct
_BOUNDARY_SIMILARITY = 0.5
_ANY_PAIR_SIMILARITY = 0.75
_TEXTURED_STD = 8.0  # a row is "textured" when its grey std exceeds this
_CONTENT_ROW_MIN = 0.03  # a row with at least this share not card white carries content
# longest content-run share that makes a 0.5..0.8-white card a photo
_CONTENT_RUN_MIN = 0.5
# CLIP's (OpenAI) normalisation mean, 255 * (0.48145466, 0.4578275, 0.40821073), rounded: the
# square padding of embed.py and the like-button mask below, both ~0 after CLIP's Normalize.
CLIP_MEAN_RGB = (123, 117, 104)
# The like-button mask (module docstring "The like button"), in card columns and in rows above
# the media's bottom edge (half-open, so rows bottom-166 .. bottom-14 are painted).
_BUTTON_COLS = (809, 962)
_BUTTON_ROWS_ABOVE_BOTTOM = (166, 13)
# half side of the heart-centred square painted when the bottom edge is unknown
_BUTTON_HALF = 76
# Like-glyph templates, oldest build first (the experiment's tie order), with the sha256 of the
# file each one must be: a recropped template moves every heart and so every card.
_LIKE_TEMPLATES = (
    (
        "pre_10_8",
        "hinge_like_button.png",
        "23e89ebc6e8bc525d71b73111d7e55a42d2c905234623343637b87d0c9ce5957",
    ),
    (
        "10_8",
        "hinge_like_button_10_8.png",
        "4a3a351653fb6f35983f6a477dfa5291ecd9acb86db854fd23c17dff96ddf8a5",
    ),
)
# hinge._LIKE_MATCH_THRESHOLD when this feature version was defined; pinned like _CONTENT_BAND
# so a driver recalibration cannot move the stored backfill and the live vector apart.
_LIKE_MATCH_THRESHOLD = 0.75
_TEMPLATE_HEAD_FRAMES = 3


class PhotoCardsError(RuntimeError):
    """The frames could not be read as a Hinge profile at all: missing cv2/numpy/PIL, a frame
    that does not decode, a frame of the wrong size, a like template that is missing or not the
    pinned file, or a segmentation that could not run."""


class NoPhotoCards(PhotoCardsError):
    """The frames were read but hold no photo card (raised by `Embedder.embed_whole_photos`).

    A Hinge profile always carries photos, so this is an extraction failure (a layout the
    extractor does not recognise), never a fact to record: nothing stores a zero-card row, an
    AUTO decision stops, and the backfill reports the label under its own category."""


@dataclass(frozen=True)
class PhotoCard:
    """One unique photo/video card: its media pixels and how they were obtained.

    `pixels` is an (height, 974, 3) uint8 RGB array of the photo media only (caption strip and
    page chrome excluded), read-only. `height` is its row count; a complete card is 974.
    `button_boxes` are the (row0, row1, col0, col1) half-open card-pixel boxes painted with
    `CLIP_MEAN_RGB` over the like button (empty when no button position was known; with
    `paint_like_button=False` they are where it was painted for measuring, not in `pixels`).
    `stitched` means rows came from more than one frame (always pixel-exact: frames are aligned
    by exact scroll shifts); `video_like` means the card's instances disagreed like an
    autoplaying video, in which case it is never stitched. `x_overlay_rows` counts rows that
    could only be taken from under the floating pass-X."""

    order: int
    pixels: np.ndarray
    height: int
    completeness: float
    complete: bool
    stitched: bool
    video_like: bool
    first_frame: int
    component: int
    page_y0: int
    top_observed: bool
    bottom_observed: bool
    caption_rows: int | None
    x_overlay_rows: int
    white_share: float
    media_row_share: float
    content_run_share: float
    button_boxes: tuple[tuple[int, int, int, int], ...]
    instances: int


# How the two pixel cues decided one card in the 0.5..0.8 white band (`PhotoCards.band_cards`).
BAND_EDGE_MEDIA = "edge_media"  # kept by edge media alone
BAND_CONTIGUOUS = "contiguous"  # kept by the contiguous-content cue alone
BAND_BOTH = "both"  # kept by both
BAND_TEXT = "text"  # neither: dropped as a prompt/vitals card


@dataclass(frozen=True)
class PhotoCards:
    """Every unique photo card of one profile, in order of first appearance.

    `frames_sha256` is the sha256 hex of the concatenation of each frame's own sha256 hex digest,
    in the order given: it identifies exactly which bytes the cards were cut from, and is what
    the store records next to the embedding. `template` names the like glyph that segmented the
    frames ("pre_10_8" or "10_8"; None with no frames). `components` is the number of scroll
    chain pieces (1 when every consecutive pair aligned). `merges` lists the rule behind each
    cross-component merge ("pixel", "hinge6_boundary", "hinge6"). `band_cards` holds one
    `BAND_*` verdict for every reconstructed card whose white share is in [0.5, 0.8), kept or
    not, in order of first appearance (a diagnostic: it is where this module departs from the
    experiment's CLIP rule)."""

    cards: tuple[PhotoCard, ...]
    frames_sha256: str
    frame_count: int
    template: str | None
    components: int
    shifts: tuple[int | None, ...]
    merges: tuple[str, ...]
    band_cards: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.cards)


def frames_sha256(frames: Sequence[bytes]) -> str:
    """sha256 hex of the concatenated per-frame sha256 hex digests, in order. Pure."""
    joined = "".join(hashlib.sha256(bytes(frame)).hexdigest() for frame in frames)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _require_libs():
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 — surfaced as PhotoCardsError, never swallowed
        raise PhotoCardsError(
            "whole-photo cards need opencv-python, numpy and pillow "
            f"(extras: operation-love[hinge,ml]); import failed: {exc}"
        ) from exc
    return cv2, np, Image


def _like_templates():
    """The shipped like-glyph templates (each checked against its pinned sha256), loaded through
    the driver's own cached loader, and the pinned match threshold."""
    from ..drivers import hinge

    out = []
    for name, filename, expected in _LIKE_TEMPLATES:
        try:
            digest = hashlib.sha256((hinge._ASSETS / filename).read_bytes()).hexdigest()
        except OSError as exc:
            raise PhotoCardsError(
                f"like-glyph template {filename} could not be read: {exc}"
            ) from exc
        if digest != expected:
            raise PhotoCardsError(
                f"like-glyph template {filename} is not the file {WHOLE_PHOTO_FEATURE_VERSION} "
                f"was defined with (sha256 {digest}, pinned {expected}); a changed template "
                "moves every card, so bump WHOLE_PHOTO_FEATURE_VERSION, pin the new digest and "
                "re-run python -m tools.backfill_photo_features"
            )
        template = hinge._load_template(filename)
        if template is None:
            raise PhotoCardsError(f"like-glyph template {filename} did not load")
        out.append((name, template))
    return tuple(out), _LIKE_MATCH_THRESHOLD


# =============================================================================================
# Scroll alignment (lib.py estimate_shift, lib2.py rescue_shift)
# =============================================================================================


@dataclass(frozen=True)
class _Band:
    full: Any  # (rows, 974) float32 grey of the content band's card columns
    half: Any  # the same at half resolution (INTER_AREA)


def _band(gray, card_x: tuple[int, int], cv2, np) -> _Band:
    height = gray.shape[0]
    r0, r1 = (
        int(round(_CONTENT_BAND[0] * height)),
        int(round(_CONTENT_BAND[1] * height)),
    )
    full = gray[r0:r1, card_x[0] : card_x[1]].astype(np.float32)
    half = cv2.resize(
        full, (full.shape[1] // 2, full.shape[0] // 2), interpolation=cv2.INTER_AREA
    )
    return _Band(full=full, half=half)


def _overlap_stats(af, bf, s: int, np):
    """Median row-MAD and textured-row count of the overlap implied by shift s (B[y]~A[y+s])."""
    if s >= 0:
        a, b = af[s:], bf[: af.shape[0] - s]
    else:
        a, b = af[: af.shape[0] + s], bf[-s:]
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    if n < 40:
        return None, 0, n
    mad = float(np.median(np.abs(a - b).mean(axis=1)))
    textured = int((b.std(axis=1) > _TEXTURED_STD).sum())
    return mad, textured, n


def _refine(af, bf, s0: int, np, radius: int = 4):
    best = (None, None, 0, 0)
    for s in range(int(s0) - radius, int(s0) + radius + 1):
        mad, textured, n = _overlap_stats(af, bf, s, np)
        if mad is not None and (best[0] is None or mad < best[0]):
            best = (mad, s, textured, n)
    return best


def _row_signature(frame, np, bins: int = 16):
    width = frame.shape[1] // bins * bins
    return frame[:, :width].reshape(frame.shape[0], bins, -1).mean(axis=2)


def _estimate_shift(a: _Band, b: _Band, cv2, np, n_strips=12, strip=24, min_score=0.97):
    """Vertical scroll shift s (full-res px) with B[y] ~= A[y + s], or None, plus the tier-1
    candidate shifts (the rescue reuses them). Tier 1: textured-strip template matches,
    verified at full res. Tier 2: exhaustive search over 16-bin row signatures, accepted only
    when pixel-exact and unambiguous. Returns (shift, tier, candidates)."""
    af, ah, bf, bh = a.full, a.half, b.full, b.half
    n = bh.shape[0]
    candidates: list[int] = []
    section = n // n_strips
    for k in range(n_strips):
        lo, hi = k * section, min(n - strip, (k + 1) * section)
        if hi <= lo:
            continue
        row_var = bh[lo : hi + strip].var(axis=1)
        window = np.convolve(row_var, np.ones(strip), "valid")[: hi - lo]
        yb = lo + int(np.argmax(window))
        template = bh[yb : yb + strip]
        if template.std() < 3:
            continue
        scores = cv2.matchTemplate(ah, template, cv2.TM_CCOEFF_NORMED)[:, 0]
        ya = int(np.argmax(scores))
        if float(scores[ya]) >= min_score:
            candidates.append((ya - yb) * 2)
    values = np.array(candidates)
    unique = list(dict.fromkeys(candidates))  # first-appearance order
    unique.sort(key=lambda v: -int(np.sum(np.abs(values - v) <= 4)))  # stable
    tried: list[int] = []
    for v in unique:
        if any(abs(v - t) <= 4 for t in tried):
            continue
        tried.append(v)
        mad, s, textured, _ = _refine(af, bf, v, np)
        if mad is None:
            continue
        support = int(np.sum(np.abs(values - v) <= 4))
        others = len(values) - support
        if (mad <= 2.0 and textured >= 30) or (
            mad <= 10.0 and support >= 2 and support > others
        ):
            return s, 1, candidates
    sa, sb = _row_signature(af, np), _row_signature(bf, np)
    height = sa.shape[0]
    dist = np.full(height, np.inf)
    for s in range(0, height - 60):
        dist[s] = np.abs(sa[s:] - sb[: height - s]).mean()
    order = np.argsort(dist, kind="stable")
    s1 = int(order[0])
    rest = [int(x) for x in order[1:50] if abs(int(x) - s1) > 8]
    d2 = float(dist[rest[0]]) if rest else np.inf
    mad, s, textured, _ = _refine(af, bf, s1, np, radius=3)
    if (
        mad is not None
        and mad <= 1.0
        and textured >= 30
        and d2 > 2.0 * float(dist[s1]) + 1.0
    ):
        return s, 2, candidates
    return None, None, candidates


def _frame_edges(segmentation) -> list[int]:
    """Observed block edges (frame rows) of one FrameSegmentation."""
    out = []
    for block in segmentation.blocks:
        if block.top.observed:
            out.append(int(block.y0))
        if block.bottom.observed:
            out.append(int(block.y1))
    return sorted(set(out))


def _edge_agreement(ea, eb, s: int, r0: int, r1: int, tol: int = 3, margin: int = 8):
    """B[y] ~ A[y+s]: B's observed edges landing inside A's band with / without an A edge within
    tol, and the same the other way round. Returns (matched, unmatched)."""
    matched = unmatched = 0
    for yb in eb:
        ya = yb + s
        if r0 + margin <= ya <= r1 - margin:
            if any(abs(ya - x) <= tol for x in ea):
                matched += 1
            else:
                unmatched += 1
    for ya in ea:
        yb = ya - s
        if r0 + margin <= yb <= r1 - margin:
            if any(abs(yb - x) <= tol for x in eb):
                matched += 1
            else:
                unmatched += 1
    return matched, unmatched


def _exact_rows(af, bf, s: int, np, c0: int = _XFREE_C0, thr: float = 1.0):
    """(count, longest run, overlap) of overlap rows implied by s that are textured and
    pixel-exact on the X-free columns."""
    height = af.shape[0]
    if s >= 0:
        a, b = af[s:, c0:], bf[: height - s, c0:]
    else:
        a, b = af[: height + s, c0:], bf[-s:, c0:]
    n = min(a.shape[0], b.shape[0])
    if n < 20:
        return 0, 0, n
    a, b = a[:n], b[:n]
    ok = (np.abs(a - b).mean(1) <= thr) & (b.std(1) > _TEXTURED_STD)
    if not ok.any():
        return 0, 0, n
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return int(ok.sum()), int(max(len(r) for r in runs)), n


def _xfree_signature(frame, np, c0: int = _XFREE_C0, bins: int = 16):
    frame = frame[:, c0:]
    width = frame.shape[1] // bins * bins
    return frame[:, :width].reshape(frame.shape[0], bins, -1).mean(axis=2), frame.std(1)


def _rescue_shift(a: _Band, b: _Band, ea, eb, candidates, band_rows, np):
    """Video-robust shift for a pair `_estimate_shift` refused: a large set of textured rows
    PIXEL-EXACT on the X-free columns, unambiguous against every other shift, and confirmed by
    the segmentation's observed card edges. Returns the shift or None."""
    af, bf = a.full, b.full
    height = af.shape[0]
    sa, _ = _xfree_signature(af, np)
    sb, sdb = _xfree_signature(bf, np)
    shifts = np.arange(-(height - 40), height - 40)
    counts = np.zeros(len(shifts), np.int32)
    textured_b = sdb > _TEXTURED_STD
    for k, s in enumerate(shifts):
        if s >= 0:
            x, y, t = sa[s:], sb[: height - s], textured_b[: height - s]
        else:
            x, y, t = sa[: height + s], sb[-s:], textured_b[-s:]
        counts[k] = int(((np.abs(x - y).max(1) <= 1.5) & t).sum())
    top = [int(shifts[k]) for k in np.argsort(-counts, kind="stable")[:12]]
    tried: list[int] = []
    for v in list(candidates) + top:
        if not any(abs(v - c) <= 2 for c in tried):
            tried.append(int(v))
    # (n_exact, run, overlap, s) at each candidate's best shift within +-3
    scored = [
        max(
            ((*_exact_rows(af, bf, s, np), s) for s in range(v - 3, v + 4)),
            key=lambda t: t[0],
        )
        for v in tried
    ]
    scored.sort(key=lambda t: -t[0])
    n1, run1, _, s1 = scored[0]
    # runner-up: best exact count at any shift more than 8px away (coarse count, full-res
    # recheck), ranked by (count, shift) descending exactly as the experiment ranked it
    far = sorted(
        (
            (int(c), int(s))
            for c, s in zip(counts, shifts, strict=True)
            if abs(int(s) - s1) > 8
        ),
        reverse=True,
    )[:5]
    n2 = max(
        [_exact_rows(af, bf, s, np)[0] for _, s in far]
        + [t[0] for t in scored[1:] if abs(t[3] - s1) > 8]
        + [0]
    )
    matched, unmatched = _edge_agreement(ea, eb, s1, *band_rows)
    if n1 >= 120 and run1 >= 60 and n2 * 3 < n1 and matched >= 1 and unmatched == 0:
        return s1
    # weaker pixel evidence (a mostly-white vitals card beside a video) but >= 3 coincident
    # observed card edges and no contradicting one; a card-pitch alias would align the edges
    # but compare two different photos, leaving ~0 exact textured rows
    if n1 >= 30 and run1 >= 20 and n2 * 8 <= n1 and matched >= 3 and unmatched == 0:
        return s1
    # small overlap inside one card (no edge observable) that is pixel-exact over a long
    # contiguous textured run, with no competing shift
    if n1 >= 150 and run1 >= 120 and n2 * 5 <= n1 and unmatched == 0:
        return s1
    return None


def _align(a: _Band, b: _Band, ea, eb, band_rows, cv2, np) -> int | None:
    shift, tier, candidates = _estimate_shift(a, b, cv2, np)
    if shift is None:
        return _rescue_shift(a, b, ea, eb, candidates, band_rows, np)
    # On low-texture overlaps the median-MAD minimum is flat over several px (measured: tier 2
    # accepted 663 where 663..667 all had MAD 0 but the textured pixel-exact rows peak sharply
    # at 665: 260 vs 42), so take the shift within +-4 that maximises pixel-exact textured rows
    # when it clearly beats the accepted one.
    exact = {
        v: _exact_rows(a.full, b.full, v, np)[0] for v in range(shift - 4, shift + 5)
    }
    best = max(exact, key=lambda v: (exact[v], -abs(v - shift)))
    if best != shift and exact[best] >= 1.5 * exact[shift] + 20:
        return best
    return shift


# =============================================================================================
# Cards (build2.py process)
# =============================================================================================


@dataclass(frozen=True)
class _Instance:
    frame: int
    y0: int
    y1: int
    top_observed: bool
    bottom_observed: bool
    component: int
    p0: int  # page rows within the component, half-open
    p1: int
    # like-heart centres (x, y) in frame coordinates
    hearts: tuple[tuple[int, int], ...] = ()


@dataclass(eq=False)
class _Card:
    """Mutable working record for one reconstructed card (a merge updates the survivor's
    first appearance and instance count). Page rows are within `component`. Compared by
    identity: it holds a pixel array, and two records are never the same card by value."""

    serial: int
    component: int
    r0: int
    height: int
    canvas: Any  # (height, 974, 3) uint8 RGB
    top: int | None
    bot: int | None
    caption: int | None
    white: float
    media: float
    first_frame: int
    first_y: int
    n_inst: int
    stitched: bool
    video_like: bool
    x_rows: int
    static_known: bool
    complete: bool
    is_photo: bool
    content: float = 0.0  # longest content-run share (module docstring "Photo or text")
    # (height, 974) bool, False on the painted like button; None = every pixel kept
    keep: Any = None
    buttons: tuple[tuple[int, int, int, int], ...] = ()
    band: str | None = None  # BAND_* verdict when 0.5 <= white < 0.8
    # the canvas before the button was painted; kept only for paint_like_button=False
    unpainted: Any = None


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _caption_rows_from_top(rows, np) -> int:
    """Rows of white caption strip at the top of a card (0 if none). `rows` are RGB rows
    starting exactly at the card's observed top edge."""
    white = (rows.min(-1) >= _WHITE_MIN).mean(1)
    if len(white) < 60 or white[:50].mean() < 0.8:
        return 0
    for r in range(_CAPTION_MIN_ROWS, min(len(white) - 10, _CAPTION_MAX_ROWS)):
        if white[r : r + 10].mean() < 0.6:
            return r
    return 0


def _media_row_share(canvas, from_x, np, keep=None) -> float:
    """Share of canvas rows whose right-edge strip (or left-edge strip, on a row not taken from
    under the pass-X) is mostly not card white: media fills a Hinge card edge to edge, a text
    card's padding is white. `keep` (bool, canvas-shaped) excludes the painted like button: a
    strip is judged on its kept pixels, and a strip with none says nothing."""
    width = canvas.shape[1]
    lo, hi = _EDGE_STRIP
    white = canvas.min(-1) >= _WHITE_MIN
    if keep is None:
        keep = np.ones(white.shape, bool)

    def media(cols):
        kept = keep[:, cols].sum(1)
        return (kept > 0) & (
            (white[:, cols] & keep[:, cols]).sum(1) < _EDGE_WHITE_MAX * kept
        )

    right = media(slice(width - hi, width - lo))
    left = media(slice(lo, hi)) & ~from_x
    return float((right | left).mean()) if len(canvas) else 0.0


def _content_run_share(canvas, np, keep=None) -> float:
    """Share of canvas rows covered by the longest run of rows that carry content (at least
    `_CONTENT_ROW_MIN` of their kept pixels not card white). One unbroken region of a photo
    gives a long run; a text card's ink lines are separated by blank rows."""
    if not len(canvas):
        return 0.0
    ink = canvas.min(-1) < _WHITE_MIN
    if keep is None:
        keep = np.ones(ink.shape, bool)
    kept = keep.sum(1)
    content = (kept > 0) & ((ink & keep).sum(1) >= _CONTENT_ROW_MIN * kept)
    if not content.any():
        return 0.0
    idx = np.flatnonzero(content)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return max(len(run) for run in runs) / len(content)


def _button_boxes(height, width, r0, media_bottom, hearts):
    """The like-button boxes (row0, row1, col0, col1), half-open, in canvas coordinates.

    `media_bottom` is the media's bottom edge in page rows (None when unknown); `hearts` are the
    card's heart centres in canvas coordinates. A known bottom edge places the one fixed
    region; only without it is each heart's square used (module docstring "The like button")."""
    if media_bottom is not None:
        bottom = media_bottom - r0
        rows_above, rows_below = _BUTTON_ROWS_ABOVE_BOTTOM
        raw = [(bottom - rows_above, bottom - rows_below, *_BUTTON_COLS)]
    else:
        raw = [
            (y - _BUTTON_HALF, y + _BUTTON_HALF, x - _BUTTON_HALF, x + _BUTTON_HALF)
            for x, y in hearts
        ]
    out = set()
    for a0, a1, b0, b1 in raw:
        a0, a1, b0, b1 = max(0, a0), min(height, a1), max(0, b0), min(width, b1)
        if a1 > a0 and b1 > b0:
            out.add((int(a0), int(a1), int(b0), int(b1)))
    return tuple(sorted(out))


def _colour_similarity(a, b, np, keep_a=None, keep_b=None) -> float:
    """Histogram intersection of two RGB canvases' 512-bin joint colour histograms, in [0, 1],
    over their kept pixels (`keep_*`: bool masks that exclude the painted like button)."""

    def hist(img, keep):
        pixels = img.reshape(-1, 3) if keep is None else img[keep]
        q = (pixels >> 5).astype(np.int32)
        h = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512)
        return h / max(1, h.sum())

    return float(np.minimum(hist(a, keep_a), hist(b, keep_b)).sum())


def _same_card_pixels(img_a, img_b, cv2, np, c0: int = _XFREE_C0) -> int:
    """Most textured rows (X-free columns) that are pixel-exact between two card canvases at one
    vertical offset, or 0 when the best offset is ambiguous."""
    ga = cv2.cvtColor(np.ascontiguousarray(img_a), cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    gb = cv2.cvtColor(np.ascontiguousarray(img_b), cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    sa, _ = _xfree_signature(ga, np)
    sb, sdb = _xfree_signature(gb, np)
    ha, hb = len(sa), len(sb)
    best = []
    for d in range(-(hb - 20), ha - 20):
        ya0, yb0 = max(0, d), max(0, -d)
        n = min(ha - ya0, hb - yb0)
        if n < 20:
            continue
        ok = (np.abs(sa[ya0 : ya0 + n] - sb[yb0 : yb0 + n]).max(1) <= 1.5) & (
            sdb[yb0 : yb0 + n] > _TEXTURED_STD
        )
        best.append((int(ok.sum()), d))
    if not best:
        return 0
    best.sort(reverse=True)
    c1, d1 = best[0]
    c2 = max([c for c, d in best[1:] if abs(d - d1) > 8] + [0])
    ya0, yb0 = max(0, d1), max(0, -d1)
    n = min(ha - ya0, hb - yb0)
    a, b = ga[ya0 : ya0 + n, c0:], gb[yb0 : yb0 + n, c0:]
    exact = int(((np.abs(a - b).mean(1) <= 1.0) & (b.std(1) > _TEXTURED_STD)).sum())
    return 0 if exact < 3 * c2 else exact


def _decode(frames: Sequence[bytes], cv2, np, Image):
    grays, rgbs = [], []
    for index, frame in enumerate(frames):
        data = bytes(frame)
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        try:
            with Image.open(io.BytesIO(data)) as image:
                rgb = np.array(image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 — re-raised as PhotoCardsError
            raise PhotoCardsError(f"frame {index} did not decode: {exc}") from exc
        if gray is None:
            raise PhotoCardsError(f"frame {index} did not decode ({len(data)} bytes)")
        size = (rgb.shape[1], rgb.shape[0])
        if size != _FRAME_SIZE or gray.shape != rgb.shape[:2]:
            raise PhotoCardsError(
                f"frame {index} is {size[0]}x{size[1]}; whole-photo cards are calibrated on "
                f"{_FRAME_SIZE[0]}x{_FRAME_SIZE[1]} screencaps"
            )
        grays.append(gray)
        rgbs.append(rgb)
    return grays, rgbs


def _segment_all(frames):
    """Segment every frame with the like template that finds more hearts on the first frames."""
    from ..drivers.segment import SegmentationError, segment_frame

    templates, threshold = _like_templates()

    def run(frame_index, template):
        try:
            return segment_frame(
                bytes(frames[frame_index]),
                content_band=_CONTENT_BAND,
                like_template=template,
                like_threshold=threshold,
            )
        except SegmentationError as exc:
            raise PhotoCardsError(
                f"frame {frame_index} could not be segmented: {exc}"
            ) from exc

    head = min(_TEMPLATE_HEAD_FRAMES, len(frames))
    trials = {
        name: [run(i, template) for i in range(head)] for name, template in templates
    }
    hearts = {name: sum(len(s.hearts) for s in segs) for name, segs in trials.items()}
    chosen = max((name for name, _ in templates), key=lambda name: hearts[name])
    template = dict(templates)[chosen]
    segs = trials[chosen] + [run(i, template) for i in range(head, len(frames))]
    return chosen, segs


def extract_photo_cards(
    frames: Sequence[bytes], *, paint_like_button: bool = True
) -> PhotoCards:
    """The profile's unique photo cards, in order of first appearance. See the module docstring.

    `frames` are the profile's full-screen screencaps in scroll order (`Profile.photos`), ALL of
    them: no quality filter, nothing reordered. Pure and deterministic; raises
    `PhotoCardsError` when the frames cannot be read, never returns a partial answer.
    `paint_like_button=False` returns the same cards with the like button left in their pixels,
    for display only (module docstring "The like button")."""
    for index, frame in enumerate(frames):
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise PhotoCardsError(f"frame {index} is {type(frame).__name__}, not bytes")
    digest = frames_sha256(frames)
    if not frames:
        return PhotoCards(
            cards=(),
            frames_sha256=digest,
            frame_count=0,
            template=None,
            components=0,
            shifts=(),
            merges=(),
        )
    cv2, np, Image = _require_libs()
    grays, rgbs = _decode(frames, cv2, np, Image)
    template, segs = _segment_all(frames)
    card_x = segs[0].card_x
    x0, x1 = card_x
    band_rows = segs[0].band
    bands = [_band(g, card_x, cv2, np) for g in grays]
    edges = [_frame_edges(s) for s in segs]
    del grays

    # ---- page offsets per chain component
    shifts: list[int | None] = []
    component, offset = [0], [0]
    for i in range(len(frames) - 1):
        s = _align(bands[i], bands[i + 1], edges[i], edges[i + 1], band_rows, cv2, np)
        shifts.append(s)
        if s is None:
            component.append(component[-1] + 1)
            offset.append(0)
        else:
            component.append(component[-1])
            offset.append(offset[-1] + s)
    del bands
    n_components = component[-1] + 1

    # ---- block instances, grouped into cards by page overlap within a component
    inst: list[_Instance] = []
    for f, seg in enumerate(segs):
        for block in seg.blocks:
            if block.height < _MIN_INSTANCE_ROWS:
                continue
            inst.append(
                _Instance(
                    frame=f,
                    y0=block.y0,
                    y1=block.y1,
                    top_observed=block.top.observed,
                    bottom_observed=block.bottom.observed,
                    component=component[f],
                    p0=block.y0 + offset[f],
                    p1=block.y1 + offset[f],
                    hearts=tuple((int(x), int(y)) for x, y in block.hearts),
                )
            )
    parent = list(range(len(inst)))
    for i in range(len(inst)):
        for j in range(i + 1, len(inst)):
            a, b = inst[i], inst[j]
            if a.frame == b.frame or a.component != b.component:
                continue
            if min(a.p1, b.p1) - max(a.p0, b.p0) > _MERGE_OVERLAP_ROWS:
                parent[_find(parent, i)] = _find(parent, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(inst)):
        groups.setdefault(_find(parent, i), []).append(i)

    cards: list[_Card] = []
    for members in groups.values():
        card = _reconstruct(
            [inst[i] for i in members],
            rgbs,
            offset,
            x0,
            x1,
            len(cards),
            np,
            keep_unpainted=not paint_like_button,
        )
        if card is not None:
            cards.append(card)

    # ---- photo or text (module docstring), then the size floor
    band_cards = tuple(
        c.band
        for c in sorted(cards, key=lambda c: (c.first_frame, c.first_y))
        if c.band is not None
    )
    photos = [c for c in cards if c.is_photo]
    photos.sort(key=lambda c: (c.first_frame, c.first_y))
    kept = [c for c in photos if c.height >= _MIN_KEEP_ROWS]

    merges: list[str] = []
    if n_components > 1 and len(kept) > 1:
        kept = _merge_across_components(kept, merges, cv2, np)
    kept.sort(key=lambda c: (c.first_frame, c.first_y))

    out = []
    for order, c in enumerate(kept):
        pixels = np.ascontiguousarray(c.canvas if c.unpainted is None else c.unpainted)
        pixels.setflags(write=False)
        out.append(
            PhotoCard(
                order=order,
                pixels=pixels,
                height=int(c.height),
                completeness=round(min(1.0, c.height / _PHOTO_H), 4),
                complete=bool(c.complete),
                stitched=bool(c.stitched),
                video_like=bool(c.video_like),
                first_frame=int(c.first_frame),
                component=int(c.component),
                page_y0=int(c.r0),
                top_observed=c.top is not None,
                bottom_observed=c.bot is not None,
                caption_rows=c.caption,
                x_overlay_rows=int(c.x_rows),
                white_share=round(float(c.white), 4),
                media_row_share=round(float(c.media), 4),
                content_run_share=round(float(c.content), 4),
                button_boxes=tuple(c.buttons),
                instances=int(c.n_inst),
            )
        )
    return PhotoCards(
        cards=tuple(out),
        frames_sha256=digest,
        frame_count=len(frames),
        template=template,
        components=n_components,
        shifts=tuple(shifts),
        merges=tuple(merges),
        band_cards=band_cards,
    )


def _reconstruct(
    group, rgbs, offset, x0, x1, serial, np, keep_unpainted: bool = False
) -> _Card | None:
    """One card's photo region from its instances, or None when too little of it was seen.
    `keep_unpainted` also keeps a copy of the canvas from before the like button is painted."""
    comp = group[0].component
    tops = [x.p0 for x in group if x.top_observed]
    bots = [x.p1 for x in group if x.bottom_observed]
    cov0, cov1 = min(x.p0 for x in group), max(x.p1 for x in group)
    top = int(np.median(tops)) if tops else None
    bot = int(np.median(bots)) if bots else None
    caption: int | None = None
    media_bottom: int | None = bot  # page row of the media's bottom edge, when known
    if top is not None and bot is not None:
        if bot - top > _PHOTO_H + 20:
            r0, r1, caption = bot - _PHOTO_H, bot, bot - top - _PHOTO_H
        else:
            r0, r1, caption = top, bot, 0
    elif bot is not None:
        r0, r1 = max(cov0, bot - _PHOTO_H), bot
    elif top is not None:
        src = max((x for x in group if x.top_observed), key=lambda x: x.y1 - x.y0)
        caption = _caption_rows_from_top(rgbs[src.frame][src.y0 : src.y1, x0:x1], np)
        r0 = top + caption
        r1 = min(cov1, r0 + _PHOTO_H)
        media_bottom = r0 + _PHOTO_H
    else:
        r0, r1 = cov0, cov1
    r0, r1 = max(r0, cov0), min(r1, cov1)
    if r1 - r0 < _MIN_INSTANCE_ROWS:
        return None

    # Consistency of the instances that overlap inside the photo region, measured on the
    # columns RIGHT of the floating pass-X (a fixed-screen overlay that otherwise reads as
    # disagreement): 0 = pixel-exact; large = autoplaying video / animation.
    omad, compared = 0.0, 0
    for ai in range(len(group)):
        for bj in range(ai + 1, len(group)):
            xa, xb = group[ai], group[bj]
            a0, a1 = max(xa.p0, xb.p0, r0), min(xa.p1, xb.p1, r1)
            if a1 - a0 < 20:
                continue
            ra = rgbs[xa.frame][
                a0 - offset[xa.frame] : a1 - offset[xa.frame], x0 + _XFREE_C0 : x1
            ].astype(np.int16)
            rb = rgbs[xb.frame][
                a0 - offset[xb.frame] : a1 - offset[xb.frame], x0 + _XFREE_C0 : x1
            ].astype(np.int16)
            omad = max(omad, float(np.median(np.abs(ra - rb).mean(axis=(1, 2)))))
            compared += 1
    video = omad > _VIDEO_MAD

    def coverage(
        x,
    ):  # rows of [r0, r1) this instance shows, and how many sit under the X
        c0, c1 = max(x.p0, r0), min(x.p1, r1)
        under = max(0, c1 - max(c0, _X_Y0 + offset[x.frame])) if c1 > c0 else 0
        return max(0, c1 - c0), under

    order = sorted(group, key=lambda x: (-coverage(x)[0], coverage(x)[1]))
    if video:
        order = order[:1]  # never stitch different video frames
    # Stitch the region in page space, best instance first. Pass 1 takes only rows that were
    # ABOVE the floating X in their frame; pass 2 fills what is left.
    canvas = np.zeros((r1 - r0, x1 - x0, 3), np.uint8)
    filled = np.zeros(r1 - r0, bool)
    from_x = np.zeros(r1 - r0, bool)
    contributors: set[int] = set()
    for xpass in (0, 1):
        for x in order:
            a0, a1 = max(x.p0, r0), min(x.p1, r1)
            if a1 <= a0:
                continue
            frame_rows = np.arange(a0, a1) - offset[x.frame]
            need = ~filled[a0 - r0 : a1 - r0]
            if xpass == 0:
                need &= frame_rows < _X_Y0
            if not need.any():
                continue
            rows = rgbs[x.frame][frame_rows[0] : frame_rows[-1] + 1, x0:x1]
            canvas[a0 - r0 : a1 - r0][need] = rows[need]
            filled[a0 - r0 : a1 - r0] |= need
            from_x[a0 - r0 : a1 - r0] |= need & (frame_rows >= _X_Y0)
            contributors.add(x.frame)
    if not filled.all():  # keep the longest contiguous filled run
        idx = np.flatnonzero(filled)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        canvas, from_x = canvas[run[0] : run[-1] + 1], from_x[run[0] : run[-1] + 1]
        r0, r1 = r0 + int(run[0]), r0 + int(run[-1]) + 1
    height = r1 - r0
    # Paint the like button before anything measures the canvas (module docstring).
    hearts = sorted(
        {(hx - x0, hy + offset[x.frame] - r0) for x in group for hx, hy in x.hearts}
    )
    buttons = _button_boxes(height, x1 - x0, r0, media_bottom, hearts)
    unpainted = canvas.copy() if keep_unpainted else None
    keep = None
    if buttons:
        keep = np.ones(canvas.shape[:2], bool)
        for a0, a1, b0, b1 in buttons:
            keep[a0:a1, b0:b1] = False
        canvas[~keep] = CLIP_MEAN_RGB
    is_white = canvas.min(-1) >= _WHITE_MIN
    white = float(
        is_white.mean()
        if keep is None
        else (is_white & keep).sum() / max(1, int(keep.sum()))
    )
    media = _media_row_share(canvas, from_x, np, keep)
    content = _content_run_share(canvas, np, keep)
    edge_media = media >= _MIN_MEDIA_ROWS
    contiguous = content >= _CONTENT_RUN_MIN
    band = None
    if height >= _MIN_INSTANCE_ROWS and _PHOTO_WHITE_MAX <= white < _TEXT_WHITE_MIN:
        band = (
            BAND_BOTH
            if edge_media and contiguous
            else BAND_EDGE_MEDIA
            if edge_media
            else BAND_CONTIGUOUS
            if contiguous
            else BAND_TEXT
        )
    is_photo = height >= _MIN_INSTANCE_ROWS and (
        white < _PHOTO_WHITE_MAX or (band is not None and band != BAND_TEXT)
    )
    first = min(group, key=lambda x: (x.frame, x.y0))
    return _Card(
        serial=serial,
        component=comp,
        r0=r0,
        height=height,
        canvas=canvas,
        top=top,
        bot=bot,
        caption=caption,
        white=white,
        media=media,
        first_frame=first.frame,
        first_y=first.y0,
        n_inst=len(group),
        stitched=len(contributors) > 1,
        video_like=bool(video),
        x_rows=int(from_x.sum()),
        static_known=bool(compared > 0 and omad <= _STATIC_MAD),
        complete=top is not None and bot is not None and height >= _PHOTO_H - 4,
        is_photo=bool(is_photo),
        content=content,
        keep=keep,
        buttons=buttons,
        band=band,
        unpainted=unpainted,
    )


def _merge_across_components(kept: list[_Card], merges: list[str], cv2, np):
    """Drop the duplicate of a card seen in two chain components (relative offset unknown).

    (1) "pixel": the two canvases share >= 60 PIXEL-EXACT textured rows at one vertical offset
        -> same card (decisive for static media, partial or not); keep the more complete one.
    (2) "hinge6": Hinge profiles carry exactly six photo/video slots, so while more than six
        survive, merge the most similar cross-component pair unless the pixels PROVED them
        distinct (both static-known, one complete, no exact rows). This is the only route for a
        video card seen in two components (two different video frames). A chain break is
        usually an overlap that could not be aligned because a VIDEO card filled it, so the
        pair straddling the break (last card of component c, first card of c+1) is the prime
        suspect and needs less similarity."""
    pair_cache: dict[tuple[int, int], tuple[float, int]] = {}

    def pair(a: _Card, b: _Card) -> tuple[float, int]:
        key = (min(a.serial, b.serial), max(a.serial, b.serial))
        if key not in pair_cache:
            pair_cache[key] = (
                _colour_similarity(a.canvas, b.canvas, np, a.keep, b.keep),
                _same_card_pixels(a.canvas, b.canvas, cv2, np),
            )
        return pair_cache[key]

    def merge(i: int, j: int, rule: str) -> list[_Card]:
        a, b = kept[i], kept[j]
        lo, hi = (a, b) if (a.height, -a.x_rows) < (b.height, -b.x_rows) else (b, a)
        if (lo.first_frame, lo.first_y) < (hi.first_frame, hi.first_y):
            hi.first_frame, hi.first_y = lo.first_frame, lo.first_y
        hi.n_inst += lo.n_inst
        merges.append(rule)
        return [c for c in kept if c is not lo]

    changed = True
    while changed:  # pixel merges first
        changed = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if kept[i].component == kept[j].component:
                    continue
                if pair(kept[i], kept[j])[1] >= _PIXEL_MERGE_ROWS:
                    kept = merge(i, j, "pixel")
                    changed = True
                    break
            if changed:
                break

    def proven_distinct(a: _Card, b: _Card, exact: int) -> bool:
        return (
            exact < _DISTINCT_MAX_ROWS
            and a.static_known
            and b.static_known
            and (a.complete or b.complete)
        )

    while len(kept) > _HINGE_PHOTO_SLOTS:
        last: dict[int, int] = {}
        first: dict[int, int] = {}
        for t, c in enumerate(kept):
            if c.component not in last or c.r0 > kept[last[c.component]].r0:
                last[c.component] = t
            if c.component not in first or c.r0 < kept[first[c.component]].r0:
                first[c.component] = t
        candidates = []
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if kept[i].component == kept[j].component:
                    continue
                similarity, exact = pair(kept[i], kept[j])
                if proven_distinct(kept[i], kept[j], exact):
                    continue
                a, b = (i, j) if kept[i].component < kept[j].component else (j, i)
                boundary = (
                    kept[b].component == kept[a].component + 1
                    and last.get(kept[a].component) == a
                    and first.get(kept[b].component) == b
                )
                if (boundary and similarity > _BOUNDARY_SIMILARITY) or (
                    similarity > _ANY_PAIR_SIMILARITY
                ):
                    candidates.append((boundary, similarity, i, j))
        if not candidates:
            break
        boundary, _, i, j = max(candidates, key=lambda t: (t[0], t[1]))
        kept = merge(i, j, "hinge6_boundary" if boundary else "hinge6")
    return kept
