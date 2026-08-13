"""A deliberately narrow, pixel-only check for doc 5.8's early item guard.

This module answers one question before a model-selected item causes a phone gesture:
does an unambiguous claim that the item is a photograph or a written prompt agree with
the crop we numbered?  It is intentionally a high-precision check.  It never attempts
to identify the subject or compare semantic content, and missing or mixed evidence is
``INCONCLUSIVE`` rather than a failure.

The public function is pure over image bytes and text.  Drivers own the lookup from a
model item number to the crop bytes; that keeps this module usable in synthetic tests
and prevents the worker from learning about any concrete driver's crop storage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Explicit vocabulary for callers and tests.  Keep these strings stable: worker.py only
# acts on MISMATCH, while MATCH and INCONCLUSIVE are both safe to continue.
MATCH = "match"
MISMATCH = "mismatch"
INCONCLUSIVE = "inconclusive"

PHOTO = "photo"
WRITTEN = "written"
UNKNOWN = "unknown"

_PHOTO_WORDS = frozenset(("photo", "photos", "photograph", "photographs", "image",
                          "images", "selfie", "selfies"))
_WRITTEN_WORDS = frozenset(("prompt", "prompts", "answer", "answers", "response",
                            "responses"))
_WORDS = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class ItemTypePreflight:
    """The complete, auditable result of one coarse item-type comparison.

    ``description_type`` and ``crop_type`` are each PHOTO, WRITTEN or UNKNOWN.  ``state``
    is MATCH only when both confident classifications agree, MISMATCH only when both are
    confident and disagree, and INCONCLUSIVE otherwise.  The last category deliberately
    passes: doc 5.8 values precision over recall because post-tap verification remains the
    actual safety backstop.
    """

    state: str
    description_type: str
    crop_type: str
    reason: str

    @property
    def mismatch(self) -> bool:
        return self.state == MISMATCH


def classify_description(item_description: str | None) -> str:
    """Return only an unambiguous photo/written claim from model free text.

    A description mentioning both vocabularies ("a photo of her prompt") is not evidence
    for either side.  We deliberately do not infer from broad words such as "card", "bio",
    "quote", or "picture": they are too easy to use metaphorically or to describe content
    shown inside a photo.
    """
    words = set(_WORDS.findall((item_description or "").lower()))
    says_photo = bool(words & _PHOTO_WORDS)
    says_written = bool(words & _WRITTEN_WORDS)
    if says_photo == says_written:
        return UNKNOWN
    return PHOTO if says_photo else WRITTEN


def classify_crop(image: bytes | None) -> str:
    """Classify a numbered crop conservatively from colour variation and edge density.

    Text cards have a dominant near-uniform background with a sparse set of sharp glyph
    edges.  A photograph has substantial colour variation *and* widespread gradients.  The
    conjunctions below are intentional: a low-detail photograph and a graphical prompt card
    both become UNKNOWN instead of triggering a false stop.
    """
    if not image:
        return UNKNOWN
    try:
        import cv2
        import numpy as np
    except ImportError:
        return UNKNOWN
    encoded = np.frombuffer(image, dtype=np.uint8)
    rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[0] < 16 or rgb.shape[1] < 16:
        return UNKNOWN

    # Work at a bounded size so this stays a cheap pre-flight check even for full card crops.
    max_side = max(rgb.shape[:2])
    if max_side > 256:
        scale = 256 / max_side
        rgb = cv2.resize(rgb, (max(16, round(rgb.shape[1] * scale)),
                               max(16, round(rgb.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    rgb_f = rgb.astype(np.float32)
    colour_std = float(rgb_f.reshape(-1, 3).std(axis=0).mean())
    median = np.median(rgb_f.reshape(-1, 3), axis=0)
    dominant_background = float((np.abs(rgb_f - median).max(axis=2) <= 24).mean())

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient = np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))
    edge_density = float((gradient >= 80).mean())

    # These deliberately leave a wide no-verdict band.  They describe the two obvious
    # synthetic/card extremes, not a universal visual taxonomy.
    if colour_std >= 38 and edge_density >= 0.12 and dominant_background <= 0.62:
        # A written prompt can put a large, calm text panel over a visually busy background.
        # Its global variation then looks photographic even though the item is plainly a card.
        # Do not try to decide that mixed composition from pixels alone: a broad uniform panel is
        # enough to make the evidence ambiguous, so return UNKNOWN and leave the mandatory
        # post-tap verifier as the safety backstop.
        if _has_large_uniform_panel(gray):
            return UNKNOWN
        return PHOTO
    if colour_std <= 52 and dominant_background >= 0.78 and 0.008 <= edge_density <= 0.22:
        return WRITTEN
    return UNKNOWN


def _has_large_uniform_panel(gray) -> bool:
    """Whether a busy crop contains a sizeable, rectangular low-variation region.

    This is intentionally a one-way ambiguity detector, not a text recognizer.  A prompt card
    can place dark glyphs on a light/coloured panel over an image, while a photo can contain a
    quiet sky or wall; both must make this coarse classifier decline the PHOTO verdict rather
    than turn a possible prompt into an AUTO hard stop.  The panel must cover a substantial,
    box-like part of the crop after only a small closing operation bridges its glyph holes.
    """
    import cv2
    import numpy as np

    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, (7, 7), normalize=True)
    mean_sq = cv2.boxFilter(gray_f * gray_f, cv2.CV_32F, (7, 7), normalize=True)
    local_std = np.sqrt(np.maximum(0, mean_sq - mean * mean))
    # 12 grey levels is deliberately tight: it identifies a panel's fill, not ordinary
    # low-contrast photographic texture.  Closing bridges only the letters inside that fill.
    flat = (local_std <= 12).astype(np.uint8)
    kernel = np.ones((7, 7), dtype=np.uint8)
    joined = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, kernel)
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(joined, connectivity=8)
    height, width = gray.shape[:2]
    crop_area = height * width
    for x, y, panel_w, panel_h, area in stats[1:count]:
        box_area = int(panel_w) * int(panel_h)
        if (area >= 0.12 * crop_area
                and panel_w >= 0.45 * width
                and panel_h >= 0.25 * height
                and box_area > 0
                and area / box_area >= 0.72):
            return True
    return False


def preflight_item_type(item_description: str | None, image: bytes | None) -> ItemTypePreflight:
    """Compare the two coarse classes without I/O, model calls, or driver state."""
    description_type = classify_description(item_description)
    crop_type = classify_crop(image)
    if description_type == UNKNOWN or crop_type == UNKNOWN:
        return ItemTypePreflight(
            INCONCLUSIVE, description_type, crop_type,
            "the item description or crop did not unambiguously identify a photo or written item")
    if description_type == crop_type:
        return ItemTypePreflight(MATCH, description_type, crop_type,
                                 f"both the description and crop classify the item as {crop_type}")
    return ItemTypePreflight(
        MISMATCH, description_type, crop_type,
        f"the model describes a {description_type}, but the numbered crop is confidently {crop_type}")
