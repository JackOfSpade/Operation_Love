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

# Logged with every Hinge crop.  The selection policy has a stable product-level identifier,
# while this names the pixel classifier revision underneath it so threshold changes are visible
# in a bug report instead of silently changing what the policy means.
CROP_CLASSIFIER_ID = "hinge_crop_type_v2"

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


def crop_type_evidence(image: bytes | None) -> dict[str, object]:
    """Return the crop verdict and the bounded pixel evidence that produced it.

    The dictionary contains only aggregate geometry/statistics, never OCR text or pixels, so it
    is safe to place in the existing non-image capture manifest.  ``classify_crop`` is a thin
    projection of this function; diagnostics and policy therefore cannot calculate subtly
    different answers.
    """
    unknown: dict[str, object] = {
        "classifier_id": CROP_CLASSIFIER_ID,
        "classification": UNKNOWN,
        "width": None,
        "height": None,
        "colour_std": None,
        "dominant_background": None,
        "edge_density": None,
        "large_uniform_panel": None,
        "text_layout": None,
    }
    if not image:
        return unknown
    try:
        import cv2
        import numpy as np
    except ImportError:
        return unknown
    encoded = np.frombuffer(image, dtype=np.uint8)
    rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[0] < 16 or rgb.shape[1] < 16:
        return unknown

    original_height, original_width = rgb.shape[:2]
    # Work at a bounded size so this stays a cheap pre-flight check even for full card crops.
    max_side = max(rgb.shape[:2])
    if max_side > 256:
        scale = 256 / max_side
        rgb = cv2.resize(rgb, (max(16, round(rgb.shape[1] * scale)),
                               max(16, round(rgb.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    height, width = rgb.shape[:2]

    # The black circular Hinge heart is UI chrome, not card content.  On a short white prompt it
    # occupies enough pixels to push global colour variation beyond the written-card bound.  Mask
    # only its stable lower-right control region for aggregate statistics; the image sent to the
    # model and the stored verification signature remain completely untouched.
    yy, xx = np.ogrid[:height, :width]
    heart_control = (((xx - 0.91 * width) / (0.13 * width)) ** 2
                     + ((yy - 0.85 * height) / (0.19 * height)) ** 2 <= 1)
    content = ~heart_control

    rgb_f = rgb.astype(np.float32)
    pixels = rgb_f[content]
    median = np.median(pixels, axis=0)
    colour_std = float(pixels.std(axis=0).mean())
    dominant_background = float(
        (np.abs(rgb_f - median).max(axis=2)[content] <= 24).mean())

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient = np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))
    edge_density = float((gradient[content] >= 80).mean())
    large_uniform_panel = bool(_has_large_uniform_panel(gray))
    text_layout = bool(_has_text_layout(gray, content, float(np.median(gray[content]))))

    classification = UNKNOWN
    if colour_std >= 38 and edge_density >= 0.12 and dominant_background <= 0.62:
        # A written prompt can put a large, calm text panel over a visually busy background.
        # Its global variation then looks photographic even though the item is plainly a card.
        # Do not try to decide that mixed composition from pixels alone: a broad uniform panel is
        # enough to make the evidence ambiguous, so return UNKNOWN and leave the mandatory
        # post-tap verifier as the safety backstop.
        # A normal photo caption or a calm sky can also form a uniform panel.  It is only
        # prompt-like when that panel is accompanied by a globally dominant background; the
        # real rectangular sunset card that exposed the numbering bug has a small white title
        # band but only 16% dominant-background pixels across the complete crop.
        if not (large_uniform_panel and dominant_background >= 0.45):
            classification = PHOTO
    elif (dominant_background >= 0.78 and 0.008 <= edge_density <= 0.22
          and (colour_std <= 52
               or (dominant_background >= 0.84 and text_layout))):
        # The second branch closes a real false negative: large serif prompt text, an emoji and
        # the heart control can make a plainly white text card exceed the old global std limit.
        # It still requires an even stronger dominant background plus affirmative multi-row glyph
        # structure; a quiet wall/sky photograph therefore remains UNKNOWN rather than being
        # discarded because it happens to have low variation.
        classification = WRITTEN
    elif (dominant_background < 0.78 and colour_std >= 18 and edge_density >= 0.02
          and not (large_uniform_panel and dominant_background >= 0.45
                   and colour_std >= 38)):
        # Low-detail portraits and landscape cards need not clear the deliberately strong global
        # PHOTO gate above.  They are still affirmative photographs when the crop has real colour
        # and edge structure, lacks a dominant card background, and is not a high-variation mixed
        # large-panel composition. A low-contrast photo can contain calm panel-like regions, so
        # panel geometry alone is not a veto. This is content evidence, not an aspect-ratio or
        # item-count shortcut.
        classification = PHOTO

    return {
        "classifier_id": CROP_CLASSIFIER_ID,
        "classification": classification,
        "width": int(original_width),
        "height": int(original_height),
        "colour_std": round(colour_std, 4),
        "dominant_background": round(dominant_background, 6),
        "edge_density": round(edge_density, 6),
        "large_uniform_panel": large_uniform_panel,
        "text_layout": text_layout,
    }


def classify_crop(image: bytes | None) -> str:
    """Classify a numbered crop conservatively from colour variation and edge density.

    Text cards have a dominant near-uniform background with a sparse set of sharp glyph
    edges.  A photograph has substantial colour variation *and* widespread gradients.  The
    conjunctions below are intentional: a low-detail photograph and a graphical prompt card
    both become UNKNOWN instead of triggering a false stop.
    """
    return str(crop_type_evidence(image)["classification"])


def _has_text_layout(gray, content, background: float) -> bool:
    """Whether a calm bright card has multiple rows of glyph-like foreground components.

    This is deliberately stronger than merely finding dark pixels.  A person against a white
    wall can have a dominant background too, but normally forms a few large connected regions;
    prompt typography forms many small components aligned into two or more text rows.
    """
    import cv2
    import numpy as np

    height, width = gray.shape[:2]
    if background < 160:
        return False
    foreground = ((gray.astype(np.float32) <= background - 40) & content).astype(np.uint8)
    count, _labels, stats, centres = cv2.connectedComponentsWithStats(
        foreground, connectivity=8)
    glyph_centres: list[float] = []
    crop_area = height * width
    for i in range(1, count):
        _x, _y, component_w, component_h, area = stats[i]
        box_area = int(component_w) * int(component_h)
        if (2 <= area <= 0.025 * crop_area
                and component_w <= 0.22 * width
                and 2 <= component_h <= 0.22 * height
                and box_area > 0 and area / box_area <= 0.92):
            glyph_centres.append(float(centres[i][1]))
    if len(glyph_centres) < 12:
        return False

    # Greedily cluster vertical centres. Four glyph components in each of two separated rows is
    # enough; punctuation fragmentation may add components, but cannot manufacture row alignment.
    rows: list[list[float]] = []
    tolerance = max(3.0, 0.055 * height)
    for centre in sorted(glyph_centres):
        if not rows or centre - sum(rows[-1]) / len(rows[-1]) > tolerance:
            rows.append([centre])
        else:
            rows[-1].append(centre)
    substantial = [row for row in rows if len(row) >= 4]
    return (len(substantial) >= 2
            and (sum(substantial[-1]) / len(substantial[-1])
                 - sum(substantial[0]) / len(substantial[0])) >= 0.08 * height)


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
    for _x, _y, panel_w, panel_h, area in stats[1:count]:
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
