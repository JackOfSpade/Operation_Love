"""The model's ACTUAL view of a profile: one cropped image per item, and nothing else.

`item_index.py` says what a profile's items are and where they are. This turns that into the
thing ops/OPENER-REDESIGN.md 5.2 and 5.3 specify sending: a numbered image per selectable item,
the heartless context blocks cropped and sent UNnumbered, and whatever policy excludes not sent
at all.

    build_item_payload(frames, index, ...) -> ItemPayload

Deliberately a leaf module on the same terms as the three below it: pure functions over frame
BYTES plus explicit calibration parameters, no device, no I/O, no global state, no `HingeDriver`
import — and, unlike `segment.py`, not even a deferred one, because nothing here needs the glyph
matcher.

WHY CROPS AND NOT FRAMES, RESTATED AS A CONSTRAINT ON THIS MODULE
------------------------------------------------------------------
Doc 5.2 gives three reasons, and the first two are the reason this module exists at all rather
than the prompt layer slicing images itself:

  1. **Index correspondence.** "With crops, image 3 in the request *is* item 3. Agreement by
     construction." So the numbering is produced HERE, next to the pixels it numbers, and
     `ItemPayload.images` emits the images in exactly that order.
  2. **Duplication bias.** "A card straddling a scroll seam appears in two or three frames...
     repetition reads as salience. We would bias selection by our own scroll cadence." So each
     item appears EXACTLY ONCE, from ONE frame, and this module holds no frame bytes at all —
     `ItemPayload` carries `frame_index` integers and crops, so there is no full frame in it to
     accidentally append alongside the crops.
  3. **The crops are the verification reference.** Doc 5.6's post-tap check is "a deterministic
     signature match between the sheet's displayed item and the stored crop of item N", so every
     crop carries a `CropSignature` computed at build time (see below).

THE FOUR RULES ABOUT WHAT MAY BE CROPPED
-----------------------------------------
1. **A crop is never a fragment.** Only a sighting that observed BOTH of a block's edges may be
   cropped (`IndexedBlock.croppable`, which is empty by construction on an `ITEM_PARTIAL`
   block). An item nobody ever bounded end to end comes back as a `CROP_UNCROPPABLE` entry
   carrying its page position and why — reported, never cropped to whatever happened to be
   visible. Cropping the visible part would send the model a half-photo it would then write an
   opener about, and store that half as 5.6's verification reference.
2. **No stitching, and the over-tall case is detected rather than assumed away.** Doc 5.10
   measured the tallest card observed end to end at 1114px against an 1800px content band (62%
   of the budget) and closed the viewport question on it, with the caveat that "a very long
   prompt answer could still exceed it, so the crop path should detect the case rather than
   assume it away". `_over_tall_failures` is that detection: a block taller than the analysed
   band of the frames in hand can never be contained by any single frame, so it is a failure
   naming stitching as deliberately unimplemented — not a taller crop assembled from two frames.
   The limit is READ OFF the segmentation rather than declared, so it follows `content_band`.
3. **One frame per item, chosen, not taken.** See `_choose_sighting`.
4. **Zero items is a failure.** A payload the model cannot answer is not a payload. It is the
   only "nothing here" case, and it is loud.

EXCLUSION IS A MECHANISM HERE, NOT A DETECTOR
-----------------------------------------------
Doc 2.4 excludes Hinge's "From people close to [name]" endorsement section entirely, because
"models reference what they are shown" and a tease built on a friend's line is the worst possible
ammunition. Doc 5.10 then measured that the endorsement block **did not appear in either
calibration capture** ("by OCR across all 115 frames in two passes... That is not proof of
absence").

So this module ships the exclusion MECHANISM and no detector for it. `exclude` is a caller-
supplied `IndexedBlock -> str | None` returning the REASON a block is dropped, `EXCLUSION_ENDORSEMENT`
is the named reason string doc 2.4's block will use when something can recognise it, and
`exclude_page_rows` builds a predicate from explicit page-row spans, which needs no detector at
all and is what a validation pass or an operator override can use today. Inventing a geometric
detector for a block type nobody has ever observed would be guessing, and `segment.py`'s own
docstring already records that no geometry it has separates the endorsement block from the vitals
block.

**Exclusion does not renumber navigation, and that is doc 5.3's index trap.** "If an excluded
block turns out to *have* a heart, the driver still counts it, because navigation counts hearts."
`ItemIndex` owns the heart ordinals and this module never touches them: every entry carries the
`heart_ordinal` its block had, excluded ones included. What exclusion DOES change is the model's
dense list, which is renumbered over the items actually sent — so `ItemPayload.translation`, not
`ItemIndex.translation`, is the table a navigation pass must use whenever anything selectable was
excluded. With no exclusions the two are identical, and the tests pin that.

THE CROP SIGNATURE
-------------------
Doc 5.6 needs to answer "is the sheet we just opened the item the opener was written for", and
doc 5.9 needs the same answer in observe mode ("the same deterministic signature match... On
mismatch the hub replaces the opener block with a warning"). Neither is built here; what is built
is the reference they compare against.

`CropSignature` is a fixed `_SIGNATURE_GRID` greyscale downsample of the crop, which is the
house's existing notion of a frame signature (`hinge._downsample`, a 24x24 grey resize whose
distance metric is `_band_dist`'s mean-absolute-difference) applied to a card instead of a screen.
Four properties are deliberate:

  * **Fixed grid, so scale is normalised.** Two items of a profile are different heights; the
    later sheet shows an item at a different size and position again. A fixed grid makes every
    signature comparable to every other, which is what a nearest-match check needs.
  * **Greyscale, matching the whole driver's vision layer.** Colour is preserved in the IMAGE the
    model sees (crops are decoded and encoded in colour — they are photographs); the signature
    drops it because the comparison it serves is a layout/content match.
  * **`signature_of` is the ONLY way to make a comparable signature from a screenshot**, and that
    is not politeness — it is a measured trap. `cv2.IMREAD_GRAYSCALE` and "decode colour, then
    `cvtColor(BGR2GRAY)`" do not agree: [corpus: 1.46 grey levels apart on average and 40 at the
    worst pixel on one real frame, and up to 9.8 apart on a crop's 32x32 signature — twice the
    4.66 that separates the two most alike items on that profile]. A verification pass that
    decoded its screenshot the other way would land nearer the wrong item than the right one. So
    the decode lives in one function that both sides call.
  * **`distance()` is pure Python over bytes**, no cv2 and no numpy, so a stored signature can be
    compared anywhere — the hub, a log replay, a test — not only where the vision extras are
    installed. `digest` is a sha256 of the same bytes, for exact identity in a log.

MEASURED ON THE CALIBRATION CORPUS (offline, 2026-08-12, geometry and counts only)
-----------------------------------------------------------------------------------
Run over the same gitignored captures the three modules below were calibrated on. The frames are
real people's profiles and never leave `ops/calibration/`.

  * `botscroll_20260811T231516Z` (profile B, 24 frames, `at_scroll_top=True`): **9 numbered items
    and 1 unnumbered context crop**, from the 11 indexed blocks — the leading chrome is dropped,
    and nothing is uncroppable. Every crop is 974px wide (the card band) and 685..1144px tall,
    all of them inside the 1800px analysed band with the tallest at 64% of it, which is doc
    5.10's no-stitching finding reproduced by the crop path itself. The 9 items come from 9
    DISTINCT frames, so no frame contributes two items and the duplication doc 5.2 warns about
    cannot arise. `payload.translation == index.translation == (1..9)`, since nothing was
    excluded. `truncated` False.
  * **STILL-PHOTO-ONLY AMENDMENT, corrected 2026-08-15.** Applying the photographic-content
    leg of `hinge_photos_only_v2` (the C1 classifier; the corpus predates the dwell and bound
    legs and cannot speak to them) to that same
    byte-exact corpus yields six numbered photos and demotes all three confidently WRITTEN prompt
    cards to unnumbered context. The production translation is therefore `(1, 3, 4, 6, 8, 9)`:
    prompt hearts 2, 5 and 7 stay in page space but can never be chosen. The first implementation
    also required a square crop because all six calibration photos happened to be 974x974. A live
    974x695 photo proved that geometry was incidental: it was demoted while a later square photo
    survived and was densely renamed item 1. The policy now uses affirmative content-based photo
    classification rather than geometry; UNKNOWN crops stay readable context instead of being
    allowed to manufacture a false photo ordinal.
    The nine-item figures above are the pre-policy measurement retained as historical evidence.
  * **Size, doc 5.2's "crops are also smaller and fewer than the frames they came from".**
    Measured: 10 crops totalling 8.53MB of PNG against the capture's 24 frames totalling
    37.15MB, i.e. 4.4x smaller and comfortably inside `opener._MAX_INLINE_REQUEST_BYTES`.
  * The same capture with `at_scroll_top=False`, and the 0.55 aliasing capture, both RAISE before
    a single crop is produced, because their indexes are unusable — the hard stop is inherited
    whole from `ItemIndex.usable` rather than re-derived here.
  * **Signature separability, the numbers the verification step needs — CORRECTED 2026-08-12,
    and the correction matters more than the original numbers.** An earlier revision of this
    docstring reported re-observation of one block from a different frame at "max 0.02 of 11
    pairs" against a 4.66 nearest-distinct-item distance, and concluded a ~230x margin. That was
    wrong in two compounding ways. It was measured on ONE profile, and it was measured only over
    the pairs where segmentation returned a byte-identical rect in both frames — which
    systematically excludes every card that is MOVING, because a moving card's segmented bottom
    edge wobbles by a pixel and drops it out of the sample. The exclusion selected for exactly
    the cards the number was supposed to be safe for.

    Re-measured by resampling each crop's own page rows in every frame that contains them
    (`_signature_drift`), at the 32x32 grid, in mean-absolute-difference grey levels (0..255):

    | quantity | profile B (24 frames, 9 items) | profile A (57 frames, 9 items) |
    |---|---|---|
    | re-observation drift, worst STATIC item | 0.018 | 0.24 |
    | re-observation drift, worst item | **2.66** (last card) | **25.0** (animated card) |
    | that same item's distance to its NEAREST other item | 44.7 | 47.9 |
    | smallest distance between ANY two items | 4.66 (items 5 and 7) | 6.27 (items 5 and 7) |
    | items the capture never re-observed, so unmeasurable | 1 of 9 | 1 of 9 |
    | items reported UNSEPARABLE | 0 | 0 |
    | median distinct pair | 79.2 | — |
    | rect located 1 / 2 / 4px off | 1.47 / 2.93 / 5.78 | — |

    Two conclusions, and they pull in opposite directions, which is why both are recorded.

    The old number was wrong by two orders of magnitude: re-observation is NOT "essentially
    exact", it reaches 2.66 on the very profile the 0.02 was measured on and 25.0 on the other.
    But the comparison that matters is PER ITEM, and on that basis both profiles still come out
    separable on every item they could measure — item 9 drifts 25.0 and sits 47.9 from its
    nearest neighbour, a margin of 1.9x rather than 230x. Comparing the worst drift against the
    smallest distance between ANY two items (25.0 against 6.27) inverts the answer and is the
    wrong comparison: item 9 is nowhere near items 5 and 7, and a verification pass asking "is
    this item 9" only has to beat item 9's own neighbours.

    So a fixed threshold is still ruled out — 0.02 would reject a correct re-observation on both
    profiles and 30 would accept a wrong item on either — and doc 5.4's alternative is what
    ships: explicit detection, per item, with no verdict baked in. Every crop carries its own
    measured drift and its distance to the nearest other item, and `ItemCrop.separable` is the
    comparison — three-valued, because an unmeasured crop is unknown rather than fine.

    The second constraint from the old table stands unchanged: a rect located 1/2/4px off costs
    1.47/2.93/5.78 grey levels, which past ~3px of error is enough to lose the two look-alike
    prompt cards on its own. That is an argument for the verification pass locating its rect by
    segmentation rather than by assumption, and for locating it from `ItemCrop.page_y0/page_y1`
    rather than the `IndexedBlock`'s (they can differ by up to `_EXTENT_TOLERANCE_PX`; measured
    maximum 1px, and the crop's is the rect the stored image actually came from).
  * **The grid barely matters, which is why it was not tuned.** [corpus: the nearest-distinct-pair
    distance runs 1.92 / 3.05 / 4.37 / 4.66 / 5.99 / 8.02 at grids 8 / 16 / 24 / 32 / 48 / 64,
    and the 1px rect-error cost runs 0.62 / 0.96 / 1.28 / 1.47 / 1.95 / 2.44 — both scale with
    the grid, so the ratio between them is 3.1..3.3 across all six.] The grid therefore buys
    absolute headroom against unmodelled noise and nothing else; see `_SIGNATURE_GRID`.
  * No threshold constant is shipped, deliberately, and the table above is now the argument for
    that rather than a caveat on it. Everything here measures crop-against-crop; doc 5.6's actual
    comparison is crop-against-like-sheet, which no capture in the corpus contains, and a
    threshold calibrated on the wrong pair of things would be a fixed number with no measurement
    behind it. What ships instead is the measurement itself, per profile and per item, so the
    verification step compares two numbers it was given rather than one number it inherited.

WHAT THIS MODULE DOES NOT DECIDE, AND WHICH LAYER HAS TO
----------------------------------------------------------
  * The PROMPT. It emits images and numbers; the sentence "the first N images are items 1..N"
    lives with the style guide, as does her name (lost by cropping, doc 5.2, already extracted at
    `hinge.py:2138`).
  * The COMPARISON. `CropSignature.distance` is the metric; what distance means "the same item",
    and the post-tap check itself, are doc 5.6's and are not built here.
  * WHAT TO DO ABOUT AN UNVERIFIABLE ITEM. `separable` is False on an animated card and None on
    an item nothing re-observed, and neither makes the payload unusable — deliberately. The other
    eight items of that profile are perfectly good references, and failing the whole payload
    would throw them away to punish one card; equally, tapping an item whose reference cannot
    distinguish it is doc 5.6's never-substitute rule to enforce, at the moment of the tap, not
    a decision to pre-empt at crop time (observe mode, doc 5.9, shows the same crops to a human
    and wants them either way). What this module owes the next layer is the number and the
    honesty that it is sometimes absent, and that is what it emits.
  * WHETHER A CAPTURE MEASURED ENOUGH. Drift needs the same page rows to appear in two frames.
    A capture that steps and never revisits measures nothing, reports `undetermined_items`, and
    is not wrong to. Sizing a scroll so items overlap between frames — which doc 5.10.1's "step
    at most about a third of the locally measured card spacing" already produces — is the closed
    loop's job.
  * WHICH context block is which. Inherited wholesale from `item_index.py` and `segment.py`:
    `CROP_CONTEXT` is one bucket, and doc 2.4's endorsement block is separated from the vitals
    block by the `exclude` argument or not at all.
  * The BUDGET. `opener._fit_images_to_budget` already recompresses an over-large request; crops
    are "smaller and fewer than the frames they came from" (doc 5.2), so nothing here resizes.
  * SCREEN IDENTITY and the SCROLL, both inherited caveats: a usable index is not "this is a
    Hinge profile", and this module is handed frames it did not capture.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from operation_love.targeting_policy import (
    HINGE_PHOTO_SELECTION_POLICY_ID, hinge_targeting_unavailable_reason)

from .item_index import (
    ITEM_CONTEXT, ITEM_LEADING_CHROME, ITEM_PARTIAL, ITEM_SELECTABLE, IndexedBlock, ItemIndex)


# =====================================================================================
# Calibration constants.
#
# Sources, same convention as segment.py / frameshift.py / item_index.py:
#   [doc]    ops/OPENER-REDESIGN.md 5.2, 5.6, 5.10.
#   [corpus] the offline run recorded in the module docstring, over the gitignored calibration
#            captures. Geometry and counts only.
# =====================================================================================

# Encoding for the crop images. PNG, lossless, deliberately: `opener._fit_images_to_budget`
# (opener.py:1169) already tries JPEG-85 recompression with no resize FIRST and doc 5.2 records
# that "that is nearly always enough", so spending quality here would spend it twice and buy
# nothing — the budget fitter cannot un-lose it. The stored crop is also what a human looks at
# when doc 5.6's verification stops a run, and an artefact in the reference is the last thing that
# investigation needs. (The SIGNATURE is unaffected either way: it is computed from the crop's
# pixels before any encoding, so no format choice here can move it.) [corpus: the 10 crops of one
# profile total 8.53MB of PNG against 37.15MB for the 24 frames they came from — 4.4x smaller, and
# inside `opener._MAX_INLINE_REQUEST_BYTES`, so on that capture the budget fitter never ran.]
_CROP_IMAGE_FORMAT = ".png"

# The (width, height) cell grid a crop is reduced to for its signature. Fixed rather than
# aspect-preserving so that every signature on a profile is comparable to every other, which is
# what a "which item is this sheet showing" nearest-match needs — the items measured 685..1144px
# tall on the one profile in hand, so an aspect-preserving grid would give them different shapes
# and nothing to compare.
#
# The SIZE is not a tuned number and the corpus says it cannot be one: separation and
# sensitivity scale together. [corpus: the nearest distinct pair measures 1.92 / 3.05 / 4.37 /
# 4.66 / 5.99 / 8.02 grey levels at grids 8 / 16 / 24 / 32 / 48 / 64, while the cost of locating
# the rect 1px off measures 0.62 / 0.96 / 1.28 / 1.47 / 1.95 / 2.44 — a ratio of 3.1..3.3 at
# every one of them.] So a finer grid buys absolute headroom against noise that nothing here
# models, and nothing else. 32x32 takes that headroom at 1KB per signature, which is negligible
# beside the ~1MB crop it summarises, and is deliberately finer than `hinge._downsample`'s 24x24
# — that grid answers "did the whole card change", this one answers "which of these nine cards
# is this", which is the harder question.
_SIGNATURE_GRID = (32, 32)


# =====================================================================================
# Result vocabulary. Plain string constants, matching the three modules below (there is no Enum
# anywhere in operation_love/) and directly loggable to the JSONL debug log.
#
# These are PAYLOAD-level classes: what happened to a block on its way to the model. They are not
# item_index.py's `ITEM_*` page classes and not segment.py's per-frame `BLOCK_*` — an
# ITEM_SELECTABLE block that policy excluded is `CROP_EXCLUDED` here, and its page class is
# unchanged and still visible on the block.
# =====================================================================================

CROP_ITEM = "item"                # cropped, sent, and NUMBERED: the model may choose it
CROP_CONTEXT = "context"          # cropped and sent, never numbered (doc 5.7: "context blocks,
                                  # cropped, unnumbered") — the vitals block is the known instance
CROP_EXCLUDED = "excluded"        # policy says do not show it; NO image is ever produced, so it
                                  # cannot be sent by accident (doc 2.4's endorsement block)
CROP_CHROME = "chrome"            # Hinge's scroll-top header, already outside both index spaces
CROP_UNCROPPABLE = "uncroppable"  # no frame ever bounded it end to end, so there is no crop that
                                  # is not a fragment: reported with its page position, not sent

# Hinge policy: the content classifier recognises photographic pixels, not still media. Neither
# low drift nor absence of Hinge's auto-hiding mute control can distinguish a still photo from a
# paused/static/low-motion video. Until a positive discriminator is calibrated, every prospective
# photo is therefore demoted to readable context. A mute match is an earlier hard exclusion, while
# this final hook blocks the otherwise indistinguishable cases. Either way the block remains in
# ItemIndex and keeps its real heart ordinal, so ambiguity cannot manufacture a photo ordinal or
# shift navigation space.
# The identifier is recorded by targeting calibration so a geometry bound cannot silently outlive
# the classifier/policy that decided which physical hearts its item numbers can name.
PHOTO_ONLY_POLICY_ID = HINGE_PHOTO_SELECTION_POLICY_ID
EXCLUSION_NON_PHOTO = "photo_only"
# Largest re-observation drift measured for a genuinely static photo across both calibration
# profiles (32x32 greyscale signature, 0..255 mean absolute difference). It is useful rejection
# evidence above the ceiling, but deliberately grants no acceptance below it: the corpus contains
# no video false-accept distribution, and a paused video can have zero drift.
_STILL_PHOTO_MAX_SIGNATURE_DRIFT = 0.24

# --- Hinge's autoplay trigger zone (owner-provided domain fact, 2026-08-21) -----------------
# Hinge autoplays a video card ONLY while that card sits at or near the CENTRE of the screen.
# That single fact inverts what a motionless dwell is worth: an off-centre video does not play,
# holds byte-exact for as long as you look at it, and would be accepted as a still photo by C2.
# So "no motion over time" is evidence of "not a video" ONLY for a card that sat inside the
# trigger zone for the whole burst, and every producer of dwell evidence has to prove it did.
#
# THE EXACT TRIGGER ZONE IS UNMEASURED.  Nobody has instrumented where Hinge starts and stops
# playback, so this is deliberately CONSERVATIVE rather than fitted: hold the card as close to
# the centre of the content band as the scroll step can put it, and require the card's centre to
# be within this fraction of the content band's height of that centre.  Widening it without a
# measurement would be widening the population that can be accepted on evidence that does not
# cover it.  One definition, imported by the driver, the offline replay and the campaign tool
# alike -- a second copy is how the producer and the consumer of this proof drift apart.
STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC = 0.15


def content_band_center_row(frame_height: int, content_band: Sequence[float]) -> float:
    """The row the autoplay trigger zone is measured around: the content band's own centre."""
    y0, y1 = float(content_band[0]) * frame_height, float(content_band[1]) * frame_height
    return (y0 + y1) / 2.0


def card_center_offset_frac(rect: tuple[int, int, int, int], *, frame_height: int,
                            content_band: Sequence[float]) -> float:
    """Signed |card centre - content centre| as a fraction of the content band's height.

    Positive means the card sits BELOW the centre (a forward scroll brings it up).  Reported as a
    fraction so one constant governs every screen size, and signed so a caller can size the
    gesture that would fix it instead of only knowing that it is wrong.
    """
    if frame_height <= 0:
        raise ItemCropError(f"frame height must be positive (got {frame_height!r})")
    band_height = (float(content_band[1]) - float(content_band[0])) * frame_height
    if band_height <= 0:
        raise ItemCropError(f"content_band must be an ordered band (got {content_band!r})")
    card_center = (rect[1] + rect[3]) / 2.0
    return (card_center - content_band_center_row(frame_height, content_band)) / band_height


def card_is_centered(rect: tuple[int, int, int, int], *, frame_height: int,
                     content_band: Sequence[float],
                     band_frac: float = STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC) -> bool:
    """Whether this card sat inside the conservative autoplay trigger zone."""
    return abs(card_center_offset_frac(
        rect, frame_height=frame_height, content_band=content_band)) <= band_frac


# The reason string doc 2.4's block is dropped under, named so the eventual detector, the debug
# log and any operator override all spell it the same way. There is deliberately NO detector for
# it here — see the module docstring.
EXCLUSION_ENDORSEMENT = "endorsement"

# Rung (g)'s refusal when no probe has run YET. Exported as a constant, not just returned as
# prose, because a producer that takes the two bursts IN SEQUENCE has to tell "every cheaper
# rung passed, go and spend the probe's real gestures" from "this card already failed
# something" -- and it must do that by identity rather than by reading the sentence.
EXCLUSION_REATTACH_PROBE_MISSING = (
    f"{EXCLUSION_NON_PHOTO}: no re-attach probe scrolled this card out of Hinge's autoplay band "
    "and back into it, so a video that was stalled, buffering, unloaded or already ended during "
    "the dwell would have held byte-exact and passed as a photograph")


class ItemCropError(RuntimeError):
    """The payload could not be built AT ALL: an unusable index, a frame list that does not match
    the index it is paired with, missing cv2/numpy, or bytes that will not decode or encode.

    Distinct from an UNUSABLE payload, which is a result carrying its evidence in
    `ItemPayload.failures` — that means the crops were produced and something about them
    contradicts the request (an over-tall block, nothing to choose from). This exception means
    there was nothing to crop or no way to crop it.

    Also raised by `ItemPayload.item` / `signature_for` on an unusable payload or an out-of-range
    number, on exactly `ItemIndex.block_for`'s reasoning: doc 5.6 says we never substitute a
    different item, and returning the nearest one — or None for a caller to handle however it
    likes — is how a substitution gets made.
    """


def _require_vision():
    """Import cv2 + numpy or raise. Same fail-loud contract as `segment._require_vision` and
    `frameshift._require_vision`, and duplicated rather than shared for the same reason those two
    are: it is six lines, and the message names the operation that could not run."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001 — surfaced as ItemCropError, never swallowed
        raise ItemCropError(
            "cropping needs opencv-python + numpy (extra: operation-love[hinge]); "
            f"import failed: {exc}") from exc
    return cv2, np


@dataclass(frozen=True)
class CropSignature:
    """A crop reduced to a fixed grid of grey cells: doc 5.6's post-tap verification reference.

    `cells` is the `grid` downsample row-major as raw uint8 bytes, `width`/`height` are the crop's
    real pixel extent (kept because a signature normalises scale away and the extent is itself a
    discriminator — a 215px vitals block and a 1000px photo card are not the same item however
    their cells compare).

    Comparison is `distance`, a mean absolute difference in grey levels, which is the same metric
    `hinge._band_dist` already uses for every pixel-signature decision in the driver. It is
    implemented over the bytes in pure Python on purpose: a stored signature is comparable in the
    hub, in a log replay or in a test, none of which have the vision extras installed.
    """
    width: int
    height: int
    grid: tuple[int, int]
    cells: bytes

    @property
    def digest(self) -> str:
        """sha256 of the cells: exact identity, for logging and for spotting the case where two
        crops are byte-identical (which on one profile means the same card was cropped twice)."""
        return hashlib.sha256(self.cells).hexdigest()

    def distance(self, other: "CropSignature") -> float:
        """Mean absolute difference in grey levels, 0..255. 0 is byte-identical.

        Raises on a grid mismatch rather than resampling one side to the other: two signatures at
        different grids are not two measurements of the same thing, and quietly rescaling one
        would return a number that looks like a distance and is not.
        """
        if self.grid != other.grid:
            raise ItemCropError(
                f"cannot compare a {self.grid[0]}x{self.grid[1]} signature with a "
                f"{other.grid[0]}x{other.grid[1]} one — resampling one to the other would return "
                "a number that is not a distance")
        return sum(
            abs(a - b) for a, b in zip(self.cells, other.cells, strict=True)
        ) / len(self.cells)


@dataclass(frozen=True)
class ItemCrop:
    """One block's fate on the way to the model: the image if it is sent, the reason if not.

    `number` is the dense 1..N the MODEL sees and is set only on `CROP_ITEM`. `heart_ordinal` is
    the driver's private one, copied unchanged from the block — it is present on every entry that
    has a heart, including an EXCLUDED one, because navigation counts hearts and an excluded
    block's heart is still on the page (doc 5.3's index trap).

    `frame_index` / `frame_y0` / `frame_y1` record which frame the image was cut from and where,
    so a debug pass can go back to the capture and see the crop in context. `page_y0`/`page_y1`
    are the page rows of THE IMAGE, i.e. the chosen sighting's — within `item_index`'s
    `_EXTENT_TOLERANCE_PX` of the block's own resolved extent by construction, since every
    complete sighting of a block has to agree with the resolved one to survive indexing.

    `image` is None for everything not sent, which is the structural half of "excluded blocks are
    not sent at all": there are no bytes to send.

    `page_y0`/`page_y1` here are THE authoritative rect for doc 5.6, in preference to the
    `IndexedBlock`'s: the block's extent is the median complete sighting's and the crop's is the
    chosen sighting's, and the two may differ by up to `item_index._EXTENT_TOLERANCE_PX`
    (measured maximum 1px over the calibration corpus). The image is what verification compares
    against, so the image's own rows are the honest ones — and a 1px rect error costs 1.47 grey
    levels against separations that can be as small as 4.66.

    `signature_drift`, `drift_frames` and `nearest_item_distance` are the evidence for whether
    this crop can serve as doc 5.6's reference at all; see `_signature_drift`, `separable`, and
    the module docstring's separability section.
    """
    kind: str
    number: int | None
    heart_ordinal: int | None
    page_y0: int
    page_y1: int
    x0: int
    x1: int
    frame_index: int | None
    frame_y0: int | None
    frame_y1: int | None
    image: bytes | None
    signature: CropSignature | None
    signature_drift: float | None
    drift_frames: tuple[int, ...]
    nearest_item_distance: float | None
    reason: str

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.page_y1 - self.page_y0

    @property
    def sent(self) -> bool:
        """Whether this entry contributes an image to the request. Exactly `image is not None`."""
        return self.image is not None

    @property
    def separable(self) -> bool | None:
        """Whether this item's own re-observation noise is smaller than its distance to the
        nearest OTHER numbered item — i.e. whether a nearest-signature match can tell it apart
        from the items it shares a page with.

        THREE-VALUED ON PURPOSE, and None is not a soft False. None means the question was not
        answered: either no other frame contained this crop's rect, so `signature_drift` was
        never measured, or there is no other item to be confused with. Reporting an unmeasured
        crop as separable is precisely the silent substitution doc 5.6 forbids, so the caller has
        to handle "unknown" explicitly rather than receive a bool that quietly means it.

        What True does NOT license: this is a crop-against-crop comparison, and doc 5.6's real
        comparison is crop-against-like-sheet, which no capture in the calibration corpus
        contains. It is a necessary condition for verification, not a sufficient one.
        """
        if self.signature_drift is None or self.nearest_item_distance is None:
            return None
        return self.signature_drift < self.nearest_item_distance


@dataclass(frozen=True)
class ItemPayload:
    """Everything the model is shown, in the order it is shown, plus what was withheld and why.

    `crops` holds an entry for EVERY block of the index, sent or not, in page order — a payload
    that silently forgot the blocks it dropped could not be audited, and doc 2.4's exclusion is a
    decision worth being able to see in a log.

    Read `usable` FIRST, exactly as with `ItemIndex`: it is the same hard gate (doc 5.3, "treat a
    missing table as a hard stop, never as a reason to fall back to a fixed coordinate"), and the
    accessors raise rather than answer when it is False.

    `usable` is about the crops, not about whether they can be VERIFIED later. Separability is a
    second, independent question, reported through `min_item_separation`,
    `max_signature_drift`, `unseparable_items` and `undetermined_items` and never folded into
    `usable` — an animated card compromises its own reference and nothing else, and refusing the
    payload would throw away eight sound references to punish the ninth. Doc 5.6 is the layer
    that decides what an unverifiable item may be used for.
    """
    crops: tuple[ItemCrop, ...]
    truncated: bool
    at_scroll_top: bool
    signature_grid: tuple[int, int]
    failures: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """Nothing about these crops contradicts the request. False is a hard stop."""
        return not self.failures

    @property
    def items(self) -> tuple[ItemCrop, ...]:
        """The numbered items, in model order: `items[k - 1]` is the model's item k."""
        return tuple(c for c in self.crops if c.kind == CROP_ITEM)

    @property
    def context(self) -> tuple[ItemCrop, ...]:
        """The unnumbered context crops, in page order (doc 5.7)."""
        return tuple(c for c in self.crops if c.kind == CROP_CONTEXT)

    @property
    def excluded(self) -> tuple[ItemCrop, ...]:
        """Blocks policy withheld, with the reason each was withheld under."""
        return tuple(c for c in self.crops if c.kind == CROP_EXCLUDED)

    @property
    def uncroppable(self) -> tuple[ItemCrop, ...]:
        """Blocks no frame ever bounded end to end. Not a failure on its own — it is the coverage
        cost of a capture that stopped early, which `truncated` already reports — but it is never
        silent: a crop of one of these would be a fragment."""
        return tuple(c for c in self.crops if c.kind == CROP_UNCROPPABLE)

    @property
    def images(self) -> tuple[bytes, ...]:
        """THE REQUEST'S IMAGE LIST: the numbered items in model order, then the context crops.

        This is the whole of what goes to the model, and doc 5.2 is explicit that full frames do
        not go with it ("Not sent: full screenshots, scroll frames, the anchor, endorsement
        blocks", doc 5.7) — this module holds no frame bytes for a caller to append.
        """
        return tuple(c.image for c in self.items + self.context if c.image is not None)

    @property
    def item_count(self) -> int:
        """N, the size of the list the model is offered. It may only answer with 1..N."""
        return len(self.items)

    @property
    def context_count(self) -> int:
        return len(self.context)

    @property
    def translation(self) -> tuple[int, ...]:
        """Model number -> heart ordinal, `translation[k - 1]` for item k. Doc 5.3's private
        table, and THE authoritative one: it is `ItemIndex.translation` renumbered over the items
        actually sent, so it is the copy that stays correct when policy excludes something
        selectable. With no exclusions the two are equal."""
        return tuple(c.heart_ordinal for c in self.items)

    def item(self, number: int) -> ItemCrop:
        """The model's item `number`, 1-based. Raises rather than returning None or the nearest
        one, on `ItemIndex.block_for`'s reasoning (doc 5.6: never substitute a different item)."""
        if not self.usable:
            raise ItemCropError(
                "this payload is unusable and carries no valid item numbers: "
                + "; ".join(self.failures))
        items = self.items
        if not 1 <= number <= len(items):
            raise ItemCropError(
                f"item number {number} is outside 1..{len(items)}; the model was shown "
                f"{len(items)} item(s) and may not be answered with anything else")
        return items[number - 1]

    def signature_for(self, number: int) -> CropSignature:
        """Item `number`'s verification reference (doc 5.6). Same raising contract as `item`, and
        always a signature: a `CROP_ITEM` entry has an image by construction.

        A signature is not on its own a usable reference. Check `item(number).separable` before
        trusting a match against it — an animated card's own re-observation noise can exceed the
        distance to a different item, measured on a real profile at 25.0 against 6.27."""
        return self.item(number).signature

    @property
    def min_item_separation(self) -> float | None:
        """The smallest signature distance between two DIFFERENT numbered items of this profile:
        how far apart the two most alike things the model may choose actually are.

        Measured per payload rather than shipped as a constant, and that is the fix for a real
        defect: this module's docstring used to quote one profile's 4.66 as though it were a
        property of Hinge. A second profile measures 6.27, and the drift it has to be compared
        against moves by an order of magnitude between them. None when there are fewer than two
        items, where the quantity does not exist.

        A profile-wide summary, and deliberately NOT the number `separable` uses: the closest
        pair on both calibration profiles is items 5 and 7, which tells you nothing about whether
        item 9's reference is trustworthy. Per item, `nearest_item_distance` is the one that
        governs, and the two answers differ — 25.0 against 6.27 inverts, 25.0 against item 9's
        own 47.9 does not.
        """
        return min((c.nearest_item_distance for c in self.items
                    if c.nearest_item_distance is not None), default=None)

    @property
    def max_signature_drift(self) -> float | None:
        """The worst re-observation noise measured on any numbered item, in grey levels. None
        when no item could be measured at all — which is NOT zero drift, it is no evidence."""
        return max((c.signature_drift for c in self.items
                    if c.signature_drift is not None), default=None)

    @property
    def unseparable_items(self) -> tuple[int, ...]:
        """Item numbers whose stored crop is a proven-unusable verification reference: their own
        re-observation noise meets or exceeds their distance to another item, so a nearest-match
        cannot tell them apart. Doc 5.6 must not tap these on the strength of a signature check.
        Distinct from `undetermined_items`, which is ignorance rather than a negative result."""
        return tuple(c.number for c in self.items if c.separable is False)

    @property
    def undetermined_items(self) -> tuple[int, ...]:
        """Item numbers whose reference could not be evaluated: nothing re-observed the crop's
        rect, so its stability is unknown. A capture that never revisits a scroll position
        measures nothing here, and silence must not be read as stability."""
        return tuple(c.number for c in self.items if c.separable is None)


def _pairs(items: Sequence):
    """Every unordered pair, in a stable order. `itertools.combinations` in one place so the
    separability numbers are reproducible run to run rather than set-ordered."""
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            yield a, b


def _signature_from_gray(gray, *, grid: tuple[int, int], cv2, np) -> CropSignature:
    """A greyscale array -> its `CropSignature`. The single definition of what a signature IS.

    `cv2.INTER_AREA` is the resize: it averages over each destination cell's source area, which is
    the correct filter for downscaling and, unlike an interpolating one, cannot let a single
    bright pixel decide a cell.
    """
    cells = cv2.resize(gray, grid, interpolation=cv2.INTER_AREA)
    return CropSignature(
        width=int(gray.shape[1]), height=int(gray.shape[0]), grid=grid,
        cells=np.ascontiguousarray(cells, dtype=np.uint8).tobytes())


def signature_of(frame: bytes, *, y0: int, y1: int, x0: int, x1: int,
                 grid: tuple[int, int] = _SIGNATURE_GRID) -> CropSignature:
    """The signature of rows `y0..y1`, columns `x0..x1` of `frame` — the OTHER half of doc 5.6.

    A stored crop signature is only useful against a signature computed the same way, and this is
    the function that computes it, so that a post-tap verification pass (5.6) or observe mode's
    pre-type check (5.9) never has to reproduce the decode. That is not tidiness: the two obvious
    ways to get a greyscale array out of a PNG disagree by enough to change the answer. [corpus:
    `cv2.IMREAD_GRAYSCALE` versus decoding colour and converting with `cvtColor(BGR2GRAY)` differ
    by 1.46 grey levels on average and 40 at the worst pixel on one real frame, and by up to 9.8
    on a 32x32 crop signature — where 4.66 is the whole distance between the two most alike items
    on that profile.] `build_item_payload` computes its stored signatures through this same
    helper, so the two sides agree by construction rather than by both remembering.

    Raises `ItemCropError` for undecodable bytes or a rect that is not inside the frame — a rect
    clipped to fit would silently describe a different region than the one asked about.
    """
    cv2, np = _require_vision()
    gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ItemCropError(f"frame bytes did not decode as an image ({len(frame)} bytes)")
    height, width = gray.shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ItemCropError(
            f"rect rows {y0}..{y1}, columns {x0}..{x1} does not fit inside the {width}x{height} "
            "frame — clipping it would return the signature of a different region")
    return _signature_from_gray(gray[y0:y1, x0:x1], grid=grid, cv2=cv2, np=np)


def exclude_page_rows(spans: Sequence[tuple[int, int]], *, reason: str = EXCLUSION_ENDORSEMENT,
                      ) -> Callable[[IndexedBlock], str | None]:
    """An `exclude` predicate that drops every block overlapping one of `spans` (page rows,
    half-open, in the index's page space).

    This is the exclusion mechanism with no detector attached: it takes the answer from the
    caller. It is what a validation pass, an operator override or a future detector's output can
    all be expressed through, and it is deliberately positional rather than geometric — doc 5.10
    measured that the endorsement block did not appear in either calibration capture, so there is
    nothing here to fit a shape to.
    """
    windows = tuple((int(y0), int(y1)) for y0, y1 in spans)

    def _exclude(block: IndexedBlock) -> str | None:
        for y0, y1 in windows:
            if block.page_y0 < y1 and y0 < block.page_y1:
                return (f"{reason}: the caller excluded page rows {y0}..{y1}, which this block "
                        f"at {block.page_y0}..{block.page_y1} overlaps")
        return None

    return _exclude


def unnumber_unless_confident_photo(image: bytes) -> str | None:
    """Number only a crop affirmatively classified as a Hinge photograph.

    Card aspect ratio is deliberately not evidence of item type.  The calibration profile happened
    to contain square photos and shorter written prompts, but a later real profile showed a
    landscape first photo (974x695) and a square final photo.  Treating that incidental geometry
    as a hard gate demoted the earlier photos and renumbered the final one as model item 1.

    ``classify_crop`` uses content variation, photo-region structure and text-card layout rather
    than card geometry. UNKNOWN is not a weak PHOTO verdict: numbering it was the root cause of
    a seven-item model list for a profile whose extra candidate was a prompt. WRITTEN and UNKNOWN
    crops become readable, unnumbered context while retaining their heart ordinal; only PHOTO
    crops receive dense model/photo numbers.
    """
    from .item_type_preflight import PHOTO, classify_crop

    item_type = classify_crop(image)
    if item_type == PHOTO:
        return None
    return (f"{EXCLUSION_NON_PHOTO}: crop classified as {item_type}; only confidently "
            "photographic cards may be numbered, while ambiguous or written cards remain "
            "readable context")


@dataclass(frozen=True)
class StillPhotoDwell:
    """What one un-interacted dwell burst observed about ONE candidate card.

    Separate from `StillPhotoEvidence` because the two halves are measured by different layers:
    drift comes from the enumeration frames this module already holds, while the dwell is
    screencaps the DRIVER takes with no input after the read (ops/STILL-PHOTO-DISCRIMINATOR.md
    C2/C3). Keeping the dwell in its own object is what lets `build_item_payload` accept it as
    one optional per-card mapping instead of growing four parallel arguments.

    Every field is three-valued on purpose: None means the observation was never made, which is
    a refusal here and must never read as a passing measurement.
    """

    dwell_frame_sha256s: tuple[str, ...] = ()
    dwell_exact: bool | None = None
    dwell_span_s: float | None = None
    mute_screens_complete: bool | None = None
    # None, not False, by default: a producer that never measured where the card sat has made no
    # observation, and every existing caller must land on the refusal rather than inherit a pass.
    centered: bool | None = None
    center_offset_frac: float | None = None
    # The re-attach probe's second burst (2026-08-21). A centred byte-exact run proves Hinge was
    # ASKED to play this card and nothing moved; it cannot prove the media ANSWERED, so a video
    # that was stalled, buffering, never attached or ended without looping holds byte-exact for
    # as long as anyone looks. The probe takes the card out of the autoplay band and puts it
    # back, which is the event Hinge re-attaches media on, then dwells again. Defaulting every
    # field to None keeps every producer that has not been taught the probe on the refusal.
    reattach_probe_ran: bool | None = None
    reattach_dwell_frame_sha256s: tuple[str, ...] = ()
    reattach_dwell_exact: bool | None = None
    reattach_dwell_span_s: float | None = None
    reattach_mute_screens_complete: bool | None = None
    reattach_centered: bool | None = None
    reattach_center_offset_frac: float | None = None


@dataclass(frozen=True)
class StillPhotoEvidence:
    """Everything the numbering gate is allowed to consider about one photographic crop.

    It used to be two positional arguments (drift and the frames that measured it), which is
    exactly the shape that made "drift is low" look like proof. Naming the object EVIDENCE and
    giving the dwell legs their own three-valued fields makes the missing observation visible at
    every call site: a caller that cannot dwell constructs one with `dwell_exact=None` and is
    refused for that reason rather than silently graded on drift alone.

      * `signature_drift` / `drift_frames` -- C1's re-observation leg, in 32x32 greyscale mean
        absolute difference and the OTHER enumeration frames that produced it.
      * `dwell_frame_sha256s` -- C2's frames, in capture order, so a log or a review can bind the
        verdict to the exact bytes it was computed from.
      * `dwell_exact` -- True only when EVERY consecutive dwell frame pair was byte-identical
        over this candidate's card rect. False is a measured refusal; None is no dwell at all.
      * `dwell_span_s` -- the wall-clock span the byte-exact run covers. Without it the run has
        no duration to compare against the artifact's worst held-out video exact run, so the
        bound that licenses the whole rule cannot be applied.
      * `mute_screens_complete` -- True only when C3's screen RAN with a geometrically complete
        ROI and a successful decode on every dwell frame and matched nothing.
    """

    signature_drift: float | None
    drift_frames: tuple[int, ...]
    dwell_frame_sha256s: tuple[str, ...] = ()
    dwell_exact: bool | None = None
    dwell_span_s: float | None = None
    mute_screens_complete: bool | None = None
    # `centered` -- True only when the card's rect sat inside the autoplay trigger zone for the
    # WHOLE burst. Hinge only plays a video near the centre of the screen, so a byte-exact run
    # measured off-centre is consistent with a video that was simply never asked to play.
    centered: bool | None = None
    center_offset_frac: float | None = None
    # `reattach_*` -- the SECOND burst, taken after the card was scrolled out of the autoplay
    # band and back into it. Same three-valued rule as every field above: None means the probe
    # never ran, which is a refusal, never an exemption.
    reattach_probe_ran: bool | None = None
    reattach_dwell_frame_sha256s: tuple[str, ...] = ()
    reattach_dwell_exact: bool | None = None
    reattach_dwell_span_s: float | None = None
    reattach_mute_screens_complete: bool | None = None
    reattach_centered: bool | None = None
    reattach_center_offset_frac: float | None = None


def still_photo_evidence_from_drift(
        signature_drift: float | None, drift_frames: tuple[int, ...],
        dwell: StillPhotoDwell | None = None) -> StillPhotoEvidence:
    """Build the gate's evidence object from a drift measurement and an optional dwell.

    Source compatibility for every caller that has enumeration frames but cannot dwell (offline
    replay, a generic non-Hinge caller, the payload builder's own default): they keep passing the
    two numbers they have and get an object whose dwell fields are None -- which is a REFUSAL the
    moment a verified bound is installed, not an exemption. That is the intended behaviour: a
    rebuild that never held the screen still has not made C2's observation.
    """
    dwell = dwell or StillPhotoDwell()
    return StillPhotoEvidence(
        signature_drift=signature_drift, drift_frames=drift_frames,
        dwell_frame_sha256s=dwell.dwell_frame_sha256s, dwell_exact=dwell.dwell_exact,
        dwell_span_s=dwell.dwell_span_s, mute_screens_complete=dwell.mute_screens_complete,
        centered=dwell.centered, center_offset_frac=dwell.center_offset_frac,
        reattach_probe_ran=dwell.reattach_probe_ran,
        reattach_dwell_frame_sha256s=dwell.reattach_dwell_frame_sha256s,
        reattach_dwell_exact=dwell.reattach_dwell_exact,
        reattach_dwell_span_s=dwell.reattach_dwell_span_s,
        reattach_mute_screens_complete=dwell.reattach_mute_screens_complete,
        reattach_centered=dwell.reattach_centered,
        reattach_center_offset_frac=dwell.reattach_center_offset_frac)


def dwell_exact_over_rect(frames_rgb: Sequence, rect: tuple[int, int, int, int]) -> bool:
    """True only if `rect` is byte-identical in EVERY consecutive pair of these frames.

    Zero tolerance, full resolution, no downsample and no distance metric: the whole point of
    C2 is that it is not a threshold anybody can tune. [corpus: a real autoplaying video's
    signature strips match CONFIDENTLY WRONG rather than weakly, so a score-based test goes
    bimodal on exactly the population it has to catch -- pixel-exactness is the only
    discriminator that does not.]

    Accepts encoded frame bytes (the house currency of this module) or already-decoded arrays,
    so the driver and an offline replay of persisted PNGs call the same function. Fewer than two
    frames, a rect outside a frame, or bytes that will not decode all return False: an
    observation that could not be made is not a passing one. A structurally invalid `rect` is a
    CALLER bug rather than a failed observation and raises.
    """
    if (not isinstance(rect, tuple) or len(rect) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in rect)
            or rect[2] <= rect[0] or rect[3] <= rect[1] or rect[0] < 0 or rect[1] < 0):
        raise ItemCropError(
            f"dwell rect must be a (x0, y0, x1, y1) tuple of ints with positive extent (got {rect!r})")
    frames = list(frames_rgb)
    if len(frames) < 2:
        return False
    x0, y0, x1, y1 = rect
    previous = None
    for frame in frames:
        if isinstance(frame, (bytes, bytearray)):
            cv2, np = _require_vision()
            image = cv2.imdecode(np.frombuffer(bytes(frame), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return False
        else:
            image = frame
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2 or y1 > shape[0] or x1 > shape[1]:
            return False
        current = image[y0:y1, x0:x1].tobytes()
        if previous is not None and current != previous:
            return False
        previous = current
    return True


def still_photo_dwell_evidence(index: ItemIndex, frames: Sequence[bytes],
                              dwell_frames: Sequence[bytes], *, dwell_span_s: float | None,
                              mute_screen: Callable[[bytes, tuple[int, int, int, int]], bool]
                              | None = None,
                              content_band: Sequence[float] | None = None,
                              ) -> dict[int, StillPhotoDwell]:
    """Per-heart-ordinal dwell evidence for the candidates a no-input dwell could observe.

    Pure, so the live driver and an offline replay of persisted dwell PNGs derive the SAME rect
    the same way rather than each measuring their own. `frames` is the indexed capture, whose
    LAST frame is the one still on screen when a dwell begins (see
    `HingeDriver._index_captured_items` on why that is true by construction), so it is the anchor
    the burst is chained to and the frame every rect is read from.

    Only cards with a COMPLETE sighting in that anchor frame get an entry. A card that scrolled
    out of view before the read ended cannot be held still and looked at, so it has no C2
    observation and its absence from this mapping is the refusal -- never an omission to paper
    over with the last rect it happened to occupy.

    `mute_screen(frame, rect) -> bool` is C3: True only when the screen ran with a complete ROI
    and a successful decode and matched nothing. Omitting it leaves `mute_screens_complete` None,
    which refuses.

    `content_band` is what makes the AUTOPLAY precondition measurable: Hinge plays a video only
    near the centre of the screen, so a byte-exact run is evidence of stillness only for a card
    that sat in the trigger zone. Omitting it leaves `centered` None, which refuses -- the
    fail-closed default for every caller that has not been taught where the card was.

    This is the FIRST burst only. Its `reattach_*` fields stay None, which refuses at the probe
    rung; `still_photo_reattach_evidence` is what folds the second burst in once a probe has
    actually taken the card out of the autoplay band and put it back.
    """
    anchor_position = len(frames) - 1
    if anchor_position < 0 or not dwell_frames:
        return {}
    sequence = [frames[anchor_position], *dwell_frames]
    frame_height = None if content_band is None else decoded_frame_height(sequence[0])
    digests = tuple(hashlib.sha256(frame).hexdigest() for frame in sequence)
    evidence: dict[int, StillPhotoDwell] = {}
    for heart_ordinal, rect in dwell_card_rects(index, anchor_position).items():
        offset = None
        centered = None
        if frame_height:
            offset = card_center_offset_frac(rect, frame_height=frame_height,
                                             content_band=content_band)
            centered = abs(offset) <= STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC
        evidence[heart_ordinal] = StillPhotoDwell(
            dwell_frame_sha256s=digests,
            dwell_exact=dwell_exact_over_rect(sequence, rect),
            dwell_span_s=dwell_span_s,
            mute_screens_complete=(None if mute_screen is None
                                   else all(bool(mute_screen(frame, rect)) for frame in sequence)),
            centered=centered,
            center_offset_frac=(None if offset is None else float(offset)))
    return evidence


@dataclass(frozen=True)
class ReattachProbe:
    """The frames one deterministic re-attach probe produced, and where the page ended up.

    The probe exists for C2's one residual: a byte-exact centred burst proves the card did not
    move, and centring proves Hinge was ASKED to play it, but neither proves the media answered.
    A video that was stalled, still buffering, never attached, or ended without looping emits no
    frame for as long as anyone looks. Taking the card OUT of the autoplay band and bringing it
    back is the event Hinge re-attaches and restarts media on, so a second burst after that
    re-entry is a second, independent chance for a non-playing video to reveal itself.

    `page_shift_px` is `frameshift.estimate_shift`'s convention measured against the FIRST
    burst's anchor: positive means the content moved UP the screen, so a rect at row `y` of that
    anchor is at row `y - page_shift_px` here. The probe carries the measured PAGE displacement
    rather than a re-segmented card, and that is the whole reason it cannot substitute a
    different card for the one it measured (owner rule 2026-08-11): the page is tracked, the
    card never re-identified.
    """

    anchor: bytes
    frames: tuple[bytes, ...]
    span_s: float | None
    page_shift_px: int


def decoded_frame_height(frame: bytes) -> int | None:
    """The pixel height of one encoded frame, or None when it will not decode.

    None is a missing observation everywhere it is used, never a default: the autoplay-centring
    precondition cannot be measured without knowing how tall the screen was.
    """
    cv2, np = _require_vision()
    image = cv2.imdecode(np.frombuffer(bytes(frame), dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else int(image.shape[0])


def dwell_card_rects(index: ItemIndex,
                     anchor_position: int) -> dict[int, tuple[int, int, int, int]]:
    """Per heart ordinal, the card rect a dwell can observe in the anchor frame.

    Only cards with exactly ONE complete sighting in that frame qualify, for the reason
    `still_photo_dwell_evidence` states: a card that scrolled out of view cannot be held still
    and looked at, so its absence here IS the refusal. Factored out because the re-attach probe
    has to translate the very same rects, and two copies of this derivation would be two things
    free to disagree about which pixels a verdict was computed over.
    """
    rects: dict[int, tuple[int, int, int, int]] = {}
    for block in index.selectable:
        if block.heart_ordinal is None:
            continue
        anchored = [obs for obs in block.croppable if obs.frame_index == anchor_position]
        if len(anchored) != 1:
            continue
        obs = anchored[0]
        rects[block.heart_ordinal] = (block.x0, obs.frame_y0, block.x1, obs.frame_y1)
    return rects


def still_photo_reattach_legs(sequence: Sequence, rect: tuple[int, int, int, int], *,
                              span_s: float | None,
                              mute_screen: Callable[[bytes, tuple[int, int, int, int]], bool]
                              | None = None,
                              frame_height: int | None = None,
                              content_band: Sequence[float] | None = None) -> dict:
    """The seven `reattach_*` fields for ONE card, measured over the probe's own frames.

    Returned as a mapping keyed by the dataclass field names so the driver, the calibration
    pre-heart proof and the offline replay all splat the SAME measurement into
    `StillPhotoDwell`/`StillPhotoEvidence` instead of each writing their own second burst by
    hand. Every leg is recomputed from bytes here; nothing is carried across as a recorded
    verdict.

    A rect that does not lie inside the frame is a card the probe pushed off the screen, which
    is a MISSING observation rather than a failed one: it reports `reattach_probe_ran=False`
    with every measurement left None, so it refuses at the probe rung and is never read as
    motion.
    """
    frames = list(sequence)
    absent = {"reattach_probe_ran": False, "reattach_dwell_frame_sha256s": (),
              "reattach_dwell_exact": None, "reattach_dwell_span_s": None,
              "reattach_mute_screens_complete": None, "reattach_centered": None,
              "reattach_center_offset_frac": None}
    if len(frames) < 2 or rect[1] < 0 or (frame_height and rect[3] > int(frame_height)):
        return absent
    offset = None
    if frame_height and content_band is not None:
        offset = card_center_offset_frac(rect, frame_height=int(frame_height),
                                         content_band=content_band)
    return {
        "reattach_probe_ran": True,
        "reattach_dwell_frame_sha256s": tuple(
            hashlib.sha256(bytes(frame)).hexdigest() for frame in frames),
        "reattach_dwell_exact": dwell_exact_over_rect(frames, rect),
        "reattach_dwell_span_s": span_s,
        "reattach_mute_screens_complete": (
            None if mute_screen is None
            else all(bool(mute_screen(frame, rect)) for frame in frames)),
        "reattach_centered": (None if offset is None
                              else abs(offset) <= STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC),
        "reattach_center_offset_frac": None if offset is None else float(offset),
    }


def still_photo_reattach_evidence(evidence: dict[int, StillPhotoDwell], index: ItemIndex, *,
                                  frame_count: int, probe: ReattachProbe,
                                  mute_screen: Callable[[bytes, tuple[int, int, int, int]], bool]
                                  | None = None,
                                  content_band: Sequence[float] | None = None,
                                  ) -> dict[int, StillPhotoDwell]:
    """Fold one probe's second burst into the first burst's evidence, per card.

    Pure, like its first-burst twin, so the live driver and an offline replay of persisted probe
    frames reach the same verdict from the same bytes. The rects come from `dwell_card_rects`
    on the SAME anchor position the first burst used and are then translated by the probe's
    measured page displacement, which is what keeps the second burst pointed at the card the
    first one measured instead of at whatever now occupies those rows.
    """
    rects = dwell_card_rects(index, frame_count - 1)
    sequence = [probe.anchor, *probe.frames]
    frame_height = None if content_band is None else decoded_frame_height(probe.anchor)
    folded: dict[int, StillPhotoDwell] = {}
    for heart_ordinal, dwell in evidence.items():
        rect = rects.get(heart_ordinal)
        if rect is None:
            folded[heart_ordinal] = dwell
            continue
        shifted = (rect[0], rect[1] - probe.page_shift_px,
                   rect[2], rect[3] - probe.page_shift_px)
        folded[heart_ordinal] = replace(dwell, **still_photo_reattach_legs(
            sequence, shifted, span_s=probe.span_s, mute_screen=mute_screen,
            frame_height=frame_height, content_band=content_band))
    return folded


def unnumber_without_still_photo_evidence(evidence: StillPhotoEvidence) -> str | None:
    """Refuse numbering unless this candidate carries the whole C1-C4 acceptance.

    The content classifier cannot distinguish one photographic video frame from a still photo,
    Hinge's mute overlay auto-hides, and a paused/static video can have arbitrarily low drift.
    Drift and mute screening can reject candidates but cannot prove the complement, so the
    positive observation is the dwell: with no input at all, the card rect byte-identical across
    every consecutive capture, screened for a mute control on each of them, over a window the
    installed bound says exceeds the worst byte-exact run any held-out video managed.

    That dwell has one residual it cannot see, and rung (g) is the answer to it: a card that is
    centred and motionless is only evidence of a photograph if the media was PLAYING while we
    watched. A stalled, buffering, unattached or ended-non-looping video emits nothing for as
    long as anybody looks. So a byte-exact centred burst buys a second look rather than an
    acceptance: the card is scrolled out of the autoplay band and back in, which is what forces
    Hinge to re-attach the media, and only a SECOND byte-exact, screened, centred burst after
    that re-entry may be numbered.

    The ladder is ordered so the CHEAP, always-available rejectors keep answering first and keep
    their exact wording: an animated card is reported as animated whether or not a bound is
    installed, and the policy blocker still speaks for a build with no artifact. Rungs (d)-(g)
    are only reachable once a bound is installed, which is why they are the last word rather
    than the first, and (g) is last of all because it is the only rung that costs real gestures.

    Drift is a REJECTOR ONLY, so its absence is not a refusal: a capture that never re-observed
    a crop's rect measured nothing, and the dwell rungs below make the positive observation
    anyway. See the comment on that rung for why refusing on silence was structurally unsatisfiable.
    """
    if not isinstance(evidence, StillPhotoEvidence):
        # Fail closed on a caller that has not been ported: a bare drift number is not evidence,
        # and guessing which argument it was would resurrect exactly the "drift proves still"
        # reading this object exists to kill.
        return (f"{EXCLUSION_NON_PHOTO}: no independent re-observation established that this "
                "photographic-looking card is stable; an auto-hidden video cannot be ruled out")
    signature_drift = evidence.signature_drift
    # AN UNMEASURABLE DRIFT FALLS THROUGH; A MEASURED MOVING CARD STILL REFUSES.  This rung used
    # to refuse whenever `signature_drift`/`drift_frames` were absent, and that was asking a
    # question a top-down read structurally CANNOT answer for the card it reaches first: drift is
    # measured by resampling a crop's page rows in the OTHER frames of the capture, and for the
    # first card no other frame's analysed band contains those rows, so `_signature_drift`
    # returns `(None, ())` by construction.  Refusing on that silence made every depth-1 target
    # (every odd ordinal) permanently unnumberable, which is not a safety property -- it is the
    # rung mistaking ignorance for a negative result, the exact confusion `undetermined_items`
    # exists to keep apart everywhere else in this module.  Nothing is given away by falling
    # through: the rungs below are a STRICTLY STRONGER independent re-observation than drift ever
    # was (a byte-exact centred dwell, screened for a mute control, then a second byte-exact
    # burst after the card was pushed out of the autoplay band and back), and an UNLICENSED build
    # still stops at the policy rung immediately below before any of them is consulted.  What is
    # NOT relaxed: a drift that WAS measured and exceeds the ceiling is a moving card and still
    # refuses, and a drift that is corrupt rather than absent -- a bool, a non-number, non-finite,
    # or negative, or a number that disagrees with its own frame list -- is refused too, because
    # a broken measurement is not a missing one.
    #
    # A MEASUREMENT AND THE FRAMES BEHIND IT STAND OR FALL TOGETHER.  "Unmeasurable" is the pair
    # `(None, ())` and nothing else: a drift carrying an EMPTY `drift_frames` claims a distance
    # computed over zero re-observations, and a frame list with no drift behind it claims
    # re-observations that produced no distance.  Both are internally INCONSISTENT -- corrupt,
    # not silent -- so they take the corrupt refusal rather than the pass-through above.  WHY
    # THIS RUNG EXISTS WITH NOTHING IN TREE FOR IT TO CATCH: `_signature_drift` and the calibrate
    # tool's `_parked_signature_drift` both return `(None, ())` or `(float, non-empty)`, so no
    # producer emits the inconsistent shape today.  But the pass-through left
    # `evidence.drift_frames` read by nothing at all, so the moment a new producer computes a
    # drift and forgets the frames behind it, this is the rung that stops the claim instead of
    # waving it through as ignorance.
    if (signature_drift is not None) != bool(evidence.drift_frames):
        return (f"{EXCLUSION_NON_PHOTO}: no independent re-observation established that this "
                "photographic-looking card is stable; an auto-hidden video cannot be ruled out")
    if signature_drift is not None:
        if (isinstance(signature_drift, bool)
                or not isinstance(signature_drift, (int, float))
                or not math.isfinite(signature_drift) or signature_drift < 0):
            return (f"{EXCLUSION_NON_PHOTO}: no independent re-observation established that this "
                    "photographic-looking card is stable; an auto-hidden video cannot be ruled out")
        if signature_drift > _STILL_PHOTO_MAX_SIGNATURE_DRIFT:
            return (f"{EXCLUSION_NON_PHOTO}: re-observation drift {signature_drift:.6g} exceeds "
                    f"the measured static-photo ceiling {_STILL_PHOTO_MAX_SIGNATURE_DRIFT:.6g}; "
                    "animated or video media is not targetable")
    policy_blocker = hinge_targeting_unavailable_reason()
    if policy_blocker is not None:
        return f"{EXCLUSION_NON_PHOTO}: {policy_blocker}; numbered targeting is disabled"
    # `is not True` rather than a truth test, here and below: an evidence object assembled with
    # a truthy placeholder must not be able to claim an observation nothing made. The digest
    # count is part of the same claim -- "byte-identical across every consecutive pair" is
    # vacuous with fewer than two frames, so a True with no frames behind it is refused too.
    if evidence.dwell_exact is not True or len(evidence.dwell_frame_sha256s) < 2:
        return (f"{EXCLUSION_NON_PHOTO}: no un-interacted dwell proved this card's rect "
                "byte-identical across consecutive captures; a video that emitted a single "
                "frame, or a dwell that never ran, cannot be excluded")
    if evidence.centered is not True:
        # Hinge autoplays a video only while the card is at or near the centre of the screen, so
        # an off-centre card that never moved has not been shown to be a photograph -- it has
        # been shown to be something nobody asked to play. Placed immediately after the
        # byte-exactness rung because it is that rung's precondition, not an extra opinion.
        return (f"{EXCLUSION_NON_PHOTO}: the un-interacted dwell did not hold this card inside "
                "Hinge's autoplay trigger zone (card centre offset "
                f"{'unmeasured' if evidence.center_offset_frac is None else format(evidence.center_offset_frac, '.3f')}"
                f", limit {STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC:.3f}), so its stillness is "
                "equally consistent with a video that was never near enough to the centre to "
                "start playing")
    if evidence.mute_screens_complete is not True:
        return (f"{EXCLUSION_NON_PHOTO}: mute-control screening did not complete over a full ROI "
                "on every dwell frame, so a control that rendered during the dwell could have "
                "gone unseen")
    span = evidence.dwell_span_s
    if (span is None or isinstance(span, bool) or not isinstance(span, (int, float))
            or not math.isfinite(span) or span <= 0):
        return (f"{EXCLUSION_NON_PHOTO}: the dwell window is missing or non-positive "
                f"({span!r}), so the byte-exact run has no measured duration to hold against "
                "the bound that licenses accepting it")
    # (g) THE RE-ATTACH PROBE. Last because it is the most expensive observation and the newest:
    # everything above answers from frames the read already had, while this rung requires two
    # real gestures and a second burst. It exists for the one residual the rungs above cannot
    # touch. Centring proves Hinge was ASKED to play this card; a byte-exact run then proves
    # nothing moved. It does NOT prove the media answered, so a video that was stalled,
    # buffering, never attached, or ended without looping sits there emitting no frame and looks
    # exactly like a photograph for as long as anybody watches. Leaving the autoplay band and
    # re-entering it is the event Hinge re-attaches and restarts media on, so the second burst
    # is a fresh chance for a non-playing video to say so.
    if evidence.reattach_probe_ran is not True:
        return EXCLUSION_REATTACH_PROBE_MISSING
    if (evidence.reattach_dwell_exact is not True
            or len(evidence.reattach_dwell_frame_sha256s) < 2):
        return (f"{EXCLUSION_NON_PHOTO}: the second dwell burst, taken after this card "
                "re-entered Hinge's autoplay band, did not prove its rect byte-identical "
                "across consecutive captures; media that starts on re-attach is video, and a "
                "second burst that never ran has not retired the stalled-video residual")
    if evidence.reattach_centered is not True:
        return (f"{EXCLUSION_NON_PHOTO}: the re-attach probe did not put this card back inside "
                "Hinge's autoplay trigger zone (card centre offset "
                f"{'unmeasured' if evidence.reattach_center_offset_frac is None else format(evidence.reattach_center_offset_frac, '.3f')}"
                f", limit {STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC:.3f}), so the second burst "
                "never asked the media to restart and cannot exclude a stalled video")
    if evidence.reattach_mute_screens_complete is not True:
        return (f"{EXCLUSION_NON_PHOTO}: mute-control screening did not complete over a full ROI "
                "on every re-attach burst frame, so a control Hinge redrew when the card came "
                "back into the autoplay band could have gone unseen")
    reattach_span = evidence.reattach_dwell_span_s
    if (reattach_span is None or isinstance(reattach_span, bool)
            or not isinstance(reattach_span, (int, float))
            or not math.isfinite(reattach_span) or reattach_span <= 0):
        return (f"{EXCLUSION_NON_PHOTO}: the re-attach dwell window is missing or non-positive "
                f"({reattach_span!r}), so the second byte-exact run has no measured duration to "
                "hold against the bound that licenses accepting it")
    return None


def _band_height(index: ItemIndex) -> int:
    """The analysed band's height in rows — the tallest thing any single frame can contain.

    Read off the segmentation rather than declared, so it follows `content_band` (and any
    operator override of it) instead of being a second copy free to drift. Every frame of one
    capture shares a band, so frame 0's is the capture's.
    """
    r0, r1 = index.frames[0].band
    return r1 - r0


def _over_tall_failures(index: ItemIndex, band_height: int) -> list[str]:
    """Blocks that no single frame could ever contain, whatever we scroll.

    Doc 5.10 closed the viewport question on a measurement — "The tallest card fully observed is
    1114px against an 1800px content band, 62% of the budget... cropping can work from single
    frames and no stitching is required" — and immediately added the caveat this function is:
    "one profile. A very long prompt answer could still exceed it, so the crop path should detect
    the case rather than assume it away."

    A block taller than the band is that case, and it is a hard failure rather than a taller crop
    assembled from two frames: stitching is not implemented, and quietly stitching would produce
    an image whose seam is a scroll artefact, sent to the model AND stored as doc 5.6's
    verification reference.

    In practice this can only fire on an `ITEM_PARTIAL` block, since a complete one was by
    definition contained by a frame; it is applied to every block anyway, because a complete
    block measuring taller than the band would be a contradiction worth hearing about rather than
    a case to skip. `ITEM_LEADING_CHROME` is exempt: it is not an item, it is never cropped, and
    its hull is whatever Hinge's header happened to look like above the topmost card corner.
    """
    failures: list[str] = []
    for block in index.blocks:
        if block.kind == ITEM_LEADING_CHROME or block.height <= band_height:
            continue
        failures.append(
            f"the block at page rows {block.page_y0}..{block.page_y1} is {block.height}px tall, "
            f"taller than the {band_height}px analysed band, so no single frame can contain it "
            "— doc 5.10 measured the tallest card end to end at 1114px against an 1800px band "
            "and single-frame crops were sized on that; stitching two frames is deliberately not "
            "implemented, because the seam would go to the model and into the post-tap "
            "verification reference")
    return failures


def _choose_sighting(block: IndexedBlock, index: ItemIndex):
    """The frame this block is cropped from: the one that shows it most completely.

    Only `croppable` sightings are candidates — those that observed BOTH edges — so every
    candidate already contains the whole card and "most completely" is a tie among them. The
    tie is broken by CLEARANCE: how many rows separate the card from the nearer edge of that
    frame's analysed band. Larger is better, and the reason is where the band's edges are: the
    band is cut to exclude the sticky header above and the bottom nav below, so a card pressed
    against either edge is a card sitting under the chrome those edges were drawn to avoid, and
    Hinge's floating pass/like buttons live there too (segment.py's trap 2 measured one of them
    spanning x=30..216 across a gutter). A card in the middle of the band is the one least likely
    to have anything of ours painted over it.

    Ties on clearance go to the LOWEST frame index, so the choice is fully deterministic and a
    re-run of the same capture produces byte-identical crops — which matters because the crop is
    stored as doc 5.6's verification reference.

    Returns None when there is no croppable sighting at all; the caller reports that rather than
    falling back to a fragment.
    """
    best = None
    best_key: tuple[int, int] | None = None
    for obs in block.croppable:
        r0, r1 = index.frames[obs.frame_index].band
        clearance = min(obs.frame_y0 - r0, r1 - obs.frame_y1)
        key = (-clearance, obs.frame_index)
        if best_key is None or key < best_key:
            best, best_key = obs, key
    return best


def _crop_image(frame_gray, frame_bgr, obs, block: IndexedBlock, *, image_format: str,
                signature_grid: tuple[int, int], cv2, np) -> tuple[bytes, CropSignature]:
    """The encoded crop and its signature, from one decoded frame and one sighting.

    The IMAGE is cut from the colour decode — these are photographs and the model reads them as
    such — while the SIGNATURE is computed from the greyscale one, through the same
    `_signature_from_gray` a later verification pass reaches via `signature_of`. Both are cut
    from the same rows, so they describe the same pixels.
    """
    y0, y1, x0, x1 = obs.frame_y0, obs.frame_y1, block.x0, block.x1
    ok, buf = cv2.imencode(image_format, frame_bgr[y0:y1, x0:x1])
    if not ok:
        raise ItemCropError(
            f"cv2 could not encode the {x1 - x0}x{y1 - y0} crop of frame {obs.frame_index} as "
            f"'{image_format}'")
    signature = _signature_from_gray(
        frame_gray[y0:y1, x0:x1], grid=signature_grid, cv2=cv2, np=np)
    return buf.tobytes(), signature


def _check_frames_are_the_indexed_frames(frames: Sequence[bytes], index: ItemIndex) -> None:
    """Refuse a frame list that is not, byte for byte, the one the index was measured on.

    THIS IS THE GUARD, and it is exact rather than nearly useless. It replaces a decoded-size
    comparison that could never fire on a real mix-up: every Hinge screencap on the calibrated
    device is 1080x2400, so pairing an index with a DIFFERENT capture of the same phone passed it
    silently. A validation pass drove exactly that — the same frames in reverse order — and got a
    usable payload with ten images, ZERO failures, and signature distances of 18.8 to 156.5 from
    the crops it should have produced, against the 4.66 that separates the two most alike genuine
    items. The model would have been shown one profile's photos while doc 5.6 stored them as the
    verification reference for another.

    A digest closes it completely and cheaply: `segment_frame` records the sha256 of the bytes it
    measured (`FrameSegmentation.frame_digest`), so matching bytes are the same frame with no
    tolerance, no false positives, and no dependence on which frames happen to get cropped. Every
    frame is checked up front rather than lazily at decode time, because a wrong frame that no
    item happens to be cropped from still makes the payload a claim about the wrong capture.
    """
    for i, (frame, seg) in enumerate(zip(frames, index.frames, strict=True)):
        digest = hashlib.sha256(frame).hexdigest()
        if digest != seg.frame_digest:
            raise ItemCropError(
                f"frame {i} is not the frame the index was built from (sha256 {digest[:12]} "
                f"against {seg.frame_digest[:12]}) — the index's page coordinates only mean "
                "anything against the exact capture they were measured on, and cropping another "
                "one at them would produce well-formed crops of the wrong profile")


def _signature_drift(obs, block: IndexedBlock, index: ItemIndex, reference: CropSignature, *,
                     decode, signature_grid: tuple[int, int], cv2, np,
                     ) -> tuple[float | None, tuple[int, ...]]:
    """How much this crop's OWN pixels move when the same page rows are looked at again.

    THE MEASUREMENT DOC 5.6 CANNOT BE SET WITHOUT, and the one this module previously got wrong.
    Its docstring shipped a re-observation figure of "max 0.02 of 11 pairs" and read it as a
    property of the signature. It was a property of one profile AND of how it was measured: the
    pairs compared were only those where segmentation happened to return a byte-identical rect in
    both frames, which systematically excluded every card that was moving, because a moving
    card's segmented bottom edge wobbles by a pixel. Re-measured by RESAMPLING the rect instead
    of asking segmentation for it, the same capture's last item drifts 2.66 and a second
    profile's animated card drifts 25.0 — two orders of magnitude above the number that shipped.
    Doc 5.4 named animated cards as needing "a tolerance band on the signature comparison, or
    explicit detection"; 0.02 would reject a correct re-observation on both profiles and 30 would
    accept a wrong item on either, so this is the explicit detection, and it ships the number
    rather than a verdict.

    Method, and why each part of it is confound-free:

      * the rect is the CHOSEN sighting's page rows, resampled at exactly those page rows in
        every other frame whose analysed band fully contains them. Identical page rows in both
        samples means zero rect error, which matters because a rect located 1/2/4px off costs
        1.47/2.93/5.78 grey levels on its own and would otherwise be indistinguishable from
        content that moved;
      * segmentation is not consulted for the other frames. That is the whole correction: the
        cards worth detecting are exactly the ones segmentation disagrees with itself about;
      * frames the rect does not fully fit inside are skipped rather than clipped, since a
        clipped rect is a different region and its distance would not be a drift.

    Returns `(None, ())` when no other frame contained the rect. That is IGNORANCE, not
    stability, and every consumer of it is written to keep the two apart — a bot-cadence capture
    that never revisits a scroll position can legitimately measure nothing at all.

    Cost is one greyscale decode per frame that contains any crop, which is why `_decoder` caches
    grey separately from colour: the colour decode is only ever needed for the one frame an item
    is cropped from.
    """
    page_y0, page_y1 = obs.page_y0, obs.page_y1
    worst: float | None = None
    sampled: list[int] = []
    for i, offset in enumerate(index.offsets):
        if offset is None or i == obs.frame_index:
            continue
        y0, y1 = page_y0 - offset, page_y1 - offset
        band0, band1 = index.frames[i].band
        if y0 < band0 or y1 > band1:
            continue
        other = _signature_from_gray(
            decode(i, colour=False)[y0:y1, block.x0:block.x1],
            grid=signature_grid, cv2=cv2, np=np)
        distance = reference.distance(other)
        worst = distance if worst is None else max(worst, distance)
        sampled.append(i)
    return worst, tuple(sampled)


def _decoder(frames: Sequence[bytes], cv2, np):
    """A lazy per-frame decoder returning greyscale or colour, decoding each at most once.

    Lazy because a 24-frame capture is cropped from a handful of its frames and decoding the rest
    is pure cost; cached because a frame that holds two items is decoded once for both. The two
    colour modes are cached separately because they are separate decodes and are wanted in
    different places: `_signature_drift` re-samples rects in every frame that contains them and
    needs only grey, while only the frame an item is actually cropped from needs colour.

    Frame IDENTITY is not checked here — `_check_frames_are_the_indexed_frames` did that for the
    whole list before any of this ran, exactly and up front.
    """
    cache: dict[tuple[int, bool], object] = {}

    def _decode(frame_index: int, *, colour: bool):
        key = (frame_index, colour)
        if key not in cache:
            img = cv2.imdecode(
                np.frombuffer(frames[frame_index], dtype=np.uint8),
                cv2.IMREAD_COLOR if colour else cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ItemCropError(
                    f"frame {frame_index} did not decode as an image "
                    f"({len(frames[frame_index])} bytes)")
            cache[key] = img
        return cache[key]

    return _decode


def build_item_payload(frames: Sequence[bytes], index: ItemIndex, *,
                       exclude: Callable[[IndexedBlock], str | None] | None = None,
                       unnumber: Callable[[bytes], str | None] | None = None,
                       unnumber_without_evidence: Callable[
                           [StillPhotoEvidence], str | None] | None = None,
                       still_photo_dwell: dict[int, StillPhotoDwell] | None = None,
                       image_format: str = _CROP_IMAGE_FORMAT,
                       signature_grid: tuple[int, int] = _SIGNATURE_GRID) -> ItemPayload:
    """Turn an item index and the frames it was built from into the model's numbered image list.

    `frames` MUST be the very sequence `index` was built from, same order and same length — the
    index's coordinates are meaningless against any other capture, and the length and per-frame
    size are both checked rather than trusted.

    `exclude` is doc 2.4's geometry policy hook: given a block, return the REASON it must not be shown, or
    None to show it. Excluded blocks get no image at all, so they cannot reach the model by
    accident, and their heart ordinals are untouched — see the module docstring on doc 5.3's index
    trap, and `exclude_page_rows` for a ready-made predicate. The default shows everything the
    index resolved, which is what both calibration captures needed.

    `unnumber` is the selection-policy twin. It sees complete encoded bytes only for
    heart-bearing selectable blocks, after cropping but before numbering. A returned reason
    demotes the crop to readable, unnumbered context and retains the original heart ordinal.
    Hinge supplies `unnumber_unless_confident_photo`; generic callers keep the historical
    all-selectable policy.

    `unnumber_without_evidence` is the independent still-media twin. It receives one
    `StillPhotoEvidence`: the measured signature drift and the source frames that produced it,
    plus whatever an un-interacted dwell of this candidate's card rect observed. Hinge uses it to
    require affirmative still-photo evidence after content classification; generic callers opt
    out. `still_photo_dwell` is where the dwell half arrives -- a mapping keyed by HEART ordinal
    (page identity, stable regardless of what policy excludes) built by
    `still_photo_dwell_evidence`. Default None means no dwell was taken, which the Hinge gate
    treats as a refusal the moment numbering is licensed; it can never read as a passing dwell.

    Raises `ItemCropError` when there is nothing to crop or no way to crop it: an UNUSABLE index
    (doc 5.3's hard stop, inherited rather than re-derived), a frame list that does not match the
    index, missing cv2/numpy, or bytes that will not decode or encode. Everything else is a
    RESULT: read `ItemPayload.usable` before reading `images`, and treat False as a hard stop.
    """
    if not index.usable:
        # Inherited hard stop. Cropping an index that contradicts itself would produce a numbered
        # list whose numbers mean nothing, and the numbering is the entire reason doc 5.2 crops.
        raise ItemCropError(
            "refusing to crop an unusable item index — its numbering is not trustworthy, and "
            "doc 5.3 treats a missing table as a hard stop rather than a reason to fall back: "
            + "; ".join(index.failures))
    if len(frames) != len(index.frames):
        raise ItemCropError(
            f"{len(frames)} frame(s) given for an index built from {len(index.frames)} — the "
            "index's page coordinates only mean anything against the capture they were measured "
            "on")
    _check_frames_are_the_indexed_frames(frames, index)
    if not frames:
        # `build_item_index` refuses an empty capture, so this only catches a hand-assembled
        # index. Named anyway rather than left to fail on `frames[0]`, because the band height
        # every measurement below is taken against is read off frame 0.
        raise ItemCropError(
            "no frames to crop from: an empty capture is a capture failure, and an empty item "
            "list for it would read as 'this profile has no items'")

    cv2, np = _require_vision()
    decode = _decoder(frames, cv2, np)
    band_height = _band_height(index)
    failures: list[str] = _over_tall_failures(index, band_height)

    crops: list[ItemCrop] = []
    number = 0
    for block in index.blocks:
        common = dict(heart_ordinal=block.heart_ordinal, page_y0=block.page_y0,
                      page_y1=block.page_y1, x0=block.x0, x1=block.x1,
                      frame_index=None, frame_y0=None, frame_y1=None,
                      image=None, signature=None, signature_drift=None, drift_frames=(),
                      nearest_item_distance=None)

        # Hinge's scroll-top header, already outside both index spaces (item_index.py classes it
        # only at a caller-confirmed scroll-top). Recorded so the payload accounts for every
        # block, never cropped.
        if block.kind == ITEM_LEADING_CHROME:
            crops.append(ItemCrop(kind=CROP_CHROME, number=None, reason=block.reason, **common))
            continue

        # Policy first, before the class dispatch, because doc 2.4's block may land in either
        # tier: heartless it is an ITEM_CONTEXT, and doc 5.3 explicitly anticipates the case
        # where it "turns out to HAVE a heart", where it is an ITEM_SELECTABLE that must still
        # not be offered. Both are dropped here, and neither renumbers the heart ordinals.
        reason = exclude(block) if exclude is not None else None
        if reason is not None:
            crops.append(ItemCrop(kind=CROP_EXCLUDED, number=None, reason=reason, **common))
            continue

        if block.kind == ITEM_PARTIAL:
            # Rule one: a crop is never a fragment. The block's own extent is a lower bound (its
            # sightings' hull), so there is no honest image to make here — only a half card that
            # would be sent to the model and stored as the verification reference.
            crops.append(ItemCrop(
                kind=CROP_UNCROPPABLE, number=None,
                reason=("no frame observed both of this block's edges, so any crop of it would "
                        "be a fragment of unknown extent; " + block.reason),
                **common))
            continue

        if block.kind not in (ITEM_SELECTABLE, ITEM_CONTEXT):
            # ITEM_AMBIGUOUS is the only class left, and it always carries a failure of its own, so
            # a USABLE index cannot contain one. Refuse rather than fall through to an unnumbered
            # crop: an unknown page class here means item_index.py grew one this module has not
            # been told about, and guessing which tier it belongs in is exactly the policy decision
            # doc 5.3 says must be explicit. Asked BEFORE any cropping, so nothing is encoded for a
            # block we are about to refuse.
            raise ItemCropError(
                f"the block at page rows {block.page_y0}..{block.page_y1} has page class "
                f"'{block.kind}', which this module has no crop policy for")

        obs = _choose_sighting(block, index)
        if obs is None:
            # Unreachable from a usable index — ITEM_SELECTABLE and ITEM_CONTEXT both require a
            # complete sighting — so reaching it means the index and this module disagree about
            # what "resolved" means, which is a contradiction to surface rather than to skip.
            failures.append(
                f"the block at page rows {block.page_y0}..{block.page_y1} is classed "
                f"'{block.kind}', which means some frame bounded it end to end, yet it offers no "
                "croppable sighting")
            crops.append(ItemCrop(
                kind=CROP_UNCROPPABLE, number=None,
                reason="classed as resolved but carries no complete sighting to crop from",
                **common))
            continue

        image, signature = _crop_image(
            decode(obs.frame_index, colour=False), decode(obs.frame_index, colour=True),
            obs, block, image_format=image_format, signature_grid=signature_grid,
            cv2=cv2, np=np)
        drift, drift_frames = _signature_drift(
            obs, block, index, signature, decode=decode, signature_grid=signature_grid,
            cv2=cv2, np=np)
        common.update(page_y0=obs.page_y0, page_y1=obs.page_y1, frame_index=obs.frame_index,
                      frame_y0=obs.frame_y0, frame_y1=obs.frame_y1,
                      image=image, signature=signature, signature_drift=drift,
                      drift_frames=drift_frames)

        if block.kind == ITEM_SELECTABLE and unnumber is not None:
            reason = unnumber(image)
            if reason is not None:
                crops.append(ItemCrop(kind=CROP_CONTEXT, number=None, reason=reason, **common))
                continue
        if block.kind == ITEM_SELECTABLE and unnumber_without_evidence is not None:
            # The dwell is looked up by heart ordinal because that is the one identity a block
            # keeps across policy: the model item number does not exist yet at this point, and
            # page rows shift with every scroll.
            reason = unnumber_without_evidence(still_photo_evidence_from_drift(
                drift, drift_frames,
                (still_photo_dwell or {}).get(block.heart_ordinal)))
            if reason is not None:
                crops.append(ItemCrop(kind=CROP_CONTEXT, number=None, reason=reason, **common))
                continue

        if block.kind == ITEM_SELECTABLE:
            number += 1
            crops.append(ItemCrop(
                kind=CROP_ITEM, number=number,
                reason=(f"item {number} (heart {block.heart_ordinal} on the page), cropped from "
                        f"frame {obs.frame_index} rows {obs.frame_y0}..{obs.frame_y1}"),
                **common))
        else:
            crops.append(ItemCrop(
                kind=CROP_CONTEXT, number=None,
                reason=(f"context, cropped from frame {obs.frame_index} rows {obs.frame_y0}.."
                        f"{obs.frame_y1}: sent unnumbered, so it can be read and referenced but "
                        "never chosen"),
                **common))

    if not number:
        # Rule four. An empty numbered list is not a request the model can answer, and shipping
        # one would spend a call to be told so.
        failures.append(
            f"no policy-approved selectable item survived to be numbered, from "
            f"{len(index.blocks)} indexed block(s) — there is nothing for the model to choose "
            "between")

    _fill_nearest_item_distance(crops)
    return ItemPayload(
        crops=tuple(crops), truncated=index.truncated, at_scroll_top=index.at_scroll_top,
        signature_grid=signature_grid, failures=tuple(failures))


def _fill_nearest_item_distance(crops: list[ItemCrop]) -> None:
    """Give every numbered item its distance to the nearest OTHER numbered item, in place.

    Done in a second pass for the same reason `item_index._assemble` numbers in one: the quantity
    is a property of the whole list, not of the crop, and it cannot be known until the last crop
    exists. It is paired with `ItemCrop.signature_drift` by `separable`, which is the comparison
    doc 5.6 has to make and the one this module used to answer with a shipped constant.

    Only ITEMS take part. A context block is never tapped, so nothing is ever verified against
    it, and letting it into the nearest-neighbour set would shrink the separation that guards a
    real choice. With one item there is no other to confuse it with and the distance stays None,
    which `separable` reports as unknown rather than as a clean bill of health.
    """
    # Positions, not the records themselves: `ItemCrop` is a frozen dataclass and compares by
    # value, so looking one up by identity in the list would silently find the wrong entry if two
    # crops ever came out equal.
    items = [(i, c) for i, c in enumerate(crops)
             if c.kind == CROP_ITEM and c.signature is not None]
    nearest: dict[int, float] = {}
    for (i, a), (j, b) in _pairs(items):
        distance = a.signature.distance(b.signature)
        nearest[i] = min(nearest.get(i, distance), distance)
        nearest[j] = min(nearest.get(j, distance), distance)
    for i, distance in nearest.items():
        crops[i] = replace(crops[i], nearest_item_distance=distance)
