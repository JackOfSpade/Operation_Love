"""Is the selected card above Hinge's inline composer the item the model actually chose?

The public ``Sheet*`` names are retained for compatibility, but the screenshots originally
described as a separate sheet were the same selected-card reflow Hinge 9.134.0 now exposes
unambiguously inline. Geometry alone is therefore only a crop locator. Production supplies an
independently detected ``ComposerSurface`` and this module requires card -> input -> CTA topology
before that geometry can license text.

ops/OPENER-REDESIGN.md 5.6 asks for exactly one thing here, and this module is only that:

    "Once items are cropped and indexed, confirming the opened sheet is a deterministic
    signature match between the sheet's displayed item and the stored crop of item N. No model
    call, no cost, no judgment. Today's `on_target` is a pre-tap screen check that can report
    success while the tap lands on the neighbour; this is a post-tap content check, which is the
    thing that was actually missing."

    locate_sheet_preview(frame) -> SheetPreview          # where the sheet renders the item
    verification_blocker(payload, k) -> str              # BEFORE the tap: can item k be checked?
    verify_sheet_item(frame, payload, k) -> SheetVerdict # AFTER the tap: is this item k?

A leaf module on the same terms as `segment.py`, `frameshift.py`, `item_index.py`,
`item_crops.py`, `item_identity.py`, `scroll_top.py` and `scroll_step.py`: pure functions over
frame BYTES plus explicit calibration parameters. No device, no I/O, no global state, no driver
state, and no judgement — every answer is a distance in grey levels against a bound derived from
the profile's own crops.

WHAT THE SHEET ACTUALLY RENDERS, MEASURED RATHER THAN ASSUMED
---------------------------------------------------------------
Doc 5.2's addendum closes with the honest admission that "5.6's real comparison is
crop-against-like-sheet, which no capture in the corpus contains". It does now: six real
`observe_like_anchor` screencaps of an open Hinge comment sheet were sitting in the local
(gitignored) debug directory, and everything below is measured off them plus the profile frames
captured either side of each one. Geometry and grey levels only; nothing about their content
appears here, in the tests, or anywhere else.

The sheet is laid out like this on the calibrated device, and the numbers are what the locator
below is built from:

  [corpus, 6 real sheets, 1080x2400] the item preview spans columns 95..985 (890px) on five of
  them and 91..989 (898px) on the sixth, with its top edge at row 236 on five and 338 on the
  sixth. Below it the comment box spans 117..963 (846px wide, 178 tall) and the "Send Like" row
  spans 140..939 (799px wide, 109 tall). The preview is the ONLY element that is both >= 870px
  wide and >= 300px tall, and it is always the topmost of them.

  [corpus] the profile card those previews were cut from spans columns 53..1027 (974px), so the
  sheet renders it at 890/974 = 0.9137 scale. The card's own margin (53) and the sheet's (95) are
  42px apart, which is what lets the locator refuse a profile screen outright.

THE PREVIEW IS A BOTTOM-ANCHORED WINDOW, AND THAT WAS THE WHOLE BALL GAME
---------------------------------------------------------------------------
The obvious comparison — the whole preview against the whole stored crop — is WRONG, and it is
wrong by enough to matter. The sheet fits the card into a fixed content region and shows the
card's BOTTOM when it does not fit, so the preview is a uniformly width-scaled window anchored at
the card's bottom edge:

    window_px = round(preview_height * crop_width / preview_width)   # the card rows shown
    reference = the crop's LAST window_px rows

  [corpus, the four sheets whose source card was also captured in a nearby profile frame, at the
  32x32 grid the stored crops use. Distance from the preview to that card's signature, three ways:

      whole card        9.296   18.836    1.849   (4th: no whole-card comparison, see below)
      BOTTOM window     1.011    0.533    1.849    4.600
      TOP window       20.108   31.661    1.849       —

  The third sheet's card fitted whole, so its three numbers coincide; that is the control. On the
  other two the bottom window is 9x and 35x nearer than the whole card, and the top window is
  worse than either.]

The rule needs no constant: `crop_width` is the stored crop's own pixel width and `preview_width`
is the located rect's, so a profile whose cards are a different width, or a device with a
different screen, re-derives the scale rather than inheriting one.

A ~1% ERROR IN THAT SCALE IS WHAT `_SCALE_TOLERANCE` EXISTS FOR
-----------------------------------------------------------------
The 4.600 above is not a content difference. The preview rect is measured to a pixel or two, so
the derived scale carries about a percent of error, and a percent of an 800-row window is eight
rows:

  [corpus] sweeping the window height by +-2% and taking the minimum turns those four numbers
  into 1.011 / 0.533 / 1.846 / 0.946 — the tallest card's 4.600 was an 8-row (1.0%) window
  error and nothing else. The same sweep applied to the WRONG cards in the same frames moves them
  by at most 0.9 grey levels (52.024 -> 51.738, 58.858 -> 58.058, 61.128 -> 60.928,
  59.354 -> 58.772), because a card that is not the one on the sheet is not nearly aligned with
  it at any window height.

The sweep is applied identically to every candidate item, so it cannot favour the one we hope to
confirm. It does weaken the uniqueness proof below from "the distance" to "the best of
`2*tol+1` alignments", and that is stated rather than hidden: the measured cost is <= 0.9 grey
levels against bounds of 4.0 and up.

WHY THERE IS NO FIXED THRESHOLD, WITH THE MATRIX THAT PROVES IT
------------------------------------------------------------------
Doc 5.6 inherits this as a blocker ("a fixed signature threshold is ruled out") and the full
re-observation matrix over both calibration profiles says why in one line:

  [corpus, `ops/calibration/`, 42 re-observations of 16 items across two profiles, each compared
  against all 9 of its profile's stored signatures at the 32x32 grid:

      profile B (24 frames, 9 items, 11 re-observations)
          diagonal      0.002 .. 2.656       off-diagonal   4.670 .. 154.577
      profile A (57 frames, 9 items, 31 re-observations)
          diagonal      0.000 .. 25.009      off-diagonal   6.266 .. 150.288

  The nearest stored signature was the right item on 42 of 42. But the diagonal MAXIMUM (25.009,
  the animated card) is five times the off-diagonal MINIMUM (4.670, two prompt cards on the other
  profile), so the two distributions OVERLAP and no single number separates them. Per item they
  never overlap: the worst item's own re-observation is 25.009 against its own nearest neighbour
  at 47.861.]

So the bound is per item, and it is derived rather than tuned — see `_SEPARATION_FRACTION`.

THE GRID IS FINER HERE THAN THE STORED ONE, AND THAT IS A MEASUREMENT
------------------------------------------------------------------------
`item_crops._SIGNATURE_GRID` is 32x32 and stays there; this module recomputes its references from
the stored crop BYTES (through the same `item_crops.signature_of`) at `_VERIFY_GRID`, because it
has to re-cut them into windows anyway and the grid is therefore free to choose.

  [corpus] the swept sheet-reproduction penalty barely moves with the grid — 1.816 / 1.849 /
  1.878 / 1.904 / 2.015 at 24 / 32 / 48 / 64 / 96 — while the distance between the two most alike
  items on a profile grows steadily: 4.372 / 4.661 / 5.990 / 8.024 / 10.021 (profile B) and
  5.578 / 6.266 / 7.273 / 8.727 / 11.195 (profile A). The ratio of the two is what verification
  has to live on, and it goes 2.41 / 2.52 / 3.19 / 4.21 / 4.97 on the worse profile.

64x64 is taken because it doubles the worst-case separation over the stored grid for a 3% rise in
the penalty, and because it is the finest grid `item_crops`' own published table measured, so it
is not an extrapolation.

WHAT THIS REFUSES TO ANSWER, AND WHY THAT IS THE POINT
---------------------------------------------------------
Two items can be too alike for this comparison to have any power — measured at 4.661 grey levels
between two prompt cards on one profile — and an animated card's own frame-to-frame noise can
exceed its distance to its neighbours (25.009 against 47.861). Doc 5.4 names both as failure
classes and offers "a tolerance band, or explicit detection and a different path". The tolerance
band is not taken: a band wide enough to accept the animated card is 25 grey levels wide, which is
five times the distance between the two most alike items on the other profile, i.e. wide enough to
accept a wrong item. `verification_blocker` is the explicit detection, it runs BEFORE the tap, and
the different path is the one doc 5.6's owner rule asks for — the run stops, having touched
nothing.

  [corpus] over both profiles' 18 items that check refuses exactly ONE: the animated card, whose
  own drift (25.009) plus the sheet's reproduction penalty (1.95) exceeds half its distance to its
  nearest neighbour (23.931). The other 17 pass, the tightest with 0.36 grey levels to spare.

THE RELATIVE TEST IS CLOSED-SET; PRODUCTION ALSO REQUIRES AN ABSOLUTE CEILING
--------------------------------------------------------------------------------
Read `_SEPARATION_FRACTION` carefully: the 0.5 proves item k is the UNIQUE NEAREST of the stored
items. By itself it says nothing about whether the sheet is any of them: content outside the
payload only has to beat the payload's own internal spacing to be accepted as the nearest item.

  [corpus, measured end to end through the shipped driver on 2026-08-12: a stale payload for
  profile B plus a sheet rendering profile A's card returned VERIFY_MATCH, and the driver typed
  the opener and sent the like. 10 of 540 foreign comparisons were accepted, reproducibly across
  five resampling kernels; the reverse direction refused, but by only 1.118x.]

Half of that was a defect and is fixed: the neighbour set was being pruned by a height filter, so
the bound grew whenever the sheet was tall (`_compare_item`). The other half is structural:
`absolute_max_dist` supplies the necessary reject option, but its value must be calibrated on the
actual device and held-out correct/foreign sheets. No guessed numeric default is shipped; the
Hinge production path refuses targeted text unless that calibrated ceiling is configured.

"NOT IN THE PAYLOAD" IS NOT ONLY ANOTHER PERSON'S CARD, EITHER. A block the index resolved as
PARTIAL or UNCROPPABLE still carries a heart, is still tappable, and by construction has no crop,
so it is not in `payload.items` and cannot be a reference. A tap that lands on one puts
out-of-payload content on the sheet from the SAME profile, where an identity gate says nothing.
The model can never choose such a block (it is not numbered), so reaching one takes a navigation
miss. Production therefore requires both the relative bound and the calibrated absolute ceiling.

SO A CALLER MUST NOT USE THIS AS THE ONLY ANSWER TO "IS THIS THE RIGHT CARD". Doc 5.3's claim
that a stale table is "a reliability bug rather than a safety one ... the post-tap signature check
compares the opened sheet against the wrong reference, fails, and stops the run" is MEASURABLY
FALSE and must not be relied on again. On the auto path `hinge._confirm_payload_profile` asks
whose profile it is before and after the tap, requires the calibrated absolute ceiling, and
`hinge._like_comment_sheet` refuses a model item number it cannot navigate to; those gates keep
this hole off a real send today.

AND FOR WHOEVER BUILDS DOC 5.9's OBSERVE INVERSION, one measurement that makes the same check
cheap there: THE SHEET DOES NOT OCCLUDE THE STICKY HEADER. `HINGE_SPEC.identity_band` cuts rows
115..226 and the sheet's own preview starts at row 236, so the strip is still the person's header
while the sheet is open.

  [corpus, the six real `observe_like_anchor` sheets: every one reads `confirmed_not_top` on that
  band, i.e. the strip carries a person rather than Hinge's filter chips, and every one measures
  **0.000** against its own profile's neighbouring scrolled frames. So the frame this function is
  handed is itself checkable, and a deck advance can be refused on identity rather than on the
  pixels of a card.]

That is the cheaper and stronger half of the answer, but it is not a complete one either: an old
3.0 identity bound false-accepted a measured different-profile pair at 2.565. Production now
requires an operator-calibrated identity ceiling strictly below that collision and still runs
both checks. Nobody may treat an item MATCH alone as evidence about whose profile is on screen.

WHAT THIS DOES NOT DO
------------------------
  * It does not replace `item_identity`'s gate, and it is not a weaker version of it. That rules
    out the wrong PERSON before any gesture; this rules out the wrong ITEM within one profile's
    payload after the tap. Neither implies the other, and this one does not degrade gracefully
    into the other -- see the section above.
  * It does not navigate, tap, type or send. It answers a question about a frame.
  * It does not repair. There is no "closest reachable item" and no re-ask: doc 5.6's owner rule
    is that a miss stops the run, and a verdict that is not MATCH has no second reading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .item_crops import (
    CropSignature, ItemCropError, ItemPayload, _signature_from_gray, signature_of)


# =====================================================================================
# Calibration constants.
#
# Sources, the same convention as the seven leaf modules beside this one:
#   [corpus] measurements taken on this machine, offline, over the gitignored real captures:
#            `ops/calibration/` (two hand/bot-scrolled profiles, 9 items each) and the six
#            `observe_like_anchor` sheet screencaps in `data/hinge_debug/`. Numbers only -- those
#            are real people's profiles and nothing but geometry and grey levels leaves them.
#   [doc]    ops/OPENER-REDESIGN.md 5.2 (crops and signatures), 5.4 (the animated-card and
#            look-alike failure classes), 5.6 (this section).
# =====================================================================================

# The (width, height) cell grid the sheet and its references are compared at. See the module
# docstring's grid section for the sweep this is read off. Deliberately NOT
# `item_crops._SIGNATURE_GRID`: the stored crops keep theirs, and this pass recomputes from the
# crop bytes because it has to re-cut them into bottom-anchored windows regardless.
_VERIFY_GRID = (64, 64)

# The sheet's own reproduction noise: how far a preview sits from the very card it renders, once
# the window is aligned. [corpus, the four real sheets whose source card was also captured in a
# profile frame: 1.011 / 0.533 / 1.846 / 0.946 at `item_crops`' stored 32x32 grid, and
# 1.154 / 0.577 / 1.904 / 0.804 at `_VERIFY_GRID`.] 1.95 is the larger of those two maxima rounded
# up -- observed rather than fitted, and four sheets is four sheets.
#
# It is never an accept threshold. It appears in `verification_blocker`'s prediction, which only
# ever REFUSES, and in the fallback bound for a payload with no second item, where it is scaled up
# rather than down. A real penalty larger than this therefore produces a stop, never an
# acceptance, which is the direction doc 5.6's owner rule asks for.
_SHEET_RENDER_DRIFT = 1.95
# A ceiling at or above this measured foreign-card acceptance can no longer distinguish the
# known wrong card.  Hinge validates the configured value too; this leaf repeats the check so a
# direct caller cannot accidentally disable the absolute reject guard with ``inf``/``nan``.
_SHEET_FALSE_MATCH_DISTANCE = 14.91

# Half the distance to the nearest other item is the accept bound, and the 0.5 is a proof rather
# than a tuning: `CropSignature.distance` is a scaled L1 norm, so it satisfies the triangle
# inequality, and if the sheet sits within `nearest_k / 2` of item k then for every other item j
#     d_j >= dist(k, j) - d_k >= nearest_k - nearest_k/2 = nearest_k/2 > d_k,
# i.e. item k is the UNIQUE nearest stored item. The argmin check in `verify_sheet_item` is
# therefore corroboration, not a second threshold. Read the other way round: for a sheet that is
# really item j to be accepted as k, item j's own reproduction noise would have to exceed half the
# distance between them, which is exactly the condition `verification_blocker` refuses on.
_SEPARATION_FRACTION = 0.5

# How far the derived window height is swept, as a fraction, before the nearest alignment is
# taken. [corpus: the largest error observed between the width-derived window and the best one is
# 8 rows of 804, i.e. 1.0%; 2% is that with a factor of two on it. Cost on a WRONG card, measured
# on the same frames: at most 0.9 grey levels.]
_SCALE_TOLERANCE = 0.02

# Hinge 9.134's inline composer can reframe a selected square photo vertically before putting
# the comment controls below it. The source crop remains complete; what changes is which interior
# strip is visible, so the legacy modal's bottom-anchor assumption is not valid there. The 2026-08-15
# Alex composer frame showed 800 of 1109 source rows (27.86% hidden); 30% admits that measured
# renderer with a small margin, while the contiguous full-width window and fixed origin sample cap
# still prevent this from becoming an arbitrary image-patch search.
_INLINE_REFRAME_MAX_HIDDEN_FRACTION = 0.30
_INLINE_REFRAME_ORIGIN_SAMPLES = 49
# The ordinary profile card draws its heart over the photo's lower-right corner. Hinge removes
# that control from the selected inline preview, so comparison excludes the fixed control lane on
# BOTH source and preview. This is verification-only; the model crop remains unchanged.
_INLINE_REFRAME_RIGHT_CONTROL_FRACTION = 0.18
# This is deliberately separate from ``_SHEET_RENDER_DRIFT``.  The latter is the measured
# legacy/modal reproduction penalty and must not grow when Hinge changes only the inline
# selected-photo renderer.  The first two held-out, full-crop inline renders after 9.134's
# reframe change measured 3.729 (Shai) and 6.703 (Malaika) at this module's 64x64 grid.  7.00 is
# the smallest tenth-level ceiling above both, while the reciprocal held-out cards measured
# 58.758 and 61.552.  It is used ONLY for the one-item fallback: a multi-item payload continues
# to require the triangle-inequality/unique-nearest separation proof below.  Production also
# supplies its independently frozen ``absolute_max_dist``; this ceiling cannot relax that guard.
_INLINE_COMPOSER_ONE_ITEM_MAX_DIST = 7.00
# Some selected photographs contain a large near-white/sky region.  The legacy row-background
# locator quite properly cannot call those rows part of an image (they are indistinguishable from
# the page), which fragments a real 856px preview into a 200px run.  This compact fallback is
# available only after an independently detected composer supplies the field/CTA geometry.  It
# still requires a substantial image run bound to the independently detected comment field.
_INLINE_COMPACT_MIN_WIDTH_FRACTION = 0.78
# Lauren's Hinge 10.1.0 selected photo kept its right edge exactly on the composer's x=985
# boundary while 73 rows of pale sky at the LEFT edge blended into the page.  The ordinary
# two-edge probe therefore saw only 661px of the 890px photo (74.27%) at the narrowest row and
# split one continuous preview at row 628.  A verified composer may use that measured one-edge
# continuation, but only while the opposite edge stays aligned and at least this much visible
# photo structure remains.  The public/legacy locator never uses this lower floor.
_INLINE_PREVIEW_ONE_EDGE_MIN_WIDTH_FRACTION = 0.74
# A selected photo can contain a page-coloured horizontal feature that makes the row-span probe
# briefly indistinguishable from background.  The Hinge 10.1.0 held-out composer measured one
# such 3-row interruption at rows 942..945 inside an otherwise continuous 236..1092 preview.
# Bridge exactly that measured maximum only after the composer has independently established the
# surrounding topology.  Four consecutive unsupported rows remain a hard boundary.
_INLINE_PREVIEW_MAX_INTERNAL_GAP_PX = 3

# --- the preview locator's geometry ---------------------------------------------------
# All [corpus, 6 real sheets]; see the module docstring's layout section for the full table.

# Rows above this are never considered. The status bar and the sheet's own title row live there
# and neither is anywhere near wide enough to be mistaken for the preview, so this is belt and
# braces rather than the discriminator -- the width and height tests below are.
_PREVIEW_SEARCH_TOP_PX = 110

# A row belongs to the preview when its non-background span is at least this wide. The preview
# measures 890 or 898; the comment box below it 846 and the send row 799. 870 sits 20px under the
# narrowest preview and 24px over the widest impostor.
_PREVIEW_MIN_WIDTH_PX = 870

# ...and the run of such rows must be at least this tall. The previews measure 735, 856 and 898;
# the comment box 178 and the send row 109. 300 is 435 under the shortest preview seen and 122
# over the tallest impostor. A hypothetical item small enough to render under 300px would be
# missed here and stop the run, which is the safe direction.
_PREVIEW_MIN_HEIGHT_PX = 300

# The preview's left edge, and how far it may sit from it. Measured at 95 on five sheets and 91 on
# the sixth. This is the test that tells a comment sheet from a PROFILE screen: a profile card's
# own left edge is `segment._CARD_MARGIN_PX` = 53, which is 42px away and 27px outside this
# window. Without it, a screen with no sheet on it at all would offer its card as a "preview".
_PREVIEW_MARGIN_PX = 95
_PREVIEW_MARGIN_TOLERANCE_PX = 15

# Columns read for the per-row background reference, and how far a pixel may sit from it and still
# count as background. Both mirror `segment._row_background_profile`: the reference is LOCAL to
# the row because the page background is a vertical gradient (doc 5.10 measured ~(255,254,253) at
# y=300 against ~(243,243,243) at y=2100), and it is read from the margins either side so the
# element being measured never contributes to its own reference.
_MARGIN_PROBE = (8, 80)
_ROW_BACKGROUND_TOLERANCE = 6


# --- the three outcomes ---------------------------------------------------------------
VERIFY_MATCH = "verify_match"                # the sheet is showing the item that was chosen
VERIFY_MISMATCH = "verify_mismatch"          # it is showing something else, or nothing near enough
VERIFY_UNVERIFIABLE = "verify_unverifiable"  # the comparison has no power for this item


class SheetVerificationError(RuntimeError):
    """The check could not LOOK: undecodable frame bytes, cv2/numpy missing, an unusable payload,
    an item number outside the list, or no preview on the screen it was handed.

    Distinct from `VERIFY_MISMATCH`, which is a verdict reached by looking. The separation is
    `segment.SegmentationError`'s and `item_identity.IdentityError`'s, for their reason: "we could
    not look" must never be reachable by a route that also produces a verdict, or a broken install
    degrades into a stream of plausible-looking answers.

    A DRIVER must treat it as a stop exactly as it treats a mismatch -- the sheet is open, the
    opener has not been typed, and there is no evidence that typing it would be right.
    """


def _require_vision():
    """Import cv2 + numpy or raise. Same fail-loud contract as `segment._require_vision` and
    `item_crops._require_vision`, and duplicated rather than shared for the same reason those two
    are: it is six lines, and the message names the operation that could not run."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001 — surfaced as SheetVerificationError, never swallowed
        raise SheetVerificationError(
            "verifying the like sheet needs opencv-python + numpy "
            f"(extra: operation-love[hinge]); import failed: {exc}") from exc
    return cv2, np


@dataclass(frozen=True)
class SheetPreview:
    """Where an open comment sheet is rendering the item, in that frame's own rows and columns.

    Half-open in both axes, exactly as `segment.Block` is. `reason` records how it was decided so
    a stop record can show the operator the geometry the run was looking at rather than only the
    verdict it reached.
    """
    y0: int
    y1: int
    x0: int
    x1: int
    reason: str

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass(frozen=True)
class ItemComparison:
    """One numbered item measured against the sheet, and everything the measurement rested on.

    `distance` is the smallest distance over the swept window alignments, `window_px` the
    width-derived window height before sweeping, `crop_px` the stored crop's own height (so
    `window_px < crop_px` says the sheet is showing a WINDOW of this card rather than all of it),
    `nearest_other` this item's distance to the nearest OTHER item measured in the same window
    space at the same grid, and `bound` the value `distance` had to come under.

    `distance` is None when this item cannot be what the sheet shows at all -- the card is
    shorter than the window the sheet is rendering -- which is evidence rather than an error, and
    keeps it out of the nearest-match. It does NOT keep it out of anybody's `nearest_other`: such
    an item still carries a reference and still bounds every other item, which is the whole of
    `_compare_item`'s 2026-08-12 fix.
    """
    number: int
    distance: float | None
    window_px: int
    crop_px: int
    nearest_other: float | None
    bound: float | None
    reason: str


@dataclass(frozen=True)
class SheetVerdict:
    """Whether the open sheet is showing the item that was chosen, and the whole case for it.

    `model_index` is what was INTENDED and `nearest_index` what the sheet actually looks most
    like -- doc 5.6 asks for both to be recorded on a miss, and they are the two numbers a stop
    line has to carry.

    `__bool__` RAISES, on `item_identity.IdentityVerdict.__bool__`'s and
    `scroll_top.ScrollTopVerdict.__bool__`'s precedent and for their reason: a frozen dataclass is
    truthy by default, so `if verify_sheet_item(...):` would read every outcome -- including "this
    comparison has no power" -- as a match, which is the single most dangerous line anyone could
    write against this API. Callers test `.matched`.
    """
    state: str
    model_index: int
    nearest_index: int | None
    distance: float | None
    bound: float | None
    comparisons: tuple[ItemComparison, ...]
    preview: SheetPreview
    grid: tuple[int, int]
    reason: str

    def __bool__(self):
        raise TypeError(
            "SheetVerdict has no truth value -- test .matched. A bare truth test would read "
            f"{self.state!r} as a match and type an opener under an item nobody checked")

    @property
    def matched(self) -> bool:
        """The one property that licenses typing. False for every other state, including the one
        that means the question could not be answered."""
        return self.state == VERIFY_MATCH


def _row_spans(gray, np, *, tolerance: float, probe: tuple[int, int]):
    """Per row, the first and last column that differs from that row's own background, or None.

    The reference is the median of the two margin strips, so the element being measured never
    contributes to the reference it is judged against, and it is re-read per row because the page
    background is a gradient (`segment._row_background_profile`'s reasoning, applied to a sheet).
    """
    lo, hi = probe
    width = gray.shape[1]
    left = gray[:, lo:hi]
    right = gray[:, width - hi:width - lo]
    reference = np.median(np.concatenate([left, right], axis=1), axis=1)
    differs = np.abs(gray.astype(np.int16) - reference[:, None]) > tolerance
    out = []
    for row in differs:
        cols = np.flatnonzero(row)
        out.append((int(cols[0]), int(cols[-1]) + 1) if cols.size else None)
    return out


def locate_sheet_preview(frame: bytes, *,
                         search_top_px: int = _PREVIEW_SEARCH_TOP_PX,
                         min_width_px: int = _PREVIEW_MIN_WIDTH_PX,
                         min_height_px: int = _PREVIEW_MIN_HEIGHT_PX,
                         margin_px: int = _PREVIEW_MARGIN_PX,
                         margin_tolerance_px: int = _PREVIEW_MARGIN_TOLERANCE_PX,
                         background_tolerance: float = _ROW_BACKGROUND_TOLERANCE,
                         margin_probe: tuple[int, int] = _MARGIN_PROBE) -> SheetPreview:
    """Where the open comment sheet is rendering the item. Raises when there is no such thing.

    The rule is the layout measurement in the module docstring turned into three tests, in the
    order that makes the failure legible: the TOPMOST run of rows at least `min_width_px` wide,
    at least `min_height_px` tall, whose left edge is `margin_px` +- `margin_tolerance_px`.

    The third test is the one that is not redundant. Width and height alone are satisfied by an
    ordinary profile card (974px wide, 685..1114 tall), so without it this would happily report a
    "preview" on a screen with no sheet on it at all -- and the caller is about to compare that
    against a stored crop of the same profile and find a match. The sheet indents its preview to
    95 where a card sits at 53, and 42px is not a tolerance anybody would widen by accident.

    Raises `SheetVerificationError` rather than returning None: doc 5.6's rule is that a check
    which cannot be made stops the run, and an Optional return is how "we did not look" gets
    quietly treated as "nothing was wrong".
    """
    cv2, np = _require_vision()
    gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SheetVerificationError(
            f"the like-sheet frame did not decode as an image ({len(frame)} bytes), so there is "
            "nothing to verify the chosen item against")
    height, width = gray.shape
    lo, hi = margin_probe
    if width < 2 * hi + 1 or hi <= lo:
        raise SheetVerificationError(
            f"a {width}px-wide frame leaves no margin strip at columns {lo}..{hi} to read a "
            "background reference from, so the preview cannot be bounded")
    spans = _row_spans(gray, np, tolerance=background_tolerance, probe=margin_probe)

    runs: list[tuple[int, int]] = []
    start = None
    for y in range(search_top_px, height):
        span = spans[y]
        wide = span is not None and (span[1] - span[0]) >= min_width_px
        if wide and start is None:
            start = y
        elif not wide and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, height))

    tall = [(a, b) for a, b in runs if b - a >= min_height_px]
    if not tall:
        widest = max(((b - a) for a, b in runs), default=0)
        raise SheetVerificationError(
            f"no comment-sheet item preview on this frame: nothing at least {min_width_px}px "
            f"wide runs for {min_height_px} rows below row {search_top_px} (the tallest such run "
            f"is {widest} rows). Doc 5.6 verifies the sheet against the stored crop, and there is "
            "no sheet here to verify")
    y0, y1 = tall[0]
    band = [spans[y] for y in range(y0, y1) if spans[y] is not None]
    x0 = int(np.median([s[0] for s in band]))
    x1 = int(np.median([s[1] for s in band]))
    if abs(x0 - margin_px) > margin_tolerance_px:
        raise SheetVerificationError(
            f"the topmost full-width block on this frame starts at column {x0}, and a comment "
            f"sheet indents its item preview to {margin_px} (+-{margin_tolerance_px}). A profile "
            "card sits at 53, so this looks like a profile screen rather than an open sheet, and "
            "comparing it against the stored crops would confirm the item we never tapped")
    return SheetPreview(
        y0=y0, y1=y1, x0=x0, x1=x1,
        reason=(f"topmost run of rows at least {min_width_px}px wide, {y1 - y0} rows tall, "
                f"indented to column {x0}"))


def _locate_inline_composer_preview(frame: bytes, composer_surface, *, cv2, np) -> SheetPreview:
    """Locate the selected photo that belongs to an independently proven inline composer.

    A Training reviewer may scroll Hinge's still-open inline surface while deciding.  In that
    state an ordinary profile card can be visible above the selected photo.  The legacy locator
    intentionally inspects the *topmost* wide block and rejects that card's x=53 margin, but a
    composer-bound lookup must instead find the nearest eligible x=95 photo above its own comment
    field.  Looking at every eligible run also recovers bright photos at the narrower measured
    inline width floor.

    This is not a looser generic-screen locator.  It is reachable only with an independently
    detected comment field and CTA, requires the preview to align with that field and sit within
    the existing card-to-field gap bound, and leaves the content/signature verification below
    unchanged.  An upper card can therefore neither hide the real preview nor license a match.
    """
    comment = getattr(composer_surface, "comment_rect", None)
    if comment is None:
        raise SheetVerificationError(
            "the supplied inline-composer surface has no comment rectangle for preview lookup")
    slack = max(8, round(comment.width * 0.03))
    minimum_width = round(comment.width * _INLINE_COMPACT_MIN_WIDTH_FRACTION)
    gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SheetVerificationError(
            f"the like-sheet frame did not decode as an image ({len(frame)} bytes), so there is "
            "nothing to verify the chosen item against")
    height, width = gray.shape
    lo, hi = _MARGIN_PROBE
    if width < 2 * hi + 1 or hi <= lo:
        raise SheetVerificationError(
            f"a {width}px-wide frame leaves no margin strip at columns {lo}..{hi} to read a "
            "background reference from, so the inline preview cannot be bounded")
    spans = _row_spans(
        gray, np, tolerance=_ROW_BACKGROUND_TOLERANCE, probe=_MARGIN_PROBE)
    runs: list[tuple[int, int]] = []
    start = None
    for y in range(_PREVIEW_SEARCH_TOP_PX, min(height, comment.y0)):
        span = spans[y]
        wide = span is not None and span[1] - span[0] >= minimum_width
        if wide and start is None:
            start = y
        elif not wide and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, min(height, comment.y0)))

    candidates: list[tuple[int, SheetPreview]] = []
    for y0, y1 in runs:
        if y1 - y0 < _PREVIEW_MIN_HEIGHT_PX:
            continue
        band = [spans[y] for y in range(y0, y1) if spans[y] is not None]
        x0 = int(np.median([span[0] for span in band]))
        x1 = int(np.median([span[1] for span in band]))
        gap = comment.y0 - y1
        max_gap = max(40, round((y1 - y0) * 0.25))
        if (abs(x0 - comment.x0) <= slack
                and abs(x1 - comment.x1) <= slack
                and 0 <= gap <= max_gap):
            candidates.append((gap, SheetPreview(
                y0=y0, y1=y1, x0=x0, x1=x1,
                reason=(f"inline-composer-bound run of rows at least {minimum_width}px wide, "
                        f"{y1 - y0} rows tall, aligned to comment field "
                        f"x={comment.x0}..{comment.x1}, with a {gap}px field gap"))))
    if not candidates:
        raise SheetVerificationError(
            "the selected-card preview is not immediately above the independently detected "
            f"inline comment field: no candidate at least {minimum_width}px wide and "
            f"{_PREVIEW_MIN_HEIGHT_PX}px tall is aligned with it")
    _gap, preview = min(candidates, key=lambda candidate: candidate[0])
    return SheetPreview(
        y0=preview.y0, y1=preview.y1, x0=preview.x0, x1=preview.x1,
        reason=(f"inline-composer compact fallback: {preview.reason}; minimum width "
                f"{minimum_width}px bound to comment x={comment.x0}..{comment.x1}"))


def _extend_inline_preview_to_block_bottom(
        frame: bytes, preview: SheetPreview, composer_surface, *, cv2, np,
        background_tolerance: float = _ROW_BACKGROUND_TOLERANCE,
        margin_probe: tuple[int, int] = _MARGIN_PROBE) -> SheetPreview:
    """Carry a located preview's BOTTOM edge down to where the selected photo actually ends.

    `locate_sheet_preview` ends its run at the first row whose non-background span falls under
    the absolute `_PREVIEW_MIN_WIDTH_PX` floor, and that floor is [corpus, 6 real sheets] 870,
    chosen with only 20px of headroom under the narrowest preview measured (890). REAL PHOTO
    CONTENT SPENDS THAT HEADROOM. On the refused Hinge 10.0.1 composer frame (frame_sha256
    29175f91..., 1080x2400, `ops/calibration/targeting_20260822T_cal-a/`) the selected photo's
    bright car window reaches the right edge, so 29 of the block's 856 rows measure only
    849..868px of non-background span -- never background, just a few pixels shy of 870 -- and
    the strict floor splits ONE contiguous image into the runs 236..856, 857..862, 875..883,
    893..902, 903..1088. The locator returns the first run at least `_PREVIEW_MIN_HEIGHT_PX`
    tall, which is 236..856: a 620px FRAGMENT of an 856px photo, reported with every appearance
    of success.

    Two things then go wrong, and only the first is visible. The topology check below measures
    the gap from that fragment's bottom to the comment field and gets 268px against a 155px
    allowance, so a completely ordinary composer is refused. Had it passed, the CONTENT
    comparison would have been worse: `_compare_item` derives the source window from
    `preview.height`, so a 620px preview asks for 620/856 of the rows the sheet is really
    showing and compares the wrong window.

    Under an independently detected composer the bottom edge does not have to be inferred from a
    width floor at all -- the layout defines it as the end of the contiguous image block sitting
    above the comment field. So walk down from the located bottom while the rows are still image
    rows: at least `_INLINE_COMPACT_MIN_WIDTH_FRACTION` of the composer's OWN comment width (the
    same composer-bound evidence rule `_locate_inline_compact_preview` already uses) and still
    left-aligned with the located preview. Stop at the first background row, or at the comment
    field itself.

    THE TOP EDGE NEVER MOVES, AT MOST THE MEASURED THREE-ROW INTERNAL INTERRUPTION IS BRIDGED,
    EVERY NON-BACKGROUND ROW KEEPS EITHER THE LEFT EDGE OR THE RIGHT EDGE AND AT LEAST THE
    MEASURED 74% OF PHOTO STRUCTURE, AND THE WALK CANNOT REACH PAST THE COMMENT FIELD, so this
    cannot wander onto a different element to manufacture adjacency: it can only finish the
    block the strict locator had already started inside. A preview that genuinely stops short
    of the composer -- a partially rendered view, a different sheet -- has more than three
    background rows between it and the field, the walk halts there, and the topology check still
    refuses. Columns are left as located: they are the median of the strict band, which is the
    higher-confidence evidence, and the added rows had to retain one of those edges to be walked
    at all.

    A no-op after `_locate_inline_compact_preview`, which already scans at this same width floor
    and therefore already ends at a background row.
    """
    comment = getattr(composer_surface, "comment_rect", None)
    if comment is None:
        return preview
    gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        # `locate_sheet_preview` decoded this same frame moments ago; an undecodable frame is
        # its refusal to report, not this helper's.
        return preview
    limit = min(gray.shape[0], comment.y0)
    if preview.y1 >= limit:
        return preview
    spans = _row_spans(gray, np, tolerance=background_tolerance, probe=margin_probe)
    minimum_width = round(comment.width * _INLINE_COMPACT_MIN_WIDTH_FRACTION)
    one_edge_minimum_width = round(
        comment.width * _INLINE_PREVIEW_ONE_EDGE_MIN_WIDTH_FRACTION)
    slack = max(8, round(comment.width * 0.03))
    y1 = preview.y1
    scan_y = y1
    last_supported_y1 = y1
    unsupported_run = 0
    bridged_rows = 0
    bridged_runs = 0
    one_edge_rows = 0
    while scan_y < limit:
        span = spans[scan_y]
        span_width = 0 if span is None else span[1] - span[0]
        left_aligned = span is not None and abs(span[0] - preview.x0) <= slack
        right_aligned = span is not None and abs(span[1] - preview.x1) <= slack
        fully_supported = span_width >= minimum_width and left_aligned
        # Pale content may erase either visible edge against the page, but it cannot erase both:
        # this lower, composer-only floor requires the other edge to remain exactly attached to
        # the strict preview we already found.  It does not bridge blank page rows (span=None).
        one_edge_supported = (
            span_width >= one_edge_minimum_width and (left_aligned or right_aligned))
        supported = fully_supported or one_edge_supported
        if supported:
            if one_edge_supported and not fully_supported:
                one_edge_rows += 1
            if unsupported_run:
                bridged_runs += 1
                bridged_rows += unsupported_run
            unsupported_run = 0
            last_supported_y1 = scan_y + 1
        else:
            unsupported_run += 1
            if unsupported_run > _INLINE_PREVIEW_MAX_INTERNAL_GAP_PX:
                break
        scan_y += 1
    y1 = last_supported_y1
    if y1 == preview.y1:
        return preview
    # `bridged_rows` counts only interruptions followed by more supported image rows; terminal
    # page background is never included in the preview.
    if (bridged_runs > 1
            or bridged_rows > _INLINE_PREVIEW_MAX_INTERNAL_GAP_PX):
        # Each individual run is bounded above, but only one measured interruption is
        # calibrated.  Failing closed here also prevents a future loop change from silently
        # widening the seam by accepting several individually-small gaps.
        return preview
    return SheetPreview(
        y0=preview.y0, y1=y1, x0=preview.x0, x1=preview.x1,
        reason=(f"{preview.reason}; bottom edge carried from row {preview.y1} to {y1}, the end "
                f"of the >={minimum_width}px image block above the composer's comment field at "
                f"row {comment.y0}, retaining one aligned edge across {one_edge_rows} rows and "
                f"bridging {bridged_rows} measured internal background rows"))


def _decode_crop(crop_bytes: bytes, number: int, cv2, np):
    """The stored crop as one greyscale array, decoded exactly the way `signature_of` decodes.

    Decoded ONCE per verification rather than once per swept window: a sweep is tens of windows
    per item and every one of them would otherwise re-decode a ~1MB PNG, which is seconds of ADB
    dead time in the middle of an open comment sheet.

    `cv2.IMREAD_GRAYSCALE`, byte for byte the call `item_crops.signature_of` makes, and the
    windows below are reduced by `item_crops._signature_from_gray`, which is that module's single
    definition of what a signature IS. Both are reached through `item_crops` rather than
    reimplemented here because the two obvious ways to get grey out of a PNG measure 1.46 levels
    apart on average and up to 9.8 apart on a signature -- where the whole distance between the
    two most alike items on a profile is 4.66 -- so a second decode path would put the two sides
    of this comparison in different units.
    """
    if not crop_bytes:
        # A CROP_ITEM carries an image by construction (`ItemPayload.signature_for` says so), so
        # this is a hand-assembled payload rather than a real one. Named rather than left to fail
        # inside numpy, because "there is no reference for this item" and "the reference does not
        # match" are different problems with the same consequence.
        raise SheetVerificationError(
            f"model item {number} carries no stored crop, so the sheet cannot be verified "
            "against it")
    gray = cv2.imdecode(np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SheetVerificationError(
            f"the stored crop for item {number} did not decode as an image "
            f"({len(crop_bytes)} bytes), so there is nothing to verify the sheet against")
    return gray


def _compare_item(crop, gray, preview: SheetPreview, sheet: CropSignature, *,
                  grid: tuple[int, int], scale_tolerance: float, cv2, np,
                  inline_reframe: bool = False):
    """One item measured against the sheet: its swept distance, its window, and its reference.

    Returns `(distance_or_None, window_px, reference_signature, reason)`. The reference is the
    UNSWEPT window, because it is what the other items are measured against for `nearest_other`
    and a swept-per-pair reference would not be one fixed quantity.

    A REFERENCE IS PRODUCED FOR EVERY ITEM, INCLUDING ONE TOO SHORT TO BE THE RENDERED WINDOW,
    AND THAT IS THE FIX FOR A MEASURED FALSE ACCEPT. `distance` stays None for such an item --
    it genuinely cannot be what is on the sheet -- but it must still take part in every OTHER
    item's `nearest_other`, because that quantity is what the accept bound is derived from. When
    it did not, a tall sheet pruned the short items out of the neighbour set and every surviving
    item's bound grew accordingly: on the calibration corpus one item's bound came out 27.6 where
    its own stored neighbour distance implies ~7, and a card belonging to a DIFFERENT PROFILE
    ENTIRELY verified as it, 10 times out of 540 (`verify_sheet_item`'s "WHAT THIS IS NOT" for
    what still does not follow from fixing it). The neighbour set is a property of the payload;
    it must not depend on how tall the sheet in front of us happens to be.
    """
    crop_height, crop_width = int(gray.shape[0]), int(gray.shape[1])
    source_x1 = crop_width
    if inline_reframe:
        source_x1 -= round(crop_width * _INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
        if source_x1 <= round(crop_width * 0.70):
            return (None, 0, None,
                    "inline composer verification would leave too little source width after "
                    "excluding the profile-card heart control lane")
        crop_width = source_x1
    scale = crop_width / preview.width
    window_px = int(round(preview.height * scale))
    tolerance = max(1, int(round(window_px * scale_tolerance)))

    def _window(rows: int, *, start: int | None = None) -> CropSignature:
        # The historical sheet exposes the bottom of a card.  Inline composer reframe mode
        # supplies an explicit source start below; both routes still use the exact same
        # grayscale/signature implementation.
        if start is None:
            start = crop_height - rows
        return _signature_from_gray(gray[start:start + rows, :source_x1], grid=grid,
                                    cv2=cv2, np=np)

    # As much of this card as the sheet would be able to show. For a card at least as tall as the
    # window that is the bottom `window_px` rows; for a shorter one it is the whole card, which is
    # the same rule with the same `min` the line below already applied.
    reference_rows = min(window_px, crop_height)
    # Neighbour separation must not depend on which candidate happens to fit THIS screenshot.
    # Keep one deterministic centered reference for each item; only intended-item distance may
    # search the tightly bounded inline reframe offsets below.
    reference = _window(reference_rows, start=(crop_height - reference_rows) // 2
                        if inline_reframe else None)
    if window_px - tolerance > crop_height:
        return (None, window_px, reference,
                f"the sheet is rendering {window_px}px of card and item {crop.number} is only "
                f"{crop_height}px tall, so this item cannot be what is on screen (it still bounds "
                f"every other item)")

    if not inline_reframe:
        best = None
        for candidate in range(window_px - tolerance, window_px + tolerance + 1):
            if not 0 < candidate <= crop_height:
                continue
            distance = sheet.distance(_window(candidate))
            best = distance if best is None else min(best, distance)
        return (best, window_px, reference,
                f"bottom {window_px}px of a {crop_height}px crop, swept +-{tolerance}px")

    # This is deliberately not a generic patch search. Every candidate keeps the full card
    # width and all but a bounded edge strip, and a changed layout that hides more than 30% of
    # the source simply fails. It exists only for a structurally proven inline composer; the
    # normal modal comparison above retains bottom anchoring exactly.
    nominal_hidden = crop_height - window_px
    if nominal_hidden > round(crop_height * _INLINE_REFRAME_MAX_HIDDEN_FRACTION):
        return (None, window_px, reference,
                f"inline composer preview would hide {nominal_hidden}px of this {crop_height}px "
                f"crop, above the {_INLINE_REFRAME_MAX_HIDDEN_FRACTION:.0%} reframe limit")
    best_distance = None
    best_start = None
    best_rows = None
    # The old corpus needed only a 2% scale tolerance. Nine evenly spaced heights retain that
    # envelope while bounding this compositor-only check to roughly 9×49 full-width windows per
    # item rather than exhaustively trying every row/height combination.
    height_lo = window_px - tolerance
    height_hi = window_px + tolerance
    height_step = max(1, math.ceil((height_hi - height_lo) / 8))
    candidates = list(range(height_lo, height_hi + 1, height_step))
    if window_px not in candidates:
        candidates.append(window_px)
    if height_hi not in candidates:
        candidates.append(height_hi)
    for candidate in candidates:
        if not 0 < candidate <= crop_height:
            continue
        max_start = crop_height - candidate
        if max_start > round(crop_height * _INLINE_REFRAME_MAX_HIDDEN_FRACTION):
            continue
        step = max(1, math.ceil(max_start / max(1, _INLINE_REFRAME_ORIGIN_SAMPLES - 1)))
        starts = list(range(0, max_start + 1, step))
        if starts[-1] != max_start:
            starts.append(max_start)
        for start in starts:
            candidate_reference = _window(candidate, start=start)
            distance = sheet.distance(candidate_reference)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_start = start
                best_rows = candidate
    if best_distance is None:
        return (None, window_px, reference,
                "inline composer reframe offered no bounded full-width source window")
    return (best_distance, window_px, reference,
            f"inline reframe rows {best_start}..{best_start + best_rows} of a {crop_height}px "
            f"crop, width-derived {window_px}px swept +-{tolerance}px")


def verification_blocker(payload: ItemPayload, model_index: int, *,
                         sheet_render_drift: float = _SHEET_RENDER_DRIFT,
                         separation_fraction: float = _SEPARATION_FRACTION) -> str:
    """Why item `model_index` cannot be verified on the sheet at all, or "" when it can.

    CALL THIS BEFORE THE TAP. Doc 5.2's addendum ends "5.6 decides what an unverifiable item may
    be used for", and this is that decision: an item whose stored crop cannot serve as a reference
    may not be liked, because the only alternatives are to substitute a different item (forbidden)
    or to send an opener under an item nobody checked (the failure this whole redesign exists to
    stop). Refusing here rather than after the tap costs nothing and leaves the screen untouched.

    The test is the accept bound read backwards. Verification can only work if the noise on a
    CORRECT reading is smaller than the bound a correct reading has to come under:

        sheet_render_drift + this item's own re-observation drift  <  separation_fraction * nearest

    where both `drift` and `nearest` are the numbers `item_crops` already measured per item.

    THIS IS THE STRICTER OF THE TWO TESTS ONLY BECAUSE BOTH NOW COUNT THE SAME ITEMS, AND THAT
    SENTENCE USED TO BE FALSE. The claim rests on `nearest` being measured over the payload's
    whole item list at 32x32 while `verify_sheet_item` re-measures in window space at a finer grid
    where the same separations are roughly twice as large. That held for the grid term and NOT for
    the item term: until 2026-08-12 `verify_sheet_item` dropped every item too short to be the
    rendered window out of its neighbour set, so a short card's true nearest neighbours could
    vanish and the post-tap bound came out up to 4x LOOSER than the prediction here -- the ordinary
    case for a short card, not a corner. `_compare_item` now produces a reference for every item,
    so the two tests range over the same list again and this really does err towards refusing an
    item that would in fact have verified -- a stop, never a pass.

    [corpus: over both calibration profiles' 18 items this refuses exactly one, the animated card,
    whose 25.009 drift plus 1.95 exceeds half of its 47.861 nearest-neighbour distance. The
    tightest pass has 0.36 grey levels of margin.]

    An UNMEASURED drift (`signature_drift is None`, i.e. no frame ever re-observed that crop) is
    treated as zero HERE and only here. That is not "silence read as stability" -- the doc's rule
    is about never REPORTING an unmeasured crop as separable, which `ItemCrop.separable` honours
    by staying None. This is a prediction about whether the post-tap check can pass, the post-tap
    check is the authority and is unaffected, and refusing every never-re-observed item would
    refuse one item per profile on no evidence at all.

    Raises `SheetVerificationError` for an unusable payload or an out-of-range number, on
    `ItemPayload.item`'s reasoning: those are not verdicts.
    """
    try:
        crop = payload.item(model_index)
    except ItemCropError as exc:
        raise SheetVerificationError(
            f"cannot check whether model item {model_index} is verifiable: {exc}") from exc
    nearest = crop.nearest_item_distance
    if nearest is None:
        # One numbered item and nothing to confuse it with. `verify_sheet_item` falls back to the
        # reproduction bound there, which is a real check, so this is not a blocker.
        return ""
    drift = crop.signature_drift or 0.0
    bound = separation_fraction * nearest
    noise = sheet_render_drift + drift
    if noise >= bound:
        return (
            f"model item {model_index} cannot be verified on the like sheet: its own stored crop "
            f"drifts {drift:.3f} grey levels between two observations of the same scroll "
            f"position, and the sheet re-renders a card {sheet_render_drift:.2f} away from it, "
            f"for {noise:.3f} of expected noise against a {bound:.3f} bound "
            f"({separation_fraction:g} x its {nearest:.3f} distance to the nearest other item). A "
            f"tolerance wide enough to accept it is wide enough to accept a different item, and "
            f"doc 5.6 forbids substituting one, so nothing is tapped")
    return ""


def verify_sheet_item(frame: bytes, payload: ItemPayload, model_index: int, *,
                      grid: tuple[int, int] = _VERIFY_GRID,
                      scale_tolerance: float = _SCALE_TOLERANCE,
                      separation_fraction: float = _SEPARATION_FRACTION,
                      sheet_render_drift: float = _SHEET_RENDER_DRIFT,
                      absolute_max_dist: float | None = None,
                      composer_surface=None,
                      **locate_kwargs) -> SheetVerdict:
    """Is the open comment sheet in `frame` showing model item `model_index`? Deterministic.

    CALL THIS AFTER THE TAP AND BEFORE ANY TYPING. It never taps, types, sends or repairs; it
    reports, and `SheetVerdict.matched` is the only thing that licenses the opener being typed.

    Every numbered item is measured, not just the chosen one, because the nearest-match is what
    turns "this is not item 3" into "this is item 4" -- doc 5.6 asks for intended AND actual to be
    recorded on a miss, and `nearest_index` is the actual.

    The verdict is MATCH only when the sheet's distance to item `model_index` is under
    `separation_fraction` x that item's distance to the nearest other item, measured in the same
    window space at the same grid. When `absolute_max_dist` is supplied, a MATCH also has to be
    strictly below that calibrated ceiling; closed-set separation alone cannot reject a foreign
    card that happens to resemble every stored crop. By the triangle inequality the relative
    test also makes it the unique nearest item, so the argmin is checked as corroboration and
    reported either way.

    Raises `SheetVerificationError` when it could not look at all -- no preview on the frame, an
    unusable payload, an out-of-range number, undecodable bytes, or missing vision extras -- or
    when a supplied absolute ceiling is not finite/positive and strictly below the known 14.91
    foreign-card collision. A caller must treat either case exactly as it treats a mismatch.
    """
    # ``None`` deliberately preserves this leaf's offline relative-comparison API.  Once a
    # caller supplies an absolute ceiling, however, it is security evidence rather than a
    # cosmetic tolerance: NaN makes ``distance >= ceiling`` false and infinity makes it always
    # false, both silently erasing the foreign-card guard below.
    if absolute_max_dist is not None:
        if (isinstance(absolute_max_dist, bool)
                or not isinstance(absolute_max_dist, (int, float))
                or not math.isfinite(absolute_max_dist)
                or absolute_max_dist <= 0
                or absolute_max_dist >= _SHEET_FALSE_MATCH_DISTANCE):
            raise SheetVerificationError(
                "absolute_max_dist must be a finite positive number strictly below the known "
                f"{_SHEET_FALSE_MATCH_DISTANCE} foreign-card false-match distance")

    try:
        chosen = payload.item(model_index)
    except ItemCropError as exc:
        raise SheetVerificationError(
            f"cannot verify model item {model_index} against this payload: {exc}") from exc

    cv2, np = _require_vision()
    try:
        preview = locate_sheet_preview(frame, **locate_kwargs)
    except SheetVerificationError:
        # Do not lower the public/legacy locator's evidence threshold.  An independently proven
        # inline composer supplies a stronger local boundary: reviewers may scroll an unrelated
        # x=53 card above the still-open selected preview, and bright photos can fall under the
        # public width floor. Bind the fallback to the fresh composer instead of letting either
        # condition mask the target.
        if composer_surface is None:
            raise
        preview = _locate_inline_composer_preview(
            frame, composer_surface, cv2=cv2, np=np)
    if composer_surface is not None:
        # The located run is where the preview STARTS; under a proven composer its bottom edge is
        # the end of the image block, not the first row a legacy width floor stumbles on. See
        # `_extend_inline_preview_to_block_bottom` -- a real 10.0.1 composer was refused below
        # because a bright photo edge cut an 856px preview down to a 620px fragment.
        preview = _extend_inline_preview_to_block_bottom(
            frame, preview, composer_surface, cv2=cv2, np=np,
            background_tolerance=locate_kwargs.get(
                "background_tolerance", _ROW_BACKGROUND_TOLERANCE),
            margin_probe=locate_kwargs.get("margin_probe", _MARGIN_PROBE))
        # Hinge 9.134 moved this UI inline.  The legacy preview geometry remains useful for
        # content comparison, but no longer proves by itself that a composer exists: an ordinary
        # card can share the same inset.  Bind the selected-card crop to an independently
        # detected composer and require the measured card -> field -> CTA topology.
        comment = getattr(composer_surface, "comment_rect", None)
        send = getattr(composer_surface, "send_rect", None)
        if comment is None or send is None:
            raise SheetVerificationError(
                "the supplied inline-composer surface has no comment/send rectangles")
        width_slack = max(8, round(preview.width * 0.03))
        gap = comment.y0 - preview.y1
        max_gap = max(40, round(preview.height * 0.25))
        controls_gap = send.y0 - comment.y1
        if (abs(preview.x0 - comment.x0) > width_slack
                or abs(preview.x1 - comment.x1) > width_slack
                or not 0 <= gap <= max_gap
                or not 0 <= controls_gap <= max(60, round(comment.height * 0.40))):
            raise SheetVerificationError(
                "the selected-card preview is not immediately above the independently detected "
                "inline comment field and Send Like CTA "
                f"(preview=({preview.x0},{preview.y0})-({preview.x1},{preview.y1}), "
                f"comment=({comment.x0},{comment.y0})-({comment.x1},{comment.y1}), "
                f"send=({send.x0},{send.y0})-({send.x1},{send.y1}), "
                f"photo-to-field gap={gap}px with allowed 0..{max_gap}px, "
                f"field-to-CTA gap={controls_gap}px)")
    try:
        sheet = signature_of(frame, y0=preview.y0, y1=preview.y1, x0=preview.x0, x1=preview.x1,
                             grid=grid)
    except ItemCropError as exc:
        raise SheetVerificationError(
            f"the sheet's item preview at rows {preview.y0}..{preview.y1}, columns "
            f"{preview.x0}..{preview.x1} could not be reduced to a signature: {exc}") from exc

    comparison_preview = preview
    if composer_surface is not None:
        # The profile-card heart is an overlaid control, not photo content. The selected inline
        # preview correctly omits it, so cut that fixed right lane from both representations
        # before comparing. Preserve ``preview`` itself in the verdict as the full inspected
        # geometry/audit record.
        right_lane = round(preview.width * _INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
        if preview.width - right_lane <= round(preview.width * 0.70):
            raise SheetVerificationError(
                "inline selected-photo preview leaves too little width after excluding the "
                "profile-card heart-control lane")
        comparison_preview = SheetPreview(
            y0=preview.y0, y1=preview.y1, x0=preview.x0, x1=preview.x1 - right_lane,
            reason=preview.reason + "; inline right control lane excluded")
        try:
            sheet = signature_of(frame, y0=comparison_preview.y0, y1=comparison_preview.y1,
                                 x0=comparison_preview.x0, x1=comparison_preview.x1, grid=grid)
        except ItemCropError as exc:
            raise SheetVerificationError(
                "the inline selected-photo content rect could not be reduced to a signature") from exc

    measured: list[tuple] = []
    for crop in payload.items:
        gray = _decode_crop(crop.image, crop.number, cv2, np)
        distance, window_px, reference, reason = _compare_item(
            crop, gray, comparison_preview, sheet, grid=grid, scale_tolerance=scale_tolerance, cv2=cv2,
            np=np, inline_reframe=(composer_surface is not None))
        measured.append((crop, distance, window_px, reference, reason, int(gray.shape[0])))

    # EVERY numbered item, without exception. An item too short to be the rendered window is still
    # a stored item this profile can be confused with, and dropping it here is what let a foreign
    # card verify (see `_compare_item`). The bound below has to be a property of the payload, not
    # of the sheet's height.
    references = {c.number: r for c, _d, _w, r, _rs, _h in measured if r is not None}
    comparisons: list[ItemComparison] = []
    for crop, distance, window_px, reference, reason, crop_px in measured:
        nearest_other = None
        if reference is not None:
            others = [reference.distance(r) for n, r in references.items() if n != crop.number]
            nearest_other = min(others) if others else None
        # With a neighbour the bound is the separation scaled DOWN by the fraction (half of it,
        # which is what makes the nearest match unique). With none -- a one-item list, or one
        # where every other item is too short to be what the sheet is rendering -- there is no
        # separation to halve, so the only quantity left is what a CORRECT reading costs, and the
        # same fraction scales it UP instead.  Legacy/modal mode retains that historic rule.  A
        # structurally proven inline composer has a distinct Hinge renderer/reframe and therefore
        # uses its own held-out one-item ceiling; it never widens a multi-item relative proof.
        # Production's independently calibrated absolute ceiling remains a second, stricter cap.
        bound = (separation_fraction * nearest_other if nearest_other is not None
                 else (sheet_render_drift + (crop.signature_drift or 0.0)) / separation_fraction)
        if composer_surface is not None and nearest_other is None:
            bound = _INLINE_COMPOSER_ONE_ITEM_MAX_DIST
        comparisons.append(ItemComparison(
            number=crop.number, distance=distance, window_px=window_px, crop_px=crop_px,
            nearest_other=nearest_other, bound=bound, reason=reason))
    by_number = {c.number: c for c in comparisons}
    mine = by_number[model_index]

    reachable = [c for c in comparisons if c.distance is not None]
    nearest_index = (min(reachable, key=lambda c: c.distance).number if reachable else None)

    def _verdict(state: str, reason: str) -> SheetVerdict:
        return SheetVerdict(
            state=state, model_index=model_index, nearest_index=nearest_index,
            distance=mine.distance, bound=mine.bound, comparisons=tuple(comparisons),
            preview=preview, grid=grid, reason=reason)

    # UNVERIFIABLE first: when the bound is not wide enough to hold a correct reading, the
    # numbers below decide nothing, and reporting a MATCH off them would be the substitution doc
    # 5.6 forbids dressed up as a measurement. `verification_blocker` normally catches this
    # before the tap, off the payload's own coarser numbers; this is the same test re-made in the
    # space the comparison actually happened in.
    noise = sheet_render_drift + (chosen.signature_drift or 0.0)
    if mine.bound is not None and mine.nearest_other is not None and noise >= mine.bound:
        return _verdict(VERIFY_UNVERIFIABLE, (
            f"model item {model_index} cannot be told from its neighbours on a like sheet: it "
            f"sits {mine.nearest_other:.3f} grey levels from the nearest other item in the window "
            f"the sheet renders, against {noise:.3f} of expected noise "
            f"({sheet_render_drift:.2f} for the sheet's own re-render plus "
            f"{chosen.signature_drift or 0.0:.3f} of the crop's own drift). Nothing is typed"))
    if mine.distance is None:
        # `nearest_index` is None when NOTHING in the payload could be measured against this
        # sheet -- not when some other item was the nearest. Those are different facts and the
        # operator gets this sentence verbatim: the 2026-08-15 observe run printed "The nearest
        # stored item is None", which reads as an item named None rather than as "no comparison
        # was possible at all". Say which one it is.
        nearest = (f"The nearest stored item is {nearest_index}" if nearest_index is not None
                   else "No stored item could be measured against this sheet either")
        return _verdict(VERIFY_MISMATCH, (
            f"the sheet is not showing model item {model_index}: {mine.reason}. {nearest}"))
    if nearest_index != model_index:
        other = by_number[nearest_index]
        return _verdict(VERIFY_MISMATCH, (
            f"the like sheet is showing model item {nearest_index}, not the item {model_index} "
            f"the opener was written about: the sheet sits {other.distance:.3f} grey levels from "
            f"item {nearest_index}'s crop and {mine.distance:.3f} from item {model_index}'s. "
            f"Nothing is sent and the sheet is left open"))
    # How the bound was arrived at, spelled out in both the accept and the reject message: with a
    # neighbour it is half the distance to it, and with none (a one-item list, or a list where
    # every other item is too short to be what the sheet is rendering) there is no neighbour to
    # halve, so the only defensible bound left is the sheet's own measured reproduction noise.
    derivation = (
        f"{separation_fraction:g} x its {mine.nearest_other:.3f} distance to the nearest other "
        f"item" if mine.nearest_other is not None else
        (f"the inline composer's independently held-out {mine.bound:.2f} one-item render ceiling, "
         f"there being no other item to bound against" if composer_surface is not None else
         f"the sheet's own {sheet_render_drift:.2f} reproduction noise plus this crop's "
         f"{chosen.signature_drift or 0.0:.3f} drift over {separation_fraction:g}, there being no "
         f"other item to bound against"))
    if mine.distance >= mine.bound:
        return _verdict(VERIFY_MISMATCH, (
            f"the like sheet does not show model item {model_index}: item {model_index} is the "
            f"nearest stored crop but at {mine.distance:.3f} grey levels, against a "
            f"{mine.bound:.3f} bound ({derivation}). Whatever is on the sheet is not any item "
            f"this profile was indexed with. Nothing is sent and the sheet is left open"))
    if absolute_max_dist is not None and mine.distance >= absolute_max_dist:
        return _verdict(VERIFY_MISMATCH, (
            f"the like sheet is too far from model item {model_index} to confirm: "
            f"{mine.distance:.3f} grey levels is within its relative {mine.bound:.3f} bound "
            f"but not below the calibrated absolute {absolute_max_dist:.3f} ceiling. Nothing "
            f"was typed and the sheet is left open"))
    return _verdict(VERIFY_MATCH, (
        f"the like sheet shows model item {model_index}: {mine.distance:.3f} grey levels from its "
        f"stored crop, against a {mine.bound:.3f} bound ({derivation}), and nearer than every "
        f"other numbered item this profile was indexed with"))
