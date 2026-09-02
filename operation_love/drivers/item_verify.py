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

  THOSE COMMENT-BOX COLUMNS ARE ITS FIRST BORDER ROW AND ARE NOT WHAT `ComposerSurface` CARRIES,
  which matters because the pillarbox regime derives a scale from the field's width. The box is a
  rounded rect, so its topmost row is inset: row 1124 reads 117..963 and by row 1146 it has
  widened to 95..985. `like_composer._comment_for_cta` reports the enclosing component's OUTER
  bounds, i.e. 95..985 (890px) -- the same content column as the preview above it, which is the
  point. Building anything on the 846 would put a 974/846 = 1.151 scale where 974/890 = 1.094
  belongs, roughly 28x the +-2% the window sweep allows.

  A PREVIEW'S INK DOES NOT HAVE TO FILL THAT COLUMN. All six of these are full-bleed photo cards,
  where the photograph happens to run the card edge to edge; a card whose photo is portrait is
  rendered whole into the same column with the photo centred inside it. See
  `_INLINE_PILLARBOX_MIN_WIDTH_FRACTION`.

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
# selected-photo renderer.  It is used ONLY for the one-item fallback: a multi-item payload
# continues to require the triangle-inequality/unique-nearest separation proof below.  Production
# also supplies its independently frozen ``absolute_max_dist``; this ceiling cannot relax that.
#
# RE-FITTED 2026-08-28, BECAUSE THE UNITS IT WAS FITTED IN NO LONGER EXIST.  The previous value
# was 7.00, justified by the first two held-out inline renders at 3.729 (Shai) and 6.703
# (Malaika).  BOTH OF THOSE FIGURES WERE DECODE BIAS, not render penalty: they were measured while
# the crop reference was recovered from a colour re-encode that had lost the device PNG's ``sRGB``
# chunk, so the two sides of every comparison were in different grey spaces (see `_decode_crop`).
# The frames behind them are not preserved, so this is inference rather than re-measurement -- but
# the direction is not in doubt, and the replacement is measured on real device data.
#
# [measured 2026-08-28 over every recoverable like-sheet verification in data/hinge_debug --
# 148 sheets across 17 runs, each re-verified through the real `verify_sheet_item` /
# `_compare_item` / `locate_inline_composer` path, and validated by reproducing production's own
# logged 10.283:
#   * CORRECT card, corrected space:  min 0.068, median 0.249, p90 0.963, MAX 1.380
#   * CORRECT card, defective space:  min 0.733, median 1.969, max 10.283, and 9 of 148 (6.1%)
#     at or above the old 7.00 -- the 2026-08-27 halt was one of nine, not a one-off
#   * FOREIGN card (218 cross-profile pairs): min 40.823, median 70.535, max 111.698
# One 219th pair scored 0.174 and was hand-checked: it is the SAME profile captured in two
# different runs, i.e. a correct match, and it is excluded from the foreign set rather than
# quietly kept as a flattering minimum.  It is also independent evidence the check works across
# sessions.]
#
# 3.00 sits 2.17x above the worst of 148 real correct readings and 13.6x below the nearest real
# foreign card, and it refuses NONE of the 148.  The point of moving it is not reject power
# against a different card -- 7.00 already had 5.8x of that -- it is that this ceiling is also the
# only test of whether the reading is TRUSTWORTHY AT ALL in the one-item regime, where there is no
# separation proof.  At roughly 0.7-0.9 grey levels per pixel of preview-rect error, 7.00 accepts
# a preview mislocated by ~8px; 3.00 accepts ~2px, against the +-1px actually observed.
#
# LOWERING THIS CANNOT CREATE A FALSE ACCEPT.  It feeds exactly one expression,
# ``if mine.distance >= mine.bound`` below, so a smaller value can only turn accepts into
# refusals.  The cost it does carry is availability: a run that would have squeaked through now
# halts with the sheet open.  That is the fail-loud direction, and one constant reverts it.
_INLINE_COMPOSER_ONE_ITEM_MAX_DIST = 3.00
# The verification reference has to reproduce the signature `item_crops` stored for the SAME rows
# at the SAME grid, and "reproduce" here means exactly: `ItemCrop.verify_image` is a lossless
# 8-bit greyscale PNG cut from the frame's own greyscale decode, so it decodes back bit for bit
# and the honest expected value of this check is 0.000. The tolerance is not a margin for the two
# to differ in -- it is a guard against a future lossless-but-not-bit-identical encoder turning a
# provenance check into a production halt, and it is far below anything that could hide the fault
# it exists to catch [measured 2026-08-27: the colour-PNG path this replaced scored 10.158 here].
_REFERENCE_PROVENANCE_MAX_DIST = 0.25
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
# Hinge 10.1.0 can insert a horizontally scrolling shelf of suggested replies between a short
# selected-photo preview and the comment field.  The 2026-09-01 held-out frame measured a 550px
# preview, an 89px shelf with 32px gutters, and therefore a 153px photo-to-field gap.  That same
# renderer shows 602 of a 974px square card (38.2% hidden), beyond the ordinary inline renderer's
# 30% envelope below.  Neither number is permission to widen the ordinary topology: the larger
# crop is available only when `_inline_gap_evidence` independently finds the measured row of
# outlined suggestion pills between this preview and this composer.
_INLINE_SUGGESTION_MAX_HIDDEN_FRACTION = 0.40
_INLINE_SUGGESTION_MIN_PILLS = 2
# A SELECTED PHOTO DOES NOT HAVE TO FILL THE COMPOSER'S CONTENT COLUMN, AND THE SIX-SHEET CORPUS
# ABOVE CONTAINS NO EXAMPLE THAT DOES NOT.  Every one of those six is a full-bleed photo card,
# where the photograph happens to run the card edge to edge, so "the preview's ink spans the
# content column" reads like a layout invariant.  It is not one: it is a property of the CARD's
# CONTENT.  The card is always rendered into the whole column; only its ink may be inset.
#
# [measured 2026-08-28, live Pixel 7a, Hinge 10.1.0, profile `Yvonne` page heart 7, run
# `data/hinge_debug/f6a15d162e68` (gitignored -- only geometry is written down here).  The card is
# a PROMPT-PHOTO card whose photograph is portrait, so Hinge fits it to the card's height and
# centres it: a 556x932 photo inside a 974px card, leaving 209px of card background on each side.
# The sheet renders that whole card into the 890px content column at the corpus's own 0.9137
# scale, so the photo's INK spans 508px at columns 286..794 -- 191px of white card on the left and
# 191px on the right, symmetric to the pixel.  Against a 694px floor and 27px of edge slack the
# locator refused a correct, ordinary composer, and the run halted with the sheet open.]
#
# RELAXING THE WIDTH AND ALIGNMENT TESTS ALONE DOES NOT FIX THIS, AND THAT IS THE WHOLE REASON
# THE RECT AND THE INK BOX ARE NOW SEPARATE THINGS.  `_compare_item` derives
# `scale = crop_width / preview.width`, and the stored crop is always the WHOLE card.  Handing it
# the 508px ink box asks a 1109px card for a window of round(837 * 974/508) = 1604 rows, so the
# comparison refuses with "this item cannot be what is on screen" -- the same halt, one stage
# later, with a message that blames the card instead of the locator.  [measured on the frame
# above: ink box -> distance None, window_px 1604; content column -> distance 0.579 with the
# reframe search landing on source rows 185..1088 of 1109, against production's 14.91 ceiling,
# while the same card mirrored scores 21.563.]
#
# [swept 2026-08-28 over the whole local debug corpus: 98 runs, 3781 frames, 297 verify-sheet
# frames, of which 291 have a composer the real `locate_inline_composer` can find.
#   * FULL-BLEED 287.  Their ink is EXACTLY the content column -- width 890, left pad 0, right
#     pad 0, on every one of the 287.  Twelve distinct rects in total.
#   * PILLARBOXED 4, and they are ONE photograph: rows 236..1073, columns 286..794, 508px wide,
#     191px of card either side, gap 51 -- byte-identical between the two runs, because it is the
#     same profile met twice.  1 of 60 distinct sheets; the underlying CARD shape is 1 of ~1033
#     distinct cards seen (0.10%).
#   * Nothing at all lies between 0.5708 and 1.0000: a 0.43-wide empty gap.
# HEAD's locator raises on exactly those 4 frames and nowhere else in the corpus, which is the
# whole population of production halts this shape has caused -- two of them, 2026-08-27 and
# 2026-08-28, the same woman's profile both times.]
#
# THE CORPUS DOES NOT BOUND THIS FROM BELOW AND 0.40 IS NOT A MEASURED FLOOR.  n = 1 render, at
# 0.5708, and it sits near the SHALLOW end of the regime rather than at its bottom: the full-bleed
# previews run 0.83..1.15 in displayed aspect and this one is 837/508 = 1.65, so Hinge starts
# pillarboxing somewhere around 1.15 and then caps the rendered height, under which the fraction
# falls without limit -- a 9:16 upload lands near 0.54 and a 1:2 near 0.48.  0.40 clears both with
# room; a taller upload than that will still REFUSE, which is the fail-loud direction and one
# constant to revisit with a frame in hand.  It is deliberately not the discriminator on its own:
# a candidate must ALSO be at least `_PREVIEW_MIN_HEIGHT_PX` tall, contained inside the content
# column, centred in it to within the same 3% slack the flush regime aligns to, and immediately
# above the independently detected comment field.
#
# A RESIDUAL THIS REGIME DOES NOT FIX, RECORDED RATHER THAN GLOSSED: the sheet is a rigid rescale
# for a chrome-less full-bleed card and a RE-LAYOUT for a chromed prompt-photo card like this one.
# [measured 2026-08-28 at full resolution, no signature grid involved: the width-derived
# `window_px` for the Yvonne render is 916 rows and the true best-fitting window is 905 at start
# 183, so the vertical scale is 0.92486 against a horizontal 0.91376 -- 1.23% of anisotropy and 11
# rows of error. The control, a full-bleed card from the same corpus (d605b0837ac2/00140 against
# its own sheet 00143, which logged a verify_match), fits in 804 rows with 0.059% of anisotropy
# and ZERO rows of error.] `_SCALE_TOLERANCE`'s +-2% sweep absorbs those 11 rows and lands on 903,
# which is why the distance comes out at 0.579. But that constant was fitted as "the largest error
# observed is 8 rows of 804, i.e. 1.0%; 2% is that with a factor of two on it", and against a
# chromed card the factor of two is now 1.65 -- measured on ONE render. A second pillarboxed card
# with more caption chrome is the thing to measure next; if the anisotropy grows past 2% the
# symptom will be a false VERIFY_MISMATCH on a correct tap, which is fail-loud but reads as a
# targeting fault rather than as a scale one.
_INLINE_PILLARBOX_MIN_WIDTH_FRACTION = 0.40

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
    # THE RECT AND THE INK ARE THE SAME THING ONLY FOR A FULL-BLEED PHOTO, and conflating them is
    # what `_locate_inline_composer_preview`'s pillarbox regime exists to undo. `x0`/`x1` are the
    # CARD's columns -- what `_compare_item` must scale the stored crop against, because the
    # stored crop is always the whole card. `ink_x0`/`ink_x1` are where that card actually painted
    # non-background pixels, which for a portrait photo centred in its card is a narrower, inset
    # box. They are None for every preview located the historic way, and `ink_bounds` then reports
    # the rect itself, so nothing that predates the pillarbox regime changes behaviour.
    ink_x0: int | None = None
    ink_x1: int | None = None

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def pillarboxed(self) -> bool:
        """Whether the card's ink was measured to be inset inside the card's own columns."""
        return self.ink_x0 is not None and self.ink_x1 is not None

    @property
    def ink_bounds(self) -> tuple[int, int]:
        """The columns that carry image structure: the inset box, or the rect when it is flush."""
        if self.pillarboxed:
            return int(self.ink_x0), int(self.ink_x1)
        return self.x0, self.x1

    @property
    def ink_width(self) -> int:
        lo, hi = self.ink_bounds
        return hi - lo


@dataclass(frozen=True)
class _InlineGapEvidence:
    """Which measured preview-to-comment topology, if any, this frame proves."""

    accepted: bool
    max_gap: int
    suggestion_shelf: tuple[int, int, int, int] | None
    reason: str

    @property
    def max_hidden_fraction(self) -> float:
        return (_INLINE_SUGGESTION_MAX_HIDDEN_FRACTION
                if self.suggestion_shelf is not None
                else _INLINE_REFRAME_MAX_HIDDEN_FRACTION)


def _inline_gap_evidence(gray, *, preview_y1: int, preview_height: int, comment,
                         cv2, np) -> _InlineGapEvidence:
    """Prove either the ordinary direct gap or Hinge's outlined suggestion-pill shelf.

    The ordinary 25%-of-preview rule protects a still-open composer from adopting an unrelated
    card that a reviewer scrolled above it.  Keep that rule byte-for-byte.  A larger gap is valid
    only when it contains the distinct 10.1.0 shelf: at least two wide, shallow, outline-density
    components on one row, spanning most of the comment column, with independently bounded
    gutters above and below.  Text alone is too narrow, a filled bar is too dense, and blank
    whitespace supplies no components, so none can opt into the larger reframe search.
    """
    gap = comment.y0 - preview_y1
    direct_max = max(40, round(preview_height * 0.25))
    if 0 <= gap <= direct_max:
        return _InlineGapEvidence(
            True, direct_max, None,
            f"direct photo-to-field gap allowed 0..{direct_max}px")
    if gap <= 0 or preview_y1 < 0 or comment.y0 > gray.shape[0]:
        return _InlineGapEvidence(
            False, direct_max, None,
            f"photo-to-field gap allowed 0..{direct_max}px")

    band = gray[preview_y1:comment.y0]
    if not band.size:
        return _InlineGapEvidence(
            False, direct_max, None,
            f"photo-to-field gap allowed 0..{direct_max}px")

    frame_width = int(gray.shape[1])
    probe_hi = min(80, max(9, frame_width // 4))
    margin_parts = [band[:, 8:probe_hi], band[:, frame_width - probe_hi:frame_width - 8]]
    margin_samples = [part.reshape(-1) for part in margin_parts if part.size]
    if not margin_samples:
        return _InlineGapEvidence(
            False, direct_max, None,
            f"photo-to-field gap allowed 0..{direct_max}px")
    background = float(np.median(np.concatenate(margin_samples)))
    mask = (band < background - _ROW_BACKGROUND_TOLERANCE).astype(np.uint8)
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_width = round(comment.width * 0.20)
    min_height = round(comment.height * 0.30)
    max_height = round(comment.height * 0.65)
    edge_slack = max(8, round(comment.width * 0.03))
    components: list[tuple[int, int, int, int]] = []
    for x, y, width, height, area in stats[1:count]:
        x, y, width, height, area = map(int, (x, y, width, height, area))
        density = area / (width * height)
        absolute_y = preview_y1 + y
        if (width >= min_width
                and min_height <= height <= max_height
                and 0.015 <= density <= 0.35
                and x < comment.x1 + edge_slack
                and x + width > comment.x0 - edge_slack):
            components.append((x, absolute_y, x + width, absolute_y + height))

    gutter_max = max(12, round(comment.height * 0.25))
    components.sort()
    for start in range(len(components)):
        row = [components[start]]
        for candidate in components[start + 1:]:
            if (abs(candidate[1] - row[0][1]) <= 8
                    and abs(candidate[3] - row[0][3]) <= 8
                    and candidate[0] >= row[-1][2] + 4):
                row.append(candidate)
        if len(row) < _INLINE_SUGGESTION_MIN_PILLS:
            continue
        shelf = (row[0][0], min(rect[1] for rect in row),
                 row[-1][2], max(rect[3] for rect in row))
        top_gutter = shelf[1] - preview_y1
        bottom_gutter = comment.y0 - shelf[3]
        spans_column = (shelf[0] <= comment.x0 + round(comment.width * 0.20)
                        and shelf[2] >= comment.x1 - edge_slack
                        and shelf[2] - shelf[0] >= round(comment.width * 0.70))
        if (spans_column and 0 <= top_gutter <= gutter_max
                and 0 <= bottom_gutter <= gutter_max):
            max_gap = (shelf[3] - shelf[1]) + 2 * gutter_max
            return _InlineGapEvidence(
                True, max_gap, shelf,
                f"suggestion shelf ({shelf[0]},{shelf[1]})-({shelf[2]},{shelf[3]}) "
                f"with {len(row)} outlined pills and gutters {top_gutter}px/{bottom_gutter}px; "
                f"photo-to-field gap allowed 0..{max_gap}px")

    return _InlineGapEvidence(
        False, direct_max, None,
        f"photo-to-field gap allowed 0..{direct_max}px; no bounded outlined suggestion shelf "
        "bridges the extra space")


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
    detected comment field and CTA, requires the preview to sit within the existing card-to-field
    gap bound, and leaves the content/signature verification below unchanged.  An upper card can
    therefore neither hide the real preview nor license a match.

    TWO REGIMES, AND THE SECOND IS ONLY TRIED WHEN THE FIRST FINDS NOTHING, so a full-bleed photo
    is still located by exactly the evidence it always was:

      * FLUSH -- the run's ink spans the comment field's own columns, both edges within 3%.  This
        is every sheet in the six-sheet corpus and 287 of the 291 composer frames on disk.
      * PILLARBOXED -- the run's ink sits INSIDE those columns and is CENTRED in them, which is
        what a portrait photograph looks like once Hinge has fitted it to its card's height.  The
        rect reported is then the CARD's columns (the field's), not the ink's, because the stored
        crop is the whole card and that is what it has to be scaled against; the ink box rides
        along in `ink_x0`/`ink_x1` for the block walk.  See
        `_INLINE_PILLARBOX_MIN_WIDTH_FRACTION` for the measurement and for what each of the two
        replacement tests is and is not carrying.
    """
    comment = getattr(composer_surface, "comment_rect", None)
    if comment is None:
        raise SheetVerificationError(
            "the supplied inline-composer surface has no comment rectangle for preview lookup")
    slack = max(8, round(comment.width * 0.03))
    minimum_width = round(comment.width * _INLINE_COMPACT_MIN_WIDTH_FRACTION)
    pillarbox_width = round(comment.width * _INLINE_PILLARBOX_MIN_WIDTH_FRACTION)
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
    bottom = min(height, comment.y0)

    def _tall_runs(floor: int) -> list[tuple[int, int]]:
        """Runs of rows at least `floor` wide that are tall enough to be a preview at all."""
        runs: list[tuple[int, int]] = []
        start = None
        for y in range(_PREVIEW_SEARCH_TOP_PX, bottom):
            span = spans[y]
            wide = span is not None and span[1] - span[0] >= floor
            if wide and start is None:
                start = y
            elif not wide and start is not None:
                runs.append((start, y))
                start = None
        if start is not None:
            runs.append((start, bottom))
        return [(a, b) for a, b in runs if b - a >= _PREVIEW_MIN_HEIGHT_PX]

    def _columns(y0: int, y1: int) -> tuple[int, int]:
        """The run's columns: the MEDIAN of its rows' spans, and never their min/max.

        This is load-bearing for the centring test below and it is a trap worth spelling out.
        Hinge draws a small dark alt-text chip on the card's white margin at the top left of the
        selected preview, so on the 2026-08-28 frame 54 of the block's 837 rows report ink
        starting at x=133..156 while the other 783 report exactly 286.  Min/max over the run
        gives a 38px left pad against a 191px right one -- an asymmetry of 153px, which would
        refuse the very frame the pillarbox regime was written for.  The median gives 286..794
        and an asymmetry of 0.  [measured over the corpus: the worst left-vs-right asymmetry on
        any pillarboxed sheet is 0px, and on its source card 0px, so anything above a few pixels
        is a measurement artefact rather than a layout.]
        """
        band = [spans[y] for y in range(y0, y1) if spans[y] is not None]
        return (int(np.median([span[0] for span in band])),
                int(np.median([span[1] for span in band])))

    def _field_gap(y0: int, y1: int) -> int | None:
        """The run's distance to the comment field, or None when it is not adjacent to it."""
        gap = comment.y0 - y1
        evidence = _inline_gap_evidence(
            gray, preview_y1=y1, preview_height=y1 - y0, comment=comment,
            cv2=cv2, np=np)
        return gap if evidence.accepted else None

    def _survey() -> str:
        """Every non-background block above the field, measured, with what it passed AND failed.

        Built only when the locator is about to refuse, and deliberately NOT a report on rejected
        candidates. On the 2026-08-28 halt no run reached the 694px width floor at all, so there
        were no candidates to reject and a message written in terms of them would have described
        nothing -- the diagnosis was entirely in ink that never became a candidate. So this walks
        blocks at NO width floor and prints their geometry against every bound the two regimes
        apply.

        The passes matter as much as the failures. "Fails width and both edges, PASSES height and
        field gap, and is inset by the same amount on both sides" is the pillarbox diagnosis
        stated in one line; the failures alone read as an ordinarily mislocated block, which is
        what a whole day of the 2026-08-28 investigation went into ruling out.
        """
        # A block shorter than this is a line of text -- the sheet's title row, a caption -- and
        # listing every one of them would bury the one block that matters.
        survey_min_height = 24
        blocks: list[tuple[int, int]] = []
        start = None
        for y in range(_PREVIEW_SEARCH_TOP_PX, bottom):
            present = spans[y] is not None
            if present and start is None:
                start = y
            elif not present and start is not None:
                blocks.append((start, y))
                start = None
        if start is not None:
            blocks.append((start, bottom))
        blocks = [(a, b) for a, b in blocks if b - a >= survey_min_height]
        if not blocks:
            return ("No non-background block at all was found between row "
                    f"{_PREVIEW_SEARCH_TOP_PX} and the field, so the frame above the composer is "
                    "blank rather than showing something this could not identify.")
        blocks.sort(key=lambda block: block[1] - block[0], reverse=True)
        described: list[str] = []
        for y0, y1 in blocks[:3]:
            x0, x1 = _columns(y0, y1)
            width, rows = x1 - x0, y1 - y0
            # Signed against the field's own edges, so a pillarboxed card reads +191/-191 (inset
            # on both sides) and an ordinary x=53 profile card reads -42/+42 (over on both).
            left_offset, right_offset = x0 - comment.x0, x1 - comment.x1
            gap = comment.y0 - y1
            gap_evidence = _inline_gap_evidence(
                gray, preview_y1=y1, preview_height=rows, comment=comment,
                cv2=cv2, np=np)
            passed: list[str] = []
            failed: list[str] = []
            for ok, text in (
                    (width >= minimum_width, f"width {width}px vs {minimum_width}px"),
                    (rows >= _PREVIEW_MIN_HEIGHT_PX,
                     f"height {rows} rows vs {_PREVIEW_MIN_HEIGHT_PX}"),
                    (gap_evidence.accepted,
                     f"field gap {gap}px: {gap_evidence.reason}"),
                    (abs(left_offset) <= slack,
                     f"left edge {left_offset:+d}px vs +-{slack}px"),
                    (abs(right_offset) <= slack,
                     f"right edge {right_offset:+d}px vs +-{slack}px")):
                (passed if ok else failed).append(text)
            note = ""
            # Only where WIDTH (and the edges that follow from it) is the single thing wrong.
            # Without this the sentence lands on the sheet's own title row -- 210px of text, 42
            # rows tall, symmetric because centred text is -- and a diagnostic that explains the
            # wrong block is worse than one that explains nothing.
            if (rows >= _PREVIEW_MIN_HEIGHT_PX and gap_evidence.accepted
                    and left_offset >= -slack and right_offset <= slack
                    and abs(left_offset + right_offset) <= slack):
                note = (f" -- it is inset by the same {left_offset}px on both sides, i.e. a card "
                        "rendered whole into the content column with a narrower photo "
                        f"pillarboxed inside it, which needs at least {pillarbox_width}px of ink "
                        f"and {_PREVIEW_MIN_HEIGHT_PX} rows")
            described.append(
                f"rows {y0}..{y1}, columns {x0}..{x1} -- fails: "
                + ("; ".join(failed) if failed else "nothing")
                + (("; passes: " + "; ".join(passed)) if passed else "") + note)
        more = ("" if len(blocks) <= 3
                else f" ({len(blocks) - 3} shorter block(s) not listed)")
        return "The blocks above it, tallest first: " + " | ".join(described) + more

    candidates: list[tuple[int, SheetPreview]] = []
    for y0, y1 in _tall_runs(minimum_width):
        x0, x1 = _columns(y0, y1)
        gap = _field_gap(y0, y1)
        if (abs(x0 - comment.x0) <= slack
                and abs(x1 - comment.x1) <= slack
                and gap is not None):
            candidates.append((gap, SheetPreview(
                y0=y0, y1=y1, x0=x0, x1=x1,
                reason=(f"inline-composer-bound run of rows at least {minimum_width}px wide, "
                        f"{y1 - y0} rows tall, aligned to comment field "
                        f"x={comment.x0}..{comment.x1}, with a {gap}px field gap"))))

    if not candidates:
        # THE PILLARBOX REGIME, AND IT IS TRIED ONLY WHEN NOTHING SAT FLUSH, so a full-bleed
        # photo continues to be located by exactly the evidence it always was. See
        # `_INLINE_PILLARBOX_MIN_WIDTH_FRACTION` for why a correct composer can present a photo
        # narrower than its own content column, and why widening the width floor alone would only
        # move the halt into `_compare_item`.
        #
        # WHAT REPLACES THE TWO EDGE-ALIGNMENT TESTS IS CONTAINMENT AND CENTRING, and it is worth
        # being exact about how much each of the three is really carrying rather than crediting
        # the new pair with the old pair's work.
        #
        # Containment refuses an ordinary FULL-BLEED profile card as decisively as alignment did:
        # a card spans x=53..1027 against a field at 95..985, so it spills 42px past both ends --
        # the same 42px the module docstring's margin test has always turned on. Centring refuses
        # a block hard against one side of the column.
        #
        # BUT NEITHER REFUSES THE ONE IMPOSTOR THAT MATTERS, AND THE GAP BOUND DOES. A PILLARBOXED
        # card scrolled above the composer is contained (262..818 inside 95..985) and centred to
        # the pixel (167px each side), because a card and the content column share a centre. What
        # excludes it is adjacency, and by TWO independent mechanisms: the gap bound refuses a
        # distant candidate outright (comment.y0 - y1 <= 0.25 * (y1 - y0)), and `min(..., key=gap)`
        # below prefers the nearest candidate when several survive. With both present either one
        # suffices, which is worth knowing before deleting either -- a test that puts the scrolled
        # card and the real preview on one frame stays green with the bound removed, because the
        # tiebreak carries it. [swept 2026-08-28 over all 298 composer
        # frames in data/hinge_debug applying exactly these tests: no frame yields more than one
        # accepted candidate, and the accepted block is the selected preview every time. The real
        # scrolled-card frames -- ce2ace4a0ee8/00224, daceece7f043/00214 -- fail containment AND
        # miss the gap allowance by 1017px against 77.] The structural version: reaching the gap
        # bound needs y1 >= 921, so an impostor can only win if the selected preview renders under
        # ~188px tall, at which point the preview fails the 300-row floor and nothing is located
        # at all. `test_a_pillarboxed_card_scrolled_above_the_composer_does_not_displace_the_...`
        # constructs the case the corpus does not contain.
        #
        # What the run is then reported as is the CARD's columns, i.e. the field's, because that
        # is the extent the stored crop has to be scaled against; the ink box is carried
        # separately for the block walk that follows.
        for y0, y1 in _tall_runs(pillarbox_width):
            x0, x1 = _columns(y0, y1)
            gap = _field_gap(y0, y1)
            left_pad, right_pad = x0 - comment.x0, comment.x1 - x1
            contained = left_pad >= -slack and right_pad >= -slack
            centred = abs(left_pad - right_pad) <= slack
            if contained and centred and gap is not None:
                candidates.append((gap, SheetPreview(
                    y0=y0, y1=y1, x0=comment.x0, x1=comment.x1, ink_x0=x0, ink_x1=x1,
                    reason=(f"inline-composer-bound pillarboxed run of rows at least "
                            f"{pillarbox_width}px wide, {y1 - y0} rows tall, its {x1 - x0}px of "
                            f"ink at x={x0}..{x1} centred inside the comment field's "
                            f"x={comment.x0}..{comment.x1} ({left_pad}px of card left, "
                            f"{right_pad}px right), with a {gap}px field gap"))))

    if not candidates:
        # A selected inline card can have a page-coloured right rail after the profile-card heart
        # disappears.  Its left card edge remains visible, but the row-background probe cannot
        # distinguish the rail from the page.  This is NOT the symmetric pillarbox regime above:
        # there is no evidence for a mirrored left-rail variant, and the inline comparison already
        # discards this exact right control lane from both the sheet and every stored card crop.
        #
        # The visible ink must nevertheless reach the START of that already-excluded lane.  Thus
        # every pixel the signature comparison will inspect is still backed by the observed card;
        # this is a geometry recovery, not permission to compare blank page.  Keep the normal
        # compact width, height, gap, composer, and content-verification gates unchanged.
        right_control_lane = round(
            comment.width * _INLINE_REFRAME_RIGHT_CONTROL_FRACTION)
        compared_x1 = comment.x1 - right_control_lane
        for y0, y1 in _tall_runs(minimum_width):
            x0, x1 = _columns(y0, y1)
            gap = _field_gap(y0, y1)
            if (abs(x0 - comment.x0) <= slack
                    and compared_x1 <= x1 <= comment.x1 + slack
                    and gap is not None):
                candidates.append((gap, SheetPreview(
                    y0=y0, y1=y1, x0=comment.x0, x1=comment.x1,
                    ink_x0=x0, ink_x1=x1,
                    reason=(f"inline-composer-bound left-edge-attached run of rows at least "
                            f"{minimum_width}px wide, {y1 - y0} rows tall, its "
                            f"{x1 - x0}px of ink at x={x0}..{x1} reaches the existing "
                            f"right control-lane start x={compared_x1} inside comment "
                            f"x={comment.x0}..{comment.x1}, with a {gap}px field gap"))))

    if not candidates:
        raise SheetVerificationError(
            "the selected-card preview is not immediately above the independently detected "
            f"inline comment field: no candidate at least {minimum_width}px wide "
            f"({_INLINE_COMPACT_MIN_WIDTH_FRACTION:.0%} of the field's own {comment.width}px), "
            f"or at least {pillarbox_width}px and centred inside it, runs for "
            f"{_PREVIEW_MIN_HEIGHT_PX} rows above it. The field is at rows "
            f"{comment.y0}..{comment.y1}, columns {comment.x0}..{comment.x1}, and a flush "
            f"candidate has to hold both edges within {slack}px of those columns. " + _survey())
    _gap, preview = min(candidates, key=lambda candidate: candidate[0])
    return SheetPreview(
        y0=preview.y0, y1=preview.y1, x0=preview.x0, x1=preview.x1,
        ink_x0=preview.ink_x0, ink_x1=preview.ink_x1,
        reason=(f"inline-composer compact fallback: {preview.reason}; minimum width "
                f"{minimum_width}px bound to comment x={comment.x0}..{comment.x1}"))


def _extend_inline_preview_to_block_edges(
        frame: bytes, preview: SheetPreview, composer_surface, *, cv2, np,
        background_tolerance: float = _ROW_BACKGROUND_TOLERANCE,
        margin_probe: tuple[int, int] = _MARGIN_PROBE) -> SheetPreview:
    """Carry a located preview's edges to where the selected photo actually begins and ends.

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

    Under an independently detected composer neither image edge has to be inferred from the
    strict width floor alone. The 2026-08-27 Lauren post-keyboard frame retained the entire
    selected photo at rows 638..1528, but pale upper content fell below the compact floor and the
    composer-bound locator seeded only its lower run at 1116..1528. Walking only downward turned
    that 412px fragment into a false 523/974px reframe refusal even though the photo was complete.

    Walk both ways from the high-confidence located run while rows are still image rows: at least
    `_INLINE_COMPACT_MIN_WIDTH_FRACTION` of the composer's OWN comment width, or the separately
    measured 74% floor while either located edge remains attached. Stop upward at the preview
    search boundary and downward at the comment field.

    AT MOST THE MEASURED THREE-ROW INTERNAL INTERRUPTION IS BRIDGED ACROSS BOTH DIRECTIONS,
    EVERY NON-BACKGROUND ROW KEEPS EITHER THE LEFT EDGE OR THE RIGHT EDGE AND AT LEAST THE
    MEASURED 74% OF PHOTO STRUCTURE, AND THE WALK CANNOT REACH PAST EITHER LAYOUT BOUNDARY, so
    this cannot wander onto a different element to manufacture adjacency: it can only finish the
    block the strict locator had already started inside. A preview separated from another block
    by more than three background rows halts there. Columns are left as located: they are the
    median of the strict band, which is the higher-confidence evidence, and the added rows had to
    retain one of those edges to be walked at all.

    NOT a no-op after `_locate_inline_composer_preview`, and the sentence that used to claim it
    was is deleted rather than reworded. Both halves of it were false and both are measurable:
    that locator scans at `_INLINE_COMPACT_MIN_WIDTH_FRACTION` (0.78) while this walk continues at
    the lower one-edge floor (0.74), so its run does NOT end at a background row -- on this
    module's own Lauren fixture it returns 713..1092 and this walk carries the top edge 477 rows
    up to 236, terminating on a row that still holds a 661px span. At HEAD the sentence named
    `_locate_inline_compact_preview`, a function that does not exist, which made it obviously
    stale; the 2026-08-27 rename swapped in the real name and left the false substance, which is
    worse -- it launders a dead sentence into a believable one. These docstrings are the
    calibration record, so a claim that cannot be reproduced is removed, not repaired.
    """
    comment = getattr(composer_surface, "comment_rect", None)
    if comment is None:
        return preview
    gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        # `locate_sheet_preview` decoded this same frame moments ago; an undecodable frame is
        # its refusal to report, not this helper's.
        return preview
    lower_limit = min(gray.shape[0], comment.y0)
    # THE MODULE CONSTANT, DELIBERATELY, AND NOT `locate_kwargs`' `search_top_px`. It looks like
    # an inconsistency -- `locate_sheet_preview` honours that kwarg and this walk does not -- and
    # forwarding it would be a BUG, not a fix: the composer fallback seed
    # `_locate_inline_composer_preview` hardcodes `_PREVIEW_SEARCH_TOP_PX` and takes no kwargs, so
    # a caller-narrowed value would stop the walk above a seed that was found below it, yielding a
    # truncated preview, a wrong window height and a false refusal. No caller passes
    # `search_top_px` today; if one ever needs to, the seed locator has to learn it FIRST and all
    # three have to move together.
    upper_limit = min(max(0, _PREVIEW_SEARCH_TOP_PX), preview.y0)
    spans = _row_spans(gray, np, tolerance=background_tolerance, probe=margin_probe)
    # THE WALK FOLLOWS THE INK, NOT THE RECT, AND FOR EVERY PREVIEW THAT PREDATES THE PILLARBOX
    # REGIME THOSE ARE THE SAME COLUMNS AND THIS IS THE SAME ARITHMETIC AS BEFORE. A pillarboxed
    # preview reports the CARD's columns as its rect -- that is what `_compare_item` needs -- but
    # its rows are made of a narrower, inset photo, so aligning to and measuring against the rect
    # would find no supported row anywhere and silently turn this walk into a no-op on exactly
    # the previews whose edges are hardest to see. `ink_bounds` is the rect itself unless the
    # locator measured an inset box, so the flush regime's floors stay derived from the comment
    # field's width and nothing about it moves.
    ink_x0, ink_x1 = preview.ink_bounds
    reference_width = preview.ink_width if preview.pillarboxed else comment.width
    minimum_width = round(reference_width * _INLINE_COMPACT_MIN_WIDTH_FRACTION)
    one_edge_minimum_width = round(
        reference_width * _INLINE_PREVIEW_ONE_EDGE_MIN_WIDTH_FRACTION)
    slack = max(8, round(reference_width * 0.03))
    def row_support(scan_y: int) -> tuple[bool, bool]:
        span = spans[scan_y]
        span_width = 0 if span is None else span[1] - span[0]
        left_aligned = span is not None and abs(span[0] - ink_x0) <= slack
        right_aligned = span is not None and abs(span[1] - ink_x1) <= slack
        # BOTH EDGES ARE BOUNDED, NOT JUST THE ALIGNED ONE, and this is what keeps the docstring's
        # "it can only finish the block the strict locator had already started inside" true.
        # Alignment alone does not say a row belongs to the preview -- it says ONE of its edges
        # does. An ordinary profile card sits at x=53..1027, which the alignment tests reject at
        # 42px against 27px of slack; but a card whose leading columns are pale enough to read as
        # page background reports a span that STARTS at the preview's own x0 and still runs 42px
        # past its x1, and that row passed every test above. [measured 2026-08-27 on the real
        # 00097 composer frame with a pale-left card painted at rows 110..232: the upward walk
        # welded a 236..1092 preview into 110..1092 and turned a `verify_match` into a
        # `verify_mismatch`.] Containment costs nothing legitimate: a genuine row of THIS photo
        # cannot extend past the photo's own columns.
        contained = (span is not None
                     and span[0] >= ink_x0 - slack
                     and span[1] <= ink_x1 + slack)
        fully_supported = span_width >= minimum_width and left_aligned and contained
        # Pale content may erase either visible edge against the page, but it cannot erase both:
        # this lower, composer-only floor requires the other edge to remain exactly attached to
        # the strict preview we already found.  It does not bridge blank page rows (span=None).
        one_edge_supported = (
            span_width >= one_edge_minimum_width and (left_aligned or right_aligned)
            and contained)
        return fully_supported or one_edge_supported, one_edge_supported and not fully_supported

    def scan_edge(start: int, stop: int, step: int, boundary: int):
        scan_y = start
        last_supported_boundary = boundary
        unsupported_run = 0
        bridged_rows = 0
        bridged_runs = 0
        one_edge_rows = 0
        while (scan_y < stop) if step > 0 else (scan_y >= stop):
            supported, one_edge_only = row_support(scan_y)
            if supported:
                if one_edge_only:
                    one_edge_rows += 1
                if unsupported_run:
                    bridged_runs += 1
                    bridged_rows += unsupported_run
                unsupported_run = 0
                last_supported_boundary = scan_y + 1 if step > 0 else scan_y
            else:
                unsupported_run += 1
                if unsupported_run > _INLINE_PREVIEW_MAX_INTERNAL_GAP_PX:
                    break
            scan_y += step
        return last_supported_boundary, bridged_rows, bridged_runs, one_edge_rows

    y0, upper_bridged_rows, upper_bridged_runs, upper_one_edge_rows = scan_edge(
        preview.y0 - 1, upper_limit, -1, preview.y0)
    y1, lower_bridged_rows, lower_bridged_runs, lower_one_edge_rows = scan_edge(
        preview.y1, lower_limit, 1, preview.y1)

    # THE SEAM BUDGET IS JUDGED PER EDGE, AND THE OFFENDING EDGE ALONE IS DISCARDED. Summing the
    # two directions and testing the sum against a single-direction budget was a REGRESSION
    # against HEAD, where only one direction existed: one legal 3-row seam below plus a single
    # bridged row above pushed `bridged_runs` to 2, threw away the whole extension, and produced
    # exactly the false refusal this walk exists to prevent. Each direction is an independent
    # extension of the same strict run, so each is held to precisely the budget HEAD held its one
    # direction to -- neither direction is allowed a wider seam than before, and the downward
    # behaviour is now byte-for-byte HEAD's again.
    #
    # This is deliberately NOT the same thing as widening the seam. What stops the walk reaching a
    # DIFFERENT block is not the number of seams it forgave, it is
    # `_INLINE_PREVIEW_MAX_INTERNAL_GAP_PX` as a cap on CONSECUTIVE unsupported rows (unchanged,
    # so a four-row gap is still a hard boundary in both directions) together with the column
    # containment in `row_support` above. Two three-row seams do not add up to permission to cross
    # a four-row one.
    def within_seam_budget(runs: int, rows: int) -> bool:
        # `bridged_rows` counts only interruptions followed by more supported image rows; terminal
        # page background is never included in the preview. Failing closed here also prevents a
        # future loop change from silently widening the seam by accepting several individually
        # small gaps in one direction.
        return runs <= 1 and rows <= _INLINE_PREVIEW_MAX_INTERNAL_GAP_PX

    if not within_seam_budget(upper_bridged_runs, upper_bridged_rows):
        y0 = preview.y0
        upper_bridged_rows = upper_bridged_runs = upper_one_edge_rows = 0
    if not within_seam_budget(lower_bridged_runs, lower_bridged_rows):
        y1 = preview.y1
        lower_bridged_rows = lower_bridged_runs = lower_one_edge_rows = 0
    bridged_rows = upper_bridged_rows + lower_bridged_rows
    one_edge_rows = upper_one_edge_rows + lower_one_edge_rows
    if y0 == preview.y0 and y1 == preview.y1:
        return preview
    carried = []
    if y0 != preview.y0:
        carried.append(f"top edge carried from row {preview.y0} to {y0}")
    if y1 != preview.y1:
        carried.append(f"bottom edge carried from row {preview.y1} to {y1}")
    return SheetPreview(
        y0=y0, y1=y1, x0=preview.x0, x1=preview.x1,
        ink_x0=preview.ink_x0, ink_x1=preview.ink_x1,
        reason=(f"{preview.reason}; {' and '.join(carried)} over the "
                f">={minimum_width}px image block bounded by search row {upper_limit} and the "
                f"composer comment field at row {comment.y0}, retaining one aligned edge across "
                f"{one_edge_rows} rows and bridging {bridged_rows} measured internal background "
                "rows"))


def _decode_crop(crop, cv2, np):
    """The stored crop as one greyscale array, in the same grey space as the sheet.

    Decoded ONCE per verification rather than once per swept window: a sweep is tens of windows
    per item and every one of them would otherwise re-decode a ~1MB PNG, which is seconds of ADB
    dead time in the middle of an open comment sheet.

    THE BYTES ARE `verify_image`, NEVER `image`, AND THAT DISTINCTION IS THE WHOLE POINT. This
    function used to decode `crop.image` with `cv2.IMREAD_GRAYSCALE` and argue that it was safe
    because `item_crops.signature_of` makes byte for byte the same call. That argument is wrong:
    calling one decoder on two DIFFERENTLY ENCODED PNGs does not produce one grey space. The
    sheet side is the frame the DEVICE wrote, which carries an `sRGB` chunk; `crop.image` is a
    re-encode by `cv2.imencode`, which writes no such chunk. `IMREAD_GRAYSCALE` is sRGB-aware, so
    it greys the two differently -- and re-inserting the chunk into the re-encode makes them
    bit-identical again, which is what pins the cause to the chunk rather than to the alpha
    channel or to the choice of conversion. The comparison therefore ran in exactly the two units
    `item_crops`' module docstring documents as incompatible.

    [measured 2026-08-27, live Pixel 7a, Tina heart 9: the two paths differ by 4.35 grey levels
    on average and 64 at the worst pixel, and by 10.158 on the 64x64 signature. The correct card
    scored 10.283 against a 7.00 ceiling and the run halted with the sheet open. The error is
    content-dependent -- under 3.2 over the whole calibration corpus, but that photo is a dark,
    saturated neon frame, which is where the two greys diverge most. On `verify_image` the same
    card scores 0.167 and the reciprocal wrong card still scores 73.965.

    NOTE FOR ANYONE ADDING A FIXTURE: no synthetic frame in this repo can reproduce the fault,
    because `cv2.imencode` never writes an `sRGB` chunk, so both paths agree on a painted PNG.
    That is exactly why the whole test corpus passed while a live run refused a correct card. The
    regression test injects the chunk by hand.]

    `verify_image` is an 8-bit greyscale PNG cut from the frame's own `IMREAD_GRAYSCALE` decode,
    so there is no colour conversion left anywhere in this path to disagree about, and the
    windows below are still reduced by `item_crops._signature_from_gray` -- that module's single
    definition of what a signature IS.
    """
    number = crop.number
    if not crop.image:
        # A CROP_ITEM carries an image by construction (`ItemPayload.signature_for` says so), so
        # this is a hand-assembled payload rather than a real one. Named rather than left to fail
        # inside numpy, because "there is no reference for this item" and "the reference does not
        # match" are different problems with the same consequence.
        raise SheetVerificationError(
            f"model item {number} carries no stored crop, so the sheet cannot be verified "
            "against it")
    if not crop.verify_image:
        # Fail closed rather than fall back to `crop.image`: that fallback IS the bug above, and
        # it is invisible in the numbers it produces -- a wrongly-scaled distance still looks
        # like a distance. `build_item_payload` fills this in for every cropped block, so a
        # payload without it is hand-assembled and cannot be verified in the sheet's grey space.
        raise SheetVerificationError(
            f"model item {number} carries no greyscale verification reference, so the sheet "
            "cannot be compared against it in the grey space it was measured in")
    gray = cv2.imdecode(np.frombuffer(crop.verify_image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SheetVerificationError(
            f"the stored verification reference for item {number} did not decode as an image "
            f"({len(crop.verify_image)} bytes), so there is nothing to verify the sheet against")
    return gray


def _check_reference_provenance(crop, gray, *, signature_grid: tuple[int, int], cv2, np) -> None:
    """Refuse a verification whose reference is not in the grey space its signature was stored in.

    THIS IS THE GUARD FOR THE FAULT CLASS `_decode_crop` DESCRIBES, and it is cheap because both
    sides already exist: `item_crops` stored `crop.signature` from the FRAME's greyscale decode
    of exactly these rows, and `gray` is what this module is about to cut every swept window out
    of. Reduce the whole of `gray` at the stored grid and the two must agree, because they are
    the same pixels of the same card measured the same way.

    When they do not, every number below is quietly in the wrong units: the sheet is measured in
    the frame's grey space and the items in some other one, so a correct card's distance is
    inflated by a content-dependent amount and the verdict is a coin toss dressed as a
    measurement. On 2026-08-27 that cost a live run -- the right card scored 10.283 against a
    7.00 ceiling, the sheet stayed open and the profile was abandoned. A wrongly-scaled distance
    is indistinguishable from an honest one by inspection, which is exactly why this is asserted
    rather than trusted.

    Fails closed, like every other refusal in this module: the caller stops with the sheet open
    and nothing typed. It cannot fire on a payload `build_item_payload` produced.
    """
    if crop.signature is None:
        # Nothing stored to check against. `_compare_item` does not need `crop.signature`, so
        # this is silence rather than a refusal -- a hand-assembled payload that carries a
        # reference but no signature is still internally consistent.
        return
    try:
        reproduced = _signature_from_gray(gray, grid=signature_grid, cv2=cv2, np=np)
    except Exception as exc:  # noqa: BLE001 -- surface as this module's typed refusal
        raise SheetVerificationError(
            f"the verification reference for item {crop.number} could not be reduced to a "
            f"signature to check its provenance against the stored one: {exc}") from exc
    distance = crop.signature.distance(reproduced)
    if distance > _REFERENCE_PROVENANCE_MAX_DIST:
        raise SheetVerificationError(
            f"item {crop.number}'s verification reference does not reproduce the signature "
            f"stored for the same rows: {distance:.3f} grey levels apart at the stored "
            f"{signature_grid[0]}x{signature_grid[1]} grid, against a "
            f"{_REFERENCE_PROVENANCE_MAX_DIST:.3f} bound. The two are the same pixels, so a "
            f"difference means they were decoded into different grey spaces and every distance "
            f"measured against this sheet would be in the wrong units. Nothing is typed")


def _compare_item(crop, gray, preview: SheetPreview, sheet: CropSignature, *,
                  grid: tuple[int, int], scale_tolerance: float, cv2, np,
                  inline_reframe: bool = False,
                  inline_reframe_max_hidden_fraction: float =
                  _INLINE_REFRAME_MAX_HIDDEN_FRACTION):
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
    # width and all but a bounded edge strip. The ordinary composer permits the measured 30%;
    # the short-photo renderer may supply its separately proven 40% only after the outlined
    # suggestion shelf is detected. The normal modal comparison above retains bottom anchoring.
    nominal_hidden = crop_height - window_px
    if nominal_hidden > round(crop_height * inline_reframe_max_hidden_fraction):
        return (None, window_px, reference,
                f"inline composer preview would hide {nominal_hidden}px of this {crop_height}px "
                f"crop, above the {inline_reframe_max_hidden_fraction:.0%} reframe limit")
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
        if max_start > round(crop_height * inline_reframe_max_hidden_fraction):
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
    inline_reframe_max_hidden_fraction = _INLINE_REFRAME_MAX_HIDDEN_FRACTION
    if composer_surface is not None:
        # The located run is where the preview STARTS; under a proven composer its bottom edge is
        # the end of the image block, not the first row a legacy width floor stumbles on. See
        # `_extend_inline_preview_to_block_edges` -- a real 10.0.1 composer was refused below
        # because a bright photo edge cut an 856px preview down to a 620px fragment.
        preview = _extend_inline_preview_to_block_edges(
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
        gap_gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gap_gray is None:
            raise SheetVerificationError(
                f"the like-sheet frame did not decode as an image ({len(frame)} bytes), so "
                "its preview-to-composer topology cannot be checked")
        width_slack = max(8, round(preview.width * 0.03))
        gap = comment.y0 - preview.y1
        gap_evidence = _inline_gap_evidence(
            gap_gray, preview_y1=preview.y1, preview_height=preview.height,
            comment=comment, cv2=cv2, np=np)
        inline_reframe_max_hidden_fraction = gap_evidence.max_hidden_fraction
        controls_gap = send.y0 - comment.y1
        if (abs(preview.x0 - comment.x0) > width_slack
                or abs(preview.x1 - comment.x1) > width_slack
                or not gap_evidence.accepted
                or not 0 <= controls_gap <= max(60, round(comment.height * 0.40))):
            raise SheetVerificationError(
                "the selected-card preview is not immediately above the independently detected "
                "inline comment field and Send Like CTA "
                f"(preview=({preview.x0},{preview.y0})-({preview.x1},{preview.y1}), "
                f"comment=({comment.x0},{comment.y0})-({comment.x1},{comment.y1}), "
                f"send=({send.x0},{send.y0})-({send.x1},{send.y1}), "
                f"photo-to-field gap={gap}px ({gap_evidence.reason}), "
                f"field-to-CTA gap={controls_gap}px)")
        if gap_evidence.suggestion_shelf is not None:
            preview = SheetPreview(
                y0=preview.y0, y1=preview.y1, x0=preview.x0, x1=preview.x1,
                ink_x0=preview.ink_x0, ink_x1=preview.ink_x1,
                reason=(preview.reason + "; " + gap_evidence.reason
                        + f"; inline reframe limit {_INLINE_SUGGESTION_MAX_HIDDEN_FRACTION:.0%}"))
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

    # THE PILLARBOX DILUTION, AND WHY THE TWO ABSOLUTE CEILINGS CANNOT BE CARRIED OVER UNCHANGED.
    # A pillarboxed comparison rect is blank on BOTH sides of the comparison: card-white in the
    # sheet and card-white in every candidate crop, so those cells contribute ~0 to every distance
    # and the whole scale contracts. [measured 2026-08-28 on the Yvonne render: the comparison rect
    # is 730px wide and carries 508px of ink, so 30.4% of it is blank and the geometry predicts a
    # 0.696 contraction. THE ACTUAL CONTRACTION IS CONTENT-DEPENDENT AND IS NOT ONE NUMBER --
    # comparing the same pair over the photo's columns alone rather than the content column gives
    # 0.749 for the correct card and 0.867 for a mirrored foreign one, while a separately
    # composited foreign card gave 0.681. It is not a clean factor because the 64x64 grid resolves
    # the photo more finely when it is cut to the photo's own columns. The geometric 0.696 sits at
    # or below all three, which is the direction that matters: it tightens by at least as much as
    # the contraction it corrects for.]
    #
    # THIS CORRECTION IMPROVES A MARGIN THAT WAS ALREADY ADEQUATE; it does not rescue a broken
    # guard. [measured 2026-08-28: 40 foreign photographs composited into this exact card template
    # at its exact photo box score min 31.32, median 58.98 through the pillarbox path, and not one
    # is under the uncorrected 14.9099. The correction moves that from 2.1x of headroom to 3.0x.]
    #
    # The relative bound does not care: `0.5 x nearest_other` is a ratio of two quantities that
    # contract together, which is exactly why a multi-item run like the one that found this looks
    # perfectly safe and hides the problem. THE ABSOLUTE CEILINGS DO CARE, because they are frozen
    # numbers in the undiluted scale, and one of them is frozen hard against a known collision:
    # `config.yaml`'s `inline_item_max_dist` is 14.9099 against the 14.91 measured foreign-card
    # false match, a margin of 0.0001, and `tools/hinge_calibrate.py` clamps it there ON PURPOSE.
    # Left uncorrected, that collision lands at 14.91 x 0.681 = 10.155 in pillarbox units, i.e.
    # 4.755 grey levels INSIDE the ceiling meant to refuse it -- so making this regime reachable
    # would have quietly disarmed the one guard that catches content the payload does not contain
    # at all, which is the case with no relative bound to fall back on.
    #
    # Scaling the ceilings by the measured content fraction puts the regime back in the units its
    # calibration was fitted in. It can only ever LOWER a ceiling, so by the same argument
    # `_INLINE_COMPOSER_ONE_ITEM_MAX_DIST` records, it cannot create a false accept; what it can
    # cost is availability, in the fail-loud direction. The exact alternative -- comparing over
    # the photo's columns on both sides, which needs no estimate at all -- is the better fix and
    # is deliberately NOT taken here: it re-cuts every item's reference and so changes what
    # `nearest_other` means, and there is exactly ONE pillarboxed render in the whole corpus to
    # validate that against.
    content_fraction = 1.0
    if composer_surface is not None and preview.pillarboxed:
        ink_lo = max(comparison_preview.x0, preview.ink_bounds[0])
        ink_hi = min(comparison_preview.x1, preview.ink_bounds[1])
        content_fraction = max(0, ink_hi - ink_lo) / comparison_preview.width
        if not 0.0 < content_fraction <= 1.0:
            raise SheetVerificationError(
                f"the pillarboxed preview's ink ({preview.ink_bounds[0]}..{preview.ink_bounds[1]}) "
                f"covers none of the {comparison_preview.width}px compared, so every distance "
                "measured through it would be between two blank regions")

    measured: list[tuple] = []
    for crop in payload.items:
        gray = _decode_crop(crop, cv2, np)
        _check_reference_provenance(
            crop, gray, signature_grid=payload.signature_grid, cv2=cv2, np=np)
        distance, window_px, reference, reason = _compare_item(
            crop, gray, comparison_preview, sheet, grid=grid, scale_tolerance=scale_tolerance, cv2=cv2,
            np=np, inline_reframe=(composer_surface is not None),
            inline_reframe_max_hidden_fraction=inline_reframe_max_hidden_fraction)
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
            bound = _INLINE_COMPOSER_ONE_ITEM_MAX_DIST * content_fraction
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
            # "CANNOT CONFIRM", not "does not show", and the difference is the whole diagnosis.
            # Reaching here means item `model_index` IS the nearest stored crop -- no other item
            # was closer -- so the honest statement is that the distance did not come under the
            # bound, which is a measurement or calibration outcome. Claiming the sheet shows
            # something else is a stronger assertion than the numbers support, and on 2026-08-27
            # it sent the reader looking for a targeting bug when the sheet had been correct all
            # along and the reference was in the wrong grey space.
            f"the like sheet cannot be confirmed as model item {model_index}: item {model_index} "
            f"IS the nearest stored crop, but at {mine.distance:.3f} grey levels, against a "
            f"{mine.bound:.3f} bound ({derivation}). Nothing else this profile was indexed with "
            f"is closer, so this is a distance/bound refusal rather than a different card. "
            f"Nothing is sent and the sheet is left open"))
    effective_absolute = (None if absolute_max_dist is None
                          else absolute_max_dist * content_fraction)
    if effective_absolute is not None and mine.distance >= effective_absolute:
        return _verdict(VERIFY_MISMATCH, (
            f"the like sheet is too far from model item {model_index} to confirm: "
            f"{mine.distance:.3f} grey levels is within its relative {mine.bound:.3f} bound "
            f"but not below the calibrated absolute {effective_absolute:.3f} ceiling"
            + ("" if content_fraction == 1.0 else
               f" (the {absolute_max_dist:.3f} calibrated ceiling scaled by the "
               f"{content_fraction:.3f} of this pillarboxed preview that carries ink, so it is "
               "applied in the units it was fitted in)")
            + ". Nothing was typed and the sheet is left open"))
    return _verdict(VERIFY_MATCH, (
        f"the like sheet shows model item {model_index}: {mine.distance:.3f} grey levels from its "
        f"stored crop, against a {mine.bound:.3f} bound ({derivation}), and nearer than every "
        f"other numbered item this profile was indexed with"))
