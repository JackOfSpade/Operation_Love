"""How far the enumeration pass may scroll NEXT, sized against the spacing it can actually see.

This is ops/OPENER-REDESIGN.md 5.5's closed loop and 5.10.1's step/spacing rule, made
executable. It answers one question — "what `frac` do I hand `_scroll_down_one` for the next
read-scroll of THIS profile?" — from the segmentation of the frame we are about to scroll away
from, and it refuses when it cannot answer safely.

    plan_scroll_step(segmentation) -> ScrollStep      # .frac and .x_frac go to _scroll_down_one

Deliberately a leaf module, on the same terms as `segment.py` / `frameshift.py` /
`item_index.py` / `item_crops.py` / `scroll_top.py`: pure functions over a `FrameSegmentation`
plus explicit calibration parameters. No device, no I/O, no global state, and no touch access of
any kind — the only thing this module can produce is a NUMBER, so the humanized gesture path is
the only way to spend it (see "HOW THE ENUMERATION LOOP CALLS THIS" below, and note the
`x_frac` trap there, which is load-bearing).

WHY A FIXED SCREEN FRACTION IS THE WRONG ANSWER, AND WHAT REPLACES IT
----------------------------------------------------------------------
Production's `read_scroll_frac` is 0.55, which on the calibrated Pixel 7a moves the content
1299px per step. Doc 5.10.1 measured what that costs: at ~1027px card spacing the step/spacing
ratio is ~1.26, "a heart translated by one step lands almost exactly where the next item's heart
already was", and the two become geometrically indistinguishable. 25% frame-to-frame match rate,
6 of 8 pairs flagged as tracking failures, a phantom tenth item fabricated. `item_index` now
REFUSES that cadence outright ("moved +1299px, beyond the 900px window", zero blocks), which is
correct and is also why nothing in Part B can run until this module exists.

The fix is not a smaller constant. 5.10.1 is explicit:

  > 363px is safe against this profile's 1027px spacing, but a short prompt card could space
  > hearts ~600px apart, where the same 363px step gives a ratio of 0.6 and lands back in the
  > aliasing band. So the rule is relative, not absolute: the enumeration pass must step at most
  > about a third of the LOCALLY MEASURED card spacing.

and its own addendum sharpens it further — the 231516Z capture's 16 measured spacings are 738,
809 and 1027px on ONE profile, so "a step sized against 1027 would already be at ratio 0.49 where
the spacing is 738". Spacing is not a per-profile constant either. It has to be re-measured every
frame, which is exactly what `segment.py` already computes for free.

THE MEASURAND: HEART-BEARING SPACING, AND WHY HEARTLESS BLOCKS ARE EXCLUDED
----------------------------------------------------------------------------
Aliasing is a CORRESPONDENCE failure between two things that look alike at a fixed period. Doc
5.10.1 frames it in exactly those terms (a heart landing where the next heart was), so the period
that matters is the one between HEART-BEARING items, and three independent measurements of it are
available on a single frame. `measure_local_spacing` takes the MINIMUM of whatever it finds,
because the smallest local period is the hazard:

  1. `SPACING_HEART_PITCH` — two like glyphs on one frame, `y2 - y1`. Exact, needs no block edge
     observed at all, and it is doc 5.10.1's own quantity.
     [corpus: 738, 809, 1027 on profile B; 828, 1027, 1162, 1165, 1166 on profile A.]
  2. `SPACING_CARD_PITCH` — a heart-bearing block and the block DIRECTLY below it, tops both
     observed, top-to-top. Exact, and it reaches the case where the lower card's heart is still
     below the analysed band, which rule 3 cannot bound and rule 1 cannot see. Only the UPPER
     block must bear a heart, and that asymmetry is the whole safety argument: the pair measures
     the upper card's own period (its height plus whatever separates it from the next block), so
     a heartless block below contributes nothing but a boundary. Reverse it and the vitals block
     becomes the upper member of a 268px pair, which is the pathological number below.
     Adjacency is required rather than "the next heart-bearing block" for the same reason: a
     block skipped over might be hiding a heart nobody bounded, and skipping it would OVERSTATE
     the period — the one direction that is unsafe.
  3. `SPACING_CARD_EXTENT` — a complete heart-bearing block's own height plus one gutter. A lower
     bound on the distance to the next heart (if a heartless block intervenes, the true heart
     pitch is larger), so it errs small, which errs safe. This is the workhorse: some frame shows
     a complete card far more often than it shows two hearts.
     [corpus: complete SELECTABLE heights are 685, 756, 775, 974, 1109, 1113 and 1114px, i.e.
     pitches 737..1166 with the 52px gutter below.]

Heartless blocks never begin a period, and that exclusion is not tidiness — it is the difference
between a workable loop and a stalled one. [corpus: profile A's vitals block measures 215px and
carries no heart in any frame, so its top-to-top pitch to the card below it is 268px. Sized
against 268 the step would be 89px, roughly 112 captures for one profile, and it would then fall
under the minimum gesture this driver is allowed to make and hard-stop the run.] Two independent
things say that 268px is not a period anything can alias against: aliasing needs a REPEATED
structure, and one 215px block between two ~1000px cards repeats nothing; and the 363px cadence
doc 5.10.1 validated ran straight over that very block at a ratio of 1.35 and still recovered all
9 of profile A's items, agreeing item for item with the independent bot-scrolled capture. The
corpus's other heartless blocks (944, 1009 and 1144px) are excluded on the same rule.

[corpus: at least one of the three measurements is available on 124 of the 148 frames — 20 of 24,
7 of 9 and 97 of 115. The other 24 are frames where the analysed band happens to slice every
block it contains; see the fallback below.]

WHERE THE JITTER LIVES: A WINDOW SIZED BY THE RATIO RULE, DRAWN IN PIXELS
--------------------------------------------------------------------------
Doc 5.5's anti-bot argument for closed-loop scrolling is that "a content-following scroll
distance varies with card height, which varies per item and per profile, and that is a WIDER
distribution than a fixed screen fraction. With jitter on top it has strictly more entropy than
what runs today". The owner rule behind it is stronger and simpler: any timing or distance
parameter must be randomized, never a fixed constant, because a fixed constant is a signature.

So the step is drawn from a WINDOW whose two ends are the content-following part — the low end is
`_STEP_RATIO_MIN` of the locally measured spacing, the high end `_STEP_RATIO_MAX` of it (capped by
`_MAX_STEP_PX`) — and the draw inside that window is uniform over its integer PIXELS. A tall card
still jitters over a proportionally wider range than a short one, because the window's width is a
fraction of the card; what the pixel draw buys is that nothing piles up on the window's edges.

**An earlier revision drew the RATIO and then clamped the resulting distance into the legal
gesture range, and that clamp was a bot signature rather than a rounding detail.** A clamp turns
every draw outside the range into the SAME delivered distance, so the realised distribution grew a
point mass at each end: at the gesture floor below (a short card licenses less than the smallest
gesture this driver may make) and at `_MAX_STEP_PX` above (a tall card licenses more than the one
cadence with end-to-end evidence). Measured over 5000 draws per spacing, ratio-draw-then-clamp
against the same window drawn in pixels:

    spacing  window     ratio-then-clamp            pixel draw
     609px   219..219   1 value, 100%               1 value, 100%   <- forced by the device
     620px   219..223   5 values, 95% on one        5 values, 21%
     650px   219..234   15 values, 78% on one       16 values, 7%
     738px   219..265   46 values, 40% on one       46 values, 4%   <- the corpus MINIMUM
    1027px   219..363   92 values, 7% on one        92 values, 2%
    1166px   219..363   57 values, 34% on one       57 values, 4%
    1400px   219..363   2 values, 87% on one        139 values, 2%

[corpus: over the three captures in order, threading `profile_min_spacing_px`, the single most
common step went from 24-37% of all gestures to 2-4%, and the cost is unchanged or slightly better
— 35 scrolls to cover profile B's 8349px against 34, and 43 for profile A's 10027px against 42,
because the floor's point mass was dragging the mean step DOWN. No ratio moved: the corpus-wide
maximum is 0.3596 before and after, and not one plan in either scheme reaches 0.4.]

Two edges of that window are worth stating explicitly, because both are where the old point masses
came from and neither is a free choice:

  * the GESTURE FLOOR wins over `_STEP_RATIO_MIN`. A spacing under ~842px puts `_STEP_RATIO_MIN`
    of it below the smallest read-scroll this driver may make, so the window narrows from below
    and, at ~609px, closes entirely onto one legal distance. That is a fact about the device, not
    a choice, and the plan REPORTS it (`ScrollStep.window_px`, and `reason` says so when the
    window holds a single value) rather than presenting a forced constant as a draw.
  * `_MAX_STEP_PX` wins over `_STEP_RATIO_MIN` too. On a card spacing past ~1396px the whole ratio
    window sits above the ceiling; the old clamp collapsed that onto 363px, and the window instead
    drops its low end back to the gesture floor. Every value in it is smaller than the ratio rule
    would have allowed, which is the safe direction.

THE TRANSPORT MODEL, MEASURED RATHER THAN ASSUMED
---------------------------------------------------
`_scroll_down_one` takes a FRACTION OF SCREEN HEIGHT, and the content does not move by that
fraction. Both transports (`Adb.scroll_up`, `UhidTouch.scroll_up`) drag from
`int(h * (0.5 + frac/2))` to `int(h * (0.5 - frac/2))`, and the content moves that distance MINUS
a constant:

    content_shift_px = int(h * (0.5 + frac/2)) - int(h * (0.5 - frac/2)) - _TOUCH_SLOP_PX

[corpus, and this is the cleanest number in the whole calibration set: the 231516Z capture issued
`frac=0.16` on a 2400px screen — a 384px drag — 24 times, and `frameshift.estimate_shift`
measures its median step at 363px. 384 - 363 = 21. The 225314Z capture issued `frac=0.55` — a
1320px drag — and measures 1299px. 1320 - 1299 = 21. Two cadences 3.4x apart, the same 21px, to
the pixel. It is a fixed offset, not a gain.]

The mechanism is almost certainly Android's `ViewConfiguration` touch slop, which is 8dp and
lands on exactly 21px at this device's 420dpi/2.625 density bucket — but the MEASUREMENT is the
authority here, not the explanation, and `step_px_for_frac` reproduces the transport's integer
truncation rather than approximating it.

THE THREE BOUNDS, AND WHICH ONE FAILS LOUD
--------------------------------------------
  * CEILING, `_MAX_STEP_PX` = 363. The tallest cards would otherwise license a ~420px step, and
    363 is the only cadence anything has end-to-end evidence for. It also bounds the one risk the
    local measurement cannot see — the card BELOW the fold being shorter than the card in front
    of us — to precisely the risk the corpus already ran, since 363 against that capture's own
    smallest 738px spacing is ratio 0.49 and it measured 23 of 23 pairs with zero tracking
    failures and no phantom.

    That below-the-fold hole is narrowed further, and by evidence rather than by a constant, via
    `profile_min_spacing_px`: the enumeration loop threads the smallest spacing it has seen
    ANYWHERE on this profile so far back into the next plan, and the step is sized against the
    smaller of that and the current frame. The loop starts permissive and tightens the moment it
    meets a short card, so a short card can cost at most the ONE step during which it was still
    below the fold. It is an optional argument because the plan is computable without it; a
    caller that omits it is left with `_MAX_STEP_PX` alone, which is why the loop in the module
    docstring's call site threads it.
  * FLOOR, derived not declared: the smallest step the sanctioned gesture window can deliver,
    `step_px_for_frac(_READ_SCROLL_FRAC_MIN, h)` = 219px on the calibrated device. The
    enumeration scroll is an ordinary read-scroll and must stay inside the same distance window
    every other read-scroll in this driver is validated against; a gesture distribution that
    dips outside it is a new and unvalidated motion signature, which the humanized-input rule
    does not allow us to invent quietly.
  * The two can CONFLICT, and that is the loud one. If even the floor step exceeds
    `_STEP_RATIO_MAX` of the measured spacing — spacing below ~608px on this device — then no
    gesture this driver is allowed to make can enumerate that profile without aliasing, and
    `plan_scroll_step` raises `ScrollStepError`. It does not step further "just this once".
    [corpus: no observed profile comes near this — the smallest spacing measured anywhere across
    148 frames and two profiles is 738px. Doc 5.10.1's HYPOTHETICAL ~600px short prompt card
    would trip it, which is the honest answer: that profile is outside what the humanized scroll
    can enumerate, and it says so.]

WHAT VIOLATING THE RATIO RULE ACTUALLY COSTS, MEASURED (and it is not what the doc says)
-----------------------------------------------------------------------------------------
Doc 5.10.1 says a step past the ratio "produces a confidently wrong index", and an earlier
revision of the refusal above repeated it. That was true of the NAIVE heart-chaining tracker doc
5.10 measured, which fabricated a phantom tenth item. It is not true of the stack that shipped,
and the difference matters in both directions, so it is recorded rather than left as a scare
quote. Sub-sampling the clean 363px capture against the same real content, re-indexed with
`build_item_index`:

  * stride 2 — 726px per pair, ratio 0.99 against that profile's own 738px minimum spacing, dead
    centre of the aliasing band. 11 of 11 pairs measured 724..727, the index came back USABLE, and
    all nine heart page rows are EXACT (1364, 2173, 4397, 5424, 6162, 7189, 7927, 8954, 9981).
    What it cost was COVERAGE: 9 selectable items became 8, because one card was never bounded
    end to end in a single frame.
  * stride 3 — 1089px per pair, ratio 1.48. All 7 pairs refused as `beyond_window`, index
    unusable, zero blocks, zero phantoms.

So in THIS architecture — strip-NCC correspondence plus folding by absolute page position, with a
broken chain yielding no items at all — an over-large step costs items and eventually costs the
whole index, but it did not misnumber anything and could not fabricate one. The rule stays exactly
where it is: it is the difference between reading a profile and reading most of a profile, the
0.99 result is one capture rather than a licence, and nothing here says the aliasing band is safe.
What it does mean is that the ratio rule is a conservatism on top of the index's own refusals
rather than the only thing standing between the run and a wrong like — and that a future pass
tempted to relax it for speed should know it would be buying frames with items.

WHEN SPACING CANNOT BE MEASURED: A CONSERVATIVE SMALL STEP, REPORTED
----------------------------------------------------------------------
16% of corpus frames offer none of the three measurements — the band sliced every block on them.
This module falls back to the SMALLEST step it is allowed to take rather than failing loud, and
the choice is deliberate:

  * failing loud there would stop roughly one profile-read in six on a condition that is normal,
    self-correcting and not an error: the recovery IS the scroll, because any step re-frames the
    band and the next frame almost always shows a complete card;
  * the fallback cannot be the unsafe direction. It is sized against `_FALLBACK_SPACING_PX`, the
    smallest spacing ever measured in the corpus, so it is the most conservative step for which
    any evidence exists, and it is then floored by the gesture window anyway;
  * it is never silent. `ScrollStep.basis` is `STEP_FALLBACK`, `spacing_px` is None and `reason`
    says so, so a debug log or a validation pass sees every fallback step individually. That is
    the distinction the owner rule draws: a reported conservative choice is not a silent
    substitution.

Fail loud is still the answer for the two cases that mean something is WRONG rather than merely
unobserved: a segmentation that contradicts itself (`FrameSegmentation.ok` False) and the
floor/ratio conflict above. Both raise.

WHAT THE THREE CALIBRATION CAPTURES MEASURE (validated offline, 2026-08-12)
----------------------------------------------------------------------------
Every frame of the gitignored corpus was segmented and handed to `plan_scroll_step`, 200 draws
each — 29,600 plans over 148 frames and two profiles — and then again as the ORDERED loop the
enumeration pass will really run, threading `profile_min_spacing_px`. Geometry and counts only.
The headline is the third line. [Re-measured 2026-08-12 after the pixel draw above replaced the
ratio-draw-then-clamp; the numbers below are the new scheme's, and the section it replaced records
what moved.]

  * Chosen steps span **219..363px**, 139 distinct values, mean 276..302 memoryless and 244..266
    once the profile's memory has bound. Against the single value 363 that `read_scroll_frac:
    0.16` emits on every gesture of every profile, and 1299 that production's 0.55 emits. The most
    common single distance is **2-4% of gestures** rather than the 24-37% the clamped draw piled
    onto one value.
  * **Zero refusals.** No frame in the corpus, on any draw, hit the floor/ratio conflict — the
    smallest spacing anywhere is 738px against a 608px threshold.
  * **Every step/spacing ratio is <= 0.360** against the spacing the plan measured, over all
    29,600 plans; median 0.29..0.32 per capture, minimum 0.259, maximum 0.3596, and NOT ONE at or
    above 0.4, let alone the 0.5 where doc 5.10.1's failure begins. For comparison the aliasing
    cadence this replaces measures 1.26 against the same corpus's 1027px spacing and 1.76 against
    its 738px spacing.
  * The 16.2% of plans that fall back (a frame offering no measurement) come out at 219..265px,
    strictly smaller than any measured plan's ceiling, exactly as the fallback rule requires.
  * Residual, stated: measured against a profile's eventual SMALLEST spacing rather than the
    spacing in front of it, the worst ratio is 0.49 — the `_MAX_STEP_PX` exposure, and precisely
    the validated 363px cadence. It occurs on ~3 steps per profile on all three captures, before
    the running minimum has met its first short card, and never after.
  * Cost: 34 scrolls to cover profile B's 8349px and 42 to cover profile A's 10027px, against the
    23 the 363px probe used. ~1.6x the captures, well inside doc 5.10.1's "~3x the frames per
    profile, which is more dwell time, which the Signals requirements reward rather than
    penalise" — but it does mean the enumeration pass needs a capture ceiling well above
    `scroll_captures` (`HINGE_SPEC`'s default is 8, `config.yaml` sets 12, and
    `hinge._capture_limit_for_profile` adds 0..2 on top, so the effective ceiling is 12..14 —
    the validated probe ran at 48).

HOW THE ENUMERATION LOOP CALLS THIS
-------------------------------------
No driver method was added — hinge.py churn is being kept minimal while other Part B work edits
it, and there is nothing here a method would encapsulate. The whole loop, with its one piece of
state:

    seen: int | None = None                     # smallest spacing seen on THIS profile
    while ...:
        frame = self._screencap()
        seg = segment_frame(frame, content_band=self.content_band,
                            like_template=self._template("like"),
                            like_threshold=_LIKE_MATCH_THRESHOLD)
        step = plan_scroll_step(seg, profile_min_spacing_px=seen)
        if step.spacing_px is not None:
            seen = step.spacing_px if seen is None else min(seen, step.spacing_px)
        self._scroll_down_one(step.frac, step.x_frac)   # humanized: jitter, zone guard, ledger

Pass `self.content_band` (the config-merged value), NOT `spec.content_band`, or an operator
override is silently ignored — the same rule `segment_frame` and `confirm_scroll_top` state.
`seen` is per PROFILE and must be cleared wherever `_current_sigs` is, including the deck-advance
path: carrying one profile's spacing into the next is not unsafe (it can only make the step
smaller) but it is wrong, and it would quietly slow every profile after a short one.

**Both arguments, always.** `_scroll_down_one(frac=None, x_frac=None)` re-samples BOTH from the
behaviour policy if EITHER is None, so `self._scroll_down_one(step.frac)` silently discards the
computed step and issues production's 0.55 instead — the exact cadence this module exists to
avoid, with no error anywhere. `ScrollStep` therefore carries `x_frac` as well as `frac`, so the
correct call has both to hand.

The loop is closed by re-segmenting the NEXT frame, which the enumeration pass captures anyway;
`step_overshoot` is the post-hoc half, comparing what the gesture actually delivered (from
`frameshift.estimate_shift`) against the bound this plan was built to respect.

WHAT THIS MODULE DOES NOT DECIDE, AND WHICH LAYER HAS TO
----------------------------------------------------------
  * WHEN TO STOP. It sizes one step; end-of-profile, the capture ceiling and the stop conditions
    belong to the capture loop. Note that a finer step needs a higher ceiling than
    `scroll_captures`'s effective 12..14 (`HINGE_SPEC` 8, `config.yaml` 12, plus 0..2 from
    `hinge._capture_limit_for_profile`; the validated 0.16 probe ran at 48), which is the caller's
    configuration problem, not a number to smuggle in here.
  * `config.yaml`'s `read_scroll_frac`. Untouched, and deliberately: production's 0.55 stays
    correct for the swipe-deck read path, which does not index anything. The enumeration pass
    computes its own step and never consults it.
  * SCREEN IDENTITY and SCROLL-TOP. Inherited wholesale from `segment.py`: `ok is True` is not
    "this is a Hinge profile", and nothing in one frame's geometry says whether it is the top.
    Settle `_identity_of`/`_screen_is` first, and `scroll_top.confirm_scroll_top` before counting.
  * WHETHER THE GESTURE HAPPENED. This returns a number. The ledger, the forbidden-zone guard and
    the transport are `_scroll_down_one`'s, and a plan that is never spent leaves no trace here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Imported rather than re-declared, on frameshift.py's and item_index.py's precedent: the gutter
# window is already measured-and-cited in segment.py and it is the same gutter that separates the
# cards whose pitch this module measures. A second copy would be free to drift.
from .segment import _GUTTER_PX, FrameSegmentation


# =====================================================================================
# Calibration constants.
#
# Sources, same convention as segment.py / frameshift.py / item_index.py / scroll_top.py:
#   [doc]    ops/OPENER-REDESIGN.md 5.5, 5.10 and 5.10.1 (+ its 2026-08-11 addendum).
#   [corpus] a measurement of the gitignored calibration captures with segment.py's own row test
#            and frameshift.py's own strip search: 148 frames over three captures and two
#            profiles (115 hand-scrolled, 24 bot-scrolled at read_scroll_frac 0.16, 9 at 0.55).
#            Geometry and counts only — the frames are real people's profiles and never leave
#            ops/calibration/.
# =====================================================================================

# The step/spacing ratios that define the ends of the draw window. This IS the jitter — the step
# is drawn uniformly over the integer pixels BETWEEN `_STEP_RATIO_MIN` and `_STEP_RATIO_MAX` of the
# locally measured spacing, so both ends follow the card while nothing piles up on either (see the
# module docstring for the measured point masses that ruled out drawing the ratio and clamping).
#
# The centre is doc 5.10.1's rule, "at most about a third of the locally measured card spacing".
# The width is set by what the corpus says the safe side of the aliasing band looks like:
#   [corpus] ratio 0.35 (the 363px cadence against 1027px spacing) measured 23 of 23 pairs, 100%
#            frame-to-frame match rate, 0 tracking failures, 9 items and no phantom; the SAME
#            cadence against that capture's own 738px spacing is ratio 0.49 and is part of the
#            same clean run. Ratio ~1.26 (the 0.55 cadence) is the failure: 25% match rate, a
#            fabricated item, and no estimator fixes it.
# So 0.36 is a measured-safe ceiling with the doc's "about a third" still honest, and 0.26 gives
# a +-16% draw around a 0.31 mean — enough entropy to matter, narrow enough that the delivered
# distance still tracks the card rather than the dice. Every realised ratio is <= _STEP_RATIO_MAX
# or the plan raises, so 0.5 is never approached from either side.
_STEP_RATIO_MIN = 0.26
_STEP_RATIO_MAX = 0.36

# Pixels of finger travel that the content does NOT move, subtracted once per gesture.
# [corpus: frac 0.16 on a 2400px screen is a 384px drag and moves the content 363px (median of 23
# measured pairs, matching the ledger's own 362.99); frac 0.55 is a 1320px drag and moves it
# 1299px (5 pairs, at full confidence, on the strips that still have content in common). 384-363
# = 1320-1299 = 21, on two cadences 3.4x apart. A fixed offset, not a gain.]
# Android's ViewConfiguration touch slop is 8dp, which is 21.0px at this device's 420dpi (2.625)
# density bucket, so the mechanism is very likely the slop the view consumes before it begins to
# scroll — but the two measurements above are the authority, and this constant must be
# RE-MEASURED rather than re-derived if the device or the transport changes.
_TOUCH_SLOP_PX = 21

# Absolute ceiling on one enumeration step, in px of content movement.
# [corpus: 363px is the only cadence with end-to-end evidence — the 231516Z probe, 23 of 23 pairs
# measured, 0 tracking failures, 9 items recovered and no phantom, agreeing item-for-item with the
# independent hand-scrolled capture.] It exists because the local measurement is blind to the card
# BELOW the fold: a tall card in view licenses ~420px at _STEP_RATIO_MAX, and if the next card is
# short that step is spent before anything can re-measure. Capping at 363 bounds that one-step
# exposure to exactly the exposure the corpus already ran and measured clean (363 against the same
# profile's smallest 738px spacing is ratio 0.49).
_MAX_STEP_PX = 363

# The spacing the fallback step is sized against when a frame offers no measurement at all.
# [corpus: 738px is the smallest heart-bearing spacing measured anywhere — 685px, the shortest
# complete SELECTABLE block in 148 frames over two profiles, plus one 52px gutter. It appears
# both as a direct heart-to-heart pitch on profile B and as a card extent on both profiles.]
# Sizing the blind step against the smallest spacing for which any evidence exists is the most
# conservative choice that is still evidence-based; it is not a guess about the frame in hand,
# and `ScrollStep.basis` reports every time it is used.
_FALLBACK_SPACING_PX = 738

# The gutter below a card, added to a complete block's own height to get its pitch to the next
# block. The SMALLER end of segment.py's measured window is used on purpose: a smaller pitch
# estimate yields a smaller step, which is the safe direction.
# [corpus: every one of the 206 real card-to-card gutters across 139 frames measured exactly 53
# with segment.py's row test, so this is a 1px conservatism, not a modelled variation.]
_GUTTER_FLOOR_PX = min(_GUTTER_PX)

# The lane window a caller may ask for, checked here so a malformed x_frac is refused with an
# explanation instead of surfacing as an `OutOfRangeTapError` from the zone guard three frames
# later. Not measured here and not this module's to choose: it is exactly the range
# `_sample_read_scroll` and `_sample_read_step` validate any policy-sampled lane against before
# issuing a gesture (hinge.py, `0.10 <= x_frac <= 0.90` at both call sites), named rather than
# inlined so the duplication is greppable if that ever moves.
_X_FRAC_WINDOW = (0.10, 0.90)

# The lane handed to `_scroll_down_one` when a caller does not supply one. 0.5 is the driver's own
# legacy read-scroll lane (`_sample_read_scroll`'s fallback returns `self.read_scroll_frac, 0.5`),
# and the transport already jitters the actual column by +-SCROLL_X_JITTER_PX inside `scroll_x`,
# so consecutive enumeration scrolls are not pixel-identical even at a fixed x_frac. This module
# owns the scroll DISTANCE and nothing else; a caller with a behaviour policy should pass that
# policy's lane through instead of accepting this default.
_DEFAULT_X_FRAC = 0.5


# =====================================================================================
# Result vocabulary. Plain string constants rather than an Enum, matching the house style and
# keeping these directly loggable to the JSONL debug log without a serializer.
# =====================================================================================

# --- which measurement produced a spacing figure ------------------------------------
SPACING_HEART_PITCH = "heart_pitch"    # two like glyphs on one frame, exact
SPACING_CARD_PITCH = "card_pitch"      # a heart-bearing block's top to the top of the block
                                       # directly below it: that card's own period, exact
SPACING_CARD_EXTENT = "card_extent"    # one complete heart-bearing block's height + one gutter;
                                       # a LOWER bound on the pitch to the next heart

# --- what sized the step ------------------------------------------------------------
STEP_MEASURED = "measured"             # sized against this frame's own smallest spacing
STEP_FALLBACK = "fallback"             # nothing measurable here: sized against _FALLBACK_SPACING_PX


class ScrollStepError(RuntimeError):
    """No safe enumeration step can be produced, and none will be substituted.

    Raised only for things that mean something is WRONG rather than merely unobserved: a frame
    whose segmentation contradicts itself; an input this module cannot do arithmetic on (a
    degenerate screen height, an inverted ratio or frac window, a lane outside the driver's own,
    a non-positive spacing); and the one substantive refusal, the floor/ratio conflict — a
    measured spacing so small that even the smallest gesture this driver is allowed to make would
    exceed `_STEP_RATIO_MAX` of it.

    "I could not measure the spacing on this frame" is NOT one of them; that is a `STEP_FALLBACK`
    plan, reported and conservative. See the module docstring for why the two are treated
    differently.
    """


@dataclass(frozen=True)
class LocalSpacing:
    """The smallest heart-bearing period visible on one frame, and everything that voted.

    `px` is None when the frame offered no measurement at all — which is a legitimate and
    frequent answer (24 of 148 corpus frames), not an error. `evidence` carries every
    `(kind, px)` pair found, unsorted-by-value in frame order, so a validation pass can see which
    measurements agreed without re-deriving them; `px` is the minimum over it.

    [corpus: of the 124 frames that do measure, 61 combine heart-pitch with a card extent, 38
    have all three rules, 24 have card-pitch plus extent with no second heart on screen, and ONE
    has nothing but a heart pitch — which is rule 1 earning its place, since that frame bounds no
    block at all. The chosen minima are 737, 808, 827, 1026, 1027, 1161, 1165 and 1166px.]
    """
    px: int | None
    evidence: tuple[tuple[str, int], ...]

    @property
    def measured(self) -> bool:
        return self.px is not None

    @property
    def kinds(self) -> tuple[str, ...]:
        """The distinct measurement kinds that contributed, in first-seen order."""
        seen: list[str] = []
        for kind, _px in self.evidence:
            if kind not in seen:
                seen.append(kind)
        return tuple(seen)


@dataclass(frozen=True)
class ScrollStep:
    """One planned enumeration scroll: what to hand `_scroll_down_one`, and why.

    `frac` and `x_frac` are the two arguments of that call and must BOTH be passed — see the
    module docstring's "Both arguments, always": passing `frac` alone makes `_scroll_down_one`
    re-sample both from the behaviour policy and silently issue production's cadence instead.

    `step_px` is what the transport model says this `frac` will actually move the content, not
    what was asked for: the request is rounded through the transport's own integer truncation
    (`step_px_for_frac`) and reported back as the truth, so `ratio` describes the gesture that
    will really be made. `spacing_px` is None on a `STEP_FALLBACK` plan and `ratio` with it,
    because there is no local measurement to be a ratio of.

    `sized_against_px` is the distance the step was actually computed from, which is the smallest
    of: this frame's measured spacing (or `_FALLBACK_SPACING_PX` when there is none), and the
    caller's `profile_min_spacing_px`. It is carried separately from `spacing_px` so that a plan
    tightened by the profile's memory, or one sized blind, is legible as such from the result
    alone rather than by re-deriving it.

    `bound_px` is the aliasing bound this plan was built to respect — the ratio window's ceiling
    applied to `sized_against_px`, and deliberately NOT reduced by `_MAX_STEP_PX`, which is a
    planning-time conservatism about cards below the fold rather than a statement about this
    frame. It is stored rather than recomputed so `step_overshoot` holds the gesture to the bound
    the plan really used, including a caller's own `ratio_window`.

    `window_px` is the closed `(low, high)` pixel interval this step was DRAWN from, after the
    ratio window, `_MAX_STEP_PX` and the driver's own gesture floor have all been applied to it.
    It is carried because the entropy of the enumeration scroll is a safety property in its own
    right (the owner rule on fixed constants), and a window that has narrowed to one value — which
    a card spacing near the ~608px refusal threshold forces, and which no draw can undo — must be
    legible from the result rather than inferred from a suspiciously repetitive log.
    """
    frac: float
    x_frac: float
    step_px: int
    spacing_px: int | None
    sized_against_px: int
    bound_px: int
    window_px: tuple[int, int]
    ratio: float | None
    basis: str
    spacing: LocalSpacing
    reason: str


def step_px_for_frac(frac: float, screen_height: int) -> int:
    """How far the CONTENT moves for a `_scroll_down_one(frac, ...)`, in px.

    A faithful mirror of the transport, not an approximation of it: both `Adb.scroll_up` and
    `UhidTouch.scroll_up` drag from `int(h * (0.5 + frac/2))` to `int(h * (0.5 - frac/2))`, and
    the integer truncation at both ends is reproduced here rather than smoothed into `frac * h`,
    so a plan's `step_px` is the number the device will really deliver. `_TOUCH_SLOP_PX` is then
    the measured constant the content does not move; see its comment for the two cadences it was
    measured on.

    Clamped at 0: a frac small enough for the slop to swallow the whole drag moves nothing, and
    reporting a negative distance would let a caller's ratio arithmetic come out backwards.
    """
    y1 = int(screen_height * (0.5 + frac / 2))
    y2 = int(screen_height * (0.5 - frac / 2))
    return max(0, y1 - y2 - _TOUCH_SLOP_PX)


def frac_for_step_px(step_px: int, screen_height: int) -> float:
    """The `frac` whose content movement is closest to `step_px`, before any clamping.

    The inverse of `step_px_for_frac` up to the transport's integer truncation, which can leave
    the delivered step 1px either side of the request. `plan_scroll_step` corrects that
    downwards against its own safety cap rather than accepting the rounding, so this stays a
    plain closed-form inverse and callers do not have to know about the correction.
    """
    if screen_height <= 0:
        raise ScrollStepError(
            f"cannot size a scroll on a {screen_height}px screen")
    return (step_px + _TOUCH_SLOP_PX) / screen_height


def measure_local_spacing(segmentation: FrameSegmentation) -> LocalSpacing:
    """The smallest heart-bearing period this frame shows, by the three rules in the module
    docstring, or `LocalSpacing(None, ())` when it shows none of them.

    Never raises and never guesses. Heartless blocks contribute nothing — see the docstring for
    the measured reason (profile A's 215px vitals block would otherwise force an 89px step and
    then a hard stop), and note that the exclusion is what keeps the measurand a PERIOD: a block
    that repeats nothing cannot alias with anything.
    """
    evidence: list[tuple[str, int]] = []
    blocks = segmentation.blocks

    # 1. Two like glyphs on one frame. Independent of every block edge, which is why it comes
    #    first: it survives frames where the band slices both cards.
    hearts = sorted(y for block in blocks for (_x, y) in block.hearts)
    for lower, upper in zip(hearts, hearts[1:]):
        gap = upper - lower
        if gap > 0:
            evidence.append((SPACING_HEART_PITCH, gap))

    for index, block in enumerate(blocks):
        heart_bearing = len(block.hearts) == 1
        if not heart_bearing:
            # Includes BLOCK_AMBIGUOUS (two hearts), whose "period" would be the distance between
            # two glyphs segment.py has already declared a failure. Rule 1 above still records
            # that distance as heart-pitch evidence, and `plan_scroll_step` refuses the frame
            # outright on `segmentation.ok`, so nothing here has to reason about it.
            continue

        # 2. This card's top to the top of the block DIRECTLY below it: this card's own period,
        #    measurable even when its bottom edge (and so its extent, and so rule 3) is off the
        #    analysed band. Only the UPPER block must bear a heart — what is below only supplies
        #    a boundary — which is exactly what keeps the heartless vitals block from ever
        #    STARTING a 268px pair. ADJACENT, not "the next heart-bearing one": an intervening
        #    block might be hiding a heart nobody bounded, and spanning it would overstate the
        #    period, which is the unsafe direction.
        if index + 1 < len(blocks):
            below = blocks[index + 1]
            if block.top.observed and below.top.observed and below.y0 > block.y0:
                evidence.append((SPACING_CARD_PITCH, below.y0 - block.y0))

        # 3. A complete card's own extent plus the gutter below it. A lower bound on the distance
        #    to the next heart, so it errs small.
        if block.complete:
            evidence.append((SPACING_CARD_EXTENT, block.height + _GUTTER_FLOOR_PX))

    if not evidence:
        return LocalSpacing(px=None, evidence=())
    return LocalSpacing(px=min(px for _kind, px in evidence), evidence=tuple(evidence))


def plan_scroll_step(segmentation: FrameSegmentation, *,
                     x_frac: float = _DEFAULT_X_FRAC,
                     profile_min_spacing_px: int | None = None,
                     screen_height: int | None = None,
                     rng: random.Random | None = None,
                     ratio_window: tuple[float, float] = (_STEP_RATIO_MIN, _STEP_RATIO_MAX),
                     max_step_px: int = _MAX_STEP_PX,
                     fallback_spacing_px: int = _FALLBACK_SPACING_PX,
                     frac_window: tuple[float, float] | None = None) -> ScrollStep:
    """Size the next enumeration read-scroll against the spacing `segmentation` can see.

    The entry point. Hand `step.frac` AND `step.x_frac` to `_scroll_down_one` — both, always;
    passing the frac alone makes that method re-sample both and issue production's 0.55 cadence
    instead, silently. See the module docstring for the full call site.

    `profile_min_spacing_px` is the loop's one piece of state: the smallest spacing measured
    anywhere on THIS profile so far. Threading it back makes the step shrink permanently once a
    short card has been seen, which is the only defence against a card that is still below the
    fold — see the module docstring's CEILING bullet. Per profile; carrying it across a deck
    advance is wrong.

    `screen_height` defaults to the segmented frame's own height, which on this device IS the
    screen (a Hinge screencap is the full 1080x2400 framebuffer). Deriving it from the frame
    rather than from `adb.screen_size()` removes a way to be wrong: the spacing was measured in
    that frame's rows, so the step is expressed in the same rows by construction. Pass it
    explicitly only if a capture is ever scaled relative to the device.

    `frac_window` defaults to the driver's own sanctioned read-scroll distance window
    (`hinge._READ_SCROLL_FRAC_MIN` / `_READ_SCROLL_FRAC_MAX`), imported lazily so this module
    stays a leaf that hinge.py may import — the same deferred-import shape, and for the same
    reason, as `segment_frame`'s call back into `_match_glyph`. The enumeration scroll is an
    ordinary read-scroll and stays inside the window every other read-scroll is validated
    against.

    Raises `ScrollStepError` when the frame contradicts itself, when the geometry cannot be
    computed, or when the measured spacing is too small for any permitted gesture to respect
    `ratio_window`'s ceiling. It never substitutes a larger step for a refused one.
    """
    if not segmentation.ok:
        raise ScrollStepError(
            "refusing to size a scroll from a frame whose segmentation contradicts itself: "
            + "; ".join(segmentation.failures))

    height = screen_height if screen_height is not None else segmentation.frame_size[1]
    if height <= 0:
        raise ScrollStepError(f"cannot size a scroll on a {height}px screen")

    ratio_lo, ratio_hi = float(ratio_window[0]), float(ratio_window[1])
    if not 0.0 < ratio_lo <= ratio_hi:
        raise ScrollStepError(
            f"ratio window {ratio_window} is not a positive, non-inverted range")

    if not _X_FRAC_WINDOW[0] <= float(x_frac) <= _X_FRAC_WINDOW[1]:
        raise ScrollStepError(
            f"lane x_frac={x_frac} is outside the {_X_FRAC_WINDOW} window the driver validates "
            "every read-scroll lane against; refusing here rather than letting the zone guard "
            "raise on the gesture")

    frac_lo, frac_hi = _frac_window() if frac_window is None else (
        float(frac_window[0]), float(frac_window[1]))
    if not 0.0 < frac_lo <= frac_hi:
        raise ScrollStepError(
            f"read-scroll frac window {(frac_lo, frac_hi)} is not a positive, non-inverted range")

    spacing = measure_local_spacing(segmentation)
    if spacing.measured:
        basis, sizing_px = STEP_MEASURED, spacing.px
    else:
        basis, sizing_px = STEP_FALLBACK, int(fallback_spacing_px)
        if sizing_px <= 0:
            raise ScrollStepError(
                f"fallback spacing {fallback_spacing_px} is not a positive distance")

    # The loop's memory: a short card seen earlier on this profile keeps binding after it has
    # scrolled out of view, because it is still on the page we are indexing.
    if profile_min_spacing_px is not None:
        if int(profile_min_spacing_px) <= 0:
            raise ScrollStepError(
                f"profile_min_spacing_px {profile_min_spacing_px} is not a positive distance")
        sizing_px = min(sizing_px, int(profile_min_spacing_px))

    # The safety cap this plan must respect no matter what the draw or the rounding does. It is
    # the whole point of the module: the delivered step, not the requested one, has to stay under
    # ratio_hi of the spacing it was sized against.
    bound_px = int(ratio_hi * sizing_px)
    cap_px = min(bound_px, int(max_step_px))

    floor_px = step_px_for_frac(frac_lo, height)
    if floor_px > cap_px:
        raise ScrollStepError(
            f"cannot enumerate this profile: the local card spacing is {sizing_px}px "
            f"({basis}), which permits a step of at most {cap_px}px, but the smallest read-scroll "
            f"this driver may make (frac {frac_lo:g} on a {height}px screen) moves {floor_px}px. "
            "Stepping further would alias the scroll against the card spacing, which measures as "
            "an index that loses items and then refuses outright (doc 5.10.1, and the module "
            "docstring's stride measurements); no larger step is substituted")

    # THE DRAW WINDOW. Its ends are the ratio rule applied to the card in front of us, so the
    # distance still follows the content; the draw inside it is over PIXELS, so neither end
    # accumulates the point mass a clamp would have put there (module docstring, with the measured
    # before/after). Two overrides of `ratio_lo`, both in the safe direction:
    #   * the gesture floor. A spacing under ~842px puts `ratio_lo` of it below the smallest
    #     read-scroll this driver may make, so the window narrows from below — and at ~609px it
    #     closes onto the single legal distance, which is the device's answer and not a choice.
    #   * `max_step_px`. Past ~1396px of spacing the whole ratio window sits above the ceiling;
    #     rather than collapsing onto it, the low end drops back to the floor. Every value in the
    #     window is then smaller than the ratio rule would have allowed.
    low_px = max(floor_px, int(ratio_lo * sizing_px))
    if low_px >= cap_px:
        low_px = floor_px
    target_px = (rng.randint if rng is not None else random.randint)(low_px, cap_px)

    frac = frac_for_step_px(target_px, height)
    frac = min(max(frac, frac_lo), frac_hi)
    step_px = step_px_for_frac(frac, height)

    # The transport truncates twice, so the delivered step can land 1px above the request and,
    # at the top of the ratio window, 1px above the cap with it. Walk the frac down one screen
    # row at a time until the DELIVERED step is inside the cap; the loop is bounded because each
    # iteration removes at least one row of drag and cannot pass the floor, which was already
    # proven to sit under the cap above.
    while step_px > cap_px and frac > frac_lo:
        frac = max(frac_lo, frac - 1.0 / height)
        step_px = step_px_for_frac(frac, height)
    if step_px > cap_px:  # pragma: no cover — unreachable: floor_px <= cap_px was checked above
        raise ScrollStepError(
            f"could not find a frac delivering at most {cap_px}px on a {height}px screen; "
            f"closest is {step_px}px at frac {frac:g}")

    ratio = (step_px / spacing.px) if spacing.measured else None
    if spacing.measured:
        reason = (f"step {step_px}px is {ratio:.2f} of the {spacing.px}px local spacing "
                  f"({'+'.join(spacing.kinds)})")
    else:
        reason = (f"no local spacing on this frame ({len(segmentation.blocks)} block(s), "
                  f"{len(segmentation.hearts)} heart(s)); stepped {step_px}px, sized against the "
                  f"{fallback_spacing_px}px corpus minimum")
    if sizing_px != (spacing.px if spacing.measured else int(fallback_spacing_px)):
        reason += f", tightened to {sizing_px}px by this profile's smallest spacing so far"
    if low_px >= cap_px:
        # Reported rather than silent, on the same rule `STEP_FALLBACK` is reported: this gesture
        # has no distance jitter at all, and a constant distance is the shape the owner's
        # randomization rule exists to forbid. Nothing here can fix it — a smaller gesture is
        # outside the driver's sanctioned window and a larger one breaks the ratio rule — so what
        # the plan owes the log is that it happened.
        reason += (f"; the {low_px}px gesture floor meets the {cap_px}px aliasing ceiling here, "
                   "so this profile's step distance has no jitter left to draw")

    return ScrollStep(frac=frac, x_frac=float(x_frac), step_px=step_px,
                      spacing_px=spacing.px, sized_against_px=sizing_px, bound_px=bound_px,
                      window_px=(low_px, cap_px), ratio=ratio, basis=basis, spacing=spacing,
                      reason=reason)


def step_overshoot(step: ScrollStep, achieved_px: int) -> str | None:
    """Why the gesture that was actually made violates the plan it came from, or None.

    The closing half of the loop. `plan_scroll_step` bounds what we ASK for; this checks what the
    device DID, from `frameshift.estimate_shift`'s measured delta on the resulting frame pair,
    against the same bound. They can differ — a fling that carries, a card that swallows the
    gesture, a transport that did not deliver.

    Returns a reason string (never a bare bool) so a caller can put it straight into a hard stop's
    message, matching `item_crops.exclude`'s `-> str | None` shape. A negative or zero delta is
    reported too: the profile not moving at all is not a safe step, it is a stalled loop, and the
    enumeration pass must see it rather than scroll again into the same frame.

    The bound is the plan's own `bound_px`, not a module constant, so a caller that narrowed
    `ratio_window` is held to the window it chose.
    """
    if achieved_px <= 0:
        return (f"the profile did not move: measured {achieved_px}px against a planned "
                f"{step.step_px}px")
    if achieved_px > step.bound_px:
        return (f"moved {achieved_px}px, which is "
                f"{achieved_px / step.sized_against_px:.2f} of the {step.sized_against_px}px "
                f"spacing this step was sized against and past its {step.bound_px}px aliasing "
                f"bound; planned {step.step_px}px")
    return None


def _frac_window() -> tuple[float, float]:
    """The driver's sanctioned read-scroll distance window, imported at call time.

    Deferred exactly as `segment_frame`'s `_match_glyph` import is, and for the same reason:
    hinge.py is the importer of this family of modules, so a module-level import back into it
    would be a cycle the moment the driver picks this one up. Reading the bounds from hinge.py
    rather than re-declaring them keeps one source of truth for the window that every other
    read-scroll in the driver is validated against, without giving that up.
    """
    from .hinge import _READ_SCROLL_FRAC_MAX, _READ_SCROLL_FRAC_MIN
    return float(_READ_SCROLL_FRAC_MIN), float(_READ_SCROLL_FRAC_MAX)
