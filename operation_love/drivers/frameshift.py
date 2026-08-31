"""How far the CONTENT moved between two consecutive Hinge screencaps, or an honest "unknown".

`segment.py` turns ONE frame into blocks. Nothing there — deliberately, see its "WHAT THIS
MODULE DOES NOT DECIDE" section — can tell whether the card at the top of frame N+1 is the card
that was in the middle of frame N or the next one down. That correspondence is what the driver-
owned item index (ops/OPENER-REDESIGN.md 5.3) and the closed-loop enumeration scroll (5.5,
5.10.1) are built on, and it reduces to a single number per frame pair: the vertical translation.

Deliberately a leaf module, on the same terms as `segment.py`: pure functions over frame BYTES
plus explicit calibration parameters, no device, no I/O, no global state, and — unlike
segment.py, which needs `_match_glyph` at runtime — no dependency on `hinge.py` at all.

WHY THIS IS A SEPARATE FILE FROM segment.py
-------------------------------------------
Two frames, not one, and a different physical model: segment.py measures a card's own geometry
by row projection against a page-background reference, while this measures a RIGID TRANSLATION
by correlation. They share no machinery beyond the card's x-extent and the content-band row
arithmetic, both imported below rather than re-declared. Folding this into segment.py would put
a second subject of comparable size, with its own calibration corpus, under a docstring that
opens "segmentation of ONE Hinge profile frame", and would give `SegmentationError` two
meanings — one of which ("we could not look") that module's own docstring is at pains to keep
distinct from its results.

SIGN CONVENTION, stated once
----------------------------
`delta_px` is POSITIVE for an ordinary forward read-scroll: content at row `y` in frame A sits
at row `y - delta_px` in frame B, because scrolling DOWN through a profile moves content UP the
screen. This is the same convention as `tools/hinge_bot_scroll_probe.estimate_vertical_delta`,
on purpose — two conventions in one repo is how a tracker ends up chaining items backwards.

WHY MULTI-STRIP NCC, AND WHAT WAS TRIED FIRST
---------------------------------------------
Both alternatives were measured on exactly these captures and both failed (doc 5.10.1):

  * `cv2.phaseCorrelate` over the whole content band. Its own response came back 0.02-0.24 on
    the large-step capture with "implausibly uniform deltas", i.e. confidently wrong, and its
    stored per-pair deltas on that capture (-501, -501, -501, -501, -555, +272) are visibly a
    fixed artefact rather than a measurement. It is NOT reintroduced here as the primary
    estimator, and a caller wanting a second opinion should treat it as the demoted one.
  * A 1D row-profile cross-correlation. It SATURATED at its search bound and reported the bound
    as though it were an answer — the specific failure this module's `SHIFT_BEYOND_WINDOW` /
    `STRIP_PINNED` handling exists to make impossible.

What works is a bank of horizontal STRIPS cut from frame A, spread over the full analysed band,
each located independently in frame B by normalized cross-correlation
(`cv2.matchTemplate` + `TM_CCOEFF_NORMED`, the same correlation `_match_glyph` already uses), and
then required to AGREE. Each of those choices is load-bearing:

  * STRIPS, not one big crop, because a profile frame is not one rigid scene from the
    estimator's point of view: some of it leaves the band, some of it is a video card that
    changes on its own, some of it is blank white card interior that matches everywhere. A
    single crop averages all of that into one number with no way to see which part disagreed.
    Strips make disagreement countable, and counting it is the whole confidence story.
  * FULL CARD WIDTH strips, one column of results, because a list scroll has no horizontal
    component. Searching x as well would only add ways to be wrong.
  * TM_CCOEFF_NORMED because it is zero-mean and unit-variance normalized, so the page
    background's vertical GRADIENT (segment.py's trap 1(a): ~RGB(255,254,253) at y=300 down to
    ~(243,243,243) at y=2100) cannot bias a match. A plain SSD would prefer the offset that
    matched the brightness, not the content.

THE ONE THING THAT ACTUALLY DISCRIMINATES IS AGREEMENT BETWEEN STRIPS
---------------------------------------------------------------------
This is the finding that shaped every threshold below, and it is worth stating before the
constants so they are not read as a stack of quality filters. A strip's peak score and its
peak-to-runner-up separation are useful diagnostics, but neither is a correctness gate: a strip
can correlate strongly to repeated content, and a weak strip is silence about the page
translation, not a contrary vote. The answer is therefore the median plus an explicit count of
the strips that independently agree with it. `_MIN_PEAK_SCORE` only removes strips that have no
credible correspondence; `_AGREEMENT_TOLERANCE_PX`, `_MIN_AGREEING_STRIPS`, and
`_MIN_CONFIDENCE` decide whether the remaining evidence is sufficient.

A current 0.50/0.75/0.90 score-floor replay after the eligible-strip correction measures
111/112/112 of 114 hand-scroll pairs, 23/23/23 of 23 bot-scroll pairs, and 2/2/3 of 8 aliasing
pairs. Lowering the floor is not strictly monotonic: on one hand-scroll pair, an extra low-score
strip becomes a dissenting voter. The floor therefore controls both coverage and the evidence
bank that is asked to agree; it cannot be described as a confidence proxy.

WHY THE SEARCH LOOKS PAST THE WINDOW IT TRUSTS
-----------------------------------------------
Each strip is searched over its full GEOMETRIC range — every offset at which the strip's
destination still lands inside frame B's analysed band — and only afterwards is the consensus
compared against `trust_window_px`. That ordering is the fix for the prior attempt's failure.
An estimator that searches only [-W, +W] and returns its argmax cannot tell "the content moved
W" from "the content moved past W and W was the closest I was allowed to look": both come back
as W. You have to look past the fence to know you are standing at it.

So a shift beyond the trust window is REPORTED as `SHIFT_BEYOND_WINDOW` with the measured
magnitude in `consensus_px` and `delta_px` left None — a refusal that carries its evidence,
never a boundary value dressed up as a measurement. [corpus: the aliasing capture's real step is
1299px, not the ~1150 doc 5.10.1 estimated from a phase correlation that was itself unreliable
there; this measures 1299 on the strips that can still see it, on 5 of its 6 over-large pairs,
and refuses all 6 rather than pinning at the 900px window.]

WHAT THE THREE CALIBRATION CAPTURES MEASURE (validated offline 2026-08-11,
re-measured after the eligible-strip correction 2026-08-13)
--------------------------------------------------------------------------
The headline is the last line of each block: every current-default delta with a checkable
like glyph was independently corroborated; this module never reads that glyph signal.

  * `botscroll_20260811T231516Z` — profile B, bot-scrolled at the read_scroll_frac 0.16 cadence
    doc 5.10.1 recommends. 23 of 23 pairs MEASURED, no refusals. 22 of them land on 361..364 for
    a nominally constant step (median 363, matching the ledger's own 362.99 exactly) and the
    23rd on 177, which is the short final step into the end of the profile. Confidence is 1.00 on
    every measured pair. The independent like-glyph replay is 30/30 within 4px.
  * `botscroll_20260811T225314Z` — profile B, the ALIASING cadence (production's 0.55). 6 of its
    8 pairs are refused: 5 as `SHIFT_BEYOND_WINDOW` carrying `consensus_px = 1299`, one as
    `SHIFT_NO_CONSENSUS`, and 6 of 8 with `saturated` set. The two it DOES measure are genuine —
    the profile's short final step (364) and a settled frame that did not move at all (0) —
    and both are confirmed by the like-glyph cross-check. Not one phantom number.
  * `scroll_20260811T211209Z` — profile A, hand-scrolled, the hostile case doc 5.10 measured at
    0..787px per step with 11% of steps over 500px. 112 of 114 pairs measured, spanning -18 to
    +787 (the doc's own 787px maximum, to the pixel). The two refusals are one
    `SHIFT_NO_EVIDENCE` (pair 112: all 13 strips weak) and one `SHIFT_NO_CONSENSUS` (pair 113:
    one matched strip at -491px, below the three-strip quorum). The correction turns 23 former
    coverage refusals into measurements; 22 are 0px in the near-static animated-card run.
    The independent like-glyph replay is 134/134 within 4px.

The old module text reported like-glyph totals of 130/130 and 29/29. The driver's current
`hinge._match_glyph` replay instead yields 134/134 and 30/30, respectively, at the same 100% rate
and 4px tolerance. The denominator discrepancy is unresolved (possibly glyph de-duplication or a
different original y-band), so the old totals are not silently treated as equivalent.

WHAT THIS MODULE REFUSES TO GUESS
---------------------------------
Fail loud, never silently degrade (owner rule):

  * cv2/numpy missing, bytes that do not decode, two frames of different sizes, a degenerate
    band => raise `ShiftEstimationError`. There is no fallback estimate, because the thing it
    would have to fake is "which item is which", and inventing that is how a run likes the
    wrong photo.
  * A delta is returned ONLY as `delta_px`, and `delta_px` is None whenever the evidence did
    not clear every gate. "I do not know how far it moved" and "it moved 363px" are different
    return values, never the same value with a flag a caller might forget to read. `ok` is the
    one-line form of that question.
  * A strip whose best offset sits at the edge of its own searchable range is `STRIP_PINNED`
    and contributes NO value to the consensus, only a count. Its offset is a lower bound on a
    shift, not a measurement of one.

WHAT THIS MODULE DOES NOT DECIDE, AND WHICH LAYER HAS TO
--------------------------------------------------------
  * WHY the frames failed to correspond. `SHIFT_NO_CONSENSUS` covers "the scroll overshot",
    "the card is animated", "the deck advanced to a different profile" and "the screen is not a
    profile at all" with one status, because from two frames those look alike. Screen identity
    is `_identity_of`/`_screen_is`'s job and must be settled separately, exactly as it must be
    before segmenting.
  * WHETHER a measured delta is the RIGHT delta for the enumeration pass. Doc 5.10.1's rule is
    that the step must stay at most about a third of locally measured card spacing; that is a
    comparison between this number and `segment.py`'s block extents, and it belongs to the
    closed loop, not here.
  * ITEM identity. Knowing the frame moved 363px does not say which heart is which; it is the
    input to that decision, not the decision.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Imported rather than re-declared: these two are already measured-and-cited in segment.py, and
# a second copy of either would be free to drift away from the module whose blocks this has to
# line up with. `_CARD_MARGIN_PX` is the page margin the strips are cut inside; `_band_rows` is
# the content_band -> rows arithmetic, which must round IDENTICALLY here and there or a strip's
# "still inside the band" test would disagree with the band a block was found in.
from .segment import _CARD_MARGIN_PX, _band_rows


# =====================================================================================
# Calibration constants.
#
# Sources, same convention as segment.py:
#   [doc]    ops/OPENER-REDESIGN.md 5.10 / 5.10.1.
#   [corpus] a measurement of the gitignored calibration captures with THIS module's own strip
#            search at the defaults below: 145 frame pairs over three captures (114 hand-scrolled
#            pairs at steps 0..787px, 23 bot-scrolled pairs at 363px, 8 bot-scrolled pairs at the
#            aliasing cadence), 1885 individual strip searches. Geometry and counts only — the
#            frames are real people's profiles and never leave ops/calibration/.
# =====================================================================================

# How many strips are cut from frame A, evenly spaced so the first starts at the band's top row
# and the last ends on its bottom row.
#
# The count is set by the QUORUM arithmetic, not by accuracy: a shift of D px pushes the top D px
# of the band out of frame, so the strips that can still witness it are those starting at least D
# below the band top, and their number falls linearly with D. With `n` strips over a band of
# usable height `U` (= band height - strip height), the count still eligible at shift D is about
# `n * (U - D) / U`. Two floors have to stay reachable: _MIN_AGREEING_STRIPS (3) out to the trust
# window, and _MIN_SATURATION_STRIPS (2) well beyond it, or an over-large shift cannot be
# REPORTED as over-large and degrades into an unexplained refusal.
# [corpus, on the calibrated 300..2100 band with 96px strips: 13 strips leave 11 eligible at a
# shift of 0, 10 at the production 363px step, 6 at the 900px window edge, and 3 at the aliasing
# capture's real 1299px step — the last comfortably clearing _MIN_SATURATION_STRIPS, which is
# what turns that capture into 5 explicit "moved 1299px, out of window" reports instead of 5
# bare "cannot tell"s. The same corpus at 9 strips leaves only 2 eligible there and reports 2.]
# Raising it further keeps paying, but at a shrinking rate and a linear cost: 21 strips converts
# one more aliasing pair and 5 more hand-scrolled pairs, for ~35% more time per pair.
_STRIP_COUNT = 13

# Strip height in px. Two-sided:
#   * too short and a strip is not locally unique — a 16px band of a photo correlates well in a
#     lot of places, and the argmax becomes a coin toss between them;
#   * too tall and the strip spans more than one card, so it straddles a gutter and dilutes the
#     texture that locates it, while also shrinking every strip's searchable range by its own
#     height (see `_strip_search_range`).
# 96 is under half the SHORTEST block segment.py found in these captures (the 215px vitals
# block) and under a tenth of the tallest card (1114px, doc 5.10's card table), so a strip lands
# inside one block's content nearly always. Its 974x96 footprint is ~93k px, three orders of
# magnitude more evidence than the like glyph `_match_glyph` already localises reliably on the
# same frames with the same correlation.
# [corpus: only 90 of 1885 strips came out too flat to search at this height, and 1126 produced
# a usable interior peak.]
_STRIP_HEIGHT_PX = 96

# Fraction of the analysed band's height that the estimate is TRUSTED out to, when a caller
# passes no explicit `trust_window_px`. On the calibrated 1800-row band this is 900px.
#
# It is a quorum bound, not an optical one — the search itself always runs wider (see the module
# docstring). Past half the band, fewer than half the strips have any content left in common
# with frame B, so the cross-strip agreement that is the ENTIRE basis for trusting the number
# starts running out of voters. Three independent facts put the line at 0.5:
#   [corpus] at a shift of 900 on this band, 6 of 13 strips are still geometrically eligible; at
#            the aliasing capture's real 1299px step only 3 are, so the quorum is one strip from
#            collapsing and every further pixel of shift removes voters;
#   [corpus] the largest step in the 114-pair hand-scrolled capture is 787px, so a 900px window
#            excludes nothing that capture actually did — even at hostile human cadence;
#   [doc 5.10.1] the enumeration pass is required to step at most about a third of local card
#            spacing (measured 363px against 1027px), so any pair beyond this window is already
#            outside the contract the closed loop is supposed to hold and must be reported, not
#            measured.
# Expressed as a fraction rather than px so it follows content_band and screen size instead of
# silently becoming wrong on a different device.
_TRUST_WINDOW_BAND_FRAC = 0.5

# Minimum standard deviation (grey levels, 0-255) of a strip's own pixels before it is searched
# at all. A flat strip — blank white card interior, or a run of page background — correlates
# equally well everywhere, so TM_CCOEFF_NORMED's near-zero denominator makes its argmax
# arbitrary, and an arbitrary vote is worse than no vote.
# [corpus: strip stddev is sharply bimodal — 79 of 1885 strips measure under 4, another 11 land
# between 4 and 8, and the 5th percentile of the 1795 above 8 is 22.3 (median 49.2). So 8 sits in
# an almost empty gap: it drops 4.8% of strips, and there is no plausible value between 8 and 22
# that would behave differently.]
_MIN_STRIP_STDDEV = 8.0

# Correlation floor for a strip's peak. Its job is ONLY to drop strips whose content is not in
# frame B at all — it does NOT identify a correct match. Agreement across independently located
# strips, rather than an individual peak score, is the evidence this module uses to trust a delta.
# 0.75 is not a new number: it is `hinge._LIKE_MATCH_THRESHOLD`, the floor already calibrated for
# TM_CCOEFF_NORMED on these very frames, reused so this module does not introduce a second
# correlation threshold free to drift from it.
# [corpus, re-measured 2026-08-13 after the eligible-strip correction: at score floors
# 0.50/0.75/0.90, measured-pair counts are 111/112/112 of 114 hand-scroll pairs, 23/23/23 of 23
# bot-scroll pairs, and 2/2/3 of 8 aliasing pairs. At the 0.75 default, 509 of 1885 strip
# searches return STRIP_WEAK. The non-monotonic hand-scroll result is real: lower score floors
# can add dissenting voters, not merely coverage.]
_MIN_PEAK_SCORE = 0.75

# How close a strip's winning offset may come to either end of its own searchable range before
# the offset is treated as a lower bound (STRIP_PINNED) rather than a measurement. This is the
# per-strip half of the saturation rule: a correlation surface whose maximum sits on the edge of
# the window is exactly as consistent with "the true match is further out" as with "the true
# match is here", and the prior estimator's documented failure was calling that a number.
# [corpus: 160 of 1885 strips pin. Of the 1126 that do NOT, the closest any came to its own bound
# is 9px, the 1st percentile is 56px and the median is 426px — so at 8 this margin has never
# clipped a real interior match, and it is not close to doing so. On the aliasing capture it is
# what separates the strip whose range tops out at 1278 (pins, contributes no value) from the
# strips that can reach the true 1299 (match, and carry the saturation report).]
_PIN_MARGIN_PX = 8

# A strip whose geometrically searchable range is narrower than this is dropped unsearched
# (STRIP_NO_RANGE). With `_PIN_MARGIN_PX` of dead zone at each end, a range this narrow leaves at
# most 48 offsets at which a peak could even be called interior — thinner than one card gutter,
# and far too little for an argmax over a handful of candidates to mean anything.
_MIN_SEARCH_SPAN_PX = 64

# How far two strips' offsets may differ and still count as the same answer. The scroll is a
# rigid translation, so on a perfect capture this would be 0.
# [corpus: within a single pair, agreeing strips are IDENTICAL to the pixel — spread exactly 0 on
# all 23 bot-scrolled pairs. Across pairs the per-pair consensus ranges 361..364 for a nominally
# constant step, i.e. about +-1.5px of capture-to-capture jitter, and doc 5.10 independently
# measured a 2.4px residual std for heart matching under an estimated offset.] 3 covers both with
# room while being far tighter than anything that could confuse two different cards: the closest
# card-to-card heart spacing measured in these captures is 738px. The corpus does not currently
# exercise it at all, which is the point — it is slack for a device or a capture path that is
# less exact than this one, not a modelled quantity.
_AGREEMENT_TOLERANCE_PX = 3

# How many strips must independently land on the consensus before it is returned as a
# measurement. Two is not enough — two strips can be fooled by the same repeated element, and
# there is no third opinion to break the tie. Three is the smallest number that can be a
# majority of itself.
# [corpus: every one of the 23 bot-scrolled pairs reached 7-13 agreeing strips, and 112 of 114
# hand-scrolled pairs reached 3 or more, so this floor is nowhere near the working point in the
# regime the enumeration pass actually runs in. The pairs that miss it are inside the run of
# near-static frames over an ANIMATED card, where the content genuinely does not repeat between
# frames and refusing is the correct answer.]
_MIN_AGREEING_STRIPS = 3

# How many strips must agree before a shift is reported as BEYOND the trust window. Lower than
# _MIN_AGREEING_STRIPS on purpose, and the asymmetry is the fail-loud rule rather than a
# concession: this number gates a REFUSAL, and a refusal costs a retry, while _MIN_AGREEING_STRIPS
# gates a MEASUREMENT, and a wrong measurement costs a wrongly-indexed item and a like on the
# wrong photo. It is also structural — a shift big enough to be out of window is, by the same
# arithmetic that sized _STRIP_COUNT, a shift few strips can still witness, so demanding the
# full quorum for it would be demanding evidence the geometry has already destroyed.
# It is not 1: a single strip agreeing with itself is not evidence, and a spurious out-of-window
# report would stop a run that was fine.
# [corpus: on the aliasing capture, whose real step is 1299px against a 900px window, 3 of 13
# strips are geometrically eligible and this floor turns 5 of the 6 unmeasurable pairs into an
# explicit "moved 1299px, out of window" instead of a bare refusal. The same floor fires ZERO
# times across the 114 hand-scrolled and 23 bot-scrolled pairs, all of which are in window.]
_MIN_SATURATION_STRIPS = 2

# Fraction of the ELIGIBLE strips — those whose searchable range actually contains the consensus,
# so they had a real chance to see it — that must agree, before a measurement is returned.
# Deliberately a coverage ratio and not an average of scores: "6 of the 7 strips that could have
# seen this did see it" is a statement a human can check against the per-strip records, and a
# blended score is not.
#
# [corpus, re-measured 2026-08-13 with the current eligible rule over 145 pairs: at 0.4 / 0.5 /
# 0.6 the measured-pair counts are unchanged — 112/114 hostile hand-scroll, 23/23 bot-scroll,
# and 2/8 aliasing. Over SHIFT_MEASURED pairs, confidence min/median/max is 0.75/1.00/1.00 for
# the hostile capture and 1.00/1.00/1.00 for each bot capture. The hostile floor first excludes
# pairs at 0.8 (109/114 measured), then 0.9 (93/114) and 1.0 (89/114); both bot captures remain
# unchanged through 1.0.]
# 0.5 leaves room for a true coverage conflict while retaining every current-corpus measurement.
_MIN_CONFIDENCE = 0.5

# Half-width of the neighbourhood around the peak that is excluded when looking for the
# runner-up. Sized at _STRIP_HEIGHT_PX / 3 so the exclusion covers the shoulder of a single
# correlation lobe rather than clipping into a genuinely separate second match.
# Reported only, never gated — see the module docstring for the measurement that says why.
# [corpus: the resulting separation is 0.454 at the median for strips that agree with the
# verified answer and 0.048 for those that dissent, but the ranges overlap end to end (0.002 to
# 0.905 against 0.005 to 0.599), so it informs a human reading a `StripMatch` and cannot decide
# anything on its own.]
_RUNNER_UP_EXCLUSION_PX = 32


# =====================================================================================
# Result vocabulary. Plain string constants, matching segment.py and the wider house style
# (no Enum anywhere in operation_love/), and directly loggable to the JSONL debug log.
# =====================================================================================

# --- what one strip's search produced -----------------------------------------------
STRIP_MATCHED = "matched"      # an interior peak above the score floor: a usable vote
STRIP_PINNED = "pinned"        # peak on the edge of the strip's own range: a LOWER BOUND, not a
                               # value, so it never enters the consensus
STRIP_WEAK = "weak"            # best peak under the score floor: this strip's content is not
                               # anywhere in frame B's band
STRIP_FLAT = "flat"            # too little texture to locate anything (blank card interior)
STRIP_NO_RANGE = "no_range"    # too little of the band left to search: not run at all

# --- what the whole pair produced ---------------------------------------------------
SHIFT_MEASURED = "measured"          # `delta_px` is a number and is trustworthy
SHIFT_BEYOND_WINDOW = "beyond_window"  # the strips agree, on a shift outside `trust_window_px`;
                                     # the magnitude is in `consensus_px`, `delta_px` is None
SHIFT_NO_CONSENSUS = "no_consensus"  # the strips did not agree, or too few could vote
SHIFT_NO_EVIDENCE = "no_evidence"    # no strip produced a usable peak at all


class ShiftEstimationError(RuntimeError):
    """The shift could not be estimated AT ALL: no cv2/numpy, undecodable frame bytes, two
    frames of different sizes, a band with no room for a strip, or calibration parameters under
    which no pair could ever be answered.

    Distinct from every `SHIFT_*` status other than MEASURED, which are results — "I looked and
    could not tell" — carrying the per-strip evidence that says why. This exception means there
    was nothing to look at, and it exists so that "we could not look" can never be mistaken for
    "the frame did not move".
    """


@dataclass(frozen=True)
class StripMatch:
    """One strip's independent answer, and enough of its working to audit it.

    `y0`/`y1` are the strip's rows in frame A, half-open. `search` is the `(lo, hi)` INCLUSIVE
    range of offsets this strip could geometrically test — its width is what decides whether the
    strip could have seen a given consensus at all, and it is the bound `STRIP_PINNED` is
    measured against.

    `delta_px` is None for every state except `STRIP_MATCHED` and `STRIP_PINNED`; for a pinned
    strip it is a LOWER bound on the magnitude, never a measurement, and no consumer may average
    it in. `score` and `runner_up` are TM_CCOEFF_NORMED correlations in [-1, 1].
    """
    y0: int
    y1: int
    state: str
    delta_px: int | None
    score: float
    runner_up: float
    stddev: float
    search: tuple[int, int]

    @property
    def separation(self) -> float:
        """Peak minus best competing peak outside `_RUNNER_UP_EXCLUSION_PX`. Reported for
        diagnosis; measured NOT to discriminate correct strips from incorrect ones on this
        corpus, so nothing in this module gates on it."""
        return self.score - self.runner_up


@dataclass(frozen=True)
class ShiftEstimate:
    """How far the content moved between two frames, or an explicit statement that it is unknown.

    `delta_px` is the whole contract: an int when the answer is trustworthy, None otherwise, with
    no third state. A caller that reads only this field can never mistake a refusal for a
    measurement, which is the distinction that stops phantom items (doc 5.10).

    `consensus_px` is what the strips agreed on REGARDLESS of whether it was trusted, and is the
    saturation report: on `SHIFT_BEYOND_WINDOW` it carries the measured out-of-window magnitude,
    so a caller can tell "it moved much too far" (retry with a smaller scroll) from "the frames
    do not correspond at all" (stop). It is None when no consensus formed.

    `confidence` is the fraction of ELIGIBLE strips that agreed — eligible meaning the strip's
    own searchable range contained the consensus, so it had a genuine chance to see it. Strips
    that could not have seen it are excluded from the denominator; strips that could have and
    did not are counted against it.
    """
    delta_px: int | None
    confidence: float
    status: str
    saturated: bool
    consensus_px: int | None
    reason: str
    strips: tuple[StripMatch, ...]
    frame_size: tuple[int, int]
    band: tuple[int, int]
    trust_window_px: int
    agreeing: int
    dissenting: int
    eligible: int

    @property
    def ok(self) -> bool:
        """The estimate is trustworthy and `delta_px` is a number.

        False is a hard stop for anything that indexes items by it — never a reason to fall back
        to an assumed step size, which is the assumption doc 5.10 measured fabricating a phantom
        item.
        """
        return self.delta_px is not None


def _require_vision():
    """Import cv2 + numpy or raise. There is deliberately no degraded path: the fallback would
    have to invent a scroll distance, and a wrong scroll distance is indistinguishable
    downstream from a correct one."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001 — surfaced as ShiftEstimationError, never swallowed
        raise ShiftEstimationError(
            "shift estimation needs opencv-python + numpy (extra: operation-love[hinge]); "
            f"import failed: {exc}") from exc
    return cv2, np


def _decode_pair(frame_a: bytes, frame_b: bytes, cv2, np):
    """Both frames as greyscale arrays, or raise.

    Greyscale for the same reason the rest of the driver's vision layer is (segment.py's own
    note): every measurement below is a correlation of structure, colour adds three times the
    work and nothing else. Differing sizes are a hard error rather than a resize — a resized
    frame's shift would be in the wrong units, and silently rescaling a number the item index
    depends on is precisely the "silently degrade" the owner rule forbids.
    """
    img_a = cv2.imdecode(np.frombuffer(frame_a, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imdecode(np.frombuffer(frame_b, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        which = "first" if img_a is None else "second"
        raise ShiftEstimationError(
            f"the {which} frame's bytes did not decode as an image "
            f"({len(frame_a)} and {len(frame_b)} bytes)")
    if img_a.shape != img_b.shape:
        raise ShiftEstimationError(
            f"frames are different sizes ({img_a.shape[1]}x{img_a.shape[0]} vs "
            f"{img_b.shape[1]}x{img_b.shape[0]}): a shift measured across a resize would be in "
            "the wrong units")
    return img_a, img_b


def _strip_tops(r0: int, r1: int, count: int, height: int) -> list[int]:
    """Top row of each strip, evenly spread so the first begins on the band's first row and the
    last ends on its last. Rounded to whole rows, and de-duplicated so a band too short for
    `count` distinct strips yields fewer strips rather than the same strip several times (which
    would let one region of the frame vote repeatedly)."""
    last = r1 - height
    if count <= 1 or last <= r0:
        return [r0]
    step = (last - r0) / (count - 1)
    tops = [int(round(r0 + i * step)) for i in range(count)]
    return sorted(set(tops))


def _strip_search_range(y0: int, height: int, r0: int, r1: int) -> tuple[int, int]:
    """The `(lo, hi)` INCLUSIVE offsets this strip can be tested at, from geometry alone.

    A strip at rows `[y0, y0 + height)` of frame A appears at `[y0 - d, y0 - d + height)` in
    frame B under a shift of `d`. We only ever compare against frame B's analysed band, so `d`
    is testable exactly when that destination lies inside `[r0, r1)`:

        y0 - d >= r0             ->  d <= y0 - r0
        y0 - d + height <= r1    ->  d >= y0 + height - r1

    Note what this makes true for free: a strip near the top of the band can only test small or
    negative offsets, and a strip near the bottom only small or positive ones. That asymmetry is
    not a limitation to work around — it is why the bank as a whole can tell a large forward
    scroll (only the low strips still have anything to find) from a frame that did not move (all
    of them do), and it is what `eligible` in `estimate_shift` is counting.
    """
    return y0 + height - r1, y0 - r0


def _search_strip(img_b, template, *, y0: int, height: int, lo: int, hi: int, card_x0: int,
                  card_x1: int, min_peak: float, pin_margin: int, exclusion: int, cv2, np):
    """Locate one strip of frame A inside frame B. Returns `(state, delta, score, runner_up)`.

    The slab handed to `matchTemplate` is exactly as wide as the template, so the correlation
    result is a single COLUMN: one score per candidate row, which is the whole search space for
    a list that only scrolls vertically. Result row `k` corresponds to the template's top-left
    landing at frame-B row `lo_row + k`, hence a shift of `y0 - (lo_row + k)`.
    """
    lo_row = y0 - hi                      # destination top row at the LARGEST tested shift
    slab = img_b[lo_row: y0 - lo + height, card_x0:card_x1].astype(np.float32)
    scores = cv2.matchTemplate(slab, template, cv2.TM_CCOEFF_NORMED)[:, 0]

    k = int(np.argmax(scores))
    score = float(scores[k])
    delta = y0 - (lo_row + k)

    # Runner-up: the best score OUTSIDE a neighbourhood of the peak, so the shoulder of the
    # winning lobe does not get reported as a competitor. -1.0 (the floor of TM_CCOEFF_NORMED)
    # when the exclusion swallows the whole surface, which only happens on a search range barely
    # wider than _MIN_SEARCH_SPAN_PX.
    masked = scores.copy()
    masked[max(0, k - exclusion): k + exclusion + 1] = -1.0
    runner_up = float(masked.max())

    if score < min_peak:
        return STRIP_WEAK, None, score, runner_up
    if delta >= hi - pin_margin or delta <= lo + pin_margin:
        # The peak is on the edge of what this strip could see. Its offset is a lower bound on
        # the magnitude of a shift, not a measurement of one — see _PIN_MARGIN_PX.
        return STRIP_PINNED, delta, score, runner_up
    return STRIP_MATCHED, delta, score, runner_up


def estimate_shift(frame_a: bytes, frame_b: bytes, *,
                   content_band: tuple[float, float],
                   trust_window_px: int | None = None,
                   card_margin_px: int = _CARD_MARGIN_PX,
                   strip_count: int = _STRIP_COUNT,
                   strip_height_px: int = _STRIP_HEIGHT_PX,
                   trust_window_band_frac: float = _TRUST_WINDOW_BAND_FRAC,
                   min_strip_stddev: float = _MIN_STRIP_STDDEV,
                   min_peak_score: float = _MIN_PEAK_SCORE,
                   pin_margin_px: int = _PIN_MARGIN_PX,
                   min_search_span_px: int = _MIN_SEARCH_SPAN_PX,
                   agreement_tolerance_px: int = _AGREEMENT_TOLERANCE_PX,
                   min_agreeing_strips: int = _MIN_AGREEING_STRIPS,
                   min_saturation_strips: int = _MIN_SATURATION_STRIPS,
                   min_confidence: float = _MIN_CONFIDENCE,
                   runner_up_exclusion_px: int = _RUNNER_UP_EXCLUSION_PX,
                   ) -> ShiftEstimate:
    """How far frame A's content moved to reach frame B, or an explicit "unknown".

    `frame_a` is the EARLIER frame. Positive `delta_px` means the content moved UP the screen,
    which is what an ordinary forward read-scroll does; content at row `y` of A is at row
    `y - delta_px` of B. See the module docstring — the convention is shared with
    `tools/hinge_bot_scroll_probe`.

    `content_band` is the `(y0, y1)` height-fraction band that scrolls — pass the driver's
    `self.content_band` (the config-merged value), NOT `spec.content_band`, or an operator
    override is silently ignored. The band restriction is not cosmetic: the status bar, the
    sticky header, the floating like/pass buttons and the bottom nav do NOT translate when the
    content scrolls (hinge.py's `_vertical_shift_match` docstring records the same finding), so
    a strip cut across them would correlate best at a shift of zero no matter what the content
    underneath did, and would vote against every real scroll.

    THAT EXCLUSION IS CURRENTLY INCOMPLETE, AND KNOWINGLY SO. Hinge 10.1.0 draws a profile header
    pinned to the SCREEN *inside* the shipped band, so the very shape this paragraph excludes is
    back within it: on the 2026-08-28 incident capture the top strip [300,396] reports `pinned`,
    delta 0, score 1.0 on all 18 pairs, and [442,538] frequently votes a structurally meaningless
    0 into the median. It is benign TODAY and was measured to be — every delta on that capture is
    correct and its confidence is 1.000 — because the quorum is over eligible strips and the
    pinned ones are excluded from `eligible` by the pin-margin test rather than by the band. It is
    recorded here because it is the same fault class, one strip away from mattering, and because
    the band cannot simply be narrowed to fix it: the header's height is not a constant (a second,
    281px-shorter header state was measured on the same build and phone). See
    `segment._unanchored_leading_island_rows` and ops/OPENER-REDESIGN.md addendum 2026-08-28.

    `trust_window_px` overrides the derived window (`trust_window_band_frac` of the band's
    height, 900px on the calibrated 1080x2400 device). The search always runs wider than this
    — the window decides what is RETURNED, not what is looked at, which is what lets an
    over-large shift be reported as `SHIFT_BEYOND_WINDOW` instead of pinned at the bound.

    Every other keyword is a calibration constant with a measured default; they are exposed so a
    validation pass can vary one without editing the module, not because any caller should.

    Raises `ShiftEstimationError` when it cannot look at all, or when it was configured so that
    it could never answer: missing cv2/numpy, undecodable bytes, mismatched frame sizes, a band
    with no room for one strip, a bank too small to reach its own quorum, or a non-positive trust
    window. Those last three are configuration errors, and they raise rather than returning a
    refusal precisely so they cannot be mistaken for "these frames do not correspond". Returns a
    result whose `delta_px` is None when it looked and could not tell.
    """
    cv2, np = _require_vision()
    img_a, img_b = _decode_pair(frame_a, frame_b, cv2, np)
    height, width = img_a.shape

    card_x0, card_x1 = card_margin_px, width - card_margin_px
    if card_x1 - card_x0 < 1:
        raise ShiftEstimationError(
            f"frame is {width}px wide with a {card_margin_px}px card margin per side — no "
            "content columns left to correlate")

    r0, r1 = _band_rows(content_band, height)
    if r1 - r0 < strip_height_px + min_search_span_px:
        # Not merely "few strips": with less than one strip plus a usable search span, EVERY
        # strip would be STRIP_NO_RANGE and the result would be an empty-evidence answer that
        # looks like a content problem rather than the configuration problem it is.
        raise ShiftEstimationError(
            f"analysed band is too short to search: content_band {content_band} on a "
            f"{height}-row frame -> rows [{r0}, {r1}), which is under a {strip_height_px}px "
            f"strip plus the {min_search_span_px}px minimum search span")

    if strip_count < min_agreeing_strips:
        # Same reasoning as the band check: a bank too small to reach its own quorum would refuse
        # EVERY pair forever, and the refusal would read as "these frames do not correspond"
        # rather than "this call was configured to be unable to answer".
        raise ShiftEstimationError(
            f"strip_count={strip_count} is below min_agreeing_strips={min_agreeing_strips}, so "
            "no bank could ever reach a quorum and every pair would be refused")

    window = (int(round((r1 - r0) * trust_window_band_frac)) if trust_window_px is None
              else int(trust_window_px))
    if window < 1:
        raise ShiftEstimationError(
            f"trust window resolved to {window}px: nothing could ever be inside it, so every "
            "pair would come back as SHIFT_BEYOND_WINDOW regardless of what the frames did")

    strips: list[StripMatch] = []
    for y0 in _strip_tops(r0, r1, strip_count, strip_height_px):
        y1 = y0 + strip_height_px
        lo, hi = _strip_search_range(y0, strip_height_px, r0, r1)
        template = img_a[y0:y1, card_x0:card_x1].astype(np.float32)
        stddev = float(template.std())
        if stddev < min_strip_stddev:
            strips.append(StripMatch(y0=y0, y1=y1, state=STRIP_FLAT, delta_px=None, score=0.0,
                                     runner_up=0.0, stddev=stddev, search=(lo, hi)))
            continue
        if hi - lo < min_search_span_px:
            strips.append(StripMatch(y0=y0, y1=y1, state=STRIP_NO_RANGE, delta_px=None,
                                     score=0.0, runner_up=0.0, stddev=stddev, search=(lo, hi)))
            continue
        state, delta, score, runner_up = _search_strip(
            img_b, template, y0=y0, height=strip_height_px, lo=lo, hi=hi, card_x0=card_x0,
            card_x1=card_x1, min_peak=min_peak_score, pin_margin=pin_margin_px,
            exclusion=runner_up_exclusion_px, cv2=cv2, np=np)
        strips.append(StripMatch(y0=y0, y1=y1, state=state, delta_px=delta, score=score,
                                 runner_up=runner_up, stddev=stddev, search=(lo, hi)))

    return _resolve(tuple(strips), frame_size=(width, height), band=(r0, r1), window=window,
                    tolerance=agreement_tolerance_px, min_agreeing=min_agreeing_strips,
                    min_saturation=min_saturation_strips, min_confidence=min_confidence,
                    pin_margin=pin_margin_px, np=np)


def _vote_clusters(voters: Sequence[StripMatch], *, tolerance: int) -> list[list[int]]:
    """The matched offsets split into disjoint groups, each internally within `tolerance`.

    Sorted, then cut wherever two neighbouring values are further apart than one agreement
    tolerance. A rigid translation puts every voter in ONE group; two groups mean two different
    things moved by two different amounts, which a single number cannot describe and a median
    across both actively misdescribes.
    """
    deltas = sorted(s.delta_px for s in voters if s.delta_px is not None)
    if not deltas:
        return []
    groups = [[deltas[0]]]
    for delta in deltas[1:]:
        if delta - groups[-1][-1] > tolerance:
            groups.append([delta])
        else:
            groups[-1].append(delta)
    return groups


def _pins_allow(strips: Sequence[StripMatch], candidate: int) -> bool:
    """Whether every pinned strip's LOWER BOUND is consistent with `candidate`.

    A pinned strip's peak sits on the edge of its own search range, so it says "the content went
    at least this far, in this direction" and never "it went exactly here" (see `_PIN_MARGIN_PX`).
    Pinned at the top of the range means the truth is at or beyond it; pinned at the bottom means
    at or below. That is a real independent constraint and it is free, so a split-bank rescue is
    held to it even though the ordinary median path has never needed it.
    """
    for strip in strips:
        if strip.state != STRIP_PINNED or strip.delta_px is None:
            continue
        low, high = strip.search
        if strip.delta_px == high and candidate < strip.delta_px:
            return False
        if strip.delta_px == low and candidate > strip.delta_px:
            return False
    return True


def _exact_cluster_shift(strips: Sequence[StripMatch], voters: Sequence[StripMatch], *,
                         tolerance: int, min_agreeing: int) -> int | None:
    """The one pixel-exact group in a SPLIT strip bank, or None if that is not unambiguous.

    THE FAILURE THIS EXISTS FOR (Grace, 2026-08-16, frames 36/37 and 26 more pairs). A profile
    whose last card is an autoplaying video was read to the bottom. The page below the video
    stopped moving, the video did not. Nine strips of page content reported the true shift
    IDENTICALLY TO THE PIXEL (+145, then +0 once the scroll saturated); the five strips lying
    over the video reported +73..+81 — a coherent-looking but non-rigid cluster, because they had
    re-correlated against the video's OWN internal motion rather than against the page. The median
    of the two groups landed at +113, a value no strip had ever reported, so every strip
    "dissented" from it and the pair was refused. Sixty-four frames, a whole profile, and no
    numbered items — for a capture in which the answer was present, unanimous and exact.

    A median is only a consensus statistic for a UNIMODAL bank. This is the bimodal repair, and
    it turns exclusively on the discriminator the corpus already established for
    `_AGREEMENT_TOLERANCE_PX`: strips that agree about a rigid translation agree TO THE PIXEL
    (spread exactly 0 on all 23 bot-scrolled pairs), so an internally-spread group is by
    definition NOT reporting one rigid translation and cannot be the page. Deliberately NOT
    gated on `separation`, which this corpus measured as unable to tell correct strips from
    incorrect ones.

    The bar is unanimity-or-nothing, because a wrong measurement here costs a wrongly indexed
    item and a like on the wrong photo (doc 5.10):
      * exactly one group is pixel-exact AND at quorum — a second such group is a genuine
        ambiguity and is refused, never broken by taking the larger one;
      * a LARGER inexact group never wins, and never blocks: size is not evidence of rigidity,
        and letting a big smear outvote an exact answer is how the video would have won here;
      * every pinned strip's bound must still admit it.
    """
    groups = _vote_clusters(voters, tolerance=tolerance)
    if len(groups) < 2:
        return None                       # unimodal: the ordinary median path owns this bank
    exact = [group[0] for group in groups
             if len(group) >= min_agreeing and group[0] == group[-1]]
    if len(exact) != 1 or not _pins_allow(strips, exact[0]):
        return None
    return exact[0]


def _resolve(strips: tuple[StripMatch, ...], *, frame_size, band, window: int, tolerance: int,
             min_agreeing: int, min_saturation: int, min_confidence: float, pin_margin: int,
             np) -> ShiftEstimate:
    """Turn the strip bank's independent answers into one estimate, or into a refusal.

    Split out from `estimate_shift` because it is pure arithmetic over `StripMatch` records —
    no image data, no cv2 — so the whole decision surface (quorum, coverage, the beyond-window
    branch) is directly testable against hand-written strip banks with no synthetic frames at
    all. Same reason `tools/hinge_bot_scroll_probe` splits `build_pair_result` off its
    estimator.
    """
    voters = [s for s in strips if s.state == STRIP_MATCHED]
    # Saturation evidence: strips that put the content OUTSIDE the trust window, counting a
    # pinned strip's lower bound as well as a matched strip's measurement.
    #
    # The test is on the OFFSET's magnitude, not on pinnedness, and that is a correction rather
    # than a refinement. Pinnedness alone is not evidence of a large shift: on a frame that did
    # not move, the topmost and bottommost strips are pinned BY CONSTRUCTION, since a shift of 0
    # is the outer edge of what each of them can test (see `_strip_search_range`).
    # [corpus: counting pinned strips instead would have flagged 29 of the hand-scrolled
    # capture's near-static frames as saturated — the exact opposite of what they are.]
    beyond = [s for s in strips if s.delta_px is not None and abs(s.delta_px) > window]
    common = dict(strips=strips, frame_size=frame_size, band=band, trust_window_px=window)

    if not voters:
        saturated = len(beyond) >= min_saturation
        return ShiftEstimate(
            delta_px=None, confidence=0.0, status=SHIFT_NO_EVIDENCE, saturated=saturated,
            consensus_px=None, agreeing=0, dissenting=0, eligible=0,
            reason=(
                f"no strip found an interior match: {_state_tally(strips)}"
                + (f" — but {len(beyond)} strips put the content past the {window}px trust "
                   "window, so it most likely moved further than this band can follow"
                   if saturated else "")),
            **common)

    consensus = int(round(float(np.median([s.delta_px for s in voters]))))
    # THE BIMODAL CORRECTION (2026-08-16). The median above is the right statistic for one
    # population and the wrong one for two, and a strip bank goes two-population whenever part
    # of the frame moves independently of the page — an autoplaying video card being the case
    # this repo has now met twice. When that happens the median lands BETWEEN the groups, on a
    # value no strip reported, and the pair is refused for want of agreement with a number
    # nothing ever measured.
    #
    # This runs ONLY when the median has already failed its own quorum, so a pair that measures
    # today measures identically tomorrow: the rescue cannot reach a bank the ordinary rule
    # accepts, and it can therefore only ever convert a REFUSAL into a measurement. See
    # `_exact_cluster_shift` for the evidence bar, which is unanimity-or-nothing.
    split_note = ""
    if len([s for s in voters if abs(s.delta_px - consensus) <= tolerance]) < min_agreeing:
        rescued = _exact_cluster_shift(strips, voters, tolerance=tolerance,
                                       min_agreeing=min_agreeing)
        if rescued is not None:
            groups = _vote_clusters(voters, tolerance=tolerance)
            split_note = (
                f"; the bank split into {len(groups)} groups "
                f"({', '.join('/'.join(f'{v:+d}' for v in g) for g in groups)}) so the "
                f"{consensus:+d}px median described none of them, and only {rescued:+d}px is "
                "reported by strips that agree to the pixel")
            consensus = rescued
    agreeing = [s for s in voters if abs(s.delta_px - consensus) <= tolerance]
    dissenting = [s for s in voters if abs(s.delta_px - consensus) > tolerance]
    # Eligible = every strip that LOCATED ITSELF in frame B (matched or pinned) AND whose range
    # contains the consensus with pin margin to spare, i.e. every strip that had a real chance to
    # return this value as an interior peak AND actually had an opinion about where it went.
    # Strips that could not have seen it are not evidence against it; strips that could have and
    # put the content somewhere else are.
    #
    # STRIP_WEAK is DELIBERATELY not in that list, and this is the 2026-08-13 correction. A weak
    # strip is one whose own content is not anywhere in frame B's band above the score floor —
    # that is SILENCE, not dissent. It says "what I was looking at is gone", which is a statement
    # about the CONTENT changing (an animated card, a video prompt, a photo that finished loading
    # between screencaps), not a statement about how far the page moved. Counting it in the
    # denominator let one animating card outvote every strip that could actually see the scroll,
    # so the pair was refused for lack of coverage while the strips that DID speak were unanimous
    # to the pixel.
    # [corpus, re-measured 2026-08-13 over the same three captures: this converts 23 of the
    # hostile hand-scrolled capture's 25 refusals into measurements (89 -> 112 of 114 pairs), and
    # every one of them is CORROBORATED by the independent like-glyph cross-check this module
    # never looks at — 180 of 180 predicted heart positions land within 4px, against 134 of 134
    # before. 22 of the 23 measure 0px, and they are the documented run of near-static frames over
    # an animated card: an independent pixel diff puts the only changing region at rows 937..1910
    # in every one of them, identical to the row, so "did not move" is the true answer and the old
    # rule was refusing it. Neither bot-scrolled capture's result changes at all (23 of 23
    # measured before and after; the aliasing capture refuses the same 6 pairs, 5 still as
    # SHIFT_BEYOND_WINDOW).]
    #
    # What still stops a frame of unrelated content from being measured is `min_agreeing` and NOT
    # this ratio: the corpus's two surviving refusals are exactly the pairs that floor catches —
    # one with 13 weak strips and no voter at all, one where a single strip matched at -491px.
    # And the deck ADVANCING mid-capture, the other way a bank goes mostly-weak, is not this
    # module's to catch and never was (see "WHAT THIS MODULE DOES NOT DECIDE"): hinge.py's
    # per-frame identity-band guard settles screen identity before these frames are ever paired.
    eligible = [s for s in strips
                if s.state in (STRIP_MATCHED, STRIP_PINNED)
                and s.search[0] + pin_margin < consensus < s.search[1] - pin_margin]
    confidence = len(agreeing) / len(eligible) if eligible else 0.0
    counts = (f"{len(agreeing)} of {len(eligible)} eligible strips agree within {tolerance}px "
              f"({len(dissenting)} dissent, {_state_tally(strips)})")
    common_counts = dict(confidence=confidence, agreeing=len(agreeing),
                         dissenting=len(dissenting), eligible=len(eligible))

    if abs(consensus) > window:
        # The saturation report. The strips agree, and they agree on something outside the range
        # this module is willing to call a measurement, so the magnitude goes in `consensus_px`
        # and `delta_px` stays None. Requiring only `min_saturation` witnesses here is
        # deliberate: see that constant's comment for why a refusal is held to a lower bar than
        # a measurement.
        if len(agreeing) >= min_saturation and confidence >= min_confidence:
            return ShiftEstimate(
                delta_px=None, status=SHIFT_BEYOND_WINDOW, saturated=True,
                consensus_px=consensus,
                reason=(f"content moved {consensus:+d}px, beyond the {window}px window this "
                        f"estimate is trusted within — {counts}"),
                **common, **common_counts)
        return ShiftEstimate(
            delta_px=None, status=SHIFT_NO_CONSENSUS,
            saturated=len(beyond) >= min_saturation, consensus_px=None,
            reason=(f"the strips point past the {window}px trust window (median "
                    f"{consensus:+d}px) but do not corroborate each other — {counts}"),
            **common, **common_counts)

    if len(agreeing) >= min_agreeing and confidence >= min_confidence:
        return ShiftEstimate(
            delta_px=consensus, status=SHIFT_MEASURED, saturated=False, consensus_px=consensus,
            reason=f"content moved {consensus:+d}px — {counts}{split_note}",
            **common, **common_counts)

    return ShiftEstimate(
        delta_px=None, status=SHIFT_NO_CONSENSUS, saturated=len(beyond) >= min_saturation,
        consensus_px=None,
        reason=(f"no trustworthy shift: median {consensus:+d}px, but {counts} — needs "
                f"{min_agreeing} agreeing strips at confidence {min_confidence:.2f}"),
        **common, **common_counts)


def _state_tally(strips: tuple[StripMatch, ...]) -> str:
    """`"5 matched, 2 weak, 1 flat"` — the strip bank's state histogram, for `reason` strings.
    Ordered by the vocabulary above rather than by count, so two reasons from different frames
    line up when read side by side."""
    order = (STRIP_MATCHED, STRIP_PINNED, STRIP_WEAK, STRIP_FLAT, STRIP_NO_RANGE)
    counts = {state: sum(1 for s in strips if s.state == state) for state in order}
    return ", ".join(f"{n} {state}" for state, n in counts.items() if n) or "no strips"
