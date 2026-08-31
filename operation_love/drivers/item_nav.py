"""Counting navigation: put a chosen item's heart on screen, and say where to tap it.

ops/OPENER-REDESIGN.md 5.5 first stated this as "scroll to top, then step forward counting
distinct hearts, tap the k-th". THAT IS NO LONGER WHAT HAPPENS, and the change is the subject of
the section below. The module still stops at "here is where item N's heart is, and here is why I
believe it" — the tap, doc 5.6's post-tap verification and the hard stop on a miss belong to the
next layer, and nothing here touches a heart.

    navigate_to_item(driver, index, model_index, entry_reference=...) -> ItemTarget   # or raises

BOTTOM-UP: THE REWIND IS GONE (owner-approved, 2026-08-12)
-----------------------------------------------------------
The enumeration read already walks the profile top to bottom and leaves the card AT THE BOTTOM.
The first version of this module then threw all of that away: `_scroll_to_top()` rewound every
one of those ~43 gestures, and the count walked forward again from the top. One mid-profile item
cost ~120 gestures and an item deep in the profile with a retry approached 200, against ~27
before Part B existed.

Navigation now WALKS UP from where the read left the card. It is cheaper, but that is the smaller
half of the argument and it is not why the owner approved it:

  * **The anchor becomes MEASURED rather than REPLAYED.** `_scroll_to_top` sizes its swipe budget
    from a counted ledger and decides it arrived with a pixel-diff settle heuristic whose result
    "every pre-existing caller ignores" (doc 5.5). Walking up instead joins this pass's page
    space to the index's with ONE `frameshift.estimate_shift` against the very frame the index's
    last offset was measured on — a measurement, with its own refusal, that is exactly 0px
    whenever nothing moved the card between the read and the like. Every step after it is
    measured the same way and cross-checked against the index by counting hearts in reverse.
  * **It removes the mechanical down-up-down pattern.** A person reads a profile, goes back to
    the thing worth commenting on, and acts. Doc 5.11 already says so ("scrolling back up to a
    specific item to comment on it is MORE human"); what it did not say is that the rewind made
    the bot do it in the one shape a person never would — a full-speed sweep to the very top,
    then a second full-speed sweep back down. That pattern is a stronger signature than the raw
    gesture count.

What did NOT change: the enumeration pass still opens with doc 5.5's affirmative filter-chips
scroll-top gate, because that is what makes heart ordinals ABSOLUTE (`at_scroll_top=True`, and
`ItemIndex.heart_ordinal_for` refuses to answer without it). The top stops being the only anchor
we will NAVIGATE from; it does not stop being the origin the ordinals are counted from.

NOT A LEAF, and it is the first module of this family that is not. `segment.py`,
`frameshift.py`, `item_index.py`, `item_crops.py`, `scroll_top.py` and `scroll_step.py` are all
pure functions over frame bytes; this one captures frames and issues gestures, because
"navigate" is a verb about a device. What it does keep from that family is the shape: every
decision is made by one of those leaf modules, this file only sequences them, and the two
device verbs it may use are `driver._screencap()` and the driver's own humanized scroll methods.
It is a module rather than a `HingeDriver` method for `scroll_step.py`'s stated reason — hinge.py
churn is being kept minimal while other Part B work edits it, and there is nothing here a method
would encapsulate that a function taking the driver does not.

THE THREE THINGS THAT MAKE COUNTING SAFE, AND WHY EACH IS A HARD STOP
----------------------------------------------------------------------
Doc 5.5 names the first, doc 5.10.1 the second, and the 2026-08-12 addendum to 5.3 the third.
All three are closed here by refusing, never by compensating.

1. **A MEASURED entry anchor, not an assumed or replayed one.** The count needs one page row it
   trusts. Under the rewind that row was "wherever `_scroll_to_top` stopped", validated by a
   settle heuristic and corroborated by the filter-chips gate. Bottom-up, it is
   `estimate_shift(entry_reference, entry_frame)` — where `entry_reference` is the LAST frame the
   index was built from, i.e. the frame `index.offsets[-1]` was measured on. That single estimate
   puts this pass's frames into the index's own page space, so the two no longer have separate
   origins to reconcile at all. It refuses (`NAV_ANCHOR_UNMEASURED`) rather than guessing, and
   the reverse heart count re-checks it against a real heart on the very first frame that shows
   one.

2. **A step small enough that the count cannot alias.** Doc 5.10.1 measured the failure exactly:
   at production's `read_scroll_frac` the step exceeds the card spacing, "a heart translated by
   one step lands almost exactly where the next item's heart already was", and the two become
   indistinguishable — 25% frame-to-frame match rate and a fabricated tenth item. Aliasing does
   not care which way the finger went, so every gesture here is sized by
   `scroll_step.plan_scroll_step` against the spacing the current frame can actually see, and
   spent through the driver's own `_scroll_up_one` so the jitter, the forbidden-zone guard and
   the scroll ledger all still apply. Both of that plan's arguments are passed, always —
   `_scroll_up_one` refuses to be called with fewer, for the reason its own docstring gives.

3. **An index whose ordinals are ABSOLUTE.** With `at_scroll_top=False` an `ItemIndex`'s heart
   ordinals are relative to whatever was in view — a validation pass measured a mid-profile
   window returning translation (2, 3, 4, 5, 6) for what were really items 5..9 — and nothing
   on the result said so. `navigate_to_item` refuses such an index before a finger moves, and
   `ItemIndex.heart_ordinal_for` refuses to answer for one at all. That assertion still has
   exactly one legitimate source, a `ScrollTopVerdict.confirmed` — it is just made by the
   ENUMERATION pass now (`hinge._confirm_enumeration_top`) rather than a second time here.

WHY THE COUNT IS RE-DERIVED AND THEN CROSS-CHECKED, RATHER THAN READ OFF THE INDEX
-----------------------------------------------------------------------------------
The index already knows the target heart's page row, and this pass now shares the index's page
space outright, so it is more tempting than ever to scroll to a computed distance and tap. That
is still not what happens, for the reason it never was: a computed distance has nothing to check
itself against, which is how a scroll that under-delivered turns into a tap on the neighbouring
card. The entry anchor is ONE measurement; a chain hung off it unchecked would inherit any error
in it silently.

So this pass counts hearts of its own, with the same machinery the index was built from —
`frameshift.estimate_shift` for the page coordinate space, `segment_frame` for the hearts, and
`item_index`'s own `_heart_clusters` for folding the same heart seen in six frames into one. The
count and the index are then two measurements of one fixed page, and they are compared on every
frame:

  * anchored on a HEART rather than on the entry offset, so the comparison does not rest on the
    quantity it exists to check. Ascending, the anchor is the BOTTOM-most heart this pass has
    seen (it is the first one it sees and it never moves, because an upward walk only ever admits
    content ABOVE what is already folded in), matched to the index by nearest page row within
    `_ENTRY_ANCHOR_RESIDUAL_PX`. Every other heart is then compared by its distance FROM that
    anchor, which is a property of the page and identical in either origin;
  * every ordinal the count has REACHED, not only those at or below the target, so a skipped or
    doubled heart is caught at the frame it happens on rather than at the end;
  * and at the landing frame, on the card itself: its height and the heart's offset within it
    must match what the index recorded, or the count found the right ORDINAL on a card that is
    not laid out like the one the index recorded.

Disagreement is a hard stop, never a correction. Doc 5.6's owner rule — "never substitute a
different item" — is what a "nearest reachable heart" would violate, and a count that disagrees
with the index is precisely the state in which "nearest" is a guess.

THE SIGN FLIPS, ALL OF THEM, IN ONE PLACE
------------------------------------------
Walking up inverts four things, and each one is where an off-by-one would hide. They are written
out here together, and each is implemented as the `ascending` branch of a function whose other
branch is the descending original, so the two conventions sit side by side and BOTH are tested.

  | quantity              | descending (the enumeration read)        | ascending (this pass)      |
  |-----------------------|------------------------------------------|----------------------------|
  | `estimate_shift`      | delta POSITIVE (content moves up)        | delta NEGATIVE             |
  | progress / stall      | `delta <= 0` is a stall                  | `-delta <= 0` is a stall   |
  | new content arrives   | at the band's BOTTOM edge                | at the band's TOP edge     |
  | a missed heart looks  | first seen below the previous band's top | ...above its bottom        |
  | the count's anchor    | heart 1, at a confirmed top              | the bottom-most heart seen |
  | ordinals run          | upward from the anchor                   | downward from it           |
  | "we went past it"     | heart above the band's top row           | heart below its bottom row |

Two of those need no code at all and it is worth saying why, because a reader expecting seven
flips and finding five will go looking for the missing two. The page-space recurrence
(`offsets[i+1] = offsets[i] + delta`) is UNCHANGED: `estimate_shift` measures the sign, so an
upward gesture produces a negative delta and the offset decreases on its own. And the
ordinal arithmetic in `_count_disagrees` is direction-free — every cluster's ordinal is
`anchor_ordinal + (its position - the anchor's position)`, which counts up for a bottom anchor
and down for a top one without a second expression to keep in step.

THE FOURTH THING, ADDED 2026-08-12: AN INDEX OF THE WRONG PROFILE
-------------------------------------------------------------------
The three checks above compare this pass against the index. None of them compares either against
the IDENTITY of the person on screen, and no geometry available here can — Hinge's cards are
stereotyped, so in the one measured cross-profile pair a foreign profile's item 1 came back with
the same 974px height, the same 885px heart inset and the same x=938 as the index's, and every
landing comparison passed identically. The only thing that caught that pair was the ANCHOR test,
by exactly 0px (218px apart against a 218px bound, which is why that comparison is `>=` and not
`>`), and a closer pair would not have been caught at all.

Geometry is not identity, so the gate is not geometric. `navigate_to_item` opens by comparing
the CURRENT screen's `identity_band` against the fingerprint the index carries of the profile it
was built from (`item_identity.compare_profile_identity`, on the driver's own already-calibrated
sticky-header primitive). It runs on the ENTRY frame — the same screencap the page anchor is
measured from, so it costs nothing extra — i.e. before the first gesture of any kind, and refuses
on MISMATCH and equally on "cannot tell": `NAV_IDENTITY_MISMATCH` and
`NAV_IDENTITY_UNCONFIRMED`.

Bottom-up navigation makes that gate strictly easier to satisfy rather than harder, which is
worth recording because the old flow's own precondition was the awkward part: identity is NOT
readable at a scroll top (there the strip is Hinge's own chrome, byte-identical across two
different people), and the rewind's very first act was to go there. The entry frame is now
wherever the enumeration read left the card — scrolled, sticky header showing — by construction
rather than by a caller remembering to order two calls correctly.

It is not skippable, and that is structural rather than a matter of remembering: the fingerprint
is a field of the `ItemIndex`, `build_item_index` requires the band that produces it, and an
index whose identity is UNKNOWN is refused here exactly as a foreign one is. There is no keyword
on this function that turns the gate off.

TWO THINGS IT DOES NOT DO, AND ONE PRECONDITION IT IMPOSES.
It does not replace doc 5.6's post-tap crop-signature check — that answers "is the opened sheet
the ITEM we chose", this one answers "is this the PERSON the table describes", and 5.6's addendum
is explicit that the signature check must not be weakened on the assumption that navigation
pre-screened anything. It does not read a name (see `item_identity`'s docstring for why the OCR
path is deliberately not load-bearing here). And it requires that navigation be entered from
where the enumeration read leaves the card — SCROLLED, sticky header showing — because at a
scroll top that strip is Hinge's own chrome, byte-identical across two different people in the
calibration corpus, and identity is genuinely not visible there. Entering at a top is
`NAV_IDENTITY_UNCONFIRMED`, deliberately: doc 5.6's owner rule is that we never substitute, and
"we could not tell whose card this is" is not a state a tap can be justified from. Under
bottom-up navigation that precondition is also what the entry ANCHOR needs, so the two
requirements are the same requirement and neither can be satisfied without the other.

THE FIFTH THING, ADDED 2026-08-22: A SCROLL TOP AT ENTRY IS RECOVERABLE, NOT TERMINAL (BLOCKER 12)
----------------------------------------------------------------------------------------------------
The paragraph above is still true as a REQUIREMENT — identity must be visible before a single
heart is counted — but it is no longer true that entering at a scroll top ends the attempt on the
spot. SHORT profiles (one photo plus a video, nothing below the fold) end their enumeration read
exactly where it began: there is nothing to scroll past, so the read never leaves the top, and the
entry frame this function captures is the app's own profile-independent filter chips every time.
Refusing unconditionally there made every such profile unreachable regardless of what the model
chose, which is a real cost, not a theoretical one — it consumed the calibration corpus's entire
bounded skip budget on a live device.

The fix does not weaken the requirement; it adds exactly ONE way to satisfy it. When the entry
frame's identity is UNKNOWN specifically because the identity band affirmatively confirms the
scroll-top filter-chips row (the same `confirm_scroll_top` check `compare_profile_identity` itself
already ran, re-asked directly here rather than inferred by parsing `identity.reason`), one
bounded, guarded forward read-scroll is issued — through `driver._scroll_down_one`, the same
humanized primitive and ledger every other read-scroll in this driver goes through, never a raw
transport call. That single gesture is what reveals Hinge's sticky per-profile header, because the
header appears "the moment the card is scrolled at all" (`item_identity`'s own measurement) —
regardless of how deep the enumeration read actually went, so the step is sized to the SMALLEST
legal read-scroll (`scroll_step._frac_window()[0]`) rather than to production's own read cadence:
there is no card spacing here to plan an aliasing-safe distance against (nothing below the fold
has been segmented yet), and a full-size step risks landing beyond `estimate_shift`'s own trust
window — measured, a 0.55 `read_scroll_frac` step moves ~1299px on the calibrated 2400px screen,
past the 900px window that estimate is trusted within, where the smallest legal step (219px) sits
safely inside it. The behaviour policy still supplies the dwell and the LANE, on the ascending
walk's own precedent of keeping those and discarding the policy's own frac. The displacement this
step produces is MEASURED with the same `estimate_shift` every other leg in this module uses,
never assumed, and folded into `entry_offset` on top of the ordinary entry anchor (which is still
taken against the frame captured BEFORE this step, so "did anything ELSE move the card" stays
exactly the check it always was). Identity is then re-checked on the frame the step produced.

Three things are unchanged by this. No counting starts until identity is CONFIRMED on a
post-step frame — a mismatch or a still-unreadable header after the one step raises the ORIGINAL
refusal, wording intact, with a note that the step was already tried appended rather than a new
diagnosis invented. Exactly ONE such step is ever taken; there is no retry loop hiding here. And a
screen that was already scrolled on entry — the ordinary case — takes no extra frame or gesture at
all, because the recovery branch is only entered when `confirm_scroll_top` affirmatively confirms
the chips row, not merely whenever identity fails to match.

That last sentence is load-bearing on a build where the chips row is pinned to the SCREEN (Hinge
10.1.0's expanded per-profile header), because there the raw gate confirms on EVERY frame and
this branch would spend its one read-scroll on every entry regardless of where the card sits. So
the check here passes the driver's `_pinned_band_evidence()` like every other scroll-top call in
this stack: a proven-pinned build answers `check_unavailable`, which is not `.confirmed`, so the
branch declines to fire and the original refusal stands with its own wording intact. Declining is
the right action rather than a lost opportunity — the branch's whole premise is "we are at the
top and one step will reveal the header", and on a pinned build nothing establishes the premise.

WHAT IS NOT DONE HERE, AND WHICH LAYER OWNS IT
------------------------------------------------
  * The TAP. `ItemTarget.point` is a screen coordinate in `ItemTarget.frame`, and this module
    never sends it anywhere. The forbidden-zone guard lives in `hinge._tap`, so a point returned
    here is still checked at the moment it is spent, not by having been returned.
  * POST-TAP VERIFICATION and the hard stop on a miss (doc 5.6). This confirms that the item is
    on screen and where its heart is; whether the sheet that opens afterwards shows that item is
    a different question, against a reference this module does not hold. Note the constraint
    recorded for it: crop signatures DRIFT on animated cards (`item_crops.py`'s measured table,
    0.018 on a static item against 25.0 on an animated one), so that check cannot be a fixed
    threshold.
  * The CAPTURE CEILING. `_ENUMERATION_CAPTURE_LIMIT` sizes the profile READ and
    `scroll_captures` sizes every other read, while this pass computes its own frame budget from
    the index geometry and the smallest gesture the driver may make, because they are different
    jobs with different right answers. See `_frame_budget`.
  * WHICH item to navigate to. The model picks it; `model_index` arrives already chosen.
  * SCREEN IDENTITY beyond the fingerprint. A paywall, a dialog or a blank framebuffer are not
    this module's to recognise. Settle `_identity_of`/`_screen_is` BEFORE calling this.
  * STOP-AWARENESS. `should_stop`, when supplied, is checked before every screencap and upward
    gesture. Cancellation raises ``ActionCancelled`` before further device input and never taps.

THE DRIVER SURFACE THIS USES, AND NOTHING ELSE
------------------------------------------------
Written out because it is the whole coupling, and because a test double is expected to implement
exactly it (tests/test_item_nav.py does):

    driver.content_band          the CONFIG-MERGED band, not spec.content_band
    driver.identity_band         likewise, or an operator override is silently ignored
    driver._template("like")     the calibrated glyph, one home, never a default here
    driver._screencap()          the blank-frame-guarded capture, never adb.screencap()
    driver._scroll_up_one(frac, x_frac)      the counting walk's gesture, both arguments always
    driver._scroll_down_one(frac, x_frac)    one bounded call per explicit entry recovery: leave a
                                             scroll-top entry or seat a lower-edge-unresolved target
                                             BEFORE counting; both arguments always, never a retry
    driver._sample_read_step(depth, hint)    the behaviour policy's dwell and LANE

`_scroll_to_top` and `_capture_scrolls` are NO LONGER USED — that is the rewind, and its removal
is the point of this revision. `scroll_captures` sizes a different pass entirely (see
`_frame_budget`) and is likewise not read here.

The last line deserves its own. The policy returns `(dwell, frac, x_frac)` and this module
uses the dwell and the lane and DISCARDS the frac, because the frac is the one quantity the
closed loop must own. Taking the lane rather than hardcoding one keeps the owner rule about
fixed constants ("any timing/probability param must be randomized/hazard-based") satisfied by
the shipped sampler instead of by a second one invented here.

WHAT THIS COSTS, PER PROFILE
-----------------------------
Gestures spent putting the chosen item's heart on screen, counted against the two designs, using
the calibration corpus's own geometry (profile A, 10027px, ~43 enumeration steps at the loop's
measured 233..262px cadence; profile B, 8349px, ~35):

  | | rewind (until 2026-08-12) | bottom-up |
  |---|---|---|
  | best case (the LAST item)   | ~51 rewind + ~1 forward = **~52** | **0** |
  | typical (a middle item)     | ~51 + ~22 = **~73**, and the doc's own worst reading of the same flow put a mid-profile item at ~120 once `_locate_target_heart`'s own scroll-to-top and search are counted | **~22** |
  | worst case (item 1)         | ~51 + ~43 = **~94**, doubled by `_TARGET_HEART_ATTEMPTS` to **~188** | **~43** |

Bottom-up's worst case is the rewind's BEST case, and its best case is free. The saving is
exactly the rewind: an upward walk to item k costs what the downward walk to it cost, and nothing
else.

WHAT THE CALIBRATION CAPTURES MEASURE (validated offline, 2026-08-12)
----------------------------------------------------------------------
The numbers below were measured against `ops/calibration/botscroll_20260811T231516Z` under the
REWIND design and are kept because what they measure — that heart counting against this index
lands on the right card — is unchanged by walking the other way. They are stated as history, not
as evidence about the current control flow, which is validated by the synthetic closed loop in
tests/test_item_nav.py and has not been near a device (doc 9, blocker 7).

  * **All 9 of that profile's items were navigated to, one at a time, and every one landed on
    the heart the index says is its own.** 9 of 9 exact: the landed heart's page row equals the
    index's to 0px on every item (1364, 2173, 4397, 5424, 6162, 7189, 7927, 8954, 9981), its x
    matches, and the card bounded under it is the index's own height (974, 756, 974, 974, 685,
    974, 685, 974, 974). Zero refusals, zero substitutions.
  * `agreement_px` came back **0 on every one of the nine runs** — that is the WORST
    disagreement over every ordinal checked on the way, not just the target — against the 16px
    tolerance. The two passes are the same measurement twice there because the replay reproduces
    the capture's own frames; a live second pass will not be exact, which is what the tolerance
    is for.
  * **The two other captures are refused before a finger moves.** The 0.55 aliasing capture and
    the 115-frame hand-scrolled one both index UNUSABLE, and navigation stops at
    `NAV_INDEX_UNUSABLE` with no gesture issued — the refusal is inherited whole from
    `ItemIndex.usable` rather than re-derived. Unchanged by this revision: it is a pure
    inspection of the index and happens before anything looks at the device.
  * **A usable but RELATIVE index is refused too**, and separately: a mid-profile 8-frame window
    of the same capture indexes usable with translation (2, 3) at `at_scroll_top=False`, and
    navigation stops at `NAV_INDEX_RELATIVE` before the first gesture. That is blocker 3 on real
    frames rather than synthetic ones, and it is likewise unchanged.
  * **The cross-profile matrix** — profile A's index driven over profile B's frames and back, 18
    navigations — refused 18 of 18 at `NAV_IDENTITY_MISMATCH` with zero gestures issued. That
    gate now reads the entry frame instead of a post-rewind one, which is the same comparison on
    a frame that is MORE favourable to it (identity is unreadable at a scroll top).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .base import ActionCancelled
from .frameshift import ShiftEstimate, estimate_shift
from .item_identity import IdentityError, IdentityVerdict, compare_profile_identity
# The private helpers are imported rather than re-implemented, on `scroll_step.py`'s and
# `item_index.py`'s own precedent for importing `segment._GUTTER_PX`: this pass must fold hearts
# by EXACTLY the rule the index it is checked against folded them by, and a second copy of that
# rule would be free to drift into disagreeing with the thing it exists to agree with.
from .item_index import (
    _EXTENT_TOLERANCE_PX, ItemIndex, ItemIndexError, _heart_clusters, _observations)
from .scroll_step import (
    _FALLBACK_SPACING_PX, ScrollStep, _frac_window, plan_scroll_step, step_overshoot,
    step_px_for_frac)
from .scroll_top import ScrollTopError, confirm_scroll_top
from .segment import Block, FrameSegmentation, segment_frame


def _driver_pinned_evidence(driver):
    """The driver's proof that the filter-chips row is pinned to the SCREEN, or None.

    `hinge.HingeDriver._pinned_band_evidence` measures this off the enumeration read's own frames
    and offsets; this module asks for it rather than deriving it, for the same reason it asks the
    driver for `content_band` and the like template -- one calibrated answer, one home.

    Reached through `getattr` because the driver here is a duck-typed collaborator (see the
    module docstring): a driver that predates the method, or a test double that only implements
    the handful of methods this function actually calls, must get the behaviour that shipped
    before the parameter existed rather than an AttributeError. None is precisely that: "nothing
    has been proven", never "proven not pinned".
    """
    getter = getattr(driver, "_pinned_band_evidence", None)
    if not callable(getter):
        return None
    return getter()


# =====================================================================================
# Calibration constants.
#
# Sources, same convention as the six modules below this one:
#   [doc]    ops/OPENER-REDESIGN.md 5.5 (counting navigation), 5.6 (never substitute) and the
#            2026-08-12 addenda to 5.3 and 5.5.
#   [corpus] the measurements those modules' docstrings carry, taken over the gitignored
#            calibration captures. Geometry and counts only.
# =====================================================================================

# The closest known different-profile identity pair.  A navigation bound is an acceptance
# threshold, so equality is unsafe too.
_IDENTITY_FALSE_MATCH_DISTANCE = 2.565

# How far this pass's count and the reference index may disagree about the distance from the
# anchor heart to heart k before the disagreement is a hard stop.
#
# DERIVED, not chosen: `item_index._EXTENT_TOLERANCE_PX` is the slack ONE chain of shift
# estimates is allowed ([doc 5.10] a 2.4px per-step residual std, i.e. ~3.3 sigma at 8px), and
# this comparison is between TWO such chains built over the same page — the enumeration pass's
# and this one's. Each contributes its own slack, so the tolerance is their sum. Against that,
# the quantity being compared is a card pitch of 738..1027px, so a genuine off-by-one-heart is
# two orders of magnitude outside this window and cannot hide in it.
# [corpus: over the 9 navigations of the pre-bottom-up validation capture the worst disagreement
# on any ordinal is 0px against this 16 — though that replay reproduces the index's own frames,
# so it measures that the comparison is correctly ANCHORED rather than how much slack a live
# second pass needs. The 16 is the derivation above, not that measurement.]
#
# This constant is the value for an index built at the DEFAULT extent tolerance, which is the
# common case and the one the derivation above describes.  `navigate_to_item` re-derives the sum
# per index, because `item_index`'s frame-omission recovery deliberately folds its blocks at
# `_RECOVERY_EXTENT_TOLERANCE_PX` instead: for such an index the true worst case is 9 + 8, and a
# fixed 16 here would hard-stop a navigation whose two passes actually agreed.
_CROSSCHECK_TOLERANCE_PX = 2 * _EXTENT_TOLERANCE_PX

# How far this pass's own page origin may sit from the index's before the ASCENDING anchor is a
# hard stop. An EXCLUSIVE bound (`>=`), on `_TOP_ORIGIN_RESIDUAL_PX`'s precedent below.
#
# DERIVED. Bottom-up, the two passes do not have two origins to reconcile: they share the index's
# page space, joined by ONE `estimate_shift` between the frame `index.offsets[-1]` was measured on
# and the frame this pass entered on. That shift is exactly 0px whenever nothing moved the card
# between the read and the like, which is what `hinge._capture_current` produces by construction —
# both of its terminating paths leave the screen showing its own last KEPT frame (the repeated-
# frame break happens before the repeat is appended; the ceiling path's final iteration issues no
# scroll). So the slack is one shift estimate's residual plus the index's own per-block fold
# tolerance: `frameshift._AGREEMENT_TOLERANCE_PX` (3, the strip-consensus window) +
# `_EXTENT_TOLERANCE_PX` (8) = 11, rounded up to the 16 above so ONE constant governs every
# page-row comparison in this module rather than two that can drift apart.
#
# WHAT MAKES IT SAFE rather than merely tight: a mis-assignment needs the WRONG index heart to be
# nearer to the anchor than the right one, which takes half a card pitch. The smallest
# heart-bearing spacing measured anywhere in the corpus is `scroll_step._FALLBACK_SPACING_PX`
# (738px), so the anchor could not pick a neighbour until 369px — 23x this bound. The assertion
# below states that rather than leaving it to be re-derived.
_ENTRY_ANCHOR_RESIDUAL_PX = _CROSSCHECK_TOLERANCE_PX
if _ENTRY_ANCHOR_RESIDUAL_PX * 2 >= _FALLBACK_SPACING_PX // 2:
    # Import-time safety invariant: unlike `assert`, this must remain active under `python -O`.
    raise RuntimeError(
        "the ascending anchor bound must stay far under half the smallest measured heart pitch, "
        "or the nearest-index-heart match could choose a neighbour")

# How far two independently confirmed scroll tops may sit apart in page rows. An EXCLUSIVE bound:
# a gap of exactly this much (plus the comparison's own tolerance) is already outside what two
# confirmations can produce, and the test below is `>=` rather than `>` for that reason.
#
# [doc 5.5's 2026-08-12 addendum, and `scroll_top.py`'s stated residual: the gate confirms that
# Hinge's sticky per-profile header has not slid in, which BOUNDS the scroll offset rather than
# pinning it at page row 0. Measured on the hand-scrolled capture, the swap happens somewhere in
# (0, 202]px — its first two frames are 0px apart and both CONFIRM, the next is 202px further on
# and REFUTES. So a confirmed top's own offset lies in [0, 202): 202 is the first value measured
# to refute, not the last one measured to confirm, and two confirmed origins therefore differ by
# strictly less than it.]
#
# It is the DESCENDING anchor bound — the case where both passes are anchored on their own
# confirmed scroll top and the first heart each of them sees is claimed to be heart 1. Bottom-up
# navigation does not take that path (see `_ENTRY_ANCHOR_RESIDUAL_PX`), and the constant stays
# because the descending branch of `_count_disagrees` stays: the two conventions are written and
# tested side by side precisely so neither can drift into agreeing with itself.
_TOP_ORIGIN_RESIDUAL_PX = 202

# Frames the budget allows beyond the number of minimum-size steps it would take to bring the
# target card's top edge into the analysed band. Four, and each one is a specific frame rather
# than a safety margin:
#   1  the entry frame itself, which is captured before any gesture;
#   1  the one extra step that finishes bringing the card's TOP edge inside the band after its
#      heart has appeared — walking up, a card's bottom edge enters the band first and its heart
#      sits ~89px above that edge (segment.py's stamping geometry), i.e. well under one minimum
#      step, so it is one frame;
#   2  spare, for gestures the transport under-delivers. This pair is the bound, not a
#      measurement; the divisor it is added to is already the SMALLEST gesture the driver may
#      make, so a real run using the loop's 219..363px steps arrives with far more headroom than
#      this.
_FRAME_BUDGET_SLACK = 4


# --- refusal codes -------------------------------------------------------------------
# Plain strings rather than an Enum, matching the house style in `segment.py` / `scroll_step.py`,
# and directly loggable to the JSONL debug log without a serializer.
NAV_INDEX_UNUSABLE = "index_unusable"                  # the index contradicts itself
NAV_INDEX_RELATIVE = "index_not_at_scroll_top"         # ordinals are relative: blocker 3
NAV_ORDINAL_OUT_OF_RANGE = "ordinal_out_of_range"      # no such model item
NAV_CALIBRATION_INVALID = "identity_calibration_invalid"  # a caller supplied no safe identity bound
NAV_IDENTITY_MISMATCH = "identity_mismatch"            # this index describes a DIFFERENT profile
NAV_IDENTITY_UNCONFIRMED = "identity_unconfirmed"      # whose card this is cannot be established
NAV_ANCHOR_UNMEASURED = "entry_anchor_unmeasured"      # the screen could not be put in the index's
                                                       # page space: bottom-up's zero point
NAV_ENTRY_STEP_UNMEASURED = "entry_step_unmeasured"    # the ONE scroll-top recovery step's own
                                                       # displacement could not be measured
                                                       # (2026-08-22, blocker 12)
NAV_ENTRY_POSITION_UNRESOLVED = "entry_position_unresolved"  # a one-shot forward positioning
                                                               # recovery could not be measured or
                                                               # did not leave the chosen card whole
NAV_ITEM_BELOW_ENTRY = "item_below_entry"              # the target is BELOW where the read ended,
                                                       # and walking up only moves it further away
NAV_FRAME_CONTRADICTS = "frame_contradicts_itself"     # segment.py failed on a frame
NAV_CHAIN_BROKEN = "chain_broken"                      # frameshift refused a pair
NAV_SCROLL_STALLED = "scroll_stalled"                  # the gesture moved nothing (or backwards)
NAV_SCROLL_OVERSHOT = "scroll_overshot"                # past the plan's own aliasing bound
NAV_HEART_MISSED = "heart_missed"                      # a heart was on screen and not counted
NAV_COUNT_DISAGREES = "count_disagrees_with_index"     # two measurements, one page, two answers
NAV_ITEM_NOT_FULLY_VISIBLE = "item_never_fully_visible"  # scrolled past it, or it never bounded
NAV_BUDGET_EXHAUSTED = "budget_exhausted"              # never reached the target ordinal


@dataclass(frozen=True)
class NavigationRecovery:
    """A measured terminal position from a navigation that correctly refused to land.

    This is deliberately *not* a target and cannot authorize a tap or a retry.  It exists for
    the still-photo candidate walk, whose navigation is optional but whose real gestures still
    owe the profile a measured return to its entry anchor.  ``page_shift_px`` is the complete
    signed displacement from this navigation's entry frame to ``frame`` in frameshift's
    convention (positive is content up).  It is present only when every leg, including the
    refusing one, was measured.
    """

    frame: bytes
    page_shift_px: int
    step_index: int
    frame_index: int
    planned_step_px: int
    achieved_step_px: int
    bound_px: int
    frac: float
    window_px: tuple[int, int]
    basis: str
    spacing_px: int | None
    sized_against_px: int
    measurement_delta_px: int
    measurement_status: str
    measurement_confidence: float
    measurement_agreeing: int
    measurement_dissenting: int
    measurement_eligible: int
    violation: str


class ItemNavigationError(RuntimeError):
    """Navigation refused, and nothing was tapped.

    `code` is one of the `NAV_*` constants above, so a caller can route the stop (the hub shows
    the reason, doc 5.5's stop-condition rule) without parsing prose. `frame` is the screencap
    the refusal was decided on, when there was one, so a bug report can show the operator what
    the run was looking at; `anchor` carries the entry `ShiftEstimate` once one has been taken,
    because "we could not join this screen to the index's page space" and "we joined it and then
    the count disagreed" are different diagnoses.

    Every one of these is a stop, not a fallback. Doc 5.6: "if we cannot land on the item the
    model chose, even after multiple tries, the run stops. No falling back to `hearts[0]`, no
    'closest reachable item'." This class exists so that rule has exactly one shape.
    """

    def __init__(self, code: str, message: str, *, frame: bytes | None = None,
                 frame_index: int | None = None, anchor: "ShiftEstimate | None" = None,
                 recovery: NavigationRecovery | None = None):
        super().__init__(message)
        self.code = code
        self.frame = frame
        self.frame_index = frame_index
        self.anchor = anchor
        self.recovery = recovery


@dataclass(frozen=True)
class ItemTarget:
    """Where item `model_index`'s heart is right now, and the whole case for believing it.

    `point` is `(x, y)` in `frame`'s own screen rows — the coordinate a tap would use, and the
    only field a caller strictly needs. Everything else is the evidence doc 5.6's verification
    step and a stop record are going to want, kept together so neither has to re-derive it.

    `frame` is the landing screencap itself, carried because the next layer's post-tap check
    needs the BEFORE picture of the card it is about to open and re-capturing it would be a
    different frame. `block_frame_rows` are that card's rows in that frame, so a crop for the
    comparison is a slice rather than a search.

    `agreement_px` is the largest disagreement between this pass's count and the reference index
    over every ordinal it checked — the headline number of the cross-check, kept so a debug log
    records how much margin there actually was rather than only that it passed.

    `identity` is the entry gate's verdict, always a MATCH here because nothing else returns an
    `ItemTarget`. The distance it matched at is the margin a debug record wants, and a run that
    later goes wrong should be able to show that the person on screen was checked before a finger
    moved rather than assumed.

    `anchor` is the `estimate_shift` against `entry_reference` from the frame captured BEFORE this
    function moved anything, and `entry_offset` is the page offset it produced for the entry
    frame. They are the whole of bottom-up's zero point, so a stop record that wants to ask "was
    the origin right" has the measurement rather than a claim. When the entry screen needed one
    scroll-top recovery step first (2026-08-22, blocker 12), `entry_offset` also folds in that
    step's OWN separately measured displacement on top of `anchor`, which stays the plain
    entry_reference-vs-first-frame measurement either way. `page_offset` is the offset of the
    LANDING frame; the two differ by exactly the sum of `shifts`, which are all negative.
    """
    point: tuple[int, int]
    model_index: int
    heart_ordinal: int
    frame: bytes
    frame_index: int
    block_frame_rows: tuple[int, int]
    block_page_rows: tuple[int, int]
    page_offset: int
    hearts_counted: int
    agreement_px: int
    identity: IdentityVerdict
    anchor: ShiftEstimate
    entry_offset: int
    steps: tuple[ScrollStep, ...]
    shifts: tuple[ShiftEstimate, ...]
    offsets: tuple[int, ...]
    reason: str

    @property
    def scrolls(self) -> int:
        """REVERSE gestures spent getting here — zero when the read already left the card on the
        chosen item, which is the ordinary case for the last item on a profile. Every one of them
        is in the driver's scroll ledger; see `hinge._scroll_up_one` for why a reverse gesture is
        appended to that ledger rather than popping a forward one off it."""
        return len(self.steps)

    @property
    def climbed_px(self) -> int:
        """How far up the page this pass actually travelled, as measured rather than as planned.
        Non-negative: every `shifts` delta is negative, so the sum is negated once here instead
        of at each of the places that want to report a distance."""
        return -sum(est.delta_px or 0 for est in self.shifts)


def _like_threshold() -> float:
    """The calibrated like-glyph match floor, imported at call time.

    Deferred exactly as `scroll_step._frac_window` and `segment_frame`'s `_match_glyph` import
    are, and for the same reason: hinge.py is the importer of this family, so a module-level
    import back into it would be a cycle the moment the driver picks this one up. Read from
    hinge.py rather than re-declared because `_LIKE_MATCH_THRESHOLD` has exactly one home and a
    default here would be a second copy free to drift — the same rule `segment_frame` and
    `build_item_index` state by refusing to default it at all.
    """
    from .hinge import _LIKE_MATCH_THRESHOLD
    return float(_LIKE_MATCH_THRESHOLD)


def index_hearts(index: ItemIndex) -> dict[int, tuple[int, int]]:
    """Heart ordinal -> `(x, page_y)` for every heart the index holds, selectable or not.

    Doc 5.3's index space in dictionary form: it spans EVERY heart on the page, including hearts
    on blocks the model may not choose and on blocks that were never bounded end to end, because
    "a heart that exists and is not counted here is an off-by-N in the tap coordinate".

    An `ITEM_AMBIGUOUS` block holds two hearts and names the first of them, so its second
    occupies the following ordinal; that block also makes the index unusable, which is checked
    before this is ever called.
    """
    out: dict[int, tuple[int, int]] = {}
    for block in index.blocks:
        if block.heart_ordinal is None:
            continue
        for offset, heart in enumerate(block.hearts):
            out[block.heart_ordinal + offset] = heart
    return out


def _min_heart_pitch(hearts: dict[int, tuple[int, int]]) -> int | None:
    """The smallest distance between consecutive hearts anywhere on the indexed page, or None.

    This is `scroll_step.SPACING_HEART_PITCH` — the exact measurand, two like glyphs' page rows
    — measured over the WHOLE profile rather than over one frame, which is something only a
    completed enumeration pass can offer. Threading it into the first plan as
    `profile_min_spacing_px` closes the one hole that module documents by construction rather
    than by a constant: "the local measurement cannot see the card under the fold", so its
    `_MAX_STEP_PX` ceiling exists to bound a short card the current frame has not met yet. Here
    the index has already met every card on the page.
    """
    ordinals = sorted(hearts)
    gaps = [hearts[b][1] - hearts[a][1]
            for a, b in zip(ordinals, ordinals[1:], strict=False)
            if hearts[b][1] > hearts[a][1]]
    return min(gaps) if gaps else None


def _frame_budget(entry_offset: int, target_page_y0: int, band_y0: int,
                  screen_height: int) -> int:
    """How many frames this navigation may capture before it is a refusal.

    Derived rather than declared, and deliberately NOT `scroll_captures`: that constant sizes the
    ordinary profile READ (12 in this tree) and `_ENUMERATION_CAPTURE_LIMIT` sizes the enumeration
    one, while this is "how many minimum-size steps could it possibly take to bring ONE known
    card's top edge into the band from where the read left off". The divisor is the smallest
    gesture the driver is allowed to make — `_READ_SCROLL_FRAC_MIN` through the measured transport
    model, 219px on the calibrated device — so the budget is an upper bound on a loop that will
    really be stepping 219..363px, and exceeding it means something other than slow progress is
    wrong.

    THE DISTANCE IS THE ASCENT, and the sign is the whole of what changed with bottom-up: it is
    how far the band's TOP row must travel UP the page to reach the target card's top edge, i.e.
    `(entry_offset + band_y0) - target_page_y0`. Under the rewind it was the descent from the
    page's first row to the target heart. A target the entry frame already shows completely gives
    0 here and a budget of exactly `_FRAME_BUDGET_SLACK` frames, which is correct: that
    navigation issues no gestures at all.

    Everything is in the INDEX's page space, which this pass shares outright (see the entry
    anchor), so unlike the rewind there is no origin residual for the slack to have to cover.
    """
    floor_px = step_px_for_frac(_frac_window()[0], screen_height)
    if floor_px <= 0:
        raise ItemNavigationError(
            NAV_BUDGET_EXHAUSTED,
            f"the smallest read-scroll this driver may make moves {floor_px}px on a "
            f"{screen_height}px screen, so no number of them reaches anything")
    distance = max(0, (entry_offset + band_y0) - target_page_y0)
    return math.ceil(distance / floor_px) + _FRAME_BUDGET_SLACK


def _no_heart_was_missed(clusters, observations, offsets, band: tuple[int, int], tolerance: int,
                         *, ascending: bool) -> str | None:
    """Why this pass's heart count cannot be trusted, or None.

    The count's silent failure mode is a heart the glyph matcher did not find while it was on
    screen: nothing contradicts it, the ordinals past it are all off by one, and the tap lands one
    card away. It is detectable because a scroll admits new content at exactly ONE edge, and WHICH
    edge is the whole of the direction dependence here:

        descending (the enumeration read)  new content arrives at the band's BOTTOM.
                                           A heart first seen in frame `i` must have been BELOW
                                           frame `i-1`'s band; one at a page row that band already
                                           covered was on screen and was not reported.
        ascending  (bottom-up navigation)  new content arrives at the band's TOP.
                                           A heart first seen in frame `i` must have been ABOVE
                                           frame `i-1`'s band; one at a page row that band already
                                           covered was on screen and was not reported.

    Both branches are exercised by the tests, deliberately, even though production only walks one
    way: this is the predicate where an inverted comparison would be invisible — it would simply
    never fire — so the descending form is kept as the executable statement of the mirror rather
    than as a comment claiming what the ascending one is the reverse of.

    Frame 0 is exempt by construction in both directions: it has no predecessor, so every heart it
    shows is legitimately new to this pass.

    Returns the reason string rather than raising, on `item_crops.exclude`'s and
    `scroll_step.step_overshoot`'s `-> str | None` shape, so the caller owns the stop.
    """
    if not clusters:
        return None
    band_y0, band_y1 = band
    first_seen: dict[int, int] = {}
    for obs in observations:
        for _x, page_y in obs.hearts:
            k = min(range(len(clusters)), key=lambda j: abs(clusters[j][1] - page_y))
            first_seen[k] = min(first_seen.get(k, obs.frame_index), obs.frame_index)
    for k, frame_index in sorted(first_seen.items()):
        if frame_index == 0:
            continue
        page_y = clusters[k][1]
        if ascending:
            previous_edge = offsets[frame_index - 1] + band_y0
            covered = page_y >= previous_edge + tolerance
            where = f"up to page row {previous_edge}"
        else:
            previous_edge = offsets[frame_index - 1] + band_y1
            covered = page_y <= previous_edge - tolerance
            where = f"down to page row {previous_edge}"
        if covered:
            return (f"the heart at page row {page_y} was first seen in frame {frame_index}, but "
                    f"frame {frame_index - 1}'s analysed band already covered that row ({where}) "
                    "and did not report it. A heart that was on screen and not counted shifts "
                    "every ordinal counted past it, which is an off-by-one tap and not a missing "
                    "item")
    return None


def _count_disagrees(clusters, hearts: dict[int, tuple[int, int]], *,
                     heart_count: int, tolerance: int, ascending: bool,
                     anchor_residual_px: int | None = None
                     ) -> tuple[str | None, int, dict[int, tuple[int, int]]]:
    """Compare this pass's count against the reference index.

    Returns `(reason, worst_px, counted)`, where `counted` maps each heart ordinal this pass has
    reached to THIS pass's own `(x, page_y)` for it. The caller navigates by `counted`, not by
    the index's rows: the index says where a heart was during the READ, this says where the
    count has just found it, and using the second is what keeps the landing decision on the
    measurement rather than on the table it is being checked against. The two agree to
    `tolerance` whenever `reason` is None, which is what makes either safe to quote in a message.

    Anchored on a HEART rather than on a page row, in both directions, because a comparison built
    on the page origin cannot check the page origin. Everything below the anchor test is a
    distance BETWEEN two hearts, which is a property of the page and therefore the same number in
    either pass's coordinates.

    WHICH heart is the anchor is the direction dependence, and it is the only one:

        descending  the FIRST heart this pass saw, claimed to be the profile's heart 1 because
                    the pass began at an affirmatively confirmed scroll top. The bound is
                    `_TOP_ORIGIN_RESIDUAL_PX + tolerance`: two confirmed tops differ by strictly
                    less than that, so a wider gap means either the first heart counted is not
                    heart 1 or the index describes a different profile.
        ascending   the LAST heart this pass has seen, i.e. the bottom-most — the first one it
                    met and one that never moves, because an upward walk only ever folds in
                    content ABOVE what it already has. Its ordinal is not known a priori (the
                    bottom of a profile is not guaranteed to show the last card), so it is matched
                    to the index by NEAREST page row within `_ENTRY_ANCHOR_RESIDUAL_PX`. That is
                    the check on the entry anchor: the shift estimate is only believed once a real
                    heart lands where it predicts.

    The ordinal arithmetic itself is direction-FREE and that is deliberate — cluster `j` is
    ordinal `anchor_ordinal + (j - anchor_position)`, which counts upward from a first-heart
    anchor and downward from a last-heart one with no second expression to keep in step. An
    off-by-one can therefore only be in the choice of anchor, which is four lines and is tested
    both ways.

    EVERY ordinal this pass has counted is checked, not only those at or below the target. A
    skipped or doubled heart then shows up at the ordinal it happened on, which is a far better
    stop message than a mismatch discovered at the end, and it stops the run before it scrolls
    further.
    """
    if not clusters:
        return None, 0, {}
    if len(clusters) > heart_count:
        return (f"this pass has counted {len(clusters)} distinct hearts but the index recorded "
                f"{heart_count} on the whole profile — one of the two is seeing an item that is "
                "not there, and neither can be preferred over the other"), 0, {}
    if not hearts:
        return ("the index holds no hearts at all, so there is nothing to anchor this pass's "
                "count against"), 0, {}

    if ascending:
        anchor_position = len(clusters) - 1
        anchor_bound = ((_ENTRY_ANCHOR_RESIDUAL_PX if anchor_residual_px is None
                         else int(anchor_residual_px)) + tolerance)
        anchor_ordinal = min(hearts, key=lambda k: (abs(hearts[k][1] - clusters[-1][1]), k))
        why = ("This pass shares the index's page space through one measured shift against the "
               "frame the index's last offset was taken on, which bounds the two origins well "
               "inside half a card pitch, so either that anchor measurement is wrong or this "
               "index describes a different profile than the one on screen")
    else:
        anchor_position = 0
        # `>=`, not `>`, and the difference is not cosmetic: `_TOP_ORIGIN_RESIDUAL_PX` is the
        # first offset MEASURED TO REFUTE, so two confirmed tops differ by strictly less than it
        # and a gap of exactly the bound is already impossible between two confirmations. A
        # validation pass measured the case it matters on — profile A's index driven over profile
        # B's frames, whose heart 1 sit exactly 218px apart against this exact 218px bound — and a
        # strict `>` admitted it by 0px.
        anchor_bound = _TOP_ORIGIN_RESIDUAL_PX + tolerance
        if 1 not in hearts:
            return ("the index holds no heart with ordinal 1, so there is nothing to anchor this "
                    "pass's count against"), 0, {}
        anchor_ordinal = 1
        why = ("Both passes began at an affirmatively confirmed scroll top, which bounds their "
               "page origins under that much apart, so either the first heart counted here is "
               "not the first heart of the profile — and every ordinal counted from it is "
               "shifted — or this index describes a different profile than the one on screen")

    anchor_page_y = clusters[anchor_position][1]
    anchor_gap = abs(anchor_page_y - hearts[anchor_ordinal][1])
    if anchor_gap >= anchor_bound:
        return (f"this pass's anchor heart is at page row {anchor_page_y} but the index's nearest "
                f"heart (ordinal {anchor_ordinal}) is at {hearts[anchor_ordinal][1]}, "
                f"{anchor_gap}px apart against a {anchor_bound}px bound. {why}"), 0, {}

    worst = 0
    counted_hearts: dict[int, tuple[int, int]] = {}
    for j, cluster in enumerate(clusters):
        k = anchor_ordinal + (j - anchor_position)
        if k not in hearts:
            return (f"this pass has counted a heart the index has no ordinal {k} for, so the two "
                    "disagree about how many hearts sit above or below the anchor"), worst, {}
        counted = cluster[1] - anchor_page_y
        indexed = hearts[k][1] - hearts[anchor_ordinal][1]
        worst = max(worst, abs(counted - indexed))
        if abs(counted - indexed) > tolerance:
            return (f"heart {k} sits {counted}px from the anchor heart {anchor_ordinal} in this "
                    f"pass but {indexed}px from it in the index, a disagreement of "
                    f"{abs(counted - indexed)}px against a {tolerance}px tolerance. Two "
                    "measurements of one page do not agree, so the count may be off by an item "
                    "and there is no basis for choosing between them"), worst, {}
        counted_hearts[k] = cluster
    return None, worst, counted_hearts


def _heart_in_frame(seg: FrameSegmentation, frame_y: int, tolerance: int
                    ) -> tuple[tuple[int, int], Block] | None:
    """This frame's own detection of the heart at frame row `frame_y`, with its block.

    The row comes from a cluster representative, which is a MEDIAN over every frame that saw
    that heart; the point returned is this frame's actual detection, because that is the row a
    tap on THIS screen has to use. Returns None when this frame does not show it.
    """
    best: tuple[tuple[int, int], Block] | None = None
    best_gap = tolerance + 1
    for block in seg.blocks:
        for heart in block.hearts:
            gap = abs(heart[1] - frame_y)
            if gap < best_gap:
                best, best_gap = (heart, block), gap
    return best


def navigate_to_item(driver, index: ItemIndex, model_index: int, *,
                     entry_reference: bytes,
                     identity_match_max_dist: float,
                     selected_model_item_index: int | None = None,
                     should_stop=None,
                     max_frames: int | None = None,
                     crosscheck_tolerance_px: int | None = None,
                     extent_tolerance_px: int = _EXTENT_TOLERANCE_PX,
                     anchor_residual_px: int = _ENTRY_ANCHOR_RESIDUAL_PX,
                     trust_window_px: int | None = None,
                     **plan_kwargs) -> ItemTarget:
    """Walk UP from where the profile read left the card to the model's item `model_index`, and
    report where its heart is. Never taps, and never substitutes a different item.

    `entry_reference` is REQUIRED and is the LAST frame the index was built from — the frame
    `index.offsets[-1]` was measured on. One `estimate_shift` against it puts the screen into the
    index's own page space, which is bottom-up's entire zero point; there is no default, because a
    caller who forgot it could otherwise be given a weaker anchor silently. `hinge` keeps it
    beside the index and clears the two together (`_invalidate_item_index`), so "the index is
    stale" and "the anchor is stale" cannot become different facts.

    `identity_match_max_dist` is also REQUIRED. It is the measured, device-bound identity
    acceptance ceiling: finite and positive, and strictly below this module's known 2.565
    different-profile collision. This leaf returns a target a caller can tap, so it refuses a
    missing, generic, or unsafe bound before taking its first screencap.

    `index` must be the index of the profile currently on screen, built with `at_scroll_top=True`
    from a confirmed `ScrollTopVerdict`. An index built with `at_scroll_top=False` is refused
    outright: its ordinals are relative to whatever was in view, so counting against it is a
    systematic off-by-N with nothing on the result to say so. NOTE that this function does not
    itself confirm a top any more and does not need to — the assertion belongs to the enumeration
    pass that built the index, which is the pass whose origin the ordinals are counted from.

    "Of the profile currently on screen" is CHECKED here before any COUNTING happens: the index
    carries a fingerprint of the sticky per-profile header it was built from, and a screen that
    does not match it is `NAV_IDENTITY_MISMATCH` / `NAV_IDENTITY_UNCONFIRMED`. Enter this from
    where the enumeration read left the card, scrolled, so the sticky header is already showing;
    see the module docstring.

    A screen that instead arrives at a card's SCROLL TOP is no longer an automatic refusal
    (2026-08-22, blocker 12): that entry requirement — identity must be visible before a single
    heart is counted — has not changed, but a scroll top is now a RECOVERABLE entry state rather
    than a terminal one, because the strip there is Hinge's own profile-independent filter chips
    and ONE bounded, guarded read-scroll is exactly what reveals the sticky header underneath.
    That step is taken through the driver's own humanized read-scroll primitive and ledger, sized
    to the SMALLEST legal read-scroll rather than production's own read cadence (the header
    appears the moment the card is scrolled at all, and the smallest step is the one guaranteed to
    stay inside the shift estimator's own trust window), MEASURED with that same estimator every
    other leg in this module uses, and folded into the entry offset before anything is refused;
    identity is then re-checked on the frame it produced. Only if identity still cannot be confirmed afterwards does the original "cannot
    tell" refusal stand, with a note that the step was already tried. Exactly ONE such step is
    ever taken, and a screen that was already scrolled on entry — the ordinary case — costs no
    extra frame or gesture at all. This is what makes a SHORT profile (a photo plus a video, with
    nothing below the fold for the read to have scrolled past) reachable: its read ends at the
    top by construction, and before this fix the entry gate refused it outright regardless of
    which item the model chose.

    A second, equally bounded entry recovery handles one distinct geometry: the target heart is
    already visible, but the frame cannot establish its card's LOWER edge (whether it runs into
    the band or ends in an untrusted blank-background run). An ascending gesture would move that
    uncertainty farther down and turn a recoverable landing into an inevitable refusal. Before
    this pass has counted or issued an ascending gesture, it may therefore spend ONE planned
    forward read-scroll only when the current frame itself proves that lower-edge shape and the
    index says the entire card fits in the band after the planned displacement. Its delivered
    positive shift and the newly complete target block are both measured; otherwise the run stops.
    This is positioning before a count, never a reversal of one.

    `model_index` is 1-based into THIS index's selectable blocks. It is NOT a number from an
    `ItemPayload` whose exclusions renumbered the model's list — when anything selectable was
    excluded the two differ and `ItemPayload.translation` is authoritative, so convert first:

        ordinal = payload.item(n).heart_ordinal
        model_index = index.translation.index(ordinal) + 1

    ``selected_model_item_index`` is optional audit provenance for that conversion.  It does
    not affect navigation: callers that began with the model's dense payload number pass it so
    the returned human-readable reason can distinguish (for example) "model item 3" from the
    full-index position used to reach its page heart (for example, item 9).  Direct index-only
    callers leave it unset.

    Returns an `ItemTarget` whose `point` is a screen coordinate in its `frame`. The card is on
    screen and fully visible at that moment, and the phone is left exactly there. `steps` may be
    EMPTY and that is the ordinary case for an item near the bottom of the profile: the read
    already left the card showing it, so the navigation is one screencap and one shift estimate.

    Raises `ItemNavigationError` for every refusal it decides itself; its `code` is one of the
    `NAV_*` constants and enumerates them. `ScrollStepError` (a card spacing no permitted gesture
    can enumerate), `SegmentationError` and `ShiftEstimationError` (the vision layer could not
    look at all) propagate from the modules that own those decisions, unchanged.

    `max_frames`, the tolerances, `trust_window_px` and any further `plan_kwargs` forwarded to
    `plan_scroll_step` are calibration parameters with derived or measured defaults, exposed on
    `segment_frame`'s precedent so a validation pass can vary one without editing the module —
    not because a production caller should.
    """
    # This helper ultimately returns a heart for its caller to tap.  It cannot safely retain a
    # generic identity default: 3.0 admits the measured 2.565 different-profile collision.
    # Hinge supplies its device-bound calibration, while offline callers must make their own
    # measured bound explicit.  Validate before even a screencap so a malformed direct call has
    # no device side effects.
    if (isinstance(identity_match_max_dist, bool)
            or not isinstance(identity_match_max_dist, (int, float))
            or not math.isfinite(identity_match_max_dist)
            or identity_match_max_dist <= 0
            or identity_match_max_dist >= _IDENTITY_FALSE_MATCH_DISTANCE):
        raise ItemNavigationError(
            NAV_CALIBRATION_INVALID,
            "refusing to navigate without a finite, positive identity_match_max_dist strictly "
            f"below the known {_IDENTITY_FALSE_MATCH_DISTANCE} different-profile false-match "
            "distance; a generic or "
            "unsafe bound could turn a foreign card into a target")

    def cancelled(*, frame: bytes | None = None, frame_index: int | None = None,
                  anchor: ShiftEstimate | None = None) -> None:
        """Stop before a capture or gesture, leaving the card untouched from this point."""
        if should_stop is not None and should_stop():
            raise ActionCancelled(
                "navigation cancelled because the run is stopping; no further screencap or "
                "scroll gesture was issued, and nothing was tapped")

    # The bound is the SUM of the two chains' slack (see `_CROSSCHECK_TOLERANCE_PX`), and only
    # one of those two is this module's own. The other belongs to the index in hand, which the
    # frame-omission recovery deliberately builds at a wider tolerance than the default — so
    # reading it off the index is what keeps the derivation true for a recovered index instead of
    # leaving it one pixel short and hard-stopping a navigation that agreed all along. An
    # explicit argument still wins, for the calibration passes that vary one bound at a time.
    if crosscheck_tolerance_px is None:
        crosscheck_tolerance_px = (
            getattr(index, "extent_tolerance_px", _EXTENT_TOLERANCE_PX) + _EXTENT_TOLERANCE_PX)

    # --- the index, before a finger moves ------------------------------------------
    if not index.usable:
        raise ItemNavigationError(
            NAV_INDEX_UNUSABLE,
            "refusing to navigate against an item index that contradicts itself: "
            + "; ".join(index.failures)
            + ". Doc 5.3: treat a missing table as a hard stop, never as a reason to fall back "
              "to a fixed coordinate")
    if not index.at_scroll_top:
        raise ItemNavigationError(
            NAV_INDEX_RELATIVE,
            "refusing to count against an item index built with at_scroll_top=False. Its heart "
            "ordinals are RELATIVE to whatever was in view when it was captured — a measured "
            "mid-profile window returns translation (2, 3, 4, 5, 6) for what are really items "
            "5..9 — so counting hearts from a real scroll top towards one of them lands a fixed "
            "number of cards away from the item the model chose. Re-index from a confirmed top "
            "(ops/OPENER-REDESIGN.md 5.3, addendum 2026-08-12)")
    try:
        target_block = index.block_for(model_index)
        ordinal = index.heart_ordinal_for(model_index)
    except ItemIndexError as exc:
        raise ItemNavigationError(NAV_ORDINAL_OUT_OF_RANGE, str(exc)) from exc

    hearts = index_hearts(index)
    if sorted(hearts) != list(range(1, index.heart_count + 1)):
        raise ItemNavigationError(
            NAV_INDEX_UNUSABLE,
            f"the index's heart ordinals are {sorted(hearts)}, which is not a gapless 1.."
            f"{index.heart_count}; a count cannot be checked against a table that skips a "
            "number")
    target_heart = hearts[ordinal]
    index_selection = f"full-index selectable item {model_index}"
    if selected_model_item_index is not None:
        index_selection += f", selected by model item {selected_model_item_index}"

    if not index.offsets or index.offsets[-1] is None:
        # Unreachable through `usable` (a broken correspondence chain is a failure and empties
        # `blocks`), and checked anyway because the entry anchor is built ON this number: an
        # anchor measured against a frame whose own page offset is unknown would be an
        # arithmetic error dressed as a measurement.
        raise ItemNavigationError(
            NAV_INDEX_UNUSABLE,
            "the index's last frame has no page offset, so there is no page row for the entry "
            "anchor to be measured against")
    reference_offset = int(index.offsets[-1])

    like_template = driver._template("like")
    like_threshold = _like_threshold()
    content_band = driver.content_band

    # --- the identity gate: whose card is this? -------------------------------------
    # LAST of the pre-flight checks and FIRST of the ones that look at the device, which is the
    # order both halves want. Everything above is a pure inspection of the index — free, and
    # unaffected by what is on screen — so an unusable or relative index says so instead of
    # spending an ADB round-trip to be told the identity does not match either. Everything below
    # moves the phone. Nothing between the two.
    #
    # This is the check doc 5.7's final audit carried forward, and it is the only thing standing
    # between a stale or foreign translation table and a confident tap at ordinal 1: measured,
    # profile A's index over profile B's frames returned a tap target because every geometric
    # comparison this module makes passed identically on two stereotyped cards. So the question
    # asked here is not geometric.
    cancelled()
    entry_frame = driver._screencap()
    try:
        identity = compare_profile_identity(
            entry_frame, index.identity, identity_band=driver.identity_band,
            match_max_dist=identity_match_max_dist)
    except IdentityError as exc:
        # "Could not look" is kept distinct in the message and collapsed into one code, because
        # to a caller about to tap a heart it calls for exactly the action "cannot tell" does.
        raise ItemNavigationError(
            NAV_IDENTITY_UNCONFIRMED,
            f"refusing to navigate: the identity of the profile on screen could not be read at "
            f"all ({type(exc).__name__}: {exc}). An index may only be navigated against the "
            "profile it was built from, and that cannot be established here "
            "(ops/OPENER-REDESIGN.md 5.7, carried-forward requirement 1)",
            frame=entry_frame) from exc

    # Bottom-up's zero point is measured against the frame captured ABOVE, before anything below
    # gets a chance to move the phone — so `entry_step_delta`/`identity_recovered_frame` default
    # to "no recovery happened" and only change inside the branch that actually earns them.
    entry_step_delta = 0
    identity_recovered_frame: bytes | None = None
    if not identity.matched:
        # BLOCKER 12 (2026-08-22): a screen at a card's SCROLL TOP is a RECOVERABLE entry state,
        # not a terminal one — see THE FIFTH THING in the module docstring for why. Every OTHER
        # cause of "not matched" (a foreign header, a moved band, an index with no fingerprint) is
        # left alone: scrolling cannot help any of those, and re-asking the same scroll-top check
        # `compare_profile_identity` already ran internally is what tells the two apart without
        # parsing prose out of `identity.reason`.
        recovery_note = ""
        if identity.unknown:
            try:
                still_at_top = confirm_scroll_top(
                    entry_frame, identity_band=driver.identity_band,
                    pinned_evidence=_driver_pinned_evidence(driver))
            except ScrollTopError:
                still_at_top = None
            if still_at_top is not None and still_at_top.confirmed:
                # ONE bounded, guarded forward read-scroll — the same humanized primitive and
                # ledger every other read-scroll on this driver goes through, never a raw
                # transport call and never a second attempt. Bounded to the SMALLEST legal
                # read-scroll rather than production's own read cadence, on the ascending walk's
                # own precedent of discarding the policy's frac: there is no card spacing to plan
                # an aliasing-safe distance against yet (nothing below the fold has been
                # segmented), Hinge's sticky header appears "the moment the card is scrolled at
                # all" regardless of how far, and the smallest legal step is the one guaranteed to
                # stay inside `estimate_shift`'s own trust window rather than risk landing beyond
                # it -- a full read-scroll can (measured: a 0.55 read_scroll_frac step moves
                # ~1299px on the calibrated 2400px screen, past the 900px window this estimate is
                # trusted within). `_sample_read_step` still owns the dwell and the LANE.
                cancelled(frame=entry_frame, frame_index=0)
                dwell, _discarded_policy_frac, x_frac = driver._sample_read_step(0, None)
                frac = _frac_window()[0]
                driver._scroll_down_one(frac, x_frac)
                if dwell and dwell > 0:
                    time.sleep(dwell)
                cancelled(frame=entry_frame, frame_index=0)
                recovered_frame = driver._screencap()
                # MEASURED like every other leg in this module: the displacement this one gesture
                # produced, never assumed. `entry_frame` (before the step) and `recovered_frame`
                # (after it) are exactly the shift estimator's ordinary pair.
                entry_step = estimate_shift(entry_frame, recovered_frame,
                                            content_band=content_band,
                                            trust_window_px=trust_window_px)
                if entry_step.delta_px is None:
                    raise ItemNavigationError(
                        NAV_ENTRY_STEP_UNMEASURED,
                        "refusing to navigate: one bounded read-scroll was taken to leave the "
                        "card's scroll top and reveal the sticky per-profile header, but the "
                        f"shift it produced could not be measured ({entry_step.status} — "
                        f"{entry_step.reason}). Guessing the displacement here is exactly how "
                        "doc 5.10's phantom item was fabricated, so this stops rather than "
                        "assumes a distance (2026-08-22)",
                        frame=recovered_frame, frame_index=0)
                try:
                    retried_identity = compare_profile_identity(
                        recovered_frame, index.identity, identity_band=driver.identity_band,
                        match_max_dist=identity_match_max_dist)
                except IdentityError:
                    retried_identity = None
                if retried_identity is not None and retried_identity.matched:
                    identity = retried_identity
                    entry_step_delta = entry_step.delta_px
                    identity_recovered_frame = recovered_frame
                else:
                    recovery_note = (
                        " A bounded entry read-scroll was already taken to try to reveal the "
                        "sticky header (2026-08-22), and identity still could not be confirmed "
                        "on the frame it produced, so this refusal stands")
        if not identity.matched:
            raise ItemNavigationError(
                NAV_IDENTITY_MISMATCH if identity.mismatched else NAV_IDENTITY_UNCONFIRMED,
                f"refusing to navigate an item index against a profile it may not describe: "
                f"{identity.reason}. Doc 5.6's owner rule is that we never substitute a different "
                "item, and counting hearts on somebody else's card would do exactly that with full "
                "confidence — every geometric check this module makes passed identically on the one "
                "measured cross-profile pair, because Hinge's cards are stereotyped. Re-read the "
                "profile rather than navigating this index" + recovery_note,
                frame=entry_frame)

    # --- the entry anchor: put this screen into the INDEX's page space ---------------
    # BOTTOM-UP'S ZERO POINT, and the whole of what replaced the rewind. `entry_reference` is the
    # last frame the index was built from, so `index.offsets[-1]` is ITS page offset; one shift
    # estimate against the screen in front of us therefore gives the screen's offset in the same
    # space, and every page row below is directly comparable with the index's without a second
    # origin to reconcile.
    #
    # It is exactly 0px whenever nothing moved the card between the read and the like, which is
    # what `hinge._capture_current` produces by construction: both of its terminating paths leave
    # the screen showing its own last KEPT frame. A SMALL non-zero measurement is honoured rather
    # than refused — it is a measurement, and the reverse heart count checks it against a real
    # heart on the first frame that shows one — while a drift at or beyond one gesture floor is
    # refused a few lines below (see THE ENTRY DRIFT BOUND in the loop, which needs the screen
    # height and therefore has to wait for the first segmentation).
    #
    # A shift the estimator cannot measure AT ALL is a hard stop here, because the alternative is
    # to assume a distance, which is precisely how doc 5.10's phantom item was fabricated.
    # `estimate_shift`'s own trust window (900px on the calibrated band) bounds a screen that has
    # been moved wholesale, and on real frames it is also what refuses a foreign profile: two
    # different people's cards do not correlate.
    anchor = estimate_shift(entry_reference, entry_frame, content_band=content_band,
                            trust_window_px=trust_window_px)
    if anchor.delta_px is None:
        raise ItemNavigationError(
            NAV_ANCHOR_UNMEASURED,
            f"refusing to navigate: the screen could not be put in the item index's page space "
            f"({anchor.status} — {anchor.reason}). Bottom-up navigation counts up from where the "
            "profile read left the card, so its zero point is one measured shift against the "
            "read's own last frame; without it there is no page row to count from and no "
            "substitute is invented (ops/OPENER-REDESIGN.md 5.5)",
            frame=entry_frame, frame_index=0, anchor=anchor)
    # `entry_step_delta` is 0 unless the scroll-top recovery branch above actually ran and
    # identity was confirmed on the frame it produced (blocker 12, 2026-08-22) — in which case it
    # is that step's OWN separately measured displacement, folded on top of the ordinary anchor
    # rather than through it, so the anchor above keeps meaning exactly what it always meant: the
    # shift between `entry_reference` and the frame captured BEFORE this function moved anything.
    entry_offset = reference_offset + anchor.delta_px + entry_step_delta

    # The one piece of loop state `scroll_step` documents, seeded from the enumeration pass
    # rather than left at None: the index already measured every card on this page, so the very
    # first gesture is sized against the profile's true smallest heart pitch instead of
    # discovering it a card at a time. Strictly smaller steps than the unseeded loop would take,
    # never larger.
    seen_spacing = _min_heart_pitch(hearts)

    # The counting walk starts from wherever the phone PHYSICALLY is right now: `entry_frame` in
    # the ordinary case, or the frame the recovery step produced when that step is what got
    # identity confirmed. `offsets[0]` (below) is `entry_offset`, which already folds in the same
    # step's measured displacement, so the two stay in the same page space either way.
    frames: list[bytes] = [identity_recovered_frame
                           if identity_recovered_frame is not None else entry_frame]
    segmentations: list[FrameSegmentation] = []
    offsets: list[int] = [entry_offset]
    steps: list[ScrollStep] = []
    shifts: list[ShiftEstimate] = []
    budget: int | None = max_frames
    agreement = 0
    heart_was_visible = False
    entry_position_attempted = False

    while True:
        i = len(segmentations)
        seg = segment_frame(frames[i], content_band=content_band, like_template=like_template,
                            like_threshold=like_threshold)
        if not seg.ok:
            raise ItemNavigationError(
                NAV_FRAME_CONTRADICTS,
                f"frame {i} of this navigation contradicts itself and cannot be counted from: "
                + "; ".join(seg.failures),
                frame=frames[i], frame_index=i, anchor=anchor)
        segmentations.append(seg)
        band_y0, band_y1 = seg.band
        if i == 0:
            # THE ENTRY DRIFT BOUND, checked here rather than beside the anchor itself only
            # because this is the first place the screen's own height is known.
            #
            # The expected drift is EXACTLY 0px: `entry_reference` is the read's last kept frame
            # and, in both of `hinge._capture_current`'s terminating paths, that frame is what is
            # still on screen. The bound is therefore not noise slack — it is "did anything MOVE
            # the card". The smallest read-scroll this driver is allowed to make is the smallest
            # displacement any gesture of ours can produce, so a drift at or beyond it cannot
            # have come from rendering, an animated card or a settle: something scrolled the
            # profile between the read and the like (a finger, or a gesture nobody recorded), and
            # a page position nobody accounted for is not one a tap can be justified from.
            #
            # WHAT THIS IS NOT. It is not a cross-profile check and must not be mistaken for one.
            # A foreign profile does not fail here — it fails at the identity gate above, and on
            # real frames it fails at the anchor itself, because `estimate_shift` requires the
            # screen to CORRELATE with the read's last frame and two different people's cards do
            # not. See the module docstring's note on what the rewind's geometric anchor used to
            # catch and what replaced it.
            drift_bound = step_px_for_frac(_frac_window()[0], seg.frame_size[1])
            if abs(anchor.delta_px) >= drift_bound > 0:
                raise ItemNavigationError(
                    NAV_ANCHOR_UNMEASURED,
                    f"refusing to navigate: the screen has moved {anchor.delta_px:+d}px since "
                    f"the profile read's last frame, which is at or beyond the {drift_bound}px "
                    "smallest gesture this driver can make. The read leaves the card exactly "
                    "where its last kept frame was, so a drift this large means something else "
                    "scrolled the profile between the read and the like, and where the count "
                    "would be starting from is no longer accounted for",
                    frame=frames[i], frame_index=i, anchor=anchor)
            budget = _frame_budget(entry_offset, target_block.page_y0, band_y0,
                                   seg.frame_size[1]) if budget is None else budget

        # A target already visible in a lower-edge-*unresolved* block is the one entry geometry
        # where an ascending gesture is provably the wrong direction. The lower edge can run into
        # the band OR be an untrusted background run inside it: the latter is the live failure --
        # the index projected the whole card into the band, but segmentation could not prove that
        # its blank lower tail had ended. A trustworthy TOP edge distinguishes this from a card
        # merely entering through the band's top. This branch runs before this frame contributes
        # observations or clusters, and before an upward gesture exists. It therefore seats a card
        # for one clean count rather than reversing a count already spent.
        if i == 0 and not entry_position_attempted and not steps:
            indexed_frame_y = target_heart[1] - offsets[0]
            target_top = target_block.page_y0 - offsets[0]
            target_bottom = target_block.page_y1 - offsets[0]
            found_entry = (_heart_in_frame(seg, indexed_frame_y, extent_tolerance_px)
                           if band_y0 <= indexed_frame_y < band_y1 else None)
            entry_block = None if found_entry is None else found_entry[1]
            lower_edge_unresolved = (
                entry_block is not None
                and not entry_block.complete
                and entry_block.top.observed
                and not entry_block.bottom.observed
                and target_block.page_y1 - target_block.page_y0 <= band_y1 - band_y0)
            if lower_edge_unresolved:
                # The scroll-top identity recovery above already spent this function's one
                # sanctioned forward entry gesture. Do not turn a pair of independently bounded
                # recoveries into a two-step forward walk: the latter has neither a clean entry
                # contract nor the FakeDriver/driver ledger guarantee the former two have.
                if identity_recovered_frame is not None:
                    raise ItemNavigationError(
                        NAV_ENTRY_POSITION_UNRESOLVED,
                        "refusing to navigate: the target's lower edge remains unresolved after "
                        "the one bounded forward entry scroll already used to reveal profile "
                        "identity. A second forward positioning gesture is not permitted before "
                        "the ascending count, and walking up would make this lower-edge evidence "
                        "worse",
                        frame=frames[0], frame_index=0, anchor=anchor)
                dwell, _discarded_policy_frac, x_frac = driver._sample_read_step(0, None)
                position_step = plan_scroll_step(
                    seg, x_frac=x_frac, profile_min_spacing_px=seen_spacing, **plan_kwargs)
                # The planned (not merely hoped-for) displacement must leave every indexed card
                # edge inside the band. A gesture too short or too long is not a reason to probe
                # in a second direction.
                if not (target_top - position_step.step_px >= band_y0
                        and target_bottom - position_step.step_px <= band_y1):
                    raise ItemNavigationError(
                        NAV_ITEM_NOT_FULLY_VISIBLE,
                        f"heart {ordinal} ({index_selection}) is visible in a card with an "
                        "unresolved lower edge, but even one sanctioned forward positioning "
                        "step would leave an edge outside the band. Walking up would make the same "
                        "edge worse, so no gesture can safely seat this item",
                        frame=frames[0], frame_index=0, anchor=anchor)
                else:
                    entry_position_attempted = True
                    cancelled(frame=frames[0], frame_index=0, anchor=anchor)
                    driver._scroll_down_one(position_step.frac, position_step.x_frac)
                    if dwell and dwell > 0:
                        time.sleep(dwell)
                    cancelled(frame=frames[0], frame_index=0, anchor=anchor)
                    positioned_frame = driver._screencap()
                    positioned_shift = estimate_shift(
                        frames[0], positioned_frame, content_band=content_band,
                        trust_window_px=trust_window_px)
                    if positioned_shift.delta_px is None:
                        raise ItemNavigationError(
                            NAV_ENTRY_POSITION_UNRESOLVED,
                            "refusing to navigate: one bounded forward entry-positioning scroll was "
                            "taken to seat the lower-edge-unresolved target card, but its shift could "
                            f"not be measured ({positioned_shift.status} — {positioned_shift.reason}). "
                            "No count had started, but guessing the new page row could tap a "
                            "different item",
                            frame=positioned_frame, frame_index=0, anchor=anchor)
                    violation = step_overshoot(position_step, positioned_shift.delta_px)
                    if violation is not None:
                        raise ItemNavigationError(
                            NAV_ENTRY_POSITION_UNRESOLVED,
                            "refusing to navigate: the one bounded forward entry-positioning scroll "
                            f"did not respect its plan ({violation}). The target is not pursued by "
                            "a second forward or reverse gesture",
                            frame=positioned_frame, frame_index=0, anchor=anchor)
                    positioned_offset = offsets[0] + positioned_shift.delta_px
                    positioned_seg = segment_frame(
                        positioned_frame, content_band=content_band, like_template=like_template,
                        like_threshold=like_threshold)
                    positioned_y = target_heart[1] - positioned_offset
                    positioned_found = (
                        _heart_in_frame(positioned_seg, positioned_y, extent_tolerance_px)
                        if positioned_seg.ok and band_y0 <= positioned_y < band_y1 else None)
                    if positioned_found is None or not positioned_found[1].complete:
                        raise ItemNavigationError(
                            NAV_ENTRY_POSITION_UNRESOLVED,
                            "refusing to navigate: the one bounded forward entry-positioning scroll "
                            f"moved page heart {ordinal} ({index_selection}), but its card still was "
                            "not bounded end to end. No count has been reused and no second gesture "
                            "is justified",
                            frame=positioned_frame, frame_index=0, anchor=anchor)
                    # Restart the normal ascending pass at this measured origin. Nothing from the
                    # clipped frame was folded into a count, and this positioning gesture is not
                    # one of the returned reverse-walk `steps`.
                    entry_offset = positioned_offset
                    frames[:] = [positioned_frame]
                    offsets[:] = [positioned_offset]
                    segmentations.clear()
                    budget = max_frames
                    continue

        observations = _observations(segmentations, offsets)
        clusters = _heart_clusters(observations, tolerance=extent_tolerance_px)

        missed = _no_heart_was_missed(clusters, observations, offsets, seg.band,
                                      extent_tolerance_px, ascending=True)
        if missed is not None:
            raise ItemNavigationError(NAV_HEART_MISSED, missed, frame=frames[i], frame_index=i,
                                      anchor=anchor)

        measured_anchor_residual = anchor_residual_px
        if anchor.delta_px is not None:
            measured_anchor_residual = max(measured_anchor_residual, abs(int(anchor.delta_px)))

        disagreement, worst, counted = _count_disagrees(
            clusters, hearts, heart_count=index.heart_count,
            tolerance=crosscheck_tolerance_px, ascending=True,
            anchor_residual_px=measured_anchor_residual)
        agreement = max(agreement, worst)
        if disagreement is not None:
            raise ItemNavigationError(NAV_COUNT_DISAGREES, disagreement, frame=frames[i],
                                      frame_index=i, anchor=anchor)

        # --- have we gone PAST it? --------------------------------------------------
        # Walking UP, content moves DOWN the screen: the target enters the band at its TOP edge
        # and would eventually leave at its BOTTOM. This is the mirror of the descending pass's
        # "scrolled off the top" test, and it is asked of the INDEX's row rather than of the
        # count because a heart below the band has not been counted at all — there is nothing to
        # ask the count about. Once `_count_disagrees` has passed, the two rows agree to
        # `crosscheck_tolerance_px`, so quoting either in the message is honest.
        indexed_frame_y = target_heart[1] - offsets[i]
        if indexed_frame_y >= band_y1:
            # Walking up only takes it further down, so there is nothing this pass can do about
            # it and nothing it may substitute. On the FIRST frame this means the read did not
            # end where the index says it ended — the anchor and the index disagree about where
            # the bottom of this profile is — which is a different diagnosis from a gesture that
            # over-delivered later, so the two get different codes.
            raise ItemNavigationError(
                NAV_ITEM_BELOW_ENTRY if not steps else NAV_ITEM_NOT_FULLY_VISIBLE,
                f"heart {ordinal} ({index_selection}) is BELOW the analysed band — the "
                f"index puts it at page row {target_heart[1]}, i.e. frame row {indexed_frame_y} "
                f"against a band ending at {band_y1} — after {len(steps)} upward gesture(s). "
                "Walking up moves it further down, and scrolling back down would re-run a count "
                "that has already been spent, so there is no safe way to reach it from here",
                frame=frames[i], frame_index=i, anchor=anchor)

        # --- have we arrived? -------------------------------------------------------
        # From here on the COUNT is the authority, not the index: `counted[ordinal]` is this
        # pass's own folded page row for that heart, and it exists only once the count has
        # actually reached the ordinal. That is the ascending mirror of the descending pass's
        # `len(clusters) >= ordinal` gate — same rule ("do not land on an ordinal the count has
        # not reached"), expressed against a map because ascending ordinals are assigned from a
        # bottom anchor rather than implied by position.
        counted_heart = counted.get(ordinal)
        if counted_heart is not None:
            frame_y = counted_heart[1] - offsets[i]
            if band_y0 <= frame_y < band_y1:
                heart_was_visible = True
                found = _heart_in_frame(seg, frame_y, extent_tolerance_px)
            else:
                found = None
            if found is not None and found[1].complete:
                heart, block = found
                landed = _confirm_landing(
                    heart=heart, block=block, index_block=target_block,
                    index_heart=target_heart, extent_tolerance_px=extent_tolerance_px,
                    crosscheck_tolerance_px=crosscheck_tolerance_px)
                if landed is not None:
                    raise ItemNavigationError(NAV_COUNT_DISAGREES, landed, frame=frames[i],
                                              frame_index=i, anchor=anchor)
                climbed = entry_offset - offsets[i]
                return ItemTarget(
                    point=(int(heart[0]), int(heart[1])), model_index=model_index,
                    heart_ordinal=ordinal, frame=frames[i], frame_index=i,
                    block_frame_rows=(block.y0, block.y1),
                    block_page_rows=(block.y0 + offsets[i], block.y1 + offsets[i]),
                    page_offset=offsets[i], hearts_counted=len(clusters),
                    agreement_px=agreement, identity=identity,
                    anchor=anchor, entry_offset=entry_offset,
                    steps=tuple(steps), shifts=tuple(shifts), offsets=tuple(offsets),
                    reason=(f"walked {climbed}px up from where the profile read ended in "
                            f"{len(steps)} gesture(s), counting {len(clusters)} heart(s) back in "
                            f"reverse; page heart {ordinal} is {index_selection}; its card is "
                            f"bounded end to end at frame rows {block.y0}..{block.y1}, and the "
                            f"count agrees with the index to {agreement}px"))

        # --- not yet: plan one more step --------------------------------------------
        # The budget is counted in FRAMES, which is what `_frame_budget` derives and what
        # `max_frames` names; this frame has already been segmented and counted, so reaching the
        # budget here means the next capture would be over it.
        if len(segmentations) >= budget:
            if heart_was_visible:
                raise ItemNavigationError(
                    NAV_ITEM_NOT_FULLY_VISIBLE,
                    f"heart {ordinal} ({index_selection}) has been on screen but its "
                    f"card was never bounded end to end in any of {len(frames)} frames, so "
                    "there is no frame in which the item is fully visible to tap from",
                    frame=frames[i], frame_index=i, anchor=anchor)
            raise ItemNavigationError(
                NAV_BUDGET_EXHAUSTED,
                f"counted {len(clusters)} heart(s) back in {len(frames)} frames and never "
                f"brought heart {ordinal} ({index_selection}) into view, against a budget "
                f"of {budget} frames derived from the index's own geometry and the smallest "
                "gesture this driver may make. Something other than slow progress is wrong",
                frame=frames[i], frame_index=i, anchor=anchor)

        # The behaviour policy owns the dwell and the LANE; the closed loop owns the DISTANCE.
        # The policy's own frac is discarded on purpose — it is production's read cadence, which
        # doc 5.10.1 measured aliasing against the card spacing, and it is the one quantity this
        # whole pass exists to replace. Aliasing is a property of step-against-spacing and does
        # not care which way the finger went, so the ratio rule is applied to the reverse gesture
        # exactly as the enumeration read applies it to the forward one.
        dwell, _discarded_policy_frac, x_frac = driver._sample_read_step(len(steps), None)
        step = plan_scroll_step(seg, x_frac=x_frac, profile_min_spacing_px=seen_spacing,
                                **plan_kwargs)
        if step.spacing_px is not None:
            seen_spacing = (step.spacing_px if seen_spacing is None
                            else min(seen_spacing, step.spacing_px))
        # THE ONLY GESTURE. `_scroll_up_one` requires both arguments, which is what makes the
        # `_scroll_down_one(frac)` trap — re-sampling BOTH from the behaviour policy when either
        # is None, and so silently issuing production's cadence — impossible to reach from here.
        cancelled(frame=frames[i], frame_index=i, anchor=anchor)
        driver._scroll_up_one(step.frac, step.x_frac)
        steps.append(step)
        if dwell and dwell > 0:
            time.sleep(dwell)

        cancelled(frame=frames[i], frame_index=i, anchor=anchor)
        nxt = driver._screencap()
        est = estimate_shift(frames[i], nxt, content_band=content_band,
                             trust_window_px=trust_window_px)
        shifts.append(est)
        if est.delta_px is None:
            raise ItemNavigationError(
                NAV_CHAIN_BROKEN,
                f"frames {i} and {i + 1} of this navigation could not be put in one coordinate "
                f"space: {est.status} — {est.reason}. Without the shift there is no way to tell "
                "a heart that moved from a heart that arrived, which is exactly how doc 5.10's "
                "phantom item was fabricated, so the count stops here",
                frame=nxt, frame_index=i + 1, anchor=anchor)
        # THE SIGN FLIP, and it is one line. `estimate_shift` is positive when content moves UP,
        # i.e. for a forward scroll; an upward gesture must therefore measure NEGATIVE, and how
        # far it climbed is the negation. `step_overshoot` is handed that magnitude, so its own
        # two tests keep their meanings exactly: `<= 0` is "the profile did not move" — which now
        # also covers "it moved the WRONG WAY", the reverse loop's specific stall, including the
        # legitimate one where the card has reached its true top and no further upward gesture
        # can do anything — and `> bound_px` is the aliasing ceiling.
        climb_px = -est.delta_px
        violation = step_overshoot(step, climb_px)
        if violation is not None:
            # The count must stop -- this step is too large to safely use for ordinal
            # navigation -- but unlike an unmeasurable chain its terminal page position is known.
            # Carry that strictly-cleanup-only fact to the optional still-photo walk so it can
            # return to its entry anchor instead of converting a rejected candidate into a
            # poisoned capture.  `offsets` does not yet contain this frame because the count must
            # not consume it; spelling the prospective offset here keeps the recovery arithmetic
            # tied to the same measured chain the normal recurrence would have used.
            recovery = NavigationRecovery(
                frame=nxt,
                page_shift_px=(offsets[-1] + int(est.delta_px)) - entry_offset,
                step_index=i,
                frame_index=i + 1,
                planned_step_px=step.step_px,
                achieved_step_px=climb_px,
                bound_px=step.bound_px,
                frac=step.frac,
                window_px=step.window_px,
                basis=step.basis,
                spacing_px=step.spacing_px,
                sized_against_px=step.sized_against_px,
                measurement_delta_px=int(est.delta_px),
                measurement_status=est.status,
                measurement_confidence=est.confidence,
                measurement_agreeing=est.agreeing,
                measurement_dissenting=est.dissenting,
                measurement_eligible=est.eligible,
                violation=violation)
            raise ItemNavigationError(
                NAV_SCROLL_STALLED if climb_px <= 0 else NAV_SCROLL_OVERSHOT,
                f"the upward gesture at frame {i} did not respect the step it was planned as: "
                f"{violation} (measured {est.delta_px:+d}px, i.e. {climb_px}px of climb). A "
                "stalled loop would re-count the same frame forever and an over-large one "
                "aliases the count against the card spacing; neither is a step to take again",
                frame=nxt, frame_index=i + 1, anchor=anchor, recovery=recovery)
        # UNCHANGED, deliberately: the page-space recurrence needs no flip because the sign is in
        # the measurement. An upward gesture produces a negative delta, so the offset decreases.
        offsets.append(offsets[-1] + est.delta_px)
        frames.append(nxt)


def _confirm_landing(*, heart: tuple[int, int], block: Block, index_block, index_heart,
                     extent_tolerance_px: int, crosscheck_tolerance_px: int) -> str | None:
    """Why the card under the counted heart is not the card the index calls this item, or None.

    The ordinal cross-check answers "is this the k-th heart"; this answers "is the k-th heart on a
    card LAID OUT like the one the index recorded as item N". They are different questions, and
    all three comparisons here are origin-free — a height, an offset within a block, and an x — so
    none of them depends on the two passes sharing a page origin, which they do not.

    WHAT IT DOES NOT CATCH, corrected 2026-08-12 after a validation pass measured the claim an
    earlier revision of this docstring made. It said this check "is the one that catches an index
    built against a DIFFERENT profile — a stale translation table surviving a deck advance". That
    is FALSE, and it was measured false on the only real cross-profile pair in the calibration
    corpus: driving profile A's index over profile B's frames, item 1's card came back 974px tall
    with its heart 885px down and at x=938 on BOTH profiles, so every comparison below passed
    identically. Hinge's cards are stereotyped — the modal card is one photo of one height with
    the heart at a fixed inset — so "the wrong card under the right ordinal" is not what an
    ordinary foreign profile looks like. What this does catch is a card whose GEOMETRY differs,
    which is the miscount case (the right ordinal on a genuinely different-sized card).

    For a foreign or stale index the guards are, in order: the IDENTITY GATE at the top of
    `navigate_to_item`, which is the one that actually answers this question and refuses before a
    finger moves; then `_count_disagrees`'s ordinal comparison, which caught 8 of the 9 items on
    that measurement; then its ANCHOR test, which catches a profile whose item 1 sits
    `_TOP_ORIGIN_RESIDUAL_PX + tolerance` or more from the index's (the measured pair is exactly
    218px apart against exactly that bound, which is why that test is `>=`). Doc 5.3's own
    mitigation — correct INVALIDATION of the table wherever `_current_sigs` is cleared — and doc
    5.6's post-tap crop-signature check remain in force behind all three: the gate rules out the
    wrong PERSON before the tap, the signature check rules out the wrong ITEM after it, and
    neither replaces the other. What must not be written is code that treats THIS function as
    having ruled any of it out.

    AND THE TWO BEHIND IT NEED CALIBRATION, corrected 2026-08-12 by a second validation pass.
    Relative `item_verify` alone accepted a foreign card in 10 of 540 comparisons, and an old
    3.0 identity bound admitted a measured different-person pair at 2.565. The production driver
    now refuses targeted work unless it can pass a calibrated identity bound below that collision
    and a calibrated absolute sheet-distance ceiling, in addition to this geometry and correct
    index invalidation. This function remains only one layer and must not be treated as proof of
    profile identity.
    """
    if abs(block.height - index_block.height) > extent_tolerance_px:
        return (f"the card under heart {index_block.heart_ordinal} is {block.height}px tall in "
                f"this frame but {index_block.height}px in the index, a disagreement of "
                f"{abs(block.height - index_block.height)}px against a {extent_tolerance_px}px "
                "tolerance. Two frames that both bounded one card are two measurements of one "
                "fixed quantity, so exactly one of them is wrong")
    inset = heart[1] - block.y0
    index_inset = index_heart[1] - index_block.page_y0
    if abs(inset - index_inset) > extent_tolerance_px:
        return (f"the heart sits {inset}px below its card's top edge in this frame but "
                f"{index_inset}px below it in the index — the count reached the right ordinal "
                "on a card that is not laid out like the one the index recorded")
    if abs(heart[0] - index_heart[0]) > crosscheck_tolerance_px:
        return (f"the heart is at x={heart[0]} in this frame but x={index_heart[0]} in the "
                "index, and a list scroll has no horizontal component, so these are not the "
                "same glyph")
    return None
