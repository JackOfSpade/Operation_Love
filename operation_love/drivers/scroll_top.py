"""Is this frame at the TOP of a Hinge profile? A positive answer, a negative one, or neither.

ops/OPENER-REDESIGN.md 5.5 specifies counting navigation as "scroll to top, then step forward
counting distinct hearts, tap the k-th", and then names the dependency that makes or breaks it:

    "Counting forward from a false top gives a systematic off-by-N. So the design needs an
    affirmative top confirmation before counting starts, and one exists: at scroll-top
    `identity_band` shows Hinge's filter-chips row rather than a name, which is a positive
    signal rather than an absence. Confirm top by that signal, and treat failure to confirm as
    a hard stop."

This module is that confirmation, and nothing else.

    confirm_scroll_top(frame, identity_band=...) -> ScrollTopVerdict
    require_scroll_top(frame, identity_band=...) -> ScrollTopVerdict   # or raises
    band_pinned_evidence(frames, identity_band=..., page_offsets=...) -> PinnedBandEvidence

Deliberately a leaf module on the same terms as `segment.py`, `frameshift.py`, `item_index.py`
and `item_crops.py`: pure functions over frame BYTES plus explicit calibration parameters, no
device, no I/O, no global state, no `HingeDriver` import, and no driver state of any kind. It
needs none: the whole signal is one already-calibrated strip of one screencap.

WHY THIS IS A POSITIVE CONFIRMATION AND `item_index._scroll_top_evidence` IS NOT
--------------------------------------------------------------------------------
The two are complements and BOTH are wanted; they are not alternatives.

`item_index._scroll_top_evidence` can only CONTRADICT a false assertion, and says so in its own
docstring: it asks whether some frame saw page background above the capture's topmost block,
which a genuine top always has and a band edge slicing a card usually does not. "Usually" is the
whole problem — measured 21 of 21 false tops refused on one capture but 9 of 10 on the other,
because a false top whose band edge lands in a gutter looks exactly like a real one, and the
gutter is 53px of a ~1027px card pitch. It converts "undetectable" into "usually detected".

This module asks a question whose YES means something. At a genuine scroll top `identity_band`
shows Hinge's own filter-chips row; the moment the card is scrolled at all, Hinge's sticky
per-profile header slides in and that same strip shows the person's NAME instead
(`HINGE_SPEC.identity_band`'s own comment records the live measurement). Chips present is
therefore evidence FOR the top rather than absence of evidence against it.

WHAT MAKES A FIXED REFERENCE LEGITIMATE HERE
---------------------------------------------
A positive test needs something to be positive about, and the reason one constant can serve every
profile is itself a measurement rather than an assumption: the filter-chips strip is Hinge's own
chrome, and it is PROFILE-INDEPENDENT.

  [corpus: `_band(frame, identity_band)` over the three gitignored calibration captures
  (ops/calibration/, two different profiles, three separate sessions 21:12, 22:54 and 23:19 UTC).
  The five frames that are at a genuine scroll top produce a BYTE-IDENTICAL band -- pairwise
  mean-abs distance 0.000 at the shipped 64x16 identity grid and 0.000 at this module's 16x4 grid
  -- while every one of the other 143 frames lands at 14.391 or further. That 0.000 across two
  different people is simultaneously the reason the constant generalises and the proof that the
  strip carries nothing about either person: a value that does not change with the profile cannot
  encode the profile.]

`_SCROLL_TOP_BAND_FINGERPRINT` below is that measurement stored as 64 grey levels, one per cell
of a 16x4 grid over a 756x111 strip -- roughly a 47x28 px block average per cell. It is a
fingerprint, not an image: nothing is recoverable from it, and it is app chrome in the first
place. No frame, crop or derivative of `ops/calibration/` appears anywhere else in this repo, and
the tests for this module use synthetic bands and a synthetic reference exactly like every other
test in Part B.

WHY 16x4 AND NOT THE 64x16 THE DRIVER ALREADY USES
---------------------------------------------------
Coarser is strictly better here, which is not the usual direction and is worth the two lines.

Both grids separate perfectly (0.000 against >=14.39), so resolution buys no separation. What it
costs is robustness to Hinge redrawing that strip a few pixels higher or lower -- a banner
appearing, a status-bar height change, a font metric shift -- which must NOT read as "not at top"
when it is only "the same chrome, moved".

  [corpus: re-cropping the reference frame's band with a deliberate offset, in mean-abs grey
  levels. A 4px vertical shift costs 2.25 at 16x4 against 6.67 at 64x16; +-2px in both axes costs
  1.17 against 3.97; additive Gaussian capture noise at sd=4 costs 2.61 against 2.35 (grid does
  not matter for noise). So the coarse grid tolerates roughly 3x the layout drift for the same
  confirm bound, while the nearest non-chips band stays 11.2 away.]

WHEN THE PREMISE ITSELF FAILS: A SCREEN-PINNED CHIPS ROW (Hinge 10.1.0, 2026-08-28)
------------------------------------------------------------------------------------
Everything above rests on one premise, and it is a property of the APP rather than of this code:
the filter-chips row is drawn at this rect ONLY at a scroll top, and every other offset covers it
with the sticky per-profile header. Hinge 10.1.0 has a per-profile header state -- the "expanded"
one, which carries the filter-chips row along with it -- that FALSIFIES that premise.

  [device, run 8fb11094ef4d, the six `item_index_refused_3fa8b8db66fd` evidence frames, captured
  mid-profile at page offsets 6180, 6615, 7110, 7637, 8152 and 8485 (2305px of proven scroll):
  frame rows 0..516 are byte-identical across all 15 pairs (max abs diff 0), and this gate
  returned `confirmed_top` at distance 0.000 on every one of them. An affirmative "we are at the
  top" was therefore being issued thousands of pixels down a profile, into a HARD GATE the rest
  of doc 5.5 leans on. On two OTHER profiles in the same run the same band correctly refuted
  (13.219 and 12.359), so this is a per-screen-state defect, not a blanket one.]

No single frame can detect this: one frame showing the chips row is exactly what a genuine top
looks like. It takes TWO frames known to sit at DIFFERENT scroll offsets, which is
`item_index._screen_fixed_islands`'s reasoning applied to this strip -- a screen-fixed element
keeps a constant FRAME position while the page moves, and page content cannot do that.
`band_pinned_evidence` is that test, and `confirm_scroll_top` takes its result as an OPTIONAL
parameter: given evidence that the chips row is pinned, a match to it is no longer allowed to
mean "at top".

WHAT "PINNED" MUST MEAN HERE, BECAUSE THE OBVIOUS READING BREAKS THE WORKING CASE
----------------------------------------------------------------------------------
The sticky per-profile header is pinned TOO, and always was: "stays pixel-identical for the whole
profile" is the device measurement `item_identity` is built on, and 143 corpus frames of one
profile read 0.000 against each other. So "the band holds constant pixels across a scroll" is
true in the state this gate still works in, and a helper that stopped there would disable the gate
on the collapsed-header profiles it reads correctly today.

The distinction is WHICH thing is pinned. This gate reads the chips row as evidence FOR the top,
so it is only dead when the CHIPS ROW is what survives the scroll. `band_pinned_evidence`
therefore requires both halves: the band holds identical pixels across proven scrolling AND every
one of those frames confirms as the chips row. Two frames at offsets more than the band's own
height apart cannot both be at the top, so a chips row on both is a direct counterexample to the
premise. The collapsed-header state fails the second half -- its band is a person, not chrome --
and is untouched.

THE FOUR OUTCOMES, AND WHY "CANNOT TELL" IS A STATE AND NOT A FALSY BOOL
--------------------------------------------------------------------------
`ScrollTopVerdict.state` is exactly one of:

  * `SCROLL_TOP_CONFIRMED`   -- the band IS the filter-chips row. Counting may start.
  * `SCROLL_TOP_REFUTED`     -- the band is something else, on a screen already established to be
                                a Hinge profile, which means the sticky per-profile header is
                                showing and we are scrolled. Scroll up and ask again.
  * `SCROLL_TOP_UNKNOWN`     -- neither. No `identity_band` declared, or the distance landed in
                                the dead zone between the two bounds. NEVER "at top".
  * `SCROLL_TOP_UNAVAILABLE` -- the caller supplied `band_pinned_evidence` proving this app
                                version pins the chips row to the screen, so the signal carries
                                no information about scroll position and this CHECK cannot answer
                                on this app version at all. Deliberately NOT folded into REFUTED:
                                "not at top" tells a scroll-up loop to scroll again, and no amount
                                of scrolling fixes a pinned band. It is a recalibration stop and
                                its reason says so. Deliberately not folded into UNKNOWN either:
                                UNKNOWN is "this frame's band is between the bounds", a per-frame
                                measurement that another frame may resolve; this is "the signal
                                itself is dead", which no frame will.

A two-valued answer would collapse the third case into one of the other two, and both collapses
are bugs: fold it into CONFIRMED and a systematic off-by-N ships silently; fold it into REFUTED
and a run scrolls up forever against a screen that will never confirm, with no reason recorded.

`ScrollTopVerdict.__bool__` RAISES rather than returning True, on numpy's precedent for an
ambiguous truth value. A frozen dataclass is truthy by default, so `if confirm_scroll_top(...):`
would silently read every outcome -- including "cannot tell" -- as a confirmed top. That is the
single most dangerous line anyone could write against this API, and it is now a TypeError at the
first execution rather than an off-by-N in production. Callers test `.confirmed`, or call
`require_scroll_top`, which turns anything but CONFIRMED into `ScrollTopUnconfirmed`.

WHAT THIS MODULE DOES NOT DECIDE, AND WHICH LAYER HAS TO
----------------------------------------------------------
  * SCREEN IDENTITY, and here it is load-bearing rather than a formality. REFUTED means "this
    strip is not the filter-chips row", which is only the same thing as "we are scrolled" once
    something else has established that this is a Hinge profile screen at all. A paywall, a
    dialog or a blank framebuffer all refute. Settle `_identity_of`/`_screen_is` FIRST -- the same
    precondition `segment.py` and `item_index.py` state for themselves.
  * The RESIDUAL SCROLL OFFSET. This confirms that Hinge's sticky header has not slid in, which
    is not the same as "the page is at row 0 to the pixel". [corpus: the hand-scrolled capture's
    first two frames are 0px apart and both confirm; the next frame is 202px further on and
    measures 17.947, so the swap happens somewhere in (0, 202].] That residual is bounded well
    below one card pitch (738..1027px measured on one profile), so it cannot change any heart
    ORDINAL, which is the only thing counting navigation reads. A caller needing exact page rows
    must get them from `item_index`, not from here.
  * WHAT TO DO ABOUT A REFUTED TOP. Scrolling up is `hinge._scroll_to_top`'s job and must keep
    going through the driver's humanized path; this module never touches the device.
  * WHETHER THE OWNER'S FILTER CHIPS CHANGED. The strip shows the account's own filter settings,
    so an owner who edits them invalidates the constant. The failure mode is UNKNOWN or REFUTED
    forever -- loud, and a hard stop at the gate -- never a false top. Re-measuring is a
    calibration task with the owner present, not something to soften a threshold over.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# =====================================================================================
# Calibration constants.
#
# Every number here was MEASURED, not chosen. Sources:
#   [doc]    ops/OPENER-REDESIGN.md 5.5 (the affirmative-confirmation requirement) and
#            `HINGE_SPEC.identity_band`'s own comment, which carries the original live
#            Pixel 7a measurement of this strip (0.00 between scroll-top frames, 17.95 against
#            a scrolled one, against a change_threshold of 9.0).
#   [corpus] a re-measurement of the gitignored calibration captures through the SHIPPED
#            `hinge._band` decode (148 frames: 115 hand-scrolled, 24 bot-scrolled at
#            read_scroll_frac 0.16, 9 bot-scrolled at 0.55, over two profiles; plus the
#            out-of-likes paywall screencap as a non-profile control). Numbers only -- the
#            frames are real people's profiles and never leave ops/calibration/.
# =====================================================================================

# Downsample grid (w, h) the identity band is reduced to before comparison. See the module
# docstring's "WHY 16x4" section for the drift measurements that chose it over the driver's own
# 64x16 `hinge._IDENTITY_DS`. 16x4 = 64 cells over the 756x111px strip `identity_band` cuts on a
# 1080x2400 screencap, i.e. roughly a 47x28px block average per cell.
_FINGERPRINT_GRID = (16, 4)

# The filter-chips row itself, as `_FINGERPRINT_GRID` grey levels in row-major order.
#
# [corpus: produced by `hinge._band(frame, HINGE_SPEC.identity_band, (16, 4))` -- the shipped
# decode, at this module's grid -- and IDENTICAL, value for value, on all five genuine scroll-top
# frames in the corpus: frame 1 of both bot-scrolled captures, frames 1 and 2 of the hand-scrolled
# capture, and its frame 115 (where the human had advanced the deck and the next card was back at
# its own top). Two different profiles, three sessions, one array.]
#
# The filter-chips row variants, as `_FINGERPRINT_GRID` grey levels in row-major order.
# Variant 1: `( Compatible ) Active today Dating Intent ...` (calibration corpus baseline)
_SCROLL_TOP_BAND_FINGERPRINT_COMPATIBLE = (
    254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254,
    254, 251, 251, 250, 251, 250, 233, 232, 232, 249, 252, 251, 251, 250, 252, 252,
    248, 244, 236, 239, 249, 238, 236, 239, 240, 237, 246, 235, 236, 243, 248, 250,
    245, 240, 186, 191, 239, 242, 215, 193, 229, 239, 242, 192, 188, 222, 245, 251,
)

# Variant 2: `( Age v ) Height v Dating Intent ...` (filter-selected layout)
_SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT = (
    254, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    253, 237, 232, 231, 243, 253, 251, 251, 250, 251, 253, 251, 251, 250, 251, 251,
    240, 236, 237, 242, 235, 248, 236, 236, 240, 247, 250, 239, 234, 242, 237, 238,
    237, 233, 185, 222, 236, 248, 207, 182, 214, 237, 253, 224, 190, 196, 198, 196,
)

# Variant 3: `( Signals ) Age v Height v Dating Intent ...` (HingeX Signals selection).
#
# Measured from the Pixel 7a at a confirmed visual card top on 2026-08-18. The HingeX
# "Most Compatible" card sits below the filter strip; it is not itself the signal the gate
# reads.  This records only the 16x4 greyscale calibration fingerprint, never the profile
# screenshot or its contents.
#
# The first calibration of this variant was 4.859 levels from a later, visibly confirmed Signals
# top. That placed a real top in the deliberate 3..9 dead zone. This fingerprint is the later
# top's shipped-band decode, so the gate is calibrated to the layout actually on the device.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS = (
    254, 255, 255, 255, 255, 255, 254, 254, 254, 255, 255, 254, 254, 254, 255, 255,
    254, 186, 150, 150, 189, 249, 232, 232, 231, 246, 236, 233, 234, 232, 240, 244,
    223, 91, 91, 88, 88, 222, 235, 232, 239, 232, 236, 225, 225, 237, 235, 234,
    220, 93, 123, 118, 95, 221, 225, 204, 236, 233, 238, 207, 201, 231, 236, 236,
)

# Variant 4: `( Signals ) Age v Height v Dating Intent ...` (HingeX Signals unselected).
#
# Captured on the same confirmed-top Most Compatible card on 2026-08-18.  It happened to be
# 2.891 levels from the older Compatible reference -- inside the 3.0 tolerance, but too close to
# the edge to leave as an accidental match.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED = (
    254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 251, 251, 250, 251, 250, 233, 232, 232, 249, 239, 233, 233, 232, 243, 247,
    248, 244, 236, 239, 249, 238, 236, 239, 240, 232, 235, 233, 233, 240, 234, 233,
    245, 240, 186, 191, 239, 242, 215, 193, 230, 233, 236, 193, 189, 223, 234, 235,
)

# Variant 5: `( Signals ) Age v Height v Dating Intent ...` (HingeX Signals selected, current
# Android rendering).
#
# A second visually confirmed selected-Signals top on 2026-08-18 measured 3.281 levels from
# Variant 3.  It is a genuine rendering variant rather than a looser threshold: the capture
# visibly shows the profile-independent filter chips in this exact band, while a scrolled card
# replaces the strip with the person's sticky header.  Keeping both calibrated fingerprints
# preserves the 3.0 confirmation bound instead of admitting every unknown band between 3 and 9.
# This records only the 16x4 greyscale filter-strip fingerprint, never a profile image or name.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_SELECTED_CURRENT = (
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 200, 168, 168, 203, 251, 232, 232, 232, 248, 238, 232, 233, 231, 242, 246,
    227, 87, 78, 75, 85, 223, 237, 237, 240, 232, 235, 230, 231, 239, 235, 233,
    215, 88, 126, 120, 89, 218, 221, 196, 232, 233, 237, 198, 193, 226, 235, 236,
)

# Variant 6: current HingeX Signals filter strip, measured from Jaden's visibly top-of-card
# Android capture on 2026-08-19.  It was 3.891 from the unselected Signals reference, just
# outside the 3.0 confirm tolerance; registering this discrete chrome variant keeps the safety
# dead zone intact rather than broadening it to admit unknown bands.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_CURRENT = (
    254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    252, 251, 251, 250, 253, 241, 232, 232, 239, 249, 233, 233, 233, 233, 249, 237,
    250, 237, 239, 241, 247, 235, 236, 242, 236, 232, 234, 233, 236, 240, 232, 235,
    253, 213, 184, 207, 247, 239, 189, 214, 234, 235, 220, 182, 206, 231, 232, 235,
)

# Variant 7: `( Signals ) Age v Height v Dating Intent ...` (HingeX Signals selected, dark/
# charcoal pill rendering).
#
# Measured from the Pixel 7a at a visually confirmed card-top on 2026-08-19 (Lana capture,
# ce1de851af2e run).  The Signals pill renders dark/charcoal rather than purple in this
# rendering mode.  Both available frames of this layout (00001 and 00008 of that run) measure
# 2.594 from each other -- within the 3.0 confirm bound -- and 17.344--17.750 from the nearest
# existing variant, confirming this is a discrete chrome state rather than capture noise.
# Before this fingerprint was added, both frames were classified as `confirmed_not_top` (the
# band exceeded the 9.0 refute threshold), causing the observe loop to stall for 5m33s while
# the card was genuinely at its scroll top the whole time.
# This records only the 16x4 greyscale filter-strip fingerprint, never a profile image or name.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL = (
    255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    219, 181, 181, 206, 254, 234, 232, 232, 248, 242, 233, 235, 232, 241, 248, 233,
     84,  65,  64,  68, 213, 239, 236, 239, 232, 233, 232, 233, 238, 234, 232, 231,
     75, 129, 134,  82, 198, 231, 183, 228, 233, 238, 197, 182, 219, 233, 236, 216,
)

# Variant 8: `( Signals ) Age v Height v Dating Intent ...` (HingeX Signals selected, dark/
# charcoal pill rendering — second calibration frame).
#
# Measured from the same Pixel 7a run (ce1de851af2e, 00008_observe_waiting_before.png, same
# 2026-08-19 session as Variant 7).  The two frames measure 2.594 from each other (within the
# 3.0 confirm bound), so they are unambiguously the same layout, but when compared against the
# Variant-7 fingerprint 00008 lands at 3.547 — just outside the 3.0 confirm bound.  Keeping
# both calibration points means each confirms the other rather than landing in the dead zone.
# This records only the 16x4 greyscale filter-strip fingerprint, never a profile image or name.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL_B = (
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    207, 164, 164, 192, 252, 233, 233, 231, 246, 241, 234, 236, 232, 240, 247, 232,
     81,  75,  74,  69, 210, 239, 232, 239, 231, 234, 228, 228, 236, 235, 233, 230,
    81, 128, 133,  86, 201, 233, 187, 231, 233, 239, 202, 187, 223, 234, 237, 220,
)

# Variant 9: `Age v  Height v  Dating Intentions v  Active …` (current non-Signals
# filter strip).
#
# Measured from a visibly top-of-card Pixel 7a capture on 2026-08-19 (Val,
# c679dfb4e458, both the capture frame and the later settled observe frame).  The filter
# controls are the familiar white outline pills, but their current text/raster layout is 7.203
# from the preceding selected-Signals reference: inside the deliberate 3..9 uncertainty gap.
# The two saved frames have an identical 16x4 fingerprint, and the screen visibly shows the
# profile-independent filter chips above the card, so this is a discrete chrome variant rather
# than evidence of a scrolled card.  Registering the measured variant preserves the narrow
# confirmation bound rather than treating the dead zone as a permissive match.
#
# This contains only the 16x4 greyscale filter-strip fingerprint, never profile pixels or text.
_SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_CURRENT = (
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    238, 233, 232, 241, 248, 233, 233, 232, 234, 249, 235, 233, 233, 233, 233, 233,
    233, 236, 242, 235, 233, 234, 233, 237, 239, 232, 234, 231, 239, 235, 235, 236,
    235, 187, 218, 234, 236, 215, 182, 210, 232, 233, 229, 191, 195, 200, 195, 199,
)

# Variant 10: `( Signals ) Age v Height v Dating Intent ...` (selected Signals, current
# Android rendering with a purple profile banner below the filter strip).
#
# Measured from Jenny's visibly top-of-card capture on 2026-08-20
# (b93807731946/00011_capture_before.png). The band visibly contains Hinge's filter controls;
# the profile header, first media card, and purple "shows thoughtful signals" banner all begin
# below it. It measured 8.953 from Variant 7, inside the deliberate 3..9 uncertainty gap, so
# the capture was correctly *not* treated as a top until this discrete chrome rendering was
# registered. Keeping the exact chrome-only fingerprint retains the 3.0 confirmation bound
# rather than weakening the gate for unknown layouts.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_PURPLE_BANNER = (
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    243, 186, 181, 184, 241, 243, 232, 232, 239, 249, 233, 233, 233, 233, 249, 237,
    159,  70,  72,  64, 149, 242, 236, 242, 236, 232, 234, 233, 236, 240, 232, 235,
    138,  99, 130, 102, 130, 246, 190, 214, 234, 235, 220, 182, 206, 231, 232, 235,
)

# Variant 11: `Signals  Age v  Height v  Dating Intent ...` (unselected Signals pill, Hinge
# 10.0.1 rendering).
#
# Measured 2026-08-21 after the Pixel 7a's Hinge app auto-updated from 9.134.0 to 10.0.1. Three
# independent captures taken at different times that day -- phone_check3.png (10:26),
# phone_after_abort2.png and phone_idle_check.png (10:33) -- all visibly show Stephanie's
# profile at a genuine scroll top (chips row, name, first photo, no sticky header) and all three
# decode to this EXACT 16x4 fingerprint, 0.000 apart, so this is a stable re-render rather than
# capture jitter. Against every fingerprint already on file it measured 6.672 from Variant 4
# (`SIGNALS_UNSELECTED`, the nearest) -- squarely inside the 3..9 dead zone, and the reason a
# live calibration capture aborted twice before this variant was registered. It is the same
# unselected-Signals chip content as Variant 4, redrawn by the updated app; keeping it as its
# own discrete fingerprint retains the 3.0 confirmation bound instead of loosening it for a
# whole app-version rendering change.
# This records only the 16x4 greyscale filter-strip fingerprint, never a profile image or name.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_10_0_1 = (
    254, 254, 253, 253, 254, 255, 251, 250, 251, 255, 253, 250, 250, 250, 254, 255,
    253, 250, 252, 253, 251, 245, 233, 236, 233, 242, 235, 236, 236, 234, 238, 240,
    247, 241, 214, 216, 244, 241, 228, 222, 236, 233, 237, 215, 215, 232, 236, 236,
    247, 247, 214, 220, 245, 241, 231, 218, 242, 234, 240, 223, 216, 240, 237, 236,
)

# Variant 12: `Signals  Age v  Height v  Dating Intentions` (current unselected-Signals
# strip).
#
# Measured from the visibly top-of-card Pixel 7a debug capture on 2026-08-24
# (`b55de0b40e1b/00001_capture_entry_refused_before.png`).  The filter chips appear above the
# profile name and first photo, with no sticky per-profile header.  Before this discrete chrome
# variant was registered, its closest existing candidate was Variant 9 at 8.859 after the
# bounded alignment sweep -- inside the deliberate 3..9 dead zone -- so a session-start rewind
# exhausted all twelve bounded strokes even though it had already reached the top.  Registering
# this chrome-only fingerprint keeps the 3.0 confirmation bound instead of weakening it for an
# unknown rendering.  No profile pixels or text are stored here.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_CURRENT = (
    254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    251, 251, 250, 252, 236, 232, 232, 245, 244, 232, 233, 232, 238, 249, 233, 233,
    234, 239, 245, 242, 235, 237, 242, 233, 234, 233, 233, 239, 237, 232, 233, 231,
    190, 187, 224, 245, 230, 186, 224, 234, 237, 205, 182, 217, 234, 234, 220, 190,
)

# Variant 13: `Signals  Age v  Height v  Dating Intentions` (Hinge 10.1.0 current
# unselected-Signals strip).
#
# Measured from a read-only, visibly confirmed Pixel 7a top on 2026-08-26
# (`/tmp/hinge_10_1_top.png` at capture time): the Signals/Age/Height chips are above the
# profile name and first photo, with no sticky per-profile header.  Its nominal crop was 7.094
# from the prior 10.0.1 Variant 11; the bounded alignment sweep found 4.625 at -12px.  Both are
# inside the deliberate 3..9 dead zone, so widening either threshold would be the wrong fix.
#
# Replaying this added candidate against all 148 hand/bot-scroll corpus frames left every prior
# classification unchanged (5 tops, 143 refuted); the closest previously-refuted frame stayed
# 10.531 above the 9.0 refute floor.  This records only the 16x4 greyscale chrome fingerprint,
# never profile pixels or text.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_10_1_0 = (
    254, 251, 251, 251, 252, 250, 236, 235, 235, 249, 241, 236, 236, 235, 244, 247,
    249, 246, 239, 241, 249, 242, 240, 242, 244, 237, 240, 237, 237, 243, 239, 238,
    248, 244, 208, 211, 244, 244, 228, 214, 238, 237, 241, 214, 210, 234, 239, 239,
    253, 251, 250, 252, 251, 246, 235, 237, 237, 244, 237, 239, 238, 238, 240, 242,
)

# Variant 14: `Age  Height  Dating Intentions` (Hinge 10.1.0 selected-Signals/filter-selected
# strip).
#
# Measured from a read-only, visibly confirmed Pixel 7a top on 2026-08-26
# (`/tmp/hinge_10_1_signals_selected_top.png` at capture time): the filter chips are above the
# profile name and first card, with no sticky per-profile header.  The selected Signals chip is
# omitted from this rendering, leaving Age/Height/Dating Intentions as the first visible chips.
# Its nominal crop was 9.375 from Variant 13; the bounded alignment sweep found 6.766 at -12px,
# inside the deliberate 3..9 dead zone.  This is therefore another discrete chrome state, not
# evidence for relaxing either safety threshold.
#
# Replaying the candidate against the canonical 148-frame hand/bot-scroll corpus produced zero
# false confirmations: the five known tops still confirm, 142 frames refute, and one prior
# refutation becomes a conservative UNKNOWN (5.781), which merely requests another bounded
# scroll/recheck.  This records only the 16x4 greyscale chrome fingerprint, never profile pixels
# or text.
_SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_10_1_0 = (
    247, 245, 252, 252, 243, 242, 249, 250, 251, 246, 241, 246, 249, 252, 250, 251,
    244, 242, 217, 236, 242, 244, 227, 215, 232, 242, 242, 236, 218, 223, 222, 221,
    250, 241, 238, 242, 243, 243, 241, 238, 242, 241, 244, 240, 241, 238, 241, 241,
    254, 250, 247, 246, 252, 253, 247, 247, 246, 250, 254, 248, 247, 247, 247, 247,
)

# Variant 15: `Age  Height  Dating Intentions` (Hinge 10.1.0 selected-filter strip after the
# profile deck has fully settled).
#
# Measured from a second read-only capture of the same visibly confirmed Pixel 7a card top on
# 2026-08-26 (`/tmp/hinge_current_top.png` at capture time), several minutes after Variant 14.
# Hinge had repainted the same profile-independent chips to this lighter settled rendering.  Its
# closest nominal registered match was the historical Age/Height Variant 2 at 4.531; the bounded
# alignment sweep still stopped at 4.375 (-11px), inside the deliberate dead zone.  Recording
# the exact settled chrome state avoids broadening the 3.0 confirmation threshold.
# Replaying it against the canonical 148-frame corpus changed no historical state (5 confirmed,
# 142 refuted, 1 already-conservative UNKNOWN) and produced zero false confirmations; the closest
# historical non-top remained 15.547 away.
# This is only the 16x4 greyscale filter-strip fingerprint, never profile pixels or text.
_SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_SETTLED_10_1_0 = (
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    253, 237, 232, 231, 244, 246, 232, 233, 232, 237, 249, 234, 233, 233, 233, 233,
    240, 236, 237, 242, 234, 234, 233, 233, 238, 237, 232, 233, 232, 239, 234, 235,
    237, 233, 185, 222, 234, 237, 208, 182, 215, 233, 234, 223, 191, 196, 199, 197,
)

# Variant 16: `Age  Height  Dating Intentions` (Hinge 10.1.0 alternate settled filter chrome).
#
# Measured from another read-only, visibly confirmed Pixel 7a card top on 2026-08-26
# (`/tmp/hinge_top_variant16.png` at capture time).  The chips sit above the profile name and
# first card, with no sticky header.  Its bounded alignment sweep reached 3.438 at -10px against
# the prior candidates: only 0.438 beyond the deliberately strict confirmation ceiling, but
# still UNKNOWN rather than permission to count.  Registering the exact chrome state preserves
# that ceiling instead of relaxing it for all unknown bands.  The canonical 148-frame replay
# changed no state (5 confirmed, 142 refuted, 1 conservative UNKNOWN), produced zero false
# confirmations, and left the closest historical non-top 13.953 away.  This is only the 16x4
# greyscale filter-strip fingerprint, never profile pixels or text.
_SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_ALT_10_1_0 = (
    255, 252, 251, 252, 255, 253, 251, 251, 251, 255, 254, 251, 251, 251, 251, 251,
    242, 234, 235, 233, 243, 234, 236, 236, 234, 240, 238, 234, 236, 235, 236, 236,
    240, 226, 226, 237, 233, 235, 215, 218, 234, 235, 237, 218, 218, 224, 217, 218,
    240, 226, 220, 242, 233, 240, 217, 218, 240, 236, 238, 226, 218, 222, 222, 224,
)

# Variant 17: `Signals  Age  Height  Dating Intentions` (Hinge 10.1.0 unselected-Signals
# alternate chrome).
#
# Measured from a read-only, visibly confirmed Pixel 7a top on 2026-08-26
# (`/tmp/hinge_top_alignment.png` at capture time).  The complete chips row sits above the
# profile name and first card, with no sticky header.  Its best existing match was 5.375 at the
# -12px alignment boundary; extending the diagnostic sweep to -60..+60 still bottomed out at
# 4.688 (-26px), proving this is a different raster/layout state rather than merely an undersized
# alignment search.  Registering the exact state retains the 3.0/9.0 safety bounds.  This stores
# only the 16x4 app-chrome fingerprint, never profile pixels or text.  The canonical 148-frame
# replay changed no state (5 confirmed, 142 refuted, 1 UNKNOWN) and produced zero false
# confirmations, but its existing UNKNOWN now sits only 3.3125 away; therefore neither the 3.0
# threshold nor the +/-12px search may be widened alongside this candidate.
_SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_ALT_10_1_0 = (
    253, 242, 242, 244, 250, 245, 242, 247, 246, 243, 243, 240, 242, 247, 242, 245,
    253, 236, 222, 233, 250, 245, 226, 237, 244, 243, 239, 222, 234, 244, 242, 244,
    253, 252, 252, 253, 253, 244, 242, 242, 243, 248, 242, 242, 243, 242, 248, 242,
    254, 254, 254, 254, 254, 254, 253, 253, 254, 255, 253, 253, 253, 253, 255, 254,
)

_SCROLL_TOP_BAND_FINGERPRINTS: tuple[tuple[int, ...], ...] = (
    _SCROLL_TOP_BAND_FINGERPRINT_COMPATIBLE,
    _SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_SELECTED_CURRENT,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_CURRENT,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL_B,
    _SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_CURRENT,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_PURPLE_BANNER,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_10_0_1,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_CURRENT,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_10_1_0,
    _SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_10_1_0,
    _SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_SETTLED_10_1_0,
    _SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_ALT_10_1_0,
    _SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_ALT_10_1_0,
)

# Maintained for backwards compatibility:
_SCROLL_TOP_BAND_FINGERPRINT = _SCROLL_TOP_BAND_FINGERPRINT_COMPATIBLE

# Bounded VERTICAL alignment search: how many pixels of crop drift `confirm_scroll_top` will
# absorb by re-cropping the identity band before falling back to the fixed dy=0 crop.
#
# Measured live 2026-08-22: a hybrid-rewind capture whose identity band genuinely WAS the
# filter-chips row still measured 5.625--6.547 against every registered fingerprint at the
# crop's nominal (dy=0) position -- squarely in the (3.0, 9.0) dead zone -- because the page had
# settled a few pixels lower than any calibration frame. Re-cropping that SAME live frame 8px
# higher (dy=-8) measured 1.0, comfortably inside confirm_max. The gate was asking "is the page
# at this precise pixel offset" when doc 5.5 only needs "is this strip the filter-chips row",
# and single-digit pixels of rendering jitter is far short of a real scroll.
#
# 12 is sized off two anchors, not chosen freeform: it comfortably covers the 8px jitter actually
# measured, while staying well under the ~111px identity-band height (16x4 grid over a 756x111
# crop -- see _FINGERPRINT_GRID's comment) so the sweep can never slide a DIFFERENT UI element
# into the band, and vastly under the ~219px minimum scroll gesture ("Scroll step vs spacing
# aliasing", measured 2026-08-11) so it can never be mistaken for a real scroll completing.
_ALIGNMENT_SEARCH_PX = 12

# At or below this mean-abs distance (0..255) from the fingerprint, the band IS the filter-chips
# row and the frame is CONFIRMED at scroll top.
#
# [corpus: every genuine scroll-top frame measures exactly 0.000, so the bound is set by how far
# a genuine top may be perturbed and still be recognised, not by where the real tops sit. The
# perturbations measured on the reference band are 2.25 for a 4px vertical layout shift, 0.86 for
# a 4px horizontal one, 1.17 for +-2px in both axes, and 2.61 for additive capture noise at
# sd=4. 3.0 clears all of them. Against that, the CLOSEST anything not-the-chips-row has ever
# measured is 11.203 (a synthetic flat white band -- the chips row is mostly white, which is why
# the nearest hazard is blank rather than textual); the out-of-likes paywall's band is 12.703 and
# the nearest real scrolled frame 14.391. So the confirm bound sits 3.7x below the nearest
# non-top and above every measured perturbation of a real one.]
_CONFIRM_MAX_DIST = 3.0

# At or above this distance the band is positively something ELSE, and the frame is CONFIRMED NOT
# at scroll top. Between the two bounds the answer is "cannot tell".
#
# 9.0 is `HINGE_SPEC.change_threshold`, the app's own calibrated "this band is showing something
# different now" constant -- the same number `_identity_of` decides same/new/top on. It is
# declared rather than imported because `hinge` imports this package's leaf modules and not the
# other way round; `tests/test_scroll_top.py` pins the two together so the copy cannot drift.
#
# [corpus: the dead zone 3.0 <= d < 9.0 is EMPTY on all 148 frames. Every frame is either exactly
# 0.000 or 14.391+. The gap exists to be honest about a band this module has never seen, not to
# classify one it has.]
_REFUTE_MIN_DIST = 9.0


# --- the four outcomes ----------------------------------------------------------------
SCROLL_TOP_CONFIRMED = "confirmed_top"       # the band IS the filter-chips row
SCROLL_TOP_REFUTED = "confirmed_not_top"     # the band is positively something else
SCROLL_TOP_UNKNOWN = "cannot_tell"           # neither — and never to be read as "at top"
# The chips row is pinned to the SCREEN on this app version, so it appears at every scroll offset
# and this check has nothing left to read. Reachable ONLY when a caller supplies
# `band_pinned_evidence` that proves it — see the module docstring's "WHEN THE PREMISE ITSELF
# FAILS" section. It is not "not at top" (scrolling cannot fix it) and not "cannot tell" (no other
# frame can resolve it); it is a recalibration stop.
SCROLL_TOP_UNAVAILABLE = "check_unavailable"


class ScrollTopError(RuntimeError):
    """The gate could not LOOK: undecodable frame bytes, PIL/numpy missing, or a fingerprint
    whose length contradicts the grid it is supposed to describe.

    Distinct from `SCROLL_TOP_UNKNOWN`, which is a verdict reached by looking, and distinct
    again from `ScrollTopUnconfirmed`, which is a caller policy. The separation is the same one
    `segment.SegmentationError` draws and for the same reason: "we could not look" must never be
    reachable by a route that also produces a verdict, or a broken install degrades into a
    stream of plausible-looking answers.
    """


class ScrollTopUnconfirmed(RuntimeError):
    """`require_scroll_top` was given a frame it could not CONFIRM as scroll top.

    Raised for `SCROLL_TOP_REFUTED` and for `SCROLL_TOP_UNKNOWN` alike, because to a caller
    about to start counting hearts they are the same thing: not confirmed. Carries the verdict
    on `.verdict` so a stop record can report which one it was and at what distance.
    """

    def __init__(self, message: str, verdict: "ScrollTopVerdict"):
        super().__init__(message)
        self.verdict = verdict


@dataclass(frozen=True)
class ScrollTopVerdict:
    """One frame's answer, with the evidence that produced it.

    `distance` is the mean-abs grey-level distance (0..255) between the frame's identity band and
    the reference fingerprint — the same metric `hinge._band_dist` uses on the same band, so the
    number is directly comparable to `change_threshold` and to the measurements in
    `HINGE_SPEC.identity_band`'s comment. It is None only when no comparison happened at all
    (no `identity_band` declared).

    `reason` is always populated, including on a CONFIRMED verdict, so a debug record or a hub
    stop line can quote why without re-deriving anything.

    `alignment_offset_px` is the vertical pixel offset (see `_ALIGNMENT_SEARCH_PX`) that produced
    `distance` -- 0 when the fixed dy=0 crop was already the best match, which is the overwhelming
    common case. Appended as the LAST field with a default so the existing positional
    constructions in `tests/test_hinge_calibrate.py` and `tests/test_hinge_operational_evidence.py`
    (7 positional args, matching the 7 fields above) keep working unchanged.
    """
    state: str
    distance: float | None
    reason: str
    band: tuple[float, float, float, float] | None
    grid: tuple[int, int]
    confirm_max: float
    refute_min: float
    alignment_offset_px: int = 0

    @property
    def confirmed(self) -> bool:
        """True ONLY for `SCROLL_TOP_CONFIRMED`. This is the value to pass as
        `item_index.build_item_index(..., at_scroll_top=...)`, and the only route by which that
        assertion should ever become True."""
        return self.state == SCROLL_TOP_CONFIRMED

    @property
    def refuted(self) -> bool:
        return self.state == SCROLL_TOP_REFUTED

    @property
    def unknown(self) -> bool:
        return self.state == SCROLL_TOP_UNKNOWN

    @property
    def unavailable(self) -> bool:
        """True ONLY for `SCROLL_TOP_UNAVAILABLE`: the chips-row signal is pinned to the screen on
        this app version, so this gate cannot answer at all. A caller that branches on
        `.refuted` to scroll again must not treat this as one — see the module docstring."""
        return self.state == SCROLL_TOP_UNAVAILABLE

    def __bool__(self) -> bool:
        # A frozen dataclass is truthy by default, so `if confirm_scroll_top(...):` would read
        # "cannot tell" and "confirmed not at top" as a confirmed top — the exact off-by-N this
        # module exists to prevent, written in the most natural-looking way possible. numpy's
        # precedent: an ambiguous truth value is a TypeError, not a guess.
        raise TypeError(
            "ScrollTopVerdict has no truth value: 'cannot tell' is not 'at top'. Test "
            "`.confirmed` explicitly, or call require_scroll_top() to make anything but a "
            f"confirmed top a hard stop. This verdict was {self.state!r}: {self.reason}")


@dataclass(frozen=True)
class PinnedBandEvidence:
    """Does a capture PROVE that the chips-row signal this gate reads is pinned to the SCREEN?

    `pinned` is the only actionable field and it is the conjunction of two things, both required
    (see the module docstring's "WHAT \"PINNED\" MUST MEAN HERE"): the identity band held IDENTICAL
    pixels across frames at proven-different scroll offsets, and every one of those frames read as
    the filter-chips row. The second half is what keeps the collapsed-header state — where the
    band is a person's pinned sticky header and the gate still works — out of this.

    Everything else is the evidence behind it, kept because this licenses a HARD GATE to stop:
    `offsets` the distinct page offsets used, `offset_span` how far apart the furthest two were,
    `band_height_px` the height the span had to clear for the sightings' page rows to be disjoint,
    `frames_compared` how many frames contributed, `distinct_bands` how many different band
    contents were seen among them (1 is the only value that can prove pinning), and
    `max_chips_distance` the worst distance any of them sat from the chips row.

    `reason` is always populated, on a negative answer as much as a positive one, because "the
    proof was unavailable" and "the band moves like page content" are different facts and an
    operator reading a refusal needs to know which one they have.
    """
    pinned: bool
    reason: str
    band: tuple[float, float, float, float] | None
    grid: tuple[int, int]
    offsets: tuple[int, ...] = ()
    offset_span: int | None = None
    band_height_px: int | None = None
    frames_compared: int = 0
    distinct_bands: int = 0
    max_chips_distance: float | None = None

    def __bool__(self) -> bool:
        # A frozen dataclass is truthy by default, so `if band_pinned_evidence(...):` would read
        # "nothing here proves anything" as "the band is pinned" — the exact inversion that would
        # disable a working gate on every capture. `ScrollTopVerdict.__bool__`'s precedent.
        raise TypeError(
            "PinnedBandEvidence has no truth value: an unproven pinning is not a proven one. "
            f"Test `.pinned` explicitly. This evidence was pinned={self.pinned!r}: {self.reason}")


def fingerprint_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Mean absolute difference (0..255) between two equal-length band fingerprints.

    Pure Python over ints — no cv2, no numpy — on `item_crops.CropSignature.distance`'s
    precedent, so a stored fingerprint can be compared anywhere the vision extras are not
    installed: a hub view, a log replay, a test. The metric is deliberately identical to
    `hinge._band_dist`'s so every distance in this module is comparable to the ones already
    recorded against this band.
    """
    if len(a) != len(b):
        raise ScrollTopError(
            f"fingerprint length mismatch: {len(a)} against {len(b)} — these describe different "
            "grids and no distance between them means anything")
    if not a:
        raise ScrollTopError("empty fingerprints have no distance")
    # The explicit length guard above is part of this metric's public contract.
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b, strict=True)) / len(a)


def band_fingerprint(frame: bytes, *, identity_band: tuple[float, float, float, float],
                     grid: tuple[int, int] = _FINGERPRINT_GRID) -> tuple[int, ...]:
    """`frame`'s identity band as a row-major tuple of `grid` grey levels.

    Goes through `hinge._band`, which is the driver's ONE decode of this rect, rather than
    re-implementing crop-and-resize here. Two decode paths that disagree by a grey level or two
    is a measured trap in this repo rather than a hypothetical one — see `_band`'s own docstring
    and `item_crops.signature_of`'s — and the whole confirm bound below is 3.0 grey levels wide,
    so it is not a trap this module could absorb.

    The import is DEFERRED for `segment.segment_frame`'s reason: `hinge` imports the leaf modules
    of this package, so a module-level import here would make the graph cyclic, and resolving the
    attribute at call time keeps tests that monkeypatch `hinge._band` working through this path.

    Raises `ScrollTopError` when the frame cannot be read at all — `_band` collapses "PIL/numpy
    missing" and "these bytes are not an image" into a single None, and both mean this gate
    could not look.
    """
    from . import hinge as _hinge

    band = _hinge._band(frame, identity_band, grid)
    if band is None:
        raise ScrollTopError(
            f"identity band {identity_band} could not be read from {len(frame)} bytes of frame "
            "— either the vision extras (PIL/numpy) are missing or the bytes are not an image. "
            "This is 'could not look', which is not a verdict")
    return tuple(int(v) for v in band.reshape(-1))


def _decode_for_alignment_sweep(frame: bytes):
    """Frame bytes -> one decoded grayscale PIL image, or None if it cannot be decoded at all.

    The SAME PIL call `hinge._band` makes before its own crop-and-resize step (see
    `hinge._band_of_image`'s docstring) — pulled out here, not reimplemented, so
    `confirm_scroll_top`'s `_ALIGNMENT_SEARCH_PX` sweep can decode a frame ONCE and hand the
    result to `_band_fingerprint_at_offset` for every one of its 25 candidate crops, instead of
    the sweep paying for a fresh PNG decode per crop (measured 468ms -> 19.6ms per
    `confirm_scroll_top` call, a 23.9x speedup, 2026-08-23 perf pass). It does no crop or resize
    of its own — that stays exclusively `hinge._band_of_image`'s job, so these crops can never
    drift from `_band`'s single-crop callers by so much as a rounding step.
    """
    try:
        from io import BytesIO

        from PIL import Image
        return Image.open(BytesIO(frame)).convert("L")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> caller treats it as "could not look"
        return None


def _band_fingerprint_at_offset(frame: bytes, *, identity_band: tuple[float, float, float, float],
                                grid: tuple[int, int], dy_px: int,
                                image=None) -> tuple[int, ...] | None:
    """`band_fingerprint`'s decode, with the identity band's crop shifted `dy_px` pixels DOWN
    (negative moves it up) in the source frame before `hinge._band_of_image` crops and resizes
    it. This is `confirm_scroll_top`'s alignment search calling the SAME shipped crop-and-resize
    at a different rect, not a second crop-and-resize implementation — see `band_fingerprint`'s
    own docstring for why that distinction matters at a 3.0-grey-level confirm bound.

    `identity_band` is normalised fractions of the frame, so an exact-PIXEL shift needs the
    frame's actual height. `image` is an already-decoded frame (`_decode_for_alignment_sweep`'s
    return value) that a caller sweeping many offsets of the SAME frame — `confirm_scroll_top` is
    the only one — has already paid to decode; passing it here is what lets the sweep avoid
    redecoding the PNG per offset. `None` (the default, and what every direct caller in
    tests/test_scroll_top.py uses) decodes `frame` itself first, unchanged from before this
    parameter existed.

    Returns None — never raises — for three distinct "skip this one offset" situations: `image`
    was not supplied and this frame could not be decoded on its own, the shift would push the
    band off the top or bottom edge of the frame, or the shifted crop could not be produced from
    an otherwise-decoded frame. All three are safe to skip because `confirm_scroll_top` always
    resolves dy=0 first and raises loudly there if the frame cannot be read at all — see its own
    docstring — so a broken PIL/numpy install can never be silently absorbed by "skip this
    offset" here.
    """
    from . import hinge as _hinge

    x0, y0, x1, y1 = identity_band
    if image is None:
        image = _decode_for_alignment_sweep(frame)
        if image is None:
            return None
    frame_h = image.size[1]
    if frame_h <= 0:
        return None

    dy_frac = dy_px / frame_h
    y0, y1 = y0 + dy_frac, y1 + dy_frac
    if y0 < 0.0 or y1 > 1.0:
        return None  # would slide the crop off the top or bottom edge of the frame

    try:
        band = _hinge._band_of_image(image, (x0, y0, x1, y1), grid)
    except Exception:  # noqa: BLE001 — malformed crop at this offset -> skip it, not a fresh
        # "could not look": dy=0 already proved (or will prove) this frame decodes.
        return None
    return tuple(int(v) for v in band.reshape(-1))


def confirm_scroll_top(frame: bytes, *,
                       identity_band: tuple[float, float, float, float] | None,
                       fingerprint: Sequence[int] | Sequence[Sequence[int]] = _SCROLL_TOP_BAND_FINGERPRINTS,
                       grid: tuple[int, int] = _FINGERPRINT_GRID,
                       confirm_max: float = _CONFIRM_MAX_DIST,
                       refute_min: float = _REFUTE_MIN_DIST,
                       pinned_evidence: PinnedBandEvidence | None = None) -> ScrollTopVerdict:
    """Affirmatively confirm — or refuse to — that `frame` shows a Hinge profile at scroll top.

    `identity_band` is the normalised `(x0, y0, x1, y1)` rect of the filter-chips / sticky-header
    strip. Pass the driver's `self.identity_band` (the config-merged value), NOT
    `spec.identity_band`, or an operator override is silently ignored — the same rule
    `segment_frame` states for `content_band`. `None` is accepted and answers
    `SCROLL_TOP_UNKNOWN`: an app with no declared band is a thing this gate cannot judge, and
    saying so is the honest answer. It is not an error, because `identity_band` is optional on
    `AndroidAppSpec` by design.

    `fingerprint`, `grid` and the two bounds are calibration constants with measured defaults,
    exposed on `segment_frame`'s precedent so a validation pass — or a test with a synthetic
    band — can vary one without editing the module. When multiple candidate fingerprints are
    configured, the nearest matching filter-chips variant determines the distance.

    Before comparing, the identity band is also re-cropped at every integer vertical offset in
    `[-_ALIGNMENT_SEARCH_PX, +_ALIGNMENT_SEARCH_PX]` and the BEST (minimum) distance across every
    offset and every candidate fingerprint wins — see `_ALIGNMENT_SEARCH_PX`'s comment for the
    live jitter this absorbs. `confirm_max` and `refute_min` are unchanged by this: only the crop
    alignment is searched, never the bounds. The winning offset is recorded on
    `ScrollTopVerdict.alignment_offset_px` and, when nonzero, named in `reason`.

    `pinned_evidence` is `band_pinned_evidence`'s answer for the capture this frame came from,
    and it is OPTIONAL with a default of None so every existing call site keeps its exact previous
    behaviour: no evidence supplied is not evidence of absence, and this gate has no way of its own
    to obtain it from one frame. When it is supplied AND proves the chips row is pinned to the
    screen, a match to that row can no longer mean "at top" and the verdict becomes
    `SCROLL_TOP_UNAVAILABLE` instead of `SCROLL_TOP_CONFIRMED` — see the module docstring's "WHEN
    THE PREMISE ITSELF FAILS". It downgrades ONLY that outcome: REFUTED and UNKNOWN are unchanged,
    because a band that is NOT the chips row is the collapsed-header state this gate still reads
    correctly, and turning its refusals into recalibration stops would break a working case to fix
    a broken one. Evidence taken at a different rect or grid than this call is a `ScrollTopError`
    rather than a silent downgrade: two different crops of a screen say nothing about each other.

    PRECONDITION, and it is not a formality: this must be a screen already established to be a
    Hinge profile. `SCROLL_TOP_REFUTED` says "this strip is not the filter-chips row", which only
    means "we are scrolled" once something else has ruled out a paywall, a dialog or a blank
    framebuffer — all of which refute here. See the module docstring.

    Returns a verdict. Raises `ScrollTopError` only when it could not look (see
    `band_fingerprint`) or when the parameters contradict each other.
    """
    if pinned_evidence is not None and pinned_evidence.pinned:
        if identity_band is not None and pinned_evidence.band != tuple(identity_band):
            raise ScrollTopError(
                f"the pinning evidence was measured on band {pinned_evidence.band} but this call "
                f"reads {tuple(identity_band)} — these are different crops of the screen, and one "
                "crop being pinned says nothing about another")
        if pinned_evidence.grid != tuple(grid):
            raise ScrollTopError(
                f"the pinning evidence was measured at grid {pinned_evidence.grid} but this call "
                f"compares at {tuple(grid)} — these describe different geometries, and the "
                "evidence's 'identical pixels' finding does not carry across them")
    if confirm_max >= refute_min:
        raise ScrollTopError(
            f"confirm_max {confirm_max} is not below refute_min {refute_min}: that collapses the "
            "'cannot tell' band, so every distance would be classified and the third outcome "
            "this gate exists for could never be reached")
    expected = grid[0] * grid[1]

    if isinstance(fingerprint, (list, tuple)) and fingerprint and isinstance(fingerprint[0], (list, tuple)):
        candidates = list(fingerprint)
    elif fingerprint == _SCROLL_TOP_BAND_FINGERPRINT:
        candidates = list(_SCROLL_TOP_BAND_FINGERPRINTS)
    else:
        candidates = [fingerprint]

    for fp in candidates:
        if len(fp) != expected:
            raise ScrollTopError(
                f"fingerprint has {len(fp)} values but grid {grid} needs {expected} — one "
                "of the two is stale, and comparing them would compare different geometries")

    if identity_band is None:
        return ScrollTopVerdict(
            state=SCROLL_TOP_UNKNOWN, distance=None, band=None, grid=grid,
            confirm_max=confirm_max, refute_min=refute_min, alignment_offset_px=0,
            reason=("no identity_band declared for this app, so the filter-chips signal doc 5.5 "
                    "requires cannot be read at all — this is 'cannot tell', never 'at top'"))

    # ONE decode for the whole sweep -- dy=0 included (2026-08-23 perf pass). Every one of the
    # 2*_ALIGNMENT_SEARCH_PX + 1 candidate crops below used to redecode the whole PNG from
    # scratch (dy=0 through band_fingerprint -> hinge._band, every other offset through its own
    # hinge._band call inside _band_fingerprint_at_offset); measured 468ms/call on
    # ops/calibration/scroll_20260811T211209Z. They now all read from this single decoded image
    # instead, via hinge._band_of_image (measured 19.6ms/call, 23.9x). A frame that cannot be
    # decoded at all must still fail LOUDLY here, before the alignment search below gets a chance
    # to treat every other offset's failure as merely "skip it" (see
    # _band_fingerprint_at_offset's docstring) -- this replaces band_fingerprint's own raising
    # contract for that same "could not look" case, word for word, so no caller sees a different
    # failure mode than before this frame was shared.
    image = _decode_for_alignment_sweep(frame)
    if image is None:
        raise ScrollTopError(
            f"identity band {identity_band} could not be read from {len(frame)} bytes of frame "
            "— either the vision extras (PIL/numpy) are missing or the bytes are not an image. "
            "This is 'could not look', which is not a verdict")

    seen = _band_fingerprint_at_offset(frame, identity_band=identity_band, grid=grid, dy_px=0,
                                       image=image)
    if seen is None:
        raise ScrollTopError(
            f"identity band {identity_band} could not be read from {len(frame)} bytes of frame "
            "— either the vision extras (PIL/numpy) are missing or the bytes are not an image. "
            "This is 'could not look', which is not a verdict")
    dist = min(fingerprint_distance(seen, fp) for fp in candidates)
    offset = 0

    for dy in range(-_ALIGNMENT_SEARCH_PX, _ALIGNMENT_SEARCH_PX + 1):
        if dy == 0:
            continue
        shifted = _band_fingerprint_at_offset(frame, identity_band=identity_band, grid=grid,
                                              dy_px=dy, image=image)
        if shifted is None:
            continue
        shifted_dist = min(fingerprint_distance(shifted, fp) for fp in candidates)
        if shifted_dist < dist:
            dist, offset = shifted_dist, dy

    offset_note = (f", after searching a {offset:+d}px vertical alignment offset within "
                   f"+/-{_ALIGNMENT_SEARCH_PX}px of layout jitter" if offset else "")

    if dist <= confirm_max and pinned_evidence is not None and pinned_evidence.pinned:
        # The distance is still reported: it is the measurement, and it is exactly the number an
        # operator needs to see that a 0.000 match to the chips row has stopped meaning anything.
        return ScrollTopVerdict(
            state=SCROLL_TOP_UNAVAILABLE, distance=dist, band=tuple(identity_band), grid=grid,
            confirm_max=confirm_max, refute_min=refute_min, alignment_offset_px=offset,
            reason=(f"the identity band matches Hinge's profile-independent filter-chips row at "
                    f"{dist:.3f} <= {confirm_max}{offset_note}, but this capture PROVES that row "
                    f"is pinned to the SCREEN rather than drawn only at the top: "
                    f"{pinned_evidence.reason}. A strip that shows the chips at every scroll "
                    "offset cannot mean 'at top', so doc 5.5's affirmative confirmation is "
                    "structurally UNAVAILABLE on this app version — not refuted, and not a frame "
                    "another scroll or another look could resolve. What needs recalibrating is "
                    "the scroll-top signal itself: apps.hinge.identity_band and this module's "
                    "chips-row fingerprints, re-measured against the app's current per-profile "
                    "header states with the owner and a device. Widening or narrowing a threshold "
                    "cannot help; the strip carries no scroll information at all"))

    if dist <= confirm_max:
        return ScrollTopVerdict(
            state=SCROLL_TOP_CONFIRMED, distance=dist, band=tuple(identity_band), grid=grid,
            confirm_max=confirm_max, refute_min=refute_min, alignment_offset_px=offset,
            reason=(f"the identity band matches Hinge's profile-independent filter-chips row at "
                    f"{dist:.3f} <= {confirm_max}{offset_note} — at any scroll offset past the "
                    "top the app's sticky per-profile header covers this strip with the person's "
                    "name instead"))
    if dist >= refute_min:
        return ScrollTopVerdict(
            state=SCROLL_TOP_REFUTED, distance=dist, band=tuple(identity_band), grid=grid,
            confirm_max=confirm_max, refute_min=refute_min, alignment_offset_px=offset,
            reason=(f"the identity band is {dist:.3f} from the filter-chips row, at or past the "
                    f"{refute_min} bound{offset_note} — on a screen already established as a "
                    "Hinge profile that means the sticky per-profile header is showing and we "
                    "are scrolled"))
    return ScrollTopVerdict(
        state=SCROLL_TOP_UNKNOWN, distance=dist, band=tuple(identity_band), grid=grid,
        confirm_max=confirm_max, refute_min=refute_min, alignment_offset_px=offset,
        reason=(f"the identity band is {dist:.3f} from the filter-chips row, inside the "
                f"({confirm_max}, {refute_min}) dead zone{offset_note} — too far to be that row "
                "and too near to be positively something else. No frame in the calibration "
                "corpus has ever landed here; this is 'cannot tell', never 'at top'"))


def _not_pinned(reason: str, *, band, grid, **evidence) -> PinnedBandEvidence:
    return PinnedBandEvidence(pinned=False, reason=reason, band=band, grid=grid, **evidence)


def band_pinned_evidence(frames: Sequence[bytes], *,
                         identity_band: tuple[float, float, float, float] | None,
                         page_offsets: Sequence[int | None],
                         fingerprint: Sequence[int] | Sequence[Sequence[int]] = _SCROLL_TOP_BAND_FINGERPRINTS,
                         grid: tuple[int, int] = _FINGERPRINT_GRID,
                         confirm_max: float = _CONFIRM_MAX_DIST,
                         refute_min: float = _REFUTE_MIN_DIST) -> PinnedBandEvidence:
    """Do these frames PROVE that the filter-chips row is pinned to the screen on this app build?

    `frames` are one capture's frames and `page_offsets` the page offset each was taken at, in the
    same order and the same length, `None` for a frame whose offset is not known (the shape
    `item_index` already carries for frames past a broken correspondence chain — those are skipped
    here exactly as `item_index._islands` skips them).

    WHAT "KNOWN TO HAVE SCROLLED" MEANS HERE, AND WHY IT IS OFFSETS RATHER THAN A FLAG OR A
    MEASUREMENT OF OUR OWN
    ---------------------------------------------------------------------------------------
    Three designs were available and only one of them is honest at a leaf module's altitude:

      * the caller asserting "these scrolled" as a bool. Rejected: the conclusion is the whole
        question, and a gate that accepts the answer it was built to compute is not a gate. It
        would also make a caller's mistake invisible, because there would be no measurement on the
        record to contradict.
      * this function measuring the scroll itself, from the pixels. Rejected: the repo has exactly
        ONE calibrated answer to "how far did the page move" (`frameshift.estimate_shift`, whose
        docstring records the two estimators that were tried and failed on these same captures),
        and its callers have already run it — `item_index` hands its result around as the offset
        chain. A second estimator inside the confirmation gate would be a second opinion on a
        question this repo has measured once, and it would give a PIL/numpy-only leaf module a
        hard cv2 dependency, so the gate's strictness would start depending on the host install.
      * the caller supplying the MEASUREMENTS it already has, and this function doing the
        deciding. Chosen, and it is `item_index._screen_fixed_islands`'s contract exactly: that
        function is handed offsets and decides screen-fixedness itself. A caller cannot obtain
        `pinned=True` by asserting anything; it has to hand over offsets that actually spread far
        enough, on frames whose bands are actually identical and actually read as the chips row.

    The error direction is what makes this safe rather than merely tidy. A wrong `pinned=True` can
    only turn a CONFIRMED into `SCROLL_TOP_UNAVAILABLE`, which is a loud stop at a hard gate — a
    false stop, never a false top. A wrong `pinned=False`, or a caller that supplies no offsets at
    all, degrades to exactly the behaviour that shipped before this function existed, which is the
    same "unavailable proof degrades to the status quo, never to a new outcome" rule
    `item_index._observations` states for its own screen-fixed strips.

    ALL FOUR CONDITIONS, AND WHY NONE IS NEGOTIABLE
    -----------------------------------------------
      * at least TWO DISTINCT page offsets. At one offset the page did not move, so a pinned strip
        and a piece of page content predict identical pixels and neither hypothesis is tested.
      * `max(offset) - min(offset) >= the band's own height in px`. Below that the two sightings'
        page extents still overlap, so one tall piece of page content could produce both. At or
        above it the page rows are disjoint and only a screen-fixed element can show the same
        pixels twice. (`item_index._screen_fixed_islands` imposes the same bound for the same
        reason. It also means two frames far enough apart cannot BOTH be at the scroll top, which
        is what makes a chips row on both a direct counterexample to this gate's premise.)
      * every sighting shows the SAME band. Unanimity, no tolerance, no quorum: this is an
        identity test on pixels, and "mostly the same" is what a scrolling page looks like. An
        animating or live-updating header reads as "not proven" and is left alone.
      * every sighting CONFIRMS as the filter-chips row, judged by `confirm_scroll_top` itself
        rather than by a second comparison written here. This is the half that keeps the working
        case working: the sticky per-profile header is pinned too (measured 0.000 across 143
        frames of one profile), so without it a collapsed-header capture — the state where this
        gate reads correctly — would prove itself "pinned" and disable the gate everywhere.

    Never raises for a frame it cannot read: an undecodable frame is skipped and named in the
    reason, on `capture_profile_identity`'s contract, because the caller for this is a capture
    description rather than a per-frame gate and a broken decode must degrade to "not proven"
    (the status quo) rather than to an exception. It DOES raise `ScrollTopError` when `frames` and
    `page_offsets` are different lengths, which is not a frame problem but a caller pairing one
    frame's band with another frame's scroll position.
    """
    if len(frames) != len(page_offsets):
        raise ScrollTopError(
            f"{len(frames)} frame(s) against {len(page_offsets)} page offset(s): these describe "
            "different captures, and pairing them would attribute one frame's band to another "
            "frame's scroll position")
    band = tuple(identity_band) if identity_band is not None else None
    grid = tuple(grid)
    if identity_band is None:
        return _not_pinned(
            "no identity_band is declared for this app, so there is no strip to test for pinning "
            "and nothing for the scroll-top gate to have been reading in the first place",
            band=None, grid=grid)

    usable: list[tuple[int, int, tuple[int, ...], ScrollTopVerdict]] = []
    unreadable: list[int] = []
    frame_size: tuple[int, int] | None = None
    geometry_mismatches: list[tuple[int, tuple[int, int]]] = []
    for i, (frame, offset) in enumerate(zip(frames, page_offsets, strict=True)):
        if offset is None:
            continue
        image = _decode_for_alignment_sweep(frame)
        if image is None:
            unreadable.append(i)
            continue
        seen = _band_fingerprint_at_offset(frame, identity_band=identity_band, grid=grid, dy_px=0,
                                           image=image)
        if seen is None:
            unreadable.append(i)
            continue
        if frame_size is None:
            frame_size = image.size
        elif image.size != frame_size:
            # Page offsets are pixels in the capture's shared coordinate space.  A resized or
            # rotated frame has no such shared space with the first frame, even if both happen
            # to reduce to the same small fingerprint grid.  Treating that coincidence as proof
            # would let a mixed-device capture disable the scroll-top gate.
            geometry_mismatches.append((i, image.size))
            continue
        usable.append((i, int(offset), seen,
                       confirm_scroll_top(frame, identity_band=identity_band,
                                          fingerprint=fingerprint, grid=grid,
                                          confirm_max=confirm_max, refute_min=refute_min)))

    trailer = (f" ({len(unreadable)} frame(s) could not be read at all: {unreadable})"
               if unreadable else "")
    if geometry_mismatches:
        assert frame_size is not None  # the mismatch is defined relative to the first readable frame
        mismatched = ", ".join(
            f"{i} ({width}x{height})" for i, (width, height) in geometry_mismatches)
        return _not_pinned(
            f"this capture's identity-band frames do not share one pixel coordinate space: the "
            f"first readable frame is {frame_size[0]}x{frame_size[1]}, but frame(s) {mismatched} "
            "have different geometry. Page offsets cannot compare those frames, so identical "
            "reduced fingerprints are not proof that the band is pinned to the screen"
            f"{trailer}",
            band=band, grid=grid, frames_compared=len(usable))
    if len(usable) < 2:
        return _not_pinned(
            f"only {len(usable)} of this capture's {len(frames)} frame(s) carry both a page offset "
            f"and a readable identity band, and one sighting proves nothing: at a single offset a "
            f"screen-pinned strip and a piece of page content predict identical pixels{trailer}",
            band=band, grid=grid, frames_compared=len(usable))

    offsets = tuple(sorted({o for _i, o, _b, _v in usable}))
    span = offsets[-1] - offsets[0]
    x0, y0, x1, y1 = identity_band
    band_height_px = round(y1 * frame_size[1]) - round(y0 * frame_size[1])   # `_band_of_image`'s crop
    distinct = {b for _i, _o, b, _v in usable}
    worst = max(v.distance for _i, _o, _b, v in usable if v.distance is not None)
    common = dict(band=band, grid=grid, offsets=offsets, offset_span=span,
                  band_height_px=band_height_px, frames_compared=len(usable),
                  distinct_bands=len(distinct), max_chips_distance=worst)

    if len(offsets) < 2:
        return _not_pinned(
            f"all {len(usable)} frames sit at the same page offset ({offsets[0]}), so the page did "
            f"not move between them and nothing distinguishes a screen-pinned strip from page "
            f"content that simply had not scrolled yet{trailer}", **common)
    if span < band_height_px:
        return _not_pinned(
            f"these {len(usable)} frames span only {span}px of scroll, less than the band's own "
            f"{band_height_px}px height, so the page rows it would have had to display still "
            f"overlap and one tall piece of page content could explain every sighting{trailer}",
            **common)
    not_chips = [i for i, _o, _b, v in usable if not v.confirmed]
    if not_chips:
        return _not_pinned(
            f"the identity band does not read as the filter-chips row on {len(not_chips)} of these "
            f"{len(usable)} frames (frame(s) {not_chips}), so the signal this gate reads as "
            f"evidence FOR the top is not what is on screen at these offsets. A band that is "
            f"constant because it holds the sticky per-profile header is the state this gate was "
            f"calibrated for and still reads correctly; only a pinned CHIPS ROW makes it "
            f"unanswerable{trailer}", **common)
    if len(distinct) != 1:
        return _not_pinned(
            f"the identity band shows {len(distinct)} different contents across these "
            f"{len(usable)} frames, so it is not a static element — an animating or live-updating "
            f"strip reads exactly like this. Unanimity, no tolerance: 'mostly the same' is what a "
            f"scrolling page looks like{trailer}", **common)

    return PinnedBandEvidence(
        pinned=True,
        reason=(f"the identity band {band} holds IDENTICAL pixels, all of them within "
                f"{confirm_max} of the filter-chips row (worst {worst:.3f}), on {len(usable)} "
                f"frames spanning {span}px of page scroll ({offsets[0]}..{offsets[-1]}) — more "
                f"than its own {band_height_px}px height, so the page rows those sightings would "
                f"otherwise have had to display are disjoint. Page content cannot do that and "
                f"chrome cannot do anything else, and two frames that far apart cannot both be at "
                f"the scroll top: this app version draws the chips row at every scroll offset, "
                f"pinned to the screen{trailer}"),
        **common)


def require_scroll_top(frame: bytes, *,
                       identity_band: tuple[float, float, float, float] | None,
                       fingerprint: Sequence[int] | Sequence[Sequence[int]] = _SCROLL_TOP_BAND_FINGERPRINTS,
                       grid: tuple[int, int] = _FINGERPRINT_GRID,
                       confirm_max: float = _CONFIRM_MAX_DIST,
                       refute_min: float = _REFUTE_MIN_DIST,
                       pinned_evidence: PinnedBandEvidence | None = None) -> ScrollTopVerdict:
    """`confirm_scroll_top`, but anything short of a confirmed top is `ScrollTopUnconfirmed`.

    This is the entry point for every caller that is about to COUNT — building an item index with
    `at_scroll_top=True`, or stepping forward to the k-th heart. Doc 5.5: "treat failure to
    confirm as a hard stop". Returning a falsy value there would leave the stop to a caller's
    `if`, which is precisely the shape of mistake `ScrollTopVerdict.__bool__` refuses to allow;
    an exception makes proceeding-by-accident impossible instead of merely discouraged.

    A caller that genuinely wants to branch — a scroll-up loop deciding whether to swipe again —
    should call `confirm_scroll_top` and read `.confirmed` / `.refuted` / `.unknown` /
    `.unavailable`, where REFUTED means "scroll further", UNKNOWN means "stop and show the frame"
    and UNAVAILABLE means "stop and recalibrate — no further gesture on this app version can make
    this signal answer", which are three different actions.
    """
    verdict = confirm_scroll_top(
        frame, identity_band=identity_band, fingerprint=fingerprint, grid=grid,
        confirm_max=confirm_max, refute_min=refute_min, pinned_evidence=pinned_evidence)
    if not verdict.confirmed:
        raise ScrollTopUnconfirmed(
            f"scroll top not confirmed ({verdict.state}): {verdict.reason}. Counting items from "
            "an unconfirmed top gives a systematic off-by-N in every heart ordinal, so this is a "
            "hard stop (ops/OPENER-REDESIGN.md 5.5)", verdict)
    return verdict
