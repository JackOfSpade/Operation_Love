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

THE THREE OUTCOMES, AND WHY "CANNOT TELL" IS A STATE AND NOT A FALSY BOOL
--------------------------------------------------------------------------
`ScrollTopVerdict.state` is exactly one of:

  * `SCROLL_TOP_CONFIRMED`  -- the band IS the filter-chips row. Counting may start.
  * `SCROLL_TOP_REFUTED`    -- the band is something else, on a screen already established to be
                               a Hinge profile, which means the sticky per-profile header is
                               showing and we are scrolled. Scroll up and ask again.
  * `SCROLL_TOP_UNKNOWN`    -- neither. No `identity_band` declared, or the distance landed in
                               the dead zone between the two bounds. NEVER "at top".

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
)

# Maintained for backwards compatibility:
_SCROLL_TOP_BAND_FINGERPRINT = _SCROLL_TOP_BAND_FINGERPRINT_COMPATIBLE

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


# --- the three outcomes ---------------------------------------------------------------
SCROLL_TOP_CONFIRMED = "confirmed_top"       # the band IS the filter-chips row
SCROLL_TOP_REFUTED = "confirmed_not_top"     # the band is positively something else
SCROLL_TOP_UNKNOWN = "cannot_tell"           # neither — and never to be read as "at top"


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
    """
    state: str
    distance: float | None
    reason: str
    band: tuple[float, float, float, float] | None
    grid: tuple[int, int]
    confirm_max: float
    refute_min: float

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

    def __bool__(self) -> bool:
        # A frozen dataclass is truthy by default, so `if confirm_scroll_top(...):` would read
        # "cannot tell" and "confirmed not at top" as a confirmed top — the exact off-by-N this
        # module exists to prevent, written in the most natural-looking way possible. numpy's
        # precedent: an ambiguous truth value is a TypeError, not a guess.
        raise TypeError(
            "ScrollTopVerdict has no truth value: 'cannot tell' is not 'at top'. Test "
            "`.confirmed` explicitly, or call require_scroll_top() to make anything but a "
            f"confirmed top a hard stop. This verdict was {self.state!r}: {self.reason}")


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


def confirm_scroll_top(frame: bytes, *,
                       identity_band: tuple[float, float, float, float] | None,
                       fingerprint: Sequence[int] | Sequence[Sequence[int]] = _SCROLL_TOP_BAND_FINGERPRINTS,
                       grid: tuple[int, int] = _FINGERPRINT_GRID,
                       confirm_max: float = _CONFIRM_MAX_DIST,
                       refute_min: float = _REFUTE_MIN_DIST) -> ScrollTopVerdict:
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

    PRECONDITION, and it is not a formality: this must be a screen already established to be a
    Hinge profile. `SCROLL_TOP_REFUTED` says "this strip is not the filter-chips row", which only
    means "we are scrolled" once something else has ruled out a paywall, a dialog or a blank
    framebuffer — all of which refute here. See the module docstring.

    Returns a verdict. Raises `ScrollTopError` only when it could not look (see
    `band_fingerprint`) or when the parameters contradict each other.
    """
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
            confirm_max=confirm_max, refute_min=refute_min,
            reason=("no identity_band declared for this app, so the filter-chips signal doc 5.5 "
                    "requires cannot be read at all — this is 'cannot tell', never 'at top'"))

    seen = band_fingerprint(frame, identity_band=identity_band, grid=grid)
    dist = min(fingerprint_distance(seen, fp) for fp in candidates)

    if dist <= confirm_max:
        return ScrollTopVerdict(
            state=SCROLL_TOP_CONFIRMED, distance=dist, band=tuple(identity_band), grid=grid,
            confirm_max=confirm_max, refute_min=refute_min,
            reason=(f"the identity band matches Hinge's profile-independent filter-chips row at "
                    f"{dist:.3f} <= {confirm_max} — at any scroll offset past the top the app's "
                    "sticky per-profile header covers this strip with the person's name instead"))
    if dist >= refute_min:
        return ScrollTopVerdict(
            state=SCROLL_TOP_REFUTED, distance=dist, band=tuple(identity_band), grid=grid,
            confirm_max=confirm_max, refute_min=refute_min,
            reason=(f"the identity band is {dist:.3f} from the filter-chips row, at or past the "
                    f"{refute_min} bound — on a screen already established as a Hinge profile "
                    "that means the sticky per-profile header is showing and we are scrolled"))
    return ScrollTopVerdict(
        state=SCROLL_TOP_UNKNOWN, distance=dist, band=tuple(identity_band), grid=grid,
        confirm_max=confirm_max, refute_min=refute_min,
        reason=(f"the identity band is {dist:.3f} from the filter-chips row, inside the "
                f"({confirm_max}, {refute_min}) dead zone — too far to be that row and too near "
                "to be positively something else. No frame in the calibration corpus has ever "
                "landed here; this is 'cannot tell', never 'at top'"))


def require_scroll_top(frame: bytes, *,
                       identity_band: tuple[float, float, float, float] | None,
                       fingerprint: Sequence[int] | Sequence[Sequence[int]] = _SCROLL_TOP_BAND_FINGERPRINTS,
                       grid: tuple[int, int] = _FINGERPRINT_GRID,
                       confirm_max: float = _CONFIRM_MAX_DIST,
                       refute_min: float = _REFUTE_MIN_DIST) -> ScrollTopVerdict:
    """`confirm_scroll_top`, but anything short of a confirmed top is `ScrollTopUnconfirmed`.

    This is the entry point for every caller that is about to COUNT — building an item index with
    `at_scroll_top=True`, or stepping forward to the k-th heart. Doc 5.5: "treat failure to
    confirm as a hard stop". Returning a falsy value there would leave the stop to a caller's
    `if`, which is precisely the shape of mistake `ScrollTopVerdict.__bool__` refuses to allow;
    an exception makes proceeding-by-accident impossible instead of merely discouraged.

    A caller that genuinely wants to branch — a scroll-up loop deciding whether to swipe again —
    should call `confirm_scroll_top` and read `.confirmed` / `.refuted` / `.unknown`, where
    REFUTED means "scroll further" and UNKNOWN means "stop and show the frame", which are not the
    same action.
    """
    verdict = confirm_scroll_top(
        frame, identity_band=identity_band, fingerprint=fingerprint, grid=grid,
        confirm_max=confirm_max, refute_min=refute_min)
    if not verdict.confirmed:
        raise ScrollTopUnconfirmed(
            f"scroll top not confirmed ({verdict.state}): {verdict.reason}. Counting items from "
            "an unconfirmed top gives a systematic off-by-N in every heart ordinal, so this is a "
            "hard stop (ops/OPENER-REDESIGN.md 5.5)", verdict)
    return verdict
