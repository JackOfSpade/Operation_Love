"""Is the profile on screen right now the profile this item index was built from?

ops/OPENER-REDESIGN.md 5.7's final audit states the requirement this module exists for, and it is
the only one of that audit's two carried-forward requirements that is about the screen rather
than about a crop:

    "A stale or foreign index is NOT caught at model index 1 today ... no geometry available to
    navigation can catch it, because Hinge's first cards are stereotyped (same 974px height, same
    885px heart inset, same x=938 across profiles). The next phase must add an IDENTITY check at
    navigation ENTRY, before any gesture, IN ADDITION to 5.6's post-tap verification."

This module is that check, and nothing else.

    capture_profile_identity(frames, identity_band=...) -> ProfileIdentity   # at index build
    compare_profile_identity(frame, identity, identity_band=...) -> IdentityVerdict

A leaf module on the same terms as `segment.py`, `frameshift.py`, `item_index.py`,
`item_crops.py`, `scroll_top.py` and `scroll_step.py`: pure functions over frame BYTES plus
explicit calibration parameters. No device, no I/O, no global state, no driver state.

GEOMETRY IS NOT IDENTITY, AND THAT IS A MEASUREMENT
-----------------------------------------------------
The failure this closes was measured, not feared. Driving profile A's index over profile B's real
frames, `item_nav` returned a tap target for model index 1: every geometric comparison it makes
passed IDENTICALLY (974px card height, 885px heart inset, x=938 on both profiles, because the
modal Hinge card is one photo of one height with the heart at a fixed inset), and its one
remaining defence — the anchor test, which compares where each pass's heart 1 sits — passed by
exactly 0px, 218px apart against a 218px bound on a strict `>`. That test is now `>=`, which
catches that ONE pair by 0px instead of missing it by 0px, and a closer pair is not caught by it
at all. Widening the bound is not the fix and is explicitly ruled out: the bound is
`_TOP_ORIGIN_RESIDUAL_PX`, a measured property of the scroll-top gate, and any value large enough
to separate two arbitrary profiles' first cards is large enough to admit the neighbouring card.

So the answer cannot come from the page's geometry at all. It comes from the one strip of the
screen that says WHO is on it.

THE PRIMITIVE IS THE DRIVER'S OWN, DELIBERATELY REUSED RATHER THAN REPLACED
----------------------------------------------------------------------------
`HINGE_SPEC.identity_band` is the app's sticky per-profile header: the person's NAME, centred,
which "appears here the moment the card is scrolled at all, and stays pixel-identical for the
whole profile". It is already the anchor observe mode decides same-person / new-person on
(`hinge._identity_of`, `hinge._identity_sig`), it is already calibrated on the real device, and
this module reads it through the same `hinge._band` decode at the same `hinge._IDENTITY_DS` grid.
The BOUND is the one thing not inherited, because measuring it against the corpus showed
`change_threshold` does not separate two people — see `_IDENTITY_MATCH_MAX_DIST`.

  [device, 2026-08-10, Pixel 7a 1080x2400, recorded in `HINGE_SPEC.identity_band`'s own comment:
  three frames of one profile at three different scroll offsets measure mean-abs 0.00 between
  them, and 17.95 against the same profile's scroll-top frame. `change_threshold` is 9.0, so the
  separation is unambiguous in both directions.]

  [corpus, `scroll_top.py`'s re-measurement over the gitignored calibration captures: the five
  frames at a genuine scroll top produce a byte-identical band across TWO DIFFERENT PEOPLE (0.000
  pairwise), while all 143 other frames land at 14.391 or further from it. That is the same fact
  from the other side — the strip carries the filter chips at the top and a person below it — and
  it is what `confirm_scroll_top` answers for this module.]

  [corpus, measured for THIS module over the same captures: one profile's settled header is
  **0.000** from itself on every one of 143 refuting frames, and the two profiles' headers are
  **7.287** apart. Those two numbers are the whole gate. The rest of the table, including why the
  bound is not `change_threshold`, is on `_IDENTITY_MATCH_MAX_DIST`.]

Nothing about this scheme is new. What is new is that the fingerprint is now an ATTRIBUTE OF THE
INDEX, captured from the frames the index was built from, so a navigator can ask "is this the
profile my table describes" without holding any driver state and without being trusted to
remember to ask.

WHY THE FINGERPRINT NEEDS BOTH STATES OF THE STRIP AND A SECOND OPINION
-------------------------------------------------------------------------
`capture_profile_identity` does not take the first frame it is given, and each of the three
conditions it does impose is a measurement rather than a precaution. In short: some frame must
CONFIRM the scroll top (proving the rect is the one Hinge draws its own chips over, so a mis-typed
override pointing at static chrome cannot become a fingerprint that matches everybody); a later
frame must REFUTE it (proving the same rect carries something else once scrolled, which is the
person); and that reading must be CORROBORATED by another refuting frame.

The third condition is the one the corpus forced. Taking the first refuting frame picked, on one
of the two real profiles, a TRANSITIONAL frame caught mid-slide-in: the only frame of 112 that
disagreed with the rest, 14.482 from the settled header the other 111 share to 0.000. An index
fingerprinted from it refuses its own profile at every later frame — a guaranteed false stop on a
good read. See `capture_profile_identity` for the full rule and the majority condition.

THE THREE OUTCOMES, AND WHY "CANNOT TELL" IS NOT "SAME PERSON"
----------------------------------------------------------------
`IdentityVerdict.state` is exactly one of:

  * `IDENTITY_MATCH`    — the strip shows the header this index was built from, to within a bound
                          that no measured non-match comes near. Navigation may proceed.
  * `IDENTITY_MISMATCH` — it shows something else. Deliberately ONE state rather than a graded
                          one: a different person, a redrawn header and a paywall are all "not
                          the card this index describes", they are not separable at this grid
                          (7.287, ~6.67 and 4.289 respectively), and the action is identical for
                          all three. The distance is on the verdict for whoever has to diagnose
                          the stop.
  * `IDENTITY_UNKNOWN`  — there is nothing to compare: the screen is at its scroll top, where the
                          strip shows the profile-independent chips row; or no band is declared;
                          or the index carries no fingerprint. `hinge._identity_of`'s "top" and
                          "unknown", which to a caller about to tap a heart are one thing.

`IdentityVerdict.__bool__` RAISES, on `ScrollTopVerdict.__bool__`'s precedent and for its reason:
a frozen dataclass is truthy by default, so `if compare_profile_identity(...):` would read every
outcome — including "cannot tell" — as a match, which is the single most dangerous line anyone
could write against this API. Callers test `.matched`.

THE ENTRY PRECONDITION THIS IMPOSES ON WHOEVER NAVIGATES, STATED PLAINLY
--------------------------------------------------------------------------
Identity is NOT readable at a scroll top. That is not a limitation of this module, it is the
shape of the screen: at the top the strip is Hinge's own chrome, byte-identical across people
(measured 0.000 between two different profiles), so there is nothing there to tell anyone apart.

Navigation must therefore be entered from where the enumeration read leaves the card — SCROLLED,
with the sticky header showing — and not from a scroll top. **That is now true by construction
rather than by a caller remembering it** (doc 5.5's bottom-up addendum, 2026-08-12): navigation
walks UP from where the read ended and no longer rewinds at all, so the entry frame IS a scrolled
one, and the same frame the entry page anchor is measured from is the one this gate reads. The
paragraph that used to stand here warned whoever wired the tap to put the navigation call above
`hinge.like()`'s `_scroll_to_top()`; on the counting-navigation path there is no `_scroll_to_top`
left to be above. The legacy capture-order path still has one, and `_confirm_payload_profile`
sits above it for exactly this reason.

Refusing there is the deliberate answer rather than a gap to paper over: doc 5.6's owner rule is
that we never substitute a different item, and "we could not tell whose card this is" is not a
state in which a tap can be justified.

WHAT THIS DOES NOT DO
-----------------------
  * It does not replace doc 5.6's POST-TAP crop-signature check. That answers a different
    question — "is the sheet that opened the item we chose" — against a reference this module
    does not hold. 5.6's own addendum is explicit that navigation cannot pre-screen identity and
    that the signature check must not be weakened on the assumption that it did. Now that this
    exists, the two are complements: this one rules out the wrong PERSON before any gesture, that
    one rules out the wrong ITEM after the tap.
  * It does not read a name. `hinge._identity_of`'s OCR corroboration is a position-tolerant
    upgrade path that can only ever conclude "same"; it is best-effort, needs tesseract on PATH,
    and giving a safety gate a component that can be absent at runtime would make the gate's
    strictness depend on the host. The pixel band alone is what ships here, and a header that
    moved a few pixels therefore reads as MISMATCH — a false stop, never a false go.
  * It does not invalidate anything. The table's lifetime is one profile and
    `hinge._invalidate_item_index` owns that (doc 5.3). This is the guard for the case where
    invalidation was missed, which is exactly the case that has no other guard before the tap.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .scroll_top import (
    ScrollTopError, band_fingerprint, band_pinned_evidence, confirm_scroll_top,
    fingerprint_distance)


# =====================================================================================
# Calibration constants.
#
# Sources, the same convention as the six leaf modules beside this one:
#   [device] the live Pixel 7a measurements recorded in `HINGE_SPEC.identity_band`'s own comment
#            (2026-08-10) and in `hinge._identity_of`'s docstring.
#   [corpus] `scroll_top.py`'s re-measurement of the gitignored calibration captures through the
#            SHIPPED `hinge._band` decode. Numbers only -- the frames are real people's profiles
#            and never leave ops/calibration/.
#   [doc]    ops/OPENER-REDESIGN.md 5.3 (the stale table), 5.5's 2026-08-12 addendum (the 218-vs-
#            218 cross-profile measurement) and 5.7's final audit (the carried-forward
#            requirement this module answers).
# =====================================================================================

# Downsample grid (w, h) the identity band is reduced to. This is `hinge._IDENTITY_DS` -- the
# grid the driver's OWN identity anchor uses and the one every measurement quoted above was taken
# at. Declared rather than imported for `scroll_top._REFUTE_MIN_DIST`'s reason: `hinge` imports
# this package's leaf modules and not the other way round, so a module-level import back into it
# would be a cycle. `tests/test_item_identity.py` pins the two together so the copy cannot drift.
_IDENTITY_GRID = (64, 16)

# At or below this mean-abs distance (0..255) from the stored fingerprint, the strip IS this
# profile's sticky header and the screen is the profile the index describes. Above it, it is not.
#
# 3.0 is `scroll_top._CONFIRM_MAX_DIST` -- the tolerance the sibling gate already applies to this
# same strip for the same shape of question ("is this strip the thing I believe it is"), so there
# is one number on this band and not two.
#
# IT IS EMPHATICALLY NOT `change_threshold`, AND THAT IS A MEASUREMENT RATHER THAN A PREFERENCE.
# An earlier revision of this module used 9.0, on the reasoning that `hinge._identity_of` decides
# same-person / new-person on it. Driving the two calibration profiles against each other found
# 9.0 does not separate them: their settled headers are **7.287** apart at this grid, so a 9.0
# bound calls two different people the same person, and the whole gate would pass for the exact
# case it exists to refuse. (That is a real property of `_identity_of` too, recorded here because
# it was found here. It is safe THERE and not here: that method's errors are deliberately
# asymmetric -- a false "same" costs at most a missed pass, and the deck-ready, settle and
# content checks downstream still have to agree -- whereas a false "same" HERE is a comment
# attached to the wrong person's card.)
#
# [corpus, all four gitignored calibration captures over two profiles, through the shipped
# `hinge._band` decode at this grid. The same profile's settled header measures **0.000** against
# itself on every one of 143 refuting frames -- 111 of them spanning ~95s of one hand-scrolled
# read -- so there is no measured nonzero same-profile distance at all, and the bound is set
# entirely by what must be kept OUT:
#     0.000   the same profile, any scroll offset, all 143 frames, four captures, two profiles
#     4.289   the out-of-likes paywall's band against profile B's header (the nearest non-match)
#     7.287   profile A's settled header against profile B's, every frame, both directions
#     9.496   that same paywall against profile A's header
#    14.482   profile A's own header MID-SLIDE-IN, i.e. the transitional frame
#    15.7..19.5  the filter-chips row against either profile's header
# 3.0 sits above zero and below every one of those, with 1.29 grey levels of margin against the
# nearest.]
#
# THAT MARGIN DOES NOT SURVIVE A THIRD PROFILE, AND THE NUMBER ABOVE IS THEREFORE KNOWN TO BE TOO
# WIDE. The table's "7.287 between two different people" is ONE PAIR. Re-measured 2026-08-12 over
# six real profiles -- the six `observe_like_anchor` sheets in the local (gitignored) debug
# directory, each definitely a different person (a like advances the deck) and each verified to
# carry a settled header by reading 0.000 against that profile's own neighbouring scrolled frames:
#
#     [corpus, 6 profiles, 15 pairs, through the shipped `hinge._band` decode at this grid
#      2.565  the CLOSEST two different people          <-- inside this bound: the gate MATCHES
#      3.752  the next closest                          <-- clears it by 25%
#      6.891  the furthest apart
#      0.000  every sheet against its own profile's scrolled frames, all six]
#
# So one pair in fifteen is a measured FALSE ACCEPT: the gate would call two different people the
# same person for exactly the case it exists to refuse. The cause is what this module's own
# docstring already warned about -- the strip is a short centred first name on flat chrome, and at
# 64x16 two names of similar length and weight barely differ. A finer grid does not rescue it
# (128x32 and 256x64 move the closest pair to 2.683 and 2.681, i.e. nowhere).
#
# THE BOUND IS DELIBERATELY LEFT AT 3.0 RATHER THAN NARROWED HERE. Same-profile distance is
# identically 0.000 on every frame ever measured, so any bound in (0, 2.565) is correct on all
# available data and there is no principled way to choose among them offline -- and the tighter it
# goes the more a redrawn header (measured at 6.67 for a 4px shift) turns into a stop. That is a
# calibration decision with the owner and a device in front of them, which is what this comment
# already said about re-measuring, and picking a number to fit six profiles is the fitted constant
# this repo does not ship.
#
# WHAT THIS MEANS FOR CALLERS, AND IT IS NOT OPTIONAL: the legacy 3.0 default is diagnostic only;
# it cannot license a targeted action. The production Hinge path supplies an operator-calibrated
# ceiling strictly below the known 2.565 collision, rechecks identity on the opened sheet, and
# also requires the separately calibrated absolute item-distance ceiling. Missing evidence blocks
# AUTO before a gesture and OBSERVE before suggestion generation. Other callers using this
# module's default still inherit the measured false-accept caveat and must not build on MATCH alone.
#
# CONSEQUENCE, STATED RATHER THAN DISCOVERED LATER: at this grid a redrawn header cannot be
# tolerated. [corpus: `scroll_top.py` measured a 4px vertical layout shift on this band at 6.67
# grey levels at 64x16 -- larger than the 4.289 that separates a paywall from a person -- so no
# bound can accept "the same chrome, moved" while rejecting a different screen.] The gate is
# therefore deliberately intolerant of drift: a header redrawn a few pixels lower is a REFUSAL,
# which is a false stop and never a false go, and doc 5.6's owner rule is that we stop rather
# than substitute. Re-measuring is a calibration task with the owner present, not a reason to
# widen this.
_IDENTITY_MATCH_MAX_DIST = 3.0


# --- the three outcomes ---------------------------------------------------------------
IDENTITY_MATCH = "identity_match"            # the screen is the profile the index describes
IDENTITY_MISMATCH = "identity_mismatch"      # it is not the header this index was built from
IDENTITY_UNKNOWN = "identity_unknown"        # cannot tell -- and never to be read as a match


class IdentityError(RuntimeError):
    """The gate could not LOOK: undecodable frame bytes, PIL/numpy missing, or a stored
    fingerprint whose band/grid contradicts the one it is being compared against.

    Distinct from `IDENTITY_UNKNOWN`, which is a verdict reached by looking. The separation is
    `segment.SegmentationError`'s and `scroll_top.ScrollTopError`'s, for their reason: "we could
    not look" must never be reachable by a route that also produces a verdict, or a broken
    install degrades into a stream of plausible-looking answers.

    A NAVIGATOR should still treat it as a refusal — see `item_nav`, which catches it and stops
    with `NAV_IDENTITY_UNCONFIRMED`, keeping the distinction in the message rather than in the
    control flow. The two call for the same action; only the diagnosis differs.
    """


@dataclass(frozen=True)
class ProfileIdentity:
    """Who the profile an `ItemIndex` describes is, as pixels, or why that is not known.

    `fingerprint` is the sticky-header strip as `grid` grey levels in row-major order — the same
    array `hinge._identity_sig` holds, through the same decode, flattened to plain ints so it can
    be compared, logged and stored anywhere without numpy. It is None when the capture could not
    establish one, and `reason` then says which of the several ways it could not.

    It is NOT a picture of anyone: 1024 block averages of a strip that holds a first name on flat
    app chrome. The rest is provenance, and it is kept because this is a safety gate whose stops
    an operator has to be able to diagnose: `frame_index` is the frame it was read from,
    `scroll_top_distance` how far that frame's band sat from the profile-independent chips row
    (the evidence that the strip really had switched to a per-profile header), `agreeing_frames`
    how many frames of the capture corroborated it, and `outlier_frames` /
    `outlier_max_distance` how many did not and by how much.
    """
    fingerprint: tuple[int, ...] | None
    band: tuple[float, float, float, float] | None
    grid: tuple[int, int]
    frame_index: int | None
    scroll_top_distance: float | None
    reason: str
    agreeing_frames: int = 0
    outlier_frames: int = 0
    outlier_max_distance: float | None = None

    @property
    def known(self) -> bool:
        """True when this index can be checked against a screen at all. False is a hard stop for
        anything that navigates: doc 5.3's "treat a missing table as a hard stop" applies to a
        table that cannot prove whose profile it is just as much as to one that is absent."""
        return self.fingerprint is not None


@dataclass(frozen=True)
class IdentityVerdict:
    """One frame's answer to "is this the profile that index came from", with its evidence.

    `distance` is the mean-abs grey-level distance (0..255) between the frame's identity band and
    the stored fingerprint — the same metric `hinge._band_dist` uses on the same band at the same
    grid, so it is directly comparable to every number in `HINGE_SPEC.identity_band`'s comment and
    to `change_threshold` (which is NOT the bound here: see `_IDENTITY_MATCH_MAX_DIST`). None when
    no comparison happened at all. `match_max` is the bound it was judged against, carried so a
    stop record reads as a comparison rather than as a bare number.

    `reason` is always populated, including on a match, so a debug record or a hub stop line can
    quote why without re-deriving anything.
    """
    state: str
    distance: float | None
    reason: str
    identity: ProfileIdentity | None
    match_max: float

    @property
    def matched(self) -> bool:
        """True ONLY for `IDENTITY_MATCH`. The only property a gate may proceed on."""
        return self.state == IDENTITY_MATCH

    @property
    def mismatched(self) -> bool:
        return self.state == IDENTITY_MISMATCH

    @property
    def unknown(self) -> bool:
        return self.state == IDENTITY_UNKNOWN

    def __bool__(self) -> bool:
        # See the module docstring. `if compare_profile_identity(...):` would read "cannot tell"
        # and "that is a different person" as "same person", in the most natural-looking way
        # possible. numpy's precedent, and `ScrollTopVerdict.__bool__`'s: an ambiguous truth
        # value is a TypeError, not a guess.
        raise TypeError(
            "IdentityVerdict has no truth value: 'cannot tell' is not 'same profile'. Test "
            "`.matched` explicitly. This verdict was "
            f"{self.state!r}: {self.reason}")


def _unknown(reason: str, *, band=None, grid=_IDENTITY_GRID) -> ProfileIdentity:
    return ProfileIdentity(fingerprint=None, band=band, grid=grid, frame_index=None,
                           scroll_top_distance=None, reason=reason)


def capture_profile_identity(frames: Sequence[bytes], *,
                             identity_band: tuple[float, float, float, float] | None,
                             page_offsets: Sequence[int | None] | None = None,
                             grid: tuple[int, int] = _IDENTITY_GRID,
                             match_max_dist: float = _IDENTITY_MATCH_MAX_DIST) -> ProfileIdentity:
    """The sticky-header fingerprint of the profile `frames` were captured from.

    `frames` are the capture's frames in order — the same ones the index is built from, so the
    fingerprint is provably about the page that was indexed rather than about whatever the driver
    happened to be looking at when someone remembered to ask.

    `identity_band` is the normalised `(x0, y0, x1, y1)` rect of the filter-chips / sticky-header
    strip. Pass the driver's `self.identity_band` (the config-merged value), NOT
    `spec.identity_band`, or an operator override is silently ignored — the same rule
    `segment_frame` states for `content_band`. `None` is accepted and yields an unknown identity:
    an app that declares no band is one this gate cannot judge, and saying so is the honest
    answer rather than an error, because the field is optional on `AndroidAppSpec` by design.

    THE RULE HAS THREE PARTS, AND EVERY ONE OF THEM IS A MEASUREMENT RATHER THAN A PRECAUTION.

    1. Some frame must CONFIRM the scroll top. That proves `identity_band` really is the strip
       Hinge draws its own filter chips over — i.e. the calibrated rect, and not an operator's
       mis-typed override pointing at static status-bar chrome. A fingerprint taken off static
       chrome would be identical on every profile and would pass this gate for everybody, which
       is worse than no check at all, and this is the only structure that rules it out without a
       second calibration.

    2. Some LATER frame must REFUTE it. That proves the same rect shows something else once the
       card is scrolled, which is the sticky per-profile header — the thing that differs between
       people. Confirmation alone would only prove the rect holds chrome.

    3. The refuting frame must be CORROBORATED by another refuting frame that agrees with it to
       within `match_max_dist`. This is the part the calibration corpus forced, and skipping it
       shipped a bug: taking the FIRST refuting frame picked, on one of the two real profiles, a
       TRANSITIONAL frame caught mid-slide-in — the only frame of 112 that disagreed with all the
       others, sitting 14.482 from the settled header the remaining 111 frames share to 0.000.
       An index fingerprinted from it would have refused its own profile at every later frame:
       a guaranteed false stop on a perfectly good read. Corroboration is the same rule
       `item_index` applies to a block's extent — "two frames that both observed one fixed
       quantity are two measurements of it, and disagreement is a failure rather than an average"
       — and it is exactly the device measurement that licenses it: the header "stays
       pixel-identical for the whole profile".

    The corroborated group must also be a strict MAJORITY of the refuting frames. A capture whose
    header readings split evenly is a capture that may span two people, and picking the larger
    half of that would fingerprint whichever one happened to have more frames. (Upstream,
    `hinge._capture_current`'s own split check and the index's correspondence chain both refuse
    such a capture first; this is the belt to those braces, and it is cheap.)

    `page_offsets` is OPTIONAL and changes no outcome — only a DIAGNOSIS, and it is the whole
    reason it exists. When every frame of a capture reads as the filter-chips row there are two
    completely different causes with opposite fixes, and this function used to name only one of
    them: either the capture genuinely never scrolled far enough to reveal the header (re-run it),
    or this capture's current toolbar state pins the chips row to the screen (Hinge 10.1.0's
    expanded header state, measured 2026-08-28 — see
    `scroll_top.band_pinned_evidence`, which is what these offsets are handed to). Blaming the
    capture in the second case sends an operator to re-run it forever, and this repo's standing
    rule is that guidance derives from its precondition. Pass the same per-frame offsets the index
    build already measured (`None` for a frame past a broken correspondence chain); omit it and
    the diagnosis stays the pre-existing one, which is correct whenever the pinning cannot be
    proven anyway. The fail-closed behaviour is IDENTICAL either way: no fingerprint, `.known`
    False, navigation refuses.

    Never raises for a frame it cannot read: an undecodable frame is skipped with the reason kept,
    because this runs inside an index build whose own answer is a RESULT, and a capture with one
    bad frame must not turn into an exception out of a function whose job is to describe it. It
    DOES raise for a `page_offsets` of the wrong length, which is not a frame problem but a caller
    pairing one frame's band with another frame's scroll position.
    Returns a `ProfileIdentity` whose `.known` is False in every case it could not establish one.
    """
    if page_offsets is not None and len(page_offsets) != len(frames):
        raise IdentityError(
            f"{len(frames)} frame(s) against {len(page_offsets)} page offset(s): these describe "
            "different captures, and pairing them would attribute one frame's band to another "
            "frame's scroll position")
    if identity_band is None:
        return _unknown(
            "no identity_band is declared for this app, so there is no strip that carries the "
            "person's sticky header and no fingerprint can be taken. Navigation against this "
            "index cannot be checked for identity and must refuse", grid=grid)

    band = tuple(identity_band)
    if not frames:
        return _unknown("this capture has no frames, so there was nothing to fingerprint",
                        band=band, grid=grid)

    confirmed_at: int | None = None
    unreadable: list[str] = []
    seen: list[tuple[int, tuple[int, ...], float | None]] = []
    for i, frame in enumerate(frames):
        try:
            verdict = confirm_scroll_top(frame, identity_band=identity_band)
            if verdict.confirmed:
                if confirmed_at is None:
                    confirmed_at = i
                continue
            if not verdict.refuted or confirmed_at is None:
                # UNKNOWN (the scroll-top gate's own dead zone) is not evidence of a header, and a
                # refuting frame BEFORE any confirmation proves nothing about the rect — static
                # chrome refutes on every frame of every profile. Keep looking.
                continue
            seen.append((i, band_fingerprint(frame, identity_band=identity_band, grid=grid),
                         verdict.distance))
        except ScrollTopError as exc:
            unreadable.append(f"frame {i} ({exc})")

    trailer = (f" ({len(unreadable)} frame(s) could not be read: {'; '.join(unreadable)})"
               if unreadable else "")
    if confirmed_at is None:
        return _unknown(
            f"no frame of this capture showed the profile-independent filter-chips row in "
            f"{band}, so nothing proves that rect is the strip Hinge draws its sticky header "
            f"over. A fingerprint taken off some other rect could be identical on every profile, "
            f"which would pass an identity check for everybody{trailer}", band=band, grid=grid)
    if not seen:
        # TWO causes, opposite fixes, and the reason must derive from which one this capture has.
        # `band_pinned_evidence` is only consulted here, where the alternative is to blame a
        # capture that may be blameless; everywhere else in this function the chips row reading as
        # the chips row is ordinary and expected.
        pinned = (band_pinned_evidence(frames, identity_band=identity_band,
                                       page_offsets=page_offsets)
                  if page_offsets is not None else None)
        if pinned is not None and pinned.pinned:
            return _unknown(
                f"the identity band {band} is PINNED to the screen in this capture's current "
                f"toolbar state, so the identity check is structurally unavailable for this "
                f"capture rather than merely unlucky: "
                f"{pinned.reason}. Every frame therefore reads as the filter-chips row no matter "
                f"how far it scrolled, the sticky per-profile header never occupies this rect, "
                f"and the observed frames cannot be fingerprinted from it. Do not target or "
                f"label from it. A fresh capture "
                f"may be considered only if it independently proves a usable scroll-top and "
                f"profile-identity signal; otherwise the rect and its signals need "
                f"recalibration with the owner and a device{trailer}",
                band=band, grid=grid)
        return _unknown(
            f"every frame of this capture reads as the filter-chips row in {band}, so the sticky "
            f"per-profile header never appeared and there is nothing that distinguishes this "
            f"person from any other. A capture that never scrolled far enough to reveal the "
            f"header cannot be checked for identity{trailer}", band=band, grid=grid)

    # Group the header readings by mutual agreement, earliest-first, and take the largest group.
    groups: list[list[int]] = []
    for position, (_i, fingerprint, _d) in enumerate(seen):
        for group in groups:
            if fingerprint_distance(fingerprint, seen[group[0]][1]) <= match_max_dist:
                group.append(position)
                break
        else:
            groups.append([position])
    best = max(groups, key=lambda g: (len(g), -g[0]))
    outliers = [p for group in groups if group is not best for p in group]
    outlier_max = max((fingerprint_distance(seen[p][1], seen[best[0]][1]) for p in outliers),
                      default=None)

    if len(best) < 2:
        return _unknown(
            f"no two frames of this capture agree on what the sticky header looks like: "
            f"{len(seen)} frame(s) showed one, and no two of them are within {match_max_dist} "
            f"grey levels of each other. One uncorroborated reading is how a frame caught "
            f"mid-slide-in becomes the fingerprint, which then refuses its own profile at every "
            f"later frame{trailer}", band=band, grid=grid)
    if len(best) * 2 <= len(seen):
        return _unknown(
            f"this capture's header readings do not agree on one person: the largest group of "
            f"{len(best)} agreeing frame(s) is not a majority of the {len(seen)} that showed a "
            f"header, and the nearest disagreeing reading is {outlier_max:.3f} grey levels away. "
            f"A capture that may span two profiles cannot be fingerprinted as one{trailer}",
            band=band, grid=grid)

    index, fingerprint, top_distance = seen[best[0]]
    note = (f"; {len(outliers)} frame(s) disagreed, the furthest by {outlier_max:.3f} grey "
            "levels, which is what a frame caught mid-slide-in looks like" if outliers else "")
    return ProfileIdentity(
        fingerprint=fingerprint, band=band, grid=grid, frame_index=index,
        scroll_top_distance=top_distance, agreeing_frames=len(best),
        outlier_frames=len(outliers), outlier_max_distance=outlier_max,
        reason=(f"frame {index} of this capture shows the app's sticky per-profile header in the "
                f"identity band — {top_distance:.3f} grey levels from the profile-independent "
                f"filter-chips row frame {confirmed_at} confirmed — and {len(best)} of the "
                f"{len(seen)} frames that showed a header agree with it to within "
                f"{match_max_dist}{note}{trailer}"))


def compare_profile_identity(frame: bytes, identity: ProfileIdentity | None, *,
                             identity_band: tuple[float, float, float, float] | None,
                             match_max_dist: float = _IDENTITY_MATCH_MAX_DIST) -> IdentityVerdict:
    """Is `frame` showing the profile `identity` was captured from? Match, mismatch, or neither.

    `identity` is `ItemIndex.identity`. `identity_band` is the driver's config-merged rect, and it
    must be the SAME rect the fingerprint was taken through — comparing two different crops of a
    screen is not a distance between anything, so a disagreement is an `IdentityError` rather than
    a quiet verdict.

    THE ORDER OF THE CHECKS IS THE CONSERVATIVE ONE. "Is the screen at its scroll top" is asked
    BEFORE the distance, because at the top the strip is app chrome that is byte-identical across
    people (measured 0.000 between two different profiles): a small distance there would be a
    property of the chrome, not evidence about anybody. A screen at the top is UNKNOWN even if
    its band happens to sit close to the stored header.

    Raises `IdentityError` when it could not look at all — including the `ScrollTopError` the
    shared decode raises on undecodable bytes, re-wrapped so this function has exactly one
    exception type and a caller does not have to know which sibling module the read went through.
    Returns a verdict otherwise; the caller proceeds only on `.matched`.
    """
    if identity is None or not identity.known:
        return IdentityVerdict(
            state=IDENTITY_UNKNOWN, distance=None, identity=identity,
            match_max=match_max_dist,
            reason=("this item index carries no identity fingerprint, so there is nothing to "
                    "check the screen against: "
                    + (identity.reason if identity is not None else "no identity was captured")))
    if identity_band is None:
        return IdentityVerdict(
            state=IDENTITY_UNKNOWN, distance=None, identity=identity,
            match_max=match_max_dist,
            reason=("no identity_band is declared for this app now, though this index carries a "
                    f"fingerprint taken through {identity.band} — the strip that would answer "
                    "cannot be read"))
    if tuple(identity_band) != identity.band:
        raise IdentityError(
            f"this index's identity fingerprint was taken through band {identity.band} but the "
            f"driver now declares {tuple(identity_band)}. Those are different crops of the "
            "screen and no distance between them measures anything — re-index the profile "
            "rather than comparing across a changed calibration")

    try:
        seen = band_fingerprint(frame, identity_band=identity_band, grid=identity.grid)
        top = confirm_scroll_top(frame, identity_band=identity_band)
    except ScrollTopError as exc:
        raise IdentityError(
            f"the identity band {tuple(identity_band)} could not be read from this frame "
            f"({exc}), so whose profile is on screen cannot be established") from exc

    if top.confirmed:
        return IdentityVerdict(
            state=IDENTITY_UNKNOWN, distance=None, identity=identity,
            match_max=match_max_dist,
            reason=("the screen is at a card's scroll top, where the identity band shows Hinge's "
                    f"profile-independent filter-chips row ({top.distance:.3f} from it). That "
                    "strip is byte-identical across two different people in the calibration "
                    "corpus, so identity is simply not visible here — this is 'cannot tell', "
                    "never 'same profile'. Navigation must be entered from where the read left "
                    "the card, scrolled, with the sticky header showing"))

    dist = fingerprint_distance(seen, identity.fingerprint)
    if dist <= match_max_dist:
        return IdentityVerdict(
            state=IDENTITY_MATCH, distance=dist, identity=identity, match_max=match_max_dist,
            reason=(f"the sticky per-profile header matches the one this index was built from at "
                    f"{dist:.3f} <= {match_max_dist} (frame {identity.frame_index} of that "
                    f"capture, agreed on by {identity.agreeing_frames} of its frames); the "
                    "header stays pixel-identical for a whole profile and changes with the "
                    "person"))
    return IdentityVerdict(
        state=IDENTITY_MISMATCH, distance=dist, identity=identity, match_max=match_max_dist,
        reason=(f"the identity band is {dist:.3f} grey levels from the sticky header this index "
                f"was built from, past the {match_max_dist} bound. Two frames of ONE profile "
                "measured exactly 0.000 apart on all 143 frames of the calibration corpus, so "
                "anything above the bound is a different draw of that strip: a different person "
                "(measured 7.287 between the two corpus profiles), a header redrawn a few pixels "
                "over, or another screen entirely (the paywall measured 4.289). None of the "
                "three is a card this index may be counted against, and geometry cannot tell any "
                "of them apart — Hinge's cards are stereotyped, and a foreign profile's item 1 "
                "measured the same card height, the same heart inset and the same x as the "
                "indexed one"))
