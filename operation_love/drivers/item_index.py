"""ONE canonical list of a profile's items, assembled from an ordered run of scroll frames.

`segment.py` turns ONE frame into blocks. `frameshift.py` says how far the content moved
between two frames. Neither can answer the question ops/OPENER-REDESIGN.md 5.3 actually asks:
"what are this profile's items, in order, and which one is item 3?" That needs every frame's
blocks folded into a single PAGE coordinate space, the same physical card recognised across the
six frames it appears in, and two index spaces laid over the result.

    build_item_index(frames, ...) -> ItemIndex

Deliberately a leaf module, on the same terms as its two dependencies: pure functions over frame
BYTES plus explicit calibration parameters, no device, no I/O, no global state, no `HingeDriver`
import.

THE TWO INDEX SPACES, AND WHY THEY ARE NOT THE SAME SPACE
----------------------------------------------------------
Doc 5.3, stated as a rule: **index space belongs to the driver, selectability is policy.**

  * the HEART ORDINAL is what navigation counts. Scroll to top, step forward counting distinct
    hearts, tap the k-th (doc 5.5). It must therefore include EVERY heart on the page, in page
    order, with no gaps — including hearts on blocks the model is not allowed to choose, and
    including any future block type nobody has thought of yet. A heart that exists and is not
    counted here is an off-by-N in the tap coordinate.
  * the MODEL INDEX is a dense 1..N over the SELECTABLE blocks only — heart-bearing, and with
    an extent this module actually observed, so it can be cropped and later verified (5.6).
    The model never sees heart ordinals. Item 3 in the model's list may be the 4th heart on the
    page.

`ItemIndex.translation` is the table between them, and it is the reason a context block (the
vitals block: heartless, freely referenced, never selectable) can occupy a page position without
consuming a model index, while a heart-bearing block that we could not fully bound still consumes
a heart ordinal without ever being offered as a choice.

THE FAILURE THIS MODULE EXISTS TO MAKE IMPOSSIBLE
--------------------------------------------------
Doc 5.10 records the exact bug, measured on a real capture: chaining frame-to-frame heart matches
across a whole scroll "recovered 9 real items **and fabricated a spurious 10th**, created
entirely by one large-jump tracking failure". 5.10.1 then found the cause — the scroll step
aliasing against the card spacing — and proved no better estimator fixes it.

So a fabricated item is not something to detect afterwards. It is prevented by construction, in
three places:

  1. **A broken correspondence chain yields NO items at all.** Every consecutive pair must
     produce a trustworthy `frameshift.estimate_shift`. One pair that does not — low confidence,
     no consensus, or a shift past the trust window — and there is no page coordinate space to
     express the later frames in, so `blocks` comes back EMPTY with the reason in `failures`.
     Not a best-effort prefix, and above all not an assumed step size: assuming the step is
     precisely what fabricated the phantom.
  2. **The same physical card seen in six frames is ONE block.** Sightings are folded by
     absolute page position, not counted.
  3. **Two blocks closer than a gutter are reported, not accepted.** The one way folding by
     overlap could still invent an item is by SPLITTING one card into two non-overlapping
     fragments. Real Hinge cards are separated by a 47..58px gutter and nothing smaller
     (segment.py's `_GUTTER_PX` +- `_GUTTER_TOLERANCE_PX`, measured exactly 53 on all 206
     card-to-card gutters in the corpus), so two resolved blocks nearer than that contradict the
     layout and are a failure. See `_MIN_ITEM_GAP_PX`.

WHY DISAGREEMENT IS AN ERROR AND NEVER AN AVERAGE
---------------------------------------------------
Two frames that both observed BOTH edges of the same card are two independent measurements of
one fixed quantity. If they disagree — 974px against 853px at the same page position — exactly
one of them is right, and the difference is a segmentation or a tracking fault. Averaging them
produces a number neither frame ever saw, which would then be cropped, sent to the model, and
stored as the post-tap verification reference (5.6), where it would fail against whichever card
is really there. So the resolved extent is always an extent some single frame actually OBSERVED
(see `_resolve_group`), and any other complete sighting that disagrees with it by more than
`_EXTENT_TOLERANCE_PX` is a `failures` entry.

The same rule applies to hearts: two heart positions more than a tolerance apart at one page
position mean the block holds two hearts, which segment.py already calls a segmentation failure
within a frame, and which is equally a failure across frames.

WHAT THE THREE CALIBRATION CAPTURES MEASURE (validated offline, 2026-08-12)
----------------------------------------------------------------------------
Run over the same gitignored corpus segment.py and frameshift.py were calibrated on. Geometry
and counts only. The headline is the last line of each block.

  * `botscroll_20260811T231516Z` — profile B, all 24 frames, bot-scrolled at the
    read_scroll_frac 0.16 cadence doc 5.10.1 recommends, `at_scroll_top=True`. USABLE, with
    **9 selectable items and no phantom** — the same 9 that doc 5.10.1's two independent
    captures agree on. 11 blocks in total: one `ITEM_LEADING_CHROME`, nine selectable and one
    1144px `ITEM_CONTEXT`, so the translation table comes out 1..9 (the context block consumes
    neither number). ZERO partial blocks — every one of the nine was bounded end to end in at
    least one frame, and seven of them in two or more, which is what makes them croppable. Item
    1 resolves to page rows 479..1453, exactly the extent segment.py's own docstring records for
    that profile's first card. `reached_end` True with a 189px tail, `truncated` False.
  * the SAME capture with `at_scroll_top=False`. UNUSABLE, on one failure: the 42px leading
    block, which is Hinge's header chrome, "was never observed complete and shows no like heart".
    That is the `at_scroll_top` assertion doing real work rather than decorating the signature —
    without the caller's top confirmation the module will not assume that block is chrome, and
    it will not quietly number around it either.
  * `botscroll_20260811T225314Z` — the ALIASING cadence (production's 0.55), the capture doc
    5.10.1 measured "a phantom item fabricated" on. UNUSABLE at the very first pair, which
    reports "content moved +1299px, beyond the 900px window". **Zero blocks, zero items, zero
    phantoms.** This is the regression this module exists for, and it is refused by construction
    rather than detected.
  * `scroll_20260811T211209Z` — profile A, 115 frames, hand-scrolled at 0..787px per step.
    UNUSABLE, refused at the pair frameshift also refuses (frames 56/57, "no trustworthy shift"),
    which is the start of the near-static run over an ANIMATED card. The 56 pairs before it chain
    cleanly to page row 10027. An enumeration pass at uncontrolled human cadence is outside what
    this can index, and it says so instead of indexing it badly.

WHAT IS UNKNOWABLE, AND HOW IT IS REPORTED RATHER THAN GUESSED
----------------------------------------------------------------
A block whose extent was never fully observed is not a crop candidate, and it may also be hiding
a heart in the rows that were never inside the analysed band. The second half of that is the
dangerous half, because a hidden heart shifts every heart ordinal BELOW it by one — silently.

So the rule is positional and precise: a heartless block that was never observed complete is
tolerated when nothing heart-bearing sits below it (the truncated tail of a capture, which costs
only coverage), and is a `failures` entry when something does (the middle of a page, where it
would corrupt the count). See `_UNCERTAIN_HEART_NOTE` at its use site.

That rule has one systematic exception, and it is why `at_scroll_top` is a required argument:
Hinge draws a filter-chips header and a name row ABOVE item 1, and on every scroll-top frame in
the calibration corpus that chrome came back as a PARTIAL block — "neither of its ends is gutter-
or corner-bounded", segment.py's own docstring notes, adding that "a caller that has confirmed
scroll-top should ignore blocks above the topmost card corner rather than trust that they
self-exclude". This module is that caller. `at_scroll_top=True` is the caller stating it has
made doc 5.5's affirmative top confirmation (the filter-chips signal in `identity_band`, which
5.5 requires before counting anyway), and it is what licenses classing a leading heartless,
never-complete block as `ITEM_LEADING_CHROME` instead of treating it as an item that might be
hiding heart #1.

A false assertion there is the most dangerous input this module takes, because a capture that
began mid-profile indexes perfectly well — it just numbers from the wrong place. `at_scroll_top`
is therefore no longer taken entirely on trust: `_scroll_top_evidence` requires that some frame
saw page background above the capture's topmost block, which a genuine scroll top always has
(Hinge draws its header below the analysed band's first row) and a band edge slicing a card
never does. It is a partial guard by construction and says so; doc 5.5's filter-chips check is
still the gate.

WHAT THIS MODULE DOES NOT DECIDE, AND WHICH LAYER HAS TO
----------------------------------------------------------
  * The SCROLL. It is handed frames; it does not capture them, does not size a step, and does
    not know when to stop. Doc 5.10.1's rule — step at most about a third of locally measured
    card spacing — is the closed loop's, and this module's `blocks` are the geometry it sizes
    against.
  * WHICH context block is which. `ITEM_CONTEXT` is one bucket, exactly as `BLOCK_CONTEXT` is:
    doc 2.4 wants the endorsement block excluded rather than sent, and no geometry here
    separates it from the vitals block.
  * The CROPS. Every block carries, per sighting, the rows it occupied in that specific frame
    (`BlockObservation.frame_y0/frame_y1`) and whether that sighting was complete, which is
    exactly what a crop pass needs to pick a frame and slice it. Producing the image is not done
    here.
  * SCREEN IDENTITY. `usable is True` is not "this is a Hinge profile" — it inherits that
    caveat wholesale from segment.py, where the out-of-likes paywall returns a clean result.
    Settle `_identity_of`/`_screen_is` BEFORE capturing.
  * Whether a card is ANIMATED. Two screencaps of the same position over animated content can
    differ. Frameshift still refuses ordinary measurements below its three-strip quorum. This
    layer can repair one narrowly bounded two- or three-pair animation run only when an exact
    two-strip alternative is independently fixed by several observed card/heart landmarks.  In
    the narrower one-strip case -- whether it is the only match or a dissenting alternative to
    a failed two-strip candidate -- all three landmark kinds (a card top, a card bottom and a
    heart) must fix it exactly and an adjacent pair must independently form part of the same
    repair run.  The complete page rebuild must then be contradiction-free.  One still narrower
    three-pair transition may begin with two NCC strips
    plus one exact shared gutter (the old heart has left before the new one arrives), but only
    when the following pair is a measured-majority override and the last is the full-landmark
    one-strip case.  A final saved shape may extend one ordinary run to four pairs only when its
    measured tail is corrected by one unique exact top/bottom/heart answer in the separately
    bounded 13..14px structural-tail window.  One final five-pair refusal island is recoverable
    only when ordinary measured pairs bracket it, exact three-or-more-strip/full-landmark
    candidates close both ends, and the fixed 3+/2/2/1/3+ evidence grammar plus complete page
    rebuild succeeds.  An app-confirmed animation marker may also authorize one bounded measured-
    bridge window connecting otherwise independent two-strip/full-layout repairs when every
    intervening pair and both outer anchors retain raw NCC quorum plus exact multi-kind geometry.
    Within that window only, a raw measured bridge may project by at most the existing fold
    tolerance onto one unique exact top/bottom/heart delta with an exact NCC witness; this treats
    stable samples inside one confirmed live-video span as corroboration rather than incorrectly
    splitting the span into separate repair runs.
    Every other animation shape remains a broken-chain refusal, never a guessed answer.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

from .frameshift import (
    _AGREEMENT_TOLERANCE_PX, SHIFT_MEASURED, SHIFT_NO_CONSENSUS, SHIFT_NO_EVIDENCE,
    STRIP_MATCHED, ShiftEstimate, estimate_shift,
)
from .item_identity import ProfileIdentity, capture_profile_identity
# `_GUTTER_PX` / `_GUTTER_TOLERANCE_PX` are imported rather than re-declared, on frameshift.py's
# precedent: that window is already measured-and-cited in segment.py, it is what CUT the blocks
# this module folds, and a second copy would be free to drift away from it.
from .segment import (
    _GUTTER_PX, _GUTTER_TOLERANCE_PX, _HEART_SEPARATED_NEAR_GUTTER_PX,
    EDGE_CARD_CORNER, RUN_TOO_LONG, FrameSegmentation, segment_frame)
# `_MAX_STEP_PX` is borrowed for the same reason: the ceiling on ONE read gesture is measured and
# cited in scroll_step.py, and it is the gesture that produced the frames folded here, so a
# second copy of it would be free to drift away from the step the reader actually makes.
from .scroll_step import MAX_SEGMENTATION_FALLBACK_FRAMES, _MAX_STEP_PX


# =====================================================================================
# Calibration constants.
#
# Sources, same convention as segment.py and frameshift.py:
#   [doc]    ops/OPENER-REDESIGN.md 5.10 / 5.10.1.
#   [corpus] the measurements those two modules' own docstrings carry, taken over the gitignored
#            calibration captures. Geometry and counts only — the frames are real people's
#            profiles and never leave ops/calibration/.
# =====================================================================================

# How far two sightings of the SAME page position may disagree — about an edge, or about a
# heart's row — before the disagreement is reported instead of absorbed.
#
# It is measurement slack on a CHAIN of shift estimates, which is why it is larger than
# frameshift's own `_AGREEMENT_TOLERANCE_PX` (3, for a single pair):
#   [doc 5.10] frame-to-frame heart matching under an independently estimated scroll offset ran
#              at a residual std of 2.4px, so 8 is ~3.3 sigma of the only per-step residual this
#              chain has ever been measured to carry;
#   [corpus]   the shift estimator itself contributes ~nothing to that: within a pair its
#              agreeing strips are identical TO THE PIXEL (spread 0 on all 23 bot-scrolled
#              pairs), and its 23 per-pair answers land on 361..364 for a step whose ledger says
#              362.99 — i.e. the spread is the phone really moving a different distance, which
#              is a correct measurement rather than an error to accumulate.
# The ceiling is what matters more than the floor: it must stay far below HALF of
# `_MIN_ITEM_GAP_PX` (47/2 = 23), or two genuinely distinct blocks could be reconciled into one.
# 8 is under half of that.
#
# Exceeding it is never absorbed silently in either direction — too-large a disagreement becomes
# a `failures` entry, and a fragment that drifts out of overlap with its own block becomes the
# `_MIN_ITEM_GAP_PX` failure below. Both are loud.
_EXTENT_TOLERANCE_PX = 8

# A direct bridge over one omitted transient frame spans TWO ordinary capture steps.  It still
# has to be independently measured and the complete reduced index still has to validate, but its
# accumulated alignment slack can be one pixel wider than a normal adjacent-frame chain:
#   [live capture 2026-08-15] both pairs around one frame fell below frameshift's three-witness
#              floor, while the direct bridge had 4/4 agreement.  That bridge placed the shared
#              bounded card 9px past the extent measured on the other side; the ordinary 8px
#              bound rejected the otherwise clean rebuild as a phantom second heart.
# Keep this recovery-only.  `_EXTENT_TOLERANCE_PX` also derives item_nav's independently measured
# 16px cross-check bound; widening that would admit a known different-profile pair at 218px.
# Nine remains far below the 47px minimum real gutter, and its doubled slack remains below it.
_RECOVERY_EXTENT_TOLERANCE_PX = 9

# How far a layout-corroborated two-strip cluster may sit from an already-MEASURED majority
# before overriding it is refused.  This is the one part of the repair that discards evidence
# frameshift's quorum accepted, so it is the one that has to justify itself hardest.
#
# The FLOOR is derived, and the derivation is what makes the override safe rather than merely
# bounded: it is one pixel past `_EXTENT_TOLERANCE_PX`, i.e. the smallest per-step error the page
# fold cannot already absorb.  That single fact rules the dangerous band out twice over:
#   * BELOW it a repair is unnecessary.  [measured 2026-08-15, over the seven-frame synthetic read
#     in tests/test_item_index.py: a wrong measured shift of 1..8px injected at ANY one of the 7
#     pairs still yields a usable index whose block count and translation are IDENTICAL to the
#     clean build; from 9px every pair refuses.]  Accumulated chain error cancels between two
#     sightings of the same card, because both inherit the same chain prefix -- so a sub-tolerance
#     step error never grows into a fold disagreement, however long the read.
#   * BELOW it a wrong repair is also undetectable.  The caller's probe rejects a proposal by
#     re-running the ordinary assembly, and an error the assembly tolerates by design produces no
#     contradiction to reject.  So the old 4px floor admitted exactly the corrections that could
#     only ever turn a correct answer into a wrong one, never rescue a capture.
# Above the floor both properties invert: the raw value would have refused the whole capture, and
# a wrong override contradicts the fold loudly enough for the probe to throw the repair away.
#
# The floor stored here is the one for a fold at the DEFAULT tolerance.  `_layout_repaired_shift`
# re-derives it from whatever tolerance its own call folds at, because the frame-omission rebuild
# uses `_RECOVERY_EXTENT_TOLERANCE_PX` instead -- the reasoning above is about the fold actually
# in use, so pinning the floor to the default would reintroduce the absorbed-and-undetectable
# band on exactly that path.
#
# The CEILING is not derived and is deliberately conservative: it is the reproduced animation's
# 10px correction with a little room, not a measured limit.  Refusing past it costs a capture
# (fail-closed), so leave it alone until a labelled corpus of animation runs says otherwise --
# and note nothing yet measures how often an ordinary strip bank holds a coincidental two-vote
# cluster this far from a correct majority.
_MAJORITY_OVERRIDE_PX = (_EXTENT_TOLERANCE_PX + 1, 12)

# A structure-only measured tail is NOT an extension of the ordinary override window above.
# It has its own two-pixel window and the fixed four-pair grammar in
# `_repair_shifts_from_layout`.  The floor is exactly one past the NCC-backed ceiling; the ceiling
# is the saved +207 -> +221 correction and remains far below half the smallest Hinge gutter.
# Nothing currently authorises a 15px structure-only correction.
_STRUCTURAL_TAIL_CORRECTION_PX = (_MAJORITY_OVERRIDE_PX[1] + 1, 14)

# `_MAX_STEP_PX` (imported above) bounds a layout-supported repair: the reader targets about one
# third of the locally observed card spacing, so a repair must still fit inside one ordinary read
# step.  It is never a back door for a large jump that frameshift deliberately refused outside its
# 900px trust window.

# The smallest page-row gap that can separate two DISTINCT blocks. Derived, not chosen: within a
# single frame segment.py only ever cuts on a background run inside its gutter window or on a
# longer run with a card's own top corner below it, so no two blocks it emits are closer than the
# bottom of that window. [corpus: every one of the 206 real card-to-card gutters across 139
# frames measured EXACTLY 53px; the tolerance either side of the canonical 52-53 is pure
# measurement slack.]
#
# Two RESOLVED blocks closer than this therefore contradict the layout, and the reading that
# matters is the dangerous one: one card whose sightings fragmented into two non-overlapping
# groups, i.e. a fabricated item of exactly the kind doc 5.10 measured. `_EXTENT_TOLERANCE_PX` is
# subtracted at the use site so chain slack cannot fire it on its own.
_MIN_ITEM_GAP_PX = min(_GUTTER_PX) - _GUTTER_TOLERANCE_PX

# Page-background rows required BELOW the final frame's last block before the capture is called
# "reached the end". Sized at the top of the gutter window: a gap wider than the widest gutter
# the layout draws cannot be the gap before a next card that the analysed band still contains.
# What it can never rule out is a card beginning past the band's bottom EDGE — see `_tail`, which
# states that limit and names the corroborating signal.
_END_TAIL_GAP_PX = max(_GUTTER_PX) + _GUTTER_TOLERANCE_PX

# Logged with every refusal dossier.  This is not a compatibility switch; it is a human-readable
# name for the loaded decision surface, paired by hinge.py with hashes of the actual in-memory
# indexer decision path and splitter.  Those values distinguish "the current source replays
# cleanly" from "the long-lived
# worker was still executing an older indexer" without trusting the working tree alone.
ITEM_INDEX_ALGORITHM_ID = "bounded-card-split-v13"


# =====================================================================================
# Result vocabulary. Plain string constants, matching segment.py and frameshift.py (there is no
# Enum anywhere in operation_love/) and directly loggable to the JSONL debug log.
#
# These are PAGE-level classes and are deliberately NOT segment.py's per-frame `BLOCK_*` values,
# even where the names rhyme: a card that came back PARTIAL on four frames and complete on the
# fifth is a resolved, croppable, SELECTABLE item here. Each sighting keeps its own per-frame
# class in `BlockObservation.kind`.
# =====================================================================================

ITEM_SELECTABLE = "selectable"      # extent resolved, exactly one heart: gets a model index
ITEM_CONTEXT = "context"            # extent resolved, no heart: read and referenced, never a
                                    # choice; consumes NO model index and NO heart ordinal
ITEM_PARTIAL = "partial"            # never observed complete: not croppable, so not selectable,
                                    # but STILL consumes a heart ordinal if it has a heart
ITEM_AMBIGUOUS = "ambiguous"        # >1 heart at one page position: a failure, not a choice
ITEM_LEADING_CHROME = "leading_chrome"  # Hinge's header above item 1, at a confirmed scroll-top;
                                    # outside both index spaces (see the module docstring)


class ItemIndexError(RuntimeError):
    """The index could not be built AT ALL: no frames to build it from.

    Deliberately narrow. Everything the two dependencies refuse to do — no cv2/numpy, undecodable
    bytes, mismatched frame sizes, a degenerate band, a missing like template — already raises
    `segment.SegmentationError` or `frameshift.ShiftEstimationError`, and those propagate through
    this module UNCHANGED rather than being rewrapped, because their messages name the specific
    thing that went wrong and a wrapper would only blur it.

    Distinct from an UNUSABLE index, which is a result: the module looked, and something it saw
    contradicts something else it saw, or the correspondence chain broke. That carries its full
    evidence in `ItemIndex.failures`. This exception means there was nothing to look at.
    """


@dataclass(frozen=True)
class BlockObservation:
    """One frame's sighting of one block, in both coordinate spaces.

    `page_y0`/`page_y1` are the shared page space (frame rows plus that frame's accumulated
    offset, half-open). `frame_y0`/`frame_y1` are the rows this block occupied in THAT frame, and
    they are carried rather than left to be recomputed because they are what a crop pass slices:
    pick a sighting with `complete` set, and its `frame_y0:frame_y1` of `frames[frame_index]` is
    the item.

    `kind` is segment.py's per-frame `BLOCK_*` for this sighting alone, which is frequently
    PARTIAL on a block that the page-level index resolves completely. `hearts` are `(x, page_y)`
    — x needs no correction because a list scroll has no horizontal component.
    """
    frame_index: int
    page_y0: int
    page_y1: int
    frame_y0: int
    frame_y1: int
    kind: str
    complete: bool
    top_observed: bool
    bottom_observed: bool
    hearts: tuple[tuple[int, int], ...]

    @property
    def height(self) -> int:
        return self.page_y1 - self.page_y0


@dataclass(frozen=True)
class IndexedBlock:
    """One physical block of the profile, folded from every sighting of it.

    `page_y0`/`page_y1` are half-open and, whenever any sighting was complete, are an extent some
    single frame actually OBSERVED — never a blend of two (see the module docstring). On a block
    no frame ever bounded they are the hull of the fragments, which is a LOWER bound on the true
    card, and `kind` is `ITEM_PARTIAL` to say so.

    `heart_ordinal` is 1-based over every heart down the page, `model_index` is 1-based over the
    `ITEM_SELECTABLE` blocks only, and either may be None — a context block has neither, a
    partial block with a heart has an ordinal but no model index. That asymmetry is doc 5.3's
    whole point and is not a special case to remove.
    """
    page_y0: int
    page_y1: int
    x0: int
    x1: int
    kind: str
    hearts: tuple[tuple[int, int], ...]
    heart_ordinal: int | None
    model_index: int | None
    observations: tuple[BlockObservation, ...]
    reason: str

    @property
    def height(self) -> int:
        return self.page_y1 - self.page_y0

    @property
    def heart(self) -> tuple[int, int] | None:
        """The single heart's `(x, page_y)`, or None. Deliberately None for a block with TWO
        hearts as well as for one with none, exactly as `segment.Block.heart` is: returning
        `hearts[0]` there would smuggle back the guess `ITEM_AMBIGUOUS` exists to prevent."""
        return self.hearts[0] if len(self.hearts) == 1 else None

    @property
    def frames(self) -> tuple[int, ...]:
        """Indices of the frames this block was seen in, ascending."""
        return tuple(sorted({o.frame_index for o in self.observations}))

    @property
    def complete(self) -> bool:
        """Some single frame observed BOTH edges, so `page_y0..page_y1` is the card's true
        extent and it can be cropped. The precondition for `ITEM_SELECTABLE`."""
        return any(o.complete for o in self.observations)

    @property
    def croppable(self) -> tuple[BlockObservation, ...]:
        """The sightings a crop pass may use: those that saw the whole block. Empty on an
        `ITEM_PARTIAL` block, which is the point — there is no frame to crop it from."""
        return tuple(o for o in self.observations if o.complete)


@dataclass(frozen=True)
class VideoMuteMarker:
    """One affirmative, positioned Hinge mute-control observation.

    Unlike the legacy frame boolean, ``x/y`` identify the physical card carrying the app-owned
    control. The indexer never changes frameshift's raw vote. It normally uses this identity
    only to bound a video-card track; the narrow direct-marker bridge additionally derives one
    auditable effective distance when two contained controls and another card-local anchor agree.
    """
    frame_index: int
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class ItemIndexRepair:
    """One accepted repair's audit record; it carries no pixels or inferred identity."""
    pair_index: int
    path: str                    # ``v12_mute_card_track`` or ``legacy_layout_grammar``
    raw_status: str
    raw_delta_px: int | None
    effective_status: str
    effective_delta_px: int | None
    marker_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class ItemIndex:
    """The profile's items, the evidence behind them, and what the capture failed to establish.

    `usable` is the contract, and it is a HARD gate: doc 5.3 says "treat a missing table as a
    hard stop, never as a reason to fall back to a fixed coordinate", and a table that
    contradicts itself is a missing table. Nothing here degrades — `blocks` is empty when the
    correspondence chain broke, and the accessors below raise rather than answer when `usable`
    is False.

    `frames` and `shifts` are the per-frame and per-pair evidence, kept so a validation pass can
    answer "why did this capture index the way it did" from the result alone.

    Read `usable` FIRST. The completeness properties (`reached_end`, `truncated`, `complete`,
    `partial`) describe the capture's coverage and are computed whether or not the index holds
    together, so on an unusable index they answer a question nobody should be asking yet — a
    refused capture can perfectly well have reached the end of the profile and still have no
    items in it.

    `identity` is WHOSE profile this is (`item_identity.ProfileIdentity`), fingerprinted from
    these very frames at build time. It is a field of the index rather than something a caller
    remembers separately because the index is the artefact navigation trusts, and doc 5.5's
    2026-08-12 addendum measured what happens when the two are allowed to drift apart: profile
    A's index driven over profile B's frames returned a confident tap target at model index 1,
    because every geometric check `item_nav` makes passed identically on two stereotyped Hinge
    cards. Geometry is not identity; this is. It may be UNKNOWN (`identity.known` False) — an app
    with no `identity_band`, a capture that never scrolled far enough to reveal the sticky
    header — and navigation refuses such an index rather than proceeding unchecked.
    """
    blocks: tuple[IndexedBlock, ...]
    frames: tuple[FrameSegmentation, ...]
    shifts: tuple[ShiftEstimate, ...]
    offsets: tuple[int | None, ...]
    page_span: tuple[int, int]
    at_scroll_top: bool
    reached_end: bool
    tail_gap_px: int | None
    failures: tuple[str, ...]
    identity: ProfileIdentity
    # The frame positions from the caller's original capture that were actually folded.  This is
    # normally ``range(len(frames))``.  An isolated bad pair may be recovered by dropping either
    # offending intermediate frame and measuring a direct bridge instead; the crop layer must use
    # the same reduced sequence, never the original sequence shifted by one position.
    source_frame_indices: tuple[int, ...] = ()
    # Per-index-frame affirmative animation markers supplied by the app driver. They never alter
    # raw frameshift or segmentation. A bounded measured-bridge repair may consult them as
    # product-specific evidence that changing pixels really come from a video UI rather than
    # from two unrelated static cards; False is absence of proof, never proof of a still image.
    animation_markers: tuple[bool, ...] = ()
    # Position-aware mute-control evidence used by v12's physical video-card tracker.  This is
    # separate from the legacy bool tuple so older callers and recorded manifests remain readable.
    video_mute_markers: tuple[VideoMuteMarker, ...] = ()
    # Accepted effective shifts and the raw evidence they replaced.  This is deliberately
    # structured so a successful capture, not only a refusal sidecar, can say whether v12's
    # physical-card tracker or the legacy compatibility grammar supplied authority.
    repair_provenance: tuple[ItemIndexRepair, ...] = ()
    recovered_from_pair: tuple[int, int] | None = None
    recovery_bridge: tuple[int, int] | None = None
    recovery_failed_shift: ShiftEstimate | None = None
    # Complete refusal provenance.  The singular fields above remain the compatibility view of
    # the first failed pair; one transient frame can make BOTH adjacent pairs refuse, so keeping
    # only one of them would make a successful recovery impossible to audit.
    recovered_from_pairs: tuple[tuple[int, int], ...] = ()
    recovery_failed_shifts: tuple[ShiftEstimate, ...] = ()
    # A segmentation contradiction can be treated like a damaged intermediate frame, but only
    # after the surrounding real frames bridge and the reduced index rebuilds.  This records
    # every omitted frame in that localized recovery window.  Usually each one had its own
    # segmentation failure; a single immediately adjacent frame may also be omitted when the
    # first rebuild proves the corrupted frame left an unresolved cross-frame contradiction.
    # Keep it separate from ``recovered_from_pair``: there need not have been a failed shift,
    # and inventing one would make the successful recovery's audit trail lie.
    recovered_from_segmentation_frames: tuple[int, ...] = ()
    # A complete page fold can prove that a seemingly-valid partial sighting crossed a card
    # boundary.  Such a frame is recoverable only when every fold failure names this exact
    # overrun, the affected frames form one short interior run, and a fresh bridge plus complete
    # reduced rebuild both succeed.  Keep that distinct from a segmenter's own failure: the
    # latter is a frame-local fact; this one is cross-frame evidence.
    recovered_from_fold_contradiction_frames: tuple[int, ...] = ()
    recovery_reason: str | None = None
    # Original frameshift evidence replaced by a layout-assisted acceptance.  This is never a
    # hidden quorum relaxation: the effective shift's reason and this immutable provenance both
    # preserve the raw two-strip evidence for a later audit.
    layout_repaired_shifts: tuple[tuple[int, ShiftEstimate], ...] = ()
    # The per-chain extent slack this index's blocks were ACTUALLY folded at.  A second pass that
    # cross-checks this index has to sum its own slack with this one's, and the frame-omission
    # recovery deliberately builds at `_RECOVERY_EXTENT_TOLERANCE_PX` rather than the default --
    # so a consumer that assumed the default would set a bound one pixel too tight for exactly
    # the indexes that were hardest to build.  Recording it keeps that derivation honest instead
    # of duplicating the assumption in each consumer (see `item_nav._CROSSCHECK_TOLERANCE_PX`).
    extent_tolerance_px: int = _EXTENT_TOLERANCE_PX
    # One string per sighting `_split_on_bounded_cards` excluded while resolving `blocks` — never
    # silent, on the owner's standing rule that no fallback or degradation happens without being
    # visible. A sighting only ever gets excluded when it BRIDGES two cards the group's own
    # complete sightings already proved are separate (the frame-19/20 shape: segment.py missed a
    # gutter in that one frame and emitted a single block spanning both cards), and excluding it
    # is what keeps `_resolve_group`'s disagreement check from reading that bridge as two cards
    # contradicting each other. `notes` is where an operator finds out it happened — a note names
    # the frame, its page rows, and the proven boundary it spanned, so the missed gutter can be
    # traced back to a specific frame rather than only showing up as "the index came out usable
    # with a gap in the evidence". Empty on any index nothing was excluded from, which includes
    # every `ItemIndex` this repo's other test suites hand-build without passing it.
    notes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Nothing this capture saw contradicts anything else it saw, and the page coordinate
        space spans every frame. False is a hard stop for anything that navigates by index."""
        return not self.failures

    @property
    def selectable(self) -> tuple[IndexedBlock, ...]:
        """The blocks the model may choose from, in model-index order. `selectable[k - 1]` is
        model item k."""
        return tuple(b for b in self.blocks if b.kind == ITEM_SELECTABLE)

    @property
    def context(self) -> tuple[IndexedBlock, ...]:
        """Heartless resolved blocks — sent to the model as unnumbered context, never as a
        choice (doc 5.3, and doc 5.7's "context blocks, cropped, unnumbered")."""
        return tuple(b for b in self.blocks if b.kind == ITEM_CONTEXT)

    @property
    def partial(self) -> tuple[IndexedBlock, ...]:
        """Blocks no frame ever bounded end to end. Part of the completeness report: they are
        real page positions with unknown extents, not items to quietly drop."""
        return tuple(b for b in self.blocks if b.kind == ITEM_PARTIAL)

    @property
    def heart_count(self) -> int:
        """Every heart on the indexed page — what a counting navigation would tick off. Counts
        hearts, not blocks, so an `ITEM_AMBIGUOUS` block contributes both of its."""
        return sum(len(b.hearts) for b in self.blocks)

    @property
    def translation(self) -> tuple[int, ...]:
        """Model index -> heart ordinal, as a tuple where `translation[k - 1]` is model item k's
        ordinal. Doc 5.3's private table: the model never sees these numbers, and item 3 in its
        list may well be the 4th heart on the page.

        On an `at_scroll_top=False` index these numbers are RELATIVE to whatever was in view and
        must not be counted against; `heart_ordinal_for` is the accessor that enforces that, and
        is the one a navigator calls. This stays answerable because it describes the capture —
        `item_crops` compares its own renumbering against it, and a validation pass reads it
        without navigating anywhere."""
        return tuple(b.heart_ordinal for b in self.selectable)

    @property
    def truncated(self) -> bool:
        """The capture does not demonstrably cover the whole profile — it did not start at a
        confirmed scroll-top, or it did not reach the end. This is doc 5.7's truncation flag,
        the one sent to the model with the items.

        Note it is about COVERAGE, not correctness: a truncated index can be perfectly usable,
        it just describes a window of the profile rather than all of it."""
        return not (self.at_scroll_top and self.reached_end)

    @property
    def complete(self) -> bool:
        """Every block was bounded end to end at least once AND the capture covers the whole
        profile. The full completeness answer in one property; `partial`, `truncated`,
        `reached_end` and `at_scroll_top` are its components."""
        return not self.partial and not self.truncated

    def block_for(self, model_index: int) -> IndexedBlock:
        """The block the model's item `model_index` refers to, 1-based.

        Raises rather than returning None on an unusable index or an out-of-range index. Doc 5.6
        is explicit that we never substitute a different item, and returning the nearest one, or
        None for a caller to handle however it likes, is how a substitution gets made.
        """
        if not self.usable:
            raise ItemIndexError(
                "this item index is unusable and carries no valid model indices: "
                + "; ".join(self.failures))
        items = self.selectable
        if not 1 <= model_index <= len(items):
            raise ItemIndexError(
                f"model index {model_index} is outside 1..{len(items)}; the model was offered "
                f"{len(items)} item(s) and may not be answered with anything else")
        return items[model_index - 1]

    def heart_ordinal_for(self, model_index: int) -> int:
        """How many hearts down the page the model's item `model_index` is — the number a
        counting navigation actually uses. Same raising contract as `block_for`, plus one more.

        REFUSES an index built with `at_scroll_top=False`, and that refusal is the point of the
        method existing separately from `block_for`. With the assertion False the ordinals are
        RELATIVE to whatever was in view: a validation pass measured a mid-profile window
        returning translation (2, 3, 4, 5, 6) for what were really items 5..9. Counting hearts
        from a real scroll top towards one of those numbers lands a fixed number of cards away
        from the item the model chose, and until this raise existed nothing on the result said
        so — the method answered with a confident int (doc 5.3, addendum 2026-08-12: "navigation
        must not count against a False index").

        `translation` still answers, deliberately: it is a description of the capture, it is what
        `item_crops` compares its own renumbering against, and it is read by validation passes
        that are not about to count anything. This is the accessor a NAVIGATOR calls, so this is
        where the enforcement belongs.

        Otherwise always an int, never None: `ITEM_SELECTABLE` means exactly one heart by
        construction, so every block `block_for` can return carries an ordinal."""
        if not self.at_scroll_top:
            raise ItemIndexError(
                "this index was built with at_scroll_top=False, so its heart ordinals are "
                "RELATIVE to whatever was in view and not the profile's own numbering — a "
                "measured mid-profile window returns translation (2, 3, 4, 5, 6) for what are "
                "really items 5..9. Counting to one of them from a genuine scroll top taps a "
                "fixed number of cards away from the item that was chosen, so there is no "
                "ordinal to return. Re-index from an affirmatively confirmed top "
                "(scroll_top.confirm_scroll_top) instead")
        return self.block_for(model_index).heart_ordinal


def _matched_delta_clusters(shift: ShiftEstimate) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """The admissible disjoint clusters among usable strip votes for layout corroboration.

    ``frameshift`` intentionally has the final word on an ordinary strip-only measurement.  This
    helper normally exposes its *two*-witness alternatives for the much narrower layout check
    below.  It exposes a single witness only when it is the pair's sole matched strip and the raw
    result is already a refusal; `_layout_repaired_shift` then requires top, bottom, and heart
    geometry, and `_repair_shifts_from_layout` still refuses it unless an adjacent pair forms the
    same bounded animation run.  Thus a lone NCC coincidence can never repair an isolated pair.

    A two-voter candidate uses the same decision surface as ``frameshift._resolve``: both values
    must sit within `_AGREEMENT_TOLERANCE_PX` of their rounded median.  Comparing the values to
    *each other* instead is too strict -- the saved `+278,+282` pair is 4px apart but both are
    only 2px from its `+280` median, so frameshift correctly counts two agreeing witnesses.

    Overlapping pairs remain inadmissible.  Votes at 0, 3 and 6px can make candidate pairs around
    2 and 4, but the middle vote belongs to both; exposing either would manufacture a repair
    candidate out of evidence that also supports its neighbour.  Disjoint alternatives are
    retained so independent layout can select between them.
    """
    deltas = sorted(s.delta_px for s in shift.strips
                    if s.state == STRIP_MATCHED and s.delta_px is not None)
    if shift.status == SHIFT_NO_CONSENSUS and len(deltas) == 1:
        return ((deltas[0], (deltas[0],)),)

    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for left in range(len(deltas)):
        for right in range(left + 1, len(deltas)):
            candidate = int(round((deltas[left] + deltas[right]) / 2))
            supporters = tuple(i for i, delta in enumerate(deltas)
                               if abs(delta - candidate) <= _AGREEMENT_TOLERANCE_PX)
            if supporters == (left, right):
                candidates.append((candidate, (left, right), (deltas[left], deltas[right])))

    overlapping: set[int] = set()
    for i, (_candidate, voter_indices, _voters) in enumerate(candidates):
        if any(set(voter_indices).intersection(other_indices)
               for j, (_other, other_indices, _other_voters) in enumerate(candidates) if i != j):
            overlapping.add(i)
    return tuple((candidate, voters) for i, (candidate, _indices, voters) in enumerate(candidates)
                 if i not in overlapping)


def _structural_landmarks(before: FrameSegmentation, after: FrameSegmentation,
                          ) -> tuple[tuple[str, int], ...]:
    """Independent same-type landmark deltas that physically fit one read-scroll.

    A top/bottom edge is used only when its own frame observed it; a heart is a separately
    detected glyph.  We pair like with like across every pair of blocks because the index does
    not yet know which blocks correspond -- that is exactly the question the candidate shift is
    answering.  The later candidate check requires several of these pairings and more than one
    landmark type, so coincidental repeated gutters alone cannot repair a shift.
    """
    landmarks: list[tuple[str, int]] = []
    for kind, attr in (("top", "top"), ("bottom", "bottom")):
        left = [getattr(block, attr).y for block in before.blocks
                if getattr(block, attr).observed]
        right = [getattr(block, attr).y for block in after.blocks
                 if getattr(block, attr).observed]
        landmarks.extend((kind, a - b) for a in left for b in right
                         if 0 <= a - b <= _MAX_STEP_PX)
    left_hearts = [heart for block in before.blocks for heart in block.hearts]
    right_hearts = [heart for block in after.blocks for heart in block.hearts]
    landmarks.extend(("heart", ay - by) for ax, ay in left_hearts for bx, by in right_hearts
                     if abs(ax - bx) <= 2 and 0 <= ay - by <= _MAX_STEP_PX)
    return tuple(landmarks)


def _observed_gutters(segmentation: FrameSegmentation) -> tuple[tuple[int, int], ...]:
    """Exact `(upper bottom, lower top)` rows for fully observed canonical gutters.

    A generic top and bottom can belong to unrelated blocks.  Keeping them paired here proves
    the specific layout object the live-video transition preserves: both sides of one 47..58px
    Hinge gutter, observed in both frames.  Band-edge fragments are excluded by the edge flags.
    """
    low = min(_GUTTER_PX) - _GUTTER_TOLERANCE_PX
    high = max(_GUTTER_PX) + _GUTTER_TOLERANCE_PX
    gutters: list[tuple[int, int]] = []
    for upper, lower in pairwise(segmentation.blocks):
        if not upper.bottom.observed or not lower.top.observed:
            continue
        bottom, top = upper.bottom.y, lower.top.y
        if low <= top - bottom <= high:
            gutters.append((bottom, top))
    return tuple(gutters)


def _shared_gutter_witnesses(before: FrameSegmentation, after: FrameSegmentation,
                             candidate: int,
                             ) -> tuple[tuple[int, int, int, int], ...]:
    """Shared canonical gutters whose two independently segmented edges move `candidate` px."""
    return tuple(
        (before_bottom, before_top, after_bottom, after_top)
        for before_bottom, before_top in _observed_gutters(before)
        for after_bottom, after_top in _observed_gutters(after)
        if before_bottom - after_bottom == candidate and before_top - after_top == candidate
    )


def _edge_only_two_strip_shift(pair_index: int, before: FrameSegmentation,
                               after: FrameSegmentation, shift: ShiftEstimate,
                               ) -> tuple[ShiftEstimate, str | None]:
    """Propose the saved transition's two-NCC plus exact-shared-gutter shift.

    This helper never grants ordinary layout acceptance.  `_repair_shifts_from_layout` admits
    its tagged proposal only as the first member of one exact three-pair animation pattern, and
    the builder still probes the complete page before committing it.
    """
    if shift.status != SHIFT_NO_CONSENSUS:
        return shift, None
    passing: list[tuple[int, tuple[int, ...], tuple[int, int, int, int]]] = []
    for candidate, voters in _matched_delta_clusters(shift):
        if len(voters) != 2 or not 0 < candidate <= _MAX_STEP_PX:
            continue
        witnesses = _shared_gutter_witnesses(before, after, candidate)
        if len(witnesses) == 1:
            passing.append((candidate, voters, witnesses[0]))
    if len(passing) != 1:
        return shift, None

    candidate, voters, witness = passing[0]
    before_bottom, before_top, after_bottom, after_top = witness
    repaired = replace(
        shift, delta_px=candidate, consensus_px=candidate, status=SHIFT_MEASURED,
        agreeing=2, dissenting=max(0, shift.eligible - 2),
        confidence=(2 / shift.eligible if shift.eligible else 0.0),
        reason=(
            f"three-pair edge companion from {shift.status}: exactly two independent NCC strips "
            f"({voters[0]:+d}px, {voters[1]:+d}px) form +{candidate}px; one canonical gutter "
            f"moves exactly from rows {before_bottom}..{before_top} to "
            f"{after_bottom}..{after_top}"))
    note = (
        f"frame {pair_index}'s pair with frame {pair_index + 1}: three-pair edge companion "
        f"+{candidate}px from exactly two NCC strips {list(voters)} and one exact shared gutter "
        f"{before_bottom}..{before_top} -> {after_bottom}..{after_top}")
    return repaired, note


def _layout_repaired_shift(pair_index: int, before: FrameSegmentation, after: FrameSegmentation,
                           shift: ShiftEstimate, *,
                           extent_tolerance_px: int = _EXTENT_TOLERANCE_PX,
                           ) -> tuple[ShiftEstimate, str | None]:
    """Return a strictly layout-corroborated two-strip repair, or the original shift.

    This does *not* relax ``frameshift``.  It may only rescue a no-consensus result, or replace
    an actually different measured majority, when exactly two NCC strips form an alternative
    cluster and three independently segmented landmarks of at least two kinds locate that exact
    candidate.  A no-consensus pair with only one matched strip has the stronger requirement that
    all three independently detected landmark kinds -- top, bottom, and heart -- agree exactly.
    The same exact-three-kind rule may select one dissenting strip only after every ordinary
    two-strip alternative has failed layout corroboration.  That covers the saved live-video
    bank ``+256,+256,+261``, whose independently segmented top, bottom and heart all move
    ``+261``.  The enclosing run check still prevents either one-strip form from repairing an
    isolated pair.  More than one passing candidate is ambiguity, hence no repair.
    """
    if shift.status not in (SHIFT_NO_CONSENSUS, SHIFT_MEASURED):
        return shift, None

    candidates = _matched_delta_clusters(shift)
    landmarks = _structural_landmarks(before, after)
    passing: list[tuple[int, tuple[int, ...], tuple[tuple[str, int], ...]]] = []
    for candidate, voters in candidates:
        if not 0 < candidate <= _MAX_STEP_PX:
            continue
        corroborating = tuple((kind, delta) for kind, delta in landmarks
                               if abs(delta - candidate) <= _AGREEMENT_TOLERANCE_PX)
        kinds = {kind for kind, _delta in corroborating}
        exact_kinds = {kind for kind, delta in landmarks if delta == candidate}
        enough_layout = (len(corroborating) >= 3 and
                         (exact_kinds == {"top", "bottom", "heart"} if len(voters) == 1
                          else len(kinds) >= 2))
        if enough_layout:
            passing.append((candidate, voters, corroborating))

    # A changing video can create a repeated wrong match while one stable strip follows the
    # actual card.  Do not put singleton alternatives into `_matched_delta_clusters`: that
    # helper's pair candidates are also used by the weaker shared-gutter exception.  Instead,
    # fall back here only when NONE of those stronger candidates passed.  There need not be an
    # ordinary pair: the saved `+286,+274` bank has two mutually dissenting matches and therefore
    # no two-strip candidate at all, while top, bottom and heart independently fix `+274`.  The
    # singleton must be a unique exact strip delta and all three independently detected landmark
    # kinds must equal it pixel-for-pixel.  Later,
    # `_repair_shifts_from_layout` still requires an adjacent two-strip repair and the builder
    # still probes the complete page before committing the run.
    if not passing and shift.status == SHIFT_NO_CONSENSUS:
        matched = tuple(s.delta_px for s in shift.strips
                        if s.state == STRIP_MATCHED and s.delta_px is not None)
        for candidate in sorted(set(matched)):
            if matched.count(candidate) != 1 or not 0 < candidate <= _MAX_STEP_PX:
                continue
            exact = tuple((kind, delta) for kind, delta in landmarks if delta == candidate)
            if {kind for kind, _delta in exact} == {"top", "bottom", "heart"}:
                passing.append((candidate, (candidate,), exact))
    if len(passing) != 1:
        return shift, None

    candidate, voters, corroborating = passing[0]
    # A normal measurement already selected this answer.  Do not emit a cosmetic repair merely
    # because its own two strips appear in a larger measured cluster.
    if shift.status == SHIFT_MEASURED:
        if shift.delta_px == candidate:
            return shift, None
        # A measured majority may be overridden only for the observed animation shape: its
        # layout-supported two-strip cluster is materially different, but still close enough to
        # be ordinary capture drift rather than a different card.  The enclosing run check below
        # prevents one stray pair from becoming an exception to frameshift's quorum.
        # The floor tracks THIS call's fold tolerance rather than the module default, because the
        # frame-omission rebuild deliberately folds at `_RECOVERY_EXTENT_TOLERANCE_PX`.  Deriving
        # it per call is what keeps `_MAJORITY_OVERRIDE_PX`'s justification -- "one pixel past
        # what the fold already absorbs" -- true on that path too, instead of only on the default
        # one.  A tolerance at or above the ceiling simply empties the window, which is the right
        # answer: a fold that absorbs that much never needs a majority overridden at all.
        correction = abs(shift.delta_px - candidate)
        if not extent_tolerance_px + 1 <= correction <= _MAJORITY_OVERRIDE_PX[1]:
            return shift, None

    types = ", ".join(sorted({kind for kind, _delta in corroborating}))
    old = (f"{shift.status} {shift.delta_px:+d}px" if shift.delta_px is not None
           else shift.status)
    witness_label = "one NCC strip" if len(voters) == 1 else "exactly two independent NCC strips"
    repaired = replace(
        shift, delta_px=candidate, consensus_px=candidate, status=SHIFT_MEASURED,
        agreeing=len(voters), dissenting=max(0, shift.eligible - len(voters)),
        confidence=(len(voters) / shift.eligible if shift.eligible else 0.0),
        reason=(
            f"layout-assisted acceptance from {old}: {witness_label} "
            f"({', '.join(f'{value:+d}px' for value in voters)}) form +{candidate}px; "
            f"{len(corroborating)} observed structural landmark pairings across {types} "
            f"also land within {_AGREEMENT_TOLERANCE_PX}px (one-step maximum {_MAX_STEP_PX}px)"))
    note = (
        f"frame {pair_index}'s pair with frame {pair_index + 1}: layout-assisted shift from {old} "
        f"to +{candidate}px using {witness_label} {list(voters)} plus "
        f"{len(corroborating)} structural landmark pairings across {types}")
    return repaired, note


def _structural_tail_shift(pair_index: int, before: FrameSegmentation, after: FrameSegmentation,
                           shift: ShiftEstimate,
                           ) -> tuple[ShiftEstimate, str | None]:
    """Propose one exact-layout tail beyond the ordinary majority-override ceiling.

    This is deliberately not part of `_layout_repaired_shift`: it may have zero NCC witnesses at
    the structural answer and is therefore weaker than every ordinary repair.  The enclosing
    run gate admits it only as the measured final member of one exact four-pair animation shape,
    after two ordinary two-strip repairs and one exact one-strip repair.  The builder then probes
    the complete page before committing anything.

    Its two-pixel correction window begins one pixel beyond the existing strip-supported ceiling
    and ends at the saved transition's +207px raw versus +221px exact geometry.  Fourteen pixels
    is still far below half one Hinge gutter; nothing authorises a wider structural-only answer.
    """
    if shift.status != SHIFT_MEASURED or shift.delta_px is None:
        return shift, None
    landmarks = _structural_landmarks(before, after)
    candidates = tuple(sorted({
        delta for _kind, delta in landmarks
        if 0 < delta <= _MAX_STEP_PX
        and {kind for kind, other in landmarks if other == delta}
        == {"top", "bottom", "heart"}
    }))
    if len(candidates) != 1:
        return shift, None
    candidate = candidates[0]
    correction = abs(shift.delta_px - candidate)
    if not _STRUCTURAL_TAIL_CORRECTION_PX[0] <= correction \
            <= _STRUCTURAL_TAIL_CORRECTION_PX[1]:
        return shift, None

    matched = tuple(s.delta_px for s in shift.strips
                    if s.state == STRIP_MATCHED and s.delta_px == candidate)
    repaired = replace(
        shift, delta_px=candidate, consensus_px=candidate, status=SHIFT_MEASURED,
        agreeing=len(matched), dissenting=max(0, shift.eligible - len(matched)),
        confidence=(len(matched) / shift.eligible if shift.eligible else 0.0),
        reason=(
            f"four-pair structural tail proposal from measured {shift.delta_px:+d}px to "
            f"+{candidate}px: exact top, bottom and heart geometry; correction {correction}px "
            f"is above the ordinary {_MAJORITY_OVERRIDE_PX[1]}px strip-supported ceiling and "
            f"inside the dedicated {_STRUCTURAL_TAIL_CORRECTION_PX[0]}.."
            f"{_STRUCTURAL_TAIL_CORRECTION_PX[1]}px structural-tail window"))
    note = (
        f"frame {pair_index}'s pair with frame {pair_index + 1}: four-pair structural tail "
        f"from measured {shift.delta_px:+d}px to +{candidate}px using exact top, bottom and "
        f"heart geometry")
    return repaired, note


def _exact_multi_strip_shift(pair_index: int, before: FrameSegmentation,
                             after: FrameSegmentation, shift: ShiftEstimate,
                             ) -> tuple[ShiftEstimate, str | None]:
    """Propose one exact 3+-strip/full-layout boundary, for the five-pair or two-pair grammars.

    This is deliberately separate from `_matched_delta_clusters`.  The ordinary helper exposes
    under-quorum two-strip alternatives; it must not start treating every local mode in a noisy
    strip bank as a measurement.  Here a candidate must be an exact repeated strip value, not a
    rounded centroid, and independently segmented top, bottom, and heart landmarks must all land
    on that same pixel.  `_repair_shifts_from_layout` still grants the proposal authority only as
    a measured-bracketed endpoint of one exact five-pair refusal island, or as one of a bare
    two-pair window where both pairs clear this same bar with no interior pair between them.
    """
    if shift.status != SHIFT_NO_CONSENSUS:
        return shift, None
    matched = tuple(s.delta_px for s in shift.strips
                    if s.state == STRIP_MATCHED and s.delta_px is not None)
    landmarks = _structural_landmarks(before, after)
    passing: list[tuple[int, int, tuple[tuple[str, int], ...]]] = []
    for candidate in sorted(set(matched)):
        supporters = matched.count(candidate)
        if supporters < 3 or not 0 < candidate <= _MAX_STEP_PX:
            continue
        exact = tuple((kind, delta) for kind, delta in landmarks if delta == candidate)
        if {kind for kind, _delta in exact} == {"top", "bottom", "heart"}:
            passing.append((candidate, supporters, exact))
    if len(passing) != 1:
        return shift, None

    candidate, supporters, exact = passing[0]
    repaired = replace(
        shift, delta_px=candidate, consensus_px=candidate, status=SHIFT_MEASURED,
        agreeing=supporters, dissenting=max(0, shift.eligible - supporters),
        confidence=(supporters / shift.eligible if shift.eligible else 0.0),
        reason=(
            f"exact-multi boundary proposal from {shift.status}: {supporters} independent NCC "
            f"strips land exactly at +{candidate}px and exact top, bottom and heart geometry "
            "selects that one repeated strip value"))
    note = (
        f"frame {pair_index}'s pair with frame {pair_index + 1}: exact-multi full-landmark "
        f"cluster boundary +{candidate}px from {supporters} exact NCC strips plus exact top, "
        "bottom and heart geometry")
    return repaired, note


def _measured_layout_bridge(pair_index: int, before: FrameSegmentation,
                            after: FrameSegmentation, shift: ShiftEstimate, *,
                            allow_full_layout_projection: bool = False,
                            extent_tolerance_px: int = _EXTENT_TOLERANCE_PX,
                            ) -> tuple[ShiftEstimate, str | None]:
    """Describe a raw measured pair strong enough to bridge two nearby repair proposals.

    It already passed frameshift's ordinary quorum.  Ordinarily this helper only adds an
    independent exact-layout requirement: at least two observed landmark kinds must sit on the
    measured delta pixel-for-pixel, and the raw shift is returned unchanged.

    Inside an affirmatively video-marked measured-bridge window only, the caller may permit one
    stronger form: a UNIQUE exact top/bottom/heart delta within the fold's existing tolerance may
    remove a small NCC centroid error when at least one matched strip lands on that exact answer.
    That projection is still only a proposal; refusal pairs on both sides plus the complete page
    rebuild remain mandatory.  Anchors never use it, and nothing here lowers frameshift quorum.
    """
    if (shift.status != SHIFT_MEASURED or shift.delta_px is None or shift.agreeing < 3
            or not 0 < shift.delta_px <= _MAX_STEP_PX):
        return shift, None
    landmarks = _structural_landmarks(before, after)
    exact = tuple((kind, delta) for kind, delta in landmarks if delta == shift.delta_px)
    kinds = {kind for kind, _delta in exact}
    if len(exact) >= 2 and len(kinds) >= 2:
        return shift, (
            f"frame {pair_index}'s pair with frame {pair_index + 1}: retained raw measured "
            f"bridge +{shift.delta_px}px from {shift.agreeing} NCC strips and {len(exact)} "
            f"exact landmark pairings across {', '.join(sorted(kinds))}")
    if not allow_full_layout_projection:
        return shift, None

    candidates = tuple(sorted({
        delta for _kind, delta in landmarks
        if 0 < delta <= _MAX_STEP_PX
        and 0 < abs(delta - shift.delta_px) <= extent_tolerance_px
        and {kind for kind, other in landmarks if other == delta}
        == {"top", "bottom", "heart"}
        and any(strip.state == STRIP_MATCHED and strip.delta_px == delta
                for strip in shift.strips)
    }))
    if len(candidates) != 1:
        return shift, None
    candidate = candidates[0]
    voters = tuple(strip.delta_px for strip in shift.strips
                   if strip.state == STRIP_MATCHED and strip.delta_px is not None
                   and abs(strip.delta_px - candidate) <= _AGREEMENT_TOLERANCE_PX)
    projected = replace(
        shift, delta_px=candidate, consensus_px=candidate,
        agreeing=len(voters), dissenting=max(0, shift.eligible - len(voters)),
        confidence=(len(voters) / shift.eligible if shift.eligible else 0.0),
        reason=(
            f"video-marked measured-bridge projection from raw measured {shift.delta_px:+d}px "
            f"({shift.agreeing} agreeing NCC strips) to +{candidate}px: exact top, bottom and "
            f"heart geometry plus exact NCC witness; correction "
            f"{abs(candidate - shift.delta_px)}px is inside the existing "
            f"{extent_tolerance_px}px fold tolerance"))
    return projected, (
        f"frame {pair_index}'s pair with frame {pair_index + 1}: video-marked measured bridge "
        f"projected from raw {shift.delta_px:+d}px to exact full-layout {candidate:+d}px; "
        f"{len(voters)} NCC strip(s) remain within {_AGREEMENT_TOLERANCE_PX}px")


def _project_to_exact_full_layout(pair_index: int, before: FrameSegmentation,
                                  after: FrameSegmentation, raw: ShiftEstimate,
                                  proposed: ShiftEstimate,
                                  ) -> tuple[ShiftEstimate, str | None]:
    """Project one two-strip proposal onto its unique exact full-layout delta, if different.

    A rounded two-vote centroid can sit between the physical answers (`+274,+275` rounds to
    `+274` while card top, card bottom and heart all move exactly `+275`).  This projection is
    used only by the measured-bridge window.  Both selected NCC voters must remain within the
    ordinary agreement tolerance of the exact structural answer, so geometry can remove the
    rounding error but cannot nominate an unrelated shift.
    """
    if (raw.status != SHIFT_NO_CONSENSUS or proposed.delta_px is None
            or proposed.agreeing != 2):
        return proposed, None
    voters = tuple(s.delta_px for s in raw.strips
                   if s.state == STRIP_MATCHED and s.delta_px is not None
                   and abs(s.delta_px - proposed.delta_px) <= _AGREEMENT_TOLERANCE_PX)
    if len(voters) != 2:
        return proposed, None
    landmarks = _structural_landmarks(before, after)
    candidates = tuple(sorted({
        delta for _kind, delta in landmarks
        if 0 < delta <= _MAX_STEP_PX
        and {kind for kind, other in landmarks if other == delta}
        == {"top", "bottom", "heart"}
        and all(abs(voter - delta) <= _AGREEMENT_TOLERANCE_PX for voter in voters)
    }))
    if len(candidates) != 1 or candidates[0] == proposed.delta_px:
        return proposed, None
    candidate = candidates[0]
    projected = replace(
        proposed, delta_px=candidate, consensus_px=candidate,
        reason=(f"{proposed.reason}; measured-bridge exact-layout projection from "
                f"{proposed.delta_px:+d}px to {candidate:+d}px using top, bottom and heart"))
    return projected, (
        f"frame {pair_index}'s pair with frame {pair_index + 1}: measured-bridge projection "
        f"from two-strip centroid {proposed.delta_px:+d}px to exact full-layout "
        f"{candidate:+d}px; voters {list(voters)} remain within "
        f"{_AGREEMENT_TOLERANCE_PX}px")


def _marker_block(segmentation: FrameSegmentation, marker: VideoMuteMarker):
    """The one segmented card containing a mute-control origin, or ``None``.

    A Hinge mute control sits inside the card, near its upper-left corner.  Containment is an
    identity check, not an attempt to classify image pixels: a marker in page chrome or in a
    gutter never starts a video track.
    """
    candidates = tuple(block for block in segmentation.blocks
                       if block.x0 <= marker.x < block.x1 and block.y0 <= marker.y < block.y1)
    return candidates[0] if len(candidates) == 1 else None


def _track_candidate_deltas(pair_index: int, before: FrameSegmentation,
                            after: FrameSegmentation, raw: ShiftEstimate, *,
                            extent_tolerance_px: int) -> tuple[int, ...]:
    """Strict candidate deltas a *known* video card may use to link adjacent sightings.

    This merely supplies alternatives for card-identity matching.  The caller separately keeps
    the raw quorum and only records a repair when the complete assembly succeeds.
    """
    candidates: set[int] = set()
    if raw.status == SHIFT_MEASURED and raw.delta_px is not None:
        candidates.add(raw.delta_px)
        projected, note = _measured_layout_bridge(
            pair_index, before, after, raw, allow_full_layout_projection=True,
            extent_tolerance_px=extent_tolerance_px)
        if note is not None and projected.delta_px is not None:
            candidates.add(projected.delta_px)
    elif raw.status == SHIFT_NO_CONSENSUS:
        proposed, note = _layout_repaired_shift(
            pair_index, before, after, raw, extent_tolerance_px=extent_tolerance_px)
        if note is not None and proposed.delta_px is not None:
            candidates.add(proposed.delta_px)
            projected, projection_note = _project_to_exact_full_layout(
                pair_index, before, after, raw, proposed)
            if projection_note is not None and projected.delta_px is not None:
                candidates.add(projected.delta_px)
    # 0 is admissible HERE and nowhere else. A page that has hit its bottom while a video keeps
    # playing produces a run of genuine 0px pairs, and excluding them did not merely skip those
    # pairs -- it broke the track permanently, because `_video_track_deltas` drops a card the
    # moment one pair offers no candidate and never re-seeds. Live 2026-08-16 (Grace) that killed
    # the mute-card track at the first static frame, discarding the identity of the card for the
    # whole rest of the profile. Letting a track CONTINUE across "it did not move" cannot invent
    # a shift: `_layout_repaired_shift` and `_exact_multi_strip_shift` both still refuse to
    # propose a non-positive candidate, so a 0 link can only ever confirm a pair frameshift
    # already measured as 0 and can never repair one it refused.
    return tuple(sorted(delta for delta in candidates if 0 <= delta <= _MAX_STEP_PX))


def _track_anchor_count(before, after, delta: int, *,
                        before_marker: VideoMuteMarker | None,
                        after_marker: VideoMuteMarker | None) -> int:
    """Count independent card-local anchors that translate by one exact delta."""
    anchors = 0
    for name, before_y, after_y in (
            ("top", before.top.y, after.top.y),
            ("bottom", before.bottom.y, after.bottom.y),
            ("heart", before.heart[1] if before.heart else None,
             after.heart[1] if after.heart else None)):
        if (before_y is not None and after_y is not None
                and ((name == "top" and before.top.observed and after.top.observed)
                     or (name == "bottom" and before.bottom.observed and after.bottom.observed)
                     or name == "heart")
                and before_y - after_y == delta):
            anchors += 1
    if (before_marker is not None and after_marker is not None
            and before_marker.y - after_marker.y == delta
            and before_marker.x == after_marker.x):
        anchors += 1
    return anchors


def _direct_marker_bridge_delta(before_segmentation: FrameSegmentation,
                                after_segmentation: FrameSegmentation, *,
                                before_marker: VideoMuteMarker | None,
                                after_marker: VideoMuteMarker | None,
                                ) -> int | None:
    """One direct mute-control bridge across a pair with no usable pixel shift.

    Animated video can make every NCC strip weak even though Hinge's app-owned mute control
    remains visible on the very same physical card.  This is deliberately *not* a replacement
    for ordinary frameshift: it exists only when both unique control sightings have the same x,
    imply one in-range scroll distance, are each contained by a segmented card, and that exact
    distance also moves a distinct card-local anchor.  The marker pins the card identity; the
    second anchor prevents one moving overlay from inventing a page offset by itself.
    """
    if before_marker is None or after_marker is None or before_marker.x != after_marker.x:
        return None
    delta = before_marker.y - after_marker.y
    # A direct-marker bridge is a synthetic repair for an otherwise refused pair, never a
    # replacement for a confirmed static observation.  Zero remains valid only when frameshift
    # itself measured it and `_track_candidate_deltas` carries that raw observation forward.
    if not 0 < delta <= _MAX_STEP_PX:
        return None
    before_block = _marker_block(before_segmentation, before_marker)
    after_block = _marker_block(after_segmentation, after_marker)
    if before_block is None or after_block is None:
        return None
    if _track_anchor_count(
            before_block, after_block, delta,
            before_marker=before_marker, after_marker=after_marker) < 2:
        return None
    return delta


def _video_exit_bridge_delta(raw: ShiftEstimate) -> int | None:
    """The one high-specificity strip value permitted immediately after a tracked exit.

    Once a contained mute control has affirmatively left the content band, the next frame may
    expose the cards below it while the video is still repainting.  Three *identical* NCC strips
    are enough to nominate a distance only at that one boundary; the caller additionally requires
    the already-tracked card's heart to move by it and a unique destination card.  This never
    weakens the ordinary frameshift quorum for an untracked pair.
    """
    if raw.status != SHIFT_NO_CONSENSUS:
        return None
    matched = tuple(strip.delta_px for strip in raw.strips
                    if strip.state == STRIP_MATCHED and strip.delta_px is not None)
    candidates = tuple(sorted(
        delta for delta in set(matched)
        if 0 < delta <= _MAX_STEP_PX and matched.count(delta) >= 3))
    return candidates[0] if len(candidates) == 1 else None


def _heart_moves_by(before, after, delta: int) -> bool:
    """One exact same-card heart anchor for the post-video exit boundary."""
    return bool(
        before.heart is not None and after.heart is not None
        and abs(before.heart[0] - after.heart[0]) <= 2
        and before.heart[1] - after.heart[1] == delta)


def _unique_markers_by_frame(markers: Sequence[VideoMuteMarker], frame_count: int,
                             ) -> dict[int, VideoMuteMarker]:
    """Keep only one positioned mute observation per valid frame.

    A second match is ambiguity, not corroboration: without an app-provided card id there is no
    safe way to decide which control should carry the video identity forward.
    """
    rows: dict[int, VideoMuteMarker | None] = {}
    for marker in markers:
        if not (0 <= marker.frame_index < frame_count):
            continue
        if marker.frame_index in rows:
            rows[marker.frame_index] = None
        else:
            rows[marker.frame_index] = marker
    return {frame_index: marker for frame_index, marker in rows.items() if marker is not None}


def _video_track_deltas(segmentations: Sequence[FrameSegmentation],
                        shifts: Sequence[ShiftEstimate],
                        markers: Sequence[VideoMuteMarker], *,
                        extent_tolerance_px: int) -> dict[int, int]:
    """Return adjacent pair -> exact delta for uniquely tracked physical video cards.

    A mute hit seeds a card, then a track may continue after that overlay scrolls out of frame
    only through two exact card-local anchors. Between two visible, uniquely contained mute hits,
    one raw refusal (``no_evidence`` or ``no_consensus``) may instead use their exact direct
    displacement plus a second card-local anchor. Immediately after a proved exit, one unique
    three-strip/heart boundary may
    rejoin cards below the video. It cannot jump to the next card merely because a frame contains
    *some* video. Ambiguity removes authority rather than choosing an identity.
    """
    by_frame = _unique_markers_by_frame(markers, len(segmentations))

    seeded: dict[int, object] = {}
    marker_blocks: dict[int, object] = {}
    for frame_index, marker in by_frame.items():
        block = _marker_block(segmentations[frame_index], marker)
        if block is not None:
            seeded[frame_index] = block
            marker_blocks[frame_index] = marker

    # All later mute hits are observations that must agree with this one physical trajectory;
    # they are not independent repair seeds.  Otherwise a broken middle link could silently
    # start a second "same video" track below it.
    first_seed = min(seeded, default=None)
    active: set[tuple[int, object]] = (
        {(first_seed, seeded[first_seed])} if first_seed is not None else set())
    # Frame indices whose active card has just been proved to leave the band.  Exactly the next
    # unmarked pair may re-establish the lower coordinate space with `_video_exit_bridge_delta`.
    # The set is intentionally not propagated: allowing a general post-video grammar would make
    # later noisy cards look related merely because an earlier card happened to be a video.
    pending_exit_frames: set[int] = set()
    links: dict[int, int] = {}
    for pair_index, (before_seg, after_seg, raw) in enumerate(
            zip(segmentations[:-1], segmentations[1:], shifts, strict=True)):
        current = tuple(block for frame, block in active if frame == pair_index)
        if not current:
            continue
        before_marker = marker_blocks.get(pair_index)  # only contained, unambiguous controls
        after_marker = marker_blocks.get(pair_index + 1)
        direct_marker_delta = _direct_marker_bridge_delta(
            before_seg, after_seg, before_marker=before_marker, after_marker=after_marker)
        exit_delta = (_video_exit_bridge_delta(raw)
                      if (pair_index in pending_exit_frames
                          and before_marker is None and after_marker is None) else None)
        candidates: list[tuple[object, object, int]] = []
        for before in current:
            deltas = set(_track_candidate_deltas(
                    pair_index, before_seg, after_seg, raw,
                    extent_tolerance_px=extent_tolerance_px))
            # A raw refusal has no trusted strip candidate. A direct marker bridge is the sole
            # exception, and only when it continues the very card that is already active; a
            # later marker may never re-seed a broken trajectory below the video.
            if (raw.status in (SHIFT_NO_EVIDENCE, SHIFT_NO_CONSENSUS)
                    and direct_marker_delta is not None
                    and before is _marker_block(before_seg, before_marker)):
                deltas.add(direct_marker_delta)
            if exit_delta is not None:
                deltas.add(exit_delta)
            # A marker disappearing while its predicted origin remains inside the next content
            # band invalidates the WHOLE track, including earlier repairs.  Otherwise a later
            # missed UI match could leave a trusted prefix that falsely connects two cards.
            if (before_marker is not None and after_marker is None
                    and any(before_marker.y - delta >= after_seg.band[0] for delta in deltas)):
                return {}
            for delta in sorted(deltas):
                # A control may disappear only by leaving the analysed band.  If its predicted
                # origin is still visible, a missing after-hit is a broken identity chain, not
                # permission to follow whichever lower card geometry happens to fit.
                # On a direct bridge the after marker identifies the card, so do not let its
                # one extra anchor make a similarly-shaped neighbour look interchangeable.
                is_direct_bridge = (raw.status in (SHIFT_NO_EVIDENCE, SHIFT_NO_CONSENSUS)
                                    and delta == direct_marker_delta)
                after_pool = ((_marker_block(after_seg, after_marker),)
                              if is_direct_bridge else after_seg.blocks)
                is_exit_bridge = delta == exit_delta
                matches = tuple(
                    after for after in after_pool if after is not None
                    if (_heart_moves_by(before, after, delta) if is_exit_bridge else
                        _track_anchor_count(
                            before, after, delta,
                            before_marker=before_marker,
                            after_marker=after_marker) >= 2))
                if len(matches) == 1:
                    candidates.append((before, matches[0], delta))
        # A direct, single-card continuation is the only authority.  Multiple valid mappings
        # are exactly the situation in which treating animation as a repair would be a guess.
        if len(candidates) != 1:
            continue
        _before, after, delta = candidates[0]
        active.add((pair_index + 1, after))
        links[pair_index] = delta
        if before_marker is not None and after_marker is None:
            # The early return above has already proved this control's expected next origin is
            # above the band, not merely missing from a visible place.
            pending_exit_frames.add(pair_index + 1)
    return links


def _repair_video_track_shifts(segmentations: Sequence[FrameSegmentation],
                               shifts: Sequence[ShiftEstimate],
                               markers: Sequence[VideoMuteMarker], *,
                               extent_tolerance_px: int,
                               ) -> tuple[tuple[ShiftEstimate, ...], tuple[str, ...],
                                          tuple[tuple[int, ShiftEstimate], ...]]:
    """Repair only strictly tracked video pairs, then require a whole-page rebuild."""
    links = _video_track_deltas(segmentations, shifts, markers,
                                extent_tolerance_px=extent_tolerance_px)
    markers_by_frame = _unique_markers_by_frame(markers, len(segmentations))
    repaired = list(shifts)
    notes: list[str] = []
    raw_provenance: list[tuple[int, ShiftEstimate]] = []
    for pair_index, delta in sorted(links.items()):
        raw = shifts[pair_index]
        before, after = segmentations[pair_index], segmentations[pair_index + 1]
        direct_delta = _direct_marker_bridge_delta(
            before, after,
            before_marker=markers_by_frame.get(pair_index),
            after_marker=markers_by_frame.get(pair_index + 1))
        if (raw.status in (SHIFT_NO_CONSENSUS, SHIFT_NO_EVIDENCE)
                and direct_delta == delta):
            # Two contained controls identify the same physical card across an otherwise noisy
            # pair. This can corroborate a raw no-consensus result just as it can bridge an
            # all-weak one; raw NCC evidence remains recorded rather than being relabelled.
            repaired[pair_index] = replace(
                raw, status=SHIFT_MEASURED, delta_px=delta, consensus_px=delta,
                saturated=False, confidence=1.0, agreeing=2, dissenting=0, eligible=2,
                reason=("v12 mute-card direct marker bridge: positioned app control plus "
                        "one exact card-local anchor; raw strips retained in "
                        "layout_repaired_shifts"))
        elif raw.status == SHIFT_NO_CONSENSUS and _video_exit_bridge_delta(raw) == delta:
            # A link can reach this branch only after the prior pair proved the mute control
            # physically exited the band. Recheck that complete, local grammar here instead of
            # inferring it from the repaired list, so a future caller cannot invoke this helper
            # with an arbitrary three-strip refusal and receive a synthetic shift.
            previous = pair_index - 1
            previous_marker = markers_by_frame.get(previous)
            if (previous < 0 or previous_marker is None
                    or markers_by_frame.get(pair_index) is not None
                    or markers_by_frame.get(pair_index + 1) is not None
                    or previous not in links
                    or previous_marker.y - links[previous] >= after.band[0]):
                continue
            repaired[pair_index] = replace(
                raw, status=SHIFT_MEASURED, delta_px=delta, consensus_px=delta,
                saturated=False, agreeing=3,
                dissenting=max(0, raw.eligible - 3),
                confidence=(3 / raw.eligible if raw.eligible else 0.0),
                reason=("v12 mute-card post-exit bridge: three exact NCC strips plus the "
                        "tracked card's exact heart anchor; raw no-consensus strips retained "
                        "in layout_repaired_shifts"))
        elif raw.status == SHIFT_NO_CONSENSUS:
            proposed, note = _layout_repaired_shift(
                pair_index, before, after, raw, extent_tolerance_px=extent_tolerance_px)
            if note is not None:
                projected, projection_note = _project_to_exact_full_layout(
                    pair_index, before, after, raw, proposed)
                if projection_note is not None:
                    proposed = projected
                    note = f"{note}; {projection_note}"
            if note is None or proposed.delta_px != delta:
                continue
            repaired[pair_index] = proposed
        elif raw.status == SHIFT_MEASURED and raw.delta_px != delta:
            proposed, note = _measured_layout_bridge(
                pair_index, before, after, raw, allow_full_layout_projection=True,
                extent_tolerance_px=extent_tolerance_px)
            if note is None or proposed.delta_px != delta:
                continue
            repaired[pair_index] = proposed
        else:
            continue
        raw_provenance.append((pair_index, raw))
        raw_value = raw.delta_px if raw.delta_px is not None else raw.status
        notes.append(
            f"frame {pair_index}'s pair with frame {pair_index + 1}: v12 mute-card track "
            f"fixed the physical video card at +{delta}px; raw {raw.status} "
            f"{raw_value} is retained for "
            "audit and complete-page assembly must still pass"
            + (f"; video-marked measured bridge projected to +{delta}px"
               if raw.status == SHIFT_MEASURED else ""))
    return tuple(repaired), tuple(notes), tuple(raw_provenance)


def _repair_shifts_from_layout(segmentations: Sequence[FrameSegmentation],
                               shifts: Sequence[ShiftEstimate], *,
                               extent_tolerance_px: int = _EXTENT_TOLERANCE_PX,
                               animation_markers: Sequence[bool] = (),
                               ) -> tuple[tuple[ShiftEstimate, ...], tuple[str, ...],
                                          tuple[tuple[int, ShiftEstimate], ...]]:
    """Propose one bounded animation run, never an isolated layout repair.

    A real animation affects neighbouring capture pairs, whereas a coincidental strip match is
    normally isolated.  Exactly one maximal run of two or three ordinary candidates is therefore
    the only generally admissible shape, and it must include a raw no-consensus pair.  A one-strip
    candidate may accompany that run but cannot bootstrap it: at least one candidate must carry
    the ordinary two-strip layout corroboration.

    One exact four-pair continuation is narrower still: two ordinary two-strip repairs, one exact
    one-strip repair, then a measured structural-only tail beyond the normal override ceiling.
    The tail cannot occur anywhere else or validate a shorter/otherwise shaped run.

    One five-pair refusal island has a different, fully bracketed grammar: exact 3+-strip/full-
    landmark candidates close both ends around the existing 2-strip, 2-strip, 1-strip sequence;
    raw measured pairs must sit immediately outside it.  Its one heartless transition must carry
    an exact shared gutter.  This does not raise the generic three-pair limit.

    A bare two-pair window of exact 3+-strip/full-landmark candidates, bracketed the same way by
    raw measured pairs, needs none of that interior sequence: with no interior pair left to
    bridge, both boundary candidates already carry more independent per-pair evidence than the
    five-pair island asks of even its own endpoints.  This is the tail of a scrolling video card:
    each pair it touches gets its own exact top/bottom/heart landmark corroboration, so there is
    nothing left for a weaker interior grammar to add.

    Ordinary two-strip proposals separated by one or two raw measured pairs are one animation
    window rather than unrelated runs only when the app supplied an affirmative animation marker,
    the whole span is at most five pairs, every gap and both immediate outer anchors have raw
    quorum plus exact multi-kind layout, and the repair endpoints include both exact full layout
    and an exact heartless shared-gutter transition. One measured gap may project within the
    existing fold tolerance only onto unique exact top/bottom/heart geometry carrying an exact
    NCC witness. The complete page probe remains the final authority.

    The edge-only helper is weaker and cannot participate in that general rule.  It is admitted
    only in the exact saved three-pair sequence: leading no-consensus two-strip/shared-gutter;
    middle measured-majority override with two strips and full top/bottom/heart layout; trailing
    no-consensus one-strip repair with full layout.  The caller still runs a complete ordinary
    page assembly before committing either proposal.
    """
    repaired: list[ShiftEstimate] = []
    notes: list[str | None] = []
    edge_only_indices: list[int] = []
    structural_tail_indices: list[int] = []
    exact_multi_indices: list[int] = []
    for i, (before, after, shift) in enumerate(zip(segmentations[:-1], segmentations[1:], shifts,
                                                     strict=True)):
        result, note = _layout_repaired_shift(i, before, after, shift,
                                              extent_tolerance_px=extent_tolerance_px)
        if note is None:
            result, note = _edge_only_two_strip_shift(i, before, after, shift)
            if note is not None:
                edge_only_indices.append(i)
        if note is None:
            result, note = _structural_tail_shift(i, before, after, shift)
            if note is not None:
                structural_tail_indices.append(i)
        if note is None:
            result, note = _exact_multi_strip_shift(i, before, after, shift)
            if note is not None:
                exact_multi_indices.append(i)
        repaired.append(result)
        notes.append(note)

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, note in enumerate(notes + [None]):
        if note is not None and start is None:
            start = i
        elif note is None and start is not None:
            runs.append((start, i))
            start = None
    window_notes = list(notes)
    measured_bridge_indices: tuple[int, ...] = ()
    projected_bridge_indices: tuple[int, ...] = ()
    if len(runs) != 1:
        # Live video can briefly leave enough stable pixels for an ordinary raw measurement,
        # producing two repair-note islands even though card/heart geometry proves one physical
        # transition.  Merge only one short, fully bracketed window of ORDINARY layout proposals;
        # none of the weaker edge-only, structural-tail, or exact-multi exceptions may borrow
        # this path.  At least two proposals are required, so a measured neighbour can never
        # bootstrap an isolated two-strip coincidence.
        proposal_indices = tuple(i for i, note in enumerate(notes) if note is not None)
        if (edge_only_indices or structural_tail_indices or exact_multi_indices
                or not 2 <= len(proposal_indices) <= 3):
            return tuple(shifts), (), ()
        start, end = proposal_indices[0], proposal_indices[-1] + 1
        measured_bridge_indices = tuple(
            i for i in range(start, end) if notes[i] is None)
        if (end - start > 5 or not 1 <= len(measured_bridge_indices) <= 2
                or start == 0 or end >= len(shifts)
                or any(shifts[i].status != SHIFT_NO_CONSENSUS
                       or shifts[i].agreeing != 2 or shifts[i].eligible != 2
                       or shifts[i].dissenting != 0
                       or sum(1 for strip in shifts[i].strips
                              if strip.state == STRIP_MATCHED) != 2
                       for i in proposal_indices)):
            return tuple(shifts), (), ()
        if (len(animation_markers) != len(segmentations)
                or not any(bool(animation_markers[i]) for i in range(start, end + 1))):
            # This broader connection of otherwise independent repair islands exists only for
            # an app-confirmed animated/video window.  Layout alone continues to use the older
            # contiguous-run grammars; absence of a marker is absence of authority.
            return tuple(shifts), (), ()
        anchor_indices = (start - 1, *measured_bridge_indices, end)
        bridge_notes: dict[int, str] = {}
        projected_bridges: list[int] = []
        for i in anchor_indices:
            bridge_shift, bridge_note = _measured_layout_bridge(
                i, segmentations[i], segmentations[i + 1], shifts[i],
                allow_full_layout_projection=(i in measured_bridge_indices),
                extent_tolerance_px=extent_tolerance_px)
            if bridge_note is None:
                return tuple(shifts), (), ()
            if i in measured_bridge_indices:
                repaired[i] = bridge_shift
                bridge_notes[i] = bridge_note
                if bridge_shift.delta_px != shifts[i].delta_px:
                    projected_bridges.append(i)
            elif bridge_shift is not shifts[i]:
                # Outer anchors must remain raw measurements.  Their job is to bound the window,
                # not borrow repair authority from it.
                return tuple(shifts), (), ()
        projected_bridge_indices = tuple(projected_bridges)

        full_layout = 0
        shared_gutter = 0
        for i in proposal_indices:
            projected, projection_note = _project_to_exact_full_layout(
                i, segmentations[i], segmentations[i + 1], shifts[i], repaired[i])
            repaired[i] = projected
            if projection_note is not None:
                notes[i] = f"{notes[i]}; {projection_note}"
            delta = repaired[i].delta_px
            exact_kinds = {
                kind for kind, landmark_delta in _structural_landmarks(
                    segmentations[i], segmentations[i + 1])
                if delta is not None and landmark_delta == delta
            }
            if exact_kinds == {"top", "bottom", "heart"}:
                full_layout += 1
            elif (exact_kinds == {"top", "bottom"} and delta is not None
                    and len(_shared_gutter_witnesses(
                        segmentations[i], segmentations[i + 1], delta)) == 1):
                shared_gutter += 1
            else:
                return tuple(shifts), (), ()
        if full_layout < 1 or shared_gutter < 1:
            return tuple(shifts), (), ()
        for i, bridge_note in bridge_notes.items():
            window_notes[i] = bridge_note
        for i in proposal_indices:
            window_notes[i] = notes[i]
        runs = [(start, end)]
    start, end = runs[0]
    max_run = 5 if (exact_multi_indices or measured_bridge_indices) \
        else (4 if structural_tail_indices else 3)
    if not 2 <= end - start <= max_run:
        return tuple(shifts), (), ()
    indices = tuple(range(start, end))
    if not any(shifts[i].status == SHIFT_NO_CONSENSUS for i in indices):
        return tuple(shifts), (), ()
    two_pair_exact_multi_window = False
    if exact_multi_indices:
        # Two admissible shapes share this evidence kind.
        #
        # The minimal one is a bare two-pair window with no interior pair at all: both pairs in
        # it already independently cleared `_exact_multi_strip_shift`'s own bar (3+ exact-
        # matching NCC strips plus exact top, bottom and heart geometry, all on the same
        # candidate).  That is strictly more per-pair evidence than the five-pair island below
        # asks of even its own two endpoints, so admitting it needs nothing beyond an ordinary
        # measured bracket on both sides -- there is no weaker interior pair whose corroboration
        # could be in question, unlike the five-pair shape this grammar was first built for.
        #
        # The five-pair island is the saved live-video transition: two exact-multi endpoints
        # bracket a specific interior grammar -- two strips with one exact shared gutter while
        # the old heart leaves, then two strips with full layout, then one exact strip with full
        # layout -- and ordinary measured shifts must bracket the whole island on both sides.
        common_guard = (
            edge_only_indices or structural_tail_indices
            or tuple(exact_multi_indices) != (start, end - 1)
            or start == 0 or end >= len(shifts)
            or shifts[start - 1].status != SHIFT_MEASURED
            or shifts[end].status != SHIFT_MEASURED
            or repaired[start].agreeing < 3 or repaired[end - 1].agreeing < 3)
        if end - start == 2:
            if common_guard:
                return tuple(shifts), (), ()
            for i in indices:
                delta = repaired[i].delta_px
                exact_kinds = {
                    kind for kind, landmark_delta in _structural_landmarks(
                        segmentations[i], segmentations[i + 1])
                    if delta is not None and landmark_delta == delta
                }
                if exact_kinds != {"top", "bottom", "heart"}:
                    return tuple(shifts), (), ()
            two_pair_exact_multi_window = True
        else:
            if (common_guard or end - start != 5
                    or any(shifts[i].status != SHIFT_NO_CONSENSUS for i in indices)
                    or tuple(repaired[i].agreeing for i in indices[1:-1]) != (2, 2, 1)):
                return tuple(shifts), (), ()
            for position, i in enumerate(indices):
                delta = repaired[i].delta_px
                exact_kinds = {
                    kind for kind, landmark_delta in _structural_landmarks(
                        segmentations[i], segmentations[i + 1])
                    if delta is not None and landmark_delta == delta
                }
                if position == 1:
                    if (exact_kinds != {"top", "bottom"} or delta is None
                            or len(_shared_gutter_witnesses(
                                segmentations[i], segmentations[i + 1], delta)) != 1):
                        return tuple(shifts), (), ()
                elif exact_kinds != {"top", "bottom", "heart"}:
                    return tuple(shifts), (), ()
    if edge_only_indices:
        # Do not generalise two edges into a third independent landmark.  The saved failure is
        # safe only as this one fully bracketed transition: the middle pair corrects a measured
        # majority with the ordinary two-strip/full-layout rule, and the trailing pair is the
        # sole-strip/exact-full-layout rule.  Neither weak pair can bootstrap the other.
        if (tuple(edge_only_indices) != (start,) or end - start != 3
                or tuple(shifts[i].status for i in indices) != (
                    SHIFT_NO_CONSENSUS, SHIFT_MEASURED, SHIFT_NO_CONSENSUS)
                or tuple(repaired[i].agreeing for i in indices) != (2, 2, 1)):
            return tuple(shifts), (), ()
        for i in indices[1:]:
            delta = repaired[i].delta_px
            corroborating_kinds = {
                kind for kind, landmark_delta in _structural_landmarks(
                    segmentations[i], segmentations[i + 1])
                if delta is not None
                and abs(landmark_delta - delta) <= _AGREEMENT_TOLERANCE_PX
            }
            if corroborating_kinds != {"top", "bottom", "heart"}:
                return tuple(shifts), (), ()
    if structural_tail_indices:
        # A structure-only answer never gains ordinary authority.  It is the terminal cross-check
        # for exactly the saved four-pair video transition, whose three preceding pairs have
        # progressively 2, 2, and 1 NCC witnesses.  The last two of those and the tail must all
        # carry exact full landmark geometry; the first already passed the ordinary multi-edge
        # layout gate.  Any other placement, status grammar, or evidence strength remains raw.
        if (edge_only_indices or tuple(structural_tail_indices) != (end - 1,)
                or end - start != 4
                or tuple(shifts[i].status for i in indices) != (
                    SHIFT_NO_CONSENSUS, SHIFT_NO_CONSENSUS,
                    SHIFT_NO_CONSENSUS, SHIFT_MEASURED)
                or tuple(repaired[i].agreeing for i in indices[:3]) != (2, 2, 1)):
            return tuple(shifts), (), ()
        for i in indices[1:]:
            delta = repaired[i].delta_px
            exact_kinds = {
                kind for kind, landmark_delta in _structural_landmarks(
                    segmentations[i], segmentations[i + 1])
                if delta is not None and landmark_delta == delta
            }
            if exact_kinds != {"top", "bottom", "heart"}:
                return tuple(shifts), (), ()
    if not two_pair_exact_multi_window and not any(repaired[i].agreeing == 2 for i in indices):
        # Two adjacent lone-strip coincidences are not independent evidence for one another.
        # The one-strip exception is a companion to an ordinary two-strip layout repair only.
        # A two-pair exact-multi window has no lone-strip repair to begin with -- both pairs
        # already cleared the stronger 3+-exact-strip bar above -- so this guard does not apply.
        return tuple(shifts), (), ()
    return (tuple(repaired),
            tuple(note for note in window_notes[start:end] if note is not None),
            tuple((i, shifts[i]) for i in indices
                  if notes[i] is not None or i in projected_bridge_indices))


def _frame_offsets(shifts: Sequence[ShiftEstimate]) -> tuple[list[int | None], list[str]]:
    """Each frame's page offset, and a failure for the first pair that broke the chain.

    Page coordinates are anchored on frame 0 (`offsets[0] == 0`), and content at row `y` of frame
    `i` is at page row `y + offsets[i]`. The recurrence is `offsets[i + 1] = offsets[i] + delta`,
    straight from frameshift's sign convention: a positive delta means content at row `y` of the
    earlier frame sits at `y - delta` in the later one, so the later frame's window has moved
    `delta` further DOWN the page.

    Once a pair refuses, every later offset is None and stays None. There is no re-anchoring on
    the next measurable pair: two disjoint coordinate spaces cannot be laid end to end without
    assuming the gap between them, and assuming the step is what doc 5.10 measured fabricating a
    phantom item.
    """
    offsets: list[int | None] = [0]
    failures: list[str] = []
    for i, est in enumerate(shifts):
        current = offsets[-1]
        if current is None:
            offsets.append(None)
            continue
        if est.delta_px is None:
            offsets.append(None)
            # `consensus_px` is quoted when there is one because it is the difference between
            # "scroll again, smaller" and "these frames do not correspond, stop" — see
            # frameshift's SHIFT_BEYOND_WINDOW. It is NEVER used as a delta.
            measured = ("" if est.consensus_px is None
                        else f" (the strips do agree on {est.consensus_px:+d}px, which is only a "
                             "magnitude and may not be used as a shift)")
            failures.append(
                f"frames {i} and {i + 1} could not be put in one coordinate space: "
                f"{est.status} — {est.reason}{measured}. Without it there is no way to tell a "
                "heart that moved from a new heart that arrived, which is exactly how doc 5.10's "
                "phantom item was fabricated, so no item may be indexed from this capture")
            continue
        offsets.append(current + est.delta_px)
    return offsets, failures


def _observations(segmentations: Sequence[FrameSegmentation],
                  offsets: Sequence[int | None]) -> list[BlockObservation]:
    """Every frame's blocks, lifted into the shared page space. Frames with no offset contribute
    nothing — they are the frames past a broken chain, whose page position is unknown."""
    out: list[BlockObservation] = []
    for i, (seg, offset) in enumerate(zip(segmentations, offsets, strict=True)):
        if offset is None:
            continue
        out.extend(
            BlockObservation(
                frame_index=i,
                page_y0=block.y0 + offset, page_y1=block.y1 + offset,
                frame_y0=block.y0, frame_y1=block.y1,
                kind=block.kind, complete=block.complete,
                top_observed=block.top.observed, bottom_observed=block.bottom.observed,
                hearts=tuple((x, y + offset) for x, y in block.hearts))
            for block in seg.blocks
        )
    return out


def _split_repeated_near_gutter_merges(
        observations: Sequence[BlockObservation],
        near_gutters: Sequence[tuple[int, int, int]], *, tolerance: int,
        ) -> tuple[tuple[BlockObservation, ...], tuple[str, ...]]:
    """Restore a missed 59..64px gutter only when the capture itself proves it.

    ``segment.py`` deliberately does not treat every slightly-long page-coloured span as a
    boundary: a blank region inside a pale card is safer merged than silently split.  The
    incident behind this repair supplied the extra evidence a single frame lacks: the same
    near-gutter recurred in several aligned frames, a complete card ended at its upper edge, and
    another frame saw the next fragment begin at the ordinary canonical-gutter distance.  That
    combination proves a card boundary without turning the generic per-frame rule permissive.

    ``near_gutters`` carries ``(frame_index, page_y0, page_y1)`` for only those repeatedly
    measured `RUN_TOO_LONG` spans.  Virtual pieces remain partial -- this pass only prevents a
    known boundary from merging two cards; it never claims to have observed either new edge.
    """
    if not observations or not near_gutters:
        return tuple(observations), ()

    # Cluster the repeated page-space runs.  Their starts are the structural fact used below;
    # a same-card blank span that drifts with a misread frame cannot establish a boundary.
    clusters: list[list[tuple[int, int, int]]] = []
    for run in sorted(near_gutters, key=lambda value: (value[1], value[2], value[0])):
        if clusters and abs(run[1] - clusters[-1][0][1]) <= tolerance:
            clusters[-1].append(run)
        else:
            clusters.append([run])

    gutter_low = min(_GUTTER_PX) - _GUTTER_TOLERANCE_PX
    gutter_high = max(_GUTTER_PX) + _GUTTER_TOLERANCE_PX
    boundaries: list[tuple[int, int, tuple[int, ...]]] = []
    for cluster in clusters:
        frame_indices = tuple(sorted({frame for frame, _y0, _y1 in cluster}))
        if len(frame_indices) < 2:
            continue
        run_y0 = sorted(y0 for _frame, y0, _y1 in cluster)[len(cluster) // 2]
        upper_ends = sorted(
            obs.page_y1 for obs in observations
            if obs.complete and abs(obs.page_y1 - run_y0) <= tolerance)
        if not upper_ends:
            continue
        upper_end = upper_ends[len(upper_ends) // 2]
        lower_starts = sorted(
            obs.page_y0 for obs in observations
            if gutter_low <= obs.page_y0 - upper_end <= gutter_high)
        if not lower_starts:
            continue
        lower_start = lower_starts[len(lower_starts) // 2]
        # A heart in the unmodelled rows would make the virtual split discard selectable
        # evidence.  Leave that capture untouched so the ordinary loud refusal remains.
        if any(
                obs.page_y0 < upper_end and obs.page_y1 > lower_start
                and any(upper_end <= y < lower_start for _x, y in obs.hearts)
                for obs in observations):
            continue
        boundaries.append((upper_end, lower_start, frame_indices))

    if not boundaries:
        return tuple(observations), ()

    repaired: list[BlockObservation] = []
    used: list[tuple[int, int, tuple[int, ...]]] = []
    for obs in observations:
        # Complete sightings are the independent evidence that licensed the repair; changing
        # one would convert a measurement into an inference.  Split only a partial that spans
        # both proven sides of the boundary.
        boundary = next((candidate for candidate in boundaries
                         if not obs.complete
                         and obs.page_y0 < candidate[0]
                         and obs.page_y1 > candidate[1]), None)
        if boundary is None:
            repaired.append(obs)
            continue
        upper_end, lower_start, frame_indices = boundary
        split_at_upper = obs.frame_y0 + (upper_end - obs.page_y0)
        split_at_lower = obs.frame_y0 + (lower_start - obs.page_y0)
        repaired.extend((
            replace(obs, page_y1=upper_end, frame_y1=split_at_upper,
                    bottom_observed=False,
                    hearts=tuple((x, y) for x, y in obs.hearts if y < upper_end)),
            replace(obs, page_y0=lower_start, frame_y0=split_at_lower,
                    top_observed=False,
                    hearts=tuple((x, y) for x, y in obs.hearts if y >= lower_start)),
        ))
        if boundary not in used:
            used.append(boundary)

    notes = tuple(
        f"repeated {min(_HEART_SEPARATED_NEAR_GUTTER_PX)}.."
        f"{max(_HEART_SEPARATED_NEAR_GUTTER_PX)}px page-background run at the complete "
        f"card end {upper_end} in frames {list(frame_indices)}, with a canonical-gutter "
        f"lower fragment beginning at {lower_start}; spanning partial sightings were split "
        "without claiming either virtual edge was observed"
        for upper_end, lower_start, frame_indices in used)
    return tuple(repaired), notes


def _overlap_groups(observations: Sequence[BlockObservation]) -> list[list[BlockObservation]]:
    """Sightings folded into groups by page-position OVERLAP: one group per physical block.

    Overlap needs no tolerance and gets none. Two DISTINCT blocks are at least `_MIN_ITEM_GAP_PX`
    (47px) apart by construction, so a single shared row already means "the same card", and
    demanding more would only start splitting cards whose visible fragment in one frame is thin.
    Groups CHAIN transitively, which is the correct reading: a card seen through a window that
    slides down the page produces a run of fragments that each overlap only their neighbours.

    The one thing this cannot catch — two fragments of one card that overlap NOTHING, because a
    scroll step jumped clean past the middle of it — is caught afterwards by the minimum-gap
    check in `_assemble`, which is where the fabricated-item guard lives.
    """
    groups: list[list[BlockObservation]] = []
    hulls: list[int] = []
    for obs in sorted(observations, key=lambda o: (o.page_y0, o.page_y1)):
        if groups and obs.page_y0 < hulls[-1]:
            groups[-1].append(obs)
            hulls[-1] = max(hulls[-1], obs.page_y1)
        else:
            groups.append([obs])
            hulls.append(obs.page_y1)
    return groups


def _heart_clusters(observations: Sequence[BlockObservation], *, tolerance: int,
                    ) -> list[tuple[int, int]]:
    """The DISTINCT hearts of one block, `(x, page_y)`, in page order.

    The same heart seen in six frames lands within `tolerance` of itself six times and is one
    heart; two hearts on one block are 738px apart at the very closest measured (doc 5.10.1's
    correction two), so the clustering has three orders of magnitude of room and the count it
    returns is the count that matters. The representative of a cluster is a MEDIAN of the
    positions actually observed rather than a mean, for the same reason `_resolve_group` picks an
    observed extent: the number is used to aim a tap, and it should be somewhere a frame saw a
    heart.
    """
    seen = sorted({h for o in observations for h in o.hearts}, key=lambda h: (h[1], h[0]))
    clusters: list[list[tuple[int, int]]] = []
    for heart in seen:
        if clusters and heart[1] - clusters[-1][-1][1] <= tolerance:
            clusters[-1].append(heart)
        else:
            clusters.append([heart])
    return [sorted(c, key=lambda h: (h[1], h[0]))[len(c) // 2] for c in clusters]


def _resolve_group(group: Sequence[BlockObservation], *, tolerance: int,
                   card_x: tuple[int, int]) -> tuple[IndexedBlock, list[str]]:
    """One group of sightings -> one block, plus whatever the sightings contradict.

    The extent comes from the COMPLETE sightings when there are any, because only they measured
    both edges; the representative is the median one by position, so the reported extent is an
    extent a frame actually observed. Every other complete sighting then has to agree with it,
    and every partial sighting has to fit INSIDE it — a fragment reaching past a bounded card is
    a merged block or a mis-tracked frame, not a wider card.

    With no complete sighting at all the extent is the hull of the fragments, which is a lower
    bound and is labelled `ITEM_PARTIAL`.
    """
    observations = tuple(sorted(group, key=lambda o: (o.frame_index, o.page_y0)))
    failures: list[str] = []
    complete = [o for o in observations if o.complete]
    bounded_by: int | None = None

    if complete:
        rep = sorted(complete, key=lambda o: (o.page_y0, o.page_y1))[len(complete) // 2]
        page_y0, page_y1 = rep.page_y0, rep.page_y1
        bounded_by = rep.frame_index
        for other in complete:
            if (abs(other.page_y0 - page_y0) > tolerance
                    or abs(other.page_y1 - page_y1) > tolerance):
                # Naming the FRAME-LOCAL rows alongside the page rows is what lets an operator go
                # find the actual pixels: page rows are only meaningful relative to this capture's
                # own offsets, frame rows are what is really on screen in `frames[frame_index]`.
                if other.frame_index == rep.frame_index:
                    if other.page_y0 < page_y1 and page_y0 < other.page_y1:
                        # segment.py emits disjoint blocks within one frame.  Overlapping complete
                        # extents from it are therefore a genuine frame-local contradiction, not
                        # the external bridging shape handled by `_split_on_bounded_cards`.
                        origin = (
                            f"both sightings are frame {rep.frame_index}'s own and their "
                            "extents overlap — segment.py's blocks are disjoint within one "
                            "frame, so this is a self-contradictory frame")
                    else:
                        # The confusing live shape: one frame bounded two separate cards, then a
                        # missed gutter in another frame bridged them into this overlap group.
                        origin = (
                            f"both sightings are frame {rep.frame_index}'s own but their "
                            "extents are disjoint: one frame bounded two separate cards, folded "
                            "into this one group by a gutter missed in some OTHER frame — not "
                            f"frame {rep.frame_index} disagreeing with itself")
                else:
                    origin = "both observed both edges, so one of them is wrong"
                failures.append(
                    f"frames {rep.frame_index} and {other.frame_index} disagree about the block "
                    f"at page rows {page_y0}..{page_y1}: {page_y1 - page_y0}px against "
                    f"{other.height}px ({other.page_y0}..{other.page_y1}) — frame "
                    f"{rep.frame_index} saw its own frame rows {rep.frame_y0}..{rep.frame_y1}, "
                    f"frame {other.frame_index} saw frame rows {other.frame_y0}..{other.frame_y1}"
                    f". {origin}; averaging them would produce an extent neither frame saw")
        for other in observations:
            if other.complete:
                continue
            if other.page_y0 < page_y0 - tolerance or other.page_y1 > page_y1 + tolerance:
                if other.frame_index == rep.frame_index:
                    # Segment.py's blocks are disjoint within one frame, so a frame contributing
                    # both the bounding sighting and an overrunning fragment to one group should
                    # not happen — spelled out rather than left to read as an ordinary cross-frame
                    # gutter miss, because it is a different and more surprising kind of fault.
                    origin = (
                        f"both are frame {rep.frame_index}'s own sightings, which should never "
                        "happen — segment.py's blocks are disjoint within one frame, so this is "
                        "not an ordinary missed gutter but a self-contradictory frame")
                else:
                    origin = "either a gutter was missed or these two sightings are not the same block"
                failures.append(
                    f"frame {other.frame_index} sees page rows {other.page_y0}..{other.page_y1} "
                    f"(its own frame rows {other.frame_y0}..{other.frame_y1}) where the block was "
                    f"bounded at {page_y0}..{page_y1} by frame {rep.frame_index} (frame rows "
                    f"{rep.frame_y0}..{rep.frame_y1}) — a fragment cannot reach past the card "
                    f"that contains it, so {origin}")
    else:
        page_y0 = min(o.page_y0 for o in observations)
        page_y1 = max(o.page_y1 for o in observations)

    hearts = tuple(_heart_clusters(observations, tolerance=tolerance))
    seen_in = f"seen in frame(s) {sorted({o.frame_index for o in observations})}"

    if len(hearts) > 1:
        kind = ITEM_AMBIGUOUS
        reason = (f"{len(hearts)} distinct like hearts at page rows "
                  f"{[y for _, y in hearts]} inside one block — every likeable Hinge item "
                  f"carries exactly one, so a gutter between them was missed; {seen_in}")
        failures.append(f"block at page rows {page_y0}..{page_y1} is ambiguous: {reason}")
    elif not complete:
        kind = ITEM_PARTIAL
        reason = (f"no frame observed both edges, so {page_y0}..{page_y1} is the hull of "
                  f"{len(observations)} fragment(s) and a lower bound on the real card; "
                  f"{len(hearts)} heart(s); {seen_in}")
    elif hearts:
        kind = ITEM_SELECTABLE
        reason = (f"bounded at {page_y0}..{page_y1} by frame {bounded_by} with exactly one "
                  f"like heart at page row {hearts[0][1]}; {seen_in}")
    else:
        kind = ITEM_CONTEXT
        reason = (f"bounded at {page_y0}..{page_y1} by frame {bounded_by} and no like "
                  f"heart — a context block, read and referenced but never selectable; {seen_in}")

    block = IndexedBlock(
        page_y0=page_y0, page_y1=page_y1, x0=card_x[0], x1=card_x[1],
        kind=kind, hearts=hearts, heart_ordinal=None, model_index=None,
        observations=observations, reason=reason)
    return block, failures


# A heartless block whose extent was never bounded may be hiding a heart in the rows that were
# never inside the analysed band, and a hidden heart shifts every ordinal BELOW it by one. So the
# test is positional: it only fires when something heart-bearing is actually below it, which
# makes the truncated tail of a capture a coverage cost (reported via `ItemIndex.partial` and
# `truncated`) rather than a hard stop, and the middle of a page a hard stop.
_UNCERTAIN_HEART_NOTE = (
    "was never observed complete and shows no like heart, so a heart may sit in the rows that "
    "were never inside the analysed band; every heart ordinal below it would then be wrong")


def _scroll_top_evidence(topmost: IndexedBlock, band_y0: int) -> bool:
    """Whether anything actually SAW page background above this capture's topmost content.

    `at_scroll_top` is an assertion the caller makes and this module cannot verify — segment.py's
    docstring is explicit that nothing in a single frame's geometry distinguishes "the header is
    above card 1" from "we scrolled past it". But the assertion has one geometric CONSEQUENCE
    that can be contradicted, and a validation pass showed why contradicting it matters: told
    `at_scroll_top=True` about a capture that started three frames down a profile, the index came
    back usable with eight items, translation 1..8, `truncated` False and ZERO failures, and its
    model item 1 was really the profile's item 2. Every completeness property corroborated the
    lie rather than staying silent, and a counting navigation would have tapped the wrong card.

    The consequence is this. If frame 0 really is the top of a profile, then Hinge's scroll-top
    header is the topmost content, and it is drawn BELOW the analysed band's first row with page
    background between the two — measured 34px on one calibration profile and 68px on the other,
    both with the shipped `content_band`. If we scrolled past the top, the band's top edge slices
    whatever card is there instead, and the topmost block starts flush ON the band's first row.
    So a topmost block that no frame ever saw with background above it is evidence AGAINST the
    assertion, and it is exactly the shape the false capture above had (its leading fragment sat
    at frame row 300, the band's own first row, against 368 for the same capture's genuine top).

    `any` over the sightings rather than a test on one of them: only the earliest frame can show
    the page's topmost row with anything above it, since every later frame has scrolled that row
    further up, so one qualifying sighting is the whole of the available evidence.

    [corpus: swept over every suffix of the 24-frame bot-scrolled capture — 21 falsely-asserted
    scroll tops, all 21 refused here, none still accepted. Swept over every fifth suffix of the
    57-frame hand-scrolled capture — 10 false tops, 9 refused and ONE accepted, plus one suffix
    that is not a false top at all (that capture opens with a zero-scroll pair, so dropping its
    first frame still starts at the genuine top, and accepting it is the right answer). Both
    genuine scroll tops are unaffected and still index 9 items with translation 1..9.]

    WHAT THIS DOES NOT CATCH, and the reason doc 5.5's filter-chips confirmation stays a hard
    gate rather than becoming optional: a false top whose band edge lands in a gutter, or in any
    other page background, puts background above the topmost block just as a genuine top does.
    The one miss above is exactly that — its topmost block began 4 rows below the band's first
    row — and the gutter is 53px of a ~1027px card pitch, so roughly 1 scroll position in 20
    looks innocent by construction. This turns "undetectable" into "usually detected", and
    nothing more.
    """
    return any(o.frame_y0 > band_y0 for o in topmost.observations)


def _split_on_bounded_cards(group: Sequence[BlockObservation], *, tolerance: int,
                            ) -> tuple[list[list[BlockObservation]], list[str]]:
    """Split ONE `_overlap_groups` group on a card boundary the group's OWN complete sightings
    prove exists, when a bridging fragment has folded two real cards into it.

    This is the fix for a live capture `_overlap_groups`'s docstring did not anticipate. Its
    "two distinct blocks are `_MIN_ITEM_GAP_PX` apart by construction" assumption holds for every
    sighting segment.py reports ON ITS OWN — but two adjacent Hinge cards (page rows 6368..6503
    and 6550..7477, a 47px gutter between) had their shared gutter MISSED in exactly two frames
    (19 and 20 of a 37-frame scroll), each of which then emitted one bottom-clipped block spanning
    both cards — 6368..6709 and 6368..6941, `complete=False` because the analysed band cut them
    off before either card's own bottom. Those two fragments overlap BOTH cards, so
    `_overlap_groups`'s transitive chaining folded every sighting of both cards into one group,
    and `_resolve_group` went on to report every sighting of the second card as contradicting the
    first card's extent, which it had picked as the group's representative.

    The fix is not to weaken `_overlap_groups` — chaining by raw overlap is exactly right for the
    ordinary case of one card sliding through a series of scroll windows, and loosening it would
    reopen the door doc 5.10's phantom item came through. It is to look again, AFTER folding, at
    what the group's own STRONGEST evidence says: a sighting is `complete` only when segment.py
    saw the block bounded by its own gutter- or corner-window on both ends (`EDGE_CARD_CORNER` /
    the analogous gutter edge in segment.py), which is corner-corroborated positive evidence that
    a card boundary is really there. A frame that merely failed to report a gutter — the bridging
    fragment's failure mode — is absence of evidence, not evidence of anything. Positive evidence
    wins: folding the COMPLETE sightings alone through the very same `_overlap_groups` sweep
    answers "how many cards did this group's strongest evidence actually see", because sightings
    of one true card agree within `tolerance` and land together, while sightings of two distinct
    cards are at least `_MIN_ITEM_GAP_PX` apart and land in separate groups. And it is still never
    an average: every sub-group returned here is resolved by `_resolve_group` exactly as an
    ungrouped block is, from a single frame's own observation, never a blend of two.

    Returns `([group], [])` — behaviour BYTE-IDENTICAL to before this function existed — whenever
    there is nothing proven to split on: fewer than two complete sightings, or the complete
    sightings all land in one `_overlap_groups(complete)` card (the ordinary case of a card seen
    complete more than once). That is also the fallback the moment this pass meets something it
    does not understand: a fragment that overlaps NONE of the proven cards by more than
    `tolerance`, or an excluded fragment's heart that no surviving sighting of its own proven card
    corroborates (see the heart-safety loop below). In both cases the whole, unsplit group is
    handed back so `_resolve_group`'s existing loud refusal fires on it, rather than this pass
    guessing where the evidence it cannot place belongs.
    """
    group = list(group)
    complete = [o for o in group if o.complete]
    if len(complete) < 2:
        return [group], []

    cards = _overlap_groups(complete)
    if len(cards) <= 1:
        return [group], []

    # Each proven card's extent is the hull of ITS OWN complete sightings — several independent
    # corner-corroborated measurements of the same card, which is what folding them found, not a
    # blend across the boundary this split exists to find.
    proven = [(min(o.page_y0 for o in card), max(o.page_y1 for o in card), card) for card in cards]
    proven.sort(key=lambda p: p[0])          # page order, so the returned subgroups are too

    def overlap_px(o: BlockObservation, y0: int, y1: int) -> int:
        return max(0, min(o.page_y1, y1) - max(o.page_y0, y0))

    # Assign every non-complete sighting to the proven card(s) it overlaps by MORE than
    # `tolerance` — a real overlap-LENGTH test, not endpoint containment, so a fragment that only
    # grazes a card's edge within chain slack is not mistaken for evidence about that card.
    fragments = [o for o in group if not o.complete]
    assigned: dict[int, int] = {}
    excluded: set[int] = set()
    for i, frag in enumerate(fragments):
        hits = [c for c, (y0, y1, _) in enumerate(proven) if overlap_px(frag, y0, y1) > tolerance]
        if not hits:
            return [group], []
        if len(hits) == 1:
            assigned[i] = hits[0]
        else:
            excluded.add(i)          # overlaps 2+ proven cards: the frame-19/20 bridging shape

    # HEART SAFETY. `heart_ordinal` and selectability are downstream of every heart this module
    # sees, so dropping an excluded sighting must never quietly drop a heart nothing else reports.
    # Every heart on an excluded sighting must fall inside a proven card's extent, and some
    # SURVIVING sighting actually assigned to that same card — a complete one, or a fragment that
    # overlapped only it — must independently report a heart at the same page row. Anything short
    # of that and the split backs out entirely, leaving `_resolve_group`'s ordinary ambiguous- or
    # fragment-past-a-bound failure to fire on the group as a whole rather than this pass silently
    # discarding evidence it could not corroborate.
    for i in excluded:
        frag = fragments[i]
        for _x, y in frag.hearts:
            card_idx = next((c for c, (y0, y1, _) in enumerate(proven)
                             if y0 - tolerance <= y <= y1 + tolerance), None)
            if card_idx is None:
                return [group], []
            _y0, _y1, complete_sightings = proven[card_idx]
            surviving = list(complete_sightings) + [
                fragments[j] for j, c in assigned.items() if c == card_idx]
            if not any(abs(hy - y) <= tolerance for s in surviving for _hx, hy in s.hearts):
                return [group], []

    subgroups = [
        list(complete_sightings) + [fragments[i] for i, c in assigned.items() if c == idx]
        for idx, (_y0, _y1, complete_sightings) in enumerate(proven)]

    # One note per excluded sighting, worded for an operator to act on: which frame, what it saw,
    # and which proven boundary it bridged — that boundary IS the gutter segmentation missed.
    notes = []
    for i in sorted(excluded):
        frag = fragments[i]
        hits = [c for c, (y0, y1, _) in enumerate(proven) if overlap_px(frag, y0, y1) > tolerance]
        boundary = " and ".join(f"{proven[c][0]}..{proven[c][1]}" for c in hits)
        notes.append(
            f"frame {frag.frame_index}'s sighting at page rows {frag.page_y0}..{frag.page_y1} "
            f"(frame rows {frag.frame_y0}..{frag.frame_y1}) spans the proven card boundary "
            f"between {boundary}: segmentation missed the gutter between them in this frame, so "
            "the fragment was excluded rather than folded into either card")

    return subgroups, notes


def _assemble(observations: Sequence[BlockObservation], *, at_scroll_top: bool,
              card_x: tuple[int, int], extent_tolerance_px: int, min_item_gap_px: int,
              band_y0: int, include_notes: bool = False,
              ) -> (tuple[tuple[IndexedBlock, ...], tuple[str, ...]]
                    | tuple[tuple[IndexedBlock, ...], tuple[str, ...], tuple[str, ...]]):
    """Sightings in page space -> the ordered block list, the two index spaces, the failures, and
    the notes — sightings `_split_on_bounded_cards` excluded on the way, reported rather than
    silently dropped.

    Split out from `build_item_index` because it is pure arithmetic over `BlockObservation`
    records — no image data, no cv2, no frames — so the whole decision surface (folding,
    disagreement, the minimum gap, the two numberings) is directly testable against hand-written
    sightings. Same reason `frameshift._resolve` is split off its estimator.

    `band_y0` is the analysed band's first row, which every frame of one capture shares (the
    shift estimator refuses two frames of different sizes, so they cannot differ). It is used
    only by `_scroll_top_evidence`, and only when `at_scroll_top` is asserted.
    """
    failures: list[str] = []
    notes: list[str] = []
    blocks: list[IndexedBlock] = []
    for group in _overlap_groups(observations):
        # `_split_on_bounded_cards` is a no-op — `[group], []` — on every group this module could
        # already fold correctly; it only does work when the group's own complete sightings prove
        # it actually holds two (or more) cards a bridging fragment merged. See its docstring for
        # the live capture that made this necessary.
        subgroups, split_notes = _split_on_bounded_cards(group, tolerance=extent_tolerance_px)
        notes.extend(split_notes)
        for subgroup in subgroups:
            block, group_failures = _resolve_group(
                subgroup, tolerance=extent_tolerance_px, card_x=card_x)
            blocks.append(block)
            failures.extend(group_failures)

    # THE SCROLL-TOP EVIDENCE CHECK. The caller's assertion cannot be verified here, but it can
    # be CONTRADICTED, and an unchallenged false assertion is the one error that produces a
    # confidently wrong index with every completeness property agreeing with it. See
    # `_scroll_top_evidence` for the geometry, the measurement and the residual it leaves.
    top_confirmed = bool(blocks) and _scroll_top_evidence(blocks[0], band_y0)
    if at_scroll_top and blocks and not top_confirmed:
        failures.append(
            f"at_scroll_top was asserted, but the topmost block at page rows "
            f"{blocks[0].page_y0}..{blocks[0].page_y1} begins on the analysed band's own first "
            f"row ({band_y0}) in every frame that saw it, which is what a card sliced by the "
            "band edge looks like rather than the top of a profile — at a genuine scroll top "
            "Hinge's header sits BELOW that row with page background above it (measured 34px "
            "and 68px on the two calibration profiles). Heart ordinal 1 would not be the "
            "profile's first heart, and nothing else in this result would say so")

    # Hinge's scroll-top header chrome, and ONLY it: the topmost block, heartless, never bounded,
    # with real items below it, at a scroll-top the caller has affirmatively confirmed AND the
    # geometry above has not contradicted. A block that ever showed a heart or was ever bounded
    # end to end is an item and survives this, which is what keeps a genuine item 1 out of the
    # exception. The `top_confirmed` term matters beyond the failure already appended: without
    # it this relabelling is what SWALLOWS the evidence, turning the sliced leading fragment of a
    # false top into a "chrome" block that no later rule looks at again.
    if (at_scroll_top and top_confirmed and len(blocks) > 1
            and blocks[0].kind == ITEM_PARTIAL and not blocks[0].hearts):
        blocks[0] = replace(
            blocks[0], kind=ITEM_LEADING_CHROME,
            reason=("the topmost block at a confirmed scroll-top, heartless and never bounded: "
                    "Hinge's filter-chips header and name row, which sit above item 1 and are "
                    "chrome rather than an item — outside both index spaces; "
                    + blocks[0].reason))

    # THE FABRICATED-ITEM GUARD. Folding by overlap can only ever merge too much; the way it
    # could invent an item is by leaving one card as two fragments that never overlapped. Two
    # real blocks are always separated by at least the bottom of the gutter window, so anything
    # closer is that split. `extent_tolerance_px` is subtracted so chain slack alone cannot fire
    # it.
    for above, below in pairwise(blocks):
        gap = below.page_y0 - above.page_y1
        if gap + extent_tolerance_px < min_item_gap_px:
            failures.append(
                f"the blocks at page rows {above.page_y0}..{above.page_y1} and "
                f"{below.page_y0}..{below.page_y1} are only {gap}px apart, closer than the "
                f"{min_item_gap_px}px minimum gutter the layout can draw — they are one card "
                "whose sightings never overlapped, and counting them as two would fabricate an "
                "item")

    for i, block in enumerate(blocks):
        if block.kind == ITEM_LEADING_CHROME or block.hearts or block.complete:
            continue
        if any(later.hearts for later in blocks[i + 1:]):
            failures.append(
                f"the block at page rows {block.page_y0}..{block.page_y1} "
                + _UNCERTAIN_HEART_NOTE)

    # The two numberings, in page order. The heart counter advances by the block's heart COUNT
    # rather than by one, so an ambiguous block's second heart still occupies its ordinal and
    # nothing below it is renumbered; `heart_ordinal` names the first of them.
    ordinal, model = 0, 0
    for i, block in enumerate(blocks):
        if block.kind == ITEM_LEADING_CHROME:
            continue
        heart_ordinal = ordinal + 1 if block.hearts else None
        ordinal += len(block.hearts)
        model_index = None
        if block.kind == ITEM_SELECTABLE:
            model += 1
            model_index = model
        blocks[i] = replace(block, heart_ordinal=heart_ordinal, model_index=model_index)

    result = (tuple(blocks), tuple(failures))
    return (*result, tuple(notes)) if include_notes else result


_FRAGMENT_OVERRUN_FAILURE = re.compile(
    r"^frame (\d+) sees page rows .* — a fragment cannot reach past the card that contains it, so ")


def _fragment_overrun_frame_indices(failures: Sequence[str]) -> tuple[int, ...]:
    """Return the exact frames a fold proved crossed a bounded card, or ``()``.

    This deliberately accepts no nearby failure wording.  An omission recovery is only justified
    when *every* preliminary fold failure is the explicit partial-sighting-overruns-its-bounded-
    card invariant from `_resolve_group`; a bad shift, two competing complete extents, uncertain
    heart, or any unrelated assembly fault remains a refusal.  The numbers are output by this
    module, not inferred from screenshot order or a heuristic score.
    """
    if not failures:
        return ()
    indices: list[int] = []
    for failure in failures:
        match = _FRAGMENT_OVERRUN_FAILURE.match(failure)
        if match is None:
            return ()
        indices.append(int(match.group(1)))
    return tuple(sorted(set(indices)))


def build_item_index(frames: Sequence[bytes], *, content_band: tuple[float, float],
                     like_template, like_threshold: float, at_scroll_top: bool,
                     identity_band: tuple[float, float, float, float] | None,
                     animation_markers: Sequence[bool] | None = None,
                     video_mute_markers: Sequence[VideoMuteMarker] | None = None,
                     trust_window_px: int | None = None,
                     extent_tolerance_px: int = _EXTENT_TOLERANCE_PX,
                     min_item_gap_px: int = _MIN_ITEM_GAP_PX,
                     end_tail_gap_px: int = _END_TAIL_GAP_PX,
                     _prefix_index: ItemIndex | None = None,
                     _allow_frame_omission_recovery: bool = True) -> ItemIndex:
    """Fold an ordered run of scroll frames into ONE list of the profile's items.

    `frames` are consecutive screencaps of the SAME profile, earliest first, captured while
    scrolling forward. What order buys is precise, and an earlier revision of this docstring
    oversold it ("a re-ordered or interleaved sequence produces refusals, not a rearranged
    index"), which a validation pass measured to be FALSE: page position is MEASURED pairwise
    rather than assumed from ordering, so a fully reversed run — and every adjacent transposition
    of a real capture — still chains and still yields the same 9 items with the same translation.
    Those answers are correct, and the safe direction, but do not rely on an ordering mistake
    being caught. What order genuinely buys is coverage, not correctness:

      * neighbours must still CORRESPOND. A pair whose content moved further than the trust
        window refuses, which is what a shuffle or a spliced-in foreign frame trips over — not
        the ordering itself;
      * `reached_end` is read off the LAST frame, so a run given backwards reports it False;
      * `at_scroll_top` is about the TOPMOST block in page space, which in a forward run is
        frame 0's. It is not read off frame 0 directly, so it survives a reordering that a human
        would call scrambled.

    `content_band` is the `(y0, y1)` height-fraction band cards occupy — pass the driver's
    `self.content_band` (the config-merged value), NOT `spec.content_band`, or an operator
    override is silently ignored. `like_template` and `like_threshold` are REQUIRED and have no
    defaults for the same reason they have none in `segment_frame`: the calibrated glyph and its
    0.75 floor have exactly one home each (`HINGE_SPEC.templates["like"]` and
    `hinge._LIKE_MATCH_THRESHOLD`), and a default here would be a second copy free to drift. Pass
    `driver._template("like")` and `hinge._LIKE_MATCH_THRESHOLD`. All three are forwarded to
    `segment_frame` unchanged.

    `at_scroll_top` is REQUIRED and is an assertion by the caller, not a hint: "this capture
    contains the top of the profile, and I confirmed it affirmatively" — doc 5.5's filter-chips
    signal in `identity_band`, which 5.5 requires before counting regardless. It buys two things.
    Heart ordinals become ABSOLUTE (heart 1 really is the profile's first heart, which is what a
    counting navigation needs), and the leading header chrome can be classed as chrome instead of
    as a block that might be hiding heart 1.

    Passing True without having confirmed it is still the most dangerous input this module takes,
    and it is now PARTLY detected rather than not at all: `_scroll_top_evidence` refuses a
    capture whose topmost block runs flush into the analysed band's first row, which is what the
    band edge slicing a card looks like and what a capture starting mid-profile normally
    produces. Read that function for what it catches, what it cannot (roughly 1 scroll position
    in 20, where the band edge lands in a gutter instead of on a card), and why doc 5.5's
    filter-chips confirmation therefore remains a hard gate and not a formality.

    With `at_scroll_top=False` the ordinals are RELATIVE to whatever was in view — a validation
    pass measured a mid-profile window returning translation (2, 3, 4, 5, 6) for what were really
    items 5..9 — so navigation must not count against a False index. That is now enforced rather
    than only documented: `heart_ordinal_for` raises on such an index (it used to answer with a
    confident int), and `item_nav.navigate_to_item` refuses one before it moves the phone.

    `identity_band` is REQUIRED and has no default, for `at_scroll_top`'s reason rather than
    `content_band`'s: it is not a tuning knob but a statement about the capture, and the thing it
    produces — `ItemIndex.identity`, the fingerprint of WHOSE profile this is — is what
    `item_nav`'s entry gate refuses to navigate without. Pass the driver's `self.identity_band`
    (config-merged), not `spec.identity_band`. `None` is accepted and yields an UNKNOWN identity,
    which navigation then refuses: an app that declares no band cannot be identity-checked, and
    that is a hard stop rather than a licence. Making it required is the structural half of doc
    5.7's carried-forward requirement — an index that cannot say whose profile it describes
    cannot be built by accident, only on purpose, and still cannot be navigated with.

    `animation_markers` is optional, frame-aligned affirmative app evidence (for Hinge, its exact
    per-card mute control). It never changes raw frameshift or segmentation and never authorizes
    an isolated repair; it is consulted only by the bounded measured-bridge window documented in
    `_repair_shifts_from_layout`. None means no marker evidence. A supplied sequence with the
    wrong length is refused rather than padded or guessed.

    `video_mute_markers` is the v12 form of that evidence: positioned, near-perfect matches of
    Hinge's own mute control. A row first binds to one segmented card and may then follow that
    same card after its overlay leaves the visible band, but only through two exact card-local
    anchors. One raw refused pair may be bridged only while that same control is affirmatively
    visible in both frames, its exact displacement agrees with another card-local anchor, and a
    complete page rebuild succeeds. The one immediately following a proved overlay exit may
    instead rejoin the lower cards only with three identical strip values and the tracked card's
    exact heart displacement. When the legacy boolean sequence is omitted, its frame-aligned
    compatibility view is derived from these rows; an explicitly supplied boolean sequence
    remains authoritative.

    `trust_window_px` is forwarded to `frameshift.estimate_shift`; the remaining keywords are
    calibration constants with measured defaults, exposed so a validation pass can vary one
    without editing the module.

    Raises `ItemIndexError` on an empty `frames`. `SegmentationError` and `ShiftEstimationError`
    propagate unchanged from the two dependencies. Everything else is a RESULT: read
    `ItemIndex.usable` before reading `ItemIndex.blocks`, and treat False as a hard stop.
    """
    if not frames:
        raise ItemIndexError(
            "no frames to index: an empty capture is a capture failure, and returning an empty "
            "item list for it would read as 'this profile has no items'")
    marker_evidence = (tuple(False for _ in frames) if animation_markers is None
                       else tuple(bool(value) for value in animation_markers))
    if len(marker_evidence) != len(frames):
        raise ItemIndexError(
            f"{len(marker_evidence)} animation marker(s) given for {len(frames)} frame(s) — "
            "video evidence must be frame-aligned and may not be guessed or padded")
    mute_markers = tuple(video_mute_markers or ())
    if any(not isinstance(marker, VideoMuteMarker) for marker in mute_markers):
        raise ItemIndexError("video mute markers must be VideoMuteMarker records, never booleans")
    if any(not (0 <= marker.frame_index < len(frames)) for marker in mute_markers):
        raise ItemIndexError("video mute marker frame index is outside this capture")
    if animation_markers is None and mute_markers:
        # Structured rows are stronger evidence; deriving their historical frame flag keeps the
        # all-or-nothing v11 fallback available after a v12 track rejects itself.
        marker_evidence = tuple(any(marker.frame_index == frame_index for marker in mute_markers)
                                for frame_index in range(len(frames)))

    # `_prefix_index` is intentionally private: the supervised calibration reader is the only
    # caller that grows one frame list by one frame at a time.  Reuse only measurements whose
    # exact source bytes are still this call's prefix.  The complete page fold below is STILL
    # rerun over every frame, so all coverage/identity/refusal decisions have exactly the same
    # inputs and result as an uncached call; this only avoids decoding and matching old pixels
    # again.  A mismatch falls back to the ordinary full analysis rather than making a best
    # effort out of a stale prefix.
    prefix_count = 0
    if _prefix_index is not None and len(_prefix_index.frames) < len(frames):
        candidate_count = len(_prefix_index.frames)
        if all(
            seg.frame_digest == hashlib.sha256(frame).hexdigest()
            for seg, frame in zip(_prefix_index.frames, frames[:candidate_count], strict=True)
        ):
            prefix_count = candidate_count

    if prefix_count:
        # A cached prefix may itself have used the layout-assisted path.  Its effective shifts
        # are safe results, but the raw evidence must be replayed here so the extended capture
        # re-applies the same bounded-run gate and retains provenance instead of silently
        # treating a synthetic acceptance as an ordinary frameshift quorum.
        prefix_raw_shifts = dict(getattr(_prefix_index, "layout_repaired_shifts", ()) or ())
        segmentations = _prefix_index.frames + tuple(
            segment_frame(
                frame, content_band=content_band, like_template=like_template,
                like_threshold=like_threshold,
                recover_leading_low_contrast_media=bool(at_scroll_top and i == 0))
            for i, frame in enumerate(frames[prefix_count:], start=prefix_count))
        shifts = tuple(prefix_raw_shifts.get(i, shift)
                       for i, shift in enumerate(_prefix_index.shifts)) + tuple(
            estimate_shift(frames[i], frames[i + 1], content_band=content_band,
                           trust_window_px=trust_window_px)
            for i in range(prefix_count - 1, len(frames) - 1))
    else:
        segmentations = tuple(
            segment_frame(
                frame, content_band=content_band, like_template=like_template,
                like_threshold=like_threshold,
                recover_leading_low_contrast_media=bool(at_scroll_top and i == 0))
            for i, frame in enumerate(frames))
        shifts = tuple(
            estimate_shift(frames[i], frames[i + 1], content_band=content_band,
                           trust_window_px=trust_window_px)
            for i in range(len(frames) - 1))

    # `frameshift` keeps its own three-strip quorum unchanged.  Only after the independent
    # segmenter has produced card edges and heart glyphs may a bounded animation-shaped run
    # *propose* an alternative.  Commit it only if the entire ordinary assembly succeeds at this
    # call's extent bound (8px by default); no omitted frames or best-effort prefix is involved.
    raw_shifts = shifts
    used_video_track = False
    accepted_repair_path: str | None = None
    if mute_markers:
        shifts, repair_notes, layout_repaired_shifts = _repair_video_track_shifts(
            segmentations, shifts, mute_markers,
            extent_tolerance_px=extent_tolerance_px)
        used_video_track = bool(repair_notes)
        if repair_notes:
            accepted_repair_path = "v12_mute_card_track"
        if not repair_notes:
            # Position-aware evidence is preferred, but a marker row can be unavailable for an
            # older replay even when its historical frame-aligned flag was recorded.  Keep that
            # compatibility path as a proposal-only fallback; live Hinge never relies on it.
            shifts, repair_notes, layout_repaired_shifts = _repair_shifts_from_layout(
                segmentations, raw_shifts, extent_tolerance_px=extent_tolerance_px,
                animation_markers=marker_evidence)
            if repair_notes:
                accepted_repair_path = "legacy_layout_grammar"
    else:
        # Compatibility for recorded/offline callers which only stored the pre-v12 frame flag.
        # Production Hinge sends positioned markers and never grants repair authority from this.
        shifts, repair_notes, layout_repaired_shifts = _repair_shifts_from_layout(
            segmentations, shifts, extent_tolerance_px=extent_tolerance_px,
            animation_markers=marker_evidence)
        if repair_notes:
            accepted_repair_path = "legacy_layout_grammar"
    def repair_probe_failed(candidate_shifts: Sequence[ShiftEstimate]) -> bool:
        probe_offsets, probe_chain_failures = _frame_offsets(candidate_shifts)
        probe_failures: tuple[str, ...] = ()
        if not probe_chain_failures and not any(seg.failures for seg in segmentations):
            _probe_blocks, probe_failures, _probe_notes = _assemble(
                _observations(segmentations, probe_offsets), at_scroll_top=at_scroll_top,
                card_x=segmentations[0].card_x, extent_tolerance_px=extent_tolerance_px,
                min_item_gap_px=min_item_gap_px, band_y0=segmentations[0].band[0],
                include_notes=True)
        return bool(probe_chain_failures or probe_failures or any(seg.failures for seg in segmentations))

    if repair_notes and repair_probe_failed(shifts):
        if used_video_track:
            # Never blend two repair theories.  A physical-track proposal that cannot rebuild
            # the page is discarded wholesale, then the independently audited historical
            # grammar gets one fresh all-or-nothing probe from the original raw shifts.
            legacy_shifts, legacy_notes, legacy_raw = _repair_shifts_from_layout(
                segmentations, raw_shifts, extent_tolerance_px=extent_tolerance_px,
                animation_markers=marker_evidence)
            if legacy_notes and not repair_probe_failed(legacy_shifts):
                shifts, repair_notes, layout_repaired_shifts = (
                    legacy_shifts, legacy_notes, legacy_raw)
                accepted_repair_path = "legacy_layout_grammar"
            else:
                shifts, repair_notes, layout_repaired_shifts = raw_shifts, (), ()
                accepted_repair_path = None
        else:
            shifts, repair_notes, layout_repaired_shifts = raw_shifts, (), ()
            accepted_repair_path = None

    repair_provenance = tuple(
        ItemIndexRepair(
            pair_index=pair_index, path=accepted_repair_path or "legacy_layout_grammar",
            raw_status=raw.status, raw_delta_px=raw.delta_px,
            effective_status=shifts[pair_index].status,
            effective_delta_px=shifts[pair_index].delta_px,
            marker_frames=tuple(marker.frame_index for marker in mute_markers
                                if marker.frame_index in (pair_index, pair_index + 1)),
        )
        for pair_index, raw in layout_repaired_shifts)

    # An ordinary read can contain one unusable *intermediate* frame while both of its neighbours
    # still correspond.  That produces either one failed pair (only one side was damaged) or two
    # adjacent failed pairs (both sides were damaged, as with a transient video/UI frame).  Do not
    # turn two votes into a shift, and do not manufacture an offset: the only omission candidates
    # are frames shared by every failed pair, and each candidate must pass a fresh direct bridge
    # plus a fully usable rebuild over the remaining real frames.  The omitted-frame provenance
    # below makes that reduction visible to every caller, including the crop layer.
    failed_pairs = [i for i, shift in enumerate(shifts) if shift.delta_px is None]
    failed_segmentation_frames = {
        i for i, segmentation in enumerate(segmentations) if segmentation.failures
    }
    failed_pair_records = tuple((i, i + 1) for i in failed_pairs)
    omission_candidates: set[int] = set()
    if failed_pairs:
        omission_candidates = {failed_pairs[0], failed_pairs[0] + 1}
        for pair_index in failed_pairs[1:]:
            omission_candidates.intersection_update((pair_index, pair_index + 1))
        omission_candidates = {
            omitted for omitted in omission_candidates if 0 < omitted < len(frames) - 1
        }
    # A missed low-contrast gutter is not always a frame-local segmentation failure.  It can
    # yield a plausible partial block until another frame bounds the true card, at which point
    # `_resolve_group` proves the partial crosses that boundary.  The reported capture had this
    # exact shape.  Let that *specific proof* nominate its own frames for the same direct-bridge
    # recovery; do not search through ostensibly good frames looking for a better answer.
    fold_contradiction_frames: tuple[int, ...] = ()
    if (_allow_frame_omission_recovery and not failed_pairs
            and not failed_segmentation_frames):
        probe_offsets, probe_chain_failures = _frame_offsets(shifts)
        if not probe_chain_failures:
            probe_observations = _observations(segmentations, probe_offsets)
            probe_near_gutters = tuple(
                (frame_index, run.y0 + offset, run.y1 + offset)
                for frame_index, (segmentation, offset) in enumerate(
                    zip(segmentations, probe_offsets, strict=True))
                if offset is not None
                for run in segmentation.runs
                if (run.kind == RUN_TOO_LONG
                    and min(_HEART_SEPARATED_NEAR_GUTTER_PX) <= run.height
                    <= max(_HEART_SEPARATED_NEAR_GUTTER_PX)))
            probe_observations, _probe_notes = _split_repeated_near_gutter_merges(
                probe_observations, probe_near_gutters, tolerance=extent_tolerance_px)
            _probe_blocks, probe_failures, _probe_assembly_notes = _assemble(
                probe_observations, at_scroll_top=at_scroll_top,
                card_x=segmentations[0].card_x,
                extent_tolerance_px=extent_tolerance_px,
                min_item_gap_px=min_item_gap_px, band_y0=segmentations[0].band[0],
                include_notes=True)
            fold_contradiction_frames = _fragment_overrun_frame_indices(probe_failures)
    # One short, contiguous contradictory run is also eligible for the same conservative
    # recovery.  A white-on-white Hinge card boundary can remain invisible for four adjacent
    # scroll positions; all failed frames must disappear from a fully rebuilt page coordinate system,
    # joined by one fresh measured bridge.  A known bad frame can also leave its immediately
    # adjacent segmentation apparently valid while its wrong edge only becomes visible during
    # the page fold.  In that case try one extra neighbour on either side -- never a free-form
    # scan -- and accept it only if it is the sole clean complete rebuild.  Anything longer,
    # non-contiguous, at an endpoint, accompanied by an unrelated shift failure, or ambiguous
    # between two clean repaired windows remains a hard refusal.
    segmentation_frames = tuple(sorted(failed_segmentation_frames))
    segmentation_recovery = (
        1 <= len(segmentation_frames) <= MAX_SEGMENTATION_FALLBACK_FRAMES
        and segmentation_frames[0] > 0
        and segmentation_frames[-1] < len(frames) - 1
        and segmentation_frames == tuple(range(segmentation_frames[0],
                                               segmentation_frames[-1] + 1)))
    if segmentation_recovery:
        omitted_set = set(segmentation_frames)
        # A pair may fail because it touches the bad run; a failure wholly outside it proves the
        # sequence has a second fault and cannot be repaired by dropping these frames.
        segmentation_recovery = all(
            pair_index in omitted_set or pair_index + 1 in omitted_set
            for pair_index in failed_pairs)

    fold_recovery = (
        1 <= len(fold_contradiction_frames) <= MAX_SEGMENTATION_FALLBACK_FRAMES
        and fold_contradiction_frames[0] > 0
        and fold_contradiction_frames[-1] < len(frames) - 1
        and fold_contradiction_frames == tuple(range(
            fold_contradiction_frames[0], fold_contradiction_frames[-1] + 1)))

    if failed_segmentation_frames:
        candidate_runs = [segmentation_frames] if segmentation_recovery else []
        if segmentation_recovery:
            # The failed frame itself is non-negotiable.  The two expansions merely test whether
            # an adjacent *apparently clean* read carried the same missed-gutter geometry.  Do
            # not expand both sides in one attempt: two extra frames would hide which local
            # observation was wrong and turn this bounded recovery into evidence selection.
            left = segmentation_frames[0] - 1
            right = segmentation_frames[-1] + 1
            if left > 0 and len(segmentation_frames) < MAX_SEGMENTATION_FALLBACK_FRAMES:
                candidate_runs.append((left,) + segmentation_frames)
            if (right < len(frames) - 1
                    and len(segmentation_frames) < MAX_SEGMENTATION_FALLBACK_FRAMES):
                candidate_runs.append(segmentation_frames + (right,))
    elif fold_recovery:
        # Unlike the segmenter-failure path, the page fold names every bad partial explicitly;
        # there is no reason to discard a neighbouring observation that did not overrun a card.
        candidate_runs = [fold_contradiction_frames]
    else:
        candidate_runs = [(candidate,) for candidate in sorted(omission_candidates)]

    if _allow_frame_omission_recovery and candidate_runs:
        candidates: list[tuple[tuple[float, float, float, int], tuple[int, ...], ItemIndex]] = []
        # With one refusal, test both sides as before.  With two adjacent refusals, their shared
        # middle frame is the sole candidate.  A non-adjacent refusal leaves the intersection
        # empty and cannot be disguised as one bad capture.  ``build_item_index`` remeasures the
        # bridge and revalidates every later page-level invariant with recursion disabled, so no
        # refused consensus or assumed sum enters the recovered coordinate space.
        for omitted_frames in candidate_runs:
            omitted_set = set(omitted_frames)
            reduced_frames = tuple(frame for i, frame in enumerate(frames) if i not in omitted_set)
            # The direct bridge spans the omitted frames plus the next real read.  Its maximum
            # is therefore a known multiple of the driver's own one-gesture ceiling, not an
            # inferred page offset.  Widen the estimator only enough to see that bridge; every
            # other reduced-chain pair is checked back against the normal one-step ceiling below.
            recovery_trust_window_px = trust_window_px
            if failed_segmentation_frames or fold_contradiction_frames:
                default_window = (segmentations[0].band[1] - segmentations[0].band[0]) // 2
                recovery_trust_window_px = max(
                    default_window if trust_window_px is None else trust_window_px,
                    (len(omitted_frames) + 1) * _MAX_STEP_PX)
            recovered = build_item_index(
                reduced_frames, content_band=content_band, like_template=like_template,
                like_threshold=like_threshold, at_scroll_top=at_scroll_top,
                identity_band=identity_band,
                animation_markers=tuple(value for i, value in enumerate(marker_evidence)
                                        if i not in omitted_set),
                video_mute_markers=tuple(
                    VideoMuteMarker(
                        frame_index=(marker.frame_index - sum(
                            omitted < marker.frame_index for omitted in omitted_frames)),
                        x=marker.x, y=marker.y, score=marker.score)
                    for marker in mute_markers if marker.frame_index not in omitted_set),
                trust_window_px=recovery_trust_window_px,
                extent_tolerance_px=max(extent_tolerance_px, _RECOVERY_EXTENT_TOLERANCE_PX),
                min_item_gap_px=min_item_gap_px,
                end_tail_gap_px=end_tail_gap_px, _allow_frame_omission_recovery=False)
            bridge = (omitted_frames[0] - 1, omitted_frames[-1] + 1)
            # The source index shifts by one after the omission, but the bridge always starts at
            # the original left neighbour.  ``usable`` also rules out any other broken pair.
            bridge_shift = recovered.shifts[bridge[0]] if recovered.usable else None
            ordinary_pairs_stay_one_step = all(
                pair_index == bridge[0] or shift.delta_px is not None
                and 0 <= shift.delta_px <= _MAX_STEP_PX
                for pair_index, shift in enumerate(recovered.shifts))
            if (bridge_shift is not None and bridge_shift.status == SHIFT_MEASURED
                    and ordinary_pairs_stay_one_step):
                candidates.append((
                    (bridge_shift.agreeing, bridge_shift.confidence,
                     -bridge_shift.dissenting, -omitted_frames[0]), omitted_frames, recovered))
        # Pair-shift recovery has a measured-fitness tie-breaker because both possible omitted
        # frames are directly implicated by the failed pair.  A clean rebuild that drops exactly
        # the explicitly contradictory segmentation frames takes priority over every expansion:
        # it preserves more observations and needs no inference about a neighbour.  If that
        # direct recovery fails, exactly one expanded segmentation window may succeed.  Two
        # clean expansions would leave us unable to tell which nominally valid neighbour
        # contained the missed gutter, so refuse that ambiguity rather than choose on
        # strip-count noise.
        candidate_pool = candidates
        if failed_segmentation_frames:
            direct_candidates = [
                candidate for candidate in candidates
                if len(candidate[1]) == len(segmentation_frames)]
            expanded_candidates = [
                candidate for candidate in candidates
                if len(candidate[1]) > len(segmentation_frames)]
            candidate_pool = (direct_candidates if direct_candidates else
                              expanded_candidates if len(expanded_candidates) == 1 else [])
        if candidate_pool:
            # Most independent agreement wins; stable secondary keys make an audit replay choose
            # the same reduction when both bridges are sound.
            _score, omitted_frames, recovered = max(candidate_pool, key=lambda candidate: candidate[0])
            bridge = (omitted_frames[0] - 1, omitted_frames[-1] + 1)
            return replace(
                recovered,
                source_frame_indices=tuple(i for i in range(len(frames)) if i not in omitted_frames),
                recovered_from_pair=(failed_pair_records[0] if failed_pair_records else None),
                recovery_bridge=bridge,
                recovery_failed_shift=(shifts[failed_pairs[0]] if failed_pairs else None),
                recovered_from_pairs=failed_pair_records,
                recovery_failed_shifts=tuple(shifts[i] for i in failed_pairs),
                recovered_from_segmentation_frames=(
                    omitted_frames if failed_segmentation_frames else ()),
                recovered_from_fold_contradiction_frames=(
                    omitted_frames if fold_contradiction_frames else ()),
                recovery_reason=(
                    f"frame{'s' if len(omitted_frames) > 1 else ''} "
                    + ", ".join(str(frame) for frame in omitted_frames) + " "
                    + ("were omitted after a segmentation contradiction and one adjacent "
                       "frame's geometry could not be reconciled with the complete page"
                       if (failed_segmentation_frames
                           and len(omitted_frames) > len(segmentation_frames)) else
                       "were omitted after their segmentations contradicted themselves"
                       if len(omitted_frames) > 1 and failed_segmentation_frames else
                       "was omitted after its segmentation contradicted itself"
                       if failed_segmentation_frames else
                       "were omitted after the complete page fold proved each partial sighting "
                       "crossed a bounded card"
                       if len(omitted_frames) > 1 and fold_contradiction_frames else
                       "was omitted after the complete page fold proved its partial sighting "
                       "crossed a bounded card"
                       if fold_contradiction_frames else
                       f"frames {failed_pairs[0]} and {failed_pairs[0] + 1} had no trustworthy "
                       "shift" if len(failed_pairs) == 1 else
                       f"pairs {', '.join(f'{a}/{b}' for a, b in failed_pair_records)} had no "
                       "trustworthy shifts")
                    + "; the fresh direct bridge from "
                    f"frame {bridge[0]} to frame {bridge[1]} and the complete rebuilt index "
                    "both passed without assuming an offset"
                    + (f" (the bridge alone was measured inside the "
                       f"{(len(omitted_frames) + 1) * _MAX_STEP_PX}px "
                       "multi-read window)" if len(omitted_frames) > 1 else "")))

    failures: list[str] = []
    for i, seg in enumerate(segmentations):
        # A frame that contradicted itself cannot be folded into a page that has to be
        # self-consistent, and segment.py's failures are exactly the index-corrupting kind (a
        # two-heart block, a heart bounded by nothing). They are carried forward with the frame
        # number attached rather than restated.
        failures.extend(f"frame {i}: {reason}" for reason in seg.failures)

    offsets, chain_failures = _frame_offsets(shifts)
    failures.extend(chain_failures)

    card_x = segmentations[0].card_x
    if chain_failures:
        # No page space spans the capture, so there is nothing to fold. Emphatically NOT a
        # best-effort prefix: a partial list looks exactly like a complete one to a caller that
        # forgot to check `usable`, and the missing items are precisely the ones the run would
        # then misnumber.
        blocks: tuple[IndexedBlock, ...] = ()
        notes: tuple[str, ...] = repair_notes
    else:
        observations = _observations(segmentations, offsets)
        # A 59..64px page-background run stays `RUN_TOO_LONG` in one frame unless that frame's
        # own hearts prove a boundary.  Preserve that conservative segmenter rule, but let the
        # page fold use the stronger cross-frame proof when it is present (see helper).
        near_gutters = tuple(
            (frame_index, run.y0 + offset, run.y1 + offset)
            for frame_index, (segmentation, offset) in enumerate(
                zip(segmentations, offsets, strict=True))
            if offset is not None
            for run in segmentation.runs
            if (run.kind == RUN_TOO_LONG
                and min(_HEART_SEPARATED_NEAR_GUTTER_PX) <= run.height
                <= max(_HEART_SEPARATED_NEAR_GUTTER_PX)))
        observations, near_gutter_notes = _split_repeated_near_gutter_merges(
            observations, near_gutters, tolerance=extent_tolerance_px)
        blocks, assembly_failures, assembly_notes = _assemble(
            observations, at_scroll_top=at_scroll_top, card_x=card_x,
            extent_tolerance_px=extent_tolerance_px, min_item_gap_px=min_item_gap_px,
            band_y0=segmentations[0].band[0], include_notes=True)
        notes = repair_notes + near_gutter_notes + assembly_notes
        failures.extend(assembly_failures)

    placed = [
        (seg, off) for seg, off in zip(segmentations, offsets, strict=True)
        if off is not None
    ]
    page_span = (min(seg.band[0] + off for seg, off in placed),
                 max(seg.band[1] + off for seg, off in placed))

    reached_end, tail_gap = _tail(segmentations[-1], end_tail_gap_px=end_tail_gap_px)
    # WHOSE profile this is, from these very frames. Never raises and never contributes a
    # `failures` entry: an index with no identity is perfectly usable for describing a capture
    # (a validation pass reads `translation` without navigating anywhere, `item_crops` renumbers
    # against it), and it is `item_nav`'s entry gate that refuses to NAVIGATE with one. Folding
    # it into `usable` would conflate "this capture contradicts itself" with "this capture cannot
    # prove whose it is", which are different facts with different fixes.
    identity = capture_profile_identity(frames, identity_band=identity_band)
    return ItemIndex(
        blocks=blocks, frames=segmentations, shifts=shifts, offsets=tuple(offsets),
        page_span=page_span, at_scroll_top=at_scroll_top, reached_end=reached_end,
        tail_gap_px=tail_gap, failures=tuple(failures), identity=identity,
        source_frame_indices=tuple(range(len(frames))), notes=notes,
        animation_markers=marker_evidence,
        video_mute_markers=mute_markers,
        repair_provenance=repair_provenance,
        layout_repaired_shifts=layout_repaired_shifts,
        extent_tolerance_px=extent_tolerance_px)


def _tail(last: FrameSegmentation, *, end_tail_gap_px: int) -> tuple[bool, int | None]:
    """`(reached_end, tail_gap_px)` from the FINAL frame alone.

    The end of a profile looks like this and nothing else: the bottommost block's own bottom edge
    observed — by its rounded CORNER specifically, since a gutter would mean another card follows
    — with more page background below it than any gutter the layout draws. `tail_gap_px` is that
    background, and is None when the last block's bottom was never bounded.

    What this cannot see is a card that begins BELOW the analysed band's bottom edge; one frame
    has no way to. The corroborating signal is the capture's own last shift: a closed loop that
    scrolled and did not move (`shifts[-1].delta_px == 0`) has hit the bottom independently, and
    a caller wanting certainty should require both.
    """
    if not last.blocks:
        return False, None
    bottom = last.blocks[-1].bottom
    if not bottom.observed or bottom.kind != EDGE_CARD_CORNER:
        return False, None
    gap = last.band[1] - last.blocks[-1].y1
    return gap > end_tail_gap_px, gap
