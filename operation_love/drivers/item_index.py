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
  * Whether a card is ANIMATED. Two screencaps of the same position over animated content
    differ, which segment.py flags as invisible to one frame and which frameshift measured as
    the entire refusal tail of the hand-scrolled capture. It surfaces here as a broken chain,
    which is a refusal, not a wrong answer.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .frameshift import ShiftEstimate, estimate_shift
from .item_identity import ProfileIdentity, capture_profile_identity
# `_GUTTER_PX` / `_GUTTER_TOLERANCE_PX` are imported rather than re-declared, on frameshift.py's
# precedent: that window is already measured-and-cited in segment.py, it is what CUT the blocks
# this module folds, and a second copy would be free to drift away from it.
from .segment import (
    _GUTTER_PX, _GUTTER_TOLERANCE_PX, EDGE_CARD_CORNER, FrameSegmentation, segment_frame)


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
# 8 is under a third of that.
#
# Exceeding it is never absorbed silently in either direction — too-large a disagreement becomes
# a `failures` entry, and a fragment that drifts out of overlap with its own block becomes the
# `_MIN_ITEM_GAP_PX` failure below. Both are loud.
_EXTENT_TOLERANCE_PX = 8

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
    for i, (seg, offset) in enumerate(zip(segmentations, offsets)):
        if offset is None:
            continue
        for block in seg.blocks:
            out.append(BlockObservation(
                frame_index=i,
                page_y0=block.y0 + offset, page_y1=block.y1 + offset,
                frame_y0=block.y0, frame_y1=block.y1,
                kind=block.kind, complete=block.complete,
                top_observed=block.top.observed, bottom_observed=block.bottom.observed,
                hearts=tuple((x, y + offset) for x, y in block.hearts)))
    return out


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
                failures.append(
                    f"frames {rep.frame_index} and {other.frame_index} disagree about the block "
                    f"at page rows {page_y0}..{page_y1}: {page_y1 - page_y0}px against "
                    f"{other.height}px ({other.page_y0}..{other.page_y1}). Both observed both "
                    "edges, so one of them is wrong; averaging them would produce an extent "
                    "neither frame saw")
        for other in observations:
            if other.complete:
                continue
            if other.page_y0 < page_y0 - tolerance or other.page_y1 > page_y1 + tolerance:
                failures.append(
                    f"frame {other.frame_index} sees page rows {other.page_y0}..{other.page_y1} "
                    f"where the block was bounded at {page_y0}..{page_y1} — a fragment cannot "
                    "reach past the card that contains it, so either a gutter was missed or "
                    "these two sightings are not the same block")
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


def _assemble(observations: Sequence[BlockObservation], *, at_scroll_top: bool,
              card_x: tuple[int, int], extent_tolerance_px: int, min_item_gap_px: int,
              band_y0: int) -> tuple[tuple[IndexedBlock, ...], tuple[str, ...]]:
    """Sightings in page space -> the ordered block list, the two index spaces, and the failures.

    Split out from `build_item_index` because it is pure arithmetic over `BlockObservation`
    records — no image data, no cv2, no frames — so the whole decision surface (folding,
    disagreement, the minimum gap, the two numberings) is directly testable against hand-written
    sightings. Same reason `frameshift._resolve` is split off its estimator.

    `band_y0` is the analysed band's first row, which every frame of one capture shares (the
    shift estimator refuses two frames of different sizes, so they cannot differ). It is used
    only by `_scroll_top_evidence`, and only when `at_scroll_top` is asserted.
    """
    failures: list[str] = []
    blocks: list[IndexedBlock] = []
    for group in _overlap_groups(observations):
        block, group_failures = _resolve_group(
            group, tolerance=extent_tolerance_px, card_x=card_x)
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
    for above, below in zip(blocks, blocks[1:]):
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

    return tuple(blocks), tuple(failures)


def build_item_index(frames: Sequence[bytes], *, content_band: tuple[float, float],
                     like_template, like_threshold: float, at_scroll_top: bool,
                     identity_band: tuple[float, float, float, float] | None,
                     trust_window_px: int | None = None,
                     extent_tolerance_px: int = _EXTENT_TOLERANCE_PX,
                     min_item_gap_px: int = _MIN_ITEM_GAP_PX,
                     end_tail_gap_px: int = _END_TAIL_GAP_PX) -> ItemIndex:
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

    segmentations = tuple(
        segment_frame(frame, content_band=content_band, like_template=like_template,
                      like_threshold=like_threshold)
        for frame in frames)
    shifts = tuple(
        estimate_shift(frames[i], frames[i + 1], content_band=content_band,
                       trust_window_px=trust_window_px)
        for i in range(len(frames) - 1))

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
    else:
        blocks, assembly_failures = _assemble(
            _observations(segmentations, offsets), at_scroll_top=at_scroll_top, card_x=card_x,
            extent_tolerance_px=extent_tolerance_px, min_item_gap_px=min_item_gap_px,
            band_y0=segmentations[0].band[0])
        failures.extend(assembly_failures)

    placed = [(seg, off) for seg, off in zip(segmentations, offsets) if off is not None]
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
        tail_gap_px=tail_gap, failures=tuple(failures), identity=identity)


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
