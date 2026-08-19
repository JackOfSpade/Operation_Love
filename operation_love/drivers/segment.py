"""Row-projection segmentation of ONE Hinge profile frame into card BLOCKS.

This is the primitive ops/OPENER-REDESIGN.md 5.3/5.4 stands on: turn a screencap into an
ordered list of blocks, each with a known y-extent, a known "did we actually see both of its
edges" flag, and a class (heart-bearing => selectable, heartless => context). Everything else
in Part B — the driver-owned index table (5.3), closed-loop scrolling sized against local card
spacing (5.10.1), per-item crops and the post-tap signature check (5.6) — consumes this.

Deliberately a leaf module: pure functions over frame BYTES plus explicit calibration
parameters, no device, no I/O, no global state, and no HingeDriver import. hinge.py imports
THIS module, never the other way round at import time (see `_match_glyph`'s deferred import in
`segment_frame` for the one runtime call back into it, and why it is deferred).

WHY ROW PROJECTION AND NOT CONTOURS
-----------------------------------
Hinge cards are rounded rectangles laid out in a single column on a contrasting page
background, with a constant-height gutter between them. That makes a horizontal projection —
"is this scanline page background across the card's full width?" — a complete description of
the layout. No contour finding, no Canny, no adaptive thresholding, and above all nothing that
learns: detection stays deterministic (owner rule).

THE TWO MEASURED TRAPS (ops/OPENER-REDESIGN.md 5.4, amendments one and two)
--------------------------------------------------------------------------
Both were measured against 115 real hand-scrolled frames and are the reason this module is
more than three lines of numpy.

1. Colour alone is NOT sufficient, in two separate ways.

   (a) The page background is a vertical GRADIENT, ~RGB(255,254,253) at y=300 down to
       ~(243,243,243) at y=2100. A single global "background colour" constant is wrong by ~11
       grey levels by the bottom of the content band, which is larger than any sane tolerance.
       So the reference is LOCAL TO THE ROW, read off the left and right page margins beside
       the card (see `_row_background_profile`) — those 53px strips are page background by
       construction on every row, so they track the gradient exactly and for free.

   (b) Even with a correct per-row reference, 11.5% of rows sampled INSIDE confirmed cards
       still pass a "matches background" test — a white prompt card near the top of the screen
       is within a couple of grey levels of the near-white page background there. One such
       contiguous span ran 192 rows and would have split a single card in two, shifting every
       index below it. The discriminator is the gutter's LENGTH: 192 is nearly 4x the canonical
       52-53px, so a background run may cut a block only when its height lands in the canonical
       window (see `_GUTTER_TOLERANCE_PX`) or when a card's own top corner starts on the row
       directly below it (see the next section, which is the list-boundary case and the only
       other way a run becomes a boundary).

2. "Any pixel differs from background" MISSES real gutters. A real gutter was missed that way
   because a narrow element (measured spanning only x=30..216, i.e. Hinge's floating pass-X
   overlapping the gutter) kept the row looking occupied. So a row counts as CARD only when its
   non-background pixels span close to the card's FULL 53..1026 width — see
   `_CARD_ROW_MIN_SPAN_FRAC`, which sits between the widest observed gutter intruder and the
   narrowest genuine card row.

THE THIRD PIECE OF EVIDENCE: A CARD'S OWN ROUNDED CORNER (doc 5.4, failure class three)
---------------------------------------------------------------------------------------
Gutters alone can never bound the FIRST or the LAST item of a profile: there is no gutter above
item 1 (Hinge's filter-chips header and the name row sit there instead) and none below item N.
An earlier revision of this module asked the caller to declare those two rows
(`list_top_y`/`list_bottom_y`). That was wrong, and the calibration corpus proved it: the
header above item 1 is PROFILE-DEPENDENT. The declared row that recovers item 1 has to land in
the background gap above the first card, and those gaps are 554..697 on one profile and
410..479 on the other — DISJOINT windows, so no constant, and nothing derived from
`identity_top_name_band`, can serve both. A fixed constant silently truncated item 1 on the
second profile while still reporting `ok`.

The root cause was not the constant, it was that the module had no way to see a card's own
edge. It does: Hinge cards are ROUNDED rectangles, and the row projection already computed for
trap 2 measures the corner arc for free. At a card's top edge the non-background span starts at
`card_width - 2r` and grows monotonically to the FULL card width exactly `r` rows later, where
`r` is the corner radius; a bottom edge is the same arc mirrored. So an edge row is a card edge
when the span it starts from and the number of rows it takes to reach full width AGREE on one
radius (see `_corner_radius`).

That test is self-evidencing — it needs no background above the row and no declaration from the
caller — and the "reaches the FULL card width" requirement is what makes it safe: only the card
itself spans x=53..1026, since anything drawn inside a card is inset by the card's own padding.

  [corpus: at the 219 card edges that a canonical gutter independently proves are card edges,
  the two radius estimates agree to within 2px on 209 tops and 213 bottoms (95-97%); the rest
  are rejected, which only leaves an edge unobserved. Against that, every non-card row that
  begins or ends a card-run in the 148 analysed frames — Hinge's header strip and its name row
  — measures a "radius" of 68.5, 72.5, 75.0 or 487px, i.e. never within 43px of the accepted
  18..25 window. And on the 224 frames whose content runs flush into the analysed band's top
  edge and the 84 that run flush into its bottom, the test fired ZERO times: a band edge slicing
  a card mid-height does not look like a corner.]

This buys the two things the declaration was for, without the declaration: item 1's top edge is
observed whenever its corner is on-screen, and item N's bottom edge is observed from the card's
own bottom corner with page background below it. It also promotes every block the analysed band
happens to clip inside a gutter, which the old code could only report as PARTIAL.

WHAT THIS MODULE REFUSES TO GUESS
---------------------------------
Fail loud, never silently degrade (owner rule):

  * cv2/numpy missing, or bytes that do not decode, or a missing like template => raise
    `SegmentationError`. An empty block list must never be reachable by any route that also
    means "we could not look", because "no cards here" is a legitimate and very different
    answer.
  * A block edge is reported as OBSERVED only when a canonical-length gutter or the card's own
    rounded corner produced it. A background run that merely runs off the end of the analysed
    band is NOT trusted on its own, even when it is longer than a gutter: from one frame we
    cannot tell "the card ended here" from "the card's blank tail continues past the band edge".
    Such a block is PARTIAL, and its `BlockEdge` carries the observed run length so a later
    layer can decide with more context.
  * Two hearts inside one block is a segmentation FAILURE, reported as one (block kind
    `BLOCK_AMBIGUOUS` plus an entry in `FrameSegmentation.failures`), never resolved by picking
    one. Same for a heart that lands in no block at all.

GREYSCALE, NOT COLOUR: the whole driver's vision layer (`_match_glyph`, `_downsample`, `_band`)
works in greyscale, the page-margin reference makes the gradient a non-issue, and the measured
separations below are all comfortable in luminance alone. A colour-matching card region would
read as background here, but that is exactly the case the gutter-length gate already covers.

WHAT THIS MODULE DOES NOT DECIDE, AND WHICH LAYER HAS TO
--------------------------------------------------------
Named here because each one was raised against this module by a validation pass over 148 real
frames and each is genuinely unanswerable from ONE frame. Do not "fix" them here.

  * `ok is True` is NOT "this is a profile screen". Segmentation is not screen recognition: fed
    the out-of-likes paywall it returns one honest PARTIAL block, no hearts, no failures — and
    so do 15 of the 148 ordinary profile frames, which complete no block either. Screen identity
    is `_identity_of`/`_screen_is`'s job and must be settled BEFORE segmenting.
  * The item INDEX. One frame yields at most 2 complete blocks (measured 15/126/7 frames with
    0/1/2 across 148). The 1..N list doc 5.3 needs is assembled ACROSS frames by the scroll
    ledger; this module feeds that, it does not close it.
  * Whether a frame is at scroll-top. Nothing in a single frame's geometry distinguishes "the
    header is above card 1" from "we scrolled past it". Doc 5.5 already requires an affirmative
    confirmation from `identity_band`'s filter-chips signal, and that remains a precondition for
    counting — it is simply no longer a precondition for BOUNDING item 1, which is what the
    corner test changed.
  * Whether a card is ANIMATED. A block over animated content measured a 1px bottom-row wobble
    across 58 consecutive zero-scroll frames, which is invisible to any single frame and will
    defeat doc 5.6's crop signature. Detecting it needs two frames at the same scroll offset.
  * WHICH heartless block a context block is. `BLOCK_CONTEXT` is one bucket, and the corpus put
    at least four distinct things in it (three heartless blocks on one profile, one on the
    other, sized 215..1144px). Doc 2.4 wants the endorsement block EXCLUDED rather than sent,
    and no geometry here separates it from the vitals block. Until some deterministic signal
    does, the policy layer must either send every context block or find that signal elsewhere.
  * That Hinge's scroll-top header chrome is chrome. It becomes a block like anything else. On
    every scroll-top frame in the corpus it came back PARTIAL — neither of its ends is gutter-
    or corner-bounded, so it can reach neither tier — but that is an observation about how
    Hinge draws its header, not a guarantee. A caller that has confirmed scroll-top should
    ignore blocks above the topmost card corner rather than trust that they self-exclude.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace


# =====================================================================================
# Calibration constants.
#
# Every number here was MEASURED, not chosen. Sources:
#   [doc]   ops/OPENER-REDESIGN.md 5.10, "Card geometry (5.4) is measurable" table + the two
#           amendments underneath it (115 real frames, one profile, hand-scrolled).
#   [corpus] a re-measurement of the gitignored calibration captures with THIS module's own row
#           test (148 frames total: 115 hand-scrolled, 24 bot-scrolled at read_scroll_frac 0.16,
#           and 9 bot-scrolled at 0.55, over two profiles). Geometry and counts only — the
#           frames are real people's profiles and never leave ops/calibration/.
# =====================================================================================

# Card inset from each screen edge, in px on the calibrated 1080x2400 Pixel 7a. [doc: "Left /
# right card margin | 53px each (screen 1080px)"]. Cards therefore occupy x 53..1026 inclusive,
# and the two 53px strips outside that are page background on EVERY row — which is what makes
# the per-row gradient reference below possible.
_CARD_MARGIN_PX = 53

# How far inside each page margin the background reference is sampled from. 8px of clearance at
# both ends of the strip keeps the probe off the phone's rounded screen corners on one side and
# off the card's own antialiased edge + drop shadow on the other, leaving 37 columns per side
# (74 total) to take a median over. [corpus: the peak-to-peak spread across those 74 columns is
# 0-2 grey levels on essentially every row of every frame, so the median is a very stable
# reference — see _MARGIN_PROBE_MAX_SPREAD for what happens when it is not.]
_MARGIN_PROBE_INSET_PX = 8

# Spread (grey levels) allowed across the margin probe before that row's background reference
# is declared untrustworthy. Measured as the MIDDLE-HALF spread (p75 - p25), not peak-to-peak,
# and that choice is load-bearing rather than statistical taste: doc 5.4's amendment-two element
# was measured spanning x=30..216, which reaches INTO the left page margin. Peak-to-peak would
# let those ~15 contaminated columns condemn the row, force it to CARD, and hide the very gutter
# that amendment exists to preserve. The middle half tolerates up to a quarter of the probe
# being occupied at each end, which is exactly the overhang case, while still seeing a margin
# that is genuinely covered. (The reference itself is a median, robust on the same grounds.)
# [corpus: the middle-half spread is 0 at the median, <=1 at p90, <=5 at p99, <=6 at p99.9, with
# a maximum of 119.] 24 is 4x the 99.9th percentile, and fires on 32 of 250,200 content-band
# rows (0.013%), all inside the two scroll-top frames whose header genuinely covers the margin.
# Such rows are treated as CARD rows, which is the conservative direction: it can only MERGE
# blocks (recoverable, and reported), never split one.
_MARGIN_PROBE_MAX_SPREAD = 24

# How far a pixel may sit from its row's background reference and still count as background.
# The vertical gradient is already removed by the per-row reference, so this only has to absorb
# screencap noise plus antialiasing. [corpus: the margin probe's own within-row spread is 0 at
# the median and <=1 at the 90th percentile.] 6 is several times that, while still resolving the
# ~9-level gap between the page background (246) and pure card white (255) at the BOTTOM of the
# content band. It deliberately does NOT resolve that gap at the TOP of the band, where the
# background is ~253.5 and card white is 255 — that is trap 1(b) above, and the gutter-LENGTH
# gate, not this tolerance, is what handles it.
_BACKGROUND_TOLERANCE = 6

# A row counts as CARD when its non-background pixels SPAN at least this fraction of the card
# width (1026 - 53 + 1 = 974px on the calibrated device). Span, not count: a prompt card's blank
# interior is legitimately background-coloured in the middle, and a card's rounded corner rows
# are only non-background between the arcs.
#   [corpus] widest non-background span ever observed INSIDE an accepted gutter: 164px = 0.168 of
#            the card width — which is trap 2's narrow element itself, doc 5.4's x=30..216
#            measurement (Hinge's floating pass-X plus its shadow) clipped to the card band.
#            Narrowest span of a genuine card row ADJACENT to a gutter, over 412 such rows:
#            926px = 0.951, and that is also the 1st percentile (the outermost row of a card's
#            ~20-23px corner arc, inset ~24px per side).
# 0.60 sits near the midpoint of that 0.168 / 0.951 gap, with 0.43 of margin on the intruder side
# and 0.35 on the card side. It is NOT a "close to 1.0" number precisely because the corner
# radius means a real card's first and last rows are never full width.
_CARD_ROW_MIN_SPAN_FRAC = 0.60

# Canonical gutter height between two cards. [doc: "Gutter height between cards | 52-53px
# canonical, 212 of 224 measured gutters within 1px"; corpus: every one of the 206 real
# card-to-card gutters across 139 frames measured EXACTLY 53 with this module's row test.]
_GUTTER_PX = (52, 53)

# Slack around _GUTTER_PX, giving an accepted window of 47..58 rows inclusive. The gutter is a
# fixed layout constant on a fixed-density device, so this is not modelling real variation (the
# corpus shows none) — it is pure measurement slack, for the case where a card's antialiased
# corner arc is too sparse to clear _CARD_ROW_MIN_SPAN_FRAC and gets counted into the gutter, or
# the reverse. The doc's own 12-of-224 gutters that missed 52-53 by more than 1px are the reason
# it is not tighter still.
#
# It is not WIDER because widening it costs safety in the direction that fails silently.
# [corpus: the nearest non-gutter background run is the 60-row whitespace between Hinge's
# scroll-top header and the name below it, on 2 of 139 frames. At a tolerance of 7 that gap is
# accepted, and the header becomes a spurious CONTEXT block whose presence renumbers nothing but
# whose crop is chrome; at 5 it is rejected and those frames instead report an honest PARTIAL
# block. Rejecting a genuine gutter costs a merged block, which shows up immediately as a
# two-heart BLOCK_AMBIGUOUS hard stop — loud, and the direction the owner rule prefers.]
# The 192-row false span inside a card from doc 5.4's amendment one is rejected by more than 3x
# either way.
_GUTTER_TOLERANCE_PX = 5

# A gutter is not merely a canonical-length run with no full-width foreground span: its pixels
# must also agree with the page background read from the side margins on the same rows.  The
# distinction matters for compound photo cards.  Hinge can draw a short caption panel above the
# media inside ONE rounded card, separated by 47px of blank card surface.  Length alone calls
# that internal seam a gutter and splits the caption from its photo.  On the incident frame the
# seam's median grey level is 2 levels away from the page reference on the median row (up to 3),
# while the genuine gutter immediately below is an exact 0-level match on all 53 rows.
#
# [corpus] the page-margin probe's within-row spread is <=1 through p90 and real gutters are the
# page itself, so one grey level absorbs ordinary capture noise without accepting the measured
# card-white seam.  A failure here deliberately MERGES regions, the conservative/loud direction;
# it can never create a new item or renumber anything below it.
_GUTTER_BACKGROUND_LEVEL_TOLERANCE = 1.0

# Corner radius window, in px, that a card's own top/bottom edge row may measure. [doc 5.10:
# "Card corner radius | ~20-23px". corpus: at the 219 edges a canonical gutter independently
# proves are card edges, the radius implied by the edge row's non-background span is 19.0 on 105
# of them, 23.0 on 167, and spread over 19/19.5/20/20.5/21/22/23/23.5/24 in between — never
# below 18.5 and never above 24 once the ramp test below has agreed. Two radii dominate because
# Hinge draws photo cards and text cards with different corners.]
# 18..25 is that measured range plus a 1px cushion at each end for the antialiased first row.
# The window is a FILTER, not an estimate: a row outside it simply leaves the edge unobserved,
# which is the conservative direction (an honest PARTIAL block). It is also nowhere near tight:
# [corpus: the nearest non-card row that begins or ends a card-run — Hinge's scroll-top header
# strip and its name row — implies radii of 68.5, 72.5, 75.0 and 487px, so the nearest false
# positive is 43px outside the window.]
_CARD_CORNER_PX = (18, 25)

# How far the two INDEPENDENT radius estimates may disagree. Estimate one is
# (card_width - span) / 2 at the edge row; estimate two is how many rows the span takes to reach
# the full card width. On a true rounded rect they are the same number, and it is their
# AGREEMENT, not either value alone, that identifies a card edge — which is why a text row
# cannot fake it: its span neither starts near full width nor ramps.
# [corpus: the disagreement is exactly 0 on 195 of 219 gutter-proven top edges, and <=1 on 208;
# bottom edges cluster at 0 and 1.5 (the 1.5 is one specific card's antialiasing, 58 repeat
# observations of it). 2 accepts 209 of 219 tops and 213 of 219 bottoms.] Widening it further
# buys back only the ~10 edges whose corner is overpainted by card content, and those are
# exactly the ones where the "is this a corner" question has no clean answer.
_CARD_CORNER_RAMP_SLACK_PX = 2

# Rows are allowed to dip by this much while walking the arc before the shape is rejected as
# non-monotonic. [corpus: 1px dips occur on real corner arcs from antialiasing; the 5 rejected
# bottom edges dipped by more.] Purely measurement slack on a shape that is monotonic by
# construction.
_CARD_CORNER_DIP_PX = 1

# A confirmed scroll-top has one layout edge that ordinary frames do not: profile-name chrome,
# then a page-background gap, then the first media card. A pale photo can make one of that
# card's corner ramps indistinguishable from the page, so `_corner_radius` correctly refuses the
# generic edge even though the initial row still has the measured 18..25px corner span. The
# fallback below is deliberately much narrower than the generic corner rule: it is opt-in for
# frame 0 of an affirmatively confirmed scroll-top, the gap is inside the measured 69..143px
# header range (with raster slack), and the candidate below it must be an exactly square media
# card ending at a canonical gutter with one normally placed Hinge heart.
_SCROLL_TOP_MEDIA_GAP_PX = (64, 160)
_SCROLL_TOP_MEDIA_SQUARE_TOLERANCE_PX = 2
_SCROLL_TOP_MEDIA_RAMP_ROWS = 8
_SCROLL_TOP_MEDIA_MIN_RAMP_GAIN_PX = 12
_SCROLL_TOP_MEDIA_HEART_BOTTOM_INSET_PX = (60, 120)

# Consecutive media cards in the live Rebecca capture have a 98..101px page-background gap,
# rather than the ordinary 52..53px gutter.  It is still a real boundary: its upper edge lands
# 89px below the prior card's lower-right like heart, inside the independently measured 60..120px
# heart-bottom inset.  Keep this deliberately narrow.  A generic 98..101px blank span is not a
# boundary, and a heart at another vertical position is not evidence that its card ended here.
_HEART_ANCHORED_MEDIA_GAP_PX = (90, 112)

# A centred, low-contrast photo can leave a few rows of its outer card surface
# indistinguishable from the page. In the reported live capture this made the visible page gap
# between a prompt and the following photo 60px rather than the usual 52..53px. This is NOT a
# wider generic gutter window: it applies only to this tiny near-gutter range when matched hearts
# prove one Hinge item lies on each side.
_HEART_SEPARATED_NEAR_GUTTER_PX = (59, 64)

# Which half of the screen a like heart may appear on. Hinge puts it at the bottom-RIGHT of
# every likeable item (hinge.py:770, enforced by _match_glyph's own side filter). Named rather
# than inlined so the one place this assumption lives is greppable.
_HEART_SIDE = "right"

# _match_glyph's non-max-suppression loop runs a fixed 12 iterations (hinge.py:843), so the
# RETURNED list can never exceed 12 hits. That is NOT the same guarantee as "12 iterations means
# 12 hits": a peak on the wrong side is retired from the correlation surface (so it can never be
# found again) but is never appended to the result (hinge.py:849-852's `continue`), so a mix of
# on-heart and off-side peaks can fully spend the 12-iteration budget while returning FEWER than
# 12 hits. The `len(hearts) >= _MATCH_GLYPH_HIT_CAP` check below therefore only catches the
# narrowest saturation case -- all 12 iterations landing a same-side peak -- and is BLIND to a
# saturated frame that also hits the wrong side even once, which returns some count under 12 and
# passes this check silently. There is no cheap way to close that gap from this module: it would
# need _match_glyph to report how many iterations it actually spent, not just the survivors,
# which touches its ~10 other call sites and is out of scope here. Accepted as a documented
# residual risk rather than a soundness guarantee. Real frames show 1-3 hearts at a time, so
# tripping this check at all means the match is misbehaving, not that the profile is unusual.
_MATCH_GLYPH_HIT_CAP = 12


# =====================================================================================
# Result vocabulary. Plain string constants rather than an Enum, matching the house style
# (there is no Enum anywhere in operation_love/) and keeping these directly loggable to the
# JSONL debug log without a serializer.
# =====================================================================================

# --- block classes, per doc 5.3 -----------------------------------------------------
BLOCK_SELECTABLE = "selectable"   # fully observed, exactly one heart: a photo or prompt card
BLOCK_CONTEXT = "context"         # fully observed, no heart: the vitals block (5.3)
BLOCK_PARTIAL = "partial"         # at least one edge not observed; true extent unknown here
BLOCK_AMBIGUOUS = "ambiguous"     # >1 heart in one block: a segmentation FAILURE, not a choice

# --- why a block edge is where it is ------------------------------------------------
EDGE_GUTTER = "gutter"            # a canonical-length page-background run between two cards
EDGE_CARD_CORNER = "card_corner"  # the card's OWN rounded corner is visible on this row: the
                                  # span it starts from and the rows it takes to reach the full
                                  # card width agree on one radius (see `_corner_radius`)
EDGE_SCROLL_TOP_MEDIA = "scroll_top_media"  # guarded frame-0 recovery for a pale square photo
EDGE_HEART_ANCHORED_MEDIA_GUTTER = "heart_anchored_media_gutter"
                                  # a tall, page-coloured gap starts at a proven media item's
                                  # lower edge even though its corner is obscured
EDGE_HEART_SEPARATED_NEAR_GUTTER = "heart_separated_near_gutter"
                                  # a 59..64px page-coloured gap lies between proven hearts
EDGE_BAND_EDGE = "band_edge"      # the block runs straight into the analysed band's edge
EDGE_BACKGROUND_RUN = "background_run"  # page background beyond this edge, but neither
                                  # gutter-length nor corner-confirmed, so what it MEANS is
                                  # unknown from one frame: "the card ended" and "the card's
                                  # blank tail continues off-band" look identical

# --- why a background run was or was not treated as a block boundary -----------------
RUN_GUTTER = "gutter"
RUN_CARD_EDGE = "card_edge"       # not gutter-length, but a card's own top corner starts on the
                                  # row directly below it — the list-top case, where Hinge's
                                  # header chrome sits above item 1 instead of a gutter
RUN_SCROLL_TOP_MEDIA = "scroll_top_media"  # distinct provenance for the guarded recovery
RUN_HEART_ANCHORED_MEDIA_GUTTER = "heart_anchored_media_gutter"
RUN_HEART_SEPARATED_NEAR_GUTTER = "heart_separated_near_gutter"
RUN_TOO_SHORT = "too_short"
RUN_TOO_LONG = "too_long"
RUN_CLIPPED = "clipped"           # touches the analysed band's edge, so its length is a lower
                                  # bound only and cannot be tested against the gutter window
RUN_CARD_SURFACE = "card_surface"  # canonical length, but its median level differs from the
                                   # page: blank space inside one compound card, not a gutter


class SegmentationError(RuntimeError):
    """Segmentation could not run at all: no cv2/numpy, undecodable frame bytes, a missing like
    template, or a degenerate band.

    Distinct from a segmentation FAILURE (two hearts in one block, an unassigned heart), which
    is reported in `FrameSegmentation.failures` rather than raised — those still carry a full
    structured result that a human or the hub can read to diagnose the frame. This exception
    means there is no result to report, and it exists so that "we could not look" can never be
    mistaken for "there are no cards on this frame".
    """


@dataclass(frozen=True)
class BackgroundRun:
    """A maximal run of consecutive background-like rows inside the analysed band.

    `y1` is EXCLUSIVE (so `height` is `y1 - y0` with no off-by-one), matching every other
    y-extent in this module. `widest_intruder_px` is the widest horizontal span of
    non-background pixels found on any row of the run — 0 for a clean gutter, and up to a few
    hundred px when Hinge's floating pass-X overlaps it (trap 2). It is recorded so a
    validation pass can see HOW close a run came to failing the span test without re-deriving.
    """
    y0: int
    y1: int
    kind: str
    widest_intruder_px: int
    # Median absolute grey-level distance between the card band and the same rows' page-margin
    # reference.  This is the affirmative page-background check for RUN_GUTTER; older callers
    # constructing records positionally remain source-compatible through the default.
    median_level_delta: float = 0.0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class BlockEdge:
    """One end of a block: where it is, whether we actually SAW it, and what put it there.

    `observed` is the load-bearing field. It is normally True for `EDGE_GUTTER`, a canonical
    background run, or `EDGE_CARD_CORNER`, the card's own rounded corner. The sole third form is
    `EDGE_SCROLL_TOP_MEDIA`: the separately gated frame-0 conjunction documented above, which
    combines a partial corner with square-card, heart and lower-gutter geometry. It is False for
    `EDGE_BAND_EDGE` (the block runs flush into the
    analysed band) and for `EDGE_BACKGROUND_RUN`, where there IS page background beyond the edge
    but nothing says whether the card ended or merely went blank, which from one frame is the
    same picture.

    `run_px` carries the length of the background run adjacent to this edge (a lower bound when
    that run leaves the band), or None when none was seen. `corner_px` is the measured corner
    radius, set only for `EDGE_CARD_CORNER`, and is there so a validation pass can see WHY an
    edge was trusted without re-deriving it.
    """
    y: int
    observed: bool
    kind: str
    run_px: int | None = None
    corner_px: int | None = None


@dataclass(frozen=True)
class Block:
    """One card-like region of the frame.

    `y0`/`y1` are frame rows, half-open. `x0`/`x1` are the card's horizontal extent, which is
    the same constant band for every block on Hinge (the 53px page margins) but is carried
    per-block so a caller cropping an item never has to re-derive it.

    `hearts` holds EVERY like-glyph centre that fell inside this block, not just the first.
    Discarding all but the first is precisely the information loss doc 5.4 calls out at every
    existing `_match_glyph` call site, and the whole point of `BLOCK_AMBIGUOUS` is that two
    hearts here means the segmentation is wrong, so the count must survive.
    """
    y0: int
    y1: int
    x0: int
    x1: int
    top: BlockEdge
    bottom: BlockEdge
    kind: str
    hearts: tuple[tuple[int, int], ...]
    reason: str

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def complete(self) -> bool:
        """Both edges actually observed, so this block's extent is trustworthy from this frame
        alone — the precondition for cropping it or measuring card spacing off it."""
        return self.top.observed and self.bottom.observed

    @property
    def heart(self) -> tuple[int, int] | None:
        """The single heart centre, or None. Deliberately None for a block with TWO hearts as
        well as for one with none: an ambiguous block has no defensible answer, and returning
        `hearts[0]` here would smuggle back exactly the guess `BLOCK_AMBIGUOUS` exists to
        prevent."""
        return self.hearts[0] if len(self.hearts) == 1 else None


@dataclass(frozen=True)
class FrameSegmentation:
    """Everything one frame yielded: the blocks, the evidence behind them, and what went wrong.

    `runs` includes the REJECTED background runs with their rejection reason, and `hearts` is
    the raw detector output before assignment. Both exist so a validation pass can answer "why
    did this frame segment the way it did" from the result alone, without re-running the
    detector or re-deriving thresholds.

    `frame_digest` is the sha256 of the exact BYTES this result was measured from, and it is the
    only field here that says anything about frame IDENTITY rather than frame content. It exists
    because everything downstream (`item_index.py`'s page space, `item_crops.py`'s rects) is a
    set of coordinates that are meaningless against any other capture, while being perfectly
    well-formed against one: a validation pass measured that handing the crop pass a DIFFERENT
    same-size capture produced ten confidently wrong crops with zero failures, because the only
    guard was the decoded frame size and every Hinge screencap on this device is 1080x2400. A
    digest makes that check exact instead of nearly useless — same bytes, same frame, no
    tolerance and no false positives.
    """
    frame_size: tuple[int, int]        # (width, height) in px
    frame_digest: str                  # sha256 hex of the frame bytes this was measured from
    band: tuple[int, int]              # analysed rows, half-open
    card_x: tuple[int, int]            # (x0, x1) card extent, half-open
    blocks: tuple[Block, ...]
    runs: tuple[BackgroundRun, ...]
    hearts: tuple[tuple[int, int], ...]
    unassigned_hearts: tuple[tuple[int, int], ...]
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """No segmentation failure was detected on this frame. A False here is a hard stop for
        anything that navigates by block index — never a reason to fall back to a fixed
        coordinate (doc 5.3: "Treat a missing table as a hard stop").

        True is much weaker than it looks and must never be read as "this is a Hinge profile".
        [corpus: the out-of-likes PAYWALL frame returns True with one PARTIAL block and no
        hearts, and so do 15 of the 148 profile frames.] Confirm the screen first, separately."""
        return not self.failures

    @property
    def selectable(self) -> tuple[Block, ...]:
        """The heart-bearing, fully-observed blocks, top to bottom. This is the subset the model
        is ever offered a choice from (doc 5.3); the driver's own index space still spans every
        block, selectable or not."""
        return tuple(b for b in self.blocks if b.kind == BLOCK_SELECTABLE)


def _require_vision():
    """Import cv2 + numpy or raise. Every other vision helper in this codebase degrades to a
    fixed-coordinate fallback when cv2 is absent; segmentation has no such fallback, because
    the thing it would have to fake is "which items exist", and inventing that is how a run
    likes the wrong photo."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001 — surfaced as SegmentationError, never swallowed
        raise SegmentationError(
            "segmentation needs opencv-python + numpy (extra: operation-love[hinge]); "
            f"import failed: {exc}") from exc
    return cv2, np


def _band_rows(content_band: tuple[float, float], height: int) -> tuple[int, int]:
    """Row range `(r0, r1)` (half-open) of `content_band`'s `(y0, y1)` height fractions.

    Mirrors `hinge._content_rows` exactly, including its clamp to a non-empty range. It is
    duplicated rather than imported because this module must stay importable BEFORE hinge.py
    (hinge.py is the importer), and four lines of arithmetic is a cheaper price than an import
    cycle. If _content_rows' contract ever changes, this must follow.
    """
    y0, y1 = content_band
    r0 = max(0, min(height - 1, round(y0 * height)))
    r1 = max(r0 + 1, min(height, round(y1 * height)))
    return r0, r1


def _row_background_profile(gray, *, card_x0: int, probe_inset: int, np):
    """Per-row page-background reference and its spread, read off the two page margins.

    Returns `(reference, spread)`, both one value per frame row: the MEDIAN grey level across
    the `2 * (card_x0 - 2 * probe_inset)` margin columns, and their MIDDLE-HALF spread
    (p75 - p25). This is the answer to trap 1(a): the reference tracks the background gradient
    row by row, at zero modelling cost, because the margins beside a card ARE the page
    background by construction.

    Both statistics are deliberately robust rather than extremal — see
    `_MARGIN_PROBE_MAX_SPREAD` for the measured case (an element overhanging the card into the
    margin) that a mean or a peak-to-peak range would get wrong.
    """
    width = gray.shape[1]
    left = gray[:, probe_inset: card_x0 - probe_inset]
    right = gray[:, width - card_x0 + probe_inset: width - probe_inset]
    probe = np.concatenate((left, right), axis=1).astype(np.float32)
    lo, mid, hi = np.percentile(probe, (25, 50, 75), axis=1)
    return mid, hi - lo


def _classify_rows(gray, *, card_x0: int, card_x1: int, reference, spread, tolerance: float,
                   min_span_frac: float, max_spread: float, np):
    """Per-row `(is_card, nonbg_span)` for the whole frame.

    `nonbg_span` is the distance from the leftmost to the rightmost non-background pixel within
    the card band — the horizontal EXTENT of whatever occupies the row, not how many pixels it
    covers. Extent is the right measure on both sides of the decision:

      * a card's rounded-corner rows are non-background only between the two arcs, and a prompt
        card's text rows are mostly background pixels, yet both extend nearly the full width;
      * the pass-X floating over a gutter is a solid blob, yet extends only ~350px.

    A row whose margin probe is too uneven to trust (`spread > max_spread`, i.e. the page margin
    beside it is not page background at all) is forced to CARD.
    That is the conservative direction: an untrustworthy row can then only merge two blocks
    (visible as a PARTIAL/oversized block, and recoverable) instead of splitting one, which
    would silently renumber every item below it.
    """
    band = gray[:, card_x0:card_x1].astype(np.float32)
    nonbg = np.abs(band - reference[:, None]) > tolerance
    any_nonbg = nonbg.any(axis=1)
    first = np.argmax(nonbg, axis=1)
    last = nonbg.shape[1] - 1 - np.argmax(nonbg[:, ::-1], axis=1)
    span = np.where(any_nonbg, last - first + 1, 0).astype(np.int32)
    is_card = (span >= min_span_frac * (card_x1 - card_x0)) | (spread > max_spread)
    return is_card, span


def _background_runs(is_card, span, level_delta, *, r0: int, r1: int, gutter_lo: int,
                     gutter_hi: int, gutter_level_tolerance: float, np,
                     ) -> list[BackgroundRun]:
    """Maximal runs of background-like rows, classified by length and actual page agreement.

    A run that touches either end of the analysed band is `RUN_CLIPPED` regardless of its
    measured height: we only see a lower bound on its length, so it cannot be tested against the
    gutter window at all. Calling a clipped 53-row run a gutter would be a guess, and the one it
    would most often get wrong is the blank tail of a card that continues past the band edge.

    `RUN_CARD_EDGE` is never produced here — length cannot see it. `segment_frame` re-labels a
    `RUN_TOO_LONG` run to it after asking `_corner_radius` about the row below it.
    """
    runs: list[BackgroundRun] = []
    y = r0
    while y < r1:
        if is_card[y]:
            y += 1
            continue
        start = y
        while y < r1 and not is_card[y]:
            y += 1
        clipped = start == r0 or y == r1
        height = y - start
        median_delta = float(np.median(level_delta[start:y])) if height else 0.0
        if clipped:
            kind = RUN_CLIPPED
        elif height < gutter_lo:
            kind = RUN_TOO_SHORT
        elif height > gutter_hi:
            kind = RUN_TOO_LONG
        elif median_delta > gutter_level_tolerance:
            kind = RUN_CARD_SURFACE
        else:
            kind = RUN_GUTTER
        runs.append(BackgroundRun(y0=start, y1=y, kind=kind,
                                  widest_intruder_px=int(span[start:y].max()) if height else 0,
                                  median_level_delta=median_delta))
    return runs


def _corner_radius(is_card, span, *, y: int, step: int, lo_row: int, hi_row: int,
                   card_width: int, radius_px: tuple[int, int], ramp_slack: int,
                   dip_px: int) -> float | None:
    """The card corner radius measured at row `y`, or None if `y` is not a card's own edge row.

    `step` is +1 to test a TOP edge and -1 to test a BOTTOM edge: either way we walk INTO the
    card, away from the page background, and the arc's inset shrinks to nothing as we go. The
    walk is confined to `[lo_row, hi_row)` — the block's own segment — so it can never read the
    shape of a neighbouring card across a gutter.

    Two independent estimates of the same radius have to agree (see `_CARD_CORNER_RAMP_SLACK_PX`
    for why agreement rather than either value is the test):

      1. from the edge row's own span. A rounded rect's outermost row is non-background only
         between its two corner arcs, so `span = card_width - 2r`.
      2. from the ramp length. Walking into the card, the span grows monotonically and reaches
         the FULL card width exactly `r` rows later, where the arcs meet the straight edges.

    Requiring the full card width — not "most of it" — is the safety property that makes this
    usable as trusted evidence: on Hinge only the card itself reaches x=53..1026, because
    anything drawn INSIDE a card is inset by the card's own padding. A rounded element within a
    photo therefore cannot forge a card edge no matter how card-shaped it is.
    """
    s0 = int(span[y])
    if s0 >= card_width:
        return None                       # already full width: a card's interior, not its edge
    radius = (card_width - s0) / 2.0
    lo, hi = radius_px
    if not lo <= radius <= hi:
        return None
    prev = s0
    for n in range(1, int(hi) + ramp_slack + 1):
        row = y + step * n
        if not lo_row <= row < hi_row or not is_card[row]:
            return None                   # the card ended before the arc closed: not an arc
        s = int(span[row])
        if s < prev - dip_px:
            return None                   # the shape turns back in: not a corner arc
        prev = s
        if s >= card_width:
            return radius if abs(n - radius) <= ramp_slack else None
    return None


def _leading_low_contrast_media_edge(
        runs: list[BackgroundRun], run_index: int, *, hearts: tuple[tuple[int, int], ...],
        is_card, span, card_x0: int, card_x1: int, radius_px: tuple[int, int],
        dip_px: int) -> bool:
    """Whether one rejected long run safely bounds the first square media card.

    This is not a relaxed corner detector. It is the conjunction supplied by an already
    confirmed Hinge scroll-top: header content above this first long run, a square card below it,
    a canonical gutter at the card bottom, one lower-right heart, and the visible half of a
    measured corner ramp. Any missing fact keeps the run rejected as before.
    """
    run = runs[run_index]
    if run.kind != RUN_TOO_LONG or not (
            _SCROLL_TOP_MEDIA_GAP_PX[0] <= run.height <= _SCROLL_TOP_MEDIA_GAP_PX[1]):
        return False
    if any(previous.kind in (RUN_GUTTER, RUN_CARD_EDGE, RUN_SCROLL_TOP_MEDIA)
           for previous in runs[:run_index]):
        return False
    if not any(bool(value) for value in is_card[:run.y0]):
        return False
    next_gutter = next((candidate for candidate in runs[run_index + 1:]
                        if candidate.kind == RUN_GUTTER), None)
    if next_gutter is None:
        return False
    width = card_x1 - card_x0
    if abs((next_gutter.y0 - run.y1) - width) > _SCROLL_TOP_MEDIA_SQUARE_TOLERANCE_PX:
        return False
    card_hearts = tuple((x, y) for x, y in hearts if run.y1 <= y < next_gutter.y0)
    if len(card_hearts) != 1:
        return False
    heart_x, heart_y = card_hearts[0]
    inset = next_gutter.y0 - heart_y
    if (heart_x < card_x0 + width // 2
            or not (_SCROLL_TOP_MEDIA_HEART_BOTTOM_INSET_PX[0] <= inset
                    <= _SCROLL_TOP_MEDIA_HEART_BOTTOM_INSET_PX[1])):
        return False

    top = run.y1
    if top + _SCROLL_TOP_MEDIA_RAMP_ROWS >= next_gutter.y0 or not is_card[top]:
        return False
    initial_span = int(span[top])
    implied_radius = (width - initial_span) / 2.0
    if not (radius_px[0] <= implied_radius <= radius_px[1]):
        return False
    ramp = [int(span[y]) for y in range(top, top + _SCROLL_TOP_MEDIA_RAMP_ROWS + 1)]
    if any(not is_card[y] for y in range(top, top + _SCROLL_TOP_MEDIA_RAMP_ROWS + 1)):
        return False
    if any(after < before - dip_px for before, after in zip(ramp, ramp[1:])):
        return False
    return ramp[-1] - ramp[0] >= _SCROLL_TOP_MEDIA_MIN_RAMP_GAIN_PX


def _heart_anchored_media_gutter(
        run: BackgroundRun, *, hearts: tuple[tuple[int, int], ...],
        gutter_level_tolerance: float) -> bool:
    """Whether a non-canonical page-coloured run is the lower edge of a media item.

    This is intentionally not a wider gutter window. A tall blank surface can occur inside one
    compound card, but a Hinge like heart sits at the lower-right of its one item. When the
    next page-coloured run begins at that measured heart-bottom inset, the heart proves the
    upper card ended there; a genuine 98..101px Rebecca gap is then enough to cut. The lower
    card need not already show its heart -- the first card must become complete before the next
    one has scrolled into view.
    """
    if run.kind != RUN_TOO_LONG or not (
            _HEART_ANCHORED_MEDIA_GAP_PX[0] <= run.height <=
            _HEART_ANCHORED_MEDIA_GAP_PX[1]):
        return False
    if run.median_level_delta > gutter_level_tolerance:
        return False
    anchored = [
        heart for heart in hearts
        if (_SCROLL_TOP_MEDIA_HEART_BOTTOM_INSET_PX[0] <= run.y0 - heart[1]
                <= _SCROLL_TOP_MEDIA_HEART_BOTTOM_INSET_PX[1])
    ]
    return len(anchored) == 1


def _heart_separated_near_gutter(
        runs: list[BackgroundRun], run_index: int, *, hearts: tuple[tuple[int, int], ...],
        gutter_level_tolerance: float) -> bool:
    """Whether a slightly overlong page gap is proven to separate two Hinge items.

    The standard length rule rightly refuses to infer a boundary from a 59..64px blank span by
    itself. Here, independently matched hearts above and below the page-coloured run provide the
    missing proof: Hinge has one like heart per item, so they cannot belong to one card.
    """
    run = runs[run_index]
    if run.kind != RUN_TOO_LONG or not (
            _HEART_SEPARATED_NEAR_GUTTER_PX[0] <= run.height <=
            _HEART_SEPARATED_NEAR_GUTTER_PX[1]):
        return False
    if run.median_level_delta > gutter_level_tolerance:
        return False
    boundary_kinds = {
        RUN_GUTTER, RUN_CARD_EDGE, RUN_SCROLL_TOP_MEDIA,
        RUN_HEART_ANCHORED_MEDIA_GUTTER,
    }
    previous = next((candidate for candidate in reversed(runs[:run_index])
                     if candidate.kind in boundary_kinds), None)
    following = next((candidate for candidate in runs[run_index + 1:]
                      if candidate.kind in boundary_kinds), None)
    # A heart farther down the profile is not evidence about this gap.  The lower card must
    # expose its own heart before the next independently confirmed boundary.
    if following is None:
        return False
    upper_start = previous.y1 if previous is not None else 0
    return (any(upper_start <= y < run.y0 for _, y in hearts)
            and any(run.y1 <= y < following.y0 for _, y in hearts))


def _block_edge(*, y: int, cut: BackgroundRun | None, gap: int,
                corner: float | None) -> BlockEdge:
    """One end of a block, resolved to a `BlockEdge`.

    Precedence, strongest evidence first:

      1. a canonical-length GUTTER on this side. Exact on every card pair in the corpus (206 of
         206 measured exactly 53px), so it settles the edge on its own.
      2. the guarded confirmed-scroll-top media conjunction, when explicitly present on the cut.
      3. the card's own rounded CORNER on the edge row. This is what bounds item 1 and item N,
         which no gutter can reach, and it also rescues any block the analysed band happened to
         clip inside a gutter.
      4. page background beyond the edge that is neither. We saw where the content stopped, we
         just cannot trust WHY it stopped — a card that ended and a card whose blank tail runs
         off the band are the same picture — so the length is reported and the edge is not
         observed.
      5. nothing at all: the block runs flush into the analysed band.

    `gap` is how many page-background rows were trimmed off this end of the segment; it is 0 by
    construction at any cut (the run was maximal), so it only carries information at the band's
    two ends.
    """
    if cut is not None and cut.kind == RUN_GUTTER:
        return BlockEdge(y=y, observed=True, kind=EDGE_GUTTER, run_px=cut.height)
    if cut is not None and cut.kind == RUN_SCROLL_TOP_MEDIA:
        return BlockEdge(y=y, observed=True, kind=EDGE_SCROLL_TOP_MEDIA, run_px=cut.height)
    if cut is not None and cut.kind == RUN_HEART_ANCHORED_MEDIA_GUTTER:
        return BlockEdge(y=y, observed=True, kind=EDGE_HEART_ANCHORED_MEDIA_GUTTER,
                         run_px=cut.height)
    if cut is not None and cut.kind == RUN_HEART_SEPARATED_NEAR_GUTTER:
        return BlockEdge(y=y, observed=True, kind=EDGE_HEART_SEPARATED_NEAR_GUTTER,
                         run_px=cut.height)
    run_px = cut.height if cut is not None else (gap or None)
    if corner is not None:
        return BlockEdge(y=y, observed=True, kind=EDGE_CARD_CORNER, run_px=run_px,
                         corner_px=round(corner))
    if run_px:
        return BlockEdge(y=y, observed=False, kind=EDGE_BACKGROUND_RUN, run_px=run_px)
    return BlockEdge(y=y, observed=False, kind=EDGE_BAND_EDGE, run_px=None)


def _classify_block(*, top: BlockEdge, bottom: BlockEdge, hearts) -> tuple[str, str]:
    """`(kind, reason)` for one block, from its edges and its heart count.

    Precedence is deliberate and is the whole of doc 5.3's two-tier rule plus the fail-loud one:

      1. more than one heart  -> BLOCK_AMBIGUOUS. Outranks everything, including PARTIAL,
         because a two-heart block is a statement that the gutter search was WRONG, and that is
         worth surfacing even on a block whose extent was already untrusted.
      2. an unobserved edge   -> BLOCK_PARTIAL. Its true extent is unknown from this frame, so
         it is neither safe to crop nor safe to count as an item, whatever its heart count.
      3. exactly one heart    -> BLOCK_SELECTABLE (a photo or prompt card).
      4. no heart             -> BLOCK_CONTEXT (the vitals block: read and referenced, never
         offered as a choice).
    """
    if len(hearts) > 1:
        return BLOCK_AMBIGUOUS, (
            f"{len(hearts)} like hearts inside one block at y={[y for _, y in hearts]} — every "
            "likeable Hinge item carries exactly one, so a gutter between them was missed")
    if not top.observed or not bottom.observed:
        detail = "; ".join(
            f"{name} edge unobserved ({edge.kind}"
            + (f", {edge.run_px}px of background beyond it)" if edge.run_px is not None else ")")
            for name, edge in (("top", top), ("bottom", bottom)) if not edge.observed)
        return BLOCK_PARTIAL, (
            f"neither a gutter nor a card corner bounds this end, true extent unknown from this "
            f"frame — {detail}; {len(hearts)} heart(s) visible")
    if hearts:
        return BLOCK_SELECTABLE, (
            f"both edges observed ({top.kind}/{bottom.kind}) and exactly one like heart at "
            f"y={hearts[0][1]}")
    return BLOCK_CONTEXT, (
        f"both edges observed ({top.kind}/{bottom.kind}) and no like heart — a context block "
        "(Hinge's vitals block is the known instance: ~215px tall, never selectable)")


def segment_frame(frame: bytes, *, content_band: tuple[float, float], like_template,
                  like_threshold: float,
                  recover_leading_low_contrast_media: bool = False,
                  card_margin_px: int = _CARD_MARGIN_PX,
                  margin_probe_inset_px: int = _MARGIN_PROBE_INSET_PX,
                  margin_probe_max_spread: float = _MARGIN_PROBE_MAX_SPREAD,
                  background_tolerance: float = _BACKGROUND_TOLERANCE,
                  card_row_min_span_frac: float = _CARD_ROW_MIN_SPAN_FRAC,
                  gutter_px: tuple[int, int] = _GUTTER_PX,
                  gutter_tolerance_px: int = _GUTTER_TOLERANCE_PX,
                  gutter_background_level_tolerance: float =
                  _GUTTER_BACKGROUND_LEVEL_TOLERANCE,
                  card_corner_px: tuple[int, int] = _CARD_CORNER_PX,
                  card_corner_ramp_slack_px: int = _CARD_CORNER_RAMP_SLACK_PX,
                  card_corner_dip_px: int = _CARD_CORNER_DIP_PX) -> FrameSegmentation:
    """Segment one screencap into blocks. The entry point; everything above is its machinery.

    `content_band` is the `(y0, y1)` height-fraction band cards may occupy — pass the driver's
    `self.content_band` (the config-merged value), NOT `spec.content_band`, or an operator
    override is silently ignored.

    `like_template` and `like_threshold` are REQUIRED and have no defaults on purpose: the
    calibrated heart glyph and its 0.75 correlation floor have exactly one home each
    (`HINGE_SPEC.templates["like"]` and `hinge._LIKE_MATCH_THRESHOLD`, whose module comment
    carries the measurement that justifies the number), and a default here would be a second
    copy free to drift. Pass `driver._template("like")` and `hinge._LIKE_MATCH_THRESHOLD`.

    `recover_leading_low_contrast_media` is False by default and may be set only for frame 0
    after the caller independently confirmed Hinge's scroll-top chrome. It enables the guarded
    pale-square-first-photo conjunction above; it does not relax ordinary corner detection.

    THERE IS DELIBERATELY NO `list_top_y` / `list_bottom_y`, and that is a fix, not an omission.
    Doc 5.4's fourth failure class — "Above the first card is Hinge's filter-chips header, and
    below the last is the dark bottom nav. Neither is page background, so the first and last
    blocks need special-casing rather than the interior gutter rule" — is handled here by
    detecting the card's own rounded corner (see `_corner_radius` and the module docstring),
    which needs nothing from the caller. An earlier revision took those two rows as parameters
    and it was measurably wrong:

      [corpus: the two profiles' scroll-top frames put item 1's top at row 697 and row 479, so
      the declared row that recovers it has to fall in 554..696 on one and 410..478 on the other
      — DISJOINT. The recommended constant (600, the bottom of `identity_top_name_band`) is
      correct on the first profile and silently returns an 853px truncation of the second
      profile's 974px card, with `ok` still True. Worse, the self-check that was supposed to
      catch a wrong declaration only fired on 22 of 139 deliberately-mis-declared frames: it can
      prevent a confidently wrong extent, but it cannot detect a wrong row.]

      [corpus: with the corner test and no declaration at all, the same two frames cut at the
      143-row and the 69-row chrome gap respectively and return item 1 at its true 697..1671 and
      479..1453, complete and SELECTABLE. Across the whole corpus item 1 goes from complete on 0
      frames to complete on 4 of the 6 it appears in (one profile) and 1 of 4 (the other), and
      item N — which needed the mirror declaration — from 0 to 58 of 63 and 1 of 4. The frames
      that still miss are the ones where the card's far edge is genuinely off-band, and they
      report an honest PARTIAL.]

    A caller must still confirm scroll-top before COUNTING items (doc 5.5, via `identity_band`'s
    filter-chips signal) — that requirement is untouched. What changed is that bounding item 1
    no longer depends on it, so a mistaken confirmation can no longer produce a wrong extent.

    Raises `SegmentationError` when it cannot look at all. Returns a result with a non-empty
    `failures` when it looked and found something contradictory. Note that an empty `failures`
    says only "nothing here contradicted itself" — it is emphatically not "this is a Hinge
    profile screen"; see the module docstring.
    """
    cv2, np = _require_vision()

    if like_template is None:
        # _match_glyph returns [] for a None template, which would read here as "this frame has
        # no hearts" and reclassify every card on it as a CONTEXT block. Refuse instead.
        raise SegmentationError(
            "no like-glyph template: segmentation cannot tell a heartless context block from a "
            "card whose heart we simply failed to load a template for")

    img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SegmentationError(
            f"frame bytes did not decode as an image ({len(frame)} bytes)")
    height, width = img.shape

    if width < 2 * card_margin_px + 1 or card_margin_px < 2 * margin_probe_inset_px + 1:
        raise SegmentationError(
            f"frame is {width}px wide with a {card_margin_px}px card margin and a "
            f"{margin_probe_inset_px}px probe inset — no page margin left to read the "
            "background reference from")

    r0, r1 = _band_rows(content_band, height)
    if r1 - r0 < 2:
        raise SegmentationError(
            f"analysed band is empty: content_band {content_band} on a {height}-row frame "
            f"-> rows [{r0}, {r1})")

    card_x0, card_x1 = card_margin_px, width - card_margin_px
    reference, spread = _row_background_profile(
        img, card_x0=card_x0, probe_inset=margin_probe_inset_px, np=np)
    is_card, span = _classify_rows(
        img, card_x0=card_x0, card_x1=card_x1, reference=reference, spread=spread,
        tolerance=background_tolerance, min_span_frac=card_row_min_span_frac,
        max_spread=margin_probe_max_spread, np=np)
    # The row classifier intentionally tolerates small colour differences so prompt whitespace
    # is not fragmented.  Gutter acceptance gets a stricter, aggregate check: the median of the
    # whole card band must actually match the page margins on those same rows.
    band_level = np.median(img[:, card_x0:card_x1].astype(np.float32), axis=1)
    level_delta = np.abs(band_level - reference)

    gutter_lo = min(gutter_px) - gutter_tolerance_px
    gutter_hi = max(gutter_px) + gutter_tolerance_px
    runs = _background_runs(
        is_card, span, level_delta, r0=r0, r1=r1, gutter_lo=gutter_lo,
        gutter_hi=gutter_hi, gutter_level_tolerance=gutter_background_level_tolerance, np=np)

    # Deferred import, and the ONLY runtime dependency this module has on hinge.py. Deferring it
    # keeps the import graph acyclic (hinge.py imports segment.py) and, as a bonus, resolves
    # _match_glyph as an attribute at CALL time, so the existing tests that monkeypatch
    # `hinge._match_glyph` keep working through this path too.
    #
    # _match_glyph swallows every failure into an empty list, which would be dangerous here —
    # "no hearts" reclassifies cards as context blocks. Only TWO of its failure paths are ruled
    # out by this point: cv2 is importable (we already used it above) and the template is not
    # None. That does NOT rule out every way `[]` can mean "something broke" rather than
    # "genuinely no hearts": _match_glyph's own `except Exception: return []` (hinge.py:856-857)
    # is a bare catch-all, so a runtime failure INSIDE its try block -- e.g. cv2.matchTemplate
    # raising on a shape/dtype mismatch -- reads here exactly like a heartless frame and silently
    # demotes every selectable card on it to context. That residual risk is ACCEPTED, not
    # eliminated: closing it for real would mean narrowing _match_glyph's except clause or having
    # it report a runtime failure separately from a genuine "no hearts" read, which touches its
    # ~10 other call sites and is out of scope here.
    from . import hinge as _hinge
    hearts = tuple(
        (int(x), int(y)) for x, y in _hinge._match_glyph(
            frame, like_template, side=_HEART_SIDE, threshold=like_threshold,
            y_band=content_band))

    failures: list[str] = []
    if len(hearts) >= _MATCH_GLYPH_HIT_CAP:
        # Not "a lot of hearts" — at the cap we do not know how many there really are, and any
        # index built on it is a guess. See _MATCH_GLYPH_HIT_CAP's own comment: this only
        # catches the narrowest saturation case (12 consecutive same-side peaks), not saturation
        # in general. Real frames show 1-3.
        failures.append(
            f"like-glyph matcher returned {len(hearts)} hits, at or above its "
            f"{_MATCH_GLYPH_HIT_CAP}-iteration non-max-suppression cap — the true count is "
            "unknown, so no index may be built from this frame")

    # WHICH BACKGROUND RUNS CUT A BLOCK. The ordinary path has exactly two kinds of evidence:
    #
    #   * a canonical-length gutter. This is amendment one of doc 5.4 — a background run that is
    #     not gutter-shaped (the measured 192-row bright span inside a single card) is recorded
    #     in `runs` with its reason and then ignored, so the rows either side of it stay in the
    #     SAME block.
    #   * a longer run with a card's own top corner on the row directly below it. This is doc
    #     5.4's list-boundary case: at scroll-top the gap between Hinge's header chrome and item
    #     1 is not a gutter (measured 143 rows on one profile, 69 on the other) and never can be,
    #     so the length gate alone leaves item 1 permanently unbounded.
    #
    # A run SHORTER than the gutter window is never allowed to cut, corner or no corner: the
    # gutter is a fixed layout constant, so a sub-canonical gap between two cards contradicts the
    # layout and is evidence of a detection error rather than of a boundary. Refusing it can only
    # merge two cards, which surfaces immediately as a two-heart BLOCK_AMBIGUOUS hard stop;
    # accepting it could split one, which silently renumbers every item below.
    # [corpus: 8 non-gutter interior runs across 148 frames, all too_long, zero too_short. The
    # corner test cuts 6 of the 8 — every scroll-top chrome gap — and correctly declines the
    # other 2, which are the whitespace between the header strip and the name row above it.]
    # The opt-in third label is the narrow scroll-top media conjunction described by
    # `_leading_low_contrast_media_edge`; unlike either ordinary rule it is never available to
    # an unconfirmed or later frame. The fourth is an equally narrow interior media rule: a
    # 90..112px run at the one measured bottom inset of the preceding like heart. It does not
    # widen generic gutter acceptance, so ordinary long card-white spans remain whole.
    cuts: list[BackgroundRun] = []
    for i, run in enumerate(runs):
        if run.kind == RUN_GUTTER:
            cuts.append(run)
            continue
        if run.kind != RUN_TOO_LONG:
            continue
        corner = _corner_radius(
            is_card, span, y=run.y1, step=1, lo_row=r0, hi_row=r1,
            card_width=card_x1 - card_x0, radius_px=card_corner_px,
            ramp_slack=card_corner_ramp_slack_px, dip_px=card_corner_dip_px)
        if corner is not None:
            runs[i] = replace(run, kind=RUN_CARD_EDGE)
        elif (recover_leading_low_contrast_media
              and _leading_low_contrast_media_edge(
                  runs, i, hearts=hearts, is_card=is_card, span=span,
                  card_x0=card_x0, card_x1=card_x1, radius_px=card_corner_px,
                  dip_px=card_corner_dip_px)):
            runs[i] = replace(run, kind=RUN_SCROLL_TOP_MEDIA)
        elif _heart_anchored_media_gutter(
                run, hearts=hearts,
                gutter_level_tolerance=gutter_background_level_tolerance):
            runs[i] = replace(run, kind=RUN_HEART_ANCHORED_MEDIA_GUTTER)
        elif _heart_separated_near_gutter(
                runs, i, hearts=hearts,
                gutter_level_tolerance=gutter_background_level_tolerance):
            runs[i] = replace(run, kind=RUN_HEART_SEPARATED_NEAR_GUTTER)
        else:
            continue
        cuts.append(runs[i])

    blocks: list[Block] = []
    assigned: set[tuple[int, int]] = set()
    # Segment bounds: band start, then each cut, then band end. Each pair of consecutive bounds
    # is one candidate block region, and `cut_above`/`cut_below` carry the run that produced it
    # (None at the two band edges) so each edge can be resolved against its own evidence.
    bounds: list[tuple[int, int]] = []
    prev_y = r0
    for cut in cuts:
        bounds.append((prev_y, cut.y0))
        prev_y = cut.y1
    bounds.append((prev_y, r1))

    for seg_i, (seg_y0, seg_y1) in enumerate(bounds):
        cut_above = cuts[seg_i - 1] if seg_i > 0 else None
        cut_below = cuts[seg_i] if seg_i < len(cuts) else None

        # Trim page background off the ends of the segment. By construction a cut-bounded end
        # needs no trimming (the run was maximal), so this only ever bites at the two band edges
        # — where it turns "the band happened to start mid-gap" into a block that starts at real
        # content. The trimmed rows are captured BEFORE hearts can widen the extent, because the
        # corner test has to be asked about the card's real edge row.
        rows = [y for y in range(seg_y0, seg_y1) if is_card[y]]
        if not rows:
            continue
        content_y0, content_y1 = rows[0], rows[-1] + 1
        y0, y1 = content_y0, content_y1
        gap_above, gap_below = y0 - seg_y0, seg_y1 - y1

        # Hearts are assigned by SEGMENT, not by the trimmed extent. A prompt card whose bottom
        # rows are blank white can put its heart below the last row that passed the span test,
        # and dropping it there would silently demote a selectable card to a context block.
        block_hearts = tuple(h for h in hearts if seg_y0 <= h[1] < seg_y1)
        assigned.update(block_hearts)
        if block_hearts:
            # Then widen the extent to cover the full glyph, so a crop of this block contains the
            # heart that made it selectable — CLAMPED to the segment, so at the analysed band's
            # edge the widening can be cut short and the glyph box then crosses the block
            # boundary. [corpus: 6 of 230 heart assignments, every one of them on a PARTIAL block
            # that is not croppable anyway.] The guarantee is unconditional only for a complete
            # block, where the heart sits ~89px above the bottom and the widening never binds.
            th = like_template.shape[0]
            y0 = max(seg_y0, min(y0, min(y for _, y in block_hearts) - th // 2))
            y1 = min(seg_y1, max(y1, max(y for _, y in block_hearts) + th // 2 + 1))

        # The corner test is asked only about an UNWIDENED end: once a heart has pushed the
        # extent past where the card's content stopped, the reported row is no longer the row the
        # corner was measured on, and calling that edge observed would report a wrong extent.
        top = _block_edge(
            y=y0, cut=cut_above, gap=gap_above,
            corner=_corner_radius(
                is_card, span, y=content_y0, step=1, lo_row=seg_y0, hi_row=seg_y1,
                card_width=card_x1 - card_x0, radius_px=card_corner_px,
                ramp_slack=card_corner_ramp_slack_px, dip_px=card_corner_dip_px,
            ) if y0 == content_y0 else None)
        bottom = _block_edge(
            y=y1, cut=cut_below, gap=gap_below,
            corner=_corner_radius(
                is_card, span, y=content_y1 - 1, step=-1, lo_row=seg_y0, hi_row=seg_y1,
                card_width=card_x1 - card_x0, radius_px=card_corner_px,
                ramp_slack=card_corner_ramp_slack_px, dip_px=card_corner_dip_px,
            ) if y1 == content_y1 else None)

        kind, reason = _classify_block(top=top, bottom=bottom, hearts=block_hearts)
        blocks.append(Block(y0=y0, y1=y1, x0=card_x0, x1=card_x1, top=top, bottom=bottom,
                            kind=kind, hearts=block_hearts, reason=reason))

    for block in blocks:
        if block.kind == BLOCK_AMBIGUOUS:
            failures.append(
                f"block y={block.y0}..{block.y1} is ambiguous: {block.reason}")

    unassigned = tuple(h for h in hearts if h not in assigned)
    if unassigned:
        # A heart with no block around it means the row test called the heart's own card rows
        # background, or the heart landed inside an accepted gutter. Either way the block table
        # would be missing an item, which is the index corruption doc 5.3 says we cannot
        # tolerate — so it is a failure, not a note.
        failures.append(
            f"{len(unassigned)} like heart(s) at y={[y for _, y in unassigned]} fell outside "
            "every block — a selectable item exists that segmentation cannot bound")

    return FrameSegmentation(
        frame_size=(width, height), frame_digest=hashlib.sha256(frame).hexdigest(),
        band=(r0, r1), card_x=(card_x0, card_x1),
        blocks=tuple(blocks), runs=tuple(runs), hearts=hearts,
        unassigned_hearts=unassigned, failures=tuple(failures))
