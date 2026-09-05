"""Automated, AI-labeled corpus campaign for Hinge's still-photo dwell bound.

Sibling of `tools/hinge_video_bound.py`, which is the design of record's harness
(ops/STILL-PHOTO-DISCRIMINATOR.md section 4).  Read that module first: this one reuses its
manifest schema, its hazard-randomized burst planner, its byte-exact-run arithmetic, its
threshold enforcement and its artifact/paste machinery unchanged.  What differs is exactly one
thing, and it is the thing section 2 objection 1 killed a whole design over:

    THE LABEL CHANNEL IS CIRCULAR, ON PURPOSE, BY OWNER DECISION (2026-08-21).

The sibling gets its ground truth from the owner tapping each card and typing what happened.
This tool has no owner in the loop.  It labels each card from the mute-glyph matcher, the burst's
own pixel-exactness, and the crop classifier -- the very signals the accept rule reads.  So the
held-out video false-accept count it produces is ZERO BY CONSTRUCTION and proves nothing about
the population the bound exists to quantify: a static video whose mute control never rendered
would be labeled `photo` here and then accepted, and both halves of that mistake are invisible to
a measurement that made them with the same pixels.  Nothing in this module hides that.  The
manifest records `ground_truth_channel: ai_mute_glyph_circular_v1`, `human_ground_truth: false`
and the owner's acceptance phrase verbatim; `measure` refuses a campaign that omits any of them;
and the caveat is printed at the top of every report and frozen inside the artifact's own digest
as `label_blind_spot`.

WHAT THE CAMPAIGN IS ACTUALLY WORTH, then, is three things the circularity does not touch:

  * `max_video_exact_run_s` measured on cards that were PLAYING.  A playing video's byte-exact
    run is a physical fact about the phone and the app, not about who labeled it, and it is the
    number the shipped dwell window is a multiple of.
  * the still-photo FALSE-REFUSAL rate, which bounds usefulness rather than safety and cannot be
    inflated by a circular label: a still photo the rule refuses is refused whatever it is called.
  * a persisted corpus.  Every frame is kept with its digest, so a later owner-labeled pass can
    re-label these exact cards and recompute a real bound without spending the screen time again.

WHY THE CARD RECTS COME FROM PER-FRAME SEGMENTATION AND NOT FROM THE ITEM INDEX
(root cause of the 2026-08-21 silent-empty live failure; do not undo this).  The first version of
this tool took its frames and its rects from `HingeDriver._index_captured_items`, hooked during
`_capture_current`.  That hook is only ever reached when the read is an ENUMERATION read, and
`_item_enumeration_blocker` turns enumeration off whenever `apps.hinge.targeting_calibration` is
absent -- a PRODUCT-POLICY gate ("no numbered item list would have a usable consumer"), not a
perception one.  On the shipped config that key is not configured, so the hook never fired, every
profile produced zero frames and zero cards, and the run passed five real profiles while
persisting nothing.  The dependency was structural and no retry could have helped.

The item index exists to COUNT items into absolute page ordinals so a model's numbered choice can
be navigated to.  This campaign never numbers anything for a model and never navigates: it needs
a card RECT on the frame in front of it and nothing else.  `segment.segment_frame` answers
exactly that from one frame, needs no scroll-top confirmation to bound a card (see its docstring
on the corner test), cannot alias across frames because it never looks at two, and depends on no
targeting policy.  Measured against the 15 real screencaps of the failed run, it returns one
complete selectable card rect on 14 of them, the mute screen runs over a complete ROI on every
one of those, and it caught a real video at 0.99999976.

WHAT THIS TOOL DOES TO THE PHONE, unlike its read-only sibling: it drives it.  Every gesture goes
through HingeDriver's own guarded, humanized primitives -- `_scroll_down_one` with a step, lane
and dwell drawn by the driver's own behaviour policy, and `dislike()` or `like()` for the advance.
There is no raw `adb` event anywhere in this module and no fallback path that degrades to one:
the standing owner rule is best humanized interaction or fail loudly, so a driver refusal halts
the campaign instead of being retried by a cruder route.  The dwell burst itself sends NOTHING;
it is `driver._screencap` on a hazard-drawn schedule, which is what makes it a dwell.

THE ADVANCE ACTION IS CHOSEN PER RUN AND HAS NO DEFAULT.  `--advance pass` spends a real Pass per
profile; `--advance like` spends a real, permanent Like and demands a second confirmation phrase
of its own.  Neither ever touches a paid control, a Rose, or a super-like (standing owner rule:
the bot does ordinary like/pass only).

STOP CONDITIONS follow the auto-mode owner rules: no volume caps, no quota stops.  The campaign
halts on an unrecognized screen, on a driver refusal, on SIGINT/SIGTERM, and -- non-negotiable
since the live failure -- on the FIRST profile that persists no frame or yields no measurable
card.  There is no strike allowance: a profile that produced nothing is not a recordable profile,
and `advance_deck` cannot be called without a `ProfileHarvest` licence that was built by the
persistence path and re-verified against the digests on disk.  Every halt leaves the screen
exactly where it is, records the reason, writes an INCOMPLETE manifest, and exits nonzero.

OFFLINE ADJUDICATION OF `unsure` CARDS (owner decision 2026-08-21).  A campaign may carry an
optional `adjudications.json` in which a second reader -- a vision model, applying the owner's
rule "videos all have a mute icon, no mute icon means picture" -- re-labels cards the capture
could not resolve.  `measure` applies it before forming any denominator, reports the capture's
own tallies beside the adjudicated ones, and `emit` freezes the applied verdicts and the
adjudicator's identity inside the artifact's own digest.  No file means byte-identical behaviour.

Two things about it are worth stating plainly here rather than in a commit message.  First, the
transitions are ONE-WAY (`unsure -> photo`, `unsure -> video`, `photo -> video`) and nothing can
demote a video, because the owner's rule is exactly the inference section 2 objection 2 refuted:
the mute control auto-hides, and complete-ROI frames of PROVEN videos score 0.267-0.305 before it
renders, so its absence is evidence of nothing.  Second, adjudication does NOT repair the
circularity -- it deepens it.  A second model reading the same mute glyph off the same pixels is
the same channel with more steps, so `label_blind_spot` stands unchanged and the video-accept
count remains zero by construction.

    python -m tools.hinge_video_bound_auto capture --out ops/calibration/videoauto_<UTC> \\
        --advance pass --confirmation I_ACCEPT_CIRCULAR_AI_LABELED_STILL_PHOTO_BOUND
    python -m tools.hinge_video_bound_auto measure ops/calibration/videoauto_<UTC>
    python -m tools.hinge_video_bound_auto emit    ops/calibration/videoauto_<UTC>

PRIVACY: identical to the sibling.  Every frame is a real person's dating profile, output is
confined to gitignored ops/calibration/, and the tool refuses any directory git does not ignore.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from operation_love import targeting_policy as _policy
from operation_love.drivers import hinge as _hinge
from operation_love.drivers.hinge import (
    HingeActionError, HingeDriver, video_mute_screen_reason)
from operation_love.drivers.item_crops import (
    STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC, card_center_offset_frac, dwell_exact_over_rect)
from operation_love.drivers.scroll_step import frac_for_step_px
from operation_love.drivers.item_type_preflight import PHOTO, WRITTEN, classify_crop
from operation_love.drivers.segment import BLOCK_SELECTABLE, SegmentationError, segment_frame
from operation_love.human import human_cooldown, human_delay
from operation_love.private_files import (
    atomic_write_private_bytes, atomic_write_private_text, ensure_private_dir)
from tools import hinge_video_bound as bound
from tools._devicelock import run_holding_the_device

_TOOL_VERSION = "2"
_CAMPAIGN_MODE = "ai_labeled_automated_v2"
_DEFAULT_PACKAGE = "co.hinge.app"

# The circular channel and its acceptance phrase are resolved from the policy module exactly as
# the sibling resolves them, then asserted equal to the sibling's copy.  Two modules that
# disagreed about which channel this is would produce a campaign one of them could not read, and
# during development the policy constants and these tools land in separate commits.
CIRCULAR_CHANNEL = getattr(_policy, "STILL_PHOTO_BOUND_CIRCULAR_CHANNEL",
                           bound.CIRCULAR_GROUND_TRUTH_CHANNEL)
CIRCULAR_ACCEPTANCE = getattr(_policy, "STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE",
                              bound.CIRCULAR_ACCEPTANCE)
if (CIRCULAR_CHANNEL != bound.CIRCULAR_GROUND_TRUTH_CHANNEL
        or CIRCULAR_ACCEPTANCE != bound.CIRCULAR_ACCEPTANCE):
    raise ImportError(
        "operation_love.targeting_policy and tools.hinge_video_bound disagree about the circular "
        "still-photo label channel or its acceptance phrase; a campaign captured under one and "
        "measured under the other would be silently unreadable")

LABEL_BLIND_SPOT = bound.LABEL_BLIND_SPOT
# Re-exported from the shared machinery so a reader of this tool can find the adjudication
# contract without knowing which of the two modules implements it.
ADJUDICATION_FILENAME = bound.ADJUDICATION_FILENAME
LEGAL_ADJUDICATIONS = bound.LEGAL_ADJUDICATIONS

ADVANCE_PASS = "pass"
ADVANCE_LIKE = "like"
ADVANCE_ACTIONS = (ADVANCE_PASS, ADVANCE_LIKE)
# Deliberately a second, DIFFERENT phrase from --confirmation, and deliberately the same phrase
# tools/hinge_calibrate.py already uses for the same authorization: a real Like is permanent and
# cannot be retracted, so accepting the circular measurement risk must not silently also accept
# spending real Likes.  Duplicated as a literal rather than imported because importing
# hinge_calibrate would pull its whole capture stack into a tool that does not use it; the test
# suite pins the two strings equal.
LIKE_CONFIRMATION = "I_ACCEPT_REAL_PRIORITY_LIKE_SEND_RISK"

LABEL_VIDEO = "video"
LABEL_PHOTO = "photo"
# A Hinge prompt/text card. Its own bucket rather than `unsure`, because `unsure` has to keep
# meaning "cannot tell": campaign 2 put 25 of 58 cards in `unsure` and a vision review found 16
# of the first 20 were simply TEXT cards the classifier had correctly refused to call
# photographic. Written cards are excluded from both denominators exactly as before -- a prompt
# card is not a photograph and the bound is about photo targeting -- they just stop hiding the
# cards that genuinely could not be resolved. Terminal: adjudication may not touch it.
LABEL_WRITTEN = "written"
LABEL_UNSURE = "unsure"
LABELS = (LABEL_VIDEO, LABEL_PHOTO, LABEL_WRITTEN, LABEL_UNSURE)

# Hinge autoplays a video card ONLY at or near the centre of the screen (owner-provided domain
# fact, 2026-08-21). So a motionless burst is evidence of "not a video" only for a card that was
# inside that zone the whole time; an off-centre video simply never starts and holds byte-exact
# forever. The zone itself is defined once, in item_crops, and imported by the driver, the
# offline replay and this tool alike.
CENTER_BAND_FRAC = STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC
# How many corrective scrolls one candidate gets before the campaign gives up on centring it and
# lets it stay unsure. Hazard-drawn per card, never a constant (auto-mode owner rule).
_CENTERING_STEPS_SPAN = (1, 3)
# The read leaves the page deep inside a profile -- centring the LAST station puts it at an
# arbitrary offset -- and Hinge's floating like heart is not drawn at every scroll position while
# the pass X is. `_require_deck_confirmed` needs BOTH glyphs, so it refused and campaign 3 halted
# at profile 14 with the screen scrolled onto a centred video prompt card. The guard was right;
# the walk simply never put the page back. This is the bounded, hazard-drawn number of guarded
# rewinds spent trying to before the advance is attempted anyway.
_REPOSITION_ATTEMPTS_SPAN = (1, 3)

# The card-rect crops this tool persists ARE the content, so the band that would select content
# out of a full screencap is the whole image here.  Recorded in the manifest so the sibling's
# `_card_stat` measures byte-exactness over exactly the card rect and nothing else -- which is
# C2's statistic.  A whole-screen band would let a static chrome region next to a playing video,
# or a second card sharing the frame, decide a card's verdict; the sibling's own hold-test row
# blocks exist because that failure is real.
_CARD_CROP_BAND = (0.0, 1.0)

# How many dwell STATIONS one profile is walked in.  Hazard-drawn per profile under the standing
# rule that no timing or count parameter may be a fixed constant.  A station is one read position
# where the screen is held still for a burst; the read-scroll between them is the driver's own
# behaviour-policy step, so consecutive stations show different cards.
_STATIONS_SPAN = (3, 6)
_READ_SCROLL_SETTLE_S = getattr(_hinge, "_READ_SCROLL_SETTLE_S", 0.4)

# The re-attach probe. A centred byte-exact burst proves Hinge was ASKED to play the card and
# that nothing moved; it does NOT prove the media answered. A video that was stalled, buffering,
# never attached or ended without looping emits no frame for as long as anybody watches, which is
# exactly the population an AI-labeled campaign cannot see (its label and its accept rule read
# the same pixels). Leaving the autoplay band and re-entering it is the event Hinge re-attaches
# and restarts media on, so every station takes a SECOND burst after that round trip and a card
# is labeled `photo` only if both bursts agree. Mirrors the driver's own constants and reasoning
# (`_REATTACH_EXIT_BAND_MULTIPLE_SPAN` in hinge.py) so a corpus stays comparable with production.
_REATTACH_EXIT_BAND_MULTIPLE_SPAN = (1.6, 2.4)
_REATTACH_MAX_EXIT_BAND_FRAC = 0.35


class AutoBoundRefused(bound.VideoBoundRefused):
    """The campaign could not be started, or a captured campaign is not readable."""


class CampaignHalt(RuntimeError):
    """An unrecognized screen or a driver refusal stopped the run. Nothing further is touched."""


class CampaignSignalled(CampaignHalt):
    """SIGINT or SIGTERM arrived. Same contract as any other halt: stop, record, write."""


class EmptyProfileHalt(CampaignHalt):
    """A profile persisted no frame or produced no measurable card.

    Its own type because it is the failure the 2026-08-21 live run had no defence against: the
    campaign recorded five profiles that contained nothing at all and spent a real Pass on each.
    There is deliberately NO strike allowance -- one empty profile means the perception path is
    not seeing what it thinks it is seeing, and every further profile would be another real
    gesture bought with nothing.
    """


# =====================================================================================
# The label rule (pure)
# =====================================================================================

def label_card(*, glyph_hit: bool, dwell_exact: bool | None,
               mute_screens_complete: bool | None,
               classifier_verdict: str | None,
               centered: bool | None,
               reattach_probe_ran: bool | None = None,
               reattach_dwell_exact: bool | None = None,
               reattach_mute_screens_complete: bool | None = None,
               reattach_centered: bool | None = None) -> tuple[str, str]:
    """Return `(label, reason)` for one card from observations only. No I/O, no driver.

    The order is the whole rule and is not interchangeable:

      1. A complete-ROI mute-glyph match at or above the shipped threshold, in ANY frame the card
         was seen in, is affirmative VIDEO evidence and outranks everything below it.  The
         matcher's silence proves nothing (the control auto-hides), but its voice is definite.
      2. A card rect that is not byte-exact across the un-interacted burst moved while nothing
         touched the screen, which is VIDEO by the only positive test this design has.  Note this
         is pixel-exactness, never a score: a real autoplaying video's signature strips match
         CONFIDENTLY WRONG, so a threshold on similarity goes bimodal on exactly the population
         it has to catch.
      3. WRITTEN is terminal and is decided before any photo reasoning: a Hinge prompt card is
         not a photograph, needs no autoplay argument, and belongs in its own bucket rather than
         polluting `unsure`. Any video affordance would already have been caught by rungs 1-2,
         which are valid from ANY screen position.
      4. PHOTO requires every positive together and is the only label that requires them:
         byte-exact over the whole burst, a mute screen that RAN complete and clean on every
         burst frame, the card held inside Hinge's autoplay trigger zone for that burst, and a
         crop the classifier calls confidently photographic. The centring rung is a
         PRECONDITION on rung 2's silence, not an extra opinion: off-centre, a video does not
         play, so its stillness says nothing at all.
      5. Everything else is UNSURE, including the case where no dwell observed the card at all.
         Unsure is excluded from both bound denominators, so a missing observation costs the
         campaign a card and can never be mistaken for a passing measurement.

    THE RE-ATTACH LEGS ARE THE SAME RULE APPLIED TWICE, not a new one.  A byte-exact centred
    burst cannot separate a photograph from a video that was stalled, buffering, unattached or
    already ended, because neither emits a frame.  So each station scrolls the card out of the
    autoplay band and back -- the event Hinge re-attaches media on -- and bursts again.  Motion
    in the second burst is VIDEO by rung 2 exactly as motion in the first is; everything else
    about the second burst is a PHOTO requirement in rung 4.  The defaults are None so a caller
    that made no probe lands on UNSURE, which is where a missing observation always lands here.

    Three-valued inputs are compared with `is True` / `is False`, never for truthiness: None
    means the observation was never made, and a card nobody looked at must not be able to borrow
    a verdict from a placeholder.
    """
    if glyph_hit:
        return LABEL_VIDEO, ("a complete-ROI mute-control match at or above the shipped "
                             "threshold was seen on this card")
    if dwell_exact is False:
        return LABEL_VIDEO, ("the card rect changed across the un-interacted burst, so this "
                             "media animates with nothing touching the screen")
    if reattach_dwell_exact is False:
        return LABEL_VIDEO, ("the card rect changed across the un-interacted burst taken after "
                             "the card re-entered the autoplay band, so this media started "
                             "playing when Hinge re-attached it")
    if dwell_exact is not True:
        return LABEL_UNSURE, ("no un-interacted dwell observed this card, so its byte-exactness "
                              "was never measured")
    if mute_screens_complete is not True:
        return LABEL_UNSURE, ("mute-control screening did not run over a complete ROI, and clean, "
                              "on every burst frame, so a control that rendered could have gone "
                              "unseen")
    if classifier_verdict == WRITTEN:
        return LABEL_WRITTEN, ("the crop classifies as a written prompt card, which is not a "
                               "photograph and is excluded from both denominators")
    if centered is not True:
        return LABEL_UNSURE, ("the card was not held inside Hinge's autoplay trigger zone for "
                              "this burst, so its stillness is equally consistent with a video "
                              "that was never near enough to the centre to start playing")
    if reattach_probe_ran is not True:
        return LABEL_UNSURE, ("no re-attach probe took this card out of the autoplay band and "
                              "back, so a video that was stalled, buffering, unloaded or already "
                              "ended was never asked to start")
    if reattach_dwell_exact is not True:
        return LABEL_UNSURE, ("the re-attach burst never measured this card's byte-exactness, "
                              "so the second look that would catch a non-playing video is "
                              "missing")
    if reattach_mute_screens_complete is not True:
        return LABEL_UNSURE, ("mute-control screening did not run over a complete ROI, and "
                              "clean, on every re-attach burst frame, so a control Hinge redrew "
                              "on re-entry could have gone unseen")
    if reattach_centered is not True:
        return LABEL_UNSURE, ("the card was not back inside Hinge's autoplay trigger zone for "
                              "the re-attach burst, so its stillness there says nothing about "
                              "media nobody asked to play")
    if classifier_verdict != PHOTO:
        return LABEL_UNSURE, "the card crop did not classify as confidently photographic"
    return LABEL_PHOTO, ("byte-exact across BOTH bursts while centred in the autoplay trigger "
                         "zone, including the burst taken after the card was scrolled out of "
                         "that zone and back, screened complete and clean on every frame of "
                         "both, and photographic")


# =====================================================================================
# Per-card observation
# =====================================================================================

@dataclass(frozen=True)
class CardObservation:
    """Everything one card contributed, plus the label those observations produced."""

    rect: tuple[int, int, int, int]
    label: str
    label_reason: str
    glyph_hit: bool
    max_glyph_score: float | None
    dwell_exact: bool
    mute_screens_complete: bool
    classifier_verdict: str
    centered: bool
    center_offset_frac: float
    dwell_span_s: float
    crops: tuple[bytes, ...]
    mute_records: tuple[dict, ...]
    # The re-attach probe's second burst. Defaults keep a station that could not probe on the
    # UNSURE label instead of letting a missing observation read as a passing one.
    reattach_probe_ran: bool = False
    reattach_dwell_exact: bool | None = None
    reattach_mute_screens_complete: bool | None = None
    reattach_centered: bool | None = None
    reattach_center_offset_frac: float | None = None
    reattach_dwell_span_s: float | None = None
    reattach_rect: tuple[int, int, int, int] | None = None
    reattach_crops: tuple[bytes, ...] = ()
    reattach_mute_records: tuple[dict, ...] = ()


def _decode(frame: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AutoBoundRefused("a captured frame could not be decoded as an image")
    return image


def _crop_png(frame: bytes, rect: tuple[int, int, int, int]) -> bytes:
    """Re-encode one card rect as its own lossless PNG.

    Lossless matters: the sibling's `measure` recomputes byte-exactness from these files, so a
    crop that round-tripped through a lossy codec would manufacture differences a still photo
    never had, and the photo false-refusal rate is one of the two numbers this campaign actually
    delivers.
    """
    import cv2

    image = _decode(frame)
    x0, y0, x1, y1 = rect
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AutoBoundRefused(f"card rect {rect!r} does not lie inside a {width}x{height} frame")
    ok, buffer = cv2.imencode(".png", image[y0:y1, x0:x1])
    if not ok:
        raise AutoBoundRefused("a card crop could not be encoded")
    return buffer.tobytes()


def _classify(crop: bytes) -> str:
    """The shipped classifier's RAW verdict (photo / written / unknown), not a boolean.

    The boolean it used to be collapsed `written` and `unknown` into one bucket, which is what
    put 16 real prompt cards into `unsure` in campaign 2. The three-way verdict is what lets a
    text card be labelled as one.
    """
    return classify_crop(crop)


def _screen_card(frame: bytes, rect: tuple[int, int, int, int], matcher) -> dict:
    """Run the SHIPPED per-card mute screen over one frame and record what it saw.

    `video_mute_screen_reason` owns the ROI arithmetic, so passing a recording wrapper as its
    matcher yields the score for the exact ROI the production screen uses instead of a second,
    independently drifting copy of that geometry.  `reason is None` is the only clean outcome:
    the screen RAN over a geometrically complete ROI, decoded, and matched nothing.
    """
    scores: list[float | None] = []

    def _recording_match(image: bytes, roi: tuple[int, int, int, int]):
        screened, score = matcher(image, roi)
        scores.append(None if score is None else float(score))
        return screened, score

    reason = video_mute_screen_reason(frame, rect, match=_recording_match)
    observed = [score for score in scores if score is not None]
    return {"screened": bool(observed), "score": max(observed) if observed else None,
            "reason": reason, "glyph_hit": bool(reason is not None and "mute control" in reason),
            "clean": reason is None, "observational_only": True}


def selectable_card_rects(frame: bytes, *, content_band, like_template,
                          like_threshold) -> list[tuple[int, int, int, int]]:
    """Card rects fully visible in ONE frame, from the shipped per-frame segmenter.

    Only `BLOCK_SELECTABLE` blocks whose BOTH edges were actually observed qualify.  A partial
    block's extent is a lower bound on the real card (see `segment.Block.complete`), so cropping
    it would measure a truncation of the card and call it the card.  Blocks are returned in page
    order; the caller assigns campaign-local card ids and never claims a page ordinal, because
    counting items from one frame is exactly the off-by-N `_confirm_enumeration_top` exists to
    prevent and this campaign has no use for ordinals anyway.
    """
    segmentation = segment_frame(frame, content_band=content_band, like_template=like_template,
                                 like_threshold=like_threshold)
    return [(block.x0, block.y0, block.x1, block.y1) for block in segmentation.blocks
            if block.kind == BLOCK_SELECTABLE and block.complete]


def observe_card(rect, sequence: list[bytes], *, dwell_span_s: float, matcher,
                 frame_height: int, content_band, classify=None,
                 reattach_sequence: list[bytes] | None = None,
                 reattach_span_s: float | None = None,
                 reattach_page_shift_px: int | None = None) -> CardObservation:
    """Label one card rect from the anchor frame plus its un-interacted burst.

    `sequence[0]` is the anchor -- the frame the rect was segmented out of, still on screen when
    the burst began -- and the rest are the burst.  Chaining the anchor into the run is what
    makes the measured span cover the whole time the screen was held, not just the burst's own
    interior.

    C2 is `dwell_exact_over_rect`, the same shared helper the driver and an offline replay use,
    so this campaign measures the exact observation production will make.  C3 is the shipped
    per-card mute screen on every frame of the same sequence.

    Centring is read off the ANCHOR rect, and that is sufficient rather than approximate: no
    input is issued during a burst, so a card that was inside the trigger zone when the burst
    opened cannot have left it. If the page moved anyway, `dwell_exact` is False and the card is
    a video by rung 2 before centring is ever consulted.

    `reattach_sequence` is the SECOND burst, taken after the station scrolled the card out of the
    autoplay band and back. It is measured over the same rect translated by the page displacement
    the probe MEASURED, so the second burst looks at the card the first one looked at rather than
    at whatever now occupies those rows -- the page is tracked, the card is never re-identified.
    Omitting it leaves every `reattach_*` leg unset, which lands the card on UNSURE.
    """
    classify = _classify if classify is None else classify
    mute_records = [_screen_card(frame, rect, matcher) for frame in sequence]
    crops = [_crop_png(frame, rect) for frame in sequence]
    dwell_exact = bool(dwell_exact_over_rect(sequence, rect))
    mute_clean = all(record["clean"] for record in mute_records)
    verdict = str(classify(crops[0]))
    offset = card_center_offset_frac(tuple(rect), frame_height=frame_height,
                                     content_band=content_band)
    centered = abs(offset) <= CENTER_BAND_FRAC
    reattach = _observe_reattach(rect, reattach_sequence, matcher=matcher,
                                 frame_height=frame_height, content_band=content_band,
                                 page_shift_px=reattach_page_shift_px)
    scores = [record["score"] for record in mute_records + list(reattach["mute_records"])
              if record["score"] is not None]
    glyph_hit = any(record["glyph_hit"]
                    for record in mute_records + list(reattach["mute_records"]))
    label, reason = label_card(
        glyph_hit=glyph_hit, dwell_exact=dwell_exact, mute_screens_complete=mute_clean,
        classifier_verdict=verdict, centered=centered,
        reattach_probe_ran=reattach["probe_ran"], reattach_dwell_exact=reattach["dwell_exact"],
        reattach_mute_screens_complete=reattach["mute_screens_complete"],
        reattach_centered=reattach["centered"])
    return CardObservation(
        rect=tuple(rect), label=label, label_reason=reason, glyph_hit=glyph_hit,
        max_glyph_score=max(scores) if scores else None, dwell_exact=dwell_exact,
        mute_screens_complete=mute_clean, classifier_verdict=verdict, centered=centered,
        center_offset_frac=float(offset), dwell_span_s=float(dwell_span_s),
        crops=tuple(crops), mute_records=tuple(mute_records),
        reattach_probe_ran=reattach["probe_ran"],
        reattach_dwell_exact=reattach["dwell_exact"],
        reattach_mute_screens_complete=reattach["mute_screens_complete"],
        reattach_centered=reattach["centered"],
        reattach_center_offset_frac=reattach["center_offset_frac"],
        reattach_dwell_span_s=(None if reattach_span_s is None else float(reattach_span_s)),
        reattach_rect=reattach["rect"], reattach_crops=tuple(reattach["crops"]),
        reattach_mute_records=tuple(reattach["mute_records"]))


def _observe_reattach(rect, sequence, *, matcher, frame_height: int, content_band,
                      page_shift_px: int | None) -> dict:
    """The second burst's legs for one card, or an all-missing record when there was no probe.

    A rect the probe pushed off the screen is a MISSING observation rather than a failed one:
    `probe_ran` stays False and every leg stays None, so the card lands on UNSURE and can never
    be read as motion. Two frames is the minimum -- one frame has zero consecutive pairs, and a
    vacuous "all pairs matched" is the unearned pass this whole design exists to prevent.
    """
    absent = {"probe_ran": False, "dwell_exact": None, "mute_screens_complete": None,
              "centered": None, "center_offset_frac": None, "rect": None, "crops": (),
              "mute_records": ()}
    if not sequence or len(sequence) < 2 or page_shift_px is None:
        return absent
    shifted = (rect[0], rect[1] - page_shift_px, rect[2], rect[3] - page_shift_px)
    if shifted[1] < 0 or shifted[3] > frame_height:
        return absent
    records = [_screen_card(frame, shifted, matcher) for frame in sequence]
    offset = card_center_offset_frac(tuple(shifted), frame_height=frame_height,
                                     content_band=content_band)
    return {"probe_ran": True,
            "dwell_exact": bool(dwell_exact_over_rect(sequence, shifted)),
            "mute_screens_complete": all(record["clean"] for record in records),
            "centered": abs(offset) <= CENTER_BAND_FRAC,
            "center_offset_frac": float(offset),
            "rect": tuple(shifted),
            "crops": tuple(_crop_png(frame, shifted) for frame in sequence),
            "mute_records": tuple(records)}


# =====================================================================================
# Driving the phone (guarded, humanized primitives only)
# =====================================================================================

def _guarded_screencap(driver, *, on_blank: str = "none") -> bytes:
    """One read-only frame through the driver's own blank-frame guard.

    A blank frame is not a shorter dwell to shrug at, it is a screen nobody recognizes, so it
    halts rather than silently shortening an exact-run measurement.
    """
    frame = driver._screencap(on_blank=on_blank)
    if frame is None:
        raise CampaignHalt("the phone returned a blank frame, so the screen is not recognizable")
    return frame


def reposition_for_advance(driver, *, rnd: random.Random, sleep_fn,
                           print_fn=print) -> tuple[int, bool]:
    """Put the page back where the deck is confirmable. Returns `(gestures, confirmed)`.

    NOTHING HERE WEAKENS THE CONFIRMATION.  `_observe_deck_ready` is the very predicate
    `_require_deck_confirmed` uses, asked passively, and `_scroll_to_top` is the driver's own
    guarded humanized rewind -- the same one this campaign already runs between profiles.  The
    caller attempts the advance whatever this returns, so a page that still will not confirm is
    refused by the driver itself, with the driver's own error, and the halt reads exactly as it
    did in the field.  This function only removes the case where the refusal was caused by the
    walk's own scroll position rather than by anything actually wrong with the screen.

    Bounded on purpose: if a couple of rewinds have not produced a confirmable deck, the screen
    is genuinely something else (a paywall, an ad, a dialog) and more gestures would be spent
    guessing at it.
    """
    frame = _guarded_screencap(driver)
    if driver._observe_deck_ready(frame):
        return 0, True
    budget = int(round(bound._hazard_value(*_REPOSITION_ATTEMPTS_SPAN, rnd)))
    budget = max(_REPOSITION_ATTEMPTS_SPAN[0], min(_REPOSITION_ATTEMPTS_SPAN[1], budget))
    gestures = 0
    for attempt in range(budget):
        # One guarded rewind per attempt. `_scroll_to_top` may issue several strokes internally,
        # all through the same transport and the same forbidden-zone guard as every other
        # gesture in this run; it is counted as one repositioning here because it is one
        # decision.
        driver._scroll_to_top()
        gestures += 1
        sleep_fn(human_delay(_READ_SCROLL_SETTLE_S))
        frame = _guarded_screencap(driver)
        if driver._observe_deck_ready(frame):
            return gestures, True
        print_fn(f"    deck still not confirmable after rewind {attempt + 1}/{budget}")
    return gestures, False


def _halt_if_screen_unrecognized(driver, *, ordinal: int) -> None:
    """The driver's own error surfaces, asked BEFORE anything is read or spent."""
    blocked = driver.blocked_reason()
    if blocked:
        raise CampaignHalt(f"profile {ordinal}: {blocked}")
    top_reason = driver._confirm_enumeration_top()
    if top_reason:
        raise CampaignHalt(
            f"profile {ordinal}: the ordinary deck top could not be confirmed, so nothing on "
            f"screen is the card this campaign thinks it is reading: {top_reason}")


@dataclass
class Station:
    """One read position: the anchor frame, its card rects, and the burst held on it."""

    position: int
    anchor: bytes
    # The anchor's OWN stamp, on the same profile clock the burst below is stamped on. A card's
    # frame list splices anchor and burst into one series, so an anchor carrying a constant
    # instead of its measurement would put the two on different timelines and inflate the card's
    # measured dwell span by the whole time since the profile started -- which grows with the
    # station index and is the number `measure` derives the licensed dwell window from.
    anchor_t: float
    rects: tuple[tuple[int, int, int, int], ...]
    burst: list[tuple[bytes, float]]
    planned_frames: int
    planned_window_s: float
    span_s: float
    centering_steps: int
    offsets_before: tuple[float, ...]
    # The re-attach probe: its own settled anchor, its own burst, and the page displacement it
    # MEASURED between the two. None/empty means the probe could not complete, which costs every
    # card at this station its `photo` eligibility and nothing else.
    reattach_anchor: bytes | None = None
    reattach_anchor_t: float | None = None
    reattach_burst: list[tuple[bytes, float]] | None = None
    reattach_span_s: float | None = None
    reattach_page_shift_px: int | None = None


def _offset_of(rect, *, frame_height: int, content_band) -> float:
    return card_center_offset_frac(tuple(rect), frame_height=frame_height,
                                   content_band=content_band)


def center_candidate(driver, rect, *, frame_height: int, content_band, rnd: random.Random,
                     sleep_fn) -> bool:
    """One humanized corrective scroll that moves `rect` toward the autoplay trigger zone.

    Returns whether a gesture was actually issued.  The distance is derived from the measured
    offset and the driver's own documented scroll geometry (a forward stroke travels `h * frac`,
    touch-down at `h*(0.5 + frac/2)`, release at `h*(0.5 - frac/2)`), then clamped into the read
    scroll envelope every other gesture in this repository lives inside.  An offset too small to
    express as a legal gesture returns False rather than reaching for a smaller, unhumanized one:
    the standing rule is best humanized interaction or fail loudly, and "cannot centre this card"
    is a perfectly good answer that costs the card its photo eligibility and nothing else.

    The LANE comes from the driver's behaviour policy, like every other read-scroll, so a
    centring stroke is not the one gesture in the run that always travels the same column.
    """
    offset = _offset_of(rect, frame_height=frame_height, content_band=content_band)
    band_height = (float(content_band[1]) - float(content_band[0])) * frame_height
    frac = abs(offset) * band_height / float(frame_height)
    if frac < _hinge._READ_SCROLL_FRAC_MIN:
        return False
    frac = min(frac, _hinge._READ_SCROLL_FRAC_MAX)
    _dwell, _step, x_frac = driver._sample_read_step(0, None)
    if offset > 0:
        driver._scroll_down_one(frac, x_frac)      # the card sits low: bring the page up
    else:
        driver._scroll_up_one(frac, x_frac)        # the card sits high: bring the page down
    sleep_fn(human_delay(_READ_SCROLL_SETTLE_S))
    return True


def reattach_stimulus(driver, anchor: bytes, rect, *, frame_height: int, content_band,
                      rnd: random.Random, sleep_fn, clock,
                      origin: float) -> tuple[bytes, int, float] | None:
    """Take `rect`'s card OUT of the autoplay band and bring it back.  `(frame, shift, frame_t)`.

    `frame_t` is the returned frame's own stamp on the caller's profile clock, taken the instant
    the read is issued, exactly as `record_spanning_burst` stamps a burst frame.  It is returned
    from in here rather than read off the clock afterwards because the measured page shift
    between two frames is estimator work, not a screencap, and charging that to the dwell would
    understate the span of the second look by however long the estimator took.

    THE RESIDUAL IT EXISTS FOR.  A centred byte-exact burst proves Hinge was asked to play the
    card and that nothing moved; it cannot prove the media answered.  Stalled, buffering,
    unattached and ended-non-looping video all emit nothing for as long as anybody watches.
    Re-entering the trigger zone is what makes Hinge attach and start the media, so the burst
    taken after this round trip is a deterministic second chance for such a video to reveal
    itself.  There is no statistic here: the campaign simply looks twice.

    EVERY LEG IS MEASURED.  `_measured_page_shift` says where the page actually went after each
    stroke, and the rect travels with that measurement rather than with the distance we asked
    for, so the second burst looks at the card the first one looked at.  The return leg is
    `center_candidate`, the same corrective read-scroll the station used to seat the card in the
    first place; the exit is one guarded reverse read-scroll sized from the measured offset.
    Any leg that cannot be measured returns None, which leaves the station without a probe and
    every card at it UNSURE.
    """
    band_px = (float(content_band[1]) - float(content_band[0])) * frame_height
    if band_px <= 0:
        return None
    offset = _offset_of(rect, frame_height=frame_height, content_band=content_band)
    target = CENTER_BAND_FRAC * bound._hazard_value(*_REATTACH_EXIT_BAND_MULTIPLE_SPAN, rnd)
    exit_px = min((target - offset) * band_px, _REATTACH_MAX_EXIT_BAND_FRAC * band_px)
    frac = min(_hinge._READ_SCROLL_FRAC_MAX,
               max(_hinge._READ_SCROLL_FRAC_MIN,
                   frac_for_step_px(int(round(exit_px)), frame_height)))
    _dwell, _step, x_frac = driver._sample_read_step(0, None)
    driver._scroll_up_one(frac, x_frac)      # content down: the card slides off centre
    sleep_fn(human_delay(_READ_SCROLL_SETTLE_S))
    frame_t = clock() - origin
    frame = _guarded_screencap(driver)
    total = driver._measured_page_shift(anchor, frame)
    if total is None:
        return None
    moved = (rect[0], rect[1] - total, rect[2], rect[3] - total)
    if abs(_offset_of(moved, frame_height=frame_height,
                      content_band=content_band)) <= CENTER_BAND_FRAC:
        # The stroke was delivered and the card never left the zone (a clamp, a swallowed fling).
        # Nothing detached, so nothing can re-attach, and calling the re-entry a re-attach anyway
        # would manufacture the very observation this probe exists to earn.
        return None
    budget = int(round(bound._hazard_value(*_CENTERING_STEPS_SPAN, rnd)))
    budget = max(_CENTERING_STEPS_SPAN[0], min(_CENTERING_STEPS_SPAN[1], budget))
    for _attempt in range(budget):
        moved = (rect[0], rect[1] - total, rect[2], rect[3] - total)
        if abs(_offset_of(moved, frame_height=frame_height,
                          content_band=content_band)) <= CENTER_BAND_FRAC:
            break
        if not center_candidate(driver, moved, frame_height=frame_height,
                                content_band=content_band, rnd=rnd, sleep_fn=sleep_fn):
            break
        following_t = clock() - origin
        following = _guarded_screencap(driver)
        step = driver._measured_page_shift(frame, following)
        if step is None:
            return None
        total += step
        frame, frame_t = following, following_t
    return frame, int(total), frame_t


def read_stations(driver, *, rnd: random.Random, sleep_fn, clock, origin: float,
                  content_band, segment=None, print_fn=print) -> tuple[list[Station], dict]:
    """Walk one profile, CENTRING a candidate card before each burst.

    This is the frame source, and it deliberately does NOT go through `_capture_current`.  That
    method's enumeration hook is gated on `_item_enumeration_blocker`, a product-policy gate that
    is false whenever `apps.hinge.targeting_calibration` is unconfigured -- which is what made the
    first version of this tool persist nothing at all on the shipped config.  See the module
    docstring.  Everything here is guarded driver transport: `_screencap` reads,
    `_sample_read_step` draws the dwell, distance and lane from the driver's own behaviour
    policy, and `_scroll_down_one`/`_scroll_up_one` are the humanized read-scrolls that keep the
    scroll ledger honest.

    WHY IT CENTRES AT ALL.  Hinge autoplays a video only at or near the centre of the screen, so
    a burst taken with the card off to one side measures a video that was never asked to play and
    would call it a photograph.  So each station picks the candidate nearest the centre, spends a
    hazard-drawn budget of corrective scrolls trying to seat it, and then bursts wherever it
    actually ended up -- the achieved offset is MEASURED per card afterwards, never assumed, and
    a card that could not be seated simply stays unsure.  This costs profile wall-clock time
    compared with a free-running walk, which is accepted.

    WHY EVERY STATION BURSTS TWICE.  Centring only proves the media was ASKED to play.  A
    station therefore follows its first burst with `reattach_stimulus` -- out of the autoplay
    band and back, the event Hinge re-attaches media on -- and bursts again, and `label_card`
    calls a card `photo` only when both bursts agree.  A station whose probe could not complete
    keeps its first burst and its cards stay unsure; nothing is discarded and nothing is assumed.

    The bottom of the profile is a frame byte-identical to the previous one: the page stopped
    moving under a real scroll, which is the same signal `_capture_current`'s own loop stops on.
    """
    segment = segment or (lambda frame: selectable_card_rects(
        frame, content_band=driver.content_band, like_template=driver._template("like"),
        like_threshold=_hinge._LIKE_MATCH_THRESHOLD))
    planned = int(round(bound._hazard_value(*_STATIONS_SPAN, rnd)))
    planned = max(_STATIONS_SPAN[0], min(_STATIONS_SPAN[1], planned))
    stations: list[Station] = []
    notes = {"planned_stations": planned, "stations_read": 0, "reached_bottom": False,
             "segmentation_failures": 0, "stations_without_a_complete_card": 0,
             "centering_steps": 0, "centering_gave_up": 0,
             "center_band_frac": CENTER_BAND_FRAC,
             "reattach_probes_ran": 0, "reattach_probes_refused": 0}
    # Every frame this walk hands on carries the stamp of the read that produced it, taken the
    # instant the read is issued -- the same convention `record_spanning_burst` uses -- because
    # whichever frame is current when a burst starts becomes that station's anchor, and the
    # anchor sits in the SAME series as the burst frames when a card is written.
    frame_t = clock() - origin
    frame = _guarded_screencap(driver)
    frame_height = bound._frame_size(frame)[1]

    def _segment(payload):
        try:
            return tuple(segment(payload))
        except SegmentationError as exc:
            # A frame the segmenter refuses is not a card-free frame; it is a frame nobody can
            # measure. Counted and skipped rather than treated as "no cards here", so the empty
            # profile halt cannot be satisfied by a page of unreadable frames.
            notes["segmentation_failures"] += 1
            print_fn(f"    segmentation refused ({type(exc).__name__}: {exc})")
            return ()

    for position in range(planned):
        notes["stations_read"] += 1
        rects = _segment(frame)
        offsets_before = tuple(
            round(_offset_of(rect, frame_height=frame_height, content_band=content_band), 6)
            for rect in rects)
        steps = 0
        if rects:
            budget = int(round(bound._hazard_value(*_CENTERING_STEPS_SPAN, rnd)))
            budget = max(_CENTERING_STEPS_SPAN[0], min(_CENTERING_STEPS_SPAN[1], budget))
            for _attempt in range(budget):
                target = min(rects, key=lambda r: abs(
                    _offset_of(r, frame_height=frame_height, content_band=content_band)))
                if abs(_offset_of(target, frame_height=frame_height,
                                  content_band=content_band)) <= CENTER_BAND_FRAC:
                    break
                if not center_candidate(driver, target, frame_height=frame_height,
                                        content_band=content_band, rnd=rnd, sleep_fn=sleep_fn):
                    notes["centering_gave_up"] += 1
                    break
                steps += 1
                notes["centering_steps"] += 1
                frame_t = clock() - origin
                frame = _guarded_screencap(driver)
                # Re-acquire by proximity rather than identity: the campaign does not need to
                # know WHICH card it seated, only which cards were seated when the burst ran,
                # and that is measured per card from this anchor.
                rects = _segment(frame)
                if not rects:
                    break
        if rects:
            plan = bound.plan_burst(rnd)
            burst_payloads = bound.record_spanning_burst(
                lambda: _guarded_screencap(driver), plan, sleep_fn=sleep_fn, clock=clock)
            shortfall = bound.burst_span_shortfall(burst_payloads, plan)
            if shortfall > bound.BURST_SPAN_TOLERANCE_S:
                # A burst that did not span its window is not a dwell.  Campaign 2 recorded nine
                # frames across ONE second of a playing video's own countdown; measure would then
                # cap the licensed window at that one second and reject the run anyway, after
                # spending every profile.  Halt while it is still one profile.
                raise CampaignHalt(
                    f"a dwell burst spanned {plan.window_s - shortfall:.2f}s of its drawn "
                    f"{plan.window_s:.2f}s window (short by {shortfall:.2f}s, tolerance "
                    f"{bound.BURST_SPAN_TOLERANCE_S:.2f}s); a burst that does not span its "
                    "window cannot bound the exact-run tail a production dwell will meet")
            # The second look. Sized, gestured and measured exactly like the first burst, so a
            # corpus captured now stays comparable with production's own two-burst rule.
            probe_anchor = probe_burst = None
            probe_span = probe_shift = probe_anchor_t = None
            # `station_anchor` is the frame the rects were segmented from and stays the station's
            # anchor whatever the probe does; `frame` goes on to be the current screen, which is
            # what the read-scroll to the next station has to continue from.  Its stamp travels
            # with it: the anchor is the first frame of this station's card series.
            station_anchor, station_anchor_t = frame, frame_t
            probe_target = min(rects, key=lambda r: abs(
                _offset_of(r, frame_height=frame_height, content_band=content_band)))
            probe = reattach_stimulus(driver, station_anchor, probe_target,
                                      frame_height=frame_height, content_band=content_band,
                                      rnd=rnd, sleep_fn=sleep_fn, clock=clock, origin=origin)
            if probe is None:
                notes["reattach_probes_refused"] += 1
            else:
                probe_anchor, probe_shift, probe_anchor_t = probe
                notes["reattach_probes_ran"] += 1
                probe_plan = bound.plan_burst(rnd)
                probe_payloads = bound.record_spanning_burst(
                    lambda: _guarded_screencap(driver), probe_plan, sleep_fn=sleep_fn,
                    clock=clock)
                probe_shortfall = bound.burst_span_shortfall(probe_payloads, probe_plan)
                if probe_shortfall > bound.BURST_SPAN_TOLERANCE_S:
                    raise CampaignHalt(
                        f"a re-attach dwell burst spanned "
                        f"{probe_plan.window_s - probe_shortfall:.2f}s of its drawn "
                        f"{probe_plan.window_s:.2f}s window (short by {probe_shortfall:.2f}s, "
                        f"tolerance {bound.BURST_SPAN_TOLERANCE_S:.2f}s); a burst that does not "
                        "span its window cannot bound the exact-run tail a production dwell "
                        "will meet")
                probe_burst = [(payload, stamp - origin) for payload, stamp in probe_payloads]
                probe_span = float(probe_plan.window_s - probe_shortfall)
                frame, frame_t = probe_payloads[-1][0], probe_payloads[-1][1] - origin
            stations.append(Station(
                position=position, anchor=station_anchor, anchor_t=station_anchor_t, rects=rects,
                burst=[(payload, stamp - origin) for payload, stamp in burst_payloads],
                planned_frames=plan.frames, planned_window_s=float(plan.window_s),
                span_s=float(plan.window_s - shortfall), centering_steps=steps,
                offsets_before=offsets_before,
                reattach_anchor=probe_anchor, reattach_anchor_t=probe_anchor_t,
                reattach_burst=probe_burst,
                reattach_span_s=probe_span, reattach_page_shift_px=probe_shift))
        else:
            notes["stations_without_a_complete_card"] += 1
        if position < planned - 1:
            dwell, frac, x_frac = driver._sample_read_step(position, None)
            sleep_fn(dwell)
            driver._scroll_down_one(frac, x_frac)
            # Let Hinge finish the scroll animation before the next frame is segmented; a frame
            # caught mid-animation segments into partial blocks and would waste the station.
            sleep_fn(human_delay(_READ_SCROLL_SETTLE_S))
            following_t = clock() - origin
            following = _guarded_screencap(driver)
            if following == frame:
                notes["reached_bottom"] = True
                break
            frame, frame_t = following, following_t
    return stations, notes


# =====================================================================================
# Persistence, and the licence that gates the advance
# =====================================================================================

@dataclass(frozen=True)
class ProfileHarvest:
    """Proof that one profile actually produced evidence. The ONLY key to the advance gesture.

    Constructed exclusively by `persist_profile`, from the records it wrote, and re-verified
    against the bytes on disk before a gesture is allowed.  This is the structural answer to the
    2026-08-21 live failure: `advance_deck` takes a harvest and calls
    `require_advance_licence()` before it can reach `dislike()`/`like()`, so there is no code
    path on which a real Pass or a real Like can be spent on a profile that persisted nothing.
    """

    profile_id: str
    root: Path
    frame_records: tuple[dict, ...]
    measurable_cards: int

    def require_advance_licence(self) -> None:
        if not self.frame_records:
            raise EmptyProfileHalt(
                f"{self.profile_id} persisted ZERO frames, so there is nothing to advance past "
                "and nothing was measured; the perception path is not seeing this screen")
        if self.measurable_cards < 1:
            raise EmptyProfileHalt(
                f"{self.profile_id} persisted frames but produced no measurable card; refusing "
                "to spend a real gesture on a profile that yielded no evidence")
        for record in self.frame_records:
            path = self.root / record["path"]
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise EmptyProfileHalt(
                    f"{self.profile_id} recorded {record['path']} but it is not on disk "
                    f"({type(exc).__name__})") from exc
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise EmptyProfileHalt(
                    f"{self.profile_id} frame {record['path']} does not match the digest it was "
                    "recorded with, so the persisted corpus is not what was measured")


def _write_frames(directory: Path, frames: list[tuple[bytes, float]], *,
                  relative: str) -> list[dict]:
    """Persist frames and return their manifest records, digest and monotonic stamp each."""
    ensure_private_dir(directory)
    records: list[dict] = []
    for position, (payload, stamp) in enumerate(frames):
        name = f"frame_{position:03d}.png"
        atomic_write_private_bytes(directory / name, payload, parent=directory)
        records.append({"path": f"{relative}/{name}",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "t": round(float(stamp), 6)})
    return records


def advance_deck(driver, advance: str, *, harvest: ProfileHarvest) -> None:
    """Spend the run's chosen advance, and only against a profile that produced evidence.

    The licence check is the FIRST statement and it raises rather than returning a verdict, so
    the gesture below is unreachable without it.  `dislike()` and `like()` are the PUBLIC driver
    methods, which keeps their own deck confirmation, forbidden-zone guard and landed
    verification in force.  Neither touches a paid control and neither is a super-like: standing
    owner rule, the bot does ordinary like/pass only.  A driver refusal is a halt, never a retry
    by a cruder route: the humanized-input rule is best interaction or fail loudly.
    """
    harvest.require_advance_licence()
    if advance not in ADVANCE_ACTIONS:
        raise AutoBoundRefused(f"unknown advance action {advance!r}")
    try:
        if advance == ADVANCE_PASS:
            driver.dislike()
        else:
            driver.like()
    except HingeActionError as exc:
        raise CampaignHalt(
            f"the driver refused the {advance} advance and left the screen untouched: "
            f"{type(exc).__name__}: {exc}") from exc


# =====================================================================================
# One profile
# =====================================================================================

def _run_one_profile(*, driver, ordinal: int, out_dir: Path, next_card_no: int, advance: str,
                     rnd: random.Random, sleep_fn, clock, classify, print_fn,
                     segment=None) -> dict:
    """Read, dwell on, label and persist one profile. Performs NO gesture that advances the deck."""
    _halt_if_screen_unrecognized(driver, ordinal=ordinal)
    origin = clock()
    content_band = driver.content_band
    stations, notes = read_stations(driver, rnd=rnd, sleep_fn=sleep_fn, clock=clock,
                                    origin=origin, content_band=content_band, segment=segment,
                                    print_fn=print_fn)

    profile_id = f"profile_{ordinal:04d}"
    profile_dir = out_dir / "profiles" / profile_id
    cards: list[dict] = []
    frame_records: list[dict] = []
    station_records: list[dict] = []
    card_no = next_card_no
    seen_crop_digests: set[str] = set()
    duplicates = 0

    for station in stations:
        tag = f"station_{station.position:02d}"
        anchor_records = _write_frames(profile_dir / tag / "anchor",
                                       [(station.anchor, station.anchor_t)],
                                       relative=f"profiles/{profile_id}/{tag}/anchor")
        burst_records = _write_frames(profile_dir / tag / "burst", station.burst,
                                      relative=f"profiles/{profile_id}/{tag}/burst")
        frame_records.extend(anchor_records + burst_records)
        sequence = [station.anchor] + [payload for payload, _t in station.burst]
        # Read back off the records rather than re-rounded here, so the frame manifest and the
        # card manifest cannot disagree about when a frame was read.  Both are the ONE profile
        # clock: the anchor is measured like every burst frame, never stamped with a constant.
        times = [anchor_records[0]["t"]] + [record["t"] for record in burst_records]
        span = (times[-1] - times[0]) if len(times) > 1 else 0.0
        # The re-attach burst is persisted exactly like the first, under its own labels, so a
        # future reader can tell which look a verdict came from without diffing timestamps.
        reattach_sequence: list[bytes] = []
        reattach_times: list[float] = []
        if station.reattach_anchor is not None and station.reattach_burst:
            reattach_anchor_records = _write_frames(
                profile_dir / tag / "reattach_anchor",
                [(station.reattach_anchor, station.reattach_anchor_t)],
                relative=f"profiles/{profile_id}/{tag}/reattach_anchor")
            reattach_burst_records = _write_frames(
                profile_dir / tag / "reattach_burst", station.reattach_burst,
                relative=f"profiles/{profile_id}/{tag}/reattach_burst")
            frame_records.extend(reattach_anchor_records + reattach_burst_records)
            reattach_sequence = ([station.reattach_anchor]
                                 + [payload for payload, _t in station.reattach_burst])
            reattach_times = ([reattach_anchor_records[0]["t"]]
                              + [record["t"] for record in reattach_burst_records])
        else:
            reattach_anchor_records = []
            reattach_burst_records = []
        frame_height = bound._frame_size(station.anchor)[1]
        station_cards: list[str] = []
        for rect in station.rects:
            observation = observe_card(rect, sequence, dwell_span_s=span,
                                       matcher=driver._match_video_mute,
                                       frame_height=frame_height, content_band=content_band,
                                       classify=classify,
                                       reattach_sequence=reattach_sequence or None,
                                       reattach_span_s=station.reattach_span_s,
                                       reattach_page_shift_px=station.reattach_page_shift_px)
            digest = hashlib.sha256(observation.crops[0]).hexdigest()
            if digest in seen_crop_digests:
                # The same physical card seen from two stations would be counted twice in a
                # denominator whose whole claim is per-CARD independence. Byte-identical anchor
                # crops are the cheap, honest test for that; near-duplicates that differ by a
                # pixel are left in and reported rather than guessed at with a threshold.
                duplicates += 1
                continue
            seen_crop_digests.add(digest)
            card_id = f"card_{card_no:04d}"
            card_no += 1
            crop_records = _write_frames(
                out_dir / "cards" / card_id, list(zip(observation.crops, times, strict=True)),
                relative=f"cards/{card_id}")
            frame_records.extend(crop_records)
            # Kept in their OWN key rather than appended to `frames`: the sibling's `measure`
            # computes the byte-exact run from `frames`, and splicing a second burst into that
            # list would silently redefine the statistic the whole campaign reports.
            reattach_crop_records = _write_frames(
                out_dir / "cards" / card_id / "reattach",
                list(zip(observation.reattach_crops, reattach_times, strict=True)),
                relative=f"cards/{card_id}/reattach") if observation.reattach_crops else []
            frame_records.extend(reattach_crop_records)
            cards.append({
                "card_id": card_id, "profile_id": profile_id, "station": station.position,
                "card_rect": list(observation.rect), "label": observation.label,
                # The autoplay precondition, per card per burst, measured not assumed.
                "centered": observation.centered,
                "center_offset_frac": round(observation.center_offset_frac, 6),
                "center_band_frac": CENTER_BAND_FRAC,
                # Deliberately never settled before the burst: a settle waits until two
                # consecutive frames are identical, which would start every burst at a proven
                # moment of stillness and bias the exactness statistic towards `photo` on
                # exactly the stalling videos the bound is about.
                "settled": None, "settle_reads": 0,
                "settle_skipped_reason": "a settle would start every burst at a moment of proven "
                                         "stillness and bias the exact-run statistic",
                "planned_frames": station.planned_frames,
                "planned_window_s": round(station.planned_window_s, 6),
                "frames": crop_records,
                "crop_sources": ([{"path": anchor_records[0]["path"], "role": "anchor"}]
                                 + [{"path": record["path"], "role": "burst"}
                                    for record in burst_records]),
                "burst_completed_t": round(times[-1], 6),
                "label_prompted_t": round(clock() - origin, 6),
                "reattach_probe_ran": observation.reattach_probe_ran,
                "reattach_rect": (list(observation.reattach_rect)
                                  if observation.reattach_rect else None),
                "reattach_page_shift_px": station.reattach_page_shift_px,
                "reattach_span_s": observation.reattach_dwell_span_s,
                "reattach_frames": reattach_crop_records,
                "reattach_crop_sources": (
                    ([{"path": reattach_anchor_records[0]["path"], "role": "reattach_anchor"}]
                     + [{"path": record["path"], "role": "reattach_burst"}
                        for record in reattach_burst_records])
                    if reattach_anchor_records else []),
                "mute_matcher_observations": [
                    {"screened": record["screened"], "score": record["score"],
                     "observational_only": True} for record in observation.mute_records],
                # `label_evidence` is the measurement record the label was derived FROM. Each
                # per-frame entry is marked observational, exactly as the owner-labeled sibling
                # marks the same matcher's output, and `circular` states plainly that here those
                # observations are ALSO the label, which is what makes the bound vacuous.
                "label_evidence": {
                    "circular": True, "channel": CIRCULAR_CHANNEL,
                    "reason": observation.label_reason, "glyph_hit": observation.glyph_hit,
                    "max_glyph_score": observation.max_glyph_score,
                    "dwell_exact": observation.dwell_exact,
                    "mute_screens_complete": observation.mute_screens_complete,
                    "classifier_verdict": observation.classifier_verdict,
                    "centered": observation.centered,
                    "center_offset_frac": round(observation.center_offset_frac, 6),
                    "dwell_span_s": observation.dwell_span_s,
                    "burst_screens": [dict(record) for record in observation.mute_records],
                    "reattach_probe_ran": observation.reattach_probe_ran,
                    "reattach_dwell_exact": observation.reattach_dwell_exact,
                    "reattach_mute_screens_complete": observation.reattach_mute_screens_complete,
                    "reattach_centered": observation.reattach_centered,
                    "reattach_center_offset_frac": (
                        None if observation.reattach_center_offset_frac is None
                        else round(observation.reattach_center_offset_frac, 6)),
                    "reattach_dwell_span_s": observation.reattach_dwell_span_s,
                    "reattach_burst_screens": [dict(record)
                                               for record in observation.reattach_mute_records],
                    "observational_only": True,
                },
            })
            station_cards.append(card_id)
        station_records.append({
            "station": station.position, "anchor_frame": anchor_records[0],
            "burst_frames": burst_records, "card_rects": [list(r) for r in station.rects],
            "cards": station_cards, "planned_frames": station.planned_frames,
            "planned_window_s": round(station.planned_window_s, 6),
            "burst_span_s": round(station.span_s, 6),
            "centering_steps": station.centering_steps,
            "offsets_before_centering": list(station.offsets_before),
            "reattach_anchor_frame": (reattach_anchor_records[0]
                                      if reattach_anchor_records else None),
            "reattach_burst_frames": reattach_burst_records,
            "reattach_span_s": (None if station.reattach_span_s is None
                                else round(station.reattach_span_s, 6)),
            "reattach_page_shift_px": station.reattach_page_shift_px})

    harvest = ProfileHarvest(profile_id=profile_id, root=out_dir,
                             frame_records=tuple(frame_records), measurable_cards=len(cards))
    print_fn(f"  profile {ordinal}: {len(station_records)} station(s), {len(cards)} card(s) "
             f"[{', '.join(card['label'] for card in cards) or 'none'}], "
             f"{len(frame_records)} frame(s) persisted, "
             f"{sum(1 for card in cards if card['centered'])} centred")
    return {
        "profile": {
            "profile_id": profile_id, "ordinal": ordinal, "stations": station_records,
            "persisted_frames": len(frame_records), "cards": len(cards),
            "duplicate_card_crops_skipped": duplicates,
            "advance": advance, "advanced": False,
            # Filled in by the campaign loop once the advance step runs; a profile that halted
            # before it keeps the zero/None it was born with, which is the honest record.
            "repositioning_gestures": 0, "deck_confirmed_before_advance": None,
            **notes,
        },
        "cards": cards, "harvest": harvest, "next_card_no": card_no,
        "frame_size_px": (list(bound._frame_size(stations[0].anchor)) if stations else None),
    }


# =====================================================================================
# Signals
# =====================================================================================

def _raise_on_signal(signum, _frame):
    """Turn SIGINT/SIGTERM into an ordinary halt so the manifest is still written.

    The 2026-08-21 run was `pkill`ed and lost everything it had captured, because the default
    SIGTERM disposition kills the process outright and no `finally` ever runs.  Raising inside
    the handler hands control to the campaign loop's own except/finally, which records the reason
    and writes the INCOMPLETE manifest exactly as any other halt does.
    """
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    raise CampaignSignalled(
        f"received {name}; stopping without another gesture and writing the incomplete manifest")


@contextlib.contextmanager
def halt_on_signals(*, signal_module=signal, print_fn=print):
    """Install the halting handlers for the run and restore whatever was there before."""
    installed: dict = {}
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal_module, name, None)
        if signum is None:
            continue
        try:
            installed[signum] = signal_module.signal(signum, _raise_on_signal)
        except (ValueError, OSError, RuntimeError):
            # Only the main thread may install handlers.  Say so rather than let a caller
            # believe an interrupted run will still write its manifest.
            print_fn(f"note: {name} could not be trapped here, so a signal will not write the "
                     "incomplete manifest")
    try:
        yield tuple(installed)
    finally:
        for signum, previous in installed.items():
            try:
                signal_module.signal(signum, previous)
            except (ValueError, OSError, RuntimeError, TypeError):
                pass


# =====================================================================================
# The campaign loop
# =====================================================================================

def run_capture(*, out_dir: Path, driver, advance: str, serial: str,
                config_sha256: str | None, target_videos: int, target_photos: int,
                max_profiles: int | None = None, hinge_version_name: str | None = None,
                driver_content_band: tuple[float, float] | None = None,
                print_fn=print, sleep_fn=time.sleep, clock=time.monotonic,
                rnd: random.Random | None = None, classify=None, segment=None,
                signal_module=signal) -> dict:
    """Drive the deck until the targets are met, the profile budget runs out, or a halt."""
    rnd = rnd or random.Random()
    ensure_private_dir(out_dir / "cards")
    ensure_private_dir(out_dir / "profiles")
    cards: list[dict] = []
    profiles: list[dict] = []
    counts = {label: 0 for label in LABELS}
    spent = {ADVANCE_PASS: 0, ADVANCE_LIKE: 0}
    frame_size: list[int] | None = None
    next_card_no = 1
    ended = "targets_reached"
    halt_reason: str | None = None
    aborted: BaseException | None = None

    print_fn(
        f"\nCIRCULAR AI LABELS ACCEPTED. Every card is labeled by the mute matcher, the burst's "
        f"own pixel-\nexactness and the crop classifier, which are the same signals the accept "
        f"rule reads. The video\nfalse-accept count this produces is ZERO BY CONSTRUCTION and is "
        f"not evidence.\nAdvance action for this run: {advance.upper()} (one real {advance} per "
        f"profile, never a paid control).\nA profile that persists no frame or yields no card "
        f"HALTS the run before any advance.\nFrames are LOCAL-ONLY under gitignored "
        f"ops/calibration/: these are real people's profiles.\n")

    try:
        with halt_on_signals(signal_module=signal_module, print_fn=print_fn):
            # Once per session, through the driver's own guarded rewind -- exactly what
            # `next_profile` does before its first read.  This tool never calls
            # `_capture_current`, so nothing else restores the invariant that a read starts at a
            # confirmed scroll top.
            if not driver._ensure_session_top():
                raise CampaignHalt(
                    "the card could not be confirmed at its scroll top at session start, so the "
                    "run would begin part way down somebody's profile")
            while True:
                if counts[LABEL_VIDEO] >= target_videos and counts[LABEL_PHOTO] >= target_photos:
                    ended = "targets_reached"
                    break
                if max_profiles is not None and len(profiles) >= max_profiles:
                    ended = "max_profiles"
                    break
                result = _run_one_profile(
                    driver=driver, ordinal=len(profiles) + 1, out_dir=out_dir,
                    next_card_no=next_card_no, advance=advance, rnd=rnd, sleep_fn=sleep_fn,
                    clock=clock, classify=classify, print_fn=print_fn, segment=segment)
                profiles.append(result["profile"])
                cards.extend(result["cards"])
                next_card_no = result["next_card_no"]
                if frame_size is None:
                    frame_size = result["frame_size_px"]
                for card in result["cards"]:
                    counts[card["label"]] += 1
                # Licence FIRST, before a single repositioning gesture is spent: an empty
                # profile halts here and the page is left exactly where the walk left it.
                result["harvest"].require_advance_licence()
                gestures, confirmed = reposition_for_advance(
                    driver, rnd=rnd, sleep_fn=sleep_fn, print_fn=print_fn)
                result["profile"]["repositioning_gestures"] = gestures
                result["profile"]["deck_confirmed_before_advance"] = confirmed
                # Attempted regardless of `confirmed`: the driver's own confirmation is the only
                # thing allowed to decide, and when it refuses the halt is the field's exact
                # UnconfirmedScreenError rather than a second opinion invented here.
                advance_deck(driver, advance, harvest=result["harvest"])
                result["profile"]["advanced"] = True
                spent[advance] += 1
                # The read leaves the card scrolled down, and the next profile's first act is an
                # affirmative scroll-top confirmation. `current_profile` maintains that invariant
                # between profiles with this same guarded rewind; calling `_capture_current`
                # directly means nothing else will. It is cheap when the new card is already at
                # its top (one stroke, two frames) and its verdict is recorded rather than
                # trusted -- the next profile's own top gate is what actually decides.
                result["profile"]["rewound_to_top"] = bool(driver._scroll_to_top())
                # Between-profile pacing is hazard-drawn from the driver's own read dwell, never
                # a constant: a fixed cadence is a bot signature in its own right (auto-mode
                # owner rule, which forbids any fixed timing parameter anywhere in this repo).
                sleep_fn(human_cooldown(getattr(driver, "dwell_s", 1.0)))
    except CampaignHalt as exc:
        # Loud, and then nothing: no recovery gesture, no retry, the screen left exactly where
        # the driver left it so the failure can be looked at (standing auto-mode owner rule).
        ended = "signalled" if isinstance(exc, CampaignSignalled) else "halted"
        if isinstance(exc, EmptyProfileHalt):
            ended = "halted_empty_profile"
        halt_reason = str(exc)
        print_fn(f"\nHALTED: {halt_reason}\nThe screen was left untouched for debugging.")
    except Exception as exc:  # noqa: BLE001 -- record what was captured, then re-raise loudly
        ended = "halted"
        halt_reason = f"{type(exc).__name__}: {exc}"
        aborted = exc

    manifest = {
        "schema_version": bound._MANIFEST_SCHEMA_VERSION, "kind": bound._CAMPAIGN_KIND,
        "tool_version": _TOOL_VERSION, "campaign_mode": _CAMPAIGN_MODE,
        "ground_truth_channel": CIRCULAR_CHANNEL,
        "human_ground_truth": False,
        "accepted_circular_risk": CIRCULAR_ACCEPTANCE,
        "label_blind_spot": LABEL_BLIND_SPOT,
        # False, and stated: on this channel the matcher's output IS part of the label. The
        # sibling records True for the opposite reason and the two must never be confused.
        "mute_matcher_is_observational_not_ground_truth": False,
        "card_rect_source": "segment.segment_frame per anchor frame (never the item index)",
        # The zone the photo labels in this campaign are licensed for. Frozen into the artifact
        # by `build_artifact`, because the accept rule these labels support is only valid for
        # cards held inside the same zone.
        "autoplay_center_band_frac": CENTER_BAND_FRAC,
        "burst_schedule": "absolute deadlines across the drawn window; per-frame monotonic "
                          "stamps taken at screencap time, never file mtimes",
        # The hazard spans every burst in this campaign was drawn from. Two sittings can only be
        # merged into one denominator when they ran the same experiment, and this is what "the
        # same experiment" means for a dwell: see `_MERGE_AGREEMENT_FIELDS`.
        "dwell_parameter_space": {
            "burst_frames_span": list(bound._BURST_FRAMES_SPAN),
            "burst_window_s_span": list(bound._BURST_WINDOW_S_SPAN),
            "burst_span_tolerance_s": bound.BURST_SPAN_TOLERANCE_S,
            "stations_span": list(_STATIONS_SPAN),
            "centering_steps_span": list(_CENTERING_STEPS_SPAN),
            "center_band_frac": CENTER_BAND_FRAC,
            # A campaign that looked ONCE and a campaign that looked twice ran different
            # experiments, so these belong in the agreement set rather than beside it: merging
            # their denominators would average a rule against the evidence for a weaker one.
            "reattach_exit_band_multiple_span": list(_REATTACH_EXIT_BAND_MULTIPLE_SPAN),
            "reattach_max_exit_band_frac": _REATTACH_MAX_EXIT_BAND_FRAC,
        },
        "advance_action": advance,
        "like_confirmation": LIKE_CONFIRMATION if advance == ADVANCE_LIKE else None,
        "completed": ended in ("targets_reached", "max_profiles"),
        "ended": ended, "halt_reason": halt_reason,
        "device": serial, "hinge_version_name": hinge_version_name,
        "frame_size_px": frame_size,
        "content_band": [_CARD_CROP_BAND[0], _CARD_CROP_BAND[1]],
        "card_frames_are_card_rect_crops": True,
        "driver_content_band": (None if driver_content_band is None
                                else [driver_content_band[0], driver_content_band[1]]),
        "config_sha256": config_sha256,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "targets": {"videos": target_videos, "photos": target_photos,
                    "max_profiles": max_profiles},
        "counts": {"profiles_seen": len(profiles), "passes": spent[ADVANCE_PASS],
                   "likes": spent[ADVANCE_LIKE], "cards": len(cards), **counts},
        "profiles": profiles,
        "cards": cards,
    }
    atomic_write_private_text(out_dir / "manifest.json",
                              json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                              parent=out_dir)
    print_fn(f"\nwrote {out_dir / 'manifest.json'} "
             f"({'complete' if manifest['completed'] else 'INCOMPLETE'})")
    if aborted is not None:
        raise aborted
    return manifest


def print_run_summary(manifest: dict, *, print_fn=print) -> None:
    counts = manifest.get("counts") or {}
    print_fn(f"\nprofiles seen: {counts.get('profiles_seen', 0)}  "
             f"passes spent: {counts.get('passes', 0)}  likes spent: {counts.get('likes', 0)}")
    print_fn(f"cards: {counts.get('cards', 0)} "
             f"(video {counts.get(LABEL_VIDEO, 0)}, photo {counts.get(LABEL_PHOTO, 0)}, "
             f"written {counts.get(LABEL_WRITTEN, 0)}, unsure {counts.get(LABEL_UNSURE, 0)})")
    print_fn(f"ended: {manifest.get('ended')}"
             + (f" -- {manifest['halt_reason']}" if manifest.get("halt_reason") else ""))
    print_fn(f"\n{LABEL_BLIND_SPOT}")


# =====================================================================================
# measure / emit: the sibling's machinery, restricted to this channel
# =====================================================================================

def measure(campaigns, *, config_path: str | None = None) -> bound.BoundResult:
    """The sibling's own offline bound, refusing anything that is not a circular campaign.

    Every threshold, the per-card statistic and the window-capping rule come from
    `tools.hinge_video_bound.measure` unchanged; keeping a second copy of that arithmetic here is
    exactly how the shipped window and the bound that licenses it would drift apart.  All this
    adds is the restriction: this subcommand measures the circular channel and nothing else, so
    an owner-labeled campaign is never quietly reported under a caveat it does not deserve.

    An optional `adjudications.json` beside each manifest is applied by that same shared path,
    before any denominator is formed, and is scoped to the directory it sits in. See the module
    docstring for why its transitions are one-way and why adjudicating cannot undo the
    circularity.

    `campaigns` may be one directory or several. Merging is what makes the section 4 thresholds
    reachable at all: 60 video cards is roughly 300 profiles at the observed rate, which is more
    sittings than one uninterrupted campaign directory can hold.
    """
    result = bound.measure(campaigns, config_path=config_path)
    channel = result.manifest.get("ground_truth_channel")
    if channel != CIRCULAR_CHANNEL:
        raise AutoBoundRefused(
            f"{campaigns} was captured on the {channel!r} channel; measure it with "
            "`python -m tools.hinge_video_bound measure`, which is the harness that channel "
            "belongs to")
    return result


def emit(campaigns, *, config_path: str,
         serial_override: str | None = None) -> tuple[dict, dict]:
    """Freeze bound.json and the paste block for one or more circular campaigns."""
    measure(campaigns, config_path=config_path)
    artifact, paste = bound.emit(campaigns, config_path=config_path,
                                 serial_override=serial_override)
    if artifact.get("accepted_circular_risk") != CIRCULAR_ACCEPTANCE:
        raise AutoBoundRefused(
            "the frozen artifact does not carry the circular acceptance phrase; config "
            "validation would refuse it and a reader could mistake it for owner-labeled evidence")
    return artifact, paste


# =====================================================================================
# CLI
# =====================================================================================

def _refuse_unconfirmed(args) -> None:
    """Every authorization check, before a line of config is read or a device is touched."""
    if getattr(args, "confirmation", "") != CIRCULAR_ACCEPTANCE:
        raise AutoBoundRefused(
            f"capture requires the exact --confirmation {CIRCULAR_ACCEPTANCE!r}. It accepts a "
            "measurement whose video labels come from the same signals the accept rule reads, "
            "so the video false-accept count it produces is zero by construction. It is "
            "deliberately not enabled by a short flag.")
    if args.advance not in ADVANCE_ACTIONS:
        raise AutoBoundRefused(f"--advance must be one of {ADVANCE_ACTIONS}")
    if args.advance == ADVANCE_LIKE and getattr(args, "like_confirmation", "") != LIKE_CONFIRMATION:
        raise AutoBoundRefused(
            f"--advance like requires the exact --like-confirmation {LIKE_CONFIRMATION!r}, "
            "separate from --confirmation. It replaces this campaign's Pass with a REAL, "
            "PERMANENT Like on every profile it reads.")
    if args.target_videos < 1 or args.target_photos < 1:
        raise AutoBoundRefused("--target-videos and --target-photos must both be at least 1")
    if args.max_profiles is not None and args.max_profiles < 1:
        raise AutoBoundRefused("--max-profiles must be at least 1 when given")


def preflight_perception(driver) -> None:
    """Refuse to spend a single gesture unless this build can actually see a card.

    The live failure spent five real Passes discovering that its frame source was unreachable.
    Everything this campaign needs is checkable before the first profile: a content band, the
    calibrated heart template the segmenter refuses to run without, and the mute template the
    per-card screen matches against.  Note what is deliberately NOT required -- an item index,
    `targeting_calibration`, or enabled openers.  Those gate numbering items for a MODEL, and
    this campaign never numbers anything for anyone.
    """
    if driver.content_band is None:
        raise AutoBoundRefused(
            "apps.hinge.content_band is unavailable, so no card rect can be segmented at all")
    if driver._template("like") is None:
        raise AutoBoundRefused(
            "the calibrated 'like' glyph template could not be loaded, so segmentation cannot "
            "tell a card from a heartless context block and every frame would yield zero cards")


def build_driver(config_path: str):
    """Load config, construct HingeDriver and open its session. Never called before the checks."""
    from operation_love import config as cfg_mod

    cfg = cfg_mod.load(config_path)
    driver = HingeDriver(cfg)
    driver.open_session()
    return driver


def _capture_command(args) -> int:
    """Drive the deck for a whole campaign, holding the device lock the entire time.

    ADDED 2026-08-22 (tools/_devicelock.py): this command spends real advance actions on real
    profiles over many profiles, and until now nothing stopped a hub run from driving the same
    phone mid-campaign.

    `_refuse_unconfirmed` runs HERE, before the lock is requested (found+fixed 2026-08-22): it
    used to live inside `_capture_command_unlocked`, so an invocation missing a confirmation
    phrase took the operator's real device lock FIRST and only then refused. A live campaign
    already holding that lock turned every such mistake into "Android device is already in
    use...", hiding the actual confirmation error the operator needed to see and act on.
    `_refuse_unconfirmed`'s own docstring already promises "before ... a device is touched" --
    the lock is a device touch too, so the call has to be out here, with the lock scoped to the
    real campaign it protects rather than to the argument validation that precedes it.
    """
    _refuse_unconfirmed(args)
    # `None`, not `args.config`, per holding_the_device's rule: the lock reads nothing out of a
    # config and this command loads/validates its own (`build_driver` -> `cfg_mod.load`), so
    # passing it a path the helper does not consume only re-arms the unloadable-config escape
    # hatch -- a config that parses as YAML but fails validation would otherwise reach
    # `_resolve_serial`'s `adb devices` UNLOCKED before the real error surfaced.
    return run_holding_the_device(None, _capture_command_unlocked, args)


def _capture_command_unlocked(args) -> int:
    cfg_map, cfg_raw = bound._load_config_mapping(args.config)
    app_cfg = bound._hinge_app_config(cfg_map)
    serial, adb_path = bound._resolve_serial(app_cfg, args.serial)
    version_name = bound._device_version_name(serial, adb_path,
                                              app_cfg.get("package", _DEFAULT_PACKAGE))
    out_dir = bound._private_out_dir(args.out, prefix="videoauto")
    driver = build_driver(args.config)
    try:
        preflight_perception(driver)
        manifest = run_capture(
            out_dir=out_dir, driver=driver, advance=args.advance, serial=serial,
            config_sha256=None if cfg_raw is None else bound._sha(cfg_raw),
            target_videos=args.target_videos, target_photos=args.target_photos,
            max_profiles=args.max_profiles, hinge_version_name=version_name,
            driver_content_band=driver.content_band)
    finally:
        driver.close()
    print_run_summary(manifest)
    return 0 if manifest["completed"] else 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m tools.hinge_video_bound_auto",
        description="Automated, AI-labeled still-photo bound campaign for Hinge. The label "
                    "channel is CIRCULAR by owner decision: the video false-accept count it "
                    "produces is zero by construction and is not evidence.")
    sub = ap.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="drive the deck and build an AI-labeled corpus")
    capture.add_argument("--out", default=None)
    capture.add_argument("--config", default="config.yaml")
    capture.add_argument("--serial", default=None)
    # required, with NO default: the advance action spends something real on every profile, so
    # it is chosen per run and can never be inherited from a default nobody typed.
    capture.add_argument("--advance", choices=list(ADVANCE_ACTIONS), required=True,
                         help="what to spend on each profile after it is captured")
    capture.add_argument("--confirmation", default="",
                         help=f"required exact phrase: {CIRCULAR_ACCEPTANCE}")
    capture.add_argument("--like-confirmation", dest="like_confirmation", default="",
                         help=f"required exact phrase for --advance like: {LIKE_CONFIRMATION}")
    capture.add_argument("--target-videos", dest="target_videos", type=int,
                         default=bound.MIN_VIDEO_CARDS)
    capture.add_argument("--target-photos", dest="target_photos", type=int,
                         default=bound.MIN_PHOTO_CARDS)
    capture.add_argument("--max-profiles", dest="max_profiles", type=int, default=None)

    measure_cmd = sub.add_parser("measure",
                                 help="offline bound over one or more captured campaigns")
    measure_cmd.add_argument("campaign", nargs="+",
                             help="one or more completed campaign directories, merged into one "
                                  "denominator; every field the artifact binds must agree")
    measure_cmd.add_argument("--config", default=None)

    emit_cmd = sub.add_parser("emit", help="freeze bound.json and print the config paste block")
    emit_cmd.add_argument("campaign", nargs="+",
                          help="the same directory list `measure` passed; bound.json is written "
                               "into the first")
    emit_cmd.add_argument("--config", default="config.yaml")
    emit_cmd.add_argument("--serial", default=None)

    args = ap.parse_args(argv)
    try:
        if args.command == "capture":
            sys.exit(_capture_command(args))
        elif args.command == "measure":
            result = measure([bound._existing_campaign_dir(path) for path in args.campaign],
                             config_path=args.config)
            bound.print_bound_report(result)
            print("\nbound holds against this campaign's own labels. Run `emit` to freeze it, "
                  "and read the\ncaveat above before quoting the video-accept line anywhere.")
        else:
            roots = [bound._existing_campaign_dir(path) for path in args.campaign]
            campaign = roots[0]
            artifact, paste = emit(roots, config_path=args.config,
                                   serial_override=args.serial)
            print(f"wrote {campaign / 'bound.json'} "
                  f"(evidence_sha256 {artifact['evidence_sha256']})\n")
            print(LABEL_BLIND_SPOT + "\n")
            print("Paste into config.yaml by hand. These numbers install NUMBERING readiness "
                  "only;\nAuto stays blocked by its own separate release chain.\n")
            print(yaml.safe_dump({"apps": {"hinge": {"still_photo_bound_evidence": paste}}},
                                 sort_keys=False))
    except bound.VideoBoundRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
