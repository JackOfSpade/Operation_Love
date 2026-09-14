"""Measure the two operator-calibrated targeting bounds ops/RUNBOOK.md's "Hinge targeted-opener
calibration" section demands before any targeted opener or targeted AUTO like may run:

    identity_match_max_dist   -- must end up strictly below the known 2.565 collision
    inline_item_max_dist      -- must end up strictly below the known 14.91 collision

These are NOT tuning knobs. They gate whether a like/opener is allowed to land on the item the
model actually chose. A wrong bound means liking the WRONG PERSON's item, which is the one
failure mode this whole family of code exists to make impossible (doc 5.6's "never substitute a
different item" rule). So this tool's own rule is the same one: when a measurement is ambiguous,
REFUSE and print nothing, rather than emit a best-guess number.

FIVE SUBCOMMANDS:

    python -m tools.hinge_calibrate capture --split calibration --profiles 4
    python -m tools.hinge_calibrate capture --split heldout --profiles 3
    python -m tools.hinge_calibrate measure ops/calibration/targeting_20260813T120000Z \\
        ops/calibration/targeting_20260813T153000Z
    python -m tools.hinge_calibrate verify-entry-anchor \\
        ops/calibration/targeting_20260813T1300_photo_calibration
    python -m tools.hinge_calibrate attach-operational-evidence \\
        ops/calibration/targeting_20260813T1300_photo_calibration \\
        ops/calibration/operational_evidence_20260813T... --config config.yaml
    python -m tools.hinge_calibrate observe-check --supervised --confirmation OBSERVE_ONLY

`capture` TOUCHES THE PHONE. It screencaps (read-only) and performs small, humanized, forward
read-scrolls through the driver's own transport (`driver._scroll_down_one`, the same private
method `hinge.py`'s ordinary profile read uses, going through the same forbidden-zone guard,
jitter and log-normal timing every other gesture in this repo gets -- never a raw `adb shell
input tap`/`input swipe`). It NEVER taps a heart or sends a comment, and
NEVER types anything: every one of those actions is done by the OWNER, by hand, on the phone,
because doing them autonomously would require the exact targeting logic (`item_nav`,
`verify_sheet_item` with a real bound) this tool exists to calibrate in the first place --
using it here would be circular, and guessing which item to tap is exactly doc 5.6's
"never substitute a different item" rule with the stakes turned up. The tool only ever prompts
clearly for what a human must do next, then screencaps once it is done.

Before it presents any supervised operational-check prompts, a completed capture automatically
writes ``entry_anchor_ledger.json`` beside ``manifest.json``.  That is a hash-bound, entirely
offline reverse replay of the exact saved card frames through production ``item_nav``: it
requires a confirmed top, complete photo-only numbering/translation, a measured bottom-up entry
anchor and shift ledger for every photo model item, and a cancellation dry run that stops before
even a replay capture or scroll.  It owns no ADB handle, cannot issue phone input, and never
emits a runtime targeting bound or YAML.  A replay refusal leaves the capture incomplete; do not
replace it with a free-form checklist assertion.

`measure` is PURE OFFLINE ANALYSIS. It NEVER opens an ADB session, NEVER constructs a touch
transport, and NEVER screencaps -- it only reads the PNG frames and manifest.json a previous
`capture` run already wrote to disk. It freezes each bound from the CALIBRATION split only, then
applies it UNCHANGED to a SEPARATELY COLLECTED held-out split (never carved retroactively out of
one pool -- that is not held-out evidence). Any foreign accept on held-out data invalidates the
bound; a false refusal is a stop/re-measure signal. Neither ever widens a bound to recover a
refusal. Only on total success does it print the exact `targeting_calibration:` YAML block
ops/RUNBOOK.md documents, for the owner to paste into config.yaml by hand -- this tool never
writes config.yaml itself, so binding a calibration to a running config stays a deliberate act.

`observe-check` is a *pre-calibration, preliminary* passive-cycle record for the one catch-22
that numeric measurement cannot solve: calibration correctly withholds numbered item payloads
and real provider requests until calibration exists, while operators need to establish that the
manual device path and Gemini credentials are live. It disables enumeration, sends one synthetic
non-personal Gemini request, and hashes screenshots around owner-performed pass/heart/send
actions. It never asks a provider about a profile, constructs the production Worker, renders a
hub opener, records a store label, targets an item, or injects device input.

Consequently this artifact is deliberately NOT an ``observe_timing_and_quota`` release check and
cannot authorize AUTO. After numeric calibration is installed, run a separate production OBSERVE
validation against that exact device/build and capture evidence of hub pre-tap publication,
post-tap item verification, real provider behaviour, store persistence, and paywall/refusal
handling. Only that post-calibration production validation can satisfy RUNBOOK's final AUTO gate.

No on-device automation server: like every tool in this package, all device I/O goes through
host-side ADB (`operation_love.drivers.adb.Adb` / `operation_love.drivers.hinge.HingeDriver`)
only -- no uiautomator2, no atx-agent, no accessibility service (ops/HINGE-PIXEL-RUNBOOK.md §5).

Privacy: every frame this tool saves is a screenshot of a real person's dating profile. The
default output lives under ops/calibration/, which is gitignored (see .gitignore's comment above
that line) -- local-only by rule, not by accident. Nothing here uploads, copies outside the
repo, or transmits a frame anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import yaml

from operation_love import config as cfg_mod
from operation_love.drivers import hinge as hinge_mod
from operation_love.drivers.adb import parse_devices_output
from operation_love.drivers.frameshift import ShiftEstimationError, estimate_shift
from operation_love.drivers.hinge import HingeDriver
from operation_love.drivers.item_crops import (
    EXCLUSION_REATTACH_PROBE_MISSING, PHOTO_ONLY_POLICY_ID,
    STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC, ItemCropError, ItemPayload, StillPhotoDwell,
    build_item_payload, card_center_offset_frac, dwell_exact_over_rect, signature_of,
    still_photo_evidence_from_drift, still_photo_reattach_legs,
    unnumber_unless_confident_photo, unnumber_without_still_photo_evidence)
from operation_love.drivers.item_identity import (
    IdentityError, ProfileIdentity, capture_profile_identity, compare_profile_identity)
from operation_love.drivers.item_index import ItemIndexError, build_item_index
from operation_love.drivers.item_nav import (
    NAV_ANCHOR_UNMEASURED, ItemNavigationError, navigate_to_item)
from operation_love.drivers.item_verify import (
    SheetVerificationError, verification_blocker, verify_sheet_item)
from operation_love.drivers.base import ActionCancelled
from operation_love.drivers.like_composer import ComposerDetectionError, locate_inline_composer
from operation_love.drivers.scroll_step import ScrollStepError, plan_scroll_step
from operation_love.drivers.scroll_top import (
    ScrollTopError, band_fingerprint, confirm_scroll_top, fingerprint_distance)
from operation_love.drivers.segment import SegmentationError, segment_frame
from operation_love.human import human_cooldown, human_delay
from operation_love.opener.opener import GeminiOpener
from operation_love.private_files import (
    atomic_write_private_bytes,
    atomic_write_private_text,
    ensure_private_dir,
    load_private_dotenv,
)
from tools._devicelock import run_holding_the_device
from tools.hinge_operational_evidence import EvidenceRefused as _DeviceEvidenceRefused
from tools.hinge_operational_evidence import _read_device_evidence as _read_shared_device_evidence

_TOOL_VERSION = "5"
_CALIBRATION_SCHEMA_VERSION = 3
_COMPOSER_LAYOUT_ID = "hinge_inline_v1"
_UNATTENDED_PROVENANCE_SCHEMA_VERSION = 1
_UNATTENDED_REVIEW_SCHEMA_VERSION = 1
_UNATTENDED_REVIEW_KIND = "hinge_unattended_calibration_independent_review"

# This is intentionally awkward: unattended capture spends real likes and passes, and its
# targeting observations are not independent of the transport/vision stack being measured.
# It is a run-time acknowledgement, not a config switch that can accidentally become sticky.
_UNATTENDED_CONFIRMATION = "I_ACCEPT_UNATTENDED_CIRCULAR_CALIBRATION_RISK"
_HYBRID_REVIEW_CONFIRMATION = "I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK"
# Owner-directed exception to the tool's default never-send design (see
# _automated_pass_from_verified_composer): a real, permanent Send Priority Like in place of
# Pass, requested for this account specifically because it runs unlimited HingeX likes. Kept
# behind its own confirmation phrase, separate from _HYBRID_REVIEW_CONFIRMATION, so opting into
# real sends is never a side effect of opting into hybrid review.
_SEND_LIKE_CONFIRMATION = "I_ACCEPT_REAL_PRIORITY_LIKE_SEND_RISK"
# Automated capture must never navigate again after Hinge has opened a persistent inline
# composer.  Alternate the one calibrated photo depth by profile instead: odd profiles cover
# photo model item 1 and even profiles cover photo model item 3.  The policy id is evidence,
# not a suggestion; reviewers and measurement derive every expected action from it. This is the
# DEFAULT strategy; its behaviour must stay byte-identical when --target-items is not passed.
_AUTOMATED_TARGET_STRATEGY_ID = "alternate_photo_1_3_by_profile_ordinal_v1"
# Owner decision 2026-08-22: two live campaigns showed real decks rarely carry three numberable
# photos (prompt cards/videos are common), so the alternating strategy burns its bounded
# per-ordinal skip budget on every even ordinal and no campaign can finish. This SECOND,
# explicitly-selected strategy always targets photo model item 1, accepting that it proves less
# about deep-item navigation. Opt in only via `capture --target-items photo-1-only`; the
# alternating id above remains the default in every other respect.
_PHOTO_1_ONLY_TARGET_STRATEGY_ID = "photo_1_only_v1"
_ACCEPTED_TARGET_STRATEGY_IDS = frozenset(
    (_AUTOMATED_TARGET_STRATEGY_ID, _PHOTO_1_ONLY_TARGET_STRATEGY_ID))
# `capture --target-items` CLI spelling -> strategy id. Kept as the single place that maps the
# operator-facing flag value to the internal id so the CLI and _cmd_capture cannot drift apart.
_TARGET_ITEMS_FLAG_STRATEGY_IDS = {
    "alternate-1-3": _AUTOMATED_TARGET_STRATEGY_ID,
    "photo-1-only": _PHOTO_1_ONLY_TARGET_STRATEGY_ID,
}
# A deliberately strict, *temporary* navigation guard.  This is not a measured calibration
# value, is never emitted, and exists only to keep the existing navigator fail-closed while
# producing the evidence from which the real bound will subsequently be measured.
_UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST = 1.0

_SPLIT_CALIBRATION = "calibration"
_SPLIT_HELDOUT = "heldout"
_SPLITS = (_SPLIT_CALIBRATION, _SPLIT_HELDOUT)

# item_identity._IDENTITY_GRID, i.e. the grid `capture_profile_identity` actually fingerprints
# at and the one `item_nav`'s production identity comparison uses. NOT the bare default
# `scroll_top.band_fingerprint`'s own `_FINGERPRINT_GRID = (16, 4)` -- that measures a coarser,
# unrelated question (is this strip the profile-independent filter-chips row at all), documented
# as a trap for exactly this tool in api_contracts.md's own research: pass `grid=(64, 16)`
# EXPLICITLY on every `band_fingerprint`/`capture_profile_identity` call in this file, always.
# Declared locally rather than imported, on `scroll_top._REFUTE_MIN_DIST`'s own precedent (the
# leaf modules do not import back into each other's private constants); a value this tool got
# wrong would silently calibrate a different, narrower question than production checks.
_IDENTITY_GRID = (64, 16)

# item_nav._IDENTITY_FALSE_MATCH_DISTANCE and item_verify._SHEET_FALSE_MATCH_DISTANCE. The
# closest measured different-profile / foreign-card collisions on real Pixel 7a captures.
# `identity_match_max_dist` / `inline_item_max_dist` must both end up STRICTLY below these.  The
# inline v2 config/driver consumer is migrated separately; this tool hard-clamps the emitted
# values now so a later consumer cannot turn a known collision into an acceptance threshold.
_IDENTITY_FALSE_MATCH_DISTANCE = 2.565
_INLINE_FALSE_MATCH_DISTANCE = 14.91

# Card-frame capture is the enumeration pass.  It must therefore use the same closed-loop
# alias-safe planner as enumeration/navigation, not a fixed "small" fraction: a 0.15 fraction
# can still move 333px on this phone, while a short visible card may permit only 281px.  The
# planner draws the distance inside its safe window and carries the profile's smallest observed
# spacing forward, so a short card remains binding after it leaves the viewport.
_CARD_SCROLL_X_FRAC = 0.5

# Ceiling on how many small forward scrolls one profile capture may take before this tool
# refuses rather than spinning the device forever on a profile that never resolves. Generous
# relative to the driver's own _ENUMERATION_CAPTURE_LIMIT (64) precedent, because a calibration
# run is supervised and offline-verified at every step, not timed.
_MAX_CARD_SCROLLS = 60

# A hybrid/automated run can be resumed while Hinge is part way down a card.  There is no
# meaningful scroll ledger across that handoff, so a replay-based rewind has no trustworthy
# distance to spend.  The visual rewind below uses this same generous, explicit card-walk cap:
# it is enough to recover from the deepest capture this tool can create, but never keeps
# swiping a screen whose state it cannot prove.
_MAX_AUTOMATED_TOP_REWIND_STEPS = _MAX_CARD_SCROLLS
# Hinge 10.0.1 settles its filter-chips strip a few pixels after a card advance, so a screencap
# taken mid-settle can land in scroll_top's deliberate 3..9 "cannot tell" dead zone even though
# the very same screen reads an exact confirmed top a moment later (measured 2026-08-21: two live
# captures aborted at 6.672 and 3.875 while the resting screen read 0.000). This re-READS the
# screen a bounded number of times; it issues NO gesture and does not widen any bound, so
# "cannot tell" still never becomes "at top" -- it just stops treating one transient frame as a
# final answer.
_MAX_UNSETTLED_TOP_REPROBES = 3
# A post-Pass identity proof used to spend exactly one planner-sized read scroll. Hinge 10.2.0's
# taller first-card header can leave that first resting frame in scroll_top's deliberate 3..9
# dead zone (measured live 2026-09-04: 7.859); one additional planner-sized scroll exposed the
# sticky name header at 12.281. Keep this separate from the general read/rewind ceilings: it is
# only the bounded proof that a terminal Pass reached a distinct next profile. Before the second
# gesture the helper below re-proves the ordinary Like+Pass deck, so a modal or composer can
# never consume the extra allowance.
_MAX_POST_ADVANCE_STICKY_SCROLLS = 2
# A low-distance sticky-header proof can be a late render rather than a failed advance.  At most
# three *read-only* looks let one transitional framebuffer fall away while still requiring two
# adjacent, independently-safe candidates before accepting it.  This is deliberately separate
# from the post-advance scroll budget: it never licenses a third scroll, edge-back, or tap.
_MAX_POST_ADVANCE_IDENTITY_REPROBES = 3
# A read-scroll frame becomes the navigator's exact zero-drift anchor. The first held-out
# Hinge 10.1.0 run still moved 556px after TWO quiet comparisons, so the app can pause before a
# delayed card snap. Require FOUR quiet comparisons and allow a bounded eight probes: this gives
# a genuinely parked screen more time to prove itself while an app that keeps moving still fails
# closed. Up to 3px is ordinary segmentation/raster jitter and far below the 219px minimum
# gesture measured by the navigator's own entry gate.
_MAX_READ_SCROLL_SETTLE_PROBES = 8
_READ_SCROLL_SETTLE_MAX_SHIFT_PX = 3
_READ_SCROLL_SETTLE_QUIET_COMPARISONS = 4

_ROUND_NDIGITS = 4

_OPERATIONAL_CHECKS = {
    "entry_anchor_scroll_ledger": (
        "Entry anchor / scroll ledger: confirmed top entry, inspected the bottom-up anchor, "
        "frame/shift ledger, item numbering, and cancellation before authorizing a gesture"),
    "item_1_inline_composer_identity": (
        "Item 1 inline composer: operator hearted item 1 without an intervening scroll and "
        "recorded Hinge's immediately auto-focused composer plus the separate profile/item "
        "proof; do not claim a nonexistent unfocused-to-focused transition"),
    "gesture_transport": (
        "Gesture transport: on a supervised disposable profile, operator observed a second "
        "heart move the inline composer and a profile advance clear it without sending; keep "
        "the trace so the operator action is distinguishable from what a screenshot proves"),
    "observe_timing_and_quota": (
        "Pre-calibration passive OBSERVE-cycle/quota bridge: record this tool's completed, "
        "self-hashed observe_check.json after a manual pass/heart/send cycle and synthetic "
        "non-personal Gemini probe. It licenses only a numeric calibration candidate for "
        "production OBSERVE; the separate observe_release_evidence gate still blocks AUTO."),
}

_OBSERVE_CHECK_SCHEMA_VERSION = 2
_OBSERVE_CHECK_CONFIRMATION = "OBSERVE_ONLY"
_ENTRY_ANCHOR_LEDGER_SCHEMA_VERSION = 1
_ENTRY_ANCHOR_LEDGER_FILE = "entry_anchor_ledger.json"
_OPERATIONAL_EVIDENCE_TOOL_VERSION = "2"
_OPERATIONAL_EVIDENCE_ROLES = (
    "confirmed_top_item1_pre",
    "composer_initial_autofocused",
    "composer_stable_autofocused",
    "other_item_composer_moved",
    "new_profile_top_clear",
    "new_sticky_identity",
)
_OPERATIONAL_EVIDENCE_ANALYSES = frozenset((*_OPERATIONAL_EVIDENCE_ROLES,
                                             "item1_profile_identity"))
_OPERATIONAL_EVIDENCE_DEVICE_KEYS = (
    "serial", "model", "display_w", "display_h", "density", "hinge_package",
    "hinge_version_name",
)
_PRELIMINARY_OBSERVE_SCOPE = "pre_calibration_passive_manual_cycle_and_synthetic_quota_v1"
_POST_CALIBRATION_OBSERVE_REQUIREMENTS = (
    "production_worker_and_hub_pre_tap_publication",
    "post_tap_target_item_verification",
    "real_profile_provider_request_and_quota_result",
    "persisted_manual_label_in_production_store",
    "rejected_send_or_paywall_logging",
)
_HYBRID_CHECKPOINT_SCHEMA_VERSION = 1
_HYBRID_CHECKPOINT_KIND = "hinge_hybrid_calibration_checkpoint"
# There is deliberately no sibling "_HYBRID_REVIEW_TOKEN_KIND" for the reviewer-decision record
# (`HybridReviewGate.checkpoint`'s `record` dict / `_approved_hybrid_decision`'s `record` arg),
# unlike _HYBRID_CHECKPOINT_KIND above which IS written (~line 617) and checked on resume
# (~line 5662). Found 2026-09-02: a `_HYBRID_REVIEW_TOKEN_KIND = "hinge_external_reviewer_
# decision"` constant sat here unread by any code or string literal, and every real on-disk
# `manifest.json` under ops/calibration/ (checked across multiple hybrid-reviewed sessions)
# stores decision records with no "kind" key at all -- only checkpoint_file/checkpoint_sha256/
# checkpoint_evidence_sha256/frame_file/frame_sha256/claimed_state/action_plan/decision/source/
# reviewer/human_ground_truth/decided_utc. Adding a `kind` check to `_approved_hybrid_decision`
# would therefore reject every hybrid-reviewed calibration session captured before this date --
# real operational evidence, not a fixture that can just be regenerated -- so the constant was
# deleted rather than wired in. If a tagged review-token schema is wanted later, it must ship
# alongside a migration for the existing decisions, not just a stricter check.
_HYBRID_MAX_ADJUSTMENTS_PER_ACTION = 3
_HYBRID_CAPTURE_MODE = "hybrid_ai_reviewed_automation"
_HYBRID_REVIEW_SOURCE = "external_ai_review"
_CARET_BLINK_RECHECK_S = 0.2
_CARET_BLINK_RECHECK_ATTEMPTS = 16
_SYSTEM_STATUS_BAR_HEIGHT_FRAC = 0.04
# Automated calibration deliberately proves only the prefix through the one photo it will
# touch.  This is NOT a replacement for production's closed-set profile payload: it exists so
# a changed/long Hinge tail cannot prevent a safe, already-visible calibration target from being
# measured.  The identifier is carried in every automated manifest and independently checked
# before a target-scoped session is measured.
_TARGET_SCOPED_PREFIX_PROOF_ID = "photo_only_confirmed_prefix_v1"
# An unsuitable card is allowed to be skipped only before the first heart.  Keep both limits
# explicit: a changed Hinge surface must not turn one requested ordinal into an unbounded run of
# real Passes.
_MAX_PREACTION_PROFILE_SKIPS_PER_ORDINAL = 3
_MAX_PREACTION_PROFILE_SKIPS_PER_SESSION = 12
# A measured large entry translation invalidates the whole enumeration/index relationship. One
# fresh driver-owned rewind and re-enumeration can clear a delayed Hinge snap; a second refusal
# advances by the normal bounded pre-action skip rather than looping on one live profile.
_MAX_ENTRY_DRIFT_REENUMERATION_RESTARTS_PER_PROFILE = 1

# A left-edge Android back gesture delivered through the driver's normal guarded, humanized
# swipe transport.  These are relative geometry, not fixed phone pixels.  It is calibration
# transport only: production ``HingeDriver.dislike`` retains its ordinary deck confirmation.
_EDGE_BACK_START_X_FRAC = 0.01
_EDGE_BACK_END_X_FRAC = 0.44
_EDGE_BACK_Y_FRAC = 0.50
_PASS_POINT_RELOCATION_MAX_FRAC = 0.025


# =====================================================================================
# Small pure helpers
# =====================================================================================

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_distance(value, *, context: str) -> float:
    """Return a finite nonnegative measured distance; NaN/inf must never erase a guard."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise _MeasureRefused(
            f"{context} produced invalid distance {value!r}; refusing rather than letting a "
            "NaN/inf/negative value disable a comparison")
    return float(value)


def _parse_item_numbers(raw: str) -> list[int]:
    """'1, 3,2' -> [1, 2, 3]. Raises ValueError on anything that is not a list of positive ints,
    so the caller can print a clear message and re-prompt rather than silently targeting nothing
    or the wrong item."""
    tokens = [tok.strip() for tok in raw.replace(";", ",").split(",") if tok.strip()]
    if not tokens:
        raise ValueError("no item numbers given")
    numbers = sorted({int(tok) for tok in tokens})
    if any(n < 1 for n in numbers):
        raise ValueError("item numbers must be positive (Hinge's model index is 1-based)")
    return numbers


def _automated_composer_items_for_ordinal(
        ordinal: int, *, strategy_id: str = _AUTOMATED_TARGET_STRATEGY_ID) -> tuple[int, ...]:
    """Return the automated photo target(s) for a 1-based profile ordinal under `strategy_id`.

    Keeping this as the single source of truth prevents an automated/hybrid capture from
    reopening navigation after a persistent composer is visible.  Manual capture deliberately
    does not call this helper: its owner-chosen item list remains part of the supervised path.

    `strategy_id` defaults to the alternating strategy, so every existing caller (and the
    default capture path) is unaffected.  Passing `_PHOTO_1_ONLY_TARGET_STRATEGY_ID` narrows
    every ordinal to item 1 only -- an explicit, owner-accepted narrowing of the evidence that
    does not exercise deep-item navigation.  Any other id is refused rather than silently
    treated as one of the two known strategies.
    """
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError(f"profile ordinal must be a positive integer, got {ordinal!r}")
    if strategy_id == _AUTOMATED_TARGET_STRATEGY_ID:
        return (1 if ordinal % 2 else 3,)
    if strategy_id == _PHOTO_1_ONLY_TARGET_STRATEGY_ID:
        return (1,)
    raise ValueError(f"unknown automated target strategy id {strategy_id!r}")


def _automated_target_depths_for_strategy(strategy_id: str) -> frozenset[int]:
    """All distinct photo model items `strategy_id` can ever target.

    Derived from `_automated_composer_items_for_ordinal` itself (ordinals 1 and 2 span every
    branch of both known strategies) rather than duplicated, so this can never drift from the
    single source of truth above.  Raises the same `ValueError` for an unknown id.
    """
    depths: set[int] = set()
    for ordinal in (1, 2):
        depths.update(_automated_composer_items_for_ordinal(ordinal, strategy_id=strategy_id))
    return frozenset(depths)


def _target_scoped_prefix_reason(index, payload, target_items: tuple[int, ...]) -> str | None:
    """Return why a photo-target prefix cannot be acted on, else ``None``.

    A normal profile read must remain closed-set: later cards can change what the model sees.
    Calibration is narrower.  It hearts exactly one already-numbered photo, then verifies the
    composer against that exact crop; no action depends on an unseen lower card.  That narrower
    claim is sound only when the capture started at a confirmed top, every block through the
    target is fully resolved, the target crop is complete/photo-classified, and the sticky
    identity belongs to this exact index.  In particular, an unresolved predecessor could hide
    a heart and silently shift the target's absolute ordinal, so it is always a refusal.

    Keep this helper in the calibration harness rather than weakening ``ItemIndex.complete`` or
    ``build_item_payload``.  Production callers still require their existing closed-set policy.
    """
    if not getattr(index, "usable", False):
        return "item index is unusable"
    if not getattr(index, "at_scroll_top", False):
        return "index does not carry confirmed absolute scroll-top numbering"
    identity = getattr(index, "identity", None)
    if not getattr(identity, "known", False):
        return "index has no corroborated sticky-header identity"
    if not getattr(payload, "usable", False):
        return "photo-only payload is unusable"
    if not target_items:
        return "target-scoped proof was given no target photo item"

    blocks = tuple(getattr(index, "blocks", ()))
    for item_number in target_items:
        try:
            crop = payload.item(item_number)
        except (ItemCropError, AttributeError) as exc:
            return f"photo item {item_number} is unavailable in the prefix payload ({exc})"
        if (getattr(crop, "image", None) is None or getattr(crop, "signature", None) is None
                or getattr(crop, "frame_index", None) is None
                or getattr(crop, "heart_ordinal", None) is None):
            return f"photo item {item_number} has no complete crop/signature/heart evidence"
        matching = [pos for pos, block in enumerate(blocks)
                    if getattr(block, "heart_ordinal", None) == crop.heart_ordinal]
        if len(matching) != 1:
            return (f"photo item {item_number}'s heart ordinal {crop.heart_ordinal!r} does not "
                    "resolve to exactly one indexed block")
        target_pos = matching[0]
        # ``complete`` is the end-to-end card proof.  Do not merely rely on a visible heart: a
        # fragment crop would make post-tap verification circular and unusable.
        if not getattr(blocks[target_pos], "complete", False):
            return f"target photo item {item_number} was not observed end-to-end"
        for predecessor in blocks[:target_pos]:
            # The confirmed-top filter/header strip is deliberately represented as an
            # unbounded ``leading_chrome`` block.  It is not a profile item and item_index has
            # already proved it cannot be a hidden heart; requiring it to be croppable would
            # make every genuine top-prefix fail for an irrelevant chrome artifact.
            if getattr(predecessor, "kind", None) == "leading_chrome":
                continue
            if not getattr(predecessor, "complete", False):
                return (f"a predecessor at page rows {getattr(predecessor, 'page_y0', '?')}.."
                        f"{getattr(predecessor, 'page_y1', '?')} was not resolved; it may hide "
                        "a heart and shift the target ordinal")
    return None


def _require_collective_target_depths(
        profiles: list, *, split: str, strategy_id: str = _AUTOMATED_TARGET_STRATEGY_ID) -> None:
    """Require every photo depth `strategy_id` can target, across any split large enough to
    cover them.

    One composer pair per profile is intentional.  At two or more profiles the split must
    nevertheless exercise every depth `strategy_id` claims to exercise, otherwise the numeric
    result cannot support that strategy's full target surface.  The requirement is derived from
    `strategy_id` (via `_automated_target_depths_for_strategy`) rather than a fixed {1, 3}: the
    depth requirement exists so a calibration's evidence spans the depths its strategy claims to
    exercise, and deriving it from the strategy keeps that meaning under a narrower strategy
    instead of demanding evidence the strategy never collects.  `photo_1_only_v1` was chosen
    2026-08-22 after item-3 targets refused three times for three different legitimate reasons on
    real decks, and it never collects item 3 at all -- a split entirely captured under it must not
    be refused for "missing" a depth it was never trying to reach.

    `strategy_id` defaults to the alternating strategy, so every existing caller is unaffected.
    """
    if len(profiles) < 2:
        return
    try:
        required = _automated_target_depths_for_strategy(strategy_id)
    except ValueError as exc:
        raise _MeasureRefused(f"{split} split: {exc}") from exc
    captured = {item for profile in profiles for item, _pre, _composer in profile.composer_pairs}
    missing = sorted(required - captured)
    if missing:
        raise _MeasureRefused(
            f"{split} split has {len(profiles)} profiles but lacks composer evidence for "
            f"required photo target depth(s) {missing} under target strategy {strategy_id!r}; "
            "retain one pair per profile but collectively cover every depth that strategy targets")


def _round_down_below(value: float, *, above: float, below: float,
                      ndigits: int = _ROUND_NDIGITS) -> float:
    """Round `value` DOWN to `ndigits` decimals for a friendlier config.yaml paste, without ever
    crossing either the calibration evidence (`above`, e.g. the calibration split's own max
    same-profile/intended-item distance) or the hard collision ceiling (`below`). A cleaner
    number is a nicety; staying strictly between the two is not negotiable, so a rounding that
    would violate either bound is discarded in favour of the untruncated value instead."""
    factor = 10 ** ndigits
    rounded = math.floor(value * factor) / factor
    if above < rounded < below:
        return rounded
    return value


def _validate_v3_calibration_block(block: dict) -> None:
    """Refuse malformed emitted v3 YAML before it leaves this offline tool.

    Runtime/config support is intentionally changed in its owning files.  This local check keeps
    the capture/measure boundary fail-closed while those consumers migrate, rather than asking
    the legacy six-key parser to silently reinterpret inline evidence.
    """
    required = {
        "schema_version", "device", "hinge_version_name", "frame_size_px",
        "composer_layout_id", "item_selection_policy_id", "identity_match_max_dist",
        "inline_item_max_dist",
        "calibrated_at", "identity_band", "content_band",
    }
    if set(block) != required:
        raise _MeasureRefused(f"v3 calibration keys are not exact: {sorted(block)}")
    if block["schema_version"] != _CALIBRATION_SCHEMA_VERSION:
        raise _MeasureRefused("emitted calibration has the wrong schema_version")
    if block["composer_layout_id"] != _COMPOSER_LAYOUT_ID:
        raise _MeasureRefused("emitted calibration has an unsupported composer_layout_id")
    if block["item_selection_policy_id"] != PHOTO_ONLY_POLICY_ID:
        raise _MeasureRefused("emitted calibration has an unsupported item_selection_policy_id")
    if (not isinstance(block["hinge_version_name"], str)
            or not block["hinge_version_name"].strip()):
        raise _MeasureRefused("emitted calibration has no exact Hinge versionName")
    if not isinstance(block["device"], str) or not block["device"].strip():
        raise _MeasureRefused("emitted calibration has no exact device serial")
    if not isinstance(block["calibrated_at"], str) or not block["calibrated_at"].strip():
        raise _MeasureRefused("emitted calibration has no calibration provenance")
    size = block["frame_size_px"]
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        raise _MeasureRefused("emitted calibration has invalid frame_size_px")
    for key, ceiling in (("identity_match_max_dist", _IDENTITY_FALSE_MATCH_DISTANCE),
                         ("inline_item_max_dist", _INLINE_FALSE_MATCH_DISTANCE)):
        value = block[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value < ceiling):
            raise _MeasureRefused(
                f"emitted calibration {key} is not finite, positive, and strictly below {ceiling}")
    for key, size in (("identity_band", 4), ("content_band", 2)):
        values = block[key]
        if (not isinstance(values, list) or len(values) != size
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in values)):
            raise _MeasureRefused(
                f"emitted calibration {key} must contain {size} finite numeric values")


class _CaptureAbort(RuntimeError):
    """The owner chose to stop the whole capture run (declined a retry prompt), or a per-profile
    refusal could not be resolved interactively. Raised deliberately rather than left to an
    uncaught exception, so `_cmd_capture` can still close the driver and write whatever manifest
    evidence exists before exiting non-zero."""


class _RestartProfile(RuntimeError):
    """A reviewer requested a fresh, driver-owned rewind before a pending action."""


class _PreActionProfileRetry(RuntimeError):
    """This profile cannot supply its requested target before any heart was sent."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _ProfileSkipped(RuntimeError):
    """A guarded public Pass advanced an unsuitable, entirely pre-action profile."""

    def __init__(self, record: dict):
        super().__init__(record.get("reason_code", "pre_action_profile_skipped"))
        self.record = record


class _UnsupportedEntryDeck(_CaptureAbort):
    """An ordinary deck is visible, but its entry layout cannot prove calibration top.

    The frame is safe only as the input to a separately audited public profile advance. It is
    never returned as a card frame and therefore can never become targeting evidence.
    """

    def __init__(self, frame: bytes, *, state: str, reason: str):
        super().__init__(f"unsupported entry deck ({state}): {reason}")
        self.frame = frame
        self.state = state
        self.reason = reason


class _MeasureRefused(RuntimeError):
    """A measurement step in `measure` could not produce trustworthy evidence -- an unreadable
    frame, an unusable item index/payload, a composer captured for an item number the index never
    produced, or similar. Caught once at the top of `_cmd_measure` and turned into a `REFUSED:`
    message plus a non-zero exit. Never partially handled: doc 5.6's rule is refuse, not guess,
    and that applies to this calibration tool's own output exactly as much as to a live run."""


class _HybridReviewGate:
    """Fail-closed stdin checkpoints for externally AI-reviewed calibration transport.

    A checkpoint is a private, atomic PNG plus JSON record.  The process does not infer who
    supplied stdin: it records the configured Codex review provenance, never human ground truth.
    EOF, a malformed answer, or a refusal stops before another device-changing action.
    """
    def __init__(self, out_dir: Path, *, device: dict, config_provenance: dict,
                 reviewer_model: str, reviewer_process: str, reviewer_id: str | None = None,
        reviewer_version: str | None = None):
        self.dir = out_dir / "hybrid_review"
        ensure_private_dir(self.dir)
        self.device = device
        self.config_provenance = config_provenance
        self.reviewer = {"source": _HYBRID_REVIEW_SOURCE, "id": reviewer_id or reviewer_model,
                         "model": reviewer_model, "version": reviewer_version or reviewer_process,
                         "process": reviewer_process}
        self.sequence = 0
        self.decisions: list[dict] = []

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Durably publish one private checkpoint member without a partially-written file."""
        try:
            atomic_write_private_bytes(path, data, parent=path.parent)
        except OSError as exc:
            raise _CaptureAbort(f"could not atomically write hybrid checkpoint {path.name}: {exc}") from exc

    def checkpoint(self, frame: bytes, *, claimed_state: str, action_plan: dict) -> dict:
        """Persist the exact visual/action binding, then require one exact stdin decision."""
        self.sequence += 1
        stem = f"{self.sequence:05d}"
        frame_name, checkpoint_name = f"{stem}.png", f"{stem}.checkpoint.json"
        frame_path, checkpoint_path = self.dir / frame_name, self.dir / checkpoint_name
        if frame_path.exists() or checkpoint_path.exists():
            raise _CaptureAbort(f"hybrid reviewer checkpoint namespace collision at {stem}")
        frame_sha256 = _sha256(frame)
        self._atomic_write(frame_path, frame)
        body = {
            "schema_version": _HYBRID_CHECKPOINT_SCHEMA_VERSION,
            "kind": _HYBRID_CHECKPOINT_KIND,
            "sequence": self.sequence,
            "frame": {"file": frame_name, "sha256": frame_sha256},
            "claimed_state": claimed_state,
            # Exact action/item/point/predicates are intentionally visible to the reviewer.
            "action_plan": action_plan,
            "device": self.device,
            "config_sha256": self.config_provenance["sha256"],
            "human_ground_truth": False,
        }
        checkpoint = {**body, "evidence_sha256": _canonical_json_digest(body)}
        checkpoint_raw = (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self._atomic_write(checkpoint_path, checkpoint_raw)
        checkpoint_sha256 = _sha256(checkpoint_raw)
        print(
            "HYBRID REVIEW REQUIRED: inspect private checkpoint "
            f"{checkpoint_path} and frame {frame_path}; then enter exactly "
            f"APPROVE {checkpoint_sha256}, REFUSE {checkpoint_sha256}, RETRY {checkpoint_sha256}, "
            f"RESTART_PROFILE {checkpoint_sha256}, SCROLL_UP {checkpoint_sha256}, "
            f"SCROLL_DOWN {checkpoint_sha256}, or ABORT {checkpoint_sha256}",
            flush=True)
        response = sys.stdin.readline()
        # Preserve every character except the terminal LF: spaces/case/extra words are refusals.
        response = response[:-1] if response.endswith("\n") else response
        commands = {
            f"APPROVE {checkpoint_sha256}": "approved",
            f"REFUSE {checkpoint_sha256}": "refused",
            f"RETRY {checkpoint_sha256}": "retry",
            f"RESTART_PROFILE {checkpoint_sha256}": "restart_profile",
            f"SCROLL_UP {checkpoint_sha256}": "scroll_up",
            f"SCROLL_DOWN {checkpoint_sha256}": "scroll_down",
            f"ABORT {checkpoint_sha256}": "aborted",
        }
        decision = commands.get(response, "eof" if response == "" else "invalid")
        record = {
            "checkpoint_file": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_evidence_sha256": checkpoint["evidence_sha256"],
            "frame_file": str(frame_path), "frame_sha256": frame_sha256,
            "claimed_state": claimed_state, "action_plan": action_plan,
            "decision": decision, "source": _HYBRID_REVIEW_SOURCE,
            "reviewer": self.reviewer, "human_ground_truth": False,
            "decided_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.decisions.append(record)
        if decision in {"refused", "aborted", "eof", "invalid"}:
            raise _CaptureAbort(
                "hybrid external AI review did not approve the next transport action "
                f"({decision}); session stopped before it could continue")
        return record


def _check_vision_stacks() -> None:
    """Both vision stacks this tool needs are independent and fail independently (see
    api_contracts.md §8): PIL+numpy for the identity-band path (`scroll_top.band_fingerprint`,
    `item_identity.capture_profile_identity`), opencv-python+numpy for the inline item path
    (`item_crops.build_item_payload`, `item_verify.verify_sheet_item`). Checked once, up front,
    with an actionable install message -- the alternative is a `_MeasureRefused`/`DriverClosed`
    surfacing many steps later with a less specific "import failed" message."""
    missing = []
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except Exception:  # noqa: BLE001 -- report clearly, exit non-zero
        missing.append("PIL + numpy (`pip install -e '.[ml]'` or `pip install pillow numpy`) "
                       "-- needed for the identity-band measurement")
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001 -- report clearly, exit non-zero
        missing.append("opencv-python + numpy (`pip install -e '.[hinge]'`) -- needed for the "
                       "inline-composer/item measurement")
    if missing:
        print("ERROR: missing vision dependencies:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)


# =====================================================================================
# capture
# =====================================================================================

def _default_out_dir(prefix: str = "targeting") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("ops/calibration") / f"{prefix}_{stamp}"


def _preflight_serial(cfg) -> tuple[str, str]:
    """The connected ADB serial must EXACTLY equal apps.hinge.serial -- refuse otherwise.

    `HingeDriver.open_session()` already refuses if the configured serial is not AMONG the ready
    devices, but that is weaker than what a calibration run needs: with a second device also
    connected, it would happily proceed against the right one while a wrong one sits on the same
    bench. This check requires the ready-device list to be EXACTLY `[configured]` -- one device,
    the right one -- before a single gesture is sent."""
    app_cfg = (getattr(cfg, "apps", {}) or {}).get("hinge", {}) or {}
    configured = app_cfg.get("serial") or None
    adb_path = app_cfg.get("adb_path", "adb")
    if not isinstance(configured, str) or not configured.strip():
        print("ERROR: apps.hinge.serial is not configured in config.yaml. A calibration "
              "measured without a fixed serial could silently apply to the wrong device. Set "
              "apps.hinge.serial to the phone's exact `adb devices` serial and retry.",
              file=sys.stderr)
        sys.exit(1)
    try:
        result = subprocess.run(
            [adb_path, "devices"], capture_output=True, timeout=10, check=False)
    except FileNotFoundError:
        print(f"ERROR: adb binary not found ({adb_path!r} not on PATH).", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: `adb devices` timed out.", file=sys.stderr)
        sys.exit(1)
    ready = parse_devices_output(result.stdout.decode("utf-8", errors="replace"))
    if ready != [configured]:
        print(
            f"ERROR: connected ADB device(s) {ready!r} do not EXACTLY match the configured "
            f"apps.hinge.serial {configured!r}. This tool measures a safety bound for one "
            "specific physical device and refuses to guess which one is on the bench when more "
            f"than one (or the wrong one) is connected. Connect exactly {configured!r} and "
            "nothing else, then retry.", file=sys.stderr)
        sys.exit(1)
    return configured, adb_path


def _device_evidence(driver: HingeDriver) -> dict:
    """Model, display size, density, and the Hinge package's versionName -- read-only `adb
    shell` queries (`getprop`, `wm density`, `dumpsys package`), none of which send any input.
    Raises if the evidence cannot be read/parsed, on the same fail-loud rule as everything else
    here: evidence a later audit cannot actually read is not evidence.

    A thin HingeDriver-shaped adapter over `tools.hinge_operational_evidence`'s own
    `_read_device_evidence` -- found 2026-09-02: this function and that one had independently
    hand-rolled the identical ~20-line probe/parse, and had already drifted (this copy was
    missing that module's density check). One implementation now backs both callers; this
    wrapper's job is only to unpack the three facts a HingeDriver already carries (`.adb`,
    `.serial`, `.package`, which -- unlike `hinge_operational_evidence`'s fixed package constant
    -- can be a config override) and to re-raise the shared helper's `EvidenceRefused` as a
    plain `RuntimeError`, so every existing caller here keeps seeing the exact exception type
    it always has."""
    try:
        return _read_shared_device_evidence(
            driver.adb, serial=driver.serial, package=driver.package)
    except _DeviceEvidenceRefused as exc:
        raise RuntimeError(str(exc)) from exc


def _plan_card_scroll(frame: bytes, *, content_band, like_template, like_threshold,
                      profile_min_spacing_px: int | None):
    """Plan one forward calibration-enumeration step from this exact saved-frame candidate.

    Returning the next profile minimum with the plan makes its state explicit at the one capture
    call site.  A planning refusal means there is no safe next frame to capture: callers retry
    the profile from its confirmed top rather than issuing a larger probe gesture and hoping the
    offline replay can repair it later.
    """
    segmentation = segment_frame(
        frame, content_band=content_band, like_template=like_template,
        like_threshold=like_threshold)
    step = plan_scroll_step(
        segmentation, x_frac=_CARD_SCROLL_X_FRAC,
        profile_min_spacing_px=profile_min_spacing_px)
    if step.spacing_px is not None:
        profile_min_spacing_px = (step.spacing_px if profile_min_spacing_px is None
                                  else min(profile_min_spacing_px, step.spacing_px))
    return step, profile_min_spacing_px


def _settled_read_scroll_frame(driver: HingeDriver) -> bytes:
    """Capture the parked result of a card-enumeration gesture, not its in-flight repaint.

    ``adb shell input swipe`` returns when the finger path ends, while Hinge can continue its
    inertial/card-snap animation. The 2026-08-26 Pixel 7a held-out run measured two immediate
    post-gesture frames moving another 554px and 559px before navigation began. That correctly
    tripped ``navigate_to_item``'s unaccounted-drift gate and spent two otherwise usable profiles.

    The pre-gesture dwell controls cadence; it cannot settle a gesture that has not happened
    yet. Keep the same bounded humanized dwell on the other side, then require four consecutive
    frame comparisons to measure no more than 3px of residual page motion. The last proven frame
    becomes both the recorded read position and the exact navigation entry reference.
    """
    time.sleep(human_delay(driver.dwell_s))
    prior = driver.adb.screencap()
    refusals: list[str] = []
    quiet_comparisons = 0
    for _probe in range(_MAX_READ_SCROLL_SETTLE_PROBES):
        time.sleep(human_delay(driver.dwell_s))
        current = driver.adb.screencap()
        if current == prior:
            quiet_comparisons += 1
        else:
            try:
                shift = estimate_shift(
                    prior, current, content_band=driver.content_band)
            except ShiftEstimationError as exc:
                quiet_comparisons = 0
                refusals.append(f"{type(exc).__name__}: {exc}")
            else:
                if shift.ok and abs(shift.delta_px) <= _READ_SCROLL_SETTLE_MAX_SHIFT_PX:
                    quiet_comparisons += 1
                else:
                    quiet_comparisons = 0
                    refusals.append(shift.reason)
        if quiet_comparisons >= _READ_SCROLL_SETTLE_QUIET_COMPARISONS:
            return current
        prior = current
    detail = refusals[-1] if refusals else "frames never became byte-stable"
    raise _CaptureAbort(
        "card-enumeration scroll did not park within the bounded "
        f"{_MAX_READ_SCROLL_SETTLE_PROBES}-comparison settle window: {detail}")


@dataclass(frozen=True)
class _TargetFrameProof:
    """What one exact-frame target screen actually established, carried to whoever claims it.

    The mute-control verdict used to be restated as a bare `True` in the pre-heart checkpoint,
    ten lines below the call that proves it.  That coupled the claim to statement ORDER: move,
    wrap or drop the screen and the checkpoint keeps publishing a claim nothing measured.  This
    repo has already paid for exactly that once (the old hardcoded `photo_only_item_verified`),
    so the verdict now travels as data.  `frame_sha256` binds it to the bytes that were screened:
    a proof taken from any other frame cannot vouch for the frame a reviewer is looking at.

    `heart_visible` is the other verdict this screen reaches and used to restate as a literal:
    the segmentation located EXACTLY ONE heart, at the navigator's point, inside the reviewed
    card's rows on these bytes.  Same rule as the mute leg -- carried as data, read back through
    a function that refuses when the proof describes some other frame.
    """
    block: object
    frame_sha256: str
    mute_control_absent: bool
    heart_visible: bool


def _screened_mute_control_absent(proof: object, frame: bytes) -> bool:
    """Report the mute-screen verdict for `frame`, refusing when nothing screened these bytes.

    A checkpoint predicate is read as evidence, both by the offline review chain and by whoever
    approves a real tap, so this is deliberately unable to answer from anything except a proof
    produced by a screen of this exact frame.  A refactor that relocates the screen therefore
    stops the run here instead of publishing an unearned claim.
    """
    if not isinstance(proof, _TargetFrameProof) or proof.frame_sha256 != _sha256(frame):
        raise _CaptureAbort(
            "checkpoint refused: no target-frame mute-control screen is bound to the exact "
            "frame whose heart would be approved")
    if not proof.mute_control_absent:
        # The screen that sees a control already refuses upstream, so arriving here means a
        # later edit let a negative verdict through.  A reviewer must never be offered a heart
        # on a frame the screen itself calls a video, so refuse rather than publish the False.
        raise _CaptureAbort(
            "checkpoint refused: the target-frame mute-control screen found a visible mute "
            "control on the exact action frame")
    return proof.mute_control_absent


def _located_target_heart_visible(proof: object, frame: bytes) -> bool:
    """Report the heart-location verdict for `frame`, refusing when nothing located these bytes.

    `target_heart_visible` is the claimed_state the reviewer is asked to approve, so it is the
    LAST predicate that may be a literal.  It answers only from a proof whose segmentation found
    exactly one heart, at the navigator's point, in the reviewed card's rows on this exact frame.
    """
    if not isinstance(proof, _TargetFrameProof) or proof.frame_sha256 != _sha256(frame):
        raise _CaptureAbort(
            "checkpoint refused: no target-heart location is bound to the exact frame whose "
            "heart would be approved")
    if not proof.heart_visible:
        raise _CaptureAbort(
            "checkpoint refused: the target-heart location did not find the reviewed heart on "
            "the exact action frame")
    return proof.heart_visible


def _verified_target_frame_proof(driver: HingeDriver, target, *, frame: bytes,
                                 content_band, like_template, like_threshold: float,
                                 expected_rows=None, expected_point=None) -> _TargetFrameProof:
    """Re-prove the exact target card/heart and absence of visible video UI on one frame.

    Returns the evidence, not merely the card block, so a caller that wants to claim the screen
    ran has to hold what the screen returned (see `_TargetFrameProof`).

    `expected_rows`/`expected_point` default to what the navigator parked, and are passed
    explicitly by the one caller that re-proves the SAME card on a DIFFERENT frame: the pre-heart
    loop, after the still-photo probe's return leg left a measured page residual.  They are the
    navigator's own rows and point carried across that measurement -- never a re-identified card
    (owner rule 2026-08-11), and never a relaxation: the segmentation still has to find exactly
    one card at exactly those rows with exactly one heart at exactly that point.
    """
    rows = tuple(target.block_frame_rows if expected_rows is None else expected_rows)
    point = tuple(target.point if expected_point is None else expected_point)
    try:
        segmentation = segment_frame(
            frame, content_band=content_band, like_template=like_template,
            like_threshold=like_threshold)
        if not segmentation.ok:
            raise SegmentationError("; ".join(segmentation.failures))
        matching_blocks = [
            block for block in segmentation.blocks
            if (block.y0, block.y1) == rows
        ]
        heart_visible = (len(matching_blocks) == 1
                         and list(matching_blocks[0].hearts) == [point])
        if not heart_visible:
            raise SegmentationError(
                "frame does not contain exactly one reviewed heart in the reviewed card")
        screen = getattr(driver, "_target_frame_video_screen_reason", None)
        if not callable(screen):
            raise SegmentationError("driver has no exact target-frame video screen")
        video_reason = screen(frame, matching_blocks[0])
        if video_reason is not None:
            raise SegmentationError(video_reason)
    except SegmentationError as exc:
        raise _CaptureAbort(
            "target photo proof refused on the exact action frame: " + str(exc)) from exc
    # `video_reason` is the screen's own verdict on these exact bytes.  Deriving the published
    # predicate from it here, rather than restating it at the checkpoint, is what keeps the two
    # from drifting apart when either moves.
    return _TargetFrameProof(block=matching_blocks[0], frame_sha256=_sha256(frame),
                             mute_control_absent=video_reason is None,
                             heart_visible=heart_visible)


@dataclass(frozen=True)
class _StillPhotoProof:
    """The C1-C3 still-photo verdict for one exact action frame, carried as data.

    `positive_still_photo_evidence_verified` was the last hardcoded `True` in the pre-heart
    checkpoint, and it carried a comment saying so.  It is now the outcome of re-running the
    whole acceptance on frames this object names by digest: the action frame the reviewer is
    about to approve, plus a no-input dwell burst taken in the pre-heart window.  Nothing about
    this can be satisfied by editing code -- with no verified bound installed the payload numbers
    nothing, so the capture path never reaches a heart to make a claim about.

    TWO FRAMES, NAMED APART.  The re-attach probe moves the screen on purpose, and its return leg
    only ever promised a MEASURED net displacement driven back under half a read-scroll quantum
    -- never byte identity.  So the screen the SECOND burst was taken on is not always the screen
    the first burst was taken on.  `pre_probe_frame_sha256` names the frame the first burst is
    chained to; `action_frame_sha256` names the frame that is actually on the device when the
    heart is offered and tapped, and that is the one a checkpoint predicate must bind.
    `page_residual_px` is the measured displacement between the two (0 when the probe did come
    back byte-for-byte), and `action_frame` carries those exact bytes so the caller can re-prove
    and bind the screen that is really up.  The digests remain the binding authority; the bytes
    are there to be re-proved, never to be taken as evidence on their own.
    """
    action_frame_sha256: str
    pre_probe_frame_sha256: str
    dwell_frame_sha256s: tuple[str, ...]
    dwell_span_s: float
    still_photo_verified: bool
    # The re-attach probe's SECOND burst, named by digest exactly as the first is. Defaulted so
    # a proof constructed without one is refused at the binding check below rather than
    # published: a card that was never re-attached has not retired the stalled-video residual.
    reattach_frame_sha256s: tuple[str, ...] = ()
    reattach_dwell_span_s: float | None = None
    # The probe's own measured net displacement over its round trip, and the bytes it came back
    # to. Defaulted to "the probe put the page back exactly", which is the only state the
    # pre-2026-08-22 producer could publish at all.
    page_residual_px: int = 0
    action_frame: bytes = b""


def _verified_still_photo_evidence(proof: object, frame: bytes) -> bool:
    """Report the still-photo verdict for `frame`, refusing when nothing proved these bytes.

    `frame` is the ACTION frame -- the screen the heart is offered on and tapped on -- which is
    the probe's POST-RETURN frame whenever its return leg left a measured residual.  The first
    burst still has to chain to the pre-probe frame and the second to the action frame, because
    those are the screens each of them was actually measured on.
    """
    if not isinstance(proof, _StillPhotoProof) or proof.action_frame_sha256 != _sha256(frame):
        raise _CaptureAbort(
            "checkpoint refused: no still-photo (C1-C3) verdict is bound to the exact frame "
            "whose heart would be approved")
    if not proof.still_photo_verified:
        # The producer already refuses a negative verdict, so a False arriving here means a later
        # edit routed one through.  Refuse rather than publish it: a reviewer reading a predicate
        # list is reading claims, and a False still-photo claim beside an offered heart is how a
        # real video got one keystroke from a like on 2026-08-21.
        raise _CaptureAbort(
            "checkpoint refused: the still-photo acceptance did not pass on the exact action "
            "frame")
    if (len(proof.dwell_frame_sha256s) < 2
            or proof.dwell_frame_sha256s[0] != proof.pre_probe_frame_sha256):
        raise _CaptureAbort(
            "checkpoint refused: the un-interacted dwell is not chained to the exact frame "
            "whose heart would be approved")
    if (len(proof.reattach_frame_sha256s) < 2
            or proof.reattach_frame_sha256s[0] != proof.action_frame_sha256):
        # The second burst must name the screen it was actually taken on, and that screen is the
        # one this checkpoint is about. The probe scrolls the card out of Hinge's autoplay band
        # and back; where the return leg leaves a residual the producer MEASURES it, carries the
        # card rect across it and the caller re-proves the card there, so the action frame is the
        # post-return one. A second burst chained to anything else was measured on a screen this
        # checkpoint is not about -- the "bound to the wrong frame" case that would let a
        # re-attach observation of one screen vouch for another.
        raise _CaptureAbort(
            "checkpoint refused: no re-attach dwell burst is chained to the exact frame whose "
            "heart would be approved, so a video that was stalled or unloaded during the first "
            "dwell was never given a second chance to reveal itself")
    if proof.page_residual_px and proof.action_frame_sha256 == proof.pre_probe_frame_sha256:
        # A proof claiming the probe displaced the page while naming ONE frame for both bursts is
        # internally inconsistent -- corrupt, not silent -- so it cannot say which screen the
        # re-attach observation describes. Refuse rather than pick one of the two readings.
        raise _CaptureAbort(
            "checkpoint refused: the still-photo proof reports a displaced page yet names one "
            "frame for both dwell bursts, so which screen the re-attach burst describes is "
            "unknown")
    return proof.still_photo_verified


def _frame_height_px(frame: bytes) -> int | None:
    """The decoded pixel height of one frame, or None when it cannot be decoded.

    None is a REFUSAL upstream, never a default: the autoplay-centring precondition cannot be
    measured without knowing how tall the screen was, and a card whose position was never
    measured has not been shown to have been inside Hinge's trigger zone.
    """
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 -- an undecodable frame is a missing observation
        return None
    return None if image is None else int(image.shape[0])


def _frame_width_px(frame: bytes) -> int | None:
    """The decoded pixel width of one frame, or None when it cannot be decoded.

    Mirrors `_frame_height_px` exactly (see its docstring): None is a refusal upstream, never a
    default width like a hardcoded 1080 -- the device in hand is not the only one this ever runs
    against.
    """
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 -- an undecodable frame is a missing observation
        return None
    return None if image is None else int(image.shape[1])


def _content_band_rect_px(frame: bytes, content_band) -> tuple[int, int, int, int] | None:
    """The full-width pixel rect of `frame`'s scrolling content band, or None if undecodable.

    Android's status bar and Hinge's bottom nav are fixed chrome: they do not translate when the
    content scrolls, which is exactly why `estimate_shift` and `hinge.py`'s
    `_vertical_shift_match` both restrict their own comparisons to `content_band`'s rows rather
    than the whole frame (see either docstring for the measured numbers). The same fact makes
    `content_band` the correct scope for a byte-identity comparison too: it is blind to rows
    that change for reasons with no bearing on the card underneath, such as the status-bar clock
    ticking a minute forward, while still covering every row the tap target could possibly sit
    on or that could cover it.
    """
    height = _frame_height_px(frame)
    width = _frame_width_px(frame)
    if not height or not width:
        return None
    return (0, int(content_band[0] * height), width, int(content_band[1] * height))


def _reviewed_target_protected_prefix_rect(
        frame: bytes, content_band, reviewed_rows) -> tuple[int, int, int, int] | None:
    """Return the exact-match prefix that protects a reviewed heart from stale pixels.

    The full content band is deliberately too broad for this particular pre-tap guard: a later,
    separate card can autoplay below an already-reviewed still-photo target.  The target's
    coordinate cannot depend on pixels below its own bottom, but it *does* depend on every row
    through that bottom: profile header/name, every intervening row, the selected card, its
    heart, and any overlay or scroll that reaches them.  Keep that whole full-width prefix
    byte-exact, excluding only later cards below the selected one. A complete target card may
    begin a few pixels above the configured content-band start, so the prefix starts at the
    earlier of those two rows rather than refusing a valid frame-bounded card.

    ``reviewed_rows`` is the effective frame-row binding.  A reattach probe may have rebound the
    same card to translated ``expected_rows``; using the navigator's original rows there would
    accidentally leave part of the actual reviewed target unprotected.
    """
    band_rect = _content_band_rect_px(frame, content_band)
    if band_rect is None:
        return None
    if (not isinstance(reviewed_rows, (tuple, list)) or len(reviewed_rows) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in reviewed_rows)):
        raise ItemCropError("reviewed target rows must be two integer frame rows")
    target_y0, target_y1 = reviewed_rows
    _x0, band_y0, width, band_y1 = band_rect
    if not (0 <= target_y0 < target_y1 <= band_y1):
        raise ItemCropError(
            "reviewed target rows must be valid frame rows ending within the configured content band")
    return (0, min(band_y0, target_y0), width, target_y1)


def _identity_band_rect_px(frame: bytes, identity_band) -> tuple[int, int, int, int] | None:
    """Return the exact configured identity-band ROI for a decodable frame.

    The protected content prefix begins below this band on the current Hinge layout.  Keep the
    identity strip separately byte-exact before the semantic sticky-header comparison: a changed
    header must never be tolerated merely because an unrelated later card is allowed to animate.
    """
    height = _frame_height_px(frame)
    width = _frame_width_px(frame)
    if not height or not width:
        return None
    if (not isinstance(identity_band, (tuple, list)) or len(identity_band) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   for value in identity_band)):
        raise ItemCropError("identity band must be four numeric fractions")
    x0_frac, y0_frac, x1_frac, y1_frac = identity_band
    rect = (int(x0_frac * width), int(y0_frac * height),
            int(x1_frac * width), int(y1_frac * height))
    if not (0 <= rect[0] < rect[2] <= width and 0 <= rect[1] < rect[3] <= height):
        raise ItemCropError("configured identity band is outside the reviewed frame")
    return rect


def _dwell_centering(rect, frame: bytes, content_band):
    """`(centered, offset)` for one card rect, or `(None, None)` when it cannot be measured.

    Hinge autoplays a video only at or near the centre of the screen (owner fact 2026-08-21), so
    a byte-exact dwell taken off-centre is consistent with a video that was never asked to play.
    The zone itself lives in item_crops and is imported, never re-stated here.
    """
    height = _frame_height_px(frame)
    if not height or content_band is None:
        return None, None
    try:
        offset = card_center_offset_frac(tuple(rect), frame_height=height,
                                         content_band=content_band)
    except ItemCropError:
        return None, None
    return abs(offset) <= STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC, float(offset)


def _parked_card_center_offset(target, proof, *, frame: bytes, content_band,
                               rows=None) -> float | None:
    """Signed autoplay-zone offset of the card navigation just parked, or None if unmeasurable.

    Cheap on purpose.  It re-uses evidence already in hand: the frame rows are the ones
    `_verified_target_frame_proof` just re-proved by EXACT match on these bytes, and the x bounds
    are the block it matched them against, so asking "is this card near enough to the centre for
    Hinge to have been asked to play it" costs no gesture, no dwell and no second segmentation.

    None means NOT MEASURED, and is never read as centred.  A caller that cannot measure the
    position falls through to `_verified_still_photo_proof`, whose own centring rung refuses an
    unmeasured card rather than accepting it -- the same fail-closed answer, reached by the path
    that also holds the rest of the ladder.
    """
    # `rows` defaults to what the navigator parked and is passed explicitly when the card has
    # been carried across the still-photo probe's measured page residual: the position that
    # matters is where the card sits on the frame the heart will actually be tapped on.
    rows = getattr(target, "block_frame_rows", None) if rows is None else tuple(rows)
    block = getattr(proof, "block", None)
    height = _frame_height_px(frame)
    if rows is None or block is None or not height or content_band is None:
        return None
    try:
        return float(card_center_offset_frac(
            (block.x0, rows[0], block.x1, rows[1]), frame_height=height,
            content_band=content_band))
    except ItemCropError:
        return None


def _parked_signature_drift(frames, rect: tuple[int, int, int, int],
                            ) -> tuple[float | None, tuple[int, ...]]:
    """C1's re-observation drift over frames taken at ONE parked card position.

    The ladder's drift rung wants "how much did this crop's own pixels move when the same rows
    were looked at again", and in the pre-heart window the re-observations are right here: the
    anchor action frame and the burst frames taken after it, all of the same unmoved screen and
    therefore all describing the SAME rect with zero rect error.  It uses item_crops'
    `signature_of` and `CropSignature.distance` -- the module's own primitives, so this is the
    same measurement the payload builder makes and not a second implementation of it -- and
    aggregates the WORST pairwise distance against the anchor, exactly as `_signature_drift`
    aggregates its own samples.

    MEASURED, NEVER ASSUMED.  These frames are separately required to be byte-exact over the
    rect, so in practice the answer is 0.0; computing it anyway is what makes the drift rung
    still load-bearing if that byte-exactness requirement is ever weakened.  Fewer than two
    frames returns `(None, ())` -- ignorance, in the same shape `_signature_drift` reports it.
    """
    frames = list(frames)
    if len(frames) < 2:
        return None, ()
    x0, y0, x1, y1 = rect
    reference = signature_of(frames[0], y0=y0, y1=y1, x0=x0, x1=x1)
    worst: float | None = None
    sampled: list[int] = []
    for position, data in enumerate(frames[1:], start=1):
        distance = reference.distance(signature_of(data, y0=y0, y1=y1, x0=x0, x1=x1))
        worst = distance if worst is None else max(worst, distance)
        sampled.append(position)
    return worst, tuple(sampled)


def _cross_position_signature_drift(frame_a: bytes, rect_a: tuple[int, int, int, int],
                                    frame_b: bytes, rect_b: tuple[int, int, int, int],
                                    ) -> tuple[float | None, tuple[int, ...]]:
    """C1's re-observation drift across a MEASURED page residual, rather than at one position.

    `_parked_signature_drift` looks at ONE unmoved rect across several frames of the same parked
    screen. Here the still-photo probe's return leg left a measured residual, so the reviewed
    card genuinely sits at different rows on the pre-probe frame and the probe's post-return
    frame -- `rect_a`/`rect_b` are those two positions, already translated by the caller using
    the same measured residual. This is otherwise the identical measurement, over the module's
    own `signature_of` and `CropSignature.distance` -- no second implementation free to disagree
    with `_parked_signature_drift` or with `_signature_drift` about what "drift" means -- reduced
    to the one pairwise distance there is ever a reason to compute here.

    A still photo re-rasterised at its new position measures a small distance; a video that
    advanced while the probe ran measures a large one, and the ladder's 0.24 ceiling
    (`_STILL_PHOTO_MAX_SIGNATURE_DRIFT`) refuses it exactly as it refuses in-place motion --
    strictly STRONGER than the byte-identity demand this replaces, and it stays a MEASUREMENT,
    never an assumption. `(1,)` mirrors `_parked_signature_drift`'s own shape for a single
    comparison: one frame compared against a reference is "position 1".
    """
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b
    reference = signature_of(frame_a, y0=ay0, y1=ay1, x0=ax0, x1=ax1)
    other = signature_of(frame_b, y0=by0, y1=by1, x0=bx0, x1=bx1)
    return reference.distance(other), (1,)


def _verified_still_photo_proof(driver: HingeDriver, *, frame: bytes, block) -> _StillPhotoProof:
    """Re-run the acceptance on the action frame plus TWO no-input dwell bursts.

    Both bursts are the driver's own `_still_photo_dwell_burst`, not a local copy: the window is
    anchored on the installed bound and both knobs are hazard-randomized there, and a second
    implementation of the same dwell is a second thing that can drift from the artifact.

    THE FIRST burst issues no input at all, so it cannot disturb the reviewed frame it is chained
    to. THE SECOND is taken after the driver's re-attach probe, which deliberately DOES move the
    screen: it scrolls the card out of Hinge's autoplay band and back, because that re-entry is
    what makes the app attach and restart the media, and a video that was stalled, buffering,
    unloaded or already ended is otherwise indistinguishable from a photograph no matter how long
    the first burst watches.

    BYTE IDENTITY WAS NEVER THE DRIVER'S CONTRACT. `_still_photo_reattach_probe`'s return leg
    only ever promised a MEASURED net displacement driven back under about half a read-scroll
    quantum -- never that the page would land back on the exact byte. Refusing whenever it did
    not made depth-3 (and deeper) targets structurally lucky: whether a heart could ever be
    offered turned on a residual nobody was even trying to make zero. The probe still measures
    its own return leg, exactly as before; what changed is that a measured non-zero residual is
    now a SECOND position to re-prove the card at, not an automatic refusal. `_fresh_reviewed_target_point`
    is unaffected: it re-binds to whichever frame is actually current -- the pre-probe frame when
    the residual is zero, the probe's post-return frame otherwise -- immediately before the tap.
    (Found live 2026-08-22, attempt 7.)

    THE C1 RE-OBSERVATION FOR THE HEART DECISION IS THIS PARKED EVIDENCE, measured here by
    `_parked_signature_drift` over the frames this proof holds.  It used to be threaded in from
    the caller's payload crop, which measures something else entirely: the READ-SCROLL drift of
    the enumeration pass, taken while the page was moving under the card.  That number could not
    do this job in either direction.  For the first card of a top-down read it does not exist at
    all -- no other enumeration frame's analysed band contains that card's page rows, so
    `_signature_drift` structurally returns `(None, ())` and every depth-1 (odd-ordinal) target
    refused here forever.  And where it does exist it is routinely far above the 0.24 ceiling on
    real still photographs (0.277 to 2.1 measured), because the ceiling was measured on PARKED
    cards, which is exactly the population this function has and the enumeration loop does not.
    The payload crop's read-scroll drift remains useful capture-context diagnostics; it is not
    evidence about the frame whose heart is about to be approved.

    WHEN THE PROBE'S RETURN LEG LEAVES A MEASURED RESIDUAL, the C1 re-observation for the FINAL
    (post-probe) rung is two MEASURED positions of the SAME card rather than one parked position
    looked at repeatedly: the crop at the pre-probe frame's rect against the crop at the probe's
    post-return frame's translated rect. See `_cross_position_signature_drift`.

    Refuses rather than returning a negative proof, so a card the acceptance rejects is skipped
    exactly like a card the mute screen rejects, and the reviewer is never offered its heart.
    """
    rect = (block.x0, block.y0, block.x1, block.y1)
    content_band = getattr(driver, "content_band", None)

    def screened_clean(data: bytes) -> bool:
        return hinge_mod.video_mute_screen_reason(
            data, rect, match=getattr(driver, "_match_video_mute", None)) is None

    burst, span_s = driver._still_photo_dwell_burst()
    frames = [frame, *burst]
    digests = tuple(_sha256(data) for data in frames)
    try:
        exact = dwell_exact_over_rect(frames, rect)
        screened = all(screened_clean(data) for data in frames)
        drift, drift_frames = _parked_signature_drift(frames, rect)
    except ItemCropError as exc:
        raise _CaptureAbort(
            f"target still-photo proof refused on the exact action frame: {exc}") from exc
    centered, offset = _dwell_centering(rect, frame, content_band)
    first = StillPhotoDwell(dwell_frame_sha256s=digests, dwell_exact=exact, dwell_span_s=span_s,
                            mute_screens_complete=screened, centered=centered,
                            center_offset_frac=offset)
    # Ask the ladder BEFORE spending the probe's real gestures (two strokes, three when the
    # first exit direction measures clamped).  Every rung above the probe
    # answers from frames we already hold, so a card that fails one of them is refused for THAT
    # reason -- the policy blocker on an unlicensed build, the mute screen, the centring -- and
    # never for a probe it was never eligible for.  The comparison is by identity against the
    # exported constant, so this can never start reading prose.
    prelim = unnumber_without_still_photo_evidence(still_photo_evidence_from_drift(
        drift, drift_frames, first))
    if prelim != EXCLUSION_REATTACH_PROBE_MISSING:
        raise _CaptureAbort(
            "target still-photo proof refused on the exact action frame: "
            + (prelim if prelim is not None else
               "the acceptance passed a card no re-attach probe had looked at, which the "
               "ladder must never do"))
    probe = driver._still_photo_reattach_probe(frame, rect)
    if probe is None:
        raise _CaptureAbort(
            "target still-photo proof refused on the exact action frame: the re-attach probe "
            "could not take this card out of Hinge's autoplay band and bring it back, so a "
            "video that was not playing during the dwell was never asked to start")
    # `frameshift.estimate_shift`'s docstring, verbatim: "frame_a is the EARLIER frame ...
    # Positive delta_px means the content moved UP the screen ... content at row y of A is at
    # row y - delta_px of B."  F0 is `frame` (earlier); F1 is `probe.anchor` (later, the frame
    # the probe's return leg actually settled on).  So a card at `rect`'s rows on F0 sits at
    # (y0 - residual, y1 - residual) on F1.
    if probe.anchor == frame:
        # The byte-identical fast path.  Every proof this ladder has ever passed took exactly
        # this route, and it must keep doing so byte-for-byte: no extra measurement, no extra
        # screencap, no behaviour change at all.
        residual = 0
        rect2 = rect
        action_frame = frame
    else:
        residual = driver._measured_page_shift(frame, probe.anchor)
        if residual is None:
            raise _CaptureAbort(
                "target still-photo proof refused on the exact action frame: the probe's "
                "return leg displacement could not be measured, so the second burst cannot be "
                "bound to the reviewed card")
        rect2 = (rect[0], rect[1] - residual, rect[2], rect[3] - residual)
        action_frame = probe.anchor
        height2 = _frame_height_px(action_frame)
        if rect2[1] < 0 or height2 is None or rect2[3] > height2:
            raise _CaptureAbort(
                "target still-photo proof refused on the exact action frame: the probe's "
                f"return leg left a {residual}px residual that puts the reviewed card partly "
                "off screen, so it cannot be re-proved there")
    reattach_frames = [action_frame, *probe.frames]
    reattach_digests = tuple(_sha256(data) for data in reattach_frames)
    try:
        reattach_legs = still_photo_reattach_legs(
            reattach_frames, rect2, span_s=probe.span_s,
            mute_screen=lambda data, _rect: screened_clean(data),
            frame_height=_frame_height_px(action_frame), content_band=content_band)
        # THE FINAL C1 MEASUREMENT.  At residual == 0 every frame this proof holds describes the
        # SAME unmoved rect -- the anchor, the first burst, and the post-probe burst the probe
        # brought back to that same anchor byte-for-byte -- so the existing worst-of-N
        # re-observation over all of them is still exactly right and stays untouched.  Where the
        # probe's return leg left a residual, the card genuinely sits at two different rows on
        # two different frames, and the only honest re-observation is a measurement ACROSS that
        # move: see `_cross_position_signature_drift`.
        if residual:
            final_drift, final_drift_frames = _cross_position_signature_drift(
                frame, rect, action_frame, rect2)
        else:
            final_drift, final_drift_frames = _parked_signature_drift(
                [*frames, *probe.frames], rect)
    except ItemCropError as exc:
        raise _CaptureAbort(
            f"target still-photo proof refused on the exact action frame: {exc}") from exc
    refusal = unnumber_without_still_photo_evidence(still_photo_evidence_from_drift(
        final_drift, final_drift_frames, replace(first, **reattach_legs)))
    if refusal is not None:
        raise _CaptureAbort(
            "target still-photo proof refused on the exact action frame: " + refusal)
    return _StillPhotoProof(
        action_frame_sha256=_sha256(action_frame), pre_probe_frame_sha256=_sha256(frame),
        dwell_frame_sha256s=digests, dwell_span_s=span_s, still_photo_verified=True,
        reattach_frame_sha256s=reattach_digests, reattach_dwell_span_s=probe.span_s,
        page_residual_px=int(residual), action_frame=action_frame)


@dataclass(frozen=True)
class _VerificationBlockerProof:
    """What doc 5.6's post-tap verification blocker screen said about one payload item.

    Bound to the item it screened and to that item's heart ordinal rather than to a frame
    digest, because the blocker is a property of the PAYLOAD (an unseparable or undetermined
    crop reference), not of any single screen.  Binding it to a frame would be a more
    impressive-looking claim than the check can support, which is the failure mode this whole
    family of proof objects exists to prevent.
    """
    item_number: int
    heart_ordinal: object
    blocker: str
    blocker_absent: bool


def _verified_blocker_absence(payload, item_number: int) -> _VerificationBlockerProof:
    """Run doc 5.6's blocker screen for one item and carry its verdict."""
    blocker = verification_blocker(payload, item_number)
    return _VerificationBlockerProof(
        item_number=item_number, heart_ordinal=payload.item(item_number).heart_ordinal,
        blocker=blocker or "", blocker_absent=not blocker)


def _screened_verification_blocker_absent(proof: object, payload, item_number: int) -> bool:
    """Report the blocker verdict for this item, refusing when nothing screened it."""
    if (not isinstance(proof, _VerificationBlockerProof) or proof.item_number != item_number
            or proof.heart_ordinal != payload.item(item_number).heart_ordinal):
        raise _CaptureAbort(
            "checkpoint refused: no verification-blocker screen is bound to the exact payload "
            "item whose heart would be approved")
    if not proof.blocker_absent:
        raise _CaptureAbort(
            "checkpoint refused: the verification-blocker screen refused this item: "
            + proof.blocker)
    return proof.blocker_absent


def _fresh_reviewed_target_point(driver: HingeDriver, target, *, reviewed_point: object,
                                 content_band, like_template,
                                 like_threshold: float, expected_point=None,
                                 expected_frame=None, expected_rows=None) -> tuple[int, int]:
    """Re-bind a reviewed heart target to the screen at the instant before its tap.

    A hybrid review can take long enough for animated media or auto-hiding controls to change
    the framebuffer.  Coordinates approved for the earlier PNG are therefore not authority for
    a later screen.  Require byte identity over the protected CONTENT-BAND PREFIX through the
    reviewed target card, plus the exact configured identity band, then independently re-run the
    card/heart and profile identity gates on those fresh bytes.  No input is issued here; every
    refusal leaves the driver's final foreground-package guard as the only operation immediately
    before a valid tap.

    FULL-FRAME identity was never the right scope, and was found live to be unsatisfiable
    (2026-08-22, campaign attempt 8): Android's status bar redraws its clock every 60 seconds
    with no bearing whatsoever on the tap target, so any reviewer careful enough to actually
    read the PNG before approving it -- exactly the behaviour this checkpoint exists to reward
    -- was near-guaranteed to cross a minute boundary and have a genuinely valid heart refused.
    A diffed real refusal showed only rows 43-76 (the clock) differing; the content band, card
    and heart included, was byte-identical. The content band is the scrolling region and, by
    construction, excludes exactly this fixed chrome -- `estimate_shift`'s docstring and
    `hinge.py`'s `_vertical_shift_match` document the same status-bar/bottom-nav exclusion for
    the same reason. The reviewed heart point always sits inside this band, so scoping the
    comparison to the prefix through the target is strictly narrower, never weaker for the
    reviewed coordinate: it still catches a scroll, any target-card autoplay/mute/control change,
    or a modal over the target, while excluding only a separate later card below it. The exact
    identity-band comparison closes the layout gap above that prefix. Everything below the
    comparisons is UNCHANGED -- `_verified_target_frame_proof` and `compare_profile_identity`
    independently re-prove the card, the heart and the profile identity on the fresh frame --
    which is what makes narrowing the content scope safe rather than a relaxation.

    `expected_point`/`expected_frame`/`expected_rows` default to the navigator's own point, frame
    and rows -- exactly today's behaviour -- and are passed explicitly by the one caller that
    re-binds this SAME card to a DIFFERENT frame: the pre-heart loop, after the still-photo
    probe's return leg left a measured page residual and re-proved the card at the shifted rows.
    They are the navigator's own point and rows carried across that measurement (owner rule
    2026-08-11: never a re-identified card), never a relaxation of the byte-identity or
    structural/identity checks below.

    THE IDENTITY REFERENCE IS THE REVIEWED FRAME, NOT THE NAVIGATOR'S FRAME. `target.identity` is
    bound at the navigator's entry gate, against the profile's index-build capture -- a frame that
    can legitimately sit at a different scroll position than `frame_to_match` once the pre-heart
    loop has rebound this card to a POST-PROBE `expected_frame` after a measured page residual.
    Hinge's identity band shows different content at different scroll positions by design (the
    profile-independent filter-chips row at the very top, the sticky per-profile header once
    scrolled at all), so comparing a fresh frame against `target.identity`'s fingerprint instead
    of against `frame_to_match`'s own measures scroll offset, not identity. Found live this way
    (2026-08-22, campaign attempt 9): a fresh frame diffed byte-identical to the reviewed
    checkpoint on both the identity band and the content band was refused at 19.156 grey levels
    because the reference was the stale, differently-scrolled index-build frame. Fingerprinting
    `frame_to_match` directly puts both sides of the comparison at the SAME scroll position, so
    the distance measures identity the way it is supposed to. "Same profile across the probe" is
    established separately and does not depend on this: the probe's own MEASURED displacement,
    plus `_verified_target_frame_proof`'s structural re-proof of exactly one card at the
    translated rows with exactly one heart at the translated point.

    THE STICKY-BAND FINGERPRINT IS COMPLEMENTARY, NOT PRIMARY, AND ONLY RUNS WHEN IT CAN MEAN
    SOMETHING. It is taken and compared only when `confirm_scroll_top(frame_to_match, ...)`
    positively REFUTES top -- the one state where the band is showing this profile's real sticky
    header rather than Hinge's own chrome. When it does not refute (`confirmed` top, where the
    band is the profile-INDEPENDENT filter-chips row, or `cannot_tell`), this gate SKIPS the
    sticky-band comparison instead of aborting: the protected-prefix byte comparison above has
    already run and already passed by the time this code is reached, Hinge renders the profile's
    name header inside that prefix, and the separate exact identity-band ROI prevents a changed
    header from passing as lower-card animation. Found live this way
    (2026-08-22, attempt 9): the reviewed frame read `cannot_tell` at 8.078 grey levels -- inside
    the deliberate 3.0-9.0 dead zone, because the card parks just below top on this campaign --
    while the fresh frame was byte-identical to it across both the identity band and the content
    band; aborting there converted one refusal into another on exactly the frames this campaign
    produces.
    """
    point = target.point if expected_point is None else tuple(expected_point)
    rows = tuple(target.block_frame_rows if expected_rows is None else expected_rows)
    if (not isinstance(reviewed_point, list) or len(reviewed_point) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in reviewed_point)
            or tuple(reviewed_point) != point):
        raise _CaptureAbort(
            "hybrid heart refused: the approved action point no longer exactly binds the "
            "navigator's reviewed target")

    frame_to_match = target.frame if expected_frame is None else expected_frame
    fresh = driver.adb.screencap()
    try:
        protected_prefix_rect = _reviewed_target_protected_prefix_rect(
            frame_to_match, content_band, rows)
        identity_rect = _identity_band_rect_px(frame_to_match, driver.identity_band)
    except ItemCropError as exc:
        raise _CaptureAbort(
            f"hybrid heart refused: the reviewed target could not scope its protected "
            f"prefix/identity comparison: {exc}") from exc
    if protected_prefix_rect is None or identity_rect is None:
        raise _CaptureAbort(
            "hybrid heart refused: the reviewed frame could not be read to scope the protected "
            "prefix/identity comparison")
    try:
        prefix_unchanged = dwell_exact_over_rect([frame_to_match, fresh], protected_prefix_rect)
        identity_unchanged = dwell_exact_over_rect([frame_to_match, fresh], identity_rect)
    except ItemCropError as exc:
        raise _CaptureAbort(f"hybrid heart refused: the reviewed target's protected "
                            f"prefix/identity bands could not be compared: {exc}") from exc
    if not prefix_unchanged:
        raise _CaptureAbort(
            "hybrid heart refused: the reviewed target's protected content prefix changed after "
            "review; refusing to spend coordinates from a stale checkpoint")
    if not identity_unchanged:
        raise _CaptureAbort(
            "hybrid heart refused: the reviewed profile identity band changed after review; "
            "refusing to spend coordinates from a stale checkpoint")

    # The identity gate's reference is fingerprinted directly off `frame_to_match` -- the frame
    # the checkpoint was reviewed against, the SAME frame the protected-prefix comparison above just
    # used -- rather than off `target.identity`, which is bound to the profile's original
    # index-build capture and can legitimately sit at a different scroll position (see the
    # docstring). `confirm_scroll_top` decides whether this fingerprint is even worth taking:
    # `refuted` means the band is showing the per-profile sticky header (real identity signal,
    # worth comparing); `confirmed` means the band is Hinge's profile-INDEPENDENT filter-chips
    # row, and fingerprinting it would silently turn this gate into one that matches every
    # profile; `cannot_tell` means there is nothing usable to fingerprint at all. The non-refuted
    # cases are handled by SKIPPING the fingerprint below, not by aborting -- see the docstring's
    # "STICKY-BAND FINGERPRINT IS COMPLEMENTARY" paragraph for why that is safe rather than a
    # relaxation: the protected-prefix byte comparison above already proves the reviewed target
    # and in-content name region unchanged, while the separate exact identity-band ROI above
    # prevents a changed sticky header from passing as lower-card animation.
    try:
        reviewed_top = confirm_scroll_top(frame_to_match, identity_band=driver.identity_band)
    except ScrollTopError as exc:
        raise _CaptureAbort(
            "hybrid heart refused: the reviewed frame's profile identity could not be re-read, "
            f"so the tap cannot be proved to land on the reviewed profile: {exc}") from exc

    reviewed_identity = None
    if reviewed_top.refuted:
        try:
            reviewed_fingerprint = band_fingerprint(
                frame_to_match, identity_band=driver.identity_band, grid=_IDENTITY_GRID)
        except ScrollTopError as exc:
            raise _CaptureAbort(
                "hybrid heart refused: the reviewed frame's profile identity could not be "
                f"re-read, so the tap cannot be proved to land on the reviewed profile: {exc}"
            ) from exc
        reviewed_identity = ProfileIdentity(
            fingerprint=reviewed_fingerprint, band=tuple(driver.identity_band),
            grid=_IDENTITY_GRID, frame_index=None, scroll_top_distance=reviewed_top.distance,
            agreeing_frames=1,
            reason=(
                "fingerprinted directly from the reviewed checkpoint frame -- the frame the tap "
                "was actually approved against -- rather than from the profile's original "
                "index-build capture, so it is comparable to a fresh frame at the SAME scroll "
                "position; confirmed to refute scroll top at "
                f"{reviewed_top.distance:.3f} grey levels from the profile-independent "
                "filter-chips row"))
    # ELSE (`confirmed` top or `cannot_tell`): no fingerprint is taken and none is compared below.
    # This is a recorded skip, not a silent one -- it is reachable only after the content-band
    # byte comparison above has already passed, which is the proof that licenses it.

    try:
        proof = _verified_target_frame_proof(
            driver, target, frame=fresh, content_band=content_band,
            like_template=like_template, like_threshold=like_threshold,
            expected_rows=rows, expected_point=expected_point)
        if reviewed_identity is not None:
            fresh_identity = compare_profile_identity(
                fresh, reviewed_identity, identity_band=driver.identity_band,
                match_max_dist=target.identity.match_max)
            if not fresh_identity.matched:
                raise IdentityError(fresh_identity.reason)
    except (IdentityError, SegmentationError) as exc:
        raise _CaptureAbort(
            "hybrid heart refused: fresh structural/identity revalidation failed: "
            f"{type(exc).__name__}: {exc}") from exc
    return proof.block.hearts[0]


def _rewind_automated_profile_to_confirmed_top(driver: HingeDriver, *, ordinal: int,
                                                identity_band, content_band, like_template,
                                                like_threshold) -> bytes:
    """Return a visually-confirmed profile top for one automated/hybrid profile entry.

    This deliberately does *not* call ``driver._scroll_to_top``.  That method correctly
    unwinds scrolls made in this process, but its ledger is empty after an operator/AI handoff
    or a restarted calibration process.  Guessing a distance from that empty ledger was the
    source of a real mid-profile hybrid entry failure.

    The closed loop owns no raw input transport: each reverse stroke is planned from the exact
    current frame and issued through the driver's guarded, humanized ``_scroll_up_one`` path.
    It ordinarily continues only while the Hinge top detector *positively refutes* top.  One
    narrowly bounded exception covers a known post-like failure mode: an UNKNOWN filter-chip
    band may receive one recovery stroke only when the current frame independently proves an
    ordinary Hinge swipe deck through its visible Like and Pass controls.  That proof excludes
    compose sheets, dialogs, paywalls and arbitrary in-app surfaces.  The allowance is spent
    wherever the UNKNOWN is first met -- on the entry frame or on a settled post-gesture frame --
    because Hinge 10.2.0's taller header puts an ordinary mid-card profile one stroke below the
    sticky-header proof position squarely in scroll_top's deliberate 3..9 dead zone (measured
    live 2026-09-04: 7.859, confirmable one further stroke up), and every automated entry rewind
    starts from exactly that refuted proof position.  Once spent, a further UNKNOWN on a proven
    deck is an unresolved layout, never a reason to keep swiping: it raises
    ``_UnsupportedEntryDeck`` so the caller can advance the profile through the separately
    audited public action instead of banking a frame that is not a top anchor.  An UNKNOWN
    without that deck proof, an unreadable detector, an unchanged post-gesture frame, a planning
    refusal, or exhaustion of the explicit cap all still stop the run.
    """
    def settled_verdict(frame: bytes, *, stage: str) -> tuple[bytes, object]:
        """Re-read an unsettled scroll-top gate without touching the screen (see the constant)."""
        for probe in range(_MAX_UNSETTLED_TOP_REPROBES + 1):
            try:
                verdict = confirm_scroll_top(frame, identity_band=identity_band)
            except ScrollTopError as exc:
                raise _CaptureAbort(
                    f"automated profile {ordinal}: could not read the scroll-top gate {stage} "
                    f"hybrid rewind: {exc}") from exc
            if verdict.confirmed or verdict.refuted or probe >= _MAX_UNSETTLED_TOP_REPROBES:
                return frame, verdict
            # Humanized so the re-look is not a fixed-interval poll; `dwell_s` is absent on the
            # narrow fakes used by the offline rewind tests, which never reach a real screen.
            time.sleep(human_delay(getattr(driver, "dwell_s", 0.6)))
            frame = driver.adb.screencap()
        raise _CaptureAbort(f"automated profile {ordinal}: unreachable settle probe exhaustion")

    def unknown_deck_recovery_allowed(candidate: bytes) -> bool:
        """Return the positive, non-actionable deck fact needed for one UNKNOWN recovery.

        `_observe_deck_ready` is deliberately perception-only and requires both floating deck
        controls.  Do not substitute package foreground, a refuted top band, or a remembered
        profile here: all can describe a dialog/paywall/other Hinge surface on which an upward
        swipe would be speculative.
        """
        detector = getattr(driver, "_observe_deck_ready", None)
        if not callable(detector):
            return False
        try:
            return detector(candidate) is True
        except Exception:  # noqa: BLE001 -- an unproven deck never licenses input
            return False

    frame = driver.adb.screencap()
    min_spacing_px = None
    unknown_deck_recovery_spent = False
    for attempt in range(_MAX_AUTOMATED_TOP_REWIND_STEPS + 1):
        frame, verdict = settled_verdict(frame, stage="during")
        if verdict.confirmed:
            return frame
        if not verdict.refuted:
            if not unknown_deck_recovery_allowed(frame):
                raise _CaptureAbort(
                    f"automated profile {ordinal}: hybrid rewind refused an unconfirmed "
                    f"scroll-top state ({verdict.state}): {verdict.reason}")
            if unknown_deck_recovery_spent:
                # Symmetric with the post-gesture branch: the deck is proven, so this is an
                # advanceable but unusable layout, not a run-ending mystery.
                raise _UnsupportedEntryDeck(frame, state=verdict.state, reason=verdict.reason)
            # An UNKNOWN band cannot establish that this is merely a scrolled profile.  Spend
            # exactly one guarded upward stroke only after the independent, current-frame deck
            # proof above; a second UNKNOWN is an unresolved layout/state change, never a reason
            # to keep swiping.
            unknown_deck_recovery_spent = True
        if attempt >= _MAX_AUTOMATED_TOP_REWIND_STEPS:
            raise _CaptureAbort(
                f"automated profile {ordinal}: hybrid rewind exceeded its bounded "
                f"{_MAX_AUTOMATED_TOP_REWIND_STEPS}-gesture budget while top remained "
                # State, not a fixed phrase: the last iteration can now be the one that spends
                # the UNKNOWN allowance, so this is no longer always a positive refutation.
                f"unconfirmed ({verdict.state}: {verdict.reason})")
        try:
            step, min_spacing_px = _plan_card_scroll(
                frame, content_band=content_band, like_template=like_template,
                like_threshold=like_threshold, profile_min_spacing_px=min_spacing_px)
        except (SegmentationError, ScrollStepError) as exc:
            raise _CaptureAbort(
                f"automated profile {ordinal}: hybrid rewind could not plan a guarded "
                f"upward scroll from the current frame: {type(exc).__name__}: {exc}") from exc
        driver._scroll_up_one(step.frac, step.x_frac)
        after = driver.adb.screencap()
        # Confirm after every real input first.  A byte-identical frame that still refutes top
        # proves the gesture made no observable progress; do not keep touching a stuck screen.
        after, after_verdict = settled_verdict(after, stage="after a gesture during")
        if after_verdict.confirmed:
            return after
        if not after_verdict.refuted:
            if unknown_deck_recovery_allowed(after):
                if not unknown_deck_recovery_spent:
                    # The ordinary case, not an exotic deck: one stroke up from the position a
                    # terminal Pass proves its sticky header at lands in scroll_top's 3..9 dead
                    # zone on Hinge 10.2 (see the docstring). Hand this frame to the same single
                    # guarded allowance the entry frame gets rather than spending a real public
                    # Pass on a profile whose confirmable top is one more stroke away.
                    frame = after
                    continue
                # Hinge 10.2's "people close to ..." recommendation deck keeps both ordinary
                # controls visible but moves the filter-chip row out of the calibrated top band.
                # It is safe to advance through the public action guard, but this nearly blank
                # band is not a positive top anchor and must never enter calibration evidence.
                raise _UnsupportedEntryDeck(
                    after, state=after_verdict.state, reason=after_verdict.reason)
            raise _CaptureAbort(
                f"automated profile {ordinal}: hybrid rewind reached an unconfirmed "
                f"scroll-top state ({after_verdict.state}): {after_verdict.reason}")
        if after == frame:
            raise _CaptureAbort(
                f"automated profile {ordinal}: hybrid rewind stalled after guarded gesture "
                f"{attempt + 1}; top remains positively refuted ({after_verdict.reason})")
        frame = after

    # The range/cap branch above is exhaustive.  Keep a fail-closed guard if this loop is
    # refactored so no caller can ever mistake an unconfirmed frame for a valid entry anchor.
    raise _CaptureAbort(f"automated profile {ordinal}: hybrid rewind ended without a confirmed top")


def _skip_automated_unsupported_entry_deck(
        driver: HingeDriver, *, ordinal: int, entry: _UnsupportedEntryDeck,
        identity_band, review_gate: _HybridReviewGate | None = None,
        send_like: bool = False, skip_reason: _PreActionProfileRetry | None = None) -> dict:
    """Advance a proven deck whose layout cannot supply a calibration top anchor.

    Unlike a normal pre-action skip, this trace makes no identity or top claim. Permission to
    act comes from fresh ordinary-deck controls plus composer absence; the public driver action
    supplies its own vision target and progress verification. The next retry must independently
    rewind and prove a normal top before saving any evidence.

    ``skip_reason`` is set when this escalation replaced a pre-action skip whose own rewind hit
    the unsupported layout. Carrying that original diagnosis into ``reason_detail`` keeps it in
    ``reason_sha256`` (which binds code+detail), so the operator still learns why the profile was
    being skipped at all instead of seeing only the layout symptom.
    """
    frame = entry.frame
    try:
        if driver._observe_deck_ready(frame) is not True:
            raise _CaptureAbort(
                f"automated profile {ordinal}: unsupported entry no longer proves an ordinary deck")
    except _CaptureAbort:
        raise
    except Exception as exc:  # noqa: BLE001 -- failed perception cannot license input
        raise _CaptureAbort(
            f"automated profile {ordinal}: unsupported-entry deck proof failed: "
            f"{type(exc).__name__}: {exc}") from exc
    try:
        locate_inline_composer(frame, driver._template("confirm"), threshold=0.8)
    except ComposerDetectionError:
        pass
    else:
        raise _CaptureAbort(
            f"automated profile {ordinal}: refusing unsupported-entry advance while an inline "
            "composer is structurally present")

    action_name = ("advance_unusable_profile_with_priority_like" if send_like
                   else "skip_profile_without_heart")
    public_transport = ("HingeDriver.like" if send_like else "HingeDriver.dislike")
    reason_code = "unsupported_entry_layout"
    reason_detail = (
        f"ordinary deck controls were proved, but schema-v3 could not positively confirm the "
        f"entry top ({entry.state}: {entry.reason})")
    if skip_reason is not None:
        reason_detail += (
            f"; escalated from a pre-action skip for {skip_reason.code}: {skip_reason.detail}")
    review = None
    if review_gate is not None:
        review = review_gate.checkpoint(
            frame, claimed_state="pre_action_unsupported_entry_deck_ready",
            action_plan={
                "action": action_name, "photo_model_item": None, "point": None,
                "point_source": f"public {public_transport}",
                "predicates": {
                    "ordinary_deck_ready": True, "inline_composer_absent": True,
                    "calibration_top_unconfirmed": True,
                    "skip_reason_code": reason_code, "skip_reason_detail": reason_detail,
                    "forbidden_zone_guarded_transport": public_transport,
                    **({"send_like_requested_for_unusable_profile": True} if send_like else {
                        "no_photo_heart_or_send_like_on_current_profile": True}),
                },
            })
        if review["decision"] != "approved":
            raise _CaptureAbort(
                "hybrid reviewer did not approve the unsupported-entry profile skip; refusing "
                "to advance this profile")

    try:
        if send_like:
            driver.like()
        else:
            driver.dislike()
    except Exception as exc:  # noqa: BLE001 -- preserve public action's fail-closed refusal
        raise _CaptureAbort(
            f"automated profile {ordinal}: public {public_transport} refused unsupported-entry "
            f"advance: {type(exc).__name__}: {exc}") from exc
    time.sleep(human_delay(driver.dwell_s))
    post = driver.adb.screencap()
    if post == frame:
        raise _CaptureAbort(
            "automated unsupported-entry skip did not produce a changed deck frame")

    # A consecutive unsupported layout is valid retry input but still not evidence. Accept it
    # only as an ordinary, composer-free deck and let the bounded outer skip budget decide
    # whether another public advance is permitted. One Hinge+ modal recovery is retained.
    initial_post = post
    modal_edge_back_used = False
    try:
        locate_inline_composer(post, driver._template("confirm"), threshold=0.8)
    except ComposerDetectionError:
        pass
    else:
        raise _CaptureAbort("automated unsupported-entry skip left an inline composer visible")
    if driver._observe_deck_ready(post) is not True:
        screen_width, screen_height = driver.adb.screen_size()
        if (not isinstance(screen_width, int) or screen_width <= 0
                or not isinstance(screen_height, int) or screen_height <= 0):
            raise _CaptureAbort(
                "automated unsupported-entry skip cannot recover modal: invalid screen size")
        driver._swipe(round(screen_width * _EDGE_BACK_START_X_FRAC),
                      round(screen_height * _EDGE_BACK_Y_FRAC),
                      round(screen_width * _EDGE_BACK_END_X_FRAC),
                      round(screen_height * _EDGE_BACK_Y_FRAC))
        modal_edge_back_used = True
        time.sleep(human_delay(driver.dwell_s))
        post = driver.adb.screencap()
        try:
            locate_inline_composer(post, driver._template("confirm"), threshold=0.8)
        except ComposerDetectionError:
            pass
        else:
            raise _CaptureAbort(
                "automated unsupported-entry modal recovery left an inline composer visible")
        if driver._observe_deck_ready(post) is not True:
            raise _CaptureAbort(
                "automated unsupported-entry modal recovery did not reach an ordinary deck")
    try:
        post_top = confirm_scroll_top(post, identity_band=identity_band)
    except ScrollTopError as exc:
        raise _CaptureAbort(
            f"automated unsupported-entry skip cannot read the next deck top state: {exc}") from exc

    reason_digest = _sha256(f"{reason_code}\n{reason_detail}".encode("utf-8"))
    return {
        "action": action_name,
        "ordinal": ordinal,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "reason_sha256": reason_digest,
        "transport": public_transport,
        "pre_frame_sha256": _sha256(frame),
        "post_frame_sha256": _sha256(post),
        "post_deck_frame_sha256": _sha256(post),
        "pre_scroll_top_state": entry.state,
        "post_scroll_top_state": post_top.state,
        "post_scroll_top_reason": post_top.reason,
        "post_pass_settle": {
            "modal_edge_back_used": modal_edge_back_used,
            "ordinary_deck_ready": True,
            "initial_post_pass_frame_sha256": _sha256(initial_post),
            "settled_post_pass_frame_sha256": _sha256(post),
        },
        "review_checkpoints": ({"before": review} if review_gate is not None else None),
        "predicates": {
            "pre_action_deck_ready": True,
            "pre_action_composer_absent": True,
            "calibration_top_unconfirmed": True,
            **({
                "send_like_requested_for_unusable_profile": True,
                "send_like_tapped": True,
                "public_action_guard_used": True,
            } if send_like else {
                "no_photo_heart_or_send_like_on_current_profile": True,
                "public_dislike_guard_used": True,
            }),
            "public_action_progress_verified": True,
            "deck_frame_changed": True,
            "new_deck_ready": True,
            "new_profile_composer_absent": True,
            "new_profile_top_confirmed": post_top.confirmed,
            "new_profile_identity_distinct": False,
            "identity_comparison_not_claimed": True,
            "modal_edge_back_used": modal_edge_back_used,
        },
    }


def _scroll_next_profile_to_sticky_header(
        driver: HingeDriver, *, top_frame: bytes, identity_band, content_band,
        like_template, like_threshold: float, context: str) -> bytes:
    """Expose a post-advance sticky header with at most two guarded read scrolls.

    The caller already proved ``top_frame`` is a fresh ordinary deck. Each gesture is sized by
    the same local-card planner as the calibration read. If the first resting frame is still
    confirmed-top or indeterminate, a second gesture is allowed only while the current frame
    independently retains both ordinary deck controls. No identity threshold is widened and an
    unresolved second frame remains a hard refusal.
    """
    frame = top_frame
    min_spacing_px = None
    verdict = None
    for attempt in range(_MAX_POST_ADVANCE_STICKY_SCROLLS):
        if attempt:
            deck_ready = getattr(driver, "_observe_deck_ready", None)
            if not callable(deck_ready) or deck_ready(frame) is not True:
                raise _CaptureAbort(
                    f"{context}: the first identity-proof scroll did not expose a sticky name "
                    "header and the ordinary deck could not be re-proved before a second scroll")
        try:
            step, min_spacing_px = _plan_card_scroll(
                frame, content_band=content_band, like_template=like_template,
                like_threshold=like_threshold, profile_min_spacing_px=min_spacing_px)
        except (SegmentationError, ScrollStepError) as exc:
            raise _CaptureAbort(
                f"{context}: cannot plan guarded post-advance identity scroll "
                f"{attempt + 1}: {type(exc).__name__}: {exc}") from exc
        driver._scroll_down_one(step.frac, step.x_frac)
        # SETTLED, not the first frame back (restored 2026-09-04). `adb shell input swipe` returns
        # when the finger path ends while Hinge is still snapping the new profile, and
        # `_settled_read_scroll_frame`'s own docstring records the measurement: two immediate
        # post-gesture frames moved another 554px and 559px on the 2026-08-26 held-out run. An
        # unsettled frame here still shows the filter-chip row, so `confirm_scroll_top` does not
        # refute it and this loop answers a mid-animation frame by spending ANOTHER real device
        # gesture -- and the frame it finally returns is consumed directly as identity evidence
        # by the caller. This is the read the pre-action-skip path used for exactly this reason.
        frame = _settled_read_scroll_frame(driver)
        try:
            verdict = confirm_scroll_top(frame, identity_band=identity_band)
        except ScrollTopError as exc:
            raise _CaptureAbort(
                f"{context}: cannot read the post-advance sticky identity after scroll "
                f"{attempt + 1}: {exc}") from exc
        if verdict.refuted:
            return frame
    assert verdict is not None
    raise _CaptureAbort(
        f"{context}: sticky name header was not proven after the bounded "
        f"{_MAX_POST_ADVANCE_STICKY_SCROLLS}-scroll probe; final state "
        f"{verdict.state}: {verdict.reason}")


def _prove_post_advance_identity_distinct(
        driver: HingeDriver, *, prior_fingerprint, initial_frame: bytes, identity_band,
        content_band, context: str) -> tuple[bytes, float, dict]:
    """Return a distinct sticky-header frame, repairing only a proven late repaint.

    ``_scroll_next_profile_to_sticky_header`` has already spent its bounded gesture budget and
    returned a positively refuted (sticky) frame.  A collision at the fixed identity threshold
    is nevertheless ambiguous: it can be a same profile, a true fingerprint collision, or a
    late header repaint.  The latter is the only recoverable case, and it receives at most three
    read-only screencaps.  Acceptance requires a final consecutive pair which is sticky,
    stationary, distinct from the prior identity, and mutually agreeing.  No retry changes either
    threshold or invokes a transport method.
    """
    prior_fingerprint_sha256 = _sha256(repr(prior_fingerprint).encode("utf-8", "replace"))

    def fingerprint_and_distance(frame: bytes, *, stage: str) -> tuple[object | None, float | None, str | None]:
        try:
            fingerprint = band_fingerprint(
                frame, identity_band=identity_band, grid=_IDENTITY_GRID)
            distance = _checked_distance(
                fingerprint_distance(prior_fingerprint, fingerprint),
                context=f"{context} {stage} identity")
        except (ScrollTopError, _MeasureRefused) as exc:
            # The eventual refusal names only the exception class: detector prose can describe a
            # real profile surface, while the hashes and numeric measurements remain sufficient
            # to correlate the private local frames.
            return None, None, type(exc).__name__
        return fingerprint, distance, None

    def stationary(left: bytes, right: bytes) -> tuple[bool, float | None, str]:
        if left == right:
            return True, 0.0, "byte_identical"
        try:
            shift = estimate_shift(left, right, content_band=content_band)
        except ShiftEstimationError as exc:
            return False, None, type(exc).__name__
        delta = getattr(shift, "delta_px", None)
        if (isinstance(delta, bool) or not isinstance(delta, (int, float))
                or not math.isfinite(delta)):
            return False, None, "invalid_shift"
        return (bool(shift.ok) and abs(delta) <= _READ_SCROLL_SETTLE_MAX_SHIFT_PX,
                float(delta), "stationary" if bool(shift.ok) else "shift_unavailable")

    initial_fp, initial_distance, initial_error = fingerprint_and_distance(
        initial_frame, stage="initial sticky")
    initial_sample = {
        "stage": "initial_sticky",
        "frame_sha256": _sha256(initial_frame),
        "fingerprint_sha256": (_sha256(repr(initial_fp).encode("utf-8", "replace"))
                               if initial_fp is not None else None),
        "distance_from_prior": initial_distance,
        "measurement_error": initial_error,
    }
    trace = {
        "read_only_reprobe_count": 0,
        "prior_identity_fingerprint_sha256": prior_fingerprint_sha256,
        "identity_distance_samples": [initial_sample],
    }
    if initial_error is not None:
        raise _CaptureAbort(
            f"{context}: cannot measure initial post-advance sticky identity "
            f"({initial_error}); no gesture was issued and identity thresholds unchanged; "
            f"prior_fingerprint_sha256={prior_fingerprint_sha256}; "
            f"initial_frame_sha256={initial_sample['frame_sha256']}")
    assert initial_distance is not None
    if initial_distance > _IDENTITY_FALSE_MATCH_DISTANCE:
        return initial_frame, initial_distance, trace

    previous_frame = initial_frame
    reprobe_samples: list[dict] = []
    # Keep raw fingerprints only until this helper returns.  The trace stores hashes, never the
    # profile-derived vectors themselves.
    reprobe_fingerprints: list[object | None] = []
    for probe in range(1, _MAX_POST_ADVANCE_IDENTITY_REPROBES + 1):
        # Keep timing humanized and bounded by the fixed probe count.  This read-only wait can
        # observe a compositor repaint but can never turn an ambiguous screen into permission
        # for another gesture.
        time.sleep(human_delay(getattr(driver, "dwell_s", 0.6)))
        frame = driver.adb.screencap()
        try:
            top = confirm_scroll_top(frame, identity_band=identity_band)
        except ScrollTopError as exc:
            top_state, top_refuted, top_error = "unreadable", False, type(exc).__name__
        else:
            top_state, top_refuted, top_error = top.state, bool(top.refuted), None
        fingerprint, distance, measurement_error = fingerprint_and_distance(
            frame, stage=f"read-only reprobe {probe}")
        is_stationary, shift_px, shift_state = stationary(previous_frame, frame)
        sample = {
            "stage": f"read_only_reprobe_{probe}",
            "frame_sha256": _sha256(frame),
            "fingerprint_sha256": (_sha256(repr(fingerprint).encode("utf-8", "replace"))
                                   if fingerprint is not None else None),
            "distance_from_prior": distance,
            "measurement_error": measurement_error,
            "scroll_top_state": top_state,
            "scroll_top_refuted": top_refuted,
            "scroll_top_error": top_error,
            "content_shift_from_previous_px": shift_px,
            "content_stationary_from_previous": is_stationary,
            "content_shift_state": shift_state,
        }
        reprobe_samples.append(sample)
        reprobe_fingerprints.append(fingerprint)
        trace["identity_distance_samples"].append(sample)
        trace["read_only_reprobe_count"] = probe

        # A first read can be transitional.  The accepting pair is always the last two reads
        # seen so far, not the low-distance initial sticky frame and one later repaint.
        if len(reprobe_samples) >= 2:
            left, right = reprobe_samples[-2:]
            left_raw, right_raw = reprobe_fingerprints[-2:]
            pair_distance = None
            if left_raw is not None and right_raw is not None:
                try:
                    pair_distance = _checked_distance(
                        fingerprint_distance(left_raw, right_raw),
                        context=f"{context} consecutive read-only reprobe agreement")
                except _MeasureRefused:
                    pair_distance = None
            trace["last_consecutive_reprobe_distance"] = pair_distance
            pair_is_safe = (
                bool(left.get("scroll_top_refuted"))
                and bool(right.get("scroll_top_refuted"))
                and left.get("distance_from_prior") is not None
                and right.get("distance_from_prior") is not None
                and left["distance_from_prior"] > _IDENTITY_FALSE_MATCH_DISTANCE
                and right["distance_from_prior"] > _IDENTITY_FALSE_MATCH_DISTANCE
                # `right` measured its content shift from `left`, exactly the pair whose
                # fingerprints are being compared.  A transition into the first member is not
                # smuggled into this condition.
                and bool(right.get("content_stationary_from_previous"))
                and pair_distance is not None
                and pair_distance <= _UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST)
            if pair_is_safe:
                trace["accepted_read_only_reprobe_pair"] = [probe - 1, probe]
                return frame, float(right["distance_from_prior"]), trace
        previous_frame = frame

    sample_summary = "; ".join(
        "{stage}(sha256={frame_sha256},distance={distance_from_prior},top={scroll_top_state},"
        "refuted={scroll_top_refuted},shift={content_shift_from_previous_px})".format(
            **sample) for sample in reprobe_samples)
    raise _CaptureAbort(
        f"{context}: post-advance sticky identity reprobe refused after "
        f"{_MAX_POST_ADVANCE_IDENTITY_REPROBES} read-only screencaps; no gesture was issued and "
        f"identity thresholds unchanged (distinct>{_IDENTITY_FALSE_MATCH_DISTANCE:.3f}, "
        f"consecutive_agreement<={_UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST:.3f}); "
        f"prior_fingerprint_sha256={prior_fingerprint_sha256}; "
        f"initial(sha256={initial_sample['frame_sha256']},distance={initial_distance:.3f}); "
        f"reprobes=[{sample_summary}]")


def _skip_automated_profile_before_heart(
        driver: HingeDriver, *, ordinal: int, reason: _PreActionProfileRetry,
        identity: ProfileIdentity, identity_band, content_band, like_template, like_threshold,
        review_gate: _HybridReviewGate | None = None, send_like: bool = False) -> dict:
    """Advance one unusable profile without pretending it was calibration evidence.

    The default remains the historical public ``HingeDriver.dislike`` transport. An explicitly
    accepted ``--send-like`` run instead uses public ``HingeDriver.like()`` with no opener and no
    target index: the owner's rule is that every encountered profile advances by a real Like,
    even when the profile is unusable as calibration evidence. It never invents a target,
    opener, or measurement. Both branches start from a confirmed ordinary deck top, retain the
    driver's production action guard/landed proof, and record the result only in
    ``skipped_attempts`` so measurement cannot count a bad target as evidence.
    """
    if (not identity.known or identity.fingerprint is None):
        raise _CaptureAbort(
            f"automated profile {ordinal}: cannot skip {reason.code} without a corroborated "
            "current-profile identity")
    top_frame = _rewind_automated_profile_to_confirmed_top(
        driver, ordinal=ordinal, identity_band=identity_band, content_band=content_band,
        like_template=like_template, like_threshold=like_threshold)
    try:
        locate_inline_composer(top_frame, driver._template("confirm"), threshold=0.8)
    except ComposerDetectionError:
        pass
    else:
        raise _CaptureAbort(
            f"automated profile {ordinal}: refusing pre-action skip while an inline composer "
            "is structurally present")

    review = None
    action_name = ("advance_unusable_profile_with_priority_like" if send_like
                   else "skip_profile_without_heart")
    public_transport = ("HingeDriver.like" if send_like else "HingeDriver.dislike")
    pre_action_predicates = {
        "confirmed_profile_top": True,
        "inline_composer_absent": True,
        "skip_reason_code": reason.code,
        "skip_reason_detail": reason.detail,
        "forbidden_zone_guarded_transport": public_transport,
    }
    if send_like:
        pre_action_predicates.update({
            "no_profile_action_sent_yet": True,
            "send_like_requested_for_unusable_profile": True,
        })
    else:
        # Preserve the established default-Pass manifest/checkpoint shape byte-for-byte in
        # semantics; the new owner rule is opt-in only through --send-like.
        pre_action_predicates["no_photo_heart_or_send_like_on_current_profile"] = True
    if review_gate is not None:
        review = review_gate.checkpoint(
            top_frame, claimed_state="pre_action_profile_skip_ready",
            action_plan={
                "action": action_name, "photo_model_item": None,
                "point": None, "point_source": f"public {public_transport}",
                # The reviewer is approving a real action on a real person. The diagnosis and
                # exact transport are inside the checkpoint hash, never inferred afterward.
                "predicates": pre_action_predicates,
            })
        if review["decision"] != "approved":
            raise _CaptureAbort(
                "hybrid reviewer did not approve the pre-action profile skip; refusing to "
                "advance this profile")

    # These are deliberately public APIs, rather than `_await_button`/`_tap`: they retain the
    # ordinary deck/action preflights and landed verification, so a composer, paywall, or other
    # unexpected surface refuses instead of licensing a private-coordinate fallback.
    try:
        if send_like:
            driver.like()
        else:
            driver.dislike()
    except Exception as exc:  # noqa: BLE001 -- preserve the public guard's fail-closed refusal
        raise _CaptureAbort(
            f"automated profile {ordinal}: public {public_transport} refused pre-action advance: "
            f"{type(exc).__name__}: {exc}") from exc
    time.sleep(human_delay(driver.dwell_s))
    raw_advanced_top = driver.adb.screencap()
    if raw_advanced_top == top_frame:
        raise _CaptureAbort("automated pre-action profile skip did not produce a changed deck frame")
    advanced_top, settle_trace = _settle_automated_post_pass_to_top(
        driver, frame=raw_advanced_top, confirm_template=driver._template("confirm"),
        identity_band=identity_band)

    # A top frame proves the ordinary deck returned. A tightly bounded planner-approved read is
    # then required to expose the sticky identity and prove this is not merely the same card
    # redrawn. The next attempt starts with its own visual rewind, so this does not leak a
    # process-local scroll estimate into the retry.
    advanced_identity = _scroll_next_profile_to_sticky_header(
        driver, top_frame=advanced_top, identity_band=identity_band,
        content_band=content_band, like_template=like_template,
        like_threshold=like_threshold,
        context=f"automated profile {ordinal} pre-action skip")
    advanced_identity, distance, sticky_header_trace = _prove_post_advance_identity_distinct(
        driver, prior_fingerprint=identity.fingerprint, initial_frame=advanced_identity,
        identity_band=identity_band, content_band=content_band,
        context=f"automated profile {ordinal} pre-action skip")

    # `reason_sha256` binds code+detail, so carrying the plaintext detail beside it makes this
    # record SELF-VERIFYING rather than weaker: any reader can recompute the digest.  Publishing
    # it is the point -- a diagnosis that exists only as a hash is invisible to the operator, and
    # a live run produced two consecutive skips that looked identical because only the generic
    # code was readable, leaving nothing to act on.
    reason_digest = _sha256(f"{reason.code}\n{reason.detail}".encode("utf-8"))
    return {
        "action": action_name,
        "ordinal": ordinal,
        "reason_code": reason.code,
        "reason_detail": reason.detail,
        "reason_sha256": reason_digest,
        "transport": public_transport,
        "pre_frame_sha256": _sha256(top_frame),
        "post_frame_sha256": _sha256(advanced_top),
        "post_identity_frame_sha256": _sha256(advanced_identity),
        "new_profile_identity_distance": distance,
        "automated_sticky_header_proof": sticky_header_trace,
        "post_pass_settle": settle_trace,
        "review_checkpoints": ({"before": review} if review_gate is not None else None),
        "predicates": {
            "pre_action_confirmed_top": True,
            "pre_action_composer_absent": True,
            **({
                "send_like_requested_for_unusable_profile": True,
                "send_like_tapped": True,
                "public_action_guard_used": True,
            } if send_like else {
                "no_photo_heart_or_send_like_on_current_profile": True,
                "public_dislike_guard_used": True,
            }),
            "deck_frame_changed": True,
            "new_profile_top_confirmed": True,
            "new_profile_composer_absent": True,
            "new_profile_identity_distinct": True,
            "modal_edge_back_used": settle_trace["modal_edge_back_used"],
        },
    }


def _verified_automated_composer(frame: bytes, *, confirm_template, payload: ItemPayload,
                                 item_number: int):
    """Return the only inline composer surface automation may clear with Pass.

    Unlike a production decision, this is deliberately scoped to calibration capture after a
    heart has already been recorded.  It proves both the measured inline layout and the selected
    photo again on every frame immediately preceding an automated transition.  A visible label
    or a remembered earlier composer is never enough permission to use the calibration-only
    edge-back/Pass path.
    """
    try:
        surface = locate_inline_composer(frame, confirm_template, threshold=0.8)
        if surface.layout_id != _COMPOSER_LAYOUT_ID:
            raise ComposerDetectionError(f"unsupported layout {surface.layout_id!r}")
        verdict = verify_sheet_item(frame, payload, item_number, composer_surface=surface)
        if not verdict.matched:
            raise SheetVerificationError(verdict.reason)
    except (ComposerDetectionError, SheetVerificationError) as exc:
        raise _CaptureAbort(
            f"automated Pass refused: current frame does not structurally prove the prior "
            f"inline composer for photo item {item_number}: {exc}") from exc
    return surface


def _only_transient_empty_comment_caret_change(reviewed_frame: bytes, fresh_frame: bytes, *,
                                                surface) -> bool:
    """Allow the one harmless framebuffer race a focused, empty composer creates.

    A Hinge screencap taken after a human hybrid-review approval can differ only because the
    Android text caret blinked.  This is not a general frame-diff tolerance: the fresh frame has
    already independently re-proved the selected item and every composer control.  Here we
    merely ensure its *remaining* changed pixels are one narrow, left-inset vertical strip within
    the comment input.  In particular, changes to Gboard, the profile card, header, CTA, or
    input border always return False.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover - capture preflight normally establishes this dependency
        return False
    reviewed = cv2.imdecode(np.frombuffer(reviewed_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    fresh = cv2.imdecode(np.frombuffer(fresh_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if reviewed is None or fresh is None or reviewed.shape != fresh.shape:
        return False
    changed = np.any(reviewed != fresh, axis=2)
    # Android's read-only status bar clock/battery/network glyphs are outside Hinge's app
    # surface and can legitimately tick during hybrid review.  Ignore only that fixed top strip;
    # every Hinge, composer, keyboard, and navigation pixel remains in the comparison.
    status_bar_bottom = round(changed.shape[0] * _SYSTEM_STATUS_BAR_HEIGHT_FRAC)
    changed[:status_bar_bottom, :] = False
    ys, xs = np.nonzero(changed)
    if not len(xs):
        # PNG encoding metadata may differ although the pixels and the re-proven controls do not.
        return True

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    comment = surface.comment_rect
    # Measured on the exact schema-v3 Pixel 7a composer: the caret is a 5x49px connected stroke
    # at (+42,+32) from the comment rectangle.  Keep only small rasterisation/layout tolerance;
    # the earlier whole-left-inset allowance could also admit a second narrow mark beside it.
    width, height = x1 - x0, y1 - y0
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8), connectivity=8)
    if component_count != 2:  # background + exactly one changed component
        return False
    component_area = int(stats[1, cv2.CC_STAT_AREA])
    location_and_shape_match = (
        comment.x0 + 39 <= x0 <= comment.x0 + 45
        and comment.x0 + 44 <= x1 <= comment.x0 + 51
        and comment.y0 + 28 <= y0 <= comment.y0 + 60
        and 3 <= width <= 8
        and 45 <= height <= 60
        and component_area >= round(width * height * 0.75)
    )
    if not location_and_shape_match:
        return False

    # A blink toggles between Hinge's dark caret and the light empty-input background.  Requiring
    # that polarity rejects small same-tone animations even at the measured location.
    reviewed_gray = cv2.cvtColor(reviewed[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    fresh_gray = cv2.cvtColor(fresh[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    dark_to_light = (
        np.mean(reviewed_gray <= 90) >= 0.75 and np.mean(fresh_gray >= 110) >= 0.75)
    light_to_dark = (
        np.mean(fresh_gray <= 90) >= 0.75 and np.mean(reviewed_gray >= 110) >= 0.75)
    return bool(dark_to_light or light_to_dark)


def _same_actionable_pixels(first_frame: bytes, second_frame: bytes) -> bool:
    """Compare every non-status-bar pixel, ignoring harmless PNG encoding metadata."""
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover - capture preflight normally establishes this dependency
        return False
    first = cv2.imdecode(np.frombuffer(first_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    second = cv2.imdecode(np.frombuffer(second_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None or second is None or first.shape != second.shape:
        return False
    status_bar_bottom = round(first.shape[0] * _SYSTEM_STATUS_BAR_HEIGHT_FRAC)
    return bool(np.array_equal(first[status_bar_bottom:], second[status_bar_bottom:]))


def _require_same_profile_header_after_edge_back(frame: bytes, *, identity: ProfileIdentity,
                                                  identity_band) -> float:
    """Prove that an edge-back left us on the indexed profile, now with its sticky header.

    On current Hinge, dismissing an auto-focused inline composer may restore the profile's
    previous scroll position.  That can legitimately move the composer out of the viewport, so
    its visibility is not a safe post-edge predicate.  The sticky identity band is instead the
    measured invariant: it must be positively non-top and within the deliberately strict,
    provisional same-profile distance used for automated navigation.
    """
    if not identity.known or identity.fingerprint is None:
        raise _CaptureAbort(
            "automated Pass refused: indexed profile has no corroborated identity fingerprint")
    try:
        top = confirm_scroll_top(frame, identity_band=identity_band)
        if not top.refuted:
            raise _CaptureAbort(
                "automated Pass refused: Android edge-back did not expose a positively "
                "refuted (sticky-header) profile identity band")
        current = band_fingerprint(frame, identity_band=identity_band, grid=_IDENTITY_GRID)
        distance = fingerprint_distance(identity.fingerprint, current)
    except ScrollTopError as exc:
        raise _CaptureAbort(
            "automated Pass refused: could not read the profile identity after Android "
            f"edge-back: {exc}") from exc
    if (isinstance(distance, bool) or not isinstance(distance, (int, float))
            or not math.isfinite(distance) or distance < 0):
        raise _CaptureAbort("automated Pass refused: post-edge profile identity distance was invalid")
    if distance > _UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST:
        raise _CaptureAbort(
            "automated Pass refused: Android edge-back exposed a different or unstable profile "
            f"header ({distance:.3f} > provisional "
            f"{_UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST:.3f})")
    return float(distance)


def _settle_automated_post_pass_to_top(driver: HingeDriver, *, frame: bytes,
                                        confirm_template, identity_band) -> tuple[bytes, dict]:
    """Accept a post-Pass deck only after one bounded, non-purchase modal recovery.

    Hinge can insert an unsolicited Hinge+ page just after a real public/calibration Pass.  It
    is not a profile and must never be tapped.  A valid new top is accepted unchanged.  Because
    the first post-Pass framebuffer can also be a transient/loading state, make a few read-only
    re-probes before considering recovery.  Any visible inline composer is refused unchanged.
    Only the LAST re-probed frame, when positively confirmed as a top but not an ordinary deck,
    may receive one guarded Android edge-back; its next frame must then prove an ordinary
    composer-free top.
    """
    def composer_absent(candidate: bytes, *, context: str) -> None:
        try:
            locate_inline_composer(candidate, confirm_template, threshold=0.8)
        except ComposerDetectionError:
            return
        raise _CaptureAbort(f"automated Pass {context} left an inline composer visible")

    def ordinary_deck_ready(candidate: bytes, *, context: str) -> bool:
        # A confirmed-looking header alone is not enough: the observed Hinge+ surface can share
        # white chrome.  Reuse the driver's exact ordinary Like+Pass detector instead of trying
        # to classify/poke a purchase surface here.
        try:
            return bool(driver._observe_deck_ready(candidate))
        except Exception as exc:  # noqa: BLE001 -- no missing/failed detector may imply deck
            raise _CaptureAbort(
                f"automated Pass cannot prove ordinary deck readiness {context}: "
                f"{type(exc).__name__}: {exc}") from exc

    initial_frame = frame
    last_top = None
    last_top_error = ""
    last_ready = False
    for probe in range(_MAX_UNSETTLED_TOP_REPROBES + 1):
        # A composer is never a recoverable modal for this path.  Check every candidate, not
        # only the one that happens to settle, so a late composer cannot be erased by recovery.
        composer_absent(frame, context=f"post-action settlement probe {probe + 1}")
        try:
            last_top = confirm_scroll_top(frame, identity_band=identity_band)
            last_top_error = ""
        except ScrollTopError as exc:
            # A loading frame can be temporarily unreadable.  It still gets only this bounded,
            # read-only retry allowance and can never license a gesture.
            last_top = None
            last_top_error = str(exc)
        else:
            if last_top.confirmed:
                last_ready = ordinary_deck_ready(
                    frame, context=f"after action settlement probe {probe + 1}")
                if last_ready:
                    return frame, {
                        "modal_edge_back_used": False,
                        "ordinary_deck_ready": True,
                        "initial_post_pass_frame_sha256": _sha256(initial_frame),
                        "settled_post_pass_frame_sha256": _sha256(frame),
                    }
            else:
                last_ready = False
        if probe < _MAX_UNSETTLED_TOP_REPROBES:
            time.sleep(human_delay(driver.dwell_s))
            frame = driver.adb.screencap()

    # A modal edge-back is a narrow recovery for the measured top-looking promo only.  A
    # refuted/sticky header or an unknown post-action screen might be a real profile state, so
    # preserve it untouched rather than applying a generic dismissal gesture.
    if last_top is None:
        raise _CaptureAbort(
            "automated Pass post-action settlement preserved state: scroll-top remained "
            f"unknown/unreadable after {_MAX_UNSETTLED_TOP_REPROBES} read-only reprobes "
            f"({last_top_error or 'no verdict'}); no modal edge-back was issued")
    if not last_top.confirmed:
        state = "refuted/scrolled" if last_top.refuted else "unknown"
        raise _CaptureAbort(
            "automated Pass post-action settlement preserved state: scroll-top remained "
            f"{state} after {_MAX_UNSETTLED_TOP_REPROBES} read-only reprobes "
            f"({last_top.reason}); no modal edge-back was issued")
    # ``composer_absent`` already proved the last candidate above.  `last_ready` is false here:
    # a true value would have returned, so this is precisely the confirmed-top promo predicate.
    screen_width, screen_height = driver.adb.screen_size()
    if (not isinstance(screen_width, int) or screen_width <= 0
            or not isinstance(screen_height, int) or screen_height <= 0):
        raise _CaptureAbort("automated Pass cannot recover modal: device returned invalid screen size")
    driver._swipe(round(screen_width * _EDGE_BACK_START_X_FRAC),
                  round(screen_height * _EDGE_BACK_Y_FRAC),
                  round(screen_width * _EDGE_BACK_END_X_FRAC),
                  round(screen_height * _EDGE_BACK_Y_FRAC))
    time.sleep(human_delay(driver.dwell_s))
    settled = driver.adb.screencap()
    if settled == frame:
        raise _CaptureAbort("automated Pass modal edge-back did not change the post-action frame")
    try:
        settled_top = confirm_scroll_top(settled, identity_band=identity_band)
    except ScrollTopError as exc:
        raise _CaptureAbort(f"automated Pass cannot read state after modal edge-back: {exc}") from exc
    settled_ready = ordinary_deck_ready(settled, context="after modal edge-back")
    if not settled_top.confirmed or not settled_ready:
        raise _CaptureAbort(
            "automated Pass modal edge-back did not reach a confirmed ordinary new profile "
            f"top ({settled_top.reason}; deck_ready={settled_ready})")
    composer_absent(settled, context="modal recovery")
    return settled, {
        "modal_edge_back_used": True,
        "ordinary_deck_ready": True,
        "initial_post_pass_frame_sha256": _sha256(initial_frame),
        "settled_post_pass_frame_sha256": _sha256(settled),
        "modal_edge_back_transport": "HingeDriver._swipe(android_edge_back)",
    }


def _automated_pass_from_verified_composer(driver: HingeDriver, *, frame: bytes,
                                            confirm_template, payload: ItemPayload,
                                            item_number: int, identity: ProfileIdentity,
                                            identity_band) -> tuple[bytes, dict]:
    """Calibration-only, fail-closed path from a verified inline composer to Pass.

    This intentionally does not call ``HingeDriver.dislike``.  Its public deck guard correctly
    rejects a persistent composer (the normal Like glyph is hidden), and weakening that guard
    would make production decisions less safe.  Instead, this narrow capture helper first proves
    the composer/item, dismisses it through the existing guarded/humanized swipe chokepoint,
    then proves the sticky header still belongs to the indexed profile before vision-locating and
    guarded-tapping the floating Pass X.  Hinge may restore the profile's old scroll position
    after the keyboard disappears, so the composer is allowed to be offscreen after edge-back;
    this trace records that fact rather than pretending it was reverified.  Every condition and
    frame hash is returned for the calibration trace; this never claims a human performed the
    action.
    """
    _verified_automated_composer(
        frame, confirm_template=confirm_template, payload=payload, item_number=item_number)
    return _automated_pass_from_confirmed_composer(
        driver, frame=frame, confirm_template=confirm_template, identity=identity,
        identity_band=identity_band, selected_item_verified=True)


def _automated_send_from_verified_composer(driver: HingeDriver, *, frame: bytes,
                                           confirm_template, payload: ItemPayload,
                                           item_number: int,
                                           reviewed_confirm_point: tuple[int, int]
                                           ) -> tuple[bytes, dict]:
    """Owner-opted-in path from a verified inline composer to a REAL, permanent Send.

    Only reachable with ``capture --send-like --send-like-confirmation
    I_ACCEPT_REAL_PRIORITY_LIKE_SEND_RISK``; the tool's default (see the Pass helper above)
    never sends. Reuses the exact tap/upsell-dismiss/landed-verification the production
    comment_sheet ``like()`` flow uses (``HingeDriver._handle_rose_upsell``,
    ``HingeDriver._verify_like_landed``), so a real send from calibration is held to the same
    safety bar as an ordinary AUTO/OBSERVE like -- never the paid Rose/upsell option, and a
    like that did not structurally land raises rather than being recorded as sent.
    """
    if driver.halt_on_error is not True:
        raise _CaptureAbort(
            "automated Send refused: halt_on_error must be enabled so delivery is structurally "
            "verified rather than merely attempted")
    reviewed_surface = _verified_automated_composer(
        frame, confirm_template=confirm_template, payload=payload, item_number=item_number)
    if reviewed_surface.confirm_point != reviewed_confirm_point:
        raise _CaptureAbort(
            "automated Send refused: the checkpoint point does not exactly match the reviewed "
            "composer confirmation point")

    # This is deliberately the final read before `_tap`.  A reviewer can take long enough for
    # video controls or the composer itself to move; never replay the approved coordinate on a
    # different framebuffer, even when a fresh detector could find some other plausible Send.
    fresh_frame = driver.adb.screencap()
    for blink_attempt in range(_CARET_BLINK_RECHECK_ATTEMPTS):
        fresh_surface = _verified_automated_composer(
            fresh_frame, confirm_template=confirm_template, payload=payload,
            item_number=item_number)
        if fresh_surface != reviewed_surface:
            raise _CaptureAbort(
                "automated Send refused: fresh composer geometry changed after review; the "
                "reviewed action is no longer exact")
        if fresh_surface.confirm_point != reviewed_confirm_point:
            # Keep this separately named even though full-surface equality above also implies it.
            # It is the invariant that licenses the actual tap and protects against future
            # surface fields becoming intentionally tolerant.
            raise _CaptureAbort(
                "automated Send refused: fresh composer detection moved the confirmation point; "
                "the reviewed action is no longer exact")
        if fresh_frame == frame or _same_actionable_pixels(frame, fresh_frame):
            break
        if not _only_transient_empty_comment_caret_change(
                frame, fresh_frame, surface=reviewed_surface):
            raise _CaptureAbort(
                "automated Send refused: the composer framebuffer changed outside the "
                "empty-comment caret blink; no stale confirmation coordinate was tapped")
        if blink_attempt == _CARET_BLINK_RECHECK_ATTEMPTS - 1:
            raise _CaptureAbort(
                "automated Send refused: the empty-comment caret did not return to the exact "
                "reviewed blink phase; no confirmation coordinate was tapped")
        # A real caret alternates back to the exact reviewed pixels.  A typed vertical mark or
        # other persistent glyph does not, so it can never license Send merely by fitting a box.
        time.sleep(_CARET_BLINK_RECHECK_S)
        fresh_frame = driver.adb.screencap()
    driver._tap(*fresh_surface.confirm_point)
    time.sleep(human_cooldown(0.6))
    driver._handle_rose_upsell()
    driver._verify_like_landed(fresh_frame)
    advance_frame = driver.adb.screencap()
    return advance_frame, {
        "action": "automated_send_priority_like",
        "transport": ["HingeDriver._tap(confirm_point)", "HingeDriver._handle_rose_upsell",
                      "HingeDriver._verify_like_landed"],
        "pre_frame_sha256": _sha256(frame),
        "pre_tap_frame_sha256": _sha256(fresh_frame),
        "post_frame_sha256": _sha256(advance_frame),
        "send_like_tapped": True,
        "confirm_point": list(fresh_surface.confirm_point),
        "predicates": {
            "inline_composer_and_selected_photo_verified_before_action": True,
            "fresh_composer_and_selected_photo_reverified_before_action": True,
            "send_like_tapped": True,
            "like_landed_verified": True,
        },
    }


def _automated_pass_from_confirmed_composer(driver: HingeDriver, *, frame: bytes,
                                             confirm_template, identity: ProfileIdentity,
                                             identity_band,
                                             selected_item_verified: bool) -> tuple[bytes, dict]:
    """Clear a structurally confirmed composer through the calibration-only Pass route.

    This is the deliberately smaller primitive below ``_automated_pass_from_verified_composer``.
    The normal path always supplies ``selected_item_verified=True`` after the exact payload
    comparison.  The abort-recovery path may supply ``False`` *only* to discard a profile whose
    heart has already opened a real inline composer but whose post-tap photo comparison refused.
    It must never become targeting evidence: its trace is kept apart from ``profiles`` and says
    explicitly that the selected item was not accepted.  Both callers retain the exact same
    identity, vision-located Pass, guarded transport, and no-Send-Like guarantees.
    """
    screen_width, screen_height = driver.adb.screen_size()
    if (not isinstance(screen_width, int) or screen_width <= 0
            or not isinstance(screen_height, int) or screen_height <= 0):
        raise _CaptureAbort("automated Pass refused: device returned an invalid screen size")
    transport: list[str] = []
    try:
        surface = locate_inline_composer(frame, confirm_template, threshold=0.8)
        if surface.layout_id != _COMPOSER_LAYOUT_ID:
            raise ComposerDetectionError(f"unsupported layout {surface.layout_id!r}")
    except ComposerDetectionError as exc:
        raise _CaptureAbort(
            "automated Pass refused: current frame no longer structurally proves the inline "
            f"composer before cleanup: {exc}") from exc

    # An unfocused composer can already expose Pass.  A second Android edge-back in that state
    # may leave Hinge altogether (observed live), so it is not a harmless idempotent dismissal.
    # First require the same sticky profile header *on the current composer frame*, then use two
    # ordinary vision locations on that state.  Locator absence/motion means "not ready" and may
    # take the existing one-edge fallback; an identity failure is a hard refusal with no gesture.
    initial_identity_distance = _require_same_profile_header_after_edge_back(
        frame, identity=identity, identity_band=identity_band)

    def stable_visible_pass() -> tuple[bytes, tuple[int, int], float] | None:
        try:
            first_point = driver._locate_button("pass")
        except Exception:  # noqa: BLE001 -- a locator failure is not a reason to tap/Send Like
            return None
        if first_point is None:
            return None
        candidate = driver.adb.screencap()
        # Do not turn a changed/misidentified state into the edge-back fallback.  We have already
        # observed a protected control; a non-matching header now is terminal without input.
        candidate_identity_distance = _require_same_profile_header_after_edge_back(
            candidate, identity=identity, identity_band=identity_band)
        try:
            refreshed_point = driver._locate_button("pass")
        except Exception:  # noqa: BLE001 -- no second location, no direct guarded tap
            return None
        if refreshed_point is None:
            return None
        max_relocation = max(1.0, screen_width * _PASS_POINT_RELOCATION_MAX_FRAC)
        if math.dist(first_point, refreshed_point) > max_relocation:
            return None
        return candidate, refreshed_point, candidate_identity_distance

    visible_pass = stable_visible_pass()
    edge_back_used = visible_pass is None

    # The system back gesture is not a raw ADB escape hatch: `_swipe` is the same
    # forbidden-zone checked, humanized touch transport used by every other explicit drag.
    # Do not infer keyboard state from a brittle absolute composer-CTA height.  The measured
    # postcondition is the re-exposed floating Pass plus an identity-matching sticky header.
    if visible_pass is None:
        driver._swipe(round(screen_width * _EDGE_BACK_START_X_FRAC),
                      round(screen_height * _EDGE_BACK_Y_FRAC),
                      round(screen_width * _EDGE_BACK_END_X_FRAC),
                      round(screen_height * _EDGE_BACK_Y_FRAC))
        transport.append("HingeDriver._swipe(android_edge_back)")
        time.sleep(human_delay(driver.dwell_s))
        post_edge_frame = driver.adb.screencap()
        post_edge_identity_distance = _require_same_profile_header_after_edge_back(
            post_edge_frame, identity=identity, identity_band=identity_band)

        # `_await_button` is the existing vision/retry locator; do not substitute a coordinate.
        # Re-locate once on the immediately preceding stable state and require agreement before
        # the guarded tap, so a floating control that moves during its settle animation cannot
        # be reused.
        located_point = driver._await_button("pass")
        pre_tap_frame = driver.adb.screencap()
        pre_tap_identity_distance = _require_same_profile_header_after_edge_back(
            pre_tap_frame, identity=identity, identity_band=identity_band)
        refreshed_point = driver._locate_button("pass")
        if refreshed_point is None:
            raise _CaptureAbort("automated Pass refused: floating Pass X disappeared before guarded tap")
        max_relocation = max(1.0, screen_width * _PASS_POINT_RELOCATION_MAX_FRAC)
        if math.dist(located_point, refreshed_point) > max_relocation:
            raise _CaptureAbort(
                "automated Pass refused: floating Pass X moved after vision location "
                f"({math.dist(located_point, refreshed_point):.1f}px > {max_relocation:.1f}px)")
    else:
        pre_tap_frame, refreshed_point, pre_tap_identity_distance = visible_pass
        post_edge_frame = frame
        post_edge_identity_distance = initial_identity_distance
        transport.extend(("HingeDriver._locate_button(pass)",
                          "HingeDriver._locate_button(pass)"))
    driver._tap(*refreshed_point)
    if edge_back_used:
        transport.extend(("HingeDriver._await_button(pass)", "HingeDriver._locate_button(pass)"))
    transport.append("HingeDriver._tap")
    time.sleep(human_delay(driver.dwell_s))
    raw_advance_frame = driver.adb.screencap()
    if raw_advance_frame == pre_tap_frame:
        raise _CaptureAbort("automated Pass did not produce a changed deck frame")
    advance_frame, settle_trace = _settle_automated_post_pass_to_top(
        driver, frame=raw_advance_frame, confirm_template=confirm_template,
        identity_band=identity_band)

    return advance_frame, {
        "action": "automated_pass",
        "transport": transport,
        "pre_frame_sha256": _sha256(frame),
        "post_edge_back_frame_sha256": _sha256(post_edge_frame),
        "pre_tap_frame_sha256": _sha256(pre_tap_frame),
        "post_frame_sha256": _sha256(advance_frame),
        "send_like_tapped": False,
        "composer_clear_visible": True,
        "pass_point": list(refreshed_point),
        "predicates": {
            "inline_composer_and_selected_photo_verified_before_action": selected_item_verified,
            "inline_composer_structurally_confirmed_before_action": True,
            "edge_back_transport_performed": edge_back_used,
            "unfocused_composer_pass_visible_before_edge_back": not edge_back_used,
            "post_edge_same_profile_sticky_header_verified": True,
            "post_edge_identity_distance": pre_tap_identity_distance,
            "initial_post_edge_identity_distance": post_edge_identity_distance,
            "post_edge_composer_reverified": False,
            "post_edge_composer_visibility": (
                "not_required_may_be_offscreen" if edge_back_used
                else "structurally_visible_unfocused_before_pass"),
            "pass_vision_relocated_before_guarded_tap": True,
            "send_like_tapped": False,
            "deck_frame_changed": True,
            "composer_clear_visible": True,
            "new_profile_top_confirmed": True,
            "modal_edge_back_used": settle_trace["modal_edge_back_used"],
        },
        "post_pass_settle": settle_trace,
    }


def _recover_automated_abort_from_open_composer(
        driver: HingeDriver, *, frame: bytes, confirm_template, payload: ItemPayload,
        item_number: int, identity: ProfileIdentity, identity_band, failure_stage: str) -> dict:
    """Best-effort, fail-closed cleanup after an automated heart left an unsent composer.

    ``capture`` normally refuses a post-tap comparison before it can call the verified-composer
    Pass helper.  Leaving that composer open strands the device: production ``dislike`` correctly
    refuses because the ordinary Like control is hidden.  This recovery is intentionally not a
    second attempt to make the heart valid.  It either proves the exact item and delegates to the
    ordinary calibration-only Pass helper, or (only when the *current* frame still structurally
    proves Hinge's inline composer) discards the already-hearted profile using the same guarded
    edge-back / identity / vision-located Pass route.  It never taps the composer CTA or Send
    Like, never weakens production ``HingeDriver.dislike``, and never adds evidence to the
    calibration profile set.

    A failed precondition returns an honest ``not_cleared`` trace without sending another touch.
    If a recovery gesture itself fails, its trace records that failure and the original capture
    remains aborted; callers must not retry the heart on that profile.
    """
    trace = {
        "kind": "automated_abort_unsent_composer_recovery_v1",
        "failure_stage": failure_stage,
        "photo_model_item": item_number,
        "initial_frame_sha256": _sha256(frame),
        "human_ground_truth": False,
        "calibration_evidence": False,
        "send_like_tapped": False,
        "outcome": "not_cleared",
    }
    try:
        surface = locate_inline_composer(frame, confirm_template, threshold=0.8)
        if surface.layout_id != _COMPOSER_LAYOUT_ID:
            raise ComposerDetectionError(f"unsupported layout {surface.layout_id!r}")
    except ComposerDetectionError as exc:
        trace["refusal"] = f"current frame has no supported inline composer: {exc}"
        return trace

    try:
        # Prefer the normal exact-item route whenever it is available.  It preserves the most
        # stringent precondition and makes recovery merely an abort-time invocation of the
        # already-audited Pass implementation.
        advance, pass_trace = _automated_pass_from_verified_composer(
            driver, frame=frame, confirm_template=confirm_template, payload=payload,
            item_number=item_number, identity=identity, identity_band=identity_band)
        trace.update({
            "outcome": "cleared",
            "selected_item_verified_before_cleanup": True,
            "post_frame_sha256": _sha256(advance),
            "cleanup_trace": pass_trace,
        })
        return trace
    except _CaptureAbort as verified_exc:
        # A *comparison* refusal is the one state this function was created for.  Do not fall
        # back after an edge-back/Pass failure: another gesture could be acting on a state the
        # normal helper has already changed or found unsafe.
        try:
            verdict = verify_sheet_item(frame, payload, item_number, composer_surface=surface)
        except Exception as exc:  # noqa: BLE001 -- no unknown verifier state licenses a Pass
            trace["refusal"] = (
                "could not classify verified-composer failure before cleanup fallback: "
                f"{type(exc).__name__}: {exc}")
            return trace
        if verdict.matched:
            trace["refusal"] = (
                "verified-composer Pass changed/refused after item verification; no fallback "
                f"gesture is permitted: {verified_exc}")
            return trace
        try:
            advance, pass_trace = _automated_pass_from_confirmed_composer(
                driver, frame=frame, confirm_template=confirm_template, identity=identity,
                identity_band=identity_band, selected_item_verified=False)
        except _CaptureAbort as cleanup_exc:
            trace["refusal"] = (
                "structural-composer cleanup refused before a safe completion: "
                f"{cleanup_exc}")
            return trace
        trace.update({
            "outcome": "cleared",
            "selected_item_verified_before_cleanup": False,
            "selected_item_verification_refusal": str(verified_exc),
            "post_frame_sha256": _sha256(advance),
            "cleanup_trace": pass_trace,
        })
        return trace


def _save_frame(out_dir: Path, frames_meta: list, frame_counter: int, png: bytes, *,
                role: str, profile_ordinal: int, profile_id: str,
                item_number: int | None, captured_utc: str | None = None) -> int:
    """Append one frame to disk + the manifest's `frames` list. Filenames are a single counter
    across the WHOLE session (not per-profile), `%05d.png`, matching every other
    `ops/calibration/` tool in this repo -- so capture order is unambiguous from the filename
    alone and `measure` can reconstruct each profile's frame sequence in the exact order it was
    captured, which is what `build_item_payload`'s frame-digest check requires."""
    frame_counter += 1
    name = f"{frame_counter:05d}.png"
    atomic_write_private_bytes(out_dir / name, png, parent=out_dir)
    frames_meta.append({
        "file": name,
        "sha256": _sha256(png),
        "role": role,
        "profile_ordinal": profile_ordinal,
        "profile_id": profile_id,
        "item_number": item_number,
        "captured_utc": captured_utc or datetime.now(timezone.utc).isoformat(),
    })
    return frame_counter


def _commit_profile_frames(out_dir: Path, frames_meta: list, frame_counter: int,
                           staged: list[tuple[bytes, str, int | None, str]], *,
                           profile_ordinal: int, profile_id: str) -> int:
    """Commit one successful attempt atomically at the manifest level.

    A refused retry must not leave frames behind under an ordinal later reused for another
    attempt. Those frames would either make the session unloadable or, worse, merge two real
    people when the operator reused a label. The capture loop therefore keeps an attempt in
    memory (it already retains the card frames for indexing) and writes it only after every
    requested inline composer and the terminal profile-advance clear proof are captured.
    """
    for png, role, item_number, captured_utc in staged:
        frame_counter = _save_frame(
            out_dir, frames_meta, frame_counter, png, role=role,
            profile_ordinal=profile_ordinal, profile_id=profile_id,
            item_number=item_number, captured_utc=captured_utc)
    return frame_counter


class _OfflineNavigationReplay:
    """Replay a captured top-to-bottom read backwards for ``item_nav`` without a phone.

    It deliberately implements only the narrow driver surface navigation consumes.  A replay
    ``_scroll_up_one`` advances to an ALREADY HASHED screenshot; it never owns an ADB object,
    never emits input, and refuses if navigation asks for a frame outside the capture.  Reversing
    the capture is the one honest offline analogue of bottom-up navigation: its entry frame is
    the read's final frame and every subsequent frame moves toward the confirmed top.
    """

    def __init__(self, frames: list[bytes], *, identity_band, content_band, like_template):
        if not frames:
            raise _MeasureRefused("entry-anchor replay was given no card-scroll frames")
        self._frames = list(reversed(frames))
        self._position = 0
        self.identity_band = identity_band
        self.content_band = content_band
        self._like_template = like_template
        self.capture_calls = 0
        self.scroll_up_calls: list[tuple[float, float]] = []

    def _template(self, name):
        if name != "like":
            raise AssertionError(f"offline navigation requested unexpected template {name!r}")
        return self._like_template

    def _screencap(self) -> bytes:
        self.capture_calls += 1
        if self._position >= len(self._frames):
            raise AssertionError(
                "offline entry-anchor replay would need a frame that was not captured; refusing "
                "rather than fabricating a navigation result")
        return self._frames[self._position]

    def _sample_read_step(self, _step_index, _profile):
        # No sleep and no policy-derived movement: replay only consumes a recorded next frame.
        return 0.0, 0.0, 0.5

    def _scroll_up_one(self, frac: float, x_frac: float) -> None:
        self.scroll_up_calls.append((float(frac), float(x_frac)))
        self._position += 1
        if self._position >= len(self._frames):
            raise AssertionError(
                "offline entry-anchor replay would scroll beyond the recorded card frames; "
                "refusing rather than issuing a live gesture")


def _entry_anchor_profile_report(*, ordinal: int, profile_id: str, card_frames: list[bytes],
                                 frame_records: list[dict], identity_band, content_band,
                                 like_template, like_threshold) -> dict:
    """Prove one captured profile is navigable bottom-up using only its recorded bytes.

    The replay-only identity ceiling is deliberately derived solely to call the production
    navigator's already-strict API.  It is neither a calibration measurement nor a runtime
    value: it is omitted from config/YAML and cannot escape this JSON evidence artifact.
    """
    if len(card_frames) != len(frame_records) or not card_frames:
        raise _MeasureRefused(f"profile {profile_id!r}: card-frame records are incomplete")
    for data, rec in zip(card_frames, frame_records, strict=True):
        if _sha256(data) != rec.get("sha256"):
            raise _MeasureRefused(
                f"profile {profile_id!r}: entry-anchor source frame {rec.get('file')!r} no "
                "longer matches its recorded sha256")
    try:
        top = confirm_scroll_top(card_frames[0], identity_band=identity_band)
    except ScrollTopError as exc:
        raise _MeasureRefused(
            f"profile {profile_id!r}: entry-anchor source cannot read the claimed top frame "
            f"({exc})") from exc
    if not top.confirmed:
        raise _MeasureRefused(
            f"profile {profile_id!r}: first card frame is not a confirmed profile top "
            f"({top.reason})")
    try:
        index = build_item_index(
            card_frames, content_band=content_band, like_template=like_template,
            like_threshold=like_threshold, at_scroll_top=True, identity_band=identity_band)
        # Classifier-only, matching the capture loops byte for byte.  This replay exists to
        # prove that the numbering capture produced is REPRODUCIBLE from the committed frames;
        # a stricter gate here would refuse items capture legitimately numbered and turn a
        # reproduction check into a different check.
        payload = build_item_payload(
            card_frames, index, unnumber=unnumber_unless_confident_photo)
    except (ItemIndexError, ItemCropError, SegmentationError, ShiftEstimationError) as exc:
        raise _MeasureRefused(
            f"profile {profile_id!r}: could not rebuild complete photo-only item numbering "
            f"for entry-anchor evidence ({type(exc).__name__}: {exc})") from exc
    if not index.usable or not index.complete or not payload.usable:
        raise _MeasureRefused(
            f"profile {profile_id!r}: entry-anchor evidence requires a usable complete index "
            f"and usable photo-only payload (index usable={index.usable}, complete={index.complete}, "
            f"payload usable={payload.usable})")
    if not index.identity.known:
        raise _MeasureRefused(
            f"profile {profile_id!r}: complete index has no corroborated sticky-header identity")
    if not payload.items:
        raise _MeasureRefused(
            f"profile {profile_id!r}: complete profile has no photo-only model items to replay")
    if payload.translation != tuple(c.heart_ordinal for c in payload.items):
        raise _MeasureRefused(
            f"profile {profile_id!r}: photo-only payload translation is internally inconsistent")

    try:
        entry_fp = band_fingerprint(card_frames[-1], identity_band=identity_band,
                                    grid=_IDENTITY_GRID)
        entry_distance = _checked_distance(
            fingerprint_distance(index.identity.fingerprint, entry_fp),
            context=f"profile {profile_id!r} offline navigation entry identity")
    except ScrollTopError as exc:
        raise _MeasureRefused(
            f"profile {profile_id!r}: could not fingerprint bottom-up replay entry ({exc})") from exc
    if entry_distance >= _IDENTITY_FALSE_MATCH_DISTANCE:
        raise _MeasureRefused(
            f"profile {profile_id!r}: replay entry identity is at/over the known foreign-profile "
            f"ceiling ({entry_distance:.3f} >= {_IDENTITY_FALSE_MATCH_DISTANCE})")
    # This midpoint is a per-artifact replay permit, not a measured runtime threshold. It is
    # strictly above this replay's observed same-profile entry and strictly below the collision.
    replay_identity_ceiling = (entry_distance + _IDENTITY_FALSE_MATCH_DISTANCE) / 2.0

    cancellation_replay = _OfflineNavigationReplay(
        card_frames, identity_band=identity_band, content_band=content_band,
        like_template=like_template)
    first_crop = payload.items[0]
    try:
        first_index_model = index.translation.index(first_crop.heart_ordinal) + 1
        navigate_to_item(
            cancellation_replay, index, first_index_model, entry_reference=card_frames[-1],
            identity_match_max_dist=replay_identity_ceiling, should_stop=lambda: True,
            max_frames=len(card_frames))
    except ActionCancelled:
        pass
    except Exception as exc:  # noqa: BLE001 -- a different refusal is not cancellation proof
        raise _MeasureRefused(
            f"profile {profile_id!r}: cancellation dry-run did not stop before navigation "
            f"({type(exc).__name__}: {exc})") from exc
    else:
        raise _MeasureRefused(
            f"profile {profile_id!r}: cancellation dry-run unexpectedly returned a target")
    if cancellation_replay.capture_calls or cancellation_replay.scroll_up_calls:
        raise _MeasureRefused(
            f"profile {profile_id!r}: cancellation dry-run touched replay input before stopping")

    targets = []
    for crop in payload.items:
        if crop.heart_ordinal not in index.translation:
            raise _MeasureRefused(
                f"profile {profile_id!r}: photo item {crop.number} maps to a heart missing from "
                "the absolute index translation")
        index_model = index.translation.index(crop.heart_ordinal) + 1
        replay = _OfflineNavigationReplay(
            card_frames, identity_band=identity_band, content_band=content_band,
            like_template=like_template)
        try:
            target = navigate_to_item(
                replay, index, index_model, entry_reference=card_frames[-1],
                identity_match_max_dist=replay_identity_ceiling, max_frames=len(card_frames))
        except (ItemNavigationError, AssertionError, ShiftEstimationError, SegmentationError) as exc:
            raise _MeasureRefused(
                f"profile {profile_id!r}: offline bottom-up replay refused photo item {crop.number} "
                f"(heart {crop.heart_ordinal}; {type(exc).__name__}: {exc})") from exc
        targets.append({
            "photo_model_item": crop.number,
            "heart_ordinal": crop.heart_ordinal,
            "index_model_item": index_model,
            "entry_anchor_delta_px": target.anchor.delta_px,
            "entry_offset_px": target.entry_offset,
            "landing_frame_sha256": _sha256(target.frame),
            "landing_frame_index": target.frame_index,
            "landing_page_offset_px": target.page_offset,
            "hearts_counted": target.hearts_counted,
            "crosscheck_max_disagreement_px": target.agreement_px,
            "replay_scroll_steps": len(target.steps),
            "replay_shift_deltas_px": [shift.delta_px for shift in target.shifts],
        })
    return {
        "ordinal": ordinal,
        "profile_id_sha256": _sha256(profile_id.encode("utf-8", "replace")),
        "source_card_frames": [
            {"file": rec["file"], "sha256": rec["sha256"]} for rec in frame_records],
        "confirmed_top": True,
        "index_complete": True,
        "index_reached_end": bool(index.reached_end),
        "index_heart_translation": list(index.translation),
        "photo_only_translation": list(payload.translation),
        "offline_replay_identity_ceiling": replay_identity_ceiling,
        "offline_replay_identity_entry_distance": entry_distance,
        "cancellation_before_capture": True,
        "cancellation_replay_capture_calls": cancellation_replay.capture_calls,
        "cancellation_replay_scroll_up_calls": len(cancellation_replay.scroll_up_calls),
        "targets": targets,
    }


def _write_entry_anchor_ledger(out_dir: Path, *, frames_meta: list[dict], profiles_meta: list[dict],
                               identity_band, content_band, like_template, like_threshold) -> dict:
    """Write a hash-bound, offline-only proof for the entry-anchor checklist item."""
    by_ordinal: dict[int, list[dict]] = {}
    for rec in frames_meta:
        if rec.get("role") == "card_scroll":
            by_ordinal.setdefault(rec["profile_ordinal"], []).append(rec)
    reports = []
    for meta in profiles_meta:
        ordinal = meta["ordinal"]
        records = by_ordinal.get(ordinal, [])
        if not records:
            raise _MeasureRefused(
                f"profile {meta['profile_id']!r}: no committed card-scroll frames for entry-anchor evidence")
        card_frames = []
        for rec in records:
            path = out_dir / rec["file"]
            if not path.is_file():
                raise _MeasureRefused(f"entry-anchor source frame {path} is missing")
            data = path.read_bytes()
            if _sha256(data) != rec["sha256"]:
                raise _MeasureRefused(f"entry-anchor source frame {path} changed after capture")
            card_frames.append(data)
        reports.append(_entry_anchor_profile_report(
            ordinal=ordinal, profile_id=meta["profile_id"], card_frames=card_frames,
            frame_records=records, identity_band=identity_band, content_band=content_band,
            like_template=like_template, like_threshold=like_threshold))
    artifact = {
        "schema_version": _ENTRY_ANCHOR_LEDGER_SCHEMA_VERSION,
        "kind": "hinge_entry_anchor_offline_replay",
        "tool_version": _TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "phone_input_issued": False,
        "runtime_targeting_calibration_emitted": False,
        "profiles": reports,
    }
    path = out_dir / _ENTRY_ANCHOR_LEDGER_FILE
    atomic_write_private_text(path, json.dumps(artifact, indent=2) + "\n", parent=out_dir)
    return {"file": path.name, "sha256": _sha256(path.read_bytes())}


def _capture_one_profile(driver: HingeDriver, out_dir: Path, *, ordinal: int,
                         frame_counter: int, frames_meta: list,
                         used_profile_ids: set[str]) -> tuple[dict, int]:
    """Capture one profile's complete card sequence plus persistent inline-composer evidence.

    Every heart-tap and profile advance is done by the OWNER, by hand -- see the module
    docstring for why this tool never targets an item itself. The only gestures THIS function
    sends are small forward read-scrolls (`driver._scroll_down_one`), used purely to reveal
    enough of the card for `item_identity.capture_profile_identity` to confirm-then-refute-then-
    corroborate the scroll top and for `item_index.build_item_index` to enumerate every item
    number the owner is about to target -- both checked live, with the SAME offline functions
    `measure` uses later, so "enough frames" is a verified fact rather than a guess.
    """
    identity_band = driver.identity_band
    content_band = driver.content_band
    like_template = driver._template("like")
    confirm_template = driver._template("confirm")
    like_threshold = hinge_mod._LIKE_MATCH_THRESHOLD

    while True:  # retry the whole profile on a refusal, at the owner's choice
        staged_frames: list[tuple[bytes, str, int | None, str]] = []
        profile_id = ""
        while not profile_id:
            candidate = input(
                f"  Profile {ordinal}: type a short LOCAL label for this profile (e.g. "
                "initials plus a distinguishing suffix -- it MUST uniquely identify this real "
                "profile across every calibration session; never committed anywhere but this "
                "session's own gitignored manifest.json) > ").strip()
            if candidate in used_profile_ids:
                print(f"  Profile id {candidate!r} was already used in this session. Use a "
                      "new, unique id so foreign-profile comparisons cannot be skipped.")
                continue
            profile_id = candidate
        try:
            target_items = _parse_item_numbers(input(
                f"  Profile {ordinal} ({profile_id}): which PHOTO number(s) will you target for "
                "the inline-composer calibration? Prompts are never numbered or targeted. "
                "Comma-separated, e.g. 1,3 > "))
        except ValueError as exc:
            print(f"  {exc}; try again.")
            continue

        input(
            "  On the phone: open a NEW real profile you have not already used in this "
            "session, and scroll it to its very TOP by hand (the filter-chips row visible, no "
            "per-profile header yet). Press ENTER once it is at the top > ")

        top_frame = driver.adb.screencap()
        top_verdict = confirm_scroll_top(top_frame, identity_band=identity_band)
        if not top_verdict.confirmed:
            print(f"  Not confirmed at scroll top ({top_verdict.reason}).")
            if input("  Retry this profile? [Y/n] > ").strip().lower().startswith("n"):
                raise _CaptureAbort("owner declined to retry after a scroll-top refusal")
            continue

        card_frames: list[bytes] = [top_frame]
        staged_frames.append((top_frame, "card_scroll", None,
                              datetime.now(timezone.utc).isoformat()))

        identity: ProfileIdentity | None = None
        index = None
        payload = None
        prior_index = None
        profile_min_spacing_px = None
        planning_refusal = None
        satisfied = False
        for _step in range(_MAX_CARD_SCROLLS):
            identity = capture_profile_identity(card_frames, identity_band=identity_band,
                                                grid=_IDENTITY_GRID)
            index_ok = False
            try:
                index = build_item_index(
                    card_frames, content_band=content_band, like_template=like_template,
                    like_threshold=like_threshold, at_scroll_top=True,
                    identity_band=identity_band, _prefix_index=prior_index)
                # The next pass may reuse only this exact successful frame prefix.  The index
                # builder hash-checks it and otherwise rebuilds from scratch; it still folds the
                # whole sequence every pass, so this cache cannot change a stop/refusal result.
                prior_index = index
                # Calibration negatives must include EVERY item available on this profile, not
                # merely enough of the prefix to reach the requested target. `complete` proves
                # the capture began at top, reached the end, and left no partial block out.
                if index.usable and index.complete:
                    # CLASSIFIER-ONLY NUMBERING, DELIBERATELY, exactly as in the automated loop
                    # below.  Numbering here selects WHICH item to navigate to; no heart is spent
                    # on its strength.  `_verified_still_photo_proof` is the sole still-photo
                    # licence for a heart and takes real PARKED evidence in the pre-heart window,
                    # once navigation has stopped the card.  Wiring the strict gate in here
                    # (e5054f7f) with `still_photo_dwell=None` made every profile skip (live
                    # failure 2026-08-22): a read scroll separates every candidate from its next
                    # frame, so this loop has no un-interacted dwell to offer and the dwell rung
                    # can never pass on read-scroll frames.
                    payload = build_item_payload(
                        card_frames, index, unnumber=unnumber_unless_confident_photo)
                    index_ok = payload.usable and len(payload.items) >= max(target_items)
                else:
                    payload = None
            except (ItemIndexError, ItemCropError, SegmentationError, ShiftEstimationError):
                index = None
                payload = None
            if identity.known and index_ok:
                satisfied = True
                break
            try:
                step, profile_min_spacing_px = _plan_card_scroll(
                    card_frames[-1], content_band=content_band, like_template=like_template,
                    like_threshold=like_threshold,
                    profile_min_spacing_px=profile_min_spacing_px)
            except (SegmentationError, ScrollStepError) as exc:
                planning_refusal = (
                    "could not safely plan the next card-enumeration scroll from the current "
                    f"frame ({type(exc).__name__}: {exc})")
                break
            time.sleep(human_delay(driver.dwell_s))
            # Both plan arguments are required.  In particular, passing just a fraction lets the
            # driver re-sample its ordinary read cadence, which can exceed this plan's aliasing
            # bound before the next recorded frame exists.
            driver._scroll_down_one(step.frac, step.x_frac)
            frame = _settled_read_scroll_frame(driver)
            card_frames.append(frame)
            staged_frames.append((frame, "card_scroll", None,
                                  datetime.now(timezone.utc).isoformat()))

        if not satisfied:
            if planning_refusal is not None:
                reason = planning_refusal
            elif identity is not None and not identity.known:
                reason = identity.reason
            elif index is not None and not index.usable:
                reason = "; ".join(index.failures)
            else:
                have = len(payload.items) if payload is not None else 0
                reason = (f"only {have} photo item(s) indexed, need >= {max(target_items)}, "
                          "or the full profile end/partial-block check was not satisfied")
            print(f"  Could not capture a complete, usable, identity-confirmed sequence "
                  f"covering every item through requested item {max(target_items)} within "
                  f"{_MAX_CARD_SCROLLS} scrolls: {reason}")
            if input("  Retry this profile from a fresh top? [Y/n] > ").strip().lower().startswith("n"):
                raise _CaptureAbort("owner declined to retry after a card-sequence refusal")
            continue

        composer_items_meta = []
        for n in target_items:
            while True:
                input(
                    f"  Position item {n} so its card is visible. Press ENTER immediately BEFORE "
                    "you tap its heart > ")
                target_pre = driver.adb.screencap()
                input(
                    f"  Manually tap item {n}'s heart now (YOUR OWN HAND -- this tool never "
                    "targets an item). The inline composer persists; do NOT try to dismiss it. "
                    "Press ENTER once Send Like is visible beneath the selected item > ")
                composer_open = driver.adb.screencap()
                try:
                    surface = locate_inline_composer(composer_open, confirm_template, threshold=0.8)
                except ComposerDetectionError as exc:
                    print(f"  That does not structurally prove an inline composer ({exc}). Try again.")
                    continue
                if surface.layout_id != _COMPOSER_LAYOUT_ID:
                    print(f"  Refusing unknown composer layout {surface.layout_id!r}; try again.")
                    continue
                break
            captured_utc = datetime.now(timezone.utc).isoformat()
            staged_frames.extend(((target_pre, "target_pre", n, captured_utc),
                                  (composer_open, "composer_open", n, captured_utc)))
            composer_items_meta.append(n)

        while True:
            input(
                "  After the final composer capture, manually advance to a NEW profile and its "
                "scroll top. Do NOT tap Send Like. Press ENTER once filter chips are visible and "
                "the previous composer is gone > ")
            advance_frame = driver.adb.screencap()
            top_verdict = confirm_scroll_top(advance_frame, identity_band=identity_band)
            if not top_verdict.confirmed:
                print(f"  New profile top was not confirmed ({top_verdict.reason}). Try again.")
                continue
            try:
                locate_inline_composer(advance_frame, confirm_template, threshold=0.8)
            except ComposerDetectionError:
                break
            print("  An inline composer is still visible; advance to a new profile before continuing.")
        staged_frames.append((advance_frame, "profile_advance_clear", None,
                              datetime.now(timezone.utc).isoformat()))
        while True:
            input(
                "  On that NEW profile, scroll just enough for its sticky name header to replace "
                "the filter chips. Press ENTER once it is visibly stable > ")
            advance_identity = driver.adb.screencap()
            try:
                identity_top = confirm_scroll_top(advance_identity, identity_band=identity_band)
                if not identity_top.refuted:
                    print("  The new profile's sticky header was not positively visible. Try again.")
                    continue
                advance_fp = band_fingerprint(
                    advance_identity, identity_band=identity_band, grid=_IDENTITY_GRID)
                distance = _checked_distance(
                    fingerprint_distance(identity.fingerprint, advance_fp),
                    context=f"profile {profile_id!r} advance identity")
            except ScrollTopError as exc:
                print(f"  Could not read the new profile identity header ({exc}). Try again.")
                continue
            if distance <= _IDENTITY_FALSE_MATCH_DISTANCE:
                print(
                    f"  The claimed next profile is only {distance:.3f} from the prior identity "
                    f"(must be > {_IDENTITY_FALSE_MATCH_DISTANCE}); do not certify a same-profile "
                    "scroll as an advance. Try again.")
                continue
            break
        staged_frames.append((advance_identity, "profile_advance_identity", None,
                              datetime.now(timezone.utc).isoformat()))

        profile_meta = {
            "ordinal": ordinal,
            "profile_id": profile_id,
            "card_scroll_frames": len(card_frames),
            "composer_items": composer_items_meta,
            "profile_advance_cleared_composer": True,
            "profile_advance_identity_mismatched": True,
            "identity_frame_index": identity.frame_index,
            "identity_agreeing_frames": identity.agreeing_frames,
        }
        frame_counter = _commit_profile_frames(
            out_dir, frames_meta, frame_counter, staged_frames,
            profile_ordinal=ordinal, profile_id=profile_id)
        used_profile_ids.add(profile_id)
        return profile_meta, frame_counter


def _capture_one_profile_unattended(driver: HingeDriver, out_dir: Path, *, ordinal: int,
                                    frame_counter: int, frames_meta: list,
                                    used_profile_ids: set[str],
                                    review_gate: _HybridReviewGate | None = None,
                                    skipped_attempts: list[dict] | None = None,
                                    abort_recoveries: list[dict] | None = None,
                                    send_like: bool = False,
                                    entry_drift_restart_attempts: int = 0,
                                    target_strategy_id: str = _AUTOMATED_TARGET_STRATEGY_ID
                                    ) -> tuple[dict, int]:
    """Capture one profile with explicitly-authorized device actions.

    This is deliberately separate from the supervised path above.  It uses the same bounded
    scroll planner, item index, bottom-up navigator, guarded driver tap transport, and composer
    detector; it does *not* recast those observations as operator actions or independent ground
    truth.  Any ambiguity leaves the composer/profile untouched and aborts the session.

    `target_strategy_id` defaults to the alternating strategy, so every existing caller is
    unaffected; `_cmd_capture` threads its own resolved choice through explicitly.
    """
    identity_band = driver.identity_band
    content_band = driver.content_band
    like_template = driver._template("like")
    confirm_template = driver._template("confirm")
    like_threshold = hinge_mod._LIKE_MATCH_THRESHOLD
    if (type(entry_drift_restart_attempts) is not int
            or not 0 <= entry_drift_restart_attempts
            <= _MAX_ENTRY_DRIFT_REENUMERATION_RESTARTS_PER_PROFILE):
        raise _CaptureAbort("invalid bounded entry-drift re-enumeration restart count")
    target_items = _automated_composer_items_for_ordinal(ordinal, strategy_id=target_strategy_id)

    def recover_unsent_composer(*, frame: bytes, item_number: int, failure_stage: str) -> dict:
        """Record the cleanup result even though this profile never becomes evidence."""
        recovery = _recover_automated_abort_from_open_composer(
            driver, frame=frame, confirm_template=confirm_template, payload=payload,
            item_number=item_number, identity=identity, identity_band=identity_band,
            failure_stage=failure_stage)
        recovery["ordinal"] = ordinal
        recovery["profile_id_sha256"] = _sha256(profile_id.encode("utf-8", "replace"))
        if abort_recoveries is not None:
            abort_recoveries.append(recovery)
        return recovery

    def assert_skip_budget() -> None:
        # Check budgets *before* the public Pass.  A cap must stop a changed implementation
        # before it spends another profile, not merely decline to record the extra advance.
        attempts = skipped_attempts or []
        ordinal_attempts = sum(1 for attempt in attempts if attempt.get("ordinal") == ordinal)
        if ordinal_attempts >= _MAX_PREACTION_PROFILE_SKIPS_PER_ORDINAL:
            raise _CaptureAbort(
                f"automated profile {ordinal}: exceeded the bounded "
                f"{_MAX_PREACTION_PROFILE_SKIPS_PER_ORDINAL}-attempt pre-action skip budget")
        if len(attempts) >= _MAX_PREACTION_PROFILE_SKIPS_PER_SESSION:
            raise _CaptureAbort(
                "automated capture exceeded the bounded session pre-action skip budget of "
                f"{_MAX_PREACTION_PROFILE_SKIPS_PER_SESSION}")

    def skip_before_heart(reason: _PreActionProfileRetry, identity: ProfileIdentity) -> _ProfileSkipped:
        assert_skip_budget()
        try:
            return _ProfileSkipped(_skip_automated_profile_before_heart(
                driver, ordinal=ordinal, reason=reason, identity=identity,
                identity_band=identity_band, content_band=content_band,
                like_template=like_template, like_threshold=like_threshold,
                review_gate=review_gate, send_like=send_like))
        except _UnsupportedEntryDeck as exc:
            # The pre-action skip's own rewind meets the same unsupported layout the entry
            # rewind does, from deeper in the card, so it must escalate the same way. Letting it
            # propagate would surface as a plain `_CaptureAbort` in `_cmd_capture`, marking the
            # whole session interrupted and discarding every profile already banked. The rewind
            # is the first thing that function does after its identity precondition, so no
            # device action has happened yet; the budget was already charged once above and this
            # escalation still emits exactly one skip record.
            return _ProfileSkipped(_skip_automated_unsupported_entry_deck(
                driver, ordinal=ordinal, entry=exc, identity_band=identity_band,
                review_gate=review_gate, send_like=send_like, skip_reason=reason))

    # Every automated/hybrid profile begins with a visually-driven rewind, including profiles
    # reached after a reviewer-directed restart.  It is intentionally independent of the
    # driver's in-process scroll ledger, which cannot represent a handoff position.
    try:
        top_frame = _rewind_automated_profile_to_confirmed_top(
            driver, ordinal=ordinal, identity_band=identity_band, content_band=content_band,
            like_template=like_template, like_threshold=like_threshold)
    except _UnsupportedEntryDeck as exc:
        assert_skip_budget()
        raise _ProfileSkipped(_skip_automated_unsupported_entry_deck(
            driver, ordinal=ordinal, entry=exc, identity_band=identity_band,
            review_gate=review_gate, send_like=send_like)) from exc

    card_frames = [top_frame]
    staged_frames: list[tuple[bytes, str, int | None, str]] = [
        (top_frame, "card_scroll", None, datetime.now(timezone.utc).isoformat())]
    identity: ProfileIdentity | None = None
    index = None
    payload = None
    prior_index = None
    min_spacing = None
    target_scope_reason = "index not yet built"
    for _step in range(_MAX_CARD_SCROLLS):
        identity = capture_profile_identity(card_frames, identity_band=identity_band,
                                            grid=_IDENTITY_GRID)
        try:
            index = build_item_index(card_frames, content_band=content_band,
                                     like_template=like_template, like_threshold=like_threshold,
                                     at_scroll_top=True, identity_band=identity_band,
                                     _prefix_index=prior_index)
            prior_index = index
            # The automated strategy has one target only.  It does not need to claim that it
            # saw the lower, untouched profile tail; it needs a complete absolute prefix through
            # this exact photo plus identity and crop evidence strong enough for bottom-up
            # navigation and post-tap verification.  ``ItemIndex.complete`` remains mandatory
            # everywhere that constructs a production closed-set payload.
            # CLASSIFIER-ONLY NUMBERING, DELIBERATELY.  Numbering here decides WHICH item this
            # profile will navigate to; no heart is ever spent on its strength.
            # `_verified_still_photo_proof` is the sole still-photo licence for a heart, and it
            # takes REAL parked evidence in the pre-heart window, after navigation stops the
            # card.  Wiring the strict gate in here instead (e5054f7f) handed it
            # `still_photo_dwell=None`, whose dwell rung can never pass on read-scroll frames --
            # a read scroll separates every candidate from its next frame, so this loop has no
            # un-interacted dwell to offer -- so every candidate refused, every payload came back
            # unusable and every profile skipped (live failure 2026-08-22).
            payload = (build_item_payload(card_frames, index,
                                          unnumber=unnumber_unless_confident_photo)
                       if index.usable else None)
            target_scope_reason = (_target_scoped_prefix_reason(index, payload, target_items)
                                   if payload is not None else "photo-only payload unavailable")
            if target_scope_reason is None:
                break
        except (ItemIndexError, ItemCropError, SegmentationError, ShiftEstimationError) as exc:
            index = payload = None
            # Say what actually happened.  Leaving the initial placeholder in place meant a run
            # where EVERY iteration raised reported "index not yet built" as its skip detail --
            # a sentence describing the state before the loop started, not the failure the
            # operator has to fix.
            target_scope_reason = f"enumeration raised {type(exc).__name__} on this frame set"
        try:
            step, min_spacing = _plan_card_scroll(
                card_frames[-1], content_band=content_band, like_template=like_template,
                like_threshold=like_threshold, profile_min_spacing_px=min_spacing)
        except (SegmentationError, ScrollStepError) as exc:
            raise _CaptureAbort(f"automated profile {ordinal}: no alias-safe read scroll: {exc}") from exc
        time.sleep(human_delay(driver.dwell_s))
        driver._scroll_down_one(step.frac, step.x_frac)
        frame = _settled_read_scroll_frame(driver)
        card_frames.append(frame)
        staged_frames.append((frame, "card_scroll", None, datetime.now(timezone.utc).isoformat()))
    else:
        if identity is None or not identity.known:
            raise _CaptureAbort(
                f"automated profile {ordinal}: did not obtain an identity-bound target prefix "
                f"for {target_items}, and has no identity safe enough to skip")
        retry = _PreActionProfileRetry(
            "target_unavailable_or_incomplete_index",
            f"did not obtain a safe target-scoped photo prefix for {target_items} within "
            f"{_MAX_CARD_SCROLLS} bounded read scrolls ({target_scope_reason})")
        raise skip_before_heart(retry, identity)

    if identity is None or not identity.known or index is None or payload is None:
        raise _CaptureAbort(f"automated profile {ordinal}: index/identity unexpectedly unavailable")
    target_scope_reason = _target_scoped_prefix_reason(index, payload, target_items)
    if target_scope_reason is not None:
        raise _CaptureAbort(f"automated profile {ordinal}: target prefix changed before action: "
                            f"{target_scope_reason}")
    profile_id = "auto-" + _sha256(b"".join(card_frames[:3]))[:20]
    if profile_id in used_profile_ids:
        raise _CaptureAbort("automated capture encountered a duplicate profile fingerprint; refusing "
                            "to reuse a profile across the evidence split")

    action_trace: list[dict] = []
    for item_number in target_items:
        blocker_proof = _verified_blocker_absence(payload, item_number)
        if not blocker_proof.blocker_absent:
            retry = _PreActionProfileRetry(
                "target_verification_blocked",
                f"item {item_number}: {blocker_proof.blocker}")
            raise skip_before_heart(retry, identity)
        review_before = None
        adjustment_count = 0
        center_offsets_tried: list[float] = []

        def _apply_bounded_centering_correction(
                center_offset: float, *, target_pre: bytes, item_number: int,
                adjustment_count: int, center_offsets_tried: list[float]) -> int:
            """One bounded corrective read-scroll, charged against the shared adjustment
            budget, or the bounded pre-action skip once that budget is exhausted.  Returns the
            incremented adjustment count; the caller stores it back.

            Shared by both centring checks in the loop below -- the one taken right after
            navigation parks the card, and the one re-run after a still-photo probe's return
            leg leaves a measured page residual -- so a card corrected from either position is
            corrected exactly the same way, against exactly the same budget.  `target_pre`,
            `item_number`, `adjustment_count` and `center_offsets_tried` are explicit parameters
            rather than closed over: all four are reassigned somewhere in the loops this is
            called from, and closing over a name a loop reassigns is exactly the late-binding
            trap ruff's B023 exists to catch.
            """
            center_offsets_tried.append(center_offset)
            # SHARED with the reviewer's budget on purpose: one heart gets one bounded
            # allowance of visual adjustments, however they were requested.
            adjustment_count += 1
            if adjustment_count > _HYBRID_MAX_ADJUSTMENTS_PER_ACTION:
                # Name every offset that was tried.  "Refused for centring" alone cannot tell a
                # hopeless geometry from a correction that was converging and ran out of
                # budget, and those need opposite responses from the operator.
                tried = " then ".join(f"{value:.3f}" for value in center_offsets_tried)
                retry = _PreActionProfileRetry(
                    "target_verification_blocked",
                    f"item {item_number}: navigation could not park this card inside "
                    f"Hinge's autoplay trigger zone (card centre offset {tried} over "
                    f"{len(center_offsets_tried) - 1} corrective scroll(s), limit "
                    f"{STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC:.3f})")
                raise skip_before_heart(retry, identity)
            try:
                step, _ = _plan_card_scroll(
                    target_pre, content_band=content_band, like_template=like_template,
                    like_threshold=like_threshold, profile_min_spacing_px=None)
            except (SegmentationError, ScrollStepError) as exc:
                raise _CaptureAbort("centring correction could not be planned as a bounded "
                                    f"scroll: {exc}") from exc
            if center_offset < 0:
                # The card sits HIGH of the content centre, so the CONTENT has to come DOWN.
                driver._scroll_up_one(step.frac, step.x_frac)
            else:
                driver._scroll_down_one(step.frac, step.x_frac)
            # Never reuse a target point across a visual adjustment; re-navigate instead.
            driver.adb.screencap()
            return adjustment_count

        while True:
            try:
                heart_ordinal = payload.item(item_number).heart_ordinal
                navigation_index = index.translation.index(heart_ordinal) + 1
                target = navigate_to_item(
                    driver, index, navigation_index, entry_reference=card_frames[-1],
                    identity_match_max_dist=_UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST)
            except (ItemCropError, ValueError, ItemNavigationError, ShiftEstimationError,
                    SegmentationError, ScrollStepError) as exc:
                # A navigation entry refusal is decided from two exact frames: the last indexed
                # read position and the navigator's fresh entry capture.  Persist both privately
                # before the skip rewinds/advances the phone; without the pair, a real -500px
                # anchor can never be distinguished offline from a false shift caused by moving
                # media.  These are forensic diagnostics only and are never listed in the
                # evidence manifest.  Keep them beneath a private child directory rather than
                # at the session root: `_load_session` deliberately requires the root PNG set
                # to be exactly the manifest's evidence frames, so a best-effort diagnostic
                # must not turn an otherwise complete future session into an unloadable one.
                if isinstance(exc, ItemNavigationError) and exc.frame is not None:
                    # The one bounded fresh enumeration may itself reach the same refusal.  Its
                    # pair is independent evidence about a different scan, so never overwrite
                    # scan 1 with scan 2 just because both target the same profile/item.
                    scan_attempt = entry_drift_restart_attempts + 1
                    stem = (f"refused_navigation_p{ordinal}_item{item_number}"
                            f"_scan{scan_attempt}")
                    try:
                        forensics_dir = ensure_private_dir(out_dir / "forensics")
                        atomic_write_private_bytes(
                            forensics_dir / f"{stem}_read_reference.png", card_frames[-1],
                            parent=forensics_dir)
                        atomic_write_private_bytes(
                            forensics_dir / f"{stem}_entry.png", exc.frame,
                            parent=forensics_dir)
                        anchor = exc.anchor
                        atomic_write_private_text(
                            forensics_dir / f"{stem}.json",
                            json.dumps({
                                "kind": "navigation_refusal_forensic_v1",
                                "ordinal": ordinal,
                                "item_number": item_number,
                                "error_code": exc.code,
                                "error": str(exc),
                                "read_reference_sha256": _sha256(card_frames[-1]),
                                "entry_sha256": _sha256(exc.frame),
                                "anchor_delta_px": getattr(anchor, "delta_px", None),
                                "anchor_status": getattr(anchor, "status", None),
                                "anchor_confidence": getattr(anchor, "confidence", None),
                                "anchor_reason": getattr(anchor, "reason", None),
                                "calibration_evidence": False,
                            }, indent=2, sort_keys=True) + "\n",
                            parent=forensics_dir)
                        print("Wrote forensic (non-evidence) navigation pair: "
                              f"{forensics_dir / stem}")
                    except Exception as diagnostic_exc:  # noqa: BLE001 -- never mask refusal
                        print(f"Could not write navigation-refusal diagnostic: {diagnostic_exc}")
                # Do not rebase a large entry delta onto this index.  Even a unanimous
                # translation only proves two rendered frames correspond; it does not make the
                # old enumeration's frame sequence the current scan.  Before ANY action or
                # reviewer checkpoint, throw the entire local index/payload/staged-frame set
                # away and give this same profile one fresh top-to-bottom read.  A recurrence is
                # then routed through the existing bounded public-Like skip, never an unbounded
                # retry loop or a nearest-item fallback.
                large_measured_entry_drift = (
                    isinstance(exc, ItemNavigationError)
                    and exc.code == NAV_ANCHOR_UNMEASURED
                    and isinstance(getattr(exc.anchor, "delta_px", None), int))
                if large_measured_entry_drift and not action_trace:
                    if entry_drift_restart_attempts < (
                            _MAX_ENTRY_DRIFT_REENUMERATION_RESTARTS_PER_PROFILE):
                        raise _RestartProfile(
                            "measured pre-action navigation entry drift "
                            f"{exc.anchor.delta_px:+d}px; discarding the scan and taking the "
                            "one bounded fresh re-enumeration") from exc
                retry = _PreActionProfileRetry(
                    "pre_heart_navigation_refused",
                    f"item {item_number}: {type(exc).__name__}: {exc}")
                raise skip_before_heart(retry, identity) from exc
            target_pre = target.frame
            # Default to the navigator's own point/rows.  Both are overwritten below only when
            # the still-photo probe's return leg leaves a measured page residual; passing them
            # through unconditionally from here on keeps the residual == 0 path a no-op.
            tap_target_point = target.point
            moved_rows = target.block_frame_rows
            try:
                target_proof = _verified_target_frame_proof(
                    driver, target, frame=target_pre, content_band=content_band,
                    like_template=like_template, like_threshold=like_threshold)
            except _CaptureAbort as exc:
                retry = _PreActionProfileRetry(
                    "target_verification_blocked",
                    f"item {item_number}: {exc}")
                raise skip_before_heart(retry, identity) from exc
            # A POSITIONAL REFUSAL IS NOT A CONTENT REFUSAL, and only one of the two is
            # correctable.  The reviewer-directed SCROLL_UP/SCROLL_DOWN machinery below already
            # knows how to move a card and re-prove it from the new position -- but a
            # still-photo refusal raises straight past that machinery into the bounded profile
            # SKIP, before any checkpoint exists to direct an adjustment.  So a card that was
            # merely PARKED IN THE WRONG PLACE could never be corrected: the profile was Passed
            # and a real person spent.  Centring is the one rung of that ladder a bounded scroll
            # can fix, so it is measured HERE, from the exact rows the target proof just
            # re-proved by exact match on this frame, and BEFORE the proof spends its dwell
            # bursts and the two real gestures of its re-attach probe on a card that cannot
            # clear the rung anyway.
            # LIVE FAILURE 2026-08-22 (campaign attempt 5): navigation parked a depth-1 target
            # one read-scroll quantum low and it measured -0.229 against the 0.150 limit, while
            # the same card at true scroll top measures -0.109 -- inside the zone, one
            # corrective step away.  Navigation only promises a residual under about half a
            # quantum while a top-parked card can have as little as ~74px of zone margin, so
            # this is the ordinary case for depth-1 targets, not a rare one.
            # The correction RE-ENTERS THE LOOP rather than translating the stale rect or point:
            # `navigate_to_item` is the already-tested re-anchoring path, and rows, offset and
            # heart are all measured again from the corrected position.  That is also why an
            # overcorrection needs no clamp -- the deck stops at the top, and the next pass
            # simply measures whatever is really on the screen.
            center_offset = _parked_card_center_offset(
                target, target_proof, frame=target_pre, content_band=content_band)
            if (center_offset is not None
                    and abs(center_offset) > STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC):
                adjustment_count = _apply_bounded_centering_correction(
                    center_offset, target_pre=target_pre, item_number=item_number,
                    adjustment_count=adjustment_count, center_offsets_tried=center_offsets_tried)
                continue
            try:
                # The dwell runs HERE, inside the pre-heart window and after the card is parked
                # by navigation, because that is the only moment the card sits still with the
                # heart in view.  Its first burst is screencaps only; its re-attach probe DOES
                # move the screen on purpose.  A probe whose return leg leaves a measured
                # residual no longer refuses the card outright (2026-08-22, attempt 7): it hands
                # back the frame the page actually settled on, re-proved below.
                still_photo_proof = _verified_still_photo_proof(
                    driver, frame=target_pre, block=target_proof.block)
            except _CaptureAbort as exc:
                retry = _PreActionProfileRetry(
                    "target_verification_blocked",
                    f"item {item_number}: {exc}")
                raise skip_before_heart(retry, identity) from exc
            if still_photo_proof.action_frame != target_pre:
                # THE GATE IS THE BYTES CHANGING, NOT THE RESIDUAL.  `action_frame` can differ
                # from `target_pre` with a MEASURED residual of exactly 0: Android's status-bar
                # clock ticks a minute forward inside the probe's own dwell window often enough
                # that the probe's anchor routinely comes back byte-different from `target_pre`
                # in the chrome rows alone, with nothing in the content band having moved at all
                # (live 2026-08-22, campaign attempt 10). `_measured_page_shift` correctly
                # reports 0px for that case -- but every predicate below is read back against
                # the exact bytes a proof names by digest, and `action_frame_sha256` names
                # `action_frame`, never the stale `target_pre`.  So the frame every checkpoint
                # predicate binds to has to be rebound here whenever the bytes changed, at
                # residual 0 exactly as at any other residual, or `_verified_still_photo_evidence`
                # and `_screened_mute_control_absent` refuse a card that never actually failed
                # anything.  The residual itself is unchanged in meaning: it is how far the
                # navigator's own rows/point must be TRANSLATED, and it is only ever spent on that
                # translation when it is non-zero -- a zero residual means nothing moved, only the
                # chrome did, so rows/point are carried across UNCHANGED (owner rule 2026-08-11:
                # never a re-identified card).
                residual = still_photo_proof.page_residual_px
                target_pre = still_photo_proof.action_frame
                if residual:
                    moved_rows = (target.block_frame_rows[0] - residual,
                                 target.block_frame_rows[1] - residual)
                    tap_target_point = (target.point[0], target.point[1] - residual)
                try:
                    target_proof = _verified_target_frame_proof(
                        driver, target, frame=target_pre, content_band=content_band,
                        like_template=like_template, like_threshold=like_threshold,
                        expected_rows=moved_rows, expected_point=tap_target_point)
                except _CaptureAbort as exc:
                    retry = _PreActionProfileRetry(
                        "target_verification_blocked",
                        f"item {item_number}: {exc}")
                    raise skip_before_heart(retry, identity) from exc
                center_offset = _parked_card_center_offset(
                    target, target_proof, frame=target_pre, content_band=content_band,
                    rows=moved_rows)
                if (center_offset is not None
                        and abs(center_offset) > STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC):
                    adjustment_count = _apply_bounded_centering_correction(
                        center_offset, target_pre=target_pre, item_number=item_number,
                        adjustment_count=adjustment_count,
                        center_offsets_tried=center_offsets_tried)
                    continue
            if review_gate is None:
                break
            review_before = review_gate.checkpoint(
                target_pre, claimed_state="target_heart_visible",
                action_plan={
                    "action": "automated_photo_heart", "photo_model_item": item_number,
                    "point": list(tap_target_point), "point_source": "navigate_to_item",
                    # Every predicate below is the RETURN VALUE of the check that establishes it,
                    # read back through a function that refuses unless the proof binds this exact
                    # frame (or, for the blocker, this exact payload item).  None of them can be
                    # made True by editing this dict, which is the whole point: the hardcoded
                    # `photo_only_item_verified` that used to live here is how a real video came
                    # one approval from being hearted.
                    "predicates": {
                        # Evidence: the C1-C3 acceptance re-run on this frame plus TWO dwell
                        # bursts taken moments ago -- the second after the card was scrolled out
                        # of Hinge's autoplay band and back, so a stalled or unloaded video had
                        # to restart to stay hidden -- with every frame named by digest.
                        "positive_still_photo_evidence_verified":
                            _verified_still_photo_evidence(still_photo_proof, target_pre),
                        # Evidence: the verdict of the mute screen that ran on these exact
                        # bytes.  Reading it out of the proof refuses when nothing screened
                        # this frame, so the claim cannot outlive the screen behind it.
                        "target_frame_mute_control_screened_absent":
                            _screened_mute_control_absent(target_proof, target_pre),
                        "verification_blocker_absent":
                            _screened_verification_blocker_absent(
                                blocker_proof, payload, item_number),
                        "target_heart_visible":
                            _located_target_heart_visible(target_proof, target_pre),
                        "forbidden_zone_guarded_transport": "HingeDriver._tap",
                    },
                })
            if review_before["decision"] == "approved":
                break
            if review_before["decision"] == "restart_profile":
                if item_number == target_items[0]:
                    retry = _PreActionProfileRetry(
                        "reviewer_requested_restart_before_first_heart",
                        "reviewer requested RESTART_PROFILE before the first photo heart")
                    raise skip_before_heart(retry, identity)
                raise _CaptureAbort("reviewer requested restart after a prior heart; refusing to hide "
                                    "a real action by replaying the profile")
            if review_before["decision"] == "retry":
                adjustment_count += 1
            elif review_before["decision"] in {"scroll_up", "scroll_down"}:
                adjustment_count += 1
                try:
                    step, _ = _plan_card_scroll(
                        target_pre, content_band=content_band, like_template=like_template,
                        like_threshold=like_threshold, profile_min_spacing_px=None)
                except (SegmentationError, ScrollStepError) as exc:
                    raise _CaptureAbort("reviewer-requested bounded scroll could not be planned: "
                                        f"{exc}") from exc
                if review_before["decision"] == "scroll_up":
                    driver._scroll_up_one(step.frac, step.x_frac)
                else:
                    driver._scroll_down_one(step.frac, step.x_frac)
                # Never reuse a target point after a reviewer-directed visual adjustment.
                driver.adb.screencap()
            if adjustment_count > _HYBRID_MAX_ADJUSTMENTS_PER_ACTION:
                raise _CaptureAbort("reviewer exceeded the bounded hybrid adjustment budget for one "
                                    "heart; abort and modify/relaunch rather than continue blind")
        # External review authorizes one exact frame and point, not a coordinate that remains
        # valid indefinitely. Re-capture and re-run the structural/identity gates immediately
        # before the driver's foreground-guarded transport choke point.
        tap_point = tap_target_point
        if review_gate is not None:
            plan = review_before.get("action_plan") if isinstance(review_before, dict) else None
            tap_point = _fresh_reviewed_target_point(
                driver, target,
                reviewed_point=plan.get("point") if isinstance(plan, dict) else None,
                content_band=content_band, like_template=like_template,
                like_threshold=like_threshold, expected_point=tap_target_point,
                expected_frame=target_pre, expected_rows=moved_rows)
        # _tap is the driver's forbidden-zone/foreground-checked humanized transport chokepoint.
        driver._tap(*tap_point)
        time.sleep(human_delay(driver.dwell_s))
        composer_open = driver.adb.screencap()
        try:
            surface = locate_inline_composer(composer_open, confirm_template, threshold=0.8)
            if surface.layout_id != _COMPOSER_LAYOUT_ID:
                raise ComposerDetectionError(f"unsupported layout {surface.layout_id!r}")
            verdict = verify_sheet_item(composer_open, payload, item_number,
                                        composer_surface=surface)
            if not verdict.matched:
                raise SheetVerificationError(verdict.reason)
        except (ComposerDetectionError, SheetVerificationError) as exc:
            recovery = recover_unsent_composer(
                frame=composer_open, item_number=item_number,
                failure_stage="post_tap_composer_or_item_verification_refused")
            # The refusal text names a MEASURED geometry -- how many rows of the stored crop
            # the inline composer preview hid -- and that geometry is the only way to
            # re-derive `_INLINE_REFRAME_MAX_HIDDEN_FRACTION` after an app update changes the
            # composer's layout.  `composer_open` used to be discarded the instant this branch
            # raised, which is exactly what made the 2026-08-22 refusal (Hinge 10.0.1 hiding
            # 34.08% of a 1109px crop against the 27.86% limit measured on 9.134) unmeasurable
            # after the session ended.  Persist it as a forensic diagnostic ONLY -- it must
            # never be mistaken for calibration evidence -- best-effort, so a write failure
            # here can never mask the real refusal raised below.
            try:
                diagnostic_stem = f"refused_composer_p{ordinal}_item{item_number}"
                diagnostic_frame_path = out_dir / f"{diagnostic_stem}.png"
                ensure_private_dir(out_dir)
                atomic_write_private_bytes(diagnostic_frame_path, composer_open, parent=out_dir)
                diagnostic_record = {
                    "ordinal": ordinal,
                    "item_number": item_number,
                    "refusal": str(exc),
                    "frame_sha256": _sha256(composer_open),
                }
                try:
                    device_evidence = _device_evidence(driver)
                    diagnostic_record["hinge_version_name"] = device_evidence["hinge_version_name"]
                    diagnostic_record["frame_size_px"] = [
                        device_evidence["display_w"], device_evidence["display_h"]]
                except Exception:  # noqa: BLE001 -- diagnostic-only; omit rather than fail
                    pass
                try:
                    diagnostic_record["stored_crop_height_px"] = payload.item(item_number).height
                except Exception:  # noqa: BLE001 -- diagnostic-only; omit rather than fail
                    pass
                diagnostic_json_path = out_dir / f"{diagnostic_stem}.json"
                atomic_write_private_bytes(
                    diagnostic_json_path,
                    (json.dumps(diagnostic_record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                    parent=out_dir)
                print(f"Wrote forensic (non-evidence) refusal diagnostic: {diagnostic_frame_path}")
            except Exception:  # noqa: BLE001 -- diagnostic-only; never mask the real refusal
                pass
            raise _CaptureAbort(f"automated profile {ordinal} item {item_number}: post-tap "
                                f"composer/item verification refused: {exc}; abort cleanup "
                                f"outcome={recovery['outcome']}") from exc
        review_after = None
        if review_gate is not None:
            post_retries = 0
            while True:
                try:
                    review_after = review_gate.checkpoint(
                        composer_open, claimed_state="composer_and_item_verified",
                        action_plan={
                            "action": "review_heart_result", "photo_model_item": item_number,
                            "point": list(tap_point), "point_source": "completed HingeDriver._tap",
                            "predicates": {"inline_composer_verified": True,
                                           "post_tap_item_relative_verified": True,
                                           "send_like_tapped": False},
                        })
                except _CaptureAbort as exc:
                    recovery = recover_unsent_composer(
                        frame=composer_open, item_number=item_number,
                        failure_stage="post_tap_reviewer_checkpoint_aborted")
                    raise _CaptureAbort(
                        "hybrid reviewer stopped after a real heart; abort cleanup "
                        f"outcome={recovery['outcome']}") from exc
                if review_after["decision"] == "approved":
                    break
                if review_after["decision"] == "retry" and post_retries < _HYBRID_MAX_ADJUSTMENTS_PER_ACTION:
                    post_retries += 1
                    continue
                recovery = recover_unsent_composer(
                    frame=composer_open, item_number=item_number,
                    failure_stage="post_tap_reviewer_nonapproval")
                raise _CaptureAbort("reviewer did not approve the verified heart result; no further "
                                    "device action is allowed in this session; abort cleanup "
                                    f"outcome={recovery['outcome']}")
        captured = datetime.now(timezone.utc).isoformat()
        staged_frames.extend(((target_pre, "target_pre", item_number, captured),
                              (composer_open, "composer_open", item_number, captured)))
        action_trace.append({"action": "automated_photo_heart", "photo_model_item": item_number,
                             "transport": "HingeDriver._tap",
                             "pre_frame_sha256": _sha256(target_pre),
                             "post_frame_sha256": _sha256(composer_open),
                             "post_tap_composer_verified": True,
                             "post_tap_item_relative_verified": True,
                             "target_scoped_prefix_proof": {
                                 "id": _TARGET_SCOPED_PREFIX_PROOF_ID,
                                 "index_complete": bool(index.complete),
                                 "target_photo_model_item": item_number,
                                 "target_heart_ordinal": payload.item(item_number).heart_ordinal,
                                 "predecessors_resolved": True,
                                 "target_crop_complete": True,
                                 "identity_known": True,
                             },
                             "review_checkpoints": ({"before": review_before,
                                                     "after": review_after}
                                                    if review_gate is not None else None)})

    # A persistent inline composer hides the normal deck Like glyph, so production
    # `HingeDriver.dislike()` correctly refuses it.  Calibration instead has one narrow,
    # separately-audited route below.  Re-prove the exact composer/item before publishing the
    # reviewer checkpoint: a reviewer may request a bounded scroll and the checkpoint's claimed
    # state must not outrun what the current frame structurally establishes.
    pass_frame = driver.adb.screencap()
    pass_item_number = target_items[-1]
    try:
        pass_surface = _verified_automated_composer(
            pass_frame, confirm_template=confirm_template, payload=payload,
            item_number=pass_item_number)
    except _CaptureAbort as exc:
        recovery = recover_unsent_composer(
            frame=pass_frame, item_number=pass_item_number,
            failure_stage="pre_pass_composer_reverification_refused")
        raise _CaptureAbort("automated profile "
                            f"{ordinal}: composer could not be reverified before Pass: {exc}; "
                            f"abort cleanup outcome={recovery['outcome']}") from exc
    pass_review = None
    if review_gate is not None:
        pass_adjustments = 0
        while True:
            try:
                pass_review = review_gate.checkpoint(
                    pass_frame, claimed_state="composer_open_before_pass",
                    action_plan={
                        "action": ("automated_send_priority_like" if send_like
                                  else "automated_pass"),
                        "photo_model_item": None,
                        "point": (list(pass_surface.confirm_point) if send_like else None),
                        "point_source": ("calibration-only verified-composer Send transport"
                                        if send_like else
                                        "calibration-only verified-composer Pass transport"),
                        "predicates": {
                            "inline_composer_and_selected_photo_verified_before_action": True,
                            "send_like_tapped": False,
                            "forbidden_zone_guarded_transport": "HingeDriver._tap",
                        },
                    })
            except _CaptureAbort as exc:
                recovery = recover_unsent_composer(
                    frame=pass_frame, item_number=pass_item_number,
                    failure_stage="pre_pass_reviewer_checkpoint_aborted")
                raise _CaptureAbort(
                    "hybrid reviewer stopped before Pass/Send after a real heart; abort cleanup "
                    f"outcome={recovery['outcome']}") from exc
            if pass_review["decision"] == "approved":
                break
            if pass_review["decision"] == "retry":
                pass_adjustments += 1
            elif pass_review["decision"] in {"scroll_up", "scroll_down"}:
                pass_adjustments += 1
                try:
                    step, _ = _plan_card_scroll(
                        pass_frame, content_band=content_band, like_template=like_template,
                        like_threshold=like_threshold, profile_min_spacing_px=None)
                except (SegmentationError, ScrollStepError) as exc:
                    raise _CaptureAbort("reviewer-requested bounded Pass scroll could not be planned: "
                                        f"{exc}") from exc
                if pass_review["decision"] == "scroll_up":
                    driver._scroll_up_one(step.frac, step.x_frac)
                else:
                    driver._scroll_down_one(step.frac, step.x_frac)
                pass_frame = driver.adb.screencap()
                pass_surface = _verified_automated_composer(
                    pass_frame, confirm_template=confirm_template, payload=payload,
                    item_number=pass_item_number)
            else:
                recovery = recover_unsent_composer(
                    frame=pass_frame, item_number=pass_item_number,
                    failure_stage="pre_pass_reviewer_nonapproval")
                raise _CaptureAbort("reviewer requested profile restart after hearting; session must "
                                    "abort rather than disguise prior real actions; abort cleanup "
                                    f"outcome={recovery['outcome']}")
            if pass_adjustments > _HYBRID_MAX_ADJUSTMENTS_PER_ACTION:
                raise _CaptureAbort("reviewer exceeded bounded hybrid adjustment budget before Pass")
    try:
        if send_like:
            plan = pass_review.get("action_plan") if isinstance(pass_review, dict) else None
            reviewed_point = plan.get("point") if isinstance(plan, dict) else None
            if (not isinstance(reviewed_point, list) or len(reviewed_point) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in reviewed_point)):
                raise _CaptureAbort(
                    "automated Send refused: no exact reviewer-approved confirmation point is "
                    "bound to the terminal checkpoint")
            advance_frame, pass_trace = _automated_send_from_verified_composer(
                driver, frame=pass_frame, confirm_template=confirm_template, payload=payload,
                item_number=pass_item_number, reviewed_confirm_point=tuple(reviewed_point))
        else:
            advance_frame, pass_trace = _automated_pass_from_verified_composer(
                driver, frame=pass_frame, confirm_template=confirm_template, payload=payload,
                item_number=pass_item_number, identity=identity, identity_band=identity_band)
    except _CaptureAbort as exc:
        # The terminal helper can still refuse after the hybrid reviewer approves: most notably
        # its final, fresh-frame proof runs immediately before a real Send.  A pre-tap refusal
        # leaves the already-hearted profile's composer open and used to strand the next capture.
        # Re-read the *current* state (the terminal helper may already have acted) and delegate
        # only to the existing Pass-only abort recovery.  If Send/Pass already advanced, the
        # absence of a structurally proven composer makes recovery a no-input `not_cleared`.
        recovery_frame = driver.adb.screencap()
        recovery = recover_unsent_composer(
            frame=recovery_frame, item_number=pass_item_number,
            failure_stage=("terminal_send_refused" if send_like else "terminal_pass_refused"))
        raise _CaptureAbort(
            f"automated profile {ordinal}: terminal "
            f"{'Send' if send_like else 'Pass'} refused after a real heart: {exc}; "
            f"abort cleanup outcome={recovery['outcome']}") from exc
    staged_frames.append((advance_frame, "profile_advance_clear", None,
                          datetime.now(timezone.utc).isoformat()))
    pass_trace["review_checkpoints"] = ({"before": pass_review}
                                          if review_gate is not None else None)
    action_trace.append(pass_trace)

    # Make at most two bounded, planner-authorized scrolls solely to prove the next profile's
    # sticky identity, then use the driver's normal humanized rewind for the following profile.
    advance_identity = _scroll_next_profile_to_sticky_header(
        driver, top_frame=advance_frame, identity_band=identity_band,
        content_band=content_band, like_template=like_template,
        like_threshold=like_threshold, context=f"automated profile {ordinal}")
    advance_identity, distance, sticky_header_trace = _prove_post_advance_identity_distinct(
        driver, prior_fingerprint=identity.fingerprint, initial_frame=advance_identity,
        identity_band=identity_band, content_band=content_band,
        context=f"automated profile {ordinal}")
    staged_frames.append((advance_identity, "profile_advance_identity", None,
                          datetime.now(timezone.utc).isoformat()))
    action_trace.append({"action": "automated_sticky_header_proof", "transport": "HingeDriver._scroll_down_one",
                         "new_profile_identity_distance": distance,
                         **sticky_header_trace})

    profile_meta = {
        "ordinal": ordinal, "profile_id": profile_id, "card_scroll_frames": len(card_frames),
        "target_strategy_id": target_strategy_id,
        "composer_items": list(target_items), "profile_advance_cleared_composer": True,
        "profile_advance_identity_mismatched": True, "identity_frame_index": identity.frame_index,
        "identity_agreeing_frames": identity.agreeing_frames,
        "action_evidence_mode": (_HYBRID_CAPTURE_MODE if review_gate is not None
                                  else "automated_circular_risk_accepted"),
        "capture_evidence_scope": _TARGET_SCOPED_PREFIX_PROOF_ID,
        "automated_actions": action_trace,
    }
    frame_counter = _commit_profile_frames(out_dir, frames_meta, frame_counter, staged_frames,
                                           profile_ordinal=ordinal, profile_id=profile_id)
    used_profile_ids.add(profile_id)
    return profile_meta, frame_counter


def _capture_out_dir(raw: str | None, *, prefix: str = "targeting") -> Path:
    """Resolve a capture directory and keep private profile frames under ops/calibration/."""
    root = Path("ops/calibration").resolve()
    out_dir = (Path(raw) if raw else _default_out_dir(prefix)).resolve()
    try:
        out_dir.relative_to(root)
    except ValueError:
        print(
            f"ERROR: --out must stay under the gitignored {root}. Profile screenshots contain "
            "private personal data and cannot be written to a tracked or external directory "
            f"(got {out_dir}).", file=sys.stderr)
        sys.exit(1)
    if out_dir.exists() and not out_dir.is_dir():
        print(f"ERROR: capture output {out_dir} exists and is not a directory.", file=sys.stderr)
        sys.exit(1)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"ERROR: capture output directory {out_dir} already exists and is not empty. "
              "Use a fresh directory so two evidence sessions cannot be merged.",
              file=sys.stderr)
        sys.exit(1)
    ensure_private_dir(out_dir)
    return out_dir


def _existing_calibration_session_dir(raw: str) -> Path:
    """Resolve an existing private capture directory without creating or merging anything."""
    root = Path("ops/calibration").resolve()
    sess_dir = Path(raw).resolve()
    try:
        sess_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"session directory must stay under private {root}; got {sess_dir}") from exc
    if not sess_dir.is_dir():
        raise RuntimeError(f"session directory does not exist: {sess_dir}")
    return sess_dir


def _record_operational_checks(enabled: bool) -> dict:
    """Collect explicit evidence references for RUNBOOK's four supervised release checks."""
    result = {
        key: {"confirmed": False, "evidence": None, "recorded_utc": None}
        for key in _OPERATIONAL_CHECKS
    }
    if not enabled:
        return result
    print("\n=== Required supervised operational checks ===")
    print("These checks license real targeting behavior in addition to the numeric bounds. "
          "Perform each check as written in ops/RUNBOOK.md, then enter a nonempty local debug "
          "run / ledger reference. An assertion without a traceable reference is not evidence.")
    for key, description in _OPERATIONAL_CHECKS.items():
        print(f"\n{description}")
        evidence = ""
        while not evidence:
            evidence = input("  Evidence reference (local path/run id; blank is not accepted) > ").strip()
        result[key] = {
            "confirmed": True,
            "evidence": evidence,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        }
    return result


def _canonical_json_digest(value) -> str:
    """Hash structured evidence without retaining profile pixels, text, or API responses."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    return _sha256(payload.encode("utf-8", "replace"))


def _config_provenance(config_path: str, cfg) -> dict:
    """Hash exact config bytes plus the effective Hinge mapping captured under them.

    The byte digest makes unattended evidence non-portable across even a later configuration
    edit.  The canonical effective mapping gives an auditor a stable, secret-free comparison
    target without putting the full config (which may contain paths or credentials) in a private
    profile manifest.
    """
    path = Path(config_path).resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read capture config for provenance {path}: {exc}") from exc
    hinge_cfg = (getattr(cfg, "apps", {}) or {}).get("hinge", {})
    return {
        "schema_version": _UNATTENDED_PROVENANCE_SCHEMA_VERSION,
        "path": str(path),
        "sha256": _sha256(raw),
        "effective_hinge_sha256": _canonical_json_digest(hinge_cfg),
    }


def _event_from_frame(name: str, frame: bytes, *, started_monotonic: float) -> dict:
    """Return a privacy-preserving, timestamped observation of a local frame.

    This deliberately records only a SHA-256 digest.  An observe-check establishes that the
    passive path saw the expected state transitions; it is not a calibration capture and must
    never create another local collection of a person's profile screenshots.
    """
    captured = datetime.now(timezone.utc)
    return {
        "event": name,
        "recorded_utc": captured.isoformat(),
        "elapsed_s": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "frame_sha256": _sha256(frame),
        "frame_bytes": len(frame),
    }


def _synthetic_gemini_quota_probe(cfg) -> dict:
    """Issue one non-personal GenerateContent request and return hashed outcome evidence.

    This is intentionally *not* an OpenerService request: it contains no Profile, ItemRequest,
    screenshot, opener prompt, or model output exposed to the hub.  It exercises exactly one
    configured Gemini endpoint so a supervised observe check can document quota/refusal timing
    before targeting calibration has licensed real profile data.
    """
    opener = cfg.opener
    if not opener.enabled or opener.provider != "gemini":
        raise RuntimeError("observe-check requires enabled opener.provider: gemini for its synthetic quota probe")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for observe-check's synthetic quota probe")
    models = opener.effective_models
    if not models:
        raise RuntimeError("observe-check found no configured Gemini model")

    # Construct only to reuse the production credential/timeout transport.  Do not call
    # `generate`: that method accepts a Profile and is therefore deliberately out of bounds.
    client = GeminiOpener(models, max_tokens=16, request_timeout_s=opener.request_timeout_s,
                          api_key=os.environ["GEMINI_API_KEY"], thinking={})
    model = models[0]
    payload = {
        "contents": [{"role": "user", "parts": [{"text": (
            "Synthetic Operation Love observe-only quota check. Return exactly "
            "{\"probe\":\"ok\"}. This request contains no dating-profile data.") }]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 16,
            "responseMimeType": "application/json",
        },
    }
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        code, response = client.transport(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent",
            payload,
            {"Content-Type": "application/json", "X-goog-api-key": client.api_key},
            opener.request_timeout_s,
        )
        code = int(code)
        outcome = "ok" if 200 <= code < 300 else "quota_refused" if code == 429 else "http_error"
        return {
            "kind": "synthetic_non_personal_gemini_generate_content",
            "model": model,
            "request_sha256": _canonical_json_digest(payload),
            "started_utc": started_utc,
            "elapsed_s": round(max(0.0, time.monotonic() - started), 3),
            "outcome": outcome,
            "http_code": code,
            "response_sha256": _canonical_json_digest(response),
        }
    except Exception as exc:  # noqa: BLE001 -- evidence must record a provider refusal safely
        return {
            "kind": "synthetic_non_personal_gemini_generate_content",
            "model": model,
            "request_sha256": _canonical_json_digest(payload),
            "started_utc": started_utc,
            "elapsed_s": round(max(0.0, time.monotonic() - started), 3),
            "outcome": "transport_or_client_error",
            "error_type": type(exc).__name__,
            # Provider strings can sometimes contain operational identifiers. Keep only a hash.
            "error_sha256": _sha256(str(exc).encode("utf-8", "replace")),
        }


def _write_observe_check_evidence(out_dir: Path, evidence: dict) -> Path:
    """Atomically publish one self-hashing, local-only observe-check artifact."""
    body = dict(evidence)
    body.pop("evidence_sha256", None)
    body["evidence_sha256"] = _canonical_json_digest(body)
    path = out_dir / "observe_check.json"
    atomic_write_private_text(
        path, json.dumps(body, indent=2, sort_keys=True) + "\n", parent=out_dir)
    return path


def _require_operator_event(prompt: str, token: str) -> None:
    """Make every claimed manual action an explicit, local operator attestation."""
    entered = input(prompt).strip()
    if entered != token:
        raise _CaptureAbort(
            f"operator did not enter {token!r}; refusing to infer a manual action or continue")


def _cmd_observe_check(args: argparse.Namespace) -> None:
    """Record a preliminary supervised passive cycle without authorizing targeting.

    The tool only reads screenshots around actions performed by the owner.  It never calls the
    Worker, OpenerService, a profile-generating method, or a driver gesture/text method.  In
    particular it disables Hinge enumeration before opening the session: no provisional state
    can become a substitute for a measured targeting calibration or the later production OBSERVE
    validation that must happen before AUTO.
    """
    if not getattr(args, "supervised", False) or args.confirmation != _OBSERVE_CHECK_CONFIRMATION:
        print("ERROR: observe-check is available only with --supervised --confirmation "
              f"{_OBSERVE_CHECK_CONFIRMATION!r}.", file=sys.stderr)
        sys.exit(2)

    cfg = cfg_mod.load(args.config)
    # Validate for the same reason capture does (see _cmd_capture): the process-local
    # still-photo licence only exists after config.validate(), and this command opens a real
    # driver session whose behaviour must match a production process, not an unvalidated one.
    try:
        cfg_mod.validate(cfg)
    except Exception as exc:  # noqa: BLE001 -- any invalid config is fatal before the device
        print(f"ERROR: config.validate() refused {args.config}: {exc}", file=sys.stderr)
        sys.exit(1)
    # Same read-only exact-device preflight as a calibration capture.  `open_session` only
    # proves the configured serial is present; evidence must prove it was the one device.
    _preflight_serial(cfg)
    out_dir = _capture_out_dir(args.out, prefix="observe_check")
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    events: list[dict] = []
    evidence: dict = {
        "schema_version": _OBSERVE_CHECK_SCHEMA_VERSION,
        "kind": "hinge_supervised_observe_only_check",
        "evidence_scope": _PRELIMINARY_OBSERVE_SCOPE,
        "release_status": "preliminary_only_not_auto_authorization",
        "does_not_prove": [
            "production_worker_or_hub_pre_tap_timing",
            "post_tap_target_item_verification",
            "real_profile_provider_request_or_opener_text",
            "production_store_label_persistence",
            "production_refusal_or_paywall_logging",
        ],
        "required_before_auto": list(_POST_CALIBRATION_OBSERVE_REQUIREMENTS),
        "started_utc": started_utc,
        "passive_guards": {
            "worker_constructed": False,
            "opener_service_called": False,
            "profile_or_item_request_sent": False,
            "driver_gesture_text_or_send_called": False,
            "enumeration_enabled": False,
            "production_store_persistence_checked": False,
            "hub_pre_tap_state_checked": False,
        },
        "events": events,
    }
    driver = HingeDriver(cfg)
    complete = False
    try:
        # This must precede open_session: current Hinge code consults the flag while deciding
        # whether an uncalibrated card should be enumerated.
        disable_enumeration = getattr(driver, "set_opener_enabled", None)
        if callable(disable_enumeration):
            disable_enumeration(False)
        driver.open_session()
        device = _device_evidence(driver)
        evidence["device"] = {
            key: device[key] for key in ("serial", "model", "display_w", "display_h",
                                         "hinge_package", "hinge_version_name")
        }
        confirm_template = driver._template("confirm")
        if confirm_template is None:
            raise RuntimeError("Hinge confirm template is unavailable; cannot prove inline composer")

        # The single provider call is deliberately synthetic and precedes any operator tap.
        evidence["quota_probe"] = _synthetic_gemini_quota_probe(cfg)
        if evidence["quota_probe"]["outcome"] not in {"ok", "quota_refused"}:
            raise RuntimeError("synthetic Gemini quota probe did not receive a usable response")

        print("OBSERVE-CHECK is preliminary and passive. The tool will not tap, swipe, type, "
              "send, construct Worker/hub/store, generate an opener, enumerate profile items, "
              "or transmit profile data. It cannot prove production OBSERVE timing or label "
              "persistence and cannot authorize AUTO. Perform all phone actions yourself; only "
              "SHA-256 frame digests are written locally.")
        _require_operator_event(
            "With a normal profile visible, type READY immediately before your manual PASS > ",
            "READY")
        events.append(_event_from_frame("before_manual_pass", driver.adb.screencap(),
                                        started_monotonic=started_monotonic))
        _require_operator_event("Manually PASS now, then type PASSED once the next card is stable > ",
                                "PASSED")
        events.append(_event_from_frame("after_manual_pass", driver.adb.screencap(),
                                        started_monotonic=started_monotonic))

        _require_operator_event(
            "On the next profile, type READY immediately before manually hearting a PHOTO > ",
            "READY")
        events.append(_event_from_frame("before_manual_heart", driver.adb.screencap(),
                                        started_monotonic=started_monotonic))
        _require_operator_event(
            "Manually heart that PHOTO now; type COMPOSER when Send Like is visibly open > ",
            "COMPOSER")
        composer_frame = driver.adb.screencap()
        surface = locate_inline_composer(composer_frame, confirm_template, threshold=0.8)
        events.append({
            **_event_from_frame("after_manual_heart_inline_composer", composer_frame,
                                started_monotonic=started_monotonic),
            "composer_layout_id": surface.layout_id,
            "comment_rect": [surface.comment_rect.x0, surface.comment_rect.y0,
                             surface.comment_rect.x1, surface.comment_rect.y1],
            "send_rect": [surface.send_rect.x0, surface.send_rect.y0,
                          surface.send_rect.x1, surface.send_rect.y1],
            "confirm_point": list(surface.confirm_point),
        })

        _require_operator_event(
            "Type READY immediately before manually pressing Send Like (or observing its refusal) > ",
            "READY")
        events.append(_event_from_frame("before_manual_send", driver.adb.screencap(),
                                        started_monotonic=started_monotonic))
        send_outcome = input(
            "Manually press Send Like once, then type SENT when it resolves (or REFUSED for a "
            "visible quota/paywall refusal) > ").strip()
        if send_outcome not in {"SENT", "REFUSED"}:
            raise _CaptureAbort("operator did not enter 'SENT' or 'REFUSED'; refusing completion")
        after_send = driver.adb.screencap()
        try:
            locate_inline_composer(after_send, confirm_template, threshold=0.8)
        except ComposerDetectionError:
            composer_cleared = True
        else:
            composer_cleared = False
        events.append({
            **_event_from_frame("after_manual_send", after_send,
                                started_monotonic=started_monotonic),
            "operator_outcome": send_outcome.lower(),
            "inline_composer_cleared": composer_cleared,
            "deck_ready": bool(driver._observe_deck_ready(after_send)),
        })
        if not composer_cleared:
            raise RuntimeError("inline composer remains visible after the claimed manual send; refusing completion")
        if send_outcome == "SENT" and not events[-1]["deck_ready"]:
            raise RuntimeError("claimed manual send did not reach a confirmed ready deck")
        if send_outcome == "REFUSED":
            blocked = getattr(driver, "blocked_reason", lambda: None)()
            events[-1]["blocked_reason_sha256"] = (
                _sha256(str(blocked).encode("utf-8", "replace")) if blocked else None)
            if not blocked:
                raise RuntimeError("claimed refusal/paywall was not affirmatively recognized by the driver")
        complete = True
    except (_CaptureAbort, ComposerDetectionError, RuntimeError) as exc:
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message_sha256": _sha256(str(exc).encode("utf-8", "replace")),
        }
        print(f"OBSERVE-CHECK REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        try:
            driver.close()
        finally:
            evidence["completed"] = complete
            evidence["ended_utc"] = datetime.now(timezone.utc).isoformat()
            evidence["elapsed_s"] = round(max(0.0, time.monotonic() - started_monotonic), 3)
            path = _write_observe_check_evidence(out_dir, evidence)
            print(f"Wrote local hashed observe-check evidence to {path}")
            print("NEXT REQUIRED BEFORE AUTO: install the measured calibration for a supervised "
                  "production OBSERVE-only run, then record Worker/hub pre-tap, post-tap item "
                  "verification, real provider/quota, store persistence, and paywall/refusal "
                  "evidence. This preliminary artifact cannot satisfy that release check.")
    if not complete:
        sys.exit(1)


def _cmd_capture(args: argparse.Namespace) -> None:
    if args.profiles < 1:
        print("ERROR: --profiles must be >= 1.", file=sys.stderr)
        sys.exit(1)

    unattended = bool(getattr(args, "unattended", False))
    hybrid_review = bool(getattr(args, "hybrid_review", False))
    automated = unattended or hybrid_review
    if unattended and hybrid_review:
        print("ERROR: choose only one of --unattended or --hybrid-review.", file=sys.stderr)
        sys.exit(1)
    if unattended and args.confirmation != _UNATTENDED_CONFIRMATION:
        print("ERROR: --unattended requires the exact --confirmation "
              f"{_UNATTENDED_CONFIRMATION!r}. It authorizes real automated hearts and Passes, "
              "and is intentionally not enabled by a short flag alone.", file=sys.stderr)
        sys.exit(1)
    if hybrid_review and args.confirmation != _HYBRID_REVIEW_CONFIRMATION:
        print("ERROR: --hybrid-review requires the exact --confirmation "
              f"{_HYBRID_REVIEW_CONFIRMATION!r}. It authorizes real automated hearts and Passes "
              "only after a reviewer approves each private frame checkpoint.", file=sys.stderr)
        sys.exit(1)
    if hybrid_review and (not str(getattr(args, "reviewer_model", "")).strip()
                           or not str(getattr(args, "reviewer_process", "")).strip()):
        print("ERROR: --hybrid-review requires nonempty --reviewer-model and --reviewer-process "
              "so the manifest does not pretend its AI review provenance is anonymous.", file=sys.stderr)
        sys.exit(1)
    if automated and getattr(args, "record_operational_checks", False):
        print("ERROR: automated capture cannot record supervised operational checks. Its manifest "
              "will label device actions and circular-risk acceptance explicitly.", file=sys.stderr)
        sys.exit(1)

    send_like = bool(getattr(args, "send_like", False))
    if send_like and not hybrid_review:
        # Deliberately narrower than the other automated flags. A real Send Priority Like is
        # permanent and cannot be retracted, so every one of them must pass a reviewer checkpoint
        # first; fully unattended sending would deliver real likes with nothing in the loop. This
        # also keeps the evidence path consistent: `_verified_automated_circular_evidence` (the
        # --unattended validator) only authenticates the Pass-without-send terminal action, so an
        # unattended send capture could never be measured anyway.
        print("ERROR: --send-like requires --hybrid-review. Real Send Priority Likes are "
              "permanent, so each one must be approved at a reviewer checkpoint; --unattended "
              "has no reviewer in the loop.", file=sys.stderr)
        sys.exit(1)
    if send_like and getattr(args, "send_like_confirmation", None) != _SEND_LIKE_CONFIRMATION:
        print("ERROR: --send-like requires the exact --send-like-confirmation "
              f"{_SEND_LIKE_CONFIRMATION!r}, separate from --confirmation. It replaces every "
              "calibration Pass with a REAL, PERMANENT Send Priority Like and is intentionally "
              "not enabled by a short flag alone.", file=sys.stderr)
        sys.exit(1)

    target_items_flag = getattr(args, "target_items", "alternate-1-3")
    try:
        target_strategy_id = _TARGET_ITEMS_FLAG_STRATEGY_IDS[target_items_flag]
    except KeyError:
        print(f"ERROR: unknown --target-items {target_items_flag!r}; choose one of "
              f"{sorted(_TARGET_ITEMS_FLAG_STRATEGY_IDS)}.", file=sys.stderr)
        sys.exit(1)
    photo_1_only = target_strategy_id == _PHOTO_1_ONLY_TARGET_STRATEGY_ID
    target_items_description = (
        "photo model item 1 on EVERY profile -- deep-item navigation is NOT exercised; this is "
        "an explicit narrowing of the evidence (--target-items photo-1-only)"
        if photo_1_only else
        "photo model item 1 on odd profile ordinals and item 3 on even ordinals")

    if not automated:
        print("HUMANIZED, NARROW-SCOPE: this tool only ever screencaps (read-only) and performs "
              "small humanized SCROLL gestures through the driver's own transport -- never a raw "
              "`adb shell input tap`/`input swipe`. It NEVER taps a heart, NEVER dismisses or "
              "sends a comment, and NEVER types anything: every one of those is done BY YOU, "
              "by hand, on the phone, when prompted.")
    if unattended:
        print("UNATTENDED / CIRCULAR-RISK ACCEPTED: this run will use HingeDriver's guarded "
              f"humanized tap/dislike transport to heart {target_items_description}, verify the "
              "inline composer structurally, then Pass without sending. The manifest records "
              "human_ground_truth=false and cannot stand in for supervised operational evidence.")
    if hybrid_review:
        print("HYBRID / AI-REVIEWED AUTOMATION: guarded HingeDriver transport will heart "
              f"{target_items_description}, then "
              f"{'send a REAL Priority Like' if send_like else 'Pass without sending'}, but only "
              "after a private PNG+JSON checkpoint is reviewed through stdin before every "
              "heart/Pass-or-Send and after every heart result. EOF, REFUSE, ABORT, or malformed "
              "input stops cleanly.")
    if send_like:
        print("SEND-LIKE ENABLED: every profile captured this run ends with a REAL, PERMANENT "
              "Send Priority Like instead of Pass -- owner-directed exception to this tool's "
              "default never-send design, requested because this account runs unlimited HingeX "
              "likes. Uses the exact same tap/upsell-dismiss/landed-verification as a normal "
              "AUTO/OBSERVE like.")
    print("Frames are LOCAL-ONLY: ops/calibration/ is gitignored -- these are real people's "
          "dating profiles, so do not upload, copy outside the repo, or transmit them anywhere.")

    cfg = cfg_mod.load(args.config)
    # VALIDATE, not just load (found live 2026-08-22, attempt 3 of the first campaign): the
    # still-photo licence is installed process-locally by config.validate() and by NOTHING
    # else, and `_verified_still_photo_proof`'s ladder consults it through
    # `hinge_targeting_unavailable_reason()`.  A capture that only loads runs the entire read,
    # numbering, and navigation flawlessly and then has the pre-heart proof refuse every heart
    # at the policy rung -- fail-closed, correct, and utterly indistinguishable from a licence
    # problem in config.yaml until the skip detail names the rung.  Validation failures are
    # config problems, so they get the tool's standard fatal treatment.
    try:
        cfg_mod.validate(cfg)
    except Exception as exc:  # noqa: BLE001 -- any invalid config is fatal before the device
        print(f"ERROR: config.validate() refused {args.config}: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        config_provenance = _config_provenance(args.config, cfg) if automated else None
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    driver = HingeDriver(cfg)
    # Calibration capture drives its own guarded scroll/heart/Pass lifecycle; it is neither
    # the retired passive Observe loop nor a Worker ranker run.  Mark its session as
    # device-driven before `open_session()` so the driver's final platform preflight uses a
    # supported live mode.  The policy is deliberately `None`: this harness supplies no
    # ranker decision and keeps its separate, checkpoint-bound calibration protocol.
    #
    # Keep the capability check soft for the deliberately small fake drivers used by offline
    # capture-manifest tests.  Real `HingeDriver` instances always provide the method.
    session_policy = getattr(driver, "set_auto_session_policy", None)
    if callable(session_policy):
        session_policy(None)
    if driver.identity_band is None or driver.content_band is None:
        print("ERROR: this app's effective identity_band/content_band is None (no band "
              "declared, or an override disabled it). Targeting calibration is meaningless "
              "without both.", file=sys.stderr)
        sys.exit(1)
    identity_band, content_band = driver.identity_band, driver.content_band
    print(f"Effective identity_band={identity_band}  content_band={content_band}  "
          "(config-merged; read from driver.identity_band/content_band, never HINGE_SPEC's own "
          "defaults)")

    serial, _adb_path = _preflight_serial(cfg)
    out_dir = _capture_out_dir(args.out)

    print(f"\nConnecting to {serial} over ADB…")
    try:
        driver.open_session()
    except Exception as exc:  # noqa: BLE001 -- report clearly, exit non-zero, never retry blind
        print(f"ERROR: open_session() raised {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    frames_meta: list[dict] = []
    profiles_meta: list[dict] = []
    skipped_attempts: list[dict] = []
    # Abort cleanup is forensic transport accounting, never a successful measurement profile.
    # Keep it outside ``profiles`` so a later reviewer/measure command cannot mistake a
    # post-verification refusal for a calibrated composer pair.
    abort_recoveries: list[dict] = []
    frame_counter = 0
    start_utc = datetime.now(timezone.utc)
    interrupted = False
    evidence: dict | None = None
    review_gate: _HybridReviewGate | None = None
    entry_anchor_ledger: dict | None = None
    operational_checks = _record_operational_checks(False)
    try:
        try:
            evidence = _device_evidence(driver)
        except Exception as exc:  # noqa: BLE001 -- report clearly, exit non-zero
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Device: {evidence['model']} serial={evidence['serial']} "
              f"{evidence['display_w']}x{evidence['display_h']} density={evidence['density']} "
              f"Hinge versionName={evidence['hinge_version_name']}")
        if hybrid_review:
            if config_provenance is None:
                raise _CaptureAbort(
                    "hybrid review cannot start without hash-bound config provenance")
            review_gate = _HybridReviewGate(
                out_dir, device=evidence, config_provenance=config_provenance,
                reviewer_model=getattr(args, "reviewer_model", _HYBRID_REVIEW_SOURCE),
                reviewer_process=getattr(args, "reviewer_process", "stdin_checkpoint_protocol"),
                reviewer_id=getattr(args, "reviewer_id", None),
                reviewer_version=getattr(args, "reviewer_version", None))
        print(f"\nCapturing split={args.split!r}, {args.profiles} profile(s) -> {out_dir}\n")

        try:
            ordinal = 0
            used_profile_ids: set[str] = set()
            # Kept per currently-read real profile attempt, not per requested output ordinal:
            # after a normal pre-action skip the next deck profile may use the same ordinal and
            # is entitled to its own one fresh re-enumeration.
            entry_drift_restarts: dict[int, int] = {}
            while ordinal < args.profiles:
                ordinal += 1
                print(f"\n=== Profile {ordinal}/{args.profiles} ({args.split}) ===")
                capture_fn = (_capture_one_profile_unattended if automated
                              else _capture_one_profile)
                try:
                    if automated:
                        automated_kwargs = {
                            "review_gate": review_gate,
                            "skipped_attempts": skipped_attempts,
                            "abort_recoveries": abort_recoveries,
                            "send_like": send_like,
                            "target_strategy_id": target_strategy_id,
                        }
                        restart_attempts = entry_drift_restarts.get(ordinal, 0)
                        # Keep pre-existing alternate capture callables compatible on the
                        # ordinary first attempt; the real recorder's default is also zero.
                        if restart_attempts:
                            automated_kwargs["entry_drift_restart_attempts"] = restart_attempts
                        profile_meta, frame_counter = capture_fn(
                            driver, out_dir, ordinal=ordinal, frame_counter=frame_counter,
                            frames_meta=frames_meta, used_profile_ids=used_profile_ids,
                            **automated_kwargs)
                    else:
                        profile_meta, frame_counter = capture_fn(
                            driver, out_dir, ordinal=ordinal, frame_counter=frame_counter,
                            frames_meta=frames_meta, used_profile_ids=used_profile_ids)
                except _RestartProfile as exc:
                    restart_attempts = entry_drift_restarts.get(ordinal, 0) + 1
                    if restart_attempts > _MAX_ENTRY_DRIFT_REENUMERATION_RESTARTS_PER_PROFILE:
                        raise _CaptureAbort(
                            f"automated profile {ordinal}: exceeded the bounded "
                            f"{_MAX_ENTRY_DRIFT_REENUMERATION_RESTARTS_PER_PROFILE}-restart "
                            "entry-drift re-enumeration limit") from exc
                    entry_drift_restarts[ordinal] = restart_attempts
                    print(f"Restarting profile {ordinal} from a driver-owned top: {exc}")
                    ordinal -= 1
                    continue
                except _ProfileSkipped as exc:
                    entry_drift_restarts.pop(ordinal, None)
                    prior_for_ordinal = sum(
                        1 for attempt in skipped_attempts if attempt.get("ordinal") == ordinal)
                    record = dict(exc.record)
                    record["attempt_number_for_ordinal"] = prior_for_ordinal + 1
                    record["session_skip_number"] = len(skipped_attempts) + 1
                    skipped_attempts.append(record)
                    # Print the detail, not only the code: the operator watching this run is
                    # the person who has to decide whether the deck, the target depth, or the
                    # app itself is the problem, and the code is the same string for every
                    # cause in its class.
                    advance_description = (
                        "after advancing it with a verified public Like"
                        if record.get("action") == "advance_unusable_profile_with_priority_like"
                        else "before any heart")
                    retry_description = (
                        "the next proven ordinary deck"
                        if record.get("reason_code") == "unsupported_entry_layout"
                        else "the distinct next profile")
                    print(
                        f"Skipped unsuitable calibration evidence {advance_description} for "
                        f"ordinal {ordinal} ({record['reason_code']}: "
                        f"{record['reason_detail']}); retrying this ordinal on "
                        f"{retry_description}.")
                    ordinal -= 1
                    continue
                profiles_meta.append(profile_meta)
                entry_drift_restarts.pop(ordinal, None)
            # The closed-set replay is intentionally not generated for automated target-scoped
            # evidence.  It proves every photo in a complete profile, a stronger claim that a
            # prefix run expressly does not make.  Each actual target was instead navigated by
            # production ``navigate_to_item`` immediately before its guarded tap, and the
            # trace/independent review bind that exact target-only proof.  Supervised captures
            # remain closed-set and retain their offline all-item ledger unchanged.
            if not automated:
                entry_anchor_ledger = _write_entry_anchor_ledger(
                    out_dir, frames_meta=frames_meta, profiles_meta=profiles_meta,
                    identity_band=identity_band, content_band=content_band,
                    like_template=driver._template("like"), like_threshold=hinge_mod._LIKE_MATCH_THRESHOLD)
                print(f"Verified offline entry-anchor replay: {out_dir / entry_anchor_ledger['file']}")
            operational_checks = _record_operational_checks(
                bool(getattr(args, "record_operational_checks", False)))
        except _CaptureAbort as exc:
            print(f"\nAborting capture: {exc}", file=sys.stderr)
            interrupted = True
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted by Ctrl-C.")
        except Exception as exc:  # noqa: BLE001 -- surface clearly, stop, never guess
            print(f"\nERROR during capture: {type(exc).__name__}: {exc}", file=sys.stderr)
            interrupted = True
    finally:
        driver.close()

    end_utc = datetime.now(timezone.utc)
    manifest = {
        "tool_version": _TOOL_VERSION,
        "calibration_schema_version": _CALIBRATION_SCHEMA_VERSION,
        "composer_layout_id": _COMPOSER_LAYOUT_ID,
        "item_selection_policy_id": PHOTO_ONLY_POLICY_ID,
        "frame_size_px": [evidence["display_w"], evidence["display_h"]] if evidence else None,
        "split": args.split,
        "capture_mode": (_HYBRID_CAPTURE_MODE if hybrid_review
                         else "automated_circular_risk_accepted" if unattended else "supervised_manual"),
        "automated_target_strategy_id": (target_strategy_id if automated else None),
        "capture_evidence_scope": (_TARGET_SCOPED_PREFIX_PROOF_ID if automated else "closed_set_profile_v1"),
        "human_ground_truth": not automated,
        "unattended_provenance_schema_version": (
            _UNATTENDED_PROVENANCE_SCHEMA_VERSION if automated else None),
        "config_provenance": config_provenance,
        "automation_acceptance": ({
            "confirmation": (_HYBRID_REVIEW_CONFIRMATION if hybrid_review
                             else _UNATTENDED_CONFIRMATION),
            "accepted_utc": start_utc.isoformat(),
            "target_strategy_id": target_strategy_id,
            "targets_photo_model_items": sorted(_automated_target_depths_for_strategy(target_strategy_id)),
            "provisional_identity_match_max_dist": _UNATTENDED_PROVISIONAL_IDENTITY_MAX_DIST,
            "not_independent_ground_truth": True,
            "not_supervised_operational_evidence": True,
            # Owner-accepted real sends replace this capture's Pass-without-send terminal action.
            # Recorded here (not inferred downstream) so `hinge_calibration_review`/`measure` can
            # gate on an explicit acceptance instead of silently tolerating a send trace, and so
            # the evidence never reads as Pass-only when real likes were actually delivered.
            "send_like_accepted": send_like,
            "send_like_confirmation": (_SEND_LIKE_CONFIRMATION if send_like else None),
            "terminal_advance_action": ("automated_send_priority_like" if send_like
                                        else "automated_pass"),
            "reviewer_protocol": ("stdin_checkpoint_sha256_v1" if review_gate is not None
                                  else None),
            "reviewer_source": (_HYBRID_REVIEW_SOURCE if review_gate is not None else None),
            "reviewer": (review_gate.reviewer if review_gate is not None else None),
        } if automated else None),
        "hybrid_review": ({
            "schema_version": _HYBRID_CHECKPOINT_SCHEMA_VERSION,
            "protocol": "stdin_checkpoint_sha256_v1",
            "reviewer": review_gate.reviewer,
            "decisions": review_gate.decisions,
            "human_ground_truth": False,
        } if review_gate is not None else None),
        "config_path": str(args.config),
        "device": evidence,
        "identity_band": list(identity_band),
        "content_band": list(content_band),
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "interrupted": interrupted,
        "requested_profiles": args.profiles,
        "profiles": profiles_meta,
        # Explicitly excluded from the measurement profile set: these records authenticate a
        # guarded advance only, never an item/composer observation.
        "skipped_attempts": skipped_attempts,
        "abort_recoveries": abort_recoveries,
        "frame_count": len(frames_meta),
        "frames": frames_meta,
        "entry_anchor_ledger": entry_anchor_ledger,
        "operational_checks": operational_checks,
    }
    atomic_write_private_text(
        out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n", parent=out_dir)
    print(f"\nWrote {len(frames_meta)} frame(s) across {len(profiles_meta)} profile(s) to "
          f"{out_dir}")
    if interrupted:
        print("Capture was interrupted/aborted before completion -- review manifest.json before "
              "using this session in `measure`.", file=sys.stderr)
        sys.exit(1)


# =====================================================================================
# measure
# =====================================================================================

@dataclass
class _ProfileData:
    ordinal: int
    profile_id: str
    card_frames: list  # list[bytes], in capture order
    composer_pairs: list  # list[tuple[int, target_pre_bytes, composer_open_bytes]], capture order
    profile_advance_clear: bytes
    profile_advance_identity: bytes
    target_scoped_prefix: bool = False


@dataclass
class _SessionData:
    dir: Path
    split: str
    device_serial: str | None
    identity_band: tuple
    content_band: tuple
    profiles: list  # list[_ProfileData]
    manifest: dict


def _validate_entry_anchor_ledger(sess_dir: Path, manifest: dict) -> None:
    """Bind the offline replay artifact to the exact card-frame digest set it replayed."""
    ref = manifest.get("entry_anchor_ledger")
    if (not isinstance(ref, dict) or set(ref) != {"file", "sha256"}
            or ref.get("file") != _ENTRY_ANCHOR_LEDGER_FILE
            or not isinstance(ref.get("sha256"), str)):
        raise RuntimeError(
            "manifest.json lacks the exact hash-bound entry_anchor_ledger reference; run "
            "verify-entry-anchor only if this is an otherwise completed v3 manifest, or "
            "recapture rather than treating a free-form checklist path as navigation evidence")
    path = sess_dir / ref["file"]
    if not path.is_file() or _sha256(path.read_bytes()) != ref["sha256"]:
        raise RuntimeError("entry-anchor ledger is missing or does not match manifest sha256")
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("entry-anchor ledger is not readable JSON") from exc
    required = {
        "schema_version", "kind", "tool_version", "created_utc", "offline_only",
        "phone_input_issued", "runtime_targeting_calibration_emitted", "profiles",
    }
    if (not isinstance(artifact, dict) or set(artifact) != required
            or artifact.get("schema_version") != _ENTRY_ANCHOR_LEDGER_SCHEMA_VERSION
            or artifact.get("kind") != "hinge_entry_anchor_offline_replay"
            or artifact.get("tool_version") != _TOOL_VERSION
            or artifact.get("offline_only") is not True
            or artifact.get("phone_input_issued") is not False
            or artifact.get("runtime_targeting_calibration_emitted") is not False):
        raise RuntimeError("entry-anchor ledger has an unsupported or unsafe schema")
    profiles = artifact.get("profiles")
    expected_profiles = manifest.get("profiles")
    if (not isinstance(profiles, list) or not isinstance(expected_profiles, list)
            or len(profiles) != len(expected_profiles)):
        raise RuntimeError("entry-anchor ledger/profile metadata is not a list")
    expected_sources: dict[int, list[dict]] = {}
    for rec in manifest.get("frames", []):
        if isinstance(rec, dict) and rec.get("role") == "card_scroll":
            expected_sources.setdefault(rec.get("profile_ordinal"), []).append(
                {"file": rec.get("file"), "sha256": rec.get("sha256")})
    by_ordinal = {rec.get("ordinal"): rec for rec in profiles if isinstance(rec, dict)}
    if len(by_ordinal) != len(profiles):
        raise RuntimeError("entry-anchor ledger has duplicate or malformed profile reports")
    report_required = {
        "ordinal", "profile_id_sha256", "source_card_frames", "confirmed_top",
        "index_complete", "index_reached_end", "index_heart_translation",
        "photo_only_translation", "offline_replay_identity_ceiling",
        "offline_replay_identity_entry_distance", "cancellation_before_capture",
        "cancellation_replay_capture_calls", "cancellation_replay_scroll_up_calls", "targets",
    }
    target_required = {
        "photo_model_item", "heart_ordinal", "index_model_item", "entry_anchor_delta_px",
        "entry_offset_px", "landing_frame_sha256", "landing_frame_index",
        "landing_page_offset_px", "hearts_counted", "crosscheck_max_disagreement_px",
        "replay_scroll_steps", "replay_shift_deltas_px",
    }
    for meta in expected_profiles:
        ordinal = meta.get("ordinal") if isinstance(meta, dict) else None
        report = by_ordinal.get(ordinal)
        if not isinstance(report, dict) or set(report) != report_required:
            raise RuntimeError(f"entry-anchor ledger lacks profile ordinal {ordinal!r}")
        if report.get("source_card_frames") != expected_sources.get(ordinal):
            raise RuntimeError(
                f"entry-anchor ledger profile ordinal {ordinal!r} does not bind the exact "
                "manifest card-frame sequence")
        translation = report.get("index_heart_translation")
        photos = report.get("photo_only_translation")
        targets = report.get("targets")
        if (report.get("confirmed_top") is not True or report.get("index_complete") is not True
                or report.get("index_reached_end") is not True
                or report.get("cancellation_before_capture") is not True
                or report.get("cancellation_replay_capture_calls") != 0
                or report.get("cancellation_replay_scroll_up_calls") != 0
                or not isinstance(translation, list) or not translation
                or not isinstance(photos, list) or not photos
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in translation)
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in photos)
                or len(set(translation)) != len(translation) or len(set(photos)) != len(photos)
                or not set(photos).issubset(translation)
                or not isinstance(targets, list) or len(targets) != len(photos)
                or any(not isinstance(t, dict) or set(t) != target_required for t in targets)):
            raise RuntimeError(
                f"entry-anchor ledger profile ordinal {ordinal!r} lacks complete top/index/"
                "cancellation/replay evidence")
        if ([t["photo_model_item"] for t in targets] != list(range(1, len(photos) + 1))
                or [t["heart_ordinal"] for t in targets] != photos
                or any(t["index_model_item"] != translation.index(t["heart_ordinal"]) + 1
                       for t in targets)):
            raise RuntimeError(
                f"entry-anchor ledger profile ordinal {ordinal!r} has inconsistent photo/index "
                "translation targets")


def _load_session(sess_dir: Path, *, require_entry_anchor_ledger: bool = True) -> _SessionData:
    """Load one capture session, re-verifying every frame's sha256 against the manifest as it
    goes -- calibration evidence must be byte-exact (api_contracts.md §6: `ItemPayload` can only
    ever be rebuilt from the EXACT original frame bytes an index was built from), so a frame
    that changed on disk since capture is refused here rather than silently measured."""
    manifest_path = sess_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"no manifest.json in {sess_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("tool_version") != _TOOL_VERSION:
        raise RuntimeError(
            f"manifest.json tool_version={manifest.get('tool_version')!r} is not the supported "
            f"{_TOOL_VERSION!r}; recapture rather than interpreting an unknown schema")
    if manifest.get("interrupted") is not False:
        raise RuntimeError(
            "manifest.json is interrupted, partial, or does not carry an explicit completed "
            "state; partial capture evidence can never calibrate a targeting bound")
    if manifest.get("calibration_schema_version") != _CALIBRATION_SCHEMA_VERSION:
        raise RuntimeError(
            "manifest.json does not carry the supported inline calibration schema v"
            f"{_CALIBRATION_SCHEMA_VERSION}; recapture rather than interpreting modal evidence")
    if manifest.get("composer_layout_id") != _COMPOSER_LAYOUT_ID:
        raise RuntimeError(
            f"manifest.json composer_layout_id={manifest.get('composer_layout_id')!r} is not "
            f"the supported {_COMPOSER_LAYOUT_ID!r}")
    if manifest.get("item_selection_policy_id") != PHOTO_ONLY_POLICY_ID:
        raise RuntimeError(
            f"manifest.json item_selection_policy_id="
            f"{manifest.get('item_selection_policy_id')!r} is not the supported "
            f"{PHOTO_ONLY_POLICY_ID!r}")
    split = manifest.get("split")
    if split not in _SPLITS:
        raise RuntimeError(f"manifest.json split={split!r} is not one of {_SPLITS}")
    try:
        identity_band = tuple(manifest["identity_band"])
        content_band = tuple(manifest["content_band"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("manifest.json does not contain readable identity/content bands") from exc
    device_serial = (manifest.get("device") or {}).get("serial")
    frame_size = manifest.get("frame_size_px")
    device = manifest.get("device") or {}
    if (not isinstance(frame_size, list) or len(frame_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in frame_size)
            or frame_size != [device.get("display_w"), device.get("display_h")]):
        raise RuntimeError(
            "manifest.json frame_size_px is not a positive exact match for device display evidence")

    profile_meta = manifest.get("profiles")
    requested_profiles = manifest.get("requested_profiles")
    if (isinstance(requested_profiles, bool) or not isinstance(requested_profiles, int)
            or requested_profiles < 1):
        raise RuntimeError("manifest.json requested_profiles is not a positive integer")
    if not isinstance(profile_meta, list) or len(profile_meta) != requested_profiles:
        raise RuntimeError(
            f"manifest.json completed {len(profile_meta) if isinstance(profile_meta, list) else 0} "
            f"profile(s), not its requested {requested_profiles}; refusing a partial session")
    expected: dict[int, dict] = {}
    expected_ids: set[str] = set()
    for meta in profile_meta:
        if not isinstance(meta, dict):
            raise RuntimeError("manifest.json profiles contains a non-mapping entry")
        ordinal = meta.get("ordinal")
        profile_id = meta.get("profile_id")
        if (isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1
                or ordinal in expected):
            raise RuntimeError(f"manifest.json has an invalid/duplicate profile ordinal {ordinal!r}")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise RuntimeError(f"profile ordinal {ordinal} has no nonempty unique profile_id")
        if profile_id in expected_ids:
            raise RuntimeError(
                f"profile_id {profile_id!r} is duplicated in {sess_dir}; foreign-profile "
                "comparisons require a unique id for every real profile")
        composer_items = meta.get("composer_items")
        if (not isinstance(composer_items, list) or not composer_items
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                       for n in composer_items)
                or len(set(composer_items)) != len(composer_items)):
            raise RuntimeError(
                f"profile {profile_id!r} has no valid, unique positive composer_items list")
        if meta.get("profile_advance_cleared_composer") is not True:
            raise RuntimeError(
                f"profile {profile_id!r} lacks explicit profile-advance composer-clear evidence")
        if meta.get("profile_advance_identity_mismatched") is not True:
            raise RuntimeError(
                f"profile {profile_id!r} lacks explicit next-profile identity-mismatch evidence")
        expected[ordinal] = meta
        expected_ids.add(profile_id)

    # Skip traces are intentionally not converted to `_ProfileData`: a guarded pre-action Pass
    # proves only that the next retry is distinct, never a target/composer sample.  The separate
    # offline reviewer authenticates their detailed transport/checkpoint records before an
    # automated manifest may be measured.
    skipped_attempts = manifest.get("skipped_attempts", [])
    if not isinstance(skipped_attempts, list):
        raise RuntimeError("manifest.json skipped_attempts is not a list")
    if len(skipped_attempts) > _MAX_PREACTION_PROFILE_SKIPS_PER_SESSION:
        raise RuntimeError("manifest.json skipped_attempts exceeds the bounded session cap")
    if manifest.get("capture_mode") == "supervised_manual" and skipped_attempts:
        raise RuntimeError("supervised capture cannot contain automated skipped_attempts")

    checks = manifest.get("operational_checks")
    if not isinstance(checks, dict) or set(checks) != set(_OPERATIONAL_CHECKS):
        raise RuntimeError(
            "manifest.json does not carry the exact supervised operational-check schema; "
            "recapture with this tool version")
    for key, rec in checks.items():
        if not isinstance(rec, dict) or not isinstance(rec.get("confirmed"), bool):
            raise RuntimeError(f"operational_checks.{key} is malformed")
        if rec["confirmed"] and (
                not isinstance(rec.get("evidence"), str) or not rec["evidence"].strip()
                or not isinstance(rec.get("recorded_utc"), str)
                or not rec["recorded_utc"].strip()):
            raise RuntimeError(
                f"operational_checks.{key} is marked confirmed without traceable evidence")

    by_profile: dict[int, dict] = {}
    frame_records = manifest.get("frames")
    if not isinstance(frame_records, list):
        raise RuntimeError("manifest.json frames is not a list")
    if manifest.get("frame_count") != len(frame_records):
        raise RuntimeError("manifest.json frame_count does not match its frames list")
    seen_files: set[str] = set()
    for sequence, rec in enumerate(frame_records, 1):
        if not isinstance(rec, dict):
            raise RuntimeError("manifest.json frames contains a non-mapping entry")
        name = rec.get("file")
        if (not isinstance(name, str) or Path(name).name != name
                or re.fullmatch(r"\d{5}\.png", name) is None or name in seen_files):
            raise RuntimeError(f"invalid, unsafe, or duplicate frame filename {name!r}")
        expected_name = f"{sequence:05d}.png"
        if name != expected_name:
            raise RuntimeError(
                f"manifest frame order is not the original contiguous capture sequence: "
                f"record {sequence} is {name!r}, expected {expected_name!r}")
        seen_files.add(name)
        path = sess_dir / name
        if not path.is_file():
            raise RuntimeError(f"{path} is listed in manifest.json but missing on disk")
        data = path.read_bytes()
        digest = _sha256(data)
        if digest != rec["sha256"]:
            raise RuntimeError(
                f"{path} sha256 mismatch (manifest {rec['sha256']}, on-disk {digest}) -- frame "
                "bytes changed since capture; refusing to measure against evidence that no "
                "longer matches what was recorded")
        pid = rec.get("profile_ordinal")
        if pid not in expected:
            raise RuntimeError(f"frame {name} refers to unknown profile ordinal {pid!r}")
        entry = by_profile.setdefault(
            pid, {"profile_id": rec.get("profile_id"), "card": [], "pairs": [],
                  "sequence": []})
        if (entry["profile_id"] != rec.get("profile_id")
                or rec.get("profile_id") != expected[pid]["profile_id"]):
            raise RuntimeError(
                f"profile ordinal {pid} has inconsistent profile_id in {sess_dir} "
                f"({entry['profile_id']!r} vs {rec.get('profile_id')!r})")
        role = rec.get("role")
        if role == "card_scroll":
            if rec.get("item_number") is not None:
                raise RuntimeError(f"card frame {name} unexpectedly carries an item number")
            entry["card"].append(data)
        elif role in ("target_pre", "composer_open"):
            item_number = rec.get("item_number")
            if (isinstance(item_number, bool) or not isinstance(item_number, int)
                    or item_number < 1):
                raise RuntimeError(f"{role} frame {name} has invalid item_number {item_number!r}")
            entry["sequence"].append((role, item_number, data))
        elif role == "profile_advance_clear":
            if rec.get("item_number") is not None:
                raise RuntimeError(f"profile-advance frame {name} unexpectedly carries an item number")
            entry["sequence"].append((role, None, data))
        elif role == "profile_advance_identity":
            if rec.get("item_number") is not None:
                raise RuntimeError(f"profile-advance identity frame {name} unexpectedly carries an item number")
            entry["sequence"].append((role, None, data))
        else:
            raise RuntimeError(f"unknown frame role {role!r} in {sess_dir}")

    actual_pngs = {p.name for p in sess_dir.glob("*.png")}
    if actual_pngs != seen_files:
        raise RuntimeError(
            f"session PNG set does not exactly match manifest frames (unlisted "
            f"{sorted(actual_pngs - seen_files)}, missing {sorted(seen_files - actual_pngs)})")
    target_scoped_automated = (
        manifest.get("capture_evidence_scope") == _TARGET_SCOPED_PREFIX_PROOF_ID
        and manifest.get("capture_mode") in {
            "automated_circular_risk_accepted", _HYBRID_CAPTURE_MODE}
        and manifest.get("human_ground_truth") is False)
    if require_entry_anchor_ledger and not target_scoped_automated:
        _validate_entry_anchor_ledger(sess_dir, manifest)
    if target_scoped_automated and manifest.get("entry_anchor_ledger") is not None:
        raise RuntimeError(
            "target-scoped automated evidence must not carry a closed-set entry-anchor ledger; "
            "that would obscure which proof the capture actually made")

    profiles = []
    for pid, v in sorted(by_profile.items()):
        expected_items = expected[pid]["composer_items"]
        expected_sequence = []
        for item_number in expected_items:
            expected_sequence.extend((("target_pre", item_number), ("composer_open", item_number)))
        expected_sequence.extend((("profile_advance_clear", None),
                                  ("profile_advance_identity", None)))
        actual_sequence = [(role, number) for role, number, _data in v["sequence"]]
        if actual_sequence != expected_sequence:
            raise RuntimeError(
                f"profile {v['profile_id']!r} inline frame roles/items {actual_sequence} do not "
                f"exactly match required persistent-composer sequence {expected_sequence}")
        pairs = []
        for offset, item_number in enumerate(expected_items):
            pre = v["sequence"][2 * offset][2]
            composer = v["sequence"][2 * offset + 1][2]
            pairs.append((item_number, pre, composer))
        profiles.append(_ProfileData(
            ordinal=pid, profile_id=v["profile_id"], card_frames=v["card"],
            composer_pairs=pairs, profile_advance_clear=v["sequence"][-2][2],
            profile_advance_identity=v["sequence"][-1][2],
            target_scoped_prefix=target_scoped_automated))
    if {p.ordinal for p in profiles} != set(expected):
        raise RuntimeError(f"{sess_dir} has completed profile metadata without matching frames")
    for p in profiles:
        if not p.card_frames:
            raise RuntimeError(
                f"profile {p.profile_id!r} in {sess_dir} has no card_scroll frames")
        if expected[p.ordinal].get("card_scroll_frames") != len(p.card_frames):
            raise RuntimeError(
                f"profile {p.profile_id!r} has {len(p.card_frames)} card frame(s), but its "
                f"completed-attempt metadata records "
                f"{expected[p.ordinal].get('card_scroll_frames')!r}")

    return _SessionData(dir=sess_dir, split=split, device_serial=device_serial,
                        identity_band=identity_band, content_band=content_band,
                        profiles=profiles, manifest=manifest)


def _cmd_verify_entry_anchor(args: argparse.Namespace) -> None:
    """Attach one offline replay ledger to a completed, previously unledgered capture.

    This is deliberately a migration *only* for a completed manifest whose frame roles and
    digests can already be verified.  In particular it cannot infer roles from a pile of PNGs:
    that would turn an ambiguous current session into invented evidence.
    """
    try:
        sess_dir = _existing_calibration_session_dir(args.session)
        manifest_path = sess_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                "no manifest.json: refusing to reconstruct card/composer/advance roles from "
                "unmanifested private PNGs")
        # First authenticate every listed card byte and the exact persistent-composer terminal
        # sequence.  Bypassing only the ledger gate is intentional: this command creates it.
        session = _load_session(sess_dir, require_entry_anchor_ledger=False)
        if session.manifest.get("entry_anchor_ledger") is not None:
            _validate_entry_anchor_ledger(sess_dir, session.manifest)
            print(f"Existing entry-anchor ledger already validates: "
                  f"{sess_dir / _ENTRY_ANCHOR_LEDGER_FILE}")
            return

        cfg = cfg_mod.load(args.config)
        replay_driver = HingeDriver(cfg)  # construction/template lookup only; never open_session()
        if (replay_driver.identity_band != session.identity_band
                or replay_driver.content_band != session.content_band):
            raise RuntimeError(
                "current config's effective identity/content bands do not exactly match this "
                "captured manifest; refusing to replay with a different detector binding")
        ref = _write_entry_anchor_ledger(
            sess_dir, frames_meta=session.manifest["frames"],
            profiles_meta=session.manifest["profiles"], identity_band=session.identity_band,
            content_band=session.content_band, like_template=replay_driver._template("like"),
            like_threshold=hinge_mod._LIKE_MATCH_THRESHOLD)
        session.manifest["entry_anchor_ledger"] = ref
        atomic_write_private_text(
            manifest_path, json.dumps(session.manifest, indent=2) + "\n", parent=sess_dir)
        _validate_entry_anchor_ledger(sess_dir, session.manifest)
    except (OSError, RuntimeError, _MeasureRefused, ItemIndexError, ItemCropError,
            SegmentationError, ShiftEstimationError) as exc:
        print(f"ERROR: offline entry-anchor verification refused: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote hash-bound offline entry-anchor replay: {sess_dir / _ENTRY_ANCHOR_LEDGER_FILE}")


def _cmd_attach_operational_evidence(args: argparse.Namespace) -> None:
    """Safely migrate a completed pre-ledger capture to the truthful v2 recorder artifact.

    This exists for an in-memory capture that was already paused at the old evidence prompts
    when the stricter v2 gate landed.  It never reconstructs screenshots or accepts prose: it
    first re-authenticates the completed capture, validates every v2 recorder frame/analysis,
    builds the normal offline entry-anchor replay, and then changes only the three evidence
    references in the existing operational-check records.  A manifest that already has a ledger
    is deliberately refused so this cannot be used to rewrite an established audit trail.
    """
    created_ledger: Path | None = None
    try:
        sess_dir = _existing_calibration_session_dir(args.session)
        session = _load_session(sess_dir, require_entry_anchor_ledger=False)
        if session.manifest.get("entry_anchor_ledger") is not None:
            raise RuntimeError(
                "session already has an entry-anchor ledger; refuse to rewrite its operational "
                "references. Measure it as-is or recapture a fresh evidence session")
        checks = session.manifest.get("operational_checks")
        required = ("entry_anchor_scroll_ledger", "item_1_inline_composer_identity",
                    "gesture_transport")
        if (not isinstance(checks, dict)
                or any(not isinstance(checks.get(key), dict)
                       or checks[key].get("confirmed") is not True for key in required)):
            raise RuntimeError(
                "completed capture lacks affirmative existing entry/item-1/gesture operational "
                "records; this migration cannot invent an owner-performed check")
        expected_device = session.manifest.get("device")
        reason = _operational_evidence_reference_reason(
            args.operational_evidence, expected_device=expected_device,
            identity_band=session.identity_band)
        if reason:
            raise RuntimeError("supplied operational recorder artifact is invalid: " + reason)
        op_manifest, op_manifest_reason = _operational_evidence_manifest_path(
            args.operational_evidence)
        if op_manifest is None:
            raise RuntimeError(
                "validated operational evidence became unavailable: "
                + (op_manifest_reason or "manifest path could not be resolved"))

        cfg = cfg_mod.load(args.config)
        replay_driver = HingeDriver(cfg)  # template lookup only; this command never opens ADB
        if (replay_driver.identity_band != session.identity_band
                or replay_driver.content_band != session.content_band):
            raise RuntimeError(
                "current config's effective identity/content bands do not exactly match this "
                "captured manifest; refusing to attach evidence under a different binding")
        entry_candidate = sess_dir / _ENTRY_ANCHOR_LEDGER_FILE
        created_ledger = entry_candidate
        entry_ref = _write_entry_anchor_ledger(
            sess_dir, frames_meta=session.manifest["frames"],
            profiles_meta=session.manifest["profiles"], identity_band=session.identity_band,
            content_band=session.content_band, like_template=replay_driver._template("like"),
            like_threshold=hinge_mod._LIKE_MATCH_THRESHOLD)
        session.manifest["entry_anchor_ledger"] = entry_ref
        checks["entry_anchor_scroll_ledger"]["evidence"] = str(
            (sess_dir / _ENTRY_ANCHOR_LEDGER_FILE).resolve())
        checks["item_1_inline_composer_identity"]["evidence"] = str(op_manifest)
        checks["gesture_transport"]["evidence"] = str(op_manifest)
        manifest_path = sess_dir / "manifest.json"
        _validate_entry_anchor_ledger(sess_dir, session.manifest)
        # Re-run the complete relation check on this single manifest; a later measure will also
        # bind it against the held-out session, but no arbitrary reference survives this write.
        _verified_operational_checks([session])
        # Commit the one changed JSON file atomically only after every new relation validates.
        # If any prior check refused, the catch below removes the newly-created ledger too, so a
        # later run never mistakes an orphan side file for an attached audit trail.
        atomic_write_private_text(
            manifest_path, json.dumps(session.manifest, indent=2) + "\n", parent=sess_dir)
    except (OSError, RuntimeError, _MeasureRefused, ItemIndexError, ItemCropError,
            SegmentationError, ShiftEstimationError) as exc:
        if created_ledger is not None:
            created_ledger.unlink(missing_ok=True)
        print(f"ERROR: offline operational-evidence attachment refused: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Attached hash-valid recorder v2 evidence and entry-anchor replay to "
          f"{sess_dir / 'manifest.json'}")


def _split_profiles(sessions: list) -> list:
    return [p for s in sessions for p in s.profiles]


def _session_sort_key(sess: _SessionData):
    m = re.search(r"(\d{8}T\d{6}Z)$", sess.dir.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(sess.dir.stat().st_mtime, tz=timezone.utc)


def _validate_profile_advance_clears(profiles: list, *, identity_band, confirm_template) -> None:
    """Re-check terminal transition evidence offline; absence from view is not a cancellation.

    Hinge's inline composer persists until the owner advances the profile.  A terminal frame is
    therefore evidence only when it both returns to confirmed top chrome with no composer AND a
    subsequent sticky-header frame is provably a different profile. This rejects a same-profile
    scroll where the composer simply moved offscreen.
    """
    for p in profiles:
        try:
            top = confirm_scroll_top(p.profile_advance_clear, identity_band=identity_band)
        except ScrollTopError as exc:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: advance-clear frame could not confirm a new profile top "
                f"({exc})") from exc
        if not top.confirmed:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: advance-clear frame is not a confirmed profile top "
                f"({top.reason})")
        try:
            locate_inline_composer(p.profile_advance_clear, confirm_template, threshold=0.8)
        except ComposerDetectionError:
            continue
        raise _MeasureRefused(
            f"profile {p.profile_id!r}: terminal advance-clear frame still contains an inline "
            "composer; do not treat a persistent/offscreen composer as cancelled")

    for p in profiles:
        try:
            prior = capture_profile_identity(
                p.card_frames, identity_band=identity_band, grid=_IDENTITY_GRID)
            if not prior.known:
                raise _MeasureRefused(
                    f"profile {p.profile_id!r}: prior card-scroll capture has no identity "
                    f"representative ({prior.reason})")
            identity_top = confirm_scroll_top(
                p.profile_advance_identity, identity_band=identity_band)
            if not identity_top.refuted:
                raise _MeasureRefused(
                    f"profile {p.profile_id!r}: next-profile identity frame does not positively "
                    f"show a sticky header ({identity_top.reason})")
            next_fp = band_fingerprint(
                p.profile_advance_identity, identity_band=identity_band, grid=_IDENTITY_GRID)
            distance = _checked_distance(
                fingerprint_distance(prior.fingerprint, next_fp),
                context=f"profile {p.profile_id!r} next-profile identity")
        except ScrollTopError as exc:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: next-profile identity frame could not be read ({exc})") from exc
        if distance <= _IDENTITY_FALSE_MATCH_DISTANCE:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: claimed next profile identity is {distance:.3f} from "
                f"the prior profile, at/under the {_IDENTITY_FALSE_MATCH_DISTANCE} ambiguity "
                "ceiling. A same-profile scroll cannot certify composer clearance")


# --- IDENTITY ------------------------------------------------------------------------

@dataclass
class _IdentitySample:
    profile_id: str
    origin: str  # "representative" or f"composer:{item_number}"
    fingerprint: tuple


def _profile_identity_samples(profiles: list, *, identity_band) -> list:
    """One representative fingerprint per profile (from its card-scroll sequence, via
    `capture_profile_identity`) plus one fingerprint per captured composer frame (via
    `band_fingerprint` directly on that frame's identity band) -- matching ops/RUNBOOK.md's
    same-profile card/composer pairs: the card representative is compared against every one of
    that profile's own composer frames, and every profile's samples are compared against
    every other profile's."""
    samples: list[_IdentitySample] = []
    for p in profiles:
        identity = capture_profile_identity(p.card_frames, identity_band=identity_band,
                                            grid=_IDENTITY_GRID)
        if not identity.known:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: card-scroll frames do not establish an identity "
                f"fingerprint ({identity.reason})")
        samples.append(_IdentitySample(p.profile_id, "representative", identity.fingerprint))
        for item_number, _target_pre, frame in p.composer_pairs:
            try:
                fp = band_fingerprint(frame, identity_band=identity_band, grid=_IDENTITY_GRID)
            except ScrollTopError as exc:
                raise _MeasureRefused(
                    f"profile {p.profile_id!r} item {item_number}: composer frame's identity band "
                    f"could not be read ({exc})") from exc
            samples.append(_IdentitySample(p.profile_id, f"composer:{item_number}", fp))
    return samples


def _identity_pair_distances(samples: list) -> tuple[list, list]:
    same, diff = [], []
    for a, b in combinations(samples, 2):
        d = _checked_distance(
            fingerprint_distance(a.fingerprint, b.fingerprint),
            context=(f"identity comparison {a.profile_id!r}/{a.origin} against "
                     f"{b.profile_id!r}/{b.origin}"))
        rec = {"a_profile": a.profile_id, "a_origin": a.origin, "b_profile": b.profile_id,
               "b_origin": b.origin, "distance": d}
        (same if a.profile_id == b.profile_id else diff).append(rec)
    return same, diff


def _freeze_identity_bound(samples: list) -> tuple[float | None, dict]:
    same, diff = _identity_pair_distances(samples)
    report = {"same_profile_pairs": same, "different_profile_pairs": diff}
    if not same:
        report["reason"] = ("no same-profile pairs on the calibration split -- need at least "
                            "one profile with >= 2 fingerprints (its card representative plus "
                            ">= 1 captured composer)")
        return None, report
    if not diff:
        report["reason"] = ("no different-profile pairs on the calibration split -- need >= 2 "
                            "distinct profiles")
        return None, report
    max_same = max(r["distance"] for r in same)
    min_diff = min(r["distance"] for r in diff)
    report["max_same_profile_distance"] = max_same
    report["min_different_profile_distance"] = min_diff
    report["margin"] = min_diff - max_same
    if max_same >= min_diff:
        report["reason"] = (
            f"not separable: max same-profile distance {max_same:.3f} >= min different-profile "
            f"distance {min_diff:.3f}")
        return None, report
    bound = (max_same + min_diff) / 2.0
    if bound >= _IDENTITY_FALSE_MATCH_DISTANCE:
        bound = math.nextafter(_IDENTITY_FALSE_MATCH_DISTANCE, 0.0)
    if bound <= max_same:
        report["reason"] = (
            f"the midpoint bound is at/above the known {_IDENTITY_FALSE_MATCH_DISTANCE} "
            f"collision ceiling, and clamping it strictly below that ceiling would put it "
            f"at/under this split's own max same-profile distance ({max_same:.3f}) -- not a "
            "usable bound")
        return None, report
    report["frozen_bound_raw"] = bound
    return bound, report


def _apply_identity_bound(samples: list, bound: float) -> tuple[bool, str, dict]:
    same, diff = _identity_pair_distances(samples)
    # Production identity comparison accepts equality (`distance <= bound`). Mirror that exact
    # boundary here: equality is a foreign accept, while a same-profile equality is accepted.
    foreign_accepts = [r for r in diff if r["distance"] <= bound]
    false_refusals = [r for r in same if r["distance"] > bound]
    detail = {"same_profile_pairs": same, "different_profile_pairs": diff,
              "foreign_accepts": foreign_accepts, "false_refusals": false_refusals}
    if foreign_accepts:
        nearest = min(r["distance"] for r in foreign_accepts)
        return False, (
            f"{len(foreign_accepts)} held-out different-profile pair(s) scored under the "
            f"frozen bound {bound:.4f} (nearest {nearest:.3f}) -- this bound would have liked "
            "the WRONG profile's item. The bound is INVALIDATED, not just this run"), detail
    if false_refusals:
        return False, (
            f"{len(false_refusals)} held-out same-profile pair(s) scored at/over the frozen "
            f"bound {bound:.4f} -- a false refusal is a stop/re-measurement signal, never a "
            "reason to widen the bound"), detail
    if not diff:
        return False, (
            "held-out split produced no different-profile pairs to test 0 foreign-profile "
            "accepts against -- need >= 2 distinct profiles in the held-out split"), detail
    return True, "0 foreign-profile accepts, 0 false refusals", detail


# --- INLINE COMPOSER ITEMS -------------------------------------------------------------

@dataclass
class _ProfilePayload:
    profile_id: str
    payload: ItemPayload
    composer_pairs: list  # list[tuple[int, target_pre_bytes, composer_open_bytes]]


def _build_profile_payloads(profiles: list, *, content_band, identity_band, like_template,
                            like_threshold) -> list:
    out: list[_ProfilePayload] = []
    for p in profiles:
        try:
            index = build_item_index(
                p.card_frames, content_band=content_band, like_template=like_template,
                like_threshold=like_threshold, at_scroll_top=True, identity_band=identity_band)
        except (ItemIndexError, SegmentationError, ShiftEstimationError) as exc:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: could not build an item index from its "
                f"card-scroll frames ({type(exc).__name__}: {exc})") from exc
        if not index.usable:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: item index is not usable: "
                + "; ".join(index.failures))
        if not index.complete and not p.target_scoped_prefix:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: item index does not cover the complete profile "
                f"(at_scroll_top={index.at_scroll_top}, reached_end={index.reached_end}, "
                f"partial_blocks={len(index.partial)}); every available foreign item must be "
                "included in calibration negatives")
        try:
            # Classifier-only, matching capture.  Measure rebuilds the numbering capture already
            # committed to; if it applied a gate capture did not, it would refuse the very items
            # capture numbered and the two would disagree about a profile they both read from
            # the same bytes.  The still-photo licence for a HEART is not this call's job -- it
            # was `_verified_still_photo_proof`'s, live, in the pre-heart window.
            payload = build_item_payload(
                p.card_frames, index, unnumber=unnumber_unless_confident_photo)
        except ItemCropError as exc:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: could not build item crops: {exc}") from exc
        if not payload.usable:
            raise _MeasureRefused(
                f"profile {p.profile_id!r}: item payload is not usable: "
                + "; ".join(payload.failures))
        target_items = tuple(item_number for item_number, _target_pre, _composer_open
                             in p.composer_pairs)
        if p.target_scoped_prefix:
            reason = _target_scoped_prefix_reason(index, payload, target_items)
            if reason is not None:
                raise _MeasureRefused(
                    f"profile {p.profile_id!r}: target-scoped prefix proof refused: {reason}")
        available = {c.number for c in payload.items}
        for item_number, _target_pre, _composer_open in p.composer_pairs:
            if item_number not in available:
                raise _MeasureRefused(
                    f"profile {p.profile_id!r}: a composer was captured for item {item_number}, "
                    f"which this profile's index/payload never produced ({sorted(available)} "
                    "available) -- capture and measure disagree about this profile's item "
                    "numbering, refusing rather than guessing")
        out.append(_ProfilePayload(profile_id=p.profile_id, payload=payload,
                                   composer_pairs=list(p.composer_pairs)))
    return out


@dataclass
class _InlineDistanceRecord:
    kind: str  # "own_intended" | "own_foreign_item" | "foreign_profile"
    composer_profile_id: str
    composer_item_number: int
    against_profile_id: str
    against_item_number: int
    distance: float


def _inline_distance_records(profiles: list, *, confirm_template) -> list:
    """For every verified inline composer: compare its selected card against every item in its own
    profile and every item in every other profile.
    (the intended item plus every other numbered item on that same profile -- one
    `verify_sheet_item` call already returns comparisons for all of them, see api_contracts.md
    §4), and its distance to every item in every OTHER profile's payload ("foreign_profile").
    Every real profile id is required to be globally unique before this function runs, so only
    the exact payload object that supplied the sheet is skipped in the foreign loop. A duplicate
    human label can therefore never suppress a negative comparison.
    `model_index` for a foreign payload is arbitrary (any number that payload actually has) --
    the comparisons returned do not depend on which one is passed, only `.matched`/`.distance`
    for THAT number would, and this function never reads either of those, only `.comparisons`.
    """
    records: list[_InlineDistanceRecord] = []
    for pp in profiles:
        for item_number, _target_pre, frame in pp.composer_pairs:
            try:
                surface = locate_inline_composer(frame, confirm_template, threshold=0.8)
                if surface.layout_id != _COMPOSER_LAYOUT_ID:
                    raise SheetVerificationError(
                        f"unsupported composer layout {surface.layout_id!r}")
                verdict = verify_sheet_item(frame, pp.payload, item_number,
                                            absolute_max_dist=None,
                                            composer_surface=surface)
            except (ComposerDetectionError, SheetVerificationError) as exc:
                raise _MeasureRefused(
                    f"profile {pp.profile_id!r} item {item_number}: composer frame could not be "
                    f"compared against its own payload ({exc})") from exc
            for comp in verdict.comparisons:
                if comp.distance is None:
                    raise _MeasureRefused(
                        f"profile {pp.profile_id!r} item {item_number}: comparison against its "
                        f"own item {comp.number} had no distance ({comp.reason}); the protocol "
                        "requires every available alternative, so it cannot be omitted")
                distance = _checked_distance(
                    comp.distance,
                    context=(f"composer {pp.profile_id!r} item {item_number} against own item "
                             f"{comp.number}"))
                kind = "own_intended" if comp.number == item_number else "own_foreign_item"
                records.append(_InlineDistanceRecord(
                    kind=kind, composer_profile_id=pp.profile_id, composer_item_number=item_number,
                    against_profile_id=pp.profile_id, against_item_number=comp.number,
                    distance=distance))
            for other in profiles:
                if other is pp:
                    continue
                probe_number = min((c.number for c in other.payload.items), default=None)
                if probe_number is None:
                    continue
                try:
                    foreign_verdict = verify_sheet_item(frame, other.payload, probe_number,
                                                        absolute_max_dist=None,
                                                        composer_surface=surface)
                except SheetVerificationError as exc:
                    raise _MeasureRefused(
                        f"profile {pp.profile_id!r} item {item_number}: composer frame could not "
                        f"be compared against profile {other.profile_id!r}'s payload "
                        f"({exc})") from exc
                for comp in foreign_verdict.comparisons:
                    if comp.distance is None:
                        raise _MeasureRefused(
                            f"profile {pp.profile_id!r} item {item_number}: comparison against "
                            f"foreign profile {other.profile_id!r} item {comp.number} had no "
                            f"distance ({comp.reason}); foreign candidates cannot be omitted")
                    distance = _checked_distance(
                        comp.distance,
                        context=(f"composer {pp.profile_id!r} item {item_number} against foreign "
                                 f"profile {other.profile_id!r} item {comp.number}"))
                    records.append(_InlineDistanceRecord(
                        kind="foreign_profile", composer_profile_id=pp.profile_id,
                        composer_item_number=item_number, against_profile_id=other.profile_id,
                        against_item_number=comp.number, distance=distance))
    return records


def _record_dicts(records: list) -> list:
    return [
        {"kind": r.kind, "composer_profile_id": r.composer_profile_id,
         "composer_item_number": r.composer_item_number, "against_profile_id": r.against_profile_id,
         "against_item_number": r.against_item_number, "distance": r.distance}
        for r in records
    ]


def _freeze_inline_bound(records: list) -> tuple[float | None, dict]:
    for r in records:
        _checked_distance(
            r.distance,
            context=(f"{r.kind} inline comparison {r.composer_profile_id!r}/"
                     f"{r.composer_item_number} against {r.against_profile_id!r}/"
                     f"{r.against_item_number}"))
    intended = [r for r in records if r.kind == "own_intended"]
    foreign = [r for r in records if r.kind in ("own_foreign_item", "foreign_profile")]
    report = {"intended": _record_dicts(intended), "foreign": _record_dicts(foreign)}
    if not intended:
        report["reason"] = "no intended-item distances measured on the calibration split"
        return None, report
    if not foreign:
        report["reason"] = (
            "no foreign-item/foreign-profile distances measured on the calibration split -- "
            "need either a profile with >= 2 items or >= 2 distinct profiles")
        return None, report
    max_correct = max(r.distance for r in intended)
    min_foreign = min(r.distance for r in foreign)
    report["max_intended_distance"] = max_correct
    report["min_foreign_distance"] = min_foreign
    report["margin"] = min_foreign - max_correct
    if max_correct >= min_foreign:
        report["reason"] = (
            f"not separable: max intended-item distance {max_correct:.3f} >= min foreign "
            f"distance {min_foreign:.3f}")
        return None, report
    bound = (max_correct + min_foreign) / 2.0
    if bound >= _INLINE_FALSE_MATCH_DISTANCE:
        bound = math.nextafter(_INLINE_FALSE_MATCH_DISTANCE, 0.0)
    if bound <= max_correct:
        report["reason"] = (
            f"the midpoint bound is at/above the known {_INLINE_FALSE_MATCH_DISTANCE} collision "
            f"ceiling, and clamping it strictly below that ceiling would put it at/under this "
            f"split's own max intended-item distance ({max_correct:.3f}) -- not a usable bound")
        return None, report
    report["frozen_bound_raw"] = bound
    return bound, report


def _apply_inline_bound(records: list, bound: float) -> tuple[bool, str, dict]:
    _checked_distance(bound, context="frozen inline-item bound")
    for r in records:
        _checked_distance(
            r.distance,
            context=(f"held-out {r.kind} inline comparison {r.composer_profile_id!r}/"
                     f"{r.composer_item_number} against {r.against_profile_id!r}/"
                     f"{r.against_item_number}"))
    intended = [r for r in records if r.kind == "own_intended"]
    foreign = [r for r in records if r.kind in ("own_foreign_item", "foreign_profile")]
    foreign_accepts = [r for r in foreign if r.distance < bound]
    false_refusals = [r for r in intended if r.distance >= bound]
    detail = {"intended": _record_dicts(intended), "foreign": _record_dicts(foreign),
              "foreign_accepts": _record_dicts(foreign_accepts),
              "false_refusals": _record_dicts(false_refusals)}
    if foreign_accepts:
        nearest = min(r.distance for r in foreign_accepts)
        return False, (
            f"{len(foreign_accepts)} held-out foreign-item/foreign-profile inline comparison(s) "
            f"scored under the frozen bound {bound:.4f} (nearest {nearest:.3f}) -- this bound "
            "would have accepted the WRONG item's composer. The bound is INVALIDATED, not just "
            "this run"), detail
    if false_refusals:
        return False, (
            f"{len(false_refusals)} held-out intended-item inline comparison(s) scored at/over "
            f"the frozen bound {bound:.4f} -- a false refusal is a stop/re-measurement signal, "
            "never a reason to widen the bound"), detail
    if not foreign:
        return False, (
            "held-out split produced no foreign-item/foreign-profile comparisons to test 0 "
            "accepts against"), detail
    return True, "0 foreign-item/foreign-profile accepts, 0 false refusals", detail


def _preliminary_observe_check_reference_reason(reference: str) -> str | None:
    """Validate the required preliminary artifact for the numeric-candidate stage.

    This is intentionally *accepted* by ``measure``: it is the safe bridge out of the circular
    gate. The separate config AUTO release gate later requires a production-OBSERVE artifact.
    """
    path = Path(reference)
    if path.is_dir():
        path /= "observe_check.json"
    if path.name != "observe_check.json" or not path.is_file():
        return (f"{path} is not a readable observe_check.json artifact; a free-form path or "
                "assertion cannot satisfy the preliminary OBSERVE gate")
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"could not read observe-check evidence {path}: {type(exc).__name__}"
    if (record.get("kind") != "hinge_supervised_observe_only_check"
            or record.get("evidence_scope") != _PRELIMINARY_OBSERVE_SCOPE):
        return f"{path} is not a recognised preliminary observe-check artifact"
    if record.get("completed") is not True:
        return f"{path} is an incomplete preliminary observe-check artifact"
    digest = record.get("evidence_sha256")
    body = dict(record)
    body.pop("evidence_sha256", None)
    if not isinstance(digest, str) or digest != _canonical_json_digest(body):
        return f"{path} has an invalid preliminary observe-check self-hash"
    return None


def _entry_anchor_reference_reason(session: _SessionData, reference: str) -> str | None:
    """Require the operator field to name this session's already hash-bound ledger.

    ``_load_session`` validates the ledger bytes and replay content.  This extra check closes
    the remaining loophole where a checked box could carry an unrelated run-id or prose instead
    of pointing back to that verified artifact.  The capture directory itself is accepted as a
    convenient shorthand, but only because it resolves to the one fixed ledger filename.
    """
    path = Path(reference).expanduser()
    if path.is_dir():
        path /= _ENTRY_ANCHOR_LEDGER_FILE
    expected = (session.dir / _ENTRY_ANCHOR_LEDGER_FILE).resolve()
    try:
        actual = path.resolve()
    except OSError as exc:
        return f"could not resolve entry-anchor evidence {path}: {type(exc).__name__}"
    if actual != expected or not actual.is_file():
        return (f"{path} is not this session's exact {_ENTRY_ANCHOR_LEDGER_FILE}; a free-form "
                "assertion cannot satisfy the entry-anchor check")
    return None


def _operational_evidence_manifest_path(reference: str) -> tuple[Path | None, str | None]:
    path = Path(reference).expanduser()
    if path.is_dir():
        path /= "manifest.json"
    if path.name != "manifest.json" or not path.is_file():
        return None, (f"{path} is not a readable operational-evidence manifest.json; a free-form "
                      "path or assertion cannot satisfy this check")
    return path.resolve(), None


def _operational_evidence_reference_reason(reference: str, *, expected_device: dict,
                                           identity_band: tuple) -> str | None:
    """Validate one v2 read-only operational recorder artifact without trusting its labels.

    Every screen is re-hashed, its geometry is checked against the recorder's device evidence,
    and the exact successful-v2 topology/analysis schema is required.  This deliberately makes
    the old v1 ``unfocused``/``focused`` trace unusable: Hinge 9.134 has an auto-focused inline
    composer, so a trace that claims a transition the app does not present cannot certify it.
    """
    path, path_reason = _operational_evidence_manifest_path(reference)
    if path_reason:
        return path_reason
    if path is None:
        return "could not resolve the operational-evidence manifest after validation"
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"could not read operational evidence {path}: {type(exc).__name__}"
    expected_keys = {
        "tool_version", "completed", "interrupted", "device", "frame_size_px",
        "identity_band", "composer_layout_id", "config_binding",
        "new_profile_min_identity_distance", "start_utc", "end_utc", "frame_count",
        "frames", "analyses", "failure", "evidence_sha256",
    }
    if not isinstance(artifact, dict) or set(artifact) != expected_keys:
        return f"{path} has an unsupported operational-evidence schema; recapture with recorder v2"
    if (artifact.get("tool_version") != _OPERATIONAL_EVIDENCE_TOOL_VERSION
            or artifact.get("completed") is not True or artifact.get("interrupted") is not False
            or artifact.get("failure") is not None):
        return f"{path} is not one completed, successful operational recorder v2 trace"
    claimed_digest = artifact.get("evidence_sha256")
    digest_body = dict(artifact)
    digest_body.pop("evidence_sha256", None)
    if (not isinstance(claimed_digest, str)
            or claimed_digest != _canonical_json_digest(digest_body)):
        return f"{path} has an invalid operational-evidence manifest self-hash"
    device = artifact.get("device")
    if (not isinstance(device, dict) or set(device) != set(_OPERATIONAL_EVIDENCE_DEVICE_KEYS)
            or device != expected_device):
        return (f"{path} device/build evidence does not exactly match the calibration/held-out "
                "sessions")
    frame_size = artifact.get("frame_size_px")
    if frame_size != [device["display_w"], device["display_h"]]:
        return f"{path} frame_size_px does not exactly match its device evidence"
    if artifact.get("identity_band") != list(identity_band):
        return f"{path} identity_band does not exactly match the effective calibration config"
    if artifact.get("composer_layout_id") != _COMPOSER_LAYOUT_ID:
        return f"{path} has an unsupported composer layout"
    binding = artifact.get("config_binding")
    expected_binding = {
        "serial": device["serial"], "identity_band": list(identity_band),
        "composer_layout_id": _COMPOSER_LAYOUT_ID, "hinge_package": device["hinge_package"],
    }
    if binding != expected_binding:
        return f"{path} config binding does not exactly match the effective calibration config"
    if artifact.get("new_profile_min_identity_distance") != _IDENTITY_FALSE_MATCH_DISTANCE:
        return f"{path} has an unsupported new-profile identity separation threshold"
    frames = artifact.get("frames")
    if not isinstance(frames, list) or len(frames) != len(_OPERATIONAL_EVIDENCE_ROLES):
        return f"{path} does not carry the complete six-frame operational trace"
    if artifact.get("frame_count") != len(frames):
        return f"{path} frame_count does not match its frame list"
    seen_names: set[str] = set()
    frame_bytes: list[bytes] = []
    for ordinal, (frame, role) in enumerate(
            zip(frames, _OPERATIONAL_EVIDENCE_ROLES, strict=True), 1):
        if (not isinstance(frame, dict) or set(frame) != {"file", "sha256", "role", "captured_utc"}
                or frame.get("role") != role or frame.get("file") != f"{ordinal:05d}.png"
                or not isinstance(frame.get("sha256"), str) or len(frame["sha256"]) != 64
                or not isinstance(frame.get("captured_utc"), str) or not frame["captured_utc"]
                or frame["file"] in seen_names):
            return f"{path} has malformed or wrongly ordered operational evidence frames"
        seen_names.add(frame["file"])
        png_path = path.parent / frame["file"]
        if not png_path.is_file():
            return f"{path} frame {frame['file']!r} is missing or its sha256 no longer matches"
        try:
            raw = png_path.read_bytes()
        except OSError as exc:
            return f"{path} frame {frame['file']!r} could not be read: {type(exc).__name__}"
        if _sha256(raw) != frame["sha256"]:
            return f"{path} frame {frame['file']!r} is missing or its sha256 no longer matches"
        try:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != tuple(frame_size):
                    return (f"{path} frame {frame['file']!r} has size {image.size}, not the "
                            "recorder's bound framebuffer")
        except Exception as exc:  # noqa: BLE001 -- malformed proof is always a refusal
            return f"{path} frame {frame['file']!r} is not a readable PNG: {type(exc).__name__}"
        frame_bytes.append(raw)
    if {entry.name for entry in path.parent.glob("*.png")} != seen_names:
        return f"{path} PNG set is not exactly the six hash-bound recorder frames"
    analyses = artifact.get("analyses")
    if not isinstance(analyses, dict) or set(analyses) != _OPERATIONAL_EVIDENCE_ANALYSES:
        return f"{path} lacks the exact v2 inline-composer/topology analyses"
    def top_claim(verdict) -> dict:
        return {"state": verdict.state, "distance": verdict.distance, "reason": verdict.reason,
                "grid": list(verdict.grid)}

    def surface_claim(surface) -> dict:
        return {
            "layout_id": surface.layout_id,
            "comment_rect": {key: getattr(surface.comment_rect, key)
                             for key in ("x0", "y0", "x1", "y1")},
            "send_rect": {key: getattr(surface.send_rect, key)
                          for key in ("x0", "y0", "x1", "y1")},
            "confirm_point": list(surface.confirm_point),
        }

    template_name = hinge_mod.HINGE_SPEC.templates.get("confirm")
    confirm_template = hinge_mod._load_template(template_name) if template_name else None
    if confirm_template is None:
        return f"{path} could not load the shipped confirmation template for replay"
    try:
        pre_top = confirm_scroll_top(frame_bytes[0], identity_band=identity_band)
        initial_top = confirm_scroll_top(frame_bytes[1], identity_band=identity_band)
        stable_top = confirm_scroll_top(frame_bytes[2], identity_band=identity_band)
        new_top = confirm_scroll_top(frame_bytes[4], identity_band=identity_band)
        sticky_top = confirm_scroll_top(frame_bytes[5], identity_band=identity_band)
        initial_surface = locate_inline_composer(frame_bytes[1], confirm_template, threshold=0.8)
        stable_surface = locate_inline_composer(frame_bytes[2], confirm_template, threshold=0.8)
        moved_surface = locate_inline_composer(frame_bytes[3], confirm_template, threshold=0.8)
        try:
            locate_inline_composer(frame_bytes[4], confirm_template, threshold=0.8)
        except ComposerDetectionError:
            composer_cleared = True
        else:
            composer_cleared = False
        identity = capture_profile_identity(frame_bytes[:3], identity_band=identity_band,
                                            grid=_IDENTITY_GRID)
        sticky_fingerprint = band_fingerprint(frame_bytes[5], identity_band=identity_band,
                                              grid=_IDENTITY_GRID)
        if not identity.known or identity.fingerprint is None:
            return f"{path} replay could not corroborate the item-1 profile identity"
        sticky_distance = fingerprint_distance(identity.fingerprint, sticky_fingerprint)
    except (ComposerDetectionError, IdentityError, ScrollTopError, ItemIndexError, ValueError) as exc:
        return f"{path} pure replay of hashed recorder frames refused: {type(exc).__name__}: {exc}"
    if (not pre_top.confirmed or not initial_top.refuted or not stable_top.refuted
            or not new_top.confirmed or not sticky_top.refuted):
        return f"{path} replayed top/sticky topology does not meet the v2 operational contract"
    if any(surface.layout_id != _COMPOSER_LAYOUT_ID
           for surface in (initial_surface, stable_surface, moved_surface)) or not composer_cleared:
        return f"{path} replayed inline-composer topology does not meet the v2 operational contract"
    if sticky_distance <= _IDENTITY_FALSE_MATCH_DISTANCE:
        return f"{path} replayed new sticky identity is not distinct from item 1"
    expected_analyses = {
        "confirmed_top_item1_pre": {"scroll_top": top_claim(pre_top)},
        "composer_initial_autofocused": {
            "composer": surface_claim(initial_surface), "scroll_top": top_claim(initial_top)},
        "composer_stable_autofocused": {
            "composer": surface_claim(stable_surface), "scroll_top": top_claim(stable_top)},
        "other_item_composer_moved": {"composer": surface_claim(moved_surface)},
        "item1_profile_identity": {
            "fingerprint": list(identity.fingerprint), "grid": list(identity.grid),
            "frame_index": identity.frame_index, "reason": identity.reason},
        "new_profile_top_clear": {
            "scroll_top": top_claim(new_top), "composer_absent": True},
        "new_sticky_identity": {
            "scroll_top": top_claim(sticky_top), "fingerprint": list(sticky_fingerprint),
            "grid": list(_IDENTITY_GRID), "distance_from_item1_profile": sticky_distance},
    }
    if analyses != expected_analyses:
        return f"{path} recorded analyses do not exactly match pure replay of its hashed frames"
    return None


def _verified_operational_checks(sessions: list[_SessionData]) -> dict:
    """Return traceable candidate-stage confirmations, or refuse.

    ``observe_timing_and_quota`` is satisfied here only by this tool's self-hashed preliminary
    passive artifact. It licenses a numeric calibration candidate for production OBSERVE; the
    separate ``observe_release_evidence`` config gate still blocks AUTO.
    """
    if not sessions:
        raise _MeasureRefused("no sessions supplied for supervised operational-check validation")
    expected_device = sessions[0].manifest.get("device")
    if (not isinstance(expected_device, dict)
            or set(expected_device) != set(_OPERATIONAL_EVIDENCE_DEVICE_KEYS)):
        raise _MeasureRefused("calibration sessions lack exact device/build evidence")
    identity_band = sessions[0].identity_band
    verified: dict[str, list[dict]] = {key: [] for key in _OPERATIONAL_CHECKS}
    operational_paths: dict[str, set[Path]] = {
        "item_1_inline_composer_identity": set(), "gesture_transport": set(),
    }
    for sess in sessions:
        if sess.manifest.get("device") != expected_device or sess.identity_band != identity_band:
            raise _MeasureRefused(
                "sessions disagree on device/build or identity-band before operational evidence "
                "can be bound")
        checks = sess.manifest.get("operational_checks") or {}
        for key in _OPERATIONAL_CHECKS:
            rec = checks.get(key)
            if (isinstance(rec, dict) and rec.get("confirmed") is True
                    and isinstance(rec.get("evidence"), str) and rec["evidence"].strip()
                    and isinstance(rec.get("recorded_utc"), str)
                    and rec["recorded_utc"].strip()):
                if key == "observe_timing_and_quota":
                    preliminary_reason = _preliminary_observe_check_reference_reason(
                        rec["evidence"].strip())
                    if preliminary_reason:
                        raise _MeasureRefused(
                            "the preliminary OBSERVE operational-check reference is invalid: "
                            + preliminary_reason)
                elif key == "entry_anchor_scroll_ledger":
                    entry_reason = _entry_anchor_reference_reason(sess, rec["evidence"].strip())
                    if entry_reason:
                        raise _MeasureRefused(
                            "the entry-anchor operational-check reference is invalid: "
                            + entry_reason)
                elif key in operational_paths:
                    operational_reason = _operational_evidence_reference_reason(
                        rec["evidence"].strip(), expected_device=expected_device,
                        identity_band=identity_band)
                    if operational_reason:
                        raise _MeasureRefused(
                            f"the {key} operational-check reference is invalid: "
                            + operational_reason)
                    manifest_path, manifest_reason = _operational_evidence_manifest_path(
                        rec["evidence"].strip())
                    if manifest_path is None:
                        raise _MeasureRefused(
                            f"the {key} operational-check manifest became unavailable after "
                            f"validation: {manifest_reason or 'path could not be resolved'}")
                    operational_paths[key].add(manifest_path)
                verified[key].append({
                    "session": str(sess.dir),
                    "evidence": rec["evidence"].strip(),
                    "recorded_utc": rec["recorded_utc"].strip(),
                })
    missing = [key for key, evidence in verified.items() if not evidence]
    if missing:
        raise _MeasureRefused(
            "the required supervised operational checks are not all confirmed with evidence "
            f"references (missing {missing}). Re-run a capture with "
            "--record-operational-checks after completing ops/RUNBOOK.md's four checks; numeric "
            "separation alone does not license a real targeting gesture")
    common_operational = (operational_paths["item_1_inline_composer_identity"]
                          & operational_paths["gesture_transport"])
    if not common_operational:
        raise _MeasureRefused(
            "item-1 identity and gesture transport must share one completed, hash-valid v2 "
            "operational-evidence manifest; recapture the six-frame auto-focused trace rather "
            "than combining unrelated assertions")
    return verified


def _verified_automated_circular_evidence(sessions: list[_SessionData]) -> dict:
    """Validate the *different* evidence contract for an explicitly accepted auto run.

    This never returns the supervised-check shape: callers and ledger readers must be able to
    distinguish a circular transport/vision trace from human ground truth.
    """
    records = []
    # `_cmd_measure` threads this back into `_require_collective_target_depths` so that check
    # asks for the depths THIS evidence's strategy actually claims, never a hardcoded pair
    # re-derived independently from a manifest. One measure invocation is one campaign, so every
    # session's already-verified strategy_id (below) must agree with every other.
    session_strategy_ids: set[str] = set()

    def target_proof_reason(action: dict) -> str | None:
        proof = action.get("target_scoped_prefix_proof")
        ordinal = proof.get("target_heart_ordinal") if isinstance(proof, dict) else None
        if (not isinstance(proof, dict)
                or proof.get("id") != _TARGET_SCOPED_PREFIX_PROOF_ID
                or proof.get("target_photo_model_item") != action.get("photo_model_item")
                or proof.get("predecessors_resolved") is not True
                or proof.get("target_crop_complete") is not True
                or proof.get("identity_known") is not True
                or isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1):
            return "missing or malformed target-scoped prefix proof"
        return None

    for sess in sessions:
        manifest = sess.manifest
        acceptance = manifest.get("automation_acceptance")
        manifest_strategy_id = manifest.get("automated_target_strategy_id")
        acceptance_strategy_id = (
            acceptance.get("target_strategy_id") if isinstance(acceptance, dict) else None)
        if (manifest.get("capture_mode") != "automated_circular_risk_accepted"
                or manifest.get("human_ground_truth") is not False
                or not isinstance(acceptance, dict)
                or acceptance.get("confirmation") != _UNATTENDED_CONFIRMATION
                or manifest_strategy_id not in _ACCEPTED_TARGET_STRATEGY_IDS
                or acceptance_strategy_id not in _ACCEPTED_TARGET_STRATEGY_IDS
                or manifest_strategy_id != acceptance_strategy_id
                or acceptance.get("not_independent_ground_truth") is not True
                or acceptance.get("not_supervised_operational_evidence") is not True):
            raise _MeasureRefused(
                f"{sess.dir} is not an exact unattended circular-risk capture manifest")
        strategy_id = manifest_strategy_id
        session_strategy_ids.add(strategy_id)
        target_scoped = manifest.get("capture_evidence_scope") == _TARGET_SCOPED_PREFIX_PROOF_ID
        for profile in manifest.get("profiles", []):
            if not isinstance(profile, dict):
                raise _MeasureRefused(f"{sess.dir} has a malformed automated profile trace")
            actions = profile.get("automated_actions") if isinstance(profile, dict) else None
            hearts = [a for a in actions or [] if a.get("action") == "automated_photo_heart"]
            ordinal = profile.get("ordinal") if isinstance(profile, dict) else None
            try:
                expected_items = _automated_composer_items_for_ordinal(
                    ordinal, strategy_id=strategy_id)
            except ValueError as exc:
                raise _MeasureRefused(f"{sess.dir} has invalid automated profile ordinal") from exc
            # Membership in the accepted set is not enough: every profile's own strategy id must
            # agree with the manifest/acceptance id already pinned above, so a session can never
            # mix strategies across its profiles.
            if (profile.get("target_strategy_id") != strategy_id
                    or (target_scoped and profile.get("capture_evidence_scope") != _TARGET_SCOPED_PREFIX_PROOF_ID)
                    or profile.get("composer_items") != list(expected_items)
                    or [a.get("photo_model_item") for a in hearts] != list(expected_items)
                    or any(a.get("post_tap_composer_verified") is not True for a in hearts)
                    or any(a.get("post_tap_item_relative_verified") is not True for a in hearts)
                    or not any(a.get("action") == "automated_pass" and
                               a.get("send_like_tapped") is False and
                               a.get("composer_clear_visible") is True for a in actions or [])):
                raise _MeasureRefused(
                    f"{sess.dir} profile {profile.get('ordinal')!r} lacks the exact automated "
                    "heart/composer/Pass trace")
            if target_scoped and any(target_proof_reason(action) is not None for action in hearts):
                raise _MeasureRefused(
                    f"{sess.dir} profile {profile.get('ordinal')!r} lacks exact target-scoped "
                    "prefix proof for its only automated heart")
        records.append({"session": str(sess.dir), "manifest_sha256": _sha256(
            (sess.dir / "manifest.json").read_bytes()), "human_ground_truth": False})
    if len(session_strategy_ids) > 1:
        raise _MeasureRefused(
            f"sessions use disagreeing automated target strategies {sorted(session_strategy_ids)}; "
            "one measure invocation must use exactly one target strategy across every session")
    return {"kind": "automated_circular_risk_accepted", "records": records,
            "not_supervised_operational_evidence": True,
            "target_strategy_id": next(iter(session_strategy_ids), _AUTOMATED_TARGET_STRATEGY_ID)}


def _approved_hybrid_decision(record: object, *, acceptance: dict, review: dict,
                              action: str, item: int | None) -> bool:
    """Return whether one ledger member is the exact approved action/item decision."""
    plan = record.get("action_plan") if isinstance(record, dict) else None
    return (isinstance(record, dict) and record.get("decision") == "approved"
            and record.get("source") == acceptance.get("reviewer_source")
            and record.get("reviewer") == acceptance.get("reviewer")
            and record.get("human_ground_truth") is False
            and isinstance(plan, dict) and plan.get("action") == action
            and plan.get("photo_model_item") == item
            and isinstance(record.get("checkpoint_evidence_sha256"), str)
            and isinstance(record.get("frame_sha256"), str)
            and record in review.get("decisions", []))


def _exact_hybrid_terminal_checkpoint(action: object, *, session_dir: Path, manifest: dict,
                                      acceptance: dict, review: dict,
                                      expected_action: str,
                                      expected_point: list[int] | None) -> bool:
    """Authenticate the checkpoint JSON, PNG, ledger and exact terminal action plan."""
    if not isinstance(action, dict):
        return False
    checks = action.get("review_checkpoints")
    before = checks.get("before") if isinstance(checks, dict) else None
    if (not _approved_hybrid_decision(
            before, acceptance=acceptance, review=review,
            action=expected_action, item=None)
            or not isinstance(before, dict)):
        return False
    checkpoint_file, frame_file = before.get("checkpoint_file"), before.get("frame_file")
    if not isinstance(checkpoint_file, str) or not isinstance(frame_file, str):
        return False
    try:
        review_dir = (session_dir / "hybrid_review").resolve(strict=True)
        checkpoint_path = Path(checkpoint_file).resolve(strict=True)
        checkpoint_frame_path = Path(frame_file).resolve(strict=True)
        if (checkpoint_path.parent != review_dir or checkpoint_frame_path.parent != review_dir
                or not checkpoint_path.is_file() or not checkpoint_frame_path.is_file()):
            return False
        checkpoint_raw = checkpoint_path.read_bytes()
        checkpoint = json.loads(checkpoint_raw)
        checkpoint_frame_sha256 = _sha256(checkpoint_frame_path.read_bytes())
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False
    if not isinstance(checkpoint, dict):
        return False
    checkpoint_body = dict(checkpoint)
    checkpoint_body.pop("evidence_sha256", None)
    plan = checkpoint.get("action_plan")
    predicates = plan.get("predicates") if isinstance(plan, dict) else None
    frame = checkpoint.get("frame")
    config_provenance = manifest.get("config_provenance")
    config_sha256 = (config_provenance.get("sha256")
                     if isinstance(config_provenance, dict) else None)
    expected_source = ("calibration-only verified-composer Send transport"
                       if expected_action == "automated_send_priority_like"
                       else "calibration-only verified-composer Pass transport")
    return (
        before.get("checkpoint_sha256") == _sha256(checkpoint_raw)
        and checkpoint.get("evidence_sha256") == before.get("checkpoint_evidence_sha256")
        and checkpoint.get("evidence_sha256") == _canonical_json_digest(checkpoint_body)
        and checkpoint.get("schema_version") == _HYBRID_CHECKPOINT_SCHEMA_VERSION
        and checkpoint.get("kind") == _HYBRID_CHECKPOINT_KIND
        and checkpoint.get("config_sha256") == config_sha256
        and checkpoint.get("human_ground_truth") is False
        and checkpoint.get("claimed_state") == "composer_open_before_pass"
        and before.get("claimed_state") == checkpoint.get("claimed_state")
        and plan == before.get("action_plan")
        and isinstance(plan, dict)
        and plan.get("action") == expected_action
        and plan.get("photo_model_item") is None
        and plan.get("point") == expected_point
        and plan.get("point_source") == expected_source
        and isinstance(predicates, dict)
        and predicates.get(
            "inline_composer_and_selected_photo_verified_before_action") is True
        and predicates.get("send_like_tapped") is False
        and predicates.get("forbidden_zone_guarded_transport") == "HingeDriver._tap"
        and isinstance(frame, dict)
        and frame.get("file") == checkpoint_frame_path.name
        and frame.get("sha256") == before.get("frame_sha256")
        and frame.get("sha256") == checkpoint_frame_sha256
        and frame.get("sha256") == action.get("pre_frame_sha256")
    )


def _verified_hybrid_reviewed_evidence(sessions: list[_SessionData]) -> dict:
    """Validate the distinct AI-reviewed transport provenance; never call it human evidence."""
    records = []
    required_reviewer = {"source", "id", "model", "version", "process"}
    # See the matching comment in `_verified_automated_circular_evidence`: threaded back into
    # `_require_collective_target_depths` so the depth requirement matches this evidence's own
    # strategy rather than a hardcoded pair or an independent manifest re-derivation.
    session_strategy_ids: set[str] = set()

    def target_proof_reason(action: dict) -> str | None:
        proof = action.get("target_scoped_prefix_proof")
        ordinal = proof.get("target_heart_ordinal") if isinstance(proof, dict) else None
        if (not isinstance(proof, dict)
                or proof.get("id") != _TARGET_SCOPED_PREFIX_PROOF_ID
                or proof.get("target_photo_model_item") != action.get("photo_model_item")
                or proof.get("predecessors_resolved") is not True
                or proof.get("target_crop_complete") is not True
                or proof.get("identity_known") is not True
                or isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1):
            return "missing or malformed target-scoped prefix proof"
        return None

    for sess in sessions:
        manifest = sess.manifest
        acceptance, review = manifest.get("automation_acceptance"), manifest.get("hybrid_review")
        manifest_strategy_id = manifest.get("automated_target_strategy_id")
        acceptance_strategy_id = (
            acceptance.get("target_strategy_id") if isinstance(acceptance, dict) else None)
        if (manifest.get("capture_mode") != "hybrid_ai_reviewed_automation"
                or manifest.get("human_ground_truth") is not False
                or not isinstance(acceptance, dict)
                or acceptance.get("confirmation") != _HYBRID_REVIEW_CONFIRMATION
                or manifest_strategy_id not in _ACCEPTED_TARGET_STRATEGY_IDS
                or acceptance_strategy_id not in _ACCEPTED_TARGET_STRATEGY_IDS
                or manifest_strategy_id != acceptance_strategy_id
                or acceptance.get("reviewer_protocol") != "stdin_checkpoint_sha256_v1"
                or acceptance.get("reviewer_source") != "external_ai_review"
                or not isinstance(acceptance.get("reviewer"), dict)
                or not required_reviewer.issubset(acceptance["reviewer"])
                or any(not isinstance(acceptance["reviewer"][key], str)
                       or not acceptance["reviewer"][key].strip() for key in required_reviewer)
                or not isinstance(review, dict) or review.get("schema_version") != 1
                or review.get("protocol") != "stdin_checkpoint_sha256_v1"
                or review.get("reviewer") != acceptance["reviewer"]
                or review.get("human_ground_truth") is not False
                or not isinstance(review.get("decisions"), list)):
            raise _MeasureRefused(f"{sess.dir} is not an exact hybrid AI-reviewed capture manifest")
        strategy_id = manifest_strategy_id
        session_strategy_ids.add(strategy_id)
        target_scoped = manifest.get("capture_evidence_scope") == _TARGET_SCOPED_PREFIX_PROOF_ID
        # See _SEND_LIKE_CONFIRMATION: a capture ends each profile with Pass-without-send unless
        # the owner explicitly accepted real sends at capture time. Read the acceptance from the
        # manifest so a send trace can never authorize itself.
        send_like_accepted = acceptance.get("send_like_accepted")
        if send_like_accepted is not None and type(send_like_accepted) is not bool:
            raise _MeasureRefused(f"{sess.dir} has a non-boolean send_like_accepted acceptance")
        send_like_accepted = bool(send_like_accepted)
        if send_like_accepted and acceptance.get("send_like_confirmation") != _SEND_LIKE_CONFIRMATION:
            raise _MeasureRefused(
                f"{sess.dir} records accepted real sends without the exact send-like confirmation")
        if not send_like_accepted and acceptance.get("send_like_confirmation") is not None:
            raise _MeasureRefused(
                f"{sess.dir} carries a send-like confirmation without accepting real sends")
        terminal_action = ("automated_send_priority_like" if send_like_accepted
                           else "automated_pass")
        declared_terminal = acceptance.get("terminal_advance_action")
        if declared_terminal is not None and declared_terminal != terminal_action:
            raise _MeasureRefused(
                f"{sess.dir} declares a terminal advance action that contradicts its acceptance")

        profiles_by_ordinal = {profile.ordinal: profile for profile in sess.profiles}

        def has_exact_send_summary(
                action: object, *, ordinal: int,
                send_like_accepted=send_like_accepted,
                profiles_by_ordinal=profiles_by_ordinal, session_dir=sess.dir,
                manifest=manifest, acceptance=acceptance, review=review) -> bool:
            """Authenticate an owner-accepted REAL Send Priority Like terminal action.

            Only reachable when this manifest carries the explicit send-like acceptance above.
            It is a separate exact shape rather than a loosened Pass check: the composer and
            selected photo are still proven before the tap, the transport is still the production
            upsell-dismiss + landed-verification chain (never the paid Rose control), and the
            advance frame is still bound.
            """
            if not isinstance(action, dict) or not send_like_accepted:
                return False
            profile_data = profiles_by_ordinal.get(ordinal)
            action_predicates, transport = action.get("predicates"), action.get("transport")
            point = action.get("confirm_point")
            if (profile_data is None
                    or not isinstance(action_predicates, dict) or not isinstance(transport, list)
                    or transport != ["HingeDriver._tap(confirm_point)",
                                     "HingeDriver._handle_rose_upsell",
                                     "HingeDriver._verify_like_landed"]
                    or action.get("send_like_tapped") is not True
                    or action_predicates.get("send_like_tapped") is not True
                    or action_predicates.get(
                        "inline_composer_and_selected_photo_verified_before_action") is not True
                    or action_predicates.get("like_landed_verified") is not True
                    or action.get("post_frame_sha256") != _sha256(profile_data.profile_advance_clear)
                    or not profile_data.profile_advance_identity):
                return False
            if (not isinstance(point, list) or len(point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in point)):
                return False
            return _exact_hybrid_terminal_checkpoint(
                action, session_dir=session_dir, manifest=manifest, acceptance=acceptance,
                review=review, expected_action="automated_send_priority_like",
                expected_point=point)

        def has_exact_pass_summary(
                action: object, *, ordinal: int,
                profiles_by_ordinal=profiles_by_ordinal, session_dir=sess.dir,
                manifest=manifest, acceptance=acceptance, review=review) -> bool:
            """Authenticate the pre-summary hybrid Pass format without inventing a summary."""
            if not isinstance(action, dict):
                return False
            missing = object()
            sent = action.get("send_like_tapped", missing)
            clear = action.get("composer_clear_visible", missing)
            if ((sent is not missing or clear is not missing)
                    and not (sent is False and clear is True)):
                return False
            profile_data = profiles_by_ordinal.get(ordinal)
            if (profile_data is None
                    or not _exact_hybrid_terminal_checkpoint(
                        action, session_dir=session_dir, manifest=manifest,
                        acceptance=acceptance, review=review,
                        expected_action="automated_pass", expected_point=None)):
                return False
            action_predicates, transport = action.get("predicates"), action.get("transport")
            if (not isinstance(action_predicates, dict) or not isinstance(transport, list)
                    or any(not isinstance(step, str) for step in transport)
                    or len(transport) < 2 or transport[-2:] != ["HingeDriver._locate_button(pass)",
                                                                 "HingeDriver._tap"]
                    or action_predicates.get("inline_composer_and_selected_photo_verified_before_action") is not True
                    or action_predicates.get("inline_composer_structurally_confirmed_before_action") is not True
                    or action_predicates.get("edge_back_transport_performed") is not True
                    or action_predicates.get("pass_vision_relocated_before_guarded_tap") is not True
                    or action_predicates.get("send_like_tapped") is not False
                    or action_predicates.get("deck_frame_changed") is not True
                    or action_predicates.get("composer_clear_visible") is not True
                    or action_predicates.get("new_profile_top_confirmed") is not True
                    or action.get("post_frame_sha256") != _sha256(profile_data.profile_advance_clear)
                    or not profile_data.profile_advance_identity):
                return False
            return True

        for profile in manifest.get("profiles", []):
            if not isinstance(profile, dict):
                raise _MeasureRefused(f"{sess.dir} has a malformed hybrid profile trace")
            actions = profile.get("automated_actions") if isinstance(profile, dict) else None
            hearts = [action for action in actions or []
                      if isinstance(action, dict) and action.get("action") == "automated_photo_heart"]
            try:
                expected_items = _automated_composer_items_for_ordinal(
                    profile.get("ordinal"), strategy_id=strategy_id)
            except ValueError as exc:
                raise _MeasureRefused(f"{sess.dir} has invalid hybrid profile ordinal") from exc
            # Every profile's own strategy id must agree with the manifest/acceptance id already
            # pinned above -- membership in the accepted set alone would let one session mix
            # strategies across its profiles.
            if (not isinstance(profile, dict)
                    or profile.get("action_evidence_mode") != "hybrid_ai_reviewed_automation"
                    or profile.get("target_strategy_id") != strategy_id
                    or (target_scoped and profile.get("capture_evidence_scope") != _TARGET_SCOPED_PREFIX_PROOF_ID)
                    or profile.get("composer_items") != list(expected_items)
                    or [action.get("photo_model_item") for action in hearts] != list(expected_items)
                    or any(action.get("post_tap_composer_verified") is not True
                           or action.get("post_tap_item_relative_verified") is not True
                           or not isinstance(action.get("review_checkpoints"), dict)
                           or not _approved_hybrid_decision(
                               action["review_checkpoints"].get("before"),
                               acceptance=acceptance, review=review,
                               action="automated_photo_heart",
                               item=action.get("photo_model_item"))
                           or not _approved_hybrid_decision(
                               action["review_checkpoints"].get("after"),
                               acceptance=acceptance, review=review,
                               action="review_heart_result",
                               item=action.get("photo_model_item"))
                           for action in hearts)):
                raise _MeasureRefused(f"{sess.dir} profile {profile.get('ordinal')!r} lacks exact "
                                      "approved hybrid heart checkpoints")
            if target_scoped and any(target_proof_reason(action) is not None for action in hearts):
                raise _MeasureRefused(f"{sess.dir} profile {profile.get('ordinal')!r} lacks exact "
                                      "target-scoped prefix proof")
            passes = [action for action in actions or []
                      if isinstance(action, dict) and action.get("action") == terminal_action]
            summary_ok = (has_exact_send_summary if send_like_accepted else has_exact_pass_summary)
            if (len(passes) != 1 or not summary_ok(passes[0], ordinal=profile.get("ordinal"))
                    or not isinstance(passes[0].get("review_checkpoints"), dict)
                    or not _approved_hybrid_decision(
                        passes[0]["review_checkpoints"].get("before"),
                        acceptance=acceptance, review=review,
                        action=terminal_action, item=None)):
                raise _MeasureRefused(
                    f"{sess.dir} profile {profile.get('ordinal')!r} lacks exact approved hybrid "
                    f"{'Send Priority Like' if send_like_accepted else 'Pass'} checkpoint")
            other_action = ("automated_pass" if send_like_accepted
                            else "automated_send_priority_like")
            if any(isinstance(action, dict) and action.get("action") == other_action
                   for action in actions or []):
                raise _MeasureRefused(
                    f"{sess.dir} profile {profile.get('ordinal')!r} mixes Pass and Send terminal "
                    "actions in one ledger")
        records.append({"session": str(sess.dir), "manifest_sha256": _sha256(
            (sess.dir / "manifest.json").read_bytes()), "human_ground_truth": False,
            "reviewer": acceptance["reviewer"]})
    if len(session_strategy_ids) > 1:
        raise _MeasureRefused(
            f"sessions use disagreeing automated target strategies {sorted(session_strategy_ids)}; "
            "one measure invocation must use exactly one target strategy across every session")
    return {"kind": "hybrid_ai_reviewed_automation", "records": records,
            "not_supervised_operational_evidence": True, "human_ground_truth": False,
            "target_strategy_id": next(iter(session_strategy_ids), _AUTOMATED_TARGET_STRATEGY_ID)}


def _unattended_review_reference_reason(reference: str, sessions: list[_SessionData], *,
                                        config_path: str) -> str | None:
    """Authenticate the separate stdlib review without re-running its capture/vision logic.

    The reviewer is purposely invoked before measure as a distinct process.  This small reader
    only checks its self-hash and the immutable bindings that make its conclusion applicable to
    *these* capture directories and config bytes; it cannot turn circular evidence into human
    evidence or make an AUTO release valid.
    """
    if not isinstance(reference, str) or not reference.strip():
        return "--unattended-review is required for automated circular-risk measurement"
    path = Path(reference)
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read unattended review artifact {path}: {exc}"
    if not isinstance(artifact, dict):
        return "unattended review artifact is not a mapping"
    claimed = artifact.get("evidence_sha256")
    body = dict(artifact)
    body.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_json_digest(body):
        return "unattended review artifact self-hash does not match its contents"
    if (body.get("schema_version") != _UNATTENDED_REVIEW_SCHEMA_VERSION
            or body.get("kind") != _UNATTENDED_REVIEW_KIND
            or body.get("human_ground_truth") is not False
            or body.get("not_independent_ground_truth") is not True):
        return "unattended review artifact has unsupported provenance/schema"
    reviewer = body.get("reviewer")
    if (not isinstance(reviewer, dict)
            or reviewer.get("deterministic_second_process") is not True
            or reviewer.get("imports_capture_or_vision_stack") is not False
            or not isinstance(reviewer.get("implementation_sha256"), str)):
        return "unattended review does not attest independent deterministic reviewer provenance"
    try:
        config_digest = _sha256(Path(config_path).read_bytes())
    except OSError as exc:
        return f"cannot hash config used for unattended review binding: {exc}"
    config = body.get("config")
    if not isinstance(config, dict) or config.get("sha256") != config_digest:
        return "unattended review config digest does not match this measure config"
    captures = body.get("captures")
    if not isinstance(captures, list) or len(captures) != len(sessions):
        return "unattended review does not cover exactly the supplied capture sessions"
    expected: dict[str, tuple[str, dict, list[int], str]] = {}
    for session in sessions:
        manifest_path = session.dir / "manifest.json"
        try:
            manifest_digest = _sha256(manifest_path.read_bytes())
        except OSError as exc:
            return f"cannot rehash supplied capture manifest {manifest_path}: {exc}"
        device = session.manifest.get("device")
        # The evidence scope belongs to THIS session: mixed-scope invocations are supported
        # (a legacy closed-set session may be measured alongside a target-scoped one), so it is
        # bound per session here rather than read back from the loop variable below.
        expected[str(session.dir.resolve())] = (
            manifest_digest, device,
            session.manifest.get("frame_size_px"),
            session.manifest.get("capture_evidence_scope", "closed_set_profile_v1"),
        )
    seen: set[str] = set()
    for record in captures:
        if not isinstance(record, dict) or record.get("human_ground_truth") is not False:
            return "unattended review contains malformed/non-automated capture record"
        session_path = record.get("session")
        if not isinstance(session_path, str) or session_path in seen or session_path not in expected:
            return "unattended review capture session does not exactly match measure input"
        seen.add(session_path)
        manifest_digest, device, frame_size, expected_scope = expected[session_path]
        if (record.get("manifest_sha256") != manifest_digest
                or record.get("device") != device
                or record.get("frame_size_px") != frame_size
                or record.get("config_sha256") != config_digest
                or record.get("capture_evidence_scope", "closed_set_profile_v1") != expected_scope):
            return "unattended review capture binding differs from exact manifest/device/frame/config"
    if seen != set(expected):
        return "unattended review omitted a supplied capture session"
    return None


# --- top level -------------------------------------------------------------------------

def _cmd_measure(args: argparse.Namespace) -> None:
    cfg = cfg_mod.load(args.config)
    driver = HingeDriver(cfg)  # inert: no adb, no touch transport -- measure never opens one
    if driver.identity_band is None or driver.content_band is None:
        print("ERROR: this app's effective identity_band/content_band is None. Targeting "
              "calibration is meaningless without both.", file=sys.stderr)
        sys.exit(1)
    effective_identity = tuple(driver.identity_band)
    effective_content = tuple(driver.content_band)
    like_template = driver._template("like")
    confirm_template = driver._template("confirm")
    like_threshold = hinge_mod._LIKE_MATCH_THRESHOLD

    configured_serial = (getattr(cfg, "apps", {}) or {}).get("hinge", {}).get("serial")
    if not isinstance(configured_serial, str) or not configured_serial.strip():
        print("ERROR: apps.hinge.serial is not set in config.yaml; the calibration block's "
              "`device` field must exactly equal it.", file=sys.stderr)
        sys.exit(1)

    loaded: list[_SessionData] = []
    for raw in args.sessions:
        sess_dir = Path(raw)
        try:
            loaded.append(_load_session(sess_dir))
        except Exception as exc:  # noqa: BLE001 -- report clearly, exit non-zero
            print(f"ERROR: could not load session {sess_dir}: {exc}", file=sys.stderr)
            sys.exit(1)

    for sess in loaded:
        if sess.identity_band != effective_identity or sess.content_band != effective_content:
            print(
                f"ERROR: session {sess.dir} was captured with identity_band={sess.identity_band} "
                f"content_band={sess.content_band}, which does not exactly match the current "
                f"effective identity_band={effective_identity} content_band={effective_content}. "
                "A calibration cannot be reused after either crop changes -- recapture.",
                file=sys.stderr)
            sys.exit(1)
        if sess.device_serial != configured_serial:
            print(
                f"ERROR: session {sess.dir} was captured on serial {sess.device_serial!r}, "
                f"which does not match apps.hinge.serial {configured_serial!r}. A calibration "
                "is only valid for the exact device it was measured on.", file=sys.stderr)
            sys.exit(1)

    calibration_sessions = [s for s in loaded if s.split == _SPLIT_CALIBRATION]
    heldout_sessions = [s for s in loaded if s.split == _SPLIT_HELDOUT]
    if not calibration_sessions or not heldout_sessions:
        print(
            "ERROR: need at least one session of EACH split. The held-out set must be "
            "SEPARATELY COLLECTED, never drawn retroactively from one pool "
            f"(calibration sessions: {len(calibration_sessions)}, heldout sessions: "
            f"{len(heldout_sessions)}).", file=sys.stderr)
        sys.exit(1)

    calibration_profiles = _split_profiles(calibration_sessions)
    heldout_profiles = _split_profiles(heldout_sessions)
    # `_require_collective_target_depths` runs further below, once the evidence mode has resolved
    # this run's exact target strategy id -- see the comment at that call site.
    calib_ids = {p.profile_id for p in calibration_profiles}
    heldout_ids = {p.profile_id for p in heldout_profiles}
    overlap = calib_ids & heldout_ids
    if overlap:
        print(
            f"ERROR: profile id(s) {sorted(overlap)} appear in BOTH the calibration and "
            "held-out splits. The held-out set must be a separately collected, disjoint set of "
            "real profiles -- relabel or recapture so no id is shared.", file=sys.stderr)
        sys.exit(1)

    for split_name, profiles in ((_SPLIT_CALIBRATION, calibration_profiles),
                                 (_SPLIT_HELDOUT, heldout_profiles)):
        ids = [p.profile_id for p in profiles]
        duplicates = sorted({profile_id for profile_id in ids if ids.count(profile_id) > 1})
        if duplicates:
            print(
                f"ERROR: profile id(s) {duplicates} are duplicated within the {split_name} "
                "split. Every real profile needs one globally unique id; duplicate labels "
                "silently suppress required foreign-profile comparisons.", file=sys.stderr)
            sys.exit(1)

    evidence_keys = ("serial", "model", "display_w", "display_h", "density",
                     "hinge_package", "hinge_version_name")
    device_evidence = []
    for sess in loaded:
        device = sess.manifest.get("device") or {}
        missing = [key for key in evidence_keys if device.get(key) in (None, "")]
        if missing:
            print(f"ERROR: session {sess.dir} is missing device/build evidence {missing}.",
                  file=sys.stderr)
            sys.exit(1)
        device_evidence.append(tuple(device[key] for key in evidence_keys))
    if len(set(device_evidence)) != 1:
        print(
            "ERROR: calibration and held-out sessions disagree on device model, display size, "
            "density, package, or Hinge build. Those change the measured pixel-distance space; "
            "recapture both splits under one exact device/build state.", file=sys.stderr)
        sys.exit(1)

    try:
        unattended_evidence = bool(getattr(args, "accept_automated_circular_evidence", False))
        hybrid_evidence = bool(getattr(args, "accept_hybrid_reviewed_evidence", False))
        if unattended_evidence and hybrid_evidence:
            raise _MeasureRefused("choose exactly one automated evidence mode, never a mixed-mode campaign")
        if unattended_evidence:
            if getattr(args, "confirmation", "") != _UNATTENDED_CONFIRMATION:
                raise _MeasureRefused("--accept-automated-circular-evidence requires exact "
                                      f"--confirmation {_UNATTENDED_CONFIRMATION!r}")
            review_reason = _unattended_review_reference_reason(
                getattr(args, "unattended_review", ""), loaded, config_path=args.config)
            if review_reason is not None:
                raise _MeasureRefused(review_reason)
            operational_check_evidence = _verified_automated_circular_evidence(loaded)
            operational_check_evidence["independent_review"] = {
                "artifact": str(getattr(args, "unattended_review", "")).strip(),
                "sha256": _sha256(Path(getattr(args, "unattended_review", "")).read_bytes()),
                "human_ground_truth": False,
            }
        elif hybrid_evidence:
            if getattr(args, "confirmation", "") != _HYBRID_REVIEW_CONFIRMATION:
                raise _MeasureRefused("--accept-hybrid-reviewed-evidence requires exact "
                                      f"--confirmation {_HYBRID_REVIEW_CONFIRMATION!r}")
            review_reason = _unattended_review_reference_reason(
                getattr(args, "unattended_review", ""), loaded, config_path=args.config)
            if review_reason is not None:
                raise _MeasureRefused(review_reason)
            operational_check_evidence = _verified_hybrid_reviewed_evidence(loaded)
            operational_check_evidence["independent_review"] = {
                "artifact": str(getattr(args, "unattended_review", "")).strip(),
                "sha256": _sha256(Path(getattr(args, "unattended_review", "")).read_bytes()),
                "human_ground_truth": False,
            }
        else:
            operational_check_evidence = _verified_operational_checks(loaded)
        # Use the strategy id `_verified_automated_circular_evidence`/`_verified_hybrid_reviewed_
        # evidence` already resolved and cross-checked above -- never re-derive it from a
        # manifest independently here, and never default it away when a capture's own strategy is
        # known. Supervised (manual) evidence carries no such id, so `.get` keeps today's
        # alternating-strategy default for that path exactly as before.
        measure_strategy_id = operational_check_evidence.get(
            "target_strategy_id", _AUTOMATED_TARGET_STRATEGY_ID)
        # One composer pair per profile is sufficient, provided every split with at least two
        # profiles collectively reaches every depth `measure_strategy_id` targets.
        _require_collective_target_depths(
            calibration_profiles, split=_SPLIT_CALIBRATION, strategy_id=measure_strategy_id)
        _require_collective_target_depths(
            heldout_profiles, split=_SPLIT_HELDOUT, strategy_id=measure_strategy_id)
        _validate_profile_advance_clears(
            calibration_profiles + heldout_profiles, identity_band=effective_identity,
            confirm_template=confirm_template)
    except _MeasureRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- IDENTITY ---
    try:
        calib_identity_samples = _profile_identity_samples(
            calibration_profiles, identity_band=effective_identity)
        heldout_identity_samples = _profile_identity_samples(
            heldout_profiles, identity_band=effective_identity)
        # Human labels are evidence aids, not an identity oracle. Detect the same real profile
        # reused under two different labels by comparing the independently captured card
        # representatives across splits at production's exact grid. The known 2.565
        # different-profile collision is only a hard upper limit; anything at/under it is too
        # ambiguous to certify as held out and therefore forces fresh collection.
        calib_representatives = [s for s in calib_identity_samples
                                 if s.origin == "representative"]
        heldout_representatives = [s for s in heldout_identity_samples
                                   if s.origin == "representative"]
        cross_split_near_matches = []
        for calibration_sample in calib_representatives:
            for heldout_sample in heldout_representatives:
                distance = _checked_distance(
                    fingerprint_distance(calibration_sample.fingerprint,
                                         heldout_sample.fingerprint),
                    context=(f"cross-split identity {calibration_sample.profile_id!r} against "
                             f"{heldout_sample.profile_id!r}"))
                if distance <= _IDENTITY_FALSE_MATCH_DISTANCE:
                    cross_split_near_matches.append({
                        "calibration_profile": calibration_sample.profile_id,
                        "heldout_profile": heldout_sample.profile_id,
                        "distance": distance,
                    })
        if cross_split_near_matches:
            nearest = min(rec["distance"] for rec in cross_split_near_matches)
            raise _MeasureRefused(
                f"{len(cross_split_near_matches)} calibration/held-out representative pair(s) "
                f"sit at/under the known {_IDENTITY_FALSE_MATCH_DISTANCE} ambiguous identity "
                f"distance (nearest {nearest:.3f}). The same real profile may have been reused "
                "under a different label, so the held-out population is not provably disjoint; "
                "collect fresh held-out profiles")
        raw_identity_bound, identity_report = _freeze_identity_bound(calib_identity_samples)
    except _MeasureRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)

    if raw_identity_bound is None:
        print(f"REFUSED: identity bound not separable on the calibration split: "
              f"{identity_report.get('reason')}", file=sys.stderr)
        sys.exit(1)
    identity_bound = _round_down_below(
        raw_identity_bound, above=identity_report["max_same_profile_distance"],
        below=_IDENTITY_FALSE_MATCH_DISTANCE)
    identity_report["frozen_bound"] = identity_bound

    try:
        identity_ok, identity_msg, identity_heldout_detail = _apply_identity_bound(
            heldout_identity_samples, identity_bound)
    except _MeasureRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
    if not identity_ok:
        print(f"REFUSED: identity bound failed held-out verification: {identity_msg}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[OK] identity_match_max_dist={identity_bound:.4f} "
          f"(calibration margin {identity_report['margin']:.3f}; held-out: {identity_msg})")

    # --- INLINE COMPOSER ITEMS ---
    try:
        calib_payloads = _build_profile_payloads(
            calibration_profiles, content_band=effective_content,
            identity_band=effective_identity, like_template=like_template,
            like_threshold=like_threshold)
        heldout_payloads = _build_profile_payloads(
            heldout_profiles, content_band=effective_content,
            identity_band=effective_identity, like_template=like_template,
            like_threshold=like_threshold)
        calib_inline_records = _inline_distance_records(
            calib_payloads, confirm_template=confirm_template)
        heldout_inline_records = _inline_distance_records(
            heldout_payloads, confirm_template=confirm_template)
        raw_inline_bound, inline_report = _freeze_inline_bound(calib_inline_records)
    except _MeasureRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)

    if raw_inline_bound is None:
        print(f"REFUSED: inline-item bound not separable on the calibration split: "
              f"{inline_report.get('reason')}", file=sys.stderr)
        sys.exit(1)
    inline_bound = _round_down_below(
        raw_inline_bound, above=inline_report["max_intended_distance"],
        below=_INLINE_FALSE_MATCH_DISTANCE)
    inline_report["frozen_bound"] = inline_bound

    try:
        inline_ok, inline_msg, inline_heldout_detail = _apply_inline_bound(
            heldout_inline_records, inline_bound)
    except _MeasureRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
    if not inline_ok:
        print(f"REFUSED: inline-item bound failed held-out verification: {inline_msg}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[OK] inline_item_max_dist={inline_bound:.4f} "
          f"(calibration margin {inline_report['margin']:.3f}; held-out: {inline_msg})")

    # Build and round-trip the exact payload before writing a ledger or printing success.  The
    # runtime driver's independent parser is the gesture-boundary authority, while safe_dump /
    # safe_load protects unusual but legal serials and session-directory names from YAML syntax.
    newest = max(loaded, key=_session_sort_key)
    measured_at = datetime.now(timezone.utc)
    calibrated_at = (f"{measured_at.isoformat()} — measured from session(s): "
                     + ", ".join(s.dir.name for s in loaded))
    calibration_block = {
        "schema_version": _CALIBRATION_SCHEMA_VERSION,
        "device": configured_serial,
        "hinge_version_name": device_evidence[0][6],
        "frame_size_px": [device_evidence[0][2], device_evidence[0][3]],
        "composer_layout_id": _COMPOSER_LAYOUT_ID,
        "item_selection_policy_id": PHOTO_ONLY_POLICY_ID,
        "identity_match_max_dist": identity_bound,
        "inline_item_max_dist": inline_bound,
        "calibrated_at": calibrated_at,
        "identity_band": list(effective_identity),
        "content_band": list(effective_content),
    }
    try:
        _validate_v3_calibration_block(calibration_block)
    except _MeasureRefused as exc:
        print(f"REFUSED: the measured v3 calibration block is invalid: {exc}", file=sys.stderr)
        sys.exit(1)
    parsed, parse_reason = hinge_mod._parse_targeting_calibration(
        calibration_block, configured_serial, identity_band=effective_identity,
        content_band=effective_content)
    if parsed is None:
        print("REFUSED: runtime rejected the measured v3 calibration block: "
              f"{parse_reason}", file=sys.stderr)
        sys.exit(1)
    yaml_document = yaml.safe_dump(
        {"apps": {"hinge": {"targeting_calibration": calibration_block}}},
        sort_keys=False, allow_unicode=True)
    try:
        round_trip = yaml.safe_load(yaml_document)["apps"]["hinge"]["targeting_calibration"]
    except Exception as exc:  # noqa: BLE001 -- malformed output must never be printed as success
        print(f"REFUSED: generated calibration YAML did not parse: {exc}", file=sys.stderr)
        sys.exit(1)
    if round_trip != calibration_block:
        print("REFUSED: generated calibration YAML did not round-trip byte-for-value; refusing "
              "to print an ambiguous paste block.", file=sys.stderr)
        sys.exit(1)

    # --- ledger ---
    ledger = {
        "tool_version": _TOOL_VERSION,
        "measured_at_utc": measured_at.isoformat(),
        "sessions": [{"dir": str(s.dir), "split": s.split, "device_serial": s.device_serial}
                     for s in loaded],
        "identity_band": list(effective_identity),
        "content_band": list(effective_content),
        "identity": {
            "calibration": identity_report,
            "heldout": {"ok": identity_ok, "reason": identity_msg, **identity_heldout_detail},
            "frozen_bound": identity_bound,
        },
        "inline": {
            "calibration": inline_report,
            "heldout": {"ok": inline_ok, "reason": inline_msg, **inline_heldout_detail},
            "frozen_bound": inline_bound,
        },
        "device_serial": configured_serial,
        "device_evidence": dict(zip(evidence_keys, device_evidence[0], strict=True)),
        "operational_checks": operational_check_evidence,
        "measurement_evidence_mode": (
            "automated_circular_risk_accepted" if unattended_evidence
            else "hybrid_ai_reviewed_automation" if hybrid_evidence
            else "supervised_operational_evidence"),
        "targeting_calibration": calibration_block,
        "auto_release_status": "pending_post_calibration_production_observe_validation",
        "auto_release_requirements": list(_POST_CALIBRATION_OBSERVE_REQUIREMENTS),
    }
    ledger_path = newest.dir / "measurement_ledger.json"
    atomic_write_private_text(
        ledger_path, json.dumps(ledger, indent=2) + "\n", parent=newest.dir)
    print(f"\nWrote measurement ledger to {ledger_path}")

    print("\n" + "=" * 78)
    if unattended_evidence or hybrid_evidence:
        print("AUTOMATED CIRCULAR-RISK NUMERIC CANDIDATE. This is NOT supervised operational "
              "evidence and does NOT release AUTO. The measurement ledger carries that exact "
              "provenance; retain it with any review of this candidate.")
    else:
        print("NUMERIC CALIBRATION CANDIDATE. Paste targeting_calibration into config.yaml under "
              "apps.hinge only for a supervised production OBSERVE validation; this tool never "
              "writes config.yaml. It does NOT release AUTO: before AUTO, record separate evidence "
              "from production Worker/hub showing pre-tap publication, post-tap item verification, "
              "real provider/quota behaviour, persisted labels, and refusal/paywall handling.")
    print("=" * 78)
    print(yaml_document.rstrip())
    print("=" * 78)


# =====================================================================================
# CLI
# =====================================================================================

def _run_holding_the_device(args: argparse.Namespace, command) -> None:
    """Run a phone-touching subcommand under the same lock a production run holds.

    `capture` and `observe-check` drive the real Pixel; `measure`, `verify-entry-anchor`, and
    `attach-operational-evidence` never open ADB and deliberately stay unlocked. See
    tools/_devicelock.py for why this exists and why the lock spans the whole command.
    """
    run_holding_the_device(args.config, command, args)


def main(argv: list[str] | None = None) -> None:
    # Match ``python -m operation_love``: credentials used by the optional supervised
    # observe-check live in this project's own .env on the owner's machine.  Never walk parent
    # directories. The shared loader also refuses link leaves and tightens a real .env before
    # reading it, so a quota probe cannot silently inherit credentials from an unsafe file.
    load_private_dotenv(Path.cwd() / ".env")

    ap = argparse.ArgumentParser(
        prog="python -m tools.hinge_calibrate",
        description="Measure identity_match_max_dist and inline_item_max_dist, the two "
                     "operator-calibrated Hinge targeting bounds ops/RUNBOOK.md's held-out "
                     "device measurement protocol requires before any targeted opener or "
                     "targeted AUTO like can run.")
    sub = ap.add_subparsers(dest="command", required=True)

    cap = sub.add_parser(
        "capture",
        help="Capture a calibration or held-out session on the real device. TOUCHES THE PHONE "
             "(read-only screencaps + small humanized scroll gestures); every heart-tap and "
             "profile advance is done by the owner, by hand, when prompted.")
    cap.add_argument("--split", choices=list(_SPLITS), required=True,
                     help="which split this session belongs to. The held-out set must be "
                          "SEPARATELY COLLECTED, never carved out of a calibration pool later")
    cap.add_argument("--profiles", type=int, default=3,
                     help="how many distinct real profiles to capture in this session "
                          "(default 3; need >= 2 for cross-profile identity pairs)")
    cap.add_argument("--config", default="config.yaml")
    cap.add_argument("--out", default=None,
                     help="output directory (default ops/calibration/targeting_<UTC "
                          "timestamp>/)")
    cap.add_argument(
        "--record-operational-checks", action="store_true",
        help="after capture, record evidence references for all four supervised operational "
             "checks required by ops/RUNBOOK.md. `measure` refuses to emit calibration YAML "
             "until the supplied sessions collectively contain all four confirmations")
    cap.add_argument("--unattended", action="store_true",
                    help="explicitly authorize automated HingeDriver hearts on photo model "
                          "item 1 for odd profile ordinals and item 3 for even ordinals, plus "
                          "a Pass-without-send. Evidence is marked circular "
                          "and is not supervised operational evidence.")
    cap.add_argument("--hybrid-review", action="store_true",
                     help="automated transport with fail-closed AI review checkpoints before every "
                          "heart/Pass and after every heart result; mutually exclusive with "
                          "--unattended")
    cap.add_argument(
        "--target-items", choices=sorted(_TARGET_ITEMS_FLAG_STRATEGY_IDS),
        default="alternate-1-3",
        help="which automated/hybrid photo-item targeting strategy to run. alternate-1-3 "
             "(default) targets photo model item 1 on odd profile ordinals and item 3 on even "
             "ordinals -- unchanged default behaviour. photo-1-only always targets item 1 and "
             "DOES NOT EXERCISE DEEP-ITEM NAVIGATION: real decks rarely carry three numberable "
             "photos, so this is an explicit, narrower calibration that proves less about deep "
             "navigation -- pick it deliberately, not as a default.")
    cap.add_argument("--confirmation", default="",
                     help="required exact phrase: "
                          f"--unattended={_UNATTENDED_CONFIRMATION}; "
                          f"--hybrid-review={_HYBRID_REVIEW_CONFIRMATION}")
    cap.add_argument("--reviewer-model", default=_HYBRID_REVIEW_SOURCE,
                     help="manifest-only AI reviewer model/id for --hybrid-review")
    cap.add_argument("--reviewer-id", default="",
                     help="manifest-only reviewer name/id for --hybrid-review (defaults to model)")
    cap.add_argument("--reviewer-version", default="",
                     help="manifest-only reviewer model/version for --hybrid-review (defaults to process)")
    cap.add_argument("--reviewer-process", default="stdin_checkpoint_protocol",
                     help="manifest-only AI reviewer process/version for --hybrid-review")
    cap.add_argument("--send-like", action="store_true",
                     help="owner-directed exception to this tool's default never-send design: "
                          "replace every calibration Pass with a REAL, PERMANENT Send Priority "
                          "Like instead. Requires --hybrid-review (every send must clear a "
                          "reviewer checkpoint) plus --send-like-confirmation")
    cap.add_argument("--send-like-confirmation", default="",
                     help=f"required exact phrase for --send-like: {_SEND_LIKE_CONFIRMATION}")

    mea = sub.add_parser(
        "measure",
        help="Offline-only: freeze both bounds from one or more previously captured session "
             "directories, spanning both splits. NEVER opens an ADB session or touches the "
             "device.")
    mea.add_argument("sessions", nargs="+",
                     help="capture session director(ies) (ops/calibration/targeting_*/)")
    mea.add_argument("--config", default="config.yaml")
    mea.add_argument("--accept-automated-circular-evidence", action="store_true",
                     help="allow only exact --unattended manifests to produce a clearly marked "
                          "numeric candidate. It does not convert them to supervised evidence "
                          "or release AUTO.")
    mea.add_argument("--accept-hybrid-reviewed-evidence", action="store_true",
                     help="allow only exact --hybrid-review manifests with accepted external "
                          "review checkpoints to produce a clearly marked numeric candidate; "
                          "it never converts AI review into human ground truth or releases AUTO")
    mea.add_argument("--confirmation", default="",
                     help="required exact phrase: "
                          f"--accept-automated-circular-evidence={_UNATTENDED_CONFIRMATION}; "
                          f"--accept-hybrid-reviewed-evidence={_HYBRID_REVIEW_CONFIRMATION}")
    mea.add_argument("--unattended-review", default="",
                     help="required with either automated-evidence flag: deterministic self-hashed "
                          "artifact from tools.hinge_calibration_review bound to these exact "
                          "captures and config bytes")

    anchor = sub.add_parser(
        "verify-entry-anchor",
        help="Offline-only: attach the required hash-bound entry-anchor replay to one completed "
             "private v3 manifest. NEVER opens ADB or touches the device.")
    anchor.add_argument("session", help="completed capture directory under ops/calibration/")
    anchor.add_argument("--config", default="config.yaml",
                        help="used only to load the existing like detector; no ADB session opens")

    attach = sub.add_parser(
        "attach-operational-evidence",
        help="Offline-only one-time migration for a completed, unledgered v3 capture paused at "
             "legacy evidence prompts. It validates recorder-v2 evidence, creates the normal "
             "entry-anchor replay, and replaces only the three trace references.")
    attach.add_argument("session", help="completed unledgered capture directory under ops/calibration/")
    attach.add_argument("operational_evidence",
                        help="completed recorder-v2 directory or its manifest.json")
    attach.add_argument("--config", default="config.yaml",
                        help="used only for offline replay/template binding; no ADB session opens")

    observe = sub.add_parser(
        "observe-check",
        help="Pre-calibration passive-cycle + synthetic-quota evidence only. It never exercises "
             "production Worker/hub/store and cannot authorize AUTO.")
    observe.add_argument("--config", default="config.yaml")
    observe.add_argument("--out", default=None,
                         help="output directory (default ops/calibration/observe_check_<UTC>/)")
    observe.add_argument("--supervised", action="store_true",
                         help="required acknowledgement that a human, not this tool, performs every action")
    observe.add_argument("--confirmation", default="",
                         help=f"required exact acknowledgement: {_OBSERVE_CHECK_CONFIRMATION}")

    args = ap.parse_args(argv)
    _check_vision_stacks()
    if args.command == "capture":
        _run_holding_the_device(args, _cmd_capture)
    elif args.command == "measure":
        _cmd_measure(args)
    elif args.command == "verify-entry-anchor":
        _cmd_verify_entry_anchor(args)
    elif args.command == "attach-operational-evidence":
        _cmd_attach_operational_evidence(args)
    else:
        _run_holding_the_device(args, _cmd_observe_check)


if __name__ == "__main__":
    main()
