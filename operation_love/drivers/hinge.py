"""The generic Android driver (+ its Hinge binding) — physical Android phone over HOST-SIDE
ADB (no on-device helper).

This module is now home to TWO things:

  AndroidDriver  — the app-agnostic driver: perception (screencap + vision) and action
                   (humanized taps/swipes) for ANY Android dating app, parameterised by an
                   AndroidAppSpec (operation_love/drivers/android_spec.py). Nothing in here
                   is Hinge-specific anymore except the flow branch selected by
                   `spec.like_flow`.
  HingeDriver    — a thin subclass binding AndroidDriver to HINGE_SPEC.

Bumble's binding (BumbleAndroidDriver + BUMBLE_SPEC) lives in
operation_love/drivers/android/ instead of here, because it needs no access to anything
below — EXCEPT AndroidDriver itself, imported from this module.

Why is the generic driver defined in a file called "hinge.py" instead of its own module?
Because tests/test_hinge_observe.py, tests/test_hinge_vision.py and tools/hinge_inspect.py
monkeypatch/call module-level helpers (`_split_diff`, `_downsample`, `_match_glyph`,
`_load_template`, `Adb`, ...) as attributes of THIS module (`operation_love.drivers.hinge`).
Python resolves a bare name inside a function/method against the globals of the module it was
DEFINED in, not the module that imported it — so `monkeypatch.setattr(hinge, "_split_diff",
fake)` only takes effect on methods whose code actually lives in hinge.py's namespace. Moving
AndroidDriver's body out to a separate module would silently break every one of those patches
(the methods would keep calling the ORIGINAL helpers, the test's fake would just sit unused on
the wrong module) without any test provably failing to construct — a nasty, quiet drift. Rather
than touch 190+ existing test assertions to route patches through a new module, the generic
driver stays here and Bumble's binding imports AndroidDriver FROM here instead.

Off the old uiautomator2/emulator path: that installed an on-device server (atx-agent + the
uiautomator2 APK) which Play Integrity can flag and which ops/HINGE-PIXEL-RUNBOOK.md §5
forbids as the main account-protection guardrail. This driver instead talks to a genuine,
stock, physical Pixel through the host-side `Adb` transport only:

  * perception  — `adb exec-out screencap` frames, deduped by a downsampled
                  signature (no accessibility tree, no resource-ids)
  * action      — humanized `input motionevent` taps/swipes (curved, jittered,
                  log-normal timing) via Adb

Hinge itself lets a user like a photo or prompt with a comment. Operation Love intentionally
targets photos only, so the opener is sent
at like-time (Signals behavior #2). Reading the whole profile slowly before
deciding (the dwell in `_capture_current`) is Signals behavior #1. We only ever
send a NORMAL like — never a Rose (Hinge's super-like) or, generically, any other paid
upgrade a given app might interstitial-upsell; Roses/boosts/etc are manual, always
(owner rule — see `_handle_rose_upsell`).

⚠️ LIVE-VERIFY: the tap COORDINATES (fractions of the screen) and the
screencap-diff thresholds below are best-effort from a partial UI map and MUST be
confirmed on a COMPLETE, Selfie-Verified profile (the like→comment→"Send Like"
sheet and the empty-deck state are gated until the profile is finished). All of
them are config-overridable under apps.hinge in config.yaml. The transport
(adb.py) and the capture/scroll mechanics are not gated and are validated.

If the USB/ADB link drops mid-run, Adb raises DriverClosed (parity with Bumble's
browser-closed path) so the worker stops cleanly and flushes buffered labels.
"""
from __future__ import annotations

from contextlib import contextmanager
import base64
import dataclasses
import difflib
import fcntl
import functools
import hashlib
import json
import marshal
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ..typography import format_duration
from ..human import human_cooldown, human_delay
from ..human_motion import tap_jitter_margin_px
from ..perception.capture import Profile
from .adb import SCROLL_X_JITTER_PX, Adb, AdbError, clamp_xy, scroll_x
from .android_spec import AndroidAppSpec
from .base import (ActionCancelled, DatingAppDriver, DeckBlockedError, DriverClosed,
                   ItemTargetingError, OBSERVE_ITEM_INCONCLUSIVE, OBSERVE_ITEM_MATCH,
                   OBSERVE_ITEM_MISMATCH, ObserveItemCheck, open_debug_log,
                   snapshot_failure_frame)
from .frameshift import SHIFT_MEASURED, ShiftEstimationError, estimate_shift
from .item_crops import (
    PHOTO_ONLY_POLICY_ID, ItemCropError, build_item_payload, unnumber_unless_confident_photo)
from .item_type_preflight import INCONCLUSIVE, ItemTypePreflight, preflight_item_type
from .item_identity import IdentityError, compare_profile_identity
from .item_index import (
    ItemIndexError, _edge_only_two_strip_shift, _exact_multi_strip_shift,
    _layout_repaired_shift, _measured_layout_bridge,
    _matched_delta_clusters, _observed_gutters, _structural_landmarks,
    _project_to_exact_full_layout, _structural_tail_shift, VideoMuteMarker, build_item_index,
    _video_track_deltas,
)
from .item_nav import ItemNavigationError, navigate_to_item
from .item_verify import (VERIFY_MISMATCH, SheetVerificationError, verification_blocker,
                          verify_sheet_item)
from .like_composer import (
    ComposerDetectionError, ComposerSurface, locate_inline_composer)
from .scroll_step import (MAX_SEGMENTATION_FALLBACK_FRAMES, STEP_SEGMENTATION_FALLBACK,
                          ScrollStepError,
                          plan_scroll_step)
from .scroll_top import ScrollTopError, confirm_scroll_top
from .segment import SegmentationError, segment_frame
from .touchwatch import TouchWatcher, TouchWatchUnavailable
from .uhid import UhidTouch, UhidUnavailable

_ASSETS = Path(__file__).parent / "assets"

# The closest known different-profile pair is 2.565 grey levels apart.  This is deliberately
# local to the driver rather than a permissive default: targeted likes only run with an
# operator-supplied, evidence-backed value strictly below it.
_TARGETING_IDENTITY_FALSE_MATCH_DISTANCE = 2.565
_TARGETING_SHEET_FALSE_MATCH_DISTANCE = 14.91
_TARGETING_CALIBRATION_KEYS = frozenset({
    "schema_version", "hinge_version_name", "frame_size_px", "composer_layout_id",
    "item_selection_policy_id",
    "identity_match_max_dist", "inline_item_max_dist", "device", "calibrated_at",
    "identity_band", "content_band",
})


def _item_index_runtime_provenance() -> dict[str, str | None]:
    """Fingerprint the item-index implementation this process is actually executing.

    A git hash or source-file hash describes the working tree, not a long-lived Python process:
    either can change after imports are cached.  The old splitter-only digest therefore left a
    particularly bad blind spot: a process could be running a changed builder, shift repair, or
    page assembler while reporting the same runtime fingerprint.  Hash the loaded builder, every
    project function it TRANSITIVELY reaches, and the simple calibration values those loaded
    functions read.

    The closure is what makes this honest.  Naming the stages explicitly still stopped at
    `segment_frame`, `estimate_shift` and `capture_profile_identity`, which are thin
    orchestrators: the row classifier, the strip matcher and the scroll-top confirmer beneath
    them decide as much of a refusal as anything in `item_index`, and swapping one left the
    digest byte-identical -- the same blind spot one call-frame down.  Following function-valued
    globals rather than sweeping each module keeps the original property that an imported but
    UNUSED helper cannot make the fingerprint drift.  Functions outside this project (stdlib,
    third-party) are recorded by identity rather than hashed, so a rebinding is still visible
    without pinning the digest to somebody else's bytecode.

    This intentionally hashes *code objects*, not source files.  Editing a checkout after this
    process imported ``item_index`` must not make its diagnostics claim that the running code
    changed; monkeypatching a loaded helper, on the other hand, must.  The provenance remains
    best-effort and deliberately tolerates a test callable whose globals do not look like the
    production item-index module.
    """
    try:
        build = build_item_index
        namespace = getattr(build, "__globals__", {})
        splitter = namespace.get("_split_on_bounded_cards")
        source = namespace.get("__file__")
        if isinstance(source, str):
            try:
                source = str(Path(source).resolve())
            except Exception:  # noqa: BLE001 -- provenance is best-effort diagnostics
                pass

        # The SEEDS of the walk below, not the whole of it.  They stay explicit so that a stage
        # disappearing from the module is itself a digest change (see `missing`), and so a
        # monkeypatched helper is hashed even when it lives in a test module the closure would
        # otherwise decline to follow.
        callable_names = (
            "build_item_index",
            "_matched_delta_clusters", "_structural_landmarks", "_layout_repaired_shift",
            "_exact_multi_strip_shift", "_measured_layout_bridge",
            "_project_to_exact_full_layout",
            "_track_candidate_deltas", "_track_anchor_count", "_video_track_deltas",
            "_repair_video_track_shifts",
            "_repair_shifts_from_layout", "_frame_offsets", "_observations",
            "_overlap_groups", "_heart_clusters", "_resolve_group",
            "_scroll_top_evidence", "_split_on_bounded_cards", "_assemble", "_tail",
            "segment_frame", "estimate_shift", "capture_profile_identity",
        )

        def nested_names(code) -> set[str]:
            """Global names a code object or one of its nested comprehensions reads."""
            names = set(getattr(code, "co_names", ()))
            for constant in getattr(code, "co_consts", ()):
                if isinstance(constant, type(code)):
                    names.update(nested_names(constant))
            return names

        unsupported = object()

        def canonical_value(value, depth: int = 0):
            """A stable marshal-able form for values that can change indexer behaviour.

            The depth ceiling is a guard, not a calibration: a self-referential container would
            otherwise recurse until `RecursionError`, and the blanket ``except`` below would turn
            one awkward global into a wholly absent provenance dict on a live refusal.
            """
            if depth > 8:
                return unsupported
            if value is None or isinstance(value, (bool, int, float, str, bytes)):
                return (type(value).__name__, value)
            if isinstance(value, tuple):
                values = tuple(canonical_value(item, depth + 1) for item in value)
                return ("tuple", values) if unsupported not in values else unsupported
            if isinstance(value, frozenset):
                values = [canonical_value(item, depth + 1) for item in value]
                if unsupported in values:
                    return unsupported
                return ("frozenset", tuple(sorted(values, key=marshal.dumps)))
            if isinstance(value, dict):
                values = [(canonical_value(key, depth + 1), canonical_value(item, depth + 1))
                          for key, item in value.items()]
                if any(unsupported in pair for pair in values):
                    return unsupported
                return ("dict", tuple(sorted(values, key=lambda pair: marshal.dumps(pair[0]))))
            return unsupported

        project = __name__.split(".", 1)[0]

        def qualified(obj, fallback: str) -> str:
            return (f"{getattr(obj, '__module__', '?')}."
                    f"{getattr(obj, '__qualname__', getattr(obj, '__name__', fallback))}")

        def is_ours(obj) -> bool:
            module = getattr(obj, "__module__", None)
            return isinstance(module, str) and (module == project
                                                or module.startswith(project + "."))

        functions: dict[str, object] = {}   # qualified name -> loaded function, hashed in full
        foreign: dict[str, str] = {}        # reading site -> identity of a function we do not own
        missing: list[str] = []             # seed stages this namespace no longer has at all
        pending: list[object] = []

        def consider(obj, *, seed: bool = False) -> bool:
            """Track a function whose code decides an index result.

            A seed is hashed wherever it lives, which is what makes a monkeypatched stage
            visible.  Anything discovered by the walk is followed only when it is ours.
            """
            if getattr(obj, "__code__", None) is None:
                return False
            if not seed and not is_ours(obj):
                return False
            key = qualified(obj, "?")
            if key not in functions:
                functions[key] = obj
                pending.append(obj)
            return True

        seen_globals: dict[str, object] = {}
        for name in callable_names:
            candidate = build if name == "build_item_index" else namespace.get(name)
            if not consider(candidate, seed=True):
                missing.append(name)

        while pending:
            current = pending.pop()
            current_globals = getattr(current, "__globals__", {})
            for global_name in nested_names(current.__code__):
                if global_name not in current_globals:
                    continue
                value = current_globals[global_name]
                if consider(value):
                    continue
                # Qualify by reading namespace: dependencies can live in another module with a
                # same-named calibration constant.
                key = f"{getattr(current, '__module__', '?')}.{global_name}"
                if getattr(value, "__code__", None) is not None:
                    foreign[key] = qualified(value, global_name)
                    continue
                encoded = canonical_value(value)
                if encoded is not unsupported:
                    seen_globals[key] = encoded

        # The algorithm label is part of the human/audit contract even though it is not itself a
        # branch in the builder.  It is normally already collected above, but make that guarantee
        # explicit for monkeypatched builder functions with sparse globals.
        algorithm_id = namespace.get("ITEM_INDEX_ALGORITHM_ID")
        encoded_algorithm = canonical_value(algorithm_id)
        if encoded_algorithm is not unsupported:
            seen_globals["item_index.ITEM_INDEX_ALGORITHM_ID"] = encoded_algorithm

        # Every section is sorted and tagged, so the digest depends only on WHAT was reached, not
        # on the order the walk happened to reach it in -- across processes and hash seeds alike.
        digest = hashlib.sha256(b"operation-love.item-indexer-runtime-v2\0")
        for name in sorted(missing):
            digest.update(b"missing\0" + name.encode("utf-8", "replace") + b"\0")
        for key in sorted(functions):
            function = functions[key]
            digest.update(b"code\0" + key.encode("utf-8", "replace") + b"\0"
                          + marshal.dumps(function.__code__))
            defaults = (getattr(function, "__defaults__", None),
                        getattr(function, "__kwdefaults__", None))
            encoded_defaults = canonical_value(defaults)
            if encoded_defaults is not unsupported:
                digest.update(b"defaults\0" + marshal.dumps(encoded_defaults))
            else:
                digest.update(b"defaults-unserializable\0")
        for key in sorted(foreign):
            digest.update(b"foreign\0" + key.encode("utf-8", "replace") + b"\0"
                          + foreign[key].encode("utf-8", "replace") + b"\0")
        for key in sorted(seen_globals):
            digest.update(b"global\0" + key.encode("utf-8", "replace") + b"\0"
                          + marshal.dumps(seen_globals[key]))

        return {
            "algorithm_id": algorithm_id,
            "build_callable": (
                f"{getattr(build, '__module__', '?')}."
                f"{getattr(build, '__qualname__', getattr(build, '__name__', '?'))}"),
            "module_path": source if isinstance(source, str) else None,
            "indexer_code_sha256": digest.hexdigest(),
            "splitter_code_sha256": (
                hashlib.sha256(marshal.dumps(getattr(splitter, "__code__", None))).hexdigest()
                if getattr(splitter, "__code__", None) is not None else None),
        }
    except Exception:  # noqa: BLE001 -- diagnostics must never alter a live refusal
        return {
            "algorithm_id": None, "build_callable": None,
            "module_path": None, "indexer_code_sha256": None,
            "splitter_code_sha256": None,
        }

# A second controller must never turn an active OBSERVE wait into its own read-and-unwind
# sequence.  The in-process map makes that refusal deterministic even when both drivers are
# constructed in one Python process; the advisory OS lock carries the same contract across the
# separate controller process used by hybrid review.  The lock file is intentionally in the
# system temp directory rather than a run/debug folder: a restarted Worker and its controller
# need one stable, per-device rendezvous point, and an old empty file is harmless once its file
# descriptor closes.
_OBSERVE_INPUT_LEASES: dict[str, tuple[int, int]] = {}
_OBSERVE_INPUT_LEASES_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True)
class TargetingCalibration:
    """Measured bounds and exact crop geometry licensing a model item on one device."""

    schema_version: int
    hinge_version_name: str
    frame_size_px: tuple[int, int]
    composer_layout_id: str
    item_selection_policy_id: str
    identity_match_max_dist: float
    inline_item_max_dist: float
    device: str
    calibrated_at: str
    identity_band: tuple[float, float, float, float]
    content_band: tuple[float, float]


def _targeting_band(value, *, length: int) -> tuple[float, ...] | None:
    """Normalise a stored/effective geometry band, or return None when it is unsafe to use."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in value):
        return None
    band = tuple(float(v) for v in value)
    if length == 4:
        x0, y0, x1, y1 = band
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            return None
    else:
        y0, y1 = band
        if not (0.0 <= y0 < y1 <= 1.0):
            return None
    return band


def _parse_targeting_calibration(raw, serial: str | None, *, identity_band, content_band
                                 ) -> tuple[TargetingCalibration | None, str | None]:
    """Return a complete targeting calibration or the reason it cannot license a like.

    Config validation gives operators a precise error.  This second check is still necessary:
    driver tests and embedding callers may construct a Config-like object directly, bypassing
    ``config.validate()``, and that must fail closed at the gesture boundary too.  The bound is
    only valid for the exact ADB device serial that produced it, so the runtime check repeats
    that binding rather than trusting a free-form evidence string.
    """
    if raw is None:
        return None, "no apps.<app>.targeting_calibration mapping is configured"
    if not isinstance(raw, dict):
        return None, "targeting_calibration is not a mapping"
    unknown = set(raw) - _TARGETING_CALIBRATION_KEYS
    missing = _TARGETING_CALIBRATION_KEYS - set(raw)
    if unknown or missing:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if unknown:
            parts.append(f"unknown {sorted(unknown)}")
        return None, f"targeting_calibration has {'; '.join(parts)}"
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 3:
        return None, "targeting_calibration.schema_version must be the exact integer 3"
    if raw["composer_layout_id"] != "hinge_inline_v1":
        return None, (
            "targeting_calibration.composer_layout_id must be the supported "
            "'hinge_inline_v1' layout")
    if raw["item_selection_policy_id"] != PHOTO_ONLY_POLICY_ID:
        return None, (
            "targeting_calibration.item_selection_policy_id must be the supported "
            f"{PHOTO_ONLY_POLICY_ID!r} policy")
    frame_size = raw["frame_size_px"]
    if (not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2
            or any(type(v) is not int or v <= 0 for v in frame_size)):
        return None, "targeting_calibration.frame_size_px must be two positive integers"
    for key in ("identity_match_max_dist", "inline_item_max_dist"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"targeting_calibration.{key} is not a number"
        if not math.isfinite(value) or value <= 0:
            return None, f"targeting_calibration.{key} is not finite and > 0"
    if raw["identity_match_max_dist"] >= _TARGETING_IDENTITY_FALSE_MATCH_DISTANCE:
        return None, (
            "targeting_calibration.identity_match_max_dist is not strictly below "
            f"the known {_TARGETING_IDENTITY_FALSE_MATCH_DISTANCE} false-match distance")
    if raw["inline_item_max_dist"] >= _TARGETING_SHEET_FALSE_MATCH_DISTANCE:
        return None, (
            "targeting_calibration.inline_item_max_dist is not strictly below "
            f"the known {_TARGETING_SHEET_FALSE_MATCH_DISTANCE} foreign-card false-match "
            "distance")
    for key in ("device", "calibrated_at", "hinge_version_name"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            return None, f"targeting_calibration.{key} is not nonempty evidence text"
    if not isinstance(serial, str) or not serial:
        return None, "targeting_calibration requires a nonempty configured ADB serial"
    if raw["device"] != serial:
        return None, (
            "targeting_calibration.device does not exactly match the configured ADB serial "
            f"({raw['device']!r} != {serial!r})")
    calibrated_identity = _targeting_band(raw["identity_band"], length=4)
    calibrated_content = _targeting_band(raw["content_band"], length=2)
    effective_identity = _targeting_band(identity_band, length=4)
    effective_content = _targeting_band(content_band, length=2)
    if calibrated_identity is None or calibrated_content is None:
        return None, "targeting_calibration geometry is not finite, normalised, and ordered"
    if effective_identity is None or effective_content is None:
        return None, "the effective targeting geometry is not finite, normalised, and ordered"
    if calibrated_identity != effective_identity:
        return None, (
            "targeting_calibration.identity_band does not exactly match the effective "
            f"identity_band ({calibrated_identity!r} != {effective_identity!r})")
    if calibrated_content != effective_content:
        return None, (
            "targeting_calibration.content_band does not exactly match the effective "
            f"content_band ({calibrated_content!r} != {effective_content!r})")
    return TargetingCalibration(
        schema_version=3,
        hinge_version_name=raw["hinge_version_name"].strip(),
        frame_size_px=(frame_size[0], frame_size[1]),
        composer_layout_id="hinge_inline_v1",
        item_selection_policy_id=PHOTO_ONLY_POLICY_ID,
        identity_match_max_dist=float(raw["identity_match_max_dist"]),
        inline_item_max_dist=float(raw["inline_item_max_dist"]),
        device=raw["device"].strip(), calibrated_at=raw["calibrated_at"].strip(),
        identity_band=calibrated_identity, content_band=calibrated_content,
    ), None


class HingeActionError(RuntimeError):
    """An autonomous action did not produce the expected on-screen change (stuck deck, missed
    tap, or an unknown screen). NOT a DriverClosed (which is a clean, restart-safe stop): this
    is unexpected, so the worker halts the run and preserves the debug logs.

    Named for Hinge (the first, and so far only calibrated, Android app) but raised by
    AndroidDriver generically — any Android app's driver instance can raise it."""


class HingeDeckBlockedError(HingeActionError, DeckBlockedError):
    """A Hinge action reached a known blocking screen instead of completing.

    Both bases are intentional: callers that already treat Hinge action failures
    as ``HingeActionError`` remain conservative, while ``Worker`` can catch the
    app-agnostic ``DeckBlockedError`` and report the existing graceful blocked
    state rather than record a like Hinge refused.
    """


class HingeTargetingError(HingeActionError, ItemTargetingError):
    """This driver could not put the like on the item the opener was written about, so it put it
    nowhere. Doc 5.6's hard stop; see base.ItemTargetingError for the rule and the fields.

    BOTH bases are load-bearing and neither is decorative. `HingeActionError` keeps the existing
    halt/preserve-the-debug-logs behaviour for any caller that has always caught this driver's
    action failures by type (and `_verify_like_landed`, `UnlocatedControlError` and the rest stay
    exactly what they were). `ItemTargetingError` is what worker.py catches, because the worker is
    app-agnostic and must not import a Hinge symbol to recognise the one failure class that has a
    specific, non-error stop to render (see _auto_loop's targeting stop).

    Raised at three places, all leaving the screen untouched-from-here-on:
      * `_verifiable_payload`, before any gesture at all -- an item whose crops cannot serve as a
        verification reference must not be tapped, and finding that out afterwards costs a stop
        with a sheet open on somebody's card;
      * `_locate_target_heart`, after the retries are spent -- scrolled, but nothing tapped, no
        sheet, no like;
      * `_verify_sheet_shows`, after the tap -- sheet open, nothing typed, Send never tapped.
    """


class ForbiddenTapError(HingeActionError):
    """A tap resolved to a point inside one of the spec's forbidden_zones and was refused.

    These zones guard PAID controls (Bumble's SuperSwipe sits between Pass and Like;
    Hinge's Rose). Tapping one spends the owner's money and breaks a standing rule that
    super-likes and boosts are manual-only. Refusing is always correct here: a missed
    like costs one profile, a mis-tap costs money and cannot be undone.

    Deliberately a HingeActionError, so it halts the run and preserves the debug logs like
    any other unexpected-screen condition rather than being swallowed as routine."""


class OutOfRangeTapError(HingeActionError):
    """A touch-down coordinate resolved to a fraction of the screen outside 0..1 -- always a
    configuration or logic error (a bad coords/*_frac value in config.yaml or an
    AndroidAppSpec, or a computed offset that pushed a touch off-screen), never a
    legitimate touch.

    This exists because BOTH real touch transports CLAMP whatever coordinate they are
    handed onto the live screen instead of raising (uhid.py's _report, Adb._clamp -- see
    clamp_xy in adb.py, which both now share). An out-of-range value therefore doesn't fail
    on a real device: it silently lands on a screen EDGE -- and that edge can be INSIDE a
    forbidden zone despite the raw value never having fallen inside one, because
    forbidden_zones' own 0..1 range check has nothing to compare an out-of-range fraction
    against. Demonstrated live: Bumble's SuperSwipe zone (0.34, 0.80, 0.66, 1.00) -- an aim
    fraction of (0.50, 1.05) clamps to pixel (538.9, 2399.0) = fraction (0.4989, 0.9996),
    inside the zone. Also reachable with no coords tuple at all: apps.bumble.read_scroll_frac
    set to 1.30 puts an ordinary read-scroll's touch-down at fraction 1.15, which clamps the
    same way.

    Deliberately a HingeActionError (not a bare RuntimeError): this must halt the run and
    preserve the debug logs like any other unexpected-safety condition, never be treated as
    a routine, retryable miss."""


class UnlocatedControlError(HingeActionError):
    """Vision could not locate a control, so the action was refused rather than guessed.

    This used to fall back to the calibrated fixed coordinate and tap it blind, on the
    reasoning that acting degraded beat not acting. That reasoning is wrong here. The
    coordinate is only valid for the app version and screen state it was measured on: Hinge
    shifts its floating buttons with the "Start sending likes" banner and per-profile photo
    heights, and the app updates on its own schedule. A blind tap at a stale point does not
    "probably still work" — it lands on whatever now occupies that pixel, which on these
    screens can be a paid control (a Rose, a SuperSwipe) or an irreversible one.

    Worse, the trigger is usually not a transient miss but a MISSING CAPABILITY: if OpenCV
    is not installed, every template match returns nothing and EVERY action silently becomes
    a blind coordinate tap, indefinitely, with no error. That exact bug shipped once already
    (a launcher omitted the `hinge` extra, so opencv was never installed).

    Failing here costs one profile. Guessing costs money or the account."""


class UnconfirmedScreenError(HingeActionError):
    """A decide gesture (like/pass) was refused because the expected swipe-deck screen could
    not be positively confirmed before acting.

    Why this exists, distinct from ForbiddenTapError: forbidden_zones is a SCREEN-AGNOSTIC
    rectangle — it forbids a coordinate no matter what is actually on screen. But the danger
    some apps pose is SCREEN-DEPENDENT, and a static rectangle cannot express that. Measured
    live on the device 2026-08-10: Bumble's like_heart (0.850, 0.900) and pass_x (0.150,
    0.900) fallback coordinates are ordinary, harmless points on the swipe deck — and BOTH
    sit ON the "Get 30 SuperSwipes for $39.99" purchase button when the SuperSwipe purchase
    sheet is up instead (that sheet's CTA spans x 0.049-0.950, y 0.899-0.951 — see
    BUMBLE_SPEC's comment block for the full measurement). Both coordinates are also OUTSIDE
    BUMBLE_SPEC.forbidden_zones (0.34, 0.80, 0.66, 1.00), which only ever covers the middle
    third: no rectangle drawn to guard the deck's paid button can also cover the deck's own
    like/pass controls, because on the purchase sheet those are the SAME pixels. Widening
    forbidden_zones to also catch (0.850, 0.900) would make an ordinary like impossible on
    the very deck it exists to protect — see this module's HINGE_SPEC/BUMBLE_SPEC comments
    and ops/RUNBOOK.md for why that path was rejected.

    The guard that actually generalises asks a different question before every decide
    gesture: not "is this point forbidden" but "am I actually looking at the deck". A spec
    that declares BOTH a 'like' and a 'pass' glyph template can answer that — the same
    perception _observe_deck_ready already uses PASSIVELY for human-driven observe mode is
    reused here as an autonomous PRE-CONDITION on every decide gesture (see
    AndroidDriver._require_deck_confirmed). If either glyph is not visible — because a
    purchase sheet, an ad, a permissions dialog, or literally anything else has come between
    the bot and the deck — this refuses instead of firing a gesture blind, the same "fail
    loud, never guess" rule UnlocatedControlError already applies to a single missing button.

    A spec declaring no 'like'/'pass' templates at all (Bumble's current, real, uncalibrated
    state) has no way to answer the question, and the check is skipped entirely — the same
    "no template -> no vision path, no fixed-coord fallback" precedent _locate_button already
    sets. That is not a loophole: such a spec cannot be marked calibrated (see
    AndroidAppSpec.__post_init__'s has_paid_upsell check) and a TAP-gesture app in that state
    already can't decide at all (_await_button refuses with UnlocatedControlError for the
    same missing-template reason). The gap this class closes is for a CALIBRATED app whose
    deck glyphs are momentarily hidden, not for a spec that was never wired up to prove
    anything in the first place.

    Ground truth from the owner (measured 2026-08-10): this guard is the ONLY protection that
    covers Bumble's non-zero-SuperSwipe-balance case. When the account holds a balance (5, at
    measurement time), a stray SuperSwipe spends SILENTLY — no sheet, nothing for
    _handle_rose_upsell to dismiss, nothing at all downstream to catch it. Refusing to act
    unless the deck is positively confirmed is what stands between a drifted/blocked screen
    and a real, irreversible, paid action in that state — the confirmation sheet itself is
    not a safety net there; it does not even exist."""


class PaidUpsellStuckError(HingeActionError):
    """A paid-upgrade sheet was positively detected (its 'upsell_dismiss' template matched)
    and remained on screen after AndroidDriver._dismiss_via_zone's bounded retry budget.

    Raised instead of retrying forever, deliberately: a dismiss tap that keeps landing just
    outside the sheet (drift, a device rotation, an app update that moved the layout) must
    not turn into an indefinite sequence of blind taps near a screen whose bottom third can
    be a purchase button (MEASURED 2026-08-10 on Bumble's real SuperSwipe sheet: the CTA
    sits at y 0.899-0.951 — see BUMBLE_SPEC). Halting and preserving the on-screen state for
    debugging is always safer than one more guess at the same modal."""


_OBSERVE_POLL_S = 0.35     # internal sampling cadence for your manual tap (not app-facing)
# The _note_observe_waiting reasons that describe a VERIFIED compose/send state rather than an
# unclassifiable screen. They repeat on the slower _OBSERVE_LIKE_NOTICE_S cadence and get their
# own wording -- see that method. ``like_candidate`` is deliberately NOT here: no composer was
# observed in that state, so it is neither a human-paced draft nor a confirmed app send.
_OBSERVE_LIKE_WAIT_REASONS = frozenset({"like_sheet", "like_sending"})
# Minimum gap between ANY two waiting notices, including ones with different reasons -- the
# backstop against a flapping screen turning the heartbeat into a firehose. See
# _note_observe_waiting.
_OBSERVE_NOTICE_FLOOR_S = 2.0
_UPSELL_DISMISS_MAX_ATTEMPTS = 3
# Hinge animates the sticky profile header and floating controls while a read-scroll is still
# resolving.  A capture taken immediately after the gesture can therefore lock an in-between
# header as this profile's identity, then call the fully-rendered header a different person on
# the next frame.  `_await_button` independently retries through 0.4s settles for the same UI
# behavior.  This is an input-free, interruptible settle before the NEXT read frame; it is not
# dwell/read time and must never be credited as such.
_READ_SCROLL_SETTLE_S = 0.4
# Bounded retries for AndroidDriver._dismiss_via_zone before it gives up (PaidUpsellStuckError)
# rather than tapping an already-detected modal again and again. See that error's docstring.

# --- a BLOCKED deck (a purchase screen standing between us and the cards) ----------------
# Added after an observe-mode incident: an attempted send was refused because the account had no
# free likes remaining, and the app displayed its Hinge+ upgrade screen. Nothing in this codebase
# recognised that screen, so _await_like_resolved kept polling like_sheet/like_sending until the
# operator stopped it. No label was written, correctly, because the like never went out; the run
# should nevertheless have stopped rather than hanging.
_PAYWALL_MATCH_THRESHOLD = 0.75
# Template-match threshold for the "paywall" role (Hinge: the "HingeX" tab wordmark -- see
# HINGE_SPEC's templates comment for why that particular 210x76 crop and not the headline or the
# CTA). MEASURED with cv2.TM_CCOEFF_NORMED against the committed reference screenshot
# ops/calibration/hinge_out_of_likes_20260811.png (Pixel 7a, 1080x2400, 2026-08-11):
#   * live paywall frame                                            1.000
#   * same frame under gain 0.35..1.4 / bias -30..+90 perturbation   0.965 .. 1.000
#     (standing in for the tab rendering purple-when-active vs grey-when-inactive)
#   * max over ALL 88 real non-paywall frames of the hung run       0.4903
# 0.75 therefore sits with a wide margin on BOTH sides -- 0.215 below the worst positive and
# 0.26 above the best negative -- rather than being tuned to just clear one of them.
_PAYWALL_MAX_Y_FRAC = 0.30
# ...and the match must also be in the right PLACE. The tab chrome sits at y 266..318 of 2400
# (y_frac 0.111..0.133) in the reference dump, so a hit whose centre is below 0.30 of the screen
# height is not the tab bar and is rejected. Same defensive idiom as
# _observe_like_sheet_visible's existing 0.25..0.85 y-gate: cv2's normalized matcher can report a
# mathematically-perfect hit on a flat/degraded frame, and a position gate costs nothing.

_OBSERVE_STUCK_S = 90.0
# The FLOOR/anchor of the stuck-screen watchdog's budget, NOT the budget itself -- every arm
# point draws its own value from _observe_stuck_budget() below, and this constant is only the
# lower bound that draw can never fall under. This is the fix for the DEEPER defect the paywall
# merely exposed: worker.py calls wait_for_decision(timeout=None), so before this existed ANY
# unrecognised screen -- a paywall, a system dialog, an app update prompt, a crash to the
# launcher -- hung the run forever with no bail-out of any kind. The paywall is only today's
# instance of it.
#
# 90s is chosen to be far longer than anything the APP legitimately takes (Hinge's like-send
# resolves in seconds) while never limiting the HUMAN: every state where a person is genuinely
# thinking or typing resets this budget instead of consuming it -- see the reset points in
# wait_for_decision and the deliberately asymmetric like_sheet/like_sending handling in
# _await_like_resolved. In the incident run the owner spent 01:22:33 -> 01:27:14, nearly five
# minutes, deciding on ONE profile: that is normal, legitimate use and must never be interrupted.
#
# THIS IS RANDOMIZED (see _observe_stuck_budget() below) -- an earlier version of this comment
# argued it should NOT be, on the grounds that a host-side-only diagnostic with no
# device-observable behaviour has nothing for Hinge to fingerprint. The owner overruled that
# case-by-case reasoning: "humanize it because it's easier to just humanize all rather than
# selectively only humanizing what we think is detectable." The standing rule -- every
# timing/probability parameter is randomized/hazard-based, never a fixed constant, because a
# fixed constant is a bot signature -- now applies UNIFORMLY, not only to the knobs any single
# pass of reasoning judges risky. That judgement call is precisely the fragile part: a blanket
# rule cannot be wrong about which knob turned out to be observable, but an argument like the one
# this comment used to make can be, silently, and the codebase would have no way to notice. See
# _observe_stuck_budget() for the drawn value and its measured distribution.


def _observe_stuck_budget() -> float:
    """Draw one stuck-screen watchdog budget, anchored at the _OBSERVE_STUCK_S floor.

    human_cooldown is the correct primitive here specifically because it NEVER returns below its
    anchor: the floor guarantee is what stops the watchdog from ever cutting short a legitimate
    wait (see _OBSERVE_STUCK_S's own comment for why that floor matters), while the log-normal
    tail above it removes the fixed-90.0s signature a bare constant would otherwise carry.

    MEASURED over 200k draws at anchor 90.0: min 90.0s, median 104.4s, p75 115.9s, p95 138.5s,
    p99 158.5s, max ~249s, mean 108.2s, and exactly 0.0000 of draws below 90s.

    Deliberately a module-level function rather than an inline `human_cooldown(_OBSERVE_STUCK_S)`
    call at each arm site, so tests can monkeypatch `hinge._observe_stuck_budget` for
    deterministic arms instead of fighting real randomness.
    """
    return human_cooldown(_OBSERVE_STUCK_S)


_OBSERVE_STUCK_CHECK_S = 5.0
# The ANCHOR for how often the stuck-screen watchdog is allowed to spend a deck-ready probe on
# the `no_change` fast path -- like _OBSERVE_STUCK_S, humanized via human_delay(...) at each use
# rather than compared against directly (same uniform-humanization reasoning; see
# _OBSERVE_STUCK_S's comment). That path returns BEFORE any classification runs, so a STATIC
# unrecognised screen (exactly what the paywall is: nothing moves on it) is indistinguishable
# from a human sitting still and thinking -- both produce `no_change` forever. The only way to
# tell them apart is to ask whether a real deck is underneath, which costs two template matches.
# At the 0.35s poll cadence that would be ~6 matches a second for the entire wait; throttled to
# roughly once per 5s it is roughly two matches per 5s, i.e. ~3% of the polls, which is nothing.

_PAYWALL_OCR_WHITE_MIN = 200
# Luminance above which a pixel of the paywall HEADLINE band counts as text, for the
# binarize-and-invert preprocessing _paywall_headline asks _ocr_band for. MEASURED 2026-08-11 on
# ops/calibration/hinge_out_of_likes_20260811.png: 180, 200 and 215 all produce the identical,
# completely clean read ("You're out of free likes for today"), so 200 is the middle of a
# measured-flat range rather than a tuned edge. See _ocr_band's `white_text_threshold` parameter
# for why the ordinary recipe cannot read this band at all.

# Hinge's video mute control is app-owned UI: a white muted-speaker glyph on a fixed black circle
# at the upper-left of the media card.  Unlike video pixels and the playback timestamp, it is
# byte-stable.  The embedded 42x42 template is the inner square of that control, wholly inside
# the black disk so none of the person's underlying video pixels are retained.  Across the four
# saved Shannon sightings it matches at 1.0000 even while every surrounding video frame changes.
# Search remains card-local and top-local, so the Android mute status icon and the per-card like
# heart are outside the ROI.  The policy threshold is intentionally near-perfect: this is an
# exclusion/targeting decision, not a fuzzy content classifier.
_VIDEO_MUTE_X_BAND = (0.0, 0.22)
_VIDEO_MUTE_Y_BAND = (0.0, 0.14)
_VIDEO_MUTE_MATCH_THRESHOLD = 0.98
_VIDEO_MUTE_TEMPLATE_SIDE_PX = 42
_VIDEO_MUTE_TEMPLATE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACoAAAAqCAAAAADgqJaHAAACiUlEQVQ4EY3BX2hNARzA8e9v"
    "uVsjaw8e+BmRFy/+JFnCOWsKZaeNrUQ6t22KJv/CuWttDzb3IMWbc5/MtHukGA8S8ciDBybK"
    "pnbuipqa1EKRP5tz98fY2cP5fESJS5S4RIlLlLhEiUuUuESJS5QZNlSvuPicWYjyn7L2bTC6"
    "7wlRovyj6MjhBKGHjUSJMs1qXcy4gQqiRJmyIr2ZSYFJlCgFyU2lhNbPIe/VaghMokS5Xsm0"
    "gVMLMxCYQCE/mFTyGRCt6OavH5cyvywPAhMSnaMNvxjXsbFuBERbmphyr30IqjIQmCQ6K3hw8"
    "DehtoP07/qCaDoJrx8RupsjVJWBwKSpBbhzbBROngBunEbUtaG7mdDclS8Ay4PAJHF9C3D7+N"
    "ihVmCgZgRR14ZsCkoONNx3AMuDnAFFfjng93UAgzWfQNS1obuZ2vPF+A5geRCYQPHNdUx4Xz0"
    "MiLo2ZFOkk+A7gOVBziA079Yq8oZqhgiJujZkU6ST4DuA5UHOIK/04WLge+U78kRdG7Ip0kn"
    "wHcDyIGeQV3dZCHW2kSfq2pBNkU6C7wCWBzmD0L4LwrjONkKirg3ZFOkk+A5geZAzgPoOoHd"
    "oJ3CtFRB1bXj9iK1rwHcAy4PAhPoOoG/3t6uVgO+AaEsTU54lv0JVBgKTvReBt7UjJLoM4Iq"
    "LqOXx18ezPWNVGQhMFtwrY7DmE1DklzNS149oQbfBtN7m5RkITFh0Z6x6mLzinmW7+kGUgv1"
    "GKaF1heS9XAuBCSz9+YEJ85e8AUSZou07mJQziBJlmnluKeMGKogS5R+Jw0cLCT3dQ5Qo/yk"
    "7sx1+Nz4mSpQZNlTP6XrDLESJS5S4RIlLlLhEiUuUuESJ6w+BFL9vsnkW/AAAAABJRU5ErkJ"
    "ggg==")

# Hinge's spec: exactly today's values (formerly the module-level `DEFAULTS` dict + the
# `apps.hinge` block in config.yaml). calibrated=True — coords/templates verified live on the
# Pixel 7a (1080x2400) 2026-06-27. Config-overridable (apps.hinge.* in config.yaml).
HINGE_SPEC = AndroidAppSpec(
    app="hinge",
    package="co.hinge.app",
    calibrated=True,
    coords={
        # Action points as FRACTIONS of the screen (x, y in 0..1). like_heart / pass_x are
        # VISION-located at runtime (template-matched glyph) — these are only the FALLBACK if
        # vision can't find the glyph. Inline composer controls are deliberately absent here:
        # HingeDriver locates them from every live frame and has no coordinate fallback.
        "like_heart": (0.868, 0.667),   # FALLBACK only — heart vision-located on the first photo
        "pass_x": (0.116, 0.848),       # FALLBACK only — X vision-located (floating, bottom-left)
    },
    templates={
        # "like" -> hinge_like_button.png, NOT hinge_heart.png. hinge_heart.png (still shipped,
        # see below) is the WRONG glyph for this role -- it is Hinge's OUTLINE heart, which is
        # what appears in the "Which do we have in common" list rows, a different, non-like
        # control. Cross-correlated against it, those outline-heart rows score 1.000 (a perfect
        # match) while the REAL per-card like button (white heart in a filled black circle,
        # bottom-right of every photo/prompt card) peaked around 0.544 -- under the 0.6 default
        # threshold, and that 0.544 peak wasn't even on a heart. Confirmed on real profiles
        # (ops/calibration/scroll_20260811T211209Z/, 115 frames, 1080x2400, gitignored): with
        # hinge_heart.png as "like", _locate_button("like") could never find the true control,
        # and — worse — would have matched the "common" widget's outline hearts perfectly, which
        # must never be tapped as a like. hinge_like_button.png (88x88 grayscale, cropped from
        # ops/calibration/scroll_20260811T211209Z/00050.png, a pixel-exact inscribed square of
        # the filled black circle so the crop is 100% button chrome -- no photo, no face, no
        # text) fixes this: 199 genuine card-heart matches across those 115 frames scored
        # 0.815..1.000 (mean 0.999), and it does NOT fire on the outline hearts at all (measured
        # correlation ~ -0.09 there, not a near-miss). See _LIKE_MATCH_THRESHOLD below for why
        # the module's 0.6 default is unsafe specifically for this template (a fixed, unrelated
        # bit of UI chrome scores 0.653 against it on every single frame) and why 0.75 was
        # chosen instead. hinge_heart.png is kept on disk, unused by any role, purely as a
        # reference for what it actually is -- do not repurpose it for "like".
        "like": "hinge_like_button.png",
        "pass": "hinge_pass_x.png",
        "confirm": "hinge_send_like.png",              # inline composer's "Send Like" glyph
        "upsell_dismiss": "hinge_send_like_anyway.png",  # "Send Like anyway" — NEVER the Rose button
        # The "HingeX" tab wordmark of Hinge's full-screen upgrade paywall — the screen Hinge
        # shows INSTEAD of the deck once the account is out of free likes for the day. Cropped
        # (210x76, grayscale) from x 705..915, y 254..330 of the committed reference screenshot
        # ops/calibration/hinge_out_of_likes_20260811.png, taken live on the Pixel 7a
        # (1080x2400) on 2026-08-11 together with a one-off read-only uiautomator dump
        # (ops/calibration/hinge_out_of_likes_20260811_uiautomator.xml) used purely to measure
        # the geometry below — exactly like the 2026-08-10 Bumble measurement recorded in
        # ops/ANTI-BOT-RESEARCH.md. The accessibility tree stays FORBIDDEN in production code;
        # nothing at runtime reads it.
        #
        # The TAB CHROME, not the headline and not the CTA, because it is the only part of that
        # screen that holds still: the hero image is rotating marketing artwork, the benefit
        # list ("Send unlimited likes*", "See everyone who likes you", ...) scrolls, and the
        # bottom CTA carries a price string ("Get 3 months for CA$99.99") that varies by
        # currency, promo and plan. The tab bar does not move. Detection-only: unlike every
        # other role here, nothing ever aims a tap at this match — the paywall is a PURCHASE
        # screen and is never dismissed automatically (see _deck_blocked_reason).
        "paywall": "hinge_upgrade_tab.png",
    },
    like_flow="comment_sheet",
    accepts_opener=True,          # Hinge sends the opener as a comment at like-time
    think_time_calibrated=True,   # human_motion._THINK was measured on this app
    change_threshold=9.0,         # mean abs grayscale delta (0..255) on a 24x24 downsample
    scroll_captures=8,            # max screencaps while reading one profile
    dwell_s=1.1,                  # per-card read dwell (humanized) — Signals behavior #1
    read_scroll_frac=0.55,        # how far each read-scroll advances the profile
    # Hinge's own sticky profile header: the person's NAME, centred, appears here the moment
    # the card is scrolled at all, and stays pixel-identical for the whole profile. Measured
    # live on the Pixel 7a (1080x2400) 2026-08-10 over three frames of one profile at three
    # different scroll offsets: mean abs delta 0.00 between them, 17.95 against the same
    # profile's scroll-top frame (which shows the profile-independent filter-chips row here
    # instead). change_threshold is 9.0, so the separation is unambiguous. x1 stops at 0.80 to
    # keep the "..." overflow button out of the band.
    identity_band=(0.10, 0.048, 0.80, 0.094),
    # Card-header NAME band, consulted when identity_band says "top" (to resolve the
    # profile-independent filter-chips row) or "new" (only to let a matching stored name veto
    # a transient pixel mismatch -- never to strengthen the mismatch). See
    # AndroidAppSpec.identity_top_name_band and _identity_of for the full asymmetric rule. This
    # is the fix for the
    # incident that motivated this field: a pass that advanced the deck from "Zorva" to
    # "Qelix" was recorded as a scroll WITHIN Zorva's profile, because nothing on screen at
    # scroll-top could name the new card and the decision fell through to a loose content
    # match. OCR-only, never a pixel signature -- MEASURED on the real Pixel 7a on 2026-08-10 by
    # running `tesseract --psm 6` over the actual failing run's frames. Reads (all correct):
    #   Zorva, scroll-top, with the purple "shows thoughtful signals" banner -> "Zorva %"
    #   Zorva, scroll-top, banner gone (content shifted up)   -> "Zorva @ | @ Signals Active today"
    #   Qelix, scroll-top, with banner (three separate frames) -> "Qelix &"
    # On SCROLLED frames the same band OCRs to garbage, which is safe: the check this feeds is
    # gated on the pixel verdict being genuinely "top".
    identity_top_name_band=(0.03, 0.130, 0.75, 0.250),
    # The headline of the out-of-free-likes paywall above: "You're out of free likes for today"
    # (plain ASCII apostrophe), a white TextView over the hero photo at bounds [116,505][964,692]
    # in the 2026-08-11 reference dump. The band is padded out to px (60,470)-(1030,720) =
    # (0.0556, 0.1958, 0.9537, 0.3000) of 1080x2400 so a slightly different two-line wrap still
    # falls inside it. BEST-EFFORT REFINEMENT ONLY: the screen is DETECTED by the "paywall"
    # template above, never by this OCR — with no tesseract on PATH the stop still happens, just
    # with the less specific message (see _deck_blocked_reason's two strings).
    paywall_headline_band=(0.0556, 0.1958, 0.9537, 0.3000),
    # Scrolling content only. Above 0.125 is the status bar + sticky header, below 0.875 is the
    # floating X/heart overlay and the dark bottom nav -- none of which translate when content
    # scrolls, which is exactly why a whole-frame shift search never matched (see the table in
    # the module comment on _vertical_shift_match).
    content_band=(0.125, 0.875),
    observe_ignore_zones=(
        (0.75, 0.030, 1.00, 0.115),   # rewind arrow + "..." overflow, top-right
        (0.00, 0.900, 1.00, 1.000),   # bottom nav bar (dark bar starts at y frac 0.906)
    ),
    # OFF, because it demonstrably cannot work on the target device. Gesture corroboration
    # (touchwatch.py) reads the phone's own touch stream to prove a human actually PRESSED
    # something. Measured on the Pixel 7a, Android 17 (SDK 37, 2026-06-05 patch), 2026-08-10:
    # the adb `shell` user IS in group 1004(input), `/dev/input/event3` is `crw-rw---- root
    # input`, and `getevent -p /dev/input/event3` reads the touchscreen's full capability set
    # -- so the node opens fine -- but with the owner deliberately tapping and scrolling for
    # 30s, ZERO lines were delivered (raw_line_count 0, not merely unparsed). Android
    # withholds the input event stream from unprivileged readers on this build; recording
    # input is a rooted-device capability. No code change here can recover it.
    #
    # With this extra layer unavailable, wait_for_decision fails closed: raw pixel identity is
    # only a candidate advance, and a PASS additionally requires the different card-header name
    # to be read cleanly on both settled frames. Ambiguous header reflows resync without a label.
    # What is still given up is the narrower set of cases only a real tap could disambiguate --
    # a rewind/nav tap that changes the card without being a decision, and the same-first-name
    # collision (see _is_current_profile_frame). Both are logged as accepted risk in
    # ops/ANTI-BOT-RESEARCH.md with a §5 re-check trigger.
    #
    # The machinery is kept and tested rather than deleted: it is correct, it costs nothing
    # while off, and it becomes available the moment this runs on a rooted device or a
    # platform that permits the read. Re-enable with apps.hinge.observe_touch_watch: true and
    # confirm with `python -m tools.touch_selftest` BEFORE trusting it -- that tool now
    # reports raw lines separately from parsed events precisely so "the platform sent nothing"
    # can never again be confused with "the parser understood nothing".
    observe_touch_watch=False,
)


def _downsample(frame: bytes, size: int = 24):
    """Frame -> small grayscale int16 array, or None if PIL/numpy unavailable.

    A 24x24 grayscale is enough to tell whole-card changes (advance / like sheet)
    from frame noise, without reading the accessibility tree.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
        im = Image.open(BytesIO(frame)).convert("L").resize((size, size))
        return np.asarray(im, dtype="int16")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> caller falls back to exact bytes
        return None


_IDENTITY_DS = (64, 16)   # identity-band downsample (w, h) -- see _band. Matches the crop's
# roughly 5.6:1 aspect ratio (identity_band spans 0.70 of the width, 0.046 of the height) at
# enough resolution for OCR (_ocr_band crops the SAME rect at full frame resolution, this
# downsample is only for the pixel-distance comparison _identity_of makes on every poll).


def _band(frame: bytes, rect: tuple[float, float, float, float],
          size: tuple[int, int] = _IDENTITY_DS):
    """Downsampled grayscale crop of the normalised rect `(x0, y0, x1, y1)` of `frame`, at
    _IDENTITY_DS resolution. None when the frame can't be decoded -- same contract as
    _downsample (every caller here already knows how to fall back to 'unknown' on None; a
    decode failure must never read as either a positive 'same' or 'new' identity signal).

    This is a SEPARATE crop from _downsample's whole-frame one, deliberately: the identity
    band is a thin strip near the top of the screen (see HINGE_SPEC's identity_band comment
    for the measured geometry), and cropping it out before downsampling is what makes the
    0.00-vs-17.95 separation measured on the real device (see the module docstring) actually
    survive to a 64x16 array -- downsampling the WHOLE frame first and then trying to compare
    a sub-region of that would throw away exactly the resolution this anchor depends on.

    `size` is the (w, h) downsample grid and defaults to _IDENTITY_DS, which is the only grid
    anything in THIS file uses. It exists so `scroll_top.py` can read the SAME band through the
    SAME decode at its own coarser grid instead of re-implementing crop-and-resize: two decode
    paths that disagree is a measured trap in this repo, not a hypothetical one (see
    `item_crops.signature_of`'s docstring, where cv2's IMREAD_GRAYSCALE and a
    decode-then-cvtColor landed 1.46 grey levels apart on average and 9.8 apart on a 32x32
    signature -- twice the distance separating the two most alike items on that profile).
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
        im = Image.open(BytesIO(frame)).convert("L")
        w, h = im.size
        x0, y0, x1, y1 = rect
        crop = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        return np.asarray(crop.resize(size), dtype="int16")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> caller falls back to 'unknown'
        return None


def _band_dist(a, b) -> float:
    """Mean abs diff (0..255) between two _band() arrays. Callers only ever call this once
    both are already confirmed non-None -- unlike _split_diff, this has no undecodable-bytes
    fallback, because there is no meaningful "distance" between a real band and "we couldn't
    decode anything"."""
    import numpy as np
    return float(np.mean(np.abs(a - b)))


# A screen-off / keyguard-locked capture. Both bounds must hold: near-BLACK and
# near-UNIFORM. Darkness alone would false-positive on legitimately dark content
# (dark mode, a night photo) — but real content has text and edges, so its
# peak-to-peak spread stays large even when its mean is low. A blank framebuffer
# has essentially no spread at all.
_BLANK_MEAN_MAX = 2.0      # 0..255 mean luminance
_BLANK_SPREAD_MAX = 2.0    # 0..255 peak-to-peak


def _is_blank_frame(frame: bytes) -> bool:
    """True when `frame` is a solid-black capture (device asleep or on the keyguard).

    Decodes independently rather than reusing _downsample, deliberately. This asks a
    question about RAW PIXELS ("is the panel dark?"), whereas _downsample serves the
    dedup/diff layer's notion of a frame SIGNATURE — a different concern that callers
    and tests legitimately substitute. Sharing the helper would let a stubbed
    signature masquerade as a dead screen (and would let a monkeypatch silently
    disable this guard).

    Returns False when PIL/numpy are unavailable or the bytes don't decode — if we
    cannot see, we must not claim "blank" and halt a healthy run.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
        arr = np.asarray(
            Image.open(BytesIO(frame)).convert("L").resize((24, 24)), dtype="int16")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> "can't tell", not "blank"
        return False
    return bool(np.mean(arr) <= _BLANK_MEAN_MAX and np.ptp(arr) <= _BLANK_SPREAD_MAX)


def _split_diff(a: bytes, b: bytes) -> tuple[float, float]:
    """(top-half delta, bottom-half delta) between two frames.

    Lets us distinguish a like sheet sliding up over the BOTTOM (photo still
    visible up top) from a whole-card advance (top changes too). Falls back to a
    binary same/different signal when PIL/numpy aren't present.
    """
    da, db = _downsample(a), _downsample(b)
    if da is None or db is None:
        return (0.0, 0.0) if a == b else (999.0, 999.0)
    import numpy as np
    d = np.abs(da - db)
    h = d.shape[0]
    return float(np.mean(d[: h // 2])), float(np.mean(d[h // 2:]))


def _frame_sig(frame: bytes) -> bytes:
    """Stable signature for dedup: the downsampled bytes, or a hash fallback."""
    arr = _downsample(frame)
    if arr is not None:
        return arr.tobytes()
    import hashlib
    return hashlib.md5(frame).digest()


# How many consecutive read-scrolls must be MEASURED to have moved the page 0px before the
# capture calls it the bottom. Not 1: a single swipe can be swallowed by a mid-animation frame or
# lost by the input transport, and ending a read on one lost gesture would silently truncate a
# real profile -- the expensive direction, since the model then chooses from an item list that
# stops part way down. Not more than 2 either: every extra confirmation is another futile swipe
# at a bottomed profile, and swiping at a page that cannot move is exactly the kind of thing
# ops/ANTI-BOT-RESEARCH.md exists to keep off the wire. Two independent quorate measurements of
# "nothing moved" across two real gestures is the smallest evidence that is not one accident.
_STATIC_PAIRS_FOR_BOTTOM = 2


# Whole-frame downsample mean-abs-difference above which two frames are too different to be the
# same page position, so the expensive `estimate_shift` probe below is skipped. Purely a cost
# gate: it may only ever say "obviously moved", never "bottom".
# [corpus: the 2026-08-16 Grace frames, the hardest real case there is -- a page that had stopped
# under a video repainting ~40% of the content band. The six STATIC pairs measure 1.24, 4.87,
# 4.91, 4.99, 6.77 and 12.49; the two genuinely SCROLLED pairs measure 49.19 and 51.72. Nothing
# lands between 12.5 and 49.] 36 sits ~2.9x above the static ceiling and ~1.4x below the scrolled
# floor, so a real 0px pair would have to animate three times harder than a playing video to be
# skipped -- and a frame that animated THAT hard has no static strips left for `estimate_shift`
# to reach quorum on, so it would be refused rather than measured either way.
_STATIC_PROBE_MAX_DIST = 36.0


def _static_pair_is_the_bottom(before: bytes, after: bytes,
                               content_band: tuple[float, float], *,
                               before_ds=None, after_ds=None) -> bool:
    """Whether frameshift AFFIRMATIVELY measured these two frames as the same page position.

    True only for a quorate `SHIFT_MEASURED` of exactly 0px. Every other outcome -- a refusal, a
    saturation report, an undecodable frame, a missing cv2 -- is False, so the capture keeps
    reading. That default is the point: this predicate can only ever STOP a read, a wrong stop
    truncates a profile the model then has to choose items from, and `estimate_shift` is the only
    comparator in this driver that says "I cannot tell" instead of guessing.

    It is deliberately not `_vertical_shift_match` or a `_band_dist` threshold. Both are
    mean-absolute-difference tests over the content band, and the frames this exists to classify
    have a VIDEO repainting up to half that band -- the mean is dominated by the one region that
    is genuinely changing, which is how it hides the fact that the page underneath it is not.
    Such a threshold appears here only as `_STATIC_PROBE_MAX_DIST`, where its one job is to skip
    work on frames that obviously moved; it is never allowed to conclude the opposite.
    """
    if before_ds is not None and after_ds is not None:
        try:
            import numpy as np
            if float(np.abs(before_ds.astype("float32")
                            - after_ds.astype("float32")).mean()) > _STATIC_PROBE_MAX_DIST:
                return False
        except Exception:  # noqa: BLE001 — the gate is an optimisation; fall through and measure
            pass
    try:
        shift = estimate_shift(before, after, content_band=content_band)
    except Exception:  # noqa: BLE001 — decode/cv2/geometry failure is "cannot tell", never "bottom"
        return False
    return shift.status == SHIFT_MEASURED and shift.delta_px == 0


# The hard bounds _sample_read_step/_sample_read_scroll validate any policy-sampled read-scroll
# distance against before issuing a gesture (see both call sites below). Named here so
# _vertical_shift_match derives its search radius from the SAME numbers instead of a second,
# independently-drifting copy of them -- two frames _capture_current() stopped at while reading
# one profile top-to-bottom can be up to _READ_SCROLL_FRAC_MAX apart.
_READ_SCROLL_FRAC_MIN = 0.10
_READ_SCROLL_FRAC_MAX = 0.75

# --- item enumeration (ops/OPENER-REDESIGN.md 5.2/5.3/5.5) --------------------------------
# The per-profile screencap ceiling for a read that is ALSO building an item index, replacing
# `scroll_captures` (config.yaml: 12) for that read only. The two are different jobs and the
# doc says so in as many words: `scroll_captures` sizes the ordinary profile read, while the
# closed loop steps at most ~1/3 of the locally measured card spacing and therefore needs
# several times as many frames for the same profile.
#
# DERIVED FROM MEASURED GEOMETRY. The original 48-frame derivation used two captures containing
# only the six-media/three-text-prompt core. A 2026-08-15 production capture disproved the
# assumption that this was the structural maximum: Hinge also permits optional video and voice
# prompt content. That profile advanced at least 11,773px in 47 gestures and still had not
# reached the bottom. Treat that observation as a new lower bound, not as an outlier to truncate:
#
#   PAGE SPAN.    The first two calibration profiles measure 8349px and 10027px and both carry
#                 nine selectable items. The production profile with optional animated prompt
#                 content traversed ~11,773px before the old ceiling cut it off, proving that
#                 10,027px and nine ordinary items were not a complete structural bound.
#   REALISED STEP. The closed loop does NOT step `scroll_step._MAX_STEP_PX` (363px). It draws
#                 uniformly inside a window whose ends are the ratio rule applied to the SMALLEST
#                 spacing seen so far on this profile, and both calibration profiles contain a
#                 685px card, so both loops end up drawing from (219, 265). Measured mean step
#                 with that memory in play: 235..262px, and the loop was measured to need 35
#                 gestures (36 frames) for profile B and 43 (44 frames) for profile A.
#   THE BOUND.    64 frames provide 63 gestures. Even if every draw lands on the 219px gesture
#                 floor, that covers 13,797px -- 17% beyond the new 11,773px observed lower
#                 bound. Normal profiles do not pay for the headroom: the repeated-frame bottom
#                 signal still ends their loop as soon as it did before.
#
# A tighter value would have to come from a faster cadence, which doc 5.10.1 forbids (a step past
# ~1/3 of the local spacing aliases the count against the card pitch), so the only honest
# tightening available is none. Reading "~28 steps for a ~10,000px profile" off `_MAX_STEP_PX`
# assumes every draw is the ceiling, which happens on no profile in the corpus.
#
# ABOVE THE CEILING THE READ IS LOUD, not silent (doc 5.11's own ask, and the reason this comment
# is not just arithmetic). `_note_enumeration_truncated` prints the profile's frame count and this
# constant, records a `capture_enumeration_truncated` debug action, and the flag still reaches the
# model on `Profile.items_truncated` per doc 5.7. It is deliberately NOT a hard stop: a profile
# longer than the ceiling is not an error, and the owner's stop-condition rule scopes stops to an
# unrecognized screen or an error rather than to a quota. What a hard stop would buy is nothing --
# the item the model picks is inside the enumerated region either way, so navigation is unaffected
# -- and what it would cost is a halted run on a legitimate profile.
#
# NOT a config key, deliberately: it is a property of the measured card geometry and the
# gesture window, not an operator preference, and `scroll_captures` must stay exactly what it
# is for the observe read and for `_ensure_session_top`'s swipe ceiling. The jitter that keeps
# profile after profile from terminating at one identical depth is unchanged and still applied
# on top of this, by `_capture_limit_for_profile`.
_ENUMERATION_CAPTURE_LIMIT = 64


def _ranker_frames_from_enumeration(frames: list[bytes], target: int) -> list[bytes]:
    """Downsample an enumeration-cadence capture back to the ranker's ordinary frame budget
    (audit fix, "BUG 3", 2026-08-12).

    THE PROBLEM. `_ENUMERATION_CAPTURE_LIMIT` (64, above) replaced `scroll_captures` (12,
    config.yaml) as the CEILING for a read that is also building an item index -- the index
    needs the finer, closed-loop cadence to avoid the step/spacing aliasing doc 5.10.1
    measured, so raising the ceiling for THAT consumer is correct and is not touched here. But
    `_capture_current` used to hand every frame it read straight to `Profile.photos` regardless
    of which ceiling was in force, and `Profile.photos` is also the RANKER's whole view of the
    profile (`decider.decide`, worker.py's auto loop). Per-profile pooling is not scale-free
    (aggregation-design.md: ArcFace is a raw MEAN over every detected face, CLIP dedups by
    cosine similarity but does not equalise weights), so a card sampled ~4x as often at the
    enumeration cadence would carry ~4x the weight it carried before Part B -- and only on
    AUTO, since observe (the only mode that produces the labels the ranker is trained against,
    per doc 4) never enumerates and always reads at the 12-cadence. Left alone, every auto
    decision would run train/serve skewed against its own labels in a way that looks like a
    ranker regression but is actually a change to the ranker's INPUT.

    THE FIX is a resample, not a smaller enumeration ceiling: the index still gets every frame
    it needs (this function never touches `photos` before `_index_captured_items` folds it),
    and only the copy handed to `Profile.photos` -- the ranker's copy -- is thinned back down.
    Evenly spaced across the WHOLE captured range and always including the first and last frame
    (`round(i * (n - 1) / (target - 1))` for `i` in `0..target-1`, deduplicated), so the ranker
    keeps seeing top-to-bottom coverage rather than just the top of the profile, which is what
    naively keeping the first `target` frames of a finer-grained read would do.

    `target` is `self.scroll_captures`, the CONFIGURED base with no jitter applied. The jitter
    `_capture_limit_for_profile` adds on top of a base exists so the DEVICE-facing scroll
    ceiling is not a fixed, externally observable bot signature; this function produces nothing
    Hinge's servers or a human ever sees (it runs entirely after every gesture for this profile
    has already happened), so that reasoning does not transfer here -- a second, unrelated
    random draw would only make the ranker's input size non-deterministic for no anti-detection
    benefit.

    NOT EXACT EQUIVALENCE to what the ranker received before Part B, and that is stated plainly
    rather than papered over. The pre-Part-B read walked the profile in big (~1299px at the
    shipped 0.55 `read_scroll_frac`) steps and stopped the moment a repeated frame signalled the
    bottom, so a SHORT profile could see fewer than `target` frames at that coarse spacing. This
    function instead resamples whatever the (finer-grained, closed-loop) capture already read,
    which on a short profile can still return close to `target` frames spanning the same short
    page -- i.e. it can hand the ranker a slightly denser sampling of a short profile than the
    old cadence ever would have, though never more frames than the device actually produced
    (`min(len(frames), target)`) and never a synthesised one: every frame returned is a genuine
    frame this capture read. The ranker's own pooling (ArcFace's mean, CLIP's dedup+GeM) already
    collapses near-duplicate detections, so an extra same-content frame that would not have
    existed under the old cadence contributes at most a near-duplicate of a frame the old
    cadence WOULD have kept -- it is not a new face, pose, or piece of content, so the SKEW this
    function exists to remove is addressed even though the exact pre-Part-B frame set cannot be
    reconstructed after the fact.
    """
    n = len(frames)
    if n <= target or target <= 0:
        return list(frames)
    if target == 1:
        return [frames[0]]
    step = (n - 1) / (target - 1)
    indices = sorted({round(i * step) for i in range(target)})
    return [frames[i] for i in indices]


# How many times `_locate_target_heart` searches for the SAME item before the run stops.
#
# The owner rule draws the line between two things a fixed retry count has to keep apart:
# retrying the item the model chose is a shaky hand and is encouraged, while landing on a
# different item is a wrong decision and is forbidden outright (doc 5.6, "no falling back to
# `hearts[0]`, no 'closest reachable item'"). So this bounds the shaky hand, and there is no
# value of it that permits a substitution.
#
# 2, not 1 and not more, and the reason is what a repeat can actually change. Every attempt is
# one affirmative return to the top followed by the same bounded downward search, so attempt 2
# re-reads the profile from a re-established zero point -- which is exactly the failure the
# search's own +3 frame slack was added for (HINGE-05: "on a real device a scroll can
# over/undershoot the intended frame"), and a single over/undershoot is the one condition a
# repeat is measured to fix. A third attempt repeats the same deterministic comparison over
# frames captured from the same top by the same gestures; it buys another whole profile's worth
# of dwell for a case nothing has measured, and the honest answer to "the second read did not
# find it either" is that the capture and the screen disagree, which is a stop, not a third try.
_TARGET_HEART_ATTEMPTS = 2


def _content_rows(content_band: tuple[float, float], size: int) -> tuple[int, int]:
    """Row range `(r0, r1)` of `content_band`'s `(y0, y1)` fractions on a `size`-row
    downsample. A shared helper so every _vertical_shift_match call site derives its crop
    from the SAME content_band field instead of a second, independently-drifting copy of the
    arithmetic -- exactly the kind of drift _READ_SCROLL_FRAC_MIN/MAX's own module comment
    above calls out. Clamped so a degenerate size (0 or 1) can't produce an empty or
    inverted slice."""
    y0, y1 = content_band
    r0 = max(0, min(size - 1, round(y0 * size)))
    r1 = max(r0 + 1, min(size, round(y1 * size)))
    return r0, r1


def _vertical_shift_match(cur, seen, *, threshold: float,
                          rows: tuple[int, int] | None = None) -> tuple[bool, int, int]:
    """Whether downsampled frame `cur` could be a vertically-scrolled view of `seen`, at ANY
    vertical offset -- not just the one wait_for_decision's exact-position check already tried.

    That exact-position check only recognizes a manual scroll that happens to land back on one
    of _capture_current's own fixed read-scroll stops. A human scrolling by hand to actually
    read a profile -- Signals behavior #1, and exactly what the owner was doing when this was
    reported -- stops wherever their thumb does, not at the bot's own stride. Everywhere in
    between the automated stops used to read as "whole card changed" and got silently recorded
    as a PASS despite no pass/like tap ever happening.

    This tries every vertical offset a real scroll between two adjacent read-scroll stops could
    produce (bounded by _READ_SCROLL_FRAC_MAX in either direction -- the search doesn't need to
    know or care which way the human actually scrolled) and matches on the best-aligned
    OVERLAPPING band at each one. A genuine scroll's shared content lines up at some offset
    wherever it stopped; two frames from actually different profiles share no such alignment at
    any offset tried here, so a real pass is still recognized as one.

    `rows`, when given, is a `(r0, r1)` row range (see `_content_rows`) BOTH frames are sliced
    to before the shift search runs -- this is the fix for a bug this helper shipped with
    once already. MEASURED on the real Pixel 7a (1080x2400) 2026-08-10, mean-abs-diff
    best-shift distance across five real frames of one profile scrolled to different offsets:

        pair (same profile, different scroll offsets)   full-frame   content-rows-only
        scrolled A vs scrolled B                            38.94          3.48
        scrolled A vs scroll-top                            41.38          5.41
        scrolled A vs scrolled N                            30.38          3.34
        scroll-top vs deep-scrolled                         32.88         11.94

    change_threshold is 9.0. Every full-frame number above is 3-4x over it, so searching the
    WHOLE downsampled frame (`rows=None`, this function's very first shipped shape) could
    never return True in production: the status bar, the sticky header, and the bottom nav
    bar do NOT translate when the content scrolls, so they dominate the mean-abs-diff at
    EVERY offset tried, no matter how well the actual content lines up underneath them.
    Restricting the search to `content_band`'s rows fixes 3 of the 4 pairs above; the 4th
    (a scroll-top frame vs one scrolled 13 of 24 downsample rows further down) still misses
    at any offset -- which is exactly why this helper is corroboration (layer 2), never the
    authoritative check: the identity-band anchor (`_identity_of`, layer 1) is what actually
    closes the bug, and wait_for_decision only reaches this helper at all once identity has
    already failed to call the frame 'same'.

    `rows=None` (the default) searches the whole frame -- kept for any caller with no
    content_band to slice against, and for the direct unit tests below that exercise the
    shift-search arithmetic itself rather than the chrome-exclusion fix.

    Returns `(matched, shift, overlap_rows)` -- a bool used to be the WHOLE return value, and
    that bool's semantics are unchanged (first offset under `threshold`, scanned in the same
    `-max_shift..max_shift` order as before). `shift` and `overlap_rows` exist because the
    bool alone is exactly what made the incident that motivated this change undiagnosable: the
    only trace it left was "matched a stored signature", with no way to see WHICH one, at what
    offset, or how much of the band actually lined up. `shift` is the offset the decision was
    made at (positive = `cur`'s content sits `shift` downsample rows below `seen`'s);
    `overlap_rows` is how many of `content_band`'s rows were actually compared at that offset
    (`band_h - abs(shift)` -- the rest fell off one end and were never compared at all). When
    NO offset matched, `shift`/`overlap_rows` describe the BEST (lowest mean-abs-diff) offset
    tried instead, purely as a diagnostic for "how close did it get" -- `matched` is still the
    only thing any caller's control flow may depend on.
    """
    import numpy as np
    size = cur.shape[0]
    r0, r1 = rows if rows is not None else (0, size)
    cur_c, seen_c = cur[r0:r1], seen[r0:r1]
    band_h = cur_c.shape[0]
    # MEASURED evidence toward a possible future tightening of the match itself (NOT done in
    # this pass -- see the module docstring's redesign notes for why): of the 4 genuine
    # same-profile shift-match pairs observed live on the Pixel 7a while diagnosing the
    # Zorva/Qelix incident, overlap_rows/band_h came out to 9/18, 11/18, 12/18, and 18/18 --
    # all >= 50%, against a floor this function has always allowed of 4/18 (~22%, the
    # `band_h - 4` term below). That is suggestive that a >=50% overlap floor could reject
    # weaker, more coincidental matches without losing a real one -- but 4 samples is far too
    # few to move a threshold that exists to guard against the ORIGINAL false-PASS bug
    # (mis-swallowing a genuine advance as "just a scroll"). This needs on-device
    # LIVE-VERIFY first (see ops/RUNBOOK.md's convention for that), which the new
    # `overlap_rows` return value is what will finally make possible to gather at scale.
    max_shift = max(1, min(band_h - 4, math.ceil(_READ_SCROLL_FRAC_MAX * size)))
    best_shift, best_overlap, best_dist = 0, band_h, None
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            cur_band, seen_band = cur_c[shift:], seen_c[:band_h - shift]
        else:
            cur_band, seen_band = cur_c[:band_h + shift], seen_c[-shift:]
        if cur_band.size == 0:
            continue
        dist = float(np.mean(np.abs(cur_band - seen_band)))
        if best_dist is None or dist < best_dist:
            best_dist, best_shift, best_overlap = dist, shift, cur_band.shape[0]
        if dist < threshold:
            return True, shift, cur_band.shape[0]
    return False, best_shift, best_overlap


# --- identity: resolving the scroll-top "top" verdict by OCR'ing the card header ---------
# See _identity_of's own "Layer 1b" block for the full rule this feeds. Module-level (not a
# class attribute like _OCR_NAME_RE) because both need to be usable without an AndroidDriver
# instance in hand -- the ratio calibration below is itself a standalone, testable fact about
# two strings, independent of any driver state.

_NAME_MATCH_RATIO = 0.6
# How close a token OCR'd from identity_top_name_band must score against the stored profile
# name (difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio(), called via
# _name_token_matches below with the OCR'd TOKEN first and the STORED name second -- this
# ratio is NOT symmetric under argument swap, so the numbers here are specifically the call's
# real order, not the reverse) to count as a match for that profile. Deliberately calibrated
# LOW -- biased toward "same" -- because the two possible errors here are not symmetric: a
# false "same" costs at most a missed pass (the loop just keeps waiting; nothing is written,
# exactly today's behaviour), while a false "new" records a PASS the human never made and
# corrupts the taste model with a decision that did not happen. MEASURED on the real
# incident's OCR reads: "Zorva" vs plausible misreads "Zorba" and "Zorna" score 0.80 --
# both MUST stay "same". The next profile's OCR'd token ("qelix") against the stored name
# ("zorva") -- SequenceMatcher(None, "qelix", "zorva"), the call's real argument order --
# scores 0.00 (as does the reverse order) -- MUST become "new". 0.6
# sits comfortably below every observed misread and far above every observed genuine
# difference, i.e. on the safe side of both real data points this fix was built to get right.
#
# Ratio alone is not enough: it was calibrated only against substitution-style misreads and
# scores BELOW 0.6 for a plain TRUNCATION of a longer name ("Zorva" read as "Zo" is 0.57,
# "Katherine" read as "Kat" is 0.50, as "Ka" is 0.36) -- a common tesseract failure (a partial
# crop at the band edge, tight kerning), and truncation gets WORSE, not better, the longer the
# stored name is. _name_token_matches below adds a prefix test specifically to close that hole
# without touching this ratio (see its own docstring for why a prefix test is the right shape
# of fix and the full set of truncation cases it was verified against).
def _name_token_matches(seen: str, stored: str) -> bool:
    """True if an OCR'd token `seen` and the profile's stored name `stored` are close enough
    to call the SAME person, biased toward "same" per this module's own reasoning above (a
    false "same" costs a wait; a false "new" writes a wrong label).

    Two independent tests, either is enough:

      1. difflib ratio >= _NAME_MATCH_RATIO -- catches substitution-style misreads
         ("Zorva"/"Zorba", "Zorva"/"Zorna").
      2. either string is a case-insensitive PREFIX of the other -- catches TRUNCATION, which
         is a common tesseract failure (a partial crop at the band edge, tight kerning) and
         scores BELOW the ratio bar for longer names (see the module comment above). A
         truncation is by definition a prefix, so this one test closes the hole in BOTH
         directions: a truncated READ this poll ("Zo" seen for a profile stored as "Zorva")
         and a truncated STORE from a bad capture-time read ("Zo" was what got stored for a
         profile actually named "Zorva", so a later full "Zorva" read must still match it --
         otherwise the bad capture poisons the whole profile with spurious "new" verdicts).

    VERIFIED (2026-08-10) against every case this fix targets -- read as seen(stored)=ratio:

      truncated reads, both directions -- all "same" via the prefix test:
        Zo(Zorva)=0.57, Kat(Katherine)=0.50, Ka(Katherine)=0.36    -- read got truncated
        Zorva(Zo)=0.57, Samantha(Sam)=0.55                         -- STORED name was truncated
      genuine differences -- still "new" (no prefix relationship, ratio stays below bar):
        Zorva(Qelix)=0.00, Qelix(Zorva)=0.00, Katherine(Michelle)=0.35,
        Zorva(Signals)=0.17
      genuine same-name misreads -- unaffected, still "same" via the ratio path alone:
        Zorba(Zorva)=0.80, Zorna(Zorva)=0.80, Zorla(Zorva)=0.80, orva(Zorva)=0.89
    """
    seen_cf, stored_cf = seen.casefold(), stored.casefold()
    if seen_cf and stored_cf and (seen_cf.startswith(stored_cf) or stored_cf.startswith(seen_cf)):
        return True
    return difflib.SequenceMatcher(None, seen_cf, stored_cf).ratio() >= _NAME_MATCH_RATIO


_TOP_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
# Tokens considered as name candidates in the scroll-top card-header OCR read: a letter
# followed by one or more letters/apostrophes/hyphens -- i.e. length >= 2, which excludes
# stray single-character OCR noise ("|", "l", "@") without needing a separate length check.
# This is the SAME-match tokenizer only -- a token this short CAN still resolve a "same"
# verdict via _name_token_matches' prefix test above (a 2-char truncation like "Al" or "Ka" is
# exactly the failure mode that test exists for). The "new"-candidate selection in
# _identity_of's Layer 1b block applies its OWN, stricter length floor (>= 3 alphabetic
# characters) on top of this regex, because a 1-2 character token is noise, never a usable
# name, and must never by itself be the basis for recording a PASS.
_TOP_NAME_CHROME_WORDS = frozenset({
    "signals", "active", "today", "shows", "thoughtful", "age", "height", "dating", "intent",
})
# Words HINGE ITSELF renders in the card-header band, observed in the real reads that built
# identity_top_name_band ("Zorva @ | @ Signals Active today", Qelix's "Signals ( Agev )
# Height v" equivalent at scroll-top): none of these is ever a person's first name. Without
# this blocklist, a frame where OCR catches the chrome text but misses the name entirely would
# take "Signals" (say) as the candidate name, score ~0.1-0.2 against the stored name -- clearing
# nothing, but ALSO not blocked -- and the "first non-chrome token" fallback below would
# misread it as a genuinely new profile: a FALSE PASS. With the blocklist, a read containing
# only chrome words yields no candidate at all and the verdict stays "top" (inconclusive) --
# the safe outcome; the pixel/content layers downstream still get to decide.


def _clean_first_line_name_candidate(text: str) -> str | None:
    """Return one clean, non-chrome name candidate from OCR's first line, or ``None``.

    Both identity OCR bands put the person's name on their first non-empty line when they are
    actually looking at a name. Requiring exactly one candidate keeps filter-chip chrome and
    photo-text garbage inconclusive; the caller still decides whether this candidate is the
    captured name or a possible next profile.
    """
    first_line_tokens = _TOP_NAME_TOKEN_RE.findall(text.splitlines()[0]) if text else []
    candidates = [
        tok for tok in first_line_tokens
        if tok.casefold() not in _TOP_NAME_CHROME_WORDS
        and sum(ch.isalpha() for ch in tok) >= 3
    ]
    return candidates[0] if len(candidates) == 1 else None


# --- vision: locate an action BUTTON by its glyph (not a fixed coord) --------------------
# Hinge's like-heart sits at the bottom-right of EACH photo and the pass-X floats bottom-left.
# A fixed fraction is unreliable: the "Start sending likes" banner and per-profile photo
# aspect ratios shift the heart vertically, and the X's white disc merges into the white
# background of a prompt card (so a plain white-blob detector loses it). We instead template-
# match the dark glyph (heart / X), which stays distinct on ANY background. The glyphs are
# fixed-resolution UI assets (this driver targets one device — the Pixel 7a at 1080x2400),
# so matches are essentially exact. Any app's spec can plug its own glyph PNGs into the same
# roles (see AndroidAppSpec.templates); an app with no template for a role simply skips
# vision-location for it (AndroidDriver._template returns None, callers fall back to a fixed
# coordinate, or — for upsell_dismiss, which has no safe fixed-coordinate fallback — no-op).

@functools.lru_cache(maxsize=None)
def _load_template(name: str):
    """Load a grayscale button-glyph template via cv2 (cached). None if cv2/asset missing."""
    try:
        import cv2
    except Exception:  # noqa: BLE001 — cv2 absent -> caller falls back to fixed coords
        return None
    return cv2.imread(str(_ASSETS / name), cv2.IMREAD_GRAYSCALE)   # None if file missing


def _match_glyph(frame: bytes, template, *, side: str, threshold: float = 0.6,
                 y_band: tuple[float, float] | None = None) -> list:
    """Locate a button glyph in a screencap by normalized cross-correlation. `side` keeps only
    matches on the right ('like' heart) or left ('pass' X) of the screen. Returns (x, y)
    centers sorted top->bottom (so [0] is the first photo's heart after scroll-to-top). Empty
    list if cv2/template/decoding is unavailable — the driver then uses its fixed-coord fallback.

    `y_band`, when given, is a `(y0, y1)` fraction-of-height band (the same shape as
    HINGE_SPEC.content_band) that a match's CENTER must fall inside. It exists for the "like"
    role specifically: Hinge's bottom nav bar carries two persistent false positives at y=2258
    (constant, static UI chrome, not photo content) -- the "Matches" tab icon (~0.65 correlation,
    previously excluded only by _LIKE_MATCH_THRESHOLD's 0.75 floor, a margin of ~0.10) and the
    "Likes" tab heart (~1.0 correlation, previously excluded only by `side`'s x >= 0.55*w cutoff,
    a margin of 54px/5% of screen width). Both sit outside content_band (y 300..2100 of 2400 on
    the calibrated device); every genuine card heart (y 570..1890, MEASURED against 115 real
    frames) sits inside it. Masking the band structurally excludes both false positives instead
    of relying on threshold/side margins that a layout shift could erode. See _match_glyph's
    call sites in _locate_button/_locate_target_heart/_observe_glyph_visible for why only "like"
    passes this (the floating pass-X can legitimately sit low, outside content_band)."""
    if template is None:
        return []
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        h, w = img.shape
        th, tw = template.shape
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        work = res.copy()
        if y_band is not None:
            # Mask OUT-OF-BAND rows to -1 (the same sentinel the loop below uses to retire an
            # already-found peak) BEFORE the non-max-suppression loop starts, rather than
            # filtering matches after minMaxLoc returns them. This matters because the loop's
            # 12-iteration budget is consumed on every pass regardless of what the body decides
            # to do with the peak it found (see the `side` check a few lines down, which DOES
            # reject a wrong-side peak only after minMaxLoc already spent an iteration finding
            # it -- an existing wart this deliberately does not reproduce for y_band). Masking
            # `work` up front means an out-of-band peak -- e.g. either nav-bar false positive
            # above -- is never returned by minMaxLoc at all, so it can never starve the budget
            # a real in-band match further down the correlation surface would need.
            r0, r1 = _content_rows(y_band, h)
            # `res`/`work` rows are the template's TOP-LEFT y (`loc[1]`), but the band is
            # defined against a match's CENTER (`loc[1] + th // 2`, same as `cy` below) -- shift
            # by that half-height before masking, or a template straddling the band edge would
            # be kept/dropped by its top-left corner instead of the center this function reports.
            wr0 = max(0, r0 - th // 2)
            wr1 = max(wr0, r1 - th // 2)
            work[:wr0, :] = -1.0
            work[wr1:, :] = -1.0
        centers: list = []
        for _ in range(12):                                   # non-max suppression loop
            _, maxv, _, loc = cv2.minMaxLoc(work)
            if maxv < threshold:
                break
            cx, cy = loc[0] + tw // 2, loc[1] + th // 2
            work[max(0, loc[1] - th // 2): loc[1] + th, max(0, loc[0] - tw // 2): loc[0] + tw] = -1.0
            if side == "right" and cx < w * 0.55:
                continue
            if side == "left" and cx > w * 0.45:
                continue
            centers.append((int(cx), int(cy)))
        centers.sort(key=lambda c: c[1])
        return centers
    except Exception:  # noqa: BLE001 — any cv2/decoding failure -> fixed-coord fallback
        return []


# Correlation floor for the "like" role specifically -- NOT a change to _match_glyph's own 0.6
# default above, which stays as-is for every other role (pass/confirm/upsell_dismiss/paywall
# weren't touched by this recalibration and their own margins against 0.6 are unverified here).
#
# MEASURED 2026-08-11 against ops/calibration/scroll_20260811T211209Z/ (115 real-profile frames,
# 1080x2400, gitignored) with hinge_like_button.png as the template:
#   - 199 genuine card-heart matches (side="right", every frame that had a button on screen)
#     scored 0.815 .. 1.000, mean 0.999. The single sub-0.93 outlier (0.815) was a button
#     straddling the very bottom screen edge, only ~2/3 visible -- an inherent template-matching
#     limitation for a partially off-screen glyph, not a template defect; every FULLY visible
#     button scored >= 0.933.
#   - The one reproducible false positive on the right side is Hinge's own bottom-nav bar glyph
#     (the "Matches" tab icon) at a constant (756, 2258): it scored 0.6527 in literally all 115
#     frames (stable to 5 decimals -- it's static UI chrome, not photo content). That is BELOW
#     0.75 but ABOVE _match_glyph's 0.6 default, which is exactly why 0.6 is unsafe here: at 0.6
#     that nav icon is a "like" button hit on every single frame.
#   - Hinge's bottom-nav "Likes" tab icon (a heart, at a constant (540, 2258)) is a near-perfect
#     match too (~1.0! it's visually almost the same glyph, just outline-on-dark instead of
#     filled-circle) but sits on the LEFT half of the screen (540 < 0.55 * 1080), so side="right"
#     already excludes it regardless of threshold. Flagged here because the margin is thin (540
#     vs the 594px cutoff, 54px / 5% of screen width) -- a layout shift that moved that icon
#     right of center would reintroduce it as a live false positive.
#
# 0.75 sits with ~0.10 of margin above the worst known false positive (0.6527) and ~0.065 under
# the worst real (edge-clipped) true positive (0.815); every fully-visible true positive clears
# it by >= 0.18. Not the same 0.6 every other role uses, and not raised further, because raising
# it past ~0.815 would start rejecting that legitimate edge-clipped case.
#
# Addendum 2026-08-11: both nav false positives above are now ALSO excluded structurally, via
# _match_glyph's y_band parameter (wired to self.content_band, (0.125, 0.875) -> y 300..2100)
# at every role=="like" call site. Both sit at y=2258, below content_band's lower edge; every
# real card heart (y 570..1890, measured against the same 115 frames) sits inside it. This
# threshold and the side="right" cutoff both stay as they were -- y_band is an added,
# independent layer, not a replacement for either -- so the "Matches" tab margin (~0.10 above
# 0.6527) and the "Likes" tab margin (54px of screen width) described above are no longer the
# ONLY thing standing between either false positive and a live match.
#
# This is keyed by ROLE NAME ("like"), not by app -- AndroidDriver (this class) is shared with
# BumbleAndroidDriver. BUMBLE_SPEC.templates is currently {} (empty, uncalibrated -- see
# operation_love/drivers/android/bumble.py), so no Bumble call ever reaches a role=="like"
# branch today. If Bumble later grows its own "like" template, this threshold (and the
# skip-inversion behaviour in _observe_glyph_visible below) would apply to it too, sight
# unseen -- whoever wires that up should re-measure against Bumble's own asset rather than
# assume this number still holds.
_LIKE_MATCH_THRESHOLD = 0.75


def _retry_until(check_fn, tries: int, delay_s: float, *, is_found=bool):
    """Call check_fn() up to `tries` times, sleeping human_delay(delay_s) after each
    attempt where `is_found(result)` is False. Returns the first result for which
    is_found(result) is True, or None once attempts run out. The shared shape behind
    _await_button/_verify_progress/_handle_rose_upsell: poll something on-screen with
    humanized pacing until it appears or we give up.

    `is_found` defaults to `bool` (any truthy result counts as found) — correct for
    every current caller (None/bool/non-empty-list are never falsy-but-genuinely-found
    here), but pass an explicit predicate (e.g. `lambda r: r is not None`) for a
    check_fn whose "found" result can legitimately be falsy, so a falsy-but-valid hit
    isn't mistaken for "not found yet" and silently retried away."""
    for _ in range(max(1, tries)):
        result = check_fn()
        if is_found(result):
            return result
        time.sleep(human_delay(delay_s))
    return None


class AndroidDriver(DatingAppDriver):
    """Drives ANY Android dating app whose UI fits the shape captured by AndroidAppSpec:
    a scrollable profile card, a like/pass control pair (vision-located with a fixed-coord
    fallback), and either a comment-sheet or a direct like flow. See HingeDriver (below, in
    this module) and BumbleAndroidDriver (operation_love/drivers/android/bumble.py) for the
    two current bindings.
    """

    # Both capture entry points (next_profile / current_profile) accept and honour a
    # `should_stop` callable, polled between screencaps, between read-scrolls, inside the read
    # dwell, and between _scroll_to_top's undo-swipes. Declared here (class-level, so both
    # Android bindings inherit it) because worker.py only passes the callable to a driver that
    # advertises it -- see DatingAppDriver.supports_interruptible_capture. It is True for the
    # whole AndroidDriver family and not for BumbleWebDriver, whose own capture waits are not
    # stop-aware: a flag claiming a capability the code does not have would be worse than no
    # flag, since the worker would then believe Stop is handled when it silently is not.
    supports_interruptible_capture = True
    supports_interruptible_like_navigation = True

    def __init__(self, cfg, spec: AndroidAppSpec):
        self.spec = spec
        self.accepts_opener = spec.accepts_opener
        # Only comment-sheet apps have an intermediate human like intent. A direct-flow
        # Android driver (Bumble) must keep its ordinary pass/like observer contract.
        self.supports_observe_like_intent = (
            spec.like_flow == "comment_sheet" and spec.accepts_opener
        )
        self.think_time_calibrated = spec.think_time_calibrated
        app_cfg = (getattr(cfg, "apps", {}) or {}).get(spec.app, {})
        self.serial = app_cfg.get("serial") or None
        self.adb_path = app_cfg.get("adb_path", "adb")
        self.package = app_cfg.get("package", spec.package)
        self.scroll_captures = max(1, int(app_cfg.get("scroll_captures", spec.scroll_captures)))
        self.dwell_s = float(app_cfg.get("dwell_s", spec.dwell_s))
        self.read_scroll_frac = float(app_cfg.get("read_scroll_frac", spec.read_scroll_frac))
        self.change_threshold = float(app_cfg.get("change_threshold", spec.change_threshold))
        self.coords = {**spec.coords, **(app_cfg.get("coords") or {})}
        # Observe-mode decision detection (see the module docstring's redesign notes): every
        # field here is config-overridable per app the SAME way change_threshold/coords are
        # above, not read off self.spec directly, so a per-device geometry tweak lives in
        # config.yaml rather than requiring a code change.
        identity_band = app_cfg.get("identity_band", spec.identity_band)
        self.identity_band = tuple(identity_band) if identity_band is not None else None
        identity_top_name_band = app_cfg.get("identity_top_name_band", spec.identity_top_name_band)
        self.identity_top_name_band = (
            tuple(identity_top_name_band) if identity_top_name_band is not None else None)
        self.content_band = tuple(app_cfg.get("content_band", spec.content_band))
        self.observe_ignore_zones = tuple(
            tuple(zone) for zone in app_cfg.get("observe_ignore_zones", spec.observe_ignore_zones))
        self.observe_touch_watch = bool(app_cfg.get("observe_touch_watch", spec.observe_touch_watch))
        # Re-run AndroidAppSpec.__post_init__'s rect/arity/pairing checks against these five
        # fields' MERGED (config.yaml-overridden) values, not just spec's own hardcoded
        # literals. __post_init__ already validates identity_band/identity_top_name_band/
        # content_band/observe_ignore_zones's shape and 0..1 range, and refuses
        # observe_touch_watch=True paired with identity_band=None (and, since
        # identity_top_name_band was added, identity_top_name_band set with identity_band=None)
        # -- but until now it only ever ran ONCE, against HINGE_SPEC's/BUMBLE_SPEC's own
        # literals at import time, because that used to be the only time an AndroidAppSpec got
        # constructed. An operator's apps.<app>.identity_band/identity_top_name_band/
        # content_band/observe_ignore_zones/observe_touch_watch override above bypassed all of
        # that: an identity_band with x0/x1 swapped (an easy typo) constructed fine and then
        # silently disabled Layer 1 forever -- _band()'s blanket `except Exception: return
        # None` (below) swallows PIL's crop error and _identity_of falls back to 'unknown'
        # with no print, no warning, nothing; a content_band with the wrong arity constructed
        # fine and only surfaced as an uncaught ValueError deep inside wait_for_decision's
        # Layer 2 fallback, mid-session; and identity_band: null in config.yaml (dict.get
        # returns the stored None, not spec's default, when the key is PRESENT with value
        # null) with observe_touch_watch left at Hinge's default True reached exactly the
        # combination __post_init__ refuses to construct -- silently, because nothing here
        # ever asked it again after the merge. identity_top_name_band flows through this SAME
        # rebuild for the identical reason: an operator override that pairs it with
        # identity_band=None (or gives it a malformed rect) must fail loudly here, not leave
        # _identity_of silently unable to ever resolve a "top" verdict by name.
        #
        # dataclasses.replace() rebuilds a frozen AndroidAppSpec from `spec` with these five
        # fields swapped in, which reruns __init__ -- and therefore __post_init__ -- against
        # the MERGED values, purely for the side effect of __post_init__'s validation raising
        # the same ValueError it already raises for a bad literal. This is the only way to get
        # that validation without a second, independently-drifting copy of the rect/arity/
        # pairing logic living here too (see AndroidAppSpec.__post_init__ for what actually
        # gets checked). The resulting spec is discarded -- self.spec keeps pointing at the
        # pre-merge instance, since only these five fields are ever config-overridden this
        # way; self.coords/self.read_scroll_frac have their own equivalent check in config.py's
        # _validate_android_fractions, deliberately not duplicated here.
        try:
            dataclasses.replace(
                spec,
                identity_band=self.identity_band,
                identity_top_name_band=self.identity_top_name_band,
                content_band=self.content_band,
                observe_ignore_zones=self.observe_ignore_zones,
                observe_touch_watch=self.observe_touch_watch,
            )
        except ValueError as exc:
            raise DriverClosed(
                f"{spec.app}: apps.{spec.app} config.yaml overrides of identity_band / "
                f"identity_top_name_band / content_band / observe_ignore_zones / "
                f"observe_touch_watch produced an invalid combination once merged with "
                f"{spec.app}'s spec defaults ({exc}). Fix the offending apps.{spec.app}.<key> "
                f"named above in config.yaml -- leaving it broken would silently disable the "
                f"identity anchor (or worse) for the whole run with no further warning."
            ) from exc
        # Parse this only AFTER the effective config-overridden geometry exists.  Bounds measured
        # over one header/content crop cannot license targeting after either crop changes.
        self.targeting_calibration, self._targeting_calibration_unavailable = (
            _parse_targeting_calibration(
                app_cfg.get("targeting_calibration"), self.serial,
                identity_band=self.identity_band, content_band=self.content_band))
        self._targeting_runtime_version_name: str | None = None
        self._targeting_runtime_frame_size: tuple[int, int] | None = None
        self.observe_name_ocr = bool(app_cfg.get("observe_name_ocr", True))
        self._touch_watcher: TouchWatcher | None = None   # started in open_session, closed in close()
        self._touch_watch_health_warned = False   # print the event_count==0 warning at most once/run
        # Whether THIS session is autonomous (Worker._auto_loop) rather than observe. The
        # driver has no `mode` of its own -- Worker never passes one in (make_driver builds a
        # bare AndroidDriver; mode lives only on Worker itself) -- so this starts False
        # (observe/unknown, the conservative default that preserves every existing caller's
        # behavior, including calibration tools like tools/hinge_inspect.py that construct a
        # driver directly and never call set_auto_session_policy at all) and flips True only
        # inside set_auto_session_policy below, the one hook Worker._auto_loop already calls,
        # unconditionally, before open_session() -- see that method's docstring for why this
        # is the cheapest correct signal rather than a new mode-passing parameter. Read by
        # open_session()'s touch-watcher gate: gesture corroboration exists to corroborate a
        # HUMAN tap against wait_for_decision's identity-and-deck-ready proof, and auto mode
        # never calls wait_for_decision at all (it drives itself via like()/dislike()), so
        # attaching TouchWatcher for it would hold a persistent `adb shell getevent` subprocess
        # open for the whole run for a reader that never exists.
        self._auto_session = False
        # Whether an opener will actually be REQUESTED for a like this session -- i.e. whether
        # anything downstream can consume the numbered item list enumeration builds. Starts True
        # (the conservative default that preserves every existing caller's behaviour), and is set
        # once per session by BOTH worker loops from
        # `opener_service is not None and not opener_service.disabled` -- the exact condition
        # that already tells "openers administratively off" apart from "a live service" at every
        # other call site in worker.py. Since doc 5.9's inversion this is the ONLY session-level
        # gate on enumeration (`_auto_session` no longer gates it, because observe enumerates
        # too), which is why a calibration tool that wants the ordinary 12-frame read has to say
        # so explicitly -- tools/hinge_bot_scroll_probe.py calls set_opener_enabled(False) for
        # exactly that reason. Read by _item_enumeration_blocker (audit
        # fix, "BUG 2", 2026-08-12): item enumeration exists solely to let the model pick an item
        # for an opener, so with opener.enabled: false there is no consumer for it at all, and it
        # must not run -- reading a profile enumerated at _ENUMERATION_CAPTURE_LIMIT (64 frames,
        # ~3x the dwell) for a run that was only ever going to send bare likes is both wasted
        # device time and, worse, the one condition doc 5.2's "never fall back to raw frames"
        # stop was never meant to fire for: nothing was ever going to consume the numbered list,
        # so its absence is not a failure. See _item_enumeration_blocker's docstring.
        self._openers_enabled = True
        # Set fresh by _capture_current every profile; None until a profile actually reveals
        # the sticky header (identity_top_sig) and, past that, until the header itself
        # (identity_sig) is seen. See _capture_current's identity-anchor block and
        # _identity_of, which is the only reader of these three.
        self._identity_top_sig = None
        self._identity_sig = None
        self._identity_name = None
        # Freshest frame that PROVED the like composer, so a resolved like is filed
        # against what the human actually had on screen. Reset per wait by
        # _wait_for_decision_unlocked; see _note_observe_like_outcome for why.
        self._observe_like_evidence = None
        # The first band that differs from the top chrome is only a CANDIDATE until a later
        # frame reproduces it. Hinge can briefly draw a half-transitioned filter-chip/header
        # strip after a scroll; treating that one frame as authoritative caused the stable
        # header on the very same Shuman profile to be reported as a deck advance.
        self._identity_anchor_confirmed = False
        self._identity_anchor_frame = None
        self._identity_anchor_frame_index = None
        # The identity OCR text _identity_of last used for a verdict (or None if neither the
        # tight header nor scroll-top card-header read produced one). Reset on every call,
        # not just every profile, so an observe_scroll debug record logged right after can
        # report exactly what was seen at the moment THAT verdict was decided.
        self._identity_top_name_read = None
        # What the identity OCR concluded ('same' / 'new'), or None if it never ran or never
        # reached a verdict this call. Reset alongside _identity_top_name_read, for the same
        # reason. wait_for_decision reads this immediately after the FIRST _identity_of(cur)
        # call of a poll to tell a name-derived 'new' apart from a pixel-derived one -- see
        # its own "name-derived 'new' must be reproduced" comment for why that distinction
        # gates whether a confirm-frame 'top' is allowed to corroborate a PASS.
        self._identity_top_name_verdict = None
        # The clean name token behind a ``new`` OCR verdict, if there was one. A no-touch-data
        # PASS requires the same candidate on both settled frames; two unrelated OCR guesses
        # are movement evidence, not proof that one new profile is on screen.
        self._identity_name_candidate = None
        # Bounded memo cache for _ocr_band: keyed on (rect, psm, sha1(frame)), holding at most
        # _OCR_BAND_CACHE_MAX entries. See _ocr_band's own comment for the measured cost this
        # exists to avoid. Reset per profile (in _capture_current) alongside the other identity
        # state, not just here, so a cached read never survives into a different profile.
        self._ocr_band_cache: dict[tuple, str | None] = {}
        self._adb: Adb | None = None
        self._capture_scrolls = 0     # read-scrolls the last _capture_current did; _scroll_to_top's ceiling
        # Exact geometry of every forward scroll since the last confirmed top.  A count alone
        # was enough while all read-scrolls had one fixed distance; auto-mode can now vary both
        # distance and lane per gesture, so undo must be based on what actually happened.
        self._capture_scroll_ledger: list[tuple[float, float]] = []
        self._profile_capture_limit = self.scroll_captures
        # --- doc 5.3's driver-owned index space, per profile ------------------------------
        # "Index space belongs to the driver. Selectability is policy." These three are the
        # driver's private table for the profile currently on screen, built by _capture_current
        # from the frames it just read, and they live and die with `_current_sigs`: reset at
        # the top of every capture, cleared on the deck-advance path, and cleared again once an
        # action (like/dislike) has moved the deck on. A STALE table is the failure this whole
        # design exists to prevent -- it would navigate by a previous profile's heart ordinals
        # and compare the opened sheet against a previous profile's crops.
        #
        # `_current_item_index` is the page (heart ordinals, page extents, per-frame evidence);
        # `_current_item_payload` is what the model was shown (the numbered crops, their
        # signatures, and `translation` from model item number to heart ordinal, which is the
        # authoritative copy whenever policy excluded anything selectable).
        # `_current_items_unavailable` is the one-sentence reason there is no payload, and
        # exactly one of it and the payload is ever set.
        #
        # `_current_item_anchor` is the FOURTH member of the same lifetime, added with doc 5.5's
        # bottom-up navigation: the last frame the index was built from, i.e. the frame
        # `ItemIndex.offsets[-1]` was measured on. `item_nav.navigate_to_item` measures ONE shift
        # against it to put the screen into the index's page space, which is what replaced the
        # rewind's `_scroll_to_top`. It is set and cleared with the other three, never separately
        # -- an anchor from one profile beside an index from another would be an arithmetic error
        # dressed as a measurement, and keeping them in one lifetime is what makes that
        # unreachable rather than merely unlikely.
        self._current_item_index = None
        self._current_item_payload = None
        self._current_item_anchor = None
        self._current_items_unavailable = (
            "no profile has been read yet, so this driver has enumerated nothing")
        self._touch = None            # touch transport: UhidTouch (genuine) or Adb (input fallback)
        self.touch_backend = app_cfg.get("touch_backend", "auto")   # auto | uhid | adb
        self._observe_ready = False   # True once open_session validated PIL/numpy + device
        # Rate-limit state for _note_observe_waiting. Both are re-anchored at the top of every
        # wait_for_decision call (one call = one profile's wait); initialised here as well so
        # the notice helper is safe to call from anywhere -- including tools and tests that
        # exercise it without going through a full wait. _observe_last_reason starts as None,
        # which no branch ever passes as a reason, so the FIRST notice of a session is always
        # emitted rather than suppressed by an accidental match.
        self._observe_last_notice = 0.0
        self._observe_last_reason: str | None = None
        # Stuck-screen watchdog state (see _OBSERVE_STUCK_S / _observe_stuck_budget).
        # `_observe_last_recognized` is "when did we last positively recognize what is on
        # screen", and like the notice anchors above it is re-anchored at the top of every
        # wait_for_decision call, so the budget is per-profile-wait and can never leak across
        # profiles. `_observe_stuck_probe_at` throttles the deck-ready probe the `no_change` fast
        # path needs, against `_observe_stuck_probe_interval_s` (anchored at
        # _OBSERVE_STUCK_CHECK_S). `_observe_stuck_budget_s` is the budget `_observe_stuck_bail`
        # actually compares elapsed time against -- a FRESH draw every time the watchdog is
        # (re)armed, never one draw reused across a whole run (see _observe_stuck_budget's
        # docstring). All four are initialised here so _observe_stuck_bail is safe to call from
        # anywhere, including tests and tools that exercise it without going through a full wait.
        self._observe_last_recognized = 0.0
        self._observe_stuck_probe_at = 0.0
        self._observe_stuck_budget_s = _OBSERVE_STUCK_S
        self._observe_stuck_probe_interval_s = _OBSERVE_STUCK_CHECK_S
        # The blocked-deck reason, once anything has proven one (a recognised paywall, or the
        # stuck-screen watchdog giving up). Memoized rather than recomputed because worker.py
        # asks blocked_reason() on every loop iteration and the answer costs a screencap plus a
        # template match plus an OCR; a deck that is blocked stays blocked until the operator
        # deals with it, so the first non-None answer is the answer.
        self._blocked_reason: str | None = None
        # Whether this session's one-shot "put the card at a confirmed scroll-top" pass has
        # been attempted yet -- see _ensure_session_top for why a session cannot assume the
        # previous one left the card where it found it.
        self._session_top_done = False
        self.debug_log = bool(app_cfg.get("debug_log", False))
        self.debug_dir = app_cfg.get("debug_dir", f"./data/{spec.app}_debug")
        self.halt_on_error = bool(app_cfg.get("halt_on_error", True))   # auto: STOP on unexpected (preserve logs)
        self._dbg = None              # HingeDebugLog (set in open_session when debug_log is on)
        # Assigned by Worker before open_session. A release verifier only accepts a debug
        # folder whose first production record binds it to this same Worker run id.
        self._debug_run_id: str | None = None
        # Process-local ownership for the per-device OBSERVE input lease. The same driver may
        # re-enter through current_profile() -> _scroll_to_top(), but a second thread or driver
        # instance must be refused before it can move the card underneath wait_for_decision().
        self._observe_input_lease_depth = 0
        self._observe_input_lease_thread: int | None = None
        self._observe_input_lease_fd: int | None = None

    def _require_vision(self) -> None:
        """Refuse to open a session if this app's declared glyph templates can't be matched.

        Every template this spec declares must actually load. Checking up front turns the
        worst failure mode in the driver — OpenCV quietly absent, so every match returns
        nothing — from "the run continues, blind, indefinitely" into "the run never starts".
        That is not hypothetical: a launcher shipped without the `hinge` extra once, so
        opencv was never installed and the vision path was dead for an unknown period.

        A spec declaring NO templates is fine here: it simply has no vision to lose, and
        _await_button refuses on its behalf if anything ever tries to aim at a control."""
        missing = [name for role, name in self.spec.templates.items()
                   if _load_template(name) is None]
        if missing:
            raise DriverClosed(
                f"{self.spec.app}: cannot load glyph template(s) {', '.join(sorted(missing))} "
                f"from {_ASSETS}. Vision-location would silently find nothing and every "
                f"action would be refused. OpenCV is the usual cause: "
                f"pip install -e '.[hinge]'"
            )

    # --- lifecycle ------------------------------------------------------
    def bind_debug_run(self, run_id: str) -> None:
        """Bind this driver's future debug session to one immutable Worker run id.

        This is metadata only: it neither opens ADB nor changes the device. Hinge's release
        verifier uses the resulting first debug action to reject a timestamp-named or unrelated
        ``actions.jsonl`` paired with another run's persisted labels.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("debug run id must be nonempty text")
        if self._debug_run_id is not None and self._debug_run_id != run_id:
            raise ValueError("a Hinge driver cannot be rebound to a different Worker run id")
        self._debug_run_id = run_id

    def open_session(self) -> None:
        # The last gate before this driver can touch a real phone with a real account on it.
        # Constructing an AndroidDriver is inert (no adb, no touch transport until below), so
        # the registry check lives HERE rather than in the factory — that keeps calibration
        # tooling able to build an uncalibrated driver, which is how one stops being
        # uncalibrated, while still making it impossible to open a session with placeholder
        # coordinates. supervisor.run() and HubState.start() check earlier and more loudly;
        # this one catches anything that reached a driver by another path.
        from .. import platforms
        reason = platforms.unavailable_reason(self.spec.app)
        if reason:
            raise DriverClosed(reason)

        import importlib.util
        if not (importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL")):
            raise DriverClosed(f"{self.spec.app} driver requires PIL and numpy. Install them via `pip install -e '.[ml]'` or `pip install pillow numpy`.")

        self._require_vision()

        self._adb = Adb(self.serial, adb_path=self.adb_path)
        ready = self._adb.devices()
        if not ready:
            raise DriverClosed(f"No ADB device connected for {self.spec.app}")
        if self.serial and self.serial not in ready:
            raise DriverClosed(f"{self.spec.app} device {self.serial} not connected (adb devices: {ready})")
        self._adb.screen_size()                       # cache geometry for clamping/coords
        if self.targeting_calibration is not None:
            self._targeting_runtime_frame_size = self._adb.screen_size()
            package_dump = self._adb.shell(f"dumpsys package {self.package}")
            match = re.search(r"(?m)^\s*versionName=(\S+)\s*$", package_dump)
            self._targeting_runtime_version_name = match.group(1) if match else None
            calibration = self.targeting_calibration
            if (self._targeting_runtime_version_name != calibration.hinge_version_name
                    or self._targeting_runtime_frame_size != calibration.frame_size_px):
                self.targeting_calibration = None
                self._targeting_calibration_unavailable = (
                    "the live app build/frame geometry does not exactly match schema-v3 "
                    f"calibration ({self._targeting_runtime_version_name!r}/"
                    f"{self._targeting_runtime_frame_size!r} != "
                    f"{calibration.hinge_version_name!r}/{calibration.frame_size_px!r})")
        self._adb.shell(f"monkey -p {self.package} -c android.intent.category.LAUNCHER 1")
        time.sleep(human_cooldown(1.5))               # let the app come to the foreground
        self._touch = self._make_touch()              # genuine UHID touches; adb input fallback
        self._observe_ready = True                    # only True once fully open (touch ready too)
        if self.observe_name_ocr and shutil.which("tesseract") is None:
            # A missing OCR binary cannot weaken the false-PASS guarantee: unresolved identity
            # stays waiting. It can reduce recall, however, because a real next card at scroll
            # top has generic filter-chip pixels and needs two matching different-name reads to
            # become a positive `new/new` decision. One line, once, so that conservative stall
            # is diagnosable rather than looking like a silent detector failure.
            print("profile-name OCR unavailable (tesseract not on PATH); identity safety still "
                  "holds, but a scroll-top card advance may remain unresolved.")
        if self.observe_touch_watch and not self._auto_session:
            # Gesture corroboration (layer 3), OBSERVE SESSIONS ONLY -- see self._auto_session
            # in __init__. Its whole job is to prove a HUMAN pressed something, which only
            # wait_for_decision ever asks; auto mode drives itself through like()/dislike()
            # and never calls wait_for_decision at all, so attaching here would hold a
            # persistent `adb shell getevent` subprocess open for an entire autonomous run
            # with no reader. That is not merely wasteful: ops/ANTI-BOT-RESEARCH.md's
            # 2026-08-10 (b) addendum scoped the accepted risk of holding that subprocess open
            # to observe sessions specifically, so quietly widening it to autonomous runs
            # would spend risk the log says was never accepted.
            #
            # Unlike the OCR probe above, this one DOES fail
            # loud: without a live touch stream, a card advance can only ever be corroborated
            # by identity + deck-ready (layers 1/2) -- narrower proof than what
            # observe_touch_watch=True was asking for. Silently running with it anyway would
            # be exactly the kind of quiet capability downgrade touch_backend's own "auto
            # REQUIRES UHID, never silently downgrades" contract exists to prevent (see
            # _make_touch above) -- so this raises the same way, naming the config key that
            # accepts the narrower proof deliberately.
            watcher = TouchWatcher(self.adb_path, self.serial, self._adb.screen_size())
            try:
                watcher.start()
            except TouchWatchUnavailable as exc:
                raise DriverClosed(
                    f"{self.spec.app}: gesture corroboration (observe_touch_watch) could not "
                    f"attach to the device's touch event stream ({exc}). Running observe mode "
                    f"without it would silently narrow PASS proof back to identity + deck-"
                    f"ready alone, with no evidence of an actual tap -- fine as a deliberate "
                    f"choice, not as a silent fallback. Fix the device/permissions (see "
                    f"ops/ANTI-BOT-RESEARCH.md), or set apps.{self.spec.app}.observe_touch_watch: "
                    f"false to accept the narrower proof on purpose."
                ) from exc
            self._touch_watcher = watcher
            print(f"{self.spec.app}: touch watcher attached on {watcher.device_path} "
                  f"({watcher.device_name!r}) — gesture corroboration is live for observe mode.")
        try:
            if self.debug_log:
                self._dbg = open_debug_log(self.debug_dir, run_id=self._debug_run_id)
                if self._dbg is not None and self._debug_run_id is not None:
                    # This must remain the first production record: release verification uses
                    # it to bind every later fact in this immutable-named debug folder to the
                    # Worker/store run id, without logging profile data or touching the phone.
                    self._dbg.action("observe_release_run_binding", run_id=self._debug_run_id,
                                     app=self.spec.app)
        except BaseException:
            # Anything after the watcher attaches must hand the subprocess back if it fails.
            # open_session() raising is NOT followed by a close() call -- the worker treats a
            # failed open as "this driver never opened" and moves on -- so without this the
            # `adb shell getevent` process started a few lines above outlives the session that
            # started it, invisibly, holding the device's touch stream until the whole hub
            # exits. close() stays idempotent, so the ordinary teardown path is unaffected.
            self.close()
            raise

    def close(self) -> None:
        if self._touch is not None and self._touch is not self._adb:
            try:
                self._touch.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the real outcome
                pass
        self._touch = None
        if self._touch_watcher is not None:
            try:
                self._touch_watcher.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the real outcome
                pass
            self._touch_watcher = None
        self._adb = None

    # --- helpers --------------------------------------------------------
    @property
    def adb(self) -> Adb:
        if self._adb is None:
            raise DriverClosed(f"{self.spec.app} session is not open")
        return self._adb

    @property
    def touch(self):
        if self._touch is None:
            raise DriverClosed(f"{self.spec.app} session is not open")
        return self._touch

    def _make_touch(self):
        """Touch transport. `auto` and `uhid` both REQUIRE the genuine UHID virtual
        touchscreen; only an explicit `adb` accepts the degraded one.

        `auto` used to mean "try UHID, quietly fall back to adb `input`". Both transports
        are humanized (curved paths, jitter, log-normal timing), but they are not
        equivalent: UHID delivers genuine kernel-level TOOL_TYPE_FINGER events with a
        variable pressure ramp at ~180Hz, while `adb shell input motionevent` cannot vary
        pressure at all — every contact reports the same synthetic value. A silent
        downgrade meant the run could look identical while emitting a materially less human
        touch signature, for an unknown length of time, on the account we care about.

        So `auto` now means "prefer UHID and fail loudly if it is unavailable". The
        degraded path is still reachable, but only by explicitly writing
        `touch_backend: adb`, which is an operator decision rather than an accident."""
        if self.touch_backend == "adb":
            print(f"{self.spec.app}: touch_backend='adb' — using the DEGRADED input transport "
                  f"(humanized paths, but constant pressure). Set 'auto' for genuine UHID touches.")
            return self._adb
        try:
            t = UhidTouch(self._adb)
            t.open()
            return t
        except (UhidUnavailable, AdbError) as exc:
            raise DriverClosed(
                f"{self.spec.app}: the genuine UHID touchscreen is unavailable ({exc}), and "
                f"touch_backend={self.touch_backend!r} will not silently downgrade to the "
                f"constant-pressure adb input transport. Fix the device/UHID path, or set "
                f"apps.{self.spec.app}.touch_backend: adb to accept the degraded transport "
                f"deliberately."
            ) from exc

    # --- tap choke point (forbidden-zone guard) -------------------------
    # EVERY gesture this driver issues goes through _tap() / _swipe() / _scroll(), never the
    # transport directly, so the no-go check cannot be bypassed by a new call site forgetting
    # about it. This originally covered taps ONLY, and the comment claimed more than the code
    # delivered: read-scrolls and scroll-to-top swipes went straight to the transport, so
    # their touch-down points were never zone-checked at all — on every profile, every run.
    #
    # A later audit found the check itself could be evaded even where it WAS wired up: it
    # computed the zone-check fraction from the RAW, UNCLAMPED (x, y), but both real
    # transports CLAMP the coordinate they actually deliver AFTER this check runs (uhid.py's
    # _report, Adb._clamp — see clamp_xy in adb.py, which both now share). forbidden_zones
    # rects are constrained to 0.0..1.0, so any coordinate whose fraction fell OUTSIDE that
    # range matched no zone and passed cleanly — then the transport clamped it onto the
    # screen edge, which can be INSIDE a zone. The point that was CHECKED was not the point
    # that got DELIVERED. See OutOfRangeTapError for the demonstrated exploit.
    _TAP_ZONE_MARGIN_PX = tap_jitter_margin_px()
    # The zone-check margin _tap() applies (see its call below): the worst-case single-axis
    # drift EITHER real touch transport's tap can add to the delivered touch-down AFTER this
    # check runs, so a zone can't be evaded by jitter the checked point never accounted for.
    # UhidTouch.tap() calls human_motion.plan_tap() at plan_tap's own default jitter_px (it
    # overrides jitter_px for swipes only, not taps), so tap_jitter_margin_px()'s default
    # argument is the correct bound for the transport this project requires by default.
    # Adb.tap()'s simpler transport (its `input tap` fallback, reachable only via an explicit
    # touch_backend: adb) jitters with a plain uniform(-jitter_px, jitter_px) of at most its
    # own jitter_px (default 2.0px) — comfortably smaller — so this one constant safely
    # covers both without needing to know which backend is actually wired up.

    def _assert_tap_allowed(self, x: int, y: int, margin_px: float = 0.0) -> None:
        """The choke point itself. Two checks, in this order:

        1. RANGE — (x, y) must already resolve to a fraction inside 0..1 of the live screen,
           checked on the RAW value before any clamping. An out-of-range value is always a
           configuration or logic error (never a legitimate touch — see OutOfRangeTapError),
           and letting the transport silently clamp it instead of refusing it here is exactly
           the gap this whole check exists to close, so it fires even for a spec that
           declares no forbidden_zones at all.

        2. ZONE — clamp (x, y) the SAME way the real transports do (clamp_xy — see the class
           comment above), then require that point, widened by `margin_px` on every side, to
           clear every forbidden zone. `margin_px` covers jitter the transport adds AFTER
           this check returns (see _tap()/_TAP_ZONE_MARGIN_PX above); callers whose delivered
           point has no such jitter (_swipe(), _scroll() below) pass 0.
        """
        w, h = self.adb.screen_size()
        fx_raw = x / w if w else 0.0
        fy_raw = y / h if h else 0.0
        if not (0.0 <= fx_raw <= 1.0 and 0.0 <= fy_raw <= 1.0):
            raise OutOfRangeTapError(
                f"{self.spec.app}: refusing a touch at raw pixel ({x}, {y}) = fraction "
                f"({fx_raw:.3f}, {fy_raw:.3f}) of the {w}x{h} screen — outside 0..1. This is "
                f"always a configuration or logic error: check apps.{self.spec.app}.coords, "
                f"read_scroll_frac, or any other *_frac knob for this app for a typo, or a "
                f"pixel value written where a fraction was expected. Refusing rather than "
                f"letting the transport silently clamp it onto a screen edge, which can land "
                f"inside a forbidden zone undetected.")
        zones = getattr(self.spec, "forbidden_zones", ())
        if not zones:
            return
        cx, cy = clamp_xy(x, y, w, h)      # the point that will ACTUALLY be delivered
        mfx = margin_px / w if w else 0.0
        mfy = margin_px / h if h else 0.0
        fx, fy = (cx / w if w else 0.0), (cy / h if h else 0.0)
        for zone in zones:
            x0, y0, x1, y1 = zone
            if x0 - mfx <= fx <= x1 + mfx and y0 - mfy <= fy <= y1 + mfy:
                margin_note = f" (+/-{margin_px:.1f}px jitter envelope)" if margin_px else ""
                raise ForbiddenTapError(
                    f"refused a tap at ({cx}, {cy}) = ({fx:.3f}, {fy:.3f}) of the "
                    f"screen{margin_note}: it lands inside, or could jitter into, "
                    f"{self.spec.app}'s forbidden zone {zone}, which guards a paid control. "
                    f"Refusing rather than risking a paid action.")

    def _tap(self, x, y) -> None:
        x, y = int(x), int(y)
        self._assert_tap_allowed(x, y, margin_px=self._TAP_ZONE_MARGIN_PX)
        self.touch.tap(x, y)

    def _swipe(self, x1, y1, x2, y2) -> None:
        """Every explicit drag goes through here, for the same reason every tap goes
        through _tap(). Only the START point is zone-checked (margin_px=0: plan_swipe's
        first sample is pinned exactly to (x1, y1) with no jitter, so there is no drift to
        cover here — see plan_swipe in human_motion.py): the touch-down claims the gesture,
        so a drag that merely travels over a control does not press it.

        That "touch-down claims the gesture" model is correct for STANDARD Android touch
        dispatch (a MotionEvent stream goes to whichever View captured ACTION_DOWN,
        regardless of where ACTION_MOVE/ACTION_UP later land) — but it is UNVERIFIED against
        Bumble's actual UI, and the stakes of being wrong are real money. _scroll_to_top's
        undo-swipes travel DOWNWARD (returning the card to the top) and routinely END with
        the finger sitting over Bumble's SuperSwipe location. If Bumble's real button turns
        out to react to where a drag ENDS — a custom touch listener, a drop target, anything
        other than plain View dispatch — this check as written would not catch it. See
        ops/RUNBOOK.md's Bumble calibration checklist: before Bumble is ever run unattended,
        deliberately drag over the SuperSwipe control on a disposable profile and confirm
        nothing is purchased."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self._assert_tap_allowed(x1, y1)
        self.touch.swipe(x1, y1, x2, y2)

    def _scroll(self, frac: float, x_frac: float = 0.5, *, reverse: bool = False) -> None:
        """touch.scroll_up() with the forbidden-zone guard every other gesture gets.

        scroll_up computes its geometry INSIDE the transport, so the driver cannot see
        where the touch-down lands without mirroring that math — which _scroll_to_top
        already does for its undo-swipes. Both transports use the same start point:
        (scroll_x(w, x_frac), h * (0.5 + frac/2)).

        This guard was missing, and the margin is thinner than it looks: at the default
        read_scroll_frac=0.55 the touch-down sits at y=0.775, just 2.5% of the screen above
        Bumble's SuperSwipe zone (which starts at 0.80). Raising read_scroll_frac to 0.65 —
        an ordinary config-level calibration tweak — would put EVERY read-scroll's
        touch-down inside the paid control's territory, on every profile, unguarded.

        scroll_x jitters the column by +/-SCROLL_X_JITTER_PX so repeated scrolls aren't
        pixel-identical, so both extremes are checked rather than the nominal centre — a
        guard that only the average case passes is not a guard.

        REVERSE (`reverse=True`) IS THE SAME STROKE PLAYED BACKWARDS, AND IT IS A DIRECTION
        OF THIS METHOD RATHER THAN A SECOND GESTURE PATH ON PURPOSE (ops/OPENER-REDESIGN.md
        5.5, bottom-up navigation). Doc 5.5's navigation used to be "scroll to top, then walk
        forward", i.e. every reverse travel went through `_scroll_to_top`'s ledger REPLAY.
        Walking up under continuous shift tracking needs one measured reverse step at a time,
        and the owner rule ("best humanized interaction or FAIL LOUDLY; no silent fallback to
        a degraded transport") means it must not reach for `self.touch` on its own. So the
        guard, the column jitter and the transport are all the forward path's, and the only
        difference is which end the finger goes down on:

          forward  touch-down at h*(0.5 + frac/2), release at h*(0.5 - frac/2)  (content up)
          reverse  touch-down at h*(0.5 - frac/2), release at h*(0.5 + frac/2)  (content down)

        BOTH END ROWS ARE ZONE-CHECKED FOR THE REVERSE STROKE, which is stricter than the
        forward one (`_swipe`, and `_decide_by_card_swipe`'s docstring, check only the START —
        "a drag that ends over a button does not press it"). Two reasons to be stricter here
        rather than consistent: the reverse stroke's touch-down and release swap, so "the
        start" is a different row than every other read-scroll in this driver puts a finger on,
        and the enumeration/navigation fracs are small (0.10..0.16 measured, i.e. both rows
        inside 0.42..0.58 of the screen) so nothing legal is refused by asking for both. The
        cost of the extra check is two array comparisons; the cost of getting it wrong is the
        owner's money."""
        w, h = self.adb.screen_size()
        y_low = int(h * (0.5 + frac / 2))     # the FORWARD stroke's touch-down, low on screen
        y_high = int(h * (0.5 - frac / 2))    # ...and its release, high on screen
        nominal = int(w * x_frac)
        for x in (nominal - SCROLL_X_JITTER_PX, nominal + SCROLL_X_JITTER_PX):
            self._assert_tap_allowed(x, y_low)
            if reverse:
                self._assert_tap_allowed(x, y_high)
        if not reverse:
            self.touch.scroll_up(frac, x_frac)
            return
        # scroll_x is the SHARED column jitter both transports' scroll_up() applies (HINGE-04),
        # called here rather than re-derived so a reverse read-scroll is not the one gesture in
        # this driver that lands on a pixel-identical column every time. `self._swipe`, not
        # `self.touch.swipe`: it re-asserts the delivered start point, which is the same
        # chokepoint every other drag in this file goes through.
        x = scroll_x(w, x_frac)
        self._swipe(x, y_high, x, y_low)

    def _tap_frac(self, frac) -> None:
        w, h = self.adb.screen_size()
        self._tap(frac[0] * w, frac[1] * h)

    def _decide_by_card_swipe(self, decision: str) -> None:
        """Deliver a like/pass by dragging the card sideways instead of tapping a control.

        Bumble places its paid SuperSwipe BETWEEN Pass and Like at the bottom of the card, so
        a placeholder or drifted coordinate can land on it. Whether that lands harmlessly or
        expensively is NOT governed by a confirmation step the way Hinge's Rose is — measured
        live on the device 2026-08-10, a SuperSwipe has two distinct outcomes depending on the
        account's SuperSwipe balance: with a non-zero balance (5, at measurement time) it is
        spent SILENTLY, no modal at all; only with a zero balance does a purchase/confirmation
        sheet appear first (see BUMBLE_SPEC's comment block and _handle_rose_upsell for the
        full two-state writeup). A drag begins in the middle of the card and cannot press a
        button it merely travels over, so the paid control is unreachable by construction
        rather than by careful aiming — this is the mechanism that actually protects the
        balance>0 case, since there is nothing downstream of the tap to catch a mistake there.

        The drag goes through the same touch transport as everything else, so it inherits
        the humanized kinematics (Fitts-law duration, curved path, tremor, pressure ramp).
        Only the START point is zone-checked: a drag that ends over a button does not
        press it, since the press was already claimed by whatever was under the start."""
        w, h = self.adb.screen_size()
        # self.coords, not self.spec.coords: config.yaml's apps.<app>.coords is layered over
        # the spec, so these can be calibrated on the device without editing code — which is
        # the whole workflow for taking Bumble from placeholder to calibrated.
        start = self.coords["swipe_start"]
        end = self.coords["swipe_like_end" if decision == "like" else "swipe_pass_end"]
        x1, y1 = int(start[0] * w), int(start[1] * h)
        x2, y2 = int(end[0] * w), int(end[1] * h)
        self._swipe(x1, y1, x2, y2)

    def _template(self, role: str):
        """Load the template PNG for a logical UI role (AndroidAppSpec.templates), or None if
        this app's spec declares no template for that role at all — _match_glyph's own
        None-template guard then makes every caller here treat it as a clean 'not found'."""
        name = self.spec.templates.get(role)
        return _load_template(name) if name else None

    def _locate_button(self, which: str):
        """Screencap and template-match the like-heart ('like' -> topmost right glyph) or the
        floating pass-X ('pass' -> left glyph). Returns (x, y) or None when not visible, OR
        when this app's spec has no template for that role at all — vision-location is then
        skipped WITHOUT even capturing a frame; _await_button falls back to the fixed coord."""
        role = "like" if which == "like" else "pass"
        if role not in self.spec.templates:
            return None
        # "like" gets its own, higher-margin threshold (_LIKE_MATCH_THRESHOLD) AND a content_band
        # y_band restriction (see _match_glyph's own comment for why: two Hinge nav-bar false
        # positives at y=2258, structurally outside content_band). "pass" keeps _match_glyph's
        # own 0.6 default and no y_band, unchanged -- the floating pass-X can legitimately sit
        # low on the screen, outside content_band, so restricting it was never verified as safe.
        kwargs = ({"threshold": _LIKE_MATCH_THRESHOLD, "y_band": self.content_band}
                 if role == "like" else {})
        centers = _match_glyph(self._screencap(), self._template(role),
                               side="right" if which == "like" else "left", **kwargs)
        return centers[0] if centers else None

    def _await_button(self, which: str, tries: int = 5, *, should_stop=None):
        """Locate a button by vision, retrying through short settle waits.

        Retries because Hinge fades the floating like/pass buttons out DURING a scroll and
        back in once it settles, so a tap fired immediately after a scroll-read can miss a
        button that is genuinely there.

        RAISES rather than falling back to the calibrated fixed coordinate. See
        UnlocatedControlError for why guessing is worse than stopping. The fixed coords stay
        in the spec as a calibration reference and as the anchor for tooling, but nothing
        taps them on this path."""
        if should_stop is None:
            pt = _retry_until(lambda: self._locate_button(which), tries, 0.4)
        else:
            pt = None
            for _ in range(max(1, tries)):
                self._raise_if_action_cancelled(should_stop, boundary="legacy button lookup")
                pt = self._locate_button(which)
                if pt is not None:
                    break
                if not self._interruptible_sleep(human_delay(0.4), should_stop):
                    self._raise_if_action_cancelled(should_stop,
                                                    boundary="legacy button lookup")
        if pt is not None:
            return pt

        role = "like" if which == "like" else "pass"
        if role not in self.spec.templates:
            why = (f"{self.spec.app} declares no '{role}' glyph template, so this control "
                   f"cannot be vision-located at all")
        elif _load_template(self.spec.templates[role]) is None:
            why = (f"the '{role}' template {self.spec.templates[role]!r} could not be loaded "
                   f"— OpenCV is very likely missing (install the extra: "
                   f"pip install -e '.[hinge]')")
        else:
            why = (f"the '{role}' glyph was not found on screen after {tries} attempts — the "
                   f"app UI may have changed, or the screen is not the swipe deck")
        # .get(), not [] — a spec is allowed to omit these calibration-reference coords, and a
        # KeyError raised while BUILDING an error message would replace the real diagnosis
        # with a confusing one.
        ref = self.coords.get("like_heart" if which == "like" else "pass_x")
        instead = f" Not falling back to the fixed coordinate {ref} —" if ref else " —"
        raise UnlocatedControlError(
            f"refusing to {which}: {why}.{instead} a blind tap at a stale point can hit a "
            f"paid or irreversible control.")

    def _await_sheet_open(self, tries: int = 5) -> ComposerSurface | None:
        """Confirm the comment / "Send Like" sheet really opened, before tapping into it.

        comment_box and send_like are FIXED coordinates rather than vision-located, on the
        grounds that the sheet's layout is consistent. That is only true WHILE THE SHEET IS
        UP. If the heart tap missed, or the sheet was slow, or the app changed, those two
        taps instead land on the profile card underneath — and on Hinge the send_like point
        sits in the middle of the card, where per-photo and per-prompt like buttons live. A
        missed heart tap could therefore like the wrong item rather than doing nothing.

        So the fixed taps are gated on the sheet being visibly present. This is the same
        `confirm` glyph _verify_like_landed already uses to decide the sheet has CLOSED;
        checking it on the way in as well costs one screencap.

        (When a device is available, consider tapping the matched glyph position instead of
        the fixed send_like coordinate — strictly more robust to drift. Not done blind:
        whether this template depicts the button itself or a label beside it needs eyes on
        the real sheet, and the fixed coordinate is at least live-validated.)"""
        if "confirm" not in self.spec.templates:
            raise UnlocatedControlError(
                f"{self.spec.app} uses the comment_sheet like flow but declares no 'confirm' "
                f"template, so there is no way to tell the sheet opened. Refusing to tap the "
                f"fixed comment box / send coordinates on an unverified screen.")
        found = _retry_until(
            lambda: _match_glyph(self._screencap(), self._template("confirm"),
                                 side="any", threshold=0.6) or None,
            tries, 0.4)
        if not found:
            raise UnlocatedControlError(
                f"refusing to continue the like: the '{self.spec.app}' comment sheet never "
                f"appeared after tapping the heart ({tries} attempts). Not tapping the fixed "
                f"comment box {self.coords.get('comment_box')} / send {self.coords.get('send_like')} "
                f"coordinates — with no sheet up they land on the profile card underneath, "
                f"where they can hit a per-item like.")
        # Generic comment-sheet apps retain their measured fractional controls.  Hinge overrides
        # this method and returns vision-located inline-composer geometry instead.
        return None

    # --- capture (blank-frame guard) -----------------------------------
    def _screencap(self, *, on_blank: str = "raise") -> bytes | None:
        """Every decision-making capture goes through here, never `self.adb.screencap()`.

        `adb exec-out screencap` on a device that is asleep or sitting on the
        keyguard SUCCEEDS and returns a solid-black PNG — it does not error. Two
        identical black frames read as "nothing changed", so a run would stall
        blind; and a good frame followed by a black one reads as a BIG delta, which
        wait_for_decision would mis-score as a card advance (a phantom "pass").

        `on_blank` picks what a persistently-blank screen means to THIS caller:

        - "raise" (default) — for paths that are about to ACT or verify an action.
          Tapping blind on a burner is never an acceptable degradation, so this
          raises regardless of `halt_on_error`. That is a deliberate departure from
          the `halt_on_error` gating used by _snap/_verify_progress: those cover
          *unexpected* errors the owner may opt out of halting on, whereas a blank
          screen is a physical impossibility — there is nothing to look at.

        - "none" — returns None for PASSIVE polling loops (wait_for_decision,
          _await_like_resolved). Those wait on a HUMAN with no timeout and send no
          touches, so nothing keeps the screen alive; a screen-off there is an
          ordinary, recoverable event. They skip the iteration and keep watching,
          which is both non-fatal AND avoids the phantom-pass above.

        Retried once after a settle so a single odd frame mid-transition can't halt
        a healthy run; only a frame that is STILL blank counts.

        The two deliberate exceptions that keep calling `self.adb.screencap()` raw:
        _dbg_action (best-effort debug capture, already swallows failures) and
        snapshot_failure (must record whatever is on screen — including black — and
        must never raise while handling another error).
        """
        def _validated(nonblank: bytes) -> bytes:
            calibration = self.targeting_calibration
            if calibration is not None:
                try:
                    from io import BytesIO

                    from PIL import Image
                    with Image.open(BytesIO(nonblank)) as image:
                        frame_size = image.size
                except Exception as exc:  # noqa: BLE001 -- an undecodable safety frame must stop
                    raise HingeActionError(
                        f"targeted frame geometry could not be decoded for schema-v3 "
                        f"calibration ({exc})") from exc
                if frame_size != calibration.frame_size_px:
                    raise HingeActionError(
                        f"live frame {frame_size!r} does not match schema-v3 targeting "
                        f"calibration {calibration.frame_size_px!r}; refusing to use calibrated "
                        "geometry")
            return nonblank

        frame = self.adb.screencap()
        if not _is_blank_frame(frame):
            return _validated(frame)
        time.sleep(human_delay(0.4))
        frame = self.adb.screencap()
        if not _is_blank_frame(frame):
            return _validated(frame)
        if on_blank == "none":
            return None
        raise HingeActionError(
            f"screen is blank ({self._blank_reason()}) — every capture is solid black, so "
            "the run would act blind; wake and unlock the device before continuing")

    def _blank_reason(self) -> str:
        """Best-effort one-line diagnosis for the blank-frame error. Never raises:
        it only ever decorates an error message that is already being raised."""
        try:
            wake = self.adb.shell("dumpsys power | grep -m1 -o 'mWakefulness=[A-Za-z]*'").strip()
            lock = self.adb.shell("dumpsys trust | grep -m1 -o 'deviceLocked=[01]'").strip()
        except Exception:  # noqa: BLE001 — diagnosis is optional, the raise is not
            return "device state unreadable"
        parts = [p for p in (wake, lock) if p]
        return ", ".join(parts) if parts else "device state unknown"

    def _await_live_frame(self, deadline, should_stop) -> bytes | None:
        """Block until a NON-blank frame is available, for passive observe loops.

        Observe waits on a human with no timeout and sends no touches, so nothing
        keeps the screen alive — it going dark mid-wait is ordinary and recoverable,
        not a run-ending error. We simply keep watching until the owner comes back
        and wakes it. Returns None if `should_stop` fires or `deadline` passes,
        which the callers already handle as "no decision".
        """
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            frame = self._screencap(on_blank="none")
            if frame is not None:
                return frame
            time.sleep(_OBSERVE_POLL_S)
        return None

    # --- debug logging + autonomous-safety (halt on unexpected) --------
    def _snap(self):
        """Screencap for debug/verify, or None when neither is active (skips the overhead)."""
        if self._dbg is None and not self.halt_on_error:
            return None
        for attempt in (1, 2):                         # retry once: a transient screencap blip must
            try:                                       # not silently disable the _verify check below
                return self._screencap()
            except DriverClosed:
                raise                                  # device truly gone -> let it propagate
            except HingeActionError:
                # Blank screen: _screencap already retried and already says exactly what
                # is wrong ("screen is blank (mWakefulness=..., deviceLocked=...)").
                # Re-raise rather than let the generic handler below bury that diagnosis
                # under "screencap failed twice in a row" — or, worse, swallow it to None
                # when halt_on_error is False and hand a caller a missing "before" frame.
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    if self.halt_on_error:
                        # Returning None here would silently defeat halt_on_error:
                        # _verify_progress/_verify_like_landed treat a missing "before"
                        # frame as "skip verification" / an automatic pass — exactly the
                        # failure mode this safety net exists to catch. Two consecutive
                        # screencap failures means the device/link is wedged, not a blip.
                        raise HingeActionError(
                            f"screencap failed twice in a row ({type(exc).__name__}: {exc}); "
                            "halting rather than acting/verifying blind") from exc
                    return None
                time.sleep(human_delay(0.3))

    def _dbg_action(self, name: str, before, **fields) -> None:
        if self._dbg is None:
            return
        try:
            after = self.adb.screencap()   # RAW: best-effort debug capture, never gates the run
        except Exception:  # noqa: BLE001
            after = None
        self._dbg.action(name, before=before, after=after, **fields)

    def snapshot_failure(self, exc: BaseException) -> None:
        """Worker hook: snapshot the on-screen state of an unexpected error into the debug log
        (so the failure is reconstructable) before the worker halts the run."""
        snapshot_failure_frame(self._dbg, exc, self.adb.screencap)

    def _verify_progress(self, before, action: str) -> None:
        """After an autonomous action the screen MUST change (a new card / a confirmation). If it
        doesn't, even after a short settle, something is wrong (missed tap, unknown modal, stuck
        deck) — raise so the worker HALTS and the debug logs are preserved instead of being
        rotated away by continued blind swiping. Only active when halt_on_error is set (in which
        case `before`, sourced from _snap(), is never None — _snap() raises instead)."""
        if not self.halt_on_error:
            return
        if _retry_until(lambda: self._changed(before, self._screencap()), 2, 0.6):
            return
        raise HingeActionError(f"{action} did not change the screen (stuck or unexpected state)")

    def _handle_rose_upsell(self, tries: int = 2) -> bool:
        """Dismiss a paid-upgrade interstitial — NEVER the paid option itself. Named for
        Hinge's "Send a Rose instead?" modal (which intercepts "Send Like" WHENEVER a Rose is
        available, since free Roses are granted periodically) but used generically by every
        like flow: we never spend a Rose, a SuperSwipe, or any other paid upsell (OWNER RULE —
        never automated, always manual). No-op when the modal isn't shown (or this app's spec
        declares no "upsell_dismiss" template at all — _template then returns None and
        _match_glyph's None-template guard reports no hits). Returns True if it dismissed a
        modal.

        Detection is ALWAYS the same: the "upsell_dismiss" template must match before this
        method taps anything at all. What happens after a match differs by app, selected by
        AndroidAppSpec.upsell_dismiss_zone:

          * zone is None (Hinge) — the template names a real, single, well-defined button
            ("Send Like anyway"); tap exactly its matched location. There is no coordinate or
            template anywhere in this driver for the PAID button beside/above it, so there is
            no code path that could tap one even by accident.

          * zone is set (Bumble, once calibrated) — there is no discrete dismiss BUTTON to
            match. Measured live on the device 2026-08-10: Bumble's SuperSwipe purchase sheet
            dismisses on a tap ANYWHERE in the dimmed area above it (x 0.000-1.000,
            y 0.000-0.394 — see BUMBLE_SPEC's comment block for the full measurement and the
            narrower, jitter-safe band actually declared). The template here identifies the
            SHEET (e.g. its heading), not a tap target; the actual dismiss tap goes through
            _dismiss_via_zone, which picks a FRESH random point inside the declared zone on
            every attempt, verifies the sheet actually cleared, and HALTS (PaidUpsellStuckError)
            rather than retrying forever if it doesn't.

        Ground truth from the owner, measured live on the device 2026-08-10: this whole
        method — and the confirmation sheet it dismisses — is NOT a general SuperSwipe safety
        net. It only ever runs AFTER a decide gesture has already landed, and Bumble's
        SuperSwipe has two distinct outcomes depending on the account's balance:

          * balance == 0 — the purchase sheet described above appears. No money has been
            spent yet; this method's job is to close that sheet without ever touching its
            "Get 30 SuperSwipes for $39.99" CTA (measured at x 0.049-0.950, y 0.899-0.951).

          * balance > 0 (5, at measurement time) — the SuperSwipe is spent SILENTLY. No
            sheet, nothing here to dismiss, nothing at all downstream to catch it. The real
            protections for THIS case are upstream of this method entirely: never aiming at
            the SuperSwipe control (decide_gesture="card_swipe" + forbidden_zones) and never
            firing a decide gesture without positively confirming the deck first
            (_require_deck_confirmed / UnconfirmedScreenError).

        Uses on_blank="none" rather than raising, deliberately: this runs AFTER the
        confirming tap has already landed. A like sent server-side but aborted here
        would never reach worker.py's store.record_decision(), which assumes a raising
        like() "never landed" — so we would silently lose the record of a real like.
        A blank screen here degrades to "no modal seen"; the flow's own verify step is
        the one that still gets to fail loudly.
        """
        template = self._template("upsell_dismiss")

        def _probe():
            frame = self._screencap(on_blank="none")
            if frame is None:
                return []                     # can't see -> assume no modal (see docstring)
            return _match_glyph(frame, template, side="any", threshold=0.6)

        hits = _retry_until(_probe, tries, 0.5)   # modal animates in (only when an upsell is offered)
        if not hits:
            return False
        if self.spec.upsell_dismiss_zone is not None:
            self._dismiss_via_zone()          # random point in the safe band, verified, bounded
        else:
            self._tap(*hits[0])               # dismiss control — NEVER the paid button above/beside it
        return True

    def _dismiss_via_zone(self) -> None:
        """Dismiss an ALREADY-DETECTED paid-upgrade sheet by tapping a fresh random point
        inside spec.upsell_dismiss_zone, re-verifying after each attempt, and halting rather
        than retrying forever. Only ever called from _handle_rose_upsell, after ITS OWN
        template match already confirmed the sheet is up — this method never probes for the
        modal itself, so it structurally cannot fire against the ordinary deck.

        Why a FRESH random point every attempt, not the zone's centre or a fixed offset: a
        repeated exact coordinate is exactly the kind of bot signature the owner's
        no-fixed-constants rule forbids (see memory/auto-mode-uncapped-volume.md). Why a
        bounded retry-then-halt instead of looping until success: a dismiss tap that keeps
        landing just outside the sheet (drift, a device rotation, an app update that moved
        the layout) must not turn into an indefinite sequence of blind taps near a screen
        whose bottom third can be a purchase button (MEASURED 2026-08-10: Bumble's SuperSwipe
        CTA sits at y 0.899-0.951 — see BUMBLE_SPEC). Stopping and preserving the on-screen
        state for debugging is always safer than one more guess.
        """
        x0, y0, x1, y1 = self.spec.upsell_dismiss_zone
        w, h = self.adb.screen_size()
        template = self._template("upsell_dismiss")
        for _attempt in range(_UPSELL_DISMISS_MAX_ATTEMPTS):
            fx = random.uniform(x0, x1)       # fresh draw every attempt -- never a fixed point
            fy = random.uniform(y0, y1)
            self._tap(fx * w, fy * h)         # still runs through _assert_tap_allowed's guard
            time.sleep(human_cooldown(0.6))   # let the dismiss animation resolve
            frame = self._screencap(on_blank="none")
            if frame is not None and not _match_glyph(frame, template, side="any", threshold=0.6):
                return                        # sheet glyph is gone -> dismissed
        raise PaidUpsellStuckError(
            f"{self.spec.app}: a paid-upgrade sheet is still on screen after "
            f"{_UPSELL_DISMISS_MAX_ATTEMPTS} dismiss attempts inside its safe zone "
            f"{self.spec.upsell_dismiss_zone} — halting rather than tapping again blindly. "
            f"Repeated taps near a modal like this one are how a purchase gets confirmed.")

    def _require_deck_confirmed(self) -> None:
        """Refuse to issue a decide gesture (like/pass) unless the swipe deck is positively
        confirmed on screen. See UnconfirmedScreenError for the full reasoning: in short,
        forbidden_zones is screen-agnostic and cannot by itself tell a safe deck coordinate
        from the identical point on a paid-upgrade sheet, so this asks a different question
        first — "is this actually the deck" — using the same glyph-based readiness check
        _observe_deck_ready already uses passively for human-driven observe mode.

        A no-op when this app's spec declares no 'like' or no 'pass' template: there is
        nothing to prove readiness against yet (see UnconfirmedScreenError's docstring for
        why that's not a loophole — such a spec cannot be calibrated or run unattended
        anyway). Called from _deliver_decision, the single chokepoint both decide gestures
        (tap and card_swipe) go through, so this covers Bumble's card-drag path — which,
        unlike Hinge's vision-located tap, had NO perceptual check at all before this."""
        if "like" not in self.spec.templates or "pass" not in self.spec.templates:
            return
        frame = self._screencap()
        if self._observe_deck_ready(frame):
            return
        raise UnconfirmedScreenError(
            f"{self.spec.app}: refusing to decide — the swipe deck (like heart + pass X) is "
            f"not positively confirmed on screen. Something else may be up (a paid-upgrade "
            f"sheet, an ad, a dialog); firing a decide gesture at deck coordinates against an "
            f"unconfirmed screen is exactly how a like/pass tap lands on a different control "
            f"instead.")

    def _changed(self, a: bytes, b: bytes) -> bool:
        top, bot = _split_diff(a, b)
        return top >= self.change_threshold or bot >= self.change_threshold

    def _auto_behavior_policy(self):
        """The optional auto-mode behavior policy, or None in observe/legacy callers.

        Worker wiring deliberately owns whether this attribute exists.  Keeping the lookup
        soft preserves the standalone driver and observe paths, including calibration tools
        that construct AndroidDriver directly.
        """
        return getattr(self, "_auto_policy", None)

    def set_auto_session_policy(self, policy) -> None:
        """Attach Worker-owned behavior state for this autonomous session.

        This redesign is calibrated for Hinge only. AndroidDriver also backs an
        experimental Bumble path whose paid-control zones and reading geometry are
        different, so it must retain its legacy plan. Observe mode never calls this
        method and likewise retains calibrated legacy behavior.

        Also the ONLY signal this driver has that it is running an AUTONOMOUS session at
        all -- see self._auto_session's docstring in __init__ and open_session()'s touch-
        watcher gate. Worker._auto_loop calls this, unconditionally for any driver that
        defines it, before open_session(); Worker._observe_loop never does. So
        `self._auto_session = True` here is UNCONDITIONAL (unlike self._auto_policy right
        below, which stays app-gated to Hinge's calibrated behavior only) -- it just
        records which loop is driving this session, independent of which app it is or
        whether that app's policy object ends up used for anything.
        """
        self._auto_session = True
        self._auto_policy = policy if self.spec.app == "hinge" else None

    def set_opener_enabled(self, enabled: bool) -> None:
        """Tell this driver whether an opener will actually be requested for a like this
        session (audit fix, "BUG 2", 2026-08-12).

        BOTH worker loops call this once, unconditionally for any driver that defines it,
        before open_session() -- _auto_loop right alongside set_auto_session_policy, and
        _observe_loop on its own (observe has no session policy to sit beside, and as of doc
        5.9's inversion it enumerates too, so this hook is the ONLY thing standing between a
        no-opener observe session and a ~40-frame read per card it would never use).
        `enabled` is
        `opener_service is not None and not opener_service.disabled`, i.e. true exactly when a
        live, administratively-enabled OpenerService exists to consume a numbered item list.
        `opener.enabled: false` in config constructs OpenerService(client=None, ...), which is
        `disabled` from construction (see OpenerService.__init__) -- a deliberate "openers do
        not exist this run" choice, not a per-call failure, so this is checked once at session
        start rather than per profile (mid-run exhaustion is a different, already-handled case:
        it also sets stop_requested, which _auto_loop checks independently and which ends the
        run before another profile is ever read).

        Read by _item_enumeration_blocker: with no consumer for a numbered item list, item
        enumeration must not run at all, not merely "must not stop the run over its absence" --
        see that method's docstring for the reasoning.
        """
        self._openers_enabled = bool(enabled)

    def observe_release_fact(self, fact: str) -> None:
        """Append one thread-safe, transport-free production-OBSERVE release fact.

        The Worker invokes this only after its real hub publication / verifier result or its
        own observed blocked state. It is deliberately a fact-name-only debug record: no opener
        text, profile identity, screenshot, or target index is needed to prove the control-flow
        transition, and the debug logger serializes the provider and device threads.
        """
        if fact not in {"hub_pre_tap_published", "post_tap_item_verified",
                        "refusal_or_paywall_logged"}:
            raise ValueError(f"unknown observe release fact {fact!r}")
        # This is intentionally *not* _dbg_action: that helper requires a ``before``
        # frame and takes a fresh ``after`` screencap.  Release facts originate on the
        # Worker/provider side and are control-flow evidence, not a device action, so a
        # screenshot here would both violate this method's transport-free contract and
        # make a missing positional ``before`` argument fail silently in Worker error
        # handling.  DebugLog.action is itself locked, so it is safe to append directly.
        if self._dbg is not None:
            self._dbg.action(f"observe_release_{fact}")

    def _sample_read_step(self, depth: int, complexity_hint: float | None):
        """Return one coherent ``(dwell, distance, lane)`` read step.

        The production policy draws all three dimensions together.  The older
        split sampler hooks remain as a defensive compatibility fallback for
        calibration tools and small test doubles.
        """
        policy = self._auto_behavior_policy()
        planner = getattr(policy, "read_step", None) if policy is not None else None
        if callable(planner):
            try:
                step = planner(depth, captured_frames=depth + 1)
                dwell = float(step.dwell_s)
                frac = float(step.fraction)
                x_frac = float(step.x_frac)
                if (math.isfinite(dwell) and dwell >= 0.0
                        and math.isfinite(frac) and _READ_SCROLL_FRAC_MIN <= frac <= _READ_SCROLL_FRAC_MAX
                        and math.isfinite(x_frac) and 0.10 <= x_frac <= 0.90):
                    return dwell, frac, x_frac
            except Exception:  # noqa: BLE001 — sampling may degrade, touching may not
                pass
        return (self._sample_read_dwell(depth, complexity_hint),
                *self._sample_read_scroll())

    def _sample_read_scroll(self) -> tuple[float, float]:
        """Choose one forward-scroll distance/lane, falling back to legacy geometry."""
        policy = self._auto_behavior_policy()
        sampler = getattr(policy, "sample_read_scroll", None) if policy is not None else None
        if callable(sampler):
            try:
                frac, x_frac = sampler(self.read_scroll_frac, len(self._capture_scroll_ledger))
                frac, x_frac = float(frac), float(x_frac)
                # Reject malformed/extreme policy output before it can create an off-screen or
                # near-edge gesture.  Paid-control intersections are still owned by _scroll's
                # forbidden-zone guard; this is only geometry validation.
                if (math.isfinite(frac) and math.isfinite(x_frac)
                        and _READ_SCROLL_FRAC_MIN <= frac <= _READ_SCROLL_FRAC_MAX and 0.10 <= x_frac <= 0.90):
                    return frac, x_frac
            except Exception:  # noqa: BLE001 — behavior sampling may degrade, touching may not
                pass
        return self.read_scroll_frac, 0.5

    def _sample_read_dwell(self, depth: int, complexity_hint: float | None) -> float:
        """Choose the pause before a read-scroll; policy failures retain legacy pacing."""
        policy = self._auto_behavior_policy()
        sampler = getattr(policy, "read_dwell", None) if policy is not None else None
        if callable(sampler):
            try:
                dwell = float(sampler(self.dwell_s, depth, complexity_hint))
                if math.isfinite(dwell) and dwell >= 0.0:
                    return dwell
            except Exception:  # noqa: BLE001 — behavior sampling may degrade, touching may not
                pass
        return human_delay(self.dwell_s)

    def _capture_limit_for_profile(self, base: int | None = None) -> int:
        """Per-profile screencap ceiling.

        Observe/legacy behavior remains exactly the configured value.  An auto policy opts
        into a small upward-only variation, so the shipped safety baseline of eight is never
        weakened and profile after profile does not terminate at one identical depth.

        `base` overrides the configured `scroll_captures` for ONE read, and exists for the item
        enumeration pass, whose ceiling is a different quantity entirely
        (`_ENUMERATION_CAPTURE_LIMIT` -- see its comment for the measured frame counts). The
        policy's upward-only jitter is applied on top of whichever base is in force, so the
        enumeration read varies its depth exactly the way the ordinary read does: a fixed
        terminal depth is the bot signature the owner rule forbids, and it does not stop being
        one because the constant got larger.
        """
        base = self.scroll_captures if base is None else max(1, int(base))
        policy = self._auto_behavior_policy()
        if policy is None:
            return base
        sampler = getattr(policy, "capture_limit", None)
        if callable(sampler):
            try:
                sampled = int(sampler(base))
                if base <= sampled <= base + 2:
                    return sampled
            except Exception:  # noqa: BLE001 — keep the capture guard if sampling fails
                pass
        return base + random.randint(0, 2)

    def _observe_input_lease_key(self) -> str:
        """Stable, non-identifying lock-file path for this app/device pair."""
        serial = self.serial or "unbound"
        digest = hashlib.sha256(f"{self.spec.app}:{serial}".encode()).hexdigest()[:24]
        return str(Path(tempfile.gettempdir()) / f"operation-love-observe-{digest}.lock")

    @contextmanager
    def _observe_input_lease(self, operation: str):
        """Serialize Hinge OBSERVE reads/scrolls with a live decision wait.

        A review controller may take screenshots while the Worker is waiting, but it must not
        call ``current_profile`` or ``_scroll_to_top``: both move the same card and can make a
        header reflow look like a PASS. This fail-closed lease covers same-process helpers and
        separately launched controllers without touching ADB. It is intentionally only used by
        Hinge's passive OBSERVE input paths; auto capture/navigation retain their existing flow.
        """
        if self.spec.app != "hinge":
            yield
            return
        thread_id = threading.get_ident()
        if self._observe_input_lease_depth:
            if self._observe_input_lease_thread != thread_id:
                raise HingeActionError(
                    f"Hinge OBSERVE input is already owned by another thread; refusing {operation}")
            self._observe_input_lease_depth += 1
            try:
                yield
            finally:
                self._observe_input_lease_depth -= 1
            return

        lock_path = self._observe_input_lease_key()
        with _OBSERVE_INPUT_LEASES_LOCK:
            owner = _OBSERVE_INPUT_LEASES.get(lock_path)
            if owner is not None:
                raise HingeActionError(
                    f"Hinge OBSERVE input is already owned by another controller; refusing {operation}. "
                    "Use a read-only screenshot while the Worker is waiting.")
            fd = None
            try:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as exc:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                raise HingeActionError(
                    f"another process owns Hinge OBSERVE input for this device; refusing {operation}. "
                    "Use a read-only screenshot while the Worker is waiting.") from exc
            _OBSERVE_INPUT_LEASES[lock_path] = (id(self), thread_id)
            self._observe_input_lease_depth = 1
            self._observe_input_lease_thread = thread_id
            self._observe_input_lease_fd = fd
        try:
            yield
        finally:
            with _OBSERVE_INPUT_LEASES_LOCK:
                self._observe_input_lease_depth = 0
                self._observe_input_lease_thread = None
                self._observe_input_lease_fd = None
                _OBSERVE_INPUT_LEASES.pop(lock_path, None)
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _scroll_down_one(self, frac: float | None = None,
                         x_frac: float | None = None) -> None:
        """One humanized forward read-scroll, ALWAYS going through here (never a bare
        `touch.scroll_up()` call) so self._capture_scrolls stays the single source of truth
        for "how far down from the last confirmed top are we right now". Both
        _capture_current's read loop and _locate_target_heart's own re-navigation scrolls use
        this — if either called touch.scroll_up() directly instead, _scroll_to_top's ceiling
        would silently under-count again exactly the way the hardcoded-distance bug did."""
        if frac is None or x_frac is None:
            frac, x_frac = self._sample_read_scroll()
        self._scroll(frac, x_frac)
        # Append only after the transport accepted the gesture: a forbidden-zone refusal or
        # transport failure must not leave a fictional scroll for _scroll_to_top to undo.
        self._capture_scroll_ledger.append((frac, x_frac))
        self._capture_scrolls = len(self._capture_scroll_ledger)

    def _scroll_up_one(self, frac: float, x_frac: float) -> None:
        """One humanized REVERSE read-scroll: the twin of `_scroll_down_one`, going back up.

        Doc 5.5's bottom-up navigation is what needs it. Until 2026-08-12 every backwards
        travel in this driver went through `_scroll_to_top`, which REPLAYS a counted ledger and
        whose arrival test is a settle heuristic "every pre-existing caller ignores"; walking up
        one measured step at a time, cross-checked against the item index, needs a single reverse
        gesture it can size itself. It goes through `_scroll(..., reverse=True)`, so the
        forbidden-zone guard, the shared column jitter and the humanized kinematics are the same
        ones every forward read-scroll gets -- nothing here touches the transport.

        BOTH ARGUMENTS ARE REQUIRED, deliberately, and this is the one place the two methods
        differ in signature. `_scroll_down_one(frac=None, x_frac=None)` re-samples BOTH from the
        behaviour policy when either is None, which is right for a read loop and catastrophic
        here: the only caller is the closed loop, which owns the distance (doc 5.10.1's ratio
        rule), and a silently re-sampled 0.55 would move the content four times further than the
        step it was planned as. Having nowhere to put a None is the cheapest way to make that
        impossible.

        THE LEDGER IS APPENDED TO, NOT POPPED, AND THAT IS NOT AN OVERSIGHT. The ledger's only
        consumer is `_scroll_to_top`, where `len(ledger)` is a CEILING on undo-swipes ("a few
        attempts beyond the recorded count are a hard safety margin ... not a target") and the
        settle check is what actually ends that loop. Overstating the outstanding downward
        travel therefore costs nothing -- at the top, one downward swipe changes no pixels and
        the loop exits on its first iteration -- while UNDERSTATING it leaves the card scrolled
        and the next capture's identity anchor seeded from a real person's sticky header instead
        of the app's chrome (see _ensure_session_top). Popping one forward entry per reverse
        gesture would understate whenever a reverse step is smaller than the forward step it is
        undoing, which is exactly what the ratio rule makes likely (both are drawn per frame
        against the card in front of them, independently). So the safe direction is chosen and
        stated rather than a distance ledger being invented for a consumer that only wants a
        bound.
        """
        self._scroll(frac, x_frac, reverse=True)
        # Same rule as _scroll_down_one: append only after the transport accepted the gesture.
        self._capture_scroll_ledger.append((frac, x_frac))
        self._capture_scrolls = len(self._capture_scroll_ledger)

    def _interruptible_sleep(self, seconds: float, should_stop=None) -> bool:
        """Sleep `seconds`, but in slices, giving up early if `should_stop` fires.
        Returns True if the full time elapsed, False if it was cut short by a stop.

        Deliberately implemented with the MODULE-LEVEL time.sleep in a loop rather than a
        threading.Event.wait: every Hinge test file neutralises real time by monkeypatching
        `hinge.time.sleep` (four separate autouse fixtures, since there is no conftest.py).
        A wait primitive those fixtures cannot see would silently turn an offline suite that
        runs in ~100s into one that genuinely sleeps through 1.1s dwells x 11 per capture --
        the kind of regression that shows up as "tests got slow", never as a failure.

        With should_stop=None this is exactly time.sleep(seconds), one call, so every
        existing caller and every existing dwell-accounting test is unaffected.
        """
        if should_stop is None:
            time.sleep(seconds)
            return True
        # Bounded by a slice COUNT as well as by the deadline. The deadline alone is correct in
        # production but not under the four autouse fixtures that patch `hinge.time.sleep` to a
        # no-op: with sleeping removed, the deadline is the only thing left and the loop spins
        # on time.monotonic() burning CPU for the full wall-clock duration -- which is the exact
        # "the suite silently starts taking real time" regression routing this through
        # time.sleep was chosen to avoid, reintroduced by the loop around it. In production the
        # two bounds coincide (each slice really does sleep ~_OBSERVE_POLL_S); with sleep
        # patched out, the count is what makes this behave like the should_stop=None branch.
        deadline = time.monotonic() + seconds
        for _ in range(max(1, math.ceil(seconds / _OBSERVE_POLL_S))):
            if should_stop():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(_OBSERVE_POLL_S, remaining))
        return not should_stop()

    @staticmethod
    def _raise_if_action_cancelled(should_stop, *, boundary: str) -> None:
        """Make Stop a hard boundary before any further autonomous device input."""
        if should_stop is not None and should_stop():
            raise ActionCancelled(
                f"action cancelled because the run is stopping before {boundary}; no further "
                "device input was issued")

    def _scroll_to_top(self, should_stop=None) -> bool:
        # This private helper is still called by operational controllers in practice. Guard it
        # as well as current_profile(), so a controller cannot bypass the public-read refusal
        # and scroll a live OBSERVE card underneath wait_for_decision().
        with self._observe_input_lease("_scroll_to_top"):
            return self._scroll_to_top_unlocked(should_stop)

    def _scroll_to_top_unlocked(self, should_stop=None) -> bool:
        """Swipe the profile back to the top (content down) until it stops moving.

        Returns True only when the top was CONFIRMED — the loop ended because the view stopped
        moving. False means the swipe ceiling ran out first, or a stop interrupted it, i.e. the
        card may still be scrolled. Every pre-existing caller ignores this; it exists for
        _ensure_session_top, which needs to say honestly whether the invariant it is there to
        restore actually got restored.

        Two independent things have to be right for this to actually land back at the top,
        and a while ago only one of them was fixed:

          COUNT — how many undo-swipes to throw. Ceiling is self._capture_scrolls: the
          number of forward read-scrolls actually performed since the last confirmed top
          (tracked by _scroll_down_one, the only place that calls touch.scroll_up()) -- not a
          fixed guess, which could not undo a capture that scrolled further than it assumed.

          DISTANCE — how far each undo-swipe travels. This used to be a hardcoded drag
          (0.35h -> 0.80h = 0.45h) while the read-scroll it must cancel travels
          read_scroll_frac of the screen (0.55h by default) -- an 18%-per-scroll shortfall
          that COUNT alone can't fix: matching the swipe COUNT to the scroll count still
          leaves the profile scrolled down by N * (read_scroll_frac - 0.45) after N swipes,
          which compounds on a long profile (up to ~0.7 screen-heights short at the default
          scroll_captures=8). The two fractions must be tied together, not maintained as
          separate magic numbers that can drift apart again. Observe mode therefore mirrors
          read_scroll_frac exactly. Auto mode instead derives a bounded mean from the real
          forward-scroll ledger, then samples broader return strokes and relies on the
          screenshot settle check below rather than replaying one exact reverse per forward.

        _changed() still ends the loop the moment the view settles (its own safety net against
        an animated/video card whose frames never settle); the count is a ceiling, not a
        target. On exit the profile IS at (or past) the top, so the counter resets to 0 -- the
        next forward scroll (whether from a new capture or from _locate_target_heart's search)
        starts counting fresh from a confirmed top.

        STOP: `should_stop` is polled before each undo-swipe, because this loop is the half of
        the ~85s stop-deaf window the operator actually SEES ("it completes the read by
        scrolling a bunch, then stops"). A stop here returns WITHOUT the counter/ledger reset
        below: those two lines are a CLAIM that the card is back at the top, and abandoning the
        undo half-way makes that claim false, so making it anyway would be recording something
        untrue about the device.

        Nothing in THIS process then goes on to act on the stale ledger.  Both capture and the
        remaining legacy navigation caller thread their stop callback here; the caller turns the
        abort into ``ActionCancelled`` before any later tap/text/send.  A phone left mid-scroll
        is the documented "a stop leaves the screen untouched for debugging" outcome;
        `_ensure_session_top` is what makes the NEXT session safe, since a stale ledger cannot
        survive process exit anyway.
        """
        w, h = self.adb.screen_size()
        ledger = list(self._capture_scroll_ledger)

        # Compatibility for tests/tools (and an in-flight pre-ledger driver) that only set
        # _capture_scrolls.  Observe mode also keeps its old exact-mirror behavior when there
        # is no auto policy, minimizing risk in the manual-label path.
        if not ledger and self._capture_scrolls:
            ledger = [(self.read_scroll_frac, 0.5)] * self._capture_scrolls
        policy_mode = self._auto_behavior_policy() is not None
        max_swipes = max(1, len(ledger))
        if policy_mode and ledger:
            # A few attempts beyond the recorded count are a hard safety margin for physical
            # under-travel, not a target.  Screenshot settling still ends the loop as soon as
            # the real top is reached.  The bound prevents animated/video cards from spinning.
            max_swipes = len(ledger) + 3
        # Mirror of touch.scroll_up(read_scroll_frac): same y-extents, reversed direction, so
        # one undo-swipe travels exactly as far as one forward read-scroll.
        legacy_frac = self.read_scroll_frac
        mean_forward = (sum(frac for frac, _lane in ledger) / len(ledger)
                        if ledger else legacy_frac)
        settled = False
        for _ in range(max_swipes):
            if should_stop is not None and should_stop():
                return False               # see STOP above: do NOT fall through to the reset
            if policy_mode and ledger:
                # Undo in broader, independently varied strokes.  It intentionally is not a
                # reverse replay of N forward gestures: real people return to the top with a
                # different hand motion/count, while the settle check below remains the source
                # of truth.  Keep enough headroom from screen edges and paid-control zones.
                undo_frac = max(0.30, min(0.72, mean_forward * random.uniform(1.10, 1.28)))
                x_frac = random.uniform(0.38, 0.62)
            else:
                undo_frac = legacy_frac
                x_frac = 0.5
            y_near = int(h * (0.5 - undo_frac / 2))
            y_far = int(h * (0.5 + undo_frac / 2))
            x = int(w * x_frac)
            before = self._screencap()
            if should_stop is not None and should_stop():
                return False
            self._swipe(x, y_near, x, y_far)
            if not self._interruptible_sleep(human_delay(0.3), should_stop):
                return False               # stop landed inside the settle wait; same rule as above
            if not self._changed(before, self._screencap()):
                settled = True
                break
        self._capture_scrolls = 0
        self._capture_scroll_ledger = []   # confirmed (or ceiling-bounded) back at top
        return settled

    def _ensure_session_top(self, should_stop=None) -> None:
        """Once per session, before the first capture, put the card at a CONFIRMED scroll-top.

        _capture_current's identity anchor rests on one invariant: frame 0 of a capture is at
        scroll-top, so the band it seeds `_identity_top_sig` from is the app's own
        profile-independent chrome (Hinge's filter-chips row) rather than a real person's
        sticky header. Within a session that invariant is maintained by current_profile()'s
        own trailing _scroll_to_top(). ACROSS sessions nothing maintained it at all --
        open_session() only foregrounds the app (`monkey ... LAUNCHER 1`), which restores
        whatever scroll position the card was left at.

        That gap was harmless while every stop happened between profiles, and it stopped being
        harmless the moment Stop could abandon a read mid-scroll (see _capture_current): the
        ordinary Stop -> Start flow then starts the next run mid-card, `_identity_top_sig` gets
        seeded with THAT PERSON's header, `_identity_sig`/`_identity_name` never get set at all,
        and layer 1 is silently dead for the first card of the run -- the one screen an operator
        is most likely to scroll through before deciding. (It was reachable before, too, just
        rarely: a mid-capture profile split or any exception mid-read leaves the card scrolled
        the same way.)

        Cost when the card is already at the top, which is the normal case: one downward swipe
        and two screencaps (~3s), once per run -- _scroll_to_top's settle check ends the loop on
        the first iteration. The ceiling below is what makes this work from an UNKNOWN position:
        _scroll_to_top sizes its swipe budget from the scroll ledger, which is empty at session
        start, so without this it would throw exactly one swipe and give up. `_capture_scrolls`
        is the documented pre-ledger compatibility input for exactly that (see _scroll_to_top's
        COUNT paragraph), and the worst case it must cover is a previous run's own read ceiling,
        hence scroll_captures.

        Not fatal when it fails: a card that never settles (an animated/video profile) leaves
        the invariant unrestored, and _identity_of's id_sig-is-None branch is what keeps that
        honest -- it can answer 'top' or 'unknown', never a manufactured 'new'.
        """
        # An open like/comment sheet is not a card, and swiping under one would drag the sheet
        # the operator is composing in. _capture_current already refuses to read or scroll on a
        # sheet for exactly this reason ("makes a capture that STARTS on a sheet completely
        # input-free") -- this pass runs BEFORE that check, so it has to make the same promise
        # itself or it would silently break it. Deliberately does NOT mark the session done:
        # nothing was restored, so the next capture (after the sheet closes) should try again.
        frame = self._screencap(on_blank="none")
        if frame is not None and self._observe_like_sheet_visible(frame):
            return
        self._session_top_done = True     # one attempt per session, even if it fails below
        self._capture_scrolls = self.scroll_captures
        self._capture_scroll_ledger = []
        if self._scroll_to_top(should_stop):
            return
        if should_stop is not None and should_stop():
            return                        # interrupted, not failed -- the run is ending anyway
        print(f"{self.spec.app}: could not confirm the card is scrolled to the top at session "
              f"start (it never stopped moving). The first profile's identity anchor may be "
              f"unavailable; decisions on it fall back to content matching.")
        if self._dbg is not None:
            try:
                self._dbg.action("session_top_unconfirmed", swipe_ceiling=self.scroll_captures)
            except Exception:  # noqa: BLE001 — debug logging must never break a session
                pass

    def _note_capture_aborted(self, frames: int) -> None:
        """Record that a profile read was abandoned because Stop was requested.

        Written to the debug log so actions.jsonl says why a `capture` record the reader was
        expecting never appeared -- without it, an interrupted run's log simply ends after the
        previous decision, which reads identically to a crash or a wedged device. Deliberately
        takes NO screenshot (unlike _dbg_action, which grabs a fresh screencap for every
        record): this fires on the shutdown path, where the entire point is to stop touching
        the device, and one more ~1s ADB round-trip per abort would eat into the very latency
        this change exists to remove. The frame count is the useful part anyway -- it says how
        far into the read the stop landed.
        """
        scrolled = len(self._capture_scroll_ledger)
        where = (f" The card is left scrolled down {scrolled} screen(s) — that is deliberate "
                 f"(a stop leaves the screen untouched); scroll it back to the top yourself "
                 f"before the next run so the first capture starts from a known position."
                 if scrolled else "")
        print(f"{self.spec.app}: stop requested while reading this profile — abandoning the "
              f"read after {frames} frame(s); nothing was recorded for it.{where}")
        if self._dbg is not None:
            try:
                self._dbg.action("capture_aborted", frames=frames,
                                 read_scrolls=len(self._capture_scroll_ledger),
                                 profile_name=self._identity_name)
            except Exception:  # noqa: BLE001 — debug logging must never break a shutdown
                pass

    # --- item enumeration: doc 5.3's driver-owned index space -----------
    #
    # Four leaf modules do every piece of thinking here (scroll_top confirms the top, segment
    # finds the cards and hearts, scroll_step sizes each gesture against the card in front of
    # us, item_index folds the frames into one page and item_crops cuts the numbered crops).
    # This driver contributes exactly two things they cannot have: a device to look at, and the
    # humanized gesture path. Nothing below issues a gesture except through `_scroll_down_one`,
    # so the ledger, the jitter and the forbidden-zone guard all still apply unchanged.
    #
    # EVERY FAILURE HERE IS A RECORDED SENTENCE, NOT AN EXCEPTION, and that is deliberate. The
    # frames are wanted by two independent consumers: the ranker (faces, embeddings, the stored
    # label) and the opener (numbered items). Enumeration failing says nothing about the first,
    # so a failed enumeration must not discard a perfectly good capture -- it produces a Profile
    # with `items_unavailable` set, and worker.py's auto loop turns that into the hard stop
    # before any like is sent. Doc 5.2's rule ("never fall back to sending raw frames") is
    # enforced at that decision point, which is where the substitution would otherwise happen,
    # rather than by crashing the read here.

    def _item_enumeration_blocker(self) -> str:
        """Why this capture will not attempt an enumeration read, or "" when it will.

        Five conditions, checked before a single frame is looked at, so nothing below has to
        re-derive them.

        OBSERVE IS NO LONGER EXCLUDED (doc 5.9's inversion, 2026-08-12). Until this workflow the
        first condition here was `_auto_session`, on the reasoning that enumeration reads a
        profile in ~37-44 frames instead of `scroll_captures`' 12 and observe would pay that on
        the one mode a human sits and waits through, "while having nothing yet to show for it".
        Observe now has everything to show for it: it runs the SAME request auto runs (numbered
        crops, no anchor), and the model's chosen item is what the hub tells the operator to
        like. Keeping the exclusion would have meant observe generating from raw scroll frames
        while auto generated from crops -- the same request-shape divergence one layer down that
        the 2026-08-12 audit named as the quiet death of the canary property.

        THE WALL CLOCK IS REAL AND IT IS THE OWNER'S TO ACCEPT. Measured against the two
        calibration profiles' page heights at the loop's own 233..262px cadence, the read goes
        from 12 frames to 36 (profile B, 8349px) and 44 (profile A, 10027px). The scroll back to
        the top does NOT scale with it: `_scroll_to_top` has no auto behaviour policy in observe,
        so it throws `read_scroll_frac`-sized (0.55h) undo strokes and ends on its own settle
        check, i.e. ~9-10 swipes over a ~10,000px page regardless of how many fine enumeration
        steps went down. So the cost is the READ, roughly 3x the ~85s per-profile figure, and doc
        5.5's bottom-up navigation buys observe nothing at all -- what it removed was the BOT's
        rewind-then-walk on the auto like path, and in observe the human does the navigating.

        OPENERS ADMINISTRATIVELY DISABLED IS THE ONE POLICY DECISION LEFT (audit fix, "BUG 2",
        2026-08-12). Item enumeration exists solely to let the model pick an item for an opener
        -- with `opener.enabled: false` there is no consumer for a numbered item list at all, so
        this refuses: not "capability" (Hinge can always attach a comment, see `accepts_opener`
        below) but "nothing downstream wants this payload". Before this check existed, a
        disabled-opener AUTO run still paid for a ~40-frame enumeration read, and any refusal in
        it (an unconfirmed scroll top, a spacing no gesture could respect, ...) reached
        worker.py's `items_unavailable` stop and halted a run that was only ever going to send
        bare likes -- see `set_opener_enabled`'s docstring for who sets `self._openers_enabled`
        and when. BOTH loops now call that hook, which is what lets this one condition carry the
        whole "does this session want a numbered list" question for observe as well as auto.

        The other four are capability, not policy: an app that cannot send an opener at swipe
        time has nothing to number items for; absent per-device targeting calibration means the
        model's numbered choice could never be safely acted on or checked in Observe (the exact
        waste exposed by the 33-frame reported run); the affirmative scroll-top gate reads
        `identity_band` and cannot run without one; and `segment_frame` refuses a None like
        template rather than reporting a page of heartless cards (which would silently
        reclassify every photo as unselectable context).
        """
        if not getattr(self, "_openers_enabled", True):
            return ("openers are disabled for this run (opener.enabled: false), so nothing "
                    "would consume a numbered item list -- reading a profile at the enumeration "
                    "cadence for a run that only ever sends bare likes would waste device time "
                    "and risk stopping a run over the absence of a payload nobody wanted")
        if not self.accepts_opener:
            return (f"{self.spec.app} cannot attach an opener to a like at swipe time, so there "
                    "is no request for numbered items to answer")
        if self.targeting_calibration is None:
            return (f"apps.{self.spec.app}.targeting_calibration is unavailable "
                    f"({self._targeting_calibration_unavailable or 'no reason was recorded'}), "
                    "so a model-selected item could not be verified or targeted and no "
                    "numbered item list would have a usable consumer")
        if self.identity_band is None:
            return (f"{self.spec.app} declares no identity_band, so doc 5.5's affirmative "
                    "filter-chips scroll-top confirmation cannot be read at all, and counting "
                    "items from an unconfirmed top gives a systematic off-by-N in every ordinal")
        if self._template("like") is None:
            return (f"{self.spec.app} declares no calibrated 'like' glyph template, so hearts "
                    "cannot be located and every card would segment as unselectable context")
        return ""

    def _confirm_enumeration_top(self) -> str:
        """Affirmatively confirm the card is at scroll top. "" when confirmed, else the reason.

        Doc 5.5: "the design needs an affirmative top confirmation before counting starts ...
        treat failure to confirm as a hard stop". The signal is positive rather than an absence
        -- at a genuine top Hinge draws its own filter-chips row in `identity_band`, and the
        sticky per-profile header covers that strip the moment the card is scrolled at all.

        This runs BEFORE the read loop, on its own screencap, because the answer decides the
        read's CEILING (`_ENUMERATION_CAPTURE_LIMIT` vs `scroll_captures`) and a ceiling cannot
        be raised half way through a loop that is already running against it. The frame is read
        with `on_blank="none"` so a screen that has gone dark is reported here as an
        unconfirmable top rather than raising out of a gate whose only job is to answer a
        question -- the loop's own `_screencap()` a moment later is the caller that is entitled
        to raise about a blank screen, and it does.

        `confirm_scroll_top`, not `require_scroll_top`: the exception form is for callers that
        must stop, and this one has a third option that is neither stopping nor proceeding
        blind, namely reading the profile without enumerating it. The three-valued verdict is
        preserved in the reason text either way -- "confirmed NOT at top" and "cannot tell" are
        different situations for whoever reads the stop line.
        """
        frame = self._screencap(on_blank="none")
        if frame is None:
            return ("the screen was blank when the scroll-top gate looked, so the filter-chips "
                    "row could not be read and the top could not be confirmed")
        try:
            verdict = confirm_scroll_top(frame, identity_band=self.identity_band)
        except ScrollTopError as exc:
            return (f"the scroll-top gate could not read the identity band ({exc}), so the top "
                    "could not be confirmed")
        if not verdict.confirmed:
            return (f"the card is not confirmed to be at its scroll top ({verdict.state}): "
                    f"{verdict.reason}")
        return ""

    def _plan_enumeration_step(self, frame: bytes, x_frac: float, min_spacing_px: int | None,
                               *, allow_segmentation_failure_fallback: bool = False):
        """Size the next enumeration scroll against the spacing THIS frame shows.

        Returns the `ScrollStep`; `.frac` and `.x_frac` go straight to `_scroll_down_one`, both
        of them, always (passing the frac alone makes that method re-sample both from the
        behaviour policy and silently issue production's 0.55 cadence instead).

        `x_frac` is the LANE the behaviour policy already drew for this step, handed through
        rather than replaced: the distance is what has to follow the card in front of us (doc
        5.10.1's ratio rule), while the column the thumb travels in is ordinary humanization and
        has no business being decided by a geometry module. `_sample_read_step` has already
        validated it into 0.10..0.90, which is exactly the window `plan_scroll_step` accepts.

        `min_spacing_px` is the smallest spacing measured anywhere on THIS profile so far, which
        is the loop's only piece of memory: it makes the step shrink permanently once a short
        card has been seen, which is the only defence against a card that is still below the
        fold. Per profile -- carrying it across a deck advance would be wrong.

        NO `**plan_kwargs` PASS-THROUGH, deliberately. `ratio_window`, `max_step_px` and
        `fallback_spacing_px` are the offline-validation door doc 5.6 flags: widening one takes
        the gesture outside the envelope every measurement in this stack was taken inside. A
        production caller passes none of them, and the way to keep that true is to have nowhere
        to put them. ``allow_segmentation_failure_fallback`` is intentionally not a geometry
        override: the capture loop may use it only for the capped contiguous run, producing an
        explicitly marked corpus-minimum step whose contradictory frames must be omitted by
        item_index's separate measured-bridge rebuild before they can ever produce an opener.
        """
        segmentation = segment_frame(frame, content_band=self.content_band,
                                     like_template=self._template("like"),
                                     like_threshold=_LIKE_MATCH_THRESHOLD)
        return plan_scroll_step(segmentation, x_frac=x_frac,
                                profile_min_spacing_px=min_spacing_px,
                                allow_segmentation_failure_fallback=
                                allow_segmentation_failure_fallback)

    @staticmethod
    def _match_video_mute(frame: bytes, rect: tuple[int, int, int, int],
                          ) -> tuple[bool, float | None]:
        """Return whether one card-local ROI contains the exact Hinge mute control.

        ``screened=False`` means decode/template/matcher failure, not "no video".  The control's
        inner square is entirely black-background UI plus white glyph, so ordinary
        ``TM_CCOEFF_NORMED`` can make the promised near-perfect comparison without a mask whose
        OpenCV support varies by method/version.
        """
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            template = cv2.imdecode(
                np.frombuffer(base64.b64decode(_VIDEO_MUTE_TEMPLATE_B64), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)
            if image is None or template is None:
                return False, None
            x0, y0, x1, y1 = rect
            if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                return False, None
            roi = image[y0:y1, x0:x1]
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                return False, None
            score = float(cv2.minMaxLoc(
                cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))[1])
            return True, score
        except Exception:  # noqa: BLE001 — selection screening fails closed; it never raises
            return False, None

    @staticmethod
    def _locate_video_mute(frame: bytes, rect: tuple[int, int, int, int],
                           ) -> tuple[bool, float | None, tuple[int, int] | None]:
        """Return the origin of the same exact mute match used by the selection screen."""
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            template = cv2.imdecode(
                np.frombuffer(base64.b64decode(_VIDEO_MUTE_TEMPLATE_B64), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)
            if image is None or template is None:
                return False, None, None
            x0, y0, x1, y1 = rect
            if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                return False, None, None
            roi = image[y0:y1, x0:x1]
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                return False, None, None
            _min, score, _min_loc, max_loc = cv2.minMaxLoc(
                cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
            return True, float(score), (x0 + max_loc[0], y0 + max_loc[1])
        except Exception:  # noqa: BLE001 — no location is no repair authority
            return False, None, None

    def _video_mute_frame_markers(self, frames: list[bytes]) -> tuple[bool, ...]:
        """Affirmative per-frame video UI evidence for the generic item indexer.

        The full-frame search is still tightly bounded: left 30% of the screen and only the
        configured content band.  That contains the card-local mute control but excludes both
        Android's top-bar mute icon and Hinge's right-side like hearts.  Matcher failure is False
        (no repair authority); the later per-card selection screen remains independently
        fail-closed before anything can be numbered.
        """
        markers: list[bool] = []
        for frame in frames:
            try:
                import cv2
                import numpy as np

                image = cv2.imdecode(
                    np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    markers.append(False)
                    continue
                height, width = image.shape
                rect = (0, round(self.content_band[0] * height), round(0.30 * width),
                        round(self.content_band[1] * height))
                screened, score = self._match_video_mute(frame, rect)
                markers.append(bool(screened and score is not None
                                    and score >= _VIDEO_MUTE_MATCH_THRESHOLD))
            except Exception:  # noqa: BLE001 — marker absence never grants repair authority
                markers.append(False)
        return tuple(markers)

    def _video_mute_marker_rows(self, frames: list[bytes]) -> tuple[VideoMuteMarker, ...]:
        """Positioned mute observations for v12's physical-card tracker.

        The old boolean reader remains for manifests/tests, but production indexing receives
        these rows: a video can carry its identity after the control itself scrolls offscreen.
        """
        markers: list[VideoMuteMarker] = []
        for frame_index, frame in enumerate(frames):
            try:
                import cv2
                import numpy as np

                image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                height, width = image.shape
                rect = (0, round(self.content_band[0] * height), round(0.30 * width),
                        round(self.content_band[1] * height))
                ok, score, origin = self._locate_video_mute(frame, rect)
                if ok and origin is not None and score is not None and score >= _VIDEO_MUTE_MATCH_THRESHOLD:
                    markers.append(VideoMuteMarker(
                        frame_index=frame_index, x=origin[0], y=origin[1], score=score))
            except Exception:  # noqa: BLE001 — no affirmative marker is no repair authority
                continue
        return tuple(markers)

    def _video_selection_exclusions(self, frames: list[bytes], index) -> dict[int, str]:
        """Map mute-marked/unscreenable heart ordinals to post-index payload exclusions.

        ItemIndex deliberately keeps video blocks and their hearts: their stable card edges and
        gutters are valid scroll evidence, and removing a heart from page space would shift every
        target below it.  This method acts later, at the product-policy boundary.  It looks only
        inside each selectable block's upper-left overlay ROI; the Android mute status icon and
        the per-card like heart are outside that rectangle.  One near-perfect app-UI match is
        affirmative video evidence.

        Neither the absence of a sufficiently visible top sighting nor matcher failures with no
        successful search prove a still photograph. That physical card is therefore excluded
        from automatic targeting. A successful full-ROI search with no mute control clears it;
        clipped slivers are skipped as geometry, and the independent photo classifier still
        decides whether the crop is actually photographic.
        """
        blocks = tuple(getattr(index, "selectable", ()) or ())
        excluded: dict[int, str] = {}
        for block in blocks:
            ordinal = block.heart_ordinal
            if ordinal is None:
                continue
            top_sightings = tuple(
                obs for obs in block.observations
                if obs.top_observed and 0 <= obs.frame_index < len(frames))
            if not top_sightings:
                excluded[ordinal] = (
                    "video_mute_v1: not targetable because no source frame observed the card "
                    "top needed for mute-control screening")
                continue

            screened = False
            failed = False
            mute_frames: list[int] = []
            insufficient_frames: list[int] = []
            outcomes: list[dict[str, object]] = []
            for obs in top_sightings:
                card_width = block.x1 - block.x0
                x0 = block.x0 + round(_VIDEO_MUTE_X_BAND[0] * card_width)
                x1 = block.x0 + round(_VIDEO_MUTE_X_BAND[1] * card_width)
                y0 = obs.frame_y0 + round(_VIDEO_MUTE_Y_BAND[0] * block.height)
                y1 = min(obs.frame_y1,
                         obs.frame_y0 + round(_VIDEO_MUTE_Y_BAND[1] * block.height))
                # A top grazing the analysed band's bottom can expose fewer rows than the
                # 42x42 mute template. That is incomplete geometry, not matcher failure: skip
                # it and use a later full sighting. With no full sighting, screening still
                # fails closed below.
                if (x1 - x0 < _VIDEO_MUTE_TEMPLATE_SIDE_PX
                        or y1 - y0 < _VIDEO_MUTE_TEMPLATE_SIDE_PX):
                    insufficient_frames.append(obs.frame_index)
                    outcomes.append({"source_frame_index": obs.frame_index,
                                     "roi": [x0, y0, x1, y1],
                                     "outcome": "insufficient_visible_roi"})
                    continue
                ok, score = self._match_video_mute(
                    frames[obs.frame_index], (x0, y0, x1, y1))
                if not ok:
                    failed = True
                    outcomes.append({"source_frame_index": obs.frame_index,
                                     "roi": [x0, y0, x1, y1],
                                     "outcome": "matcher_failed"})
                    continue
                screened = True
                outcomes.append({"source_frame_index": obs.frame_index,
                                 "roi": [x0, y0, x1, y1],
                                 "outcome": ("mute_matched" if score is not None
                                             and score >= _VIDEO_MUTE_MATCH_THRESHOLD
                                             else "no_mute_match"),
                                 "score": (round(float(score), 6)
                                           if score is not None else None)})
                if score is not None and score >= _VIDEO_MUTE_MATCH_THRESHOLD:
                    mute_frames.append(obs.frame_index)

            def outcome_summary(values: list[dict[str, object]]) -> str:
                # Geometry and UI-match outcomes only: no crop bytes, image content, OCR text,
                # or matcher exceptions enter a profile manifest.
                return "; ".join(
                    f"f{value['source_frame_index']}:{value['outcome']}@"
                    f"{value['roi']}"
                    + (f" score={value['score']}" if value.get("score") is not None else "")
                    for value in values[:12])

            if mute_frames:
                excluded[ordinal] = (
                    "video_mute_v1: upper-left Hinge mute control matched in card source "
                    f"frame(s) {sorted(set(mute_frames))}; videos are retained for scroll "
                    "geometry but are never numbered or likeable; screen outcomes: "
                    + outcome_summary(outcomes))
            elif not screened:
                failed_frames = [value["source_frame_index"] for value in outcomes
                                 if value["outcome"] == "matcher_failed"]
                blocker = (
                    "upper-left mute-control screening failed in source frame(s) "
                    f"{failed_frames}" if failed else
                    f"no source frame exposed the {_VIDEO_MUTE_TEMPLATE_SIDE_PX}x"
                    f"{_VIDEO_MUTE_TEMPLATE_SIDE_PX} upper-left mute-control template area; "
                    f"clipped source frame(s) {insufficient_frames}")
                excluded[ordinal] = (
                    f"video_mute_v1: not targetable because {blocker}; screen outcomes: "
                    + outcome_summary(outcomes))
        return excluded

    def _index_captured_items(self, photos: list[bytes]) -> str:
        """Fold this capture's frames into doc 5.3's index and doc 5.2's crops.

        Sets `_current_item_index` / `_current_item_payload` on success and returns ""; on any
        refusal it leaves both None and returns the reason, which becomes
        `Profile.items_unavailable` and then worker.py's stop line. Called exactly once per
        capture, after the read loop, with the frames that were actually kept.

        `at_scroll_top=True` is passed because `_confirm_enumeration_top` confirmed it
        affirmatively for this very read -- that argument is an assertion by the caller and this
        is the caller doc 5.5 had in mind. It is what buys ABSOLUTE heart ordinals, which is
        what a counting navigation needs; an index built with it False numbers hearts relative
        to whatever happened to be in view.

        `identity_band` is passed for the same kind of reason and is REQUIRED by the builder: the
        index carries a fingerprint of this profile's sticky header, taken from these very
        frames, and `item_nav`'s entry gate refuses to navigate an index that cannot say whose
        profile it describes (doc 5.7's carried-forward requirement 1 -- geometry cannot tell two
        stereotyped Hinge cards apart, so the check has to be this strip).

        FOUR FAILURE FAMILIES, all of them results rather than crashes:
          * the index refuses (aliasing, a broken correspondence chain, a block two frames
            disagree about, a heart nothing bounded). `usable` is False and its own failure list
            is the reason. Doc 5.10.1 measured what an over-large step costs: coverage first,
            then the whole index -- it cannot misnumber, and it cannot fabricate an item;
          * the index holds together but carries no IDENTITY (no `identity_band` declared, or a
            capture in which the sticky header never appeared). Nothing is wrong with the
            numbering; it simply could never be navigated safely, so it is refused before a
            billed opener call rather than after one;
          * the crops refuse (a block taller than the analysed band, nothing selectable left to
            number). Same contract, one layer up;
          * a dependency raises (`SegmentationError` on a frame that will not decode,
            `ShiftEstimationError` when no page space spans the capture, `ItemIndexError` /
            `ItemCropError` on a capture that cannot be indexed at all, including a missing
            cv2/numpy). Caught by name and turned into the same sentence.

        Note `build_item_payload` re-checks that these frames really are the ones the index was
        built from, by a sha256 per frame recorded at segmentation time. That guard exists
        because an offline validation pass once drove the same capture in REVERSE order past a
        weaker one and got ten confidently wrong crops with zero failures.
        """
        # Keep `index` outside the try so every refusal, including a dependency exception before
        # a result exists, can leave the best evidence we have in actions.jsonl.  The logger is
        # deliberately an observer: no failure in this diagnostic path may change the read's
        # fail-loud indexing outcome.
        index = None
        try:
            video_mute_markers = self._video_mute_marker_rows(photos)
            animation_markers = tuple(any(marker.frame_index == frame_index
                                          for marker in video_mute_markers)
                                      for frame_index in range(len(photos)))
            index = build_item_index(
                photos, content_band=self.content_band,
                like_template=self._template("like"), like_threshold=_LIKE_MATCH_THRESHOLD,
                at_scroll_top=True, identity_band=self.identity_band,
                animation_markers=animation_markers,
                video_mute_markers=video_mute_markers)
            if not index.usable:
                return self._item_index_refused(
                    photos, "the item index this capture produced contradicts itself, so its "
                    "numbering cannot be trusted: " + "; ".join(index.failures), index)
            if not index.identity.known:
                # Refused HERE rather than left for navigation, and the difference is a billed
                # call: an index that cannot say whose profile it describes is one
                # `item_nav.navigate_to_item` will refuse at its entry gate, so producing crops
                # from it would buy an opener for a profile that can never be targeted. The same
                # placement argument as every other refusal in this method -- the ranker's frames
                # are untouched, and worker.py stops before the model is asked anything.
                return self._item_index_refused(
                    photos, "this capture could not be fingerprinted for identity, so a navigation "
                    "pass could never confirm the card it counts on is this profile's: "
                    + index.identity.reason, index)
            source_indices = tuple(getattr(index, "source_frame_indices", ()) or ())
            if source_indices:
                if (len(source_indices) != len(index.frames)
                        or any(i < 0 or i >= len(photos) for i in source_indices)):
                    return self._item_index_refused(
                        photos, "this capture's recovered item index has invalid frame provenance "
                        "and cannot be cropped safely", index)
                indexed_photos = [photos[i] for i in source_indices]
            else:
                # Compatibility with a legacy/index test double that predates frame provenance.
                indexed_photos = photos
            video_exclusions = self._video_selection_exclusions(indexed_photos, index)
            payload = build_item_payload(
                indexed_photos, index,
                exclude=lambda block: video_exclusions.get(block.heart_ordinal),
                unnumber=unnumber_unless_confident_photo)
            if not payload.usable:
                return self._item_index_refused(
                    photos, "the item crops this capture produced are not a request the model can "
                    "be asked to answer: " + "; ".join(payload.failures), index)
        except (ItemIndexError, ItemCropError, SegmentationError, ShiftEstimationError) as exc:
            return self._item_index_refused(
                photos, f"this capture could not be indexed into items "
                f"({type(exc).__name__}: {exc})", index)
        self._current_item_index = index
        self._current_item_payload = payload
        self._record_item_index_recovery(photos, index)
        self._record_item_index_notes(photos, index)
        # Doc 5.5's bottom-up entry anchor. `photos[-1]` is the last frame the index was folded
        # from, so `index.offsets[-1]` is ITS page offset -- and it is also, by construction, the
        # frame still on screen when this capture returns: `_capture_current`'s repeated-frame
        # break happens BEFORE the repeat is appended, and its ceiling path issues no scroll on
        # the final iteration. So the shift `navigate_to_item` measures against it is 0px in the
        # ordinary case, and a measurement rather than an assumption in every case.
        self._current_item_anchor = photos[-1] if photos else None
        return ""

    @staticmethod
    def _item_payload_debug_manifest(payload, index) -> list[dict]:
        """Non-image provenance for every indexed crop, in page order.

        A bare ``items=1, item_context=10`` capture record cannot explain which physical
        card survived policy filtering, which heart it maps to, or why the other crops were
        left unnumbered.  Keep that mapping in actions.jsonl without saving any additional
        profile imagery: hashes identify repeated crop bytes, while page/frame coordinates and
        the policy reason explain the numbering decision.

        Best-effort by construction.  This is diagnostic metadata written after the payload is
        already valid; a malformed legacy/test-double crop must never break a live capture.
        """
        try:
            source_indices = tuple(getattr(index, "source_frame_indices", ()) or ())
            manifest: list[dict] = []
            for crop in payload.crops:
                local_frame = crop.frame_index
                source_frame = local_frame
                if (isinstance(local_frame, int) and 0 <= local_frame < len(source_indices)):
                    source_frame = source_indices[local_frame]
                classifier = None
                if crop.image is not None and crop.heart_ordinal is not None:
                    # Aggregate-only evidence: no OCR text and no additional saved image.  This
                    # is calculated through the classifier's own entry point so the manifest
                    # cannot drift from the verdict used by the policy.
                    from .item_type_preflight import crop_type_evidence
                    classifier = crop_type_evidence(crop.image)
                manifest.append({
                    "kind": crop.kind,
                    "model_item": crop.number,
                    "heart_ordinal": crop.heart_ordinal,
                    "source_frame_index": source_frame,
                    "page_rows": [crop.page_y0, crop.page_y1],
                    "crop_size": [crop.width, crop.height],
                    "crop_sha256": (hashlib.sha256(crop.image).hexdigest()[:16]
                                    if crop.image is not None else None),
                    "selection_evidence": classifier,
                    "reason": crop.reason,
                })
            return manifest
        except Exception:  # noqa: BLE001 — diagnostics never alter a successful capture
            return []

    def _record_item_index_recovery(self, photos: list[bytes], index) -> None:
        """Log an already-verified isolated-frame recovery without affecting the result.

        The recovery is safe only because `build_item_index` rebuilt and validated the entire
        reduced sequence, and the crop + identity gates above also passed. This record preserves
        the omitted raw frame as its anchor and both bridge endpoints for later review;
        diagnostics remain strictly best-effort, like `_item_index_refused`.
        """
        pair = getattr(index, "recovered_from_pair", None)
        pairs = tuple(getattr(index, "recovered_from_pairs", ()) or ())
        segmentation_frames = tuple(
            getattr(index, "recovered_from_segmentation_frames", ()) or ())
        bridge = getattr(index, "recovery_bridge", None)
        failed_shift = getattr(index, "recovery_failed_shift", None)
        failed_shifts = tuple(getattr(index, "recovery_failed_shifts", ()) or ())
        source = tuple(getattr(index, "source_frame_indices", ()) or ())
        omitted = [i for i in range(len(photos)) if i not in source]
        if (self._dbg is None or not isinstance(bridge, tuple) or len(bridge) != 2
                or not omitted or omitted != list(range(omitted[0], omitted[-1] + 1))):
            return
        left, right = bridge
        if not (0 <= left < omitted[0] <= omitted[-1] < right < len(photos)):
            return
        try:
            fields = {
                "omitted_frame_indices": omitted,
                "recovery_bridge": list(bridge),
                "original_status": getattr(failed_shift, "status", None),
                "original_reason": getattr(failed_shift, "reason", None),
                "original_agreeing": getattr(failed_shift, "agreeing", None),
                "original_dissenting": getattr(failed_shift, "dissenting", None),
                "original_eligible": getattr(failed_shift, "eligible", None),
                "recovery_reason": getattr(index, "recovery_reason", None),
            }
            if isinstance(pair, tuple) and len(pair) == 2:
                fields["recovered_from_pair"] = list(pair)
            if segmentation_frames:
                fields["recovered_from_segmentation_frames"] = list(segmentation_frames)
            if len(pairs) > 1 and len(pairs) == len(failed_shifts):
                fields["recovered_from_pairs"] = [list(item) for item in pairs]
                fields["original_failures"] = [{
                    "pair": list(failed_pair),
                    "status": getattr(shift, "status", None),
                    "reason": getattr(shift, "reason", None),
                    "agreeing": getattr(shift, "agreeing", None),
                    "dissenting": getattr(shift, "dissenting", None),
                    "eligible": getattr(shift, "eligible", None),
                } for failed_pair, shift in zip(pairs, failed_shifts, strict=True)]
            self._dbg.action("item_index_recovered", before=photos[left],
                             after=photos[right], anchor=photos[omitted[0]], **fields)
        except Exception:  # noqa: BLE001 -- diagnostics must never alter a live-run result
            pass

    def _record_item_index_notes(self, photos: list[bytes], index) -> None:
        """Make a successful, conservative segmentation repair visible in actions.jsonl.

        ``ItemIndex.notes`` names *index-local* frames.  An omission recovery can make those
        differ from the original capture positions, so retain both coordinates here rather than
        making an investigator guess which PNG ``frame 19`` meant.  This is diagnostic only: a
        malformed test double, logger, or note must not turn an otherwise usable index back into
        a refusal.
        """
        if self._dbg is None:
            return
        try:
            raw_notes = tuple(getattr(index, "notes", ()) or ())
            source = tuple(getattr(index, "source_frame_indices", ()) or ())
            local_count = len(tuple(getattr(index, "frames", ()) or ()))
            if not raw_notes or len(source) != local_count:
                return
            if any(not isinstance(i, int) or i < 0 or i >= len(photos) for i in source):
                return

            note_frames: list[dict[str, int]] = []
            notes: list[str] = []
            for note in raw_notes[:16]:  # a corrupt producer must not make one JSONL line huge
                if not isinstance(note, str):
                    continue
                match = re.match(r"^frame\s+(\d+)\b", note)
                if match is None:
                    notes.append(note[:800])
                    continue
                local = int(match.group(1))
                if not 0 <= local < len(source):
                    notes.append(note[:800])
                    continue
                original = source[local]
                note_frames.append({"local_frame_index": local,
                                    "source_frame_index": original})
                notes.append(re.sub(r"^frame\s+\d+\b",
                                    f"source frame {original} (index frame {local})",
                                    note, count=1)[:800])
            if not notes:
                return
            first = note_frames[0]["source_frame_index"] if note_frames else None
            runtime = _item_index_runtime_provenance()
            mute_markers = tuple(getattr(index, "video_mute_markers", ()) or ())
            repairs = []
            for repair in tuple(getattr(index, "repair_provenance", ()) or ())[:16]:
                local = getattr(repair, "pair_index", None)
                if not isinstance(local, int) or not 0 <= local + 1 < len(source):
                    continue
                marker_frames = tuple(getattr(repair, "marker_frames", ()) or ())
                marker_evidence = [
                    {"local_frame_index": marker.frame_index,
                     "source_frame_index": source[marker.frame_index],
                     "x": marker.x, "y": marker.y, "score": marker.score}
                    for marker in mute_markers
                    if marker.frame_index in marker_frames
                    and 0 <= marker.frame_index < len(source)]
                repairs.append({
                    "path": getattr(repair, "path", None),
                    "local_pair": [local, local + 1],
                    "source_pair": [source[local], source[local + 1]],
                    "raw": {"status": getattr(repair, "raw_status", None),
                            "delta_px": getattr(repair, "raw_delta_px", None)},
                    "effective": {"status": getattr(repair, "effective_status", None),
                                  "delta_px": getattr(repair, "effective_delta_px", None)},
                    "mute_markers": marker_evidence,
                })
            self._dbg.action("item_index_repaired",
                             before=(photos[first] if first is not None else None),
                             notes=notes, note_frames=note_frames,
                             source_frame_indices=list(source), item_index_runtime=runtime,
                             repairs=repairs)
        except Exception:  # noqa: BLE001 -- diagnostics must never affect a usable capture
            pass

    @staticmethod
    def _item_index_source_indices(index, photos: list[bytes]) -> tuple[int, ...]:
        """Index-local -> original-capture frame positions, conservatively normalised."""
        frames = tuple(getattr(index, "frames", ()) or ())
        source = tuple(getattr(index, "source_frame_indices", ()) or ())
        if (frames and len(source) == len(frames)
                and all(isinstance(i, int) and 0 <= i < len(photos) for i in source)):
            return source
        # An unrecovered / legacy index uses the original capture order.  Do not invent mapping
        # for a malformed index with more local frames than capture images.
        return tuple(range(min(len(frames), len(photos)))) if frames else tuple(range(len(photos)))

    def _save_item_index_refusal_evidence(
            self, photos: list[bytes], reason: str, index, cited_local: list[int],
            runtime_provenance: dict[str, str | None],
            priority_local: tuple[int, ...] = ()) -> tuple[str | None, list[str]]:
        """Persist a small, standalone refusal dossier when this is a real DebugLog.

        The regular action logger intentionally owns normal rotating screenshots.  These source
        frames are exceptional forensic evidence and need stable, human-readable names; keep the
        direct writes narrowly bounded and entirely behind this best-effort helper.
        """
        try:
            debug_dir = getattr(self._dbg, "dir", None)
            if debug_dir is None:
                return None, []
            debug_dir = Path(debug_dir)
            source = self._item_index_source_indices(index, photos)
            # A run can refuse more than one profile.  Keep each dossier immutable and make its
            # filenames traceable from the JSONL record, rather than letting a later refusal
            # overwrite the evidence an earlier record points to.
            dossier_id = hashlib.sha256(
                (photos[0] if photos else b"") + b"\0" + (photos[-1] if photos else b"")
                + b"\0" + str(reason).encode("utf-8", "replace")).hexdigest()[:12]
            local_candidates = sorted({i for i in cited_local if 0 <= i < len(source)})
            if not local_candidates and source:
                count = min(8, len(source))
                local_candidates = ([0] if count == 1 else
                                    [round(i * (len(source) - 1) / (count - 1))
                                     for i in range(count)])
            # Preserve the breadth of a long cited run (including both endpoints), not merely
            # its first eight frames -- but never at the cost of the frames the refusal is ABOUT.
            #
            # The even stride below walks POSITIONS in the merged candidate list and knows
            # nothing about which of them form a refused pair. Live 2026-08-16 (Grace) that cost
            # exactly the two frames the dossier existed to preserve: six refused pairs merged
            # into a 25-frame pool, the stride landed on positions 0 and 3 of the first pair's
            # five-frame window, and the bundle shipped frames 35 and 38 -- the NEIGHBOURS of
            # failing pair (36, 37) -- while both frames of the pair itself were dropped. The
            # refusal could not be replayed from its own evidence. Seed the selection with the
            # named frames first, then let the stride spend what is left on breadth.
            if len(local_candidates) > 8:
                cited = sorted({i for i in cited_local if 0 <= i < len(source)})
                chosen = {i for i in dict.fromkeys(priority_local) if i in set(cited)}
                budget = max(0, 8 - len(chosen))
                # The stride still spans the whole pool including both endpoints; it just runs
                # over the slots the named frames left it.
                if budget > 1:
                    chosen.update(cited[round(i * (len(cited) - 1) / (budget - 1))]
                                  for i in range(budget))
                elif budget == 1:
                    chosen.add(cited[0])
                # A named frame that the stride would also have picked costs a slot rather than
                # shrinking the dossier, so the cap stays a floor as well as a ceiling.
                for value in cited:
                    if len(chosen) >= 8:
                        break
                    chosen.add(value)
                local_candidates = sorted(chosen)

            offsets = tuple(getattr(index, "offsets", ()) or ())
            frames = tuple(getattr(index, "frames", ()) or ())
            shifts = tuple(getattr(index, "shifts", ()) or ())
            animation_markers = tuple(getattr(index, "animation_markers", ()) or ())
            mute_markers = tuple(getattr(index, "video_mute_markers", ()) or ())
            try:
                track_deltas = _video_track_deltas(
                    frames, shifts, mute_markers, extent_tolerance_px=getattr(
                        index, "extent_tolerance_px", 8))
            except Exception:  # noqa: BLE001 -- provenance cannot affect a refusal
                track_deltas = {}

            def geometry_record(local: int, screenshot: str | None = None) -> dict:
                original = source[local] if local < len(source) else local
                segmentation = frames[local] if local < len(frames) else None
                blocks = []
                for block in tuple(getattr(segmentation, "blocks", ()) or ())[:64]:
                    y0, y1 = getattr(block, "y0", None), getattr(block, "y1", None)
                    offset = offsets[local] if local < len(offsets) else None
                    top = getattr(block, "top", None)
                    bottom = getattr(block, "bottom", None)
                    blocks.append({
                        "frame_rows": [y0, y1],
                        "page_rows": ([y0 + offset, y1 + offset]
                                      if isinstance(y0, int) and isinstance(y1, int)
                                      and isinstance(offset, int) else None),
                        "kind": getattr(block, "kind", None),
                        "complete": bool(getattr(block, "complete", False)),
                        "top_observed": bool(getattr(top, "observed", False)),
                        "bottom_observed": bool(getattr(bottom, "observed", False)),
                        "top_kind": getattr(top, "kind", None),
                        "bottom_kind": getattr(bottom, "kind", None),
                        "hearts": {
                            "frame_rows": [list(h) for h in tuple(getattr(block, "hearts", ()) or ())[:12]],
                            "page_rows": ([list((x, y + offset)) for x, y in
                                           tuple(getattr(block, "hearts", ()) or ())[:12]]
                                          if isinstance(offset, int) else None),
                        },
                    })
                runs = []
                for run in tuple(getattr(segmentation, "runs", ()) or ())[:64]:
                    runs.append({
                        "frame_rows": [getattr(run, "y0", None), getattr(run, "y1", None)],
                        "kind": getattr(run, "kind", None),
                        "widest_intruder_px": getattr(run, "widest_intruder_px", None),
                        "median_level_delta": getattr(run, "median_level_delta", None),
                    })
                record = {
                    "local_frame_index": local, "source_frame_index": original,
                    "offset_px": offsets[local] if local < len(offsets) else None,
                    "animation_marker": (animation_markers[local]
                                         if local < len(animation_markers) else None),
                    "video_mute_markers": [
                        {"x": getattr(marker, "x", None), "y": getattr(marker, "y", None),
                         "score": getattr(marker, "score", None)}
                        for marker in mute_markers
                        if getattr(marker, "frame_index", None) == local],
                    "blocks": blocks, "background_runs": runs,
                }
                if screenshot is not None:
                    record["screenshot"] = screenshot
                return record

            records = []
            saved = []
            for local in local_candidates:
                original = source[local]
                filename = f"item_index_refused_{dossier_id}_frame_{original}.png"
                (debug_dir / filename).write_bytes(photos[original])
                saved.append(filename)
                records.append(geometry_record(local, filename))

            # Screenshots remain capped at eight, but geometry is small and is the evidence a
            # replay actually needs.  Preserve every folded frame (bounded again per frame)
            # instead of silently omitting a non-cited fragment that made a repair back out.
            geometry_count = min(len(frames), len(source))
            all_frame_geometry = [geometry_record(local) for local in range(geometry_count)]

            def pair_record(local: int) -> dict:
                """Numeric raw evidence and local proposals for one adjacent frame pair."""
                shift = shifts[local]
                before, after = frames[local], frames[local + 1]
                strips = []
                for strip in tuple(getattr(shift, "strips", ()) or ())[:32]:
                    strips.append({
                        "frame_rows": [getattr(strip, "y0", None), getattr(strip, "y1", None)],
                        "state": getattr(strip, "state", None),
                        "delta_px": getattr(strip, "delta_px", None),
                        "score": getattr(strip, "score", None),
                        "runner_up": getattr(strip, "runner_up", None),
                        "stddev": getattr(strip, "stddev", None),
                        "search": list(getattr(strip, "search", ()) or ())[:2],
                    })
                proposals = []
                for kind, proposer in (
                    ("layout", lambda: _layout_repaired_shift(local, before, after, shift)),
                    ("edge", lambda: _edge_only_two_strip_shift(local, before, after, shift)),
                    ("structural_tail", lambda: _structural_tail_shift(
                        local, before, after, shift)),
                    ("exact_multi", lambda: _exact_multi_strip_shift(
                        local, before, after, shift)),
                ):
                    proposed, note = proposer()
                    if note is not None:
                        proposals.append({
                            "kind": kind, "delta_px": getattr(proposed, "delta_px", None),
                            "agreeing": getattr(proposed, "agreeing", None), "note": note[:1000],
                        })
                        if kind == "layout":
                            projected, projection_note = _project_to_exact_full_layout(
                                local, before, after, shift, proposed)
                            if projection_note is not None:
                                proposals.append({
                                    "kind": "measured_bridge_projection",
                                    "delta_px": getattr(projected, "delta_px", None),
                                    "agreeing": getattr(projected, "agreeing", None),
                                    "note": projection_note[:1000],
                                })
                bridge_shift, bridge_note = _measured_layout_bridge(
                    local, before, after, shift,
                    allow_full_layout_projection=True)
                if bridge_note is not None:
                    proposals.append({
                        "kind": "measured_bridge",
                        "delta_px": getattr(bridge_shift, "delta_px", None),
                        "agreeing": getattr(bridge_shift, "agreeing", None),
                        "note": bridge_note[:1000],
                    })
                return {
                    "pair": [local, local + 1],
                    "source_pair": [source[local], source[local + 1]],
                    "status": getattr(shift, "status", None),
                    "delta_px": getattr(shift, "delta_px", None),
                    "consensus_px": getattr(shift, "consensus_px", None),
                    "confidence": getattr(shift, "confidence", None),
                    "agreeing": getattr(shift, "agreeing", None),
                    "dissenting": getattr(shift, "dissenting", None),
                    "eligible": getattr(shift, "eligible", None),
                    "reason": str(getattr(shift, "reason", ""))[:1000],
                    "strips": strips,
                    "matched_delta_clusters": [
                        {"delta_px": delta, "voters": list(voters)}
                        for delta, voters in _matched_delta_clusters(shift)
                    ],
                    "structural_landmarks": [list(value)
                                             for value in _structural_landmarks(before, after)],
                    "observed_gutters": {
                        "before": [list(value) for value in _observed_gutters(before)],
                        "after": [list(value) for value in _observed_gutters(after)],
                    },
                    "local_proposals": proposals,
                    "v12_video_track_delta_px": track_deltas.get(local),
                }

            # Every pair is bounded to 32 numeric strip records; unlike screenshots this carries
            # no profile pixels or text.  Keeping measured pairs is essential: the fifth saved
            # refusal's first broken chain hid a later +207px measured majority whose exact
            # top/bottom/heart geometry was +221px, and v2 had discarded its strip bank.
            pair_count = min(len(shifts), max(0, len(frames) - 1), max(0, len(source) - 1))
            pair_evidence = [pair_record(local) for local in range(pair_count)]
            sidecar = f"item_index_refused_{dossier_id}_evidence.json"
            # The image payload is capped at eight frames; structured geometry is bounded by the
            # capture ceiling, 64 blocks/runs per frame, and 12 hearts/block.  Reason remains
            # short because the full prose is already in actions.jsonl.
            (debug_dir / sidecar).write_text(json.dumps({
                "schema_version": 6, "reason": str(reason)[:2000],
                "runtime": runtime_provenance, "frames": records,
                "all_frame_geometry": all_frame_geometry,
                "pair_evidence": pair_evidence}, separators=(",", ":")))
            return sidecar, saved
        except Exception:  # noqa: BLE001 -- disk trouble cannot change a hard refusal
            return None, []

    def _item_index_refused(self, photos: list[bytes], reason: str, index=None) -> str:
        """Best-effort forensic record for a refused enumeration, then return its reason.

        A normal ``capture`` record intentionally saves only its first frame: that is enough to
        identify the profile the ranker saw, but not enough to reconstruct a broken coordinate
        pair later in a long enumeration.  ``ItemIndex`` retains that pair evidence, so preserve
        it here while it still exists.  This helper must stay observational: `DebugLog` itself is
        best-effort, and the outer guard also protects a future log implementation or test double
        from ever turning a safe refusal into a live-run exception.
        """
        if self._dbg is None:
            return reason

        shifts = tuple(getattr(index, "shifts", ()) or ())
        steps_px = [getattr(shift, "delta_px", None) for shift in shifts]
        refused_pairs = []
        failing_pair = None
        for pair_index, shift in enumerate(shifts):
            delta_px = getattr(shift, "delta_px", None)
            if delta_px is not None:
                continue
            if failing_pair is None:
                failing_pair = pair_index
            states: dict[str, int] = {}
            for strip in tuple(getattr(shift, "strips", ()) or ()):
                state = getattr(strip, "state", None)
                if isinstance(state, str):
                    states[state] = states.get(state, 0) + 1
            refused_pairs.append({
                "pair": [pair_index, pair_index + 1],
                "status": getattr(shift, "status", None),
                "consensus_px": getattr(shift, "consensus_px", None),
                "confidence": getattr(shift, "confidence", None),
                "agreeing": getattr(shift, "agreeing", None),
                "dissenting": getattr(shift, "dissenting", None),
                "eligible": getattr(shift, "eligible", None),
                "strip_states": states,
                "reason": getattr(shift, "reason", None),
            })

        # A malformed/legacy index can lack shifts but still carry the chain's first unknown
        # offset.  Offset j is reached by pair j-1, hence this conversion back to frame indices.
        if failing_pair is None:
            for offset_index, offset in enumerate(tuple(getattr(index, "offsets", ()) or ())):
                if offset is None and offset_index:
                    failing_pair = offset_index - 1
                    break

        cited_local: list[int] = []
        named_in_reason: list[int] = []
        # Refusal prose is deliberately human-readable, so retain every frame it names.  Pair
        # ledger fields cover older/newer message wording that does not say "frame".
        for text in (reason, *tuple(getattr(index, "failures", ()) or ())):
            if isinstance(text, str):
                for match in re.finditer(r"\bframe(?:s)?\s+(\d+)(?:\s+and\s+(\d+))?", text):
                    named_in_reason.extend(int(v) for v in match.groups() if v is not None)
        cited_local.extend(named_in_reason)
        if failing_pair is not None:
            cited_local.extend((failing_pair, failing_pair + 1))
        for rec in refused_pairs:
            pair = rec.get("pair")
            if isinstance(pair, list):
                cited_local.extend(v for v in pair if isinstance(v, int))
                if (len(pair) == 2 and all(isinstance(v, int) for v in pair)):
                    # Preserve one anchor before and two frames after every refused pair.  The
                    # look-ahead is still subject to the existing eight-image cap, but catches a
                    # measured downstream contradiction: v2 stopped at frame 10 while frame 11
                    # was the first screenshot proving +207px should have been +221px.
                    cited_local.extend(range(
                        max(0, pair[0] - 1), min(len(photos), pair[1] + 3)))
        runtime_provenance = _item_index_runtime_provenance()
        sidecar, evidence_frames = self._save_item_index_refusal_evidence(
            photos, reason, index, cited_local, runtime_provenance,
            # The pair the refusal is named after outranks breadth when the eight-image cap
            # has to drop something.  A malformed/legacy index that yields no failing pair still
            # has the frames its own prose names, which is the same evidence by a weaker route.
            priority_local=((failing_pair, failing_pair + 1) if failing_pair is not None
                            else tuple(named_in_reason[:2])))

        before = after = None
        if failing_pair is not None and 0 <= failing_pair and failing_pair + 1 < len(photos):
            before, after = photos[failing_pair], photos[failing_pair + 1]
        elif index is None and photos:
            # An exception has no pair ledger.  Save the capture boundaries rather than claiming
            # they are the failing pair; it is still the evidence that existed at failure time.
            before, after = photos[0], (photos[-1] if len(photos) > 1 else None)

        fields = {
            "reason": reason,
            "photos": len(photos),
            "steps_px": steps_px,
            "refused_pairs": refused_pairs,
            "item_index_runtime": runtime_provenance,
        }
        if failing_pair is not None:
            fields["failing_pair"] = [failing_pair, failing_pair + 1]
        if sidecar is not None:
            fields["evidence_sidecar"] = sidecar
            fields["evidence_frames"] = evidence_frames
        try:
            self._dbg.action("item_index_refused", before=before, after=after, **fields)
        except Exception:  # noqa: BLE001 -- diagnostics must never alter a live-run refusal
            pass
        return reason

    def _note_enumeration_truncated(self, frames: int) -> None:
        """Say out loud that an ENUMERATION read hit its ceiling without reaching the bottom.

        `_ENUMERATION_CAPTURE_LIMIT`'s comment carries the derivation and why exceeding it is not
        a hard stop; this is the other half of that decision. Truncation was already recorded --
        `ItemIndex.truncated` -> `Profile.items_truncated` -> the model, per doc 5.7 -- but only
        the MODEL was told, and only as a flag beside a numbered list. The operator saw nothing,
        which is what makes "it truncates silently" a fair description even though a flag exists.

        So the console gets a sentence naming the frame count against the derived ceiling, and
        actions.jsonl gets its own record. What the operator does about it is a judgement call
        this method deliberately does not make for them: a profile past the ceiling still yields a
        usable numbered list of everything above the cut, and the item the model picks from it is
        inside the enumerated region by construction, so navigation is unaffected. What is lost is
        Connect material from the tail of the profile (doc 4's own reason Part B exists), which is
        a quality cost rather than a correctness one.

        Only ever called for a read that raised the ceiling. An ordinary 12-frame read that hits
        `scroll_captures` is the pre-existing `capture_truncated` case and is untouched.
        """
        # Names the LIMIT THIS READ ACTUALLY RAN AT, not just the constant it was derived from:
        # `_capture_limit_for_profile` jitters the ceiling upward per profile, so quoting the
        # bare constant misreports the read by up to two frames in the one message whose job is
        # to let the owner judge whether that constant is measured against the wrong profiles.
        applied = self._profile_capture_limit
        ceiling = (f"the derived ceiling of {_ENUMERATION_CAPTURE_LIMIT}" if applied is None
                   or applied == _ENUMERATION_CAPTURE_LIMIT else
                   f"this read's ceiling of {applied}, jittered up from the derived "
                   f"{_ENUMERATION_CAPTURE_LIMIT}")
        print(f"{self.spec.app}: this profile is longer than the enumeration read could cover -- "
              f"{frames} frame(s) at {ceiling} and the "
              f"bottom was never reached, so the numbered item list stops part way down the "
              f"profile and the model is told so. See _ENUMERATION_CAPTURE_LIMIT for the "
              f"derivation; if this is not rare, that constant is measured against the wrong "
              f"profiles.")
        if self._dbg is not None:
            try:
                self._dbg.action("capture_enumeration_truncated", frames=frames,
                                 ceiling=_ENUMERATION_CAPTURE_LIMIT,
                                 limit=self._profile_capture_limit,
                                 profile_name=self._identity_name)
            except Exception:  # noqa: BLE001 — debug logging must never break a capture
                pass

    def _invalidate_item_index(self, reason: str) -> None:
        """Drop the driver-owned index, with the reason a later reader would need.

        Doc 5.3: the table's "lifetime is exactly one profile, and it must be invalidated
        wherever `_current_sigs` is today, including the deck-advance path". A stale table would
        navigate by the previous profile's heart ordinals and then verify against the previous
        profile's crops -- doc 5.6's own addendum measures that at ordinal 1 nothing before the
        tap catches it, which is what makes correct invalidation load-bearing rather than tidy.

        Sets the reason as `_current_items_unavailable` rather than clearing it, so the state is
        never "no payload and no explanation": every reader gets either crops or a sentence.
        """
        self._current_item_index = None
        self._current_item_payload = None
        self._current_item_anchor = None
        self._current_items_unavailable = reason

    # --- capture (Signals #1: read the whole profile, human-paced) ------
    def _capture_current(self, should_stop=None) -> Profile | None:
        """Read the profile currently on screen, human-paced, returning its frames.

        STOP: `should_stop` (when supplied -- see DatingAppDriver.supports_interruptible_capture)
        is polled at the top of every read iteration and inside the read dwell, and a stop
        abandons the read and returns None. Before this, the ONLY stop check between one profile
        and the next lived in worker.py, on the line AFTER this call returned -- so a Stop pressed
        while a profile was being read was not seen until the whole read finished: 12 screencaps,
        11 humanized read-scrolls, and then a full scroll back to the top, measured at ~85s of
        visible scrolling after the operator asked it to stop.

        Abandoning a read costs nothing: no decision has been made, nothing has been written, and
        None is already this method's established "nothing usable here" answer (a like sheet is
        up, a static screen, a mid-capture profile split), which both worker loops handle by
        recapturing -- and which, with the stop event set, means they simply leave their loop.
        The partial frames are dropped rather than returned precisely because a half-read profile
        must never reach the ranker as if it were a whole one.

        ENUMERATION (ops/OPENER-REDESIGN.md 5.2/5.3/5.5): an AUTO read is also the pass that
        builds the profile's item index. It confirms the scroll top affirmatively before the
        first gesture, sizes every step against the card spacing the current frame shows instead
        of a screen fraction, and folds the frames it kept into the driver-owned index plus the
        numbered crops the opener sends. See the block of `_item_enumeration_*` /
        `_index_captured_items` methods immediately above for the reasoning, including why every
        failure in there is a sentence on the Profile rather than an exception out of here: these
        frames have TWO consumers with independent needs, and the one that wants faces is not the
        one that wants item numbers.
        """
        photos: list[bytes] = []
        self._current_sigs = []
        self._current_capture_truncated = False   # reset: see the for/else below
        self._capture_scrolls = 0     # reset: _scroll_to_top must undo THIS capture, not a stale one
        self._capture_scroll_ledger = []
        # Identity anchor (see _identity_of): reset every capture, unconditionally, so a
        # profile with no identity_band declared or one that never scrolls far enough to
        # reveal the sticky header never inherits a STALE signature from the profile before
        # it -- that would be worse than no anchor at all (a false 'same' against the wrong
        # profile).
        self._identity_top_sig = None
        self._identity_sig = None
        self._identity_name = None
        self._identity_anchor_confirmed = False
        self._identity_anchor_frame = None
        self._identity_anchor_frame_index = None
        # Same reasoning as the three resets just above: a cached OCR read is keyed on frame
        # bytes, not profile identity, so a coincidental byte-identical crop from the NEXT
        # profile (unlikely but not impossible for a mostly-blank band) could otherwise return
        # a stale answer. See _ocr_band_cache's own comment in __init__.
        self._ocr_band_cache = {}
        self._current_capture_split = False   # set if the deck advanced mid-capture; see the loop
        self._capture_split_frame = None      # rejected trigger frame; retained for debug evidence
        self._capture_split_evidence = {}     # scalar-only evidence paired with that frame
        # Doc 5.3's table dies with the profile it described. Reset HERE, alongside
        # _current_sigs and the identity anchors above and for exactly their reason: the frames
        # about to be read belong to a different person, so anything left over from the last
        # read is a table pointing at somebody else's card.
        self._invalidate_item_index(
            "this profile's read has not finished, so nothing has been enumerated for it yet")
        # ENUMERATION (doc 5.2/5.3/5.5), decided BEFORE the loop because it sets the ceiling.
        # `enumeration_reason` is "" while the read is still on track to produce an item index
        # and a sentence the moment it is not -- the first sentence wins, so a later step's
        # failure never overwrites the reason the read stopped enumerating in the first place.
        enumeration_reason = self._item_enumeration_blocker()
        if not enumeration_reason:
            if should_stop is not None and should_stop():
                # The loop's own first check returns None one line below, so the gate's answer
                # would be discarded -- and the gate costs a full ADB round-trip, which is
                # exactly the latency a Stop is not supposed to wait through.
                enumeration_reason = ("the run is stopping, so this profile was never "
                                      "enumerated")
            else:
                enumeration_reason = self._confirm_enumeration_top()
        enumerating = not enumeration_reason
        # Audit fix, "BUG 3" (2026-08-12): `enumerating` can flip False MID-LOOP (a scroll that
        # cannot be sized ends the enumeration but lets the read finish at the ordinary cadence
        # -- see the `except (ScrollStepError, SegmentationError)` branch below), but
        # `_profile_capture_limit` is fixed for the whole capture right here, before that can
        # happen. So whether the RANKER needs its frames thinned back down (see
        # _ranker_frames_from_enumeration) depends on whether the ceiling was raised for this
        # read at all, not on whether enumeration was still running when the read stopped --
        # a read that raised the ceiling to 48 and then fell back to the ordinary cadence at
        # frame 20 can still walk all the way to frame 48 at that ordinary cadence, and the
        # ranker must not see all 48 of those either.
        enumeration_ceiling_raised = enumerating
        self._profile_capture_limit = self._capture_limit_for_profile(
            _ENUMERATION_CAPTURE_LIMIT if enumerating else None)
        # The smallest heart-bearing card spacing measured anywhere on THIS profile so far; see
        # _plan_enumeration_step. None until the first frame that can measure one.
        enum_min_spacing_px: int | None = None
        # A short contiguous contradictory run can continue at the corpus-minimum cadence, but
        # only so item_index gets a consecutive sequence in which to independently omit it and
        # remeasure its direct bridge. A fifth contradiction ends enumeration as before.
        enum_segmentation_fallback_frames: list[int] = []
        read_dwell_s_total = 0.0
        seen = set()
        # Consecutive pairs frameshift has MEASURED as exactly 0px since the last frame that
        # actually moved. See _static_pair_is_the_bottom for why the byte-identical `seen` test
        # below cannot see this profile's bottom on its own.
        static_pairs = 0
        prev_ds = None                  # last APPENDED frame's downsample, for that probe's gate
        for i in range(self._profile_capture_limit):
            if should_stop is not None and should_stop():
                # Checked BEFORE the screencap, so a stop that lands during the previous
                # dwell/scroll costs one poll rather than another full ADB round-trip. The
                # partial read is discarded (see the docstring); the scroll ledger is left
                # alone so _scroll_to_top still knows how far down the card actually is.
                self._note_capture_aborted(len(photos))
                return None
            frame = self._screencap()
            # A comment sheet is not profile content.  In observe mode it can be left open
            # while Hinge is still composing/sending a like; treating its changing pixels as
            # a card and then read-scrolling would move the sheet underneath the operator.
            # Stop before recording this frame or issuing another scroll.  In particular this
            # makes a capture that STARTS on a sheet completely input-free.
            if self._observe_like_sheet_visible(frame):
                return None
            ds = _downsample(frame)                    # None if PIL/numpy unavailable or undecodable
            # A profile boundary reached MID-CAPTURE. The deck can advance while this loop is
            # still reading -- Hinge draws no on-screen busy overlay (worker.py's WAIT cue
            # lives on the hub, not the phone), and the human's finger and the bot's own
            # digitizer are concurrent input streams (see touchwatch.py) -- so a tap landing
            # here is a real possibility, not a hypothetical. Nothing else in this loop would
            # notice: it stops only on a repeated frame or the ceiling, so the frames after
            # the advance get appended as though they were more of the SAME person, and the
            # returned Profile then carries two different people's photos into one mean-pooled
            # embedding and one stored label. Once this capture has locked an identity, a
            # frame whose header matches NEITHER that identity nor the scroll-top chrome is a
            # different card: stop before appending it and mark the capture, so the worker
            # discards and recaptures instead of scoring a chimera.
            if self._identity_sig is not None:
                band = _band(frame, self.identity_band)
                if band is not None:
                    identity_dist = _band_dist(band, self._identity_sig)
                    top_dist = (None if self._identity_top_sig is None
                                else _band_dist(band, self._identity_top_sig))
                    if identity_dist < self.change_threshold:
                        # A second matching observation promotes the provisional first header
                        # to an identity anchor. Keep the later, settled pixels and OCR read:
                        # they are better evidence than the frame captured immediately after
                        # the first scroll animation.
                        if not self._identity_anchor_confirmed:
                            self._identity_anchor_confirmed = True
                            self._identity_sig = band
                            self._identity_name = self._ocr_band(frame, self.identity_band)
                            self._identity_anchor_frame = frame
                            self._identity_anchor_frame_index = len(photos)
                    elif top_dist is None or top_dist >= self.change_threshold:
                        # Before the first header has been reproduced, a mismatch is ambiguous:
                        # it can be a real card boundary, or the stable header replacing a
                        # transient strip that was captured just after scrolling. Resolve that
                        # ambiguity with the independently measured content motion. Adjacent
                        # frames from one scrolled profile align under a vertical shift; a new
                        # profile does not. Once confirmed, the identity band remains the hard
                        # boundary it was before this fix.
                        content_match = False
                        content_shift = None
                        content_overlap = None
                        if (not self._identity_anchor_confirmed and ds is not None
                                and self._current_sigs and self._current_sigs[-1] is not None
                                and ds.shape == self._current_sigs[-1].shape):
                            try:
                                content_match, content_shift, content_overlap = (
                                    _vertical_shift_match(
                                        ds, self._current_sigs[-1],
                                        threshold=self.change_threshold,
                                        rows=_content_rows(self.content_band, ds.shape[0])))
                            except Exception:  # noqa: BLE001 -- corroboration fails closed
                                content_match = False
                        if content_match:
                            old_anchor = self._identity_anchor_frame
                            old_index = self._identity_anchor_frame_index
                            old_name = self._identity_name
                            self._identity_sig = band
                            self._identity_name = self._ocr_band(frame, self.identity_band)
                            self._identity_anchor_frame = frame
                            self._identity_anchor_frame_index = len(photos)
                            if self._dbg is not None:
                                self._dbg.action(
                                    "identity_anchor_replaced", before=old_anchor, after=frame,
                                    old_anchor_frame_index=old_index,
                                    new_anchor_frame_index=len(photos),
                                    old_profile_name=old_name,
                                    new_profile_name=self._identity_name,
                                    identity_dist=round(identity_dist, 3),
                                    top_dist=(None if top_dist is None else round(top_dist, 3)),
                                    content_shift=content_shift,
                                    content_overlap_rows=content_overlap)
                            # This frame is still ordinary profile content. It is appended below,
                            # and the next matching header observation will confirm the anchor.
                        else:
                            # Preserve the exact rejected frame and the distances that made this
                            # a boundary.  The old capture_split record saved only frame 0, so a
                            # real card advance and a transient sticky-header animation were
                            # impossible to distinguish after the fact.
                            evidence = {
                                "trigger_frame_index": len(photos),
                                "captured_frames": len(photos),
                                "read_scrolls": len(self._capture_scroll_ledger),
                                "identity_dist": round(identity_dist, 3),
                                "top_dist": None if top_dist is None else round(top_dist, 3),
                                "identity_anchor_frame_index": self._identity_anchor_frame_index,
                                "identity_anchor_confirmed": self._identity_anchor_confirmed,
                                "identity_anchor_name": self._identity_name,
                                "content_match": content_match,
                                "content_shift": content_shift,
                                "content_overlap_rows": content_overlap,
                            }
                            try:
                                top_verdict = confirm_scroll_top(
                                    frame, identity_band=self.identity_band)
                            except ScrollTopError as exc:
                                evidence["scroll_top_state"] = "unreadable"
                                evidence["scroll_top_reason"] = str(exc)
                            else:
                                evidence["scroll_top_state"] = top_verdict.state
                                evidence["scroll_top_distance"] = (
                                    None if top_verdict.distance is None
                                    else round(top_verdict.distance, 3))
                                evidence["scroll_top_reason"] = top_verdict.reason
                            self._capture_split_frame = frame
                            self._capture_split_evidence = evidence
                            self._current_capture_split = True
                            break
            sig = _frame_sig(frame)
            if sig in seen:
                # A repeat frame = reached the bottom (the screen stopped changing). Only treat
                # the very FIRST scroll repeating as a static screen (Out of Profiles / Loading)
                # when the frame actually DECODES. An undecodable repeat is a degraded/wedged
                # capture (e.g. a dropped device returning empty/truncated screencap on exit 0) —
                # it must fall through to the H1 guard below, not masquerade as an empty deck
                # (which would silently livelock the observe worker instead of stopping cleanly).
                if i == 1 and ds is not None:
                    return None
                break
            # The bottom, when an ANIMATION is repainting the page. `sig` above is a whole-frame
            # 24x24 compared for EXACT byte equality, so one autoplaying video card keeps every
            # frame "new" forever and the bottom signal never fires. Measured live 2026-08-16
            # (Grace): the page saturated at frame 37 and the loop still ran to its 64-frame
            # ceiling, issuing 26 futile swipes at an already-bottomed profile and then reporting
            # the profile as LONGER than the read could cover -- the exact opposite of the truth,
            # into the hub banner and BigQuery's capture_truncated column.
            #
            # frameshift is the authority instead of a looser pixel threshold because it is the
            # one comparator here that REFUSES when it cannot tell: a measured 0 means 3+ strips
            # independently agreed to the pixel that nothing moved, and anything less certain
            # returns None and simply keeps the read going. That asymmetry is the whole safety
            # argument -- a false 0 would truncate a real profile, so only an affirmative,
            # quorate measurement is allowed to stop the loop, and it must happen TWICE in a row
            # so one dropped or swallowed gesture cannot end a read on its own.
            #
            # The count is of failed GESTURES, not of idle frames, which is why this does not
            # short-circuit the dwell and scroll below: each static pair is measured across a
            # read-scroll that was actually issued, so two of them mean two real swipes moved
            # nothing. Skipping ahead to the next screencap instead would prove only that the
            # screen was idle while nothing was asked of it.
            if photos and _static_pair_is_the_bottom(
                    photos[-1], frame, self.content_band,
                    before_ds=prev_ds, after_ds=ds):
                static_pairs += 1
                if static_pairs >= _STATIC_PAIRS_FOR_BOTTOM:
                    # Deliberately not appended: a measured 0px translation of the previous
                    # frame carries no page content the index has not already seen.
                    break
            else:
                static_pairs = 0
            seen.add(sig)
            photos.append(frame)
            prev_ds = ds
            # Lock the identity anchor INLINE as frames arrive, not retroactively after the
            # loop. The boundary check above can only fire once _identity_sig exists, and a
            # post-hoc scan would establish it only after the foreign frames had already been
            # appended -- i.e. exactly too late to keep them out.
            if self.identity_band is not None:
                band = _band(frame, self.identity_band)
                if band is not None:
                    if self._identity_top_sig is None:
                        self._identity_top_sig = band     # frame 0: the app's scroll-top chrome
                    elif (self._identity_sig is None
                            and _band_dist(band, self._identity_top_sig) >= self.change_threshold):
                        self._identity_sig = band         # first frame showing the sticky header
                        self._identity_name = self._ocr_band(frame, self.identity_band)
                        self._identity_anchor_frame = frame
                        self._identity_anchor_frame_index = len(photos) - 1
            # Keep _current_sigs index-ALIGNED with photos: append ds even when None (an
            # undecodable frame). _locate_target_heart looks its target up here — a gap would
            # desync the two and target the WRONG photo. Consumers below filter/guard the Nones.
            #
            # This list is a CAPTURE-ORDER (0-based, per scroll frame) space, and as of
            # 2026-08-12 the opener no longer speaks it: OpenerResult.item_index is 1-based over
            # the numbered ITEMS the model was shown (ops/OPENER-REDESIGN.md 5.1/5.7). Nothing
            # converts between the two here, and nothing may: the crossing happens exactly once,
            # in opener.service.OpenerPick.capture_order_index, which knows which list the
            # model's number counted and refuses rather than guessing when it cannot say. Doc
            # 5.3's driver-owned translation table (model item -> heart ordinal) now EXISTS
            # alongside this list -- see _current_item_payload, built below from these same
            # frames -- but it is a different table in a different space, and nothing converts
            # between the two: the crop-shape index resolves to a HEART ORDINAL, never to a
            # position in this frame list.
            #
            # Why this list cannot police the difference itself, since it looks like it could:
            # its only bound is its own length, and there are always more frames than items (24
            # frames for 9 items on the calibration capture), so every out-of-space value it
            # could receive looks perfectly in range. That is not a missing check here, it is
            # why the space has to be stated by the producer.
            self._current_sigs.append(ds)

            if i < self._profile_capture_limit - 1:
                complexity_hint = None
                if ds is not None:
                    try:
                        complexity_hint = float(ds.std()) / 255.0
                    except Exception:  # noqa: BLE001 — hint is optional, capture is not
                        pass
                dwell, frac, x_frac = self._sample_read_step(i, complexity_hint)
                if enumerating:
                    # THE CLOSED LOOP (doc 5.5 / 5.10.1). The distance stops being a screen
                    # fraction and becomes a fraction of the card actually in front of us,
                    # because the two must not alias: a step near the item spacing makes "the
                    # same heart moved" and "the next heart arrived" geometrically
                    # indistinguishable, which no better estimator can fix. The lane and the
                    # dwell keep coming from the behaviour policy exactly as they did.
                    #
                    # A refusal here normally ENDS THE ENUMERATION and lets the read finish at
                    # the ordinary cadence. The sole bounded exception is one short, contiguous
                    # contradictory segmentation run, carried at the corpus-minimum cadence so
                    # item_index can independently omit it and rebuild across a measured bridge. Any
                    # other refusal builds no item payload: its reason is recorded, crops are
                    # never produced, and nothing downstream can mistake the frames for a
                    # numbered list.
                    try:
                        step = self._plan_enumeration_step(
                            frame, x_frac, enum_min_spacing_px,
                            allow_segmentation_failure_fallback=
                            len(enum_segmentation_fallback_frames)
                            < MAX_SEGMENTATION_FALLBACK_FRAMES)
                    except (ScrollStepError, SegmentationError) as exc:
                        enumerating = False
                        enumeration_reason = (
                            f"the enumeration scroll could not be sized against frame "
                            f"{len(photos) - 1} of this profile ({type(exc).__name__}: {exc})")
                    else:
                        frac, x_frac = step.frac, step.x_frac
                        if step.basis == STEP_SEGMENTATION_FALLBACK:
                            # This is only reachable inside the explicitly capped bad-frame run:
                            # the next call after the cap disables the opt-in and raises normally.
                            # Save every raw frame permanently for the direct-bridge audit.
                            fallback_frame = len(photos) - 1
                            enum_segmentation_fallback_frames.append(fallback_frame)
                            if self._dbg is not None:
                                self._dbg.action(
                                    "enumeration_segmentation_fallback", before=frame,
                                    keep_before=True,
                                    frame_index=fallback_frame,
                                    fallback_frame_indices=enum_segmentation_fallback_frames,
                                    reason=step.reason,
                                    profile_min_spacing_px=enum_min_spacing_px,
                                    step_px=step.step_px,
                                    sized_against_px=step.sized_against_px)
                        if step.spacing.measured:
                            enum_min_spacing_px = (
                                step.spacing.px if enum_min_spacing_px is None
                                else min(enum_min_spacing_px, step.spacing.px))
                # The read dwell is the single longest stretch of this loop (dwell_s=1.1
                # humanized, x11), so it is where a Stop most often lands. Credit
                # read_dwell_s_total only with time actually spent: this counter is the
                # Signals behaviour-#1 "did we really read the profile" metric, and an
                # interrupted capture that reported the full sampled dwell would be claiming
                # reading time that never happened.
                started = time.monotonic()
                completed = self._interruptible_sleep(dwell, should_stop)
                read_dwell_s_total += dwell if completed else max(0.0, time.monotonic() - started)
                if not completed:
                    self._note_capture_aborted(len(photos))
                    return None
                self._scroll_down_one(frac, x_frac)
                # The dwell above intentionally happens BEFORE the read gesture: it is the
                # time spent looking at content before moving on.  This separate wait exists
                # solely to let Hinge finish the scroll/header animation before the next frame
                # can become an identity anchor.  It remains stop-aware, and is deliberately
                # not included in read_dwell_s_total.
                if not self._interruptible_sleep(human_delay(_READ_SCROLL_SETTLE_S), should_stop):
                    self._note_capture_aborted(len(photos))
                    return None
        else:
            # The loop ran out its full range() without ever finding a repeated frame (the
            # signal that the profile's true bottom was reached) -- this profile has MORE
            # content than self._profile_capture_limit could read. _current_sigs then does not
            # cover the whole scrollable range, so a human reading further down than the bot
            # did can land on pixels _vertical_shift_match has genuinely never seen (not a
            # granularity gap this project can shift-search its way out of). Recorded for
            # wait_for_decision's PASS diagnostic and Profile.meta below -- monitoring only,
            # never changes what gets decided.
            self._current_capture_truncated = bool(photos)
            # ...except that an ENUMERATION read hitting its ceiling is worth saying out loud,
            # because that ceiling is derived from measured page geometry and a profile past it
            # is either genuinely unusual or evidence the derivation is wrong. See
            # _note_enumeration_truncated and _ENUMERATION_CAPTURE_LIMIT.
            if enumeration_ceiling_raised and photos:
                self._note_enumeration_truncated(len(photos))
        # H1: in a real run (open_session validated PIL/numpy), every frame should
        # downsample. If none did, decode is broken at runtime (PIL/numpy failure OR a wedged
        # device returning empty/truncated screencap) — refuse to continue in a degraded mode
        # where scroll-detection is off and manual scrolls mislabel as PASS / the worker
        # silently no-ops forever. (The decodable-static case returned None above.)
        if self._observe_ready and photos and not any(s is not None for s in self._current_sigs):
            raise DriverClosed(
                "screencap frames could not be decoded (PIL/numpy runtime failure or wedged "
                "device); refusing to run degraded — it would corrupt training labels")
        # Identity anchor (layer 1 of the observe-mode redesign): current_profile()/next_profile()
        # both go through here, and _scroll_to_top() puts the PREVIOUS capture at the top before
        # this one starts reading -- so photos[0] is guaranteed to be at scroll-top, i.e. showing
        # the app's own scroll-top chrome (Hinge's filter-chips row) in the identity band, never a
        # real per-profile header. The sticky header only appears once the card is scrolled, so
        # the true identity signature is the first LATER frame whose band reads as meaningfully
        # different from that chrome (>= change_threshold) -- see HINGE_SPEC's identity_band
        # comment for the measured 0.00-vs-17.95 separation this relies on. Left None when the
        # profile was never scrolled far enough during THIS capture to reveal it; _identity_of
        # then reports 'unknown' rather than manufacturing a false 'same'/'new' from nothing.
        identity_seen = self._identity_sig is not None    # locked inline in the loop above
        if self._current_capture_split:
            # This capture spans two profiles: the deck advanced part-way through the read.
            # Returning the frames anyway would hand worker.py a Profile whose photos are two
            # different people, which it would mean-pool into ONE embedding and store against
            # ONE label -- silently poisoning the training set with a face that does not exist.
            # Return None, the same signal current_profile() already uses for "nothing usable
            # here"; worker.py's _observe_loop treats it as `continue` and recaptures the card
            # that is actually on screen now (worker.py:215-216), which is exactly right.
            if self._dbg is not None and photos:
                self._dbg.action("capture_split", before=photos[0],
                                 after=self._capture_split_frame,
                                 anchor=self._identity_anchor_frame, photos=len(photos),
                                 profile_name=self._identity_name,
                                 **self._capture_split_evidence)
            print(f"{self.spec.app}: the deck advanced while reading this profile "
                  f"(captured {len(photos)} frame(s) spanning two cards); discarding and "
                  f"recapturing rather than mixing two people into one label.")
            # THE DECK-ADVANCE PATH, named explicitly by doc 5.3 as a place the table must be
            # invalidated. Nothing was indexed on this path anyway (the build below never runs),
            # but the point is that the state must not be left describing whatever was here
            # before: these frames span two people, so no index over them could be right, and an
            # index from the PREVIOUS profile surviving into the recapture is the exact stale
            # table doc 5.3 says a wrong like is built from.
            self._invalidate_item_index(
                "the deck advanced while this profile was being read, so the capture spans two "
                "cards and nothing about it can be enumerated")
            return None
        # THE INDEX, from the frames that were actually kept (doc 5.2/5.3). Only when the whole
        # read was an enumeration read: a capture that stopped enumerating half way through has
        # a mixed cadence, and folding those frames would be indexing a page nobody scrolled.
        if enumerating:
            enumeration_reason = self._index_captured_items(photos)
        if enumeration_reason:
            self._invalidate_item_index(enumeration_reason)
            print(f"{self.spec.app}: no numbered item list for this profile -- "
                  f"{enumeration_reason}")
        else:
            self._current_items_unavailable = ""
        payload = self._current_item_payload
        # THE RANKER'S COPY (audit fix, "BUG 3", 2026-08-12). `photos` above -- and everything
        # already built from it (`_current_sigs`, the item index, the crops) -- stays the FULL
        # enumeration-cadence capture; only what goes to `Profile.photos` below is thinned back
        # down to what the ranker saw before Part B raised the enumeration ceiling. See
        # _ranker_frames_from_enumeration's docstring for the full reasoning and what is NOT
        # exactly reproduced. A capture that never raised the ceiling (observe, or any read that
        # was blocked from enumerating before the first frame) is untouched: `ranker_photos is
        # photos` in that case, so nothing about this file changes for it.
        ranker_photos = (
            _ranker_frames_from_enumeration(photos, self.scroll_captures)
            if enumeration_ceiling_raised else photos)
        if self._dbg is not None and photos:
            # profile_name, not name: DebugLog.action's own first positional parameter IS
            # called `name` (the action-type string, "capture" here) -- a fields key of
            # literally `name` would collide with it (TypeError: multiple values for
            # argument 'name'). Same reason wait_for_decision's debug records below use
            # profile_name too.
            self._dbg.action("capture", before=photos[0], photos=len(photos),   # first frame = who was scored
                             capture_truncated=self._current_capture_truncated,
                             identity_seen=identity_seen,
                             identity_confirmed=self._identity_anchor_confirmed,
                             profile_name=self._identity_name,
                             # The enumeration's own verdict, so a debug replay can tell "this
                             # profile was never enumerated" from "it was, and here is what it
                             # found" without re-running any vision.
                             items=(payload.item_count if payload is not None else 0),
                             item_context=(payload.context_count if payload is not None else 0),
                             item_translation=(list(payload.translation)
                                               if payload is not None else []),
                             item_manifest=(self._item_payload_debug_manifest(
                                 payload, self._current_item_index)
                                 if payload is not None and self._current_item_index is not None
                                 else []),
                             items_unavailable=self._current_items_unavailable or None,
                             # "BUG 3" fix: only present (and only ever < photos) when this read
                             # raised the enumeration ceiling and the ranker's copy was thinned
                             # back down for it -- absent, not equal to photos, on every ordinary
                             # (non-enumerating) read, so a debug replay can tell "this read never
                             # needed thinning" from "it did, and here is what survived".
                             ranker_photos=(len(ranker_photos)
                                            if enumeration_ceiling_raised else None))
        return Profile(
            photos=ranker_photos,
            prompts=[],
            meta={
                "app": self.spec.app,
                "capture_frames": len(photos),
                "read_scrolls": len(self._capture_scroll_ledger),
                "read_dwell_s_total": read_dwell_s_total,
                "capture_truncated": self._current_capture_truncated,
            },
            # THE ITEM PAYLOAD (doc 5.7's request shape), copied out of the driver's own table
            # as plain bytes. The driver keeps the index, the crops' signatures and
            # `translation` (model item number -> heart ordinal) for navigation and doc 5.6's
            # verification; the Profile carries only what the MODEL is shown, because that is
            # all the opener layer has any business seeing.
            #
            # `payload.images` is deliberately not used here even though it is the same bytes in
            # the same order: it concatenates the two tiers, and the split is what stops a
            # context crop from being numbered. Exactly one of `items` and `items_unavailable`
            # is ever non-empty.
            name=self._identity_name or "",
            items=tuple(c.image for c in payload.items) if payload is not None else (),
            item_context=tuple(c.image for c in payload.context) if payload is not None else (),
            items_truncated=bool(payload.truncated) if payload is not None else False,
            items_unavailable=self._current_items_unavailable,
        )

    def next_profile(self, *, should_stop=None) -> Profile | None:
        if self.out_of_profiles():
            return None
        if not self._session_top_done:
            self._ensure_session_top(should_stop)   # see its docstring: once per session
        profile = self._capture_current(should_stop)
        self._recover_capture_split(should_stop)
        return profile

    def _recover_capture_split(self, should_stop=None) -> None:
        """Restore scroll top only after a capture proved the deck advanced mid-read.

        Both public capture paths return ``None`` for a split so their workers discard the
        mixed profile and recapture the new card.  The forward-scroll ledger still describes
        where that new card is positioned, however.  Without this narrow recovery either path
        would immediately seed the next capture's top/chips anchor from a sticky header and the
        affirmative scroll-top gate would (correctly) refuse enumeration.

        Do not generalise this to every ``None`` capture: an open Send Like sheet and an
        operator Stop both intentionally return None without touching the screen.  The flag is
        set only by the identity-proven split branch in `_capture_current`; `_scroll_to_top`
        itself remains stop-aware, so a Stop that arrives before or during recovery issues no
        further gesture and leaves the phone where the owner asked.
        """
        if self._current_capture_split:
            self._scroll_to_top(should_stop)

    def current_profile(self, *, should_stop=None) -> Profile | None:
        # Observe mode's only capture path: worker.py prints "READY - swipe this profile"
        # right after this returns, so the phone must be back at the top for that swipe to
        # land on the card the operator actually read (bug 2). next_profile() (auto) does NOT
        # get this: like() already calls _scroll_to_top() itself before acting, so adding it
        # here too would just be a redundant extra scroll on that path.
        with self._observe_input_lease("current_profile"):
            if self.out_of_profiles():
                return None
            if not self._session_top_done:
                self._ensure_session_top(should_stop)   # see its docstring: once per session
            profile = self._capture_current(should_stop)
            if profile is not None:
                # Also stop-aware: this unwind is roughly half of the total per-profile dead time
                # (11 undo-swipes, each with two settle screencaps), and it is the part the
                # operator actually watches happen after pressing Stop. Reaching it at all means
                # the read itself completed, so the profile IS returned and worker.py's own stop
                # check on the next line decides what happens to it -- an interrupted unwind never
                # discards a complete capture, it only declines to keep scrolling.
                self._scroll_to_top(should_stop)
            else:
                self._recover_capture_split(should_stop)
            return profile

    def current_profile_reviewed(self, *, should_stop=None) -> Profile | None:
        """Capture one card for the explicit non-manual Observe action bridge.

        Manual Observe publishes a card only after returning it to scroll top, because that is
        the state a person expects to inspect and act from.  A reviewed bridge has no second
        controller touching the phone: its Worker waits on an exact mailbox capability instead.
        Retain the completed capture at its own final scrolled frame so the driver-owned item
        anchor and sticky-header identity can be used directly by ``navigate_to_item``.  This
        avoids trying to reconstruct an old page position from a top-unwound card.

        This is deliberately a distinct capability rather than a flag on ``current_profile``.
        Worker selects it only for its explicit non-manual bridge; direct/manual callers keep
        the existing top-unwind contract.  `_capture_current` still performs the affirmative
        scroll-top gate before it creates any absolute item index.  If that gate, the capture,
        or indexing refuses, there is no actionable payload and no reviewed heart can follow.
        """
        with self._observe_input_lease("current_profile_reviewed"):
            if self.out_of_profiles():
                return None
            # Unlike the manual path, a bridge pre-tap refusal may leave this same card at an
            # indexed intermediate position and immediately request a recapture.  Re-establish
            # the top attempt before *every* reviewed capture, not merely once per process, so
            # `_capture_current` can build absolute heart ordinals only from a fresh affirmative
            # top gate.  On the common already-top next card this is one bounded settle probe.
            self._ensure_session_top(should_stop)
            profile = self._capture_current(should_stop)
            if profile is None:
                self._recover_capture_split(should_stop)
            return profile

    def out_of_profiles(self) -> bool:
        # LIVE-VERIFY: the empty-deck screen is gated until the profile is finished,
        # so we can't yet match its signature. Until calibrated this returns False
        # and the run stops via the rate limiter (auto) or the operator (observe).
        return False

    # --- is something STANDING BETWEEN us and the deck? -----------------
    # A different question from out_of_profiles() above (which means the deck ran dry — normal
    # end of supply, nothing wrong). See DatingAppDriver.blocked_reason for the contract and
    # _PAYWALL_MATCH_THRESHOLD for the 2026-08-11 incident these exist for. Everything in this
    # block is READ-ONLY perception: it screencaps, template-matches and OCRs, and NEVER taps,
    # swipes or types. That is not a style preference here — the one screen it recognises is a
    # PURCHASE screen, and observe mode is strictly passive besides.

    def _paywall_visible(self, frame: bytes) -> bool:
        """Whether `frame` is an app upgrade/paywall screen, by the spec's "paywall" template.

        For Hinge that template is the "HingeX" tab wordmark of the out-of-free-likes upgrade
        screen. The tab CHROME was chosen over the two more obvious candidates because it is the
        only fixed part of that screen: the hero image is rotating marketing artwork (a template
        cut from it would stop matching the next time Hinge changes the campaign), and the
        benefit list below it scrolls, as does the price string in the bottom CTA. The tab bar
        holds still — MEASURED at y 266..318 of 2400 in the 2026-08-11 reference dump, which is
        also what makes the position gate below meaningful.

        MEASURED discrimination with cv2.TM_CCOEFF_NORMED (Pixel 7a, 1080x2400, 2026-08-11):
        1.000 on the live paywall; 0.965..1.000 with the frame perturbed over gain 0.35..1.4 and
        bias -30..+90 (standing in for the tab rendering purple-when-active vs
        grey-when-inactive); and a maximum of 0.4903 over ALL 88 real non-paywall frames of the
        hung run this fix comes from. _PAYWALL_MATCH_THRESHOLD (0.75) sits with a wide margin on
        both sides.

        An app whose spec declares no "paywall" template at all (every app but Hinge today)
        gets None from `_template`, `_match_glyph`'s own None-template guard returns no hits,
        and this is simply always False — no behaviour change for Bumble or the web drivers.

        False on ANY exception: inability to PROVE a paywall must never break observation. The
        cost of a false negative here is only that the generic stuck-screen watchdog stops the
        run 90s later with a vaguer message; the cost of raising would be a crashed run.
        """
        try:
            hits = _match_glyph(frame, self._template("paywall"), side="any",
                                threshold=_PAYWALL_MATCH_THRESHOLD)
            if not hits:
                return False
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return False
            height = image.shape[0]
            return any(y <= height * _PAYWALL_MAX_Y_FRAC for _x, y in hits)
        except Exception:  # noqa: BLE001 — see the docstring: never break observation over this
            return False

    def _paywall_headline(self, frame: bytes) -> str | None:
        """Best-effort OCR of the paywall's headline (spec.paywall_headline_band), or None.

        NEVER load-bearing, and deliberately not a detector: `_paywall_visible` above has
        already decided whether this screen IS a paywall, from pixels alone. This only refines
        the operator-facing message from "the deck is not available" into "out of free likes for
        today", so no `tesseract` on PATH costs message specificity and nothing else.

        The ordinary `_ocr_band` recipe cannot read this band — MEASURED 2026-08-11, it returns
        garbage ("“Tikes for today") — because the headline is WHITE text over a PHOTOGRAPH,
        where every other band this driver OCRs is dark text on flat chrome. Hence
        `white_text_threshold`: binarize at a high luminance and invert, so tesseract gets the
        black-on-white it wants. See that parameter's own documentation on `_ocr_band` for the
        measured before/after, and `_PAYWALL_OCR_WHITE_MIN` for why 200.

        `psm="6"` (a uniform block of text) rather than the `_ocr_band` default of `"7"` (a
        single line): the headline wraps onto two lines ("You're out of free / likes for today"),
        and forcing multi-line content through a single-line segmentation is exactly the failure
        already documented for identity_top_name_band.

        Note that this inherits `_ocr_band`'s `observe_name_ocr` gate: an operator who has turned
        host-side OCR off gets None here too, and therefore the generic message. That is the
        honest outcome — with OCR off, nothing has actually read the headline — and it is not a
        loss of detection, only of specificity.

        Never raises: `_ocr_band` swallows every failure into None by contract.
        """
        band = self.spec.paywall_headline_band
        if band is None:                       # this app declares no headline to refine with
            return None
        return self._ocr_band(frame, band, psm="6",
                              white_text_threshold=_PAYWALL_OCR_WHITE_MIN)

    def _deck_blocked_reason(self, frame: bytes) -> str | None:
        """`frame` -> an operator-facing sentence saying what is standing between us and the
        deck, or None if nothing recognisable is.

        Two outcomes only, and the difference between them is purely how much we can honestly
        claim: the OCR-refined string asserts WHICH paywall this is, the generic one asserts
        only that the upgrade screen is up. Nothing here ever dismisses the screen — it is a
        purchase screen, and the standing owner rule is that paid controls are manual, always.

        The "is this the out-of-likes one" test is deliberately loose: lowercase the read and
        require "out of" AND "likes" anywhere in it. Hard-matching the full sentence would throw
        away a perfectly good read the moment tesseract drops the apostrophe, splits a word, or
        picks up a stray glyph from the photo behind the text — and the penalty for being wrong
        in either direction is one adjective in a message, never a wrong action.

        That is not a hypothetical tolerance: MEASURED 2026-08-11 against the reference
        screenshot, tesseract reads the headline's two wrapped lines correctly. `_ocr_band`
        preserves that boundary now, but older OCR/cache behavior welded the words into
        "freelikes" and imperfect reads can still vary their spacing. Both substrings this test
        looks for survive either form, which is the point; a random hero-image OCR result that
        happens to contain just one generic word does not.

        Both strings name Hinge outright even though this method lives on the app-agnostic
        AndroidDriver, because both are reachable only through a spec that declares a "paywall"
        template and HINGE_SPEC is the only one that does (2026-08-11). The day a second app
        gets one, these need to become per-spec wording rather than being left to tell a Bumble
        operator about Hinge+.
        """
        if not self._paywall_visible(frame):
            return None
        headline = (self._paywall_headline(frame) or "").casefold()
        if "out of" in headline and "likes" in headline:
            return "Hinge is out of free likes for today — the Hinge+ upgrade screen is up"
        return "Hinge's Hinge+ upgrade screen is up — the deck is not available"

    def blocked_reason(self) -> str | None:
        """Worker-facing: is the deck unavailable for a reason the operator must be told about?

        Contract (DatingAppDriver.blocked_reason): never raises, never touches the screen, and
        is called on EVERY iteration of both worker loops — hence the memo. Once a paywall (or
        the stuck-screen watchdog) has established a reason, that reason stands until the
        operator deals with it, so re-screencapping and re-OCRing it on every loop would burn
        seconds per iteration to re-derive an answer we already have. `self._blocked_reason` is
        also where `_observe_stuck_bail` deposits its verdict, so a watchdog stop the driver
        already decided on is reported through this same channel.

        `on_blank="none"`: a blank/asleep screen is the owner having stepped away, not a blocked
        deck. Reporting one as blocked would stop a perfectly healthy run.
        """
        if self._blocked_reason is not None:
            return self._blocked_reason
        try:
            frame = self._screencap(on_blank="none")
            if frame is None:
                return None
            reason = self._deck_blocked_reason(frame)
        except Exception:  # noqa: BLE001 — a failed probe is not evidence of a blocked deck
            # Deliberately catches DriverClosed too, despite that being a real, meaningful stop:
            # this method's contract is that it never raises, and a dropped ADB link surfaces
            # from the very next capture the loop makes anyway (that is how it has always been
            # reported). Letting it out HERE would turn a diagnostic probe into a second,
            # competing place the run can die from, for no earlier warning.
            return None
        if reason is not None:
            self._blocked_reason = reason
        return reason

    def _locate_target_heart(self, item_index: int | None, *, should_stop=None) -> tuple[int, int]:
        """comment_sheet flow only. Locate the heart of the numbered photo the opener is about --
        or STOP. This method never returns a DIFFERENT item's heart than the one it was asked for.

        item_index is a 0-based index into CAPTURE ORDER (this driver's `_current_sigs`), and
        that is the ONLY space it is ever in -- it is not the model's item number, which counts
        a different list from a different base (ops/OPENER-REDESIGN.md 5.1/5.7). Callers cross
        the two spaces exactly once, through opener.service.OpenerPick.capture_order_index, and
        that method returns None rather than guessing when no sound conversion exists. We
        re-navigate to the named captured frame by matching its downsample signature, then take
        its heart.

        NO SUBSTITUTION, AND THAT IS WHAT CHANGED HERE (doc 5.6, standing owner rule). Until
        2026-08-12 every route that could not reach the named item -- an out-of-range index, an
        undecodable target signature, a search that never matched, a matched frame carrying no
        heart -- fell back to `hearts[0]`/`_await_button("like")`, the topmost heart on screen,
        and reported the miss as `on_target=False` for the caller to repair the TEXT against.
        Repairing the text does not undo attaching the like to an item the model never chose, so
        all four now raise `HingeTargetingError` and the run stops. There is no "and by the way
        this is the wrong item" flag in the return value any more, because there is no path that
        can produce one: this returns the heart of item `item_index`, or it raises.

        RETRYING THE SAME ITEM IS ALLOWED AND IS TRIED FIRST. `_TARGET_HEART_ATTEMPTS` whole
        searches, each starting from its own `_scroll_to_top()`, before the stop -- a shaky hand
        is not a wrong decision (that constant carries why the number is 2). Every gesture is the
        driver's own humanized `_scroll_to_top` / `_scroll_down_one`, so the ledger, the jitter
        and the forbidden-zone guard all still apply, and a retry's scrolls are tracked exactly
        like the first attempt's.

        `item_index is None` MEANS NOBODY SAID WHICH ITEM, and it is deliberately not the same
        input as 0. It is legal only where there is no opener to misplace -- Hinge with
        `opener.enabled: false`, where a like is a plain like and no item was ever chosen, so
        there is nothing for a substitution rule to protect. `_like_comment_sheet` REFUSES the
        combination of an opener and a None index before any of this runs, so an opener can never
        ride on this branch. Here it simply takes the topmost heart, because something has to
        open the sheet.

        `item_index == 0` needs no navigation at all: `_like_comment_sheet` scrolls to the top
        immediately before calling this, so the topmost heart on screen IS item 0's. That is a
        fast path, not a fallback -- the item asked for and the item found are the same item.

        RESIDUAL, UNCHANGED AND STILL ACCEPTED ON THIS PATH: the heart taken from a matched frame
        is the topmost one on it, so a scroll position showing two items at once (a photo AND a
        prompt) can still land the tap on the neighbour. That is a property of the capture-order
        space itself, which numbers FRAMES rather than items; doc 5.6 closes it with counting
        navigation plus the post-tap crop check, and `model_item_index` is the parameter that
        turns the latter on. Nothing here can detect it, and nothing here pretends to."""
        self._raise_if_action_cancelled(should_stop, boundary="legacy target lookup")
        sigs = getattr(self, "_current_sigs", None)
        if item_index is None:
            # Recorded whether or not we have sigs: unlike every other branch this one is not a
            # search that failed, it is about never having been given anything to search for, and
            # a bug report must be able to tell those apart (HINGE-05). Not an error HERE --
            # _like_comment_sheet already refused any call that carries an opener, so what is
            # left is a plain, itemless like with nothing to misplace.
            self._raise_if_action_cancelled(should_stop, boundary="legacy target debug capture")
            self._dbg_action("locate_target_heart", self._snap(), item_index=None,
                             outcome="no_item_named", reason="no_target_index")
            return self._await_button("like", should_stop=should_stop)
        if item_index == 0:
            # At the scroll top the topmost heart is item 0's.
            return self._await_button("like", should_stop=should_stop)
        # THE FOUR REFUSALS THAT USED TO BE FALLBACKS. Each names what specifically went wrong,
        # because the operator's next move differs: a negative or out-of-range index is a caller
        # or conversion bug, an empty sig list means the profile on screen was never captured by
        # this driver instance, and an undecodable target frame is a capture-path failure.
        # `< 0` is separate from the fast path above deliberately: a negative index is invalid
        # input, not "the first item", and collapsing the two is how an unusable index would once
        # again resolve to a confident tap on card 1.
        if item_index < 0:
            refusal = f"item {item_index} is not a valid capture-order index"
        elif not sigs:
            refusal = ("this driver holds no captured frames for the profile on screen, so there "
                       "is nothing to navigate back to")
        elif item_index >= len(sigs):
            refusal = (f"item {item_index} is outside the {len(sigs)} frame(s) captured for this "
                       f"profile")
        elif sigs[item_index] is None:
            refusal = (f"the captured frame for item {item_index} could not be decoded, so there "
                       f"is no signature to navigate back to")
        else:
            refusal = ""
        if refusal:
            self._raise_if_action_cancelled(should_stop, boundary="legacy target debug capture")
            self._dbg_action("locate_target_heart", self._snap(), item_index=item_index,
                             outcome="stop", reason="unresolvable_target_index")
            raise HingeTargetingError(
                f"{self.spec.app}: cannot target the item the opener was written about -- "
                f"{refusal}. Nothing was tapped and the like is NOT sent; liking a different "
                f"item instead is never an option (ops/OPENER-REDESIGN.md 5.6).",
                stage="navigate", intended=item_index, index_space="capture_order")
        import numpy as np
        target = sigs[item_index]
        self._raise_if_action_cancelled(should_stop, boundary="legacy target debug capture")
        before = self._snap()
        # like() always _scroll_to_top()s right before calling this, so item_index (a capture-
        # order index counted down from the top) should need about that many scroll_up()s to
        # reach. Cap the search there (+ slack) instead of sweeping the whole scroll_captures
        # depth: on a real device a scroll can over/undershoot the intended frame, and without a
        # cap a target we've scrolled past costs a full wasted sweep before we even find out
        # (HINGE-05). That same over/undershoot is what the outer retry below exists for.
        tries = min(self._profile_capture_limit + 1, item_index + 3)
        reason = ""
        for attempt in range(1, _TARGET_HEART_ATTEMPTS + 1):
            self._raise_if_action_cancelled(should_stop, boundary="legacy target retry")
            if attempt > 1:
                # Re-establish the zero point and search for THE SAME ITEM again. This is the
                # sanctioned retry: same target, same comparison, a fresh read. `_scroll_to_top`
                # undoes exactly the tracked scrolls the previous attempt made.
                self._scroll_to_top(should_stop)
                if not self._interruptible_sleep(human_delay(0.3), should_stop):
                    self._raise_if_action_cancelled(should_stop, boundary="legacy target retry")
            matched_frame_no_heart = False
            for _ in range(tries):
                self._raise_if_action_cancelled(should_stop, boundary="legacy target capture")
                frame = self._screencap()
                ds = _downsample(frame)
                if ds is not None and float(np.mean(np.abs(ds - target))) < self.change_threshold:
                    hearts = _match_glyph(frame, self._template("like"), side="right",
                                          threshold=_LIKE_MATCH_THRESHOLD, y_band=self.content_band)
                    if hearts:
                        return hearts[0]              # the referenced item's heart, now in view
                    matched_frame_no_heart = True
                    break
                self._raise_if_action_cancelled(should_stop, boundary="legacy target scroll")
                self._scroll_down_one()          # tracked, so the retry's _scroll_to_top() above
                if not self._interruptible_sleep(human_delay(self.dwell_s * 0.4), should_stop):
                    self._raise_if_action_cancelled(should_stop, boundary="legacy target scroll")
            reason = ("heart_not_visible_on_matched_frame" if matched_frame_no_heart
                      else "target_frame_not_found")
            self._dbg_action("locate_target_heart", before, item_index=item_index,
                             outcome=("retry" if attempt < _TARGET_HEART_ATTEMPTS else "stop"),
                             reason=reason, attempt=attempt,
                             attempts=_TARGET_HEART_ATTEMPTS)
        detail = ("its frame was found but no like heart was visible on it"
                  if reason == "heart_not_visible_on_matched_frame"
                  else "none of the frames read back matched the one it was captured on")
        raise HingeTargetingError(
            f"{self.spec.app}: could not reach item {item_index}, the item the opener was "
            f"written about, in {_TARGET_HEART_ATTEMPTS} attempts -- {detail}. Nothing was "
            f"tapped and the like is NOT sent; the profile is left scrolled where the search "
            f"ended, for debugging. Liking whichever item we could reach instead is never an "
            f"option (ops/OPENER-REDESIGN.md 5.6).",
            stage="navigate", intended=item_index, index_space="capture_order")

    def _navigate_to_model_item(self, model_item_index: int, *, should_stop=None) -> tuple[int, int]:
        """Doc 5.5's counting navigation, wired: put model item N's heart on screen and return
        the point to tap. Never taps, never substitutes, and never scrolls back to the top.

        THIS IS THE HANDOVER `_like_comment_sheet` USED TO REFUSE AT. Until now a
        `model_item_index` with no capture-order index beside it was a hard stop naming
        `item_nav.navigate_to_item` as the missing piece; this is that call.

        BOTTOM-UP (ops/OPENER-REDESIGN.md 5.5, owner-approved 2026-08-12). The enumeration read
        leaves the card at the BOTTOM of the profile, and navigation walks back UP from there
        rather than rewinding to the top and walking down again. Two consequences for this
        method's placement, and both are why it is called where it is:

          * IT MUST RUN BEFORE ANY `_scroll_to_top`, and there is no longer one on this path at
            all. The old flow's rewind is gone; what is left is the entry frame the read left us
            on, which is the only frame the entry anchor can be measured against and the only one
            the identity strip is readable on (`item_identity`: at a scroll top that strip is
            Hinge's own chrome, byte-identical across two different people).
          * IT SPENDS `_current_item_anchor`, which lives and dies with the index and the crops.
            A missing anchor beside a present index is not a state `_invalidate_item_index` can
            produce, and it is checked anyway: doc 5.3's rule is that a missing table is a hard
            stop, never a reason to fall back to a fixed coordinate, and half a table is a
            missing table.

        EVERY REFUSAL BECOMES `HingeTargetingError`, which is what closes doc 9's blocker 4.
        `item_nav` refuses with a coded `ItemNavigationError`, but three of its dependencies
        refuse UNCODED and by design -- `ScrollStepError` (a card spacing no permitted gesture
        can enumerate), `SegmentationError` and `ShiftEstimationError` (the vision layer could
        not look at all) -- and `IdentityError` joins them. All four are correct hard stops and
        all four mean the same thing to the operator: the like was not put on the item the opener
        was written about, so it was not put anywhere. Translating them here rather than at the
        worker keeps `worker.py` free of Hinge symbols, which is the reason `ItemTargetingError`
        exists in `base.py`.

        NO `**plan_kwargs` PASS-THROUGH, on `_plan_enumeration_step`'s own precedent and for its
        reason: `ratio_window` / `max_step_px` / `fallback_spacing_px` are the offline-validation
        door doc 5.6 flags, a production caller passes none of them, and the way to keep that
        true is to have nowhere to put them (doc 9, blocker 8).
        """
        index = self._current_item_index
        anchor = self._current_item_anchor
        calibration = self._require_targeting_calibration(model_item_index)
        payload = self._current_item_payload
        if index is None or anchor is None or payload is None:
            missing = ("index" if index is None else
                       "entry anchor frame" if anchor is None else "numbered item payload")
            raise HingeTargetingError(
                f"{self.spec.app}: the opener targets model item {model_item_index}, but this "
                f"driver holds no item {missing} for the profile on screen "
                f"({self._current_items_unavailable or 'it was cleared without a reason'}), so "
                f"there is no way to count to that item. Nothing was tapped and the like is NOT "
                f"sent (ops/OPENER-REDESIGN.md 5.3/5.5).",
                stage="navigate", intended=model_item_index, index_space="model_items")
        # `model_item_index` counts only the crops actually sent to the model.  The navigation
        # index counts selectable blocks in the full ItemIndex, including any excluded crop.
        # They coincide on the common all-photo profile, but treating that coincidence as an
        # invariant silently taps a later item as soon as one selectable block is withheld.
        # Cross through the payload's authoritative model-number -> heart-ordinal table, then
        # locate that same ordinal in the index's selectable-block table.
        try:
            heart_ordinal = payload.item(model_item_index).heart_ordinal
            if heart_ordinal is None:
                raise ItemCropError("the numbered crop has no heart ordinal")
            navigation_index = index.translation.index(heart_ordinal) + 1
        except (ItemCropError, ValueError, AttributeError) as exc:
            raise HingeTargetingError(
                f"{self.spec.app}: model item {model_item_index} cannot be translated into the "
                f"current item index ({exc}). Nothing was tapped and the like is NOT sent; "
                "there is no safe substitute for a missing model-item-to-heart mapping.",
                stage="navigate", intended=model_item_index, index_space="model_items") from exc
        # No pre-capture. `navigate_to_item`'s own first act is a screencap of the entry frame,
        # and it hands that frame back on every refusal it decides (`ItemNavigationError.frame`)
        # and the LANDING frame on success -- both of which are the frame a reader of the debug
        # log actually wants. A `_snap()` here would be a second ADB round-trip for a worse
        # picture, on every navigation, whether or not debug logging is on.
        try:
            target = navigate_to_item(self, index, navigation_index, entry_reference=anchor,
                                      identity_match_max_dist=calibration.identity_match_max_dist,
                                      should_stop=should_stop)
        except ItemNavigationError as exc:
            self._dbg_action("navigate_to_item", exc.frame, item=model_item_index,
                             outcome="stop", reason=exc.code)
            raise HingeTargetingError(
                f"{self.spec.app}: could not put the tap on model item {model_item_index}, the "
                f"item the opener was written about -- {exc} [{exc.code}]. Nothing was tapped "
                f"and the like is NOT sent; the profile is left where the walk stopped, for "
                f"debugging. Liking whichever item we could reach instead is never an option "
                f"(ops/OPENER-REDESIGN.md 5.6).",
                stage="navigate", intended=model_item_index,
                index_space="model_items") from exc
        except (ScrollStepError, SegmentationError, ShiftEstimationError, IdentityError) as exc:
            self._dbg_action("navigate_to_item", None, item=model_item_index,
                             outcome="stop", reason=type(exc).__name__)
            raise HingeTargetingError(
                f"{self.spec.app}: could not put the tap on model item {model_item_index} -- the "
                f"navigation pass could not look at the screen at all ({type(exc).__name__}: "
                f"{exc}). Nothing was tapped and the like is NOT sent "
                f"(ops/OPENER-REDESIGN.md 5.6).",
                stage="navigate", intended=model_item_index,
                index_space="model_items") from exc
        self._dbg_action("navigate_to_item", target.frame, item=model_item_index,
                         outcome="located", heart_ordinal=target.heart_ordinal,
                         point=list(target.point), scrolls=target.scrolls,
                         climbed_px=target.climbed_px, agreement_px=target.agreement_px,
                         hearts_counted=target.hearts_counted, reason=target.reason)
        return target.point

    def _verify_like_landed(self, before) -> None:
        """comment_sheet flow only. A like is COMPLETE only when the comment sheet AND any
        paid-upsell modal are gone AND the deck has moved off the pre-tap card. If the sheet is
        still up (missed Send Like tap) or the screen never changed (missed heart tap), raise so
        the worker HALTS instead of counting a like that never sent. A paid-upsell modal that
        animates in LATE — after _handle_rose_upsell's own poll window already gave up and
        returned False — is tolerated here rather than treated as a dead run: we dismiss it
        ourselves (same rule: never the paid option) and keep checking (HINGE-07). Only a
        modal/sheet that genuinely won't clear, or a deck that never advances, still raises — an
        unsent like must never be mislabelled as sent (it would corrupt the taste model).

        The *generic* sheet/advance verification remains opt-in for the human-supervised
        ``halt_on_error=False`` mode. Paywall detection does not: Hinge's known refusal screen
        is affirmative evidence that this particular like did NOT land, so returning normally
        there would let even a direct driver caller record a completed action that Hinge refused.
        Unlike a bare change-check, the scroll-to-top can't spoof this."""
        sheet_up = modal_up = landed_candidate = False
        for attempt in range(3):  # extra passes tolerate late-animating modals and paywalls
            frame = self._screencap()
            # A recognised paywall is a purchase screen, so it wins over every other post-send
            # interpretation of this frame.  In particular, do this BEFORE the late-upsell
            # branch below: a loose dismiss-glyph match on a paid screen must not turn a
            # read-only rejection detector into one more tap on that screen.
            blocked = self._deck_blocked_reason(frame)
            if blocked is not None:
                self._blocked_reason = blocked
                raise HingeDeckBlockedError(blocked)
            if not self.halt_on_error:
                # A screen change alone is NOT proof that Hinge accepted the like: the measured
                # out-of-free-likes path closes the sheet and replaces the card with its Hinge+
                # paywall.  This check is deliberately BEFORE the halt_on_error return: a
                # recognised rejection is not an optional generic verification failure.
                if attempt < 2:
                    time.sleep(human_delay(0.6))
                continue
            sheet_up = self._observe_like_sheet_visible(frame)
            modal_hits = _match_glyph(frame, self._template("upsell_dismiss"), side="any", threshold=0.6)
            modal_up = bool(modal_hits)
            if modal_up:
                self._tap(*modal_hits[0])             # late-animating upsell -> dismiss, never the paid option
            if sheet_up or modal_up:
                time.sleep(human_delay(0.6))
                continue
            if before is not None and not self._changed(before, frame):
                time.sleep(human_delay(0.6))
                continue
            # Require the apparently-advanced deck to survive one more poll. A transition frame
            # can be both changed and free of sheet glyphs just before a paywall animates in.
            if landed_candidate or attempt == 2:
                return                                # stable closed sheet + advanced deck -> sent
            landed_candidate = True
            time.sleep(human_delay(0.6))
        if not self.halt_on_error:
            return
        if sheet_up or modal_up:
            raise HingeActionError("like did not complete — the like sheet / upsell modal is still open")
        raise HingeActionError("like did not change the screen (missed tap or stuck)")

    # --- actions (NORMAL like only — never a paid upgrade) ----------------------
    def like(self, opener: str | None = None, item_index: int | None = None, *,
             model_item_index: int | None = None, should_stop=None) -> None:
        # item_index defaults to None ("nobody said which item"), NOT to 0 ("the first captured
        # frame"). See base.Driver.like and _locate_target_heart: the two are different inputs
        # and only one of them licenses attaching an opener to what gets tapped.
        #
        # model_item_index is the OTHER space and is doc 5.6's input: the 1-based number of the
        # item in the list the model was actually shown, which this driver holds the crops for in
        # `_current_item_payload`. When it is given, the sheet that opens is VERIFIED against
        # that item's stored crop before a character is typed -- see _like_comment_sheet.
        #
        # RAISES HingeTargetingError (a HingeActionError AND a base.ItemTargetingError) when the
        # chosen item cannot be reached or the opened sheet is not showing it. Never a different
        # item, never a rewritten opener, never a commentless like: doc 5.6's hard stop, which
        # worker.py catches by the base type and renders as a stop rather than a crash.
        if item_index is not None and model_item_index is not None:
            raise HingeTargetingError(
                f"{self.spec.app}: like() received both capture-order item_index "
                f"{item_index} and model_item_index {model_item_index}. They name different "
                "index spaces, so the target is ambiguous. Nothing was tapped and the like is "
                "NOT sent; supply exactly one target index.",
                stage="preflight", intended=model_item_index, index_space="model_items")
        try:
            if self.spec.like_flow == "comment_sheet":
                self._like_comment_sheet(opener, item_index,
                                         model_item_index=model_item_index,
                                         should_stop=should_stop)
            else:
                self._like_direct(opener, item_index, model_item_index=model_item_index)
        finally:
            # The deck has moved on (or an action failed part way through it, which is worse:
            # nobody knows where the deck is). Either way the item table describes a card that
            # is no longer the current one, so it is dropped here rather than left to be
            # overwritten by the next capture -- doc 5.3's invalidation rule, applied at the
            # other place the profile on screen changes. In `finally` deliberately: an exception
            # is exactly when a stale table would survive longest.
            self._invalidate_item_index(
                "the deck advanced after this like, so anything enumerated for the previous "
                "profile no longer describes what is on screen")

    def _require_targeting_calibration(self, model_item_index: int) -> TargetingCalibration:
        """Return the evidence-backed bounds or stop before a model-selected item is touched."""
        calibration = self.targeting_calibration
        if calibration is not None:
            if self._adb is not None:
                package_dump = self.adb.shell(f"dumpsys package {self.package}")
                match = re.search(r"(?m)^\s*versionName=(\S+)\s*$", package_dump)
                version = match.group(1) if match else None
            else:
                version = self._targeting_runtime_version_name
            size = (self.adb.screen_size() if self._adb is not None
                    else self._targeting_runtime_frame_size)
            if version != calibration.hinge_version_name or size != calibration.frame_size_px:
                raise HingeTargetingError(
                    f"{self.spec.app}: model item {model_item_index} cannot be targeted because "
                    "the live app build/frame geometry does not exactly match targeting "
                    f"calibration ({version!r}/{size!r} != "
                    f"{calibration.hinge_version_name!r}/{calibration.frame_size_px!r}). "
                    "Nothing was tapped and the like is NOT sent; recapture schema-v3 inline "
                    "calibration for this build.", stage="preflight", intended=model_item_index,
                    index_space="model_items")
            return calibration
        raise HingeTargetingError(
            f"{self.spec.app}: model item {model_item_index} cannot be targeted because "
            f"apps.{self.spec.app}.targeting_calibration is unavailable "
            f"({self._targeting_calibration_unavailable or 'no reason was recorded'}). Nothing "
            f"was tapped and the like is NOT sent; provide measured identity and sheet-item "
            f"bounds, exact effective crop geometry, device, and calibrated_at evidence before "
            f"enabling targeted likes.",
            stage="preflight", intended=model_item_index, index_space="model_items")

    def targeted_suggestion_blocker(self) -> str:
        """Why an Observe model-item suggestion is unavailable, or ``""`` when licensed.

        Observe generates before the human opens a sheet.  This pure capability lets the
        worker refuse the provider call itself when the calibrated identity and sheet bounds
        are absent; waiting until ``observe_item_mismatch`` would briefly expose unchecked text.
        """
        if self.targeting_calibration is not None:
            return ""
        return (f"targeted suggestion is unavailable because apps.{self.spec.app}."
                f"targeting_calibration is unavailable "
                f"({self._targeting_calibration_unavailable or 'no reason was recorded'}); "
                "no opener text is offered")

    def _verifiable_payload(self, model_item_index: int | None):
        """The crops doc 5.6 will verify the sheet against, or None when nobody named an item.

        Runs BEFORE the scroll, the heart search and the tap, so every refusal it makes leaves the
        screen exactly as it was: no gesture, no opened sheet, no like spent. That placement is
        the point -- an item that cannot be verified after the tap is an item that must not be
        tapped, and finding out afterwards costs a stop with a sheet open on somebody's card.

        Three refusals, all `HingeTargetingError`, which is the same halt every other
        "we could not put the like on the chosen item" outcome raises -- worker.py catches it by
        its `base.ItemTargetingError` half and turns it into a stop with the reason on the hub
        banner rather than a crash (see _auto_loop's targeting stop):

          * a model item number with no payload behind it. The table's lifetime is one profile and
            `_invalidate_item_index` records WHY it went, so the reason is quoted rather than
            replaced -- "the deck advanced" and "this capture could not be indexed" call for very
            different next moves by the operator;
          * a number outside 1..N, or a payload that is unusable. `verification_blocker` raises
            `SheetVerificationError` for both rather than answering, on `ItemPayload.item`'s
            reasoning (doc 5.6: never substitute a different item);
          * an item whose stored crop cannot serve as a verification reference at all -- doc 5.4's
            animated-card class, detected explicitly from the numbers `item_crops` measured for
            this very profile rather than assumed away by a wider tolerance.
        """
        if model_item_index is None:
            return None
        self._require_targeting_calibration(model_item_index)
        payload = self._current_item_payload
        if payload is None:
            raise HingeTargetingError(
                f"{self.spec.app}: the opener targets model item {model_item_index}, but this "
                f"driver holds no item crops for the profile on screen, so the sheet that opens "
                f"cannot be checked against the item the opener was written about — "
                f"{self._current_items_unavailable or 'no reason was recorded'}. Doc 5.3 treats a "
                f"missing table as a hard stop, never as a reason to fall back to a fixed "
                f"coordinate, so nothing is tapped and the like is NOT sent.",
                stage="verify", intended=model_item_index, index_space="model_items")
        try:
            blocker = verification_blocker(payload, model_item_index)
        except SheetVerificationError as exc:
            raise HingeTargetingError(
                f"{self.spec.app}: refusing to like model item {model_item_index} — {exc}. "
                f"Nothing was tapped and the like is NOT sent.",
                stage="verify", intended=model_item_index, index_space="model_items") from exc
        if blocker:
            raise HingeTargetingError(
                f"{self.spec.app}: refusing to like model item {model_item_index} — {blocker}. "
                f"Nothing was tapped and the like is NOT sent.",
                stage="verify", intended=model_item_index, index_space="model_items")
        return payload

    def item_type_preflight(self, item_description: str, model_item_index: int) -> ItemTypePreflight:
        """Pure doc 5.8 check against the numbered crop the model was shown.

        Method presence is the optional driver capability: generic drivers deliberately do not
        inherit a stub, so existing drivers and test doubles remain compatible.  This method
        neither screencaps nor logs nor touches ADB; it only reads the current profile's stored
        crop and returns INCONCLUSIVE when that evidence is unavailable.
        """
        payload = self._current_item_payload
        if payload is None:
            return ItemTypePreflight(
                INCONCLUSIVE, "unknown", "unknown",
                "this driver has no numbered crop for the current profile")
        try:
            crop = payload.item(model_item_index)
        except ItemCropError as exc:
            return ItemTypePreflight(
                INCONCLUSIVE, "unknown", "unknown",
                f"model item {model_item_index} has no readable numbered crop ({exc})")
        return preflight_item_type(item_description, crop.image)

    def _confirm_payload_profile(self, model_item_index: int) -> None:
        """Is the person on screen still the person the stored crops describe? Or STOP.

        Runs BEFORE the scroll and the tap, on the same terms as `_verifiable_payload`: every
        refusal leaves the screen exactly as it was. It is the same check, at the same point in
        the sequence, that `item_nav.navigate_to_item` makes as the first thing it does -- so
        whoever wires counting navigation inherits this rather than replacing it.

        WHY THE POST-TAP CHECK IS NOT ENOUGH ON ITS OWN, measured rather than reasoned. Doc 5.3
        argued a stale table was "a reliability bug rather than a safety one" because the stored
        crops would be stale too and the sheet comparison would fail. It does not always fail: a
        validation pass drove a stale payload for one profile against a sheet rendering another
        profile's card through this very method and got a MATCH, a typed opener and a SENT like,
        10 times in 540 comparisons when only its relative, closed-set rule was used. Production
        now also requires the per-device absolute ceiling, but the profile check remains an
        independent requirement rather than assuming one calibrated metric replaces the other.

        AND THIS CHECK IS NOT ENOUGH ON ITS OWN EITHER, which is why both run. Over six real
        profiles an old 3.0 identity bound admitted a different-person pair at 2.565. Targeted
        work is now licensed only by a measured bound strictly below that collision. Even then,
        the independent sheet check and correct `_invalidate_item_index` lifecycle remain
        mandatory.

        IDENTITY IS NOT READABLE AT A SCROLL TOP, and that is why this is here rather than after
        the `_scroll_to_top` below: there the strip is Hinge's own filter-chips row, identical for
        everybody, so there is nothing to tell two people apart. An enumeration read leaves the
        card scrolled with the sticky header showing, which is exactly where a like starts. If the
        screen is somewhere else, the answer is "cannot tell" and this stops -- never a pass,
        because a check that cannot be made must not read as a check that succeeded.
        """
        index = self._current_item_index
        if index is None:
            # Cannot happen through `_index_captured_items`, which sets the index and the payload
            # together and clears them together, but it is stated rather than assumed: a payload
            # whose index went missing is a payload nobody can attribute to a profile.
            raise HingeTargetingError(
                f"{self.spec.app}: the opener targets model item {model_item_index} and this "
                f"driver holds crops for it but no index behind them, so there is no fingerprint "
                f"to check the profile on screen against. Nothing was tapped and the like is NOT "
                f"sent.",
                stage="navigate", intended=model_item_index, index_space="model_items")
        frame = self._screencap()
        try:
            verdict = compare_profile_identity(
                frame, index.identity, identity_band=self.identity_band,
                match_max_dist=self._require_targeting_calibration(
                    model_item_index).identity_match_max_dist)
        except IdentityError as exc:
            raise HingeTargetingError(
                f"{self.spec.app}: the profile on screen could not be checked against the one "
                f"model item {model_item_index}'s crops were cut from ({exc}). Nothing was tapped "
                f"and the like is NOT sent.",
                stage="navigate", intended=model_item_index, index_space="model_items") from exc
        self._dbg_action("confirm_payload_profile", frame, item=model_item_index,
                         outcome=verdict.state, distance=verdict.distance)
        if verdict.matched:
            return
        raise HingeTargetingError(
            f"{self.spec.app}: the profile on screen is not the one model item "
            f"{model_item_index}'s crops were cut from, so the opener would be attached to a "
            f"card this driver has never seen. Nothing was tapped and the like is NOT sent. "
            f"{verdict.reason}",
            stage="navigate", intended=model_item_index, index_space="model_items")

    def _verify_sheet_shows(self, sheet: bytes, payload, model_item_index: int, before, *,
                            composer_surface: ComposerSurface | None = None,
                            allow_top_identity: bool = False) -> None:
        """Doc 5.6's post-tap check. Returns only when the sheet IS showing item `model_item_index`.

        `sheet` is the screencap taken once `_await_sheet_open` confirmed the comment sheet is up
        — the picture of the item this comment is about to attach to. The comparison is a
        deterministic signature match against the crop this driver stored for that item while it
        enumerated the profile; there is no model call, no judgement and no repair.

        On anything else this raises, which is what makes "verify, then type" a property of the
        code rather than a convention: the caller's typing lives below this line, so a miss cannot
        reach it. The screen is deliberately left exactly as it is — sheet open, nothing typed,
        Send never tapped — matching the rest of this driver's halt behaviour, and the message
        carries INTENDED and ACTUAL item numbers because those are the two things doc 5.6 asks a
        stop record to hold.

        `SheetVerificationError` ("could not look": no preview on the frame, undecodable bytes,
        missing vision extras) becomes the same stop as a mismatch. To a run that is one tap away
        from typing an opener under an unverified card the two call for the same action, and only
        the diagnosis differs.
        """
        # `before` is the PRE-TAP card and `_dbg_action` captures the screen as it is now, so one
        # debug entry holds both sides of the tap: what was under the heart and what the sheet
        # opened on. That pair is the whole diagnosis when a verification stop has to be read back
        # a day later.
        calibration = self._require_targeting_calibration(model_item_index)
        index = self._current_item_index
        if index is None:
            raise HingeTargetingError(
                f"{self.spec.app}: the like sheet cannot be attributed to model item "
                f"{model_item_index}'s profile because its identity index is missing. The opener "
                f"is NOT typed and the like is NOT sent.",
                stage="verify", intended=model_item_index, index_space="model_items")
        # The deck can advance between the pre-tap identity check and the sheet arriving. Check
        # the sticky header again before the crop verifier: a closed-set item comparison may pass
        # for a foreign card, but it cannot license typing on another person's profile.
        try:
            identity = compare_profile_identity(
                sheet, index.identity, identity_band=self.identity_band,
                match_max_dist=calibration.identity_match_max_dist)
        except IdentityError as exc:
            self._dbg_action("verify_sheet_identity", before, item=model_item_index,
                             outcome="unreadable", reason=str(exc))
            raise HingeTargetingError(
                f"{self.spec.app}: the profile on the like sheet could not be checked against "
                f"model item {model_item_index}'s profile ({exc}). The opener is NOT typed and "
                f"the like is NOT sent; the sheet is left open for debugging.",
                stage="verify", intended=model_item_index, index_space="model_items") from exc
        self._dbg_action("verify_sheet_identity", before, item=model_item_index,
                         outcome=identity.state, distance=identity.distance,
                         bound=identity.match_max)
        if identity.mismatched or (identity.unknown and not allow_top_identity):
            raise HingeTargetingError(
                f"{self.spec.app}: the like sheet is not on the profile model item "
                f"{model_item_index} was cropped from, so the opener is NOT typed and the like "
                f"is NOT sent. {identity.reason}",
                stage="verify", intended=model_item_index, index_space="model_items")
        try:
            verdict = verify_sheet_item(
                sheet, payload, model_item_index,
                absolute_max_dist=calibration.inline_item_max_dist,
                composer_surface=composer_surface)
        except SheetVerificationError as exc:
            self._dbg_action("verify_sheet_item", before, item=model_item_index,
                             outcome="unreadable", reason=str(exc))
            raise HingeTargetingError(
                f"{self.spec.app}: the like sheet could not be checked against model item "
                f"{model_item_index} ({exc}). The opener is NOT typed and the like is NOT sent; "
                f"the sheet is left open on screen for debugging.",
                stage="verify", intended=model_item_index, index_space="model_items") from exc
        self._dbg_action("verify_sheet_item", before, item=model_item_index,
                         outcome=verdict.state, nearest=verdict.nearest_index,
                         distance=verdict.distance, bound=verdict.bound,
                         preview=[verdict.preview.y0, verdict.preview.y1,
                                  verdict.preview.x0, verdict.preview.x1])
        if verdict.matched:
            return
        if verdict.distance is None:
            intended = next(c for c in verdict.comparisons if c.number == model_item_index)
            raise HingeTargetingError(
                f"{self.spec.app}: the like sheet could not confirm model item "
                f"{model_item_index}: {intended.reason}. The opener is NOT typed and the like "
                f"is NOT sent; the sheet is left open for debugging.",
                stage="verify", intended=model_item_index, index_space="model_items")
        # Never send a commentless like, never ship a comment attached to the wrong item, never
        # rewrite the opener to match whatever we hit (all three are owner rules, and the third is
        # why there is no anchored re-ask here -- that repair callback was removed outright on
        # 2026-08-12, see base.Driver.like). Every remaining option violates one of them, so the
        # run stops with the screen exactly as it is. `intended` and `actual` ride on the
        # exception as well as in the sentence, because worker.py records them in the stop.
        raise HingeTargetingError(
            f"{self.spec.app}: the like sheet is NOT showing the item the opener was written "
            f"about — intended model item {model_item_index}, actual {verdict.nearest_index} — so "
            f"the opener is NOT typed and the like is NOT sent. {verdict.reason}",
            stage="verify", intended=model_item_index, actual=verdict.nearest_index,
            index_space="model_items")

    # --- doc 5.9's observe-side mismatch guard ---------------------------------------
    # Same two comparisons `like()` makes, in the same order, on a sheet a HUMAN opened. The
    # difference is only what a refusal costs: auto stops the run, observe refuses to show text.
    # There is deliberately no `supports_*` capability flag beside it: the method's PRESENCE is
    # the capability, worker.py tests exactly that, and a second declaration of the same fact is
    # one more thing that can drift out of agreement with the first.
    def observe_item_check(self, sheet: bytes, model_item_index: int) -> ObserveItemCheck:
        """Classify the open composer's selected item as match, mismatch, or inconclusive.

        DOC 5.9's MISMATCH GUARD, and the reason it exists is that the inversion removed a
        guarantee. While observe generated AFTER the tap, the suggestion was right by
        construction ("exact by construction (a human tapped it)"). Generating BEFORE the tap
        means the human may heart a different item than the one the hub named -- and the owner
        explicitly declined to make that a prompt, a question, or training data. So it is
        DETECTED and SURFACED and nothing else: the hub replaces the opener with this sentence
        and offers no text to type. Silence is the one outcome that is not allowed.

        TWO CHECKS, AND EITHER REFUSING MEANS REFUSE. This is not belt-and-braces, it is the
        only honest reading of the measurements this project has:

          * `compare_profile_identity` answers WHOSE profile the sheet belongs to. Doc 5.9 asks
            for the deck-advance race to be answered on identity rather than on card pixels, and
            it can be, on the frame already in hand: the comment sheet does NOT occlude the
            sticky header (`identity_band` cuts rows 115..226; the sheet's preview starts at row
            236), measured on six real sheets, every one of which read `confirmed_not_top` and
            sat 0.000 from its own profile's neighbouring scrolled frames. No extra screencap.
          * `verify_sheet_item` answers WHICH item of that profile is on the sheet.

          NEITHER IS SUFFICIENT ALONE. An old 3.0 identity bound admitted a measured different
          person at 2.565, while the relative-only sheet test accepted a foreign card 10 times in
          540. This method therefore uses the configured identity bound and absolute sheet ceiling
          together; either refusal withholds the text, and missing calibration blocks suggestion
          generation before a provider call.

        IT NEVER RAISES AND IT NEVER STOPS THE RUN. Observe is a labelling session; ending it
        over a cosmetic display failure is the mistake the advisory plumbing already exists to
        prevent. Every "could not look" -- a missing payload, an unreadable band, no preview on
        the frame, missing vision extras -- comes back as a REASON, which the hub renders exactly
        like a mismatch, because to an operator about to type they call for the same action.

        IT NEVER TOUCHES THE TRANSPORT AND IT NEVER LOGS. Both are deliberate: this can be called
        from the worker's suggestion thread (when the model's answer lands while the sheet is
        already open) while the worker thread is inside `wait_for_decision` screencapping and
        writing its own debug records. Pure vision over a frame the caller already holds is the
        only shape that is safe there -- `DebugLog` appends to one file from one thread by
        design, and a second writer would interleave records in the artefact a bug report is
        reconstructed from. The console line the caller prints is this check's record.
        """
        def result(state: str, reason: str = "") -> ObserveItemCheck:
            return ObserveItemCheck(state, reason)

        calibration = self.targeting_calibration
        if calibration is None:
            return result(OBSERVE_ITEM_INCONCLUSIVE, self.targeted_suggestion_blocker())
        payload = self._current_item_payload
        index = self._current_item_index
        if payload is None or index is None:
            return result(
                OBSERVE_ITEM_INCONCLUSIVE,
                f"this driver holds no numbered items for the profile on screen, so there is "
                f"no way to tell whether the sheet is showing item {model_item_index}: "
                f"{self._current_items_unavailable or 'no reason was recorded'}")
        try:
            composer = locate_inline_composer(
                sheet, self._template("confirm"), threshold=0.8)
        except ComposerDetectionError as exc:
            return result(
                OBSERVE_ITEM_INCONCLUSIVE,
                f"the inline Send Like composer could not be structurally confirmed "
                f"({exc}), so the suggestion is not offered")
        try:
            verdict = compare_profile_identity(
                sheet, index.identity, identity_band=self.identity_band,
                match_max_dist=calibration.identity_match_max_dist)
        except IdentityError as exc:
            return result(
                OBSERVE_ITEM_INCONCLUSIVE,
                f"the profile on the like sheet could not be checked against the one item "
                f"{model_item_index} was cropped from ({exc}), so the suggestion is not "
                f"offered")
        if not verdict.matched:
            reason = (f"the like sheet is not on the profile item {model_item_index} was cropped "
                      f"from, so this suggestion is about a card you are no longer looking at. "
                      f"{verdict.reason}")
            return result(
                OBSERVE_ITEM_MISMATCH if verdict.mismatched else OBSERVE_ITEM_INCONCLUSIVE,
                reason)
        try:
            blocker = verification_blocker(payload, model_item_index)
            if blocker:
                return result(OBSERVE_ITEM_INCONCLUSIVE, blocker)
            sheet_verdict = verify_sheet_item(
                sheet, payload, model_item_index,
                absolute_max_dist=calibration.inline_item_max_dist,
                composer_surface=composer)
        except (SheetVerificationError, ItemCropError) as exc:
            return result(
                OBSERVE_ITEM_INCONCLUSIVE,
                f"the like sheet could not be checked against model item "
                f"{model_item_index} ({exc}), so the suggestion is not offered")
        if sheet_verdict.matched:
            return result(OBSERVE_ITEM_MATCH)
        # A nearest *reachable* crop is not the item the human opened when the intended crop
        # could not be measured at all (for example a still-settling inline reframe).  It is
        # merely the closest remaining candidate.  Calling that a positive identification was
        # the Alex report's false "you opened item 6" while the sheet was actually item 3.
        if sheet_verdict.distance is None:
            intended = next(c for c in sheet_verdict.comparisons
                            if c.number == model_item_index)
            return result(
                OBSERVE_ITEM_INCONCLUSIVE,
                f"the selected image could not yet be confirmed as model item "
                f"{model_item_index}: {intended.reason}. The suggestion is withheld until "
                f"the sheet can be checked again")
        if sheet_verdict.state != VERIFY_MISMATCH:
            return result(OBSERVE_ITEM_INCONCLUSIVE, sheet_verdict.reason)
        actual = ("nothing this profile was indexed with" if sheet_verdict.nearest_index is None
                  else f"item {sheet_verdict.nearest_index}")
        return result(
            OBSERVE_ITEM_MISMATCH,
            f"you opened {actual}, but this suggestion was written about item "
            f"{model_item_index}. {sheet_verdict.reason}")

    def observe_item_mismatch(self, sheet: bytes, model_item_index: int) -> str:
        """Compatibility wrapper: empty on match, otherwise the operator-facing refusal.

        New Observe state machines consume :meth:`observe_item_check` so a transient unreadable
        refresh is not confused with positive evidence of the wrong item. Auto and existing
        external/test callers retain the original string contract.
        """
        check = self.observe_item_check(sheet, model_item_index)
        return "" if check.state == OBSERVE_ITEM_MATCH else check.reason

    # --- reviewed Observe action bridge ---------------------------------
    def observe_open_targeted_like(self, model_item_index: int, *, should_stop=None) -> None:
        """Worker-owned wrapper that excludes a second OBSERVE controller mid-action."""
        with self._observe_input_lease("observe_open_targeted_like"):
            return self._observe_open_targeted_like_unlocked(
                model_item_index, should_stop=should_stop)

    def _observe_open_targeted_like_unlocked(self, model_item_index: int, *, should_stop=None) -> None:
        """Open and verify Hinge's comment sheet for ONE indexed model item.

        This is intentionally not a generic tap API.  It spends the current driver's own item
        index and entry anchor, then leaves the verified sheet open.  No character is typed and
        no like is sent here; ``observe_send_targeted_like`` is the separate irreversible step.
        """
        self._raise_if_action_cancelled(should_stop, boundary="reviewed targeting preflight")
        payload = self._verifiable_payload(model_item_index)
        if payload is None:
            raise HingeTargetingError("reviewed targeted like requires a verifiable indexed item",
                                      stage="preflight", intended=model_item_index,
                                      index_space="model_items")
        self._confirm_payload_profile(model_item_index)
        heart = self._navigate_to_model_item(model_item_index, should_stop=should_stop)
        self._raise_if_action_cancelled(should_stop, boundary="reviewed target heart tap")
        before = self._snap()
        self._tap(*heart)
        if not self._interruptible_sleep(human_cooldown(0.8), should_stop):
            self._raise_if_action_cancelled(should_stop, boundary="reviewed sheet confirmation")
        composer = self._await_sheet_open()
        sheet = self._screencap()
        if composer is not None:
            try:
                composer = locate_inline_composer(sheet, self._template("confirm"), threshold=0.8)
            except ComposerDetectionError as exc:
                raise HingeTargetingError("the reviewed like composer could not be verified; "
                                          "nothing was typed or sent", stage="verify",
                                          intended=model_item_index, index_space="model_items") from exc
        self._verify_sheet_shows(sheet, payload, model_item_index, before,
                                 composer_surface=composer, allow_top_identity=True)
        # The action bridge has no passive wait in front of it to emit the usual
        # like-intent anchor.  Write the verified sheet before declaring the
        # post-tap fact, so release evidence has the same causal order as manual
        # Observe: published suggestion -> sheet anchor -> item verification.
        if self._dbg is not None:
            self._dbg.action("observe_like_anchor", before=sheet)
        # Commit the fact while this method's OBSERVE lease is still held. Returning
        # first would leave a gap in which another controller could move the card
        # before the Worker advertised a verified, sendable sheet.
        self.observe_release_fact("post_tap_item_verified")
        self._observe_reviewed_sheet = (model_item_index, payload, before)
        self._dbg_action("observe_reviewed_open", before, model_item_index=model_item_index,
                         verified=True)

    def observe_send_targeted_like(self, opener: str, model_item_index: int, *, should_stop=None) -> None:
        """Worker-owned wrapper that excludes a second OBSERVE controller mid-send."""
        with self._observe_input_lease("observe_send_targeted_like"):
            return self._observe_send_targeted_like_unlocked(
                opener, model_item_index, should_stop=should_stop)

    def _observe_send_targeted_like_unlocked(self, opener: str, model_item_index: int, *, should_stop=None) -> None:
        """Re-verify the already-open reviewed sheet, then type and send its current opener."""
        pending = getattr(self, "_observe_reviewed_sheet", None)
        if not opener or pending is None or pending[0] != model_item_index:
            raise HingeTargetingError("there is no matching reviewed, verified comment sheet open; "
                                      "nothing was typed or sent", stage="verify",
                                      intended=model_item_index, index_space="model_items")
        _item, payload, before = pending
        try:
            self._raise_if_action_cancelled(should_stop, boundary="reviewed comment entry")
            composer = self._await_sheet_open(tries=3)
            if composer is None:
                raise HingeTargetingError("the reviewed comment composer disappeared; nothing was typed or sent",
                                          stage="verify", intended=model_item_index,
                                          index_space="model_items")
            sheet = self._screencap()
            composer = locate_inline_composer(sheet, self._template("confirm"), threshold=0.8)
            self._verify_sheet_shows(sheet, payload, model_item_index, before,
                                     composer_surface=composer, allow_top_identity=False)
            self._tap(*composer.comment_rect.center)
            if not self._interruptible_sleep(human_delay(0.5), should_stop):
                self._raise_if_action_cancelled(should_stop, boundary="reviewed comment entry")
            focused = self._screencap()
            focused_composer = locate_inline_composer(focused, self._template("confirm"), threshold=0.8)
            self._verify_sheet_shows(focused, payload, model_item_index, before,
                                     composer_surface=focused_composer, allow_top_identity=False)
            self._raise_if_action_cancelled(should_stop, boundary="reviewed text entry")
            self.adb.text(opener)
            if not self._interruptible_sleep(human_delay(0.6), should_stop):
                self._raise_if_action_cancelled(should_stop, boundary="reviewed send")
            send_composer = self._await_sheet_open(tries=3)
            self._raise_if_action_cancelled(should_stop, boundary="reviewed send")
            if send_composer is None:
                self._tap_frac(self.coords["send_like"])
            else:
                self._tap(*send_composer.confirm_point)
            time.sleep(human_cooldown(0.6))
            self._record_reviewed_like_sending_if_observed(before)
            self._dbg_action("observe_reviewed_like_attempt", before,
                             model_item_index=model_item_index, opener_chars=len(opener), verified=True)
            self._handle_rose_upsell()
            self._verify_like_landed(before)
            self._dbg_action("observe_decision", before, decision="like", reviewed=True,
                             model_item_index=model_item_index, opener_chars=len(opener))
            self._dbg_action("observe_reviewed_like", before,
                             model_item_index=model_item_index, opener_chars=len(opener), verified=True)
        finally:
            self._observe_reviewed_sheet = None
            self._invalidate_item_index("reviewed Observe like completed or stopped; indexed card is no longer current")

    def _record_reviewed_like_sending_if_observed(self, before: bytes) -> bool:
        """Log a bridge send transition only when its captured frame actually proves one."""
        # The active bridge does not re-enter passive observation after Send. A
        # straight-to-next-card success is not evidence that Hinge was observed
        # resolving, so release validation must honestly refuse that abbreviated
        # trace instead of fabricating it.
        post_send = self._screencap()
        sheet_still_open = self._observe_like_sheet_visible(post_send)
        current_card = (post_send == before
                        or self._is_current_profile_frame(post_send, require_content=True))
        next_card_ready = not current_card and self._observe_deck_ready(post_send)
        if self._dbg is None or sheet_still_open or current_card or next_card_ready:
            return False
        self._dbg.action("observe_waiting", before=post_send, reason="like_sending")
        return True

    def _like_comment_sheet(self, opener: str | None, item_index: int | None, *,
                            model_item_index: int | None = None, should_stop=None) -> None:
        """Hinge's flow: heart -> inline "Send Like" composer appears -> optionally type the
        opener into its comment box (Signals #2: the opener is sent WITH the like) -> tap
        Send -> handle a paid-upsell interstitial (never tap the paid option) -> verify.

        NOTHING HERE EVER SUBSTITUTES AN ITEM, AND THERE IS NO LONGER A REPAIR PATH THAT COULD.
        Doc 5.6, standing owner rule: "no falling back to `hearts[0]`, no 'closest reachable
        item', no rewriting the opener to match whatever we hit". Until 2026-08-12 an
        `anchored_opener` callback sat below the tap and re-asked the model for text about
        whichever item the sheet had actually opened on; it is removed, because repairing the TEXT
        does not undo spending the LIKE on an item the model never chose. What is left is: reach
        the chosen item or stop (`_locate_target_heart`), and -- when we hold crops for it --
        prove the sheet is showing it or stop (`_verify_sheet_shows`).

        TWO TARGETING CONTRACTS LIVE HERE, AND `model_item_index` PICKS BETWEEN THEM.

        Given a `model_item_index`, doc 5.6's POST-TAP CHECK is in force and is the only thing
        that licenses typing: the sheet that opens is matched, deterministically and without a
        model call, against this profile's stored crop of that item (`item_verify`), and a miss
        is a hard stop with nothing typed and the sheet left open. The check that used to stand
        here was a PRE-tap screen comparison which `_locate_target_heart` still admits "can land
        the tap on the neighbour" when one frame shows two items; a post-tap CONTENT check is the
        thing that was actually missing.

        Whether item `model_item_index` can be verified at all is settled BEFORE anything is
        touched, by `verification_blocker` -- an animated card whose own frame-to-frame drift
        exceeds half its distance to its neighbours has a stored crop that cannot serve as a
        reference, and doc 5.4 asks for that to be detected explicitly rather than papered over
        with a tolerance band wide enough to accept a different item.

        AND SO IS WHOSE PROFILE THIS IS, by `_confirm_payload_profile`, which runs before any
        gesture because the identity strip is only readable while the card is scrolled -- which,
        with bottom-up navigation, is exactly where the profile read leaves it. The order is
        deliberate and is the order `item_nav.navigate_to_item` uses too: whose profile, then can
        this item be checked at all, then -- after the tap -- which item. The post-tap check does
        NOT subsume the first of those; it was measured accepting another profile's card outright
        (see that method's docstring for the 10-of-540).

        HOW THE TAP POINT IS REACHED DEPENDS ON WHETHER a model item needs bottom-up counting.
        A `model_item_index` with no capture-order index beside it uses doc 5.5's counting
        navigation: `_navigate_to_model_item` walks UP from where the read left the card,
        counting hearts in reverse against the driver-owned index.  The older capture-order
        helper remains only for plain likes (and a defensive explicit dual-index call whose
        post-tap model-item verifier still runs); it can never by itself license opener text.

        EVERY OPENER MUST NAME A MODEL ITEM.  Capture-order frames are not a safe targeting space:
        one frame can contain two selectable cards, and that old branch had neither the calibrated
        profile identity check nor a numbered-crop sheet verifier.  A caller may still send a
        plain like with no item at all, but any text requires ``model_item_index`` and therefore
        takes the calibrated counting-navigation/verify path below."""
        self._raise_if_action_cancelled(should_stop, boundary="targeting preflight")
        if opener and model_item_index is None:
            raise HingeTargetingError(
                f"{self.spec.app}: an opener was supplied without a model item index. "
                "Capture-order targeting is retired because it cannot prove which selectable "
                "item the sheet represents. Nothing was tapped and the like is NOT sent; a "
                "model_item_index with calibrated identity and sheet verification is required.",
                stage="preflight", intended=item_index, index_space="capture_order")
        payload = self._verifiable_payload(model_item_index)   # raises before anything is touched
        if payload is not None:
            # Whose profile is this, before a finger moves and while the sticky header is still
            # readable. See _confirm_payload_profile for why the post-tap crop check does not
            # cover this and why this does not cover the post-tap crop check.
            #
            # KEPT even though `item_nav.navigate_to_item` opens with the very same comparison.
            # The duplication is one band decode and it is the only identity gate on this path if
            # `like()` is ever called with a model item number by something that is not the
            # counting-navigation branch below -- `like()` is a public driver method, and the two
            # guards historically failed in the same direction on a known collision, so neither
            # may be dropped merely because both now use calibrated bounds.
            self._confirm_payload_profile(model_item_index)
        if model_item_index is not None and item_index is None:
            # DOC 5.5'S COUNTING NAVIGATION, AND THIS IS WHERE THE HANDOVER LANDED. Until
            # 2026-08-12 this branch was a hard stop naming `item_nav.navigate_to_item` as the
            # missing piece; it is now that call, and the shape it refused is the shape that
            # runs.
            #
            # NO `_scroll_to_top` ON THIS PATH, deliberately, and that is the change rather than
            # an omission (ops/OPENER-REDESIGN.md 5.5, bottom-up navigation, owner-approved).
            # The enumeration read leaves the card at the bottom of the profile; navigation walks
            # back UP from there under continuous shift tracking, cross-checked by counting
            # hearts in reverse against the index. Rewinding first would throw away the one
            # anchor that is MEASURED rather than replayed, would put the identity strip on a
            # screen where it carries no identity at all, and would cost ~51 gestures to arrive
            # somewhere the read had already been.
            heart = self._navigate_to_model_item(model_item_index, should_stop=should_stop)
        else:
            # Plain likes may still use the old generic heart lookup because no text is being
            # attached to a model-selected item.  An unusual dual-index caller keeps the model
            # crop/identity/sheet gates above; an opener with only a capture-order index was
            # refused before any navigation.
            self._scroll_to_top(should_stop)
            self._raise_if_action_cancelled(should_stop, boundary="legacy navigation")
            if not self._interruptible_sleep(human_delay(0.4), should_stop):
                self._raise_if_action_cancelled(should_stop, boundary="legacy navigation")
            # Raises HingeTargetingError rather than returning a different item's heart. Anything
            # below this line is therefore working with the heart of the item the opener is about
            # -- as far as the capture-order space can tell, see that method's stated residual.
            heart = self._locate_target_heart(item_index, should_stop=should_stop)
        self._raise_if_action_cancelled(should_stop, boundary="heart tap")
        before = self._snap()                         # baseline AFTER navigation: the pre-tap card
        self._raise_if_action_cancelled(should_stop, boundary="heart tap")
        self._tap(*heart)                             # opens the comment / "Send Like" sheet
        if not self._interruptible_sleep(human_cooldown(0.8), should_stop):
            self._raise_if_action_cancelled(should_stop, boundary="like-sheet confirmation")
        composer = self._await_sheet_open()
        self._raise_if_action_cancelled(should_stop, boundary="comment entry")
        sheet = self._screencap()           # the like screen: shows the item this comment attaches to
        if composer is not None:
            try:
                composer = locate_inline_composer(
                    sheet, self._template("confirm"), threshold=0.8)
            except ComposerDetectionError as exc:
                raise UnlocatedControlError(
                    "the Hinge inline composer changed between detection and verification; "
                    "nothing was typed or sent") from exc
        # DOC 5.6'S POST-TAP CHECK, AND IT IS THE FIRST THING THAT LOOKS AT THE OPEN SHEET.
        # Placed above every branch that can type: the ordering IS the guarantee ("verify, then
        # type"), and there is no path from here to `self.adb.text(...)` that does not pass
        # through `verdict.matched`. `_verify_sheet_shows` raises on anything but a match.
        if payload is not None:
            # Item 1 opens with Hinge's profile-independent filter chips in identity_band.  That
            # is UNKNOWN, never a match; the selected-card verifier may establish the item now,
            # but no text is licensed until focusing the field shifts the sticky name into the
            # calibrated band and the full identity+item pair is repeated below.
            if composer is None:
                self._verify_sheet_shows(sheet, payload, model_item_index, before)
            else:
                self._verify_sheet_shows(
                    sheet, payload, model_item_index, before, composer_surface=composer,
                    allow_top_identity=bool(opener))
        if self._dbg is not None:
            # Best-effort, never raising (DebugLog.action swallows its own I/O failures) — a
            # broken debug log must never take down a like that would otherwise send cleanly.
            # Kept after the anchor mechanism itself went away: this frame is a picture of the
            # item a real comment is about to attach to, and doc 5.6's addendum records the
            # observe-side twin of it as the only corpus of open like sheets anyone has.
            self._dbg.action("like_anchor", before=sheet, item_index=item_index,
                             model_item_index=model_item_index, verified=payload is not None)
        if opener:
            self._raise_if_action_cancelled(should_stop, boundary="comment entry")
            if composer is None:
                self._tap_frac(self.coords["comment_box"])
            else:
                self._tap(*composer.comment_rect.center)
            if not self._interruptible_sleep(human_delay(0.5), should_stop):
                self._raise_if_action_cancelled(should_stop, boundary="comment entry")
            if payload is not None and composer is not None:
                focused = self._screencap()
                try:
                    focused_composer = locate_inline_composer(
                        focused, self._template("confirm"), threshold=0.8)
                except ComposerDetectionError as exc:
                    raise HingeTargetingError(
                        f"{self.spec.app}: focusing the inline comment field did not leave a "
                        "structurally verified composer on screen. The opener is NOT typed and "
                        "the like is NOT sent.", stage="verify", intended=model_item_index,
                        index_space="model_items") from exc
                self._verify_sheet_shows(
                    focused, payload, model_item_index, before,
                    composer_surface=focused_composer, allow_top_identity=False)
            self._raise_if_action_cancelled(should_stop, boundary="text entry")
            self.adb.text(opener)                     # opener sent WITH the like (Signals #2)
            if not self._interruptible_sleep(human_delay(0.6), should_stop):
                self._raise_if_action_cancelled(should_stop, boundary="send like")
        self._raise_if_action_cancelled(should_stop, boundary="send like")
        # Focusing the inline field opens the keyboard and moves the whole composer.  Locate it
        # again immediately before the irreversible send; the surface returned before typing is
        # deliberately not reused across that layout transition.
        send_composer = self._await_sheet_open(tries=3)
        self._raise_if_action_cancelled(should_stop, boundary="send like")
        if send_composer is None:
            self._tap_frac(self.coords["send_like"])
        else:
            self._tap(*send_composer.confirm_point)
        time.sleep(human_cooldown(0.6))               # let the send register / upsell modal animate in
        # Sending is only an attempt until the post-send frame proves Hinge accepted it. Keep a
        # separate attempt record because a paywall can appear only AFTER the send tap; the
        # completed ``like`` record below must never claim that refused action landed.  This is
        # intentionally before the Rose-modal helper: that helper may refuse a stuck paid
        # interstitial, but the Send Like tap has already been issued and must remain diagnosable.
        attempt_fields = dict(heart=list(heart), opener_chars=len(opener or ""),
                              item_index=item_index, model_item_index=model_item_index,
                              verified=payload is not None)
        self._dbg_action("like_attempt", before, **attempt_fields)
        rose = self._handle_rose_upsell()             # paid-upsell interstitial: dismiss, NEVER pay
        action_fields = {**attempt_fields, "rose_modal": rose}
        try:
            self._verify_like_landed(before)
        except HingeDeckBlockedError as exc:
            self._dbg_action("like_rejected", before, **action_fields,
                             rejection="deck_blocked", reason=str(exc))
            raise
        self._dbg_action("like", before, **action_fields)

    def _deliver_decision(self, decision: str):
        """Issue one like/pass, by whichever gesture this app's spec calls for.

        Returns the tapped point, or None when the decision was delivered as a card drag
        (there is no single point to log in that case). Both paths are equally humanized;
        they differ only in what the phone receives, and therefore in what can go wrong:
        a tap can land on a neighbouring control, a drag cannot.

        _require_deck_confirmed() runs FIRST, unconditionally, for both gestures — this is
        the single chokepoint every autonomous decide passes through (see
        UnconfirmedScreenError), so a card_swipe app (Bumble) gets the same "prove this is
        the deck before acting" guarantee a tap app gets implicitly from vision-locating its
        button."""
        self._require_deck_confirmed()
        if self.spec.decide_gesture == "card_swipe":
            self._decide_by_card_swipe(decision)
            return None
        point = self._await_button("like" if decision == "like" else "pass")
        self._tap(*point)
        return point

    def _like_direct(self, opener: str | None, item_index: int | None, *,
                     model_item_index: int | None = None) -> None:
        """Bumble's flow: one like, no comment sheet, no per-item targeting.
        `model_item_index` joins the list of parameters accepted for parity and ignored: doc
        5.6's post-tap check verifies the COMMENT SHEET against a stored crop, and this flow has
        no sheet — `_item_enumeration_blocker` refuses to enumerate for an app that cannot attach
        an opener at swipe time, so there are no crops to verify against either.
        `opener`/`item_index` are accepted only for interface parity with the comment_sheet flow
        and are otherwise unused — accepts_opener is False for every spec using this flow (Bumble
        is match-first-then-message, so there is no swipe-time opener to attach), so worker.py
        never actually passes a real `opener` here.

        Consequently this flow can never raise HingeTargetingError: there is no per-item target to
        miss, so the never-substitute rule has nothing to protect here."""
        before = self._snap()
        like_btn = self._deliver_decision("like")
        time.sleep(human_cooldown(0.6))                # let it register / an upsell modal animate in
        upsell = self._handle_rose_upsell()             # paid-upsell interstitial: dismiss, NEVER pay
        self._dbg_action("like", before, like=list(like_btn) if like_btn else None,
                         gesture=self.spec.decide_gesture, opener_chars=0, upsell_dismissed=upsell)
        self._verify_progress(before, "like")

    def dislike(self) -> None:
        before = self._snap()                         # snapped immediately before acting (no scroll between)
        try:
            x = self._deliver_decision("pass")        # vision-located X, or a card drag per spec
            self._dbg_action("dislike", before, x=list(x) if x else None,
                             gesture=self.spec.decide_gesture)
            self._verify_progress(before, "dislike")
        finally:
            # Same rule as like(): the card on screen is no longer the card that was enumerated.
            # See _invalidate_item_index and doc 5.3.
            self._invalidate_item_index(
                "the deck advanced after this pass, so anything enumerated for the previous "
                "profile no longer describes what is on screen")

    def observe_pass(self, *, should_stop=None) -> None:
        """Perform and verify a reviewed Observe pass with a frame-bound decision record."""
        with self._observe_input_lease("observe_pass"):
            self._raise_if_action_cancelled(should_stop, boundary="reviewed pass preflight")
            before = self._snap()
            self._raise_if_action_cancelled(should_stop, boundary="reviewed pass input")
            try:
                x = self._deliver_decision("pass")
                if not self._interruptible_sleep(human_cooldown(0.6), should_stop):
                    self._raise_if_action_cancelled(should_stop, boundary="reviewed pass verification")
                self._verify_progress(before, "dislike")
                self._dbg_action("observe_decision", before, decision="pass", reviewed=True,
                                 x=list(x) if x else None, gesture=self.spec.decide_gesture)
            finally:
                self._invalidate_item_index(
                    "the deck advanced after this reviewed Observe pass, so anything enumerated "
                    "for the previous profile no longer describes what is on screen")

    # --- observe mode: identity anchor + gesture corroboration ---------
    # See the module docstring's redesign notes and HINGE_SPEC's identity_band/content_band/
    # observe_ignore_zones/observe_touch_watch comments for the measured ground truth these
    # methods are built on. Everything below is READ-ONLY perception; nothing here taps,
    # swipes, or types (see wait_for_decision's own contract, unchanged).

    _OCR_NAME_RE = re.compile(r"[^A-Za-z0-9 '\-]")
    # Cache sizing for _ocr_band's memo below (see that method's own comment for the measured
    # cost it exists to avoid). A sentinel object, not None, marks "not cached" -- a genuine
    # OCR miss is itself represented as a cached None, and dict.get needs a default that can
    # never collide with a real stored value to tell the two apart.
    _OCR_BAND_CACHE_MAX = 4
    _OCR_BAND_CACHE_MISS = object()

    def _ocr_band(self, frame: bytes, rect: tuple[float, float, float, float], *,
                  psm: str = "7", white_text_threshold: int | None = None) -> str | None:
        """Best-effort OCR of `rect` via the host's `tesseract` binary (MEASURED reliable
        recipe on the Pixel 7a 2026-08-10: crop tightly -- a wider crop that includes photo
        content makes tesseract's page segmentation fail -- upscale 3x LANCZOS, `tesseract
        stdin stdout --psm <psm>`).

        `psm` defaults to `"7"` (treat the crop as a single text line), which keeps every
        existing call site -- the sticky-header identity_band, whose whole point is that it
        renders the name alone on one line -- byte-identical to before this parameter existed.
        identity_top_name_band's card-header crop is a DIFFERENT shape: it holds two or three
        short lines (the name, then a "Signals Active today" row, depending on layout -- see
        that field's own comment), and `--psm 7` MEASURED empty or garbled output against it
        on the Pixel 7a 2026-08-10 -- forcing a single line onto multi-line content confuses
        tesseract's segmentation. `--psm 6` ("assume a single uniform block of text") is what
        actually reads it correctly, on every scroll-top frame tested, in both layouts Hinge
        renders. Callers pass `psm="6"` explicitly for that band; nothing here chooses it
        automatically, because doing so from `rect` alone would silently couple this method's
        behavior to which band happens to be which shape today, instead of to a measured fact
        the caller already knows.

        NEVER sufficient on its own to write a label: the pixel/deck/settle gates still apply,
        and a different card-header name must reproduce on the independent confirm frame. OCR
        is nevertheless load-bearing for *recall* when a real next card lands at scroll top:
        the pixel band then contains profile-independent filter chips and cannot positively say
        who is shown. Any failure here (no `tesseract` on PATH, decode error, garbled read,
        timeout) returns None; the caller keeps waiting rather than weakening the false-PASS
        guarantee. This method must never raise.

        MEMOIZED, bounded to `_OCR_BAND_CACHE_MAX` entries, keyed on `(rect, psm, sha1(frame))`.
        MEASURED on this machine 2026-08-10: ~103ms per call at psm 7 (the sticky identity_band)
        and ~110ms at psm 6 (the card-header identity_top_name_band) -- and both bands can be
        OCR'd on the SAME poll (the identity_band corroboration above Layer 1b, plus Layer 1b
        itself), against an `_OBSERVE_POLL_S` of 0.35s. That is up to ~220ms of subprocess time
        inside a 350ms poll -- a 60% duty cycle that can stall the loop and cost frames. The
        repeat this cache targets is common, not hypothetical: wait_for_decision's own
        settle/confirm frame 0.5s later is usually the identical screen, and
        `_is_current_profile_frame` re-enters `_identity_of` (and so this method) on frames
        already seen earlier in the same poll. A cache hit skips the PIL crop/resize AND the
        tesseract subprocess entirely -- it is a pure memo, never a behavior change: identical
        `(rect, psm, frame bytes)` always yields whatever the first real OCR of that input
        produced (including a cached `None` miss), and any input not seen before takes the
        exact same path as before this cache existed.

        `white_text_threshold` is OFF by default (None), and both pre-existing call sites leave
        it that way, so their behaviour -- and their cache entries -- are identical to before it
        existed. When set, the crop is binarized at that luminance (pixels ABOVE it are taken to
        be the text) and INVERTED before the upscale, i.e. tesseract is handed black text on a
        white background. This exists for one MEASURED reason: the recipe above assumes DARK
        text on FLAT chrome, which is exactly what identity_band and identity_top_name_band are,
        and it fails outright on WHITE text over a PHOTOGRAPH. On Hinge's out-of-free-likes
        paywall headline (2026-08-11, ops/calibration/hinge_out_of_likes_20260811.png) the plain
        recipe returns garbage -- MEASURED, literally "“Tikes for today" -- while
        binarize-and-invert at 180, 200 or 215 all return the exact headline, "You're out of
        free likes for today". Rather than duplicating this method's tesseract/cache/timeout
        plumbing in a second OCR helper, the one genuinely different STEP is a parameter; it is
        part of the cache key below, so a preprocessed read can never be served out of a plain
        read of the same frame (or vice versa).
        """
        if not self.observe_name_ocr:
            return None
        tesseract = shutil.which("tesseract")
        if tesseract is None:
            return None
        cache_key = (rect, psm, white_text_threshold, hashlib.sha1(frame).digest())
        cached = self._ocr_band_cache.get(cache_key, self._OCR_BAND_CACHE_MISS)
        if cached is not self._OCR_BAND_CACHE_MISS:
            return cached
        try:
            from io import BytesIO

            from PIL import Image
            im = Image.open(BytesIO(frame)).convert("L")
            w, h = im.size
            x0, y0, x1, y1 = rect
            crop = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
            if white_text_threshold is not None:
                # Binarize AND invert in one pass: a pixel brighter than the threshold is text
                # and becomes black (0), everything else becomes white (255). Done BEFORE the
                # upscale, which is the order the recipe was measured in -- LANCZOS on the
                # already-binary image keeps the glyph edges clean, while binarizing after an
                # interpolating resize would threshold the interpolated halo instead.
                crop = crop.point(lambda p: 0 if p > white_text_threshold else 255)
            crop = crop.resize((max(1, crop.width * 3), max(1, crop.height * 3)), Image.LANCZOS)
            buf = BytesIO()
            crop.save(buf, format="PNG")
            result = subprocess.run(
                [tesseract, "stdin", "stdout", "--psm", psm],
                input=buf.getvalue(), capture_output=True, timeout=5.0,
            )
            # Preserve Tesseract's line segmentation.  The scroll-top card-header crop is a
            # small block, not a single line: line 1 is the profile name, while line 2 holds
            # pronouns/activity metadata and line 3 can already contain the first card title.
            # Flattening that structure made a perfectly clean read such as
            # ``Jen &\nshe/her Active now`` become ``Jen sheher Active now``.  The name parser
            # then saw several plausible words, refused to identify Jen, and a real X press was
            # swallowed by the content-scroll fallback.  Replace punctuation with spaces (so
            # ``she/her`` cannot weld into ``sheher``) and normalize each line independently.
            # Single-line callers remain byte-for-byte equivalent apart from punctuation now
            # becoming a separator instead of disappearing; the paywall caller benefits too
            # because wrapped ``free\nlikes`` no longer becomes ``freelikes``.
            raw_text = result.stdout.decode("utf-8", "replace")
            lines = [
                " ".join(self._OCR_NAME_RE.sub(" ", line).split())
                for line in raw_text.splitlines()
            ]
            cleaned = "\n".join(line for line in lines if line)
            value = cleaned or None
        except Exception:  # noqa: BLE001 — OCR failure must stay fail-closed, never raise
            value = None
        self._ocr_band_cache[cache_key] = value
        if len(self._ocr_band_cache) > self._OCR_BAND_CACHE_MAX:
            self._ocr_band_cache.pop(next(iter(self._ocr_band_cache)))   # evict oldest (FIFO)
        return value

    def _identity_of(self, frame: bytes) -> tuple[str, float | None]:
        """('same' | 'new' | 'top' | 'unknown', distance).

        'same'    — the app's sticky per-profile header pixels match the captured profile.
                    Normally this is the same card and never a decision. The one deliberately
                    separate signal is ``_identity_name_candidate``: the mostly-white bands of
                    two different names can fall under the pixel threshold, so a clean tight
                    OCR mismatch remains eligible for wait_for_decision's repeated-name proof.
                    This method still returns ``same`` until that caller proves the pair.
        'top'     — the band shows the app's own scroll-top chrome (profile-independent), so
                    identity is simply not visible right now; the caller falls back to content
                    matching (_vertical_shift_match) rather than treating this as a mismatch.
                    UNLESS identity_top_name_band is declared and can resolve it by OCR'ing
                    the card header instead -- see the "Layer 1b" block below, added for the
                    incident where a pass at scroll-top (Zorva -> Qelix) had no name visible
                    in identity_band and fell through to a spurious content match.
        'new'     — the band shows a DIFFERENT profile's header.
        'unknown' — no identity_band declared for this app, the frame didn't decode, or this
                    profile never scrolled far enough during capture to reveal its own header
                    (self._identity_sig is still None) -- there is nothing to compare against
                    either way, so this must not be read as either 'same' or 'new'.

        `distance` is the mean abs diff (0..255) the verdict was decided on, or None when
        there was nothing to compare (declared for logging -- see wait_for_decision's
        observe_decision/observe_resync records).
        """
        # Reset on EVERY call, not just every profile: a call that doesn't reach the Layer 1b
        # block below (wrong state, OCR off, band not declared, ...) must not leave a stale
        # read from an EARLIER call sitting here for the next observe_scroll/observe_waiting
        # debug record to misattribute to a frame it was never actually read from. Same for
        # _identity_top_name_verdict: wait_for_decision reads it immediately after calling this
        # method, so a stale value from an earlier call must never survive to be misread as
        # THIS call's provenance.
        self._identity_top_name_read = None
        self._identity_top_name_verdict = None
        self._identity_name_candidate = None
        if self.identity_band is None:
            return "unknown", None
        band = _band(frame, self.identity_band)
        if band is None:
            return "unknown", None

        top_sig, id_sig = self._identity_top_sig, self._identity_sig
        state, dist = "unknown", None
        if id_sig is not None:
            dist = _band_dist(band, id_sig)
            if dist < self.change_threshold:
                state = "same"
            elif top_sig is not None and _band_dist(band, top_sig) < self.change_threshold:
                state = "top"
            else:
                state = "new"
        elif top_sig is not None:
            # No per-profile header was ever locked for this capture (self._identity_sig is
            # None), so the ONLY thing the scroll-top chrome can positively prove is 'top':
            # this frame IS a card's scroll-top. It cannot prove 'new'. A frame that merely
            # differs from the chrome is any scrolled frame of ANY card -- including the very
            # card being watched -- so answering 'new' there manufactures a verdict from
            # nothing, which is exactly what this method's own docstring forbids for the
            # id_sig-is-None case ("there is nothing to compare against either way, so this
            # must not be read as either 'same' or 'new'") and what the OCR block below
            # refuses to do on much stronger evidence ("OCR gets a veto on 'new' and no power
            # to create one").
            #
            # It is not academic: 'new' is the one verdict that SKIPS layer 2 entirely (see
            # wait_for_decision's `identity_state != "new"` guard), so a manufactured 'new'
            # goes straight to the deck-ready/settle check and can record a PASS for a human
            # who only scrolled. Two ways to get here with no id_sig: a profile short enough
            # that the sticky header never appeared during its capture (latent since the
            # identity redesign), and a capture whose frame 0 was not at scroll-top -- which a
            # Stop abandoned mid-read now makes reachable for the FIRST card of the next run
            # (see _ensure_session_top, which exists to stop that happening in the first
            # place; this is the belt to its braces).
            #
            # 'unknown' is the honest answer and is NOT a weaker outcome for a genuine card
            # advance: it falls through to layer 2's content match, and a real advance matches
            # nothing there and still reaches the same deck-ready + settle proof, corroborated
            # rather than assumed.
            dist = _band_dist(band, top_sig)
            state = "top" if dist < self.change_threshold else "unknown"

        # OCR corroboration: position-tolerant where the pixel band above is not, so it can
        # recognise the SAME profile through a header that shifted a few px (a banner
        # appearing/disappearing) and would therefore fail the pixel compare.
        #
        # Deliberately ASYMMETRIC: a name match normally only upgrades the verdict TO 'same'.
        # The narrow exception is a pixel ``same`` whose tight sticky-header OCR cleanly reads
        # one different name; that becomes only a candidate ``new`` and must survive the
        # caller's two-frame/deck-ready/name-repeat gates. The two errors are not equally costly. A false 'same'
        # costs at most a missed pass -- the loop keeps waiting, and the deck-ready + settle +
        # content checks downstream still have to agree before anything is recorded. A false
        # 'new' writes a WRONG TRAINING LABEL, which is the entire class of bug this redesign
        # exists to eliminate. And a mismatch here is a genuinely weak signal for 'new': the
        # band legitimately reads as the app's filter-chips row at scroll-top (measured
        # 2026-08-10: OCRs as "Signals ( Agev ) Height v", which matches no name and would
        # have flipped a correct 'top' verdict straight to 'new'), and tesseract garbles
        # perfectly ordinary names often enough that a non-match proves nothing on its own.
        # So an arbitrary OCR mismatch gets no power to create ``new``.
        identity_band_named_candidate = False
        if self.observe_name_ocr and self._identity_name:
            seen_name = self._ocr_band(frame, self.identity_band)
            if seen_name:
                stored = self._identity_name.strip()
                seen_tokens = _TOP_NAME_TOKEN_RE.findall(seen_name)
                if stored and any(_name_token_matches(tok, stored) for tok in seen_tokens):
                    state = "same"
                elif state == "same":
                    # The signature is mostly white header chrome. In the measured Allison ->
                    # Brittany incident two genuinely different sticky headers were only 6.81
                    # apart, below the 9.0 pixel ``same`` threshold. A tight psm-7 read of one
                    # different name is useful evidence against that background-dominated
                    # match, but it does NOT change this method's pixel verdict. The caller
                    # alone may combine the candidate with the same name on a settled,
                    # deck-ready confirm frame before a no-touch PASS can be recorded.
                    candidate = _clean_first_line_name_candidate(seen_name)
                    if candidate is not None:
                        self._identity_top_name_read = seen_name
                        self._identity_top_name_verdict = "new"
                        self._identity_name_candidate = candidate
                        identity_band_named_candidate = True

        # Layer 1b: OCR the CARD HEADER. It can resolve a verified scroll-top chrome state,
        # and it can veto a pixel-derived ``new`` when it sees the stored name.
        #
        # At scroll-top identity_band shows Hinge's profile-independent filter chips, rather
        # than a name.  The card header lower down does carry the name there.  It is also useful
        # when a scroll/header transition has left the thin pixel identity band in a transient
        # state that reads as ``new``: seeing the captured name in the card header is positive
        # evidence that this is still the current profile and must veto that pixel verdict.
        #
        # Gated on state in {"top", "new"} / observe_name_ocr / a stored name / a declared
        # band:
        #
        #   ``top`` is the geometry this crop was measured against, so an actual different-name
        #   candidate there may resolve the otherwise generic scroll-top pixels to ``new``.
        #   ``new`` is included only for the asymmetric, safe direction: a matching stored name
        #   vetoes a potentially transient pixel mismatch. On a scrolled frame the crop can be
        #   photo content and OCR can be garbage, so a NON-match is deliberately ignored when
        #   the pixels already said ``new``; it never strengthens that verdict.
        #
        #   observe_name_ocr / self._identity_name / identity_top_name_band all being set is
        #   the same "nothing to work with, don't pretend otherwise" guard the OCR
        #   corroboration above already applies: no stored name means nothing to compare
        #   against, and no declared band means this app was never measured for one (None is
        #   the default -- see that field's docstring).
        if (state in {"top", "new"} and not identity_band_named_candidate
                and self.observe_name_ocr and self._identity_name
                and self.identity_top_name_band is not None):
            text = self._ocr_band(frame, self.identity_top_name_band, psm="6")
            self._identity_top_name_read = text
            if text:
                tokens = _TOP_NAME_TOKEN_RE.findall(text)
                stored = self._identity_name.strip()
                # Match against the STORED profile name via _name_token_matches -- ratio OR
                # prefix, deliberately biased toward "same" -- see _NAME_MATCH_RATIO's own
                # module comment and _name_token_matches' docstring for the full calibration.
                # Any token clearing the bar is enough: the header holds at most a first name
                # plus a line or two of chrome, so there is no ambiguity in "which token was
                # the name" once one clears it.
                if stored and any(_name_token_matches(tok, stored) for tok in tokens):
                    state = "same"
                    self._identity_top_name_verdict = "same"
                elif state == "top":
                    # The scroll-top geometry is the one this crop was measured against, so a
                    # CLEAN different name can identify a real advance. Exactly one non-chrome
                    # candidate is required: the 2026-08-16 Julia failure's crop included photo
                    # texture and OCR'd dozens of unrelated words. Taking the first one would
                    # turn arbitrary image noise into a person's name and recreate the false
                    # PASS even after the repeated-name gate below. A measured real header read
                    # may include "Signals Active today", but those are all filtered as chrome,
                    # leaving exactly the name ("Zorva", "Qelix", ...). Zero or multiple
                    # candidates is inconclusive. It is still only a first observation:
                    # wait_for_decision requires the clean ``new`` read on the settle frame too.
                    # Tesseract's line boundary is important evidence here: the measured Hinge
                    # geometry puts the person's name on the FIRST non-empty line.  Pronouns,
                    # "Active now/today", and occasionally the first card title occupy later
                    # lines.  Looking for candidates across the flattened whole crop caused the
                    # reported Sammy -> Jen pass to be missed even though OCR had read ``Jen``
                    # correctly on both settled frames.  Restrict NEW-name candidates to the
                    # first line.  SAME-name vetoes above still inspect every line, preserving
                    # the deliberately conservative false-PASS guard if OCR reflows a stored
                    # name downward.  If OCR emits one noisy line, the existing exactly-one-
                    # candidate rule still refuses it (the Julia regression).
                    candidate = _clean_first_line_name_candidate(text)
                    if candidate is not None:
                        state = "new"
                        self._identity_top_name_verdict = "new"
                        self._identity_name_candidate = candidate
                # When the original pixel state was ``new``, a nonmatch might just be OCR of
                # photo content. It deliberately does nothing: only a matching stored name can
                # override pixels in that direction.
        # RESIDUAL LIMITATION, stated plainly: two consecutive profiles sharing one first name
        # still OCR to the same token here and still read as "same" -- this layer, like the
        # identity-band pixel check above it, only ever holds a FIRST name, so it cannot
        # distinguish "the profile didn't change" from "the profile changed to someone with the
        # same first name". That case still hangs (a false 'same' costs a wait, per the ratio
        # calibration's own reasoning above -- never a false PASS). Gesture corroboration
        # (observe_touch_watch, see _observe_gesture_verdict) is the layer that WOULD cover it,
        # by proving a human tap actually happened independent of what any OCR read says -- but
        # it is off on this device (Android withholds the touch event stream here; see
        # HINGE_SPEC's observe_touch_watch comment), so this gap is accepted risk, not solved.
        return state, dist

    def _observe_tap_slop_px(self) -> float:
        """How far a finger may travel and still be a TAP rather than a scroll.

        This is NOT the same number as _observe_tap_radius_px below, and conflating them is a
        real mislabelling risk rather than a style nit: the control radius is deliberately
        generous (12% of screen height, ~288px here) because it must absorb where a human's
        thumb lands ON a big floating button. Reused as tap slop it would classify a 200px
        flick -- an ordinary short read-scroll -- as a "tap", which then only has to land
        within the (equally generous) radius of the pass-X to be corroborated as a deliberate
        PASS. A scroll must never be able to become a decision, so tap-ness gets its own,
        much tighter threshold: 2% of screen height (~48px on the Pixel 7a's 2400px panel),
        comfortably above finger tremor on a stationary press and far below the shortest
        gesture anyone makes to move content."""
        _, h = self.adb.screen_size()
        return max(12.0, min(0.02 * h, 90.0))

    def _observe_tap_radius_px(self) -> float:
        """How close a corroborating tap/drag-end must land to count as "on" a control.
        12% of screen height (~288px on the Pixel 7a's 2400px-tall panel) -- generous enough
        to absorb ordinary Fitts-law tap variance on a floating bottom control, clamped to a
        sane absolute range so a degenerate screen_size() (a stub, a test double) can't
        produce a zero or unbounded radius."""
        _, h = self.adb.screen_size()
        return max(40.0, min(0.12 * h, 500.0))

    def _observe_locate_pass_x(self, frame: bytes):
        """Locate the pass-X glyph on `frame`, for gesture-corroboration's tap-radius check
        ONLY -- never used as an action target (see the class docstring of _tap/_await_button
        for why a vision hit is never blindly trusted as a target on its own; this call site
        never taps anything at all).

        Duplicates _observe_glyph_visible's contrast-tolerant matching (Hinge can render the
        glyph either polarity: a dark X, or a white outline in a dark circle) rather than
        reusing it directly, because that method returns a bool and _observe_deck_ready is
        off limits to change (see this module's redesign notes) -- but it calls the SAME
        underlying _match_glyph/_template primitives _observe_deck_ready already used to
        confirm this exact frame is deck-ready, so this is not a second independent vision
        pipeline, just the one extra call needed to get a POINT instead of a bool."""
        template = self._template("pass")
        hits = _match_glyph(frame, template, side="left", threshold=0.6)
        if hits:
            return hits[0]
        try:
            import numpy as np
            inverted = np.bitwise_not(template)
        except Exception:  # noqa: BLE001 — no usable template -> no location to corroborate against
            return None
        hits = _match_glyph(frame, inverted, side="left", threshold=0.6)
        return hits[0] if hits else None

    def _observe_gesture_verdict(self, confirm_frame: bytes) -> str:
        """Classify the human's own touch-stream gestures since this profile became ready for
        a decision (self._observe_since), to corroborate -- or refuse to corroborate -- the
        identity-and-deck-proven card advance wait_for_decision has already established by the
        time this runs. Returns:

          'pass'    — affirmative evidence of a real decision: a tap that landed within
                      _observe_tap_radius_px of the located pass-X (tap-gesture apps), or, for
                      decide_gesture='card_swipe', a drag whose END landed within that radius
                      of coords['swipe_pass_end']. wait_for_decision reports PASS.
          'resync'  — the card changed but nothing corroborates a HUMAN DECISION causing it:
                      every gesture since ready was a drag with no tap at all (a tap-gesture
                      app's card cannot advance from a drag), or the human's last tap did not
                      land on the pass control. worker.py already treats a returned None as
                      "recapture, record nothing" (worker._observe_loop's `liked is None` branch) -- exactly right here.
          'no_data' — nothing to corroborate WITH: the watcher isn't configured/running, or it
                      has parsed zero events for the whole run (the health exception -- a
                      broken sensor is not affirmative evidence either way, see the printed
                      warning below). Callers may accept the advance only when the different
                      profile name was also positively read on both settled frames; a
                      pixel-only identity-band change resyncs without recording.
        """
        watcher = self._touch_watcher
        if not self.observe_touch_watch or watcher is None or not watcher.alive:
            return "no_data"
        if watcher.event_count == 0:
            # The stream has never delivered a single parsed line this whole run -- proof the
            # stream itself isn't working on this device (wrong node selected, a permissions
            # change mid-run, ...), not "the human genuinely never touched the screen". Only
            # the FIRST such run is worth a print; every subsequent poll would just repeat it.
            if not self._touch_watch_health_warned:
                self._touch_watch_health_warned = True
                print(f"{self.spec.app}: touch watcher has seen no events this run; falling "
                      f"back to repeated next-profile name proof (check `adb shell getevent` "
                      f"on this device).")
            return "no_data"

        radius = self._observe_tap_radius_px()      # "landed ON the control" -- generous
        slop = self._observe_tap_slop_px()          # "was a press, not a scroll" -- tight
        gestures = watcher.gestures_since(self._observe_since)
        w, h = self.adb.screen_size()

        if self.spec.decide_gesture == "card_swipe":
            drags = [g for g in gestures if not g.is_tap(slop)]
            pass_end = self.coords.get("swipe_pass_end")
            if not drags or pass_end is None:
                return "resync"          # nothing but taps (or nothing at all) can't drag a card
            target = (pass_end[0] * w, pass_end[1] * h)
            last = drags[-1]
            dist = math.hypot(last.up[0] - target[0], last.up[1] - target[1])
            return "pass" if dist <= radius else "resync"

        taps = [g for g in gestures if g.is_tap(slop)]
        if not taps:
            return "resync"              # only drags (or nothing) -- a tap deck can't advance from one
        last = taps[-1]
        for x0, y0, x1, y1 in self.observe_ignore_zones:
            if x0 * w <= last.up[0] <= x1 * w and y0 * h <= last.up[1] <= y1 * h:
                return "resync"          # rewind / overflow / bottom nav -- a known non-decision tap
        pass_point = self._observe_locate_pass_x(confirm_frame)
        if pass_point is None:
            return "no_data"             # can't locate the control to corroborate against -- degrade, don't veto

        # More than one tap landed ON the pass control during a single wait. The identity
        # anchor's one blind spot is that its band holds only a FIRST NAME, so two adjacent
        # profiles who share one render an identical header and a real advance can read as
        # 'same' -- the loop then keeps waiting through a decision it should have reported.
        # The touch stream is the independent witness to that: the human does not press pass
        # twice on one card, so a second pass-tap in the same window is direct evidence at
        # least one earlier decision was swallowed and this wait no longer knows which profile
        # it is holding. Resync (record nothing, recapture) rather than attach the accumulated
        # ambiguity to whatever Profile the worker captured before the wait began -- a missed
        # pass is recoverable, a label on the wrong person's photos is not.
        #
        # Only PASS-control taps count here. A heart tap followed by a pass tap is the
        # ordinary "opened the comment sheet, changed my mind, dismissed it, pressed pass"
        # sequence on ONE card -- two decisive taps, one decision -- and _await_like_resolved
        # has already resolved the sheet half of that before this ever runs.
        on_pass = [g for g in taps
                   if math.hypot(g.up[0] - pass_point[0], g.up[1] - pass_point[1]) <= radius]
        if len(on_pass) > 1:
            return "resync"

        dist = math.hypot(last.up[0] - pass_point[0], last.up[1] - pass_point[1])
        return "pass" if dist <= radius else "resync"

    _OBSERVE_WAIT_NOTICE_S = 15.0
    # At most how often wait_for_decision's rate-limited "still watching" notice may repeat.
    # The incident this exists for was TOTALLY silent: the owner pressed X, the misclassified
    # scroll match kept the wait going, and nothing at all appeared in the console, the hub, or
    # actions.jsonl for the whole time before Stop was pressed by hand -- there was no signal
    # to distinguish "working as intended, still reading" from "stuck, go look at the phone".
    # This does not fix that misclassification (the identity/OCR/shift-match layers above are
    # what do); it exists purely so the NEXT time anything stalls -- for this reason or a new
    # one nobody has hit yet -- there is at least something to look at instead of silence.

    _OBSERVE_LIKE_NOTICE_S = 30.0
    # Same idea as _OBSERVE_WAIT_NOTICE_S above, but for the window in which the human has an
    # open like sheet and is composing a comment. Deliberately slower: composing a message
    # legitimately takes minutes (measured 3m44s in the audited run of 2026-08-10), so a 15s
    # repeat there would be nagging rather than reassuring. It is not zero, because that
    # compose window was the ONE wait in observe mode with no heartbeat at all -- the longest
    # wait in the whole mode, and the only one where a wedged sheet (glyph match stuck, Hinge's
    # sending UI never resolving) would show the operator nothing whatsoever, in a loop whose
    # observe-mode deadline is None, i.e. forever. That is precisely the silence the sibling
    # notice was written to eliminate, left in place in the worst spot for it.

    def _note_observe_waiting(self, reason: str, frame: bytes | None = None) -> None:
        """Rate-limited operator print + matching debug record for wait_for_decision's "still
        waiting, nothing recorded yet" branches. `reason` is one of the plain, per-branch
        strings the call sites below pass -- `same` (identity says this is still the captured
        profile), `scroll` (layer 2 recognised this as a scroll, not a card change), `no_change`
        (nothing has moved since the last poll yet), `not_deck_ready` (the card changed but the
        next screen isn't confirmed ready), `not_settled` (deck-ready once, but the settle
        recheck hasn't agreed yet), `like_candidate` (a bottom-only change occurred but no
        composer was observed), `like_sheet` (the human has a like sheet open and is composing),
        `like_sending` (a VERIFIED sheet closed and Hinge is still resolving the send) -- so a
        human reading the console or actions.jsonl mid-run gets the SAME vocabulary this file's
        own comments already use for these states, not a fresh set of words to map back onto
        them.

        The two like-sheet reasons repeat on the slower _OBSERVE_LIKE_NOTICE_S cadence and get
        their own wording: they are not "nothing could be classified", they are a state this
        driver understands perfectly well and is deliberately waiting out.

        `self._observe_last_notice` is set fresh at the top of every wait_for_decision call
        (one call = one profile's wait), so this can never go permanently quiet across
        profiles: a notice suppressed near the end of one profile's wait cannot suppress the
        FIRST notice of the next one.

        The rate limit applies only to REPEATS OF THE SAME reason. A change of reason is always
        announced, because the change is the informative part: `no_change` -> `not_deck_ready`
        is the transition that says the screen finally started moving, and suppressing it left
        the console and actions.jsonl asserting a stale reason for up to 15s -- or, for a
        short-lived state, omitting it from the record entirely, which is the one thing a
        heartbeat must never do.

        `frame`, when supplied, is the frame the verdict was actually computed from. It is
        logged as-is instead of taking a fresh screencap, which is both cheaper (one less ADB
        round-trip per notice, on the hot polling path) and more truthful: a full 1080x2400
        screencap costs ~1-2s here, so the freshly-grabbed image used to show the screen 1-2s
        AFTER the verdict -- and if the human tapped in that gap, the saved picture showed the
        NEXT state while the record next to it claimed "nothing has moved". That is actively
        misleading in the one artifact you open when diagnosing a stall.
        """
        like_wait = reason in _OBSERVE_LIKE_WAIT_REASONS
        interval = self._OBSERVE_LIKE_NOTICE_S if like_wait else self._OBSERVE_WAIT_NOTICE_S
        if reason != self._observe_last_reason:
            # A changed reason bypasses the full interval (the transition is the informative
            # part) but NOT this floor. Without it, two branches that disagree on a flapping
            # screen -- a sheet glyph matching on one poll and not the next -- would alternate
            # reasons every _OBSERVE_POLL_S and turn an anti-silence heartbeat into ~3 console
            # lines a second. A transition that flaps faster than the floor is noise, and the
            # next notice reports whatever the reason is by then anyway.
            interval = _OBSERVE_NOTICE_FLOOR_S
        now = time.monotonic()
        if now - self._observe_last_notice < interval:
            return
        self._observe_last_notice = now
        self._observe_last_reason = reason
        if like_wait:
            detail = ("your like sheet is still open -- take your time; I'm waiting for you to "
                      "tap Send Like or dismiss it" if reason == "like_sheet"
                      else "the sheet closed and Hinge hasn't shown the next card yet")
            print(f"{self.spec.app}: still watching -- {detail} (reason={reason}). Nothing is "
                  f"recorded until this resolves. If it stays like this with the phone showing "
                  f"something else, check the phone, or press Stop.")
        elif reason == "like_candidate":
            # Do not borrow `like_sending`'s wording here.  This state is reached from a
            # bottom-only pixel delta before a structurally verified composer has ever been
            # observed; asserting that a sheet "closed" was the misleading diagnostic in the
            # Hayley false-LIKE report.  It uses the ordinary (15s) cadence above because it is
            # not a human-paced compose state and has no confirmed app send to wait for.
            print(f"{self.spec.app}: still watching -- a bottom-only change looks like a possible "
                  f"like, but no Send Like sheet has been observed (reason=like_candidate). "
                  f"Nothing is recorded unless the sheet is actually seen and a new card is "
                  f"proved. If it repeats, check the phone, or press Stop.")
        else:
            # "no_change" is the one reason that means the screen has NOT moved at all, so it
            # must not be announced as "the screen changed" -- an operator who just tapped X and
            # reads that the screen changed would reasonably conclude the tap registered and the
            # driver merely failed to classify it, when in fact nothing happened on the phone
            # and the tap is the thing to retry. Getting that backwards sends them debugging the
            # wrong half of the system, which is exactly how the incident this notice exists for
            # was misread.
            observed = ("nothing has moved on screen yet" if reason == "no_change"
                        else "the screen changed but nothing could yet be")
            verb = "" if reason == "no_change" else " classified as a like or a pass"
            print(f"{self.spec.app}: still watching -- {observed}{verb} (reason={reason}). "
                  f"Observation is continuing; if you already tapped X or the heart, this is "
                  f"normal while the screen settles. If it repeats for a while, check the phone "
                  f"matches what you expect, or press Stop.")
        if frame is not None and self._dbg is not None:
            # Straight to DebugLog.action, bypassing _dbg_action: the whole point is to log the
            # frame already in hand rather than let _dbg_action grab a second, later one.
            try:
                self._dbg.action("observe_waiting", before=frame, reason=reason)
            except Exception:  # noqa: BLE001 — debug logging must never break the wait
                pass
            return
        self._dbg_action("observe_waiting", None, reason=reason)

    def _note_observe_like_outcome(self, base: bytes, sent: bool, *, sheet_seen: bool = True,
                                   top: float | None = None, bot: float | None = None) -> None:
        """Record how an opened like sheet RESOLVED: sent (a LIKE) or dismissed.

        The PASS path has written a full `observe_decision` record since the observe redesign;
        the LIKE path wrote nothing at all. `observe_like_anchor` fires on INTENT (the sheet was
        detected), not on resolution, so an actions.jsonl reader could not distinguish the four
        outcomes of an opened sheet -- sent, dismissed, stop/timeout, resync. Without this
        record, a capture of one profile can be followed by an anchor and a capture of another,
        with no on-disk evidence that the first profile was liked. The LABEL was never at risk
        (worker.py stores it either way); what was
        missing was the diagnostic trail, on the rarer and higher-value of the two decisions,
        and it broke the invariant bugreport.py documents (exactly one decision record per
        capture).

        `gesture` is deliberately reported as "not_checked" rather than run: Layer 3's
        corroboration (_observe_gesture_verdict) measures the distance from the last touch-up to
        the PASS control, so on a like it would report a confident "resync" for a tap that
        correctly landed on the heart -- a wrong answer is worse in a debug log than an honest
        absence.

        The attached frame is the freshest one that PROVED the composer, not the caller's
        `base` anchor. The PASS path's `base` is the card as it looked immediately before it
        advanced, because that loop refreshes it every scroll poll; on this path
        _await_like_resolved deliberately holds `base` frozen as a dismissal comparand for the
        whole wait, so it can be minutes stale by the time a like resolves. In the audited
        2026-08-14 run it was 5 minutes old and predated the heart tap entirely, showing a
        scrolled profile instead of the item and opener that were actually sent -- evidence for
        a screen the human had already left. `base` remains the fallback for the paths that
        never observed a composer at all.
        """
        evidence = self._observe_like_evidence or base
        fields = {
            "capture_truncated": getattr(self, "_current_capture_truncated", None),
            "profile_name": self._identity_name,
            "gesture": "not_checked", "watcher": self.observe_touch_watch,
            # A bottom-only delta is only a candidate.  Keep the evidence that promoted it
            # (or did not) beside every resolution record so a later report can distinguish a
            # verified composer dismissal from a benign scroll/toast candidate.
            "sheet_seen": sheet_seen,
        }
        if top is not None:
            fields["top"] = round(top, 2)
        if bot is not None:
            fields["bot"] = round(bot, 2)
        if sent:
            self._dbg_action("observe_decision", evidence, decision="like", **fields)
        elif sheet_seen:
            # Not a decision -- the human opened the sheet and backed out, and the wait
            # continues on the SAME card. Logged under its own action name so it can never be
            # miscounted as a decision, while still leaving a trace that the sheet was up.
            self._dbg_action("observe_like_dismissed", evidence, **fields)
        else:
            # The bottom half changed and resolved back to the same card, but the Send Like
            # glyph was never actually matched on ANY poll -- so calling this a dismissed like
            # sheet would assert something nobody observed. A snackbar ("Your like was sent"),
            # a toast, or a keyboard dismissal all land here. Recorded under a name that claims
            # only what happened, so the log stays trustworthy about the rarer, higher-value
            # class of event it sits next to.
            self._dbg_action("observe_bottom_delta", base, **fields)
        # This composer is resolved either way. A dismissal leaves the wait running on the
        # SAME card, so without this a later resolution on that card could still be filed
        # against the sheet the human already backed out of.
        self._observe_like_evidence = None

    def _observe_recognized(self) -> None:
        """Mark the frame just classified as POSITIVELY RECOGNIZED, re-arming the stuck-screen
        watchdog with a FRESH budget draw (see _OBSERVE_STUCK_S, _observe_stuck_budget, and
        _observe_stuck_bail).

        Called from every branch of the observe loops that can say what it is looking at: the
        like sheet is open, identity says this is still the captured profile, layer 2 matched a
        scroll, the deck is confirmed ready, or the like flow resolved onto a known screen.
        Deliberately NOT called from the branches that only know what the screen ISN'T -- those
        are the ones the budget is counting."""
        self._observe_last_recognized = time.monotonic()
        self._observe_stuck_budget_s = _observe_stuck_budget()

    def _observe_stuck_bail(self, frame: bytes) -> str | None:
        """Has observe mode been staring at a screen it cannot recognize for longer than this
        arm's drawn budget (self._observe_stuck_budget_s, floored at _OBSERVE_STUCK_S -- see
        _observe_stuck_budget)? Returns the operator-facing reason (having recorded and printed
        it) when the caller must give up, or None to keep watching.

        This is the bail-out that did not exist on 2026-08-11: worker.py calls
        wait_for_decision(timeout=None), so every "keep watching" branch of both observe loops
        was, before this, unbounded for ANY screen the driver could not classify. When Hinge
        replaced the deck with its out-of-free-likes paywall the run polled for 2.5 minutes and
        only stopped because the owner pressed Stop by hand.

        The reason is the SPECIFIC one when the paywall is recognisable, and otherwise a generic
        one that is careful to claim only what is actually true -- that a screen has been up for
        the ACTUAL elapsed time and could not be classified -- because the whole class of failure
        this guards is "something nobody has seen before is on screen", and a message that
        guessed at which thing would be wrong exactly when it matters most.

        Both the debug record and the print exist because they answer different questions later:
        the record carries the FRAME (what was actually on screen, the single most useful thing
        in a bug report about an unrecognised screen), and the print is what the operator sees
        live -- the hub tees stdout into its log panel. The action NAME distinguishes the two
        cases so `actions.jsonl` can be grepped for either.
        """
        if not self._observe_last_recognized:
            # Never armed -- something reached an observe wait without going through
            # wait_for_decision (a calibration tool, a test exercising _await_like_resolved
            # directly). Arm it here instead of reading "has recognized nothing since the epoch"
            # as "stuck": a watchdog whose unarmed state is 'already expired' would stop a run
            # the instant any path that forgot to arm it was taken, which is exactly backwards.
            # The safe failure for a guard like this is to keep watching.
            self._observe_last_recognized = time.monotonic()
            self._observe_stuck_budget_s = _observe_stuck_budget()
            return None
        stuck_s = time.monotonic() - self._observe_last_recognized
        if stuck_s <= self._observe_stuck_budget_s:
            return None
        paywall_reason = self._deck_blocked_reason(frame)
        # No trailing full stop, matching _deck_blocked_reason's two strings: these are published
        # verbatim as the hub's stop_reason, and the sentence they are embedded in supplies its
        # own punctuation (below, and in the hub).
        reason = paywall_reason or (
            f"{self.spec.app} has been showing a screen I can't recognize for "
            f"{format_duration(stuck_s)} — stopping so nothing is mislabelled; the phone "
            f"is untouched, check what's on screen"
        )
        self._blocked_reason = reason
        self._dbg_action("observe_blocked" if paywall_reason else "observe_stuck", frame,
                         reason=reason, stuck_s=round(stuck_s, 1))
        print(f"{self.spec.app}: STOPPING — {reason}. Nothing was recorded for this card: "
              f"whatever you last tapped, this driver never saw it complete, so recording a "
              f"decision here would be inventing one. The phone has not been touched.")
        return reason

    # --- observe mode (shadow learning) --------------------------------
    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None,
                          on_like_intent=None) -> bool | None:
        """Wait passively while holding the per-device OBSERVE input lease."""
        with self._observe_input_lease("wait_for_decision"):
            return self._wait_for_decision_unlocked(
                timeout=timeout, should_stop=should_stop, on_like_intent=on_like_intent)

    def _wait_for_decision_unlocked(self, timeout: float | None = 120.0, should_stop=None,
                                    on_like_intent=None) -> bool | None:
        """Block until you manually like/pass the current card, inferred from
        screencap deltas (no accessibility tree):

          LIKE — tapping a heart slides the comment / "Send Like" sheet up over the
                 BOTTOM while the photo stays up top (bottom changes, top doesn't).
                 We surface an opener through the callback, then PASSIVELY wait
                 for the human to type and tap Send Like. Only a new card -> True.
          PASS — the whole card advances to a DIFFERENT, deck-ready, SETTLED profile,
                 positively proven by the identity anchor and/or content match below
                 (never by "changed and unrecognised" alone) -> False.
          none — stop requested, deck empty, timeout, a card change with no positive
                 decide evidence (a resync -- see _observe_gesture_verdict), or the
                 stuck-screen watchdog giving up on a screen it cannot recognize
                 (see _observe_stuck_bail, which also leaves an operator-facing reason
                 in self._blocked_reason for worker.py to publish) -> None.

        ⚠️ LIVE-VERIFY: the like-sheet geometry is gated until the profile is
        finished, so the top/bottom thresholds must be confirmed on-device before
        trusting observe labels. After "READY", tap Hinge's X or heart -- reading the
        profile by scrolling first is fine and expected (Signals behavior #1): the
        identity anchor below is exactly what makes that safe to do.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        # Cleared per wait, never merely overwritten: a leaked frame would file THIS
        # profile's like against the previous profile's composer.
        self._observe_like_evidence = None
        base = self._await_live_frame(deadline, should_stop)
        if base is None:
            return None
        # "When this profile became READY" for _observe_gesture_verdict's gesture-window
        # query below -- set once per call, before the poll loop, so a tap the human made
        # while still reading (before this wait even started watching) can never be
        # mistaken for corroboration of a LATER, unrelated card advance.
        self._observe_since = time.monotonic()
        # Rate-limit anchor for _note_observe_waiting -- reset here (once per call = once per
        # profile's wait), never inside the loop, so it "can never go permanently quiet" is
        # true across the WHOLE run, not just within one profile: see that method's docstring.
        # The reason is cleared alongside it so the first notice for a NEW profile is never
        # suppressed as a repeat of whatever the previous profile happened to end on.
        self._observe_last_notice = time.monotonic()
        self._observe_last_reason = None
        # Stuck-screen watchdog, armed here for the same reason and in the same place as the
        # notice anchor above: one call = one profile's wait, so the budget is per-profile and
        # never leaks across profiles. A FRESH budget is drawn here too (see
        # _observe_stuck_budget) rather than reusing whatever the previous profile's wait last
        # drew. Every branch below that can say WHAT it is looking at calls _observe_recognized()
        # to re-arm it, which draws its own fresh budget in turn; the branches that only know
        # what the screen is not are exactly the ones it counts. See _OBSERVE_STUCK_S.
        self._observe_last_recognized = time.monotonic()
        self._observe_stuck_budget_s = _observe_stuck_budget()
        self._observe_stuck_probe_at = self._observe_last_recognized
        self._observe_stuck_probe_interval_s = human_delay(_OBSERVE_STUCK_CHECK_S)
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            cur = self._screencap(on_blank="none")
            if cur is None:                           # screen asleep: the owner stepped away.
                time.sleep(_OBSERVE_POLL_S)           # keep watching — do NOT diff a black frame
                continue                              # against `base` (that reads as a phantom pass)

            # Checked here, at the top of the poll, against everything the PREVIOUS polls
            # concluded -- so it fires only after this arm's drawn budget (floored at
            # _OBSERVE_STUCK_S -- see _observe_stuck_budget) of unbroken failure to recognize the
            # screen, whichever branches those polls took.
            # Returns None, never True/False: the whole point is that nothing on screen was
            # understood, and None is worker.py's "record nothing". A LIKE label here would be
            # fabricated -- in the incident this fixes, the like the owner tried to send was
            # REFUSED by Hinge and never went out.
            if self._observe_stuck_bail(cur) is not None:
                return None

            # Check the actual compose sheet BEFORE interpreting aggregate screen deltas.
            # Opening the keyboard can change both halves at once, which the generic branch
            # below would otherwise call a card advance/PASS.  A visible sheet is instead an
            # in-progress human like until it is dismissed or a ready next deck card appears.
            if self._observe_like_sheet_visible(cur):
                # `cur` is passed as the anchor because it is ALREADY the frame that just
                # proved the Send Like glyph is visible — no extra screencap is taken for
                # this. The sheet slides up over the BOTTOM while the liked photo stays up
                # top (see this method's own docstring above), so this one full-screen frame
                # already contains both the item the comment attaches to AND the composer
                # itself. Re-capturing here would add ADB latency and an animation-timing
                # race for zero gain over what's already in hand.
                self._observe_recognized()            # an open like sheet is a screen we know
                self._observe_like_evidence = cur     # freshest proof of the composer so far
                self._notify_observe_like_intent(on_like_intent, True, cur)
                sent, intent_notified = self._await_like_resolved(
                    base, deadline, should_stop, on_like_intent=on_like_intent,
                    intent_notified=True,
                )
                if sent is None:
                    return None
                self._note_observe_like_outcome(base, sent)
                if sent:
                    if intent_notified:
                        self._notify_observe_like_intent(on_like_intent, False)
                    return True
                if intent_notified:
                    self._notify_observe_like_intent(on_like_intent, False)
                base = self._await_live_frame(deadline, should_stop)
                if base is None:
                    return None
                continue
            top, bot = _split_diff(base, cur)
            if top < self.change_threshold and bot < self.change_threshold:
                # THE SUBTLE CASE for the watchdog. This fast path returns BEFORE any
                # classification runs, so a STATIC unrecognised screen -- which is exactly what
                # the paywall is, nothing on it moves -- produces `no_change` forever and looks
                # identical to a human sitting still and thinking. Both must be handled, and
                # they pull in opposite directions: a human deliberating on a real deck may take
                # as long as they like (in the incident run the owner spent 01:22:33 -> 01:27:14
                # on ONE profile, nearly five minutes, which is normal use and must never be
                # interrupted), while a screen that isn't the deck at all must not be waited on
                # forever. So the timer is NOT re-armed unconditionally here: instead, at most
                # once every drawn probe interval (anchored at _OBSERVE_STUCK_CHECK_S and
                # humanized the same uniform way as the stuck budget itself -- see
                # _OBSERVE_STUCK_S's comment; this costs two template matches roughly per 5s, not
                # per 0.35s poll), ask whether a real deck is actually underneath. Deck ready ->
                # the human is deliberating, re-arm. Not ready -> leave the budget running and
                # draw the next interval.
                #
                # The deck-ready probe is necessary but NOT sufficient, and assuming it was cost
                # a second false stop of its own kind. Hinge hides the floating like heart at
                # some scroll offsets -- measured on the 2026-08-15 report's own frames, where
                # 00020/00021 are ordinary mid-read positions with the pass X visible and NO
                # heart, so _observe_deck_ready is False on them. An owner who scrolls to such a
                # position and then simply READS for the drawn budget (>= 90s, median ~104s --
                # entirely normal on a long prompt answer; this very run held one screen for
                # 2m20s) would never re-arm, and the watchdog would stop the run claiming it
                # could not recognize the screen -- while the screen was the profile being
                # watched, controls and all.
                #
                # So ask the identity anchor as well. A band that still matches the captured
                # profile's sticky header IS a positive recognition of exactly the thing this
                # loop is watching, which is what _observe_recognized documents itself as
                # meaning, and it is the same verdict Layer 1 below already treats as
                # authoritative. It does not weaken the paywall guard this watchdog exists for:
                # a paywall's band matches no captured header, so it still counts down and still
                # bails. Deck-ready is tried FIRST because it is pure template matching, so the
                # healthy case never reaches _identity_of's OCR at all.
                now = time.monotonic()
                if now - self._observe_stuck_probe_at >= self._observe_stuck_probe_interval_s:
                    self._observe_stuck_probe_at = now
                    self._observe_stuck_probe_interval_s = human_delay(_OBSERVE_STUCK_CHECK_S)
                    if self._observe_deck_ready(cur) or self._identity_of(cur)[0] == "same":
                        self._observe_recognized()
                self._note_observe_waiting("no_change", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue

            # A LIKE opens the comment / "Send Like" sheet: the photo (top half) stays put
            # while the BOTTOM changes. Detect this BEFORE the scroll check — otherwise, on a
            # light/gray profile, a like-sheet frame can match a stored full-frame signature
            # and be mis-read as a scroll, silently dropping the LIKE (bug C1).
            if bot >= self.change_threshold and top < self.change_threshold:
                # A bottom-only delta starts a *candidate* sheet flow. Do not spend
                # opener budget or replace the hub instructions until the actual
                # Send Like glyph corroborates it; a bottom animation/scroll alone
                # is not a human intent to like.
                sheet_visible = self._observe_like_sheet_visible(cur)
                if sheet_visible:
                    # Same reason as the sibling call site: record the proof now, in case the
                    # sheet closes before the resolver's first poll ever sees it.
                    self._observe_like_evidence = cur
                    self._notify_observe_like_intent(on_like_intent, True, cur)   # cur already proves the sheet -- see the sibling call site above
                sent, intent_notified = self._await_like_resolved(
                    base, deadline, should_stop, on_like_intent=on_like_intent,
                    intent_notified=sheet_visible,
                )
                if sent is None:
                    return None
                # sheet_seen comes from _await_like_resolved's own return, not from the single
                # `sheet_visible` probe above: the glyph can start matching on a LATER poll (the
                # sheet was still animating up when the bottom delta was first noticed), and it
                # can also never match at all -- a snackbar or toast produces the same
                # bottom-only delta with no sheet behind it. Only the resolver knows which
                # happened across every poll it made.
                self._note_observe_like_outcome(base, sent, sheet_seen=intent_notified,
                                                top=top, bot=bot)
                if sent:
                    if intent_notified:
                        self._notify_observe_like_intent(on_like_intent, False)
                    return True                       # like sheet resolved to a new card
                # Dismissal is not a pass. Clear the suggestion and continue waiting on
                # the same profile for the operator's next X/heart decision.
                if intent_notified:
                    self._notify_observe_like_intent(on_like_intent, False)
                base = self._await_live_frame(deadline, should_stop)   # cancelled -> resync
                if base is None:
                    return None
                continue

            # Top changed (whole card moved). LAYER 1 (identity) is authoritative and runs
            # FIRST: a matching sticky-header pixel signature normally proves this is still the
            # captured card -- the fix for manual scrolls once being recorded as PASS. The one
            # exception is a clean different-name candidate from that same tight band: shared
            # white chrome can make two names pixel-``same``, so the caller keeps that candidate
            # alive only long enough to demand the same name again on a settled, deck-ready
            # confirm frame. Layers 2/3 cannot override ``same`` without that named evidence.
            identity_state, identity_dist = self._identity_of(cur)
            # `_identity_of` resets its OCR provenance on every call, including the confirm
            # call below. Preserve this frame's read now so the decision/resync record says
            # whether header OCR rescued (or was unable to rescue) the pixel verdict.
            identity_name_read = self._identity_top_name_read
            identity_name_verdict = self._identity_top_name_verdict
            identity_name_candidate = self._identity_name_candidate
            if identity_state == "same" and identity_name_candidate is None:
                base = cur                            # scroll within the SAME profile -> keep waiting
                self._observe_recognized()            # identity named this card: recognised
                self._note_observe_waiting("same", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue

            # LAYER 2: legacy content match. An exact stored-signature hit, then a
            # vertical-shift search restricted to the scrolling CONTENT rows only -- see
            # _vertical_shift_match's docstring for the measured table explaining why the
            # earlier whole-frame version of this search could never fire in production.
            # Still reached (rather than short-circuited) when identity_state is 'top' or
            # 'unknown': a profile that hasn't revealed its sticky header yet, or an app with
            # no identity_band at all, gets exactly this content-only fallback, which is the
            # ONLY scroll-vs-pass discriminator this file had before the identity anchor.
            # ...but NOT when identity has already answered 'new'. Layer 2 compares whole
            # PHOTOS, and this file's own _is_current_profile_frame docstring records why that
            # is a weak cross-profile signal: dating first-photos are alike (centred face,
            # light background), so a genuinely different card can land under
            # change_threshold against one of the previous profile's captured frames. If that
            # collision were allowed to overrule the sticky header -- which is measured
            # 0.00-vs-~18 separated and is the whole basis of this redesign -- a real advance
            # would be swallowed as "just a scroll", the human's decision on the profile they
            # actually left would never be recorded, and their NEXT decision would be
            # attributed to the previous profile's photos by worker.py (which holds the
            # Profile object captured before this call and never re-derives it). The sibling
            # _is_current_profile_frame already returns on BOTH 'same' and 'new' before
            # consulting _current_sigs; this, the more consequential call site, simply never
            # had that precedence ported to it. 'top'/'unknown' still fall through, which is
            # the case Layer 2 exists for: the header isn't visible, so photos are all there is.
            ds_cur = _downsample(cur)
            # Indexed pairs (original _current_sigs position, sig), not a flat filtered list:
            # the whole point of the new sig_index debug field below is to name WHICH captured
            # frame matched (photos/_current_sigs are index-aligned -- see _capture_current's
            # own comment on that), which a re-numbered filtered list would get wrong the
            # moment any earlier frame in the profile was undecodable (a None slot).
            current_sigs = getattr(self, "_current_sigs", None) or []
            seen_pairs = [(idx, s) for idx, s in enumerate(current_sigs) if s is not None]
            min_dist = None
            shift_matched = False
            if (identity_state != "new" and identity_name_candidate is None
                    and ds_cur is not None and seen_pairs):
                import numpy as np
                dists = [(idx, float(np.mean(np.abs(ds_cur - ds_seen)))) for idx, ds_seen in seen_pairs]
                min_idx, min_dist = min(dists, key=lambda pair: pair[1])
                if min_dist < self.change_threshold:
                    # Swallowing a screen change is exactly as consequential as concluding a
                    # PASS from one, and until now it was the only silent path left in this
                    # function -- the same absence of a paper trail that made the original
                    # bug so hard to diagnose from a debug log. Best-effort, never gates.
                    self._dbg_action("observe_scroll", cur, reason="content_sig",
                                     identity=identity_state, min_sig_dist=round(min_dist, 2),
                                     sig_index=min_idx, name_read=self._identity_top_name_read)
                    base = cur                        # it's a scroll -> keep waiting
                    self._observe_recognized()        # layer 2 recognised this frame
                    self._note_observe_waiting("scroll", cur)
                    time.sleep(_OBSERVE_POLL_S)
                    continue
                content_rows = _content_rows(self.content_band, ds_cur.shape[0])
                band_rows = content_rows[1] - content_rows[0]
                # First match wins (same iteration-order semantics _vertical_shift_match itself
                # always had) -- sig_index/shift/overlap_rows below describe THAT match, which
                # is exactly the "matched stored frame 6 at a 12-row shift with only 4 rows
                # overlapping" detail the original incident's log had no way to show (see
                # _vertical_shift_match's own docstring for the return shape this reads).
                shift_hit = None
                for idx, ds_seen in seen_pairs:
                    matched, shift, overlap_rows = _vertical_shift_match(
                        ds_cur, ds_seen, threshold=self.change_threshold, rows=content_rows)
                    if matched:
                        shift_hit = (idx, shift, overlap_rows)
                        break
                shift_matched = shift_hit is not None
                if shift_matched:
                    sig_index, shift, overlap_rows = shift_hit
                    self._dbg_action("observe_scroll", cur, reason="shift_match",
                                     identity=identity_state,
                                     min_sig_dist=None if min_dist is None else round(min_dist, 2),
                                     sig_index=sig_index, shift=shift, overlap_rows=overlap_rows,
                                     band_rows=band_rows, name_read=self._identity_top_name_read)
                    base = cur                        # it's a scroll -> keep waiting
                    self._observe_recognized()        # layer 2 recognised this frame
                    self._note_observe_waiting("scroll", cur)
                    time.sleep(_OBSERVE_POLL_S)
                    continue

            # Neither identity nor content recognised `cur` as the captured profile. That is
            # NECESSARY for a PASS but not SUFFICIENT -- the OLD rule stopped right here
            # ("changed and unrecognised -> PASS"), which is exactly what let a human's manual
            # scroll into territory nothing had captured yet (identity 'top'/'unknown'/'new'
            # with no content match either) masquerade as a decision. LAYER 2's positive proof:
            # this must be a DIFFERENT, deck-READY, SETTLED card -- no like sheet, both deck
            # glyphs visible, and -- after a short settle -- the SAME still true on a SECOND
            # capture, with identity confirming that second capture is not the captured
            # profile either. Any single failure here means "not settled yet": keep watching,
            # conclude nothing, and do NOT advance `base` (a transient/animating frame is not
            # a safe anchor for the next diff).
            if self._observe_like_sheet_visible(cur) or not self._observe_deck_ready(cur):
                self._note_observe_waiting("not_deck_ready", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue
            # Falling through means the deck's own controls are BOTH visibly on screen: whatever
            # this card turns out to be, we are demonstrably still looking at Hinge's deck and
            # not at some screen nobody has seen before. That is a positive recognition, and it
            # is what keeps the settle/confirm loop below (which can legitimately re-poll for a
            # while on an animating deck) from ever being counted as a stuck screen.
            self._observe_recognized()

            time.sleep(0.5)
            confirm = self._screencap(on_blank="none")
            if confirm is None:
                continue                              # can't confirm blind -- re-poll
            confirm_identity_state, confirm_identity_dist = self._identity_of(confirm)
            confirm_identity_name_read = self._identity_top_name_read
            confirm_identity_name_verdict = self._identity_top_name_verdict
            confirm_identity_name_candidate = self._identity_name_candidate
            # A PASS label demands a positive, stable different-profile identity, not merely
            # two frames that failed to match the captured profile. In particular, Hinge fades
            # the filter chips/sticky header during a manual scroll; that transient band can be
            # just over the 9.0 pixel threshold and read ``new`` for one poll, while a later
            # scroll-top frame is only ``top``/``unknown``. Treating `!= "same"` as proof here
            # recorded exactly that manual read as a PASS. Requiring `new` on BOTH independently
            # captured frames makes the safe error a missed/resync decision, never a label on
            # the wrong person's profile.
            candidate_identity_advance = (
                identity_name_verdict == "new"
                and confirm_identity_name_verdict == "new"
                and identity_name_candidate is not None
                and confirm_identity_name_candidate is not None
                and identity_name_candidate.casefold()
                    == confirm_identity_name_candidate.casefold()
            )
            stable_identity_advance = (
                not self._observe_like_sheet_visible(confirm)
                and self._observe_deck_ready(confirm)
                and not self._changed(cur, confirm)
                and (identity_state == "new" or candidate_identity_advance)
                and (confirm_identity_state == "new" or candidate_identity_advance)
            )
            if not stable_identity_advance:
                self._note_observe_waiting("not_settled", confirm)
                time.sleep(_OBSERVE_POLL_S)
                continue                              # still settling / reverted -- keep watching

            # LAYER 3: corroborate the now identity-and-deck-proven advance against the
            # human's OWN touch stream, when this app is configured to read one (read-only --
            # see touchwatch.py; this driver never injects anything on this path).
            # A positively located pass-control gesture is sufficient corroboration even if
            # OCR could not read the next name.  With no gesture data, however, raw pixel
            # identity is NOT sufficient: the 2026-08-16 Julia incident produced two stable
            # ``new`` pixel verdicts from nothing more than Hinge vertically reflowing its
            # filter chips and card header. Both screenshots still showed Julia. In that
            # fallback state require the independent card-header OCR to have positively read
            # a different name on BOTH frames. Failure is an unlabeled resync, never a PASS.
            verdict = self._observe_gesture_verdict(confirm)
            # profile_name, not name: DebugLog.action's own first positional parameter is
            # `name` (the action-type string, "observe_decision"/"observe_resync" below) --
            # a fields key of literally `name` collides with it (see _capture_current's
            # identical note above).
            fields = dict(
                top=round(top, 2), bot=round(bot, 2),
                min_sig_dist=None if min_dist is None else round(min_dist, 2),
                shift_matched=shift_matched,
                capture_truncated=getattr(self, "_current_capture_truncated", None),
                identity=identity_state,
                identity_dist=None if identity_dist is None else round(identity_dist, 2),
                identity_name_read=identity_name_read,
                identity_name_verdict=identity_name_verdict,
                identity_name_candidate=identity_name_candidate,
                confirm_identity=confirm_identity_state,
                confirm_identity_dist=(None if confirm_identity_dist is None
                                       else round(confirm_identity_dist, 2)),
                confirm_identity_name_read=confirm_identity_name_read,
                confirm_identity_name_verdict=confirm_identity_name_verdict,
                confirm_identity_name_candidate=confirm_identity_name_candidate,
                profile_name=self._identity_name, gesture=verdict, watcher=self.observe_touch_watch,
            )
            if verdict == "resync":
                # The card DID change, but nothing corroborates a human decision causing it
                # (only a drag, or a tap that didn't land on the pass control). worker.py
                # already treats a returned None as "recapture, record nothing" (worker._observe_loop's `liked is None` branch)
                # -- exactly the right outcome for a resync, never a silent mislabel.
                self._dbg_action("observe_resync", base, **fields)
                return None
            name_advance_proven = candidate_identity_advance
            if verdict == "no_data" and not name_advance_proven:
                self._dbg_action(
                    "observe_resync", base,
                    reason="pass_identity_name_unconfirmed",
                    **fields,
                )
                return None
            self._dbg_action("observe_decision", base, decision="pass", **fields)
            return False                              # identity + deck-ready + settle + gesture/name -> pass
        return None

    def _notify_observe_like_intent(self, callback, active: bool,
                                    anchor: bytes | None = None, *, refresh: bool = False) -> None:
        """Best-effort notification for the passive Hinge observe flow.

        This helper intentionally performs no ADB input. If displaying or generating
        a suggestion fails, the human can still write their own message or dismiss
        the sheet, so observation must continue normally.

        `anchor` is a screencap of the like sheet as it is open on screen — the "which item does
        this comment attach to" picture. None when clearing (`active=False`) — there is nothing
        left on screen to take a picture of once the sheet has closed.

        WHAT THIS FRAME IS FOR CHANGED WITH DOC 5.9's INVERSION, AND THE PARAMETER DELIBERATELY
        DID NOT. It used to be the model's INPUT: observe generated after the tap, and the frame
        told the model which item the human had chosen. Observe now generates BEFORE the tap,
        from the same numbered crops auto sends, so the model is not told anything by this frame
        at all. It is now the EVIDENCE the worker checks the human's tap against
        (`observe_item_mismatch`): the item the human opened, compared against the item the
        suggestion was written for, with a warning and no text to type on a mismatch. Same frame,
        same call site, same one-shot notification — the receiver's job is what inverted.

        The frame is passed rather than re-captured for the reason it always was: `cur` has
        ALREADY proved the Send Like glyph is visible, the sheet slides up over the BOTTOM while
        the liked item stays up top, and `identity_band` (rows 115..226) sits above the sheet's
        preview (row 236), so this one full-screen frame carries the item, the composer AND the
        sticky header the identity check reads. A second screencap would add ADB latency and an
        animation-timing race for nothing.

        Instance method rather than the `@staticmethod` this used to be, so it can reach
        `self._dbg` for the best-effort anchor log below.

        Swallow-AND-PRINT below, not swallow-and-vanish: this hook is the ONLY way the
        operator's suggestion reaches the hub, and the old bare `except Exception: pass`
        meant a stale callback signature (an ordinary `TypeError` after some future
        refactor, say) would make suggestions vanish forever with zero evidence anything
        was ever wrong — nothing printed, nothing logged, nothing to grep for. The
        failure must still be non-fatal (observation must keep running with no
        suggestion rather than stop), but it must not stay invisible.
        """
        if callback is None:
            return
        # A sheet can settle its preview after its Send Like controls are already visible.  The
        # worker needs those later frames to correct a provisional item reading, but repeating
        # the provenance action on every 0.4 s poll would turn one human tap into a noisy log.
        if self._dbg is not None and active and anchor is not None and not refresh:
            self._dbg.action("observe_like_anchor", before=anchor)   # best-effort; DebugLog.action never raises
        try:
            callback(active, anchor)
        except Exception as exc:  # noqa: BLE001 — non-input side channel must not break observation
            print(f"{self.spec.app}: observe like-intent callback failed "
                  f"({type(exc).__name__}: {exc}); continuing observation without the suggestion.")

    def _observe_like_sheet_visible(self, frame: bytes) -> bool:
        """Whether Hinge's ``Send Like`` confirmation is still on screen.

        A keyboard can move enough of the profile to create a large top-region
        diff while the comment sheet remains open. The confirmation glyph is the
        authoritative guard against calling that an already-sent like.
        """
        if self.spec.like_flow != "comment_sheet":
            return False
        try:
            hits = _match_glyph(frame, self._template("confirm"), side="any", threshold=0.6)
            if not hits:
                return False
            # cv2's normalized matcher can report a mathematically-perfect hit on a tiny
            # flat synthetic/degraded frame (where the template cannot physically fit).  The
            # real Hinge control is in the central/lower compose sheet, so reject impossible
            # or top-bar positions before allowing this guard to block capture/decision flow.
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return False
            height = image.shape[0]
            return any(height * 0.25 <= y <= height * 0.85 for _x, y in hits)
        except Exception:  # noqa: BLE001 — retain delta fallback if visual matching is unavailable
            return False

    def _observe_deck_ready(self, frame: bytes) -> bool:
        """Whether ``frame`` is visibly a swipe deck ready for the next decision.

        A closed comment sheet is not, by itself, evidence that Hinge has advanced: its
        network/animation "sending" state also closes the Send Like control and can stay
        visually stable for seconds.  Require the existing independent pass-X and like-heart
        glyphs instead.  This method is observe-only perception; it never uses the matched
        positions as input targets.
        """
        try:
            like = self._observe_glyph_visible(frame, "like", side="right")
            passed = self._observe_glyph_visible(frame, "pass", side="left")
            return bool(like and passed)
        except Exception:  # noqa: BLE001 — inability to prove a ready deck must fail closed
            return False

    def _observe_glyph_visible(self, frame: bytes, role: str, *, side: str) -> bool:
        """Passive control detection, accepting either UI contrast polarity.

        This inverted-template fallback was written against the OLD "like" template
        (hinge_heart.png, a dark outline glyph) to catch the live deck's real button, which
        actually renders the opposite polarity (a white heart in a filled black circle) --
        see the templates dict comment on HINGE_SPEC and _LIKE_MATCH_THRESHOLD above for the
        full story. hinge_like_button.png (the CORRECT "like" template, now wired in) is
        already cropped at that same live polarity, so it no longer needs inverting -- and
        MUST NOT be inverted: bitwise-NOT of a white-heart-on-black-circle glyph reproduces
        Hinge's OUTLINE heart almost exactly (measured correlation ~1.0 against the "Which do
        we have in common" row hearts, the same false-positive the new template was built to
        avoid — see the templates dict comment). Testing the inverted template for "like"
        would therefore silently resurrect that exact bug for this perception-only path, so
        role == "like" skips it entirely and relies solely on the primary (uninverted) match,
        which measurement showed is already reliable (0.815..1.000 across 115 real frames).
        "pass" is unaffected -- its own template/polarity wasn't touched by this recalibration,
        so it keeps the original both-polarities behaviour.  Autonomous ACTIONS intentionally
        keep `_match_glyph`'s calibrated, single-polarity matcher regardless of role:
        broadening it would turn this perception-only readiness check into a new tap target.
        Here we only need evidence that a future deck has loaded.
        """
        template = self._template(role)
        threshold = _LIKE_MATCH_THRESHOLD if role == "like" else 0.6
        # y_band (content_band) is "like"-only, same rationale as _locate_button/
        # _locate_target_heart: the floating pass-X can legitimately sit outside content_band,
        # so "pass" gets no y_band restriction here either.
        y_band = self.content_band if role == "like" else None
        if _match_glyph(frame, template, side=side, threshold=threshold, y_band=y_band):
            return True
        if role == "like":
            return False
        try:
            import numpy as np
            inverted = np.bitwise_not(template)
        except Exception:  # noqa: BLE001 — no usable template means no proof of a deck
            return False
        return bool(_match_glyph(frame, inverted, side=side, threshold=threshold))

    def _await_like_resolved(self, base: bytes, deadline, should_stop,
                             *, on_like_intent=None,
                             intent_notified: bool = False) -> tuple[bool | None, bool]:
        """After the like sheet appears, wait for a human send or dismissal.

        Do not use a top-region change alone as evidence of sending: focusing the
        text field can shift the profile behind an otherwise-still-open sheet.
        Once the sheet glyph is gone, a *stable, visibly ready* new deck card is a sent
        like; closing the sheet onto Hinge's transient sending UI is deliberately neither.
        The base card (or another captured frame of the current profile) is a dismissal.

        This is also entered SPECULATIVELY, from wait_for_decision's bottom-only-delta branch,
        before any composer has been seen -- because a real one may still be animating up. In
        that state (`intent_notified` False) it is a *candidate* resolver, not a like resolver:
        it can only ever conclude "no like happened, keep waiting on this card" (False) or
        "the card is no longer the one we captured" (None -> resync), never a LIKE. Every
        verdict below that reads differently depending on which of the two states it is in says
        so explicitly; see the `require_content` comment in the loop for the measured incident
        on each side.

        This loop shares wait_for_decision's stuck-screen budget (self._observe_stuck_budget_s,
        floored at _OBSERVE_STUCK_S -- see _observe_stuck_budget), armed once per profile-wait by
        that method and re-armed with a fresh draw by every _observe_recognized() call, with a
        deliberate ASYMMETRY between its two waiting states -- see the `like_sheet` and
        `like_sending` branches below. This is where the 2026-08-11 incident actually hung, so it
        matters more here than anywhere else.
        """
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None, intent_notified
            cur = self._screencap(on_blank="none")
            if cur is None:                           # screen asleep mid-wait: keep watching
                time.sleep(_OBSERVE_POLL_S)           # (never diff a black frame against base)
                continue
            # The screen that can make this resolver wait forever is recognisable now.  Do not
            # make a human wait for the generic stuck-screen budget just because the paywall
            # appeared mid-wait (after Send Like rather than between profile loops).  Memoize the
            # exact same reason ``blocked_reason()`` would report; Worker sees it on the next loop
            # iteration and stops as blocked without recording a decision.
            blocked = self._deck_blocked_reason(cur)
            if blocked is not None:
                self._blocked_reason = blocked
                return None, intent_notified
            # (None, ...) on expiry, never (True, ...): the caller turns a True into a LIKE
            # label, and the whole reason we are giving up is that we never saw the like land.
            # In the incident, Hinge had REFUSED it (out of free likes) -- a LIKE label here
            # would have been fabricated from a like that never went out.
            if self._observe_stuck_bail(cur) is not None:
                return None, intent_notified
            if self._observe_like_sheet_visible(cur):
                # Advanced on EVERY sheet poll, not just the first: the composer's last
                # observed state is what the human actually sent, and it is minutes newer
                # than the `base` anchor this resolver deliberately holds frozen.
                self._observe_like_evidence = cur
                if not intent_notified:
                    self._notify_observe_like_intent(on_like_intent, True, cur)   # cur already proves the sheet -- see wait_for_decision's call sites
                    intent_notified = True
                else:
                    # The first structurally-valid composer frame can still be mid-animation:
                    # its selected-photo preview may be clipped differently from the settled
                    # sheet.  Re-check the live anchor while the sheet remains visible, so one
                    # such frame cannot leave the hub claiming a different item until the human
                    # manually scrolls/reopens it.  This is a refresh of one tap, not another
                    # tap, hence no duplicate `observe_like_anchor` debug action.
                    self._notify_observe_like_intent(on_like_intent, True, cur, refresh=True)
                # HALF of the watchdog's deliberate ASYMMETRY here (the other half is at the
                # `like_sending` notice at the bottom of this loop). `like_sheet` means the sheet
                # is OPEN and the HUMAN is composing a comment, which is human-paced and must
                # stay completely unbounded -- the owner may spend minutes writing one (measured
                # 3m44s in the audited run of 2026-08-10). So this re-arms the budget on every
                # poll it holds.
                self._observe_recognized()
                # The keyboard/sheet may radically alter the top half. It is still
                # an unsent human draft while the Send Like control is visible.
                self._note_observe_waiting("like_sheet", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue
            # A single negative glyph read is not enough to call a human's compose sheet closed.
            # The 2026-08-15 trace contained exactly that flap (like_sheet -> like_sending ->
            # like_sheet three seconds later).  Confirm the negative reading before entering
            # closed/sending resolution; a reappearing sheet is still an open, human-paced draft.
            if intent_notified:
                time.sleep(_OBSERVE_POLL_S)
                confirm_sheet = self._screencap(on_blank="none")
                if confirm_sheet is None:
                    continue
                if self._observe_like_sheet_visible(confirm_sheet):
                    self._observe_like_evidence = confirm_sheet
                    self._notify_observe_like_intent(
                        on_like_intent, True, confirm_sheet, refresh=True)
                    self._observe_recognized()
                    self._note_observe_waiting("like_sheet", confirm_sheet)
                    time.sleep(_OBSERVE_POLL_S)
                    continue
                cur = confirm_sheet

            # `require_content` is ASYMMETRIC on purpose, and the axis is `intent_notified`,
            # because the two states of this resolver pay opposite prices for a wrong verdict.
            #
            # SHEET OBSERVED (intent_notified) -- the strict two-signal rule, unchanged: a wrong
            # "still the current profile" throws away a like the human really sent, and a
            # same-first-name next card can make the header collide (see the flag's docstring).
            #
            # NO SHEET EVER OBSERVED -- there is no like to throw away. This branch can only
            # return False (nothing happened, keep waiting on this card) or None (resync); it
            # can never return True, so no content check is protecting a label here. The prices
            # are therefore REVERSED: the costly error is a wrong "not current", and it costs
            # the whole profile. That is the 2026-08-15 Alex report -- the owner only scrolled
            # to read, the sticky header stayed pixel-identical the entire time (measured 0.000
            # across every frame of that read, and wait_for_decision's Layer 1 had used exactly
            # that 'same' verdict to keep waiting three seconds earlier), but the manual scroll
            # offset matched none of the capture-time downsamples. require_content=True turned
            # an identity-proven SAME CARD into "not current", the deck glyphs were legitimately
            # on screen, and the no-sheet branch below resynced a profile nobody had decided on,
            # discarding a five-minute read and its opener.
            #
            # Deferring to the identity anchor here introduces no new risk class: it is the same
            # verdict the outer loop already treats as authoritative on every single poll. What
            # it removes is this resolver being quietly STRICTER than that loop while running on
            # WEAKER evidence -- a bottom-only delta, which a fling settling, a snackbar, or a
            # keyboard transition all produce.
            require_content = intent_notified
            current = cur == base or self._is_current_profile_frame(
                cur, require_content=require_content)
            ready = not current and self._observe_deck_ready(cur)
            if current or ready:
                self._observe_recognized()            # back on the known card, or on a ready deck
                # Both a dismissal and a ready deck must settle.  This rejects a single
                # transition frame and, for the ready case, proves the deck controls remain
                # present after Hinge's sending animation has completed.
                time.sleep(0.5)
                confirm = self._screencap(on_blank="none")
                if confirm is None:
                    continue                          # can't confirm blind -> re-poll
                if self._observe_like_sheet_visible(confirm):
                    continue                          # sheet reappeared / animation still resolving
                confirm_current = confirm == base or self._is_current_profile_frame(
                    confirm, require_content=require_content)   # same asymmetry as `current`
                if current and confirm_current:
                    return False, intent_notified     # genuinely back on the current profile
                if (ready and not confirm_current and self._observe_deck_ready(confirm)
                        and not self._changed(cur, confirm)):
                    # A bottom-only delta is deliberately only a *candidate*: a manual read
                    # scroll, snackbar, keyboard transition, or other bottom chrome movement
                    # can all satisfy it.  It may wake this resolver so a composer that is
                    # still animating in can be observed on a later poll, but it must NEVER
                    # become a LIKE merely because a later steady deck frame does not happen
                    # to match one of the capture-time downsampled frames.  The latter is the
                    # exact false-LIKE path from the 2026-08-14 Hayley incident.
                    #
                    # `intent_notified` flips only after `_observe_like_sheet_visible` has
                    # structurally found Hinge's inline input + CTA + Send Like glyph.  No
                    # sheet evidence means no human like evidence.  Return None (worker
                    # resyncs and records no preference) rather than False: False asserts a
                    # dismissal of a sheet we never observed, while True would invent a LIKE.
                    #
                    # Reaching here now MEANS the identity anchor could not name this card as
                    # the captured profile on either frame (`current` above consults it without
                    # the content requirement when no sheet was seen), so the card really may
                    # have moved on beneath us. Resync is still the right answer for that: the
                    # alternative -- keep waiting -- would leave the loop watching a card whose
                    # `_current_sigs`/profile are stale, and attribute the human's NEXT decision
                    # to the person they already left, which is the corrupted-label failure this
                    # whole design exists to prevent.
                    #
                    # The identity states are logged because their ABSENCE is what made the
                    # 2026-08-15 report expensive to diagnose: the record said only
                    # `current=False, deck_ready=True`, while the decisive fact -- that the
                    # sticky header still read the captured name -- was nowhere in it.
                    if not intent_notified:
                        identity_state, _identity_dist = self._identity_of(cur)
                        confirm_identity_state, _confirm_dist = self._identity_of(confirm)
                        self._dbg_action(
                            "observe_resync", base,
                            reason="like_candidate_without_observed_sheet",
                            sheet_seen=False,
                            profile_name=self._identity_name,
                            identity=identity_state,
                            confirm_identity=confirm_identity_state,
                            current=False,
                            deck_ready=True,
                        )
                        return None, False

                    # An observed composer establishes intent/opening, not a completed send.
                    # A stable deck is still not enough to prove it belongs to a DIFFERENT
                    # profile: a dismissed sheet followed by a manual scroll can make the old
                    # card look ready, and a same-first-name next card is ambiguous to this
                    # driver's identity anchor.  Reuse the PASS path's affirmative two-frame
                    # identity rule.  Without an anchor there is no way to distinguish that
                    # dismissal-plus-scroll sequence, so fail closed there too.  A missed LIKE
                    # is recoverable; assigning it to the profile held before this wait is not.
                    identity_state, _identity_dist = self._identity_of(cur)
                    confirm_identity_state, _confirm_identity_dist = self._identity_of(confirm)
                    if identity_state == "new" and confirm_identity_state == "new":
                        return True, True              # observed sheet -> stable, proven new deck
                    self._dbg_action(
                        "observe_resync", base,
                        reason=("like_send_identity_unavailable" if self._identity_sig is None
                                else "like_send_identity_unproven"),
                        sheet_seen=True,
                        profile_name=self._identity_name,
                        identity=identity_state,
                        confirm_identity=confirm_identity_state,
                        current=False,
                        deck_ready=True,
                    )
                    return None, True
            # A closed VERIFIED sheet but no current card and no ready deck = Hinge is still
            # processing.  Without that proof this is only a bottom-delta candidate (often a
            # human read-scroll), and must not claim a sheet closed or a send is in progress.
            # Keep observing; a timeout is unresolved, never a false cancellation/label.
            #
            # The OTHER half of the watchdog's asymmetry, and the exact state the 2026-08-11
            # incident hung in: `like_sending` is the APP working, not the human, and it must
            # resolve in seconds. So it deliberately does NOT re-arm the budget -- unlike
            # `like_sheet` above, which re-arms it on every poll. In the incident it never
            # resolved at all, because Hinge had refused the like and silently swapped the deck
            # for the out-of-free-likes paywall; this loop continued until the operator stopped
            # it.
            self._note_observe_waiting(
                "like_sending" if intent_notified else "like_candidate", cur)
            time.sleep(_OBSERVE_POLL_S)
        return None, intent_notified

    def _is_current_profile_frame(self, frame: bytes, *, require_content: bool = False) -> bool:
        """True if `frame` matches a captured frame of the CURRENT profile — i.e. an
        'advance' that is really a scroll within the same profile, not a new card.
        Guards the C1 reorder against a uniform-top scroll reading as a LIKE (#6).
        Conservatively False when undecodable (treat as a genuine advance).

        The audit-flagged weakness this used to carry: full-frame min-over-ALL-captured-sigs
        was the SOLE like-vs-scroll discriminator here, and dating first-photos are visually
        similar (centred face, light background), so a genuinely NEW card could collide with
        one of the current profile's sigs — a real LIKE then mis-read as a scroll, the like was
        dropped, and the observe loop desynced (the NEXT decision attributed to the wrong
        profile's photos). The fix was deferred for want of a signal that distinguishes two
        profiles better than "their photos look different".

        The identity anchor is exactly that signal, and it costs nothing to consult here: the
        app's sticky per-profile header (see _identity_of) is measured pixel-identical across
        every scroll offset of one profile and completely different across profiles, so it
        answers "same card or not" without depending on how alike two people's photos are.
        When it can answer ('same' or 'new') it is authoritative and the photo comparison is
        not consulted at all; a 'top'/'unknown' verdict — the header isn't visible, or this app
        declares no identity_band — falls through to the original full-frame behaviour, which
        is exactly what this method did before and no worse.

        `require_content` exists because the identity anchor has one blind spot, and the two
        callers pay wildly different prices for it. The band contains only the person's FIRST
        NAME, and it was measured WITHIN one profile (0.00 across three scroll offsets vs
        ~18 against that profile's own scroll-top chrome) -- never BETWEEN two people. Two
        different profiles who happen to share a first name render that header identically, so
        'same' can be wrong for a genuinely new card. In wait_for_decision's scroll-vs-pass
        check a wrong 'same' merely defers (keep waiting, decide nothing). In
        _await_like_resolved, ONCE A COMPOSER HAS BEEN OBSERVED, it DISCARDS: a like the human
        actually sent reads as a dismissal and is silently dropped. So that caller passes
        require_content=True in that state and gets 'same' only when the header AND the photos
        agree -- two independent signals that would both have to collide at once -- while the
        cheap identity-only path stays for the callers whose worst case is patience.

        That same resolver passes require_content=False BEFORE any sheet has been observed, and
        the reason is that the sentence above stops being true there: with no composer seen it
        cannot return a LIKE at all, so a wrong 'same' discards nothing and merely keeps
        watching, while a wrong 'not current' resyncs a card the owner is still reading. See its
        own `require_content` comment for the 2026-08-15 report that mis-set asymmetry produced."""
        state, _dist = self._identity_of(frame)
        if state == "new":
            return False                              # the header proves it is NOT this card
        ds = _downsample(frame)
        sigs = [s for s in getattr(self, "_current_sigs", None) or [] if s is not None]
        content_match = False
        if ds is not None and sigs:
            import numpy as np
            content_match = min(float(np.mean(np.abs(ds - s))) for s in sigs) < self.change_threshold
        if state == "same":
            return content_match if require_content else True
        return content_match                          # 'top'/'unknown': photos are all there is


class HingeDriver(AndroidDriver):
    """Hinge binding, including its versioned inline post-heart composer."""

    def __init__(self, cfg):
        super().__init__(cfg, HINGE_SPEC)

    def _await_sheet_open(self, tries: int = 5) -> ComposerSurface:
        """Return fresh, vision-located Hinge inline-composer geometry or refuse.

        Since Hinge 9.134.0 the selected item is reflowed in the profile and the controls are
        created beneath it; there is no modal to dismiss.  The old fixed comment/send fractions
        now land inside the selected card, so this override must never fall back to them.
        """
        surface = _retry_until(
            lambda: self._locate_inline_composer(self._screencap()), tries, 0.4,
            is_found=lambda value: value is not None)
        if surface is None:
            raise UnlocatedControlError(
                f"refusing to continue the like: the '{self.spec.app}' inline Send Like "
                f"composer was not structurally confirmed after {tries} attempts. Nothing was "
                "typed or sent, and the legacy fixed comment/send coordinates are not used.")
        return surface

    def _locate_inline_composer(self, frame: bytes) -> ComposerSurface | None:
        try:
            return locate_inline_composer(frame, self._template("confirm"), threshold=0.8)
        except ComposerDetectionError:
            return None

    def _observe_like_sheet_visible(self, frame: bytes) -> bool:
        """Passive proof of Hinge's inline composer, never a bare text-template match."""
        return self._locate_inline_composer(frame) is not None
