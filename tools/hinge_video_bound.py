"""Held-out video false-accept bound campaign for Hinge's still-photo dwell discriminator.

This implements section 4 (measurement protocol) of ops/STILL-PHOTO-DISCRIMINATOR.md, the
design of record. Read that file first; this module is only its harness.

WHY THIS TOOL EXISTS AT ALL. Section 2 objection 1 killed the previous candidate design with
GROUND-TRUTH CIRCULARITY: every "proven video" frame on disk exists because the 42x42 mute
template matched, so a bound measured on that corpus structurally excludes the exact population
it claims to bound -- videos whose control never rendered. The fix is not a better matcher. It
is a label channel the matcher cannot reach: the OWNER, at the phone, taps the card, watches
whether it plays, and types the answer. The mute matcher still runs here and its scores are
still persisted, but they are recorded as OBSERVATIONAL METADATA and are never, at any point,
allowed to become a label. `measure` and `emit` read the owner's typed label and nothing else.

WHAT THIS TOOL DOES TO THE PHONE: it reads. `adb exec-out screencap -p` for frames and one
`adb shell dumpsys package ... | grep versionName` for the build string. That is the complete
list of device commands. It performs no gesture of any kind and issues no event to the
touchscreen: the owner does every scroll and every card interaction by hand, exactly as in the
protocol, and the harness only screencaps and prompts on stdin. It therefore sits entirely
outside the humanized-interaction rules -- there is no gesture to humanize. It also never
likes, passes, comments, or sends anything, and holds no driver, ADB session, or touch
transport object. That is enforced structurally rather than by comment: nothing under
operation_love.drivers is imported at module scope. The three pure helpers this module does
borrow (a static template matcher, a band-arithmetic helper, an `adb devices` text parser) are
reached through narrow function-local imports at their single call sites, and none of them
constructs a device handle. tests/test_hinge_video_bound.py asserts both properties.

FOUR SUBCOMMANDS:

    python -m tools.hinge_video_bound hold-test --out ops/calibration/videobound_hold_<UTC>
    python -m tools.hinge_video_bound capture --out ops/calibration/videobound_<UTC> --profiles 70
    python -m tools.hinge_video_bound measure ops/calibration/videobound_<UTC> --config config.yaml
    python -m tools.hinge_video_bound emit    ops/calibration/videobound_<UTC> --config config.yaml

`hold-test` is the cheap falsifier of protocol step 5 and must be run FIRST. Park one known
video fully in view, hands off, and it screencaps for a minute. If any two consecutive frames
come back byte-exact over the content band, the dwell window has to grow or the design is
abandoned honestly -- not patched. It also reports the same statistic per 8-connected
changed-pixel row block, because a whole-band verdict lets a large static chrome region sit
next to a card that is quietly holding a frame; the per-block view is the sensitive one.

`capture` is the corpus campaign. Per card it waits for the screen to settle, records a
hazard-randomized dwell burst, and only THEN prompts for the label. The ordering is the
contamination guarantee: the burst is complete and closed before the prompt is printed, so the
owner's tap-to-play cannot appear in any dwell frame, and every frame captured after the label
belongs to the next card's burst. `measure` re-checks that ordering from the recorded
timestamps and refuses a session that violates it.

`measure` is PURE OFFLINE ANALYSIS: no ADB, no screencap, no device. It reads the PNGs and
manifest a previous `capture` wrote, re-verifies every frame digest, and computes the bound PER
CARD (never per frame pair -- see protocol step 3). It refuses, nonzero and without writing
anything, unless the section 4 thresholds are all met.

`emit` re-runs exactly that measurement and, only if it passes, freezes bound.json and prints
the `apps.hinge.still_photo_bound_evidence` block for the owner to paste into config.yaml by
hand. This tool never writes config.yaml, so binding a bound to a running config stays a
deliberate act -- and per section 3's gate split, those numbers install NUMBERING readiness
only. Auto stays blocked by its own separate release chain regardless of what is measured here.

PRIVACY: every frame saved here is a screenshot of a real person's dating profile. Output is
confined to gitignored ops/calibration/ and the tool refuses to run against any directory git
does not actually ignore. Nothing is uploaded or copied outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from operation_love import targeting_policy as _policy
from operation_love.private_files import (
    atomic_write_private_bytes, atomic_write_private_text, ensure_private_dir)
from tools._devicelock import holding_the_device

_TOOL_VERSION = "1"
_MANIFEST_SCHEMA_VERSION = 1
_ARTIFACT_SCHEMA_VERSION = 1
_CAMPAIGN_KIND = "hinge_still_photo_bound_campaign"
_HOLD_TEST_KIND = "hinge_still_photo_hold_falsifier"

# Section 4's thresholds have exactly one home: operation_love/targeting_policy.py, which config
# validation reads too.  They are resolved with getattr defaults purely so this module still
# imports against an older policy module; _POLICY_FALLBACKS names any that were not found, and
# the test suite asserts equality for every constant the policy module does export.
_POLICY_DEFAULTS = {
    "STILL_PHOTO_BOUND_MIN_VIDEO_CARDS": 60,
    "STILL_PHOTO_BOUND_MIN_PHOTO_CARDS": 60,
    "STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC": 0.05,
    "STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS": 0,
    "STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL": "owner_tap_to_play_v1",
    "STILL_PHOTO_DWELL_WINDOW_SAFETY_FACTOR": 3.0,
}
_POLICY_FALLBACKS = tuple(sorted(n for n in _POLICY_DEFAULTS if not hasattr(_policy, n)))
MIN_VIDEO_CARDS = getattr(_policy, "STILL_PHOTO_BOUND_MIN_VIDEO_CARDS",
                          _POLICY_DEFAULTS["STILL_PHOTO_BOUND_MIN_VIDEO_CARDS"])
MIN_PHOTO_CARDS = getattr(_policy, "STILL_PHOTO_BOUND_MIN_PHOTO_CARDS",
                          _POLICY_DEFAULTS["STILL_PHOTO_BOUND_MIN_PHOTO_CARDS"])
MAX_PHOTO_FALSE_REFUSAL_FRAC = getattr(
    _policy, "STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC",
    _POLICY_DEFAULTS["STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC"])
REQUIRED_VIDEO_ACCEPTS = getattr(_policy, "STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS",
                                 _POLICY_DEFAULTS["STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS"])
GROUND_TRUTH_CHANNEL = getattr(_policy, "STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL",
                               _POLICY_DEFAULTS["STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL"])
DWELL_WINDOW_SAFETY_FACTOR = getattr(_policy, "STILL_PHOTO_DWELL_WINDOW_SAFETY_FACTOR",
                                     _POLICY_DEFAULTS["STILL_PHOTO_DWELL_WINDOW_SAFETY_FACTOR"])

# --- the second, owner-accepted label channel (2026-08-21) ---------------------------------
# `tools/hinge_video_bound_auto.py` runs this same protocol unattended and derives each label
# from the mute matcher, the burst's own pixel-exactness and the crop classifier -- the very
# signals the accept rule reads.  That is section 2 objection 1's circularity, deliberately, and
# the owner accepted it in writing.  So `measure` reads such a campaign, but ONLY when its
# manifest carries the acceptance phrase verbatim, and every report and artifact it produces
# states the blind spot below.  Deliberately NOT in `_POLICY_DEFAULTS`: that dict is section 4's
# threshold set, which the test suite pins to the policy module in full, and these two are a
# label-channel identity rather than a threshold.
_CIRCULAR_POLICY_DEFAULTS = {
    "STILL_PHOTO_BOUND_CIRCULAR_CHANNEL": "ai_mute_glyph_circular_v1",
    "STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE": "I_ACCEPT_CIRCULAR_AI_LABELED_STILL_PHOTO_BOUND",
}
CIRCULAR_GROUND_TRUTH_CHANNEL = getattr(
    _policy, "STILL_PHOTO_BOUND_CIRCULAR_CHANNEL",
    _CIRCULAR_POLICY_DEFAULTS["STILL_PHOTO_BOUND_CIRCULAR_CHANNEL"])
CIRCULAR_ACCEPTANCE = getattr(
    _policy, "STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE",
    _CIRCULAR_POLICY_DEFAULTS["STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE"])
# One short factual sentence, frozen into the artifact and printed by every report of a circular
# campaign.  A reader who sees "0 video accepts" without it will read a false-accept bound that
# is simply not there, so it names what the campaign DOES deliver in the same breath.
LABEL_BLIND_SPOT = (
    "AI-labeled channel: the label and the accept rule read the same pixels, so the video "
    "accept count is zero BY CONSTRUCTION and this campaign is not evidence against a static "
    "video whose mute control never rendered. Its real deliverables are max_video_exact_run_s "
    "measured on playing videos, the still-photo false-refusal rate, and the persisted corpus.")

# Auto-mode owner rule: no timing parameter in this repository may be a fixed constant, because a
# fixed cadence is itself a bot signature.  These are the SPANS the per-burst hazard draws from;
# the drawn values differ every burst and the shape parameter of the law is itself redrawn.
_BURST_FRAMES_SPAN = (6, 10)
_BURST_WINDOW_S_SPAN = (8.0, 15.0)
_SETTLE_GAP_S_SPAN = (0.30, 0.90)
_SETTLE_MAX_READS = 60

_LABELS = ("video", "photo", "unsure", "skip", "done")
# The labels a CARD RECORD may carry.  Deliberately a separate tuple from `_LABELS`, which is the
# owner's typed prompt vocabulary: `written` is produced by the automated harness's classifier
# rung and is never something the owner is asked to type.  Like `unsure` and `skip` it is
# excluded from both denominators -- a Hinge prompt card is not a photograph, and the bound is
# about photo targeting -- but it is its OWN bucket so that `unsure` keeps meaning "cannot tell".
CARD_LABELS = ("video", "photo", "unsure", "skip", "written")
_DENOMINATOR_LABELS = ("video", "photo")
_LABEL_PROMPT = "card label? video/photo/unsure/skip/done > "
_READY_PROMPT = ("park the next card fully in view, take your hands off the screen, then press "
                 "Enter (or type done) > ")

_CALIBRATION_ROOT = Path("ops/calibration")
_PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_DEFAULT_PACKAGE = "co.hinge.app"

BOUND_ARTIFACT_KEYS = {
    "schema_version", "ground_truth_channel", "human_ground_truth", "device",
    "hinge_version_name", "frame_size_px", "captured_at", "config_sha256", "video_cards",
    "video_accepts", "photo_cards", "photo_false_refusals", "max_video_exact_run_s", "dwell",
    "cards", "evidence_sha256",
}
BOUND_CARD_KEYS = {"card_id", "label", "frames", "longest_exact_run_s", "accept"}
# Two keys the circular channel MUST carry and the owner channel must NOT.  The asymmetry is the
# same one operation_love/config.py enforces on the pasted mapping: an owner-labeled bound
# accepted no circularity at all, so silently carrying the field would misdescribe it.
CIRCULAR_ARTIFACT_KEYS = {"accepted_circular_risk", "label_blind_spot"}
# Config validation for this exact key set is implemented separately in operation_love/config.py.
# Keep the tuple and the validator's expectation identical; a paste block that does not match is
# a bug in one of the two, never something to paper over by hand-editing config.yaml.
PASTE_KEYS = ("artifact_path", "artifact_sha256", "ground_truth_channel", "video_cards",
              "video_accepts", "photo_cards", "photo_false_refusals", "max_video_exact_run_s",
              "captured_at", "device", "hinge_version_name")
# Appended, in this order, on the circular channel only -- again matching config.py's optional
# key.  The owner must re-state the acceptance in the file they paste, so binding a circular
# bound to a running config can never be a silent consequence of re-running the tool.
CIRCULAR_PASTE_KEYS = ("accepted_circular_risk",)

# --- offline adjudication of a captured campaign's own labels -------------------------------
# An optional <campaign>/adjudications.json lets a second reader (owner decision 2026-08-21: a
# vision model, on the owner's rule "videos all have a mute icon, no mute icon means picture")
# re-label cards the capture could not resolve.  It is an OFFLINE input only: it never touches
# the phone, it cannot add a card, and it cannot invent a frame.
#
# THE DIRECTION RULE IS THE WHOLE SAFETY ARGUMENT and it is not symmetric.  Only these three
# transitions exist:
#
#   unsure -> photo   grows the still-photo denominator, which bounds USEFULNESS (the false
#                     refusal rate).  It cannot manufacture a video accept, because a card
#                     labelled photo is measured by the same dwell arithmetic as every other
#                     photo and is refused if it behaves like a video.
#   unsure -> video   grows the video denominator, which is the side the bound is claimed on.
#   photo  -> video   strictly safety-increasing: it removes a card from the accept-eligible set.
#
# Everything else refuses, and `video -> anything` refuses hardest: a reader who could demote a
# video to a photo could erase exactly the population section 4 exists to count, and the owner's
# stated rule ("no mute icon means picture") is precisely the inference the design of record
# refuted in section 2 objection 2 -- complete-ROI pre-match frames of PROVEN videos score
# 0.267-0.305, and the control auto-hides.  So absence of the icon is allowed to move a card
# toward caution and is never allowed to move one away from it.
ADJUDICATION_FILENAME = "adjudications.json"
ADJUDICATION_DOCUMENT_KEYS = {"adjudicator", "entries"}
ADJUDICATION_ADJUDICATOR_KEYS = {"model", "process"}
ADJUDICATION_ENTRY_KEYS = {"card_id", "from_label", "to_label", "frame_sha256s", "rationale"}
LEGAL_ADJUDICATIONS = (("unsure", "photo"), ("unsure", "video"), ("photo", "video"))
# Present in the artifact only when a campaign was actually adjudicated, so a campaign with no
# adjudications.json produces byte-identical evidence to one measured before this existed.
ADJUDICATION_ARTIFACT_KEYS = {"adjudications"}
# Present when the campaign measured Hinge's autoplay-centring precondition (owner fact
# 2026-08-21: a video only plays near the centre of the screen, so byte-exactness measured
# off-centre proves nothing).  Frozen into the artifact because the accept rule it licenses is
# only valid for cards held inside the same zone, and a later reader has to be able to see which
# zone that was without re-deriving it from a tool version.
AUTOPLAY_ARTIFACT_KEYS = {"autoplay_center_band_frac"}
# Present only when more than one campaign directory was merged, so a single-directory artifact
# stays byte-identical to one produced before merging existed.
MERGE_ARTIFACT_KEYS = {"campaigns"}


class VideoBoundRefused(RuntimeError):
    """The campaign could not be run, or its evidence did not prove the section 4 bound."""


# =====================================================================================
# Digests, canonical JSON, repo-relative paths
# =====================================================================================

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _inside_repo(path: Path, *, label: str) -> str:
    root = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as exc:
        raise VideoBoundRefused(f"{label} must be inside the repository") from exc


def _read_json(path: Path, label: str):
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoBoundRefused(f"could not read {label}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise VideoBoundRefused(f"{label} must be a JSON object")
    return value, raw


# =====================================================================================
# Private, gitignored output
# =====================================================================================

def _ignored_by_git(path: Path) -> bool | None:
    """`git check-ignore`'s verdict for one path, or None when git cannot answer.

    Checking the resolved path rather than trusting the ops/calibration prefix is the point: if
    the ignore rule is ever edited away, this refuses instead of silently writing strangers'
    profile screenshots into a tracked directory.
    """
    try:
        result = subprocess.run(["git", "check-ignore", "-q", str(path)],
                                capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _private_out_dir(raw: str | None, *, prefix: str) -> Path:
    """Resolve a fresh output directory, refusing anything that is not actually gitignored."""
    root = _CALIBRATION_ROOT.resolve()
    if raw:
        out_dir = Path(raw).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = (_CALIBRATION_ROOT / f"{prefix}_{stamp}").resolve()
    try:
        out_dir.relative_to(root)
    except ValueError as exc:
        raise VideoBoundRefused(
            f"--out must stay under the gitignored {root}; every frame here is a real person's "
            f"dating profile and cannot be written to a tracked or external directory "
            f"(got {out_dir})") from exc
    if _ignored_by_git(out_dir) is False:
        raise VideoBoundRefused(
            f"git does not ignore {out_dir}; refusing to write private profile frames to a "
            "path that would be committable. Restore the ops/calibration/ rule in .gitignore.")
    if out_dir.exists() and not out_dir.is_dir():
        raise VideoBoundRefused(f"output {out_dir} exists and is not a directory")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise VideoBoundRefused(
            f"output directory {out_dir} already exists and is not empty; use a fresh directory "
            "so two campaigns cannot be merged into one bound")
    ensure_private_dir(out_dir)
    return out_dir


def _existing_campaign_dir(raw: str) -> Path:
    root = _CALIBRATION_ROOT.resolve()
    campaign = Path(raw).resolve()
    try:
        campaign.relative_to(root)
    except ValueError as exc:
        raise VideoBoundRefused(
            f"campaign directory must stay under private {root}; got {campaign}") from exc
    if not campaign.is_dir():
        raise VideoBoundRefused(f"campaign directory {campaign} does not exist")
    return campaign


# =====================================================================================
# Read-only device access
#
# Everything below shells out to adb directly.  There is deliberately no driver, no ADB session
# object, and no transport here: the complete device vocabulary of this tool is `exec-out
# screencap -p`, `devices`, and one `shell dumpsys package ... | grep versionName`.
# =====================================================================================

def _load_config_mapping(path: str | None) -> tuple[dict, bytes | None]:
    """Read config.yaml as plain data.

    Deliberately not `operation_love.config.load`: this campaign has to be runnable BEFORE the
    config key it exists to produce is present, so it must not depend on that key's validator.
    """
    if path is None:
        return {}, None
    try:
        raw = Path(path).read_bytes()
        value = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise VideoBoundRefused(f"could not read config {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise VideoBoundRefused(f"config {path} must be a mapping")
    return value, raw


def _hinge_app_config(cfg_map: dict) -> dict:
    apps = cfg_map.get("apps") or {}
    app = apps.get("hinge") if isinstance(apps, dict) else None
    return app if isinstance(app, dict) else {}


def _ready_devices(adb_path: str) -> list[str]:
    # The `adb devices` text format has one canonical parser in this repository; a second ad-hoc
    # one is how the `-l` trailing-column bug got written three times already.
    from operation_love.drivers.adb import parse_devices_output

    try:
        result = subprocess.run([adb_path, "devices"], capture_output=True, timeout=15,
                                check=False)
    except FileNotFoundError as exc:
        raise VideoBoundRefused(f"adb binary not found ({adb_path!r} not on PATH)") from exc
    except subprocess.SubprocessError as exc:
        raise VideoBoundRefused(f"`adb devices` failed: {type(exc).__name__}") from exc
    return parse_devices_output(result.stdout.decode("utf-8", errors="replace"))


def _resolve_serial(app_cfg: dict, override: str | None) -> tuple[str, str]:
    """One device, named explicitly. Refuse to guess which phone is on the bench."""
    adb_path = app_cfg.get("adb_path", "adb")
    if not isinstance(adb_path, str) or not adb_path.strip():
        raise VideoBoundRefused("apps.hinge.adb_path must be a nonempty string")
    configured = app_cfg.get("serial")
    configured = configured.strip() if isinstance(configured, str) else ""
    serial = (override or "").strip() or configured
    if not serial:
        raise VideoBoundRefused(
            "apps.hinge.serial is not configured and --serial was not given; a bound measured "
            "without a fixed serial could silently be attributed to the wrong device")
    ready = _ready_devices(adb_path)
    if len(ready) > 1 and not override:
        raise VideoBoundRefused(
            f"multiple ADB devices are ready {ready!r}; pass --serial to name the one phone this "
            "campaign is measured on")
    if serial not in ready:
        raise VideoBoundRefused(
            f"requested serial {serial!r} is not among the ready ADB devices {ready!r}")
    return serial, adb_path


def _screencap(serial: str, adb_path: str, *, timeout: float = 30.0) -> bytes:
    """One read-only frame. This is the only frame source in the entire module."""
    try:
        result = subprocess.run([adb_path, "-s", serial, "exec-out", "screencap", "-p"],
                                capture_output=True, timeout=timeout, check=False)
    except subprocess.SubprocessError as exc:
        raise VideoBoundRefused(f"screencap failed: {type(exc).__name__}") from exc
    if result.returncode != 0 or not result.stdout.startswith(_PNG_MAGIC):
        raise VideoBoundRefused(
            "screencap did not return a PNG frame "
            f"(rc={result.returncode}, {len(result.stdout)} bytes)")
    return result.stdout


def _device_version_name(serial: str, adb_path: str, package: str) -> str | None:
    """The live Hinge build string, or None when the phone cannot be asked.

    The parse is the same anchored end-of-line regex `operation_love/drivers/hinge.py` and
    `tools/hinge_operational_evidence.py` use, and for the reason the 2026-09-02 consolidation
    recorded: a bare `.split("=", 1)[1].strip()` silently accepts trailing garbage on the same
    dumpsys line, and this value is recorded verbatim into `bound.json`'s `hinge_version_name`,
    which config validation then binds the whole still-photo bound to.  What is deliberately NOT
    shared is that module's probe itself: it takes an `Adb` object, and this tool's complete
    device vocabulary is three raw adb commands and no driver session (see the module docstring).
    """
    if not _PACKAGE_RE.match(package):
        raise VideoBoundRefused(f"apps.hinge.package {package!r} is not an Android package id")
    try:
        result = subprocess.run(
            [adb_path, "-s", serial, "shell", f"dumpsys package {package} | grep versionName"],
            capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"(?m)^\s*versionName=(\S+)\s*$",
                      result.stdout.decode("utf-8", errors="replace"))
    return match.group(1) if match else None


# =====================================================================================
# Frames: decoding, the content band, byte-exact runs
# =====================================================================================

def _decode(frame: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise VideoBoundRefused("a saved frame could not be decoded as an image")
    return image


def _frame_size(frame: bytes) -> tuple[int, int]:
    image = _decode(frame)
    return int(image.shape[1]), int(image.shape[0])


def _effective_content_band(app_cfg: dict, override: str | None) -> tuple[float, float]:
    """The band the driver would actually use: the apps.hinge override, else HINGE_SPEC's."""
    if override:
        parts = override.split(",")
        try:
            band = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise VideoBoundRefused("--band must be two comma-separated fractions y0,y1") from exc
    else:
        configured = app_cfg.get("content_band")
        if configured is None:
            from operation_love.drivers.hinge import HINGE_SPEC

            band = tuple(HINGE_SPEC.content_band)
        else:
            try:
                band = tuple(float(part) for part in configured)
            except (TypeError, ValueError) as exc:
                raise VideoBoundRefused(
                    "apps.hinge.content_band must be two numeric fractions") from exc
    if len(band) != 2 or not 0.0 <= band[0] < band[1] <= 1.0:
        raise VideoBoundRefused(f"content_band must be ordered fractions in [0,1]; got {band!r}")
    return band


def _band_rows(band: tuple[float, float], height: int) -> tuple[int, int]:
    # Same arithmetic as the driver's own crops, from the driver's own helper, so a band edge
    # can never drift by a row between what is measured here and what ships.
    from operation_love.drivers.hinge import _content_rows

    return _content_rows(band, height)


def _band_view(frame: bytes, band: tuple[float, float]):
    image = _decode(frame)
    r0, r1 = _band_rows(band, int(image.shape[0]))
    return image[r0:r1]


def _exact_pairs(bands) -> list[bool]:
    """Byte-exactness of every consecutive pair. Full resolution, zero tolerance (C2)."""
    pairs: list[bool] = []
    for previous, current in zip(bands, bands[1:], strict=False):
        if previous.shape != current.shape:
            raise VideoBoundRefused("frames in one burst do not share a frame geometry")
        pairs.append(previous.tobytes() == current.tobytes())
    return pairs


def _longest_exact_run_s(times: list[float], pairs: list[bool]) -> tuple[float, int]:
    """The longest run of consecutive byte-exact frames, in seconds and in frames.

    Per protocol step 3 this is a PER-CARD statistic: one number per card, never a population of
    frame pairs, so a card with many frames cannot outvote a card with few.
    """
    if not times:
        return 0.0, 0
    best_s, best_frames = 0.0, 1
    start = 0
    for index, exact in enumerate(pairs, start=1):
        if not exact:
            best_s = max(best_s, times[index - 1] - times[start])
            best_frames = max(best_frames, index - start)
            start = index
    best_s = max(best_s, times[len(pairs)] - times[start])
    best_frames = max(best_frames, len(pairs) + 1 - start)
    return float(best_s), int(best_frames)


def _changed_mask(bands):
    """Union of every consecutive-pair difference: which band pixels moved at all."""
    import numpy as np

    changed = np.zeros(bands[0].shape[:2], dtype=bool)
    for previous, current in zip(bands, bands[1:], strict=False):
        difference = previous != current
        if difference.ndim == 3:
            difference = difference.any(axis=2)
        changed |= difference
    return changed


def _changed_row_blocks(changed) -> list[tuple[int, int]]:
    """Row spans of the 8-connected components of everything that moved during the hold.

    A whole-band verdict is dominated by whichever pixel changed anywhere in it: a ticking clock
    or a shimmering chrome element keeps the band "not exact" for the whole minute while a card
    underneath quietly holds one frame.  Grouping the changed pixels into 8-connected components
    and collapsing each to its row span gives a per-region view where that card shows up.
    """
    import cv2
    import numpy as np

    if changed is None or not changed.any():
        return []
    count, labels = cv2.connectedComponents(changed.astype(np.uint8), connectivity=8)
    spans: list[tuple[int, int]] = []
    for component in range(1, count):
        rows = np.flatnonzero((labels == component).any(axis=1))
        if rows.size:
            spans.append((int(rows[0]), int(rows[-1]) + 1))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _unchanged_row_spans(changed) -> list[tuple[int, int]]:
    """Contiguous band rows that never changed across the whole hold.

    This is the other half of the falsifier and the more alarming one: a card region that emits
    no pixel change for the entire window would not appear as a changed block at all, so a
    report built only from changed regions would stay silent about the exact condition section 4
    step 5 is looking for.
    """
    if changed is None:
        return []
    spans: list[tuple[int, int]] = []
    start = None
    for row in range(int(changed.shape[0])):
        still = not bool(changed[row].any())
        if still and start is None:
            start = row
        elif not still and start is not None:
            spans.append((start, row))
            start = None
    if start is not None:
        spans.append((start, int(changed.shape[0])))
    return spans


def _mute_observation(frame: bytes, band: tuple[float, float], size: tuple[int, int]) -> dict:
    """Run the shipped mute matcher and record its score. OBSERVATIONAL METADATA ONLY.

    This is never a label and must never become one: section 2 objection 1 is that a corpus
    labeled by this matcher structurally excludes the false-accept population being bounded.
    It is persisted so that a later reader can see what the matcher would have said about a card
    the OWNER labeled independently -- which is the only direction the inference can run.
    """
    from operation_love.drivers.hinge import HingeDriver

    width, height = size
    rect = (0, round(band[0] * height), round(0.30 * width), round(band[1] * height))
    screened, score = HingeDriver._match_video_mute(frame, rect)
    return {"screened": bool(screened), "score": None if score is None else float(score),
            "observational_only": True}


# =====================================================================================
# Hazard-randomized burst planning (auto-mode owner rule)
# =====================================================================================

@dataclass(frozen=True)
class BurstPlan:
    frames: int
    window_s: float
    gaps_s: tuple[float, ...]


def _hazard_value(low: float, high: float, rnd: random.Random) -> float:
    """Draw one value on [low, high] from a truncated constant-hazard law.

    Auto-mode owner rule: every timing or probability parameter is randomized per draw and never
    a fixed constant, because a fixed cadence is a bot signature in its own right.  A constant
    hazard means the chance of the burst ending in the next instant does not depend on how long
    it has already run, so a sequence of draws carries no learnable period; the rate itself is
    redrawn per call so even the shape of the law varies between bursts.
    """
    rate = 0.5 + 1.5 * rnd.random()
    u = rnd.random()
    shaped = (1.0 - math.exp(-rate * u)) / (1.0 - math.exp(-rate))
    return low + (high - low) * shaped


def plan_burst(rnd: random.Random) -> BurstPlan:
    """n frames over a window w, both hazard-drawn, with hazard-drawn gaps between them."""
    low_n, high_n = _BURST_FRAMES_SPAN
    frames = int(round(_hazard_value(low_n, high_n, rnd)))
    frames = max(low_n, min(high_n, frames))
    window = _hazard_value(*_BURST_WINDOW_S_SPAN, rnd)
    weights = [0.25 + rnd.random() for _ in range(frames - 1)]
    total = sum(weights)
    gaps = tuple(window * weight / total for weight in weights)
    return BurstPlan(frames=frames, window_s=float(window), gaps_s=gaps)


# =====================================================================================
# The capture loop
# =====================================================================================

def settle(capture_fn, *, sleep_fn, rnd: random.Random,
           max_reads: int | None = None) -> tuple[bytes, bool, int]:
    """Read frames until two consecutive whole-frame screencaps come back byte-identical.

    Returns the settled frame, whether it actually settled, and how many reads it took.  The
    bound is not decoration: a card that is PLAYING never settles, and that population is the
    entire point of this campaign, so a settle timeout is recorded as a fact rather than being
    treated as an error.  `measure` sees `settled` per card.
    """
    max_reads = _SETTLE_MAX_READS if max_reads is None else max_reads
    previous = capture_fn()
    for read in range(2, max_reads + 1):
        sleep_fn(_hazard_value(*_SETTLE_GAP_S_SPAN, rnd))
        current = capture_fn()
        if current == previous:
            return current, True, read
        previous = current
    return previous, False, max_reads


def record_burst(capture_fn, plan: BurstPlan, *, sleep_fn, clock) -> list[tuple[bytes, float]]:
    """Capture the planned dwell burst, stamping every frame with a monotonic time.

    The burst is complete and returned before its caller prints a single character of the label
    prompt.  That ordering IS the non-contamination property of protocol step 2: the owner's
    tap-to-play happens strictly after the last dwell frame exists, so no dwell frame can carry
    it, and frames captured after a label always belong to the next card.

    NO CAMPAIGN RECORDS WITH THIS ANY MORE.  Its relative sleeps compress under a slow screencap
    (see `record_spanning_burst`, which both harnesses now use), and it is kept only as the
    documented counter-example that failure is measured against.
    """
    frames: list[tuple[bytes, float]] = []
    for index in range(plan.frames):
        if index:
            elapsed = clock() - frames[-1][1]
            sleep_fn(max(0.0, plan.gaps_s[index - 1] - elapsed))
        started = clock()
        frames.append((capture_fn(), started))
    return frames


# A burst is allowed to finish this far short of its drawn window before it stops counting as
# a burst at all.  It exists for scheduler noise, not for slippage: the 2026-08-21 campaign
# recorded nine frames whose on-screen video countdown advanced ONE second, and a one-second
# look cannot bound the exact-run tail a production dwell spanning the full window will meet.
BURST_SPAN_TOLERANCE_S = 0.5

# The other side of the same window.  A card's frame list may OPEN with one pre-burst anchor
# frame -- the automated harness records its station anchor as the first crop, so the
# anchor/first-burst pair is itself an exactness observation -- and the gap between that
# screencap and the burst's own first frame is a segmentation pass, not a schedule, so a card's
# measured span legitimately runs a little past the window it was drawn for.  Past THIS much it
# is not one clock's measurement of one dwell any more: a frame list stamped on two clocks
# carries a whole station offset, which is tens of seconds, not seconds.
CARD_SPAN_OVERRUN_TOLERANCE_S = 5.0


def record_spanning_burst(capture_fn, plan: BurstPlan, *, sleep_fn, clock,
                          ) -> list[tuple[bytes, float]]:
    """Capture the planned burst on ABSOLUTE deadlines, so it cannot compress.

    `record_burst` sizes each sleep from the previous frame's stamp, which is correct arithmetic
    but accumulates whatever the capture itself costs: every screencap that runs long eats into
    the gap after it, and nothing downstream can tell a burst that spanned its window from one
    that did not.  Here the n target times are laid out once, as offsets from a single origin,
    and each frame waits until its own deadline.  A slow capture therefore shortens the NEXT
    sleep and never the total, and the drawn window is the schedule rather than a hope.

    The gaps are still `plan.gaps_s`: hazard-drawn, non-uniform and re-drawn per burst, under
    the standing owner rule that no timing parameter anywhere in this repository may be a fixed
    constant, because a regular cadence is a bot signature in its own right.

    Every stamp is taken AT SCREENCAP TIME -- immediately before the read is issued, the same
    convention for every frame -- so the recorded span is a measurement of the observation and
    never a file mtime, which is a batch-write artefact and proves nothing.
    """
    targets = [0.0]
    for gap in plan.gaps_s:
        targets.append(targets[-1] + gap)
    origin = clock()
    frames: list[tuple[bytes, float]] = []
    for target in targets:
        remaining = target - (clock() - origin)
        if remaining > 0:
            sleep_fn(remaining)
        stamp = clock()
        frames.append((capture_fn(), stamp))
    return frames


def burst_span_shortfall(frames: list[tuple[bytes, float]], plan: BurstPlan) -> float:
    """How far short of its drawn window a recorded burst fell, in seconds. <= 0 is fine."""
    if len(frames) < 2:
        return float(plan.window_s)
    return float(plan.window_s) - (frames[-1][1] - frames[0][1])


def prompt_label(input_fn, *, print_fn=print) -> str:
    """Read one owner label. Garbage re-prompts; EOF ends the session as an abort."""
    print_fn("\nTap the card and watch it. Does it PLAY (motion, sound, or a scrubber)?")
    while True:
        try:
            answer = input_fn(_LABEL_PROMPT)
        except EOFError:
            return "done"
        answer = (answer or "").strip().lower()
        if answer in _LABELS:
            return answer
        print_fn(f"  Invalid label. Answer exactly one of {'/'.join(_LABELS)}.")


def run_capture(*, out_dir: Path, profiles: int, serial: str, adb_path: str,
                band: tuple[float, float], package: str, config_sha256: str | None,
                capture_fn=None, input_fn=input, print_fn=print, sleep_fn=time.sleep,
                clock=time.monotonic, rnd: random.Random | None = None) -> dict:
    """Run the owner-labeled corpus campaign and write manifest.json."""
    rnd = rnd or random.Random()
    if capture_fn is None:
        def capture_fn():
            return _screencap(serial, adb_path)
    cards_dir = ensure_private_dir(out_dir / "cards")
    version_name = _device_version_name(serial, adb_path, package)
    frame_size: list[int] | None = None
    cards: list[dict] = []
    short_bursts = 0
    ended = "profiles_reached"

    print_fn(
        "\nThis tool NEVER touches the screen. You scroll, you tap, you answer; it only reads "
        "frames and asks.\nFor each card: park it in view hands-off, let the harness record its "
        "dwell burst, and only THEN tap it\nto see whether it plays. The label you type is the "
        "only ground truth in this campaign; the mute\nmatcher's scores are recorded beside it "
        "as observation, never as a label.\n")

    # An interrupted campaign must not lose an hour of owner labeling: whatever was already
    # recorded is written out as an explicitly INCOMPLETE manifest (which `measure` refuses)
    # and the original failure is then re-raised, so the run still fails loudly.
    aborted: BaseException | None = None
    try:
        for ordinal in range(1, profiles + 1):
            try:
                ready = input_fn(_READY_PROMPT)
            except EOFError:
                ended = "eof_at_ready_prompt"
                break
            if (ready or "").strip().lower() == "done":
                ended = "done_at_ready_prompt"
                break
            frame, settled, reads = settle(capture_fn, sleep_fn=sleep_fn, rnd=rnd)
            if not settled:
                print_fn(f"  note: the screen never produced two identical frames in {reads} reads; "
                         "recording the burst anyway and marking settled=false.")
            if frame_size is None:
                frame_size = list(_frame_size(frame))
            plan = plan_burst(rnd)
            print_fn(f"  recording {plan.frames} frames over ~{plan.window_s:.1f}s. Hands off.")
            burst = record_spanning_burst(capture_fn, plan, sleep_fn=sleep_fn, clock=clock)
            shortfall = burst_span_shortfall(burst, plan)
            if shortfall > BURST_SPAN_TOLERANCE_S:
                # The 2026-08-21 failure, refused at the card rather than at the campaign: this
                # harness has the owner standing at the phone, and aborting the sitting would
                # throw away every card they already labeled by hand.  So the card is dropped,
                # counted, and the next one is prompted for -- but it is never recorded, because
                # a burst that did not span its window cannot bound the exact-run tail a
                # production dwell will meet, and a compressed one reaching `measure` licenses a
                # dwell window nothing was ever watched for.
                short_bursts += 1
                print_fn(f"  REFUSED card_{ordinal:04d}: the burst spanned "
                         f"{plan.window_s - shortfall:.2f}s of its drawn {plan.window_s:.2f}s "
                         f"window (short by {shortfall:.2f}s, tolerance "
                         f"{BURST_SPAN_TOLERANCE_S:.2f}s). Nothing recorded; park the next card.")
                continue
            burst_completed_t = clock()

            card_id = f"card_{ordinal:04d}"
            card_dir = ensure_private_dir(cards_dir / card_id)
            origin = burst[0][1]
            frames_meta: list[dict] = []
            mute_meta: list[dict] = []
            for index, (payload, stamp) in enumerate(burst):
                name = f"frame_{index:03d}.png"
                atomic_write_private_bytes(card_dir / name, payload, parent=card_dir)
                frames_meta.append({"path": f"cards/{card_id}/{name}", "sha256": _sha(payload),
                                    "t": round(stamp - origin, 6)})
                mute_meta.append(_mute_observation(payload, band, (frame_size[0], frame_size[1])))

            # The prompt is printed only now, with the burst already closed on disk.
            label_prompted_t = clock()
            label = prompt_label(input_fn, print_fn=print_fn)
            recorded = "skip" if label == "done" else label
            cards.append({
                "card_id": card_id, "label": recorded, "settled": settled, "settle_reads": reads,
                "planned_frames": plan.frames, "planned_window_s": round(plan.window_s, 6),
                "frames": frames_meta,
                "burst_completed_t": round(burst_completed_t - origin, 6),
                "label_prompted_t": round(label_prompted_t - origin, 6),
                "mute_matcher_observations": mute_meta,
            })
            print_fn(f"  recorded {card_id} as {recorded} ({len(frames_meta)} frames).")
            if label == "done":
                ended = "done_at_label_prompt"
                break
    except (KeyboardInterrupt, VideoBoundRefused) as exc:
        ended = f"aborted_{type(exc).__name__}"
        aborted = exc

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION, "kind": _CAMPAIGN_KIND,
        "tool_version": _TOOL_VERSION, "ground_truth_channel": GROUND_TRUTH_CHANNEL,
        "human_ground_truth": True,
        "mute_matcher_is_observational_not_ground_truth": True,
        "completed": ended in ("profiles_reached", "done_at_label_prompt",
                               "done_at_ready_prompt"),
        "ended": ended, "device": serial, "hinge_version_name": version_name,
        "frame_size_px": frame_size, "content_band": [band[0], band[1]],
        "config_sha256": config_sha256,
        # Cards the sitting threw away because their burst did not span its drawn window. A
        # dropped card leaves no other trace, and "how often did this phone fail to hold a
        # schedule" is exactly what a reader of the corpus needs in order to trust the rest.
        "refused_short_bursts": short_bursts,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cards": cards,
    }
    atomic_write_private_text(out_dir / "manifest.json",
                              json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                              parent=out_dir)
    if aborted is not None:
        raise aborted
    return manifest


# =====================================================================================
# hold-test: the cheap falsifier (protocol step 5)
# =====================================================================================

def run_hold_test(*, out_dir: Path, seconds: float, interval: float,
                  band: tuple[float, float], serial: str, adb_path: str,
                  capture_fn=None, print_fn=print, sleep_fn=time.sleep,
                  clock=time.monotonic) -> dict:
    """Screencap a parked, known video hands-off and report its byte-exact runs."""
    if capture_fn is None:
        def capture_fn():
            return _screencap(serial, adb_path)
    if seconds <= 0 or interval <= 0:
        raise VideoBoundRefused("--seconds and --interval must both be positive")
    frames_dir = ensure_private_dir(out_dir / "frames")
    print_fn(f"Park ONE KNOWN VIDEO fully in view and take your hands off the screen.\n"
             f"Reading every {interval:.2f}s for {seconds:.0f}s. This tool sends nothing to the "
             "phone.\n")
    payloads: list[bytes] = []
    times: list[float] = []
    meta: list[dict] = []
    origin = clock()
    index = 0
    while True:
        now = clock()
        if now - origin > seconds:
            break
        payload = capture_fn()
        stamp = clock()
        name = f"frame_{index:04d}.png"
        atomic_write_private_bytes(frames_dir / name, payload, parent=frames_dir)
        payloads.append(payload)
        times.append(stamp - origin)
        meta.append({"path": f"frames/{name}", "sha256": _sha(payload),
                     "t": round(stamp - origin, 6)})
        index += 1
        elapsed = clock() - stamp
        sleep_fn(max(0.0, interval - elapsed))

    if len(payloads) < 2:
        raise VideoBoundRefused("hold test captured fewer than two frames")
    bands = [_band_view(payload, band) for payload in payloads]
    pairs = _exact_pairs(bands)
    whole_run_s, whole_run_frames = _longest_exact_run_s(times, pairs)
    exact_pairs = sum(1 for pair in pairs if pair)

    changed = _changed_mask(bands)
    blocks: list[dict] = []
    for r0, r1 in _changed_row_blocks(changed):
        block_bands = [view[r0:r1] for view in bands]
        block_pairs = _exact_pairs(block_bands)
        run_s, run_frames = _longest_exact_run_s(times, block_pairs)
        blocks.append({"rows": [r0, r1], "longest_exact_run_s": round(run_s, 6),
                       "longest_exact_run_frames": run_frames,
                       "exact_pairs": sum(1 for pair in block_pairs if pair)})
    unchanged_spans = _unchanged_row_spans(changed)
    longest_unchanged = max(unchanged_spans, key=lambda span: span[1] - span[0], default=None)

    report = {
        "schema_version": _MANIFEST_SCHEMA_VERSION, "kind": _HOLD_TEST_KIND,
        "tool_version": _TOOL_VERSION, "device": serial, "content_band": [band[0], band[1]],
        "seconds": seconds, "interval": interval, "frame_count": len(payloads),
        "frames": meta,
        "band_longest_exact_run_s": round(whole_run_s, 6),
        "band_longest_exact_run_frames": whole_run_frames,
        "band_exact_pairs": exact_pairs, "pair_count": len(pairs),
        "changed_row_blocks": blocks,
        "band_rows": int(bands[0].shape[0]),
        "unchanged_row_count": sum(span[1] - span[0] for span in unchanged_spans),
        "longest_unchanged_row_span": (None if longest_unchanged is None
                                       else [longest_unchanged[0], longest_unchanged[1]]),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_private_text(out_dir / "holdtest.json",
                              json.dumps(report, indent=2, sort_keys=True) + "\n",
                              parent=out_dir)
    print_hold_report(report, print_fn=print_fn)
    return report


def print_hold_report(report: dict, *, print_fn=print) -> None:
    print_fn(f"\nframes {report['frame_count']} over {report['seconds']:.0f}s "
             f"({report['pair_count']} consecutive pairs), band "
             f"{report['content_band'][0]}..{report['content_band'][1]}")
    print_fn(f"whole band: {report['band_exact_pairs']} byte-exact pairs, longest exact run "
             f"{report['band_longest_exact_run_s']:.3f}s "
             f"({report['band_longest_exact_run_frames']} frames)")
    blocks = report["changed_row_blocks"]
    print_fn(f"changed-pixel row blocks (8-connected): {len(blocks)}")
    for block in sorted(blocks, key=lambda b: -b["longest_exact_run_s"])[:5]:
        print_fn(f"  rows {block['rows'][0]}..{block['rows'][1]}: longest exact run "
                 f"{block['longest_exact_run_s']:.3f}s "
                 f"({block['longest_exact_run_frames']} frames, "
                 f"{block['exact_pairs']} exact pairs)")
    span = report["longest_unchanged_row_span"]
    print_fn(f"band rows that never changed: {report['unchanged_row_count']} of "
             f"{report['band_rows']}"
             + ("" if span is None else f" (longest contiguous span rows {span[0]}..{span[1]})"))
    if report["band_exact_pairs"]:
        print_fn(
            "\nVERDICT: a known video produced consecutive byte-exact frames over the FULL "
            "content band.\nPer ops/STILL-PHOTO-DISCRIMINATOR.md section 4 step 5 the dwell "
            "window W must grow past this\nrun with margin, or the dwell design is abandoned "
            "honestly. Do not patch around this result.")
    else:
        print_fn("\nVERDICT: no consecutive byte-exact pair over the full content band. The "
                 "dwell falsifier did\nnot fire; proceed to the corpus campaign, which is what "
                 "actually bounds the false-accept rate.")
    if blocks and max(block["exact_pairs"] for block in blocks):
        print_fn("NOTE: at least one changed-pixel row block held byte-exact across consecutive "
                 "frames. Region-\nlocal stillness inside a moving band is exactly what a "
                 "whole-band verdict hides; read the block\ntable above before trusting the "
                 "whole-band line.")


# =====================================================================================
# measure: the offline bound
# =====================================================================================

@dataclass(frozen=True)
class CardStat:
    card_id: str
    label: str
    frames: tuple[dict, ...]
    longest_exact_run_s: float
    longest_exact_run_frames: int
    burst_span_s: float
    all_pairs_exact: bool


@dataclass(frozen=True)
class BoundResult:
    manifest: dict
    stats: tuple[CardStat, ...]
    video_cards: int
    photo_cards: int
    unsure_cards: int
    skipped_cards: int
    video_accepts: int
    photo_false_refusals: int
    photo_false_refusals_at_window: int
    max_video_exact_run_s: float
    observed_window_s: float
    candidate_window_s: float
    dwell_min_window_s: float
    dwell_min_frames: int
    accepts: dict
    # Defaulted so every existing constructor and every caller that predates offline
    # adjudication keeps working unchanged, and so a campaign with no adjudications.json is
    # indistinguishable from one measured before this feature existed.
    written_cards: int = 0
    adjudications: dict | None = None
    original_label_counts: dict | None = None
    # One entry per merged campaign directory, or None for a single-directory measurement --
    # which keeps a single-dir artifact byte-identical to one produced before merging existed.
    campaigns: tuple[dict, ...] | None = None


# =====================================================================================
# Offline adjudication (optional <campaign>/adjudications.json)
# =====================================================================================

def _adjudication_entry(entry, position: int, cards_by_id: dict, claimed: set) -> dict:
    """Validate one entry against the MANIFEST and return it canonicalised."""
    where = f"adjudications.json entry {position}"
    if not isinstance(entry, dict) or set(entry) != ADJUDICATION_ENTRY_KEYS:
        raise VideoBoundRefused(
            f"{where} must carry exactly {sorted(ADJUDICATION_ENTRY_KEYS)}")
    card_id = entry["card_id"]
    if not isinstance(card_id, str) or card_id not in cards_by_id:
        raise VideoBoundRefused(
            f"{where} names card {card_id!r}, which this campaign does not contain; an "
            "adjudication may re-label a captured card and may never add one")
    if card_id in claimed:
        raise VideoBoundRefused(
            f"{where} is a second verdict for card {card_id}; one card gets one adjudication")
    card = cards_by_id[card_id]
    from_label, to_label = entry["from_label"], entry["to_label"]
    current = card.get("label")
    if from_label != current:
        raise VideoBoundRefused(
            f"{where} claims card {card_id} is currently labelled {from_label!r}, but the "
            f"campaign labelled it {current!r}; the adjudicator was shown a different card "
            "than the one this manifest records")
    if (from_label, to_label) not in LEGAL_ADJUDICATIONS:
        raise VideoBoundRefused(
            f"{where} asks for {from_label!r} -> {to_label!r}, which is not one of the legal "
            f"transitions {[f'{a} -> {b}' for a, b in LEGAL_ADJUDICATIONS]}. Adjudication is "
            "deliberately one-way: it may move a card toward caution and never away from it, "
            "because the rule being applied (no mute icon means picture) is the inference "
            "ops/STILL-PHOTO-DISCRIMINATOR.md section 2 objection 2 refuted -- the control "
            "auto-hides, and proven videos score 0.267-0.305 before it renders")
    digests = entry["frame_sha256s"]
    if (not isinstance(digests, list) or not digests
            or not all(isinstance(digest, str) and digest for digest in digests)):
        raise VideoBoundRefused(
            f"{where} must cite at least one frame sha256 the verdict was formed from")
    # Checked against the MANIFEST's recorded digests, not against whatever is on disk: the
    # filesystem is re-verified separately by `_card_stat`, and an adjudication that cited a
    # frame only present on disk would be evidence about bytes this campaign never recorded.
    recorded = {frame.get("sha256") for frame in (card.get("frames") or [])
                if isinstance(frame, dict)}
    unknown = [digest for digest in digests if digest not in recorded]
    if unknown:
        raise VideoBoundRefused(
            f"{where} cites frame sha256 {unknown[0]} which is not among card {card_id}'s "
            "recorded frames")
    rationale = entry["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise VideoBoundRefused(f"{where} must carry a nonempty rationale")
    return {"card_id": card_id, "from_label": from_label, "to_label": to_label,
            "frame_sha256s": list(digests), "rationale": rationale}


def load_adjudications(campaign: Path, manifest: dict) -> dict | None:
    """Read and fully validate <campaign>/adjudications.json, or None when there is none.

    Absence is the normal case and is not an error: with no file, `measure` and `emit` behave
    byte-identically to a tree that never had this feature.
    """
    path = campaign / ADJUDICATION_FILENAME
    if not path.exists():
        return None
    # The owner channel's entire claim is that a HUMAN typed every label after watching the card
    # play.  A model re-labelling some of them would make `human_ground_truth: true` false while
    # every other number in the artifact still agreed, which is exactly the undetectable lie the
    # channel split exists to prevent.  Adjudication belongs to the circular channel, which has
    # already declared that its labels come from a machine.
    if manifest.get("ground_truth_channel") == GROUND_TRUTH_CHANNEL:
        raise VideoBoundRefused(
            f"{ADJUDICATION_FILENAME} cannot be applied to a {GROUND_TRUTH_CHANNEL!r} campaign: "
            "its labels are the owner's own tap-to-play verdicts, and re-labelling any of them "
            "would leave the artifact claiming human ground truth it no longer has")
    document, _raw = _read_json(path, ADJUDICATION_FILENAME)
    if set(document) != ADJUDICATION_DOCUMENT_KEYS:
        raise VideoBoundRefused(
            f"{ADJUDICATION_FILENAME} must carry exactly {sorted(ADJUDICATION_DOCUMENT_KEYS)}")
    adjudicator = document["adjudicator"]
    if (not isinstance(adjudicator, dict)
            or set(adjudicator) != ADJUDICATION_ADJUDICATOR_KEYS
            or not all(isinstance(adjudicator[key], str) and adjudicator[key].strip()
                       for key in ADJUDICATION_ADJUDICATOR_KEYS)):
        raise VideoBoundRefused(
            f"{ADJUDICATION_FILENAME} must name its adjudicator with nonempty "
            f"{sorted(ADJUDICATION_ADJUDICATOR_KEYS)}; anonymous re-labelling is not evidence")
    entries = document["entries"]
    if not isinstance(entries, list) or not entries:
        raise VideoBoundRefused(
            f"{ADJUDICATION_FILENAME} carries no entries; delete the file rather than binding "
            "an empty verdict into the artifact digest")
    cards_by_id = {card.get("card_id"): card for card in manifest.get("cards") or []
                   if isinstance(card, dict)}
    claimed: set = set()
    applied = []
    for position, entry in enumerate(entries):
        record = _adjudication_entry(entry, position, cards_by_id, claimed)
        claimed.add(record["card_id"])
        applied.append(record)
    return {"adjudicator": {key: adjudicator[key] for key in
                            sorted(ADJUDICATION_ADJUDICATOR_KEYS)},
            "entries": applied}


def _relabelled(manifest: dict, adjudications: dict) -> dict:
    """A copy of the manifest with the adjudicated cards' labels rewritten. Never mutates."""
    verdicts = {entry["card_id"]: entry["to_label"] for entry in adjudications["entries"]}
    cards = []
    for card in manifest["cards"]:
        if isinstance(card, dict) and card.get("card_id") in verdicts:
            card = dict(card)
            card["label_before_adjudication"] = card["label"]
            card["label"] = verdicts[card["card_id"]]
        cards.append(card)
    updated = dict(manifest)
    updated["cards"] = cards
    return updated


# Multi-campaign merge (2026-08-21).  At the observed ~1-video-in-5-profiles rate, 60 owner- or
# AI-labelled video cards is ~300 profiles, which is several sittings.  One campaign directory per
# sitting is right -- a sitting is the unit that can be interrupted, halted and re-read -- so the
# BOUND has to be able to span them.  What it must never do is span two things that measured
# different worlds, so every field the artifact BINDS has to be identical across the set before a
# single card is merged.  A disagreement is refused by name rather than reconciled: a bound whose
# device serial or Hinge build is "one of these two" licenses nothing.
_MERGE_AGREEMENT_FIELDS = (
    "ground_truth_channel",
    "accepted_circular_risk",
    "human_ground_truth",
    "device",
    "hinge_version_name",
    "frame_size_px",
    "config_sha256",
    # The band the per-card byte-exactness is measured over. Two campaigns that cropped
    # differently would be contributing incomparable statistics to one denominator.
    "content_band",
    "autoplay_center_band_frac",
    # The hazard spans the bursts were drawn from. The bound's whole claim is about a dwell of a
    # particular size; cards watched under a different span are not the same experiment.
    "dwell_parameter_space",
)


# A campaign that STOPPED SAFELY is still evidence.  The auto harness halts on the first
# unrecognized screen, the first empty profile, or a signal -- all deliberate, all leaving a
# fully written prefix behind -- and refusing to measure those threw away a whole sitting for
# doing exactly what it was told to do.  At ~1 video in 5 profiles a sitting is expensive.
#
# What is NOT relaxed: every card is still verified end to end against its own recorded digests.
# The `ended` flag is not trusted for anything except deciding whether a TRAILING partially
# written card may be dropped; a card that fails verification anywhere else is corruption and
# still refuses.  So "measurable" means "every card kept was proven", not "the manifest says so".
_MEASURABLE_HALT_ENDED = ("halted", "halted_empty_profile", "signalled", "interrupted")


@dataclass(frozen=True)
class CampaignSource:
    """One completed campaign directory, loaded and adjudicated, ready to be merged."""

    root: Path
    name: str
    manifest: dict
    adjudications: dict | None
    original_label_counts: dict
    manifest_sha256: str
    stats: tuple[CardStat, ...]
    ended: str
    halt_reason: str | None
    dropped_trailing_cards: int


def _verified_stats(campaign: Path, cards: list, band: tuple[float, float], *,
                    allow_trailing_drop: bool) -> tuple[list[CardStat], int]:
    """Verify every card end to end; return the proven ones and how many trailing were dropped.

    A campaign that halted can have been interrupted mid-write on its LAST card, which is a
    truncation and not a lie: the cards before it are complete and digest-verified.  So a
    contiguous run of failures at the very end may be dropped.  A failure anywhere else is
    corruption -- somebody edited or lost a frame in the middle of a finished record -- and
    refuses with the original reason, exactly as a completed campaign would.
    """
    outcomes: list = []
    for card in cards:
        try:
            outcomes.append(_card_stat(campaign, card, band))
        except VideoBoundRefused as exc:
            outcomes.append(exc)
    failed = [index for index, outcome in enumerate(outcomes)
              if isinstance(outcome, VideoBoundRefused)]
    if not failed:
        return list(outcomes), 0
    first = failed[0]
    if not allow_trailing_drop:
        raise outcomes[first]
    if set(failed) != set(range(first, len(outcomes))):
        intact_after = next(index for index in range(first, len(outcomes))
                            if index not in set(failed))
        raise VideoBoundRefused(
            f"card {cards[first].get('card_id')!r} failed verification but card "
            f"{cards[intact_after].get('card_id')!r} after it is intact, so this is not an "
            "interrupted write at the end of a halted campaign; it is a corrupted record "
            f"({outcomes[first]})") from outcomes[first]
    return list(outcomes[:first]), len(failed)


def _load_campaign(campaign: Path) -> CampaignSource:
    """Load, verify and adjudicate ONE directory. Every kept card is proven from its own bytes."""
    manifest = _load_manifest(campaign)
    raw = (campaign / "manifest.json").read_bytes()
    ended = manifest.get("ended")
    halted = manifest.get("completed") is not True
    band = (float(manifest["content_band"][0]), float(manifest["content_band"][1]))
    stats, dropped = _verified_stats(campaign, manifest["cards"], band,
                                     allow_trailing_drop=halted)
    if dropped:
        # The dropped cards leave the manifest view too, so an adjudication cannot name one and
        # the tallies cannot count one.
        manifest = dict(manifest)
        manifest["cards"] = manifest["cards"][:len(stats)]
    if not manifest["cards"]:
        raise VideoBoundRefused(
            f"{campaign.name} has no card that survives verification; a halted campaign is "
            "evidence only for the cards it finished writing")
    # Applied BEFORE anything is grouped by label, so the denominators, the accept verdicts and
    # the artifact all describe one consistent set of labels.  Scoped to THIS directory: an
    # adjudications.json may only re-label cards of the campaign it sits beside, which is why it
    # is resolved here rather than against the merged set.
    adjudications = load_adjudications(campaign, manifest)
    original = {label: sum(1 for card in manifest["cards"]
                           if isinstance(card, dict) and card.get("label") == label)
                for label in CARD_LABELS}
    if adjudications is not None:
        manifest = _relabelled(manifest, adjudications)
        verdicts = {entry["card_id"]: entry["to_label"] for entry in adjudications["entries"]}
        stats = [replace(stat, label=verdicts.get(stat.card_id, stat.label)) for stat in stats]
    return CampaignSource(root=campaign, name=campaign.name, manifest=manifest,
                          adjudications=adjudications, original_label_counts=original,
                          manifest_sha256=_sha(raw), stats=tuple(stats),
                          ended=ended if isinstance(ended, str) else "unknown",
                          halt_reason=manifest.get("halt_reason"),
                          dropped_trailing_cards=dropped)


def _require_agreement(sources: list[CampaignSource]) -> None:
    """Refuse unless every directory measured the same world, naming the field that differs."""
    seen_names: dict[str, Path] = {}
    for source in sources:
        previous = seen_names.get(source.name)
        if previous is not None:
            raise VideoBoundRefused(
                f"two campaign directories share the basename {source.name!r} ({previous} and "
                f"{source.root}); card ids are namespaced by basename, so they would collide")
        seen_names[source.name] = source.root
    first = sources[0]
    for field in _MERGE_AGREEMENT_FIELDS:
        reference = first.manifest.get(field)
        for source in sources[1:]:
            value = source.manifest.get(field)
            if value != reference:
                raise VideoBoundRefused(
                    f"campaigns disagree on {field}: {first.name} records {reference!r} but "
                    f"{source.name} records {value!r}. Every field the artifact binds must be "
                    "identical across a merged set, because a bound that cannot name one device, "
                    "one build and one dwell is not evidence about any of them")


def _merged_adjudications(sources: list[CampaignSource], *, namespaced: bool) -> dict | None:
    """One adjudication block for the merged set, with card ids namespaced when merging.

    Each directory keeps its OWN adjudications.json and each entry was already validated against
    its own manifest, so an entry in one campaign can never reach a card in another even when the
    bare card ids collide -- they always do, since every campaign numbers from card_0001.
    """
    carrying = [source for source in sources if source.adjudications is not None]
    if not carrying:
        return None
    adjudicator = carrying[0].adjudications["adjudicator"]
    for source in carrying[1:]:
        if source.adjudications["adjudicator"] != adjudicator:
            raise VideoBoundRefused(
                f"campaigns disagree on the adjudicator: {carrying[0].name} names "
                f"{adjudicator!r} but {source.name} names "
                f"{source.adjudications['adjudicator']!r}; one merged bound records one reviewer")
    entries = []
    for source in carrying:
        for entry in source.adjudications["entries"]:
            record = dict(entry)
            if namespaced:
                record["card_id"] = f"{source.name}/{entry['card_id']}"
            entries.append(record)
    return {"adjudicator": adjudicator, "entries": entries}


def _load_manifest(campaign: Path) -> dict:
    manifest, _raw = _read_json(campaign / "manifest.json", "campaign manifest")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION or manifest.get("kind") != _CAMPAIGN_KIND:
        raise VideoBoundRefused("campaign manifest has an unsupported schema/kind")
    if manifest.get("completed") is not True and manifest.get("ended") not in _MEASURABLE_HALT_ENDED:
        raise VideoBoundRefused(
            "campaign manifest is not completed and did not stop at a recognized safety halt; "
            "an aborted session is not evidence")
    channel = manifest.get("ground_truth_channel")
    if channel == CIRCULAR_GROUND_TRUTH_CHANNEL:
        # The circular channel is readable only because the owner opted into it by name, and the
        # phrase is demanded from the CAMPAIGN rather than from a flag: a flag would let a later
        # re-run of `measure` opt them in retroactively over evidence captured under a different
        # understanding.  Claiming human ground truth here would be a straight lie, so it is
        # refused rather than coerced.
        if manifest.get("accepted_circular_risk") != CIRCULAR_ACCEPTANCE:
            raise VideoBoundRefused(
                f"a {CIRCULAR_GROUND_TRUTH_CHANNEL!r} campaign labels its cards with the same "
                "signals the accept rule reads, so it is measurable only when its manifest "
                f"carries the acceptance phrase {CIRCULAR_ACCEPTANCE!r} verbatim")
        if manifest.get("human_ground_truth") is not False:
            raise VideoBoundRefused(
                f"a {CIRCULAR_GROUND_TRUTH_CHANNEL!r} campaign has no human in its label loop "
                "and must record human_ground_truth false")
    elif channel != GROUND_TRUTH_CHANNEL:
        raise VideoBoundRefused(
            f"campaign ground_truth_channel must be {GROUND_TRUTH_CHANNEL!r}; the whole design "
            "rests on labels the mute matcher cannot produce")
    elif manifest.get("human_ground_truth") is not True:
        raise VideoBoundRefused("campaign manifest does not claim human ground truth")
    band = manifest.get("content_band")
    if (not isinstance(band, list) or len(band) != 2
            or not all(isinstance(value, (int, float)) for value in band)
            or not 0.0 <= band[0] < band[1] <= 1.0):
        raise VideoBoundRefused("campaign manifest has no usable content_band")
    if not isinstance(manifest.get("cards"), list) or not manifest["cards"]:
        raise VideoBoundRefused("campaign manifest records no cards")
    return manifest


def _card_stat(campaign: Path, card: dict, band: tuple[float, float]) -> CardStat:
    card_id = card.get("card_id")
    label = card.get("label")
    if not isinstance(card_id, str) or label not in CARD_LABELS:
        raise VideoBoundRefused(f"card {card_id!r} has no usable id/owner label")
    frames = card.get("frames")
    if not isinstance(frames, list) or not frames:
        raise VideoBoundRefused(f"card {card_id} records no frames")
    payloads: list[bytes] = []
    times: list[float] = []
    for record in frames:
        if (not isinstance(record, dict) or set(record) != {"path", "sha256", "t"}
                or not isinstance(record["path"], str)
                or not isinstance(record["sha256"], str)
                or not isinstance(record["t"], (int, float))):
            raise VideoBoundRefused(f"card {card_id} has a malformed frame record")
        path = campaign / record["path"]
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise VideoBoundRefused(
                f"card {card_id} frame {record['path']} is missing: {type(exc).__name__}") from exc
        if _sha(payload) != record["sha256"]:
            raise VideoBoundRefused(
                f"card {card_id} frame {record['path']} does not match its manifest sha256; the "
                "campaign on disk is not the campaign that was captured")
        payloads.append(payload)
        times.append(float(record["t"]))
    if times != sorted(times):
        raise VideoBoundRefused(f"card {card_id} frame timestamps are not monotonic")
    prompted = card.get("label_prompted_t")
    if label in _DENOMINATOR_LABELS:
        if not isinstance(prompted, (int, float)) or prompted < times[-1]:
            raise VideoBoundRefused(
                f"card {card_id} was labeled before its dwell burst closed; the burst may carry "
                "the owner's tap and cannot be used as dwell evidence")
    # The one place BOTH harnesses' spans have to survive, and the reason it lives here rather
    # than at either recorder: `burst_span_s` is what `observed_window_s`, `accepted()` and the
    # margin guard are all computed from, so a span that is not a measurement of the window the
    # card was actually scheduled for must never reach them -- and no future call site can pick
    # a recorder that bypasses this.  Short means the burst compressed (2026-08-21: nine frames
    # across ONE second of a playing video's own countdown); long means two clocks were spliced
    # into one frame list, which inflates every deep card's dwell and makes the accept rule that
    # ships not the rule that was validated.
    planned = card.get("planned_window_s")
    if isinstance(planned, bool) or not isinstance(planned, (int, float)) or planned <= 0:
        raise VideoBoundRefused(
            f"card {card_id} records no usable planned_window_s, so nothing says what window its "
            "frames were supposed to span")
    span = float(times[-1] - times[0])
    if span < float(planned) - BURST_SPAN_TOLERANCE_S:
        raise VideoBoundRefused(
            f"card {card_id} spanned {span:.3f}s of the {float(planned):.3f}s window it was "
            f"drawn for (tolerance {BURST_SPAN_TOLERANCE_S:.2f}s); a burst that did not span its "
            "window cannot bound the exact-run tail a production dwell will meet")
    if span > float(planned) + CARD_SPAN_OVERRUN_TOLERANCE_S:
        raise VideoBoundRefused(
            f"card {card_id} spanned {span:.3f}s, well past the {float(planned):.3f}s window it "
            f"was drawn for (tolerance {CARD_SPAN_OVERRUN_TOLERANCE_S:.2f}s); its frame "
            "timestamps are not one clock's measurement of one dwell")
    if len(payloads) < 2:
        return CardStat(card_id=card_id, label=label, frames=tuple(frames),
                        longest_exact_run_s=0.0, longest_exact_run_frames=1,
                        burst_span_s=0.0, all_pairs_exact=False)
    bands = [_band_view(payload, band) for payload in payloads]
    pairs = _exact_pairs(bands)
    run_s, run_frames = _longest_exact_run_s(times, pairs)
    return CardStat(card_id=card_id, label=label, frames=tuple(frames),
                    longest_exact_run_s=run_s, longest_exact_run_frames=run_frames,
                    burst_span_s=float(times[-1] - times[0]), all_pairs_exact=all(pairs))


def measure(campaigns, *, config_path: str | None = None) -> BoundResult:
    """Compute the held-out bound offline over ONE OR MORE campaigns, or refuse. Writes nothing.

    `campaigns` is a directory or a sequence of them.  Merging exists because 60 video cards is
    roughly 300 profiles at the observed rate, which is several sittings, and a sitting is the
    right unit for a campaign directory: it is what can be interrupted, halted and re-read.  What
    a merge must never do is blur two different worlds together, so `_require_agreement` refuses
    unless every field the artifact binds is identical across the set.

    A SINGLE directory is byte-identical to what this function did before merging existed: no
    namespacing, no `campaigns` block, the same digest.  Namespacing card ids as
    `<dirname>/card_0001` only starts once there is more than one directory to tell apart -- and
    there always is something to tell apart, because every campaign numbers from card_0001.
    """
    sources = [campaigns] if isinstance(campaigns, (str, Path)) else list(campaigns)
    if not sources:
        raise VideoBoundRefused("no campaign directory was given")
    sources = [_load_campaign(Path(root)) for root in sources]
    _require_agreement(sources)
    namespaced = len(sources) > 1
    representative = sources[0].manifest
    adjudications = _merged_adjudications(sources, namespaced=namespaced)
    original_label_counts = {
        label: sum(source.original_label_counts.get(label, 0) for source in sources)
        for label in CARD_LABELS}
    if config_path is not None:
        _cfg_map, cfg_raw = _load_config_mapping(config_path)
        # Every campaign agreed on config_sha256 above, so checking the representative checks
        # all of them.
        recorded = representative.get("config_sha256")
        if isinstance(recorded, str) and cfg_raw is not None and recorded != _sha(cfg_raw):
            raise VideoBoundRefused(
                "campaign was captured against different config bytes than the config supplied "
                "here; a bound cannot be carried across a config change")
    stats: list[CardStat] = []
    campaign_records: list[dict] = []
    for source in sources:
        # Already verified against THIS directory's own recorded digests by `_load_campaign`.
        local = list(source.stats)
        if namespaced:
            local = [replace(stat, card_id=f"{source.name}/{stat.card_id}") for stat in local]
        stats.extend(local)
        campaign_records.append({
            "dir": source.name,
            "manifest_sha256": source.manifest_sha256,
            "captured_at": source.manifest.get("captured_at"),
            # Provenance, not decoration: a halted sitting is measurable, and a reader of the
            # artifact has to be able to see which of its cards came from one and why it stopped.
            "ended": source.ended,
            "halt_reason": source.halt_reason,
            "dropped_trailing_cards": source.dropped_trailing_cards,
            "cards": len(local),
            **{f"{label}_cards": sum(1 for stat in local if stat.label == label)
               for label in CARD_LABELS},
        })
    stats = tuple(stats)
    by_label = {label: [stat for stat in stats if stat.label == label] for label in CARD_LABELS}
    videos, photos = by_label["video"], by_label["photo"]
    if not videos:
        raise VideoBoundRefused("campaign contains no owner-labeled video cards")
    if not photos:
        raise VideoBoundRefused("campaign contains no owner-labeled still-photo cards")

    max_video_exact_run_s = max(stat.longest_exact_run_s for stat in videos)
    # Every labeled card was watched for at least this long, so this is the longest dwell window
    # the campaign can license without extrapolating past its own observations.
    observed_window_s = min(stat.burst_span_s for stat in videos + photos)
    candidate_window_s = DWELL_WINDOW_SAFETY_FACTOR * max_video_exact_run_s
    # The window that is actually evaluated is capped by observation, deliberately.  Deriving it
    # only from SAFETY_FACTOR x the worst observed run would make "zero video accepts" true by
    # construction and the Rule-of-Three claim vacuous.  Capping means a video that held one
    # frame for the whole burst DOES accept here and the campaign refuses, which is the outcome
    # section 4 wants; the separate margin check below then refuses whenever the safety factor
    # cannot be achieved inside what was observed.  A zero worst-case run is the best possible
    # result, not a licence to ship a zero-length dwell, so it falls back to the same cap.
    dwell_min_window_s = (min(candidate_window_s, observed_window_s)
                          if candidate_window_s > 0 else observed_window_s)

    def accepted(stat: CardStat) -> bool:
        return (stat.burst_span_s >= dwell_min_window_s
                and stat.longest_exact_run_s >= dwell_min_window_s)

    accepts = {stat.card_id: accepted(stat) for stat in stats}
    video_accepts = sum(1 for stat in videos if accepts[stat.card_id])
    photo_false_refusals = sum(1 for stat in photos if not stat.all_pairs_exact)
    photo_false_refusals_at_window = sum(1 for stat in photos if not accepts[stat.card_id])
    dwell_min_frames = min(len(stat.frames) for stat in videos + photos)

    result = BoundResult(
        manifest=representative, stats=stats, video_cards=len(videos), photo_cards=len(photos),
        unsure_cards=len(by_label["unsure"]), skipped_cards=len(by_label["skip"]),
        written_cards=len(by_label["written"]),
        video_accepts=video_accepts, photo_false_refusals=photo_false_refusals,
        photo_false_refusals_at_window=photo_false_refusals_at_window,
        max_video_exact_run_s=max_video_exact_run_s, observed_window_s=observed_window_s,
        candidate_window_s=candidate_window_s, dwell_min_window_s=dwell_min_window_s,
        dwell_min_frames=dwell_min_frames, accepts=accepts, adjudications=adjudications,
        original_label_counts=original_label_counts,
        # Recorded whenever there is more than one sitting to tell apart OR any sitting stopped
        # at a safety halt.  A single COMPLETED campaign keeps no block at all, which is what
        # makes its artifact byte-identical to one produced before any of this existed.
        campaigns=(tuple(campaign_records)
                   if (namespaced or any(record["ended"] in _MEASURABLE_HALT_ENDED
                                         for record in campaign_records))
                   else None))
    _enforce_thresholds(result)
    return result


def _enforce_thresholds(result: BoundResult) -> None:
    if result.video_cards < MIN_VIDEO_CARDS:
        raise VideoBoundRefused(
            f"only {result.video_cards} owner-labeled video cards; section 4 requires "
            f">= {MIN_VIDEO_CARDS} for the Rule-of-Three 95% upper bound this artifact claims")
    if result.photo_cards < MIN_PHOTO_CARDS:
        raise VideoBoundRefused(
            f"only {result.photo_cards} owner-labeled still-photo cards; section 4 requires "
            f">= {MIN_PHOTO_CARDS}")
    if result.video_accepts != REQUIRED_VIDEO_ACCEPTS:
        raise VideoBoundRefused(
            f"{result.video_accepts} owner-labeled video card(s) would have been ACCEPTED at a "
            f"{result.dwell_min_window_s:.3f}s dwell window; section 4 requires exactly "
            f"{REQUIRED_VIDEO_ACCEPTS}. Do not widen the window to recover this: a video accept "
            "is the failure mode the whole discriminator exists to exclude.")
    if result.candidate_window_s > result.observed_window_s:
        raise VideoBoundRefused(
            f"the dwell window this campaign licenses ({result.candidate_window_s:.3f}s = "
            f"{DWELL_WINDOW_SAFETY_FACTOR} x the worst-case video byte-exact run "
            f"{result.max_video_exact_run_s:.3f}s) exceeds the {result.observed_window_s:.3f}s "
            "every card was actually watched for. The safety factor cannot be achieved inside "
            "what was observed; capture longer bursts rather than shrinking the margin.")
    ceiling = MAX_PHOTO_FALSE_REFUSAL_FRAC * result.photo_cards
    if result.photo_false_refusals > ceiling:
        raise VideoBoundRefused(
            f"{result.photo_false_refusals} of {result.photo_cards} owner-labeled still photos "
            f"would be falsely refused, above the {MAX_PHOTO_FALSE_REFUSAL_FRAC:.0%} ceiling "
            f"({ceiling:.2f} cards)")


def _adjudicated_suffix(result: BoundResult, label: str) -> str:
    """" (+N adjudicated)" when N cards were re-labelled INTO this label, else ""."""
    if not result.adjudications:
        return ""
    gained = sum(1 for entry in result.adjudications["entries"]
                 if entry["to_label"] == label)
    return f" (+{gained} adjudicated)" if gained else ""


def print_bound_report(result: BoundResult, *, print_fn=print) -> None:
    channel = result.manifest.get("ground_truth_channel")
    circular = channel == CIRCULAR_GROUND_TRUTH_CHANNEL
    if circular:
        # Printed FIRST, before a single number, because the numbers below are the part a reader
        # will quote and the caveat is the part that says which of them mean anything.
        print_fn("\n" + "=" * 88)
        print_fn("STRUCTURAL VACUITY CAVEAT")
        print_fn(LABEL_BLIND_SPOT)
        print_fn("=" * 88)
    print_fn(f"\ncampaign device {result.manifest.get('device')} build "
             f"{result.manifest.get('hinge_version_name')} "
             f"frame {result.manifest.get('frame_size_px')}")
    print_fn(f"ground truth: {channel} "
             + ("(AI-labeled from the same pixels the accept rule reads; the video-accept line "
                "below is vacuous)" if circular
                else "(owner-typed; the mute matcher's scores are observation only)"))
    if result.campaigns:
        print_fn(f"campaign directories in this bound: {len(result.campaigns)}")
        for record in result.campaigns:
            note = "" if record["ended"] not in _MEASURABLE_HALT_ENDED else (
                f"  [HALTED: {record['ended']}"
                + (f" -- {record['halt_reason']}" if record.get("halt_reason") else "")
                + (f"; dropped {record['dropped_trailing_cards']} trailing partial card(s)"
                   if record["dropped_trailing_cards"] else "") + "]")
            print_fn(f"  {record['dir']}: {record['cards']} cards "
                     f"({record['video_cards']} video, {record['photo_cards']} photo, "
                     f"{record['written_cards']} written, {record['unsure_cards']} unsure)"
                     + note)
    print_fn(f"cards: {result.video_cards} video{_adjudicated_suffix(result, 'video')}, "
             f"{result.photo_cards} photo{_adjudicated_suffix(result, 'photo')} "
             f"(excluded from both denominators: {result.unsure_cards} unsure, "
             f"{result.written_cards} written, {result.skipped_cards} skipped)")
    if result.adjudications:
        adjudicator = result.adjudications["adjudicator"]
        moves = {}
        for entry in result.adjudications["entries"]:
            key = f"{entry['from_label']} -> {entry['to_label']}"
            moves[key] = moves.get(key, 0) + 1
        original = result.original_label_counts or {}
        print_fn(f"adjudicated OFFLINE by {adjudicator['model']} via "
                 f"{adjudicator['process']}: "
                 + ", ".join(f"{count} x {move}" for move, count in sorted(moves.items())))
        print_fn("  the capture's own labels were: "
                 + ", ".join(f"{original.get(label, 0)} {label}"
                             for label in ("video", "photo", "unsure")))
    print_fn(f"worst-case video byte-exact run: {result.max_video_exact_run_s:.3f}s")
    if result.candidate_window_s > 0:
        print_fn(f"candidate dwell window: {result.candidate_window_s:.3f}s "
                 f"({DWELL_WINDOW_SAFETY_FACTOR} x that worst case)")
    else:
        print_fn("candidate dwell window: no video ever held two byte-identical frames, so the "
                 "safety factor has\n  nothing to multiply and the window is whatever the "
                 "campaign actually watched")
    print_fn(f"evaluated at: {result.dwell_min_window_s:.3f}s, capped by the "
             f"{result.observed_window_s:.3f}s every card was actually watched for")
    print_fn(f"video accepts at that window: {result.video_accepts} "
             f"(required {REQUIRED_VIDEO_ACCEPTS})")
    print_fn(f"photo false refusals: {result.photo_false_refusals} of {result.photo_cards} "
             f"(ceiling {MAX_PHOTO_FALSE_REFUSAL_FRAC:.0%}; "
             f"{result.photo_false_refusals_at_window} refused at the candidate window)")
    print_fn(f"dwell to ship: min_frames {result.dwell_min_frames}, "
             f"min_window_s {result.dwell_min_window_s:.3f}")
    if circular:
        print_fn("\nREMINDER: the video-accept count above is zero by construction on this "
                 "channel. What this\ncampaign actually delivers is the worst-case byte-exact "
                 "run measured on playing videos, the\nstill-photo false-refusal rate, and the "
                 "persisted corpus a later owner-labeled run can reuse.")


# =====================================================================================
# emit: freeze the artifact and print the config paste block
# =====================================================================================

def build_artifact(result: BoundResult, *, config_sha256: str,
                   hinge_version_name: str) -> dict:
    cards = []
    for stat in result.stats:
        card = {"card_id": stat.card_id, "label": stat.label,
                "frames": [dict(frame) for frame in stat.frames],
                "longest_exact_run_s": round(stat.longest_exact_run_s, 6),
                "accept": bool(result.accepts[stat.card_id])}
        if set(card) != BOUND_CARD_KEYS:
            raise AssertionError("bound card schema drift")
        cards.append(card)
    # The channel is read off the CAMPAIGN rather than pinned to a constant.  `_load_manifest`
    # has already proven it is one of the two accepted channels and that a circular one carries
    # the acceptance phrase, and an artifact that renamed its own label channel would be the one
    # claim config validation could not catch: every other number it mirrors would still agree.
    channel = result.manifest.get("ground_truth_channel")
    circular = channel == CIRCULAR_GROUND_TRUTH_CHANNEL
    artifact = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "ground_truth_channel": channel,
        "human_ground_truth": not circular,
        "device": result.manifest.get("device"),
        "hinge_version_name": hinge_version_name,
        "frame_size_px": result.manifest.get("frame_size_px"),
        "captured_at": _artifact_captured_at(result),
        "config_sha256": config_sha256,
        "video_cards": result.video_cards,
        "video_accepts": result.video_accepts,
        "photo_cards": result.photo_cards,
        "photo_false_refusals": result.photo_false_refusals,
        "max_video_exact_run_s": round(result.max_video_exact_run_s, 6),
        "dwell": {"min_frames": result.dwell_min_frames,
                  "min_window_s": round(result.dwell_min_window_s, 6)},
        "cards": cards,
        "evidence_sha256": "",
    }
    if circular:
        # Both are inside the digest, so a hand edit that strips the caveat or the acceptance
        # invalidates the artifact rather than quietly producing a cleaner-looking one.
        artifact["accepted_circular_risk"] = CIRCULAR_ACCEPTANCE
        artifact["label_blind_spot"] = LABEL_BLIND_SPOT
    if result.adjudications:
        # Inside the digest for the same reason: the counts above are POST-adjudication, so the
        # verdicts that produced them have to be part of what the digest attests. Editing an
        # entry, adding one, or deleting the block all invalidate the artifact.
        artifact["adjudications"] = result.adjudications
    center_band = result.manifest.get("autoplay_center_band_frac")
    if center_band is not None:
        artifact["autoplay_center_band_frac"] = float(center_band)
    if result.campaigns:
        # Inside the digest: the counts above are the UNION of these directories, so which
        # directories they came from is part of what the digest attests. Each entry carries its
        # own manifest sha256, so a merged artifact names exactly the sittings it merged and a
        # later reader can re-verify every one of them.
        artifact["campaigns"] = [dict(record) for record in result.campaigns]
    if set(artifact) != _artifact_keys(circular, bool(result.adjudications),
                                       center_band is not None, bool(result.campaigns)):
        raise AssertionError("bound artifact schema drift")
    artifact["evidence_sha256"] = _sha(_canonical(artifact))
    return artifact


_OPTIONAL_ARTIFACT_KEY_SETS = (CIRCULAR_ARTIFACT_KEYS, ADJUDICATION_ARTIFACT_KEYS,
                               AUTOPLAY_ARTIFACT_KEYS, MERGE_ARTIFACT_KEYS)


def _artifact_captured_at(result: BoundResult):
    """The artifact's single capture timestamp.

    For one directory it is that directory's own, unchanged.  For a merged set it is the
    EARLIEST, because that is when the evidence this artifact rests on started being collected;
    each directory's own timestamp is kept in `campaigns[]` so nothing is lost.
    """
    if not result.campaigns:
        return result.manifest.get("captured_at")
    stamps = [record.get("captured_at") for record in result.campaigns
              if isinstance(record.get("captured_at"), str)]
    return min(stamps) if stamps else result.manifest.get("captured_at")


def _artifact_keys(circular: bool, adjudicated: bool = False, autoplay: bool = False,
                   merged: bool = False) -> set:
    keys = set(BOUND_ARTIFACT_KEYS)
    for present, optional in zip((circular, adjudicated, autoplay, merged),
                                 _OPTIONAL_ARTIFACT_KEY_SETS, strict=True):
        if present:
            keys |= optional
    return keys


def verify_artifact_digest(artifact: dict) -> bool:
    """Recompute the self-digest exactly as build_artifact wrote it."""
    if set(artifact) not in [_artifact_keys(*flags) for flags in
                            itertools.product((False, True), repeat=4)]:
        return False
    subject = dict(artifact)
    claimed = subject["evidence_sha256"]
    subject["evidence_sha256"] = ""
    return isinstance(claimed, str) and _sha(_canonical(subject)) == claimed


def emit(campaigns, *, config_path: str, serial_override: str | None = None) -> tuple[dict, dict]:
    """Re-run the measurement over the same directory list and freeze bound.json + the paste block.

    The artifact is written into the FIRST directory of the list, which is deterministic and
    self-describing: a merged artifact carries `campaigns[]` naming every directory it merged, so
    where the file happens to live never has to be inferred.
    """
    cfg_map, cfg_raw = _load_config_mapping(config_path)
    if cfg_raw is None:
        raise VideoBoundRefused("emit requires --config")
    roots = [campaigns] if isinstance(campaigns, (str, Path)) else list(campaigns)
    if not roots:
        raise VideoBoundRefused("no campaign directory was given")
    campaign = Path(roots[0])
    result = measure(roots, config_path=config_path)
    app_cfg = _hinge_app_config(cfg_map)
    package = app_cfg.get("package", _DEFAULT_PACKAGE)
    version_name = None
    try:
        serial, adb_path = _resolve_serial(app_cfg, serial_override)
        # Only the phone the campaign was actually measured on may name its build.  A different
        # handset on the bench would otherwise stamp its own versionName onto someone else's
        # evidence, which is exactly the kind of silent cross-device binding the serial checks
        # everywhere else in this repository exist to prevent.
        if serial == result.manifest.get("device"):
            version_name = _device_version_name(serial, adb_path, package)
    except VideoBoundRefused:
        version_name = None
    if not version_name:
        version_name = result.manifest.get("hinge_version_name")
    if not isinstance(version_name, str) or not version_name.strip():
        raise VideoBoundRefused(
            "no Hinge versionName is available from the phone or the campaign manifest; an "
            "artifact that cannot name the build it was measured on is not evidence")
    artifact = build_artifact(result, config_sha256=_sha(cfg_raw),
                              hinge_version_name=version_name)
    output = campaign / "bound.json"
    payload = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_private_bytes(output, payload, parent=campaign)
    paste = {
        "artifact_path": _inside_repo(output, label="bound artifact"),
        "artifact_sha256": _sha(payload),
        "ground_truth_channel": artifact["ground_truth_channel"],
        "video_cards": artifact["video_cards"],
        "video_accepts": artifact["video_accepts"],
        "photo_cards": artifact["photo_cards"],
        "photo_false_refusals": artifact["photo_false_refusals"],
        "max_video_exact_run_s": artifact["max_video_exact_run_s"],
        "captured_at": artifact["captured_at"],
        "device": artifact["device"],
        "hinge_version_name": artifact["hinge_version_name"],
    }
    expected_paste_keys = PASTE_KEYS
    if artifact["ground_truth_channel"] == CIRCULAR_GROUND_TRUTH_CHANNEL:
        paste["accepted_circular_risk"] = artifact["accepted_circular_risk"]
        expected_paste_keys = PASTE_KEYS + CIRCULAR_PASTE_KEYS
    if tuple(paste) != expected_paste_keys:
        raise AssertionError("still_photo_bound_evidence paste schema drift")
    return artifact, paste


# =====================================================================================
# CLI
# =====================================================================================

def _band_and_device(args, *, need_device: bool):
    cfg_map, cfg_raw = _load_config_mapping(args.config)
    app_cfg = _hinge_app_config(cfg_map)
    band = _effective_content_band(app_cfg, getattr(args, "band", None))
    if not need_device:
        return app_cfg, band, None, None, (None if cfg_raw is None else _sha(cfg_raw))
    serial, adb_path = _resolve_serial(app_cfg, getattr(args, "serial", None))
    return app_cfg, band, serial, adb_path, (None if cfg_raw is None else _sha(cfg_raw))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m tools.hinge_video_bound",
        description="Owner-labeled held-out video false-accept bound for Hinge's still-photo "
                    "dwell discriminator (ops/STILL-PHOTO-DISCRIMINATOR.md section 4). Reads "
                    "frames and asks questions; it never sends anything to the phone.")
    sub = ap.add_subparsers(dest="command", required=True)

    hold = sub.add_parser("hold-test", help="60-second parked-video falsifier (run this first)")
    hold.add_argument("--out", default=None)
    hold.add_argument("--config", default="config.yaml")
    hold.add_argument("--serial", default=None)
    hold.add_argument("--band", default=None, help="content band fractions y0,y1")
    hold.add_argument("--seconds", type=float, default=60.0)
    hold.add_argument("--interval", type=float, default=0.5)

    capture = sub.add_parser("capture", help="owner-labeled bound corpus campaign")
    capture.add_argument("--out", default=None)
    capture.add_argument("--config", default="config.yaml")
    capture.add_argument("--serial", default=None)
    capture.add_argument("--band", default=None, help="content band fractions y0,y1")
    capture.add_argument("--profiles", type=int, required=True)

    measure_cmd = sub.add_parser("measure",
                                 help="offline bound over one or more captured campaigns")
    measure_cmd.add_argument("campaign", nargs="+",
                             help="one or more completed campaign directories; every field the "
                                  "artifact binds must be identical across them")
    measure_cmd.add_argument("--config", default=None)

    emit_cmd = sub.add_parser("emit", help="freeze bound.json and print the config paste block")
    emit_cmd.add_argument("campaign", nargs="+",
                          help="the same directory list `measure` passed; bound.json is written "
                               "into the first")
    emit_cmd.add_argument("--config", default="config.yaml")
    emit_cmd.add_argument("--serial", default=None)

    args = ap.parse_args(argv)
    try:
        if args.command == "hold-test":
            # Device lock (tools/_devicelock.py): this reads only, but a hub run scrolling the
            # deck underneath an owner-labeled hold would silently corrupt the evidence rather
            # than fail, which is the worst way for a measurement campaign to go wrong.
            #
            # `None`, not `args.config`, per holding_the_device's rule: this tool reads its
            # config INDEPENDENTLY of the lock (`_load_config_mapping` parses plain YAML on
            # purpose, so the campaign runs before the key it produces exists), so handing the
            # helper a path it does not consume would only re-arm the unloadable-config escape
            # hatch -- turning an unrelated validation error into a silently UNLOCKED campaign.
            # `None` takes the identical lock; the lock path reads nothing out of a config.
            with holding_the_device(None):
                _app_cfg, band, serial, adb_path, _sha256 = _band_and_device(args,
                                                                            need_device=True)
                out_dir = _private_out_dir(args.out, prefix="videobound_hold")
                run_hold_test(out_dir=out_dir, seconds=args.seconds, interval=args.interval,
                              band=band, serial=serial, adb_path=adb_path)
        elif args.command == "capture":
            if args.profiles < 1:
                raise VideoBoundRefused("--profiles must be at least 1")
            # `None` for the same reason as hold-test above: this campaign never calls
            # config.load, so passing a path through the lock helper buys nothing and arms the
            # escape hatch that would run the whole campaign unlocked.
            with holding_the_device(None):
                app_cfg, band, serial, adb_path, config_sha = _band_and_device(args,
                                                                              need_device=True)
                out_dir = _private_out_dir(args.out, prefix="videobound")
                manifest = run_capture(out_dir=out_dir, profiles=args.profiles, serial=serial,
                                       adb_path=adb_path, band=band,
                                       package=app_cfg.get("package", _DEFAULT_PACKAGE),
                                       config_sha256=config_sha)
            print(f"\nwrote {out_dir / 'manifest.json'} with {len(manifest['cards'])} cards")
        elif args.command == "measure":
            result = measure([_existing_campaign_dir(path) for path in args.campaign],
                             config_path=args.config)
            print_bound_report(result)
            print("\nbound holds. Run `emit` to freeze the artifact.")
        else:
            roots = [_existing_campaign_dir(path) for path in args.campaign]
            campaign = roots[0]
            artifact, paste = emit(roots, config_path=args.config,
                                   serial_override=args.serial)
            print(f"wrote {campaign / 'bound.json'} "
                  f"(evidence_sha256 {artifact['evidence_sha256']})\n")
            print("Paste into config.yaml by hand. These numbers install NUMBERING readiness "
                  "only;\nAuto stays blocked by its own separate release chain.\n")
            print(yaml.safe_dump({"apps": {"hinge": {"still_photo_bound_evidence": paste}}},
                                 sort_keys=False))
    except VideoBoundRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
