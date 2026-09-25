"""One-click bug report — a redacted markdown diagnostic snapshot.

Modeled on infinite-canvas's bug-report feature, adapted to this app. Captures
system + device info, the running build (git commit + a stale-code check), key
dependency versions, runtime CAPABILITY presence (the tesseract OCR binary,
opencv, and which /dev/input/event* device the touch watcher would attach
to — see _capabilities_md), the config (secrets stripped — presence only,
never a value or even a derived prefix), the live run status (phase, labels,
ranker, per-app decisions, budget, last error), a STALL SUMMARY distilled from
each app's on-disk actions.jsonl (the most recent same-reason observe_waiting
repeats, most recent first — see _stall_summary_md), item-index refusal pair evidence
and realised-step ranges, contemporaneous ADB screencap-timeout recovery evidence, and recent
log lines. Output
is markdown the owner can paste to a developer to debug. The report includes
an explicit reporter-follow-up section when its human description is too brief
to provide expected/actual behaviour or a reproduction path, alongside an
instruction to improve this collector when it lacks enough context, and it
caps output at 50k lines by dropping the oldest captured lines first.

The hub serves it at GET /api/bugreport; install_log_capture() (called by the
hub at startup) tees stdout/stderr into a ring buffer so "recent logs" has
content. Pure stdlib so it imports anywhere — the capability probes below
import cv2/touchwatch lazily and swallow their own absence rather than making
that a hard dependency of this module.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

from .typography import format_duration

_MAX_REPORT_LINES = 50_000
_DEBUG_ACTION_TAIL = 30          # actions.jsonl DISPLAY entries to inline from the latest run
# ^ Used to be "raw lines", and the comment here claimed a profile logs at most 2 of those, so
# 30 always reached back ~15 profiles. That broke the day wait_for_decision grew the
# observe_waiting heartbeat (_note_observe_waiting, hinge.py): it fires roughly every 15s of
# human deliberation even while reason="no_change" proves nothing moved, so ONE profile's wait
# alone can burn the entire raw-line budget (a real audited 3-minute decision logged 12 of
# them — data/hinge_debug/run_20260810_203956) and push its own "capture" record — the thing a
# developer needs sitting right next to the decision — out of the tail entirely.
# The real invariant now: _collapse_action_tail groups ADJACENT actions.jsonl lines that share
# the same "action" AND "reason" (i.e. repeats of the same heartbeat) into ONE display entry
# before this budget is ever applied, so 30 means "the last 30 GENUINE events", not "the last
# 30 raw lines, however many of them are the same heartbeat repeated". Every action that isn't
# a repeated action+reason pair (capture, observe_decision, observe_resync, locate_target_heart,
# like, ...) still costs exactly one raw JSON line, unchanged. The very first and last entries
# shown are always kept as literal, uncollapsed records — even when they'd otherwise be part of
# a run — specifically so a live stall's most recent state (its own screenshot filename
# included) is never hidden behind a "repeated: N" summary with no filename in it.
_RECENT_OPENERS_SHOWN = 10        # cap on _recent_openers_md rows -- see its docstring
_RECENT_OPENER_TEXT_CHARS = 240   # per-opener cap so one runaway response can't blow up the report
_RECENT_OPENER_MONITOR_MARKERS_SHOWN = 4  # compact cap for redundancy-monitor details per row
_RECENT_OPENER_MONITOR_MARKER_CHARS = 120  # one malformed marker cannot dominate the section
_RECENT_REJECTIONS_SHOWN = 10     # cap on _recent_opener_rejections_md rows -- mirrors the above
_STALL_STRETCHES_SHOWN = 3        # cap on _stall_summary_md rows -- see its docstring
_ABANDONED_CARDS_SHOWN = 3        # cap on _abandoned_card_summary_md rows -- see its docstring
_CAPTURE_SPLITS_SHOWN = 3          # most recent split/recovery pairs to show — see below
_SHIFT_TRACE_STRIPS_SHOWN = 12     # per-direction cap on rendered strips in a shift-estimator
                                  # trace. frameshift ships a 13-strip bank, so this shows a
                                  # whole ordinary bank and truncates (audibly, with "+N more")
                                  # only a bank taken at a caller-varied strip count.
_MIN_ACTIONABLE_DESCRIPTION_CHARS = 20
_ITEM_INDEX_REASON_INLINE_LIMIT = 360
_COMPLETION_EVIDENCE_MAX_BYTES = 8_000_000
# A 1080x2400 RGBA framebuffer is 10,368,000 bytes before PNG encoding.  32 MiB leaves ample
# headroom for an incompressible capture and PNG framing while preventing a JSONL filename from
# turning report generation into an unbounded file read.
_DEBUG_SCREENSHOT_MAX_BYTES = 32 * 1024 * 1024
# Keep this tied to the public AdbError wording rather than an exception import: actions.jsonl
# is intentionally plain diagnostic data, and older runs may have been written by a process
# whose classes are no longer importable in this one.
_ADB_SCREENCAP_TIMEOUT_RE = re.compile(
    r"\b(?:ADB command timed out after\s+[^:]+:|ADB screencap recovery exhausted after\s+"
    r"\d+\s+read-only attempts;).*\bexec-out\s+screencap\s+-p\b",
    re.IGNORECASE,
)
# These are recovered failures: the opener service logs them while it tries the next model,
# rather than publishing a terminal AppStatus error. Keep this matcher narrow so an unrelated
# diagnostic mentioning a failure cannot change the completion verdict.
#
# It keys on "trying the next configured model" -- the ONE phrase every non-fatal cascade
# branch in opener/opener.py shares (transport failure, a 5xx, a per-minute/unclassified 429, a
# per-day 429, a retired-model 404, a rejected thinking level) -- rather than on any one
# branch's own prose. The shipped matcher named only the TRANSPORT wording, so the other
# branches missed entirely: run 257bdd639ca5 cascaded off a 503, printed it in this very
# report's "Recent logs", and was still stamped COMPLETED CLEANLY. This module is deliberately
# stdlib-only (see the module docstring) so it cannot import those messages from the opener
# package; tests/test_bugreport.py reads opener.py's SOURCE and pins every cascade branch's
# message against this pattern, so prose drift fails the gate rather than silently
# resurrecting that blind spot.
#
# The phrase stands ALONE -- no leading `gemini|provider` requirement. That prefix looked like
# free narrowing and was not: _Tee.write splits captured stdout on "\n" before it reaches the
# ring, and the anchor phrase sits AFTER interpolated text in most branches (the transport one
# interpolates `{type(exc).__name__}: {exc}`), so an exception whose str() carries a newline
# lands the phrase on a ring line with no "gemini" anywhere on it -- and the fault went
# uncounted all over again. The phrase occurs nowhere else in the codebase (opener.py's cascade
# prints and this matcher are the only places it appears at all), so requiring it alone is
# already as narrow as the prefix ever made it.
#
# Attributing a match to a run: as of the run_id threading added to GeminiOpener.generate()
# (opener.py) and OpenerService (service.py:1003, service.py:1301), a cascade print DOES carry
# a `Run {run_id}: ` prefix -- but ONLY when the caller that reached generate() passed one, and
# only in a build that ships the prefixing at all. An older build's ring line, a direct
# `GeminiOpener` construction, or any caller that passes no run id still prints the exact same
# phrase with NO tag at all. See _lines_not_attributed_to_another_run (below) and
# _run_completion_assessment_md for the counting rule this asymmetry forces: a plain "keep only
# this run's own tag" filter would silently drop every one of those untagged lines, which is
# the exact under-report this whole matcher exists to prevent.
_RECOVERED_PROVIDER_FAILURE_RE = re.compile(
    r"\btrying the next configured model\b", re.IGNORECASE)
# supervisor.py's shutdown line for rows the store permanently gave up on. It imports this
# exact constant (bugreport is pure stdlib, so anything may import it; the reverse would drag
# the whole runtime into a diagnostic module), which is what keeps the printer and this reader
# from drifting apart -- the wording is the only channel there is, because the run-level tally
# lives on the store and never reaches RunStatus. tests/test_supervisor.py runs a real
# shutdown's printed line through this pattern so a reworded print cannot quietly stop being
# detected here.
DROPPED_ROWS_NOTICE = "row(s) were PERMANENTLY DROPPED and never written to"
_DROPPED_ROWS_RE = re.compile(
    re.escape(DROPPED_ROWS_NOTICE) + r"\s+\S+\s+\(([^)]*)\)")
_ITEM_INDEX_GEOMETRY_SIDECARS_SHOWN = 2    # cap on _item_index_geometry_md blocks -- one
                                  # refused capture per block, most recent last. Two is the
                                  # incident shape: a dwell walk refuses, the retry refuses.
_ITEM_INDEX_GEOMETRY_ROWS_SHOWN = 6        # per-line cap inside that section: named frame
                                  # indices, distinct leading rows, run extents, and
                                  # unanchored strips. A capture with more than this many
                                  # distinct shapes has no single leading edge to report.
_ITEM_INDEX_GEOMETRY_SIDECAR_BYTES = 8_000_000   # a sidecar larger than this is not read at
                                  # all. Bounded by hinge.py at 64 blocks/runs per frame and
                                  # 32 strips per pair, so a real one is ~1MB; anything past
                                  # this is not the file this section was written against.
# segment.py's block/run vocabulary reaches this module only as JSON tokens. Naming the one
# token this section keys on keeps bugreport.py pure-stdlib (segment.py needs cv2/numpy,
# which this module deliberately never hard-imports) while making the coupling greppable.
_SIDECAR_UNANCHORED_BLOCK_KIND = "unanchored"    # == segment.BLOCK_UNANCHORED
# The same hold-out evidence, written by hinge.py's `geometry_record` on EVERY block
# unconditionally (both keys, `unanchored_reason` merely None off an unanchored strip). Their
# PRESENCE is therefore what dates a sidecar against the screen-fixed island check: they shipped
# in the same commit as `segment.BLOCK_UNANCHORED` and `item_index._screen_fixed_islands`, so a
# block carrying either key was written by a writer that already had the check. See
# `_sidecar_geometry_lines`' final branch for why neither `schema_version` nor the runtime hashes
# can date it -- both predate that commit.
_SIDECAR_SCREEN_FIXED_EVIDENCE_KEYS = ("unanchored_reason", "content_digest")

# The driver outcome token an operator Stop mid-navigation writes in place of
# "navigation_refused" (hinge.py's `_dbg_still_photo_walk_candidate`, 2026-09-16). Named here so
# both the dwell-navigation renderer and the Stop-attribution fallback key on ONE constant: a
# token missing from the renderer's accepted set renders NOTHING at all for the event.
_DWELL_NAVIGATION_CANCELLED = "navigation_cancelled"

# The five exception classes hinge.py's two uncoded dwell-navigation handlers catch, recorded on
# the row verbatim as `reason=type(exc).__name__`. None of them carries a measurement, so a row
# naming one of these has no plan/climb/anchor-return telemetry BY CONSTRUCTION -- it is not an
# old row, and describing it as a "legacy/incomplete trace" asserted an age nothing on the row
# supports (found 2026-09-16 on a row the CURRENT build had written minutes earlier).
# A DUPLICATED LIST, deliberately: this module is pure-stdlib on purpose and never imports the
# drivers (hinge.py pulls in cv2/numpy), so the names are copied rather than derived. It fails
# SOFT if hinge.py ever widens that catch tuple -- an unlisted class simply gets the neutral
# "carries no structured refusal telemetry" sentence, which is still true of it. The one thing
# this list must never do is gain a name hinge.py does NOT raise, which would assert an absence
# of measurement for a refusal that had one.
_UNCODED_DWELL_NAVIGATION_REFUSAL_CLASSES = frozenset({
    "ActionCancelled", "ScrollStepError", "SegmentationError", "ShiftEstimationError",
    "IdentityError",
})
_LOG_RING: deque[str] = deque(maxlen=_MAX_REPORT_LINES)
_LOG_LOCK = threading.Lock()
_PROCESS_START = time.time()
_REPO = Path(__file__).resolve().parent.parent
_PKG = Path(__file__).resolve().parent


# ── log capture ────────────────────────────────────────────────────────────
class _Tee:
    """Mirror a stream to the ring buffer (timestamped) and through to the original."""

    def __init__(self, stream):
        self._s = stream
        self._buffers: dict[int, str] = {}

    def write(self, data):
        text = str(data)
        try:
            ident = threading.get_ident()
            with _LOG_LOCK:
                chunk = self._buffers.get(ident, "") + text
                lines = chunk.split("\n")
                self._buffers[ident] = lines.pop()
                if not self._buffers[ident]:
                    self._buffers.pop(ident, None)
                for line in lines:
                    line = line.rstrip("\r")
                    if line.strip():
                        _LOG_RING.append(f"{datetime.now().strftime('%H:%M:%S')} {line}")
        except Exception:  # noqa: BLE001
            pass
        return self._s.write(data)

    def flush(self):
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


def install_log_capture() -> None:
    """Tee stdout/stderr into the ring so reports include recent logs. Idempotent.

    Test runners, notebooks, and process supervisors can replace ``sys.stdout`` or
    ``sys.stderr`` after startup.  A process-global installed flag alone is therefore not
    enough: it would say capture is active while writes are going to a newer, unwrapped stream.
    Check the live streams independently on every call, avoiding nested tees when they are
    already wrapped and restoring capture when either one has been replaced.
    """
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr)


def recent_logs(limit: int = 120) -> list[str]:
    # Hold the lock for the snapshot: _Tee.write appends under _LOG_LOCK, and the hub
    # polls this every ~1.5s during an actively-logging run — without the lock,
    # list(_LOG_RING) can race an append and raise "deque mutated during iteration".
    with _LOG_LOCK:
        return list(_LOG_RING)[-limit:]


# ── section builders ───────────────────────────────────────────────────────
def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                           text=True, timeout=4, check=False)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _newest_source_mtime() -> float:
    newest = 0.0
    for p in _PKG.rglob("*.py"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return newest


def _device() -> str:
    try:
        from .device import best_device
        return best_device()
    except Exception:  # noqa: BLE001
        return "unknown"


def _dep_versions() -> dict[str, str]:
    deps = ["playwright", "torch", "insightface", "open_clip", "onnxruntime",
            "google.cloud.bigquery", "dotenv", "PIL", "sklearn", "yaml"]
    out: dict[str, str] = {}
    for d in deps:
        try:
            out[d] = getattr(importlib.import_module(d), "__version__", "(installed)")
        except Exception:  # noqa: BLE001
            out[d] = "— not installed"
    return out


def _build_md() -> str:
    commit = _git("rev-parse", "--short", "HEAD") or "(unknown)"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "(unknown)"
    dirty = _git("status", "--porcelain")
    clean = "clean" if dirty == "" else f"DIRTY ({len(dirty.splitlines())} file(s) changed)" if dirty is not None else "(unknown)"
    uptime = int(time.time() - _PROCESS_START)
    stale = _newest_source_mtime() > _PROCESS_START
    stale_note = ("⚠️ a source file changed AFTER this process started — restart "
                  "the hub/app to run the new code" if stale else "up to date")
    return (f"- git: `{commit}` on `{branch}` — {clean}\n"
            f"- process uptime: {uptime}s · running code: {stale_note}")


def _system_md() -> str:
    return (f"- platform: {platform.system()} {platform.machine()} ({platform.release()})\n"
            f"- python: {platform.python_version()} ({sys.executable})\n"
            f"- device: {_device()}\n"
            f"- cpu count: {os.cpu_count()}")


def _deps_md() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in _dep_versions().items())


# ── capability probes ──────────────────────────────────────────────────────
# The three things the observe-mode redesign (see ops/ANTI-BOT-RESEARCH.md, 2026-08-10) leans
# on that are otherwise invisible anywhere else in this report: none of them show up in
# _dep_versions (tesseract is a system binary, not a Python import; cv2 is deliberately NOT in
# that list because a missing opencv install is not a soft degradation for this project — see
# _opencv_md) or in _config_md (the config can declare observe_touch_watch=true while the
# actual device probe silently fails or picks a surprising node). This is the bug report's
# owner-filed self-improvement in action: the original report gave no way to tell "OCR is off
# because tesseract isn't installed" apart from "OCR is off because nothing scrolled the
# profile far enough yet", nor whether a gesture-corroboration failure traced back to the
# wrong /dev/input node. Presence-only, same redaction contract as _secrets_md: a path or a
# device NAME is not a secret, but this never becomes a place to grow raw device output.
def _tesseract_md() -> str:
    path = shutil.which("tesseract")
    return (f"- tesseract: {path or 'absent'} — profile-name OCR (see hinge.py's _ocr_band) "
            f"can veto a transient pixel mismatch and must confirm a different scroll-top "
            f"name twice; when absent, identity safety remains fail-closed but a real advance "
            f"may stay unresolved")


def _opencv_md() -> str:
    try:
        import cv2
        return f"- opencv (cv2): present ({getattr(cv2, '__version__', '(installed)')})"
    except Exception as exc:  # noqa: BLE001 — absence itself IS the diagnostic, not a crash
        return (f"- opencv (cv2): absent ({type(exc).__name__}: {exc}) — every Android "
                f"driver refuses to open a session without it (AndroidDriver._require_vision "
                f"in drivers/hinge.py); a launcher shipped without the `hinge` extra has left "
                f"this silently missing before, so this line exists to make that visible "
                f"without waiting for every glyph-match call to fail")


def _first_android_app_cfg(apps: dict, enabled_apps: list) -> tuple[str, str | None] | None:
    """(adb_path, serial) for the ADB target a live run would actually probe.

    Android-driven apps declare `adb_path`, so its presence distinguishes a device entry
    without hardcoding app names here. Prefers the first ENABLED app that declares one — that
    is the device a real run would actually talk to — and falls back to ANY app with
    `adb_path` so a report generated with a stale/mismatched `enabled_apps` (or none at all)
    can still say something concrete about the phone this project talks to, rather than
    reporting "not configured" when it plainly is.
    """
    candidates = [apps.get(name) for name in (enabled_apps or [])]
    candidates += list(apps.values())
    for opts in candidates:
        if isinstance(opts, dict) and "adb_path" in opts:
            return opts.get("adb_path", "adb"), (opts.get("serial") or None)
    return None


def _touch_watcher_probe_md(config_path: str) -> str:
    """Which /dev/input/event* device layer 3's gesture corroboration (touchwatch.py) would
    attach to, or why none qualified — WITHOUT ever leaving anything attached.

    Runs the SAME read-only `adb shell getevent -p` device-selection TouchWatcher.start()
    itself runs first, through the real class rather than a reimplementation, so this can
    never drift from what a live session actually selects. It deliberately goes through
    start()/close() rather than reaching into TouchWatcher's private probe helper: start() is
    the one place that owns "pick a device, then decide whether to keep going", and using the
    public method means a future change to that decision can't silently stop being reflected
    here. The stream it briefly attaches (`getevent -lt`, read-only — see touchwatch.py's
    module docstring; this never writes to /dev/input) is torn down in the `finally` before
    this function returns, so a bug report can never leave a background reader thread/process
    attached to the phone. Safe with no device connected or `adb` itself missing:
    TouchWatchUnavailable is exactly the "reason none qualified" case this reports as a plain
    line, not a crash — the same contract AndroidDriver.open_session relies on.
    """
    try:
        from . import config as cfg_mod
        c = cfg_mod.load(config_path)
    except Exception as exc:  # noqa: BLE001
        return (f"- touch watcher: could not load `{config_path}` to find the ADB target "
                f"({_sanitize_inline(str(exc))})")
    apps = c.apps if isinstance(c.apps, dict) else {}
    target = _first_android_app_cfg(apps, getattr(c, "enabled_apps", None) or [])
    if target is None:
        return "- touch watcher: no Android app configured (no `apps.*.adb_path` in config)"
    adb_path, serial = target

    from .drivers.touchwatch import TouchWatcher, TouchWatchUnavailable
    # screen_size only scales GESTURE coordinates after a device has already been selected
    # (TouchWatcher.start()); device selection itself never reads it, so a placeholder is
    # fine — this probe never reaches the point that would need a real one.
    watcher = TouchWatcher(adb_path, serial, (0, 0), probe_timeout=4.0)
    try:
        watcher.start()
    except TouchWatchUnavailable as exc:
        return f"- touch watcher: unavailable — `{_sanitize_inline(str(exc))}`"
    finally:
        # Best-effort teardown on EITHER path: on success this stops the live stream start()
        # just attached; on failure close() is a documented no-op (nothing was ever attached,
        # since the device-selection probe runs before the Popen it would need to stop).
        watcher.close()
    return (f"- touch watcher: would select `{watcher.device_path}` "
            f"(name: `{_sanitize_inline(watcher.device_name or '')}`)")


def _capabilities_md(config_path: str) -> str:
    return "\n".join([_tesseract_md(), _opencv_md(), _touch_watcher_probe_md(config_path)])


def _secrets_md() -> str:
    def shown(name: str) -> str:
        key = os.environ.get(name, "")
        return "present" if key else "unset (needed only for openers / auto mode)"
    # Never include credentials or even a credential-derived prefix.
    return f"- GEMINI_API_KEY: {shown('GEMINI_API_KEY')}"


_TARGETING_CALIBRATION_UNAVAILABLE = "apps.hinge.targeting_calibration is unavailable"


def _runtime_targeting_calibration_rejection(hub_state) -> str | None:
    """Return the latest recorded live-calibration refusal, without reading the device.

    Config validation can establish that a calibration mapping is internally sound, but only the
    running driver knows whether that mapping is bound to the Hinge build and frame currently on
    the phone.  Keep this narrowly keyed to the driver's explicit refusal so unrelated opener
    failures do not turn into misleading calibration guidance.
    """
    if hub_state is None:
        return None
    try:
        snapshot = hub_state.snapshot()
        status = snapshot.get("status") if isinstance(snapshot, dict) else None
        apps = status.get("apps") if isinstance(status, dict) else None
        hinge = apps.get("hinge") if isinstance(apps, dict) else None
        reason = hinge.get("stop_reason") if isinstance(hinge, dict) else None
        kind = hinge.get("stop_kind") if isinstance(hinge, dict) else None
    except Exception:  # noqa: BLE001 -- diagnostics must not depend on a stable hub snapshot
        return None
    if not isinstance(reason, str):
        return None
    # The typed kind is the current contract. Keep the driver sentence as a fallback so reports
    # generated from status snapshots written before the kind existed remain actionable.
    if kind != "targeting_calibration" and _TARGETING_CALIBRATION_UNAVAILABLE not in reason:
        return None
    return _sanitize_inline(reason)


def _content_band_rows(band: object, frame_size_px: object) -> tuple[int, int, int] | None:
    """`content_band`'s ``(r0, r1, frame_height)`` pixel rows, or None if it cannot be derived.

    Deliberately the SAME arithmetic as ``hinge._content_rows`` (round, then clamp so a
    degenerate size can never produce an empty or inverted slice), because a report that
    printed a second, independently-drifting derivation of the analysed band would be worse
    than printing nothing. Never raises: a hand-edited config is still a reportable config.
    """
    if not (isinstance(band, (list, tuple)) and len(band) == 2):
        return None
    if not (isinstance(frame_size_px, (list, tuple)) and len(frame_size_px) == 2):
        return None
    # ``bool`` subclasses ``int`` and would otherwise turn a malformed calibration into a
    # plausible-looking one-pixel/whole-frame geometry line.
    if any(isinstance(value, bool) for value in (*band, frame_size_px[1])):
        return None
    try:
        y0, y1 = float(band[0]), float(band[1])
        size = int(frame_size_px[1])
    except Exception:  # noqa: BLE001 -- a malformed calibration reports without this line
        return None
    if size <= 1 or not (math.isfinite(y0) and math.isfinite(y1)):
        return None
    r0 = max(0, min(size - 1, round(y0 * size)))
    r1 = max(r0 + 1, min(size, round(y1 * size)))
    return r0, r1, size


def _targeting_readiness_md(config_path: str, hub_state=None) -> str:
    """Why Hinge is or is not offering numbered targeted openers, as three separable facts.

    ADDED 2026-08-22 with the fix for the bug this section exists to have caught. The report
    that prompted it said `opener: enabled=True`, `GEMINI_API_KEY: present`, and zero provider
    calls, and left the actual cause -- one absent config key -- to be reconstructed from a
    driver sentence buried in the run log. Worse, the two facts that together explain it sat in
    different places and contradicted each other on a skim: the log announced that targeted
    suggestions were ENABLED under an accepted assumption, while the hub banner said still-photo
    proof was still required.

    So report the whole chain, in the order it gates: the still-photo licence, config-calibration
    validity, then the runtime binding and the one step that would clear a refusal. Any of those
    can be the answer, and which one it is has never been visible here.

    Deliberately side-effect free: it reads the config file and the process-local licence slot
    but never calls ``config.validate()``, which would clear and reinstall that slot underneath
    a live run. When the report is generated from the hub the slot is already populated by the
    run's own validation, which is exactly the state worth reporting; from a bare CLI it is
    empty, and the config-key line below is what carries the answer instead.
    """
    from . import targeting_policy as tp
    lines: list[str] = []
    try:
        import yaml
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
        app = ((raw.get("apps") or {}).get("hinge") or {}) if isinstance(raw, dict) else {}
        if not isinstance(app, dict):
            app = {}
    except Exception as exc:  # noqa: BLE001 -- a diagnostic section must not raise over config
        return f"- ⚠️ could not read `{config_path}` for targeting state: {exc}"

    licence_keys = [k for k in ("still_photo_bound_evidence", "still_photo_assumption_acceptance")
                    if k in app]
    lines.append("- still-photo licence key in config: "
                 + (", ".join(f"`apps.hinge.{k}`" for k in licence_keys) if licence_keys
                    else "none — numbering is refused everywhere until one is installed"))
    provenance = tp.still_photo_licence_provenance()
    lines.append("- licence installed in THIS process: "
                 + (provenance if provenance else
                    "no (expected when the report is generated outside a validated run)"))
    calibration = app.get("targeting_calibration")
    calibration_valid = False
    calibration_problem = ""
    if isinstance(calibration, dict) and tp.hinge_targeting_unavailable_reason() is None:
        # The report must not mistake a mapping's mere presence for a calibration licence.  This
        # narrow validator reads no device and changes no process-global readiness state; it
        # verifies exactly the mapping that a live driver would rely on.  Full
        # config.validate() is intentionally not called here because it clears/reinstalls the
        # live process's still-photo licence.
        try:
            from . import config as cfg_mod
            cfg_mod._validate_targeting_calibration(cfg_mod.load(config_path))
            calibration_valid = True
        except Exception as exc:  # noqa: BLE001 -- a malformed report config remains reportable
            calibration_problem = _sanitize_inline(str(exc))
    if calibration_valid:
        calibration_line = "present and validated (static config check)"
        # The NUMBERS, not just the verdict. The 2026-08-27 stop was refused at 7.000 while this
        # config's own `inline_item_max_dist` was 14.9099, and a reader with only "validated"
        # printed here had no way to see that the operator's calibrated ceiling was not the
        # bound that fired — it is a second, stricter cap applied after a bound derived inside
        # `item_verify`. These are geometry and thresholds, never secrets.
        if isinstance(calibration, dict):
            shown = [key for key in ("hinge_version_name", "frame_size_px", "content_band",
                                     "composer_layout_id", "inline_item_max_dist",
                                     "identity_match_max_dist",
                                     "calibrated_at") if calibration.get(key) is not None]
            if shown:
                calibration_line += "; " + ", ".join(
                    f"{key}={_sanitize_inline(json.dumps(calibration[key]))}" for key in shown)
            # `content_band` is two fractions, and every row number the geometry sections below
            # print is measured INSIDE the band those fractions cut. Nobody reading "block top
            # at frame row 368" can check it against [0.125, 0.875] in their head, so derive the
            # rows here, once, the same way `hinge._content_rows` does.
            band_rows = _content_band_rows(calibration.get("content_band"),
                                           calibration.get("frame_size_px"))
            if band_rows is not None:
                r0, r1, size = band_rows
                calibration_line += (f"; that content_band analyses frame rows {r0}..{r1} of "
                                     f"{size} — every geometry row reported elsewhere in this "
                                     f"report is measured inside it")
    elif isinstance(calibration, dict):
        calibration_line = ("present but not validated"
                            + (f": {calibration_problem}" if calibration_problem else ""))
    else:
        calibration_line = ("ABSENT — no numbered item list, so no targeted opener is generated "
                            "or offered")
    lines.append("- `apps.hinge.targeting_calibration`: " + calibration_line)
    blocker = tp.hinge_targeting_unavailable_reason()
    lines.append("- targeting-policy blocker: " + (blocker if blocker else "none"))
    runtime_rejection = _runtime_targeting_calibration_rejection(hub_state)
    if runtime_rejection:
        lines.append("- runtime calibration: REJECTED in the latest hub snapshot: `"
                     + runtime_rejection + "`")
        lines.append("- next step: recapture and validate a schema-v3 targeting calibration for "
                     "the live Hinge build/frame geometry; also recapture measured still-photo "
                     "evidence or explicitly re-accept the unmeasured still-photo assumption "
                     "for that exact build/device, then resume training")
    elif blocker is not None and licence_keys:
        # Do not let an unvalidated reporting process contradict the config two lines above.
        # The blocker and next step are read from THIS process's slot; a config that carries a
        # licence key would install one the moment a run validated it, and printing the
        # pre-licence next step without saying so is precisely the stale-guidance failure this
        # section was added to catch.
        lines.append("- ⚠️ the blocker above reflects this unvalidated reporting process, not a "
                     "run: the config key above would install a licence when a run validates "
                     "it. Re-check inside the run, or with `config.validate(config.load(...))`")
    elif calibration_valid and blocker is None:
        lines.append("- targeting config readiness: ready — the still-photo licence and the "
                     "calibration mapping are valid in config. A live session still requires an "
                     "exact Hinge app-version/frame match.")
    elif calibration is not None and blocker is None:
        lines.append("- next step: repair or replace the invalid targeting calibration before "
                     "treating numbered targeted suggestions as ready")
    else:
        lines.append("- next step: " + tp.targeting_setup_next_step())
    return "\n".join(lines)


def _diagnostic_improvement_md() -> str:
    return (
        "- If this report is not enough to diagnose the issue, improve "
        "`operation_love/bugreport.py` to capture the missing diagnostics "
        "(state, logs, config/build context, or reproduction details) while "
        "keeping secrets redacted, then include that reporting improvement "
        "with the bug fix."
    )


def _training_alerts_md() -> str:
    """Render bounded, secret-free notification and sound outcomes from this Hub process."""
    from .notifications import recent_training_alerts
    alerts = recent_training_alerts()
    if not alerts:
        return "- No Training alert attempt has been recorded in this Hub process."
    lines = []
    for row in alerts[-12:]:
        at = _sanitize_inline(row.get("at") or "time unavailable")
        channel = _sanitize_inline(row.get("channel") or "unknown")
        outcomes = [
            f"{name}={_sanitize_inline(value)}"
            for name, value in row.items()
            if name not in {"at", "channel"}
        ]
        lines.append(f"- `{at}` · {channel} · " + ("; ".join(outcomes) or "no outcome"))
    return "\n".join(lines)


def _reporter_follow_up_md(description: str) -> str:
    """Explain precisely which human-side evidence is absent from a terse report.

    Runtime snapshots can tell us what the app was doing, but cannot infer what the
    operator expected to happen or the interaction that made them report a problem.
    A one-word description (the real report that prompted this addition was simply
    ``bug``) leaves every captured log line compatible with both a defect and normal
    observe-mode waiting. Do not pretend this is recoverable from telemetry: call it
    out, and make the requested follow-up bounded and copy/paste-friendly.

    This is deliberately a conservative length check rather than keyword guessing.
    A short description cannot contain all of expected behaviour, observed behaviour,
    triggering action, and recurrence/timing; a longer description is allowed through
    unchanged because only its author can judge whether it is sufficient.
    """
    if len(_sanitize_inline(description)) >= _MIN_ACTIONABLE_DESCRIPTION_CHARS:
        return "- Reporter description is present; runtime diagnostics are captured below."
    return (
        "- ⚠️ The report description is too brief to diagnose from telemetry alone.\n"
        "- Add: the expected result; the actual result; the last action or steps that led "
        "to it; the phone/app screen shown; and whether it repeats (with an approximate time)."
    )


def _config_md(config_path: str) -> str:
    try:
        from . import config as cfg_mod
        c = cfg_mod.load(config_path)
        bq = c.storage.bigquery if isinstance(c.storage.bigquery, dict) else {}
        return (f"- enabled_apps: {c.enabled_apps}\n"
                f"- mode: {c.mode}\n"
                f"- storage: {c.storage.backend} (project_id={bq.get('project_id', '?')}, "
                f"photo_bucket={bq.get('photo_bucket', '?')})\n"
                f"- ranker: min_labels={c.ranker.min_labels_to_engage}, "
                f"threshold={c.ranker.like_threshold}, retrain_every={c.ranker.retrain_every}\n"
                f"- quality_filter: enabled={c.quality_filter.enabled}, "
                f"metric={c.quality_filter.metric}, min_score={c.quality_filter.min_score}\n"
                f"- opener: enabled={c.opener.enabled}, provider={c.opener.provider}, "
                f"models={c.opener.effective_models}, "
                f"max_tokens={c.opener.max_tokens}\n"
                f"- budget: run_budget_usd={c.budget.run_budget_usd}, "
                f"opener.max_attempts={c.opener.max_attempts}")
    except Exception as exc:  # noqa: BLE001
        return f"- ⚠️ could not load `{config_path}`: {exc}"


def _sanitize_inline(text: str) -> str:
    """Make free text that we did NOT author (provider error strings, opener-retry-exhaustion
    summaries, HALT-on-unexpected exception text) safe to embed as a single markdown line.
    Two hazards: an embedded newline could start what reads as a new bullet/heading and
    restructure the report around it, and an embedded backtick could prematurely close the
    inline-code span this text is rendered inside (letting the rest of the string escape into
    literal markdown). ``" ".join(text.split())`` collapses every whitespace run — including
    newlines/tabs — to a single space, and the backtick swap neutralises the other hazard. A
    stray `|` is left untouched: it's harmless prose once rendered outside a table cell (see
    _app_diagnostics_md), which is exactly why that section exists instead of a wide table
    column."""
    return " ".join(text.split()).replace("`", "'")


_SECRET_ENV_NAME_RE = re.compile(
    r"(?:^|_)(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|credential|password|"
    r"secret|token)(?:$|_)",
    re.IGNORECASE,
)
_AUTH_HEADER_RE = re.compile(
    r"(?i)(\b(?:proxy-)?authorization\s*[:=]\s*)(?:bearer|token|basic)\s+[^\s,;`]+",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;`]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|token|secret|password|"
    r"credential)\b\s*[=:]\s*)(?:[\"']?)([^\s&#,\"'`]+)",
)


def _redact_report_output(text: str) -> str:
    """Remove credentials from the COMPLETE rendered report, not just its secrets section.

    Diagnostics collect third-party errors, JSONL rows and stdout/stderr.  Any of those can echo a
    credential even when ``_secrets_md`` correctly reports presence only.  Redacting at the final
    rendering boundary makes every existing and future section safe by default, while the section
    builders may still retain their useful, typed presentation logic.  Literal values are taken
    only from plausibly-secret environment-variable names; redacting all environment values would
    turn ordinary paths and configuration into misleading ``[REDACTED]`` text.
    """
    values = sorted({value for name, value in os.environ.items()
                     if value and _SECRET_ENV_NAME_RE.search(name)}, key=len, reverse=True)
    for value in values:
        # Do not derive or display any part of the credential.  The length guard avoids replacing
        # incidental one-character values such as a test flag throughout normal prose.
        if len(value) >= 4:
            text = text.replace(value, "[REDACTED]")
    text = _AUTH_HEADER_RE.sub(r"\1[REDACTED]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)


def _capture_window_has_dwell_walk_stop(records: list[dict], capture_index: int) -> bool:
    """Whether Stop interrupted the candidate walk for one completed capture window.

    TWO tokens, because the walk has two ways of observing the same Stop. "stop" is the
    COOPERATIVE one: the loop polled `should_stop` at the top of an iteration and never started
    the hop. "navigation_cancelled" (2026-09-16) is the one raised THROUGH `navigate_to_item`
    after the hop began, which the driver distinguishes from a genuine measurement fault by
    reading `should_stop` in the handler that caught the cancellation. Both say the operator
    ended the walk; only the second one existed in the run that exposed this gap.
    """
    for prior in reversed(records[:capture_index]):
        if prior.get("action") == "capture":
            break
        if (prior.get("action") == "still_photo_dwell_walk_candidate"
                and prior.get("outcome") in {"stop", _DWELL_NAVIGATION_CANCELLED}):
            return True
    return False


def _capture_stop_interrupted_gap_count(coverage: object, records: list[dict],
                                        capture_index: int) -> int:
    """Count only capture-scoped gaps interrupted by Stop, with an old-log fallback."""
    if not isinstance(coverage, dict):
        return 0
    gaps = coverage.get("no_dwell_coverage_page_hearts")
    if not isinstance(gaps, list):
        return 0
    interruption = coverage.get("dwell_walk_interruption")
    if isinstance(interruption, dict) and interruption.get("reason") == "stop_requested":
        unattempted = interruption.get("unattempted_within_remaining_limit_page_hearts")
        interrupted = interruption.get("interrupted_page_hearts")
        if not isinstance(unattempted, list) and not isinstance(interrupted, list):
            return 0
        gap_ordinals = {value for value in gaps
                        if isinstance(value, int) and not isinstance(value, bool)}
        interrupted_ordinals = {
            value
            for values in (unattempted, interrupted)
            if isinstance(values, list)
            for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return len(gap_ordinals & interrupted_ordinals)
    # Historic logs have no capture-scoped list. Within their bounded capture window, a `stop`
    # row is the best available evidence; attribute its unobserved gaps but never spill into a
    # preceding capture or another app.
    return len(gaps) if _capture_window_has_dwell_walk_stop(records, capture_index) else 0


def _completion_capture_facts(st: dict, config_path: str) -> dict[str, object]:
    """Read only the active/last run's structured capture facts, best-effort.

    A debug base can contain many old runs, so this intentionally refuses to fall back to its
    newest folder: only a directory named by the status snapshot's run_id belongs to the run
    being assessed. The facts already exist in the driver JSONL; this extracts its last capture.

    ``items_unavailable`` IS READ HERE BECAUSE THE COVERAGE CHANNEL IS ANTI-CORRELATED WITH IT
    (found 2026-09-16, run f78ca90856b4). This function used to read only ``item_coverage`` and
    ``capture_truncated``, and hinge.py writes ``item_coverage=None`` exactly when the payload is
    None -- i.e. whenever enumeration produced nothing at all. So the one Stop-aware limitation
    the verdict had could only fire for a capture that still enumerated SOME items, while a Stop
    that destroyed the item index ENTIRELY left every fact at zero and the durable verdict read
    "COMPLETED CLEANLY" over an abandoned profile. The refusal was never invisible -- three other
    sections render it -- but the verdict line is the durable one, and it was silent.
    """
    run_id = st.get("run_id")
    if (not isinstance(run_id, str) or not run_id or run_id in {".", ".."}
            or Path(run_id).name != run_id):
        return {}
    try:
        from . import config as cfg_mod
        cfg = cfg_mod.load(config_path)
    except Exception:  # noqa: BLE001 -- completion reporting must not depend on config loading
        return {}
    apps = cfg.apps if isinstance(cfg.apps, dict) else {}
    facts: dict[str, object] = {"coverage_gaps": 0, "coverage_candidates": 0,
                                "coverage_stop_interrupted_gaps": 0,
                                "coverage_stop_interrupted_candidates": 0,
                                "capture_truncated": False,
                                "items_unavailable": None,
                                "items_unavailable_kind": None,
                                "items_unavailable_profile": None,
                                "items_unavailable_after_stop": False}
    for app in cfg.enabled_apps:
        opts = apps.get(app)
        if not isinstance(opts, dict) or not opts.get("debug_log"):
            continue
        base = Path(opts.get("debug_dir", f"./data/{app}_debug"))
        if not base.is_absolute():
            base = Path.cwd() / base
        run = base / run_id
        log = run / "actions.jsonl"
        try:
            # The run id is status input, not authority to follow a debug-directory symlink.
            # The same guard also rules out FIFOs/devices, whose ``read_text`` could block a
            # bug-report request indefinitely instead of examining bounded JSONL evidence.
            if (run.is_symlink() or not run.is_dir() or log.is_symlink() or not log.is_file()
                    or log.stat().st_size > _COMPLETION_EVIDENCE_MAX_BYTES):
                continue
            rows = log.read_text().splitlines()
        except Exception:  # noqa: BLE001 -- optional evidence may be absent or mid-write
            continue
        records = _action_records(rows)
        for capture_index in range(len(records) - 1, -1, -1):
            record = records[capture_index]
            if record.get("action") != "capture":
                continue
            coverage = record.get("item_coverage")
            if isinstance(coverage, dict):
                gaps = coverage.get("no_dwell_coverage_page_hearts")
                candidates = coverage.get("photo_candidate_page_hearts")
                if isinstance(gaps, list):
                    facts["coverage_gaps"] = int(facts["coverage_gaps"]) + len(gaps)
                if isinstance(candidates, list):
                    facts["coverage_candidates"] = int(facts["coverage_candidates"]) + len(candidates)
                interrupted_gaps = _capture_stop_interrupted_gap_count(
                    coverage, records, capture_index)
                if interrupted_gaps:
                    facts["coverage_stop_interrupted_gaps"] = (
                        int(facts["coverage_stop_interrupted_gaps"]) + interrupted_gaps)
                    if isinstance(candidates, list):
                        facts["coverage_stop_interrupted_candidates"] = (
                            int(facts["coverage_stop_interrupted_candidates"]) + len(candidates))
            facts["capture_truncated"] = bool(facts["capture_truncated"] or
                                               record.get("capture_truncated"))
            # The SAME row's top-level refusal fields (hinge.py writes them beside
            # `item_coverage`, deliberately NOT inside it -- see that call site's own comment:
            # "`item_coverage` is None whenever the payload is -- which is exactly the refusal
            # case this aggregate exists to explain"). Only the first enabled app to report one
            # wins, matching every other fact here: this is a verdict line, not a per-app table.
            unavailable = record.get("items_unavailable")
            if unavailable and facts["items_unavailable"] is None:
                facts["items_unavailable"] = str(unavailable)
                kind = record.get("items_unavailable_kind")
                facts["items_unavailable_kind"] = str(kind) if kind else None
                profile = record.get("profile_name")
                facts["items_unavailable_profile"] = str(profile) if profile else None
                # WHY A STOP IS ASSERTED ONLY FROM EVIDENCE. The two causes read identically on
                # this row -- an abandoned index says nothing about who abandoned it -- so the
                # cause is taken from the driver's own walk rows inside THIS capture's window:
                # a cooperative "stop" or the mid-navigation "navigation_cancelled" token. With
                # no such row the verdict says targeting failure, never "probably a Stop".
                facts["items_unavailable_after_stop"] = _capture_window_has_dwell_walk_stop(
                    records, capture_index)
            break
    return facts


def _lines_for_run(lines: list[str], run_id: object) -> list[str]:
    """Only the log lines the named run itself printed -- `[]` when the run cannot be named.

    _LOG_RING is process-global and installed ONCE by install_log_capture() at hub startup; it
    holds _MAX_REPORT_LINES lines and nothing ever clears it, while supervisor.run() runs on a
    thread inside that same hub process. So every run of a hub session writes into one ring,
    and a report built for run B can read a line run A printed. That is not hypothetical
    house-keeping: the dropped-rows notice below is a claim that THIS run's records are
    incomplete forever, and an unscoped read of the ring would put run A's permanent data loss
    on run B's otherwise clean report.

    Scopes on `Run {run_id}: `, the prefix supervisor.py stamps on its run-level lines. That
    prefix is a fixed contract between the printer and this reader -- do not change it on
    either side without changing both.

    An unnameable run (no run_id in the snapshot, or a non-string one) gets NO lines rather
    than all of them: a line this report cannot attribute to the run it is assessing is
    evidence about some other run, and reporting it here would be exactly the contamination
    this function exists to stop. Use this ONLY where under-reporting is the safe direction
    (the dropped-rows tally just below: a line without this run's own tag says nothing about
    THIS run's data loss, so it is correctly excluded). opener.py's cascade prints CAN carry
    this same prefix now (GeminiOpener.generate() accepts an optional run_id -- see
    service.py:1003 and :1301), but not every caller supplies one and older rings may hold
    lines from before that prefixing shipped, so a plain intersection filter on THIS function
    would silently drop those untagged lines. The provider-fault count in
    _run_completion_assessment_md needs the opposite bias (over-report, never under-report) and
    uses _lines_not_attributed_to_another_run below instead of this function for exactly that
    reason.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        return []
    prefix = re.compile(r"\bRun\s+" + re.escape(run_id.strip()) + r":")
    return [line for line in lines if prefix.search(line)]


# Any "Run <token>: " tag at all, capturing the token -- as opposed to _lines_for_run's regex,
# which only ever tests for ONE specific, already-known run_id. This one has to discover
# whatever id (if any) a line is tagged with, so it can be compared against the run actually
# being assessed.
_ANY_RUN_TAG_RE = re.compile(r"\bRun\s+(\S+):")


def _lines_not_attributed_to_another_run(lines: list[str], run_id: object) -> list[str]:
    """Every line EXCEPT one explicitly tagged for a run other than the one being assessed.

    This is the counting rule the recovered-provider-failure tally in
    _run_completion_assessment_md needs, and it is deliberately NOT _lines_for_run: that
    function keeps only a line carrying THIS run's own `Run {run_id}: ` tag, which silently
    drops every cascade line that carries NO tag at all -- an older build, a direct
    `GeminiOpener` construction, or any caller that passes no run id (see
    OpenerClient.generate's own docstring in opener.py for why run_id is optional there). An
    uncounted cascade is the ORIGINAL defect this whole counter exists to catch (a 503 that let
    a run get stamped "COMPLETED CLEANLY" -- run 257bdd639ca5), so under-reporting by filtering
    too hard is not an acceptable trade for a tidier scope label.

    The rule actually applied, three cases:
      1. A line carrying THIS run's own `Run {run_id}: ` tag -- kept (it is unambiguously this
         run's cascade).
      2. A line carrying NO run tag at all -- kept. Over-reporting an earlier run's untagged
         cascade as if it might be this run's is the safe direction; the alternative is
         dropping a real fault this run caused.
      3. A line carrying a DIFFERENT run's tag -- excluded. This is the one case that IS
         provably not about the run being assessed, so excluding it is not a guess.

    An unnameable assessed run (no run_id in the snapshot, or a non-string one) cannot prove
    case 3 for anything -- there is no name to compare a tag against -- so every line is kept.
    """
    own = run_id.strip() if isinstance(run_id, str) and run_id.strip() else None
    if own is None:
        return list(lines)
    kept = []
    for line in lines:
        match = _ANY_RUN_TAG_RE.search(line)
        if match is not None and match.group(1) != own:
            continue
        kept.append(line)
    return kept


def _dropped_row_tallies(lines: list[str]) -> list[str]:
    """Per-table tallies of PERMANENTLY DROPPED rows, read out of the shutdown log lines.

    A dropped row is one the store accepted, never wrote, and never will (see
    ranker/__init__.py's Store.dropped_rows). Nothing about it survives into the hub snapshot:
    the tally lives on the store object, which this module never holds, and by shutdown the
    buffer is empty and flush() has returned cleanly -- so the supervisor's own printed line is
    the only terminal evidence the loss happened at all. Reads it back rather than re-deriving
    it, matching how recovered provider faults are counted just below.

    Parses whatever lines it is handed and attributes nothing: the ring spans every run of the
    hub session, so callers reporting on ONE run must hand it `_lines_for_run(...)` output, not
    the raw ring.

    Returns each distinct tally text ("openers=1, labels=2") in the order seen, never raising:
    an absent line means the shutdown never reported loss, which is the healthy case.
    """
    out: list[str] = []
    for line in lines:
        match = _DROPPED_ROWS_RE.search(line)
        if not match:
            continue
        tally = _sanitize_inline(match.group(1))
        if tally and tally not in out:
            out.append(tally)
    return out


def _run_completion_assessment_md(hub_state, config_path: str) -> str:
    """Give a concise, deterministic answer to whether a run finished smoothly.

    A stopped phase is meaningful: supervisor.py reaches it only after workers exit and store
    flush succeeds. It must not be conflated with save_failed or wedged. Separately name known
    limitations so a durable save cannot disguise them as a perfect run.

    "The flush succeeded" is NOT "every row landed", and this used to read as if it were: a row
    BigQuery permanently rejects is dropped from the buffer so the rest can drain, the run
    continues on a warning, and shutdown then flushes an empty buffer successfully. Terminal
    phase stopped, no app error, no save error -- and the verdict came out COMPLETED CLEANLY on
    top of lost data. Dropped rows are therefore a limitation of their own below.
    """
    if hub_state is None:
        return "- **Outcome: NOT ASSESSED** — no hub snapshot is available."
    try:
        snap = hub_state.snapshot()
    except Exception as exc:  # noqa: BLE001 -- keep a failing hub diagnostic-safe
        return f"- **Outcome: NOT ASSESSED** — hub snapshot failed: {type(exc).__name__}."
    if not isinstance(snap, dict):
        return "- **Outcome: NOT ASSESSED** — hub snapshot is not a mapping."
    st = snap.get("status")
    if not isinstance(st, dict):
        return "- **Outcome: NOT ASSESSED** — no active or last run was recorded."

    phase = str(st.get("phase") or "unknown")
    apps = st.get("apps") if isinstance(st.get("apps"), dict) else {}
    app_rows = [row for row in apps.values() if isinstance(row, dict)]
    app_states = {str(row.get("state") or "unknown") for row in app_rows}
    app_errors = any(row.get("error") for row in app_rows)
    stopped = phase == "stopped" and not st.get("running") and not snap.get("running")
    run_id = st.get("run_id")
    run_note = (f" for run `{_sanitize_inline(run_id)}`"
                if isinstance(run_id, str) and run_id else "")

    if phase == "save_failed":
        return ("- **Outcome: SAVE FAILED** — the shutdown reached persistence, but flush "
                "failed; buffered data may be incomplete.")
    if phase == "wedged" or "wedged" in app_states:
        return ("- **Outcome: DEGRADED SHUTDOWN** — persistence was attempted, but at least one "
                "worker did not exit; do not treat the run as fully complete.")
    if st.get("stopping"):
        return (f"- **Outcome: STOPPING / INDETERMINATE**{run_note} — a stop was requested "
                f"while phase was {phase!r}; no durable completion verdict is available until "
                "shutdown and persistence finish.")
    if st.get("running") or snap.get("running") or phase != "stopped":
        return (f"- **Outcome: IN PROGRESS / INDETERMINATE**{run_note} — phase is {phase!r}; no durable "
                "completion verdict is available yet.")
    if snap.get("error") or app_errors or "error" in app_states:
        return ("- **Outcome: COMPLETED WITH ERRORS** — shutdown reached stopped, but the hub "
                "or an app recorded an error; saved records may be durable, but the run was not "
                "clean.")
    if not stopped:
        return ("- **Outcome: INDETERMINATE** — terminal fields disagree, so this report will "
                "not infer a durable save.")
    if not app_rows:
        # A terminal phase alone cannot prove that any worker actually reached it.  Treating an
        # empty/malformed app snapshot as a clean completion would make the headline claim
        # "all workers exited" when the snapshot names no workers at all.
        return ("- **Outcome: INDETERMINATE** — the terminal snapshot contains no app status, "
                "so worker completion cannot be verified.")
    terminal_app_states = {
        "error", "out_of_profiles", "rate_limited", "stopped", "wedged", "blocked",
    }
    nonterminal_app_states = sorted(app_states - terminal_app_states)
    if nonterminal_app_states:
        states = ", ".join(repr(state) for state in nonterminal_app_states)
        return ("- **Outcome: INDETERMINATE** — the global phase is stopped, but app status "
                f"still contains non-terminal state(s): {states}; the snapshot is internally "
                "inconsistent, so durable completion will not be inferred.")

    limitations: list[str] = []
    recent = recent_logs(_MAX_REPORT_LINES)
    # Loss leads: everything else in this list is a run that did less than it could have, while
    # this one is a run whose records are incomplete forever. Read from the shutdown log because
    # the store's tally has no route into the hub snapshot -- see _dropped_row_tallies.
    #
    # Scoped to THIS run's lines, never the whole ring: one hub process runs many runs through
    # one never-cleared _LOG_RING, so an unscoped read hands run B the permanent data loss run
    # A suffered -- see _lines_for_run.
    limitations.extend(
        f"the store PERMANENTLY DROPPED rows it never wrote ({tally}) — that data is LOST "
        "and this run's records are incomplete"
        for tally in _dropped_row_tallies(_lines_for_run(recent, run_id)))
    stop_kinds = {str(row.get("stop_kind")) for row in app_rows if row.get("stop_kind")}
    if "opener" in stop_kinds:
        limitations.append("an opener/provider condition stopped an app")
    elif stop_kinds:
        limitations.append("a safety or deck condition stopped an app")
    elif "rate_limited" in app_states:
        limitations.append("an app was rate limited")
    elif "blocked" in app_states:
        limitations.append("the deck was blocked")

    # _lines_not_attributed_to_another_run, NOT _lines_for_run -- the opposite of the
    # dropped-rows tally above, and deliberately so. Cascade prints in opener.py's
    # GeminiOpener.generate() CAN carry the same `Run {run_id}: ` tag supervisor.py's own lines
    # do, now that service.py:1003 and :1301 thread a run_id into every self.client.generate(...)
    # call -- but that tag is OPTIONAL on the client side (OpenerClient.generate's run_id kwarg
    # defaults to ""), and an older ring line, a direct `GeminiOpener` construction, or any
    # caller that never supplies a run_id still prints the identical phrase with NO tag at all.
    # A plain `_lines_for_run` filter keeps only lines carrying the ASSESSED run's own tag, so
    # it would silently drop every one of those untagged lines -- reporting ZERO faults for a
    # run that actually cascaded, which is the exact defect this counter exists to catch (run
    # 257bdd639ca5 cascaded off a 503, printed it in this report's own "Recent logs", and was
    # stamped COMPLETED CLEANLY anyway).
    #
    # So the rule is the one _lines_not_attributed_to_another_run implements: keep this run's
    # own tagged lines AND every untagged line, and exclude only a line explicitly tagged for a
    # DIFFERENT run (see that function's docstring for the three-way split). An earlier run's
    # UNTAGGED cascade line (an older build, or any ring predating this fix) still counts here
    # on that basis -- recoverable by reading the log; silently hiding THIS run's own fault is
    # not, so over-reporting remains the accepted trade, just narrower than a whole-ring scan.
    provider_faults = sum(
        bool(_RECOVERED_PROVIDER_FAILURE_RE.search(line))
        for line in _lines_not_attributed_to_another_run(recent, run_id))
    if provider_faults:
        # Not "transport failure(s)" any more: the matcher now catches every non-fatal cascade
        # branch (transport, 5xx, 429, retired-model 404), and naming one of them would be a
        # confidently wrong label on the other three -- the same mistake the old matcher made by
        # only ever finding transport failures in the first place.
        limitations.append(
            f"{provider_faults} provider failure(s) recovered by cascading to the next "
            "configured model appear in the recent process log (this run's own tagged cascade "
            "lines, plus any UNTAGGED one -- a cascade line explicitly tagged for a DIFFERENT "
            "run is excluded, never one merely lacking a tag)")
    facts = _completion_capture_facts(st, config_path)
    gaps = facts.get("coverage_gaps", 0)
    candidates = facts.get("coverage_candidates", 0)
    if isinstance(gaps, int) and gaps:
        total = f" of {candidates}" if isinstance(candidates, int) and candidates else ""
        stop_gaps = facts.get("coverage_stop_interrupted_gaps", 0)
        stop_candidates = facts.get("coverage_stop_interrupted_candidates", 0)
        if isinstance(stop_gaps, int) and stop_gaps > 0:
            stop_total = (f" of {stop_candidates}"
                          if isinstance(stop_candidates, int) and stop_candidates else "")
            limitations.append(
                "the requested Stop interrupted the latest still-photo dwell walk, leaving "
                f"{stop_gaps}{stop_total} photo candidate(s) uncovered")
            remaining_gaps = gaps - stop_gaps
            if remaining_gaps > 0:
                limitations.append(
                    f"still-photo coverage skipped {remaining_gaps} photo candidate(s)")
        else:
            limitations.append(f"still-photo coverage skipped {gaps}{total} photo candidate(s)")
    # ITEMS_UNAVAILABLE IS ITS OWN CHANNEL, not a coverage gap. The gap counters above read
    # `item_coverage`, which hinge.py sets to None exactly when the payload is None -- so the
    # WORSE outcome (no item index at all, nothing an opener could be aimed at) could never
    # reach them and the verdict came out COMPLETED CLEANLY over it. See
    # `_completion_capture_facts`.
    unavailable = facts.get("items_unavailable")
    kind = facts.get("items_unavailable_kind")
    # ...BUT "NOTHING ASKED FOR A NUMBERED LIST" IS NOT A LIMITATION OF THE RUN (second
    # fix-review pass, 2026-09-16). The wording below stopped this line CLAIMING the profile
    # was abandoned; it still DEGRADED the verdict. hinge.py's `_item_enumeration_blocker`
    # records `items_unavailable` for five conditions, and its first two -- `opener.enabled:
    # false`, and an app that cannot attach an opener at swipe time -- mean no consumer for a
    # numbered list existed in the first place. worker.py agrees and deliberately does not stop
    # for them (its "BUG 2" guard, the `not disabled` check in front of the `items_unavailable`
    # stop): it sends the bare like and carries on. So a run configured with openers off wrote
    # that sentence on EVERY capture and came out "COMPLETED SAFELY, WITH LIMITATIONS" over
    # profiles it handled exactly as configured -- noise, and the same "the verdict cannot tell
    # correct behaviour from a fault" failure this whole channel was added to fix. The driver
    # now says which it was, and only that one is skipped: the other three conditions mean the
    # run WANTED a numbered list and could not safely produce one, and they keep their
    # limitation.
    #
    # THE CONFIG HALF IS GONE (2026-09-17). This skip used to also fire for a row carrying no
    # kind at all, reading `opener.enabled: false` out of config.yaml AT REPORT TIME on the
    # premise that a kind-less row PREDATES the kind and is therefore a historic openers-off
    # capture -- an age marker. That premise is false for the current build:
    # `_item_enumeration_unavailable_kind` (hinge.py) returns "" for THREE of the blocker's five
    # LIVE conditions, not only for rows written before the kind existed, and every one of those
    # genuine refusals also reaches this function with `items_unavailable_kind=None`. Reading
    # the config at report time then muted them identically to a real openers-off row -- and
    # every `items_unavailable` capture row on disk under data/hinge_debug (10 of them) carries
    # kind None and is a genuine fault, none an openers-off policy row, so this was not a
    # theoretical risk. Using kind-falsiness as an age marker is the same "records must not
    # assert what they never knew" defect commit 64e5d6b6 fixed elsewhere. The skip now keys on
    # the kind alone: it is stamped by the driver AT CAPTURE TIME, so a current-build
    # openers-off row is still muted correctly, and the only cost is one spurious limitation
    # line on a HISTORIC openers-off row written before the kind existed -- over-reporting,
    # which every comment in this channel already names as the cheap direction.
    no_opener_consumer = kind == "no_opener_consumer"
    if isinstance(unavailable, str) and unavailable and not no_opener_consumer:
        profile = facts.get("items_unavailable_profile")
        who = (f" for profile `{_sanitize_inline(str(profile))}`"
               if isinstance(profile, str) and profile else "")
        kind_text = (f" (kind: `{_sanitize_inline(str(kind))}`)"
                     if isinstance(kind, str) and kind else "")
        detail = f"`{_compact_item_index_refusal_text(unavailable)}`{kind_text}"
        # WORDED FROM EVIDENCE, NEVER FROM AN ASSUMPTION. `items_unavailable_after_stop` is true
        # only when the driver's own walk rows inside that capture's window recorded a Stop
        # (cooperative "stop", or the mid-navigation "navigation_cancelled" token). The other
        # candidate evidence -- supervisor.py's `Run {id}: stop requested ...` print -- is
        # deliberately NOT consulted: its single raise site is the STARTUP abort, which by
        # construction runs before any worker and therefore before any capture row could exist,
        # so matching on it here would only ever be a false positive.
        #
        # AND THE CONSEQUENCE IS NAMED, NOT THE OUTCOME (fix-review 2026-09-16). Both sentences
        # ended "so that profile was abandoned", which is a claim about what the WORKER did next
        # and nothing on this row can see it. hinge.py writes `items_unavailable` for POLICY
        # reasons too, not only for refusals: `_item_enumeration_blocker`'s first two conditions
        # are `opener.enabled: false` and an app that cannot attach an opener at swipe time, and
        # in both of those worker.py deliberately does NOT stop -- its own "BUG 2" guard (the
        # `not disabled` check before the `items_unavailable` stop) sends the bare like and moves
        # on, so with openers switched off EVERY completed run would have been stamped WITH
        # LIMITATIONS over a profile it handled exactly as configured. THAT SHAPE NO LONGER GETS
        # THIS FAR -- the `no_opener_consumer` skip above, added the same day, drops it entirely
        # -- but the restraint below still earns its place, because what remains here is the
        # residue the skip cannot attribute: a historic row written before the kind existed,
        # whose run's config no longer says openers were off, reads identically to a genuine
        # refusal. Observe is the third case: there the human does the liking and this refusal
        # only makes the item check inconclusive.
        # What the row DOES establish is the targeting consequence, and it establishes it for all
        # three: with no numbered item payload there is nothing for an opener to be written about
        # or verified against -- hinge.py's navigate and verify paths raise `HingeTargetingError`
        # quoting this very sentence -- so say that much and let the quoted reason carry the rest.
        if facts.get("items_unavailable_after_stop"):
            limitations.append(
                f"the requested Stop left the latest capture{who} with no numbered items at "
                f"all, so no opener could be targeted at that profile: {detail}")
        else:
            limitations.append(
                f"the latest capture{who} produced no numbered items at all, so no opener "
                f"could be targeted at that profile: {detail}")
    if facts.get("capture_truncated"):
        limitations.append("the latest capture was truncated")

    if not limitations:
        return ("- **Outcome: COMPLETED CLEANLY** — all workers exited and the store flush "
                "succeeded before the terminal stopped snapshot; no captured app failure or "
                "coverage limitation was found.")
    return ("- **Outcome: COMPLETED SAFELY, WITH LIMITATIONS** — all workers exited and the "
            "store flush succeeded before the terminal stopped snapshot, but "
            + "; ".join(limitations) + ".")


def _compact_item_index_refusal_text(value: object) -> str:
    """Replace only an oversized item-index geometry wall with a clear pointer."""
    text = _sanitize_inline(str(value))
    if "item index" not in text.lower() or len(text) <= _ITEM_INDEX_REASON_INLINE_LIMIT:
        return text
    return (text[:_ITEM_INDEX_REASON_INLINE_LIMIT].rstrip() +
            "… (full geometry is in the item-index summary below)")


def _app_diagnostics_md(apps: dict) -> str:
    """Per-app stop_reason / error, one bullet each, rendered BELOW the run-status table rather
    than as extra table columns. Both fields are free text (see AppStatus.stop_reason/.error in
    status.py) sourced from provider messages and tracebacks — not guaranteed to be short,
    single-line, or free of `|`. Cramming that into a 6-column table would either wrap
    illegibly or, worse, a literal `|` inside the text would be indistinguishable from a column
    separator and silently corrupt the table's column count. A labelled bullet list has no such
    ambiguity and reads as clearly as the table itself.

    The stop reason bullet also carries AppStatus.stop_kind (status.py) — "opener" (an
    opener-side stop: capacity exhausted, or no usable opener for this profile), "deck_blocked"
    (worker.py's blocked-deck check: DatingAppDriver.blocked_reason() found something on screen
    standing between us and the deck, e.g. Hinge's out-of-free-likes Hinge+ paywall), or
    "targeting"
    (ops/OPENER-REDESIGN.md 5.6: the like could not be put on the item its opener was written
    about, so it was not sent — the phone is left exactly as the driver stopped it). The value
    is rendered verbatim rather than mapped, so a kind added later reaches the report without
    an edit here. Before stop_kind existed, all of those
    rendered as an identical "stop reason: <free text>" bullet with no way to tell which
    SUBSYSTEM decided to stop without reading the message and guessing — exactly the ambiguity
    this report's own STALL SUMMARY (see _stall_summary_md) was filed to remove one layer down,
    for the same incident. When stop_kind is None (a stop path that predates it, or a manual
    Stop click that never sets stop_reason at all) the bullet says so explicitly as
    "unlabelled" rather than silently dropping the bracket — omitting it here would just
    reintroduce that same "which subsystem?" guesswork one field over. Routed through
    _sanitize_inline like every other free-text field in this file, even though stop_kind is
    presently a fixed small vocabulary and not itself provider-sourced text.

    Returns "" (no heading, no bullets) when no app has anything to report — the common healthy
    run must render nothing extra here, not an empty section."""
    lines: list[str] = []
    for name, a in apps.items():
        reason = a.get("stop_reason")
        if reason:
            # "**{name}** stop reason:" is kept as an unbroken substring (kind appended AFTER
            # the value, not spliced in before the colon) so this stays the same stable anchor
            # older callers/tests already grep for; the new information is additive, not a
            # reformat of what was already there.
            kind = a.get("stop_kind")
            kind_label = f"`{_sanitize_inline(str(kind))}`" if kind else "unlabelled"
            lines.append(f"- ⚠️ **{name}** stop reason: `{_compact_item_index_refusal_text(reason)}` "
                         f"(kind: {kind_label})")
        err = a.get("error")
        if err:
            lines.append(f"- ⚠️ **{name}** error: `{_sanitize_inline(str(err))}`")
    if not lines:
        return ""
    return "\n".join(["", "**Stop reasons / errors:**", *lines])


def _hub_guidance_md(apps: dict) -> str:
    """Render the *current* per-app guidance the hub is showing, not only its coarse state.

    ``state=waiting`` is intentionally broad: it can mean a person is simply reading a card,
    that an advisory opener is still being generated, or that the safety gate withheld a
    suggestion altogether.  The latter was the important fact in the owner-filed ``bug?``
    report, but the old bug report rendered only the word ``waiting`` even though the hub had
    already displayed "no suggestion to type" and its reason.  That made an expected safety
    refusal look indistinguishable from a stalled worker.

    These fields are a snapshot of the hub state, not a new device read: say that explicitly
    rather than pretending the reporter's phone is being inspected while the markdown is
    assembled.  All text fields originate outside this module (model output or driver/provider
    messages), so they use the same one-line/backtick-safe sanitiser as stop reasons.  In
    particular, this remains diagnostic-only and never exposes a credential value.
    """
    lines: list[str] = []
    for name, app in apps.items():
        if not isinstance(app, dict):
            continue
        state = str(app.get("state") or "unknown")
        mode = str(app.get("mode") or "")
        prefix = f"- **{_sanitize_inline(str(name))}**"
        if mode == "training" and state == "waiting_approval":
            lines.append(f"{prefix}: hub says a typed target opener is ready; choose Like to "
                         "send it or Dislike to pass this profile. No decision has been "
                         "recorded yet.")

        # 2026-09-06: the opener_warning / opener_pending / opener_suggestion branches that used
        # to live here were the last reader of the Observe ADVISORY SUGGESTION flow, whose call
        # site was removed by commit ea6756e8 (2026-08-26) and whose service parameter and config
        # knobs were removed outright this date. Nothing has written any of opener_warning,
        # opener_pending, opener_suggestion, opener_item, opener_item_description or
        # opener_referenced into a hub app dict since that commit, so all three branches were
        # unreachable and their "advisory suggestion is currently shown" text could only ever
        # have described a feature that no longer exists. Dead diagnostics that read as live are
        # exactly what made this file's own "advisory-blind" explanation for the empty
        # opener_rejections table survive, and misdirect, for months.
    return "\n".join(lines)


def _status_md(hub_state) -> str:
    if hub_state is None:
        return "- (no hub — run via `python -m operation_love hub` for live run status)"
    snap = hub_state.snapshot()
    st = snap.get("status")
    lines = [f"- hub running: {snap.get('running')}"]
    if snap.get("error"):
        lines.append(f"- ⚠️ last run error: `{snap['error']}`")
    if not st:
        lines.append("- no active/last run")
        return "\n".join(lines)
    run_id = st.get("run_id")
    if isinstance(run_id, str) and run_id:
        uptime = st.get("uptime_s")
        uptime_note = (f" · status uptime: {format_duration(float(uptime))}"
                       if isinstance(uptime, (int, float)) and not isinstance(uptime, bool)
                       and math.isfinite(float(uptime)) and uptime >= 0 else "")
        lines.append(f"- status provenance: run `{_sanitize_inline(run_id)}`{uptime_note}")
    ready = "ready" if st["ranker_ready"] else "defer"
    cap = f" / ${st['budget_cap']:.2f}" if st.get("budget_cap") is not None else ""
    # These are intentionally separate ledgers.  ``labels`` is the whole ranker's loaded
    # dataset, while AppStatus.swipes_run is the count of actual preference decisions made in
    # THIS run.  CostTracker.calls/spend are provider/billing telemetry: a staged opener draft
    # or a Training checkpoint can legitimately incur those without a landed decision or label.
    # ``calls`` is historical naming, not an HTTP-attempt counter: it advances only when a model
    # result carries usage into CostTracker.record().  Gemini's internal 503/429/404/transport
    # fallback attempts are intentionally visible in Recent logs and excluded from this ledger.
    apps = st.get("apps") or {}
    decisions = sum(
        int(a.get("swipes_run", 0))
        for a in apps.values()
        if isinstance(a, dict) and isinstance(a.get("swipes_run", 0), int)
    )
    lines += [
        f"- phase: {st.get('phase')} · mode: {st.get('mode')}",
        f"- labels: {st['labels']} / {st['min_labels']} ({ready}) "
        "(ranker dataset total, not this run)",
        f"- preference decisions recorded this run: {decisions}",
        f"- provider / billing telemetry: {st.get('openers', 0)} accounted model result(s) "
        "with usage (HTTP fallback failures are logged separately and excluded) · "
        f"tracked spend: ${st['budget_spent']:.2f}{cap}",
    ]
    if decisions == 0:
        lines.append("- No pass/like preference decision was recorded. A generated but unacted "
                     "opener draft (including a Training checkpoint) or its provider cost is "
                     "not a profile, photo, label, or decision record.")
    # `stopping` (RunStatus.snapshot(), status.py) marks a stop that's been requested but hasn't
    # unwound yet -- distinct from `phase == "stopped"`, which only appears once it actually has.
    # Read with .get() rather than st["stopping"]: this field is landing in a concurrent change,
    # so an older/mismatched snapshot dict must still render every other line here instead of
    # KeyError-ing the whole section, and once it lands this needs no further change to pick it
    # up. 🔴 per the owner's GO/WAIT status-circle convention -- never a hand emoji.
    if st.get("stopping"):
        lines.append("- 🔴 stopping: stop requested, winding down (not halted yet)")
    lines += [
        "",
        "| app | mode | state | last | score | decisions |",
        "|---|---|---|---|---|---|",
    ]
    for name, a in apps.items():
        score = "" if a.get("last_score") is None else f"{a['last_score']:.2f}"
        lines.append(f"| {name} | {a.get('mode','')} | {a.get('state','')} "
                     f"| {a.get('last_decision') or '—'} | {score or '—'} | {a.get('swipes_run',0)} |")
    diag = _app_diagnostics_md(apps)
    if diag:
        lines.append(diag)
    guidance = _hub_guidance_md(apps)
    if guidance:
        lines.extend(["", "**Current hub guidance (snapshot, not a new phone read):**", guidance])
    return "\n".join(lines)


def _recent_openers_md(hub_state) -> str:
    """WHAT the opener said, and what the model was looking at when it wrote it -- the
    diagnostic this report was missing. The bug this collector improvement was filed against
    showed only `openers: 1` in `## Run status`: that count proves a call happened, but says
    nothing about what the model wrote or which item the comment attaches to.

    The second half of that sentence used to read "whether it was ANCHORED to the like screen",
    and doc 5.9's observe inversion retired that framing along with the anchor's last caller:
    AUTO, Training, and advisory modes now send numbered item crops and the model CHOOSES the
    item, so the useful fact
    is the request shape (`index_space`) rather than an anchor flag. Older entries still carry
    the flag and still render by it -- see the anchor_note branch below.

    DATA ROUTE: PRIMARY source is hub_state.recent_openers() -- HubState now captures a live
    reference to the running OpenerService (supervisor.run()'s on_opener_service callback,
    plumbed through hub/state.py's HubState.start()) and this reads its real ring buffer of
    the last several SUCCESSFUL generations (see opener/service.py's `recent_openers` deque
    and OpenerService.recent_openers_snapshot), each with a timestamp, model id, the request
    item space, the model's own `referenced` claim, which profile item it attaches to (`index`),
    and the opener text itself. That answers the question that broke diagnosis before in
    full: what did the model say, which item space did it use, and which model produced it --
    not just a guess from whatever suggestion happens to still be live
    on the hub right now.

    Dropped the old fallback (reading each app's CURRENT live suggestion off the
    AppStatus.opener_* set) rather than keeping it as a secondary line: those fields are
    published from the exact same successful generate() call that appends to OpenerService's
    ring buffer (worker.py's _ObserveSuggestion publishes the whole set together -- see
    status.cleared_opener_fields), so the live-status fields are always either identical to the
    newest ring-buffer entry for that app or already stale/cleared -- never something extra.
    Keeping both would just show the same generation twice. The one thing the live set carries
    that the ring buffer does not is doc 5.9's MISMATCH (opener_warning: the human opened a
    different item than the suggestion names), and that is a fact about the human's tap rather
    than about a generation, so it does not belong in this section either.

    Best-effort, returns markdown -- may raise (hub_state.recent_openers() is documented
    never to, but this function does not re-guard that promise); wrapped by _safe_section
    like every other section in this file, so a raise here degrades to a warning line, not a
    broken report."""
    if hub_state is None:
        return "- (no hub — run via `python -m operation_love hub` for live run status)"
    entries = hub_state.recent_openers()
    if not entries:
        return ("- (no committed opener records in the active/last run; an unacted staged "
                "opener draft is intentionally absent)")
    newest_first = list(reversed(entries))[:_RECENT_OPENERS_SHOWN]   # ring buffer is newest-LAST
    # Entries committed before session_mode was added can still be present in the Hub's frozen
    # last-run snapshot after a code reload.  The status snapshot belongs to the same active or
    # last run as the opener ring, so its per-app mode is a safe compatibility fallback.
    legacy_app_modes: dict[str, str] = {}
    snapshot = getattr(hub_state, "snapshot", None)
    if callable(snapshot):
        try:
            status = snapshot().get("status") or {}
            for app_name, app_status in (status.get("apps") or {}).items():
                if isinstance(app_status, dict) and app_status.get("mode") in {"training", "auto"}:
                    legacy_app_modes[str(app_name)] = str(app_status["mode"])
        except Exception:  # noqa: BLE001 -- optional compatibility context only
            pass
    lines = []
    for e in newest_first:
        if not isinstance(e, dict):
            continue
        ts = _sanitize_inline(str(e.get("ts") or "unknown time"))
        app_name = str(e.get("app") or "?")
        app = _sanitize_inline(app_name)
        model = _sanitize_inline(str(e.get("model") or "?"))
        index = e.get("index")
        # WHICH LIST that number counts (opener.INDEX_SPACE_*). A bare small int is
        # uninterpretable on its own -- "3" means a different thing depending on whether the
        # request numbered her scroll frames or her item crops, and a reader comparing two
        # reports has no other way to tell. Absent from every entry written before 2026-08-12,
        # which renders as no suffix rather than as a wrong one.
        index_space = e.get("index_space")
        index_space = index_space.strip() if isinstance(index_space, str) else ""
        referenced = e.get("referenced")
        referenced = referenced.strip() if isinstance(referenced, str) else ""
        opener = str(e.get("opener") or "")
        if len(opener) > _RECENT_OPENER_TEXT_CHARS:
            opener = opener[:_RECENT_OPENER_TEXT_CHARS] + "…"
        explicit_mode = e.get("session_mode")
        # Only "training"/"auto" are recognized here. "advisory" was a real value back when
        # this line read `mode_note = "advisory" if advisory else "auto"`; once that flag was
        # dropped in favour of the legacy_app_modes fallback above, "advisory" stopped being
        # something session_mode could ever hold. Confirmed by grep: OpenerService.maybe_opener
        # and .commit_opener (opener/service.py) are the only two writers of this dict's
        # session_mode key and both write only "auto"/"training" literals, and this whole
        # ring buffer (HubState.recent_openers / _completed_openers) is in-memory only for the
        # life of one hub process -- never serialized to or reloaded from a persisted run log
        # -- so no historical log can hand this function an "advisory" value either. An
        # unrecognized explicit_mode (missing key, or any other stray string) falls back to the
        # last-run status snapshot exactly like a pre-session_mode legacy entry would.
        if explicit_mode not in {"training", "auto"}:
            explicit_mode = legacy_app_modes.get(app_name, "auto")
        mode_note = explicit_mode
        # WHAT THE MODEL WAS LOOKING AT: request shape, not a retired live-sheet flag.
        if index_space == "model_items":
            anchor_note = "🟢 chose from numbered item crops"
        else:
            anchor_note = "🔴 no numbered item crops"
        about = f" · about: {_sanitize_inline(referenced)}" if referenced else ""
        space_note = f" ({_sanitize_inline(index_space)})" if index_space else ""
        # Compact era suffix -- absent (not guessed) when the entry predates `prompt_sha256`
        # (2026-09-06), same absent-field convention as `index_space` immediately above.
        prompt_note = _prompt_provenance_suffix(e.get("prompt_sha256"))
        # These fields are already retained in OpenerService's committed ring. They are
        # diagnostics only: the redundancy detector is an uncalibrated lower-bound monitor and
        # the entropy guard redraws at most once. Render only non-empty outcomes, compactly, so a
        # report can assess the monitor after the live-log ring rolls over without making every
        # ordinary opener row noisier or implying that any line was rejected.
        monitor_bits: list[str] = []
        raw_markers = e.get("redundancy_markers")
        markers = []
        if isinstance(raw_markers, (list, tuple)):
            for marker in raw_markers:
                if not isinstance(marker, str) or not marker.strip():
                    continue
                markers.append(_sanitize_inline(marker)[:_RECENT_OPENER_MONITOR_MARKER_CHARS])
        if markers:
            shown = markers[:_RECENT_OPENER_MONITOR_MARKERS_SHOWN]
            more = len(markers) - len(shown)
            marker_note = "; ".join(shown)
            if more:
                marker_note += f"; +{more} more"
            monitor_bits.append(f"redundancy: {marker_note}")
        collision = e.get("entropy_collision")
        if isinstance(collision, str) and collision.strip():
            collision_text = _sanitize_inline(collision)[:_RECENT_OPENER_MONITOR_MARKER_CHARS]
            outcome = "regenerated once" if e.get("entropy_regenerated") else "kept original"
            monitor_bits.append(f'opening collision "{collision_text}" ({outcome})')
        elif e.get("entropy_regenerated"):
            # Defensive compatibility for a malformed/older entry which retained the outcome but
            # not the colliding n-gram. Never invent a collision string.
            monitor_bits.append("opening regenerated once (collision text unavailable)")
        monitor_note = (
            "\n  _Monitor only; never a rejection:_ " + " · ".join(monitor_bits)
            if monitor_bits else ""
        )
        lines.append(
            f"- `{ts}` · **{app}** · model: `{model}` · {mode_note} · {anchor_note} · "
            f"index: {index}{space_note}{about}{prompt_note}\n"
            f"  > {_sanitize_inline(opener)}{monitor_note}"
        )
    if not lines:
        return ("- (no committed opener records in the active/last run; an unacted staged "
                "opener draft is intentionally absent)")
    return "\n".join(lines)


def _recent_opener_rejections_md(hub_state) -> str:
    """The other half of the opener paper trail: attempts the deterministic guards in
    opener.py's _parse REJECTED (scaffolding text, an emoji/undeliverable char, an over-long
    opener, bad JSON, ...), not just the successes _recent_openers_md above shows. Before
    this section existed, a rejected opener was printed to the console and then lost forever
    -- BigQuery only ever recorded SUCCEEDED openers (record_opener), so there was no way to
    answer "how often does any guard fire" or "is the scaffolding/sentence-cap detector too
    strict" from a bug report alone.

    DATA ROUTE: reads hub_state.recent_opener_rejections() -- an in-memory ring buffer on the
    live OpenerService (recent_rejections / recent_rejections_snapshot, see its __init__
    docstring), the same "no BigQuery round-trip needed" design as _recent_openers_md's own
    ring buffer. reason_code/raw_opener may be missing or None (see OpenerParseError's own
    docstring for exactly which guards leave raw_opener empty), which is rendered as a plain
    placeholder rather than the literal string "None".

    Best-effort, returns markdown -- may raise; wrapped by _safe_section like every other
    section in this file, so a raise here degrades to a warning line, not a broken report."""
    if hub_state is None:
        return "- (no hub — run via `python -m operation_love hub` for live run status)"
    entries = hub_state.recent_opener_rejections()
    if not entries:
        return "- (no committed opener rejections in the active/last run)"
    newest_first = list(reversed(entries))[:_RECENT_REJECTIONS_SHOWN]   # ring buffer is newest-LAST
    lines = []
    for e in newest_first:
        if not isinstance(e, dict):
            continue
        ts = _sanitize_inline(str(e.get("ts") or "unknown time"))
        app = _sanitize_inline(str(e.get("app") or "?"))
        model = _sanitize_inline(str(e.get("model") or "?"))
        attempt = e.get("attempt")
        reason_code = e.get("reason_code") or "(no reason code)"
        reason_code = _sanitize_inline(str(reason_code))
        raw_opener = e.get("raw_opener")
        raw_opener = str(raw_opener) if raw_opener is not None else "(no candidate text)"
        if len(raw_opener) > _RECENT_OPENER_TEXT_CHARS:
            raw_opener = raw_opener[:_RECENT_OPENER_TEXT_CHARS] + "…"
        # Same compact era suffix, and same absent-field convention, as _recent_openers_md.
        prompt_note = _prompt_provenance_suffix(e.get("prompt_sha256"))
        lines.append(
            f"- `{ts}` · **{app}** · model: `{model}` · attempt {attempt} · "
            f"reason: `{reason_code}`{prompt_note}\n"
            f"  > {_sanitize_inline(raw_opener)}"
        )
    if not lines:
        return "- (no opener rejections recorded this run, or no run is active)"
    return "\n".join(lines)


# `verify_sheet_item`/`verify_sheet_identity` rows carry the `before`/`after` screenshot
# filenames that are the whole diagnosis of a targeting stop (see _sheet_verification_md): one
# names the pre-tap card under the heart, the other the sheet the verdict was actually taken
# on. Both actions now log a `reason` on every outcome, not only the exception paths (see
# AndroidDriver._sheet_verification_evidence), which makes them newly eligible to match on
# `_action_reason_key` and collapse together below -- and `_render_run` keeps only
# ts/action/reason/repeated, dropping before/after along with everything else. Two adjacent
# verifications of the same sheet can easily share the same reason text (a retried tap, or the
# doc-5.9 observe-side mismatch guard re-checking the same card), so these two actions are
# excluded here, unconditionally, rather than trusting reason strings to stay distinct.
_NEVER_COLLAPSED_ACTIONS = frozenset({"verify_sheet_item", "verify_sheet_identity"})


def _action_reason_key(raw: str) -> tuple[str, str] | None:
    """(action, reason) for a raw actions.jsonl line, or None if it can't merge with a
    neighbour: invalid JSON, a record with no "reason" field at all (capture,
    observe_decision, observe_resync, locate_target_heart, like, ... — every action that isn't
    the observe_waiting heartbeat), or an action in `_NEVER_COLLAPSED_ACTIONS`. Only records
    that match on BOTH fields ever collapse together; returning None here is what keeps
    everything else exactly as raw, individual JSON lines. Deliberately swallows every parse
    failure — a malformed or older-format line must pass through untouched rather than raise,
    same contract as the rest of this best-effort collector."""
    try:
        rec = json.loads(raw)
        action = str(rec["action"])
        if action in _NEVER_COLLAPSED_ACTIONS:
            return None
        return (action, str(rec["reason"]))
    except Exception:  # noqa: BLE001
        return None


def _rec_ts(raw: str) -> str:
    try:
        return str(json.loads(raw).get("ts", "?"))
    except Exception:  # noqa: BLE001
        return "?"


def _render_run(run: list[str], key: tuple[str, str] | None) -> str:
    """A run of >=1 raw lines that all share the same (action, reason) -> one display line.
    A singleton (or a line that never had a key to begin with) renders as its own untouched raw
    JSON line; a genuine run of repeats collapses into one summary object carrying the count and
    the timestamp span, e.g. {"ts": "20:46:13-20:49:02", "action": "observe_waiting",
    "reason": "no_change", "repeated": 12} — real numbers from the audited run this fix is for."""
    if len(run) == 1 or key is None:
        return run[0]
    action, reason = key
    return json.dumps({"ts": f"{_rec_ts(run[0])}-{_rec_ts(run[-1])}",
                        "action": action, "reason": reason, "repeated": len(run)})


def _group_action_reason_runs(lines: list[str]) -> tuple[list[list[str]], list[tuple[str, str] | None]]:
    """Group FILE-ORDER-ADJACENT lines that share an `_action_reason_key` into runs. A line
    with no key (None) never merges with anything, even another None-keyed line right next to
    it — each such line stays its own run of 1, i.e. exactly today's raw output."""
    runs: list[list[str]] = []
    keys: list[tuple[str, str] | None] = []
    for raw in lines:
        key = _action_reason_key(raw)
        if key is not None and keys and keys[-1] == key:
            runs[-1].append(raw)
        else:
            runs.append([raw])
            keys.append(key)
    return runs, keys


def _collapse_action_tail(lines: list[str], limit: int) -> list[str]:
    """The actions.jsonl tail, collapsed: group the WHOLE file into (action, reason) runs
    first, THEN take the last `limit` runs — collapsing before windowing (rather than windowing
    first, as the old raw-line slice did) is what lets a fixed display budget reach back past a
    single oversized observe_waiting run instead of being entirely consumed by it.

    The first and last runs actually selected are never displayed as a single collapsed summary
    even if they qualify — their own boundary raw line is peeled back out (see the loop below)
    so the very edge of what's shown is always a literal record: the newest one in particular,
    so a live stall's current state (its own "after" screenshot filename) stays visible instead
    of being flattened into a "repeated: N" count that names no file at all.

    Never reorders, never drops a non-adjacent record, and a line that fails to parse (or has no
    action/reason key) simply can't join a run — see `_action_reason_key`."""
    if limit <= 0 or not lines:
        return []
    runs, keys = _group_action_reason_runs(lines)
    tail_runs, tail_keys = runs[-limit:], keys[-limit:]
    last_i = len(tail_runs) - 1
    out: list[str] = []
    for i, (run, key) in enumerate(zip(tail_runs, tail_keys, strict=True)):
        peel_first = i == 0
        peel_last = i == last_i
        if len(run) == 1 or key is None or not (peel_first or peel_last):
            out.append(_render_run(run, key))
            continue
        start = 1 if peel_first else 0
        end = len(run) - 1 if peel_last else len(run)
        mid = run[start:end]
        if peel_first:
            out.append(run[0])
        if mid:
            out.append(_render_run(mid, key))
        if peel_last:
            out.append(run[-1])
    return out


def _action_counts_line(lines: list[str]) -> str | None:
    """One-line action-type histogram for the latest run's WHOLE actions.jsonl (not just the
    displayed tail below it) — e.g. "capture 4 · observe_waiting 18 · observe_decision 2".
    That instantly answers the question the tail alone makes a developer eyeball-count for:
    was this run stalling (observe_waiting dominates), deciding normally (capture and
    observe_decision roughly track each other), or erroring out — without reading a single
    JSON line. Ordered by first appearance in the file, i.e. capture-then-wait-then-decide,
    the same per-profile order this driver's own comments already describe (see
    wait_for_decision's docstring in hinge.py). Returns None (render nothing) when there's
    nothing to count — same "no extra section when there's nothing to say" contract as
    `_app_diagnostics_md`. A line that fails to parse or has no "action" key is silently
    skipped, never raises."""
    counts: dict[str, int] = {}
    for raw in lines:
        try:
            action = str(json.loads(raw)["action"])
        except Exception:  # noqa: BLE001
            continue
        # Pre-dedupe drivers encoded a dwell frame's ordinal into its action label. New drivers
        # keep the ordinal in its existing structured field and use one stable label so identical
        # PNGs can dedupe. Normalize both spellings here: a restarted run may legitimately contain
        # old and new rows, and its histogram should remain one count rather than dozens of
        # position-shaped action types. Raw tail rows remain untouched for forensic compatibility.
        if re.fullmatch(r"still_photo_dwell_\d+", action):
            action = "still_photo_dwell_frame"
        elif re.fullmatch(r"still_photo_reattach_\d+", action):
            action = "still_photo_reattach_frame"
        counts[action] = counts.get(action, 0) + 1
    if not counts:
        return None
    return "action counts: " + " · ".join(f"{k} {v}" for k, v in counts.items())


# ── capture-split recovery summary ─────────────────────────────────────────
def _capture_split_summary_md(lines: list[str]) -> str:
    """Summarise mid-read deck advances and whether their recapture completed.

    ``capture_split`` is deliberately a non-error action: Hinge can advance while a profile is
    being read, and the driver drops the mixed frames rather than poisoning a label.  Until this
    summary, an incident report made the important follow-up question need a manual timeline
    reconstruction: did the worker recapture the new card, or did that capture fail the
    scroll-top gate and return a truncated, unnumbered profile?

    The action carries the old-card evidence screenshot in ``before`` and, in current runs, the
    foreign boundary-trigger frame in ``after``.  Older logs have only ``before``; report that
    honestly rather than requiring the new field.  For every split, locate the first later
    completed ``capture`` action in the raw file (not the display-collapsed tail), then state
    its profile/read outcome.  Invalid JSON or malformed fields are ignored or rendered
    conservatively: generating a bug report must never be stricter than the log.
    """
    records: list[tuple[int, dict]] = []
    for index, raw in enumerate(lines):
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 — best-effort diagnostic over a live JSONL file
            continue
        if isinstance(rec, dict):
            records.append((index, rec))
    splits = [(index, rec) for index, rec in records if rec.get("action") == "capture_split"]
    if not splits:
        return ""

    out: list[str] = []
    for index, split in splits[-_CAPTURE_SPLITS_SHOWN:]:
        ts = _sanitize_inline(str(split.get("ts") or "unknown time"))
        frames = split.get("captured_frames", split.get("photos"))
        frame_text = (f" after {frames} captured frame(s)" if isinstance(frames, int)
                      and not isinstance(frames, bool) else "")
        evidence: list[str] = []
        before = split.get("before")
        after = split.get("after")
        anchor = split.get("anchor")
        if before:
            evidence.append(f"source screenshot `{_sanitize_inline(str(before))}`")
        if anchor:
            anchor_text = f"identity-anchor screenshot `{_sanitize_inline(str(anchor))}`"
            anchor_index = split.get("identity_anchor_frame_index")
            if isinstance(anchor_index, int) and not isinstance(anchor_index, bool):
                anchor_text += f" (frame {anchor_index})"
            evidence.append(anchor_text)
        if after:
            evidence.append(f"boundary-trigger screenshot `{_sanitize_inline(str(after))}`")
        identity_dist = split.get("identity_dist")
        top_dist = split.get("top_dist")
        if isinstance(identity_dist, (int, float)) and not isinstance(identity_dist, bool):
            identity_evidence = f"identity distance {identity_dist:g}"
            if isinstance(top_dist, (int, float)) and not isinstance(top_dist, bool):
                identity_evidence += f", scroll-top distance {top_dist:g}"
            evidence.append(identity_evidence)
        if "identity_anchor_confirmed" in split:
            evidence.append("identity anchor " + (
                "confirmed" if split.get("identity_anchor_confirmed") else "still provisional"))
        if "content_match" in split:
            evidence.append("adjacent content " + (
                "aligned" if split.get("content_match") else "did not align"))
        evidence_text = "; ".join(evidence) if evidence else "no split evidence was saved"
        profile_name = split.get("profile_name")
        name_text = (f"; identity read `{_sanitize_inline(str(profile_name))}`"
                     if profile_name else "")

        recovery = next(
            (rec for later_index, rec in records
             if later_index > index and rec.get("action") == "capture"),
            None,
        )
        if recovery is None:
            recovery_text = "no later completed capture was recorded"
        else:
            parts: list[str] = []
            photos = recovery.get("photos")
            if isinstance(photos, int) and not isinstance(photos, bool):
                parts.append(f"{photos} photo(s)")
            if recovery.get("capture_truncated") is True:
                parts.append("capture truncated")
            elif recovery.get("capture_truncated") is False:
                parts.append("capture reached its natural end")
            items = recovery.get("items")
            if isinstance(items, int) and not isinstance(items, bool):
                item_context = recovery.get("item_context")
                context_suffix = (f", {item_context} context crop(s)"
                                  if isinstance(item_context, int) and not isinstance(item_context, bool)
                                  else "")
                parts.append(f"{items} numbered item(s){context_suffix}")
            unavailable = recovery.get("items_unavailable")
            # `items_unavailable` (enumeration could not produce a payload) and
            # `items_unnumbered` (enumeration finished and legitimately numbered nothing --
            # e.g. every card was a still-photo-discriminator video, ops/STILL-PHOTO-
            # DISCRIMINATOR.md 5d) are never both set (see Profile's own docstring). Report
            # them as the two different things they are: only one reads as a failure.
            unnumbered = recovery.get("items_unnumbered")
            if unavailable:
                parts.append(f"items unavailable: `{_compact_item_index_refusal_text(unavailable)}`")
            elif unnumbered:
                parts.append("enumeration completed but numbered nothing: "
                             f"`{_compact_item_index_refusal_text(unnumbered)}`")
            detail = "; ".join(parts) if parts else "no capture details were logged"
            recovery_text = f"later capture/recovery followed: {detail}"
        out.append(f"- `{ts}`: deck advanced mid-read{frame_text}; {evidence_text}{name_text}; "
                   f"{recovery_text}")
    return "\n".join(out)


# ── item-index refusal summary ─────────────────────────────────────────────
def _item_index_refusal_summary_md(lines: list[str]) -> str:
    """Render every item-index refusal in the latest run without making a developer decode its
    compact per-pair ledger by hand.

    ``hinge._item_index_refused`` records the frames that first broke the shared page space,
    every measured pair delta in ``steps_px``, and detail for every refused pair.  This reporter
    derives min/median/max only from numeric measured steps: ``null`` means "unknown", never a
    zero step.  Older or manually edited logs can omit any field; they still produce an honest
    reason-only line rather than making a bug report fail.
    """
    refusals: list[dict] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- a partial JSONL write must not hide a later refusal
            continue
        if isinstance(rec, dict) and rec.get("action") == "item_index_refused":
            refusals.append(rec)
    if not refusals:
        return ""

    out: list[str] = []
    for rec in refusals:
        pair = rec.get("failing_pair")
        if (isinstance(pair, list) and len(pair) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in pair)):
            pair_text = f"frames {pair[0]} and {pair[1]}"
        else:
            pair_text = "no specific failing pair recorded"

        steps = rec.get("steps_px")
        numbered = [(i, float(step)) for i, step in enumerate(steps)
                    if isinstance(step, (int, float)) and not isinstance(step, bool)] \
            if isinstance(steps, list) else []

        def _median(values: list[float]) -> float:
            ordered = sorted(values)
            middle = len(ordered) // 2
            return (ordered[middle] if len(ordered) % 2 else
                    (ordered[middle - 1] + ordered[middle]) / 2)

        # A capture that hits the card's bottom keeps swiping a page that cannot move: every
        # step from there to the end is either a ~0px measured step or a refused pair, never a
        # mid-capture spacing glitch.  Walk backwards from the very last step and grow that run
        # for as long as each measured step is near-zero relative to the median of whatever
        # still precedes the run (scale-free, so it works at any cadence). A refused pair
        # (``None``) never breaks the run -- it sits inside the same saturated stretch.
        run_indices: list[int] = []
        if isinstance(steps, list):
            i = len(steps) - 1
            while i >= 0:
                value = steps[i]
                if value is None:
                    run_indices.append(i)
                    i -= 1
                    continue
                if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
                    break
                prior_values = [v for idx, v in numbered if idx < i]
                if not prior_values or float(value) >= _median(prior_values) * 0.25:
                    break
                run_indices.append(i)
                i -= 1
        run_indices.reverse()
        run_set = set(run_indices)
        run_measured = [(idx, value) for idx, value in numbered if idx in run_set]

        trailing_saturation = None      # a single clamped final gesture (legacy phrasing)
        trailing_run = None             # a genuine multi-step trailing saturation run
        if len(run_measured) >= 2:
            # Two or more near-zero measured steps in the run is what tells this apart from an
            # ordinary final gesture clamping short: name the whole stretch as one fact instead
            # of the single last element, and keep every one of its steps out of the cadence.
            trailing_run = (run_indices[0], len(run_measured),
                             len(run_indices) - len(run_measured),
                             [value for _idx, value in run_measured])
            cadence = [(idx, value) for idx, value in numbered if idx not in run_set]
        elif len(run_measured) == 1:
            trailing_saturation = run_measured[0]
            cadence = [(idx, value) for idx, value in numbered
                       if idx != trailing_saturation[0]]
        else:
            cadence = numbered

        cadence_values = [step for _i, step in cadence]
        if cadence_values:
            median = _median(cadence_values)
            label = ("realised steps" if trailing_saturation is None and trailing_run is None
                      else "main realised cadence")
            step_text = (f"{label} ({len(cadence_values)} measured): "
                         f"min {min(cadence_values):g}px, median {median:g}px, "
                         f"max {max(cadence_values):g}px")
            if trailing_saturation is not None:
                step_text += (f"; trailing scroll saturation at step {trailing_saturation[0]} "
                              f"was {trailing_saturation[1]:g}px")
            if trailing_run is not None:
                start, measured_count, refused_count, run_values = trailing_run
                value_text = (f"{run_values[0]:g}px" if len(set(run_values)) == 1 else
                              f"{min(run_values):g}-{max(run_values):g}px")
                refused_clause = f", {refused_count} refused" if refused_count else ""
                step_text += (
                    f"; scroll saturated from step {start} to the end of the capture "
                    f"({measured_count} measured steps at {value_text}{refused_clause}): "
                    f"the profile's bottom was reached there and the remaining frames repeat "
                    f"the same page position")
            small = [(i, value) for i, value in cadence if value < median * 0.5]
            if small:
                step_text += ("; mid-run small-step anomalies" if len(small) > 1
                              else "; mid-run small-step anomaly")
                step_text += " at " + ", ".join(f"step {i}={value:g}px" for i, value in small[:4])
        else:
            step_text = "realised-step stats unavailable (no measured pair delta was logged)"

        evidence = []
        if rec.get("before"):
            evidence.append(f"before `{_sanitize_inline(str(rec['before']))}`")
        if rec.get("after"):
            evidence.append(f"after `{_sanitize_inline(str(rec['after']))}`")
        saved = rec.get("evidence_frames")
        if isinstance(saved, list):
            names = [f"`{_sanitize_inline(str(name))}`" for name in saved[:8] if name]
            if names:
                evidence.append("saved capture evidence " + ", ".join(names))
        sidecar = rec.get("evidence_sidecar")
        if sidecar:
            evidence.append(f"geometry sidecar `{_sanitize_inline(str(sidecar))}`")
        runtime = rec.get("item_index_runtime")
        if isinstance(runtime, dict):
            algorithm = runtime.get("algorithm_id")
            module_path = runtime.get("module_path")
            indexer_hash = runtime.get("indexer_code_sha256")
            splitter_hash = runtime.get("splitter_code_sha256")
            runtime_bits = []
            if algorithm:
                runtime_bits.append(f"algorithm `{_sanitize_inline(str(algorithm))}`")
            if module_path:
                runtime_bits.append(f"loaded module `{_sanitize_inline(str(module_path))}`")
            if indexer_hash:
                runtime_bits.append(
                    f"in-memory indexer `{_sanitize_inline(str(indexer_hash))[:12]}`")
            if splitter_hash:
                runtime_bits.append(
                    f"in-memory splitter `{_sanitize_inline(str(splitter_hash))[:12]}`")
            if runtime_bits:
                evidence.append("runtime provenance " + ", ".join(runtime_bits))
        evidence_text = "; ".join(evidence) if evidence else "no pair screenshots saved"
        reason = _sanitize_inline(str(rec.get("reason") or "no refusal reason logged"))
        regions = sorted(set(re.findall(r"page rows\s+(\d+\.\.\d+)", reason)))
        frame_numbers = []
        for match in re.finditer(r"\bframe(?:s)?\s+(\d+)(?:\s+and\s+(\d+))?", reason):
            frame_numbers.extend(value for value in match.groups() if value is not None)
        frames = sorted(set(frame_numbers), key=int)
        geometry = []
        if regions:
            geometry.append("distinct page regions " + ", ".join(regions[:8]))
        if frames:
            geometry.append("cited frames " + ", ".join(frames[:8]))
        compact_geometry = "; ".join(geometry) if geometry else "no structured geometry cited"
        # This is the one deliberate full rendering.  Other report sections/tail records use a
        # pointer so an incident's repeated prose cannot dominate the whole report.
        out.append(f"- {pair_text}: {compact_geometry}; {step_text}; {evidence_text}. "
                   f"Full refusal: `{reason}`")
    return "\n".join(out)


def _plain_int(value: object) -> int | None:
    """``value`` as an int when it really is one. ``bool`` is not: ``True`` is not row 1."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _row_pair(value: object) -> tuple[int, int] | None:
    """A sidecar ``[y0, y1]`` row pair, or None when either end is missing/not an int."""
    if not (isinstance(value, list) and len(value) == 2):
        return None
    y0, y1 = _plain_int(value[0]), _plain_int(value[1])
    return None if y0 is None or y1 is None or y1 < y0 else (y0, y1)


def _geometry_sidecar_frames(run: Path, name: object) -> list | None:
    """``all_frame_geometry`` out of a refusal's geometry sidecar, or None. Never raises.

    The filename arrives from an actions.jsonl record, so it gets the SAME treatment
    ``_shot_digest`` gives a screenshot name: a bare filename inside the run directory or
    nothing at all.  A record is diagnostic input, never authority to read an arbitrary path.
    """
    if not isinstance(name, str) or not name:
        return None
    candidate = Path(name)
    if candidate.name != name or candidate.suffix.lower() != ".json":
        return None
    try:
        path = run / name
        # A bare name alone does not confine a symlink: ``run / evidence.json`` can still
        # resolve outside the run.  Evidence filenames arrive from JSONL, so do not let one
        # turn this best-effort report into an arbitrary-file reader.
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size > _ITEM_INDEX_GEOMETRY_SIDECAR_BYTES):
            return None
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 -- absent, truncated or malformed evidence stays silent
        return None
    if not isinstance(payload, dict):
        return None
    frames = payload.get("all_frame_geometry")
    return frames if isinstance(frames, list) else None


def _named_frames(indices: list[int]) -> str:
    """``(frames 0, 6)`` for a short list, and nothing at all for a long one."""
    if not indices or len(indices) > _ITEM_INDEX_GEOMETRY_ROWS_SHOWN:
        return ""
    return (" (frame" + ("" if len(indices) == 1 else "s") + " "
            + ", ".join(str(i) for i in indices) + ")")


def _sidecar_geometry_lines(frames: list) -> list[str]:
    """The three derived facts about one refused capture's leading edge. See the caller."""
    leading: dict[int, tuple[int, int, str, bool]] = {}   # frame -> (y0, y1, top_kind, observed)
    offsets: dict[int, int] = {}                          # frame -> page offset
    first_run: dict[int, tuple[int, int, str]] = {}       # frame -> (y0, y1, run kind)
    islands: dict[tuple[int, int], list[tuple[int, int | None, object]]] = {}
    unanchored_rows_unusable = False
    # DOES THIS SIDECAR'S WRITER POSTDATE THE SCREEN-FIXED ISLAND CHECK? Keyed on the PRESENCE of
    # the hold-out evidence keys on any block, which is the only thing in the file that dates it:
    # hinge.py writes both on every block unconditionally, and they shipped in the same commit as
    # `segment.BLOCK_UNANCHORED` and `item_index._screen_fixed_islands`. Presence, never
    # truthiness -- `unanchored_reason` is legitimately None off an unanchored strip, and
    # `content_digest` is None on every ordinary block. Confirmed against all 7 sidecars on disk
    # 2026-09-16: 78d364c5527d (74 blocks) and 8fb11094ef4d (51 blocks) carry neither key on any
    # block and are genuinely pre-check; f78ca90856b4 carries them on 54 of 54.
    writer_records_screen_fixed_evidence = False

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        index = _plain_int(frame.get("local_frame_index"))
        if index is None:
            continue
        offset = _plain_int(frame.get("offset_px"))
        raw_blocks = frame.get("blocks")
        blocks = ([b for b in raw_blocks if isinstance(b, dict)]
                  if isinstance(raw_blocks, list) else [])
        for block in blocks:
            # Asked of EVERY block, before the unanchored filter below: an ordinary `selectable`
            # block carries the keys too, and a post-check capture whose segmentation happened to
            # place everything has no unanchored block at all -- which is precisely the case this
            # flag exists to describe.
            if any(key in block for key in _SIDECAR_SCREEN_FIXED_EVIDENCE_KEYS):
                writer_records_screen_fixed_evidence = True
            if block.get("kind") != _SIDECAR_UNANCHORED_BLOCK_KIND:
                continue
            rows = _row_pair(block.get("frame_rows"))
            if rows is None:
                unanchored_rows_unusable = True
                continue
            islands.setdefault(rows, []).append((index, offset, block.get("content_digest")))
        head = _row_pair(blocks[0].get("frame_rows")) if blocks else None
        if head is None:
            continue
        if offset is not None:
            offsets[index] = offset
        leading[index] = (head[0], head[1], str(blocks[0].get("top_kind") or "unrecorded"),
                          bool(blocks[0].get("top_observed")))
        raw_runs = frame.get("background_runs")
        runs = ([r for r in raw_runs if isinstance(r, dict)]
                if isinstance(raw_runs, list) else [])
        below = sorted(
            (rows for rows, kind in
             ((_row_pair(r.get("frame_rows")), str(r.get("kind") or "unrecorded")) for r in runs)
             if rows is not None and rows[0] > head[0]),
            key=lambda rows: rows[0])
        for r in runs:
            rows = _row_pair(r.get("frame_rows"))
            if below and rows == below[0]:
                first_run[index] = (rows[0], rows[1], str(r.get("kind") or "unrecorded"))
                break

    out: list[str] = []
    if not leading:
        return out
    total = len(leading)

    # (1) THE TOP EDGE, aggregated. The incident's tell: one row, every frame, never observed.
    tops = Counter((y0, kind, observed) for y0, _y1, kind, observed in leading.values())
    (row, kind, observed), count = tops.most_common(1)[0]
    line = (f"- leading block began at frame row {row} with "
            f"{'an observed' if observed else 'an UNOBSERVED'} top edge "
            f"({_sanitize_inline(kind)}) on {count} of {total} frames")
    others = sorted({y0 for y0, _y1, _k, _o in leading.values()} - {row})
    if others:
        shown = ", ".join(str(v) for v in others[:_ITEM_INDEX_GEOMETRY_ROWS_SHOWN])
        line += (f"; the remaining {total - count} began at {len(others)} other row(s) "
                 f"({shown}), so NO leading strip holds a fixed frame position in this capture")
    elif len(set(offsets.values())) >= 2:
        line += (f", while the page offset moved across {min(offsets.values())}.."
                 f"{max(offsets.values())}px — a frame row that does not move while the page "
                 f"does is what screen-pinned chrome looks like")
    out.append(line)

    # (2) THE RUN BELOW IT. `too_long` is not a gutter, so it never cut, so the strip above it
    # was merged downward -- and that merge is what put a block top above any real content.
    if first_run:
        starts = Counter(v[0] for v in first_run.values())
        # One start row on every frame that has one is a FIXED run and may be named as such. A
        # run that sits at a different row on nearly every frame is just page content scrolling
        # past, and reporting its modal row (2 of 26, on the second incident capture) would
        # dress noise up as a measurement -- so say what actually varies instead.
        group = ({i: v for i, v in first_run.items() if v[0] == starts.most_common(1)[0][0]}
                 if len(starts) == 1 else dict(first_run))
        kinds = Counter(v[2] for v in group.values())
        kind_bits = [
            f"`{_sanitize_inline(name)}` on {n}"
            + _named_frames(sorted(i for i, v in group.items() if v[2] == name))
            for name, n in kinds.most_common()]
        if len(starts) == 1:
            where = (f"began at frame row {starts.most_common(1)[0][0]} on "
                     f"{len(group)} of {total} frames")
        else:
            where = (f"sat at {len(starts)} different frame rows across {len(group)} of {total} "
                     f"frames, so it moves with the page rather than holding one position")
        line = (f"- the first background run below that top edge {where}, classified "
                + ", ".join(kind_bits))
        extents = Counter((v[0], v[1]) for v in group.values())
        if len(starts) == 1 and len(extents) > 1:
            line += ("; it spanned " + ", ".join(
                f"{y0}..{y1} on {n}" for (y0, y1), n in extents.most_common(
                    _ITEM_INDEX_GEOMETRY_ROWS_SHOWN)))
        absorbed = sorted(i for i, v in group.items() if v[0] < leading[i][1])
        if absorbed:
            line += (f"; on {len(absorbed)} of those {len(group)} frames the run lay INSIDE the "
                     f"leading block instead of bounding it, so it did not cut and the strip "
                     f"above it was merged into the content below")
        out.append(line)

    # (3) THE SCREEN-FIXED VERDICT, re-derived from the same three conditions item_index's
    # `_screen_fixed_islands` applies. An absent verdict is reported as absent, never as "no".
    if islands:
        for rows, sightings in sorted(islands.items())[:_ITEM_INDEX_GEOMETRY_ROWS_SHOWN]:
            height = rows[1] - rows[0]
            seen = sorted({o for _i, o, _d in sightings if o is not None})
            digests = {d for _i, _o, d in sightings}
            where = (f"- screen-fixed verdict for the unanchored strip at frame rows "
                     f"{rows[0]}..{rows[1]}"
                     + _named_frames(sorted({i for i, _o, _d in sightings})) + ": ")
            if len(seen) < 2:
                only = f"one page offset ({seen[0]})" if seen else "no recorded page offset"
                out.append(where + f"NOT proven — it was seen at {only}, so nothing here "
                                   "distinguishes screen-pinned chrome from page content and it "
                                   "is placed on the page")
            elif seen[-1] - seen[0] < height:
                out.append(where + f"NOT proven — it was seen across only {seen[-1] - seen[0]}px "
                                   f"of scroll, less than its own {height}px height, so the page "
                                   "rows it would have shown still overlap and one piece of page "
                                   "content could explain both sightings")
            elif any(not isinstance(d, str) or not d for d in digests):
                out.append(where + f"UNDECIDABLE FROM THIS FILE — identical frame rows at "
                                   f"{len(seen)} page offsets spanning {seen[-1] - seen[0]}px, "
                                   f"more than its own {height}px height, but this sidecar "
                                   "carries no per-strip content digest, so the pixel-identity "
                                   "leg of the proof cannot be re-checked here")
            elif len(digests) != 1:
                out.append(where + f"NOT proven — it showed {len(digests)} different pixel "
                                   f"contents across {len(seen)} offsets, so it is not a static "
                                   "element; an animating or live-updating header reads exactly "
                                   "like this and it is placed on the page")
            else:
                out.append(where + f"PROVEN screen-fixed — identical pixels at {len(seen)} page "
                                   f"offsets spanning {seen[-1] - seen[0]}px, more than its own "
                                   f"{height}px height; it has no page position at all and is "
                                   "held out of the index entirely")
    elif unanchored_rows_unusable:
        # The strips ARE recorded here, so this writer plainly has the check; only these
        # particular rows cannot be parsed back. Keep this ahead of both branches below so a
        # half-written sidecar is never described by either of their provenance claims.
        out.append(f"- screen-fixed verdict: unavailable — this capture's geometry records "
                   f"`{_SIDECAR_UNANCHORED_BLOCK_KIND}` strip(s) but none with usable frame "
                   "rows, so the verdict cannot be re-derived from this file. Nothing here "
                   "decides whether the leading block above is page content or chrome pinned "
                   "to the screen")
    elif writer_records_screen_fixed_evidence:
        # THE COMMON PATH, and until 2026-09-16 it was the one described as "predates ... (or
        # that check found nothing to place)" -- a provenance claim this very file refutes, with
        # the true cause demoted to a parenthetical. A block carrying the hold-out evidence keys
        # was written by a writer that already had the check, so there is no age question left:
        # the check ran and placed everything.
        out.append("- screen-fixed verdict: none to give — this capture's segmentation placed "
                   f"every block on the page and recorded no `{_SIDECAR_UNANCHORED_BLOCK_KIND}` "
                   "strip to decide on, so the leading block above is page content as far as "
                   "this check is concerned")
    else:
        # STILL LIVE, and keyed on something real: no block in the whole sidecar carries either
        # hold-out key, which is the shape of `data/hinge_debug/78d364c5527d` and
        # `data/hinge_debug/8fb11094ef4d`. Deliberately NOT keyed on `schema_version` or the
        # runtime hashes -- both already existed before the check shipped, so neither dates the
        # file. The module name is `item_index._screen_fixed_islands`, matching the caller's
        # docstring: segment.py supplies the `unanchored` block kind, item_index applies the
        # three-condition check.
        out.append("- screen-fixed verdict: unavailable — no block in this capture's geometry "
                   "carries the screen-fixed hold-out evidence "
                   f"(`{'`/`'.join(_SIDECAR_SCREEN_FIXED_EVIDENCE_KEYS)}`), so this sidecar was "
                   "written before item_index._screen_fixed_islands existed. Nothing here "
                   "decides whether the leading block above is page content or chrome pinned "
                   "to the screen")
    return out


def _item_index_geometry_md(lines: list[str], run: Path) -> str:
    """Open the geometry sidecar an item-index refusal only NAMES, and print its leading edge.

    ADDED 2026-08-28, the day a Hinge 10.1.0 capture hard-refused to build an item index and the
    generated bug report could not lead an operator to the cause.  Every fact needed was already
    on disk in `item_index_refused_<id>_evidence.json`, which the refusal section above cites by
    filename and never reads.  The cause was that 10.1.0 pins a per-profile header INSIDE the
    analysed content band: frame rows 300..516 do not move across a whole scroll, the 106px page
    background run under the header is not gutter-length so it never cut, and segment.py
    therefore merged 43 rows of non-scrolling chrome into the clipped top of the next card and
    reported a block top 149px above any real content.

    So print the three facts that name that shape, each derived from the file and each saying
    what it measured:

      1. where the leading block began and whether its top edge was ever OBSERVED, aggregated
         over every frame -- one constant row across a moving page is the whole signature;
      2. how the first background run below that top edge was classified, and whether it lay
         inside the leading block (absorbed, so it did not cut) or bounded it;
      3. the screen-fixed verdict for any `unanchored` strip, re-derived from the same three
         conditions `item_index._screen_fixed_islands` applies -- and, when it was not proven,
         WHICH condition failed, so the next step derives from its precondition instead of being
         a fixed sentence.  A sidecar written before that check existed says so rather than
         printing a false negative.

    PRIVACY: verified by enumerating every leaf value of both real incident sidecars
    (`data/hinge_debug/8fb11094ef4d`, `data/hinge_debug/78d364c5527d`) on 2026-08-28 -- the
    `all_frame_geometry` sub-tree this reads is ints, floats, bools and a closed vocabulary of
    segment.py kind tokens (`gutter`, `too_long`, `card_edge`, `clipped`, `partial`,
    `selectable`, `context`, `band_edge`, `card_corner`, `background_run`).  No profile pixels
    and no profile text.  Every token is still passed through `_sanitize_inline` because a
    sidecar is on-disk input, not something this module authored.

    Best-effort throughout: an absent, unreadable, oversized, truncated or malformed sidecar
    degrades to silence, and a filename that is not a bare `.json` inside the run directory is
    never opened at all.
    """
    named: list[object] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- a partial JSONL write must not hide a later refusal
            continue
        if (isinstance(rec, dict) and rec.get("action") == "item_index_refused"
                and rec.get("evidence_sidecar")):
            named.append(rec["evidence_sidecar"])

    out: list[str] = []
    for name in named[-_ITEM_INDEX_GEOMETRY_SIDECARS_SHOWN:]:
        frames = _geometry_sidecar_frames(run, name)
        if not frames:
            continue
        body = _sidecar_geometry_lines(frames)
        if not body:
            continue
        out.append(f"- `{_sanitize_inline(str(name))}` ({len(frames)} frames of geometry):")
        out.extend(f"  {line}" for line in body)
    return "\n".join(out)


def _dwell_navigation_refusal_summary_md(lines: list[str]) -> str:
    """Render a failed still-photo candidate walk separately from index construction.

    An ``item_index_refused`` record is also the terminal envelope used when the dwell walk loses
    its measured return anchor.  That does *not* imply a frame pair in the original enumeration
    index failed: the index may be complete and every one of its deltas measured.

    WHICH ROWS CARRY ``navigation_refusal`` IS A PROPERTY OF THE REFUSAL, NOT OF THE ROW'S AGE
    (corrected 2026-09-16).  This docstring used to say "new candidate rows carry
    ``navigation_refusal`` ... older rows retain only a code", and the renderer below acted on
    that premise by calling every dict-less row a "legacy/incomplete trace".  The premise is
    false for the CURRENT build: the two uncoded-exception handlers in
    ``_still_photo_dwell_candidate_walk`` / ``_still_photo_dwell_progressive_sweep`` record only
    ``reason=type(exc).__name__`` because those five classes carry no measurement at all, and the
    ``return_unverified`` rows carry no ``reason`` either.  Run f78ca90856b4 rendered a row the
    then-current build had written MINUTES earlier as "legacy".  Nothing on any row carries an
    age marker, so this renderer no longer infers one: it says what the refusal class can and
    cannot supply, and stays neutral when it cannot tell.
    """
    records: list[dict] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- a partial action row must not hide a later one
            continue
        if (isinstance(rec, dict)
                and rec.get("action") == "still_photo_dwell_walk_candidate"
                and rec.get("outcome") in {
                    "navigation_refused", "navigation_refused_returned",
                    "navigation_refused_return_unverified", "return_unverified",
                    # A below-entry refusal that had already spent a gesture. Its harmless
                    # sibling `unreachable_below_entry` is deliberately absent -- that one is a
                    # card stepped over, not a failure -- but this one ends the capture and
                    # stops the run, so leaving it out rendered nothing at all for the event
                    # the operator is reading the report to understand (2026-09-04).
                    "below_entry_after_gesture",
                    # A mid-navigation operator Stop (2026-09-16): the literal driver token
                    # "navigation_cancelled", held in `_DWELL_NAVIGATION_CANCELLED` so this set
                    # and the Stop-attribution fallback in
                    # `_capture_window_has_dwell_walk_stop` cannot drift apart. It reaches the
                    # same terminal envelope as its `navigation_refused` sibling and must be
                    # explained for the same reason, so the token belongs in this set even
                    # though it is NOT a fault -- omitting it renders nothing at all for the
                    # event, the exact regression this comment and the accepted-outcome
                    # assertion in tests/test_hinge_item_capture.py exist to prevent. It is
                    # rendered as a Stop below, never as a refusal.
                    _DWELL_NAVIGATION_CANCELLED,
                }):
            records.append(rec)
    if not records:
        return ""

    def number(value: object) -> int | float | None:
        return (value if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None)

    out: list[str] = []
    for rec in records[-_CAPTURE_SPLITS_SHOWN:]:
        heart = number(rec.get("heart_ordinal"))
        candidate = f"page heart {heart:g}" if heart is not None else "unknown page heart"
        telemetry = rec.get("navigation_refusal")
        if not isinstance(telemetry, dict):
            reason = rec.get("reason")
            code = _sanitize_inline(str(reason or "not recorded"))
            outcome_token = str(rec.get("outcome") or "unknown")
            outcome = _sanitize_inline(outcome_token)
            if outcome_token == _DWELL_NAVIGATION_CANCELLED:
                # This module's own norm, already stated for an aborted read in
                # `_item_manifest_md`: "A Stop is not a failure to diagnose, so this says so
                # plainly rather than borrowing the refusal wording." The dwell-navigation
                # renderer used to violate it, filing a Stop under `navigation_refused` and then
                # calling the row legacy.
                out.append(f"- {candidate}: dwell-navigation CANCELLED by the requested Stop "
                           f"(`{code}`); the walk was asked to end while moving to this card, so "
                           "there is no plan, measured climb, or anchor-return telemetry to show "
                           "and nothing here is a targeting or measurement fault")
                continue
            if (isinstance(reason, str)
                    and reason in _UNCODED_DWELL_NAVIGATION_REFUSAL_CLASSES):
                # Not an old row: these five classes are raised by the vision/should_stop layers
                # UNDER `navigate_to_item` and carry no position, plan or measurement of any
                # kind, so a current-build row naming one is complete exactly as written.
                out.append(f"- {candidate}: dwell-navigation {outcome} (`{code}`); this refusal "
                           "class carries no measurement, so no structured plan, measured climb "
                           "or anchor-return telemetry exists for it")
                continue
            # Everything else -- including `return_unverified`, which the driver writes with no
            # `reason` key at all. Say only what is true of the row in hand; do not guess at its
            # age, because no row records one.
            out.append(f"- {candidate}: dwell-navigation {outcome} (`{code}`); this row carries "
                       "no structured refusal telemetry, so no plan, measured climb or "
                       "anchor-return figures can be shown for it")
            continue

        code = _sanitize_inline(str(telemetry.get("code") or rec.get("reason") or "not recorded"))
        frame_index = number(telemetry.get("frame_index"))
        frame_note = f" at navigation frame {frame_index:g}" if frame_index is not None else ""
        planned = telemetry.get("planned") if isinstance(telemetry.get("planned"), dict) else {}
        achieved = telemetry.get("achieved") if isinstance(telemetry.get("achieved"), dict) else {}
        plan_bits: list[str] = []
        for key, label in (("step_px", "step"), ("bound_px", "bound"),
                           ("spacing_px", "spacing"), ("sized_against_px", "sized against")):
            value = number(planned.get(key))
            if value is not None:
                plan_bits.append(f"{label} {value:g}px")
        window = planned.get("window_px")
        if (isinstance(window, (list, tuple)) and len(window) == 2
                and number(window[0]) is not None and number(window[1]) is not None):
            plan_bits.append(f"window {number(window[0]):g}..{number(window[1]):g}px")
        basis = planned.get("basis")
        if isinstance(basis, str) and basis:
            plan_bits.append(f"basis `{_sanitize_inline(basis)}`")
        plan_text = "; ".join(plan_bits) if plan_bits else "planned step telemetry unavailable"

        achieved_bits: list[str] = []
        delta = number(achieved.get("delta_px", achieved.get("measurement_delta_px")))
        climb = number(achieved.get("climb_px"))
        overshoot = number(achieved.get("overshoot_px"))
        if delta is not None:
            achieved_bits.append(f"shift {delta:+g}px")
        if climb is not None:
            achieved_bits.append(f"climb {climb:g}px")
        if overshoot is not None:
            achieved_bits.append(f"overshoot {overshoot:g}px")
        status = achieved.get("status", achieved.get("measurement_status"))
        if isinstance(status, str) and status:
            achieved_bits.append(f"estimator `{_sanitize_inline(status)}`")
        measurement_reason = achieved.get("measurement_reason")
        if isinstance(measurement_reason, str) and measurement_reason:
            achieved_bits.append(f"reason `{_sanitize_inline(measurement_reason)}`")
        reverse = achieved.get("reverse")
        if isinstance(reverse, dict):
            reverse_status = reverse.get("status")
            reverse_reason = reverse.get("reason")
            reverse_delta = number(reverse.get("delta_px"))
            reverse_bits: list[str] = []
            if isinstance(reverse_status, str) and reverse_status:
                reverse_bits.append(_sanitize_inline(reverse_status))
            if reverse_delta is not None:
                reverse_bits.append(f"{reverse_delta:+g}px")
            if isinstance(reverse_reason, str) and reverse_reason:
                reverse_bits.append(_sanitize_inline(reverse_reason))
            if reverse_bits:
                achieved_bits.append("reverse estimator `" + "; ".join(reverse_bits) + "`")
        achieved_text = "; ".join(achieved_bits) if achieved_bits else "achieved-climb telemetry unavailable"

        return_outcome = _sanitize_inline(str(telemetry.get("return_outcome") or "not recorded"))
        restored = number(telemetry.get("restored_page_shift_px"))
        anchor_text = (f"anchor return `{return_outcome}`"
                       + (f" at {restored:+g}px" if restored is not None else ""))

        # `walk` is new telemetry (2026-08-27): a verified return can let the candidate walk
        # continue past a refusal instead of abandoning it outright, bounded by a small budget of
        # returned refusals.  Rows without it -- every historic run, and the telemetry-less
        # branches above that return before reaching here -- simply have no "walk" key, so this
        # stays a no-op and the rendered line is byte-identical to before the budget existed.
        walk_text = ""
        walk = telemetry.get("walk")
        if isinstance(walk, dict):
            walk_outcome = _sanitize_inline(str(walk.get("outcome") or "unknown"))
            spent = number(walk.get("returned_refusals_spent"))
            budget = number(walk.get("returned_refusal_budget"))
            budget_text = (f" (returned-refusal budget {spent:g}/{budget:g})"
                           if spent is not None and budget is not None else "")
            if walk_outcome == "continued":
                walk_text = f"; walk continued to the next candidate{budget_text}"
            elif walk_outcome == "abandoned_budget_spent":
                walk_text = f"; ⚠️ walk abandoned: returned-refusal budget spent{budget_text}"
            elif walk_outcome == "abandoned_return_unverified":
                walk_text = f"; ⚠️ walk abandoned: return unverified{budget_text}"
            else:
                walk_text = f"; walk outcome `{walk_outcome}`{budget_text}"

        pair_bits: list[str] = []
        before_name = rec.get("before")
        after_name = rec.get("after")
        if isinstance(before_name, str) and before_name:
            pair_bits.append(f"before `{_sanitize_inline(before_name)}`")
        if isinstance(after_name, str) and after_name:
            pair_bits.append(f"after `{_sanitize_inline(after_name)}`")
        pair_text = ("; failing pair " + " / ".join(pair_bits)) if pair_bits else ""

        out.append(f"- {candidate}{frame_note}: dwell-navigation refusal `{code}`; {plan_text}; "
                   f"{achieved_text}; {anchor_text}{walk_text}{pair_text}")
        # schema_version 3 (2026-09-04) carries whole `frameshift.trace_estimate` records under
        # achieved["forward"]/["reverse"] -- the exact shape `_shift_trace_lines_md` consumes, so
        # the per-strip bank prints here too. That bank is what tells "the strips disagreed" apart
        # from "the bank split into a page cluster and a video cluster over an autoplaying card";
        # the nine projected scalars above cannot. Gated on the version rather than on key
        # presence: a v2 row's `reverse` is the smaller scalar dict, and its line above stays
        # byte-identical.
        schema_version = number(telemetry.get("schema_version"))
        if schema_version is not None and schema_version >= 3:
            out.extend(_shift_trace_lines_md(achieved))
    return "\n".join(out)


def _shift_trace_lines_md(measurement: object) -> list[str]:
    """Indented sub-bullets saying what the SHIFT ESTIMATOR saw, one per direction.

    Renders `frameshift.trace_estimate` records (driver key ``measurement``). It exists because
    a return-chain refusal used to report only what the driver INTENDED -- attempts, requested
    step, leg-by-leg terminal shifts -- so "the phone did not move" and "the phone moved exactly
    the 630px asked for, but an autoplaying video left too few static strips to reach quorum"
    printed identically (found 2026-09-04, run 1d84909bf1bb; diagnosing it took the retained
    PNGs and an offline re-run of the estimator).

    The strip bank is rendered as the MATCHED and PINNED strips only, with weak/flat ones
    collapsed to counts: a weak strip's whole content is "this strip saw nothing", which the
    tally already says, while the matched offsets are what show a bank split in two -- the page
    at one value and a video's own internal motion at another. Bounded at
    `_SHIFT_TRACE_STRIPS_SHOWN` rendered strips per direction with an explicit "+N more", so a
    caller-varied strip count cannot grow this section without saying so.
    """
    if not isinstance(measurement, dict):
        return []
    out: list[str] = []
    error = measurement.get("error")
    if isinstance(error, str) and error:
        out.append(f"  - estimator could not look at all: `{_sanitize_inline(error)}`")
    for direction in ("forward", "reverse"):
        trace = measurement.get(direction)
        if not isinstance(trace, dict):
            continue
        status = trace.get("status")
        head = f"  - {direction} estimator `{_sanitize_inline(str(status))}`" if isinstance(
            status, str) and status else f"  - {direction} estimator"
        counts: list[str] = []
        for key, label in (("agreeing", "agreeing"), ("dissenting", "dissenting"),
                           ("eligible", "eligible")):
            value = trace.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                counts.append(f"{value} {label}")
        confidence = trace.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            counts.append(f"confidence {float(confidence):.2f}")
        window = trace.get("trust_window_px")
        if isinstance(window, int) and not isinstance(window, bool):
            counts.append(f"trust window {window}px")
        band = trace.get("band")
        if (isinstance(band, (list, tuple)) and len(band) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in band)):
            counts.append(f"band rows {band[0]}..{band[1]}")
        if counts:
            head += " (" + ", ".join(counts) + ")"
        reason = trace.get("reason")
        if isinstance(reason, str) and reason:
            head += f": `{_sanitize_inline(reason)}`"
        out.append(head)
        strips = trace.get("strips")
        if not isinstance(strips, list) or not strips:
            continue
        located: list[str] = []
        other: dict[str, int] = {}
        for strip in strips:
            if not (isinstance(strip, (list, tuple)) and len(strip) >= 5):
                continue
            y0, y1, state, delta, score = strip[0], strip[1], strip[2], strip[3], strip[4]
            state_text = state if isinstance(state, str) else "?"
            if not (isinstance(delta, int) and not isinstance(delta, bool)):
                other[state_text] = other.get(state_text, 0) + 1
                continue
            score_text = (f" ({float(score):.2f})"
                          if isinstance(score, (int, float))
                          and not isinstance(score, bool) else "")
            marker = "" if state_text == "matched" else f" [{_sanitize_inline(state_text)}]"
            located.append(f"{y0}-{y1} {delta:+d}px{score_text}{marker}")
        if not located and not other:
            continue
        shown = located[:_SHIFT_TRACE_STRIPS_SHOWN]
        strip_text = "; ".join(shown) if shown else "no strip located anything"
        if len(located) > len(shown):
            strip_text += f"; +{len(located) - len(shown)} more"
        tally = ", ".join(f"{n} {state}" for state, n in sorted(other.items()))
        if tally:
            strip_text += f"; plus {tally}"
        out.append(f"    located strips: {strip_text}")
    return out


def _dwell_return_chain_refusal_summary_md(lines: list[str]) -> str:
    """Render failed measured cleanup chains without pretending planned movement was measured.

    ``still_photo_dwell_return_chain`` is emitted by the common return helper, so it covers both
    a proved candidate's cleanup and a post-gesture navigation refusal's cleanup.  It intentionally
    has no heart ordinal: the helper does not own candidate selection, and joining it to a nearby
    candidate row by order would make a malformed/interleaved JSONL log identify the wrong card.
    Historic runs have no such row at all; they retain the legacy summary above unchanged.
    """
    records: list[dict] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- a partial row must not hide a later terminal trace
            continue
        if (isinstance(rec, dict)
                and rec.get("action") == "still_photo_dwell_return_chain"
                and rec.get("outcome") == "refused"):
            records.append(rec)
    if not records:
        return ""

    def number(value: object) -> int | float | None:
        return (value if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None)

    out: list[str] = []
    for rec in records[-_CAPTURE_SPLITS_SHOWN:]:
        reason = _sanitize_inline(str(rec.get("reason") or "not recorded"))
        attempts = number(rec.get("attempts"))
        initial = number(rec.get("initial_terminal_shift_px"))
        terminal = number(rec.get("terminal_shift_px"))
        bound = number(rec.get("drift_bound_px"))
        direct = number(rec.get("direct_shift_px"))
        before = rec.get("before") if isinstance(rec.get("before"), str) else None
        after = rec.get("after") if isinstance(rec.get("after"), str) else None

        details: list[str] = []
        if attempts is not None:
            details.append(f"after {attempts:g} return attempt(s)")
        if initial is not None:
            details.append(f"initial entry-relative shift {initial:+g}px")
        if terminal is not None:
            # On an unmeasurable leg this is the state BEFORE that gesture.  Calling it a final
            # position would quietly turn an unknown landing into a measurement.
            details.append(f"last known entry-relative shift {terminal:+g}px")
        if bound is not None:
            details.append(f"return gate <{bound:g}px")
        if direct is not None:
            details.append(f"direct entry-frame check {direct:+g}px")
        if before:
            details.append(f"evidence before `{_sanitize_inline(before)}`")
        if after:
            details.append(f"after `{_sanitize_inline(after)}`")
        detail_text = "; ".join(details) if details else "numeric return telemetry unavailable"
        out.append(f"- return chain refused (`{reason}`): {detail_text}")
        # The estimator's own account of the leg that refused, when the driver recorded one.
        # Historic runs carry no `measurement` key and render exactly as they did before.
        out.extend(_shift_trace_lines_md(rec.get("measurement")))
    return "\n".join(out)


_ITEM_INDEX_NOTE_ACTIONS = ("item_index_repaired", "item_index_notes",
                            "item_index_screen_fixed_notes")

# Text fingerprint of a `_screen_fixed_islands`-produced note (item_index.py's `where = (f"the
# leading strip at frame rows {rows[0]}..{rows[1]}, seen in frame(s) ..."` local, ~line 2096),
# used ONLY as a fallback for a record written before `ItemIndex.screen_fixed_notes` existed
# (2026-09-17). Verified unique across every other note producer in item_index.py: neither
# `_assemble`'s ("the heartless partial block at page rows ...", "the block at page rows ..."),
# `_split_repeated_near_gutter_merges`'s ("the complete sighting in frame ... bounded the lower
# card ..."), `_split_long_background_card_top_merges`'s, nor the video/layout repair prose
# ("frame N's pair with frame N+1: positioned mute-card track fixed ...") begins with this
# phrase. The
# nearest neighbour is the DEFENCE-IN-DEPTH string at item_index.py:~3695 ("the strip at frame
# rows ... was proven fixed to the screen and held out ..."), which is a `failures` entry, never
# a `notes` entry, and does not share this prefix ("the strip", not "the leading strip"). A
# record's `screen_fixed_notes` field -- present on every row emitted after this fix -- is always
# preferred when it exists; this text match is read-only compatibility for historic rows only.
_SCREEN_FIXED_NOTE_TEXT_PREFIX = "the leading strip at frame rows "


def _item_index_note_bearing_records(lines: list[str]) -> list[dict]:
    """Every screen-fixed-island / assembly record, from ANY note-bearing action, undivided.

    Cross-agent contract (2026-09-17, corrected same day TWICE). hinge.py's emit site
    (~line 8628, `_record_item_index_notes`) names a record "item_index_repaired" when its
    `repairs` list (mirrors `ItemIndex.repair_provenance` -- video-track/layout SHIFT repairs
    ONLY) is non-empty; otherwise "item_index_screen_fixed_notes" when at least one of its notes
    came from `_screen_fixed_islands`; otherwise the neutral "item_index_notes". `notes` is a
    SEPARATE field (`repair_notes + fold_notes`, item_index.py:~3666), a concatenation from FIVE
    unrelated producers (`_screen_fixed_islands`, `_assemble`, `_split_repeated_near_gutter_merges`,
    `_split_long_background_card_top_merges`, and the video/layout repair prose). `repairs` and
    `notes` are ORTHOGONAL: a record can carry either, both, or neither, and the presence of one
    says nothing about the other.

    The FIRST cut of this fix split records into a "repaired" bucket and an "unresolved strip"
    bucket keyed on `repairs` emptiness alone, and reused each bucket's `notes` as if EVERY note
    in it described that bucket's mechanism. That key was wrong in both directions on real
    on-disk data (run 5554d6fc51aa's only note is an ordinary `_assemble` success with no
    screen-fixed mechanism at all, filed under a heading asserting a failed placement; run
    0e4257f20e64's genuine screen-fixed note rode with an unrelated shift repair and hid under
    "conservative repairs"), and it double-printed a note shared between a repaired and an
    unrepaired record (run 6de69d385cc6).

    But a per-RECORD key can never be right, because `repairs` and note PROVENANCE are properties
    of individual notes, not of the record they happen to ride in together (exactly what run
    0e4257f20e64 demonstrates). This function therefore stops splitting records at all -- it
    returns every note-bearing record undivided -- and the three summaries below bucket PER NOTE:
    `_item_index_repair_summary_md` reads only `repairs`; `_item_index_screen_fixed_notes_summary_md`
    and `_item_index_construction_notes_summary_md` partition `notes` itself using
    `_item_index_note_is_screen_fixed`, each deduplicated globally across every record regardless
    of which action name wrote it or what else that same record carried.
    """
    return [rec for rec in _action_records(lines) if rec.get("action") in _ITEM_INDEX_NOTE_ACTIONS]


def _item_index_note_is_screen_fixed(rec: dict, note: str) -> bool:
    """Whether ONE note string in ONE record came from `_screen_fixed_islands`.

    Bucketing is PER NOTE, not per record (2026-09-17, corrected same day): a record can carry a
    shift repair AND a screen-fixed note at once (run 0e4257f20e64), and a record's OTHER notes
    (from the remaining unrelated producers `_fold_page`/`build_item_index` also concatenate into
    `notes`) must not be swept into the screen-fixed heading just because they share a record
    with one that IS screen-fixed, or vice versa.

    `screen_fixed_notes` (present on rows written after this fix) is authoritative -- hinge.py
    populates it as an exact-string subset of that SAME record's `notes` (see
    `AndroidDriver._record_item_index_notes`). A record without the field is a historic row that
    predates it; `_SCREEN_FIXED_NOTE_TEXT_PREFIX` recovers the same classification from the note
    text alone for those.
    """
    screen_fixed = rec.get("screen_fixed_notes")
    if isinstance(screen_fixed, list):
        return note in screen_fixed
    return note.startswith(_SCREEN_FIXED_NOTE_TEXT_PREFIX)


def _item_index_repair_summary_md(lines: list[str]) -> str:
    """Structured shift-repair geometry ONLY -- never the place a bare note lands.

    Renders `repairs` entries (mirrors `ItemIndex.repair_provenance`) from every note-bearing
    record that has any (see `_item_index_note_bearing_records`). `notes` is never consulted
    here: a note names no shift-repair mechanism this heading could honestly claim, whether or
    not the same record also carries one.
    """
    structured: list[str] = []
    for rec in _item_index_note_bearing_records(lines):
        # v12 writes the same field name as refusal records.  Keep the older
        # key as a read-only compatibility fallback so historic debug runs
        # remain useful in newly generated reports.
        runtime = rec.get("item_index_runtime", rec.get("runtime"))
        algorithm = runtime.get("algorithm_id") if isinstance(runtime, dict) else None
        repairs = rec.get("repairs") if isinstance(rec.get("repairs"), list) else ()
        for repair in repairs[:16]:
            if not isinstance(repair, dict):
                continue
            path = repair.get("path")
            source_pair = repair.get("source_pair")
            raw_shift = repair.get("raw") if isinstance(repair.get("raw"), dict) else {}
            effective = (repair.get("effective")
                         if isinstance(repair.get("effective"), dict) else {})
            if not isinstance(path, str) or not isinstance(source_pair, list) or len(source_pair) != 2:
                continue
            # Both shifts go through _delta_px_or_dash: a repair record exists precisely
            # BECAUSE the raw estimator refused, and a no_consensus refusal always carries
            # delta_px=None, so interpolating it printed "raw no_consensus Nonepx" -- a
            # refusal dressed up as a measurement of the value None (run 257bdd639ca5 printed
            # exactly that line).
            line = (
                f"{algorithm or 'unknown indexer'}: `{path}` source frames "
                f"{source_pair[0]}→{source_pair[1]}; raw "
                f"{raw_shift.get('status')} {_delta_px_or_dash(raw_shift.get('delta_px'))} → "
                f"effective {effective.get('status')} "
                f"{_delta_px_or_dash(effective.get('delta_px'))}")
            clean = _sanitize_inline(line)
            if clean and clean not in structured:
                structured.append(clean)
    if not structured:
        return ""
    shown = structured[:8]
    hidden = len(structured) - len(shown)
    suffix = f"; {hidden} more distinct repair record(s) in actions.jsonl" if hidden else ""
    return "\n".join(f"- {line}" for line in shown) + suffix


def _render_deduped_item_index_notes(notes: list[str]) -> str:
    """Shared tail for the two note summaries below: cap at 8, name how many more exist."""
    if not notes:
        return ""
    shown = notes[:8]
    hidden = len(notes) - len(shown)
    suffix = f"; {hidden} more distinct note(s) in actions.jsonl" if hidden else ""
    return "\n".join(f"- `{note}`" for note in shown) + suffix


def _item_index_screen_fixed_notes_summary_md(lines: list[str]) -> str:
    """Notes `_screen_fixed_islands` produced, deduplicated GLOBALLY across every record.

    Deliberately WORDED FROM WHAT THE CHECK CONCLUDED, never asserting a direction: the check's
    do-nothing branches PLACE a strip on the page (they decline to hold it OUT, which "could not
    place" said backwards), while its proof branch does the opposite and holds one out entirely.
    Both conclusions are screen-fixed-check notes; the prose of each note already says which one
    it is, so the heading names only the mechanism, not the outcome.
    """
    notes: list[str] = []
    for rec in _item_index_note_bearing_records(lines):
        for note in rec.get("notes", ()) if isinstance(rec.get("notes"), list) else ():
            if isinstance(note, str) and _item_index_note_is_screen_fixed(rec, note):
                clean = _sanitize_inline(note)
                if clean and clean not in notes:
                    notes.append(clean)
    return _render_deduped_item_index_notes(notes)


def _item_index_construction_notes_summary_md(lines: list[str]) -> str:
    """Every OTHER free-text note, deduplicated GLOBALLY across every note-bearing record.

    "Other" means: not classified screen-fixed by `_item_index_note_is_screen_fixed`. This still
    includes the video/layout repair prose ("frame N's pair with frame N+1: positioned mute-card
    track fixed ...") on a record whose `repairs` is also non-empty -- that prose and the
    structured
    `repairs` entry describe the SAME accepted repair in two formats, so it belongs here rather
    than nowhere, but it is exactly why this heading never claims "no shift repair applied" or
    any other single mechanism: the four remaining producers this bucket can hold
    (`_assemble`, `_split_repeated_near_gutter_merges`, `_split_long_background_card_top_merges`,
    and the repair prose) do not share one description in common beyond "not the screen-fixed
    check". Global dedup (one seen-set across every record, not per-bucket) is what fixes the
    double-print reproduced on run 6de69d385cc6: a note attached to both a repaired and an
    unrepaired record used to print once under each of two headings, reading as two distinct
    strips instead of one.
    """
    notes: list[str] = []
    for rec in _item_index_note_bearing_records(lines):
        for note in rec.get("notes", ()) if isinstance(rec.get("notes"), list) else ():
            if isinstance(note, str) and not _item_index_note_is_screen_fixed(rec, note):
                clean = _sanitize_inline(note)
                if clean and clean not in notes:
                    notes.append(clean)
    return _render_deduped_item_index_notes(notes)


def _manifest_capture(lines: list[str]) -> dict | None:
    """The ONE capture whose item manifest the summary section expands.

    The raw-tail compactor has to reach the same answer this section displays. It replaces a
    manifest with a pointer to the table "above", and a run that captured several profiles inside
    the tail window has several manifest-bearing records but only ever ONE expanded table -- so a
    pointer on any other record names a table that is not that capture's.
    """
    capture = None
    for rec in _action_records(lines):
        if (rec.get("action") == "capture"
                and isinstance(rec.get("item_manifest"), list) and rec.get("item_manifest")):
            capture = rec
    return capture


def _item_manifest_summary_md(lines: list[str]) -> str:
    """Explain how page crops became dense model item numbers.

    Counts alone hid the incident where an ordinary rectangular first photo was among ten
    context crops and the last square photo became the sole survivor named ``item 1``.  New
    capture records retain a bounded, non-image manifest; surface its page/heart/model mapping
    once and compact the duplicate copy in the raw tail below.
    """
    # Both action names END a profile read: `capture` writes whatever the index produced (an
    # empty manifest when it refused), and `capture_aborted` is the Stop path, which by design
    # writes no `capture` record at all.  Reading only the former is what let the 2026-08-15
    # report print a previous profile's item table directly under a card the run never finished
    # reading -- with no warning, because the abort was invisible to this scan.
    records = _action_records(lines)
    attempts = [rec for rec in records
                if rec.get("action") in ("capture", "capture_aborted")]
    capture = _manifest_capture(lines)
    if capture is None:
        return ""
    manifest = [row for row in capture["item_manifest"] if isinstance(row, dict)]
    if not manifest:
        return ""
    translation = capture.get("item_translation")
    out = []
    # A refusal capture deliberately writes ``item_manifest=[]``.  Keep the immediately prior
    # successful mapping useful for comparison, but never let its page numbers appear to describe
    # the current card.  The raw-tail compactor's "see ... above" pointer remains accurate because
    # this explicit provenance line lives in the same manifest section.
    latest_capture = attempts[-1] if attempts else None
    # Compared by VALUE, not identity: `_manifest_capture` parses the same lines independently,
    # so the record it returns is an equal dict rather than the same object as the one in
    # `attempts`. An identity test here reads every successful capture as a stale one.
    if latest_capture is not None and latest_capture != capture:
        aborted = latest_capture.get("action") == "capture_aborted"
        prior_bits = ["prior successful capture"]
        prior_ts = capture.get("ts")
        if prior_ts:
            prior_bits.append(f"at `{_sanitize_inline(str(prior_ts))}`")
        prior_profile = capture.get("profile_name")
        if prior_profile:
            prior_bits.append(f"for profile `{_sanitize_inline(str(prior_profile))}`")
        latest_bits = ["the read that followed it was abandoned" if aborted
                       else "the latest capture"]
        latest_ts = latest_capture.get("ts")
        if latest_ts:
            latest_bits.append(f"at `{_sanitize_inline(str(latest_ts))}`")
        latest_profile = latest_capture.get("profile_name")
        if latest_profile:
            latest_bits.append(f"for profile `{_sanitize_inline(str(latest_profile))}`")
        if aborted:
            # A Stop is not a failure to diagnose, so this says so plainly rather than borrowing
            # the refusal wording.  The frame count is the useful part: it says how far into the
            # read the Stop landed, which is also how far down the card was left scrolled.
            frames = latest_capture.get("frames")
            depth = (f" after {_sanitize_inline(str(frames))} frame(s)"
                     if isinstance(frames, int) else "")
            out.append("- ⚠️ " + " ".join(prior_bits) + " — the manifest below belongs to it; "
                       + " ".join(latest_bits) + f" on Stop{depth}, so it has no manifest of its "
                       "own and none of the numbers below describe it")
        else:
            latest_reason = latest_capture.get("items_unavailable")
            # An empty manifest means either state 2 (`items_unavailable`, enumeration failed)
            # or state 3 (`items_unnumbered`, enumeration finished and legitimately numbered
            # nothing) of Profile's three-state contract -- never both. Falling back to "no
            # items_unavailable reason was logged" when `items_unnumbered` was in fact the
            # live field would misreport a normal outcome as an unexplained failure.
            latest_unnumbered = latest_capture.get("items_unnumbered")
            if latest_reason:
                reason_text = f"; items unavailable: `{_compact_item_index_refusal_text(latest_reason)}`"
            elif latest_unnumbered:
                reason_text = ("; enumeration completed but numbered nothing: "
                               f"`{_compact_item_index_refusal_text(latest_unnumbered)}`")
            else:
                reason_text = "; no items_unavailable reason was logged"
            out.append("- ⚠️ " + " ".join(prior_bits) + " — the manifest below belongs to it; "
                       + " ".join(latest_bits) + " had no numbered manifest" + reason_text)
    if isinstance(translation, list):
        out.append("- model item → page heart translation: `"
                   + _sanitize_inline(json.dumps(translation)) + "`")
    # The terminal capture row carries the coverage counts, while a preceding candidate-walk
    # row records whether Stop interrupted that same capture.  Keep the scan bounded by the
    # preceding capture: an older cancelled profile must not re-label a later ordinary gap.
    capture_index = max(
        (index for index, rec in enumerate(records)
         if rec.get("action") == "capture" and rec == capture),
        default=-1,
    )
    coverage_interrupted_by_stop = (
        _capture_stop_interrupted_gap_count(
            capture.get("item_coverage"), records, capture_index) > 0
        if capture_index >= 0 else False)
    out.extend(_item_coverage_lines(
        capture, coverage_interrupted_by_stop=coverage_interrupted_by_stop))
    for row in manifest[:24]:
        number = row.get("model_item")
        kind = _sanitize_inline(str(row.get("kind") or "unknown"))
        label = f"model item {number}" if isinstance(number, int) and not isinstance(number, bool) \
            else f"unnumbered {kind}"
        bits = []
        heart = row.get("heart_ordinal")
        if isinstance(heart, int) and not isinstance(heart, bool):
            bits.append(f"page heart {heart}")
        source = row.get("source_frame_index")
        if isinstance(source, int) and not isinstance(source, bool):
            bits.append(f"source frame {source}")
        page_rows = row.get("page_rows")
        if (isinstance(page_rows, list) and len(page_rows) == 2
                and all(isinstance(value, int) and not isinstance(value, bool)
                        for value in page_rows)):
            bits.append(f"page rows {page_rows[0]}..{page_rows[1]}")
        crop_size = row.get("crop_size")
        if (isinstance(crop_size, list) and len(crop_size) == 2
                and all(isinstance(value, int) and not isinstance(value, bool)
                        for value in crop_size)):
            bits.append(f"crop {crop_size[0]}x{crop_size[1]}")
        digest = row.get("crop_sha256")
        if digest:
            bits.append(f"sha256 `{_sanitize_inline(str(digest))}`")
        evidence = row.get("selection_evidence")
        if isinstance(evidence, dict):
            classifier_id = evidence.get("classifier_id")
            classification = evidence.get("classification")
            classifier_bits = []
            if classifier_id:
                classifier_bits.append(_sanitize_inline(str(classifier_id)))
            if classification:
                classifier_bits.append(_sanitize_inline(str(classification)))
            for key, metric_label in (("colour_std", "std"),
                                      ("dominant_background", "background"),
                                      ("edge_density", "edges")):
                value = evidence.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    classifier_bits.append(f"{metric_label}={value}")
            if evidence.get("large_uniform_panel") is not None:
                classifier_bits.append(
                    f"panel={str(bool(evidence['large_uniform_panel'])).lower()}")
            if evidence.get("text_layout") is not None:
                classifier_bits.append(
                    f"text={str(bool(evidence['text_layout'])).lower()}")
            if classifier_bits:
                bits.append("classifier `" + ", ".join(classifier_bits) + "`")
        reason = row.get("reason")
        if reason:
            bits.append(f"reason `{_sanitize_inline(str(reason))}`")
        detail = "; ".join(bits) if bits else "no provenance detail logged"
        out.append(f"- {label}: {detail}")
    if len(manifest) > 24:
        out.append(f"- {len(manifest) - 24} more manifest row(s) retained in actions.jsonl")
    return "\n".join(out)


def _compact_debug_tail_line(raw: str, expanded: dict | None = None) -> str:
    """Keep raw JSON useful while avoiding another full copy of a long refusal wall.

    ``expanded`` is the one capture record whose manifest the section above actually printed
    (`_manifest_capture`).  Only that record may be replaced with a pointer to it; every other
    manifest-bearing capture in the tail is summarised in place, because its own table is NOT
    above.  Passing nothing keeps the record's manifest raw rather than guessing.
    """
    try:
        rec = json.loads(raw)
    except Exception:  # noqa: BLE001
        return raw
    if not isinstance(rec, dict):
        return raw
    if rec.get("action") == "item_index_refused" and rec.get("reason"):
        rec["reason"] = _compact_item_index_refusal_text(rec["reason"])
    elif rec.get("action") == "capture" and rec.get("items_unavailable"):
        rec["items_unavailable"] = _compact_item_index_refusal_text(rec["items_unavailable"])
    # `items_unnumbered` is the different, non-failure state-3 sentence (enumeration finished
    # and legitimately numbered nothing) -- never set alongside `items_unavailable` above, but
    # can grow just as long, so it gets the same compaction rather than being left as a wall.
    elif rec.get("action") == "capture" and rec.get("items_unnumbered"):
        rec["items_unnumbered"] = _compact_item_index_refusal_text(rec["items_unnumbered"])
    if rec.get("action") == "capture" and rec.get("item_manifest"):
        if expanded is not None and rec == expanded:
            rec["item_manifest"] = "see item-numbering manifest above"
        else:
            rows = len(rec["item_manifest"]) if isinstance(rec["item_manifest"], list) else 0
            rec["item_manifest"] = (
                f"{rows} manifest row(s) in actions.jsonl; the expanded table above belongs to a "
                f"different capture")
    elif rec.get("action") in _ITEM_INDEX_NOTE_ACTIONS and rec.get("notes"):
        # PER NOTE, not per record (2026-09-17, corrected same day TWICE): this record's own
        # notes can straddle both headings at once (run 0e4257f20e64 has both a screen-fixed
        # note and repair-prose construction notes on the SAME record), so name whichever
        # heading(s) this record's notes actually landed under rather than assuming one. A
        # record's separate `repairs` entries (if any) still land under "item-index conservative
        # repairs", but that is the untouched `repairs` field printed below, not this pointer.
        raw_notes = rec["notes"] if isinstance(rec["notes"], list) else []
        has_screen_fixed = any(isinstance(n, str) and _item_index_note_is_screen_fixed(rec, n)
                               for n in raw_notes)
        has_other = any(isinstance(n, str) and not _item_index_note_is_screen_fixed(rec, n)
                        for n in raw_notes)
        headings = []
        if has_screen_fixed:
            headings.append("item-index screen-fixed check notes")
        if has_other:
            headings.append("item-index construction notes")
        rec["notes"] = [f"see {' and '.join(headings) or 'item-index construction notes'} "
                        "summary above"]
    return json.dumps(rec)


# ── stall summary ──────────────────────────────────────────────────────────
# Filed after an observe-mode incident where an unrecognised Hinge+ paywall left
# _await_like_resolved polling like_sheet / like_sending until the operator stopped it.
# actions.jsonl recorded every poll faithfully, but diagnosing the hang from the raw report
# meant reading its tail by eye and noticing observe_waiting repeatedly firing with the same
# reason. The functions below turn that pattern into one summary line at the top of the
# debug-log section.
def _parse_action_ts(raw_ts: object) -> datetime | None:
    """Best-effort parse of an actions.jsonl "ts" field into a datetime for wall-clock
    arithmetic. Every action logger in hinge.py (_note_observe_waiting included) writes
    datetime.now().isoformat(timespec="seconds")-shaped strings. A bug report must
    never crash on a malformed, missing, or older-format timestamp (not a string at all, not
    ISO-parseable, the "?" placeholder _rec_ts substitutes elsewhere for a field that's plain
    absent), so every failure here returns None rather than raising. Every caller treats None
    as "duration unknown", never as zero — a missing timestamp is not the same claim as an
    instantaneous stall."""
    if not isinstance(raw_ts, str):
        return None
    try:
        return datetime.fromisoformat(raw_ts)
    except ValueError:
        return None


_OBSERVE_WAITING_TELEMETRY_ACTIONS = frozenset({
    # These rows are emitted by the suggestion/verification thread while the driver's existing
    # wait_for_decision call keeps polling the SAME card. They are not decision boundaries.
    "observe_release_hub_pre_tap_published",
    "observe_release_post_tap_item_verified",
})


def _is_capture_abort_timing(rec: dict) -> bool:
    """Whether this is an end-of-capture timing row that can follow a Stop abort.

    `_capture_current` writes `capture_aborted` before its ``finally`` block emits the
    iteration timing row.  Timing is diagnostic bookkeeping, not a new capture lifecycle
    event, so it must not hide the terminal abort from the report.  Keep this deliberately
    narrow: ordinary inputs and any future non-timing capture action still break the suffix.
    """
    action = rec.get("action")
    return (isinstance(action, str) and action.startswith("capture_")
            and "timing" in action)


def _terminal_capture_abort(records: list[dict]) -> dict | None:
    """Return a final Stop-aborted capture obscured only by trailing timing diagnostics."""
    for rec in reversed(records):
        if _is_capture_abort_timing(rec):
            continue
        return rec if rec.get("action") == "capture_aborted" else None
    return None


def _is_observe_waiting_telemetry(rec: dict) -> bool:
    """Whether `rec` is a known in-wait fact rather than a card/lifecycle boundary."""
    return rec.get("action") in _OBSERVE_WAITING_TELEMETRY_ACTIONS


def _observe_waiting_stretches(lines: list[str]) -> list[list[dict]]:
    """Split a run's actions.jsonl into observed-decision waits.

    `observe_waiting` rows form a stretch, with the two known release facts allowed between
    them because they are generated on a sibling thread during the same wait. Every other
    action -- including capture, decision, resync, a like anchor, malformed JSON, or a future
    unclassified action -- closes the stretch rather than risking a merge across cards.
    Returns parsed waiting rows only, since callers need their reason/timestamps rather than
    the interleaved telemetry itself.
    """
    stretches: list[list[dict]] = []
    current: list[dict] = []
    for raw in lines:
        rec = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                rec = parsed
        except Exception:  # noqa: BLE001
            rec = None
        if rec is not None and rec.get("action") == "observe_waiting":
            current.append(rec)
            continue
        if current and rec is not None and _is_observe_waiting_telemetry(rec):
            continue
        if current:
            stretches.append(current)
            current = []
    if current:
        stretches.append(current)
    return stretches


def _final_observe_wait(records: list[dict]) -> tuple[list[dict], int | None]:
    """The final logged wait and its first record index, allowing in-wait telemetry.

    This mirrors `_observe_waiting_stretches`; otherwise the stall summary could correctly
    count one wait while the latest-context narrative reports only its suffix.
    """
    wait_reversed: list[dict] = []
    first_index: int | None = None
    started = False
    for index in range(len(records) - 1, -1, -1):
        rec = records[index]
        if rec.get("action") == "observe_waiting":
            wait_reversed.append(rec)
            first_index = index
            started = True
            continue
        if started and _is_observe_waiting_telemetry(rec):
            continue
        break
    return list(reversed(wait_reversed)), first_index


def _stall_candidates(lines: list[str]) -> list[tuple[int, str, int, float | None]]:
    """(stretch index in file order, reason, record count, duration in seconds or None) for
    every reason that repeats (>=2 records) within a single observe_waiting stretch — one
    profile's wait, per `_observe_waiting_stretches`. The stretch index is what lets
    `_stall_summary_md` rank a still-unresolved wait above an old one that took just as long
    but eventually got a decision -- see that function's docstring for why raw duration alone
    is the WRONG primary sort key, measured against this exact incident.

    Records need not be strictly ADJACENT within the stretch to count together -- this is the
    one place this module deliberately does NOT reuse `_group_action_reason_runs`'s strict
    adjacency, and the reason is measured, not stylistic: the incident run's final stretch
    alternates `like_sheet` (composing) with `like_sending` (Hinge resolving the send) every
    ~30s while stuck on a SINGLE like, e.g.
        01:42:43 like_sheet, 01:43:13 like_sheet, 01:43:16 like_sending, 01:43:19 like_sheet,
        01:43:49 like_sheet, 01:44:20 like_sheet, 01:44:39 like_sending
    A strict-adjacency grouping (matching this file's OWN "collapsed record" display format —
    see _render_run's "repeated": N objects, lines ~521-573) would fragment that into four
    separate runs of 2/1/3/1 records and miss the actual story: Hinge was stuck resolving ONE
    like for 1m37s across five like_sheet polls, not several shorter unrelated stalls. Grouping
    by reason across the WHOLE stretch (rather than the whole file — bounding to the stretch is
    what stops an unrelated, already-resolved like from an earlier profile at 01:19:51 from
    being counted into the same total) reproduces that reading exactly: 5 records,
    01:42:43 -> 01:44:20, 97s.

    So this function's own record/duration arithmetic never touches the file's collapsed
    "repeated": N / "ts": "<start>-<end>" display format at all — it always counts and dates
    genuinely raw, uncollapsed per-poll records straight out of `_observe_waiting_stretches`.
    That is the correct way to "account for both collapsed and uncollapsed records" the task
    warns about: by construction, every record counted here IS an individual, uncollapsed
    actions.jsonl line, and duration/count fall straight out of len(group) and the first/last
    entries' own timestamps — there is no separate collapsed-record code path to reconcile
    against.

    Duration is (last occurrence's ts) - (first occurrence's ts); None if either timestamp is
    missing or fails to parse (see _parse_action_ts), or if the clock appears to run backwards
    (a malformed-timestamp case too, not a negative stall). A reason seen only once within a
    stretch is not a stall — nothing repeated — and is dropped here rather than reported with a
    meaningless 0s duration."""
    out: list[tuple[int, str, int, float | None]] = []
    for stretch_index, stretch in enumerate(_observe_waiting_stretches(lines)):
        by_reason: dict[str, list[dict]] = {}
        for rec in stretch:
            reason = rec.get("reason")
            if isinstance(reason, str):
                by_reason.setdefault(reason, []).append(rec)
        for reason, recs in by_reason.items():
            if len(recs) < 2:
                continue
            start = _parse_action_ts(recs[0].get("ts"))
            end = _parse_action_ts(recs[-1].get("ts"))
            duration = (end - start).total_seconds() if start is not None and end is not None else None
            if duration is not None and duration < 0:
                duration = None
            out.append((stretch_index, reason, len(recs), duration))
    return out


# Compact "1m37s" wall-clock durations, and the "unknown duration" a malformed/missing
# timestamp pair produces (see _stall_candidates). Shared with hinge.py's stuck-screen
# watchdog message so a stall reads identically live and in this report; it lives in
# typography.py rather than here because a driver must not import the bug reporter — see
# format_duration's own docstring for why that direction breaks.
_format_stall_duration = format_duration

_STALL_RECENCY_LABELS = ("most recent", "2nd most recent", "3rd most recent")


def _stall_summary_md(lines: list[str]) -> str:
    """Repeated observe waits: the top `_STALL_STRETCHES_SHOWN` same-reason waits in a
    run's actions.jsonl, most recent first — see _stall_candidates for exactly what counts as one.
    This is the bug-report's own self-improvement (_diagnostic_improvement_md), added after an
    unrecognised Hinge paywall produced repeated observe_waiting/like_sheet records. It turns
    an eyeball pattern-match over a log tail into one line at the top of the debug-log section.

    Ranked by STRETCH RECENCY FIRST (the most recent profile-wait's own repeats outrank every
    earlier one), THEN by duration/count within that -- deliberately NOT by raw duration alone
    across the whole file. Measured on that exact incident run, sorting by raw duration alone
    gets this backwards: an entirely benign 4m08s / 17-record `no_change` streak sits earlier
    in the same file (01:22:36-01:26:44, one profile's long — and successful — human read time,
    eventually followed by a normal decision) and would outrank the actual 1m37s / 5-record
    `like_sheet` stall that never resolved, burying the real incident under stale, already-
    resolved deliberation from ~16 minutes earlier in the SAME run (01:26:44 -> 01:42:43).
    Recency fixes this on a principle broader than this one run: every stretch except the run's
    last is, by construction, followed by something that resolved it (an observe_decision, or
    the next profile's capture) -- however long it took, it was not what was still stuck when
    the run stopped. Only the run's current/final stretch is possibly still open, so its own
    repeats are what a developer investigating "the run hung" actually needs first; older
    resolved waits are still shown (this is "top few", not "only the last one") as lower-ranked
    context, not hidden. Unknown-duration entries (malformed/missing timestamps) sort after
    every known duration within the same stretch, then by record count as the next-best signal.

    Returns "" (no heading, no bullets) when nothing repeated — the common healthy run, or a
    run that stalled but never on the SAME reason twice in a row, must render nothing extra
    here, same "quiet when healthy" contract as `_app_diagnostics_md` and
    `_action_counts_line`. Never raises: `_stall_candidates` / `_parse_action_ts` already
    swallow every parse failure, so there is nothing left here that can throw on malformed
    input."""
    candidates = _stall_candidates(lines)
    if not candidates:
        return ""
    candidates.sort(key=lambda c: (-c[0], c[3] is None, -(c[3] or 0.0), -c[2]))
    out = []
    for i, (_stretch_index, reason, count, duration) in enumerate(candidates[:_STALL_STRETCHES_SHOWN]):
        # Candidates are intentionally sorted by stretch recency, not global duration.  Do not
        # call this row "longest": a newer 31s wait can correctly precede an older 63s one.
        # The ordinal describes that actual order and remains meaningful if a report shows
        # several resolved waits alongside its still-open final wait.
        label = (_STALL_RECENCY_LABELS[i] if i < len(_STALL_RECENCY_LABELS)
                 else f"#{i + 1} most recent")
        record_word = "record" if count == 1 else "records"
        out.append(
            f"- {label} repeated observe wait: reason=`{_sanitize_inline(reason)}` for "
            f"{_format_stall_duration(duration)} ({count} {record_word})"
        )
    return "\n".join(out)


# A resync record written from hinge.py's PASS-path fork carries no `reason` key at all --
# `_dbg_action("observe_resync", base, **fields)` there (hinge.py ~6532) never sets one, unlike
# the two LIKE-path call sites below it. That is not missing data to paper over: the card
# genuinely changed and the identity+deck-ready proof held, but the human's own touch stream did
# not corroborate a decision, so worker.py records nothing rather than guess pass or like. Naming
# that fork explicitly -- rather than leaving its bullet's reason blank -- is the difference this
# whole section exists to make: an absent field must never render as absent TEXT.
_GESTURE_UNCORROBORATED_REASON = "gesture-uncorroborated"

# The complete reason vocabulary `observe_resync` emits today (there are exactly four call
# sites in hinge.py -- one of them, the plain gesture-uncorroborated pass, sets no `reason` key
# at all and is handled by the `_GESTURE_UNCORROBORATED_REASON` branch below instead; grep
# "observe_resync" in hinge.py to re-confirm this list is still exhaustive before trusting it).
# A reason not in this dict is not swallowed -- see _abandoned_card_summary_md -- it renders
# with its raw string and no explanation, so a reason added later reaches the report before
# anyone remembers to update this dict.
_RESYNC_REASON_EXPLANATIONS = {
    "like_candidate_without_observed_sheet": (
        "a bottom-only screen change was investigated as a possible like, but no Send Like "
        "sheet was ever observed and the identity anchor could not confirm the card was still "
        "the captured profile; the card was abandoned and recaptured with nothing recorded"),
    "like_send_identity_unproven": (
        "a real Send Like sheet was observed, but the deck that followed could not be "
        "positively proven to be a DIFFERENT profile, so no LIKE was recorded rather than risk "
        "filing it against the wrong person"),
    "like_send_identity_unavailable": (
        "same as like_send_identity_unproven, except this profile never revealed its sticky "
        "header during capture, so there was no identity anchor to prove anything with"),
    "pass_identity_name_unconfirmed": (
        "the pixel identity check read a new, different profile on two independent frames, but "
        "there was nothing to corroborate that a human caused it: observe_touch_watch is off "
        "for Hinge (Android withholds the touch input stream on this device, so gesture "
        "corroboration is unavailable by design -- see HINGE_SPEC's observe_touch_watch "
        "comment), and OCR could not positively read the same new name on both settled frames "
        "either; rather than guess pass, the driver recaptured the card with nothing recorded"),
    _GESTURE_UNCORROBORATED_REASON: (
        "the card genuinely changed, but the human's own touch stream did not corroborate a "
        "decision causing it -- the pass path, not a like path"),
}


def _abandoned_card_summary_md(lines: list[str]) -> str:
    """ABANDONED CARDS: every `observe_resync` record in a run's actions.jsonl, most recent
    first, capped at `_ABANDONED_CARDS_SHOWN`.

    Filed against the 2026-08-15 report "it moved on again without waiting for my like or
    dislike": the owner was only scrolling to read a profile, pressed nothing, and the run
    abandoned the card and recaptured it, recording nothing. worker.py already does the SAFE
    thing on a resync -- a returned None means "recapture, record nothing" (worker._observe_loop's `liked is None` branch) -- so
    no like/pass was mislabelled by this incident. But the event that would have EXPLAINED the
    complaint sat only inside the raw `actions.jsonl (tail)` code block, indistinguishable by eye
    from any other line, so a developer had to spot "observe_resync" themselves and then
    reverse-engineer what it meant before they could even tell the owner's complaint apart from a
    real bug. This section names the event and turns its `reason` into the plain-English claim it
    is actually making -- the same translation `_OBSERVE_WAIT_EXPLANATIONS` below already does
    for observe_waiting.

    Deliberately does NOT claim every resync is a bug. Two of the three named reasons
    (`like_send_identity_unproven` / `_unavailable`) are the identity anchor correctly refusing to
    file a LIKE it could not prove belonged to a different profile -- the conservative, INTENDED
    outcome, not a failure. This section only narrates what the log says happened; it is not a
    verdict on whether a given resync was the bug a report was filed about -- that judgement still
    needs the surrounding context (profile_name, the latest observe context above, screenshots).

    Most-recent-first, same rationale as `_stall_summary_md`'s recency-first ordering: a developer
    chasing "why did it just move on" needs the LATEST resync first, not one from three profiles
    ago in the same run. Unlike the stall summary there is no "still open" distinction to make --
    a resync is, by construction, already resolved (the card was recaptured) -- so plain recency
    is the whole ranking, with no secondary sort needed.

    Returns "" (no heading, no bullets) when there are none -- the common healthy run, where every
    card ended in a proven capture or decision, must render nothing extra here, same "quiet when
    healthy" contract as `_stall_summary_md` and `_app_diagnostics_md`. Never raises:
    `_action_records` already swallows JSON parse failures, and every field read below is
    defensively type/truthiness-checked before display, so a record missing several keys entirely
    -- as logs from before this provenance existed do -- renders with those fields omitted rather
    than throwing or printing a bare `None`."""
    records = [rec for rec in _action_records(lines) if rec.get("action") == "observe_resync"]
    if not records:
        return ""
    records.reverse()                                      # most recent logged resync first
    shown = records[:_ABANDONED_CARDS_SHOWN]
    out: list[str] = []
    if len(records) > len(shown):
        out.append(f"- {len(records)} card(s) abandoned without a decision (resync) logged in "
                   f"this run; showing the most recent {len(shown)}:")
    for rec in shown:
        reason = rec.get("reason")
        has_reason = isinstance(reason, str) and bool(reason)
        gesture = rec.get("gesture")

        bits: list[str] = []
        if has_reason:
            bits.append(f"reason=`{_sanitize_inline(reason)}`")
        profile_name = rec.get("profile_name")
        if profile_name:
            bits.append(f"profile_name=`{_sanitize_inline(str(profile_name))}`")
        sheet_seen = rec.get("sheet_seen")
        if isinstance(sheet_seen, bool):                   # False is a real, meaningful value --
            bits.append(f"sheet_seen=`{'true' if sheet_seen else 'false'}`")   # never drop it
        identity = rec.get("identity")
        if identity:
            bits.append(f"identity=`{_sanitize_inline(str(identity))}`")
        confirm_identity = rec.get("confirm_identity")
        if confirm_identity:
            bits.append(f"confirm_identity=`{_sanitize_inline(str(confirm_identity))}`")
        if gesture:
            bits.append(f"gesture=`{_sanitize_inline(str(gesture))}`")
        detail = ", ".join(bits) if bits else "no further detail was logged"

        if has_reason:
            explanation = _RESYNC_REASON_EXPLANATIONS.get(reason)
            tail = (f" — {explanation}." if explanation is not None
                    else " — unrecognised resync reason (no explanation on file for it yet).")
        elif gesture == "resync":
            tail = f" — {_RESYNC_REASON_EXPLANATIONS[_GESTURE_UNCORROBORATED_REASON]}."
        else:
            tail = " — no reason or gesture verdict was logged for this resync."

        out.append(f"- `{_record_time(rec)}`: {detail}{tail}")
    return "\n".join(out)


_OBSERVE_WAIT_EXPLANATIONS = {
    "no_change": "the frame has not visibly changed since this card became READY; no manual "
                 "pass/like has been proven",
    "same": "the screen changed within the same captured profile (for example, a manual "
            "scroll), not to a proven next card",
    "scroll": "the movement matched a manual scroll within the captured profile, not a "
              "pass/like decision",
    "not_deck_ready": "the screen changed, but a stable swipe deck has not yet been proven",
    "not_settled": "a possible next deck card was seen once but did not yet pass the settle "
                   "recheck",
    "like_candidate": "a bottom-only screen change looked like a possible like, but no Send "
                      "Like sheet was observed; no manual decision has been proven",
    "like_sheet": "the app's like/comment sheet is visibly open; it is waiting for Send Like "
                  "or dismissal",
    "like_sending": "the like sheet closed, but the app has not yet shown a stable next card",
}


def _action_records(lines: list[str]) -> list[dict]:
    """Best-effort parsed records in file order; malformed partial JSONL is ignored."""
    records: list[dict] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- a report must survive an in-progress append
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def _finite_nonnegative_action_seconds(rec: dict, key: str) -> float | None:
    """Return a trustworthy non-negative duration from an untrusted JSONL row."""
    value = rec.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _nonnegative_action_int(rec: dict, key: str) -> int | None:
    """Return a JSON integer only when it is safe to use as capture provenance."""
    value = rec.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _format_capture_timing_seconds(seconds: float) -> str:
    """Keep the report concise while preserving a useful sub-minute comparison."""
    return f"{seconds:.1f}s"


def _round_or_dash(value, places: int = 3) -> str:
    """A measured number at fixed precision, or an em dash when there is not one.

    Absent and zero are different facts here — a 0.000 distance is a perfect reproduction and a
    missing one means the comparison could not be made at all — so this never coerces None into
    a number, and never prints a bare `None` at a reader either.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    try:
        if not math.isfinite(value):
            return "—"
    except TypeError:  # noqa: PERF203 -- a non-float numeric that isfinite rejects
        return "—"
    return f"{value:.{places}f}"


def _delta_px_or_dash(value) -> str:
    """`440px` for a measured pixel shift, a bare em dash for a refusal that measured none.

    The unit belongs to the MEASUREMENT, not to the slot: an estimator that refused (every
    `no_consensus` record carries delta_px=None) did not measure "None pixels", it measured
    nothing, and "Nonepx" reads as the former. Whole pixels, not thousandths -- these are
    frame offsets, and a `.000` tail on one would only suggest a precision the estimator never
    claimed.

    Derives its dash from `_round_or_dash(None)` rather than repeating the glyph, so the two
    renderings of "there is no number here" cannot drift apart.
    """
    rendered = _round_or_dash(value, 0)
    return rendered if rendered == _round_or_dash(None) else f"{rendered}px"


def _item_coverage_lines(capture: dict, *, coverage_interrupted_by_stop: bool = False) -> list[str]:
    """How many photo cards the still-photo dwell actually reached, and what that cost.

    `item_coverage` has always been in the log and has never been interpreted, which on
    2026-08-27 hid the SECOND half of a stop. Six of that profile's nine cards were photo
    candidates; the bounded dwell walk reached one; the other five went unnumbered as a stated
    coverage gap rather than a judgement. A one-item payload is not a neutral outcome for doc
    5.6 — `verify_sheet_item` derives its accept bound from the distance to the NEAREST OTHER
    item, so with no other item there is no separation to halve and the weakest available
    ceiling applies instead. The refusal that followed was measured against that ceiling.

    So this is not a capture statistic. It is the reason the bound in the verification section
    below is the one it is, and it earns a ⚠️ whenever photo candidates were left unobserved.
    """
    coverage = capture.get("item_coverage")
    if not isinstance(coverage, dict):
        return []
    def _ordinals(key):
        value = coverage.get(key)
        return value if isinstance(value, list) else []
    candidates = _ordinals("photo_candidate_page_hearts")
    dwelled = _ordinals("dwell_covered_page_hearts")
    missed = _ordinals("no_dwell_coverage_page_hearts")
    numbered = _ordinals("numbered_page_hearts")
    if not candidates and not dwelled and not missed:
        return []
    limit = coverage.get("still_photo_dwell_candidate_limit")
    limit_text = f", candidate limit {limit}" if isinstance(limit, int) else ""
    line = (f"- still-photo coverage: {len(dwelled)} of {len(candidates)} photo candidate(s) "
            f"were dwelled{limit_text}; {len(numbered)} card(s) ended up numbered")
    out = [line]
    if missed:
        stop_cause = ("because the requested Stop interrupted this in-progress dwell walk; "
                      "they could not be" if coverage_interrupted_by_stop
                      else "and so they could not be")
        out.append(f"  - ⚠️ page heart(s) {_sanitize_inline(json.dumps(missed))} were photo "
                   f"candidates the dwell walk never reached {stop_cause} numbered — a coverage "
                   "gap, not a judgement about the cards")
    if len(numbered) == 1:
        out.append("  - ⚠️ exactly ONE numbered item, so a later like-sheet verification has no "
                   "neighbour to derive a separation bound from and falls back to the weakest "
                   "ceiling available. See the post-tap sheet verification section.")
    return out


def _verify_mismatch_shape(item: dict) -> str:
    """Which of the three shapes a `verify_mismatch` record is, in plain words.

    FILED AGAINST THE 2026-08-27 HALT: "intended model item 1, actual 1" read as nonsense
    because a reader could not tell "the sheet had the right card and the bound was too tight"
    apart from "the sheet had the wrong card" -- the two numbers alone look identical either
    way. `nearest` (the sheet's closest stored crop) and `item` (what the model intended) settle
    it, three ways:

      * `nearest == item` -- the CORRECT item is the nearest stored crop; its distance simply
        did not come under the bound. A possible FALSE REFUSAL: a measurement/calibration
        problem, not a targeting miss.
      * `nearest != item` -- a DIFFERENT item is on the sheet. A real targeting miss.
      * `nearest` is absent/None -- nothing in the payload could be measured against this sheet
        at all (see item_verify.verify_sheet_item: `nearest_index` is None only when no stored
        item had a usable distance).

    Tolerates missing/None/malformed `nearest` and `item` without raising -- this renders a
    stop, so explaining it must never itself become a new failure.
    """
    nearest = item.get("nearest")
    if nearest is None:
        return ("nothing in this payload could be measured against this sheet at all (no "
                "stored item had a usable distance)")
    intended = item.get("item")
    try:
        same = nearest == intended
    except Exception:  # noqa: BLE001 -- malformed logged values must not break the report
        same = False
    if same:
        return ("the CORRECT item IS the nearest stored crop, but its distance did not come "
                "under the bound -- a possible FALSE REFUSAL (a measurement/calibration "
                "problem, not a targeting miss)")
    return "a DIFFERENT item is on the sheet -- a real targeting miss"


def _sheet_verification_md(lines: list[str], run: Path) -> str:
    """The post-tap verification family, rendered as the arithmetic behind a targeting stop.

    FILED AGAINST THE 2026-08-27 REPORT, which carried the halt sentence and nothing to check it
    with. That report said "intended model item 1, actual 1 ... 10.283 grey levels, against a
    7.000 bound" and stopped there, so the two possibilities a reader has to separate --
    the sheet was showing the wrong card, or the measurement was wrong -- looked identical. They
    are separated by exactly three things, all of which the driver computes and now records:

      * WHICH BOUND REGIME produced the number. 7.000 is the inline one-item ceiling, held out on
        two renders; a half-nearest-neighbour bound on a nine-item payload is a far stronger
        piece of evidence. `reason` says which in words.
      * THE PER-ITEM TABLE, whose `why` names the source rows the inline reframe sweep chose. On
        that run it read "inline reframe rows 37..974 of a 974px crop" -- a full-height window at
        the search boundary, which said the geometry had found the right card and the units were
        wrong. That turned out to be the bug (`item_verify._decode_crop`).
      * THE GEOMETRY THE COMPARISON IS BOUND TO: the preview's own derivation and the composer
        rectangles every bound in the check is measured against.

    Renders the newest `verify_sheet_item` record, and the identity check before it when one is
    present, because a sheet can fail either. Silent when the newest verification matched: a run
    that verified needs no explanation, and this section is for the stop.

    ADDENDUM (2026-08-28), filed against the pillarbox halt: an `unreadable` outcome is a refusal
    to LOOK, not a comparison that failed, and this rendered the two the same way -- the outcome
    line printed `nearest stored item None; distance — against bound —`, three dashes standing
    where three numbers stand on every other verdict. A reader could reasonably take that for a
    comparison that ran and returned nothing, when in fact no card had been compared at all. An
    `unreadable` now says so in words and prints no measurement it does not have, and the reason
    line is labelled as the refusal it is rather than as a verdict.

    ADDENDUM (2026-08-28), filed against the same incident: "intended model item 1, actual 1"
    was still ambiguous even with the numbers above it, because a `verify_mismatch` can mean
    either of two very different things and rendered identically either way. Right after the
    outcome line this now says, in plain words, which one it was -- see `_verify_mismatch_shape`.
    It also names `before`/`after` by ROLE (the pre-tap card under the heart vs. the sheet the
    verdict was actually taken on) so a reader knows which screenshot to open for which question,
    routed through the same `_shot_digest` path guard as every other screenshot this module
    prints so nothing outside the run directory is ever named.
    """
    records = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- partial JSONL rows are skipped, never guessed at
            continue
        if isinstance(rec, dict) and rec.get("action") in (
                "verify_sheet_item", "verify_sheet_identity"):
            records.append(rec)
    item = next((r for r in reversed(records) if r.get("action") == "verify_sheet_item"), None)
    if item is None or item.get("outcome") == "verify_match":
        return ""
    outcome = item.get("outcome")
    if outcome == "unreadable":
        # A "COULD NOT LOOK" IS NOT A MEASUREMENT, AND IT USED TO BE PRINTED AS ONE. The line
        # below this one rendered `nearest stored item None; distance — against bound —` for a
        # refusal, which reads as a comparison that ran and came back empty -- three dashes where
        # three numbers go. [2026-08-28: the report of the pillarbox halt said exactly that, and
        # the numbers a reader needed were not missing, they had never been computed, because the
        # locator refused before any card was compared.] Nothing was compared; the refusal's own
        # geometry, rendered below, is the entire record.
        out = [f"- outcome: `unreadable` for model item {item.get('item')} — the sheet could not "
               "be LOOKED AT, so no card was compared and there is no distance or bound to read. "
               "The refusal below carries the measurement instead."]
    else:
        out = [f"- outcome: `{outcome}` for model item {item.get('item')}; "
               f"nearest stored item {item.get('nearest')}; "
               f"distance {_round_or_dash(item.get('distance'))} against bound "
               f"{_round_or_dash(item.get('bound'))}"]
    if outcome == "verify_mismatch":
        out.append(f"- shape: {_verify_mismatch_shape(item)}")
    before_name, after_name = item.get("before"), item.get("after")
    if before_name or after_name:
        shot_cache: dict[str, str | None] = {}
        unsafe = "not available (missing or unsafe filename)"
        before_display = (before_name if _shot_digest(run, before_name, shot_cache) is not None
                           else unsafe)
        after_display = (after_name if _shot_digest(run, after_name, shot_cache) is not None
                          else unsafe)
        out.append(f"- pre-tap card under the heart: `{before_display}`; sheet the verdict was "
                   f"taken on: `{after_display}` — open both before reading a number.")
    identity = next(
        (r for r in reversed(records) if r.get("action") == "verify_sheet_identity"), None)
    if identity is not None:
        # The identity check gates the item check, so a reader has to know it passed before the
        # item numbers mean anything: a foreign profile explains a mismatch by itself.
        out.append(f"- profile identity on the same sheet: `{identity.get('outcome')}` at "
                   f"{_round_or_dash(identity.get('distance'))} against "
                   f"{_round_or_dash(identity.get('bound'))}")
    reason = item.get("reason")
    if reason:
        out.append(
            f"- why the sheet could not be read, and the geometry it measured: {reason}"
            if outcome == "unreadable"
            else f"- verdict, including which bound regime this was: {reason}")
    preview = item.get("preview")
    if isinstance(preview, list) and len(preview) == 4:
        y0, y1, x0, x1 = preview
        try:
            out.append(f"- preview compared: rows {y0}..{y1}, columns {x0}..{x1} "
                       f"({int(x1) - int(x0)}x{int(y1) - int(y0)}px)")
        except (TypeError, ValueError):
            out.append(f"- preview compared: {preview}")
    if item.get("preview_reason"):
        out.append(f"- how that preview was arrived at: {item['preview_reason']}")
    composer = item.get("composer")
    if isinstance(composer, dict):
        out.append(f"- composer the comparison is bound to: layout `{composer.get('layout_id')}`; "
                   f"comment {composer.get('comment')}; send {composer.get('send')}")
    grid = item.get("grid")
    if isinstance(grid, list) and len(grid) == 2:
        out.append(f"- comparison grid: {grid[0]}x{grid[1]}")
    comparisons = item.get("comparisons")
    if isinstance(comparisons, list) and comparisons:
        out.append("- per-item comparison table (every numbered item, as the bound is a property "
                   "of the payload and not of this sheet):")
        for row in comparisons:
            if not isinstance(row, dict):
                continue
            out.append(
                f"  - item {row.get('item')}: distance {_round_or_dash(row.get('distance'))}, "
                f"bound {_round_or_dash(row.get('bound'))}, nearest other "
                f"{_round_or_dash(row.get('nearest_other'))}, window {row.get('window_px')}px of "
                f"a {row.get('crop_px')}px crop; {row.get('why')}")
        omitted = item.get("comparisons_omitted")
        if omitted:
            out.append(f"  - ⚠️ {omitted} further item(s) not listed (record is capped)")
    if len(comparisons or []) == 1:
        # The one-item regime is not a detail: it is what selects the weakest bound available,
        # and on the 2026-08-27 run it was the consequence of a dwell-coverage shortfall five
        # cards wide rather than of the profile only having one photo.
        out.append("- ⚠️ this payload carried ONE numbered item, so there was no neighbour to "
                   "derive a separation bound from and the weakest available ceiling applied. "
                   "Check the still-photo coverage line above for why the other photo cards "
                   "were never numbered.")
    return "\n".join(out)


def _latest_completed_capture_timing_md(lines: list[str]) -> str:
    """Summarise the newest completed capture only when its timing rows can be paired safely.

    A capture's read roll-up precedes its fold row, which precedes its ``capture`` record.  The
    log is append-only and may contain partial/corrupt rows, so never borrow a duration across a
    completed/aborted capture boundary or infer absent values from per-iteration diagnostics.
    """
    # Keep invalid/non-object raw rows as explicit sentinels here.  Elsewhere the report can
    # still make best-effort use of the records around a partial append, but timing provenance
    # cannot safely cross unknown data between this capture and its own timing rows.
    records: list[dict | None] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 -- partial JSONL is a timing-pairing boundary
            records.append(None)
        else:
            records.append(rec if isinstance(rec, dict) else None)
    capture_index = next((
        index for index in range(len(records) - 1, -1, -1)
        if records[index] is not None and records[index].get("action") == "capture"
    ), None)
    if capture_index is None:
        return ""

    capture = records[capture_index]
    assert capture is not None  # narrowed by the capture-index predicate above
    capture_photos = _nonnegative_action_int(capture, "photos")
    read_record: dict | None = None
    fold_record: dict | None = None
    read_index: int | None = None
    fold_index: int | None = None
    for index in range(capture_index - 1, -1, -1):
        rec = records[index]
        if rec is None:
            return ""
        action = rec.get("action")
        if action in {"capture", "capture_aborted"}:
            break
        if action == "capture_fold_timing" and fold_record is None:
            fold_record, fold_index = rec, index
        elif action == "capture_timing_summary" and read_record is None:
            read_record, read_index = rec, index

    # Matching the fold's photo count to the completed capture prevents a malformed/interleaved
    # diagnostic row from being attributed to the wrong profile.  The roll-up itself has no
    # photo count, so its documented position before that paired fold is the remaining evidence.
    if (read_record is None or fold_record is None or read_index is None or fold_index is None
            or not read_index < fold_index < capture_index
            or capture_photos is None
            or _nonnegative_action_int(fold_record, "photos") != capture_photos):
        return ""
    read_s = _finite_nonnegative_action_seconds(read_record, "iter_wall_s_total")
    fold_s = _finite_nonnegative_action_seconds(fold_record, "fold_wall_s")
    if read_s is None or fold_s is None:
        return ""
    total_s = read_s + fold_s
    if not math.isfinite(total_s):
        return ""

    # Optional per-candidate detail emitted by newer Hinge drivers.  Keep the established fold
    # total authoritative and accept a candidate row only when its own named pieces reconstruct
    # its wall clock; old logs simply have no rows and render byte-for-byte as before.  A deep
    # progressive sweep defers cleanup until all of its candidates finish, so its one terminal
    # row carries return time that intentionally belongs to no individual candidate wall clock.
    candidate_timings: list[tuple[float, float, float, float]] = []
    sweep_return_timings: list[float] = []
    for rec in records[read_index + 1:fold_index]:
        if rec is None:
            continue
        if rec.get("action") == "still_photo_dwell_progressive_sweep":
            if rec.get("outcome") in {"returned", "return_unverified"}:
                sweep_return_s = _finite_nonnegative_action_seconds(rec, "return_s")
                if sweep_return_s is not None:
                    sweep_return_timings.append(sweep_return_s)
            continue
        if rec.get("action") != "still_photo_dwell_walk_candidate_timing":
            continue
        candidate_wall_s = _finite_nonnegative_action_seconds(rec, "candidate_wall_s")
        navigation_s = _finite_nonnegative_action_seconds(rec, "navigation_s")
        proof_s = _finite_nonnegative_action_seconds(rec, "proof_s")
        return_s = _finite_nonnegative_action_seconds(rec, "return_s")
        other_s = _finite_nonnegative_action_seconds(rec, "unattributed_s")
        if (None in {candidate_wall_s, navigation_s, proof_s, return_s, other_s}
                or not math.isclose(
                    navigation_s + proof_s + return_s + other_s,
                    candidate_wall_s, abs_tol=0.000003)):
            continue
        candidate_timings.append((navigation_s, proof_s, return_s, other_s))

    details: list[str] = []
    profile_name = capture.get("profile_name")
    if isinstance(profile_name, str) and profile_name.strip():
        details.append(f"profile `{_sanitize_inline(profile_name)}`")
    details.extend((
        f"read {_format_capture_timing_seconds(read_s)}",
        f"fold {_format_capture_timing_seconds(fold_s)}",
    ))
    dwell_s = _finite_nonnegative_action_seconds(fold_record, "still_photo_dwell_s")
    if dwell_s is not None and dwell_s <= fold_s:
        dwell_detail = "still-photo safety checks " + _format_capture_timing_seconds(dwell_s)
        share_detail = ""
        if fold_s > 0:
            # Divide before scaling: valid large finite durations must not overflow merely
            # because the derived percentage is being rendered.
            share = (dwell_s / fold_s) * 100
            if math.isfinite(share):
                share_detail = f"{share:.1f}% of fold"
        breakdown = fold_record.get("still_photo_dwell_breakdown")
        passive_s = (_finite_nonnegative_action_seconds(breakdown, "passive_observation_s")
                     if isinstance(breakdown, dict) else None)
        remainder_s = (
            _finite_nonnegative_action_seconds(breakdown, "navigation_and_overhead_s")
            if isinstance(breakdown, dict) else None)
        # The split is optional, diagnostic detail.  Accept it only when it reconstructs the
        # compatibility total closely enough for the row's six-decimal JSON rounding; a torn or
        # hand-edited object must not make the report confidently misattribute time.
        if (passive_s is not None and remainder_s is not None
                and math.isclose(passive_s + remainder_s, dwell_s, abs_tol=0.000002)):
            split_detail = (
                "passive observation " + _format_capture_timing_seconds(passive_s)
                + "; navigation/overhead " + _format_capture_timing_seconds(remainder_s))
            share_detail = f"{share_detail}; {split_detail}" if share_detail else split_detail
        if candidate_timings:
            navigation_s = sum(row[0] for row in candidate_timings)
            proof_s = sum(row[1] for row in candidate_timings)
            # Exactly one terminal row is the only valid progressive-sweep shape.  Duplicate
            # terminal rows are a torn/interleaved diagnostic sequence, so omit their optional
            # detail instead of double-counting cleanup the authoritative fold total already has.
            sweep_return_s = (sweep_return_timings[0]
                              if len(sweep_return_timings) == 1 else 0.0)
            return_s = sum(row[2] for row in candidate_timings) + sweep_return_s
            other_s = sum(row[3] for row in candidate_timings)
            candidate_detail = (
                f"{len(candidate_timings)} candidate hop(s): navigation "
                f"{_format_capture_timing_seconds(navigation_s)}; proof "
                f"{_format_capture_timing_seconds(proof_s)}; return "
                f"{_format_capture_timing_seconds(return_s)}; other "
                f"{_format_capture_timing_seconds(other_s)}")
            share_detail = (f"{share_detail}; {candidate_detail}"
                            if share_detail else candidate_detail)
        if share_detail:
            dwell_detail += f" ({share_detail})"
        details.append(dwell_detail)
    details.append(f"total {_format_capture_timing_seconds(total_s)}")
    return "- " + "; ".join(details) + "."


def _record_time(rec: dict) -> str:
    return _sanitize_inline(str(rec.get("ts") or "unknown time"))


def _training_probe_bool(value: object) -> str:
    """Render a probe predicate without promoting a missing field to a verdict."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "not recorded"


_TRAINING_PROBE_RECIPES = frozenset({
    "identity_band_psm7_3x",
    "top_card_header_psm6_3x",
    "top_card_header_psm6_native",
    "top_card_header_fallback_psm6_3x",
    "top_card_header_fallback_psm6_native",
    "top_card_header_take_another_look_psm6_3x",
    "top_card_header_take_another_look_psm6_native",
})
_TRAINING_PROBE_VERDICTS = frozenset({"same", "new"})
_TRAINING_IDENTITY_STATES = frozenset({
    "matched", "mismatched", "identity_unknown", "same", "new", "top", "unknown",
})
_TRAINING_NAME_VERDICTS = frozenset({"same", "new", "exact_name_conflict_after_content"})
_DEBUG_SCREENSHOT_NAME_RE = re.compile(r"\A\d{5}_[A-Za-z0-9_]+\.png\Z")
_GENERATION_CONTEXT_TEXT_LIMIT = 2_000
_GENERATION_INDEX_SPACES = frozenset({"model_items", "profile_photos"})


def _training_probe_token(value: object, allowed: frozenset[str]) -> str:
    """Render only known machine tokens from an untrusted Training action row.

    OCR text, a candidate name, and even a future free-text parser result must never appear in
    this compact report.  A schema change should read as ``unrecognised`` until the report has
    an explicit redaction-aware rendering for it.
    """
    if value is None:
        return "not recorded"
    if isinstance(value, str) and value in allowed:
        return value
    return "unrecognised"


# The prompt-era registry (tools/backfill_prompt_eras.py), resolved the same way _git() resolves
# the repo root: relative to this module's own location, never a hardcoded absolute path, so the
# report works from any checkout. See _prompt_era_description's docstring for WHY this exists --
# a real bug report about a bad opener could not say which prompt RULES were on wire when it was
# generated, even though `prompt_sha256` was sitting on the very evidence row already rendered.
_PROMPT_ERAS_PATH = _REPO / "ops" / "prompt-eras.json"


def _prompt_era_description(prompt_sha256: object) -> str:
    """Resolve a `prompt_stamp()` digest (opener/opener.py) to a short, human-readable era
    description via ops/prompt-eras.json.

    Matches a shipped era first (`eras[].prompt_sha256`), then the uncommitted working-tree
    entry -- marked provisional/uncommitted here explicitly rather than trusting that file's own
    `label` wording to always say so, since a digest that only matches `working_tree` was never
    released and must never read like a shipped era on a skim. Renders only the era's rule COUNT
    plus its `label`: a full 37-rule dump belongs to the registry itself, not to one report line.
    An unknown digest says so plainly rather than guessing.

    Totally failure-tolerant, matching this file's whole contract (see _safe_section): a missing
    ops/prompt-eras.json, an unreadable one, or malformed JSON all degrade to a plain sentence.
    Never raises.
    """
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        return "no prompt digest recorded"
    try:
        raw = json.loads(_PROMPT_ERAS_PATH.read_text())
    except Exception:  # noqa: BLE001 -- missing file, permissions, malformed JSON, anything
        return "ops/prompt-eras.json could not be read"
    if not isinstance(raw, dict):
        return "ops/prompt-eras.json could not be read"
    eras = raw.get("eras")
    if isinstance(eras, list):
        for era in eras:
            if not isinstance(era, dict) or era.get("prompt_sha256") != prompt_sha256:
                continue
            label = era.get("label")
            label_text = (_sanitize_inline(label) if isinstance(label, str) and label
                         else "(no label)")
            rules = era.get("rules")
            count = len(rules) if isinstance(rules, list) else "?"
            return f"shipped era: {label_text} ({count} rule(s) on wire)"
    working_tree = raw.get("working_tree")
    if isinstance(working_tree, dict) and working_tree.get("prompt_sha256") == prompt_sha256:
        rules = working_tree.get("rules")
        count = len(rules) if isinstance(rules, list) else "?"
        return (f"UNCOMMITTED WORKING TREE — provisional, not a shipped era "
                f"({count} rule(s) on wire)")
    return "digest not in ops/prompt-eras.json"


def _current_prompt_stamp(config_path: str) -> str | None:
    """The `prompt_stamp()` digest for the prompt that is CHECKED OUT right now, so a report
    can say whether the opener under investigation was written under the exact rules a reader
    is about to look at, or a different, no-longer-live set -- the single most valuable line
    for a wording bug (see opener/opener.py's own "OFFLINE REPRODUCTION" note: load the config
    and call ``prompt_stamp(cfg.opener.style)``; this is that recipe).

    Lazy local imports matching this module's existing convention (e.g. `from . import config
    as cfg_mod` in `_targeting_readiness_md`), wrapped so any failure -- an unreadable config, an
    opener-module import error -- degrades to None ("could not determine") rather than raising.
    Callers must treat None as "unknown", never as "no prompt digest exists".
    """
    try:
        from . import config as cfg_mod
        from .opener.opener import prompt_stamp
        cfg = cfg_mod.load(config_path)
        return prompt_stamp(cfg.opener.style)
    except Exception:  # noqa: BLE001
        return None


def _prompt_provenance_lines(record_prompt_sha256: object,
                             current_prompt_sha256: str | None) -> list[str]:
    """Full-form prompt provenance for the pre-send evidence block: the raw digest, its
    resolved era (`_prompt_era_description`), and whether it is the SAME prompt checked out
    right now (`_current_prompt_stamp`) -- so a reader knows whether the rules they are about
    to read are the rules that actually produced this text.

    Renders nothing when the record carries no `prompt_sha256` at all (older rows predate the
    field) rather than guessing -- the same absent-field convention `_recent_openers_md` already
    applies to `index_space`.
    """
    if not isinstance(record_prompt_sha256, str) or not record_prompt_sha256:
        return []
    digest = _sanitize_inline(record_prompt_sha256)
    lines = [f"- prompt era: `{digest}` — {_prompt_era_description(record_prompt_sha256)}"]
    if not isinstance(current_prompt_sha256, str) or not current_prompt_sha256:
        lines.append("- prompt drift: could not determine the checked-out prompt "
                     "(config or opener module unavailable)")
    elif current_prompt_sha256 == record_prompt_sha256:
        lines.append("- prompt drift: same as the checked-out prompt "
                     "(these are the rules that generated this opener)")
    else:
        lines.append("- ⚠️ prompt drift: DIFFERENT from the checked-out prompt "
                     "(the rules on wire now are NOT the rules that generated this opener)")
    return lines


def _prompt_provenance_suffix(record_prompt_sha256: object) -> str:
    """Compact list-row suffix: an abbreviated digest plus its resolved era, or '' when the
    record carries no `prompt_sha256` -- same absent-field convention as `_prompt_provenance_
    lines` above. Deliberately never repeats the drift sentence: that belongs to the one full
    pre-send evidence block, not to every row of a list.
    """
    if not isinstance(record_prompt_sha256, str) or not record_prompt_sha256:
        return ""
    digest = _sanitize_inline(record_prompt_sha256)
    abbrev = digest[:12] + "…" if len(digest) > 12 else digest
    return f" · prompt: `{abbrev}` ({_prompt_era_description(record_prompt_sha256)})"


def _replay_corpus_pointer_md() -> str:
    """Point at the opener replay corpus (operation_love/opener/replay_corpus.py) as this
    opener's reproduction path, without pretending to correlate it to this exact record.

    A capture's on-disk id (`replay_id`) is a content hash of the request crops/name/context/
    truncation flag ALONE -- by that module's own PRIVACY section, a manifest carries no run id,
    no profile id, and no evidence id, so nothing an actions.jsonl row carries can be joined back
    to one specific capture directory. Rather than invent a fragile heuristic (nearest
    `captured_at` to this record's `ts`, matching item counts, ...), this names the corpus
    directory, how many captures are retained right now (a cheap directory listing, not a claim
    of correlation), and the tool that replays them.
    """
    try:
        from .opener import replay_corpus
        root = Path(replay_corpus.DEFAULT_CORPUS_DIR)
        if not root.is_absolute():
            root = Path.cwd() / root
        count = len(replay_corpus.list_replay_ids(root))
    except Exception:  # noqa: BLE001 -- this is a cheap pointer, never worth failing the report
        return ("- replay corpus: could not be listed -- reproduce a historical draft via "
                "`tools/opener_replay.py` against `data/opener_replay_corpus/` if a capture "
                "was retained")
    if count == 0:
        return (f"- replay corpus: no captures retained under `{root}` (disabled, or none have "
                f"landed yet) -- see `tools/opener_replay.py`")
    return (f"- replay corpus: {count} capture(s) retained under `{root}`, NOT correlated to "
            f"this specific record (the manifest stores no run/profile/evidence id, only a "
            f"content-derived `replay_id` -- see replay_corpus.py's PRIVACY section). Reproduce "
            f"via `tools/opener_replay.py`")


def _generation_context_md(record: dict, *,
                           current_prompt_sha256: str | None = None) -> list[str]:
    """Render the private structured fields retained with a staged Training draft, plus the
    prompt-era provenance for this evidence row (``_prompt_provenance_lines``) -- the latter
    renders for AUTO and Training alike, since `prompt_sha256` is stamped on every
    ``auto_opener_pre_send``/``auto_opener_resumed_send`` row regardless of session mode, even
    though the legacy model-private fields below remain Training-only in practice (AUTO never
    populates ``generation_context``, so ``context`` is simply empty for it and only the
    provenance lines render).

    ``actions.jsonl`` is untrusted report input, even though the real driver writes these
    values.  Accept only a flat mapping and bounded strings; model prose is additionally passed
    through the report's Markdown sanitizer and the final whole-report credential redactor.
    Older evidence rows simply have no generation context and retain their existing layout.
    """
    context = record.get("generation_context")
    # The first deployment writes flat keys in the action row.  Accept that bounded, explicit
    # shape too so a partial evidence writer still leaves useful forensic data rather than a
    # silently empty report.
    if not isinstance(context, dict):
        context = {key: record.get(key) for key in (
            "model", "index_space", "referenced", "angle", "item_description")
            if key in record}

    out: list[str] = []
    if context:
        def text(key: str) -> str:
            value = context.get(key)
            if not isinstance(value, str) or not value:
                return "not recorded"
            return _sanitize_inline(value)[:_GENERATION_CONTEXT_TEXT_LIMIT]

        index_space = context.get("index_space")
        index_display = (index_space if isinstance(index_space, str)
                         and index_space in _GENERATION_INDEX_SPACES else "not recorded")
        out = [
            "- generation context (model-private fields, not a sent/committed opener):",
            f"  - model: `{text('model')}`",
            f"  - index space: `{_sanitize_inline(index_display)}`",
            f"  - referenced: `{text('referenced')}`",
            f"  - angle: `{text('angle')}`",
            f"  - item description: `{text('item_description')}`",
        ]
    out.extend(_prompt_provenance_lines(record.get("prompt_sha256"), current_prompt_sha256))
    return out


def _training_probe_ocr_md(diagnostics: object) -> str:
    """Summarise OCR recipe outcomes while keeping OCR text and names out of the report."""
    if not isinstance(diagnostics, dict):
        return "not recorded"
    attempts = diagnostics.get("ocr_attempts")
    if not isinstance(attempts, list):
        return "not recorded"
    summary: list[str] = []
    for attempt in attempts[:5]:
        if not isinstance(attempt, dict):
            continue
        recipe = _training_probe_token(attempt.get("recipe"), _TRAINING_PROBE_RECIPES)
        verdict = _training_probe_token(attempt.get("verdict"), _TRAINING_PROBE_VERDICTS)
        # `candidate_sha256` proves only that a normalised candidate existed.  Deliberately do
        # not render either that digest or the candidate itself: a report reader needs to know
        # whether OCR found a candidate, not a stable cross-report join key for a person's name.
        candidate = ("candidate redacted" if isinstance(attempt.get("candidate_sha256"), str)
                     and attempt["candidate_sha256"] else "no candidate")
        summary.append(f"{recipe}: verdict={verdict}, {candidate}")
    return "; ".join(summary) if summary else "not recorded"


def _training_probe_frame_md(probe: dict, run: Path, keys: tuple[str, ...],
                             shot_cache: dict[str, str | None]) -> str:
    """Name one observation's retained frame through the normal bounded PNG guard."""
    for key in keys:
        name = probe.get(key)
        # DebugLog owns a deliberately boring filename grammar.  In addition to avoiding
        # Markdown surprises, reject an action-row filename that tries to smuggle personal text
        # into a report even if a same-named file exists under the run directory.
        if not isinstance(name, str) or _DEBUG_SCREENSHOT_NAME_RE.fullmatch(name) is None:
            continue
        if _shot_digest(run, name, shot_cache) is None:
            continue
        # `_shot_digest` already established that this is a bounded, regular, non-symlink PNG
        # under `run`; sanitize too because filenames are still untrusted Markdown text.
        return f"`{_sanitize_inline(str(name))}` (retained on disk; field `{key}`)"
    return "not retained or unsafe"


def _training_probe_observation_md(label: str, diagnostics: object, frame: str) -> str:
    """Render a frame-bound, privacy-safe Training observation."""
    values = diagnostics if isinstance(diagnostics, dict) else {}
    state = (
        "composer_open=" + _training_probe_bool(values.get("composer_open"))
        + "; deck_ready=" + _training_probe_bool(values.get("deck_ready"))
        + "; current_profile=" + _training_probe_bool(values.get("current_profile"))
    )
    content = (
        "exact=" + _training_probe_bool(values.get("current_content_exact_matched"))
        + "; shifted=" + _training_probe_bool(values.get("current_content_shift_matched"))
    )
    identity_value = values.get("identity_verdict")
    if identity_value is None:
        identity_value = values.get("current_identity_state")
    identity = _training_probe_token(identity_value, _TRAINING_IDENTITY_STATES)
    name = _training_probe_token(values.get("name_verdict"), _TRAINING_NAME_VERDICTS)
    return (
        f"- {label} post-X observation: frame {frame}; {state}; "
        f"content({content}); identity={identity}; name_verdict={name}; "
        f"OCR attempts: {_training_probe_ocr_md(values)}."
    )


def _training_dislike_unverified_landing_md(
        records: list[dict], opener_index: int, outcome: dict, run: Path) -> list[str]:
    """Explain the final semantic refusal after an irreversible Training X tap.

    The outcome row's ``before`` image is intentionally the pre-tap, human-reviewed checkpoint.
    The final ``training_advance_probe`` instead records what the deck looked like after that X.
    Keep those two facts visibly separate: neither an unverified action nor an absent OCR name is
    evidence that a label may be recorded.
    """
    try:
        outcome_index = next(index for index, record in enumerate(records) if record is outcome)
    except StopIteration:  # defensive only; callers pass a member of `records`
        return ["- post-X landing verification: the unverified outcome was not bound to a "
                "readable action-log position."]
    final_probe = next(
        (record for record in reversed(records[opener_index + 1:outcome_index])
         if record.get("action") == "training_advance_probe"),
        None)
    if final_probe is None:
        return ["- post-X landing verification: no final Training advance probe was recorded; "
                "the X tap remains unverified and no training label was recorded."]

    outcome_name = _training_probe_token(final_probe.get("outcome"), frozenset({"retry"}))
    attempt_value = final_probe.get("attempt")
    attempt = (str(attempt_value) if isinstance(attempt_value, int)
               and not isinstance(attempt_value, bool) and 1 <= attempt_value <= 3
               else "unrecognised")
    shot_cache: dict[str, str | None] = {}
    first_frame = _training_probe_frame_md(
        final_probe, run, ("kept_before", "before"), shot_cache)
    heading = ("final refused Training advance probe"
               if outcome_name == "retry"
               else "final Training advance probe with an unrecognised outcome")
    observations = [
        f"- post-X landing verification ({heading}; "
        f"attempt `{attempt}`, outcome `{outcome_name}`).",
        _training_probe_observation_md("first", final_probe.get("first"), first_frame),
    ]
    if isinstance(final_probe.get("second"), dict):
        second_frame = _training_probe_frame_md(
            final_probe, run, ("kept_after", "after"), shot_cache)
        observations.append(_training_probe_observation_md(
            "second", final_probe["second"], second_frame))
        observations.append(
            "- pair verdict: stable=" + _training_probe_bool(final_probe.get("stable"))
            + "; names_agree=" + _training_probe_bool(final_probe.get("names_agree")) + ".")
    observations.extend([
        "- the checkpoint snapshot above is pre-action evidence; this probe is the separate "
        "post-X landing observation. The action remains unverified and no training label was "
        "recorded.",
    ])
    return observations


def _latest_auto_opener_evidence_md(lines: list[str], run: Path, *,
                                    current_prompt_sha256: str | None = None) -> str:
    """Render the latest AUTO or Training opener's private evidence and outcome.

    The screenshot proves which selected card was on screen but Hinge's composer may show only
    the tail of a long opener. ``auto_opener_pre_send`` therefore binds the full text and review
    PNG by content hash; when Training resumes after human review,
    ``auto_opener_resumed_send`` records the freshly verified tap-time frame and links it back to
    that approval. Later like rows repeat the applicable evidence ID. Keep this parser defensive:
    actions.jsonl can be partially appended and a debug directory is still untrusted input.

    ``current_prompt_sha256`` (the digest of the prompt checked out RIGHT NOW -- see
    ``_current_prompt_stamp``) is optional and keyword-only so every existing caller/test keeps
    working unchanged; omitting it simply renders the drift verdict as "could not determine".
    """
    records = _action_records(lines)
    # In Training, the reviewer can move the profile while the checkpoint is open.
    # The original pre-send record remains the immutable human-reviewed checkpoint, but it is
    # not necessarily the frame that the eventual Send Like tap used.  Prefer that later,
    # re-verified ``auto_opener_resumed_send`` record whenever it is the latest opener-evidence
    # event. Old/ordinary AUTO logs contain only ``auto_opener_pre_send`` and therefore retain
    # their established rendering and linkage semantics. ``session_mode`` was introduced after
    # these records, so omitted/unknown values intentionally render as AUTO for compatibility.
    candidates = [
        (index, rec) for index, rec in enumerate(records)
        if rec.get("action") in {"auto_opener_pre_send", "auto_opener_resumed_send"}
    ]
    if not candidates:
        return ""
    record_index, record = candidates[-1]
    session_mode = record.get("session_mode")
    session_mode = "training" if session_mode == "training" else "auto"

    opener = record.get("opener")
    opener = opener if isinstance(opener, str) else None
    evidence_id = record.get("evidence_id")
    evidence_id = evidence_id if isinstance(evidence_id, str) else None
    expected_opener_hash = record.get("opener_sha256")
    expected_frame_hash = record.get("frame_sha256")

    opener_integrity = "not verifiable (full opener or SHA-256 was not recorded)"
    if opener is not None and isinstance(expected_opener_hash, str):
        actual = hashlib.sha256(opener.encode("utf-8")).hexdigest()
        opener_integrity = "verified" if actual == expected_opener_hash else "⚠️ SHA-256 mismatch"

    shot_name = record.get("before")
    shot_digest = (_shot_digest(run, shot_name, {})
                   if isinstance(shot_name, str)
                   and _DEBUG_SCREENSHOT_NAME_RE.fullmatch(shot_name) is not None else None)
    if shot_digest is None:
        # Do not echo an unsafe path back into the report.  Apart from misleading the reader
        # about its availability, a flat-but-hostile filename can contain Markdown control text
        # or personal data even though the action row itself is diagnostic input.
        shot_display = "not recorded or unsafe"
        frame_integrity = "⚠️ screenshot missing or unsafe"
    else:
        shot_display = _sanitize_inline(str(shot_name))
        if isinstance(expected_frame_hash, str):
            frame_integrity = ("verified" if shot_digest == expected_frame_hash
                               else "⚠️ SHA-256 mismatch")
        else:
            frame_integrity = "not verifiable (SHA-256 was not recorded)"

    resumed = record.get("action") == "auto_opener_resumed_send"
    outcome_evidence_key = "resumed_send_evidence_id" if resumed else "pre_send_evidence_id"
    outcome = "no linked send outcome was logged"
    legacy_training_dislike = False
    checkpoint_tail: list[dict] = []
    unverified_landing: list[str] = []
    if evidence_id:
        linked = [
            rec for rec in records[record_index + 1:]
            if rec.get(outcome_evidence_key) == evidence_id
            and rec.get("action") in {
                "like_attempt", "like_rejected", "like", "training_dislike",
                "training_dislike_unverified", "training_cancelled",
            }
        ]
        if not linked and session_mode == "training" and not resumed:
            # Rows written before training_dislike carried pre_send_evidence_id can still be
            # joined safely inside one checkpoint: stop at the next capture/opener boundary
            # and require the same model item.  This repairs the exact misleading wording in
            # the 2026-08-26 report without treating an arbitrary later Dislike as this draft's
            # outcome.  New rows use the explicit evidence-ID branch above.
            for candidate in records[record_index + 1:]:
                if candidate.get("action") in {
                        "capture", "auto_opener_pre_send", "auto_opener_resumed_send"}:
                    break
                checkpoint_tail.append(candidate)
            linked = [
                rec for rec in checkpoint_tail
                if rec.get("action") == "training_dislike"
                and rec.get("model_item_index") == record.get("model_item_index")
            ]
            legacy_training_dislike = bool(linked)
        if linked:
            result = linked[-1]
            labels = {
                "like": "LIKE verified as landed",
                "like_rejected": "LIKE rejected",
                "like_attempt": "Send Like attempted; final result not logged",
                "training_dislike": (
                    "DISLIKE verified as landed; typed opener was not sent or committed"
                ),
                "training_dislike_unverified": (
                    "X/Dislike tap was issued but its landing could not be semantically "
                    "verified; no training label was recorded; typed opener was not sent"
                ),
            }
            action = result["action"]
            if action == "training_cancelled":
                label = (
                    "Training decision cancelled by Stop; typed opener was not sent or committed"
                    if result.get("reason") == "stop_requested"
                    else "Training decision cancelled before send; typed opener was not sent or committed"
                )
            else:
                label = labels[action]
            outcome = f"{label} at `{_record_time(result)}`"
            if legacy_training_dislike:
                outcome += " (legacy sequence; the outcome row predates evidence-ID linkage)"
            if action == "training_dislike_unverified":
                unverified_landing = _training_dislike_unverified_landing_md(
                    records, record_index, result, run)
        elif session_mode == "training" and not resumed:
            # Before the cancellation-classification fix, an intentional Hub Stop while the
            # reviewer held this checkpoint was retained as an ``unexpected`` error record.
            # It still establishes the important outcome: the driver raised before the Send
            # Like boundary, so this typed draft was neither sent nor committed. Limit the
            # inference to this checkpoint; a cancellation after another capture/evidence row
            # belongs to a different profile.
            stopped = next((candidate for candidate in checkpoint_tail
                            if candidate.get("action") == "unexpected"
                            and isinstance(candidate.get("error"), str)
                            and "actioncancelled" in candidate["error"].lower()
                            and "stopp" in candidate["error"].lower()), None)
            if stopped is not None:
                outcome = ("Training decision cancelled by Stop; typed opener was not sent or "
                           f"committed at `{_record_time(stopped)}`")

    target = record.get("model_item_index")
    target_display = str(target) if isinstance(target, int) and not isinstance(target, bool) else "not recorded"
    evidence_display = _sanitize_inline(evidence_id) if evidence_id else "not recorded"
    opener_display = (_sanitize_inline(opener) if opener is not None
                      else "not recorded (legacy evidence row)")
    out = [
        f"- session mode: {session_mode}",
        f"- evidence ID: `{evidence_display}`",
        f"- {'pre-action checkpoint snapshot' if unverified_landing else 'snapshot'}: "
        f"`{shot_display}` ({frame_integrity})",
        f"- target: model item `{target_display}`",
        f"- full opener: `{opener_display}` ({opener_integrity})",
    ]
    # Unconditional, not training-only: `prompt_sha256` is stamped on every AUTO row exactly
    # like Training's (see hinge.py's _record_auto_opener_pre_send), and the legacy
    # model-private fields this call also renders stay quiet on their own for AUTO because
    # production AUTO never populates `generation_context` in the first place.
    out.extend(_generation_context_md(record, current_prompt_sha256=current_prompt_sha256))
    out.append(_replay_corpus_pointer_md())
    if resumed:
        approval_evidence_id = record.get("approval_evidence_id")
        approval_display = (_sanitize_inline(approval_evidence_id)
                            if isinstance(approval_evidence_id, str) and approval_evidence_id
                            else "not recorded")
        out.append(f"- human-reviewed approval evidence ID: `{approval_display}`")
    out.append(f"- linked outcome: {outcome}")
    out.extend(unverified_landing)
    return "\n".join(out)


def _latest_auto_opener_evidence_mode(lines: list[str]) -> str:
    """Return the current diagnostic label, treating pre-migration rows as AUTO."""
    records = _action_records(lines)
    for record in reversed(records):
        if record.get("action") in {"auto_opener_pre_send", "auto_opener_resumed_send"}:
            return "Training" if record.get("session_mode") == "training" else "AUTO"
    return "AUTO"


def _safe_action_scalar(rec: dict, key: str) -> str:
    """Return a JSON-scalar action field fit for an inline diagnostic.

    The action log is useful forensic input, not trusted report markup.  In particular, do
    not stringify a list or mapping from a partially written/corrupt row: apart from making
    the concise end-of-log diagnosis noisy, its nested contents might be misleading or
    sensitive.  Scalars are enough for the transport metadata this summary needs.
    """
    value = rec.get(key)
    if isinstance(value, (str, int, float, bool)):
        text = _sanitize_inline(str(value))
        if text:
            return text
    return "not recorded"


def _latest_observe_context_md(lines: list[str], run: Path) -> str:
    """Turn the newest observe trace into the answer a terse report actually needs.

    The raw JSON tail is retained below as forensic evidence, but a report should not require a
    developer to reverse-engineer the latest capture -> READY -> wait sequence from it.  This
    deliberately describes the *latest logged evidence*, rather than claiming it is a fresh
    screenshot of the phone at report-generation time.  That distinction matters if logging has
    stopped or the file belongs to an earlier run.
    """
    records = _action_records(lines)
    if not records:
        return ""
    latest = records[-1]
    action = _sanitize_inline(str(latest.get("action") or "unknown action"))
    latest_time = (_safe_action_scalar(latest, "ts")
                   if latest.get("action") == "device_input" else _record_time(latest))
    out = [f"- latest logged action: `{action}` at `{latest_time}`"]

    # A Stop during `_capture_current` logs `capture_aborted` before the loop's finally-block
    # timing record.  Do not infer a phone state from either record; this only restores the
    # terminal lifecycle fact and the scalar progress the abort itself logged.
    aborted_capture = _terminal_capture_abort(records)
    if aborted_capture is not None:
        out.append("- terminal capture state: Stop abandoned the in-progress profile read; "
                   "no profile capture or decision was recorded for it.")
        details: list[str] = []
        profile_name = aborted_capture.get("profile_name")
        if isinstance(profile_name, str) and profile_name.strip():
            details.append(f"profile identity `{_sanitize_inline(profile_name)}`")
        for label, key in (("captured frame(s)", "frames"),
                           ("read scroll(s)", "read_scrolls")):
            value = aborted_capture.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                details.append(f"{value} {label}")
        if details:
            out.append("- abandoned-capture details from the log: " + "; ".join(details) + ".")
        else:
            out.append("- the abort row did not record profile identity, frames, or read-scrolls.")
        return "\n".join(out)

    # `observe_stopped` is a terminal marker written by Hinge after Worker has noticed a Stop
    # during the manual-decision wait.  The preceding observe_waiting record is still valuable
    # historical evidence, but it must not be called a *current* unresolved wait in a report
    # produced after the run has stopped.  It was exactly that wording mismatch which made a
    # clean Stop on a READY second card look like an observe hang.
    if latest.get("action") == "observe_stopped":
        out.append("- observe wait ended because Stop was requested; the READY card was "
                   "intentionally abandoned and no pass/like decision was recorded for it.")
        profile_name = latest.get("profile_name")
        if profile_name:
            out.append("- the stopped wait belonged to the captured profile identity `"
                       f"{_sanitize_inline(str(profile_name))}`.")
        return "\n".join(out)

    # `device_input` is deliberately emitted only after the synchronous transport call
    # returns.  A trace ending here can prove its final input completed, but it has no evidence
    # for the code/device state which prevented the next log write; never recast that gap as an
    # interrupted gesture.  Include only scalar metadata so an arbitrary action row cannot
    # turn this compact diagnostic into a dump of untrusted nested data.
    if latest.get("action") == "device_input":
        details = "; ".join(
            f"{label}=`{_safe_action_scalar(latest, key)}`"
            for label, key in (("transport", "transport"), ("kind", "kind"),
                               ("source", "source"), ("direction", "direction"),
                               ("time", "ts"))
        )
        out.append(f"- final completed device input: {details}.")
        out.append("- logging ended after that completed input; the trace cannot pinpoint "
                   "what subsequently blocked progress.")
        return "\n".join(out)

    # Only the final logged wait is relevant. A decision/capture/resync before it resolved an
    # older wait; known release telemetry is deliberately transparent (see helper). The action
    # log has no lifecycle authority after its final write: a manual Stop can end the worker
    # immediately after a waiting heartbeat, so do not call this state "current" or imply the
    # driver remains alive when a report is generated later.
    wait, first_wait_index = _final_observe_wait(records)
    if not wait:
        return "\n".join(out)
    final_wait = wait[-1]
    reason = final_wait.get("reason")
    reason_text = _sanitize_inline(str(reason or "not recorded"))
    explanation = _OBSERVE_WAIT_EXPLANATIONS.get(
        reason if isinstance(reason, str) else "",
        "the driver had not logged a proven manual decision before logging ended",
    )
    out.append(f"- final logged observe state: waiting (`{reason_text}`) — {explanation}.")
    if len(wait) > 1:
        out.append(f"- this final logged wait began at `{_record_time(wait[0])}` and has "
                   f"{len(wait)} heartbeat record(s), latest at `{_record_time(final_wait)}`.")
    else:
        out.append("- this is the first logged waiting heartbeat before logging ended.")

    # The immediately preceding capture is the most useful reproduction context: it says what
    # was successfully read before the app entered READY, without inventing a current screen.
    before_wait = records[:first_wait_index] if first_wait_index is not None else []
    capture = next((rec for rec in reversed(before_wait) if rec.get("action") == "capture"), None)
    if capture is not None:
        bits: list[str] = []
        if capture.get("profile_name"):
            bits.append(f"identity read `{_sanitize_inline(str(capture['profile_name']))}`")
        photos = capture.get("photos")
        if isinstance(photos, int) and not isinstance(photos, bool):
            bits.append(f"{photos} captured photo(s)")
        items = capture.get("items")
        if isinstance(items, int) and not isinstance(items, bool):
            item_context = capture.get("item_context")
            context_suffix = (f", {item_context} context crop(s)"
                              if isinstance(item_context, int) and not isinstance(item_context, bool)
                              else "")
            bits.append(f"{items} numbered item(s){context_suffix}")
        if capture.get("items_unavailable"):
            bits.append("numbered items unavailable: `"
                        f"{_compact_item_index_refusal_text(capture['items_unavailable'])}`")
        elif capture.get("items_unnumbered"):
            # Distinct from the failure above: enumeration finished and legitimately numbered
            # nothing (Profile's state 3), so this must never read as "something broke".
            bits.append("enumeration completed but numbered nothing: `"
                        f"{_compact_item_index_refusal_text(capture['items_unnumbered'])}`")
        detail = "; ".join(bits) if bits else "no capture detail was logged"
        out.append(f"- last capture before this wait (`{_record_time(capture)}`): {detail}.")

    shot = next((final_wait.get(key) for key in ("before", "after", "screenshot")
                 if final_wait.get(key)), None)
    if shot:
        shot_text = _sanitize_inline(str(shot))
        availability = "present" if (run / str(shot)).is_file() else "not present (possibly rotated)"
        out.append(f"- screen evidence for that waiting verdict: `{shot_text}` ({availability}).")
    else:
        out.append("- no screenshot filename was recorded with the latest waiting verdict.")
    out.append("- reproduction sequence from the log: capture completed → READY/manual decision "
               "prompt → no pass/like record before the final logged wait above.")
    return "\n".join(out)


def _foreground_input_attribution_md(lines: list[str]) -> str:
    """Correlate a foreground interruption with the driver's last completed input.

    Screenshots prove *what* covered Hinge but not *who moved it*.  Current Android drivers
    therefore append ``device_input`` only after each synchronous gesture succeeds.  This
    summary makes the useful negative evidence explicit: if the shade appeared much later with
    no intervening input row, Operation Love did not pull it down.  Older runs are labelled as
    lacking the audit trail rather than being retroactively exonerated.
    """
    records = _action_records(lines)
    event_indexes = [
        index for index, rec in enumerate(records)
        if rec.get("action") in {"observe_foreground_paused", "foreground_blocked"}
    ]
    if not event_indexes:
        return ""
    index = event_indexes[-1]
    event = records[index]
    prior_inputs = [rec for rec in records[:index] if rec.get("action") == "device_input"]
    resumed_index = next((later_index for later_index in range(index + 1, len(records))
                          if records[later_index].get("action") ==
                          "observe_foreground_resumed"), None)
    interruption_tail = records[index + 1:resumed_index]
    later_inputs = [rec for rec in interruption_tail if rec.get("action") == "device_input"]
    package = _sanitize_inline(str(event.get("package") or "package not recorded"))
    out = [f"- foreground interruption at `{_record_time(event)}`: `{package}`."]
    if prior_inputs:
        last = prior_inputs[-1]
        kind = _sanitize_inline(str(last.get("kind") or "input"))
        source = _sanitize_inline(str(last.get("source") or "source not recorded"))
        event_ts = _parse_action_ts(event.get("ts"))
        input_ts = _parse_action_ts(last.get("ts"))
        gap = None if event_ts is None or input_ts is None else (event_ts - input_ts).total_seconds()
        gap_text = (f"{format_duration(max(0.0, gap))} earlier"
                    if gap is not None else "at an unknown time before it")
        geometry = ""
        if last.get("start") is not None or last.get("end") is not None:
            geometry = (f", start=`{_sanitize_inline(str(last.get('start')))}`"
                        f", end=`{_sanitize_inline(str(last.get('end')))}`")
        out.append(
            f"- last completed Operation Love device input: `{kind}` from `{source}` at "
            f"`{_record_time(last)}` ({gap_text}){geometry}.")
        if later_inputs:
            out.append(f"- ⚠️ {len(later_inputs)} Operation Love device input(s) were logged "
                       "after the foreground interruption; inspect the action tail below.")
        else:
            out.append("- no Operation Love device input was logged after the interruption.")
    else:
        out.append("- this run contains no outbound `device_input` audit row before the "
                   "interruption; input attribution is unavailable (legacy/incomplete trace).")
    resumed = records[resumed_index] if resumed_index is not None else None
    if resumed is not None:
        out.append(f"- Hinge regained foreground at `{_record_time(resumed)}`; Observe "
                   "recaptured without recording a decision.")
    return "\n".join(out)


# ── evidence anomalies ─────────────────────────────────────────────────────
# Filed after the 2026-08-14 run, where this report printed every record faithfully and still
# could not surface either of the two bugs sitting in it. Both were found only by hashing the
# PNGs by hand and reading the phone's own status-bar clock out of the pixels:
#   * the composer detector called the like sheet CLOSED for one poll while the owner was
#     editing the opener, because Android's text-selection handle welded the comment input to
#     the Send Like CTA into one connected component; and
#   * the LIKE's evidence frame was a five-minute-stale screenshot from before the heart was
#     even tapped, because _await_like_resolved holds its `base` anchor frozen by design.
# Neither is visible by eye in an actions.jsonl tail. These two checks say them in words.
_DECISION_ACTIONS = ("observe_decision", "observe_like_dismissed", "observe_bottom_delta")


def _composer_flap_md(lines: list[str]) -> str:
    """Report every `like_sending` -> `like_sheet` regression in the run.

    `like_sending` asserts the composer CLOSED and Hinge is now sending; `like_sheet` asserts
    it is open with the human still composing. Hinge cannot reopen a sheet by itself, so the
    forward transition is one-way in reality and every regression means at least one poll
    misread the composer. That is worth its own line rather than being left inside the stall
    summary, because the two states are NOT symmetric in cost: `like_sheet` re-arms the
    stuck-screen budget on every poll (a human may take minutes writing an opener) while
    `like_sending` deliberately does not (see _await_like_resolved). A false `like_sending`
    therefore spends a watchdog allowance that belongs to the human who is still typing.
    """
    stretches = _observe_waiting_stretches(lines)
    flaps: list[str] = []
    for stretch in stretches:
        for earlier, later in pairwise(stretch):
            if earlier.get("reason") != "like_sending" or later.get("reason") != "like_sheet":
                continue
            flaps.append(f"`like_sending` at `{earlier.get('ts', '?')}` -> "
                         f"`like_sheet` at `{later.get('ts', '?')}`")
    if not flaps:
        return ""
    out = [f"- ⚠️ the composer was reported CLOSED and then OPEN again {len(flaps)}x -- a like "
           "sheet cannot reopen itself, so at least one of these polls misread it:"]
    out.extend(f"  - {flap}" for flap in flaps)
    out.append("  - each false `like_sending` also burns stuck-screen budget that `like_sheet` "
               "would have re-armed, so this can end a wait the human was still composing in")
    return "\n".join(out)


def _shot_digest(run: Path, name: object, cache: dict[str, str | None]) -> str | None:
    """sha256 of a screenshot referenced by an actions.jsonl record, or None.

    Cached per filename because a run's records reference the same shot repeatedly (the debug
    log reuses one file for consecutive identical frames).  The record is untrusted input, so a
    symlink, non-regular file, empty file, or oversized PNG is never opened.  Any unsafe,
    unreadable, or missing file is None -- a bug report must never raise or escape its run
    directory while describing a bug."""
    if not isinstance(name, str) or not name:
        return None
    candidate = Path(name)
    # actions.jsonl is diagnostic input, not authority to read arbitrary filesystem paths.
    # Debug screenshots are flat PNG names inside the run directory; anything else is
    # malformed evidence and must stay unread rather than following traversal/absolute paths.
    if candidate.name != name or candidate.suffix.lower() != ".png":
        return None
    if name not in cache:
        try:
            path = run / name
            # A bare name alone does not confine a symlink: a debug row can point it outside the
            # run directory.  Match the sidecar reader's regular-file and bounded-size guard.
            if path.is_symlink() or not path.is_file():
                cache[name] = None
            else:
                size = path.stat().st_size
                cache[name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                               if 0 < size <= _DEBUG_SCREENSHOT_MAX_BYTES else None)
        except Exception:  # noqa: BLE001
            cache[name] = None
    return cache[name]


def _adb_screencap_timeout_md(lines: list[str], run: Path) -> str:
    """Explain the last retained ADB screencap timeout in actionable transport terms.

    The raw ``unexpected`` row records the failed request but used to leave the operator to
    infer whether the phone disconnected or merely stalled.  The driver's failure snapshot
    performs a screencap *after* that exception; a saved error shot is therefore direct recovery
    evidence for this path.  A missing shot deliberately stays indeterminate rather than being
    called a disconnect -- disk logging and the follow-up capture are both best effort.
    """
    matches = [rec for rec in _action_records(lines)
               if rec.get("action") == "unexpected"
               and isinstance(rec.get("error"), str)
               and _ADB_SCREENCAP_TIMEOUT_RE.search(rec["error"])]
    if not matches:
        return ""
    latest = matches[-1]
    at = _sanitize_inline(str(latest.get("ts") or "time not recorded"))
    screenshot = latest.get("screenshot")
    digest = _shot_digest(run, screenshot, {})
    if digest is not None:
        return (
            f"- ADB screencap timeout at `{at}`: a post-timeout failure snapshot was "
            f"saved as `{_sanitize_inline(str(screenshot))}`. The follow-up raw screencap "
            "command completed and retained non-empty screenshot bytes, so ADB was responsive "
            "again: this was a transient screencap/ADB stall rather than a persistent "
            "disconnect while the failure handler ran."
        )
    if isinstance(screenshot, str) and screenshot:
        evidence = (f"`{_sanitize_inline(screenshot)}` was named but is missing or unsafe to "
                    "read")
    else:
        evidence = "no post-timeout screenshot was saved"
    return (
        f"- ADB screencap timeout at `{at}`: {evidence}; recovery versus disconnect is "
        "indeterminate because the best-effort failure snapshot did not leave usable evidence."
    )


# like_candidate is deliberately absent: it means only that the lower screen changed, without
# structural proof that a composer ever opened. Treating it as proof manufactures severe stale-
# evidence warnings for benign snackbars and other bottom-only deltas.
_COMPOSER_WAIT_REASONS = frozenset({"like_sheet", "like_sending"})


def _stale_evidence_md(lines: list[str], run: Path) -> str:
    """Flag a resolved composer whose evidence frame PREDATES that composer.

    A like record claims the human sent something through the inline composer, so its one
    stored frame should show that composer. Frames are compared by CONTENT, not filename: the
    debug log keys dedup on (action label, digest), so the same pixels logged under a different
    action get a fresh filename -- which is how the 2026-08-14 stale LIKE evidence hid in plain
    sight as `00019_observe_decision_before.png` while being byte-identical to a waiting shot
    from five minutes earlier.

    "Predates the composer" is the discriminator, rather than the simpler "these two frames are
    identical", and the difference is measured rather than stylistic. Reusing an earlier frame
    is usually CORRECT: a decision's evidence is the card as it looked just before it advanced,
    and on a motionless screen that is legitimately the previous waiting shot. Across this
    project's runs the naive rule fired on ~10 of 16, nearly all benign, and a check that cries
    wolf teaches the reader to skip it. Byte-difference alone is no better -- the status-bar
    clock ticks below the driver's own change threshold, so frames differ while the screen has
    not meaningfully moved. Evidence captured before the composer episode even began is
    unambiguous: whatever it shows, it cannot show what was sent.
    """
    cache: dict[str, str | None] = {}
    first_seen: dict[str, int] = {}
    records: list[dict] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(rec, dict):
            records.append(rec)
    findings: list[str] = []
    composer_since: int | None = None          # where the in-flight composer episode began
    for index, rec in enumerate(records):
        action = rec.get("action")
        if action == "observe_like_anchor" or (
                action == "observe_waiting" and rec.get("reason") in _COMPOSER_WAIT_REASONS):
            if composer_since is None:
                composer_since = index
        digest = _shot_digest(run, rec.get("before"), cache)
        if digest is not None:
            first_seen.setdefault(digest, index)
        if action not in _DECISION_ACTIONS:
            continue
        opened_at, composer_since = composer_since, None
        if opened_at is None or digest is None or first_seen[digest] >= opened_at:
            continue
        origin = records[first_seen[digest]]
        gap = ""
        then, now = _parse_action_ts(origin.get("ts")), _parse_action_ts(rec.get("ts"))
        if then is not None and now is not None:
            gap = f", captured {_format_stall_duration((now - then).total_seconds())} earlier"
        findings.append(
            f"`{action}` at `{rec.get('ts', '?')}` is evidenced by `{rec.get('before')}`, "
            f"pixel-identical to `{origin.get('before')}` from `{origin.get('action')}` at "
            f"`{origin.get('ts', '?')}`{gap} -- before this composer opened at "
            f"`{records[opened_at].get('ts', '?')}`")
    if not findings:
        return ""
    out = ["- ⚠️ a resolved like is evidenced by a frame from BEFORE its composer existed, so "
           "the one stored frame cannot show what the human actually sent:"]
    out.extend(f"  - {finding}" for finding in findings)
    return "\n".join(out)


def _debug_log_md(config_path: str, hub_state=None) -> str:
    """Surface the on-disk action/screenshot debug log (Hinge's silent auto-mode logging) so the
    report points a developer straight at a failure: the latest run folder, the tail of its
    actions.jsonl, and any error screenshots (which are kept un-rotated). Screenshots are binary,
    so we list their paths rather than inline them. Best-effort; never raises."""
    try:
        from . import config as cfg_mod
        cfg = cfg_mod.load(config_path)
        apps = cfg.apps
        enabled_apps = set(cfg.enabled_apps)
    except Exception as exc:  # noqa: BLE001
        return f"- (could not load config to locate debug logs: {exc})"
    # Computed once for every app's evidence block below, rather than per-block, so a report
    # with several enabled apps doesn't reload config.yaml and re-hash the prompt once per app.
    current_prompt_sha256 = _current_prompt_stamp(config_path)
    current_run_id = None
    if hub_state is not None:
        try:
            hub_snapshot = hub_state.snapshot()
            status_snapshot = (hub_snapshot.get("status")
                               if isinstance(hub_snapshot, dict) else None)
            candidate = (status_snapshot.get("run_id")
                         if isinstance(status_snapshot, dict) else None)
            if isinstance(candidate, str) and candidate:
                current_run_id = candidate
        except Exception:  # noqa: BLE001 -- provenance is optional diagnostic context
            pass
    apps = apps if isinstance(apps, dict) else {}
    sections = [_one_debug_dir_md(app, (opts or {}), current_run_id=current_run_id,
                                  current_prompt_sha256=current_prompt_sha256)
                for app, opts in apps.items()
                if app in enabled_apps and (opts or {}).get("debug_log")]
    if not sections:
        return "- (no enabled app has `debug_log` enabled — nothing on-disk to include)"
    return "\n".join(sections)


def _one_debug_dir_md(app: str, opts: dict, *, current_run_id: str | None = None,
                      current_prompt_sha256: str | None = None) -> str:
    base = Path(opts.get("debug_dir", f"./data/{app}_debug"))
    if not base.is_absolute():
        base = Path.cwd() / base
    try:
        if not base.exists():
            return f"- **{app}**: `debug_log` on, but no folder yet at `{base}` (no run has logged here)"
        runs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not runs:
            return f"- **{app}**: `{base}` exists but has no run folders yet"
        run = runs[-1]                                       # most recent run
        pngs = sorted(run.glob("*.png"))
        errors = [p.name for p in pngs if p.name.endswith("_error.png")]
        if current_run_id is None:
            provenance = ""
        elif run.name == current_run_id:
            provenance = " · provenance: current status run"
        else:
            provenance = (" · provenance: previous on-disk run; current status run is `"
                          f"{_sanitize_inline(current_run_id)}`")
        out = [f"- **{app}** · latest run: `{run}`{provenance} · screenshots: {len(pngs)}"
               + (f" · ⚠️ error shots (kept): {', '.join(errors)}" if errors else "")]
        log = run / "actions.jsonl"
        if log.exists():
            try:
                raw_lines = log.read_text().splitlines()
            except Exception:  # noqa: BLE001
                raw_lines = []
            adb_timeout = _adb_screencap_timeout_md(raw_lines, run)
            if adb_timeout:
                out.append("  - ADB timeout diagnosis:")
                out.extend(f"    {line}" for line in adb_timeout.splitlines())
            # Repeated waits go FIRST, ahead of the action-counts histogram and the tail --
            # it is the one line a developer needs before anything else if this run hung (see
            # _stall_summary_md's docstring for the incident this is filed against). Rendered
            # as its own nested bullet block only when there's something to say; a healthy run
            # (nothing repeated on the same reason) adds nothing here.
            stall = _stall_summary_md(raw_lines)
            if stall:
                out.append("  - repeated observe waits:")
                out.extend(f"    {line}" for line in stall.splitlines())
            # Names the specific event the 2026-08-15 "it moved on again without waiting for my
            # like or dislike" report needed but never got: the resync record was in the log,
            # but nothing in the report named it (see _abandoned_card_summary_md's docstring).
            # Quiet on a run where every card ended in a proven capture or decision.
            abandoned = _abandoned_card_summary_md(raw_lines)
            if abandoned:
                out.append("  - cards abandoned without a decision (resync):")
                out.extend(f"    {line}" for line in abandoned.splitlines())
            foreground = _foreground_input_attribution_md(raw_lines)
            if foreground:
                out.append("  - Android foreground/input attribution:")
                out.extend(f"    {line}" for line in foreground.splitlines())
            # Beside the stall summary, and for the same reason: these are the anomalies a
            # developer cannot see by reading the tail, because both look like ordinary
            # records until you hash the screenshots. Quiet on a healthy run.
            anomalies = "\n".join(part for part in
                                  (_composer_flap_md(raw_lines), _stale_evidence_md(raw_lines, run))
                                  if part)
            if anomalies:
                out.append("  - evidence anomalies:")
                out.extend(f"    {line}" for line in anomalies.splitlines())
            capture_timing = _latest_completed_capture_timing_md(raw_lines)
            if capture_timing:
                out.append("  - latest completed capture timing:")
                out.extend(f"    {line}" for line in capture_timing.splitlines())
            # Immediately before the opener evidence, because verification is the gate the opener
            # is waiting on: when this section is present, the opener below was NOT sent.
            verification = _sheet_verification_md(raw_lines, run)
            if verification:
                out.append("  - post-tap sheet verification (the gate the opener never passed):")
                out.extend(f"    {line}" for line in verification.splitlines())
            opener_evidence = _latest_auto_opener_evidence_md(
                raw_lines, run, current_prompt_sha256=current_prompt_sha256)
            if opener_evidence:
                evidence_mode = _latest_auto_opener_evidence_mode(raw_lines)
                out.append(f"  - latest {evidence_mode} opener pre-send evidence:")
                out.extend(f"    {line}" for line in opener_evidence.splitlines())
            observe_context = _latest_observe_context_md(raw_lines, run)
            if observe_context:
                out.append("  - latest observe context (logged evidence, not a new phone read):")
                out.extend(f"    {line}" for line in observe_context.splitlines())
            splits = _capture_split_summary_md(raw_lines)
            if splits:
                out.append("  - capture-split recovery:")
                out.extend(f"    {line}" for line in splits.splitlines())
            refusals = _item_index_refusal_summary_md(raw_lines)
            if refusals:
                out.append("  - item-index refusals and realised-step stats:")
                out.extend(f"    {line}" for line in refusals.splitlines())
            # Directly under the refusal that names the sidecar, because this is that
            # refusal's own evidence file opened: the block top it reports is derived here.
            geometry = _item_index_geometry_md(raw_lines, run)
            if geometry:
                out.append("  - refused-capture leading-edge geometry (read from the sidecar "
                           "the refusal names):")
                out.extend(f"    {line}" for line in geometry.splitlines())
            dwell_navigation = _dwell_navigation_refusal_summary_md(raw_lines)
            if dwell_navigation:
                # "and cancellations" is not padding (2026-09-16): `navigation_cancelled` rows
                # render here, and the bullet under this heading says plainly that a requested
                # Stop is "not a targeting or measurement fault". A heading reading only
                # "refusals" reasserts that exact mislabel one level up, which is the thing
                # this module's own norm forbids -- "A Stop is not a failure to diagnose, so
                # this says so plainly rather than borrowing the refusal wording".
                out.append("  - dwell-navigation refusals and cancellations "
                           "(separate from item-index correspondence):")
                out.extend(f"    {line}" for line in dwell_navigation.splitlines())
            dwell_return_chains = _dwell_return_chain_refusal_summary_md(raw_lines)
            if dwell_return_chains:
                out.append("  - still-photo return-chain refusals (measured cleanup after dwell):")
                out.extend(f"    {line}" for line in dwell_return_chains.splitlines())
            repairs = _item_index_repair_summary_md(raw_lines)
            if repairs:
                out.append("  - item-index conservative repairs:")
                out.extend(f"    {line}" for line in repairs.splitlines())
            # Sibling headings (2026-09-17, corrected same day TWICE): `repairs`, screen-fixed
            # notes, and every other note are three ORTHOGONAL views of the same note-bearing
            # records -- see `_item_index_note_bearing_records` and
            # `_item_index_note_is_screen_fixed` for why the split is per-NOTE, not per-record,
            # and reads every note-bearing record rather than only the ones the repairs heading
            # above skipped.
            screen_fixed_notes = _item_index_screen_fixed_notes_summary_md(raw_lines)
            if screen_fixed_notes:
                out.append("  - item-index screen-fixed check notes:")
                out.extend(f"    {line}" for line in screen_fixed_notes.splitlines())
            construction_notes = _item_index_construction_notes_summary_md(raw_lines)
            if construction_notes:
                out.append("  - item-index construction notes (not from the screen-fixed check):")
                out.extend(f"    {line}" for line in construction_notes.splitlines())
            manifest = _item_manifest_summary_md(raw_lines)
            if manifest:
                out.append("  - item-numbering manifest (non-image capture provenance):")
                out.extend(f"    {line}" for line in manifest.splitlines())
            counts_line = _action_counts_line(raw_lines)
            if counts_line:
                out.append(f"  - {counts_line}")
            tail = [_compact_debug_tail_line(line, _manifest_capture(raw_lines))
                    for line in _collapse_action_tail(raw_lines, _DEBUG_ACTION_TAIL)]
            if tail:
                out.append("  - actions.jsonl (tail):\n```\n" + "\n".join(tail) + "\n```")
        else:
            out.append("  - (no actions.jsonl in the latest run)")
        return "\n".join(out)
    except Exception as exc:  # noqa: BLE001
        return f"- **{app}**: could not read debug dir `{base}`: {exc}"


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _logs_md(max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    logs = recent_logs(_MAX_REPORT_LINES)
    if not logs:
        return "_(no logs captured this session)_"
    if max_lines < 3:
        return "_(omitted - report at 50,000-line cap)_"

    available_body_lines = max_lines - 2        # opening + closing code fences
    prefix: list[str] = []
    if len(logs) > available_body_lines:
        available_body_lines -= 1               # the omission notice itself costs a body slot
        shown = max(0, available_body_lines)    # derive the count from what's ACTUALLY shown
        prefix = [
            f"... {len(logs) - shown} older log line(s) omitted to keep the report under "
            f"{_MAX_REPORT_LINES:,} lines ..."
        ]

    body_lines = prefix
    if available_body_lines > 0:
        # The exact long refusal is already rendered once in the on-disk item-index summary.
        # Console logs mirror that same sentence, so retain their timestamp/context but replace
        # only an oversized geometry wall with the same explicit pointer.  Short/ordinary logs
        # remain byte-for-byte unchanged.
        body_lines += [_compact_item_index_refusal_text(line)
                       for line in logs[-available_body_lines:]]
    return "```\n" + "\n".join(body_lines) + "\n```"


def _cap_report_lines(report: str) -> str:
    lines = report.splitlines()
    if len(lines) <= _MAX_REPORT_LINES:
        return report if report.endswith("\n") else report + "\n"

    pinned = lines[:2]
    marker = (
        f"... {len(lines) - _MAX_REPORT_LINES + 1} older report line(s) "
        f"omitted to keep the report under {_MAX_REPORT_LINES:,} lines ..."
    )
    tail_budget = _MAX_REPORT_LINES - len(pinned) - 1
    if tail_budget <= 0:
        return "\n".join(lines[-_MAX_REPORT_LINES:]) + "\n"
    return "\n".join([*pinned, marker, *lines[-tail_budget:]]) + "\n"


def _opener_rejection_deadletter_md(config_path: str = "config.yaml") -> str:
    """Surface the local rejection dead-letter, because a diagnostic nobody reads is not one.

    THE INCIDENT: production BigQuery `opener_rejections` sat at ZERO rows for the table's
    entire history while `spend` showed at least 159 billed OpenerParseError events -- so
    rejections were happening and the durable write was failing. The exception text that would
    have named the cause went to a bare print() nothing captured, and two separate fixes shipped
    against theories because nobody ever saw it. `OpenerService` now writes that exception to a
    bounded local JSONL (see _write_opener_rejection_deadletter in opener/service.py, wired from
    supervisor.py off cfg.data_dir).

    This section is the last link in that chain. A file that only fills up during an incident is
    worth nothing if the report filed about that incident does not mention it, so this renders
    LOUDLY when there are entries and stays a single quiet line when there are none -- the empty
    case is the healthy case and must not read as a finding.

    Shows the newest entries' exception identity rather than whole rows: `cause_type`/`cause_str`
    is where BigQuery client errors bury the real reason, so that is what a reader needs first.
    Never raises; a missing or unreadable file degrades to a plain line like every other section.
    """
    try:
        from . import config as cfg_mod
        path = Path(cfg_mod.load(config_path).data_dir) / "opener_rejection_deadletter.jsonl"
    except Exception as exc:  # noqa: BLE001
        return f"- could not resolve the dead-letter path: {type(exc).__name__}: {exc}"
    if not path.exists():
        return ("- none: no opener-rejection row has failed to reach the store "
                f"(`{path}` does not exist)")
    try:
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        return f"- ⚠️ `{path}` exists but could not be read: {type(exc).__name__}: {exc}"
    if not entries:
        return f"- none: `{path}` is empty"
    out = [f"- ⚠️ **{len(entries)} opener-rejection row(s) FAILED to reach the store** and were "
           f"written to `{path}`.",
           "  - This is the evidence three prior investigations of the empty "
           "`opener_rejections` table lacked: the exception each failed write actually raised. "
           "Nothing here diagnoses the cause yet -- read `exc_type`/`exc_str`, then "
           "`cause_type`/`cause_str`, which is where a chained cause hides.",
           "  - newest first:"]
    for entry in reversed(entries[-5:]):
        if not isinstance(entry, dict):
            continue
        row = entry.get("row") if isinstance(entry.get("row"), dict) else {}
        cause = entry.get("cause_type") or "no chained cause"
        out.append(
            f"    - `{_sanitize_inline(str(entry.get('ts', '?')))}` · branch="
            f"`{_sanitize_inline(str(entry.get('branch', '?')))}` · store="
            f"`{_sanitize_inline(str(entry.get('store_class', '?')))}` · reason_code="
            f"`{_sanitize_inline(str(row.get('reason_code', '?')))}` · "
            f"{_sanitize_inline(str(entry.get('exc_type', '?')))}: "
            f"{_sanitize_inline(str(entry.get('exc_str', '')))[:200]} · cause: "
            f"{_sanitize_inline(str(cause))}: "
            f"{_sanitize_inline(str(entry.get('cause_str', '')))[:200]}")
    if len(entries) > 5:
        out.append(f"    - ... {len(entries) - 5} older entry(ies) in the file")
    return "\n".join(out)


def _safe_section(fn, *args) -> str:
    """Call a report-section builder, rendering a warning instead of crashing the whole
    report if it raises. The bug-report endpoint is the tool reached for when something's
    already broken — it must never itself be a single point of failure."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        return f"- ⚠️ this section failed to generate: {type(exc).__name__}: {exc}"


# ── assembly ───────────────────────────────────────────────────────────────
def build_report(hub_state=None, description: str = "", config_path: str = "config.yaml") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    desc = (description or "").strip() or "_(none provided)_"
    head = (
        f"# Operation Love — Bug Report\n_Generated {now}_\n\n"
        f"## What happened\n{desc}\n\n"
        f"## Reporter follow-up\n{_safe_section(_reporter_follow_up_md, description or '')}\n\n"
        f"## Build\n{_safe_section(_build_md)}\n\n"
        f"## System\n{_safe_section(_system_md)}\n\n"
        f"## Dependencies\n{_safe_section(_deps_md)}\n\n"
        f"## Capabilities\n{_safe_section(_capabilities_md, config_path)}\n\n"
        f"## Config (config.yaml)\n{_safe_section(_config_md, config_path)}\n\n"
        f"## Hinge targeting readiness\n{_safe_section(_targeting_readiness_md, config_path, hub_state)}\n\n"
        f"## Secrets (presence only — never raw values)\n{_safe_section(_secrets_md)}\n\n"
        f"## Diagnostic improvement\n{_safe_section(_diagnostic_improvement_md)}\n\n"
        f"## Training alerts\n{_safe_section(_training_alerts_md)}\n\n"
        f"## Run completion assessment\n{_safe_section(_run_completion_assessment_md, hub_state, config_path)}\n\n"
        f"## Run status\n{_safe_section(_status_md, hub_state)}\n\n"
        f"## Recent openers\n{_safe_section(_recent_openers_md, hub_state)}\n\n"
        f"## Recent opener rejections\n{_safe_section(_recent_opener_rejections_md, hub_state)}\n\n"
        f"## Opener rejections that never reached the store\n{_safe_section(_opener_rejection_deadletter_md, config_path)}\n\n"
        f"## Debug log (on-disk actions + screenshots)\n{_safe_section(_debug_log_md, config_path, hub_state)}\n\n"
        f"## Recent logs\n"
    )
    report = head + _safe_section(_logs_md, _MAX_REPORT_LINES - _line_count(head))
    return _cap_report_lines(_redact_report_output(report))
