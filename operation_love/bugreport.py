"""One-click bug report — a redacted markdown diagnostic snapshot.

Modeled on infinite-canvas's bug-report feature, adapted to this app. Captures
system + device info, the running build (git commit + a stale-code check), key
dependency versions, runtime CAPABILITY presence (the tesseract OCR binary,
opencv, and which /dev/input/event* device the touch watcher would attach
to — see _capabilities_md), the config (secrets stripped — presence only,
never a value or even a derived prefix), the live run status (phase, labels,
ranker, per-app decisions, budget, last error), a STALL SUMMARY distilled from
each app's on-disk actions.jsonl (the longest same-reason observe_waiting
repeats, worst first — see _stall_summary_md), item-index refusal pair evidence
and realised-step ranges, and recent log lines. Output
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
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
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
_RECENT_REJECTIONS_SHOWN = 10     # cap on _recent_opener_rejections_md rows -- mirrors the above
_STALL_STRETCHES_SHOWN = 3        # cap on _stall_summary_md rows -- see its docstring
_ABANDONED_CARDS_SHOWN = 3        # cap on _abandoned_card_summary_md rows -- see its docstring
_CAPTURE_SPLITS_SHOWN = 3          # most recent split/recovery pairs to show — see below
_MIN_ACTIONABLE_DESCRIPTION_CHARS = 20
_ITEM_INDEX_REASON_INLINE_LIMIT = 360
_LOG_RING: deque[str] = deque(maxlen=_MAX_REPORT_LINES)
_LOG_LOCK = threading.Lock()
_PROCESS_START = time.time()
_REPO = Path(__file__).resolve().parent.parent
_PKG = Path(__file__).resolve().parent
_installed = False


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
    global _installed
    _installed = True
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
                           text=True, timeout=4)
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

    Only Android-driven apps declare `adb_path` at all (bumble_web is a Playwright browser
    and has none), so its presence is what distinguishes an Android app entry from a web one
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


def _diagnostic_improvement_md() -> str:
    return (
        "- If this report is not enough to diagnose the issue, improve "
        "`operation_love/bugreport.py` to capture the missing diagnostics "
        "(state, logs, config/build context, or reproduction details) while "
        "keeping secrets redacted, then include that reporting improvement "
        "with the bug fix."
    )


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
                f"opener.max_attempts={c.opener.max_attempts}, "
                f"opener.advisory_max_attempts={c.opener.advisory_max_attempts}, "
                f"opener.advisory_deadline_s={c.opener.advisory_deadline_s}")
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
        if mode == "observe" and state == "waiting":
            lines.append(f"{prefix}: hub says READY for a manual pass/like; no decision has "
                         "been recorded for the current card.")
        elif mode == "observe" and state == "waiting_for_send":
            lines.append(f"{prefix}: hub says a like/comment sheet is open; waiting for Send "
                         "Like or dismissal, with no decision recorded yet.")

        warning = app.get("opener_warning")
        if warning:
            lines.append(f"{prefix}: ⚠️ no suggestion to type — "
                         f"`{_compact_item_index_refusal_text(warning)}`")
        elif app.get("opener_pending"):
            lines.append(f"{prefix}: advisory suggestion generation is still pending; this does "
                         "not block a manual pass/like.")
        elif app.get("opener_suggestion"):
            item = app.get("opener_item")
            item_note = f" for item {item}" if item is not None else ""
            desc = app.get("opener_item_description")
            desc_note = (f" ({_sanitize_inline(str(desc))})" if desc else "")
            reference = app.get("opener_referenced")
            reference_note = (f" about `{_sanitize_inline(str(reference))}`"
                              if reference else "")
            lines.append(f"{prefix}: advisory suggestion is currently shown{item_note}"
                         f"{desc_note}{reference_note}.")
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
    ready = "ready" if st["ranker_ready"] else "defer"
    cap = f" / ${st['budget_cap']:.2f}" if st.get("budget_cap") is not None else ""
    # These are intentionally separate ledgers.  ``labels`` is the whole ranker's loaded
    # dataset, while AppStatus.swipes_run is the count of actual preference decisions made in
    # THIS run.  CostTracker.calls/spend are provider/billing telemetry: an Observe suggestion
    # can legitimately incur those without the owner sending a like or recording a label.
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
        f"- provider / billing telemetry: {st.get('openers', 0)} model response(s) · "
        f"tracked spend: ${st['budget_spent']:.2f}{cap}",
    ]
    if decisions == 0:
        lines.append("- No pass/like preference decision was recorded. A generated Observe "
                     "suggestion or its provider cost is not a profile, photo, label, or "
                     "decision record.")
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
    both modes now send numbered item crops and the model CHOOSES the item, so the useful fact
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
        return ("- (no committed opener records in the active/last run; an unacted Observe "
                "suggestion is intentionally absent)")
    newest_first = list(reversed(entries))[:_RECENT_OPENERS_SHOWN]   # ring buffer is newest-LAST
    lines = []
    for e in newest_first:
        if not isinstance(e, dict):
            continue
        ts = _sanitize_inline(str(e.get("ts") or "unknown time"))
        app = _sanitize_inline(str(e.get("app") or "?"))
        model = _sanitize_inline(str(e.get("model") or "?"))
        advisory = bool(e.get("advisory"))
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
        mode_note = "advisory" if advisory else "auto"
        # WHAT THE MODEL WAS LOOKING AT: request shape, not a retired live-sheet flag.
        if index_space == "model_items":
            anchor_note = "🟢 chose from numbered item crops"
        else:
            anchor_note = "🔴 no numbered item crops"
        about = f" · about: {_sanitize_inline(referenced)}" if referenced else ""
        space_note = f" ({_sanitize_inline(index_space)})" if index_space else ""
        lines.append(
            f"- `{ts}` · **{app}** · model: `{model}` · {mode_note} · {anchor_note} · "
            f"index: {index}{space_note}{about}\n"
            f"  > {_sanitize_inline(opener)}"
        )
    if not lines:
        return ("- (no committed opener records in the active/last run; an unacted Observe "
                "suggestion is intentionally absent)")
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
        lines.append(
            f"- `{ts}` · **{app}** · model: `{model}` · attempt {attempt} · "
            f"reason: `{reason_code}`\n"
            f"  > {_sanitize_inline(raw_opener)}"
        )
    if not lines:
        return "- (no opener rejections recorded this run, or no run is active)"
    return "\n".join(lines)


def _action_reason_key(raw: str) -> tuple[str, str] | None:
    """(action, reason) for a raw actions.jsonl line, or None if it can't merge with a
    neighbour: invalid JSON, or a record with no "reason" field at all (capture,
    observe_decision, observe_resync, locate_target_heart, like, ... — every action that isn't
    the observe_waiting heartbeat). Only records that match on BOTH fields ever collapse
    together; returning None here is what keeps everything else exactly as raw, individual
    JSON lines. Deliberately swallows every parse failure — a malformed or older-format line
    must pass through untouched rather than raise, same contract as the rest of this best-effort
    collector."""
    try:
        rec = json.loads(raw)
        return (str(rec["action"]), str(rec["reason"]))
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
    for i, (run, key) in enumerate(zip(tail_runs, tail_keys)):
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
            if unavailable:
                parts.append(f"items unavailable: `{_compact_item_index_refusal_text(unavailable)}`")
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
        trailing_saturation = None
        cadence = numbered
        # A final short gesture is the natural signature of a scroll clamping at the card's
        # bottom.  It is not a mid-capture spacing anomaly, so describe it separately rather
        # than letting it poison min/median/max for the ordinary cadence.
        if len(numbered) >= 4 and numbered[-1][0] == len(steps) - 1:
            prior = [value for _i, value in numbered[:-1]]
            ordered_prior = sorted(prior)
            middle_prior = len(ordered_prior) // 2
            prior_median = (ordered_prior[middle_prior] if len(ordered_prior) % 2 else
                            (ordered_prior[middle_prior - 1] + ordered_prior[middle_prior]) / 2)
            if numbered[-1][1] < prior_median * 0.25:
                trailing_saturation = numbered[-1]
                cadence = numbered[:-1]
        cadence_values = [step for _i, step in cadence]
        if cadence_values:
            ordered = sorted(cadence_values)
            middle = len(ordered) // 2
            median = (ordered[middle] if len(ordered) % 2 else
                      (ordered[middle - 1] + ordered[middle]) / 2)
            label = "realised steps" if trailing_saturation is None else "main realised cadence"
            step_text = (f"{label} ({len(cadence_values)} measured): min {ordered[0]:g}px, "
                         f"median {median:g}px, max {ordered[-1]:g}px")
            if trailing_saturation is not None:
                step_text += (f"; trailing scroll saturation at step {trailing_saturation[0]} "
                              f"was {trailing_saturation[1]:g}px")
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


def _item_index_repair_summary_md(lines: list[str]) -> str:
    """Show conservative usable-index repairs, with both source and local frame coordinates."""
    notes: list[str] = []
    structured: list[str] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(rec, dict) or rec.get("action") != "item_index_repaired":
            continue
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
            line = (
                f"{algorithm or 'unknown indexer'}: `{path}` source frames "
                f"{source_pair[0]}→{source_pair[1]}; raw "
                f"{raw_shift.get('status')} {raw_shift.get('delta_px')}px → effective "
                f"{effective.get('status')} {effective.get('delta_px')}px")
            clean = _sanitize_inline(line)
            if clean and clean not in structured:
                structured.append(clean)
        for note in rec.get("notes", ()) if isinstance(rec.get("notes"), list) else ():
            if isinstance(note, str):
                clean = _sanitize_inline(note)
                if clean and clean not in notes:
                    notes.append(clean)
    if not notes and not structured:
        return ""
    shown_structured = structured[:8]
    shown_notes = notes[:max(0, 8 - len(shown_structured))]
    rendered = [f"- {line}" for line in shown_structured] + [f"- `{note}`" for note in shown_notes]
    hidden = len(structured) - len(shown_structured) + len(notes) - len(shown_notes)
    suffix = f"; {hidden} more distinct repair record(s) in actions.jsonl" if hidden else ""
    return "\n".join(rendered) + suffix


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
    attempts = [rec for rec in _action_records(lines)
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
            reason_text = (
                f"; items unavailable: `{_compact_item_index_refusal_text(latest_reason)}`"
                if latest_reason else "; no items_unavailable reason was logged")
            out.append("- ⚠️ " + " ".join(prior_bits) + " — the manifest below belongs to it; "
                       + " ".join(latest_bits) + " had no numbered manifest" + reason_text)
    if isinstance(translation, list):
        out.append("- model item → page heart translation: `"
                   + _sanitize_inline(json.dumps(translation)) + "`")
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
    if rec.get("action") == "capture" and rec.get("item_manifest"):
        if expanded is not None and rec == expanded:
            rec["item_manifest"] = "see item-numbering manifest above"
        else:
            rows = len(rec["item_manifest"]) if isinstance(rec["item_manifest"], list) else 0
            rec["item_manifest"] = (
                f"{rows} manifest row(s) in actions.jsonl; the expanded table above belongs to a "
                f"different capture")
    elif rec.get("action") == "item_index_repaired" and rec.get("notes"):
        rec["notes"] = ["see item-index conservative repairs summary above"]
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


def _observe_waiting_stretches(lines: list[str]) -> list[list[dict]]:
    """Split a run's actions.jsonl into maximal stretches of consecutive records whose
    "action" is "observe_waiting" — i.e. one profile's entire wait for a decision, bounded by
    whatever resolves it (observe_decision, capture, observe_like_anchor, ...) on either side.
    A line that fails to parse, or parses to something with no "observe_waiting" action, closes
    whatever stretch is currently open rather than silently vanishing into it — a malformed
    line must never let two unrelated waits (different profiles, potentially minutes apart)
    merge into one bogus stall. Returns parsed record dicts, not raw strings, since every
    caller needs both "reason" and "ts" out of each one."""
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
        if current:
            stretches.append(current)
            current = []
    if current:
        stretches.append(current)
    return stretches


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


_STALL_ORDINALS = ["longest", "2nd-longest", "3rd-longest"]   # matches _STALL_STRETCHES_SHOWN


def _stall_summary_md(lines: list[str]) -> str:
    """STALL SUMMARY: the top `_STALL_STRETCHES_SHOWN` same-reason observe_waiting repeats in a
    run's actions.jsonl, worst first — see _stall_candidates for exactly what counts as one.
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
        label = _STALL_ORDINALS[i] if i < len(_STALL_ORDINALS) else f"{i + 1}th-longest"
        record_word = "record" if count == 1 else "records"
        out.append(
            f"- {label} observe stall: reason=`{_sanitize_inline(reason)}` for "
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

# The complete reason vocabulary `observe_resync` emits today (there are exactly three call
# sites in hinge.py; grep "observe_resync" there to re-confirm before trusting this list is
# still exhaustive). A reason not in this dict is not swallowed -- see
# _abandoned_card_summary_md -- it renders with its raw string and no explanation, so a reason
# added later reaches the report before anyone remembers to update this dict.
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


def _record_time(rec: dict) -> str:
    return _sanitize_inline(str(rec.get("ts") or "unknown time"))


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
    out = [f"- latest logged action: `{action}` at `{_record_time(latest)}`"]

    if latest.get("action") != "observe_waiting":
        return "\n".join(out)

    # Only the final contiguous observe_waiting records are a still-open wait.  An earlier
    # stretch followed by a decision/capture was resolved and must not be presented as current.
    wait: list[dict] = []
    for rec in reversed(records):
        if rec.get("action") != "observe_waiting":
            break
        wait.append(rec)
    wait.reverse()
    reason = latest.get("reason")
    reason_text = _sanitize_inline(str(reason or "not recorded"))
    explanation = _OBSERVE_WAIT_EXPLANATIONS.get(
        reason if isinstance(reason, str) else "",
        "the driver is still observing and has not logged a proven manual decision",
    )
    out.append(f"- current logged observe state: waiting (`{reason_text}`) — {explanation}.")
    if len(wait) > 1:
        out.append(f"- this unresolved wait began at `{_record_time(wait[0])}` and has "
                   f"{len(wait)} heartbeat record(s), latest at `{_record_time(latest)}`.")
    else:
        out.append("- this is the first logged waiting heartbeat for the unresolved wait.")

    # The immediately preceding capture is the most useful reproduction context: it says what
    # was successfully read before the app entered READY, without inventing a current screen.
    before_wait = records[:len(records) - len(wait)]
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
        detail = "; ".join(bits) if bits else "no capture detail was logged"
        out.append(f"- last capture before this wait (`{_record_time(capture)}`): {detail}.")

    shot = next((latest.get(key) for key in ("before", "after", "screenshot")
                 if latest.get(key)), None)
    if shot:
        shot_text = _sanitize_inline(str(shot))
        availability = "present" if (run / str(shot)).is_file() else "not present (possibly rotated)"
        out.append(f"- screen evidence for that waiting verdict: `{shot_text}` ({availability}).")
    else:
        out.append("- no screenshot filename was recorded with the latest waiting verdict.")
    out.append("- reproduction sequence from the log: capture completed → READY/manual decision "
               "prompt → no pass/like record yet → current observe wait above.")
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
        for earlier, later in zip(stretch, stretch[1:]):
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
    log reuses one file for consecutive identical frames). Any unreadable/missing file is None
    -- a bug report must never raise while describing a bug."""
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
            cache[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001
            cache[name] = None
    return cache[name]


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


def _debug_log_md(config_path: str) -> str:
    """Surface the on-disk action/screenshot debug log (Hinge's silent auto-mode logging) so the
    report points a developer straight at a failure: the latest run folder, the tail of its
    actions.jsonl, and any error screenshots (which are kept un-rotated). Screenshots are binary,
    so we list their paths rather than inline them. Best-effort; never raises."""
    try:
        from . import config as cfg_mod
        apps = cfg_mod.load(config_path).apps
    except Exception as exc:  # noqa: BLE001
        return f"- (could not load config to locate debug logs: {exc})"
    apps = apps if isinstance(apps, dict) else {}
    sections = [_one_debug_dir_md(app, (opts or {}))
                for app, opts in apps.items() if (opts or {}).get("debug_log")]
    if not sections:
        return "- (no app has `debug_log` enabled — nothing on-disk to include)"
    return "\n".join(sections)


def _one_debug_dir_md(app: str, opts: dict) -> str:
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
        out = [f"- **{app}** · latest run: `{run}` · screenshots: {len(pngs)}"
               + (f" · ⚠️ error shots (kept): {', '.join(errors)}" if errors else "")]
        log = run / "actions.jsonl"
        if log.exists():
            try:
                raw_lines = log.read_text().splitlines()
            except Exception:  # noqa: BLE001
                raw_lines = []
            # Stall summary goes FIRST, ahead of the action-counts histogram and the tail --
            # it is the one line a developer needs before anything else if this run hung (see
            # _stall_summary_md's docstring for the incident this is filed against). Rendered
            # as its own nested bullet block only when there's something to say; a healthy run
            # (nothing repeated on the same reason) adds nothing here.
            stall = _stall_summary_md(raw_lines)
            if stall:
                out.append("  - stall summary:")
                out.extend(f"    {line}" for line in stall.splitlines())
            # Names the specific event the 2026-08-15 "it moved on again without waiting for my
            # like or dislike" report needed but never got: the resync record was in the log,
            # but nothing in the report named it (see _abandoned_card_summary_md's docstring).
            # Quiet on a run where every card ended in a proven capture or decision.
            abandoned = _abandoned_card_summary_md(raw_lines)
            if abandoned:
                out.append("  - cards abandoned without a decision (resync):")
                out.extend(f"    {line}" for line in abandoned.splitlines())
            # Beside the stall summary, and for the same reason: these are the anomalies a
            # developer cannot see by reading the tail, because both look like ordinary
            # records until you hash the screenshots. Quiet on a healthy run.
            anomalies = "\n".join(part for part in
                                  (_composer_flap_md(raw_lines), _stale_evidence_md(raw_lines, run))
                                  if part)
            if anomalies:
                out.append("  - evidence anomalies:")
                out.extend(f"    {line}" for line in anomalies.splitlines())
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
            repairs = _item_index_repair_summary_md(raw_lines)
            if repairs:
                out.append("  - item-index conservative repairs:")
                out.extend(f"    {line}" for line in repairs.splitlines())
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
        f"## Secrets (presence only — never raw values)\n{_safe_section(_secrets_md)}\n\n"
        f"## Diagnostic improvement\n{_safe_section(_diagnostic_improvement_md)}\n\n"
        f"## Run status\n{_safe_section(_status_md, hub_state)}\n\n"
        f"## Recent openers\n{_safe_section(_recent_openers_md, hub_state)}\n\n"
        f"## Recent opener rejections\n{_safe_section(_recent_opener_rejections_md, hub_state)}\n\n"
        f"## Debug log (on-disk actions + screenshots)\n{_safe_section(_debug_log_md, config_path)}\n\n"
        f"## Recent logs\n"
    )
    return _cap_report_lines(head + _safe_section(_logs_md, _MAX_REPORT_LINES - _line_count(head)))
