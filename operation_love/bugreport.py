"""One-click bug report — a redacted markdown diagnostic snapshot.

Modeled on infinite-canvas's bug-report feature, adapted to this app. Captures
system + device info, the running build (git commit + a stale-code check), key
dependency versions, runtime CAPABILITY presence (the tesseract OCR binary,
opencv, and which /dev/input/event* device the touch watcher would attach
to — see _capabilities_md), the config (secrets stripped — presence only,
never a value or even a derived prefix), the live run status (phase, labels,
ranker, per-app decisions, budget, last error), and recent log lines. Output
is markdown the owner can paste to a developer to debug. The report includes
an instruction to improve this collector when it lacks enough context, and it
caps output at 50k lines by dropping the oldest captured lines first.

The hub serves it at GET /api/bugreport; install_log_capture() (called by the
hub at startup) tees stdout/stderr into a ring buffer so "recent logs" has
content. Pure stdlib so it imports anywhere — the capability probes below
import cv2/touchwatch lazily and swallow their own absence rather than making
that a hard dependency of this module.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

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
    """Tee stdout/stderr into the ring so reports include recent logs. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    sys.stdout = _Tee(sys.stdout)
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
    return (f"- tesseract: {path or 'absent'} — best-effort profile-name OCR (identity "
            f"corroboration only, see hinge.py's _ocr_band) degrades silently to the "
            f"pixel-signature-only identity anchor when absent; never load-bearing")


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


def _app_diagnostics_md(apps: dict) -> str:
    """Per-app stop_reason / error, one bullet each, rendered BELOW the run-status table rather
    than as extra table columns. Both fields are free text (see AppStatus.stop_reason/.error in
    status.py) sourced from provider messages and tracebacks — not guaranteed to be short,
    single-line, or free of `|`. Cramming that into a 6-column table would either wrap
    illegibly or, worse, a literal `|` inside the text would be indistinguishable from a column
    separator and silently corrupt the table's column count. A labelled bullet list has no such
    ambiguity and reads as clearly as the table itself.

    Returns "" (no heading, no bullets) when no app has anything to report — the common healthy
    run must render nothing extra here, not an empty section."""
    lines: list[str] = []
    for name, a in apps.items():
        reason = a.get("stop_reason")
        if reason:
            lines.append(f"- ⚠️ **{name}** stop reason: `{_sanitize_inline(str(reason))}`")
        err = a.get("error")
        if err:
            lines.append(f"- ⚠️ **{name}** error: `{_sanitize_inline(str(err))}`")
    if not lines:
        return ""
    return "\n".join(["", "**Stop reasons / errors:**", *lines])


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
    lines += [
        f"- phase: {st.get('phase')} · mode: {st.get('mode')}",
        f"- labels: {st['labels']} / {st['min_labels']} ({ready})",
        f"- budget: ${st['budget_spent']:.2f}{cap} · openers: {st.get('openers', 0)}",
    ]
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
    apps = st.get("apps") or {}
    for name, a in apps.items():
        score = "" if a.get("last_score") is None else f"{a['last_score']:.2f}"
        lines.append(f"| {name} | {a.get('mode','')} | {a.get('state','')} "
                     f"| {a.get('last_decision') or '—'} | {score or '—'} | {a.get('swipes_run',0)} |")
    diag = _app_diagnostics_md(apps)
    if diag:
        lines.append(diag)
    return "\n".join(lines)


def _recent_openers_md(hub_state) -> str:
    """WHAT the opener said, and whether it was anchored to the like screen it will actually
    hang under -- the diagnostic this report was missing. The bug this collector improvement
    was filed against showed only `openers: 1` in `## Run status`: that count proves a call
    happened, but says nothing about what the model wrote or whether it was told which photo
    the comment attaches to (the anchor mechanism this whole feature is about -- see
    worker.py's _wait_for_observed_decision and opener/service.py's OpenerPick.referenced).

    DATA ROUTE: PRIMARY source is hub_state.recent_openers() -- HubState now captures a live
    reference to the running OpenerService (supervisor.run()'s on_opener_service callback,
    plumbed through hub/state.py's HubState.start()) and this reads its real ring buffer of
    the last several SUCCESSFUL generations (see opener/service.py's `recent_openers` deque
    and OpenerService.recent_openers_snapshot), each with a timestamp, model id, anchored
    flag, the model's own `referenced` claim, which profile item it attaches to (`index`),
    and the opener text itself. That answers the question that broke diagnosis before in
    full: what did the model say, was it anchored to what you actually hearted, and which
    model produced it -- not just a guess from whatever suggestion happens to still be live
    on the hub right now.

    Dropped the old fallback (reading each app's CURRENT live suggestion off
    AppStatus.opener_suggestion / opener_referenced / opener_anchored) rather than keeping it
    as a secondary line: worker.py sets those three fields from the exact same successful
    generate() call that appends to OpenerService's ring buffer (see worker.py's
    `_stat(state="waiting_for_send", opener_suggestion=..., opener_referenced=...,
    opener_anchored=...)`), so the live-status fields are always either identical to the
    newest ring-buffer entry for that app or already stale/cleared -- never something extra.
    Keeping both would just show the same generation twice.

    Best-effort, returns markdown -- may raise (hub_state.recent_openers() is documented
    never to, but this function does not re-guard that promise); wrapped by _safe_section
    like every other section in this file, so a raise here degrades to a warning line, not a
    broken report."""
    if hub_state is None:
        return "- (no hub — run via `python -m operation_love hub` for live run status)"
    entries = hub_state.recent_openers()
    if not entries:
        return "- (no openers generated yet this run, or no run is active)"
    newest_first = list(reversed(entries))[:_RECENT_OPENERS_SHOWN]   # ring buffer is newest-LAST
    lines = []
    for e in newest_first:
        if not isinstance(e, dict):
            continue
        ts = _sanitize_inline(str(e.get("ts") or "unknown time"))
        app = _sanitize_inline(str(e.get("app") or "?"))
        model = _sanitize_inline(str(e.get("model") or "?"))
        anchored = bool(e.get("anchored"))
        advisory = bool(e.get("advisory"))
        index = e.get("index")
        referenced = e.get("referenced")
        referenced = referenced.strip() if isinstance(referenced, str) else ""
        opener = str(e.get("opener") or "")
        if len(opener) > _RECENT_OPENER_TEXT_CHARS:
            opener = opener[:_RECENT_OPENER_TEXT_CHARS] + "…"
        mode_note = "advisory" if advisory else "auto"
        # 🟢/🔴 circles per the owner's status-indicator convention -- never a hand emoji
        # (👍/👎 etc.), which the owner has flagged before as hard to tell apart at a glance.
        anchor_note = ("🟢 anchored to the live like screen" if anchored
                       else "🔴 blind — no like-screen anchor")
        about = f" · about: {_sanitize_inline(referenced)}" if referenced else ""
        lines.append(
            f"- `{ts}` · **{app}** · model: `{model}` · {mode_note} · {anchor_note} · "
            f"index: {index}{about}\n"
            f"  > {_sanitize_inline(opener)}"
        )
    if not lines:
        return "- (no openers generated yet this run, or no run is active)"
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
            counts_line = _action_counts_line(raw_lines)
            if counts_line:
                out.append(f"  - {counts_line}")
            tail = _collapse_action_tail(raw_lines, _DEBUG_ACTION_TAIL)
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
        body_lines += logs[-available_body_lines:]
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
        f"## Build\n{_safe_section(_build_md)}\n\n"
        f"## System\n{_safe_section(_system_md)}\n\n"
        f"## Dependencies\n{_safe_section(_deps_md)}\n\n"
        f"## Capabilities\n{_safe_section(_capabilities_md, config_path)}\n\n"
        f"## Config (config.yaml)\n{_safe_section(_config_md, config_path)}\n\n"
        f"## Secrets (presence only — never raw values)\n{_safe_section(_secrets_md)}\n\n"
        f"## Diagnostic improvement\n{_safe_section(_diagnostic_improvement_md)}\n\n"
        f"## Run status\n{_safe_section(_status_md, hub_state)}\n\n"
        f"## Recent openers\n{_safe_section(_recent_openers_md, hub_state)}\n\n"
        f"## Debug log (on-disk actions + screenshots)\n{_safe_section(_debug_log_md, config_path)}\n\n"
        f"## Recent logs\n"
    )
    return _cap_report_lines(head + _safe_section(_logs_md, _MAX_REPORT_LINES - _line_count(head)))
