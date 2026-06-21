"""One-click bug report — a redacted markdown diagnostic snapshot.

Modeled on infinite-canvas's bug-report feature, adapted to this app. Captures
system + device info, the running build (git commit + a stale-code check), key
dependency versions, the config (secrets stripped — only presence + a short
prefix), the live run status (phase, labels, ranker, per-app decisions, budget,
last error), and recent log lines. Output is markdown the owner can paste to a
developer to debug. The report includes an instruction to improve this
collector when it lacks enough context, and it caps output at 50k lines by
dropping the oldest captured lines first.

The hub serves it at GET /api/bugreport; install_log_capture() (called by the
hub at startup) tees stdout/stderr into a ring buffer so "recent logs" has
content. Pure stdlib so it imports anywhere.
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_MAX_REPORT_LINES = 50_000
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
            "google.cloud.bigquery", "anthropic", "uiautomator2", "sklearn", "yaml"]
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


def _secrets_md() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    shown = f"present ({key[:7]}…)" if key else "unset (needed only for openers / auto mode)"
    return f"- ANTHROPIC_API_KEY: {shown}"      # never the raw value


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
                f"- opener: enabled={c.opener.enabled}, model={c.opener.model}, "
                f"max_tokens={c.opener.max_tokens}\n"
                f"- budget: run_budget_usd={c.budget.run_budget_usd}, "
                f"on_exhausted={c.budget.on_exhausted}")
    except Exception as exc:  # noqa: BLE001
        return f"- ⚠️ could not load `{config_path}`: {exc}"


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
        "",
        "| app | mode | state | last | score | swipes |",
        "|---|---|---|---|---|---|",
    ]
    for name, a in (st.get("apps") or {}).items():
        score = "" if a.get("last_score") is None else f"{a['last_score']:.2f}"
        lines.append(f"| {name} | {a.get('mode','')} | {a.get('state','')} "
                     f"| {a.get('last_decision') or '—'} | {score or '—'} | {a.get('swipes_run',0)} |")
    return "\n".join(lines)


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
    omitted = max(0, len(logs) - available_body_lines)
    prefix: list[str] = []
    if omitted:
        prefix = [
            f"... {omitted} older log line(s) omitted to keep the report under "
            f"{_MAX_REPORT_LINES:,} lines ..."
        ]
        available_body_lines -= len(prefix)

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


# ── assembly ───────────────────────────────────────────────────────────────
def build_report(hub_state=None, description: str = "", config_path: str = "config.yaml") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    desc = (description or "").strip() or "_(none provided)_"
    head = (
        f"# Operation Love — Bug Report\n_Generated {now}_\n\n"
        f"## What happened\n{desc}\n\n"
        f"## Build\n{_build_md()}\n\n"
        f"## System\n{_system_md()}\n\n"
        f"## Dependencies\n{_deps_md()}\n\n"
        f"## Config (config.yaml)\n{_config_md(config_path)}\n\n"
        f"## Secrets (presence only — never raw values)\n{_secrets_md()}\n\n"
        f"## Diagnostic improvement\n{_diagnostic_improvement_md()}\n\n"
        f"## Run status\n{_status_md(hub_state)}\n\n"
        f"## Recent logs\n"
    )
    return _cap_report_lines(head + _logs_md(_MAX_REPORT_LINES - _line_count(head)))
