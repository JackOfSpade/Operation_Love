"""Local control hub — double-click to open, no terminal needed.

Starts a localhost HTTP server and opens your browser to a control panel that
shows live status for every app and starts/stops runs (observe/auto). Pure
stdlib (http.server), so there's no extra dependency. The run executes in a
background thread in THIS process; the page polls /api/status.

    python -m operation_love hub                  # open the hub
    python -m operation_love hub --make-launchers # write a double-click launcher

The browser is only the face: start/stop POST to this local server, which does
the real work (launch the Bumble browser, and — once an AVD exists — boot the
Hinge emulator). UI choice has no bearing on what the backend can do.
"""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config as cfg_mod
from . import supervisor


class HubState:
    """Owns the (at most one) active run and exposes a JSON-able snapshot."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._status = None                 # RunStatus, captured via on_status
        self._error: str | None = None
        self._eval: dict | None = None      # cached model-quality CV (eval_snapshot)
        self._eval_at: float = 0.0
        self._eval_labels: int | None = None  # label count when _eval was computed (cadence gate)
        self._eval_refreshing = False
        self._live_store = None             # the running supervisor's store (live in-memory labels)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, mode: str | None = None, apps=None,
              max_per_run: int | None = None) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "a run is already active"
            self._stop = threading.Event()
            self._status = None
            self._error = None
            self._live_store = None
            stop = self._stop

            def _capture(st):
                with self._lock:
                    self._status = st

            def _capture_store(store):
                with self._lock:
                    self._live_store = store

            def _target():
                try:
                    supervisor.run(self.config_path, stop_event=stop, on_status=_capture,
                                   on_store=_capture_store,
                                   mode=mode, enabled_apps=apps, max_per_run=max_per_run)
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"
                finally:
                    with self._lock:
                        self._live_store = None   # supervisor closed it on exit; don't read a dead store

            self._thread = threading.Thread(target=_target, name="hub-run", daemon=True)
            self._thread.start()
            return True, "started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if self._stop:
                self._stop.set()
        return True, "stopping"

    def snapshot(self) -> dict:
        with self._lock:
            running = self.is_running()
            status = self._status
            error = self._error
        return {
            "running": running,
            "error": error,
            "status": status.snapshot() if status else None,
        }

    def config_defaults(self) -> dict:
        try:
            cfg = cfg_mod.load(self.config_path)
            return {
                "mode": cfg.mode,
                "enabled_apps": cfg.enabled_apps,
                "all_apps": list(cfg.apps.keys()) or cfg.enabled_apps,
                "min_labels": cfg.ranker.min_labels_to_engage,
                "backend": cfg.storage.backend,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def eval_snapshot(self, every: int = 5, max_age: float = 300.0) -> dict:
        """Leakage-free, identity-grouped CV of the ranker, for the GUI's model-quality
        card. Recomputed only when `every` new labels have been recorded since the last
        run (read from the live in-memory label counter — no extra store/BigQuery read
        just to check the gate), with `max_age` seconds as an idle fallback. Once a
        cached card exists, threshold refreshes run in the background so the UI can show
        every/every progress until the new metrics land. Never raises."""
        every = max(1, every)
        with self._lock:
            cached, at, base, status = self._eval, self._eval_at, self._eval_labels, self._status
        live = getattr(status, "labels", None) if status is not None else None

        fresh_enough = cached is not None and (time.time() - at) < max_age
        if live is not None and base is not None:
            enough_new = (live - base) >= every
        else:
            enough_new = cached is None           # no live counter -> lean on max_age
        if cached is not None:
            if enough_new:
                self._start_eval_refresh(every)
                return self._attach_refresh(cached, every, live, base, status)
            if fresh_enough:
                return self._attach_refresh(cached, every, live, base, status)

        return self._compute_eval_snapshot(every)

    def _start_eval_refresh(self, every: int) -> None:
        with self._lock:
            if self._eval_refreshing:
                return
            self._eval_refreshing = True

        def _target():
            try:
                self._compute_eval_snapshot(every)
            finally:
                with self._lock:
                    self._eval_refreshing = False

        threading.Thread(target=_target, name="hub-eval-refresh", daemon=True).start()

    def _compute_eval_snapshot(self, every: int) -> dict:
        with self._lock:
            base, status, live_store = self._eval_labels, self._status, self._live_store
        live = getattr(status, "labels", None) if status is not None else None
        running = self.is_running()
        try:
            from .ranker import make_store
            from .ranker.evaluate import evaluate, marginal_return_summary
            cfg = cfg_mod.load(self.config_path)
            # FRESHNESS: when a run is live, read its in-memory store so the card sees
            # EVERY swipe (incl. the unflushed buffer) — not a re-query of BigQuery, which
            # only returns committed rows and lags behind the streaming buffer. When idle,
            # a fresh read-only store is correct (shutdown already flushed everything).
            samples = None
            if running and live_store is not None:
                try:
                    samples = live_store.load_labels()
                except Exception:  # noqa: BLE001 — run ended + store closed mid-read; fall back
                    samples = None
            if samples is None:
                store = make_store(cfg, ensure=False)   # read-only; don't run DDL just to eval
                try:
                    samples = store.load_labels()
                finally:
                    store.close()
                live_store = None                        # don't reuse a dead store for the chart
            result = evaluate(samples)
            result = {**result, "marginal_return": marginal_return_summary(samples, eval_result=result)}
            result["trajectory"] = self._eval_trajectory(cfg, samples, result, every,
                                                         live_store if running else None)
            # Gate baseline must use the SAME counter the gate compares against: the live
            # swipe counter (status.labels), not len(samples). Using len(samples) would lag
            # status.labels by the worker's unflushed buffer and re-fire the gate every poll.
            computed_at = live if live is not None else len(samples)
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "message": f"{type(exc).__name__}: {exc}",
                      "labels": None, "identities": None, "folds": 0,
                      "roc_auc": None, "pr_auc": None, "brier": None, "base_rate": 0.0,
                      "marginal_return": {
                          "status": "error", "batch": 20, "marginal_return": None,
                          "within_noise": None, "confidence": "low",
                          "message": "diminishing-returns estimate failed",
                      }}
            computed_at = live if live is not None else base
        with self._lock:
            self._eval, self._eval_at, self._eval_labels = result, time.time(), computed_at
            status = self._status
        live = getattr(status, "labels", None) if status is not None else None
        return self._attach_refresh(result, every, live, computed_at, status)

    def _eval_trajectory(self, cfg, samples, result, every: int, live_store=None) -> list:
        """Historical PR-AUC/ROC/Brier curve vs label count, for the hub chart. Reads the
        COMMITTED labels in swipe order and recomputes grouped CV at each prefix; then ties
        the final point to the live full-set result so the curve ends exactly on the card.
        Reuses the running store when given (its ordered read is off the worker lock for
        BigQuery), else opens a fresh read-only store. Never raises — returns [] on any
        problem (incl. a store closed mid-read as a run ends)."""
        try:
            from .ranker import make_store
            from .ranker.evaluate import quality_trajectory
            store = live_store if live_store is not None else make_store(cfg, ensure=False)
            close_after = live_store is None
            try:
                loader = getattr(store, "load_labels_ordered", None)
                ordered = loader() if loader is not None else None
            except Exception:  # noqa: BLE001 — store closed mid-read; chart just sits out a cycle
                ordered = None
            finally:
                if close_after:
                    store.close()
            if not ordered:
                return []
            traj = quality_trajectory(ordered, step=every)
            if result.get("status") == "ok" and result.get("pr_auc"):
                pr = result.get("pr_auc") or [None, None]
                roc = result.get("roc_auc") or [None, None]
                brier = result.get("brier") or [None, None]
                live_point = {"labels": len(samples), "identities": result.get("identities"),
                              "pr_auc": pr[0], "pr_std": pr[1], "roc_auc": roc[0],
                              "brier": brier[0], "base_rate": result.get("base_rate")}
                if traj and traj[-1]["labels"] >= live_point["labels"]:
                    traj[-1] = live_point     # live full-set supersedes the committed tail
                else:
                    traj.append(live_point)
            return traj
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _attach_refresh(result: dict, every: int, live, base, status) -> dict:
        """Return a shallow copy of the cached/fresh eval with a `refresh` countdown
        attached (the cached dict itself stays refresh-free so the count stays live)."""
        every = max(1, every)
        running = bool(getattr(status, "running", False)) if status is not None else False
        mode = getattr(status, "mode", None) if status is not None else None
        if live is None or base is None or not running:
            refresh = {"every": every, "remaining": None, "since": None,
                       "live": False, "mode": mode}
        else:
            since = max(0, int(live) - int(base))
            refresh = {"every": every, "since": since,
                       "remaining": max(0, every - since) or every,
                       "live": True, "mode": mode}
        return {**result, "refresh": refresh}


class _Handler(BaseHTTPRequestHandler):
    state: HubState | None = None           # set by serve()

    def _send(self, code: int, body, ctype: str) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")   # always serve the live build (no stale UI)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, _PAGE, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)             # no icon; avoids a console 404
            self.end_headers()
        elif path == "/api/status":
            self._json(self.state.snapshot())
        elif path == "/api/config":
            self._json(self.state.config_defaults())
        elif path == "/api/eval":
            self._json(self.state.eval_snapshot())
        elif path == "/api/logs":
            from .bugreport import recent_logs   # tee'd stdout/stderr ring (install_log_capture)
            self._json({"lines": recent_logs(300)})
        elif path == "/api/bugreport":
            from .bugreport import build_report
            desc = ""
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse
                desc = (parse_qs(urlparse(self.path).query).get("desc", [""])[0])
            md = build_report(self.state, description=desc)
            self._send(200, md, "text/markdown; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {}
        if self.path == "/api/start":
            mpr = body.get("max_per_run")           # auto-mode per-run cap (0 = unlimited)
            try:
                mpr = int(mpr) if mpr is not None else None
            except (TypeError, ValueError):
                mpr = None
            ok, msg = self.state.start(body.get("mode"), body.get("apps"), mpr)
            self._json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif self.path == "/api/stop":
            ok, msg = self.state.stop()
            self._json({"ok": ok, "msg": msg})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *_):              # silence default stderr request logging
        pass


def _bind(host: str, port: int) -> ThreadingHTTPServer:
    for p in range(port, port + 200):       # find a free port near the default
        try:
            return ThreadingHTTPServer((host, p), _Handler)
        except OSError:
            continue
    raise SystemExit(f"Hub: no free port in {port}..{port + 199}")


def serve(config_path: str = "config.yaml", host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True) -> None:
    from ._warnings import configure_warnings
    from .bugreport import install_log_capture
    configure_warnings()
    install_log_capture()                   # capture logs so bug reports include them
    _Handler.state = HubState(config_path)
    httpd = _bind(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"Hub: Operation Love control hub → {url}   (Ctrl-C to quit)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nHub: shutting down…")
    finally:
        if _Handler.state:
            _Handler.state.stop()
        httpd.shutdown()


# Portable launcher body — written INTO the project folder and committed, so it
# travels with the repo and works on any machine. It resolves the project from
# the script's OWN location (no absolute paths) and uses a project-local .venv.
# __EXTRAS__ is substituted by make_launchers.
#
# The single launcher: install deps only when they change, then launch. The app
# runs from source (editable install), so code changes are live with no rebuild;
# we re-run pip only when pyproject.toml is newer than the last install (stamped
# inside .venv) or there's no .venv yet. Then it execs the hub, so the window
# stays open running the server.
_MAC_UPDATE_RUN = r'''#!/bin/zsh
# Operation Love — set up if needed, then launch, in one double-click. Portable:
# resolves the project from this script's own location, so it works on any
# machine. Dependencies are (re)installed ONLY when they change; otherwise it
# launches straight away. The app runs from source, so there's no build step.
cd "${0:A:h}" || exit 1
notify() { osascript -e "display notification \"$1\" with title \"Operation Love\" sound name \"$2\"" >/dev/null 2>&1; }

PY=".venv/bin/python"
STAMP=".venv/.oplove-deps-stamp"
need_install=0
if [ ! -x "$PY" ]; then
  echo "> Creating project virtualenv (.venv)..."
  python3 -m venv .venv || { echo "x venv creation failed (is python3 installed?)"; notify "Setup failed: venv creation." "Basso"; exit 1; }
  need_install=1
elif [ ! -e "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then
  need_install=1                       # deps changed since the last install
fi

if [ "$need_install" -eq 1 ]; then
  echo "> Installing / refreshing dependencies (first run can take a few minutes)..."
  "$PY" -m pip install -e ".[__EXTRAS__]" || { echo "x pip install failed."; notify "Setup failed at pip install." "Basso"; exit 1; }
  "$PY" -m playwright install chromium >/dev/null 2>&1
  "$PY" -m operation_love.runtime || { echo "x runtime check failed."; notify "Setup: runtime check failed." "Basso"; exit 1; }
  touch "$STAMP"
  notify "Operation Love is ready - launching." "Glass"
else
  echo "> Dependencies already up to date."
fi

echo "OK - launching the control hub (Ctrl-C to quit)..."
exec "$PY" -m operation_love hub
'''

_LINUX_UPDATE_RUN = ('#!/bin/sh\ncd "$(dirname "$0")" || exit 1\n'
                     'PY=".venv/bin/python"\nSTAMP=".venv/.oplove-deps-stamp"\nNEED=0\n'
                     'if [ ! -x "$PY" ]; then python3 -m venv .venv || exit 1; NEED=1\n'
                     'elif [ ! -e "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then NEED=1; fi\n'
                     'if [ "$NEED" -eq 1 ]; then\n'
                     '  "$PY" -m pip install -e ".[__EXTRAS__]" || exit 1\n'
                     '  "$PY" -m playwright install chromium >/dev/null 2>&1\n'
                     '  "$PY" -m operation_love.runtime || exit 1\n'
                     '  touch "$STAMP"\nfi\n'
                     'exec "$PY" -m operation_love hub\n')

_WIN_UPDATE_RUN = ('@echo off\r\ncd /d "%~dp0"\r\n'
                   'set "PY=.venv\\Scripts\\python.exe"\r\n'
                   'set "STAMP=.venv\\.oplove-deps-stamp"\r\n'
                   'set NEED=0\r\n'
                   'if not exist "%PY%" ( python -m venv .venv || exit /b 1 & set NEED=1 ) else (\r\n'
                   '  powershell -NoProfile -Command "if(!(Test-Path \'%STAMP%\') -or (Get-Item \'pyproject.toml\').LastWriteTime -gt (Get-Item \'%STAMP%\').LastWriteTime){exit 1}else{exit 0}"\r\n'
                   '  if errorlevel 1 set NEED=1\r\n'
                   ')\r\n'
                   'if "%NEED%"=="1" (\r\n'
                   '  "%PY%" -m pip install -e ".[__EXTRAS__]" || exit /b 1\r\n'
                   '  "%PY%" -m playwright install chromium\r\n'
                   '  "%PY%" -m operation_love.runtime || exit /b 1\r\n'
                   '  echo ok> "%STAMP%"\r\n'
                   ')\r\n'
                   '"%PY%" -m operation_love hub\r\n')


def make_launchers(config_path: str = "config.yaml", extras: str = "ml,bq,bumble") -> None:
    """Write ONE portable double-click launcher INTO the project folder.

    It resolves the project from the script's own location (no absolute paths)
    and uses a project-local .venv, so the committed file works on any machine.
    The app runs from source, so there's no build step; dependencies are
    (re)installed only when they actually change (pyproject.toml newer than the
    last install, or no .venv yet). It then opens the hub.
    """
    import stat
    import sys
    from pathlib import Path

    proj = Path(config_path).resolve().parent
    plat = sys.platform

    def _write(name: str, body: str, executable: bool) -> None:
        p = proj / name
        p.write_text(body.replace("__EXTRAS__", extras))
        if executable:
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Hub: wrote: {p.name}")

    if plat == "darwin":
        _write("Operation Love.command", _MAC_UPDATE_RUN, True)
    elif plat.startswith("win"):
        _write("Operation Love.bat", _WIN_UPDATE_RUN, False)
    else:  # linux / *bsd
        _write("operation-love.sh", _LINUX_UPDATE_RUN, True)

    print("      One launcher: installs deps only when they change, then opens the hub.")
    print("      Portable across machines (relative paths + project-local .venv).")


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Operation Love — hub</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0f0f14; color:#e8e8ea;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:680px; margin:0 auto; padding:22px 18px 60px; }
  h1 { font-size:18px; margin:0 0 2px; letter-spacing:.3px; }
  .sub { opacity:.55; font-size:12px; margin-bottom:18px; }
  .card { background:#17171d; border:1px solid rgba(255,255,255,.09);
          border-radius:14px; padding:14px 16px; margin-bottom:14px; }
  .row { display:flex; justify-content:space-between; align-items:center; gap:10px; }
  .muted { opacity:.6; } .b { font-weight:600; }
  .pill { font-size:11px; padding:2px 9px; border-radius:999px; }
  .run { background:rgba(57,217,138,.2); color:#39d98a; }
  .stop { background:rgba(255,255,255,.12); color:#c9c9cf; }
  .bar { height:6px; background:rgba(255,255,255,.12); border-radius:4px; margin:8px 0 4px; overflow:hidden; }
  .bar > div { height:100%; border-radius:4px; transition:width .3s; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  select, button { font:inherit; border-radius:9px; border:1px solid rgba(255,255,255,.14);
         background:#22222b; color:#e8e8ea; padding:8px 12px; }
  button { cursor:pointer; }
  button.primary { background:#2f6df0; border-color:#2f6df0; }
  button.danger { background:#b3322f; border-color:#b3322f; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  label.chk { display:inline-flex; align-items:center; gap:6px; padding:6px 10px;
         background:#22222b; border:1px solid rgba(255,255,255,.14); border-radius:9px; cursor:pointer; }
  .err { color:#ff8585; font-size:12px; margin-top:8px; }
  .meta { font-size:12px; opacity:.7; }
</style></head>
<body><div class="wrap">
  <div class="row"><h1>Operation&nbsp;Love</h1><span id="runpill" class="pill stop">stopped</span></div>
  <div class="sub" id="sub">control hub · live status</div>

  <div class="card">
    <div class="row"><span class="muted">budget</span><span id="budget" class="b">—</span></div>
  </div>

  <div class="card">
    <div class="row"><span class="muted">model quality · leakage-free CV</span><span class="meta" id="evalhint"></span></div>
    <div class="meta" id="evalbody" style="margin-top:6px">evaluating…</div>
  </div>

  <div class="card">
    <div class="row"><span class="muted">live log</span><span class="meta" id="loghint"></span></div>
    <pre id="logs" style="max-height:320px;overflow:auto;background:#0c0c10;border:1px solid rgba(255,255,255,.09);border-radius:9px;padding:10px;margin-top:8px;font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;color:#cfcfd6">waiting for output…</pre>
  </div>

  <div class="card">
    <div class="controls">
      <span class="muted">mode</span>
      <select id="mode"><option value="observe">observe (you swipe)</option>
        <option value="auto">auto (bot swipes)</option></select>
      <span id="appchecks"></span>
    </div>
    <div class="controls" id="maxrow" style="margin-top:10px;display:none">
      <span class="muted">max profiles this run</span>
      <input type="text" inputmode="numeric" id="maxrun" value="8"
        style="width:72px;text-align:center;background:#22222b;color:#e8e8ea;border:1px solid rgba(255,255,255,.14);border-radius:7px;padding:5px 7px;font:inherit">
      <label class="chk"><input type="checkbox" id="unlimited"> unlimited</label>
      <span class="meta">(the per-day cap still applies)</span>
    </div>
    <div class="controls" style="margin-top:12px">
      <button id="start" class="primary">▶ Start</button>
      <button id="stop" class="danger" disabled>■ Stop</button>
      <span class="meta" id="hint"></span>
    </div>
    <div class="err" id="err"></div>
  </div>

  <div class="card">
    <div class="row"><span class="muted">report a bug</span><span class="meta" id="bughint"></span></div>
    <textarea id="bugdesc" rows="2" placeholder="What went wrong? (optional)"
      style="width:100%;margin:8px 0;background:#22222b;color:#e8e8ea;border:1px solid rgba(255,255,255,.14);border-radius:9px;padding:8px;font:inherit;resize:vertical"></textarea>
    <div class="controls">
      <button id="bugcopy">📋 Copy report</button>
      <button id="bugdl">⬇ Download .md</button>
    </div>
    <pre id="bugout" style="display:none;max-height:260px;overflow:auto;background:#0c0c10;border:1px solid rgba(255,255,255,.09);border-radius:9px;padding:10px;margin-top:10px;font-size:11px;white-space:pre-wrap"></pre>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
async function getJSON(u){ const r = await fetch(u); return r.json(); }
async function postJSON(u,b){ const r = await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}); return r.json(); }

let cfg = {all_apps:['bumble'], enabled_apps:['bumble'], mode:'observe', min_labels:40};

function renderGlobal(snap){
  const running = snap && snap.running;
  const s = snap && snap.status;
  const phase = s && s.phase;
  $('#runpill').textContent = running ? ((phase && phase!=='live') ? phase : 'running') : 'stopped';
  $('#runpill').className = 'pill ' + (running ? 'run' : 'stop');
  $('#start').disabled = running; $('#stop').disabled = !running;
  if(phase === 'saving data') $('#hint').textContent = 'saving data…';
  else if(!running && $('#hint').textContent === 'saving data…') $('#hint').textContent = '';
  if(s){
    const cap = s.budget_cap!=null ? ' / $'+Number(s.budget_cap).toFixed(2) : '';
    $('#budget').textContent = `$${Number(s.budget_spent).toFixed(2)}${cap}`;
  } else {
    $('#budget').textContent='—';
  }
  $('#err').textContent = (snap && snap.error) ? ('error: '+snap.error) : '';
}

async function tick(){ try { const snap = await getJSON('/api/status'); renderGlobal(snap); } catch(e){} }

function renderEval(e){
  if(!e) return;
  const esc = s => String(s).replace(/[&<>"]/g, ch => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]
  ));
  const gainFmt = v => {
    const x = Math.abs(Number(v));
    if(!Number.isFinite(x) || x === 0) return null;
    const places = Math.max(2, 1 - Math.floor(Math.log10(x)));   // ~2 sig figs, never 0.00
    if(places > 9) return x.toExponential(1);
    return x.toFixed(places).replace(/0+$/,'').replace(/\\.$/,'');
  };
  // One value-only indicator; the magnitude IS the signal (smaller = more diminished).
  const diminishingHtml = m => {
    const g = (m && m.status === 'ok') ? gainFmt(m.marginal_return) : null;
    if(!g) return 'diminishing returns <b>—</b>';
    const b = Number(m.batch) || 20;
    return `diminishing returns <b>${g}</b> <span class="muted">PR-AUC / +${b} labels</span>`;
  };
  const mr = diminishingHtml(e.marginal_return);
  // Quality-over-labels trend: how grouped-CV PR-AUC / ROC-AUC have moved as labels
  // accumulated (reconstructed from swipe order), with the per-point base rate as the floor.
  const trajSvg = traj => {
    if(!Array.isArray(traj)) return '';
    const pts = traj.filter(p => p && p.pr_auc != null && isFinite(p.pr_auc));
    if(pts.length < 2) return '';
    const W=600, H=150, L=8, Rr=8, T=16, B=22;
    const xs = pts.map(p => p.labels);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const vv = [];
    pts.forEach(p => [p.pr_auc, p.roc_auc, p.base_rate].forEach(v => { if(v!=null && isFinite(v)) vv.push(v); }));
    let lo = Math.min(...vv), hi = Math.max(...vv);
    const padv = (hi-lo)*0.18 || 0.05; lo = Math.max(0, lo-padv); hi = Math.min(1, hi+padv);
    const X = v => L + (xmax===xmin ? 0.5 : (v-xmin)/(xmax-xmin)) * (W-L-Rr);
    const Y = v => T + (1-(v-lo)/((hi-lo)||1)) * (H-T-B);
    const poly = (key, color, dash) => {
      const s = pts.filter(p => p[key]!=null && isFinite(p[key]))
                   .map(p => `${X(p.labels).toFixed(1)},${Y(p[key]).toFixed(1)}`).join(' ');
      return s ? `<polyline fill="none" stroke="${color}" stroke-width="1.6"${dash?' stroke-dasharray="3 3" opacity="0.55"':''} points="${s}"/>` : '';
    };
    const txt = (x,y,t,anchor,col) => `<text x="${x}" y="${y}" fill="${col||'#6a6a72'}" font-size="10"${anchor?` text-anchor="${anchor}"`:''}>${t}</text>`;
    const svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block;background:#0c0c10;border:1px solid rgba(255,255,255,.09);border-radius:9px">`
      + poly('base_rate','#6a6a72',true) + poly('roc_auc','#9a9aa2',false) + poly('pr_auc','#5ab0ff',false)
      + txt(L, T-5, hi.toFixed(2)) + txt(L, H-7, lo.toFixed(2))
      + txt(L+18, H-7, xmin+' labels') + txt(W-Rr, H-7, xmax+'', 'end')
      + `</svg>`;
    return `<div style="margin-top:8px"><div class="meta" style="margin-bottom:3px">quality over labels`
      + ` &nbsp;<span style="color:#5ab0ff">PR-AUC</span> · <span style="color:#9a9aa2">ROC-AUC</span> · <span style="color:#6a6a72">base</span></div>`
      + svg + `</div>`;
  };
  // Mini-goal: refresh is gated on new labels (every N), not the clock — so show progress
  // toward the next recompute. Only while a live observe run is feeding labels (auto mode
  // never adds training labels).
  const refreshHtml = () => {
    const r = e.refresh;
    if(!r || !r.live || r.mode !== 'observe' || r.since == null) return '';
    const every = Math.max(1, Math.floor(Number(r.every) || 1));
    const progress = Math.min(every, Math.max(0, Math.floor(Number(r.since) || 0)));
    return `<div class="muted" style="margin-top:4px">↻ ${progress}/${every} till next refresh</div>`;
  };
  if(e.status !== 'ok'){
    $('#evalhint').textContent = (e.identities!=null) ? `${e.identities} identities` : '';
    $('#evalbody').style.color = '#9a9aa2';
    $('#evalbody').innerHTML = `<div>${esc(e.message || 'evaluating…')}</div>`
      + `<div>${mr}</div>` + refreshHtml();
    return;
  }
  const f = m => `${m[0].toFixed(2)}±${m[1].toFixed(2)}`;
  // HIGH bar: green = confidently production-ready, so grade the PESSIMISTIC edge of the
  // ±std band (mean-std, or mean+std for Brier where lower is better). A wide band can't
  // be green no matter how good the point estimate looks.
  const G='#39d98a', A='#f0b429', R='#ff6b6b';
  const tier = (v, good, ok) => v>=good ? G : (v>=ok ? A : R);
  // PR-AUC: lift over the no-skill base rate (precision–recall's "good" scales with
  // prevalence), evaluated at the lower band edge.
  const liftLo = (e.pr_auc[0]-e.pr_auc[1] - e.base_rate) / Math.max(1e-9, 1 - e.base_rate);
  const prCol = tier(liftLo, 0.60, 0.35);
  // ROC-AUC: 0.5 = chance; grade the lower band edge.
  const rocCol = tier(e.roc_auc[0]-e.roc_auc[1], 0.85, 0.72);
  // Brier: lower is better → Brier Skill Score vs the no-skill base·(1-base), at the
  // upper (worst) band edge. Note: class_weight=balanced decalibrates probabilities, so
  // Brier can stay red even when ranking (AUC) is strong.
  const baseBrier = e.base_rate * (1 - e.base_rate);
  const bssLo = (baseBrier - (e.brier[0]+e.brier[1])) / Math.max(1e-9, baseBrier);
  const brierCol = tier(bssLo, 0.35, 0.12);
  $('#evalhint').textContent = `${e.folds}-fold · ${e.identities} identities`;
  $('#evalbody').style.color = '#e8e8ea';
  $('#evalbody').innerHTML =
    `<div>PR-AUC <b style="color:${prCol}">${f(e.pr_auc)}</b>`
    + ` <span class="muted">(base ${e.base_rate.toFixed(2)})</span></div>`
    + `<div>ROC-AUC <b style="color:${rocCol}">${f(e.roc_auc)}</b></div>`
    + `<div>Brier score <b style="color:${brierCol}">${f(e.brier)}</b></div>`
    + `<div>${mr}</div>` + refreshHtml() + trajSvg(e.trajectory);
}
async function tickEval(){ try { renderEval(await getJSON('/api/eval')); } catch(e){} }

function _nearBottom(el){ return el.scrollHeight - el.scrollTop - el.clientHeight < 40; }
async function tickLogs(){
  try {
    const r = await getJSON('/api/logs'); const el = $('#logs');
    const stick = _nearBottom(el);                 // only auto-scroll if user is already at the bottom
    el.textContent = (r.lines||[]).join('\\n');
    $('#loghint').textContent = (r.lines||[]).length + ' lines';
    if(stick) el.scrollTop = el.scrollHeight;
  } catch(e){}
}

function appChecks(){
  $('#appchecks').innerHTML = cfg.all_apps.map(a =>
    `<label class="chk"><input type="checkbox" value="${a}" ${cfg.enabled_apps.includes(a)?'checked':''}>${a}</label>`
  ).join('');
}
function chosenApps(){ return [...document.querySelectorAll('#appchecks input:checked')].map(i=>i.value); }

function syncModeUI(){                         // the cap only applies in auto mode
  $('#maxrow').style.display = $('#mode').value === 'auto' ? '' : 'none';
  const off = $('#unlimited').checked;         // unlimited on -> show ∞, grey out + disable the number box
  const box = $('#maxrun');
  if (off) {
    if (box.value !== '∞') box.dataset.prev = box.value;   // remember the number to restore later
    box.value = '∞';
  } else if (box.value === '∞') {
    box.value = box.dataset.prev || '8';
  }
  box.disabled = off;
  box.style.opacity = off ? '.45' : '1';
  box.style.cursor = off ? 'not-allowed' : 'text';
}
$('#mode').onchange = syncModeUI;
$('#unlimited').onchange = syncModeUI;
$('#start').onclick = async () => {
  $('#hint').textContent='starting…';
  const mode = $('#mode').value;
  let max_per_run = null;                       // null = use config; 0 = unlimited; N = cap
  if (mode === 'auto') max_per_run = $('#unlimited').checked ? 0 : (parseInt($('#maxrun').value,10) || 0);
  const r = await postJSON('/api/start', {mode, apps: chosenApps(), max_per_run});
  $('#hint').textContent = r.ok ? '' : (r.msg||'could not start');
  tick();
};
$('#stop').onclick = async () => { $('#hint').textContent='stopping…'; await postJSON('/api/stop',{}); tick(); };

async function getReport(){          // generate fresh each time (Copy/Download do this implicitly)
  $('#bughint').textContent = 'generating…';
  const r = await fetch('/api/bugreport?desc=' + encodeURIComponent($('#bugdesc').value || ''));
  const md = await r.text();
  $('#bughint').textContent = md.length + ' chars';
  return md;
}
$('#bugcopy').onclick = async () => {
  const md = await getReport();
  try { await navigator.clipboard.writeText(md); $('#bughint').textContent = 'copied ✓'; }
  catch(e){ const out = $('#bugout'); out.textContent = md; out.style.display = 'block';
            $('#bughint').textContent = 'copy blocked — select the text below'; }
};
$('#bugdl').onclick = async () => {
  const md = await getReport();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([md], {type:'text/markdown'}));
  a.download = 'operation-love-bug-report.md'; a.click(); URL.revokeObjectURL(a.href);
  $('#bughint').textContent = 'downloaded ✓';
};

(async () => {
  const c = await getJSON('/api/config');
  if(!c.error){ cfg = Object.assign(cfg, c); $('#mode').value = cfg.mode; $('#sub').textContent = `control hub · storage: ${cfg.backend}`; }
  syncModeUI();
  appChecks();
  tick(); setInterval(tick, 1000);
  tickEval(); setInterval(tickEval, 5000);    // model quality: poll often for live label progress; the CV itself only recomputes every N new labels (server-gated)
  tickLogs(); setInterval(tickLogs, 1500);    // live log: tail the captured stdout/stderr
})();
</script>
</body></html>
"""
