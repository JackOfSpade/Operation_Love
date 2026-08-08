"""Local control hub — double-click to open, no terminal needed.

Starts a localhost HTTP server and opens your browser to a control panel that
shows live status for every app and starts/stops runs (observe/auto). Pure
stdlib (http.server), so there's no extra dependency. The run executes in a
background thread in THIS process; the page polls /api/status.

    python -m operation_love hub                  # open the hub
    python -m operation_love hub --make-launchers # write a double-click launcher

The browser is only the face: start/stop POST to this local server, which does
the real work (launch the Bumble browser and the Hinge ADB session). UI choice
has no bearing on what the backend can do.
"""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config as cfg_mod
from . import supervisor


# Chrome (and others) throttle setInterval in a hidden tab to ~once/minute after 5min hidden —
# very plausible during a real run (owner watching the phone/Playwright window, or the display
# asleep). The stale window has to clear that worst case with margin, or a throttled tab looks
# "gone" and stop()s a live run out from under the owner.
# The grace is a different question: it only runs after an EXPLICIT close beacon, where a reload
# re-registers over localhost in well under a second. It stays short so deliberately closing the
# tab doesn't feel hung — the sleeping-tab case is covered by the stale window above, not here.
_BROWSER_SHUTDOWN_GRACE_S = 3.0
_BROWSER_CLIENT_STALE_S = 120.0
_BROWSER_STALE_CHECK_S = 5.0
_CLOSED_BROWSER_CLIENT_TTL_S = 30.0
# Sane upper bound so a wedged worker (e.g. an ADB/Playwright call with no timeout of its own)
# can't block quit forever. Shared by BOTH shutdown paths — Ctrl-C in serve() and the
# tab-close watchdog below — so they can't drift out of sync again.
_RUN_WAIT_TIMEOUT_S = 90.0
_EVAL_COLD_WAIT_S = 60.0  # bound on a cold-eval waiter so a dead computer thread can't hang it


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
        self._eval_cold_event: threading.Event | None = None  # cold-start single-flight
        self._live_store = None             # the running supervisor's store (live in-memory labels)
        self._browser_clients: dict[str, float] = {}
        self._closed_browser_clients: dict[str, float] = {}
        self._browser_shutdown_requested = False
        self._browser_stale_watch_active = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, mode: str | None = None, apps=None,
              max_per_run: int | None = None) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "a run is already active"
            if apps is not None and len(apps) == 0:
                # chosenApps() sends [] when every app checkbox is unchecked. supervisor.run
                # treats a falsy enabled_apps override as "not overridden" and falls back to
                # config.enabled_apps — so silently starting here would run the apps the user
                # just deselected. Refuse instead; apps=None (no override argument at all)
                # still falls through to config, unchanged.
                return False, "no apps selected — check at least one app before starting"
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
                except (Exception, SystemExit) as exc:  # noqa: BLE001
                    # supervisor.run raises SystemExit (a BaseException, not Exception) for a
                    # startup failure it treats as fatal (e.g. missing cloud deps) — catch it
                    # here too so it surfaces in self._error like any other failure, instead of
                    # threading's default SystemExit handling silently swallowing it. Don't
                    # widen this to bare BaseException: that would also eat KeyboardInterrupt /
                    # genuine interpreter shutdown.
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

    def wait_for_run(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout)

    def browser_client_opened(self, client_id: str | None) -> bool:
        """Mark a hub page as alive. Return True when a stale-client watch should start."""
        if not client_id:
            return False
        now = time.time()
        with self._lock:
            self._prune_closed_browser_clients_locked(now)
            if client_id in self._closed_browser_clients:
                return False
            self._browser_clients[client_id] = now
            self._browser_shutdown_requested = False
            if self._browser_stale_watch_active:
                return False
            self._browser_stale_watch_active = True
            return True

    def browser_client_ping(self, client_id: str | None) -> bool:
        """Heartbeat fallback for browsers that drop the pagehide close beacon."""
        return self.browser_client_opened(client_id)

    def browser_client_closed(self, client_id: str | None) -> bool:
        """Return True once, when the last known hub page has gone away."""
        if not client_id:
            return False
        now = time.time()
        with self._lock:
            self._prune_closed_browser_clients_locked(now)
            self._browser_clients.pop(client_id, None)
            self._closed_browser_clients[client_id] = now
            if self._browser_clients or self._browser_shutdown_requested:
                return False
            self._browser_stale_watch_active = False
            self._browser_shutdown_requested = True
            return True

    def has_browser_clients(self) -> bool:
        with self._lock:
            return bool(self._browser_clients)

    def browser_stale_watch_active(self) -> bool:
        with self._lock:
            return self._browser_stale_watch_active

    def expire_stale_browser_clients(self) -> bool:
        """Drop hub pages that stopped heartbeating.

        Return True once if that leaves no live browser pages and should shut down the hub.
        """
        now = time.time()
        cutoff = now - _BROWSER_CLIENT_STALE_S
        with self._lock:
            self._browser_clients = {
                client_id: seen_at
                for client_id, seen_at in self._browser_clients.items()
                if seen_at >= cutoff
            }
            if self._browser_clients:
                return False
            self._browser_stale_watch_active = False
            if self._browser_shutdown_requested:
                return False
            self._browser_shutdown_requested = True
            return True

    def _prune_closed_browser_clients_locked(self, now: float) -> None:
        cutoff = now - _CLOSED_BROWSER_CLIENT_TTL_S
        for client_id, closed_at in list(self._closed_browser_clients.items()):
            if closed_at < cutoff:
                self._closed_browser_clients.pop(client_id, None)

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

        return self._compute_eval_snapshot_singleflight(every)

    def _compute_eval_snapshot_singleflight(self, every: int) -> dict:
        """Cold path (no cached eval yet — e.g. hub just started): the page polls /api/eval
        every 5s and ThreadingHTTPServer gives every request its own thread, so without a
        guard here N overlapping cold polls each run a full grouped CV (and, on BigQuery,
        each opens its own read-only store). Make concurrent cold callers share ONE
        computation: the first caller computes, everyone else just waits on it and reads
        the result it left behind. Same never-raises contract as eval_snapshot()."""
        with self._lock:
            event = self._eval_cold_event
            if event is None:
                event = self._eval_cold_event = threading.Event()
                is_computer = True
            else:
                is_computer = False
        if not is_computer:
            finished = event.wait(timeout=_EVAL_COLD_WAIT_S)
            with self._lock:
                cached, computed_at, status = self._eval, self._eval_labels, self._status
            live = getattr(status, "labels", None) if status is not None else None
            if not finished or cached is None:
                # Either the wait timed out, or the computing thread died via a BaseException
                # (e.g. SystemExit) that _compute_eval_snapshot's `except Exception` doesn't
                # catch, so it never populated self._eval before its `finally: event.set()`
                # ran. Degrade to the documented error dict instead of handing None to
                # _attach_refresh, which would break eval_snapshot's "never raises" contract.
                cached = {"status": "error", "message": "eval computation did not complete",
                          "labels": None, "identities": None, "folds": 0,
                          "roc_auc": None, "pr_auc": None, "brier": None, "base_rate": 0.0}
            return self._attach_refresh(cached, every, live, computed_at, status)
        try:
            return self._compute_eval_snapshot(every)
        finally:
            event.set()               # wake waiters first...
            with self._lock:
                self._eval_cold_event = None   # ...then clear the slot for the next cold start

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
            from .ranker.evaluate import evaluate
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
            result["trajectory"] = self._eval_trajectory(cfg, samples, result, every,
                                                         live_store if running else None)
            # Gate baseline must use the SAME counter the gate compares against: the live
            # swipe counter (status.labels), not len(samples). Using len(samples) would lag
            # status.labels by the worker's unflushed buffer and re-fire the gate every poll.
            computed_at = live if live is not None else len(samples)
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "message": f"{type(exc).__name__}: {exc}",
                      "labels": None, "identities": None, "folds": 0,
                      "roc_auc": None, "pr_auc": None, "brier": None, "base_rate": 0.0}
            computed_at = live if live is not None else base
        with self._lock:
            self._eval, self._eval_at, self._eval_labels = result, time.time(), computed_at
            status = self._status
        live = getattr(status, "labels", None) if status is not None else None
        return self._attach_refresh(result, every, live, computed_at, status)

    def _eval_trajectory(self, cfg, samples, result, every: int, live_store=None) -> list:
        """Historical ranker accuracy curve vs label count, for the hub chart. Reads the
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
            if result.get("status") == "ok" and result.get("roc_auc"):
                roc = result.get("roc_auc") or [None, None]
                live_point = {"labels": len(samples), "identities": result.get("identities"),
                              "roc_auc": roc[0], "roc_std": roc[1]}
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
            md = build_report(self.state, description=desc, config_path=self.state.config_path)
            self._send(200, md, "text/markdown; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def _schedule_shutdown_if_tab_stayed_closed(self) -> None:
        state = self.state
        server = self.server

        def _target():
            time.sleep(_BROWSER_SHUTDOWN_GRACE_S)  # reload/new tab gets a moment to register
            if state and state.has_browser_clients():
                return
            if state and state.is_running():
                print("Hub: browser hub tab closed; stopping active run before shutdown.")
                state.stop()
                # Bounded the same way as the Ctrl-C path (_RUN_WAIT_TIMEOUT_S): a wedged
                # worker must not block this thread forever, or server.shutdown() below is
                # never reached and the process lives on invisibly after the tab is gone —
                # defeating the "close tab -> hub exits -> launcher closes the Terminal tab" UX.
                state.wait_for_run(timeout=_RUN_WAIT_TIMEOUT_S)
            else:
                print("Hub: browser hub tab closed; shutting down.")
            server.shutdown()

        threading.Thread(target=_target, name="hub-tab-close-shutdown", daemon=True).start()

    def _start_stale_browser_watch(self) -> None:
        state = self.state

        def _target():
            while state and state.browser_stale_watch_active():
                time.sleep(_BROWSER_STALE_CHECK_S)
                if state.expire_stale_browser_clients():
                    self._schedule_shutdown_if_tab_stayed_closed()
                    return

        threading.Thread(target=_target, name="hub-tab-stale-watch", daemon=True).start()

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
        elif self.path == "/api/hub/open":
            start_watch = self.state.browser_client_opened(body.get("id"))
            self._json({"ok": True})
            if start_watch:
                self._start_stale_browser_watch()
        elif self.path == "/api/hub/ping":
            start_watch = self.state.browser_client_ping(body.get("id"))
            self._json({"ok": True})
            if start_watch:
                self._start_stale_browser_watch()
        elif self.path == "/api/hub/closed":
            should_shutdown = self.state.browser_client_closed(body.get("id"))
            self._json({"ok": True})
            if should_shutdown:
                self._schedule_shutdown_if_tab_stayed_closed()
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
        state = _Handler.state
        if state:
            state.stop()
            if state.is_running():
                # Mirror the tab-close path (below): the run thread is daemon=True, so if we
                # return now the interpreter kills it before supervisor.run's finally reaches
                # store.flush()/close() — silently dropping buffered rows and skipping the
                # "data saved" line. Bounded so a wedged worker can't hang quit forever; a
                # second Ctrl-C just stops waiting rather than blowing up out of serve().
                print("Hub: waiting for the run to save…")
                try:
                    state.wait_for_run(timeout=_RUN_WAIT_TIMEOUT_S)
                except KeyboardInterrupt:
                    print("\nHub: quitting without waiting further — data may not be fully saved.")
        httpd.shutdown()


# Portable launcher body — written INTO the project folder and committed, so it
# travels with the repo and works on any machine. It resolves the project from
# the script's OWN location (no absolute paths) and uses a project-local .venv.
# __EXTRAS__ is substituted by make_launchers.
#
# The single launcher: install deps only when they change, then launch. The app
# runs from source (editable install), so code changes are live with no rebuild;
# we re-run pip only when pyproject.toml is newer than the last install (stamped
# inside .venv) or there's no .venv yet. Then it launches the hub; when the
# browser hub tab closes, the hub exits and the launcher closes this Terminal tab.
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
TTY_NAME="$(tty)"
"$PY" -m operation_love hub
status=$?
if [ "$status" -eq 0 ] && [ -n "$TTY_NAME" ]; then
  /usr/bin/nohup /usr/bin/osascript \
    -e 'delay 0.2' \
    -e 'tell application "Terminal"' \
    -e 'repeat with w in windows' \
    -e 'repeat with t in tabs of w' \
    -e "if tty of t is \"$TTY_NAME\" then" \
    -e 'close t' \
    -e 'return' \
    -e 'end if' \
    -e 'end repeat' \
    -e 'end repeat' \
    -e 'end tell' >/dev/null 2>&1 &
fi
exit "$status"
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


def make_launchers(config_path: str = "config.yaml", extras: str = "ml,bq,bumble,hinge") -> None:
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


# The hub page (HTML/CSS/JS) lives in its own file, not inlined here: it gets real
# editor tooling (syntax highlighting, linting, formatting) and, critically, no
# Python-level string escaping — a doubled backslash in a Python string is an easy,
# silent way to corrupt the JS. Do not inline it back.
_PAGE = (Path(__file__).parent / "assets" / "hub.html").read_text(encoding="utf-8")
