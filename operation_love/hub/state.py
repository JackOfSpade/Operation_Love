"""HubState — owns the (at most one) active run and exposes a JSON-able snapshot.

Also tracks live hub-page (browser tab) liveness so the server can shut itself
down once the last browser tab goes away.
"""
from __future__ import annotations

import threading
import time

from .. import config as cfg_mod
from .. import platforms
from .. import supervisor

# Chrome (and others) throttle setInterval in a hidden tab to ~once/minute after 5min hidden —
# very plausible during a real run (owner watching the phone/Playwright window, or the display
# asleep). The stale window has to clear that worst case with margin, or a throttled tab looks
# "gone" and stop()s a live run out from under the owner.
_BROWSER_CLIENT_STALE_S = 120.0
_CLOSED_BROWSER_CLIENT_TTL_S = 30.0
_EVAL_COLD_WAIT_S = 60.0  # bound on a cold-eval waiter so a dead computer thread can't hang it


def _selected_platform(enabled_apps: list[str]) -> "platforms.Platform":
    """Which platform the hub should show pre-selected: the FIRST entry of enabled_apps, if
    it names a real registry id. If enabled_apps is empty or names something the registry
    doesn't know (a stale/hand-edited config.yaml), fall back to the first AVAILABLE
    platform so the hub doesn't default to a selection that would just fail on Start; if
    nothing is available at all, fall back to the first platform overall."""
    first = enabled_apps[0] if enabled_apps else None
    if first in platforms.KNOWN_APPS:
        return platforms.get(first)
    available = [p for p in platforms.all_platforms() if p.available]
    if available:
        return available[0]
    return platforms.all_platforms()[0]


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
        self._opener_service = None         # the running supervisor's OpenerService (recent_openers_snapshot for the bug report)
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
            if apps is not None:
                # Registry guard, before anything else happens: an unavailable platform
                # (uncalibrated Android target, or a web target with no live site behind
                # it) or two Android platforms requested together must never get as far as
                # a thread/driver. Verbatim reason — the hub renders it straight into
                # #hint, which is how "Additional work needed to get this to run..." shows
                # up for the Web-based button with no frontend-side special-casing.
                # (apps=None means "no override, fall back to config" and is checked at
                # supervisor.run() instead, same as the empty-list case above.)
                reason = platforms.check_runnable(apps)
                if reason:
                    return False, reason
            self._stop = threading.Event()
            self._status = None
            self._error = None
            self._live_store = None
            self._opener_service = None
            stop = self._stop

            def _capture(st):
                with self._lock:
                    self._status = st

            def _capture_store(store):
                with self._lock:
                    self._live_store = store

            def _capture_opener_service(svc):
                with self._lock:
                    self._opener_service = svc

            def _target():
                try:
                    supervisor.run(self.config_path, stop_event=stop, on_status=_capture,
                                   on_store=_capture_store,
                                   on_opener_service=_capture_opener_service,
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
                        self._opener_service = None   # same reason: don't read a torn-down object after the supervisor tore the run down

            self._thread = threading.Thread(target=_target, name="hub-run", daemon=True)
            self._thread.start()
            return True, "started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            stop = self._stop
            # Liveness, not merely "an Event object exists": self._stop is assigned in start()
            # and never reset, so after a run ends on its own (deck exhausted, rate limit, an
            # error halt) a gate on `stop is None` alone would keep answering (True,
            # "stopping") forever, i.e. reporting that a stop it never performed had worked.
            if stop is None or not self.is_running():
                # No run has ever started (or the last one already finished and cleared
                # self._stop -- start() only replaces it, never resets it to None itself,
                # but a fresh HubState/never-started hub has it unset). Pre-fix this always
                # returned (True, "stopping") here too, so a stray /api/stop with nothing
                # active looked like it worked.
                return False, "no run is active"
            stop.set()
        # Printed (not just returned in the API response) so Stop lands in the SAME place
        # every OTHER shutdown trigger already announces itself: the tab-close path
        # (server.py's _schedule_shutdown_if_tab_stayed_closed), SIGINT
        # (supervisor.py's _install_signal_handlers), and a Stop-during-startup abort
        # (supervisor.py's _abort_startup) all print a line. Stop via the hub button was the
        # one silent path -- on the terminal AND on the hub's own live-log panel, since
        # bugreport.install_log_capture tees stdout into the ring /api/logs reads, so an
        # operator watching only the browser tab (not a terminal) saw nothing happen for the
        # whole worker-join tail. The decision to actually discard an in-flight swipe is
        # worker.py's, not this line's (see its stop_event re-checks) -- this just says so.
        print("Hub: stop requested -- waiting for the active run to finish what it's doing. "
              "A swipe made from now on will NOT be recorded.")
        return True, "stopping"

    def wait_for_run(self, timeout: float | None = None) -> bool:
        """Block until the active run's thread finishes (or `timeout` elapses).
        Returns True if the thread is no longer alive (already stopped, or just
        finished); False if `timeout` elapsed while it was still running."""
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

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
            stop_pending = self._stop is not None and self._stop.is_set()
        snap = status.snapshot() if status else None
        if snap is not None and running and stop_pending:
            # RunStatus.stopping is written by supervisor.run()'s shutdown finally, which is
            # only reached once the run has fully STARTED. A Stop pressed during startup
            # (loading labels, training the ranker, warming the ML models) is handled by
            # _abort_startup instead and never passes through that finally — so the page saw
            # running=true / stopping=false and kept the Stop button live and the pill on
            # "loading ML models" for the whole warmup, exactly the "did my click do anything?"
            # gap the stopping state exists to close. Derived here, at read time, rather than
            # adding a second writer: the stop Event IS the authority on "a stop is pending",
            # and this needs no clearing logic (running goes false when the thread ends).
            snap["stopping"] = True
        return {
            "running": running,
            "error": error,
            "status": snap,
        }

    def recent_openers(self) -> list[dict]:
        """The openers generated this run, newest LAST, or [] when no run is/was active."""
        with self._lock:
            svc = self._opener_service
        # getattr, not a plain attribute access: a duck-typed/older OpenerService (or a test
        # double that doesn't bother implementing this) must degrade to "no data" rather than
        # crash a bug report — the one caller of this method — over a missing method.
        snapshot_fn = getattr(svc, "recent_openers_snapshot", None)
        if snapshot_fn is None:
            return []
        try:
            return snapshot_fn()
        except Exception:  # noqa: BLE001 — never raise into a bug report; see this method's docstring
            return []

    def recent_opener_rejections(self) -> list[dict]:
        """The REJECTED opener attempts this run, newest LAST, or [] when no run is/was
        active -- same shape and same reasoning as recent_openers above, just reading
        OpenerService.recent_rejections_snapshot instead of recent_openers_snapshot (see
        opener/service.py's recent_rejections docstring in __init__)."""
        with self._lock:
            svc = self._opener_service
        snapshot_fn = getattr(svc, "recent_rejections_snapshot", None)
        if snapshot_fn is None:
            return []
        try:
            return snapshot_fn()
        except Exception:  # noqa: BLE001 — never raise into a bug report; see this method's docstring
            return []

    def config_defaults(self) -> dict:
        try:
            cfg = cfg_mod.load(self.config_path)
            kinds = [
                {
                    "kind": kind,
                    "label": platforms.KIND_LABELS[kind],
                    "platforms": [
                        {"app": p.app, "label": p.label, "available": p.available, "reason": p.reason}
                        for p in platforms.for_kind(kind)
                    ],
                }
                for kind in platforms.kinds()
            ]
            selected = _selected_platform(cfg.enabled_apps)
            return {
                "mode": cfg.mode,
                "backend": cfg.storage.backend,
                "enabled_apps": cfg.enabled_apps,
                "all_apps": [p.app for p in platforms.all_platforms()],
                "kinds": kinds,
                "selected": {"kind": selected.kind, "app": selected.app},
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
            from ..ranker import make_store
            from ..ranker.evaluate import evaluate
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
            from ..ranker import make_store
            from ..ranker.evaluate import quality_trajectory
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
