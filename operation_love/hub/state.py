"""HubState — owns the (at most one) active run and exposes a JSON-able snapshot.

Also tracks live hub-page (browser tab) liveness so the server can shut itself
down once the last browser tab goes away.
"""
from __future__ import annotations

import threading
import time

from .. import config as cfg_mod
from .. import supervisor

_BROWSER_CLIENT_STALE_S = 20.0
_CLOSED_BROWSER_CLIENT_TTL_S = 30.0


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

    def wait_for_run(self) -> None:
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join()

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
