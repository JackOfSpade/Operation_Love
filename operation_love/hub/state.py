"""HubState — owns the (at most one) active run and exposes a JSON-able snapshot.

Also tracks live hub-page (browser tab) liveness so the server can shut itself
down once the last browser tab goes away, except at a live Training approval
checkpoint that must remain available for the owner to reopen and decide.
"""
from __future__ import annotations

import copy
import math
import threading
import time

from .. import config as cfg_mod
from .. import platforms
from .. import supervisor
from ..training_actions import TrainingActionBridge

# Chrome (and others) throttle setInterval in a hidden tab to ~once/minute after 5min hidden —
# very plausible during a real run (owner watching the phone or another window, or the display
# asleep). The stale window has to clear that worst case with margin, or a throttled tab looks
# "gone" and stop()s a live run out from under the owner.
_BROWSER_CLIENT_STALE_S = 120.0
_CLOSED_BROWSER_CLIENT_TTL_S = 30.0
_EVAL_COLD_WAIT_S = 60.0  # bound on a cold-eval waiter so a dead computer thread can't hang it


_MAX_BROWSER_CLIENT_ID_LENGTH = 128


def validate_browser_client_id(value: object) -> tuple[bool, str | None, str | None]:
    """Validate the opaque per-tab identifier accepted by the liveness API.

    Browser IDs are dictionary keys for the lifetime of the page. Rejecting malformed and
    needlessly large values prevents unhashable JSON values from becoming handler errors and
    bounds the memory retained by a local client that never closes its tab.
    """
    if not isinstance(value, str) or not value.strip():
        return False, None, "browser client id must be a non-empty string"
    if len(value) > _MAX_BROWSER_CLIENT_ID_LENGTH:
        return False, None, (
            f"browser client id must not exceed {_MAX_BROWSER_CLIENT_ID_LENGTH} characters")
    return True, value, None


def validate_max_per_run(value: object) -> tuple[bool, int | None, str | None]:
    """Normalize the optional Hub run-cap override without coercing malformed JSON.

    ``None`` delegates to config, zero explicitly removes a configured cap for this run,
    and a positive integer sets a temporary cap. Booleans are rejected even though Python
    treats them as integers.
    """
    try:
        return True, supervisor.normalize_max_per_run(value), None
    except ValueError as exc:
        return False, None, str(exc)


def validate_apps(value: object) -> tuple[bool, list[str] | None, str | None]:
    """Validate the optional API app-id list before selection logic calls ``len`` on it."""
    if value is None:
        return True, None, None
    if not isinstance(value, list):
        return False, None, "apps must be null or a list of app ids"
    if any(not isinstance(app, str) or not app.strip() for app in value):
        return False, None, "apps must contain only non-empty app ids"
    if len(set(value)) != len(value):
        return False, None, "apps must not contain duplicate app ids"
    return True, list(value), None


def validate_stop_after_seconds(value: object) -> tuple[bool, int | None, str | None]:
    """Normalize the hub's optional local timed-stop duration.

    ``None`` and zero deliberately mean unlimited.  Do not coerce strings, floats, or
    booleans here: accepting a surprising JSON value for an operation that can stop a
    live run makes a typo look like a successful configuration.
    """
    if value is None:
        return True, None, None
    if type(value) is int and value == 0:
        return True, None, None
    if type(value) is not int or value < 1:
        return False, None, "stop_after_seconds must be null, 0, or a positive integer"
    # Event.wait (used by threading.Timer) rejects an excessively large timeout on some
    # platforms.  Keep API validation deterministic instead of letting a background thread
    # crash after Start appeared to succeed.
    if value > threading.TIMEOUT_MAX:
        return False, None, f"stop_after_seconds must not exceed {int(threading.TIMEOUT_MAX)}"
    return True, value, None


def _selected_platform(enabled_apps: list[str]) -> "platforms.Platform":
    """Which platform the hub should show pre-selected: the FIRST entry of enabled_apps, if
    it names a real registry id. If enabled_apps is empty or names something the registry
    doesn't know (a stale/hand-edited config.yaml), fall back to the first AVAILABLE
    platform so the hub doesn't default to a selection that would just fail on Start; if
    nothing is available at all, fall back to the first platform overall."""
    first = enabled_apps[0] if enabled_apps else None
    if first in platforms.KNOWN_APPS and platforms.get(first).available:
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
        self._run_generation = 0
        self._timed_stop_timer: threading.Timer | None = None
        self._timed_stop_deadline: float | None = None
        self._timed_stop_duration_seconds: int | None = None
        self._status = None                 # RunStatus, captured via on_status
        self._error: str | None = None
        self._eval: dict | None = None      # cached model-quality CV (eval_snapshot)
        self._eval_at: float = 0.0
        self._eval_labels: int | None = None  # label count when _eval was computed (cadence gate)
        self._eval_refreshing = False
        self._eval_cold_event: threading.Event | None = None  # cold-start single-flight
        self._live_store = None             # the running supervisor's store (live in-memory labels)
        self._opener_service = None         # the running supervisor's OpenerService (recent_openers_snapshot for the bug report)
        # Plain, detached copies from the most recently completed run.  The live service owns
        # locks and references to a store which supervisor.run() closes during shutdown, so the
        # hub must never keep that object merely to make a later bug report nicer.  Freeze its
        # two diagnostic rings just before dropping the reference instead.
        self._completed_openers: list[dict] = []
        self._completed_opener_rejections: list[dict] = []
        self._browser_clients: dict[str, float] = {}
        self._closed_browser_clients: dict[str, float] = {}
        self._browser_shutdown_requested = False
        self._browser_stale_watch_active = False
        # Storage mutations run outside this lock because they may involve BigQuery. This flag
        # serializes them with start(), so persisted labels cannot change underneath a newly
        # created in-memory model.
        self._training_data_mutating = False
        self._training_actions = TrainingActionBridge()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, mode: str | None = None, apps=None,
              max_per_run: int | None = None,
              stop_after_seconds: int | None = None) -> tuple[bool, str]:
        valid_cap, max_per_run, cap_error = validate_max_per_run(max_per_run)
        if not valid_cap:
            return False, cap_error or "invalid max-per-run override"
        valid_timeout, stop_after_seconds, timeout_error = validate_stop_after_seconds(
            stop_after_seconds)
        if not valid_timeout:
            return False, timeout_error or "invalid timed-stop duration"
        valid_apps, apps, apps_error = validate_apps(apps)
        if not valid_apps:
            return False, apps_error or "invalid app selection"
        with self._lock:
            if self.is_running():
                return False, "a run is already active"
            if self._training_data_mutating:
                return False, "training data is being updated — wait for it to finish before starting"
            if apps is not None and len(apps) == 0:
                # chosenApps() sends [] when every app checkbox is unchecked. The shared
                # supervisor gate also rejects this, but the Hub can give the owner a clearer
                # selection-specific message. apps=None (no override argument at all) still
                # falls through to config, unchanged.
                return False, "no apps selected — check at least one app before starting"
            try:
                # Validate the complete EFFECTIVE config synchronously. Registry readiness
                # alone is insufficient: a platform can implement Auto while config-level
                # release evidence still deliberately blocks it. This shared helper also
                # resolves apps=None through config.yaml and honors per-app mode overrides,
                # exactly as supervisor.run() does in its direct-caller backstop.
                effective_cfg = supervisor.load_effective_config(
                    self.config_path, mode=mode, enabled_apps=apps)
                # A mode chosen in the Hub is an explicit operator instruction, not merely a
                # suggestion for config.yaml.  Per-app overrides remain useful for unattended
                # CLI/config runs, but silently changing an explicit Hub mode selection is a
                # dangerous control-surface mismatch. Refuse before the
                # background thread/device setup; the operator can select the shown mode or
                # remove the per-app override deliberately.
                if mode is not None and effective_cfg is not None:
                    effective_modes = {
                        app: ((effective_cfg.apps or {}).get(app, {}) or {}).get(
                            "mode", effective_cfg.mode)
                        for app in effective_cfg.enabled_apps
                    }
                    mismatched = {
                        app: actual for app, actual in effective_modes.items() if actual != mode
                    }
                    if mismatched:
                        details = ", ".join(
                            f"{app}={actual}" for app, actual in sorted(mismatched.items()))
                        return False, (
                            f"Hub requested {mode!r}, but per-app mode override(s) resolve to "
                            f"{details}; choose that mode explicitly or remove the override")
            except Exception as exc:  # noqa: BLE001 — configuration/load errors are user-facing
                return False, str(exc)
            # Treat background allocation as a transaction. A completed run's status/error
            # remains the Hub's last diagnostic snapshot unless both the optional timer and
            # the replacement run thread start successfully.
            previous_run = (
                self._run_generation, self._thread, self._stop, self._status, self._error,
                self._live_store, self._opener_service,
            )

            def _restore_failed_start() -> None:
                self._cancel_timed_stop_locked(clear=True)
                (
                    self._run_generation, self._thread, self._stop, self._status, self._error,
                    self._live_store, self._opener_service,
                ) = previous_run

            # No prior timer should survive into a replacement run.  The generation check in
            # its callback is a second guard for a callback that was already queued when it
            # was cancelled.
            self._cancel_timed_stop_locked(clear=True)
            self._run_generation += 1
            generation = self._run_generation
            self._stop = threading.Event()
            self._status = None
            self._error = None
            self._live_store = None
            self._opener_service = None
            stop = self._stop

            if stop_after_seconds is not None:
                self._timed_stop_duration_seconds = stop_after_seconds
                self._timed_stop_deadline = time.monotonic() + stop_after_seconds
                timer = threading.Timer(
                    stop_after_seconds, self._timed_stop_elapsed, args=(generation, stop))
                # A timed stop is strictly a run helper.  It must not keep the local hub
                # process alive after a normal shutdown or a test teardown.
                timer.daemon = True
                self._timed_stop_timer = timer

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
                                   on_worker=self._bind_training_worker,
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
                    # Snapshot before forgetting the live service, but retain only data.  In
                    # particular, do not keep an OpenerService after supervisor has closed its
                    # store: a completed-run report must be useful without reading a torn-down
                    # object (or retaining its locks/client/store).
                    self._freeze_completed_opener_telemetry()
                    with self._lock:
                        if generation == self._run_generation and stop is self._stop:
                            # Completion (including a supervisor-side stop) makes the timer
                            # irrelevant.  Clear it so a completed snapshot is unambiguously
                            # inactive, and so it cannot later wake up a new run.
                            self._cancel_timed_stop_locked(clear=True)
                        self._live_store = None   # supervisor closed it on exit; don't read a dead store
                        self._opener_service = None   # same reason: don't read a torn-down object after the supervisor tore the run down

            self._thread = threading.Thread(target=_target, name="hub-run", daemon=True)
            # Start the optional timer first while holding _lock. Its callback also needs
            # _lock, so it cannot observe a half-started run. If timer allocation/start fails,
            # no worker exists yet. If the worker start then fails, cancellation plus the stop
            # identity reset makes even an already-queued timer callback inert.
            if self._timed_stop_timer is not None:
                try:
                    self._timed_stop_timer.start()
                except Exception as exc:  # noqa: BLE001 — user-facing lifecycle refusal
                    _restore_failed_start()
                    return False, (
                        "could not start timed-stop timer: "
                        f"{type(exc).__name__}: {exc}")
            try:
                self._thread.start()
            except Exception as exc:  # noqa: BLE001 — never leave half-started Hub state
                _restore_failed_start()
                return False, f"could not start run thread: {type(exc).__name__}: {exc}"
            # Only a successfully started run supersedes the last completed run's diagnostic
            # paper trail. While active, reports must not present prior-run opener text as if
            # it belonged to the card currently on screen.
            self._completed_openers = []
            self._completed_opener_rejections = []
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
            # A manual stop wins over the scheduled one.  In addition to cancelling the
            # wait, clearing the status prevents a stale countdown from surviving the
            # shutdown tail.
            self._cancel_timed_stop_locked(clear=True)
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

    def _timed_stop_elapsed(self, generation: int, stop: threading.Event) -> None:
        """Request a stop only for the exact run that created this timer."""
        with self._lock:
            if (generation != self._run_generation or stop is not self._stop
                    or not self.is_running() or stop.is_set()):
                return
            stop.set()
            # Keep the duration/deadline until the run's thread finishes, so the active
            # stopping snapshot can show a truthful zero-second countdown.  The callback is
            # the timer thread itself, so there is nothing left to cancel.
            self._timed_stop_timer = None
            duration = self._timed_stop_duration_seconds
        print("Hub: timed stop reached after "
              f"{duration} seconds -- waiting for the active run to finish what it's doing. "
              "A swipe made from now on will NOT be recorded.")

    def _cancel_timed_stop_locked(self, *, clear: bool) -> None:
        """Cancel the scheduled callback.  Caller must hold ``self._lock``."""
        timer = self._timed_stop_timer
        self._timed_stop_timer = None
        if timer is not None:
            timer.cancel()
        if clear:
            self._timed_stop_deadline = None
            self._timed_stop_duration_seconds = None

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
        valid, client_id, _error = validate_browser_client_id(client_id)
        if not valid or client_id is None:
            return False
        now = time.monotonic()
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
        valid, client_id, _error = validate_browser_client_id(client_id)
        if not valid or client_id is None:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune_closed_browser_clients_locked(now)
            self._browser_clients.pop(client_id, None)
            self._closed_browser_clients[client_id] = now
            if self._browser_clients or self._browser_shutdown_requested:
                return False
            self._browser_stale_watch_active = False
            self._browser_shutdown_requested = True
            return True

    def browser_close_shutdown_disposition(self) -> str:
        """Classify the delayed last-tab-close check without exposing run internals.

        The server calls this after its short reload grace.  Keeping the client check and the
        approval-boundary check together prevents a stale close callback from stopping a run
        after another page has reopened, and makes the decision boundary durable if a preview
        tab is unloaded.  A Training worker deliberately pauses at ``waiting_approval`` with a
        verified composer/checkpoint still live; losing the browser must not turn that pause
        into a Stop because the only safe way forward is for the owner to reopen the Hub and
        choose, or explicitly press Stop.
        """
        with self._lock:
            if self._browser_clients:
                return "client_reopened"
            if not self.is_running() or self._stop is None or self._stop.is_set():
                return "shutdown"

            status = self._status
            if status is not None:
                try:
                    apps = status.snapshot().get("apps", {})
                except Exception:  # noqa: BLE001 -- lifecycle must retain normal shutdown fallback
                    apps = {}
                if not isinstance(apps, dict):
                    apps = {}
                if any(
                    isinstance(app_status, dict)
                    and app_status.get("mode") == "training"
                    and app_status.get("state") == "waiting_approval"
                    for app_status in apps.values()
                ):
                    return "preserve_training_approval"

            # publish_checkpoint() precedes the worker's waiting_approval status update.  The
            # bridge query covers that tiny but safety-critical hand-off window without using a
            # JSON snapshot as a synchronization primitive.
            if self._training_actions.has_actionable_checkpoint():
                return "preserve_training_approval"
            return "shutdown"

    def browser_stale_watch_start_failed(self) -> None:
        """Release the stale-watch claim after its watcher thread could not start.

        The next heartbeat can then retry. Otherwise the live page would be permanently marked
        as watched, while a dropped close beacon could leave the local hub running forever.
        """
        with self._lock:
            self._browser_stale_watch_active = False

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
        now = time.monotonic()
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
            timed_stop_duration = self._timed_stop_duration_seconds if running else None
            timed_stop_deadline = self._timed_stop_deadline if running else None
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
            "timed_stop": (
                {
                    "duration_seconds": timed_stop_duration,
                    "remaining_seconds": max(0, math.ceil(timed_stop_deadline - time.monotonic())),
                }
                if timed_stop_duration is not None and timed_stop_deadline is not None else None
            ),
        }

    def training_action_snapshot(self, *, run_id: str | None = None,
                                 app: str | None = None) -> dict:
        return self._training_actions.snapshot(run_id=run_id, app=app)

    def training_profile_review_image(self, *, run_id: str, app: str,
                                      profile_token: str, index: int) -> bytes | None:
        return self._training_actions.profile_review_image(
            run_id=run_id, app=app, profile_token=profile_token, index=index)

    def submit_training_action(self, body: dict) -> tuple[bool, dict, int]:
        return self._training_actions.submit(body)

    def _training_store_mutation(self, method_name: str) -> tuple[bool, dict | str]:
        """Run an explicit training-set mutation only while no model is live.

        A running supervisor keeps labels and a fitted model in memory.  Altering the
        persisted set underneath it would make the hub claim a reset while that old model
        still decides profiles, so the local control surface refuses until Stop completes.
        """
        with self._lock:
            if self.is_running():
                return False, "stop the active run before changing training data"
            if self._training_data_mutating:
                return False, "a training-data change is already in progress"
            self._training_data_mutating = True
        try:
            from ..ranker import make_store
            cfg = cfg_mod.load(self.config_path)
            store = make_store(cfg)
            try:
                result = getattr(store, method_name)()
            finally:
                store.close()
            with self._lock:
                self._eval = None
                self._eval_at = 0.0
                self._eval_labels = None
            return True, result
        except Exception as exc:  # noqa: BLE001 - surface storage errors to the local operator
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._training_data_mutating = False

    def remove_latest_training_label(self) -> tuple[bool, dict | str]:
        ok, result = self._training_store_mutation("remove_latest_training_label")
        if not ok:
            return False, result
        if result is None:
            return False, "there is no saved training label to remove"
        name = str(result.get("profile_name") or "<name unavailable>")
        print(f"Training label removed for profile: {name}")
        return True, {"profile_name": name}

    def _bind_training_worker(self, worker) -> None:
        # Called by supervisor before Thread.start(), so no hub request can observe a half-bound
        # worker.  The worker subsequently registers itself at run entry as a harmless idempotent
        # backstop for direct/test construction.
        worker.training_action_bridge = self._training_actions
        worker.training_action_supported = bool(
            worker.mode == "training" and worker.app == "hinge"
            and getattr(worker.driver, "supports_training_decision", False))
        self._training_actions.register(worker)

    @staticmethod
    def _service_snapshot(service, method_name: str) -> list[dict]:
        """Return a detached diagnostic ring from ``service``, or [] on any bad hand-off.

        OpenerService itself returns a list while holding its own lock.  ``deepcopy`` here is
        deliberate: its entries contain nested telemetry lists, and the completed-run cache
        must not share any mutable object with an object about to be torn down.
        """
        snapshot_fn = getattr(service, method_name, None)
        if snapshot_fn is None:
            return []
        try:
            rows = snapshot_fn()
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                return []
            return copy.deepcopy(rows)
        except Exception:  # noqa: BLE001 — diagnostics must never break the hub
            return []

    def _freeze_completed_opener_telemetry(self) -> None:
        """Detach final opener/rejection snapshots without retaining the live service."""
        with self._lock:
            service = self._opener_service
        openers = self._service_snapshot(service, "recent_openers_snapshot")
        rejections = self._service_snapshot(service, "recent_rejections_snapshot")
        with self._lock:
            # start() cannot replace a still-running target thread, but retain this identity
            # guard so a future lifecycle change cannot publish one run's rows into another.
            if self._opener_service is service:
                self._completed_openers = openers
                self._completed_opener_rejections = rejections

    def recent_openers(self) -> list[dict]:
        """Committed opener records from the active or most recently completed run.

        A completed run uses the detached snapshot made during HubState's shutdown hand-off;
        this method never reads a torn-down OpenerService.
        """
        with self._lock:
            svc = self._opener_service
            completed = copy.deepcopy(self._completed_openers)
        return self._service_snapshot(svc, "recent_openers_snapshot") if svc is not None else completed

    def recent_opener_rejections(self) -> list[dict]:
        """Committed rejected-opener records from the active or last completed run.

        Same detached-snapshot contract as recent_openers(), using OpenerService's rejection
        ring rather than retaining a shut-down service object.
        """
        with self._lock:
            svc = self._opener_service
            completed = copy.deepcopy(self._completed_opener_rejections)
        return self._service_snapshot(svc, "recent_rejections_snapshot") if svc is not None else completed

    def config_defaults(self) -> dict:
        try:
            cfg = cfg_mod.load(self.config_path)

            def _mode_status(app: str, mode: str) -> tuple[bool, str | None]:
                try:
                    # Use the shared startup gate. A fresh process has no installed Hinge
                    # still-photo licence until config validation reads its accepted
                    # assumption/bound; probing the registry first would reject a valid AUTO
                    # config merely because the picker examined Auto before a validating path.
                    # load_effective_config validates, then invokes the registry before any
                    # construction, so it is the authoritative readiness answer.
                    effective = supervisor.load_effective_config(
                        self.config_path, mode=mode, enabled_apps=[app])
                except Exception as exc:  # noqa: BLE001 — exact start reason is picker help
                    return False, str(exc)
                # ``load_effective_config`` intentionally retains per-app precedence for
                # unattended CLI runs.  The Hub, however, refuses an explicit mode click that
                # would resolve to another mode in start(). Reflect that exact refusal in the
                # picker instead of advertising an option that Start will immediately reject.
                actual = ((getattr(effective, "apps", {}) or {}).get(app, {}) or {}).get(
                    "mode", getattr(effective, "mode", None))
                if actual != mode:
                    return False, (
                        f"Hub requested {mode!r}, but per-app mode override resolves to "
                        f"{app}={actual}; choose that mode explicitly or remove the override")
                return True, None

            def _platform_payload(p: "platforms.Platform") -> dict:
                status = {mode: _mode_status(p.app, mode)
                          for mode in ("training", "auto")}
                return {
                    "app": p.app,
                    "label": p.label,
                    "available": p.available,
                    "reason": p.reason,
                    # Keep the established boolean contract and add diagnostic detail beside it.
                    "modes": {mode: available for mode, (available, _) in status.items()},
                    "mode_reasons": {mode: reason for mode, (_, reason) in status.items()},
                }

            # The hub is an operational control surface, not a calibration console. Do not
            # send unavailable apps to its picker: a dimmed option still looks selectable and
            # made Bumble appear usable before its safety calibration was complete.
            kinds = [
                {
                    "kind": kind,
                    "label": platforms.KIND_LABELS[kind],
                    "platforms": [
                        _platform_payload(p)
                        for p in platforms.for_kind(kind) if p.available
                    ],
                }
                for kind in platforms.kinds()
                if any(p.available for p in platforms.for_kind(kind))
            ]
            pending = [
                {"app": p.app, "label": p.label, "reason": p.reason}
                for p in platforms.all_platforms() if not p.available
            ]
            selected = _selected_platform(cfg.enabled_apps)
            return {
                "mode": cfg.mode,
                "backend": cfg.storage.backend,
                "enabled_apps": cfg.enabled_apps,
                "all_apps": [p.app for p in platforms.all_platforms()],
                "kinds": kinds,
                "pending_platforms": pending,
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
                return self._attach_refresh(cached, every, live, base, status,
                                            training=self._live_training_mix(status))
            if fresh_enough:
                return self._attach_refresh(cached, every, live, base, status,
                                            training=self._live_training_mix(status))

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
                          "roc_auc": None, "pr_auc": None, "brier": None, "base_rate": 0.0,
                          "like_threshold": None, "accepted_recall": None,
                          "false_dislike_rate": None, "confusion": None}
            return self._attach_refresh(cached, every, live, computed_at, status,
                                        training=self._live_training_mix(status))
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

        try:
            threading.Thread(target=_target, name="hub-eval-refresh", daemon=True).start()
        except Exception:  # noqa: BLE001 — preserve eval_snapshot's never-raise contract
            # A failed Thread.start() runs none of _target's finally block.  Clear this guard
            # ourselves so a transient resource failure does not make every later eval request
            # believe a refresh is still running forever.
            with self._lock:
                self._eval_refreshing = False

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
            # This evaluates the ranker's configured base threshold. The stored dataset has
            # labels and embeddings, but not the live profile/session state required to replay
            # AutoSessionPolicy's additional conservative demotions, so this result is useful
            # baseline evidence rather than an AUTO release decision.
            ranker_cfg = getattr(cfg, "ranker", None)
            like_threshold = getattr(ranker_cfg, "like_threshold", 0.5)
            result = evaluate(samples, like_threshold=like_threshold)
            training = self._training_mix(samples)
            # Gate baseline must use the SAME counter the gate compares against: the live
            # swipe counter (status.labels), not len(samples). Using len(samples) would lag
            # status.labels by the worker's unflushed buffer and re-fire the gate every poll.
            computed_at = live if live is not None else len(samples)
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "message": f"{type(exc).__name__}: {exc}",
                      "labels": None, "identities": None, "folds": 0,
                      "roc_auc": None, "pr_auc": None, "brier": None, "base_rate": 0.0,
                      "like_threshold": None, "accepted_recall": None,
                      "false_dislike_rate": None, "confusion": None}
            computed_at = live if live is not None else base
            training = None
        with self._lock:
            self._eval, self._eval_at, self._eval_labels = result, time.time(), computed_at
            status = self._status
        live = getattr(status, "labels", None) if status is not None else None
        return self._attach_refresh(result, every, live, computed_at, status, training=training)

    @staticmethod
    def _training_mix(samples) -> dict:
        """Return current label totals for the hub without running another CV."""
        rows = list(samples or [])
        # Stores always return (liked, embedding) pairs. Keep the display helper tolerant of
        # a malformed/testing row nevertheless: count the record but do not guess its class.
        likes = sum(1 for row in rows if isinstance(row, tuple) and row and bool(row[0]))
        return {"labels": len(rows), "likes": likes, "passes": len(rows) - likes}

    def _live_training_mix(self, status) -> dict | None:
        """Read the running store's cached labels, including its unflushed buffer.

        This keeps the count that confirms a just-made swipe current while the grouped CV
        refresh remains deliberately throttled. BigQueryStore serves this from its in-memory
        cache after startup, so polling does not issue a BigQuery query every five seconds.
        """
        if not bool(getattr(status, "running", False)) or getattr(status, "mode", None) != "training":
            return None
        with self._lock:
            live_store = self._live_store
        if live_store is None:
            return None
        try:
            return self._training_mix(live_store.load_labels())
        except Exception:  # noqa: BLE001 - the run may close the store between poll and read
            return None

    @staticmethod
    def _attach_refresh(result: dict, every: int, live, base, status,
                        training: dict | None = None) -> dict:
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
        attached = {**result, "refresh": refresh}
        if training is not None:
            attached["training"] = training
        return attached
