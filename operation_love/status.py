"""Live run status — one thread-safe snapshot the in-page overlay and the hub read.

Workers update their per-app slice each loop (state, last decision, counts); the
supervisor/decider set the global slice (labels, ranker readiness, budget). Pure
Python, no deps, so it imports anywhere and is unit-testable. Read the whole run
with snapshot(), or one app's view (global + that app) with app_view(app).
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class AppStatus:
    app: str
    mode: str = "observe"
    # starting | capturing | waiting | suggesting | waiting_for_send | scoring | acting |
    # out_of_profiles | rate_limited | saving | stopped | error | wedged
    # "suggesting": observe-mode Hinge only -- the worker is blocked inside the (up to
    # opener.request_timeout_s) advisory maybe_opener() call, generating the post-heart
    # comment-sheet suggestion. Published immediately before that call and cleared by the
    # unconditional state transition right after it (success or exception -- see worker.py's
    # _wait_for_observed_decision), so the hub never keeps rendering the stale "click pass X
    # or heart" banner while the operator is actually waiting on a live request.
    state: str = "starting"
    last_decision: str | None = None       # like | pass | dislike | defer | no_face | no_photos
    last_score: float | None = None
    swipes_run: int = 0                     # decisions recorded this run
    # Observe-mode Hinge comment-sheet guidance.  The worker owns this ephemeral value:
    # it publishes the generated text while the human is deciding whether to send it, and
    # clears it as soon as the sheet/card is no longer current.  The hub is the only display.
    opener_suggestion: str | None = None
    error: str | None = None
    # Human-readable cause of a "stopped" outcome that did NOT come from an exception --
    # today, exclusively OpenerService exhausting its opener capacity (run budget reached,
    # provider credit exhausted, or a permanent provider failure) and asking every worker to
    # stop. `error` already covers the exception path (worker.py's HALT-on-unexpected
    # handler); this is the equivalent for a clean stop, so the hub can tell "operator
    # clicked Stop" apart from "every Gemini free-tier model is out of quota" instead of
    # rendering both as a bare "stopped". None for every other terminal state
    # (out_of_profiles/rate_limited/error already explain themselves via `state`/`error`).
    stop_reason: str | None = None
    updated_at: float = field(default_factory=time.time)


class RunStatus:
    """Mutable, lock-guarded run state shared across worker threads + readers."""

    def __init__(self, run_id: str, apps, *, min_labels: int, mode: str,
                 budget_cap: float | None = None, labels: int = 0, ranker_ready: bool = False):
        self._lock = threading.RLock()
        self.run_id = run_id
        self.started_at = time.time()
        self.mode = mode
        self.min_labels = int(min_labels)
        self.budget_cap = budget_cap
        # global slice
        self.phase = "starting"             # starting | loading saved data | training ranker | launching app | live | saving data | stopped
        self.labels = int(labels)
        self.ranker_ready = bool(ranker_ready)
        self.budget_spent = 0.0
        self.openers = 0
        self.running = True
        self._apps: dict[str, AppStatus] = {a: AppStatus(app=a) for a in apps}

    # --- per-app updates (workers) -------------------------------------
    def set_app(self, app: str, **fields) -> None:
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            for k, v in fields.items():
                setattr(s, k, v)
            # A suggestion belongs only to Hinge's still-open comment sheet.  Clearing it
            # with a normal state transition makes the safe default "never show it for the
            # next card", even if a worker's dismissal path does not mention the field.
            # The transition that opens the sheet supplies opener_suggestion explicitly.
            if (fields.get("state") in {"waiting", "capturing", "suggesting", "acting",
                                        "out_of_profiles", "rate_limited", "saving", "stopped",
                                        "error", "wedged"}
                    and "opener_suggestion" not in fields):
                s.opener_suggestion = None
            s.updated_at = time.time()

    def record_swipe(self, app: str, decision: str, score: float | None = None) -> None:
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            s.last_decision = decision
            s.last_score = score
            s.swipes_run += 1
            s.state = "acting"
            s.opener_suggestion = None       # a completed manual decision consumes the sheet
            s.updated_at = time.time()

    # --- global updates (supervisor / decider) -------------------------
    def inc_labels(self, n: int = 1) -> None:
        with self._lock:
            self.labels += n

    def set_global(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(self, k, v)

    # --- reads ----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "uptime_s": round(time.time() - self.started_at, 1),
                "mode": self.mode,
                "phase": self.phase,
                "running": self.running,
                "labels": self.labels,
                "min_labels": self.min_labels,
                "ranker_ready": self.ranker_ready,
                "labels_needed": max(0, self.min_labels - self.labels),
                "budget_spent": round(self.budget_spent, 4),
                "budget_cap": self.budget_cap,
                "openers": self.openers,
                "apps": {a: asdict(s) for a, s in self._apps.items()},
            }

    def app_view(self, app: str) -> dict:
        """snapshot() plus an `app` key with just this app's slice (for the overlay)."""
        with self._lock:
            snap = self.snapshot()
            s = self._apps.get(app)
            snap["app"] = asdict(s) if s else None
            return snap
