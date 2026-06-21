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
    # starting | capturing | waiting | scoring | acting | out_of_profiles | rate_limited | saving | stopped | error
    state: str = "starting"
    last_decision: str | None = None       # like | pass | dislike | defer | no_face | no_photos
    last_score: float | None = None
    swipes_run: int = 0                     # decisions recorded this run
    error: str | None = None
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
            s.updated_at = time.time()

    def record_swipe(self, app: str, decision: str, score: float | None = None) -> None:
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            s.last_decision = decision
            s.last_score = score
            s.swipes_run += 1
            s.state = "acting"
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
