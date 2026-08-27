"""Live run status — one thread-safe snapshot the in-page overlay and the hub read.

Workers update their per-app slice each loop (state, last decision, counts); the
supervisor/decider set the global slice (labels, ranker readiness, budget). Pure
Python, no deps, so it imports anywhere and is unit-testable. Read the whole run
with snapshot(), or one app's view (global + that app) with app_view(app).
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from numbers import Real


@dataclass
class AppStatus:
    app: str
    mode: str = "training"
    # starting | waiting_approval | scoring | acting |
    # out_of_profiles | rate_limited | saving | stopped | error | wedged | blocked
    # "blocked": a DIFFERENT terminal state from out_of_profiles -- that one means the deck
    # ran dry (a normal end of supply, nothing wrong); this one means something is ON SCREEN
    # standing BETWEEN us and the deck and the driver can name it (DatingAppDriver.
    # blocked_reason) -- the measured case being Hinge's "out of free likes for today"
    # Hinge+ upgrade screen. A graceful stop, not an error: the phone is in
    # a perfectly normal state, nothing is broken, nothing should be retried, and the screen
    # is left exactly as found -- see worker.py's blocked-deck check for why.
    state: str = "starting"
    last_decision: str | None = None       # like | pass | dislike | defer | no_face | no_photos
    last_score: float | None = None
    swipes_run: int = 0                     # decisions recorded this run
    # Run-level provenance for Hinge numbered targeting, published once at the start of a run and
    # never cleared per card (it is a property of the RUN, not of one profile). Set only when
    # readiness was licensed
    # by the owner's UNMEASURED centered-autoplay assumption rather than by a measured held-out
    # bound -- see targeting_policy.still_photo_licence_operator_notice().  The hub renders it as
    # neutral fine print BESIDE the decision cue, never as a WAIT/warning box: the operator is
    # not being asked to stop, they are being told what licensed the numbering they are seeing.
    targeting_licence_notice: str | None = None
    error: str | None = None
    # Human-readable cause of a "stopped" outcome that did NOT come from an exception --
    # `error` already covers the exception path (worker.py's HALT-on-unexpected handler);
    # this is the equivalent for a clean stop, so the hub can tell "operator clicked Stop"
    # apart from a specific, nameable cause instead of rendering everything as a bare
    # "stopped". None for every other terminal state (out_of_profiles/rate_limited/error
    # already explain themselves via `state`/`error`).
    #
    # THIS FIELD USED TO HAVE EXACTLY ONE SOURCE: OpenerService exhausting its opener
    # capacity (run budget reached, provider credit exhausted, or a permanent provider
    # failure) and asking every worker to stop. That invariant is deliberately broken as of
    # 2026-08-11: worker.py's blocked-deck check (see stop_kind below) now also populates
    # this field, with a driver-supplied reason like Hinge's "out of free likes for today"
    # Hinge+ upgrade screen. A consumer that still
    # assumes "stop_reason set" means "OpenerService" will misreport a blocked deck as a
    # quota/credit problem -- `stop_kind` is what you must branch on now, not the mere
    # presence of a reason string.
    stop_reason: str | None = None
    # Disambiguates stop_reason's SOURCE now that more than one code path populates it:
    # "opener" -- an OpenerService-side stop: either it exhausted its capacity and asked
    #             every worker to stop (the original, sole source of stop_reason -- see its
    #             comment above), or a per-profile opener could not be produced or could not
    #             be targeted (a per-profile OpenerError, a sub-latch failure, or an opener
    #             whose item number has no driver-owned translation into a tappable item yet
    #             -- see worker.py's _auto_loop opener guards).
    # "deck_blocked" -- worker.py's blocked-deck check: the driver's blocked_reason()
    #             reported something on screen standing between us and the deck (Hinge's
    #             out-of-free-likes paywall being the measured case). A graceful stop, not
    #             an error -- see AppStatus.state's "blocked" entry above.
    # "targeting" -- ops/OPENER-REDESIGN.md 5.6's hard stop: an opener WAS produced and the
    #             driver could not put the like on the item it was written about, so the like
    #             was not sent at all (base.ItemTargetingError; worker.py's _auto_loop catches
    #             it around its one driver.like call). Deliberately NOT "opener": nothing
    #             opener-side was exhausted or failed, and the operator's next move is to go
    #             and read the phone, which the driver leaves exactly as it stopped. It was
    #             published as "opener" until this kind existed, which made it render under
    #             the hub's "opener capacity exhausted" title -- a confidently wrong label on
    #             the one stop that exists to prove we never comment on the wrong item.
    # "targeting_calibration" -- numbered targeting was unavailable before an opener request
    #             because the installed calibration was absent or rejected for the live app
    #             build/frame. No opener, device action, or label was produced; the operator's
    #             next step is to capture and validate fresh calibration evidence.
    # "approval" -- the Hub training-decision bridge could not safely publish or complete a
    #             checkpoint, so no physical action or label was issued.
    # None for every stop that isn't one of the above (a manual Stop click,
    # out_of_profiles, rate_limited, defer/cold-start) -- stop_reason is also None in all
    # of those, so there is nothing to disambiguate.
    stop_kind: str | None = None
    updated_at: float = field(default_factory=time.time)


_APP_MUTABLE_FIELDS = frozenset(AppStatus.__dataclass_fields__) - {"app", "updated_at"}
_GLOBAL_MUTABLE_FIELDS = frozenset({
    "mode", "min_labels", "budget_cap", "phase", "labels", "ranker_ready",
    "budget_spent", "openers", "running", "stopping",
})
_RUN_MODES = frozenset({"training", "auto"})
_APP_STATES = frozenset({
    "starting", "waiting_approval", "scoring", "acting", "out_of_profiles",
    "rate_limited", "saving", "stopped", "error", "wedged", "blocked",
})
_DECISIONS = frozenset({"like", "pass", "dislike", "defer", "no_face", "no_photos"})
_STOP_KINDS = frozenset({
    "approval", "opener", "deck_blocked", "targeting", "targeting_calibration",
})


def _finite_nonnegative_float(name: str, value: object, *, none_ok: bool = False) -> float | None:
    """Validate a status value before it reaches the JSON-facing snapshot."""
    if value is None and none_ok:
        return None
    requirement = f"{name} must be a finite nonnegative number" + (" or None" if none_ok else "")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(requirement)
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(requirement) from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(requirement)
    return resolved


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be exactly bool")
    return value


def _run_mode(value: object) -> str:
    if not isinstance(value, str) or value not in _RUN_MODES:
        if value == "auto_testing":
            raise ValueError("mode 'auto_testing' was retired; use 'training' or 'auto'")
        raise ValueError("mode must be exactly 'training' or 'auto'")
    return value


def _optional_text(name: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be text or None")
    return value


class RunStatus:
    """Mutable, lock-guarded run state shared across worker threads + readers."""

    def __init__(self, run_id: str, apps, *, min_labels: int, mode: str,
                 budget_cap: float | None = None, labels: int = 0, ranker_ready: bool = False):
        if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
            raise ValueError("run_id must be a nonempty string without surrounding whitespace")
        mode = _run_mode(mode)
        if isinstance(apps, (str, bytes)) or not isinstance(apps, Collection):
            raise ValueError("apps must be a finite collection of app-name strings")
        app_names = list(apps)
        if any(not isinstance(app, str) or not app or app != app.strip() for app in app_names):
            raise ValueError(
                "app names must be nonempty strings without surrounding whitespace")
        if len(set(app_names)) != len(app_names):
            raise ValueError("app names must not contain duplicates")
        min_labels = _nonnegative_int("min_labels", min_labels)
        labels = _nonnegative_int("labels", labels)
        ranker_ready = _exact_bool("ranker_ready", ranker_ready)
        resolved_budget_cap = _finite_nonnegative_float("budget_cap", budget_cap, none_ok=True)
        self._lock = threading.RLock()
        self.run_id = run_id
        self.started_at = time.time()
        self.mode = mode
        self.min_labels = min_labels
        self.budget_cap = resolved_budget_cap
        # global slice
        self.phase = "starting"             # starting | loading saved data | training ranker | loading ML models | live | stopping | saving data | stopped | wedged | save_failed
        self.labels = labels
        self.ranker_ready = ranker_ready
        self.budget_spent = 0.0
        self.openers = 0
        self.running = True
        # True from the moment run()'s shutdown `finally` sets stop_event until the run
        # reaches its terminal phase (stopped/wedged/save_failed) -- strictly between
        # running and stopped. Exists because "saving data" used to be published for the
        # WHOLE worker-join window (up to join_timeout_s -- 105s with the shipped opener
        # config) while a worker could still be mid-profile-read: the hub could not tell
        # "everything is already flushed" from "still waiting on a worker to notice
        # stop_event", and kept rendering a live-looking GO cue the entire time even though
        # any decision made in that window is discarded. Set True at the
        # same moment stop_event.set() runs (supervisor.py), set False only once the terminal
        # phase is published -- so it covers the join-wait AND the flush/save tail, the whole
        # span during which nothing the operator does on screen will be recorded.
        self.stopping = False
        self._apps: dict[str, AppStatus] = {a: AppStatus(app=a) for a in app_names}

    # --- per-app updates (workers) -------------------------------------
    def set_app(self, app: str, **fields) -> None:
        if not isinstance(app, str) or not app or app != app.strip():
            raise ValueError("app must be a nonempty string without surrounding whitespace")
        unknown = set(fields) - _APP_MUTABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown AppStatus field(s): {sorted(unknown)}")
        normalized = dict(fields)
        if "mode" in normalized:
            normalized["mode"] = _run_mode(normalized["mode"])
        if "state" in normalized and (
                not isinstance(normalized["state"], str)
                or normalized["state"] not in _APP_STATES):
            raise ValueError(f"state must be one of {sorted(_APP_STATES)}")
        if "last_decision" in normalized:
            decision = normalized["last_decision"]
            if decision is not None and (
                    not isinstance(decision, str) or decision not in _DECISIONS):
                raise ValueError(f"last_decision must be one of {sorted(_DECISIONS)} or None")
        if "last_score" in normalized:
            normalized["last_score"] = _finite_nonnegative_float(
                "last_score", normalized["last_score"], none_ok=True)
        if "swipes_run" in normalized:
            normalized["swipes_run"] = _nonnegative_int("swipes_run", normalized["swipes_run"])
        for name in ("targeting_licence_notice", "error", "stop_reason"):
            if name in normalized:
                normalized[name] = _optional_text(name, normalized[name])
        if "stop_kind" in normalized:
            stop_kind = normalized["stop_kind"]
            if stop_kind is not None and stop_kind not in _STOP_KINDS:
                raise ValueError(f"stop_kind must be one of {sorted(_STOP_KINDS)} or None")
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            for k, v in normalized.items():
                setattr(s, k, v)
            s.updated_at = time.time()

    def record_swipe(self, app: str, decision: str, score: float | None = None) -> None:
        if not isinstance(app, str) or not app or app != app.strip():
            raise ValueError("app must be a nonempty string without surrounding whitespace")
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError("decision must be nonempty text")
        resolved_score = _finite_nonnegative_float("score", score, none_ok=True)
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            s.last_decision = decision
            s.last_score = resolved_score
            s.swipes_run += 1
            s.state = "acting"
            s.updated_at = time.time()

    # --- global updates (supervisor / decider) -------------------------
    def inc_labels(self, n: int = 1) -> None:
        n = _nonnegative_int("label increment", n)
        with self._lock:
            self.labels += n

    def set_global(self, **fields) -> None:
        unknown = set(fields) - _GLOBAL_MUTABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown RunStatus global field(s): {sorted(unknown)}")
        normalized = dict(fields)
        for name in ("labels", "min_labels", "openers"):
            if name in normalized:
                normalized[name] = _nonnegative_int(name, normalized[name])
        for name in ("budget_cap", "budget_spent"):
            if name in normalized:
                normalized[name] = _finite_nonnegative_float(
                    name, normalized[name], none_ok=name == "budget_cap")
        for name in ("ranker_ready", "running", "stopping"):
            if name in normalized:
                normalized[name] = _exact_bool(name, normalized[name])
        if "mode" in normalized:
            normalized["mode"] = _run_mode(normalized["mode"])
        if "phase" in normalized:
            if not isinstance(normalized["phase"], str) or not normalized["phase"].strip():
                raise ValueError("phase must be nonempty text")
        with self._lock:
            for k, v in normalized.items():
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
                "stopping": self.stopping,
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
