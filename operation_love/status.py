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


# The per-card opener suggestion, as a SET, with each field's blank value. Every one of them
# describes one suggestion for one card, so they are published together and cleared together --
# see RunStatus.set_app's clearing net and AppStatus's own field comments. Declared once, here,
# rather than spelled out at each of the four sites that clear them (two in status.py, two in
# worker.py), because the bug this guards against is precisely a site that clears three of them
# and forgets the fourth: an adversarial review found exactly that when the set was three fields
# long, and doc 5.9's inversion took it to seven.
_OPENER_FIELDS: dict[str, object] = {
    "opener_suggestion": None,
    "opener_referenced": None,
    "opener_item": None,
    "opener_media_ordinal": None,
    "opener_item_description": None,
    "opener_warning": None,
    "opener_pending": False,
}


def cleared_opener_fields() -> dict:
    """A fresh dict of every opener field at its blank value, for a caller that must clear them
    EXPLICITLY -- i.e. one that names `opener_suggestion` in its own set_app call and therefore
    opts itself out of the clearing net above (see worker.py's per-card reset and its observe
    loop's `finally`). Returned as a new dict every call so a caller can add fields to it."""
    return dict(_OPENER_FIELDS)


@dataclass
class AppStatus:
    app: str
    mode: str = "observe"
    # starting | capturing | waiting | suggesting | waiting_for_send | scoring | acting |
    # out_of_profiles | rate_limited | saving | stopped | error | wedged | blocked
    # "suggesting": NO LONGER PUBLISHED, as of doc 5.9's observe inversion. It meant "the worker
    # is blocked inside the advisory maybe_opener() call", which was true only while observe
    # generated AFTER the human tapped a heart. Generation now happens on its own thread, before
    # the human acts, so there is nothing to block on and nothing to hold the operator up: what
    # replaced it is `opener_pending` beside a live GO cue. Left in this list because it is still
    # a legal value an older snapshot can carry and the hub still renders it.
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
    # Observe-mode Hinge comment-sheet guidance.  The worker owns this ephemeral value:
    # it publishes the generated text while the human is deciding whether to send it, and
    # clears it as soon as the sheet/card is no longer current.  The hub is the only display.
    opener_suggestion: str | None = None
    # The model's own one-line statement of which profile detail opener_suggestion is about
    # (OpenerPick.referenced, echoed here verbatim) -- shown by the hub under the suggestion
    # text so the operator can instantly see whether the suggestion matches the photo they
    # actually hearted, rather than trusting it blind.
    opener_referenced: str | None = None
    # --- doc 5.9's inverted observe suggestion -----------------------------------------
    # The MODEL ITEM NUMBER the opener was written about (OpenerPick.index, 1-based over the
    # numbered crops the model was sent -- ops/OPENER-REDESIGN.md 5.1/5.7). This is the whole
    # instruction the inversion produces: "like item 3". None when there is no suggestion.
    opener_item: int | None = None
    # A user-facing media number (photos and confirmed videos, never written prompts), set only
    # when the driver has proved every preceding heart-bearing block’s type. Unlike opener_item,
    # this is a display aid; None means the hub must use the visual description alone.
    opener_media_ordinal: int | None = None
    # The model's own short description of that item (OpenerPick.item_description), so the
    # operator can find it on the card without counting hearts. Display only -- nothing branches
    # on it in status; worker.py's separate doc 5.8 coarse type cross-check may use it to
    # withhold an unsafe AUTO action or Observe suggestion before a gesture.
    opener_item_description: str | None = None
    # Why there is NO text to type, when there is none. Set for two different situations that
    # call for the same operator action (type your own words):
    #   * doc 5.9's MISMATCH -- the human opened a different item than the suggestion was written
    #     for, or the sheet is not on this profile at all, or the sheet could not be checked.
    #     "The mismatch must be DETECTED and SURFACED": the hub replaces the opener with this and
    #     offers nothing to type. It is never accompanied by opener_suggestion -- the worker
    #     publishes one or the other, never both.
    #   * NO SUGGESTION AT ALL -- the capture could not be enumerated into numbered items, or the
    #     opener call produced nothing. Observe does not stop for either (that is AUTO's rule),
    #     but going quiet about it is what made the old failure mode invisible.
    opener_warning: str | None = None
    # True while the suggestion for THIS card is still being generated. The operator is not
    # asked to wait for it (doc 5.9: publish READY immediately and let the suggestion fill in
    # behind it), so this is a note beside a live GO cue rather than a WAIT state of its own --
    # which is exactly why it is a flag and not a `state` value. The old "suggesting" state,
    # which DID block the operator, has no producer left in observe.
    opener_pending: bool = False
    # Run-level provenance for Hinge numbered targeting, published once at the start of a run and
    # never cleared per card (it is a property of the RUN, not of one suggestion, so it is
    # deliberately outside _OPENER_FIELDS' clearing net).  Set only when readiness was licensed
    # by the owner's UNMEASURED centered-autoplay assumption rather than by a measured held-out
    # bound -- see targeting_policy.still_photo_licence_operator_notice().  The hub renders it as
    # neutral fine print BESIDE the decision cue, never as a WAIT/warning box: the operator is
    # not being asked to stop, they are being told what licensed the numbering they are seeing.
    targeting_licence_notice: str | None = None
    # The one action that would restore numbered suggestions, published alongside a run-level
    # targeting blocker (targeting_policy.targeting_setup_next_step()).  Run-level like the
    # notice above and equally outside _OPENER_FIELDS' clearing net: the blocker holds for every
    # card of the run, so a per-card clear would blank the guidance mid-run.
    #
    # It is PUBLISHED rather than written into the hub because the correct next step depends on
    # which still-photo licence is installed, which only the process that ran config validation
    # knows.  The hub hardcoded the pre-licence answer and went on giving it after the licence
    # landed -- see targeting_policy.targeting_setup_next_step's comment for what that cost.
    targeting_setup_next_step: str | None = None
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


class RunStatus:
    """Mutable, lock-guarded run state shared across worker threads + readers."""

    def __init__(self, run_id: str, apps, *, min_labels: int, mode: str,
                 budget_cap: float | None = None, labels: int = 0, ranker_ready: bool = False):
        if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
            raise ValueError("run_id must be a nonempty string without surrounding whitespace")
        if not isinstance(mode, str) or mode not in {"observe", "auto"}:
            raise ValueError("mode must be exactly 'observe' or 'auto'")
        if isinstance(apps, (str, bytes)) or not isinstance(apps, Collection):
            raise ValueError("apps must be a finite collection of app-name strings")
        app_names = list(apps)
        if any(not isinstance(app, str) or not app or app != app.strip() for app in app_names):
            raise ValueError(
                "app names must be nonempty strings without surrounding whitespace")
        if len(set(app_names)) != len(app_names):
            raise ValueError("app names must not contain duplicates")
        if type(min_labels) is not int or min_labels < 0:
            raise ValueError("min_labels must be a nonnegative integer")
        if type(labels) is not int or labels < 0:
            raise ValueError("labels must be a nonnegative integer")
        if type(ranker_ready) is not bool:
            raise ValueError("ranker_ready must be exactly bool")
        if budget_cap is None:
            resolved_budget_cap = None
        else:
            if isinstance(budget_cap, bool) or not isinstance(budget_cap, Real):
                raise ValueError("budget_cap must be a finite nonnegative number or None")
            try:
                resolved_budget_cap = float(budget_cap)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "budget_cap must be a finite nonnegative number or None") from exc
            if not math.isfinite(resolved_budget_cap) or resolved_budget_cap < 0:
                raise ValueError("budget_cap must be a finite nonnegative number or None")
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
        # any decision made in that window is discarded (worker.py's observe loop re-checks
        # stop_event and drops the in-flight card rather than recording it). Set True at the
        # same moment stop_event.set() runs (supervisor.py), set False only once the terminal
        # phase is published -- so it covers the join-wait AND the flush/save tail, the whole
        # span during which nothing the operator does on screen will be recorded.
        self.stopping = False
        self._apps: dict[str, AppStatus] = {a: AppStatus(app=a) for a in app_names}

    # --- per-app updates (workers) -------------------------------------
    def set_app(self, app: str, **fields) -> None:
        unknown = set(fields) - _APP_MUTABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown AppStatus field(s): {sorted(unknown)}")
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            for k, v in fields.items():
                setattr(s, k, v)
            # A suggestion belongs only to ONE card. Clearing it with a normal state transition
            # makes the safe default "never show it for the next card", even if a worker's
            # dismissal path does not mention the field. The transition that publishes a
            # suggestion supplies opener_suggestion explicitly and so opts out of this net.
            #
            # EVERY FIELD IN _OPENER_FIELDS IS CLEARED TOGETHER, ALWAYS. They describe ONE
            # suggestion between them -- its text, what it claims to be about, which item it
            # names, that item's description, why there is no text, and whether one is still
            # coming -- so any of them outliving the rest is a lie of exactly the kind this net
            # exists to prevent: a stale "about her beach photo" caption under a new card's
            # suggestion, or (since doc 5.9) a stale "like item 3" instruction pointing at a
            # profile that is no longer on screen. Named as one set here and in worker.py's own
            # explicit clears so a field added to one cannot be forgotten by the other.
            if (fields.get("state") in {"waiting", "capturing", "suggesting", "acting",
                                        "out_of_profiles", "rate_limited", "saving", "stopped",
                                        "error", "wedged", "blocked"}
                    and "opener_suggestion" not in fields):
                for name, blank in _OPENER_FIELDS.items():
                    setattr(s, name, blank)
            s.updated_at = time.time()

    def record_swipe(self, app: str, decision: str, score: float | None = None) -> None:
        with self._lock:
            s = self._apps.setdefault(app, AppStatus(app=app))
            s.last_decision = decision
            s.last_score = score
            s.swipes_run += 1
            s.state = "acting"
            # A completed decision consumes the suggestion, whole. Same set, same rule as
            # set_app's clearing net above: they describe one suggestion for one card.
            for name, blank in _OPENER_FIELDS.items():
                setattr(s, name, blank)
            s.updated_at = time.time()

    # --- global updates (supervisor / decider) -------------------------
    def inc_labels(self, n: int = 1) -> None:
        if type(n) is not int or n < 0:
            raise ValueError("label increment must be a nonnegative integer")
        with self._lock:
            self.labels += n

    def set_global(self, **fields) -> None:
        unknown = set(fields) - _GLOBAL_MUTABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown RunStatus global field(s): {sorted(unknown)}")
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
