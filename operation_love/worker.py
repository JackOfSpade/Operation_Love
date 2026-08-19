"""Per-app worker thread, in one of two modes.

observe  — SHADOW LEARNING. You use the app's own controls on real profiles;
           the worker captures each profile, watches your like/pass, embeds it,
           stores it as a label, and retrains the ranker live. No autonomous
           input. This is how the model learns your taste — from your real
           usage, not stock images.
           On Hinge it ALSO runs auto's whole opener pipeline as a canary
           (ops/OPENER-REDESIGN.md 5.9): the capture is enumerated into numbered
           item crops, the model is asked to CHOOSE one and write the opener, and
           the hub conditionally suggests an item and text if YOU choose to
           like — before you touch anything. It is not a recommendation or
           decision; you still tap and still type, and nothing is automated for
           you. If you then open a DIFFERENT item than the one the
           suggestion names, the hub replaces the opener with a warning and
           offers no text: observe honours auto's never-attach-a-comment-to-the-
           wrong-item rule by having nothing to copy, rather than by stopping.
           This is the only way the owner can test what auto will actually
           send, so the two modes must issue the SAME request — see
           _ObserveSuggestion for what is shared and the one thing that is not
           (observe is advisory: a short, time-bounded retry budget, and its failures never
           stop the human observation run).
auto     — AUTONOMOUS. The worker captures, scores with the trained ranker,
           and likes/dislikes itself (sending openers where the app allows).
           On an opener-capable app (Hinge), a like is normally sent WITH its
           opener or not sent at all — if OpenerService cannot produce one for
           a profile (global exhaustion — including every retry attempt for
           that profile failing, see OpenerService.maybe_opener — a
           per-profile OpenerError, or a single sub-latch 400/transient
           failure) the loop stops rather than substitute a bare like, since
           the opener is also what drives Hinge's "commented like" behavior
           signal. It also stops rather than send an opener whose item number
           cannot be turned into a tap on the phone — the model gave no item
           at all, or named one in an index space this build has no
           driver-owned translation table for yet (ops/OPENER-REDESIGN.md
           5.3) — since sending it anyway would mean guessing which item the
           comment attaches to. And it stops one step earlier still when the
           capture could not be enumerated into numbered items at all (doc
           5.2): the request is crops precisely so that image k IS item k, and
           falling back to the raw scroll frames would hand the model a
           numbering nothing downstream can act on. The narrow exception is a
           Gemini safety-policy block: the AUTO ranker already decided to like
           the profile, so that decision proceeds without a comment. This is a
           structured per-call outcome, not a configurable fallback for other
           opener failures. A like whose target cannot be honoured still never
           sends. See _auto_loop's opener guards.
           And it stops one step LATER too, when the driver reports it could
           not put the like on the chosen item after all — it could not reach
           that item, or the sheet that opened was not showing it
           (base.ItemTargetingError, doc 5.6). Never a different item, never
           a rewritten opener: the run stops with intended and actual both
           recorded and the screen left exactly as it is.
           Apps that don't accept openers (Bumble) are unaffected — they
           never call the opener service in auto mode.

Multiple workers run concurrently in one process, sharing the ranker, store, and
global budget. Failures are isolated and auto-restarted with backoff.
"""
from __future__ import annotations

import random
import inspect
import threading
import time
import traceback
import uuid
from datetime import date

from .config import PacingCfg
from .drivers.base import (ActionCancelled, DatingAppDriver, DeckBlockedError, DriverClosed,
                           ItemTargetingError, OBSERVE_ITEM_INCONCLUSIVE, OBSERVE_ITEM_MATCH,
                           OBSERVE_ITEM_MISMATCH, ObserveItemCheck)
from .human import human_cooldown, human_delay
from .human_motion import think_time_s
from .interaction import AutoSessionPolicy
from .opener.opener import INDEX_SPACE_MODEL_ITEMS, ITEM_INDEX_ABSENT, ItemRequest
from .ranker.decider import Decider, Decision
from .status import cleared_opener_fields

_OBSERVE_CAPTURE_BUSY = "Capturing profile — please wait before your next decision"
_OBSERVE_PROCESSING_BUSY = "Processing — please wait before your next decision"
_NO_PHOTO_RETRY_S = 0.5
_PROFILE_LOG_WIDTH = 72
# A reviewed action can refuse before it has touched a heart.  That is not a
# human pass, an app decision, or a worker fault: the card must be retired from
# the action bridge and read again, with no training record.  Keep this private
# sentinel distinct from ``None`` (ordinary manual resync / stop) so the outer
# Observe loop can report the correct cause rather than masking it as a card
# change.
_OBSERVE_PRE_TAP_TARGETING_REFUSED = object()
# think_time_s() is calibrated to real measured Hinge dwell data around this many
# seconds — pacing.swipe_delay_s scales it proportionally, so the config knob still
# speeds up/slows down pacing (and a test fixture's swipe_delay_s=0.0 still paces
# instantly) while think_time_s supplies the measured like-vs-pass asymmetry shape.
# Derived from PacingCfg's own default so the baseline can't drift from the config.
_THINK_TIME_BASELINE_S = PacingCfg().swipe_delay_s


def _operator_items_unavailable_warning(reason: str) -> str:
    """Turn a capture-only index failure into useful Observe-mode guidance.

    The full reason belongs in Hinge's debug log and bug report, where frame numbers, strip
    votes, and geometry are actionable. On the hub it only tells the person to count opaque
    implementation details. A contradictory coordinate space can result from a changing profile
    (commonly an active video) *or* an ambiguous card boundary, so do not present either one as
    a fact. Neither makes the person's manual choice unsafe.
    """
    if "item index this capture produced contradicts itself" in reason:
        return ("this profile's cards could not be reliably counted for an opener from this read. "
                "You can still pass or like manually.")
    return reason


def _accepts_keywords(callback, *names: str) -> bool:
    """Whether a duck-typed persistence seam accepts all named keyword arguments.

    The production stores carry action lineage, while older plugins and deliberately tiny test
    stores implement the long-standing positional protocol.  Do not discover that difference by
    catching a ``TypeError`` from the call itself: a real store may raise TypeError *inside* its
    implementation, which must remain a visible persistence failure after a landed action.
    """
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        # Opaque callables (C extensions/proxies) cannot be inspected.  Prefer the modern
        # protocol; their own call error is then truthful rather than silently discarding lineage.
        return True
    return (any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
            or all(name in {p.name for p in parameters} for name in names))


def _record_decision_with_lineage(store, run_id: str, app: str, decision: str, score: float,
                                  *, source: str, profile_id: str, created_at: float) -> None:
    """Record a landed action, carrying lineage when its store supports the modern protocol."""
    record = store.record_decision
    if _accepts_keywords(record, "profile_id", "created_at"):
        record(run_id, app, decision, score, source=source,
               profile_id=profile_id, created_at=created_at)
    else:
        record(run_id, app, decision, score, source=source)


def _item_type_preflight_mismatch(driver, pick) -> str:
    """Return doc 5.8's mismatch reason, or ``""`` when it is safe to continue.

    The capability is method-presence rather than a base-class abstract method so every existing
    generic driver remains valid.  Its contract is pure: it must only inspect the crop it already
    owns and may not capture, navigate, tap, or otherwise touch the phone.  A broken optional
    implementation is deliberately treated as inconclusive; this early coarse check is an extra
    guard, while the driver-owned post-tap verifier remains the mandatory safety mechanism.
    """
    # The optional driver hook owns NUMBERED ITEM crops.  A legacy profile-photo pick counts
    # raw capture frames instead, so handing its integer to this hook would compare unrelated
    # things and could manufacture a hard stop.  AUTO has a separate explicit legacy targeting
    # path; this coarse model-item guard must stay out of it.
    if getattr(pick, "index_space", None) != INDEX_SPACE_MODEL_ITEMS:
        return ""
    check = getattr(driver, "item_type_preflight", None)
    if not callable(check):
        return ""
    try:
        result = check(pick.item_description, pick.index)
    except Exception:  # noqa: BLE001 — optional early check must fail open to 5.6's verifier
        return ""
    if getattr(result, "mismatch", False):
        return str(getattr(result, "reason", "the crop and description disagree on item type"))
    return ""


def _display_media_ordinal(driver, pick) -> int | None:
    """Return a proved human-countable photo/video number, otherwise no number at all."""
    if getattr(pick, "index_space", None) != INDEX_SPACE_MODEL_ITEMS:
        return None
    resolve = getattr(driver, "model_item_media_ordinal", None)
    if not callable(resolve):
        return None
    try:
        ordinal = resolve(pick.index)
    except Exception:  # noqa: BLE001 -- display advice must not affect suggestion safety
        return None
    return ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal > 0 else None


class _ObserveSuggestion:
    """Doc 5.9's INVERTED observe suggestion, for exactly one profile.

    THE INVERSION, in one sentence: the system chooses an item and opener for an optional like,
    and tells the human before they tap. Until 2026-08-12 observe generated
    AFTER the tap, anchored on the sheet the human had already opened -- which made the
    suggestion right by construction, and made observe a canary for nothing: auto enumerates
    numbered crops and asks the model to CHOOSE, so the two modes issued different images, a
    different closing instruction and a different cognitive task. Testing observe told the owner
    nothing about auto. Now both modes send `ItemRequest` -- same crops, same schema, same prompt
    -- and the only difference left is who moves the phone.

    WHAT THE INVERSION COSTS, AND WHY IT IS PAID HERE. Generating before the tap removes the
    guarantee that the suggestion is about the item the comment will hang under, on the one path
    where a real message reaches a real person. The owner declined to log the human's disagreement
    as training data and declined any added friction, so the rule this class implements is
    narrower and harder: DETECT an affirmative mismatch and SURFACE it. On a confirmed wrong
    item the hub gets a warning and NO text to type. A sheet that cannot yet be measured is not
    affirmative evidence of a wrong item: Hinge reflows the selected-card preview while the
    keyboard opens, so Observe keeps the already-generated advisory text visible and rechecks
    later sheet frames. Never a silent wrong-item opener, never a question for the human, never
    a stop (that is AUTO's answer to the same rule -- observe honours it by refusing to show
    text only after proof of the mismatch).

    THE RACE IS REAL, WHICH IS WHY THIS IS A STATE MACHINE AND NOT A FUNCTION. Generation can take
    up to `opener.request_timeout_s` (90s shipped) and nothing bounds how fast a human who has
    already read the profile may tap. So the two halves -- "the model answered" and "the human
    opened a sheet" -- arrive in EITHER order, from two different threads, and the display is a
    pure function of both. Every transition republishes through one method, so there is no
    ordering a caller has to get right and no second code path for the other order.

    WHY A THREAD AT ALL. Doc 5.9 asks for READY immediately with the suggestion filling in behind
    it, and the alternative is worse than slow: blocking the loop for up to 90s between publishing
    READY and entering `wait_for_decision` means a human who passes in that window is never
    observed at all -- the driver's first frame after the wait starts is already the NEXT card, so
    the decision is lost and the one after it is attributed to the wrong profile. That is a
    corrupted training label, which is the one thing observe mode exists to avoid.

    WHAT THE THREAD MAY TOUCH, stated because it is the whole safety argument. It calls
    `OpenerService.maybe_opener` (serialised on that service's own lock, already shared by
    concurrent app workers), `RunStatus.set_app` (lock-guarded), `print`, and -- only through
    `_publish` under this object's own lock -- the driver's `observe_item_mismatch`, which is pure
    vision over a frame already in hand and touches neither the transport nor the debug log. It
    never captures, never gestures, never types, and never repaints the driver's overlay. `cancel`
    takes the same lock, so once it returns no driver access can be in flight or can start, which
    is what makes it safe for the loop to go on and capture the next profile.
    """

    def __init__(self, worker: "Worker", profile):
        self._worker = worker
        self._profile = profile
        # RLock, not Lock: _publish is reached both directly and from the mutators below, and a
        # re-entrant call is easier to allow than to prove impossible.
        self._lock = threading.RLock()
        self._pick = None          # the OpenerPick, once the model has answered
        self._warning = None       # why there is no text to type, when there is none
        self._sheet = None         # the like-sheet frame the human opened, once they have
        # Three-valued check latched within one uninterrupted sheet session. A later
        # INCONCLUSIVE typing frame preserves MATCH/MISMATCH; only affirmative evidence may
        # change an undecided state. The item number binds the proof to the pick it licensed.
        self._sheet_item_check: tuple[int, ObserveItemCheck] | None = None
        self._pending = False      # a generation call is in flight
        self._cancelled = False
        # This is deliberately separate from `_cancelled`/`_lock`: a provider call can be
        # waiting on OpenerService's shared budget lock while the human has already left this
        # card.  It needs a cheap predicate the service can observe BEFORE it spends a provider
        # slot, without waiting for this object's display lock.
        self._cancel_event = threading.Event()
        self._announced = None     # the last warning printed, so a republish does not repeat it
        self._thread: threading.Thread | None = None
        self._release_pre_tap_recorded = False
        self._release_post_tap_recorded = False

    # --- the two inputs -------------------------------------------------
    def start(self) -> "_ObserveSuggestion":
        """Publish the card's opener state and, when there is one to make, start the call.

        Returns self so the loop reads as one statement. Every refusal below is a WARNING on the
        hub rather than silence: "no suggestion" and "a suggestion that must not be typed" are
        different facts, and an operator who cannot tell them apart learns to ignore both.

        The ONE thing that is not a warning is an app or a run that has no opener feature at all
        (Bumble, or `opener.enabled: false`). Nothing was expected there, so nothing is published
        and the hub shows its ordinary GO cue -- a red "no suggestion to type" box on every card
        of a run that was never going to have one is the kind of standing warning that teaches an
        operator to stop reading warnings.
        """
        if not self._applicable():
            return self
        reason = self._blocker()
        if reason:
            with self._lock:
                self._warning = reason
            self._publish()
            return self
        with self._lock:
            self._pending = True
        self._publish()
        self._thread = threading.Thread(target=self._generate, name=f"suggest-{self._worker.app}",
                                        daemon=True)
        self._thread.start()
        return self

    def sheet_opened(self, frame: bytes | None) -> None:
        """The human opened the app's own comment sheet on some item. `frame` is that screen.

        A None frame is not a lesser version of this: it remains an inconclusive item check, not
        a claim that the suggestion matches. Unlike a confirmed mismatch it does not hide advice:
        the human already has a generated optional opener and Hinge may publish a measurable
        preview on the next sheet frame.
        """
        with self._lock:
            if self._sheet is None:
                # First frame of a newly opened sheet. Any proof belonged to the previous
                # continuous composer session and must not cross a close/reopen boundary.
                self._sheet_item_check = None
            self._sheet = frame if frame else b""
        self._publish()

    def sheet_closed(self) -> None:
        """The sheet went away without a send (dismissed, or resolved). Back to the instruction.

        The PICK is deliberately kept: the card has not changed, so the optional item/text help
        is still current if the human decides to like. Only the evidence about what they opened
        is dropped.
        """
        with self._lock:
            self._sheet = None
            self._sheet_item_check = None
        self._publish()

    def cancel(self) -> None:
        """Stop publishing for this card. Called on every exit from the profile's wait.

        Takes the lock, so it cannot return while a `_publish` (and therefore a driver call) is
        in flight -- which is the property the loop relies on before it captures the next profile
        and invalidates the driver's item table underneath us. The thread itself is NOT joined: it
        may be blocked in a provider call for up to `request_timeout_s`, and waiting for that is
        the very stall this whole design exists to avoid. It is a daemon, it holds nothing, and
        its only remaining act is a `_publish` that returns immediately.
        """
        # Set this first.  A generation blocked behind another card's serialized opener call
        # can then return at `maybe_opener`'s before-attempt stop check instead of issuing a
        # stale, billable request merely because it has not yet acquired our display lock.
        self._cancel_event.set()
        with self._lock:
            self._cancelled = True

    def commit_if_liked(self, *, profile_id: str, decision_source: str,
                        decision_created_at: float) -> None:
        """Commit this card's staged advisory opener only after a real Like is confirmed.

        A generated suggestion is a draft.  The Worker calls this after its decision boundary,
        so Pass, Stop, resync, and every failed/pre-tap action leave no profile-attributable
        opener record.  Billing telemetry is handled independently by OpenerService.
        """
        with self._lock:
            pick = self._pick
        commit = getattr(self._worker.opener_service, "commit_advisory_opener", None)
        if pick is not None and callable(commit):
            if _accepts_keywords(commit, "profile_id", "decision", "decision_source",
                                 "decision_created_at"):
                commit(pick, profile_id=profile_id, decision="like",
                       decision_source=decision_source, decision_created_at=decision_created_at)
            else:
                # Older advisory-service seams still receive the same confirmed-like boundary;
                # they simply do not persist the optional action-lineage columns.
                commit(pick)

    # --- internals ------------------------------------------------------
    def _applicable(self) -> bool:
        """Whether opener suggestions are a THING for this run at all -- as opposed to one that
        could not be produced for this card. Both of these are per-RUN and neither is news."""
        worker = self._worker
        return bool(getattr(worker.driver, "accepts_opener", False)
                    and worker.opener_service is not None
                    and not getattr(worker.opener_service, "disabled", True))

    def _blocker(self) -> str:
        """Why no suggestion can be made for THIS card, or "" when one can."""
        worker = self._worker
        unavailable = getattr(self._profile, "items_unavailable", "")
        if unavailable:
            return _operator_items_unavailable_warning(unavailable)
        if not getattr(self._profile, "items", ()):
            return ("this capture produced no numbered items, so there is nothing for the model "
                    "to choose from")
        # The canary property is the whole point of this workflow, so the check that protects it
        # is structural rather than a comment: a suggestion whose item nobody can verify against
        # the sheet is a suggestion doc 5.9 forbids showing, and refusing to GENERATE it is
        # cheaper and clearer than generating one and warning about it every time.
        if not callable(getattr(worker.driver, "observe_item_mismatch", None)):
            return ("this driver cannot check which item you opened against the item a "
                    "suggestion names, and doc 5.9 forbids offering text that cannot be checked")
        # Some drivers require measured, per-device evidence before their identity and sheet
        # checks can license targeted text.  Ask before the provider call: checking only after
        # the human opens a sheet would let unlicensed opener text appear on the hub meanwhile.
        readiness = getattr(worker.driver, "targeted_suggestion_blocker", None)
        if callable(readiness):
            try:
                reason = readiness()
            except Exception as exc:  # noqa: BLE001 — an optional safety gate fails closed
                return ("this driver's targeted-suggestion readiness check failed "
                        f"({type(exc).__name__}: {exc}), so no opener text is offered")
            if reason:
                return str(reason)
        return ""

    def _generate(self) -> None:
        """The provider call. Runs on its own thread; publishes whatever it ends up with."""
        worker = self._worker
        pick = None
        failure = None

        def should_stop() -> bool:
            return self._cancel_event.is_set() or worker.stop_event.is_set()

        # Do not even join OpenerService's shared queue if the card left before this daemon got
        # CPU time.  The service receives the same predicate below for the remaining race where
        # cancellation happens while it is waiting on its own budget lock or inside a cooperative
        # provider cascade.
        if should_stop():
            return
        try:
            pick = worker.opener_service.maybe_opener(
                worker.run_id, worker.app, self._profile,
                items=ItemRequest.from_profile(self._profile),
                should_stop=should_stop,
                advisory=True)
        except Exception as exc:  # noqa: BLE001 — a suggestion must never break a labelling run
            failure = f"the opener call failed ({type(exc).__name__}: {exc})"
        if pick is not None and pick.index == ITEM_INDEX_ABSENT:
            # An opener with no item number is exactly as unusable here as in AUTO, and for the
            # same reason: there is nothing to tell the human to like and nothing to check their
            # tap against. AUTO stops the run over it; observe declines to show it.
            failure = ("the model wrote an opener but named no item for it, so there is nothing "
                       "to tell you to like")
            pick = None
        index_space = getattr(pick, "index_space", None) if pick is not None else None
        if pick is not None and index_space != INDEX_SPACE_MODEL_ITEMS:
            # Observe's crop request has exactly one valid numbering: its own 1-based
            # model-item list.  `observe_item_mismatch` receives that same number and compares
            # the opened sheet with one of those crops, so accepting a legacy capture-order
            # value here would make the hub label/check one item while the text was generated
            # for another.  AUTO can still route a legacy response through its explicit
            # capture-order branch; Observe has no equivalent safe interpretation and must
            # withhold the text.
            failure = ("the model returned an opener in the unsupported index space "
                       f"{index_space!r}; Observe can only verify numbered model items, "
                       "so no opener is offered")
            pick = None
        with self._lock:
            # `cancel()` takes this same lock before the next capture is allowed to invalidate
            # the driver's per-profile crops.  Run the pure crop check under it, therefore: a
            # late model answer can never classify the NEXT person's item after this card has
            # already left the screen.  It is safe to hold because the documented capability
            # cannot capture, navigate, log, or otherwise touch the transport.
            if self._cancelled:
                return
            if pick is not None:
                mismatch = _item_type_preflight_mismatch(worker.driver, pick)
                if mismatch:
                    # Doc 5.8 is deliberately earlier than observe's sheet check: the model
                    # chose a numbered crop whose own coarse type contradicts its description,
                    # so this is not advice the human should receive.  Observe warns and
                    # withholds text; AUTO makes the corresponding ItemTargetingError/stop.
                    failure = ("the model's chosen item failed the pre-flight type check, so no "
                               f"opener is offered: {mismatch}")
                    pick = None
            self._pending = False
            self._pick = pick
            if pick is None:
                self._warning = failure or (
                    getattr(worker.opener_service, "last_skip_reason", None)
                    or getattr(worker.opener_service, "exhausted_reason", None)
                    or "no opener was produced for this card")
        # Announcing under `_publish`'s lock closes the cancellation race: after a cancelled
        # generation's status update has been suppressed, it must not leave a stale "ready" line
        # in the console/bug report.  It also prevents announcing a pick whose already-open sheet
        # made `_display` replace the text with a mismatch warning.
        self._publish(announce_pick=pick)

    def _display(self) -> dict:
        """The opener fields the hub should show right now, from both halves of the state.

        Called with the lock held. Returns a FULL set of opener fields every time (see
        status.cleared_opener_fields) so a transition can never leave one of them describing a
        previous state -- the class of bug an adversarial review already found once when the set
        was three fields long.
        """
        fields = cleared_opener_fields()
        fields["opener_pending"] = self._pending
        pick = self._pick
        if pick is None:
            fields["opener_warning"] = self._warning
            return fields
        fields["opener_item"] = pick.index
        fields["opener_media_ordinal"] = _display_media_ordinal(self._worker.driver, pick)
        fields["opener_item_description"] = pick.item_description or None
        if self._sheet is not None:
            mismatch = self._mismatch(pick)
            if mismatch:
                # DOC 5.9's WHOLE POINT: the opener is REPLACED by the warning, not annotated
                # with it. Leaving the text up beside a caveat is how a wrong-item opener gets
                # typed anyway, and the owner's rule is that observe honours auto's
                # never-attach-a-comment-to-the-wrong-item rule by having nothing to copy.
                fields["opener_warning"] = mismatch
                return fields
        fields["opener_suggestion"] = pick.text
        fields["opener_referenced"] = pick.referenced or None
        return fields

    def _mismatch(self, pick) -> str:
        """"" when the sheet the human opened is confirmed to be showing `pick.index`.

        The check itself lives on the driver (`observe_item_mismatch`) because that is where the
        crops, the index and this profile's identity fingerprint live. It is the same pair of
        comparisons, in the same order, that `driver.like()` makes on the auto path; only the
        consequence differs. An inconclusive layout/identity read remains advisory: it may be a
        keyboard reflow rather than a different item, so it preserves the already-generated text
        and is retried whenever the observer provides a fresher open-sheet frame.
        """
        if not self._sheet:
            current = ObserveItemCheck(
                OBSERVE_ITEM_INCONCLUSIVE,
                "the app's comment sheet is open but this driver could not hand over a "
                "picture of it, so there is no way to tell which item you opened")
        else:
            try:
                typed_check = getattr(self._worker.driver, "observe_item_check", None)
                if callable(typed_check):
                    current = typed_check(self._sheet, pick.index)
                    if not isinstance(current, ObserveItemCheck):
                        raise TypeError("observe_item_check did not return ObserveItemCheck")
                else:
                    # Compatibility for existing third-party drivers/test doubles. Their old
                    # string API cannot distinguish "wrong" from "could not look", so a reason
                    # remains a fail-closed mismatch. Production Hinge exposes the typed method.
                    reason = self._worker.driver.observe_item_mismatch(self._sheet, pick.index)
                    current = ObserveItemCheck(
                        OBSERVE_ITEM_MISMATCH if reason else OBSERVE_ITEM_MATCH,
                        reason or "")
            except Exception as exc:  # noqa: BLE001 — a display check never breaks the run
                current = ObserveItemCheck(
                    OBSERVE_ITEM_INCONCLUSIVE,
                    f"the item you opened could not be checked against this suggestion "
                    f"({type(exc).__name__}: {exc})")

        previous = (self._sheet_item_check[1]
                    if self._sheet_item_check is not None
                    and self._sheet_item_check[0] == pick.index else None)
        if previous is not None and previous.state == OBSERVE_ITEM_MISMATCH:
            effective = previous              # an affirmative wrong-item result stays refused
        elif current.state == OBSERVE_ITEM_INCONCLUSIVE and previous is not None:
            effective = previous              # typing/layout noise cannot revoke real evidence
        else:
            effective = current
            self._sheet_item_check = (pick.index, current)

        if effective.state == OBSERVE_ITEM_MATCH:
            if not self._release_post_tap_recorded:
                record = getattr(self._worker.driver, "observe_release_fact", None)
                if callable(record):
                    try:
                        record("post_tap_item_verified")
                    except Exception:  # noqa: BLE001 -- debug evidence never changes observe safety
                        pass
                self._release_post_tap_recorded = True
            return ""
        return effective.reason if effective.state == OBSERVE_ITEM_MISMATCH else ""

    def _publish(self, *, announce_pick=None) -> None:
        with self._lock:
            if self._cancelled:
                return
            fields = self._display()
            state = "waiting_for_send" if self._sheet is not None else "waiting"
            # set_app, not Worker._stat: the driver's status overlay must not be repainted from
            # this object, which is reachable from the suggestion thread while the worker thread
            # is inside wait_for_decision talking to the same device. On Hinge render_status is a
            # no-op anyway (its status lives on the hub), so nothing is lost; what is bought is
            # that no transport is touched from two threads.
            self._worker._publish_status(state=state, **fields)
            # This contains only the item number and is lock-safe like status publication.
            # It lets the reviewed-action mailbox bind an approval to the one suggestion that
            # is actually visible, without exposing a second route to the driver.
            # ``_ObserveSuggestion`` is also a deliberately small unit-test seam.  The
            # reviewed-action bridge is a Worker capability, not a requirement of every
            # object that can exercise suggestion display.  Real Workers always expose this
            # method and still publish every transition to the bridge; minimal synthetic
            # workers may omit it without turning a harmless no-suggestion warning into an
            # exception.
            publish_bridge_suggestion = getattr(self._worker, "_observe_action_suggestion", None)
            if callable(publish_bridge_suggestion):
                publish_bridge_suggestion(
                    getattr(self._worker, "_active_observe_action_profile_token", None),
                    self._pick if fields["opener_suggestion"] else None)
            if (fields["opener_suggestion"] and self._sheet is None
                    and not self._release_pre_tap_recorded):
                record = getattr(self._worker.driver, "observe_release_fact", None)
                recorded = False
                if callable(record):
                    try:
                        record("hub_pre_tap_published")
                        recorded = True
                    except Exception:  # noqa: BLE001 -- debug evidence never changes observe safety
                        pass
                if recorded:
                    self._worker._observe_action_pre_tap_published(
                        getattr(self._worker, "_active_observe_action_profile_token", None))
                self._release_pre_tap_recorded = True
            # EVERY warning reaches the console too, and this is the only place that can do it:
            # a mismatch is not decided until both halves of the state are in, and which thread
            # completes the pair is a race. Printed once per DISTINCT warning, because a dismissed
            # sheet reopened on the same wrong item would otherwise repeat itself. The hub is the
            # surface doc 5.9 names; stdout is what a bug report keeps, and "silent is the one
            # thing it must not be" is a claim about both.
            warning = fields["opener_warning"]
            if warning and warning != self._announced:
                self._announced = warning
                print(f"   💬 no suggestion to type — {warning}")
            # The text deliberately does not go to the console.  Announce only the instruction,
            # and only while the same lock proves the suggestion is still current and visible.
            # If the sheet check replaced it with a warning, there is intentionally nothing to
            # announce as ready.
            if announce_pick is not None and fields["opener_suggestion"] == announce_pick.text:
                media_ordinal = fields["opener_media_ordinal"]
                if media_ordinal is not None:
                    target = (f"media item {media_ordinal} ({announce_pick.item_description})"
                              if announce_pick.item_description else f"media item {media_ordinal}")
                else:
                    target = (f"the photo described as {announce_pick.item_description}"
                              if announce_pick.item_description else "the described photo")
                print(f"   💬 optional suggestion ready: if you choose to like, use {target}; "
                      f"the text to type is on the hub")


class Worker(threading.Thread):
    def __init__(self, app, driver: DatingAppDriver, decider: Decider, opener_service,
                 store, run_id, pacing, stop_event: threading.Event, mode: str = "observe",
                 retrain_every: int = 1, limiter=None, max_restarts: int = 5, status=None,
                 observe_source: str = "manual", observe_action_bridge=None):
        super().__init__(name=f"worker-{app}", daemon=True)
        self.app = app
        self.driver = driver
        self.decider = decider
        self.opener_service = opener_service
        self.store = store
        self.run_id = run_id
        self.pacing = pacing
        self.stop_event = stop_event
        self.mode = mode
        self.retrain_every = max(1, int(retrain_every))
        self.limiter = limiter
        self.max_restarts = max_restarts
        self.status = status              # optional RunStatus (live overlay/hub); None in tests
        # Observe normally records manual choices.  The explicit non-manual values are for a
        # separately reviewed external controller; their release verifier rejects manual rows,
        # so this is provenance rather than a cosmetic label.
        if observe_source not in {"manual", "external_ai_review", "automation"}:
            raise ValueError("observe_source must be manual, external_ai_review, or automation")
        self.observe_source = observe_source
        # Hub requests are a mailbox, never a second device reader.  It is optional so the
        # CLI/manual Observe path retains its existing behaviour byte-for-byte when no hub owns
        # this worker.
        self.observe_action_bridge = observe_action_bridge
        self.observe_action_supported = bool(
            mode == "observe" and app == "hinge"
            and observe_source in {"external_ai_review", "automation"}
            and callable(getattr(driver, "observe_pass", None))
            and callable(getattr(driver, "observe_open_targeted_like", None))
            and callable(getattr(driver, "observe_send_targeted_like", None)))

    # --- live status (overlay + hub); no-ops when status is unset ---------
    def _stat(self, **fields) -> None:
        if self.status:
            self.status.set_app(self.app, **fields)
        self._render()

    def _publish_status(self, **fields) -> None:
        """Update the shared status WITHOUT repainting the driver's in-app overlay.

        The overlay repaint (`_render`) reaches into the driver, and doc 5.9's observe suggestion
        publishes from its own thread while this worker's thread is inside `wait_for_decision`
        driving the same device. `RunStatus.set_app` is lock-guarded and touches nothing else, so
        this half is safe from anywhere; `render_status` is not, and is a no-op on the one driver
        that can produce a suggestion at all. See _ObserveSuggestion's docstring."""
        if self.status:
            self.status.set_app(self.app, **fields)

    def _render(self) -> None:
        if self.status:
            self.driver.render_status(self.status.app_view(self.app))

    def _install_opener_flag(self) -> None:
        """Tell the driver whether an opener will actually be requested this session.

        Called once per session by BOTH loops, before open_session(). An optional hook (absent on
        any fake or third-party driver that has no enumeration subsystem to gate), exactly like
        set_auto_session_policy. `opener.enabled: false` constructs a live-but-DISABLED
        OpenerService (client=None, not None, `disabled` from construction), which is why the
        condition is `not disabled` rather than `is not None` -- the same condition every other
        opener guard in this file uses.

        Observe needs this as much as auto does, and as of doc 5.9's inversion needs it MORE: it
        is now the only session-level gate on enumeration, and an observe run with openers off
        would otherwise pay a ~40-frame read per card for a numbered list nobody would ever
        look at."""
        install = getattr(self.driver, "set_opener_enabled", None)
        if callable(install):
            install(bool(self.opener_service is not None
                         and not getattr(self.opener_service, "disabled", True)))

    def _bind_debug_run(self) -> None:
        """Give an opt-in driver the immutable Worker id before it opens a debug session.

        Hinge writes a transport-free first debug record from this value. Other drivers remain
        unaffected: this optional metadata hook is not a device capability requirement.
        """
        bind = getattr(self.driver, "bind_debug_run", None)
        if callable(bind):
            bind(self.run_id)

    def _announce_observe_targeting_setup(self) -> None:
        """Explain a run-level targeted-opener prerequisite once, without blocking labels.

        Hinge's model-item suggestions are deliberately withheld until its per-device targeting
        calibration can prove that the item named before a manual tap is the item shown in the
        later comment sheet.  The per-card warning remains useful evidence, but it is a poor
        first explanation: an operator otherwise learns about this prerequisite only after a
        full capture and may mistake it for a profile-specific failure.  This startup notice is
        informational only -- observe remains a fully usable manual labelling session.

        Do not call an optional driver's readiness hook when no live opener service could
        consume it.  ``opener.enabled: false`` is an intentional run-wide choice, not a request
        to configure targeting, and drivers without targeted suggestions must retain their
        existing silent observe behaviour.
        """
        if not (getattr(self.driver, "accepts_opener", False)
                and self.opener_service is not None
                and not getattr(self.opener_service, "disabled", True)):
            return
        blocker = getattr(self.driver, "targeted_suggestion_blocker", None)
        if not callable(blocker):
            return
        try:
            reason = blocker()
        except Exception:  # noqa: BLE001 -- an optional startup notice must not break Observe
            return
        if reason:
            print(f"{self.app.title()} observe: targeted opener suggestions need setup "
                  f"({reason}). Manual pass/like labels still work. Complete the Hinge "
                  "targeted-opener calibration in ops/RUNBOOK.md to enable suggestions.")

    def _capture_failure(self, exc: BaseException) -> None:
        """Let the driver snapshot the on-screen failure state into its debug log. Called from
        inside the loop's except — WHILE the transport is still live — because the loop's finally
        closes the driver before run()'s handler sees the exception."""
        snap = getattr(self.driver, "snapshot_failure", None)
        if callable(snap):
            try:
                snap(exc)
            except Exception:  # noqa: BLE001 — debug capture must never mask the real error
                pass

    def _capture_failure_if_unexpected(self, exc: BaseException) -> None:
        """Shared except-clause body for _observe_loop/_auto_loop: snapshot the on-screen
        state for any failure that isn't a clean DriverClosed stop."""
        if not isinstance(exc, DriverClosed):
            self._capture_failure(exc)

    def _finish_session(self, state: str = "stopped", *, stop_reason: str | None = None,
                        stop_kind: str | None = None) -> None:
        """Shared unconditional cleanup for _observe_loop/_auto_loop's finally: publish the
        loop's TERMINAL state (rate_limited / out_of_profiles / blocked / stopped — not a
        blanket "stopped", which would hide why the run ended) and release the driver.

        stop_reason carries a human-readable explanation for a "stopped"/"blocked" outcome
        that has no exception behind it -- state="error" already has AppStatus.error for
        that. Optional: every other terminal path (a manual Stop click, out_of_profiles,
        rate_limited) passes nothing, so the hub still shows a bare "stopped" for those
        exactly as before.

        stop_kind disambiguates stop_reason's SOURCE now that several different code paths
        populate it (see AppStatus.stop_kind): "opener" for an opener-side stop (see
        _opener_stop_reason and _auto_loop's other opener guards), "deck_blocked" for the
        blocked-deck check added 2026-08-11 (see _observe_loop/_auto_loop's blocked_reason()
        check), and "targeting" for doc 5.6's hard stop on a like that could not be put on the
        item its opener was written about (see _targeting_stop_reason). Threaded through
        the same way as stop_reason -- only included when truthy, so a terminal path that
        never set one leaves the field at its AppStatus default (None).
        """
        fields = {"state": state}
        if stop_reason:
            fields["stop_reason"] = stop_reason
        if stop_kind:
            fields["stop_kind"] = stop_kind
        self._stat(**fields)
        try:
            self.driver.close()
        finally:
            if self.observe_action_bridge:
                self.observe_action_bridge.unregister(self)

    def _opener_stop_reason(self) -> str | None:
        """Human-readable cause of an OpenerService-triggered stop, or None if the service
        hasn't recorded one. OpenerService.exhausted_reason is first-writer-wins and shared
        by every worker, so whichever worker reads it here always sees the SAME original
        cause -- not whichever symptom that particular worker's own opener call hit."""
        return getattr(self.opener_service, "exhausted_reason", None)

    @staticmethod
    def _targeting_stop_reason(exc: ItemTargetingError) -> str:
        """Operator-facing sentence for doc 5.6's hard stop: we could not put the like on the item
        the opener was written about, so we put it nowhere.

        INTENDED AND ACTUAL LEAD, because those are the two things doc 5.6 asks a stop record to
        hold and they are what the operator has to reconstruct otherwise. They are read off the
        exception's fields rather than parsed back out of its message: the driver already knows
        both numbers and which numbering they are in, and re-deriving them from prose is how a
        stop record drifts out of agreement with the stop.

        `actual` is deliberately rendered as "never got that far" rather than as a number when the
        driver never reached anything -- "we could not reach item 4" and "we reached item 6 while
        aiming at item 4" are different diagnoses and the second is a much stronger signal that
        the item list itself is wrong.

        The driver's own message is appended whole. It names the app, the stage, and what the
        screen was left showing, and it is written to be read by the person walking over to the
        phone."""
        stage = {"preflight": "checking", "navigate": "reaching", "verify": "confirming"}.get(
            exc.stage, "targeting")
        space = f" ({exc.index_space})" if exc.index_space else ""
        intended = "none was named" if exc.intended is None else f"item {exc.intended}{space}"
        actual = ("the run never got far enough to see what it would have hit"
                  if exc.actual is None else f"item {exc.actual}{space}")
        return (f"the like was NOT sent: {stage} the item the opener was written about failed. "
                f"Intended: {intended}. Actual: {actual}. Liking a different item instead is "
                f"never an option (ops/OPENER-REDESIGN.md 5.6), so the run stops with the screen "
                f"left exactly as it is for debugging. {exc}")

    def _profile_separator(self) -> None:
        print("-" * _PROFILE_LOG_WIDTH)

    def _capture_profile(self, method: str):
        """Call the driver's capture (`current_profile` for observe, `next_profile` for auto),
        handing it this run's stop signal when — and only when — the driver says it honours one.

        Capture is the longest stop-deaf stretch of a run: on Hinge it is 12 screencaps plus 11
        humanized read-scrolls, and observe mode then scrolls the card back to the top, measured
        at ~85s in which this worker's own stop checks (the loop condition, and the one on the
        very next line of both loops) simply do not run. That is the reported bug — Stop pressed
        mid-read visibly kept scrolling to the end of the profile before anything stopped.

        Gated on the capability flag rather than passed unconditionally, following
        `on_like_intent`'s precedent a few methods down (and unlike `should_stop` on
        wait_for_decision, which IS unconditional and is exactly why every observe test double
        in the suite had to grow that keyword). A driver or fake that never declares
        `supports_interruptible_capture` keeps its zero-argument capture and its existing
        between-profiles Stop behaviour, so this cannot break third-party drivers or the
        hand-driven calibration tools that call current_profile() with no arguments at all.

        Nothing here needs to distinguish "returned None because Stop fired" from any other
        None: both loops re-check stop_event on the line right after this call, and both already
        treat None as "recapture" — so an interrupted capture is silent, records nothing, and
        falls straight out of the loop, which is precisely the wanted behaviour for a read that
        was abandoned before any decision existed to save.
        """
        capture = getattr(self.driver, method)
        if getattr(self.driver, "supports_interruptible_capture", False):
            return capture(should_stop=self.stop_event.is_set)
        return capture()

    def _observe_capture_method(self) -> str:
        """Choose the capture contract for this Observe worker, never for a caller request.

        Only an explicit, non-manual bridge can retain Hinge at its indexed bottom anchor.  The
        dedicated driver method is capability-checked so an older driver/fake remains on the
        manual-compatible ``current_profile`` contract rather than failing at run time.
        """
        if (self.observe_action_bridge is not None
                and self.observe_action_supported
                and callable(getattr(self.driver, "current_profile_reviewed", None))):
            return "current_profile_reviewed"
        return "current_profile"

    def _block_observe_capture(self, **status_fields) -> None:
        self.driver.render_busy(_OBSERVE_CAPTURE_BUSY)
        self._stat(state="capturing", **status_fields)

    def _block_observe_processing(self) -> None:
        # Same "don't swipe yet" signal as capture, but for the embed/store window after a
        # manual decision. Drives BOTH channels identically for every app: the in-page busy
        # modal (Bumble, when enabled) AND the shared status state the hub banner reads —
        # so apps with no on-screen overlay (Hinge) still show WAIT during the slow embed
        # instead of a stale decision prompt that would mis-attribute the next action.
        self.driver.render_busy(_OBSERVE_PROCESSING_BUSY)
        self._stat(state="acting")

    def run(self) -> None:
        def leave() -> None:
            if self.observe_action_bridge:
                self.observe_action_bridge.unregister(self)

        backoff, restarts = 2.0, 0
        # Publish the configured mode without touching the driver. In particular this must
        # happen before open_session(): if opening an auto session fails, the error banner's
        # per-app selector must not mistake the still-default AppStatus mode for "observe".
        if self.status:
            self.status.set_app(self.app, mode=self.mode)
        self._bind_debug_run()
        if self.observe_action_bridge:
            self.observe_action_bridge.register(self)
        while not self.stop_event.is_set():
            try:
                self._observe_loop() if self.mode == "observe" else self._auto_loop()
                leave()
                return
            except ActionCancelled:
                # Stop is neither a failed target nor a driver fault.  The action boundary that
                # raised this already guaranteed no further input, so end normally without a
                # failure snapshot, invented stop reason, or decision/counter record.
                self.stop_event.set()
                leave()
                return
            except DriverClosed as exc:
                print(f"{exc}; Stopping run so buffered data can be saved.")
                self.stop_event.set()
                leave()
                return
            except Exception:  # noqa: BLE001
                # NOTE: the on-screen failure screenshot is captured INSIDE the loop's except
                # (_capture_failure), BEFORE the loop's finally closes the driver — by here the
                # transport is already closed, so a snapshot would be blank.
                # HALT-on-unexpected: any AUTO run that hits an unexpected error must STOP, for
                # EVERY app — an autonomous loop that restart-and-continues would keep swiping
                # blindly (risky) and rotate the crucial failure logs away. The HALT line + the
                # traceback go to stdout/stderr, which the hub tees into the live-log panel, so
                # the stop is visible on the GUI terminal view (and lands in the bug report).
                # Observe mode keeps per-driver behavior via halt_on_error — which now
                # DEFAULTS TO TRUE on the driver ABC (see DatingAppDriver.halt_on_error).
                # The default used to be False here, so a driver that never declared the
                # attribute (PlaywrightDriver did not) got restart-with-backoff by omission
                # rather than by decision: the riskier path was what you got by forgetting.
                # Restarting is not free even in observe mode, where the bot only reads — the
                # worker re-attaches to whatever is on screen, and a driver that was confused
                # about which card it was on then mis-attributes the manual swipes it records,
                # corrupting the taste model permanently. Restart is now reachable only by
                # explicitly setting apps.<app>.halt_on_error: false, which config validation
                # allows in observe mode only.
                if self.mode == "auto" or getattr(self.driver, "halt_on_error", True):
                    print(f"{self.app.title()} unexpected error in {self.mode} mode; HALTING "
                          f"(no restart) so nothing swipes blindly and the debug logs survive:")
                    traceback.print_exc()
                    # Publish error to hub so the banner shows the failure (not a silent stop).
                    self._stat(state="error", error=traceback.format_exc().strip().splitlines()[-1])
                    self.stop_event.set()
                    leave()
                    return
                restarts += 1
                print(f"{self.app.title()} worker error (restart {restarts}/{self.max_restarts}):")
                traceback.print_exc()
                if restarts > self.max_restarts or self.stop_event.is_set():
                    print(f"{self.app.title()} worker giving up.")
                    leave()
                    return
                self.stop_event.wait(human_cooldown(min(backoff, 60)))
                backoff *= 2
        leave()

    # --- shadow learning: you decide, the bot learns --------------------
    def _observe_loop(self) -> None:
        print(f"{self.app.title()} observe mode — use the app's pass/like controls; "
              "I'll learn from each decision.")
        # Same hook, same place in the sequence, as _auto_loop's -- see _install_opener_flag.
        # Doc 5.9's inversion made observe an enumerating mode, so this is now what stands
        # between a no-opener observe run and a ~40-frame read per card it would never use.
        self._install_opener_flag()
        self.driver.open_session()
        self._stat(mode="observe")             # set mode for the hub; the loop owns per-card WAIT/SWIPE
        self._announce_observe_targeting_setup()
        added = 0
        last_retrained = 0
        pending_error = False
        terminal_state = "stopped"            # overwritten below when the loop ends for a known reason
        stop_reason = None                    # set for an OpenerService- or deck-blocked stop; see _finish_session
        stop_kind = None                      # disambiguates stop_reason's source; see _finish_session/status.py
        try:
            while not self.stop_event.is_set():
                # Ask the driver whether something is STANDING BETWEEN us and the deck --
                # deliberately checked BEFORE out_of_profiles() below, because it is the more
                # specific answer (out_of_profiles just means the deck ran dry; this means
                # there is a screen up that isn't the deck and isn't empty either). Default
                # implementation (DatingAppDriver.blocked_reason) returns None for every
                # driver but Hinge, so this is a no-op for Bumble/web.
                #
                # THIS IS A GRACEFUL STOP, NOT AN ERROR: Hinge's own "you're out of free likes
                # for today" Hinge+ upgrade screen can refuse a like. The phone is in a
                # perfectly normal state -- nothing is broken and
                # nothing here should be retried -- so this must NOT go through the
                # HALT-on-unexpected exception path (that path is for something actually
                # wrong). Before this check existed, worker.py called
                # wait_for_decision(timeout=None) with no bail-out at all, so this exact
                # screen hung observe mode for 2.5 minutes polling for a decision that could
                # never come, until the owner pressed Stop by hand. And above all: the
                # paywall is a PURCHASE screen, observe mode is strictly passive, and it must
                # never be tapped/dismissed automatically -- so this branch only reads the
                # screen and stops; it does not touch it.
                blocked = self.driver.blocked_reason()
                if blocked is not None:
                    record = getattr(self.driver, "observe_release_fact", None)
                    if callable(record):
                        try:
                            record("refusal_or_paywall_logged")
                        except Exception:  # noqa: BLE001 -- debug proof must not affect passive stop
                            pass
                    terminal_state = "blocked"
                    stop_reason = blocked
                    stop_kind = "deck_blocked"
                    self._stat(state=terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                    break
                if self.driver.out_of_profiles():
                    terminal_state = "out_of_profiles"
                    self._stat(state=terminal_state)
                    break
                self._profile_separator()
                self._block_observe_capture()                # WAIT cue for every card (capturing state)
                # A reviewed bridge is the sole phone controller for this card.  It can retain
                # Hinge's completed capture at the exact scrolled entry anchor the item index
                # was measured on; manual Observe must keep the long-standing top-unwind
                # contract for a person inspecting the card.  Capability-check the dedicated
                # method so older/duck-typed drivers retain their normal capture path.
                profile = self._capture_profile(self._observe_capture_method())
                if self.stop_event.is_set():
                    break
                if profile is None:
                    continue
                if not profile.photos:
                    print("Captured 0 profile photos; waiting to recapture before learning.")
                    self._stat(last_decision="no_photos")     # stays WAIT (busy still up from this iteration)
                    self.stop_event.wait(_NO_PHOTO_RETRY_S)
                    continue
                self.driver.render_busy(None)                 # processing done -> OK to decide now
                # EVERY opener field is cleared together, always. RunStatus.set_app's own
                # auto-clear only fires when opener_suggestion is ABSENT from the update, so
                # naming it here (as this call has always done) opts this site out of that
                # safety net -- and clearing the text while leaving the rest behind would let the
                # PREVIOUS card's "about: her dog" caption, or its "like item 3" instruction, sit
                # against the next card. status.cleared_opener_fields() is the one place that
                # list lives, so a field added there cannot be forgotten here.
                self._stat(state="waiting", **cleared_opener_fields())
                # READY IS PUBLISHED BEFORE THE SUGGESTION IS ASKED FOR, deliberately (doc 5.9):
                # generation can take up to opener.request_timeout_s and the operator must not be
                # made to wait for it. More than a courtesy -- blocking here would mean a human
                # who acts during the wait is never observed at all, because wait_for_decision
                # would start against a card they had already left, losing that decision and
                # mis-attributing the next one. The suggestion fills in behind this line.
                instruction = getattr(
                    self.driver, "observe_decision_instruction",
                    "use the app's pass/like controls")
                print(f"✅ READY — {instruction} for this profile.")
                self._warn_if_capture_truncated(profile)
                action_card = (self.observe_action_bridge.begin_card(self)
                               if self.observe_action_bridge and self.observe_action_supported else None)
                self._active_observe_action_profile_token = (
                    action_card["profile_token"] if action_card else None)
                suggestion = _ObserveSuggestion(self, profile).start()
                try:
                    liked = self._wait_for_observed_decision(
                        suggestion, action_card["profile_token"] if action_card else None)
                finally:
                    # Unconditional, on every exit from the wait (decision, resync, stop, raise).
                    # After this returns, nothing can publish for this card and nothing can call
                    # into the driver from the suggestion thread -- which is the precondition for
                    # the next iteration capturing a new profile and invalidating the driver's
                    # item table underneath it. See _ObserveSuggestion.cancel.
                    suggestion.cancel()
                    if action_card:
                        self.observe_action_bridge.end_card(self, action_card["profile_token"])
                    self._active_observe_action_profile_token = None
                if liked is _OBSERVE_PRE_TAP_TARGETING_REFUSED:
                    # The reviewed bridge refused BEFORE a heart tap.  The driver may have
                    # scrolled while proving the advertised item unreachable, but it did not
                    # send a pass or a like; re-capture through the normal owner path rather
                    # than manufacturing either decision.  The ``finally`` above has already
                    # cancelled the suggestion and ended this card's capability tokens.
                    print("Reviewed target refused before a heart tap — recapturing, nothing recorded.")
                    self._stat(last_decision="targeting_refused")
                    continue
                if liked is None:                             # card changed / deck empty / stop -> recapture
                    # wait_for_decision's None return covers SEVERAL different situations, not
                    # just one (see base.py's docstring): stop requested, deck emptied, timeout,
                    # or a card that genuinely changed with nothing corroborating it as a human
                    # decision (a RESYNC -- a drag-only touch, a tap outside the tap-radius
                    # tolerance; see hinge.py's _observe_gesture_verdict and its observe_resync
                    # debug record). Exactly one of those is something this worker itself
                    # cheaply knows, rather than has to guess at: `should_stop` passed into
                    # wait_for_decision above is self.stop_event.is_set, so if self.stop_event
                    # is set here, THIS None is the operator's own Stop (or a supervisor
                    # shutdown) firing mid-wait -- not anything the driver observed on screen.
                    # That case used to be labelled "resync" along with everything else, and it
                    # actively misled a real investigation: a bug report's run ended on a plain
                    # manual Stop, and the last console line still read "Card changed without a
                    # corroborated decision (resync)" -- a developer reading that log reasonably
                    # went looking for a perception bug that had never happened at that moment.
                    # A Stop is the operator's own action, not evidence about the screen, so it
                    # is split out below and reported as nothing more than what it is.
                    #
                    # The REMAINING cases -- deck emptied, timeout, a genuine uncorroborated
                    # advance -- are still deliberately reported together as "resync", because
                    # telling THEM apart would need the driver to report WHY through a signature
                    # two other drivers and two tools also call with fewer arguments. In the
                    # deck-emptied/timeout cases this label is only momentarily stale anyway:
                    # the loop's own out_of_profiles handling overwrites `state` on the very
                    # next check, right above.
                    if self.stop_event.is_set():
                        continue                              # operator's own Stop -- not a resync; nothing to report
                    print("Card changed without a corroborated decision (resync) — "
                          "recapturing, nothing recorded.")
                    self._stat(last_decision="resync")
                    continue                                  # next iteration re-blocks + recaptures the card
                # A bool here is a confirmed preference action.  Do NOT drop it merely because
                # Stop arrived in the tiny interval after the driver/bridge returned that fact:
                # a Hinge Like may already be server-side, and its staged advisory opener must
                # cross the same decision boundary exactly once.  The ``liked is None`` path
                # above remains the Stop-before-decision path and discards the draft.
                #
                # This is deliberately different from the check immediately after capture:
                # there, no decision exists yet.  Here the card's action is already complete;
                # finish its durable accounting, then the while condition ends the run before
                # any new capture can begin.
                # block the next decision while this one embeds (avoids mis-attribution)
                decision = "LIKE" if liked else "PASS"
                decision_source = getattr(self, "_observe_action_decision_source", self.observe_source)
                self._observe_action_decision_source = self.observe_source
                print(f"Got {decision} — processing, don't decide again yet…")
                self._block_observe_processing()
                # Persist the completed physical action BEFORE any optional archive/embedding
                # work and before committing its staged opener.  A real Like/Pass remains a
                # decision even when there is no usable face vector or the photo archive is
                # unavailable.  Most importantly, this gives every committed opener a durable
                # Like decision to join to: cleanup/reporting must never have to infer an action
                # from a generated message alone.
                #
                # No broad try/except belongs around this call.  If durable decision storage is
                # down after a real swipe, continuing to archive/label or commit an opener would
                # manufacture a misleading partial history.  The normal worker failure path
                # leaves the device snapshot and the storage error visible instead.
                profile_id = uuid.uuid4().hex
                decision_created_at = time.time()
                _record_decision_with_lineage(
                    self.store, self.run_id, self.app, "like" if liked else "dislike",
                    1.0 if liked else 0.0, source=decision_source,
                    profile_id=profile_id, created_at=decision_created_at)
                if liked:
                    # The advisory opener was only a draft until this exact physical-action
                    # boundary.  It follows the decision row (rather than preceding it), so a
                    # store failure cannot leave an opener claiming a Like that has no matching
                    # durable decision.  Pass, Stop-before-decision, resync, and failed/pre-tap
                    # bridge actions never reach either call.
                    suggestion.commit_if_liked(profile_id=profile_id, decision_source=decision_source,
                                               decision_created_at=decision_created_at)
                if self.status:
                    self.status.record_swipe(self.app, "like" if liked else "pass")
                metadata = self._label_metadata(profile)
                archived = self.store.record_profile(self.run_id, self.app, profile_id, liked,
                                                     source=decision_source, photos=profile.photos, **metadata)
                vec = self.decider.embed(profile)
                if vec is None:                               # no face -> not a useful label
                    self._stat(last_decision="no_face")
                    continue
                if archived is False:                         # images couldn't be saved -> no label without them
                    self._stat(last_decision="archive_failed")
                    continue
                # NOTE: deliberately no stop_event check here. A manual swipe already
                # happened and its photos + embedding are already committed above — Stop
                # must end the loop AFTER this label is saved (the while-loop condition
                # below does that without starting a NEW capture), not discard work in hand.
                self.store.add_label(self.run_id, self.app, liked, vec, source=decision_source,
                                     profile_id=profile_id, **metadata)
                if self.status:
                    self.status.inc_labels(1)
                self._render()
                # A budget/credit stop can be discovered while generating the Hinge
                # suggestion. This human decision is already durably recorded above,
                # so honour the stop only now rather than interrupting the open sheet.
                if getattr(self.opener_service, "stop_requested", False):
                    stop_reason = self._opener_stop_reason()
                    stop_kind = "opener"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                added += 1
                if added % self.retrain_every == 0:
                    self._retrain_after_observe_labels(added)
                    last_retrained = added
        except BaseException as exc:
            pending_error = True
            if isinstance(exc, Exception):
                self._capture_failure_if_unexpected(exc)      # snapshot WHILE the transport is live
            raise
        finally:
            try:
                if added and added != last_retrained:
                    try:
                        self._retrain_after_observe_labels(added)
                    except Exception:  # noqa: BLE001
                        if not pending_error:
                            raise
                        print(f"{self.app.title()} final retrain skipped after shutdown error:")
                        traceback.print_exc()
            finally:
                # Same set-clearing rule as the per-card reset above (see status.py): naming
                # opener_suggestion opts this call out of set_app's auto-clear, so EVERY OTHER
                # field in the set must be named too, or a finished run leaves the hub showing
                # what the last suggestion was supposedly about -- or, since doc 5.9's
                # inversion, a "like item 3" instruction and its description -- with no
                # suggestion under any of it. status.cleared_opener_fields() is the single list
                # (the set was three fields when this comment first said "the other two"; it is
                # seven now, which is exactly why the list lives in one place and is splatted
                # rather than re-typed at each site).
                self._stat(**cleared_opener_fields())
                self.driver.render_busy(None)
                self._finish_session(terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)

    def _observe_action_suggestion(self, profile_token: str | None, pick) -> None:
        """Publish only the item binding; text stays inside the existing suggestion/status path."""
        if profile_token and self.observe_action_bridge:
            self.observe_action_bridge.update_suggestion(self, profile_token, pick)

    def _observe_action_pre_tap_published(self, profile_token: str | None) -> None:
        """Unlock a reviewed heart only after its hub-publication fact was written."""
        if profile_token and self.observe_action_bridge:
            self.observe_action_bridge.mark_pre_tap_published(self, profile_token)

    def _perform_observe_action(self, action: dict, suggestion: "_ObserveSuggestion"):
        """Run one approved action on this Worker thread at its waiting boundary."""
        bridge = self.observe_action_bridge
        try:
            if self.stop_event.is_set():
                bridge.complete(action, "aborted", "run is stopping")
                return None
            command = action["command"]
            if command == "pass":
                self.driver.observe_pass(should_stop=self.stop_event.is_set)
                bridge.complete(action, "completed", phase="terminal")
                self._observe_action_decision_source = self.observe_source
                return False
            pick = getattr(suggestion, "_pick", None)
            if (pick is None or not getattr(pick, "text", None)
                    or getattr(pick, "index", None) != action["item"]):
                bridge.complete(action, "rejected", "suggestion is no longer current")
                return "continue"
            if command == "targeted_like_open":
                # Hinge owns its current numbered-item index and entry anchor.  The bridge
                # supplies only the model item number it already advertised; no screen points.
                self.driver.observe_open_targeted_like(action["item"], should_stop=self.stop_event.is_set)
                bridge.complete(action, "completed", phase="sheet_open")
                return "continue"
            if command == "send_current_suggestion":
                # The driver re-verifies the current composer/item before typing or sending.
                self.driver.observe_send_targeted_like(pick.text, action["item"],
                                                       should_stop=self.stop_event.is_set)
                bridge.complete(action, "completed", phase="terminal")
                self._observe_action_decision_source = self.observe_source
                return True
            bridge.complete(action, "rejected", "unsupported command")
            return "continue"
        except ActionCancelled:
            bridge.complete(action, "aborted", "run stopped before device input")
            raise
        except ItemTargetingError as exc:
            # A targeting refusal at preflight/navigation is the one driver-guaranteed
            # no-tap outcome.  In AI-reviewed Observe it is recoverable: mark this exact
            # bridge action terminal, retire its one-card capability in the caller, and
            # recapture.  Never turn it into a synthetic Pass and never persist a label.
            #
            # ``verify`` is intentionally *not* included.  That happens after the heart can
            # have been tapped and leaves a sheet open; treating it as a benign retry could
            # conceal a partial irreversible action.  It retains the normal fail-closed worker
            # failure behaviour below.
            if (action.get("command") == "targeted_like_open"
                    and getattr(exc, "stage", "") in {"preflight", "navigate"}):
                bridge.complete(action, "failed", f"{type(exc).__name__}: {exc}")
                return _OBSERVE_PRE_TAP_TARGETING_REFUSED
            bridge.complete(action, "failed", f"{type(exc).__name__}: {exc}")
            raise
        except Exception as exc:  # driver preserves the screen for targeting failures
            bridge.complete(action, "failed", f"{type(exc).__name__}: {exc}")
            raise

    def _wait_for_observed_decision(self, suggestion: "_ObserveSuggestion", profile_token: str | None = None):
        """Wait for one human decision, checking their tap against the suggestion already shown.

        Hinge is the only current driver that exposes its intermediate comment sheet. This
        callback publishes shared status and runs one deterministic comparison on a frame the
        driver already holds; it never calls a tap, swipe, or text method, and it no longer calls
        the opener service at all. Other drivers and older fakes retain their existing
        ``wait_for_decision`` signature.

        THE CALLBACK'S JOB INVERTED WITH DOC 5.9, and this is the whole diff. It used to be
        "generate an opener now that we know which item the human chose", with the sheet frame
        handed to the model as an ANCHOR. The suggestion is now already on the hub before the
        human touches anything (see _ObserveSuggestion, started by the loop right after READY),
        so the frame is no longer an input to anything -- it is the EVIDENCE that the item the
        human opened is the item the suggestion was written for. On a mismatch the hub replaces
        the opener with a warning and offers no text to type: doc 5.9's rule, and the reason the
        inversion does not quietly reintroduce the out-of-place openers it was built to fix.

        The `profile` argument is gone: this method no longer builds a request, so it no longer
        needs one. `suggestion` already holds the profile it was built from, which is what keeps
        "the suggestion on the hub" and "the card in front of the human" the same card by
        construction rather than by two arguments agreeing.

        NOTHING IN HERE CAN END THE SESSION. The generation call that used to sit in this
        callback is on its own thread and is advisory (short bounded retries,
        `request_stop=False`), so an
        opener failure remains what it always was -- cosmetic. The observe loop's own
        `stop_requested` check stays, because a DIFFERENT worker (an auto-mode app sharing this
        OpenerService) can still legitimately ask everyone to stop.
        """
        bridge = self.observe_action_bridge if profile_token else None
        if bridge is None and not getattr(self.driver, "supports_observe_like_intent", False):
            return self.driver.wait_for_decision(timeout=None,
                                                 should_stop=self.stop_event.is_set)

        # `anchor: bytes | None = None` keeps this callback callable by any driver that has
        # not adopted the two-argument hook yet -- an old-style `on_like_intent(active)` call
        # (a single positional bool) still binds fine, rather than raising a TypeError that
        # the driver's own notifier would swallow (see _notify_observe_like_intent's
        # best-effort contract) and thereby kill suggestions silently, with nothing on the
        # hub to say why. An old-style caller then reaches sheet_opened(None), which is a
        # WARNING rather than a pass: "we could not look" must never render as "we looked".
        def on_like_intent(active: bool, anchor: bytes | None = None) -> None:
            if active:
                suggestion.sheet_opened(anchor)
            else:
                suggestion.sheet_closed()

        def interrupted() -> bool:
            return self.stop_event.is_set() or bool(bridge and bridge.has_pending(self, profile_token))

        while True:
            action = bridge.claim(self, profile_token) if bridge else None
            if action:
                outcome = self._perform_observe_action(action, suggestion)
                if outcome == "continue":
                    continue
                return outcome
            if not getattr(self.driver, "supports_observe_like_intent", False):
                result = self.driver.wait_for_decision(timeout=None, should_stop=interrupted)
            else:
                result = self.driver.wait_for_decision(timeout=None, should_stop=interrupted,
                                                       on_like_intent=on_like_intent)
            if self.stop_event.is_set():
                return None
            # A pending command makes the driver return None through should_stop; claim it
            # immediately instead of treating it as a manual resync/capture race.
            if bridge and bridge.has_pending(self, profile_token):
                continue
            return result

    @staticmethod
    def _warn_if_capture_truncated(profile) -> None:
        """Tell the operator when THIS card was only partially read, right at the moment they
        are about to decide on it.

        A truncated capture does not corrupt the label — every frame captured is genuinely this
        person. What it degrades is the driver's ability to tell a manual SCROLL apart from a
        PASS for the rest of this card's wait: the scroll matcher can only recognise territory
        the bot already captured, so if you scroll below where the read stopped, the identity
        band is the only signal left. Silence here is what made that invisible; a line at READY
        costs nothing and tells the operator the one thing they can act on — that scrolling far
        down THIS card is the case most likely to need a re-decision.
        """
        meta = getattr(profile, "meta", None) or {}
        if not meta.get("capture_truncated"):
            return
        # "screencaps", not "screens": each read-scroll advances a FRACTION of a screen height
        # (Hinge's read_scroll_frac is 0.55), so 12 screencaps is roughly 7 screen-heights of
        # profile, not 12. Naming the unit the config and the driver already use ("max
        # screencaps while reading one profile") keeps the operator from over-estimating how
        # much of the card was actually covered.
        frames = meta.get("capture_frames", len(profile.photos))
        print(f"   ⚠️  only the first {frames} screencaps of this profile were read (it is "
              f"longer than the configured ceiling), so scroll detection is weaker for this card.")

    def _retrain_after_observe_labels(self, added: int) -> None:
        ready = self.decider.retrain(self.store)
        if self.status:
            self.status.set_global(ranker_ready=ready)
        self._render()
        print(f"Learned {added} labels this run; ranker ready={ready}")

    # --- autonomous: the bot swipes ------------------------------------
    def _auto_loop(self) -> None:
        # Annotated `int` rather than left to inference on purpose. Bare `= 0` makes a type
        # checker infer the literal type Literal[0], and a `> 0` test does not widen a literal
        # back to int -- so the guarded `liked / acted` below still got reported as a
        # division by a literal zero even though the guard makes it unreachable. Declaring
        # the counters as plain ints states what they actually are (running tallies, not the
        # constant 0) and removes the false positive without weakening any runtime guard.
        acted: int = 0
        liked: int = 0
        # This state belongs to one auto session, not to the learned preference
        # model.  It may only make a marginal model like more conservative; it
        # never manufactures a like.  Fakes/third-party deciders need not expose
        # the underlying model, hence the guarded configured-threshold lookup.
        configured_threshold = getattr(getattr(self.decider, "model", None), "threshold", 0.5)
        try:
            configured_threshold = float(configured_threshold)
        except (TypeError, ValueError):
            configured_threshold = 0.5
        if not 0.0 < configured_threshold < 1.0:
            configured_threshold = 0.5
        self._auto_policy = AutoSessionPolicy(base_threshold=configured_threshold)
        # AndroidDriver exposes the policy through a deliberately optional
        # hook.  Other drivers can ignore it, while lightweight fakes and older
        # plugins still receive it through a harmless instance attribute.
        install_policy = getattr(self.driver, "set_auto_session_policy", None)
        if callable(install_policy):
            install_policy(self._auto_policy)
        else:
            try:
                setattr(self.driver, "_auto_policy", self._auto_policy)
            except (AttributeError, TypeError):
                pass
        # Tell the driver whether an opener will actually be requested this session (audit fix,
        # "BUG 2", 2026-08-12) -- mirrors install_policy immediately above: an optional hook,
        # called once before the loop starts rather than per profile (opener.enabled is a static
        # run-start config choice, not something that flips mid-run -- see
        # HingeDriver.set_opener_enabled's docstring for why a one-time check is correct even
        # though `disabled` can also become True later, via mid-run exhaustion). Absent on any
        # driver that has no enumeration subsystem to gate (every non-Hinge driver today), so
        # this is a no-op for them, exactly like install_policy above.
        self._install_opener_flag()
        # Session micro-break fatigue model: re-rolled each stretch so a long run's
        # break pattern isn't governed by one fixed hazard rate for its whole duration
        # (see _maybe_session_break).
        self._actions_since_break = 0
        self._break_hazard = random.uniform(0.04, 0.14)
        self._break_due_after = random.uniform(12, 35)
        # A no-op RateLimiter (all fields None, the shipped default) is still truthy.
        # Daily history is needed only for max_per_day; querying it for an uncapped run
        # adds startup latency and can turn an irrelevant store read failure into a halt.
        has_daily_limit = self.limiter is not None and self.limiter.max_per_day is not None
        today0 = self.store.count_today(self.app) if has_daily_limit else 0
        today_acted = 0            # actions this worker made since today0 was last measured
        today_date = date.today()  # LOCAL day — matches the store's count_today() day boundary
        self.driver.open_session()
        self._stat(mode="auto", state="scoring")
        terminal_state = "stopped"            # overwritten below when the loop ends for a known reason
        stop_reason = None                    # set for an OpenerService- or deck-blocked stop; see _finish_session
        stop_kind = None                      # disambiguates stop_reason's source; see _finish_session/status.py
        try:
            while not self.stop_event.is_set():
                if has_daily_limit:
                    today = date.today()
                    if today != today_date:
                        # Local midnight crossed mid-run — re-baseline against the new day
                        # instead of forever charging new-day actions against a day that's
                        # already over. Only queries the store on the rollover itself, not
                        # every profile.
                        today0 = self.store.count_today(self.app)
                        today_acted = 0
                        today_date = today
                # Ask the driver whether something is STANDING BETWEEN us and the deck --
                # deliberately checked BEFORE out_of_profiles() below, because it is the more
                # specific answer (out_of_profiles just means the deck ran dry; this means
                # there is a screen up that isn't the deck and isn't empty either). Default
                # implementation (DatingAppDriver.blocked_reason) returns None for every
                # driver but Hinge, so this is a no-op for Bumble/web.
                #
                # THIS IS A GRACEFUL STOP, NOT AN ERROR: Hinge's own "you're out of free likes
                # for today" Hinge+ upgrade screen. The phone is in a perfectly normal
                # state -- nothing is broken and nothing here should be retried -- so this
                # must NOT go through the HALT-on-unexpected exception path (that path is
                # for something actually wrong). See _observe_loop's matching check for the
                # full incident writeup; the same reasoning applies here even though AUTO
                # mode's own actions (not a human's) are what triggered Hinge's paywall this
                # time. blocked_reason() itself never taps/swipes/types (see its contract on
                # DatingAppDriver) so this check is safe even mid-auto-session.
                blocked = self.driver.blocked_reason()
                if blocked is not None:
                    terminal_state = "blocked"
                    stop_reason = blocked
                    stop_kind = "deck_blocked"
                    self._stat(state=terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                    break
                if self.driver.out_of_profiles():
                    terminal_state = "out_of_profiles"
                    self._stat(state=terminal_state)
                    break
                if self.limiter and not self.limiter.allow(acted, today0 + today_acted):
                    terminal_state = "rate_limited"
                    self._stat(state=terminal_state)
                    print(f"{self.app.title()} rate limit reached ({self.limiter.describe()}); "
                          f"stopping {self.app}.")
                    break
                profile = self._capture_profile("next_profile")
                if self.stop_event.is_set():
                    break
                if profile is None:
                    break

                self._stat(state="scoring")
                d = self.decider.decide(profile)
                if self.stop_event.is_set():
                    break
                if d.decision == "defer":
                    self._stat(last_decision="defer", state="stopped")
                    print(f"{self.app.title()} ranker not ready (cold-start) — run in observe "
                          f"mode and make decisions manually to seed it. Stopping {self.app}.")
                    break

                # A session can decline a marginal model like according to its
                # bounded fatigue/context state.  This happens before every
                # limit check so a policy-demoted like neither consumes a like
                # budget nor trips the running like-ratio guard.
                d = self._auto_policy.apply_decision(d, profile).decision

                # Per-run like budget: stop the run rather than mislabel a wanted
                # like as a pass (keeps the right-swipe ratio human; see limits.py).
                if d.decision == "like" and self.limiter and not self.limiter.allow_like(liked):
                    terminal_state = "rate_limited"
                    self._stat(state=terminal_state)
                    print(f"{self.app.title()} per-run like budget reached "
                          f"({self.limiter.describe()}); stopping {self.app}.")
                    break

                # Ratio shape: demote this like to a pass when the running like-rate
                # is at the ceiling (soft cap — does NOT halt the run) AND this like is
                # only marginally over the ranker's own threshold. A clearly strong
                # like (score well above configured_threshold) survives the ceiling
                # instead — see limits.py's module docstring and allow_like_ratio's
                # docstring for the worked numbers and the score-blind bug this
                # replaced (an earlier version of this shaper demoted the STRONGEST of
                # four real scores purely because of arrival order). configured_threshold
                # (the plain model threshold, not AutoSessionPolicy's per-card *lifted*
                # effective_threshold from apply_decision above) is deliberately what is
                # passed here — see _STRONG_LIKE_MARGIN's comment in limits.py for why
                # that avoids double-counting AutoSessionPolicy's own separate demotion.
                # `acted > 0` is duplicated deliberately: allow_like_ratio() already
                # returns True when acted==0 (so this branch can't actually run with
                # acted==0), but that safety lives inside a DIFFERENT function, which is
                # exactly why a static analyzer flags `liked / acted` below as a possible
                # division by zero, and an `assert` alone would vanish under `python -O`.
                # Repeating the guard here, in the condition itself, makes the division
                # provably safe at its own call site regardless of either of those.
                if d.decision == "like" and self.limiter and acted > 0 \
                        and not self.limiter.allow_like_ratio(
                            liked, acted, score=d.score, like_threshold=configured_threshold):
                    ratio_pct = f"{liked / acted:.0%}"
                    print(f"{self.app.title()} like-ratio ceiling "
                          f"({ratio_pct} ≥ {self.limiter.target_like_ratio:.0%}, score "
                          f"{d.score:.3f} marginal vs threshold {configured_threshold:.3f}) "
                          f"— demoting to pass")
                    # Tagged like interaction.py's own contextual demotion
                    # (f"{source}_contextual") rather than left as the raw ranker source,
                    # so a future consumer of Decision.source can tell a shaped decision
                    # apart from a raw model one. No functional effect today:
                    # Decision.source isn't read anywhere downstream and the stored row
                    # hardcodes source="auto" (see record_decision below) — this is
                    # forward-looking consistency, not a behavior change.
                    d = Decision("dislike", d.score, d.embedding, f"{d.source}_ratio_ceiling")

                if self.stop_event.is_set():
                    break

                if d.decision == "like":
                    # Only generate a provider-backed opener for apps that can actually send one
                    # at swipe time (Hinge). On Bumble we'd just discard it — wasted credits.
                    accepts_opener = getattr(self.driver, "accepts_opener", True)
                    # `self.opener_service is not None` guard: not reachable via supervisor.run()
                    # today (it always constructs a real OpenerService, even with openers
                    # disabled -- see OpenerService(client=None, ...) for that case), but this
                    # Worker is a public class any other caller can construct directly, and an
                    # audit proved the unguarded call raises a confusing
                    # `AttributeError: 'NoneType' object has no attribute 'maybe_opener'`
                    # instead of this codebase's usual clear, actionable message. The observe
                    # path already guards the equivalent call the same way (see
                    # _wait_for_observed_decision's `self.opener_service is not None` check).
                    #
                    # THE REQUEST SHAPE IS DECIDED HERE, and there are only two outcomes: the
                    # numbered item crops the driver enumerated, or a hard stop. Doc 5.2 sends
                    # crops precisely so that image k IS item k; falling back to the raw scroll
                    # frames when the crops are missing would hand the model a numbering nothing
                    # downstream can act on (one card appears in several frames, one frame can
                    # hold two cards), which is the ambiguity this whole redesign removes. So a
                    # driver that ENUMERATES and could not produce a payload stops the run with
                    # the reason it recorded, rather than quietly reverting to profile.photos.
                    #
                    # A driver that does not enumerate at all -- every non-Hinge driver today --
                    # leaves both fields empty and keeps the pre-existing frame shape, which is
                    # the honest reading of "this capture has no item space": there is nothing to
                    # refuse. `items_unavailable` is what distinguishes the two, which is why the
                    # driver must set exactly one of the pair (see perception.capture.Profile).
                    #
                    # `not disabled` (audit fix, "BUG 2", 2026-08-12): `opener.enabled: false`
                    # constructs a live-but-disabled OpenerService (client=None,
                    # self.opener_service is not None, disabled=True from construction -- see the
                    # comment two paragraphs up), which this condition used to treat the same as
                    # a live, enabled one. With openers off there is no consumer for a numbered
                    # item list at all, so `items_unavailable` -- set by the driver whenever
                    # enumeration was attempted and failed -- must not be read as a reason to
                    # stop a run that was never going to send an opener anyway. The driver-side
                    # fix (HingeDriver.set_opener_enabled) stops enumeration from even running in
                    # this case, but this guard is the one that actually prevents the hard stop:
                    # `set_opener_enabled` is best-effort (an optional hook, absent on any fake or
                    # future driver that does not define it), while `disabled` is this service's
                    # own authoritative state and costs nothing extra to check here, exactly like
                    # the identical `not disabled` guard a few lines below this block already uses
                    # for the "pick is None" case.
                    items = None
                    if accepts_opener and self.opener_service is not None \
                            and not getattr(self.opener_service, "disabled", True):
                        unavailable = getattr(profile, "items_unavailable", "")
                        if unavailable:
                            stop_reason = (
                                f"this profile could not be enumerated into numbered items, so "
                                f"there is no honest opener request to make and the like is not "
                                f"sent: {unavailable}. (ops/OPENER-REDESIGN.md 5.2 -- sending "
                                f"the raw scroll frames instead would give the model a numbering "
                                f"nothing can act on, so this stops rather than degrades.)"
                            )
                            # stop_kind="opener" rather than a new kind: the hub branches on it
                            # (assets/hub.html) and this IS the opener path refusing to send --
                            # the like is withheld because no opener request can be built, which
                            # is the same operator-facing situation as every other opener stop.
                            stop_kind = "opener"
                            self._stat(state="stopped", stop_reason=stop_reason,
                                       stop_kind=stop_kind)
                            self.stop_event.set()
                            break
                        if getattr(profile, "items", ()):
                            # Raises if the payload is somehow empty-but-not-refused; that is a
                            # driver contract violation, not a condition to handle here.
                            items = ItemRequest.from_profile(profile)
                    opener_kwargs = {"items": items, "should_stop": self.stop_event.is_set}
                    # Stage only when both halves of the modern protocol are present.  This
                    # preserves old third-party/test opener services whose maybe_opener method
                    # predates the keyword (rather than crashing a landed-action loop with an
                    # unexpected ``stage`` kwarg); OpenerService itself has both and is the only
                    # production implementation that persists opener rows.
                    if (callable(getattr(self.opener_service, "commit_opener", None))
                            and _accepts_keywords(self.opener_service.maybe_opener, "stage")):
                        opener_kwargs["stage"] = True
                    pick = (self.opener_service.maybe_opener(
                                self.run_id, self.app, profile, **opener_kwargs)
                            if accepts_opener and self.opener_service is not None else None)
                    # An opener call can discover that every configured provider/model is
                    # exhausted. In AUTO mode, honour its global stop BEFORE calling
                    # like(): a bare like is not an acceptable substitute for the opener
                    # the worker decided to send. OBSERVE deliberately handles this only
                    # after the human's already-completed action has been persisted.
                    if getattr(self.opener_service, "stop_requested", False):
                        stop_reason = self._opener_stop_reason()
                        stop_kind = "opener"
                        self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                        break
                    # COMPLETED RULE: on an opener-capable app, an AUTO like is normally sent
                    # WITH its opener or not sent at all. The one narrow exception is a Gemini
                    # safety-policy block: the ranker has already decided to like this profile,
                    # so the service may explicitly license that decision to proceed without a
                    # comment. This is not configurable and does not revive the removed generic
                    # budget.on_exhausted="swipe_without_opener" mode. The guard above catches
                    # GLOBAL exhaustion
                    # (stop_requested, which _exhaust() now sets unconditionally); maybe_opener()
                    # can also return None for reasons that are narrow to THIS profile and leave
                    # the service otherwise healthy -- a per-profile OpenerError, a single
                    # sub-latch HTTP 400, or a single sub-latch transient failure (an unparseable
                    # response, OpenerParseError, is no longer one of these: it is now retried by
                    # maybe_opener() itself, and either produces an opener or exhausts the
                    # service, so it never reaches here as a narrow, still-enabled failure -- see
                    # service.py's maybe_opener docstring). Falling straight through to a bare
                    # driver.like(None, ...) here would silently drop the opener the worker just
                    # decided this profile deserved -- and, on Hinge, silently drop the
                    # "commented like" behavior signal that opener is also there to produce.
                    # `disabled` is what tells this case apart from the ONE case it must NOT
                    # fire for: opener.enabled=false in config (self.opener_service.disabled is
                    # True from construction, client is None) -- there was never an opener to
                    # send here, so auto mode must run exactly as if openers didn't exist, not
                    # halt on the very first like. There, self.opener_service.disabled is True,
                    # so `not disabled` is False and this guard is a no-op. A live,
                    # still-enabled service that just failed on THIS call normally stops here;
                    # the structured safety-block permission is the sole exception. Apps
                    # that do not accept openers (Bumble) are untouched: accepts_opener is False
                    # for them, so pick is always None by construction and this condition never
                    # evaluates true.
                    allows_commentless_like = bool(getattr(
                        self.opener_service, "last_skip_allows_commentless_like", False))
                    if accepts_opener and pick is None and self.opener_service is not None \
                            and not getattr(self.opener_service, "disabled", True) \
                            and not allows_commentless_like:
                        stop_reason = getattr(self.opener_service, "last_skip_reason", None) or (
                            "opener service returned no opener for this profile and no "
                            "specific reason was recorded"
                        )
                        stop_kind = "opener"
                        self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                        break
                    # A pick CAN exist (an opener was generated for this profile) and still
                    # carry no honest way to know which item to tap: `pick.capture_order_index`
                    # is the ONE sanctioned crossing from the model's index space into the
                    # driver's (see its docstring), and it returns None whenever no sound
                    # conversion exists -- either the model gave no item number at all
                    # (ITEM_INDEX_ABSENT) or it answered in a space this build has no
                    # driver-owned translation table for yet (INDEX_SPACE_MODEL_ITEMS;
                    # ops/OPENER-REDESIGN.md 5.3).
                    #
                    # AUTO MUST STOP HERE, before the driver is ever called, rather than handing
                    # it None and letting it tap the first item to open the sheet and decide only
                    # THEN whether it can honour the pick. That older shape touched the screen
                    # (and, on a failed anchor re-ask, left the sheet open) before this method
                    # even knew whether targeting was possible -- and on a SUCCESSFUL re-ask it
                    # would still send a like attached to an item the model never chose, which is
                    # a silent substitution, not the hard stop doc 5.3 requires ("treat a missing
                    # table as a hard stop, never as a reason to fall back to a fixed
                    # coordinate"). Until the translation table exists (the next workflow's job
                    # -- see the STILL NOT BUILT paragraph below), auto mode genuinely cannot
                    # target correctly here, so it stops the whole run instead of mis-liking,
                    # leaving the screen exactly as it was: no tap, no opened sheet, no anchor
                    # re-ask, no swipe.
                    #
                    # Two different causes, named differently in stop_reason, because the
                    # operator's next move differs: "the model never named an item" is an
                    # opener-layer problem (retrying won't help until the prompt/model does
                    # better), while "the model named one but we can't act on it" is a missing-
                    # infrastructure problem (the driver-owned translation table has to land
                    # first) -- see hinge.py's `_like_comment_sheet` for the same distinction
                    # made about a DIFFERENT failure (a targeting miss discovered at swipe time).
                    #
                    # WHAT CHANGED 2026-08-12, and it is the reason auto can like at all again:
                    # INDEX_SPACE_MODEL_ITEMS is no longer untranslatable-in-practice. It still
                    # has no CAPTURE-ORDER equivalent -- `capture_order_index` returns None for
                    # it, correctly and unchanged, because the model's number resolves to a HEART
                    # ORDINAL and not to a frame -- but the driver now accepts that number
                    # directly as `model_item_index` and reaches it with doc 5.5's counting
                    # navigation (hinge._navigate_to_model_item). So the stop below narrowed from
                    # "any pick without a capture-order index" to "any pick this worker has no
                    # way at all to name an item with", which is ITEM_INDEX_ABSENT and any space
                    # nothing here recognises. `targeted` is what the two branches produce:
                    # exactly one of the two index arguments, never both, never neither.
                    targeted: dict = {}
                    if accepts_opener and pick is not None:
                        if pick.index != ITEM_INDEX_ABSENT \
                                and pick.index_space == INDEX_SPACE_MODEL_ITEMS:
                            targeted = {"item_index": None, "model_item_index": pick.index}
                        elif pick.capture_order_index is not None:
                            targeted = {"item_index": pick.capture_order_index}
                    if accepts_opener and pick is not None and not targeted:
                        if pick.index == ITEM_INDEX_ABSENT:
                            stop_reason = (
                                "opener generated no item number to target (the model could "
                                "not identify which numbered photo it was about), so there is "
                                "no item to attach the comment to and the like is not sent"
                            )
                        else:
                            stop_reason = (
                                f"opener targets model item {pick.index} in index space "
                                f"'{pick.index_space}', which nothing in this build can turn "
                                f"into a tapped heart (ops/OPENER-REDESIGN.md 5.3/5.6) -- the "
                                f"two spaces this worker can act on are the model's own item "
                                f"numbers (counting navigation) and the driver's capture order, "
                                f"and this pick is in neither, so the like is not sent"
                            )
                        stop_kind = "opener"
                        self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                        break
                    # item_index lets the driver attach the comment to the numbered photo the
                    # opener is actually about, not blindly the first one.
                    #
                    # THE TWO SIDES ARE IN DIFFERENT INDEX SPACES, AND THE CONVERSION IS
                    # EXPLICIT. `pick.index` is the MODEL ITEM INDEX (1-based over the numbered
                    # images the model was sent, ops/OPENER-REDESIGN.md 5.1/5.7); this driver
                    # parameter is a 0-based index into the profile CAPTURE ORDER, which
                    # HingeDriver resolves against its per-frame `_current_sigs` list (see
                    # hinge.py's _locate_target_heart). `capture_order_index` is the one
                    # sanctioned crossing between them and it refuses (None) rather than
                    # guessing whenever no sound conversion exists -- read its docstring before
                    # touching this line, and never write a +1/-1 here instead.
                    #
                    # THIS LINE WAS WRONG BETWEEN 2026-08-12 AND THE FIX, and the shape of the
                    # bug is worth keeping: it passed `pick.index` through unconverted, and the
                    # note that used to sit here claimed the mismatch was "bounded and visible
                    # rather than silent" because the driver reported a boolean `on_target=False`
                    # (a return value that no longer exists -- see below) for anything it could
                    # not honour. That was false for the common case. Its guard was
                    # `item_index >= len(_current_sigs)`, so on a 9-frame capture every value
                    # 1..8 was IN range: a 1-based item number resolved to a real, adjacent,
                    # wrong frame and came back ON TARGET, the repair hatch never fired, and the
                    # comment landed one card down with full confidence and no debug record.
                    # Only the last item and genuinely out-of-range values were ever caught. The
                    # lesson is not "add a bounds check", it is that a small int carrying no
                    # statement of its own space cannot be validated by the side that receives
                    # it. The flag itself is gone as of the 2026-08-12 substitution removal:
                    # `_locate_target_heart` now lands on the item it was asked for or raises, so
                    # "we are about to tap the wrong item" is no longer a state a caller has to
                    # remember to check.
                    #
                    # NOW BUILT (2026-08-12): doc 5.3's table EXISTS and doc 5.5's counting
                    # navigation SPENDS it. The model-item space no longer has to be converted at
                    # all -- it is handed to the driver as `model_item_index` and resolved to a
                    # heart ordinal there, by walking UP from where the profile read left the
                    # card. `capture_order_index` still returns None for that space, correctly:
                    # there is no frame the model's number names, and inventing one was the bug.
                    # Every line below this point may assume `targeted` names exactly one index
                    # in exactly one space, or that `pick` itself is None (no opener at all) --
                    # never "an opener exists but its target is unknown", which the stop above
                    # takes.
                    #
                    # THE `anchored_opener` REPAIR HATCH IS GONE (removed 2026-08-12, doc 5.1
                    # and 5.6). A closure used to be handed to the driver here so that, when its
                    # heart targeting missed, it could re-ask the model for text about whatever
                    # item the sheet had actually opened on. It was the last substitution path in
                    # the auto flow: the LIKE still landed on an item the model never chose, and
                    # only the wording was made to agree with it after the fact. Under the owner's
                    # never-substitute rule a targeting miss is a stop, so the driver now raises
                    # ItemTargetingError and the handler below turns it into one.
                    try:
                        # DOC 5.8: compare the model's coarse description with the exact
                        # numbered crop it selected BEFORE any navigation or tap.  A confident
                        # photo-vs-written disagreement means item numbering is suspect; it is
                        # the same never-substitute stop as a targeting miss, just discovered
                        # while the phone is still untouched.  Drivers that do not own numbered
                        # crops simply have no optional method and pass through unchanged.
                        mismatch = (_item_type_preflight_mismatch(self.driver, pick)
                                    if pick is not None else "")
                        if mismatch:
                            raise ItemTargetingError(
                                f"the model's chosen item failed the pre-flight type check: "
                                f"{mismatch}. Nothing was tapped and the like is NOT sent.",
                                stage="preflight", intended=pick.index,
                                index_space=pick.index_space)
                        # `**targeted` is empty when there is no pick at all, which leaves BOTH
                        # index arguments at their `None` default -- "nobody said which item",
                        # which is legal precisely because there is no opener to misplace. It is
                        # deliberately not `item_index=0`: 0 is a legal first frame in the
                        # driver's space and would read as "the opener is about item 1".
                        like_kwargs = dict(targeted)
                        if getattr(self.driver, "supports_interruptible_like_navigation", False):
                            like_kwargs["should_stop"] = self.stop_event.is_set
                        self.driver.like(pick.text if pick else None, **like_kwargs)
                    except DeckBlockedError as exc:
                        # Send Like can be refused only AFTER the action started: Hinge closes
                        # the comment sheet and puts up its out-of-free-likes paywall.  This is
                        # a known, normal blocking screen, not an unexpected action failure;
                        # above all it is NOT a completed like.  Stop before the counter and
                        # record_decision calls below, preserving the exact driver-facing reason
                        # that the between-profile blocked_reason() path already publishes.
                        terminal_state = "blocked"
                        stop_reason = str(exc)
                        stop_kind = "deck_blocked"
                        self._stat(state=terminal_state, stop_reason=stop_reason,
                                   stop_kind=stop_kind)
                        self.stop_event.set()
                        break
                    except ItemTargetingError as exc:
                        # DOC 5.6'S HARD STOP. The driver could not put this like on the item the
                        # opener was written about -- either it could not reach that item, or the
                        # sheet that opened was not showing it -- so it put it nowhere and left
                        # the screen exactly where it stopped, for debugging. Nothing was typed,
                        # nothing was sent, and the deck has not advanced.
                        #
                        # Caught here rather than left to run()'s generic handler, which would
                        # render it as state="error" with one traceback line: this is a DECISION
                        # the bot made correctly under a standing rule, not a crash, and an
                        # operator who cannot tell those apart will start ignoring red banners.
                        # The failure screenshot is still taken (the generic handler's own first
                        # act), because the screen is the whole diagnosis here.
                        #
                        # stop_kind="targeting", NOT the "opener" the stops above publish, and
                        # that is the whole of the fix this line used to carry a note about. It
                        # was "opener" because that was the channel the hub already branched on;
                        # the hub titles that branch "opener capacity exhausted", so the one stop
                        # whose entire point is that we refused to attach a real message to the
                        # wrong item rendered as a quota problem, with the truth demoted to the
                        # sub-line. Two things make this its own kind rather than better wording
                        # there: the CAUSE is not OpenerService at all (nothing was exhausted --
                        # the opener exists and is fine, the driver could not reach the item it
                        # is about), and the operator's next move is different (walk to the phone
                        # and read what is on it, possibly an open comment sheet with nothing
                        # typed in it, rather than check a quota). assets/hub.html renders it
                        # 'idle' rather than as an error box, for the same reason this handler
                        # exists at all -- see the paragraph above.
                        self._capture_failure(exc)
                        stop_reason = self._targeting_stop_reason(exc)
                        stop_kind = "targeting"
                        self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                        break
                    liked += 1
                else:
                    self.driver.dislike()

                # AUTO mode is pure INFERENCE: log the decision (for stats + the daily rate
                # limit) AFTER the action actually landed — do NOT store it as a training label
                # (training on the model's own prediction would create a self-reinforcing feedback
                # loop), and do NOT record a phantom if like()/dislike() raised (e.g. a
                # HingeActionError halt on an unsent like) and so never landed.
                # The decision and a committed opener share this opaque per-card id and exact
                # action timestamp. AUTO intentionally does not archive profiles as training
                # data, so this is lineage rather than an auto training profile.
                lineage_profile_id = uuid.uuid4().hex
                lineage_decision_at = time.time()
                _record_decision_with_lineage(
                    self.store, self.run_id, self.app, d.decision, d.score, source="auto",
                    profile_id=lineage_profile_id, created_at=lineage_decision_at)
                if d.decision == "like":
                    # The physical like returned successfully above, and its action row is now
                    # durable.  Commit the staged opener only after that row: a storage failure
                    # recording the decision must not leave an opener that downstream cleanup
                    # could mistake for evidence of a Like.  Stop arriving between the driver
                    # boundary and here does not matter -- the Like already landed.  Every
                    # preflight/targeting/paywall/transport failure broke out before either the
                    # decision or this commit.
                    commit = getattr(self.opener_service, "commit_opener", None)
                    if pick is not None and callable(commit):
                        if _accepts_keywords(commit, "profile_id", "decision", "decision_source",
                                             "decision_created_at"):
                            commit(pick, profile_id=lineage_profile_id, decision="like",
                                   decision_source="auto", decision_created_at=lineage_decision_at)
                        else:
                            commit(pick)
                if self.status:
                    self.status.record_swipe(self.app, d.decision, d.score)
                self._render()
                acted += 1
                today_acted += 1

                # no_face still lands as a conservative pass in the pre-existing
                # worker flow, even though its audit decision stays "no_face".
                # The policy must count the physical action only after it landed.
                landed_action = "like" if d.decision == "like" else "dislike"
                self._auto_policy.record_landed_action(landed_action)

                # getattr(..., False): runs after EVERY action, including a dislike-only run
                # that never once called maybe_opener() above -- so, unlike that call site,
                # self.opener_service being None must not even be a conditional branch here, it
                # must never be dereferenced unguarded at all. An audit proved the unguarded
                # `self.opener_service.stop_requested` raised `AttributeError: 'NoneType' object
                # has no attribute 'stop_requested'` on the very first dislike of a run
                # constructed with opener_service=None -- see the maybe_opener guard above for
                # the matching fix on the like path, and the observe loop's equivalent check
                # (_observe_loop, after a label is saved) for the pattern this mirrors.
                if getattr(self.opener_service, "stop_requested", False):
                    stop_reason = self._opener_stop_reason()
                    stop_kind = "opener"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                self._pace(landed_action, profile=profile, score=d.score)
                self._maybe_session_break()
        except ActionCancelled:
            # This is an operator Stop observed at a driver action boundary, not an unexpected
            # failure.  Let run() finish it normally without capturing a failure frame.
            raise
        except Exception as exc:  # noqa: BLE001
            self._capture_failure_if_unexpected(exc)          # snapshot WHILE the transport is live
            raise                                             # (the finally below closes it)
        finally:
            self._finish_session(terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)

    def _pace(self, decision: str, *, profile=None, score: float | None = None) -> None:
        # config.validate() already rejects a negative pacing.swipe_delay_s on every
        # path that goes through it (see config.py), but Worker is a public class any
        # caller can construct directly with a pacing object that skipped validation
        # (the same class of gap an audit already found for opener_service=None — see
        # the maybe_opener guards above). Event.wait() on a negative timeout returns
        # immediately, so an unguarded negative anchor here would silently produce
        # machine-speed swiping — a stronger bot signature than any jittered delay —
        # instead of the loud failure the calibrated branch below already gets for
        # free (post_action_delay_s's own `scale < 0` check, since scale is derived
        # from this same swipe_delay_s). Raise here too so BOTH branches fail exactly
        # the same way, not just the one that happens to route through that function.
        if self.pacing.swipe_delay_s < 0:
            raise ValueError("pacing.swipe_delay_s must be non-negative")
        # ``0`` is the documented test/no-pacing setting.  Do not consume the
        # session policy's random state merely to wait for zero seconds.
        if self.pacing.swipe_delay_s == 0 and profile is not None:
            return
        if not getattr(self.driver, "think_time_calibrated", False):
            # No app-specific calibration for this driver -> the flat, decision-agnostic
            # anchor (unchanged pre-existing behavior for e.g. Bumble).
            self.stop_event.wait(human_delay(self.pacing.swipe_delay_s))
            return
        # Decision-aware "think time" (measured like/pass dwell asymmetry), scaled by the
        # configured anchor so pacing.swipe_delay_s is a real knob rather than an on/off
        # switch. config.validate() bounds it: an unbounded scale lets a negative or
        # near-zero value collapse the wait to ~0 (Event.wait treats a negative timeout as
        # "return now"), i.e. machine-speed swiping on a live account.
        scale = self.pacing.swipe_delay_s / _THINK_TIME_BASELINE_S
        policy = getattr(self, "_auto_policy", None)
        if policy is not None and profile is not None:
            # Keep the old direct-call behavior for diagnostics and legacy tests;
            # actual auto-loop calls carry the real captured profile and score.
            delay = policy.post_action_delay_s(decision, profile,
                                                0.5 if score is None else score,
                                                scale=scale)
            self.stop_event.wait(delay)
            return
        self.stop_event.wait(think_time_s("like" if decision == "like" else "pass") * scale)

    def _maybe_session_break(self) -> None:
        """Randomized micro-break between profiles — mimics stepping away.

        Break likelihood ramps up the longer it's been since the last break (real
        attention fatigue isn't memoryless), and both the hazard rate and the typical
        interval are re-rolled after every break. A constant per-swipe probability
        would instead produce an exactly geometric gap distribution — a detectable,
        single-parameter bot signature no real human's break pattern has.
        """
        if self.pacing.swipe_delay_s == 0:
            return
        self._actions_since_break += 1
        fatigue = self._actions_since_break / self._break_due_after
        p = self._break_hazard * min(fatigue, 2.5)
        if random.random() < p:
            pause = human_delay(45.0, sigma=0.5)
            print(f"{self.app.title()} session micro-break: {pause:.0f}s")
            self._actions_since_break = 0
            self._break_hazard = random.uniform(0.04, 0.14)
            self._break_due_after = random.uniform(12, 35)
            self.stop_event.wait(pause)

    @staticmethod
    def _label_metadata(profile) -> dict:
        """Per-label provenance, passed to BOTH record_profile and add_label.

        capture_truncated says the driver hit its per-profile screencap ceiling without ever
        reaching the profile's bottom — the label is real, but it was made from an incomplete
        read of the person. It already reached the local debug log and Profile.meta; it stops
        here otherwise, and the local debug dir rotates and is not the system of record, so
        "which labels came from a partial read?" was unanswerable a week later. That is not a
        hypothetical: in the audited run of 2026-08-10 one of three profiles (12 photos, the
        configured ceiling exactly) was truncated.

        .get with a default rather than an index: `meta` is driver-authored, and the only
        other capture path (Bumble web) populates a different key set entirely — a missing key
        must mean "not truncated as far as anyone knows", never a KeyError in the label path.
        """
        meta = getattr(profile, "meta", None) or {}
        return {"photo_count": len(profile.photos),
                "capture_truncated": bool(meta.get("capture_truncated", False))}
