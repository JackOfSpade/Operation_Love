"""Per-app worker thread with two supported modes.

training — Hub-reviewed learning: the worker prepares a targeted, typed opener, but
           a human chooses Like or Dislike and only that verified result is labelled.
auto     — Autonomous scoring and actions with targeting safety checks before each like.

Multiple workers may share a ranker, store, and global budget. Unexpected errors halt
the affected worker so its failure state and diagnostics remain visible.
"""
from __future__ import annotations

import random
import inspect
import threading
import time
import traceback
import uuid
from datetime import date

from .config import PacingCfg, normalize_swipe_delay_s
from .drivers.base import (ActionCancelled, DatingAppDriver, DeckBlockedError, DriverClosed,
                           ItemTargetingError)
from .human import human_delay
from .human_motion import think_time_s
from .interaction import AutoSessionPolicy
from .notifications import notify_training_decision_ready
from .opener.opener import INDEX_SPACE_MODEL_ITEMS, ITEM_INDEX_ABSENT, ItemRequest
# The two `openers.decision` values a discard may carry, imported rather than spelled here so
# this module, OpenerService.discard_opener, and tools/opener_corpus_report.py's buckets can
# never drift into three copies of one vocabulary -- see the constants' own comments in
# opener/service.py. Which of the two a given abandonment gets is _post_like_discard_decision().
from .opener.service import DECISION_NEVER_SENT, DECISION_SEND_UNVERIFIED
from .ranker.decider import Decider, Decision
from .ranker.profile_key import profile_key_from_identity
from .targeting_policy import (
    installed_still_photo_licence, still_photo_licence_operator_notice,
    still_photo_licence_provenance, use_run_still_photo_licence)

_NO_PHOTO_RETRY_S = 0.5
# think_time_s() is calibrated to real measured Hinge dwell data around this many
# seconds — pacing.swipe_delay_s scales it proportionally, so the config knob still
# speeds up/slows down pacing (and a test fixture's swipe_delay_s=0.0 still paces
# instantly) while think_time_s supplies the measured like-vs-pass asymmetry shape.
# Derived from PacingCfg's own default so the baseline can't drift from the config.
_THINK_TIME_BASELINE_S = PacingCfg().swipe_delay_s
_STATUS_ERROR_MAX_CHARS = 600
_TARGETING_LICENCE_UNSET = object()


def _terminal_error_summary(exc: BaseException) -> str:
    """Return a compact, status-safe summary without losing a multiline error's cause.

    ``traceback.format_exc().splitlines()[-1]`` used to be convenient, but an ``AdbError``
    appends its stderr on a second line.  The Hub then displayed only that trailing stderr and
    discarded both ``AdbError`` and the timed-out command.  Status is a one-line UI/report
    field, not a traceback: retain the exception type and complete message, collapse whitespace,
    include PEP-678 notes (used by retry paths), and cap an unexpectedly verbose device error.
    The full traceback is still printed by ``run()``.
    """
    message = " ".join(str(exc).split())
    summary = type(exc).__name__
    if message:
        summary += f": {message}"
    notes = getattr(exc, "__notes__", ())
    if isinstance(notes, (list, tuple)):
        rendered_notes = [" ".join(note.split()) for note in notes if isinstance(note, str)]
        rendered_notes = [note for note in rendered_notes if note]
        if rendered_notes:
            summary += " [notes: " + "; ".join(rendered_notes) + "]"
    if len(summary) > _STATUS_ERROR_MAX_CHARS:
        summary = summary[:_STATUS_ERROR_MAX_CHARS - 1].rstrip() + "…"
    return summary


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


def _explicitly_accepts_keyword(callback, name: str) -> bool:
    """Whether an optional, observational keyword is declared by this store.

    Unlike action lineage, a progress callback is not part of the long-standing store protocol.
    Do not leak it through a legacy ``**metadata`` seam: only the BigQuery store that explicitly
    advertises it should receive progress updates.
    """
    try:
        return name in inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False


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


def _staged_opener_generation_context(pick) -> dict[str, object] | None:
    """Return the model's existing private draft fields for local pre-send diagnostics.

    Training intentionally does not commit an opener row before the reviewer acts, so the
    Hinge pre-send checkpoint is otherwise the last durable place these fields can be retained.
    Read only the already-created ``OpenerPick`` and its staging envelope; no new provider call,
    profile read, or reconstruction from the public opener text is permitted.  Older services
    and test picks may lack the envelope/model, in which case the available fields still travel
    honestly and no value is invented.
    """
    if pick is None:
        return None
    context: dict[str, object] = {}
    staged = getattr(pick, "_staged_record", None)
    model = getattr(staged, "model", None)
    if isinstance(model, str) and model:
        context["model"] = model
    for key in ("index_space", "referenced", "angle", "item_description"):
        value = getattr(pick, key, None)
        if isinstance(value, str) and value:
            context[key] = value
    return context or None


def _model_item_media_ordinal(driver, pick) -> int | None:
    """The picked item's photo/video position for the Hub review card, or ``None``.

    A REVIEW HINT, NEVER A GATE.  The training card names the model's item number and its
    description, but the reviewer is holding the phone and can only count what Hinge draws --
    and Hinge's per-card hearts run across photos, videos and written prompts alike.  This
    optional driver hook re-counts the same capture over media only, so the card can say which
    photo/video to look at.  Confirming the right item IS the point of the card
    (ops/OPENER-REDESIGN.md 5.6, "never substitute the liked item"), so the affordance is worth
    asking for -- but a missing one costs a hint and nothing else, and must never delay, halt or
    alter the checkpoint.  Hence every failure here answers ``None`` and the caller simply omits
    the field.

    Shaped exactly like ``_item_type_preflight_mismatch`` above and for the same reasons:
    method-presence rather than a base-class method so every existing generic driver stays
    valid; the same pure contract (inspect the payload already held, never capture, navigate or
    tap); the same refusal to hand a legacy profile-photo index to a hook that counts NUMBERED
    ITEM crops, which would number a different card entirely; and the same treatment of a broken
    optional implementation as simply unavailable.  The driver's own answer is already
    fail-closed, so a non-positive or non-integer return is discarded here too.
    """
    if getattr(pick, "index_space", None) != INDEX_SPACE_MODEL_ITEMS:
        return None
    count = getattr(driver, "model_item_media_ordinal", None)
    if not callable(count):
        return None
    try:
        ordinal = count(pick.index)
    except Exception:  # noqa: BLE001 — an optional review hint may never break a checkpoint
        return None
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        return None
    return ordinal

class Worker(threading.Thread):
    def __init__(self, app, driver: DatingAppDriver, decider: Decider, opener_service,
                 store, run_id, pacing, stop_event: threading.Event, mode: str = "training",
                 retrain_every: int = 1, limiter=None, max_restarts: int = 5, status=None,
                 training_action_bridge=None, targeting_licence=_TARGETING_LICENCE_UNSET):
        super().__init__(name=f"worker-{app}", daemon=True)
        self.app = app
        self.driver = driver
        self.decider = decider
        self.opener_service = opener_service
        self.store = store
        self.run_id = run_id
        self.pacing = pacing
        self.stop_event = stop_event
        if mode not in {"training", "auto"}:
            if mode == "auto_testing":
                raise ValueError("Worker mode 'auto_testing' was retired; use 'training' or 'auto'")
            raise ValueError("Worker mode must be 'training' or 'auto'")
        self.mode = mode
        self.retrain_every = max(1, int(retrain_every))
        self.limiter = limiter
        # Retained only for source compatibility with older AUTO callers. Active modes never
        # restart on unexpected errors, so it intentionally has no worker state or effect.
        _ = max_restarts
        self.status = status
        self.training_action_bridge = training_action_bridge
        self.training_action_supported = bool(
            mode == "training" and app == "hinge"
            and getattr(driver, "supports_training_decision", False))
        self._training_claimed_action = None
        # Set only after open_session() returns.  The normal loop finalizers
        # clear it via _finish_session(); the outer finally owns the narrow
        # post-open/pre-loop failure gap.
        self._session_opened = False
        # Capture the licence after supervisor's validated startup gate and before this thread
        # can touch the device. The Hub may inspect/revalidate config while we run; that must
        # never replace (or briefly clear) this session's safety basis.
        self._targeting_licence = (
            installed_still_photo_licence()
            if targeting_licence is _TARGETING_LICENCE_UNSET else targeting_licence)

    # --- live status (overlay + hub); no-ops when status is unset ---------
    def _stat(self, **fields) -> None:
        if self.status:
            self.status.set_app(self.app, **fields)
        self._render()

    def _publish_status(self, **fields) -> None:
        """Update shared status without repainting the driver overlay."""
        if self.status:
            self.status.set_app(self.app, **fields)

    def _render(self) -> None:
        if self.status:
            self.driver.render_status(self.status.app_view(self.app))

    def _install_opener_flag(self) -> None:
        """Tell the driver whether this Training or AUTO session can request an opener."""
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

    def _bind_opener_prompt_stamp(self) -> None:
        """Give an opt-in driver the prompt era this whole process's openers are generated
        under, so its own LOCAL debug rows (Hinge's actions.jsonl) can be grouped by era
        exactly like the durable `openers` table already can.

        A one-time bind, mirroring _bind_debug_run immediately above, is exactly as fresh as
        reading the value at generation time would be: OpenerService.prompt_sha256 is computed
        ONCE in its own __init__ and is fixed for the life of the process (see that attribute's
        own comment) -- a running process keeps the same style/prompt until restarted, so
        nothing is lost by handing it to the driver once here rather than threading it through
        every like() call. This deliberately does NOT hand the driver a reference to
        self.opener_service itself (a much wider surface than one digest), and every non-Hinge
        driver, every third-party/legacy opener service, and every test fake that lacks this
        optional hook or that `prompt_sha256` attribute is unaffected -- `getattr(..., None)`
        degrades to a local row with no era digest, exactly like a row written before this
        stamp existed.
        """
        bind = getattr(self.driver, "set_opener_prompt_sha256", None)
        if callable(bind):
            bind(getattr(self.opener_service, "prompt_sha256", None))

    def _announce_targeting_licence_provenance(self) -> None:
        """Publish the Hinge numbering licence once before the active session starts."""
        if self.app != "hinge":
            return
        notice = still_photo_licence_operator_notice()
        if not notice:
            return
        provenance = still_photo_licence_provenance()
        print(f"{self.app.title()}: {notice} [{provenance}].")
        if self.status:
            self.status.set_app(self.app, targeting_licence_notice=notice)

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
        """Snapshot an unexpected failure while the driver transport is still live."""
        if not isinstance(exc, DriverClosed):
            self._capture_failure(exc)

    def _finish_session(self, state: str = "stopped", *, stop_reason: str | None = None,
                        stop_kind: str | None = None) -> None:
        """Publish a terminal active-mode state, close the driver, and release Training actions."""
        # ``detail`` describes only in-flight work.  In particular, a long Hinge capture can
        # publish photo-item progress just before an opener failure; never leave that old
        # capture message displayed as the explanation for a terminal result.
        fields = {"state": state, "detail": None}
        if stop_reason:
            fields["stop_reason"] = stop_reason
        if stop_kind:
            fields["stop_kind"] = stop_kind
        self._stat(**fields)
        # The Hub and bug report retain this typed status snapshot, but their Recent logs are
        # deliberately only stdout/stderr.  Emit the terminal calibration refusal once here,
        # rather than at each preflight/capture call site, so every safe refusal is visible to
        # the operator alongside the exact live-build mismatch that caused it.
        if stop_kind == "targeting_calibration" and stop_reason:
            print(f"{self.app.title()}: stopped — targeting calibration must be renewed: "
                  f"{stop_reason}")
        try:
            self.driver.close()
        finally:
            self._session_opened = False
            if self.training_action_bridge:
                self.training_action_bridge.unregister(self)

    def _open_session(self) -> None:
        """Open the driver and record ownership for the outer teardown guard."""
        self.driver.open_session()
        self._session_opened = True

    def _opener_stop_reason(self) -> str | None:
        """Human-readable cause of an OpenerService-triggered stop, or None if the service
        hasn't recorded one. OpenerService.exhausted_reason is first-writer-wins and shared
        by every worker, so whichever worker reads it here always sees the SAME original
        cause -- not whichever symptom that particular worker's own opener call hit."""
        return getattr(self.opener_service, "exhausted_reason", None)

    def _discard_staged_opener(self, pick, *, decision_source: str, profile_key: str = "",
                               decision: str = DECISION_NEVER_SENT) -> None:
        """Best-effort telemetry for a staged AUTO/Training opener draft that will never be
        committed: an explicit Dislike, a Stop, a targeting refusal, or a driver exception -- see
        OpenerService.discard_opener's docstring for why this is the other half of commit_opener
        and why every one of those outcomes used to leave no durable trace at all.

        `decision` (2026-09-15) defaults to DECISION_NEVER_SENT because the overwhelming majority
        of call sites here refuse BEFORE the phone is touched at all -- no opener, no target, no
        like allowance, a Stop observed at the boundary -- and for those "never sent" is simply
        what happened. The three sites that wrap `driver.like()` itself cannot assume it, and
        must pass `_post_like_discard_decision()` instead: see that method.

        Guarded exactly like every commit_opener call site in this module (`callable(...)` on a
        duck-typed service): `discard_opener` is a newer, optional method, and a legacy/third-party
        opener service or test fake that predates it must degrade to a no-op here, not an
        AttributeError. `pick is None` (no opener was ever generated for this profile) is
        likewise a no-op, and a `pick` whose staged record is already spent (committed or
        discarded earlier) is a no-op inside discard_opener itself, so calling this
        defensively at more than one abandonment point is always safe.

        `profile_key` (2026-09-06): the caller's already-captured attribution key for THIS
        profile (see `_current_profile_key`) -- never re-derived here. Passed only when the
        service declares the keyword, exactly like every other optional lineage field this
        module threads through a duck-typed opener service.

        No try/except here, deliberately, mirroring every existing commit_opener call site in
        this module none of which wrap the call either: discard_opener carries its OWN
        best-effort try/except around the store write (see its docstring), so a failure there
        already cannot raise into this method's caller.
        """
        discard = getattr(self.opener_service, "discard_opener", None)
        if pick is not None and callable(discard):
            discard_kwargs = {"decision": decision, "decision_source": decision_source,
                              "decision_created_at": time.time()}
            if _accepts_keywords(discard, "profile_key"):
                discard_kwargs["profile_key"] = profile_key
            discard(pick, **discard_kwargs)

    def _post_like_discard_decision(self) -> str:
        """Which `decision` a draft abandoned by a RAISING `driver.like()` must be filed under.

        `driver.like()` IS NOT ATOMIC, and this is the whole point. HingeDriver types the opener
        into the composer, taps Send Like, and only then runs its verification -- the Rose-upsell
        dismissal, `_verify_like_landed`, and in Training `_verify_training_like_landed`'s stable
        next-card proof. Each of those raises HingeActionError on a POST-SEND failure ("like did
        not complete -- the like composer is still open"; "training action did not reach a
        stable, semantically different ready deck card"), and the handlers around like() used to
        answer all of them with `decision="never_sent"` -- a durable row, in the one table
        tools/opener_corpus_report.py and the outcome join trust, asserting that a message which
        physically went out to a real person was only ever a draft. The comments at those handlers
        claimed "anything raised ... means no reviewed Like landed for this profile either", which
        is true, and then filed a row that says something else: NOT LANDED AND NEVER SENT ARE
        DIFFERENT FACTS, and only the driver knows which side of its own irreversible boundary it
        got to.

        So ask it. DECISION_SEND_UNVERIFIED is the honest third value for the post-send window:
        the send happened, the outcome could not be verified. It is deliberately NOT "like"
        (nothing verified a landed Like, and "like" is the literal that decides which opener may
        own an owner-observed outcome, so an unconfirmed send must not be able to claim one) and
        deliberately not silence (a billed draft with no durable row at all is the survivorship
        hole discard_opener exists to close).

        Optional, duck-typed driver capability read with `getattr`, exactly like every other
        one this module consults (`supports_interruptible_like_navigation`,
        `landed_auto_opener_evidence`, `current_profile_identity`):
        `DatingAppDriver.like_send_attempted` defaults to False, so a driver that never
        implements it -- and a duck-typed test double that does not subclass the contract at all
        -- degrades to today's meaning, "nothing said a send happened", i.e.
        DECISION_NEVER_SENT, rather than raising an AttributeError inside a handler where
        another exception is already in flight.

        ONLY CALL THIS FROM A HANDLER AROUND THIS PROFILE'S OWN `like()` CALL. The driver marker
        is scoped to one like ATTEMPT and is cleared as the next one begins, not when one ends, so
        a successful send stays True until the following attempt clears it; reading it at an
        unrelated later refusal (which is why the default above is a constant and not this call)
        would describe THIS profile's untouched draft with the PREVIOUS profile's send.
        """
        attempted = getattr(self.driver, "like_send_attempted", None)
        if callable(attempted) and attempted():
            return DECISION_SEND_UNVERIFIED
        return DECISION_NEVER_SENT

    def _current_profile_key(self) -> str:
        """The STABLE, cross-time attribution key (ranker/profile_key.py) for the profile this
        driver most recently captured, or "" when none is derivable -- an honestly
        unattributable row, never a fabricated key.

        MUST BE CALLED BEFORE `driver.like()`/`driver.dislike()` RUNS FOR THIS PROFILE.
        HingeDriver's own item index -- the thing `current_profile_identity` reads -- is
        invalidated in EACH of those methods' own `finally` clause the instant the physical
        action completes (doc 5.3), so the identity for THIS profile is already gone by the
        time either call returns. `_training_loop`/`_auto_loop` therefore call this exactly
        once, right after the profile is captured and well before any device action, and carry
        the resulting plain string through to whichever of commit_opener/discard_opener this
        profile eventually reaches -- never re-derive it afterward.

        Optional, duck-typed driver hook, exactly like `_model_item_media_ordinal`/
        `_item_type_preflight_mismatch` above: a driver that does not enumerate items (every
        non-Hinge driver today) simply has no `current_profile_identity` method, and this
        degrades to "" rather than raising or guessing.
        """
        getter = getattr(self.driver, "current_profile_identity", None)
        if not callable(getter):
            return ""
        try:
            identity = getter()
            return profile_key_from_identity(identity) or ""
        except Exception:  # noqa: BLE001 -- optional telemetry must never break a live run
            return ""

    def _live_targeting_calibration_blocker(self) -> str:
        """Ask an opener-capable driver to re-check its live targeting licence."""
        if not getattr(self.driver, "accepts_opener", False):
            return ""
        blocker = getattr(self.driver, "targeted_suggestion_blocker", None)
        if not callable(blocker):
            return ""
        reason = blocker()
        return reason if isinstance(reason, str) else ""

    def _targeting_calibration_stop_reason(self, details: str) -> str:
        mode = "Training" if self.mode == "training" else "AUTO"
        outcome = ("No opener, action, or label was produced."
                   if self.mode == "training" else "No opener or like was sent.")
        return (
            f"{mode} cannot prepare a verifiable targeted opener because the live Hinge "
            "session has no valid schema-v3 targeting calibration. Capture and validate a "
            f"fresh targeting calibration for the live Hinge build/frame before resuming "
            f"{mode}. Details: {details}. {outcome}")

    def _request_training_decision(self, profile, pick, pre_send_frame: bytes,
                                   evidence=None) -> str:
        """Publish one typed, verified composer and wait for its human decision.

        The driver invokes this only after its keyboard-hide/re-verification boundary.  The
        returned string is consumed by the driver, which performs the actual verified Like or
        Dislike; the Worker records the label only after that call returns successfully.
        """
        bridge = self.training_action_bridge
        if bridge is None:
            self.stop_event.set()
            return "stop"
        # This is the one place that holds BOTH the live driver and the model's pick, which is
        # what counting the target's media position needs; the bridge only ever receives the
        # worker.  Computed before the publish call and outside its try, deliberately: the hint
        # is read-only and phone-free, and folding it into the call would let its absence share
        # a failure path with a checkpoint that genuinely could not be published.
        media_ordinal = _model_item_media_ordinal(self.driver, pick)
        # An older or substitute bridge that never learned the field must still get today's
        # exact call.  `_explicitly_accepts_keyword`, not `_accepts_keywords`: an optional
        # review hint must be DECLARED to be delivered -- it may not ride in through a legacy
        # `**kwargs` seam that would swallow it, and an uninspectable bridge is treated as not
        # having it rather than being handed a keyword whose TypeError would stop the run.
        ordinal_kwargs = (
            {"item_media_ordinal": media_ordinal}
            if media_ordinal is not None
            and _explicitly_accepts_keyword(
                bridge.publish_checkpoint, "item_media_ordinal") else {})
        try:
            # Profile.photos is the capture's existing top-to-bottom phone-scroll sequence.
            # The verified post-type frame remains the approval image; these additional frames
            # let the reviewer inspect the rest of the same profile without touching the phone.
            card = bridge.publish_checkpoint(
                self, pre_send_frame, pick, evidence,
                profile_frames=getattr(profile, "photos", ()) or (), **ordinal_kwargs)
        except (RuntimeError, ValueError) as exc:
            self._stat(state="stopped", stop_reason=(
                "Training checkpoint could not be published; no action was issued: "
                f"{exc}"), stop_kind="approval")
            self.stop_event.set()
            return "stop"
        self._stat(state="waiting_approval", detail=None)
        # The checkpoint is now complete and actionable, and the worker is about to block for
        # the operator.  Keep this best-effort host alert outside TrainingActionBridge: bridge
        # publication also happens in protocol tests and carries no macOS/UI responsibility.
        notify_training_decision_ready()
        action = bridge.wait_for_action(self, card["profile_token"], self.stop_event)
        if action is None or self.stop_event.is_set():
            bridge.cancel_checkpoint(self, card["profile_token"])
            self.stop_event.set()
            return "stop"
        self._training_claimed_action = action
        self._stat(state="acting", detail=None)
        return action["command"]

    def _training_persistence_status(self, outcome: str, stage: str,
                                     completed: int | None = None,
                                     total: int | None = None) -> None:
        """Show durable-write progress after the phone has already accepted a choice.

        The Hinge decision is irreversible before profile archival begins.  Keeping that fact in
        every subsequent live detail prevents a slow cloud write from looking like a stuck tap or
        an unanswered decision.  ``stage`` is intentionally a small Worker-facing protocol so
        stores can optionally add per-photo progress without deciding Hub wording themselves.
        """
        landed = f"{outcome.title()} landed in {self.app.title()}"
        if stage == "profile_upload":
            if (type(completed) is int and type(total) is int and total > 0
                    and 0 <= completed <= total):
                detail = (f"{landed}; archiving profile screenshots "
                          f"({completed}/{total})")
            else:
                detail = f"{landed}; archiving profile screenshots"
        elif stage == "profile_uploaded":
            detail = f"{landed}; profile archive complete—recording the training label"
        elif stage == "opener_evidence":
            detail = f"{landed}; archiving the typed opener and send evidence"
        elif stage == "label":
            detail = f"{landed}; recording the decision and training label"
        elif stage == "flush":
            detail = f"{landed}; archive complete—flushing the label and evidence to storage"
        else:
            detail = f"{landed}; archiving the reviewed profile and training data"
        self._publish_status(mode="training", state="acting", detail=detail)

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


    def _capture_profile(self, method: str):
        """Capture a profile, passing stop/progress callbacks only to opt-in drivers."""
        capture = getattr(self.driver, method)
        if self.mode != "training":
            if getattr(self.driver, "supports_interruptible_capture", False):
                return capture(should_stop=self.stop_event.is_set)
            return capture()
        set_progress = getattr(self.driver, "set_capture_progress_callback", None)
        progress_installed = False

        def progress(detail: str) -> None:
            if not self.stop_event.is_set():
                try:
                    self._publish_status(mode=self.mode, state="scoring", detail=detail)
                except Exception:  # noqa: BLE001 -- capture-status plumbing is observational
                    pass

        if callable(set_progress):
            try:
                set_progress(progress)
                progress_installed = True
            except Exception:  # noqa: BLE001 -- optional status hook must not block capture
                pass
        self._publish_status(mode=self.mode, state="scoring",
                             detail="reading this profile and verifying its photo items")
        try:
            if getattr(self.driver, "supports_interruptible_capture", False):
                return capture(should_stop=self.stop_event.is_set)
            return capture()
        finally:
            if progress_installed:
                try:
                    set_progress(None)
                except Exception:  # noqa: BLE001 -- optional status hook must not block capture
                    pass
            # The callback can have last reported an individual photo while the final frame was
            # being processed.  Clear it before the caller moves into local validation or the
            # potentially slow Gemini model cascade, rather than letting a completed capture
            # look alive while the Hub's timestamp goes stale.
            self._publish_status(mode=self.mode, state="scoring", detail=None)




    def run(self) -> None:
        """Run under the targeting licence captured before this thread started."""
        with use_run_still_photo_licence(self._targeting_licence):
            self._run_with_targeting_licence()

    def _run_with_targeting_licence(self) -> None:
        def leave() -> None:
            if self.training_action_bridge:
                self.training_action_bridge.unregister(self)

        if self.status:
            self.status.set_app(self.app, mode=self.mode)
        self._bind_debug_run()
        self._bind_opener_prompt_stamp()
        self._announce_targeting_licence_provenance()
        if self.stop_event.is_set():
            leave()
            return
        if self.training_action_bridge:
            self.training_action_bridge.register(self)
        try:
            self._training_loop() if self.mode == "training" else self._auto_loop()
        except ActionCancelled:
            self.stop_event.set()
        except DriverClosed as exc:
            print(f"{exc}; Stopping run so buffered data can be saved.")
            self.stop_event.set()
        except Exception as exc:  # noqa: BLE001
            print(f"{self.app.title()} unexpected error in {self.mode} mode; HALTING "
                  f"(no restart) so nothing swipes blindly and the debug logs survive:")
            traceback.print_exc()
            self._stat(state="error", detail=None,
                       error=_terminal_error_summary(exc))
            self.stop_event.set()
        finally:
            try:
                # _training_loop/_auto_loop normally close through
                # _finish_session().  Their session setup precedes those inner
                # try/finally blocks, though, so a post-open calibration/status
                # error otherwise leaked a live transport.  Close exactly once
                # here for that ownership gap.
                if self._session_opened:
                    self.driver.close()
                    self._session_opened = False
            finally:
                leave()

    def _retrain_after_labels(self, added: int) -> None:
        ready = self.decider.retrain(self.store)
        if self.status:
            self.status.set_global(ranker_ready=ready)
        self._render()
        print(f"Learned {added} labels this run; ranker ready={ready}")

    # --- Hub-reviewed training: prepare every profile, human decides -----------------
    def _training_loop(self) -> None:
        """Create manual training labels without consulting the preference model.

        This intentionally does not share AUTO's decision section: even a seemingly harmless
        call to ``decider.decide`` would make a model prediction part of a human-ground-truth
        workflow.  Hinge owns all device input after the callback command is returned; this
        method owns the durable decision/profile/embedding transaction afterwards.
        """
        if self.training_action_bridge is None:
            self._stat(state="stopped", stop_reason=(
                "Training requires the local Hub decision bridge; no automation was started"))
            self.stop_event.set()
            return
        if (not self.training_action_supported
                or not callable(getattr(self.driver, "set_training_decision", None))):
            self._stat(state="stopped", stop_reason=(
                "Training requires Hinge's verified Hub Like/Dislike decision hook; "
                "no automation was started"), stop_kind="targeting")
            self.stop_event.set()
            return
        if (self.opener_service is None or getattr(self.opener_service, "disabled", True)
                or not getattr(self.driver, "accepts_opener", False)):
            self._stat(state="stopped", stop_reason=(
                "Training requires an enabled opener service and an opener-capable driver; "
                "no automation was started"), stop_kind="opener")
            self.stop_event.set()
            return

        self._install_opener_flag()
        # Classify the whole run before the first read, so the device-input ledger does not
        # label a supervised run's scrolls as autonomous ones.
        begin_training = getattr(self.driver, "begin_training_session", None)
        if callable(begin_training):
            begin_training()
        # AndroidDriver uses this hook to mark the session as device-driven before it chooses
        # whether to attach its touch watcher.
        # ``None`` deliberately installs no AutoSessionPolicy: training must not consult or
        # adapt to the preference model.
        set_policy = getattr(self.driver, "set_auto_session_policy", None)
        if callable(set_policy):
            set_policy(None)
        self._open_session()
        targeting_blocker = self._live_targeting_calibration_blocker()
        if targeting_blocker:
            stop_reason = self._targeting_calibration_stop_reason(targeting_blocker)
            self.stop_event.set()
            self._finish_session(
                "stopped", stop_reason=stop_reason, stop_kind="targeting_calibration")
            return
        self._stat(mode="training", state="scoring")
        self._actions_since_break = 0
        self._break_hazard = random.uniform(0.04, 0.14)
        self._break_due_after = random.uniform(12, 35)
        acted = liked = 0
        labels_added = 0
        last_retrained = 0
        pending_error = False
        has_daily_limit = self.limiter is not None and self.limiter.max_per_day is not None
        # Training decisions carry ``source=manual``.  Do not ask the historical AUTO-only
        # counter here: that would make a daily cap silently reset between Training runs.
        today0 = (self.store.count_today(self.app, source="manual")
                  if has_daily_limit else 0)
        today_acted = 0
        today_date = date.today()
        terminal_state, stop_reason, stop_kind = "stopped", None, None
        try:
            while not self.stop_event.is_set():
                targeting_blocker = self._live_targeting_calibration_blocker()
                if targeting_blocker:
                    terminal_state = "stopped"
                    stop_reason = self._targeting_calibration_stop_reason(targeting_blocker)
                    stop_kind = "targeting_calibration"
                    self._stat(state=terminal_state, stop_reason=stop_reason,
                               stop_kind=stop_kind)
                    self.stop_event.set()
                    break
                if has_daily_limit:
                    today = date.today()
                    if today != today_date:
                        today0, today_acted, today_date = (
                            self.store.count_today(self.app, source="manual"), 0, today)
                blocked = self.driver.blocked_reason()
                if blocked is not None:
                    terminal_state, stop_reason, stop_kind = "blocked", blocked, "deck_blocked"
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
                    break

                profile = self._capture_profile("next_profile")
                if self.stop_event.is_set():
                    break
                if profile is None:
                    blocked = self.driver.blocked_reason()
                    if blocked is not None:
                        # NOT ALWAYS A BLOCKED DECK (2026-09-15). `blocked_reason` is the only
                        # channel a capture that returns None has for carrying a sentence, so a
                        # FAILED COLD-RELAUNCH RECOVERY latches its reason here too -- and this
                        # branch used to hardcode stop_kind="deck_blocked", making the Hub
                        # headline "stopped -- deck blocked" when nothing was blocking the deck
                        # at all. That is precisely the surface the owner rule "the hub must
                        # clearly show why a run stopped" is about, so ask the driver which KIND
                        # of stop it latched. Optional and duck-typed exactly like every other
                        # driver hook in this module: a driver without the method (every
                        # non-Hinge driver today) keeps the historical "deck_blocked" answer,
                        # and so does one that latched a reason without classifying it.
                        stop_kind = getattr(
                            self.driver, "blocked_stop_kind", lambda: None)() or "deck_blocked"
                        terminal_state, stop_reason = "blocked", blocked
                        self._stat(state=terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                    break
                # Captured HERE, before anything else touches this profile: the driver's item
                # index (and the identity fingerprint it carries) is invalidated the instant
                # like()/dislike() completes (see _current_profile_key's own docstring), so this
                # is the last point this profile's attribution key can be read at all.
                profile_key = self._current_profile_key()
                self._publish_status(
                    mode="training", state="scoring",
                    detail="checking the captured profile for a safe target")
                if not profile.photos:
                    self._stat(last_decision="no_photos", state="scoring", detail=None)
                    self.stop_event.wait(_NO_PHOTO_RETRY_S)
                    continue

                unavailable = getattr(profile, "items_unavailable", "")
                unnumbered = getattr(profile, "items_unnumbered", "")
                unavailable_kind = getattr(profile, "items_unavailable_kind", "")
                if unavailable_kind == "targeting_calibration":
                    stop_reason = self._targeting_calibration_stop_reason(unavailable)
                    stop_kind = "targeting_calibration"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                    break
                if unavailable or unnumbered:
                    stop_reason = (
                        "Training could not prepare a verifiable targeted opener for this profile: "
                        f"{unavailable or unnumbered}. No action or label was recorded.")
                    stop_kind = "opener"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                    break
                items = ItemRequest.from_profile(profile) if getattr(profile, "items", ()) else None
                # The loop-top probe protects the whole capture. Recheck at the provider
                # boundary too: Hinge can update or the display mode can change while the
                # profile is being read, and an opener must never be requested from crops whose
                # calibration became invalid during that read.
                targeting_blocker = self._live_targeting_calibration_blocker()
                if targeting_blocker:
                    stop_reason = self._targeting_calibration_stop_reason(targeting_blocker)
                    stop_kind = "targeting_calibration"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                    break
                opener_kwargs = {"items": items, "should_stop": self.stop_event.is_set}
                if (callable(getattr(self.opener_service, "commit_opener", None))
                        and _accepts_keywords(self.opener_service.maybe_opener, "stage")):
                    opener_kwargs["stage"] = True
                # This status deliberately covers the whole configured Gemini cascade,
                # including a timeout or model fallback.  It is observational only: no request
                # shape, retry policy, timing, or model selection changes here.
                self._publish_status(
                    mode="training", state="scoring", detail="generating a targeted opener")
                pick = self.opener_service.maybe_opener(
                    self.run_id, self.app, profile, **opener_kwargs)
                if self.stop_event.is_set() or getattr(self.opener_service, "stop_requested", False):
                    stop_reason, stop_kind = self._opener_stop_reason(), "opener"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                    # EVERY post-generation exit in this loop discards the staged draft, for the
                    # reason the Dislike branch below spells out at length: the provider call
                    # above already finished and was already BILLED (record_spend runs inside
                    # maybe_opener), so a draft abandoned here reached neither the durable
                    # `openers` table nor recent_openers -- it existed in no bug report and in no
                    # corpus report, which is precisely the survivorship bias discard_opener was
                    # added to end. AUTO has discarded its equivalent abandonment paths since
                    # that path shipped; Training's did not until now. This particular exit is
                    # the common one: an operator Stop cannot abort a provider request already on
                    # the wire, so maybe_opener hands back a complete staged OpenerPick after the
                    # flag has already won and no review checkpoint will ever be prepared for it.
                    #
                    # decision_source="manual", NOT "training": that is the lineage value the
                    # commit/discard sites at the bottom of this loop already pass, and it is
                    # what commit_opener turns into session_mode="training" for the recent-opener
                    # view.  Calling this blindly is safe -- see _discard_staged_opener's
                    # docstring: `pick is None` and an already-spent staged record are both
                    # no-ops, and it never raises into this loop.
                    self._discard_staged_opener(
                        pick, decision_source="manual", profile_key=profile_key)
                    break
                text = getattr(pick, "text", None)
                if not isinstance(text, str) or not text.strip():
                    stop_reason = ("Training requires a complete generated opener before a profile "
                                   "can be reviewed; no action or label was recorded")
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind="opener")
                    self.stop_event.set()
                    # A malformed pick is still a real billed attempt, and a draft whose TEXT is
                    # unusable is exactly the kind of row the corpus report most needs to see.
                    # A no-pick/unstaged `pick` degrades to a no-op here (see the first exit).
                    self._discard_staged_opener(
                        pick, decision_source="manual", profile_key=profile_key)
                    break
                targeted = {}
                if (getattr(pick, "index", ITEM_INDEX_ABSENT) != ITEM_INDEX_ABSENT
                        and getattr(pick, "index_space", None) == INDEX_SPACE_MODEL_ITEMS):
                    targeted = {"item_index": None, "model_item_index": pick.index}
                elif (getattr(self.driver, "supports_capture_order_training_target", False)
                      and getattr(pick, "capture_order_index", None) is not None):
                    targeted = {"item_index": pick.capture_order_index}
                if not targeted:
                    stop_reason = ("Training opener did not name a target the driver can verify; "
                                   "no action or label was recorded")
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind="opener")
                    self.stop_event.set()
                    # The opener itself generated fine; only its target could not be resolved, so
                    # no review checkpoint was ever prepared for this profile. AUTO discards its
                    # identical "not targeted" refusal for the same reason (see _auto_loop).
                    self._discard_staged_opener(
                        pick, decision_source="manual", profile_key=profile_key)
                    break
                if self.limiter and not self.limiter.allow_like(liked):
                    terminal_state = "rate_limited"
                    self._stat(state=terminal_state)
                    # The like allowance ran out AFTER this profile's draft was generated and
                    # billed, so the draft is abandoned with the card exactly like the refusals
                    # above -- a rate-limited run must not be a hole in the opener corpus.
                    self._discard_staged_opener(
                        pick, decision_source="manual", profile_key=profile_key)
                    break

                self._training_claimed_action = None
                # The staged record is intentionally not committed until a reviewed Like lands.
                # Give an opt-in driver its private generation notes only for this checkpoint so
                # an unacted draft remains diagnosable without pretending it is an opener row.
                generation_context_hook = getattr(
                    self.driver, "set_staged_opener_generation_context", None)
                if callable(generation_context_hook):
                    try:
                        generation_context_hook(_staged_opener_generation_context(pick))
                    except Exception:  # noqa: BLE001 -- optional diagnostics cannot block review
                        generation_context_hook = None
                training_decision_installed = False
                try:
                    self.driver.set_training_decision(
                        lambda frame, evidence, profile=profile, pick=pick: self._request_training_decision(
                            profile, pick, frame, evidence))
                    training_decision_installed = True
                    self._publish_status(
                        mode="training", state="acting",
                        detail="preparing the review checkpoint")
                    like_kwargs = dict(targeted)
                    if getattr(self.driver, "supports_interruptible_like_navigation", False):
                        like_kwargs["should_stop"] = self.stop_event.is_set
                    outcome = self.driver.like(text, **like_kwargs)
                except ActionCancelled:
                    # A Stop arriving while the driver waits at the review checkpoint leaves the
                    # phone untouched by the driver contract, so the billed draft is abandoned
                    # and must still be written down. The original exception is re-raised
                    # completely unchanged.
                    #
                    # THE DECISION IS ASKED FOR, NOT ASSUMED (2026-09-15): the sentence above
                    # describes the checkpoint Stop, which is where this handler fires almost
                    # always -- but ActionCancelled is also raised at the driver's LAST
                    # pre-input boundary checks, and one of those sits between the Send Like tap
                    # and the verification that follows it. `_post_like_discard_decision` reads
                    # the driver's own send marker rather than trusting this comment to stay
                    # true of every path that can reach it.
                    #
                    # BEFORE the bridge completion below, deliberately, and the same at the two
                    # abandonment sites that follow: this loop's shape is that every durable
                    # write for an action happens first and ``complete`` is the last thing the
                    # worker says about it (see the success path, which completes only after the
                    # decision, opener, label and flush have all landed). An abandoned draft is
                    # the same kind of record, just a different `decision`.
                    self._discard_staged_opener(
                        pick, decision_source="manual", profile_key=profile_key,
                        decision=self._post_like_discard_decision())
                    action = self._training_claimed_action
                    if action is not None:
                        self.training_action_bridge.complete(
                            action, status="aborted", reason="training decision was cancelled")
                    raise
                except Exception:
                    # Same reasoning as AUTO's generic like() handler: anything raised while
                    # installing the checkpoint or attempting the physical action means no
                    # reviewed Like landed for this profile either, so its draft is abandoned.
                    # Scoped, like AUTO's, to the try block above and nothing after it -- the
                    # durable archival below runs only once a real action has physically landed
                    # and must never be able to make that landed action look abandoned.
                    #
                    # "NO REVIEWED LIKE LANDED" IS NOT "NOTHING WAS SENT", and Training is where
                    # the gap is widest: after the reviewer chooses Like, the driver taps Send
                    # Like and only then demands a stable, semantically different ready deck
                    # card, raising HingeActionError when it cannot get one. The opener is out
                    # by then. `_post_like_discard_decision` asks the driver which side of that
                    # boundary this exception came from instead of filing every one of them as
                    # a draft nobody ever received.
                    self._discard_staged_opener(
                        pick, decision_source="manual", profile_key=profile_key,
                        decision=self._post_like_discard_decision())
                    action = self._training_claimed_action
                    if action is not None:
                        self.training_action_bridge.complete(
                            action, status="failed", reason="the device action did not complete")
                    raise
                finally:
                    # Keep the established callback semantics: if installation itself raised,
                    # do not issue a second ``None`` call against a driver whose state is
                    # unknown.  The optional generation context is different: it was already
                    # bound before installation, so it must be cleared on every later path.
                    if training_decision_installed:
                        self.driver.set_training_decision(None)
                    if callable(generation_context_hook):
                        try:
                            generation_context_hook(None)
                        except Exception:  # noqa: BLE001 -- best-effort diagnostic cleanup
                            pass
                action = self._training_claimed_action
                self._training_claimed_action = None
                if outcome not in {"like", "dislike"} or action is None:
                    # A stop/cancellation leaves the phone untouched by the driver contract.
                    # No verified Like/Dislike means no reviewed decision exists for this
                    # profile, so its billed draft is abandoned here too -- on the clean Stop
                    # below and on the RuntimeError, and before either, because the raise would
                    # otherwise carry this row straight past the only place it can be written
                    # down. Ordered ahead of the bridge completion for the same reason as the
                    # two handlers above.
                    #
                    # ONLY THE OUTCOME HALF OF THIS DISJUNCTION MAY DISCARD (2026-09-15): the
                    # condition is entered by two independent failures, and they disagree about
                    # what happened on the phone. `outcome not in {...}` means the driver never
                    # performed a verified action, so nothing was sent. `action is None` with a
                    # like/dislike outcome means the OPPOSITE -- the opener was typed and
                    # physically SENT, and it is the Hub claim that went missing (a stale or
                    # replaced bridge registration). Discarding there wrote a durable
                    # `decision="never_sent"` opener row for an opener that actually landed:
                    # wrong data in the one table the corpus report trusts, published moments
                    # before the RuntimeError below. That half now writes no decision row at all
                    # -- an unattributed sent opener is recoverable, a row asserting it was
                    # never sent is not.
                    if outcome not in {"like", "dislike"}:
                        self._discard_staged_opener(
                            pick, decision_source="manual", profile_key=profile_key)
                    if action is not None:
                        self.training_action_bridge.complete(
                            action, status="aborted", reason="training decision was cancelled")
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError("training driver returned no verified Like/Dislike outcome")
                if outcome != action.get("command"):
                    self.training_action_bridge.complete(
                        action, status="failed", reason="driver outcome disagreed with the Hub decision")
                    raise RuntimeError("training driver outcome disagreed with the Hub decision")

                # The verified Hinge action has now physically landed.  All following work is
                # durable archival, not device input; publish that boundary before a potentially
                # slow image upload so Hub wording cannot imply that the Like/Dislike is pending.
                self._training_persistence_status(outcome, "archive")
                try:
                    profile_id, decided_at = uuid.uuid4().hex, time.time()
                    # Validate the local, pure parts of a training datum before writing any
                    # durable action lineage.  The phone action is already irreversible, but
                    # an unavailable embedder must not also consume a manual daily allowance or
                    # leave an opener/decision fragment that the Hub correctly reports failed.
                    metadata = self._label_metadata(profile)
                    vec = self.decider.embed(profile)
                    if vec is None:
                        raise RuntimeError(
                            "the training profile could not be embedded; no label was saved")
                    record_profile = self.store.record_profile
                    profile_kwargs = dict(source="manual", photos=profile.photos, **metadata)
                    # BigQuery can report each image upload.  Local/legacy stores deliberately
                    # do not receive this optional UI-only callback through ``**metadata``.
                    if _explicitly_accepts_keyword(record_profile, "progress"):
                        profile_kwargs["progress"] = (
                            lambda stage, completed=None, total=None, outcome=outcome:
                            self._training_persistence_status(
                                outcome, stage, completed, total))
                    archived = record_profile(
                        self.run_id, self.app, profile_id, outcome == "like", **profile_kwargs)
                    if archived is False:
                        raise RuntimeError(
                            "the training profile could not be archived; no label was saved")
                    if outcome == "like":
                        self._training_persistence_status(outcome, "opener_evidence")
                        commit = getattr(self.opener_service, "commit_opener", None)
                        if callable(commit):
                            landed_evidence = None
                            evidence_hook = getattr(
                                self.driver, "landed_auto_opener_evidence", None)
                            if callable(evidence_hook):
                                landed_evidence = evidence_hook()
                            if _accepts_keywords(commit, "profile_id", "decision", "decision_source",
                                                 "decision_created_at"):
                                commit_kwargs = {
                                    "profile_id": profile_id,
                                    "decision": "like",
                                    "decision_source": "manual",
                                    "decision_created_at": decided_at,
                                }
                                if (landed_evidence is not None
                                        and _accepts_keywords(commit, "pre_send_evidence")):
                                    commit_kwargs["pre_send_evidence"] = landed_evidence
                                if _accepts_keywords(commit, "profile_key"):
                                    commit_kwargs["profile_key"] = profile_key
                                committed = commit(pick, **commit_kwargs)
                            else:
                                committed = commit(pick)
                            # OpenerService deliberately turns a post-send storage outage into
                            # ``False`` so AUTO can keep its already-landed action.  Training's
                            # Hub result has a stricter promise: it cannot say the reviewed Like
                            # completed when its typed opener/evidence was not persisted.
                            if committed is False:
                                raise RuntimeError(
                                    "the landed training opener could not be persisted")
                    else:
                        # The reviewer chose Dislike. Until now this staged draft was simply
                        # dropped with the card: the durable `openers` table only ever recorded
                        # a Like, so "how often does the model write a bad opener" could not be
                        # answered from it -- the bad ones were exactly the ones never written
                        # down. discard_opener is the other half of the commit_opener call
                        # immediately above (see its docstring): same staging envelope, same
                        # prompt_sha256 captured at generation, same best-effort persistence, and
                        # it must never turn a completed Dislike into a failed Hub action --
                        # which is exactly why, unlike the Like branch above, its return value is
                        # not inspected and a discard failure never raises here.
                        discard = getattr(self.opener_service, "discard_opener", None)
                        if callable(discard):
                            discard_kwargs = {
                                "profile_id": profile_id, "decision": "dislike",
                                "decision_source": "manual", "decision_created_at": decided_at,
                            }
                            if _accepts_keywords(discard, "profile_key"):
                                discard_kwargs["profile_key"] = profile_key
                            discard(pick, **discard_kwargs)
                    self._training_persistence_status(outcome, "label")
                    _record_decision_with_lineage(
                        self.store, self.run_id, self.app, outcome,
                        1.0 if outcome == "like" else 0.0, source="manual",
                        profile_id=profile_id, created_at=decided_at)
                    self.store.add_label(
                        self.run_id, self.app, outcome == "like", vec, source="manual",
                        profile_id=profile_id, profile_name=profile.name, **metadata)
                    flush = getattr(self.store, "flush", None)
                    if not callable(flush):
                        raise RuntimeError(
                            "Training requires a store with a durable flush operation")
                    # BigQuery deliberately buffers each table independently.  A Hub action is
                    # not completed until its decision, opener/evidence, archive, and label have
                    # crossed that boundary; otherwise a process loss can turn a green Hub
                    # result into missing training data.
                    self._training_persistence_status(outcome, "flush")
                    flush()
                    self.training_action_bridge.complete(action, status="completed")
                except Exception:
                    self.training_action_bridge.complete(
                        action, status="failed", reason="training action could not be persisted")
                    raise
                # Completion means durable action data exists.  Training/repainting is useful
                # follow-up work, but a retrain/status failure must never rewrite that fact as a
                # failed Hub action.
                labels_added += 1
                if self.status:
                    self.status.record_swipe(self.app, outcome)
                    self.status.inc_labels(1)
                # Every durable write for this action has now crossed the flush boundary, so stop
                # claiming one is in flight: retrain, pacing, and an occasional session break
                # publish nothing, and the next status write is the following capture's. Published
                # AFTER record_swipe on purpose -- that call re-stamps state="acting" and leaves
                # ``detail`` alone, so a clear placed before it would leave the Hub asserting an
                # unfinished device action for the whole pacing window instead of a finished one.
                self._publish_status(
                    mode="training", state="scoring",
                    detail=f"{outcome.title()} recorded and saved; pacing before the next profile")
                if labels_added % self.retrain_every == 0:
                    self._retrain_after_labels(labels_added)
                    last_retrained = labels_added
                acted += 1
                today_acted += 1
                if outcome == "like":
                    liked += 1
                self._pace(outcome, profile=profile,
                            score=1.0 if outcome == "like" else 0.0)
                self._maybe_session_break()
        except ActionCancelled:
            # Stop can arrive while the driver is waiting for the Hub's Training
            # decision.  That is an intentional shutdown at an action boundary,
            # not a device or targeting failure.  In particular, do not retain an
            # ``unexpected`` failure screenshot for the normal Stop path.
            raise
        except DeckBlockedError as exc:
            terminal_state, stop_reason, stop_kind = "blocked", str(exc), "deck_blocked"
            self._stat(state=terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)
            self.stop_event.set()
        except Exception as exc:
            pending_error = True
            self._capture_failure_if_unexpected(exc)
            raise
        finally:
            try:
                # Match the manual-label contract: normal runs retrain in bounded batches,
                # then flush a final partial batch before the session closes.
                if labels_added and labels_added != last_retrained:
                    try:
                        self._retrain_after_labels(labels_added)
                    except Exception:
                        if not pending_error:
                            raise
                        print(f"{self.app.title()} final training retrain skipped after shutdown error:")
                        traceback.print_exc()
            finally:
                self._finish_session(terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)

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
        # This state belongs to one auto session, not to the learned preference model. It may
        # only make a marginal model like more conservative; it never manufactures a like.
        # Fakes/third-party deciders need not expose the underlying model, hence the guarded
        # configured-threshold lookup.
        configured_threshold = getattr(getattr(self.decider, "model", None), "threshold", 0.5)
        try:
            configured_threshold = float(configured_threshold)
        except (TypeError, ValueError):
            configured_threshold = 0.5
        if not 0.0 < configured_threshold < 1.0:
            configured_threshold = 0.5
        self._auto_policy = AutoSessionPolicy(base_threshold=configured_threshold)
        # AndroidDriver uses this optional hook for both the contextual policy and the
        # unconditional AUTO-session marker. Other drivers can ignore it, while lightweight
        # fakes and older plugins receive the value through a harmless instance attribute.
        install_policy = getattr(self.driver, "set_auto_session_policy", None)
        if callable(install_policy):
            install_policy(self._auto_policy)
        else:
            try:
                self.driver._auto_policy = self._auto_policy
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
        has_daily_limit = (self.limiter is not None
                           and self.limiter.max_per_day is not None)
        today0 = self.store.count_today(self.app) if has_daily_limit else 0
        today_acted = 0            # actions this worker made since today0 was last measured
        today_date = date.today()  # LOCAL day — matches the store's count_today() day boundary
        self._open_session()
        targeting_blocker = self._live_targeting_calibration_blocker()
        if targeting_blocker:
            stop_reason = self._targeting_calibration_stop_reason(targeting_blocker)
            self.stop_event.set()
            self._finish_session(
                "stopped", stop_reason=stop_reason, stop_kind="targeting_calibration")
            return
        self._stat(mode=self.mode, state="scoring")
        terminal_state = "stopped"            # overwritten below when the loop ends for a known reason
        stop_reason = None                    # set for an OpenerService- or deck-blocked stop; see _finish_session
        stop_kind = None                      # disambiguates stop_reason's source; see _finish_session/status.py
        try:
            while not self.stop_event.is_set():
                targeting_blocker = self._live_targeting_calibration_blocker()
                if targeting_blocker:
                    terminal_state = "stopped"
                    stop_reason = self._targeting_calibration_stop_reason(targeting_blocker)
                    stop_kind = "targeting_calibration"
                    self._stat(state=terminal_state, stop_reason=stop_reason,
                               stop_kind=stop_kind)
                    self.stop_event.set()
                    break
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
                # for something actually wrong). The same reasoning applies here even though
                # AUTO mode's own actions are what triggered Hinge's paywall this time.
                # blocked_reason() itself never taps/swipes/types (see its contract on
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
                    # A foreground/package race can be discovered *inside* capture, after the
                    # between-profile blocked_reason() probe above. AndroidDriver latches that
                    # precise reason and returns no profile without issuing input. Re-read it
                    # before treating ``None`` as an ordinary end of the deck, otherwise the
                    # hub reports a clean stop for a run that is actually waiting on System UI.
                    blocked = self.driver.blocked_reason()
                    if blocked is not None:
                        terminal_state = "blocked"
                        stop_reason = blocked
                        # Same classification as _training_loop's own capture-returned-None
                        # branch, for the same reason spelled out there: a latched reason is not
                        # proof of a blocked deck, and a failed cold relaunch must not be
                        # headlined as one. Absent hook -> the historical "deck_blocked".
                        stop_kind = getattr(
                            self.driver, "blocked_stop_kind", lambda: None)() or "deck_blocked"
                        self._stat(state=terminal_state, stop_reason=stop_reason,
                                   stop_kind=stop_kind)
                        self.stop_event.set()
                    break

                # Captured HERE, before anything else touches this profile -- see
                # _current_profile_key's own docstring for why this is the last point this
                # profile's attribution key can be read at all (like()/dislike() invalidate the
                # driver's index, and the identity fingerprint it carries, the instant either
                # completes).
                profile_key = self._current_profile_key()
                self._stat(state="scoring")
                d = self.decider.decide(profile)
                if self.stop_event.is_set():
                    break
                if d.decision == "defer":
                    self._stat(last_decision="defer", state="stopped")
                    print(f"{self.app.title()} ranker not ready (cold-start) — run training "
                          f"mode and make Hub decisions manually to seed it. Stopping {self.app}.")
                    break

                # A session can decline a marginal model like according to its bounded
                # fatigue/context state. This happens before every limit check so a
                # policy-demoted like neither consumes a like budget nor trips the running
                # like-ratio guard.
                d = self._auto_policy.apply_decision(d, profile).decision

                # Per-run like budget: stop the run rather than mislabel a wanted like as a
                # pass (keeps the right-swipe ratio human; see limits.py).
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
                    # instead of this codebase's usual clear, actionable message.
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
                    # A THIRD state exists (found+fixed 2026-08-22, see `Profile.items_unnumbered`):
                    # enumeration ran to completion and legitimately numbered nothing (a profile
                    # of videos, say). This whole opener block only runs for a LIKE decision (see
                    # the `if d.decision == "like":` a few lines up) -- a PASS decision never
                    # reaches this code at all, so it sails straight past a zero-item profile
                    # unconditionally, which is the correct, preserved behaviour
                    # (ops/STILL-PHOTO-DISCRIMINATOR.md 5d measured exactly this: a normal
                    # profile, not a failure).
                    #
                    # For a LIKE decision, though, this state gets its OWN hard stop immediately
                    # below, right after the `items_unavailable` one (found+fixed 2026-08-22,
                    # second pass, same day -- the first pass left this branch as the comment used
                    # to read here: "`items` stays `None`... no opener is requested"). That claim
                    # was false. With `items` left `None` and nothing else stopping the run,
                    # control fell straight through to `maybe_opener(..., items=None)`, which DOES
                    # request an opener -- just from the raw scroll frames instead of numbered
                    # crops, exactly the ambiguity doc 5.2 exists to remove ("one card appears in
                    # several frames, one frame can hold two cards, and the returned item number
                    # would be confidently meaningless"). Worse, on Hinge the pick that comes back
                    # would still be treated as naming a real item and a like would be sent against
                    # it, in direct violation of the owner's never-substitute-liked-item rule: if
                    # we cannot land on the exact item the model chose, the run stops rather than
                    # sending a like attached to something else. The block below closes that gap.
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
                        targeting_blocker = self._live_targeting_calibration_blocker()
                        if targeting_blocker:
                            stop_reason = self._targeting_calibration_stop_reason(
                                targeting_blocker)
                            stop_kind = "targeting_calibration"
                            self._stat(state="stopped", stop_reason=stop_reason,
                                       stop_kind=stop_kind)
                            self.stop_event.set()
                            break
                        unavailable = getattr(profile, "items_unavailable", "")
                        if unavailable:
                            unavailable_kind = getattr(profile, "items_unavailable_kind", "")
                            if unavailable_kind == "targeting_calibration":
                                stop_reason = self._targeting_calibration_stop_reason(unavailable)
                                stop_kind = "targeting_calibration"
                            else:
                                stop_reason = (
                                    f"this profile could not be enumerated into numbered items, so "
                                    f"there is no honest opener request to make and the like is not "
                                    f"sent: {unavailable}. (ops/OPENER-REDESIGN.md 5.2 -- sending "
                                    f"the raw scroll frames instead would give the model a numbering "
                                    f"nothing can act on, so this stops rather than degrades.)"
                                )
                                stop_kind = "opener"
                            self._stat(state="stopped", stop_reason=stop_reason,
                                       stop_kind=stop_kind)
                            self.stop_event.set()
                            break
                        unnumbered = getattr(profile, "items_unnumbered", "")
                        if unnumbered:
                            # The `items_unavailable` stop above is "the capture failed"; this
                            # one is "the capture worked and nothing survived policy" -- a
                            # different operator situation, so it gets its own reason rather than
                            # being folded into the wording above. Today that usually means dwell
                            # coverage (ops/STILL-PHOTO-DISCRIMINATOR.md 5d: a burst that only
                            # ever reached one card of fifteen), not anything broken, which is
                            # exactly why the sentence is quoted from the driver's own per-capture
                            # `items_unnumbered` rather than a hardcoded guess at the cause --
                            # this repo's standing rule that guidance must derive from the
                            # condition it describes (see `Profile.items_unnumbered`'s docstring).
                            stop_reason = (
                                f"the ranker wants to LIKE this profile, but enumeration ran to "
                                f"completion and legitimately numbered nothing, so there is no "
                                f"verifiable item an opener request could name and the like is "
                                f"not sent: {unnumbered}. (Owner rule: if we cannot land on the "
                                f"exact item the model would choose, the run stops rather than "
                                f"substituting a different one or falling back to the raw scroll "
                                f"frames -- ops/OPENER-REDESIGN.md 5.2.)"
                            )
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
                    # A Stop can land while a single already-started provider request is on the
                    # wire.  The opener service cannot abort that request synchronously, so it
                    # may hand back a valid staged draft after the flag has won.  Do not let that
                    # result flow into item preflight/navigation: no new device action may begin
                    # after Stop, and the staged envelope remains uncommitted because no Like
                    # landed.  Training has the equivalent check immediately after its opener
                    # call; AUTO needs the same boundary before it evaluates `stop_requested`,
                    # which is a different service-health signal and remains false for an
                    # operator-initiated stop.
                    if self.stop_event.is_set():
                        # This draft was generated but will never reach a device: no like was
                        # attempted and none ever will be for this profile. See
                        # _discard_staged_opener's docstring for why this is safe even though a
                        # legacy/fake opener service may not implement discard_opener at all.
                        self._discard_staged_opener(
                            pick, decision_source="auto", profile_key=profile_key)
                        break
                    # An opener call can discover that every configured provider/model is
                    # exhausted. In AUTO mode, honour its global stop BEFORE calling
                    # like(): a bare like is not an acceptable substitute for the opener
                    # the worker decided to send.
                    if getattr(self.opener_service, "stop_requested", False):
                        stop_reason = self._opener_stop_reason()
                        stop_kind = "opener"
                        self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                        # Exhaustion (e.g. the run budget, see maybe_opener's docstring) can fire
                        # on the SAME call that produced this profile's staged draft -- the
                        # generation itself succeeded, only the service now refuses to let AUTO
                        # act on it. Same reasoning as the stop_event branch above.
                        self._discard_staged_opener(
                            pick, decision_source="auto", profile_key=profile_key)
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
                        # The opener itself generated fine; only its target could not be
                        # resolved, so no like was ever attempted for this profile.
                        self._discard_staged_opener(
                            pick, decision_source="auto", profile_key=profile_key)
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
                    pre_send_evidence = None
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
                        # The stop check above covers a completed provider request; repeat it
                        # at the physical-action boundary because target preflight may itself
                        # take enough time for an operator Stop to arrive.  Interruptible
                        # drivers receive the callback as their final in-driver guard, but this
                        # also keeps a non-interruptible legacy driver from being entered when
                        # Stop was already observed before the call.
                        if self.stop_event.is_set():
                            self._discard_staged_opener(
                                pick, decision_source="auto", profile_key=profile_key)
                            break
                        self.driver.like(pick.text if pick else None, **like_kwargs)
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
                        # stop_kind="targeting", NOT the "opener" or "targeting_calibration"
                        # kinds the pre-opener stops above publish. The hub gives this failure
                        # its own title because its CAUSE is not OpenerService or stale setup --
                        # nothing was exhausted and calibration was usable:
                        # the opener exists and is fine, the driver could not reach the item it
                        # is about. The operator's next move is different too: walk to the phone
                        # and read what is on it, possibly an open comment sheet with nothing
                        # typed in it, rather than renew calibration. assets/hub.html renders it
                        # 'idle' rather than as an error box, for the same reason this handler
                        # exists at all -- see the paragraph above.
                        self._capture_failure(exc)
                        stop_reason = self._targeting_stop_reason(exc)
                        stop_kind = "targeting"
                        self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                        self.stop_event.set()
                        self._discard_staged_opener(
                            pick, decision_source="auto", profile_key=profile_key)
                        break
                    except Exception:
                        # Anything else raised while attempting the physical action -- a paywall
                        # DeckBlockedError, an operator Stop arriving mid-navigation as
                        # ActionCancelled, an unsent-like RuntimeError/driver error -- means no
                        # VERIFIED like exists for this profile. The outer handlers below still
                        # decide how the RUN reacts (stopped/blocked/error); this only records
                        # that this profile's own staged draft was abandoned, and the original
                        # exception is re-raised completely unchanged.
                        #
                        # THIS HANDLER USED TO SAY "the like was never sent", AND FOR PART OF
                        # ITS RANGE THAT WAS FALSE (2026-09-15). The paywall case is the clearest
                        # example: `_verify_like_landed` runs AFTER the Send Like tap, and it is
                        # the post-send frame that reveals the Hinge+ screen. Filing every one of
                        # these as `never_sent` put openers that had already gone out into the
                        # durable table as drafts nobody received.
                        # `_post_like_discard_decision` reads the driver's own send marker and
                        # files the post-send window as `send_unverified` instead -- still not a
                        # Like (nothing verified one), just no longer a lie in the other
                        # direction. A driver with no marker at all is unchanged.
                        #
                        # Deliberately scoped to ONLY the preflight check and the driver.like()
                        # call above (see the try block this pairs with): the evidence hook right
                        # below runs AFTER driver.like() already returned, i.e. after a real Like
                        # physically landed, and must never be able to make a landed action look
                        # abandoned merely because that OPTIONAL diagnostic read raised.
                        self._discard_staged_opener(
                            pick, decision_source="auto", profile_key=profile_key,
                            decision=self._post_like_discard_decision())
                        raise
                    evidence_hook = getattr(self.driver, "landed_auto_opener_evidence", None)
                    if pick is not None and callable(evidence_hook):
                        pre_send_evidence = evidence_hook()
                    liked += 1
                else:
                    dislike = self.driver.dislike
                    if (not getattr(self.driver, "supports_interruptible_dislike", False)
                            or not _accepts_keywords(dislike, "should_stop")):
                        raise RuntimeError(
                            f"{self.app} driver cannot perform a Stop-safe AUTO pass: it must "
                            "declare supports_interruptible_dislike and accept should_stop")
                    # The driver owns the final check immediately before physical
                    # input.  A second worker-side `is_set()` check cannot close
                    # the race this capability exists to prevent.
                    dislike(should_stop=self.stop_event.is_set)

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
                            commit_kwargs = {
                                "profile_id": lineage_profile_id, "decision": "like",
                                "decision_source": "auto",
                                "decision_created_at": lineage_decision_at,
                            }
                            if (pre_send_evidence is not None
                                    and _accepts_keywords(commit, "pre_send_evidence")):
                                commit_kwargs["pre_send_evidence"] = pre_send_evidence
                            if _accepts_keywords(commit, "profile_key"):
                                commit_kwargs["profile_key"] = profile_key
                            commit(pick, **commit_kwargs)
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
                if self._auto_policy is not None:
                    self._auto_policy.record_landed_action(landed_action)

                # getattr(..., False): runs after EVERY action, including a dislike-only run
                # that never once called maybe_opener() above -- so, unlike that call site,
                # self.opener_service being None must not even be a conditional branch here, it
                # must never be dereferenced unguarded at all. An audit proved the unguarded
                # `self.opener_service.stop_requested` raised `AttributeError: 'NoneType' object
                # has no attribute 'stop_requested'` on the very first dislike of a run
                # constructed with opener_service=None -- see the maybe_opener guard above for
                # the matching fix on the like path for the same terminal-state pattern.
                if getattr(self.opener_service, "stop_requested", False):
                    stop_reason = self._opener_stop_reason()
                    stop_kind = "opener"
                    self._stat(state="stopped", stop_reason=stop_reason, stop_kind=stop_kind)
                    self.stop_event.set()
                self._pace(landed_action, profile=profile, score=d.score)
                self._maybe_session_break()
        except ActionCancelled:
            # This is an operator Stop detected at a driver action boundary, not an unexpected
            # failure.  Let run() finish it normally without capturing a failure frame.
            raise
        except DeckBlockedError as exc:
            # A known blocking screen can surface at any final Android input boundary: not only
            # Send Like, but a pass gesture or a foreground race during navigation. It is an
            # ordinary operator-actionable stop, never a completed decision and never an
            # unexpected failure snapshot. Centralising the catch here keeps like and dislike
            # semantically identical and prevents either from falling through run()'s red error
            # path merely because their driver calls live in different branches above.
            terminal_state = "blocked"
            stop_reason = str(exc)
            stop_kind = "deck_blocked"
            self._stat(state=terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)
            self.stop_event.set()
        except Exception as exc:  # noqa: BLE001
            self._capture_failure_if_unexpected(exc)          # snapshot WHILE the transport is live
            raise                                             # (the finally below closes it)
        finally:
            self._finish_session(terminal_state, stop_reason=stop_reason, stop_kind=stop_kind)

    def _pace(self, decision: str, *, profile=None, score: float | None = None) -> None:
        # Worker is public and can be constructed without config.validate(). Reuse the exact
        # config invariant so strings/bools/non-finite values, near-zero anti-bot bypasses,
        # and platform-timeout-sized values all fail before either pacing branch executes.
        swipe_delay_s = normalize_swipe_delay_s(self.pacing.swipe_delay_s)
        # ``0`` is the documented test/no-pacing setting.  Do not consume the
        # session policy's random state merely to wait for zero seconds.
        if swipe_delay_s == 0 and profile is not None:
            return
        if not getattr(self.driver, "think_time_calibrated", False):
            # No app-specific calibration for this driver -> the flat, decision-agnostic
            # anchor (unchanged pre-existing behavior for e.g. Bumble).
            self.stop_event.wait(human_delay(swipe_delay_s))
            return
        # Decision-aware "think time" (measured like/pass dwell asymmetry), scaled by the
        # configured anchor so pacing.swipe_delay_s is a real knob rather than an on/off
        # switch. config.validate() bounds it: an unbounded scale lets a negative or
        # near-zero value collapse the wait to ~0 (Event.wait treats a negative timeout as
        # "return now"), i.e. machine-speed swiping on a live account.
        scale = swipe_delay_s / _THINK_TIME_BASELINE_S
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
        future capture paths may populate a different key set entirely — a missing key must
        mean "not truncated as far as anyone knows", never a KeyError in the label path.
        """
        meta = getattr(profile, "meta", None) or {}
        return {"photo_count": len(profile.photos),
                "capture_truncated": bool(meta.get("capture_truncated", False))}
