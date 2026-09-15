"""Common interface every app driver implements.

The worker (worker.py) is app-agnostic: it only knows these methods. Current concrete
drivers use host-side ADB (screencap perception + humanized `input motionevent` gestures
at screen-fraction coordinates — no uiautomator2 / on-device helper, see
ops/HINGE-PIXEL-RUNBOOK.md §5). A generic Playwright base remains reference-only.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

from ..perception.capture import Profile


@contextmanager
def _time_bucket(stamps: dict[str, float] | None, key: str):
    """Add this block's wall clock to `stamps[key]`, or do nothing at all if `stamps` is None.

    The shared primitive behind every timing ledger in this codebase -- originally written for
    `hinge.py`'s per-frame read-loop ledger (see `HingeDriver._capture_current`'s and
    `_index_captured_items`' own TIMING LEDGER paragraphs) and now reused, unchanged, one level
    down for the per-gesture ledger inside the humanized touch transport (uhid.py, adb.py).
    Living here rather than in hinge.py is what makes that reuse possible without a circular
    import: hinge.py already imports from uhid.py and adb.py, so those two could not import a
    hinge.py-defined helper back without a cycle. base.py is the one module every one of them
    already imports from (DriverClosed), so the shared primitive lives here instead of being
    duplicated three times -- which is exactly the kind of copy this task was told to avoid.

    `stamps is None` is how a caller with no debug log active opts out: no `time.monotonic()`
    call, no dict write, nothing but the contextmanager protocol itself -- matching every other
    diagnostic in this codebase, which goes fully quiet the moment its own `self._dbg is None`.
    Accumulates with `stamps.get` rather than overwriting, because a bucket can legitimately be
    split across more than one code path in a single iteration or gesture (identity tracking
    runs in two separate `if` blocks in the read loop; a reverse gesture's foreground recheck
    and zone check each run in two different methods) and the ledger wants ONE number for the
    concept, not whichever call happened to run last. A `break`/`return`/raised exception inside
    the block still reaches the `finally` below -- the contextmanager protocol guarantees that --
    so a boundary that cuts an iteration or a gesture short is still timed accurately up to the
    point it left.
    """
    if stamps is None:
        yield
        return
    started = time.monotonic()
    try:
        yield
    finally:
        stamps[key] = stamps.get(key, 0.0) + (time.monotonic() - started)


class DriverClosed(RuntimeError):
    """The user closed the app/browser window during a run."""


class ActionCancelled(RuntimeError):
    """The operator stopped an action before its next device input.

    This is neither a targeting refusal nor a driver fault.  No further capture, gesture, text
    entry, or send may occur; workers end normally without recording a decision or a failure.
    """


class DeckBlockedError(RuntimeError):
    """A known screen is standing between the driver and its swipe deck.

    This is a normal, operator-actionable terminal condition, not an unexpected
    driver failure: for example, Hinge may replace the deck with its out-of-free-
    likes upgrade screen after Send Like.  The worker catches this app-agnostic
    type to publish its existing ``blocked`` terminal state without recording the
    action that caused the screen to appear.
    """


class ItemTargetingError(RuntimeError):
    """The like could NOT be put on the item the opener was written about, so it was not put
    anywhere. Doc 5.6's hard stop, in the one shape worker.py can catch without importing a
    concrete driver.

    Raised by a driver that targets per item (Hinge) at either of the two moments the question
    can be answered: BEFORE the tap, when the chosen item cannot be reached or its crops cannot
    serve as a verification reference, and AFTER the tap, when the opened like/comment sheet is
    not showing that item. Both are the same operator-facing fact -- "we could not honour the
    choice the model made" -- and both call for the same next move, so they are one type carrying
    `stage` rather than two.

    THE OWNER RULE THIS EXISTS TO ENFORCE (ops/OPENER-REDESIGN.md 5.6): never substitute a
    different item. There is no `hearts[0]`, no "closest reachable item", and no rewriting the
    opener to match whatever we hit -- every one of those ships a real message about the wrong
    thing to a real person. Retrying the SAME item is allowed and is what the driver does first;
    this is what is left when the retries are spent.

    The screen is deliberately left exactly as it is when this is raised (a scrolled profile, or
    an open sheet with nothing typed) so the failure can be read off the phone.

    `intended` and `actual` are the two numbers doc 5.6 asks a stop record to hold. `actual` is
    None whenever we never got far enough to see what we hit -- "we could not reach item 4" and
    "we reached something and it was item 6" are different diagnoses, and None says which one this
    is rather than pretending to an answer. `stage` is "navigate" (before the tap) or "verify"
    (after it); `index_space` names WHICH numbering `intended` is in, because this codebase has
    two and a bare small int that does not state its space is the exact bug the 2026-08-12 audit
    found (see opener.py's INDEX_SPACE_* constants)."""

    def __init__(self, message: str, *, stage: str = "", intended=None, actual=None,
                 index_space: str = "") -> None:
        super().__init__(message)
        self.stage = stage
        self.intended = intended
        self.actual = actual
        self.index_space = index_space


OBSERVE_ITEM_MATCH = "match"
OBSERVE_ITEM_MISMATCH = "mismatch"
OBSERVE_ITEM_INCONCLUSIVE = "inconclusive"
_OBSERVE_ITEM_STATES = frozenset({
    OBSERVE_ITEM_MATCH, OBSERVE_ITEM_MISMATCH, OBSERVE_ITEM_INCONCLUSIVE,
})


@dataclass(frozen=True)
class ObserveItemCheck:
    """Three-valued result for checking an Observe composer against a suggested item.

    ``mismatch`` means the check positively identified the wrong profile/item. ``inconclusive``
    means the frame could not answer (animation, keyboard/selection overlay, missing geometry or
    unavailable vision). Conflating those states made a transient typing frame revoke an item
    that an earlier frame of the same continuously open composer had already verified.
    """

    state: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state not in _OBSERVE_ITEM_STATES:
            raise ValueError(f"unknown Observe item-check state {self.state!r}")
        if self.state != OBSERVE_ITEM_MATCH and not self.reason:
            raise ValueError(f"Observe item-check state {self.state!r} requires a reason")

    def __bool__(self):
        raise TypeError("ObserveItemCheck has no truth value -- inspect .state")


def open_debug_log(debug_dir: str, *, run_id: str | None = None,
                   keep_runs: int | None = None, protect_runs=()):
    """Best-effort DebugLog bootstrap shared by every driver's open_session().

    Returns a DebugLog instance, or None if construction fails. Debug logging must
    never block a run — but failing silently would leave the operator thinking
    diagnostics are on when they're not, so a construction failure is printed.
    """
    try:
        from .debuglog import DebugLog
        return DebugLog(debug_dir, run_id=run_id, keep_runs=keep_runs,
                        protect_runs=protect_runs)
    except Exception as exc:  # noqa: BLE001 — logging must never break a run
        print(f"Debug log unavailable ({type(exc).__name__}: {exc}); continuing without it.")
        return None


def snapshot_failure_frame(dbg, exc: BaseException, capture_frame) -> None:
    """Worker hook shared by every driver's snapshot_failure(): best-effort capture of
    the on-screen state into the debug log so an unexpected error is reconstructable.

    `capture_frame` is a zero-arg callable returning the current frame as bytes (a
    screenshot/screencap); if it raises, the error is still logged, just without a
    frame. No-op when debug logging is off (`dbg` is None). Never raises — this runs
    inside the worker's except handler and must not mask the real error.
    """
    if dbg is None:
        return
    try:
        frame = capture_frame()
    except Exception:  # noqa: BLE001 — page/device may already be gone; log the error anyway
        frame = None
    try:
        dbg.error("unexpected", frame, exc)
    except Exception:  # noqa: BLE001 — logging must never mask the real error being reported
        pass


class DatingAppDriver(ABC):
    # Whether this app lets you attach a written opener AT SWIPE TIME (Hinge's
    # "send a like with a comment" model). When False, the worker must NOT call
    # the opener service in auto mode — generating a Gemini opener we can't send
    # just wastes spend. Bumble is False (you match first, then message).
    accepts_opener: bool = True

    # Legacy manual-observation integrations can report the intermediate "Send Like" sheet.
    # Keep the opt-in capability for their compatibility surface; it is not a runnable mode.
    supports_observe_like_intent: bool = False

    # Whether next_profile()/current_profile() honour a `should_stop` callable, i.e.
    # whether a Stop pressed WHILE a profile is being captured is noticed during the
    # capture instead of only after it finishes.
    #
    # This matters because capture is by far the longest uninterruptible stretch in a
    # run. On Hinge, reading one profile is 12 screencaps + 11 humanized read-scrolls,
    # and the legacy manual-observation path then scrolls the whole card back to the top again — measured at
    # ~85s end to end, during which the worker's only stop check (the one right after
    # current_profile() returns) cannot run. The operator's report was exactly that:
    # "when I hit stop, it doesn't stop while it's reading a profile, it completes the
    # read (by scrolling a bunch) then stops."
    #
    # Opt-in, for the same reason supports_observe_like_intent is (see above): drivers
    # and lightweight test doubles that declare a zero-argument capture keep working
    # untouched. Unlike halt_on_error's original default (see its comment below), False
    # here is the SAFE value, not the risky one — it means "Stop is honoured between
    # profiles, exactly as it always was", never a new failure mode. Worker only passes
    # should_stop to a driver that declares True, so a flag that lies (True on a driver
    # that ignores the callable) is worse than no flag at all.
    supports_interruptible_capture: bool = False

    # Whether like() accepts a should_stop callable and honours it while doing its
    # pre-tap navigation. Kept opt-in so existing drivers and lightweight fakes retain
    # their public like() signature.
    supports_interruptible_like_navigation: bool = False

    # Whether dislike() accepts a should_stop callable and honours it at the final
    # pre-input boundary.  Unlike a capture, a pass must not be allowed to begin
    # after an operator has pressed Stop: a worker-side check alone has a race
    # between that check and the device gesture.  This is deliberately opt-in so
    # an older driver cannot claim a guarantee it does not implement; AUTO fails
    # closed instead of issuing an uninterruptible pass through such a driver.
    supports_interruptible_dislike: bool = False

    # Whether human_motion.think_time_s()'s per-decision "think time" (measured
    # like-vs-pass dwell asymmetry) is calibrated for THIS app's real behavior.
    # True only for Hinge, which it was actually measured on; other drivers get
    # the flat, decision-agnostic pacing anchor instead of a borrowed model.
    think_time_calibrated: bool = False

    # On an UNEXPECTED error, should the worker halt the run instead of restarting the
    # session and carrying on? Declared here, defaulting to True, so the safe behaviour is
    # what a driver gets by DEFAULT rather than what it has to remember to ask for.
    #
    # It used to live only on the Android driver, and worker.py read it as
    # `getattr(driver, "halt_on_error", False)` — so a driver that simply never mentioned
    # the attribute (PlaywrightDriver did not) silently received restart-with-backoff. That
    # is fail-open: the riskier behaviour was the one you got by forgetting.
    #
    # Restarting is not free even in a manual-observation compatibility session, where the bot
    # only reads. The worker
    # re-attaches to whatever is on screen; if the driver was confused about which card it
    # was looking at, the manual swipes it records afterwards are attributed to the wrong
    # profile — which corrupts the taste model permanently, long after the session that
    # caused it. Ending a seeding session early is cheap by comparison.
    #
    # Concrete drivers may lower it from config (apps.<app>.halt_on_error), but config
    # validation refuses `false` for anything running in auto mode.
    halt_on_error: bool = True

    # Optional doc 5.8 capability: a concrete, crop-owning driver may expose
    # `item_type_preflight(item_description, model_item_index) -> ItemTypePreflight`.
    # It MUST be pure over its already-stored crop: no capture, gesture, typing, transport, or
    # logging.  This is intentionally only documented, not supplied as a base method, so method
    # presence remains the capability and generic drivers/test doubles need no implementation.

    @abstractmethod
    def open_session(self) -> None:
        """Attach to the app or physical device and reach the swipe deck."""

    @abstractmethod
    def next_profile(self, *, should_stop=None) -> Profile | None:
        """Capture the current profile (all photos + text). None when the deck is empty.

        `should_stop` is an optional zero-argument callable polled BETWEEN the individual
        screencaps/scrolls a capture is made of; when it returns True the driver abandons
        the capture and returns None promptly. It is only ever passed to drivers that
        declare ``supports_interruptible_capture`` (see that attribute), so implementations
        that ignore it — and test doubles that don't declare it at all — are unaffected.

        A None returned because should_stop fired is deliberately indistinguishable from
        None for any other reason: both worker loops already treat None as "nothing usable
        here, look again", and both re-check the stop event immediately after this call, so
        no new return value or exception is needed to carry the difference."""

    @abstractmethod
    def like(self, opener: str | None = None, item_index: int | None = None, *,
             model_item_index: int | None = None, should_stop=None) -> str | None:
        """Like the current profile, optionally sending an opener message. item_index is the
        0-based index (capture order) of the photo/prompt the opener is about, so drivers that
        comment per-item (Hinge) can target it; drivers without that notion ignore it.

        THIS PARAMETER IS IN THE DRIVER'S OWN CAPTURE-ORDER SPACE AND NOTHING ELSE. It is NOT
        the model's item number: since 2026-08-12 the opener answers with a 1-based index over
        the numbered ITEMS it was shown (ops/OPENER-REDESIGN.md 5.1/5.7), which counts a
        different list from a different base. Callers convert with
        opener.service.OpenerPick.capture_order_index, which is the only sanctioned crossing
        and which refuses (returns None) whenever no sound conversion exists. Do NOT add a
        +1/-1 anywhere else to paper over a mismatch.

        None means NO ITEM WAS SPECIFIED -- the caller could not say which item this opener is
        about. It is deliberately distinct from 0, which is a perfectly legal first captured
        frame: collapsing the two is how an opener the model could not attach to any item got
        treated as an opener about item 1 and sent with full confidence. A driver that targets
        per item must REFUSE to send an `opener` alongside a None index, never treat None as an
        index and never repair the text against whatever it happened to land on.

        NEVER SUBSTITUTE A DIFFERENT ITEM (ops/OPENER-REDESIGN.md 5.6, owner rule). A driver that
        targets per item and cannot put the like on the item it was told to -- even after
        retrying that same item -- raises ItemTargetingError instead of liking a different one.
        Retrying the SAME item is a shaky hand and is encouraged; landing on a neighbour is a
        wrong decision and is forbidden.

        `model_item_index` is the OTHER index space, and it is the one doc 5.6 verifies against:
        the 1-based number of the item in the numbered list the MODEL was shown
        (`opener.ItemRequest`, i.e. `OpenerPick.index` when `index_space` is
        INDEX_SPACE_MODEL_ITEMS). It is deliberately a separate parameter rather than a
        re-interpretation of `item_index`, because the two count different lists from different
        bases and a small int carrying no statement of its own space cannot be validated by the
        side that receives it -- that is exactly the bug the 2026-08-12 audit found.

        `should_stop`, when a driver advertises ``supports_interruptible_like_navigation``, is
        polled before every capture and gesture in its pre-tap item navigation. A stop is a
        distinct navigation cancellation: no further device input is issued and nothing is
        tapped. Workers pass it only to opt-in drivers, preserving existing driver/fake APIs.

        Most drivers return ``None`` after a completed like. A driver that supports the
        Hub-reviewed Training boundary returns the verified human outcome instead: ``"like"``
        after Send Like lands or ``"dislike"`` after the verified pass. The worker relies on
        that closed result to record the manual label only after the corresponding device action
        has actually completed.

        A driver that holds per-item crops for the profile on screen MUST, when given one, verify
        the like/comment screen against that item's stored crop BEFORE typing anything, and stop
        rather than type on a mismatch (never a commentless like, never a comment under an item
        nobody checked, never a rewritten opener to match whatever was hit). None means no such
        verification is possible or asked for; drivers with no per-item notion ignore it.

        THAT POST-TAP CHECK IS NOT A LICENCE TO AIM LOOSELY, and the two parameters do not
        substitute for one another. A driver given a `model_item_index` it has no way to NAVIGATE
        to must refuse before it touches the screen rather than tap something and let the check
        sort it out: the relative check discriminates items within one profile's payload and was
        measured accepting a card from a DIFFERENT profile. Concrete drivers must add their
        calibrated absolute/profile gates; even then, aiming at the topmost heart and verifying
        afterwards is the `hearts[0]` substitution with an extra step."""

    @abstractmethod
    def dislike(self, *, should_stop=None) -> None:
        """Pass on the current profile.

        Drivers advertising ``supports_interruptible_dislike`` must poll
        ``should_stop`` immediately before their first pass input.  If it fires,
        they raise ActionCancelled without touching the device.  Once the input
        has been issued, the pass is a landed action and must return normally so
        the worker records it rather than creating a phantom unrecorded swipe.
        """

    @abstractmethod
    def out_of_profiles(self) -> bool:
        """True if there are no more profiles to swipe right now."""

    def blocked_reason(self) -> str | None:
        """Is the deck unavailable for a reason the operator needs to be told about, and
        what is that reason? Returns an operator-facing sentence describing what's on
        screen, or None when the deck is fine.

        This is a SEPARATE question from out_of_profiles(): out_of_profiles means the
        deck ran dry — a normal end of supply, nothing is wrong. blocked_reason means
        something is ON SCREEN standing BETWEEN us and the deck — the measured case
        being Hinge's "out of free likes for today" Hinge+ paywall: the like
        was refused, the paywall came up, and nothing in the codebase recognized it, so
        the run hung polling for a decision that could never come.

        Defaults to None rather than being abstract because every non-Hinge driver
        (Bumble Android, the Playwright web drivers) is uncalibrated for this and must
        be completely unaffected by its existence. A driver that has never been
        calibrated for a blocking screen honestly does not know whether one is up, and
        None is that honest answer — not a guess dressed up as one.

        MUST NEVER raise and MUST NEVER touch the screen: worker.py calls this on every
        loop iteration of every worker loop. A driver that taps, swipes, or types from inside
        this method would
        violate that on every single iteration, not just the blocked ones.
        """
        return None

    def like_send_attempted(self) -> bool:
        """Did the most recent like() call get as far as ISSUING ITS SEND?

        `like()` is not atomic, and the half of it that matters is irreversible. A driver that
        types an opener into a composer and taps Send has already put that text in front of a
        real person; everything it does afterwards (dismissing an upsell, proving the sheet
        closed, proving the deck advanced) only establishes whether it can CONFIRM that. When
        one of those post-send checks raises, the worker learns "this like did not verify" and
        nothing else -- and it used to record that profile's opener as `decision="never_sent"`,
        which for that window is simply false.

        True therefore means "the send boundary was crossed and the outcome is whatever the
        exception says", NOT "the like landed": a driver that can answer this question must set
        the marker BEFORE its send input, never after, because a raise between the tap and the
        bookkeeping is exactly the case this exists to describe. False means no send input was
        ever issued for the current attempt.

        Scoped to ONE like attempt: an implementation resets it as that attempt begins, so a
        failure can never inherit the previous profile's answer -- the same discipline
        HingeDriver already applies to its landed-opener evidence. It is therefore only
        meaningful to a caller inside a handler around its OWN `like()` call; asking later (say
        at an unrelated refusal one profile further on) reads a stale attempt.

        Defaults to False rather than being abstract, exactly like blocked_reason() above and
        for the same reason: every driver that has never been taught to track this -- and every
        lightweight test double -- keeps today's meaning ("nothing said a send happened")
        instead of crashing or being forced to implement a marker it has no boundary for. A
        swipe-only flow (Bumble) has no composer and no send to attempt, so False is also the
        permanently correct answer there rather than a degraded one.

        MUST NEVER raise and MUST NEVER touch the screen: this is read from inside a worker's
        exception handler, where a second failure would mask the real one.
        """
        return False

    # --- legacy manual-observation compatibility API ---
    def current_profile(self, *, should_stop=None) -> Profile | None:
        """Capture the card currently shown WITHOUT acting (you swipe manually).

        `should_stop` behaves exactly as documented on next_profile above, including the
        rule that a driver only receives it if it declares supports_interruptible_capture.
        `should_stop=None` must reproduce the pre-existing behaviour exactly — tools that
        drive profile perception by hand (tools/hinge_inspect.py) call this with no
        arguments at all."""
        raise NotImplementedError("manual profile observation is not supported by this driver")

    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None,
                          on_like_intent=None) -> bool | None:
        """Block until the user manually likes/passes the current card.

        Returns True (liked), False (passed), or None (card changed, timeout,
        deck empty, or stop requested). Pass timeout=None to wait indefinitely.
        `should_stop` is an optional callable polled in the wait loop; when it
        returns True the driver returns None promptly so a Stop request isn't
        blocked behind your next manual swipe.

        Drivers that opt in with ``supports_observe_like_intent`` may call
        ``on_like_intent(active, anchor)`` — both positional, always — once a human opens
        an intermediate like sheet (``active=True``) and again when that sheet closes,
        whether it is sent or dismissed (``active=False``). ``anchor`` is a screenshot
        (bytes) of the like sheet as it is open on screen — showing the specific item
        the pending comment will attach to — or ``None`` when clearing (``active=False``;
        there is nothing left on screen to show a picture of once the sheet has closed).
        This is only a UI-notification hook; it must not cause device input.
        """
        raise NotImplementedError("manual profile observation is not supported by this driver")

    def render_status(self, status: dict) -> None:
        """Optionally paint a live status overlay in the app's own UI.

        Web drivers may inject a HUD into the page so you see progress
        without watching the terminal. No-op by default — e.g. the Hinge
        Android driver has no injectable page (its status shows in the hub).
        Must never raise: a failed overlay must not break the swipe loop.
        """

    def render_busy(self, message: str | None = None) -> None:
        """Show/hide a blocking 'processing, please wait' overlay in the app UI.

        Called while a training decision is being embedded/stored so you don't
        like/dislike the NEXT card mid-processing (which would mis-attribute it).
        `message` shows the overlay; None hides it. No-op by default (e.g. Hinge).
        Must never raise.
        """

    def close(self) -> None:  # optional cleanup
        pass
