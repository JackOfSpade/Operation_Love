"""Common interface every app driver implements.

The worker (worker.py) is app-agnostic: it only knows these methods. Concrete drivers
differ in transport: Bumble via Playwright (DOM selectors); Hinge via host-side
ADB only (screencap perception + humanized `input motionevent` gestures at
screen-fraction coordinates — no uiautomator2 / on-device helper, see
ops/HINGE-PIXEL-RUNBOOK.md §5).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..perception.capture import Profile


class DriverClosed(RuntimeError):
    """The user closed the app/browser window during a run."""


def open_debug_log(debug_dir: str):
    """Best-effort DebugLog bootstrap shared by every driver's open_session().

    Returns a DebugLog instance, or None if construction fails. Debug logging must
    never block a run — but failing silently would leave the operator thinking
    diagnostics are on when they're not, so a construction failure is printed.
    """
    try:
        from .debuglog import DebugLog
        return DebugLog(debug_dir)
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

    # Hinge can observe the intermediate "Send Like" sheet. It is opt-in so
    # existing drivers and lightweight test doubles retain their current
    # two-outcome observe API.
    supports_observe_like_intent: bool = False

    # Whether next_profile()/current_profile() honour a `should_stop` callable, i.e.
    # whether a Stop pressed WHILE a profile is being captured is noticed during the
    # capture instead of only after it finishes.
    #
    # This matters because capture is by far the longest uninterruptible stretch in a
    # run. On Hinge, reading one profile is 12 screencaps + 11 humanized read-scrolls,
    # and observe mode then scrolls the whole card back to the top again — measured at
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
    # Restarting is not free even in observe mode, where the bot only reads. The worker
    # re-attaches to whatever is on screen; if the driver was confused about which card it
    # was looking at, the manual swipes it records afterwards are attributed to the wrong
    # profile — which corrupts the taste model permanently, long after the session that
    # caused it. Ending a seeding session early is cheap by comparison.
    #
    # Concrete drivers may lower it from config (apps.<app>.halt_on_error), but config
    # validation refuses `false` for anything running in auto mode.
    halt_on_error: bool = True

    @abstractmethod
    def open_session(self) -> None:
        """Attach to the app (launch browser / connect to emulator) and reach the swipe deck."""

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
    def like(self, opener: str | None = None, item_index: int = 0, *,
             anchored_opener=None) -> None:
        """Like the current profile, optionally sending an opener message. item_index is the
        0-based index (capture order) of the photo/prompt the opener is about, so drivers that
        comment per-item (Hinge) can target it; drivers without that notion ignore it.

        `anchored_opener` is a callback the driver MAY invoke, once its like/comment screen is
        actually open, with a screenshot (bytes) of that screen — which visually shows the
        item the comment is about to attach to — and which returns replacement opener text
        grounded in what that screenshot shows (or None/empty if it can't produce one).

        It exists for the case where the driver could not land the comment on the item
        `item_index` names (per-item targeting is best-effort, not guaranteed) and the sheet
        that just opened is now anchored to a DIFFERENT item than the one `opener` was written
        about — an out-of-place message (an opener about one photo landing under an unrelated
        one) that a driver able to detect the mismatch should repair via this callback rather
        than ship blind. Drivers that cannot land a comment on a specific item at all (e.g.
        Bumble, which likes the whole profile) accept and ignore this parameter."""

    @abstractmethod
    def dislike(self) -> None:
        """Pass on the current profile."""

    @abstractmethod
    def out_of_profiles(self) -> bool:
        """True if there are no more profiles to swipe right now."""

    # --- observe mode (shadow learning); only needed when mode="observe" ---
    def current_profile(self, *, should_stop=None) -> Profile | None:
        """Capture the card currently shown WITHOUT acting (you swipe manually).

        `should_stop` behaves exactly as documented on next_profile above, including the
        rule that a driver only receives it if it declares supports_interruptible_capture.
        `should_stop=None` must reproduce the pre-existing behaviour exactly — tools that
        drive observe-mode perception by hand (tools/hinge_inspect.py) call this with no
        arguments at all."""
        raise NotImplementedError("observe mode is not supported by this driver")

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
        raise NotImplementedError("observe mode is not supported by this driver")

    def render_status(self, status: dict) -> None:
        """Optionally paint a live status overlay in the app's own UI.

        Web drivers (Bumble) inject a HUD into the page so you see progress
        without watching the terminal. No-op by default — e.g. the Hinge
        Android driver has no injectable page (its status shows in the hub).
        Must never raise: a failed overlay must not break the swipe loop.
        """

    def render_busy(self, message: str | None = None) -> None:
        """Show/hide a blocking 'processing, please wait' overlay in the app UI.

        Called in observe mode while a swipe is being embedded/stored so you don't
        like/dislike the NEXT card mid-processing (which would mis-attribute it).
        `message` shows the overlay; None hides it. No-op by default (e.g. Hinge).
        Must never raise.
        """

    def close(self) -> None:  # optional cleanup
        pass
