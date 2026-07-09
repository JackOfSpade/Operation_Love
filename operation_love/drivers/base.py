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
    # the opener service in auto mode — generating a Claude opener we can't send
    # just wastes credits. Bumble is False (you match first, then message).
    accepts_opener: bool = True

    # Whether human_motion.think_time_s()'s per-decision "think time" (measured
    # like-vs-pass dwell asymmetry) is calibrated for THIS app's real behavior.
    # True only for Hinge, which it was actually measured on; other drivers get
    # the flat, decision-agnostic pacing anchor instead of a borrowed model.
    think_time_calibrated: bool = False

    @abstractmethod
    def open_session(self) -> None:
        """Attach to the app (launch browser / connect to emulator) and reach the swipe deck."""

    @abstractmethod
    def next_profile(self) -> Profile | None:
        """Capture the current profile (all photos + text). None when the deck is empty."""

    @abstractmethod
    def like(self, opener: str | None = None, item_index: int = 0) -> None:
        """Like the current profile, optionally sending an opener message. item_index is the
        0-based index (capture order) of the photo/prompt the opener is about, so drivers that
        comment per-item (Hinge) can target it; drivers without that notion ignore it."""

    @abstractmethod
    def dislike(self) -> None:
        """Pass on the current profile."""

    @abstractmethod
    def out_of_profiles(self) -> bool:
        """True if there are no more profiles to swipe right now."""

    # --- observe mode (shadow learning); only needed when mode="observe" ---
    def current_profile(self) -> Profile | None:
        """Capture the card currently shown WITHOUT acting (you swipe manually)."""
        raise NotImplementedError("observe mode is not supported by this driver")

    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None) -> bool | None:
        """Block until the user manually likes/passes the current card.

        Returns True (liked), False (passed), or None (card changed, timeout,
        deck empty, or stop requested). Pass timeout=None to wait indefinitely.
        `should_stop` is an optional callable polled in the wait loop; when it
        returns True the driver returns None promptly so a Stop request isn't
        blocked behind your next manual swipe.
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
