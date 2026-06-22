"""Common interface every app driver implements.

The orchestrator is app-agnostic: it only knows these methods. Concrete drivers
(Bumble via Playwright, Hinge via Appium/uiautomator2) find UI elements by
id/text — never fixed pixel coordinates — so they're resolution-independent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..perception.capture import Profile


class DriverClosed(RuntimeError):
    """The user closed the app/browser window during a run."""


class DatingAppDriver(ABC):
    # Whether this app lets you attach a written opener AT SWIPE TIME (Hinge's
    # "send a like with a comment" model). When False, the worker must NOT call
    # the opener service in auto mode — generating a Claude opener we can't send
    # just wastes credits. Bumble is False (you match first, then message).
    accepts_opener: bool = True

    @abstractmethod
    def open_session(self) -> None:
        """Attach to the app (launch browser / connect to emulator) and reach the swipe deck."""

    @abstractmethod
    def next_profile(self) -> Profile | None:
        """Capture the current profile (all photos + text). None when the deck is empty."""

    @abstractmethod
    def like(self, opener: str | None = None) -> None:
        """Like the current profile, optionally sending an opener message."""

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
