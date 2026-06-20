"""Common interface every app driver implements.

The orchestrator is app-agnostic: it only knows these methods. Concrete drivers
(Bumble via Playwright, Hinge via Appium/uiautomator2) find UI elements by
id/text — never fixed pixel coordinates — so they're resolution-independent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..perception.capture import Profile


class DatingAppDriver(ABC):
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

    def wait_for_decision(self, timeout: float = 120.0, should_stop=None) -> bool | None:
        """Block until the user manually likes/passes the current card.

        Returns True (liked), False (passed), or None (no decision before the
        card changed / timeout). `should_stop` is an optional callable polled in
        the wait loop; when it returns True the driver returns None promptly so
        a Stop request isn't blocked behind your next manual swipe.
        """
        raise NotImplementedError("observe mode is not supported by this driver")

    def render_status(self, status: dict) -> None:
        """Optionally paint a live status overlay in the app's own UI.

        Web drivers (Bumble) inject a HUD into the page so you see progress
        without watching the terminal. No-op by default — e.g. the Hinge
        Android driver has no injectable page (its status shows in the hub).
        Must never raise: a failed overlay must not break the swipe loop.
        """

    def close(self) -> None:  # optional cleanup
        pass
