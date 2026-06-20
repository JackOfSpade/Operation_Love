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

    def close(self) -> None:  # optional cleanup
        pass
