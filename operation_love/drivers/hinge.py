"""Hinge driver — Android emulator / device over ADB (uiautomator2).

Phase 5. Hinge is mobile-only, so this drives a virtual Android phone (an
emulator — no physical Android device required) via ADB. Element-based
(resource-id / text), OS-agnostic (ADB runs on macOS/Windows/Linux). The
emulator host is launched separately; this connects to it.
"""
from __future__ import annotations

from .base import DatingAppDriver
from ..perception.capture import Profile


class HingeDriver(DatingAppDriver):
    def __init__(self, cfg):
        self.cfg = cfg

    def open_session(self) -> None:  # pragma: no cover - Phase 5
        raise NotImplementedError("Hinge driver lands in Phase 5 (Android emulator + uiautomator2)")

    def next_profile(self) -> Profile | None:  # pragma: no cover - Phase 5
        raise NotImplementedError

    def like(self, opener: str | None = None) -> None:  # pragma: no cover - Phase 5
        raise NotImplementedError

    def dislike(self) -> None:  # pragma: no cover - Phase 5
        raise NotImplementedError

    def out_of_profiles(self) -> bool:  # pragma: no cover - Phase 5
        raise NotImplementedError
