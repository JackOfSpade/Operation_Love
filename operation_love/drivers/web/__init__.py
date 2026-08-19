"""Reusable Playwright/patchright base for future browser-based dating apps."""
from __future__ import annotations

from .playwright_base import (
    ActionNotLandedError,
    HumanInputUnavailable,
    PlatformUnavailable,
    PlaywrightDriver,
)

__all__ = [
    "PlaywrightDriver",
    "PlatformUnavailable",
    "ActionNotLandedError",
    "HumanInputUnavailable",
]
