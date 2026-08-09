"""Reusable web (Playwright/patchright) driving layer for browser-based dating apps.

Bumble discontinued its web app in August 2026
(https://support.bumble.com/hc/en-us/articles/30996192802973-An-update-on-Bumble-web),
so operation_love/drivers/web/bumble_web.py has no live target right now. See
operation_love/platforms.py, which registers the "bumble_web" platform with
available=False and that fact as the reason.

This package is kept -- and was split out of the old monolithic
operation_love/drivers/bumble.py -- on purpose: the anti-detection hardening in it
(persistent-context launch hygiene that hides navigator.webdriver, human-like cursor
movement, the injected-JS status HUD, the card-identity anti-phantom-swipe check) has
nothing to do with Bumble specifically. It is the starting point for driving ANY
web-based dating platform we pick up later, not dead Bumble-shaped code.

  playwright_base.py -- PlaywrightDriver: the reusable, site-agnostic base.
  bumble_web.py      -- BumbleWebDriver: what's actually Bumble-shaped (selectors,
                         photo-album capture, the observe-mode click listener).

Nothing here is runnable today -- PlaywrightDriver.open_session() refuses to start
for any platform the registry marks unavailable, "bumble_web" included -- but it is
still exercised by tests/test_bumble_observe.py and tests/test_web_base.py against a
fake Playwright object graph, so it doesn't silently bit-rot while it waits for a
future web target.
"""
from __future__ import annotations

from .bumble_web import BumbleWebDriver
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
    "BumbleWebDriver",
]
