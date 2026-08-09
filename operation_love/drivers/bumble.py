"""Compatibility shim.

The Bumble web driver's real implementation now lives under
operation_love/drivers/web/ (playwright_base.py = the reusable Playwright base,
bumble_web.py = the Bumble-specific remainder). See that package's docstring for
why this code is kept even though Bumble discontinued its web app in August 2026
and the platform has no live target (operation_love/platforms.py registers
"bumble_web" with available=False).

This module re-exports the old names so nothing that imports
`operation_love.drivers.bumble` needs to change:

  * tools/bumble_inspect.py.
  * tests/test_status.py.

New code should import from operation_love.drivers.web.bumble_web directly.
"""
from __future__ import annotations

import time  # noqa: F401 -- re-exported so `bumble.time` still resolves to the real
             # stdlib module for any caller/test that patches time.sleep through it
             # (patching the shared module object works regardless of which name
             # points at it).

from .web.bumble_web import (
    BumbleActionError,
    BumbleWebDriver as BumbleDriver,
    DEFAULT_SELECTORS,
    _BUSY_JS,
    _CARD_PHOTO_IDS_JS,
    _OVERLAY_JS,
    _PHOTO_LOADED_JS,
    _STARTUP_INTERSTITIALS,
)

__all__ = [
    "BumbleDriver",
    "BumbleActionError",
    "DEFAULT_SELECTORS",
    "_BUSY_JS",
    "_CARD_PHOTO_IDS_JS",
    "_OVERLAY_JS",
    "_PHOTO_LOADED_JS",
    "_STARTUP_INTERSTITIALS",
]
