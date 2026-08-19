"""Driver factory — one driver per registered platform.

The mapping is keyed by the ids in operation_love/platforms.py, so "which platforms exist"
is stated once (the registry) rather than duplicated here and drifting. Imports stay lazy
inside each branch: Android drivers pull in cv2/PIL/numpy only when used.
"""
from __future__ import annotations

from .. import platforms
from .base import DatingAppDriver


def make_driver(app: str, cfg) -> DatingAppDriver:
    """Build the driver for `app`. Validates the id; does NOT check availability.

    Availability is enforced at open_session() (plus supervisor.run() and HubState.start()),
    deliberately not here. Constructing a driver is inert — an AndroidDriver has no adb or
    touch transport until open_session() supplies one — whereas refusing construction would
    also block the calibration tooling that is the only way an uncalibrated platform ever
    becomes calibrated. Guard the thing that acts, not the thing that exists.
    """
    platforms.get(app)          # raises ValueError, with the known-ids list, on an unknown id

    if app == "hinge":
        from .hinge import HingeDriver
        return HingeDriver(cfg)
    if app == "bumble":
        from .android.bumble import BumbleAndroidDriver
        return BumbleAndroidDriver(cfg)
    # Registered in platforms.py but with no branch here — a half-added platform.
    raise ValueError(
        f"No driver is wired for registered platform '{app}'. Add a branch in "
        f"{__name__}.make_driver()."
    )
