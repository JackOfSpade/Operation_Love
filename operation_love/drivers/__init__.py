"""Driver factory — one driver per app. Element-based, OS-agnostic."""
from __future__ import annotations

from .base import DatingAppDriver


def make_driver(app: str, cfg) -> DatingAppDriver:
    if app == "bumble":
        from .bumble import BumbleDriver
        return BumbleDriver(cfg)
    if app == "hinge":
        from .hinge import HingeDriver
        return HingeDriver(cfg)
    raise ValueError(f"Unknown app '{app}'. Supported: bumble, hinge.")
