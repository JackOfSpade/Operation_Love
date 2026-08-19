"""The Android driver package: ties each dating app's AndroidAppSpec to the shared
AndroidDriver (defined in operation_love/drivers/hinge.py — see that module's docstring for
why the generic driver lives there rather than here) and registers calibration with the
platform registry.

Importing this package is what makes operation_love.platforms reflect reality: a spec's
`calibrated` flag is the ONLY thing that flips a platform from refused to runnable (see
platforms._apply_calibration's docstring), so anything that asks the registry whether
Hinge or Bumble-on-Android is runnable must import this package first.
"""
from __future__ import annotations

from ... import platforms
from ..android_spec import AndroidAppSpec
from ..hinge import HINGE_SPEC, AndroidDriver, HingeDriver
from .bumble import BUMBLE_SPEC, BumbleAndroidDriver

platforms._apply_calibration({
    "hinge": {"observe": HINGE_SPEC.observe_ready, "auto": HINGE_SPEC.calibrated},
    # Bumble's direct card-drag Auto path is supported. Manual observation is deliberately
    # absent, so it must never inherit Auto readiness just because both share one spec.
    "bumble": {"observe": False, "auto": True},
})

__all__ = [
    "AndroidAppSpec",
    "AndroidDriver",
    "HingeDriver",
    "HINGE_SPEC",
    "BumbleAndroidDriver",
    "BUMBLE_SPEC",
]
