"""The Android driver package: ties each dating app's AndroidAppSpec to the shared
AndroidDriver (defined in operation_love/drivers/hinge.py — see that module's docstring for
why the generic driver lives there rather than here) and registers calibration with the
platform registry.

Importing this package is what makes operation_love.platforms reflect reality: a spec's
`calibrated` and resolved `observe_ready` values are the only inputs that license Auto and
Observe respectively (see platforms._apply_calibration's docstring). The registry imports
this package lazily before answering readiness queries.
"""
from __future__ import annotations

from ... import platforms
from ..android_spec import AndroidAppSpec
from ..hinge import HINGE_SPEC, AndroidDriver, HingeDriver
from .bumble import BUMBLE_SPEC, BumbleAndroidDriver

platforms._apply_calibration({
    "hinge": {
        "observe": HINGE_SPEC.observe_ready,
        # Statically False, and deliberately not derived from numbering readiness (see the
        # "Gate split" in ops/STILL-PHOTO-DISCRIMINATOR.md section 3).  Action coordinates
        # remain calibrated and a verified still-photo bound may well be installed, but a bound
        # licenses NUMBERED SUGGESTIONS in Observe and calibration capture only.  Auto needs a
        # fresh production-OBSERVE release chain on this exact device/build plus a deliberate
        # edit here, so a weak or forged bound artifact can never turn Auto on by itself.
        "auto": False,
    },
    # Both modes derive from the spec.  In particular, the existence of testable card-drag
    # mechanics is not a licence to run a real Bumble session: BUMBLE_SPEC stays fail-closed
    # until its coordinates and required paid-upsell template have been calibrated.
    "bumble": {"observe": BUMBLE_SPEC.observe_ready, "auto": BUMBLE_SPEC.calibrated},
})

__all__ = [
    "AndroidAppSpec",
    "AndroidDriver",
    "HingeDriver",
    "HINGE_SPEC",
    "BumbleAndroidDriver",
    "BUMBLE_SPEC",
]
