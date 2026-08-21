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
from ...targeting_policy import hinge_targeting_unavailable_reason
from ..android_spec import AndroidAppSpec
from ..hinge import HINGE_SPEC, AndroidDriver, HingeDriver
from .bumble import BUMBLE_SPEC, BumbleAndroidDriver

platforms._apply_calibration({
    "hinge": {
        "observe": HINGE_SPEC.observe_ready,
        # Action coordinates remain calibrated, but Auto also needs a positive still-photo
        # discriminator before numbered targeting can be release-licensed.
        "auto": HINGE_SPEC.calibrated and hinge_targeting_unavailable_reason() is None,
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
