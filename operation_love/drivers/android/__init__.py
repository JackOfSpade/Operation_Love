"""The Android driver package: ties each dating app's AndroidAppSpec to the shared
AndroidDriver (defined in operation_love/drivers/hinge.py — see that module's docstring for
why the generic driver lives there rather than here) and registers calibration with the
platform registry.

Importing this package is what makes operation_love.platforms reflect driver readiness: a spec's
`calibrated` values license the device mechanics for Training and Auto (see
platforms._apply_calibration's docstring). Hinge AUTO's separate,
config-bound production-OBSERVE release artifact is validated by supervisor before a run starts.
The registry imports this package lazily before answering readiness queries.
"""
from __future__ import annotations

from ... import platforms
from ..android_spec import AndroidAppSpec
from ..hinge import HINGE_SPEC, AndroidDriver, HingeDriver
from .bumble import BUMBLE_SPEC, BumbleAndroidDriver

platforms._apply_calibration({
    "hinge": {
        "training": HINGE_SPEC.calibrated,
        # HINGE_SPEC proves the mechanical input geometry. A still-photo licence is checked by
        # the registry; the exact manual/AI release artifact is checked by config validation
        # before a Worker or driver exists, so this cannot bypass the production release gate.
        "auto": HINGE_SPEC.calibrated,
    },
    # Both modes derive from the spec.  In particular, the existence of testable card-drag
    # mechanics is not a licence to run a real Bumble session: BUMBLE_SPEC stays fail-closed
    # until its coordinates and required paid-upsell template have been calibrated.
    "bumble": {"training": BUMBLE_SPEC.calibrated, "auto": BUMBLE_SPEC.calibrated},
})

__all__ = [
    "AndroidAppSpec",
    "AndroidDriver",
    "HingeDriver",
    "HINGE_SPEC",
    "BumbleAndroidDriver",
    "BUMBLE_SPEC",
]
