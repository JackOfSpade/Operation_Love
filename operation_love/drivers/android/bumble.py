"""Bumble's Android binding: AndroidDriver + BUMBLE_SPEC.

⚠️ UNCALIBRATED. Bumble discontinued its web app in August 2026 (see
operation_love/platforms.py), so it is becoming a second Android target on the same
physical Pixel that Hinge already runs on. Nobody has run this against the real Bumble
Android app yet: every coordinate below is a PLACEHOLDER GUESS, not a measurement, and
`calibrated=False` is what keeps operation_love.platforms refusing to start it (see this
package's __init__, which feeds BUMBLE_SPEC.calibrated into platforms._apply_calibration).
Do not flip that flag until coords + templates are actually verified live on the device —
see operation_love/drivers/hinge.py's HINGE_SPEC for what "calibrated" is supposed to mean.
"""
from __future__ import annotations

from ..android_spec import AndroidAppSpec
from ..hinge import AndroidDriver

BUMBLE_SPEC = AndroidAppSpec(
    app="bumble",
    package="com.bumble.app",
    calibrated=False,   # <-- the only thing stopping this from firing real touches; see module docstring
    coords={
        # PLACEHOLDER GUESSES ONLY — nobody has looked at the real screen. Recalibrate live
        # before ever flipping `calibrated` above.
        #
        # The drag endpoints are what actually get used (see decide_gesture below): start
        # mid-card, well ABOVE the bottom action row, and travel horizontally off the edge.
        "swipe_start": (0.50, 0.55),
        "swipe_like_end": (0.92, 0.52),   # rightward drag = like
        "swipe_pass_end": (0.08, 0.52),   # leftward drag  = pass
        # Button positions, kept only as a fallback if this app is ever switched to
        # decide_gesture="tap". Bumble's action row runs [ X ] [ SuperSwipe ] [ like ], so
        # these MUST stay outside forbidden_zones below.
        "like_heart": (0.85, 0.90),
        "pass_x": (0.15, 0.90),
    },
    forbidden_zones=(
        # PLACEHOLDER. Bumble's PAID SuperSwipe control sits in the middle of the bottom
        # action row, physically BETWEEN Pass and Like — reviewers report accidental taps
        # (and accidental charges) precisely because it occupies the space a thumb crosses.
        # Unlike Hinge's Rose there is no confirmation modal afterwards to catch a mistake,
        # so this rect is the backstop: any tap resolving inside it raises instead of firing.
        # Generously oversized on purpose — a refused like costs one profile, a mis-tap costs
        # money and violates the owner's never-super-like rule. Tighten only after measuring
        # the real button on the device.
        (0.34, 0.80, 0.66, 1.00),
    ),
    like_flow="direct",           # one like, no swipe-time comment sheet (see decide_gesture)
    decide_gesture="card_swipe",  # drag the card, do NOT aim at the button row — a drag cannot
                                   # press the paid SuperSwipe sitting between Pass and Like,
                                   # so the rule holds by construction, not by careful aiming
    accepts_opener=False,         # Bumble: match first, then message — no swipe-time opener
                                   # (don't spend Gemini quota/spend on text that can never be
                                   # sent at like-time; same reason BumbleDriver.accepts_opener
                                   # is already False on the web driver)
    think_time_calibrated=False,  # human_motion's dwell asymmetry was only ever measured on
                                   # Hinge; Bumble gets the flat, decision-agnostic pacing
    change_threshold=9.0,         # inherited Hinge default; unverified for Bumble's UI
    scroll_captures=8,            # inherited Hinge default; unverified for Bumble's UI
    dwell_s=1.1,                  # inherited Hinge default; unverified for Bumble's UI
    read_scroll_frac=0.55,        # inherited Hinge default; unverified for Bumble's UI
)


class BumbleAndroidDriver(AndroidDriver):
    """Thin binding: AndroidDriver + BUMBLE_SPEC. Not the same class as
    operation_love.drivers.bumble.BumbleDriver (the Playwright/web driver) — that one drives
    Bumble's now-dead web app; this one drives the Bumble Android app over the same
    host-side-ADB transport Hinge uses. Refused at the platform-registry level
    (operation_love.platforms) until BUMBLE_SPEC.calibrated is True."""

    def __init__(self, cfg):
        super().__init__(cfg, BUMBLE_SPEC)
