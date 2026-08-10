"""Bumble's Android binding: AndroidDriver + BUMBLE_SPEC.

⚠️ UNCALIBRATED. Bumble discontinued its web app in August 2026 (see
operation_love/platforms.py), so it is becoming a second Android target on the same
physical Pixel that Hinge already runs on. Nobody has run this against the real Bumble
Android app yet: every coordinate below is a PLACEHOLDER GUESS, not a measurement, and
`calibrated=False` is what keeps operation_love.platforms refusing to start it (see this
package's __init__, which feeds BUMBLE_SPEC.calibrated into platforms._apply_calibration).
Do not flip that flag until coords + templates are actually verified live on the device —
see operation_love/drivers/hinge.py's HINGE_SPEC for what "calibrated" is supposed to mean.

The one exception to "everything below is a placeholder" is the SuperSwipe geometry
recorded next to `forbidden_zones` below: that block is MEASURED, not guessed (read-only
screencap + uiautomator dump on the real Pixel 7a, 2026-08-10) and is called out as such —
see that comment for the full writeup, including why the purchase sheet it describes is
NOT a general safety net (it only exists on one of Bumble's two SuperSwipe outcomes).
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
    # --- MEASURED 2026-08-10 (Pixel 7a, 1080x2400) -----------------------------------------
    # Everything in this block came from a read-only screencap + uiautomator dump against the
    # real device with `com.badoo.mobile.payments.flow.bumble.BumblePaymentFlowActivity` in
    # the foreground — NOT a guess, unlike every other coordinate in this file. No `adb` write
    # command (tap/swipe/text) was ever issued to get it.
    #
    #   Purchase CTA ("Get 30 SuperSwipes for $39.99"): x 0.049-0.950, y 0.899-0.951
    #   Sheet's own X (close) button:                    x 0.855-0.971, y 0.407-0.460
    #   Sheet heading ("Stand out with SuperSwipes"):     y 0.551-0.591
    #   "Close sheet" dimmed overlay (tap anywhere dismisses): x 0.000-1.000, y 0.000-0.394
    #
    # THE TWO-STATE TRUTH (owner-corrected, do not re-simplify this to "there's a
    # confirmation"): the purchase sheet above appears in exactly ONE of Bumble's two
    # SuperSwipe outcomes.
    #
    #   * balance == 0 -- the sheet appears (this is the state the geometry above describes).
    #     No money has been spent yet; it is a hazard, not a safety net, because the CTA sits
    #     almost exactly where the deck's own like/pass controls do (like_heart 0.850/0.900
    #     and pass_x 0.150/0.900 above BOTH land on this CTA, and BOTH sit outside
    #     forbidden_zones below — see UnconfirmedScreenError in hinge.py for why a static
    #     rect can't fix that).
    #   * balance > 0 (5, at measurement time, on the owner's account) -- the SuperSwipe is
    #     spent SILENTLY. No sheet, no prompt, nothing on screen to catch it. This is the
    #     state with NO safety net at all: the only protections are never aiming at the
    #     SuperSwipe control (decide_gesture="card_swipe" + forbidden_zones below) and never
    #     firing a decide gesture without positively confirming the deck first
    #     (AndroidDriver._require_deck_confirmed).
    #
    # "The confirmation sheet always appears" is therefore true ONLY for the zero-balance /
    # purchase path -- do not read it as a general guarantee anywhere else in this codebase.
    #
    # upsell_dismiss_zone below is derived from the measured "Close sheet" overlay
    # (x 0-1, y 0-0.394), narrowed with margin, NOT copied from it directly -- see its own
    # comment for the arithmetic.
    forbidden_zones=(
        # PLACEHOLDER. Bumble's PAID SuperSwipe control sits in the middle of the bottom
        # action row, physically BETWEEN Pass and Like — reviewers report accidental taps
        # (and accidental charges) precisely because it occupies the space a thumb crosses.
        # Whether a mis-tap there is silently spent or opens the purchase sheet above depends
        # on the account's SuperSwipe balance (see the MEASURED block above) — the rect below
        # is the backstop either way: any tap resolving inside it raises instead of firing.
        # Generously oversized on purpose — a refused like costs one profile, a mis-tap costs
        # money and violates the owner's never-super-like rule. Tighten only after measuring
        # the real button on the device.
        #
        # NOTE on why this rect is not "the fix" for the purchase-sheet hazard above: it is
        # SCREEN-AGNOSTIC (it forbids a coordinate no matter what is on screen), but the CTA
        # danger is SCREEN-DEPENDENT (the identical point is a harmless like on the deck and a
        # $39.99 purchase on the sheet). Widening this rect to also cover the CTA
        # (x 0.049-0.950, y 0.899-0.951) would forbid like_heart/pass_x on the ordinary deck
        # too — see UnconfirmedScreenError in hinge.py for the guard that actually addresses
        # this (confirm the deck is really on screen before any decide gesture, rather than
        # trying to make one rectangle describe two different screens).
        (0.34, 0.80, 0.66, 1.00),
    ),
    # MEASURED 2026-08-10, derived (not copied) from the "Close sheet" overlay above
    # (x 0.000-1.000, y 0.000-0.394). Narrowed to x 0.15-0.85, y 0.10-0.27 rather than using
    # the full measured overlay, for three reasons:
    #   * clearance below the band before the overlay ends: 0.394 - 0.27 = 0.124 (~298px on a
    #     2400px-tall screen) -- comfortably more than the ~9.6px (~0.004 fraction)
    #     tap_jitter_margin_px() envelope _tap() already accounts for, so a jittered tap
    #     inside this band cannot drift into the sheet's own X button at y 0.407-0.460.
    #   * clearance at the top avoids the status bar / notification shade.
    #   * clearance on both sides (0.15/0.85, not 0.0/1.0) avoids Android's system
    #     back-gesture strips, which live within ~24dp of each screen edge.
    #   * the band STOPS at y 0.27, short of the y 0.28-0.36 occupied by the "SuperSwipe
    #     N left" card on the profile screen that was sitting BEHIND the sheet when this was
    #     measured. While the sheet is up that card is unreachable (the Close-sheet overlay
    #     swallows the tap), so this costs nothing -- but it means that even if detection is
    #     ever wrong and we tap with no sheet present, the tap cannot land on the one control
    #     that OPENS the purchase flow. Sampling the band 20,000 times put ~25% of points in
    #     that y-range before this narrowing. Defence in depth behind the detection gate,
    #     not a substitute for it.
    # See AndroidAppSpec.upsell_dismiss_zone and AndroidDriver._dismiss_via_zone (hinge.py)
    # for how this is used: a FRESH random point inside this rect every dismiss attempt,
    # never a fixed coordinate, only ever after the "upsell_dismiss" template positively
    # confirms the sheet is up. templates={} below means this is currently INERT (no template
    # captured yet -- see ops/RUNBOOK.md's Bumble calibration checklist); declaring the zone
    # now means the geometry is on record and reviewed before that template exists, not typed
    # in a hurry once it does.
    upsell_dismiss_zone=(0.15, 0.10, 0.85, 0.27),
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
