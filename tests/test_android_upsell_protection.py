"""Bumble SuperSwipe, corrected: the purchase sheet is not a safety net, so the real
protections have to live upstream of it.

Ground truth measured live on the real Pixel 7a (1080x2400) 2026-08-10 (read-only screencap
+ uiautomator dump, no `adb` write command issued): Bumble's SuperSwipe has two outcomes.
With a non-zero balance it is spent SILENTLY -- no modal, nothing to dismiss, nothing
downstream to catch a mistake. Only with a zero balance does a purchase/confirmation sheet
appear first, and even then its "Get 30 SuperSwipes for $39.99" CTA (x 0.049-0.950,
y 0.899-0.951) sits almost exactly where the deck's own like/pass fallback coordinates do
(like_heart 0.850/0.900, pass_x 0.150/0.900 -- both OUTSIDE BUMBLE_SPEC.forbidden_zones,
(0.34, 0.80, 0.66, 1.00)).

That is why forbidden_zones alone can't be "the fix": it is a SCREEN-AGNOSTIC rectangle, and
the danger above is SCREEN-DEPENDENT (the identical point is a harmless like on the deck and
a $39.99 purchase on the sheet). This module pins the two mechanisms that actually address
that:

  1. AndroidDriver._require_deck_confirmed() / UnconfirmedScreenError (hinge.py) -- refuse
     ANY decide gesture (tap or card_swipe) unless the deck's own like+pass glyphs are
     positively confirmed on screen first. Covers the balance>0 silent-spend case, which
     has no modal of any kind for anything else to react to.
  2. AndroidAppSpec.upsell_dismiss_zone / AndroidDriver._dismiss_via_zone / PaidUpsellStuckError
     (android_spec.py, hinge.py) -- for a paid-upgrade sheet whose real dismiss control is a
     large "tap outside the sheet" overlay rather than a discrete button, detect it first
     (the existing 'upsell_dismiss' template), then dismiss with a FRESH random point inside a
     declared safe band every attempt, verify it actually cleared, and halt rather than retry
     forever. Covers the balance==0 purchase-sheet case.

Bumble itself stays uncalibrated and templateless (real, current state -- see BUMBLE_SPEC),
so these are exercised here against purpose-built specs, the same pattern
tests/test_android_spec.py already uses for '_UPSELL_SPEC' / '_DIRECT_WITH_UPSELL_SPEC'.
"""
from __future__ import annotations

import pytest

from operation_love.drivers import hinge
from operation_love.drivers.android.bumble import BUMBLE_SPEC
from operation_love.drivers.android_spec import AndroidAppSpec
from operation_love.drivers.hinge import (
    AndroidDriver,
    PaidUpsellStuckError,
    UnconfirmedScreenError,
)
from operation_love.human_motion import tap_jitter_margin_px


class FakeAdb:
    """Records what actually reached the phone. Same shape every other driver test file in
    this suite uses -- see tests/test_android_safety.py's docstring for why each file keeps
    its own copy rather than sharing one."""

    def __init__(self, frames):
        self.frames = list(frames) or [b""]
        self.i = 0
        self.taps = []
        self.swipes = []
        self.scrolls = 0
        self.texts = []

    def screen_size(self):
        return (1080, 2400)

    def devices(self):
        return ["dev"]

    def shell(self, command="", **_):
        return ""

    def screencap(self):
        return self.frames[min(self.i, len(self.frames) - 1)]

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, **_k):
        self.swipes.append((x1, y1, x2, y2))

    def scroll_up(self, *_a, **_k):
        self.scrolls += 1

    def text(self, s):
        self.texts.append(s)


class _ClearsAfterFirstTapAdb(FakeAdb):
    """A modal that is up until the first tap, then gone -- models a single successful
    dismiss so a test can inspect exactly where that one tap landed."""

    def __init__(self, up_frame, down_frame):
        super().__init__([up_frame])
        self.up_frame = up_frame
        self.down_frame = down_frame

    def screencap(self):
        return self.down_frame if self.taps else self.up_frame


def _drv(spec, adb, **overrides):
    # halt_on_error False: these tests are about the NEW guards, which are unconditional
    # (like ForbiddenTapError), not about _verify_progress's separate halt_on_error-gated path.
    class C:
        apps = {spec.app: {"halt_on_error": False, **overrides}}
    d = AndroidDriver(C(), spec)
    d._adb = adb
    d._touch = adb
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


# --- frame builders: real, template-detectable glyphs on synthetic screens ------------------

def _deck_ready_frame(heart_xy=(930, 1600), pass_xy=(130, 2030), seed=41):
    """A decodable frame carrying BOTH the like-heart and pass-X glyphs -- a positively
    confirmed swipe deck, the thing _require_deck_confirmed exists to prove is on screen."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in (("hinge_heart.png", heart_xy), ("hinge_pass_x.png", pass_xy)):
        t = hinge._load_template(name)
        th, tw = t.shape
        canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _no_deck_frame(seed=42):
    """A decodable frame with NEITHER deck glyph -- stands in for a purchase sheet, an ad,
    a dialog, or literally anything else that isn't the ordinary swipe deck."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _modal_frame(glyph_xy=(540, 1350), seed=43):
    """A decodable frame carrying the shared 'upsell sheet' detection glyph, reused generically
    (not through HINGE_SPEC) -- for a zone-dismiss app this glyph identifies the SHEET (e.g.
    its heading), not a tap target; see AndroidAppSpec.upsell_dismiss_zone."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    t = hinge._load_template("hinge_send_like_anyway.png")
    th, tw = t.shape
    cx, cy = glyph_xy
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _no_modal_frame(seed=44):
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


# =============================================================================================
# Mechanism 1: _require_deck_confirmed() / UnconfirmedScreenError -- refuse ANY decide gesture
# unless the deck is positively confirmed first. Covers Bumble's balance>0 silent-spend case,
# which has no modal for anything else to react to.
# =============================================================================================

_DECK_CONFIRM_SWIPE_SPEC = AndroidAppSpec(
    app="deckconfirmswipe",
    package="com.example.deckswipe",
    calibrated=False,
    decide_gesture="card_swipe",
    coords={"swipe_start": (0.50, 0.55), "swipe_like_end": (0.92, 0.52), "swipe_pass_end": (0.08, 0.52)},
    templates={"like": "hinge_heart.png", "pass": "hinge_pass_x.png"},
    forbidden_zones=BUMBLE_SPEC.forbidden_zones,   # the real Bumble no-go rect, unchanged
    like_flow="direct",
)


def test_card_swipe_decide_refused_and_screen_untouched_when_deck_not_confirmed():
    # This is the exact gap the ground truth exposed: Bumble decides by CARD DRAG, and a drag
    # start has never been vision-checked against anything before this guard existed -- only
    # zone-checked against a screen-agnostic rectangle. Prove it's now refused instead.
    adb = FakeAdb([_no_deck_frame()])
    drv = _drv(_DECK_CONFIRM_SWIPE_SPEC, adb)
    with pytest.raises(UnconfirmedScreenError):
        drv.dislike()
    assert adb.taps == [] and adb.swipes == [], "nothing may reach the phone while unconfirmed"


def test_card_swipe_decide_refused_for_like_too():
    adb = FakeAdb([_no_deck_frame()])
    drv = _drv(_DECK_CONFIRM_SWIPE_SPEC, adb)
    with pytest.raises(UnconfirmedScreenError):
        drv.like()
    assert adb.taps == [] and adb.swipes == []


def test_card_swipe_decide_proceeds_normally_once_deck_is_confirmed():
    # The other half of the pin: this must not become a permanent refusal machine. A normal,
    # confirmed deck must decide exactly as it did before this guard was added.
    adb = FakeAdb([_deck_ready_frame()])
    drv = _drv(_DECK_CONFIRM_SWIPE_SPEC, adb)
    drv.dislike()
    assert adb.taps == [], "card_swipe never taps"
    assert len(adb.swipes) == 1
    x1, _y1, x2, _y2 = adb.swipes[0]
    assert x2 < x1, "pass drags leftward"


def test_unconfirmed_screen_error_halts_the_run_rather_than_being_routine():
    assert issubclass(UnconfirmedScreenError, hinge.HingeActionError)


def test_deck_confirm_guard_is_a_noop_for_a_spec_declaring_no_like_or_pass_template():
    # BUMBLE_SPEC's real, current state: no templates at all. This is deliberately NOT a
    # loophole -- see UnconfirmedScreenError's docstring -- but it must not regress the
    # already-pinned behaviour in tests/test_android_safety.py (a card_swipe like/pass
    # issues exactly one drag and zero taps).
    adb = FakeAdb([b""])
    drv = _drv(BUMBLE_SPEC, adb)
    drv.like()
    assert adb.taps == []
    assert len(adb.swipes) == 1


# --- the measured CTA coordinates specifically: proven to clear forbidden_zones, but the
# decide flow that would aim near them is still refused whenever the deck isn't confirmed ---

_MEASURED_TAP_SPEC = AndroidAppSpec(
    app="measuredtap",
    package="com.example.measuredtap",
    calibrated=False,
    decide_gesture="tap",
    coords={
        # BUMBLE_SPEC's actual fallback coordinates, reused verbatim: MEASURED 2026-08-10 to
        # land ON Bumble's "Get 30 SuperSwipes for $39.99" purchase CTA (x 0.049-0.950,
        # y 0.899-0.951) whenever the purchase sheet is up, while sitting OUTSIDE
        # forbidden_zones (0.34, 0.80, 0.66, 1.00) either way.
        "like_heart": (0.850, 0.900),
        "pass_x": (0.150, 0.900),
    },
    templates={"like": "hinge_heart.png", "pass": "hinge_pass_x.png"},
    forbidden_zones=BUMBLE_SPEC.forbidden_zones,
    like_flow="direct",
)


def test_measured_cta_fallback_coords_clear_forbidden_zones_on_their_own():
    # Self-consistency pin, same style as test_android_safety.py's
    # test_bumble_fallback_button_coords_stay_outside_the_paid_zone: forbidden_zones alone
    # does NOT catch these points. That is exactly why it cannot be "the fix" -- widening it
    # to also cover (0.850, 0.900)/(0.150, 0.900) would forbid an ordinary like/pass on the
    # swipe deck too, where the identical points are safe.
    drv = _drv(_MEASURED_TAP_SPEC, FakeAdb([b""]))
    w, h = drv.adb.screen_size()
    for name in ("like_heart", "pass_x"):
        fx, fy = _MEASURED_TAP_SPEC.coords[name]
        drv._assert_tap_allowed(int(fx * w), int(fy * h))   # must not raise


def test_decide_refused_while_upsell_state_active_even_though_measured_coords_clear_the_zone():
    # The upsell state: neither deck glyph is visible (the purchase sheet, or anything else,
    # is covering the deck). _require_deck_confirmed must refuse BEFORE _await_button ever
    # gets a chance to vision-locate and tap near the measured, zone-clearing coordinates.
    adb = FakeAdb([_no_deck_frame()])
    drv = _drv(_MEASURED_TAP_SPEC, adb)
    with pytest.raises(UnconfirmedScreenError):
        drv.dislike()
    assert adb.taps == []


def test_decide_still_works_normally_on_a_confirmed_deck_with_the_same_spec():
    heart_pt, pass_pt = (930, 1600), (130, 2030)
    adb = FakeAdb([_deck_ready_frame(heart_pt, pass_pt)])
    drv = _drv(_MEASURED_TAP_SPEC, adb)
    drv.dislike()
    assert adb.taps == [pass_pt]   # the vision-located pass-X, not the fallback coordinate


# --- has_paid_upsell=True with no upsell_dismiss template still blocks calibration ----------
# Already enforced by AndroidAppSpec.__post_init__ (pinned generically in
# tests/test_android_safety.py and against BUMBLE_SPEC specifically); restated here, scoped to
# this module's subject, so this file is self-contained proof the guarantee still holds
# alongside the new upsell_dismiss_zone field.

def test_has_paid_upsell_without_upsell_dismiss_template_still_blocks_calibration():
    with pytest.raises(ValueError, match="upsell_dismiss"):
        AndroidAppSpec(app="pinned", package="x.y", calibrated=True, has_paid_upsell=True,
                       templates={"like": "hinge_heart.png", "pass": "hinge_pass_x.png"})


def test_bumble_spec_still_cannot_be_marked_calibrated_without_the_template():
    assert "upsell_dismiss" not in BUMBLE_SPEC.templates
    with pytest.raises(ValueError, match="upsell_dismiss"):
        AndroidAppSpec(**{**BUMBLE_SPEC.__dict__, "calibrated": True})


# =============================================================================================
# Mechanism 2: AndroidAppSpec.upsell_dismiss_zone / AndroidDriver._dismiss_via_zone /
# PaidUpsellStuckError -- dismiss an ALREADY-DETECTED paid-upgrade sheet whose real control is
# a large "tap outside the sheet" overlay (Bumble's SuperSwipe purchase sheet), by tapping a
# fresh random point inside a declared safe band, verifying it landed, and halting instead of
# retrying forever. Covers Bumble's balance==0 purchase-sheet case.
# =============================================================================================

_ZONE_UPSELL_SPEC = AndroidAppSpec(
    app="zoneupsell",
    package="com.example.zoneupsell",
    calibrated=False,
    templates={"upsell_dismiss": "hinge_send_like_anyway.png"},
    # MEASURED-derived band, identical to BUMBLE_SPEC's -- see that spec's comment for the
    # margin arithmetic (jitter envelope, status bar, edge back-gesture strips).
    upsell_dismiss_zone=(0.15, 0.10, 0.85, 0.34),
    like_flow="direct",
)

# Bumble's measured purchase CTA, for the "never produces a CTA point" regression below.
_CTA_RECT = (0.049, 0.899, 0.950, 0.951)


def test_handle_upsell_never_taps_when_detection_fails():
    adb = FakeAdb([_no_modal_frame()])
    drv = _drv(_ZONE_UPSELL_SPEC, adb)
    assert drv._handle_rose_upsell(tries=1) is False
    assert adb.taps == []


def test_handle_upsell_dismisses_via_zone_only_after_positive_detection():
    adb = _ClearsAfterFirstTapAdb(_modal_frame(), _no_modal_frame())
    drv = _drv(_ZONE_UPSELL_SPEC, adb)
    assert drv._handle_rose_upsell() is True
    assert len(adb.taps) == 1
    x0, y0, x1, y1 = _ZONE_UPSELL_SPEC.upsell_dismiss_zone
    w, h = adb.screen_size()
    tx, ty = adb.taps[0]
    assert x0 * w <= tx <= x1 * w
    assert y0 * h <= ty <= y1 * h


def test_dismiss_via_zone_point_stays_inside_the_measured_safe_band_across_many_draws():
    # "Randomize per attempt" pinned as an actual distribution property, not just one sample:
    # every draw across many independent dismiss attempts must land inside the declared band,
    # with the tap-jitter envelope accounted for on the boundary closest to danger (0.34, just
    # below the sheet's own X button at 0.407-0.460 and comfortably clear of the 0.394 overlay
    # edge -- see BUMBLE_SPEC's margin arithmetic).
    x0, y0, x1, y1 = _ZONE_UPSELL_SPEC.upsell_dismiss_zone
    w, h = 1080, 2400
    margin_frac_y = tap_jitter_margin_px() / h
    points = []
    for _ in range(300):
        adb = _ClearsAfterFirstTapAdb(_modal_frame(), _no_modal_frame())
        drv = _drv(_ZONE_UPSELL_SPEC, adb)
        drv._dismiss_via_zone()
        assert len(adb.taps) == 1
        points.append(adb.taps[0])

    fracs = [(tx / w, ty / h) for tx, ty in points]
    assert all(x0 <= fx <= x1 for fx, _fy in fracs)
    assert all(y0 <= fy <= y1 for _fx, fy in fracs)
    # Even with the jitter envelope added on top, every draw stays clear of the measured
    # overlay boundary (0.394) and nowhere near the purchase CTA.
    assert all(fy + margin_frac_y < 0.394 for _fx, fy in fracs)
    cta_x0, cta_y0, cta_x1, cta_y1 = _CTA_RECT
    assert not any(cta_x0 <= fx <= cta_x1 and cta_y0 <= fy <= cta_y1 for fx, fy in fracs)
    # Not a fixed point: a real spread of draws, not the same coordinate every time.
    assert len({round(fx, 4) for fx, _fy in fracs}) > 5
    assert len({round(fy, 4) for _fx, fy in fracs}) > 5


def test_dismiss_via_zone_halts_after_bounded_attempts_instead_of_tapping_forever():
    # The modal NEVER clears (screencap always returns the same glyph-bearing frame,
    # regardless of taps) -- the guard must give up after a bounded number of attempts rather
    # than tapping the same still-present modal indefinitely.
    adb = FakeAdb([_modal_frame()])
    drv = _drv(_ZONE_UPSELL_SPEC, adb)
    with pytest.raises(PaidUpsellStuckError):
        drv._dismiss_via_zone()
    assert len(adb.taps) == hinge._UPSELL_DISMISS_MAX_ATTEMPTS


def test_paid_upsell_stuck_error_halts_the_run_rather_than_being_routine():
    assert issubclass(PaidUpsellStuckError, hinge.HingeActionError)


def test_zone_dismiss_never_taps_outside_its_declared_zone_even_under_a_forbidden_zone():
    # Defense in depth: _dismiss_via_zone's tap still goes through the normal _tap() choke
    # point, so a malformed/overlapping zone would still be caught by the ordinary
    # forbidden-zone guard rather than silently firing. Declare a forbidden rect that exactly
    # covers the dismiss zone and confirm the dismiss now refuses instead of tapping into it.
    covering = AndroidAppSpec(
        app="zoneupsellblocked",
        package="com.example.zoneupsellblocked",
        calibrated=False,
        templates={"upsell_dismiss": "hinge_send_like_anyway.png"},
        upsell_dismiss_zone=(0.15, 0.10, 0.85, 0.34),
        forbidden_zones=((0.0, 0.0, 1.0, 0.40),),
        like_flow="direct",
    )
    adb = FakeAdb([_modal_frame()])
    drv = _drv(covering, adb)
    with pytest.raises(hinge.ForbiddenTapError):
        drv._dismiss_via_zone()
    assert adb.taps == []
