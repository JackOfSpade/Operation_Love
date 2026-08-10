"""Never-tap-the-paid-button, enforced rather than requested.

The owner rule is that super-likes and boosts are manual only: the bot does normal
like/pass and nothing else. On Hinge that was easy — the Rose upsell is a modal that
appears AFTER the action, and we dismiss it. Bumble is structurally worse: its paid
SuperSwipe control sits in the bottom action row physically BETWEEN Pass and Like, and
whether a placeholder or drifted coordinate lands on it is caught by anything downstream
depends on the account's SuperSwipe balance — measured live on the device 2026-08-10 (see
BUMBLE_SPEC in operation_love/drivers/android/bumble.py), a non-zero balance spends the
SuperSwipe SILENTLY with no modal at all; only a zero balance shows a purchase/confirmation
sheet first, and even that sheet's CTA sits almost exactly where the deck's own like/pass
coordinates do (see tests/test_android_upsell_protection.py). Either way, nothing
downstream of the tap can be relied on to catch a mistake here.

Two independent mechanisms answer that, and these tests pin both:

  1. decide_gesture="card_swipe" — Bumble decides by dragging the card, not by aiming at
     the button row at all. A drag cannot press a button it merely travels over.
  2. forbidden_zones — a declared no-go rect over the paid control. Any tap resolving
     inside it RAISES instead of firing, whatever aimed it there.

Mechanism 2 exists because mechanism 1 is a choice a future edit could reverse; the zone
check is the backstop that makes reversing it fail loudly instead of silently.

A later audit found mechanism 2 itself could be evaded: `_assert_tap_allowed` checked the
RAW, unclamped coordinate, but both real transports CLAMP the coordinate they actually
deliver -- so a coordinate whose fraction fell OUTSIDE 0..1 (impossible for a legitimate
value, since forbidden_zones rects are themselves constrained to 0..1) matched no zone,
passed the old check cleanly, and then got clamped onto a screen edge that CAN be inside a
zone. The "--- out-of-range coordinates" section below pins the fix: the checked point is
now the delivered point (clamp_xy, shared with both transports), an out-of-range value
raises loudly instead of silently clamping, and a zone check accounts for the jitter a real
tap can still add AFTER the check runs.
"""
import pytest

from operation_love.drivers import hinge
from operation_love.drivers.adb import clamp_xy
from operation_love.drivers.android.bumble import BUMBLE_SPEC
from operation_love.drivers.android_spec import AndroidAppSpec
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.hinge import (
    HINGE_SPEC,
    AndroidDriver,
    ForbiddenTapError,
    HingeActionError,
    OutOfRangeTapError,
    UnlocatedControlError,
)


class FakeAdb:
    """Records what actually reached the phone. Taps and swipes are counted separately
    because the whole point here is which of the two was issued."""

    def __init__(self):
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
        return b""

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, **_k):
        self.swipes.append((x1, y1, x2, y2))

    def scroll_up(self, *_a, **_k):
        self.scrolls += 1

    def text(self, s):
        self.texts.append(s)


def _drv(spec, adb, **overrides):
    class C:
        apps = {spec.app: {"halt_on_error": False, **overrides}}
    d = AndroidDriver(C(), spec)
    d._adb = adb
    d._touch = adb
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


# --- spec-level validation (fails at import, not mid-swipe) -----------------

def test_card_swipe_spec_without_drag_endpoints_is_rejected():
    # Discovering a missing endpoint mid-run would leave the driver falling back to a tap —
    # exactly what card_swipe exists to prevent. Catch it at construction instead.
    with pytest.raises(ValueError, match="swipe_start"):
        AndroidAppSpec(app="x", package="x.y", calibrated=False, decide_gesture="card_swipe")


def test_unknown_decide_gesture_is_rejected():
    with pytest.raises(ValueError, match="decide_gesture"):
        AndroidAppSpec(app="x", package="x.y", calibrated=False, decide_gesture="mash")


def test_malformed_forbidden_zone_is_rejected():
    # A silently-ignored bad rect would read as protection while providing none.
    with pytest.raises(ValueError, match="forbidden_zones"):
        AndroidAppSpec(app="x", package="x.y", calibrated=False,
                       forbidden_zones=((0.7, 0.1, 0.3, 0.9),))   # x0 > x1
    with pytest.raises(ValueError, match="forbidden_zones"):
        AndroidAppSpec(app="x", package="x.y", calibrated=False,
                       forbidden_zones=((0.1, 0.1, 1.5, 0.9),))   # outside 0..1


# --- Bumble decides by dragging, never by aiming at the button row ----------

def test_bumble_like_is_a_card_drag_and_issues_no_tap_at_all():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    drv.like()
    assert adb.taps == [], "a like must not tap: SuperSwipe sits between Pass and Like"
    assert len(adb.swipes) == 1
    x1, _y1, x2, _y2 = adb.swipes[0]
    assert x2 > x1, "like drags rightward"


def test_bumble_pass_is_a_card_drag_and_issues_no_tap_at_all():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    drv.dislike()
    assert adb.taps == []
    assert len(adb.swipes) == 1
    x1, _y1, x2, _y2 = adb.swipes[0]
    assert x2 < x1, "pass drags leftward"


def test_bumble_drag_starts_clear_of_its_own_paid_zone():
    # The drag start is the point that actually receives the gesture, so it is the one that
    # must not be over the paid control. Self-consistency check on the spec's own numbers.
    drv = _drv(BUMBLE_SPEC, FakeAdb())
    sx, sy = BUMBLE_SPEC.coords["swipe_start"]
    drv._assert_tap_allowed(int(sx * 1080), int(sy * 2400))   # must not raise


def test_bumble_fallback_button_coords_stay_outside_the_paid_zone():
    # These are only used if someone ever flips Bumble to decide_gesture="tap". If a future
    # calibration drifts them into the SuperSwipe rect, that must surface here rather than
    # on the phone.
    drv = _drv(BUMBLE_SPEC, FakeAdb())
    for name in ("like_heart", "pass_x"):
        fx, fy = BUMBLE_SPEC.coords[name]
        drv._assert_tap_allowed(int(fx * 1080), int(fy * 2400))


# --- the forbidden-zone backstop -------------------------------------------

def test_a_tap_inside_a_forbidden_zone_raises_and_never_reaches_the_phone():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x0, y0, x1, y1 = BUMBLE_SPEC.forbidden_zones[0]
    cx = int(((x0 + x1) / 2) * 1080)
    cy = int(((y0 + y1) / 2) * 2400)
    with pytest.raises(ForbiddenTapError, match="paid control"):
        drv._tap(cx, cy)
    assert adb.taps == [], "the refused tap must not have been issued anyway"


def test_tap_frac_is_guarded_too_not_just_raw_tap():
    # _tap_frac is how every fixed-coordinate action fires, so it must share the choke point.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x0, y0, x1, y1 = BUMBLE_SPEC.forbidden_zones[0]
    with pytest.raises(ForbiddenTapError):
        drv._tap_frac(((x0 + x1) / 2, (y0 + y1) / 2))
    assert adb.taps == []


def test_taps_outside_every_zone_go_through_normally():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    drv._tap(10, 10)
    assert adb.taps == [(10, 10)]


def test_a_spec_declaring_no_zones_is_unaffected():
    # Hinge declares none (its Rose is handled by the dismiss-modal path instead), so the
    # guard must be a no-op there rather than a behaviour change.
    assert HINGE_SPEC.forbidden_zones == ()
    adb = FakeAdb()
    drv = _drv(HINGE_SPEC, adb)
    drv._tap(540, 2200)
    assert adb.taps == [(540, 2200)]


def test_forbidden_tap_halts_the_run_rather_than_being_routine():
    # Subclassing HingeActionError is what makes the worker halt and preserve debug logs;
    # if this were a bare RuntimeError the run would treat it as an ordinary hiccup.
    assert issubclass(ForbiddenTapError, hinge.HingeActionError)


# --- out-of-range coordinates: checked point == delivered point -----------------------------
# The core of the fix: _assert_tap_allowed used to compute the zone-check fraction from the
# RAW, unclamped coordinate, while both real transports CLAMP the coordinate they actually
# deliver. A fraction outside 0..1 can never match a zone (forbidden_zones rects are
# themselves constrained to 0..1), so it sailed through the old check and then landed wherever
# the transport's clamp put it -- which can be inside a zone. These tests reproduce the
# demonstrated exploit against BUMBLE_SPEC's real zone (0.34, 0.80, 0.66, 1.00).

def test_clamp_xy_is_the_exact_arithmetic_both_transports_apply():
    # adb.py's Adb._clamp and uhid.py's _report both delegate to this now; pin its own
    # behaviour directly so the two transports and the zone check can never drift apart on
    # what "the point that will actually be delivered" means.
    assert clamp_xy(-5, -5, 1080, 2400) == (0, 0)
    assert clamp_xy(2000, 3000, 1080, 2400) == (1079, 2399)
    assert clamp_xy(540.4, 1200.6, 1080, 2400) == (540, 1201)


def test_out_of_range_fraction_raises_instead_of_silently_clamping_into_the_zone():
    # The demonstrated exploit: (540, 2520) = fraction (0.50, 1.05) of the 1080x2400 screen.
    # The OLD _assert_tap_allowed let this straight through (1.05 matches no zone whose own
    # range is <= 1.0) and the real transport then clamped y to 2399 -- fraction 0.9996,
    # INSIDE Bumble's zone (0.34, 0.80, 0.66, 1.00). It must now raise before either transport
    # is ever called.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x, y = int(0.50 * 1080), int(1.05 * 2400)
    with pytest.raises(OutOfRangeTapError, match=r"outside 0\.\.1"):
        drv._tap(x, y)
    assert adb.taps == [] and adb.swipes == [], "nothing may reach the phone"


def test_out_of_range_error_names_the_offending_value_and_app():
    # "raise a clear, actionable error naming the offending value and where it came from" --
    # an operator staring at this message needs to know WHAT was wrong and WHERE to look.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x, y = int(0.50 * 1080), int(1.05 * 2400)
    with pytest.raises(OutOfRangeTapError) as exc:
        drv._tap(x, y)
    msg = str(exc.value)
    assert str(x) in msg and str(y) in msg
    assert "bumble" in msg
    assert "coords" in msg or "frac" in msg   # points at the likely config culprit


def test_out_of_range_tap_error_is_a_hinge_action_error():
    # Same reasoning as ForbiddenTapError above: must halt the run and preserve debug logs,
    # not be swallowed as a routine, retryable miss.
    assert issubclass(OutOfRangeTapError, hinge.HingeActionError)


def test_out_of_range_check_fires_even_when_the_spec_declares_no_zones():
    # This is always a config/logic error regardless of whether forbidden_zones exists --
    # Hinge declares none, but an out-of-range tap on Hinge is exactly as nonsensical.
    assert HINGE_SPEC.forbidden_zones == ()
    adb = FakeAdb()
    drv = _drv(HINGE_SPEC, adb)
    with pytest.raises(OutOfRangeTapError):
        drv._tap(540, int(1.2 * 2400))
    assert adb.taps == []


def test_negative_fraction_also_raises_out_of_range():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    with pytest.raises(OutOfRangeTapError):
        drv._tap(-10, 1000)
    assert adb.taps == []


def test_read_scroll_frac_out_of_range_raises_the_same_way_at_runtime():
    # The other demonstrated exploit path: no `coords` tuple involved at all --
    # apps.bumble.read_scroll_frac=1.30 alone pushes an ordinary read-scroll's touch-down to
    # fraction 1.15. This bypasses AndroidAppSpec's own read_scroll_frac validation
    # deliberately (via an app-config override, exactly how config.yaml's
    # apps.bumble.read_scroll_frac reaches the driver -- see AndroidDriver.__init__) to prove
    # the RUNTIME check still catches it even if config.validate() were somehow skipped.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb, read_scroll_frac=1.30)
    with pytest.raises(OutOfRangeTapError):
        drv._scroll_down_one()
    assert adb.taps == [] and adb.swipes == [] and adb.scrolls == 0


def test_nan_coordinate_still_raises_before_any_touch_call():
    # Regression: NaN must still raise via int(nan) inside _tap -- BEFORE
    # _assert_tap_allowed's checks ever run, exactly as before this fix.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    with pytest.raises(ValueError, match="NaN"):
        drv._tap(float("nan"), 100)
    assert adb.taps == []


# --- jitter envelope: the checked point must cover plan_tap's plausible drift ---------------
# plan_tap's aim-radius/micro-slip jitter is applied INSIDE touch.tap(), AFTER
# _assert_tap_allowed returns -- so a nominal point that itself clears a zone can still have
# its REAL, delivered touch-down land inside one. _tap() closes this by widening the
# zone check by _TAP_ZONE_MARGIN_PX (human_motion.tap_jitter_margin_px()) on every side.

def test_tap_margin_rejects_an_aim_point_whose_jitter_could_enter_the_zone():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x0, y0, x1, y1 = BUMBLE_SPEC.forbidden_zones[0]
    w, h = 1080, 2400
    margin = drv._TAP_ZONE_MARGIN_PX
    assert margin > 0
    x_mid = int(((x0 + x1) / 2) * w)
    y_edge = int(y0 * h) - int(margin) + 2   # just inside the margin, just outside the zone
    # Sanity: the NOMINAL point itself must be outside the raw zone -- otherwise this is just
    # re-testing the plain zone check above, not the margin.
    assert not (y0 <= y_edge / h <= y1), "test setup must aim outside the raw zone"
    with pytest.raises(ForbiddenTapError):
        drv._tap(x_mid, y_edge)
    assert adb.taps == [], "the refused tap must not have been issued anyway"


def test_tap_margin_still_allows_a_point_clearly_outside_any_plausible_jitter():
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x0, y0, x1, y1 = BUMBLE_SPEC.forbidden_zones[0]
    w, h = 1080, 2400
    margin = drv._TAP_ZONE_MARGIN_PX
    x_mid = int(((x0 + x1) / 2) * w)
    y_clear = int(y0 * h) - int(margin) - 20   # comfortably clear of the widened zone
    drv._tap(x_mid, y_clear)
    assert adb.taps == [(x_mid, y_clear)]


def test_swipe_and_scroll_are_not_widened_by_the_tap_margin():
    # _swipe()/_scroll() pass margin_px=0: plan_swipe's very first sample is pinned exactly
    # to the start coordinate with zero jitter (see plan_swipe in human_motion.py), so a
    # point that clears the raw zone by less than the tap margin must still be ALLOWED for a
    # drag start -- widening it there would be over-cautious, not more correct.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    x0, y0, x1, y1 = BUMBLE_SPEC.forbidden_zones[0]
    w, h = 1080, 2400
    margin = drv._TAP_ZONE_MARGIN_PX
    x_mid = int(((x0 + x1) / 2) * w)
    y_edge = int(y0 * h) - int(margin) + 2   # inside the TAP margin, outside the raw zone
    drv._swipe(x_mid, y_edge, x_mid, y_edge - 200)   # must not raise
    assert adb.swipes == [(x_mid, y_edge, x_mid, y_edge - 200)]


# --- fail loud, not fallback: _await_button never taps a control it could not locate --------
# The owner's rule: either use the best humanized interaction we have, or fail loudly. This
# used to fall back to the calibrated fixed coordinate when vision missed; that fallback is
# now gone, because a stale coordinate can be occupied by a paid control by the time it fires,
# and the usual trigger is a MISSING CAPABILITY (OpenCV absent), not a one-off miss.

def test_await_button_raises_and_taps_nothing_when_control_cannot_be_located():
    # BUMBLE_SPEC declares no 'like' template at all, so there is no vision path AND -- since
    # the fixed-coordinate fallback is gone -- no way to act on this role at all. The important
    # guarantee is not just that this raises, it's that NOTHING reached the fake transport.
    adb = FakeAdb()
    drv = _drv(BUMBLE_SPEC, adb)
    with pytest.raises(UnlocatedControlError):
        drv._await_button("like", tries=1)
    assert adb.taps == [], "a control vision could not locate must never be tapped blind"


def test_await_button_message_blames_opencv_when_the_template_itself_cannot_load():
    # The far likelier real-world trigger than a one-off vision miss: OpenCV (or the asset
    # file) is missing, so EVERY match silently returns nothing, forever. The refusal message
    # must name that cause so an operator fixes the actual problem instead of assuming a
    # transient UI glitch and retrying.
    spec = AndroidAppSpec(app="badtemplate", package="x.y", calibrated=False,
                          coords={"like_heart": (0.5, 0.5), "pass_x": (0.1, 0.9)},
                          templates={"like": "does_not_exist_glyph.png"})
    adb = FakeAdb()
    drv = _drv(spec, adb)
    with pytest.raises(UnlocatedControlError, match="OpenCV"):
        drv._await_button("like", tries=1)
    assert adb.taps == []


def test_await_button_message_blames_the_screen_when_the_glyph_is_simply_not_there(monkeypatch):
    # Contrast case: the template loads fine (a real, working capability) -- the glyph just
    # is not on THIS screen. The message must say so, and must NOT blame OpenCV: a missing
    # capability and an ordinary "wrong screen / UI changed" miss are different problems that
    # need different fixes, and the message is the only thing an operator has to go on.
    adb = FakeAdb()
    drv = _drv(HINGE_SPEC, adb)
    monkeypatch.setattr(drv, "_locate_button", lambda _which: None)
    with pytest.raises(UnlocatedControlError) as exc:
        drv._await_button("like", tries=1)
    assert "OpenCV" not in str(exc.value)
    assert "not found on screen" in str(exc.value)
    assert adb.taps == []


def test_unlocated_control_error_is_a_hinge_action_error():
    # Same reasoning as ForbiddenTapError above: this must halt the run and preserve the
    # debug logs rather than being swallowed as a routine, retryable hiccup.
    assert issubclass(UnlocatedControlError, HingeActionError)


# --- _require_vision(): refuse to open a session rather than run with dead vision -----------
# The worst failure mode this closes: OpenCV silently absent, so every template match returns
# nothing and (before the fallback removal above) every action became a blind coordinate tap,
# indefinitely, with no error. Checking up front turns that into "the run never starts".

def test_require_vision_is_a_noop_for_a_spec_declaring_no_templates():
    spec = AndroidAppSpec(app="noviz", package="x.y", calibrated=False, templates={})
    drv = _drv(spec, FakeAdb())
    drv._require_vision()          # must not raise: there is no vision to lose


def test_require_vision_raises_when_a_declared_template_cannot_load():
    spec = AndroidAppSpec(app="badtemplate2", package="x.y", calibrated=False,
                          templates={"like": "does_not_exist_glyph.png"})
    drv = _drv(spec, FakeAdb())
    with pytest.raises(DriverClosed, match="does_not_exist_glyph.png"):
        drv._require_vision()


def test_require_vision_raises_from_open_session_when_a_template_cannot_load():
    # open_session() must call _require_vision() and let it stop the session BEFORE any ADB
    # connection is attempted -- refusing the run before it can act blind, not merely having
    # the helper available and unused. app="hinge" is registered and available in the
    # platform registry (calibrated=True there), so the platform check and the PIL/numpy
    # check both pass and _require_vision is what actually stops this one.
    # has_paid_upsell=False so the spec is constructible without an upsell_dismiss template:
    # this test is about _require_vision, and a calibrated spec otherwise has to prove it can
    # dismiss a paid modal (see test_calibrated_spec_without_an_upsell_dismiss_template_*).
    spec = AndroidAppSpec(app="hinge", package="co.hinge.app", calibrated=True,
                          has_paid_upsell=False,
                          templates={"like": "does_not_exist_glyph.png"})

    class C:
        apps = {}
    drv = AndroidDriver(C(), spec)
    with pytest.raises(DriverClosed, match="does_not_exist_glyph.png"):
        drv.open_session()
    assert drv._adb is None        # never got as far as constructing a real ADB connection


# --- _await_sheet_open(): comment_sheet's FIXED taps are gated on the sheet being open -------
# comment_box / send_like are fixed coordinates, safe to tap only while the sheet is actually
# up -- on Hinge, send_like sits mid-card, where per-photo/per-prompt like buttons live, so a
# missed heart tap could otherwise like the WRONG item instead of doing nothing.

def _heart_only_frame(cx=900, cy=1200):
    """A decodable frame carrying ONLY the heart glyph: the tap that opens the sheet succeeds,
    but nothing on screen proves the sheet actually opened afterwards."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(21)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    t = hinge._load_template("hinge_heart.png")
    th, tw = t.shape
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


class _ConstFrameAdb(FakeAdb):
    """A FakeAdb whose screencap always returns the same fixed frame."""
    def __init__(self, frame):
        super().__init__()
        self._frame = frame

    def screencap(self):
        return self._frame


def test_await_sheet_open_raises_when_the_sheet_never_appears_and_taps_nothing_after():
    heart_pt = (900, 1200)
    adb = _ConstFrameAdb(_heart_only_frame(*heart_pt))
    drv = _drv(HINGE_SPEC, adb)

    with pytest.raises(UnlocatedControlError):
        drv.like()

    # the heart tap is real and vision-located; the gate must stop BEFORE either of the
    # sheet's fixed-coordinate taps (comment_box / send_like) is issued
    assert adb.taps == [heart_pt]


def test_comment_sheet_spec_without_confirm_template_is_rejected():
    # comment_sheet taps FIXED coordinates into the sheet, and 'confirm' is the only way
    # _await_sheet_open can verify the sheet is actually open before doing so. Refuse the
    # spec at construction, not mid-swipe on a real device.
    with pytest.raises(ValueError, match="confirm"):
        AndroidAppSpec(app="x", package="x.y", calibrated=False, like_flow="comment_sheet")


# --- a spec cannot be declared "calibrated" while blind to paid upsells -------

def test_calibrated_spec_without_an_upsell_dismiss_template_is_rejected():
    # With no template, _handle_rose_upsell silently no-ops AND _verify_progress's only
    # question is "did the screen change" — which a modal appearing satisfies. So an
    # undetected paid-upgrade modal would be recorded as a SUCCESSFUL decision and left on
    # screen for the next gesture to hit unpredictably. Marking a spec calibrated is the
    # claim that it is ready for a real account, so that gap must block it.
    with pytest.raises(ValueError, match="upsell_dismiss"):
        AndroidAppSpec(app="x", package="x.y", calibrated=True,
                       templates={"like": "hinge_heart.png"})


def test_an_app_with_genuinely_no_paid_upsell_can_say_so_explicitly():
    # The escape hatch is a deliberate claim, not an omission — which is the whole point of
    # defaulting has_paid_upsell to True.
    spec = AndroidAppSpec(app="x", package="x.y", calibrated=True, has_paid_upsell=False,
                          templates={"like": "hinge_heart.png"})
    assert spec.has_paid_upsell is False


def test_an_uncalibrated_spec_may_still_be_incomplete():
    # Work in progress is expected to have gaps; calibrated=False already blocks it running.
    spec = AndroidAppSpec(app="x", package="x.y", calibrated=False, templates={})
    assert spec.calibrated is False


def test_hinge_ships_calibrated_and_can_dismiss_its_paid_upsell():
    assert HINGE_SPEC.calibrated is True
    assert HINGE_SPEC.has_paid_upsell is True
    assert "upsell_dismiss" in HINGE_SPEC.templates


def test_bumble_must_gain_an_upsell_template_before_it_can_be_calibrated():
    # Bumble's SuperSwipe purchase modal is the analogue of Hinge's Rose. This is the guard
    # that stops the calibration pass forgetting it.
    assert "upsell_dismiss" not in BUMBLE_SPEC.templates
    with pytest.raises(ValueError, match="upsell_dismiss"):
        AndroidAppSpec(**{**BUMBLE_SPEC.__dict__, "calibrated": True})


# --- scrolls are gestures too: their touch-down must be zone-checked ----------
# The choke-point comment originally claimed "EVERY tap goes through _tap()", which was
# true and beside the point: read-scrolls and scroll-to-top swipes went straight to the
# transport, so their touch-DOWN points were never zone-checked at all -- on every profile,
# every run. The margin was thinner than it looked: at the default read_scroll_frac=0.55
# the touch-down sits at y=0.775, 2.5% of the screen above Bumble's SuperSwipe zone.

def _spec_with_scroll(frac):
    return AndroidAppSpec(
        app="bumble", package="com.bumble.app", calibrated=False,
        decide_gesture="card_swipe", read_scroll_frac=frac,
        coords=dict(BUMBLE_SPEC.coords),
        forbidden_zones=BUMBLE_SPEC.forbidden_zones,
    )


def test_read_scroll_refuses_when_its_touch_down_lands_in_the_paid_zone():
    # read_scroll_frac is documented as config-overridable and calibrated on-device, so
    # raising it is an ordinary tweak -- not an abuse. At 0.65 the scroll's touch-down is
    # y=0.825, inside Bumble's SuperSwipe rect (0.80..1.00). It must refuse, not scroll.
    adb = FakeAdb()
    drv = _drv(_spec_with_scroll(0.65), adb)
    with pytest.raises(ForbiddenTapError):
        drv._scroll_down_one()
    assert adb.swipes == [] and adb.taps == [], "nothing may reach the phone"


def test_read_scroll_is_allowed_when_clear_of_the_zone():
    adb = FakeAdb()
    drv = _drv(_spec_with_scroll(0.55), adb)     # shipped default -> y=0.775, just clear
    drv._scroll_down_one()
    assert adb.scrolls == 1


def test_scroll_guard_accounts_for_the_column_jitter_not_just_the_centre():
    # scroll_x jitters the column by +/-SCROLL_X_JITTER_PX so repeated scrolls aren't
    # pixel-identical. A guard that only the nominal centre passes would let the jittered
    # extremes stray into a zone -- so both extremes are checked.
    from operation_love.drivers.adb import SCROLL_X_JITTER_PX
    w = 1080
    centre_frac = 0.5
    # A zone that the centre misses but the right-hand jitter extreme enters.
    edge = (int(w * centre_frac) + 1) / w
    spec = AndroidAppSpec(
        app="bumble", package="com.bumble.app", calibrated=False,
        decide_gesture="card_swipe", coords=dict(BUMBLE_SPEC.coords),
        forbidden_zones=((edge, 0.0, 1.0, 1.0),),
    )
    adb = FakeAdb()
    drv = _drv(spec, adb)
    assert SCROLL_X_JITTER_PX > 1                 # the extremes really do differ
    with pytest.raises(ForbiddenTapError):
        drv._scroll_down_one()
    assert adb.scrolls == 0


def test_scroll_to_top_undo_swipes_go_through_the_guard():
    # The undo-swipe is the mirror of a read-scroll, so it STARTS near the top of the screen
    # (y_near) and travels down. That start is nowhere near Bumble's bottom-of-card zone, so
    # the real spec must NOT refuse it -- a guard that blocked the undo path would strand the
    # profile mid-scroll. Prove the call is routed through the guard anyway by declaring a
    # zone over the top of the screen, where this gesture actually begins.
    ok_adb = FakeAdb()
    _drv(_spec_with_scroll(0.65), ok_adb)._scroll_to_top()
    assert ok_adb.swipes, "the real bottom-of-card zone must not block scroll-to-top"

    top_zone = AndroidAppSpec(
        app="bumble", package="com.bumble.app", calibrated=False,
        decide_gesture="card_swipe", read_scroll_frac=0.65,
        coords=dict(BUMBLE_SPEC.coords),
        forbidden_zones=((0.0, 0.0, 1.0, 0.30),),   # covers y_near, where the undo starts
    )
    adb = FakeAdb()
    drv = _drv(top_zone, adb)
    drv._capture_scrolls = 1
    with pytest.raises(ForbiddenTapError):
        drv._scroll_to_top()
    assert adb.swipes == [], "routed through _swipe, so a zone hit refuses before the phone"


def test_no_driver_gesture_reaches_the_transport_ungarded():
    # Structural guard: the ONLY places allowed to call the transport directly are the three
    # choke points (_tap/_swipe/_scroll). A new call site that forgets is what this catches.
    import inspect
    import re as _re
    from operation_love.drivers import hinge as _hinge
    src = inspect.getsource(_hinge.AndroidDriver)
    direct = _re.findall(r"self\.touch\.(tap|swipe|scroll_up)\(", src)
    assert len(direct) == 3, f"expected exactly the 3 choke points, found {direct}"
