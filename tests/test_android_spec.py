"""AndroidAppSpec / AndroidDriver — offline tests for the Hinge -> Android generalisation.

No device: a FakeAdb (same shape as tests/test_hinge_observe.py's, kept local rather than
imported — every file in this suite owns its own fake transport) scripts screencap frames and
records taps/swipes/text. These tests cover what's NEW in the generalisation and is NOT
already exercised bit-for-bit by test_hinge_observe.py:

  * spec wiring        — HINGE_SPEC / BUMBLE_SPEC field values, HingeDriver / BumbleAndroidDriver
                          actually bind to them, calibration flows into operation_love.platforms.
  * like_flow dispatch  — comment_sheet vs direct, driven by the spec, not hardcoded to "hinge".
  * optional templates  — a role missing from spec.templates skips vision-location ENTIRELY
                          (no screencap even attempted) and falls back to the fixed coordinate.
  * the paid-upsell rule — _handle_rose_upsell only ever taps a vision-matched "upsell_dismiss"
                          hit; with no such template declared (Bumble's current, real state) it
                          is structurally unable to tap anything at all.
"""
import pytest

from operation_love.drivers.android.bumble import BUMBLE_SPEC, BumbleAndroidDriver
from operation_love.drivers.android_spec import AndroidAppSpec
from operation_love.drivers.hinge import HINGE_SPEC, AndroidDriver, HingeDriver
from operation_love.drivers import hinge


class FakeAdb:
    def __init__(self, frames):
        self.frames = list(frames) or [b""]
        self.i = 0
        self.taps = []
        self.swipes = 0
        self.scrolls = 0
        self.texts = []
        self.screencaps = 0

    def screen_size(self):
        return (1080, 2400)

    def devices(self):
        return ["dev"]

    def shell(self, command="", **_):
        return ""

    def screencap(self):
        self.screencaps += 1
        return self.frames[min(self.i, len(self.frames) - 1)]

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, *_a, **_k):
        self.swipes += 1

    def scroll_up(self, *_a, **_k):
        self.scrolls += 1

    def text(self, s):
        self.texts.append(s)


class _Cfg:
    apps = {}


def _drv(spec, adb, **overrides):
    # halt_on_error defaults False so these tap-focused tests don't trigger the
    # post-action screencap-and-compare verification (same convention as _drv in
    # tests/test_hinge_observe.py).
    class C:
        apps = {spec.app: {"halt_on_error": False, **overrides}}
    d = AndroidDriver(C(), spec)
    d._adb = adb
    d._touch = adb
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


# --- spec wiring ------------------------------------------------------------
def test_android_app_spec_is_frozen():
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError (an AttributeError)
        HINGE_SPEC.calibrated = False


def test_android_app_spec_rejects_an_unknown_like_flow():
    with pytest.raises(ValueError, match="like_flow"):
        AndroidAppSpec(app="x", package="x.y", calibrated=False, like_flow="swipe_left_right")


def test_hinge_spec_matches_the_original_hardcoded_defaults():
    assert HINGE_SPEC.app == "hinge"
    assert HINGE_SPEC.package == "co.hinge.app"
    assert HINGE_SPEC.calibrated is True
    assert HINGE_SPEC.like_flow == "comment_sheet"
    assert HINGE_SPEC.accepts_opener is True
    assert HINGE_SPEC.think_time_calibrated is True
    assert HINGE_SPEC.change_threshold == 9.0
    assert HINGE_SPEC.scroll_captures == 8
    assert HINGE_SPEC.dwell_s == 1.1
    assert HINGE_SPEC.read_scroll_frac == 0.55
    assert set(HINGE_SPEC.templates) == {"like", "pass", "confirm", "upsell_dismiss"}
    assert set(HINGE_SPEC.coords) == {"like_heart", "pass_x", "comment_box", "send_like"}


def test_bumble_spec_is_an_uncalibrated_direct_placeholder():
    assert BUMBLE_SPEC.app == "bumble"
    assert BUMBLE_SPEC.package == "com.bumble.app"
    assert BUMBLE_SPEC.calibrated is False          # the guard: this is what keeps it refused
    assert BUMBLE_SPEC.like_flow == "direct"
    assert BUMBLE_SPEC.accepts_opener is False       # match-first-then-message: no swipe-time opener
    assert BUMBLE_SPEC.think_time_calibrated is False
    assert BUMBLE_SPEC.templates == {}               # no glyph assets exist yet


def test_hinge_driver_binds_hinge_spec():
    drv = HingeDriver(_Cfg())
    assert drv.spec is HINGE_SPEC
    assert drv.accepts_opener is True
    assert drv.supports_observe_like_intent is True
    assert drv.think_time_calibrated is True


def test_bumble_android_driver_binds_bumble_spec():
    drv = BumbleAndroidDriver(_Cfg())
    assert drv.spec is BUMBLE_SPEC
    assert drv.accepts_opener is False
    assert drv.supports_observe_like_intent is False
    assert drv.think_time_calibrated is False


def test_bumble_calibration_flag_gates_platform_availability():
    # Importing the android package (which BUMBLE_SPEC above already pulled in) feeds each
    # spec's `calibrated` flag into operation_love.platforms. Bumble's False is the thing
    # that actually stops it running -- see operation_love/platforms.py.
    from operation_love import platforms
    assert platforms.unavailable_reason("hinge") is None
    reason = platforms.unavailable_reason("bumble")
    assert reason and "not calibrated" in reason


# --- like_flow dispatch: driven by the spec, not hardcoded to "hinge" -------
_SHEET_SPEC = AndroidAppSpec(
    app="sheetapp",
    package="com.example.sheet",
    calibrated=False,
    coords={
        "like_heart": (0.5, 0.5),
        "pass_x": (0.1, 0.9),
        "comment_box": (0.5, 0.4),
        "send_like": (0.6, 0.6),
    },
    # A real 'like' template is required now: _await_button no longer falls back to the fixed
    # coordinate, so a comment_sheet flow can't tap a heart at all without one. 'confirm' is
    # required unconditionally by AndroidAppSpec for like_flow="comment_sheet" (see
    # __post_init__): comment_box/send_like are FIXED coordinates, safe to tap only while the
    # sheet is actually open, and 'confirm' is what _await_sheet_open matches to prove that.
    # Both reuse the real, shipped glyph assets rather than needing bespoke fixture PNGs.
    templates={"like": "hinge_heart.png", "confirm": "hinge_send_like.png"},
    like_flow="comment_sheet",
    accepts_opener=True,
)

_DIRECT_SPEC = AndroidAppSpec(
    app="directapp",
    package="com.example.direct",
    calibrated=False,
    coords={"like_heart": (0.5, 0.9), "pass_x": (0.5, 0.1)},
    # As above: a direct flow now needs a real 'like' template to tap anything at all.
    templates={"like": "hinge_heart.png"},
    like_flow="direct",
    accepts_opener=False,
)


def _sheet_frame(heart_xy, confirm_xy=(540, 1900)):
    """A decodable frame carrying the heart glyph (so the vision-located 'like' tap succeeds)
    and the comment-sheet 'confirm' glyph (so _await_sheet_open's gate, run right after the
    heart tap and before the fixed comment_box/send_like taps, finds the sheet actually up).
    heart_xy must sit right of the vision side-filter (x > 0.55 * screen width)."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(11)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in (("hinge_heart.png", heart_xy), ("hinge_send_like.png", confirm_xy)):
        t = hinge._load_template(name)
        th, tw = t.shape
        canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _direct_frame(like_xy):
    """A decodable frame carrying just the heart glyph, for a direct-flow fixture spec whose
    'like' tap is now vision-located rather than falling back to a fixed coordinate."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(12)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    t = hinge._load_template("hinge_heart.png")
    th, tw = t.shape
    cx, cy = like_xy
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def test_comment_sheet_flow_types_the_opener_between_heart_and_send():
    heart_pt = (900, 1200)      # right of the vision side-filter (x > 0.55 * 1080)
    adb = FakeAdb([_sheet_frame(heart_pt)])
    drv = _drv(_SHEET_SPEC, adb)
    drv.like("great smile in photo 2")

    w, h = adb.screen_size()
    box = (int(0.5 * w), int(0.4 * h))
    send = (int(0.6 * w), int(0.6 * h))
    assert adb.taps == [heart_pt, box, send]
    assert adb.texts == ["great smile in photo 2"]


def test_comment_sheet_flow_skips_comment_box_without_an_opener():
    heart_pt = (900, 1200)
    adb = FakeAdb([_sheet_frame(heart_pt)])
    drv = _drv(_SHEET_SPEC, adb)
    drv.like()

    w, h = adb.screen_size()
    send = (int(0.6 * w), int(0.6 * h))
    assert adb.taps == [heart_pt, send]     # no comment_box tap
    assert adb.texts == []


def test_direct_flow_taps_the_like_control_once_no_comment_box_no_text():
    like_pt = (900, 1200)
    adb = FakeAdb([_direct_frame(like_pt)])
    drv = _drv(_DIRECT_SPEC, adb)
    drv.like("this opener must be ignored -- accepts_opener is False for direct flows")

    assert adb.taps == [like_pt]         # exactly one tap: the like control itself
    assert adb.texts == []               # a direct flow has no comment box to type into


# --- templates are optional: missing role -> vision is skipped, not attempted --------------
def test_locate_button_skips_vision_entirely_when_spec_has_no_template():
    adb = FakeAdb([b"frame"])
    drv = _drv(BUMBLE_SPEC, adb)
    assert drv._locate_button("like") is None
    assert adb.screencaps == 0           # never even captured a frame to look at


def test_await_button_refuses_when_no_template_exists():
    # A spec declaring no template for this role has no vision path for it at all -- and
    # there is consequently no safe fixed-coordinate fallback left to use either: refuse
    # rather than tap the stale coordinate blind (it could now be a paid or irreversible
    # control by the time this runs).
    from operation_love.drivers.hinge import UnlocatedControlError
    adb = FakeAdb([b"frame"])
    drv = _drv(BUMBLE_SPEC, adb)

    with pytest.raises(UnlocatedControlError):
        drv._await_button("like", tries=2)

    assert adb.screencaps == 0           # no template -> not even a frame was captured to look at
    assert adb.taps == []                # and, above all, nothing was tapped


# --- the paid-upsell rule: never tap the paid option, structurally -------------------------
def test_handle_upsell_never_taps_anything_when_spec_has_no_dismiss_template():
    # Bumble's real, current state: no glyph assets exist yet, so BUMBLE_SPEC.templates has
    # no "upsell_dismiss" entry. There is consequently no coordinate ANYWHERE in the driver
    # for a paid button (no fixed-coord fallback exists for this role, unlike like/pass) --
    # it is structurally impossible for this to tap one, not merely unlikely.
    adb = FakeAdb([b"frame"])
    drv = _drv(BUMBLE_SPEC, adb)
    assert drv._handle_rose_upsell(tries=1) is False
    assert adb.taps == []


def _frame_with_glyph_at(cx, cy, seed=3):
    """A decodable frame with the (real, shipped) 'send anyway' dismiss glyph pasted at a
    known spot -- used generically here, not through HINGE_SPEC, to prove the never-tap-paid
    behavior isn't special-cased to Hinge."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    t = hinge._load_template("hinge_send_like_anyway.png")
    th, tw = t.shape
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


_UPSELL_SPEC = AndroidAppSpec(
    app="upsellapp",
    package="com.example.upsell",
    calibrated=False,
    templates={"upsell_dismiss": "hinge_send_like_anyway.png"},   # reuses the shipped asset
    like_flow="direct",
)


def test_handle_upsell_taps_only_the_matched_dismiss_glyph_never_elsewhere():
    cx, cy = 300, 1900
    adb = FakeAdb([_frame_with_glyph_at(cx, cy)])
    drv = _drv(_UPSELL_SPEC, adb)

    assert drv._handle_rose_upsell() is True
    assert adb.taps == [(cx, cy)]        # the ONE thing tapped is the dismiss glyph, nothing else


_DIRECT_WITH_UPSELL_SPEC = AndroidAppSpec(
    app="directupsell",
    package="com.example.du",
    calibrated=False,
    coords={"like_heart": (0.5, 0.9), "pass_x": (0.5, 0.1)},
    # A real 'like' template is required now: the direct like tap is vision-located, with no
    # fixed-coordinate fallback left, so this spec needs a real glyph to tap anything at all.
    templates={"like": "hinge_heart.png", "upsell_dismiss": "hinge_send_like_anyway.png"},
    like_flow="direct",
)


def _direct_upsell_frame(like_xy, dismiss_xy, seed=4):
    """A decodable frame carrying BOTH the heart glyph (so the direct like tap is
    vision-located) and the 'Send Like anyway' dismiss glyph (so the same scripted frame also
    drives the post-like upsell dismissal)."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in (
        ("hinge_heart.png", like_xy),
        ("hinge_send_like_anyway.png", dismiss_xy),
    ):
        t = hinge._load_template(name)
        th, tw = t.shape
        canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def test_direct_like_dismisses_the_upsell_and_taps_nothing_else():
    like_pt = (900, 1200)        # right of the vision side-filter (x > 0.55 * 1080)
    dismiss_pt = (250, 2100)
    adb = FakeAdb([_direct_upsell_frame(like_pt, dismiss_pt)])
    drv = _drv(_DIRECT_WITH_UPSELL_SPEC, adb)

    drv.like()

    assert adb.taps == [like_pt, dismiss_pt]   # like control, then the dismiss glyph -- never a 3rd tap
