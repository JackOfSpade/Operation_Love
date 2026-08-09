"""The generic Android driver (+ its Hinge binding) — physical Android phone over HOST-SIDE
ADB (no on-device helper).

This module is now home to TWO things:

  AndroidDriver  — the app-agnostic driver: perception (screencap + vision) and action
                   (humanized taps/swipes) for ANY Android dating app, parameterised by an
                   AndroidAppSpec (operation_love/drivers/android_spec.py). Nothing in here
                   is Hinge-specific anymore except the flow branch selected by
                   `spec.like_flow`.
  HingeDriver    — a thin subclass binding AndroidDriver to HINGE_SPEC.

Bumble's binding (BumbleAndroidDriver + BUMBLE_SPEC) lives in
operation_love/drivers/android/ instead of here, because it needs no access to anything
below — EXCEPT AndroidDriver itself, imported from this module.

Why is the generic driver defined in a file called "hinge.py" instead of its own module?
Because tests/test_hinge_observe.py, tests/test_hinge_vision.py and tools/hinge_inspect.py
monkeypatch/call module-level helpers (`_split_diff`, `_downsample`, `_match_glyph`,
`_load_template`, `Adb`, ...) as attributes of THIS module (`operation_love.drivers.hinge`).
Python resolves a bare name inside a function/method against the globals of the module it was
DEFINED in, not the module that imported it — so `monkeypatch.setattr(hinge, "_split_diff",
fake)` only takes effect on methods whose code actually lives in hinge.py's namespace. Moving
AndroidDriver's body out to a separate module would silently break every one of those patches
(the methods would keep calling the ORIGINAL helpers, the test's fake would just sit unused on
the wrong module) without any test provably failing to construct — a nasty, quiet drift. Rather
than touch 190+ existing test assertions to route patches through a new module, the generic
driver stays here and Bumble's binding imports AndroidDriver FROM here instead.

Off the old uiautomator2/emulator path: that installed an on-device server (atx-agent + the
uiautomator2 APK) which Play Integrity can flag and which ops/HINGE-PIXEL-RUNBOOK.md §5
forbids as the main account-protection guardrail. This driver instead talks to a genuine,
stock, physical Pixel through the host-side `Adb` transport only:

  * perception  — `adb exec-out screencap` frames, deduped by a downsampled
                  signature (no accessibility tree, no resource-ids)
  * action      — humanized `input motionevent` taps/swipes (curved, jittered,
                  log-normal timing) via Adb

Hinge lets you like a specific photo/prompt WITH a comment, so the opener is sent
at like-time (Signals behavior #2). Reading the whole profile slowly before
deciding (the dwell in `_capture_current`) is Signals behavior #1. We only ever
send a NORMAL like — never a Rose (Hinge's super-like) or, generically, any other paid
upgrade a given app might interstitial-upsell; Roses/boosts/etc are manual, always
(owner rule — see `_handle_rose_upsell`).

⚠️ LIVE-VERIFY: the tap COORDINATES (fractions of the screen) and the
screencap-diff thresholds below are best-effort from a partial UI map and MUST be
confirmed on a COMPLETE, Selfie-Verified profile (the like→comment→"Send Like"
sheet and the empty-deck state are gated until the profile is finished). All of
them are config-overridable under apps.hinge in config.yaml. The transport
(adb.py) and the capture/scroll mechanics are not gated and are validated.

If the USB/ADB link drops mid-run, Adb raises DriverClosed (parity with Bumble's
browser-closed path) so the worker stops cleanly and flushes buffered labels.
"""
from __future__ import annotations

import functools
import math
import random
import time
from pathlib import Path

from ..human import human_cooldown, human_delay
from ..perception.capture import Profile
from .adb import SCROLL_X_JITTER_PX, Adb, AdbError
from .android_spec import AndroidAppSpec
from .base import DatingAppDriver, DriverClosed, open_debug_log, snapshot_failure_frame
from .uhid import UhidTouch, UhidUnavailable

_ASSETS = Path(__file__).parent / "assets"


class HingeActionError(RuntimeError):
    """An autonomous action did not produce the expected on-screen change (stuck deck, missed
    tap, or an unknown screen). NOT a DriverClosed (which is a clean, restart-safe stop): this
    is unexpected, so the worker halts the run and preserves the debug logs.

    Named for Hinge (the first, and so far only calibrated, Android app) but raised by
    AndroidDriver generically — any Android app's driver instance can raise it."""


class ForbiddenTapError(HingeActionError):
    """A tap resolved to a point inside one of the spec's forbidden_zones and was refused.

    These zones guard PAID controls (Bumble's SuperSwipe sits between Pass and Like;
    Hinge's Rose). Tapping one spends the owner's money and breaks a standing rule that
    super-likes and boosts are manual-only. Refusing is always correct here: a missed
    like costs one profile, a mis-tap costs money and cannot be undone.

    Deliberately a HingeActionError, so it halts the run and preserves the debug logs like
    any other unexpected-screen condition rather than being swallowed as routine."""


class UnlocatedControlError(HingeActionError):
    """Vision could not locate a control, so the action was refused rather than guessed.

    This used to fall back to the calibrated fixed coordinate and tap it blind, on the
    reasoning that acting degraded beat not acting. That reasoning is wrong here. The
    coordinate is only valid for the app version and screen state it was measured on: Hinge
    shifts its floating buttons with the "Start sending likes" banner and per-profile photo
    heights, and the app updates on its own schedule. A blind tap at a stale point does not
    "probably still work" — it lands on whatever now occupies that pixel, which on these
    screens can be a paid control (a Rose, a SuperSwipe) or an irreversible one.

    Worse, the trigger is usually not a transient miss but a MISSING CAPABILITY: if OpenCV
    is not installed, every template match returns nothing and EVERY action silently becomes
    a blind coordinate tap, indefinitely, with no error. That exact bug shipped once already
    (a launcher omitted the `hinge` extra, so opencv was never installed).

    Failing here costs one profile. Guessing costs money or the account."""


_OBSERVE_POLL_S = 0.35     # internal sampling cadence for your manual tap (not app-facing)

# Hinge's spec: exactly today's values (formerly the module-level `DEFAULTS` dict + the
# `apps.hinge` block in config.yaml). calibrated=True — coords/templates verified live on the
# Pixel 7a (1080x2400) 2026-06-27. Config-overridable (apps.hinge.* in config.yaml).
HINGE_SPEC = AndroidAppSpec(
    app="hinge",
    package="co.hinge.app",
    calibrated=True,
    coords={
        # Action points as FRACTIONS of the screen (x, y in 0..1). like_heart / pass_x are
        # VISION-located at runtime (template-matched glyph) — these are only the FALLBACK if
        # vision can't find the glyph. comment_box / send_like are fixed (the like-sheet
        # layout is consistent).
        "like_heart": (0.868, 0.667),   # FALLBACK only — heart vision-located on the first photo
        "pass_x": (0.116, 0.848),       # FALLBACK only — X vision-located (floating, bottom-left)
        "comment_box": (0.500, 0.529),  # comment field in the like sheet
        "send_like": (0.643, 0.576),    # "Send Like" button (kept clear of the 🌷Rose button)
    },
    templates={
        "like": "hinge_heart.png",
        "pass": "hinge_pass_x.png",
        "confirm": "hinge_send_like.png",              # the "Send Like" sheet's own glyph
        "upsell_dismiss": "hinge_send_like_anyway.png",  # "Send Like anyway" — NEVER the Rose button
    },
    like_flow="comment_sheet",
    accepts_opener=True,          # Hinge sends the opener as a comment at like-time
    think_time_calibrated=True,   # human_motion._THINK was measured on this app
    change_threshold=9.0,         # mean abs grayscale delta (0..255) on a 24x24 downsample
    scroll_captures=8,            # max screencaps while reading one profile
    dwell_s=1.1,                  # per-card read dwell (humanized) — Signals behavior #1
    read_scroll_frac=0.55,        # how far each read-scroll advances the profile
)


def _downsample(frame: bytes, size: int = 24):
    """Frame -> small grayscale int16 array, or None if PIL/numpy unavailable.

    A 24x24 grayscale is enough to tell whole-card changes (advance / like sheet)
    from frame noise, without reading the accessibility tree.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
        im = Image.open(BytesIO(frame)).convert("L").resize((size, size))
        return np.asarray(im, dtype="int16")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> caller falls back to exact bytes
        return None


# A screen-off / keyguard-locked capture. Both bounds must hold: near-BLACK and
# near-UNIFORM. Darkness alone would false-positive on legitimately dark content
# (dark mode, a night photo) — but real content has text and edges, so its
# peak-to-peak spread stays large even when its mean is low. A blank framebuffer
# has essentially no spread at all.
_BLANK_MEAN_MAX = 2.0      # 0..255 mean luminance
_BLANK_SPREAD_MAX = 2.0    # 0..255 peak-to-peak


def _is_blank_frame(frame: bytes) -> bool:
    """True when `frame` is a solid-black capture (device asleep or on the keyguard).

    Decodes independently rather than reusing _downsample, deliberately. This asks a
    question about RAW PIXELS ("is the panel dark?"), whereas _downsample serves the
    dedup/diff layer's notion of a frame SIGNATURE — a different concern that callers
    and tests legitimately substitute. Sharing the helper would let a stubbed
    signature masquerade as a dead screen (and would let a monkeypatch silently
    disable this guard).

    Returns False when PIL/numpy are unavailable or the bytes don't decode — if we
    cannot see, we must not claim "blank" and halt a healthy run.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
        arr = np.asarray(
            Image.open(BytesIO(frame)).convert("L").resize((24, 24)), dtype="int16")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> "can't tell", not "blank"
        return False
    return bool(np.mean(arr) <= _BLANK_MEAN_MAX and np.ptp(arr) <= _BLANK_SPREAD_MAX)


def _split_diff(a: bytes, b: bytes) -> tuple[float, float]:
    """(top-half delta, bottom-half delta) between two frames.

    Lets us distinguish a like sheet sliding up over the BOTTOM (photo still
    visible up top) from a whole-card advance (top changes too). Falls back to a
    binary same/different signal when PIL/numpy aren't present.
    """
    da, db = _downsample(a), _downsample(b)
    if da is None or db is None:
        return (0.0, 0.0) if a == b else (999.0, 999.0)
    import numpy as np
    d = np.abs(da - db)
    h = d.shape[0]
    return float(np.mean(d[: h // 2])), float(np.mean(d[h // 2:]))


def _frame_sig(frame: bytes) -> bytes:
    """Stable signature for dedup: the downsampled bytes, or a hash fallback."""
    arr = _downsample(frame)
    if arr is not None:
        return arr.tobytes()
    import hashlib
    return hashlib.md5(frame).digest()


# --- vision: locate an action BUTTON by its glyph (not a fixed coord) --------------------
# Hinge's like-heart sits at the bottom-right of EACH photo and the pass-X floats bottom-left.
# A fixed fraction is unreliable: the "Start sending likes" banner and per-profile photo
# aspect ratios shift the heart vertically, and the X's white disc merges into the white
# background of a prompt card (so a plain white-blob detector loses it). We instead template-
# match the dark glyph (heart / X), which stays distinct on ANY background. The glyphs are
# fixed-resolution UI assets (this driver targets one device — the Pixel 7a at 1080x2400),
# so matches are essentially exact. Any app's spec can plug its own glyph PNGs into the same
# roles (see AndroidAppSpec.templates); an app with no template for a role simply skips
# vision-location for it (AndroidDriver._template returns None, callers fall back to a fixed
# coordinate, or — for upsell_dismiss, which has no safe fixed-coordinate fallback — no-op).

@functools.lru_cache(maxsize=None)
def _load_template(name: str):
    """Load a grayscale button-glyph template via cv2 (cached). None if cv2/asset missing."""
    try:
        import cv2
    except Exception:  # noqa: BLE001 — cv2 absent -> caller falls back to fixed coords
        return None
    return cv2.imread(str(_ASSETS / name), cv2.IMREAD_GRAYSCALE)   # None if file missing


def _match_glyph(frame: bytes, template, *, side: str, threshold: float = 0.6) -> list:
    """Locate a button glyph in a screencap by normalized cross-correlation. `side` keeps only
    matches on the right ('like' heart) or left ('pass' X) of the screen. Returns (x, y)
    centers sorted top->bottom (so [0] is the first photo's heart after scroll-to-top). Empty
    list if cv2/template/decoding is unavailable — the driver then uses its fixed-coord fallback."""
    if template is None:
        return []
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        h, w = img.shape
        th, tw = template.shape
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        work = res.copy()
        centers: list = []
        for _ in range(12):                                   # non-max suppression loop
            _, maxv, _, loc = cv2.minMaxLoc(work)
            if maxv < threshold:
                break
            cx, cy = loc[0] + tw // 2, loc[1] + th // 2
            work[max(0, loc[1] - th // 2): loc[1] + th, max(0, loc[0] - tw // 2): loc[0] + tw] = -1.0
            if side == "right" and cx < w * 0.55:
                continue
            if side == "left" and cx > w * 0.45:
                continue
            centers.append((int(cx), int(cy)))
        centers.sort(key=lambda c: c[1])
        return centers
    except Exception:  # noqa: BLE001 — any cv2/decoding failure -> fixed-coord fallback
        return []


def _retry_until(check_fn, tries: int, delay_s: float, *, is_found=bool):
    """Call check_fn() up to `tries` times, sleeping human_delay(delay_s) after each
    attempt where `is_found(result)` is False. Returns the first result for which
    is_found(result) is True, or None once attempts run out. The shared shape behind
    _await_button/_verify_progress/_handle_rose_upsell: poll something on-screen with
    humanized pacing until it appears or we give up.

    `is_found` defaults to `bool` (any truthy result counts as found) — correct for
    every current caller (None/bool/non-empty-list are never falsy-but-genuinely-found
    here), but pass an explicit predicate (e.g. `lambda r: r is not None`) for a
    check_fn whose "found" result can legitimately be falsy, so a falsy-but-valid hit
    isn't mistaken for "not found yet" and silently retried away."""
    for _ in range(max(1, tries)):
        result = check_fn()
        if is_found(result):
            return result
        time.sleep(human_delay(delay_s))
    return None


class AndroidDriver(DatingAppDriver):
    """Drives ANY Android dating app whose UI fits the shape captured by AndroidAppSpec:
    a scrollable profile card, a like/pass control pair (vision-located with a fixed-coord
    fallback), and either a comment-sheet or a direct like flow. See HingeDriver (below, in
    this module) and BumbleAndroidDriver (operation_love/drivers/android/bumble.py) for the
    two current bindings.
    """

    def __init__(self, cfg, spec: AndroidAppSpec):
        self.spec = spec
        self.accepts_opener = spec.accepts_opener
        self.think_time_calibrated = spec.think_time_calibrated
        app_cfg = (getattr(cfg, "apps", {}) or {}).get(spec.app, {})
        self.serial = app_cfg.get("serial") or None
        self.adb_path = app_cfg.get("adb_path", "adb")
        self.package = app_cfg.get("package", spec.package)
        self.scroll_captures = max(1, int(app_cfg.get("scroll_captures", spec.scroll_captures)))
        self.dwell_s = float(app_cfg.get("dwell_s", spec.dwell_s))
        self.read_scroll_frac = float(app_cfg.get("read_scroll_frac", spec.read_scroll_frac))
        self.change_threshold = float(app_cfg.get("change_threshold", spec.change_threshold))
        self.coords = {**spec.coords, **(app_cfg.get("coords") or {})}
        self._adb: Adb | None = None
        self._capture_scrolls = 0     # read-scrolls the last _capture_current did; _scroll_to_top's ceiling
        # Exact geometry of every forward scroll since the last confirmed top.  A count alone
        # was enough while all read-scrolls had one fixed distance; auto-mode can now vary both
        # distance and lane per gesture, so undo must be based on what actually happened.
        self._capture_scroll_ledger: list[tuple[float, float]] = []
        self._profile_capture_limit = self.scroll_captures
        self._touch = None            # touch transport: UhidTouch (genuine) or Adb (input fallback)
        self.touch_backend = app_cfg.get("touch_backend", "auto")   # auto | uhid | adb
        self._observe_ready = False   # True once open_session validated PIL/numpy + device
        self.debug_log = bool(app_cfg.get("debug_log", False))
        self.debug_dir = app_cfg.get("debug_dir", f"./data/{spec.app}_debug")
        self.halt_on_error = bool(app_cfg.get("halt_on_error", True))   # auto: STOP on unexpected (preserve logs)
        self._dbg = None              # HingeDebugLog (set in open_session when debug_log is on)

    def _require_vision(self) -> None:
        """Refuse to open a session if this app's declared glyph templates can't be matched.

        Every template this spec declares must actually load. Checking up front turns the
        worst failure mode in the driver — OpenCV quietly absent, so every match returns
        nothing — from "the run continues, blind, indefinitely" into "the run never starts".
        That is not hypothetical: a launcher shipped without the `hinge` extra once, so
        opencv was never installed and the vision path was dead for an unknown period.

        A spec declaring NO templates is fine here: it simply has no vision to lose, and
        _await_button refuses on its behalf if anything ever tries to aim at a control."""
        missing = [name for role, name in self.spec.templates.items()
                   if _load_template(name) is None]
        if missing:
            raise DriverClosed(
                f"{self.spec.app}: cannot load glyph template(s) {', '.join(sorted(missing))} "
                f"from {_ASSETS}. Vision-location would silently find nothing and every "
                f"action would be refused. OpenCV is the usual cause: "
                f"pip install -e '.[hinge]'"
            )

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        # The last gate before this driver can touch a real phone with a real account on it.
        # Constructing an AndroidDriver is inert (no adb, no touch transport until below), so
        # the registry check lives HERE rather than in the factory — that keeps calibration
        # tooling able to build an uncalibrated driver, which is how one stops being
        # uncalibrated, while still making it impossible to open a session with placeholder
        # coordinates. supervisor.run() and HubState.start() check earlier and more loudly;
        # this one catches anything that reached a driver by another path.
        from .. import platforms
        reason = platforms.unavailable_reason(self.spec.app)
        if reason:
            raise DriverClosed(reason)

        import importlib.util
        if not (importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL")):
            raise DriverClosed(f"{self.spec.app} driver requires PIL and numpy. Install them via `pip install -e '.[ml]'` or `pip install pillow numpy`.")

        self._require_vision()

        self._adb = Adb(self.serial, adb_path=self.adb_path)
        ready = self._adb.devices()
        if not ready:
            raise DriverClosed(f"No ADB device connected for {self.spec.app}")
        if self.serial and self.serial not in ready:
            raise DriverClosed(f"{self.spec.app} device {self.serial} not connected (adb devices: {ready})")
        self._adb.screen_size()                       # cache geometry for clamping/coords
        self._adb.shell(f"monkey -p {self.package} -c android.intent.category.LAUNCHER 1")
        time.sleep(human_cooldown(1.5))               # let the app come to the foreground
        self._touch = self._make_touch()              # genuine UHID touches; adb input fallback
        self._observe_ready = True                    # only True once fully open (touch ready too)
        if self.debug_log:
            self._dbg = open_debug_log(self.debug_dir)

    def close(self) -> None:
        if self._touch is not None and self._touch is not self._adb:
            try:
                self._touch.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the real outcome
                pass
        self._touch = None
        self._adb = None

    # --- helpers --------------------------------------------------------
    @property
    def adb(self) -> Adb:
        if self._adb is None:
            raise DriverClosed(f"{self.spec.app} session is not open")
        return self._adb

    @property
    def touch(self):
        if self._touch is None:
            raise DriverClosed(f"{self.spec.app} session is not open")
        return self._touch

    def _make_touch(self):
        """Touch transport. `auto` and `uhid` both REQUIRE the genuine UHID virtual
        touchscreen; only an explicit `adb` accepts the degraded one.

        `auto` used to mean "try UHID, quietly fall back to adb `input`". Both transports
        are humanized (curved paths, jitter, log-normal timing), but they are not
        equivalent: UHID delivers genuine kernel-level TOOL_TYPE_FINGER events with a
        variable pressure ramp at ~180Hz, while `adb shell input motionevent` cannot vary
        pressure at all — every contact reports the same synthetic value. A silent
        downgrade meant the run could look identical while emitting a materially less human
        touch signature, for an unknown length of time, on the account we care about.

        So `auto` now means "prefer UHID and fail loudly if it is unavailable". The
        degraded path is still reachable, but only by explicitly writing
        `touch_backend: adb`, which is an operator decision rather than an accident."""
        if self.touch_backend == "adb":
            print(f"{self.spec.app}: touch_backend='adb' — using the DEGRADED input transport "
                  f"(humanized paths, but constant pressure). Set 'auto' for genuine UHID touches.")
            return self._adb
        try:
            t = UhidTouch(self._adb)
            t.open()
            return t
        except (UhidUnavailable, AdbError) as exc:
            raise DriverClosed(
                f"{self.spec.app}: the genuine UHID touchscreen is unavailable ({exc}), and "
                f"touch_backend={self.touch_backend!r} will not silently downgrade to the "
                f"constant-pressure adb input transport. Fix the device/UHID path, or set "
                f"apps.{self.spec.app}.touch_backend: adb to accept the degraded transport "
                f"deliberately."
            ) from exc

    # --- tap choke point (forbidden-zone guard) -------------------------
    # EVERY gesture this driver issues goes through _tap() / _swipe() / _scroll(), never the
    # transport directly, so the no-go check cannot be bypassed by a new call site forgetting
    # about it. This originally covered taps ONLY, and the comment claimed more than the code
    # delivered: read-scrolls and scroll-to-top swipes went straight to the transport, so
    # their touch-down points were never zone-checked at all — on every profile, every run.
    def _assert_tap_allowed(self, x: int, y: int) -> None:
        zones = getattr(self.spec, "forbidden_zones", ())
        if not zones:
            return
        w, h = self.adb.screen_size()
        fx = x / w if w else 0.0
        fy = y / h if h else 0.0
        for zone in zones:
            x0, y0, x1, y1 = zone
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                raise ForbiddenTapError(
                    f"refused a tap at ({x}, {y}) = ({fx:.3f}, {fy:.3f}) of the screen: it "
                    f"lands inside {self.spec.app}'s forbidden zone {zone}, which guards a "
                    f"paid control. Refusing rather than risking a paid action.")

    def _tap(self, x, y) -> None:
        x, y = int(x), int(y)
        self._assert_tap_allowed(x, y)
        self.touch.tap(x, y)

    def _swipe(self, x1, y1, x2, y2) -> None:
        """Every explicit drag goes through here, for the same reason every tap goes
        through _tap(). Only the START point is zone-checked: the touch-down claims the
        gesture, so a drag that merely travels over a control does not press it."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self._assert_tap_allowed(x1, y1)
        self.touch.swipe(x1, y1, x2, y2)

    def _scroll(self, frac: float, x_frac: float = 0.5) -> None:
        """touch.scroll_up() with the forbidden-zone guard every other gesture gets.

        scroll_up computes its geometry INSIDE the transport, so the driver cannot see
        where the touch-down lands without mirroring that math — which _scroll_to_top
        already does for its undo-swipes. Both transports use the same start point:
        (scroll_x(w, x_frac), h * (0.5 + frac/2)).

        This guard was missing, and the margin is thinner than it looks: at the default
        read_scroll_frac=0.55 the touch-down sits at y=0.775, just 2.5% of the screen above
        Bumble's SuperSwipe zone (which starts at 0.80). Raising read_scroll_frac to 0.65 —
        an ordinary config-level calibration tweak — would put EVERY read-scroll's
        touch-down inside the paid control's territory, on every profile, unguarded.

        scroll_x jitters the column by +/-SCROLL_X_JITTER_PX so repeated scrolls aren't
        pixel-identical, so both extremes are checked rather than the nominal centre — a
        guard that only the average case passes is not a guard."""
        w, h = self.adb.screen_size()
        y = int(h * (0.5 + frac / 2))
        nominal = int(w * x_frac)
        for x in (nominal - SCROLL_X_JITTER_PX, nominal + SCROLL_X_JITTER_PX):
            self._assert_tap_allowed(x, y)
        self.touch.scroll_up(frac, x_frac)

    def _tap_frac(self, frac) -> None:
        w, h = self.adb.screen_size()
        self._tap(frac[0] * w, frac[1] * h)

    def _decide_by_card_swipe(self, decision: str) -> None:
        """Deliver a like/pass by dragging the card sideways instead of tapping a control.

        Bumble places its paid SuperSwipe BETWEEN Pass and Like at the bottom of the card,
        so a placeholder or drifted coordinate can land on it, and unlike Hinge's Rose
        there is no confirmation modal afterwards to catch the mistake. A drag begins in
        the middle of the card and cannot press a button it merely travels over, so the
        paid control is unreachable by construction rather than by careful aiming.

        The drag goes through the same touch transport as everything else, so it inherits
        the humanized kinematics (Fitts-law duration, curved path, tremor, pressure ramp).
        Only the START point is zone-checked: a drag that ends over a button does not
        press it, since the press was already claimed by whatever was under the start."""
        w, h = self.adb.screen_size()
        # self.coords, not self.spec.coords: config.yaml's apps.<app>.coords is layered over
        # the spec, so these can be calibrated on the device without editing code — which is
        # the whole workflow for taking Bumble from placeholder to calibrated.
        start = self.coords["swipe_start"]
        end = self.coords["swipe_like_end" if decision == "like" else "swipe_pass_end"]
        x1, y1 = int(start[0] * w), int(start[1] * h)
        x2, y2 = int(end[0] * w), int(end[1] * h)
        self._swipe(x1, y1, x2, y2)

    def _template(self, role: str):
        """Load the template PNG for a logical UI role (AndroidAppSpec.templates), or None if
        this app's spec declares no template for that role at all — _match_glyph's own
        None-template guard then makes every caller here treat it as a clean 'not found'."""
        name = self.spec.templates.get(role)
        return _load_template(name) if name else None

    def _locate_button(self, which: str):
        """Screencap and template-match the like-heart ('like' -> topmost right glyph) or the
        floating pass-X ('pass' -> left glyph). Returns (x, y) or None when not visible, OR
        when this app's spec has no template for that role at all — vision-location is then
        skipped WITHOUT even capturing a frame; _await_button falls back to the fixed coord."""
        role = "like" if which == "like" else "pass"
        if role not in self.spec.templates:
            return None
        centers = _match_glyph(self._screencap(), self._template(role),
                               side="right" if which == "like" else "left")
        return centers[0] if centers else None

    def _await_button(self, which: str, tries: int = 5):
        """Locate a button by vision, retrying through short settle waits.

        Retries because Hinge fades the floating like/pass buttons out DURING a scroll and
        back in once it settles, so a tap fired immediately after a scroll-read can miss a
        button that is genuinely there.

        RAISES rather than falling back to the calibrated fixed coordinate. See
        UnlocatedControlError for why guessing is worse than stopping. The fixed coords stay
        in the spec as a calibration reference and as the anchor for tooling, but nothing
        taps them on this path."""
        pt = _retry_until(lambda: self._locate_button(which), tries, 0.4)
        if pt is not None:
            return pt

        role = "like" if which == "like" else "pass"
        if role not in self.spec.templates:
            why = (f"{self.spec.app} declares no '{role}' glyph template, so this control "
                   f"cannot be vision-located at all")
        elif _load_template(self.spec.templates[role]) is None:
            why = (f"the '{role}' template {self.spec.templates[role]!r} could not be loaded "
                   f"— OpenCV is very likely missing (install the extra: "
                   f"pip install -e '.[hinge]')")
        else:
            why = (f"the '{role}' glyph was not found on screen after {tries} attempts — the "
                   f"app UI may have changed, or the screen is not the swipe deck")
        # .get(), not [] — a spec is allowed to omit these calibration-reference coords, and a
        # KeyError raised while BUILDING an error message would replace the real diagnosis
        # with a confusing one.
        ref = self.coords.get("like_heart" if which == "like" else "pass_x")
        instead = f" Not falling back to the fixed coordinate {ref} —" if ref else " —"
        raise UnlocatedControlError(
            f"refusing to {which}: {why}.{instead} a blind tap at a stale point can hit a "
            f"paid or irreversible control.")

    def _await_sheet_open(self, tries: int = 5) -> None:
        """Confirm the comment / "Send Like" sheet really opened, before tapping into it.

        comment_box and send_like are FIXED coordinates rather than vision-located, on the
        grounds that the sheet's layout is consistent. That is only true WHILE THE SHEET IS
        UP. If the heart tap missed, or the sheet was slow, or the app changed, those two
        taps instead land on the profile card underneath — and on Hinge the send_like point
        sits in the middle of the card, where per-photo and per-prompt like buttons live. A
        missed heart tap could therefore like the wrong item rather than doing nothing.

        So the fixed taps are gated on the sheet being visibly present. This is the same
        `confirm` glyph _verify_like_landed already uses to decide the sheet has CLOSED;
        checking it on the way in as well costs one screencap.

        (When a device is available, consider tapping the matched glyph position instead of
        the fixed send_like coordinate — strictly more robust to drift. Not done blind:
        whether this template depicts the button itself or a label beside it needs eyes on
        the real sheet, and the fixed coordinate is at least live-validated.)"""
        if "confirm" not in self.spec.templates:
            raise UnlocatedControlError(
                f"{self.spec.app} uses the comment_sheet like flow but declares no 'confirm' "
                f"template, so there is no way to tell the sheet opened. Refusing to tap the "
                f"fixed comment box / send coordinates on an unverified screen.")
        found = _retry_until(
            lambda: _match_glyph(self._screencap(), self._template("confirm"),
                                 side="any", threshold=0.6) or None,
            tries, 0.4)
        if not found:
            raise UnlocatedControlError(
                f"refusing to continue the like: the '{self.spec.app}' comment sheet never "
                f"appeared after tapping the heart ({tries} attempts). Not tapping the fixed "
                f"comment box {self.coords.get('comment_box')} / send {self.coords.get('send_like')} "
                f"coordinates — with no sheet up they land on the profile card underneath, "
                f"where they can hit a per-item like.")

    # --- capture (blank-frame guard) -----------------------------------
    def _screencap(self, *, on_blank: str = "raise") -> bytes | None:
        """Every decision-making capture goes through here, never `self.adb.screencap()`.

        `adb exec-out screencap` on a device that is asleep or sitting on the
        keyguard SUCCEEDS and returns a solid-black PNG — it does not error. Two
        identical black frames read as "nothing changed", so a run would stall
        blind; and a good frame followed by a black one reads as a BIG delta, which
        wait_for_decision would mis-score as a card advance (a phantom "pass").

        `on_blank` picks what a persistently-blank screen means to THIS caller:

        - "raise" (default) — for paths that are about to ACT or verify an action.
          Tapping blind on a burner is never an acceptable degradation, so this
          raises regardless of `halt_on_error`. That is a deliberate departure from
          the `halt_on_error` gating used by _snap/_verify_progress: those cover
          *unexpected* errors the owner may opt out of halting on, whereas a blank
          screen is a physical impossibility — there is nothing to look at.

        - "none" — returns None for PASSIVE polling loops (wait_for_decision,
          _await_like_resolved). Those wait on a HUMAN with no timeout and send no
          touches, so nothing keeps the screen alive; a screen-off there is an
          ordinary, recoverable event. They skip the iteration and keep watching,
          which is both non-fatal AND avoids the phantom-pass above.

        Retried once after a settle so a single odd frame mid-transition can't halt
        a healthy run; only a frame that is STILL blank counts.

        The two deliberate exceptions that keep calling `self.adb.screencap()` raw:
        _dbg_action (best-effort debug capture, already swallows failures) and
        snapshot_failure (must record whatever is on screen — including black — and
        must never raise while handling another error).
        """
        frame = self.adb.screencap()
        if not _is_blank_frame(frame):
            return frame
        time.sleep(human_delay(0.4))
        frame = self.adb.screencap()
        if not _is_blank_frame(frame):
            return frame
        if on_blank == "none":
            return None
        raise HingeActionError(
            f"screen is blank ({self._blank_reason()}) — every capture is solid black, so "
            "the run would act blind; wake and unlock the device before continuing")

    def _blank_reason(self) -> str:
        """Best-effort one-line diagnosis for the blank-frame error. Never raises:
        it only ever decorates an error message that is already being raised."""
        try:
            wake = self.adb.shell("dumpsys power | grep -m1 -o 'mWakefulness=[A-Za-z]*'").strip()
            lock = self.adb.shell("dumpsys trust | grep -m1 -o 'deviceLocked=[01]'").strip()
        except Exception:  # noqa: BLE001 — diagnosis is optional, the raise is not
            return "device state unreadable"
        parts = [p for p in (wake, lock) if p]
        return ", ".join(parts) if parts else "device state unknown"

    def _await_live_frame(self, deadline, should_stop) -> bytes | None:
        """Block until a NON-blank frame is available, for passive observe loops.

        Observe waits on a human with no timeout and sends no touches, so nothing
        keeps the screen alive — it going dark mid-wait is ordinary and recoverable,
        not a run-ending error. We simply keep watching until the owner comes back
        and wakes it. Returns None if `should_stop` fires or `deadline` passes,
        which the callers already handle as "no decision".
        """
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            frame = self._screencap(on_blank="none")
            if frame is not None:
                return frame
            time.sleep(_OBSERVE_POLL_S)
        return None

    # --- debug logging + autonomous-safety (halt on unexpected) --------
    def _snap(self):
        """Screencap for debug/verify, or None when neither is active (skips the overhead)."""
        if self._dbg is None and not self.halt_on_error:
            return None
        for attempt in (1, 2):                         # retry once: a transient screencap blip must
            try:                                       # not silently disable the _verify check below
                return self._screencap()
            except DriverClosed:
                raise                                  # device truly gone -> let it propagate
            except HingeActionError:
                # Blank screen: _screencap already retried and already says exactly what
                # is wrong ("screen is blank (mWakefulness=..., deviceLocked=...)").
                # Re-raise rather than let the generic handler below bury that diagnosis
                # under "screencap failed twice in a row" — or, worse, swallow it to None
                # when halt_on_error is False and hand a caller a missing "before" frame.
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    if self.halt_on_error:
                        # Returning None here would silently defeat halt_on_error:
                        # _verify_progress/_verify_like_landed treat a missing "before"
                        # frame as "skip verification" / an automatic pass — exactly the
                        # failure mode this safety net exists to catch. Two consecutive
                        # screencap failures means the device/link is wedged, not a blip.
                        raise HingeActionError(
                            f"screencap failed twice in a row ({type(exc).__name__}: {exc}); "
                            "halting rather than acting/verifying blind") from exc
                    return None
                time.sleep(human_delay(0.3))

    def _dbg_action(self, name: str, before, **fields) -> None:
        if self._dbg is None:
            return
        try:
            after = self.adb.screencap()   # RAW: best-effort debug capture, never gates the run
        except Exception:  # noqa: BLE001
            after = None
        self._dbg.action(name, before=before, after=after, **fields)

    def snapshot_failure(self, exc: BaseException) -> None:
        """Worker hook: snapshot the on-screen state of an unexpected error into the debug log
        (so the failure is reconstructable) before the worker halts the run."""
        snapshot_failure_frame(self._dbg, exc, self.adb.screencap)

    def _verify_progress(self, before, action: str) -> None:
        """After an autonomous action the screen MUST change (a new card / a confirmation). If it
        doesn't, even after a short settle, something is wrong (missed tap, unknown modal, stuck
        deck) — raise so the worker HALTS and the debug logs are preserved instead of being
        rotated away by continued blind swiping. Only active when halt_on_error is set (in which
        case `before`, sourced from _snap(), is never None — _snap() raises instead)."""
        if not self.halt_on_error:
            return
        if _retry_until(lambda: self._changed(before, self._screencap()), 2, 0.6):
            return
        raise HingeActionError(f"{action} did not change the screen (stuck or unexpected state)")

    def _handle_rose_upsell(self, tries: int = 2) -> bool:
        """Dismiss a paid-upgrade interstitial by tapping its "send anyway"/dismiss control —
        NEVER the paid option itself. Named for Hinge's "Send a Rose instead?" modal (which
        intercepts "Send Like" WHENEVER a Rose is available, since free Roses are granted
        periodically) but used generically by every like flow: we never spend a Rose, a
        SuperSwipe, or any other paid upsell (OWNER RULE — never automated, always manual).
        This is enforced structurally, not just by convention: the only template role this
        method (or _verify_like_landed's late-modal handling) EVER matches against or taps is
        "upsell_dismiss" — there is no coordinate or template anywhere in this driver for a
        paid button, so there is no code path that could tap one even by accident. No-op when
        the modal isn't shown (or this app's spec declares no "upsell_dismiss" template at
        all — _template then returns None and _match_glyph's None-template guard reports no
        hits). Returns True if it dismissed a modal.

        Uses on_blank="none" rather than raising, deliberately: this runs AFTER the
        confirming tap has already landed. A like sent server-side but aborted here
        would never reach worker.py's store.record_decision(), which assumes a raising
        like() "never landed" — so we would silently lose the record of a real like.
        A blank screen here degrades to "no modal seen"; the flow's own verify step is
        the one that still gets to fail loudly.
        """
        template = self._template("upsell_dismiss")

        def _probe():
            frame = self._screencap(on_blank="none")
            if frame is None:
                return []                     # can't see -> assume no modal (see docstring)
            return _match_glyph(frame, template, side="any", threshold=0.6)

        hits = _retry_until(_probe, tries, 0.5)   # modal animates in (only when an upsell is offered)
        if hits:
            self._tap(*hits[0])               # dismiss control — NEVER the paid button above/beside it
            return True
        return False

    def _changed(self, a: bytes, b: bytes) -> bool:
        top, bot = _split_diff(a, b)
        return top >= self.change_threshold or bot >= self.change_threshold

    def _auto_behavior_policy(self):
        """The optional auto-mode behavior policy, or None in observe/legacy callers.

        Worker wiring deliberately owns whether this attribute exists.  Keeping the lookup
        soft preserves the standalone driver and observe paths, including calibration tools
        that construct AndroidDriver directly.
        """
        return getattr(self, "_auto_policy", None)

    def set_auto_session_policy(self, policy) -> None:
        """Attach Worker-owned behavior state for this autonomous session.

        This redesign is calibrated for Hinge only. AndroidDriver also backs an
        experimental Bumble path whose paid-control zones and reading geometry are
        different, so it must retain its legacy plan. Observe mode never calls this
        method and likewise retains calibrated legacy behavior.
        """
        self._auto_policy = policy if self.spec.app == "hinge" else None

    def _sample_read_step(self, depth: int, complexity_hint: float | None):
        """Return one coherent ``(dwell, distance, lane)`` read step.

        The production policy draws all three dimensions together.  The older
        split sampler hooks remain as a defensive compatibility fallback for
        calibration tools and small test doubles.
        """
        policy = self._auto_behavior_policy()
        planner = getattr(policy, "read_step", None) if policy is not None else None
        if callable(planner):
            try:
                step = planner(depth, captured_frames=depth + 1)
                dwell = float(step.dwell_s)
                frac = float(step.fraction)
                x_frac = float(step.x_frac)
                if (math.isfinite(dwell) and dwell >= 0.0
                        and math.isfinite(frac) and 0.10 <= frac <= 0.75
                        and math.isfinite(x_frac) and 0.10 <= x_frac <= 0.90):
                    return dwell, frac, x_frac
            except Exception:  # noqa: BLE001 — sampling may degrade, touching may not
                pass
        return (self._sample_read_dwell(depth, complexity_hint),
                *self._sample_read_scroll())

    def _sample_read_scroll(self) -> tuple[float, float]:
        """Choose one forward-scroll distance/lane, falling back to legacy geometry."""
        policy = self._auto_behavior_policy()
        sampler = getattr(policy, "sample_read_scroll", None) if policy is not None else None
        if callable(sampler):
            try:
                frac, x_frac = sampler(self.read_scroll_frac, len(self._capture_scroll_ledger))
                frac, x_frac = float(frac), float(x_frac)
                # Reject malformed/extreme policy output before it can create an off-screen or
                # near-edge gesture.  Paid-control intersections are still owned by _scroll's
                # forbidden-zone guard; this is only geometry validation.
                if (math.isfinite(frac) and math.isfinite(x_frac)
                        and 0.10 <= frac <= 0.75 and 0.10 <= x_frac <= 0.90):
                    return frac, x_frac
            except Exception:  # noqa: BLE001 — behavior sampling may degrade, touching may not
                pass
        return self.read_scroll_frac, 0.5

    def _sample_read_dwell(self, depth: int, complexity_hint: float | None) -> float:
        """Choose the pause before a read-scroll; policy failures retain legacy pacing."""
        policy = self._auto_behavior_policy()
        sampler = getattr(policy, "read_dwell", None) if policy is not None else None
        if callable(sampler):
            try:
                dwell = float(sampler(self.dwell_s, depth, complexity_hint))
                if math.isfinite(dwell) and dwell >= 0.0:
                    return dwell
            except Exception:  # noqa: BLE001 — behavior sampling may degrade, touching may not
                pass
        return human_delay(self.dwell_s)

    def _capture_limit_for_profile(self) -> int:
        """Per-profile screencap ceiling.

        Observe/legacy behavior remains exactly the configured value.  An auto policy opts
        into a small upward-only variation, so the shipped safety baseline of eight is never
        weakened and profile after profile does not terminate at one identical depth.
        """
        base = self.scroll_captures
        policy = self._auto_behavior_policy()
        if policy is None:
            return base
        sampler = getattr(policy, "capture_limit", None)
        if callable(sampler):
            try:
                sampled = int(sampler(base))
                if base <= sampled <= base + 2:
                    return sampled
            except Exception:  # noqa: BLE001 — keep the capture guard if sampling fails
                pass
        return base + random.randint(0, 2)

    def _scroll_down_one(self, frac: float | None = None,
                         x_frac: float | None = None) -> None:
        """One humanized forward read-scroll, ALWAYS going through here (never a bare
        `touch.scroll_up()` call) so self._capture_scrolls stays the single source of truth
        for "how far down from the last confirmed top are we right now". Both
        _capture_current's read loop and _locate_target_heart's own re-navigation scrolls use
        this — if either called touch.scroll_up() directly instead, _scroll_to_top's ceiling
        would silently under-count again exactly the way the hardcoded-distance bug did."""
        if frac is None or x_frac is None:
            frac, x_frac = self._sample_read_scroll()
        self._scroll(frac, x_frac)
        # Append only after the transport accepted the gesture: a forbidden-zone refusal or
        # transport failure must not leave a fictional scroll for _scroll_to_top to undo.
        self._capture_scroll_ledger.append((frac, x_frac))
        self._capture_scrolls = len(self._capture_scroll_ledger)

    def _scroll_to_top(self) -> None:
        """Swipe the profile back to the top (content down) until it stops moving.

        Two independent things have to be right for this to actually land back at the top,
        and a while ago only one of them was fixed:

          COUNT — how many undo-swipes to throw. Ceiling is self._capture_scrolls: the
          number of forward read-scrolls actually performed since the last confirmed top
          (tracked by _scroll_down_one, the only place that calls touch.scroll_up()) -- not a
          fixed guess, which could not undo a capture that scrolled further than it assumed.

          DISTANCE — how far each undo-swipe travels. This used to be a hardcoded drag
          (0.35h -> 0.80h = 0.45h) while the read-scroll it must cancel travels
          read_scroll_frac of the screen (0.55h by default) -- an 18%-per-scroll shortfall
          that COUNT alone can't fix: matching the swipe COUNT to the scroll count still
          leaves the profile scrolled down by N * (read_scroll_frac - 0.45) after N swipes,
          which compounds on a long profile (up to ~0.7 screen-heights short at the default
          scroll_captures=8). The two fractions must be tied together, not maintained as
          separate magic numbers that can drift apart again. Observe mode therefore mirrors
          read_scroll_frac exactly. Auto mode instead derives a bounded mean from the real
          forward-scroll ledger, then samples broader return strokes and relies on the
          screenshot settle check below rather than replaying one exact reverse per forward.

        _changed() still ends the loop the moment the view settles (its own safety net against
        an animated/video card whose frames never settle); the count is a ceiling, not a
        target. On exit the profile IS at (or past) the top, so the counter resets to 0 -- the
        next forward scroll (whether from a new capture or from _locate_target_heart's search)
        starts counting fresh from a confirmed top.
        """
        w, h = self.adb.screen_size()
        ledger = list(self._capture_scroll_ledger)

        # Compatibility for tests/tools (and an in-flight pre-ledger driver) that only set
        # _capture_scrolls.  Observe mode also keeps its old exact-mirror behavior when there
        # is no auto policy, minimizing risk in the manual-label path.
        if not ledger and self._capture_scrolls:
            ledger = [(self.read_scroll_frac, 0.5)] * self._capture_scrolls
        policy_mode = self._auto_behavior_policy() is not None
        max_swipes = max(1, len(ledger))
        if policy_mode and ledger:
            # A few attempts beyond the recorded count are a hard safety margin for physical
            # under-travel, not a target.  Screenshot settling still ends the loop as soon as
            # the real top is reached.  The bound prevents animated/video cards from spinning.
            max_swipes = len(ledger) + 3
        # Mirror of touch.scroll_up(read_scroll_frac): same y-extents, reversed direction, so
        # one undo-swipe travels exactly as far as one forward read-scroll.
        legacy_frac = self.read_scroll_frac
        mean_forward = (sum(frac for frac, _lane in ledger) / len(ledger)
                        if ledger else legacy_frac)
        for _ in range(max_swipes):
            if policy_mode and ledger:
                # Undo in broader, independently varied strokes.  It intentionally is not a
                # reverse replay of N forward gestures: real people return to the top with a
                # different hand motion/count, while the settle check below remains the source
                # of truth.  Keep enough headroom from screen edges and paid-control zones.
                undo_frac = max(0.30, min(0.72, mean_forward * random.uniform(1.10, 1.28)))
                x_frac = random.uniform(0.38, 0.62)
            else:
                undo_frac = legacy_frac
                x_frac = 0.5
            y_near = int(h * (0.5 - undo_frac / 2))
            y_far = int(h * (0.5 + undo_frac / 2))
            x = int(w * x_frac)
            before = self._screencap()
            self._swipe(x, y_near, x, y_far)
            time.sleep(human_delay(0.3))
            if not self._changed(before, self._screencap()):
                break
        self._capture_scrolls = 0
        self._capture_scroll_ledger = []   # confirmed (or ceiling-bounded) back at top

    # --- capture (Signals #1: read the whole profile, human-paced) ------
    def _capture_current(self) -> Profile | None:
        photos: list[bytes] = []
        self._current_sigs = []
        self._capture_scrolls = 0     # reset: _scroll_to_top must undo THIS capture, not a stale one
        self._capture_scroll_ledger = []
        self._profile_capture_limit = self._capture_limit_for_profile()
        read_dwell_s_total = 0.0
        seen = set()
        for i in range(self._profile_capture_limit):
            frame = self._screencap()
            ds = _downsample(frame)                    # None if PIL/numpy unavailable or undecodable
            sig = _frame_sig(frame)
            if sig in seen:
                # A repeat frame = reached the bottom (the screen stopped changing). Only treat
                # the very FIRST scroll repeating as a static screen (Out of Profiles / Loading)
                # when the frame actually DECODES. An undecodable repeat is a degraded/wedged
                # capture (e.g. a dropped device returning empty/truncated screencap on exit 0) —
                # it must fall through to the H1 guard below, not masquerade as an empty deck
                # (which would silently livelock the observe worker instead of stopping cleanly).
                if i == 1 and ds is not None:
                    return None
                break
            seen.add(sig)
            photos.append(frame)
            # Keep _current_sigs index-ALIGNED with photos: append ds even when None (an
            # undecodable frame). The opener's referenced_index indexes photos, and
            # _locate_target_heart looks it up here — a gap would desync them and target the
            # WRONG photo. Consumers below filter/guard the Nones.
            self._current_sigs.append(ds)

            if i < self._profile_capture_limit - 1:
                complexity_hint = None
                if ds is not None:
                    try:
                        complexity_hint = float(ds.std()) / 255.0
                    except Exception:  # noqa: BLE001 — hint is optional, capture is not
                        pass
                dwell, frac, x_frac = self._sample_read_step(i, complexity_hint)
                time.sleep(dwell)
                read_dwell_s_total += dwell
                self._scroll_down_one(frac, x_frac)
        # H1: in a real run (open_session validated PIL/numpy), every frame should
        # downsample. If none did, decode is broken at runtime (PIL/numpy failure OR a wedged
        # device returning empty/truncated screencap) — refuse to continue in a degraded mode
        # where scroll-detection is off and manual scrolls mislabel as PASS / the worker
        # silently no-ops forever. (The decodable-static case returned None above.)
        if self._observe_ready and photos and not any(s is not None for s in self._current_sigs):
            raise DriverClosed(
                "screencap frames could not be decoded (PIL/numpy runtime failure or wedged "
                "device); refusing to run degraded — it would corrupt training labels")
        if self._dbg is not None and photos:
            self._dbg.action("capture", before=photos[0], photos=len(photos))   # first frame = who was scored
        return Profile(
            photos=photos,
            prompts=[],
            meta={
                "app": self.spec.app,
                "capture_frames": len(photos),
                "read_scrolls": len(self._capture_scroll_ledger),
                "read_dwell_s_total": read_dwell_s_total,
            },
        )

    def next_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def current_profile(self) -> Profile | None:
        # Observe mode's only capture path: worker.py prints "READY - swipe this profile"
        # right after this returns, so the phone must be back at the top for that swipe to
        # land on the card the operator actually read (bug 2). next_profile() (auto) does NOT
        # get this: like() already calls _scroll_to_top() itself before acting, so adding it
        # here too would just be a redundant extra scroll on that path.
        if self.out_of_profiles():
            return None
        profile = self._capture_current()
        if profile is not None:
            self._scroll_to_top()
        return profile

    def out_of_profiles(self) -> bool:
        # LIVE-VERIFY: the empty-deck screen is gated until the profile is finished,
        # so we can't yet match its signature. Until calibrated this returns False
        # and the run stops via the rate limiter (auto) or the operator (observe).
        return False

    def _locate_target_heart(self, item_index: int):
        """comment_sheet flow only. Locate the heart of the photo/prompt the opener is about.
        item_index is the 0-based index (capture order) the opener returned. We re-navigate to
        that captured frame by matching its downsample signature, then take its heart. Falls
        back to the topmost heart (first photo) when targeting isn't possible (index 0, no
        sigs, no match) — so it is never worse than the old 'always first photo' behavior.
        Every genuine fallback (as opposed to the ordinary index-0 fast path) is recorded via
        _dbg_action so a bug report can tell "target not found" apart from "no sigs to search
        at all" (HINGE-05). (LIVE-VERIFY during observe seeding.)"""
        sigs = getattr(self, "_current_sigs", None)
        if not sigs or item_index <= 0 or item_index >= len(sigs) or sigs[item_index] is None:
            if sigs and item_index > 0:                # a genuine fallback, not the index-0 fast path
                self._dbg_action("locate_target_heart", self._snap(), item_index=item_index,
                                  outcome="fallback", reason="out_of_range_or_undecodable_target")
            return self._await_button("like")         # out of range / undecodable target -> first photo
        import numpy as np
        target = sigs[item_index]
        before = self._snap()
        # like() always _scroll_to_top()s right before calling this, so item_index (a capture-
        # order index counted down from the top) should need about that many scroll_up()s to
        # reach. Cap the search there (+ slack) instead of sweeping the whole scroll_captures
        # depth: on a real device a scroll can over/undershoot the intended frame, and without a
        # cap a target we've scrolled past costs a full wasted sweep before the fallback below
        # even starts (HINGE-05).
        tries = min(self._profile_capture_limit + 1, item_index + 3)
        matched_frame_no_heart = False
        for _ in range(tries):
            frame = self._screencap()
            ds = _downsample(frame)
            if ds is not None and float(np.mean(np.abs(ds - target))) < self.change_threshold:
                hearts = _match_glyph(frame, self._template("like"), side="right")
                if hearts:
                    return hearts[0]                  # the referenced item's heart, now in view
                matched_frame_no_heart = True
                break
            self._scroll_down_one()          # tracked, so the fallback _scroll_to_top() below (if it
            time.sleep(human_delay(self.dwell_s * 0.4))  # comes to that) undoes exactly these scrolls too
        reason = "heart_not_visible_on_matched_frame" if matched_frame_no_heart else "target_frame_not_found"
        self._dbg_action("locate_target_heart", before, item_index=item_index,
                          outcome="fallback", reason=reason)
        self._scroll_to_top()                         # no match: reset and take the first photo
        time.sleep(human_delay(0.3))
        return self._await_button("like")

    def _verify_like_landed(self, before) -> None:
        """comment_sheet flow only. A like is COMPLETE only when the comment sheet AND any
        paid-upsell modal are gone AND the deck has moved off the pre-tap card. If the sheet is
        still up (missed Send Like tap) or the screen never changed (missed heart tap), raise so
        the worker HALTS instead of counting a like that never sent. A paid-upsell modal that
        animates in LATE — after _handle_rose_upsell's own poll window already gave up and
        returned False — is tolerated here rather than treated as a dead run: we dismiss it
        ourselves (same rule: never the paid option) and keep checking (HINGE-07). Only a
        modal/sheet that genuinely won't clear, or a deck that never advances, still raises — an
        unsent like must never be mislabelled as sent (it would corrupt the taste model). Only
        active under halt_on_error. Unlike a bare change-check, the scroll-to-top can't spoof
        this."""
        if not self.halt_on_error:
            return
        sheet_up = modal_up = False
        for _ in range(3):        # a couple of extra passes tolerate a late-animating upsell modal
            frame = self._screencap()
            sheet_up = bool(_match_glyph(frame, self._template("confirm"), side="any", threshold=0.6))
            modal_hits = _match_glyph(frame, self._template("upsell_dismiss"), side="any", threshold=0.6)
            modal_up = bool(modal_hits)
            if not sheet_up and not modal_up and (before is None or self._changed(before, frame)):
                return                                # sheet/modal closed AND advanced -> sent
            if modal_up:
                self._tap(*modal_hits[0])             # late-animating upsell -> dismiss, never the paid option
            time.sleep(human_delay(0.6))
        if sheet_up or modal_up:
            raise HingeActionError("like did not complete — the like sheet / upsell modal is still open")
        raise HingeActionError("like did not change the screen (missed tap or stuck)")

    # --- actions (NORMAL like only — never a paid upgrade) ----------------------
    def like(self, opener: str | None = None, item_index: int = 0) -> None:
        if self.spec.like_flow == "comment_sheet":
            self._like_comment_sheet(opener, item_index)
        else:
            self._like_direct(opener, item_index)

    def _like_comment_sheet(self, opener: str | None, item_index: int) -> None:
        """Hinge's flow: heart -> comment/"Send Like" sheet opens -> optionally type the
        opener into the comment box (Signals #2: the opener is sent WITH the like) -> tap
        Send -> handle a paid-upsell interstitial (never tap the paid option) -> verify."""
        self._scroll_to_top()
        time.sleep(human_delay(0.4))
        heart = self._locate_target_heart(item_index)  # heart of the photo the opener is about
        before = self._snap()                         # baseline AFTER navigation: the pre-tap card
        self._tap(*heart)                             # opens the comment / "Send Like" sheet
        time.sleep(human_cooldown(0.8))               # sheet animates in; you read/think
        self._await_sheet_open()                      # gate: the fixed taps below are only
                                                       # valid while the sheet is actually up
        if opener:
            self._tap_frac(self.coords["comment_box"])
            time.sleep(human_delay(0.5))
            self.adb.text(opener)                     # opener sent WITH the like (Signals #2)
            time.sleep(human_delay(0.6))
        self._tap_frac(self.coords["send_like"])
        time.sleep(human_cooldown(0.6))               # let the send register / upsell modal animate in
        rose = self._handle_rose_upsell()             # paid-upsell interstitial: dismiss, NEVER pay
        self._dbg_action("like", before, heart=list(heart), opener_chars=len(opener or ""), rose_modal=rose)
        self._verify_like_landed(before)

    def _deliver_decision(self, decision: str):
        """Issue one like/pass, by whichever gesture this app's spec calls for.

        Returns the tapped point, or None when the decision was delivered as a card drag
        (there is no single point to log in that case). Both paths are equally humanized;
        they differ only in what the phone receives, and therefore in what can go wrong:
        a tap can land on a neighbouring control, a drag cannot."""
        if self.spec.decide_gesture == "card_swipe":
            self._decide_by_card_swipe(decision)
            return None
        point = self._await_button("like" if decision == "like" else "pass")
        self._tap(*point)
        return point

    def _like_direct(self, opener: str | None, item_index: int) -> None:
        """Bumble's flow: one like, no comment sheet, no per-item targeting.
        `opener`/`item_index` are accepted only for interface parity with the comment_sheet
        flow and are otherwise unused — accepts_opener is False for every spec using this
        flow (Bumble is match-first-then-message, so there is no swipe-time opener to attach),
        so worker.py never actually passes a real `opener` here."""
        before = self._snap()
        like_btn = self._deliver_decision("like")
        time.sleep(human_cooldown(0.6))                # let it register / an upsell modal animate in
        upsell = self._handle_rose_upsell()             # paid-upsell interstitial: dismiss, NEVER pay
        self._dbg_action("like", before, like=list(like_btn) if like_btn else None,
                         gesture=self.spec.decide_gesture, opener_chars=0, upsell_dismissed=upsell)
        self._verify_progress(before, "like")

    def dislike(self) -> None:
        before = self._snap()                         # snapped immediately before acting (no scroll between)
        x = self._deliver_decision("pass")            # vision-located X, or a card drag per spec
        self._dbg_action("dislike", before, x=list(x) if x else None,
                         gesture=self.spec.decide_gesture)
        self._verify_progress(before, "dislike")

    # --- observe mode (shadow learning) --------------------------------
    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None) -> bool | None:
        """Block until you manually like/pass the current card, inferred from
        screencap deltas (no accessibility tree):

          LIKE — tapping a heart slides the comment / "Send Like" sheet up over the
                 BOTTOM while the photo stays up top (bottom changes, top doesn't);
                 we then wait for it to resolve to a new card -> True.
          PASS — the whole card advances to a new profile (top changes) -> False.
          none — stop requested, deck empty, or timeout -> None.

        ⚠️ LIVE-VERIFY: the like-sheet geometry is gated until the profile is
        finished, so the top/bottom thresholds must be confirmed on-device before
        trusting observe labels. After "READY", swipe (don't keep scrolling) — a
        manual scroll changes the top region and reads as a pass.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        base = self._await_live_frame(deadline, should_stop)
        if base is None:
            return None
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            cur = self._screencap(on_blank="none")
            if cur is None:                           # screen asleep: the owner stepped away.
                time.sleep(_OBSERVE_POLL_S)           # keep watching — do NOT diff a black frame
                continue                              # against `base` (that reads as a phantom pass)
            top, bot = _split_diff(base, cur)
            if top < self.change_threshold and bot < self.change_threshold:
                time.sleep(_OBSERVE_POLL_S)
                continue

            # A LIKE opens the comment / "Send Like" sheet: the photo (top half) stays put
            # while the BOTTOM changes. Detect this BEFORE the scroll check — otherwise, on a
            # light/gray profile, a like-sheet frame can match a stored full-frame signature
            # and be mis-read as a scroll, silently dropping the LIKE (bug C1).
            if bot >= self.change_threshold and top < self.change_threshold:
                sent = self._await_like_resolved(base, deadline, should_stop)
                if sent is None:
                    return None
                if sent:
                    return True                       # like sheet resolved to a new card
                base = self._await_live_frame(deadline, should_stop)   # cancelled -> resync
                if base is None:
                    return None
                continue

            # Top changed (whole card moved): a manual SCROLL of the SAME profile (matches a
            # captured frame) -> ignore; a genuinely new profile -> pass.
            ds_cur = _downsample(cur)
            seen_sigs = [s for s in getattr(self, "_current_sigs", None) or [] if s is not None]
            if ds_cur is not None and seen_sigs:
                import numpy as np
                dists = [float(np.mean(np.abs(ds_cur - ds_seen))) for ds_seen in seen_sigs]
                if dists and min(dists) < self.change_threshold:
                    base = cur                        # it's a scroll -> keep waiting
                    time.sleep(_OBSERVE_POLL_S)
                    continue
            return False                              # whole card changed -> pass
        return None

    def _await_like_resolved(self, base: bytes, deadline, should_stop) -> bool | None:
        """After the like sheet appears, wait for it to close: a NEW card means the
        like was sent (True); reverting to `base` means it was cancelled (False)."""
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            cur = self._screencap(on_blank="none")
            if cur is None:                           # screen asleep mid-wait: keep watching
                time.sleep(_OBSERVE_POLL_S)           # (never diff a black frame against base)
                continue
            top, bot = _split_diff(base, cur)
            if top >= self.change_threshold:
                # Wait 0.5s to verify the transition has stabilized
                time.sleep(0.5)
                confirm = self._screencap(on_blank="none")
                if confirm is None:
                    continue                          # can't confirm blind -> re-poll
                c_top, c_bot = _split_diff(base, confirm)
                if c_top >= self.change_threshold:
                    if self._is_current_profile_frame(confirm):
                        return False                  # same profile, scrolled (uniform top) -> not a like (#6)
                    return True                       # advanced to a NEW profile -> like sent
            if bot < self.change_threshold:
                # Wait 0.5s to verify if it's a temporary transition or a genuine cancellation
                time.sleep(0.5)
                confirm = self._screencap(on_blank="none")
                if confirm is None:
                    continue                          # can't confirm blind -> re-poll
                c_top, c_bot = _split_diff(base, confirm)
                if c_top >= self.change_threshold:
                    if self._is_current_profile_frame(confirm):
                        return False                  # same profile, scrolled -> not a like (#6)
                    return True                       # actually advanced to a NEW profile
                if c_bot < self.change_threshold:
                    return False                      # back to the original card -> cancelled
            time.sleep(_OBSERVE_POLL_S)               # sheet still up (bottom busy, top calm)
        return False

    def _is_current_profile_frame(self, frame: bytes) -> bool:
        """True if `frame` matches a captured frame of the CURRENT profile — i.e. an
        'advance' that is really a scroll within the same profile, not a new card.
        Guards the C1 reorder against a uniform-top scroll reading as a LIKE (#6).
        Conservatively False when undecodable (treat as a genuine advance).

        ⚠️ LIVE-VERIFY (gated): this is the SOLE like-vs-scroll discriminator, and it uses
        min-over-ALL captured sigs at change_threshold. Dating first-photos are visually
        similar (centred face, light background), so a genuinely NEW card can collide with
        one of the current profile's sigs — a real LIKE then mis-reads as a scroll, the like
        is dropped, and the observe loop desyncs (the next decision is attributed to the wrong
        profile's photos). Audit-flagged. The fix (match only the top-most sig; or require a
        corroborating 'sheet was present' signal; or bias toward LIKE on a near-miss) needs
        real like-sheet vs scroll frames to calibrate, so it's deferred to live-finalize."""
        ds = _downsample(frame)
        sigs = [s for s in getattr(self, "_current_sigs", None) or [] if s is not None]
        if ds is None or not sigs:
            return False
        import numpy as np
        return min(float(np.mean(np.abs(ds - s))) for s in sigs) < self.change_threshold


class HingeDriver(AndroidDriver):
    """Thin binding: AndroidDriver + HINGE_SPEC. All behavior lives in AndroidDriver above;
    this class exists so `operation_love.drivers.hinge.HingeDriver(cfg)` keeps working exactly
    as it always has (driver factory, tools/hinge_inspect.py, the test suite)."""

    def __init__(self, cfg):
        super().__init__(cfg, HINGE_SPEC)
