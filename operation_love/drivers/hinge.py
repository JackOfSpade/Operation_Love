"""Hinge driver — physical Android phone over HOST-SIDE ADB (no on-device helper).

Rewritten off the old uiautomator2/emulator path: that installed an on-device
server (atx-agent + the uiautomator2 APK) which Play Integrity can flag and which
ops/HINGE-PIXEL-RUNBOOK.md §5 forbids as the main account-protection guardrail.
This driver instead talks to a genuine, stock, physical Pixel through the
host-side `Adb` transport only:

  * perception  — `adb exec-out screencap` frames, deduped by a downsampled
                  signature (no accessibility tree, no resource-ids)
  * action      — humanized `input motionevent` taps/swipes (curved, jittered,
                  log-normal timing) via Adb

Hinge lets you like a specific photo/prompt WITH a comment, so the opener is sent
at like-time (Signals behavior #2). Reading the whole profile slowly before
deciding (the dwell in `_capture_current`) is Signals behavior #1. We only ever
send a NORMAL like — never a Rose (Hinge's super-like); Roses/boosts are manual.

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
import time
from pathlib import Path

from ..human import human_cooldown, human_delay
from ..perception.capture import Profile
from .adb import Adb, AdbError
from .base import DatingAppDriver, DriverClosed, open_debug_log, snapshot_failure_frame
from .uhid import UhidTouch, UhidUnavailable

_ASSETS = Path(__file__).parent / "assets"


class HingeActionError(RuntimeError):
    """An autonomous action did not produce the expected on-screen change (stuck deck, missed
    tap, or an unknown screen). NOT a DriverClosed (which is a clean, restart-safe stop): this
    is unexpected, so the worker halts the run and preserves the debug logs."""

_OBSERVE_POLL_S = 0.35     # internal sampling cadence for your manual tap (not app-facing)

DEFAULTS = {
    "package": "co.hinge.app",
    "scroll_captures": 8,          # max screencaps while reading one profile
    "dwell_s": 1.1,                # per-card read dwell (humanized) — Signals behavior #1
    "read_scroll_frac": 0.55,      # how far each read-scroll advances the profile
    # Action points as FRACTIONS of the screen (x, y in 0..1). Calibrated live on the Pixel 7a
    # (1080x2400) 2026-06-27, config-overridable. like_heart / pass_x are VISION-located at
    # runtime (template-matched glyph) — these are only the FALLBACK if vision can't find the
    # glyph. comment_box / send_like are fixed (the like-sheet layout is consistent).
    "coords": {
        "like_heart": [0.868, 0.667],   # FALLBACK only — heart vision-located on the first photo
        "pass_x": [0.116, 0.848],       # FALLBACK only — X vision-located (floating, bottom-left)
        "comment_box": [0.500, 0.529],  # comment field in the like sheet
        "send_like": [0.643, 0.576],    # "Send Like" button (kept clear of the 🌷Rose button)
    },
    # Mean abs grayscale delta (0..255) on a 24x24 downsample to call a region "changed".
    "change_threshold": 9.0,
}


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


# --- vision: locate the like/pass action BUTTONS by their glyph (not a fixed coord) -----
# Hinge's like-heart sits at the bottom-right of EACH photo and the pass-X floats bottom-left.
# A fixed fraction is unreliable: the "Start sending likes" banner and per-profile photo
# aspect ratios shift the heart vertically, and the X's white disc merges into the white
# background of a prompt card (so a plain white-blob detector loses it). We instead template-
# match the dark glyph (heart / X), which stays distinct on ANY background. The glyphs are
# fixed-resolution UI assets (this driver targets one device — the Pixel 7a at 1080x2400),
# so matches are essentially exact.

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


def _retry_until(check_fn, tries: int, delay_s: float):
    """Call check_fn() up to `tries` times, sleeping human_delay(delay_s) after each
    falsy attempt. Returns the first truthy result, or None once attempts run out.
    The shared shape behind _await_button/_verify_progress/_handle_rose_upsell: poll
    something on-screen with humanized pacing until it appears or we give up."""
    for _ in range(max(1, tries)):
        result = check_fn()
        if result:
            return result
        time.sleep(human_delay(delay_s))
    return None


class HingeDriver(DatingAppDriver):
    accepts_opener = True   # Hinge sends the opener as a comment at like-time

    def __init__(self, cfg):
        app_cfg = (getattr(cfg, "apps", {}) or {}).get("hinge", {})
        self.serial = app_cfg.get("serial") or None
        self.adb_path = app_cfg.get("adb_path", "adb")
        self.package = app_cfg.get("package", DEFAULTS["package"])
        self.scroll_captures = max(1, int(app_cfg.get("scroll_captures", DEFAULTS["scroll_captures"])))
        self.dwell_s = float(app_cfg.get("dwell_s", DEFAULTS["dwell_s"]))
        self.read_scroll_frac = float(app_cfg.get("read_scroll_frac", DEFAULTS["read_scroll_frac"]))
        self.change_threshold = float(app_cfg.get("change_threshold", DEFAULTS["change_threshold"]))
        self.coords = {**DEFAULTS["coords"], **(app_cfg.get("coords") or {})}
        self._adb: Adb | None = None
        self._touch = None            # touch transport: UhidTouch (genuine) or Adb (input fallback)
        self.touch_backend = app_cfg.get("touch_backend", "auto")   # auto | uhid | adb
        self._observe_ready = False   # True once open_session validated PIL/numpy + device
        self.debug_log = bool(app_cfg.get("debug_log", False))
        self.debug_dir = app_cfg.get("debug_dir", "./data/hinge_debug")
        self.halt_on_error = bool(app_cfg.get("halt_on_error", True))   # auto: STOP on unexpected (preserve logs)
        self._dbg = None              # HingeDebugLog (set in open_session when debug_log is on)

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        import importlib.util
        if not (importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL")):
            raise DriverClosed("Hinge driver requires PIL and numpy. Install them via `pip install -e '.[ml]'` or `pip install pillow numpy`.")

        self._adb = Adb(self.serial, adb_path=self.adb_path)
        ready = self._adb.devices()
        if not ready:
            raise DriverClosed("No ADB device connected for Hinge")
        if self.serial and self.serial not in ready:
            raise DriverClosed(f"Hinge device {self.serial} not connected (adb devices: {ready})")
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
            raise DriverClosed("Hinge session is not open")
        return self._adb

    @property
    def touch(self):
        if self._touch is None:
            raise DriverClosed("Hinge session is not open")
        return self._touch

    def _make_touch(self):
        """Touch transport: prefer the genuine UHID virtual touchscreen; fall back to the
        adb `input` transport if UHID is unavailable. touch_backend forces a choice."""
        if self.touch_backend == "adb":
            return self._adb
        try:
            t = UhidTouch(self._adb)
            t.open()
            return t
        except (UhidUnavailable, AdbError) as exc:    # probe/geometry error -> fall back, don't escape
            if self.touch_backend == "uhid":
                raise DriverClosed(f"UHID touch required but unavailable: {exc}") from exc
            print(f"Hinge: UHID touch unavailable ({exc}); using adb input transport.")
            return self._adb

    def _tap_frac(self, frac) -> None:
        w, h = self.adb.screen_size()
        self.touch.tap(int(frac[0] * w), int(frac[1] * h))

    def _locate_button(self, which: str):
        """Screencap and template-match the like-heart ('like' -> topmost right glyph) or the
        floating pass-X ('pass' -> left glyph). Returns (x, y) or None when not visible."""
        template = _load_template("hinge_heart.png" if which == "like" else "hinge_pass_x.png")
        centers = _match_glyph(self.adb.screencap(), template,
                               side="right" if which == "like" else "left")
        return centers[0] if centers else None

    def _await_button(self, which: str, tries: int = 5):
        """Locate a button, retrying through short settle waits — Hinge fades the floating
        like/pass buttons out DURING a scroll and back in once it settles, so a tap fired
        immediately after a scroll-read can miss. Falls back to the calibrated fixed coord if
        vision can't find it (degraded, but better than not acting)."""
        pt = _retry_until(lambda: self._locate_button(which), tries, 0.4)
        if pt is not None:
            return pt
        frac = self.coords["like_heart" if which == "like" else "pass_x"]
        w, h = self.adb.screen_size()
        print(f"Hinge: {which} button not found by vision; using fallback coord {frac}")
        return (int(frac[0] * w), int(frac[1] * h))

    # --- debug logging + autonomous-safety (halt on unexpected) --------
    def _snap(self):
        """Screencap for debug/verify, or None when neither is active (skips the overhead)."""
        if self._dbg is None and not self.halt_on_error:
            return None
        for attempt in (1, 2):                         # retry once: a transient screencap blip must
            try:                                       # not silently disable the _verify check below
                return self.adb.screencap()
            except DriverClosed:
                raise                                  # device truly gone -> let it propagate
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
            after = self.adb.screencap()
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
        rotated away by continued blind swiping. Only active when halt_on_error is set."""
        if not self.halt_on_error or before is None:
            return
        if _retry_until(lambda: self._changed(before, self.adb.screencap()), 2, 0.6):
            return
        raise HingeActionError(f"{action} did not change the screen (stuck or unexpected state)")

    def _handle_rose_upsell(self, tries: int = 2) -> bool:
        """Hinge intercepts "Send Like" with a "Send a Rose instead?" modal WHENEVER a Rose is
        available (free Roses are granted periodically). We NEVER send a Rose (owner rule), so we
        tap "Send Like anyway" to send the NORMAL like. Vision-matched (the modal's text), so we
        never risk the "Send a Rose" button sitting just above it. No-op when the modal isn't
        shown. Returns True if it dismissed the modal."""
        template = _load_template("hinge_send_like_anyway.png")
        hits = _retry_until(
            lambda: _match_glyph(self.adb.screencap(), template, side="any", threshold=0.6),
            tries, 0.5)                       # modal animates in (only when a Rose is available)
        if hits:
            self.touch.tap(*hits[0])          # "Send Like anyway" — NEVER the Rose button above it
            return True
        return False

    def _changed(self, a: bytes, b: bytes) -> bool:
        top, bot = _split_diff(a, b)
        return top >= self.change_threshold or bot >= self.change_threshold

    def _scroll_to_top(self) -> None:
        """Swipe the profile back to the top (content down) until it stops moving."""
        w, h = self.adb.screen_size()
        max_swipes = min(4, self.scroll_captures)
        for _ in range(max_swipes):
            before = self.adb.screencap()
            self.touch.swipe(w // 2, int(h * 0.35), w // 2, int(h * 0.80))
            time.sleep(human_delay(0.3))
            if not self._changed(before, self.adb.screencap()):
                break

    # --- capture (Signals #1: read the whole profile, human-paced) ------
    def _capture_current(self) -> Profile | None:
        photos: list[bytes] = []
        self._current_sigs = []
        seen = set()
        for i in range(self.scroll_captures):
            frame = self.adb.screencap()
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

            if i < self.scroll_captures - 1:
                time.sleep(human_delay(self.dwell_s))  # actually read this card before scrolling
                self.touch.scroll_up(self.read_scroll_frac)
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
        return Profile(photos=photos, prompts=[], meta={"app": "hinge"})

    def next_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def current_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def out_of_profiles(self) -> bool:
        # LIVE-VERIFY: the empty-deck screen is gated until the profile is finished,
        # so we can't yet match its signature. Until calibrated this returns False
        # and the run stops via the rate limiter (auto) or the operator (observe).
        return False

    def _locate_target_heart(self, item_index: int):
        """Locate the heart of the photo/prompt the opener is about. item_index is the 0-based
        index (capture order) the opener returned. We re-navigate to that captured frame by
        matching its downsample signature, then take its heart. Falls back to the topmost heart
        (first photo) when targeting isn't possible (index 0, no sigs, no match) — so it is never
        worse than the old 'always first photo' behavior. (LIVE-VERIFY during observe seeding.)"""
        sigs = getattr(self, "_current_sigs", None)
        if not sigs or item_index <= 0 or item_index >= len(sigs) or sigs[item_index] is None:
            return self._await_button("like")         # out of range / undecodable target -> first photo
        import numpy as np
        target = sigs[item_index]
        for _ in range(self.scroll_captures + 1):
            frame = self.adb.screencap()
            ds = _downsample(frame)
            if ds is not None and float(np.mean(np.abs(ds - target))) < self.change_threshold:
                hearts = _match_glyph(frame, _load_template("hinge_heart.png"), side="right")
                if hearts:
                    return hearts[0]                  # the referenced item's heart, now in view
                break
            self.touch.scroll_up(self.read_scroll_frac)
            time.sleep(human_delay(self.dwell_s * 0.4))
        self._scroll_to_top()                         # no match: reset and take the first photo
        time.sleep(human_delay(0.3))
        return self._await_button("like")

    def _verify_like_landed(self, before) -> None:
        """A like is COMPLETE only when the comment sheet AND the Rose modal are gone AND the deck
        has moved off the pre-tap card. If a sheet/modal is still up (missed Send Like tap, or a
        Rose modal that out-raced _handle_rose_upsell) or the screen never changed (missed heart
        tap), raise so the worker HALTS instead of counting a like that never sent. Only active
        under halt_on_error. Unlike a bare change-check, the scroll-to-top can't spoof this."""
        if not self.halt_on_error:
            return
        sheet_up = modal_up = False
        for _ in range(2):
            frame = self.adb.screencap()
            sheet_up = bool(_match_glyph(frame, _load_template("hinge_send_like.png"), side="any", threshold=0.6))
            modal_up = bool(_match_glyph(frame, _load_template("hinge_send_like_anyway.png"), side="any", threshold=0.6))
            if not sheet_up and not modal_up and (before is None or self._changed(before, frame)):
                return                                # sheet/modal closed AND advanced -> sent
            time.sleep(human_delay(0.6))
        if sheet_up or modal_up:
            raise HingeActionError("like did not complete — the like sheet / Rose modal is still open")
        raise HingeActionError("like did not change the screen (missed tap or stuck)")

    # --- actions (NORMAL like only — never a Rose) ----------------------
    def like(self, opener: str | None = None, item_index: int = 0) -> None:
        self._scroll_to_top()
        time.sleep(human_delay(0.4))
        heart = self._locate_target_heart(item_index)  # heart of the photo the opener is about
        before = self._snap()                         # baseline AFTER navigation: the pre-tap card
        self.touch.tap(*heart)                        # opens the comment / "Send Like" sheet
        time.sleep(human_cooldown(0.8))               # sheet animates in; you read/think
        if opener:
            self._tap_frac(self.coords["comment_box"])
            time.sleep(human_delay(0.5))
            self.adb.text(opener)                     # opener sent WITH the like (Signals #2)
            time.sleep(human_delay(0.6))
        self._tap_frac(self.coords["send_like"])
        time.sleep(human_cooldown(0.6))               # let the send register / Rose modal animate in
        rose = self._handle_rose_upsell()             # if a Rose is available: "Send Like anyway", never a Rose
        self._dbg_action("like", before, heart=list(heart), opener_chars=len(opener or ""), rose_modal=rose)
        self._verify_like_landed(before)

    def dislike(self) -> None:
        before = self._snap()                         # snapped immediately before the tap (no scroll between)
        x = self._await_button("pass")                # vision-located floating X (presence-checked)
        self.touch.tap(*x)
        self._dbg_action("dislike", before, x=list(x))
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
        base = self.adb.screencap()
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            cur = self.adb.screencap()
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
                base = self.adb.screencap()           # cancelled -> resync, keep watching
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
            cur = self.adb.screencap()
            top, bot = _split_diff(base, cur)
            if top >= self.change_threshold:
                # Wait 0.5s to verify the transition has stabilized
                time.sleep(0.5)
                confirm = self.adb.screencap()
                c_top, c_bot = _split_diff(base, confirm)
                if c_top >= self.change_threshold:
                    if self._is_current_profile_frame(confirm):
                        return False                  # same profile, scrolled (uniform top) -> not a like (#6)
                    return True                       # advanced to a NEW profile -> like sent
            if bot < self.change_threshold:
                # Wait 0.5s to verify if it's a temporary transition or a genuine cancellation
                time.sleep(0.5)
                confirm = self.adb.screencap()
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
