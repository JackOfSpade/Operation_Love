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

import dataclasses
import difflib
import functools
import hashlib
import math
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from ..human import human_cooldown, human_delay
from ..human_motion import tap_jitter_margin_px
from ..perception.capture import Profile
from .adb import SCROLL_X_JITTER_PX, Adb, AdbError, clamp_xy
from .android_spec import AndroidAppSpec
from .base import DatingAppDriver, DriverClosed, open_debug_log, snapshot_failure_frame
from .touchwatch import TouchWatcher, TouchWatchUnavailable
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


class OutOfRangeTapError(HingeActionError):
    """A touch-down coordinate resolved to a fraction of the screen outside 0..1 -- always a
    configuration or logic error (a bad coords/*_frac value in config.yaml or an
    AndroidAppSpec, or a computed offset that pushed a touch off-screen), never a
    legitimate touch.

    This exists because BOTH real touch transports CLAMP whatever coordinate they are
    handed onto the live screen instead of raising (uhid.py's _report, Adb._clamp -- see
    clamp_xy in adb.py, which both now share). An out-of-range value therefore doesn't fail
    on a real device: it silently lands on a screen EDGE -- and that edge can be INSIDE a
    forbidden zone despite the raw value never having fallen inside one, because
    forbidden_zones' own 0..1 range check has nothing to compare an out-of-range fraction
    against. Demonstrated live: Bumble's SuperSwipe zone (0.34, 0.80, 0.66, 1.00) -- an aim
    fraction of (0.50, 1.05) clamps to pixel (538.9, 2399.0) = fraction (0.4989, 0.9996),
    inside the zone. Also reachable with no coords tuple at all: apps.bumble.read_scroll_frac
    set to 1.30 puts an ordinary read-scroll's touch-down at fraction 1.15, which clamps the
    same way.

    Deliberately a HingeActionError (not a bare RuntimeError): this must halt the run and
    preserve the debug logs like any other unexpected-safety condition, never be treated as
    a routine, retryable miss."""


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


class UnconfirmedScreenError(HingeActionError):
    """A decide gesture (like/pass) was refused because the expected swipe-deck screen could
    not be positively confirmed before acting.

    Why this exists, distinct from ForbiddenTapError: forbidden_zones is a SCREEN-AGNOSTIC
    rectangle — it forbids a coordinate no matter what is actually on screen. But the danger
    some apps pose is SCREEN-DEPENDENT, and a static rectangle cannot express that. Measured
    live on the device 2026-08-10: Bumble's like_heart (0.850, 0.900) and pass_x (0.150,
    0.900) fallback coordinates are ordinary, harmless points on the swipe deck — and BOTH
    sit ON the "Get 30 SuperSwipes for $39.99" purchase button when the SuperSwipe purchase
    sheet is up instead (that sheet's CTA spans x 0.049-0.950, y 0.899-0.951 — see
    BUMBLE_SPEC's comment block for the full measurement). Both coordinates are also OUTSIDE
    BUMBLE_SPEC.forbidden_zones (0.34, 0.80, 0.66, 1.00), which only ever covers the middle
    third: no rectangle drawn to guard the deck's paid button can also cover the deck's own
    like/pass controls, because on the purchase sheet those are the SAME pixels. Widening
    forbidden_zones to also catch (0.850, 0.900) would make an ordinary like impossible on
    the very deck it exists to protect — see this module's HINGE_SPEC/BUMBLE_SPEC comments
    and ops/RUNBOOK.md for why that path was rejected.

    The guard that actually generalises asks a different question before every decide
    gesture: not "is this point forbidden" but "am I actually looking at the deck". A spec
    that declares BOTH a 'like' and a 'pass' glyph template can answer that — the same
    perception _observe_deck_ready already uses PASSIVELY for human-driven observe mode is
    reused here as an autonomous PRE-CONDITION on every decide gesture (see
    AndroidDriver._require_deck_confirmed). If either glyph is not visible — because a
    purchase sheet, an ad, a permissions dialog, or literally anything else has come between
    the bot and the deck — this refuses instead of firing a gesture blind, the same "fail
    loud, never guess" rule UnlocatedControlError already applies to a single missing button.

    A spec declaring no 'like'/'pass' templates at all (Bumble's current, real, uncalibrated
    state) has no way to answer the question, and the check is skipped entirely — the same
    "no template -> no vision path, no fixed-coord fallback" precedent _locate_button already
    sets. That is not a loophole: such a spec cannot be marked calibrated (see
    AndroidAppSpec.__post_init__'s has_paid_upsell check) and a TAP-gesture app in that state
    already can't decide at all (_await_button refuses with UnlocatedControlError for the
    same missing-template reason). The gap this class closes is for a CALIBRATED app whose
    deck glyphs are momentarily hidden, not for a spec that was never wired up to prove
    anything in the first place.

    Ground truth from the owner (measured 2026-08-10): this guard is the ONLY protection that
    covers Bumble's non-zero-SuperSwipe-balance case. When the account holds a balance (5, at
    measurement time), a stray SuperSwipe spends SILENTLY — no sheet, nothing for
    _handle_rose_upsell to dismiss, nothing at all downstream to catch it. Refusing to act
    unless the deck is positively confirmed is what stands between a drifted/blocked screen
    and a real, irreversible, paid action in that state — the confirmation sheet itself is
    not a safety net there; it does not even exist."""


class PaidUpsellStuckError(HingeActionError):
    """A paid-upgrade sheet was positively detected (its 'upsell_dismiss' template matched)
    and remained on screen after AndroidDriver._dismiss_via_zone's bounded retry budget.

    Raised instead of retrying forever, deliberately: a dismiss tap that keeps landing just
    outside the sheet (drift, a device rotation, an app update that moved the layout) must
    not turn into an indefinite sequence of blind taps near a screen whose bottom third can
    be a purchase button (MEASURED 2026-08-10 on Bumble's real SuperSwipe sheet: the CTA
    sits at y 0.899-0.951 — see BUMBLE_SPEC). Halting and preserving the on-screen state for
    debugging is always safer than one more guess at the same modal."""


_OBSERVE_POLL_S = 0.35     # internal sampling cadence for your manual tap (not app-facing)
# The _note_observe_waiting reasons that describe an OPEN LIKE SHEET rather than an
# unclassifiable screen. They repeat on the slower _OBSERVE_LIKE_NOTICE_S cadence and get
# their own wording -- see that method.
_OBSERVE_LIKE_WAIT_REASONS = frozenset({"like_sheet", "like_sending"})
# Minimum gap between ANY two waiting notices, including ones with different reasons -- the
# backstop against a flapping screen turning the heartbeat into a firehose. See
# _note_observe_waiting.
_OBSERVE_NOTICE_FLOOR_S = 2.0
_UPSELL_DISMISS_MAX_ATTEMPTS = 3
# Bounded retries for AndroidDriver._dismiss_via_zone before it gives up (PaidUpsellStuckError)
# rather than tapping an already-detected modal again and again. See that error's docstring.

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
    # Hinge's own sticky profile header: the person's NAME, centred, appears here the moment
    # the card is scrolled at all, and stays pixel-identical for the whole profile. Measured
    # live on the Pixel 7a (1080x2400) 2026-08-10 over three frames of one profile at three
    # different scroll offsets: mean abs delta 0.00 between them, 17.95 against the same
    # profile's scroll-top frame (which shows the profile-independent filter-chips row here
    # instead). change_threshold is 9.0, so the separation is unambiguous. x1 stops at 0.80 to
    # keep the "..." overflow button out of the band.
    identity_band=(0.10, 0.048, 0.80, 0.094),
    # Card-header NAME band, consulted only when identity_band's own verdict above is exactly
    # "top" (scroll-top, where identity_band shows the profile-independent filter-chips row
    # instead of a name -- see AndroidAppSpec.identity_top_name_band's docstring for the full
    # mechanism and _identity_of below for the resolution rule). This is the fix for the
    # incident that motivated this field: a pass that advanced the deck from "Alina" to
    # "jessica" was recorded as a scroll WITHIN Alina's profile, because nothing on screen at
    # scroll-top could name the new card and the decision fell through to a loose content
    # match. OCR-only, never a pixel signature -- MEASURED on the real Pixel 7a on 2026-08-10 by
    # running `tesseract --psm 6` over the actual failing run's frames. Reads (all correct):
    #   Alina, scroll-top, with the purple "shows thoughtful signals" banner -> "Alina %"
    #   Alina, scroll-top, banner gone (content shifted up)   -> "Alina @ | @ Signals Active today"
    #   jessica, scroll-top, with banner (three separate frames) -> "jessica &"
    # On SCROLLED frames the same band OCRs to garbage, which is safe: the check this feeds is
    # gated on the pixel verdict being genuinely "top".
    identity_top_name_band=(0.03, 0.130, 0.75, 0.250),
    # Scrolling content only. Above 0.125 is the status bar + sticky header, below 0.875 is the
    # floating X/heart overlay and the dark bottom nav -- none of which translate when content
    # scrolls, which is exactly why a whole-frame shift search never matched (see the table in
    # the module comment on _vertical_shift_match).
    content_band=(0.125, 0.875),
    observe_ignore_zones=(
        (0.75, 0.030, 1.00, 0.115),   # rewind arrow + "..." overflow, top-right
        (0.00, 0.900, 1.00, 1.000),   # bottom nav bar (dark bar starts at y frac 0.906)
    ),
    # OFF, because it demonstrably cannot work on the target device. Gesture corroboration
    # (touchwatch.py) reads the phone's own touch stream to prove a human actually PRESSED
    # something. Measured on the Pixel 7a, Android 17 (SDK 37, 2026-06-05 patch), 2026-08-10:
    # the adb `shell` user IS in group 1004(input), `/dev/input/event3` is `crw-rw---- root
    # input`, and `getevent -p /dev/input/event3` reads the touchscreen's full capability set
    # -- so the node opens fine -- but with the owner deliberately tapping and scrolling for
    # 30s, ZERO lines were delivered (raw_line_count 0, not merely unparsed). Android
    # withholds the input event stream from unprivileged readers on this build; recording
    # input is a rooted-device capability. No code change here can recover it.
    #
    # This is a lost EXTRA layer, not the fix: the identity anchor above is what stops a human
    # scroll being recorded as a PASS, and it is fully verified against real device frames.
    # What is given up is the narrower set of cases only a real tap could disambiguate -- a
    # rewind/nav tap that changes the card without being a decision, and the same-first-name
    # collision (see _is_current_profile_frame). Both are logged as accepted risk in
    # ops/ANTI-BOT-RESEARCH.md with a §5 re-check trigger.
    #
    # The machinery is kept and tested rather than deleted: it is correct, it costs nothing
    # while off, and it becomes available the moment this runs on a rooted device or a
    # platform that permits the read. Re-enable with apps.hinge.observe_touch_watch: true and
    # confirm with `python -m tools.touch_selftest` BEFORE trusting it -- that tool now
    # reports raw lines separately from parsed events precisely so "the platform sent nothing"
    # can never again be confused with "the parser understood nothing".
    observe_touch_watch=False,
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


_IDENTITY_DS = (64, 16)   # identity-band downsample (w, h) -- see _band. Matches the crop's
# roughly 5.6:1 aspect ratio (identity_band spans 0.70 of the width, 0.046 of the height) at
# enough resolution for OCR (_ocr_band crops the SAME rect at full frame resolution, this
# downsample is only for the pixel-distance comparison _identity_of makes on every poll).


def _band(frame: bytes, rect: tuple[float, float, float, float]):
    """Downsampled grayscale crop of the normalised rect `(x0, y0, x1, y1)` of `frame`, at
    _IDENTITY_DS resolution. None when the frame can't be decoded -- same contract as
    _downsample (every caller here already knows how to fall back to 'unknown' on None; a
    decode failure must never read as either a positive 'same' or 'new' identity signal).

    This is a SEPARATE crop from _downsample's whole-frame one, deliberately: the identity
    band is a thin strip near the top of the screen (see HINGE_SPEC's identity_band comment
    for the measured geometry), and cropping it out before downsampling is what makes the
    0.00-vs-17.95 separation measured on the real device (see the module docstring) actually
    survive to a 64x16 array -- downsampling the WHOLE frame first and then trying to compare
    a sub-region of that would throw away exactly the resolution this anchor depends on.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
        im = Image.open(BytesIO(frame)).convert("L")
        w, h = im.size
        x0, y0, x1, y1 = rect
        crop = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        return np.asarray(crop.resize(_IDENTITY_DS), dtype="int16")
    except Exception:  # noqa: BLE001 — any decode/dep failure -> caller falls back to 'unknown'
        return None


def _band_dist(a, b) -> float:
    """Mean abs diff (0..255) between two _band() arrays. Callers only ever call this once
    both are already confirmed non-None -- unlike _split_diff, this has no undecodable-bytes
    fallback, because there is no meaningful "distance" between a real band and "we couldn't
    decode anything"."""
    import numpy as np
    return float(np.mean(np.abs(a - b)))


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


# The hard bounds _sample_read_step/_sample_read_scroll validate any policy-sampled read-scroll
# distance against before issuing a gesture (see both call sites below). Named here so
# _vertical_shift_match derives its search radius from the SAME numbers instead of a second,
# independently-drifting copy of them -- two frames _capture_current() stopped at while reading
# one profile top-to-bottom can be up to _READ_SCROLL_FRAC_MAX apart.
_READ_SCROLL_FRAC_MIN = 0.10
_READ_SCROLL_FRAC_MAX = 0.75


def _content_rows(content_band: tuple[float, float], size: int) -> tuple[int, int]:
    """Row range `(r0, r1)` of `content_band`'s `(y0, y1)` fractions on a `size`-row
    downsample. A shared helper so every _vertical_shift_match call site derives its crop
    from the SAME content_band field instead of a second, independently-drifting copy of the
    arithmetic -- exactly the kind of drift _READ_SCROLL_FRAC_MIN/MAX's own module comment
    above calls out. Clamped so a degenerate size (0 or 1) can't produce an empty or
    inverted slice."""
    y0, y1 = content_band
    r0 = max(0, min(size - 1, round(y0 * size)))
    r1 = max(r0 + 1, min(size, round(y1 * size)))
    return r0, r1


def _vertical_shift_match(cur, seen, *, threshold: float,
                          rows: tuple[int, int] | None = None) -> tuple[bool, int, int]:
    """Whether downsampled frame `cur` could be a vertically-scrolled view of `seen`, at ANY
    vertical offset -- not just the one wait_for_decision's exact-position check already tried.

    That exact-position check only recognizes a manual scroll that happens to land back on one
    of _capture_current's own fixed read-scroll stops. A human scrolling by hand to actually
    read a profile -- Signals behavior #1, and exactly what the owner was doing when this was
    reported -- stops wherever their thumb does, not at the bot's own stride. Everywhere in
    between the automated stops used to read as "whole card changed" and got silently recorded
    as a PASS despite no pass/like tap ever happening.

    This tries every vertical offset a real scroll between two adjacent read-scroll stops could
    produce (bounded by _READ_SCROLL_FRAC_MAX in either direction -- the search doesn't need to
    know or care which way the human actually scrolled) and matches on the best-aligned
    OVERLAPPING band at each one. A genuine scroll's shared content lines up at some offset
    wherever it stopped; two frames from actually different profiles share no such alignment at
    any offset tried here, so a real pass is still recognized as one.

    `rows`, when given, is a `(r0, r1)` row range (see `_content_rows`) BOTH frames are sliced
    to before the shift search runs -- this is the fix for a bug this helper shipped with
    once already. MEASURED on the real Pixel 7a (1080x2400) 2026-08-10, mean-abs-diff
    best-shift distance across five real frames of one profile scrolled to different offsets:

        pair (same profile, different scroll offsets)   full-frame   content-rows-only
        scrolled A vs scrolled B                            38.94          3.48
        scrolled A vs scroll-top                            41.38          5.41
        scrolled A vs scrolled N                            30.38          3.34
        scroll-top vs deep-scrolled                         32.88         11.94

    change_threshold is 9.0. Every full-frame number above is 3-4x over it, so searching the
    WHOLE downsampled frame (`rows=None`, this function's very first shipped shape) could
    never return True in production: the status bar, the sticky header, and the bottom nav
    bar do NOT translate when the content scrolls, so they dominate the mean-abs-diff at
    EVERY offset tried, no matter how well the actual content lines up underneath them.
    Restricting the search to `content_band`'s rows fixes 3 of the 4 pairs above; the 4th
    (a scroll-top frame vs one scrolled 13 of 24 downsample rows further down) still misses
    at any offset -- which is exactly why this helper is corroboration (layer 2), never the
    authoritative check: the identity-band anchor (`_identity_of`, layer 1) is what actually
    closes the bug, and wait_for_decision only reaches this helper at all once identity has
    already failed to call the frame 'same'.

    `rows=None` (the default) searches the whole frame -- kept for any caller with no
    content_band to slice against, and for the direct unit tests below that exercise the
    shift-search arithmetic itself rather than the chrome-exclusion fix.

    Returns `(matched, shift, overlap_rows)` -- a bool used to be the WHOLE return value, and
    that bool's semantics are unchanged (first offset under `threshold`, scanned in the same
    `-max_shift..max_shift` order as before). `shift` and `overlap_rows` exist because the
    bool alone is exactly what made the incident that motivated this change undiagnosable: the
    only trace it left was "matched a stored signature", with no way to see WHICH one, at what
    offset, or how much of the band actually lined up. `shift` is the offset the decision was
    made at (positive = `cur`'s content sits `shift` downsample rows below `seen`'s);
    `overlap_rows` is how many of `content_band`'s rows were actually compared at that offset
    (`band_h - abs(shift)` -- the rest fell off one end and were never compared at all). When
    NO offset matched, `shift`/`overlap_rows` describe the BEST (lowest mean-abs-diff) offset
    tried instead, purely as a diagnostic for "how close did it get" -- `matched` is still the
    only thing any caller's control flow may depend on.
    """
    import numpy as np
    size = cur.shape[0]
    r0, r1 = rows if rows is not None else (0, size)
    cur_c, seen_c = cur[r0:r1], seen[r0:r1]
    band_h = cur_c.shape[0]
    # MEASURED evidence toward a possible future tightening of the match itself (NOT done in
    # this pass -- see the module docstring's redesign notes for why): of the 4 genuine
    # same-profile shift-match pairs observed live on the Pixel 7a while diagnosing the
    # Alina/jessica incident, overlap_rows/band_h came out to 9/18, 11/18, 12/18, and 18/18 --
    # all >= 50%, against a floor this function has always allowed of 4/18 (~22%, the
    # `band_h - 4` term below). That is suggestive that a >=50% overlap floor could reject
    # weaker, more coincidental matches without losing a real one -- but 4 samples is far too
    # few to move a threshold that exists to guard against the ORIGINAL false-PASS bug
    # (mis-swallowing a genuine advance as "just a scroll"). This needs on-device
    # LIVE-VERIFY first (see ops/RUNBOOK.md's convention for that), which the new
    # `overlap_rows` return value is what will finally make possible to gather at scale.
    max_shift = max(1, min(band_h - 4, math.ceil(_READ_SCROLL_FRAC_MAX * size)))
    best_shift, best_overlap, best_dist = 0, band_h, None
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            cur_band, seen_band = cur_c[shift:], seen_c[:band_h - shift]
        else:
            cur_band, seen_band = cur_c[:band_h + shift], seen_c[-shift:]
        if cur_band.size == 0:
            continue
        dist = float(np.mean(np.abs(cur_band - seen_band)))
        if best_dist is None or dist < best_dist:
            best_dist, best_shift, best_overlap = dist, shift, cur_band.shape[0]
        if dist < threshold:
            return True, shift, cur_band.shape[0]
    return False, best_shift, best_overlap


# --- identity: resolving the scroll-top "top" verdict by OCR'ing the card header ---------
# See _identity_of's own "Layer 1b" block for the full rule this feeds. Module-level (not a
# class attribute like _OCR_NAME_RE) because both need to be usable without an AndroidDriver
# instance in hand -- the ratio calibration below is itself a standalone, testable fact about
# two strings, independent of any driver state.

_NAME_MATCH_RATIO = 0.6
# How close a token OCR'd from identity_top_name_band must score against the stored profile
# name (difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio(), called via
# _name_token_matches below with the OCR'd TOKEN first and the STORED name second -- this
# ratio is NOT symmetric under argument swap, so the numbers here are specifically the call's
# real order, not the reverse) to count as a match for that profile. Deliberately calibrated
# LOW -- biased toward "same" -- because the two possible errors here are not symmetric: a
# false "same" costs at most a missed pass (the loop just keeps waiting; nothing is written,
# exactly today's behaviour), while a false "new" records a PASS the human never made and
# corrupts the taste model with a decision that did not happen. MEASURED on the real
# incident's OCR reads: "Alina" vs a plausible misread "Alma" scores 0.67, and vs "Aiina"
# scores 0.80 -- both MUST stay "same". The actual next profile's OCR'd token ("jessica")
# against the stored name ("Alina") -- SequenceMatcher(None, "jessica", "alina"), the call's
# real argument order -- scores 0.33 (VERIFIED 2026-08-10; a naive reverse-order read would
# suggest 0.17, which is not what this code computes) -- MUST become "new" either way. 0.6
# sits comfortably below every observed misread and far above every observed genuine
# difference, i.e. on the safe side of both real data points this fix was built to get right.
#
# Ratio alone is not enough: it was calibrated only against substitution-style misreads and
# scores BELOW 0.6 for a plain TRUNCATION of a longer name ("Alina" read as "Al" is 0.57,
# "Katherine" read as "Kat" is 0.50, as "Ka" is 0.36) -- a common tesseract failure (a partial
# crop at the band edge, tight kerning), and truncation gets WORSE, not better, the longer the
# stored name is. _name_token_matches below adds a prefix test specifically to close that hole
# without touching this ratio (see its own docstring for why a prefix test is the right shape
# of fix and the full set of truncation cases it was verified against).
def _name_token_matches(seen: str, stored: str) -> bool:
    """True if an OCR'd token `seen` and the profile's stored name `stored` are close enough
    to call the SAME person, biased toward "same" per this module's own reasoning above (a
    false "same" costs a wait; a false "new" writes a wrong label).

    Two independent tests, either is enough:

      1. difflib ratio >= _NAME_MATCH_RATIO -- catches substitution-style misreads
         ("Alina"/"Aiina", "Alina"/"Alma").
      2. either string is a case-insensitive PREFIX of the other -- catches TRUNCATION, which
         is a common tesseract failure (a partial crop at the band edge, tight kerning) and
         scores BELOW the ratio bar for longer names (see the module comment above). A
         truncation is by definition a prefix, so this one test closes the hole in BOTH
         directions: a truncated READ this poll ("Al" seen for a profile stored as "Alina")
         and a truncated STORE from a bad capture-time read ("Al" was what got stored for a
         profile actually named "Alina", so a later full "Alina" read must still match it --
         otherwise the bad capture poisons the whole profile with spurious "new" verdicts).

    VERIFIED (2026-08-10) against every case this fix targets -- read as seen(stored)=ratio:

      truncated reads, both directions -- all "same" via the prefix test:
        Al(Alina)=0.57, Kat(Katherine)=0.50, Ka(Katherine)=0.36    -- read got truncated
        Alina(Al)=0.57, Samantha(Sam)=0.55                        -- STORED name was truncated
      genuine differences -- still "new" (no prefix relationship, ratio stays below bar):
        Alina(jessica)=0.17, jessica(Alina)=0.33, Katherine(Michelle)=0.35, Alina(Signals)=0.33
      genuine same-name misreads -- unaffected, still "same" via the ratio path alone:
        Aiina(Alina)=0.80, Alma(Alina)=0.67, Aline(Alina)=0.80, lina(Alina)=0.89
    """
    seen_cf, stored_cf = seen.casefold(), stored.casefold()
    if seen_cf and stored_cf and (seen_cf.startswith(stored_cf) or stored_cf.startswith(seen_cf)):
        return True
    return difflib.SequenceMatcher(None, seen_cf, stored_cf).ratio() >= _NAME_MATCH_RATIO


_TOP_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
# Tokens considered as name candidates in the scroll-top card-header OCR read: a letter
# followed by one or more letters/apostrophes/hyphens -- i.e. length >= 2, which excludes
# stray single-character OCR noise ("|", "l", "@") without needing a separate length check.
# This is the SAME-match tokenizer only -- a token this short CAN still resolve a "same"
# verdict via _name_token_matches' prefix test above (a 2-char truncation like "Al" or "Ka" is
# exactly the failure mode that test exists for). The "new"-candidate selection in
# _identity_of's Layer 1b block applies its OWN, stricter length floor (>= 3 alphabetic
# characters) on top of this regex, because a 1-2 character token is noise, never a usable
# name, and must never by itself be the basis for recording a PASS.
_TOP_NAME_CHROME_WORDS = frozenset({
    "signals", "active", "today", "shows", "thoughtful", "age", "height", "dating", "intent",
})
# Words HINGE ITSELF renders in the card-header band, observed in the real reads that built
# identity_top_name_band ("Alina @ | @ Signals Active today", jessica's "Signals ( Agev )
# Height v" equivalent at scroll-top): none of these is ever a person's first name. Without
# this blocklist, a frame where OCR catches the chrome text but misses the name entirely would
# take "Signals" (say) as the candidate name, score ~0.1-0.2 against the stored name -- clearing
# nothing, but ALSO not blocked -- and the "first non-chrome token" fallback below would
# misread it as a genuinely new profile: a FALSE PASS. With the blocklist, a read containing
# only chrome words yields no candidate at all and the verdict stays "top" (inconclusive) --
# the safe outcome; the pixel/content layers downstream still get to decide.


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

    # Both capture entry points (next_profile / current_profile) accept and honour a
    # `should_stop` callable, polled between screencaps, between read-scrolls, inside the read
    # dwell, and between _scroll_to_top's undo-swipes. Declared here (class-level, so both
    # Android bindings inherit it) because worker.py only passes the callable to a driver that
    # advertises it -- see DatingAppDriver.supports_interruptible_capture. It is True for the
    # whole AndroidDriver family and not for BumbleWebDriver, whose own capture waits are not
    # stop-aware: a flag claiming a capability the code does not have would be worse than no
    # flag, since the worker would then believe Stop is handled when it silently is not.
    supports_interruptible_capture = True

    def __init__(self, cfg, spec: AndroidAppSpec):
        self.spec = spec
        self.accepts_opener = spec.accepts_opener
        # Only comment-sheet apps have an intermediate human like intent. A direct-flow
        # Android driver (Bumble) must keep its ordinary pass/like observer contract.
        self.supports_observe_like_intent = (
            spec.like_flow == "comment_sheet" and spec.accepts_opener
        )
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
        # Observe-mode decision detection (see the module docstring's redesign notes): every
        # field here is config-overridable per app the SAME way change_threshold/coords are
        # above, not read off self.spec directly, so a per-device geometry tweak lives in
        # config.yaml rather than requiring a code change.
        identity_band = app_cfg.get("identity_band", spec.identity_band)
        self.identity_band = tuple(identity_band) if identity_band is not None else None
        identity_top_name_band = app_cfg.get("identity_top_name_band", spec.identity_top_name_band)
        self.identity_top_name_band = (
            tuple(identity_top_name_band) if identity_top_name_band is not None else None)
        self.content_band = tuple(app_cfg.get("content_band", spec.content_band))
        self.observe_ignore_zones = tuple(
            tuple(zone) for zone in app_cfg.get("observe_ignore_zones", spec.observe_ignore_zones))
        self.observe_touch_watch = bool(app_cfg.get("observe_touch_watch", spec.observe_touch_watch))
        # Re-run AndroidAppSpec.__post_init__'s rect/arity/pairing checks against these five
        # fields' MERGED (config.yaml-overridden) values, not just spec's own hardcoded
        # literals. __post_init__ already validates identity_band/identity_top_name_band/
        # content_band/observe_ignore_zones's shape and 0..1 range, and refuses
        # observe_touch_watch=True paired with identity_band=None (and, since
        # identity_top_name_band was added, identity_top_name_band set with identity_band=None)
        # -- but until now it only ever ran ONCE, against HINGE_SPEC's/BUMBLE_SPEC's own
        # literals at import time, because that used to be the only time an AndroidAppSpec got
        # constructed. An operator's apps.<app>.identity_band/identity_top_name_band/
        # content_band/observe_ignore_zones/observe_touch_watch override above bypassed all of
        # that: an identity_band with x0/x1 swapped (an easy typo) constructed fine and then
        # silently disabled Layer 1 forever -- _band()'s blanket `except Exception: return
        # None` (below) swallows PIL's crop error and _identity_of falls back to 'unknown'
        # with no print, no warning, nothing; a content_band with the wrong arity constructed
        # fine and only surfaced as an uncaught ValueError deep inside wait_for_decision's
        # Layer 2 fallback, mid-session; and identity_band: null in config.yaml (dict.get
        # returns the stored None, not spec's default, when the key is PRESENT with value
        # null) with observe_touch_watch left at Hinge's default True reached exactly the
        # combination __post_init__ refuses to construct -- silently, because nothing here
        # ever asked it again after the merge. identity_top_name_band flows through this SAME
        # rebuild for the identical reason: an operator override that pairs it with
        # identity_band=None (or gives it a malformed rect) must fail loudly here, not leave
        # _identity_of silently unable to ever resolve a "top" verdict by name.
        #
        # dataclasses.replace() rebuilds a frozen AndroidAppSpec from `spec` with these five
        # fields swapped in, which reruns __init__ -- and therefore __post_init__ -- against
        # the MERGED values, purely for the side effect of __post_init__'s validation raising
        # the same ValueError it already raises for a bad literal. This is the only way to get
        # that validation without a second, independently-drifting copy of the rect/arity/
        # pairing logic living here too (see AndroidAppSpec.__post_init__ for what actually
        # gets checked). The resulting spec is discarded -- self.spec keeps pointing at the
        # pre-merge instance, since only these five fields are ever config-overridden this
        # way; self.coords/self.read_scroll_frac have their own equivalent check in config.py's
        # _validate_android_fractions, deliberately not duplicated here.
        try:
            dataclasses.replace(
                spec,
                identity_band=self.identity_band,
                identity_top_name_band=self.identity_top_name_band,
                content_band=self.content_band,
                observe_ignore_zones=self.observe_ignore_zones,
                observe_touch_watch=self.observe_touch_watch,
            )
        except ValueError as exc:
            raise DriverClosed(
                f"{spec.app}: apps.{spec.app} config.yaml overrides of identity_band / "
                f"identity_top_name_band / content_band / observe_ignore_zones / "
                f"observe_touch_watch produced an invalid combination once merged with "
                f"{spec.app}'s spec defaults ({exc}). Fix the offending apps.{spec.app}.<key> "
                f"named above in config.yaml -- leaving it broken would silently disable the "
                f"identity anchor (or worse) for the whole run with no further warning."
            ) from exc
        self.observe_name_ocr = bool(app_cfg.get("observe_name_ocr", True))
        self._touch_watcher: TouchWatcher | None = None   # started in open_session, closed in close()
        self._touch_watch_health_warned = False   # print the event_count==0 warning at most once/run
        # Whether THIS session is autonomous (Worker._auto_loop) rather than observe. The
        # driver has no `mode` of its own -- Worker never passes one in (make_driver builds a
        # bare AndroidDriver; mode lives only on Worker itself) -- so this starts False
        # (observe/unknown, the conservative default that preserves every existing caller's
        # behavior, including calibration tools like tools/hinge_inspect.py that construct a
        # driver directly and never call set_auto_session_policy at all) and flips True only
        # inside set_auto_session_policy below, the one hook Worker._auto_loop already calls,
        # unconditionally, before open_session() -- see that method's docstring for why this
        # is the cheapest correct signal rather than a new mode-passing parameter. Read by
        # open_session()'s touch-watcher gate: gesture corroboration exists to corroborate a
        # HUMAN tap against wait_for_decision's identity-and-deck-ready proof, and auto mode
        # never calls wait_for_decision at all (it drives itself via like()/dislike()), so
        # attaching TouchWatcher for it would hold a persistent `adb shell getevent` subprocess
        # open for the whole run for a reader that never exists.
        self._auto_session = False
        # Set fresh by _capture_current every profile; None until a profile actually reveals
        # the sticky header (identity_top_sig) and, past that, until the header itself
        # (identity_sig) is seen. See _capture_current's identity-anchor block and
        # _identity_of, which is the only reader of these three.
        self._identity_top_sig = None
        self._identity_sig = None
        self._identity_name = None
        # The scroll-top card-header OCR text _identity_of last read (or None if it never ran
        # this call -- see that method's "Layer 1b" block). Reset on every _identity_of call,
        # not just every profile, so an observe_scroll debug record logged right after can
        # report exactly what was seen at the moment THAT verdict was decided.
        self._identity_top_name_read = None
        # What Layer 1b itself concluded ('same' / 'new'), or None if it never ran or never
        # reached a verdict this call. Reset alongside _identity_top_name_read, for the same
        # reason. wait_for_decision reads this immediately after the FIRST _identity_of(cur)
        # call of a poll to tell a name-derived 'new' apart from a pixel-derived one -- see
        # its own "name-derived 'new' must be reproduced" comment for why that distinction
        # gates whether a confirm-frame 'top' is allowed to corroborate a PASS.
        self._identity_top_name_verdict = None
        # Bounded memo cache for _ocr_band: keyed on (rect, psm, sha1(frame)), holding at most
        # _OCR_BAND_CACHE_MAX entries. See _ocr_band's own comment for the measured cost this
        # exists to avoid. Reset per profile (in _capture_current) alongside the other identity
        # state, not just here, so a cached read never survives into a different profile.
        self._ocr_band_cache: dict[tuple, str | None] = {}
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
        # Rate-limit state for _note_observe_waiting. Both are re-anchored at the top of every
        # wait_for_decision call (one call = one profile's wait); initialised here as well so
        # the notice helper is safe to call from anywhere -- including tools and tests that
        # exercise it without going through a full wait. _observe_last_reason starts as None,
        # which no branch ever passes as a reason, so the FIRST notice of a session is always
        # emitted rather than suppressed by an accidental match.
        self._observe_last_notice = 0.0
        self._observe_last_reason: str | None = None
        # Whether this session's one-shot "put the card at a confirmed scroll-top" pass has
        # been attempted yet -- see _ensure_session_top for why a session cannot assume the
        # previous one left the card where it found it.
        self._session_top_done = False
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
        if self.observe_name_ocr and shutil.which("tesseract") is None:
            # Best-effort only (see _ocr_band) -- the sticky-header PIXEL signature is the
            # authoritative identity anchor either way, so a missing binary degrades the
            # observe-mode name log/OCR corroboration, never the core PASS guarantee. One
            # line, once, so it's visible without being noisy on every profile.
            print("profile-name OCR unavailable (tesseract not on PATH); identity still "
                  "enforced by the sticky-header signature.")
        if self.observe_touch_watch and not self._auto_session:
            # Gesture corroboration (layer 3), OBSERVE SESSIONS ONLY -- see self._auto_session
            # in __init__. Its whole job is to prove a HUMAN pressed something, which only
            # wait_for_decision ever asks; auto mode drives itself through like()/dislike()
            # and never calls wait_for_decision at all, so attaching here would hold a
            # persistent `adb shell getevent` subprocess open for an entire autonomous run
            # with no reader. That is not merely wasteful: ops/ANTI-BOT-RESEARCH.md's
            # 2026-08-10 (b) addendum scoped the accepted risk of holding that subprocess open
            # to observe sessions specifically, so quietly widening it to autonomous runs
            # would spend risk the log says was never accepted.
            #
            # Unlike the OCR probe above, this one DOES fail
            # loud: without a live touch stream, a card advance can only ever be corroborated
            # by identity + deck-ready (layers 1/2) -- narrower proof than what
            # observe_touch_watch=True was asking for. Silently running with it anyway would
            # be exactly the kind of quiet capability downgrade touch_backend's own "auto
            # REQUIRES UHID, never silently downgrades" contract exists to prevent (see
            # _make_touch above) -- so this raises the same way, naming the config key that
            # accepts the narrower proof deliberately.
            watcher = TouchWatcher(self.adb_path, self.serial, self._adb.screen_size())
            try:
                watcher.start()
            except TouchWatchUnavailable as exc:
                raise DriverClosed(
                    f"{self.spec.app}: gesture corroboration (observe_touch_watch) could not "
                    f"attach to the device's touch event stream ({exc}). Running observe mode "
                    f"without it would silently narrow PASS proof back to identity + deck-"
                    f"ready alone, with no evidence of an actual tap -- fine as a deliberate "
                    f"choice, not as a silent fallback. Fix the device/permissions (see "
                    f"ops/ANTI-BOT-RESEARCH.md), or set apps.{self.spec.app}.observe_touch_watch: "
                    f"false to accept the narrower proof on purpose."
                ) from exc
            self._touch_watcher = watcher
            print(f"{self.spec.app}: touch watcher attached on {watcher.device_path} "
                  f"({watcher.device_name!r}) — gesture corroboration is live for observe mode.")
        try:
            if self.debug_log:
                self._dbg = open_debug_log(self.debug_dir)
        except BaseException:
            # Anything after the watcher attaches must hand the subprocess back if it fails.
            # open_session() raising is NOT followed by a close() call -- the worker treats a
            # failed open as "this driver never opened" and moves on -- so without this the
            # `adb shell getevent` process started a few lines above outlives the session that
            # started it, invisibly, holding the device's touch stream until the whole hub
            # exits. close() stays idempotent, so the ordinary teardown path is unaffected.
            self.close()
            raise

    def close(self) -> None:
        if self._touch is not None and self._touch is not self._adb:
            try:
                self._touch.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the real outcome
                pass
        self._touch = None
        if self._touch_watcher is not None:
            try:
                self._touch_watcher.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the real outcome
                pass
            self._touch_watcher = None
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
    #
    # A later audit found the check itself could be evaded even where it WAS wired up: it
    # computed the zone-check fraction from the RAW, UNCLAMPED (x, y), but both real
    # transports CLAMP the coordinate they actually deliver AFTER this check runs (uhid.py's
    # _report, Adb._clamp — see clamp_xy in adb.py, which both now share). forbidden_zones
    # rects are constrained to 0.0..1.0, so any coordinate whose fraction fell OUTSIDE that
    # range matched no zone and passed cleanly — then the transport clamped it onto the
    # screen edge, which can be INSIDE a zone. The point that was CHECKED was not the point
    # that got DELIVERED. See OutOfRangeTapError for the demonstrated exploit.
    _TAP_ZONE_MARGIN_PX = tap_jitter_margin_px()
    # The zone-check margin _tap() applies (see its call below): the worst-case single-axis
    # drift EITHER real touch transport's tap can add to the delivered touch-down AFTER this
    # check runs, so a zone can't be evaded by jitter the checked point never accounted for.
    # UhidTouch.tap() calls human_motion.plan_tap() at plan_tap's own default jitter_px (it
    # overrides jitter_px for swipes only, not taps), so tap_jitter_margin_px()'s default
    # argument is the correct bound for the transport this project requires by default.
    # Adb.tap()'s simpler transport (its `input tap` fallback, reachable only via an explicit
    # touch_backend: adb) jitters with a plain uniform(-jitter_px, jitter_px) of at most its
    # own jitter_px (default 2.0px) — comfortably smaller — so this one constant safely
    # covers both without needing to know which backend is actually wired up.

    def _assert_tap_allowed(self, x: int, y: int, margin_px: float = 0.0) -> None:
        """The choke point itself. Two checks, in this order:

        1. RANGE — (x, y) must already resolve to a fraction inside 0..1 of the live screen,
           checked on the RAW value before any clamping. An out-of-range value is always a
           configuration or logic error (never a legitimate touch — see OutOfRangeTapError),
           and letting the transport silently clamp it instead of refusing it here is exactly
           the gap this whole check exists to close, so it fires even for a spec that
           declares no forbidden_zones at all.

        2. ZONE — clamp (x, y) the SAME way the real transports do (clamp_xy — see the class
           comment above), then require that point, widened by `margin_px` on every side, to
           clear every forbidden zone. `margin_px` covers jitter the transport adds AFTER
           this check returns (see _tap()/_TAP_ZONE_MARGIN_PX above); callers whose delivered
           point has no such jitter (_swipe(), _scroll() below) pass 0.
        """
        w, h = self.adb.screen_size()
        fx_raw = x / w if w else 0.0
        fy_raw = y / h if h else 0.0
        if not (0.0 <= fx_raw <= 1.0 and 0.0 <= fy_raw <= 1.0):
            raise OutOfRangeTapError(
                f"{self.spec.app}: refusing a touch at raw pixel ({x}, {y}) = fraction "
                f"({fx_raw:.3f}, {fy_raw:.3f}) of the {w}x{h} screen — outside 0..1. This is "
                f"always a configuration or logic error: check apps.{self.spec.app}.coords, "
                f"read_scroll_frac, or any other *_frac knob for this app for a typo, or a "
                f"pixel value written where a fraction was expected. Refusing rather than "
                f"letting the transport silently clamp it onto a screen edge, which can land "
                f"inside a forbidden zone undetected.")
        zones = getattr(self.spec, "forbidden_zones", ())
        if not zones:
            return
        cx, cy = clamp_xy(x, y, w, h)      # the point that will ACTUALLY be delivered
        mfx = margin_px / w if w else 0.0
        mfy = margin_px / h if h else 0.0
        fx, fy = (cx / w if w else 0.0), (cy / h if h else 0.0)
        for zone in zones:
            x0, y0, x1, y1 = zone
            if x0 - mfx <= fx <= x1 + mfx and y0 - mfy <= fy <= y1 + mfy:
                margin_note = f" (+/-{margin_px:.1f}px jitter envelope)" if margin_px else ""
                raise ForbiddenTapError(
                    f"refused a tap at ({cx}, {cy}) = ({fx:.3f}, {fy:.3f}) of the "
                    f"screen{margin_note}: it lands inside, or could jitter into, "
                    f"{self.spec.app}'s forbidden zone {zone}, which guards a paid control. "
                    f"Refusing rather than risking a paid action.")

    def _tap(self, x, y) -> None:
        x, y = int(x), int(y)
        self._assert_tap_allowed(x, y, margin_px=self._TAP_ZONE_MARGIN_PX)
        self.touch.tap(x, y)

    def _swipe(self, x1, y1, x2, y2) -> None:
        """Every explicit drag goes through here, for the same reason every tap goes
        through _tap(). Only the START point is zone-checked (margin_px=0: plan_swipe's
        first sample is pinned exactly to (x1, y1) with no jitter, so there is no drift to
        cover here — see plan_swipe in human_motion.py): the touch-down claims the gesture,
        so a drag that merely travels over a control does not press it.

        That "touch-down claims the gesture" model is correct for STANDARD Android touch
        dispatch (a MotionEvent stream goes to whichever View captured ACTION_DOWN,
        regardless of where ACTION_MOVE/ACTION_UP later land) — but it is UNVERIFIED against
        Bumble's actual UI, and the stakes of being wrong are real money. _scroll_to_top's
        undo-swipes travel DOWNWARD (returning the card to the top) and routinely END with
        the finger sitting over Bumble's SuperSwipe location. If Bumble's real button turns
        out to react to where a drag ENDS — a custom touch listener, a drop target, anything
        other than plain View dispatch — this check as written would not catch it. See
        ops/RUNBOOK.md's Bumble calibration checklist: before Bumble is ever run unattended,
        deliberately drag over the SuperSwipe control on a disposable profile and confirm
        nothing is purchased."""
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

        Bumble places its paid SuperSwipe BETWEEN Pass and Like at the bottom of the card, so
        a placeholder or drifted coordinate can land on it. Whether that lands harmlessly or
        expensively is NOT governed by a confirmation step the way Hinge's Rose is — measured
        live on the device 2026-08-10, a SuperSwipe has two distinct outcomes depending on the
        account's SuperSwipe balance: with a non-zero balance (5, at measurement time) it is
        spent SILENTLY, no modal at all; only with a zero balance does a purchase/confirmation
        sheet appear first (see BUMBLE_SPEC's comment block and _handle_rose_upsell for the
        full two-state writeup). A drag begins in the middle of the card and cannot press a
        button it merely travels over, so the paid control is unreachable by construction
        rather than by careful aiming — this is the mechanism that actually protects the
        balance>0 case, since there is nothing downstream of the tap to catch a mistake there.

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
        """Dismiss a paid-upgrade interstitial — NEVER the paid option itself. Named for
        Hinge's "Send a Rose instead?" modal (which intercepts "Send Like" WHENEVER a Rose is
        available, since free Roses are granted periodically) but used generically by every
        like flow: we never spend a Rose, a SuperSwipe, or any other paid upsell (OWNER RULE —
        never automated, always manual). No-op when the modal isn't shown (or this app's spec
        declares no "upsell_dismiss" template at all — _template then returns None and
        _match_glyph's None-template guard reports no hits). Returns True if it dismissed a
        modal.

        Detection is ALWAYS the same: the "upsell_dismiss" template must match before this
        method taps anything at all. What happens after a match differs by app, selected by
        AndroidAppSpec.upsell_dismiss_zone:

          * zone is None (Hinge) — the template names a real, single, well-defined button
            ("Send Like anyway"); tap exactly its matched location. There is no coordinate or
            template anywhere in this driver for the PAID button beside/above it, so there is
            no code path that could tap one even by accident.

          * zone is set (Bumble, once calibrated) — there is no discrete dismiss BUTTON to
            match. Measured live on the device 2026-08-10: Bumble's SuperSwipe purchase sheet
            dismisses on a tap ANYWHERE in the dimmed area above it (x 0.000-1.000,
            y 0.000-0.394 — see BUMBLE_SPEC's comment block for the full measurement and the
            narrower, jitter-safe band actually declared). The template here identifies the
            SHEET (e.g. its heading), not a tap target; the actual dismiss tap goes through
            _dismiss_via_zone, which picks a FRESH random point inside the declared zone on
            every attempt, verifies the sheet actually cleared, and HALTS (PaidUpsellStuckError)
            rather than retrying forever if it doesn't.

        Ground truth from the owner, measured live on the device 2026-08-10: this whole
        method — and the confirmation sheet it dismisses — is NOT a general SuperSwipe safety
        net. It only ever runs AFTER a decide gesture has already landed, and Bumble's
        SuperSwipe has two distinct outcomes depending on the account's balance:

          * balance == 0 — the purchase sheet described above appears. No money has been
            spent yet; this method's job is to close that sheet without ever touching its
            "Get 30 SuperSwipes for $39.99" CTA (measured at x 0.049-0.950, y 0.899-0.951).

          * balance > 0 (5, at measurement time) — the SuperSwipe is spent SILENTLY. No
            sheet, nothing here to dismiss, nothing at all downstream to catch it. The real
            protections for THIS case are upstream of this method entirely: never aiming at
            the SuperSwipe control (decide_gesture="card_swipe" + forbidden_zones) and never
            firing a decide gesture without positively confirming the deck first
            (_require_deck_confirmed / UnconfirmedScreenError).

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
        if not hits:
            return False
        if self.spec.upsell_dismiss_zone is not None:
            self._dismiss_via_zone()          # random point in the safe band, verified, bounded
        else:
            self._tap(*hits[0])               # dismiss control — NEVER the paid button above/beside it
        return True

    def _dismiss_via_zone(self) -> None:
        """Dismiss an ALREADY-DETECTED paid-upgrade sheet by tapping a fresh random point
        inside spec.upsell_dismiss_zone, re-verifying after each attempt, and halting rather
        than retrying forever. Only ever called from _handle_rose_upsell, after ITS OWN
        template match already confirmed the sheet is up — this method never probes for the
        modal itself, so it structurally cannot fire against the ordinary deck.

        Why a FRESH random point every attempt, not the zone's centre or a fixed offset: a
        repeated exact coordinate is exactly the kind of bot signature the owner's
        no-fixed-constants rule forbids (see memory/auto-mode-uncapped-volume.md). Why a
        bounded retry-then-halt instead of looping until success: a dismiss tap that keeps
        landing just outside the sheet (drift, a device rotation, an app update that moved
        the layout) must not turn into an indefinite sequence of blind taps near a screen
        whose bottom third can be a purchase button (MEASURED 2026-08-10: Bumble's SuperSwipe
        CTA sits at y 0.899-0.951 — see BUMBLE_SPEC). Stopping and preserving the on-screen
        state for debugging is always safer than one more guess.
        """
        x0, y0, x1, y1 = self.spec.upsell_dismiss_zone
        w, h = self.adb.screen_size()
        template = self._template("upsell_dismiss")
        for _attempt in range(_UPSELL_DISMISS_MAX_ATTEMPTS):
            fx = random.uniform(x0, x1)       # fresh draw every attempt -- never a fixed point
            fy = random.uniform(y0, y1)
            self._tap(fx * w, fy * h)         # still runs through _assert_tap_allowed's guard
            time.sleep(human_cooldown(0.6))   # let the dismiss animation resolve
            frame = self._screencap(on_blank="none")
            if frame is not None and not _match_glyph(frame, template, side="any", threshold=0.6):
                return                        # sheet glyph is gone -> dismissed
        raise PaidUpsellStuckError(
            f"{self.spec.app}: a paid-upgrade sheet is still on screen after "
            f"{_UPSELL_DISMISS_MAX_ATTEMPTS} dismiss attempts inside its safe zone "
            f"{self.spec.upsell_dismiss_zone} — halting rather than tapping again blindly. "
            f"Repeated taps near a modal like this one are how a purchase gets confirmed.")

    def _require_deck_confirmed(self) -> None:
        """Refuse to issue a decide gesture (like/pass) unless the swipe deck is positively
        confirmed on screen. See UnconfirmedScreenError for the full reasoning: in short,
        forbidden_zones is screen-agnostic and cannot by itself tell a safe deck coordinate
        from the identical point on a paid-upgrade sheet, so this asks a different question
        first — "is this actually the deck" — using the same glyph-based readiness check
        _observe_deck_ready already uses passively for human-driven observe mode.

        A no-op when this app's spec declares no 'like' or no 'pass' template: there is
        nothing to prove readiness against yet (see UnconfirmedScreenError's docstring for
        why that's not a loophole — such a spec cannot be calibrated or run unattended
        anyway). Called from _deliver_decision, the single chokepoint both decide gestures
        (tap and card_swipe) go through, so this covers Bumble's card-drag path — which,
        unlike Hinge's vision-located tap, had NO perceptual check at all before this."""
        if "like" not in self.spec.templates or "pass" not in self.spec.templates:
            return
        frame = self._screencap()
        if self._observe_deck_ready(frame):
            return
        raise UnconfirmedScreenError(
            f"{self.spec.app}: refusing to decide — the swipe deck (like heart + pass X) is "
            f"not positively confirmed on screen. Something else may be up (a paid-upgrade "
            f"sheet, an ad, a dialog); firing a decide gesture at deck coordinates against an "
            f"unconfirmed screen is exactly how a like/pass tap lands on a different control "
            f"instead.")

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

        Also the ONLY signal this driver has that it is running an AUTONOMOUS session at
        all -- see self._auto_session's docstring in __init__ and open_session()'s touch-
        watcher gate. Worker._auto_loop calls this, unconditionally for any driver that
        defines it, before open_session(); Worker._observe_loop never does. So
        `self._auto_session = True` here is UNCONDITIONAL (unlike self._auto_policy right
        below, which stays app-gated to Hinge's calibrated behavior only) -- it just
        records which loop is driving this session, independent of which app it is or
        whether that app's policy object ends up used for anything.
        """
        self._auto_session = True
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
                        and math.isfinite(frac) and _READ_SCROLL_FRAC_MIN <= frac <= _READ_SCROLL_FRAC_MAX
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
                        and _READ_SCROLL_FRAC_MIN <= frac <= _READ_SCROLL_FRAC_MAX and 0.10 <= x_frac <= 0.90):
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

    def _interruptible_sleep(self, seconds: float, should_stop=None) -> bool:
        """Sleep `seconds`, but in slices, giving up early if `should_stop` fires.
        Returns True if the full time elapsed, False if it was cut short by a stop.

        Deliberately implemented with the MODULE-LEVEL time.sleep in a loop rather than a
        threading.Event.wait: every Hinge test file neutralises real time by monkeypatching
        `hinge.time.sleep` (four separate autouse fixtures, since there is no conftest.py).
        A wait primitive those fixtures cannot see would silently turn an offline suite that
        runs in ~100s into one that genuinely sleeps through 1.1s dwells x 11 per capture --
        the kind of regression that shows up as "tests got slow", never as a failure.

        With should_stop=None this is exactly time.sleep(seconds), one call, so every
        existing caller and every existing dwell-accounting test is unaffected.
        """
        if should_stop is None:
            time.sleep(seconds)
            return True
        # Bounded by a slice COUNT as well as by the deadline. The deadline alone is correct in
        # production but not under the four autouse fixtures that patch `hinge.time.sleep` to a
        # no-op: with sleeping removed, the deadline is the only thing left and the loop spins
        # on time.monotonic() burning CPU for the full wall-clock duration -- which is the exact
        # "the suite silently starts taking real time" regression routing this through
        # time.sleep was chosen to avoid, reintroduced by the loop around it. In production the
        # two bounds coincide (each slice really does sleep ~_OBSERVE_POLL_S); with sleep
        # patched out, the count is what makes this behave like the should_stop=None branch.
        deadline = time.monotonic() + seconds
        for _ in range(max(1, math.ceil(seconds / _OBSERVE_POLL_S))):
            if should_stop():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(_OBSERVE_POLL_S, remaining))
        return not should_stop()

    def _scroll_to_top(self, should_stop=None) -> bool:
        """Swipe the profile back to the top (content down) until it stops moving.

        Returns True only when the top was CONFIRMED — the loop ended because the view stopped
        moving. False means the swipe ceiling ran out first, or a stop interrupted it, i.e. the
        card may still be scrolled. Every pre-existing caller ignores this; it exists for
        _ensure_session_top, which needs to say honestly whether the invariant it is there to
        restore actually got restored.

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

        STOP: `should_stop` is polled before each undo-swipe, because this loop is the half of
        the ~85s stop-deaf window the operator actually SEES ("it completes the read by
        scrolling a bunch, then stops"). A stop here returns WITHOUT the counter/ledger reset
        below: those two lines are a CLAIM that the card is back at the top, and abandoning the
        undo half-way makes that claim false, so making it anyway would be recording something
        untrue about the device.

        Nothing in THIS process then goes on to act on the stale ledger — reaching this abort
        means the stop event is set, so the observe loop breaks and the session ends (only
        current_profile passes should_stop down here; the auto path's like()/`_locate_target_heart`
        unwinds never receive it and so can never be interrupted mid-commit). Correctness here is
        about not lying, and about `_note_capture_aborted` being able to tell the operator how
        many screens down the card was actually left — not about a live caller that would
        otherwise be misled. A phone left mid-scroll is the documented "a stop leaves the screen
        untouched for debugging" outcome; `_ensure_session_top` is what makes the NEXT session
        safe, since a stale ledger cannot survive process exit anyway.
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
        settled = False
        for _ in range(max_swipes):
            if should_stop is not None and should_stop():
                return False               # see STOP above: do NOT fall through to the reset
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
            if not self._interruptible_sleep(human_delay(0.3), should_stop):
                return False               # stop landed inside the settle wait; same rule as above
            if not self._changed(before, self._screencap()):
                settled = True
                break
        self._capture_scrolls = 0
        self._capture_scroll_ledger = []   # confirmed (or ceiling-bounded) back at top
        return settled

    def _ensure_session_top(self, should_stop=None) -> None:
        """Once per session, before the first capture, put the card at a CONFIRMED scroll-top.

        _capture_current's identity anchor rests on one invariant: frame 0 of a capture is at
        scroll-top, so the band it seeds `_identity_top_sig` from is the app's own
        profile-independent chrome (Hinge's filter-chips row) rather than a real person's
        sticky header. Within a session that invariant is maintained by current_profile()'s
        own trailing _scroll_to_top(). ACROSS sessions nothing maintained it at all --
        open_session() only foregrounds the app (`monkey ... LAUNCHER 1`), which restores
        whatever scroll position the card was left at.

        That gap was harmless while every stop happened between profiles, and it stopped being
        harmless the moment Stop could abandon a read mid-scroll (see _capture_current): the
        ordinary Stop -> Start flow then starts the next run mid-card, `_identity_top_sig` gets
        seeded with THAT PERSON's header, `_identity_sig`/`_identity_name` never get set at all,
        and layer 1 is silently dead for the first card of the run -- the one screen an operator
        is most likely to scroll through before deciding. (It was reachable before, too, just
        rarely: a mid-capture profile split or any exception mid-read leaves the card scrolled
        the same way.)

        Cost when the card is already at the top, which is the normal case: one downward swipe
        and two screencaps (~3s), once per run -- _scroll_to_top's settle check ends the loop on
        the first iteration. The ceiling below is what makes this work from an UNKNOWN position:
        _scroll_to_top sizes its swipe budget from the scroll ledger, which is empty at session
        start, so without this it would throw exactly one swipe and give up. `_capture_scrolls`
        is the documented pre-ledger compatibility input for exactly that (see _scroll_to_top's
        COUNT paragraph), and the worst case it must cover is a previous run's own read ceiling,
        hence scroll_captures.

        Not fatal when it fails: a card that never settles (an animated/video profile) leaves
        the invariant unrestored, and _identity_of's id_sig-is-None branch is what keeps that
        honest -- it can answer 'top' or 'unknown', never a manufactured 'new'.
        """
        # An open like/comment sheet is not a card, and swiping under one would drag the sheet
        # the operator is composing in. _capture_current already refuses to read or scroll on a
        # sheet for exactly this reason ("makes a capture that STARTS on a sheet completely
        # input-free") -- this pass runs BEFORE that check, so it has to make the same promise
        # itself or it would silently break it. Deliberately does NOT mark the session done:
        # nothing was restored, so the next capture (after the sheet closes) should try again.
        frame = self._screencap(on_blank="none")
        if frame is not None and self._observe_like_sheet_visible(frame):
            return
        self._session_top_done = True     # one attempt per session, even if it fails below
        self._capture_scrolls = self.scroll_captures
        self._capture_scroll_ledger = []
        if self._scroll_to_top(should_stop):
            return
        if should_stop is not None and should_stop():
            return                        # interrupted, not failed -- the run is ending anyway
        print(f"{self.spec.app}: could not confirm the card is scrolled to the top at session "
              f"start (it never stopped moving). The first profile's identity anchor may be "
              f"unavailable; decisions on it fall back to content matching.")
        if self._dbg is not None:
            try:
                self._dbg.action("session_top_unconfirmed", swipe_ceiling=self.scroll_captures)
            except Exception:  # noqa: BLE001 — debug logging must never break a session
                pass

    def _note_capture_aborted(self, frames: int) -> None:
        """Record that a profile read was abandoned because Stop was requested.

        Written to the debug log so actions.jsonl says why a `capture` record the reader was
        expecting never appeared -- without it, an interrupted run's log simply ends after the
        previous decision, which reads identically to a crash or a wedged device. Deliberately
        takes NO screenshot (unlike _dbg_action, which grabs a fresh screencap for every
        record): this fires on the shutdown path, where the entire point is to stop touching
        the device, and one more ~1s ADB round-trip per abort would eat into the very latency
        this change exists to remove. The frame count is the useful part anyway -- it says how
        far into the read the stop landed.
        """
        scrolled = len(self._capture_scroll_ledger)
        where = (f" The card is left scrolled down {scrolled} screen(s) — that is deliberate "
                 f"(a stop leaves the screen untouched); scroll it back to the top yourself "
                 f"before the next run so the first capture starts from a known position."
                 if scrolled else "")
        print(f"{self.spec.app}: stop requested while reading this profile — abandoning the "
              f"read after {frames} frame(s); nothing was recorded for it.{where}")
        if self._dbg is not None:
            try:
                self._dbg.action("capture_aborted", frames=frames,
                                 read_scrolls=len(self._capture_scroll_ledger),
                                 profile_name=self._identity_name)
            except Exception:  # noqa: BLE001 — debug logging must never break a shutdown
                pass

    # --- capture (Signals #1: read the whole profile, human-paced) ------
    def _capture_current(self, should_stop=None) -> Profile | None:
        """Read the profile currently on screen, human-paced, returning its frames.

        STOP: `should_stop` (when supplied -- see DatingAppDriver.supports_interruptible_capture)
        is polled at the top of every read iteration and inside the read dwell, and a stop
        abandons the read and returns None. Before this, the ONLY stop check between one profile
        and the next lived in worker.py, on the line AFTER this call returned -- so a Stop pressed
        while a profile was being read was not seen until the whole read finished: 12 screencaps,
        11 humanized read-scrolls, and then a full scroll back to the top, measured at ~85s of
        visible scrolling after the operator asked it to stop.

        Abandoning a read costs nothing: no decision has been made, nothing has been written, and
        None is already this method's established "nothing usable here" answer (a like sheet is
        up, a static screen, a mid-capture profile split), which both worker loops handle by
        recapturing -- and which, with the stop event set, means they simply leave their loop.
        The partial frames are dropped rather than returned precisely because a half-read profile
        must never reach the ranker as if it were a whole one.
        """
        photos: list[bytes] = []
        self._current_sigs = []
        self._current_capture_truncated = False   # reset: see the for/else below
        self._capture_scrolls = 0     # reset: _scroll_to_top must undo THIS capture, not a stale one
        self._capture_scroll_ledger = []
        # Identity anchor (see _identity_of): reset every capture, unconditionally, so a
        # profile with no identity_band declared or one that never scrolls far enough to
        # reveal the sticky header never inherits a STALE signature from the profile before
        # it -- that would be worse than no anchor at all (a false 'same' against the wrong
        # profile).
        self._identity_top_sig = None
        self._identity_sig = None
        self._identity_name = None
        # Same reasoning as the three resets just above: a cached OCR read is keyed on frame
        # bytes, not profile identity, so a coincidental byte-identical crop from the NEXT
        # profile (unlikely but not impossible for a mostly-blank band) could otherwise return
        # a stale answer. See _ocr_band_cache's own comment in __init__.
        self._ocr_band_cache = {}
        self._current_capture_split = False   # set if the deck advanced mid-capture; see the loop
        self._profile_capture_limit = self._capture_limit_for_profile()
        read_dwell_s_total = 0.0
        seen = set()
        for i in range(self._profile_capture_limit):
            if should_stop is not None and should_stop():
                # Checked BEFORE the screencap, so a stop that lands during the previous
                # dwell/scroll costs one poll rather than another full ADB round-trip. The
                # partial read is discarded (see the docstring); the scroll ledger is left
                # alone so _scroll_to_top still knows how far down the card actually is.
                self._note_capture_aborted(len(photos))
                return None
            frame = self._screencap()
            # A comment sheet is not profile content.  In observe mode it can be left open
            # while Hinge is still composing/sending a like; treating its changing pixels as
            # a card and then read-scrolling would move the sheet underneath the operator.
            # Stop before recording this frame or issuing another scroll.  In particular this
            # makes a capture that STARTS on a sheet completely input-free.
            if self._observe_like_sheet_visible(frame):
                return None
            # A profile boundary reached MID-CAPTURE. The deck can advance while this loop is
            # still reading -- Hinge draws no on-screen busy overlay (worker.py's WAIT cue
            # lives on the hub, not the phone), and the human's finger and the bot's own
            # digitizer are concurrent input streams (see touchwatch.py) -- so a tap landing
            # here is a real possibility, not a hypothetical. Nothing else in this loop would
            # notice: it stops only on a repeated frame or the ceiling, so the frames after
            # the advance get appended as though they were more of the SAME person, and the
            # returned Profile then carries two different people's photos into one mean-pooled
            # embedding and one stored label. Once this capture has locked an identity, a
            # frame whose header matches NEITHER that identity nor the scroll-top chrome is a
            # different card: stop before appending it and mark the capture, so the worker
            # discards and recaptures instead of scoring a chimera.
            if self._identity_sig is not None:
                band = _band(frame, self.identity_band)
                if (band is not None
                        and _band_dist(band, self._identity_sig) >= self.change_threshold
                        and (self._identity_top_sig is None
                             or _band_dist(band, self._identity_top_sig) >= self.change_threshold)):
                    self._current_capture_split = True
                    break
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
            # Lock the identity anchor INLINE as frames arrive, not retroactively after the
            # loop. The boundary check above can only fire once _identity_sig exists, and a
            # post-hoc scan would establish it only after the foreign frames had already been
            # appended -- i.e. exactly too late to keep them out.
            if self.identity_band is not None:
                band = _band(frame, self.identity_band)
                if band is not None:
                    if self._identity_top_sig is None:
                        self._identity_top_sig = band     # frame 0: the app's scroll-top chrome
                    elif (self._identity_sig is None
                            and _band_dist(band, self._identity_top_sig) >= self.change_threshold):
                        self._identity_sig = band         # first frame showing the sticky header
                        self._identity_name = self._ocr_band(frame, self.identity_band)
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
                # The read dwell is the single longest stretch of this loop (dwell_s=1.1
                # humanized, x11), so it is where a Stop most often lands. Credit
                # read_dwell_s_total only with time actually spent: this counter is the
                # Signals behaviour-#1 "did we really read the profile" metric, and an
                # interrupted capture that reported the full sampled dwell would be claiming
                # reading time that never happened.
                started = time.monotonic()
                completed = self._interruptible_sleep(dwell, should_stop)
                read_dwell_s_total += dwell if completed else max(0.0, time.monotonic() - started)
                if not completed:
                    self._note_capture_aborted(len(photos))
                    return None
                self._scroll_down_one(frac, x_frac)
        else:
            # The loop ran out its full range() without ever finding a repeated frame (the
            # signal that the profile's true bottom was reached) -- this profile has MORE
            # content than self._profile_capture_limit could read. _current_sigs then does not
            # cover the whole scrollable range, so a human reading further down than the bot
            # did can land on pixels _vertical_shift_match has genuinely never seen (not a
            # granularity gap this project can shift-search its way out of). Recorded for
            # wait_for_decision's PASS diagnostic and Profile.meta below -- monitoring only,
            # never changes what gets decided.
            self._current_capture_truncated = bool(photos)
        # H1: in a real run (open_session validated PIL/numpy), every frame should
        # downsample. If none did, decode is broken at runtime (PIL/numpy failure OR a wedged
        # device returning empty/truncated screencap) — refuse to continue in a degraded mode
        # where scroll-detection is off and manual scrolls mislabel as PASS / the worker
        # silently no-ops forever. (The decodable-static case returned None above.)
        if self._observe_ready and photos and not any(s is not None for s in self._current_sigs):
            raise DriverClosed(
                "screencap frames could not be decoded (PIL/numpy runtime failure or wedged "
                "device); refusing to run degraded — it would corrupt training labels")
        # Identity anchor (layer 1 of the observe-mode redesign): current_profile()/next_profile()
        # both go through here, and _scroll_to_top() puts the PREVIOUS capture at the top before
        # this one starts reading -- so photos[0] is guaranteed to be at scroll-top, i.e. showing
        # the app's own scroll-top chrome (Hinge's filter-chips row) in the identity band, never a
        # real per-profile header. The sticky header only appears once the card is scrolled, so
        # the true identity signature is the first LATER frame whose band reads as meaningfully
        # different from that chrome (>= change_threshold) -- see HINGE_SPEC's identity_band
        # comment for the measured 0.00-vs-17.95 separation this relies on. Left None when the
        # profile was never scrolled far enough during THIS capture to reveal it; _identity_of
        # then reports 'unknown' rather than manufacturing a false 'same'/'new' from nothing.
        identity_seen = self._identity_sig is not None    # locked inline in the loop above
        if self._current_capture_split:
            # This capture spans two profiles: the deck advanced part-way through the read.
            # Returning the frames anyway would hand worker.py a Profile whose photos are two
            # different people, which it would mean-pool into ONE embedding and store against
            # ONE label -- silently poisoning the training set with a face that does not exist.
            # Return None, the same signal current_profile() already uses for "nothing usable
            # here"; worker.py's _observe_loop treats it as `continue` and recaptures the card
            # that is actually on screen now (worker.py:215-216), which is exactly right.
            if self._dbg is not None and photos:
                self._dbg.action("capture_split", before=photos[0], photos=len(photos),
                                 profile_name=self._identity_name)
            print(f"{self.spec.app}: the deck advanced while reading this profile "
                  f"(captured {len(photos)} frame(s) spanning two cards); discarding and "
                  f"recapturing rather than mixing two people into one label.")
            return None
        if self._dbg is not None and photos:
            # profile_name, not name: DebugLog.action's own first positional parameter IS
            # called `name` (the action-type string, "capture" here) -- a fields key of
            # literally `name` would collide with it (TypeError: multiple values for
            # argument 'name'). Same reason wait_for_decision's debug records below use
            # profile_name too.
            self._dbg.action("capture", before=photos[0], photos=len(photos),   # first frame = who was scored
                             capture_truncated=self._current_capture_truncated,
                             identity_seen=identity_seen, profile_name=self._identity_name)
        return Profile(
            photos=photos,
            prompts=[],
            meta={
                "app": self.spec.app,
                "capture_frames": len(photos),
                "read_scrolls": len(self._capture_scroll_ledger),
                "read_dwell_s_total": read_dwell_s_total,
                "capture_truncated": self._current_capture_truncated,
            },
        )

    def next_profile(self, *, should_stop=None) -> Profile | None:
        if self.out_of_profiles():
            return None
        if not self._session_top_done:
            self._ensure_session_top(should_stop)   # see its docstring: once per session
        return self._capture_current(should_stop)

    def current_profile(self, *, should_stop=None) -> Profile | None:
        # Observe mode's only capture path: worker.py prints "READY - swipe this profile"
        # right after this returns, so the phone must be back at the top for that swipe to
        # land on the card the operator actually read (bug 2). next_profile() (auto) does NOT
        # get this: like() already calls _scroll_to_top() itself before acting, so adding it
        # here too would just be a redundant extra scroll on that path.
        if self.out_of_profiles():
            return None
        if not self._session_top_done:
            self._ensure_session_top(should_stop)   # see its docstring: once per session
        profile = self._capture_current(should_stop)
        if profile is not None:
            # Also stop-aware: this unwind is roughly half of the total per-profile dead time
            # (11 undo-swipes, each with two settle screencaps), and it is the part the
            # operator actually watches happen after pressing Stop. Reaching it at all means
            # the read itself completed, so the profile IS returned and worker.py's own stop
            # check on the next line decides what happens to it -- an interrupted unwind never
            # discards a complete capture, it only declines to keep scrolling.
            self._scroll_to_top(should_stop)
        return profile

    def out_of_profiles(self) -> bool:
        # LIVE-VERIFY: the empty-deck screen is gated until the profile is finished,
        # so we can't yet match its signature. Until calibrated this returns False
        # and the run stops via the rate limiter (auto) or the operator (observe).
        return False

    def _locate_target_heart(self, item_index: int) -> tuple[tuple[int, int], bool]:
        """comment_sheet flow only. Locate the heart of the photo/prompt the opener is about.
        item_index is the 0-based index (capture order) the opener returned. We re-navigate to
        that captured frame by matching its downsample signature, then take its heart. Falls
        back to the topmost heart (first photo) when targeting isn't possible (index 0, no
        sigs, no match) — so it is never worse than the old 'always first photo' behavior.
        Every genuine fallback (as opposed to the ordinary index-0 fast path) is recorded via
        _dbg_action so a bug report can tell "target not found" apart from "no sigs to search
        at all" (HINGE-05). (LIVE-VERIFY during observe seeding.)

        Returns `(point, on_target)`. `on_target` is what this method's whole contract is FOR:
        whether `point` actually IS the item the opener was written about, or a first-photo
        fallback that merely has the same shape as a real answer. When it's False, the caller
        is one tap away from attaching a comment written about item `item_index` to item 0
        instead — the exact out-of-place-message failure this feature exists to catch (an
        opener about a beach photo landing under a dining-table photo) — and `_like_comment_sheet`
        must repair the mismatch (see its `anchored_opener` parameter) rather than ship it.
        index 0 is the one case where "fell back to the first photo" and "the opener genuinely
        was about the first photo" are the same outcome, so that fast path is on target by
        construction; every OTHER path that reaches the first-photo fallback below is a real
        miss, because the opener was written about a different item than the one about to be
        tapped."""
        sigs = getattr(self, "_current_sigs", None)
        if not sigs or item_index <= 0 or item_index >= len(sigs) or sigs[item_index] is None:
            if sigs and item_index > 0:                # a genuine fallback, not the index-0 fast path
                self._dbg_action("locate_target_heart", self._snap(), item_index=item_index,
                                  outcome="fallback", reason="out_of_range_or_undecodable_target")
            # item_index == 0 -> index 0 legitimately IS the topmost heart, so this fast path
            # lands on exactly the item the opener was about (on target). Anything else that
            # reaches this branch (no sigs, out-of-range, or an undecodable target signature)
            # falls back to the first photo despite the opener being about a DIFFERENT item —
            # not on target.
            #
            # `== 0`, not `<= 0`, even though the branch condition above is `<= 0`: a NEGATIVE
            # index is invalid input, not the index-0 fast path, and the two must not report
            # the same verdict. opener.py clamps referenced_index with max(0, ...) so the live
            # path can't produce one today, but the honest answer for an index we cannot honour
            # is "off target" — that routes it into the anchored_opener repair (see
            # _like_comment_sheet), which re-reads the like screen and rewrites the message
            # against whatever the heart actually landed on. Claiming on_target for a nonsense
            # index would instead ship the original text unchecked, which is precisely the
            # silent mismatch this flag exists to prevent.
            return self._await_button("like"), item_index == 0
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
                    return hearts[0], True            # the referenced item's heart, now in view
                matched_frame_no_heart = True
                break
            self._scroll_down_one()          # tracked, so the fallback _scroll_to_top() below (if it
            time.sleep(human_delay(self.dwell_s * 0.4))  # comes to that) undoes exactly these scrolls too
        reason = "heart_not_visible_on_matched_frame" if matched_frame_no_heart else "target_frame_not_found"
        self._dbg_action("locate_target_heart", before, item_index=item_index,
                          outcome="fallback", reason=reason)
        self._scroll_to_top()                         # no match: reset and take the first photo
        time.sleep(human_delay(0.3))
        return self._await_button("like"), False

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
    def like(self, opener: str | None = None, item_index: int = 0, *,
             anchored_opener: Callable[[bytes], str | None] | None = None) -> None:
        if self.spec.like_flow == "comment_sheet":
            self._like_comment_sheet(opener, item_index, anchored_opener=anchored_opener)
        else:
            self._like_direct(opener, item_index, anchored_opener=anchored_opener)

    def _like_comment_sheet(self, opener: str | None, item_index: int, *,
                            anchored_opener: Callable[[bytes], str | None] | None = None) -> None:
        """Hinge's flow: heart -> comment/"Send Like" sheet opens -> optionally type the
        opener into the comment box (Signals #2: the opener is sent WITH the like) -> tap
        Send -> handle a paid-upsell interstitial (never tap the paid option) -> verify.

        `anchored_opener`, when given, is the repair path for a targeting miss. On Hinge a
        like's comment is attached to ONE specific photo or prompt — every photo AND every
        prompt card has its own heart — and the opener writer picks which item it's about
        before this method ever runs. `_locate_target_heart` usually lands the tap on that
        exact item, but when it can't (see its `on_target` return value) the tap falls back
        to the first photo instead, and the comment sheet that just opened is now anchored to
        a DIFFERENT item than the one the opener text describes — an obviously out-of-place
        message (a beach-photo opener landing under a dining-table photo) shipped with full
        confidence. `anchored_opener` is called with a screencap of the open sheet itself
        (which visually shows the item the comment will attach to) and returns replacement
        opener text grounded in what that screencap actually shows, so the message that gets
        typed matches the item the heart tap actually landed on."""
        self._scroll_to_top()
        time.sleep(human_delay(0.4))
        heart, on_target = self._locate_target_heart(item_index)  # heart of the photo the opener is about
        before = self._snap()                         # baseline AFTER navigation: the pre-tap card
        self._tap(*heart)                             # opens the comment / "Send Like" sheet
        time.sleep(human_cooldown(0.8))               # sheet animates in; you read/think
        self._await_sheet_open()                      # gate: the fixed taps below are only
                                                       # valid while the sheet is actually up
        anchor = self._screencap()          # the like screen: shows the item this comment attaches to
        reasked = False
        # Re-ask ONLY when targeting missed (`not on_target`) — never unconditionally, even
        # though `anchored_opener` is available on every call. Each re-ask is a SECOND billed
        # provider request against a free tier whose per-minute cap can be as low as 5 (see
        # the Gemini opener migration notes); when targeting succeeded, the sheet is already
        # anchored to the very item the opener was written about, so a second call would buy
        # nothing but spend budget the run may not have. Residual, deliberately-accepted risk,
        # stated honestly rather than papered over: `_locate_target_heart` takes the TOPMOST
        # heart on the matched frame, so a scroll position that shows two items in one frame
        # (a photo AND a prompt, say) can still land the tap on the neighbour even when
        # on_target reports True — that miss is invisible to this check. Observe mode reads
        # the real like screen and is exact by construction (a human tapped it); auto mode is
        # best-effort here.
        if opener and not on_target and anchored_opener is not None:
            replacement = anchored_opener(anchor)
            if not replacement or not str(replacement).strip():
                # Never send a commentless like, and never silently ship a comment attached to
                # the wrong photo/prompt (both owner rules). Heart targeting already fell back
                # to the first item, so the opener written about item item_index no longer
                # matches what the comment is about to attach to — and the replacement request
                # that was supposed to fix that produced nothing usable either. The only choice
                # left that doesn't violate one of those two rules is to not send the like at
                # all. The sheet is intentionally left open on screen for debugging, consistent
                # with the rest of this driver's halt behaviour (see HingeActionError).
                raise HingeActionError(
                    f"{self.spec.app}: heart targeting fell back to the first item, so the "
                    f"opener written about item {item_index} no longer matches what the "
                    f"comment attaches to, and the replacement (anchored) opener request "
                    f"produced no usable opener — so the like is deliberately NOT sent. Never "
                    f"send a commentless like; never silently ship one attached to the wrong "
                    f"photo/prompt. The sheet is left open on screen for debugging."
                )
            opener = str(replacement)
            reasked = True
        if self._dbg is not None:
            # Best-effort, never raising (DebugLog.action swallows its own I/O failures) — a
            # broken debug log must never take down a like that would otherwise send cleanly.
            self._dbg.action("like_anchor", before=anchor, item_index=item_index,
                             on_target=on_target, reasked=reasked)
        if opener:
            self._tap_frac(self.coords["comment_box"])
            time.sleep(human_delay(0.5))
            self.adb.text(opener)                     # opener sent WITH the like (Signals #2)
            time.sleep(human_delay(0.6))
        self._tap_frac(self.coords["send_like"])
        time.sleep(human_cooldown(0.6))               # let the send register / upsell modal animate in
        rose = self._handle_rose_upsell()             # paid-upsell interstitial: dismiss, NEVER pay
        self._dbg_action("like", before, heart=list(heart), opener_chars=len(opener or ""), rose_modal=rose,
                         on_target=on_target, anchor_reask=reasked)
        self._verify_like_landed(before)

    def _deliver_decision(self, decision: str):
        """Issue one like/pass, by whichever gesture this app's spec calls for.

        Returns the tapped point, or None when the decision was delivered as a card drag
        (there is no single point to log in that case). Both paths are equally humanized;
        they differ only in what the phone receives, and therefore in what can go wrong:
        a tap can land on a neighbouring control, a drag cannot.

        _require_deck_confirmed() runs FIRST, unconditionally, for both gestures — this is
        the single chokepoint every autonomous decide passes through (see
        UnconfirmedScreenError), so a card_swipe app (Bumble) gets the same "prove this is
        the deck before acting" guarantee a tap app gets implicitly from vision-locating its
        button."""
        self._require_deck_confirmed()
        if self.spec.decide_gesture == "card_swipe":
            self._decide_by_card_swipe(decision)
            return None
        point = self._await_button("like" if decision == "like" else "pass")
        self._tap(*point)
        return point

    def _like_direct(self, opener: str | None, item_index: int, *,
                     anchored_opener: Callable[[bytes], str | None] | None = None) -> None:
        """Bumble's flow: one like, no comment sheet, no per-item targeting.
        `opener`/`item_index`/`anchored_opener` are accepted only for interface parity with
        the comment_sheet flow and are otherwise unused — accepts_opener is False for every
        spec using this flow (Bumble is match-first-then-message, so there is no swipe-time
        opener to attach), so worker.py never actually passes a real `opener` here, and there
        is no comment sheet here to screenshot in the first place, so `anchored_opener` has
        nothing to anchor a replacement opener against even if it were called."""
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

    # --- observe mode: identity anchor + gesture corroboration ---------
    # See the module docstring's redesign notes and HINGE_SPEC's identity_band/content_band/
    # observe_ignore_zones/observe_touch_watch comments for the measured ground truth these
    # methods are built on. Everything below is READ-ONLY perception; nothing here taps,
    # swipes, or types (see wait_for_decision's own contract, unchanged).

    _OCR_NAME_RE = re.compile(r"[^A-Za-z0-9 '\-]")
    # Cache sizing for _ocr_band's memo below (see that method's own comment for the measured
    # cost it exists to avoid). A sentinel object, not None, marks "not cached" -- a genuine
    # OCR miss is itself represented as a cached None, and dict.get needs a default that can
    # never collide with a real stored value to tell the two apart.
    _OCR_BAND_CACHE_MAX = 4
    _OCR_BAND_CACHE_MISS = object()

    def _ocr_band(self, frame: bytes, rect: tuple[float, float, float, float], *,
                  psm: str = "7") -> str | None:
        """Best-effort OCR of `rect` via the host's `tesseract` binary (MEASURED reliable
        recipe on the Pixel 7a 2026-08-10: crop tightly -- a wider crop that includes photo
        content makes tesseract's page segmentation fail -- upscale 3x LANCZOS, `tesseract
        stdin stdout --psm <psm>`).

        `psm` defaults to `"7"` (treat the crop as a single text line), which keeps every
        existing call site -- the sticky-header identity_band, whose whole point is that it
        renders the name alone on one line -- byte-identical to before this parameter existed.
        identity_top_name_band's card-header crop is a DIFFERENT shape: it holds two or three
        short lines (the name, then a "Signals Active today" row, depending on layout -- see
        that field's own comment), and `--psm 7` MEASURED empty or garbled output against it
        on the Pixel 7a 2026-08-10 -- forcing a single line onto multi-line content confuses
        tesseract's segmentation. `--psm 6` ("assume a single uniform block of text") is what
        actually reads it correctly, on every scroll-top frame tested, in both layouts Hinge
        renders. Callers pass `psm="6"` explicitly for that band; nothing here chooses it
        automatically, because doing so from `rect` alone would silently couple this method's
        behavior to which band happens to be which shape today, instead of to a measured fact
        the caller already knows.

        NEVER load-bearing: the pixel-band signature `_identity_of` computes from `_band` is
        the authoritative identity anchor on its own. This exists only because a pixel
        signature is POSITION-sensitive (a header that shifts a few px between frames reads
        as a mismatch) where a name string is not -- so when both names are available they
        corroborate the pixel check with something more tolerant. Any failure here (no
        `tesseract` on PATH, a decode error, a garbled read, a timeout) returns None and the
        caller falls straight back to the pixel band; this method must never raise.

        MEMOIZED, bounded to `_OCR_BAND_CACHE_MAX` entries, keyed on `(rect, psm, sha1(frame))`.
        MEASURED on this machine 2026-08-10: ~103ms per call at psm 7 (the sticky identity_band)
        and ~110ms at psm 6 (the card-header identity_top_name_band) -- and both bands can be
        OCR'd on the SAME poll (the identity_band corroboration above Layer 1b, plus Layer 1b
        itself), against an `_OBSERVE_POLL_S` of 0.35s. That is up to ~220ms of subprocess time
        inside a 350ms poll -- a 60% duty cycle that can stall the loop and cost frames. The
        repeat this cache targets is common, not hypothetical: wait_for_decision's own
        settle/confirm frame 0.5s later is usually the identical screen, and
        `_is_current_profile_frame` re-enters `_identity_of` (and so this method) on frames
        already seen earlier in the same poll. A cache hit skips the PIL crop/resize AND the
        tesseract subprocess entirely -- it is a pure memo, never a behavior change: identical
        `(rect, psm, frame bytes)` always yields whatever the first real OCR of that input
        produced (including a cached `None` miss), and any input not seen before takes the
        exact same path as before this cache existed.
        """
        if not self.observe_name_ocr:
            return None
        tesseract = shutil.which("tesseract")
        if tesseract is None:
            return None
        cache_key = (rect, psm, hashlib.sha1(frame).digest())
        cached = self._ocr_band_cache.get(cache_key, self._OCR_BAND_CACHE_MISS)
        if cached is not self._OCR_BAND_CACHE_MISS:
            return cached
        try:
            from io import BytesIO

            from PIL import Image
            im = Image.open(BytesIO(frame)).convert("L")
            w, h = im.size
            x0, y0, x1, y1 = rect
            crop = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
            crop = crop.resize((max(1, crop.width * 3), max(1, crop.height * 3)), Image.LANCZOS)
            buf = BytesIO()
            crop.save(buf, format="PNG")
            result = subprocess.run(
                [tesseract, "stdin", "stdout", "--psm", psm],
                input=buf.getvalue(), capture_output=True, timeout=5.0,
            )
            text = self._OCR_NAME_RE.sub("", result.stdout.decode("utf-8", "replace"))
            cleaned = " ".join(text.split())
            value = cleaned or None
        except Exception:  # noqa: BLE001 — OCR is best-effort, never load-bearing
            value = None
        self._ocr_band_cache[cache_key] = value
        if len(self._ocr_band_cache) > self._OCR_BAND_CACHE_MAX:
            self._ocr_band_cache.pop(next(iter(self._ocr_band_cache)))   # evict oldest (FIFO)
        return value

    def _identity_of(self, frame: bytes) -> tuple[str, float | None]:
        """('same' | 'new' | 'top' | 'unknown', distance).

        'same'    — the app's sticky per-profile header matches the captured profile. Whatever
                    else moved on screen, this is still the same card: never a decision. THIS
                    is the fix for the reported bug (a human scrolling to read a profile, no
                    tap at all, recorded as a PASS) -- wait_for_decision checks this FIRST,
                    before any pixel-delta reasoning, on every poll.
        'top'     — the band shows the app's own scroll-top chrome (profile-independent), so
                    identity is simply not visible right now; the caller falls back to content
                    matching (_vertical_shift_match) rather than treating this as a mismatch.
                    UNLESS identity_top_name_band is declared and can resolve it by OCR'ing
                    the card header instead -- see the "Layer 1b" block below, added for the
                    incident where a pass at scroll-top (Alina -> jessica) had no name visible
                    in identity_band and fell through to a spurious content match.
        'new'     — the band shows a DIFFERENT profile's header.
        'unknown' — no identity_band declared for this app, the frame didn't decode, or this
                    profile never scrolled far enough during capture to reveal its own header
                    (self._identity_sig is still None) -- there is nothing to compare against
                    either way, so this must not be read as either 'same' or 'new'.

        `distance` is the mean abs diff (0..255) the verdict was decided on, or None when
        there was nothing to compare (declared for logging -- see wait_for_decision's
        observe_decision/observe_resync records).
        """
        # Reset on EVERY call, not just every profile: a call that doesn't reach the Layer 1b
        # block below (wrong state, OCR off, band not declared, ...) must not leave a stale
        # read from an EARLIER call sitting here for the next observe_scroll/observe_waiting
        # debug record to misattribute to a frame it was never actually read from. Same for
        # _identity_top_name_verdict: wait_for_decision reads it immediately after calling this
        # method, so a stale value from an earlier call must never survive to be misread as
        # THIS call's provenance.
        self._identity_top_name_read = None
        self._identity_top_name_verdict = None
        if self.identity_band is None:
            return "unknown", None
        band = _band(frame, self.identity_band)
        if band is None:
            return "unknown", None

        top_sig, id_sig = self._identity_top_sig, self._identity_sig
        state, dist = "unknown", None
        if id_sig is not None:
            dist = _band_dist(band, id_sig)
            if dist < self.change_threshold:
                state = "same"
            elif top_sig is not None and _band_dist(band, top_sig) < self.change_threshold:
                state = "top"
            else:
                state = "new"
        elif top_sig is not None:
            # No per-profile header was ever locked for this capture (self._identity_sig is
            # None), so the ONLY thing the scroll-top chrome can positively prove is 'top':
            # this frame IS a card's scroll-top. It cannot prove 'new'. A frame that merely
            # differs from the chrome is any scrolled frame of ANY card -- including the very
            # card being watched -- so answering 'new' there manufactures a verdict from
            # nothing, which is exactly what this method's own docstring forbids for the
            # id_sig-is-None case ("there is nothing to compare against either way, so this
            # must not be read as either 'same' or 'new'") and what the OCR block below
            # refuses to do on much stronger evidence ("OCR gets a veto on 'new' and no power
            # to create one").
            #
            # It is not academic: 'new' is the one verdict that SKIPS layer 2 entirely (see
            # wait_for_decision's `identity_state != "new"` guard), so a manufactured 'new'
            # goes straight to the deck-ready/settle check and can record a PASS for a human
            # who only scrolled. Two ways to get here with no id_sig: a profile short enough
            # that the sticky header never appeared during its capture (latent since the
            # identity redesign), and a capture whose frame 0 was not at scroll-top -- which a
            # Stop abandoned mid-read now makes reachable for the FIRST card of the next run
            # (see _ensure_session_top, which exists to stop that happening in the first
            # place; this is the belt to its braces).
            #
            # 'unknown' is the honest answer and is NOT a weaker outcome for a genuine card
            # advance: it falls through to layer 2's content match, and a real advance matches
            # nothing there and still reaches the same deck-ready + settle proof, corroborated
            # rather than assumed.
            dist = _band_dist(band, top_sig)
            state = "top" if dist < self.change_threshold else "unknown"

        # OCR corroboration: position-tolerant where the pixel band above is not, so it can
        # recognise the SAME profile through a header that shifted a few px (a banner
        # appearing/disappearing) and would therefore fail the pixel compare.
        #
        # Deliberately ASYMMETRIC: a name match can only ever upgrade the verdict TO 'same',
        # never downgrade one to 'new'. The two errors are not equally costly. A false 'same'
        # costs at most a missed pass -- the loop keeps waiting, and the deck-ready + settle +
        # content checks downstream still have to agree before anything is recorded. A false
        # 'new' writes a WRONG TRAINING LABEL, which is the entire class of bug this redesign
        # exists to eliminate. And a mismatch here is a genuinely weak signal for 'new': the
        # band legitimately reads as the app's filter-chips row at scroll-top (measured
        # 2026-08-10: OCRs as "Signals ( Agev ) Height v", which matches no name and would
        # have flipped a correct 'top' verdict straight to 'new'), and tesseract garbles
        # perfectly ordinary names often enough that a non-match proves nothing on its own.
        # So OCR gets a veto on 'new' and no power to create one.
        if self.observe_name_ocr and self._identity_name and state != "same":
            seen_name = self._ocr_band(frame, self.identity_band)
            if seen_name:
                stored = self._identity_name.strip().casefold()
                seen = seen_name.strip().casefold()
                if stored and seen and seen == stored:
                    state = "same"

        # Layer 1b: resolve a "top" verdict by OCR'ing the CARD HEADER, not the identity band.
        #
        # This is the actual fix for the reported incident: at scroll-top, identity_band shows
        # Hinge's own profile-independent filter-chips row ("Signals / Age / Height / Dating
        # Intent"), not a name -- so the pixel logic above can only ever say "top", and the
        # OCR corroboration just above reads that SAME band, so it inherits the same blind
        # spot. The card header lower down on screen DOES carry the name even at scroll-top
        # (see AndroidAppSpec.identity_top_name_band's docstring for the measured geometry and
        # the two on-screen layouts it was read from) -- this is the one place in the file that
        # consults it.
        #
        # Gated on ALL of state == "top" / observe_name_ocr / a stored name / a declared band:
        #
        #   state == "top" specifically (not "unknown", not "new") -- that verdict means the
        #   pixel band positively matched the app's own scroll-top chrome, i.e. we are
        #   DEMONSTRABLY looking at a card's scroll-top, which is exactly the screen state
        #   identity_top_name_band was measured against. On a SCROLLED frame this same crop is
        #   photo content, not header text, and OCRs to garbage (measured) -- consulting it
        #   there could only inject noise into a state ("new"/"unknown") that already has a
        #   more reliable signal or none at all.
        #
        #   observe_name_ocr / self._identity_name / identity_top_name_band all being set is
        #   the same "nothing to work with, don't pretend otherwise" guard the OCR
        #   corroboration above already applies: no stored name means nothing to compare
        #   against, and no declared band means this app was never measured for one (None is
        #   the default -- see that field's docstring).
        if (state == "top" and self.observe_name_ocr and self._identity_name
                and self.identity_top_name_band is not None):
            text = self._ocr_band(frame, self.identity_top_name_band, psm="6")
            self._identity_top_name_read = text
            if text:
                tokens = _TOP_NAME_TOKEN_RE.findall(text)
                stored = self._identity_name.strip()
                # Match against the STORED profile name via _name_token_matches -- ratio OR
                # prefix, deliberately biased toward "same" -- see _NAME_MATCH_RATIO's own
                # module comment and _name_token_matches' docstring for the full calibration.
                # Any token clearing the bar is enough: the header holds at most a first name
                # plus a line or two of chrome, so there is no ambiguity in "which token was
                # the name" once one clears it.
                if stored and any(_name_token_matches(tok, stored) for tok in tokens):
                    state = "same"
                    self._identity_top_name_verdict = "same"
                else:
                    # No token matched the stored name closely enough. Only conclude "new" if
                    # there is an actual NAME CANDIDATE left after removing Hinge's own chrome
                    # words (see _TOP_NAME_CHROME_WORDS) AND that candidate has at least 3
                    # alphabetic characters. The chrome-word filter alone is not enough: a read
                    # that caught nothing but chrome (the name itself went unread, e.g. cut off
                    # or misrecognised as something not alphabetic) can still leave behind a
                    # 1-2 character speckle token ("l", "@", OCR noise the >=2-char token regex
                    # let through) that is not chrome-listed but is also never a usable name --
                    # taking it as "the different person's name" would be a FALSE PASS on pure
                    # noise. Either failure mode -- no candidate at all, or only a noise-length
                    # one -- proves nothing about who is on screen and must stay "top"
                    # (inconclusive), not be promoted to a false "new".
                    candidate = next(
                        (tok for tok in tokens
                         if tok.casefold() not in _TOP_NAME_CHROME_WORDS
                         and sum(ch.isalpha() for ch in tok) >= 3),
                        None)
                    if candidate is not None:
                        state = "new"
                        self._identity_top_name_verdict = "new"
                        # A name-derived "new" is a noisier signal than a pixel-derived one
                        # (measured separation ~0 vs ~18) and is NOT trusted alone -- see
                        # wait_for_decision's confirm-frame corroboration requirement, gated on
                        # this verdict, for why a second independent read is required before
                        # this can become a PASS.
        # RESIDUAL LIMITATION, stated plainly: two consecutive profiles sharing one first name
        # still OCR to the same token here and still read as "same" -- this layer, like the
        # identity-band pixel check above it, only ever holds a FIRST name, so it cannot
        # distinguish "the profile didn't change" from "the profile changed to someone with the
        # same first name". That case still hangs (a false 'same' costs a wait, per the ratio
        # calibration's own reasoning above -- never a false PASS). Gesture corroboration
        # (observe_touch_watch, see _observe_gesture_verdict) is the layer that WOULD cover it,
        # by proving a human tap actually happened independent of what any OCR read says -- but
        # it is off on this device (Android withholds the touch event stream here; see
        # HINGE_SPEC's observe_touch_watch comment), so this gap is accepted risk, not solved.
        return state, dist

    def _observe_tap_slop_px(self) -> float:
        """How far a finger may travel and still be a TAP rather than a scroll.

        This is NOT the same number as _observe_tap_radius_px below, and conflating them is a
        real mislabelling risk rather than a style nit: the control radius is deliberately
        generous (12% of screen height, ~288px here) because it must absorb where a human's
        thumb lands ON a big floating button. Reused as tap slop it would classify a 200px
        flick -- an ordinary short read-scroll -- as a "tap", which then only has to land
        within the (equally generous) radius of the pass-X to be corroborated as a deliberate
        PASS. A scroll must never be able to become a decision, so tap-ness gets its own,
        much tighter threshold: 2% of screen height (~48px on the Pixel 7a's 2400px panel),
        comfortably above finger tremor on a stationary press and far below the shortest
        gesture anyone makes to move content."""
        _, h = self.adb.screen_size()
        return max(12.0, min(0.02 * h, 90.0))

    def _observe_tap_radius_px(self) -> float:
        """How close a corroborating tap/drag-end must land to count as "on" a control.
        12% of screen height (~288px on the Pixel 7a's 2400px-tall panel) -- generous enough
        to absorb ordinary Fitts-law tap variance on a floating bottom control, clamped to a
        sane absolute range so a degenerate screen_size() (a stub, a test double) can't
        produce a zero or unbounded radius."""
        _, h = self.adb.screen_size()
        return max(40.0, min(0.12 * h, 500.0))

    def _observe_locate_pass_x(self, frame: bytes):
        """Locate the pass-X glyph on `frame`, for gesture-corroboration's tap-radius check
        ONLY -- never used as an action target (see the class docstring of _tap/_await_button
        for why a vision hit is never blindly trusted as a target on its own; this call site
        never taps anything at all).

        Duplicates _observe_glyph_visible's contrast-tolerant matching (Hinge can render the
        glyph either polarity: a dark X, or a white outline in a dark circle) rather than
        reusing it directly, because that method returns a bool and _observe_deck_ready is
        off limits to change (see this module's redesign notes) -- but it calls the SAME
        underlying _match_glyph/_template primitives _observe_deck_ready already used to
        confirm this exact frame is deck-ready, so this is not a second independent vision
        pipeline, just the one extra call needed to get a POINT instead of a bool."""
        template = self._template("pass")
        hits = _match_glyph(frame, template, side="left", threshold=0.6)
        if hits:
            return hits[0]
        try:
            import numpy as np
            inverted = np.bitwise_not(template)
        except Exception:  # noqa: BLE001 — no usable template -> no location to corroborate against
            return None
        hits = _match_glyph(frame, inverted, side="left", threshold=0.6)
        return hits[0] if hits else None

    def _observe_gesture_verdict(self, confirm_frame: bytes) -> str:
        """Classify the human's own touch-stream gestures since this profile became ready for
        a decision (self._observe_since), to corroborate -- or refuse to corroborate -- the
        identity-and-deck-proven card advance wait_for_decision has already established by the
        time this runs. Returns:

          'pass'    — affirmative evidence of a real decision: a tap that landed within
                      _observe_tap_radius_px of the located pass-X (tap-gesture apps), or, for
                      decide_gesture='card_swipe', a drag whose END landed within that radius
                      of coords['swipe_pass_end']. wait_for_decision reports PASS.
          'resync'  — the card changed but nothing corroborates a HUMAN DECISION causing it:
                      every gesture since ready was a drag with no tap at all (a tap-gesture
                      app's card cannot advance from a drag), or the human's last tap did not
                      land on the pass control. worker.py already treats a returned None as
                      "recapture, record nothing" (worker.py:226) -- exactly right here.
          'no_data' — nothing to corroborate WITH: the watcher isn't configured/running, or it
                      has parsed zero events for the whole run (the health exception -- a
                      broken sensor must never veto a real advance, see the printed warning
                      below). Callers fall back to the identity-and-deck-proven advance alone,
                      the same as if observe_touch_watch were off entirely.
        """
        watcher = self._touch_watcher
        if not self.observe_touch_watch or watcher is None or not watcher.alive:
            return "no_data"
        if watcher.event_count == 0:
            # The stream has never delivered a single parsed line this whole run -- proof the
            # stream itself isn't working on this device (wrong node selected, a permissions
            # change mid-run, ...), not "the human genuinely never touched the screen". Only
            # the FIRST such run is worth a print; every subsequent poll would just repeat it.
            if not self._touch_watch_health_warned:
                self._touch_watch_health_warned = True
                print(f"{self.spec.app}: touch watcher has seen no events this run; falling "
                      f"back to identity-only decision proof (check `adb shell getevent` on "
                      f"this device).")
            return "no_data"

        radius = self._observe_tap_radius_px()      # "landed ON the control" -- generous
        slop = self._observe_tap_slop_px()          # "was a press, not a scroll" -- tight
        gestures = watcher.gestures_since(self._observe_since)
        w, h = self.adb.screen_size()

        if self.spec.decide_gesture == "card_swipe":
            drags = [g for g in gestures if not g.is_tap(slop)]
            pass_end = self.coords.get("swipe_pass_end")
            if not drags or pass_end is None:
                return "resync"          # nothing but taps (or nothing at all) can't drag a card
            target = (pass_end[0] * w, pass_end[1] * h)
            last = drags[-1]
            dist = math.hypot(last.up[0] - target[0], last.up[1] - target[1])
            return "pass" if dist <= radius else "resync"

        taps = [g for g in gestures if g.is_tap(slop)]
        if not taps:
            return "resync"              # only drags (or nothing) -- a tap deck can't advance from one
        last = taps[-1]
        for x0, y0, x1, y1 in self.observe_ignore_zones:
            if x0 * w <= last.up[0] <= x1 * w and y0 * h <= last.up[1] <= y1 * h:
                return "resync"          # rewind / overflow / bottom nav -- a known non-decision tap
        pass_point = self._observe_locate_pass_x(confirm_frame)
        if pass_point is None:
            return "no_data"             # can't locate the control to corroborate against -- degrade, don't veto

        # More than one tap landed ON the pass control during a single wait. The identity
        # anchor's one blind spot is that its band holds only a FIRST NAME, so two adjacent
        # profiles who share one render an identical header and a real advance can read as
        # 'same' -- the loop then keeps waiting through a decision it should have reported.
        # The touch stream is the independent witness to that: the human does not press pass
        # twice on one card, so a second pass-tap in the same window is direct evidence at
        # least one earlier decision was swallowed and this wait no longer knows which profile
        # it is holding. Resync (record nothing, recapture) rather than attach the accumulated
        # ambiguity to whatever Profile the worker captured before the wait began -- a missed
        # pass is recoverable, a label on the wrong person's photos is not.
        #
        # Only PASS-control taps count here. A heart tap followed by a pass tap is the
        # ordinary "opened the comment sheet, changed my mind, dismissed it, pressed pass"
        # sequence on ONE card -- two decisive taps, one decision -- and _await_like_resolved
        # has already resolved the sheet half of that before this ever runs.
        on_pass = [g for g in taps
                   if math.hypot(g.up[0] - pass_point[0], g.up[1] - pass_point[1]) <= radius]
        if len(on_pass) > 1:
            return "resync"

        dist = math.hypot(last.up[0] - pass_point[0], last.up[1] - pass_point[1])
        return "pass" if dist <= radius else "resync"

    _OBSERVE_WAIT_NOTICE_S = 15.0
    # At most how often wait_for_decision's rate-limited "still watching" notice may repeat.
    # The incident this exists for was TOTALLY silent: the owner pressed X, the misclassified
    # scroll match kept the wait going, and nothing at all appeared in the console, the hub, or
    # actions.jsonl for the whole time before Stop was pressed by hand -- there was no signal
    # to distinguish "working as intended, still reading" from "stuck, go look at the phone".
    # This does not fix that misclassification (the identity/OCR/shift-match layers above are
    # what do); it exists purely so the NEXT time anything stalls -- for this reason or a new
    # one nobody has hit yet -- there is at least something to look at instead of silence.

    _OBSERVE_LIKE_NOTICE_S = 30.0
    # Same idea as _OBSERVE_WAIT_NOTICE_S above, but for the window in which the human has an
    # open like sheet and is composing a comment. Deliberately slower: composing a message
    # legitimately takes minutes (measured 3m44s in the audited run of 2026-08-10), so a 15s
    # repeat there would be nagging rather than reassuring. It is not zero, because that
    # compose window was the ONE wait in observe mode with no heartbeat at all -- the longest
    # wait in the whole mode, and the only one where a wedged sheet (glyph match stuck, Hinge's
    # sending UI never resolving) would show the operator nothing whatsoever, in a loop whose
    # observe-mode deadline is None, i.e. forever. That is precisely the silence the sibling
    # notice was written to eliminate, left in place in the worst spot for it.

    def _note_observe_waiting(self, reason: str, frame: bytes | None = None) -> None:
        """Rate-limited operator print + matching debug record for wait_for_decision's "still
        waiting, nothing recorded yet" branches. `reason` is one of the plain, per-branch
        strings the call sites below pass -- `same` (identity says this is still the captured
        profile), `scroll` (layer 2 recognised this as a scroll, not a card change), `no_change`
        (nothing has moved since the last poll yet), `not_deck_ready` (the card changed but the
        next screen isn't confirmed ready), `not_settled` (deck-ready once, but the settle
        recheck hasn't agreed yet), `like_sheet` (the human has a like sheet open and is
        composing), `like_sending` (the sheet closed and Hinge is still resolving the send) --
        so a human reading the console or actions.jsonl mid-run gets the SAME vocabulary this
        file's own comments already use for these states, not a fresh set of words to map back
        onto them.

        The two like-sheet reasons repeat on the slower _OBSERVE_LIKE_NOTICE_S cadence and get
        their own wording: they are not "nothing could be classified", they are a state this
        driver understands perfectly well and is deliberately waiting out.

        `self._observe_last_notice` is set fresh at the top of every wait_for_decision call
        (one call = one profile's wait), so this can never go permanently quiet across
        profiles: a notice suppressed near the end of one profile's wait cannot suppress the
        FIRST notice of the next one.

        The rate limit applies only to REPEATS OF THE SAME reason. A change of reason is always
        announced, because the change is the informative part: `no_change` -> `not_deck_ready`
        is the transition that says the screen finally started moving, and suppressing it left
        the console and actions.jsonl asserting a stale reason for up to 15s -- or, for a
        short-lived state, omitting it from the record entirely, which is the one thing a
        heartbeat must never do.

        `frame`, when supplied, is the frame the verdict was actually computed from. It is
        logged as-is instead of taking a fresh screencap, which is both cheaper (one less ADB
        round-trip per notice, on the hot polling path) and more truthful: a full 1080x2400
        screencap costs ~1-2s here, so the freshly-grabbed image used to show the screen 1-2s
        AFTER the verdict -- and if the human tapped in that gap, the saved picture showed the
        NEXT state while the record next to it claimed "nothing has moved". That is actively
        misleading in the one artifact you open when diagnosing a stall.
        """
        like_wait = reason in _OBSERVE_LIKE_WAIT_REASONS
        interval = self._OBSERVE_LIKE_NOTICE_S if like_wait else self._OBSERVE_WAIT_NOTICE_S
        if reason != self._observe_last_reason:
            # A changed reason bypasses the full interval (the transition is the informative
            # part) but NOT this floor. Without it, two branches that disagree on a flapping
            # screen -- a sheet glyph matching on one poll and not the next -- would alternate
            # reasons every _OBSERVE_POLL_S and turn an anti-silence heartbeat into ~3 console
            # lines a second. A transition that flaps faster than the floor is noise, and the
            # next notice reports whatever the reason is by then anyway.
            interval = _OBSERVE_NOTICE_FLOOR_S
        now = time.monotonic()
        if now - self._observe_last_notice < interval:
            return
        self._observe_last_notice = now
        self._observe_last_reason = reason
        if like_wait:
            detail = ("your like sheet is still open -- take your time; I'm waiting for you to "
                      "tap Send Like or dismiss it" if reason == "like_sheet"
                      else "the sheet closed and Hinge hasn't shown the next card yet")
            print(f"{self.spec.app}: still watching -- {detail} (reason={reason}). Nothing is "
                  f"recorded until this resolves. If it stays like this with the phone showing "
                  f"something else, check the phone, or press Stop.")
        else:
            # "no_change" is the one reason that means the screen has NOT moved at all, so it
            # must not be announced as "the screen changed" -- an operator who just tapped X and
            # reads that the screen changed would reasonably conclude the tap registered and the
            # driver merely failed to classify it, when in fact nothing happened on the phone
            # and the tap is the thing to retry. Getting that backwards sends them debugging the
            # wrong half of the system, which is exactly how the incident this notice exists for
            # was misread.
            observed = ("nothing has moved on screen yet" if reason == "no_change"
                        else "the screen changed but nothing could yet be")
            verb = "" if reason == "no_change" else " classified as a like or a pass"
            print(f"{self.spec.app}: still watching -- {observed}{verb} (reason={reason}). "
                  f"Observation is continuing; if you already tapped X or the heart, this is "
                  f"normal while the screen settles. If it repeats for a while, check the phone "
                  f"matches what you expect, or press Stop.")
        if frame is not None and self._dbg is not None:
            # Straight to DebugLog.action, bypassing _dbg_action: the whole point is to log the
            # frame already in hand rather than let _dbg_action grab a second, later one.
            try:
                self._dbg.action("observe_waiting", before=frame, reason=reason)
            except Exception:  # noqa: BLE001 — debug logging must never break the wait
                pass
            return
        self._dbg_action("observe_waiting", None, reason=reason)

    def _note_observe_like_outcome(self, base: bytes, sent: bool, *, sheet_seen: bool = True,
                                   top: float | None = None, bot: float | None = None) -> None:
        """Record how an opened like sheet RESOLVED: sent (a LIKE) or dismissed.

        The PASS path has written a full `observe_decision` record since the observe redesign;
        the LIKE path wrote nothing at all. `observe_like_anchor` fires on INTENT (the sheet was
        detected), not on resolution, so an actions.jsonl reader could not distinguish the four
        outcomes of an opened sheet -- sent, dismissed, stop/timeout, resync -- and a real run
        (2026-08-10) shows exactly that: capture(jessica) -> observe_waiting ->
        observe_like_anchor -> capture(Victoria), with no on-disk evidence that jessica was
        liked at all. The LABEL was never at risk (worker.py stores it either way); what was
        missing was the diagnostic trail, on the rarer and higher-value of the two decisions,
        and it broke the invariant bugreport.py documents (exactly one decision record per
        capture).

        `gesture` is deliberately reported as "not_checked" rather than run: Layer 3's
        corroboration (_observe_gesture_verdict) measures the distance from the last touch-up to
        the PASS control, so on a like it would report a confident "resync" for a tap that
        correctly landed on the heart -- a wrong answer is worse in a debug log than an honest
        absence.
        """
        fields = {
            "capture_truncated": getattr(self, "_current_capture_truncated", None),
            "profile_name": self._identity_name,
            "gesture": "not_checked", "watcher": self.observe_touch_watch,
        }
        if top is not None:
            fields["top"] = round(top, 2)
        if bot is not None:
            fields["bot"] = round(bot, 2)
        if sent:
            self._dbg_action("observe_decision", base, decision="like", **fields)
        elif sheet_seen:
            # Not a decision -- the human opened the sheet and backed out, and the wait
            # continues on the SAME card. Logged under its own action name so it can never be
            # miscounted as a decision, while still leaving a trace that the sheet was up.
            self._dbg_action("observe_like_dismissed", base, **fields)
        else:
            # The bottom half changed and resolved back to the same card, but the Send Like
            # glyph was never actually matched on ANY poll -- so calling this a dismissed like
            # sheet would assert something nobody observed. A snackbar ("Your like was sent"),
            # a toast, or a keyboard dismissal all land here. Recorded under a name that claims
            # only what happened, so the log stays trustworthy about the rarer, higher-value
            # class of event it sits next to.
            self._dbg_action("observe_bottom_delta", base, **fields)

    # --- observe mode (shadow learning) --------------------------------
    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None,
                          on_like_intent=None) -> bool | None:
        """Block until you manually like/pass the current card, inferred from
        screencap deltas (no accessibility tree):

          LIKE — tapping a heart slides the comment / "Send Like" sheet up over the
                 BOTTOM while the photo stays up top (bottom changes, top doesn't).
                 We surface an opener through the callback, then PASSIVELY wait
                 for the human to type and tap Send Like. Only a new card -> True.
          PASS — the whole card advances to a DIFFERENT, deck-ready, SETTLED profile,
                 positively proven by the identity anchor and/or content match below
                 (never by "changed and unrecognised" alone) -> False.
          none — stop requested, deck empty, timeout, or a card change with no positive
                 decide evidence (a resync -- see _observe_gesture_verdict) -> None.

        ⚠️ LIVE-VERIFY: the like-sheet geometry is gated until the profile is
        finished, so the top/bottom thresholds must be confirmed on-device before
        trusting observe labels. After "READY", tap Hinge's X or heart -- reading the
        profile by scrolling first is fine and expected (Signals behavior #1): the
        identity anchor below is exactly what makes that safe to do.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        base = self._await_live_frame(deadline, should_stop)
        if base is None:
            return None
        # "When this profile became READY" for _observe_gesture_verdict's gesture-window
        # query below -- set once per call, before the poll loop, so a tap the human made
        # while still reading (before this wait even started watching) can never be
        # mistaken for corroboration of a LATER, unrelated card advance.
        self._observe_since = time.monotonic()
        # Rate-limit anchor for _note_observe_waiting -- reset here (once per call = once per
        # profile's wait), never inside the loop, so it "can never go permanently quiet" is
        # true across the WHOLE run, not just within one profile: see that method's docstring.
        # The reason is cleared alongside it so the first notice for a NEW profile is never
        # suppressed as a repeat of whatever the previous profile happened to end on.
        self._observe_last_notice = time.monotonic()
        self._observe_last_reason = None
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            cur = self._screencap(on_blank="none")
            if cur is None:                           # screen asleep: the owner stepped away.
                time.sleep(_OBSERVE_POLL_S)           # keep watching — do NOT diff a black frame
                continue                              # against `base` (that reads as a phantom pass)

            # Check the actual compose sheet BEFORE interpreting aggregate screen deltas.
            # Opening the keyboard can change both halves at once, which the generic branch
            # below would otherwise call a card advance/PASS.  A visible sheet is instead an
            # in-progress human like until it is dismissed or a ready next deck card appears.
            if self._observe_like_sheet_visible(cur):
                # `cur` is passed as the anchor because it is ALREADY the frame that just
                # proved the Send Like glyph is visible — no extra screencap is taken for
                # this. The sheet slides up over the BOTTOM while the liked photo stays up
                # top (see this method's own docstring above), so this one full-screen frame
                # already contains both the item the comment attaches to AND the composer
                # itself. Re-capturing here would add ADB latency and an animation-timing
                # race for zero gain over what's already in hand.
                self._notify_observe_like_intent(on_like_intent, True, cur)
                sent, intent_notified = self._await_like_resolved(
                    base, deadline, should_stop, on_like_intent=on_like_intent,
                    intent_notified=True,
                )
                if sent is None:
                    return None
                self._note_observe_like_outcome(base, sent)
                if sent:
                    if intent_notified:
                        self._notify_observe_like_intent(on_like_intent, False)
                    return True
                if intent_notified:
                    self._notify_observe_like_intent(on_like_intent, False)
                base = self._await_live_frame(deadline, should_stop)
                if base is None:
                    return None
                continue
            top, bot = _split_diff(base, cur)
            if top < self.change_threshold and bot < self.change_threshold:
                self._note_observe_waiting("no_change", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue

            # A LIKE opens the comment / "Send Like" sheet: the photo (top half) stays put
            # while the BOTTOM changes. Detect this BEFORE the scroll check — otherwise, on a
            # light/gray profile, a like-sheet frame can match a stored full-frame signature
            # and be mis-read as a scroll, silently dropping the LIKE (bug C1).
            if bot >= self.change_threshold and top < self.change_threshold:
                # A bottom-only delta starts a *candidate* sheet flow. Do not spend
                # opener budget or replace the hub instructions until the actual
                # Send Like glyph corroborates it; a bottom animation/scroll alone
                # is not a human intent to like.
                sheet_visible = self._observe_like_sheet_visible(cur)
                if sheet_visible:
                    self._notify_observe_like_intent(on_like_intent, True, cur)   # cur already proves the sheet -- see the sibling call site above
                sent, intent_notified = self._await_like_resolved(
                    base, deadline, should_stop, on_like_intent=on_like_intent,
                    intent_notified=sheet_visible,
                )
                if sent is None:
                    return None
                # sheet_seen comes from _await_like_resolved's own return, not from the single
                # `sheet_visible` probe above: the glyph can start matching on a LATER poll (the
                # sheet was still animating up when the bottom delta was first noticed), and it
                # can also never match at all -- a snackbar or toast produces the same
                # bottom-only delta with no sheet behind it. Only the resolver knows which
                # happened across every poll it made.
                self._note_observe_like_outcome(base, sent, sheet_seen=intent_notified,
                                                top=top, bot=bot)
                if sent:
                    if intent_notified:
                        self._notify_observe_like_intent(on_like_intent, False)
                    return True                       # like sheet resolved to a new card
                # Dismissal is not a pass. Clear the suggestion and continue waiting on
                # the same profile for the operator's next X/heart decision.
                if intent_notified:
                    self._notify_observe_like_intent(on_like_intent, False)
                base = self._await_live_frame(deadline, should_stop)   # cancelled -> resync
                if base is None:
                    return None
                continue

            # Top changed (whole card moved). LAYER 1 (identity) is authoritative and runs
            # FIRST: whatever else changed on screen, a frame whose sticky per-profile header
            # still matches the CAPTURED profile is still the same card, full stop -- this is
            # the actual fix for the reported bug (the owner scrolling to read a profile, no
            # tap of any kind, recorded as a PASS). Layers 2/3 below only ever CORROBORATE an
            # identity-proven advance; neither can override a 'same' verdict here.
            identity_state, identity_dist = self._identity_of(cur)
            # Captured immediately, into a LOCAL, because self._identity_top_name_verdict is
            # reset on every _identity_of call -- including the confirm-frame call further
            # below -- and would otherwise no longer describe THIS frame's verdict by the time
            # the settle/confirm block reads it. See that block's own comment for why this
            # distinction (pixel-derived 'new' vs name-derived 'new') matters.
            first_new_is_name_derived = (
                identity_state == "new" and self._identity_top_name_verdict == "new"
            )
            if identity_state == "same":
                base = cur                            # scroll within the SAME profile -> keep waiting
                self._note_observe_waiting("same", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue

            # LAYER 2: legacy content match. An exact stored-signature hit, then a
            # vertical-shift search restricted to the scrolling CONTENT rows only -- see
            # _vertical_shift_match's docstring for the measured table explaining why the
            # earlier whole-frame version of this search could never fire in production.
            # Still reached (rather than short-circuited) when identity_state is 'top' or
            # 'unknown': a profile that hasn't revealed its sticky header yet, or an app with
            # no identity_band at all, gets exactly this content-only fallback, which is the
            # ONLY scroll-vs-pass discriminator this file had before the identity anchor.
            # ...but NOT when identity has already answered 'new'. Layer 2 compares whole
            # PHOTOS, and this file's own _is_current_profile_frame docstring records why that
            # is a weak cross-profile signal: dating first-photos are alike (centred face,
            # light background), so a genuinely different card can land under
            # change_threshold against one of the previous profile's captured frames. If that
            # collision were allowed to overrule the sticky header -- which is measured
            # 0.00-vs-~18 separated and is the whole basis of this redesign -- a real advance
            # would be swallowed as "just a scroll", the human's decision on the profile they
            # actually left would never be recorded, and their NEXT decision would be
            # attributed to the previous profile's photos by worker.py (which holds the
            # Profile object captured before this call and never re-derives it). The sibling
            # _is_current_profile_frame already returns on BOTH 'same' and 'new' before
            # consulting _current_sigs; this, the more consequential call site, simply never
            # had that precedence ported to it. 'top'/'unknown' still fall through, which is
            # the case Layer 2 exists for: the header isn't visible, so photos are all there is.
            ds_cur = _downsample(cur)
            # Indexed pairs (original _current_sigs position, sig), not a flat filtered list:
            # the whole point of the new sig_index debug field below is to name WHICH captured
            # frame matched (photos/_current_sigs are index-aligned -- see _capture_current's
            # own comment on that), which a re-numbered filtered list would get wrong the
            # moment any earlier frame in the profile was undecodable (a None slot).
            current_sigs = getattr(self, "_current_sigs", None) or []
            seen_pairs = [(idx, s) for idx, s in enumerate(current_sigs) if s is not None]
            min_dist = None
            shift_matched = False
            if identity_state != "new" and ds_cur is not None and seen_pairs:
                import numpy as np
                dists = [(idx, float(np.mean(np.abs(ds_cur - ds_seen)))) for idx, ds_seen in seen_pairs]
                min_idx, min_dist = min(dists, key=lambda pair: pair[1])
                if min_dist < self.change_threshold:
                    # Swallowing a screen change is exactly as consequential as concluding a
                    # PASS from one, and until now it was the only silent path left in this
                    # function -- the same absence of a paper trail that made the original
                    # bug so hard to diagnose from a debug log. Best-effort, never gates.
                    self._dbg_action("observe_scroll", cur, reason="content_sig",
                                     identity=identity_state, min_sig_dist=round(min_dist, 2),
                                     sig_index=min_idx, name_read=self._identity_top_name_read)
                    base = cur                        # it's a scroll -> keep waiting
                    self._note_observe_waiting("scroll", cur)
                    time.sleep(_OBSERVE_POLL_S)
                    continue
                content_rows = _content_rows(self.content_band, ds_cur.shape[0])
                band_rows = content_rows[1] - content_rows[0]
                # First match wins (same iteration-order semantics _vertical_shift_match itself
                # always had) -- sig_index/shift/overlap_rows below describe THAT match, which
                # is exactly the "matched stored frame 6 at a 12-row shift with only 4 rows
                # overlapping" detail the original incident's log had no way to show (see
                # _vertical_shift_match's own docstring for the return shape this reads).
                shift_hit = None
                for idx, ds_seen in seen_pairs:
                    matched, shift, overlap_rows = _vertical_shift_match(
                        ds_cur, ds_seen, threshold=self.change_threshold, rows=content_rows)
                    if matched:
                        shift_hit = (idx, shift, overlap_rows)
                        break
                shift_matched = shift_hit is not None
                if shift_matched:
                    sig_index, shift, overlap_rows = shift_hit
                    self._dbg_action("observe_scroll", cur, reason="shift_match",
                                     identity=identity_state,
                                     min_sig_dist=None if min_dist is None else round(min_dist, 2),
                                     sig_index=sig_index, shift=shift, overlap_rows=overlap_rows,
                                     band_rows=band_rows, name_read=self._identity_top_name_read)
                    base = cur                        # it's a scroll -> keep waiting
                    self._note_observe_waiting("scroll", cur)
                    time.sleep(_OBSERVE_POLL_S)
                    continue

            # Neither identity nor content recognised `cur` as the captured profile. That is
            # NECESSARY for a PASS but not SUFFICIENT -- the OLD rule stopped right here
            # ("changed and unrecognised -> PASS"), which is exactly what let a human's manual
            # scroll into territory nothing had captured yet (identity 'top'/'unknown'/'new'
            # with no content match either) masquerade as a decision. LAYER 2's positive proof:
            # this must be a DIFFERENT, deck-READY, SETTLED card -- no like sheet, both deck
            # glyphs visible, and -- after a short settle -- the SAME still true on a SECOND
            # capture, with identity confirming that second capture is not the captured
            # profile either. Any single failure here means "not settled yet": keep watching,
            # conclude nothing, and do NOT advance `base` (a transient/animating frame is not
            # a safe anchor for the next diff).
            if self._observe_like_sheet_visible(cur) or not self._observe_deck_ready(cur):
                self._note_observe_waiting("not_deck_ready", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue

            time.sleep(0.5)
            confirm = self._screencap(on_blank="none")
            if confirm is None:
                continue                              # can't confirm blind -- re-poll
            confirm_identity_state = self._identity_of(confirm)[0]
            # A name-derived 'new' (Layer 1b's OCR of the card header, at scroll-top) is a
            # noisier signal than a pixel-derived one -- see _identity_of's Layer 1b comment
            # for the measured ~0-vs-~18 separation the pixel band gets that a fuzzy-matched
            # name string never can. A transient misread (garbled tesseract output on one
            # frame) must not by itself be able to write a label, so when the FIRST frame's
            # 'new' verdict came from Layer 1b, the confirm frame -- captured independently,
            # 0.5s later -- must ALSO read 'new' by the same OCR path; a mere 'top' (identity
            # simply not resolved this time) is no longer good enough to corroborate it, even
            # though 'top' passes the plain `!= "same"` test below on its own. This is the
            # cheapest available corroboration: a second, independent read of the same claim.
            # Honesty about its limits: a SYSTEMATIC misread -- a crop boundary that is
            # consistently off, not a one-off garble -- will reproduce identically on the
            # confirm frame too and this check will not catch it. That failure mode is what
            # the prefix rule in _name_token_matches covers instead (a truncated read is a
            # prefix of the real name and matches 'same' regardless of how many times it
            # reproduces); this check only ever guards against a NON-reproducing misread.
            proven = (
                not self._observe_like_sheet_visible(confirm)
                and self._observe_deck_ready(confirm)
                and not self._changed(cur, confirm)
                and confirm_identity_state != "same"
                and (not first_new_is_name_derived or confirm_identity_state == "new")
            )
            if not proven:
                self._note_observe_waiting("not_settled", confirm)
                time.sleep(_OBSERVE_POLL_S)
                continue                              # still settling / reverted -- keep watching

            # LAYER 3: corroborate the now identity-and-deck-proven advance against the
            # human's OWN touch stream, when this app is configured to read one (read-only --
            # see touchwatch.py; this driver never injects anything on this path).
            # 'no_data' (watcher off, not running, or healthy-but-empty for this window) falls
            # through to the identity-proven advance alone -- corroboration INFRASTRUCTURE
            # being unavailable must never itself veto a real decision, only affirmative
            # evidence of a NON-decision (a resync) does.
            verdict = self._observe_gesture_verdict(confirm)
            # profile_name, not name: DebugLog.action's own first positional parameter is
            # `name` (the action-type string, "observe_decision"/"observe_resync" below) --
            # a fields key of literally `name` collides with it (see _capture_current's
            # identical note above).
            fields = dict(
                top=round(top, 2), bot=round(bot, 2),
                min_sig_dist=None if min_dist is None else round(min_dist, 2),
                shift_matched=shift_matched,
                capture_truncated=getattr(self, "_current_capture_truncated", None),
                identity=identity_state,
                identity_dist=None if identity_dist is None else round(identity_dist, 2),
                profile_name=self._identity_name, gesture=verdict, watcher=self.observe_touch_watch,
            )
            if verdict == "resync":
                # The card DID change, but nothing corroborates a human decision causing it
                # (only a drag, or a tap that didn't land on the pass control). worker.py
                # already treats a returned None as "recapture, record nothing" (worker.py:226)
                # -- exactly the right outcome for a resync, never a silent mislabel.
                self._dbg_action("observe_resync", base, **fields)
                return None
            self._dbg_action("observe_decision", base, decision="pass", **fields)
            return False                              # identity + deck-ready + settle (+ gesture) -> pass
        return None

    def _notify_observe_like_intent(self, callback, active: bool,
                                    anchor: bytes | None = None) -> None:
        """Best-effort notification for the passive Hinge observe flow.

        This helper intentionally performs no ADB input. If displaying or generating
        a suggestion fails, the human can still write their own message or dismiss
        the sheet, so observation must continue normally.

        `anchor` is a screencap of the like sheet as it is open on screen — the same
        "which item does this comment attach to" picture `_like_comment_sheet`'s
        `anchored_opener` repair path uses in auto mode, surfaced here so observe mode's
        suggestion can be grounded in the real item too instead of guessing. None when
        clearing (`active=False`) — there is nothing left on screen to anchor a picture
        of once the sheet has closed.

        Instance method rather than the `@staticmethod` this used to be, so it can reach
        `self._dbg` for the best-effort anchor log below.

        Swallow-AND-PRINT below, not swallow-and-vanish: this hook is the ONLY way the
        operator's suggestion reaches the hub, and the old bare `except Exception: pass`
        meant a stale callback signature (an ordinary `TypeError` after some future
        refactor, say) would make suggestions vanish forever with zero evidence anything
        was ever wrong — nothing printed, nothing logged, nothing to grep for. The
        failure must still be non-fatal (observation must keep running with no
        suggestion rather than stop), but it must not stay invisible.
        """
        if callback is None:
            return
        if self._dbg is not None and active and anchor is not None:
            self._dbg.action("observe_like_anchor", before=anchor)   # best-effort; DebugLog.action never raises
        try:
            callback(active, anchor)
        except Exception as exc:  # noqa: BLE001 — non-input side channel must not break observation
            print(f"{self.spec.app}: observe like-intent callback failed "
                  f"({type(exc).__name__}: {exc}); continuing observation without the suggestion.")

    def _observe_like_sheet_visible(self, frame: bytes) -> bool:
        """Whether Hinge's ``Send Like`` confirmation is still on screen.

        A keyboard can move enough of the profile to create a large top-region
        diff while the comment sheet remains open. The confirmation glyph is the
        authoritative guard against calling that an already-sent like.
        """
        if self.spec.like_flow != "comment_sheet":
            return False
        try:
            hits = _match_glyph(frame, self._template("confirm"), side="any", threshold=0.6)
            if not hits:
                return False
            # cv2's normalized matcher can report a mathematically-perfect hit on a tiny
            # flat synthetic/degraded frame (where the template cannot physically fit).  The
            # real Hinge control is in the central/lower compose sheet, so reject impossible
            # or top-bar positions before allowing this guard to block capture/decision flow.
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return False
            height = image.shape[0]
            return any(height * 0.25 <= y <= height * 0.85 for _x, y in hits)
        except Exception:  # noqa: BLE001 — retain delta fallback if visual matching is unavailable
            return False

    def _observe_deck_ready(self, frame: bytes) -> bool:
        """Whether ``frame`` is visibly a swipe deck ready for the next decision.

        A closed comment sheet is not, by itself, evidence that Hinge has advanced: its
        network/animation "sending" state also closes the Send Like control and can stay
        visually stable for seconds.  Require the existing independent pass-X and like-heart
        glyphs instead.  This method is observe-only perception; it never uses the matched
        positions as input targets.
        """
        try:
            like = self._observe_glyph_visible(frame, "like", side="right")
            passed = self._observe_glyph_visible(frame, "pass", side="left")
            return bool(like and passed)
        except Exception:  # noqa: BLE001 — inability to prove a ready deck must fail closed
            return False

    def _observe_glyph_visible(self, frame: bytes, role: str, *, side: str) -> bool:
        """Passive control detection, accepting either UI contrast polarity.

        The stored heart template is a dark glyph, while the live Hinge deck can render the
        same outline white inside a black circle.  Autonomous actions intentionally retain
        `_match_glyph`'s calibrated, single-polarity matcher: broadening it would turn this
        perception-only readiness check into a new tap target.  Here we only need evidence
        that a future deck has loaded, so testing the contrast-inverted template is safe.
        """
        template = self._template(role)
        if _match_glyph(frame, template, side=side, threshold=0.6):
            return True
        try:
            import numpy as np
            inverted = np.bitwise_not(template)
        except Exception:  # noqa: BLE001 — no usable template means no proof of a deck
            return False
        return bool(_match_glyph(frame, inverted, side=side, threshold=0.6))

    def _await_like_resolved(self, base: bytes, deadline, should_stop,
                             *, on_like_intent=None,
                             intent_notified: bool = False) -> tuple[bool | None, bool]:
        """After the like sheet appears, wait for a human send or dismissal.

        Do not use a top-region change alone as evidence of sending: focusing the
        text field can shift the profile behind an otherwise-still-open sheet.
        Once the sheet glyph is gone, a *stable, visibly ready* new deck card is a sent
        like; closing the sheet onto Hinge's transient sending UI is deliberately neither.
        The base card (or another captured frame of the current profile) is a dismissal.
        """
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None, intent_notified
            cur = self._screencap(on_blank="none")
            if cur is None:                           # screen asleep mid-wait: keep watching
                time.sleep(_OBSERVE_POLL_S)           # (never diff a black frame against base)
                continue
            if self._observe_like_sheet_visible(cur):
                if not intent_notified:
                    self._notify_observe_like_intent(on_like_intent, True, cur)   # cur already proves the sheet -- see wait_for_decision's call sites
                    intent_notified = True
                # The keyboard/sheet may radically alter the top half. It is still
                # an unsent human draft while the Send Like control is visible.
                self._note_observe_waiting("like_sheet", cur)
                time.sleep(_OBSERVE_POLL_S)
                continue
            # require_content=True: at THIS call site a wrong "still the current profile"
            # verdict throws away a like the human really sent (see the flag's docstring).
            current = cur == base or self._is_current_profile_frame(cur, require_content=True)
            ready = not current and self._observe_deck_ready(cur)
            if current or ready:
                # Both a dismissal and a ready deck must settle.  This rejects a single
                # transition frame and, for the ready case, proves the deck controls remain
                # present after Hinge's sending animation has completed.
                time.sleep(0.5)
                confirm = self._screencap(on_blank="none")
                if confirm is None:
                    continue                          # can't confirm blind -> re-poll
                if self._observe_like_sheet_visible(confirm):
                    continue                          # sheet reappeared / animation still resolving
                confirm_current = confirm == base or self._is_current_profile_frame(
                    confirm, require_content=True)
                if current and confirm_current:
                    return False, intent_notified     # genuinely back on the current profile
                if (ready and not confirm_current and self._observe_deck_ready(confirm)
                        and not self._changed(cur, confirm)):
                    return True, intent_notified      # stable, non-current, ready next deck card
            # Closed sheet but no current card and no ready deck = Hinge is still processing.
            # Keep observing; a timeout is unresolved, never a false cancellation/label.
            self._note_observe_waiting("like_sending", cur)
            time.sleep(_OBSERVE_POLL_S)
        return None, intent_notified

    def _is_current_profile_frame(self, frame: bytes, *, require_content: bool = False) -> bool:
        """True if `frame` matches a captured frame of the CURRENT profile — i.e. an
        'advance' that is really a scroll within the same profile, not a new card.
        Guards the C1 reorder against a uniform-top scroll reading as a LIKE (#6).
        Conservatively False when undecodable (treat as a genuine advance).

        The audit-flagged weakness this used to carry: full-frame min-over-ALL-captured-sigs
        was the SOLE like-vs-scroll discriminator here, and dating first-photos are visually
        similar (centred face, light background), so a genuinely NEW card could collide with
        one of the current profile's sigs — a real LIKE then mis-read as a scroll, the like was
        dropped, and the observe loop desynced (the NEXT decision attributed to the wrong
        profile's photos). The fix was deferred for want of a signal that distinguishes two
        profiles better than "their photos look different".

        The identity anchor is exactly that signal, and it costs nothing to consult here: the
        app's sticky per-profile header (see _identity_of) is measured pixel-identical across
        every scroll offset of one profile and completely different across profiles, so it
        answers "same card or not" without depending on how alike two people's photos are.
        When it can answer ('same' or 'new') it is authoritative and the photo comparison is
        not consulted at all; a 'top'/'unknown' verdict — the header isn't visible, or this app
        declares no identity_band — falls through to the original full-frame behaviour, which
        is exactly what this method did before and no worse.

        `require_content` exists because the identity anchor has one blind spot, and the two
        callers pay wildly different prices for it. The band contains only the person's FIRST
        NAME, and it was measured WITHIN one profile (0.00 across three scroll offsets vs
        ~18 against that profile's own scroll-top chrome) -- never BETWEEN two people. Two
        different profiles who happen to share a first name render that header identically, so
        'same' can be wrong for a genuinely new card. In wait_for_decision's scroll-vs-pass
        check a wrong 'same' merely defers (keep waiting, decide nothing). Here in
        _await_like_resolved it DISCARDS: a like the human actually sent reads as a dismissal
        and is silently dropped. So that caller passes require_content=True and gets 'same'
        only when the header AND the photos agree -- two independent signals that would both
        have to collide at once -- while the cheap identity-only path stays for the caller
        whose worst case is patience."""
        state, _dist = self._identity_of(frame)
        if state == "new":
            return False                              # the header proves it is NOT this card
        ds = _downsample(frame)
        sigs = [s for s in getattr(self, "_current_sigs", None) or [] if s is not None]
        content_match = False
        if ds is not None and sigs:
            import numpy as np
            content_match = min(float(np.mean(np.abs(ds - s))) for s in sigs) < self.change_threshold
        if state == "same":
            return content_match if require_content else True
        return content_match                          # 'top'/'unknown': photos are all there is


class HingeDriver(AndroidDriver):
    """Thin binding: AndroidDriver + HINGE_SPEC. All behavior lives in AndroidDriver above;
    this class exists so `operation_love.drivers.hinge.HingeDriver(cfg)` keeps working exactly
    as it always has (driver factory, tools/hinge_inspect.py, the test suite)."""

    def __init__(self, cfg):
        super().__init__(cfg, HINGE_SPEC)
