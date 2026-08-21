"""READ-ONLY observer of the phone's own touchscreen event stream via `adb shell getevent`.

Screencap deltas can prove a card CHANGED but never prove a human actually pressed
anything, or where -- a stray notification redraw, a system toast, or (see the
identity-band bug this module exists to help close) a manual scroll can all produce a
frame delta with zero taps behind it. This module answers that question directly by
reading the touchscreen's own kernel input stream over `adb shell getevent -lt`, the
same unrooted, `shell`-domain-readable device confirmed live on the Pixel 7a
(ops/ANTI-BOT-RESEARCH.md, 2026-08-10): `/dev/input/event*` is `root:input
crw-rw----` and the adb `shell` user can read (not write) it. Real touchscreen there is
`goodix_ts0`, `INPUT_PROP_DIRECT`, `ABS_MT_POSITION_X` max 1079 / `ABS_MT_POSITION_Y`
max 2399 -- 1:1 with the 1080x2400 framebuffer -- and it declares `BTN_TOUCH` (KEY
014a).

Hard contract: this is an OBSERVER, never an actor. It never writes to /dev/input,
never runs `input`, and never installs anything on the device -- a `TouchWatcher`
issues exactly two kinds of read-only adb invocations: `getevent -p` once, to pick the
device, then `getevent -lt <device>` streamed for the life of the session. Confusing
this module with a touch INJECTOR would reintroduce exactly the on-device-agent risk
uhid.py/adb.py were built to avoid (see adb.py's module docstring).

Two touch streams can exist on this phone at once: the human's real finger on
`goodix_ts0`, and -- while a bot swipe is in flight -- our OWN virtual digitizer
(uhid.py's UhidTouch), which registers as `og_touch_<8 hex>` (`secrets.token_hex(4)`).
`exclude_name_prefixes` exists so a TouchWatcher never mistakes our own synthetic
gesture for the human's; the default excludes exactly that prefix.

`Adb` (adb.py) is a one-shot `subprocess.run` transport BY DESIGN -- it must not grow a
streaming API just for this one caller. This module owns its own `subprocess.Popen`
instead of routing through Adb.
"""
from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass


class TouchWatchUnavailable(RuntimeError):
    """The touch event stream could not be attached: no device in `adb shell getevent -p`
    declared both ABS_MT_POSITION_X and ABS_MT_POSITION_Y (after excluding our own UHID
    digitizer), or the probe/stream process itself could not be started.

    Same "explicit operator decision, never a silent downgrade" contract as
    UhidUnavailable (uhid.py): the caller (hinge.py's open_session) treats this as fatal
    when gesture corroboration is configured on, naming the config key to disable it in
    the DriverClosed it raises. Quietly running WITHOUT gesture corroboration because the
    stream failed to attach would silently narrow observe mode's PASS proof back to
    content-only matching -- the exact false-PASS bug this module exists to help close.
    """


@dataclass(frozen=True)
class Gesture:
    """One DOWN..UP touch on the phone's own screen, as read off the device's touch
    stream.

    Timestamps are the HOST's `time.monotonic()` at the moment each boundary event was
    read off the adb pipe, not the device's own event clock (getevent's bracketed
    `[   nnnn.nnnnnn]` prefix, which is device uptime -- a different, unsynchronized
    clock domain that would need a rebase against wall time neither side commits to
    holding stable for the length of a session). Every comparison this module's callers
    make (`gestures_since`) is against another host `time.monotonic()` reading, so
    staying entirely in the host's clock is sufficient and avoids that reconciliation.
    """

    t_down: float
    t_up: float
    down: tuple[int, int]   # screen px
    up: tuple[int, int]     # screen px
    travel_px: float        # max euclidean distance from the down point

    def is_tap(self, slop_px: float) -> bool:
        """True when the finger never travelled more than `slop_px` from its down point.

        No baked-in default: "how much wobble still counts as a tap" is a UI-geometry
        decision (hinge.py uses a fraction of screen height) that belongs with the
        caller, not the parser.
        """
        return self.travel_px <= slop_px


# getevent -lt line shape, values in HEX:
# [   12345.678901] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    000001f4
_LT_LINE_RE = re.compile(
    # AOSP's getevent normally expands event and code names under ``-l``.  Recent builds
    # expose the same stream as numeric Linux input codes even with that option, however:
    # ``0003 0035 000001f4`` rather than ``EV_ABS ABS_MT_POSITION_X 000001f4``.  Both
    # representations are the same event; accepting both keeps the observer tied to the
    # actual kernel stream rather than a toolbox formatting detail.
    r"^\[\s*[\d.]+\]\s+\S+:\s+(?P<ev>EV_\w+|[0-9a-fA-F]{4})\s+"
    r"(?P<code>\S+)\s+(?P<value>\S+)\s*$"
)

# getevent -p device-selection shape. Real transcript (Pixel 7a, 2026-08-10):
#   add device 3: /dev/input/event3
#     name:     "goodix_ts0"
#     events:
#       KEY (0001): 014a
#       ABS (0003): 0035  : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
#                   0036  : value 0, min 0, max 2399, fuzz 0, flat 0, resolution 0
#     input props:
#       INPUT_PROP_DIRECT
_ADD_DEVICE_RE = re.compile(r"^add device \d+:\s*(/dev/input/event\d+)\s*$", re.MULTILINE)
_NAME_RE = re.compile(r'^\s*name:\s*"([^"]*)"', re.MULTILINE)
# Matches every "<4-hex-code>  : value V, min N, max M, ..." axis line inside an ABS
# block, regardless of exactly how the surrounding "ABS (0003):" header wraps around the
# first axis -- that header never itself matches this pattern (its own trailing "):" is
# followed by an axis code, not literally by "value").
_ABS_AXIS_RE = re.compile(r"([0-9a-fA-F]{4})\s*:\s*value\s+-?\d+,\s*min\s+-?\d+,\s*max\s+(-?\d+)")

ABS_MT_POSITION_X = "0035"
ABS_MT_POSITION_Y = "0036"
EV_ABS = "0003"
EV_KEY = "0001"
BTN_TOUCH = "014a"
ABS_MT_TRACKING_ID = "0039"


def _iter_device_blocks(getevent_p_output: str):
    """Yield (dev_path, block_text) for each `add device N: /dev/input/eventX` section of
    `adb shell getevent -p` output, in the order getevent printed them."""
    matches = list(_ADD_DEVICE_RE.finditer(getevent_p_output))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(getevent_p_output)
        yield m.group(1), getevent_p_output[start:end]


def select_touch_device(
    getevent_p_output: str,
    exclude_name_prefixes: tuple[str, ...] = ("og_touch_",),
) -> tuple[str, str, int, int]:
    """Pick the real touchscreen out of `adb shell getevent -p` output.

    Returns the first device (in getevent's own print order) that declares BOTH
    ABS_MT_POSITION_X (0035) and ABS_MT_POSITION_Y (0036) among its ABS axes, skipping
    any whose `name:` starts with one of `exclude_name_prefixes` -- our own UHID virtual
    touchscreen (uhid.py's UhidTouch) registers as `og_touch_<8 hex>` and must never be
    mistaken for the human's real finger. Returns (dev_path, name, x_max, y_max), the
    axis maxima TouchWatcher.start() uses to scale raw device coordinates onto the live
    screen. Raises TouchWatchUnavailable if nothing qualifies.
    """
    for dev_path, block in _iter_device_blocks(getevent_p_output):
        name_m = _NAME_RE.search(block)
        name = name_m.group(1) if name_m else ""
        if any(name.startswith(prefix) for prefix in exclude_name_prefixes):
            continue
        axes = dict(_ABS_AXIS_RE.findall(block))
        if ABS_MT_POSITION_X in axes and ABS_MT_POSITION_Y in axes:
            return dev_path, name, int(axes[ABS_MT_POSITION_X]), int(axes[ABS_MT_POSITION_Y])
    raise TouchWatchUnavailable(
        "no ABS_MT_POSITION_X/Y touch device found in `adb shell getevent -p` output "
        f"(excluded name prefixes: {exclude_name_prefixes!r})"
    )


class TouchWatcher:
    """READ-ONLY observer of the phone's own touchscreen event stream.

    Answers the one question screencap deltas cannot: did the human actually PRESS
    something, and where. It runs `adb shell getevent -lt <device>` and parses it; it
    never writes to /dev/input, never runs `input`, and never installs anything on the
    device (host-side ADB only, same transport contract as the rest of this driver).
    """

    def __init__(
        self,
        adb_path: str,
        serial: str | None,
        screen_size: tuple[int, int],
        *,
        exclude_name_prefixes: tuple[str, ...] = ("og_touch_",),
        gesture_cap: int = 256,
        probe_timeout: float = 10.0,
        close_timeout: float = 2.0,
    ):
        self._adb_path = adb_path
        self._serial = serial
        self._screen_size = screen_size
        self._exclude_name_prefixes = tuple(exclude_name_prefixes)
        self._probe_timeout = float(probe_timeout)
        self._close_timeout = float(close_timeout)

        # Set by start(); readable afterwards for logging (hinge.py's open_session names
        # the chosen device in its one-line success print, mirroring the existing
        # "Hinge ADB preflight OK: ..." style).
        self.device_path: str | None = None
        self.device_name: str | None = None

        # Identity scale until start() rescales against the selected device's real axis
        # maxima. That also happens to be exactly the scale the Pixel 7a needs in
        # practice -- ABS_MT_POSITION_X/Y already run 0..1079 / 0..2399, 1:1 with the
        # 1080x2400 framebuffer (see module docstring) -- so a TouchWatcher fed lines
        # directly (tests) behaves like the real device without first having to fake a
        # getevent -p probe.
        self._scale_x = 1.0
        self._scale_y = 1.0

        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._alive = False

        self._lock = threading.Lock()
        self._gestures: deque[Gesture] = deque(maxlen=gesture_cap)
        self._event_count = 0
        # Every line read off the pipe, matched by _LT_LINE_RE or not. Without this, "the
        # device sent nothing" and "the device sent plenty and this parser understood none of
        # it" are indistinguishable -- both show event_count == 0 -- and they call for
        # opposite responses (accept that the platform withholds the stream vs. fix the regex).
        # Measured on the Pixel 7a / Android 17 2026-08-10: both counters stayed 0 through
        # real screen touches, which is what settled it as a platform restriction rather than a
        # parsing bug. See tools/touch_selftest.py, which reports the two separately.
        self._raw_line_count = 0
        self._unparsed_line_samples: deque[str] = deque(maxlen=3)

        # In-flight gesture-tracking state, all guarded by _lock.
        self._x: float | None = None
        self._y: float | None = None
        self._down_point: tuple[float, float] | None = None
        self._down_pending = False   # DOWN seen, but no coordinate yet to anchor it on
        self._down_t: float | None = None
        self._last_point: tuple[float, float] | None = None
        self._max_travel = 0.0

    # --- lifecycle -----------------------------------------------------
    def start(self) -> None:
        """Probe the device's input nodes, pick the touchscreen, and start the reader
        thread. Raises TouchWatchUnavailable rather than silently running unattached --
        same "explicit operator decision, never a silent downgrade" contract as
        UhidUnavailable (uhid.py)."""
        if self._proc is not None:
            return   # idempotent: already attached, nothing to do
        probe = self._run_getevent_p()
        dev_path, name, x_max, y_max = select_touch_device(probe, self._exclude_name_prefixes)
        self.device_path = dev_path
        self.device_name = name
        w, h = self._screen_size
        # x_max/y_max are the device's own logical maxima (values run 0..max, i.e. max+1
        # distinct positions); scaling by w/(max+1) is a no-op (1.0) exactly when the
        # device already reports 1:1 with the framebuffer, which is the case on the
        # Pixel 7a (max 1079/2399 against a 1080x2400 screen).
        self._scale_x = w / (x_max + 1)
        self._scale_y = h / (y_max + 1)

        argv = [self._adb_path]
        if self._serial:
            argv += ["-s", self._serial]
        # No PTY (`adb shell -tt`) here, deliberately, because the obvious worry about
        # streaming a remote command turns out not to apply. The worry: Android's getevent
        # writes through stdio, and stdio block-buffers (4KB) when stdout is not a terminal,
        # which for a low-rate signal like "the human tapped twice" would strand a whole
        # session's events in a device-side buffer and then lose them at close() -- a silent
        # failure indistinguishable from "nobody touched anything", the one distinction this
        # signal exists to make. Measured on the Pixel 7a 2026-08-10 against the real device:
        # it does NOT buffer. `adb shell getevent -lt` (no PTY) delivered its output at 0.074s
        # and `adb shell -tt getevent -lt` at 0.079s -- identical, so getevent is flushing per
        # line to a plain pipe already. Left as a plain pipe because a PTY would buy nothing
        # and cost something: it rewrites newlines to \r\n and merges stderr into stdout,
        # both of which only make this parser's input harder to reason about.
        argv += ["shell", "getevent", "-lt", dev_path]
        try:
            self._proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except OSError as exc:
            raise TouchWatchUnavailable(f"could not start `{' '.join(argv)}`: {exc}") from exc

        self._alive = True
        self._thread = threading.Thread(
            target=self._read_loop, name="touchwatch-reader", daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the reader and terminate the getevent process. Best-effort and idempotent,
        same as UhidTouch.close(): session teardown must not blow up just because the
        stream had already died on its own."""
        self._alive = False
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 — best-effort; the process may already be gone
                pass
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._close_timeout)
        if proc is not None:
            try:
                proc.wait(timeout=self._close_timeout)
            except Exception:  # noqa: BLE001 — a stuck adb process must not hang teardown
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    # --- reading ---------------------------------------------------------
    def _run_getevent_p(self) -> str:
        argv = [self._adb_path]
        if self._serial:
            argv += ["-s", self._serial]
        argv += ["shell", "getevent", "-p"]
        try:
            result = subprocess.run(
                argv, capture_output=True, timeout=self._probe_timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TouchWatchUnavailable(
                f"could not probe input devices (`{' '.join(argv)}`): {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", "replace").strip()
            raise TouchWatchUnavailable(
                f"`{' '.join(argv)}` failed (exit {result.returncode}): {stderr}"
            )
        return result.stdout.decode("utf-8", "replace")

    def _read_loop(self) -> None:
        """Runs on a daemon thread for the life of the getevent process. Must NEVER raise
        into the interpreter: an exception on a daemon thread with no handler just
        vanishes silently while `alive` stays True forever, which would make "the stream
        died" indistinguishable from "genuinely idle but still working" to every caller
        that checks `alive`/`event_count`. Routing every exit path -- normal EOF, a read
        error, anything -- through this `finally` is what keeps `alive` honest."""
        proc = self._proc
        try:
            if proc is None or proc.stdout is None:
                raise TouchWatchUnavailable(
                    "getevent reader started without a live process/stdout stream")
            for line in proc.stdout:
                self._feed_line(line)
        except Exception:  # noqa: BLE001 — see docstring: must never raise into the process
            pass
        finally:
            self._alive = False

    def _feed_line(self, line: str) -> None:
        """Parse one `getevent -lt` line and fold it into gesture-tracking state.

        Deliberately usable standalone (no Popen/thread required) so a canned transcript
        can be fed straight through in tests -- normal operation only ever calls it from
        _read_loop, one physical line at a time.
        """
        with self._lock:
            self._raw_line_count += 1
        m = _LT_LINE_RE.match(line.strip())
        if not m:
            # Keep only a scrubbed *format* sample for incident reports.  It contains no
            # coordinates or profile pixels, but lets a future Android toolbox spelling be
            # fixed from a report instead of asking the owner to reproduce it in a terminal.
            sample = re.sub(r"\b(?:[0-9a-fA-F]{2,}|\d+(?:\.\d+)?)\b", "<n>", line.strip())
            sample = re.sub(r"\s+", " ", sample)
            if sample:
                with self._lock:
                    self._unparsed_line_samples.append(sample[:180])
            return
        ev, code, value = m.group("ev"), m.group("code"), m.group("value")
        with self._lock:
            self._event_count += 1
            numeric = len(ev) == 4 and all(ch in "0123456789abcdefABCDEF" for ch in ev)
            if (ev == "EV_ABS" or (numeric and ev.lower() == EV_ABS)) and (
                    code == "ABS_MT_POSITION_X" or code.lower() == ABS_MT_POSITION_X):
                self._note_axis_locked("x", value)
            elif (ev == "EV_ABS" or (numeric and ev.lower() == EV_ABS)) and (
                    code == "ABS_MT_POSITION_Y" or code.lower() == ABS_MT_POSITION_Y):
                self._note_axis_locked("y", value)
            elif (ev == "EV_KEY" or (numeric and ev.lower() == EV_KEY)) and (
                    code == "BTN_TOUCH" or code.lower() == BTN_TOUCH):
                if value == "DOWN" or value.lower() in {"1", "00000001"}:
                    self._begin_gesture_locked()
                elif value == "UP" or value.lower() in {"0", "00000000"}:
                    self._end_gesture_locked()
            elif ((ev == "EV_ABS" or (numeric and ev.lower() == EV_ABS)) and
                  (code == "ABS_MT_TRACKING_ID" or code.lower() == ABS_MT_TRACKING_ID) and
                  value.lower() == "ffffffff"):
                # Devices that omit BTN_TOUCH altogether signal release by resetting the
                # multitouch tracking ID to -1 (the ffffffff 32-bit wraparound) instead --
                # same boundary, different spelling. goodix_ts0 DOES declare BTN_TOUCH
                # (KEY 014a, confirmed live on the Pixel 7a), so this is a defensive
                # fallback path, not the primary one.
                self._end_gesture_locked()

    # --- gesture state machine (all callers hold _lock) ------------------
    def _note_axis_locked(self, axis: str, raw_hex: str) -> None:
        try:
            raw = int(raw_hex, 16)
        except ValueError:
            return   # a malformed value must not take the reader thread down with it
        scaled = raw * (self._scale_x if axis == "x" else self._scale_y)
        if axis == "x":
            self._x = scaled
        else:
            self._y = scaled
        if self._down_pending and self._x is not None and self._y is not None:
            # A DOWN arrived before this gesture's first coordinate pair, so _begin_gesture_locked
            # had nothing real to anchor on. Adopt the first complete pair as the down point.
            # Without this the gesture would anchor on whatever _x/_y happened to hold -- the
            # PREVIOUS gesture's release point, or (0.0, 0.0) for the very first touch of a
            # session. Either one puts the down point somewhere the finger never was, which
            # inflates travel_px (a genuine tap reads as a long drag) and misplaces
            # `down`/`up`. goodix_ts0 happens to emit ABS_MT_POSITION_X/Y before BTN_TOUCH
            # within the same SYN frame, so on the measured device this never fires -- it is
            # here because the opposite ordering is equally legal and a wrong anchor is
            # silent, not loud.
            self._down_pending = False
            self._down_point = (self._x, self._y)
            self._last_point = self._down_point
            self._max_travel = 0.0
            return
        if self._down_point is not None and self._x is not None and self._y is not None:
            point = (self._x, self._y)
            self._last_point = point
            dist = math.hypot(point[0] - self._down_point[0], point[1] - self._down_point[1])
            if dist > self._max_travel:
                self._max_travel = dist

    def _begin_gesture_locked(self) -> None:
        self._down_t = time.monotonic()
        self._max_travel = 0.0
        if self._x is None or self._y is None:
            # No coordinate belonging to THIS gesture has been seen yet -- either the first
            # touch of the session, a stream attached mid-gesture, or (the common case) simply
            # a device that emits BTN_TOUCH before the position pair inside a SYN frame, since
            # _end_gesture_locked clears _x/_y so a finished gesture's coordinates can never
            # leak into the next one. Anchor nothing yet: _note_axis_locked adopts the first
            # complete pair as the down point via _down_pending. Anchoring on a fabricated
            # (0, 0) -- or on the previous gesture's release point -- would make a stationary
            # tap read as a long drag, and a tap on the pass-X is the ONLY affirmative PASS
            # evidence hinge.py's gesture corroboration accepts.
            self._down_pending = True
            self._down_point = None
            self._last_point = None
            return

        self._down_pending = False
        self._down_point = (self._x, self._y)
        self._last_point = self._down_point

    def _end_gesture_locked(self) -> None:
        if self._down_point is None or self._down_t is None:
            # An UP with no anchored DOWN: either the stream started mid-gesture, or a DOWN
            # arrived and the finger lifted before a single coordinate pair ever landed. Emit
            # nothing (a gesture with no known position corroborates nothing), but clear the
            # pending flag so the NEXT gesture's first coordinate isn't consumed as this
            # dead one's down point.
            self._down_pending = False
            self._down_t = None
            self._x = self._y = None
            return
        up_point = self._last_point if self._last_point is not None else self._down_point
        gesture = Gesture(
            t_down=self._down_t,
            t_up=time.monotonic(),
            down=(int(round(self._down_point[0])), int(round(self._down_point[1]))),
            up=(int(round(up_point[0])), int(round(up_point[1]))),
            travel_px=self._max_travel,
        )
        self._gestures.append(gesture)
        self._down_point = None
        self._down_pending = False
        self._down_t = None
        self._last_point = None
        self._max_travel = 0.0
        # Retire this gesture's coordinates too. A touchscreen reports absolute positions and
        # simply stops reporting on lift, so _x/_y would otherwise still hold the release
        # point when the NEXT down arrives -- and on a device that emits BTN_TOUCH before its
        # position pair, that stale point silently becomes the next gesture's down anchor.
        self._x = self._y = None

    # --- public reads ----------------------------------------------------
    def gestures_since(self, t: float) -> list[Gesture]:
        """Gestures whose DOWN happened at or after host time `t`. Both `t` and each
        gesture's `t_down` are `time.monotonic()` readings from this same process, so
        they compare directly with no clock reconciliation."""
        with self._lock:
            return [g for g in self._gestures if g.t_down >= t]

    @property
    def event_count(self) -> int:
        """Total getevent lines successfully parsed, ever -- a health signal. Callers
        (hinge.py's gesture corroboration) treat 0 for a whole run as proof the stream
        itself isn't delivering anything on this device, distinct from "no gestures
        happened to land in this window."""
        with self._lock:
            return self._event_count

    @property
    def raw_line_count(self) -> int:
        """Lines read off the stream, whether or not they parsed -- see _raw_line_count."""
        with self._lock:
            return self._raw_line_count

    @property
    def unparsed_line_samples(self) -> tuple[str, ...]:
        """A tiny redacted format sample for diagnosing an unsupported getevent spelling."""
        with self._lock:
            return tuple(self._unparsed_line_samples)

    @property
    def alive(self) -> bool:
        return self._alive
