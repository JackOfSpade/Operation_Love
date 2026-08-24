"""Genuine-to-the-kernel touch via a non-root UHID virtual touchscreen.

High-fidelity input for the physical Pixel: a virtual multitouch digitizer created
through the STOCK `/system/bin/hid` tool over `/dev/uhid` (writable by the
unprivileged `shell` domain — SELinux `allow shell uhid_device`, shell in group
`uhid`). No root, no installed helper/APK, no accessibility service. Unlike the
`adb input` transport (Adb), UHID events flow through the real kernel input
pipeline, so to the app they are genuine hardware touches: `TOOL_TYPE_FINGER`,
VARIABLE pressure, VSYNC-batched, not injection-flagged. Motion comes from
human_motion.py (curved path, asymmetric-lognormal velocity, OU tremor, dynamic
pressure), calibrated to real getevent data.

Delivery model — PER-GESTURE FILE (proven on the Pixel 7a, see memory
uhid-touch-recipe). `hid <file>` reads a JSON file (register + delay + report/delay
commands), owns the virtual device for the file's duration, then destroys it at EOF.
A *persistent* device would need to hold a FIFO open, but `mkfifo` is SELinux-denied
to the shell domain (verified on-device) — so each gesture re-creates the device via
its own file: register -> short enumerate delay -> the report stream -> a flush delay.
Cost: ~enumerate_ms re-enumeration per gesture; invisible to apps (they can't see
input-device add/remove). The file is a regular file written with `cat` (allowed).

Descriptor: single-finger multitouch digitizer, Usage(Touch Screen), with
Tip/Confidence/ContactID/X/Y/Pressure + Contact Count, and CRUCIALLY no
Contact-Count-Maximum *feature* report (that triggers a kernel GET_FEATURE the hid
stream can't answer, killing the device mid-enumeration). X/Y logical maxima are
built from the live screen size (not hard-coded) so coordinates stay 1:1 on any
resolution. NOTE: the descriptor carries pressure but not contact-size; size in the
sample stream is intentionally unused at the report level.

Addendum 2026-08-24 -- a SECOND delivery mechanism, PersistentUhidTouch, was added below.
It is additive: everything above this paragraph still describes UhidTouch exactly as it
has always behaved, and this module's per-gesture-file design remains the default for
`touch_backend: auto`/`uhid`. PersistentUhidTouch instead holds ONE registered virtual
device open for a whole session via a persistent PTY-backed `hid` process, reachable only
through an explicit `touch_backend: uhid_persistent` (see hinge.py's `_make_touch`) --
see that class's own docstring for the live validation this rests on, and for the one
edge case it does not directly prove.
"""
from __future__ import annotations

import json
import random
import secrets
import shlex
import subprocess
import threading
import time

from ..human_motion import REPORT_HZ, plan_swipe, plan_tap
from .adb import Adb, AdbError, clamp_xy, scroll_x
from .base import DriverClosed, _time_bucket


class UhidUnavailable(RuntimeError):
    """UHID can't be used on this device (e.g. no /system/bin/hid).

    This does NOT authorise a silent downgrade. AndroidDriver._make_touch treats it as
    fatal under `touch_backend: auto` (the default) and under `uhid`, because the adb
    `input` transport cannot vary pressure and swapping to it invisibly would weaken the
    touch signature for an unknown length of time. Only an explicit `touch_backend: adb`
    accepts that transport. Distinct from DriverClosed (a real device loss)."""


def _le16(v: int) -> list[int]:
    v = max(0, int(v))
    return [v & 0xFF, (v >> 8) & 0xFF]


def _build_descriptor(w: int, h: int) -> list[int]:
    """Single-finger multitouch touchscreen digitizer with X/Y logical maxima set to
    the live screen (w-1, h-1), so device coordinates map 1:1 to pixels."""
    return [
        0x05, 0x0D, 0x09, 0x04, 0xA1, 0x01, 0x85, 0x01, 0x09, 0x22, 0xA1, 0x02,
        0x09, 0x42, 0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x01, 0x81, 0x02,   # Tip Switch
        0x09, 0x47, 0x81, 0x02,                                                    # Confidence
        0x95, 0x06, 0x81, 0x03,                                                    # 6-bit pad
        0x95, 0x01, 0x75, 0x08, 0x09, 0x51, 0x25, 0x7F, 0x81, 0x02,                # Contact ID
        0x05, 0x01, 0x26, *_le16(w - 1), 0x75, 0x10, 0x09, 0x30, 0x81, 0x02,       # X 0..w-1
        0x26, *_le16(h - 1), 0x09, 0x31, 0x81, 0x02,                               # Y 0..h-1
        0x05, 0x0D, 0x26, 0xFF, 0x00, 0x75, 0x08, 0x09, 0x30, 0x81, 0x02,          # Tip Pressure 0..255
        0xC0,
        0x05, 0x0D, 0x09, 0x54, 0x95, 0x01, 0x75, 0x08, 0x25, 0x7F, 0x81, 0x02,    # Contact Count
        0xC0,
    ]


def _report(sample, w: int, h: int) -> list[int]:
    """A 9-byte HID input report (incl. report id) for one TouchSample, clamped to the
    screen (clamp_xy, shared with Adb._clamp -- see that function's docstring for why),
    pressure scaled from normalized 0..1 to raw 0..255 (contact-size dropped)."""
    x, y = clamp_xy(sample.x, sample.y, w, h)
    p = max(0, min(255, int(round(sample.pressure * 255))))
    flags = 0x03 if sample.tip else 0x00          # bit0 Tip, bit1 Confidence
    count = 1 if sample.tip else 0
    return [0x01, flags, 0x00, x & 0xFF, (x >> 8) & 0xFF, y & 0xFF, (y >> 8) & 0xFF, p, count]


def _register_command(name: str, vid: int, pid: int, w: int, h: int) -> dict:
    """The single `register` command every gesture-script variant sends -- identical fields
    (name/vid/pid/bus/descriptor) whether it is issued once PER GESTURE (UhidTouch's
    per-gesture-file design, below) or once PER SESSION (PersistentUhidTouch, further
    below): the virtual device this dict describes to `/system/bin/hid` is the same either
    way, only how often it gets sent differs. Factored out so the two classes cannot drift
    apart on what "register" means."""
    return {"id": 1, "command": "register", "name": name, "vid": vid, "pid": pid,
            "bus": "usb", "descriptor": _build_descriptor(w, h)}


def _report_stream_commands(samples, w: int, h: int, flush_ms: int) -> list[dict]:
    """report/delay commands for one gesture's sample stream, ending with a flush delay --
    the part of a gesture script that is identical regardless of whether the
    register+enumerate-delay in front of it happens once per gesture (UhidTouch) or once per
    session (PersistentUhidTouch). Factored out of UhidTouch's original inline loop so the
    two classes cannot drift apart on the one thing that MUST stay identical between them:
    the actual HID report bytes and inter-report timing delivered to the kernel are the same
    planned motion (human_motion.plan_swipe/plan_tap, unchanged), no matter which delivery
    mechanism carries them. `flush_ms` is a parameter, not computed here, because each
    caller derives it from ITS OWN jittered range (UhidTouch.flush_ms /
    PersistentUhidTouch.flush_ms happen to share the same default today, but this function
    has no opinion on that -- it only appends whatever duration it is given)."""
    cmds: list[dict] = []
    for i, s in enumerate(samples):
        cmds.append({"id": 1, "command": "report", "report": _report(s, w, h)})
        if i < len(samples) - 1:
            dt_ms = max(1, int(round((samples[i + 1].t - s.t) * 1000.0)))
            cmds.append({"id": 1, "command": "delay", "duration": dt_ms})
    cmds.append({"id": 1, "command": "delay", "duration": flush_ms})   # let the last event flush
    return cmds


class UhidTouch:
    """A non-root virtual touchscreen. Mirrors Adb's tap/swipe/scroll_up surface so a
    driver can swap transports. Each gesture is delivered as its own `hid <file>` run
    (see module docstring). Uses the given Adb for device I/O."""

    def __init__(self, adb: Adb, *, hz: float = REPORT_HZ, rng=None, jitter_px: float = 2.2,
                 width_px: float = 180.0, enumerate_ms: int = 700, flush_ms: int = 150,
                 file_path: str = "/data/local/tmp/og_uhid_g.json",
                 name: str | None = None, vid: int = 0x18D1, pid: int = 0x0C10):
        self.adb = adb
        self.hz = float(hz)
        self._rng = rng if rng is not None else random
        self.jitter_px = float(jitter_px)
        self.width_px = float(width_px)
        self.enumerate_ms = int(enumerate_ms)
        self.flush_ms = int(flush_ms)
        self.file_path = file_path
        # A run keeps one identity, but unrelated sessions do not all register the same
        # globally fixed virtual-device name. Explicit names remain exact for calibration.
        self.name = name if name is not None else f"og_touch_{secrets.token_hex(4)}"
        self.vid, self.pid = vid, pid
        self._lock = threading.Lock()   # serialize gestures: one `hid <file>` run at a time

    # --- lifecycle (no persistent device; just geometry/cleanup) --------
    def open(self) -> None:
        # The probe ALWAYS exits 0 (the `|| echo no` branch), so absence of /system/bin/hid is
        # signaled purely via stdout ("no") -> UhidUnavailable (clean fall back), NOT via a
        # non-zero remote exit that modern adb propagates and Adb._run surfaces as AdbError.
        # A genuine device loss still raises DriverClosed from the shell call; a real command
        # failure still raises AdbError. (test -e + && would exit 1 when absent -> AdbError.)
        if "yes" not in self.adb.shell("[ -e /system/bin/hid ] && echo yes || echo no"):
            raise UhidUnavailable("/system/bin/hid not present")   # device-loss raises DriverClosed instead
        self.adb.screen_size()                    # cache geometry

    def close(self) -> None:
        try:
            self.adb.shell(f"rm -f {shlex.quote(self.file_path)}")
        except Exception:  # noqa: BLE001 — best-effort cleanup must not mask the real outcome
            pass

    # --- gesture -> hid file -------------------------------------------
    def _gesture_script(self, samples) -> bytes:
        w, h = self.adb.screen_size()
        # File-backed hid cannot be made persistent under the stock shell constraints. Keep
        # the proven baseline delays, but vary them upward in a bounded range per gesture so
        # the registration/flush cadence is not a single fixed pair.
        enumerate_ms = self.enumerate_ms + self._rng.randint(0, 180)
        flush_ms = self.flush_ms + self._rng.randint(0, 90)
        # Register+enumerate-delay (once per gesture, this class's whole design -- see module
        # docstring), then the shared report/delay/flush loop (_report_stream_commands, also
        # used by PersistentUhidTouch's per-gesture command builder further below). Extracting
        # that loop changed WHERE this code lives, not what it produces: the dicts built here
        # are byte-for-byte identical, in the same order, to what this method built inline
        # before 2026-08-24 -- pinned by tests/test_uhid.py's existing structural assertions,
        # which were not touched by that extraction and still pass unchanged.
        cmds = [_register_command(self.name, self.vid, self.pid, w, h),
                {"id": 1, "command": "delay", "duration": enumerate_ms}]
        cmds.extend(_report_stream_commands(samples, w, h, flush_ms))
        return ("\n".join(json.dumps(c) for c in cmds) + "\n").encode()

    def _run_gesture(self, samples, *, _timing: dict[str, float] | None = None) -> None:
        """`_timing`, when given, is the SAME per-gesture stamps dict `hinge.py`'s
        `_scroll`/`_swipe` are already accumulating into (see their own docstrings) -- this
        method just contributes its own leaf buckets to it, never wraps a span another layer
        already timed.  `None` (every caller before 2026-08-23, and every caller today whose
        `self.touch` is not one of the two real transports -- see hinge.py's
        `_touch_supports_timing`) makes `_time_bucket` a true no-op: not one extra
        `time.monotonic()` call, exactly as before this ledger existed.

        The two bucket names below are this transport's answer to "how many `adb shell` round
        trips does one gesture cost": `uhid_write_file_s` (write the gesture's HID script) and
        `uhid_hid_run_s` (run it -- this is the one call that actually BLOCKS for the gesture's
        duration; the module docstring's "enumerate -> report stream -> flush" all happens
        device-side inside this single call, so it cannot be decomposed further without either
        a second device call this ledger is not allowed to add, or an on-device timestamp
        stream this transport was deliberately never given).

        `uhid_hid_run_s` NOW COVERS CLEANUP TOO (2026-08-24, owner-approved marginal lever,
        ~2s/profile). Until this date `hid <file>` and `rm -f <file>` were two SEPARATE `adb
        shell` round trips -- the second, in a `finally`, unconditionally deleting the script
        file after the first ran (or failed). They are now one remote shell invocation:

            hid <file>; ec=$?; rm -f <file>; exit $ec

        `ec` captures `hid`'s OWN exit status before `rm -f` ever runs, and the compound script
        exits with `$ec`, never `rm -f`'s -- so a `hid` failure still surfaces as a non-zero
        remote exit (Adb.shell -> AdbError -> the same DriverClosed below, same message) no
        matter what `rm -f` itself does, and a clean `hid` run still reports success even if
        `rm -f` cannot remove the file (never something a caller could act on anyway -- see
        close()'s identical best-effort swallow of that same failure). REJECTED shape: `hid
        <file>; rm -f <file>` with no `$?` capture -- that reports `rm -f`'s exit status, not
        `hid`'s, so a real `hid` failure followed by a successful `rm -f` would silently read as
        success. (The POSIX semantics behind this choice are pinned with plain `sh -c` in
        tests/test_uhid.py -- no device needed, this isn't Android-specific.)

        One consequence: there is no longer a separate `uhid_cleanup_s` bucket. The two costs
        now happen inside ONE opaque remote call with no on-device timestamp between them to
        split on -- reporting a fake split would misattribute worse than admitting they're
        fused, so `uhid_hid_run_s` honestly reports the WHOLE compound script's wall clock.
        `hid`'s own device-side cost still dominates it in practice: the 2026-08-23 instrumented
        run that motivated this ledger measured 2.706s for `hid` alone against 0.058s for the
        separate cleanup call it used to need -- `rm -f` was always the rounding error, not the
        thing worth its own bucket."""
        if not samples:
            return
        with _time_bucket(_timing, "uhid_script_build_s"):
            script = self._gesture_script(samples)
        quoted = shlex.quote(self.file_path)
        # Serialize: each gesture is its own `hid <file>` run; a concurrent caller must not
        # truncate the file while another gesture's hid is still reading it.
        with self._lock:
            # True once the script file is confirmed written, i.e. once the compound
            # `hid ...; rm -f ...` command below is about to run -- that command now owns
            # cleanup for every path that reaches it (success OR a `hid` failure; `;` sequences
            # unconditionally, unlike `&&`). False only when BOTH write_file attempts below
            # failed: in that one narrow path the compound command never runs at all, so a
            # `cat` redirect that errored partway through could still have left a truncated
            # script file on the device with nothing downstream left to remove it -- the
            # `finally` covers exactly that one remaining case with its own best-effort
            # `rm -f`, same as every path got before this collapse.
            wrote_file = False
            try:
                # Rewriting the file is safe to retry: no input delivery can begin until the
                # separate `hid` command below starts.  Once `hid` has started, however, a
                # non-zero exit is ambiguous -- the kernel may already have received part or all
                # of the report stream.  Replaying that script could turn one irreversible tap
                # into two, so execution failure always stops without a second `hid` invocation.
                for attempt in (1, 2):
                    try:
                        with _time_bucket(_timing, "uhid_write_file_s"):
                            self.adb.write_file(self.file_path, script)
                    except AdbError as exc:
                        if attempt == 2:
                            raise DriverClosed(
                                f"UHID gesture file write failed before delivery: {exc}") from exc
                        time.sleep(0.3)
                        continue
                    break
                wrote_file = True
                try:
                    with _time_bucket(_timing, "uhid_hid_run_s"):
                        # See this method's docstring for why this is `; ec=$?; ...; exit $ec`
                        # and not the naive (rejected) `hid ...; rm -f ...`.
                        self.adb.shell(f"hid {quoted}; ec=$?; rm -f {quoted}; exit $ec")
                except AdbError as exc:
                    raise DriverClosed(
                        "UHID gesture delivery became uncertain after `hid` started; refusing "
                        f"to replay the gesture: {exc}") from exc
            finally:
                if not wrote_file:
                    try:
                        self.adb.shell(f"rm -f {quoted}")
                    except Exception:  # noqa: BLE001 — best-effort cleanup
                        pass

    # --- public surface (matches Adb) ----------------------------------
    def tap(self, x: int, y: int) -> None:
        self._run_gesture(plan_tap(x, y, hz=self.hz, rng=self._rng))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 450, *,
             _timing: dict[str, float] | None = None) -> None:
        # Keep interface parity with Adb.  450ms is this transport's ordinary Fitts-law
        # gesture class; a caller may explicitly request a shorter measured flick, which is
        # scaled in the pure planner without compromising curved endpoints or pressure data.
        #
        # `_timing` (see `_run_gesture`'s docstring): `uhid_plan_swipe_s` isolates the pure-CPU
        # cost of synthesizing the curved path/velocity/tremor/pressure sample stream
        # (human_motion.plan_swipe) from the device I/O `_run_gesture` goes on to do with the
        # result -- the "is the humanized motion planner itself slow" question this ledger
        # exists to answer, kept as its own bucket rather than folded into the device calls it
        # has nothing to do with.
        duration_scale = max(0.20, min(2.0, float(duration_ms) / 450.0))
        with _time_bucket(_timing, "uhid_plan_swipe_s"):
            samples = plan_swipe(x1, y1, x2, y2, hz=self.hz, jitter_px=self.jitter_px,
                                 width_px=self.width_px, duration_scale=duration_scale,
                                 rng=self._rng)
        self._run_gesture(samples, _timing=_timing)

    def scroll_up(self, distance_frac: float = 0.55, x_frac: float = 0.5, *,
                 _timing: dict[str, float] | None = None) -> None:
        # x jitter shared with Adb.scroll_up via adb.scroll_x() (HINGE-04): UHID is the
        # genuine, proven transport, so it must not be the one emitting a pixel-identical
        # column every scroll.
        #
        # `_timing`'s "screen_size_s" bucket is the SAME key hinge.py's `_scroll` already
        # writes (accumulated, not overwritten -- see _time_bucket's own docstring): a prior
        # analysis suspected `adb.screen_size()` is called twice per gesture, once in the
        # driver and once here in the transport. That call count is confirmed by this second
        # site existing at all -- but `Adb.screen_size()` caches after its first-ever call for
        # the whole session (see adb.py), so every one of these calls after session open is a
        # dict lookup, not a device round trip. This bucket is how that gets PROVEN rather than
        # assumed: if it ever reads meaningfully above zero, the cache assumption broke.
        with _time_bucket(_timing, "screen_size_s"):
            w, h = self.adb.screen_size()
        x = scroll_x(w, x_frac)
        self.swipe(x, int(h * (0.5 + distance_frac / 2)), x, int(h * (0.5 - distance_frac / 2)),
                  _timing=_timing)


# The persistent process's local teardown timeout (PersistentUhidTouch.close_timeout's
# default). This is a LOCAL `proc.kill()` + `proc.wait()`, not a remote adb round trip --
# `proc.kill()` sends SIGKILL to the local `adb shell` client, and the remote virtual device
# dies as a side effect of that connection dropping (empirically verified, 4 separate live
# kills, 2026-08-24: `/proc/bus/input/devices` confirmed EMPTY after every one, no dangling
# device). That is why this is nowhere near Adb.default_timeout (10.0s, adb.py) -- that
# timeout bounds an actual blocking round trip over USB to a REMOTE command, which can
# legitimately take a while; this one only has to wait for the local OS to finish reaping a
# process that was just SIGKILLed, which is normally a matter of milliseconds. 3.0s follows
# touchwatch.py's own `close_timeout: float = 2.0` precedent for the identical shape (a
# persistent Popen this driver owns, torn down at session close) -- kept a little larger here
# because TouchWatcher's close() tries a graceful `terminate()` FIRST and only escalates to
# `kill()` if that stalls, so its 2.0s budget is "how long to wait for graceful shutdown"; this
# class goes straight to `kill()` (see PersistentUhidTouch's own docstring: no graceful
# in-protocol shutdown command exists or was found), so its whole budget is "how long teardown
# of an already-killed process may reasonably take" -- a smaller number in principle, but 3.0s
# costs nothing extra in the common case (wait() returns as soon as the process is reaped, it
# does not sleep the full budget) and buys more margin against a loaded CI box or a slow PTY
# teardown than shaving it down would save.
_DEFAULT_CLOSE_TIMEOUT_S = 3.0

# How often PersistentUhidTouch._run_gesture polls the remote process during its post-write
# wait, instead of one blind time.sleep(wait_s) followed by a single poll() at the end. A
# never-substitute review (2026-08-24) flagged that a single end-of-wait check leaves a
# process that died EARLY in a gesture's wait window undetected for up to the rest of that
# gesture's own duration, plus the guard check of whatever gesture comes next -- several
# consecutive gestures could be silently "delivered" against an already-dead session before
# the class ever notices. Slicing the wait into small steps and polling after each one
# shrinks that detection-lag window to roughly one interval, without changing what a
# poll()-based check actually PROVES (still an inference from the local adb client noticing a
# dead remote link, never a positive confirmation the device received anything -- see
# PersistentUhidTouch's class docstring, "not directly proven"). 30ms is short enough to
# meaningfully shrink the window (well under human-perceptible gesture timing, and small next
# to the multi-hundred-ms waits typical gestures encode) without turning every gesture into a
# many-hundred-call poll() loop.
_PERSISTENT_POLL_INTERVAL_S = 0.03


class PersistentUhidTouch:
    """A non-root virtual touchscreen held open for an ENTIRE SESSION instead of being
    re-registered per gesture. Same public surface as UhidTouch (open/close/tap/swipe/
    scroll_up, identical signatures) so AndroidDriver can select either transport
    interchangeably (see hinge.py's `_make_touch`) -- this is the SECOND real touch
    transport, opted into only by an explicit `touch_backend: uhid_persistent`, never by
    `auto`/`uhid`. It delivers the exact SAME planned motion as UhidTouch
    (human_motion.plan_tap/plan_swipe, `_build_descriptor`, `_report` -- all reused
    unchanged, imported from this same module): this class is a new DELIVERY mechanism for
    that motion, never a new motion model.

    Delivery model -- PERSISTENT PTY (validated live on the Pixel 7a, 2026-08-24; additive
    to, never a replacement for, UhidTouch's per-gesture-file design above). UhidTouch's own
    module docstring already ruled out a device-side named pipe (`mkfifo` is SELinux-denied
    to the `shell` domain). What this addendum rules out, and then finds, is different: a
    PLAIN pipe cannot be REOPENED by path on this device either -- `adb shell "hid
    /dev/stdin"` measured ENXIO live, 2026-08-24. What DOES work: `adb shell -tt "hid
    /proc/self/fd/0"`. The `-tt` forces a PTY even though nothing about this call looks
    interactive; `/proc/self/fd/0` gives the remote `hid` process a path back to ITS OWN
    stdin (the PTY's slave side), which is exactly the thing a plain reopened pipe could not
    give it. Spawned host-side via `subprocess.Popen(stdin=PIPE)`, the remote
    `com.android.commands.hid.Hid` process (confirmed via logcat: `/system/bin/hid` is a
    379-byte shell script that execs `app_process ... com.android.commands.hid.Hid` -- a
    thin Java wrapper, not a native binary) stayed alive across one `register` + one
    `delay`, a SECOND `delay` with NO re-registration ~1s later, and TWO full read-scroll
    gesture streams (built from the exact production `human_motion.plan_swipe`/`_report`
    this class still uses) sent back to back through the same connection with no
    re-registration between them. A THIRD test sent ONE real read-scroll against LIVE Hinge
    (profile "Meg", card at scroll-top) through the persistent connection: a screenshot
    before/after CONFIRMED the scroll landed correctly -- the page advanced from the top
    photo to the prompt cards below, exactly as the per-gesture `hid <file>` transport
    already does, with no mis-tap and no stray action.

    Why this works at all: `com.android.commands.hid.Hid`'s command loop reads
    newline-delimited JSON (register/delay/report) and, for a LIVE/STREAMING source (a PTY,
    not a static file), BLOCKS waiting for the next command instead of exiting the moment
    there are no more bytes right now -- a plain file hits real EOF and the process exits
    normally once it does, which is UhidTouch's whole reason for re-registering every
    gesture. Given an invalid path it throws an uncaught `FileNotFoundException` that
    crashes the WHOLE process with exit code 1 (confirmed via logcat during earlier
    non-persistent testing) -- there is no in-protocol error recovery on the remote side,
    which is why this class treats a dead process or a failed write as terminal
    (DriverClosed) rather than something to retry against.

    What is validated and what is NOT, precisely, matters here, because this class trades
    UhidTouch's ~enumerate_ms-per-gesture re-registration cost for a failure mode UhidTouch
    never had to consider: a session-long remote process whose death must be inferred from
    the HOST side, with no on-device helper to ask directly.
      - Clean teardown: EMPIRICALLY VERIFIED, 4 separate live kills, 2026-08-24 -- a local
        `proc.kill()` reliably tears the kernel-side virtual device down every time tested;
        `/proc/bus/input/devices` confirmed EMPTY after each one. No graceful in-protocol
        "unregister" command exists or was found; close() below does not go looking for one.
      - A REMOTE-side crash being noticed LOCALLY (via `poll()`) was NOT directly proven --
        a `ps`-based PID lookup meant to kill the remote process independently and watch the
        local side notice failed to reliably identify the right PID via this device's
        available `ps` output columns, and the live test's own safety check correctly
        refused to blind-kill an unverified PID rather than risk killing the wrong process.
        This is INFERRED, not proven, from a mechanism the CURRENTLY SHIPPED code already
        depends on: adb's `shell -t`/`-tt` protocol propagates the remote command's exit
        back to the local `adb shell` client process -- UhidTouch._run_gesture's
        non-persistent `self.adb.shell(f"hid {quoted}")` call already relies on exactly
        this (a remote non-zero exit -> a local `AdbError` -> the same `DriverClosed` its
        own docstring describes). Every `poll()`-based death check in this class rests on
        that same inference, unverified independently, rather than on a live-proven
        mechanism of its own. `_run_gesture`'s post-write wait polls in short slices
        (`_PERSISTENT_POLL_INTERVAL_S`) rather than once at the very end -- added 2026-08-24
        in response to a never-substitute review -- which shrinks how long a death can go
        unnoticed, but it is still exactly this same inference, checked more often, not a
        different or stronger kind of evidence.

    Config validation and `_make_touch` gate this behind an explicit `touch_backend:
    uhid_persistent` for exactly that reason (see config.yaml's own LIVE-VERIFY note on this
    key): what was validated live above is real, but it is a handful of gestures against one
    profile, not the scale of confidence this repo's LIVE-VERIFY convention reserves for
    changing a DEFAULT. `auto` and `uhid` keep selecting UhidTouch, byte-for-byte unchanged.
    """

    def __init__(self, adb: Adb, *, hz: float = REPORT_HZ, rng=None, jitter_px: float = 2.2,
                 width_px: float = 180.0, enumerate_ms: int = 700, flush_ms: int = 150,
                 name: str | None = None, vid: int = 0x18D1, pid: int = 0x0C10,
                 close_timeout: float = _DEFAULT_CLOSE_TIMEOUT_S):
        self.adb = adb
        self.hz = float(hz)
        self._rng = rng if rng is not None else random
        self.jitter_px = float(jitter_px)
        self.width_px = float(width_px)
        self.enumerate_ms = int(enumerate_ms)
        self.flush_ms = int(flush_ms)
        # Same "a run keeps one identity, unrelated sessions do not all collide on a single
        # globally fixed name" reasoning as UhidTouch.name (see its own comment) -- this is
        # the SAME kind of virtual device, just held open longer.
        self.name = name if name is not None else f"og_touch_{secrets.token_hex(4)}"
        self.vid, self.pid = vid, pid
        self.close_timeout = float(close_timeout)
        # RLock, not Lock: open()'s own failure paths call _kill_proc_best_effort() while
        # ALREADY holding this lock (see both methods below) -- a plain Lock would deadlock
        # on that same-thread reentry. This lock now guards every mutator of self._proc/
        # self._dead (open, close, _kill_proc_best_effort, _run_gesture), not just
        # gesture-vs-gesture ordering as the original comment here said: a thread-safety
        # review (2026-08-24) found that close() running on a second thread while a gesture
        # was mid-flight on this one could observe self._proc flip to None between
        # _run_gesture's guard check (line ~596) and its later use a few lines on, raising a
        # raw AttributeError instead of the documented DriverClosed contract every OTHER
        # failure path in this class uses. Not reachable from today's single call site
        # (Worker._finish_session calls close() from the same thread that ran every gesture,
        # only after that thread's own loop has already exited) but nothing about the class
        # itself enforced that invariant, so a future caller (e.g. a stall/timeout watchdog
        # closing a hung session from a second thread) would have silently inherited the gap.
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        # Sticky "this session is unusable" flag. `self._proc.poll() is not None` alone is
        # not quite enough to detect every dead-session case: a write that raised
        # (BrokenPipeError/ValueError) means delivery is ALREADY ambiguous even in the brief
        # window before the OS has finished reaping the child and `poll()` reflects it. This
        # flag makes "mark the session dead" an explicit, immediate fact rather than
        # something inferred a moment later from a `poll()` call that might still race.
        self._dead = False

    # --- lifecycle -------------------------------------------------------
    def open(self) -> None:
        """Probe for /system/bin/hid (same clean, expected-fallback signal as
        UhidTouch.open() -- see that method's own comment for why absence is read from
        stdout, never a non-zero exit code), then spawn the persistent PTY-backed `hid`
        process and register ONE virtual device for the life of this object.

        Every failure path here raises UhidUnavailable, never DriverClosed: "the persistent
        mechanism did not work THIS session" (a clean, expected fallback -- `_make_touch`
        treats it exactly like UhidTouch's own open() failure) is a different fact from "a
        session that WAS working broke" (DriverClosed, which is what every failure AFTER
        open() succeeds raises instead -- see `_run_gesture`)."""
        # Same probe UhidTouch.open() uses, verbatim: absence signals via stdout ("no"), not
        # a non-zero remote exit, so a genuine device-loss/command failure still raises
        # AdbError (not caught here, propagates to `_make_touch`'s except clause) rather than
        # being misread as "hid absent".
        if "yes" not in self.adb.shell("[ -e /system/bin/hid ] && echo yes || echo no"):
            raise UhidUnavailable("/system/bin/hid not present")
        w, h = self.adb.screen_size()   # cache geometry; also needed to build the register descriptor

        # Everything from here on mutates self._proc (and, on failure, self._dead via
        # _kill_proc_best_effort) -- the same state _run_gesture and close() touch under
        # self._lock, so this whole sequence is locked too (see __init__'s lock comment).
        # Safe to call _kill_proc_best_effort() from inside this block: self._lock is an
        # RLock precisely so that reentrant call does not deadlock.
        with self._lock:
            # Refuse to spawn a second persistent process out from under a live one. Without
            # this guard, calling open() twice on the same instance would silently overwrite
            # self._proc at the Popen assignment below, orphaning the FIRST `adb shell -tt
            # hid ...` process (and its still-registered virtual device) with nothing left
            # holding a reference to kill it -- a leak, not a crash, so nothing else in the
            # class would ever have surfaced it. Not reachable from today's single call site
            # (`_make_touch` constructs a fresh instance and calls open() on it exactly
            # once) but the class itself provided no defense if a future caller changed that.
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(
                    "PersistentUhidTouch.open() called on a session that is already open; "
                    "close() it first -- calling open() again here would silently orphan "
                    "the existing `adb shell hid` process and its registered virtual device")

            argv = [self.adb.adb_path]
            if self.adb.serial:
                argv += ["-s", self.adb.serial]
            # `-tt` forces a PTY even though this is not an interactive session -- without it,
            # this is the SAME plain-pipe shape that measured ENXIO live on 2026-08-24 (see
            # class docstring). `/proc/self/fd/0` is the remote `hid` process reading its OWN
            # stdin (the PTY's slave side) by path, which a plain reopened pipe could not do.
            argv += ["shell", "-tt", "hid /proc/self/fd/0"]
            try:
                # text=True, not bytes: every payload this transport ever sends or reads is a
                # JSON command line or PTY status text -- never a binary blob (unlike
                # Adb.screencap's PNG, which stays bytes for exactly that reason). Staying in
                # `str` end to end means `json.dumps(...)` writes straight to stdin with no
                # per-call `.encode()`, matching touchwatch.py's own persistent-Popen
                # precedent (also `text=True`, for the identical reason: its stream is text,
                # not binary).
                self._proc = subprocess.Popen(
                    argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1,
                )
            except OSError as exc:
                raise UhidUnavailable(
                    f"could not spawn persistent `adb shell hid`: {exc}") from exc

            # Same jittered enumerate range UhidTouch's per-gesture script uses
            # (self.enumerate_ms + rng.randint(0, 180)) -- this session validated
            # correctness with that range: no reason to assume a persistent connection
            # needs a different one.
            enumerate_ms = self.enumerate_ms + self._rng.randint(0, 180)
            cmds = [_register_command(self.name, self.vid, self.pid, w, h),
                    {"id": 1, "command": "delay", "duration": enumerate_ms}]
            try:
                self._write_commands(cmds)
            except (OSError, ValueError) as exc:
                # The process was JUST spawned; a write failure this early means it never
                # even accepted registration -- this attempt at the persistent mechanism
                # simply did not work THIS session, not that a working session broke. Same
                # UhidUnavailable clean-fallback verdict as every other failure path here.
                self._kill_proc_best_effort()
                raise UhidUnavailable(
                    f"persistent `hid` process rejected registration: {exc}") from exc
            time.sleep(enumerate_ms / 1000.0)
            if self._proc.poll() is not None:
                # Confirmed dead during its OWN enumeration window: this device/PTY approach
                # did not work this session. (See the class docstring's ps/PID caveat:
                # detecting death via `poll()` at all rests on an inference this class never
                # independently proved -- this is the first place that inference gets leaned
                # on.) Still routed through the shared teardown helper rather than a bare
                # `self._proc = None`: the process is already dead, so `kill()` here is a
                # harmless no-op, but `wait()` still needs to run to reap it rather than
                # leaving a zombie behind.
                self._kill_proc_best_effort()
                raise UhidUnavailable(
                    "persistent `hid` process exited during registration/enumeration")

    def close(self) -> None:
        """Best-effort, idempotent teardown -- mirrors UhidTouch.close()'s
        `except Exception: pass` swallow, and for the same reason: process teardown failing
        here is not something a caller can act on, so it must not mask whatever the real
        outcome already was. Safe to call twice, and safe to call on a session that never
        opened (`self._proc is None` either way) -- see `_kill_proc_best_effort`."""
        self._kill_proc_best_effort()

    def _kill_proc_best_effort(self) -> None:
        """Shared by close() and open()'s failure paths. `proc.kill()` alone is EMPIRICALLY
        VERIFIED sufficient to tear the kernel-side virtual device down cleanly -- 4 separate
        live kills, 2026-08-24, `/proc/bus/input/devices` confirmed EMPTY after every one.
        No graceful in-protocol "unregister" command exists or was found; this does not go
        looking for one. Must never raise (both callers depend on that -- close() as a
        public promise, open()'s except-block as a hard requirement of not shadowing the
        real exception being raised there).

        Locked (self._lock is an RLock, see __init__) so the self._proc, self._proc = None
        swap below can never interleave with _run_gesture's guard-check-then-use of the same
        attribute on another thread -- that interleaving was a thread-safety review finding
        (2026-08-24): a concurrent close() could previously null self._proc between
        _run_gesture's `self._proc.poll() is not None` guard and a later use of self._proc a
        few lines on, raising a raw AttributeError instead of the class's documented
        DriverClosed contract. The RLock also makes this method safe to call reentrantly from
        inside open()'s own `with self._lock:` block (its failure paths call this directly
        while already holding the lock)."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=self.close_timeout)
        except Exception:  # noqa: BLE001 — best-effort teardown must not mask the real outcome
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001 — same reasoning, one stream at a time
                pass

    # --- gesture -> persistent stdin write -------------------------------
    def _write_commands(self, cmds: list[dict]) -> None:
        """Write newline-delimited JSON commands to the persistent process's stdin and
        flush. `proc.stdin` is always a live pipe here: every `self._proc` this object ever
        holds was constructed with `stdin=subprocess.PIPE` in open(), above. A write/flush
        failure (BrokenPipeError, or ValueError on a stream already closed out from under
        this object) is caught by CALLERS, not here -- open() and `_run_gesture` each need a
        DIFFERENT verdict for the identical failure (UhidUnavailable during registration vs.
        DriverClosed mid-session), so this method stays a thin, unopinionated write."""
        proc = self._proc
        for c in cmds:
            proc.stdin.write(json.dumps(c) + "\n")
        proc.stdin.flush()

    def _run_gesture(self, samples, *, _timing: dict[str, float] | None = None) -> None:
        """Deliver one gesture's report/delay stream through the already-open persistent
        connection -- NO register command here; that happened once, in open(). See
        UhidTouch._run_gesture's own docstring for `_timing`'s general contract (a per-gesture
        stamps dict `_scroll`/`_swipe` may thread in); this method's own bucket names
        (`uhid_persistent_script_build_s`/`uhid_persistent_write_s`/`uhid_persistent_wait_s`)
        are deliberately DISTINCT from UhidTouch's (`uhid_script_build_s`/`uhid_write_file_s`/
        `uhid_hid_run_s`) even though the two transports are conceptually parallel -- they are
        NOT the same cost: UhidTouch's buckets time a file write plus a blocking `adb shell`
        round trip that itself covers register+enumerate+report+flush+cleanup; this class's
        buckets time a stdin write plus a plain host-side `time.sleep`, with no register/
        enumerate/cleanup cost anywhere in a per-gesture call at all. Reusing the old names
        for a materially different mechanism would make a future timing analysis compare
        two different things as if they were one -- keeping them separate is what keeps
        that comparison honest if this class is ever wired into the ledger. `uhid_plan_swipe_s`
        (planning, in `swipe()` below) is the one bucket name that genuinely IS shared with
        UhidTouch, because it is the identical `plan_swipe` call doing the identical work."""
        if not samples:
            return
        with self._lock:
            # Guard FIRST, before writing anything: a session already found dead must never
            # silently attempt to reopen (that would mean re-registering per gesture again,
            # defeating this class's entire point) or silently resend. This repeats the
            # owner's "best humanized interaction or FAIL LOUDLY; no silent fallback to a
            # degraded transport" rule one level further: no silent fallback to a FRESH
            # SESSION either. `self._dead` catches the write-failure case `poll()` alone
            # might not yet reflect (see its own comment in __init__); `poll() is not None`
            # catches everything else.
            if self._dead or self._proc is None or self._proc.poll() is not None:
                raise DriverClosed(
                    "persistent UHID session is not open (or has already died); refusing "
                    "to silently reopen or resend -- a dead session must be recreated by "
                    "the caller, and a gesture already handed to this transport is never "
                    "replayed against a new one")
            with _time_bucket(_timing, "uhid_persistent_script_build_s"):
                w, h = self.adb.screen_size()
                # Same jittered flush range UhidTouch's per-gesture script uses -- this
                # session validated correctness WITH that flush present; removing or
                # shrinking it is unverified, so it stays exactly as it is upstream.
                flush_ms = self.flush_ms + self._rng.randint(0, 90)
                cmds = _report_stream_commands(samples, w, h, flush_ms)
                wait_s = sum(c["duration"] for c in cmds if c["command"] == "delay") / 1000.0
            try:
                with _time_bucket(_timing, "uhid_persistent_write_s"):
                    self._write_commands(cmds)
            except (OSError, ValueError) as exc:
                # Ambiguous, exactly like UhidTouch._run_gesture's "delivery became
                # uncertain" case (see its own docstring): the write may have delivered
                # some, all, or none of this gesture's reports before failing. Never
                # retried -- replaying could double an irreversible tap, or move the page an
                # unknown amount from an unknown starting point. `self._dead = True` so the
                # VERY NEXT call hits the guard above rather than attempting a second write.
                self._dead = True
                raise DriverClosed(
                    "persistent UHID gesture delivery became uncertain (stdin write "
                    f"failed); refusing to replay the gesture: {exc}") from exc
            # Host-side equivalent of what the CURRENT blocking `adb.shell(f"hid {quoted}")`
            # call already does for the per-gesture-file transport: it blocks for exactly the
            # gesture's own encoded duration. Deliberately NOT a should_stop-aware
            # interruptible wait -- the per-gesture `hid` call has no such mechanism either
            # (UhidTouch's own docstring: "a gesture already handed to `adb shell hid` is
            # never interrupted"), so adding one here would be a behavior CHANGE, not parity.
            #
            # Sliced into _PERSISTENT_POLL_INTERVAL_S steps, polling after each one, rather
            # than a single time.sleep(wait_s) followed by one poll() at the end (see that
            # constant's own comment for why -- a never-substitute review, 2026-08-24: a
            # single end-of-wait check leaves an early-in-the-wait death undetected for the
            # rest of THIS gesture's duration, and possibly into the NEXT gesture's guard
            # check too). This still cannot turn "the local adb client hasn't noticed a dead
            # link yet" into positive proof of delivery -- it only shrinks how long that
            # ambiguity window can silently persist before this class notices and stops.
            with _time_bucket(_timing, "uhid_persistent_wait_s"):
                remaining_s = wait_s
                while remaining_s > 0:
                    step_s = min(_PERSISTENT_POLL_INTERVAL_S, remaining_s)
                    time.sleep(step_s)
                    remaining_s -= step_s
                    if self._proc.poll() is not None:
                        break   # confirmed dead already; the check just below re-observes
                                # the same (idempotent) result and raises -- see that check
            if self._proc.poll() is not None:
                # Died during or immediately after this gesture: "delivery became uncertain"
                # again, for the same never-replay reason (never-substitute: a half-delivered
                # scroll must not be silently retried, which could double a gesture or move
                # the page an unknown amount from an unknown starting point). See the class
                # docstring's ps/PID caveat -- detecting a REMOTE crash this way is inferred,
                # not proven, from the same exit-propagation contract UhidTouch's own
                # `adb.shell` call already leans on.
                self._dead = True
                raise DriverClosed(
                    "persistent UHID gesture delivery became uncertain (the `hid` process "
                    "died during or after delivery); refusing to replay the gesture")

    # --- public surface (matches UhidTouch) -------------------------------
    def tap(self, x: int, y: int) -> None:
        self._run_gesture(plan_tap(x, y, hz=self.hz, rng=self._rng))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 450, *,
             _timing: dict[str, float] | None = None) -> None:
        # Identical to UhidTouch.swipe (see its own comment for the Fitts-law/duration_scale
        # reasoning) -- planning is the one piece of this call that really is shared work,
        # not merely parallel work, hence the shared "uhid_plan_swipe_s" bucket name (see
        # _run_gesture's docstring).
        duration_scale = max(0.20, min(2.0, float(duration_ms) / 450.0))
        with _time_bucket(_timing, "uhid_plan_swipe_s"):
            samples = plan_swipe(x1, y1, x2, y2, hz=self.hz, jitter_px=self.jitter_px,
                                 width_px=self.width_px, duration_scale=duration_scale,
                                 rng=self._rng)
        self._run_gesture(samples, _timing=_timing)

    def scroll_up(self, distance_frac: float = 0.55, x_frac: float = 0.5, *,
                 _timing: dict[str, float] | None = None) -> None:
        # Identical to UhidTouch.scroll_up, including the shared x-column jitter (HINGE-04,
        # adb.scroll_x) -- see that method's own comment for why a genuine transport must not
        # emit a pixel-identical swipe column every scroll.
        with _time_bucket(_timing, "screen_size_s"):
            w, h = self.adb.screen_size()
        x = scroll_x(w, x_frac)
        self.swipe(x, int(h * (0.5 + distance_frac / 2)), x, int(h * (0.5 - distance_frac / 2)),
                  _timing=_timing)
