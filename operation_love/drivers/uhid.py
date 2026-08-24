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
"""
from __future__ import annotations

import json
import random
import secrets
import shlex
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
        cmds = [
            {"id": 1, "command": "register", "name": self.name, "vid": self.vid,
             "pid": self.pid, "bus": "usb", "descriptor": _build_descriptor(w, h)},
            {"id": 1, "command": "delay", "duration": enumerate_ms},
        ]
        for i, s in enumerate(samples):
            cmds.append({"id": 1, "command": "report", "report": _report(s, w, h)})
            if i < len(samples) - 1:
                dt_ms = max(1, int(round((samples[i + 1].t - s.t) * 1000.0)))
                cmds.append({"id": 1, "command": "delay", "duration": dt_ms})
        cmds.append({"id": 1, "command": "delay", "duration": flush_ms})   # let the last event flush
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
