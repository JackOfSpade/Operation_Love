"""Thin host-side ADB wrapper for Android device control — NO on-device helper.

This is the genuine-input control surface for the physical Pixel. It deliberately
avoids uiautomator2 / atx-agent / an accessibility service: those install an
on-device server that Play Integrity can flag (appAccessRiskVerdict), the single
biggest way to burn the account (ops/HINGE-PIXEL-RUNBOOK.md §5). Input goes through
Android's own `input` injector; screenshots through `adb exec-out screencap`.

Swipes are HUMANIZED: `swipe` emits a multi-step `input motionevent DOWN/MOVE/UP`
gesture — a curved quadratic-Bezier path, ease-in-out velocity, per-point coordinate
jitter, and log-normal timing (../human.py) — not a straight `input swipe`.
`input motionevent` keeps one gesture continuous across invocations, so the whole
path is piped to a single `adb shell`. Taps use a single jittered `input tap` (one
fast fork; a multi-step motionevent tap risked crossing Android's ~500ms long-press
threshold on slower devices). NOTE: the genuine-to-the-kernel path (variable
pressure/size at ~180Hz, no FLAG_INJECTED) is the UHID transport (uhid.py), which is what
the driver REQUIRES by default. This `input`-based transport is validated and humanized
(curved paths, jitter, log-normal timing) but cannot vary pressure, so it is reachable only
via an explicit `touch_backend: adb` — never as an automatic fallback.

Why not raw `sendevent` (which could also vary pressure): writing /dev/input/eventX
is blocked by SELinux for the non-root `shell` domain (verified on the Pixel 7a),
and we never root. `input motionevent` routes through system_server's injector,
which is permitted. The accepted no-root limitation: touch pressure can't be varied.

Coordinates are screen pixels (the Pixel 7a touchscreen reports X/Y 1:1 with the
1080x2400 framebuffer).
"""
from __future__ import annotations

import math
import random
import re
import shlex
import subprocess
from collections.abc import Sequence

from ..human import human_delay
from ..typography import (
    describe_char,
    fold_to_ascii,
    undeliverable_chars,
    undeliverable_sequences,
)
from .base import DriverClosed, _time_bucket

_DEVICE_LOST_PHRASES = (
    "device not found",
    "device offline",
    "no devices/emulators found",
    "device unauthorized",
    "unauthorized",
)

# A screenshot cannot deliver input or otherwise change device state.  A single retry is
# therefore safe after a host-side transport timeout, unlike retrying any action command.
# Keep this a fixed bound: an unavailable or wedged device must still stop the run.
_SCREENCAP_ATTEMPTS = 2

_TEXT_SHELL_SPECIALS = frozenset("\\'\"`$&|;<>(){}[]*?!#~")
_ANDROID_PACKAGE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_WM_SIZE_RE = re.compile(r"\b(?:physical|override)\s+size\s*:\s*(\d+)\s*x\s*(\d+)\b", re.IGNORECASE)


def validate_android_package_id(value: object, *, context: str = "Android package") -> str:
    """Return one conservative dotted Android/Java package id or raise before shell use."""
    if not isinstance(value, str) or _ANDROID_PACKAGE_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be a dotted Android identifier containing only letters, digits, "
            f"and underscores (got {value!r})")
    return value


def quote_android_package_id(value: object, *, context: str = "Android package") -> str:
    """Validate and shell-quote a package token for :meth:`Adb.shell` command strings."""
    return shlex.quote(validate_android_package_id(value, context=context))


class AdbError(RuntimeError):
    """An ADB command failed before a usable device response was produced."""

    def __init__(self, argv: Sequence[str], message: str, stderr: str = ""):
        self.argv = list(argv)
        self.stderr = stderr
        detail = f"{message}: {_format_argv(self.argv)}"
        if stderr:
            detail += f"\nstderr: {stderr}"
        super().__init__(detail)


def _bezier(s: tuple[float, float], c: tuple[float, float],
            e: tuple[float, float], t: float) -> tuple[float, float]:
    mt = 1.0 - t
    x = mt * mt * s[0] + 2 * mt * t * c[0] + t * t * e[0]
    y = mt * mt * s[1] + 2 * mt * t * c[1] + t * t * e[1]
    return x, y


def clamp_xy(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """Clamp a coordinate onto the live `w`x`h` screen: the exact arithmetic BOTH real touch
    transports apply to the point they actually deliver -- Adb._clamp (below) and uhid.py's
    _report. Factored out here (adb.py, which uhid.py already imports from) so the two
    transports and hinge.py's forbidden-zone check (_assert_tap_allowed) can never drift
    apart on what "the point that will actually reach the phone" means; before this, each
    of the three call sites re-derived the same `max(0, min(dim-1, round(v)))` arithmetic by
    hand, which is exactly the kind of duplication that lets a checked point silently stop
    matching the delivered one.

    Rounds before clamping (not the reverse): a fraction just past 1.0 must clamp to the
    last valid pixel, not `int()`-truncate to a value that then falls just short of it.
    """
    cx = max(0, min(w - 1, int(round(x))))
    cy = max(0, min(h - 1, int(round(y))))
    return cx, cy


SCROLL_X_JITTER_PX = 25   # scroll_up column jitter, shared by Adb and UhidTouch (HINGE-04)


def scroll_x(width: int, x_frac: float) -> int:
    """The x column for a scroll_up swipe: `x_frac` of `width`, jittered by +/-
    SCROLL_X_JITTER_PX px so repeated scrolls aren't pixel-identical. Shared between Adb and
    UhidTouch (uhid.py imports this) so the two touch transports' humanization can't drift
    apart again — a pixel-identical swipe column is exactly the machine-like signature the
    humanized path exists to avoid."""
    return int(width * x_frac) + random.randint(-SCROLL_X_JITTER_PX, SCROLL_X_JITTER_PX)


def plan_path(x1: float, y1: float, x2: float, y2: float, steps: int,
              curve: float = 0.12, jitter: float = 2.0) -> list[tuple[int, int]]:
    """Waypoints along a quadratic Bezier from (x1,y1) to (x2,y2).

    The control point is offset perpendicular to the straight line by a random
    fraction of its length (the path bows like a real finger); `t` is sampled
    ease-in-out (slow at both ends, faster mid = human velocity); interior points
    get +/- `jitter`px of noise. Endpoints stay exact. Returns steps+1 int points.
    """
    steps = max(2, int(steps))
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length            # unit perpendicular
    offset = random.uniform(-curve, curve) * length
    cx, cy = mx + px * offset, my + py * offset
    pts: list[tuple[int, int]] = []
    for i in range(steps + 1):
        u = i / steps
        t = 0.5 * (1.0 - math.cos(math.pi * u))   # ease-in-out
        bx, by = _bezier((x1, y1), (cx, cy), (x2, y2), t)
        if 0 < i < steps:
            bx += random.uniform(-jitter, jitter)
            by += random.uniform(-jitter, jitter)
        pts.append((int(round(bx)), int(round(by))))
    pts[0] = (int(round(x1)), int(round(y1)))
    pts[-1] = (int(round(x2)), int(round(y2)))
    return pts


class Adb:
    def __init__(
        self,
        serial: str | None = None,
        adb_path: str = "adb",
        default_timeout: float = 10.0,
        *,
        jitter_px: float = 2.0,
    ):
        self.serial = serial
        self.adb_path = adb_path
        self.default_timeout = default_timeout
        self.jitter_px = float(jitter_px)
        self._size: tuple[int, int] | None = None

    # --- humanized input ----------------------------------------------
    def tap(self, x: int, y: int) -> None:
        """A human-ish tap: jittered coordinate, executed in a single fast fork."""
        jx, jy = self._clamp(self._jit(x), self._jit(y))
        self._run_device(["shell", "input", "tap", str(jx), str(jy)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 450, *,
             _timing: dict[str, float] | None = None) -> None:
        """A curved, variable-velocity drag from (x1,y1) to (x2,y2). `duration_ms`
        is the human-anchor for total gesture time (spread log-normally).

        `_timing` (see uhid.py's UhidTouch, this transport's genuine-kernel sibling, and
        hinge.py's `_scroll`/`_swipe`, its only callers): `adb_plan_path_s` isolates the
        pure-CPU Bezier-path synthesis (plan_path) from `adb_script_build_s`, which builds the
        `input motionevent`/`sleep` script text those points are turned into, and both are kept
        separate from `adb_run_script_s` -- the ONE blocking `adb shell` round trip that
        actually delivers the whole gesture. UNLIKE UHID's transport, this one issues no
        separate write-then-run-then-remove sequence: the entire scripted gesture, embedded
        `sleep`s included, is piped to a single `adb shell` invocation and executed by the
        device's own shell, so `adb_run_script_s` is this transport's one and only per-gesture
        device round trip, and its internal step-by-step timing is exactly as undecomposable
        here as UHID's `uhid_hid_run_s` is, for the same reason: decomposing it would need
        either a second device call this ledger may not add, or a behavior change."""
        duration = max(0.12, human_delay(max(1, int(duration_ms)) / 1000.0))
        # Each motionevent step is a separate on-device `cmd input` fork (~100ms),
        # so step count, not the sleeps, dominates gesture time. ~1 step / 60px
        # keeps the path visibly curved while staying ~1.5-2s (deliberate, human).
        with _time_bucket(_timing, "adb_plan_path_s"):
            dist = math.hypot(x2 - x1, y2 - y1)
            steps = max(8, min(22, int(dist / 60)))
            pts = [self._clamp(px, py) for px, py in
                   plan_path(x1, y1, x2, y2, steps, jitter=self.jitter_px)]
        with _time_bucket(_timing, "adb_script_build_s"):
            per = duration / max(1, len(pts) - 1)
            lines = [f"input motionevent DOWN {pts[0][0]} {pts[0][1]}"]
            for i in range(1, len(pts)):
                lines.append(f"sleep {max(0.005, human_delay(per, sigma=0.3)):.3f}")
                kind = "UP" if i == len(pts) - 1 else "MOVE"
                lines.append(f"input motionevent {kind} {pts[i][0]} {pts[i][1]}")
        with _time_bucket(_timing, "adb_run_script_s"):
            self._run_script(lines)

    def scroll_up(self, distance_frac: float = 0.55, x_frac: float = 0.5, *,
                 _timing: dict[str, float] | None = None) -> None:
        """Scroll content up (reveal what's below) — a humanized swipe low->high.

        `_timing`'s "screen_size_s" bucket is the SAME key hinge.py's `_scroll` and this
        transport's UHID sibling already write (accumulated across every call site in one
        gesture -- see uhid.py's `scroll_up` for the full "is this actually a device round
        trip" answer, which applies identically here: `Adb.screen_size()` caches after its
        first-ever call for the session, so this is a dict lookup by the time any gesture runs).
        """
        with _time_bucket(_timing, "screen_size_s"):
            w, h = self.screen_size()
        x = scroll_x(w, x_frac)
        y1 = int(h * (0.5 + distance_frac / 2))
        y2 = int(h * (0.5 - distance_frac / 2))
        self.swipe(x, y1, x, y2, _timing=_timing)

    def text(self, s: str) -> None:
        """Type a string via ``adb shell input text``.

        Checked BEFORE anything is folded or sent: if `s` still contains a character the
        device keyboard cannot render even after ASCII folding (typography.fold_to_ascii --
        curly quotes, dashes, accents, ligatures, etc.), this raises AdbError naming it and
        NO adb command is issued at all. This is the owner's "best humanized interaction or
        fail loudly" rule applied to text input: the old behaviour silently dropped the
        undeliverable character and typed a truncated message, so the text recorded in
        BigQuery / shown in the hub could read differently from what the device actually
        typed to a real person. Failing here instead makes that impossible -- what got
        recorded as sent is always what was actually sent, or nothing was sent at all.

        A literal ``%`` is otherwise left completely alone by fold_to_ascii (the owner
        rejected an earlier version that rewrote it to the word " percent" -- unnatural: "50%"
        must type as "50%") -- it round-trips fine through ``adb shell input text``'s own
        ``%s`` space escape in every case measured EXCEPT one: a literal ``%`` immediately
        followed by a lowercase ``s`` collides with the escape and is silently eaten
        (measured via a faithful port of Android's sendText() unescaper against this file's
        own _escape_input_text -- see typography.fold_to_ascii's docstring point 4 for the
        full round-trip table). That one narrow case is checked for explicitly, right below,
        and rejected the same way an undeliverable character is: loudly, with nothing typed.
        """
        bad = undeliverable_chars(s)
        if bad:
            names = ", ".join(describe_char(ch) for ch in bad)
            raise AdbError(
                ["shell", "input", "text", s],
                f"cannot type this text: it contains character(s) the device keyboard "
                f"cannot render even after ASCII folding: {names}. The text was NOT typed.",
            )
        bad_seqs = undeliverable_sequences(s)
        if bad_seqs:
            seqs = ", ".join(repr(seq) for seq in bad_seqs)
            raise AdbError(
                ["shell", "input", "text", s],
                f"cannot type this text: it contains sequence(s) that collide with "
                f"adb's own %s space escape and cannot be delivered as written, even though "
                f"every character in them is individually typeable: {seqs}. "
                f"The text was NOT typed.",
            )
        cleaned = _clean_text_for_input(s)
        if cleaned:
            self._run_device(["shell", "input", "text", _escape_input_text(cleaned)])

    def keyevent(self, keycode: int) -> None:
        """Deliver one Android key event through the device-owned input transport.

        This deliberately exposes no arbitrary shell fragment to callers.  The Hinge training
        checkpoint uses ``KEYCODE_BACK`` (4) to dismiss the IME after its opener is typed; the
        driver, not a call site, remains responsible for foreground ownership and audit logging
        around that input.
        """
        if type(keycode) is not int or keycode < 0 or keycode > 0xffff:
            raise ValueError("Android keycode must be an integer in 0..65535")
        self._run_device(["shell", "input", "keyevent", str(keycode)])

    # --- screen --------------------------------------------------------
    def screencap(self) -> bytes:
        """Return one PNG frame, retrying one host-side timeout of this read-only command.

        ``exec-out screencap -p`` is the sole command eligible for this recovery path: it
        observes the screen and cannot replay a tap, swipe, typed text, or shell write.  A
        nonzero ADB response is not assumed transient, and neither are device-loss errors;
        both fail closed without a second command.  On an exhausted retry, retain the final
        ADB error and add the initial timeout as traceback context for diagnosis.
        """
        args = ["exec-out", "screencap", "-p"]
        first_error: AdbError | None = None
        for attempt in range(_SCREENCAP_ATTEMPTS):
            try:
                return self._run_device(args)
            except DriverClosed:
                raise
            except AdbError as error:
                # _run chains the original TimeoutExpired, so do not classify arbitrary ADB
                # failures from their text.  This is intentionally narrower than a generic
                # "retry on AdbError" policy.
                if (attempt + 1 < _SCREENCAP_ATTEMPTS
                        and isinstance(error.__cause__, subprocess.TimeoutExpired)):
                    if first_error is None:
                        first_error = error
                    continue
                if first_error is not None:
                    terminal_error = AdbError(
                        error.argv,
                        "ADB screencap recovery exhausted after "
                        f"{_SCREENCAP_ATTEMPTS} read-only attempts; final failure: "
                        f"{str(error).split(':', 1)[0]}",
                        error.stderr,
                    )
                    terminal_error.add_note(
                        "Read-only screencap retry after initial timeout: "
                        f"{first_error}")
                    raise terminal_error from error
                raise

        raise AssertionError("screencap retry loop exhausted without returning or raising")

    def screen_size(self) -> tuple[int, int]:
        if self._size is None:
            out = _decode(self._run_device(["shell", "wm", "size"]))
            w = h = 0
            for line in out.splitlines():
                match = _WM_SIZE_RE.search(line)
                if match is not None:
                    w, h = (int(value) for value in match.groups())
                    # Android lists the optional override after the physical size, so the
                    # last matching line is the actual coordinate space for input.
            if not (w and h):
                raise AdbError(["wm", "size"], f"could not parse screen size from {out!r}")
            self._size = (w, h)
        return self._size

    def devices(self) -> list[str]:
        out = self._run(["devices"])
        return parse_devices_output(_decode(out))

    def shell(self, cmd: str) -> str:
        """Run a shell command on the device and return its decoded output."""
        return _decode(self._run_device(["shell", cmd]))

    def foreground_package(self) -> str | None:
        """Best-effort package currently holding Android's focused window.

        This is deliberately only a package-level state probe, not an accessibility-tree read.
        It lets a driver distinguish its own UI from Android System UI (for example the
        notification shade) before a capture mistakes system chrome for app content.

        ``dumpsys window windows`` is authoritative when it reports a focus field: its
        ``mCurrentFocus`` names an overlay such as the notification shade, whereas the
        underlying activity can still be Hinge.  Android 17 on the live Pixel no longer puts
        either focus field in that particular dump, however; its ``dumpsys activity
        activities`` output does, under ActivityManager's OWN field names
        (``mResumedActivity:``/``mFocusedActivity:`` -- see
        :func:`parse_foreground_activity_package` for the shape and its LIVE-VERIFY caveat;
        an earlier version of this fallback grepped the WindowManager names against this dump
        by mistake and silently returned None every time -- Finding 1, 2026-09-02). Use the
        activity dump only when the window dump contains neither labelled focus field.  A
        present-but-malformed/conflicting window focus is deliberately *not* retried through
        activity -- it remains indeterminate rather than risking a stale underlying activity
        being treated as the foreground surface.

        DO NOT CACHE WHICH DUMP ANSWERED, and do not reorder these two probes.  This runs
        once per captured frame and once per gesture, so remembering that the activity dump
        answered last time and asking it FIRST looks like an obvious saving of one always-empty
        window round trip on an Android-17 session.  It was written that way on 2026-09-02 and
        reverted the same day, because the two dumps do not answer the same question and are
        not interchangeable sources for one fact: the window dump reports WHICH WINDOW HAS
        FOCUS (the notification shade, when it is open), while the activity dump reports which
        ACTIVITY IS RESUMED -- and Hinge's activity stays resumed underneath the open shade.
        Ask activity first and it answers ``co.hinge.app`` truthfully; return that and the
        screen-ownership gate that guards touch input concludes Hinge owns a screen the shade
        is actually covering.  That is precisely the case this probe exists to catch, so the
        window dump is asked FIRST on every call, unconditionally, and one extra round trip on
        a window-incapable device is the correct price.
        """
        window_answered, window_package = self._probe_window_foreground()
        if window_answered:
            return window_package
        activity_answered, activity_package = self._probe_activity_foreground()
        return activity_package if activity_answered else None

    def _probe_window_foreground(self) -> tuple[bool, str | None]:
        """One ``dumpsys window windows`` round trip: (either focus field present, package)."""
        window_output = _decode(self._run_device(["shell", "dumpsys window windows"]))
        current_present, _ = _package_from_focus_lines(window_output, "mCurrentFocus")
        focused_present, _ = _package_from_focus_lines(window_output, "mFocusedApp")
        if current_present or focused_present:
            return True, parse_foreground_package(window_output)
        return False, None

    def _probe_activity_foreground(self) -> tuple[bool, str | None]:
        """One ``dumpsys activity activities`` round trip: (either focus field present,
        package). See :func:`parse_foreground_activity_package` for the field names this
        checks and the LIVE-VERIFY caveat on this dump shape."""
        activity_output = _decode(self._run_device(["shell", "dumpsys activity activities"]))
        resumed_present, _ = _package_from_focus_lines(
            activity_output, "mResumedActivity", separator=":")
        focused_present, _ = _package_from_focus_lines(
            activity_output, "mFocusedActivity", separator=":")
        if resumed_present or focused_present:
            return True, parse_foreground_activity_package(activity_output)
        return False, None

    def write_file(self, path: str, data: bytes) -> None:
        """Write bytes to a device file via `cat` redirect. Works for regular files in
        shell-writable dirs (e.g. /data/local/tmp) — unlike `mkfifo`, which SELinux denies
        to the shell domain. Raises DriverClosed if the device drops."""
        self._run(["shell", f"cat > {shlex.quote(path)}"], target_device=True, input_bytes=data)

    # --- helpers -------------------------------------------------------
    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        w, h = self._size if self._size else (1080, 2400)
        return clamp_xy(x, y, w, h)

    def _jit(self, v: float) -> int:
        return int(round(v + random.uniform(-self.jitter_px, self.jitter_px)))

    def _run_device(self, args: Sequence[str]) -> bytes:
        return self._run(args, target_device=True)

    def _run_script(self, lines: Sequence[str]) -> None:
        """Pipe a multi-line shell script to one `adb shell` (one round-trip)."""
        script = ("\n".join(lines) + "\n").encode()
        self._run(["shell"], target_device=True, input_bytes=script)

    def _run(self, args: Sequence[str], target_device: bool = False,
             input_bytes: bytes | None = None) -> bytes:
        argv = [self.adb_path]
        if target_device and self.serial:
            argv.extend(["-s", self.serial])
        argv.extend(args)
        try:
            result = subprocess.run(
                argv,
                input=input_bytes,
                capture_output=True,
                timeout=self.default_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = _decode(exc.stderr)
            # ADB can report an already-disconnected device while its client is still being
            # waited on.  Preserve the same terminal semantic used for a completed nonzero
            # command, rather than treating that report as a recoverable timeout.  Do not
            # inspect timed-out screencap stdout: it can be an arbitrarily large partial PNG,
            # not a textual ADB diagnostic, so decoding it is both wasteful and could
            # accidentally match an ordinary pixel sequence.
            reported = stderr
            if tuple(args) != ("exec-out", "screencap", "-p"):
                stdout_text = _decode(exc.stdout)
                reported = "\n".join(part for part in (stderr, stdout_text) if part)
            if _is_device_lost_message(reported):
                raise DriverClosed("ADB device was disconnected") from None
            raise AdbError(
                argv,
                f"ADB command timed out after {self.default_timeout:g}s",
                stderr,
            ) from exc
        except FileNotFoundError as exc:
            raise AdbError(argv, "ADB binary not found") from exc
        except OSError as exc:
            # ``subprocess.run`` can also fail before a child starts when an explicit
            # adb_path is not executable (PermissionError), is a directory, or the host
            # otherwise rejects execve. Keep those host-side launch failures on the same
            # public AdbError surface as a missing binary instead of leaking an unrelated
            # OSError from beneath the driver API.
            raise AdbError(argv, f"could not start ADB binary: {exc}") from exc

        if result.returncode != 0:
            # Only decode on the error path — never decode a successful payload
            # (e.g. a multi-MB screencap PNG) just to inspect it for failure text.
            stderr = _decode(result.stderr)
            stdout_text = _decode(result.stdout)
            reported = "\n".join(p for p in (stderr, stdout_text) if p)
            if _is_device_lost_message(reported):
                raise DriverClosed("ADB device was disconnected") from None
            raise AdbError(
                argv,
                f"ADB command failed with exit code {result.returncode}",
                stderr,
            )
        return result.stdout


def parse_devices_output(stdout: str) -> list[str]:
    """Canonical parser for `adb devices` / `adb devices -l` stdout (X8): the serials whose
    state is exactly "device" (ready) — states like "unauthorized" / "offline" are excluded.

    Skips the "List of devices attached" header, blank lines, and the daemon-startup chatter
    adb prints to stdout on a cold start (lines like "* daemon not running; starting now at
    tcp:5037" / "* daemon started successfully"). Checks the SECOND whitespace-split field, not
    the last: `-l` appends trailing `key:value` columns (usb:... product:... model:...
    device:... transport_id:...), so a `parts[-1] == "device"` check (as used by two other
    ad-hoc parsers of this same command elsewhere in the codebase) breaks on that output.
    """
    devices: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*") or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


_FOREGROUND_COMPONENT_RE = re.compile(
    r"(?<![\w.])([A-Za-z][\w.]*)/(?:[A-Za-z_$][\w.$]*|\.[A-Za-z_$][\w.$]*)"
)


def _package_from_focus_lines(
        stdout: str, label: str, *, separator: str = "=") -> tuple[bool, str | None]:
    """Return ``(label_present, unambiguous_package)`` for one dumpsys focus field.

    ``separator`` distinguishes WindowManager's ``label=value`` shape (``dumpsys window
    windows``'s ``mCurrentFocus``/``mFocusedApp``, the default here) from ActivityManager's
    ``label: value`` shape (``dumpsys activity activities``'s ``mResumedActivity``/
    ``mFocusedActivity`` -- see :func:`parse_foreground_activity_package`). Both dumps wrap
    the package the same way (a ``pkg/.Class`` or ``pkg/pkg.Class`` component later on the
    same line), so only the marker differs between the two shapes; the component regex and
    every ambiguity rule below apply identically to both (Finding 1, 2026-09-02).

    Android can report more than one display.  Repeated lines that all name the same package
    are harmless, but conflicting or partially unreadable focus lines are not evidence that
    any one package owns the screen and therefore fail closed to ``None``.
    """
    marker = re.compile(rf"\b{re.escape(label)}\s*{re.escape(separator)}")
    packages: list[str] = []
    present = False
    for line in stdout.splitlines():
        marker_match = marker.search(line)
        if marker_match is None:
            continue
        present = True
        component = _FOREGROUND_COMPONENT_RE.search(line, marker_match.end())
        if component is None:
            return True, None
        packages.append(component.group(1))
    if not present:
        return False, None
    unique = set(packages)
    return True, packages[0] if len(unique) == 1 else None


def parse_foreground_package(stdout: str) -> str | None:
    """Extract the focused window's package from Android's ``dumpsys window`` output.

    Both the current-focus and focused-app forms include a ``package/class`` component. The
    focused *window* is authoritative: while System UI covers an app, ``mCurrentFocus`` can name
    System UI while ``mFocusedApp`` still names the underlying activity. Prefer it regardless of
    textual order, and use the activity only as a compatibility fallback when the
    ``mCurrentFocus`` field is absent entirely. A present-but-null, malformed, or conflicting
    current-focus field must not fall through to a stale underlying activity. Unlabelled
    package-looking strings can never claim the foreground.
    """
    current_present, package = _package_from_focus_lines(stdout, "mCurrentFocus")
    if current_present:
        return package
    _focused_present, package = _package_from_focus_lines(stdout, "mFocusedApp")
    return package


def parse_foreground_activity_package(stdout: str) -> str | None:
    """Extract the resumed/focused activity's package from ``dumpsys activity activities``.

    LIVE-VERIFY (Finding 1, 2026-09-02): this is ActivityManager/ActivityTaskManager's OWN
    dump, and it names focus with ITS OWN field names -- ``mResumedActivity:`` /
    ``mFocusedActivity:``, colon-separated, each wrapping an
    ``ActivityRecord{hash u0 pkg/.Class taskId}`` payload. The fallback this function replaces
    grepped this dump for ``mCurrentFocus``/``mFocusedApp`` instead -- those are WindowManager's
    field names, from ``dumpsys window windows`` (see :func:`parse_foreground_package`), and
    they never appear here -- so on the one device that actually needed this fallback (Android
    17 on the live Pixel, where the window dump has stopped printing either focus field) it
    silently returned None on every single call, and the caller's "no focus info" and "focus
    info says nobody" cases became indistinguishable.

    The field names and colon separator above are the documented AOSP shape (the dump text
    ActivityTaskSupervisor/ActivityStackSupervisor's own ``dump()`` prints), but nobody has
    captured this exact string off the live Android-17 Pixel yet -- CONFIRM it against a real
    capture before trusting this path in production. Until then this fails closed to None on
    anything that does not match, exactly like the window-dump parser above: a wrong guess is
    worse than no guess (see ``humanized-input-only`` / fail-loud rule).

    Prefer ``mResumedActivity`` (the activity actually running/visible) and fall back to
    ``mFocusedActivity`` only when resumed is entirely absent, mirroring
    :func:`parse_foreground_package`'s current-focus-first precedence for the window dump. A
    present-but-null/malformed/conflicting resumed field must not fall through to a stale
    focused activity.
    """
    resumed_present, package = _package_from_focus_lines(
        stdout, "mResumedActivity", separator=":")
    if resumed_present:
        return package
    _focused_present, package = _package_from_focus_lines(
        stdout, "mFocusedActivity", separator=":")
    return package


def _escape_input_text(s: str) -> str:
    """Best-effort escaping for ``adb shell input text``.

    Whitespace -> ``%s`` (input text's space escape) and shell metacharacters are
    backslash-escaped so the on-device shell doesn't interpret them. Caveat: a literal ``%``
    immediately followed by a lowercase ``s`` in ``s`` is NOT round-trip-safe -- it collides
    with the space encoding produced right here (any other literal ``%``, including one
    followed by uppercase ``S``, round-trips fine; see typography.fold_to_ascii's docstring
    point 4 for the measured table). :meth:`Adb.text` does NOT work around this by rewriting
    the text -- the owner rejected that as unnatural -- it instead checks for the collision
    via :func:`typography.undeliverable_sequences` BEFORE this function ever runs, and raises
    :class:`AdbError` naming it rather than typing something silently different from what was
    asked for.
    """
    escaped: list[str] = []
    for ch in str(s):
        if ch.isspace():
            escaped.append("%s")
        elif ch in _TEXT_SHELL_SPECIALS:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def _is_device_lost_message(message: str) -> bool:
    lowered = message.lower()
    return (
        any(phrase in lowered for phrase in _DEVICE_LOST_PHRASES)
        or ("device" in lowered and "not found" in lowered)
    )


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace").strip()


def _format_argv(argv: Sequence[str]) -> str:
    return shlex.join(argv)


# Owner rule (b): no dash of any kind may survive (reads as AI). Folding (quotes, dashes,
# ellipsis, exotic spaces, ligatures, accents) is delegated entirely to
# operation_love.typography.fold_to_ascii -- the SAME function opener.py's _sanitize() uses
# -- so this device-input safety net and the LLM-output sanitizer can't silently disagree
# again on which codepoints count as a dash, an accent, or anything else.


def _clean_text_for_input(s: str) -> str:
    """Fold `s` to its ASCII-typeable equivalent via typography.fold_to_ascii (curly quotes,
    ellipsis, exotic spaces, every dash variant, and NFKD accent-stripping -- e.g. an
    accented "e" becomes a plain ASCII "e", not a mangled or truncated string).

    Unlike the version of this function that shipped before typography.py's fold_to_ascii
    existed, it no longer silently DROPS whatever is left over after folding: a character
    fold_to_ascii cannot reduce to printable ASCII is now Adb.text()'s problem, not this
    function's -- Adb.text() calls typography.undeliverable_chars() on the raw input BEFORE
    this function ever runs (undeliverable_chars folds internally too, so the check is
    equivalent to checking this function's own output) and raises AdbError naming the
    offending character(s) rather than typing a silently truncated message. See adb.py's
    module docstring and the owner's "best humanized interaction or fail loudly" rule for
    why a silent drop is not acceptable here: the text recorded in BigQuery / shown in the
    hub must always be exactly what the device actually typed."""
    return fold_to_ascii(s)
