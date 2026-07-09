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
pressure/size at ~180Hz, no FLAG_INJECTED) is the planned UHID transport (uhid.py);
this `input`-based transport is the validated fallback.

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
import shlex
import subprocess
from collections.abc import Sequence

from ..human import human_delay
from .base import DriverClosed

_DEVICE_LOST_PHRASES = (
    "device not found",
    "device offline",
    "no devices/emulators found",
    "device unauthorized",
    "unauthorized",
)

_TEXT_SHELL_SPECIALS = frozenset("\\'\"`$&|;<>(){}[]*?!#~")


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

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 450) -> None:
        """A curved, variable-velocity drag from (x1,y1) to (x2,y2). `duration_ms`
        is the human-anchor for total gesture time (spread log-normally)."""
        duration = max(0.12, human_delay(max(1, int(duration_ms)) / 1000.0))
        # Each motionevent step is a separate on-device `cmd input` fork (~100ms),
        # so step count, not the sleeps, dominates gesture time. ~1 step / 60px
        # keeps the path visibly curved while staying ~1.5-2s (deliberate, human).
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(8, min(22, int(dist / 60)))
        pts = [self._clamp(px, py) for px, py in
               plan_path(x1, y1, x2, y2, steps, jitter=self.jitter_px)]
        per = duration / max(1, len(pts) - 1)
        lines = [f"input motionevent DOWN {pts[0][0]} {pts[0][1]}"]
        for i in range(1, len(pts)):
            lines.append(f"sleep {max(0.005, human_delay(per, sigma=0.3)):.3f}")
            kind = "UP" if i == len(pts) - 1 else "MOVE"
            lines.append(f"input motionevent {kind} {pts[i][0]} {pts[i][1]}")
        self._run_script(lines)

    def scroll_up(self, distance_frac: float = 0.55, x_frac: float = 0.5) -> None:
        """Scroll content up (reveal what's below) — a humanized swipe low->high."""
        w, h = self.screen_size()
        x = int(w * x_frac) + random.randint(-25, 25)
        y1 = int(h * (0.5 + distance_frac / 2))
        y2 = int(h * (0.5 - distance_frac / 2))
        self.swipe(x, y1, x, y2)

    def key(self, keycode: int | str) -> None:
        self._run_device(["shell", "input", "keyevent", str(keycode)])

    def text(self, s: str) -> None:
        """Type a string via ``adb shell input text``.

        NOT round-trip-safe for every string: ``adb shell input text`` reserves
        ``%s`` as its space escape. This method cleans the input to keep printable ASCII
        and avoids the space encoding collision.
        """
        cleaned = _clean_text_for_input(s)
        if cleaned:
            self._run_device(["shell", "input", "text", _escape_input_text(cleaned)])

    # --- screen --------------------------------------------------------
    def screencap(self) -> bytes:
        return self._run_device(["exec-out", "screencap", "-p"])

    def screen_size(self) -> tuple[int, int]:
        if self._size is None:
            out = _decode(self._run_device(["shell", "wm", "size"]))
            w = h = 0
            for line in out.splitlines():
                low = line.lower()
                if "size:" in low and "x" in low:
                    try:
                        ws, hs = line.split("size:")[1].strip().lower().split("x")
                        w, h = int(ws), int(hs)        # Override size, if present, wins (parsed last)
                    except (ValueError, IndexError):
                        continue
            if not (w and h):
                raise AdbError(["wm", "size"], f"could not parse screen size from {out!r}")
            self._size = (w, h)
        return self._size

    def devices(self) -> list[str]:
        out = self._run(["devices"])
        devices: list[str] = []
        for raw_line in _decode(out).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("List of devices"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    def shell(self, cmd: str) -> str:
        """Run a shell command on the device and return its decoded output."""
        return _decode(self._run_device(["shell", cmd]))

    def write_file(self, path: str, data: bytes) -> None:
        """Write bytes to a device file via `cat` redirect. Works for regular files in
        shell-writable dirs (e.g. /data/local/tmp) — unlike `mkfifo`, which SELinux denies
        to the shell domain. Raises DriverClosed if the device drops."""
        self._run(["shell", f"cat > {shlex.quote(path)}"], target_device=True, input_bytes=data)

    # --- helpers -------------------------------------------------------
    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        w, h = self._size if self._size else (1080, 2400)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return int(x), int(y)

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
            )
        except subprocess.TimeoutExpired as exc:
            stderr = _decode(exc.stderr)
            raise AdbError(
                argv,
                f"ADB command timed out after {self.default_timeout:g}s",
                stderr,
            ) from exc
        except FileNotFoundError as exc:
            raise AdbError(argv, "ADB binary not found") from exc

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


def _escape_input_text(s: str) -> str:
    """Best-effort escaping for ``adb shell input text``.

    Whitespace -> ``%s`` (input text's space escape) and shell metacharacters are
    backslash-escaped so the on-device shell doesn't interpret them. Caveat: a
    literal ``%`` / ``%s`` in ``s`` is NOT round-trip-safe (it collides with the
    space encoding); see :meth:`Adb.text`. Use the clipboard path when fidelity
    matters.
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


def _clean_text_for_input(s: str) -> str:
    cleaned = []
    for ch in s:
        o = ord(ch)
        if 32 <= o <= 126:
            if ch == "%":
                cleaned.append(" percent")
            else:
                cleaned.append(ch)
        elif ch == "\n":
            cleaned.append(" ")
    return "".join(cleaned)
