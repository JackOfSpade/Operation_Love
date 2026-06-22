"""Thin host-side ADB wrapper for Android device control."""
from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence

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


class Adb:
    def __init__(
        self,
        serial: str | None = None,
        adb_path: str = "adb",
        default_timeout: float = 10.0,
    ):
        self.serial = serial
        self.adb_path = adb_path
        self.default_timeout = default_timeout

    def tap(self, x: int, y: int) -> None:
        self._run_device(["shell", "input", "tap", _int_arg(x), _int_arg(y)])

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 200,
    ) -> None:
        self._run_device([
            "shell",
            "input",
            "swipe",
            _int_arg(x1),
            _int_arg(y1),
            _int_arg(x2),
            _int_arg(y2),
            _int_arg(duration_ms),
        ])

    def key(self, keycode: int | str) -> None:
        self._run_device(["shell", "input", "keyevent", str(keycode)])

    def text(self, s: str) -> None:
        """Type a string via ``adb shell input text``.

        NOT round-trip-safe for every string: ``adb shell input text`` reserves
        ``%s`` as its space escape, so a literal ``%`` (or the literal sequence
        ``%s``) in ``s`` collides with the space encoding and cannot be faithfully
        transmitted, and non-ASCII / emoji are unreliable. For high-fidelity text
        (openers, bios) prefer the scrcpy clipboard path rather than this
        primitive; this is fine for ASCII control text.
        """
        self._run_device(["shell", "input", "text", _escape_input_text(s)])

    def screencap(self) -> bytes:
        return self._run_device(["exec-out", "screencap", "-p"])

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

    def _run_device(self, args: Sequence[str]) -> bytes:
        return self._run(args, target_device=True)

    def _run(self, args: Sequence[str], target_device: bool = False) -> bytes:
        argv = [self.adb_path]
        if target_device and self.serial:
            argv.extend(["-s", self.serial])
        argv.extend(args)
        try:
            result = subprocess.run(
                argv,
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


def _int_arg(value: int) -> str:
    return str(int(value))


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
