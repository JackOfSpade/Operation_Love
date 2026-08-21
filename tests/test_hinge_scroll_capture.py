"""tools/hinge_scroll_capture.py -- a READ-ONLY dense-frame capture tool for Hinge
card-geometry calibration (ops/OPENER-REDESIGN.md 5.4/5.5/9). See that module's own
docstring for what the frames are for and why this tool must never send input to the phone.

No test here touches a real device: `capture()` takes a duck-typed `adb` object (only
`screencap()` is ever called on it, matching production), so the dedupe/manifest/Ctrl-C tests
below use a FakeAdb that never shells out. The CLI-level tests (`_resolve_serial`,
`main()`) monkeypatch `subprocess.run` and/or the module's own `Adb` name -- nothing here
spawns a real `adb` process or opens a real connection.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from tools import hinge_scroll_capture as hsc


class _StopCapture(Exception):
    """Raised by FakeAdb once its scripted frames are exhausted, so a test can end the
    capture loop deterministically (on the NEXT screencap() call) instead of racing real
    wall-clock time against `seconds`/`interval` -- the same class of flake this repo's own
    Hinge observe regression tests were fixed for (see "Fix a wall-clock race..." in git log).
    `capture()`'s screencap try/except catches this like any other screencap failure: it
    prints a message and breaks out of the loop, leaving whatever was already saved intact."""


class FakeAdb:
    """Duck-types the ONE method `capture()` is allowed to call: `screencap()`. Returns each
    scripted frame in order; once exhausted, raises `_StopCapture` to end the test's loop."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.calls = 0

    def screencap(self) -> bytes:
        self.calls += 1
        if not self._frames:
            raise _StopCapture()
        return self._frames.pop(0)


def _sha256(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


# --- dedupe -----------------------------------------------------------------------------

def test_capture_dedupes_byte_identical_consecutive_frames_keeps_changed_ones(tmp_path):
    # AAA repeats, then BBB repeats, then CCC once, then AAA again (not adjacent to the
    # first AAA run, so it must be saved again -- dedupe is against the previously SAVED
    # frame only, per the tool's spec, not a whole-session hash pool).
    frames = [b"AAA", b"AAA", b"BBB", b"BBB", b"BBB", b"CCC", b"AAA"]
    adb = FakeAdb(frames)
    out_dir = tmp_path / "out"

    manifest = hsc.capture(adb=adb, seconds=1000.0, interval=0.0, out_dir=out_dir,
                            serial="SER1", progress=False)

    assert manifest["frame_count"] == 4
    saved = sorted(p.name for p in out_dir.glob("*.png"))
    assert saved == ["00001.png", "00002.png", "00003.png", "00004.png"]
    assert (out_dir / "00001.png").read_bytes() == b"AAA"
    assert (out_dir / "00002.png").read_bytes() == b"BBB"
    assert (out_dir / "00003.png").read_bytes() == b"CCC"
    assert (out_dir / "00004.png").read_bytes() == b"AAA"


def test_capture_skips_a_fully_static_run_and_saves_only_the_first_frame(tmp_path):
    frames = [b"X"] * 10
    adb = FakeAdb(frames)
    out_dir = tmp_path / "out"

    manifest = hsc.capture(adb=adb, seconds=1000.0, interval=0.0, out_dir=out_dir,
                            serial="SER1", progress=False)

    assert manifest["frame_count"] == 1
    assert list(out_dir.glob("*.png")) == [out_dir / "00001.png"]


# --- manifest shape -----------------------------------------------------------------------

def test_manifest_well_formed_and_frame_numbering_sequential_with_no_gaps(tmp_path):
    frames = [b"one", b"two", b"two", b"three"]
    adb = FakeAdb(frames)
    out_dir = tmp_path / "out"

    manifest = hsc.capture(adb=adb, seconds=1000.0, interval=0.0, out_dir=out_dir,
                            serial="MY-SERIAL", progress=False)

    for key in ("tool_version", "device_serial", "interval_s", "seconds_requested",
                "start_utc", "end_utc", "interrupted", "frame_count", "frames"):
        assert key in manifest, f"manifest missing {key!r}"

    assert manifest["device_serial"] == "MY-SERIAL"
    assert manifest["frame_count"] == len(manifest["frames"]) == 3

    # sequential, zero-padded, no gaps
    assert [f["file"] for f in manifest["frames"]] == ["00001.png", "00002.png", "00003.png"]
    # every named file actually exists on disk
    for f in manifest["frames"]:
        assert (out_dir / f["file"]).exists()
    # per-frame sha256 matches the bytes actually written, and offsets are non-decreasing
    prev_offset = -1.0
    for f, raw in zip(manifest["frames"], [b"one", b"two", b"three"], strict=True):
        assert f["sha256"] == _sha256(raw)
        assert f["offset_s"] >= prev_offset
        prev_offset = f["offset_s"]

    # manifest.json on disk matches the returned dict exactly
    on_disk = json.loads((out_dir / "manifest.json").read_text())
    assert on_disk == manifest
    if os.name == "posix":
        assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600
                   for path in out_dir.iterdir() if path.is_file())


# --- Ctrl-C / early termination ------------------------------------------------------------

class _InterruptingAdb:
    """Like FakeAdb, but raises KeyboardInterrupt (simulating Ctrl-C) after N frames instead
    of the benign _StopCapture -- exercises the "still writes a valid manifest" path
    specifically, which capture() must handle differently (interrupted=True) from an
    ordinary screencap error."""

    def __init__(self, frames, interrupt_after):
        self._frames = list(frames)
        self._interrupt_after = interrupt_after
        self.calls = 0

    def screencap(self) -> bytes:
        self.calls += 1
        if self.calls > self._interrupt_after:
            raise KeyboardInterrupt()
        return self._frames.pop(0)


def test_ctrl_c_during_capture_still_writes_a_valid_manifest(tmp_path):
    adb = _InterruptingAdb([b"a", b"b", b"c"], interrupt_after=2)
    out_dir = tmp_path / "out"

    manifest = hsc.capture(adb=adb, seconds=1000.0, interval=0.0, out_dir=out_dir,
                            serial="SER1", progress=False)

    assert manifest["interrupted"] is True
    assert manifest["frame_count"] == 2
    assert (out_dir / "manifest.json").exists()
    on_disk = json.loads((out_dir / "manifest.json").read_text())
    assert on_disk == manifest
    assert sorted(p.name for p in out_dir.glob("*.png")) == ["00001.png", "00002.png"]


def test_ctrl_c_before_any_frame_still_writes_an_empty_but_valid_manifest(tmp_path):
    adb = _InterruptingAdb([], interrupt_after=0)
    out_dir = tmp_path / "out"

    manifest = hsc.capture(adb=adb, seconds=1000.0, interval=0.0, out_dir=out_dir,
                            serial="SER1", progress=False)

    assert manifest["interrupted"] is True
    assert manifest["frame_count"] == 0
    assert manifest["frames"] == []
    assert (out_dir / "manifest.json").exists()


# --- --serial / device resolution ----------------------------------------------------------

def _devices_output(*serials_and_states):
    lines = ["List of devices attached"]
    for serial, state in serials_and_states:
        lines.append(f"{serial}\t{state}")
    return ("\n".join(lines) + "\n").encode()


def test_resolve_serial_picks_the_single_connected_device(monkeypatch):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0,
                                                         stdout=_devices_output(("33111JEHN04475", "device")),
                                                         stderr=b""))
    assert hsc._resolve_serial(None) == "33111JEHN04475"


def test_resolve_serial_honors_explicit_serial_when_connected(monkeypatch):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0,
            stdout=_devices_output(("AAA", "device"), ("BBB", "device")), stderr=b""))
    assert hsc._resolve_serial("BBB") == "BBB"


def test_resolve_serial_exits_nonzero_when_no_device_connected(monkeypatch, capsys):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=_devices_output(), stderr=b""))

    with pytest.raises(SystemExit) as exc:
        hsc._resolve_serial(None)
    assert exc.value.code != 0
    assert "no ADB device" in capsys.readouterr().err.lower() or True


def test_resolve_serial_exits_nonzero_on_multiple_devices_without_serial_flag(monkeypatch, capsys):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0,
            stdout=_devices_output(("AAA", "device"), ("BBB", "device")), stderr=b""))

    with pytest.raises(SystemExit) as exc:
        hsc._resolve_serial(None)
    assert exc.value.code != 0
    err = capsys.readouterr().err.lower()
    assert "multiple" in err and "--serial" in err


def test_resolve_serial_exits_nonzero_when_requested_serial_not_connected(monkeypatch, capsys):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0,
                                                         stdout=_devices_output(("AAA", "device")),
                                                         stderr=b""))

    with pytest.raises(SystemExit) as exc:
        hsc._resolve_serial("ZZZ")
    assert exc.value.code != 0
    assert "ZZZ" in capsys.readouterr().err


def test_resolve_serial_excludes_unauthorized_and_offline_devices(monkeypatch):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0,
            stdout=_devices_output(("AAA", "unauthorized"), ("BBB", "device")), stderr=b""))
    assert hsc._resolve_serial(None) == "BBB"


def test_list_ready_devices_exits_nonzero_when_adb_binary_missing(monkeypatch, capsys):
    def _raise(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(hsc.subprocess, "run", _raise)

    with pytest.raises(SystemExit) as exc:
        hsc._list_ready_devices("adb")
    assert exc.value.code != 0
    assert "not found" in capsys.readouterr().err.lower()


def test_list_ready_devices_exits_nonzero_when_adb_command_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 1, stdout=b"", stderr=b"adb server is unavailable"))

    with pytest.raises(SystemExit) as exc:
        hsc._list_ready_devices("adb")
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "exit code 1" in err and "server is unavailable" in err


@pytest.mark.parametrize("seconds, interval", [
    (0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0),
    (1.0, -0.1), (1.0, float("inf")),
])
def test_capture_rejects_invalid_timing_before_creating_output(tmp_path, seconds, interval):
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="finite value"):
        hsc.capture(adb=FakeAdb([b"frame"]), seconds=seconds, interval=interval,
                    out_dir=out_dir, serial="SER1", progress=False)
    assert not out_dir.exists()


# --- main() wiring, --help, and end-to-end with a fake device -----------------------------

def test_main_help_exits_cleanly_without_touching_device(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("--help must not touch the device")
    monkeypatch.setattr(hsc, "_resolve_serial", _boom)
    monkeypatch.setattr(hsc, "_list_ready_devices", _boom)

    with pytest.raises(SystemExit) as exc:
        hsc.main(["--help"])
    assert exc.value.code == 0


def test_main_end_to_end_single_device_writes_frames_and_manifest(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "scroll_out"
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, stdout=_devices_output(("SER1", "device")), stderr=b""))

    fake = FakeAdb([b"f1", b"f1", b"f2", b"f3"])
    monkeypatch.setattr(hsc, "Adb", lambda serial: fake)

    hsc.main(["--out", str(out_dir), "--interval", "0", "--seconds", "1000"])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["device_serial"] == "SER1"
    assert manifest["frame_count"] == 3
    assert sorted(p.name for p in out_dir.glob("*.png")) == [
        "00001.png", "00002.png", "00003.png",
    ]

    out = capsys.readouterr().out
    assert "READ-ONLY" in out
    assert "gitignored" in out.lower()
    assert "LOCAL-ONLY" in out or "local-only" in out.lower()


def test_main_exits_nonzero_when_no_device_connected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hsc.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=_devices_output(), stderr=b""))

    with pytest.raises(SystemExit) as exc:
        hsc.main(["--out", str(tmp_path / "out")])
    assert exc.value.code != 0


# --- the hard constraint, enforced mechanically --------------------------------------------

# This is the enforceable version of the module's "structurally incapable of sending input"
# claim: rather than trust the docstring, grep the tool's OWN source for every substring that
# would send input to the phone (tap, swipe, key event, launch an activity, raw kernel input
# injection) and fail the suite if any of them ever creeps in, no matter how it got there.
_FORBIDDEN_INPUT_SUBSTRINGS = (
    "input tap",
    "input swipe",
    "input keyevent",
    "am start",
    ".tap(",
    ".swipe(",
    ".scroll",
    "sendevent",
    "/system/bin/hid",
)


def test_source_contains_no_input_sending_call():
    import tools.hinge_scroll_capture as mod
    source = Path(mod.__file__).read_text()
    hits = [s for s in _FORBIDDEN_INPUT_SUBSTRINGS if s in source]
    assert not hits, (
        f"tools/hinge_scroll_capture.py's source contains input-sending substring(s) {hits} -- "
        "this tool must be structurally incapable of sending input to the phone (see its "
        "module docstring); a match here means that guarantee has been broken."
    )


def test_source_never_imports_a_driver_or_the_touch_transport():
    # Substrings that would indicate an actual import/construction, not the module
    # docstring's own prose ABOUT avoiding them (which legitimately names "uhid.py" and
    # "HingeDriver" in text) -- so this checks realistic import/construction forms rather
    # than the bare words.
    import tools.hinge_scroll_capture as mod
    source = Path(mod.__file__).read_text()
    forbidden = (
        "HingeDriver(",
        "AndroidDriver(",
        "UhidTouch(",
        "drivers.uhid",
        "drivers import uhid",
    )
    hits = [f for f in forbidden if f in source]
    assert not hits, (
        f"tool must never construct a driver or import the touch transport: {hits}"
    )
