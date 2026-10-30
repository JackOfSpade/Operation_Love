"""Exit-zero ADB reads must still contain complete, decodable screenshot pixels."""

import struct
import subprocess
import zlib

import pytest

from operation_love.drivers.adb import Adb, AdbError
from operation_love.drivers.base import DriverClosed


def _chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _frame(*, pixels=b"\x00\x00", compressed=None):
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(pixels) if compressed is None else compressed)
        + _chunk(b"IEND", b"")
    )


def _reads(monkeypatch, *results):
    remaining = list(results)
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        assert remaining, "screencap exceeded its read-only retry budget"
        result = remaining.pop(0)
        if isinstance(result, BaseException):
            raise result
        return subprocess.CompletedProcess(argv, 0, stdout=result, stderr=b"")

    monkeypatch.setattr("operation_love.drivers.adb.subprocess.run", run)
    return calls


@pytest.mark.parametrize(
    "broken",
    [
        b"",
        b"\x89PNG\r\n\x1a\n",
        _frame()[:45],  # partial IDAT despite a valid IHDR, as in run 11d40827ac78
        _frame()[:-12],  # all pixels present, but no terminal IEND
        _frame(compressed=b"not a zlib stream"),  # valid framing and CRCs
        _frame(pixels=b"\x00"),  # zlib EOF but incomplete raster
        _frame(pixels=b"\x05\x00"),  # correct row length, invalid reconstruction filter
        _frame()[:-1] + b"\x00",  # bad chunk CRC
    ],
    ids=[
        "empty",
        "header",
        "truncated-idat",
        "missing-iend",
        "bad-zlib",
        "short-raster",
        "invalid-filter",
        "crc",
    ],
)
def test_incomplete_exit_zero_screencap_gets_one_read_only_retry(monkeypatch, broken):
    good = _frame()
    calls = _reads(monkeypatch, broken, good)

    assert Adb(serial="pixel").screencap() == good

    assert calls == [["adb", "-s", "pixel", "exec-out", "screencap", "-p"]] * 2


def test_blank_complete_screen_is_valid_and_keeps_original_bytes(monkeypatch):
    blank = _frame()
    calls = _reads(monkeypatch, blank)

    assert Adb(serial="pixel").screencap() is blank
    assert len(calls) == 1


def test_two_corrupt_frames_fail_with_bounded_diagnostics(monkeypatch):
    broken = _frame()[:45]
    calls = _reads(monkeypatch, broken, broken)

    with pytest.raises(
        AdbError, match="recovery exhausted after 2 read-only attempts"
    ) as exc:
        Adb(serial="pixel").screencap()

    assert len(calls) == 2
    assert "invalid or incomplete PNG (45 bytes)" in str(exc.value)
    assert "initial invalid PNG" in exc.value.__notes__[0]
    assert "invalid or incomplete PNG" in str(exc.value.__cause__)


@pytest.mark.parametrize("timeout_first", [False, True])
def test_corrupt_frames_and_timeouts_share_one_total_retry(monkeypatch, timeout_first):
    timeout = subprocess.TimeoutExpired(["adb", "exec-out", "screencap", "-p"], 10)
    broken = _frame()[:45]
    results = [timeout, broken] if timeout_first else [broken, timeout]
    calls = _reads(monkeypatch, *results)

    with pytest.raises(AdbError, match="recovery exhausted after 2 read-only attempts"):
        Adb(serial="pixel").screencap()

    assert len(calls) == 2


def test_device_loss_on_corrupt_frame_retry_remains_terminal(monkeypatch):
    calls = _reads(
        monkeypatch,
        _frame()[:45],
        subprocess.TimeoutExpired(["adb"], 10, stderr=b"error: device offline"),
    )

    with pytest.raises(DriverClosed, match="device was disconnected"):
        Adb(serial="pixel").screencap()

    assert len(calls) == 2
