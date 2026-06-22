import subprocess

import pytest

from operation_love.drivers import adb as adb_mod
from operation_love.drivers.adb import Adb, AdbError
from operation_love.drivers.base import DriverClosed


def _ok(stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(["adb"], 0, stdout=stdout, stderr=stderr)


def _fail(stderr=b"boom", stdout=b""):
    return subprocess.CompletedProcess(["adb"], 1, stdout=stdout, stderr=stderr)


class FakeRun:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, *, capture_output, timeout):
        self.calls.append((list(argv), capture_output, timeout))
        if not self.results:
            raise AssertionError("unexpected subprocess.run call")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    @property
    def argv(self):
        return [call[0] for call in self.calls]


def test_device_methods_build_expected_argv(monkeypatch):
    run = FakeRun(*[_ok() for _ in range(5)])
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    d = Adb(serial="pixel-7a", adb_path="/opt/android/adb", default_timeout=3.5)

    d.tap(10, 20)
    d.swipe(1, 2, 3, 4)
    d.key("HOME")
    d.key(66)
    d.text("hi there&you")

    assert run.calls == [
        ([
            "/opt/android/adb",
            "-s",
            "pixel-7a",
            "shell",
            "input",
            "tap",
            "10",
            "20",
        ], True, 3.5),
        ([
            "/opt/android/adb",
            "-s",
            "pixel-7a",
            "shell",
            "input",
            "swipe",
            "1",
            "2",
            "3",
            "4",
            "200",
        ], True, 3.5),
        ([
            "/opt/android/adb",
            "-s",
            "pixel-7a",
            "shell",
            "input",
            "keyevent",
            "HOME",
        ], True, 3.5),
        ([
            "/opt/android/adb",
            "-s",
            "pixel-7a",
            "shell",
            "input",
            "keyevent",
            "66",
        ], True, 3.5),
        ([
            "/opt/android/adb",
            "-s",
            "pixel-7a",
            "shell",
            "input",
            "text",
            "hi%sthere\\&you",
        ], True, 3.5),
    ]


def test_swipe_uses_custom_duration(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").swipe(10, 20, 30, 40, duration_ms=450)

    assert run.argv == [[
        "adb",
        "-s",
        "pixel",
        "shell",
        "input",
        "swipe",
        "10",
        "20",
        "30",
        "40",
        "450",
    ]]


def test_screencap_uses_exec_out_and_returns_raw_bytes(monkeypatch):
    png = b"\x89PNG\r\n\x1a\nraw\r\nbytes"
    run = FakeRun(_ok(stdout=png))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    got = Adb(serial="pixel").screencap()

    assert got == png
    assert run.argv == [["adb", "-s", "pixel", "exec-out", "screencap", "-p"]]


def test_devices_parses_only_ready_devices(monkeypatch):
    run = FakeRun(_ok(stdout=b"""List of devices attached
pixel-1	device
pixel-2	offline
pixel-3	unauthorized
pixel-4	device product:foo model:bar

"""))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    assert Adb(serial="ignored").devices() == ["pixel-1", "pixel-4"]
    assert run.argv == [["adb", "devices"]]


def test_non_zero_exit_raises_adb_error(monkeypatch):
    run = FakeRun(_fail(stderr=b"bad command"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb().tap(1, 2)

    assert exc.value.argv == ["adb", "shell", "input", "tap", "1", "2"]
    assert exc.value.stderr == "bad command"
    assert "ADB command failed with exit code 1" in str(exc.value)
    assert "stderr: bad command" in str(exc.value)


def test_timeout_raises_adb_error(monkeypatch):
    timeout = subprocess.TimeoutExpired(
        ["adb", "shell", "input", "tap", "1", "2"],
        timeout=10,
        stderr=b"still waiting",
    )
    run = FakeRun(timeout)
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb(default_timeout=10).tap(1, 2)

    assert exc.value.argv == ["adb", "shell", "input", "tap", "1", "2"]
    assert exc.value.stderr == "still waiting"
    assert "timed out after 10s" in str(exc.value)


@pytest.mark.parametrize("stderr", [
    b"error: device offline",
    b"error: device 'pixel' not found",
    b"error: no devices/emulators found",
    b"error: device unauthorized",
    b"error: unauthorized",
])
def test_device_lost_stderr_raises_driver_closed(monkeypatch, stderr):
    run = FakeRun(_fail(stderr=stderr))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(DriverClosed):
        Adb(serial="pixel").key("HOME")


def test_device_lost_via_stdout_raises_driver_closed(monkeypatch):
    # Detection folds in stdout too (some adb builds emit the message there, not
    # on stderr). Guards against a regression that only inspected stderr.
    run = FakeRun(_fail(stderr=b"", stdout=b"error: device offline"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(DriverClosed):
        Adb(serial="pixel").key("HOME")
