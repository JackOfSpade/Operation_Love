"""Offline tests for the UHID transport (per-gesture-file design).

No device: a FakeAdb captures write_file payloads + shell calls, so we verify the HID
report bytes, the dynamic descriptor sizing, the register+report/delay gesture script,
and that each gesture writes a file then runs `hid <file>`. The live `hid`/`/dev/uhid`
behaviour is verified on-device separately.
"""
import json
import random

import pytest

from operation_love.drivers import uhid
from operation_love.drivers.adb import AdbError
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.uhid import UhidTouch, _build_descriptor, _report
from operation_love.human_motion import TouchSample


class FakeAdb:
    adb_path = "adb"
    serial = "pixel"

    def __init__(self):
        self.shell_calls = []
        self.files = {}

    def screen_size(self):
        return (1080, 2400)

    def shell(self, cmd):
        self.shell_calls.append(cmd)
        return "yes" if "system/bin/hid" in cmd else ""   # hid present by default

    def write_file(self, path, data):
        self.files[path] = data


def _cmds(data: bytes):
    return [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]


# --- report bytes -------------------------------------------------------
def test_report_layout_and_pressure_scaling():
    down = _report(TouchSample(0, 540, 1200, 0.706, 0.5, True), 1080, 2400)
    assert down[0] == 0x01 and down[1] == 0x03 and down[2] == 0x00 and down[8] == 1
    assert down[3] == 540 & 0xFF and down[4] == (540 >> 8) & 0xFF
    assert down[5] == 1200 & 0xFF and down[6] == (1200 >> 8) & 0xFF
    assert down[7] == round(0.706 * 255)                                       # -> 180
    up = _report(TouchSample(0, 540, 1200, 0.0, 0.0, False), 1080, 2400)
    assert up[1] == 0x00 and up[7] == 0 and up[8] == 0


def test_report_clamps_coords_and_pressure():
    r = _report(TouchSample(0, -5, 9999, 2.0, 0.0, True), 1080, 2400)
    x = r[3] | (r[4] << 8)
    y = r[5] | (r[6] << 8)
    assert 0 <= x <= 1079 and 0 <= y <= 2399
    assert r[7] == 255


# --- descriptor uses live screen dims (Codex fix #1) -------------------
def test_descriptor_encodes_screen_dimensions():
    d = _build_descriptor(1080, 2400)
    ix = d.index(0x26)                                                          # 1st 0x26 = X logical max
    assert d[ix + 1: ix + 3] == [0x37, 0x04]                                    # X max = 1079
    iy = d.index(0x26, ix + 1)                                                  # 2nd 0x26 = Y logical max
    assert d[iy + 1: iy + 3] == [0x5F, 0x09]                                    # Y max = 2399
    d2 = _build_descriptor(1440, 3120)
    assert d2 != d                                                              # different resolution -> different descriptor
    jx = d2.index(0x26)
    assert d2[jx + 1: jx + 3] == [0x9F, 0x05]                                   # X max 1439 -> 0x059F LE
    jy = d2.index(0x26, jx + 1)
    assert d2[jy + 1: jy + 3] == [0x2F, 0x0C]                                   # Y max 3119 -> 0x0C2F LE


# --- gesture -> register + report/delay script -------------------------
def test_gesture_script_structure():
    drv = UhidTouch(FakeAdb())
    samples = [
        TouchSample(0.000, 100, 200, 0.5, 0.3, True),
        TouchSample(0.010, 110, 220, 0.7, 0.4, True),
        TouchSample(0.020, 110, 220, 0.0, 0.0, False),
    ]
    cmds = _cmds(drv._gesture_script(samples))
    assert cmds[0]["command"] == "register" and cmds[0]["descriptor"]
    assert drv.enumerate_ms <= cmds[1]["duration"] <= drv.enumerate_ms + 180
    assert drv.flush_ms <= cmds[-1]["duration"] <= drv.flush_ms + 90
    report_delays = [c["duration"] for c in cmds[2:-1] if c["command"] == "delay"]
    assert report_delays == [10, 10]                                              # 0.01s -> 10ms
    reports = [c["report"] for c in cmds if c["command"] == "report"]
    assert reports[0][8] == 1 and reports[-1][8] == 0                           # down ... up


def test_uhid_delays_vary_per_gesture_but_never_undercut_safe_baselines():
    drv = UhidTouch(FakeAdb(), rng=random.Random(21), name="explicit_name")
    samples = [TouchSample(0.0, 100, 200, 0.5, 0.3, True),
               TouchSample(0.02, 100, 200, 0.0, 0.0, False)]
    pairs = []
    for _ in range(30):
        cmds = _cmds(drv._gesture_script(samples))
        pairs.append((cmds[1]["duration"], cmds[-1]["duration"]))
    assert len(set(pairs)) > 5
    assert all(drv.enumerate_ms <= e <= drv.enumerate_ms + 180 and
               drv.flush_ms <= f <= drv.flush_ms + 90 for e, f in pairs)


def test_default_uhid_name_is_session_stable_and_explicit_name_is_preserved():
    first, second = UhidTouch(FakeAdb()), UhidTouch(FakeAdb())
    assert first.name.startswith("og_touch_") and first.name != second.name
    samples = [TouchSample(0.0, 100, 200, 0.5, 0.3, True),
               TouchSample(0.02, 100, 200, 0.0, 0.0, False)]
    names = [_cmds(first._gesture_script(samples))[0]["name"] for _ in range(2)]
    assert names == [first.name, first.name]  # one session identity, not per-gesture identity
    explicit = UhidTouch(FakeAdb(), name="calibration_touch", vid=7, pid=9)
    assert (explicit.name, explicit.vid, explicit.pid) == ("calibration_touch", 7, 9)


def test_gesture_script_empty_is_noop():
    fa = FakeAdb()
    UhidTouch(fa)._run_gesture([])
    assert fa.files == {} and fa.shell_calls == []


# --- swipe/tap write a file then run hid -------------------------------
def test_swipe_writes_file_then_runs_hid():
    fa = FakeAdb()
    drv = UhidTouch(fa, rng=random.Random(1))
    drv.swipe(540, 1700, 540, 700)
    assert drv.file_path in fa.files
    cmds = _cmds(fa.files[drv.file_path])
    assert cmds[0]["command"] == "register"
    reports = [c["report"] for c in cmds if c["command"] == "report"]
    assert reports[0][8] == 1 and reports[-1][8] == 0                           # down ... release
    for r in reports:
        x = r[3] | (r[4] << 8)
        y = r[5] | (r[6] << 8)
        assert 0 <= x <= 1079 and 0 <= y <= 2399
    assert f"hid {drv.file_path}" in fa.shell_calls                             # ran hid on the file
    assert f"rm -f {drv.file_path}" in fa.shell_calls                           # cleaned up after


def test_scroll_up_jitters_x_column():
    # HINGE-04: UHID is the PROVEN, genuine-to-the-kernel transport, so it must not be the one
    # emitting a pixel-identical swipe column every scroll -- Adb.scroll_up jitters x by +/-25px
    # (adb.scroll_x, shared) but UhidTouch previously did not.
    fa = FakeAdb()
    drv = UhidTouch(fa)
    xs = set()
    for _ in range(30):
        drv.scroll_up()
        cmds = _cmds(fa.files[drv.file_path])
        reports = [c["report"] for c in cmds if c["command"] == "report"]
        x0 = reports[0][3] | (reports[0][4] << 8)   # first sample's x is exact (no path jitter at k=0)
        xs.add(x0)
    assert len(xs) > 1     # jittered run to run, not a fixed column every time


def test_tap_writes_down_then_release():
    fa = FakeAdb()
    UhidTouch(fa, rng=random.Random(2)).tap(540, 1200)
    cmds = _cmds(next(iter(fa.files.values())))
    reports = [c["report"] for c in cmds if c["command"] == "report"]
    assert reports[0][8] == 1 and reports[-1][8] == 0
    assert max(r[7] for r in reports) > 0                                       # a real pressure pulse


# --- lifecycle ---------------------------------------------------------
def test_open_caches_screen_size():
    fa = FakeAdb()
    UhidTouch(fa).open()                                                        # hid present -> no raise


def test_open_raises_uhid_unavailable_when_hid_missing():
    from operation_love.drivers.uhid import UhidUnavailable

    class NoHid(FakeAdb):
        def shell(self, cmd):
            self.shell_calls.append(cmd)
            return "no"          # always-exit-0 probe -> stdout "no" when /system/bin/hid is absent

    with pytest.raises(UhidUnavailable):
        UhidTouch(NoHid()).open()


def test_open_probe_does_not_depend_on_exit_code():
    # Regression (audit): the presence probe must signal absence via STDOUT, not a non-zero
    # exit. An Adb that raises AdbError whenever a command would exit non-zero (the real adb
    # behavior `test -e ... &&` triggers on absence) must NOT surface during the probe.
    from operation_love.drivers.uhid import UhidUnavailable

    class ExitCodeAdb(FakeAdb):
        def shell(self, cmd):
            self.shell_calls.append(cmd)
            if "&&" in cmd and "||" not in cmd:                                # `test -e .. && echo` would exit 1
                raise AdbError(["adb", "shell", cmd], "exit code 1")
            return "no"                                                        # exit-0 form -> stdout "no"

    with pytest.raises(UhidUnavailable):                                       # clean fall back, not AdbError
        UhidTouch(ExitCodeAdb()).open()


def test_close_removes_the_gesture_file():
    fa = FakeAdb()
    drv = UhidTouch(fa)
    drv.close()
    assert any(c == f"rm -f {drv.file_path}" for c in fa.shell_calls)


def test_device_loss_propagates_from_write_file():
    class Lost(FakeAdb):
        def write_file(self, path, data):
            raise DriverClosed("ADB device was disconnected")

    with pytest.raises(DriverClosed):
        UhidTouch(Lost(), rng=random.Random(3)).swipe(540, 1700, 540, 700)


def test_hid_failure_retries_once_then_raises_driver_closed(monkeypatch):
    # A non-zero `hid` exit (e.g. a prior virtual device still tearing down) is retried
    # once, then surfaced as DriverClosed (clean stop) rather than a raw AdbError crash.
    class Flaky(FakeAdb):
        def shell(self, cmd):
            self.shell_calls.append(cmd)
            if cmd.startswith("hid "):
                raise AdbError(["adb", "shell", cmd], "hid failed")
            return "yes" if "system/bin/hid" in cmd else ""

    fa = Flaky()
    monkeypatch.setattr(uhid.time, "sleep", lambda *a, **k: None)               # no real backoff wait
    with pytest.raises(DriverClosed):
        UhidTouch(fa, rng=random.Random(4)).swipe(540, 1700, 540, 700)
    assert sum(1 for c in fa.shell_calls if c.startswith("hid ")) == 2          # tried twice
    assert any(c.startswith("rm -f") for c in fa.shell_calls)                   # still cleaned up
