"""Offline tests for the UHID transport (per-gesture-file design).

No device: a FakeAdb captures write_file payloads + shell calls, so we verify the HID
report bytes, the dynamic descriptor sizing, the register+report/delay gesture script,
and that each gesture writes a file then runs `hid <file>`. The live `hid`/`/dev/uhid`
behaviour is verified on-device separately.
"""
import json
import random
import shlex

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
    # ONE compound remote call now runs hid AND removes the file -- see _run_gesture's
    # docstring for why it is `hid <file>; ec=$?; rm -f <file>; exit $ec`, not two round trips.
    expected = f"hid {drv.file_path}; ec=$?; rm -f {drv.file_path}; exit $ec"
    assert fa.shell_calls == [expected]


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


def test_close_quotes_the_gesture_file_as_one_literal_shell_target():
    fa = FakeAdb()
    file_path = "/data/local/tmp/gesture file; echo injected.json"
    UhidTouch(fa, file_path=file_path).close()

    assert fa.shell_calls == [f"rm -f {shlex.quote(file_path)}"]
    assert shlex.split(fa.shell_calls[0]) == ["rm", "-f", file_path]


def test_device_loss_propagates_from_write_file():
    class Lost(FakeAdb):
        def write_file(self, path, data):
            raise DriverClosed("ADB device was disconnected")

    with pytest.raises(DriverClosed):
        UhidTouch(Lost(), rng=random.Random(3)).swipe(540, 1700, 540, 700)


def test_write_file_permanent_failure_still_attempts_a_fallback_cleanup(monkeypatch):
    # The compound `hid ...; rm -f ...` command (see _run_gesture's docstring) only runs once
    # write_file has succeeded -- if BOTH write attempts fail, that command never gets a chance
    # to clean anything up. A `cat` redirect that errors partway through can still leave a
    # truncated script file on the device, so this one narrow path keeps its own separate
    # best-effort `rm -f`, exactly as every path got before the hid+rm collapse.
    class AlwaysFailsWrite(FakeAdb):
        def write_file(self, path, data):
            raise AdbError(["adb", "shell", "cat"], "write failed")

    monkeypatch.setattr(uhid.time, "sleep", lambda *a, **k: None)
    fa = AlwaysFailsWrite()

    with pytest.raises(DriverClosed, match="file write failed before delivery"):
        UhidTouch(fa, rng=random.Random(7)).swipe(540, 1700, 540, 700)

    assert not any(c.startswith("hid ") for c in fa.shell_calls)   # hid never ran
    assert any(c.startswith("rm -f") for c in fa.shell_calls)      # but cleanup was still attempted


def test_hid_failure_is_never_replayed_after_delivery_may_have_started():
    # A non-zero `hid` exit is ambiguous: it may have emitted every report before failing.
    # Replaying the file could duplicate an irreversible tap, so stop after one invocation.
    class Flaky(FakeAdb):
        def shell(self, cmd):
            self.shell_calls.append(cmd)
            if cmd.startswith("hid "):
                raise AdbError(["adb", "shell", cmd], "hid failed")
            return "yes" if "system/bin/hid" in cmd else ""

    fa = Flaky()
    with pytest.raises(DriverClosed, match="delivery became uncertain.*refusing to replay"):
        UhidTouch(fa, rng=random.Random(4)).swipe(540, 1700, 540, 700)
    assert sum(1 for c in fa.shell_calls if c.startswith("hid ")) == 1
    # Cleanup is now PART OF the one compound command sent for `hid` (see _run_gesture's
    # docstring) -- `rm -f` is unconditionally sequenced with `;`, not `&&`, so it is still
    # asked for even though `hid` itself failed. FakeAdb only records the command text (it does
    # not interpret shell syntax), so this proves we SENT the cleanup clause; the exit-code
    # semantics that make it actually run remotely regardless of `hid`'s outcome are pinned
    # separately below, against a real POSIX shell.
    assert "rm -f" in fa.shell_calls[0]


def test_hid_failure_cause_chain_names_hid_not_the_folded_in_cleanup():
    # The compound command folds `rm -f` into the same string as `hid`, but the raised
    # exception must still read -- and chain -- as a `hid` delivery failure specifically, not
    # something that could be misread as a cleanup problem now that cleanup shares the call.
    class Flaky(FakeAdb):
        def shell(self, cmd):
            self.shell_calls.append(cmd)
            if cmd.startswith("hid "):
                raise AdbError(["adb", "shell", cmd], "hid failed")
            return "yes" if "system/bin/hid" in cmd else ""

    fa = Flaky()
    with pytest.raises(DriverClosed, match="delivery became uncertain") as caught:
        UhidTouch(fa, rng=random.Random(6)).swipe(540, 1700, 540, 700)

    assert isinstance(caught.value.__cause__, AdbError)
    assert "hid failed" in str(caught.value.__cause__)
    assert sum(1 for c in fa.shell_calls if c.startswith("hid ")) == 1          # no replay


def test_compound_shell_command_preserves_hid_exit_code_regardless_of_rm_outcome():
    """The subtle part of this collapse: `ec=$?` must capture `hid`'s OWN exit status BEFORE
    `rm -f` runs, so the compound command's final exit code is always `hid`'s, never `rm -f`'s
    -- whatever `rm -f` itself does. FakeAdb cannot prove this: its fake `shell()` only records
    the command string, it never actually interprets it as shell syntax, and this suite has no
    real device to run `adb shell` against either. What it CAN do without either: the compound
    command is plain POSIX shell, and its exit-code composition is POSIX shell semantics, not
    anything Android- or adb-specific -- so running the exact shape `_run_gesture` sends through
    a local `sh -c`, with `hid`/`rm -f` swapped for the `true`/`false` builtins, proves the same
    claim a real remote shell will honour on-device.
    """
    import subprocess

    def run(cmd: str) -> int:
        return subprocess.run(["sh", "-c", cmd], check=False).returncode

    def compound(hid_cmd: str, rm_cmd: str) -> str:
        return f"{hid_cmd}; ec=$?; {rm_cmd}; exit $ec"

    # hid succeeds, rm -f fails -- caller must still see success (rm's failure is swallowed).
    assert run(compound("true", "false")) == 0
    # hid fails, rm -f succeeds -- caller must still see hid's failure.
    assert run(compound("false", "true")) != 0
    # hid fails, rm -f ALSO fails -- caller must still see hid's failure, never rm's.
    assert run(compound("false", "false")) != 0

    # Non-vacuity (required by the task this collapse was built for): the REJECTED naive shape
    # `hid ...; rm -f ...` -- no `$?` capture -- really does leak rm -f's exit status instead of
    # hid's, proving the assertions above actually distinguish the safe shape from the unsafe
    # one rather than passing no matter which is used.
    naive = "false; true"          # hid "fails", rm -f "succeeds"
    assert run(naive) == 0   # the bug: hid's failure vanishes


# --- per-gesture timing ledger (2026-08-23, one level down from hinge.py's own) --------
def test_swipe_records_the_named_timing_buckets_when_given_a_stamps_dict():
    fa = FakeAdb()
    drv = UhidTouch(fa, rng=random.Random(1))
    stamps: dict[str, float] = {}

    drv.swipe(540, 1700, 540, 700, _timing=stamps)

    # Exactly the buckets this transport can name from the host side: the pure-CPU planner,
    # building the gesture script, and the two `adb shell` round trips write-file / hid+rm-f --
    # see _run_gesture's own docstring for why there are exactly two now (2026-08-24: `hid` and
    # `rm -f` collapsed into one remote call, so there is no separate "uhid_cleanup_s" bucket
    # to report any more -- pinned here by the exact-set equality below, not just an absence
    # check, so a regression that reintroduces a standalone cleanup bucket fails this test).
    assert set(stamps) == {"uhid_plan_swipe_s", "uhid_script_build_s", "uhid_write_file_s",
                          "uhid_hid_run_s"}
    assert all(v >= 0.0 for v in stamps.values())


def test_scroll_up_adds_its_own_screen_size_bucket_on_top_of_swipes(monkeypatch):
    fa = FakeAdb()
    drv = UhidTouch(fa, rng=random.Random(1))
    stamps: dict[str, float] = {}

    drv.scroll_up(_timing=stamps)

    assert "screen_size_s" in stamps
    assert set(stamps) >= {"screen_size_s", "uhid_plan_swipe_s", "uhid_script_build_s",
                          "uhid_write_file_s", "uhid_hid_run_s"}
    assert "uhid_cleanup_s" not in stamps


def test_swipe_with_no_timing_dict_is_a_true_noop(monkeypatch):
    """The default -- every caller before 2026-08-23, and every caller today whose transport
    is not asked to report -- costs nothing extra: no dict, no `time.monotonic()` call."""
    calls = []
    monkeypatch.setattr(uhid.time, "monotonic", lambda: calls.append(1) or 0.0)
    fa = FakeAdb()
    UhidTouch(fa, rng=random.Random(1)).swipe(540, 1700, 540, 700)
    assert calls == []


def test_run_gesture_empty_samples_records_nothing_even_with_a_stamps_dict():
    fa = FakeAdb()
    stamps: dict[str, float] = {}
    UhidTouch(fa)._run_gesture([], _timing=stamps)
    assert stamps == {} and fa.files == {} and fa.shell_calls == []


def test_gesture_file_write_may_retry_before_any_hid_delivery(monkeypatch):
    class WriteFlaky(FakeAdb):
        def __init__(self):
            super().__init__()
            self.write_calls = 0

        def write_file(self, path, data):
            self.write_calls += 1
            if self.write_calls == 1:
                raise AdbError(["adb", "shell", "cat"], "write failed")
            super().write_file(path, data)

    fa = WriteFlaky()
    monkeypatch.setattr(uhid.time, "sleep", lambda *a, **k: None)

    UhidTouch(fa, rng=random.Random(5)).swipe(540, 1700, 540, 700)

    assert fa.write_calls == 2
    assert sum(1 for c in fa.shell_calls if c.startswith("hid ")) == 1
