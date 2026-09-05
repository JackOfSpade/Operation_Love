"""Offline tests for the UHID transport (per-gesture-file design).

No device: a FakeAdb captures write_file payloads + shell calls, so we verify the HID
report bytes, the dynamic descriptor sizing, the register+report/delay gesture script,
and that each gesture writes a file then runs `hid <file>`. The live `hid`/`/dev/uhid`
behaviour is verified on-device separately.
"""
import functools
import json
import random
import shlex

import pytest

from operation_love.drivers import uhid
from operation_love.drivers.adb import AdbError
from operation_love.drivers.base import DriverClosed
from operation_love.drivers.uhid import PersistentUhidTouch, UhidTouch, UhidUnavailable, _build_descriptor, _report
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


# =====================================================================================
# PersistentUhidTouch (2026-08-24) -- one virtual device registered ONCE per session, not
# per gesture. No real device or real `adb`: subprocess.Popen is monkeypatched with a fake
# that captures every write to a fake stdin and exposes a controllable poll()/kill(), the
# same convention tests/test_touchwatch.py already uses for its own persistent-Popen class
# (a different module, the identical shape: a long-lived Popen this driver owns).
# =====================================================================================

def _cmds_from_strings(lines: list) -> list:
    """Same job as this file's `_cmds(data: bytes)` above, for PersistentUhidTouch's
    str-mode stdin writes instead of UhidTouch's one-shot bytes payload: `lines` is every
    individual `.write()` call's argument, in order; joining and re-splitting recovers the
    full command sequence regardless of exactly how many separate write() calls produced it."""
    return [json.loads(ln) for ln in "".join(lines).splitlines() if ln.strip()]


def _popen_must_not_be_called(*_a, **_k):
    raise AssertionError(
        "subprocess.Popen must not be called on this path (a real adb process would "
        "otherwise be spawned by this test)")


class _FakePersistentStdin:
    """Stands in for Popen.stdin: captures every write() call's argument verbatim, and can
    be told to raise (simulating a broken pipe, or a stream already closed out from under
    this object) starting from the `fail_after`-th write -- `fail_after=0` means "fail on
    the very first write", matching how a registration- or gesture-time failure is tested."""

    def __init__(self, fail_after: int | None = None, raise_type=BrokenPipeError):
        self.written: list = []
        self.flush_count = 0
        self.closed = False
        self._fail_after = fail_after
        self._raise_type = raise_type

    def write(self, s):
        if self._fail_after is not None and len(self.written) >= self._fail_after:
            raise self._raise_type("broken pipe")
        self.written.append(s)

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class _FakePersistentPopen:
    """Stands in for subprocess.Popen for PersistentUhidTouch. `poll()` returns None while
    alive; `die_after_polls`, when given, makes the N-th `poll()` call (and every one after
    it) report death -- this is what lets a test put the process's death at an EXACT point
    in a gesture's guard-check / post-wait-check sequence without a real clock or a real
    subprocess. `start_alive=False` is the simpler "already dead from the very first check"
    case (dies during open()'s own registration/enumeration window)."""

    instances: list = []   # every instance constructed since the last `.clear()` -- lets a
    # test assert exactly one process was ever spawned for a whole session (no re-registration).

    def __init__(self, argv, *, start_alive: bool = True, die_after_polls: int | None = None,
                 stdin_fail_after: int | None = None, stdin_raise_type=BrokenPipeError, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.stdin = _FakePersistentStdin(fail_after=stdin_fail_after, raise_type=stdin_raise_type)
        # The real persistent PTY command deliberately discards its unused output.  Mirroring
        # Popen's `None` stream attributes here makes close() exercise that production shape.
        self.stdout = None
        self.stderr = None
        self._alive = start_alive
        self._die_after_polls = die_after_polls
        self._poll_calls = 0
        self.kill_called = False
        self.wait_calls = 0
        _FakePersistentPopen.instances.append(self)

    def poll(self):
        self._poll_calls += 1
        if self._die_after_polls is not None and self._poll_calls >= self._die_after_polls:
            self._alive = False
        return None if self._alive else 1

    def kill(self):
        self.kill_called = True
        self._alive = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        return 0


@pytest.fixture(autouse=True)
def _reset_fake_persistent_popen_instances():
    _FakePersistentPopen.instances.clear()
    yield
    _FakePersistentPopen.instances.clear()


def _quiet_sleep(monkeypatch):
    """PersistentUhidTouch sleeps for real durations in open() (enumerate_ms) and every
    gesture (the sum of that gesture's own delay durations) -- monkeypatched to a no-op in
    every test below, same as this file's existing UhidTouch tests already do for their own
    retry-backoff sleep (`test_write_file_permanent_failure_still_attempts_a_fallback_cleanup`,
    `test_gesture_file_write_may_retry_before_any_hid_delivery`)."""
    monkeypatch.setattr(uhid.time, "sleep", lambda *a, **k: None)


# --- open(): registration, no re-registration on later gestures ------------------------
def test_persistent_open_registers_once_and_gestures_never_reregister(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    fa = FakeAdb()
    drv = PersistentUhidTouch(fa, rng=random.Random(1))
    drv.open()

    assert len(_FakePersistentPopen.instances) == 1
    proc = _FakePersistentPopen.instances[0]
    assert proc.kwargs["stdin"] is uhid.subprocess.PIPE
    assert proc.kwargs["stdout"] is uhid.subprocess.DEVNULL
    assert proc.kwargs["stderr"] is uhid.subprocess.DEVNULL
    open_cmds = _cmds_from_strings(proc.stdin.written)
    assert [c["command"] for c in open_cmds] == ["register", "delay"]
    assert open_cmds[0]["descriptor"] and open_cmds[0]["name"] == drv.name
    assert drv.enumerate_ms <= open_cmds[1]["duration"] <= drv.enumerate_ms + 180

    before = len(proc.stdin.written)
    drv.swipe(540, 1700, 540, 700)
    drv.swipe(540, 700, 540, 1700)

    assert len(_FakePersistentPopen.instances) == 1        # same process, no re-spawn
    gesture_cmds = _cmds_from_strings(proc.stdin.written[before:])
    assert gesture_cmds                                     # gestures actually wrote something
    assert all(c["command"] != "register" for c in gesture_cmds)


def test_persistent_two_gestures_write_distinct_report_streams_no_register(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb(), rng=random.Random(2))
    drv.open()
    proc = _FakePersistentPopen.instances[0]

    mark = len(proc.stdin.written)
    drv.tap(200, 900)
    first = _cmds_from_strings(proc.stdin.written[mark:])
    mark = len(proc.stdin.written)
    drv.tap(800, 1900)
    second = _cmds_from_strings(proc.stdin.written[mark:])

    assert all(c["command"] != "register" for c in first + second)
    first_reports = [c["report"] for c in first if c["command"] == "report"]
    second_reports = [c["report"] for c in second if c["command"] == "report"]
    assert first_reports and second_reports
    assert first_reports != second_reports     # different tap coords -> genuinely distinct streams


# --- open() failure modes: all UhidUnavailable, never DriverClosed ---------------------
def test_persistent_open_raises_uhid_unavailable_when_hid_missing(monkeypatch):
    monkeypatch.setattr(uhid.subprocess, "Popen", _popen_must_not_be_called)

    class NoHid(FakeAdb):
        def shell(self, cmd):
            self.shell_calls.append(cmd)
            return "no"

    with pytest.raises(UhidUnavailable):
        PersistentUhidTouch(NoHid()).open()


def test_persistent_open_raises_uhid_unavailable_when_popen_cannot_start(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("adb binary not found")

    monkeypatch.setattr(uhid.subprocess, "Popen", boom)
    with pytest.raises(UhidUnavailable):
        PersistentUhidTouch(FakeAdb()).open()


def test_persistent_open_raises_uhid_unavailable_when_registration_write_fails(monkeypatch):
    monkeypatch.setattr(
        uhid.subprocess, "Popen", functools.partial(_FakePersistentPopen, stdin_fail_after=0))

    with pytest.raises(UhidUnavailable, match="rejected registration"):
        PersistentUhidTouch(FakeAdb()).open()

    proc = _FakePersistentPopen.instances[0]
    assert proc.kill_called      # the half-registered process is torn down, not left dangling


def test_persistent_open_raises_uhid_unavailable_when_process_dies_during_enumerate(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(
        uhid.subprocess, "Popen", functools.partial(_FakePersistentPopen, start_alive=False))

    with pytest.raises(UhidUnavailable, match="exited during registration"):
        PersistentUhidTouch(FakeAdb()).open()


# --- gesture delivery: success, and both "never replay" failure modes ------------------
def test_persistent_gesture_succeeds_silently_when_process_stays_alive(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb(), rng=random.Random(3))
    drv.open()
    drv.swipe(540, 1700, 540, 700)          # must not raise
    drv.tap(540, 1200)                       # must not raise


def test_persistent_death_after_gesture_marks_session_dead_and_next_call_never_writes(monkeypatch):
    _quiet_sleep(monkeypatch)
    # poll() call #1 is open()'s own post-enumerate check (must read alive); #2 is the NEXT
    # gesture's guard check (must also still read alive, or delivery would never be
    # attempted); #3 is that same gesture's post-wait check, where death is revealed.
    monkeypatch.setattr(
        uhid.subprocess, "Popen", functools.partial(_FakePersistentPopen, die_after_polls=3))
    drv = PersistentUhidTouch(FakeAdb(), rng=random.Random(4))
    drv.open()
    proc = _FakePersistentPopen.instances[0]

    with pytest.raises(DriverClosed, match="died during or after delivery"):
        drv.swipe(540, 1700, 540, 700)

    # THE most important property in this set: a session found dead must never be silently
    # reopened or resent against -- the very next call raises immediately, before writing
    # anything at all, proven here by the write count not moving.
    written_after_death = len(proc.stdin.written)
    with pytest.raises(DriverClosed, match="already died"):
        drv.swipe(540, 700, 540, 1700)
    assert len(proc.stdin.written) == written_after_death


def test_persistent_write_failure_marks_session_dead_and_next_call_never_writes(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb(), rng=random.Random(5))
    drv.open()
    proc = _FakePersistentPopen.instances[0]
    written_after_open = len(proc.stdin.written)
    proc.stdin._fail_after = 0     # every write from here on raises BrokenPipeError

    with pytest.raises(DriverClosed, match="stdin write"):
        drv.swipe(540, 1700, 540, 700)
    assert len(proc.stdin.written) == written_after_open   # nothing from this gesture landed

    with pytest.raises(DriverClosed, match="already died"):
        drv.swipe(540, 700, 540, 1700)
    assert len(proc.stdin.written) == written_after_open   # still nothing -- no retry, no reopen


def test_persistent_run_gesture_empty_samples_is_a_true_noop(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb())
    drv.open()
    proc = _FakePersistentPopen.instances[0]
    before = len(proc.stdin.written)

    drv._run_gesture([])

    assert len(proc.stdin.written) == before


# --- flush delay: same jittered range as UhidTouch's, not weakened ---------------------
def test_persistent_gesture_flush_delay_matches_uhid_touchs_jittered_range(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb(), rng=random.Random(6))
    drv.open()
    proc = _FakePersistentPopen.instances[0]

    seen = set()
    for _ in range(30):
        mark = len(proc.stdin.written)
        drv.swipe(540, 1700, 540, 700)
        cmds = _cmds_from_strings(proc.stdin.written[mark:])
        seen.add(cmds[-1]["duration"])     # the trailing flush delay is always the last command

    assert len(seen) > 5                                              # genuinely jittered
    assert all(drv.flush_ms <= v <= drv.flush_ms + 90 for v in seen)   # never below the safe baseline


# --- the gesture surface is ONE surface, and the shipped transport is the tested one -----
#
# Until 2026-09-04 tap/swipe/scroll_up were a character-identical SECOND copy on
# PersistentUhidTouch, and not one test in this file ever called that copy's scroll_up or
# exercised its duration scaling -- so the transport config.yaml actually ships was the untested
# one, and a humanization edit made to either class alone would have been silent in both
# directions. The identity test below is what makes the drift impossible; the two after it are
# the coverage that would have caught it anyway.

def test_both_uhid_transports_are_literally_the_same_gesture_surface():
    """`touch_backend: uhid` is documented as the instant revert to the proven per-gesture
    transport, which is only true while the two emit the same touch signature. The humanization
    (Fitts-law duration scaling, HINGE-04 x-column jitter, the plan_swipe parameters) is
    therefore ONE function each, shared by inheritance, not two that happen to match today."""
    for method in ("tap", "swipe", "scroll_up"):
        assert getattr(UhidTouch, method) is getattr(PersistentUhidTouch, method), method
        assert method not in UhidTouch.__dict__, f"{method} was re-overridden on UhidTouch"
        assert method not in PersistentUhidTouch.__dict__, \
            f"{method} was re-overridden on PersistentUhidTouch"
    # ...and delivery, the one thing they genuinely differ on, IS overridden by each of them.
    assert "_run_gesture" in UhidTouch.__dict__
    assert "_run_gesture" in PersistentUhidTouch.__dict__
    assert UhidTouch._run_gesture is not PersistentUhidTouch._run_gesture


def test_persistent_scroll_up_jitters_x_column(monkeypatch):
    """HINGE-04 on the SHIPPED transport, mirroring `test_scroll_up_jitters_x_column` above:
    the genuine-to-the-kernel transport must not be the one emitting a pixel-identical swipe
    column every scroll."""
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb())
    drv.open()
    proc = _FakePersistentPopen.instances[0]

    xs = set()
    for _ in range(30):
        mark = len(proc.stdin.written)
        drv.scroll_up()
        cmds = _cmds_from_strings(proc.stdin.written[mark:])
        reports = [c["report"] for c in cmds if c["command"] == "report"]
        # first sample's x is exact (no path jitter at k=0), same as the UhidTouch test
        xs.add(reports[0][3] | (reports[0][4] << 8))

    assert len(xs) > 1


def test_persistent_swipe_scales_its_duration_and_clamps_at_both_ends(monkeypatch):
    """`duration_scale = max(0.20, min(2.0, duration_ms / 450.0))` on the shipped transport.

    Asserted through what actually reaches the kernel -- the number of HID reports in the
    stream, which at a fixed report rate IS the gesture's duration -- rather than by reading the
    scale back off the object, so a planner wired up with the wrong scale fails here. Both
    clamps are checked, since an unclamped version would pass a monotonicity test alone.
    """
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)

    def reports_for(duration_ms):
        drv = PersistentUhidTouch(FakeAdb(), rng=random.Random(7))
        drv.open()
        proc = _FakePersistentPopen.instances[-1]
        mark = len(proc.stdin.written)
        drv.swipe(540, 1700, 540, 700, duration_ms=duration_ms)
        cmds = _cmds_from_strings(proc.stdin.written[mark:])
        return sum(1 for c in cmds if c["command"] == "report")

    default, short = reports_for(450), reports_for(150)
    assert short < default, "a shorter requested duration must deliver a shorter gesture"
    # The LOW clamp: 0.20 * 450 = 90ms is the floor, so anything under it plans identically
    # and a 1ms request can never collapse the gesture to a teleport (humanized-input rule).
    assert reports_for(90) == reports_for(45) == reports_for(1) < short
    # The HIGH clamp: 2.0 * 450 = 900ms, so a 5000ms request plans exactly the 900ms gesture.
    assert reports_for(900) == reports_for(1800) == reports_for(5000) > default


# --- close(): idempotent, best-effort, never raises -------------------------------------
def test_persistent_close_is_idempotent_and_never_raises_even_when_never_opened():
    drv = PersistentUhidTouch(FakeAdb())
    drv.close()     # never opened at all -- must not raise
    drv.close()     # idempotent


def test_persistent_close_kills_and_waits_on_the_process_and_closes_its_pipes(monkeypatch):
    _quiet_sleep(monkeypatch)
    monkeypatch.setattr(uhid.subprocess, "Popen", _FakePersistentPopen)
    drv = PersistentUhidTouch(FakeAdb())
    drv.open()
    proc = _FakePersistentPopen.instances[0]

    drv.close()

    assert proc.kill_called and proc.wait_calls == 1
    assert proc.stdin.closed
    assert proc.stdout is None and proc.stderr is None

    drv.close()     # idempotent: second close is a pure no-op, not a second kill/wait
    assert proc.wait_calls == 1


def test_persistent_close_swallows_kill_and_wait_failures(monkeypatch):
    _quiet_sleep(monkeypatch)

    class BoomOnTeardown(_FakePersistentPopen):
        def kill(self):
            raise OSError("no such process")

        def wait(self, timeout=None):
            raise TimeoutError("still running")

    monkeypatch.setattr(uhid.subprocess, "Popen", BoomOnTeardown)
    drv = PersistentUhidTouch(FakeAdb())
    drv.open()

    drv.close()     # must not raise despite kill()/wait() both blowing up
    assert drv._proc is None    # still torn down from this object's point of view
