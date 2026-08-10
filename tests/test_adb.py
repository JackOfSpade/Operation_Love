import re
import subprocess

import pytest

from operation_love.drivers import adb as adb_mod
from operation_love.drivers.adb import (
    Adb,
    AdbError,
    _clean_text_for_input,
    parse_devices_output,
    plan_path,
)
from operation_love.drivers.base import DriverClosed


def _ok(stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(["adb"], 0, stdout=stdout, stderr=stderr)


def _fail(stderr=b"boom", stdout=b""):
    return subprocess.CompletedProcess(["adb"], 1, stdout=stdout, stderr=stderr)


class FakeRun:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, *, capture_output=True, timeout=None, input=None):
        self.calls.append((list(argv), capture_output, timeout, input))
        if not self.results:
            raise AssertionError("unexpected subprocess.run call")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    @property
    def argv(self):
        return [call[0] for call in self.calls]

    @property
    def inputs(self):
        return [call[3] for call in self.calls]


def test_text_builds_expected_argv(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    d = Adb(serial="pixel-7a", adb_path="/opt/android/adb", default_timeout=3.5)

    d.text("hi there&you")

    assert run.argv == [
        ["/opt/android/adb", "-s", "pixel-7a", "shell", "input", "text", "hi%sthere\\&you"],
    ]
    assert all(c[1] is True for c in run.calls)     # capture_output
    assert all(c[2] == 3.5 for c in run.calls)      # timeout threaded through


# --- _clean_text_for_input: fold non-ASCII typography, never delete it (bug 3) ----------
def test_clean_text_folds_curly_quotes_and_ellipsis_instead_of_deleting():
    out = _clean_text_for_input("that’s a “bold” choice… love it")
    assert out == "that's a \"bold\" choice... love it"


def test_clean_text_folds_exotic_spaces_to_a_normal_space():
    out = _clean_text_for_input("a b c d")     # nbsp, narrow nbsp, thin space
    assert out == "a b c d"


def test_clean_text_removes_every_dash_variant():
    # Owner rule (b): openers must contain no dash of any kind (reads as AI). Consistent with
    # opener.py's _sanitize(), not a second policy.
    dashes = "—–‒―−‑‐-"   # em en figure horiz-bar minus nb-hyphen hyphen ascii
    out = _clean_text_for_input(f"a{dashes}b")
    for ch in "-—–‐‑‒―−":
        assert ch not in out


def test_clean_text_collapses_double_space_artifacts():
    # An unmapped codepoint (still dropped, as before) must not leave a double space where
    # it used to sit.
    out = _clean_text_for_input("choice \U0001F600 love")
    assert "  " not in out
    assert out == "choice love"


def test_clean_text_realistic_gemini_opener_reads_naturally():
    opener = "that’s a bold choice — I love it… truly “unique”"
    out = _clean_text_for_input(opener)
    assert "  " not in out
    assert not any(ch in out for ch in "—–‐‑‒―−-")
    # No stray space before the comma the em dash folded into: this layer now shares
    # opener.py's punctuation tidy-up, so both paths spell the same sentence.
    assert out == "that's a bold choice, I love it... truly \"unique\""


def test_clean_text_ascii_passthrough_unchanged():
    # Regression guard: the only case previously covered (via text(), see below) must still
    # pass through untouched now that folding is in the mix.
    assert _clean_text_for_input("hi there&you") == "hi there&you"


def test_text_folds_gemini_opener_typography_then_escapes_for_shell(monkeypatch):
    # Full-pipeline regression: fold -> escape -> shell argv, for typography Gemini actually
    # produces. Shell-escaping (%s for spaces, backslash for shell metacharacters) must still
    # work on the FOLDED string.
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").text("that’s bold — love it…")

    assert run.argv[0][:6] == ["adb", "-s", "pixel", "shell", "input", "text"]
    sent = run.argv[0][6]
    assert "’" not in sent and "—" not in sent and "…" not in sent
    assert "%s" in sent                              # spaces still escaped (existing behaviour)
    assert "\\'" in sent                              # folded apostrophe still shell-escaped


def test_tap_is_single_fork_tap(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").tap(500, 1200)

    assert run.argv[0][:4] == ["adb", "-s", "pixel", "shell"]
    assert run.argv[0][4] == "input"
    assert run.argv[0][5] == "tap"
    jx, jy = int(run.argv[0][6]), int(run.argv[0][7])
    assert abs(jx - 500) <= 2 and abs(jy - 1200) <= 2


def test_shell_runs_shell_command(monkeypatch):
    run = FakeRun(_ok(stdout=b"hello\nworld\n"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    res = Adb(serial="pixel").shell("echo hello")
    assert res == "hello\nworld"
    assert run.argv == [["adb", "-s", "pixel", "shell", "echo hello"]]


def test_swipe_is_humanized_bezier(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").swipe(540, 1800, 540, 700, duration_ms=500)

    assert run.argv == [["adb", "-s", "pixel", "shell"]]
    script = run.inputs[0].decode()
    assert script.count("input motionevent DOWN") == 1
    assert script.count("input motionevent UP") == 1
    assert script.count("input motionevent MOVE") >= 5     # curved multi-step path
    assert script.count("sleep") >= 5
    dx, dy = map(int, re.search(r"DOWN (\d+) (\d+)", script).groups())
    ux, uy = map(int, re.search(r"UP (\d+) (\d+)", script).groups())
    assert (dx, dy) == (540, 1800)                          # endpoints exact
    assert (ux, uy) == (540, 700)


def test_swipe_clamps_to_screen_when_size_known(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    d = Adb(serial="pixel")
    d._size = (1080, 2400)

    d.swipe(-50, -50, 5000, 5000, duration_ms=300)

    coords = re.findall(r"motionevent \w+ (-?\d+) (-?\d+)", run.inputs[0].decode())
    assert coords
    for xs, ys in coords:
        assert 0 <= int(xs) <= 1079 and 0 <= int(ys) <= 2399


def test_plan_path_endpoints_exact_and_count():
    pts = plan_path(100, 200, 900, 2000, steps=12)
    assert len(pts) == 13
    assert pts[0] == (100, 200)
    assert pts[-1] == (900, 2000)
    assert all(isinstance(p[0], int) and isinstance(p[1], int) for p in pts)


def test_screencap_uses_exec_out_and_returns_raw_bytes(monkeypatch):
    png = b"\x89PNG\r\n\x1a\nraw\r\nbytes"
    run = FakeRun(_ok(stdout=png))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    got = Adb(serial="pixel").screencap()

    assert got == png
    assert run.argv == [["adb", "-s", "pixel", "exec-out", "screencap", "-p"]]


def test_screen_size_parses_and_caches(monkeypatch):
    run = FakeRun(_ok(stdout=b"Physical size: 1080x2400\nOverride size: 1080x2160\n"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    d = Adb(serial="pixel")

    assert d.screen_size() == (1080, 2160)      # an Override size, if present, wins
    assert d.screen_size() == (1080, 2160)      # cached
    assert len(run.calls) == 1


def test_devices_parses_only_ready_devices(monkeypatch):
    run = FakeRun(_ok(stdout=b"""List of devices attached
pixel-1\tdevice
pixel-2\toffline
pixel-3\tunauthorized
pixel-4\tdevice product:foo model:bar

"""))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    assert Adb(serial="ignored").devices() == ["pixel-1", "pixel-4"]
    assert run.argv == [["adb", "devices"]]


# --- parse_devices_output: the ONE canonical `adb devices` parser (X8) --------------------
# adb devices was previously parsed three different, disagreeing ways across the codebase
# (drivers/adb.py, supervisor.py, tools/hinge_inspect.py); this is the single source of truth
# the other two are being rewired to import.
def test_parse_devices_output_handles_dash_l_extra_columns():
    # `adb devices -l` appends trailing key:value columns. A naive `parts[-1] == "device"`
    # check (used by the old ad-hoc parsers) is flat-out wrong on this output.
    out = ("List of devices attached\n"
           "0A051FDD4003ZR       device usb:1-1 product:panther model:Pixel_7a "
           "device:panther transport_id:3\n")
    assert parse_devices_output(out) == ["0A051FDD4003ZR"]


def test_parse_devices_output_excludes_unauthorized_and_offline():
    out = ("List of devices attached\n"
           "pixel-1\tunauthorized\n"
           "pixel-2\toffline\n"
           "pixel-3\tdevice\n")
    assert parse_devices_output(out) == ["pixel-3"]


def test_parse_devices_output_skips_daemon_startup_chatter():
    # A cold `adb devices` call prints daemon-startup lines to stdout BEFORE the header.
    out = ("* daemon not running; starting now at tcp:5037\n"
           "* daemon started successfully\n"
           "List of devices attached\n"
           "pixel-1\tdevice\n"
           "\n")
    assert parse_devices_output(out) == ["pixel-1"]


def test_parse_devices_output_empty_when_no_devices():
    assert parse_devices_output("List of devices attached\n\n") == []


# --- scroll_up: x-column jitter, shared by Adb and UhidTouch (HINGE-04) -------------------
def test_scroll_up_jitters_x_column(monkeypatch):
    # UHID (uhid.py) previously had NO x jitter on scroll_up, unlike this one -- a repeated,
    # pixel-identical swipe column is exactly the machine-like signature the humanized path
    # exists to avoid. Both transports now share adb.scroll_x(); this pins Adb's side of it.
    run = FakeRun(*[_ok() for _ in range(30)])
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    d = Adb(serial="pixel")
    d._size = (1080, 2400)

    xs = set()
    for _ in range(30):
        d.scroll_up()
        dx = int(re.search(r"DOWN (\d+) \d+", run.inputs[-1].decode()).group(1))
        xs.add(dx)
    assert len(xs) > 1     # jittered run to run, not a fixed column every time


def test_non_zero_exit_raises_adb_error(monkeypatch):
    run = FakeRun(_fail(stderr=b"bad command"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb().text("HOME")

    assert exc.value.argv == ["adb", "shell", "input", "text", "HOME"]
    assert exc.value.stderr == "bad command"
    assert "ADB command failed with exit code 1" in str(exc.value)
    assert "stderr: bad command" in str(exc.value)


def test_timeout_raises_adb_error(monkeypatch):
    timeout = subprocess.TimeoutExpired(
        ["adb", "shell", "input", "text", "HOME"],
        timeout=10,
        stderr=b"still waiting",
    )
    run = FakeRun(timeout)
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb(default_timeout=10).text("HOME")

    assert exc.value.argv == ["adb", "shell", "input", "text", "HOME"]
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
        Adb(serial="pixel").text("HOME")

    assert run.argv == [["adb", "-s", "pixel", "shell", "input", "text", "HOME"]]


def test_device_lost_via_stdout_raises_driver_closed(monkeypatch):
    # Detection folds in stdout too (some adb builds emit the message there, not
    # on stderr). Guards against a regression that only inspected stderr.
    run = FakeRun(_fail(stderr=b"", stdout=b"error: device offline"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(DriverClosed):
        Adb(serial="pixel").text("HOME")

    assert run.argv == [["adb", "-s", "pixel", "shell", "input", "text", "HOME"]]


def test_write_file_cat_redirect_argv_and_raw_bytes(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").write_file("/data/local/tmp/g.json", b"\x00\x01")

    assert run.argv == [["adb", "-s", "pixel", "shell", "cat > /data/local/tmp/g.json"]]
    assert run.inputs == [b"\x00\x01"]      # raw bytes piped via stdin, untouched


def test_write_file_quotes_path_with_spaces(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").write_file("/data/local/tmp/has space.json", b"x")

    assert run.argv[0][4] == "cat > '/data/local/tmp/has space.json'"     # shlex.quote


def test_write_file_device_lost_raises_driver_closed(monkeypatch):
    run = FakeRun(_fail(stderr=b"error: device offline"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(DriverClosed):
        Adb(serial="pixel").write_file("/data/local/tmp/g.json", b"x")
