import re
import subprocess

import pytest

from operation_love.drivers import adb as adb_mod
from operation_love.drivers.adb import (
    Adb,
    AdbError,
    _clean_text_for_input,
    parse_devices_output,
    parse_foreground_package,
    plan_path,
    quote_android_package_id,
    validate_android_package_id,
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

    def __call__(self, argv, *, capture_output=True, timeout=None, input=None, check=None):
        self.calls.append((list(argv), capture_output, timeout, input, check))
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
    assert all(c[4] is False for c in run.calls)    # return code is inspected by Adb._run


def test_keyevent_uses_a_closed_integer_android_input_argv(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel-7a", adb_path="/opt/android/adb").keyevent(4)

    assert run.argv == [["/opt/android/adb", "-s", "pixel-7a", "shell", "input",
                        "keyevent", "4"]]


def test_host_side_adb_launch_error_uses_driver_error_contract(monkeypatch):
    run = FakeRun(PermissionError("permission denied"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError, match="could not start ADB binary") as exc:
        Adb(adb_path="/blocked/adb").devices()

    assert "/blocked/adb devices" in str(exc.value)


@pytest.mark.parametrize("keycode", [None, True, -1, 65536, "4"])
def test_keyevent_rejects_non_android_keycodes_without_starting_adb(monkeypatch, keycode):
    run = FakeRun()
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(ValueError, match="Android keycode"):
        Adb(serial="pixel-7a").keyevent(keycode)

    assert run.calls == []


@pytest.mark.parametrize(("dump", "expected"), [
    ("mCurrentFocus=Window{123 u0 com.android.systemui/.shade.NotificationShadeWindowView}\n",
     "com.android.systemui"),
    ("mFocusedApp=ActivityRecord{abc u0 co.hinge.app/.MainActivity t42}\n", "co.hinge.app"),
    ("Window #0 unrelated.example/.Elsewhere\n", None),
])
def test_parse_foreground_package_accepts_only_labeled_focus_lines(dump, expected):
    assert parse_foreground_package(dump) == expected


@pytest.mark.parametrize("current", [
    "mCurrentFocus=null",
    "mCurrentFocus=Window{123 u0 NotificationShade}",
])
def test_present_but_unreadable_current_focus_never_falls_back_to_stale_activity(current):
    dump = f"{current}\nmFocusedApp=ActivityRecord{{abc u0 co.hinge.app/.MainActivity t42}}\n"

    assert parse_foreground_package(dump) is None


def test_foreground_parser_accepts_fully_qualified_component_class():
    dump = "mCurrentFocus=Window{123 u0 co.hinge.app/co.hinge.app.ui.MainActivity}\n"

    assert parse_foreground_package(dump) == "co.hinge.app"


@pytest.mark.parametrize(
    "value", [None, 7, "hinge", ".co.hinge", "co..hinge", "co.hinge-app", "co.hinge; id"])
def test_android_package_validation_rejects_non_identifiers_before_shell_use(value):
    with pytest.raises(ValueError, match="dotted Android identifier"):
        validate_android_package_id(value)


def test_android_package_shell_token_is_validated_and_quoted():
    assert validate_android_package_id("co.hinge_app.v10") == "co.hinge_app.v10"
    assert quote_android_package_id("co.hinge_app.v10") == "co.hinge_app.v10"


def test_conflicting_current_focus_lines_are_ambiguous_even_if_one_is_hinge():
    dump = "\n".join([
        "mCurrentFocus=Window{123 u0 co.hinge.app/.MainActivity}",
        "mCurrentFocus=Window{456 u0 com.android.systemui/.StatusBar}",
        "mFocusedApp=ActivityRecord{abc u0 co.hinge.app/.MainActivity t42}",
    ])

    assert parse_foreground_package(dump) is None


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


def test_clean_text_no_longer_drops_an_unmapped_codepoint():
    # WYSIWYG regression guard: _clean_text_for_input used to silently DELETE any codepoint
    # it didn't recognise (an emoji, here) -- the exact bug this module was rewritten to fix
    # (see typography.fold_to_ascii's docstring). It must now pass an unmapped character
    # through UNCHANGED; turning that into a loud failure is Adb.text()'s job now (see
    # test_text_raises_adb_error_and_sends_nothing_for_an_undeliverable_emoji below), not
    # this function's.
    out = _clean_text_for_input("choice \U0001F600 love")
    assert out == "choice \U0001F600 love"


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


# --- Adb.text: fail loudly on an undeliverable character, never drop it silently ---------
# The bug this whole feature exists to fix: _clean_text_for_input's old silent-drop behaviour
# meant the opener recorded in BigQuery / shown in the hub could differ from what actually
# got typed on the device. Owner hard rule: best humanized interaction or FAIL LOUDLY.
def test_text_raises_adb_error_and_sends_nothing_for_an_undeliverable_emoji(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb(serial="pixel").text("great smile \U0001F600 love it")

    assert run.calls == []                            # NO adb command issued at all
    msg = str(exc.value)
    assert "\\U0001f600" in msg                        # named codepoint escape
    assert "GRINNING FACE" in msg                      # unicodedata.name
    assert "not typed" in msg.lower() or "not been typed" in msg.lower() \
        or "was not typed" in msg.lower()


def test_text_raises_adb_error_for_cjk_text(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb(serial="pixel").text("你好")

    assert run.calls == []
    assert "\\u4f60" in str(exc.value)                 # first CJK codepoint named


def test_text_types_accented_text_as_its_ascii_fold(monkeypatch):
    # The headline WYSIWYG behaviour: an accented name is not undeliverable -- it has a
    # perfectly good ASCII substitute (typography.fold_to_ascii's NFKD pass), so it types
    # cleanly rather than raising.
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").text("Your trip to São Paulo, Chloé")

    assert run.argv == [
        ["adb", "-s", "pixel", "shell", "input", "text", "Your%strip%sto%sSao%sPaulo,%sChloe"],
    ]


# --- Android's real sendText() unescaping, ported for tests --------------------------------
# The owner rejected the old blanket "% -> percent" rewrite in typography.fold_to_ascii as
# unnatural ("50%" must type as "50%"). What actually collides with adb's %s space escape is
# narrower: a literal '%' immediately followed by a lowercase 's' (see
# typography.fold_to_ascii's docstring point 4 and typography.undeliverable_sequences'
# docstring for the measured table this module reproduces). These two helpers are a faithful
# port of Android's `Input.java` / `InputShellCommand` `sendText()` unescaping and of the
# on-device shell's backslash-unescaping, so the round-trip test below pins the ACTUAL device
# semantics rather than our assumption about them -- this is the highest-value test in this
# file precisely because it is the thing a live on-device check (LIVE-VERIFY, device
# currently disconnected) would confirm, not a guess about Android's internals.
def shell_strip(escaped: str) -> str:
    """Undo _escape_input_text's backslash-escaping of shell metacharacters -- what the
    on-device shell does when it parses the `adb shell input text <escaped>` command line
    before invoking the `input` binary: a backslash immediately preceding one of
    adb._TEXT_SHELL_SPECIALS is consumed and only the literal character remains. Space
    encoding (%s) is left untouched here -- decoding that is Android's own sendText() job,
    handled by android_unescape below, not the on-device shell's.
    """
    out: list[str] = []
    i, n = 0, len(escaped)
    while i < n:
        ch = escaped[i]
        if ch == "\\" and i + 1 < n and escaped[i + 1] in adb_mod._TEXT_SHELL_SPECIALS:
            out.append(escaped[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def android_unescape(s: str) -> str:
    """A faithful port of Android's `Input.java` / `InputShellCommand` `sendText()`
    unescaping: a literal '%' arms an escape flag; on the very NEXT character, a lowercase
    's' becomes a space and the '%' is deleted (consumed together); any other character
    instead flushes the armed '%' back out unconsumed, and that same character is then
    re-examined as a possible NEW escape-starter in its own right (this is what makes
    'a%%b' round-trip unchanged: the first '%' arms, the second '%' doesn't match 's' so the
    first '%' flushes -- but the second '%' immediately re-arms for 'b', which also doesn't
    match, so it flushes too, leaving both '%' intact). A '%' armed at the very end of the
    string with no following character flushes literally.

    This is the thing a LIVE on-device check would confirm (see typography.fold_to_ascii's
    docstring point 4 and typography.undeliverable_sequences' docstring -- both flagged
    LIVE-VERIFY, not yet confirmed on the physical Pixel 7a).
    """
    result: list[str] = []
    escape = False
    for ch in s:
        if escape:
            escape = False
            if ch == "s":
                result.append(" ")
                continue
            result.append("%")
        if ch == "%":
            escape = True
        else:
            result.append(ch)
    if escape:
        result.append("%")
    return "".join(result)


@pytest.mark.parametrize("s", [
    "50%", "50% off", "up 30% today", "50%.", "a%%b", "50%S",
    "hi there&you", "plain text", "no percent here at all",
])
def test_escape_input_text_round_trips_through_android_sendtext_for_safe_strings(s):
    assert android_unescape(shell_strip(adb_mod._escape_input_text(s))) == s


# --- % is left ALONE by fold_to_ascii; only the narrow %+lowercase-s collision is rejected --
@pytest.mark.parametrize("s", ["50%", "50% off", "up 30% today", "50%."])
def test_text_percent_survives_unrewritten(monkeypatch, s):
    # Owner decision: "50%" must type as "50%", never "50 percent". Confirms both that the
    # literal '%' is still present in what's actually sent (no silent rewording back to the
    # word "percent"), and -- via the round-trip helpers above -- that it reaches the device
    # exactly as written.
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").text(s)

    sent = run.argv[0][6]
    assert "%" in sent
    assert "percent" not in sent
    assert android_unescape(shell_strip(sent)) == s


def test_text_uppercase_percent_s_is_safe_and_survives(monkeypatch):
    # Android's sendText() comparison is against lowercase 's' only -- '%S' does NOT collide.
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    Adb(serial="pixel").text("50%S accuracy")

    sent = run.argv[0][6]
    assert android_unescape(shell_strip(sent)) == "50%S accuracy"


@pytest.mark.parametrize("s", ["100%sure thing", "20%stake in it", "%s", "hey 50%s off"])
def test_text_percent_lowercase_s_collision_raises_and_sends_nothing(monkeypatch, s):
    # The one real collision: a literal '%' directly against a following lowercase 's' is
    # undeliverable as written (it silently becomes a space and eats both characters on the
    # real device). Fail loudly instead of silently mistyping -- nothing is sent at all.
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    with pytest.raises(AdbError) as exc:
        Adb(serial="pixel").text(s)

    assert run.calls == []                             # NO adb command issued at all
    msg = str(exc.value)
    assert "'%s'" in msg
    assert "not typed" in msg.lower()


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


# --- per-gesture timing ledger (2026-08-23, the ADB fallback's own -- see uhid.py's for the
# genuine-transport version this mirrors, and hinge.py's _scroll/_swipe, the only callers) ----
def test_swipe_records_the_named_timing_buckets_when_given_a_stamps_dict(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    stamps: dict[str, float] = {}

    Adb(serial="pixel").swipe(540, 1800, 540, 700, duration_ms=500, _timing=stamps)

    # Unlike UHID's three device round trips, this transport pipes the whole scripted gesture
    # (embedded `sleep`s included) to ONE `adb shell` call -- see Adb.swipe's own docstring.
    assert set(stamps) == {"adb_plan_path_s", "adb_script_build_s", "adb_run_script_s"}
    assert all(v >= 0.0 for v in stamps.values())
    assert run.argv == [["adb", "-s", "pixel", "shell"]]     # still exactly one device call


def test_scroll_up_adds_its_own_screen_size_bucket_on_top_of_swipe(monkeypatch):
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    d = Adb(serial="pixel")
    d._size = (1080, 2400)          # cached: screen_size_s should still appear (this IS the call)
    stamps: dict[str, float] = {}

    d.scroll_up(_timing=stamps)

    assert set(stamps) == {"screen_size_s", "adb_plan_path_s", "adb_script_build_s",
                          "adb_run_script_s"}


def test_swipe_with_no_timing_dict_is_a_true_noop(monkeypatch):
    """The default -- every caller before 2026-08-23 -- costs nothing extra: `_time_bucket`
    (base.py, shared with uhid.py and hinge.py) makes `stamps is None` a true no-op, not even a
    `time.monotonic()` call -- see its own docstring. `time` is a process-wide singleton module,
    so patching it here catches a stray monotonic() call from ANY of the three modules."""
    import time as time_mod
    run = FakeRun(_ok())
    monkeypatch.setattr(adb_mod.subprocess, "run", run)
    calls = []
    monkeypatch.setattr(time_mod, "monotonic", lambda: calls.append(1) or 0.0)

    Adb(serial="pixel").swipe(540, 1800, 540, 700, duration_ms=500)

    assert calls == []
    assert run.argv == [["adb", "-s", "pixel", "shell"]]     # unchanged: one device round trip


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


def test_screen_size_parses_case_and_spacing_variants(monkeypatch):
    # `wm` output is a shell-facing diagnostic, not a stable serialization.  The previous
    # parser noticed case-insensitively, then split the original line case-sensitively and
    # rejected a valid `Physical Size` spelling.
    run = FakeRun(_ok(stdout=b"Physical Size : 1080 x 2400\nOVERRIDE SIZE: 1080 x 2160\n"))
    monkeypatch.setattr(adb_mod.subprocess, "run", run)

    assert Adb(serial="pixel").screen_size() == (1080, 2160)


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
