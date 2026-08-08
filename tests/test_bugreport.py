"""bugreport — redacted markdown diagnostic. Offline."""
import io
import os
import threading

from operation_love import bugreport


def _ring_bodies():
    return [line.split(" ", 1)[1] for line in bugreport._LOG_RING]


def test_report_has_core_sections():
    md = bugreport.build_report(None, description="it broke")
    for h in ["# Operation Love — Bug Report", "## What happened", "it broke",
              "## Build", "## System", "## Dependencies", "## Config",
              "## Secrets", "## Diagnostic improvement", "## Run status",
              "## Debug log (on-disk actions + screenshots)", "## Recent logs"]:
        assert h in md, f"missing section: {h}"
    assert "improve `operation_love/bugreport.py`" in md


def test_secrets_are_redacted():
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-SECRETVALUE123"
    try:
        md = bugreport.build_report(None)
        assert "SECRETVALUE123" not in md          # raw key never leaks
        assert "sk-ant-" in md and "present" in md  # presence + short prefix only
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_unset_key_shows_unset():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    assert "ANTHROPIC_API_KEY: unset" in bugreport.build_report(None)


def test_log_capture_roundtrip():
    bugreport.install_log_capture()
    print("OPLOVE_TEST_LOGLINE_marker")
    assert any("OPLOVE_TEST_LOGLINE_marker" in line for line in bugreport.recent_logs())
    assert "OPLOVE_TEST_LOGLINE_marker" in bugreport.build_report(None)


def test_tee_keeps_multi_arg_print_as_one_line():
    bugreport._LOG_RING.clear()
    stream = io.StringIO()
    tee = bugreport._Tee(stream)

    print("find model:", "/path/x.onnx", "landmark_3d_68", ["None", 3, 192, 192],
          0.0, 1.0, file=tee)

    expected = "find model: /path/x.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0"
    assert stream.getvalue() == expected + "\n"
    assert _ring_bodies() == [expected]


def test_tee_emits_each_complete_multiline_write():
    bugreport._LOG_RING.clear()
    tee = bugreport._Tee(io.StringIO())

    tee.write("alpha\n\n beta \ngamma\n")

    assert _ring_bodies() == ["alpha", " beta ", "gamma"]


def test_tee_buffers_partial_until_newline():
    bugreport._LOG_RING.clear()
    tee = bugreport._Tee(io.StringIO())

    tee.write("partial")
    assert _ring_bodies() == []
    tee.flush()
    assert _ring_bodies() == []
    tee.write(" line\n")

    assert _ring_bodies() == ["partial line"]


def test_tee_concurrent_writes_stay_whole():
    bugreport._LOG_RING.clear()
    tee = bugreport._Tee(io.StringIO())
    threads = []
    per_thread = 40

    def write_lines(tid):
        for i in range(per_thread):
            tee.write(f"T{tid}-")
            tee.write(f"{i}\n")

    for tid in range(5):
        thread = threading.Thread(target=write_lines, args=(tid,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    bodies = _ring_bodies()
    expected = {f"T{tid}-{i}" for tid in range(5) for i in range(per_thread)}
    assert len(bodies) == len(expected)
    assert set(bodies) == expected


class _FakeHub:
    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "live", "mode": "observe", "running": True, "labels": 12, "min_labels": 40,
            "ranker_ready": False, "labels_needed": 28, "budget_spent": 0.5, "budget_cap": 5.0,
            "openers": 1,
            "apps": {"bumble": {"app": "bumble", "mode": "observe", "state": "waiting",
                                "last_decision": "like", "last_score": 0.0, "swipes_run": 12}}}}


def test_status_section_renders_apps():
    md = bugreport.build_report(_FakeHub())
    assert "phase: live" in md
    assert "labels: 12 / 40 (defer)" in md
    assert "| bumble |" in md


def test_line_cap_enforced_by_dropping_oldest_logs():
    bugreport._LOG_RING.clear()
    for i in range(bugreport._MAX_REPORT_LINES + 200):
        bugreport._LOG_RING.append(f"OPLOVE_LOG_{i:05d}")
    oldest_captured = bugreport._LOG_RING[0]
    newest_captured = bugreport._LOG_RING[-1]

    md = bugreport.build_report(None)
    assert len(md.splitlines()) <= bugreport._MAX_REPORT_LINES
    assert oldest_captured not in md
    assert newest_captured in md
    assert "older log line(s) omitted" in md


def test_line_cap_handles_oversized_description():
    bugreport._LOG_RING.clear()
    desc = "\n".join(f"OPLOVE_DESC_{i:05d}" for i in range(bugreport._MAX_REPORT_LINES + 200))

    md = bugreport.build_report(None, description=desc)
    assert len(md.splitlines()) <= bugreport._MAX_REPORT_LINES
    assert "OPLOVE_DESC_00000" not in md
    assert "OPLOVE_DESC_50199" in md
    assert "older report line(s) omitted" in md


def test_debug_log_section_tails_actions_and_flags_error_shots(tmp_path):
    """The on-disk debug log (Hinge's silent auto-mode logging) is surfaced: latest run folder,
    screenshot count, the kept error shot, and the tail of actions.jsonl — so a report points a
    developer straight at the failure even though the screenshots themselves are binary."""
    run = tmp_path / "run_20260627_120000"
    run.mkdir(parents=True)
    (run / "00001_capture_before.png").write_bytes(b"x")
    (run / "00002_like_before.png").write_bytes(b"x")
    (run / "00003_unexpected_error.png").write_bytes(b"x")        # kept-forever failure shot
    (run / "actions.jsonl").write_text(
        '{"action": "capture", "photos": 6}\n'
        '{"action": "like", "heart": [922, 1617]}\n'
        '{"action": "unexpected", "error": "HingeActionError: like did not land"}\n')

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert "run_20260627_120000" in md and "latest run" in md
    assert "screenshots: 3" in md
    assert "error shots (kept): 00003_unexpected_error.png" in md
    assert "like did not land" in md                              # actions.jsonl tail inlined


def test_debug_log_section_handles_missing_dir():
    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": "/no/such/oplove/debug/dir"})
    assert "hinge" in md and "no folder yet" in md                # graceful, no raise


class _FakeHubError:
    def snapshot(self):
        return {"running": False, "error": "RuntimeError: boom", "status": None}


def test_status_section_renders_error_and_no_run():
    md = bugreport.build_report(_FakeHubError())
    assert "last run error" in md
    assert "boom" in md
    assert "no active/last run" in md


def test_omitted_log_count_is_exact():
    import re

    bugreport._LOG_RING.clear()
    for i in range(20):
        bugreport._LOG_RING.append(f"L{i:02d}")
    try:
        md = bugreport._logs_md(7)
    finally:
        bugreport._LOG_RING.clear()

    m = re.search(r"\.\.\. (\d+) older log line\(s\) omitted", md)
    assert m is not None
    assert int(m.group(1)) == 16                 # locks the arithmetic (old undercount said 15)

    shown = [ln for ln in md.splitlines() if re.fullmatch(r"L\d\d", ln)]
    assert shown == ["L16", "L17", "L18", "L19"]  # exactly the 4 newest real lines
    assert len(md.splitlines()) <= 7
