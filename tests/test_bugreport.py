"""bugreport — redacted markdown diagnostic. Offline."""
import os

from operation_love import bugreport


def test_report_has_core_sections():
    md = bugreport.build_report(None, description="it broke")
    for h in ["# Operation Love — Bug Report", "## What happened", "it broke",
              "## Build", "## System", "## Dependencies", "## Config",
              "## Secrets", "## Run status", "## Recent logs"]:
        assert h in md, f"missing section: {h}"


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


def test_char_cap_enforced():
    for _ in range(3000):
        bugreport._LOG_RING.append("x" * 80)
    md = bugreport.build_report(None)
    assert len(md) <= bugreport._MAX_CHARS


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
