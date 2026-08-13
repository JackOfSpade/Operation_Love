"""bugreport — redacted markdown diagnostic. Offline."""
import datetime
import io
import json
import os
import re
import sys
import threading
import types

from operation_love import bugreport, config as oplove_config
from operation_love.drivers import touchwatch


def _ring_bodies():
    return [line.split(" ", 1)[1] for line in bugreport._LOG_RING]


def test_report_has_core_sections():
    md = bugreport.build_report(None, description="it broke")
    for h in ["# Operation Love — Bug Report", "## What happened", "it broke",
              "## Build", "## System", "## Dependencies", "## Capabilities", "## Config",
              "## Secrets", "## Diagnostic improvement", "## Run status", "## Recent openers",
              "## Recent opener rejections",
              "## Debug log (on-disk actions + screenshots)", "## Recent logs"]:
        assert h in md, f"missing section: {h}"
    assert "improve `operation_love/bugreport.py`" in md


def test_unset_key_shows_unset():
    # Gemini is the only opener provider (the Anthropic/Claude path -- and its
    # ANTHROPIC_API_KEY secrets-section entry -- was removed entirely), so GEMINI_API_KEY
    # is the only opener credential the secrets section can report on.
    os.environ.pop("GEMINI_API_KEY", None)
    assert "GEMINI_API_KEY: unset" in bugreport.build_report(None)


def test_gemini_key_is_redacted():
    os.environ["GEMINI_API_KEY"] = "AIzaGeminiSecretValue"
    try:
        md = bugreport.build_report(None)
        assert "GeminiSecretValue" not in md
        assert "GEMINI_API_KEY: present" in md
    finally:
        del os.environ["GEMINI_API_KEY"]


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
    assert "| app | mode | state | last | score | decisions |" in md
    assert "| app | mode | state | last | score | swipes |" not in md
    assert "| bumble |" in md


def test_status_section_healthy_run_has_no_diagnostics_section():
    # The common case: no app has a stop_reason or error. The "Stop reasons / errors" block
    # must not appear at all -- not as an empty heading, not as a stray blank bullet list.
    md = bugreport.build_report(_FakeHub())
    assert "Stop reasons / errors" not in md


class _StoppedHub:
    """A hinge auto run halted by opener retry-exhaustion -- state alone ("stopped") can't
    tell the owner WHY, which is exactly the gap the audit found: stop_reason carried the
    explanation but _status_md never rendered it."""
    STOP_REASON = ("5 consecutive AI opener attempts for this profile were all rejected as "
                   "unusable; stopping so nothing swipes blind")

    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "stopped", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                                "last_decision": None, "last_score": None, "swipes_run": 12,
                                "error": None, "stop_reason": _StoppedHub.STOP_REASON}}}}


def test_status_section_renders_stop_reason_verbatim():
    md = bugreport.build_report(_StoppedHub())
    assert _StoppedHub.STOP_REASON in md
    assert "**hinge** stop reason:" in md


class _ErroredHub:
    """HALT-on-unexpected: worker.py publishes the last traceback line to AppStatus.error."""
    ERROR = "HingeActionError: like did not land after 3 retries"

    def snapshot(self):
        return {"running": False, "error": None, "status": {
            "phase": "stopped", "mode": "auto", "running": False, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "error",
                                "last_decision": None, "last_score": None, "swipes_run": 12,
                                "error": _ErroredHub.ERROR, "stop_reason": None}}}}


def test_status_section_renders_per_app_error_verbatim():
    md = bugreport.build_report(_ErroredHub())
    assert _ErroredHub.ERROR in md
    assert "**hinge** error:" in md
    # This app has no stop_reason -- only the error bullet must appear for it.
    assert "**hinge** stop reason:" not in md


class _NoReasonAppHub:
    """One app is clean, the other has a stop_reason. The clean app must not get a bullet --
    "never print a reason for an app that doesn't have one"."""
    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "live", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            "apps": {
                "hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                          "last_decision": None, "last_score": None, "swipes_run": 12,
                          "error": None, "stop_reason": "every configured Gemini model has "
                          "exhausted its per-day free-tier quota"},
                "bumble": {"app": "bumble", "mode": "observe", "state": "waiting",
                           "last_decision": "like", "last_score": 0.5, "swipes_run": 3,
                           "error": None, "stop_reason": None},
            }}}


def test_status_section_omits_bullet_for_apps_without_a_reason():
    md = bugreport.build_report(_NoReasonAppHub())
    assert "**hinge** stop reason:" in md
    assert "**bumble** stop reason:" not in md
    assert "**bumble** error:" not in md


class _MalformedReasonHub:
    """stop_reason is provider-sourced free text, not something we control -- it can contain
    table-breaking `|`, embedded newlines that look like new markdown lines, and backticks
    that could escape the inline-code span it's rendered inside. None of that may corrupt the
    report's structure (sections after it must still parse, the app table above it must keep
    exactly 6 columns per data row)."""
    REASON = "quota exceeded | model=`gemini-2.0-flash`\n# fake heading\n| injected | row |"

    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "stopped", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                                "last_decision": None, "last_score": None, "swipes_run": 12,
                                "error": None, "stop_reason": _MalformedReasonHub.REASON}}}}


def test_status_section_stop_reason_with_pipes_and_newlines_does_not_corrupt_report():
    md = bugreport.build_report(_MalformedReasonHub())
    # The report must still parse into the normal downstream sections.
    assert "## Debug log (on-disk actions + screenshots)" in md
    assert "## Recent logs" in md
    # The app table above the diagnostics block keeps its real 6-column row intact.
    assert "| hinge | auto | stopped | — | — | 12 |" in md
    # No literal newline was injected into the report as a bare markdown line/heading: the
    # raw multi-line reason must not appear anywhere unsanitized...
    assert _MalformedReasonHub.REASON not in md
    # ...and in particular the embedded "# fake heading" line was collapsed into prose, not
    # left to stand alone at the start of a line where a markdown renderer would treat it as
    # a real heading.
    assert not any(ln.strip().startswith("# fake heading") for ln in md.splitlines())
    # ...but the meaningful content survived, collapsed onto one line inside a code span.
    assert "quota exceeded | model='gemini-2.0-flash' # fake heading | injected | row |" in md


# ── stop_kind: disambiguating stop_reason's SOURCE (2026-08-11, deck-blocked addition) ──────
# stop_reason used to have exactly one cause (OpenerService exhaustion). worker.py's blocked-
# deck check (Hinge's out-of-free-likes Hinge+ paywall)
# gave it a second, with its own stop_kind -- see status.py's AppStatus.stop_kind and
# _app_diagnostics_md's own docstring for the full "which subsystem?" reasoning this rendering
# exists to remove.
class _DeckBlockedHub:
    """A Hinge observe run halted by worker.py's blocked-deck check: the driver recognised
    Hinge's out-of-free-likes paywall standing between us and the deck. Pins that the bullet
    names BOTH the reason and its kind, not just the reason alone (as it did before stop_kind
    existed -- see test_status_section_renders_stop_reason_verbatim above)."""
    REASON = "Hinge is out of free likes for today — the Hinge+ upgrade screen is up"

    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "stopped", "mode": "observe", "running": True, "labels": 12, "min_labels": 40,
            "ranker_ready": False, "labels_needed": 28, "budget_spent": 0.0, "budget_cap": None,
            "openers": 0,
            "apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "blocked",
                                "last_decision": None, "last_score": None, "swipes_run": 3,
                                "error": None, "stop_reason": _DeckBlockedHub.REASON,
                                "stop_kind": "deck_blocked"}}}}


def test_status_section_renders_stop_kind_alongside_stop_reason():
    md = bugreport.build_report(_DeckBlockedHub())
    assert _DeckBlockedHub.REASON in md
    assert "**hinge** stop reason:" in md
    assert "(kind: `deck_blocked`)" in md


class _StopReasonWithoutKindHub:
    """A stop_reason from a code path that has no stop_kind of its own (an opener-exhaustion
    stop predating 2026-08-11, or any future one that simply forgets to set it) must render
    honestly as "unlabelled" rather than silently dropping the bracket -- omitting it here
    would just move the "which subsystem caused this?" guesswork one field over, exactly the
    ambiguity stop_kind exists to remove (see _app_diagnostics_md's docstring)."""
    REASON = "every configured Gemini model has exhausted its per-day free-tier quota"

    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "stopped", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            # stop_kind deliberately absent from this dict -- not merely None -- to also pin
            # the .get("stop_kind") lookup itself, not just a falsy value stored under the key.
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                                "last_decision": None, "last_score": None, "swipes_run": 12,
                                "error": None,
                                "stop_reason": _StopReasonWithoutKindHub.REASON}}}}


def test_status_section_missing_stop_kind_renders_as_unlabelled():
    md = bugreport.build_report(_StopReasonWithoutKindHub())
    assert _StopReasonWithoutKindHub.REASON in md
    assert "(kind: unlabelled)" in md


class _MalformedStopKindHub:
    """stop_kind is presently a fixed small vocabulary ("opener" / "deck_blocked" /
    "targeting" -- ops/OPENER-REDESIGN.md 5.6's hard stop), not
    provider-sourced text -- but _app_diagnostics_md routes it through _sanitize_inline anyway
    (see that function's docstring for why), and this pins that decision: embedded newlines,
    pipes, and backticks in EITHER field must not corrupt the report structure or escape the
    inline-code span they're rendered inside, the same guarantee
    test_status_section_stop_reason_with_pipes_and_newlines_does_not_corrupt_report already
    pins for stop_reason alone."""
    REASON = "quota exceeded | model=`gemini-2.0-flash`\n# fake heading"
    KIND = "deck_blocked`\n# also fake"

    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "stopped", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                                "last_decision": None, "last_score": None, "swipes_run": 12,
                                "error": None, "stop_reason": _MalformedStopKindHub.REASON,
                                "stop_kind": _MalformedStopKindHub.KIND}}}}


def test_status_section_stop_kind_with_pipes_newlines_backticks_does_not_corrupt_report():
    md = bugreport.build_report(_MalformedStopKindHub())
    # The report must still parse into the normal downstream sections.
    assert "## Debug log (on-disk actions + screenshots)" in md
    assert "## Recent logs" in md
    # Neither raw, unsanitized field appears anywhere...
    assert _MalformedStopKindHub.REASON not in md
    assert _MalformedStopKindHub.KIND not in md
    # ...and in particular neither embedded "# fake heading"/"# also fake" line was left to
    # stand alone at the start of a line where a markdown renderer would treat it as real.
    assert not any(ln.strip().startswith("# fake heading") for ln in md.splitlines())
    assert not any(ln.strip().startswith("# also fake") for ln in md.splitlines())
    # ...but the meaningful content survived, sanitized onto one line inside its own code span.
    assert "(kind: `deck_blocked' # also fake`)" in md


class _MalformedHub:
    """snapshot() with a status dict missing keys _status_md assumes exist — this must
    not crash the whole report (Bug H): the section renders a warning and every other
    section (build/system/deps/...) still comes through."""
    def snapshot(self):
        return {"running": True, "error": None, "status": {"phase": "live"}}   # no "labels" etc.


def test_malformed_status_section_does_not_crash_whole_report():
    md = bugreport.build_report(_MalformedHub())
    assert "## Run status" in md
    assert "⚠️" in md                     # the broken section renders a warning...
    assert "## Build" in md               # ...but every other section still renders
    assert "## System" in md
    assert "## Recent logs" in md


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


def test_debug_log_section_summarises_item_index_refusals_and_realised_steps(tmp_path):
    """A long capture's failure must answer both questions that its first-frame-only capture
    record cannot: which adjacent frames broke, and whether the realised step was otherwise
    stable.  Unknown pair deltas stay absent from min/median/max rather than becoming zero.
    """
    run = tmp_path / "run_item_index_refused"
    run.mkdir(parents=True)
    refusal = {
        "action": "item_index_refused", "reason": "no trustworthy shift",
        "failing_pair": [34, 35], "before": "00001_item_index_refused_before.png",
        "after": "00002_item_index_refused_after.png",
        "steps_px": [209, None, 211, 209],
        "refused_pairs": [{"pair": [34, 35], "status": "no_consensus", "agreeing": 4,
                            "dissenting": 0, "eligible": 4}],
    }
    (run / "actions.jsonl").write_text(json.dumps(refusal) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "item-index refusals and realised-step stats:" in md
    assert "frames 34 and 35" in md and "no trustworthy shift" in md
    assert "realised steps (3 measured): min 209px, median 209px, max 211px" in md
    assert "before `00001_item_index_refused_before.png`" in md
    assert "after `00002_item_index_refused_after.png`" in md


def test_debug_log_capture_split_summary_links_boundary_evidence_and_recovery(tmp_path):
    """A split report answers both halves of the incident: what triggered it, and did the
    worker's next capture get back to a usable profile rather than silently falling back to a
    truncated, unnumbered read?  The current logger stores the source/foreign frames as
    before/after; those filenames must be surfaced without making a developer decode the tail.
    """
    run = tmp_path / "run_split_recovery"
    run.mkdir(parents=True)
    split = {
        "ts": "2026-08-13T00:20:55", "action": "capture_split", "photos": 2,
        "profile_name": "Ada", "before": "00001_capture_split_before.png",
        "after": "00002_capture_split_after.png", "identity_dist": 17.95,
        "top_dist": 15.73,
    }
    recovery = {
        "ts": "2026-08-13T00:21:54", "action": "capture", "photos": 12,
        "capture_truncated": True, "items": 0,
        "items_unavailable": "the card is not confirmed to be at its scroll top",
    }
    (run / "actions.jsonl").write_text(json.dumps(split) + "\n" + json.dumps(recovery) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "capture-split recovery:" in md
    assert "deck advanced mid-read after 2 captured frame(s)" in md
    assert "source screenshot `00001_capture_split_before.png`" in md
    assert "boundary-trigger screenshot `00002_capture_split_after.png`" in md
    assert "identity distance 17.95, scroll-top distance 15.73" in md
    assert "identity read `Ada`" in md
    assert "later capture/recovery followed: 12 photo(s); capture truncated; 0 numbered item(s)" in md
    assert "items unavailable: `the card is not confirmed to be at its scroll top`" in md


def test_debug_log_capture_split_summary_says_when_no_recovery_capture_followed(tmp_path):
    """Older logs only have the source frame, and a split need not have been followed by a
    completed capture before Stop.  Both facts must be explicit rather than read as recovery.
    """
    run = tmp_path / "run_split_unrecovered"
    run.mkdir(parents=True)
    split = {
        "ts": "2026-08-13T00:20:55", "action": "capture_split", "photos": 2,
        "before": "00001_capture_split_before.png",
    }
    (run / "actions.jsonl").write_text(json.dumps(split) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "source screenshot `00001_capture_split_before.png`" in md
    assert "boundary-trigger screenshot" not in md
    assert "no later completed capture was recorded" in md


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


# ── Capabilities section (§3.12: tesseract / opencv / touch-watcher presence) ───────────────
def test_capabilities_section_reports_tesseract_path(monkeypatch):
    monkeypatch.setattr(bugreport.shutil, "which",
                         lambda name: "/opt/homebrew/bin/tesseract" if name == "tesseract" else None)
    assert "tesseract: /opt/homebrew/bin/tesseract" in bugreport.build_report(None)


def test_capabilities_section_reports_tesseract_absent(monkeypatch):
    monkeypatch.setattr(bugreport.shutil, "which", lambda name: None)
    assert "tesseract: absent" in bugreport.build_report(None)


def test_capabilities_section_reports_opencv_present(monkeypatch):
    # A fake module in sys.modules, not a real (un)install -- deterministic regardless of
    # whether this environment actually has the `hinge` extra, and reverted automatically by
    # monkeypatch after the test (see test_capabilities_section_reports_opencv_absent below,
    # which relies on that same revert to not permanently break cv2 for later tests).
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.__version__ = "9.9.9-test"
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    assert "opencv (cv2): present (9.9.9-test)" in bugreport.build_report(None)


def test_capabilities_section_reports_opencv_absent(monkeypatch):
    # sys.modules[name] = None is the documented way to make `import cv2` raise without
    # touching whatever is actually installed (see importlib docs on the import system).
    monkeypatch.setitem(sys.modules, "cv2", None)
    md = bugreport.build_report(None)
    assert "opencv (cv2): absent" in md
    assert "_require_vision" in md                # points a developer at WHERE this gates


def test_first_android_app_cfg_prefers_the_enabled_app():
    apps = {"bumble": {"adb_path": "adb", "serial": "BUMBLE-SERIAL"},
            "hinge": {"adb_path": "adb", "serial": "HINGE-SERIAL"}}
    assert bugreport._first_android_app_cfg(apps, ["hinge"]) == ("adb", "HINGE-SERIAL")


def test_first_android_app_cfg_falls_back_past_a_web_only_enabled_app():
    # bumble_web has no adb_path (it's a browser) -- the fallback must still find hinge's.
    apps = {"bumble_web": {"url": "https://bumble.com/app"},
            "hinge": {"adb_path": "adb", "serial": "HINGE-SERIAL"}}
    assert bugreport._first_android_app_cfg(apps, ["bumble_web"]) == ("adb", "HINGE-SERIAL")


def test_first_android_app_cfg_none_when_nothing_declares_adb_path():
    assert bugreport._first_android_app_cfg({"bumble_web": {"url": "x"}}, ["bumble_web"]) is None


class _FakeTouchWatcherSelects:
    """Stands in for touchwatch.TouchWatcher: start() 'finds' a device instantly (no real adb
    call), close() just records it ran. Tracks the last instance so tests can assert close()
    was actually reached -- the whole point of §3.12's probe is that it never leaves the
    stream attached, on EITHER outcome."""
    last_instance = None

    def __init__(self, adb_path, serial, screen_size, *, probe_timeout=10.0, **kw):
        self.adb_path = adb_path
        self.serial = serial
        self.device_path = None
        self.device_name = None
        self.closed = False
        _FakeTouchWatcherSelects.last_instance = self

    def start(self):
        self.device_path = "/dev/input/event3"
        self.device_name = "goodix_ts0"

    def close(self):
        self.closed = True


class _FakeTouchWatcherUnavailable:
    last_instance = None
    REASON = "no ABS_MT_POSITION_X/Y touch device found | pipe\nand a newline `backtick`"

    def __init__(self, adb_path, serial, screen_size, *, probe_timeout=10.0, **kw):
        self.closed = False
        _FakeTouchWatcherUnavailable.last_instance = self

    def start(self):
        raise touchwatch.TouchWatchUnavailable(_FakeTouchWatcherUnavailable.REASON)

    def close(self):
        self.closed = True


def test_touch_watcher_probe_reports_the_selected_device(monkeypatch):
    monkeypatch.setattr(oplove_config, "load", lambda path: types.SimpleNamespace(
        apps={"hinge": {"adb_path": "adb", "serial": "ABC123"}}, enabled_apps=["hinge"]))
    monkeypatch.setattr(touchwatch, "TouchWatcher", _FakeTouchWatcherSelects)
    md = bugreport.build_report(None)
    assert "touch watcher: would select `/dev/input/event3`" in md
    assert "goodix_ts0" in md
    assert _FakeTouchWatcherSelects.last_instance.closed is True   # never left attached


def test_touch_watcher_probe_reports_the_reason_when_unavailable(monkeypatch):
    monkeypatch.setattr(oplove_config, "load", lambda path: types.SimpleNamespace(
        apps={"hinge": {"adb_path": "adb"}}, enabled_apps=["hinge"]))
    monkeypatch.setattr(touchwatch, "TouchWatcher", _FakeTouchWatcherUnavailable)
    md = bugreport.build_report(None)
    assert "touch watcher: unavailable" in md
    # sanitized onto one line: newline collapsed, backtick swapped, pipe left alone (see
    # _sanitize_inline) -- a missing device is a reported line here, never a raised exception.
    assert "no ABS_MT_POSITION_X/Y touch device found | pipe and a newline 'backtick'" in md
    assert _FakeTouchWatcherUnavailable.last_instance.closed is True   # close() ran in `finally`


def test_touch_watcher_probe_when_no_android_app_is_configured(monkeypatch):
    monkeypatch.setattr(oplove_config, "load", lambda path: types.SimpleNamespace(
        apps={"bumble_web": {"url": "https://bumble.com/app"}}, enabled_apps=["bumble_web"]))
    assert "touch watcher: no Android app configured" in bugreport.build_report(None)


def test_touch_watcher_probe_when_config_cannot_load(monkeypatch):
    def _boom(path):
        raise ValueError("bad yaml")
    monkeypatch.setattr(oplove_config, "load", _boom)
    md = bugreport.build_report(None)
    assert "touch watcher: could not load" in md
    assert "bad yaml" in md


# ── Debug-log tail: new observe_decision/observe_resync fields (§3.12) ──────────────────────
def test_debug_log_tail_surfaces_observe_decision_fields(tmp_path):
    """hinge.py's wait_for_decision writes exactly these fields on a resolved PASS (see its
    `fields = dict(...)` block) -- lock that the raw tail keeps every one of them intact,
    since a developer reading a report needs identity/identity_dist/profile_name/gesture/
    watcher to tell a real advance apart from a false one."""
    run = tmp_path / "run_20260810_120000"
    run.mkdir(parents=True)
    capture = {"ts": "t0", "action": "capture", "photos": 6,
               "capture_truncated": False, "identity_seen": True, "profile_name": "Qelix"}
    decision = {"ts": "t1", "action": "observe_decision", "decision": "pass",
                "top": 12.3, "bot": 1.1, "min_sig_dist": None, "shift_matched": False,
                "capture_truncated": False, "identity": "new", "identity_dist": 22.4,
                "profile_name": "Amanda", "gesture": "tap_pass", "watcher": True}
    (run / "actions.jsonl").write_text(json.dumps(capture) + "\n" + json.dumps(decision) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert '"action": "capture"' in md and '"action": "observe_decision"' in md
    for field in ('"identity": "new"', '"identity_dist": 22.4', '"profile_name": "Amanda"',
                  '"gesture": "tap_pass"', '"watcher": true'):
        assert field in md, f"missing {field}"


def test_debug_log_tail_surfaces_observe_resync_fields(tmp_path):
    """A resync (worker.py treats the returned None as 'recapture, record nothing') is its own
    record, distinct from observe_decision -- must survive the tail the same way."""
    run = tmp_path / "run_20260810_121000"
    run.mkdir(parents=True)
    resync = {"ts": "t1", "action": "observe_resync", "identity": "new", "identity_dist": 30.1,
              "profile_name": None, "gesture": "resync", "watcher": True}
    (run / "actions.jsonl").write_text(json.dumps(resync) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert '"action": "observe_resync"' in md
    assert '"gesture": "resync"' in md
    assert '"watcher": true' in md


def test_debug_log_tail_shows_a_decision_and_its_preceding_capture_despite_older_filler(tmp_path):
    """Regression for the tail-length half of §3.12: even after _DEBUG_ACTION_TAIL's worth of
    unrelated older history, the most recent decision AND the capture immediately before it
    (the pairing a developer actually needs -- see the _DEBUG_ACTION_TAIL comment in
    bugreport.py) both survive being tailed, while genuinely old filler is dropped."""
    run = tmp_path / "run_20260810_130000"
    run.mkdir(parents=True)
    lines = [json.dumps({"ts": "t", "action": "capture", "photos": 1, "n": i}) for i in range(40)]
    lines.append(json.dumps({"ts": "t", "action": "capture", "photos": 6,
                              "identity_seen": True, "profile_name": "Priya"}))
    lines.append(json.dumps({"ts": "t", "action": "observe_decision", "decision": "pass",
                              "identity": "new", "identity_dist": 19.0, "profile_name": "Priya",
                              "gesture": "tap_pass", "watcher": True}))
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert '"action": "observe_decision"' in md
    assert md.count('"profile_name": "Priya"') == 2       # both the decision AND its capture
    assert '"n": 39' in md                                 # newest filler line still in range
    assert '"n": 0' not in md                              # oldest filler genuinely dropped


# ── Debug-log tail: collapsing the observe_waiting heartbeat (audited run_20260810_203956) ──
# The heartbeat (_note_observe_waiting, hinge.py) fires every ~15s of human deliberation even
# on reason="no_change", which PROVES the screen hasn't moved. A single 3-minute decision logs
# 12 of these -- enough, pre-fix, to swamp the whole raw-line tail budget and push the
# "capture" record for that same profile out of the report entirely.
def test_debug_log_tail_collapses_repeated_observe_waiting_heartbeats(tmp_path):
    run = tmp_path / "run_20260810_203956"
    run.mkdir(parents=True)
    lines = [json.dumps({"ts": "capture0", "action": "capture", "photos": 9,
                          "profile_name": "Example Profile"})]
    # 12 identical no_change heartbeats, ~15s apart -- exactly the audited-run shape.
    for i in range(12):
        lines.append(json.dumps({"ts": f"20:46:{13 + i * 15:02d}", "action": "observe_waiting",
                                  "reason": "no_change"}))
    lines.append(json.dumps({"ts": "decision0", "action": "observe_decision", "decision": "pass",
                              "profile_name": "Example Profile"}))
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    # The 12 raw heartbeat lines collapse into exactly one summary line...
    assert md.count('"reason": "no_change"') == 1
    assert '"repeated": 12' in md
    # ...spanning the first and last heartbeat timestamps...
    assert '"ts": "20:46:13-20:46:' in md
    # ...while the capture and decision either side of the run are untouched raw JSON, still
    # visibly paired (the whole point of the fix).
    assert '"action": "capture"' in md and '"profile_name": "Example Profile"' in md
    assert '"action": "observe_decision"' in md


def test_debug_log_tail_keeps_first_and_last_record_of_a_run_that_spans_the_whole_tail(tmp_path):
    """When collapsing would otherwise swallow the ENTIRE displayed tail into one summary line
    (e.g. a stall that has been going on since before the window even starts), the very first
    and last raw records must still be peeled back out as literal JSON -- so the newest state
    (with its own screenshot filename) is never hidden behind a bare "repeated: N" count."""
    run = tmp_path / "run_stalled"
    run.mkdir(parents=True)
    lines = [json.dumps({"ts": f"t{i}", "action": "observe_waiting", "reason": "no_change",
                          "after": f"shot_{i}.png"}) for i in range(20)]
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    tail = bugreport._collapse_action_tail(lines, 30)
    assert json.loads(tail[0]) == json.loads(lines[0])        # first record: untouched, literal
    assert json.loads(tail[-1]) == json.loads(lines[-1])       # last record: untouched, literal
    assert tail[-1] != tail[0]
    # exactly one middle summary line covers everything strictly between the two boundaries
    middle = [t for t in tail if '"repeated"' in t]
    assert len(middle) == 1
    assert json.loads(middle[0])["repeated"] == 18


def test_debug_log_tail_only_merges_matching_action_and_reason(tmp_path):
    """Two different reasons for the SAME action must never merge -- only an exact
    (action, reason) match collapses (see hinge.py's _note_observe_waiting vocabulary: same,
    scroll, no_change, not_deck_ready, not_settled are all distinct signals)."""
    lines = [json.dumps({"ts": "t0", "action": "observe_waiting", "reason": "no_change"}),
             json.dumps({"ts": "t1", "action": "observe_waiting", "reason": "same"}),
             json.dumps({"ts": "t2", "action": "observe_waiting", "reason": "no_change"})]
    out = bugreport._collapse_action_tail(lines, 30)
    assert out == lines                                        # nothing adjacent actually matched


def test_debug_log_tail_never_reorders_or_merges_non_adjacent_repeats(tmp_path):
    """Two identical (action, reason) records separated by something else must stay as two
    separate raw lines, in their original order -- collapsing is adjacency-only."""
    lines = [json.dumps({"ts": "t0", "action": "observe_waiting", "reason": "no_change"}),
             json.dumps({"ts": "t1", "action": "capture", "photos": 1}),
             json.dumps({"ts": "t2", "action": "observe_waiting", "reason": "no_change"})]
    out = bugreport._collapse_action_tail(lines, 30)
    assert out == lines


def test_debug_log_tail_malformed_lines_pass_through_untouched(tmp_path):
    """A line that isn't valid JSON, or is JSON but has no "action"/"reason" key, must pass
    through as-is rather than raise -- this collector is best-effort, not a strict parser."""
    lines = ["not valid json {{{",
             json.dumps({"no_action_key": True}),
             json.dumps({"action": "capture"}),          # has "action", no "reason" -- fine
             json.dumps({"action": "observe_waiting", "reason": "no_change"})]
    out = bugreport._collapse_action_tail(lines, 30)
    assert out == lines                                        # nothing here can legally merge


def test_debug_log_section_shows_action_counts_histogram_above_the_tail(tmp_path):
    run = tmp_path / "run_counts"
    run.mkdir(parents=True)
    lines = ([json.dumps({"action": "capture", "n": i}) for i in range(4)]
             + [json.dumps({"action": "observe_waiting", "reason": "no_change"}) for _ in range(18)]
             + [json.dumps({"action": "observe_decision"}) for _ in range(2)])
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert "action counts: capture 4 · observe_waiting 18 · observe_decision 2" in md
    # the histogram is a summary of the WHOLE file, not just what made it into the tail
    counts_pos = md.index("action counts:")
    tail_pos = md.index("actions.jsonl (tail):")
    assert counts_pos < tail_pos                               # directly above the inlined tail


# ── Stall summary ─────────────────────────────────────────────────────────────────────────
# An unrecognised Hinge+ paywall can leave _await_like_resolved polling like_sheet until the
# operator stops the run. actions.jsonl records every poll, but diagnosing the hang from the
# raw tail requires noticing observe_waiting repeating with the same reason. These tests pin
# _stall_summary_md's worked example from its docstring: "longest observe stall:
# reason=`like_sheet` for 1m37s (5 records)" for that exact run's repeats.
def test_stall_summary_names_the_reason_and_its_duration_from_uncollapsed_records():
    """UNCOLLAPSED shape: individual, raw actions.jsonl lines -- one per poll, real ISO
    timestamps -- exactly as hinge.py's _note_observe_waiting actually writes them, and exactly
    what _one_debug_dir_md hands to _stall_summary_md (never the collapsed display form; see
    the sibling test below for that shape). Reproduces the audited run's own like_sheet repeats
    verbatim: 01:42:43 -> 01:44:20, 97s, 5 records -- the exact numbers _stall_summary_md's own
    docstring worked example quotes."""
    lines = [
        json.dumps({"ts": "2026-08-11T01:42:36", "action": "observe_like_anchor"}),
        json.dumps({"ts": "2026-08-11T01:42:43", "action": "observe_waiting", "reason": "like_sheet"}),
        json.dumps({"ts": "2026-08-11T01:43:13", "action": "observe_waiting", "reason": "like_sheet"}),
        json.dumps({"ts": "2026-08-11T01:43:19", "action": "observe_waiting", "reason": "like_sheet"}),
        json.dumps({"ts": "2026-08-11T01:43:49", "action": "observe_waiting", "reason": "like_sheet"}),
        json.dumps({"ts": "2026-08-11T01:44:20", "action": "observe_waiting", "reason": "like_sheet"}),
    ]

    md = bugreport._stall_summary_md(lines)

    assert "longest observe stall" in md
    assert "reason=`like_sheet`" in md
    assert "1m37s" in md            # 01:42:43 -> 01:44:20
    assert "(5 records)" in md


def test_stall_summary_and_the_collapsed_tail_entry_agree_on_the_same_stall(tmp_path):
    """COLLAPSED shape: the SAME repeats, independently reduced by _collapse_action_tail /
    _render_run into bugreport.py's own display object -- {"ts": "<start>-<end>", "action":
    ..., "reason": ..., "repeated": N} -- for the "actions.jsonl (tail)" section further down
    the same report. The stall summary (computed from the raw, uncollapsed lines) and the tail
    (computed from the same lines but collapsed for display) must describe the SAME stall
    consistently -- same reason, same record count -- not silently disagree because one code
    path counts differently than the other."""
    run = tmp_path / "example_run"
    run.mkdir(parents=True)
    start = datetime.datetime(2026, 8, 11, 1, 22, 13)
    lines = [json.dumps({"ts": "2026-08-11T01:19:51", "action": "capture", "photos": 9,
                          "profile_name": "Example Profile"})]
    for i in range(12):
        ts = (start + datetime.timedelta(seconds=15 * i)).isoformat()
        lines.append(json.dumps({"ts": ts, "action": "observe_waiting", "reason": "no_change"}))
    lines.append(json.dumps({"ts": (start + datetime.timedelta(seconds=15 * 11 + 5)).isoformat(),
                              "action": "observe_decision", "decision": "pass",
                              "profile_name": "Example Profile"}))
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    # (1) the UNCOLLAPSED-derived stall summary, at the top of the section...
    assert "stall summary:" in md
    assert "reason=`no_change`" in md
    assert "2m45s" in md           # 11 * 15s between the first and last of the 12 repeats
    assert "(12 records)" in md
    # (2) ...and the SAME 12 repeats, independently collapsed for the tail further down.
    assert '"repeated": 12' in md
    assert '"reason": "no_change"' in md
    stall_pos = md.index("stall summary:")
    tail_pos = md.index("actions.jsonl (tail):")
    assert stall_pos < tail_pos    # the summary sits above the raw/collapsed tail, not below it


def test_stall_summary_is_empty_for_a_healthy_run_with_nothing_repeated():
    """The common case -- a run that never waited on the same reason twice in a row -- must
    render nothing extra, the same "quiet when healthy" contract as _app_diagnostics_md and
    _action_counts_line."""
    lines = [
        json.dumps({"ts": "2026-08-11T01:00:00", "action": "capture", "photos": 6}),
        json.dumps({"ts": "2026-08-11T01:00:05", "action": "observe_waiting", "reason": "no_change"}),
        json.dumps({"ts": "2026-08-11T01:00:10", "action": "observe_decision", "decision": "like"}),
    ]
    assert bugreport._stall_summary_md(lines) == ""


def test_stall_summary_survives_malformed_and_missing_timestamps():
    """A bug report that crashes while reporting a bug is the worst possible failure mode (see
    _diagnostic_improvement_md). actions.jsonl "ts" fields are free-form -- _parse_action_ts's
    own docstring names three failure shapes that must all degrade to "unknown duration" rather
    than raise: the key missing entirely, a non-string value, and a string that isn't
    ISO-parseable. All three appear here on repeats of the SAME reason (so _stall_candidates
    still has something to report), mixed with one line that isn't even valid JSON."""
    lines = [
        "not valid json {{{",
        json.dumps({"action": "observe_waiting", "reason": "not_deck_ready"}),        # no ts key at all
        json.dumps({"ts": None, "action": "observe_waiting", "reason": "not_deck_ready"}),
        json.dumps({"ts": "not-a-timestamp", "action": "observe_waiting", "reason": "not_deck_ready"}),
        json.dumps({"ts": 12345, "action": "observe_waiting", "reason": "not_deck_ready"}),  # not a string
    ]

    md = bugreport._stall_summary_md(lines)      # must not raise

    assert "reason=`not_deck_ready`" in md
    assert "unknown duration" in md
    assert "(4 records)" in md                    # the unparsable JSON line never joins the stretch


def test_stall_summary_malformed_timestamps_do_not_crash_the_full_debug_section(tmp_path):
    """The same guarantee, exercised through the real on-disk path (_one_debug_dir_md) rather
    than calling _stall_summary_md directly -- a malformed timestamp in a real actions.jsonl
    file must not take down the rest of the debug-log section, or the report around it."""
    run = tmp_path / "run_malformed_ts"
    run.mkdir(parents=True)
    lines = [
        json.dumps({"action": "observe_waiting", "reason": "no_change"}),
        json.dumps({"ts": "garbage", "action": "observe_waiting", "reason": "no_change"}),
        json.dumps({"ts": "still garbage", "action": "observe_waiting", "reason": "no_change"}),
    ]
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})   # must not raise

    assert "reason=`no_change`" in md
    assert "unknown duration" in md


# ── Run status: defensive `stopping` flag (landing concurrently in status.py) ───────────────
class _StoppingHub:
    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "stopping", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12, "stopping": True,
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "live",
                                "last_decision": "pass", "last_score": None, "swipes_run": 12,
                                "error": None, "stop_reason": None}}}}


def test_status_section_surfaces_stopping_flag_when_present():
    md = bugreport.build_report(_StoppingHub())
    assert "stopping: stop requested, winding down" in md


class _NoStoppingKeyHub:
    """A status dict shaped like BEFORE the `stopping` field landed -- must render fine with no
    KeyError, and simply omit the stopping line (falsy via .get(), not missing-key crash)."""
    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "live", "mode": "auto", "running": True, "labels": 40, "min_labels": 40,
            "ranker_ready": True, "labels_needed": 0, "budget_spent": 1.2, "budget_cap": 5.0,
            "openers": 12,
            "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "live",
                                "last_decision": "pass", "last_score": None, "swipes_run": 12,
                                "error": None, "stop_reason": None}}}}


def test_status_section_handles_missing_stopping_key_gracefully():
    md = bugreport.build_report(_NoStoppingKeyHub())
    assert "stopping:" not in md
    assert "| hinge |" in md                                    # rest of the section still renders


# ── Recent openers: the real OpenerService ring buffer, plumbed via HubState.recent_openers ─
# This is the section that used to fall back to each app's live opener_suggestion/
# opener_referenced status field (at most one row per app, no model name)
# because HubState never captured a reference to the running OpenerService. supervisor.py now
# takes on_opener_service and hub/state.py captures it, so hub_state.recent_openers() is the
# real thing: a snapshot of OpenerService.recent_openers_snapshot(), newest entry LAST.
def _opener_entry(ts="2026-08-10T12:00:00", app="hinge", model="gemini-2.5-flash",
                   advisory=False, index=0, referenced="the beach photo",
                   opener="hey, love the beach shot"):
    return {"ts": ts, "app": app, "model": model, "advisory": advisory,
            "index": index, "referenced": referenced, "opener": opener}


class _FakeHubOpeners:
    """Stands in for HubState: build_report/_recent_openers_md only ever call
    .recent_openers() on it, never .snapshot() -- this section no longer reads the live
    per-app status fields at all."""
    def __init__(self, entries):
        self._entries = entries

    def recent_openers(self):
        return self._entries


def test_recent_openers_section_handles_no_hub_gracefully():
    """No hub at all (bugreport used outside the hub process) must render a plain
    explanatory line, not raise -- matching every other section's `hub_state is None`
    handling in this file."""
    md = bugreport._recent_openers_md(None)
    assert "no hub" in md


def test_recent_openers_section_handles_empty_list_gracefully():
    """Either no run has ever started, or one is live but hasn't generated an opener yet --
    either way [] is not an error and must render a graceful explanatory line."""
    md = bugreport._recent_openers_md(_FakeHubOpeners([]))
    assert "no openers generated" in md


def test_recent_openers_section_renders_newest_first_and_caps_at_the_shown_limit():
    """recent_openers_snapshot() is a ring buffer that appends newest LAST -- a human reading
    a bug report needs the most recent generation first. Also must not dump the whole
    history: capped at _RECENT_OPENERS_SHOWN so one long run can't blow up the report."""
    total = bugreport._RECENT_OPENERS_SHOWN + 4
    entries = [_opener_entry(index=i, opener=f"opener-body-{i}") for i in range(total)]

    md = bugreport._recent_openers_md(_FakeHubOpeners(entries))

    order = [int(tok.rsplit("-", 1)[1]) for tok in re.findall(r"opener-body-\d+", md)]
    assert len(order) == bugreport._RECENT_OPENERS_SHOWN                # capped, not the full history
    assert order == list(range(total - 1, total - 1 - bugreport._RECENT_OPENERS_SHOWN, -1))  # newest first


def test_recent_openers_section_marks_numbered_item_crops_with_status_circles():
    """WHAT THE MODEL SAW is the whole point of this section -- it must be unmissable and use
    the owner's 🟢/🔴 status-circle convention, never a hand emoji (owner rule: hard to tell
    thumbs-up/down apart at a glance).

    The three states are not symmetrical, and doc 5.9's observe inversion is why. Both modes now
    send numbered item crops and NOTHING passes an anchor, so `model_items` is the strongest
    shape this pipeline has and must not print as "blind"."""
    entries = [
        _opener_entry(index=0, opener="alpha"),
        _opener_entry(index=1, opener="beta"),
        _opener_entry(index=2, opener="gamma"),
    ]
    entries[2]["index_space"] = "model_items"

    md = bugreport._recent_openers_md(_FakeHubOpeners(entries))

    lines = md.splitlines()
    alpha_header = next(ln for ln in lines if "index: 0" in ln)
    beta_header = next(ln for ln in lines if "index: 1" in ln)
    gamma_header = next(ln for ln in lines if "index: 2" in ln)
    assert "🔴 no numbered item crops" in alpha_header
    assert "🔴 no numbered item crops" in beta_header
    assert "🟢 chose from numbered item crops" in gamma_header
    assert "blind" not in gamma_header       # the crop shape is the opposite of blind
    assert "👍" not in md and "👎" not in md          # never the banned hand-emoji convention


def test_recent_openers_section_names_the_index_space_when_the_entry_carries_one():
    """`index: 3` is uninterpretable on its own -- 3 means a different item depending on
    whether the request numbered her scroll frames or her item crops, and a reader comparing
    two reports has no other way to tell (that ambiguity is what let the two spaces be confused
    in code in the first place). Entries written before 2026-08-12 carry no space and must
    render with no suffix rather than a guessed one."""
    entry = _opener_entry(index=3, opener="with a space")
    entry["index_space"] = "profile_photos"
    older = _opener_entry(index=4, opener="without a space")   # no index_space key at all

    md = bugreport._recent_openers_md(_FakeHubOpeners([entry, older]))

    assert "index: 3 (profile_photos)" in md
    assert "index: 4 ·" in md or "index: 4\n" in md


def test_recent_opener_text_is_truncated_at_the_configured_character_cap():
    """One runaway response must not blow up the report -- opener text is capped at
    _RECENT_OPENER_TEXT_CHARS, same contract the old per-app fallback enforced."""
    long_opener = "x" * (bugreport._RECENT_OPENER_TEXT_CHARS + 50)
    entries = [_opener_entry(opener=long_opener)]

    md = bugreport._recent_openers_md(_FakeHubOpeners(entries))

    assert long_opener not in md
    assert ("x" * bugreport._RECENT_OPENER_TEXT_CHARS) + "…" in md


def test_recent_openers_section_sanitizes_model_and_referenced_free_text():
    """model/referenced/opener are free text the model itself produced -- none of it is
    trusted. It must not be able to inject a fake heading, break out of the blockquote it's
    rendered inside, or smuggle a raw backtick past the inline-code spans (see
    _sanitize_inline, and the equivalent stop_reason test for ## Run status)."""
    entries = [_opener_entry(
        model="gemini-2.5-flash`\n# fake heading",
        referenced="the `beach` shot\nwith a newline",
        opener="hey! loved the beach\nkeep swimming `champ`",
    )]

    md = bugreport._recent_openers_md(_FakeHubOpeners(entries))

    assert not any(ln.strip().startswith("# fake heading") for ln in md.splitlines())
    assert "gemini-2.5-flash' # fake heading" in md
    assert "the 'beach' shot with a newline" in md
    assert "hey! loved the beach keep swimming 'champ'" in md


class _RaisingHubOpeners:
    """recent_openers() itself blows up -- e.g. a torn-down service the caller failed to
    guard, or (this class) a HubState stand-in that simply doesn't implement the method."""
    def recent_openers(self):
        raise RuntimeError("opener service torn down mid-read")


def test_recent_openers_section_survives_a_raising_recent_openers_call():
    """HubState.recent_openers() is documented to never raise, but this section's own
    docstring is explicit that it does not re-guard that promise -- _safe_section is the
    report's actual safety net here, exactly like every other section (see
    test_malformed_status_section_does_not_crash_whole_report)."""
    md = bugreport.build_report(_RaisingHubOpeners())
    assert "## Recent openers" in md
    assert "⚠️ this section failed to generate" in md


# ── Recent opener rejections: the OTHER half of the opener paper trail -- attempts the
# deterministic guards in opener.py's _parse REJECTED, not just the successes above. Before
# this section existed, a rejection was printed to the console and then lost forever.
def _rejection_entry(ts="2026-08-10T12:00:00", app="hinge", model="gemini-2.5-flash",
                     attempt=2, reason_code="scaffolding",
                     reason="Gemini's opener contained scaffolding text",
                     raw_opener="Here's the response: love the beach shot"):
    return {"ts": ts, "app": app, "model": model, "attempt": attempt,
            "reason_code": reason_code, "reason": reason, "raw_opener": raw_opener}


class _FakeHubRejections:
    """Stands in for HubState: build_report/_recent_opener_rejections_md only ever call
    .recent_opener_rejections() on it."""
    def __init__(self, entries):
        self._entries = entries

    def recent_opener_rejections(self):
        return self._entries


def test_recent_opener_rejections_section_handles_no_hub_gracefully():
    md = bugreport._recent_opener_rejections_md(None)
    assert "no hub" in md


def test_recent_opener_rejections_section_handles_empty_list_gracefully():
    md = bugreport._recent_opener_rejections_md(_FakeHubRejections([]))
    assert "no opener rejections recorded" in md


def test_recent_opener_rejections_section_renders_newest_first_and_caps_at_the_shown_limit():
    total = bugreport._RECENT_REJECTIONS_SHOWN + 4
    entries = [_rejection_entry(attempt=i, raw_opener=f"opener-body-{i}") for i in range(total)]

    md = bugreport._recent_opener_rejections_md(_FakeHubRejections(entries))

    order = [int(tok.rsplit("-", 1)[1]) for tok in re.findall(r"opener-body-\d+", md)]
    assert len(order) == bugreport._RECENT_REJECTIONS_SHOWN
    assert order == list(range(total - 1, total - 1 - bugreport._RECENT_REJECTIONS_SHOWN, -1))


def test_recent_opener_rejections_section_shows_reason_code_and_attempt():
    entries = [_rejection_entry(attempt=4, reason_code="undeliverable_chars",
                                raw_opener="love the \U0001F384 vibes")]

    md = bugreport._recent_opener_rejections_md(_FakeHubRejections(entries))

    assert "attempt 4" in md
    assert "undeliverable_chars" in md
    assert "love the" in md


def test_recent_opener_rejections_section_renders_none_raw_opener_as_a_placeholder():
    """no_text/max_tokens rejections carry raw_opener=None -- must render a plain
    placeholder, never the literal string "None"."""
    entries = [_rejection_entry(reason_code="no_text", raw_opener=None)]

    md = bugreport._recent_opener_rejections_md(_FakeHubRejections(entries))

    assert "no candidate text" in md
    assert "> None" not in md


def test_recent_opener_rejection_text_is_truncated_at_the_configured_character_cap():
    long_text = "x" * (bugreport._RECENT_OPENER_TEXT_CHARS + 50)
    entries = [_rejection_entry(raw_opener=long_text)]

    md = bugreport._recent_opener_rejections_md(_FakeHubRejections(entries))

    assert long_text not in md
    assert ("x" * bugreport._RECENT_OPENER_TEXT_CHARS) + "…" in md


def test_recent_opener_rejections_section_sanitizes_free_text():
    entries = [_rejection_entry(
        model="gemini-2.5-flash`\n# fake heading",
        raw_opener="hey! loved the beach\nkeep swimming `champ`",
    )]

    md = bugreport._recent_opener_rejections_md(_FakeHubRejections(entries))

    assert not any(ln.strip().startswith("# fake heading") for ln in md.splitlines())
    assert "gemini-2.5-flash' # fake heading" in md
    assert "hey! loved the beach keep swimming 'champ'" in md


class _RaisingHubRejections:
    def recent_opener_rejections(self):
        raise RuntimeError("opener service torn down mid-read")


def test_recent_opener_rejections_section_survives_a_raising_call():
    md = bugreport.build_report(_RaisingHubRejections())
    assert "## Recent opener rejections" in md
    assert "⚠️ this section failed to generate" in md
    assert "## Debug log" in md               # the rest of the report still renders
