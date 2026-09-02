"""bugreport — redacted markdown diagnostic. Offline."""
import datetime
import hashlib
import io
import json
import os
import re
import sys
import threading
import types
from pathlib import Path

import pytest

from operation_love import bugreport, config as oplove_config
from operation_love.drivers import touchwatch


def _ring_bodies():
    return [line.split(" ", 1)[1] for line in bugreport._LOG_RING]


def test_report_has_core_sections():
    md = bugreport.build_report(None, description="it broke")
    for h in ["# Operation Love — Bug Report", "## What happened", "it broke",
              "## Reporter follow-up", "## Build", "## System", "## Dependencies", "## Capabilities", "## Config",
              "## Hinge targeting readiness",
              "## Secrets", "## Diagnostic improvement", "## Run status", "## Recent openers",
              "## Run completion assessment",
              "## Recent opener rejections",
              "## Debug log (on-disk actions + screenshots)", "## Recent logs"]:
        assert h in md, f"missing section: {h}"
    assert "improve `operation_love/bugreport.py`" in md


# ── Run completion assessment ───────────────────────────────────────────────
class _CompletionHub:
    def __init__(self, *, phase="stopped", running=False, hub_error=None, app_state="stopped",
                 app_error=None, stop_kind=None):
        self._snapshot = {
            "running": running,
            "error": hub_error,
            "status": {
                "run_id": "completion-test-run",
                "phase": phase,
                "running": running,
                "stopping": False,
                "apps": {"hinge": {
                    "app": "hinge", "state": app_state, "error": app_error,
                    "stop_kind": stop_kind,
                }},
            },
        }

    def snapshot(self):
        return self._snapshot


def test_completion_assessment_calls_stopped_a_durable_clean_completion(monkeypatch):
    monkeypatch.setattr(bugreport, "_completion_capture_facts", lambda *_: {})
    bugreport._LOG_RING.clear()

    md = bugreport._run_completion_assessment_md(_CompletionHub(), "unused.yaml")

    assert "Outcome: COMPLETED CLEANLY" in md
    assert "all workers exited" in md
    assert "store flush succeeded" in md


@pytest.mark.parametrize(
    ("hub", "expected"),
    [
        (_CompletionHub(phase="save_failed"), "Outcome: SAVE FAILED"),
        (_CompletionHub(phase="wedged", app_state="wedged"), "Outcome: DEGRADED SHUTDOWN"),
        (_CompletionHub(app_state="error", app_error="driver failed"),
         "Outcome: COMPLETED WITH ERRORS"),
        (_CompletionHub(phase="live", running=True, app_state="acting"),
         "Outcome: IN PROGRESS / INDETERMINATE"),
    ],
)
def test_completion_assessment_does_not_mislabel_failure_or_live_run(monkeypatch, hub, expected):
    monkeypatch.setattr(bugreport, "_completion_capture_facts", lambda *_: {})

    assert expected in bugreport._run_completion_assessment_md(hub, "unused.yaml")


def test_completion_assessment_names_pending_stop_and_current_run(monkeypatch):
    hub = _CompletionHub(phase="loading saved data", running=True, app_state="starting")
    hub._snapshot["status"]["stopping"] = True
    monkeypatch.setattr(bugreport, "_completion_capture_facts", lambda *_: {})

    md = bugreport._run_completion_assessment_md(hub, "unused.yaml")

    assert "Outcome: STOPPING / INDETERMINATE" in md
    assert "for run `completion-test-run`" in md
    assert "stop was requested while phase was 'loading saved data'" in md
    assert "shutdown and persistence finish" in md


def test_completion_assessment_requires_at_least_one_app_status(monkeypatch):
    hub = _CompletionHub()
    hub._snapshot["status"]["apps"] = {}
    monkeypatch.setattr(bugreport, "_completion_capture_facts", lambda *_: {})

    assert "Outcome: INDETERMINATE" in bugreport._run_completion_assessment_md(
        hub, "unused.yaml")
    assert "no app status" in bugreport._run_completion_assessment_md(hub, "unused.yaml")


def test_completion_assessment_names_recovered_provider_faults_and_coverage(monkeypatch):
    monkeypatch.setattr(
        bugreport, "_completion_capture_facts",
        lambda *_: {"coverage_gaps": 3, "coverage_candidates": 5, "capture_truncated": False},
    )
    bugreport._LOG_RING.clear()
    bugreport._LOG_RING.extend([
        "12:00:00 Gemini opener: gemini-test failed at the transport level (TimeoutError)",
        "12:00:01 ordinary diagnostic",
    ])
    try:
        md = bugreport._run_completion_assessment_md(_CompletionHub(), "unused.yaml")
    finally:
        bugreport._LOG_RING.clear()

    assert "Outcome: COMPLETED SAFELY, WITH LIMITATIONS" in md
    assert "1 recovered provider transport failure" in md
    assert "still-photo coverage skipped 3 of 5 photo candidate(s)" in md


def test_completion_facts_will_not_follow_a_status_named_run_symlink(tmp_path, monkeypatch):
    """Run IDs are diagnostic input, so their path must not escape the configured debug root."""
    debug_root = tmp_path / "debug"
    debug_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "actions.jsonl").write_text(json.dumps({
        "action": "capture",
        "item_coverage": {"no_dwell_coverage_page_hearts": [1],
                          "photo_candidate_page_hearts": [1]},
        "capture_truncated": True,
    }) + "\n")
    (debug_root / "completion-test-run").symlink_to(outside, target_is_directory=True)
    cfg = types.SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"debug_log": True, "debug_dir": str(debug_root)}},
    )
    monkeypatch.setattr(oplove_config, "load", lambda _path: cfg)

    assert bugreport._completion_capture_facts(
        {"run_id": "completion-test-run"}, "unused.yaml") == {
            "coverage_gaps": 0, "coverage_candidates": 0, "capture_truncated": False,
        }


def test_reporter_follow_up_makes_a_one_word_description_actionable():
    md = bugreport.build_report(None, description="bug")

    assert "## Reporter follow-up" in md
    assert "description is too brief to diagnose from telemetry alone" in md
    assert "expected result" in md
    assert "actual result" in md
    assert "last action or steps" in md
    assert "whether it repeats" in md


def test_reporter_follow_up_does_not_falsely_flag_a_substantive_description():
    md = bugreport.build_report(
        None,
        description="After I tap Pass, the card stays visible instead of advancing.",
    )

    assert "Reporter description is present" in md
    assert "description is too brief" not in md


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


def test_final_report_redacts_secrets_from_debug_rows_logs_and_hub_free_text(monkeypatch):
    """The secrets section was already presence-only, but arbitrary diagnostic inputs can echo
    a provider credential.  The final report boundary must protect every section at once."""
    secret = "AIzaGeminiSecretValueForRegression"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    bugreport._LOG_RING.clear()
    bugreport._LOG_RING.extend([
        f"provider said Authorization: Bearer {secret}",
        f"request failed at https://example.test/?access_token={secret}",
        f"debug row api_key={secret}",
    ])

    class _SensitiveHub:
        def snapshot(self):
            return {"running": False, "error": None, "status": {
                "phase": "stopped", "mode": "auto", "running": False,
                "labels": 0, "min_labels": 40, "ranker_ready": False,
                "budget_spent": 0.0, "budget_cap": 5.0, "openers": 0,
                "apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "error",
                                     "last_decision": None, "last_score": None, "swipes_run": 0,
                                     "stop_reason": f"provider api_key={secret}",
                                     "error": f"Authorization: Bearer {secret}"}}}}

    monkeypatch.setattr(
        bugreport, "_debug_log_md",
        lambda _config, _hub=None: f"debug JSONL row: {{\"token\": \"{secret}\"}}",
    )
    try:
        md = bugreport.build_report(_SensitiveHub(), description=f"my key was {secret}")
    finally:
        bugreport._LOG_RING.clear()

    assert secret not in md
    assert "GEMINI_API_KEY: present" in md
    assert "Authorization: [REDACTED]" in md
    assert "api_key=[REDACTED]" in md
    assert "access_token=[REDACTED]" in md
    assert '"token": "[REDACTED]"' in md


def test_log_capture_roundtrip():
    bugreport.install_log_capture()
    print("OPLOVE_TEST_LOGLINE_marker")
    assert any("OPLOVE_TEST_LOGLINE_marker" in line for line in bugreport.recent_logs())
    assert "OPLOVE_TEST_LOGLINE_marker" in bugreport.build_report(None)


def test_log_capture_reinstalls_after_the_host_replaces_stdout(monkeypatch):
    """The installed flag must not outlive the stream wrapper it is meant to describe."""
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stdout", replacement)

    bugreport.install_log_capture()
    print("OPLOVE_REPLACED_STDOUT_marker")

    assert isinstance(sys.stdout, bugreport._Tee)
    assert replacement.getvalue() == "OPLOVE_REPLACED_STDOUT_marker\n"
    assert any("OPLOVE_REPLACED_STDOUT_marker" in line for line in bugreport.recent_logs())


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


def test_status_section_names_the_snapshot_run_and_uptime():
    snapshot = _FakeHub().snapshot()
    hub_snapshot = snapshot["status"]
    hub_snapshot["run_id"] = "current-run-123"
    hub_snapshot["uptime_s"] = 2.4
    hub = types.SimpleNamespace(snapshot=lambda: snapshot)

    md = bugreport._status_md(hub)

    assert "status provenance: run `current-run-123` · status uptime: 2s" in md


def test_status_section_healthy_run_has_no_diagnostics_section():
    # The common case: no app has a stop_reason or error. The "Stop reasons / errors" block
    # must not appear at all -- not as an empty heading, not as a stray blank bullet list.
    md = bugreport.build_report(_FakeHub())
    assert "Stop reasons / errors" not in md


class _TrainingWarningHub:
    """The report must preserve the hub's actual no-suggestion explanation, rather than
    reducing an intentional safety refusal to the ambiguous word ``waiting``."""
    WARNING = "item index refused | no trustworthy shift\n`do not offer text`"

    def snapshot(self):
        return {"running": True, "error": None, "status": {
            "phase": "live", "mode": "training", "running": True, "labels": 12,
            "min_labels": 40, "ranker_ready": False, "labels_needed": 28,
            "budget_spent": 0.0, "budget_cap": 5.0, "openers": 0,
            "apps": {"hinge": {"app": "hinge", "mode": "training", "state": "waiting_approval",
                                "last_decision": None, "last_score": None, "swipes_run": 0,
                                "opener_warning": self.WARNING, "opener_pending": False}}}}


def test_status_section_surfaces_current_training_guidance_safely():
    md = bugreport.build_report(_TrainingWarningHub())

    assert "Current hub guidance (snapshot, not a new phone read)" in md
    assert "typed target opener is ready; choose Like to send it or Dislike to pass" in md
    assert "no suggestion to type" in md
    assert "item index refused | no trustworthy shift 'do not offer text'" in md
    assert _TrainingWarningHub.WARNING not in md
    assert not any(line.strip() == "`do not offer text`" for line in md.splitlines())


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


def test_debug_log_section_marks_previous_run_separately_from_live_status(tmp_path):
    previous = tmp_path / "previous-run"
    previous.mkdir(parents=True)
    (previous / "actions.jsonl").write_text('{"action": "capture"}\n')

    md = bugreport._one_debug_dir_md(
        "hinge", {"debug_dir": str(tmp_path)}, current_run_id="new-startup-run")

    assert "latest run:" in md
    assert "provenance: previous on-disk run" in md
    assert "current status run is `new-startup-run`" in md


def test_debug_log_section_marks_matching_status_run_as_current(tmp_path):
    current = tmp_path / "current-run"
    current.mkdir(parents=True)
    (current / "actions.jsonl").write_text('{"action": "capture"}\n')

    md = bugreport._one_debug_dir_md(
        "hinge", {"debug_dir": str(tmp_path)}, current_run_id="current-run")

    assert "provenance: current status run" in md
    assert "previous on-disk run" not in md


def test_debug_log_links_full_auto_opener_snapshot_and_landed_outcome(tmp_path):
    run = tmp_path / "run_auto_opener_evidence"
    run.mkdir(parents=True)
    frame = b"typed opener on target photo"
    opener = ("This looks like the ideal setting for a crisp fall walk. Do you prefer "
              "exploring quiet trails for hours or heading straight for a warm coffee?")
    evidence_id = hashlib.sha256(frame + b"\0" + opener.encode()).hexdigest()
    shot = "00066_auto_opener_pre_send_before.png"
    (run / shot).write_bytes(frame)
    records = [
        {
            "ts": "2026-08-25T04:06:28", "action": "auto_opener_pre_send",
            "before": shot, "opener": opener,
            "opener_sha256": hashlib.sha256(opener.encode()).hexdigest(),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "evidence_id": evidence_id, "model_item_index": 3,
        },
        {
            "ts": "2026-08-25T04:06:37", "action": "like",
            "pre_send_evidence_id": evidence_id,
        },
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "latest AUTO opener pre-send evidence:" in md
    assert "session mode: auto" in md  # pre-migration rows remain readable as AUTO
    assert f"evidence ID: `{evidence_id}`" in md
    assert f"snapshot: `{shot}` (verified)" in md
    assert "target: model item `3`" in md
    assert f"full opener: `{opener}` (verified)" in md
    assert "linked outcome: LIKE verified as landed at `2026-08-25T04:06:37`" in md
    assert "human-reviewed approval evidence ID" not in md


def test_debug_log_renders_training_resumed_send_snapshot_and_preserves_approval_link(tmp_path):
    """Training's final Send tap must be tied to its fresh, re-verified frame.

    The earlier frame is still important: it is what the human approved.  The report must keep
    its ID without mistaking it for the final pre-tap snapshot or looking for the result through
    the ordinary AUTO linkage field.
    """
    run = tmp_path / "run_auto_opener_resumed_send_evidence"
    run.mkdir(parents=True)
    opener = "A fresh opener, checked again after the reviewer returned."
    approval_frame = b"the snapshot shown to the reviewer"
    resumed_frame = b"the snapshot immediately before the resumed send tap"
    approval_id = hashlib.sha256(approval_frame + b"\0" + opener.encode()).hexdigest()
    resumed_id = hashlib.sha256(resumed_frame + b"\0" + opener.encode()).hexdigest()
    approval_shot = "00066_auto_opener_pre_send_before.png"
    resumed_shot = "00067_auto_opener_resumed_send_before.png"
    (run / approval_shot).write_bytes(approval_frame)
    (run / resumed_shot).write_bytes(resumed_frame)
    records = [
        {
            "ts": "2026-08-25T04:06:28", "action": "auto_opener_pre_send",
            "before": approval_shot, "opener": opener,
            "opener_sha256": hashlib.sha256(opener.encode()).hexdigest(),
            "frame_sha256": hashlib.sha256(approval_frame).hexdigest(),
            "evidence_id": approval_id, "model_item_index": 3, "session_mode": "training",
        },
        {
            "ts": "2026-08-25T04:07:02", "action": "auto_opener_resumed_send",
            "before": resumed_shot, "opener": opener,
            "opener_sha256": hashlib.sha256(opener.encode()).hexdigest(),
            "frame_sha256": hashlib.sha256(resumed_frame).hexdigest(),
            "evidence_id": resumed_id, "approval_evidence_id": approval_id,
            "model_item_index": 3, "session_mode": "training",
        },
        {
            "ts": "2026-08-25T04:07:10", "action": "like_attempt",
            "pre_send_evidence_id": approval_id,
            "resumed_send_evidence_id": resumed_id,
        },
        {
            "ts": "2026-08-25T04:07:13", "action": "like",
            "pre_send_evidence_id": approval_id,
            "resumed_send_evidence_id": resumed_id,
        },
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    lines = (run / "actions.jsonl").read_text().splitlines()
    evidence = bugreport._latest_auto_opener_evidence_md(lines, run)

    assert f"evidence ID: `{resumed_id}`" in evidence
    assert "session mode: training" in evidence
    assert f"snapshot: `{resumed_shot}` (verified)" in evidence
    assert f"human-reviewed approval evidence ID: `{approval_id}`" in evidence
    assert "linked outcome: LIKE verified as landed at `2026-08-25T04:07:13`" in evidence
    assert approval_shot not in evidence
    assert bugreport._latest_auto_opener_evidence_mode(lines) == "Training"
    report = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert "latest Training opener pre-send evidence:" in report


def test_debug_log_links_training_dislike_without_calling_it_a_missing_send(tmp_path):
    run = tmp_path / "run_training_dislike_evidence"
    run.mkdir(parents=True)
    frame = b"typed opener shown at the training checkpoint"
    opener = "A draft the reviewer chose not to send."
    evidence_id = hashlib.sha256(frame + b"\0" + opener.encode()).hexdigest()
    shot = "00016_auto_opener_pre_send_before.png"
    (run / shot).write_bytes(frame)
    records = [
        {
            "ts": "2026-08-26T21:01:11", "action": "auto_opener_pre_send",
            "before": shot, "opener": opener,
            "opener_sha256": hashlib.sha256(opener.encode()).hexdigest(),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "evidence_id": evidence_id, "model_item_index": 3,
            "session_mode": "training",
        },
        {
            "ts": "2026-08-26T21:02:29", "action": "training_dislike",
            "model_item_index": 3, "advance_proof": "name",
        },
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    evidence = bugreport._latest_auto_opener_evidence_md(
        (run / "actions.jsonl").read_text().splitlines(), run)

    assert "session mode: training" in evidence
    assert "DISLIKE verified as landed" in evidence
    assert "typed opener was not sent or committed" in evidence
    assert "legacy sequence" in evidence
    assert "no linked send outcome was logged" not in evidence


def test_debug_log_classifies_stop_cancelled_training_draft_as_unsent_and_uncommitted(tmp_path):
    """The 2026-08-31 Stop happened after typing, but before any Send Like tap.

    Legacy runs retained that expected ActionCancelled as ``unexpected``.  The report must use
    that existing checkpoint-local evidence instead of making the operator infer an ambiguous
    missing outcome from the absence of a like row.
    """
    run = tmp_path / "run_training_stop_before_send"
    run.mkdir(parents=True)
    frame = b"typed opener awaiting the Hub training decision"
    opener = "A bridge question that remained a draft."
    evidence_id = hashlib.sha256(frame + b"\0" + opener.encode()).hexdigest()
    shot = "00430_auto_opener_pre_send_before.png"
    (run / shot).write_bytes(frame)
    records = [
        {
            "ts": "2026-08-31T02:44:05", "action": "auto_opener_pre_send",
            "before": shot, "opener": opener,
            "opener_sha256": hashlib.sha256(opener.encode()).hexdigest(),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "evidence_id": evidence_id, "model_item_index": 1,
            "session_mode": "training",
        },
        {
            "ts": "2026-08-31T03:00:08", "action": "unexpected",
            "error": ("ActionCancelled: action cancelled because the run is stopping before "
                      "training decision; no further device input was issued"),
        },
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    evidence = bugreport._latest_auto_opener_evidence_md(
        (run / "actions.jsonl").read_text().splitlines(), run)

    assert ("linked outcome: Training decision cancelled by Stop; typed opener was not sent or "
            "committed at `2026-08-31T03:00:08`") in evidence
    assert "no linked send outcome was logged" not in evidence


def test_debug_log_links_current_training_cancellation_action_to_its_presend_evidence(tmp_path):
    """New runs have an explicit checkpoint-local cancellation action, not `unexpected`."""
    run = tmp_path / "run_training_cancellation_evidence"
    run.mkdir(parents=True)
    frame = b"typed opener awaiting a stopped review"
    opener = "A draft that Stop prevented from sending."
    evidence_id = hashlib.sha256(frame + b"\0" + opener.encode()).hexdigest()
    shot = "00430_auto_opener_pre_send_before.png"
    (run / shot).write_bytes(frame)
    records = [
        {
            "ts": "2026-08-31T02:44:05", "action": "auto_opener_pre_send",
            "before": shot, "opener": opener,
            "opener_sha256": hashlib.sha256(opener.encode()).hexdigest(),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "evidence_id": evidence_id, "model_item_index": 1,
            "session_mode": "training",
        },
        {
            "ts": "2026-08-31T03:00:08", "action": "training_cancelled",
            "pre_send_evidence_id": evidence_id, "model_item_index": 1,
            "reason": "stop_requested",
        },
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    evidence = bugreport._latest_auto_opener_evidence_md(
        (run / "actions.jsonl").read_text().splitlines(), run)

    assert ("linked outcome: Training decision cancelled by Stop; typed opener was not sent or "
            "committed at `2026-08-31T03:00:08`") in evidence
    assert "no linked send outcome was logged" not in evidence


def test_debug_log_flags_tampered_auto_opener_evidence(tmp_path):
    run = tmp_path / "run_tampered_auto_opener_evidence"
    run.mkdir(parents=True)
    shot = "00001_auto_opener_pre_send_before.png"
    (run / shot).write_bytes(b"changed frame")
    record = {
        "action": "auto_opener_pre_send", "before": shot, "opener": "changed text",
        "opener_sha256": hashlib.sha256(b"original text").hexdigest(),
        "frame_sha256": hashlib.sha256(b"original frame").hexdigest(),
        "evidence_id": "evidence",
    }
    (run / "actions.jsonl").write_text(json.dumps(record) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert md.count("SHA-256 mismatch") == 2
    assert "no linked send outcome was logged" in md


def test_debug_log_section_excludes_disabled_apps_from_active_run_diagnosis(tmp_path):
    """An old debug folder for a configured-but-disabled app is not this run's evidence."""
    import yaml

    hinge_dir = tmp_path / "hinge_debug"
    bumble_dir = tmp_path / "bumble_debug"
    (hinge_dir / "run").mkdir(parents=True)
    (bumble_dir / "old_run").mkdir(parents=True)
    (hinge_dir / "run" / "actions.jsonl").write_text('{"action": "capture"}\n')
    (bumble_dir / "old_run" / "actions.jsonl").write_text('{"action": "capture"}\n')
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "enabled_apps": ["hinge"],
        "apps": {
            "hinge": {"debug_log": True, "debug_dir": str(hinge_dir)},
            "bumble": {"debug_log": True, "debug_dir": str(bumble_dir)},
        },
    }))

    md = bugreport._debug_log_md(str(path))

    assert "**hinge**" in md
    assert "**bumble**" not in md
    assert str(bumble_dir) not in md


def test_debug_log_section_explains_latest_observe_wait_and_reproduction_context(tmp_path):
    """A vague report must say what the latest evidence actually proves: this card captured,
    then became READY, then received no observed decision -- not merely "waiting"."""
    run = tmp_path / "run_observe_context"
    run.mkdir(parents=True)
    (run / "00002_observe_waiting_before.png").write_bytes(b"screen")
    capture = {"ts": "2026-08-14T13:20:24", "action": "capture", "photos": 38,
               "profile_name": "Anita", "items": 0,
               "items_unavailable": "item index could not be trusted"}
    wait_1 = {"ts": "2026-08-14T13:21:04", "action": "observe_waiting",
              "reason": "no_change"}
    wait_2 = {"ts": "2026-08-14T13:21:19", "action": "observe_waiting",
              "reason": "no_change", "before": "00002_observe_waiting_before.png"}
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, [capture, wait_1, wait_2])) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "latest observe context (logged evidence, not a new phone read):" in md
    assert "final logged observe state: waiting (`no_change`)" in md
    assert "frame has not visibly changed" in md
    assert "final logged wait began at `2026-08-14T13:21:04` and has 2 heartbeat record(s)" in md
    assert "identity read `Anita`; 38 captured photo(s); 0 numbered item(s)" in md
    assert "numbered items unavailable: `item index could not be trusted`" in md
    assert "00002_observe_waiting_before.png` (present)" in md
    assert "capture completed → READY/manual decision prompt → no pass/like record before the final logged wait" in md


def test_debug_log_section_marks_a_stop_after_ready_as_terminal_not_an_open_wait(tmp_path):
    """A stopped run can legitimately end on a READY card.  Its earlier wait evidence must not
    be rendered as an ongoing decision prompt after Worker has discarded that card."""
    run = tmp_path / "run_observe_stopped"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-24T13:11:55", "action": "capture", "profile_name": "Sandra"},
        {"ts": "2026-08-24T13:13:35", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:13:50", "action": "observe_stopped",
         "reason": "stop_requested", "profile_name": "Sandra"},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "latest logged action: `observe_stopped`" in md
    assert "observe wait ended because Stop was requested" in md
    assert "intentionally abandoned" in md
    assert "current logged observe state: waiting" not in md


def test_latest_context_keeps_stop_aborted_capture_terminal_after_trailing_timing(tmp_path):
    """The capture loop writes its timing finally-block after the terminal abort row.

    The timing row is useful diagnostics, but it must not turn a clean Stop into an ambiguous
    ``latest logged action`` report or invent a live device state.
    """
    run = tmp_path / "run_stopped_capture_with_timing"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-24T20:33:21", "action": "capture_aborted", "frames": 2,
         "read_scrolls": 2, "profile_name": "Jenna"},
        {"ts": "2026-08-24T20:33:21", "action": "capture_iteration_timing",
         "frame_index": 1, "exit_reason": "stopped_during_settle"},
        {"ts": "2026-08-24T20:33:21", "action": "capture_timing_summary",
         "iterations": 2},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "latest logged action: `capture_timing_summary`" in md
    assert "terminal capture state: Stop abandoned the in-progress profile read" in md
    assert "no profile capture or decision was recorded" in md
    assert "profile identity `Jenna`; 2 captured frame(s); 2 read scroll(s)" in md
    assert "final logged observe state: waiting" not in md
    assert "current phone" not in md


def test_debug_report_summarises_latest_completed_capture_timing_before_stop(tmp_path):
    """A terminal aborted next read must not hide the previous completed capture's cost."""
    run = tmp_path / "run_capture_timing_before_stop"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-25T01:07:05", "action": "capture_timing_summary",
         "iterations": 12, "iter_wall_s_total": 56.521766},
        {"ts": "2026-08-25T01:10:18", "action": "capture_fold_timing", "photos": 23,
         "fold_wall_s": 193.776865, "still_photo_dwell_s": 181.294415},
        {"ts": "2026-08-25T01:10:18", "action": "capture", "photos": 23,
         "profile_name": "Emma"},
        {"ts": "2026-08-25T01:11:38", "action": "capture_aborted", "frames": 3,
         "read_scrolls": 3, "profile_name": "Elena"},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "latest completed capture timing:" in md
    assert "profile `Emma`; read 56.5s; fold 193.8s" in md
    assert "still-photo safety checks 181.3s (93.6% of fold); total 250.3s" in md


def test_capture_timing_summary_splits_passive_observation_from_navigation_overhead():
    lines = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 76.121569}),
        json.dumps({"action": "capture_fold_timing", "photos": 17,
                    "fold_wall_s": 196.794552, "still_photo_dwell_s": 186.624577,
                    "still_photo_dwell_breakdown": {
                        "passive_observation_s": 51.866999,
                        "navigation_and_overhead_s": 134.757578}}),
        json.dumps({"action": "capture", "photos": 17, "profile_name": "Manon"}),
    ]

    summary = bugreport._latest_completed_capture_timing_md(lines)

    assert "still-photo safety checks 186.6s (94.8% of fold" in summary
    assert "passive observation 51.9s" in summary
    assert "navigation/overhead 134.8s" in summary


def test_capture_timing_summary_reports_only_arithmetic_complete_candidate_rows():
    lines = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 45.0}),
        json.dumps({"action": "still_photo_dwell_walk_candidate_timing",
                    "heart_ordinal": 8, "candidate_wall_s": 40.0,
                    "navigation_s": 10.0, "proof_s": 20.0, "return_s": 8.0,
                    "unattributed_s": 2.0}),
        json.dumps({"action": "still_photo_dwell_walk_candidate_timing",
                    "heart_ordinal": 6, "candidate_wall_s": 61.0,
                    "navigation_s": 20.0, "proof_s": 25.0, "return_s": 15.0,
                    "unattributed_s": 1.0}),
        # A torn row must not be partly added to otherwise valid candidate telemetry.
        json.dumps({"action": "still_photo_dwell_walk_candidate_timing",
                    "candidate_wall_s": 99.0, "navigation_s": 1.0, "proof_s": 1.0,
                    "return_s": 1.0, "unattributed_s": 1.0}),
        json.dumps({"action": "capture_fold_timing", "photos": 19,
                    "fold_wall_s": 146.0, "still_photo_dwell_s": 134.0,
                    "still_photo_dwell_breakdown": {
                        "passive_observation_s": 51.0,
                        "navigation_and_overhead_s": 83.0}}),
        json.dumps({"action": "capture", "photos": 19, "profile_name": "Jordan"}),
    ]

    summary = bugreport._latest_completed_capture_timing_md(lines)

    assert ("2 candidate hop(s): navigation 30.0s; proof 45.0s; return 23.0s; "
            "other 3.0s") in summary


def test_capture_timing_summary_fails_closed_for_unpaired_or_optional_bad_fields():
    """Corrupt timing rows must not be paired across captures or make reports fail."""
    unpaired = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 12}),
        json.dumps({"action": "capture_fold_timing", "photos": 3, "fold_wall_s": 8}),
        json.dumps({"action": "capture", "photos": 4}),
    ]
    assert bugreport._latest_completed_capture_timing_md(unpaired) == ""

    optional_bad_dwell = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 12}),
        json.dumps({"action": "capture_fold_timing", "photos": 4, "fold_wall_s": 8,
                    "still_photo_dwell_s": "unknown"}),
        json.dumps({"action": "capture", "photos": 4}),
    ]
    summary = bugreport._latest_completed_capture_timing_md(optional_bad_dwell)
    assert "read 12.0s; fold 8.0s; total 20.0s" in summary
    assert "still-photo safety checks" not in summary


@pytest.mark.parametrize("bad_line", [
    '{"action": "capture"',
    json.dumps(["not", "an", "action", "record"]),
])
def test_capture_timing_summary_fails_closed_on_invalid_candidate_window_line(bad_line):
    """Unknown raw data between timings and a capture can conceal a lifecycle boundary."""
    lines = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 12}),
        json.dumps({"action": "capture_fold_timing", "photos": 4, "fold_wall_s": 8}),
        bad_line,
        json.dumps({"action": "capture", "photos": 4}),
    ]

    assert bugreport._latest_completed_capture_timing_md(lines) == ""


def test_capture_timing_summary_keeps_best_effort_outside_candidate_window():
    """A later partial append cannot change already-completed capture provenance."""
    lines = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 12}),
        json.dumps({"action": "capture_fold_timing", "photos": 4, "fold_wall_s": 8}),
        json.dumps({"action": "capture", "photos": 4}),
        '{"action": "still-writing"',
    ]

    assert "total 20.0s" in bugreport._latest_completed_capture_timing_md(lines)


def test_capture_timing_summary_keeps_derived_total_and_share_finite():
    """Finite raw values must not become an infinite report value during arithmetic."""
    share_safe = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 1}),
        json.dumps({"action": "capture_fold_timing", "photos": 4, "fold_wall_s": 1e308,
                    "still_photo_dwell_s": 1e308}),
        json.dumps({"action": "capture", "photos": 4}),
    ]
    summary = bugreport._latest_completed_capture_timing_md(share_safe)
    assert "(100.0% of fold)" in summary
    assert "inf" not in summary

    overflowing_total = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 1e308}),
        json.dumps({"action": "capture_fold_timing", "photos": 4, "fold_wall_s": 1e308}),
        json.dumps({"action": "capture", "photos": 4}),
    ]
    assert bugreport._latest_completed_capture_timing_md(overflowing_total) == ""


def test_capture_timing_summary_omits_undefined_zero_fold_share():
    lines = [
        json.dumps({"action": "capture_timing_summary", "iter_wall_s_total": 0}),
        json.dumps({"action": "capture_fold_timing", "photos": 4, "fold_wall_s": 0,
                    "still_photo_dwell_s": 0}),
        json.dumps({"action": "capture", "photos": 4}),
    ]

    summary = bugreport._latest_completed_capture_timing_md(lines)
    assert "still-photo safety checks 0.0s" in summary
    assert "% of fold" not in summary


def test_latest_context_explains_a_persistent_uhid_input_as_completed_but_post_input_unknown(tmp_path):
    """A trace can end directly after the persistent transport returns.

    `device_input` is written after delivery, so the report must preserve that limited proof
    and name the useful scalar context without inventing an unfinished gesture or its cause.
    """
    run = tmp_path / "run_persistent_uhid_tail"
    run.mkdir(parents=True)
    final_input = {
        "ts": "2026-08-24T20:17:58", "action": "device_input",
        "transport": "PersistentUhidTouch", "kind": "scroll",
        "source": "_scroll_down_one", "direction": "forward",
    }
    (run / "actions.jsonl").write_text(json.dumps(final_input) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "latest logged action: `device_input` at `2026-08-24T20:17:58`" in md
    assert "final completed device input:" in md
    assert "transport=`PersistentUhidTouch`" in md
    assert "kind=`scroll`" in md
    assert "source=`_scroll_down_one`" in md
    assert "direction=`forward`" in md
    assert "time=`2026-08-24T20:17:58`" in md
    assert "logging ended after that completed input" in md
    assert "cannot pinpoint what subsequently blocked progress" in md
    assert "incomplete gesture" not in md


def test_latest_context_keeps_final_input_metadata_to_safe_scalars(tmp_path):
    run = tmp_path / "run_malformed_input_tail"
    run.mkdir(parents=True)
    final_input = {
        "ts": ["not", "a", "time"], "action": "device_input",
        "transport": {"transport": "do not render"}, "kind": ["scroll"],
        "source": None, "direction": {"direction": "forward"},
    }
    (run / "actions.jsonl").write_text(json.dumps(final_input) + "\n")

    md = bugreport._latest_observe_context_md([json.dumps(final_input)], run)

    assert "transport=`not recorded`" in md
    assert "kind=`not recorded`" in md
    assert "source=`not recorded`" in md
    assert "direction=`not recorded`" in md
    assert "time=`not recorded`" in md
    assert "do not render" not in md
    assert "['not', 'a', 'time']" not in md


def test_debug_log_section_distinguishes_items_unnumbered_from_items_unavailable(tmp_path):
    """`items_unnumbered` (enumeration ran to completion and legitimately numbered nothing,
    e.g. every card was a still-photo-discriminator video) must read as a normal outcome, not
    as the failure `items_unavailable` describes -- see Profile's three-state docstring and
    ops/STILL-PHOTO-DISCRIMINATOR.md 5d. This is the sibling of the `items_unavailable` case
    covered above; the wording must never overlap so a reader cannot mistake one for the other.
    """
    run = tmp_path / "run_observe_context_unnumbered"
    run.mkdir(parents=True)
    capture = {"ts": "2026-08-14T13:20:24", "action": "capture", "photos": 38,
               "profile_name": "Anita", "items": 0,
               "items_unnumbered": "12 selectable card(s) were considered; all excluded."}
    wait = {"ts": "2026-08-14T13:21:04", "action": "observe_waiting", "reason": "no_change"}
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, [capture, wait])) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert ("identity read `Anita`; 38 captured photo(s); 0 numbered item(s); enumeration "
            "completed but numbered nothing: `12 selectable card(s) were considered; all "
            "excluded.`" in md)
    assert "numbered items unavailable" not in md
    assert "items unavailable" not in md


def test_debug_log_section_explains_like_candidate_without_claiming_a_sheet(tmp_path):
    """A bottom-delta candidate is specifically NOT evidence that a composer opened or
    closed. The report must preserve that distinction from `like_sending`."""
    run = tmp_path / "run_like_candidate"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-14T21:54:36", "action": "observe_waiting",
         "reason": "like_candidate"},
        {"ts": "2026-08-14T21:54:51", "action": "observe_waiting",
         "reason": "like_candidate"},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "final logged observe state: waiting (`like_candidate`)" in md
    assert "bottom-only screen change looked like a possible like" in md
    assert "no Send Like sheet was observed" in md
    assert "sheet closed" not in md


def test_debug_log_section_names_an_abandoned_card_resync(tmp_path):
    """The 2026-08-15 report ("it moved on again without waiting for my like or dislike") was
    filed against an `observe_resync` record that existed only in the raw actions.jsonl tail --
    nothing in the report named it. This is that record, taken from hinge.py's
    like_candidate_without_observed_sheet call site, and the report must now name it, the
    profile it happened to, and the identity verdicts that decided it -- not just leave it for a
    developer to spot in the tail.

    The identity verdicts here are `unknown`/`new` rather than `same` on purpose: that call site
    is now only reachable when the anchor could NOT name the card as the captured profile (a
    'same' verdict keeps the wait alive instead -- see hinge.py's `require_content` comment), so
    a fixture pinning `identity: same` would be pinning a record production can no longer
    write."""
    run = tmp_path / "run_abandoned_card"
    run.mkdir(parents=True)
    resync = {
        "ts": "2026-08-15T09:12:03", "action": "observe_resync",
        "reason": "like_candidate_without_observed_sheet",
        "sheet_seen": False, "profile_name": "Jamie",
        "identity": "unknown", "confirm_identity": "new",
        "current": False, "deck_ready": True,
    }
    (run / "actions.jsonl").write_text(json.dumps(resync) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "cards abandoned without a decision (resync):" in md
    assert "2026-08-15T09:12:03" in md
    assert "reason=`like_candidate_without_observed_sheet`" in md
    assert "profile_name=`Jamie`" in md
    assert "identity=`unknown`" in md and "confirm_identity=`new`" in md
    assert "sheet_seen=`false`" in md
    assert "no Send Like sheet was ever observed" in md


def test_debug_log_section_names_a_pass_identity_name_unconfirmed_resync(tmp_path):
    """Filed against the 2026-08-20 report ("i didn't press like or dislike, but app proceeded
    forward"): hinge.py's pass-path resync at the `verdict == "no_data" and not
    name_advance_proven` call site sets `reason="pass_identity_name_unconfirmed"`, which this
    dict did not yet explain -- the report rendered "unrecognised resync reason (no explanation
    on file for it yet)" instead of telling the owner why the card advanced with no logged
    decision. It must now explain that observe_touch_watch is off for Hinge by design (Android
    withholds the touch stream on this device) and that OCR failed to corroborate a new name on
    both frames, so nothing was mislabelled -- the card was recaptured instead."""
    run = tmp_path / "run_pass_identity_name_unconfirmed"
    run.mkdir(parents=True)
    resync = {
        "ts": "2026-08-20T03:33:16", "action": "observe_resync",
        "reason": "pass_identity_name_unconfirmed",
        "identity": "new", "identity_name_read": "ov",
        "confirm_identity": "new", "confirm_identity_name_read": "ov",
        "profile_name": "Elly", "gesture": "no_data", "watcher": False,
    }
    (run / "actions.jsonl").write_text(json.dumps(resync) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "cards abandoned without a decision (resync):" in md
    assert "reason=`pass_identity_name_unconfirmed`" in md
    assert "profile_name=`Elly`" in md
    assert "identity=`new`" in md and "confirm_identity=`new`" in md
    assert "gesture=`no_data`" in md
    assert "unrecognised resync reason" not in md
    assert "observe_touch_watch is off for Hinge" in md
    assert "OCR could not positively read the same new name" in md


def test_debug_report_attributes_foreground_interruption_to_last_completed_input(tmp_path):
    """The 2026-08-21 report showed the shade but had no outbound-input timeline.

    A future report must say when the driver's last gesture completed, explicitly confirm the
    quiet interval while System UI was foreground, and ignore ordinary recapture input that
    happens only after Hinge returns.
    """
    run = tmp_path / "run_foreground_pause"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-21T01:37:45", "action": "device_input", "kind": "swipe",
         "source": "_scroll_to_top_unlocked", "start": [540, 264], "end": [540, 2136]},
        {"ts": "2026-08-21T01:38:41", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-21T01:38:48", "action": "observe_foreground_paused",
         "package": "com.android.systemui", "reason": "notification shade"},
        {"ts": "2026-08-21T01:39:10", "action": "observe_foreground_resumed",
         "package": "co.hinge.app", "result": "recapture_without_decision"},
        {"ts": "2026-08-21T01:39:15", "action": "device_input", "kind": "scroll",
         "source": "_capture_current"},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "Android foreground/input attribution:" in md
    assert "foreground interruption at `2026-08-21T01:38:48`: `com.android.systemui`" in md
    assert "last completed Operation Love device input: `swipe`" in md
    assert "`_scroll_to_top_unlocked`" in md
    assert "no Operation Love device input was logged after the interruption" in md
    assert "Hinge regained foreground at `2026-08-21T01:39:10`" in md
    assert "recaptured without recording a decision" in md
    assert "⚠️ 1 Operation Love device input" not in md


def test_debug_log_section_is_quiet_when_no_card_was_ever_abandoned(tmp_path):
    """A run where every card ended in a proven capture/decision must render nothing extra --
    same 'quiet when healthy' contract every other debug-log section here follows."""
    run = tmp_path / "run_healthy"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-15T09:00:00", "action": "capture", "profile_name": "Riley"},
        {"ts": "2026-08-15T09:00:20", "action": "observe_decision", "decision": "pass"},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "cards abandoned without a decision (resync):" not in md


def test_debug_log_section_handles_gesture_path_resync_without_a_reason_key(tmp_path):
    """hinge.py's PASS-path resync (the `_dbg_action("observe_resync", base, **fields)` call
    right after `if verdict == "resync":`) never sets a `reason` key at all -- only `gesture`.
    That must render as a labelled, sensible bullet (a card that changed without the human's
    touch stream corroborating a decision), never as a blank or `None` reason."""
    run = tmp_path / "run_gesture_resync"
    run.mkdir(parents=True)
    resync = {
        "ts": "2026-08-15T09:20:00", "action": "observe_resync",
        "top": 0.1, "bot": 12.3, "identity": "new", "confirm_identity": "new",
        "profile_name": "Alex", "gesture": "resync", "watcher": "healthy",
    }
    (run / "actions.jsonl").write_text(json.dumps(resync) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "cards abandoned without a decision (resync):" in md
    assert "gesture=`resync`" in md
    assert "profile_name=`Alex`" in md
    assert "reason=`" not in md                     # no reason key was logged -- must not invent one
    assert "touch stream did not corroborate" in md
    assert "None" not in md


def test_debug_log_section_renders_an_unknown_resync_reason_without_swallowing_it(tmp_path):
    """A future reason string this dict doesn't know about yet must still get its own bullet
    with the raw reason -- never silently dropped, never a crash."""
    run = tmp_path / "run_unknown_reason"
    run.mkdir(parents=True)
    resync = {
        "ts": "2026-08-15T09:30:00", "action": "observe_resync",
        "reason": "some_future_reason_nobody_has_seen_yet", "profile_name": "Sam",
    }
    (run / "actions.jsonl").write_text(json.dumps(resync) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "cards abandoned without a decision (resync):" in md
    assert "reason=`some_future_reason_nobody_has_seen_yet`" in md


def test_debug_log_section_survives_malformed_jsonl_around_a_resync(tmp_path):
    """A malformed/partial line elsewhere in actions.jsonl (a live in-progress append is the
    common real cause) must never break this section -- same contract `_action_records`
    guarantees for every other summary here."""
    run = tmp_path / "run_malformed"
    run.mkdir(parents=True)
    good = {"ts": "2026-08-15T09:12:03", "action": "observe_resync",
            "reason": "like_candidate_without_observed_sheet", "profile_name": "Jamie"}
    (run / "actions.jsonl").write_text(
        "{not valid json\n" + json.dumps(good) + "\n" + "\n" + '{"incomplete":\n')

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})   # must not raise

    assert "cards abandoned without a decision (resync):" in md
    assert "profile_name=`Jamie`" in md


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
        "item_index_runtime": {
            "algorithm_id": "bounded-card-split-v3",
            "module_path": "/repo/operation_love/drivers/item_index.py",
            "indexer_code_sha256": "b" * 64,
            "splitter_code_sha256": "a" * 64,
        },
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
    assert "algorithm `bounded-card-split-v3`" in md
    assert "loaded module `/repo/operation_love/drivers/item_index.py`" in md
    assert "in-memory indexer `bbbbbbbbbbbb`" in md
    assert "in-memory splitter `aaaaaaaaaaaa`" in md


def test_debug_log_separates_dwell_navigation_overshoot_from_index_refusal(tmp_path):
    """A complete item index can later lose its dwell-walk anchor.  The report must name the
    navigation loop (and its actual plan/measurement), not invent a broken index frame pair."""
    run = tmp_path / "run_dwell_navigation_refusal"
    run.mkdir(parents=True)
    records = [
        {"action": "still_photo_dwell_walk_candidate", "heart_ordinal": 6,
         "outcome": "navigation_refused_return_unverified", "reason": "scroll_overshot",
         "navigation_refusal": {
             "schema_version": 1, "code": "scroll_overshot", "frame_index": 1,
             "return_outcome": "unverified", "restored_page_shift_px": None,
             "planned": {"step_px": 292, "bound_px": 292, "spacing_px": 812,
                         "sized_against_px": 812, "frac": 0.13, "window_px": [219, 292],
                         "basis": "card_extent"},
             "achieved": {"measurement_status": "measured", "measurement_delta_px": -500,
                          "climb_px": 500, "overshoot_px": 208,
                          "measurement_confidence": 0.98, "measurement_agreeing": 6,
                          "measurement_dissenting": 0, "measurement_eligible": 6},
         }},
        # The normal terminal envelope is deliberately present too: this must not cause the
        # renderer to call it an item-index correspondence failure.
        {"action": "item_index_refused", "reason": "the dwell return is unsafe",
         "steps_px": [257, 455, 540], "refused_pairs": []},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "dwell-navigation refusals (separate from item-index correspondence):" in md
    assert "page heart 6 at navigation frame 1" in md
    assert "dwell-navigation refusal `scroll_overshot`" in md
    assert "step 292px; bound 292px; spacing 812px; sized against 812px" in md
    assert "window 219..292px; basis `card_extent`" in md
    assert "shift -500px; climb 500px; overshoot 208px; estimator `measured`" in md
    assert "anchor return `unverified`" in md
    assert "no specific failing pair recorded" in md  # honest index summary, not a false pair


def test_dwell_navigation_refusal_summary_tolerates_legacy_and_malformed_rows(tmp_path):
    run = tmp_path / "run_dwell_navigation_legacy"
    run.mkdir(parents=True)
    records = [
        {"action": "still_photo_dwell_walk_candidate", "heart_ordinal": 2,
         "outcome": "navigation_refused", "reason": "scroll_overshot"},
        {"action": "still_photo_dwell_walk_candidate", "heart_ordinal": True,
         "outcome": "return_unverified", "navigation_refusal": {
             "code": "bad`\n# fake heading", "frame_index": True,
             "planned": {"step_px": True, "window_px": ["bad", None]},
             "achieved": {"climb_px": "bad"}, "return_outcome": ["bad"]}},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "page heart 2: dwell-navigation navigation_refused (`scroll_overshot`)" in md
    assert "legacy/incomplete trace has no structured plan" in md
    assert "unknown page heart: dwell-navigation refusal `bad' # fake heading`" in md
    assert "planned step telemetry unavailable" in md
    assert not any(line.strip().startswith("# fake heading") for line in md.splitlines())


def test_dwell_navigation_chain_refusal_reports_pair_and_both_estimator_directions(tmp_path):
    """A chain refusal must no longer collapse to a code after its frames disappear."""
    run = tmp_path / "run_dwell_navigation_chain_refusal"
    run.mkdir(parents=True)
    record = {
        "action": "still_photo_dwell_walk_candidate", "heart_ordinal": 8,
        "outcome": "navigation_refused", "reason": "chain_broken",
        "before": "00100_still_photo_dwell_walk_candidate_before.png",
        "after": "00101_still_photo_dwell_walk_candidate_after.png",
        "navigation_refusal": {
            "schema_version": 2, "code": "chain_broken", "frame_index": 2,
            "return_outcome": "unavailable_unmeasured", "restored_page_shift_px": None,
            "planned": {"step_px": 300, "bound_px": 360, "spacing_px": 1027,
                        "sized_against_px": 1027, "window_px": [219, 360],
                        "basis": "measured"},
            "achieved": {
                "measurement_status": "no_consensus",
                "measurement_reason": "forward video strips split",
                "reverse": {"status": "no_consensus", "delta_px": None,
                            "reason": "reverse video strips also split"},
            },
        },
    }
    (run / "actions.jsonl").write_text(json.dumps(record) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    row = next(line for line in md.splitlines() if "page heart 8" in line)
    assert "dwell-navigation refusal `chain_broken`" in row
    assert "estimator `no_consensus`" in row
    assert "reason `forward video strips split`" in row
    assert "reverse estimator `no_consensus; reverse video strips also split`" in row
    assert "anchor return `unavailable_unmeasured`" in row
    assert "before `00100_still_photo_dwell_walk_candidate_before.png`" in row
    assert "after `00101_still_photo_dwell_walk_candidate_after.png`" in row


def test_dwell_navigation_refusal_summary_shows_walk_continue_and_abandon_outcomes(tmp_path):
    """2026-08-27: a verified return can let the still-photo candidate walk continue past a
    refusal instead of abandoning it, bounded by a small returned-refusal budget.  The report
    must say which of the three walk outcomes happened and, for the two abandoned ones -- lost
    still-photo coverage -- must flag it with the warning glyph the continued case does not
    get."""
    run = tmp_path / "run_dwell_navigation_walk_outcomes"
    run.mkdir(parents=True)

    def refusal_record(heart_ordinal, walk):
        return {
            "action": "still_photo_dwell_walk_candidate", "heart_ordinal": heart_ordinal,
            "outcome": "navigation_refused_returned", "reason": "scroll_overshot",
            "navigation_refusal": {
                "schema_version": 1, "code": "scroll_overshot", "frame_index": 2,
                "return_outcome": "verified", "restored_page_shift_px": 0,
                "planned": {"step_px": 292, "bound_px": 292, "spacing_px": 812,
                            "sized_against_px": 812, "frac": 0.13, "window_px": [219, 292],
                            "basis": "card_extent"},
                "achieved": {"measurement_status": "measured", "measurement_delta_px": -500,
                             "climb_px": 500, "overshoot_px": 208,
                             "measurement_confidence": 0.98, "measurement_agreeing": 6,
                             "measurement_dissenting": 0, "measurement_eligible": 6},
                "walk": walk,
            },
        }

    records = [
        refusal_record(1, {"outcome": "continued", "returned_refusals_spent": 1,
                            "returned_refusal_budget": 2}),
        refusal_record(2, {"outcome": "abandoned_budget_spent", "returned_refusals_spent": 3,
                            "returned_refusal_budget": 2}),
        refusal_record(3, {"outcome": "abandoned_return_unverified", "returned_refusals_spent": 0,
                            "returned_refusal_budget": 2}),
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    continued_line = next(line for line in md.splitlines() if "page heart 1" in line)
    budget_line = next(line for line in md.splitlines() if "page heart 2" in line)
    unverified_line = next(line for line in md.splitlines() if "page heart 3" in line)

    assert ("walk continued to the next candidate (returned-refusal budget 1/2)"
            in continued_line)
    assert "⚠️" not in continued_line

    assert ("⚠️ walk abandoned: returned-refusal budget spent "
            "(returned-refusal budget 3/2)" in budget_line)

    assert ("⚠️ walk abandoned: return unverified "
            "(returned-refusal budget 0/2)" in unverified_line)


def test_dwell_navigation_refusal_summary_omits_walk_clause_without_walk_key(tmp_path):
    """Every historic run -- and the legacy/incomplete branch above it -- predates the `walk`
    telemetry entirely.  Those rows must render byte-identical to before the budget existed: no
    crash, no invented walk clause, no warning glyph."""
    run = tmp_path / "run_dwell_navigation_no_walk_key"
    run.mkdir(parents=True)
    records = [
        {"action": "still_photo_dwell_walk_candidate", "heart_ordinal": 6,
         "outcome": "navigation_refused_return_unverified", "reason": "scroll_overshot",
         "navigation_refusal": {
             "schema_version": 1, "code": "scroll_overshot", "frame_index": 1,
             "return_outcome": "unverified", "restored_page_shift_px": None,
             "planned": {"step_px": 292, "bound_px": 292, "spacing_px": 812,
                         "sized_against_px": 812, "frac": 0.13, "window_px": [219, 292],
                         "basis": "card_extent"},
             "achieved": {"measurement_status": "measured", "measurement_delta_px": -500,
                          "climb_px": 500, "overshoot_px": 208,
                          "measurement_confidence": 0.98, "measurement_agreeing": 6,
                          "measurement_dissenting": 0, "measurement_eligible": 6},
         }},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    row = next(line for line in md.splitlines() if "page heart 6" in line)
    assert row.endswith("anchor return `unverified`")
    assert "walk" not in row
    assert "⚠️" not in row


def test_debug_log_summarises_structured_dwell_return_chain_refusal(tmp_path):
    """The return helper's row explains a proved candidate's failed cleanup without blaming
    the already-complete enumeration index or promoting the planned stroke to a measurement.
    """
    run = tmp_path / "run_dwell_return_chain"
    run.mkdir(parents=True)
    records = [
        # Successful cleanup rows are routine and deliberately stay out of a failure report.
        {"action": "still_photo_dwell_return_chain", "outcome": "returned",
         "reason": "direct_remeasure_matched", "attempts": 1,
         "initial_terminal_shift_px": -450, "terminal_shift_px": 0,
         "direct_shift_px": 0, "drift_bound_px": 219},
        {"action": "still_photo_dwell_return_chain", "outcome": "refused",
         "reason": "return_leg_unmeasurable", "attempts": 4,
         "initial_terminal_shift_px": -4243, "terminal_shift_px": -2353,
         "drift_bound_px": 219,
         "before": "00011_still_photo_dwell_return_chain_before.png",
         "after": "00012_still_photo_dwell_return_chain_after.png"},
    ]
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "still-photo return-chain refusals (measured cleanup after dwell):" in md
    row = next(line for line in md.splitlines() if "return chain refused" in line)
    assert "`return_leg_unmeasurable`" in row
    assert "after 4 return attempt(s)" in row
    assert "initial entry-relative shift -4243px" in row
    assert "last known entry-relative shift -2353px" in row
    assert "return gate <219px" in row
    assert "evidence before `00011_still_photo_dwell_return_chain_before.png`" in row
    assert "after `00012_still_photo_dwell_return_chain_after.png`" in row
    assert "final entry-relative shift" not in row
    assert "direct_remeasure_matched" not in bugreport._dwell_return_chain_refusal_summary_md(
        [json.dumps(record) for record in records])


def test_dwell_return_chain_summary_tolerates_missing_and_malformed_fields(tmp_path):
    """Old runs omit the action entirely; partly written future rows remain a safe, useful hint."""
    run = tmp_path / "run_dwell_return_chain_legacy"
    run.mkdir(parents=True)
    records = [
        {"action": "still_photo_dwell_return_chain", "outcome": "refused",
         "reason": "bad`\n# fake heading", "attempts": True,
         "initial_terminal_shift_px": "unknown", "terminal_shift_px": True,
         "drift_bound_px": None, "direct_shift_px": False},
        {"action": "still_photo_dwell_return_chain", "outcome": "not_a_failure",
         "reason": "must_not_render"},
    ]
    (run / "actions.jsonl").write_text("{partial\n" +
                                         "\n".join(map(json.dumps, records)) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "return chain refused (`bad' # fake heading`)" in md
    assert "numeric return telemetry unavailable" in md
    assert "must_not_render" not in bugreport._dwell_return_chain_refusal_summary_md(
        [json.dumps(record) for record in records])
    assert not any(line.strip().startswith("# fake heading") for line in md.splitlines())


def test_compact_debug_tail_line_truncates_long_items_unnumbered_like_items_unavailable():
    """The raw tail compactor already replaces an oversized `items_unavailable` geometry wall
    with a pointer (see the long-reason test below). `items_unnumbered` -- the different,
    non-failure state-3 reason (Profile's docstring) -- can grow just as long when many
    selectable cards each carry their own exclusion reason, and deserves the identical
    treatment rather than being left as an unformatted wall just because its field name
    differs. Compacting one field must never touch or invent the other."""
    long_reason = "this capture's item index carries " + "many selectable card(s); " * 40
    raw_unnumbered = json.dumps({"action": "capture", "items": 0,
                                 "items_unnumbered": long_reason})
    raw_unavailable = json.dumps({"action": "capture", "items": 0,
                                  "items_unavailable": long_reason})

    compacted_unnumbered = json.loads(bugreport._compact_debug_tail_line(raw_unnumbered))
    compacted_unavailable = json.loads(bugreport._compact_debug_tail_line(raw_unavailable))

    expected = bugreport._compact_item_index_refusal_text(long_reason)
    assert compacted_unnumbered["items_unnumbered"] == expected
    assert compacted_unnumbered["items_unnumbered"] != long_reason
    assert "full geometry is in the item-index summary below" in compacted_unnumbered["items_unnumbered"]
    assert "items_unavailable" not in compacted_unnumbered
    assert "items_unnumbered" not in compacted_unavailable


def test_compact_debug_tail_line_leaves_a_short_items_unnumbered_untouched():
    raw = json.dumps({"action": "capture", "items": 0,
                       "items_unnumbered": "1 selectable card(s) were considered; all excluded."})
    compacted = json.loads(bugreport._compact_debug_tail_line(raw))
    assert compacted["items_unnumbered"] == "1 selectable card(s) were considered; all excluded."


def test_item_index_summary_keeps_long_geometry_once_and_separates_trailing_saturation(tmp_path):
    """The full refusal is useful once; repeating it in capture context and raw tail turns a
    report into a wall.  A final clamp step is also not evidence of unstable mid-run spacing."""
    run = tmp_path / "run_item_index_compact"
    run.mkdir(parents=True)
    reason = ("the item index this capture produced contradicts itself: frame 19 sees page rows "
              "6368..6709 while frame 20 bounded page rows 6368..6503; " + "geometry detail " * 80)
    capture = {"action": "capture", "photos": 37, "items": 0, "items_unavailable": reason}
    refusal = {
        "action": "item_index_refused", "reason": reason,
        "steps_px": [240, 250, 245, 15],
        "evidence_frames": ["item_index_refused_deadbeef_frame_19.png"],
        "evidence_sidecar": "item_index_refused_deadbeef_evidence.json",
    }
    repaired = {"action": "item_index_repaired", "notes": [
        "source frame 20 (index frame 19)'s sighting spans the proven card boundary",
        "source frame 20 (index frame 19)'s sighting spans the proven card boundary",
    ]}
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, [capture, refusal, repaired])) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert md.count(bugreport._sanitize_inline(reason)) == 1
    assert "distinct page regions 6368..6503, 6368..6709" in md
    assert "cited frames 19, 20" in md
    assert "saved capture evidence `item_index_refused_deadbeef_frame_19.png`" in md
    assert "geometry sidecar `item_index_refused_deadbeef_evidence.json`" in md
    assert "main realised cadence (3 measured): min 240px, median 245px, max 250px" in md
    assert "trailing scroll saturation at step 3 was 15px" in md
    assert "item-index conservative repairs:" in md
    assert md.count("source frame 20 (index frame 19)") == 1


def test_item_index_summary_names_a_multi_step_trailing_saturation_run(tmp_path):
    """The real incident this guards against: a profile's scroll reached the card's bottom at
    step 36, and the capture kept swiping a page that could not move for the rest of the run
    (measured 0px steps interleaved with refused pairs). The old code only ever recognised the
    single LAST step as saturation and folded everything else -- including 21 measured 0px
    steps -- into "realised steps", reporting a nonsensical `min 0px` cadence and mislabelling
    the run's own head as scattered "mid-run small-step anomalies". The whole contiguous run
    must be named as one fact, and excluded wholesale from the cadence stats.
    """
    run = tmp_path / "run_item_index_trailing_saturation"
    run.mkdir(parents=True)
    steps_px = [265, 302, 268, 256, 270, 266, 263, 233, 231, 221, 247, 232, 274, 243, 262, 266,
                234, 239, 218, 226, 279, 274, 236, 249, 232, 280, 281, 243, 279, 223, 256, 252,
                274, 224, 223, 231, None, 0, 0, 0, 0, 0, 0, None, 0, 0, None, 0, 0, 0, 0, 0, 0,
                None, 0, 0, 0, None, 0, 0, None, 0, 0]
    assert len(steps_px) == 63
    refusal = {"action": "item_index_refused", "reason": "no trustworthy shift",
               "failing_pair": [61, 62], "steps_px": steps_px}
    (run / "actions.jsonl").write_text(json.dumps(refusal) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert ("scroll saturated from step 36 to the end of the capture (21 measured steps at "
            "0px, 6 refused): the profile's bottom was reached there and the remaining frames "
            "repeat the same page position") in md
    assert "mid-run small-step anomal" not in md
    assert "main realised cadence (36 measured): min 218px, median 250.5px, max 302px" in md
    # The old bug's signature: folding the trailing zeros into the cadence produced `min 0px`.
    assert "min 0px" not in md
    assert "trailing scroll saturation at step 62" not in md


def test_item_index_summary_still_pins_a_single_short_final_gesture(tmp_path):
    """A lone clamped final gesture (one near-zero measured step at the very end, nothing else
    saturated) must keep reading exactly as it always has -- the multi-step run wording is only
    for runs of two or more measured near-zero steps."""
    run = tmp_path / "run_item_index_single_clamp"
    run.mkdir(parents=True)
    refusal = {"action": "item_index_refused", "reason": "no trustworthy shift",
               "steps_px": [240, 250, 245, 15]}
    (run / "actions.jsonl").write_text(json.dumps(refusal) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "main realised cadence (3 measured): min 240px, median 245px, max 250px" in md
    assert "trailing scroll saturation at step 3 was 15px" in md
    assert "scroll saturated from step" not in md
    assert "mid-run small-step anomal" not in md


def test_item_index_summary_still_flags_genuine_mid_run_small_step_anomalies(tmp_path):
    """A small step surrounded by a healthy cadence on both sides -- including a healthy tail --
    is a real spacing anomaly, not saturation, and must keep being called out as one."""
    run = tmp_path / "run_item_index_mid_run_anomaly"
    run.mkdir(parents=True)
    refusal = {"action": "item_index_refused", "reason": "no trustworthy shift",
               "steps_px": [260, 255, 40, 270, 265, 258, 262]}
    (run / "actions.jsonl").write_text(json.dumps(refusal) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "realised steps (7 measured): min 40px, median 260px, max 270px" in md
    assert "mid-run small-step anomaly at step 2=40px" in md
    assert "scroll saturated from step" not in md
    assert "trailing scroll saturation" not in md


def test_item_index_summary_reports_plain_cadence_with_no_saturation(tmp_path):
    """A healthy capture with no trailing saturation and no anomalies reports only the plain
    cadence, unchanged from today's behaviour."""
    run = tmp_path / "run_item_index_healthy"
    run.mkdir(parents=True)
    refusal = {"action": "item_index_refused", "reason": "no trustworthy shift",
               "steps_px": [260, 255, 270, 265, 258]}
    (run / "actions.jsonl").write_text(json.dumps(refusal) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "realised steps (5 measured): min 255px, median 260px, max 270px" in md
    assert "scroll saturated from step" not in md
    assert "trailing scroll saturation" not in md
    assert "mid-run small-step anomal" not in md


@pytest.mark.parametrize("runtime_key", ("item_index_runtime", "runtime"))
def test_item_index_repair_summary_reads_v12_runtime_with_legacy_fallback(runtime_key):
    """Repair telemetry changed keys in v12 without making old runs opaque."""
    summary = bugreport._item_index_repair_summary_md([json.dumps({
        "action": "item_index_repaired",
        runtime_key: {"algorithm_id": "bounded-card-split-v13"},
        "repairs": [{
            "path": "v12_mute_card_track",
            "local_pair": [1, 2], "source_pair": [7, 9],
            "raw": {"status": "no_consensus", "delta_px": None},
            "effective": {"status": "measured", "delta_px": 234},
            "mute_markers": [{"local_frame_index": 1, "source_frame_index": 7,
                              "x": 106, "y": 876, "score": 1.0}],
        }],
    })])

    assert "bounded-card-split-v13: 'v12_mute_card_track'" in summary
    assert "source frames 7→9" in summary
    assert "raw no_consensus Nonepx → effective measured 234px" in summary


def test_debug_report_surfaces_item_numbering_manifest_and_compacts_raw_tail(tmp_path):
    """A count like 1 item / 10 context crops cannot reveal that a late crop was densely
    renamed item 1.  Preserve the model-to-heart/page mapping without duplicating the full
    manifest in the raw JSON tail.
    """
    run = tmp_path / "run_item_manifest"
    run.mkdir(parents=True)
    capture = {
        "ts": "t0", "action": "capture", "photos": 36, "items": 1,
        "item_context": 1, "item_translation": [7],
        "item_manifest": [
            {"kind": "context", "model_item": None, "heart_ordinal": 1,
             "source_frame_index": 2, "page_rows": [300, 995],
             "crop_size": [974, 695], "crop_sha256": "aaaa1111bbbb2222",
             "reason": "photo_only: rectangular crop demoted"},
            {"kind": "item", "model_item": 1, "heart_ordinal": 7,
             "source_frame_index": 31, "page_rows": [7000, 7974],
             "crop_size": [974, 974], "crop_sha256": "cccc3333dddd4444",
             "selection_evidence": {
                 "classifier_id": "hinge_crop_type_v2", "classification": "photo",
                 "colour_std": 61.25, "dominant_background": 0.14,
                 "edge_density": 0.31, "large_uniform_panel": False,
                 "text_layout": False,
             },
             "reason": "item 1 (heart 7 on the page)"},
        ],
    }
    (run / "actions.jsonl").write_text(json.dumps(capture) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "item-numbering manifest (non-image capture provenance):" in md
    assert "model item → page heart translation: `[7]`" in md
    assert "unnumbered context: page heart 1; source frame 2; page rows 300..995" in md
    assert "crop 974x695" in md and "rectangular crop demoted" in md
    assert "model item 1: page heart 7; source frame 31; page rows 7000..7974" in md
    assert ("classifier `hinge_crop_type_v2, photo, std=61.25, background=0.14, "
            "edges=0.31, panel=false, text=false`" in md)
    assert '"item_manifest": "see item-numbering manifest above"' in md
    assert md.count("aaaa1111bbbb2222") == 1


def test_debug_report_marks_prior_manifest_when_latest_capture_refused(tmp_path):
    """A prior card's item table is useful evidence, never the current refused card's table."""
    run = tmp_path / "run_prior_item_manifest"
    run.mkdir(parents=True)
    refusal_reason = ("the item index this capture produced contradicts itself: frames 7 and 8 "
                      "could not be put in one coordinate space; " + "geometry detail " * 80)
    successful = {
        "ts": "2026-08-15T06:10:25", "action": "capture", "profile_name": "Winnie",
        "items": 1, "item_translation": [1],
        "item_manifest": [{
            "kind": "item", "model_item": 1, "heart_ordinal": 1,
            "source_frame_index": 0, "page_rows": [553, 1527],
            "crop_size": [974, 974], "crop_sha256": "prior-winnie-crop",
            "reason": "item 1 (heart 1 on the page)",
        }],
    }
    refused = {
        "ts": "2026-08-15T06:17:55", "action": "capture", "profile_name": "Shannon",
        "items": 0, "item_translation": [], "item_manifest": [],
        "items_unavailable": refusal_reason,
    }
    (run / "actions.jsonl").write_text(
        json.dumps(successful) + "\n"
        + json.dumps({"action": "item_index_refused", "reason": refusal_reason}) + "\n"
        + json.dumps(refused) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert ("prior successful capture at `2026-08-15T06:10:25` for profile `Winnie`" in md)
    assert ("latest capture at `2026-08-15T06:17:55` for profile `Shannon` had no numbered "
            "manifest; items unavailable: `" +
            bugreport._compact_item_index_refusal_text(refusal_reason) + "`" in md)
    assert "model item → page heart translation: `[1]`" in md
    assert "prior-winnie-crop" in md
    assert md.count(bugreport._sanitize_inline(refusal_reason)) == 1
    assert "full geometry is in the item-index summary below" in md
    # Only the successful capture points at the summary; the refused capture's empty manifest
    # remains raw evidence of why its own item table is absent.
    assert md.count('"item_manifest": "see item-numbering manifest above"') == 1
    assert '"item_manifest": []' in md


def test_debug_report_marks_prior_manifest_when_latest_capture_numbered_nothing(tmp_path):
    """An empty manifest is not always a refusal: `items_unnumbered` means enumeration RAN TO
    COMPLETION and legitimately numbered nothing (Profile's state 3), which is a normal outcome
    and must never be reported as "no items_unavailable reason was logged" -- that fallback
    text is for when NEITHER field was set, not for when the other one was."""
    run = tmp_path / "run_prior_item_manifest_unnumbered"
    run.mkdir(parents=True)
    successful = {
        "ts": "2026-08-15T06:10:25", "action": "capture", "profile_name": "Winnie",
        "items": 1, "item_translation": [1],
        "item_manifest": [{
            "kind": "item", "model_item": 1, "heart_ordinal": 1,
            "source_frame_index": 0, "page_rows": [553, 1527],
            "crop_size": [974, 974], "crop_sha256": "prior-winnie-crop",
            "reason": "item 1 (heart 1 on the page)",
        }],
    }
    numbered_nothing = {
        "ts": "2026-08-15T06:17:55", "action": "capture", "profile_name": "Shannon",
        "items": 0, "item_translation": [], "item_manifest": [],
        "items_unnumbered": "9 selectable card(s) were considered; all excluded.",
    }
    (run / "actions.jsonl").write_text(
        json.dumps(successful) + "\n" + json.dumps(numbered_nothing) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "prior successful capture at `2026-08-15T06:10:25` for profile `Winnie`" in md
    assert ("latest capture at `2026-08-15T06:17:55` for profile `Shannon` had no numbered "
            "manifest; enumeration completed but numbered nothing: `9 selectable card(s) were "
            "considered; all excluded.`" in md)
    assert "no items_unavailable reason was logged" not in md
    assert "items unavailable" not in md


def test_debug_report_marks_prior_manifest_when_the_next_read_was_stopped(tmp_path):
    """A Stop mid-read writes `capture_aborted` and NO capture record at all.

    The 2026-08-15 report is the live shape: Mariya captured six items, the operator pressed Stop
    eleven frames into Julia, and the manifest section then printed Mariya's page/heart numbers
    directly under Julia's story with nothing saying so. An abandoned read ends a profile exactly
    as a refusal does, so it has to reach this warning by the same route.
    """
    run = tmp_path / "run_stopped_next_read"
    run.mkdir(parents=True)
    successful = {
        "ts": "2026-08-15T11:46:57", "action": "capture", "profile_name": "Mariya",
        "items": 1, "item_translation": [1],
        "item_manifest": [{
            "kind": "item", "model_item": 1, "heart_ordinal": 1,
            "source_frame_index": 0, "page_rows": [517, 1491],
            "crop_size": [974, 974], "crop_sha256": "prior-mariya-crop",
            "reason": "item 1 (heart 1 on the page)",
        }],
    }
    aborted = {"ts": "2026-08-15T12:08:33", "action": "capture_aborted", "frames": 11,
               "read_scrolls": 11, "profile_name": "Julia"}
    (run / "actions.jsonl").write_text(
        json.dumps(successful) + "\n" + json.dumps(aborted) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "prior successful capture at `2026-08-15T11:46:57` for profile `Mariya`" in md
    assert ("the read that followed it was abandoned at `2026-08-15T12:08:33` for profile "
            "`Julia` on Stop after 11 frame(s), so it has no manifest of its own and none of "
            "the numbers below describe it" in md)
    assert "prior-mariya-crop" in md
    # A Stop is not a refusal, so it must not borrow the refusal wording or invent a reason.
    assert "had no numbered manifest" not in md
    assert "items unavailable" not in md


def test_only_the_expanded_captures_manifest_is_replaced_by_a_pointer_to_it(tmp_path):
    """The tail says "see the manifest above", and exactly one manifest is ever above.

    A run that captured several profiles inside the tail window has several manifest-bearing
    records but only the LAST one's table is expanded. Pointing every one of them at "above"
    tells a reader that an earlier capture's rows are the ones printed, which is the same
    misdirection as showing a stale table with no provenance line at all.
    """
    run = tmp_path / "run_two_manifests"
    run.mkdir(parents=True)

    def capture(ts, name, sha):
        return {"ts": ts, "action": "capture", "profile_name": name, "items": 1,
                "item_translation": [1],
                "item_manifest": [{"kind": "item", "model_item": 1, "heart_ordinal": 1,
                                   "source_frame_index": 0, "page_rows": [517, 1491],
                                   "crop_size": [974, 974], "crop_sha256": sha,
                                   "reason": "item 1 (heart 1 on the page)"}]}

    (run / "actions.jsonl").write_text(
        json.dumps(capture("2026-08-15T11:20:01", "Ellie", "ellie-crop")) + "\n"
        + json.dumps({"ts": "2026-08-15T11:30:00", "action": "observe_decision",
                      "decision": "like", "profile_name": "Ellie"}) + "\n"
        + json.dumps(capture("2026-08-15T11:46:57", "Mariya", "mariya-crop")) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    # The expanded table is the latest capture's, so only that record may point at it...
    assert md.count('"item_manifest": "see item-numbering manifest above"') == 1
    assert "mariya-crop" in md
    # ...and the earlier capture says where its own rows actually live.
    assert ('"item_manifest": "1 manifest row(s) in actions.jsonl; the expanded table above '
            'belongs to a different capture"' in md)
    assert "ellie-crop" not in md
    # Both captures succeeded, so there is no stale-manifest warning to raise.
    assert "prior successful capture" not in md


def test_recent_logs_points_to_an_oversized_item_index_wall_already_shown_in_summary(tmp_path):
    """The console mirrors a driver refusal, but the report must keep its one full copy in
    the structured item-index summary rather than printing the geometry wall again at the end."""
    run = tmp_path / "run_item_index_recent_logs"
    run.mkdir(parents=True)
    reason = ("the item index this capture produced contradicts itself: frame 19 sees page rows "
              "6368..6709 while frame 20 bounded page rows 6368..6503; " + "geometry detail " * 80)
    refusal = {"action": "item_index_refused", "reason": reason, "steps_px": [240, 250, 245, 15]}
    (run / "actions.jsonl").write_text(json.dumps(refusal) + "\n")
    bugreport._LOG_RING.clear()
    bugreport._LOG_RING.append("14:05:22 hinge: no numbered item list — " + reason)
    try:
        report_parts = (bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
                        + "\n## Recent logs\n" + bugreport._logs_md(10))
    finally:
        bugreport._LOG_RING.clear()

    assert report_parts.count(bugreport._sanitize_inline(reason)) == 1
    recent_logs = report_parts.split("## Recent logs\n", 1)[1]
    assert reason not in recent_logs
    assert "full geometry is in the item-index summary below" in recent_logs


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
        "anchor": "00002_capture_split_anchor.png", "identity_anchor_frame_index": 1,
        "identity_anchor_confirmed": False, "content_match": False,
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
    assert "identity-anchor screenshot `00002_capture_split_anchor.png` (frame 1)" in md
    assert "boundary-trigger screenshot `00002_capture_split_after.png`" in md
    assert "identity distance 17.95, scroll-top distance 15.73" in md
    assert "identity read `Ada`" in md
    assert "identity anchor still provisional" in md
    assert "adjacent content did not align" in md
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


def test_debug_log_capture_split_recovery_distinguishes_unnumbered_from_unavailable(tmp_path):
    """The recovery capture after a split can land in either of Profile's zero-item states, and
    they must not read alike: `items_unavailable` is the enumeration failure that halts an auto
    LIKE, `items_unnumbered` is a normal "nothing survived policy" outcome (see the sibling
    `items_unavailable` case in the recovery test above, and Profile's own docstring)."""
    run = tmp_path / "run_split_recovery_unnumbered"
    run.mkdir(parents=True)
    split = {"ts": "2026-08-13T00:20:55", "action": "capture_split", "photos": 2,
             "before": "00001_capture_split_before.png"}
    recovery = {
        "ts": "2026-08-13T00:21:54", "action": "capture", "photos": 12,
        "capture_truncated": True, "items": 0,
        "items_unnumbered": "3 selectable card(s) were considered; all excluded.",
    }
    (run / "actions.jsonl").write_text(json.dumps(split) + "\n" + json.dumps(recovery) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert ("later capture/recovery followed: 12 photo(s); capture truncated; 0 numbered "
            "item(s); enumeration completed but numbered nothing: `3 selectable card(s) were "
            "considered; all excluded.`" in md)
    assert "items unavailable" not in md


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
    # A web-only target has no adb_path; the fallback must still find an Android target's.
    apps = {"web_only": {"url": "https://example.invalid/app"},
            "hinge": {"adb_path": "adb", "serial": "HINGE-SERIAL"}}
    assert bugreport._first_android_app_cfg(apps, ["web_only"]) == ("adb", "HINGE-SERIAL")


def test_first_android_app_cfg_none_when_nothing_declares_adb_path():
    assert bugreport._first_android_app_cfg(
        {"web_only": {"url": "https://example.invalid/app"}}, ["web_only"]
    ) is None


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
        apps={"web_only": {"url": "https://example.invalid/app"}},
        enabled_apps=["web_only"],
    ))
    assert "touch watcher: no Android app configured" in bugreport.build_report(None)


def test_touch_watcher_probe_when_config_cannot_load(monkeypatch):
    def _boom(path):
        raise ValueError("bad yaml")
    monkeypatch.setattr(oplove_config, "load", _boom)
    md = bugreport.build_report(None)
    assert "touch watcher: could not load" in md
    assert "bad yaml" in md


# ── Evidence anomalies: the two 2026-08-14 bugs a faithful tail could not show ──────────────
def test_composer_reported_closed_then_open_again_is_flagged(tmp_path):
    """A like sheet cannot reopen itself, so like_sending -> like_sheet is a misread poll.

    The 2026-08-14 run printed both records correctly and the regression still read as ordinary
    waiting. It is not cosmetic: like_sheet re-arms the stuck-screen budget on every poll and
    like_sending deliberately does not, so a false like_sending spends the allowance belonging
    to a human who is still typing."""
    run = tmp_path / "run_20260814_233454"
    run.mkdir(parents=True)
    waits = [{"ts": "2026-08-14T23:43:51", "action": "observe_waiting", "reason": "like_sheet"},
             {"ts": "2026-08-14T23:44:01", "action": "observe_waiting", "reason": "like_sending"},
             {"ts": "2026-08-14T23:44:06", "action": "observe_waiting", "reason": "like_sheet"}]
    (run / "actions.jsonl").write_text("".join(json.dumps(w) + "\n" for w in waits))

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "evidence anomalies:" in md
    assert "reported CLOSED and then OPEN again 1x" in md
    assert "2026-08-14T23:44:01" in md and "2026-08-14T23:44:06" in md


def test_a_forward_only_composer_close_is_not_flagged(tmp_path):
    """like_sheet -> like_sending is the NORMAL resolution and must stay quiet."""
    run = tmp_path / "run_20260814_233455"
    run.mkdir(parents=True)
    waits = [{"ts": "2026-08-14T23:43:51", "action": "observe_waiting", "reason": "like_sheet"},
             {"ts": "2026-08-14T23:44:01", "action": "observe_waiting", "reason": "like_sending"}]
    (run / "actions.jsonl").write_text("".join(json.dumps(w) + "\n" for w in waits))

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "reported CLOSED and then OPEN again" not in md


def _stale_evidence_run(tmp_path, name, records, shots):
    run = tmp_path / name
    run.mkdir(parents=True)
    for shot, payload in shots.items():
        (run / shot).write_bytes(payload)
    (run / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})


def test_like_evidenced_by_a_frame_older_than_its_composer_is_flagged(tmp_path):
    """The exact 2026-08-14 shape: a LIKE filed against a pre-tap frame.

    Compared by CONTENT, not filename -- the debug log keys dedup on (action label, digest), so
    the same pixels logged under a different action get a fresh filename, which is how this hid
    as `00019_observe_decision_before.png`."""
    md = _stale_evidence_run(
        tmp_path, "run_20260814_233456",
        [{"ts": "2026-08-14T23:41:06", "action": "observe_waiting", "reason": "no_change",
          "before": "00007_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:41:48", "action": "observe_waiting", "reason": "like_sheet",
          "before": "00009_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:46:07", "action": "observe_decision", "decision": "like",
          "before": "00019_observe_decision_before.png"}],
        {"00007_observe_waiting_before.png": b"identical-pixels",
         "00009_observe_waiting_before.png": b"the composer, open",
         "00019_observe_decision_before.png": b"identical-pixels"})

    assert "BEFORE its composer existed" in md
    assert "00019_observe_decision_before.png" in md
    assert "00007_observe_waiting_before.png" in md
    assert "5m01s earlier" in md


def test_like_evidenced_by_its_own_composer_frame_is_not_flagged(tmp_path):
    """The fixed behaviour: evidence captured DURING the composer episode is correct."""
    md = _stale_evidence_run(
        tmp_path, "run_20260814_233457",
        [{"ts": "2026-08-14T23:41:06", "action": "observe_waiting", "reason": "no_change",
          "before": "00007_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:41:48", "action": "observe_waiting", "reason": "like_sheet",
          "before": "00009_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:46:07", "action": "observe_decision", "decision": "like",
          "before": "00019_observe_decision_before.png"}],
        {"00007_observe_waiting_before.png": b"a scrolled profile",
         "00009_observe_waiting_before.png": b"the composer, open",
         "00019_observe_decision_before.png": b"the composer, open"})

    assert "BEFORE its composer existed" not in md


def test_like_candidate_without_a_verified_sheet_is_not_a_composer_episode(tmp_path):
    md = _stale_evidence_run(
        tmp_path, "run_20260814_233457_candidate",
        [{"ts": "2026-08-14T23:41:06", "action": "observe_waiting", "reason": "no_change",
          "before": "00007_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:41:48", "action": "observe_waiting", "reason": "like_candidate",
          "before": "00009_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:42:07", "action": "observe_bottom_delta",
          "before": "00010_observe_bottom_delta_before.png"}],
        {"00007_observe_waiting_before.png": b"a motionless card",
         "00009_observe_waiting_before.png": b"a bottom-only delta",
         "00010_observe_bottom_delta_before.png": b"a motionless card"})

    assert "BEFORE its composer existed" not in md


def test_screenshot_digest_rejects_paths_outside_the_flat_run_directory(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private pixels")
    run = tmp_path / "run"
    run.mkdir()

    assert bugreport._shot_digest(run, str(outside), {}) is None
    assert bugreport._shot_digest(run, "../outside.png", {}) is None
    assert bugreport._shot_digest(run, "nested/shot.png", {}) is None
    assert bugreport._shot_digest(run, "actions.jsonl", {}) is None


def test_a_pass_reusing_the_previous_still_frame_is_not_flagged(tmp_path):
    """Reusing an earlier frame is usually CORRECT -- a decision's evidence is the card as it
    looked just before it advanced, and on a motionless screen that IS the last waiting shot.
    Flagging it fired on ~10 of 16 real runs, nearly all benign; a check that cries wolf gets
    skipped."""
    md = _stale_evidence_run(
        tmp_path, "run_20260814_233458",
        [{"ts": "2026-08-14T23:40:04", "action": "observe_waiting", "reason": "no_change",
          "before": "00002_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:40:20", "action": "observe_waiting", "reason": "no_change",
          "before": "00002_observe_waiting_before.png"},
         {"ts": "2026-08-14T23:40:36", "action": "observe_decision", "decision": "pass",
          "before": "00003_observe_decision_before.png"}],
        {"00002_observe_waiting_before.png": b"a motionless screen",
         "00003_observe_decision_before.png": b"a motionless screen"})

    assert "BEFORE its composer existed" not in md


def test_evidence_anomaly_checks_survive_missing_screenshots(tmp_path):
    """A report must never raise while describing a bug: rotated-away shots are just unknown."""
    run = tmp_path / "run_20260814_233458"
    run.mkdir(parents=True)
    records = [{"ts": "2026-08-14T23:41:06", "action": "observe_waiting", "reason": "no_change",
                "before": "00007_rotated_away.png"},
               {"ts": "2026-08-14T23:46:07", "action": "observe_decision", "decision": "like",
                "before": "00019_also_gone.png"}]
    (run / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert "cached, not fresh" not in md
    assert '"action": "observe_decision"' in md          # the rest of the section still renders


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
                "identity_name_read": "Qelix", "identity_name_verdict": "new",
                "confirm_identity": "new", "confirm_identity_dist": 22.2,
                "confirm_identity_name_read": "Qelix",
                "confirm_identity_name_verdict": "new",
                "profile_name": "Amanda", "gesture": "tap_pass", "watcher": True}
    (run / "actions.jsonl").write_text(json.dumps(capture) + "\n" + json.dumps(decision) + "\n")

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})
    assert '"action": "capture"' in md and '"action": "observe_decision"' in md
    for field in ('"identity": "new"', '"identity_dist": 22.4',
                  '"identity_name_read": "Qelix"', '"confirm_identity": "new"',
                  '"confirm_identity_dist": 22.2',
                  '"confirm_identity_name_read": "Qelix"', '"profile_name": "Amanda"',
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
    lines.extend(
        json.dumps({"ts": f"20:46:{13 + i * 15:02d}", "action": "observe_waiting",
                    "reason": "no_change"})
        for i in range(12)
    )
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


def test_action_counts_merge_legacy_position_labels_with_deduped_dwell_labels():
    lines = [
        json.dumps({"action": "still_photo_dwell_00", "dwell_frame_index": 0}),
        json.dumps({"action": "still_photo_dwell_01", "dwell_frame_index": 1}),
        json.dumps({"action": "still_photo_dwell_frame", "dwell_frame_index": 2}),
        json.dumps({"action": "still_photo_reattach_00", "reattach_frame_index": 0}),
        json.dumps({"action": "still_photo_reattach_frame", "reattach_frame_index": 1}),
    ]

    assert bugreport._action_counts_line(lines) == (
        "action counts: still_photo_dwell_frame 3 · still_photo_reattach_frame 2")


# ── Stall summary ─────────────────────────────────────────────────────────────────────────
# An unrecognised Hinge+ paywall can leave _await_like_resolved polling like_sheet until the
# operator stops the run. actions.jsonl records every poll, but diagnosing the hang from the
# raw tail requires noticing observe_waiting repeating with the same reason. These tests pin
# _stall_summary_md's worked example from its docstring: "most recent repeated observe wait:
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

    assert "most recent repeated observe wait" in md
    assert "reason=`like_sheet`" in md
    assert "1m37s" in md            # 01:42:43 -> 01:44:20
    assert "(5 records)" in md


def test_stall_summary_labels_recency_ranking_without_claiming_duration_ranking():
    """The final wait is the incident-priority row even when an older wait lasted longer.

    The report must not call either recency-ranked row "longest"; doing so made a real report
    say a 31s wait was longer than the older 1m03s wait displayed immediately below it.
    """
    lines = [
        json.dumps({"ts": "2026-08-24T13:00:00", "action": "observe_waiting", "reason": "older"}),
        json.dumps({"ts": "2026-08-24T13:01:03", "action": "observe_waiting", "reason": "older"}),
        json.dumps({"ts": "2026-08-24T13:01:05", "action": "observe_decision", "decision": "pass"}),
        json.dumps({"ts": "2026-08-24T13:13:20", "action": "observe_waiting", "reason": "newer"}),
        json.dumps({"ts": "2026-08-24T13:13:51", "action": "observe_waiting", "reason": "newer"}),
    ]

    md = bugreport._stall_summary_md(lines)

    assert "most recent repeated observe wait: reason=`newer` for 31s" in md
    assert "2nd most recent repeated observe wait: reason=`older` for 1m03s" in md
    assert "observe stall" not in md


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
    assert "repeated observe waits:" in md
    assert "reason=`no_change`" in md
    assert "2m45s" in md           # 11 * 15s between the first and last of the 12 repeats
    assert "(12 records)" in md
    # (2) ...and the SAME 12 repeats, independently collapsed for the tail further down.
    assert '"repeated": 12' in md
    assert '"reason": "no_change"' in md
    stall_pos = md.index("repeated observe waits:")
    tail_pos = md.index("actions.jsonl (tail):")
    assert stall_pos < tail_pos    # the summary sits above the raw/collapsed tail, not below it


def test_release_publication_inside_an_observe_wait_does_not_split_the_stall_or_context(tmp_path):
    """A suggestion publication is a same-card event, not a decision or a fresh capture."""
    run = tmp_path / "run_release_interleaved_wait"
    run.mkdir(parents=True)
    records = [
        {"ts": "2026-08-24T13:11:55", "action": "capture", "photos": 14},
        {"ts": "2026-08-24T13:12:01", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:12:16", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:12:32", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:12:48", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:13:04", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:13:18", "action": "observe_release_hub_pre_tap_published"},
        {"ts": "2026-08-24T13:13:20", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:13:35", "action": "observe_waiting", "reason": "no_change"},
        {"ts": "2026-08-24T13:13:51", "action": "observe_waiting", "reason": "no_change"},
    ]
    lines = [json.dumps(record) for record in records]
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n")

    summary = bugreport._stall_summary_md(lines)
    context = bugreport._latest_observe_context_md(lines, run)

    assert "most recent repeated observe wait: reason=`no_change` for 1m50s (8 records)" in summary
    assert "final logged wait began at `2026-08-24T13:12:01` and has 8 heartbeat record(s)" in context


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
    """Minimal HubState stand-in without optional last-run mode compatibility context."""
    def __init__(self, entries):
        self._entries = entries

    def recent_openers(self):
        return self._entries


class _FakeHubOpenersWithStatus(_FakeHubOpeners):
    def __init__(self, entries, mode):
        super().__init__(entries)
        self._mode = mode

    def snapshot(self):
        return {"status": {"apps": {"hinge": {"mode": self._mode}}}}


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
    assert "no committed opener records" in md
    assert "unacted staged opener draft" in md


def test_recent_openers_section_reports_explicit_training_mode():
    entry = _opener_entry(advisory=False)
    entry["session_mode"] = "training"

    md = bugreport._recent_openers_md(_FakeHubOpeners([entry]))

    assert " · training · " in md
    assert " · auto · " not in md


def test_recent_openers_section_uses_last_run_mode_for_legacy_training_entry():
    md = bugreport._recent_openers_md(
        _FakeHubOpenersWithStatus([_opener_entry(advisory=False)], "training"))

    assert " · training · " in md
    assert " · auto · " not in md


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


def test_recent_openers_render_compact_monitor_only_metadata_without_implying_rejection():
    entry = _opener_entry(opener="That lake backdrop is unreal. Where was this?")
    entry.update({
        "redundancy_markers": [
            'opener restates the referenced word "lake"',
            'opener restates the referenced word "backdrop"',
            "third", "fourth", "fifth",
        ],
        "entropy_collision": "that lake backdrop",
        "entropy_regenerated": True,
    })

    md = bugreport._recent_openers_md(_FakeHubOpeners([entry]))

    assert "Monitor only; never a rejection" in md
    assert 'opener restates the referenced word "lake"' in md
    assert "+1 more" in md
    assert 'opening collision "that lake backdrop" (regenerated once)' in md


def test_recent_openers_omit_monitor_line_when_no_monitor_fired():
    entry = _opener_entry()
    entry.update({"redundancy_markers": [], "entropy_collision": "",
                  "entropy_regenerated": False})

    md = bugreport._recent_openers_md(_FakeHubOpeners([entry]))

    assert "Monitor only" not in md


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
    assert "no committed opener rejections" in md


def test_status_separates_zero_preference_decisions_from_provider_billing_telemetry():
    """An unacted staged opener can produce billed usage without being a swipe/label.

    This is deliberately a report-level test: the desired fix is not to hide a real provider
    charge, mistake the accounted-result ledger for all HTTP attempts, or let a returning
    operator read it as an unwanted profile decision.
    """
    class _NoDecisionChargedHub:
        def snapshot(self):
            return {"running": False, "error": None, "status": {
                "phase": "stopped", "mode": "observe", "running": False,
                "labels": 351, "min_labels": 40, "ranker_ready": True,
                "budget_spent": 0.01, "budget_cap": 5.0, "openers": 1,
                "apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "stopped",
                                     "last_decision": None, "last_score": None,
                                     "swipes_run": 0, "error": None,
                                     "stop_reason": None}}}}

    md = bugreport._status_md(_NoDecisionChargedHub())
    assert "labels: 351 / 40 (ready) (ranker dataset total, not this run)" in md
    assert "preference decisions recorded this run: 0" in md
    assert "provider / billing telemetry: 1 accounted model result(s) with usage" in md
    assert "HTTP fallback failures are logged separately and excluded" in md
    assert "No pass/like preference decision was recorded" in md


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


def _targeting_config(tmp_path, *, licence_key=None, calibration=False):
    import yaml
    app = {"serial": "synthetic-pixel"}
    if licence_key:
        app[licence_key] = {"acceptance": "placeholder"}
    if calibration:
        app["targeting_calibration"] = {"schema_version": 3}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"enabled_apps": ["hinge"], "apps": {"hinge": app}}))
    return str(path)


def _valid_targeting_calibration(device="synthetic-pixel"):
    return {
        "schema_version": 3,
        "device": device,
        "hinge_version_name": "10.0.1",
        "frame_size_px": [1080, 2400],
        "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 2.5,
        "inline_item_max_dist": 14.9,
        "calibrated_at": "2026-08-24T00:00:00Z",
        "identity_band": [0.1, 0.048, 0.8, 0.094],
        "content_band": [0.125, 0.875],
    }


def test_the_report_names_which_link_of_the_targeting_chain_is_missing(tmp_path):
    """BUG REPORT 2026-08-22: the cause was one absent config key and the report never said so.

    It showed opener enabled, a present API key, and zero provider calls, leaving the reader to
    reconstruct the cause from a driver sentence in the log tail. All three gates now report
    separately, because any of them can be the answer.
    """
    from operation_love import targeting_policy as tp

    path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
    md = bugreport._targeting_readiness_md(path)

    assert "`apps.hinge.still_photo_assumption_acceptance`" in md
    assert "`apps.hinge.targeting_calibration`: ABSENT" in md
    assert "no targeted opener is generated or offered" in md
    # No licence in this process, but the config carries one: say that rather than printing a
    # next step that contradicts the config line two rows above it.
    assert tp.hinge_targeting_unavailable_reason() in md
    assert "reflects this unvalidated reporting process" in md
    assert tp.TARGETING_SETUP_NEXT_STEP_BLOCKED not in md


def test_the_report_names_the_calibration_as_the_next_step_once_a_licence_is_installed(tmp_path):
    from operation_love import targeting_policy as tp

    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION, device="synthetic-pixel",
        hinge_version_name="10.0.1", accepted_at="2026-08-21",
        rationale="owner judged the held-out campaign not worth ~420 real passes"))
    try:
        path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
        md = bugreport._targeting_readiness_md(path)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "UNMEASURED" in md            # the provenance, never silently upgraded to a bound
    assert "targeting-policy blocker: none" in md
    assert tp.TARGETING_SETUP_NEXT_STEP_CALIBRATE in md
    assert "unvalidated reporting process" not in md


def test_targeting_readiness_does_not_recommend_calibration_that_is_already_valid(tmp_path):
    """A valid mapping is config-ready, while a live session still binds it exactly."""
    from operation_love import targeting_policy as tp
    import yaml

    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION, device="synthetic-pixel",
        hinge_version_name="10.0.1", accepted_at="2026-08-21", rationale="accepted"))
    try:
        path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
        raw = yaml.safe_load(Path(path).read_text())
        raw["apps"]["hinge"]["targeting_calibration"] = _valid_targeting_calibration()
        Path(path).write_text(yaml.safe_dump(raw))
        md = bugreport._targeting_readiness_md(path)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "targeting-policy blocker: none" in md
    assert "targeting_calibration`: present and validated (static config check)" in md
    assert "targeting config readiness: ready" in md
    assert "exact Hinge app-version/frame match" in md
    assert tp.TARGETING_SETUP_NEXT_STEP_CALIBRATE not in md


def test_targeting_readiness_overrides_config_ready_when_live_calibration_is_rejected(tmp_path):
    """A stale version-bound mapping must point to recapture, never claim setup is complete."""
    from operation_love import targeting_policy as tp
    import yaml

    class _RuntimeCalibrationRejectedHub:
        REASON = ("Training could not prepare a verifiable targeted opener for this profile: "
                  "apps.hinge.targeting_calibration is unavailable "
                  "(the live app build/frame geometry does not exactly match schema-v3 "
                  "calibration ('10.1.0'/(1080, 2400) != '10.0.1'/(1080, 2400))), so a "
                  "model-selected item could not be verified or targeted")

        def snapshot(self):
            return {"running": False, "error": None, "status": {
                "phase": "stopped", "mode": "training", "running": False,
                "labels": 59, "min_labels": 40, "ranker_ready": True,
                "budget_spent": 0.0, "budget_cap": 5.0, "openers": 0,
                "apps": {"hinge": {"app": "hinge", "mode": "training", "state": "stopped",
                                    "last_decision": None, "last_score": None, "swipes_run": 0,
                                    "error": None, "stop_reason": self.REASON,
                                    "stop_kind": "targeting_calibration"}}}}

    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION, device="synthetic-pixel",
        hinge_version_name="10.0.1", accepted_at="2026-08-21", rationale="accepted"))
    try:
        path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
        raw = yaml.safe_load(Path(path).read_text())
        raw["apps"]["hinge"]["targeting_calibration"] = _valid_targeting_calibration()
        Path(path).write_text(yaml.safe_dump(raw))
        md = bugreport.build_report(_RuntimeCalibrationRejectedHub(), config_path=path)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "targeting config readiness: ready" not in md
    assert "runtime calibration: REJECTED in the latest hub snapshot" in md
    assert "targeting_calibration" in md
    assert "'10.1.0'/(1080, 2400) != '10.0.1'/(1080, 2400)" in md
    assert "next step: recapture and validate a schema-v3 targeting calibration" in md
    assert "explicitly re-accept the unmeasured still-photo assumption" in md
    assert "for that exact build/device" in md
    assert "no targeting setup action remains" not in md


def test_targeting_readiness_does_not_treat_an_invalid_mapping_as_ready(tmp_path):
    from operation_love import targeting_policy as tp
    import yaml

    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION, device="synthetic-pixel",
        hinge_version_name="10.0.1", accepted_at="2026-08-21", rationale="accepted"))
    try:
        path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
        raw = yaml.safe_load(Path(path).read_text())
        raw["apps"]["hinge"]["targeting_calibration"] = {"schema_version": 3}
        Path(path).write_text(yaml.safe_dump(raw))
        md = bugreport._targeting_readiness_md(path)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "targeting-policy blocker: none" in md
    assert "present but not validated" in md
    assert "targeting readiness: ready" not in md
    assert "repair or replace the invalid targeting calibration" in md


def test_the_report_says_when_no_licence_key_exists_at_all(tmp_path):
    from operation_love import targeting_policy as tp

    md = bugreport._targeting_readiness_md(_targeting_config(tmp_path))

    assert "still-photo licence key in config: none" in md
    assert tp.TARGETING_SETUP_NEXT_STEP_BLOCKED in md


def test_the_targeting_section_never_installs_a_licence_as_a_side_effect(tmp_path):
    """A diagnostic must not clear and reinstall the slot a live run is reading."""
    from operation_love import targeting_policy as tp

    path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
    bugreport._targeting_readiness_md(path)

    assert tp.still_photo_licence_provenance() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


# --- post-tap sheet verification (filed against the 2026-08-27 report) --------------------


def _verify_lines(**overrides):
    record = {"action": "verify_sheet_item", "item": 1, "outcome": "verify_mismatch",
              "nearest": 1, "distance": 10.283447265625, "bound": 7.0,
              "preview": [236, 1092, 95, 985]}
    record.update(overrides)
    identity = {"action": "verify_sheet_identity", "item": 1, "outcome": "identity_match",
                "distance": 0.0, "bound": 0.8852}
    return [json.dumps(identity), json.dumps(record)]


def test_sheet_verification_section_names_the_bound_regime_and_the_item_table(tmp_path):
    """The three things the 2026-08-27 report should have carried and did not.

    Without them "10.283 against 7.000" cannot be told from a genuine wrong-item tap: the reader
    cannot see WHICH bound regime fired, what the per-item comparison actually measured, or what
    geometry the whole check was bound to.
    """
    md = bugreport._sheet_verification_md(_verify_lines(
        reason=("the like sheet does not show model item 1: item 1 is the nearest stored crop "
                "but at 10.283 grey levels, against a 7.000 bound (the inline composer's "
                "independently held-out 7.00 one-item render ceiling, there being no other item "
                "to bound against)"),
        preview_reason="topmost run of rows at least 870px wide; bottom edge carried",
        grid=[64, 64],
        composer={"layout_id": "hinge_inline_v1", "comment": [95, 1124, 985, 1302],
                  "send": [390, 1334, 985, 1443]},
        comparisons=[{"item": 1, "distance": 10.283447265625, "window_px": 937, "crop_px": 974,
                      "nearest_other": None, "bound": 7.0,
                      "why": "inline reframe rows 37..974 of a 974px crop"}]), tmp_path)
    assert "one-item render ceiling" in md
    assert "inline reframe rows 37..974" in md
    assert "window 937px of a 974px crop" in md
    assert "hinge_inline_v1" in md
    assert "comparison grid: 64x64" in md
    assert "profile identity on the same sheet: `identity_match`" in md
    # The one-item regime is the weakest bound available and must be called out as such.
    assert "ONE numbered item" in md


def test_an_unreadable_sheet_is_not_rendered_as_a_comparison_that_returned_nothing(tmp_path):
    """A refusal to LOOK and a comparison that FAILED are different events, and read alike.

    Filed against the 2026-08-28 pillarbox report. The locator refused before any card was
    compared, and the outcome line still printed the verdict shape -- `nearest stored item None;
    distance — against bound —`, three dashes standing exactly where three numbers stand on every
    other verdict this section renders. Nothing had been measured, but the report looked like
    something had been measured and had come back empty, which points a reader at the comparison
    when the whole story is in the geometry.
    """
    md = bugreport._sheet_verification_md(_verify_lines(
        outcome="unreadable", nearest=None, distance=None, bound=None,
        reason=("the selected-card preview is not immediately above the independently detected "
                "inline comment field: no candidate at least 694px wide (78% of the field's own "
                "890px), or at least 356px and centred inside it, runs for 300 rows above it")),
        tmp_path)

    assert "could not be LOOKED AT" in md
    assert "no card was compared" in md
    # the three quantities that do not exist are not printed as though they might
    assert "nearest stored item" not in md
    assert "against bound" not in md
    # ...and the refusal is labelled as one rather than as a verdict
    assert "why the sheet could not be read" in md
    assert "verdict, including which bound regime" not in md
    assert "no candidate at least 694px wide" in md


def test_a_real_verdict_still_carries_its_three_numbers(tmp_path):
    """The negative of the test above: nothing about a genuine comparison may be suppressed."""
    md = bugreport._sheet_verification_md(_verify_lines(), tmp_path)
    assert "nearest stored item 1" in md
    assert "against bound" in md
    assert "verdict" in md or "shape:" in md
    assert "could not be LOOKED AT" not in md


def test_sheet_verification_section_reads_a_log_written_before_the_new_fields_existed(tmp_path):
    """Old runs stay readable: the extra evidence is additive, never required to render."""
    md = bugreport._sheet_verification_md(_verify_lines(), tmp_path)
    assert "verify_mismatch" in md
    assert "10.283" in md and "7.000" in md
    assert "rows 236..1092, columns 95..985 (890x856px)" in md
    assert "bound regime" not in md


def test_a_verified_sheet_prints_no_verification_section(tmp_path):
    """This section explains a stop. A run that verified has nothing to explain."""
    assert bugreport._sheet_verification_md(
        _verify_lines(outcome="verify_match", distance=0.167), tmp_path) == ""


# --- the three shapes of verify_mismatch (2026-08-28 addendum) -----------------------------
# "intended model item 1, actual 1" -- the sentence the 2026-08-27 halt actually printed -- is
# ambiguous on its own: it could mean the sheet had the right card and the bound was simply too
# tight, or it could mean the sheet had a different card entirely. `nearest`/`item` settle it.


def test_verify_mismatch_shape_is_a_possible_false_refusal_when_nearest_is_the_intended_item():
    """nearest == item: the correct card was on the sheet; only the bound refused it."""
    shape = bugreport._verify_mismatch_shape({"item": 1, "nearest": 1})
    assert "FALSE REFUSAL" in shape
    assert "not a targeting miss" in shape
    assert "DIFFERENT item" not in shape


def test_verify_mismatch_shape_is_a_real_targeting_miss_when_nearest_differs():
    """nearest != item: a different card was actually on the sheet."""
    shape = bugreport._verify_mismatch_shape({"item": 1, "nearest": 2})
    assert "DIFFERENT item is on the sheet" in shape
    assert "targeting miss" in shape
    assert "FALSE REFUSAL" not in shape


def test_verify_mismatch_shape_says_nothing_was_measurable_when_nearest_is_none():
    """nearest absent/None: nothing in the payload could be compared to this sheet at all --
    not "the nearest item is None", which reads as a real item number (see item_verify.py's own
    2026-08-15 note about exactly this confusion)."""
    assert "nothing in this payload could be measured" in bugreport._verify_mismatch_shape(
        {"item": 1, "nearest": None})
    # Absent is the same fact as explicit None -- a record from before the field existed must
    # classify identically, never raise, and never be mistaken for the "different item" shape.
    assert "nothing in this payload could be measured" in bugreport._verify_mismatch_shape(
        {"item": 1})


def test_verify_mismatch_shape_tolerates_malformed_item_and_nearest_without_raising():
    """A malformed logged value must degrade gracefully, never raise, while generating a report
    about a stop -- raising here would destroy the evidence it exists to explain."""
    shape = bugreport._verify_mismatch_shape({"item": "not-a-number", "nearest": []})
    assert isinstance(shape, str) and shape
    # None of the three helpers may ever throw for `_sheet_verification_md` either.
    bugreport._sheet_verification_md(_verify_lines(item="not-a-number", nearest=[]), Path("."))


def test_sheet_verification_section_states_the_shape_right_after_the_outcome_line(tmp_path):
    """Wired into the real section, in the position the task asked for: its own line, right
    after the existing outcome line, with the existing numbers still intact."""
    md = bugreport._sheet_verification_md(_verify_lines(nearest=1, item=1), tmp_path)
    lines = md.splitlines()
    outcome_i = next(i for i, ln in enumerate(lines) if ln.startswith("- outcome:"))
    assert lines[outcome_i + 1].startswith("- shape:")
    assert "FALSE REFUSAL" in lines[outcome_i + 1]
    # The existing numbers are untouched.
    assert "10.283" in md and "7.000" in md


# --- naming before/after by role (2026-08-28 addendum) --------------------------------------


def test_screenshot_roles_are_named_when_both_before_and_after_are_recorded(tmp_path):
    (tmp_path / "before.png").write_bytes(b"before-bytes")
    (tmp_path / "after.png").write_bytes(b"after-bytes")
    md = bugreport._sheet_verification_md(
        _verify_lines(before="before.png", after="after.png"), tmp_path)
    assert ("pre-tap card under the heart: `before.png`; sheet the verdict was taken on: "
            "`after.png` — open both before reading a number.") in md


def test_screenshot_roles_line_is_omitted_when_neither_filename_is_recorded(tmp_path):
    md = bugreport._sheet_verification_md(_verify_lines(), tmp_path)
    assert "pre-tap card under the heart" not in md


def test_screenshot_roles_use_the_shot_digest_guard_and_never_print_an_unsafe_path(tmp_path):
    """Same acceptance as `_shot_digest`: a name that resolves outside the run directory, or a
    file that isn't actually there, must never be printed verbatim -- it degrades to a plain
    "not available" note instead."""
    (tmp_path / "outside.png").write_bytes(b"secret")
    md = bugreport._sheet_verification_md(
        _verify_lines(before="../outside.png", after="never_written.png"), tmp_path)
    assert "pre-tap card under the heart: `not available" in md
    assert "../outside.png" not in md
    assert "sheet the verdict was taken on: `not available" in md
    assert "never_written.png" not in md


def test_screenshot_roles_line_shows_whichever_side_is_actually_safe(tmp_path):
    """Only one side missing/unsafe must not swallow the side that IS good evidence."""
    (tmp_path / "before.png").write_bytes(b"before-bytes")
    md = bugreport._sheet_verification_md(
        _verify_lines(before="before.png", after=None), tmp_path)
    assert "pre-tap card under the heart: `before.png`" in md
    assert "sheet the verdict was taken on: `not available" in md


# --- verify rows never collapse, even once they carry a `reason` (2026-08-28, TASK 3) -------
# `before`/`after` are the whole diagnosis for a targeting stop. `_render_run` drops every field
# but ts/action/reason/repeated, so a `verify_sheet_item`/`verify_sheet_identity` row that
# collapses into a repeat run loses the exact screenshots this section just learned to name.


def test_action_reason_key_never_matches_verify_sheet_item_even_with_a_shared_reason():
    raw = json.dumps({"action": "verify_sheet_item", "reason": "same text",
                       "before": "b.png", "after": "a.png"})
    assert bugreport._action_reason_key(raw) is None


def test_action_reason_key_never_matches_verify_sheet_identity_even_with_a_shared_reason():
    raw = json.dumps({"action": "verify_sheet_identity", "reason": "same text",
                       "before": "b.png", "after": "a.png"})
    assert bugreport._action_reason_key(raw) is None


def test_two_verify_sheet_item_rows_with_the_same_reason_do_not_collapse():
    """Without the exclusion these would merge into one `{"repeated": 2, ...}` summary and both
    sets of before/after filenames would vanish."""
    lines = [
        json.dumps({"ts": "t0", "action": "verify_sheet_item", "reason": "same text",
                    "before": "b0.png", "after": "a0.png"}),
        json.dumps({"ts": "t1", "action": "verify_sheet_item", "reason": "same text",
                    "before": "b1.png", "after": "a1.png"}),
    ]
    out = bugreport._collapse_action_tail(lines, 30)
    assert out == lines
    assert "b0.png" in out[0] and "a0.png" in out[0]
    assert "b1.png" in out[1] and "a1.png" in out[1]


def test_two_verify_sheet_identity_rows_with_the_same_reason_do_not_collapse():
    lines = [
        json.dumps({"ts": "t0", "action": "verify_sheet_identity", "reason": "unreadable: x",
                    "before": "b0.png", "after": "a0.png"}),
        json.dumps({"ts": "t1", "action": "verify_sheet_identity", "reason": "unreadable: x",
                    "before": "b1.png", "after": "a1.png"}),
    ]
    out = bugreport._collapse_action_tail(lines, 30)
    assert out == lines


def test_item_coverage_warns_when_a_dwell_shortfall_leaves_one_numbered_item():
    """The upstream half of the same stop: a one-item payload is what selects the weak bound."""
    lines = bugreport._item_coverage_lines({"item_coverage": {
        "still_photo_dwell_candidate_limit": 3,
        "photo_candidate_page_hearts": [1, 3, 4, 6, 8, 9],
        "dwell_covered_page_hearts": [9],
        "no_dwell_coverage_page_hearts": [1, 3, 4, 6, 8],
        "numbered_page_hearts": [9]}})
    md = "\n".join(lines)
    assert "1 of 6 photo candidate(s) were dwelled" in md
    assert "candidate limit 3" in md
    assert "[1, 3, 4, 6, 8]" in md
    assert "coverage gap, not a judgement" in md
    assert "exactly ONE numbered item" in md


def test_item_coverage_is_quiet_when_every_photo_candidate_was_observed():
    lines = bugreport._item_coverage_lines({"item_coverage": {
        "photo_candidate_page_hearts": [1, 2],
        "dwell_covered_page_hearts": [1, 2],
        "no_dwell_coverage_page_hearts": [],
        "numbered_page_hearts": [1, 2]}})
    md = "\n".join(lines)
    assert "2 of 2 photo candidate(s) were dwelled" in md
    assert "⚠️" not in md


def test_a_validated_calibration_prints_its_numbers_not_just_its_verdict(tmp_path):
    """The 2026-08-27 report said "present and validated" and showed no value.

    That mattered because the halt was refused at 7.000 while this config's own
    `inline_item_max_dist` was 14.9099 — a reader with only the verdict could not see that the
    operator's calibrated ceiling was NOT the bound that fired (it is a second, stricter cap
    applied after a bound derived inside `item_verify`). These are geometry and thresholds, so
    printing them costs no secret.
    """
    from operation_love import targeting_policy as tp
    import yaml

    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION, device="synthetic-pixel",
        hinge_version_name="10.0.1", accepted_at="2026-08-21", rationale="accepted"))
    try:
        path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
        raw = yaml.safe_load(Path(path).read_text())
        calibration = _valid_targeting_calibration()
        raw["apps"]["hinge"]["targeting_calibration"] = calibration
        Path(path).write_text(yaml.safe_dump(raw))
        md = bugreport._targeting_readiness_md(path)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "present and validated (static config check)" in md
    for key in ("hinge_version_name", "inline_item_max_dist", "identity_match_max_dist"):
        assert key in md, key
    assert str(calibration["inline_item_max_dist"]) in md


# ── refused-capture leading-edge geometry (sidecar) ─────────────────────────
def _geometry_frame(index, offset, blocks, runs):
    """One `all_frame_geometry` record in the shape `hinge.geometry_record` writes."""
    return {
        "local_frame_index": index, "source_frame_index": index, "offset_px": offset,
        "animation_marker": False, "video_mute_markers": [],
        "blocks": [dict(b) for b in blocks],
        "background_runs": [{"frame_rows": list(rows), "kind": kind,
                             "widest_intruder_px": 0, "median_level_delta": 0.0}
                            for rows, kind in runs],
    }


def _block(y0, y1, *, kind="partial", top_kind="background_run", top_observed=False,
           bottom_kind="gutter", bottom_observed=True, digest=None):
    record = {"frame_rows": [y0, y1], "page_rows": None, "kind": kind, "complete": False,
              "top_observed": top_observed, "bottom_observed": bottom_observed,
              "top_kind": top_kind, "bottom_kind": bottom_kind,
              "hearts": {"frame_rows": [], "page_rows": []}}
    if digest is not None:
        record["content_digest"] = digest
    return record


def _pinned_header_geometry(frames=6, *, unanchored=False, digest=None):
    """The 2026-08-28 shape: a 43px strip at frame rows 368..411 that never moves, an over-long
    106px background run beneath it that is not gutter-length, and scrolling content below."""
    records = []
    for i in range(frames):
        offset = i * 500
        if unanchored:
            blocks = [_block(368, 411, kind="unanchored", top_kind="background_run",
                             bottom_kind="unanchored_island", bottom_observed=False,
                             digest=digest),
                      _block(517, 1400 + i, kind="selectable", top_kind="unanchored_island")]
            runs = [((300, 368), "clipped"), ((411, 517), "unanchored_island")]
        else:
            # Pre-check shape: the strip was MERGED into the clipped card below it, which is
            # exactly the bug — one block claiming a top 149px above any real content.
            blocks = [_block(368, 1000 + i, kind="partial")]
            runs = [((300, 368), "clipped"), ((411, 517), "too_long")]
        records.append(_geometry_frame(i, offset, blocks, runs))
    return records


def _refused_run(tmp_path, geometry, *, sidecar="item_index_refused_ab12_evidence.json",
                 named=None, name="run_geometry_sidecar"):
    run = tmp_path / name
    run.mkdir(parents=True)
    if geometry is not None:
        (run / sidecar).write_text(json.dumps(
            {"schema_version": 7, "reason": "index contradicts itself", "runtime": {},
             "frames": [], "all_frame_geometry": geometry, "pair_evidence": []}))
    (run / "actions.jsonl").write_text(json.dumps(
        {"action": "item_index_refused", "reason": "index contradicts itself",
         "steps_px": [500, 500], "evidence_sidecar": sidecar if named is None else named}) + "\n")
    return run


def _geometry_md(run):
    return bugreport._item_index_geometry_md(
        (run / "actions.jsonl").read_text().splitlines(), run)


def test_geometry_section_names_the_pinned_top_edge_the_refusal_only_hinted_at(tmp_path):
    """BUG REPORT 2026-08-28: an item-index refusal printed its own prose and the frame geometry
    summary, and an operator could not have found the cause from either. The cause -- Hinge
    10.1.0 pinning a profile header INSIDE the analysed band, whose non-gutter background run
    got absorbed so 43 rows of non-scrolling chrome merged into the card below -- was fully
    determined by the sidecar the report NAMED and never opened."""
    run = _refused_run(tmp_path, _pinned_header_geometry(frames=6))

    md = _geometry_md(run)

    assert "`item_index_refused_ab12_evidence.json` (6 frames of geometry)" in md
    assert ("leading block began at frame row 368 with an UNOBSERVED top edge "
            "(background_run) on 6 of 6 frames") in md
    assert "the page offset moved across 0..2500px" in md
    assert ("the first background run below that top edge began at frame row 411 on 6 of 6 "
            "frames, classified `too_long` on 6") in md
    assert "the run lay INSIDE the leading block instead of bounding it" in md
    assert "on 6 of those 6 frames" in md


def test_geometry_section_says_an_old_sidecar_predates_the_screen_fixed_check(tmp_path):
    """No `unanchored` record at all is NOT proof that the strip is page content. Say the check
    was unavailable, never print a false negative."""
    run = _refused_run(tmp_path, _pinned_header_geometry(frames=4))

    md = _geometry_md(run)

    assert "screen-fixed verdict: unavailable" in md
    assert "records no `unanchored` block at all" in md
    assert "predates segment.py's screen-fixed island check" in md
    assert "PROVEN screen-fixed" not in md
    assert "NOT proven" not in md


def test_geometry_section_proves_a_screen_fixed_strip_when_the_sidecar_carries_digests(tmp_path):
    """The three conditions `item_index._screen_fixed_islands` applies, re-derived here: two
    distinct offsets, a span at least the strip's own height, and one unanimous digest."""
    run = _refused_run(tmp_path, _pinned_header_geometry(
        frames=6, unanchored=True, digest="d" * 64))

    md = _geometry_md(run)

    assert ("screen-fixed verdict for the unanchored strip at frame rows 368..411 "
            "(frames 0, 1, 2, 3, 4, 5): PROVEN screen-fixed") in md
    assert "identical pixels at 6 page offsets spanning 2500px, more than its own 43px height" in md
    assert "held out of the index entirely" in md


def test_geometry_section_names_which_screen_fixed_condition_failed(tmp_path):
    """Each refusal reason must derive from ITS OWN precondition, not a fixed sentence."""
    one_offset = [f for f in _pinned_header_geometry(frames=3, unanchored=True, digest="a" * 64)]
    for frame in one_offset:
        frame["offset_px"] = 900
    assert "was seen at one page offset (900)" in _geometry_md(
        _refused_run(tmp_path, one_offset, name="one_offset"))

    narrow = _pinned_header_geometry(frames=2, unanchored=True, digest="a" * 64)
    narrow[1]["offset_px"] = 20                       # 20px of scroll against a 43px strip
    assert ("was seen across only 20px of scroll, less than its own 43px height"
            in _geometry_md(_refused_run(tmp_path, narrow, name="narrow")))

    animating = _pinned_header_geometry(frames=3, unanchored=True, digest="a" * 64)
    animating[1]["blocks"][0]["content_digest"] = "b" * 64
    assert ("showed 2 different pixel contents across 3 offsets"
            in _geometry_md(_refused_run(tmp_path, animating, name="animating")))

    undecidable = _pinned_header_geometry(frames=3, unanchored=True)   # no digest recorded
    assert ("UNDECIDABLE FROM THIS FILE" in _geometry_md(
        _refused_run(tmp_path, undecidable, name="undecidable")))


def test_geometry_section_never_claims_a_pinned_strip_a_capture_does_not_have(tmp_path):
    """The second real refusal on disk (`data/hinge_debug/78d364c5527d`) has a leading block
    that MOVES: 21 of its 26 frames start at the band edge and 5 start at 5 different rows. A
    section that reported its modal row as a fixed position would invent the incident's cause
    on a capture that never had it."""
    geometry = [
        _geometry_frame(i, i * 400,
                        [_block(300 + i * 137, 1500 + i * 20, top_kind="band_edge")],
                        [((300 + i * 137 - 30, 300 + i * 137), "clipped"),
                         ((1500 + i * 20, 1553 + i * 20), "gutter")])
        for i in range(5)]
    run = _refused_run(tmp_path, geometry, name="unpinned")

    md = _geometry_md(run)

    assert "NO leading strip holds a fixed frame position in this capture" in md
    assert "the remaining 4 began at 4 other row(s)" in md
    assert "what screen-pinned chrome looks like" not in md
    assert "sat at 5 different frame rows" in md
    assert "so it moves with the page rather than holding one position" in md
    assert "PROVEN screen-fixed" not in md


def test_geometry_section_is_silent_when_the_sidecar_is_absent(tmp_path):
    run = _refused_run(tmp_path, None, name="absent")           # record names it; file is gone

    assert _geometry_md(run) == ""
    assert "refused-capture leading-edge geometry" not in bugreport._one_debug_dir_md(
        "hinge", {"debug_dir": str(tmp_path)})


@pytest.mark.parametrize("body", [
    "{not json at all",
    '{"schema_version": 7, "all_frame_geometry": [{"local_frame_index": 0, "blocks":',  # truncated
    '{"all_frame_geometry": "not a list"}',
    '{"all_frame_geometry": 12345}',
    '["a", "top-level list"]',
    '{"all_frame_geometry": [null, 7, {"blocks": []}, {"blocks": [{"frame_rows": ["x", 4]}]}]}',
    # Rows that are not integers, and rows that are inverted, on a frame that IS indexed --
    # so the row parser is genuinely reached rather than skipped for want of a frame index.
    '{"all_frame_geometry": [{"local_frame_index": 0, "offset_px": 0,'
    ' "blocks": [{"frame_rows": ["x", 4], "top_kind": "background_run"}]}]}',
    '{"all_frame_geometry": [{"local_frame_index": 0, "offset_px": 0,'
    ' "blocks": [{"frame_rows": [900, 4], "top_kind": "background_run"}]}]}',
    '{"all_frame_geometry": [{"local_frame_index": true, "offset_px": 0,'
    ' "blocks": [{"frame_rows": [368, 411], "top_kind": "background_run"}]}]}',
])
def test_geometry_section_degrades_to_silence_on_a_malformed_sidecar(tmp_path, body):
    """A bug report must never raise while describing a bug, and a half-written sidecar is the
    normal shape of one: the refusal writes it while the run is still failing."""
    run = _refused_run(tmp_path, _pinned_header_geometry(frames=2), name=f"bad{abs(hash(body))}")
    (run / "item_index_refused_ab12_evidence.json").write_text(body)

    assert _geometry_md(run) == ""


def test_geometry_section_refuses_a_sidecar_path_outside_the_run_directory(tmp_path):
    """A sidecar filename comes out of a JSONL log, which is diagnostic input and never
    authority to read an arbitrary path -- the same rule `_shot_digest` applies to screenshots."""
    outside = tmp_path / "outside_evidence.json"
    outside.write_text(json.dumps({"all_frame_geometry": _pinned_header_geometry(frames=3)}))
    reads = []
    real_read = Path.read_text

    def spy(self, *a, **kw):
        reads.append(str(self))
        return real_read(self, *a, **kw)

    for traversal in ("../outside_evidence.json", str(outside), "/etc/passwd",
                      "evidence.json.png", "evidence", "sub/dir/evidence.json"):
        run = _refused_run(tmp_path, _pinned_header_geometry(frames=3),
                           named=traversal, name=f"traversal_{abs(hash(traversal))}")
        reads.clear()
        original = Path.read_text
        Path.read_text = spy
        try:
            md = _geometry_md(run)
        finally:
            Path.read_text = original
        assert md == "", traversal
        assert "leading block began at frame row 368" not in md
        # actions.jsonl is read by the test itself; nothing else may be opened.
        assert [r for r in reads if r.endswith(".json")] == [], (traversal, reads)


def test_geometry_section_refuses_a_sidecar_symlink_outside_the_run_directory(tmp_path):
    """A bare filename is still unsafe when it is a symlink to an arbitrary JSON file."""
    outside = tmp_path / "outside_evidence.json"
    outside.write_text(json.dumps({"all_frame_geometry": _pinned_header_geometry(frames=3)}))
    run = _refused_run(tmp_path, None, name="symlink")
    (run / "item_index_refused_ab12_evidence.json").symlink_to(outside)

    assert _geometry_md(run) == ""


def test_geometry_section_reads_only_the_most_recent_refusals(tmp_path):
    """Capped like every other repeated-row section in this file."""
    run = tmp_path / "many"
    run.mkdir(parents=True)
    records = []
    for i in range(5):
        sidecar = f"item_index_refused_{i:04d}_evidence.json"
        (run / sidecar).write_text(json.dumps(
            {"all_frame_geometry": _pinned_header_geometry(frames=3)}))
        records.append({"action": "item_index_refused", "reason": "r",
                        "evidence_sidecar": sidecar})
    (run / "actions.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")

    md = _geometry_md(run)

    assert md.count("frames of geometry") == bugreport._ITEM_INDEX_GEOMETRY_SIDECARS_SHOWN
    assert "item_index_refused_0004_evidence.json" in md      # newest kept
    assert "item_index_refused_0000_evidence.json" not in md  # oldest dropped


def test_geometry_section_is_wired_into_the_debug_log_block(tmp_path):
    _refused_run(tmp_path, _pinned_header_geometry(frames=6))

    md = bugreport._one_debug_dir_md("hinge", {"debug_dir": str(tmp_path)})

    assert ("refused-capture leading-edge geometry (read from the sidecar the refusal names):"
            in md)
    assert "leading block began at frame row 368" in md
    # It sits with the refusal it explains, not after the tail.
    assert md.index("leading block began at frame row 368") < md.index("actions.jsonl (tail)")


def test_geometry_section_will_not_read_an_oversized_sidecar(tmp_path, monkeypatch):
    """The cap must be what stops the read, so this file is one the section WOULD render:
    same bytes, cap raised, and the pinned top edge comes straight back."""
    run = _refused_run(tmp_path, _pinned_header_geometry(frames=6), name="huge")
    sidecar = run / "item_index_refused_ab12_evidence.json"
    sidecar.write_text(sidecar.read_text() + " " * 4096)     # trailing space: still valid JSON
    size = sidecar.stat().st_size

    monkeypatch.setattr(bugreport, "_ITEM_INDEX_GEOMETRY_SIDECAR_BYTES", size - 1)
    assert _geometry_md(run) == ""

    monkeypatch.setattr(bugreport, "_ITEM_INDEX_GEOMETRY_SIDECAR_BYTES", size)
    assert "leading block began at frame row 368" in _geometry_md(run)


def test_targeting_readiness_renders_the_content_band_in_pixel_rows(tmp_path):
    """`content_band: [0.125, 0.875]` is load-bearing context for every geometry row this report
    prints, and nobody checks two fractions against "frame row 368" in their head."""
    from operation_love import targeting_policy as tp
    import yaml

    tp.install_accepted_still_photo_assumption(tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION, device="synthetic-pixel",
        hinge_version_name="10.0.1", accepted_at="2026-08-21", rationale="accepted"))
    try:
        path = _targeting_config(tmp_path, licence_key="still_photo_assumption_acceptance")
        raw = yaml.safe_load(Path(path).read_text())
        raw["apps"]["hinge"]["targeting_calibration"] = _valid_targeting_calibration()
        Path(path).write_text(yaml.safe_dump(raw))
        md = bugreport._targeting_readiness_md(path)
    finally:
        tp._reset_installed_still_photo_bound_for_tests()

    assert "content_band=[0.125, 0.875]" in md
    assert "that content_band analyses frame rows 300..2100 of 2400" in md


@pytest.mark.parametrize("band,size,expected", [
    ([0.125, 0.875], [1080, 2400], (300, 2100, 2400)),
    ([0.0, 1.0], [1080, 2400], (0, 2400, 2400)),
    ([0.9, 0.1], [1080, 2400], (2160, 2161, 2400)),   # clamped, never inverted or empty
    (None, [1080, 2400], None),
    ([0.1], [1080, 2400], None),
    (["a", "b"], [1080, 2400], None),
    ([0.1, 0.9], None, None),
    ([0.1, 0.9], [1080, 1], None),
    ([float("nan"), 0.9], [1080, 2400], None),
    ([True, 0.9], [1080, 2400], None),
    ([0.1, 0.9], [1080, True], None),
])
def test_content_band_rows_matches_the_driver_and_never_raises(band, size, expected):
    """Same arithmetic as `hinge._content_rows`; a second drifting copy would be worse than
    printing nothing."""
    assert bugreport._content_band_rows(band, size) == expected


def test_content_band_rows_agrees_with_the_driver_it_mirrors():
    from operation_love.drivers import hinge

    for band in ([0.125, 0.875], [0.0, 1.0], [0.3, 0.31], [0.9, 0.1]):
        for size in (2400, 1920, 2):
            assert (bugreport._content_band_rows(band, [1080, size])[:2]
                    == hinge._content_rows(tuple(band), size))
