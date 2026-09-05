"""Native operator notification behavior, without invoking the host UI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import operation_love.notifications as notifications
import pytest


def test_training_decision_notification_uses_macos_banner_and_independent_sound(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = "presented\n" if argv[0] == "/usr/bin/osascript" else None
        return subprocess.CompletedProcess(argv, 0, stdout=stdout)

    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)
    monkeypatch.setattr(notifications, "_MACOS_SOUND", Path(__file__))

    assert notifications.notify_training_decision_ready() is True
    notification_argv, notification_kwargs = calls[0]
    sound_argv, sound_kwargs = calls[1]
    assert notification_argv[:4] == ["/usr/bin/osascript", "-l", "JavaScript", "-e"]
    assert "NSUserNotificationCenter" in notification_argv[4]
    assert "notification.presented" in notification_argv[4]
    assert "soundName" not in notification_argv[4]
    assert sound_argv == ["/usr/bin/afplay", str(notifications._MACOS_SOUND)]
    assert notification_kwargs["timeout"] == sound_kwargs["timeout"] == 5


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/osascript").exists(),
    reason="requires the macOS JavaScript for Automation parser",
)
def test_training_decision_jxa_is_accepted_by_the_real_macos_parser():
    # Keep the production statements in a dead branch: this validates their syntax without
    # posting a real notification during the test suite.
    result = subprocess.run(
        ["/usr/bin/osascript", "-l", "JavaScript", "-e",
         f"if (false) {{ {notifications._TRAINING_NOTIFICATION_JXA} }}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_training_decision_notification_is_a_noop_off_macos(monkeypatch):
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(
        notifications.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    assert notifications.notify_training_decision_ready() is False


def test_training_decision_notification_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_MACOS_SOUND", Path(__file__))
    monkeypatch.setattr(
        notifications.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("osascript", 5)),
    )

    assert notifications.notify_training_decision_ready() is False


def test_sound_still_plays_when_notification_center_suppresses_banner(monkeypatch):
    def fake_run(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout="not-presented\n" if argv[0] == "/usr/bin/osascript" else None)

    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_MACOS_SOUND", Path(__file__))
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify_training_decision_ready() is True
    assert notifications.recent_training_alerts()[-1] == {
        "at": notifications.recent_training_alerts()[-1]["at"],
        "channel": "macos-host",
        "notification": "not-presented",
        "sound": "played",
    }


@pytest.mark.parametrize(
    "outcome", ["requested", "denied", "permission-default", "unsupported", "error"])
def test_browser_notification_outcomes_are_bounded_and_recorded(outcome):
    assert notifications.record_training_browser_notification(outcome) is True
    assert notifications.recent_training_alerts()[-1]["notification"] == outcome


def test_browser_notification_rejects_unbounded_diagnostic_text():
    before = notifications.recent_training_alerts()
    assert notifications.record_training_browser_notification("raw browser error: secret") is False
    assert notifications.recent_training_alerts() == before
