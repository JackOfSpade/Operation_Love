"""Best-effort operator alerts for actionable Training checkpoints.

The Hub browser owns the primary visible notification because it has a stable, user-visible
notification permission.  On macOS this module also posts a native fallback and plays the sound
independently: Notification Center is allowed to suppress a banner *and* its attached sound, so
one deprecated API call cannot honestly prove either signal reached the operator.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


_ALERT_HISTORY: deque[dict[str, str]] = deque(maxlen=32)
_ALERT_HISTORY_LOCK = threading.Lock()
_MACOS_SOUND = Path("/System/Library/Sounds/Glass.aiff")

# NSUserNotificationCenter is only a fallback for a browser notification. Apple has deprecated
# it and explicitly permits Notification Center to suppress presentation. Checking ``presented``
# after delivery is still more truthful than the old implementation, which treated osascript's
# zero exit status as proof that the operator saw and heard an alert.
_TRAINING_NOTIFICATION_JXA = (
    'ObjC.import("Foundation"); '
    'const notification = $.NSUserNotification.alloc.init; '
    'notification.title = "Operation Love"; '
    'notification.subtitle = "Training profile ready"; '
    'notification.informativeText = "Choose Like or Dislike in the Hub."; '
    'notification.identifier = "operation-love-training-" + Date.now(); '
    'const center = $.NSUserNotificationCenter.defaultUserNotificationCenter; '
    'center.deliverNotification(notification); '
    '$.NSThread.sleepForTimeInterval(0.35); '
    'notification.presented ? "presented" : "not-presented";'
)


def _record_alert(channel: str, **outcomes: str) -> None:
    row = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel": channel,
        **{name: str(value) for name, value in outcomes.items()},
    }
    with _ALERT_HISTORY_LOCK:
        _ALERT_HISTORY.append(row)


def recent_training_alerts() -> list[dict[str, str]]:
    """Return a detached, redaction-safe alert history for diagnostics."""
    with _ALERT_HISTORY_LOCK:
        return [dict(row) for row in _ALERT_HISTORY]


def record_training_browser_notification(outcome: str) -> bool:
    """Record a browser's bounded notification outcome.

    The HTTP boundary accepts only these fixed values, so neither browser-controlled prose nor
    checkpoint capability tokens can leak into logs or bug reports.
    """
    if outcome not in {"requested", "denied", "permission-default", "unsupported", "error"}:
        return False
    _record_alert("hub-browser", notification=outcome)
    print(f"Training alert: Hub browser notification={outcome}.")
    return True


def _macos_notification_outcome() -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", _TRAINING_NOTIFICATION_JXA],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error-{type(exc).__name__}"
    if result.returncode != 0:
        return f"error-exit-{result.returncode}"
    reported = (result.stdout or "").strip()
    return reported if reported in {"presented", "not-presented"} else "submitted"


def _macos_sound_outcome() -> str:
    if not _MACOS_SOUND.is_file():
        return "unavailable"
    try:
        result = subprocess.run(
            ["/usr/bin/afplay", str(_MACOS_SOUND)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error-{type(exc).__name__}"
    return "played" if result.returncode == 0 else f"error-exit-{result.returncode}"


def notify_training_decision_ready() -> bool:
    """Alert a macOS operator that a Training card is ready for review.

    This attention aid never changes the approval protocol or cancels a checkpoint. The sound is
    deliberately separate from Notification Center so disabled banners, Focus, or a legacy API
    no-op cannot silence it as a side effect.
    """
    if sys.platform != "darwin":
        _record_alert("host", notification="unsupported", sound="unsupported")
        return False

    notification = _macos_notification_outcome()
    sound = _macos_sound_outcome()
    _record_alert("macos-host", notification=notification, sound=sound)
    print(f"Training alert: macOS fallback notification={notification}; sound={sound}.")
    return notification == "presented" or sound == "played"
