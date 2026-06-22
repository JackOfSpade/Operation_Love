"""Hinge driver — Android (emulator or phone) over ADB via uiautomator2.

Element-based (resource-id / text), OS-agnostic (ADB runs on macOS/Windows/Linux).
Hinge is mobile-only, so this drives a virtual Android phone (an emulator — no
physical Android device required). Unlike Bumble, Hinge lets you like a specific
photo/prompt WITH a comment, so the opener is sent at like-time here.

If the emulator is closed or the ADB / uiautomator2 connection drops mid-run, the
device calls below raise DriverClosed (parity with Bumble's browser-closed path)
so the worker stops the run cleanly and flushes buffered labels, instead of
treating a disconnect as a crash and restarting with backoff.

⚠️ NEEDS LIVE VERIFICATION ON YOUR MACHINE (batched with the other live steps).
Resource-ids / gestures below are best-effort and config-overridable
(config.yaml -> apps.hinge). Confirm with `uiautomator2`'s inspector / `uiautodev`.
"""
from __future__ import annotations

import time

from .base import DatingAppDriver, DriverClosed
from ..human import human_delay
from ..perception.capture import Profile

_OBSERVE_POLL_S = 0.35      # internal sampling cadence for your manual tap (not app-facing)

DEFAULTS = {
    "package": "co.hinge.app",
    "scroll_captures": 10,       # maximum screenshots taken while scrolling a profile
    "ids": {
        "like": "co.hinge.app:id/like_button",
        "pass": "co.hinge.app:id/skip_button",
        "comment_box": "co.hinge.app:id/comment_edit_text",
        "send_like": "co.hinge.app:id/send_like_button",
        "prompt_text": "co.hinge.app:id/prompt_answer",
        "empty": "co.hinge.app:id/empty_state",
    },
}

# Exception class names that mean the emulator/ADB/uiautomator2 link is gone (not a
# transient app error the worker should retry). Matched by name so we don't have to
# import uiautomator2 / adbutils / requests just to classify their errors.
_DEVICE_LOST_TYPES = frozenset({
    "DeviceError",            # uiautomator2.exceptions.DeviceError
    "GatewayError",           # uiautomator2 atx-agent gateway gone
    "UiautomatorQuitError",   # uiautomator2 service quit
    "ConnectError",           # uiautomator2 / httpx connect failure
    "AdbError",               # adbutils
    "AdbConnectionError",
    "ConnectionError",        # builtin + requests.exceptions.ConnectionError
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
})

# Conservative message fragments (lower-cased) for the same condition, to catch
# disconnects surfaced as a plain RuntimeError/OSError with a descriptive message.
_DEVICE_LOST_PHRASES = (
    "device offline",
    "device not found",
    "no devices/emulators found",
    "connection refused",
    "connection reset",
    "connection aborted",
    "cannot connect to",
    "failed to connect",
    "broken pipe",
    "remote disconnected",
)


def _is_device_lost_error(error: BaseException) -> bool:
    """True only for a genuine device/connection loss — kept conservative so a real
    app bug is never silently swallowed as a clean 'closed' stop."""
    if type(error).__name__ in _DEVICE_LOST_TYPES:
        return True
    msg = str(error).lower()
    if any(p in msg for p in _DEVICE_LOST_PHRASES):
        return True
    if "uiautomator" in msg and ("not running" in msg or "crash" in msg or "quit" in msg):
        return True
    if "adb" in msg and ("offline" in msg or "not found" in msg or "closed" in msg):
        return True
    return False


def _raise_driver_closed_if_device_lost(exc: Exception) -> None:
    if _is_device_lost_error(exc):
        raise DriverClosed("Hinge device/emulator was disconnected") from exc


class HingeDriver(DatingAppDriver):
    def __init__(self, cfg):
        app_cfg = (getattr(cfg, "apps", {}) or {}).get("hinge", {})
        self.serial = app_cfg.get("serial") or None
        self.package = app_cfg.get("package", DEFAULTS["package"])
        self.scroll_captures = max(1, int(app_cfg.get("scroll_captures", DEFAULTS["scroll_captures"])))
        self.ids = {**DEFAULTS["ids"], **app_cfg.get("ids", {})}
        self.d = None

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        import uiautomator2 as u2  # type: ignore  # lazy: optional [hinge] extra, runtime only

        self.d = u2.connect(self.serial) if self.serial else u2.connect()
        self.d.app_start(self.package, use_monkey=True)

    def close(self) -> None:
        self.d = None  # uiautomator2 connection is stateless; nothing to tear down

    # --- device I/O (all device access funnels through these so a dropped
    #     emulator/ADB link becomes DriverClosed instead of a raw crash) -----
    def _screenshot(self) -> bytes:
        try:
            return self.d.screenshot(format="raw")              # full-screen frame as bytes
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_device_lost(exc)
            raise

    def _exists(self, rid: str) -> bool:
        try:
            return bool(self.d(resourceId=rid).exists)
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_device_lost(exc)
            raise

    def _prompt_texts(self) -> list[str]:
        try:
            out: list[str] = []
            for el in self.d(resourceId=self.ids["prompt_text"]):
                txt = (el.get_text() or "").strip()
                if txt:
                    out.append(txt)
            return out
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_device_lost(exc)
            raise

    def _click(self, rid: str) -> None:
        try:
            self.d(resourceId=rid).click()
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_device_lost(exc)
            raise

    def _set_text(self, rid: str, text: str) -> None:
        try:
            self.d(resourceId=rid).set_text(text)
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_device_lost(exc)
            raise

    def _swipe_up(self) -> None:
        try:
            self.d.swipe_ext("up", scale=0.8)                   # scroll the profile
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_device_lost(exc)
            raise

    # --- capture --------------------------------------------------------
    def _capture_current(self) -> Profile:
        photos: list[bytes] = []
        prompts: list[tuple[str, str]] = []
        seen_frames: set[tuple[bytes, tuple[str, ...]]] = set()
        for i in range(self.scroll_captures):
            shot = self._screenshot()
            texts = self._prompt_texts()
            sig = (shot, tuple(texts))
            if sig in seen_frames:
                break
            seen_frames.add(sig)
            photos.append(shot)
            prompts.extend(("", txt) for txt in texts)
            if i < self.scroll_captures - 1:
                self._swipe_up()
                time.sleep(human_delay(0.5))                    # settle + "look" before next frame
        # dedupe prompts preserving order
        seen, uniq = set(), []
        for q, a in prompts:
            if a not in seen:
                seen.add(a)
                uniq.append((q, a))
        return Profile(photos=photos, prompts=uniq, meta={"app": "hinge"})

    def next_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def out_of_profiles(self) -> bool:
        return self._exists(self.ids["empty"])

    # --- actions --------------------------------------------------------
    def like(self, opener: str | None = None) -> None:
        # NORMAL like only — never send a Rose (Hinge's super-like). Roses/boosts
        # are the owner's manual call.
        self._click(self.ids["like"])
        if opener and self._exists(self.ids["comment_box"]):
            time.sleep(human_delay(0.8))                        # comment box animates in; you read/think
            self._set_text(self.ids["comment_box"], opener)     # opener sent with the like
            time.sleep(human_delay(0.6))                        # pause after typing, before sending
        self._click(self.ids["send_like"])

    def dislike(self) -> None:
        self._click(self.ids["pass"])

    # --- observe mode (shadow learning) ---------------------------------
    def current_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None) -> bool | None:
        """Block until YOU manually like/pass the current card.

        Android has no global tap callback (unlike Bumble's DOM click listener),
        so we poll the UI for each action's observable effect:
          LIKE — tapping a like heart opens Hinge's comment / "Send Like" sheet
                 (send_like / comment_box appear). We wait for that sheet to
                 close with the deck advanced (the like was actually sent) ->
                 True. A cancelled like (sheet closes, same card stays) is
                 ignored and we keep watching.
          PASS — the card is dismissed and the next profile loads with no like
                 sheet -> the prompt-text signature changes -> False.
          none — the deck empties, stop is requested, or we hit the timeout -> None.
                 timeout=None waits indefinitely.

        Returns True (liked), False (passed), or None. A device disconnect during
        the wait raises DriverClosed (handled by the worker as a clean stop).

        ⚠️ The tap-detection signals (which ids the like sheet exposes) need live
        confirmation on a real Hinge build; ids are config-overridable (see
        ops/RUNBOOK.md). The polling/return logic is unit-tested offline.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        baseline = self._observe_state()[1]
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():         # Stop pressed -> don't wait for a tap
                return None
            if self.out_of_profiles():
                return None
            like_open, sig = self._observe_state()
            if like_open:
                sent = self._await_like_sent(baseline, deadline, should_stop)
                if sent is None:
                    return None
                if sent:
                    return True                       # like sent
                baseline = self._observe_state()[1]   # cancelled -> resync, keep watching
            elif sig and sig != baseline:             # card advanced, no like sheet -> pass
                return False
            time.sleep(_OBSERVE_POLL_S)
        return None

    def _observe_state(self) -> tuple[bool, tuple[str, ...]]:
        """Cheap UI probes for observe mode: (is the like/send sheet open?, the
        current card's prompt-text signature). The signature changes when the
        deck advances to a new profile, which is how a pass is detected."""
        like_open = self._exists(self.ids["send_like"]) or self._exists(self.ids["comment_box"])
        return like_open, tuple(self._prompt_texts())

    def _await_like_sent(self, baseline: tuple[str, ...], deadline: float | None,
                         should_stop=None) -> bool | None:
        """Once the send-like sheet is open, wait for it to close. True if the
        like was sent (deck advanced or emptied); False if cancelled (same card)."""
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():
                return None
            if self.out_of_profiles():
                return True                           # last like sent; deck now empty
            like_open, sig = self._observe_state()
            if not like_open:
                return bool(sig) and sig != baseline  # advanced -> sent; unchanged -> cancelled
            time.sleep(_OBSERVE_POLL_S)
        return False
