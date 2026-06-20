"""Hinge driver — Android (emulator or phone) over ADB via uiautomator2.

Element-based (resource-id / text), OS-agnostic (ADB runs on macOS/Windows/Linux).
Hinge is mobile-only, so this drives a virtual Android phone (an emulator — no
physical Android device required). Unlike Bumble, Hinge lets you like a specific
photo/prompt WITH a comment, so the opener is sent at like-time here.

⚠️ NEEDS LIVE VERIFICATION ON YOUR MACHINE (batched with the other live steps).
Resource-ids / gestures below are best-effort and config-overridable
(config.yaml -> apps.hinge). Confirm with `uiautomator2`'s inspector / `uiautodev`.
"""
from __future__ import annotations

import time

from .base import DatingAppDriver
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

    # --- capture --------------------------------------------------------
    def _capture_current(self) -> Profile:
        photos: list[bytes] = []
        prompts: list[tuple[str, str]] = []
        seen_frames: set[tuple[bytes, tuple[str, ...]]] = set()
        for i in range(self.scroll_captures):
            shot = self.d.screenshot(format="raw")              # full-screen frame as bytes
            texts: list[str] = []
            for el in self.d(resourceId=self.ids["prompt_text"]):
                txt = (el.get_text() or "").strip()
                if txt:
                    texts.append(txt)
            sig = (shot, tuple(texts))
            if sig in seen_frames:
                break
            seen_frames.add(sig)
            photos.append(shot)
            prompts.extend(("", txt) for txt in texts)
            if i < self.scroll_captures - 1:
                self.d.swipe_ext("up", scale=0.8)               # scroll the profile
                time.sleep(human_delay(0.5))                    # settle + "look" before next frame
        # dedupe prompts preserving order
        seen, uniq = set(), []
        for q, a in prompts:
            if a not in seen:
                seen.add(a); uniq.append((q, a))
        return Profile(photos=photos, prompts=uniq, meta={"app": "hinge"})

    def next_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def out_of_profiles(self) -> bool:
        return self.d(resourceId=self.ids["empty"]).exists

    # --- actions --------------------------------------------------------
    def like(self, opener: str | None = None) -> None:
        # NORMAL like only — never send a Rose (Hinge's super-like). Roses/boosts
        # are the owner's manual call.
        self.d(resourceId=self.ids["like"]).click()
        if opener:
            box = self.d(resourceId=self.ids["comment_box"])
            if box.exists:
                time.sleep(human_delay(0.8))                    # comment box animates in; you read/think
                box.set_text(opener)                            # opener sent with the like
                time.sleep(human_delay(0.6))                    # pause after typing, before sending
        self.d(resourceId=self.ids["send_like"]).click()

    def dislike(self) -> None:
        self.d(resourceId=self.ids["pass"]).click()

    # --- observe mode (shadow learning) ---------------------------------
    def current_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def wait_for_decision(self, timeout: float = 120.0, should_stop=None) -> bool | None:
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
          none — the deck empties or we hit the timeout -> None.

        Returns True (liked), False (passed), or None.

        ⚠️ The tap-detection signals (which ids the like sheet exposes) need live
        confirmation on a real Hinge build; ids are config-overridable (see
        ops/RUNBOOK.md). The polling/return logic is unit-tested offline.
        """
        deadline = time.monotonic() + timeout
        baseline = self._observe_state()[1]
        while time.monotonic() < deadline:
            if should_stop and should_stop():         # Stop pressed -> don't wait for a tap
                return None
            if self.out_of_profiles():
                return None
            like_open, sig = self._observe_state()
            if like_open:
                if self._await_like_sent(baseline, deadline):
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
        like_open = (self.d(resourceId=self.ids["send_like"]).exists
                     or self.d(resourceId=self.ids["comment_box"]).exists)
        texts = [t for t in (
            (el.get_text() or "").strip() for el in self.d(resourceId=self.ids["prompt_text"])
        ) if t]
        return like_open, tuple(texts)

    def _await_like_sent(self, baseline: tuple[str, ...], deadline: float) -> bool:
        """Once the send-like sheet is open, wait for it to close. True if the
        like was sent (deck advanced or emptied); False if cancelled (same card)."""
        while time.monotonic() < deadline:
            if self.out_of_profiles():
                return True                           # last like sent; deck now empty
            like_open, sig = self._observe_state()
            if not like_open:
                return bool(sig) and sig != baseline  # advanced -> sent; unchanged -> cancelled
            time.sleep(_OBSERVE_POLL_S)
        return False
