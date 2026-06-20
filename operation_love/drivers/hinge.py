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

from .base import DatingAppDriver
from ..perception.capture import Profile

DEFAULTS = {
    "package": "co.hinge.app",
    "scroll_captures": 5,        # screenshots taken while scrolling a profile
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
        self.scroll_captures = int(app_cfg.get("scroll_captures", DEFAULTS["scroll_captures"]))
        self.ids = {**DEFAULTS["ids"], **app_cfg.get("ids", {})}
        self.d = None

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        import uiautomator2 as u2  # lazy: only needed at runtime

        self.d = u2.connect(self.serial) if self.serial else u2.connect()
        self.d.app_start(self.package, use_monkey=True)

    def close(self) -> None:
        self.d = None  # uiautomator2 connection is stateless; nothing to tear down

    # --- capture --------------------------------------------------------
    def _capture_current(self) -> Profile:
        photos: list[bytes] = []
        prompts: list[tuple[str, str]] = []
        for i in range(self.scroll_captures):
            photos.append(self.d.screenshot(format="raw"))      # full-screen frame as bytes
            for el in self.d(resourceId=self.ids["prompt_text"]):
                txt = (el.get_text() or "").strip()
                if txt:
                    prompts.append(("", txt))
            if i < self.scroll_captures - 1:
                self.d.swipe_ext("up", scale=0.8)               # scroll the profile
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
        self.d(resourceId=self.ids["like"]).click()
        if opener:
            box = self.d(resourceId=self.ids["comment_box"])
            if box.exists:
                box.set_text(opener)                            # opener sent with the like
        self.d(resourceId=self.ids["send_like"]).click()

    def dislike(self) -> None:
        self.d(resourceId=self.ids["pass"]).click()

    # --- observe mode (shadow learning) ---------------------------------
    def current_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def wait_for_decision(self, timeout: float = 120.0) -> bool | None:
        """Detect YOUR manual like/pass.

        TODO(live): watch which control you tap (uiautomator2 watchers on the
        like/pass ids) or the screen transition. Returns True/False/None.
        Batched with the other live-verification steps — see ops/RUNBOOK.md.
        """
        raise NotImplementedError(
            "Hinge observe hook needs live verification (tap detection)."
        )
