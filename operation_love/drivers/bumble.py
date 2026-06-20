"""Bumble driver — web app via Playwright (sync API), element-based.

OS-agnostic: Playwright downloads the right browser per OS. Runs headful (a
real, backgrounded window — better for avoiding bot-detection, and it never
steals your cursor because it's driven over CDP) or headless (for an always-on
server). Login persists via a user-data dir, so you sign in once by hand.

⚠️ PHASE 1 — STILL NEEDS LIVE VERIFICATION ON YOUR MACHINE.
The CSS selectors below are best-effort guesses; Bumble's DOM must be inspected
live to confirm them. They're config-overridable (config.yaml -> apps.bumble.
selectors) so you can fix them without touching code. Run on your Mac with
headless:false, open DevTools, and adjust.

NOTE on openers: on Bumble (hetero mode) you can't send a message at swipe time
— matches message post-match (women-first), or via Bumble's profile-level
"Opening Moves". So `like(opener)` records/uses the opener where the flow allows
and otherwise ignores it. The per-swipe opener flow is Hinge's model; this is
wired through for parity and future Bumble "Opening Moves" support.
"""
from __future__ import annotations

from pathlib import Path

from .base import DatingAppDriver
from ..perception.capture import Profile

DEFAULT_SELECTORS = {
    "photo": '.encounters-album__story-content, [data-qa-role="encounters-story-photo"]',
    "bio": '[data-qa-role="encounters-story-about"], .encounters-story-about',
    "prompt": '.encounters-story-section--profile .pill, .encounters-story-about__field',
    "like": '[data-qa-role="encounters-action-like"]',
    "pass": '[data-qa-role="encounters-action-dislike"]',
    "empty": '[data-qa-role="encounters-out-of-people"], .encounters-out-of-people',
}


class BumbleDriver(DatingAppDriver):
    def __init__(self, cfg):
        app_cfg = (getattr(cfg, "apps", {}) or {}).get("bumble", {})
        self.url = app_cfg.get("url", "https://bumble.com/app")
        self.headless = bool(app_cfg.get("headless", False))
        self.user_data_dir = Path(app_cfg.get("user_data_dir", "./data/bumble_profile"))
        self.selectors = {**DEFAULT_SELECTORS, **app_cfg.get("selectors", {})}
        self.nav_timeout_ms = int(app_cfg.get("nav_timeout_ms", 15000))
        self._pw = None
        self._ctx = None
        self.page = None

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        from playwright.sync_api import sync_playwright  # lazy: only needed at runtime

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        # Persistent context => your manual login is remembered across runs.
        self._ctx = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            viewport={"width": 1280, "height": 900},
        )
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self.page.set_default_timeout(self.nav_timeout_ms)
        self.page.goto(self.url, wait_until="domcontentloaded")
        # First run: if not logged in, sign in by hand in the opened window; the
        # user-data dir persists the session for subsequent runs.

    def close(self) -> None:
        try:
            if self._ctx:
                self._ctx.close()
        finally:
            if self._pw:
                self._pw.stop()
            self._pw = self._ctx = self.page = None

    # --- capture --------------------------------------------------------
    def _capture_current(self) -> Profile:
        return Profile(
            photos=self._capture_photos(),
            bio=self._text(self.selectors["bio"]),
            prompts=self._capture_prompts(),
            meta={"app": "bumble"},
        )

    def next_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    # --- observe mode (shadow learning) ---------------------------------
    def current_profile(self) -> Profile | None:
        if self.out_of_profiles():
            return None
        return self._capture_current()

    def wait_for_decision(self, timeout: float = 120.0) -> bool | None:
        """Detect YOUR manual like/pass on the current card.

        TODO(live): the robust implementation watches the vote network request
        Bumble fires when you like/pass (Playwright `page.expect_request` /
        `page.on("request")`), or hooks the like/pass buttons. The exact
        endpoint + how it encodes like-vs-pass must be confirmed live (Network
        tab) — same one-time, config-driven step as the selectors. Until then,
        observe mode for Bumble is not wired.
        """
        raise NotImplementedError(
            "Bumble observe hook needs live verification (vote request / button events)."
        )

    def _capture_photos(self) -> list[bytes]:
        # TODO(live): Bumble shows photos in a carousel; may need to click through
        # to load each. For now, screenshot every visible photo element.
        shots: list[bytes] = []
        for el in self.page.query_selector_all(self.selectors["photo"]):
            try:
                shots.append(el.screenshot())
            except Exception:  # noqa: BLE001
                continue
        return shots

    def _capture_prompts(self) -> list[tuple[str, str]]:
        # TODO(live): map Bumble's profile fields/prompts to (label, value) pairs.
        out: list[tuple[str, str]] = []
        for el in self.page.query_selector_all(self.selectors["prompt"]):
            txt = (el.inner_text() or "").strip()
            if txt:
                out.append(("", txt))
        return out

    def _text(self, selector: str) -> str:
        el = self.page.query_selector(selector)
        return (el.inner_text().strip() if el else "")

    # --- actions --------------------------------------------------------
    def like(self, opener: str | None = None) -> None:
        self.page.click(self.selectors["like"])
        # NOTE: opener is not sent here on standard Bumble (post-match / women-first).
        # Hook point for Bumble "Opening Moves" once that flow is mapped live.

    def dislike(self) -> None:
        self.page.click(self.selectors["pass"])

    def out_of_profiles(self) -> bool:
        return self.page.query_selector(self.selectors["empty"]) is not None
