"""Bumble driver — web app via Playwright (sync API), element-based.

OS-agnostic: Playwright downloads the right browser per OS. Runs headful (a
real, backgrounded window — better for avoiding bot-detection, and it never
steals your cursor because it's driven over CDP) or headless (for an always-on
server). Login persists via a user-data dir, so you sign in once by hand.

⚠️ STILL NEEDS LIVE VERIFICATION ON YOUR MACHINE (one step).
The CSS selectors below are best-effort guesses; Bumble's DOM must be inspected
live to confirm them. They're config-overridable (config.yaml -> apps.bumble.
selectors) so you can fix them without touching code. Run headless:false, open
DevTools, and adjust — or just run `python -m tools.bumble_inspect`, which probes
every selector for you. Observe-mode like/pass detection rides on the SAME
like/pass selectors (see wait_for_decision), so confirming them is the only
live step — no network reverse-engineering.

NOTE on openers: on Bumble (hetero mode) you can't send a message at swipe time
— matches message post-match (women-first), or via Bumble's profile-level
"Opening Moves". So `like(opener)` records/uses the opener where the flow allows
and otherwise ignores it. The per-swipe opener flow is Hinge's model; this is
wired through for parity and future Bumble "Opening Moves" support.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import DatingAppDriver
from ..perception.capture import Profile

_OBSERVE_POLL_S = 0.15      # internal sampling cadence for your manual swipe (not app-facing)

DEFAULT_SELECTORS = {
    "photo": '.encounters-album__story-content, [data-qa-role="encounters-story-photo"]',
    "bio": '[data-qa-role="encounters-story-about"], .encounters-story-about',
    "prompt": '.encounters-story-section--profile .pill, .encounters-story-about__field',
    "like": '[data-qa-role="encounters-action-like"]',
    "pass": '[data-qa-role="encounters-action-dislike"]',
    # superlike is OBSERVE-ONLY: the bot never clicks it (super-likes/boosts are
    # the owner's manual decision). It exists here solely so observe mode can read
    # a manual super-swipe as a 'like' signal when learning taste.
    "superlike": '[data-qa-role="encounters-action-superswipe"]',
    "empty": '[data-qa-role="encounters-out-of-people"], .encounters-out-of-people',
}

# In-page status HUD: a fixed, click-through (pointer-events:none) overlay updated
# after every swipe so you watch progress on the page, not the terminal. Takes an
# app_view() snapshot. Builds the element once, then just refreshes its contents.
_OVERLAY_JS = """
(s) => {
  const a = s.app || {};
  let el = document.getElementById('oplove-hud');
  if (!el) {
    el = document.createElement('div');
    el.id = 'oplove-hud';
    el.style.cssText = [
      'position:fixed','top:14px','right:14px','z-index:2147483647','pointer-events:none',
      'font:12px/1.45 -apple-system,Segoe UI,Roboto,sans-serif','color:#e8e8ea',
      'background:rgba(18,18,22,0.88)','border:1px solid rgba(255,255,255,0.12)',
      'border-radius:12px','padding:10px 12px','min-width:198px',
      'box-shadow:0 6px 24px rgba(0,0,0,0.35)'
    ].join(';');
    document.body.appendChild(el);
  }
  const ready = !!s.ranker_ready;
  const pct = s.min_labels ? Math.min(100, Math.round(100 * s.labels / s.min_labels)) : 100;
  const dec = a.last_decision ? String(a.last_decision).toUpperCase() : '—';
  const decColor = a.last_decision === 'like' ? '#39d98a'
                 : (a.last_decision === 'pass' || a.last_decision === 'dislike') ? '#ff6b6b' : '#c9c9cf';
  const score = (a.last_score == null) ? '' : ' · ' + Number(a.last_score).toFixed(2);
  const budget = (s.budget_cap != null)
    ? '$' + Number(s.budget_spent).toFixed(2) + ' / $' + Number(s.budget_cap).toFixed(2)
    : '$' + Number(s.budget_spent).toFixed(2);
  el.innerHTML =
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
      '<b style="letter-spacing:.3px">Operation&nbsp;Love</b>' +
      '<span style="font-size:10px;padding:1px 6px;border-radius:999px;background:rgba(120,120,255,.25)">' +
        String(s.mode || '').toUpperCase() + '</span></div>' +
    '<div style="opacity:.7">bumble · ' + (a.state || '—') + '</div>' +
    '<div style="margin-top:5px">labels <b>' + s.labels + '</b> / ' + s.min_labels +
      ' <span style="color:' + (ready ? '#39d98a' : '#f0b429') + '">· ' + (ready ? 'ready' : 'defer') + '</span></div>' +
    '<div style="height:5px;background:rgba(255,255,255,.12);border-radius:3px;margin:4px 0 6px">' +
      '<div style="height:100%;width:' + pct + '%;background:' + (ready ? '#39d98a' : '#f0b429') + ';border-radius:3px"></div></div>' +
    '<div>last <b style="color:' + decColor + '">' + dec + '</b>' + score + '</div>' +
    '<div style="opacity:.7">swipes this run <b>' + (a.swipes_run || 0) + '</b></div>' +
    '<div style="opacity:.7">budget ' + budget + '</div>' +
    (ready ? '' : '<div style="margin-top:6px;opacity:.6;font-size:11px">swipe — learning your taste</div>');
}
"""


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
        from playwright.sync_api import sync_playwright  # type: ignore  # lazy: optional [bumble] extra, runtime only

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

    def render_status(self, status: dict) -> None:
        # Paint/refresh the in-page HUD. Best-effort: a navigation mid-eval or a
        # closed page must never interrupt the swipe loop.
        if not self.page:
            return
        try:
            self.page.evaluate(_OVERLAY_JS, status)
        except Exception:  # noqa: BLE001
            pass

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

    def _install_observe_listener(self) -> None:
        """Inject a one-time capture-phase click listener on the like/pass buttons.

        Records 'like'/'pass' on window when you click either action. Depends
        ONLY on the like/pass selectors (the same ones used to act), so once
        those are confirmed live there's nothing else to reverse-engineer — no
        network sniffing needed. Survives card changes (window-scoped).
        """
        self.page.evaluate(
            """([likeSel, passSel, superSel]) => {
                if (window.__oplove_obs) return;
                window.__oplove_obs = true;
                window.__oplove_decision = null;
                document.addEventListener('click', (e) => {
                    const t = e.target;
                    if (!t || !t.closest) return;
                    // a manual super-swipe is still a positive 'like' signal for taste-learning
                    if (t.closest(likeSel) || t.closest(superSel)) window.__oplove_decision = 'like';
                    else if (t.closest(passSel)) window.__oplove_decision = 'pass';
                }, true);   // capture phase: fires even if the app stops propagation
            }""",
            [self.selectors["like"], self.selectors["pass"], self.selectors["superlike"]],
        )

    def wait_for_decision(self, timeout: float = 120.0) -> bool | None:
        """Block until YOU manually like/pass the current card.

        Returns True (liked), False (passed), or None (deck emptied / timeout).
        Mouse clicks on the like/pass controls are detected; if you also use
        keyboard shortcuts, confirm coverage with tools/bumble_inspect.py.
        """
        self._install_observe_listener()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            decision = self.page.evaluate(
                "() => { const v = window.__oplove_decision; window.__oplove_decision = null; return v; }"
            )
            if decision == "like":
                return True
            if decision == "pass":
                return False
            if self.out_of_profiles():
                return None
            time.sleep(_OBSERVE_POLL_S)
        return None

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
        # NORMAL like only — never the super-swipe. Super-likes/boosts are the
        # owner's manual call (see DEFAULT_SELECTORS["superlike"]).
        self.page.click(self.selectors["like"])
        # NOTE: opener is not sent here on standard Bumble (post-match / women-first).
        # Hook point for Bumble "Opening Moves" once that flow is mapped live.

    def dislike(self) -> None:
        self.page.click(self.selectors["pass"])

    def out_of_profiles(self) -> bool:
        return self.page.query_selector(self.selectors["empty"]) is not None
