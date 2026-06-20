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

from .base import DatingAppDriver, DriverClosed
from ..perception.capture import Profile

_OBSERVE_POLL_S = 0.15      # internal sampling cadence for your manual swipe (not app-facing)
_MIN_PROFILE_PHOTO_SIDE_PX = 160
_SIDEBAR_EXCLUSION_RATIO = 0.30
_DEFAULT_PHOTO_CAPTURE_STEPS = 8
_PHOTO_ADVANCE_SETTLE_S = 0.25
_ALBUM_PRELOADED_MIN = 3      # if the first frame yields >= this many photos, the album is fully loaded
_MAX_PHOTOS_PER_PROFILE = 12  # hard cap; real profiles top out ~6, so this bounds runaway re-capture


def _is_browser_closed_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "TargetClosedError":
        return True
    msg = str(exc).lower()
    return (
        "target page, context or browser has been closed" in msg
        or "browser has been closed" in msg
        or "page has been closed" in msg
    )


def _raise_driver_closed_if_browser_closed(exc: Exception) -> None:
    if _is_browser_closed_error(exc):
        raise DriverClosed("Bumble browser was closed") from exc


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
  el.style.display = 'block';   // re-show if a photo capture temporarily hid it
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
    '<div style="margin-top:5px">labels <b>' + s.labels + '</b> total ' +
      '<span style="color:' + (ready ? '#39d98a' : '#f0b429') + '">· ' +
      (ready ? 'ready' : ('ready after ' + s.min_labels)) + '</span></div>' +
    '<div style="height:5px;background:rgba(255,255,255,.12);border-radius:3px;margin:4px 0 6px">' +
      '<div style="height:100%;width:' + pct + '%;background:' + (ready ? '#39d98a' : '#f0b429') + ';border-radius:3px"></div></div>' +
    '<div>last <b style="color:' + decColor + '">' + dec + '</b>' + score + '</div>' +
    '<div style="opacity:.7">swipes this run <b>' + (a.swipes_run || 0) + '</b></div>' +
    '<div style="opacity:.7">budget ' + budget + '</div>' +
    (ready ? '' : '<div style="margin-top:6px;opacity:.6;font-size:11px">swipe — learning your taste</div>');
}
"""

# Blocking "processing" modal: a full-screen, click-intercepting backdrop with a
# spinner so a swipe can't land on the next card mid-embed. Pass a message to
# show it, null to hide. Idempotent (builds once, then toggles).
_BUSY_JS = """
(msg) => {
  let el = document.getElementById('oplove-busy');
  if (!msg) { if (el) el.style.display = 'none'; return; }
  if (!el) {
    const style = document.createElement('style');
    style.textContent = '@keyframes oplovespin{to{transform:rotate(360deg)}}';
    document.head.appendChild(style);
    el = document.createElement('div');
    el.id = 'oplove-busy';
    el.style.cssText = [
      'position:fixed','inset:0','z-index:2147483646','display:flex',
      'align-items:center','justify-content:center','pointer-events:auto',
      'background:rgba(10,10,14,0.55)',
      'font:600 17px/1.4 -apple-system,Segoe UI,Roboto,sans-serif','color:#fff'
    ].join(';');
    el.innerHTML =
      '<div style="background:rgba(20,20,26,0.96);border:1px solid rgba(255,255,255,0.14);' +
      'border-radius:16px;padding:22px 28px;text-align:center;box-shadow:0 12px 44px rgba(0,0,0,0.55)">' +
      '<div style="width:34px;height:34px;margin:0 auto 12px;border:3px solid rgba(255,255,255,0.25);' +
      'border-top-color:#fff;border-radius:50%;animation:oplovespin 0.8s linear infinite"></div>' +
      '<div id="oplove-busy-msg"></div></div>';
    document.body.appendChild(el);
  }
  el.style.display = 'flex';
  document.getElementById('oplove-busy-msg').textContent = msg;
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
        self.photo_capture_steps = max(
            1, int(app_cfg.get("photo_capture_steps", _DEFAULT_PHOTO_CAPTURE_STEPS))
        )
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
                try:
                    self._ctx.close()
                except Exception as exc:  # noqa: BLE001
                    if not _is_browser_closed_error(exc):
                        raise
        finally:
            if self._pw:
                try:
                    self._pw.stop()
                except Exception as exc:  # noqa: BLE001
                    if not _is_browser_closed_error(exc):
                        raise
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

    def render_busy(self, message: str | None = None) -> None:
        # Full-screen click-blocking modal so you can't swipe the next card while
        # this one is still embedding. message shows it; None hides it.
        if not self.page:
            return
        try:
            self.page.evaluate(_BUSY_JS, message)
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
        try:
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
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise

    def wait_for_decision(self, timeout: float = 120.0, should_stop=None) -> bool | None:
        """Block until YOU manually like/pass the current card.

        Returns True (liked), False (passed), or None (deck emptied / timeout /
        stop requested). Mouse clicks on the like/pass controls are detected; if
        you also use keyboard shortcuts, confirm coverage with bumble_inspect.py.
        """
        self._install_observe_listener()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if should_stop and should_stop():        # Stop pressed -> don't wait for a swipe
                return None
            try:
                decision = self.page.evaluate(
                    "() => { const v = window.__oplove_decision; window.__oplove_decision = null; return v; }"
                )
            except Exception as exc:  # noqa: BLE001
                _raise_driver_closed_if_browser_closed(exc)
                raise
            if decision == "like":
                return True
            if decision == "pass":
                return False
            if self.out_of_profiles():
                return None
            time.sleep(_OBSERVE_POLL_S)
        return None

    def _is_profile_photo_element(self, el) -> bool:
        try:
            box = el.bounding_box()
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            return False
        if not box:
            return False

        width = float(box.get("width") or 0)
        height = float(box.get("height") or 0)
        if width < _MIN_PROFILE_PHOTO_SIDE_PX or height < _MIN_PROFILE_PHOTO_SIDE_PX:
            return False

        viewport = getattr(self.page, "viewport_size", None) or {}
        viewport_width = float(viewport.get("width") or 0)
        if viewport_width:
            center_x = float(box.get("x") or 0) + width / 2
            if center_x < viewport_width * _SIDEBAR_EXCLUSION_RATIO:
                return False

        return True

    def _profile_photo_elements(self):
        return [
            el for el in self._query_selector_all(self.selectors["photo"])
            if self._is_profile_photo_element(el)
        ]

    def _largest_profile_photo_element(self):
        best = None
        best_area = 0.0
        for el in self._profile_photo_elements():
            try:
                box = el.bounding_box() or {}
            except Exception as exc:  # noqa: BLE001
                _raise_driver_closed_if_browser_closed(exc)
                continue
            area = float(box.get("width") or 0) * float(box.get("height") or 0)
            if area > best_area:
                best = el
                best_area = area
        return best

    def _advance_photo_album(self) -> bool:
        el = self._largest_profile_photo_element()
        if not el:
            return False
        try:
            box = el.bounding_box()
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            return False
        if not box:
            return False
        try:
            self.page.mouse.click(
                float(box.get("x") or 0) + float(box.get("width") or 0) * 0.88,
                float(box.get("y") or 0) + float(box.get("height") or 0) * 0.50,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            return False

    def _capture_visible_profile_photos(self, shots: list[bytes], seen: set[bytes]) -> tuple[int, int]:
        added = matched = 0
        for el in self._profile_photo_elements():
            matched += 1
            try:
                shot = el.screenshot()
            except Exception as exc:  # noqa: BLE001
                _raise_driver_closed_if_browser_closed(exc)
                continue
            if shot in seen:                 # exact-byte dup of an earlier frame -> drop
                continue
            seen.add(shot)
            shots.append(shot)
            added += 1
        return added, matched

    def _capture_photos(self) -> list[bytes]:
        # CRITICAL: el.screenshot() captures the PAGE pixels under the element, so
        # our own overlays (the centered "Processing" modal, the HUD) would be
        # baked into the shot and hide the face -> no_face. Hide them first.
        # Bumble preloads the whole album as <img>s, so the first frame usually has
        # every photo; we only click through the carousel if it didn't, and we cap
        # the total so a progress-bar animation can't yield dozens of near-dup frames.
        self._hide_oplove_overlays()
        shots: list[bytes] = []
        seen: set[bytes] = set()
        raw_matched = filtered_first = 0
        for i in range(self.photo_capture_steps):
            if i == 0:
                raw_matched = len(self._query_selector_all(self.selectors["photo"]))
            added, filtered = self._capture_visible_profile_photos(shots, seen)
            if i == 0:
                filtered_first = filtered
                if len(shots) >= _ALBUM_PRELOADED_MIN:   # got the whole album at once -> don't re-capture
                    break
            if len(shots) >= _MAX_PHOTOS_PER_PROFILE:    # hard cap (no profile has this many)
                break
            if i > 0 and added == 0:
                break
            if i >= self.photo_capture_steps - 1 or not self._advance_photo_album():
                break
            time.sleep(_PHOTO_ADVANCE_SETTLE_S)
        print(f"[bumble] photos: selector matched {raw_matched}, "
              f"{filtered_first} passed the profile-photo filter, "
              f"{len(shots)} distinct captured")
        self._maybe_dump_photos(shots)
        return shots

    def _hide_oplove_overlays(self) -> None:
        # Make our injected overlays invisible so they're never screenshotted on
        # top of the profile photos. The worker re-shows the HUD on its next
        # render_status, and the busy modal on its next render_busy.
        try:
            self.page.evaluate(
                "() => { for (const id of ['oplove-hud','oplove-busy']) {"
                " const e = document.getElementById(id); if (e) e.style.display = 'none'; } }"
            )
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)

    def _maybe_dump_photos(self, shots: list[bytes]) -> None:
        # Opt-in (OPLOVE_DEBUG_CAPTURE=1): save the exact images the embedder will
        # see, so you can eyeball whether they're distinct profile photos (not the
        # same picture repeated, and not UI chrome with no face).
        import os
        if not os.environ.get("OPLOVE_DEBUG_CAPTURE"):
            return
        try:
            from datetime import datetime
            d = Path("data/debug/captures") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            d.mkdir(parents=True, exist_ok=True)
            for idx, shot in enumerate(shots):
                (d / f"{idx:02d}.png").write_bytes(shot)
            print(f"[bumble] DEBUG: wrote {len(shots)} captured photo(s) to {d}")
        except Exception as exc:  # noqa: BLE001
            print(f"[bumble] DEBUG photo dump failed: {exc}")

    def _capture_prompts(self) -> list[tuple[str, str]]:
        # TODO(live): map Bumble's profile fields/prompts to (label, value) pairs.
        out: list[tuple[str, str]] = []
        for el in self._query_selector_all(self.selectors["prompt"]):
            try:
                txt = (el.inner_text() or "").strip()
            except Exception as exc:  # noqa: BLE001
                _raise_driver_closed_if_browser_closed(exc)
                continue
            if txt:
                out.append(("", txt))
        return out

    def _text(self, selector: str) -> str:
        el = self._query_selector(selector)
        if not el:
            return ""
        try:
            return el.inner_text().strip()
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise

    # --- actions --------------------------------------------------------
    def like(self, opener: str | None = None) -> None:
        # NORMAL like only — never the super-swipe. Super-likes/boosts are the
        # owner's manual call (see DEFAULT_SELECTORS["superlike"]).
        try:
            self.page.click(self.selectors["like"])
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise
        # NOTE: opener is not sent here on standard Bumble (post-match / women-first).
        # Hook point for Bumble "Opening Moves" once that flow is mapped live.

    def dislike(self) -> None:
        try:
            self.page.click(self.selectors["pass"])
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise

    def out_of_profiles(self) -> bool:
        return self._query_selector(self.selectors["empty"]) is not None

    def _query_selector(self, selector: str):
        try:
            return self.page.query_selector(selector)
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise

    def _query_selector_all(self, selector: str):
        try:
            return self.page.query_selector_all(selector)
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise
