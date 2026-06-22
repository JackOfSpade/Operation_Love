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

import random
import time
from pathlib import Path

from .base import DatingAppDriver, DriverClosed
from ..human import human_delay
from ..perception.capture import Profile

_OBSERVE_POLL_S = 0.15      # internal sampling cadence for your manual swipe (not app-facing)
_MIN_PROFILE_PHOTO_SIDE_PX = 160
_SIDEBAR_EXCLUSION_RATIO = 0.30
_DEFAULT_PHOTO_CAPTURE_STEPS = 8
_PHOTO_ADVANCE_SETTLE_S = 0.25
_ALBUM_PRELOADED_MIN = 3      # if the first frame yields >= this many photos, the album is fully loaded
_MAX_PHOTOS_PER_PROFILE = 12  # hard cap; real profiles top out ~6, so this bounds runaway re-capture
_PHOTO_READY_TIMEOUT_S = 8.0
_PHOTO_READY_POLL_S = 0.25
_PHOTO_READY_SETTLE_S = 0.75
_STARTUP_INTERSTITIAL_TIMEOUT_MS = 500

_STARTUP_INTERSTITIALS = (
    ("cookie accept", '[data-qa-role="cookie-accept"], button:has-text("Accept")'),
    ("cookie allow all", '[data-qa-role="cookie-allow-all"], button:has-text("Allow all")'),
    ("cookie reject", '[data-qa-role="cookie-reject"], button:has-text("Reject")'),
    ("not now", 'button:has-text("Not now")'),
    ("maybe later", 'button:has-text("Maybe later")'),
    ("close dialog", '[data-qa-role="modal-close"], button[aria-label="Close"]'),
    # TODO(live): add the exact selector here once a specific in-app modal is
    # identified by screenshot.
)


def _is_browser_closed_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "TargetClosedError":
        return True
    msg = str(exc).lower()
    return (
        "target page, context or browser has been closed" in msg
        or "browser has been closed" in msg
        or "page has been closed" in msg
        # Ctrl-C/SIGINT also kills patchright/playwright's node driver subprocess,
        # so a later close() sees the driver gone — treat as already-closed.
        or "connection closed" in msg
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
    # a manual super-swipe as a 'like' signal when learning taste. Matches EITHER the
    # action button's data-qa-role OR the icon's data-qa-icon-name (the latter
    # confirmed live 2026-06-21), so detection survives Bumble button-DOM tweaks.
    "superlike": '[data-qa-role="encounters-action-superswipe"], [data-qa-icon-name="floating-action-superswipe"]',
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

_PHOTO_LOADED_JS = """
async (node) => {
  const rect = node.getBoundingClientRect();
  const minW = Math.max(96, rect.width * 0.60);
  const minH = Math.max(96, rect.height * 0.60);
  const imgs = [];
  if (node instanceof HTMLImageElement) imgs.push(node);
  if (node.querySelectorAll) imgs.push(...node.querySelectorAll('img'));
  if (imgs.length) {
    return imgs.some((img) => img.complete
      && img.naturalWidth >= minW && img.naturalHeight >= minH);
  }
  const nodes = [node];
  if (node.querySelectorAll) nodes.push(...node.querySelectorAll('*'));
  const urls = [];
  for (const n of nodes) {
    const bg = window.getComputedStyle(n).backgroundImage;
    const m = bg && bg.match(/url\\(["']?([^"')]+)["']?\\)/);
    if (m) urls.push(m[1]);
  }
  if (!urls.length) return false;
  const checks = urls.map((url) => new Promise((resolve) => {
    const img = new Image();
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      resolve(!!ok && img.naturalWidth >= minW && img.naturalHeight >= minH);
    };
    img.onload = () => finish(true);
    img.onerror = () => finish(false);
    img.src = url;
    if (img.complete) finish(true);
    setTimeout(() => finish(false), 1500);
  }));
  return (await Promise.all(checks)).some(Boolean);
}
"""


class BumbleDriver(DatingAppDriver):
    accepts_opener = False          # Bumble: match first, then message — no swipe-time opener (don't spend Claude credits)

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
        # Drive the real Chrome binary by default (more authentic than bundled
        # Chromium); "" forces bundled. The in-page HUD is OFF by default so we
        # don't mutate Bumble's DOM with our own element (status lives in the hub).
        self.browser_channel = app_cfg.get("browser_channel", "chrome")
        self.inpage_overlays = bool(app_cfg.get("inpage_overlays", False))
        self._pw = None
        self._ctx = None
        self.page = None
        self._active_channel = None
        self._mouse_xy = None        # last known cursor position (for human-like moves)

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        sync_playwright, engine = self._import_playwright()

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        try:
            # Anti-automation hygiene (see Bumble risk review): hide the
            # AutomationControlled blink feature so navigator.webdriver isn't set,
            # and drop the --enable-automation switch. A patched engine (patchright
            # / rebrowser-playwright, preferred in _import_playwright) additionally
            # routes script evaluation through isolated worlds, avoiding the
            # Runtime.enable / consoleAPICalled CDP leak that flags vanilla Playwright.
            launch_kwargs = dict(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                viewport={"width": 1280, "height": 900},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--deny-permission-prompts",
                ],
                ignore_default_args=["--enable-automation"],
            )
            # Persistent context => your manual login is remembered across runs.
            # Prefer the real Google Chrome binary over bundled Chromium for
            # authentic WebGL/plugins/window.chrome; fall back if Chrome's absent.
            self._ctx = self._launch_context(launch_kwargs)
            print(f"Bumble browser engine={engine}, channel={self._active_channel}")
            self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            self.page.set_default_timeout(self.nav_timeout_ms)
            self.page.goto(self.url, wait_until="domcontentloaded")
            self._dismiss_startup_interstitials()
            # First run: if not logged in, sign in by hand in the opened window; the
            # user-data dir persists the session for subsequent runs.
        except BaseException:
            # A failed launch/goto must not orphan the Playwright driver subprocess:
            # the worker only calls close() inside its own try/finally, AFTER
            # open_session returns. close() is safe on a partially-started session.
            self.close()
            raise

    @staticmethod
    def _import_playwright():
        """Prefer a stealth-patched Playwright (patchright, then rebrowser-playwright)
        that avoids the Runtime.enable CDP leak; fall back to vanilla playwright.
        All expose the same sync_api surface."""
        from importlib import import_module
        for mod, name in (("patchright", "patchright"),
                          ("rebrowser_playwright", "rebrowser-playwright")):
            try:
                return import_module(f"{mod}.sync_api").sync_playwright, name
            except Exception:  # noqa: BLE001
                continue
        from playwright.sync_api import sync_playwright  # type: ignore  # lazy: [bumble] extra
        return sync_playwright, "playwright (unpatched — pip install patchright for stealth)"

    def _launch_context(self, launch_kwargs: dict):
        channel = (self.browser_channel or "").strip()
        if channel:
            try:
                ctx = self._pw.chromium.launch_persistent_context(channel=channel, **launch_kwargs)
                self._active_channel = channel
                return ctx
            except Exception as exc:  # noqa: BLE001
                print(f"Bumble channel='{channel}' unavailable ({type(exc).__name__}); "
                      f"falling back to bundled Chromium.")
        ctx = self._pw.chromium.launch_persistent_context(**launch_kwargs)
        self._active_channel = "chromium (bundled)"
        return ctx

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
        # Paint/refresh the in-page HUD. OFF by default on Bumble (inpage_overlays)
        # so we don't mutate Bumble's DOM with our own element — status shows in the
        # hub instead. Best-effort: a mid-eval navigation must not break the loop.
        if not self.page or not self.inpage_overlays:
            return
        try:
            self.page.evaluate(_OVERLAY_JS, status)
        except Exception:  # noqa: BLE001
            pass

    def render_busy(self, message: str | None = None) -> None:
        # Full-screen click-blocking modal so you can't swipe the next card while
        # this one is still embedding. OFF by default on Bumble (inpage_overlays)
        # to avoid injecting into Bumble's DOM. message shows it; None hides it.
        if not self.page or not self.inpage_overlays:
            return
        try:
            self.page.evaluate(_BUSY_JS, message)
        except Exception:  # noqa: BLE001
            pass

    def _dismiss_startup_interstitials(self) -> None:
        """Best-effort startup cleanup for non-native cookie/modals.

        Native Chrome permission prompts are handled by --deny-permission-prompts.
        This only tries short, guarded clicks against known/likely in-page banners.
        It must never block or break a run when Bumble changes markup.
        """
        if not self.page:
            return
        try:
            for label, selector in _STARTUP_INTERSTITIALS:
                try:
                    self.page.click(selector, timeout=_STARTUP_INTERSTITIAL_TIMEOUT_MS)
                    print(f"Bumble dismissed startup interstitial: {label}")
                    time.sleep(0.1)
                except Exception:  # noqa: BLE001
                    pass
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

    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None) -> bool | None:
        """Block until YOU manually like/pass the current card.

        Returns True (liked), False (passed), or None (deck emptied / timeout /
        stop requested). timeout=None waits indefinitely. Mouse clicks on the
        like/pass controls are detected; if you also use keyboard shortcuts,
        confirm coverage with bumble_inspect.py.
        """
        self._install_observe_listener()
        # Discard any decision recorded during the PREVIOUS card's capture/embed
        # window. With the in-page busy modal off (inpage_overlays default False),
        # you can physically swipe the next card while we're still embedding the
        # last one; that stray swipe sets window.__oplove_decision and, left
        # uncleared, would be mis-attributed to the card we're about to wait on ->
        # corrupted labels. Clearing here means only a swipe made AFTER this
        # card's capture can count for it.
        try:
            self.page.evaluate("() => { window.__oplove_decision = null; }")
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
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

    def _profile_photo_element_loaded(self, el) -> bool:
        try:
            return bool(el.evaluate(_PHOTO_LOADED_JS))
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            # Older/fake element handles may not support evaluate; fail open so
            # capture still works instead of spinning until timeout.
            return True

    def _profile_photo_counts(self, *, require_loaded: bool = False) -> tuple[int, int, int]:
        raw = self._query_selector_all(self.selectors["photo"])
        filtered = loaded = 0
        for el in raw:
            if not self._is_profile_photo_element(el):
                continue
            filtered += 1
            if not require_loaded or self._profile_photo_element_loaded(el):
                loaded += 1
        return len(raw), filtered, loaded

    def _wait_for_profile_album_ready(self) -> tuple[int, int]:
        """Wait until Bumble's album elements have appeared and had time to paint."""
        deadline = time.monotonic() + _PHOTO_READY_TIMEOUT_S
        stable_counts = None
        stable_since = None
        last_raw = last_filtered = 0
        while True:
            raw, filtered, loaded = self._profile_photo_counts(require_loaded=True)
            last_raw, last_filtered = raw, filtered
            now = time.monotonic()
            ready = filtered > 0 and loaded >= filtered
            counts = (filtered, loaded)
            if ready:
                if counts != stable_counts:
                    stable_counts = counts
                    stable_since = now
                elif stable_since is not None and now - stable_since >= _PHOTO_READY_SETTLE_S:
                    return raw, filtered
                if _PHOTO_READY_SETTLE_S <= 0:
                    return raw, filtered
            else:
                stable_counts = None
                stable_since = None
            if now >= deadline:
                return last_raw, last_filtered
            time.sleep(_PHOTO_READY_POLL_S)

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
            self._human_mouse_click(
                float(box.get("x") or 0) + float(box.get("width") or 0) * 0.88,
                float(box.get("y") or 0) + float(box.get("height") or 0) * 0.50,
            )
            return True
        except DriverClosed:
            raise
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
        # baked into the shot and hide the face -> no_face. Keep the blocking
        # overlay visible while waiting for photos to load, then hide overlays
        # only immediately before screenshots.
        # Bumble preloads the whole album as <img>s, so the first frame usually has
        # every photo; we only click through the carousel if it didn't, and we cap
        # the total so a progress-bar animation can't yield dozens of near-dup frames.
        raw_matched, filtered_first = self._wait_for_profile_album_ready()
        self._hide_oplove_overlays()
        shots: list[bytes] = []
        seen: set[bytes] = set()
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
        print(f"Photos: selector matched {raw_matched}, "
              f"{filtered_first} passed the profile-photo filter, "
              f"{len(shots)} distinct captured")
        self._maybe_dump_photos(shots)
        return shots

    def _hide_oplove_overlays(self) -> None:
        # Make our injected overlays invisible so they're never screenshotted on
        # top of the profile photos. No-op when in-page overlays are disabled
        # (nothing was injected). The worker re-shows them on its next render.
        if not self.inpage_overlays:
            return
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
            print(f"Debug: wrote {len(shots)} captured photo(s) to {d}")
        except Exception as exc:  # noqa: BLE001
            print(f"Debug photo dump failed: {exc}")

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

    # --- human-like pointer (AUTO mode only; observe mode = your real cursor) ---
    def _human_target(self, box: dict) -> tuple[float, float]:
        """A jittered point inside the element, not dead-center — humans don't click
        the exact middle every time."""
        x, y = float(box.get("x") or 0), float(box.get("y") or 0)
        w, h = float(box.get("width") or 0), float(box.get("height") or 0)
        return (x + w * (0.5 + random.uniform(-0.22, 0.22)),
                y + h * (0.5 + random.uniform(-0.22, 0.22)))

    def _human_move(self, x: float, y: float) -> None:
        """Move the cursor to (x, y) along a curved (quadratic-Bezier) path with a
        slight overshoot, instead of teleporting — defeats zero-transition-time
        pointer detection. No-op on fakes/engines without a real mouse.move."""
        mouse = getattr(self.page, "mouse", None)
        move = getattr(mouse, "move", None)
        if not callable(move):
            return
        vp = getattr(self.page, "viewport_size", None) or {"width": 1280, "height": 900}
        sx, sy = self._mouse_xy or (float(vp.get("width", 1280)) * 0.5,
                                    float(vp.get("height", 900)) * 0.85)
        cx = (sx + x) / 2 + random.uniform(-60, 60)     # random control point...
        cy = (sy + y) / 2 + random.uniform(-60, 60)
        ox, oy = x + random.uniform(-4, 4), y + random.uniform(-4, 4)   # ...and overshoot
        steps = random.randint(14, 26)
        try:
            for i in range(1, steps + 1):
                t = i / steps
                mt = 1 - t
                px = mt * mt * sx + 2 * mt * t * cx + t * t * ox
                py = mt * mt * sy + 2 * mt * t * cy + t * t * oy
                move(px, py)
                time.sleep(human_delay(0.012))
            move(x, y)                                  # settle on the (jittered) target
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise
        self._mouse_xy = (x, y)

    def _human_mouse_click(self, x: float, y: float) -> None:
        """Human-ish move to (x, y), a brief hesitation, then click there."""
        self._human_move(x, y)
        time.sleep(human_delay(0.09))
        try:
            self.page.mouse.click(x, y)
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise
        self._mouse_xy = (x, y)

    def _box_center_in_viewport(self, box: dict) -> bool:
        vp = getattr(self.page, "viewport_size", None)
        if not vp:
            return True                          # unknown viewport -> don't block the human path
        cx = float(box.get("x") or 0) + float(box.get("width") or 0) / 2
        cy = float(box.get("y") or 0) + float(box.get("height") or 0) / 2
        return 0 <= cx <= float(vp.get("width", 0)) and 0 <= cy <= float(vp.get("height", 0))

    def _human_click(self, selector: str) -> None:
        """Click an element with a human-like cursor path. Falls back to a plain
        Playwright click — which keeps Playwright's actionability checks + auto
        scroll-into-view — when the element box, a real mouse API, or an on-screen
        target isn't available (e.g. in tests, or a control that's off-screen or
        covered). Preserves the browser-closed -> DriverClosed contract on every path."""
        el = self._query_selector(selector)
        box = None
        if el is not None:
            try:
                scroll = getattr(el, "scroll_into_view_if_needed", None)
                if callable(scroll):
                    scroll()                     # bring the control into view first
                box = el.bounding_box()
            except Exception as exc:  # noqa: BLE001
                _raise_driver_closed_if_browser_closed(exc)
                raise
        mouse = getattr(self.page, "mouse", None)
        if (box and self._box_center_in_viewport(box)
                and callable(getattr(mouse, "move", None))
                and callable(getattr(mouse, "click", None))):
            x, y = self._human_target(box)
            self._human_mouse_click(x, y)
            return
        # No box / off-screen / no mouse API -> raw-coordinate clicking isn't safe;
        # page.click does its own actionability + scroll-into-view.
        try:
            self.page.click(selector)
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise

    # --- actions --------------------------------------------------------
    def like(self, opener: str | None = None) -> None:
        # NORMAL like only — never the super-swipe. Super-likes/boosts are the
        # owner's manual call (see DEFAULT_SELECTORS["superlike"]).
        self._human_click(self.selectors["like"])
        # NOTE: opener is not sent here on standard Bumble (post-match / women-first).
        # Hook point for Bumble "Opening Moves" once that flow is mapped live.

    def dislike(self) -> None:
        self._human_click(self.selectors["pass"])

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
