"""Generic Playwright/patchright driving base for browser-based dating platforms.

Everything here is site-agnostic:

  * browser lifecycle (_import_playwright, _launch_context, open_session/close)
    and the anti-automation launch hygiene that goes with it
  * the _browser_guard() / DriverClosed translation for a closed browser/page
  * human-like pointer movement (_human_target/_human_move/_human_mouse_click/
    _human_click) -- Bezier path + overshoot instead of teleporting the cursor
  * the injected-JS status HUD (_OVERLAY_JS/_BUSY_JS) and render_status/render_busy
    -- inpage_overlays defaults OFF (a deliberate anti-detection decision: don't
    mutate the target site's DOM unless a subclass' config opts in)
  * the card-identity anti-phantom-swipe pattern (_card_fingerprint /
    _verify_swipe_landed) -- confirms a like/pass click actually changed the
    on-screen profile before letting the caller record it as a real decision
  * debug-log wiring (_dbg_action) and snapshot_failure

No CSS selector, site URL, or capture logic lives here. Those belong to a concrete
subclass via the `selectors` dict and the overridable hooks.
"""
from __future__ import annotations

import random
import time
from contextlib import contextmanager
from pathlib import Path

from ... import platforms
from ...private_files import ensure_private_dir
from ..base import DatingAppDriver, DriverClosed, open_debug_log, snapshot_failure_frame
from ...human import human_delay

# Anti-phantom-swipe re-check knobs (see _verify_swipe_landed): a like/pass click's
# effect can lag a beat behind an SPA re-render, so a couple of quick re-checks ride
# that out before concluding the click was a no-op.
_SWIPE_VERIFY_TRIES = 2       # re-checks before concluding a like/pass never landed
_SWIPE_VERIFY_SETTLE_S = 0.5  # gap between re-checks (SPA re-render lag)


class PlatformUnavailable(RuntimeError):
    """Raised when a web driver is asked to actually start for a platform the
    registry (operation_love/platforms.py) marks unavailable.

    Belt and braces: the supervisor already checks platforms.unavailable_reason()
    before ever constructing a driver, so this should never fire on the normal run
    path. It exists for whoever builds a driver directly -- a script, a REPL, a
    future caller that forgets -- so an unrunnable driver says so itself rather
    than silently trying (and failing in some more confusing way, or worse,
    "succeeding" at driving nothing)."""


class ActionNotLandedError(RuntimeError):
    """A like/pass click did not actually produce a change on screen (covered hit
    target, mid-animation card, stale element, a modal that swallowed the click).

    NOT a DriverClosed (which is a clean, restart-safe stop): this is unexpected, so
    the caller should halt the run rather than record a phantom decision. See
    _verify_swipe_landed below and worker.py's "do not record a phantom" guard,
    which relies on like()/dislike() raising instead of returning normally when a
    swipe silently failed to land."""


class HumanInputUnavailable(RuntimeError):
    """The humanized input path could not be used, so the action was refused.

    The alternative was `page.click()`, which dispatches at the element with no cursor
    travel. ops/ANTI-BOT-RESEARCH.md §1 lists cursor path ("Bezier vs zero-time teleport")
    as a HIGH-confidence behavioural detection vector, so quietly substituting it would
    trade an invisible, open-ended increase in detectability for one extra swipe.

    Like ActionNotLandedError this is NOT a DriverClosed: it is unexpected, so the run
    halts and the debug logs survive rather than the driver carrying on degraded."""


def _is_browser_closed_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "TargetClosedError":
        return True
    msg = str(exc).lower()
    return (
        "target page, context or browser has been closed" in msg
        or "browser has been closed" in msg
        or "page has been closed" in msg
        # Ctrl-C/SIGINT also kills patchright/playwright's node driver subprocess,
        # so a later close() sees the driver gone -- treat as already-closed.
        or "connection closed" in msg
    )


def _raise_driver_closed_if_browser_closed(exc: Exception) -> None:
    if _is_browser_closed_error(exc):
        raise DriverClosed("Browser was closed") from exc


@contextmanager
def _browser_guard():
    """Run a Playwright call, translating a closed-browser error into DriverClosed;
    any other exception re-raises unchanged. Collapses the except/translate/raise
    boilerplate that would otherwise be hand-rolled at every Playwright call site
    that must propagate its failure (call sites that instead degrade to a fallback
    value keep their own try/except, since the fallback differs per site)."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        _raise_driver_closed_if_browser_closed(exc)
        raise


# In-page status HUD: a fixed, click-through (pointer-events:none) overlay updated
# after every swipe so you watch progress on the page, not the terminal. Takes an
# app_view() snapshot plus the driver's `hud_label` (see render_status). Builds the
# element once, then just refreshes its contents.
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
    '<div style="opacity:.7">' + (s.hud_label || '') + ' · ' + (a.state || '—') + '</div>' +
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

# Per-card identity for _card_fingerprint: one round trip (not one per element) that
# reads back each matched photo node's actual image source -- <img> src/currentSrc/
# srcset, or the CSS background-image url() for a div-based card -- so two different
# profiles are told apart even when their bio and photo COUNT happen to coincide
# (common on Bumble: bio is optional/often blank, and photo counts cluster at the
# app's max). Relies only on `sel` (the subclass's
# selectors["photo"]), so it carries no site-specific knowledge itself.
_CARD_PHOTO_IDS_JS = """
(sel) => Array.from(document.querySelectorAll(sel)).map((el) => {
  const img = el.tagName === 'IMG' ? el : (el.querySelector ? el.querySelector('img') : null);
  if (img) {
    return img.currentSrc || img.getAttribute('src') || img.getAttribute('srcset') || '';
  }
  const bg = window.getComputedStyle(el).backgroundImage;
  return (bg && bg !== 'none') ? bg : '';
}).filter(Boolean);
"""


class PlaywrightDriver(DatingAppDriver):
    """Reusable Playwright/patchright base. Not directly instantiable in practice
    (like(), dislike(), out_of_profiles(), next_profile() stay abstract -- a
    concrete site subclass provides them); construct a subclass instead."""

    # Registry app id (operation_love/platforms.py) this driver answers to for the
    # open_session() availability guard. None disables the check; every real
    # subclass should set this to its own registry id.
    platform_app: str | None = None

    # Text shown in the in-page HUD's second line ("<hud_label> · <state>") and in a
    # couple of log lines. Override per platform (kept out of _OVERLAY_JS itself so
    # the JS stays site-agnostic).
    hud_label: str = "web"

    def __init__(
        self,
        cfg,
        *,
        config_key: str,
        default_url: str,
        default_user_data_dir: str,
        default_debug_dir: str,
    ) -> None:
        app_cfg = (getattr(cfg, "apps", {}) or {}).get(config_key, {})
        self.app_cfg = app_cfg          # subclass reads its own extra keys from this
        self.url = app_cfg.get("url", default_url)
        self.headless = bool(app_cfg.get("headless", False))
        self.user_data_dir = Path(app_cfg.get("user_data_dir", default_user_data_dir))
        self.nav_timeout_ms = int(app_cfg.get("nav_timeout_ms", 15000))
        # Drive the real Chrome binary by default (more authentic than bundled
        # Chromium); "" forces bundled.
        self.browser_channel = app_cfg.get("browser_channel", "chrome")
        # The in-page HUD is OFF by default so a run doesn't mutate the target
        # site's DOM with our own element (status lives in the hub instead). This
        # is a deliberate anti-detection decision -- keep the default.
        self.inpage_overlays = bool(app_cfg.get("inpage_overlays", False))
        # Halt the run on an unexpected error rather than restarting the session and
        # carrying on. Read here so apps.<app>.halt_on_error works uniformly across web and
        # Android, and DEFAULTS TO TRUE like the ABC — this driver previously declared the
        # attribute nowhere at all, so worker.py's `getattr(..., False)` handed it
        # restart-with-backoff by omission. Config validation refuses `false` in auto mode.
        self.halt_on_error = bool(app_cfg.get("halt_on_error", True))
        # Silent debug log: a text action trail + a page screenshot ONLY on failure
        # (the browser is watchable, so per-action shots aren't worth the overhead).
        self.debug_log = bool(app_cfg.get("debug_log", False))
        self.debug_dir = app_cfg.get("debug_dir", default_debug_dir)
        self._dbg = None             # DebugLog (set in open_session when debug_log is on)
        self._pw = None
        self._ctx = None
        self.page = None
        self._active_channel = None
        self._mouse_xy = None        # last known cursor position (for human-like moves)
        # Subclass fills this in (its own DEFAULT_SELECTORS merged with config
        # overrides) before any selector-driven method is called.
        self.selectors: dict = {}

    # --- availability guard ----------------------------------------------
    def _check_platform_unavailable(self) -> None:
        """Raise PlatformUnavailable if the registry says this platform can't run.

        Checked in open_session(), not __init__: constructing a driver to
        introspect it (its .selectors, its class hierarchy -- exactly what this
        package's tests do) must keep working even for an unavailable platform;
        only actually starting a browser session is refused.
        """
        if self.platform_app is None:
            return
        reason = platforms.unavailable_reason(self.platform_app)
        if reason:
            raise PlatformUnavailable(reason)

    # --- lifecycle ------------------------------------------------------
    def open_session(self) -> None:
        # Guard first, before touching Playwright at all: an unavailable platform
        # must never launch so much as the driver subprocess.
        self._check_platform_unavailable()

        sync_playwright, engine = self._import_playwright()

        # A persistent context carries authenticated account cookies/session storage. Tighten
        # only this configured leaf before Playwright can populate it; never chmod its parent.
        ensure_private_dir(self.user_data_dir)
        self._pw = sync_playwright().start()
        try:
            # Anti-automation hygiene: hide the AutomationControlled blink feature
            # so navigator.webdriver isn't set, and drop the --enable-automation
            # switch. A patched engine (patchright / rebrowser-playwright,
            # preferred in _import_playwright) additionally routes script
            # evaluation through isolated worlds, avoiding the Runtime.enable /
            # consoleAPICalled CDP leak that flags vanilla Playwright.
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
            # Persistent context => a manual login is remembered across runs.
            # Prefer the real Google Chrome binary over bundled Chromium for
            # authentic WebGL/plugins/window.chrome; fall back if Chrome's absent.
            self._ctx = self._launch_context(launch_kwargs)
            print(f"{self.hud_label} browser engine={engine}, channel={self._active_channel}")
            self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            self.page.set_default_timeout(self.nav_timeout_ms)
            self.page.goto(self.url, wait_until="domcontentloaded")
            self._after_goto()
            if self.debug_log:
                self._dbg = open_debug_log(self.debug_dir)
            # First run: if not logged in, sign in by hand in the opened window; the
            # user-data dir persists the session for subsequent runs.
        except BaseException:
            # A failed launch/goto must not orphan the Playwright driver subprocess:
            # the worker only calls close() inside its own try/finally, AFTER
            # open_session returns. close() is safe on a partially-started session.
            self.close()
            raise

    def _after_goto(self) -> None:
        """Hook for site-specific post-navigation cleanup (e.g. cookie banners /
        startup interstitials). No-op by default."""

    @staticmethod
    def _import_playwright():
        """Prefer a stealth-patched Playwright (patchright) that avoids the Runtime.enable
        CDP leak; fall back to vanilla playwright. Both expose the same sync_api surface.
        (rebrowser-playwright was considered too, but isn't in the reference-only `web` extra --
        pyproject only installs patchright/playwright -- so it's not offered here as a
        silent, untested fallback tier.)"""
        try:
            from patchright.sync_api import sync_playwright
            return sync_playwright, "patchright"
        except Exception:  # noqa: BLE001
            pass
        from playwright.sync_api import sync_playwright  # type: ignore  # lazy: [web] extra
        return sync_playwright, "playwright (unpatched — pip install patchright for stealth)"

    def _launch_context(self, launch_kwargs: dict):
        channel = (self.browser_channel or "").strip()
        if channel:
            try:
                ctx = self._pw.chromium.launch_persistent_context(channel=channel, **launch_kwargs)
                self._active_channel = channel
                return ctx
            except Exception as exc:  # noqa: BLE001
                print(f"{self.hud_label} channel='{channel}' unavailable ({type(exc).__name__}); "
                      f"falling back to bundled Chromium.")
        ctx = self._pw.chromium.launch_persistent_context(**launch_kwargs)
        self._active_channel = "chromium (bundled)"
        return ctx

    def close(self) -> None:
        # Total + idempotent: every step below must run, and the refs must be cleared,
        # even if an earlier step raises — otherwise a raising _pw.stop() would abandon
        # cleanup with _pw/_ctx/page still set, orphaning the browser process while the
        # driver looks alive. A genuine (non-"already closed") failure is still
        # surfaced, just AFTER all cleanup has been attempted, not instead of it.
        first_exc = None
        if self._ctx:
            try:
                self._ctx.close()
            except Exception as exc:  # noqa: BLE001
                if not _is_browser_closed_error(exc):
                    first_exc = exc
        if self._pw:
            try:
                self._pw.stop()
            except Exception as exc:  # noqa: BLE001
                if not _is_browser_closed_error(exc) and first_exc is None:
                    first_exc = exc
        self._pw = self._ctx = self.page = None
        if first_exc is not None:
            raise first_exc

    # --- injected-JS status overlay --------------------------------------
    def render_status(self, status: dict) -> None:
        # Paint/refresh the in-page HUD. OFF by default (inpage_overlays) so a run
        # doesn't mutate the target site's DOM — status shows in the hub instead.
        # Best-effort: a mid-eval navigation must not break the loop.
        if not self.page or not self.inpage_overlays:
            return
        try:
            self.page.evaluate(_OVERLAY_JS, {**status, "hud_label": self.hud_label})
        except Exception:  # noqa: BLE001
            pass

    def render_busy(self, message: str | None = None) -> None:
        # Full-screen click-blocking modal so you can't swipe the next card while
        # this one is still embedding. OFF by default (inpage_overlays) to avoid
        # injecting into the target site's DOM. message shows it; None hides it.
        if not self.page or not self.inpage_overlays:
            return
        try:
            self.page.evaluate(_BUSY_JS, message)
        except Exception:  # noqa: BLE001
            pass

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
        with _browser_guard():
            for i in range(1, steps + 1):
                t = i / steps
                mt = 1 - t
                px = mt * mt * sx + 2 * mt * t * cx + t * t * ox
                py = mt * mt * sy + 2 * mt * t * cy + t * t * oy
                move(px, py)
                time.sleep(human_delay(0.012))
            move(x, y)                                  # settle on the (jittered) target
        self._mouse_xy = (x, y)

    def _human_mouse_click(self, x: float, y: float) -> None:
        """Human-ish move to (x, y), a brief hesitation, then click there."""
        self._human_move(x, y)
        time.sleep(human_delay(0.09))
        with _browser_guard():
            self.page.mouse.click(x, y)
        self._mouse_xy = (x, y)

    def _box_center_in_viewport(self, box: dict) -> bool:
        vp = getattr(self.page, "viewport_size", None)
        if not vp:
            return True                          # unknown viewport -> don't block the human path
        cx = float(box.get("x") or 0) + float(box.get("width") or 0) / 2
        cy = float(box.get("y") or 0) + float(box.get("height") or 0) / 2
        return 0 <= cx <= float(vp.get("width", 0)) and 0 <= cy <= float(vp.get("height", 0))

    def _human_click(self, selector: str, *, timeout_ms: float | None = None) -> None:
        """Click an element by moving a real cursor to it along a human path.

        `timeout_ms` bounds the actionability wait (scroll-into-view). Leave it None for
        real like/pass controls, which SHOULD wait the page default. Pass a short bound for
        speculative clicks on elements that may legitimately be absent or inert — see
        _dismiss_startup_interstitials, which would otherwise stall startup for up to the
        page default PER candidate selector.

        RAISES rather than falling back to `page.click()`. That fallback used to fire
        whenever the element had no box, sat off-screen, or the page exposed no mouse API —
        but `page.click()` dispatches at the element with NO cursor travel at all, and a
        zero-time teleport straight to a control is one of the better-documented
        automation tells (ops/ANTI-BOT-RESEARCH.md §1 lists cursor path, "Bezier vs
        zero-time teleport", as a HIGH-confidence behavioural vector). Substituting it for
        the human path is exactly the silent downgrade this driver exists to avoid.

        Reaching this state means the page is not where we think it is: scroll-into-view
        has already run by the time the box is measured, so a still-off-screen control is a
        signal to stop and look, not to click harder. Preserves the browser-closed ->
        DriverClosed contract on every path."""
        el = self._query_selector(selector)
        box = None
        if el is not None:
            with _browser_guard():
                scroll = getattr(el, "scroll_into_view_if_needed", None)
                if callable(scroll):
                    # Playwright waits for actionability here and, with no explicit timeout,
                    # falls back to the PAGE default (set to nav_timeout_ms, 15s by default).
                    # An element that is present but never becomes actionable therefore
                    # blocks for the full default unless a caller bounds it.
                    scroll(timeout=timeout_ms) if timeout_ms is not None else scroll()
                box = el.bounding_box()
        mouse = getattr(self.page, "mouse", None)
        if (box and self._box_center_in_viewport(box)
                and callable(getattr(mouse, "move", None))
                and callable(getattr(mouse, "click", None))):
            x, y = self._human_target(box)
            self._human_mouse_click(x, y)
            return
        if not box:
            why = "the element has no bounding box (not rendered, or detached)"
        elif not self._box_center_in_viewport(box):
            why = (f"the element's centre is outside the viewport even after "
                   f"scroll-into-view (box={box})")
        else:
            why = "the page exposes no usable mouse API"
        raise HumanInputUnavailable(
            f"refusing to click {selector!r}: {why}. Not falling back to page.click() — "
            f"that dispatches with no cursor travel, and a zero-time teleport is a "
            f"documented automation tell."
        )

    # --- generic Playwright query helpers ---------------------------------
    def _query_selector(self, selector: str):
        with _browser_guard():
            return self.page.query_selector(selector)

    def _query_selector_all(self, selector: str):
        with _browser_guard():
            return self.page.query_selector_all(selector)

    def _text(self, selector: str) -> str:
        el = self._query_selector(selector)
        if not el:
            return ""
        with _browser_guard():
            return el.inner_text().strip()

    # --- debug log (silent) ----------------------------------------------
    def _dbg_action(self, name: str, **fields) -> None:
        """Append a TEXT-only record to the debug log (no per-action screenshot — the browser is
        watchable; a shot is taken only on failure). Best-effort; never raises."""
        if self._dbg is None:
            return
        try:
            self._dbg.action(name, **fields)
        except Exception:  # noqa: BLE001 — logging must never break a swipe
            pass

    def snapshot_failure(self, exc: BaseException) -> None:
        """On an unexpected error (the caller invokes this WHILE the browser is still live,
        before close()), save a viewport screenshot + the error to the debug log so the bug
        report can point at what the bot was looking at. Best-effort; never raises (it runs
        inside the caller's except — it must not mask the real error)."""
        if self._dbg is None or not self.page:
            return
        snapshot_failure_frame(self._dbg, exc, self.page.screenshot)

    # --- card-identity anti-phantom-swipe check ---------------------------
    # _human_click's box-in-viewport test (_box_center_in_viewport) is a PURE
    # coordinate check — it proves nothing about whether the click actually hit a
    # live, unobstructed target. A no-op click (covered by a modal, or on a card
    # mid-animation) would otherwise be recorded as a real swipe, corrupting the
    # daily rate-limit count and stats. So like()/dislike() snapshot a cheap "which
    # profile is on screen" fingerprint before clicking and confirm it changed.
    #
    # The fingerprint is (out_of_profiles, bio, photo identities) rather than just
    # (out_of_profiles, bio, photo COUNT): a bio is often optional/blank, and photo
    # counts cluster tightly (many profiles show exactly the site's max), so two
    # distinct consecutive profiles sharing e.g. (False, "", 6) is entirely
    # plausible — that collision would read as "the card didn't change" and raise
    # ActionNotLandedError on a swipe that genuinely landed (this is BUMBLE-7's
    # regression, fixed by adding _CARD_PHOTO_IDS_JS's per-photo image identity).
    # Bio+count are still useful (free — already backed
    # by the selectors dict) but are no longer load-bearing alone.
    def _card_fingerprint(self) -> tuple | None:
        """Best-effort snapshot of the on-screen profile (bio text, empty-deck state, and the
        set of on-screen photo identities). Returns None when it can't be read reliably (thin
        test double, or a DOM shape that hasn't been live-verified) — callers must fail OPEN in
        that case, same as _box_center_in_viewport's "unknown viewport -> don't block the human
        path". Fail-open is deliberate here, not just convenient: on a live account, a false
        "didn't land" (this fingerprint wrongly claims no change) HALTS the entire auto run AND
        drops the swipe from the daily rate-limit / like-ratio counters (the caller's "don't
        record a phantom" guard skips logging when like()/dislike() raises) even though the
        click already reached the site's servers — so those safety counters go stale-low, the
        opposite of what they're for. A false "landed" (missing a genuine no-op click) only
        records one phantom decision, which makes the counters slightly OVER-conservative, not
        under. Given that asymmetry, every failure mode here (unreadable DOM, a stale handle, a
        closed page) resolves to "assume it landed" (None), never to a raised error.
        """
        try:
            empty = self.out_of_profiles()
            bio = self._text(self.selectors["bio"])
            photo_ids = self._card_photo_ids()
        except DriverClosed:
            raise
        except Exception:  # noqa: BLE001 — no reliable signal -> caller fails open
            return None
        if not isinstance(photo_ids, list):
            photo_ids = []
        return (empty, bio, tuple(sorted(str(p) for p in photo_ids)))

    def _card_photo_ids(self) -> list:
        try:
            return self.page.evaluate(_CARD_PHOTO_IDS_JS, self.selectors["photo"])
        except Exception as exc:  # noqa: BLE001
            _raise_driver_closed_if_browser_closed(exc)
            raise

    def _verify_swipe_landed(self, before: tuple | None, action: str) -> None:
        """Raise if the on-screen profile is provably UNCHANGED after a like/pass click, so the
        caller's phantom-decision guard engages instead of recording a swipe that never
        happened. Re-checks a couple of times to ride out SPA re-render lag before concluding
        the click was a no-op."""
        if before is None:
            return
        for attempt in range(_SWIPE_VERIFY_TRIES):
            after = self._card_fingerprint()
            if after is None or after != before:
                return
            if attempt < _SWIPE_VERIFY_TRIES - 1:
                time.sleep(_SWIPE_VERIFY_SETTLE_S)
        raise ActionNotLandedError(f"{action} did not land — the on-screen profile did not change")
