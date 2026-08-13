"""Bumble web driver — Playwright (sync API), element-based.

⚠️ NO LIVE TARGET. Bumble discontinued its web app in August 2026
(https://support.bumble.com/hc/en-us/articles/30996192802973-An-update-on-Bumble-web).
operation_love/platforms.py registers "bumble_web" with available=False, and
PlaywrightDriver.open_session() (this class's base) refuses to start for it. This
file is kept anyway: it is the Bumble-shaped remainder of the old
operation_love/drivers/bumble.py after the reusable, site-agnostic parts moved to
playwright_base.py (see this package's __init__.py). If Bumble web ever comes back,
or another web-based platform shows up, this is the reference for how the site
knowledge (selectors, photo-album capture, the observe-mode click listener) plugs
into PlaywrightDriver.

The selectors below were last live-verified on Bumble web in 2026-06, before the
shutdown; they are not expected to still be accurate if this ever runs again and
would need re-confirming with tools/bumble_inspect.py.

NOTE on openers: on Bumble (hetero mode) you can't send a message at swipe time
— matches message post-match (women-first), or via Bumble's profile-level
"Opening Moves". So `like(opener)` records/uses the opener where the flow allows
and otherwise ignores it. The per-swipe opener flow is Hinge's model; this is
wired through for parity and future Bumble "Opening Moves" support.
"""
from __future__ import annotations

import time

# _CARD_PHOTO_IDS_JS/_BUSY_JS/_OVERLAY_JS/BumbleActionError aren't used directly in
# this file's own code (the methods that use them live on the PlaywrightDriver base)
# -- they're re-imported here so `operation_love.drivers.bumble` (the compat shim,
# see that module) and this module's own namespace can re-export them under their
# original names for tests/tools written against the old drivers/bumble.py layout.
from .playwright_base import (
    ActionNotLandedError as BumbleActionError,  # noqa: F401
    PlaywrightDriver,
    _browser_guard,
    _CARD_PHOTO_IDS_JS,  # noqa: F401
    _BUSY_JS,  # noqa: F401
    _OVERLAY_JS,  # noqa: F401
    _raise_driver_closed_if_browser_closed,
)
from ..base import DriverClosed
from ...perception.capture import Profile

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
)


DEFAULT_SELECTORS = {
    "photo": '.encounters-album__story-content, [data-qa-role="encounters-story-photo"]',
    "bio": '[data-qa-role="encounters-story-about"], .encounters-story-about',
    "prompt": '.encounters-story-section--profile .pill, .encounters-story-about__field',
    "like": '[data-qa-role="encounters-action-like"]',
    "pass": '[data-qa-role="encounters-action-dislike"]',
    # superlike is OBSERVE-ONLY: the bot never clicks it (super-likes/boosts are
    # the owner's manual decision — see operation_love's never-superlike rule).
    # It exists here solely so observe mode can read a manual super-swipe as a
    # 'like' signal when learning taste. Matches EITHER the action button's
    # data-qa-role OR the icon's data-qa-icon-name (the latter confirmed live
    # 2026-06-21), so detection survives Bumble button-DOM tweaks.
    "superlike": '[data-qa-role="encounters-action-superswipe"], [data-qa-icon-name="floating-action-superswipe"]',
    "empty": '[data-qa-role="encounters-out-of-people"], .encounters-out-of-people',
}

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


class BumbleWebDriver(PlaywrightDriver):
    accepts_opener = False          # Bumble: match first, then message — no swipe-time opener (don't spend Gemini quota/spend)
    platform_app = "bumble_web"     # registry id checked by PlaywrightDriver.open_session()
    hud_label = "bumble"

    def __init__(self, cfg):
        super().__init__(
            cfg,
            # `apps.bumble` is the ANDROID Bumble block now that Bumble web shut down;
            # the web settings (headless, browser_channel, selectors, ...) moved to
            # `apps.bumble_web`, matching this driver's registry id. Reading "bumble"
            # here would silently hand the web driver the phone driver's config.
            config_key="bumble_web",
            default_url="https://bumble.com/app",
            default_user_data_dir="./data/bumble_profile",
            default_debug_dir="./data/bumble_debug",
        )
        self.selectors = {**DEFAULT_SELECTORS, **self.app_cfg.get("selectors", {})}
        self.photo_capture_steps = max(
            1, int(self.app_cfg.get("photo_capture_steps", _DEFAULT_PHOTO_CAPTURE_STEPS))
        )

    # --- lifecycle hook ----------------------------------------------------
    def _after_goto(self) -> None:
        self._dismiss_startup_interstitials()

    def _dismiss_startup_interstitials(self) -> None:
        """Best-effort startup cleanup for non-native cookie/consent banners.

        Native Chrome permission prompts are handled by --deny-permission-prompts.

        Clicks go through the HUMAN cursor path like every other click in this driver.
        They used to use page.click(), which dispatches with no cursor travel — and while
        a cookie banner is not a like or a paid control, a session whose very first
        interactions are zero-time teleports and whose later ones are Bezier paths is
        arguably more distinctive than one that is consistently either. Consistency is
        free here, so take it.

        A banner we cannot click humanly is simply left alone: an undismissed cookie
        banner is a cosmetic problem, and this is startup cleanup for markup that may not
        even be present. That is why HumanInputUnavailable is swallowed HERE and nowhere
        else — the alternative to clicking is skipping, not clicking worse.
        """
        if not self.page:
            return
        try:
            for label, selector in _STARTUP_INTERSTITIALS:
                try:
                    if self._query_selector(selector) is None:
                        continue
                    # Bounded: these selectors are speculative, and an element that is
                    # PRESENT but never becomes actionable would otherwise stall on the page
                    # default timeout (nav_timeout_ms, 15s) per selector. The original
                    # page.click() passed this same bound; routing through _human_click lost
                    # it until the timeout was threaded through.
                    self._human_click(selector, timeout_ms=_STARTUP_INTERSTITIAL_TIMEOUT_MS)
                    print(f"Bumble dismissed startup interstitial: {label}")
                    time.sleep(0.1)
                except DriverClosed:
                    raise
                except Exception:  # noqa: BLE001 — absent/unclickable banner: skip, never degrade
                    pass
        except DriverClosed:
            raise
        except Exception:  # noqa: BLE001
            pass

    # --- capture --------------------------------------------------------
    # NOT stop-aware, deliberately: this driver leaves DatingAppDriver's
    # supports_interruptible_capture at its inherited False, so the worker keeps calling both
    # capture entry points below with no arguments and Stop is honoured between profiles, as it
    # always was here. Declaring the flag without threading the callable would be worse than
    # leaving it off — the worker would believe Stop is handled during a capture while it
    # silently is not. The two waits a future implementation would need to reach are the album
    # readiness wait (_wait_for_profile_album_ready) and the per-photo advance settle; the
    # priority is low while Bumble web has no live target (see platforms.py).
    def _capture_current(self) -> Profile:
        return Profile(
            photos=self._capture_photos(),
            bio=self._text(self.selectors["bio"]),
            prompts=self._capture_prompts(),
            # Registry pools Bumble web + the Bumble Android app under one
            # store_key ("bumble") -- it's the same person's taste either way, so
            # the ranker should keep training on all of it (see platforms.py's
            # Platform.bucket). Keep stamping "bumble" here even though this
            # driver's own platform id is "bumble_web"; do NOT change this to
            # self.hud_label or self.platform_app.
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
        with _browser_guard():
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

    def wait_for_decision(self, timeout: float | None = 120.0, should_stop=None) -> bool | None:
        """Block until YOU manually like/pass the current card.

        Returns True (liked), False (passed), or None (card changed under us /
        deck emptied / timeout / stop requested). timeout=None waits
        indefinitely. Mouse clicks on the like/pass controls are detected; if
        you also use keyboard shortcuts, confirm coverage with bumble_inspect.py.
        """
        self._install_observe_listener()
        # Discard any decision recorded during the PREVIOUS card's capture/embed
        # window. With the in-page busy modal off (inpage_overlays default False),
        # you can physically swipe the next card while we're still embedding the
        # last one; that stray swipe sets window.__oplove_decision and, left
        # uncleared, would be mis-attributed to the card we're about to wait on ->
        # corrupted labels. Clearing here means only a swipe made AFTER this
        # card's capture can count for it.
        with _browser_guard():
            self.page.evaluate("() => { window.__oplove_decision = null; }")
        # Card-identity signal (BUMBLE-2): the click listener only fires on a
        # like/pass BUTTON click, so a swipe gesture, keyboard shortcut, or
        # app-driven advance would sail past it undetected — the deck moves on
        # to a new card, but we'd keep waiting, and the NEXT click we observe
        # (on the new card) would get attributed to THIS one: a silently
        # mislabelled training example. _card_fingerprint (already used by
        # _verify_swipe_landed) gives a cheap "which profile is on screen"
        # signal; None means it couldn't be read reliably, so we fail open
        # (skip the check) exactly like _verify_swipe_landed does.
        card_before = self._card_fingerprint()
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            if should_stop and should_stop():        # Stop pressed -> don't wait for a swipe
                return None
            with _browser_guard():
                decision = self.page.evaluate(
                    "() => { const v = window.__oplove_decision; window.__oplove_decision = null; return v; }"
                )
            if decision == "like":
                return True
            if decision == "pass":
                return False
            if self.out_of_profiles():
                return None
            if card_before is not None:
                card_now = self._card_fingerprint()
                if card_now is not None and card_now != card_before:
                    # The card changed without a like/pass click firing our
                    # listener -> it's gone. Report "no decision" so the caller
                    # re-captures instead of attributing the next click to it.
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

    def _wait_for_profile_album_ready(self) -> None:
        """Wait until Bumble's album elements have appeared and had time to paint.

        Its raw/filtered counts used to be returned, but the sole caller
        (_capture_photos) always recomputes both itself on the very next line
        (post overlay-hide, at actual capture time -- a more accurate moment
        for the log line than "when readiness was confirmed"), so the return
        value was dead. Fixed (BUMBLE-6): this is now a pure wait.
        """
        deadline = time.monotonic() + _PHOTO_READY_TIMEOUT_S
        stable_counts = None
        stable_since = None
        while True:
            raw, filtered, loaded = self._profile_photo_counts(require_loaded=True)
            now = time.monotonic()
            ready = filtered > 0 and loaded >= filtered
            counts = (filtered, loaded)
            if ready:
                if counts != stable_counts:
                    stable_counts = counts
                    stable_since = now
                elif stable_since is not None and now - stable_since >= _PHOTO_READY_SETTLE_S:
                    return
                if _PHOTO_READY_SETTLE_S <= 0:
                    return
            else:
                stable_counts = None
                stable_since = None
            if now >= deadline:
                return
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
        self._wait_for_profile_album_ready()
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
            from pathlib import Path
            d = Path("data/debug/captures") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            d.mkdir(parents=True, exist_ok=True)
            for idx, shot in enumerate(shots):
                (d / f"{idx:02d}.png").write_bytes(shot)
            print(f"Debug: wrote {len(shots)} captured photo(s) to {d}")
        except Exception as exc:  # noqa: BLE001
            print(f"Debug photo dump failed: {exc}")

    def _capture_prompts(self) -> list[tuple[str, str]]:
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

    # --- actions --------------------------------------------------------
    def like(self, opener: str | None = None, item_index: int | None = None, *,
             model_item_index: int | None = None, should_stop=None) -> None:
        # item_index is ignored: Bumble likes the whole profile (no per-photo comment). The
        # signature still tracks base.Driver.like's `int | None` (None = "nobody said which
        # item", distinct from 0 = "the first one") so no caller has to know which driver it is
        # holding before choosing what to pass.
        # model_item_index is ignored one step further on: doc 5.6's post-tap check verifies a
        # per-item comment screen against a stored crop of that item, and there is neither here.
        # For the same reason this flow can never raise base.ItemTargetingError -- there is no
        # per-item target to miss, so there is nothing for the never-substitute rule to protect.
        # NORMAL like only — never the super-swipe. Super-likes/boosts are the
        # owner's manual call (see DEFAULT_SELECTORS["superlike"]).
        before = self._card_fingerprint()
        self._human_click(self.selectors["like"])
        self._dbg_action("like", opener_chars=len(opener or ""))
        # NOTE: opener is not sent here on standard Bumble (post-match / women-first).
        # Hook point for Bumble "Opening Moves" once that flow is mapped live.
        self._verify_swipe_landed(before, "like")

    def dislike(self) -> None:
        before = self._card_fingerprint()
        self._human_click(self.selectors["pass"])
        self._dbg_action("dislike")
        self._verify_swipe_landed(before, "dislike")

    def out_of_profiles(self) -> bool:
        return self._query_selector(self.selectors["empty"]) is not None
