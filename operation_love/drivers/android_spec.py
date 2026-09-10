"""AndroidAppSpec — declarative description of one Android dating app.

This is what makes operation_love/drivers/hinge.py's driver reusable across more than one
Android target: everything that used to be hardcoded Hinge constants (the old module-level
`DEFAULTS` dict, mirrored by config.yaml's `apps.hinge`) is now a spec instance, and the
driver (`AndroidDriver`, defined in hinge.py — see that module's docstring for why) reads
`self.spec.*` instead of literals. Adding a second app (Bumble) means adding a second spec,
not forking the perception/action code.

Pure data: no adb/cv2/touch imports here, so importing a spec can never fire a touch or open
a device. `calibrated` licenses both Training and Auto readiness — see
operation_love/platforms.py's `_apply_calibration` and
operation_love/drivers/android/__init__.py, which wire both values into the registry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_LIKE_FLOWS = frozenset({"comment_sheet", "direct"})
_DECIDE_GESTURES = frozenset({"tap", "card_swipe"})


def _finite(value: object) -> bool:
    """``math.isfinite`` that answers False instead of raising on an oversized Python int.

    Every range check in ``__post_init__`` below promises the same thing: a bad field value
    is refused with a ValueError naming the field. An oversized int (a config typo such as an
    extra zero, ``10**400``) is a perfectly valid Python int, so bare ``math.isfinite`` raises
    ``OverflowError: int too large to convert to float`` and the construction dies with the
    wrong exception and no field name -- past every caller that catches ValueError to report
    which spec is malformed. Same fix, same reason, as ``config._is_finite_number`` and
    ``hinge._targeting_finite`` (2026-09-02).
    """
    try:
        return math.isfinite(value)
    except (TypeError, OverflowError):
        return False


@dataclass(frozen=True)
class AndroidAppSpec:
    """One Android dating app's identity, action points, and behavioral knobs.

    Fields fall into three groups:

      identity   — app, package, calibrated
      UI         — coords, templates, like_flow
      driver cfg — accepts_opener, think_time_calibrated, change_threshold, scroll_captures,
                   dwell_s, read_scroll_frac (all config.yaml-overridable per app, same as
                   Hinge's DEFAULTS always were)
    """

    app: str
    # Registry id ("hinge" / "bumble") — also the `apps.<app>` key in config.yaml and the
    # key operation_love.platforms uses to bucket labels/decisions.

    package: str
    # Android package name (`co.hinge.app` / `com.bumble.app`), launched via
    # `monkey -p <package> -c android.intent.category.LAUNCHER 1`.

    calibrated: bool
    # True ONLY once coords + templates below have been verified against a real device.
    # False fails the platform closed (operation_love.platforms) before any driver is even
    # constructed — see that module's docstring. Never flip this by hand for an app whose
    # numbers are still guesses; it is the one thing standing between a placeholder
    # coordinate and a real touch on a real account.

    coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Action points as FRACTIONS of the screen (x, y in 0..1). Some are fixed taps (e.g.
    # Hinge's comment box / send button); others are only the FALLBACK used when vision
    # (`templates`) can't locate the real button on-screen.

    templates: dict[str, str] = field(default_factory=dict)
    # Logical UI role -> PNG filename under operation_love/drivers/assets/. Roles the driver
    # knows about: "like" (the like glyph), "pass" (the pass glyph), "confirm" (the
    # send/"Send Like" button used to detect a still-open confirmation sheet), "upsell_dismiss"
    # (the dismiss/"send anyway" control on a paid-upgrade interstitial — NEVER the paid option
    # itself), and "paywall" (fixed chrome used to detect that a full-screen upgrade paywall is
    # covering the deck — Hinge's is hinge_upgrade_tab.png, the "HingeX" tab wordmark cropped
    # from the tab bar; chosen because the tab bar is FIXED chrome while the hero art below it
    # is rotating marketing imagery — MEASURED 2026-08-11: cv2.TM_CCOEFF_NORMED 0.965-1.000 on
    # the live paywall under gain/bias perturbation vs. a max of 0.4903 over all 88 real
    # non-paywall frames of the run this template exists to fix — see hinge.py's
    # _PAYWALL_MATCH_THRESHOLD / _paywall_visible). Unlike the other roles, "paywall" names no
    # tap target: it is detection-only, because the whole point is that observe mode must
    # recognise the screen and STOP rather than touch it — the paywall is a purchase screen and
    # must never be dismissed automatically. A role with no entry here means vision-location for
    # that role is skipped entirely (no template to match against); the driver falls back to the
    # fixed coordinate in `coords` for a tappable role (or, for upsell_dismiss, simply does
    # nothing — there is no safe fixed-coordinate fallback for a button we must never mis-tap).
    # "paywall" has no coords fallback at all: an unmatched paywall template just means this
    # detection path stays silent, never that some control gets tapped in its place.

    like_flow: str = "direct"
    # "comment_sheet" — Hinge: tap the heart -> a comment/"Send Like" sheet opens ->
    #                    optionally type the opener into it -> tap Send. The opener is
    #                    attached AT LIKE TIME.
    # "direct"        — Bumble: tap the like control, done. No comment box, no opener.

    decide_gesture: str = "tap"
    # HOW a like/pass is delivered, independent of like_flow (which is about what happens
    # after).
    #   "tap"        — aim at the like/pass control (vision-located, coord fallback). Hinge.
    #   "card_swipe" — drag the card body sideways instead. Needed wherever a PAID control
    #                  sits close enough to the like/pass targets that a placeholder or
    #                  drifted coordinate could land on it: a drag starts in the middle of
    #                  the card and cannot press a button it merely passes over, so the paid
    #                  control becomes unreachable by construction rather than by care.
    # A "card_swipe" spec must define the swipe_start / swipe_like_end / swipe_pass_end
    # coords; __post_init__ enforces that so the gap is caught at import, not mid-run.

    forbidden_zones: tuple[tuple[float, float, float, float], ...] = ()
    # Normalised (x0, y0, x1, y1) rects, 0..1, that a tap must NEVER land inside. The driver
    # raises rather than tapping when a target resolves into one — see hinge.py's
    # _assert_tap_allowed. This exists because "never tap the paid button" cannot be
    # guaranteed by only ever *aiming* elsewhere: vision can mis-match, and an uncalibrated
    # coordinate is a guess by definition. Declaring the paid control's territory turns the
    # owner's never-super-like rule into something the code refuses to violate.

    upsell_dismiss_zone: tuple[float, float, float, float] | None = None
    # Normalised (x0, y0, x1, y1) rect: a safe region to tap to dismiss a paid-upgrade
    # interstitial whose real dismiss control is not a single button but a large "tap
    # anywhere outside the sheet" overlay. Bumble's SuperSwipe purchase sheet is exactly
    # this shape (MEASURED live on the device 2026-08-10: the dimmed area above the sheet,
    # x 0.000-1.000, y 0.000-0.394, closes it on tap — see BUMBLE_SPEC in
    # operation_love/drivers/android/bumble.py for the full measurement and the narrower,
    # jitter-safe band actually declared there).
    #
    # When set, AndroidDriver._handle_rose_upsell (hinge.py) taps a FRESH RANDOM point
    # inside this rect on every dismiss attempt, instead of a fixed coordinate or a
    # vision-matched glyph location: a fixed point here would be exactly the kind of bot
    # signature the owner's no-fixed-constants rule forbids, and there is no single button
    # glyph to match a tap target against in the first place — the whole dimmed area IS the
    # control. Detecting that the sheet is actually up STILL requires the 'upsell_dismiss'
    # template (see `templates` above); with a zone declared but no template, this is inert,
    # matching every other vision-gated action in this driver: no detection, no tap, ever.
    # After each tap, the driver re-checks for the template and, if the sheet is still up
    # after a bounded number of attempts, HALTS rather than tapping again (see
    # AndroidDriver._dismiss_via_zone / PaidUpsellStuckError) — repeated blind taps near a
    # modal like this one are how a purchase gets confirmed, not how one gets avoided.
    #
    # None (the default) keeps the original behaviour: tap the vision-matched
    # 'upsell_dismiss' glyph location directly. That is correct for Hinge, where the glyph
    # names a real, single, well-defined "Send Like anyway" button — there is nothing to
    # randomise a search over.

    has_paid_upsell: bool = True
    # Does this app interrupt a like with a paid-upgrade interstitial (Hinge's "Send a Rose
    # instead?", Bumble's SuperSwipe purchase)? When True, the spec MUST declare an
    # "upsell_dismiss" template before it can be marked calibrated.
    #
    # Why this is a field rather than an inference from `templates`: with no template,
    # _handle_rose_upsell silently no-ops, and _verify_progress's only question is "did the
    # screen change" — which a modal appearing satisfies. So an undetected upsell would be
    # recorded as a SUCCESSFUL decision and left on screen for the next gesture to hit
    # unpredictably. Defaulting to True means forgetting to capture the template blocks
    # calibration; setting it False is a claim someone made deliberately after looking.

    accepts_opener: bool = False
    # Can a written opener be attached AT SWIPE TIME? True only for comment_sheet-style
    # flows. Mirrors DatingAppDriver.accepts_opener (worker.py skips generating an opener
    # entirely when this is False, to avoid spending Gemini quota/spend on text that can
    # never be sent).

    think_time_calibrated: bool = False
    # Is human_motion.think_time_s()'s like-vs-pass dwell asymmetry actually measured for
    # THIS app? True only for Hinge, which it was measured on.

    change_threshold: float = 9.0
    # Mean abs grayscale delta (0..255) on a 24x24 downsample to call a screen region
    # "changed" — see hinge.py's _split_diff / _changed.

    scroll_captures: int = 8
    # Max screencaps while reading one profile.

    dwell_s: float = 1.1
    # Per-card read dwell, humanized (Signals #1 on Hinge: read the whole profile before
    # deciding).

    read_scroll_frac: float = 0.55
    # How far each read-scroll advances the profile (fraction of screen height).

    identity_band: tuple[float, float, float, float] | None = None
    # Normalised (x0, y0, x1, y1) of the app's own STICKY PER-PROFILE HEADER: a region that
    # stays pixel-identical while one profile is open and changes completely the moment the
    # deck advances to a new profile (see hinge.py's _identity_of). This is the authoritative
    # anchor for observe mode: whatever else moved on screen, a frame whose identity band still
    # matches the captured profile is still the same card, never a decision -- this is what
    # fixes the bug where a human scrolling to read a profile (Hinge Signals behavior #1) got
    # recorded as a PASS with no tap of any kind (see the observe-mode redesign notes at the top
    # of hinge.py). None means this app has no positive identity anchor. Content matching can
    # still recognise a scroll and prevent a false label, but an otherwise ambiguous manual
    # PASS remains unresolved rather than falling back to "changed and unrecognised" proof.

    identity_top_name_band: tuple[float, float, float, float] | None = None
    # Normalised (x0, y0, x1, y1) of the app's own CARD HEADER name as rendered when the card
    # is at SCROLL-TOP -- the position where identity_band (above) shows profile-independent
    # chrome (Hinge's "Signals / Age / Height / Dating Intent" filter-chips row) instead of the
    # person's name, which is precisely the state in which observe mode otherwise cannot tell
    # one profile from the next. This is what fixes the bug where a pass advanced the deck from
    # "Zorva" to "Qelix" and got recorded as a scroll WITHIN Zorva's profile: identity_band's
    # verdict at scroll-top is the inconclusive "top" (nothing on screen names the card), so the
    # decision fell through to a loose pixel content-match that mistook the new woman's card for
    # a continuation of the old one -- see hinge.py's module docstring and _identity_of for the
    # full account.
    #
    # OCR-only, deliberately -- never a pixel signature, unlike identity_band/upsell_dismiss_zone
    # above. This band's content shifts vertically depending on whether Hinge is showing its
    # per-profile "shows thoughtful signals" banner for THIS card (MEASURED on the Pixel 7a
    # 2026-08-10: "Zorva %" with the banner present vs. "Zorva @ | @ Signals Active today" once
    # the banner is gone and content shifts up) -- a fixed pixel crop over it would therefore read
    # as a MISMATCH for the very same profile depending on which layout happened to render, while
    # OCR reads the name correctly either way because it is position-tolerant within the crop
    # (tesseract --psm 6 read the name correctly on every scroll-top frame tested, both layouts,
    # including three separate "Qelix &" frames after the deck advanced).
    #
    # None (the default) means this app declares no such band; a scroll-top identity stays the
    # inconclusive "top" verdict and cannot by itself authorize a PASS label.

    identity_top_name_fallback_band: tuple[float, float, float, float] | None = None
    # Optional, separately calibrated compact card-header OCR crop.  It is consulted only after
    # both recipes for identity_top_name_band are inconclusive while the identity layer has
    # already established a genuine scroll-top.  This preserves the broad band's two-layout
    # coverage while providing a deliberately smaller retry when the broad crop reaches prompt
    # text/photo texture and Tesseract drops the plainly visible name line.

    identity_top_name_take_another_look_band: tuple[float, float, float, float] | None = None
    # Optional, separately calibrated card-header OCR crop for Hinge's ``Take another look``
    # layout.  That banner sits inside identity_top_name_band and pushes the person's name below
    # it, so this lower crop is consulted ONLY after BOTH primary-band OCR recipes read that
    # exact banner at a genuine scroll top.  It is not a generic shifted detector: without the
    # structural banner gate it could inspect arbitrary photo/prompt content and manufacture a
    # different-profile name.  Its own two OCR recipes must agree before a name verdict is
    # committed, and the result remains subject to canonical-top, ready-deck, repeated-source,
    # and stable-frame gates.

    paywall_headline_band: tuple[float, float, float, float] | None = None
    # Normalised (x0, y0, x1, y1) crop of Hinge's "You're out of free likes for today" Hinge+
    # upgrade headline (MEASURED live on the Pixel 7a, 1080x2400, 2026-08-11 -- see
    # ops/calibration/hinge_out_of_likes_20260811.png and the read-only uiautomator dump
    # committed alongside it: band = (0.0556, 0.1958, 0.9537, 0.3000), i.e. px
    # (60,470)-(1030,720)).
    #
    # OCR'd BEST-EFFORT ONLY -- the screen itself is detected by the "paywall" TEMPLATE (see
    # `templates` above), never by OCR over this band: detection must not depend on tesseract
    # being installed. This band exists purely to refine the operator-facing stop message from
    # the generic "the deck is not available" into the specific "Hinge is out of free likes for
    # today" once OCR confirms the headline (see AndroidDriver._paywall_headline in hinge.py,
    # which never raises -- a missing or failed tesseract read costs message specificity only,
    # never detection).
    #
    # None (the default) means no headline refinement is attempted even when a "paywall"
    # template is declared and matches; the stop message then falls back to the generic
    # wording.

    content_band: tuple[float, float] = (0.125, 0.875)
    # (y0, y1) fractions bounding the SCROLLING content only -- excluding fixed chrome (status
    # bar, sticky header, floating like/pass overlay, bottom nav) that does NOT translate when
    # the profile scrolls. Used exclusively by _vertical_shift_match (hinge.py) to slice both
    # frames before searching for a shift match: searching the whole frame mixes in that fixed
    # chrome and inflates the measured shift distance 3-4x past change_threshold, which is why
    # the whole-frame version of that helper could never fire in production (see the measured
    # shift-distance table in hinge.py's module docstring). The default here is a placeholder
    # value, not a measurement -- each app should declare its own measured band.

    observe_ignore_zones: tuple[tuple[float, float, float, float], ...] = ()
    # Normalised (x0, y0, x1, y1) rects whose taps are known NOT to be decisions: rewind arrow,
    # overflow "...", bottom nav bar -- controls that sit on the card screen but do not advance
    # the deck the way a like/pass does. wait_for_decision's gesture-corroboration step
    # (hinge.py) treats a card advance whose last tap landed inside one of these as a resync
    # (`return None`) rather than a label, the same "recapture, record nothing" contract
    # worker.py already gives any other None decision. Same shape and validation as
    # forbidden_zones above; a separate field because these are about labelling a decision
    # correctly, not about refusing to tap a paid control.

    observe_touch_watch: bool = False
    # Read the device's own touch event stream (host-side `adb shell getevent`, read-only --
    # see touchwatch.py) to corroborate observe-mode decisions with actual gesture evidence
    # instead of pixels alone. True only for an app whose identity_band is declared: the touch
    # stream corroborates an identity-proven advance, it is not a standalone decision source
    # (enforced in __post_init__ below).

    auto_policy_calibrated: bool = False
    # Is the calibrated auto-mode BEHAVIOR POLICY (Worker._auto_loop's per-run tuning of read
    # geometry/dwell/capture-ceiling -- see hinge.py's set_auto_session_policy) actually
    # MEASURED and safe to apply on THIS app? True only for Hinge, the app the redesign was
    # calibrated against, mirroring think_time_calibrated above. AndroidDriver.
    # set_auto_session_policy installs the policy object only when this is True; every other
    # app -- today just the experimental Bumble path, whose paid-control zones and reading
    # geometry are different -- retains its legacy, un-policy-tuned behavior even when a
    # Worker session hands one in. Observe mode never calls set_auto_session_policy at all and
    # likewise keeps calibrated legacy behavior regardless of this flag. Formerly a bare
    # `self.spec.app == "hinge"` check inside hinge.py; moved here 2026-09 so the gate is a
    # declared capability of the spec, not a string comparison against the app's name.

    observe_input_serialized: bool = False
    # Does this app's passive OBSERVE mode need a cross-process input LEASE serializing
    # reads/scrolls against a live human decision wait (hinge.py's _observe_input_lease)? A
    # review controller may take screenshots while the Worker is waiting, but on an app where
    # this is True it must not also call current_profile/_scroll_to_top: both would move the
    # same card underneath wait_for_decision and can make a header reflow look like a PASS.
    # True only for Hinge, the app this OBSERVE redesign exists for; False keeps the lease a
    # no-op context manager everywhere else -- today just the experimental Bumble path, whose
    # observe machinery this redesign was never calibrated against. Formerly a bare
    # `self.spec.app != "hinge"` early-return inside hinge.py; moved here for the same reason
    # as auto_policy_calibrated above.

    safe_rewind_max_frac: float | None = None
    # Ceiling (fraction of screen height) on the RECOVERY rewind stroke's DISTANCE and, when
    # set, ALSO selects the bounded central-corridor LANE (fixed x=0.5, no independently
    # varied width/position) for that stroke instead of the generic wider/randomized undo
    # geometry -- see AndroidDriver._scroll_to_top_unlocked in hinge.py, which computes
    # `min(self.read_scroll_frac, self.spec.safe_rewind_max_frac)`.
    #
    # Rewind strokes must stay in Hinge's central, scrollable card region. A former
    # 0.78-screen ``fast flick`` ran from y=263 to y=2136 on the calibrated 2400px display:
    # its release was inside Hinge's fixed bottom navigation (the exact gesture that failed
    # to move the Vicki card in run de89edfe05e1). The driver's generic start-only zone check
    # cannot establish that an endpoint is safe for a custom Hinge view, so this is a per-app
    # driver invariant, not a config.yaml-overridable preference -- it is deliberately absent
    # from config.py's generic `*_frac` override validation. At Hinge's 0.55 the stroke runs
    # y=540..1860 -- the established read-scroll corridor. A smaller configured
    # read_scroll_frac shrinks it further (the min() above); `rewind_scroll_frac` can never
    # enlarge this Hinge gesture again. The bounded retry ceiling, and affirmative top verdict
    # after each try, remain the authority on arrival rather than a longer physical flick.
    #
    # None (the default) means no cap: the generic path keeps its original
    # max(read_scroll_frac, rewind_scroll_frac) undo distance and independently varied lane --
    # correct for an app with no measured incident forcing it into a narrower corridor. Was a
    # bare module constant (_HINGE_SAFE_REWIND_MAX_FRAC) in hinge.py, read through a
    # `self.spec.app == "hinge"` check; moved here for the same reason as
    # auto_policy_calibrated above.

    def __post_init__(self) -> None:
        if self.like_flow not in _LIKE_FLOWS:
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).like_flow must be one of "
                f"{sorted(_LIKE_FLOWS)} (got {self.like_flow!r})")
        if self.decide_gesture not in _DECIDE_GESTURES:
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).decide_gesture must be one of "
                f"{sorted(_DECIDE_GESTURES)} (got {self.decide_gesture!r})")
        if self.calibrated and self.has_paid_upsell and "upsell_dismiss" not in self.templates:
            # Only enforced at calibrated=True: an in-progress spec is expected to be
            # incomplete, but "ready for a real account" must mean it can recognise and
            # dismiss the modal that costs money.
            raise ValueError(
                f"AndroidAppSpec({self.app!r}) is marked calibrated but declares no "
                f"'upsell_dismiss' template while has_paid_upsell is True. A paid-upgrade "
                f"modal would go undetected, be recorded as a successful decision, and be "
                f"left on screen for the next gesture. Capture the template, or set "
                f"has_paid_upsell=False if this app genuinely has no such interstitial.")
        if self.like_flow == "comment_sheet" and "confirm" not in self.templates:
            # The comment_sheet flow taps FIXED coordinates into the sheet. Without a way to
            # confirm the sheet is actually up, those taps land on the profile card instead
            # (see AndroidDriver._await_sheet_open). Refuse the spec rather than the swipe.
            raise ValueError(
                f"AndroidAppSpec({self.app!r}) uses like_flow='comment_sheet' but declares no "
                f"'confirm' template — there would be no way to verify the sheet opened before "
                f"tapping into it")
        if self.decide_gesture == "card_swipe":
            # Fail at import rather than discovering a missing endpoint mid-swipe, where the
            # fallback would be to tap -- exactly what card_swipe exists to avoid.
            missing = [k for k in ("swipe_start", "swipe_like_end", "swipe_pass_end")
                       if k not in self.coords]
            if missing:
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}) uses decide_gesture='card_swipe' but is "
                    f"missing coords: {', '.join(missing)}")
        for zone in self.forbidden_zones:
            x0, y0, x1, y1 = zone
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).forbidden_zones entry {zone} is not a "
                    f"normalised (x0, y0, x1, y1) rect with x0<x1 and y0<y1 inside 0..1")
        if self.upsell_dismiss_zone is not None:
            # Same normalised-rect shape as forbidden_zones above -- a malformed zone here
            # would be silently unusable (or worse, an inverted/degenerate rect that
            # random.uniform can't sample sensibly), so catch it at import rather than at the
            # first dismiss attempt on a real device.
            x0, y0, x1, y1 = self.upsell_dismiss_zone
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).upsell_dismiss_zone {self.upsell_dismiss_zone} "
                    f"is not a normalised (x0, y0, x1, y1) rect with x0<x1 and y0<y1 inside 0..1")
        if self.identity_band is not None:
            # Same normalised-rect shape as upsell_dismiss_zone above -- this band is what
            # _identity_of (hinge.py) diffs on every poll to prove "still the same card" no
            # matter what else on screen changed. A malformed rect here would silently defeat
            # that anchor instead of raising, which is worse than declaring no anchor at all
            # (identity_band=None at least keeps the legacy content-only rule intact).
            x0, y0, x1, y1 = self.identity_band
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).identity_band {self.identity_band} is not a "
                    f"normalised (x0, y0, x1, y1) rect with x0<x1 and y0<y1 inside 0..1")
        if self.identity_top_name_band is not None:
            # Same normalised-rect shape as identity_band above -- a malformed rect here would
            # silently defeat the scroll-top name check (OCR over garbage, or over the wrong
            # part of the screen) rather than raise, which is worse than declaring no band at
            # all (identity_top_name_band=None at least keeps the legacy scroll-top "top"
            # verdict intact instead of quietly mis-reading it as some other profile's name).
            x0, y0, x1, y1 = self.identity_top_name_band
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).identity_top_name_band "
                    f"{self.identity_top_name_band} is not a normalised (x0, y0, x1, y1) rect "
                    f"with x0<x1 and y0<y1 inside 0..1")
        if self.identity_top_name_band is not None and self.identity_band is None:
            # The name check is a REFINEMENT of the pixel identity verdict, not a standalone
            # source: it only ever runs when that verdict is exactly "top" (hinge.py's
            # _identity_of), and "top" cannot be produced without identity_band declared in the
            # first place. Declaring identity_top_name_band with no identity_band is therefore a
            # configuration error -- refused here the same way observe_touch_watch=True paired
            # with identity_band=None is refused below, rather than left to silently never fire.
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).identity_top_name_band is set but identity_band "
                f"is None -- the scroll-top name check refines the pixel identity verdict, it "
                f"is not a standalone source, and that verdict cannot be produced without "
                f"identity_band declared")
        if self.identity_top_name_fallback_band is not None:
            x0, y0, x1, y1 = self.identity_top_name_fallback_band
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).identity_top_name_fallback_band "
                    f"{self.identity_top_name_fallback_band} is not a normalised "
                    "(x0, y0, x1, y1) rect with x0<x1 and y0<y1 inside 0..1")
        if (self.identity_top_name_fallback_band is not None
                and (self.identity_band is None or self.identity_top_name_band is None)):
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).identity_top_name_fallback_band is set but "
                "identity_band and identity_top_name_band are required -- this compact OCR "
                "crop is only a fallback refinement of the canonical card-header name path")
        if self.identity_top_name_fallback_band is not None:
            bx0, by0, bx1, by1 = self.identity_top_name_band
            fx0, fy0, fx1, fy1 = self.identity_top_name_fallback_band
            # This is not a separately authorised OCR detector: it may only remove the
            # primary crop's lower portion, where the no-banner layout reaches prompt text.
            # A contained left/right/top shift would silently inspect a new region and could
            # manufacture a name candidate from unrelated UI or photo content.
            if not (fx0 == bx0 and fx1 == bx1 and fy0 == by0 and fy1 <= by1):
                raise ValueError(
                f"AndroidAppSpec({self.app!r}).identity_top_name_fallback_band must be "
                    "the primary band with only y1 allowed to shrink -- a retry with shifted "
                    "geometry would be a separately unlicensed detector")
        if self.identity_top_name_take_another_look_band is not None:
            x0, y0, x1, y1 = self.identity_top_name_take_another_look_band
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).identity_top_name_take_another_look_band "
                    f"{self.identity_top_name_take_another_look_band} is not a normalised "
                    "(x0, y0, x1, y1) rect with x0<x1 and y0<y1 inside 0..1")
        if (self.identity_top_name_take_another_look_band is not None
                and (self.identity_band is None or self.identity_top_name_band is None)):
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).identity_top_name_take_another_look_band is "
                "set but identity_band and identity_top_name_band are required -- this lower "
                "OCR crop is only a structurally gated refinement of the canonical card-header "
                "name path")
        if self.identity_top_name_take_another_look_band is not None:
            if self.app != "hinge":
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).identity_top_name_take_another_look_band "
                    "is a Hinge-only calibrated layout and must not authorize OCR geometry for "
                    "another app")
            bx0, by0, bx1, by1 = self.identity_top_name_band
            lx0, ly0, lx1, ly1 = self.identity_top_name_take_another_look_band
            # The panel inserts itself ABOVE the name.  Keep the same horizontal span and make
            # the lower crop overlap the old bottom edge before extending below it: this is the
            # only measured relationship that can recover the displaced header without turning
            # the option into an arbitrary OCR rectangle elsewhere on the card.
            # The calibration's .035 bottom extension leaves a small per-device remeasurement
            # envelope, but a crop that reaches arbitrarily far into the first photo turns the
            # panel gate into an OCR-from-photo detector.  These normalized limits retain a
            # 120px upward / 144px downward adjustment window on the 2400px target device while
            # keeping the crop tightly bound to the displaced header.
            if not (lx0 == bx0 and lx1 == bx1 and by1 - 0.05 <= ly0 < by1 < ly1
                    <= by1 + 0.06 and ly1 - ly0 <= 0.10):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).identity_top_name_take_another_look_band "
                    "must keep the primary card-header band's horizontal span and overlap its "
                    "bottom edge within the calibrated tight-header envelope -- the exact Take "
                    "another look OCR gate is not a licence for arbitrary shifted geometry or "
                    "first-photo content")
        if self.paywall_headline_band is not None:
            # Same normalised-rect shape as identity_top_name_band above -- a malformed rect
            # here would silently defeat the OCR refinement (garbage text, or a crop over the
            # wrong part of the screen) rather than raise. Unlike identity_top_name_band, a bad
            # rect here is never the difference between detecting the screen and not -- that is
            # what the "paywall" template is for, checked separately below -- but it should
            # still fail at import rather than at the first live paywall.
            x0, y0, x1, y1 = self.paywall_headline_band
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).paywall_headline_band "
                    f"{self.paywall_headline_band} is not a normalised (x0, y0, x1, y1) rect "
                    f"with x0<x1 and y0<y1 inside 0..1")
        if self.paywall_headline_band is not None and "paywall" not in self.templates:
            # The headline band is an OCR REFINEMENT of the paywall verdict, not a standalone
            # detector -- it only ever runs once the "paywall" template has already matched
            # (AndroidDriver._deck_blocked_reason, hinge.py). Declaring a headline band with no
            # "paywall" template to detect the screen it sits on is therefore a mis-wiring: a
            # band with nothing to refine -- refused here the same way identity_top_name_band
            # with no identity_band is refused above.
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).paywall_headline_band is set but no 'paywall' "
                f"template is declared -- the headline band refines the paywall verdict, it is "
                f"not a standalone detector, and there is nothing to detect the screen it sits "
                f"on without a 'paywall' template")
        for zone in self.observe_ignore_zones:
            # Same normalised-rect shape as forbidden_zones above -- these are the taps
            # wait_for_decision (hinge.py) must recognise as a resync rather than a decision
            # (rewind arrow, overflow "...", bottom nav). A malformed entry here would silently
            # fail to recognise a non-decision tap and could mislabel a resync as a real PASS.
            x0, y0, x1, y1 = zone
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).observe_ignore_zones entry {zone} is not a "
                    f"normalised (x0, y0, x1, y1) rect with x0<x1 and y0<y1 inside 0..1")
        # content_band: same "fraction, not pixel, and must actually be finite" reasoning as
        # the coords / read_scroll_frac check below, but for a (y0, y1) pair rather than a
        # single value or an (x0,y0,x1,y1) rect -- _vertical_shift_match (hinge.py) slices both
        # frames to these rows before searching, and a bad pair here would silently reintroduce
        # the fixed-chrome bug this field exists to fix (see the measured shift-distance table
        # in hinge.py's module docstring).
        y0, y1 = self.content_band
        if not (_finite(y0) and _finite(y1) and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).content_band {self.content_band} must be a "
                f"finite (y0, y1) pair with 0.0 <= y0 < y1 <= 1.0")
        if self.observe_touch_watch and self.identity_band is None:
            # The touch stream corroborates an identity-proven card advance (wait_for_decision's
            # gesture-corroboration step, hinge.py) -- it is not a standalone decision source.
            # Without identity_band there is no identity-proven advance for a tap or drag to
            # corroborate in the first place, so this combination is refused at construction
            # rather than left to silently never corroborate anything at runtime.
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).observe_touch_watch is True but identity_band "
                f"is None -- the touch stream corroborates an identity-proven advance, it is "
                f"not a standalone decision source")
        # coords / read_scroll_frac range check. This is the SPEC-level half of a two-part
        # fix (the other half is config.py's _validate_android_fractions, which covers an
        # OPERATOR's config.yaml override of these same values): an out-of-range coordinate
        # here was previously caught nowhere until hinge.py's _assert_tap_allowed ran it
        # against a live screen -- and even then, only because that function now also checks
        # the RAW value before clamping (see its docstring). A bad literal baked into a spec
        # (a pixel value typo'd where a fraction was meant, e.g. 868 instead of 0.868) is a
        # code-review-time mistake, not a runtime one, so it belongs here: fail at import,
        # not at the first live tap. Every `coords` entry is a FRACTION of the screen (x, y
        # in 0..1), never a pixel -- an out-of-range value is never legitimate, because the
        # real touch transport clamps rather than refuses it, which can silently land inside
        # a forbidden_zones rect (see hinge.py's _assert_tap_allowed for the full mechanism
        # this guards against).
        for key, value in self.coords.items():
            if (not isinstance(value, tuple) or len(value) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
                raise ValueError(
                    f"AndroidAppSpec({self.app!r}).coords[{key!r}] must be an (x, y) tuple "
                    f"of numbers (got {value!r})")
            # The tuple-length check above establishes this 2:2 invariant. Keep strict=True so
            # a future coords shape change cannot silently skip an axis validation.
            for axis, v in zip("xy", value, strict=True):
                if not (_finite(v) and 0.0 <= v <= 1.0):
                    raise ValueError(
                        f"AndroidAppSpec({self.app!r}).coords[{key!r}].{axis} = {v!r} is not "
                        f"a fraction in 0..1. coords are fractions of the SCREEN, never "
                        f"pixels -- a value outside 0..1 would pass this rect-based check "
                        f"cleanly and then be silently clamped onto a screen edge by the real "
                        f"touch transport, which can land inside a forbidden zone.")
        # read_scroll_frac: same reasoning as the coords check above, for a non-coords fraction
        # field. Named generically ("*_frac") in config.py's sibling check because a future
        # field could add another; this one is checked by name since config.py's check only
        # covers `apps.<app>.*` OVERRIDES, not a spec's own hardcoded default.
        if not (_finite(self.read_scroll_frac) and 0.0 <= self.read_scroll_frac <= 1.0):
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).read_scroll_frac must be a fraction in 0..1 "
                f"(got {self.read_scroll_frac!r})")
        # safe_rewind_max_frac: same shape check, but optional -- None means "no cap" (see the
        # field's own comment above), so only a NON-None value is range-checked. This field is
        # not config.yaml-overridable (a driver invariant, not operator tuning -- see its
        # comment), so unlike read_scroll_frac it is never reached by config.py's generic
        # `*_frac` sibling check either; this is its only validation.
        if (self.safe_rewind_max_frac is not None
                and not (_finite(self.safe_rewind_max_frac)
                         and 0.0 <= self.safe_rewind_max_frac <= 1.0)):
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).safe_rewind_max_frac must be a fraction in "
                f"0..1 or None (got {self.safe_rewind_max_frac!r})")
