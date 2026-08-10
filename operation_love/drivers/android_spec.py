"""AndroidAppSpec — declarative description of one Android dating app.

This is what makes operation_love/drivers/hinge.py's driver reusable across more than one
Android target: everything that used to be hardcoded Hinge constants (the old module-level
`DEFAULTS` dict, mirrored by config.yaml's `apps.hinge`) is now a spec instance, and the
driver (`AndroidDriver`, defined in hinge.py — see that module's docstring for why) reads
`self.spec.*` instead of literals. Adding a second app (Bumble) means adding a second spec,
not forking the perception/action code.

Pure data: no adb/cv2/touch imports here, so importing a spec can never fire a touch or open
a device. `calibrated` is the field that matters for safety — see
operation_love/platforms.py's `_apply_calibration`, which derives whether a platform is even
allowed to start from this flag, and operation_love/drivers/android/__init__.py, which wires
each spec's flag into that registry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_LIKE_FLOWS = frozenset({"comment_sheet", "direct"})
_DECIDE_GESTURES = frozenset({"tap", "card_swipe"})


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
    # send/"Send Like" button used to detect a still-open confirmation sheet), and
    # "upsell_dismiss" (the dismiss/"send anyway" control on a paid-upgrade interstitial —
    # NEVER the paid option itself). A role with no entry here means vision-location for
    # that role is skipped entirely (no template to match against); the driver falls back
    # to the fixed coordinate in `coords` (or, for upsell_dismiss, simply does nothing —
    # there is no safe fixed-coordinate fallback for a button we must never mis-tap).

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
            for axis, v in zip("xy", value):
                if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                    raise ValueError(
                        f"AndroidAppSpec({self.app!r}).coords[{key!r}].{axis} = {v!r} is not "
                        f"a fraction in 0..1. coords are fractions of the SCREEN, never "
                        f"pixels -- a value outside 0..1 would pass this rect-based check "
                        f"cleanly and then be silently clamped onto a screen edge by the real "
                        f"touch transport, which can land inside a forbidden zone.")
        # read_scroll_frac: same reasoning, for the one non-coords fraction field this spec
        # declares today. Named generically ("*_frac") in config.py's sibling check because
        # a future field could add another; this one is checked by name since it is the only
        # dataclass field of this shape.
        if not (math.isfinite(self.read_scroll_frac) and 0.0 <= self.read_scroll_frac <= 1.0):
            raise ValueError(
                f"AndroidAppSpec({self.app!r}).read_scroll_frac must be a fraction in 0..1 "
                f"(got {self.read_scroll_frac!r})")
