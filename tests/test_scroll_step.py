"""The closed-loop enumeration scroll (operation_love/drivers/scroll_step.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures that this
module's constants were measured against are real people's dating profiles and are gitignored
(ops/calibration/, .gitignore:26); only geometry and counts from them appear anywhere in this
repo. So the fixtures rebuild the measured LAYOUT from first principles — a gradient page
background, rounded-rect cards inset 53px per side, canonical 53px gutters, the genuine shipped
like glyph stamped bottom-right on the likeable ones — and the ground truth is therefore known by
construction, which is what lets the assertions name exact pixel spacings.

The frames go through the REAL `segment_frame`, not hand-written `FrameSegmentation` records, on
purpose: the thing under test is a measurement of card spacing, and a hand-built segmentation
would let the test agree with itself about geometry that the segmenter is the actual authority
on. The one place this file builds a result by hand is `step_overshoot`, which is arithmetic over
a plan and takes no pixels at all.

Two fixture profiles carry most of the file, and the pairing is the point:

  * `_tall_frame()` — ~974px cards, the 1027px spacing doc 5.10.1 measured, where the validated
    363px cadence sits at a safe ratio 0.35;
  * `_short_frame()` — 620px cards, a 673px spacing, where that SAME 363px step would be ratio
    0.54 and would land back in the aliasing band. The logic must choose smaller, and the two
    step distributions must not even overlap.

Every positive is paired with a negative plus a control that proves WHICH mechanism did the
excluding — the heartless-block rule in particular, which is asserted by giving the same block a
heart and watching the plan refuse.
"""
import math
import random
import subprocess
import sys

import cv2
import numpy as np
import pytest

from operation_love.drivers import frameshift, hinge, scroll_step, segment
from operation_love.drivers.hinge import HINGE_SPEC, AndroidDriver

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 11
_CONTENT_BAND = HINGE_SPEC.content_band                # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - segment._CARD_MARGIN_PX                # 1027
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px"
_HEART_CX = 937                                        # bottom-right, as test_segment.py stamps it
_HEART_ABOVE_BOTTOM = 90                               # a complete card's heart sits ~89px up
_PAGE_TOP, _PAGE_BOTTOM = 254, 243                     # doc 5.10's background gradient

_TEMPLATE = hinge._load_template(HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"

# The two cadences the transport model was measured on (doc 5.10.1 + its addendum, re-measured
# through frameshift.estimate_shift): read_scroll_frac 0.16 moves the content 363px and 0.55
# moves it 1299px, both on this 2400px screen.
_MEASURED_CADENCES = ((0.16, 363), (0.55, 1299))


def _page_column():
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, _H).round().astype(np.uint8)


class _Frame:
    """A synthetic Hinge profile frame, built by painting onto a gradient page background.

    Same construction as tests/test_segment.py's fixture, including the `ceil`-carved corner arc:
    on a true rounded rect the edge row's inset and the rows the arc takes to close agree on one
    radius, which is the invariant segment.py's card-corner test keys on. A fixture that only
    approximated it would stop bounding item 1 and item N, and this module's whole measurand is
    the extent of bounded cards.
    """

    def __init__(self):
        self.col = _page_column()
        self.gray = np.repeat(self.col[:, None], _W, axis=1)
        self.rng = np.random.default_rng(_SEED)

    def card(self, y0, y1, *, heart=True, radius=_CORNER_RADIUS_PX):
        """Rows [y0, y1), clipped to the frame — a card whose bottom runs off the screen is drawn
        as far as it goes, which is what the lowest card of any real frame looks like."""
        v0, v1 = max(0, y0), min(_H, y1)
        if v1 > v0:
            self.gray[v0:v1, _CARD_X0:_CARD_X1] = self.rng.integers(
                60, 200, size=(v1 - v0, _CARD_X1 - _CARD_X0), dtype=np.uint8)
        for i in range(radius):                        # carve the four corner arcs back to page
            dy = radius - i
            inset = int(math.ceil(radius - math.sqrt(max(0.0, radius ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for y in (y0 + i, y1 - 1 - i):
                if not 0 <= y < _H:
                    continue
                self.gray[y, _CARD_X0:_CARD_X0 + inset] = self.col[y]
                self.gray[y, _CARD_X1 - inset:_CARD_X1] = self.col[y]
        if heart:
            self.heart(y1 - _HEART_ABOVE_BOTTOM)
        return self

    def heart(self, cy, cx=_HEART_CX):
        """Stamp the real shipped like glyph centred at (cx, cy); silently a no-op when the glyph
        would fall off the frame, which is how a card whose bottom is off-screen shows none."""
        th, tw = _TEMPLATE.shape
        y0, x0 = cy - th // 2, cx - tw // 2
        if y0 < 0 or y0 + th > _H or x0 < 0 or x0 + tw > _W:
            return self
        self.gray[y0:y0 + th, x0:x0 + tw] = _TEMPLATE
        return self

    def png(self):
        ok, buf = cv2.imencode(".png", self.gray)
        assert ok
        return buf.tobytes()

    def segment(self):
        return segment.segment_frame(
            self.png(), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD)


def _stack(heights, *, hearts=None, top=500):
    """A frame holding `heights` gutter-separated cards starting at row `top`.

    `hearts` is a per-card override; the default gives every card one. Cards that run past the
    analysed band are drawn anyway and come back PARTIAL, exactly as a real frame's lowest card
    does.
    """
    frame, y = _Frame(), top
    for i, height in enumerate(heights):
        wants_heart = True if hearts is None else hearts[i]
        frame.card(y, y + height, heart=wants_heart)
        y += height + _GUTTER
    return frame


# ~974px cards: doc 5.10's most common card height, giving the 1027px spacing 5.10.1 measured.
# One card complete inside the band, the next sliced by the band edge.
def _tall_frame():
    return _stack((974, 974))


# 620px cards: a 673px spacing. Chosen because 363 / 673 = 0.54, i.e. the validated fixed cadence
# would sit ABOVE the 0.5 line and back inside the aliasing band on this profile, while the
# spacing is still large enough for a legal gesture to respect the ratio rule.
def _short_frame():
    return _stack((620, 620, 620))


def _draws(seg, n=200, **kw):
    rng = random.Random(20260812)
    return [scroll_step.plan_scroll_step(seg, rng=rng, **kw) for _ in range(n)]


# =====================================================================================
# The transport model: the one place a wrong number would silently mis-size every step
# =====================================================================================

def test_the_transport_model_reproduces_both_measured_cadences():
    """`frac -> px` is calibration, not arithmetic. Both corpus cadences must come back exactly.

    These two numbers are the entire evidence for `_TOUCH_SLOP_PX`: a 384px drag moved the
    content 363px and a 1320px drag moved it 1299px, so the loss is a constant 21px rather than
    a gain. If this test ever fails, the constant has been re-derived instead of re-measured.
    """
    for frac, expected in _MEASURED_CADENCES:
        assert scroll_step.step_px_for_frac(frac, _H) == expected


def test_the_model_mirrors_the_transports_own_integer_truncation():
    """Not `frac * h`: both `Adb.scroll_up` and `UhidTouch.scroll_up` truncate each endpoint
    independently, and a plan's `step_px` claims to be what the device will really deliver."""
    for frac in (0.1037, 0.16, 0.2231, 0.3, 0.55):
        y1 = int(_H * (0.5 + frac / 2))
        y2 = int(_H * (0.5 - frac / 2))
        assert scroll_step.step_px_for_frac(frac, _H) == y1 - y2 - scroll_step._TOUCH_SLOP_PX


def test_the_inverse_round_trips_within_the_truncation():
    for px in range(200, 460, 7):
        frac = scroll_step.frac_for_step_px(px, _H)
        assert abs(scroll_step.step_px_for_frac(frac, _H) - px) <= 1


def test_a_frac_the_slop_swallows_reports_no_movement_rather_than_a_negative_distance():
    assert scroll_step.step_px_for_frac(0.001, _H) == 0


def test_a_degenerate_screen_raises_rather_than_dividing_by_it():
    with pytest.raises(scroll_step.ScrollStepError):
        scroll_step.frac_for_step_px(300, 0)


# =====================================================================================
# Local spacing: what is measured, and what is deliberately not
# =====================================================================================

def test_a_complete_card_gives_its_own_extent_plus_one_gutter():
    spacing = scroll_step.measure_local_spacing(_tall_frame().segment())
    assert spacing.measured
    assert spacing.px == 974 + min(segment._GUTTER_PX)
    assert scroll_step.SPACING_CARD_EXTENT in spacing.kinds


def test_two_hearts_on_one_frame_measure_the_pitch_directly():
    """The doc's own quantity, and it needs no block edge observed at all."""
    spacing = scroll_step.measure_local_spacing(_short_frame().segment())
    assert scroll_step.SPACING_HEART_PITCH in spacing.kinds
    pitches = [px for kind, px in spacing.evidence
               if kind == scroll_step.SPACING_HEART_PITCH]
    assert pitches == [620 + _GUTTER] * len(pitches)


def test_two_adjacent_card_tops_measure_the_pitch_when_the_lower_heart_is_off_band():
    """The case `SPACING_CARD_PITCH` exists for: the second card's heart sits below the analysed
    band, so heart-to-heart cannot see it, and its own extent is unbounded, so the extent rule
    cannot either. Only the UPPER block of the pair has to bear a heart."""
    seg = _tall_frame().segment()
    assert len(seg.hearts) == 1, "the fixture's lower card must show no heart"
    spacing = scroll_step.measure_local_spacing(seg)
    pitches = [px for kind, px in spacing.evidence if kind == scroll_step.SPACING_CARD_PITCH]
    assert pitches == [974 + _GUTTER]


def test_the_minimum_wins_because_the_smallest_period_is_the_hazard():
    """A tall card above two short ones. The heart-to-heart pitch spans the SHORT card's bottom
    edge to the tall card's, so it reports the short card's 673px period even though that card's
    own extent is never bounded on this frame — and 673 is what the step is sized against, not
    the 1026/1027 the tall card offers."""
    seg = _stack((974, 620, 620)).segment()
    spacing = scroll_step.measure_local_spacing(seg)
    assert spacing.px == min(px for _kind, px in spacing.evidence)
    assert spacing.px == 620 + _GUTTER
    assert max(px for _kind, px in spacing.evidence) == 974 + _GUTTER


def test_a_heartless_block_does_not_shrink_the_step():
    """doc 5.10's 215px vitals block sits between two cards and carries no heart in any frame.
    Counting it as a period would size the step at 89px — roughly 112 captures for one profile,
    and then a hard stop under the minimum legal gesture."""
    seg = _stack((974, 215, 974), hearts=(True, False, True)).segment()
    kinds = [b.kind for b in seg.blocks]
    assert segment.BLOCK_CONTEXT in kinds, "the fixture's vitals block must really be heartless"
    spacing = scroll_step.measure_local_spacing(seg)
    assert spacing.px == 974 + min(segment._GUTTER_PX)
    assert all(px > 215 + _GUTTER for _kind, px in spacing.evidence)


def test_the_control_the_same_block_with_a_heart_does_shrink_it():
    """The control that proves WHICH mechanism excluded the vitals block above. Give the very
    same 215px block a heart and it becomes a period — small enough that no legal gesture can
    respect the ratio rule, so the plan refuses rather than stepping further."""
    seg = _stack((974, 215, 974), hearts=(True, True, True)).segment()
    spacing = scroll_step.measure_local_spacing(seg)
    assert spacing.px == 215 + min(segment._GUTTER_PX)
    with pytest.raises(scroll_step.ScrollStepError, match="alias"):
        scroll_step.plan_scroll_step(seg)


def test_a_frame_that_offers_nothing_measurable_says_so_rather_than_guessing():
    """One card taller than the whole analysed band: no gutter, no corner, no heart in band."""
    frame = _Frame()
    frame.card(200, 2300, heart=True)                  # heart at 2210, below the band
    seg = frame.segment()
    assert not any(b.complete for b in seg.blocks)
    spacing = scroll_step.measure_local_spacing(seg)
    assert not spacing.measured and spacing.px is None and spacing.evidence == ()


# =====================================================================================
# The headline: the step follows the content, and a short card forces a smaller one
# =====================================================================================

def test_a_short_card_profile_steps_smaller_than_the_fixed_cadence_that_would_realias():
    """doc 5.10.1's own worked example, made executable: "a short prompt card could space hearts
    ~600px apart, where the same 363px step gives a ratio of 0.6 and lands back in the aliasing
    band"."""
    seg = _short_frame().segment()
    spacing = scroll_step.measure_local_spacing(seg).px
    assert spacing == 620 + min(segment._GUTTER_PX)

    # The counterfactual this test exists for, asserted rather than asserted about: the validated
    # FIXED cadence is unsafe here.
    fixed = scroll_step.step_px_for_frac(0.16, _H)
    assert fixed / spacing > 0.5

    for plan in _draws(seg):
        assert plan.step_px < fixed
        assert plan.ratio <= scroll_step._STEP_RATIO_MAX
        assert plan.step_px * 2 < spacing, "well under half the local spacing"


def test_a_tall_card_profile_is_allowed_the_larger_step():
    """The other half of the pair: the logic is not just "always step small". The same code on a
    1026px spacing spends nearly the whole budget the corpus validated — nearly, because
    `_STEP_DELIVERY_JITTER_PX` (2026-08-27) now reserves headroom under whichever ceiling binds,
    `_MAX_STEP_PX` included, so the window's top is `_MAX_STEP_PX` minus that margin rather than
    `_MAX_STEP_PX` itself.

    Asserted on the draw WINDOW rather than on the largest of 200 draws: the ceiling is the top of
    a uniform window now, not a clamp every over-large draw lands on, so "the biggest draw was
    exactly 363" would be a statement about the seed."""
    plans = _draws(_tall_frame().segment())
    spacing = scroll_step.measure_local_spacing(_tall_frame().segment()).px
    low_px = int(scroll_step._STEP_RATIO_MIN * spacing)
    cap_px = scroll_step._MAX_STEP_PX
    margin = min(scroll_step._STEP_DELIVERY_JITTER_PX, (cap_px - low_px) // 2)
    assert {p.window_px for p in plans} == {(low_px, cap_px - margin)}
    assert max(p.step_px for p in plans) <= scroll_step._MAX_STEP_PX
    assert min(p.step_px for p in plans) > 250


def test_the_two_profiles_step_distributions_do_not_even_overlap():
    """The load-bearing property of a content-following scroll: the distance is a measurement of
    the page, not a draw from a fixed distribution that happens to be wide."""
    short = [p.step_px for p in _draws(_short_frame().segment())]
    tall = [p.step_px for p in _draws(_tall_frame().segment())]
    assert max(short) < min(tall)


def test_every_ratio_lands_well_under_half_the_local_spacing():
    """The aliasing rule, over every fixture and many draws. 0.5 is where doc 5.10.1's failure
    begins; the ratio window's ceiling is `_STEP_RATIO_MAX` and nothing may exceed it."""
    assert scroll_step._STEP_RATIO_MAX < 0.4
    for seg in (_tall_frame().segment(), _short_frame().segment(),
                _stack((974, 620, 620)).segment(), _stack((1114, 1114)).segment()):
        for plan in _draws(seg, n=100):
            assert plan.ratio is not None
            assert plan.ratio <= scroll_step._STEP_RATIO_MAX
            assert plan.step_px <= plan.bound_px


def test_the_step_is_jittered_rather_than_a_fixed_content_locked_distance():
    """doc 5.5's anti-bot argument only holds if the content-following distance ALSO varies.
    A fixed step per card would be a per-card constant, which is a signature."""
    steps = {p.step_px for p in _draws(_tall_frame().segment(), n=200)}
    assert len(steps) > 30, steps


def test_the_jitter_scales_with_the_card_rather_than_being_a_fixed_wobble():
    """Because the draw window's ends are fractions of the local spacing, a tall card jitters
    across a proportionally wider pixel range than a short one — entropy that a fixed +-Npx wobble
    would not have."""
    short = [p.step_px for p in _draws(_stack((620, 620, 620)).segment(), n=200)]
    tall = [p.step_px for p in _draws(_stack((900, 900)).segment(), n=200)]
    assert (max(tall) - min(tall)) > (max(short) - min(short))


def _modal_share(steps):
    return max(steps.count(s) for s in set(steps)) / len(steps)


def test_no_single_step_distance_dominates_at_either_end_of_the_window():
    """THE REGRESSION. An earlier revision drew the step/spacing RATIO and then clamped the
    resulting distance into the legal gesture range, which turned every draw outside that range
    into the same delivered pixel count: a point mass at the gesture floor for short cards and one
    at `_MAX_STEP_PX` for tall ones. [measured, 5000 draws per spacing: 40% of gestures on ONE
    value at the corpus's smallest 738px spacing, 34% at 1166px, 95% at 620px.] A scroll distance
    that is one constant most of the time is exactly the fixed-constant signature the owner's
    randomization rule exists to forbid, so the draw is over the window's PIXELS instead.

    Both ends are covered here: `_short_frame` is bounded from below by the gesture floor and
    `_stack((1114, 1114))` from above by `_MAX_STEP_PX`."""
    for seg in (_short_frame().segment(), _stack((1114, 1114)).segment()):
        steps = [p.step_px for p in _draws(seg, n=2000)]
        window = _draws(seg, n=1)[0].window_px
        assert min(steps) >= window[0] and max(steps) <= window[1]
        assert _modal_share(steps) < 0.15, (window, _modal_share(steps))
        assert len(set(steps)) > (window[1] - window[0]) * 0.8


def test_the_draw_window_is_the_ratio_rule_applied_to_the_local_spacing():
    """The window is what keeps the distance content-following: both ends are fractions of the
    card in front of us, and the draw only decides where inside it to land — short of the bound
    itself, now that `_STEP_DELIVERY_JITTER_PX` (2026-08-27) reserves delivery headroom under
    it."""
    seg = _short_frame().segment()
    spacing = scroll_step.measure_local_spacing(seg).px
    plan = _plan_on(seg)
    cap_px = min(int(scroll_step._STEP_RATIO_MAX * spacing), scroll_step._MAX_STEP_PX)
    assert cap_px == plan.bound_px            # nothing else binds at this spacing
    low_px = max(scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H),
                int(scroll_step._STEP_RATIO_MIN * spacing))
    assert plan.window_px[0] == low_px
    margin = min(scroll_step._STEP_DELIVERY_JITTER_PX, (cap_px - low_px) // 2)
    assert plan.window_px[1] == cap_px - margin
    assert plan.window_px[1] < plan.bound_px, (
        "the draw must leave headroom under the bound it is judged against")


def test_a_spacing_that_leaves_exactly_one_legal_step_says_so_instead_of_pretending_to_draw():
    """The one place the jitter genuinely cannot exist, reported rather than hidden. Just above
    the ~608px refusal threshold the gesture floor (219px) meets the aliasing ceiling
    (0.36 x 609 = 219px), so every gesture on such a profile is the same distance. Nothing can fix
    that — a smaller gesture is outside the driver's sanctioned window and a larger one breaks the
    ratio rule — so what the plan owes the debug log is the fact.

    The control is 3px of extra spacing, which re-opens the window and drops the message."""
    forced = _stack((557, 557, 557)).segment()           # 557 + the 52px gutter floor = 609px
    assert scroll_step.measure_local_spacing(forced).px == 609
    plans = _draws(forced, n=50)
    assert {p.window_px for p in plans} == {(219, 219)}
    assert {p.step_px for p in plans} == {219}
    assert "no jitter left to draw" in plans[0].reason

    opened = _stack((560, 560, 560)).segment()           # 612px spacing
    assert scroll_step.measure_local_spacing(opened).px == 612
    wider = _draws(opened, n=50)
    assert {p.window_px for p in wider} == {(219, 220)}
    assert len({p.step_px for p in wider}) == 2
    assert "no jitter left" not in wider[0].reason


def test_a_spacing_taller_than_the_whole_ratio_window_does_not_collapse_onto_the_ceiling():
    """The other collapse the clamp produced, and the rarer one. Past ~1396px of card spacing even
    `_STEP_RATIO_MIN` of it exceeds `_MAX_STEP_PX`, so the old clamp delivered 363px on 87% of
    draws. The ceiling still wins — it is the below-the-fold bound — but the window's LOW end
    drops back to the gesture floor rather than onto it, and every value in it is smaller than the
    ratio rule would have allowed, which is the safe direction."""
    seg = _stack((1345, 1345)).segment()                 # 1397px spacing
    spacing = scroll_step.measure_local_spacing(seg).px
    assert int(scroll_step._STEP_RATIO_MIN * spacing) >= scroll_step._MAX_STEP_PX

    plans = _draws(seg, n=500)
    floor_px = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    cap_px = scroll_step._MAX_STEP_PX
    margin = min(scroll_step._STEP_DELIVERY_JITTER_PX, (cap_px - floor_px) // 2)
    assert {p.window_px for p in plans} == {(floor_px, cap_px - margin)}
    assert _modal_share([p.step_px for p in plans]) < 0.15
    assert all(p.step_px / spacing < scroll_step._STEP_RATIO_MAX for p in plans)


# =====================================================================================
# DELIVERY HEADROOM (`_STEP_DELIVERY_JITTER_PX`, 2026-08-27): the draw must stop short of the
# bound `step_overshoot` judges the delivered gesture against, never land exactly on it. See that
# constant's own comment for the two live `scroll_overshot` refusals this fixes, both of which had
# `planned_step_px == bound_px`.
# =====================================================================================

def test_the_live_refusal_geometry_never_draws_the_bound_itself():
    """The exact case `_STEP_DELIVERY_JITTER_PX`'s own comment cites: a profile whose memory has
    tightened the sizing spacing to 646px, giving `bound_px` 232 and gesture floor 219. Both real
    on-device `scroll_overshot` refusals had `planned_step_px == bound_px == 232` — this is the
    geometry that produced them, run for real through the seeded planner: no draw may reach 232,
    the window is exactly `219..229`, and jitter still exists (more than one distinct value)."""
    seg = _tall_frame().segment()                          # local spacing 1026px, well above 646
    plans = _draws(seg, n=500, profile_min_spacing_px=646)

    assert plans[0].bound_px == 232
    assert {p.window_px for p in plans} == {(219, 229)}
    assert max(p.step_px for p in plans) <= plans[0].bound_px - 3
    assert len({p.step_px for p in plans}) > 1, "the window must still have real jitter in it"


def test_the_commanded_step_keeps_the_full_headroom_for_every_drawable_target():
    """The headroom has to survive the frac ROUND-TRIP, not just the draw, and that is a
    separate guarantee from the one above.

    `target_px` is drawn inside the window, but what gets commanded is
    `step_px_for_frac(frac_for_step_px(target_px))`, and that round-trip can come back +1px
    [measured on the calibrated 2400px screen: +1 for 3 of the 230 targets in 150..379px, 0 or -1
    for the rest]. Clamping the walk-down loop at `cap_px` would therefore hand a whole pixel of
    the reservation straight back and leave only 2px of real headroom -- under the +3px
    over-delivery actually seen on device, i.e. precisely the case it was reserved for. So the
    loop clamps at `draw_cap_px`.

    Asserted over EVERY value the window can draw rather than over a sample, because the round-up
    happens for a handful of specific targets and a seeded sample can miss all of them -- which
    is exactly what makes this worth pinning separately.
    """
    seg = _tall_frame().segment()
    plan = _plan_on(seg, profile_min_spacing_px=646)
    low_px, draw_cap_px = plan.window_px
    assert (low_px, draw_cap_px, plan.bound_px) == (219, 229, 232)

    for target_px in range(low_px, draw_cap_px + 1):
        frac = scroll_step.frac_for_step_px(target_px, _H)
        commanded = scroll_step.step_px_for_frac(frac, _H)
        frac_lo = hinge._READ_SCROLL_FRAC_MIN
        while commanded > draw_cap_px and frac > frac_lo:       # the planner's own walk-down
            frac = max(frac_lo, frac - 1.0 / _H)
            commanded = scroll_step.step_px_for_frac(frac, _H)
        assert commanded <= draw_cap_px, target_px
        assert plan.bound_px - commanded >= scroll_step._STEP_DELIVERY_JITTER_PX, (
            f"target {target_px} commands {commanded}px, leaving less than the reserved "
            f"{scroll_step._STEP_DELIVERY_JITTER_PX}px under the {plan.bound_px}px bound")


def test_the_hard_bound_and_its_refusal_line_are_unchanged_by_the_headroom_reservation():
    """The headroom reservation narrows the DRAW window; it must never touch the bound
    `step_overshoot` judges the DELIVERED step against — the module's own "THIS DOES NOT RELAX
    THE BOUND" comment, checked rather than trusted. `bound_px` is still
    `int(ratio_hi * sized_against_px)`, and a delivered step exactly AT the bound is still not an
    overshoot; only one strictly past it is. The "we did not relax the guard" test."""
    seg = _tall_frame().segment()
    plan = _plan_on(seg, profile_min_spacing_px=646)

    assert plan.bound_px == int(scroll_step._STEP_RATIO_MAX * plan.sized_against_px)
    assert plan.bound_px == 232
    assert scroll_step.step_overshoot(plan, plan.bound_px) is None
    assert scroll_step.step_overshoot(plan, plan.bound_px + 1) is not None
    assert plan.window_px[1] < plan.bound_px, (
        "the draw window itself now stops short of the bound")


def test_a_window_too_narrow_for_full_headroom_still_keeps_at_least_two_drawable_values():
    """AT MOST HALF THE WINDOW: reserving delivery headroom must never collapse the draw down to
    one constant value, which is exactly the shape the owner's randomization rule forbids. At a
    615px spacing the window before headroom is only 2px wide (219..221), so the full 3px jitter
    margin does not fit; the planner must clip the margin rather than swallow the whole window,
    keep at least two distinct drawable values, and say so in `reason`."""
    seg = _stack((563, 563, 563)).segment()                # 563 + 52 = 615px spacing
    assert scroll_step.measure_local_spacing(seg).px == 615
    plan = _plan_on(seg)

    assert plan.bound_px == 221
    assert plan.window_px == (219, 220)
    assert plan.window_px[1] > plan.window_px[0], "must keep at least two distinct drawable values"
    assert plan.window_px[1] < plan.bound_px, "headroom is still reserved here, just clipped"
    assert "only 1px of the 3px delivery headroom fits" in plan.reason
    assert "over-deliver past the aliasing bound" in plan.reason

    # And the draw genuinely uses both values rather than landing on one by construction.
    steps = {p.step_px for p in _draws(seg, n=200)}
    assert len(steps) > 1, steps


def test_the_floor_meets_ceiling_degenerate_case_is_unchanged_by_the_headroom_reservation():
    """The one case headroom reservation must leave completely alone: when the gesture floor
    already meets the aliasing ceiling, `low_px` is reset to the floor and there is no window left
    to reserve anything out of. `cap_px - low_px == 0` here, so `jitter_margin_px` is 0 and
    `draw_cap_px == cap_px` — byte-for-byte the pre-change behaviour and its "no jitter left to
    draw" message, with no partial-headroom clause appended alongside it."""
    seg = _stack((557, 557, 557)).segment()                # 557 + 52 = 609px spacing
    assert scroll_step.measure_local_spacing(seg).px == 609
    plans = _draws(seg, n=50)

    assert {p.window_px for p in plans} == {(219, 219)}
    assert {p.step_px for p in plans} == {219}
    assert plans[0].bound_px == 219
    assert plans[0].window_px[1] == plans[0].bound_px, "no headroom exists to reserve here"
    assert "no jitter left to draw" in plans[0].reason
    assert "delivery headroom" not in plans[0].reason, (
        "the partial-headroom clause must not also fire alongside the zero-headroom one")


# =====================================================================================
# The bounds, and which one fails loud
# =====================================================================================

def test_the_ceiling_is_the_one_cadence_with_end_to_end_evidence():
    """`_MAX_STEP_PX` is not a chosen round number: it is exactly what read_scroll_frac 0.16
    delivers, the cadence that measured 23 of 23 pairs with zero tracking failures."""
    assert scroll_step._MAX_STEP_PX == scroll_step.step_px_for_frac(0.16, _H)


def test_the_ceiling_binds_before_a_very_tall_card_licenses_a_bigger_step():
    """`_MAX_STEP_PX` is what caps the window here, but the window's own top still sits
    `_STEP_DELIVERY_JITTER_PX` below it (2026-08-27) rather than at it — headroom reservation does
    not distinguish which ceiling (`bound_px` or `_MAX_STEP_PX`) produced `cap_px`."""
    seg = _stack((1114, 1114)).segment()                # doc 5.10's tallest card observed
    spacing = scroll_step.measure_local_spacing(seg).px
    assert scroll_step._STEP_RATIO_MAX * spacing > scroll_step._MAX_STEP_PX
    plans = _draws(seg)
    low_px = int(scroll_step._STEP_RATIO_MIN * spacing)
    cap_px = scroll_step._MAX_STEP_PX
    margin = min(scroll_step._STEP_DELIVERY_JITTER_PX, (cap_px - low_px) // 2)
    assert {p.window_px[1] for p in plans} == {cap_px - margin}
    assert max(p.step_px for p in plans) <= scroll_step._MAX_STEP_PX


def test_the_floor_is_the_drivers_own_sanctioned_read_scroll_window():
    """The enumeration scroll is an ordinary read-scroll and must stay inside the same distance
    window every other read-scroll is validated against — inventing a smaller gesture would be a
    new, unvalidated motion signature."""
    lo, hi = scroll_step._frac_window()
    assert (lo, hi) == (hinge._READ_SCROLL_FRAC_MIN, hinge._READ_SCROLL_FRAC_MAX)
    for seg in (_tall_frame().segment(), _short_frame().segment()):
        for plan in _draws(seg, n=100):
            assert lo <= plan.frac <= hi


def test_a_spacing_too_small_for_any_legal_gesture_raises_instead_of_stepping_further():
    """The one bound that fails loud. Below ~608px on this device even the smallest permitted
    read-scroll exceeds the ratio rule, and doc 5.10.1's whole finding is that a step past the
    ratio produces a confidently wrong index. No larger step is substituted."""
    seg = _stack((500, 500, 500)).segment()
    spacing = scroll_step.measure_local_spacing(seg).px
    floor = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)
    assert floor > scroll_step._STEP_RATIO_MAX * spacing
    with pytest.raises(scroll_step.ScrollStepError) as excinfo:
        scroll_step.plan_scroll_step(seg)
    message = str(excinfo.value)
    assert str(spacing) in message and str(floor) in message


def test_the_control_just_above_that_boundary_still_plans():
    """Paired with the refusal above so the test proves a BOUNDARY, not a blanket refusal of
    short cards."""
    plan = scroll_step.plan_scroll_step(_short_frame().segment())
    assert plan.basis == scroll_step.STEP_MEASURED
    assert plan.step_px >= scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)


# =====================================================================================
# The fallback: conservative, and never silent
# =====================================================================================

def test_an_unmeasurable_frame_falls_back_small_and_reports_it():
    frame = _Frame()
    frame.card(200, 2300, heart=True)
    plan = scroll_step.plan_scroll_step(frame.segment())
    assert plan.basis == scroll_step.STEP_FALLBACK
    assert plan.spacing_px is None and plan.ratio is None
    assert plan.sized_against_px == scroll_step._FALLBACK_SPACING_PX
    assert "no local spacing" in plan.reason


def test_the_fallback_is_sized_against_the_smallest_spacing_ever_measured():
    """"Conservative" has to mean something checkable: the blind step is the ratio rule applied
    to `_FALLBACK_SPACING_PX`, the smallest heart-bearing spacing in the whole corpus, so it
    cannot alias against any spacing the corpus has ever seen."""
    frame = _Frame()
    frame.card(200, 2300, heart=True)
    seg = frame.segment()
    for plan in _draws(seg):
        assert plan.step_px <= scroll_step._STEP_RATIO_MAX * scroll_step._FALLBACK_SPACING_PX
        assert plan.step_px < scroll_step._MAX_STEP_PX


def test_the_fallback_is_never_larger_than_a_measured_step_on_the_same_geometry():
    """"Never fall back to a large step", asserted as an ordering rather than a constant."""
    frame = _Frame()
    frame.card(200, 2300, heart=True)
    blind = [p.step_px for p in _draws(frame.segment())]
    measured = [p.step_px for p in _draws(_tall_frame().segment())]
    assert max(blind) < max(measured)


# =====================================================================================
# What refuses outright
# =====================================================================================

def test_a_segmentation_that_contradicts_itself_is_refused():
    """Two hearts in one block is segment.py's own hard failure. Sizing a scroll off a frame
    whose geometry does not hold together would be measuring a period that does not exist."""
    frame = _tall_frame()
    frame.heart(974 + 500 - _HEART_ABOVE_BOTTOM - 200)  # a second heart inside card one
    seg = frame.segment()
    assert not seg.ok
    with pytest.raises(scroll_step.ScrollStepError, match="contradicts itself"):
        scroll_step.plan_scroll_step(seg)


def test_one_explicit_segmentation_recovery_step_is_blind_and_marked():
    """The live capture loop may carry one bad frame forward, never size it from bad geometry."""
    frame = _tall_frame()
    frame.heart(974 + 500 - _HEART_ABOVE_BOTTOM - 200)
    seg = frame.segment()
    assert not seg.ok

    plan = scroll_step.plan_scroll_step(seg, rng=random.Random(3),
                                        allow_segmentation_failure_fallback=True)

    assert plan.basis == scroll_step.STEP_SEGMENTATION_FALLBACK
    assert plan.spacing_px is None and not plan.spacing.measured
    assert plan.sized_against_px == scroll_step._FALLBACK_SPACING_PX
    assert plan.step_px <= plan.bound_px
    assert "segmentation contradicted itself" in plan.reason


def test_a_zero_height_screen_raises():
    with pytest.raises(scroll_step.ScrollStepError):
        scroll_step.plan_scroll_step(_tall_frame().segment(), screen_height=0)


def test_an_inverted_ratio_window_raises_rather_than_silently_reordering():
    with pytest.raises(scroll_step.ScrollStepError, match="ratio window"):
        scroll_step.plan_scroll_step(_tall_frame().segment(), ratio_window=(0.4, 0.2))


def test_an_inverted_frac_window_raises():
    with pytest.raises(scroll_step.ScrollStepError, match="frac window"):
        scroll_step.plan_scroll_step(_tall_frame().segment(), frac_window=(0.5, 0.2))


def test_a_non_positive_profile_minimum_raises_rather_than_being_ignored():
    with pytest.raises(scroll_step.ScrollStepError, match="profile_min_spacing_px"):
        scroll_step.plan_scroll_step(_tall_frame().segment(), profile_min_spacing_px=0)


# =====================================================================================
# The loop's memory: a short card keeps binding after it leaves the screen
# =====================================================================================

def test_a_short_card_seen_earlier_keeps_the_step_small_on_a_later_tall_frame():
    """The only defence against a card that is still BELOW the fold. Without the memory the tall
    frame licenses close to the full ceiling; with it, the step stays inside the ratio rule for
    the short card that is still on the page even though it is no longer on the screen.

    "Close to" rather than exactly `_MAX_STEP_PX`: `_STEP_DELIVERY_JITTER_PX` (2026-08-27)
    reserves headroom under whichever ceiling binds, so the unaware plan's own draw window now
    tops out below `_MAX_STEP_PX`, never at it."""
    tall = _tall_frame().segment()
    short_spacing = scroll_step.measure_local_spacing(_short_frame().segment()).px

    unaware = _draws(tall)
    aware = _draws(tall, profile_min_spacing_px=short_spacing)

    unaware_cap = unaware[0].window_px[1]
    assert unaware_cap < scroll_step._MAX_STEP_PX, (
        "the top-of-window draw must leave delivery headroom")
    assert max(p.step_px for p in unaware) == unaware_cap
    assert max(p.step_px for p in aware) <= scroll_step._STEP_RATIO_MAX * short_spacing
    assert max(p.step_px for p in aware) < min(p.step_px for p in unaware)
    assert all(p.sized_against_px == short_spacing for p in aware)
    assert "tightened" in aware[0].reason


def test_the_memory_never_loosens_a_step():
    """It can only take the minimum, so a LARGER remembered spacing changes nothing — otherwise
    one tall card early in a profile could license a step the current frame forbids."""
    short = _short_frame().segment()
    tight = _draws(short)
    loose = _draws(short, profile_min_spacing_px=5000)
    assert [p.step_px for p in tight] == [p.step_px for p in loose]


def test_the_memory_also_binds_a_blind_frame():
    frame = _Frame()
    frame.card(200, 2300, heart=True)
    plan = scroll_step.plan_scroll_step(frame.segment(), profile_min_spacing_px=672)
    assert plan.basis == scroll_step.STEP_FALLBACK
    assert plan.sized_against_px == 672
    assert plan.step_px <= scroll_step._STEP_RATIO_MAX * 672


# =====================================================================================
# Closing the loop: holding the gesture that was actually made to the plan's own bound
# =====================================================================================

def _plan_on(seg, **kw):
    return scroll_step.plan_scroll_step(seg, rng=random.Random(3), **kw)


def test_a_gesture_inside_the_bound_is_not_an_overshoot():
    plan = _plan_on(_tall_frame().segment())
    assert scroll_step.step_overshoot(plan, plan.step_px) is None
    assert scroll_step.step_overshoot(plan, plan.bound_px) is None


def test_a_gesture_past_the_bound_is_reported_with_its_numbers():
    plan = _plan_on(_tall_frame().segment())
    reason = scroll_step.step_overshoot(plan, plan.bound_px + 1)
    assert reason is not None
    assert str(plan.bound_px) in reason and str(plan.sized_against_px) in reason


def test_a_profile_that_did_not_move_is_a_stalled_loop_not_a_safe_step():
    """Zero is inside every ratio bound and is still wrong: scrolling again into the same frame
    is how an enumeration pass spins instead of stopping."""
    plan = _plan_on(_tall_frame().segment())
    for achieved in (0, -12):
        reason = scroll_step.step_overshoot(plan, achieved)
        assert reason is not None and "did not move" in reason


def test_the_overshoot_bound_follows_a_callers_own_ratio_window():
    """`bound_px` is stored on the plan rather than recomputed from module constants, so a caller
    that narrowed the window is held to the window it chose."""
    seg = _tall_frame().segment()
    narrow = _plan_on(seg, ratio_window=(0.22, 0.24))
    wide = _plan_on(seg)
    assert narrow.bound_px < wide.bound_px
    assert scroll_step.step_overshoot(narrow, narrow.bound_px + 1) is not None
    assert scroll_step.step_overshoot(wide, narrow.bound_px + 1) is None


# =====================================================================================
# The call site: the plan is only ever spent through the humanized path
# =====================================================================================

class _RecordingAdb:
    """Records the read-scrolls that actually reached the transport."""

    def __init__(self):
        self.scrolls = []
        self.taps = []
        self.swipes = []

    def screen_size(self):
        return (_W, _H)

    def scroll_up(self, distance_frac=0.55, x_frac=0.5):
        self.scrolls.append((distance_frac, x_frac))

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, **_k):
        self.swipes.append((x1, y1, x2, y2))

    def shell(self, command="", **_k):
        return ""

    def screencap(self):
        return b""


def _drv(adb):
    class C:
        apps = {HINGE_SPEC.app: {"halt_on_error": False}}
    driver = AndroidDriver(C(), HINGE_SPEC)
    driver._adb = adb
    driver._touch = adb
    return driver


def test_the_planned_step_reaches_the_transport_through_scroll_down_one():
    """The gesture is the driver's existing humanized read-scroll — no new primitive, and the
    ledger `_scroll_to_top` counts against still gets its entry."""
    adb = _RecordingAdb()
    driver = _drv(adb)
    plan = _plan_on(_short_frame().segment())

    driver._scroll_down_one(plan.frac, plan.x_frac)

    assert adb.scrolls == [(plan.frac, plan.x_frac)]
    assert driver._capture_scroll_ledger == [(plan.frac, plan.x_frac)]
    assert driver._capture_scrolls == 1


def test_passing_the_frac_without_the_x_frac_silently_discards_it():
    """The trap the module docstring warns about, pinned so it cannot rot: `_scroll_down_one`
    re-samples BOTH arguments if EITHER is None, so `_scroll_down_one(step.frac)` issues
    production's 0.55 cadence — the exact cadence this module exists to avoid — with no error
    anywhere. A `ScrollStep` therefore carries `x_frac` as well as `frac`."""
    adb = _RecordingAdb()
    driver = _drv(adb)
    plan = _plan_on(_short_frame().segment())
    assert plan.frac != HINGE_SPEC.read_scroll_frac

    driver._scroll_down_one(plan.frac)

    assert adb.scrolls == [(HINGE_SPEC.read_scroll_frac, 0.5)]


def test_the_plans_lane_is_the_drivers_legacy_read_scroll_lane():
    """This module owns the DISTANCE and nothing else. The column is the driver's own default,
    and the transport jitters it by +-SCROLL_X_JITTER_PX on top, so consecutive enumeration
    scrolls are still not pixel-identical."""
    assert scroll_step._DEFAULT_X_FRAC == 0.5
    assert _plan_on(_tall_frame().segment()).x_frac == 0.5
    assert _plan_on(_tall_frame().segment(), x_frac=0.42).x_frac == 0.42


def test_a_lane_outside_the_drivers_own_window_is_refused_here_not_at_the_zone_guard():
    """`_sample_read_scroll` validates any policy-sampled lane against 0.10..0.90 before issuing
    a gesture; a plan that could not legally be spent is not a plan."""
    for bad in (0.05, 0.95):
        with pytest.raises(scroll_step.ScrollStepError, match="x_frac"):
            _plan_on(_tall_frame().segment(), x_frac=bad)


# =====================================================================================
# Provenance: constants shared rather than re-declared, and the leaf property
# =====================================================================================

def test_the_gutter_is_shared_with_segment_and_not_re_declared():
    """The pitch this module measures is a card's height plus the gutter that SEPARATED it, so a
    second copy of that window would be free to drift from the one that cut the block."""
    assert scroll_step._GUTTER_FLOOR_PX == min(segment._GUTTER_PX)


def test_importing_this_module_does_not_pull_in_the_driver():
    """A leaf, like segment.py / frameshift.py / item_index.py: importable with no device and no
    driver. The read-scroll window is fetched at CALL time (the same deferred-import shape
    segment_frame uses for `_match_glyph`) so there is one source of truth without a cycle."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import operation_love.drivers.scroll_step as m; "
         "print('operation_love.drivers.hinge' in sys.modules); "
         "print(m.plan_scroll_step.__name__)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "plan_scroll_step"], out.stdout + out.stderr


def test_production_read_scroll_frac_is_untouched_by_this_module():
    """doc: production's 0.55 stays correct for the swipe-deck read path, which indexes nothing.
    The enumeration pass computes its own step and never consults it."""
    assert HINGE_SPEC.read_scroll_frac == 0.55
    for plan in _draws(_tall_frame().segment(), n=50):
        assert plan.frac < HINGE_SPEC.read_scroll_frac


def test_the_fallback_spacing_is_the_corpus_minimum_and_the_ratio_window_is_about_a_third():
    """Pinning the two numbers whose provenance is a measurement rather than an argument, so a
    later "tuning" pass has to come back to this file and read why."""
    assert scroll_step._FALLBACK_SPACING_PX == 738
    assert scroll_step._STEP_RATIO_MIN < 1 / 3 < scroll_step._STEP_RATIO_MAX
    assert scroll_step._STEP_RATIO_MAX <= 0.36


# =====================================================================================
# COVERAGE-AIMED STEP (plan_coverage_step, 2026-08-24) — the enumeration read's REPLACEMENT
# rule. `plan_scroll_step` above is untouched and is what `item_nav.py`'s counting-navigation
# climb still calls; every test below exercises the new, separate function only.
#
# Ground truth for the two hard bounds, on this file's calibrated 1080x2400 device
# (`_BAND0, _BAND1 = 300, 2100`, band height 1800):
#   trust ceiling  = round(0.30 * 1800) = 540   (`_ENUM_TRUST_CEILING_BAND_FRAC`)
#   blind cap      = round((0.5 - 0.30) * 1800) = 360   (the DIRECT-BRIDGE BUDGET: what
#                    frameshift's `_TRUST_WINDOW_BAND_FRAC` window has left once the enumeration
#                    ceiling is spent, so a frame-omission recovery can still bridge a blind step.
#                    Replaced `1800 - _MAX_CARD_HEIGHT_PX = 333` on 2026-08-28 -- that constant
#                    was both wrong by 142px and anti-correlated with its own trigger; the module
#                    carries the full argument at its deletion site.)
#   gesture floor   = 219                        (`step_px_for_frac(_READ_SCROLL_FRAC_MIN, _H)`)
# =====================================================================================

_BAND_H = _BAND1 - _BAND0                                                          # 1800
_TRUST_CEILING_PX = round(scroll_step._ENUM_TRUST_CEILING_BAND_FRAC * _BAND_H)     # 540
_BLIND_CAP_PX = round((frameshift._TRUST_WINDOW_BAND_FRAC
                       - scroll_step._ENUM_TRUST_CEILING_BAND_FRAC) * _BAND_H)   # 360
_GESTURE_FLOOR_PX = scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H)  # 219
_SEGMENTATION_FALLBACK_CAP_PX = round(
    ((scroll_step._STEP_RATIO_MIN + scroll_step._STEP_RATIO_MAX) / 2)
    * scroll_step._FALLBACK_SPACING_PX)                                             # 229


def _coverage_draws(seg, n=200, **kw):
    rng = random.Random(20260824)
    return [scroll_step.plan_coverage_step(seg, rng=rng, **kw) for _ in range(n)]


def test_the_two_bounds_match_the_derivation():
    """Pins the two numbers this whole rule is built from, so a later "tuning" pass has to come
    back and read why, on `test_the_fallback_spacing_is_the_corpus_minimum...`'s own precedent."""
    assert _TRUST_CEILING_PX == 540
    assert _BLIND_CAP_PX == 360
    # THE BLIND CAP'S REASON FOR EXISTING, pinned so a "simplification" has to answer it. The cap
    # is what lets `item_index`'s frame-omission recovery drop one unusable frame and re-measure a
    # single bridge from i-1 to i+1: that bridge spans the previous throttled step PLUS this blind
    # one, so the two together must fit inside frameshift's trust window. Capping the blind branch
    # at the bare trust ceiling instead would give 540 + 540 = 1080 against a 900px window and make
    # the recovery unreachable.
    assert _BLIND_CAP_PX + _TRUST_CEILING_PX <= round(frameshift._TRUST_WINDOW_BAND_FRAC * _BAND_H)
    assert _TRUST_CEILING_PX + _TRUST_CEILING_PX > round(frameshift._TRUST_WINDOW_BAND_FRAC * _BAND_H)
    # And it stays above the driver's own smallest sanctioned gesture, which a card-height margin
    # did not: the deleted `band_height - 1609` would have been 191px against this 219px floor.
    assert _BLIND_CAP_PX > _GESTURE_FLOOR_PX
    assert _TRUST_CEILING_PX < round(0.5 * _BAND_H)          # meaningfully under frameshift's 900
    # A contradictory frame can be carried four times and then omitted as one run, so it must
    # use the corpus-minimum recovery cadence rather than this ordinary one-bad-frame budget.
    assert _SEGMENTATION_FALLBACK_CAP_PX == 229
    assert _GESTURE_FLOOR_PX < _SEGMENTATION_FALLBACK_CAP_PX < _BLIND_CAP_PX


def test_an_open_frame_with_no_trailing_card_is_capped_at_the_trust_ceiling():
    """A single short card, fully bounded well inside the band, with page background below it:
    nothing here constrains the next step, so it is capped only by bound 1."""
    seg = _stack((600,), top=500).segment()
    assert scroll_step._open_trailing_block_depth(seg) is None
    for plan in _coverage_draws(seg, n=50):
        assert plan.basis == scroll_step.COVERAGE_STEP_OPEN
        assert plan.depth_px is None
        assert plan.cap_px == _TRUST_CEILING_PX
        assert _GESTURE_FLOOR_PX <= plan.step_px <= _TRUST_CEILING_PX


def test_an_open_card_below_the_trust_ceiling_throttles_the_step_to_its_own_depth():
    """A short first card puts the second (incomplete, band-cut) card's own top row 500px
    below the band's own top row — inside the 540px trust ceiling, so bound 2 actually binds
    this step."""
    seg = _stack((247, 2000), top=500).segment()
    depth = scroll_step._open_trailing_block_depth(seg)
    assert depth == 500
    for plan in _coverage_draws(seg, n=50):
        assert plan.basis == scroll_step.COVERAGE_STEP_THROTTLED
        assert plan.depth_px == depth
        assert plan.cap_px == depth
        assert plan.step_px <= depth < _TRUST_CEILING_PX


def test_a_throttled_step_never_spends_the_open_cards_whole_depth():
    """The delivery headroom, on the branch where the cap has NO slack of its own.

    Found 2026-09-04: `plan_scroll_step` reserves `_STEP_DELIVERY_JITTER_PX` under its bound and
    `plan_coverage_step` did not, so a throttled draw could command exactly `depth` — and the
    module's own bound-2 proof spends every pixel of that, so the +1..+3px over-delivery measured
    on device puts the open card's top row ABOVE the new band and it can never be observed
    complete. This asserts the reservation over the WHOLE window rather than a lucky draw: the
    top of the draw window is `depth - 3`, and enough draws land on it that a planner without the
    reservation fails here rather than merely getting unlucky.

    Depth 500 against a 219px floor is a wide window, so the FULL headroom fits and there is no
    shortfall note; the narrow-window case is `..._when_the_ceiling_is_tight` above.
    """
    seg = _stack((247, 2000), top=500).segment()
    depth = scroll_step._open_trailing_block_depth(seg)
    assert depth == 500
    plans = _coverage_draws(seg, n=400)
    assert {p.window_px[1] for p in plans} == {depth - scroll_step._STEP_DELIVERY_JITTER_PX}
    for plan in plans:
        assert plan.cap_px == depth, "the hard bound itself must not have been relaxed"
        assert plan.step_px <= depth - scroll_step._STEP_DELIVERY_JITTER_PX, plan.reason
        assert "over-deliver" not in plan.reason
    # ...and the window is genuinely exercised up to its new top, so this is the reservation
    # doing the work and not the draw never getting there.
    assert max(p.step_px for p in plans) > depth - 2 * scroll_step._STEP_DELIVERY_JITTER_PX


def test_a_coverage_window_too_narrow_to_reserve_the_headroom_says_so_rather_than_collapsing():
    """The owner randomization rule wins over the headroom on a window that cannot afford both.

    A depth only a few pixels above the gesture floor cannot give up 3px without the draw
    collapsing onto one distance, so at most HALF the window is reserved and the shortfall is
    reported instead of silently accepted — the same trade `plan_scroll_step` makes, in the same
    words.
    """
    # depth = top + 453 - _BAND0 (the sibling fixture's arithmetic), so top=69 puts the open
    # card's own top row 222px down: 3px above the 219px gesture floor, a window too narrow to
    # give up the full 3px and still be a draw.
    seg = _stack((400, 2000), top=69).segment()
    depth = scroll_step._open_trailing_block_depth(seg)
    assert depth == 222, "the fixture must sit just above the gesture floor"
    plans = _coverage_draws(seg, n=100)
    reserved = (depth - _GESTURE_FLOOR_PX) // 2
    assert 0 < reserved < scroll_step._STEP_DELIVERY_JITTER_PX
    for plan in plans:
        assert plan.window_px == (_GESTURE_FLOOR_PX, depth - reserved)
        assert plan.step_px <= depth - reserved
        assert f"only {reserved}px of the" in plan.reason and "over-deliver" in plan.reason
    assert len({p.step_px for p in plans}) > 1, "the window must keep some jitter"


def test_an_open_card_beyond_the_trust_ceiling_is_capped_at_bound_one_not_its_own_depth():
    """The mirror of the previous test: `_tall_frame`'s second card sits 1227px below the band's
    own top row — past the trust ceiling — so bound 1 binds instead, exactly as the module
    docstring's case analysis says it must (the throttle can only ever narrow a step, never
    widen it past bound 1)."""
    seg = _tall_frame().segment()
    depth = scroll_step._open_trailing_block_depth(seg)
    assert depth == 1227 > _TRUST_CEILING_PX
    for plan in _coverage_draws(seg, n=50):
        assert plan.basis == scroll_step.COVERAGE_STEP_THROTTLED
        assert plan.depth_px == depth
        assert plan.cap_px == _TRUST_CEILING_PX
        assert plan.step_px <= _TRUST_CEILING_PX


def _reseg_after_scroll(card1_h, card2_h, *, top, cumulative_px):
    """Re-cut `_stack((card1_h, card2_h), top=...)` as it would appear after the page has
    already scrolled forward `cumulative_px`: every card's frame-local row is `top - moved`,
    exactly the rigid translation a real forward scroll applies. `_stack`/`_Frame.card` already
    clip a card's rows to `[0, _H)`, so a first card scrolled mostly off the top degrades the way
    a real one does (fewer of its own rows visible) rather than raising -- this fixture only
    needs its SECOND card to stay meaningful, which is what every assertion below reads.
    """
    return _stack((card1_h, card2_h), top=top - cumulative_px).segment()


def test_the_adaptive_throttle_never_lets_a_step_scroll_past_an_open_cards_own_top():
    """The module docstring's proof, run for real: repeatedly re-plan and re-segment the SAME
    open card as the simulated band top advances, and confirm two things every step: the band top
    never advances past the card's own top row (`depth_px` never goes negative), and the card
    does eventually complete. A generous step ceiling on the loop itself turns a hang into a
    reported failure instead.

    Starts the open (second) card 1653px below the band's own top — past the 540px trust
    ceiling, so the first steps are bound-1-limited exactly as
    `test_an_open_card_beyond_the_trust_ceiling_is_capped_at_bound_one_not_its_own_depth` proved
    in isolation — and gives it a 1750px height, taller than the 1467px worst case the blind
    fallback assumes (but, necessarily, still under the 1800px band itself: nothing can ever
    complete a card taller than the band regardless of step size, which is a structural
    impossibility rather than a step-sizing question), specifically to show the adaptive path
    does not need the 1467px figure to be correct — see the module docstring: it is sound for
    any height under the full band.
    """
    card1_h, card2_h, top = 1400, 1750, 500
    rng = random.Random(20260824)
    cumulative_px = 0
    depths: list[int] = []
    for _ in range(20):
        seg = _reseg_after_scroll(card1_h, card2_h, top=top, cumulative_px=cumulative_px)
        depth = scroll_step._open_trailing_block_depth(seg)
        if depth is None:
            assert seg.blocks[-1].complete, "the only way depth is None is a resolved trailing card"
            break
        assert depth >= 0, "the simulated band top has scrolled past the open card's own top row"
        depths.append(depth)
        plan = scroll_step.plan_coverage_step(seg, rng=rng)
        assert plan.step_px <= depth, "one step must not exceed what the proof allows"
        cumulative_px += plan.step_px
    else:
        pytest.fail(f"the card never completed in 20 steps; depths were {depths}")

    # The proof's own shape: depth is non-increasing (this design never backs off), and it does
    # shrink — a throttle that let depth stall while the card stayed incomplete would be the bug
    # this whole rule exists to prevent.
    assert depths == sorted(depths, reverse=True)
    assert depths[0] > depths[-1]


def test_a_frame_with_no_blocks_at_all_falls_back_to_the_blind_coverage_margin():
    """A blank band (no card drawn) offers no trailing block to read a depth off — the coverage
    rule's analogue of `plan_scroll_step`'s STEP_FALLBACK, and it is bounded by the SAME derived
    worst-case margin (333px) rather than the trust ceiling, since there is no local evidence to
    trust further than that."""
    seg = _Frame().segment()
    assert seg.ok and not seg.blocks
    for plan in _coverage_draws(seg, n=50):
        assert plan.basis == scroll_step.COVERAGE_STEP_FALLBACK
        assert plan.depth_px is None
        assert plan.cap_px == _BLIND_CAP_PX
        assert plan.step_px <= _BLIND_CAP_PX


def test_a_self_contradictory_frame_is_refused_without_the_opt_in():
    """Mirrors `plan_scroll_step`'s own refusal: a frame whose OWN segmentation contradicts
    itself (two hearts in one block) is not something this rule will size a blind step from
    unless the caller explicitly opts in, on `MAX_SEGMENTATION_FALLBACK_FRAMES`'s bounded-run
    precedent."""
    frame = _Frame().card(500, 1474, heart=True)
    frame.heart(1474 - 300)                    # a second heart inside the same card
    seg = frame.segment()
    assert not seg.ok
    with pytest.raises(scroll_step.ScrollStepError):
        scroll_step.plan_coverage_step(seg)


def test_the_opted_in_segmentation_fallback_is_recovery_conservative_and_jittered():
    """A contradictory run gets its own small, non-fixed cadence; an ordinary blind frame keeps
    the 360px one-bridge budget.  Coupling both assertions prevents another accidental merge of
    their two different recovery proofs."""
    frame = _Frame().card(500, 1474, heart=True)
    frame.heart(1474 - 300)
    seg = frame.segment()
    plans = _coverage_draws(seg, n=100, allow_segmentation_failure_fallback=True)
    assert {plan.basis for plan in plans} == {
        scroll_step.COVERAGE_STEP_SEGMENTATION_FALLBACK}
    assert {plan.cap_px for plan in plans} == {_SEGMENTATION_FALLBACK_CAP_PX}
    steps = {plan.step_px for plan in plans}
    assert min(steps) >= _GESTURE_FLOOR_PX
    assert max(steps) <= _SEGMENTATION_FALLBACK_CAP_PX
    assert len(steps) > 1, "the recovery cadence must retain a randomized legal window"
    assert all("contradicted itself" in plan.reason
               and "corpus-minimum recovery cadence" in plan.reason for plan in plans)

    ordinary = _Frame().segment()
    ordinary_plans = _coverage_draws(ordinary, n=20)
    assert {plan.basis for plan in ordinary_plans} == {scroll_step.COVERAGE_STEP_FALLBACK}
    assert {plan.cap_px for plan in ordinary_plans} == {_BLIND_CAP_PX}
    assert max(plan.step_px for plan in ordinary_plans) > _SEGMENTATION_FALLBACK_CAP_PX


def test_a_block_with_neither_edge_observed_requires_the_bounded_recovery_opt_in():
    """Regression for the Laura frame-5 refusal.

    A missed white-on-white gutter can make two real cards look like one plausible partial block
    spanning the whole analysed band. That is not a real depth of zero: it is a frame with no
    usable local edge. Without the capture loop's bounded-recovery opt-in it still refuses.
    """
    frame = _Frame()
    frame.card(100, 1425, heart=False, radius=0)
    frame.card(1490, 2300, heart=False, radius=0)
    seg = frame.segment()
    last = seg.blocks[-1]
    assert seg.ok and len(seg.blocks) == 1
    assert [(run.y0, run.y1, run.kind) for run in seg.runs] == [
        (1425, 1490, segment.RUN_TOO_LONG)]
    assert (last.y0, last.y1) == seg.band
    assert not last.top.observed and not last.bottom.observed
    with pytest.raises(scroll_step.ScrollStepError, match="neither edge observed"):
        scroll_step.plan_coverage_step(seg)


def test_a_block_with_neither_edge_observed_uses_the_omission_recovery_cadence():
    """The same incident shape advances by one small, marked step so a clean next frame can let
    item_index independently reconcile or omit/rebuild the bad observation. It never falls
    through to the 0.55 read cadence, and four consecutive fallback frames remain the driver's
    existing hard bound."""
    frame = _Frame()
    frame.card(100, 1425, heart=False, radius=0)
    frame.card(1490, 2300, heart=False, radius=0)
    seg = frame.segment()
    plans = _coverage_draws(
        seg, n=50, allow_segmentation_failure_fallback=True)
    assert {plan.basis for plan in plans} == {
        scroll_step.COVERAGE_STEP_SEGMENTATION_FALLBACK}
    assert {plan.cap_px for plan in plans} == {_SEGMENTATION_FALLBACK_CAP_PX}
    assert min(plan.step_px for plan in plans) >= _GESTURE_FLOOR_PX
    assert max(plan.step_px for plan in plans) <= _SEGMENTATION_FALLBACK_CAP_PX
    assert all(plan.depth_px is None for plan in plans)
    assert all("neither edge observed" in plan.reason
               and "omitted/rebuilt" in plan.reason for plan in plans)


def test_a_rounded_context_bottom_prevents_a_false_shallow_open_card_refusal():
    """Laura frame 5 merged a details card and the pale photo below across a 65px gap.

    Once segmentation uses the details card's independently measured bottom corner to split the
    regions, the lower region honestly has neither edge observed. The existing bounded recovery
    cadence can carry that frame; the planner must not mistake the details card's 29px top depth
    for the open photo's depth and hard-refuse the profile.
    """
    frame = _Frame()
    frame.card(329, 1473, heart=False)
    frame.card(1538, 2300, heart=False, radius=0)
    seg = frame.segment()

    assert [(run.y0, run.y1, run.kind) for run in seg.runs] == [
        (300, 329, segment.RUN_CLIPPED),
        (1473, 1538, segment.RUN_CARD_EDGE),
    ]
    assert [(block.y0, block.y1) for block in seg.blocks] == [
        (329, 1473), (1538, _BAND1)]
    trailing = seg.blocks[-1]
    assert not trailing.top.observed and not trailing.bottom.observed
    plans = _coverage_draws(
        seg, n=50, allow_segmentation_failure_fallback=True)
    assert {plan.basis for plan in plans} == {
        scroll_step.COVERAGE_STEP_SEGMENTATION_FALLBACK}
    assert min(plan.step_px for plan in plans) >= _GESTURE_FLOOR_PX
    assert max(plan.step_px for plan in plans) <= _SEGMENTATION_FALLBACK_CAP_PX


def test_a_top_clipped_block_with_an_observed_bottom_does_not_throttle_the_next_step():
    """A card leaving through the viewport's top is historical coverage, not an open trailing
    card. Once its bottom is observed, moving forward cannot make that observation less complete;
    the planner should size the next step for the content below instead of inventing depth zero."""
    seg = _stack((900,), top=100).segment()
    last = seg.blocks[-1]
    assert not last.top.observed and last.bottom.observed
    assert scroll_step._open_trailing_block_depth(seg) is None
    plans = _coverage_draws(seg, n=20)
    assert {plan.basis for plan in plans} == {scroll_step.COVERAGE_STEP_OPEN}


def test_a_trust_ceiling_that_swallows_the_whole_window_leaves_no_bridgeable_blind_step():
    """The boundary of the blind cap's derivation, replacing the deleted card-height one.

    REPLACED 2026-08-28. The old test drove `max_card_height_px=_BAND_H` and asserted the
    `band_height - max_card_height_px` margin refused. That parameter and that margin are gone:
    the margin guarded the blind branch against a card that, being taller than the band, would
    have produced no blocks only by exceeding it -- for which the margin is negative and its own
    premise void. What bounds the branch now is whether a frame-omission recovery could still
    bridge across the step, so the boundary to pin is the one where frameshift's window has
    nothing left after the enumeration ceiling.
    """
    seg = _stack((600,), top=500).segment()
    with pytest.raises(scroll_step.ScrollStepError, match="frame-omission recovery could still"):
        scroll_step.plan_coverage_step(
            seg, trust_ceiling_band_frac=frameshift._TRUST_WINDOW_BAND_FRAC)


def test_the_step_is_hazard_drawn_not_a_fixed_content_locked_distance():
    """The owner rule against a fixed constant, checked the same way
    `test_the_step_is_jittered_rather_than_a_fixed_content_locked_distance` checks it for the
    other rule: many draws on the identical frame must not collapse onto one repeated value."""
    seg = _stack((600,), top=500).segment()
    plans = _coverage_draws(seg, n=200)
    distinct = {plan.step_px for plan in plans}
    assert len(distinct) > 20, "too many draws are landing on the same few pixel values"
    most_common = max({v: sum(1 for p in plans if p.step_px == v) for v in distinct}.values())
    assert most_common / len(plans) < 0.10


def test_the_jitter_window_narrows_from_the_gesture_floor_when_the_ceiling_is_tight():
    """`window_low_frac` narrows the draw window's low end toward the ceiling as usual, but the
    gesture floor always wins over it — the same override `plan_scroll_step`'s own ratio-window
    low end respects, on the same device-imposed reasoning."""
    # A depth just above the gesture floor: 0.75 * 250 = 187.5 < 219, so the floor wins and the
    # window closes onto a narrow band anchored at the floor rather than at 0.75 * cap. (The
    # second card's frame-local top is `top + 400 + _GUTTER`, so depth = top + 453 - _BAND0;
    # top=97 is what puts it at exactly 250.)
    seg = _stack((400, 2000), top=97).segment()
    depth = scroll_step._open_trailing_block_depth(seg)
    assert depth == 250
    plan = scroll_step.plan_coverage_step(seg, rng=random.Random(2))
    assert plan.window_px[0] == _GESTURE_FLOOR_PX
    # The window's high end is the cap MINUS the delivery headroom, not the cap: `cap_px` is the
    # hard bound and `window_px` is what was actually drawn from, so the entropy of the draw
    # stays legible from the result (`CoverageStep.window_px`).
    assert plan.cap_px == depth
    assert plan.window_px[1] == depth - scroll_step._STEP_DELIVERY_JITTER_PX


def test_importing_the_coverage_rule_does_not_pull_in_the_driver():
    """The same leaf property `test_importing_this_module_does_not_pull_in_the_driver` checks for
    `plan_scroll_step`, run against the new function -- both share this module's deferred
    `_frac_window` import, so neither should ever import hinge.py at module load time."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import operation_love.drivers.scroll_step as m; "
         "print('operation_love.drivers.hinge' in sys.modules); "
         "print(m.plan_coverage_step.__name__)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "plan_coverage_step"], out.stdout + out.stderr


# =====================================================================================
# THE HONEST SCROLLING VIEWPORT (`_scrolling_viewport_top`, 2026-08-28)
#
# Added the day Hinge 10.1.0 was found to pin a profile header INSIDE the analysed band, and
# immediately load-bearing for `_open_trailing_block_depth` — yet it shipped with NO test in this
# module, which an adversarial review caught. That is exactly the shape of this repo's standing
# "fixtures can miss the branch they name" rule, so these tests are written mutation-first.
# =====================================================================================

def _pinned_header_frame(strip=(368, 411), gap=106, card_h=974):
    """A frame shaped like a real 10.1.0 mid-scroll one: the band opens on page background, then a
    short screen-pinned strip narrower than the full card width, then `gap` more rows of page
    background, then a corner-less card clipped by that run.

    The strip is drawn with `radius=0` and deliberately inset from the card's own x-extent so it
    cannot pass segment.py's full-width test — that is the property that makes it an unplaceable
    leading island rather than a card slice.
    """
    f = _Frame()
    s0, s1 = strip
    inset = 40                                   # keeps the strip's span under the card width
    f.gray[s0:s1, _CARD_X0 + inset:_CARD_X1 - inset] = f.rng.integers(
        60, 200, size=(s1 - s0, (_CARD_X1 - inset) - (_CARD_X0 + inset)), dtype=np.uint8)
    content_top = s1 + gap
    f.card(content_top, content_top + card_h, radius=0)
    return f, content_top


def test_the_scrolling_viewport_starts_below_a_screen_pinned_strip_not_at_the_band_top():
    """The measurand: with pinned chrome covering the top of the band, the first row the PAGE can
    occupy is below the strip and the page background under it — not the band's own first row.

    On the 2026-08-28 incident capture the two differed by 217px (band row 300 against a real
    content top of 517), and measuring depth from the band top is 217px of licence to take a
    bigger step than the open card can survive. That is why its 1109px card was never observed
    end to end: the honest completeness window was 474 rows, not the 691 the band claimed.
    """
    f, content_top = _pinned_header_frame()
    seg = f.segment()
    assert seg.blocks and seg.blocks[0].kind == segment.BLOCK_UNANCHORED, \
        "the fixture must actually produce an unplaceable leading strip"
    assert scroll_step._scrolling_viewport_top(seg) == content_top
    assert scroll_step._scrolling_viewport_top(seg) > seg.band[0], \
        "a viewport equal to the band top is the bug this function exists to fix"


def test_an_ordinary_frame_keeps_the_band_top_as_its_viewport():
    """The control. With no pinned strip the two are the same number, so this function can only
    ever move the answer on frames that actually carry one."""
    seg = _stack((600, 700), top=500).segment()
    assert not any(b.kind == segment.BLOCK_UNANCHORED for b in seg.blocks)
    assert scroll_step._scrolling_viewport_top(seg) == seg.band[0]


def test_the_open_card_depth_is_measured_from_the_viewport_not_the_band():
    """The consequence, and the thing a mutation must break: `_open_trailing_block_depth` prices
    the next step against the honest viewport. Reverting it to the band top inflates every depth
    on a pinned-header frame by exactly the strip's own reach — which is 217px of licence to take
    a bigger step than the open card can survive.

    FIXTURE NOTE, written after this test's first draft silently passed under the mutation. The
    trailing card must have an OBSERVED top (a real gutter above it), because
    `_open_trailing_block_depth` returns 0 outright for an unobserved one and 0 is the same number
    whichever viewport you measure from. The first draft put the open card directly under the
    strip, where its top edge is the island cut and therefore unobserved, so the assertions were
    comparing 0 to 0 and the mutation could not redden them.
    """
    f, content_top = _pinned_header_frame(card_h=700)            # a bounded card under the strip
    open_top = content_top + 700 + _GUTTER                       # then a gutter, then the open one
    f.card(open_top, open_top + 2000, radius=_CORNER_RADIUS_PX)  # runs off the band: stays open
    seg = f.segment()
    last = seg.blocks[-1]
    assert last.top.observed and not last.complete, \
        "the open trailing card must be gutter-bounded above, or depth short-circuits to 0"
    depth = scroll_step._open_trailing_block_depth(seg)
    assert depth == max(0, last.y0 - content_top)                # honest
    assert depth < max(0, last.y0 - seg.band[0])                 # what the pre-fix code returned
    assert max(0, last.y0 - seg.band[0]) - depth == content_top - seg.band[0]
