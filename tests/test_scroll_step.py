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

from operation_love.drivers import hinge, scroll_step, segment
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
    1026px spacing spends the whole budget the corpus validated.

    Asserted on the draw WINDOW rather than on the largest of 200 draws: the ceiling is the top of
    a uniform window now, not a clamp every over-large draw lands on, so "the biggest draw was
    exactly 363" would be a statement about the seed."""
    plans = _draws(_tall_frame().segment())
    spacing = scroll_step.measure_local_spacing(_tall_frame().segment()).px
    assert {p.window_px for p in plans} == {
        (int(scroll_step._STEP_RATIO_MIN * spacing), scroll_step._MAX_STEP_PX)}
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
    card in front of us, and the draw only decides where inside it to land."""
    seg = _short_frame().segment()
    spacing = scroll_step.measure_local_spacing(seg).px
    plan = _plan_on(seg)
    assert plan.window_px[1] == min(int(scroll_step._STEP_RATIO_MAX * spacing),
                                    scroll_step._MAX_STEP_PX)
    assert plan.window_px[1] == plan.bound_px            # nothing else binds at this spacing
    assert plan.window_px[0] == max(scroll_step.step_px_for_frac(hinge._READ_SCROLL_FRAC_MIN, _H),
                                    int(scroll_step._STEP_RATIO_MIN * spacing))


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
    assert {p.window_px for p in plans} == {(floor_px, scroll_step._MAX_STEP_PX)}
    assert _modal_share([p.step_px for p in plans]) < 0.15
    assert all(p.step_px / spacing < scroll_step._STEP_RATIO_MAX for p in plans)


# =====================================================================================
# The bounds, and which one fails loud
# =====================================================================================

def test_the_ceiling_is_the_one_cadence_with_end_to_end_evidence():
    """`_MAX_STEP_PX` is not a chosen round number: it is exactly what read_scroll_frac 0.16
    delivers, the cadence that measured 23 of 23 pairs with zero tracking failures."""
    assert scroll_step._MAX_STEP_PX == scroll_step.step_px_for_frac(0.16, _H)


def test_the_ceiling_binds_before_a_very_tall_card_licenses_a_bigger_step():
    seg = _stack((1114, 1114)).segment()                # doc 5.10's tallest card observed
    spacing = scroll_step.measure_local_spacing(seg).px
    assert scroll_step._STEP_RATIO_MAX * spacing > scroll_step._MAX_STEP_PX
    plans = _draws(seg)
    assert {p.window_px[1] for p in plans} == {scroll_step._MAX_STEP_PX}
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
    frame licenses the full ceiling; with it, the step stays inside the ratio rule for the short
    card that is still on the page even though it is no longer on the screen."""
    tall = _tall_frame().segment()
    short_spacing = scroll_step.measure_local_spacing(_short_frame().segment()).px

    unaware = _draws(tall)
    aware = _draws(tall, profile_min_spacing_px=short_spacing)

    assert max(p.step_px for p in unaware) == scroll_step._MAX_STEP_PX
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
    assert scroll_step._TOUCH_SLOP_PX == 21
