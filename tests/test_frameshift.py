"""Cross-frame vertical shift estimation (operation_love/drivers/frameshift.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures the module's
constants were measured against are real people's dating profiles and are gitignored
(ops/calibration/, .gitignore:26); only geometry and counts from them appear anywhere in this
repo. So the fixtures build a tall scrollable WORLD from first principles — a gradient page
background, rounded-rect cards inset 53px per side, 53px gutters, one blank-white prompt card in
every five — and cut two 1080x2400 windows out of it at offsets chosen by the test. The shift is
therefore known BY CONSTRUCTION, exactly, and the assertions can be `== 363` rather than
`approx(363)`.

Two fixture details are load-bearing enough to state up front:

  * static CHROME is painted over the rows outside `content_band` on every frame, identical each
    time. That is what the real device does (status bar, sticky header, floating buttons, bottom
    nav) and it is the reason the module takes a band at all; `test_static_chrome_...` uses it as
    the control.
  * the world is textured with a seeded RNG. A flat fill would give TM_CCOEFF_NORMED spurious
    perfect scores at every offset — the same trap tests/test_hinge_vision.py and
    tests/test_segment.py call out — and would make a correct answer indistinguishable from
    luck.

Following the house pattern, every positive is paired with a negative plus a control that proves
WHICH mechanism did the excluding.
"""
import math
import dataclasses
import subprocess
import sys

import cv2
import numpy as np
import pytest

from operation_love.drivers import frameshift, segment
from operation_love.drivers.hinge import HINGE_SPEC

_W, _H = 1080, 2400                       # the calibrated Pixel 7a screencap size
_WORLD_H = 6000                           # tall enough for a 2400px window plus any tested shift
_CONTENT_BAND = HINGE_SPEC.content_band   # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - segment._CARD_MARGIN_PX                # 1027
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px"
_SEED = 11

# The page background is a GRADIENT (doc 5.10: ~RGB(255,254,253) at y=300 down to ~(243,243,243)
# at y=2100). Reproduced because TM_CCOEFF_NORMED's zero-mean normalization is the reason it
# does not matter, and a flat fixture would not exercise that.
_PAGE_TOP, _PAGE_BOTTOM = 254, 243

# A scroll offset into the world that puts several whole cards inside the window. Every test
# measures FROM here so they all share one cached pair of frames where possible.
_BASE_SCROLL = 1500

# Card heights cycled down the world, from doc 5.10's measured range: the 215px vitals block at
# one end, the tallest card observed end to end (1114px) at the other.
_CARD_HEIGHTS = (700, 980, 1114, 215, 860)


def _page_column():
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, _WORLD_H).round().astype(np.uint8)


def _build_world():
    """One tall textured page: gradient background, rounded-rect cards, canonical gutters."""
    rng = np.random.default_rng(_SEED)
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    y, i = 120, 0
    while y + 200 < _WORLD_H:
        y1 = min(_WORLD_H - 1, y + _CARD_HEIGHTS[i % len(_CARD_HEIGHTS)])
        world[y:y1, _CARD_X0:_CARD_X1] = rng.integers(
            60, 200, size=(y1 - y, _CARD_X1 - _CARD_X0), dtype=np.uint8)
        for k in range(_CORNER_RADIUS_PX):             # carve the four corner arcs back to page
            dy = _CORNER_RADIUS_PX - k
            inset = int(math.ceil(_CORNER_RADIUS_PX
                                  - math.sqrt(max(0.0, _CORNER_RADIUS_PX ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for yy in (y + k, y1 - 1 - k):
                world[yy, _CARD_X0:_CARD_X0 + inset] = col[yy]
                world[yy, _CARD_X1 - inset:_CARD_X1] = col[yy]
        if i % len(_CARD_HEIGHTS) == 3:
            # A blank white card interior — the real thing a short prompt card looks like, and
            # the reason _MIN_STRIP_STDDEV exists. Without one of these the fixture would never
            # produce a STRIP_FLAT and the flat path would go untested by every other test here.
            world[y + 30:y1 - 30, _CARD_X0 + 30:_CARD_X1 - 30] = 255
        y, i = y1 + _GUTTER, i + 1
    return world


_WORLD = _build_world()
# Static chrome, textured so it is not itself flat, and IDENTICAL on every frame — the point is
# that it does not translate when the content does.
_CHROME_TOP = np.random.default_rng(3).integers(0, 60, size=(_BAND0, _W), dtype=np.uint8)
_CHROME_BOTTOM = np.random.default_rng(4).integers(0, 60, size=(_H - _BAND1, _W), dtype=np.uint8)
_FRAME_CACHE: dict[int, bytes] = {}


def _frame(scroll: int) -> bytes:
    """The 1080x2400 window of the world at `scroll`, with static chrome painted on, as PNG.

    Content at world row `w` lands on frame row `w - scroll`, so between `_frame(s)` and
    `_frame(s + d)` the content has moved UP by exactly `d` — which is frameshift.py's positive
    `delta_px` by its stated sign convention.
    """
    if scroll not in _FRAME_CACHE:
        gray = _WORLD[scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAME_CACHE[scroll] = buf.tobytes()
    return _FRAME_CACHE[scroll]


def _estimate(shift: int, base: int = _BASE_SCROLL, **kw):
    kw.setdefault("content_band", _CONTENT_BAND)
    return frameshift.estimate_shift(_frame(base), _frame(base + shift), **kw)


def _states(result):
    return [s.state for s in result.strips]


# =====================================================================================
# The clean case: a known shift comes back exactly, with its sign
# =====================================================================================

@pytest.mark.parametrize("shift", [0, 1, 120, 363, 787, 900])
def test_known_shift_is_measured_exactly(shift):
    """0 through the trust window's own edge, including the production 363px/step cadence
    (doc 5.10.1) and the 787px worst case the hand-scrolled corpus reached. Exact, not
    approximate: the shift is a slice offset, so anything but equality is a bug."""
    r = _estimate(shift)

    assert r.status == frameshift.SHIFT_MEASURED, r.reason
    assert r.ok and r.delta_px == shift
    assert r.consensus_px == shift
    assert not r.saturated
    assert r.agreeing >= frameshift._MIN_AGREEING_STRIPS
    assert r.dissenting < r.agreeing             # a stray strip may not outvote the bank
    assert r.confidence >= frameshift._MIN_CONFIDENCE
    assert r.frame_size == (_W, _H) and r.band == (_BAND0, _BAND1)
    # Every non-zero shift here carries exactly one dissenter, and it is always the topmost
    # strip, whose content has scrolled off the band and so has nothing true to find. It costs
    # nothing: its own search range cannot reach the consensus, so it never enters `eligible`
    # and the confidence stays at 1.0. That is the eligibility rule doing its job, not luck.
    assert r.confidence == 1.0
    assert all(s.search[1] < shift for s in r.strips
               if s.state == frameshift.STRIP_MATCHED and abs(s.delta_px - shift) > 3)


def test_the_sign_convention_is_content_moving_up():
    """A forward read-scroll is POSITIVE, and the docstring's promise — content at row `y` of A
    sits at row `y - delta_px` of B — is checked against the fixture's own geometry rather than
    restated. Scrolling BACK up the profile must come back negative with the same machinery."""
    forward = _estimate(400)
    assert forward.delta_px == 400

    # The same two frames in the other order: B first, so the content moved DOWN the screen.
    backward = frameshift.estimate_shift(_frame(_BASE_SCROLL + 400), _frame(_BASE_SCROLL),
                                         content_band=_CONTENT_BAND)
    assert backward.ok and backward.delta_px == -400

    # And the promise itself, on a row of real card texture picked from the middle of the band.
    a = cv2.imdecode(np.frombuffer(_frame(_BASE_SCROLL), np.uint8), cv2.IMREAD_GRAYSCALE)
    b = cv2.imdecode(np.frombuffer(_frame(_BASE_SCROLL + 400), np.uint8), cv2.IMREAD_GRAYSCALE)
    y = 1500
    assert np.array_equal(a[y, _CARD_X0:_CARD_X1], b[y - 400, _CARD_X0:_CARD_X1])


def test_a_still_frame_measures_zero_and_is_not_called_saturated():
    """`delta_px == 0` is a real answer and must not be confused with a refusal, and the two
    strips that are pinned at zero BY CONSTRUCTION (the topmost strip can only test offsets <= 0,
    the bottommost only >= 0 — see `_strip_search_range`) must not be read as evidence that the
    content ran off the band."""
    r = _estimate(0)

    assert r.ok and r.delta_px == 0
    assert not r.saturated, r.reason
    assert _states(r).count(frameshift.STRIP_PINNED) == 2
    assert all(s.delta_px == 0 for s in r.strips if s.state == frameshift.STRIP_PINNED)


# =====================================================================================
# Saturation: a shift the method cannot resolve must never come back as a number
# =====================================================================================

@pytest.mark.parametrize("shift", [1200, 1500])
def test_a_shift_past_the_trust_window_reports_saturation_not_a_number(shift):
    """THE regression this module exists for. ops/OPENER-REDESIGN.md 5.10.1 records a prior
    estimator that "saturated at its search bound" and returned the bound as though it were a
    measurement. Here the content really did move further than the window, and the result must
    be a refusal that carries the measured magnitude — never a delta, and never the window."""
    r = _estimate(shift)

    assert r.status == frameshift.SHIFT_BEYOND_WINDOW, r.reason
    assert r.delta_px is None and not r.ok
    assert r.saturated

    # The magnitude is a MEASUREMENT, not the boundary: it equals the true shift and is nowhere
    # near `trust_window_px`. That inequality is the whole point of searching past the window.
    assert r.consensus_px == shift
    assert r.consensus_px != r.trust_window_px
    assert r.trust_window_px == 900              # 0.5 of the 1800-row band
    assert str(shift) in r.reason


def test_the_same_shift_is_measured_once_the_window_is_widened():
    """The control for the test above: nothing about 1200px is unSEEABLE, it is only untrusted.
    Widen `trust_window_px` past it on the identical frames and the same machinery returns it as
    a measurement — which proves the refusal came from the window and not from an inability to
    correlate that far."""
    refused = _estimate(1200)
    allowed = _estimate(1200, trust_window_px=1400)

    assert refused.status == frameshift.SHIFT_BEYOND_WINDOW
    assert allowed.status == frameshift.SHIFT_MEASURED
    assert allowed.ok and allowed.delta_px == 1200
    assert allowed.consensus_px == refused.consensus_px == 1200


def test_a_shift_past_the_whole_band_returns_no_number_at_all():
    """Beyond the trust window is one failure; beyond the band is another. At 1800px on an
    1800-row band no strip has any content left in common, so there is nothing to measure and
    nothing to report a magnitude for. It must still refuse — with `delta_px` None and
    `consensus_px` None — rather than emit the best of a field of noise."""
    r = _estimate(1800)

    assert not r.ok and r.delta_px is None
    assert r.status == frameshift.SHIFT_NO_CONSENSUS
    assert r.consensus_px is None
    assert r.agreeing < frameshift._MIN_AGREEING_STRIPS


# =====================================================================================
# Refusals: what "I do not know" looks like, and that it is never a number
# =====================================================================================

def test_unrelated_frames_are_refused():
    """Two frames of pure noise share no content. Nothing may be returned, and in particular no
    strip may clear the score floor by accident."""
    rng = np.random.default_rng(99)
    junk = [cv2.imencode(".png", rng.integers(0, 255, size=(_H, _W), dtype=np.uint8))[1].tobytes()
            for _ in range(2)]
    r = frameshift.estimate_shift(junk[0], junk[1], content_band=_CONTENT_BAND)

    assert not r.ok and r.delta_px is None and r.consensus_px is None
    assert r.status == frameshift.SHIFT_NO_EVIDENCE
    assert not r.saturated                       # no evidence is not evidence of a big shift
    assert r.confidence == 0.0


def test_a_featureless_frame_is_refused_by_the_texture_floor():
    """A uniform frame correlates perfectly at EVERY offset, so its argmax is arbitrary. The
    texture floor has to catch it before the peak score ever gets a say — the control being that
    the peak score would have said 1.0."""
    flat = cv2.imencode(".png", np.full((_H, _W), 200, np.uint8))[1].tobytes()
    r = frameshift.estimate_shift(flat, flat, content_band=_CONTENT_BAND)

    assert not r.ok and r.status == frameshift.SHIFT_NO_EVIDENCE
    assert set(_states(r)) == {frameshift.STRIP_FLAT}
    assert all(s.stddev < frameshift._MIN_STRIP_STDDEV for s in r.strips)


def test_ok_is_exactly_delta_px_being_present():
    """`ok` must not drift into meaning anything else — it is the one-line form of "is
    `delta_px` a number", on every status."""
    for r in (_estimate(363), _estimate(1200), _estimate(1800)):
        assert r.ok is (r.delta_px is not None)
    assert _estimate(363).ok
    assert not _estimate(1200).ok


# =====================================================================================
# The band restriction, and why it is not cosmetic
# =====================================================================================

def test_static_chrome_outside_the_band_is_excluded_by_the_band_and_not_by_luck():
    """The status bar and bottom nav do NOT translate when the content scrolls. Strips cut
    across them vote for a shift of zero however far the content moved, so `content_band` has to
    keep them out. The control is the same frames re-run with the band opened to the whole
    frame: the answer survives — the median is robust — but the chrome strips show up as
    dissenters and drag the confidence down, which is exactly the evidence the band removes."""
    banded = _estimate(363)
    whole_frame = _estimate(363, content_band=(0.0, 1.0))

    assert banded.delta_px == whole_frame.delta_px == 363
    assert banded.confidence == 1.0
    assert whole_frame.dissenting > banded.dissenting
    assert whole_frame.confidence < banded.confidence

    # ...and the dissenters are specifically the chrome, sitting at a shift of zero.
    stuck = [s for s in whole_frame.strips
             if s.state == frameshift.STRIP_MATCHED and s.delta_px == 0]
    assert stuck, "the fixture's chrome did not stay put, so this proves nothing"


# =====================================================================================
# The decision surface, tested directly on hand-built strip banks (no images, no cv2)
# =====================================================================================

def _strip(y0, state, delta, *, search=(-1000, 1000), score=1.0, runner_up=0.0, stddev=50.0):
    return frameshift.StripMatch(y0=y0, y1=y0 + 96, state=state, delta_px=delta, score=score,
                                 runner_up=runner_up, stddev=stddev, search=search)


def _resolve(strips, **kw):
    kw.setdefault("window", 900)
    kw.setdefault("tolerance", frameshift._AGREEMENT_TOLERANCE_PX)
    kw.setdefault("min_agreeing", frameshift._MIN_AGREEING_STRIPS)
    kw.setdefault("min_saturation", frameshift._MIN_SATURATION_STRIPS)
    kw.setdefault("min_confidence", frameshift._MIN_CONFIDENCE)
    kw.setdefault("pin_margin", frameshift._PIN_MARGIN_PX)
    return frameshift._resolve(tuple(strips), frame_size=(_W, _H), band=(_BAND0, _BAND1),
                               np=np, **kw)


def test_pinned_strips_never_contribute_a_value():
    """A pinned offset is a LOWER BOUND on how far the content went, not a measurement of it.
    Five strips pinned at 400 plus two matched at 363 must not average, must not out-vote, and
    must not turn 400 into an answer."""
    strips = [_strip(300 + i * 200, frameshift.STRIP_PINNED, 400, search=(-1000, 400))
              for i in range(5)]
    strips += [_strip(1400, frameshift.STRIP_MATCHED, 363),
               _strip(1600, frameshift.STRIP_MATCHED, 363)]
    r = _resolve(strips)

    assert r.delta_px is None                    # two matched strips is under the quorum
    assert r.consensus_px is None
    assert 400 not in (r.delta_px, r.consensus_px)
    assert r.status == frameshift.SHIFT_NO_CONSENSUS


def test_the_quorum_is_what_separates_a_measurement_from_a_refusal():
    """Positive and negative on the single gate that matters: identical strips, one fewer
    voter. Nothing else changes between the two calls."""
    voters = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(3)]
    assert _resolve(voters).delta_px == 363
    assert _resolve(voters[:2]).delta_px is None
    assert _resolve(voters[:2]).status == frameshift.SHIFT_NO_CONSENSUS


def test_weak_strips_are_silence_not_confidence_dissent_even_when_in_range():
    """A weak strip is silence about page translation, not dissent: its own content was not
    found in frame B above the score floor. Therefore it must not enter coverage even when its
    searchable range includes the consensus. The two banks below differ only in that range, and
    neither weak strip gets a vote or changes the denominator."""
    voters = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(3)]

    out_of_reach = _resolve(voters + [_strip(2000, frameshift.STRIP_WEAK, None,
                                             search=(-1000, 100), score=0.2)])
    in_reach = _resolve(voters + [_strip(2000, frameshift.STRIP_WEAK, None,
                                         search=(-1000, 1000), score=0.2)])

    assert out_of_reach.eligible == in_reach.eligible == 3
    assert out_of_reach.confidence == in_reach.confidence == 1.0
    assert out_of_reach.delta_px == in_reach.delta_px == 363


def test_unanimous_matched_strips_measure_despite_a_majority_of_weak_strips():
    """The corrected eligible rule must let the three-strip quorum speak for itself. More
    strips in this bank are weak than matched, but weak means the changing content is gone from
    frame B — silence, not an opposing measurement — so the unanimous matched strips measure.
    """
    matched = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(3)]
    weak = [_strip(1000 + i * 120, frameshift.STRIP_WEAK, None, score=0.2)
            for i in range(4)]

    r = _resolve(matched + weak)

    assert len(weak) > len(matched)
    assert r.status == frameshift.SHIFT_MEASURED
    assert r.delta_px == r.consensus_px == 363
    assert r.agreeing == r.eligible == len(matched)
    assert r.dissenting == 0 and r.confidence == 1.0


def test_saturation_is_measured_on_the_offset_and_not_on_pinnedness():
    """Pinnedness alone cannot mean "it moved too far": on a still frame the outermost strips
    are pinned at zero by construction. `saturated` therefore keys on strips putting the content
    OUTSIDE the window, and the pair below is the same bank with only the offsets moved."""
    near = [_strip(300, frameshift.STRIP_PINNED, 0, search=(-1000, 0)),
            _strip(2000, frameshift.STRIP_PINNED, 0, search=(0, 1000))]
    far = [_strip(300, frameshift.STRIP_PINNED, 1400, search=(-1000, 1400)),
           _strip(2000, frameshift.STRIP_PINNED, 1400, search=(0, 1400))]

    assert not _resolve(near).saturated
    assert _resolve(far).saturated
    assert _resolve(far).delta_px is None        # saturated is still never a number
    assert _resolve(near).status == _resolve(far).status == frameshift.SHIFT_NO_EVIDENCE


def test_dissenting_strips_are_outvoted_but_counted():
    """A single strip landing somewhere else must not veto a clear majority — it is the wrong
    ones being wrong in DIFFERENT ways that makes the median trustworthy — but it must be
    reported so a caller can see the frame was not unanimous."""
    strips = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(5)]
    strips.append(_strip(1500, frameshift.STRIP_MATCHED, -871))
    r = _resolve(strips)

    assert r.delta_px == 363
    assert r.dissenting == 1 and r.agreeing == 5
    assert r.confidence == pytest.approx(5 / 6)


# =====================================================================================
# The 2026-08-16 bimodal-bank rescue: `_vote_clusters`, `_pins_allow` and
# `_exact_cluster_shift`, spliced into `_resolve` at the point where the plain median has
# failed its ordinary quorum OR confidence gate. A median is the right consensus statistic for
# ONE moving population; a strip bank that straddles a still page and an autoplaying video is
# TWO, and the median can either land on a value nothing reported or retain a real cluster that
# is diluted by the non-rigid one. The first two tests below are the exact live evidence —
# a real Hinge profile capture that produced a whole 64-frame profile with zero numbered
# items before this fix — reproduced strip-for-strip from the driver's own debug log. The
# rest are hand-built banks that isolate one clause of the evidence bar each
# (`_exact_cluster_shift`'s docstring: exactly one group pixel-exact AND at quorum; a
# second such group is ambiguity, not a tiebreak; a larger inexact group neither wins nor
# blocks; every pinned bound must still admit the winner), so a future change to one clause
# cannot silently loosen another without a test noticing.
# =====================================================================================

def test_regression_bimodal_video_bank_pair_36_37_now_measures_145():
    """THE failure this repair exists for, not a hypothetical: a 2026-08-16 Hinge profile
    whose last card was an autoplaying video, captured mid read-scroll at pair 36/37 of 64
    frames. Nine strips lying over ordinary page content — including the pinned strip at
    y=442, whose peak sits on its own range's edge and so is a LOWER bound, not a vote —
    measured the true shift identically to the pixel (+145, four times over). Five strips
    lying over the playing video re-correlated against the video's own internal motion and
    landed on a coherent-LOOKING but non-rigid cluster (+73..+81, no two exactly equal).
    Before this fix the plain median of all ten voters landed at +113 — a value not one of
    them reported — so the pair came back `SHIFT_NO_CONSENSUS` and the whole profile
    produced zero numbered items.

    +145 is independently corroborated by two things this module never looks at: the
    Hinge mute-control glyph on that video card sat at y=1135 in frame 36 and y=990 in
    frame 37 (exactly 145px), and `segment.py`'s own top/bottom/heart landmarks for the
    surrounding cards all shifted 145px too. This is the live per-strip record from the
    driver's debug log, unedited.
    """
    strips = [
        _strip(300, frameshift.STRIP_WEAK, None,
              search=(-1704, 0), score=0.2611, runner_up=0.1994),
        _strip(442, frameshift.STRIP_PINNED, 142,
              search=(-1562, 142), score=0.8001, runner_up=0.6468),
        _strip(584, frameshift.STRIP_MATCHED, 145,
              search=(-1420, 284), score=1.0000, runner_up=0.3939),
        _strip(726, frameshift.STRIP_MATCHED, 145,
              search=(-1278, 426), score=1.0000, runner_up=0.4205),
        _strip(868, frameshift.STRIP_MATCHED, 145,
              search=(-1136, 568), score=1.0000, runner_up=0.8414),
        _strip(1010, frameshift.STRIP_MATCHED, 145,
              search=(-994, 710), score=0.9779, runner_up=0.3344),
        _strip(1152, frameshift.STRIP_MATCHED, 78,
              search=(-852, 852), score=0.9290, runner_up=0.8091),
        _strip(1294, frameshift.STRIP_MATCHED, 81,
              search=(-710, 994), score=0.9897, runner_up=0.8680),
        _strip(1436, frameshift.STRIP_MATCHED, 80,
              search=(-568, 1136), score=0.9169, runner_up=0.7147),
        _strip(1578, frameshift.STRIP_MATCHED, 77,
              search=(-426, 1278), score=0.8966, runner_up=0.7671),
        _strip(1720, frameshift.STRIP_MATCHED, 73,
              search=(-284, 1420), score=0.8952, runner_up=0.7354),
        _strip(1862, frameshift.STRIP_WEAK, None,
              search=(-142, 1562), score=0.7273, runner_up=0.4965),
        _strip(2004, frameshift.STRIP_MATCHED, 145,
              search=(0, 1704), score=0.7985, runner_up=0.7171),
    ]
    r = _resolve(strips)

    assert r.status == frameshift.SHIFT_MEASURED, r.reason
    assert r.ok and r.delta_px == 145
    assert r.consensus_px == 145
    # Only the nine page strips agree; the five video strips and the two weak strips do
    # not, which is exactly the 5-of-10 split the corpus recorded (STRIP_WEAK is excluded
    # from the eligible denominator on its own separate grounds — see
    # `test_weak_strips_are_silence_not_confidence_dissent_even_when_in_range` — leaving
    # 10 of the bank's 13 strips as eligible: 9 matched + the 1 pinned strip whose range
    # admits 145).
    assert r.confidence == 0.5
    assert r.agreeing == 5
    assert r.eligible == 10
    # The reason string has to say WHY a number came back where the old code refused: it
    # must name the split, not just assert the delta.
    assert "split into" in r.reason


def test_regression_bimodal_saturated_bank_pair_43_44_measures_zero():
    """The same profile, 7 pairs later: the read-scroll has hit the bottom of the page
    while the video keeps playing. Ten strips over the now-motionless page unanimously
    report +0 (two of them pinned there BY CONSTRUCTION — see
    `test_a_still_frame_measures_zero_and_is_not_called_saturated` — which is fine, since
    `_exact_cluster_shift`'s exact-group filter only requires internal agreement, not that
    every member be `STRIP_MATCHED`... except pinned strips do not enter `voters` at all,
    so the five 0-valued MATCHED strips plus a 0-valued pinned strip is what actually
    forms the exact winning cluster below). Five strips over the still-playing video
    report a coherent-looking -54..-57. The plain median before this fix landed at -27 — a
    value no strip reported — refusing the one pair whose answer, 0px, IS the signal that
    the profile has reached its bottom. Refusing it is not a neutral non-answer: it is
    what let the capture keep issuing scroll commands against a page that could no longer
    move, burning the rest of the run on a wall it could not detect.
    """
    strips = [
        _strip(300, frameshift.STRIP_PINNED, 0,
              search=(-1704, 0), score=1.0000, runner_up=0.6531),
        _strip(442, frameshift.STRIP_MATCHED, 0,
              search=(-1562, 142), score=1.0000, runner_up=0.3876),
        _strip(584, frameshift.STRIP_MATCHED, 0,
              search=(-1420, 284), score=1.0000, runner_up=0.4255),
        _strip(726, frameshift.STRIP_MATCHED, 0,
              search=(-1278, 426), score=1.0000, runner_up=0.8465),
        _strip(868, frameshift.STRIP_MATCHED, 0,
              search=(-1136, 568), score=0.9782, runner_up=0.3694),
        _strip(1010, frameshift.STRIP_MATCHED, -57,
              search=(-994, 710), score=0.8653, runner_up=0.7682),
        _strip(1152, frameshift.STRIP_MATCHED, -54,
              search=(-852, 852), score=0.9284, runner_up=0.8669),
        _strip(1294, frameshift.STRIP_MATCHED, -54,
              search=(-710, 994), score=0.9176, runner_up=0.6924),
        _strip(1436, frameshift.STRIP_MATCHED, -56,
              search=(-568, 1136), score=0.9102, runner_up=0.7931),
        _strip(1578, frameshift.STRIP_MATCHED, -56,
              search=(-426, 1278), score=0.8795, runner_up=0.7449),
        _strip(1720, frameshift.STRIP_WEAK, None,
              search=(-284, 1420), score=0.7408, runner_up=0.5247),
        _strip(1862, frameshift.STRIP_MATCHED, 0,
              search=(-142, 1562), score=0.9886, runner_up=0.8156),
        _strip(2004, frameshift.STRIP_PINNED, 0,
              search=(0, 1704), score=1.0000, runner_up=0.1848),
    ]
    r = _resolve(strips)

    assert r.status == frameshift.SHIFT_MEASURED, r.reason
    assert r.ok and r.delta_px == 0
    assert r.consensus_px == 0
    assert "split into" in r.reason


def test_regression_exact_static_zero_cluster_overrides_only_the_confidence_floor():
    """Lea's saved 2026-08-31 tail pair: a unique exact zero cluster is the page.

    The seven matched NCC strips were exactly ``[-5, 0, 0, 6, 5, 0, 11]``.  Their ordinary
    median is already zero and reaches the three-strip quorum, but only 3/7 eligible strips
    agree, so the old resolver refused solely on its 0.50 confidence floor.  The other four
    video-polluted answers form no second exact quorate cluster.  A pinned zero bound is also
    present and admits zero.  The exact-cluster rule therefore has one principled answer and
    must report it while retaining the raw coverage telemetry.
    """
    strips = [
        _strip(442, frameshift.STRIP_MATCHED, -5),
        _strip(726, frameshift.STRIP_MATCHED, 0),
        _strip(868, frameshift.STRIP_MATCHED, 0),
        _strip(1152, frameshift.STRIP_MATCHED, 6),
        _strip(1294, frameshift.STRIP_MATCHED, 5),
        _strip(1720, frameshift.STRIP_MATCHED, 0),
        _strip(1862, frameshift.STRIP_MATCHED, 11),
        # The saved pair's bottom strip was pinned at the LOW end of this exact range.
        _strip(2004, frameshift.STRIP_PINNED, 0, search=(0, 1704)),
    ]
    r = _resolve(strips)

    assert r.status == frameshift.SHIFT_MEASURED, r.reason
    assert r.delta_px == r.consensus_px == 0
    assert r.agreeing == 3 and r.dissenting == 4 and r.eligible == 7
    assert r.confidence == pytest.approx(3 / 7)
    assert "split into" in r.reason and "confidence gate" in r.reason


def test_low_confidence_inexact_median_stays_refused():
    """Confidence alone is never waived: the rescued cluster must be pixel-exact.

    The median +2 has three nearby votes, but they are +0/+1/+2 rather than one exact answer.
    With all seven strips eligible its 3/7 confidence is below the ordinary floor, and no exact
    quorate cluster exists to override that refusal.
    """
    r = _resolve([
        _strip(442, frameshift.STRIP_MATCHED, -10),
        _strip(726, frameshift.STRIP_MATCHED, 0),
        _strip(868, frameshift.STRIP_MATCHED, 1),
        _strip(1152, frameshift.STRIP_MATCHED, 2),
        _strip(1294, frameshift.STRIP_MATCHED, 10),
        _strip(1720, frameshift.STRIP_MATCHED, 11),
        _strip(1862, frameshift.STRIP_MATCHED, 12),
    ])

    assert r.status == frameshift.SHIFT_NO_CONSENSUS, r.reason
    assert r.delta_px is None and r.consensus_px is None
    assert r.confidence == pytest.approx(3 / 7)


def test_two_pixel_exact_quorate_groups_are_ambiguous_and_refused():
    """The safety rail `_exact_cluster_shift`'s docstring states in words: a SECOND
    pixel-exact quorate group is a genuine ambiguity, and is refused rather than resolved
    by preferring the bigger one. Built as one group of 3 votes at +300 and one of 4 at
    +900, both far apart and both internally unanimous — the same shape as the real
    bimodal bank, except this one has no principled winner.

    A bare 3-vs-4 split of nothing else would not even reach this code: with a clean
    strict majority, the plain unweighted median already IS 900 (the literal 4th of 7
    sorted values) and already has its own 4-strip quorum, so `_resolve` would return a
    measurement without ever asking `_exact_cluster_shift` anything — see
    `test_a_larger_inexact_group_does_not_win_or_block_the_exact_one` for the sibling test
    where that is exactly the wrong outcome to want. A lone spoiler vote at -600, too small
    a group (1) to ever reach quorum itself, is added only to pull the raw median off both
    real clusters (to +600, a value nothing reported) so the ordinary path fails its own
    quorum first and the ambiguity test is the one actually exercised.
    """
    group_a = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 300) for i in range(3)]
    group_b = [_strip(1400 + i * 200, frameshift.STRIP_MATCHED, 900) for i in range(4)]
    spoiler = [_strip(2200, frameshift.STRIP_MATCHED, -600)]
    r = _resolve(group_a + group_b + spoiler)

    assert r.status == frameshift.SHIFT_NO_CONSENSUS, r.reason
    assert r.delta_px is None and not r.ok
    assert r.consensus_px is None
    assert 300 not in (r.delta_px, r.consensus_px)
    assert 900 not in (r.delta_px, r.consensus_px)


def test_a_larger_inexact_group_does_not_win_or_block_the_exact_one():
    """Size is not evidence of rigidity — the video's strips in the real bank
    outnumbered nothing here, but a future bank could easily have more video-straddling
    strips than page strips, and the rescue must not let a bigger SPREAD cluster either
    win outright or veto the smaller exact one. 3 votes sit exactly on +200; 6 more are
    spread +90..+110 (20px of internal spread, comfortably outside the module's own
    finding that agreeing strips land within 0px of each other — see
    `_AGREEMENT_TOLERANCE_PX`'s corpus note), so `_vote_clusters` never welds them into
    one block, let alone a pixel-exact one. The correct answer is +200 regardless of the
    6-vs-3 headcount.

    Each spread vote is given a search range (-50, 150) that geometrically cannot reach
    200 — modelling a strip too low in the band to have searched that far — so the
    `eligible` denominator counts only the 3 strips that could and did agree, rather than
    being diluted by 6 strips that could never have voted for the winner in the first
    place. Without that, `confidence` sits at 3/9 and the pair is refused for coverage
    even though the winner is unambiguous, which is a different failure than the one this
    test is pinning.
    """
    exact = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 200) for i in range(3)]
    spread = [_strip(1400 + i * 150, frameshift.STRIP_MATCHED, d, search=(-50, 150))
              for i, d in enumerate((90, 94, 98, 102, 106, 110))]
    r = _resolve(exact + spread)

    assert r.status == frameshift.SHIFT_MEASURED, r.reason
    assert r.ok and r.delta_px == 200
    assert r.consensus_px == 200
    assert r.agreeing == 3 and r.eligible == 3
    assert r.confidence == 1.0
    assert r.dissenting == 6                      # counted, not silenced, just outvoted
    assert "split into" in r.reason


def test_no_group_pixel_exact_stays_refused():
    """The rescue's bar is unanimity to the pixel, not merely tight agreement. Group A
    spreads 199/200/201 and group B spreads 798/799/800/801 — both internally within
    `_AGREEMENT_TOLERANCE_PX` of each other, so `_vote_clusters` still welds each into one
    block, but neither block's own members are IDENTICAL, so neither clears
    `_exact_cluster_shift`'s `group[0] == group[-1]` bar. With no exact candidate at all,
    the pair must stay refused — there is nothing here as trustworthy as the real bank's
    unanimous +145.

    The spoiler at -600 plays the same role as in the ambiguity test above: without it the
    4-vote group B would already form its own quorum under the plain median (its members
    are within tolerance of each other even though not pixel-identical) and the ordinary
    path would return a measurement before this test ever touched the rescue.
    """
    group_a = [_strip(300 + i * 150, frameshift.STRIP_MATCHED, 199 + i) for i in range(3)]
    group_b = [_strip(1400 + i * 150, frameshift.STRIP_MATCHED, 798 + i) for i in range(4)]
    spoiler = [_strip(2200, frameshift.STRIP_MATCHED, -600)]
    r = _resolve(group_a + group_b + spoiler)

    assert r.status == frameshift.SHIFT_NO_CONSENSUS, r.reason
    assert r.delta_px is None and not r.ok
    assert r.consensus_px is None


def test_the_only_exact_group_under_quorum_stays_refused():
    """Two strips agreeing to the pixel is not evidence — it is the same 2-is-not-enough
    rule `test_the_quorum_is_what_separates_a_measurement_from_a_refusal` pins for the
    ordinary path, and the rescue must hold the exact SAME line rather than relaxing it
    because the two votes happen to be unanimous. Group A is 2 votes, pixel-exact, at
    +300 — below `_MIN_AGREEING_STRIPS` — so `_exact_cluster_shift`'s own quorum filter
    drops it before the "exactly one" check ever runs. Group B is 5 votes, quorate, but
    spread +700..+712 (not pixel-exact), so it fails the OTHER half of the bar. Neither
    group qualifies, `exact` ends up empty, and the pair is refused precisely because the
    only unanimous evidence in the bank is too thin to trust.
    """
    group_a = [_strip(300 + i * 150, frameshift.STRIP_MATCHED, 300) for i in range(2)]
    group_b = [_strip(1400 + i * 150, frameshift.STRIP_MATCHED, 700 + 3 * i) for i in range(5)]
    spoiler = [_strip(2200, frameshift.STRIP_MATCHED, -100)]
    r = _resolve(group_a + group_b + spoiler)

    assert r.status == frameshift.SHIFT_NO_CONSENSUS, r.reason
    assert r.delta_px is None and not r.ok
    assert r.consensus_px is None


def test_the_rescue_cannot_touch_a_bank_the_median_already_resolves():
    """The load-bearing guarantee stated in `_resolve`'s own comment: the rescue runs
    ONLY after the plain median has already failed its quorum OR confidence gate, so a pair that
    ordinary rules already accept must measure identically tomorrow — same delta, same reason,
    no split note appended — because the rescue was never consulted. This reuses
    `test_dissenting_strips_are_outvoted_but_counted`'s unimodal bank (5 strips agreeing
    at +363, one dissenting at -871): the plain median is already +363 with a 5-strip
    quorum, so `_exact_cluster_shift` never runs, and the reason string must contain no
    trace of clustering language.
    """
    strips = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(5)]
    strips.append(_strip(1500, frameshift.STRIP_MATCHED, -871))
    r = _resolve(strips)

    assert r.status == frameshift.SHIFT_MEASURED, r.reason
    assert r.delta_px == 363
    assert r.reason == "content moved +363px — 5 of 6 eligible strips agree within 3px (1 dissent, 6 matched)"
    assert "split into" not in r.reason


def test_pins_allow_vetoes_a_rescue_that_contradicts_a_pinned_bound():
    """`_pins_allow` is the free extra constraint the rescue is held to even though the
    ordinary median path has never needed it: a pinned strip's offset is a lower or upper
    BOUND on the true shift, and the rescue must not report a value a pin has already
    ruled out. This takes the passing bank from
    `test_a_larger_inexact_group_does_not_win_or_block_the_exact_one` (which measures
    +200 on its own) and adds one more strip, pinned at the HIGH end of its own search
    range (`search=(0, 500)`, `delta_px=500 == hi`) — meaning that strip's true match is
    AT LEAST 500, strictly more than the +200 the exact cluster would otherwise win on.
    `_pins_allow` must veto it, `_exact_cluster_shift` must return None, and the pair must
    fall back to the ordinary (still-failing) refusal rather than reporting a number one
    of its own strips has already contradicted.
    """
    exact = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 200) for i in range(3)]
    spread = [_strip(1400 + i * 150, frameshift.STRIP_MATCHED, d, search=(-50, 150))
              for i, d in enumerate((90, 94, 98, 102, 106, 110))]
    contradicting_pin = _strip(2600, frameshift.STRIP_PINNED, 500, search=(0, 500))

    # Control: without the pin, this exact bank measures +200 (proven above).
    assert _resolve(exact + spread).delta_px == 200

    r = _resolve(exact + spread + [contradicting_pin])
    assert r.status == frameshift.SHIFT_NO_CONSENSUS, r.reason
    assert r.delta_px is None and not r.ok
    assert 200 not in (r.delta_px, r.consensus_px)


def test_a_pin_one_pixel_inside_its_bound_vetoes_just_as_hard():
    """The margin, not equality, is what makes a strip pinned — so it is what must bind here.

    Found 2026-09-04: every pinned fixture in this file happens to sit EXACTLY on its own bound,
    so an `== high` test passed all of them while ignoring 8 of the 9 offsets per end that
    `_search_strip` actually calls pinned. This is the same bank as the test above with the pin
    moved one pixel inside its range — still `STRIP_PINNED` by `_search_strip`'s own rule
    (`delta >= hi - _PIN_MARGIN_PX`), still saying the content went at least 499px, and the
    rescue's +200 still contradicts it. The near-bound pin is the routine case, not the exotic
    one: strip bounds sit one strip pitch apart, so a true shift landing a few pixels inside some
    strip's bound produces exactly this.
    """
    exact = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 200) for i in range(3)]
    spread = [_strip(1400 + i * 150, frameshift.STRIP_MATCHED, d, search=(-50, 150))
              for i, d in enumerate((90, 94, 98, 102, 106, 110))]
    near_bound_pin = _strip(2600, frameshift.STRIP_PINNED, 499, search=(0, 500))

    # The premise: `_search_strip` would have labelled this strip pinned, so `_pins_allow` is
    # being handed the same state the estimator really produces.
    assert 499 >= 500 - frameshift._PIN_MARGIN_PX
    assert not frameshift._pins_allow([near_bound_pin], 200,
                                      pin_margin=frameshift._PIN_MARGIN_PX)
    # ...and the pin's own offset is never what it rejects: a candidate at or past the bound,
    # which is what the true shift looks like when a strip pins one pixel short of it, passes.
    assert frameshift._pins_allow([near_bound_pin], 499, pin_margin=frameshift._PIN_MARGIN_PX)
    assert frameshift._pins_allow([near_bound_pin], 700, pin_margin=frameshift._PIN_MARGIN_PX)

    r = _resolve(exact + spread + [near_bound_pin])
    assert r.status == frameshift.SHIFT_NO_CONSENSUS, r.reason
    assert r.delta_px is None and not r.ok
    assert 200 not in (r.delta_px, r.consensus_px)

    # The low end is symmetric: a strip pinned one pixel above the bottom of its range says the
    # content went at most -499px, so a rescue at -200 is equally ruled out.
    low_pin = _strip(2600, frameshift.STRIP_PINNED, -499, search=(-500, 0))
    assert not frameshift._pins_allow([low_pin], -200, pin_margin=frameshift._PIN_MARGIN_PX)
    assert frameshift._pins_allow([low_pin], -499, pin_margin=frameshift._PIN_MARGIN_PX)


def test_vote_clusters_splits_on_gaps_over_tolerance_and_keeps_exact_ties_together():
    """`_vote_clusters` in isolation, with no `_resolve` machinery around it. Values
    exactly `tolerance` apart must stay in ONE group — the boundary is inclusive
    (`delta - groups[-1][-1] > tolerance`, not `>=`) — while a gap one pixel wider must
    start a new one. `STRIP_WEAK` strips carry `delta_px=None` and must be silently
    dropped rather than raising or being treated as a zero.
    """
    voters = [
        _strip(0, frameshift.STRIP_MATCHED, 100),
        _strip(1, frameshift.STRIP_MATCHED, 103),   # exactly 3px from 100: same group
        _strip(2, frameshift.STRIP_MATCHED, 107),   # 4px from 103: new group
        _strip(3, frameshift.STRIP_WEAK, None),     # no vote at all
    ]
    groups = frameshift._vote_clusters(voters, tolerance=3)

    assert groups == [[100, 103], [107]]


# =====================================================================================
# Fail loud: the cases where there is no result at all
# =====================================================================================

def test_undecodable_bytes_raise_rather_than_return_a_result():
    with pytest.raises(frameshift.ShiftEstimationError) as exc:
        frameshift.estimate_shift(b"not a png", _frame(_BASE_SCROLL),
                                  content_band=_CONTENT_BAND)
    assert "first" in str(exc.value)

    with pytest.raises(frameshift.ShiftEstimationError) as exc:
        frameshift.estimate_shift(_frame(_BASE_SCROLL), b"not a png",
                                  content_band=_CONTENT_BAND)
    assert "second" in str(exc.value)


def test_mismatched_frame_sizes_raise_rather_than_resize():
    """Silently rescaling would return a number in the wrong units, which downstream is
    indistinguishable from a correct one."""
    small = cv2.imencode(".png", np.random.default_rng(5).integers(
        0, 255, size=(_H // 2, _W // 2), dtype=np.uint8))[1].tobytes()
    with pytest.raises(frameshift.ShiftEstimationError, match="different sizes"):
        frameshift.estimate_shift(_frame(_BASE_SCROLL), small, content_band=_CONTENT_BAND)


def test_a_band_too_short_to_search_raises_rather_than_reporting_no_evidence():
    """A configuration mistake must not come back looking like a content problem: with no room
    for one strip plus a search span, every strip would be STRIP_NO_RANGE and the result would
    read as "the frames do not correspond"."""
    with pytest.raises(frameshift.ShiftEstimationError, match="too short to search"):
        frameshift.estimate_shift(_frame(_BASE_SCROLL), _frame(_BASE_SCROLL + 100),
                                  content_band=(0.50, 0.55))


def test_a_bank_too_small_to_reach_its_own_quorum_raises():
    """A configuration that can never answer must not masquerade as a frame that cannot be
    read. With fewer strips than `min_agreeing_strips` every pair would come back
    SHIFT_NO_CONSENSUS forever, and a caller would go looking at the phone."""
    with pytest.raises(frameshift.ShiftEstimationError, match="quorum"):
        _estimate(363, strip_count=2)


def test_a_non_positive_trust_window_raises():
    """Same class of mistake in the other direction: with a window of zero, even a perfectly
    measured shift of 1px is "beyond the window", so every pair would report saturation."""
    with pytest.raises(frameshift.ShiftEstimationError, match="trust window"):
        _estimate(363, trust_window_px=0)


def test_a_frame_narrower_than_its_own_margins_raises():
    narrow = cv2.imencode(".png", np.random.default_rng(6).integers(
        0, 255, size=(_H, 80), dtype=np.uint8))[1].tobytes()
    with pytest.raises(frameshift.ShiftEstimationError, match="no content columns"):
        frameshift.estimate_shift(narrow, narrow, content_band=_CONTENT_BAND)


# =====================================================================================
# Leaf-module property
# =====================================================================================

def test_the_module_is_a_leaf_and_does_not_pull_in_the_driver():
    """frameshift.py must stay importable without hinge.py, both because it is pure geometry and
    because hinge.py is the eventual IMPORTER. Unlike segment.py it does not even need
    `_match_glyph` at runtime, so nothing here should ever reach for it — checked in a fresh
    interpreter, since this test session has hinge.py loaded already."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import operation_love.drivers.frameshift as f; "
         "print('operation_love.drivers.hinge' in sys.modules); print(f.estimate_shift.__name__)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "estimate_shift"], out.stdout + out.stderr


def test_calibration_geometry_is_shared_with_segment_and_not_re_declared():
    """The card margin and the band arithmetic must be ONE definition, or a strip could be cut
    inside a band that segment.py drew somewhere else."""
    assert frameshift._CARD_MARGIN_PX is segment._CARD_MARGIN_PX
    assert frameshift._band_rows is segment._band_rows
    assert frameshift._band_rows(_CONTENT_BAND, _H) == (_BAND0, _BAND1)


# =====================================================================================
# THE CROSS-DIRECTION QUORUM RESCUE (Maja, 2026-09-04, run 1d84909bf1bb)
# =====================================================================================
# The strip banks below are transcribed from that run's own refused return leg -- the pair
# retained as 00009/00010_still_photo_dwell_return_chain_{before,after}.png -- exactly as the
# 2026-08-16 bimodal-video regressions above transcribe theirs. The frames themselves live under
# the gitignored data/ tree, so the bank is the fixture.

def _maja_forward():
    """Strips cut from the BEFORE frame, searched in the after frame. Median +630, 2 agree."""
    return (
        _strip(300, frameshift.STRIP_FLAT, None, score=0.0, stddev=2.9, search=(-1704, 0)),
        _strip(442, frameshift.STRIP_WEAK, None, score=0.447, search=(-1562, 142)),
        _strip(584, frameshift.STRIP_WEAK, None, score=0.330, search=(-1420, 284)),
        _strip(726, frameshift.STRIP_MATCHED, -1178, score=0.961, search=(-1278, 426)),
        _strip(868, frameshift.STRIP_WEAK, None, score=0.290, search=(-1136, 568)),
        _strip(1010, frameshift.STRIP_MATCHED, 630, score=0.977, search=(-994, 710)),
        _strip(1152, frameshift.STRIP_MATCHED, 630, score=0.841, search=(-852, 852)),
        _strip(1294, frameshift.STRIP_MATCHED, 723, score=0.861, search=(-710, 994)),
        _strip(1436, frameshift.STRIP_WEAK, None, score=0.555, search=(-568, 1136)),
    )


def _maja_reverse():
    """Strips cut from the AFTER frame, searched in the before frame. Median -630, 2 agree."""
    return (
        _strip(300, frameshift.STRIP_MATCHED, -630, score=0.998, search=(-1704, 0)),
        _strip(442, frameshift.STRIP_MATCHED, -630, score=0.910, search=(-1562, 142)),
        _strip(584, frameshift.STRIP_MATCHED, -747, score=0.881, search=(-1420, 284)),
        _strip(726, frameshift.STRIP_WEAK, None, score=0.528, search=(-1278, 426)),
        _strip(1152, frameshift.STRIP_MATCHED, -796, score=0.797, search=(-852, 852)),
        _strip(1294, frameshift.STRIP_MATCHED, -9, score=0.802, search=(-710, 994)),
        _strip(1436, frameshift.STRIP_MATCHED, 1122, score=0.782, search=(-568, 1136)),
        _strip(2004, frameshift.STRIP_MATCHED, 1178, score=0.915, search=(0, 1704)),
    )


def _recovery(forward_strips, reverse_strips, **kwargs):
    """Drive `estimate_shift_with_reverse_recovery` over two hand-built banks."""
    banks = iter((_resolve(forward_strips), _resolve(reverse_strips)))
    return frameshift.estimate_shift_with_reverse_recovery(
        b"before", b"after", estimator=lambda *_a, **_kw: next(banks), **kwargs)


def test_each_maja_direction_alone_is_still_refused():
    """The premise. Neither bank reaches quorum on its own, and that must not have changed."""
    forward, reverse = _resolve(_maja_forward()), _resolve(_maja_reverse())
    assert forward.status == frameshift.SHIFT_NO_CONSENSUS
    assert reverse.status == frameshift.SHIFT_NO_CONSENSUS
    assert forward.delta_px is None and reverse.delta_px is None
    assert forward.agreeing == 2 and reverse.agreeing == 2
    assert frameshift._MIN_AGREEING_STRIPS == 3, "the quorum itself must not have been lowered"


def test_two_directions_that_both_centre_on_one_exact_shift_measure_it():
    """The rescue. Four strips from two banks with no strip in common, all reporting +630."""
    result, forward, reverse = _recovery(_maja_forward(), _maja_reverse())
    assert result.status == frameshift.SHIFT_MEASURED
    assert result.delta_px == 630
    assert result.consensus_px == 630
    assert "median" in result.reason and "+630px" in result.reason
    # The RAW per-direction estimates are handed back untouched, so a debug row still records
    # what each bank actually said rather than the repaired answer.
    assert forward.status == frameshift.SHIFT_NO_CONSENSUS
    assert reverse.status == frameshift.SHIFT_NO_CONSENSUS


def test_the_rescue_never_runs_when_either_direction_can_answer():
    """It lives inside the both-refused branch, so it can only turn a refusal into a number."""
    measured = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(5)]
    result, _f, reverse = _recovery(measured, _maja_reverse())
    assert result.status == frameshift.SHIFT_MEASURED and result.delta_px == 363
    assert reverse is None, "a forward measurement must not even look at the other direction"
    # And an ordinary reverse measurement still wins by the pre-existing 2026-09-02 rule.
    result, _f, _r = _recovery(_maja_forward(), measured)
    assert result.delta_px == -363
    assert "reverse source strips measured the same pair" in result.reason


@pytest.mark.parametrize("status_strips", [
    # a saturation report must never be talked round by a cross-direction cluster...
    [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 1500, search=(-2000, 2000))
     for i in range(5)],
    # ...nor must silence.
    [_strip(300 + i * 200, frameshift.STRIP_WEAK, None) for i in range(5)],
])
def test_the_rescue_never_overrules_saturation_or_silence(status_strips):
    forward = _resolve(status_strips)
    assert forward.status in (frameshift.SHIFT_BEYOND_WINDOW, frameshift.SHIFT_NO_EVIDENCE)
    result, _f, reverse = _recovery(status_strips, _maja_reverse())
    assert result.delta_px is None
    assert reverse is None, "only SHIFT_NO_CONSENSUS earns a second direction at all"


def _forced_no_consensus(strips):
    """A bank held at SHIFT_NO_CONSENSUS so ONE clause of the cross-direction rule is isolated.

    The rule's clauses overlap by design -- several of them reject the same pathological bank --
    so a fixture aimed at one of them is easily rejected by another and passes for the wrong
    reason. Driving the helper directly, with the surrounding preconditions held fixed, is what
    makes each mutation test mean what its name says.
    """
    return dataclasses.replace(_resolve(tuple(strips)),
                               status=frameshift.SHIFT_NO_CONSENSUS, delta_px=None)


def _cross(forward, reverse, min_agreeing=None):
    return frameshift._cross_direction_quorum_shift(
        forward, reverse, tolerance=frameshift._AGREEMENT_TOLERANCE_PX,
        min_agreeing=frameshift._MIN_AGREEING_STRIPS if min_agreeing is None else min_agreeing,
        pin_margin=frameshift._PIN_MARGIN_PX)


def test_a_cluster_neither_direction_centred_on_is_refused():
    """The median clause, and it is what keeps a video correspondence out.

    Both banks refuse; +200 is exact, seen from BOTH directions and the only such group; the two
    medians even mirror each other at +-450. It is still refused, because +200 is not the number
    either bank centred on -- which is precisely the shape of the Maja video's own +-1178
    correspondence.
    """
    forward = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, 200),
        _strip(500, frameshift.STRIP_MATCHED, 200),
        _strip(700, frameshift.STRIP_MATCHED, 700),
        _strip(900, frameshift.STRIP_MATCHED, 800),
    ))
    reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -200),
        _strip(500, frameshift.STRIP_MATCHED, -700),
    ))
    # The mirror clause is satisfied, so ONLY "the candidate is the median" can be what refuses.
    assert frameshift._matched_median(forward) == 450
    assert frameshift._matched_median(reverse) == -450
    assert _cross(forward, reverse) is None


def test_both_banks_must_centre_on_the_same_translation():
    """The mirror clause. Here +400 IS the forward bank's median and is witnessed both ways, but
    the reverse bank centred somewhere else, so the two are not describing one page."""
    forward = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, 300),
        _strip(500, frameshift.STRIP_MATCHED, 400),
        _strip(700, frameshift.STRIP_MATCHED, 400),
        _strip(900, frameshift.STRIP_MATCHED, 500),
    ))
    reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -400),
        _strip(500, frameshift.STRIP_MATCHED, -100),
        _strip(700, frameshift.STRIP_MATCHED, 0),
    ))
    assert frameshift._matched_median(forward) == 400, "the candidate IS the forward median"
    assert frameshift._matched_median(reverse) == -100, "but the reverse bank centred elsewhere"
    assert _cross(forward, reverse) is None
    # Move the reverse bank's own centre onto it and the same evidence measures.
    agreeing_reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -400),
        _strip(500, frameshift.STRIP_MATCHED, -300),
        _strip(700, frameshift.STRIP_MATCHED, -500),
    ))
    assert _cross(forward, agreeing_reverse) == 400


def test_a_shift_only_one_direction_witnesses_is_refused():
    """Both banks must see it — a head-count inside one bank is what the quorum already governs.

    Driven against the helper rather than the pipeline: a bank with three pixel-exact agreeing
    strips is normally rescued before this rule is reached (by `_resolve`'s own bimodal repair,
    or by the reverse-measured branch), so reaching this clause through `estimate_shift` would
    take a bank the estimator does not produce. The clause is defence in depth and is tested as
    such.
    """
    forward = dataclasses.replace(
        _resolve((_strip(300, frameshift.STRIP_MATCHED, 400),
                  _strip(500, frameshift.STRIP_MATCHED, 400),
                  _strip(700, frameshift.STRIP_MATCHED, 400))),
        status=frameshift.SHIFT_NO_CONSENSUS, delta_px=None)
    reverse = dataclasses.replace(
        _resolve((_strip(300, frameshift.STRIP_MATCHED, -300),
                  _strip(500, frameshift.STRIP_MATCHED, -500))),
        status=frameshift.SHIFT_NO_CONSENSUS, delta_px=None)
    # Both medians line up, so ONLY the missing reverse witness can be what refuses it.
    assert frameshift._matched_median(forward) == 400
    assert frameshift._matched_median(reverse) == -400
    assert frameshift._cross_direction_quorum_shift(
        forward, reverse, tolerance=frameshift._AGREEMENT_TOLERANCE_PX,
        min_agreeing=frameshift._MIN_AGREEING_STRIPS,
        pin_margin=frameshift._PIN_MARGIN_PX) is None
    # ...and with one of those three votes coming from the other bank instead, it measures.
    shared_reverse = dataclasses.replace(
        _resolve((_strip(300, frameshift.STRIP_MATCHED, -400),
                  _strip(500, frameshift.STRIP_MATCHED, -300),
                  _strip(700, frameshift.STRIP_MATCHED, -500))),
        status=frameshift.SHIFT_NO_CONSENSUS, delta_px=None)
    two_forward = dataclasses.replace(
        _resolve((_strip(300, frameshift.STRIP_MATCHED, 400),
                  _strip(500, frameshift.STRIP_MATCHED, 400),
                  _strip(700, frameshift.STRIP_MATCHED, 300),
                  _strip(900, frameshift.STRIP_MATCHED, 500))),
        status=frameshift.SHIFT_NO_CONSENSUS, delta_px=None)
    assert frameshift._cross_direction_quorum_shift(
        two_forward, shared_reverse, tolerance=frameshift._AGREEMENT_TOLERANCE_PX,
        min_agreeing=frameshift._MIN_AGREEING_STRIPS,
        pin_margin=frameshift._PIN_MARGIN_PX) == 400


def test_an_inexact_cross_direction_group_is_refused():
    """Spread, not size, is the discriminator: +630/+631 is not one rigid translation."""
    forward = tuple(
        _strip(s.y0, s.state, 631 if s.delta_px == 630 and s.y0 == 1152 else s.delta_px,
               score=s.score, search=s.search)
        for s in _maja_forward())
    result, _f, _r = _recovery(forward, _maja_reverse())
    assert result.delta_px is None


def test_a_pinned_bound_in_either_direction_vetoes_the_rescue():
    """Every pinned strip's lower bound still constrains the answer, in BOTH banks."""
    veto = _strip(2004, frameshift.STRIP_PINNED, 900, search=(0, 900))
    result, _f, _r = _recovery(_maja_forward() + (veto,), _maja_reverse())
    assert result.delta_px is None, "forward pins say the content went at least 900px"
    result, _f, _r = _recovery(_maja_forward(), _maja_reverse() + (veto,))
    assert result.delta_px is None, "and the reverse bank's pins bind just as hard"


@pytest.mark.parametrize("delta", [900, 899, 892])
def test_a_near_bound_pin_vetoes_the_cross_direction_rescue_too(delta):
    """The pin veto is one of only two evidence clauses this rescue has beyond the median test,
    so the 2026-09-04 exact-equality gap mattered most here. `_search_strip` calls a strip pinned
    anywhere in `[hi - _PIN_MARGIN_PX, hi]`, so all three of these are the same statement — "the
    content went at least this far" — and all three must refuse the +630 the rescue would
    otherwise return. Only the first was ever tested; 899 is the case reproduced against the
    shipped code, which returned True and imposed no bound at all."""
    veto = _strip(2004, frameshift.STRIP_PINNED, delta, search=(0, 900))
    # The premise: this is a state `_search_strip` really produces.
    assert delta >= 900 - frameshift._PIN_MARGIN_PX
    # Control: the identical run without the pin measures +630, so the pin is doing the work.
    assert _recovery(_maja_forward(), _maja_reverse())[0].delta_px == 630

    result, _f, _r = _recovery(_maja_forward() + (veto,), _maja_reverse())
    assert result.delta_px is None, "forward pins say the content went at least this far"
    result, _f, _r = _recovery(_maja_forward(), _maja_reverse() + (veto,))
    assert result.delta_px is None, "and the reverse bank's pins bind just as hard"


def test_the_rescue_cannot_answer_outside_the_trust_window():
    """A shift the ordinary path would have reported as saturated is not measured here either.

    Every other clause admits +1500: it is exact, unique, cross-witnessed and the shared median.
    The window is the only thing that refuses it.
    """
    forward = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, 1500, search=(-3000, 3000)),
        _strip(500, frameshift.STRIP_MATCHED, 1500, search=(-3000, 3000)),
    ))
    reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -1500, search=(-3000, 3000)),
        _strip(500, frameshift.STRIP_MATCHED, -1500, search=(-3000, 3000)),
    ))
    assert forward.trust_window_px == 900
    assert frameshift._matched_median(forward) == 1500
    assert frameshift._matched_median(reverse) == -1500
    assert _cross(forward, reverse) is None
    # The identical bank inside the window measures, so nothing else here is doing the work.
    near = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, 500, search=(-3000, 3000)),
        _strip(500, frameshift.STRIP_MATCHED, 500, search=(-3000, 3000)),
    ))
    near_reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -500, search=(-3000, 3000)),
        _strip(500, frameshift.STRIP_MATCHED, -500, search=(-3000, 3000)),
    ))
    assert _cross(near, near_reverse) == 500


def test_a_reverse_bank_that_is_not_merely_short_of_quorum_is_never_pooled(monkeypatch):
    """The precondition is a GATE on reaching the rule at all, not one more clause inside it.

    Asserted by making the helper unreachable: a reverse direction that saturated or saw nothing
    is a different statement from "two strips agreed and a third was missing", and its strips
    must never be pooled with the forward bank's, whatever they happen to say.
    """
    def unreachable(*_args, **_kwargs):
        raise AssertionError("the cross-direction rule must not be consulted for this pair")

    monkeypatch.setattr(frameshift, "_cross_direction_quorum_shift", unreachable)
    for other in ([_strip(300 + i * 200, frameshift.STRIP_MATCHED, 1500, search=(-2000, 2000))
                   for i in range(5)],
                  [_strip(300 + i * 200, frameshift.STRIP_WEAK, None) for i in range(5)]):
        assert _resolve(other).status in (frameshift.SHIFT_BEYOND_WINDOW,
                                          frameshift.SHIFT_NO_EVIDENCE)
        result, _f, _r = _recovery(_maja_forward(), other)
        assert result.delta_px is None


def test_two_exact_cross_direction_clusters_are_ambiguous_and_refused():
    """Uniqueness, for the cross-direction rule specifically.

    Both +400 and -400 are pixel-exact, quorate and witnessed from both banks, and the medians
    line up on +400. A second such cluster is a genuine ambiguity about which of two things
    moved, and is refused rather than broken by taking the larger or the central one.
    """
    # -400 is the SHARED median of both banks, so every other clause admits it: taking "the"
    # exact group instead of refusing an ambiguous bank would return it. Only uniqueness stands
    # between this bank and a wrong answer.
    forward = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -400),
        _strip(500, frameshift.STRIP_MATCHED, -400),
        _strip(700, frameshift.STRIP_MATCHED, -400),
        _strip(900, frameshift.STRIP_MATCHED, 400),
        _strip(1100, frameshift.STRIP_MATCHED, 400),
    ))
    reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -400),
        _strip(500, frameshift.STRIP_MATCHED, 400),
        _strip(700, frameshift.STRIP_MATCHED, 400),
    ))
    assert frameshift._matched_median(forward) == -400
    assert frameshift._matched_median(reverse) == 400
    assert _cross(forward, reverse) is None
    # ...and with the +400 cluster dropped below quorum it is no longer a competing answer.
    unambiguous = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, -400),
        _strip(500, frameshift.STRIP_MATCHED, -400),
        _strip(700, frameshift.STRIP_MATCHED, -400),
        _strip(900, frameshift.STRIP_MATCHED, 400),
    ))
    one_reverse = _forced_no_consensus((
        _strip(300, frameshift.STRIP_MATCHED, 400),
        _strip(500, frameshift.STRIP_MATCHED, 500),
        _strip(700, frameshift.STRIP_MATCHED, 300),
    ))
    assert frameshift._matched_median(unambiguous) == -400
    assert frameshift._matched_median(one_reverse) == 400
    assert _cross(unambiguous, one_reverse) == -400


def test_a_reverse_direction_that_refuses_for_any_other_reason_ends_it():
    """Only SHIFT_NO_CONSENSUS in BOTH directions opens the cross-direction branch.

    A reverse bank that saturated or saw nothing is not a bank whose head-count merely fell
    short, and its strips must not be pooled with the forward bank's.
    """
    saturated = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 1500, search=(-2000, 2000))
                 for i in range(5)]
    assert _resolve(saturated).status == frameshift.SHIFT_BEYOND_WINDOW
    result, forward, reverse = _recovery(_maja_forward(), saturated)
    assert forward.status == frameshift.SHIFT_NO_CONSENSUS, "the forward side did open it"
    assert reverse.status == frameshift.SHIFT_BEYOND_WINDOW
    assert result.delta_px is None, "a saturated reverse bank ends the pair, it never pools"
