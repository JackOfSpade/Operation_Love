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


def test_confidence_counts_only_the_strips_that_could_have_seen_the_answer():
    """A strip whose searchable range does not reach the consensus is not evidence against it —
    it never had the chance. A strip that COULD have seen it and came back weak is. The two
    banks below differ only in that one strip's `search` range, and that must move the
    denominator by one."""
    voters = [_strip(300 + i * 200, frameshift.STRIP_MATCHED, 363) for i in range(3)]

    out_of_reach = _resolve(voters + [_strip(2000, frameshift.STRIP_WEAK, None,
                                             search=(-1000, 100), score=0.2)])
    in_reach = _resolve(voters + [_strip(2000, frameshift.STRIP_WEAK, None,
                                         search=(-1000, 1000), score=0.2)])

    assert out_of_reach.eligible == 3 and out_of_reach.confidence == 1.0
    assert in_reach.eligible == 4 and in_reach.confidence == pytest.approx(0.75)
    assert out_of_reach.delta_px == in_reach.delta_px == 363


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
