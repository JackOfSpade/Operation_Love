"""The affirmative scroll-top gate (operation_love/drivers/scroll_top.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures the module's
constants were measured against are real people's dating profiles and are gitignored
(ops/calibration/, .gitignore:26); only geometry and counts from them appear anywhere in this
repo. So the fixtures paint a 1080x2400 frame from first principles and fill the rows and columns
`HINGE_SPEC.identity_band` cuts with a pattern the test chose, which makes the resulting
fingerprint — and therefore every distance asserted below — known BY CONSTRUCTION rather than
approximated.

The reference the gate compares against is a PARAMETER (`fingerprint=`), which is what lets these
tests be synthetic end to end: a test builds a frame, reads its own fingerprint back through the
shipped decode, and then hands the gate a reference offset by an exact number of grey levels. A
flat band downsamples to exactly its own value under any resampling filter, so "offset every cell
by 5" really is a mean-abs distance of 5.000 and the three outcomes can be driven to their
boundaries rather than near them.

The shipped `_SCROLL_TOP_BAND_FINGERPRINT` is exercised too, but only ever against synthetic
frames — a blank screen, a black screen — because the property worth pinning about it is that
nothing empty may ever be mistaken for Hinge's filter-chips row.

Following the house pattern, every positive is paired with a negative plus a control that proves
WHICH mechanism did the excluding.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import hinge, scroll_top
from operation_love.drivers.hinge import HINGE_SPEC

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_IB = HINGE_SPEC.identity_band             # (0.10, 0.048, 0.80, 0.094) -> x 108..864, y 115..226
_GRID = scroll_top._FINGERPRINT_GRID       # (16, 4)
_CONFIRM = scroll_top._CONFIRM_MAX_DIST    # 3.0
_REFUTE = scroll_top._REFUTE_MIN_DIST      # 9.0

# Mid-grey, so a reference offset by +-12 levels stays inside 0..255 and the offset the test asked
# for is the offset the metric sees. A band painted near white would clip and quietly shrink the
# distance under test.
_BASE = 128


def _band_rows_cols():
    """The exact pixel rect `_band` will cut for `identity_band`, mirroring its own rounding."""
    x0, y0, x1, y1 = _IB
    return round(y0 * _H), round(y1 * _H), round(x0 * _W), round(x1 * _W)


def _frame(band_paint=None, *, background=40) -> bytes:
    """A decodable 1080x2400 greyscale PNG whose identity band is painted by `band_paint`.

    `band_paint` is called with the (rows, cols) sub-array of the identity band and mutates it in
    place; None leaves the band at `background` like the rest of the frame. The background is
    deliberately NOT the band value, so a test that accidentally measured the wrong rect would
    read a wildly different fingerprint rather than a plausible one.
    """
    arr = np.full((_H, _W), background, dtype="uint8")
    r0, r1, c0, c1 = _band_rows_cols()
    if band_paint is not None:
        band_paint(arr[r0:r1, c0:c1])
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _flat(value):
    def paint(band):
        band[:, :] = value
    return paint


def _gradient(top_value, per_row, shift_rows=0):
    """A vertical gradient of `per_row` grey levels per row, optionally moved down `shift_rows`.

    Used for the layout-drift case: shifting a gradient by N rows changes EVERY cell of the
    fingerprint by exactly `N * per_row`, so "the same chrome redrawn a few pixels lower" has an
    exactly known cost instead of a plausible one.
    """
    def paint(band):
        rows = np.arange(band.shape[0], dtype="float64") - shift_rows
        band[:, :] = np.clip(top_value + rows * per_row, 0, 255).astype("uint8")[:, None]
    return paint


def _fingerprint_of(frame) -> tuple:
    return scroll_top.band_fingerprint(frame, identity_band=_IB, grid=_GRID)


def _offset(fingerprint, delta) -> tuple:
    """The same fingerprint with every cell moved `delta` levels — a reference at an exactly
    known mean-abs distance from the frame it came from."""
    return tuple(int(v) + delta for v in fingerprint)


# =====================================================================================
# The three outcomes.
# =====================================================================================

def test_a_band_matching_the_reference_confirms_the_top():
    frame = _frame(_flat(_BASE))
    verdict = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=_fingerprint_of(frame), grid=_GRID)

    assert verdict.state == scroll_top.SCROLL_TOP_CONFIRMED
    assert verdict.confirmed is True
    assert verdict.refuted is False and verdict.unknown is False
    assert verdict.distance == 0.0
    assert verdict.band == tuple(_IB) and verdict.grid == _GRID
    assert "filter-chips" in verdict.reason


def test_a_band_far_from_the_reference_confirms_NOT_at_the_top():
    frame = _frame(_flat(_BASE))
    reference = _offset(_fingerprint_of(frame), 12)      # 12.0 > refute_min 9.0

    verdict = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.state == scroll_top.SCROLL_TOP_REFUTED
    assert verdict.refuted is True
    assert verdict.confirmed is False and verdict.unknown is False
    assert verdict.distance == 12.0


def test_a_band_between_the_two_bounds_answers_cannot_tell():
    """The outcome the gate exists to keep distinct: 5.0 is too far to be the filter-chips row
    and too near to be positively something else."""
    frame = _frame(_flat(_BASE))
    reference = _offset(_fingerprint_of(frame), 5)       # 3.0 < 5.0 < 9.0

    verdict = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.state == scroll_top.SCROLL_TOP_UNKNOWN
    assert verdict.unknown is True
    assert verdict.confirmed is False and verdict.refuted is False
    assert verdict.distance == 5.0
    assert "dead zone" in verdict.reason


def test_cannot_tell_is_never_reported_as_at_the_top():
    """The single failure this gate must not have. `.confirmed` is what a caller passes as
    `build_item_index(at_scroll_top=...)`, so it must be False for BOTH non-confirmations."""
    frame = _frame(_flat(_BASE))
    base = _fingerprint_of(frame)

    for delta, expected in ((5, scroll_top.SCROLL_TOP_UNKNOWN),
                            (12, scroll_top.SCROLL_TOP_REFUTED)):
        verdict = scroll_top.confirm_scroll_top(
            frame, identity_band=_IB, fingerprint=_offset(base, delta), grid=_GRID)
        assert verdict.state == expected
        assert verdict.confirmed is False


def test_the_two_bounds_are_inclusive_on_the_confirming_side_and_on_the_refuting_side():
    """Exactly at `confirm_max` still confirms; exactly at `refute_min` refutes. Pinned because
    the dead zone is open at both ends and an off-by-one there silently moves the gate."""
    frame = _frame(_flat(_BASE))
    base = _fingerprint_of(frame)

    at_confirm = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=_offset(base, int(_CONFIRM)), grid=_GRID)
    just_past = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=_offset(base, int(_CONFIRM) + 1), grid=_GRID)
    at_refute = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=_offset(base, int(_REFUTE)), grid=_GRID)
    just_under = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=_offset(base, int(_REFUTE) - 1), grid=_GRID)

    assert at_confirm.distance == _CONFIRM and at_confirm.confirmed
    assert just_past.unknown
    assert at_refute.distance == _REFUTE and at_refute.refuted
    assert just_under.unknown


def test_no_declared_identity_band_answers_cannot_tell_rather_than_raising():
    """`identity_band` is optional on AndroidAppSpec, so an app without one is a thing this gate
    cannot judge — not a broken install. It must still never read as a top."""
    verdict = scroll_top.confirm_scroll_top(_frame(_flat(_BASE)), identity_band=None)

    assert verdict.state == scroll_top.SCROLL_TOP_UNKNOWN
    assert verdict.confirmed is False
    assert verdict.distance is None and verdict.band is None
    assert "no identity_band declared" in verdict.reason


# =====================================================================================
# Fail loud: a caller cannot proceed on anything but a confirmation by accident.
# =====================================================================================

def test_the_verdict_has_no_truth_value():
    """`if confirm_scroll_top(...):` is the most natural way to write the off-by-N this module
    exists to prevent — a frozen dataclass is truthy, so it would read every outcome as a top."""
    frame = _frame(_flat(_BASE))
    base = _fingerprint_of(frame)

    for delta in (0, 5, 12):
        verdict = scroll_top.confirm_scroll_top(
            frame, identity_band=_IB, fingerprint=_offset(base, delta), grid=_GRID)
        with pytest.raises(TypeError) as excinfo:
            bool(verdict)
        assert "no truth value" in str(excinfo.value)
        assert verdict.state in str(excinfo.value)


def test_require_scroll_top_returns_the_verdict_on_a_confirmed_top():
    frame = _frame(_flat(_BASE))
    verdict = scroll_top.require_scroll_top(
        frame, identity_band=_IB, fingerprint=_fingerprint_of(frame), grid=_GRID)

    assert verdict.confirmed is True


@pytest.mark.parametrize("delta,state", [(5, scroll_top.SCROLL_TOP_UNKNOWN),
                                         (12, scroll_top.SCROLL_TOP_REFUTED)])
def test_require_scroll_top_raises_on_cannot_tell_and_on_refuted_alike(delta, state):
    """Doc 5.5: "treat failure to confirm as a hard stop". To a caller about to count hearts,
    "cannot tell" and "definitely not" are the same thing — neither may proceed."""
    frame = _frame(_flat(_BASE))
    reference = _offset(_fingerprint_of(frame), delta)

    with pytest.raises(scroll_top.ScrollTopUnconfirmed) as excinfo:
        scroll_top.require_scroll_top(
            frame, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert excinfo.value.verdict.state == state
    assert "off-by-N" in str(excinfo.value)


def test_require_scroll_top_raises_when_no_identity_band_is_declared():
    with pytest.raises(scroll_top.ScrollTopUnconfirmed) as excinfo:
        scroll_top.require_scroll_top(_frame(_flat(_BASE)), identity_band=None)

    assert excinfo.value.verdict.state == scroll_top.SCROLL_TOP_UNKNOWN


# =====================================================================================
# "Could not look" is not a verdict.
# =====================================================================================

def test_undecodable_bytes_raise_rather_than_answering_cannot_tell():
    """`SCROLL_TOP_UNKNOWN` means "I looked and could not tell". A broken decode must not be able
    to produce it, or a missing PIL degrades the gate into a stream of plausible refusals with no
    single loud failure anywhere — segment.SegmentationError draws the same line."""
    with pytest.raises(scroll_top.ScrollTopError) as excinfo:
        scroll_top.confirm_scroll_top(b"not a png at all", identity_band=_IB)

    assert "could not look" in str(excinfo.value)


def test_a_band_the_driver_could_not_read_raises_rather_than_answering(monkeypatch):
    """The control for the test above, isolating the mechanism: `_band` returning None is what
    "could not look" means, whatever caused it."""
    monkeypatch.setattr(hinge, "_band", lambda *a, **k: None)

    with pytest.raises(scroll_top.ScrollTopError):
        scroll_top.confirm_scroll_top(_frame(_flat(_BASE)), identity_band=_IB)


def test_a_fingerprint_that_does_not_match_its_grid_raises():
    with pytest.raises(scroll_top.ScrollTopError) as excinfo:
        scroll_top.confirm_scroll_top(
            _frame(_flat(_BASE)), identity_band=_IB, fingerprint=(1, 2, 3), grid=_GRID)

    assert "needs 64" in str(excinfo.value)


def test_bounds_that_collapse_the_dead_zone_raise():
    """With `confirm_max >= refute_min` every distance is classified and the third outcome
    becomes unreachable, which is the gate silently reverting to a two-valued answer."""
    with pytest.raises(scroll_top.ScrollTopError) as excinfo:
        scroll_top.confirm_scroll_top(
            _frame(_flat(_BASE)), identity_band=_IB, confirm_max=9.0, refute_min=9.0)

    assert "cannot tell" in str(excinfo.value)


def test_fingerprint_distance_refuses_mismatched_and_empty_inputs():
    with pytest.raises(scroll_top.ScrollTopError):
        scroll_top.fingerprint_distance((1, 2, 3), (1, 2))
    with pytest.raises(scroll_top.ScrollTopError):
        scroll_top.fingerprint_distance((), ())


# =====================================================================================
# The signal itself: what it tolerates, what it does not, and what the shipped constant is.
# =====================================================================================

def test_the_same_chrome_redrawn_a_few_pixels_lower_still_confirms():
    """The reason the grid is 16x4 rather than the driver's 64x16: a banner appearing or a font
    metric changing moves this strip a few px, and that must read as "the same chrome, moved",
    not as "not at top". A 0.5-level-per-row gradient shifted 4 rows costs exactly 2.0."""
    reference = _fingerprint_of(_frame(_gradient(160, 0.5)))
    shifted = _frame(_gradient(160, 0.5, shift_rows=4))

    verdict = scroll_top.confirm_scroll_top(
        shifted, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.confirmed is True
    assert verdict.distance == pytest.approx(2.0, abs=0.25)


def test_the_control_a_shift_large_enough_to_matter_does_not_confirm():
    """A 6-row shift of a 2-levels-per-row gradient: 12.0. The tolerance above is a bound on the
    CHANGE, not an amnesty on shifting — otherwise the test above would be passing because the
    gate ignores position. (Both gradients are chosen to span 15..225, so nothing clips and the
    shift really does move every cell by the full amount.)"""
    reference = _fingerprint_of(_frame(_gradient(15, 2.0)))
    shifted = _frame(_gradient(15, 2.0, shift_rows=6))

    verdict = scroll_top.confirm_scroll_top(
        shifted, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.refuted is True
    assert verdict.distance == pytest.approx(12.0, abs=0.5)


@pytest.mark.parametrize("value", [255, 250, 0])
def test_a_blank_screen_is_never_confirmed_as_the_top(value):
    """Against the SHIPPED reference, not a synthetic one. The filter-chips row is mostly white,
    so a blank white framebuffer is the nearest thing to it anything has ever measured — and it
    still must not confirm."""
    verdict = scroll_top.confirm_scroll_top(_frame(_flat(value)), identity_band=_IB)

    assert verdict.confirmed is False


def test_the_shipped_fingerprint_matches_its_grid_and_is_in_range():
    fingerprint = scroll_top._SCROLL_TOP_BAND_FINGERPRINT

    assert len(fingerprint) == _GRID[0] * _GRID[1] == 64
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in fingerprint)


def test_the_refute_bound_is_the_apps_own_change_threshold():
    """`_REFUTE_MIN_DIST` is a declared COPY of `HINGE_SPEC.change_threshold` — it cannot be
    imported, because hinge imports this module's package and not the other way round. This is
    what stops the copy drifting away from the original."""
    assert scroll_top._REFUTE_MIN_DIST == HINGE_SPEC.change_threshold


def test_the_metric_is_the_same_one_the_driver_already_uses_on_this_band():
    """`fingerprint_distance` is pure Python; `hinge._band_dist` is numpy. They must agree, or
    every distance recorded by this gate would be incomparable to the measurements in
    `HINGE_SPEC.identity_band`'s own comment."""
    a, b = _frame(_flat(_BASE)), _frame(_flat(_BASE + 7))
    fa, fb = _fingerprint_of(a), _fingerprint_of(b)
    na = hinge._band(a, _IB, _GRID)
    nb = hinge._band(b, _IB, _GRID)

    assert scroll_top.fingerprint_distance(fa, fb) == pytest.approx(
        hinge._band_dist(na, nb), abs=1e-9)


def test_band_fingerprint_reads_the_declared_band_at_the_declared_grid(monkeypatch):
    """The control that proves the gate goes through the driver's ONE decode of this rect, with
    the rect and the grid it was given — a second crop-and-resize here would be free to disagree
    with `_identity_of`'s by a grey level, and the confirm bound is only 3 wide."""
    seen = {}
    real = hinge._band

    def spy(frame, rect, size=hinge._IDENTITY_DS):
        seen["rect"], seen["size"] = rect, size
        return real(frame, rect, size)

    monkeypatch.setattr(hinge, "_band", spy)
    scroll_top.band_fingerprint(_frame(_flat(_BASE)), identity_band=_IB, grid=(8, 2))

    assert seen == {"rect": _IB, "size": (8, 2)}


def test_the_default_grid_and_reference_are_used_when_the_caller_names_neither():
    """A caller that passes only the band gets the calibrated constants, so the gate cannot be
    accidentally run against whatever reference happened to be lying around."""
    frame = _frame(_flat(_BASE))
    explicit = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=scroll_top._SCROLL_TOP_BAND_FINGERPRINT,
        grid=scroll_top._FINGERPRINT_GRID)
    implicit = scroll_top.confirm_scroll_top(frame, identity_band=_IB)

    assert implicit == explicit
