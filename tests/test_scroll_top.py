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
    """An 8-row shift of a 2-levels-per-row gradient: 16.0 at the fixed dy=0 crop. The tolerance
    above is a bound on the CHANGE, not an amnesty on shifting — otherwise the test above would
    be passing because the gate ignores position. (Both gradients are chosen to span 15..225, so
    nothing clips and the shift really does move every cell by the full amount.)

    8 rows rather than the previous 6: this fixture paints the gradient ONLY inside the exact
    identity-band rect with flat background on both sides, so `_ALIGNMENT_SEARCH_PX`'s sweep can
    trade a few of the shifted gradient's most-displaced rows for background and claw back part
    of a 6-row shift's distance (12.0 -> 7.75, landing in the dead zone) — an edge effect of this
    fixture's sharp-edged patch, not of a real device frame where content extends continuously
    past the crop. 8 rows keeps the post-search distance safely past `refute_min` (16.0 -> 11.0),
    so this control still demonstrates what it always did: a shift this size is never tolerated.
    """
    reference = _fingerprint_of(_frame(_gradient(15, 2.0)))
    shifted = _frame(_gradient(15, 2.0, shift_rows=8))

    verdict = scroll_top.confirm_scroll_top(
        shifted, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.refuted is True
    assert verdict.distance == pytest.approx(11.0, abs=0.5)


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


def test_all_shipped_fingerprint_variants_match_grid_and_are_in_range():
    for fp in scroll_top._SCROLL_TOP_BAND_FINGERPRINTS:
        assert len(fp) == _GRID[0] * _GRID[1] == 64
        assert all(isinstance(v, int) and 0 <= v <= 255 for v in fp)


def test_either_shipped_variant_confirms_scroll_top_under_default_matching(monkeypatch):
    for fp in scroll_top._SCROLL_TOP_BAND_FINGERPRINTS:
        arr = np.array(fp, dtype="uint8").reshape(_GRID[1], _GRID[0])
        monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size, arr=arr: arr)
        verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)
        assert verdict.confirmed is True
        assert verdict.distance == 0.0


@pytest.mark.parametrize("fp", [
    scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS,
    scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED,
    scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_SELECTED_CURRENT,
    scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_CURRENT,
])
def test_hingex_signals_chip_variants_are_default_scroll_top_candidates(monkeypatch, fp):
    """Selected and unselected HingeX Signals chips change the fixed filter strip.

    The accompanying Most Compatible banner is below ``identity_band``.  This regression
    confirms the actual top-of-card signal that must remain registered, without placing a real
    dating-profile screenshot in the test suite.
    """
    assert fp in scroll_top._SCROLL_TOP_BAND_FINGERPRINTS
    arr = np.array(fp, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.confirmed is True
    assert verdict.distance == 0.0


def test_observed_selected_signals_top_is_not_rejected_into_the_dead_zone(monkeypatch):
    """Regression for the live Signals top that formerly measured 3.281 and was refused.

    This is the filter-strip fingerprint only: no profile pixels or identifying data are stored.
    """
    observed = (
        254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 200, 168, 168, 203, 251, 232, 232, 232, 248, 238, 232, 233, 231, 242, 246,
        227, 87, 78, 75, 85, 223, 237, 237, 240, 232, 235, 230, 231, 239, 235, 233,
        215, 88, 126, 120, 89, 218, 221, 196, 232, 233, 237, 198, 193, 226, 235, 236,
    )
    arr = np.array(observed, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    assert scroll_top.fingerprint_distance(
        observed, scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS) == 3.28125
    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.confirmed is True
    assert verdict.distance == 0.0


@pytest.mark.parametrize("fp", [
    scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL,
    scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL_B,
])
def test_hingex_signals_dark_pill_variants_are_default_scroll_top_candidates(monkeypatch, fp):
    """Dark/charcoal Signals pill rendering (Variants 7 & 8) must be registered candidates.

    Regression for the 2026-08-19 Lana capture (run ce1de851af2e) where the dark-pill
    rendering measured 17.344--17.750 from all existing variants and was misclassified as
    ``confirmed_not_top``, stalling observe mode for 5m33s while the card was genuinely at
    its scroll top.  This test stores only the 16x4 greyscale filter-strip fingerprint,
    never any profile pixels.
    """
    assert fp in scroll_top._SCROLL_TOP_BAND_FINGERPRINTS
    arr = np.array(fp, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.confirmed is True
    assert verdict.distance == 0.0


def test_dark_pill_frames_measure_within_confirm_bound_of_each_other():
    """The two dark-pill calibration frames are 2.594 apart — within the 3.0 confirm bound.

    Regression: these two frames from the same run must be treated as the same layout and
    never straddle the dead zone.  This records only the inter-frame distance measurement
    (no profile pixels — both fingerprints are filter-strip chrome only).
    """
    d = scroll_top.fingerprint_distance(
        scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL,
        scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL_B,
    )
    assert d < scroll_top._CONFIRM_MAX_DIST, (
        f"Dark-pill frames are {d:.3f} apart, which exceeds the {scroll_top._CONFIRM_MAX_DIST} "
        "confirm bound — one would fall in the dead zone and stall observe mode again"
    )


def test_observed_current_age_height_top_is_not_refused_in_dead_zone(monkeypatch):
    """Current white Age/Height chips are a known top, not an ambiguous scroll state.

    Regression for the settled Val capture in the 2026-08-19 ``c679dfb4e458`` observe run.
    Its top-of-card filter strip was 7.203 from the old nearest candidate, which is deliberately
    inside the dead zone; the registered fingerprint is chrome-only calibration evidence.
    """
    observed = scroll_top._SCROLL_TOP_BAND_FINGERPRINT_AGE_HEIGHT_CURRENT
    arr = np.array(observed, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    assert observed in scroll_top._SCROLL_TOP_BAND_FINGERPRINTS
    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.confirmed is True
    assert verdict.distance == 0.0


def test_observed_signals_purple_banner_top_is_not_refused_in_dead_zone(monkeypatch):
    """A current selected-Signals filter strip is known top chrome, not an unknown scroll.

    Regression for the visibly top-of-card 2026-08-20 Jenny capture. It was 8.953 from the
    previous closest candidate, just inside the 3..9 dead zone. The fixture is only the 16x4
    profile-independent filter-strip fingerprint; no dating-profile pixels are stored here.
    """
    observed = (
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        243, 186, 181, 184, 241, 243, 232, 232, 239, 249, 233, 233, 233, 233, 249, 237,
        159,  70,  72,  64, 149, 242, 236, 242, 236, 232, 234, 233, 236, 240, 232, 235,
        138,  99, 130, 102, 130, 246, 190, 214, 234, 235, 220, 182, 206, 231, 232, 235,
    )
    assert observed == scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_PURPLE_BANNER
    assert scroll_top.fingerprint_distance(
        observed, scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_DARK_PILL) == 8.953125
    arr = np.array(observed, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.confirmed is True
    assert verdict.distance == 0.0


def test_observed_hinge_10_0_1_signals_unselected_top_is_not_refused_in_dead_zone(monkeypatch):
    """The Hinge Android app auto-updating 9.134.0 -> 10.0.1 redrew the unselected-Signals chip
    row and pushed it into the dead zone, not past the refute bound.

    Regression for 2026-08-21: three independent Pixel 7a captures taken after the update
    (phone_check3.png, phone_after_abort2.png, phone_idle_check.png) all visibly showed a
    genuine scroll top and all three decoded to this exact fingerprint, 0.000 apart from each
    other -- a stable re-render, not capture jitter. Against Variant 4 (`SIGNALS_UNSELECTED`),
    the nearest fingerprint on file at the time, it measured 6.671875: squarely inside the 3..9
    dead zone, which aborted a live calibration capture twice before this variant was
    registered. The fixture is only the 16x4 profile-independent filter-strip fingerprint; no
    dating-profile pixels are stored here.
    """
    observed = (
        254, 254, 253, 253, 254, 255, 251, 250, 251, 255, 253, 250, 250, 250, 254, 255,
        253, 250, 252, 253, 251, 245, 233, 236, 233, 242, 235, 236, 236, 234, 238, 240,
        247, 241, 214, 216, 244, 241, 228, 222, 236, 233, 237, 215, 215, 232, 236, 236,
        247, 247, 214, 220, 245, 241, 231, 218, 242, 234, 240, 223, 216, 240, 237, 236,
    )
    assert observed == scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_10_0_1
    assert scroll_top.fingerprint_distance(
        observed, scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED) == 6.671875
    assert scroll_top._CONFIRM_MAX_DIST < 6.671875 < scroll_top._REFUTE_MIN_DIST
    arr = np.array(observed, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.confirmed is True
    assert verdict.distance == 0.0


def test_hinge_10_0_1_variant_does_not_pull_a_scrolled_frame_into_the_dead_zone(monkeypatch):
    """The new variant must not narrow the gap for frames that are genuinely scrolled.

    Regression built from the real negative check: on the physical Pixel 7a, composer-open
    (keyboard visible) frames from the same 2026-08-21 session measured 15.1--16.7 from every
    registered fingerprint including this new one -- comfortably above `refute_min`. This test
    pins the mechanism with a synthetic stand-in at exactly the refute bound: a band already
    correctly refuted against the nearest OLD variant must not move into the dead zone just
    because the NEW variant was added, since `confirm_scroll_top` takes the MINIMUM distance
    across all candidates.
    """
    new_variant = scroll_top._SCROLL_TOP_BAND_FINGERPRINT_SIGNALS_UNSELECTED_10_0_1
    # -16 rather than +16: the new variant's cells run up to 255, so a positive offset this size
    # would clip against the uint8 ceiling and silently understate the distance. Subtracting
    # keeps every cell in range (min cell 214 -> 198) while still landing comfortably past
    # `refute_min`, matching the 15.1--16.7 margins measured on the real composer-open frames.
    scrolled = _offset(new_variant, -16)
    arr = np.array(scrolled, dtype="uint8").reshape(_GRID[1], _GRID[0])
    monkeypatch.setattr(hinge, "_band", lambda _frame, _rect, _size: arr)

    verdict = scroll_top.confirm_scroll_top(b"synthetic-frame", identity_band=_IB)

    assert verdict.refuted is True
    assert verdict.confirmed is False
    assert verdict.distance == pytest.approx(16.0, abs=1e-9)


# =====================================================================================
# The bounded vertical alignment search (_ALIGNMENT_SEARCH_PX).
#
# Live failure, 2026-08-22: every profile advance that night landed in the (3.0, 9.0) dead zone
# -- measurements 5.547, 5.625, 5.656, 6.547 -- aborting "hybrid rewind reached an unconfirmed
# scroll-top state" on frames that were visibly at the top. Re-cropping the identity band with a
# small vertical offset (dy=-8px) turned a 5.625 dead-zone measurement into 1.0, comfortably
# inside confirm_max: the filter-chips row was really there, just a few pixels lower than the
# fingerprint's calibration crop -- far below the ~219px minimum scroll gesture and well under
# the ~111px band height, so no gesture could have corrected it and the rewind could only refuse.
# =====================================================================================

def _frame_with_band_at(fill, *, dy: int, background=40) -> bytes:
    """Like `_frame`, but the identity band's content is painted `dy` PIXELS away from the
    nominal `identity_band` crop -- simulating chrome that rendered a few pixels off the position
    the fingerprint was calibrated against, exactly the live 2026-08-22 failure. `confirm_scroll_
    top`'s own (unshifted) crop is untouched by this helper's `dy`; only where the content
    actually sits in the frame moves, so a caller comparing against the content's OWN dy=0
    fingerprint reproduces the live "cannot tell at the nominal crop, confirms once re-aligned"
    shape from first principles.
    """
    arr = np.full((_H, _W), background, dtype="uint8")
    r0, r1, c0, c1 = _band_rows_cols()
    arr[r0 + dy:r1 + dy, c0:c1] = fill
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_a_band_settled_a_few_pixels_off_calibration_confirms_after_the_alignment_search():
    """The live regression. The filter-chips row rendered 8px away from the fingerprint's
    calibration crop -- `_frame_with_band_at(..., dy=-8)` mirrors the 2026-08-22 measurements
    (5.625 at the fixed dy=0 crop, 1.0 once re-cropped 8px higher) closely enough that the
    synthetic distances land in the same shape: 5.75 at dy=0 (dead zone, would have aborted the
    rewind exactly like that night), 0.0 at dy=-8 (comfortably inside confirm_max)."""
    reference = _fingerprint_of(_frame_with_band_at(_BASE, dy=0))
    live_frame = _frame_with_band_at(_BASE, dy=-8)

    at_nominal_crop = scroll_top.fingerprint_distance(_fingerprint_of(live_frame), reference)
    assert at_nominal_crop == pytest.approx(5.75, abs=1e-9)
    assert scroll_top._CONFIRM_MAX_DIST < at_nominal_crop < scroll_top._REFUTE_MIN_DIST

    verdict = scroll_top.confirm_scroll_top(
        live_frame, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.state == scroll_top.SCROLL_TOP_CONFIRMED
    assert verdict.confirmed is True
    assert verdict.distance == pytest.approx(0.0, abs=1e-9)
    assert verdict.alignment_offset_px == -8
    assert "-8px" in verdict.reason


def test_the_alignment_search_never_pulls_a_genuinely_scrolled_band_into_confirmation():
    """Safety pin: the whole gate exists so navigation never counts hearts on somebody else's
    card, so a REFUTED band must stay refuted across the ENTIRE sweep, not just at dy=0.

    Constructed so no offset helps: the frame's band is flat at `_BASE` over its full painted
    extent (not just the declared identity_band rect plus a shifted patch), so re-cropping it at
    any offset only trades in MORE of the surrounding `background` -- which sits even further
    from a reference offset positive from `_BASE`, never closer. This reproduces the real
    negative check (a scrolled/composer-open frame measured 10.031-16.7 at dy=0) with a
    synthetic band whose distance is provably monotonic in the offset instead of merely observed
    to be so on one capture.
    """
    frame = _frame(_flat(_BASE))
    reference = _offset(_fingerprint_of(frame), 12)      # 12.0 at dy=0, same as the plain test

    for dy in range(-scroll_top._ALIGNMENT_SEARCH_PX, scroll_top._ALIGNMENT_SEARCH_PX + 1):
        seen = (_fingerprint_of(frame) if dy == 0 else
                scroll_top._band_fingerprint_at_offset(frame, identity_band=_IB, grid=_GRID,
                                                       dy_px=dy))
        assert scroll_top.fingerprint_distance(seen, reference) >= 12.0, (
            f"offset {dy:+d}px found a closer match than dy=0 -- the sweep would be pulling a "
            "scrolled frame toward confirmation")

    verdict = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.state == scroll_top.SCROLL_TOP_REFUTED
    assert verdict.refuted is True
    assert verdict.confirmed is False
    assert verdict.distance == 12.0
    assert verdict.alignment_offset_px == 0


def test_the_dead_zone_survives_the_alignment_search():
    """The dead zone is a deliberate THIRD outcome (see the module docstring), not a threshold
    the alignment search is allowed to search its way out of. A band that is ambiguous at every
    offset -- 5.0 to 13.0 across the full +/-12px sweep, never dropping to confirm_max -- must
    still come back `cannot_tell`, exactly as it would with no alignment search at all."""
    frame = _frame(_flat(_BASE))
    reference = _offset(_fingerprint_of(frame), 5)       # 5.0 at dy=0: inside (3.0, 9.0)

    for dy in range(-scroll_top._ALIGNMENT_SEARCH_PX, scroll_top._ALIGNMENT_SEARCH_PX + 1):
        seen = (_fingerprint_of(frame) if dy == 0 else
                scroll_top._band_fingerprint_at_offset(frame, identity_band=_IB, grid=_GRID,
                                                       dy_px=dy))
        assert scroll_top.fingerprint_distance(seen, reference) > scroll_top._CONFIRM_MAX_DIST, (
            f"offset {dy:+d}px dropped to or below confirm_max -- the dead zone must never be "
            "searchable into a false confirmation")

    verdict = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=reference, grid=_GRID)

    assert verdict.state == scroll_top.SCROLL_TOP_UNKNOWN
    assert verdict.unknown is True
    assert verdict.distance == 5.0
    assert verdict.alignment_offset_px == 0
    assert "dead zone" in verdict.reason


def test_alignment_offset_defaults_to_zero_and_is_silent_in_the_reason_when_unsearched():
    """An exact dy=0 match is the overwhelming common case, so the new field and the new reason
    clause must both stay invisible when nothing was searched -- reusing the plain confirming
    test's fixture rather than duplicating its assertions."""
    frame = _frame(_flat(_BASE))
    verdict = scroll_top.confirm_scroll_top(
        frame, identity_band=_IB, fingerprint=_fingerprint_of(frame), grid=_GRID)

    assert verdict.alignment_offset_px == 0
    assert "px" not in verdict.reason


def test_band_fingerprint_at_offset_skips_a_shift_that_would_fall_off_the_frame():
    """`confirm_scroll_top`'s sweep only ever asks for `_ALIGNMENT_SEARCH_PX` (12px), which never
    approaches the frame edge for `_IB` -- so this exercises `_band_fingerprint_at_offset`
    directly, at shifts large enough to prove the clamp exists rather than trusting it never
    fires in practice."""
    frame = _frame(_flat(_BASE))

    # _IB's y1 (0.094) is much nearer 0 than 1, so pushing the shifted y1 past 1.0 needs a much
    # larger dy_px than pushing y0 below 0 -- (1.0 - 0.094) * 2400 = 2174.4px versus 0.048 * 2400
    # = 115.2px. Both are picked comfortably past their respective thresholds.
    off_top = scroll_top._band_fingerprint_at_offset(
        frame, identity_band=_IB, grid=_GRID, dy_px=-1000)
    off_bottom = scroll_top._band_fingerprint_at_offset(
        frame, identity_band=_IB, grid=_GRID, dy_px=2200)
    in_bounds = scroll_top._band_fingerprint_at_offset(
        frame, identity_band=_IB, grid=_GRID, dy_px=4)

    assert off_top is None
    assert off_bottom is None
    assert in_bounds is not None


def test_band_fingerprint_at_offset_dy_zero_agrees_with_band_fingerprint():
    """`band_fingerprint` remains the dy=0 public entry point; `_band_fingerprint_at_offset` is
    only ever exercised at nonzero shifts by `confirm_scroll_top`, but it must not silently
    disagree with the public function if a caller ever passes dy_px=0 directly."""
    frame = _frame(_flat(_BASE))

    assert (scroll_top._band_fingerprint_at_offset(frame, identity_band=_IB, grid=_GRID, dy_px=0)
            == scroll_top.band_fingerprint(frame, identity_band=_IB, grid=_GRID))
