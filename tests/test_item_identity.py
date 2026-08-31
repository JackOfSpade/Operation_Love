"""The identity gate (operation_love/drivers/item_identity.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures the module's
constants were measured against are real people's dating profiles and are gitignored
(ops/calibration/, .gitignore:26); only geometry and counts from them appear anywhere in this
repo. So the fixtures paint a 1080x2400 frame from first principles and fill the rows and columns
`HINGE_SPEC.identity_band` cuts with a pattern the test chose, which makes every distance asserted
below known BY CONSTRUCTION rather than approximated: a flat band downsamples to exactly its own
value under any resampling filter, so "two headers 40 grey levels apart" really is a mean-abs
distance of 40.000.

The ONE piece of real calibration data in the fixtures is `_SCROLL_TOP_BAND_FINGERPRINT` — the
shipped constant, which is Hinge's own filter-chips chrome and is already in the repo. Painting it
back into a frame is what lets the real scroll-top half of this gate run with its real reference,
which matters here because "the screen is at its top" is one of the three outcomes.

Following the house pattern, every positive is paired with a negative plus a control that proves
WHICH mechanism did the refusing.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import hinge, item_identity, scroll_top
from operation_love.drivers.hinge import HINGE_SPEC

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_IB = HINGE_SPEC.identity_band             # (0.10, 0.048, 0.80, 0.094) -> x 108..864, y 115..226
_GRID = item_identity._IDENTITY_GRID       # (64, 16)
_MATCH_MAX = item_identity._IDENTITY_MATCH_MAX_DIST     # 3.0

# Two people's sticky headers, as flat bands. 40 grey levels apart: far outside the bound, and far
# inside 0..255 so nothing clips and the distance the test asked for is the distance the metric
# sees. (The two REAL profiles in the calibration corpus are 7.287 apart, which is why the bound
# is 3.0 and not the app's 9.0 change_threshold; see `_IDENTITY_MATCH_MAX_DIST`.)
_HER = 128
_HIM = 168


def _band_rows_cols(rect=_IB):
    """The exact pixel rect `_band` will cut for `rect`, mirroring its own rounding."""
    x0, y0, x1, y1 = rect
    return round(y0 * _H), round(y1 * _H), round(x0 * _W), round(x1 * _W)


def _chips(rows, cols):
    """Hinge's filter-chips strip, reconstructed from the SHIPPED fingerprint constant.

    `scroll_top`'s reference is 64 grey levels on a 16x4 grid over this rect, so painting those
    cells back as blocks and letting `hinge._band` downsample them reproduces the real reference
    through the real decode. It is app chrome, not profile content — the constant's own comment
    records it as byte-identical across two different people, which is why one array serves every
    profile and why nothing about anyone is in it.
    """
    cells = np.array(scroll_top._SCROLL_TOP_BAND_FINGERPRINT, dtype="uint8").reshape(4, 16)
    return np.asarray(Image.fromarray(cells, mode="L").resize((cols, rows), Image.NEAREST))


def _frame(header=None, *, background=40, rect=_IB) -> bytes:
    """A decodable 1080x2400 greyscale PNG. `header=None` paints the filter-chips row into `rect`
    (a frame at a card's scroll top); an int paints that flat value (a scrolled frame showing that
    person's sticky header).

    The background is deliberately NOT either of those, so a comparison that accidentally read the
    wrong rect would land wildly away rather than plausibly near.
    """
    arr = np.full((_H, _W), background, dtype="uint8")
    r0, r1, c0, c1 = _band_rows_cols(rect)
    arr[r0:r1, c0:c1] = _chips(r1 - r0, c1 - c0) if header is None else header
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _capture(header=_HER, *, rect=_IB, band=_IB):
    """A whole profile read, as the driver produces one: frame 0 at the confirmed scroll top,
    every later frame showing that person's sticky header."""
    frames = [_frame(None, rect=rect)] + [_frame(header, rect=rect) for _ in range(3)]
    return frames, item_identity.capture_profile_identity(frames, identity_band=band)


def _compare(frame, identity, *, band=_IB):
    return item_identity.compare_profile_identity(frame, identity, identity_band=band)


# =====================================================================================
# The primitive is the driver's own, not a second one
# =====================================================================================

def test_the_grid_is_the_drivers_own_and_the_bound_is_deliberately_not_change_threshold():
    """The grid is DECLARED here rather than imported, because hinge imports this package and not
    the other way round, so this is the test that stops the copies drifting — and the point of the
    module is that it reuses the identity primitive the driver already ships, so a divergence
    would mean it had quietly become a second scheme.

    The BOUND is the one thing not inherited, and that is a measurement: the two calibration
    profiles' settled headers are 7.287 grey levels apart at this grid, so `change_threshold`
    (9.0, what `hinge._identity_of` decides same-person on) calls two different people the same
    person. This assertion is what stops anyone "restoring consistency" by putting it back."""
    assert _GRID == hinge._IDENTITY_DS
    assert _MATCH_MAX == scroll_top._CONFIRM_MAX_DIST      # one tolerance on this strip, not two
    assert _MATCH_MAX < 7.287 < HINGE_SPEC.change_threshold


def test_the_distance_is_the_same_number_hinge_band_dist_would_have_produced():
    """One decode path, one metric. `item_crops.signature_of`'s docstring records what two decode
    paths cost on this codebase (1.46 grey levels apart on average, 9.8 on one signature), against
    a match bound of 3.0 — so a second decode would be several times the whole tolerance."""
    her, him = _frame(_HER), _frame(_HIM)
    _frames, identity = _capture(_HER)

    verdict = _compare(him, identity)
    direct = hinge._band_dist(hinge._band(him, _IB), hinge._band(her, _IB))

    assert verdict.distance == pytest.approx(direct)
    assert verdict.distance == pytest.approx(abs(_HIM - _HER))


# =====================================================================================
# Capturing the fingerprint: both states of the strip, or nothing
# =====================================================================================

def test_the_fingerprint_is_the_earliest_corroborated_scrolled_frame_after_a_confirmed_top():
    frames, identity = _capture(_HER)

    assert identity.known
    assert identity.frame_index == 1                  # frame 0 is the chips row
    assert identity.band == _IB and identity.grid == _GRID
    assert identity.fingerprint == scroll_top.band_fingerprint(
        frames[1], identity_band=_IB, grid=_GRID)
    assert identity.scroll_top_distance >= scroll_top._REFUTE_MIN_DIST
    assert (identity.agreeing_frames, identity.outlier_frames) == (3, 0)
    assert "sticky per-profile header" in identity.reason


def test_an_uncorroborated_transitional_frame_is_not_the_fingerprint():
    """THE BUG THE CALIBRATION CORPUS FOUND, synthesised. Taking the first refuting frame picked,
    on one of the two real profiles, a frame caught MID-SLIDE-IN: the only one of 112 that
    disagreed with the rest, 14.482 from the settled header the other 111 share to 0.000. An index
    fingerprinted from it refuses its own profile at every later frame — a guaranteed false stop
    on a good read.

    Corroboration is what fixes it, and it is the same rule `item_index` applies to a block's
    extent: two frames that both observed one fixed quantity are two measurements of it."""
    transitional = (_HER + _HIM) // 2        # neither the chips row nor the settled header
    frames = [_frame(None), _frame(transitional)] + [_frame(_HER) for _ in range(4)]
    identity = item_identity.capture_profile_identity(frames, identity_band=_IB)

    assert identity.known
    assert identity.frame_index == 2                  # NOT 1, which nothing agrees with
    assert (identity.agreeing_frames, identity.outlier_frames) == (4, 1)
    assert identity.outlier_max_distance == pytest.approx(abs(transitional - _HER))
    assert "mid-slide-in" in identity.reason

    # And the consequence, which is the whole point: every later frame of her own profile matches.
    assert all(_compare(f, identity).matched for f in frames[2:])


def test_a_single_uncorroborated_header_reading_is_no_fingerprint_at_all():
    """One reading is exactly the state the transitional frame above was in. With nothing to
    corroborate it there is no way to tell a settled header from a frame caught mid-animation, so
    the capture reports no identity rather than a reading it cannot stand behind."""
    identity = item_identity.capture_profile_identity(
        [_frame(None), _frame(_HER)], identity_band=_IB)

    assert not identity.known
    assert "no two frames" in identity.reason


def test_a_capture_whose_headers_split_evenly_is_refused_rather_than_halved():
    """A capture that may span two people cannot be fingerprinted as one, and picking the larger
    half would fingerprint whichever happened to have more frames. Upstream,
    `hinge._capture_current`'s split check and the index's own correspondence chain refuse such a
    capture first; this is the belt to those braces."""
    frames = [_frame(None), _frame(_HER), _frame(_HER), _frame(_HIM), _frame(_HIM)]
    identity = item_identity.capture_profile_identity(frames, identity_band=_IB)

    assert not identity.known
    assert "may span two profiles" in identity.reason

    # The control: one more frame of hers makes it a majority, and it fingerprints her.
    tipped = item_identity.capture_profile_identity(frames + [_frame(_HER)], identity_band=_IB)
    assert tipped.known and tipped.agreeing_frames == 3 and tipped.outlier_frames == 2


def test_two_people_fingerprint_differently_and_one_person_does_not():
    """The measured premise, in both directions. [device: three frames of one profile at three
    different scroll offsets measure 0.00 between them.] The header is a property of the person,
    so it must be constant down a profile and different across two."""
    _f, hers = _capture(_HER)
    _g, his = _capture(_HIM)

    assert hers.fingerprint != his.fingerprint
    assert _compare(_frame(_HER), hers).matched                    # a later frame of her profile
    assert _compare(_frame(_HIM), hers).mismatched


def test_a_rect_that_never_shows_the_chips_row_yields_no_fingerprint():
    """The half that stops a fingerprint being taken off static chrome. A band that never confirms
    a top is a band nothing proves is Hinge's header strip — and a fingerprint of, say, the status
    bar would be IDENTICAL on every profile, so the gate would pass for everybody. Refusing to
    take one is the only structure that rules that out without a second calibration."""
    elsewhere = (0.10, 0.020, 0.80, 0.066)            # the paint is at _IB, the band is not
    frames = [_frame(None), _frame(_HER), _frame(_HER)]
    identity = item_identity.capture_profile_identity(frames, identity_band=elsewhere)

    assert not identity.known
    assert "filter-chips row" in identity.reason and "for everybody" in identity.reason

    # The control: the same frames, read through the rect they were painted at.
    assert item_identity.capture_profile_identity(frames, identity_band=_IB).known


def test_a_capture_that_never_scrolled_far_enough_yields_no_fingerprint():
    """The other half. At a scroll top the strip is chrome that is byte-identical across two
    different people, so a capture that only ever saw the top has seen nothing about anyone."""
    identity = item_identity.capture_profile_identity(
        [_frame(None), _frame(None)], identity_band=_IB)

    assert not identity.known
    assert "never appeared" in identity.reason


def test_a_pinned_band_is_diagnosed_as_pinned_rather_than_blamed_on_the_capture():
    """THE TWO CAUSES OF ONE SYMPTOM, and the reason the offsets exist. "Every frame reads as the
    filter-chips row" happens for two completely different reasons with opposite fixes: the
    capture never scrolled far enough (re-run it), or the app PINS the chips row to the screen so
    that no capture on this build will ever reveal a header (Hinge 10.1.0's expanded header,
    measured 2026-08-28 -- the gate above returned `confirmed_top` at 0.000 on six frames taken
    6180..8485px down one profile). Blaming the capture in the second case sends an operator to
    re-run it forever, which is this repo's standing "guidance must derive from its precondition"
    rule being violated in production.

    The OUTCOME is identical and deliberately untouched: no fingerprint, `.known` False,
    navigation refuses. Only the diagnosis changes."""
    frames = [_frame(None), _frame(None), _frame(None)]     # the chips row on every frame

    pinned = item_identity.capture_profile_identity(
        frames, identity_band=_IB, page_offsets=[0, 600, 1200])

    assert not pinned.known and pinned.fingerprint is None   # fail-closed, exactly as before
    assert "is PINNED to the screen on this app version" in pinned.reason
    assert "structurally unavailable" in pinned.reason
    assert "1200px of page scroll" in pinned.reason          # the evidence, not an assertion
    assert "apps.hinge.identity_band" in pinned.reason       # what actually needs recalibrating
    assert "never scrolled far enough" not in pinned.reason

    # THE CONTROL, and it is the whole point of pairing them: the SAME frames, on a capture that
    # provably did not scroll, keep the original message. Nothing here proves anything about the
    # app -- at one offset a pinned strip and page content predict identical pixels.
    still = item_identity.capture_profile_identity(
        frames, identity_band=_IB, page_offsets=[900, 900, 900])
    assert not still.known
    assert "never scrolled far enough" in still.reason
    assert "PINNED" not in still.reason

    # ...as does a caller that supplies no offsets at all, which is every call site that has not
    # been taught to pass them: an absent proof degrades to the pre-existing diagnosis.
    silent = item_identity.capture_profile_identity(frames, identity_band=_IB)
    assert not silent.known and "never scrolled far enough" in silent.reason


def test_offsets_that_do_not_line_up_with_the_frames_raise_rather_than_being_zipped():
    """Not a frame problem -- a caller pairing one frame's band with another frame's scroll
    position -- so it is an error rather than the skipped-and-noted treatment an unreadable frame
    gets. Zipping to the shorter of the two would attribute a real scroll to frames that never
    made it."""
    with pytest.raises(item_identity.IdentityError, match="different captures"):
        item_identity.capture_profile_identity(
            [_frame(None), _frame(_HER)], identity_band=_IB, page_offsets=[0])


def test_no_band_and_no_frames_are_both_unknown_rather_than_errors():
    """`identity_band` is optional on `AndroidAppSpec` by design, so an app without one is a thing
    this gate cannot judge rather than a bug. Both answers are UNKNOWN, which navigation
    refuses."""
    none_declared = item_identity.capture_profile_identity([_frame(_HER)], identity_band=None)
    assert not none_declared.known and "no identity_band" in none_declared.reason

    empty = item_identity.capture_profile_identity([], identity_band=_IB)
    assert not empty.known and "no frames" in empty.reason


def test_an_undecodable_frame_is_skipped_rather_than_fatal():
    """This runs inside an index build whose own answer is a RESULT. One unreadable frame must not
    turn into an exception out of a function whose job is to describe the capture — but it is
    recorded, so a capture that produced no fingerprint says how many frames it could not read."""
    frames = [_frame(None), b"not an image at all", _frame(_HER), _frame(_HER)]
    identity = item_identity.capture_profile_identity(frames, identity_band=_IB)

    assert identity.known and identity.frame_index == 2 and identity.agreeing_frames == 2

    only_junk = item_identity.capture_profile_identity(
        [_frame(None), b"not an image at all"], identity_band=_IB)
    assert not only_junk.known and "could not be read" in only_junk.reason


# =====================================================================================
# Comparing: match, mismatch, cannot tell
# =====================================================================================

def test_the_same_profile_matches_at_the_distance_it_really_is():
    _frames, identity = _capture(_HER)
    verdict = _compare(_frame(_HER), identity)

    assert verdict.state == item_identity.IDENTITY_MATCH
    assert (verdict.matched, verdict.mismatched, verdict.unknown) == (True, False, False)
    assert verdict.distance == 0.0
    assert "matches the one this index was built from" in verdict.reason


def test_the_bound_is_inclusive_and_one_grey_level_past_it_is_a_mismatch():
    """At the bound matches, past it does not. Driven to the boundary rather than near it: a flat
    band downsamples to exactly its own value, so "+3" really is a mean-abs distance of 3.000."""
    _frames, identity = _capture(_HER)

    at_bound = _compare(_frame(_HER + int(_MATCH_MAX)), identity)
    assert at_bound.matched and at_bound.distance == _MATCH_MAX
    past = _compare(_frame(_HER + int(_MATCH_MAX) + 1), identity)
    assert past.mismatched and past.distance == _MATCH_MAX + 1


def test_a_different_profile_is_a_mismatch_and_says_why_geometry_could_not_tell():
    _frames, identity = _capture(_HER)
    verdict = _compare(_frame(_HIM), identity)

    assert verdict.state == item_identity.IDENTITY_MISMATCH
    assert verdict.distance == pytest.approx(abs(_HIM - _HER))
    assert "stereotyped" in verdict.reason


def test_a_screen_at_its_scroll_top_is_cannot_tell_even_when_the_distance_is_small():
    """THE ORDER OF THE CHECKS, pinned. The stored fingerprint here IS the chips row — which
    `capture_profile_identity` would never produce, so it is constructed directly — and the frame
    is the chips row too, so a naive distance-first implementation would report a confident 0.000
    MATCH. It is UNREADABLE, because at the top that strip is chrome that two different people
    share to the byte, and a small distance there is a property of the chrome and not evidence
    about anyone."""
    top = _frame(None)
    chips = item_identity.ProfileIdentity(
        fingerprint=scroll_top.band_fingerprint(top, identity_band=_IB, grid=_GRID),
        band=_IB, grid=_GRID, frame_index=0, scroll_top_distance=None, reason="constructed")

    verdict = _compare(top, chips)

    assert verdict.state == item_identity.IDENTITY_UNKNOWN
    assert verdict.distance is None
    assert "scroll top" in verdict.reason and "cannot tell" in verdict.reason

    # The control: the identical stored fingerprint against a SCROLLED frame does compare.
    assert _compare(_frame(_HER), chips).mismatched


def test_an_unknown_identity_and_an_undeclared_band_are_both_cannot_tell():
    _frames, identity = _capture(_HER)
    unknown = item_identity.capture_profile_identity([], identity_band=_IB)

    no_fingerprint = _compare(_frame(_HER), unknown)
    assert no_fingerprint.unknown and "no identity fingerprint" in no_fingerprint.reason
    assert _compare(_frame(_HER), None).unknown

    no_band = item_identity.compare_profile_identity(
        _frame(_HER), identity, identity_band=None)
    assert no_band.unknown and "no identity_band is declared" in no_band.reason


def test_a_band_that_moved_since_capture_is_an_error_not_a_distance():
    """Two different crops of a screen have no distance between them that measures anything, so an
    operator editing `apps.hinge.identity_band` between the read and the tap gets "could not look"
    rather than a comparison that happens to fall one side of a bound."""
    _frames, identity = _capture(_HER)
    with pytest.raises(item_identity.IdentityError, match="different crops"):
        _compare(_frame(_HER), identity, band=(0.10, 0.020, 0.80, 0.066))


def test_bytes_that_are_not_an_image_are_could_not_look_and_one_exception_type():
    """"We could not look" must never be reachable by a route that also produces a verdict. The
    read goes through `scroll_top`'s shared decode, and its `ScrollTopError` is re-wrapped so a
    caller has one exception type to route rather than needing to know which sibling module the
    bytes went through."""
    _frames, identity = _capture(_HER)
    with pytest.raises(item_identity.IdentityError, match="could not be read"):
        _compare(b"not an image at all", identity)


# =====================================================================================
# The API cannot be misused quietly
# =====================================================================================

def test_a_verdict_has_no_truth_value():
    """`if compare_profile_identity(...):` would read "cannot tell" and "that is a different
    person" as "same person", in the most natural-looking way possible. numpy's precedent, and
    `ScrollTopVerdict.__bool__`'s: an ambiguous truth value is a TypeError, not a guess."""
    _frames, identity = _capture(_HER)
    for frame in (_frame(_HER), _frame(_HIM), _frame(None)):
        with pytest.raises(TypeError, match="no truth value"):
            bool(_compare(frame, identity))


def test_the_fingerprint_is_plain_ints_and_carries_no_image():
    """It is stored on an `ItemIndex`, logged, and compared in places numpy may not be. 1024 block
    averages of a strip of flat app chrome with a first name on it — comparable anywhere, and
    nothing recoverable from it."""
    _frames, identity = _capture(_HER)

    assert len(identity.fingerprint) == _GRID[0] * _GRID[1]
    assert all(isinstance(v, int) for v in identity.fingerprint)
    assert item_identity.ProfileIdentity(
        fingerprint=identity.fingerprint, band=identity.band, grid=identity.grid,
        frame_index=1, scroll_top_distance=None, reason="") == item_identity.ProfileIdentity(
        fingerprint=identity.fingerprint, band=identity.band, grid=identity.grid,
        frame_index=1, scroll_top_distance=None, reason="")
