"""The driver-owned item index (operation_love/drivers/item_index.py).

Every frame here is SYNTHESISED, never a real screencap. The calibration captures that
segment.py's and frameshift.py's constants were measured against are real people's dating
profiles and are gitignored (ops/calibration/, .gitignore:26); only geometry and counts from them
appear anywhere in this repo. So the fixtures build a tall scrollable WORLD from first principles
— gradient page background, rounded-rect cards inset 53px per side, canonical 53px gutters, the
genuine shipped like glyph stamped bottom-right on the likeable ones and a heartless 215px vitals
block among them — and cut 1080x2400 windows out of it at offsets the test chooses.

That makes the ground truth exact and known BY CONSTRUCTION: the page layout is a table at the
top of this file, so the assertions can name the items, their page rows, their heart ordinals and
their model indices rather than approximating any of it.

Two mechanisms are tested at two levels on purpose, following the house pattern that
tests/test_frameshift.py uses for `_resolve`:

  * end to end, through `build_item_index`, where the shift estimator and the segmenter are real
    and the only input is pixels;
  * and directly on `_assemble`, which is pure arithmetic over `BlockObservation` records. That
    is where the numbering, the disagreement rules and the fabricated-item guard can be driven to
    the exact contradiction they exist to catch, with no dependence on what a correlation happens
    to do.

Every positive is paired with a negative plus a control that proves WHICH mechanism did the
excluding.
"""
import dataclasses
import hashlib
import math
import subprocess
import sys

import cv2
import numpy as np
import pytest

from operation_love.drivers import frameshift, hinge, item_index, segment

_W, _H = 1080, 2400                        # the calibrated Pixel 7a screencap size
_SEED = 13
_CONTENT_BAND = hinge.HINGE_SPEC.content_band          # (0.125, 0.875) -> rows 300..2100
_BAND0, _BAND1 = 300, 2100
_BAND_H = _BAND1 - _BAND0
_CARD_X0 = segment._CARD_MARGIN_PX                     # 53
_CARD_X1 = _W - segment._CARD_MARGIN_PX                # 1027
_GUTTER = max(segment._GUTTER_PX)                      # 53, the canonical measured gutter
_CORNER_RADIUS_PX = 22                                 # doc 5.10: "~20-23px"
_HEART_CX = 937                                        # bottom-right, as test_segment.py stamps it
_HEART_ABOVE_BOTTOM = 90                               # segment.py: the heart sits ~89px above a
                                                       # complete card's bottom edge
_PAGE_TOP, _PAGE_BOTTOM = 254, 243                     # doc 5.10's background gradient
_WORLD_H = 5400

# The step doc 5.10.1 measured at 100% frame-to-frame heart tracking with zero phantoms
# (read_scroll_frac 0.16 -> a rock-steady 363px/step).
_STEP = 363

# THE PAGE, in world rows. Five blocks: four likeable cards and, between the second and third,
# the heartless 215px vitals block doc 5.10 tracked across 7 consecutive frames. Heights are from
# doc 5.10's card table (nothing here exceeds the 1114px tallest card observed end to end).
_PAGE_TOP_GAP = 400                        # page background above card 1, where Hinge's header is
_LAYOUT = (("card", 900), ("card", 760), ("context", 215), ("card", 1000), ("card", 820))


def _layout_rows():
    """`[(kind, y0, y1), ...]` in world rows, gutter-separated exactly as the layout draws."""
    rows, y = [], _PAGE_TOP_GAP
    for kind, height in _LAYOUT:
        rows.append((kind, y, y + height))
        y += height + _GUTTER
    return rows


_ROWS = _layout_rows()
_CARD1, _CARD2, _VITALS, _CARD3, _CARD4 = _ROWS
assert _CARD4[2] + 400 < _WORLD_H, "the world must have page background below the last card"


def _page_column(height=_WORLD_H):
    return np.linspace(_PAGE_TOP, _PAGE_BOTTOM, height).round().astype(np.uint8)


_TEMPLATE = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
assert _TEMPLATE is not None, "the shipped like glyph must load"


def _build_world(*, short_card2=0):
    """The tall page. `short_card2` shortens the second card by that many rows, moving its bottom
    edge (and its heart) up while leaving every other block exactly where it was — the ONE way to
    make two frames disagree about one card's height without also moving everything below it."""
    rng = np.random.default_rng(_SEED)
    col = _page_column()
    world = np.repeat(col[:, None], _W, axis=1)
    for kind, y0, y1 in _ROWS:
        if (kind, y0) == (_CARD2[0], _CARD2[1]):
            y1 -= short_card2
        world[y0:y1, _CARD_X0:_CARD_X1] = rng.integers(
            60, 200, size=(y1 - y0, _CARD_X1 - _CARD_X0), dtype=np.uint8)
        for i in range(_CORNER_RADIUS_PX):             # carve the four corner arcs back to page
            dy = _CORNER_RADIUS_PX - i
            inset = int(math.ceil(_CORNER_RADIUS_PX
                                  - math.sqrt(max(0.0, _CORNER_RADIUS_PX ** 2 - dy ** 2))))
            if inset <= 0:
                continue
            for y in (y0 + i, y1 - 1 - i):
                world[y, _CARD_X0:_CARD_X0 + inset] = col[y]
                world[y, _CARD_X1 - inset:_CARD_X1] = col[y]
        if kind == "card":
            th, tw = _TEMPLATE.shape
            cy = y1 - _HEART_ABOVE_BOTTOM
            world[cy - th // 2: cy - th // 2 + th,
                  _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = _TEMPLATE
    return world


_WORLDS = {0: _build_world()}
# Static chrome outside the content band, textured but IDENTICAL on every frame — the real
# device's status bar, sticky header, floating buttons and bottom nav, none of which translate.
_CHROME_TOP = np.random.default_rng(3).integers(0, 60, size=(_BAND0, _W), dtype=np.uint8)
_CHROME_BOTTOM = np.random.default_rng(4).integers(0, 60, size=(_H - _BAND1, _W), dtype=np.uint8)
_FRAMES: dict[tuple[int, int], bytes] = {}


def _frame(scroll: int, *, short_card2: int = 0) -> bytes:
    """The 1080x2400 window of the world at `scroll`, as PNG. Content at world row `w` lands on
    frame row `w - scroll`, so `_frame(s)` then `_frame(s + d)` is a forward scroll of `d`."""
    key = (scroll, short_card2)
    if key not in _FRAMES:
        if short_card2 not in _WORLDS:
            _WORLDS[short_card2] = _build_world(short_card2=short_card2)
        gray = _WORLDS[short_card2][scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        _FRAMES[key] = buf.tobytes()
    return _FRAMES[key]


def _index(scrolls, *, at_scroll_top, **kw):
    return item_index.build_item_index(
        [_frame(s) for s in scrolls], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=at_scroll_top,
        identity_band=kw.pop("identity_band", None), **kw)


def _shift_with_votes(base, votes, *, status, delta=None):
    """A hand-controlled frameshift result over otherwise-real synthetic frame geometry."""
    strips = [dataclasses.replace(strip, state=frameshift.STRIP_WEAK, delta_px=None)
              for strip in base.strips]
    for i, vote in enumerate(votes):
        strips[i] = dataclasses.replace(
            strips[i], state=frameshift.STRIP_MATCHED, delta_px=vote, score=0.99)
    return dataclasses.replace(
        base, strips=tuple(strips), status=status, delta_px=delta, consensus_px=delta,
        agreeing=(sum(abs(vote - delta) <= 3 for vote in votes) if delta is not None else 0),
        dissenting=(sum(abs(vote - delta) > 3 for vote in votes) if delta is not None else 0),
        eligible=len(votes), confidence=(1.0 if delta is None else
                                         sum(abs(vote - delta) <= 3 for vote in votes) / len(votes)),
        reason="synthetic strip-bank evidence")


# Scroll offsets: a full read from the top of the profile to past its last card, at the cadence
# doc 5.10.1 validated. Frame 0 sits at world row 0 so its band opens on the page background
# above card 1; the last frame clears card 4's bottom by 334 rows.
_FULL_SCROLL = tuple(_STEP * i for i in range(8))
assert _FULL_SCROLL[-1] + _H <= _WORLD_H
_CACHE: dict[str, object] = {}


def _full():
    """The clean whole-profile index, built once — eight frames means seven real shift estimates
    and eight real segmentations, which is worth about four seconds and is not worth repeating."""
    if "full" not in _CACHE:
        _CACHE["full"] = _index(_FULL_SCROLL, at_scroll_top=True)
    return _CACHE["full"]


def test_incremental_prefix_reuses_measurements_without_changing_the_index(monkeypatch):
    """A growing capture must not re-segment its already hash-bound frame prefix."""
    baseline = _full()
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_segment = item_index.segment_frame
    real_shift = item_index.estimate_shift
    calls = {"segment": 0, "shift": 0}

    def counted_segment(*args, **kwargs):
        calls["segment"] += 1
        return real_segment(*args, **kwargs)

    def counted_shift(*args, **kwargs):
        calls["shift"] += 1
        return real_shift(*args, **kwargs)

    monkeypatch.setattr(item_index, "segment_frame", counted_segment)
    monkeypatch.setattr(item_index, "estimate_shift", counted_shift)
    prior = None
    for size in range(1, len(frames) + 1):
        prior = item_index.build_item_index(
            frames[:size], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
            identity_band=None, _prefix_index=prior)

    assert prior == baseline
    # An uncached pass at every prefix would call these 36 and 28 times respectively.
    assert calls == {"segment": len(frames), "shift": len(frames) - 1}


def test_one_isolated_bad_intermediate_frame_is_rebuilt_over_a_measured_bridge(monkeypatch):
    """A single failed pair is recoverable only by discarding one real intermediate frame.

    This models the reported ordinary +229px Hinge step: two strips were unanimous but below the
    three-witness floor, so the original pair stays a refusal.  The replacement must instead be
    a fresh, fully usable index whose direct neighbouring bridge is measured; it may not reuse
    the failed pair's consensus or assume a two-step offset.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    failed_pair = (3, 4)
    real_shift = item_index.estimate_shift

    def one_bad_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) == (frames[failed_pair[0]], frames[failed_pair[1]]):
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic two-witness refusal")
        return result

    calls = []

    def counted_one_bad_pair(frame_a, frame_b, **kwargs):
        calls.append((frame_a, frame_b))
        return one_bad_pair(frame_a, frame_b, **kwargs)

    monkeypatch.setattr(item_index, "estimate_shift", counted_one_bad_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.recovered_from_pair == failed_pair
    assert index.recovery_bridge is not None
    assert index.recovery_failed_shift is not None
    assert index.recovery_failed_shift.reason == "synthetic two-witness refusal"
    omitted = tuple(i for i in range(len(frames)) if i not in index.source_frame_indices)
    assert omitted == (3,)                 # equal bridges choose the lower original index
    left, right = index.recovery_bridge
    assert right == left + 2 and omitted[0] == left + 1
    bridge = index.shifts[left]
    assert bridge.status == frameshift.SHIFT_MEASURED
    assert bridge.delta_px == _STEP * 2
    assert "without assuming an offset" in index.recovery_reason
    # Both candidate omissions were actually measured before the deterministic choice.
    assert (frames[2], frames[4]) in calls and (frames[3], frames[5]) in calls


def test_one_contradictory_segmentation_frame_is_rebuilt_over_a_measured_bridge(monkeypatch):
    """One frame that missed a gutter may be omitted only after a full direct-bridge rebuild."""
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    bad_frame = 4
    real_segment = item_index.segment_frame

    def one_contradictory_frame(frame, **kwargs):
        result = real_segment(frame, **kwargs)
        if frame == frames[bad_frame]:
            return dataclasses.replace(
                result, failures=("synthetic two hearts in one block",))
        return result

    monkeypatch.setattr(item_index, "segment_frame", one_contradictory_frame)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 2, 3, 5, 6, 7)
    assert index.recovered_from_pair is None
    assert index.recovered_from_segmentation_frames == (bad_frame,)
    assert index.recovery_bridge == (3, 5)
    assert index.shifts[3].status == frameshift.SHIFT_MEASURED
    assert "segmentation contradicted itself" in index.recovery_reason


def test_segmentation_recovery_rebuilds_two_adjacent_contradictory_frames(monkeypatch):
    """A short white-on-white boundary run needs one direct bridge across both bad captures."""
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_segment = item_index.segment_frame

    def two_contradictory_frames(frame, **kwargs):
        result = real_segment(frame, **kwargs)
        if frame in (frames[3], frames[4]):
            return dataclasses.replace(result, failures=("synthetic contradiction",))
        return result

    monkeypatch.setattr(item_index, "segment_frame", two_contradictory_frames)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 2, 5, 6, 7)
    assert index.recovered_from_segmentation_frames == (3, 4)
    assert index.recovery_bridge == (2, 5)
    assert index.shifts[2].status == frameshift.SHIFT_MEASURED
    assert "frames 3, 4" in index.recovery_reason


def test_segmentation_recovery_rebuilds_three_adjacent_contradictory_frames(monkeypatch):
    """The observed white-on-white run is rebuilt only over a four-read measured bridge."""
    # Rebecca's captured recovery steps were 220..231px. At that cadence a four-read bridge
    # retains enough overlapping content for frameshift's ordinary 3-witness quorum.
    frames = [_frame(230 * i) for i in range(8)]
    real_segment = item_index.segment_frame

    def three_contradictory_frames(frame, **kwargs):
        result = real_segment(frame, **kwargs)
        if frame in (frames[2], frames[3], frames[4]):
            return dataclasses.replace(result, failures=("synthetic contradiction",))
        return result

    monkeypatch.setattr(item_index, "segment_frame", three_contradictory_frames)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 5, 6, 7)
    assert index.recovered_from_segmentation_frames == (2, 3, 4)
    assert index.recovery_bridge == (1, 5)
    assert index.shifts[1].status == frameshift.SHIFT_MEASURED
    assert "frames 2, 3, 4" in index.recovery_reason


def test_segmentation_recovery_rebuilds_four_adjacent_contradictory_frames(monkeypatch):
    """A four-frame run is safe only when its five-read bridge is freshly measured."""
    # Rebecca's live failure was four 227..230px fallback steps in one missed-gutter run.
    # Keep the bridge small enough for the ordinary strip quorum to prove it, rather than
    # widening a matching rule to make this test pass.
    frames = [_frame(230 * i) for i in range(10)]
    real_segment = item_index.segment_frame

    def four_contradictory_frames(frame, **kwargs):
        result = real_segment(frame, **kwargs)
        if frame in (frames[2], frames[3], frames[4], frames[5]):
            return dataclasses.replace(result, failures=("synthetic contradiction",))
        return result

    monkeypatch.setattr(item_index, "segment_frame", four_contradictory_frames)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 6, 7, 8, 9)
    assert index.recovered_from_segmentation_frames == (2, 3, 4, 5)
    assert index.recovery_bridge == (1, 6)
    assert index.shifts[1].status == frameshift.SHIFT_MEASURED
    assert "frames 2, 3, 4, 5" in index.recovery_reason


def test_frame_omission_recovery_refuses_when_no_direct_bridge_is_measured(monkeypatch):
    """The recovery cannot turn a failed pair into a permission to drop evidence blindly."""
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift

    def failed_pair_and_bridges(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        positions = {(frames[3], frames[4]), (frames[2], frames[4]), (frames[3], frames[5])}
        if (frame_a, frame_b) in positions:
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic bridge refusal")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", failed_pair_and_bridges)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable
    assert index.source_frame_indices == tuple(range(len(frames)))
    assert index.recovered_from_pair is None and index.recovery_bridge is None


def test_one_transient_frame_can_recover_both_adjacent_refused_pairs(monkeypatch):
    """A damaged intermediate frame naturally poisons both of its pairwise comparisons.

    Recovery still requires independent evidence: the shared frame is omitted, its two real
    neighbours are correlated directly, and the entire reduced index must validate.  This is
    the shape of the reported animated-card capture; the old exactly-one-refusal gate never
    attempted the valid bridge.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift
    refused = {(frames[3], frames[4]), (frames[4], frames[5])}

    def transient_middle(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) in refused:
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic transient-frame refusal")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", transient_middle)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 2, 3, 5, 6, 7)
    assert index.recovery_bridge == (3, 5)
    assert index.recovered_from_pairs == ((3, 4), (4, 5))
    assert len(index.recovery_failed_shifts) == 2
    assert index.shifts[3].status == frameshift.SHIFT_MEASURED
    assert index.shifts[3].delta_px == _STEP * 2
    assert "pairs 3/4, 4/5" in index.recovery_reason


def test_transient_frame_recovery_allows_measured_bridge_alignment_slack(monkeypatch):
    """A strong direct bridge may carry a little more drift than an ordinary one-step chain.

    The reported 2026-08-15 capture had two adjacent two-witness refusals around one frame.  Its
    direct neighbour bridge was independently measured by 4/4 eligible strips, but put the
    shared bounded card 9px past the extent seen on the other side.  That is still far inside a
    real Hinge gutter and must be absorbed as chain slack; rejecting the complete rebuild loses
    every item even though the recovery used no refused consensus or assumed offset.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift
    refused = {(frames[3], frames[4]), (frames[4], frames[5])}
    bridge = (frames[3], frames[5])

    def transient_middle_with_drifted_bridge(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) in refused:
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic transient-frame refusal")
        if (frame_a, frame_b) == bridge:
            return dataclasses.replace(
                result, delta_px=result.delta_px + 9, consensus_px=result.consensus_px + 9,
                reason="synthetic measured bridge with 9px alignment slack")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", transient_middle_with_drifted_bridge)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 2, 3, 5, 6, 7)
    assert index.recovery_bridge == (3, 5)
    assert index.shifts[3].status == frameshift.SHIFT_MEASURED
    assert index.shifts[3].delta_px == _STEP * 2 + 9
    assert item_index._EXTENT_TOLERANCE_PX == 8
    assert item_index._RECOVERY_EXTENT_TOLERANCE_PX == 9
    assert item_index._RECOVERY_EXTENT_TOLERANCE_PX * 2 < item_index._MIN_ITEM_GAP_PX
    # The widened slack is a property of THIS index, and a second pass cross-checking it has to
    # add its own to this one rather than assume the default (see item_nav's crosscheck bound).
    assert index.extent_tolerance_px == item_index._RECOVERY_EXTENT_TOLERANCE_PX


def test_two_adjacent_refusals_still_fail_when_the_direct_bridge_refuses(monkeypatch):
    """Two adjacent failures identify a candidate frame; they do not authorize dropping it."""
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift
    refused = {(frames[3], frames[4]), (frames[4], frames[5]), (frames[3], frames[5])}

    def transient_and_bridge(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) in refused:
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic direct-bridge refusal")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", transient_and_bridge)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable
    assert index.source_frame_indices == tuple(range(len(frames)))
    assert index.recovered_from_pairs == () and index.recovery_bridge is None


def test_layout_assisted_animation_run_repairs_two_refusals_and_wrong_measured_majority(monkeypatch):
    """Three neighbouring pair failures are repaired only as one layout-proven animation run.

    This is the earlier saved-capture shape in synthetic form: a 2/2 refusal, a 2/3 refusal, and a
    wrong three-strip majority whose two dissenting strips are the ones card geometry proves.
    The original frameshift quorum remains untouched; the index carries all three raw estimates.
    """
    frames = [_frame(_STEP * i) for i in range(5)]
    real_shift = item_index.estimate_shift

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(4) if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 0:  # exactly two agreeing strips: raw no-consensus
            return _shift_with_votes(result, [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 1:  # exactly two agree, one different matched strip: still raw no-consensus
            return _shift_with_votes(result, [_STEP, _STEP, _STEP - 16],
                                     status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 2:  # raw three-strip majority is wrong by ten pixels; two stable strips win
            return _shift_with_votes(result, [_STEP - 10, _STEP - 10, _STEP - 10, _STEP, _STEP],
                                     status=frameshift.SHIFT_MEASURED, delta=_STEP - 10)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert [shift.delta_px for shift in index.shifts[:3]] == [_STEP, _STEP, _STEP]
    assert len(index.layout_repaired_shifts) == 3
    assert [raw.status for _pair, raw in index.layout_repaired_shifts] == [
        frameshift.SHIFT_NO_CONSENSUS, frameshift.SHIFT_NO_CONSENSUS, frameshift.SHIFT_MEASURED]
    assert index.layout_repaired_shifts[2][1].delta_px == _STEP - 10
    assert len(index.notes) >= 3
    assert all("layout-assisted" in note for note in index.notes[:3])


def test_layout_assisted_two_pair_run_repairs_two_strips_then_one_strip(monkeypatch):
    """The live-video failure is repaired without weakening frameshift's raw quorum.

    The first pair has the existing two-NCC plus layout proof.  Animation leaves only one stable
    NCC strip in the adjacent pair, where exact top, bottom, and heart geometry supplies the
    stronger cross-check.  This is the saved 2026-08-15 frames 8/9/10 shape.
    """
    frames = [_frame(_STEP * i) for i in range(5)]
    real_shift = item_index.estimate_shift

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(4) if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 1:
            return _shift_with_votes(result, [_STEP, _STEP],
                                     status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 2:
            return _shift_with_votes(result, [_STEP], status=frameshift.SHIFT_NO_CONSENSUS)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert [shift.delta_px for shift in index.shifts[1:3]] == [_STEP, _STEP]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [1, 2]
    assert [raw.agreeing for _pair, raw in index.layout_repaired_shifts] == [0, 0]
    assert "exactly two independent NCC strips" in index.notes[0]
    assert "one NCC strip" in index.notes[1]


def test_two_pair_video_run_repairs_exact_structural_dissent_then_two_strips(monkeypatch):
    """A stable strip plus exact full layout can beat a repeated animated-video match.

    This is the fourth saved Shannon transition in synthetic form.  The first raw pair votes
    twice for the video phase five pixels away from the real scroll, while one strip and the
    independently segmented card top, card bottom, and heart all land exactly on the real step.
    Its adjacent pair has the ordinary two-strip/full-layout proof, so the existing run and
    complete-assembly gates can corroborate the weaker first member.
    """
    frames = [_frame(_STEP * i) for i in range(3)]
    real_shift = item_index.estimate_shift

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(2) if frame_a == frames[i] and frame_b == frames[i + 1])
        votes = [_STEP - 5, _STEP - 5, _STEP] if pair == 0 else [_STEP, _STEP]
        return _shift_with_votes(result, votes, status=frameshift.SHIFT_NO_CONSENSUS)

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.offsets == (0, _STEP, _STEP * 2)
    assert [shift.delta_px for shift in index.shifts] == [_STEP, _STEP]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [0, 1]
    assert "one NCC strip" in index.notes[0]
    assert "exactly two independent NCC strips" in index.notes[1]


def test_dissenting_single_strip_requires_exact_top_bottom_and_heart(monkeypatch):
    """The minority-strip fallback is exact full geometry, not a wider tolerance."""
    before = item_index.segment_frame(
        _frame(0), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    after = item_index.segment_frame(
        _frame(_STEP), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    raw = _shift_with_votes(
        item_index.estimate_shift(_frame(0), _frame(_STEP), content_band=_CONTENT_BAND),
        [_STEP - 5, _STEP - 5, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP + 1)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP)))
    repaired, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert repaired.delta_px == _STEP and repaired.agreeing == 1
    assert note is not None and "one NCC strip" in note


def test_dissenting_single_strip_cannot_repair_an_isolated_pair():
    """Exact minority geometry remains only a companion to a stronger adjacent repair."""
    frames = [_frame(0), _frame(_STEP)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [_STEP - 5, _STEP - 5, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)

    repaired, notes, provenance = item_index._repair_shifts_from_layout(segmentations, (raw,))
    assert repaired == (raw,)
    assert notes == () and provenance == ()


def test_dissenting_single_strip_refuses_two_exact_structural_answers(monkeypatch):
    """Two exact singleton alternatives are ambiguity, even beside a repeated NCC cluster."""
    frames = [_frame(0), _frame(_STEP)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [_STEP - 5, _STEP - 5, _STEP, _STEP - 12],
        status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP),
        ("top", _STEP - 12), ("bottom", _STEP - 12), ("heart", _STEP - 12)))

    repaired, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert repaired is raw and note is None


def test_three_pair_video_run_repairs_two_mutually_dissenting_tail_strips(monkeypatch):
    """Exact full layout selects one tail strip even when no two-strip cluster exists.

    This is the fifth saved Shannon transition in synthetic form: two ordinary two-NCC layout
    repairs followed by raw votes twelve pixels apart.  Top, bottom and heart all identify the
    lower tail vote exactly, and only the complete three-pair run may commit it.
    """
    scrolls = (0, _STEP, _STEP * 2, _STEP * 3 - 12)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(3) if frame_a == frames[i] and frame_b == frames[i + 1])
        votes = [_STEP, _STEP] if pair < 2 else [_STEP, _STEP - 12]
        return _shift_with_votes(result, votes, status=frameshift.SHIFT_NO_CONSENSUS)

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [_STEP, _STEP, _STEP - 12]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [0, 1, 2]
    assert item_index._matched_delta_clusters(index.layout_repaired_shifts[2][1]) == ()
    assert "one NCC strip" in index.notes[2]


def test_four_pair_video_run_accepts_one_exact_structural_measured_tail(monkeypatch):
    """The full saved transition may end in one separately bounded structure-only correction."""
    scrolls = (0, _STEP, _STEP * 2, _STEP * 3 - 12, _STEP * 4 - 26)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(4) if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair < 2:
            return _shift_with_votes(result, [_STEP, _STEP],
                                     status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 2:
            return _shift_with_votes(result, [_STEP, _STEP - 12],
                                     status=frameshift.SHIFT_NO_CONSENSUS)
        true_step = scrolls[4] - scrolls[3]
        raw_step = true_step - item_index._STRUCTURAL_TAIL_CORRECTION_PX[1]
        return _shift_with_votes(
            result, [raw_step, raw_step, raw_step],
            status=frameshift.SHIFT_MEASURED, delta=raw_step)

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [
        _STEP, _STEP, _STEP - 12, _STEP - 14]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [0, 1, 2, 3]
    assert index.layout_repaired_shifts[3][1].delta_px == _STEP - 28
    assert index.shifts[3].agreeing == 0
    assert "four-pair structural tail" in index.notes[3]


@pytest.mark.parametrize("probe_failure", [False, True])
def test_five_pair_video_refusal_island_accepts_exact_multi_strip_boundaries(
        monkeypatch, probe_failure):
    """The sixth saved Shannon rupture is one measured-bracketed five-pair video island.

    The endpoint strip banks mirror the production evidence's `265x4` and `232x3` modes: the
    raw global medians are untrustworthy, but one exact repeated value at each end is selected
    independently by top, bottom, and heart geometry.  The middle is the already-supported
    2-strip, 2-strip, 1-strip grammar; its heartless transition carries one exact shared gutter.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift
    real_landmarks = item_index._structural_landmarks
    real_gutters = item_index._shared_gutter_witnesses
    heartless_digest = hashlib.sha256(frames[2]).hexdigest()

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(7) if frame_a == frames[i] and frame_b == frames[i + 1])
        votes = {
            1: [_STEP] * 4 + [_STEP - 18, _STEP - 17, _STEP - 15, _STEP - 15],
            2: [_STEP, _STEP],
            3: [_STEP, _STEP],
            4: [_STEP, _STEP + 1, _STEP + 7],
            5: [_STEP] * 3 + [_STEP - 18] * 3 + [_STEP - 14, _STEP - 10],
        }.get(pair)
        return (result if votes is None else
                _shift_with_votes(result, votes, status=frameshift.SHIFT_NO_CONSENSUS))

    def video_landmarks(before, after):
        if before.frame_digest == heartless_digest:
            return (("top", _STEP), ("top", _STEP), ("bottom", _STEP))
        return real_landmarks(before, after)

    def video_gutter(before, after, candidate):
        if before.frame_digest == heartless_digest and candidate == _STEP:
            return ((760, 813, 397, 450),)
        return real_gutters(before, after, candidate)

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    monkeypatch.setattr(item_index, "_structural_landmarks", video_landmarks)
    monkeypatch.setattr(item_index, "_shared_gutter_witnesses", video_gutter)
    if probe_failure:
        real_assemble = item_index._assemble
        calls = 0

        def contradictory_probe(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return (), ("synthetic five-pair page contradiction",), ()
            return real_assemble(*args, **kwargs)

        monkeypatch.setattr(item_index, "_assemble", contradictory_probe)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    if probe_failure:
        assert not index.usable
        assert index.layout_repaired_shifts == ()
        assert any("could not be put in one coordinate space" in failure
                   for failure in index.failures)
        return
    assert index.usable, index.failures
    assert index.offsets == _FULL_SCROLL
    assert [shift.delta_px for shift in index.shifts] == [_STEP] * 7
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [1, 2, 3, 4, 5]
    assert [shift.agreeing for shift in index.shifts[1:6]] == [4, 2, 2, 1, 3]
    assert all(raw.status == frameshift.SHIFT_NO_CONSENSUS
               for _pair, raw in index.layout_repaired_shifts)
    assert sum("five-pair full-landmark cluster boundary" in note
               for note in index.notes) == 2


def test_exact_multi_strip_boundary_is_never_a_standalone_layout_repair(monkeypatch):
    """The endpoint helper requires exact 3+ NCC plus one unambiguous full-layout answer."""
    frames = [_frame(0), _frame(_STEP)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    base = item_index.estimate_shift(*frames, content_band=_CONTENT_BAND)

    only_two = _shift_with_votes(
        base, [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
    refused, note = item_index._exact_multi_strip_shift(0, before, after, only_two)
    assert refused is only_two and note is None

    three_distinct = _shift_with_votes(
        base, [_STEP - 1, _STEP, _STEP + 1], status=frameshift.SHIFT_NO_CONSENSUS)
    refused, note = item_index._exact_multi_strip_shift(0, before, after, three_distinct)
    assert refused is three_distinct and note is None

    repeated = _shift_with_votes(
        base, [_STEP, _STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP + 1)))
    refused, note = item_index._exact_multi_strip_shift(0, before, after, repeated)
    assert refused is repeated and note is None

    ambiguous = _shift_with_votes(
        base, [_STEP] * 3 + [_STEP - 18] * 3,
        status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP),
        ("top", _STEP - 18), ("bottom", _STEP - 18), ("heart", _STEP - 18)))
    refused, note = item_index._exact_multi_strip_shift(0, before, after, ambiguous)
    assert refused is ambiguous and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP)))
    proposed, note = item_index._exact_multi_strip_shift(0, before, after, repeated)
    assert proposed.delta_px == _STEP and proposed.agreeing == 3 and note is not None
    repaired, notes, provenance = item_index._repair_shifts_from_layout(
        (before, after), (repeated,))
    assert repaired == (repeated,)
    assert notes == () and provenance == ()


def test_five_pair_video_grammar_requires_both_measured_brackets(monkeypatch):
    """Even the exact 4/2/2/1/3 evidence shape cannot begin at an unanchored capture edge."""
    frames = [_frame(_STEP * i) for i in range(6)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    bases = tuple(item_index.estimate_shift(
        frames[i], frames[i + 1], content_band=_CONTENT_BAND) for i in range(5))
    shifts = (
        _shift_with_votes(bases[0], [_STEP] * 4, status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(bases[1], [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(bases[2], [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(
            bases[3], [_STEP, _STEP + 1, _STEP + 7],
            status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(bases[4], [_STEP] * 3, status=frameshift.SHIFT_NO_CONSENSUS),
    )
    heartless_digest = segmentations[1].frame_digest
    real_landmarks = item_index._structural_landmarks
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda before, after: (
        (("top", _STEP), ("top", _STEP), ("bottom", _STEP))
        if before.frame_digest == heartless_digest else real_landmarks(before, after)))
    monkeypatch.setattr(item_index, "_shared_gutter_witnesses", lambda *_: (
        (760, 813, 397, 450),))

    repaired, notes, provenance = item_index._repair_shifts_from_layout(segmentations, shifts)
    assert repaired == shifts
    assert notes == () and provenance == ()


@pytest.mark.parametrize("break_gate", (None, "marker", "gutter", "probe"))
def test_two_video_refusals_can_share_one_raw_measured_geometry_bridge(monkeypatch, break_gate):
    """Two strict repair proposals may belong to one bounded live-video window.

    This is the seventh saved Shannon shape without profile pixels: measured anchors surround
    NCC+240, measured+263, NCC(+274,+275), measured+231.  The left refusal has exact top/bottom
    plus one shared gutter; the raw bridge has exact top/heart; the right refusal's two-vote
    centroid projects onto its unique exact +275 top/bottom/heart geometry.  The measured pair is
    retained raw and only the two refusal pairs appear in repair provenance.
    """
    scrolls = (0, 222, 462, 725, 1000, 1231, 1482)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift
    digests = [hashlib.sha256(frame).hexdigest() for frame in frames]

    def video_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 1:
            return dataclasses.replace(
                _shift_with_votes(result, [240, 240],
                                  status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        if pair == 3:
            return dataclasses.replace(
                _shift_with_votes(result, [274, 275],
                                  status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        return result

    landmarks = {
        0: (("top", 222), ("bottom", 222)),
        1: (("top", 240), ("top", 240), ("bottom", 240)),
        2: (("top", 263), ("heart", 263)),
        3: (("top", 275), ("bottom", 275), ("heart", 275)),
        4: (("top", 231), ("bottom", 231)),
        5: (("top", 251), ("bottom", 251)),
    }

    def video_landmarks(before, after):
        pair = digests.index(before.frame_digest)
        return landmarks[pair]

    def video_gutters(before, after, candidate):
        pair = digests.index(before.frame_digest)
        return (((794, 847, 554, 607),)
                if break_gate != "gutter" and pair == 1 and candidate == 240 else ())

    monkeypatch.setattr(item_index, "estimate_shift", video_pair)
    monkeypatch.setattr(item_index, "_structural_landmarks", video_landmarks)
    monkeypatch.setattr(item_index, "_shared_gutter_witnesses", video_gutters)
    if break_gate == "probe":
        real_assemble = item_index._assemble
        calls = 0

        def contradictory_probe(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return (), ("synthetic measured-bridge page contradiction",), ()
            return real_assemble(*args, **kwargs)

        monkeypatch.setattr(item_index, "_assemble", contradictory_probe)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        animation_markers=(False,) * len(frames) if break_gate == "marker" else
        tuple(i == 2 for i in range(len(frames))))

    if break_gate is not None:
        assert not index.usable
        assert index.layout_repaired_shifts == ()
        return
    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [222, 240, 263, 275, 231, 251]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [1, 3]
    assert [raw.status for _pair, raw in index.layout_repaired_shifts] == [
        frameshift.SHIFT_NO_CONSENSUS, frameshift.SHIFT_NO_CONSENSUS]
    assert [(repair.pair_index, repair.path, repair.raw_status,
             repair.raw_delta_px, repair.effective_status, repair.effective_delta_px,
             repair.marker_frames) for repair in index.repair_provenance] == [
        (1, "legacy_layout_grammar", frameshift.SHIFT_NO_CONSENSUS, None,
         frameshift.SHIFT_MEASURED, 240, ()),
        (3, "legacy_layout_grammar", frameshift.SHIFT_NO_CONSENSUS, None,
         frameshift.SHIFT_MEASURED, 275, ()),
    ]
    assert index.shifts[2].status == frameshift.SHIFT_MEASURED
    assert index.shifts[2].delta_px == 263 and index.shifts[2].agreeing >= 3
    assert any("retained raw measured bridge +263px" in note for note in index.notes)
    assert any("projection" in note and "+275px" in note for note in index.notes)


@pytest.mark.parametrize("break_gate", (None, "marker", "residual", "witness", "probe"))
def test_video_marked_two_bridge_window_projects_one_measured_video_centroid(
        monkeypatch, break_gate):
    """A confirmed video window may correct one small, fully structural measured bridge.

    This mirrors the v10 refusal at frames 6..13 without profile pixels.  The two under-quorum
    endpoints remain ordinary strict layout proposals.  Between them, +233 is already exact;
    raw +257 has a valid five-strip majority but the physical card top, bottom, heart and one NCC
    strip all say +262.  Marker evidence plus the whole-page probe are mandatory.
    """
    scrolls = (0, 261, 480, 713, 975, 1220, 1471, 1726)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift
    digests = [hashlib.sha256(frame).hexdigest() for frame in frames]

    def video_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 1:
            return dataclasses.replace(
                _shift_with_votes(result, [219, 219],
                                  status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        if pair == 3:
            if break_gate == "residual":
                votes, raw_delta = [252, 252, 253, 253, 255, 262], 253
            elif break_gate == "witness":
                votes, raw_delta = [256, 256, 257, 257, 259, 260], 257
            else:
                votes, raw_delta = [256, 256, 257, 257, 259, 262], 257
            return _shift_with_votes(
                result, votes, status=frameshift.SHIFT_MEASURED, delta=raw_delta)
        if pair == 4:
            return dataclasses.replace(
                _shift_with_votes(result, [242, 245],
                                  status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        return result

    landmarks = {
        0: (("top", 261), ("bottom", 261), ("heart", 261)),
        1: (("top", 219), ("top", 219), ("bottom", 219)),
        2: (("top", 233), ("bottom", 233), ("heart", 233)),
        3: (("top", 262), ("bottom", 262), ("heart", 262)),
        4: (("top", 245), ("bottom", 245), ("heart", 245)),
        5: (("top", 251), ("bottom", 251), ("heart", 251)),
        6: (("top", 255), ("bottom", 255), ("heart", 255)),
    }

    monkeypatch.setattr(item_index, "estimate_shift", video_pair)
    monkeypatch.setattr(
        item_index, "_structural_landmarks",
        lambda before, after: landmarks[digests.index(before.frame_digest)])
    monkeypatch.setattr(
        item_index, "_shared_gutter_witnesses",
        lambda before, after, candidate: (
            ((765, 818, 546, 599),)
            if digests.index(before.frame_digest) == 1 and candidate == 219 else ()))
    if break_gate == "probe":
        calls = 0

        def contradictory_probe(*args, **kwargs):
            nonlocal calls
            calls += 1
            return (), ("synthetic projected-bridge page contradiction",), ()

        monkeypatch.setattr(item_index, "_assemble", contradictory_probe)

    mute_markers = (() if break_gate == "marker" else (
        item_index.VideoMuteMarker(frame_index=1, x=106, y=1150, score=1.0),
        item_index.VideoMuteMarker(frame_index=2, x=106, y=931, score=1.0),
        item_index.VideoMuteMarker(frame_index=3, x=106, y=698, score=1.0),
        item_index.VideoMuteMarker(frame_index=4, x=106, y=436, score=1.0)))
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=mute_markers, _allow_frame_omission_recovery=False)

    if break_gate is not None:
        assert not index.usable
        assert index.layout_repaired_shifts == ()
        return
    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [261, 219, 233, 262, 245, 251, 255]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [1, 3, 4]
    assert index.layout_repaired_shifts[1][1].delta_px == 257
    assert any("v12 mute-card track" in note and "+262px" in note
               for note in index.notes)


@pytest.mark.parametrize("break_gate", (None, "visible_exit", "out_of_card", "witness"))
def test_positioned_mute_card_track_repairs_latest_two_refusal_capture_shape(
        monkeypatch, break_gate):
    """Only one physically tracked mute card may repair the latest Shannon shape.

    The real capture's card-local mute origins move 234px then 241px; its final origin would
    land above the content band on the +239px transition.  Once the overlay has genuinely left
    view, the same card's exact bottom/heart geometry permits a one-pixel NCC-centroid repair
    (+229 raw to +233 physical).  This must not be a boolean ``some video existed`` grammar.
    """
    scrolls = (0, 225, 459, 700, 939, 1172, 1419)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def latest_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair in (1, 3):
            step = scrolls[pair + 1] - scrolls[pair]
            return dataclasses.replace(
                _shift_with_votes(result, [step, step], status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        if pair == 4:
            # Saved pair 10->11: the animated NCC centroid is +229, but its two nearest
            # physical-card strips are +230/+233 and card bottom plus heart translate +233.
            votes = ([228, 228, 230, 230] if break_gate == "witness"
                     else [228, 228, 230, 233])
            return _shift_with_votes(result, votes, status=frameshift.SHIFT_MEASURED,
                                     delta=229)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", latest_pair)
    marker_rows = (1200, 966, 725, 486)
    if break_gate == "visible_exit":
        # All direct marker deltas remain exact, but the predicted next origin (487) is still
        # inside the content band: absence cannot be explained by the tracked overlay leaving.
        marker_rows = tuple(row + 240 for row in marker_rows)
    markers = tuple(
        item_index.VideoMuteMarker(frame_index=frame_index, x=(10 if break_gate == "out_of_card"
                                                                 else 106),
                                   y=row, score=1.0)
        for frame_index, row in zip(range(1, 5), marker_rows, strict=True))
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=markers)

    if break_gate == "witness":
        # Raw quorum can still fit the page inside its ordinary fold slack, but no exact NCC
        # witness may convert that centroid to +233.
        assert index.usable, index.failures
        assert index.shifts[4].delta_px == 229
        assert 4 not in {pair for pair, _raw in index.layout_repaired_shifts}
        return
    if break_gate is not None:
        assert not index.usable
        assert index.layout_repaired_shifts == ()
        return
    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [225, 234, 241, 239, 233, 247]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [1, 3, 4]
    assert index.layout_repaired_shifts[-1][1].delta_px == 229
    assert [(repair.pair_index, repair.path, repair.raw_status,
             repair.raw_delta_px, repair.effective_status, repair.effective_delta_px,
             repair.marker_frames) for repair in index.repair_provenance] == [
        (1, "v12_mute_card_track", frameshift.SHIFT_NO_CONSENSUS, None,
         frameshift.SHIFT_MEASURED, 234, (1, 2)),
        (3, "v12_mute_card_track", frameshift.SHIFT_NO_CONSENSUS, None,
         frameshift.SHIFT_MEASURED, 239, (3, 4)),
        (4, "v12_mute_card_track", frameshift.SHIFT_MEASURED, 229,
         frameshift.SHIFT_MEASURED, 233, (4,)),
    ]
    assert any("v12 mute-card track" in note and "+233px" in note for note in index.notes)


def test_positioned_mute_card_track_rolls_every_repair_back_on_page_contradiction(monkeypatch):
    """A real card track is evidence, not a bypass of the final whole-page assembly probe."""
    scrolls = (0, 225, 459, 700, 939, 1172, 1419)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def latest_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair in (1, 3):
            step = scrolls[pair + 1] - scrolls[pair]
            return dataclasses.replace(
                _shift_with_votes(result, [step, step], status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        if pair == 4:
            return _shift_with_votes(result, [228, 228, 230, 233],
                                     status=frameshift.SHIFT_MEASURED, delta=229)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", latest_pair)
    real_assemble = item_index._assemble
    calls = 0

    def contradictory_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (), ("synthetic mute-track page contradiction",), ()
        return real_assemble(*args, **kwargs)

    monkeypatch.setattr(item_index, "_assemble", contradictory_probe)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=tuple(
            item_index.VideoMuteMarker(frame_index=frame_index, x=106, y=row, score=1.0)
            for frame_index, row in zip(range(1, 5), (1200, 966, 725, 486), strict=True)))

    assert calls >= 1
    assert not index.usable
    assert index.layout_repaired_shifts == ()


def test_positioned_mute_card_track_survives_a_static_page_stretch(monkeypatch):
    """The track's authority must reach PAST a run of genuine zero-pixel pairs.

    Live 2026-08-16 ("Grace"): the reader had scrolled into card 2's neighbourhood and then
    dwelled there for two whole frames while its mute-marked video kept playing -- the page
    itself did not move, so both intervening pairs measured a genuine, unanimous +0px. The old
    `_track_candidate_deltas` filter (`0 < delta`) discarded that +0px candidate outright, so
    `_video_track_deltas` found zero candidates for the first static pair, its
    `len(candidates) != 1` guard fired, and the physical card was never re-added to `active` --
    killing the identity chain for the rest of the profile, even though nothing was ever wrong
    with the measurement itself. The fix widens the filter to `0 <= delta`, which lets a pair
    that TRULY measured no motion continue the same chain instead of ending it.

    The card here scrolls normally for two pairs (+225px, +234px -- both raw, unforced
    `frameshift` measurements), then the page holds still for two pairs (the marker's
    frame-local row is unchanged and the frames are pixel-identical, so `frameshift` itself
    reports two genuine, unforced `+0px` measured pairs -- nothing here is synthesised), and
    only then does the read resume with one more +241px step. That final pair is forced to
    `SHIFT_NO_CONSENSUS` with a two-strip cluster below `frameshift`'s own quorum, so the ONLY
    way it can be corrected is the video track's layout-corroborated repair -- and the ONLY way
    the track can still be holding this card's identity by that point is by having crossed the
    two static pairs first. If the track's authority stopped at the first static pair, as it did
    live, this final refusal would have nothing left to repair it.
    """
    scrolls = (0, 225, 459, 459, 459, 700)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def latest_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 4:
            return dataclasses.replace(
                _shift_with_votes(result, [241, 241], status=frameshift.SHIFT_NO_CONSENSUS),
                agreeing=2, dissenting=0)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", latest_pair)
    markers = tuple(
        item_index.VideoMuteMarker(frame_index=i, x=106, y=1425 - scroll, score=1.0)
        for i, scroll in enumerate(scrolls))
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=markers)

    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [225, 234, 0, 0, 241]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [4]
    assert index.layout_repaired_shifts[0][1].status == frameshift.SHIFT_NO_CONSENSUS
    assert index.layout_repaired_shifts[0][1].delta_px is None
    assert any("v12 mute-card track" in note and "+241px" in note for note in index.notes)

    # The control that proves the fix is what did it: restore the OLD `0 < delta` filter --
    # copied from the current `_track_candidate_deltas` source with only its final line
    # reverted, everything else identical -- and rebuild the exact same capture. The static
    # pairs no longer offer a continuation candidate, `len(candidates) != 1` fires on the very
    # first one, frame 3 never re-enters `active`, and the later refusal at pair 4 has no track
    # left to repair it: the whole capture becomes unusable, exactly as it did live.
    def old_track_candidate_deltas(pair_index, before, after, raw, *, extent_tolerance_px):
        candidates: set[int] = set()
        if raw.status == frameshift.SHIFT_MEASURED and raw.delta_px is not None:
            candidates.add(raw.delta_px)
            projected, note = item_index._measured_layout_bridge(
                pair_index, before, after, raw, allow_full_layout_projection=True,
                extent_tolerance_px=extent_tolerance_px)
            if note is not None and projected.delta_px is not None:
                candidates.add(projected.delta_px)
        elif raw.status == frameshift.SHIFT_NO_CONSENSUS:
            proposed, note = item_index._layout_repaired_shift(
                pair_index, before, after, raw, extent_tolerance_px=extent_tolerance_px)
            if note is not None and proposed.delta_px is not None:
                candidates.add(proposed.delta_px)
                projected, projection_note = item_index._project_to_exact_full_layout(
                    pair_index, before, after, raw, proposed)
                if projection_note is not None and projected.delta_px is not None:
                    candidates.add(projected.delta_px)
        return tuple(sorted(delta for delta in candidates
                            if 0 < delta <= item_index._MAX_STEP_PX))

    monkeypatch.setattr(item_index, "_track_candidate_deltas", old_track_candidate_deltas)
    regressed = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=markers)

    assert not regressed.usable
    assert regressed.layout_repaired_shifts == ()


def test_static_track_continuation_cannot_manufacture_a_zero_repair(monkeypatch):
    """Admitting +0px as a track CONTINUATION delta must never invent a +0px REPAIR.

    `_track_candidate_deltas`'s comment argues this holds because `_layout_repaired_shift` (and
    `_exact_multi_strip_shift`, used elsewhere in the same repair vocabulary) both still refuse
    every non-positive candidate. Reading their source confirms it: `_layout_repaired_shift`
    filters `0 < candidate <= _MAX_STEP_PX` on both its ordinary two-strip candidates and its
    exact-landmark singleton fallback, and `_exact_multi_strip_shift` filters the same way. So a
    +0px link can only ever be taken from a pair `frameshift` ALREADY measured as +0px (the raw
    `SHIFT_MEASURED` branch of `_track_candidate_deltas`, which merely records what was already
    measured) -- it is never something the layout-repair branch can propose for a pair that
    failed to measure at all. This is what makes the previous test's fix safe rather than merely
    convenient: letting a track ride through a CONFIRMED zero cannot be repurposed to invent one.

    The card scrolls normally for one pair (+225px, unforced), then the page holds for two
    pairs. The FIRST of those two static pairs is forced to `SHIFT_NO_CONSENSUS` with a
    two-strip cluster that itself sits at +0px -- the strongest possible temptation for exactly
    the bug this guards against, since the real page geometry (top, bottom and heart alike) truly
    is +0px here too, and every landmark a repair would want lines up. If the `0 < candidate`
    guard in `_layout_repaired_shift` were ever bypassed for this call site, this is precisely
    the pair that would turn into a fabricated +0px repair. It must not: `_layout_repaired_shift`
    and `_track_candidate_deltas` are checked directly first, then the full pipeline is checked
    to confirm the pair comes out of `build_item_index` exactly as `frameshift` left it -- a
    refusal, with no note claiming a track repair for it.
    """
    scrolls = (0, 225, 459, 459, 459)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def zero_no_consensus(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        return dataclasses.replace(
            _shift_with_votes(result, [0, 0], status=frameshift.SHIFT_NO_CONSENSUS),
            agreeing=2, dissenting=0)

    before_seg, after_seg = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in (frames[2], frames[3]))
    raw_pair2 = zero_no_consensus(frames[2], frames[3], content_band=_CONTENT_BAND)

    repaired, note = item_index._layout_repaired_shift(2, before_seg, after_seg, raw_pair2)
    assert note is None and repaired is raw_pair2
    assert item_index._track_candidate_deltas(
        2, before_seg, after_seg, raw_pair2, extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX
    ) == ()

    def latest_pair(frame_a, frame_b, **kwargs):
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 2:
            return zero_no_consensus(frame_a, frame_b, **kwargs)
        return real_shift(frame_a, frame_b, **kwargs)

    monkeypatch.setattr(item_index, "estimate_shift", latest_pair)
    markers = tuple(
        item_index.VideoMuteMarker(frame_index=i, x=106, y=1425 - scroll, score=1.0)
        for i, scroll in enumerate(scrolls))
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=markers, _allow_frame_omission_recovery=False)

    assert index.shifts[2].status == frameshift.SHIFT_NO_CONSENSUS
    assert index.shifts[2].delta_px is None
    assert 2 not in {pair for pair, _raw in index.layout_repaired_shifts}
    assert 2 not in {repair.pair_index for repair in index.repair_provenance}
    assert not any("frame 2's pair with frame 3" in note for note in index.notes)


@pytest.mark.parametrize("correction", (12, 15))
def test_structural_measured_tail_keeps_its_own_two_pixel_window(correction):
    """The structural tail neither borrows nor widens the ordinary 9..12px override band."""
    true_step = _STEP - 14
    frames = [_frame(0), _frame(true_step)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw_step = true_step - correction
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [raw_step, raw_step, raw_step], status=frameshift.SHIFT_MEASURED, delta=raw_step)

    repaired, note = item_index._structural_tail_shift(0, before, after, raw)
    assert repaired is raw and note is None


def test_structural_measured_tail_requires_one_exact_full_layout_answer(monkeypatch):
    """An inexact heart or two full structural answers cannot nominate a tail shift."""
    true_step = _STEP - 14
    frames = [_frame(0), _frame(true_step)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw_step = true_step - 14
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [raw_step, raw_step, raw_step], status=frameshift.SHIFT_MEASURED, delta=raw_step)

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", true_step), ("bottom", true_step), ("heart", true_step + 1)))
    refused, note = item_index._structural_tail_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", true_step), ("bottom", true_step), ("heart", true_step),
        ("top", true_step - 20), ("bottom", true_step - 20), ("heart", true_step - 20)))
    refused, note = item_index._structural_tail_shift(0, before, after, raw)
    assert refused is raw and note is None


def test_structural_measured_tail_cannot_repair_an_isolated_pair():
    """A valid local structural tail is still powerless without the exact preceding run."""
    true_step = _STEP - 14
    frames = [_frame(0), _frame(true_step)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw_step = true_step - 14
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [raw_step, raw_step, raw_step], status=frameshift.SHIFT_MEASURED, delta=raw_step)

    repaired, notes, provenance = item_index._repair_shifts_from_layout(segmentations, (raw,))
    assert repaired == (raw,)
    assert notes == () and provenance == ()


def test_nonclustered_single_strip_still_requires_an_exact_heart(monkeypatch):
    """Two far-apart matches do not widen the exact-three-kind singleton rule."""
    frames = [_frame(0), _frame(_STEP - 12)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [_STEP, _STEP - 12], status=frameshift.SHIFT_NO_CONSENSUS)
    assert item_index._matched_delta_clusters(raw) == ()

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP - 12), ("bottom", _STEP - 12), ("heart", _STEP - 11)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP - 12), ("bottom", _STEP - 12), ("heart", _STEP - 12)))
    repaired, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert repaired.delta_px == _STEP - 12 and repaired.agreeing == 1
    assert note is not None and "one NCC strip" in note


def test_nonclustered_single_strip_refuses_two_exact_answers(monkeypatch):
    """Independent full geometry must select one of the two dissenting strips, not both."""
    frames = [_frame(0), _frame(_STEP - 12)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [_STEP, _STEP - 12], status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP),
        ("top", _STEP - 12), ("bottom", _STEP - 12), ("heart", _STEP - 12)))

    repaired, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert repaired is raw and note is None


def test_nonclustered_single_strip_cannot_repair_an_isolated_pair():
    """Exact full geometry without an adjacent two-strip repair remains a hard refusal."""
    frames = [_frame(0), _frame(_STEP - 12)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw = _shift_with_votes(
        item_index.estimate_shift(*frames, content_band=_CONTENT_BAND),
        [_STEP, _STEP - 12], status=frameshift.SHIFT_NO_CONSENSUS)

    repaired, notes, provenance = item_index._repair_shifts_from_layout(segmentations, (raw,))
    assert repaired == (raw,)
    assert notes == () and provenance == ()


def test_two_strip_candidate_uses_the_same_median_tolerance_as_frameshift():
    """Votes 4px apart both agree with their median under the shared ±3px rule."""
    base = item_index.estimate_shift(_frame(0), _frame(_STEP), content_band=_CONTENT_BAND)
    raw = _shift_with_votes(
        base, [_STEP - 2, _STEP + 2], status=frameshift.SHIFT_NO_CONSENSUS)
    assert item_index._matched_delta_clusters(raw) == ((_STEP, (_STEP - 2, _STEP + 2)),)


def test_two_strip_candidate_rejects_overlapping_median_pairs():
    """A middle vote supporting two candidates cannot be spent as evidence for either one."""
    base = item_index.estimate_shift(_frame(0), _frame(_STEP), content_band=_CONTENT_BAND)
    raw = _shift_with_votes(
        base, [_STEP - 4, _STEP, _STEP + 4], status=frameshift.SHIFT_NO_CONSENSUS)
    assert item_index._matched_delta_clusters(raw) == ()


def test_three_pair_video_run_accepts_two_votes_straddling_their_median(monkeypatch):
    """The third saved Shannon capture is handled by the general bounded animation repair."""
    frames = [_frame(_STEP * i) for i in range(5)]
    real_shift = item_index.estimate_shift

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(4) if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 0:
            return _shift_with_votes(
                result, [_STEP - 2, _STEP + 2], status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 1:
            return _shift_with_votes(
                result, [_STEP + 10, _STEP + 10, _STEP + 10, _STEP + 10, _STEP, _STEP],
                status=frameshift.SHIFT_MEASURED, delta=_STEP + 10)
        if pair == 2:
            return _shift_with_votes(
                result, [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert [shift.delta_px for shift in index.shifts[:3]] == [_STEP, _STEP, _STEP]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [0, 1, 2]
    assert [raw.status for _pair, raw in index.layout_repaired_shifts] == [
        frameshift.SHIFT_NO_CONSENSUS, frameshift.SHIFT_MEASURED,
        frameshift.SHIFT_NO_CONSENSUS]


def test_three_pair_video_transition_accepts_one_exact_shared_gutter_companion(monkeypatch):
    """The second saved Shannon capture's complete animation transition is recoverable.

    Pair 0 has two competing two-strip clusters, but only the true shift moves both observed
    sides of one canonical gutter exactly.  That deliberately weaker proposal is bracketed by
    the already-supported measured-majority override and one-strip/full-layout repair.
    """
    frames = [_frame(_STEP * i) for i in range(5)]
    first_digest = hashlib.sha256(frames[0]).hexdigest()
    real_shift = item_index.estimate_shift
    real_landmarks = item_index._structural_landmarks

    def animated_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(4) if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 0:
            return _shift_with_votes(
                result, [_STEP, _STEP, _STEP + 7, _STEP + 7, _STEP + 13],
                status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 1:
            return _shift_with_votes(
                result, [_STEP - 10, _STEP - 10, _STEP - 10, _STEP, _STEP],
                status=frameshift.SHIFT_MEASURED, delta=_STEP - 10)
        if pair == 2:
            return _shift_with_votes(result, [_STEP], status=frameshift.SHIFT_NO_CONSENSUS)
        return result

    def old_heart_left_before_new_heart_arrived(before, after):
        if before.frame_digest == first_digest:
            return (("top", _STEP), ("bottom", _STEP))
        return real_landmarks(before, after)

    monkeypatch.setattr(item_index, "estimate_shift", animated_pair)
    monkeypatch.setattr(item_index, "_structural_landmarks",
                        old_heart_left_before_new_heart_arrived)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert [shift.delta_px for shift in index.shifts[:3]] == [_STEP, _STEP, _STEP]
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [0, 1, 2]
    assert [raw.status for _pair, raw in index.layout_repaired_shifts] == [
        frameshift.SHIFT_NO_CONSENSUS, frameshift.SHIFT_MEASURED,
        frameshift.SHIFT_NO_CONSENSUS]
    assert index.layout_repaired_shifts[1][1].delta_px == _STEP - 10
    assert "three-pair edge companion" in index.notes[0]
    assert "exact shared gutter" in index.notes[0]


def test_edge_only_candidate_cannot_bootstrap_a_short_animation_run(monkeypatch):
    """Two NCC strips plus one gutter are not a general substitute for a third landmark."""
    frames = [_frame(_STEP * i) for i in range(3)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    bases = tuple(item_index.estimate_shift(
        frames[i], frames[i + 1], content_band=_CONTENT_BAND) for i in range(2))
    edge_only = _shift_with_votes(
        bases[0], [_STEP, _STEP, _STEP + 7, _STEP + 7],
        status=frameshift.SHIFT_NO_CONSENSUS)
    one_strip = _shift_with_votes(
        bases[1], [_STEP], status=frameshift.SHIFT_NO_CONSENSUS)
    real_landmarks = item_index._structural_landmarks
    monkeypatch.setattr(
        item_index, "_structural_landmarks",
        lambda before, after: (("top", _STEP), ("bottom", _STEP))
        if before is segmentations[0] else real_landmarks(before, after))

    repaired, notes, raw = item_index._repair_shifts_from_layout(
        segmentations, (edge_only, one_strip))
    assert repaired == (edge_only, one_strip)
    assert notes == () and raw == ()


def test_edge_only_three_pair_pattern_requires_a_measured_middle_override(monkeypatch):
    """Three weak refusals cannot validate one another merely because their run length is three."""
    frames = [_frame(_STEP * i) for i in range(4)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    bases = tuple(item_index.estimate_shift(
        frames[i], frames[i + 1], content_band=_CONTENT_BAND) for i in range(3))
    shifts = (
        _shift_with_votes(bases[0], [_STEP, _STEP, _STEP + 7, _STEP + 7],
                          status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(bases[1], [_STEP, _STEP],
                          status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(bases[2], [_STEP], status=frameshift.SHIFT_NO_CONSENSUS),
    )
    real_landmarks = item_index._structural_landmarks
    monkeypatch.setattr(
        item_index, "_structural_landmarks",
        lambda before, after: (("top", _STEP), ("bottom", _STEP))
        if before is segmentations[0] else real_landmarks(before, after))

    repaired, notes, raw = item_index._repair_shifts_from_layout(segmentations, shifts)
    assert repaired == shifts
    assert notes == () and raw == ()


def test_edge_only_candidate_rejects_two_structurally_plausible_strip_clusters(monkeypatch):
    """One exact gutter must disambiguate the strip bank; two passing clusters are unknowable."""
    frames = [_frame(0), _frame(_STEP)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    base = item_index.estimate_shift(*frames, content_band=_CONTENT_BAND)
    raw = _shift_with_votes(
        base, [_STEP - 7, _STEP - 7, _STEP, _STEP],
        status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(
        item_index, "_shared_gutter_witnesses",
        lambda _before, _after, candidate: ((100, 153, 100 - candidate, 153 - candidate),))

    repaired, note = item_index._edge_only_two_strip_shift(0, before, after, raw)
    assert repaired is raw and note is None


def test_one_strip_layout_candidate_requires_all_three_landmark_kinds(monkeypatch):
    """One NCC match is proposed only with the full top/bottom/heart structural cross-check.

    This tests the stronger local gate.  The separate bounded-run gate remains responsible for
    refusing this proposal when it has no adjacent layout-assisted pair.
    """
    before = item_index.segment_frame(_frame(0), content_band=_CONTENT_BAND,
                                      like_template=_TEMPLATE,
                                      like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    after = item_index.segment_frame(_frame(_STEP), content_band=_CONTENT_BAND,
                                     like_template=_TEMPLATE,
                                     like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    raw = _shift_with_votes(
        item_index.estimate_shift(_frame(0), _frame(_STEP), content_band=_CONTENT_BAND),
        [_STEP], status=frameshift.SHIFT_NO_CONSENSUS)

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("top", _STEP)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP + 1)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP)))
    repaired, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert repaired.status == frameshift.SHIFT_MEASURED
    assert repaired.delta_px == _STEP and repaired.agreeing == 1
    assert note is not None and "one NCC strip" in note


def test_isolated_one_strip_layout_candidate_remains_a_refusal(monkeypatch):
    """A lone NCC coincidence plus layout may not create a shift without an adjacent repair."""
    frames = [_frame(_STEP * i) for i in range(3)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    base = item_index.estimate_shift(frames[0], frames[1], content_band=_CONTENT_BAND)
    isolated = _shift_with_votes(base, [_STEP], status=frameshift.SHIFT_NO_CONSENSUS)
    ordinary = item_index.estimate_shift(frames[1], frames[2], content_band=_CONTENT_BAND)

    repaired, notes, raw = item_index._repair_shifts_from_layout(
        segmentations, (isolated, ordinary))
    assert repaired == (isolated, ordinary)
    assert notes == () and raw == ()


def test_two_adjacent_one_strip_layout_candidates_cannot_corroborate_each_other(monkeypatch):
    """The companion exception still needs one independently stronger two-strip pair."""
    frames = [_frame(_STEP * i) for i in range(3)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    raw_shifts = tuple(_shift_with_votes(
        item_index.estimate_shift(frames[i], frames[i + 1], content_band=_CONTENT_BAND),
        [_STEP], status=frameshift.SHIFT_NO_CONSENSUS) for i in range(2))

    repaired, notes, raw = item_index._repair_shifts_from_layout(segmentations, raw_shifts)
    assert repaired == raw_shifts
    assert notes == () and raw == ()


@pytest.mark.parametrize("landmarks", [
    # Not enough independent pairings, even though two landmark kinds agree.
    (("top", 363), ("bottom", 363)),
    # Three pairings from one kind are not independent enough.
    (("top", 363), ("top", 363), ("top", 363)),
])
def test_layout_assisted_pair_requires_three_landmarks_across_two_types(monkeypatch, landmarks):
    before = item_index.segment_frame(_frame(0), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
                                      like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    after = item_index.segment_frame(_frame(_STEP), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
                                     like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    raw = _shift_with_votes(
        item_index.estimate_shift(_frame(0), _frame(_STEP), content_band=_CONTENT_BAND),
        [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: landmarks)
    repaired, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert repaired is raw and note is None


def test_layout_assisted_pair_rejects_large_or_ambiguous_two_strip_candidates(monkeypatch):
    before = item_index.segment_frame(_frame(0), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
                                      like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    after = item_index.segment_frame(_frame(_STEP), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
                                     like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    base = item_index.estimate_shift(_frame(0), _frame(_STEP), content_band=_CONTENT_BAND)

    too_large = _shift_with_votes(base, [400, 400], status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks",
                        lambda *_: (("top", 400), ("bottom", 400), ("heart", 400)))
    repaired, note = item_index._layout_repaired_shift(0, before, after, too_large)
    assert repaired is too_large and note is None

    ambiguous = _shift_with_votes(base, [300, 300, 363, 363], status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_: (
        ("top", 300), ("bottom", 300), ("heart", 300),
        ("top", 363), ("bottom", 363), ("heart", 363)))
    repaired, note = item_index._layout_repaired_shift(0, before, after, ambiguous)
    assert repaired is ambiguous and note is None


def test_layout_assisted_repair_rejects_isolated_runs_and_small_or_large_overrides(monkeypatch):
    """`_MAJORITY_OVERRIDE_PX` must stay a DERIVED floor, and the floor is a hard boundary.

    A correction the page fold already absorbs (<= `_EXTENT_TOLERANCE_PX`) must never override a
    measured majority: it is both unnecessary (the fold cancels it) and undetectable (the caller's
    probe has nothing to reject). One pixel past that, at `_EXTENT_TOLERANCE_PX + 1`, is the
    smallest correction the fold cannot already absorb, so it is the first one allowed through.
    """
    frames = [_frame(_STEP * i) for i in range(3)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    base = item_index.estimate_shift(frames[0], frames[1], content_band=_CONTENT_BAND)
    isolated = _shift_with_votes(base, [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
    ordinary = item_index.estimate_shift(frames[1], frames[2], content_band=_CONTENT_BAND)
    repaired, notes, raw = item_index._repair_shifts_from_layout(segmentations, (isolated, ordinary))
    assert repaired == (isolated, ordinary) and notes == () and raw == ()

    # The floor must be derived from `_EXTENT_TOLERANCE_PX`, never a hardcoded literal that could
    # silently drift out of sync with the tolerance it exists to sit one pixel past.
    assert item_index._MAJORITY_OVERRIDE_PX[0] == item_index._EXTENT_TOLERANCE_PX + 1

    before, after = segmentations[:2]
    for correction in (3, item_index._EXTENT_TOLERANCE_PX, 13):
        measured = _shift_with_votes(
            base, [_STEP - correction, _STEP - correction, _STEP - correction, _STEP, _STEP],
            status=frameshift.SHIFT_MEASURED, delta=_STEP - correction)
        shifted, note = item_index._layout_repaired_shift(0, before, after, measured)
        assert shifted is measured and note is None

    # Exactly one pixel past the tolerance is the first correction that is both necessary (the
    # fold cannot absorb it) and detectable (a wrong override would contradict the fold loudly),
    # so it must be the first one accepted.
    correction = item_index._EXTENT_TOLERANCE_PX + 1
    measured = _shift_with_votes(
        base, [_STEP - correction, _STEP - correction, _STEP - correction, _STEP, _STEP],
        status=frameshift.SHIFT_MEASURED, delta=_STEP - correction)
    shifted, note = item_index._layout_repaired_shift(0, before, after, measured)
    assert shifted is not measured and note is not None
    assert shifted.delta_px == _STEP and shifted.status == frameshift.SHIFT_MEASURED


def test_the_majority_override_floor_follows_the_folds_own_tolerance(monkeypatch):
    """The floor is "one past what the fold absorbs", so it must move when the fold does.

    The frame-omission rebuild deliberately folds at `_RECOVERY_EXTENT_TOLERANCE_PX` (9) instead
    of the default 8. A floor pinned to the module default would let that rebuild override a
    measured majority by exactly 9px — a correction its own fold cancels, which is precisely the
    unnecessary-and-undetectable case the floor exists to exclude.
    """
    frames = [_frame(_STEP * i) for i in range(2)]
    before, after = (item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    base = item_index.estimate_shift(frames[0], frames[1], content_band=_CONTENT_BAND)

    def overridden(correction, *, extent_tolerance_px):
        measured = _shift_with_votes(
            base, [_STEP - correction, _STEP - correction, _STEP - correction, _STEP, _STEP],
            status=frameshift.SHIFT_MEASURED, delta=_STEP - correction)
        shifted, _note = item_index._layout_repaired_shift(
            0, before, after, measured, extent_tolerance_px=extent_tolerance_px)
        return shifted is not measured

    recovery = item_index._RECOVERY_EXTENT_TOLERANCE_PX
    assert overridden(recovery, extent_tolerance_px=item_index._EXTENT_TOLERANCE_PX)
    assert not overridden(recovery, extent_tolerance_px=recovery)
    assert overridden(recovery + 1, extent_tolerance_px=recovery)

    # A fold that absorbs as much as the ceiling allows needs no override at all, and an empty
    # window is the fail-closed answer rather than an inverted comparison that lets everything by.
    ceiling = item_index._MAJORITY_OVERRIDE_PX[1]
    assert not overridden(ceiling, extent_tolerance_px=ceiling)


@pytest.mark.parametrize("candidate_pairs", (1, 4), ids=("isolated", "four_pair_run"))
def test_layout_assisted_run_gates_reject_nonconforming_capture_end_to_end(monkeypatch,
                                                                            candidate_pairs):
    """The builder itself must not commit an isolated or overlong repair proposal."""
    frames = [_frame(_STEP * i) for i in range(candidate_pairs + 1)]
    real_shift = item_index.estimate_shift

    def candidate_pairs_only(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        return _shift_with_votes(result, [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)

    monkeypatch.setattr(item_index, "estimate_shift", candidate_pairs_only)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable
    assert index.layout_repaired_shifts == () and index.notes == ()
    assert all(shift.status == frameshift.SHIFT_NO_CONSENSUS for shift in index.shifts)


def test_layout_assisted_prefix_replays_raw_evidence_and_preserves_provenance(monkeypatch):
    frames = [_frame(_STEP * i) for i in range(5)]
    real_shift = item_index.estimate_shift

    def repaired_prefix_pairs(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(4) if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair in (0, 1):
            return _shift_with_votes(result, [_STEP, _STEP], status=frameshift.SHIFT_NO_CONSENSUS)
        if pair == 2:
            return _shift_with_votes(result, [_STEP - 10, _STEP - 10, _STEP - 10, _STEP, _STEP],
                                     status=frameshift.SHIFT_MEASURED, delta=_STEP - 10)
        return result

    monkeypatch.setattr(item_index, "estimate_shift", repaired_prefix_pairs)
    prefix = item_index.build_item_index(
        frames[:4], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)
    extended = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        _prefix_index=prefix)

    assert prefix.usable and extended.usable
    assert [pair for pair, _raw in extended.layout_repaired_shifts] == [0, 1, 2]
    assert extended.layout_repaired_shifts[2][1].delta_px == _STEP - 10
    assert all("layout-assisted" in note for note in extended.notes[:3])


def _kinds(index):
    return [b.kind for b in index.blocks]


# =====================================================================================
# Observation-level helpers, for driving `_assemble` directly
# =====================================================================================

def _obs(frame_index, page_y0, page_y1, *, complete=True, hearts=(), kind=None):
    """One hand-written sighting. `hearts` are page rows; x is fixed because a list scroll has
    no horizontal component."""
    if kind is None:
        kind = (segment.BLOCK_SELECTABLE if hearts else segment.BLOCK_CONTEXT) if complete \
            else segment.BLOCK_PARTIAL
    return item_index.BlockObservation(
        frame_index=frame_index, page_y0=page_y0, page_y1=page_y1,
        frame_y0=page_y0, frame_y1=page_y1, kind=kind, complete=complete,
        top_observed=complete, bottom_observed=complete,
        hearts=tuple((_HEART_CX, y) for y in hearts))


def _assemble(observations, *, at_scroll_top=True, **kw):
    """Wraps `item_index._assemble`, which returns `(blocks, failures, notes)`. Most tests here
    only care about the first two, so `blocks, failures = _assemble(...)` still works: unpacking
    into two names raises on a 3-tuple, so this drops `notes` for callers that did not ask for it
    via `full=True`, rather than making every existing call site carry a third name it ignores."""
    kw.setdefault("card_x", (_CARD_X0, _CARD_X1))
    kw.setdefault("extent_tolerance_px", item_index._EXTENT_TOLERANCE_PX)
    kw.setdefault("min_item_gap_px", item_index._MIN_ITEM_GAP_PX)
    kw.setdefault("band_y0", _BAND0)
    full = kw.pop("full", False)
    blocks, failures, notes = item_index._assemble(
        list(observations), at_scroll_top=at_scroll_top, include_notes=True, **kw)
    return (blocks, failures, notes) if full else (blocks, failures)


# =====================================================================================
# The clean scroll: N items recovered exactly, once, with both index spaces
# =====================================================================================

def test_a_clean_scroll_recovers_exactly_the_items_that_were_planted():
    """THE headline. Eight frames, five physical blocks, each of them seen in three to five
    frames — and the index must contain five blocks at the world rows they were painted at, not
    the twenty-odd sightings they were seen as. Exact rows, not approximate: page coordinates are
    frame rows plus a measured shift, and the shift here is a slice offset."""
    index = _full()

    assert index.usable, index.failures
    assert len(index.blocks) == len(_ROWS)
    assert [(b.page_y0, b.page_y1) for b in index.blocks] == [(y0, y1) for _, y0, y1 in _ROWS]
    assert _kinds(index) == [item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_CONTEXT, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_SELECTABLE]

    # Every block really was seen several times, so the deduplication is doing work rather than
    # the capture happening to show each card once.
    assert all(len(b.frames) >= 3 for b in index.blocks), [b.frames for b in index.blocks]
    assert sum(len(b.observations) for b in index.blocks) > 3 * len(index.blocks)


def test_the_two_index_spaces_and_the_translation_between_them():
    """Doc 5.3: the model gets a dense 1..N over the selectable blocks, navigation counts hearts,
    and the driver keeps the private table between them. Here the vitals block sits third on the
    page and consumes NEITHER number, so model item 3 is the page's third heart but the page's
    FOURTH block."""
    index = _full()

    assert [b.model_index for b in index.blocks] == [1, 2, None, 3, 4]
    assert [b.heart_ordinal for b in index.blocks] == [1, 2, None, 3, 4]
    assert index.translation == (1, 2, 3, 4)
    assert index.heart_count == 4

    third = index.block_for(3)
    assert (third.page_y0, third.page_y1) == (_CARD3[1], _CARD3[2])
    assert index.blocks.index(third) == 3          # fourth block on the page, third model item
    assert index.heart_ordinal_for(3) == 3


def test_each_block_carries_the_frames_it_was_seen_in_and_a_frame_to_crop_it_from():
    """Requirement of doc 5.6's crop-and-verify path: a block is only croppable from a frame that
    saw ALL of it, and the index has to say which frame that is in that frame's own rows."""
    index = _full()
    card1 = index.block_for(1)

    assert card1.frames == tuple(range(len(card1.frames)))       # seen from frame 0 onward
    assert card1.complete and card1.croppable
    for sighting in card1.croppable:
        scroll = _FULL_SCROLL[sighting.frame_index]
        assert sighting.frame_y0 == _CARD1[1] - scroll
        assert sighting.frame_y1 == _CARD1[2] - scroll
        # ...and slicing those rows out of that very frame really is the card.
        frame = cv2.imdecode(np.frombuffer(_frame(scroll), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(frame[sighting.frame_y0:sighting.frame_y1, _CARD_X0:_CARD_X1],
                              _WORLDS[0][_CARD1[1]:_CARD1[2], _CARD_X0:_CARD_X1])

    # The negative: a sighting that did NOT see both edges is not offered as a crop source.
    assert any(not o.complete for o in card1.observations)
    assert all(o.complete for o in card1.croppable)


def test_the_heart_of_each_selectable_block_is_where_it_was_stamped():
    """The heart's page position is what a counting navigation eventually taps, so it is folded
    across frames like the extent is, and must come back at the world row it was painted at."""
    index = _full()
    for block, (_, _, y1) in zip(index.selectable, [r for r in _ROWS if r[0] == "card"]):
        assert block.heart is not None
        x, page_y = block.heart
        assert x == pytest.approx(_HEART_CX, abs=2)
        assert page_y == pytest.approx(y1 - _HEART_ABOVE_BOTTOM, abs=2)


def test_completeness_is_reported_and_a_short_capture_is_reported_as_truncated():
    """Requirement four: the index says what it does NOT establish. The full scroll starts at a
    confirmed top and clears the last card, so nothing is partial and nothing is truncated; the
    control is the same machinery on a prefix of the same frames, which is honest about both."""
    full, prefix = _full(), _index(_FULL_SCROLL[:3], at_scroll_top=True)

    assert full.at_scroll_top and full.reached_end and not full.truncated
    assert full.partial == () and full.complete
    assert full.tail_gap_px is not None and full.tail_gap_px > item_index._END_TAIL_GAP_PX

    assert prefix.usable, prefix.failures
    assert not prefix.reached_end and prefix.truncated and not prefix.complete
    assert prefix.partial, "the card the capture stopped inside is a partial block, not a gap"
    assert all(b.kind != item_index.ITEM_PARTIAL for b in prefix.selectable)


def test_the_same_profile_at_a_different_cadence_gives_the_same_items():
    """The invariant that says the item list belongs to the PROFILE and not to the capture. Two
    reads of the same page — one at doc 5.10.1's 363px step, one at double that, still inside
    frameshift's 900px trust window — must agree on every block, every extent and both index
    spaces. A capture-dependent answer is a fabricated or a dropped item by another name."""
    fine = _full()
    coarse = _index(tuple(2 * _STEP * i for i in range(5)), at_scroll_top=True)

    assert coarse.usable, coarse.failures
    assert len(coarse.frames) < len(fine.frames)          # genuinely fewer looks at the page
    assert [(b.page_y0, b.page_y1, b.kind, b.heart_ordinal, b.model_index) for b in coarse.blocks] \
        == [(b.page_y0, b.page_y1, b.kind, b.heart_ordinal, b.model_index) for b in fine.blocks]
    assert coarse.translation == fine.translation
    assert coarse.reached_end and not coarse.truncated

    # The blocks are also strictly ordered and non-overlapping in page space, which is what makes
    # "the k-th heart" a well-defined thing to count down to.
    rows = [(b.page_y0, b.page_y1) for b in coarse.blocks]
    assert rows == sorted(rows)
    assert all(a[1] < b[0] for a, b in zip(rows, rows[1:]))


def test_page_coordinates_come_from_the_measured_shift_and_not_from_an_assumed_step():
    """The offsets are the accumulated shift estimates. They must equal the scroll offsets the
    fixture cut the frames at — exactly — because everything the index says about position is
    downstream of them."""
    index = _full()

    assert index.offsets == _FULL_SCROLL
    assert [s.delta_px for s in index.shifts] == [_STEP] * (len(_FULL_SCROLL) - 1)
    assert index.page_span == (_BAND0, _FULL_SCROLL[-1] + _BAND1)


# =====================================================================================
# Fail loud: an unmeasurable shift must produce NO items, never an extra one
# =====================================================================================

def test_a_saturated_pair_makes_the_index_unusable_instead_of_fabricating_an_item():
    """THE regression this module exists for. Doc 5.10 measured a chained tracker recovering 9
    real items and fabricating a spurious 10th out of one large-jump tracking failure. Here the
    third frame is 1300px on from the second — past frameshift's 900px trust window — so the
    shift comes back as a refusal carrying a magnitude, and the index must contain NOTHING
    rather than a best guess at what moved."""
    index = _index((0, _STEP, _STEP + 1300), at_scroll_top=True)

    assert not index.usable
    assert index.blocks == ()
    assert index.selectable == () and index.translation == () and index.heart_count == 0

    # The refusal is the shift estimator's, quoted with the pair that produced it...
    assert index.shifts[-1].status == frameshift.SHIFT_BEYOND_WINDOW
    assert index.shifts[-1].delta_px is None and index.shifts[-1].consensus_px == 1300
    assert index.offsets == (0, _STEP, None)
    assert any("frames 1 and 2" in f and "beyond_window" in f for f in index.failures), \
        index.failures
    # ...and the measured magnitude is quoted as a magnitude, explicitly not as a shift.
    assert any("1300" in f and "may not be used as a shift" in f for f in index.failures)


def test_the_same_frames_index_cleanly_once_the_shift_is_measurable():
    """The control for the test above: nothing about those three frames is unreadable, the
    middle jump is only untrusted. Widen frameshift's window past it and the identical pixels
    produce the same five blocks — which proves the empty index came from the refusal and not
    from an inability to segment or to fold."""
    frames = (0, _STEP, _STEP + 1300)
    refused = _index(frames, at_scroll_top=True)
    allowed = _index(frames, at_scroll_top=True, trust_window_px=1500)

    assert not refused.usable and refused.blocks == ()
    assert allowed.usable, allowed.failures
    assert len(allowed.blocks) == len(_ROWS)
    assert [(b.page_y0, b.page_y1) for b in allowed.blocks[:4]] == \
        [(y0, y1) for _, y0, y1 in _ROWS[:4]]
    # Card 4 is the one this shorter capture stops inside, above its heart, so it is honestly
    # PARTIAL with no heart known — tolerated only because nothing heart-bearing is below it,
    # and reported through `partial` and `truncated` rather than through a failure.
    assert allowed.blocks[4].kind == item_index.ITEM_PARTIAL
    assert allowed.blocks[4].hearts == () and allowed.blocks[4].heart_ordinal is None
    assert allowed.translation == (1, 2, 3)
    assert allowed.heart_count == 3
    assert allowed.truncated and allowed.partial == (allowed.blocks[4],)


def test_unrelated_frames_are_refused_rather_than_indexed():
    """Low confidence, not saturation: two frames that share no content give the estimator
    nothing to agree on. Same outcome, different reason, and still not one item."""
    rng = np.random.default_rng(97)
    junk = cv2.imencode(".png", rng.integers(0, 255, size=(_H, _W), dtype=np.uint8))[1].tobytes()
    index = item_index.build_item_index(
        [_frame(0), junk], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable and index.blocks == ()
    assert index.offsets == (0, None)
    assert any("could not be put in one coordinate space" in f for f in index.failures)


def test_the_accessors_refuse_to_answer_from_an_unusable_index():
    """Doc 5.3: "treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate". A table that contradicts itself is a missing table, so the lookups raise rather
    than return None for a caller to interpret however it likes."""
    index = _index((0, _STEP, _STEP + 1300), at_scroll_top=True)

    with pytest.raises(item_index.ItemIndexError, match="unusable"):
        index.block_for(1)
    with pytest.raises(item_index.ItemIndexError, match="unusable"):
        index.heart_ordinal_for(1)

    # ...and on a perfectly good index, an index the model was never offered is still refused,
    # rather than clamped to the nearest item (doc 5.6: never substitute a different item).
    good = _full()
    assert good.block_for(4) is good.selectable[3]
    for bad in (0, -1, len(good.selectable) + 1):
        with pytest.raises(item_index.ItemIndexError, match="outside 1.."):
            good.block_for(bad)


def test_an_empty_capture_raises_rather_than_returning_an_empty_index():
    """"We captured nothing" and "this profile has no items" must never be the same value."""
    with pytest.raises(item_index.ItemIndexError, match="no frames"):
        item_index.build_item_index(
            [], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)


# =====================================================================================
# A context block occupies a page position without consuming a model index
# =====================================================================================

def test_a_context_block_takes_a_page_position_but_neither_number():
    """Doc 5.3's two tiers, at the level of the numbering. The vitals block is a real block with
    a real extent that the model may read, and it must not shift the model's 1..N — while the
    heart-bearing block below it keeps its place in BOTH sequences."""
    index = _full()
    vitals = index.blocks[2]

    assert vitals.kind == item_index.ITEM_CONTEXT
    assert (vitals.page_y0, vitals.page_y1) == (_VITALS[1], _VITALS[2])
    assert vitals.height == 215                    # doc 5.10's measured vitals block
    assert vitals.hearts == () and vitals.heart is None
    assert vitals.model_index is None and vitals.heart_ordinal is None
    assert vitals in index.context and vitals not in index.selectable

    # The block after it carries the next value of BOTH counters: a context block consumes
    # neither, so nothing below it is renumbered by its presence.
    assert index.blocks[3].model_index == 3 and index.blocks[3].heart_ordinal == 3


def test_a_heart_on_an_unbounded_block_still_consumes_a_heart_ordinal():
    """The other half of "index space belongs to the driver, selectability is policy", and the
    reason the translation table is not the identity function. A card the capture never bounded
    end to end cannot be cropped, so it is not offered to the model — but its heart is on the
    page, a counting navigation will tick it off, and dropping it from the ordinals would put
    every tap below it one card out.

    Built by starting the capture one step INTO the profile, so card 1's top corner is never
    seen, and stopping before card 4's bottom."""
    index = _index(_FULL_SCROLL[1:7], at_scroll_top=False)
    assert index.usable, index.failures

    assert _kinds(index) == [item_index.ITEM_PARTIAL, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_CONTEXT, item_index.ITEM_SELECTABLE,
                             item_index.ITEM_PARTIAL]
    assert [b.heart_ordinal for b in index.blocks] == [1, 2, None, 3, 4]
    assert [b.model_index for b in index.blocks] == [None, 1, None, 2, None]

    # The table is therefore NOT the identity: the model's item 1 is the page's SECOND heart.
    # (Read off `translation`, which describes the capture. `heart_ordinal_for` refuses on this
    # index because at_scroll_top is False — see the test directly below.)
    assert index.translation == (2, 3)
    assert index.heart_count == 4                  # every heart counted, selectable or not
    assert len(index.selectable) == 2

    # ...and the two partial blocks are reported as the incompleteness they are.
    assert len(index.partial) == 2 and index.truncated and not index.complete


def test_heart_ordinal_for_refuses_a_relative_index_while_translation_still_describes_it():
    """Doc 5.3's addendum 2026-08-12, third blocker: with `at_scroll_top=False` the ordinals are
    RELATIVE — this very capture's model item 1 is really the profile's item 2 — and the accessor
    a counting navigation calls used to answer with a confident int anyway. It now raises.

    The negative that proves WHICH mechanism refuses: the same index is otherwise perfectly
    usable, `block_for` still answers (crops and the model's list are unaffected by where the
    capture started), and `translation` still describes the capture. Only the ordinal lookup —
    the number a tap is counted towards — is withheld."""
    relative = _index(_FULL_SCROLL[1:7], at_scroll_top=False)
    assert relative.usable and not relative.at_scroll_top

    with pytest.raises(item_index.ItemIndexError, match="at_scroll_top=False"):
        relative.heart_ordinal_for(1)

    # Everything else on the same index keeps working, so this is a targeted refusal rather
    # than the whole result being condemned.
    assert relative.block_for(1) is relative.selectable[0]
    assert relative.translation == (2, 3)

    # ...and the identical capture asserted at a confirmed top answers normally, which is what
    # makes the refusal about the assertion and not about this profile.
    absolute = _full()
    assert absolute.at_scroll_top and absolute.heart_ordinal_for(1) == 1


# =====================================================================================
# Disagreement is an error, never an average
# =====================================================================================

def test_two_frames_that_disagree_about_one_cards_height_report_it():
    """Requirement three, end to end. Both frames bound card 2 end to end — the first in the
    ordinary world, the second in one where that card is 120px shorter and nothing else moved —
    so two frames measure different heights at the same page position. Exactly one of them is
    right; the index must say so rather than pick or blend, and both heights must appear in the
    complaint."""
    short = 120
    frames = [_frame(_STEP), _frame(2 * _STEP, short_card2=short)]
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=False, identity_band=None)

    height = _CARD2[2] - _CARD2[1]
    assert index.shifts[0].delta_px == _STEP, index.shifts[0].reason
    assert not index.usable
    conflicts = [f for f in index.failures if "disagree about the block" in f]
    assert conflicts, index.failures
    assert f"{height}px against {height - short}px" in conflicts[0] \
        or f"{height - short}px against {height}px" in conflicts[0], conflicts[0]

    # The control: the same two scroll positions in one consistent world are clean, so the
    # failure came from the disagreement and not from the two-frame capture.
    assert _index((_STEP, 2 * _STEP), at_scroll_top=False).usable


def test_conflicting_extents_are_reported_at_the_observation_level_too():
    """The same rule driven directly, with no correlation involved: two sightings that both claim
    to have bounded the block at one page position, at heights 121px apart. The resolved extent
    must be one a frame actually SAW — never the mean, which is the number that would then be
    cropped and stored as the post-tap verification reference."""
    blocks, failures = _assemble([
        _obs(0, 700, 1674, hearts=(1584,)),        # 974px
        _obs(1, 700, 1553, hearts=(1584,)),        # 853px
    ])

    assert len(blocks) == 1                        # one page position, not two items
    assert any("disagree about the block" in f for f in failures), failures
    assert any("averaging them would produce an extent neither frame saw" in f for f in failures)
    assert (blocks[0].page_y0, blocks[0].page_y1) in [(700, 1674), (700, 1553)]
    assert blocks[0].height != (974 + 853) // 2

    # The control: the same two sightings agreeing to within the chain tolerance are ONE block
    # with no complaint at all, so it is the size of the disagreement doing the work.
    agreeing, ok = _assemble([_obs(0, 700, 1674, hearts=(1584,)),
                              _obs(1, 702, 1676, hearts=(1586,))])
    assert ok == () and len(agreeing) == 1
    assert agreeing[0].kind == item_index.ITEM_SELECTABLE


def test_two_hearts_at_one_page_position_are_a_failure_and_not_a_choice():
    """A block with two hearts is a statement that a gutter between two cards was missed. It must
    surface, it must not be resolved by taking the first heart, and the count must still advance
    by two so nothing below it is renumbered."""
    blocks, failures = _assemble([
        _obs(0, 700, 2700, hearts=(1600, 2610)),
        _obs(1, 2800, 3600, hearts=(3510,)),
    ])

    assert blocks[0].kind == item_index.ITEM_AMBIGUOUS
    assert len(blocks[0].hearts) == 2
    assert blocks[0].heart is None                 # never hearts[0]
    assert blocks[0].model_index is None
    assert any("ambiguous" in f for f in failures), failures

    # The block below is still the THIRD heart on the page, because both of the ambiguous
    # block's hearts occupied an ordinal.
    assert blocks[0].heart_ordinal == 1 and blocks[1].heart_ordinal == 3


def test_a_fragment_reaching_past_a_bounded_card_is_reported():
    """A sighting that overruns a card whose both edges were observed is a merged block or a
    mis-tracked frame — a fragment cannot be bigger than the thing it is a fragment of."""
    blocks, failures, notes = _assemble([
        _obs(0, 700, 1674, hearts=(1584,)),
        _obs(1, 700, 2400, complete=False),
    ], full=True)

    assert len(blocks) == 1
    assert any("cannot reach past the card that contains it" in f for f in failures), failures
    assert notes == ()                    # one proven card is never a licence to discard data


def test_live_f19_to_f29_bridging_shape_is_split_on_its_bounded_cards():
    """One missed gutter must not merge two otherwise independently bounded cards.

    The two incomplete sightings model the live capture: each spans the 47px gutter, while the
    complete sightings independently establish both cards' exact extents.  The fragments are
    excluded (with provenance), not assigned to either card or silently discarded.
    """
    blocks, failures, notes = _assemble([
        # The precise live shape: f19/f20 span the missed 47px gutter; f24 independently
        # sees BOTH cards.  The rest are the visible fragments carried across f21–f29.
        _obs(19, 6368, 6709, complete=False),
        _obs(20, 6368, 6941, complete=False),
        _obs(21, 6550, 7200, complete=False),
        _obs(22, 6550, 7466, complete=False),
        _obs(23, 6550, 7477, hearts=(7387,)),
        _obs(24, 6368, 6503, hearts=(6413,)),
        _obs(24, 6550, 7477, hearts=(7387,)),
        _obs(25, 6550, 7477, hearts=(7387,)),
        _obs(26, 6643, 7477, complete=False),
        _obs(27, 6906, 7477, complete=False),
        _obs(28, 7126, 7477, complete=False),
        _obs(29, 7358, 7477, complete=False),
    ], full=True)

    assert failures == ()
    assert [(block.page_y0, block.page_y1) for block in blocks] == [
        (6368, 6503), (6550, 7477)]
    assert [block.kind for block in blocks] == [
        item_index.ITEM_SELECTABLE, item_index.ITEM_SELECTABLE]
    assert len(notes) == 2
    assert all("segmentation missed the gutter" in note for note in notes)
    assert {"frame 19", "frame 20"} == {note.split("'s")[0] for note in notes}


def test_live_f18_to_f29_caption_seam_refusal_geometry_is_repaired_conservatively():
    """Pin the exact observation ledger saved by run 34b23877bdf8.

    The segmentation-layer regression keeps the `Selfie #503` caption attached to its media in
    new captures.  This lower-level control proves that an older/foreign segmentation carrying
    the already-split evidence still fails safe without the giant contradiction shown to the
    operator: the independently bounded pieces survive in page order and only the two clipped
    bridge fragments are excluded with provenance.
    """
    blocks, failures, notes = _assemble([
        _obs(18, 6406, 6661, complete=False),
        _obs(19, 6406, 6905, complete=False),
        _obs(20, 6406, 6541),
        _obs(20, 6588, 7162, complete=False),
        _obs(21, 6406, 6541),
        _obs(21, 6588, 7383, complete=False),
        _obs(22, 6588, 7515, hearts=(7426,)),
        _obs(23, 6406, 6541),
        _obs(23, 6588, 7515, hearts=(7426,)),
        _obs(24, 6406, 6541),
        _obs(24, 6588, 7515, hearts=(7426,)),
        _obs(25, 6588, 7515, hearts=(7426,)),
        _obs(26, 6773, 7515, complete=False, hearts=(7426,)),
        _obs(27, 7002, 7515, complete=False, hearts=(7426,)),
        _obs(28, 7251, 7515, complete=False),
        _obs(29, 7496, 7515, complete=False),
    ], full=True)

    assert failures == ()
    assert [(block.page_y0, block.page_y1, block.kind, block.heart_ordinal,
             block.model_index) for block in blocks] == [
        (6406, 6541, item_index.ITEM_CONTEXT, None, None),
        (6588, 7515, item_index.ITEM_SELECTABLE, 1, 1),
    ]
    assert len(notes) == 2
    assert {"frame 18", "frame 19"} == {note.split("'s")[0] for note in notes}
    assert all("spans the proven card boundary" in note for note in notes)


def test_a_bridging_fragment_with_an_uncorroborated_heart_is_not_discarded():
    """The split needs independent evidence for every heart it would exclude."""
    blocks, failures, notes = _assemble([
        _obs(23, 6368, 6503, hearts=(6413,)),
        _obs(24, 6550, 7477, hearts=(7387,)),
        _obs(19, 6368, 6709, complete=False, hearts=(6600,)),
    ], full=True)

    assert len(blocks) == 1
    assert failures
    assert notes == ()


def test_a_fragment_that_cannot_be_placed_on_a_proven_card_is_not_discarded():
    """A grazing fragment cannot be guessed onto either side of a proven boundary."""
    blocks, failures, notes = _assemble([
        _obs(23, 6368, 6503, hearts=(6413,)),
        _obs(24, 6550, 7477, hearts=(7387,)),
        _obs(19, 6368, 6709, complete=False),
        _obs(20, 6503, 6555, complete=False),  # only 5px into card two: tolerance, not evidence
    ], full=True)

    assert len(blocks) == 1
    assert failures
    assert notes == ()


def test_two_complete_extents_from_the_same_frame_remain_a_hard_failure():
    """The bridge repair must not turn a genuinely self-contradictory frame into a split."""
    blocks, failures = _assemble([
        _obs(24, 700, 1674, hearts=(1584,)),
        _obs(24, 700, 1684, hearts=(1594,)),
    ])

    assert len(blocks) == 1
    assert any("self-contradictory frame" in failure for failure in failures), failures


# =====================================================================================
# The fabricated-item guard
# =====================================================================================

def test_two_fragments_of_one_card_that_never_overlapped_are_reported_not_counted_twice():
    """The one way folding by overlap could still invent an item: a card whose sightings never
    share a row. Two blocks closer than the smallest gutter the layout draws contradict the
    layout, and counting them as two is exactly doc 5.10's phantom."""
    gap = 20
    blocks, failures = _assemble([_obs(0, 700, 1200, complete=False, hearts=(1150,)),
                                  _obs(1, 1200 + gap, 1700, complete=False, hearts=(1650,))])

    assert any(f"only {gap}px apart" in f for f in failures), failures
    assert any("would fabricate an item" in f for f in failures)

    # The control: push them a canonical gutter apart and they ARE two items, silently.
    _, ok = _assemble([_obs(0, 700, 1200, complete=False, hearts=(1150,)),
                       _obs(1, 1200 + _GUTTER, 1700, complete=False, hearts=(1650,))])
    assert ok == ()
    assert item_index._MIN_ITEM_GAP_PX == 47       # min(52,53) - 5, from segment.py's own window
    assert gap + item_index._EXTENT_TOLERANCE_PX < item_index._MIN_ITEM_GAP_PX <= _GUTTER


def test_the_same_card_seen_in_many_frames_is_one_block():
    """Deduplication by page position, driven directly: six overlapping sightings of one card,
    from six frames, are one block that names all six."""
    blocks, failures = _assemble(
        [_obs(i, 700 + 40 * i, 1674, complete=(i == 0), hearts=(1584,)) for i in range(6)])

    assert failures == () and len(blocks) == 1
    assert blocks[0].frames == (0, 1, 2, 3, 4, 5)
    assert (blocks[0].page_y0, blocks[0].page_y1) == (700, 1674)   # the one bounded sighting
    assert blocks[0].kind == item_index.ITEM_SELECTABLE
    assert len(blocks[0].hearts) == 1


# =====================================================================================
# A heart that may be hiding, and the one place it is tolerated
# =====================================================================================

def test_an_unbounded_heartless_block_above_a_heart_is_a_failure():
    """A heartless block nobody ever bounded may be hiding a heart in the rows that were never
    in band, and a hidden heart shifts every ordinal below it. Where there IS something below,
    that is a hard stop; the control is the same block with nothing heart-bearing under it,
    which costs only coverage and is reported through `partial` instead."""
    dangerous, failures = _assemble(
        [_obs(0, 300, 900, complete=False), _obs(1, 1000, 1900, hearts=(1810,))],
        at_scroll_top=False)
    assert any("a heart may sit in the rows that were never inside" in f for f in failures)

    harmless, ok = _assemble(
        [_obs(0, 300, 900, hearts=(810,)), _obs(1, 1000, 1900, complete=False)],
        at_scroll_top=False)
    assert ok == ()
    assert harmless[1].kind == item_index.ITEM_PARTIAL
    assert dangerous[0].kind == item_index.ITEM_PARTIAL


def test_a_confirmed_scroll_top_excuses_hinges_header_and_only_that():
    """segment.py's docstring hands this job to its caller: at scroll-top Hinge's filter-chips
    header and name row sit above item 1 and came back PARTIAL on every scroll-top frame in the
    corpus. With the top affirmatively confirmed (doc 5.5) the leading block is chrome, outside
    both index spaces; without that confirmation the identical observations are an item that
    might be hiding heart 1.

    The header starts 34 rows below the band's own first row, which is the smaller of the two
    clearances the corpus measured (34px and 68px). That is not decoration: a header flush ON
    that row is the false-scroll-top shape, and the test below drives it."""
    leading = [_obs(0, _BAND0 + 34, 480, complete=False), _obs(0, 700, 1674, hearts=(1584,)),
               _obs(1, 1727, 2500, hearts=(2410,))]

    confirmed, ok = _assemble(leading, at_scroll_top=True)
    assert ok == ()
    assert confirmed[0].kind == item_index.ITEM_LEADING_CHROME
    assert confirmed[0].heart_ordinal is None and confirmed[0].model_index is None
    assert [b.model_index for b in confirmed] == [None, 1, 2]
    assert [b.heart_ordinal for b in confirmed] == [None, 1, 2]

    unconfirmed, failures = _assemble(leading, at_scroll_top=False)
    assert unconfirmed[0].kind == item_index.ITEM_PARTIAL
    assert any("a heart may sit in the rows that were never inside" in f for f in failures)

    # ...and the exception is narrow: a leading block that ever showed a heart, or was ever
    # bounded end to end, is an item and stays one even at a confirmed scroll-top.
    with_heart, _ = _assemble(
        [_obs(0, _BAND0 + 34, 480, complete=False, hearts=(400,))] + leading[1:],
        at_scroll_top=True)
    assert with_heart[0].kind == item_index.ITEM_PARTIAL and with_heart[0].heart_ordinal == 1
    bounded, _ = _assemble([_obs(0, _BAND0 + 34, 480)] + leading[1:], at_scroll_top=True)
    assert bounded[0].kind == item_index.ITEM_CONTEXT


def test_a_topmost_block_flush_against_the_band_edge_refuses_the_scroll_top_claim():
    """THE FALSE-SCROLL-TOP REGRESSION. A validation pass asserted `at_scroll_top=True` about a
    capture that began three frames down a real profile and got back a usable index with eight
    items, translation 1..8, `truncated` False and ZERO failures — whose model item 1 was really
    the profile's item 2. Every completeness property corroborated the lie, and the leading-chrome
    relabelling actively swallowed the one piece of evidence against it by calling the sliced
    fragment "chrome".

    The evidence is that a genuine scroll top has page background above its topmost block
    (measured 34px and 68px on the two calibration profiles) while a band edge slicing a card
    does not. So the identical observations, moved up by 34 rows to sit ON the band's first row,
    must now refuse — and must NOT be relabelled chrome on the way."""
    sliced = [_obs(0, _BAND0, 480 - 34, complete=False), _obs(0, 700, 1674, hearts=(1584,)),
              _obs(1, 1727, 2500, hearts=(2410,))]

    blocks, failures = _assemble(sliced, at_scroll_top=True)

    assert any("at_scroll_top was asserted" in f for f in failures), failures
    assert any("begins on the analysed band's own first row" in f for f in failures)
    assert blocks[0].kind == item_index.ITEM_PARTIAL, "the evidence must not be relabelled chrome"

    # The control that proves WHICH mechanism refused: one row of background above the same
    # fragment is all the evidence the check asks for, and the claim stands again.
    clear, ok = _assemble(
        [_obs(0, _BAND0 + 1, 480 - 34, complete=False)] + sliced[1:], at_scroll_top=True)
    assert ok == ()
    assert clear[0].kind == item_index.ITEM_LEADING_CHROME

    # And the check is scoped to the claim: with no scroll-top asserted, flush content is just
    # the top of a window and the ordinals are relative, which is what `at_scroll_top=False` is.
    relative, failures = _assemble(sliced, at_scroll_top=False)
    assert not any("at_scroll_top was asserted" in f for f in failures)


# =====================================================================================
# Per-frame segmentation failures are carried forward, not swallowed
# =====================================================================================

def test_a_frame_that_contradicted_itself_makes_the_index_unusable():
    """segment.py's own failures are the index-corrupting kind — a two-heart block, a heart no
    block contains. They must reach `ItemIndex.failures` with the frame number attached, because
    a page that has to be self-consistent cannot be folded out of a frame that is not."""
    world = _WORLDS[0].copy()
    # A second heart on card 1, inside the same block: a gutter would have to be missed for this
    # to happen on the phone, and that is precisely what makes it a failure.
    th, tw = _TEMPLATE.shape
    cy = _CARD1[1] + 200
    world[cy - th // 2: cy - th // 2 + th, _HEART_CX - tw // 2: _HEART_CX - tw // 2 + tw] = \
        _TEMPLATE
    gray = world[0:_H].copy()
    gray[:_BAND0] = _CHROME_TOP
    gray[_BAND1:] = _CHROME_BOTTOM
    two_hearts = cv2.imencode(".png", gray)[1].tobytes()

    index = item_index.build_item_index(
        [two_hearts, _frame(_STEP)], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert not index.usable
    assert any(f.startswith("frame 0:") and "ambiguous" in f for f in index.failures), \
        index.failures
    assert not index.frames[0].ok and index.frames[1].ok


# =====================================================================================
# Module properties
# =====================================================================================

def test_the_module_is_a_leaf_and_does_not_pull_in_the_driver():
    """item_index.py must stay importable without hinge.py — hinge.py is the eventual IMPORTER,
    and segment.py's one call back into it is deliberately deferred to call time. Checked in a
    fresh interpreter, since this test session has hinge.py loaded already."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import operation_love.drivers.item_index as m; "
         "print('operation_love.drivers.hinge' in sys.modules); "
         "print(m.build_item_index.__name__)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "build_item_index"], out.stdout + out.stderr


def test_the_gutter_geometry_is_shared_with_segment_and_not_re_declared():
    """The minimum gap between two blocks IS segment.py's gutter window, or the fabricated-item
    guard would be measuring against a constant free to drift from the one that cut the blocks."""
    assert item_index._MIN_ITEM_GAP_PX == min(segment._GUTTER_PX) - segment._GUTTER_TOLERANCE_PX
    assert item_index._END_TAIL_GAP_PX == max(segment._GUTTER_PX) + segment._GUTTER_TOLERANCE_PX
    assert item_index._EXTENT_TOLERANCE_PX * 2 < item_index._MIN_ITEM_GAP_PX


def test_usable_is_exactly_the_absence_of_failures():
    """`usable` must not drift into meaning anything else — it is the one-line form of "did
    anything contradict anything", on every index this file builds."""
    for index in (_full(), _index((0, _STEP, _STEP + 1300), at_scroll_top=True)):
        assert index.usable is (index.failures == ())
