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
from itertools import pairwise

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
_FRAMES: dict[tuple[int, int, bool], bytes] = {}

# HINGE 10.1.0'S PINNED PER-PROFILE HEADER, to scale, INSIDE the analysed band.
#
# `_CHROME_TOP`/`_CHROME_BOTTOM` above are static chrome OUTSIDE the band, which no rule here has
# ever had to think about. This is the defect the 2026-08-28 incident is about, and the difference
# is the whole point: a strip that does not scroll but IS inside the analysed rows has a page
# position of `frame_row + offset` — a different, fabricated page row on every frame.
#
# Geometry from the incident capture `data/hinge_debug/8fb11094ef4d` (segment.py's
# `_unanchored_leading_island_rows` docstring): the band opens on 68 rows of page background, then
# a 43px header strip at frame rows 368..411, then 106 more rows of page background, then the
# scrolling content clipped at row 517. The strip is painted narrower than the card (801px of the
# 974px card width, against the measured 961..968) so segment.py's clause (c) can see it is not a
# card slice, and it is heartless (clause (e)). It is painted from one fixed buffer at full frame
# width, so every frame's rows 368..411 are BYTE-identical — which is what makes its
# `content_digest` the same on all of them, exactly as measured on the real capture.
_PINNED_BG = 250                                   # one flat page level, identical on every frame
_PINNED_Y0, _PINNED_Y1 = 368, 411                  # the incident capture's own frame rows
_PINNED_CLEARANCE_PX = 106                         # its measured gap above the clipped card
_PINNED_CONTENT_Y0 = _PINNED_Y1 + _PINNED_CLEARANCE_PX      # 517: where real content resumes
_PINNED_X0, _PINNED_X1 = 100, 900                  # 801px of 974: never the card's full width
_PINNED_STRIP = np.full((_PINNED_Y1 - _PINNED_Y0, _W), _PINNED_BG, dtype=np.uint8)
_PINNED_STRIP[:, _PINNED_X0:_PINNED_X1] = np.random.default_rng(7).integers(
    60, 200, size=(_PINNED_Y1 - _PINNED_Y0, _PINNED_X1 - _PINNED_X0), dtype=np.uint8)


def _frame(scroll: int, *, short_card2: int = 0, pinned_prefix: bool = False) -> bytes:
    """The 1080x2400 window of the world at `scroll`, as PNG. Content at world row `w` lands on
    frame row `w - scroll`, so `_frame(s)` then `_frame(s + d)` is a forward scroll of `d`.

    `pinned_prefix` paints Hinge 10.1.0's screen-pinned profile header over the top of the
    analysed band — see `_PINNED_STRIP` — which does NOT translate with `scroll`. The world's own
    content is unchanged below row `_PINNED_CONTENT_Y0`, so the same page is still there to be
    indexed; what changes is that every frame now opens with a strip whose page position cannot be
    derived from that frame alone."""
    key = (scroll, short_card2, pinned_prefix)
    if key not in _FRAMES:
        if short_card2 not in _WORLDS:
            _WORLDS[short_card2] = _build_world(short_card2=short_card2)
        gray = _WORLDS[short_card2][scroll:scroll + _H].copy()
        gray[:_BAND0] = _CHROME_TOP
        gray[_BAND1:] = _CHROME_BOTTOM
        if pinned_prefix:
            gray[_BAND0:_PINNED_CONTENT_Y0] = _PINNED_BG
            gray[_PINNED_Y0:_PINNED_Y1] = _PINNED_STRIP
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


def test_one_failed_pair_recovery_widens_a_two_read_bridge_past_normal_trust_window(monkeypatch):
    """A normal enumeration step may be 540px, so omitting one frame can require a 1080px
    bridge even though *each* real gesture stayed inside frameshift's 900px ordinary window.

    Mallory's refusal had this shape: frames 11/12 had two good strip witnesses at an ordinary
    roughly-493px movement, while the surrounding planned gestures were 456px and 469px.  The
    old 363px fixture above never exposed the resulting 900px boundary: its two-read bridge is
    only 726px.  The bridge is a fresh measurement, not a presumed sum, so its trust window must
    be widened to the known two-gesture enumeration ceiling before deciding whether either
    implicated intermediate frame can be omitted.
    """
    step = 480
    scrolls = tuple(step * i for i in range(6))
    frames = [_frame(scroll) for scroll in scrolls]
    failed_pair = (2, 3)
    real_shift = item_index.estimate_shift
    bridge_windows = []

    def one_two_witness_pair(frame_a, frame_b, **kwargs):
        if (frame_a, frame_b) in ((frames[1], frames[3]), (frames[2], frames[4])):
            bridge_windows.append(kwargs.get("trust_window_px"))
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) == (frames[failed_pair[0]], frames[failed_pair[1]]):
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic two-witness refusal")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", one_two_witness_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.recovered_from_pair == failed_pair
    assert index.recovery_bridge is not None
    bridge = index.shifts[index.recovery_bridge[0]]
    assert bridge.status == frameshift.SHIFT_MEASURED
    assert bridge.delta_px == step * 2
    assert bridge_windows and all(window >= step * 2 for window in bridge_windows)


def test_failed_pair_recovery_rejects_bridge_above_its_multi_read_ceiling(monkeypatch):
    """A caller's wider window may expose a bridge, never authorize an impossible one."""
    frames = [_frame(_STEP * i) for i in range(6)]
    failed_pair = (2, 3)
    oversized_bridge = 2 * item_index._enum_step_ceiling((
        item_index.segment_frame(
            frames[0], content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD),)) + 1
    real_shift = item_index.estimate_shift

    def refusal_with_oversized_bridges(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next((i for i in range(5)
                     if frame_a == frames[i] and frame_b == frames[i + 1]), None)
        if pair == failed_pair[0]:
            return dataclasses.replace(
                result, delta_px=None, status=frameshift.SHIFT_NO_CONSENSUS,
                consensus_px=None, reason="synthetic initial refusal")
        # These are precisely the two fresh bridge measurements attempted after omitting either
        # implicated frame. A deliberately permissive fold below isolates the bridge-cap gate:
        # without it, a caller's 2000px window would let this otherwise rebuildable candidate in.
        if ((frame_a, frame_b) in ((frames[1], frames[3]), (frames[2], frames[4]))):
            return dataclasses.replace(
                result, delta_px=oversized_bridge, status=frameshift.SHIFT_MEASURED,
                consensus_px=oversized_bridge, reason="synthetic oversized bridge")
        return result

    def permissive_assemble(*_args, include_notes=False, **_kwargs):
        return ((), (), ()) if include_notes else ((), ())

    monkeypatch.setattr(item_index, "estimate_shift", refusal_with_oversized_bridges)
    monkeypatch.setattr(item_index, "_assemble", permissive_assemble)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        trust_window_px=2000)

    assert not index.usable
    assert index.recovered_from_pair is None
    assert index.source_frame_indices == tuple(range(len(frames)))


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


def test_segmentation_recovery_can_include_one_adjacent_hidden_missed_gutter(monkeypatch):
    """The real missed gutter may be flagged in one frame and only contradict the fold in its
    neighbour.  Recover only when omitting that exact two-frame window yields the sole clean,
    freshly bridged page; an apparently clean segmentation is not otherwise disposable.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    bad_frame, hidden_neighbour = 4, 5
    real_segment = item_index.segment_frame

    def one_flagged_and_one_hidden_contradiction(frame, **kwargs):
        result = real_segment(frame, **kwargs)
        if frame == frames[bad_frame]:
            return dataclasses.replace(result, failures=("synthetic two hearts in one block",))
        if frame == frames[hidden_neighbour]:
            # A wrong top edge is plausible after the same low-contrast gutter.  It is not a
            # per-frame contradiction; only the page fold can prove it disagrees with frame 3.
            return dataclasses.replace(
                result, blocks=tuple(dataclasses.replace(block, y0=block.y0 + 20)
                                     for block in result.blocks))
        return result

    monkeypatch.setattr(item_index, "segment_frame", one_flagged_and_one_hidden_contradiction)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 2, 3, 6, 7)
    assert index.recovered_from_segmentation_frames == (bad_frame, hidden_neighbour)
    assert index.recovery_bridge == (3, 6)
    assert index.shifts[3].status == frameshift.SHIFT_MEASURED
    assert "adjacent frame's geometry could not be reconciled" in index.recovery_reason


def test_fold_proven_partial_overruns_rebuild_over_their_exact_measured_bridge(monkeypatch):
    """A gutter can look locally plausible until another frame bounds its card.

    This is the Elise failure class: no segmenter error and no failed shift, but two partial
    sightings are proved by the page fold to extend past a bounded card.  Only their exact,
    contiguous interior run may be dropped; the reduced capture must still measure its bridge
    and rebuild cleanly.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    bad_frames = (4, 5)
    real_assemble = item_index._assemble
    calls = 0

    def fold_reports_exact_overruns(*args, **kwargs):
        nonlocal calls
        calls += 1
        blocks, failures, notes = real_assemble(*args, **kwargs)
        if calls == 1:
            failures = (
                "frame 4 sees page rows 2800..3200 (its own frame rows 1000..1400) where "
                "the block was bounded at 2850..3100 by frame 6 (frame rows 700..950) — a "
                "fragment cannot reach past the card that contains it, so either a gutter was "
                "missed or these two sightings are not the same block",
                "frame 5 sees page rows 2800..3240 (its own frame rows 900..1340) where the "
                "block was bounded at 2850..3100 by frame 6 (frame rows 700..950) — a fragment "
                "cannot reach past the card that contains it, so either a gutter was missed or "
                "these two sightings are not the same block",
            )
        return blocks, failures, notes

    monkeypatch.setattr(item_index, "_assemble", fold_reports_exact_overruns)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.source_frame_indices == (0, 1, 2, 3, 6, 7)
    assert index.recovered_from_fold_contradiction_frames == bad_frames
    assert index.recovered_from_segmentation_frames == ()
    assert index.recovery_bridge == (3, 6)
    assert index.shifts[3].status == frameshift.SHIFT_MEASURED
    assert "partial sighting crossed a bounded card" in index.recovery_reason


def test_fold_omission_recovery_rejects_any_unrelated_failure():
    overrun = (
        "frame 29 sees page rows 8658..8816 (its own frame rows 1942..2100) where the block "
        "was bounded at 8710..9632 by frame 34 (frame rows 989..1911) — a fragment cannot "
        "reach past the card that contains it, so either a gutter was missed or these two "
        "sightings are not the same block")

    assert item_index._fragment_overrun_frame_indices((overrun,)) == (29,)
    assert item_index._fragment_overrun_frame_indices(
        (overrun, "block at page rows 8000..9000 is ambiguous")) == ()


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

    Only `estimate_shift`'s own status/delta are forced to refuse here; the underlying strip
    bank is real, untouched geometry, so each pair independently clears the two-pair exact-multi
    grammar and is repaired in place.  That is strictly better than omitting the shared frame:
    every frame's blocks are kept instead of one being discarded.  The old exactly-one-refusal
    gate never attempted any bridge at all for this shape; frame omission remains the fallback
    for when a pair's own evidence -- not just its raw quorum -- is genuinely insufficient (see
    `test_transient_frame_recovery_allows_measured_bridge_alignment_slack`).
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
    assert index.source_frame_indices == tuple(range(len(frames)))
    assert index.recovery_bridge is None
    assert index.recovered_from_pairs == ()
    assert index.shifts[3].status == frameshift.SHIFT_MEASURED
    assert index.shifts[3].delta_px == _STEP
    assert index.shifts[4].status == frameshift.SHIFT_MEASURED
    assert index.shifts[4].delta_px == _STEP
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [3, 4]
    assert sum("exact-multi full-landmark cluster boundary" in note
               for note in index.notes) == 2


def test_transient_frame_recovery_allows_measured_bridge_alignment_slack(monkeypatch):
    """A strong direct bridge may carry a little more drift than an ordinary one-step chain.

    The reported 2026-08-15 capture had two adjacent two-witness refusals around one frame.  Its
    direct neighbour bridge was independently measured by 4/4 eligible strips, but put the
    shared bounded card 9px past the extent seen on the other side.  That is still far inside a
    real Hinge gutter and must be absorbed as chain slack; rejecting the complete rebuild loses
    every item even though the recovery used no refused consensus or assumed offset.

    Both refused pairs here carry an emptied strip bank (no votes at all), unlike the sibling
    `test_one_transient_frame_can_recover_both_adjacent_refused_pairs`: this is specifically
    exercising frame OMISSION's own slack tolerance, so each pair's own evidence must be too
    weak for the two-pair exact-multi grammar to repair it directly -- otherwise that stronger,
    frame-preserving path would recover it before omission is ever attempted, and the drifted
    bridge measured here would never be reached at all.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift
    refused = {(frames[3], frames[4]), (frames[4], frames[5])}
    bridge = (frames[3], frames[5])

    def transient_middle_with_drifted_bridge(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) in refused:
            return _shift_with_votes(result, [], status=frameshift.SHIFT_NO_CONSENSUS)
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
    """Two adjacent failures identify a candidate frame; they do not authorize dropping it.

    All three pairs -- both refusals and the direct bridge -- carry an emptied strip bank, so
    neither the two-pair exact-multi grammar nor frame omission has any real evidence to work
    from; this stays a hard refusal on every recovery path.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift
    refused = {(frames[3], frames[4]), (frames[4], frames[5]), (frames[3], frames[5])}

    def transient_and_bridge(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        if (frame_a, frame_b) in refused:
            return _shift_with_votes(result, [], status=frameshift.SHIFT_NO_CONSENSUS)
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

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP + 1)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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

    def video_landmarks(before, after, **__):
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
    assert sum("exact-multi full-landmark cluster boundary" in note
               for note in index.notes) == 2


def test_two_pair_video_tail_accepts_exact_multi_strip_boundaries(monkeypatch):
    """The Hailey capture: a scrolling video card's own motion pollutes two adjacent pairs'
    raw votes, each still leaving 3+ NCC strips landing exactly on the true step with full
    top/bottom/heart corroboration.  Unlike the five-pair island, there is no interior pair to
    bridge here -- every pair in this window already independently clears
    `_exact_multi_strip_shift`'s own bar -- so it needs only an ordinary measured bracket on
    both sides, not the five-pair island's interior 2-strip/2-strip/1-strip shape.
    """
    frames = [_frame(scroll) for scroll in _FULL_SCROLL]
    real_shift = item_index.estimate_shift

    def video_tail_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(7) if frame_a == frames[i] and frame_b == frames[i + 1])
        votes = {
            2: [_STEP, _STEP, _STEP, _STEP - 40, _STEP - 75, _STEP - 110],
            3: [_STEP, _STEP, _STEP, _STEP - 55, _STEP - 95, _STEP - 130],
        }.get(pair)
        return (result if votes is None else
                _shift_with_votes(result, votes, status=frameshift.SHIFT_NO_CONSENSUS))

    monkeypatch.setattr(item_index, "estimate_shift", video_tail_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert index.offsets == _FULL_SCROLL
    assert [shift.delta_px for shift in index.shifts] == [_STEP] * 7
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [2, 3]
    assert [shift.agreeing for shift in index.shifts[2:4]] == [3, 3]
    assert all(raw.status == frameshift.SHIFT_NO_CONSENSUS
               for _pair, raw in index.layout_repaired_shifts)
    assert sum("exact-multi full-landmark cluster boundary" in note
               for note in index.notes) == 2


def test_two_pair_exact_multi_window_needs_measured_brackets():
    """The same two-pair evidence shape at the unanchored capture edge still cannot be admitted.

    Every other repair grammar in this module requires an ordinary raw-measured pair immediately
    outside the window it rebuilds; the two-pair exact-multi grammar is no exception even though
    neither of its own two pairs is individually weak.  Here the corrupted pair-shape consumes
    the entire shifts list, so there is no room for either bracket.
    """
    frames = [_frame(_STEP * i) for i in range(3)]
    segmentations = tuple(item_index.segment_frame(
        frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames)
    bases = tuple(item_index.estimate_shift(
        frames[i], frames[i + 1], content_band=_CONTENT_BAND) for i in range(2))
    shifts = (
        _shift_with_votes(
            bases[0], [_STEP, _STEP, _STEP, _STEP - 40, _STEP - 75, _STEP - 110],
            status=frameshift.SHIFT_NO_CONSENSUS),
        _shift_with_votes(
            bases[1], [_STEP, _STEP, _STEP, _STEP - 55, _STEP - 95, _STEP - 130],
            status=frameshift.SHIFT_NO_CONSENSUS),
    )
    repaired, notes, provenance = item_index._repair_shifts_from_layout(segmentations, shifts)
    assert repaired == shifts
    assert notes == () and provenance == ()


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
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP + 1)))
    refused, note = item_index._exact_multi_strip_shift(0, before, after, repeated)
    assert refused is repeated and note is None

    ambiguous = _shift_with_votes(
        base, [_STEP] * 3 + [_STEP - 18] * 3,
        status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP),
        ("top", _STEP - 18), ("bottom", _STEP - 18), ("heart", _STEP - 18)))
    refused, note = item_index._exact_multi_strip_shift(0, before, after, ambiguous)
    assert refused is ambiguous and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda before, after, **__: (
        (("top", _STEP), ("top", _STEP), ("bottom", _STEP))
        if before.frame_digest == heartless_digest else real_landmarks(before, after)))
    monkeypatch.setattr(item_index, "_shared_gutter_witnesses", lambda *_, **__: (
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

    def video_landmarks(before, after, **__):
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
        lambda before, after, **__: landmarks[digests.index(before.frame_digest)])
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


@pytest.mark.parametrize("raw_status", (frameshift.SHIFT_NO_EVIDENCE,
                                         frameshift.SHIFT_NO_CONSENSUS))
@pytest.mark.parametrize("break_gate", (None, "marker_x", "anchor", "probe"))
def test_positioned_mute_card_track_bridges_one_directly_proved_video_pair(
        monkeypatch, break_gate, raw_status):
    """A video may bridge one directly proved refusal without dropping cards below it.

    This is the Marina shape: the animated card leaves no usable NCC strip between frames 2 and
    3, but its app-owned control moves +241px at the same x, remains inside one segmented card
    in both frames, and one card edge/heart corroborates that distance. This is sufficient for
    either an all-weak or an under-quorum raw refusal. The test uses a normal complete synthetic
    profile around that one forced failure so a positive proves the rebuilt index keeps items
    both above and below the video rather than returning a trusted prefix.
    """
    scrolls = (0, 225, 459, 700, 939, 1172)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def video_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 2:
            return dataclasses.replace(
                _shift_with_votes(result, [], status=raw_status),
                confidence=0.0,
                reason="synthetic animated card obscured every NCC strip")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", video_pair)
    if break_gate == "probe":
        monkeypatch.setattr(
            item_index, "_assemble",
            lambda *args, **kwargs: ((), ("synthetic direct-marker page contradiction",), ()))

    marker_x = 107 if break_gate == "marker_x" else 106
    markers = tuple(
        item_index.VideoMuteMarker(
            frame_index=frame_index,
            x=(marker_x if frame_index == 3 else 106),
            y=1425 - scroll,
            score=1.0)
        for frame_index, scroll in zip(range(5), scrolls[:5], strict=True))
    if break_gate == "anchor":
        real_anchor_count = item_index._track_anchor_count

        def marker_only_anchor_count(before, after, delta, *, before_marker, after_marker):
            if before_marker is not None and after_marker is not None:
                return 1
            return real_anchor_count(
                before, after, delta,
                before_marker=before_marker, after_marker=after_marker)

        monkeypatch.setattr(item_index, "_track_anchor_count", marker_only_anchor_count)

    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=markers, _allow_frame_omission_recovery=False)

    if break_gate is not None:
        assert not index.usable
        assert index.layout_repaired_shifts == ()
        return
    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [225, 234, 241, 239, 233]
    assert [(pair, raw.status, raw.delta_px)
            for pair, raw in index.layout_repaired_shifts] == [
                (2, raw_status, None)]
    repair = index.repair_provenance[0]
    assert (repair.pair_index, repair.path, repair.raw_status, repair.effective_status,
            repair.effective_delta_px, repair.marker_frames) == (
                2, "v12_mute_card_track", raw_status,
                frameshift.SHIFT_MEASURED, 241, (2, 3))
    assert "direct marker bridge" in index.shifts[2].reason


@pytest.mark.parametrize("break_gate", (None, "still_visible", "weak_cluster"))
def test_positioned_mute_card_track_reconnects_one_boundary_below_a_video(
        monkeypatch, break_gate):
    """The first transition below a proved video exit may rejoin the ordinary card sequence.

    The mute control is present through frame 4. Its measured +233px exit puts its predicted
    origin at row 253, above the 300px content band. The following pair has a deliberately
    refused raw result despite three identical +247px strips; that one value may be used only
    because the actively tracked card's heart also translates by +247px. This is the bridge that
    preserves cards *below* a video rather than stopping at a safe prefix.
    """
    scrolls = (0, 225, 459, 700, 939, 1172, 1419)
    frames = [_frame(scroll) for scroll in scrolls]
    real_shift = item_index.estimate_shift

    def exit_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next(i for i in range(len(frames) - 1)
                    if frame_a == frames[i] and frame_b == frames[i + 1])
        if pair == 5:
            votes = [247, 247, 247] if break_gate != "weak_cluster" else [247, 247, 248]
            return dataclasses.replace(
                _shift_with_votes(result, votes, status=frameshift.SHIFT_NO_CONSENSUS),
                confidence=1 / 3, agreeing=3, dissenting=6, eligible=9,
                reason="synthetic video-exit strip contradiction")
        return result

    monkeypatch.setattr(item_index, "estimate_shift", exit_pair)
    marker_rows = tuple(1425 - scroll for scroll in scrolls[:5])
    if break_gate == "still_visible":
        # The control's expected next row remains in the band, so an absent observation is an
        # identity break, not permission to bridge below it.
        marker_rows = tuple(row + 240 for row in marker_rows)
    markers = tuple(
        item_index.VideoMuteMarker(frame_index=i, x=106, y=row, score=1.0)
        for i, row in enumerate(marker_rows))
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        video_mute_markers=markers, _allow_frame_omission_recovery=False)

    if break_gate is not None:
        assert not index.usable
        assert index.layout_repaired_shifts == ()
        return
    assert index.usable, index.failures
    assert index.offsets == scrolls
    assert [shift.delta_px for shift in index.shifts] == [225, 234, 241, 239, 233, 247]
    assert [(pair, raw.status, raw.delta_px)
            for pair, raw in index.layout_repaired_shifts] == [
                (5, frameshift.SHIFT_NO_CONSENSUS, None)]
    assert "post-exit bridge" in index.shifts[5].reason


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
    def old_track_candidate_deltas(pair_index, before, after, raw, *, extent_tolerance_px,
                                   max_step_px=None):
        # `max_step_px` is intentionally unused: this reimplementation reproduces the OLD,
        # pre-fix behaviour being regression-tested (the `0 < delta` filter, not the later
        # pitch-relative bound), and that behaviour hardcoded `item_index._MAX_STEP_PX` rather
        # than threading a caller-supplied bound.
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

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", true_step), ("bottom", true_step), ("heart", true_step + 1)))
    refused, note = item_index._structural_tail_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", _STEP - 12), ("bottom", _STEP - 12), ("heart", _STEP - 11)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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

    def old_heart_left_before_new_heart_arrived(before, after, **__):
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
        lambda before, after, **__: (("top", _STEP), ("bottom", _STEP))
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
        lambda before, after, **__: (("top", _STEP), ("bottom", _STEP))
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

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", _STEP), ("bottom", _STEP), ("top", _STEP)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
        ("top", _STEP), ("bottom", _STEP), ("heart", _STEP + 1)))
    refused, note = item_index._layout_repaired_shift(0, before, after, raw)
    assert refused is raw and note is None

    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: landmarks)
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
                        lambda *_, **__: (("top", 400), ("bottom", 400), ("heart", 400)))
    repaired, note = item_index._layout_repaired_shift(0, before, after, too_large)
    assert repaired is too_large and note is None

    ambiguous = _shift_with_votes(base, [300, 300, 363, 363], status=frameshift.SHIFT_NO_CONSENSUS)
    monkeypatch.setattr(item_index, "_structural_landmarks", lambda *_, **__: (
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


def test_isolated_two_strip_full_layout_repair_needs_full_measured_brackets(monkeypatch):
    """One repaint-damaged pair is safe only when complete geometry brackets it.

    This is the Mallory frames 11/12 shape: only two eligible NCC strips survive, but both
    independently land on the exact top/bottom/heart shift and the immediately adjacent raw
    measurements carry that same complete structural proof.  A missing kind on either bracket
    must leave the original no-consensus result untouched.
    """
    frames = [_frame(_STEP * i) for i in range(5)]
    real_shift = item_index.estimate_shift

    def one_repaint_pair(frame_a, frame_b, **kwargs):
        result = real_shift(frame_a, frame_b, **kwargs)
        pair = next((i for i in range(4)
                     if frame_a == frames[i] and frame_b == frames[i + 1]), None)
        return (dataclasses.replace(
                    _shift_with_votes(result, [_STEP, _STEP],
                                      status=frameshift.SHIFT_NO_CONSENSUS),
                    agreeing=2, dissenting=0, eligible=2, confidence=1.0)
                if pair == 2 else result)

    monkeypatch.setattr(item_index, "estimate_shift", one_repaint_pair)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        scroll_top_signal_confirmed=True, _allow_frame_omission_recovery=False)

    assert index.usable, index.failures
    assert [shift.delta_px for shift in index.shifts] == [_STEP] * 4
    assert [pair for pair, _raw in index.layout_repaired_shifts] == [2]
    assert index.layout_repaired_shifts[0][1].eligible == 2

    monkeypatch.setattr(item_index, "_scroll_top_evidence",
                        lambda *_args, **_kwargs: "synthetic missing scroll-top corner")
    unconfirmed = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        _allow_frame_omission_recovery=False)
    assert not unconfirmed.usable
    assert unconfirmed.layout_repaired_shifts == ()

    real_landmarks = item_index._structural_landmarks
    right_bracket_digest = hashlib.sha256(frames[3]).hexdigest()

    def incomplete_right_bracket(before, after, **kwargs):
        landmarks = real_landmarks(before, after, **kwargs)
        return (tuple((kind, delta) for kind, delta in landmarks if kind != "heart")
                if before.frame_digest == right_bracket_digest else landmarks)

    monkeypatch.setattr(item_index, "_structural_landmarks", incomplete_right_bracket)
    refused = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
        scroll_top_signal_confirmed=True, _allow_frame_omission_recovery=False)

    assert not refused.usable
    assert refused.layout_repaired_shifts == ()
    assert refused.shifts[2].status == frameshift.SHIFT_NO_CONSENSUS


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

def _obs(frame_index, page_y0, page_y1, *, complete=True, hearts=(), kind=None, top_kind=None,
         top_observed=None, bottom_observed=None):
    """One hand-written sighting. `hearts` are page rows; x is fixed because a list scroll has
    no horizontal component.

    `top_kind` follows `top_observed` by default, because segment.py cannot report an observed
    top edge without saying which evidence produced it: an observed one is a gutter or the card's
    own corner, an unobserved one is background or a band edge. `EDGE_CARD_CORNER` is the default
    for an observed top since that is the only kind that can bound the FIRST item of a profile —
    `_scroll_top_evidence` reads it, and a fixture that left it blank would be modelling a
    sighting segment.py cannot emit. Pass it explicitly to model an interior card (a gutter top)
    or a card the analysed band sliced (background)."""
    if kind is None:
        kind = (segment.BLOCK_SELECTABLE if hearts else segment.BLOCK_CONTEXT) if complete \
            else segment.BLOCK_PARTIAL
    if top_observed is None:
        top_observed = complete
    if bottom_observed is None:
        bottom_observed = complete
    if top_kind is None:
        top_kind = segment.EDGE_CARD_CORNER if top_observed else segment.EDGE_BACKGROUND_RUN
    return item_index.BlockObservation(
        frame_index=frame_index, page_y0=page_y0, page_y1=page_y1,
        frame_y0=page_y0, frame_y1=page_y1, kind=kind, complete=complete,
        top_observed=top_observed, bottom_observed=bottom_observed,
        hearts=tuple((_HEART_CX, y) for y in hearts), top_kind=top_kind)


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
    for block, (_, _, y1) in zip(
            index.selectable, [r for r in _ROWS if r[0] == "card"], strict=True):
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
    assert all(a[1] < b[0] for a, b in pairwise(rows))


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


def test_animation_markers_of_the_wrong_length_are_refused():
    """`animation_markers` is frame-aligned evidence -- one value per frame, in the same order --
    and the docstring is explicit that a supplied sequence of the wrong length "is refused rather
    than padded or guessed". Two frames, one marker: the mismatch must stop the build rather than
    be zipped short or padded with an assumed value."""
    frames = [_frame(_FULL_SCROLL[0]), _frame(_FULL_SCROLL[1])]
    with pytest.raises(item_index.ItemIndexError, match="frame-aligned"):
        item_index.build_item_index(
            frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None,
            animation_markers=(False,))


def test_video_mute_markers_must_be_records_not_plain_booleans():
    """The v12 marker parameter takes positioned `VideoMuteMarker` records; the legacy per-frame
    boolean is `animation_markers`'s job, not this one's. A caller that hands `video_mute_markers`
    a plain bool or an ad hoc tuple standing in for a record is almost certainly holding the wrong
    sequence entirely, and treating every entry as truthy (which `any(...)` would do for a
    truthy-non-empty tuple row) would hide exactly that mistake."""
    frames = [_frame(_FULL_SCROLL[0]), _frame(_FULL_SCROLL[1])]
    for bad_markers in ((True, False), ((0, 106, 1150, 1.0),)):
        with pytest.raises(item_index.ItemIndexError, match="never booleans"):
            item_index.build_item_index(
                frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
                like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
                identity_band=None, video_mute_markers=bad_markers)


def test_video_mute_marker_frame_index_outside_the_capture_is_refused():
    """A marker naming a frame at or past the end of `frames` -- or before its start -- cannot be
    positioned evidence about THIS capture, and is refused rather than silently ignored by
    whatever loop would otherwise index past the frame list with it."""
    frames = [_frame(_FULL_SCROLL[0]), _frame(_FULL_SCROLL[1])]
    for bad_frame_index in (len(frames), -1):
        markers = (item_index.VideoMuteMarker(
            frame_index=bad_frame_index, x=106, y=1150, score=1.0),)
        with pytest.raises(item_index.ItemIndexError, match="outside this capture"):
            item_index.build_item_index(
                frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
                like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True,
                identity_band=None, video_mute_markers=markers)


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


def test_repeated_near_gutter_with_cross_frame_boundary_proof_splits_only_partial_merges():
    """A real 64px page-background run may overcount a gutter by a few pale card rows.

    This is the Mackenzie regression: one frame fully bounded the heartless context card at
    2224..2828, two others repeatedly saw a 64px page-coloured span at that exact bottom, and
    earlier frames saw the next fragment begin 53px below it.  The spanning sightings are not a
    changing profile -- they are a single-frame segmenter miss.  Keep their virtual pieces
    partial, so this repair cannot fabricate a model-selectable item by itself.
    """
    observations = [
        _obs(4, 2224, 2828),
        _obs(4, 2881, 2997, complete=False),
        _obs(5, 2224, 2828),
        _obs(5, 2881, 3229, complete=False),
        _obs(6, 2224, 3449, complete=False),
        _obs(7, 2224, 3674, complete=False),
        _obs(8, 2224, 3855, complete=False, hearts=(3766,)),
    ]
    repaired, notes = item_index._split_repeated_near_gutter_merges(
        observations, ((6, 2828, 2892), (7, 2828, 2892), (8, 2828, 2892)),
        tolerance=item_index._EXTENT_TOLERANCE_PX)
    blocks, failures = _assemble(repaired)

    assert failures == ()
    assert [(block.page_y0, block.page_y1, block.kind) for block in blocks] == [
        (2224, 2828, item_index.ITEM_CONTEXT),
        (2881, 3855, item_index.ITEM_PARTIAL),
    ]
    assert blocks[1].hearts == ((_HEART_CX, 3766),
                                ) and blocks[1].heart_ordinal == 1
    assert len(notes) == 1 and "frames [6, 7, 8]" in notes[0]

    # A lone 64px pale span is still just a segmenter ambiguity, never a licence to split.
    unchanged, no_notes = item_index._split_repeated_near_gutter_merges(
        observations, ((6, 2828, 2892),), tolerance=item_index._EXTENT_TOLERANCE_PX)
    assert unchanged == tuple(observations) and no_notes == ()


def test_long_background_card_top_merge_splits_the_reported_heartless_hybrid():
    """The refusal's exact f0/f3/f4 geometry is repairable without discarding f0.

    Frame 0's 1964..2100 partial contained a 76px ``RUN_TOO_LONG`` page-background interval at
    1970..2046.  Frames 3 and 4 independently bounded the lower card at 2046..3020.  The
    segmenter's conservative local rule correctly declined to call 76px a generic gutter, but
    the page fold can now prove that the lower side is this one complete card. Keep both content
    fragments partial, remove only the known background rows, and never assign an observed edge
    that no frame actually saw.

    The upper 6px remainder deliberately stays in the record.  Dropping it would hide evidence;
    the neighbouring complete cards plus f0's failure-free scan instead prove it cannot conceal a
    heart, so it is ordinal-safe but still uncroppable.  The lower partial then corroborates the
    complete card rather than overrunning it.
    """
    observations = [
        _obs(0, 694, 1803, hearts=(1714,)),
        _obs(0, 1964, 2100, complete=False),
        _obs(3, 2046, 3020, hearts=(2931,)),
        _obs(4, 2046, 3020, hearts=(2931,)),
    ]

    repaired, notes = item_index._split_long_background_card_top_merges(
        observations, ((0, 1970, 2046),), tolerance=item_index._EXTENT_TOLERANCE_PX)

    assert [(o.frame_index, o.page_y0, o.page_y1, o.complete, o.hearts) for o in repaired] == [
        (0, 694, 1803, True, ((_HEART_CX, 1714),)),
        (0, 1964, 1970, False, ()),
        (0, 2046, 2100, False, ()),
        (3, 2046, 3020, True, ((_HEART_CX, 2931),)),
        (4, 2046, 3020, True, ((_HEART_CX, 2931),)),
    ]
    assert len(notes) == 1
    assert "frame 0" in notes[0]
    assert "1964..2100" in notes[0] and "1970..2046" in notes[0]
    assert "2046..3020" in notes[0]

    blocks, failures, assembly_notes = _assemble(
        repaired, at_scroll_top=False, page_coverage=((1803, 2046),), full=True)

    assert failures == ()
    assert [(block.page_y0, block.page_y1, block.kind) for block in blocks] == [
        (694, 1803, item_index.ITEM_SELECTABLE),
        (1964, 1970, item_index.ITEM_PARTIAL),
        (2046, 3020, item_index.ITEM_SELECTABLE),
    ]
    # No new item was invented and no existing card was renumbered by the retained fringe.
    assert [block.heart_ordinal for block in blocks] == [1, None, 2]
    assert [block.model_index for block in blocks] == [1, None, 2]
    assert any("retained as uncroppable but is ordinal-safe" in note for note in assembly_notes)


@pytest.mark.parametrize(
    "case", ("heart", "no_complete_card", "single_complete_card", "misaligned_card",
             "disagreeing_complete_cards", "tiny_overlap"))
def test_long_background_card_top_merge_requires_heartless_exactly_aligned_complete_card(case):
    """This narrow repair must not turn an arbitrary long blank span into a boundary."""
    upper = _obs(0, 694, 1803, hearts=(1714,))
    hybrid = _obs(0, 1964, 2100, complete=False,
                  hearts=(1984,) if case == "heart" else ())
    if case == "no_complete_card":
        observations = [upper, hybrid]
    elif case == "single_complete_card":
        observations = [upper, hybrid, _obs(4, 2046, 3020, hearts=(2931,))]
    elif case == "misaligned_card":
        # More than chain slack away from the long run's lower edge; it could be another card.
        observations = [upper, hybrid, _obs(3, 2070, 3044, hearts=(2955,)),
                        _obs(4, 2070, 3044, hearts=(2955,))]
    elif case == "disagreeing_complete_cards":
        observations = [upper, hybrid, _obs(3, 2046, 3020, hearts=(2931,)),
                        _obs(4, 2046, 3100, hearts=(3011,))]
    elif case == "tiny_overlap":
        hybrid = _obs(0, 1964, 2050, complete=False)
        observations = [upper, hybrid, _obs(3, 2046, 3020, hearts=(2931,)),
                        _obs(4, 2046, 3020, hearts=(2931,))]
    else:
        observations = [upper, hybrid, _obs(3, 2046, 3020, hearts=(2931,)),
                        _obs(4, 2046, 3020, hearts=(2931,))]

    repaired, notes = item_index._split_long_background_card_top_merges(
        observations, ((0, 1970, 2046),), tolerance=item_index._EXTENT_TOLERANCE_PX)

    assert repaired == tuple(observations)
    assert notes == ()


@pytest.mark.parametrize("splitter", ("long_background_run", "repeated_near_gutter"))
def test_a_virtual_split_never_carries_the_original_top_edge_onto_the_lower_fragment(splitter):
    """A split moves `page_y0` to a row no frame bounded, so it must claim no top evidence.

    Both virtual splits build their fragments by copying the ORIGINAL sighting's fields.
    `top_kind` describes the row at `page_y0` and only that row, so the fragment whose
    `page_y0` became the invented seam must carry none — while the upper fragment, whose
    `page_y0` is unmoved, keeps whatever segment.py really saw there.
    """
    hybrid = _obs(0, 1964, 2100, complete=False,
                  top_observed=True, top_kind=segment.EDGE_CARD_CORNER)
    if splitter == "long_background_run":
        observations = [hybrid,
                        *(_obs(frame, 2046, 3020, hearts=(2931,), top_kind=segment.EDGE_GUTTER)
                          for frame in (3, 4))]
        repaired, notes = item_index._split_long_background_card_top_merges(
            observations, ((0, 1970, 2046),), tolerance=item_index._EXTENT_TOLERANCE_PX)
        expected = [(1964, 1970), (2046, 2100)]
    else:
        # The near-gutter splitter's own licence: a repeated 59..64px run at a complete card's
        # end, with the next fragment beginning a canonical 53px gutter below it.
        observations = [_obs(5, 1900, 1970, top_kind=segment.EDGE_GUTTER), hybrid,
                        *(_obs(frame, 2023, 3020, hearts=(2931,), top_kind=segment.EDGE_GUTTER)
                          for frame in (3, 4))]
        repaired, notes = item_index._split_repeated_near_gutter_merges(
            observations, ((6, 1970, 2032), (7, 1970, 2032)),
            tolerance=item_index._EXTENT_TOLERANCE_PX)
        expected = [(1964, 1970), (2023, 2100)]
    assert len(notes) == 1, notes

    fragments = [o for o in repaired if o.frame_index == 0]
    assert [(o.page_y0, o.page_y1) for o in fragments] == expected
    assert fragments[0].top_kind == segment.EDGE_CARD_CORNER   # its page_y0 is unmoved
    assert fragments[1].top_kind == ""                         # its page_y0 is invented
    assert fragments[1].top_observed is False


def test_a_split_lower_fragment_cannot_donate_a_card_corner_to_the_scroll_top_test():
    """The leak the split's `top_kind` copy would open, priced through `_assemble`.

    `_scroll_top_evidence` reads `top_kind` WITHOUT consulting `top_observed` — deliberately,
    because the screen-fixed island disposition pairs a REAL edge kind with an unobserved top.
    So a lower fragment that inherited `EDGE_CARD_CORNER` from a top row 82px above it is read
    as "some frame saw the first item's OWN top edge", and the 2026-08-28 guard passes on
    evidence no frame produced.

    Here the heartless partial's real top is the band-clipped card corner of the scrolling
    section heading, and the card below the seam was bounded by GUTTERS in both frames that
    saw it whole — the Hinge 10.1.0 pinned-header shape, where a false `at_scroll_top` shifts
    every heart ordinal while every completeness property still agrees with it.
    """
    repaired, notes = item_index._split_long_background_card_top_merges(
        [_obs(0, 1964, 2100, complete=False,
              top_observed=True, top_kind=segment.EDGE_CARD_CORNER),
         *(_obs(frame, 2046, 3020, hearts=(2931,), top_kind=segment.EDGE_GUTTER)
           for frame in (3, 4))],
        ((0, 1970, 2046),), tolerance=item_index._EXTENT_TOLERANCE_PX)
    assert len(notes) == 1

    blocks, failures = _assemble(repaired, at_scroll_top=True,
                                 page_coverage=((1803, 2046),))

    # The 6px heading remnant is the leading heartless partial the guard skips, so the block it
    # is talking about is the card below the seam.
    assert [(block.page_y0, block.page_y1) for block in blocks] == [(1964, 1970), (2046, 3020)]
    assert any("never a card corner" in failure for failure in failures), failures
    assert any("['gutter']" in failure for failure in failures), failures


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
    # `at_scroll_top=False` because this fixture does not model a scroll top and never did: both
    # sightings are unbounded fragments, so no frame ever saw the first item's own top edge, and
    # `_scroll_top_evidence` rightly refuses to corroborate a top claim from that. Asserting one
    # here would be testing the min-gap guard through an unrelated failure.
    gap = 20
    blocks, failures = _assemble([_obs(0, 700, 1200, complete=False, hearts=(1150,)),
                                  _obs(1, 1200 + gap, 1700, complete=False, hearts=(1650,))],
                                 at_scroll_top=False)

    assert any(f"only {gap}px apart" in f for f in failures), failures
    assert any("would fabricate an item" in f for f in failures)

    # The control: push them a canonical gutter apart and they ARE two items, silently.
    _, ok = _assemble([_obs(0, 700, 1200, complete=False, hearts=(1150,)),
                       _obs(1, 1200 + _GUTTER, 1700, complete=False, hearts=(1650,))],
                      at_scroll_top=False)
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


def test_a_fully_scanned_partial_between_complete_cards_cannot_hide_an_ordinal():
    """The 2026-08-30 live refusal, in its measured page geometry.

    Hinge inserted the 37px ``From people close to Katie`` heading between a complete photo and
    a complete endorsement card. Its text pixels were stable at page rows 1800..1837 across four
    scroll positions, but neither edge was a canonical gutter/card corner, so the block correctly
    remained PARTIAL. Frame 0 nevertheless analysed page rows 300..2100 in one piece: that covers
    the entire 1671..1914 interval between the independently bounded neighbours, and heart
    matching searched that whole band before block assignment. There are therefore no unseen rows
    in which this partial can hide an extra heart ordinal.

    Coverage resolves only that counting question. The heading stays partial, uncroppable and out
    of both model/context lists; the endorsement card below keeps heart/model ordinal 2.
    """
    observations = [
        _obs(0, 697, 1671, hearts=(1582,)),
        *[_obs(frame, 1800, 1837, complete=False) for frame in range(4)],
        _obs(2, 1914, 2888, hearts=(2799,)),
    ]

    blocks, failures, notes = _assemble(
        observations, page_coverage=((300, 2100),), full=True)

    assert failures == ()
    heading = blocks[1]
    assert heading.kind == item_index.ITEM_PARTIAL
    assert not heading.complete and heading.croppable == ()
    assert heading.heart_ordinal is None and heading.model_index is None
    assert heading not in tuple(block for block in blocks if block.kind == item_index.ITEM_CONTEXT)
    assert [block.heart_ordinal for block in blocks] == [1, None, 2]
    assert [block.model_index for block in blocks] == [1, None, 2]
    assert any("retained as uncroppable but is ordinal-safe" in note for note in notes)


def test_fully_scanned_partial_with_its_own_observed_edges_cannot_hide_an_ordinal():
    """The 2026-09-01 Madeleine refusal, reduced to its measured geometry.

    The 1279px heartless block at 2282..3561 was never complete in a single frame, but a
    canonical gutter observed its top in frames 2/3 and another observed its bottom in frame 4.
    Frame 3's clean band 1812..3612 then heart-scanned every row of the resolved extent. The
    lower complete card starts at 3614, two rows past that band, so the older whole-neighbour-
    interval exception correctly cannot apply; this separate proof is enough for ordinal safety
    only while the same immediate complete-neighbour bracket remains in force.
    """
    observations = [
        _obs(1, 1544, 2229, hearts=(2140,)),
        _obs(2, 2282, 3093, complete=False, top_observed=True),
        _obs(3, 2282, 3561, complete=False, top_observed=True),
        _obs(4, 2283, 3561, complete=False, bottom_observed=True),
        _obs(6, 3614, 4588, hearts=(4499,)),
    ]

    blocks, failures, notes = _assemble(
        observations, at_scroll_top=False, page_coverage=((1812, 3612),), full=True)

    assert failures == ()
    partial = blocks[1]
    assert (partial.page_y0, partial.page_y1) == (2282, 3561)
    assert partial.kind == item_index.ITEM_PARTIAL
    assert not partial.complete and partial.croppable == ()
    assert [block.heart_ordinal for block in blocks] == [1, None, 2]
    assert [block.model_index for block in blocks] == [1, None, 2]
    assert any("its own observed edges" in note for note in notes)


def test_own_edge_ordinal_safety_requires_complete_neighbours_gutters_edges_and_one_interval():
    """The Madeleine exception needs its bracket/gutters, does not infer edges or stitch scans."""
    bounded_above_and_below = [
        _obs(1, 1544, 2229, hearts=(2140,)),
        _obs(2, 2282, 3093, complete=False, top_observed=True),
        _obs(3, 2282, 3561, complete=False, top_observed=True),
        _obs(4, 2283, 3561, complete=False, bottom_observed=True),
        _obs(6, 3614, 4588, hearts=(4499,)),
    ]

    missing_bottom = bounded_above_and_below[:-2] + bounded_above_and_below[-1:]
    _blocks, missing_edge_failures = _assemble(
        missing_bottom, at_scroll_top=False, page_coverage=((1812, 3612),))
    assert any("a heart may sit in the rows that were never inside" in failure
               for failure in missing_edge_failures)

    incomplete_lower = bounded_above_and_below[:-1] + [
        _obs(6, 3614, 4588, complete=False, hearts=(4499,)),
    ]
    _blocks, incomplete_neighbour_failures = _assemble(
        incomplete_lower, at_scroll_top=False, page_coverage=((1812, 3612),))
    assert any("a heart may sit in the rows that were never inside" in failure
               for failure in incomplete_neighbour_failures)

    large_upper_gap = [_obs(1, 1544, 2100, hearts=(2010,))] + bounded_above_and_below[1:]
    _blocks, large_gap_failures = _assemble(
        large_upper_gap, at_scroll_top=False, page_coverage=((1812, 3612),))
    assert any("a heart may sit in the rows that were never inside" in failure
               for failure in large_gap_failures)

    for coverage in (((1812, 3560),), ((1812, 3000), (3000, 3612))):
        _blocks, coverage_failures = _assemble(
            bounded_above_and_below, at_scroll_top=False, page_coverage=coverage)
        assert any("a heart may sit in the rows that were never inside" in failure
                   for failure in coverage_failures)


def test_partial_ordinal_safety_needs_one_complete_clean_coverage_interval():
    """Do not assemble a proof from two frames or from a partial neighbouring card.

    A single failure-free frame must have searched every possible row between both complete
    physical bounds. Otherwise the original hidden-heart refusal remains byte-for-byte active.
    """
    heading = [_obs(0, 1800, 1837, complete=False)]
    complete_neighbours = [
        _obs(0, 697, 1671, hearts=(1582,)), *heading,
        _obs(2, 1914, 2888, hearts=(2799,)),
    ]

    _blocks, split_coverage_failures = _assemble(
        complete_neighbours, page_coverage=((300, 1850), (1850, 2100)))
    assert any("a heart may sit in the rows that were never inside" in failure
               for failure in split_coverage_failures)

    incomplete_above = [
        _obs(0, 697, 1671, complete=False, hearts=(1582,)), *heading,
        _obs(2, 1914, 2888, hearts=(2799,)),
    ]
    _blocks, neighbour_failures = _assemble(
        incomplete_above, at_scroll_top=False, page_coverage=((300, 2100),))
    assert any("a heart may sit in the rows that were never inside" in failure
               for failure in neighbour_failures)


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

    # Extended 2026-08-28, when a SECOND test was added alongside this one (the first item's own
    # card corner — see `test_background_above_without_the_first_items_corner_is_no_longer_accepted`
    # for why the background test had gone vacuous). Two tests mean two possible messages, and
    # each has to name the thing an operator should go and look at. This capture's topmost block
    # really is flush on the band's first row, so it must get the band-edge wording and NOT the
    # corner one — a message naming the wrong test sends the operator to the wrong row.
    contradiction = next(f for f in failures if "at_scroll_top was asserted" in f)
    assert "no frame ever saw the first item's OWN top edge" not in contradiction
    assert sum("at_scroll_top was asserted" in f for f in failures) == 1

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


# =====================================================================================
# PITCH-RELATIVE LANDMARK BOUND (2026-08-24) — `_pitch_relative_max_step`, replacing the fixed
# `_MAX_STEP_PX` scalar as `_structural_landmarks`'s cross-card defence now that
# `scroll_step.plan_coverage_step` routinely asks for steps well past 363px. See that function's
# own module comment for the full derivation; these tests pin the two numbers it produces on
# this file's own WORLD and prove the defence still discriminates real single-card shifts from
# implausible cross-card ones at the new, larger cadence.
# =====================================================================================

def _blank_segmentation():
    """A band with no card drawn at all: zero blocks, `ok` True, nothing measurable — the same
    "no local evidence" case `plan_coverage_step`'s own blind fallback exists for."""
    col = _page_column(_H)
    gray = np.repeat(col[:, None], _W, axis=1)
    gray[:_BAND0] = _CHROME_TOP
    gray[_BAND1:] = _CHROME_BOTTOM
    png = cv2.imencode(".png", gray)[1].tobytes()
    return item_index.segment_frame(
        png, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)


def test_pitch_relative_bound_uses_the_profiles_own_minimum_measured_spacing():
    """Over this file's own WORLD, the smallest measured local spacing anywhere is 812px (card 2,
    760px tall plus the 52px gutter floor) — smaller than every other card's own pitch — and the
    bound is that minimum less `_MIN_ITEM_GAP_PX` (47px), not a fixed scalar and not the smallest
    CARD's own extent alone (`measure_local_spacing` already takes the minimum across three
    measurement kinds; this function's job is only to take the minimum ACROSS FRAMES on top of
    that, then subtract the safety margin once)."""
    scrolls = (0, 900, 1800, 2700, 3000)
    segmentations = [item_index.segment_frame(
        _frame(s), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for s in scrolls]
    assert item_index._pitch_relative_max_step(segmentations) == 812 - item_index._MIN_ITEM_GAP_PX
    assert item_index._pitch_relative_max_step(segmentations) == 765
    # And it is comfortably above the OLD fixed 363px ceiling: this is the whole point of the
    # replacement — a bound this profile's own geometry can afford, not one every profile shares.
    assert item_index._pitch_relative_max_step(segmentations) > item_index._MAX_STEP_PX


def test_pitch_relative_bound_falls_back_to_the_corpus_minimum_when_nothing_measures():
    """No frame in the read offers any of the three local-spacing measurements (a blank band):
    the bound falls back to `_FALLBACK_SPACING_PX` — the smallest pitch ever measured across this
    repo's whole calibration corpus — minus the same margin, exactly `plan_scroll_step`'s own
    blind-fallback philosophy ("no evidence, be the most conservative thing the corpus has ever
    justified") applied one layer up."""
    blank = _blank_segmentation()
    assert blank.ok and not blank.blocks
    bound = item_index._pitch_relative_max_step([blank, blank])
    assert bound == item_index._FALLBACK_SPACING_PX - item_index._MIN_ITEM_GAP_PX
    assert bound == 691


def test_pitch_relative_bound_floors_at_one_rather_than_going_non_positive():
    """A margin larger than the measured pitch must not produce a zero or negative bound, which
    would make every repair function's own `0 < candidate <= bound` check vacuous instead of
    reporting a real, positive (if useless) ceiling."""
    scrolls = (0, 900, 1800)
    segmentations = [item_index.segment_frame(
        _frame(s), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD) for s in scrolls]
    assert item_index._pitch_relative_max_step(segmentations, margin_px=100_000) == 1


def test_enum_step_ceiling_is_the_coverage_rules_own_flat_bound_not_the_pitch():
    """`_enum_step_ceiling` answers a different question from `_pitch_relative_max_step` (how far
    could N ordinary GESTURES have moved the content, not how close two landmarks may be) and
    must not accidentally collapse onto it. On this file's calibrated 1800-row band it is
    `scroll_step._ENUM_TRUST_CEILING_BAND_FRAC` (0.30) of the band height — 540px — regardless of
    what any particular profile's card pitch happens to be."""
    segmentations = [item_index.segment_frame(
        _frame(0), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)]
    assert item_index._enum_step_ceiling(segmentations) == 540
    assert item_index._enum_step_ceiling(segmentations) != item_index._pitch_relative_max_step(
        segmentations)


def test_a_candidate_between_the_old_and_new_bound_is_refused_at_363_and_accepted_past_it(
        monkeypatch):
    """The TRAP, demonstrated directly: a +500px candidate — physically impossible for the OLD
    363px ceiling to ever consider, but well inside this profile's own 765px pitch-relative
    bound — must be refused when `max_step_px` is still the old fixed scalar and accepted once
    it is threaded through as the new, profile-derived one. This is `_layout_repaired_shift`
    itself, not a synthetic stand-in, and the landmarks are the real segmented geometry (this
    fixture's cards are far enough apart that no monkeypatch is needed to construct a genuine
    +500px top/bottom/heart landmark triple)."""
    before = item_index.segment_frame(
        _frame(0), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    after = item_index.segment_frame(
        _frame(500), content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD)
    base = item_index.estimate_shift(_frame(0), _frame(500), content_band=_CONTENT_BAND)
    raw = _shift_with_votes(base, [500, 500], status=frameshift.SHIFT_NO_CONSENSUS)

    refused, note = item_index._layout_repaired_shift(
        0, before, after, raw, max_step_px=item_index._MAX_STEP_PX)
    assert refused is raw and note is None

    pitch_bound = item_index._pitch_relative_max_step([before, after])
    assert item_index._MAX_STEP_PX < 500 <= pitch_bound
    repaired, note = item_index._layout_repaired_shift(
        0, before, after, raw, max_step_px=pitch_bound)
    assert repaired.delta_px == 500 and note is not None


def test_build_item_index_computes_the_pitch_bound_itself_not_the_default(monkeypatch):
    """End to end: `build_item_index` must thread its OWN computed `_pitch_relative_max_step`
    into the repair machinery rather than leaving every function at its bare-test default. Proven
    by capturing what `max_step_px` each call actually receives."""
    seen: list[int] = []
    real = item_index._structural_landmarks

    def spy(before, after, *, max_step_px=item_index._MAX_STEP_PX):
        seen.append(max_step_px)
        return real(before, after, max_step_px=max_step_px)

    monkeypatch.setattr(item_index, "_structural_landmarks", spy)
    frames = [_frame(0), _frame(_STEP), _frame(_STEP * 2)]
    item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert seen, "the repair path must consult the landmark bound at least once"
    assert all(value == seen[0] for value in seen), "one profile, one bound, every call"
    assert seen[0] != item_index._MAX_STEP_PX
    assert seen[0] == item_index._pitch_relative_max_step(
        tuple(item_index.segment_frame(
            frame, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
            like_threshold=hinge._LIKE_MATCH_THRESHOLD) for frame in frames))


# =====================================================================================
# COVERAGE COST AT A COARSE CADENCE: a card genuinely never observed complete is REFUSED as a
# selectable choice — demoted to ITEM_PARTIAL — never mis-numbered. This is the existing safety
# net (`IndexedBlock.kind`/`heart_ordinal`/`model_index`, unchanged by this pass) exercised at
# the new, coarser cadence `scroll_step.plan_coverage_step` can now produce, rather than a new
# mechanism: doc 5.10's fabricated-tenth-item bug is a NAIVE heart-tracker failure this
# architecture (folding by absolute page position, never by chaining) was already immune to — see
# item_index.py's own module docstring and scroll_step.py's "WHAT VIOLATING THE RATIO RULE
# ACTUALLY COSTS" section for the stride-2/stride-3 measurements this test's geometry mirrors.
# =====================================================================================

def test_a_card_never_observed_complete_at_a_coarse_cadence_is_partial_not_misnumbered():
    """Five frames at 700/600/850/750/100px steps — every one comfortably under frameshift's
    900px trust window, so the whole chain measures cleanly — are phased so that card 3
    (2434..3434, 1000px tall) is NEVER shown whole: frame 2 (scroll 1300, band 1600..3400) sees
    its top but not its bottom; frame 3 (scroll 2150, band 2450..4250) sees its bottom but not
    its top, because the band's own top row (2450) has already scrolled past card 3's own top
    row (2434). No frame in between could have shown it either, since none exists.

    The index still builds (`usable`), every OTHER card keeps its correct heart ordinal and a
    DENSE model index over exactly the selectable ones (card 3 consumes ordinal 3 but no model
    slot, and card 4 — heart ordinal 4 — becomes model index 3, not 4), and card 3 itself is
    reported, not silently dropped: `ITEM_PARTIAL`, ordinal 3, `model_index is None`.
    """
    scrolls = (0, 700, 1300, 2150, 2900, 3000)
    steps = [b - a for a, b in zip(scrolls, scrolls[1:], strict=False)]
    assert all(step <= 900 for step in steps), "every pair must stay inside the trust window"
    frames = [_frame(s) for s in scrolls]

    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=True, identity_band=None)

    assert index.usable, index.failures
    assert [shift.delta_px for shift in index.shifts] == steps

    by_row = {(block.page_y0, block.page_y1): block for block in index.blocks}
    card3 = by_row[(_CARD3[1], _CARD3[2])]
    assert card3.kind == item_index.ITEM_PARTIAL
    assert card3.heart_ordinal == 3 and card3.model_index is None
    assert all(not observation.complete for observation in card3.observations), (
        "the whole point of this fixture: card 3 must never be observed complete anywhere")

    card4 = by_row[(_CARD4[1], _CARD4[2])]
    assert card4.kind == item_index.ITEM_SELECTABLE
    assert card4.heart_ordinal == 4 and card4.model_index == 3, (
        "the model index must stay DENSE over selectable items and skip card 3's consumed slot")

    card1 = by_row[(_CARD1[1], _CARD1[2])]
    card2 = by_row[(_CARD2[1], _CARD2[2])]
    assert (card1.heart_ordinal, card1.model_index) == (1, 1)
    assert (card2.heart_ordinal, card2.model_index) == (2, 2)
    # Every heart ordinal 1..4 is accounted for exactly once — the never-substitute invariant
    # this whole architecture exists to protect does not depend on every item being selectable.
    assert sorted(block.heart_ordinal for block in index.blocks if block.heart_ordinal) == [
        1, 2, 3, 4]


# =====================================================================================
# HINGE 10.1.0'S SCREEN-PINNED PROFILE HEADER (the 2026-08-28 incident)
#
# WHAT HAPPENED. 10.1.0 pins a per-profile header — filter chips, the name, a verified badge, a
# back arrow, an overflow menu, a pronoun row — to the SCREEN, inside the analysed content band.
# It does not scroll, so its page position is `frame_row + offset`: a different, FABRICATED page
# row on every frame of the capture. segment.py merged that 43px strip into the clipped top of the
# card below (the 106px gap between them is not gutter-length, so the ordinary rule absorbed it),
# producing a block top 149px too high, which in page space landed inside the card ABOVE and
# bridged two cards into one fold group. `_split_on_bounded_cards` could not rescue it — only ONE
# of the two cards had a complete sighting — and `_resolve_group` hard-refused the whole index.
#
# THE FIX, in two halves. segment.py now reports such a strip as `BLOCK_UNANCHORED` with both
# edges UNOBSERVED and a `content_digest`, saying only "one frame cannot place this". This module
# decides, from the capture's own frames, whether it is pinned to the screen or is page content —
# and holds a PROVEN one out of page space entirely rather than asserting a row for it.
#
# Everything below is measured on run dir `data/hinge_debug/8fb11094ef4d/`, frames 13..18 at page
# offsets 6180, 6615, 7110, 7637, 8152, 8485, whose strip is frame rows 368..411 with one
# identical digest on all six.
# =====================================================================================

def _island(frame_index, offset, *, rows=(_PINNED_Y0, _PINNED_Y1), digest="d0"):
    """One hand-built `UnanchoredIsland`: where on the SCREEN, how far down the PAGE, what pixels.
    Those three fields are the entire input to `_screen_fixed_islands`, by design."""
    return item_index.UnanchoredIsland(
        frame_index=frame_index, frame_y0=rows[0], frame_y1=rows[1], offset=offset, digest=digest)


def _note_for(notes, rows=(_PINNED_Y0, _PINNED_Y1)):
    """The single note about the strip at `rows`. Asserting there is exactly one is part of the
    contract: `_screen_fixed_islands` reports one verdict per distinct frame extent."""
    hits = [n for n in notes if f"frame rows {rows[0]}..{rows[1]}" in n]
    assert len(hits) == 1, notes
    return hits[0]


def test_identical_pixels_at_four_far_apart_offsets_prove_a_strip_is_screen_fixed():
    """THE PROOF, in the shape the incident capture actually supplied it.

    A screen-fixed element keeps a constant FRAME position while its page position moves with
    every scroll; page content does the exact opposite. So identical pixels at the same frame rows
    across a scroll span WIDER than the strip's own height cannot be page content — the page rows
    it would have had to display at the two ends are disjoint.

    [measured: rows 368..411 (43px), byte-identical across the six evidence frames, at offsets
    6180..8485 = 2305px of scroll.] Four of those offsets are used here; the span is 53x the
    strip's own height, so this is nowhere near the bound and is meant not to be.

    If this stopped returning the strip as proven, `_observations` would place it at
    `frame_row + offset` on every frame and the 2026-08-28 refusal returns."""
    proven, notes = item_index._screen_fixed_islands([
        _island(13, 6180), _island(14, 6615), _island(16, 7637), _island(18, 8485)])

    assert proven == frozenset({(_PINNED_Y0, _PINNED_Y1)})
    note = _note_for(notes)
    assert "showed identical pixels at 4 page offsets spanning 2305px" in note
    assert "more than its own 43px height" in note
    assert "fixed to the SCREEN, not to the page" in note
    assert "held out of the index entirely" in note


def test_one_page_offset_can_never_prove_anything_about_a_strip():
    """A repeat at ONE offset proves nothing: the page did not move between the two frames, so
    "screen-pinned chrome" and "page content" predict identical pixels. Two frames of a capture
    that did not scroll are two frames, not two measurements.

    The note has to say THAT specific reason. This repo's standing rule is that guidance derives
    from its precondition (a hardcoded "next step" once named an already-satisfied prerequisite and
    hid the real fix across five surfaces for a day), and an operator reading a refusal needs to
    know whether to scroll further or to stop trusting the strip."""
    proven, notes = item_index._screen_fixed_islands([_island(13, 6180), _island(14, 6180)])

    assert proven == frozenset()
    note = _note_for(notes)
    assert "was only ever seen at one page offset (6180)" in note
    assert "nothing distinguishes screen-pinned chrome from page content here" in note
    assert "placed on the page exactly as it would have been before this check existed" in note
    # ...and specifically NOT either of the other two reasons.
    assert "less than its own" not in note and "different pixel contents" not in note


def test_a_scroll_shorter_than_the_strips_own_height_is_not_a_proof():
    """The bound is `span >= height`, and it is geometric rather than chosen. Below it the two
    sightings' PAGE extents still overlap, so one tall band of page content sitting still-ish
    behind a 10px scroll could produce both. At or above it the page rows are disjoint and only a
    screen-fixed element can show the same pixels twice.

    43px strip, 10px of scroll: `_island` at offsets 7110 and 7120. Not proven, and the note names
    the height reason rather than the offset-count one."""
    proven, notes = item_index._screen_fixed_islands([_island(13, 7110), _island(14, 7120)])

    assert proven == frozenset()
    note = _note_for(notes)
    assert "was seen across only 10px of scroll, less than its own 43px height" in note
    assert "one piece of page content could explain both sightings" in note
    assert "only ever seen at one page offset" not in note


def test_a_strip_whose_pixels_changed_is_not_a_static_element():
    """Unanimity on the digest, no tolerance and no quorum: this is an identity test on pixels, and
    "mostly the same" is what a scrolling page looks like. A header that animates or live-updates
    (a countdown, a carousel dot) reads exactly like this and must not be held out — holding out
    something that is really page content would drop whatever it holds."""
    proven, notes = item_index._screen_fixed_islands([
        _island(13, 6180, digest="d0"), _island(14, 6615, digest="d0"),
        _island(16, 7637, digest="CHANGED")])

    assert proven == frozenset()
    note = _note_for(notes)
    assert "showed 2 different pixel contents across 3 offsets" in note
    assert "an animating or live-updating header reads exactly like this" in note
    assert "only ever seen at one page offset" not in note and "less than its own" not in note


def test_strips_at_different_frame_rows_are_different_strips():
    """Grouping is by EXACT frame extent. A strip that changes height between frames is a different
    strip, and saying so is more honest than clustering two shapes together and calling them one —
    a taller sighting could be the header plus a slice of the card behind it.

    Here neither extent ever reaches two offsets, so neither is proven and each gets its own
    one-offset note. The two notes must be about the two different extents, not one about both."""
    proven, notes = item_index._screen_fixed_islands([
        _island(13, 6180, rows=(368, 411)), _island(14, 6615, rows=(368, 430))])

    assert proven == frozenset()
    assert len(notes) == 2
    assert "one page offset (6180)" in _note_for(notes, (368, 411))
    assert "one page offset (6615)" in _note_for(notes, (368, 430))


def test_a_missing_digest_is_never_proof():
    """`content_digest` is None on every block segment.py did NOT label `BLOCK_UNANCHORED`, so a
    None here means the evidence this test rests on was not collected. Absence of pixels is not
    sameness of pixels: with no digest there is nothing to compare, and the strip is placed on the
    page exactly as it was before the check existed."""
    proven, notes = item_index._screen_fixed_islands([
        _island(13, 6180, digest=None), _island(14, 8485, digest=None)])

    assert proven == frozenset()
    note = _note_for(notes)
    assert "at least one sighting carries no pixel digest, so there is nothing to compare" in note
    # It fails CLOSED on its own reason rather than borrowing the changed-pixels one: two
    # un-hashed sightings are not a match, and they are not a mismatch either.
    assert "different pixel contents" not in note
    assert "only ever seen at one page offset" not in note and "less than its own" not in note


# =====================================================================================
# "Unproven means status quo" — checkable rather than claimed
# =====================================================================================

def _edge(y, kind, *, observed, run_px=None):
    return segment.BlockEdge(y=y, observed=observed, kind=kind, run_px=run_px)


def _seg_block(y0, y1, *, kind, top, bottom, hearts=(), digest=None):
    return segment.Block(y0=y0, y1=y1, x0=_CARD_X0, x1=_CARD_X1, top=top, bottom=bottom,
                         kind=kind, hearts=hearts, reason="synthetic", content_digest=digest)


def _segmentation(blocks, *, digest="a" * 64):
    return segment.FrameSegmentation(
        frame_size=(_W, _H), frame_digest=digest, band=(_BAND0, _BAND1),
        card_x=(_CARD_X0, _CARD_X1), blocks=tuple(blocks), runs=(), hearts=(),
        unassigned_hearts=(), failures=())


def _pinned_strip_block(digest="d0"):
    """segment.py's own output shape for the pinned header: `BLOCK_UNANCHORED`, page background
    above it (`EDGE_BACKGROUND_RUN`, UNOBSERVED — one frame cannot say the strip ended, only that
    something stopped) and the island cut below it."""
    return _seg_block(
        _PINNED_Y0, _PINNED_Y1, kind=segment.BLOCK_UNANCHORED,
        top=_edge(_PINNED_Y0, segment.EDGE_BACKGROUND_RUN, observed=False, run_px=68),
        bottom=_edge(_PINNED_Y1, segment.EDGE_UNANCHORED_ISLAND, observed=False,
                     run_px=_PINNED_CLEARANCE_PX),
        digest=digest)


def test_an_unproven_strip_merged_below_reproduces_the_pre_fix_record_field_for_field():
    """THE BIT-EXACT FALLBACK. `_observations`' disposition 2 is the promise that an unavailable
    proof — a one-frame capture, a header that animates, a per-frame predicate that fired on
    something that was really page content — degrades to the behaviour that SHIPPED, never to a
    third new outcome. A promise like that is worth nothing unless it is compared against the old
    record field by field, so that is what this does.

    Before the strip had a name, segment.py never split it out at all: it returned ONE block from
    the strip's first row to the card's last, whose top edge was whatever sat above the strip
    (`EDGE_BACKGROUND_RUN`, unobserved) and whose bottom, kind and hearts were the card's own. The
    expected record here is written out literally rather than derived, because deriving it from
    the code under test is how a fallback quietly stops being bit-exact.

    A false positive in the screen-fixed proof costs nothing; a false negative costs the old bug,
    loudly — and that asymmetry only holds while this is true."""
    card = _seg_block(
        _PINNED_CONTENT_Y0, 1300, kind=segment.BLOCK_PARTIAL,
        top=_edge(_PINNED_CONTENT_Y0, segment.EDGE_UNANCHORED_ISLAND, observed=False),
        bottom=_edge(1300, segment.EDGE_CARD_CORNER, observed=True),
        hearts=((_HEART_CX, 1210),))
    seg = _segmentation([_pinned_strip_block(), card])

    observations = item_index._observations([seg], [6870], frozenset())

    assert observations == [item_index.BlockObservation(
        frame_index=0,
        page_y0=_PINNED_Y0 + 6870, page_y1=1300 + 6870,
        frame_y0=_PINNED_Y0, frame_y1=1300,
        kind=segment.BLOCK_PARTIAL, complete=False,
        top_observed=False, bottom_observed=True,
        hearts=((_HEART_CX, 1210 + 6870),),
        top_kind=segment.EDGE_BACKGROUND_RUN)]


def test_an_unproven_strip_split_off_by_a_card_corner_is_emitted_on_its_own():
    """DISPOSITION 3, the other bit-exact half. On the frames where the card below the strip is
    bounded by its OWN corner, segment.py's corner rescue already split the strip out before this
    incident existed — and those frames are precisely the ones that place a fabricated page
    position for it today. Unproven, the strip therefore stays an ordinary observation at
    `frame_row + offset`, exactly as it used to be, and the card keeps its own complete extent.

    The contrast with the merged case above is the point: which disposition applies is read off the
    NEXT block's own top edge kind, never guessed."""
    card = _seg_block(
        _PINNED_CONTENT_Y0, 1300, kind=segment.BLOCK_SELECTABLE,
        top=_edge(_PINNED_CONTENT_Y0, segment.EDGE_CARD_CORNER, observed=True),
        bottom=_edge(1300, segment.EDGE_CARD_CORNER, observed=True),
        hearts=((_HEART_CX, 1210),))
    seg = _segmentation([_pinned_strip_block(), card])

    strip, item = item_index._observations([seg], [6870], frozenset())

    assert strip == item_index.BlockObservation(
        frame_index=0,
        page_y0=_PINNED_Y0 + 6870, page_y1=_PINNED_Y1 + 6870,
        frame_y0=_PINNED_Y0, frame_y1=_PINNED_Y1,
        kind=segment.BLOCK_UNANCHORED, complete=False,
        top_observed=False, bottom_observed=False, hearts=(),
        top_kind=segment.EDGE_BACKGROUND_RUN)
    assert (item.page_y0, item.page_y1, item.complete) == (
        _PINNED_CONTENT_Y0 + 6870, 1300 + 6870, True)


def test_a_proven_screen_fixed_strip_becomes_no_observation_at_all():
    """DISPOSITION 1, and the whole fix in one line: a proven strip is not placed anywhere. It has
    no page position, so the only safe thing to do is never assert one. Because it never becomes an
    observation it cannot chain into a card's fold group, cannot reach past a bounded card, and
    cannot be a neighbour in the fabricated-item guard — see the objection test below, which drives
    all three of those failures out of the very same strip when the proof is withheld."""
    card = _seg_block(
        _PINNED_CONTENT_Y0, 1300, kind=segment.BLOCK_PARTIAL,
        top=_edge(_PINNED_CONTENT_Y0, segment.EDGE_UNANCHORED_ISLAND, observed=False),
        bottom=_edge(1300, segment.EDGE_CARD_CORNER, observed=True),
        hearts=((_HEART_CX, 1210),))
    seg = _segmentation([_pinned_strip_block(), card])

    observations = item_index._observations(
        [seg], [6870], frozenset({(_PINNED_Y0, _PINNED_Y1)}))

    assert len(observations) == 1
    only = observations[0]
    assert (only.page_y0, only.page_y1) == (_PINNED_CONTENT_Y0 + 6870, 1300 + 6870)
    assert only.frame_y0 == _PINNED_CONTENT_Y0, "the card must not inherit the strip's rows"


# =====================================================================================
# THE OBJECTION: what a placed strip actually does to the fold
# =====================================================================================

def test_a_placed_strip_straddling_a_proven_cards_edge_is_the_incidents_own_refusal():
    """WHY the hold-out exists, not merely that it works. This is the incident's exact arithmetic.

    Card at page rows 7387..8072, bounded end to end. A LATER frame's strip, placed at
    `frame_row + offset` = 8052..8095, straddles that card's bottom edge — a page row it never
    occupied, since it is pinned to the screen and the page moved 8000+px underneath it. The fold
    reads it as a fragment of the card that reaches 23px past the card's own measured bottom, which
    is the invariant `_resolve_group` refuses on: a fragment cannot be bigger than the thing it is a
    fragment of.

    On the real capture that refusal took the WHOLE index with it (`usable` False, no items), and
    `_split_on_bounded_cards` could not rescue it because only one of the two bridged cards had a
    complete sighting."""
    blocks, failures = _assemble([
        _obs(13, 7387, 8072, hearts=(7983,)),
        _obs(16, 8052, 8095, complete=False, kind=segment.BLOCK_UNANCHORED,
             top_kind=segment.EDGE_BACKGROUND_RUN),
    ], at_scroll_top=False)

    assert len(blocks) == 1
    assert any("a fragment cannot reach past the card that contains it" in f for f in failures), \
        failures
    assert any("frame 16 sees page rows 8052..8095" in f for f in failures)


def test_a_placed_strip_landing_in_the_gutter_fires_the_fabricated_item_guard_instead():
    """The SAME strip, five pixels further down the page, breaks a different invariant — which is
    the honest summary of what a fabricated page row does: it is not one failure mode, it is
    whatever the arithmetic happens to hit.

    Placed at 8077..8120 it clears the card at 7387..8072 entirely, so `_overlap_groups` makes it
    its own block 5px below a real one. Two resolved blocks that close contradict the layout (every
    one of the 206 real card-to-card gutters in the corpus measured exactly 53px), and the reading
    that matters is the dangerous one: one card whose sightings fragmented in two, i.e. doc 5.10's
    phantom item. So the guard fires and the index refuses."""
    blocks, failures = _assemble([
        _obs(13, 7387, 8072, hearts=(7983,)),
        _obs(16, 8077, 8120, complete=False, kind=segment.BLOCK_UNANCHORED,
             top_kind=segment.EDGE_BACKGROUND_RUN),
    ], at_scroll_top=False)

    assert len(blocks) == 2
    assert any("are only 5px apart, closer than the 47px minimum gutter" in f
               for f in failures), failures
    assert any("counting them as two would fabricate an item" in f for f in failures)


def test_a_proven_strip_produces_neither_refusal_because_it_never_becomes_an_observation():
    """The two refusals above are what the fix PREVENTS, and it prevents them at the earliest
    possible point rather than by widening a tolerance downstream. Same two frames, same strip,
    same digest — the only difference is whether the capture's own offsets proved it screen-fixed.

    Proven, the fold sees two cards and nothing else. Withheld, the strip from frame 1 lands at
    8052..8095 and reproduces the straddle refusal above exactly. Widening `_EXTENT_TOLERANCE_PX`
    to absorb the straddle would have been the tempting fix and is the wrong one: the strip's
    fabricated row moves with the scroll, so the next capture's overrun is a different size."""
    def frame_at(offset):
        return _segmentation([
            _pinned_strip_block(),
            _seg_block(_PINNED_CONTENT_Y0, _PINNED_CONTENT_Y0 + 685,
                       kind=segment.BLOCK_SELECTABLE,
                       top=_edge(_PINNED_CONTENT_Y0, segment.EDGE_CARD_CORNER, observed=True),
                       bottom=_edge(_PINNED_CONTENT_Y0 + 685, segment.EDGE_CARD_CORNER,
                                    observed=True),
                       hearts=((_HEART_CX, _PINNED_CONTENT_Y0 + 595),)),
        ], digest=f"{offset:064d}")

    segmentations = [frame_at(6870), frame_at(7684)]
    offsets = [6870, 7684]
    proven, _notes = item_index._screen_fixed_islands(
        item_index._islands(segmentations, offsets))
    assert proven == frozenset({(_PINNED_Y0, _PINNED_Y1)}), "the capture must prove it first"

    held_out, failures = _assemble(
        item_index._observations(segmentations, offsets, proven), at_scroll_top=False)
    assert failures == ()
    assert [(b.page_y0, b.page_y1) for b in held_out] == [(7387, 8072), (8201, 8886)]

    # The control that proves WHICH mechanism kept those failures away: withhold the proof and the
    # second frame's strip lands at 8052..8095, straddling the first frame's bounded card.
    placed = item_index._observations(segmentations, offsets, frozenset())
    assert (8052, 8095) in {(o.page_y0, o.page_y1) for o in placed}
    _blocks, placed_failures = _assemble(placed, at_scroll_top=False)
    assert any("a fragment cannot reach past the card that contains it" in f
               for f in placed_failures), placed_failures


# =====================================================================================
# End to end, with pixels
# =====================================================================================

_PINNED_SCROLLS = (0, 300, 900, 1500)
# Frame 2 (scroll 900) is THE BRIDGE, and its arithmetic is the incident's. Its band opens on the
# pinned strip at frame rows 368..411; its real content resumes at frame row 517, which is world
# row 1417 — 64 rows INSIDE card 2. Merge the strip into that clipped card, as segment.py did
# before this fix, and the block's top becomes page row 900 + 368 = 1268, which is 32 rows inside
# card 1 (400..1300). That single sighting then chains card 1's fragments and card 2's complete
# sightings into one fold group, and card 1 — never bounded at this cadence, since the strip covers
# its own top corner in every frame — gives `_split_on_bounded_cards` nothing to split on.
assert _PINNED_SCROLLS[2] + _PINNED_CONTENT_Y0 > _CARD2[1], "the bridge frame must clip card 2"
assert _PINNED_SCROLLS[2] + _PINNED_Y0 < _CARD1[2], "and its merged top must land inside card 1"


def test_a_screen_pinned_header_capture_indexes_cleanly_and_places_the_strip_nowhere():
    """THE TEST THAT PINS THE WHOLE FIX, on real pixels rather than hand-written records.

    Four frames of the synthetic world with Hinge 10.1.0's pinned header painted over the top of
    the analysed band — the same strip at the same frame rows with the same pixels on all four,
    while the page scrolls 1500px underneath it. segment.py reports it as `BLOCK_UNANCHORED` on
    every frame; the capture's own offsets prove it screen-fixed; and the index that comes out is
    the page that was really painted, with the strip at no page position at all.

    Card 1 is `ITEM_PARTIAL` here and that is correct rather than a compromise: the strip covers
    its own top corner in every frame, so no frame ever bounded it. It still consumes heart ordinal
    1 while the model indices stay dense over the two selectable cards — translation (2, 3) — which
    is doc 5.3's asymmetry doing exactly its job under a defect that would otherwise have renumbered
    the profile.

    The control at the bottom is the load-bearing half: the SAME segmentations and the SAME offsets,
    re-folded with the proof withheld, reproduce the incident's own refusal."""
    frames = [_frame(s, pinned_prefix=True) for s in _PINNED_SCROLLS]

    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=False, identity_band=None)

    assert index.usable, index.failures
    assert index.offsets == _PINNED_SCROLLS
    assert index.source_frame_indices == tuple(range(len(frames))), \
        "a clean capture must not need a frame-omission recovery"
    assert index.recovered_from_fold_contradiction_frames == ()

    assert [(b.page_y0, b.page_y1, b.kind) for b in index.blocks] == [
        (_CARD1[1] + _PINNED_CONTENT_Y0 - _CARD1[1], _CARD1[2], item_index.ITEM_PARTIAL),
        (_CARD2[1], _CARD2[2], item_index.ITEM_SELECTABLE),
        (_VITALS[1], _VITALS[2], item_index.ITEM_CONTEXT),
        (_CARD3[1], _CARD3[2], item_index.ITEM_SELECTABLE),
        (_CARD4[1], _CARD4[1] + 113, item_index.ITEM_PARTIAL),   # the capture's truncated tail
    ]
    assert len(index.selectable) == 2
    assert index.translation == (2, 3)

    # NO block, and no SIGHTING inside one, may carry the strip's rows. Page rows would be the
    # fabricated `frame_row + offset`; frame rows are how the fabrication gets in.
    strip_page_rows = {_PINNED_Y0 + off for off in index.offsets}
    assert not any(b.page_y0 in strip_page_rows for b in index.blocks)
    assert all(o.frame_y0 >= _PINNED_CONTENT_Y0
               for b in index.blocks for o in b.observations), \
        "no sighting may begin above the row where real content resumes"

    assert any("fixed to the SCREEN, not to the page" in n for n in index.notes), index.notes

    # THE CONTROL. Same segmentations, same offsets, proof withheld: this is byte for byte what
    # this module did before 2026-08-28, and it is the incident.
    pre_fix = item_index._observations(index.frames, index.offsets, frozenset())
    _blocks, pre_fix_failures = _assemble(pre_fix, at_scroll_top=False)
    assert any("a fragment cannot reach past the card that contains it" in f
               for f in pre_fix_failures), pre_fix_failures
    assert any("frame 2 sees page rows 1268..2113" in f for f in pre_fix_failures), \
        "the bridging sighting is the merged strip, 149px above any real content"
    assert item_index._fragment_overrun_frame_indices(pre_fix_failures) == (2,), \
        "and only the BRIDGING frame is nominated, never the three it dragged in"


def test_a_heart_inside_a_proven_screen_fixed_strip_is_a_hard_failure(monkeypatch):
    """DEFENCE IN DEPTH, deliberately unreachable today.

    segment.py's clause (e) refuses to label any strip that holds a heart, or whose adjacent run
    does, so nothing can currently reach this check. It exists because the consequence of a future
    loosening is the single error class this whole module is built to make impossible: a dropped
    heart renumbers every ordinal below it, silently, and the model's item 3 then taps item 4.

    Holding a strip out of the page is a DISCARD, and a discard has to be able to fail loudly. The
    message names the frame rows so an operator can go and look at the pixels, which is the only
    way to tell a loosened predicate from a real heart drawn on real chrome."""
    frames = [_frame(s, pinned_prefix=True) for s in _PINNED_SCROLLS[:2]]
    real_segment = item_index.segment_frame
    labelled = []

    def heart_on_the_island(*args, **kwargs):
        seg = real_segment(*args, **kwargs)
        islands = [b for b in seg.blocks if b.kind == segment.BLOCK_UNANCHORED]
        labelled.extend(islands)
        return dataclasses.replace(seg, blocks=tuple(
            dataclasses.replace(b, hearts=((_HEART_CX, b.y0 + 20),))
            if b.kind == segment.BLOCK_UNANCHORED else b for b in seg.blocks))

    monkeypatch.setattr(item_index, "segment_frame", heart_on_the_island)
    index = item_index.build_item_index(
        frames, content_band=_CONTENT_BAND, like_template=_TEMPLATE,
        like_threshold=hinge._LIKE_MATCH_THRESHOLD, at_scroll_top=False, identity_band=None)

    assert labelled, "the fixture must actually have produced an unanchored strip to poison"
    assert not index.usable
    assert any(f"the strip at frame rows {_PINNED_Y0}..{_PINNED_Y1} was proven fixed to the "
               "screen and held out of the page" in f for f in index.failures), index.failures
    assert any("would drop a likeable item and renumber every heart ordinal below it" in f
               for f in index.failures)


# =====================================================================================
# `_scroll_top_evidence`: the background test alone had gone VACUOUS
# =====================================================================================

_SCROLL_TOP_ITEMS = [_obs(0, 700, 1674, hearts=(1584,)), _obs(1, 1727, 2500, hearts=(2410,))]
_SCROLL_TOP_CHROME = _obs(0, _BAND0 + 34, 480, complete=False)


def _top_evidence(observations):
    """`_scroll_top_evidence` over the blocks these sightings fold into, asked in the state
    `_assemble` itself asks it in: BEFORE the leading-chrome relabelling, which only happens once
    this check has already declined to contradict the claim. Folding at `at_scroll_top=False`
    is how that state is reached, since the relabelling is gated on the assertion."""
    blocks, _failures = _assemble(observations, at_scroll_top=False)
    return item_index._scroll_top_evidence(blocks, _BAND0)


def test_background_above_plus_the_first_items_own_corner_corroborates_a_scroll_top():
    """The POSITIVE control for the two tests below, stated as the conjunction the check now is.

    Both halves are present: page background above the topmost block (34 rows, the smaller of the
    two clearances the corpus measured), and the first real item bounded by its OWN rounded corner
    — which at a genuine list top is visible, and is how item 1 gets bounded at all. Nothing is
    contradicted, so nothing is reported, and Hinge's header is relabelled chrome."""
    sightings = [_SCROLL_TOP_CHROME] + _SCROLL_TOP_ITEMS
    blocks, failures = _assemble(sightings, at_scroll_top=True)

    assert _top_evidence(sightings) is None
    assert failures == ()
    assert blocks[0].kind == item_index.ITEM_LEADING_CHROME


def test_background_above_without_the_first_items_corner_is_no_longer_accepted():
    """THE REGRESSION THE PINNED HEADER CREATED, and the reason a second test was needed at all.

    The background test is purely NEGATIVE: it asks whether anything sat above the topmost content
    and treats "yes" as consistent with a scroll top. Hinge 10.1.0's pinned header sits below the
    band's first row on EVERY frame of a capture, top or not — so the test the corpus credits with
    refusing 21 of 21 and 9 of 10 falsely-asserted tops began returning True unconditionally.
    [measured on the incident capture: the topmost block starts at frame row 368 against a band
    opening at 300, at page offsets from 0 to 8485. It was catching nothing, and nothing said so.]

    So the first item must ALSO have been seen with its own top edge. Here it has a gutter above it
    instead — which is what every card except item 1 has, and therefore what a window that started
    below the top looks like. The failure must NOT claim the block was flush against the band edge:
    it is not, that is precisely the trap, and sending an operator to look at the band edge is the
    wrong place. [guidance derives from its precondition; a fixed sentence here would have named
    the wrong one of two tests.]"""
    interior = [dataclasses.replace(o, top_kind=segment.EDGE_GUTTER) for o in _SCROLL_TOP_ITEMS]
    sightings = [_SCROLL_TOP_CHROME] + interior
    blocks, failures = _assemble(sightings, at_scroll_top=True)

    contradiction = _top_evidence(sightings)
    assert contradiction is not None
    assert "no frame ever saw the first item's OWN top edge" in contradiction
    assert "its top read as ['gutter'] every time, never a card corner" in contradiction
    assert "a pinned header puts it there on every frame of a capture" in contradiction
    assert "begins on the analysed band's own first row" not in contradiction, \
        "the band-edge wording sends an operator to a row that is not the problem"
    assert contradiction in failures
    assert blocks[0].kind == item_index.ITEM_PARTIAL, "the evidence must not be relabelled chrome"

    # ...and it is scoped to the claim, exactly as the band-edge test is.
    _relative, relative_failures = _assemble(sightings, at_scroll_top=False)
    assert not any("at_scroll_top was asserted" in f for f in relative_failures)


def test_reconfirmed_unpinned_scroll_top_can_resolve_a_missing_photo_corner():
    """The driver may supply a second, independent top proof after the ordinary fold refuses.

    Hinge's name panel can border the first photo closely enough that segmentation sees the
    photo's upper gutter but not its rounded corner. The default remains a refusal (the test
    above protects that guard for every generic/offline caller); only a live recheck of the
    filter-chip detector on frame zero, after a whole-capture pinning check, may authorize this
    retry. The leading heartless name panel then becomes chrome and cannot consume a heart
    ordinal.
    """
    interior = [dataclasses.replace(o, top_kind=segment.EDGE_GUTTER)
                for o in _SCROLL_TOP_ITEMS]
    sightings = [_SCROLL_TOP_CHROME] + interior

    blocks, failures = _assemble(
        sightings, at_scroll_top=True, scroll_top_signal_confirmed=True)

    assert failures == ()
    assert blocks[0].kind == item_index.ITEM_LEADING_CHROME


def test_the_corner_is_looked_for_below_a_heartless_never_bounded_leading_block():
    """THE 10.0.x SHAPE, which must keep working. A page-anchored header — Hinge's filter chips and
    name row, which SCROLL — comes back as a heartless, never-bounded leading block, and the corner
    that proves a list top belongs to the card BELOW it. So that block is skipped when looking for
    the first item, deliberately the same shape the `ITEM_LEADING_CHROME` relabelling looks for.

    The skip is narrow, and the second half here is what makes that checkable: give the leading
    block a heart and it is an ITEM, not chrome, so it is no longer skipped — and having no corner
    of its own it now refuses. A skip that fired on anything at the top of a capture would hand the
    corner test to whichever block happened to be second, which past a scroll top is any card at
    all."""
    _skipped, failures = _assemble(
        [_SCROLL_TOP_CHROME] + _SCROLL_TOP_ITEMS, at_scroll_top=True)
    assert _top_evidence([_SCROLL_TOP_CHROME] + _SCROLL_TOP_ITEMS) is None
    assert failures == ()

    contradiction = _top_evidence(
        [_obs(0, _BAND0 + 34, 480, complete=False, hearts=(400,))] + _SCROLL_TOP_ITEMS)
    assert contradiction is not None
    assert f"the block at page rows {_BAND0 + 34}..480" in contradiction
    assert "never a card corner" in contradiction


def test_the_guarded_scroll_top_media_edge_counts_as_the_first_items_own_top():
    """`EDGE_SCROLL_TOP_MEDIA` is segment.py's separately gated frame-0 conjunction for a pale
    square photo whose corner arc is too low-contrast to measure directly — a partial corner plus
    square-card, heart and lower-gutter geometry. It is a corner proved by other means, not a
    weaker signal, and it is the ONLY other way item 1 can be bounded at a real scroll top.

    Rejecting it here would refuse exactly the captures segment.py went out of its way to rescue,
    which is why `_LIST_TOP_EDGE_KINDS` is a set of two and is imported from segment.py rather than
    re-listed."""
    media = [dataclasses.replace(_SCROLL_TOP_ITEMS[0], top_kind=segment.EDGE_SCROLL_TOP_MEDIA),
             _SCROLL_TOP_ITEMS[1]]
    sightings = [_SCROLL_TOP_CHROME] + media
    blocks, failures = _assemble(sightings, at_scroll_top=True)

    assert _top_evidence(sightings) is None
    assert failures == ()
    assert blocks[0].kind == item_index.ITEM_LEADING_CHROME


# =====================================================================================
# `_fragment_overrun_frame_indices`: which frames a bridge actually implicates
# =====================================================================================

def _bridged_group_failures():
    """The incident's own fold, in records rather than pixels: one card proven at 7387..8072 by two
    frames, a second card at 8125..9234 that no frame ever bounded (it is 1109px and needs an
    offset in 7134..7608; the capture's offsets straddle that at 7110 and 7637), one BRIDGING
    sighting whose extent spans both, and three ordinary sightings of the lower card that the
    bridge dragged into the same group.

    `_split_on_bounded_cards` cannot split this — only ONE of the two cards has a complete sighting
    — so `_resolve_group` emits four overrun failures plus the co-emitted ambiguity, which is
    exactly the shape the live capture produced."""
    return _assemble([
        _obs(13, 7387, 8072, hearts=(7983,)),
        _obs(14, 7387, 8072, hearts=(7983,)),
        _obs(15, 8125, 8600, complete=False),
        _obs(16, 8020, 9234, complete=False, hearts=(9145,)),        # the bridge
        _obs(17, 8125, 9234, complete=False, hearts=(9145,)),
        _obs(18, 8125, 9000, complete=False),
    ], at_scroll_top=False)[1]


def test_only_the_bridging_frame_is_nominated_never_the_ones_it_dragged_in():
    """CORRECTION TWO, made on a live capture. A bridge drags its neighbours' ordinary sightings
    into its group, and those get an overrun failure too — six frames were nominated when exactly
    one had crossed anything. Dropping all six would have removed the only two complete sightings
    of the card below AND the last frame of the capture; dropping the one bridge yields a correct
    index.

    A fragment that OVERLAPS the bounded card and runs past it is the one that crossed the
    boundary. A fragment DISJOINT from it crossed nothing. Both extents are printed in the failure
    itself, so the distinction is read off the numbers an operator can see — the same four numbers
    `_resolve_group` used to word the message, never a second guess at the geometry.

    Here frames 15, 17 and 18 are ordinary sightings of the card below (all starting at 8125,
    clear of the card bounded at 7387..8072) and frame 16 is the only extent that spans the
    boundary. Nominating all four would drop the whole run; nominating frame 16 leaves a capture
    that rebuilds."""
    failures = _bridged_group_failures()

    overruns = [f for f in failures
                if "a fragment cannot reach past the card that contains it" in f]
    assert len(overruns) == 4
    assert sum("it is a sighting of the neighbouring card, dragged into this group" in f
               for f in overruns) == 3
    assert item_index._fragment_overrun_frame_indices(failures) == (16,)


def test_the_co_emitted_ambiguity_only_vetoes_when_it_names_a_different_block():
    """CORRECTION ONE. The group-ambiguity failure reads as an independent second fault, and the
    original rule refused anything it did not recognise for exactly that reason. But it is not
    independent — it is a DETERMINISTIC CONSEQUENCE of the same merge: two cards folded into one
    group put two hearts in one group, so `_resolve_group` co-emits it on every bridge whose lower
    card carries a heart, which is most of them. The recovery was therefore unreachable in the
    common case while appearing to be available.

    [measured on the incident: `block at page rows 7387..8072 is ambiguous: 2 distinct like hearts
    at page rows [7983, 9145]` — naming the very block the overrun failures name.] An ambiguous
    block ANYWHERE ELSE is a second fault and still refuses; the difference between the two is
    whether the extent it names is one the overruns already bound."""
    failures = _bridged_group_failures()
    ambiguity = next(f for f in failures if "is ambiguous:" in f)
    assert "block at page rows 7387..8072 is ambiguous: 2 distinct like hearts at page rows " \
        "[7983, 9145]" in ambiguity

    assert item_index._fragment_overrun_frame_indices(failures) == (16,)

    elsewhere = [ambiguity.replace("7387..8072", "9287..10396") if f is ambiguity else f
                 for f in failures]
    assert item_index._fragment_overrun_frame_indices(elsewhere) == ()


def test_any_unrelated_failure_still_refuses_the_omission_recovery():
    """The recovery drops real frames, so it is only ever justified when EVERY fold failure is the
    same merge. A bad shift, two competing complete extents, an uncertain heart or any other
    assembly fault means something else is also wrong, and dropping frames would be guessing."""
    failures = _bridged_group_failures()
    uncertain_heart = ("the block at page rows 6697..7334 " + item_index._UNCERTAIN_HEART_NOTE)

    assert item_index._fragment_overrun_frame_indices((*failures, uncertain_heart)) == ()
    assert item_index._fragment_overrun_frame_indices(()) == ()


# =====================================================================================
# `_resolve_group`: whose fault a fold contradiction is
# =====================================================================================

def test_two_disjoint_sightings_from_one_frame_are_not_a_self_contradictory_frame():
    """THE WRONG-FRAME REFUSAL, fixed 2026-08-28. On the incident capture the fold told frame 14 it
    "contradicted itself" when frame 14 had done nothing wrong: it had bounded one card AND owned a
    perfectly ordinary DISJOINT fragment of the next one, and some OTHER frame's missed boundary had
    folded the two cards into a single group. An operator following that message goes and stares at
    the one frame in the capture that is correct.

    segment.py's blocks are disjoint within a frame, so "same frame" alone says nothing; whether the
    two extents OVERLAP is what separates a genuinely self-contradictory frame from a victim of
    someone else's bridge. The mirror of this test already existed in the complete-vs-complete loop
    and simply had no counterpart in the fragment loop."""
    _blocks, failures = _assemble([
        _obs(14, 7387, 8072, hearts=(7983,)),          # frame 14 bounds one card...
        _obs(14, 8125, 8500, complete=False),          # ...and sees a disjoint fragment of the next
        _obs(16, 7975, 8600, complete=False),          # ...while frame 16 is the one that bridges
    ], at_scroll_top=False)

    victim = next(f for f in failures if "frame 14 sees page rows 8125..8500" in f)
    assert "self-contradictory" not in victim
    assert "frame 14 is not disagreeing with itself" in victim
    assert "look for the frame whose own block spans this boundary" in victim

    culprit = next(f for f in failures if "frame 16 sees page rows 7975..8600" in f)
    assert "either a gutter was missed" in culprit

    # The wording and the nomination have to agree: the frame the message exonerates is the frame
    # a recovery must not drop. Frame 14 owns the only two complete sightings in this group.
    assert item_index._fragment_overrun_frame_indices(failures) == (16,)


def test_two_overlapping_sightings_from_one_frame_still_say_self_contradictory():
    """The other side of the same gate, and the reason it is a gate rather than a deletion. A frame
    that contributes BOTH the bounding sighting and an OVERLAPPING fragment to one group should not
    happen at all — segment.py's blocks are disjoint within one frame — so it is a different and
    more surprising fault than an ordinary missed gutter, and it keeps its own wording."""
    _blocks, failures = _assemble([
        _obs(14, 7387, 8072, hearts=(7983,)),
        _obs(14, 8000, 8600, complete=False),
    ], at_scroll_top=False)

    contradiction = next(f for f in failures if "frame 14 sees page rows 8000..8600" in f)
    assert "their extents OVERLAP, which should never happen" in contradiction
    assert "self-contradictory frame" in contradiction
    # A sighting that really does span the boundary is a bridge, whoever's frame it came from.
    assert item_index._fragment_overrun_frame_indices(failures) == (14,)
