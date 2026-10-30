"""Extra marker witnesses must locate distinct pixels with the calibrated matcher."""

import dataclasses
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from operation_love.drivers import frameshift, hinge
from tests.test_hinge_item_capture import WorldAdb, _drv, _frame


def _png(image):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize("delta", (-73, 73))
def test_interleaved_strips_find_a_real_translation_with_default_matcher(delta):
    """A stable patch between normal strip origins gets a genuine full-band search."""
    rng = np.random.default_rng(482)
    before = rng.integers(40, 220, (900, 180), dtype=np.uint8)
    after = rng.integers(40, 220, before.shape, dtype=np.uint8)
    tops = frameshift._strip_tops(90, 810, 13, 96)
    y0 = (tops[5] + tops[6]) // 2
    after[y0 - delta : y0 - delta + 96, 53:127] = before[y0 : y0 + 96, 53:127]

    matches = frameshift.interleaved_strip_matches(
        _png(before), _png(after), content_band=(0.1, 0.9)
    )

    witness = next(s for s in matches if s.y0 == y0)
    assert witness.y1 - witness.y0 == 96
    assert witness.state == frameshift.STRIP_MATCHED
    assert witness.delta_px == delta
    assert witness.score > 0.99
    assert y0 not in tops


@pytest.mark.parametrize("scene", ("flat", "unrelated", "pinned"))
def test_interleaved_strips_keep_flat_weak_and_boundary_matches_unusable(scene):
    rng = np.random.default_rng(688)
    before = rng.integers(40, 220, (900, 180), dtype=np.uint8)
    after = rng.integers(40, 220, before.shape, dtype=np.uint8)
    tops = frameshift._strip_tops(90, 810, 13, 96)
    y0 = (tops[5] + tops[6]) // 2
    if scene == "flat":
        before[:] = after[:] = 128
    elif scene == "pinned":
        after[90:186, 53:127] = before[y0 : y0 + 96, 53:127]

    matches = frameshift.interleaved_strip_matches(
        _png(before), _png(after), content_band=(0.1, 0.9)
    )

    witness = next(s for s in matches if s.y0 == y0)
    assert (
        witness.state
        == {
            "flat": frameshift.STRIP_FLAT,
            "unrelated": frameshift.STRIP_WEAK,
            "pinned": frameshift.STRIP_PINNED,
        }[scene]
    )


def _strip(y0, delta, *, state=frameshift.STRIP_MATCHED):
    return frameshift.StripMatch(
        y0=y0,
        y1=y0 + 96,
        state=state,
        delta_px=delta,
        score=1.0,
        runner_up=0.0,
        stddev=30.0,
        search=(-1500, 1500),
    )


def _driver():
    drv = hinge.HingeDriver.__new__(hinge.HingeDriver)
    drv.content_band = (0.125, 0.875)
    return drv


def _raw(*strips):
    return SimpleNamespace(
        strips=strips, status=frameshift.SHIFT_NO_CONSENSUS, saturated=False
    )


@pytest.mark.parametrize("reverse_seed", (False, True))
@pytest.mark.parametrize("ascending", (False, True))
def test_extra_video_witness_preserves_direction_and_uses_distinct_pixels(
    monkeypatch, reverse_seed, ascending
):
    frames = (b"before", b"after")
    if reverse_seed:
        raw = (_raw(_strip(900, 25)), _raw(_strip(400, -100)))
        extra = {frames: (_strip(800, 100),), frames[::-1]: ()}
    else:
        raw = (_raw(_strip(400, 100)), _raw(_strip(900, 25)))
        extra = {frames: (), frames[::-1]: (_strip(800, -100),)}
    if ascending:
        frames = frames[::-1]
        raw = raw[::-1]
    monkeypatch.setattr(
        hinge,
        "interleaved_strip_matches",
        lambda before, after, **_kw: extra[(before, after)],
    )

    assert set(_driver()._independent_interleaved_video_deltas(frames, *raw)) == {
        -100 if ascending else 100
    }


@pytest.mark.parametrize(
    "extra_y,second_seed,expected",
    (
        (300, None, set()),  # Same pixels as source 400..496.
        (395, None, set()),  # A one-row overlap is still correlated evidence.
        (396, None, {100}),  # Half-open intervals may meet without overlapping.
        (500, 600, set()),  # Must be disjoint from every supporting regular strip.
    ),
)
def test_extra_video_witness_cannot_recount_a_regular_patch(
    monkeypatch, extra_y, second_seed, expected
):
    seeds = (_strip(400, 100),)
    if second_seed is not None:
        seeds += (_strip(second_seed, 100),)
    monkeypatch.setattr(
        hinge,
        "interleaved_strip_matches",
        lambda before, after, **_kw: (
            (_strip(extra_y, -100),) if before == b"after" else ()
        ),
    )

    assert (
        set(
            _driver()._independent_interleaved_video_deltas(
                (b"before", b"after"), _raw(*seeds), _raw(_strip(900, 25))
            )
        )
        == expected
    )


@pytest.mark.parametrize("missing", ("forward", "reverse", "reverse_result"))
def test_extra_video_witness_requires_both_original_banks(monkeypatch, missing):
    def unexpected_search(*_args, **_kwargs):
        pytest.fail("one original pixel bank cannot authorize an added witness")

    monkeypatch.setattr(hinge, "interleaved_strip_matches", unexpected_search)
    forward = _raw() if missing == "forward" else _raw(_strip(400, 100))
    reverse = (
        None
        if missing == "reverse_result"
        else (_raw() if missing == "reverse" else _raw(_strip(900, 25)))
    )
    assert (
        set(
            _driver()._independent_interleaved_video_deltas(
                (b"before", b"after"), forward, reverse
            )
        )
        == set()
    )


@pytest.mark.parametrize("state", (frameshift.STRIP_WEAK, frameshift.STRIP_PINNED))
def test_extra_video_witness_needs_an_interior_match(monkeypatch, state):
    monkeypatch.setattr(
        hinge,
        "interleaved_strip_matches",
        lambda before, after, **_kw: (
            (_strip(800, -100, state=state),) if before == b"after" else ()
        ),
    )
    assert (
        set(
            _driver()._independent_interleaved_video_deltas(
                (b"before", b"after"), _raw(_strip(400, 100)), _raw(_strip(900, 25))
            )
        )
        == set()
    )


def test_extra_video_witness_requires_the_exact_reciprocal_delta(monkeypatch):
    monkeypatch.setattr(
        hinge,
        "interleaved_strip_matches",
        lambda before, after, **_kw: (_strip(800, -101),) if before == b"after" else (),
    )
    assert (
        set(
            _driver()._independent_interleaved_video_deltas(
                (b"before", b"after"), _raw(_strip(400, 100)), _raw(_strip(900, 25))
            )
        )
        == set()
    )


def test_extra_video_witness_retains_competing_exact_translations(monkeypatch):
    monkeypatch.setattr(
        hinge,
        "interleaved_strip_matches",
        lambda before, after, **_kw: (
            (_strip(1100, -100), _strip(1400, -200)) if before == b"after" else ()
        ),
    )
    assert set(
        _driver()._independent_interleaved_video_deltas(
            (b"before", b"after"),
            _raw(_strip(400, 100), _strip(800, 200)),
            _raw(_strip(900, 25)),
        )
    ) == {100, 200}


@pytest.mark.parametrize("direction", (0, 1))
@pytest.mark.parametrize(
    "status",
    (
        frameshift.SHIFT_NO_EVIDENCE,
        frameshift.SHIFT_BEYOND_WINDOW,
        frameshift.SHIFT_MEASURED,
    ),
)
def test_extra_video_witness_does_not_override_other_refusal_classes(
    monkeypatch, direction, status
):
    def unexpected_search(*_args, **_kwargs):
        pytest.fail("only unresolved unsaturated consensus earns a supplemental search")

    monkeypatch.setattr(hinge, "interleaved_strip_matches", unexpected_search)
    banks = [_raw(_strip(400, 100)), _raw(_strip(900, 25))]
    banks[direction].status = status
    assert (
        set(
            _driver()._independent_interleaved_video_deltas(
                (b"before", b"after"), *banks
            )
        )
        == set()
    )


@pytest.mark.parametrize("saturated_direction", ("forward", "reverse"))
def test_video_interleaving_cannot_override_a_saturated_refusal(
    monkeypatch, saturated_direction
):
    drv = _driver()
    drv.spec = hinge.HINGE_SPEC

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("saturation must stop before any marker or extra pixel search")

    monkeypatch.setattr(drv, "_video_mute_marker_rows", unexpected_search)
    base = SimpleNamespace(status=frameshift.SHIFT_NO_CONSENSUS, saturated=False)
    forward = SimpleNamespace(**vars(base))
    reverse = SimpleNamespace(**vars(base))
    if saturated_direction == "forward":
        forward.saturated = True
    else:
        reverse.saturated = True
    assert (
        drv._recover_video_marker_page_shift(b"before", b"after", forward, reverse)
        is forward
    )


def _marker_bridge_frames(*, missing_control=False, broken_trajectory=False):
    frames = []
    mute = hinge._video_mute_template()
    for i, scroll in enumerate((700, 933)):
        image = cv2.imdecode(
            np.frombuffer(_frame(scroll), np.uint8), cv2.IMREAD_GRAYSCALE
        )
        y = 1403 - scroll + (7 if i == 1 and broken_trajectory else 0)
        if not (i == 1 and missing_control):
            image[y : y + mute.shape[0], 106 : 106 + mute.shape[1]] = mute
        frames.append(_png(image))
    return tuple(frames)


@pytest.mark.parametrize("ascending", (False, True))
@pytest.mark.parametrize(
    "gate", (None, "overlapping", "wrong_delta", "missing_control", "broken_anchor")
)
def test_an_unproved_regular_reciprocal_candidate_does_not_hide_the_extra_witness(
    monkeypatch, ascending, gate
):
    """A shared +126 strip peak has no corresponding control; +233 still earns proof.

    The original strip bank is deterministic here, while the markers, containing cards,
    exact card anchors and candidate-control searches all read actual synthetic pixels.
    """
    drv = _drv(WorldAdb())
    records = []
    drv._dbg = SimpleNamespace(
        action=lambda action, **fields: records.append((action, fields))
    )
    frames = _marker_bridge_frames(
        missing_control=gate == "missing_control",
        broken_trajectory=gate == "broken_anchor",
    )
    raw = dataclasses.replace(
        frameshift.estimate_shift(*frames, content_band=drv.content_band),
        status=frameshift.SHIFT_NO_CONSENSUS,
        delta_px=None,
        consensus_px=None,
        saturated=False,
        strips=(_strip(1000, 233), _strip(300, 126)),
    )
    reverse = dataclasses.replace(raw, strips=(_strip(1200, -126),))
    extra_y = 767 if gate == "overlapping" else 1500
    extra_delta = -234 if gate == "wrong_delta" else -233
    extra_by_pair = {frames: (), frames[::-1]: (_strip(extra_y, extra_delta),)}
    monkeypatch.setattr(
        hinge,
        "interleaved_strip_matches",
        lambda before, after, **_kw: extra_by_pair[(before, after)],
    )
    if ascending:
        frames = frames[::-1]
        raw, reverse = reverse, raw

    result = drv._recover_video_marker_page_shift(*frames, raw, reverse)

    expected = (-233 if ascending else 233) if gate is None else None
    assert result.delta_px == expected
    assert result.status == (
        frameshift.SHIFT_MEASURED if gate is None else frameshift.SHIFT_NO_CONSENSUS
    )
    if gate is None:
        action, evidence = records[-1]
        assert action == "page_shift_video_marker_measurement"
        assert evidence["forward"]["status"] == frameshift.SHIFT_NO_CONSENSUS
        witness = evidence["supplemental_witnesses"][0]
        assert witness["seed_frame"] == (1 if ascending else 0)
        assert witness["seed_rows"] == [[1000, 1096]]
        assert witness["supplemental_rows"] == [1500, 1596]
        assert witness["mapped_rows"] == [1733, 1829]
    else:
        assert records == []


@pytest.mark.parametrize("ascending", (False, True))
def test_a_proved_regular_video_pair_keeps_the_original_path(monkeypatch, ascending):
    drv = _drv(WorldAdb())
    frames = _marker_bridge_frames()
    raw = dataclasses.replace(
        frameshift.estimate_shift(*frames, content_band=drv.content_band),
        status=frameshift.SHIFT_NO_CONSENSUS,
        delta_px=None,
        consensus_px=None,
        saturated=False,
        strips=(_strip(300, 233), _strip(1000, 126)),
    )
    reverse = dataclasses.replace(raw, strips=(_strip(900, -233), _strip(1200, -126)))

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("a regular marker-and-anchor proof does not earn extra sampling")

    monkeypatch.setattr(hinge, "interleaved_strip_matches", unexpected_search)
    if ascending:
        frames = frames[::-1]
        raw, reverse = reverse, raw

    assert drv._recover_video_marker_page_shift(*frames, raw, reverse).delta_px == (
        -233 if ascending else 233
    )
