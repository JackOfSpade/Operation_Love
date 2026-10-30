"""Multiple dwell candidates share measured navigation and one final cleanup."""

from types import SimpleNamespace

import pytest

from operation_love.drivers import hinge, item_index, item_nav


def _index():
    return item_index.ItemIndex(
        blocks=tuple(
            item_index.IndexedBlock(
                page_y0=ordinal * 1000,
                page_y1=ordinal * 1000 + 900,
                x0=53,
                x1=1027,
                kind=item_index.ITEM_SELECTABLE,
                hearts=((937, ordinal * 1000 + 810),),
                heart_ordinal=ordinal,
                model_index=ordinal,
                observations=(),
                reason="synthetic navigation fixture",
            )
            for ordinal in range(1, 6)
        ),
        frames=(),
        shifts=(),
        offsets=(10000,),
        page_span=(0, 12000),
        at_scroll_top=True,
        reached_end=True,
        tail_gap_px=100,
        failures=(),
        identity=None,
    )


def _proved():
    return SimpleNamespace(
        dwell_exact=True,
        dwell_frame_sha256s=("first", "second"),
        centered=True,
        mute_screens_complete=True,
        dwell_span_s=1.0,
        reattach_probe_ran=True,
        reattach_dwell_frame_sha256s=("third", "fourth"),
        reattach_dwell_exact=True,
        reattach_centered=True,
        reattach_mute_screens_complete=True,
        reattach_dwell_span_s=1.0,
    )


def _driver():
    driver = object.__new__(hinge.HingeDriver)
    driver.still_photo_dwell_candidates = 3
    driver._dbg = None
    driver._device_inputs_delivered = 0
    driver._current_dwell_walk_interruption = None
    driver._require_targeting_calibration = lambda _n: SimpleNamespace(
        identity_match_max_dist=1.0
    )
    driver._note_capture_progress = lambda _text: None
    # These candidates all fit the old short-hop envelope. Dispatch must no longer
    # consult it before deciding that multiple visits should share a return.
    driver._still_photo_dwell_walk_return_budget = lambda *_a: pytest.fail(
        "multiple typed candidates must share a return before the short-hop budget"
    )
    return driver


def _walk(driver, index, banked, *, should_stop=None):
    return driver._still_photo_dwell_candidate_walk(
        {5: banked},
        frames=[b"entry"],
        index=index,
        eligible_heart_ordinals={1, 2, 3, 4},
        mute_screen=lambda *_a: True,
        entry_anchor=hinge._MeasuredItemAnchor(b"entry", -30),
        should_stop=should_stop,
    )


def test_shallow_candidates_rebase_each_measured_terminal_and_share_one_return(
    monkeypatch,
):
    driver, index, banked = _driver(), _index(), object()
    visits, cleanups = [], []
    proof = _proved()

    def navigate(_driver, rebased, model_index, *, entry_reference, **_kwargs):
        visits.append((model_index, rebased.offsets[-1], entry_reference))
        return SimpleNamespace(
            frame=f"landing-{model_index}".encode(),
            page_offset=rebased.offsets[-1] - 500,
            climbed_px=500,
        )

    def dwell(target, _block, *, heart_ordinal, **_kwargs):
        correction = hinge._CenteringCorrection(
            total_px=-20 if heart_ordinal == 4 else 10,
            frame=f"centered-{heart_ordinal}".encode(),
        )
        probe = (
            SimpleNamespace(page_shift_px=5, frames=(b"probe-4",))
            if heart_ordinal == 4
            else None
        )
        return proof, probe, correction

    def cleanup(origin, **kwargs):
        cleanups.append((origin, kwargs))
        return hinge._MeasuredItemAnchor(b"returned", -25)

    monkeypatch.setattr(hinge, "navigate_to_item", navigate)
    driver._still_photo_dwell_over_navigated_target = dwell
    driver._return_to_entry_from_measured_position = cleanup

    evidence, anchor = _walk(driver, index, banked)

    assert visits == [(4, 9970, b"entry"), (3, 9455, b"probe-4")]
    assert evidence == {5: banked, 4: proof, 3: proof}
    assert cleanups == [
        (
            hinge._MeasuredItemAnchor(b"entry", -30),
            {"frame": b"centered-3", "terminal_shift_px": -1005},
        )
    ]
    assert anchor == hinge._MeasuredItemAnchor(b"returned", -25)
    assert index.offsets == (10000,), (
        "the capture's original coordinate space is retained"
    )


def test_early_sweep_stop_returns_from_last_measured_candidate(monkeypatch):
    driver, index, banked = _driver(), _index(), object()
    visits, cleanups = [], []
    stopped = False
    proof = _proved()

    def navigate(_driver, rebased, model_index, **_kwargs):
        visits.append(model_index)
        return SimpleNamespace(frame=b"landed", page_offset=9470, climbed_px=500)

    def dwell(*_args, **_kwargs):
        nonlocal stopped
        stopped = True
        return proof, None, hinge._CenteringCorrection(total_px=0, frame=b"centered")

    def cleanup(origin, **kwargs):
        cleanups.append((origin, kwargs))
        return origin

    monkeypatch.setattr(hinge, "navigate_to_item", navigate)
    driver._still_photo_dwell_over_navigated_target = dwell
    driver._return_to_entry_from_measured_position = cleanup

    evidence, anchor = _walk(driver, index, banked, should_stop=lambda: stopped)

    assert visits == [4]
    assert evidence == {5: banked, 4: proof}
    assert anchor == hinge._MeasuredItemAnchor(b"entry", -30)
    assert cleanups[0][1] == {"frame": b"centered", "terminal_shift_px": -500}
    assert driver._current_dwell_walk_interruption[
        "unattempted_within_remaining_limit_page_hearts"
    ] == [3]


def test_early_sweep_unmeasured_navigation_refuses_without_inventing_cleanup(
    monkeypatch,
):
    driver, index, banked = _driver(), _index(), object()
    proof = _proved()
    visits = []

    def navigate(_driver, rebased, model_index, **_kwargs):
        visits.append(model_index)
        driver._device_inputs_delivered += 1
        if model_index == 3:
            raise item_nav.ItemNavigationError(item_nav.NAV_CHAIN_BROKEN, "unmeasured")
        return SimpleNamespace(frame=b"landed", page_offset=9470, climbed_px=500)

    monkeypatch.setattr(hinge, "navigate_to_item", navigate)
    driver._still_photo_dwell_over_navigated_target = lambda *_a, **_kw: (
        proof,
        None,
        hinge._CenteringCorrection(total_px=0, frame=b"centered"),
    )
    driver._return_to_entry_from_measured_position = lambda *_a, **_kw: pytest.fail(
        "an unmeasured navigation must never invent a cleanup distance"
    )

    evidence, anchor = _walk(driver, index, banked)

    assert visits == [4, 3]
    assert evidence == {5: banked, 4: proof}
    assert anchor is None


def test_gesture_free_refusal_keeps_the_exact_entry_without_cleanup(monkeypatch):
    driver, index, banked = _driver(), _index(), object()
    entry = hinge._MeasuredItemAnchor(b"entry", -30)

    def refuse(*_args, **_kwargs):
        raise item_nav.ItemNavigationError(
            item_nav.NAV_ANCHOR_UNMEASURED,
            "the entry could not be measured before input",
            anchor=SimpleNamespace(delta_px=None),
        )

    monkeypatch.setattr(hinge, "navigate_to_item", refuse)
    driver._return_to_entry_from_measured_position = lambda *_a, **_kw: pytest.fail(
        "a rescued entry that never moved needs no cleanup"
    )

    evidence, anchor = driver._still_photo_dwell_candidate_walk(
        {5: banked},
        frames=[entry.frame],
        index=index,
        mute_screen=lambda *_a: True,
        entry_anchor=entry,
    )

    assert evidence == {5: banked}
    assert anchor is entry


@pytest.mark.parametrize("missing", ["calibration", "entry_anchor"])
def test_shared_return_preserves_preflight_gates(missing):
    driver, index, banked = _driver(), _index(), object()
    entry = hinge._MeasuredItemAnchor(b"entry", -30)
    driver._still_photo_dwell_progressive_sweep = lambda *_a, **_kw: pytest.fail(
        "a sweep requires compatible calibration and a measured entry"
    )
    if missing == "calibration":

        def refuse(_n):
            raise hinge.HingeTargetingError("incompatible calibration")

        driver._require_targeting_calibration = refuse
    else:
        entry = None

    evidence, anchor = driver._still_photo_dwell_candidate_walk(
        {5: banked},
        frames=[b"entry"],
        index=index,
        mute_screen=lambda *_a: True,
        entry_anchor=entry,
    )

    assert evidence == {5: banked}
    assert anchor == entry
