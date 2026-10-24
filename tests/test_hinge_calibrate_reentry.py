"""Measured pre-heart re-entry, with a phone whose position changes after every gesture.

Only synthetic frames/worlds are used. The capture-path double enforces the navigator's
frame/offset contract instead of returning a fresh-looking target for every stale call.
The final test also runs the real navigator, index, segmenter and shift estimator.
"""

from __future__ import annotations

import hashlib
import json
import random
from types import SimpleNamespace

import pytest

from operation_love.drivers import hinge, item_nav
from tests.test_hinge_calibrate import _unattended_single_item_fixtures
from tests.test_item_nav import FakeDriver, _entry_reference, _reference_index
from tools import hinge_calibrate as cal


class _Gate:
    def __init__(self, decisions=()):
        self.decisions = iter(decisions)
        self.before = []

    def checkpoint(self, frame, *, claimed_state, action_plan):
        decision = "approved"
        if claimed_state == "target_heart_visible":
            self.before.append((frame, action_plan))
            decision = next(self.decisions, "approved")
        return {"decision": decision, "action_plan": action_plan}


def _moving_capture(monkeypatch, *, off_center=True, residual=40):
    driver, pass_calls, _ = _unattended_single_item_fixtures(monkeypatch)
    index = _reference_index()
    original_offset = index.offsets[-1]
    driver.position = original_offset
    driver.adjustments = []
    driver.post_heart_adjustments = []
    driver.terminal_passes = pass_calls
    driver.measurements = []
    driver.navigations = []
    driver.hearted = False

    def frame():
        return f"page:{driver.position}".encode()

    driver.adb.screencap = lambda: b"composer" if driver.hearted else frame()

    def adjust(direction):
        (
            driver.post_heart_adjustments if driver.hearted else driver.adjustments
        ).append(direction)
        # Deliberately differs from the requested 0.1 fraction. Position must be measured.
        driver.position += -400 if direction == "up" else 100

    driver._scroll_up_one = lambda *_: adjust("up")
    driver._scroll_down_one = lambda *_: adjust("down")

    def measured(before, after):
        driver.measurements.append((before, after))
        return int(after.split(b":")[1]) - int(before.split(b":")[1])

    driver._measured_page_shift = measured

    def navigate(_driver, walk_index, number, *, entry_reference, **_kwargs):
        assert walk_index.offsets[-1] == driver.position, "stale navigation page offset"
        assert entry_reference == frame(), "stale navigation reference frame"
        assert walk_index.translation == index.translation
        assert (
            walk_index.blocks is index.blocks and walk_index.identity is index.identity
        )
        assert number == 1
        driver.navigations.append((entry_reference, walk_index.offsets[-1]))
        if len(driver.navigations) == 1:
            # The initial successful navigation itself changes the page position.
            driver.position = 1000 if off_center else 600
        rows = (1500 - driver.position, 1700 - driver.position)
        return SimpleNamespace(
            frame=frame(),
            page_offset=driver.position,
            point=(500, rows[1] - 20),
            block_frame_rows=rows,
        )

    def proof(
        _driver, target, *, frame, expected_rows=None, expected_point=None, **_kw
    ):
        rows = target.block_frame_rows if expected_rows is None else expected_rows
        point = target.point if expected_point is None else expected_point
        return cal._TargetFrameProof(
            SimpleNamespace(x0=53, x1=1027, y0=rows[0], y1=rows[1], hearts=(point,)),
            cal._sha256(frame),
            True,
            True,
        )

    proofs = []

    def still(_driver, *, frame: bytes, block):
        delta = residual if not proofs else 0
        driver.position += delta
        action_frame = f"page:{driver.position}".encode()
        proofs.append((frame, action_frame))
        return cal._StillPhotoProof(
            action_frame_sha256=cal._sha256(action_frame),
            pre_probe_frame_sha256=cal._sha256(frame),
            dwell_frame_sha256s=(cal._sha256(frame), cal._sha256(frame)),
            dwell_span_s=6.0,
            still_photo_verified=True,
            reattach_frame_sha256s=(
                cal._sha256(action_frame),
                cal._sha256(action_frame),
            ),
            reattach_dwell_span_s=6.0,
            page_residual_px=delta,
            action_frame=action_frame,
        )

    def tap(*point):
        driver.taps.append(point)
        driver.hearted = True

    driver._tap = tap
    monkeypatch.setattr(cal, "_frame_height_px", lambda _: 2000)
    monkeypatch.setattr(
        cal, "_rewind_automated_profile_to_confirmed_top", lambda *_a, **_k: frame()
    )
    monkeypatch.setattr(cal, "build_item_index", lambda *_a, **_kw: index)
    monkeypatch.setattr(cal, "navigate_to_item", navigate)
    monkeypatch.setattr(cal, "_verified_target_frame_proof", proof)
    monkeypatch.setattr(cal, "_verified_still_photo_proof", still)
    monkeypatch.setattr(
        cal,
        "_fresh_reviewed_target_point",
        lambda _d, target, **kw: kw["expected_point"],
    )
    return driver, index


def _capture(driver, tmp_path, gate, **kwargs):
    return cal._capture_one_profile_unattended(
        driver,
        tmp_path,
        ordinal=1,
        frame_counter=0,
        frames_meta=[],
        used_profile_ids=set(),
        review_gate=gate,
        send_like=False,
        **kwargs,
    )


def test_capture_reentry_tracks_navigation_centering_probe_retry_and_reviewer_scroll(
    monkeypatch, tmp_path
):
    driver, index = _moving_capture(monkeypatch)
    original_offsets = index.offsets
    gate = _Gate(["retry", "scroll_down", "approved"])

    _capture(driver, tmp_path, gate)

    assert [offset for _, offset in driver.navigations] == [
        original_offsets[-1],
        600,
        640,
        740,
    ]
    assert driver.adjustments == ["up", "down"]
    assert driver.measurements == [
        (b"page:1000", b"page:600"),
        (b"page:640", b"page:740"),
    ]
    assert driver.taps == [(500, 940)]
    assert index.offsets == original_offsets
    assert [frame for frame, _plan in gate.before] == [
        b"page:640",
        b"page:640",
        b"page:740",
    ]


@pytest.mark.parametrize(
    "measurement", [None, True, 1.5], ids=["unknown", "bool", "fraction"]
)
def test_an_unmeasured_adjustment_aborts_before_any_heart(
    monkeypatch, tmp_path, measurement
):
    driver, _ = _moving_capture(monkeypatch)
    driver._measured_page_shift = lambda *_: measurement
    gate = _Gate()

    with pytest.raises(cal._CaptureAbort, match="displacement could not be measured"):
        _capture(driver, tmp_path, gate)

    assert driver.taps == [] and gate.before == []
    assert driver.adjustments == ["up"]


def test_a_reentry_that_cannot_rebase_the_original_index_aborts_before_heart(
    monkeypatch, tmp_path
):
    driver, _ = _moving_capture(monkeypatch)
    rebase = hinge.HingeDriver._rebase_item_index_for_anchor
    monkeypatch.setattr(
        hinge.HingeDriver,
        "_rebase_item_index_for_anchor",
        staticmethod(lambda index, shift: rebase(index, shift) if shift == 0 else None),
    )
    with pytest.raises(cal._CaptureAbort, match="cannot be rebased"):
        _capture(driver, tmp_path, _Gate())
    assert driver.taps == []


def test_reviewer_scroll_budget_is_checked_before_the_next_gesture(
    monkeypatch, tmp_path
):
    driver, _ = _moving_capture(monkeypatch, off_center=False, residual=0)
    monkeypatch.setattr(cal, "_HYBRID_MAX_ADJUSTMENTS_PER_ACTION", 1)

    with pytest.raises(cal._CaptureAbort, match="bounded hybrid adjustment budget"):
        _capture(driver, tmp_path, _Gate(["scroll_down", "scroll_down"]))

    assert driver.adjustments == ["down"]
    assert driver.taps == []


def test_terminal_reviewer_budget_refuses_before_an_extra_scroll_or_send(
    monkeypatch, tmp_path
):
    driver, _ = _moving_capture(monkeypatch, off_center=False, residual=0)
    monkeypatch.setattr(cal, "_HYBRID_MAX_ADJUSTMENTS_PER_ACTION", 1)

    class TerminalGate(_Gate):
        attempts = 0

        def checkpoint(self, frame, *, claimed_state, action_plan):
            if claimed_state == "composer_open_before_pass":
                self.attempts += 1
                return {
                    "decision": "scroll_down" if self.attempts <= 2 else "approved",
                    "action_plan": action_plan,
                }
            return super().checkpoint(
                frame, claimed_state=claimed_state, action_plan=action_plan
            )

    with pytest.raises(
        cal._CaptureAbort, match="bounded hybrid adjustment budget before Pass"
    ):
        _capture(driver, tmp_path, TerminalGate())

    assert len(driver.taps) == 1  # only the already-approved photo heart
    assert driver.post_heart_adjustments == ["down"]
    assert driver.terminal_passes == []


def test_a_reentry_refusal_records_its_actual_measured_reference(monkeypatch, tmp_path):
    driver, _ = _moving_capture(monkeypatch)
    navigate = cal.navigate_to_item

    def refuse_second(*args, **kwargs):
        if driver.navigations:
            raise item_nav.ItemNavigationError(
                item_nav.NAV_ANCHOR_UNMEASURED,
                "unexpected external drift",
                frame=b"untracked-motion",
                anchor=SimpleNamespace(
                    delta_px=595, status="measured", confidence=1.0, reason="drift"
                ),
            )
        return navigate(*args, **kwargs)

    monkeypatch.setattr(cal, "navigate_to_item", refuse_second)
    monkeypatch.setattr(
        cal, "_skip_automated_profile_before_heart", lambda *_a, **_k: {}
    )
    with pytest.raises(cal._ProfileSkipped):
        _capture(driver, tmp_path, _Gate(), entry_drift_restart_attempts=1)

    base = tmp_path / "forensics" / "refused_navigation_p1_item1_scan2"
    reference = base.with_name(base.name + "_read_reference.png").read_bytes()
    assert reference == b"page:600"
    assert (
        json.loads(base.with_suffix(".json").read_text())["read_reference_sha256"]
        == hashlib.sha256(reference).hexdigest()
    )
    assert driver.taps == []


def test_measured_anchor_reentry_uses_the_real_navigator_and_preserves_drift_guard(
    monkeypatch,
):
    random.seed(17)  # the synthetic world's calibrated deterministic navigation fixture
    index = _reference_index()
    driver = FakeDriver(allow_entry_scroll=True)
    first = item_nav.navigate_to_item(
        driver,
        index,
        1,
        entry_reference=_entry_reference(),
        identity_match_max_dist=1.0,
    )
    driver.adb = SimpleNamespace(screencap=driver._screencap)
    driver.dwell_s = 0.0
    driver._measured_page_shift = lambda before, after: (
        cal.estimate_shift(before, after, content_band=driver.content_band).delta_px
    )
    monkeypatch.setattr(cal, "human_delay", lambda _: 0.0)
    driver._scroll_up_one(0.1, 0.5)
    frame, page_offset = cal._adjusted_navigation_anchor(
        driver, before=first.frame, page_offset=first.page_offset
    )
    rebased = hinge.HingeDriver._rebase_item_index_for_anchor(
        index, page_offset - index.offsets[-1]
    )
    second = item_nav.navigate_to_item(
        driver, rebased, 1, entry_reference=frame, identity_match_max_dist=1.0
    )
    assert second.heart_ordinal == first.heart_ordinal
    assert second.anchor.delta_px == 0
    assert rebased.blocks is index.blocks and rebased.identity is index.identity

    # A new, untracked movement is still rejected by the navigator's original entry guard.
    driver.scroll += 400
    with pytest.raises(item_nav.ItemNavigationError, match="screen has moved"):
        item_nav.navigate_to_item(
            driver, rebased, 1, entry_reference=frame, identity_match_max_dist=1.0
        )
