"""Bounded passive (gesture-free) recovery of one ascending chain refusal in item navigation.

All frames here are synthetic: the navigation tests drive `test_item_nav`'s arithmetic phone, and
the helper tests use opaque byte stand-ins with stubbed estimator/identity collaborators.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

try:  # pytest's default rootdir insertion
    import test_item_nav as nav_fx
except ImportError:  # pragma: no cover - alternative import layout
    from tests import test_item_nav as nav_fx

from operation_love.drivers import frameshift, item_nav
from operation_love.drivers.base import ActionCancelled
from operation_love.drivers.item_identity import IdentityError


@pytest.fixture(autouse=True)
def _seeded_and_no_real_sleep(monkeypatch):
    random.seed(nav_fx._SEED)
    sleeps: list[float] = []
    monkeypatch.setattr(item_nav, "_passive_sleep", sleeps.append)
    return sleeps


def _shift(status, delta=None, *, saturated=False, consensus=None, reason="synthetic"):
    return item_nav.ShiftEstimate(
        delta_px=delta,
        confidence=1.0 if status == frameshift.SHIFT_MEASURED else 0.25,
        status=status,
        saturated=saturated,
        consensus_px=delta if consensus is None else consensus,
        reason=reason,
        strips=(),
        frame_size=(1080, 2400),
        band=(nav_fx._BAND0, nav_fx._BAND1),
        trust_window_px=900,
        agreeing=8 if status == frameshift.SHIFT_MEASURED else 1,
        dissenting=0,
        eligible=8,
    )


def _no_evidence(result):
    return dataclasses.replace(
        result,
        delta_px=None,
        consensus_px=None,
        status=frameshift.SHIFT_NO_EVIDENCE,
        saturated=False,
        reason="forced: synthetic no-evidence refusal",
    )


class _RecordingDriver(nav_fx.FakeDriver):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.frames_read: list[bytes] = []

    def _screencap(self):
        # A distinct object per read (the fixture caches frames per scroll), so `is` checks
        # below identify exactly which screenshot was threaded where.
        frame = bytes(bytearray(super()._screencap()))
        self.frames_read.append(frame)
        return frame


def _patch_estimator(monkeypatch, transform):
    """Wrap the real estimator; `transform(call_number, before, after, result)` may replace it."""
    real = item_nav.estimate_shift
    calls: list[tuple[bytes, bytes]] = []

    def wrapped(before, after, **kwargs):
        calls.append((before, after))
        result = real(before, after, **kwargs)
        replaced = transform(len(calls), before, after, result)
        return result if replaced is None else replaced

    monkeypatch.setattr(item_nav, "estimate_shift", wrapped)
    return calls


def _baseline_target():
    random.seed(nav_fx._SEED)
    driver = _RecordingDriver()
    target = nav_fx._navigate(driver, 1)
    random.seed(nav_fx._SEED)
    return driver, target


# =====================================================================================
# Real navigation
# =====================================================================================


def test_ordinary_navigation_never_waits_or_resamples(_seeded_and_no_real_sleep):
    driver, target = _baseline_target()
    assert target.model_index == 1 and target.agreement_px == 0
    assert _seeded_and_no_real_sleep == []
    # One capture per frame of the walk (entry + one per gesture), nothing extra.
    assert driver.captures == len(driver.gestures) + 1
    assert len(driver.gestures) >= 2, "fixture must take several ascending steps"


def test_passive_recovery_continues_real_navigation_on_the_fresh_terminal_frame(
    monkeypatch, _seeded_and_no_real_sleep
):
    base_driver, base_target = _baseline_target()

    def force(n, _before, _after, result):
        if n == 2:  # first ascending leg's forward pass
            assert result.status == frameshift.SHIFT_MEASURED
            return _no_evidence(result)
        return None

    calls = _patch_estimator(monkeypatch, force)
    driver = _RecordingDriver()
    target = nav_fx._navigate(driver, 1)

    assert _seeded_and_no_real_sleep == [0.25, 0.25]
    # Zero extra gestures, exactly two extra screenshots.
    assert driver.gestures == base_driver.gestures
    assert driver.captures == base_driver.captures + 2
    # Stationarity measured fresh1 -> fresh2; terminal measured ORIGINAL frame 0 -> fresh2.
    fresh1, fresh2 = driver.frames_read[2], driver.frames_read[3]
    assert calls[2] == (fresh1, fresh2)
    assert calls[3] == (driver.frames_read[0], fresh2)
    # The next ascending leg starts from fresh2, not the refused post-gesture frame.
    assert calls[4][0] is fresh2
    assert calls[4][1] is driver.frames_read[4]
    # Shifts and landing identical to the never-refused walk.
    assert [s.delta_px for s in target.shifts] == [
        s.delta_px for s in base_target.shifts
    ]
    assert target.shifts[0].status == frameshift.SHIFT_MEASURED
    assert target.offsets == base_target.offsets
    assert target.point == base_target.point
    assert target.agreement_px == 0


def test_overshoot_after_valid_recovery_still_refuses_with_fresh_frame(monkeypatch):
    forced = {}

    def force(n, _before, _after, result):
        if n == 2:
            return _no_evidence(result)
        if n == 4:  # terminal: original before -> fresh2
            forced["delta"] = result.delta_px * 10
            return dataclasses.replace(
                result, delta_px=forced["delta"], consensus_px=forced["delta"]
            )
        return None

    _patch_estimator(monkeypatch, force)
    driver = _RecordingDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_SCROLL_OVERSHOT
    assert exc.value.frame is driver.frames_read[3]
    assert exc.value.recovery is not None
    assert exc.value.recovery.frame is driver.frames_read[3]
    assert exc.value.recovery.measurement_delta_px == forced["delta"]
    assert len(driver.gestures) == 1 and driver.captures == 4


def test_persistent_refusal_reports_latest_frame_and_traces(monkeypatch):
    def force(n, _before, _after, result):
        if n in (2, 4):
            return _no_evidence(result)
        return None

    _patch_estimator(monkeypatch, force)
    driver = _RecordingDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    err = exc.value
    assert err.code == item_nav.NAV_CHAIN_BROKEN
    assert err.recovery is None
    assert driver.captures == 4 and len(driver.gestures) == 1
    assert err.frame is driver.frames_read[3]
    assert err.pair_before is driver.frames_read[0]
    m = err.measurement
    json.dumps(m)
    assert m["achieved"]["forward"]["status"] == frameshift.SHIFT_NO_EVIDENCE
    assert m["latest_frame_source"] == "passive_resample"
    p = m["passive_resample"]
    assert p["schema_version"] == 1
    assert p["attempts"] == 1 and p["frames_captured"] == 2
    assert p["outcome"] == item_nav.PASSIVE_TERMINAL_UNMEASURED
    assert p["recovered"] is False
    assert p["stationarity"]["result"]["delta_px"] == 0
    assert p["terminal"]["available"] is True
    assert p["terminal"]["authorizing"] is False
    assert p["terminal"]["result"]["status"] == frameshift.SHIFT_NO_EVIDENCE


@pytest.mark.parametrize(
    "refusal",
    [
        lambda r: dataclasses.replace(
            r,
            delta_px=None,
            status=frameshift.SHIFT_BEYOND_WINDOW,
            saturated=True,
            consensus_px=-1400,
            reason="forced beyond window",
        ),
        lambda r: dataclasses.replace(
            _no_evidence(r), saturated=True, reason="forced saturated no evidence"
        ),
    ],
    ids=["beyond_window", "saturated"],
)
def test_ineligible_refusals_take_no_extra_capture(
    monkeypatch, refusal, _seeded_and_no_real_sleep
):
    _patch_estimator(monkeypatch, lambda n, b, a, r: refusal(r) if n == 2 else None)
    driver = _RecordingDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_CHAIN_BROKEN
    assert driver.captures == 2 and len(driver.gestures) == 1
    assert _seeded_and_no_real_sleep == []
    assert exc.value.frame is driver.frames_read[1]
    p = exc.value.measurement["passive_resample"]
    assert p["attempts"] == 0 and p["frames_captured"] == 0
    assert p["outcome"] == item_nav.PASSIVE_INELIGIBLE
    assert p["reason"]
    assert exc.value.measurement["latest_frame_source"] == "initial_post_gesture"


def test_budget_is_spent_once_per_navigation(monkeypatch, _seeded_and_no_real_sleep):
    # Call 2 = first leg (recovered by calls 3/4), call 5 = second leg forward.
    _patch_estimator(
        monkeypatch, lambda n, b, a, r: _no_evidence(r) if n in (2, 5) else None
    )
    driver = _RecordingDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    assert exc.value.code == item_nav.NAV_CHAIN_BROKEN
    assert len(driver.gestures) == 2
    assert driver.captures == 5
    assert _seeded_and_no_real_sleep == [0.25, 0.25]
    assert exc.value.frame is driver.frames_read[4]
    assert exc.value.pair_before is driver.frames_read[3]
    p = exc.value.measurement["passive_resample"]
    assert p["outcome"] == item_nav.PASSIVE_BUDGET_EXHAUSTED
    assert p["attempts"] == 0 and p["frames_captured"] == 0


def test_cancellation_during_passive_wait_stops_captures_and_gestures(monkeypatch):
    stop = {"now": False}

    def sleep(_seconds):
        stop["now"] = True

    monkeypatch.setattr(item_nav, "_passive_sleep", sleep)
    _patch_estimator(
        monkeypatch, lambda n, b, a, r: _no_evidence(r) if n == 2 else None
    )
    driver = _RecordingDriver()
    with pytest.raises(ActionCancelled):
        nav_fx._navigate(driver, 1, should_stop=lambda: stop["now"])
    assert driver.captures == 2 and len(driver.gestures) == 1


def test_terminal_unmeasured_main_trace_is_terminal_not_initial(monkeypatch):
    def force(n, _before, _after, result):
        if n == 2:
            return dataclasses.replace(
                _no_evidence(result), reason="initial-only reason"
            )
        if n == 4:
            return dataclasses.replace(
                _no_evidence(result), reason="terminal-only reason"
            )
        return None

    _patch_estimator(monkeypatch, force)
    driver = _RecordingDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    err = exc.value
    assert err.code == item_nav.NAV_CHAIN_BROKEN and err.recovery is None
    assert err.frame is driver.frames_read[3]
    m = err.measurement
    json.dumps(m)
    achieved = m["achieved"]
    assert achieved["measurement_reason"] == "terminal-only reason"
    assert achieved["forward"]["reason"] == "terminal-only reason"
    assert achieved["source"] == "passive_terminal"
    assert achieved["authorizing"] is False
    initial = m["passive_resample"]["initial"]
    assert initial["measurement_reason"] == "initial-only reason"
    assert initial["forward"]["reason"] == "initial-only reason"
    assert "source" not in initial
    msg = str(err)
    assert "(the initial post-gesture pair)" in msg
    assert "rejected (terminal_unmeasured" in msg


def test_nonstationary_with_measured_terminal_stays_chain_broken(monkeypatch):
    seen = {}

    def force(n, _before, _after, result):
        if n == 2:
            return _no_evidence(result)
        if n == 3:  # stationarity fresh1 -> fresh2: force motion
            return dataclasses.replace(
                result, status=frameshift.SHIFT_MEASURED, delta_px=-12, consensus_px=-12
            )
        if n == 4:  # terminal: genuinely measured, left untouched
            seen["terminal"] = result
        return None

    _patch_estimator(monkeypatch, force)
    driver = _RecordingDriver()
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    err = exc.value
    assert seen["terminal"].status == frameshift.SHIFT_MEASURED
    assert seen["terminal"].delta_px is not None and seen["terminal"].delta_px != 0
    assert err.code == item_nav.NAV_CHAIN_BROKEN
    assert err.recovery is None
    assert err.frame is driver.frames_read[3]
    assert err.pair_before is driver.frames_read[0]
    assert driver.captures == 4 and len(driver.gestures) == 1
    m = err.measurement
    json.dumps(m)
    p = m["passive_resample"]
    assert p["outcome"] == item_nav.PASSIVE_NONSTATIONARY
    assert p["recovered"] is False
    assert p["terminal"]["authorizing"] is False
    # Main trace = original before -> latest frame (measured), not the initial refusal.
    achieved = m["achieved"]
    assert achieved["measurement_status"] == frameshift.SHIFT_MEASURED
    assert achieved["measurement_delta_px"] == seen["terminal"].delta_px
    assert achieved["forward"]["delta_px"] == seen["terminal"].delta_px
    assert achieved["authorizing"] is False
    # Initial preserved separately, and it differs.
    assert p["initial"]["measurement_status"] == frameshift.SHIFT_NO_EVIDENCE
    assert p["initial"]["measurement_delta_px"] is None
    msg = str(err)
    assert "rejected (nonstationary" in msg
    assert "diagnostic only, not authorizing" in msg


def test_identity_mismatch_main_trace_unavailable_initial_preserved(monkeypatch):
    calls = _patch_estimator(
        monkeypatch, lambda n, b, a, r: _no_evidence(r) if n == 2 else None
    )
    driver = _RecordingDriver()
    real_identity = item_nav.compare_profile_identity

    def identity(frame, ref, **kwargs):
        if any(frame is f for f in driver.frames_read[2:]):
            return _Verdict(False, distance=9.0, reason="foreign")
        return real_identity(frame, ref, **kwargs)

    monkeypatch.setattr(item_nav, "compare_profile_identity", identity)
    with pytest.raises(item_nav.ItemNavigationError) as exc:
        nav_fx._navigate(driver, 1)

    err = exc.value
    assert err.code == item_nav.NAV_CHAIN_BROKEN and err.recovery is None
    assert driver.captures == 3 and len(driver.gestures) == 1
    assert err.frame is driver.frames_read[2]
    assert len(calls) == 2  # nothing measured against the foreign frame
    m = err.measurement
    json.dumps(m)
    p = m["passive_resample"]
    assert p["outcome"] == item_nav.PASSIVE_IDENTITY_MISMATCH
    assert p["terminal"]["available"] is False
    achieved = m["achieved"]
    assert achieved["measurement_status"] == "unavailable"
    assert achieved["measurement_unavailable"] is True
    assert achieved["measurement_delta_px"] is None
    assert achieved["forward"] is None and achieved["reverse"] is None
    assert "identity_mismatch" in achieved["measurement_reason"]
    assert p["initial"]["measurement_status"] == frameshift.SHIFT_NO_EVIDENCE
    assert p["initial"]["forward"]["status"] == frameshift.SHIFT_NO_EVIDENCE
    msg = str(err)
    assert "rejected (identity_mismatch" in msg
    assert "was not measured because its identity" in msg


# =====================================================================================
# The helper's gates
# =====================================================================================


class _StubDriver:
    identity_band = (0, 10)

    def __init__(self, frames):
        self._frames = list(frames)
        self.captures = 0

    def _screencap(self):
        self.captures += 1
        return self._frames[self.captures - 1]

    def _recover_video_marker_page_shift(self, *_a, **_k):
        raise AssertionError("marker bridge must not run in this gate test")


@dataclasses.dataclass
class _Verdict:
    matched: bool
    unknown: bool = False
    distance: float | None = 0.5
    reason: str = "synthetic verdict"


def _run_helper(
    monkeypatch, *, verdicts, shifts, frames=(b"f1", b"f2"), cancelled=None
):
    verdict_iter = iter(verdicts)

    def identity(frame, ref, *, identity_band, match_max_dist):
        assert ref == "IDENT" and identity_band == (0, 10) and match_max_dist == 1.5
        v = next(verdict_iter)
        if isinstance(v, Exception):
            raise v
        return v

    est_calls = []

    def estimator(before, after, **kwargs):
        est_calls.append((before, after))
        assert kwargs == {"content_band": (1, 2), "trust_window_px": 900}
        return shifts[(before, after)]

    monkeypatch.setattr(item_nav, "compare_profile_identity", identity)
    monkeypatch.setattr(item_nav, "estimate_shift", estimator)
    driver = _StubDriver(frames)
    result = item_nav._passive_resample(
        driver,
        before=b"orig",
        reference_identity="IDENT",
        identity_match_max_dist=1.5,
        content_band=(1, 2),
        trust_window_px=900,
        cancelled=cancelled or (lambda: None),
    )
    return result, driver, est_calls


_STILL = _shift(frameshift.SHIFT_MEASURED, 0)
_TERMINAL = _shift(frameshift.SHIFT_MEASURED, -280)


def test_helper_recovers_with_stationary_matched_frames(
    monkeypatch, _seeded_and_no_real_sleep
):
    monkeypatch.setattr(
        _StubDriver, "_recover_video_marker_page_shift", lambda self, b, a, r, rev: r
    )
    result, driver, calls = _run_helper(
        monkeypatch,
        verdicts=[_Verdict(True), _Verdict(True)],
        shifts={(b"f1", b"f2"): _STILL, (b"orig", b"f2"): _TERMINAL},
    )
    assert result.recovered and result.outcome == item_nav.PASSIVE_RECOVERED
    assert result.estimate.delta_px == -280
    assert result.frame == b"f2" and result.latest_frame == b"f2"
    assert calls == [(b"f1", b"f2"), (b"orig", b"f2")]
    assert driver.captures == 2
    assert _seeded_and_no_real_sleep == [0.25, 0.25]
    json.dumps(result.metadata)
    assert result.metadata["terminal"]["authorizing"] is True


def test_helper_refuses_nonstationary_frames_but_keeps_raw_terminal(monkeypatch):
    monkeypatch.setattr(
        _StubDriver, "_recover_video_marker_page_shift", lambda self, b, a, r, rev: r
    )
    result, driver, _calls = _run_helper(
        monkeypatch,
        verdicts=[_Verdict(True), _Verdict(True)],
        shifts={
            (b"f1", b"f2"): _shift(frameshift.SHIFT_MEASURED, -12),
            (b"orig", b"f2"): _TERMINAL,
        },
    )
    assert not result.recovered and result.estimate is None and result.frame is None
    assert result.outcome == item_nav.PASSIVE_NONSTATIONARY
    assert result.latest_frame == b"f2"
    assert result.metadata["terminal"]["result"]["delta_px"] == -280
    assert result.metadata["terminal"]["authorizing"] is False
    assert result.metadata["stationarity"]["result"]["delta_px"] == -12


def test_helper_refuses_unmeasured_stationarity(monkeypatch):
    monkeypatch.setattr(
        _StubDriver, "_recover_video_marker_page_shift", lambda self, b, a, r, rev: r
    )
    result, _driver, _calls = _run_helper(
        monkeypatch,
        verdicts=[_Verdict(True), _Verdict(True)],
        shifts={
            (b"f1", b"f2"): _shift(frameshift.SHIFT_NO_EVIDENCE),
            (b"orig", b"f2"): _TERMINAL,
        },
    )
    assert result.outcome == item_nav.PASSIVE_NONSTATIONARY and not result.recovered


@pytest.mark.parametrize(
    "verdicts, outcome, captures",
    [
        ([_Verdict(False)], item_nav.PASSIVE_IDENTITY_MISMATCH, 1),
        ([_Verdict(True), _Verdict(False)], item_nav.PASSIVE_IDENTITY_MISMATCH, 2),
        ([_Verdict(False, unknown=True)], item_nav.PASSIVE_IDENTITY_UNCONFIRMED, 1),
        ([IdentityError("unreadable")], item_nav.PASSIVE_IDENTITY_UNCONFIRMED, 1),
    ],
)
def test_helper_identity_gate_fails_closed_without_measuring(
    monkeypatch, verdicts, outcome, captures
):
    result, driver, calls = _run_helper(monkeypatch, verdicts=verdicts, shifts={})
    assert result.outcome == outcome and not result.recovered
    assert driver.captures == captures
    assert result.latest_frame == (b"f1", b"f2")[captures - 1]
    assert calls == []  # no estimator, no marker bridge on a foreign/unknown frame
    assert result.metadata["terminal"]["available"] is False
    assert result.metadata["frames_captured"] == captures
    json.dumps(result.metadata)


def test_helper_terminal_unmeasured_fails_closed(monkeypatch):
    monkeypatch.setattr(
        _StubDriver, "_recover_video_marker_page_shift", lambda self, b, a, r, rev: r
    )
    result, _driver, _calls = _run_helper(
        monkeypatch,
        verdicts=[_Verdict(True), _Verdict(True)],
        shifts={
            (b"f1", b"f2"): _STILL,
            (b"orig", b"f2"): _shift(frameshift.SHIFT_NO_EVIDENCE),
        },
    )
    assert result.outcome == item_nav.PASSIVE_TERMINAL_UNMEASURED
    assert result.estimate is None and result.latest_frame == b"f2"


# Cancellation checks in order: 1 before wait, 2 after wait, 3 after capture 1,
# 4 before second wait, 5 after second wait, 6 after capture 2.
@pytest.mark.parametrize(
    "stop_at_check, captures, sleeps",
    [(1, 0, 0), (2, 0, 1), (3, 1, 1), (4, 1, 1), (5, 1, 2), (6, 2, 2)],
)
def test_helper_cancellation_is_cooperative(
    monkeypatch, stop_at_check, captures, sleeps, _seeded_and_no_real_sleep
):
    checks = {"n": 0}

    def cancelled():
        checks["n"] += 1
        if checks["n"] >= stop_at_check:
            raise ActionCancelled("stop")

    estimator_calls = []
    monkeypatch.setattr(
        item_nav, "compare_profile_identity", lambda *a, **k: _Verdict(True)
    )
    monkeypatch.setattr(
        item_nav, "estimate_shift", lambda *a, **k: estimator_calls.append(a)
    )
    driver = _StubDriver([b"f1", b"f2"])
    with pytest.raises(ActionCancelled):
        item_nav._passive_resample(
            driver,
            before=b"orig",
            reference_identity="IDENT",
            identity_match_max_dist=1.5,
            content_band=(1, 2),
            trust_window_px=900,
            cancelled=cancelled,
        )
    assert checks["n"] == stop_at_check
    assert driver.captures == captures
    assert len(_seeded_and_no_real_sleep) == sleeps
    assert estimator_calls == []


def test_ineligibility_rules():
    no_cons = _shift(frameshift.SHIFT_NO_CONSENSUS)
    clean = {"forward": {"status": "no_consensus", "saturated": False}, "reverse": None}
    assert item_nav._passive_ineligibility(no_cons, clean) is None
    assert (
        item_nav._passive_ineligibility(_shift(frameshift.SHIFT_NO_EVIDENCE), clean)
        is None
    )
    assert (
        item_nav._passive_ineligibility(
            no_cons,
            {
                "forward": clean["forward"],
                "reverse": {"status": "beyond_window", "saturated": False},
            },
        )
        is not None
    )
    assert (
        item_nav._passive_ineligibility(
            no_cons,
            {"forward": {"status": "no_consensus", "saturated": True}, "reverse": None},
        )
        is not None
    )
    assert (
        item_nav._passive_ineligibility(
            dataclasses.replace(no_cons, saturated=True), clean
        )
        is not None
    )
    assert (
        item_nav._passive_ineligibility(
            _shift(frameshift.SHIFT_BEYOND_WINDOW, consensus=-1400), clean
        )
        is not None
    )
