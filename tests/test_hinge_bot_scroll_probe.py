"""tools/hinge_bot_scroll_probe.py — fully mocked, no device.

Three things this test file exists to prove, matching the harness's own safety/measurement
contract:

  1. The tap-neutralising guard (`neutralize_unsafe_methods`) actually raises for every method
     it claims to neutralise, and leaves swipe/scroll alone.
  2. The tracking/chaining analysis (`build_pair_result`/`chain_items`) is correct against
     KNOWN deltas and KNOWN heart positions, including a deliberately large/unreliable jump
     that must be reported as a tracking failure rather than silently chained into a phantom
     item — the exact failure mode ops/OPENER-REDESIGN.md 5.10 measured under human scrolling.
  3. `main()` exits non-zero (never retries) when the driver raises, whether that happens at
     `open_session()` or at `current_profile()`.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

import tools.hinge_bot_scroll_probe as probe
from operation_love.drivers.hinge import HingeDriver


class _Cfg:
    apps = {"hinge": {"serial": "pixel"}}


# --- 1. the tap-neutralising guard --------------------------------------------------------

class _FakeTransport:
    """Records calls that get through; `keyevent` is deliberately absent, so the guard's
    hasattr-gated loop must skip it without error rather than assume every method exists."""

    def __init__(self):
        self.calls = []

    def tap(self, x, y):
        self.calls.append(("tap", x, y))

    def text(self, s):
        self.calls.append(("text", s))

    def swipe(self, *a):
        self.calls.append(("swipe", a))
        return "swiped"

    def scroll_up(self, *a, **k):
        self.calls.append(("scroll_up", a, k))
        return "scrolled"


def _drv_with_transports(adb, touch):
    d = HingeDriver(_Cfg())
    d._adb = adb
    d._touch = touch
    return d


def test_neutralize_raises_for_every_claimed_method_and_lists_them():
    adb = _FakeTransport()
    touch = _FakeTransport()   # a DIFFERENT object from adb (UhidTouch case)
    drv = _drv_with_transports(adb, touch)

    neutralized = probe.neutralize_unsafe_methods(drv)

    # Exactly the surfaces the docstring promises: driver decisions + both transports' tap/text
    # (keyevent absent on both fakes, same as production Adb/UhidTouch today).
    assert set(neutralized) == {
        "driver.like()", "driver.dislike()",
        "driver.adb.tap()", "driver.adb.text()",
        "driver.touch.tap()", "driver.touch.text()",
    }

    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.like("hello", 0)
    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.dislike()
    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.adb.tap(10, 20)
    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.adb.text("hi")
    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.touch.tap(10, 20)
    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.touch.text("hi")

    # scroll/swipe are the one approved action -- must be untouched, on BOTH transports.
    assert drv.adb.swipe(1, 2, 3, 4) == "swiped"
    assert drv.adb.scroll_up(0.5, 0.5) == "scrolled"
    assert drv.touch.swipe(1, 2, 3, 4) == "swiped"
    assert drv.touch.scroll_up(0.5, 0.5) == "scrolled"


def test_neutralize_does_not_double_report_when_touch_is_the_same_object_as_adb():
    # touch_backend: adb -- _make_touch() literally returns self._adb (hinge.py's own
    # _make_touch), so driver.touch IS driver.adb, same object. The guard must not claim two
    # neutralisations for one method.
    adb = _FakeTransport()
    drv = _drv_with_transports(adb, adb)

    neutralized = probe.neutralize_unsafe_methods(drv)

    assert neutralized.count("driver.adb.tap()") == 1
    assert not any(n.startswith("driver.touch.") for n in neutralized)
    with pytest.raises(probe.ScrollProbeGuardTripped):
        drv.touch.tap(1, 1)   # still guarded -- it's the same object as adb


# --- 2. tracking/chaining analysis, known deltas + known heart positions -----------------

_BAND_PX = (300, 2100)          # content_band_px used throughout
_TOL = 15.0
_MIN_RESPONSE = 0.5
_MAX_DELTA = 1800.0             # == band height, same derivation analyze_frames uses


def test_build_pair_result_matches_within_tolerance_and_flags_all_matched():
    hearts_a = [(500, 1000)]
    hearts_b = [(500, 850)]     # exactly at the predicted position (delta 150)
    pr = probe.build_pair_result(0, 1, hearts_a, hearts_b, delta_px=150.0, response=0.9,
                                 content_band_px=_BAND_PX, tolerance_px=_TOL,
                                 min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pr.reliable is True
    assert pr.matched == [(0, 0)]
    assert pr.unmatched_a == []
    assert pr.left_band_a == []
    assert pr.all_matched is True


def test_build_pair_result_distinguishes_left_band_from_a_genuine_miss():
    # heart 0 predicted off the top of the band (scrolled away -- expected, not a failure);
    # heart 1 predicted still in-band but nothing in frame B lands near it (a real miss).
    hearts_a = [(500, 310), (500, 1200)]
    hearts_b = [(500, 9999)]    # nowhere near either prediction
    pr = probe.build_pair_result(0, 1, hearts_a, hearts_b, delta_px=100.0, response=0.9,
                                 content_band_px=_BAND_PX, tolerance_px=_TOL,
                                 min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pr.reliable is True
    assert pr.matched == []
    assert pr.left_band_a == [0]     # 310 - 100 = 210 < r0 (300)
    assert pr.unmatched_a == [1]     # 1200 - 100 = 1100, well inside the band, but no match
    assert pr.all_matched is False


def test_build_pair_result_unreliable_on_low_response_alone():
    hearts_a = [(500, 1000)]
    hearts_b = [(500, 850)]
    pr = probe.build_pair_result(0, 1, hearts_a, hearts_b, delta_px=150.0, response=0.1,
                                 content_band_px=_BAND_PX, tolerance_px=_TOL,
                                 min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pr.reliable is False
    assert pr.matched == []          # refuses to match at all under a distrusted delta
    assert pr.all_matched is False


def test_build_pair_result_unreliable_on_implausible_delta_magnitude_alone():
    hearts_a = [(500, 1000)]
    hearts_b = [(500, 850)]
    pr = probe.build_pair_result(0, 1, hearts_a, hearts_b, delta_px=5000.0, response=0.95,
                                 content_band_px=_BAND_PX, tolerance_px=_TOL,
                                 min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pr.reliable is False
    assert pr.matched == []


def test_chain_items_recovers_distinct_items_across_a_clean_bounded_scroll():
    """3 frames, bounded 150px steps (well under _MAX_DELTA), one item present throughout and
    a second item entering at frame 1 -- both should chain cleanly to 2 distinct items, with
    no ambiguity and no tracking failures."""
    pair0 = probe.build_pair_result(
        0, 1, hearts_a=[(500, 1000)], hearts_b=[(500, 850), (500, 1900)],
        delta_px=150.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    pair1 = probe.build_pair_result(
        1, 2, hearts_a=[(500, 850), (500, 1900)], hearts_b=[(500, 700), (500, 1750)],
        delta_px=150.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)

    result = probe.chain_items([pair0, pair1])

    assert result.distinct_confirmed == 2
    assert result.ambiguous_new == 0
    assert result.orphaned_tracks == 0
    assert result.tracking_failures == []


def test_chain_items_reports_a_large_jump_as_a_tracking_failure_not_a_phantom():
    """The exact bug ops/OPENER-REDESIGN.md 5.10 found under human scrolling: a large jump
    that breaks the single-delta assumption must NOT be silently accepted as "item left, new
    item entered" (which is precisely how the naive method fabricated a spurious 10th item).
    One item, tracked cleanly for one pair, then a deliberately implausible jump. Chaining
    must report the jump as a tracking failure and hold the total at 1 confirmed item plus an
    explicitly ambiguous one -- never confidently 2."""
    pair0 = probe.build_pair_result(
        0, 1, hearts_a=[(500, 1000)], hearts_b=[(500, 850)],
        delta_px=150.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    # A jump far beyond anything a bounded bot read-scroll could produce (or, equivalently, a
    # very low phase-correlation response) -- either alone is enough to distrust this pair.
    pair1 = probe.build_pair_result(
        1, 2, hearts_a=[(500, 850)], hearts_b=[(500, 700)],
        delta_px=5000.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pair1.reliable is False   # sanity: this IS the scenario under test

    result = probe.chain_items([pair0, pair1])

    assert result.tracking_failures == [(1, 2)]
    assert result.distinct_confirmed == 1     # NOT 2 -- no phantom silently added
    assert result.ambiguous_new == 1          # the frame-2 heart is flagged, not counted
    assert result.orphaned_tracks == 1        # the frame-1 track's fate is left unresolved


def test_chain_items_does_not_promote_a_heart_born_behind_an_unreliable_pair():
    """Found 2026-09-02: the old `active = {}` reset after an unreliable pair dropped the new
    frame's hearts out of the tracking dict entirely, and `active.get(ai, True)` defaults a
    MISSING key to True (confirmed) -- so a heart that first appeared behind an unreliable pair
    (honestly `ambiguous_new`, never `distinct_confirmed`) got silently promoted to "confirmed"
    the moment the NEXT reliable pair matched it forward. If that promoted track then missed
    (pair 3), the miss was wrongly folded into `orphaned_tracks`, a statistic ChainResult's own
    docstring defines as counting only PREVIOUSLY-CONFIRMED tracks.

    Four frames, three pairs, deliberately isolating the two sources of orphaning so a
    regression shows up as a count, not just a crash:
      pair0 (0->1) UNRELIABLE: frame-0's one confirmed heart X is orphaned for real (its fate
        genuinely is unknown across an unreliable pair) -- that is 1 legitimate orphan no fix
        should remove. Frame 1's heart Y is `ambiguous_new`, not confirmed.
      pair1 (1->2) reliable: Y tracks forward to Z. Never re-added to distinct_confirmed either
        way -- matches never are -- but under the bug Z's active-entry becomes True (wrongly
        "confirmed") instead of staying False (still ambiguous).
      pair2 (2->3) reliable: Z genuinely misses (still predicted in-band, nothing matches).
        Buggy code: Z's wrongly-True entry counts this miss as a SECOND orphaned track (2
        total). Fixed code: Z's correctly-False entry means an already-ambiguous track that
        also misses is simply dropped (see the loop's own comment) -- orphaned stays at 1.
    """
    pair0 = probe.build_pair_result(
        0, 1, hearts_a=[(500, 1000)], hearts_b=[(500, 850)],
        delta_px=5000.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pair0.reliable is False   # sanity: this IS the unreliable pair under test

    pair1 = probe.build_pair_result(
        1, 2, hearts_a=[(500, 850)], hearts_b=[(500, 700)],
        delta_px=150.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pair1.reliable is True and pair1.matched == [(0, 0)]   # Y tracks forward to Z

    # predicted_y = 700 - 150 = 550, still well inside the band (r0=300) -- a genuine miss,
    # not a left-band exit -- and hearts_b is empty so nothing can match it.
    pair2 = probe.build_pair_result(
        2, 3, hearts_a=[(500, 700)], hearts_b=[],
        delta_px=150.0, response=0.9, content_band_px=_BAND_PX, tolerance_px=_TOL,
        min_response=_MIN_RESPONSE, max_delta_px=_MAX_DELTA)
    assert pair2.reliable is True and pair2.unmatched_a == [0] and pair2.left_band_a == []

    result = probe.chain_items([pair0, pair1, pair2])

    assert result.tracking_failures == [(0, 1)]
    assert result.distinct_confirmed == 1   # only frame-0's X ever confirmed
    assert result.ambiguous_new == 1        # Y, born behind the unreliable pair
    assert result.orphaned_tracks == 1      # X only -- NOT 2 (Z's miss must not be re-counted)


def test_estimate_vertical_delta_sign_and_magnitude(monkeypatch):
    """Real phase correlation (cv2.phaseCorrelate), not a mock, against two synthetic PNGs
    where the true shift is known by construction: frame_b's content_band rows are frame_a's,
    rolled down by exactly 20px (i.e. an ordinary forward/down read-scroll)."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(400, 300), dtype="uint8")
    shifted = np.zeros_like(base)
    shift = 20
    shifted[:-shift, :] = base[shift:, :]
    shifted[-shift:, :] = rng.integers(0, 255, size=(shift, 300), dtype="uint8")

    ok_a, buf_a = cv2.imencode(".png", base)
    ok_b, buf_b = cv2.imencode(".png", shifted)
    assert ok_a and ok_b

    delta_px, response = probe.estimate_vertical_delta(
        buf_a.tobytes(), buf_b.tobytes(), content_band=(0.0, 1.0))

    assert delta_px == pytest.approx(shift, abs=2.0)
    assert response > 0.5


# --- 3. main() exits non-zero when the driver raises, and never retries ------------------

class _OpenRaisesDriver:
    def open_session(self):
        raise RuntimeError("no ADB device connected")

    def close(self):  # pragma: no cover -- must not even be reached
        raise AssertionError("close() must not be called when open_session() itself raised")


def test_main_exits_nonzero_when_open_session_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.cfg_mod, "load", lambda path: _Cfg())
    monkeypatch.setattr(probe, "HingeDriver", lambda cfg: _OpenRaisesDriver())

    with pytest.raises(SystemExit) as exc:
        probe.main(["--out", str(tmp_path / "out")])
    assert exc.value.code != 0


class _FakeOpenAdb:
    def screen_size(self):
        return (1080, 2400)


class _CaptureRaisesDriver:
    def __init__(self):
        self.adb = _FakeOpenAdb()
        self.touch = self.adb
        self.content_band = (0.125, 0.875)
        self.closed = False
        # Defaults matching the shipped config.yaml, so a test that never passes an override
        # flag can assert main() left these exactly at "whatever the driver was built with".
        self.read_scroll_frac = 0.55
        self.scroll_captures = 12
        self._capture_scroll_ledger: list[tuple[float, float]] = []
        # Recorded by current_profile() below, at the moment it's called -- lets a test prove
        # main() applied a --read-scroll-frac/--scroll-captures override to the DRIVER'S OWN
        # attribute (the one _sample_read_scroll/_capture_limit_for_profile actually read)
        # BEFORE the capture path ever ran, not just that the CLI parsed the flag.
        self.observed_read_scroll_frac = None
        self.observed_scroll_captures = None

    def open_session(self):
        pass

    def close(self):
        self.closed = True

    def blocked_reason(self):
        return None

    def like(self, *a, **k):  # pragma: no cover -- guarded before it could ever fire
        raise AssertionError("like() must never be reachable")

    def dislike(self):  # pragma: no cover
        raise AssertionError("dislike() must never be reachable")

    def _scroll_to_top(self, should_stop=None):
        # install_ledger_capture wraps this attribute unconditionally (before
        # current_profile() is ever called), so it must exist even on a fake driver whose
        # current_profile() raises before this would ever actually run.
        return True

    def current_profile(self, **kw):
        self.observed_read_scroll_frac = self.read_scroll_frac
        self.observed_scroll_captures = self.scroll_captures
        raise RuntimeError("UnlocatedControlError-style stand-in: the deck is unreadable")


def test_main_exits_nonzero_when_current_profile_raises(monkeypatch, tmp_path, capsys):
    drv = _CaptureRaisesDriver()
    monkeypatch.setattr(probe.cfg_mod, "load", lambda path: _Cfg())
    monkeypatch.setattr(probe, "HingeDriver", lambda cfg: drv)
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    with pytest.raises(SystemExit) as exc:
        probe.main(["--out", str(tmp_path / "out")])
    assert exc.value.code != 0
    assert drv.closed is True   # cleanup still ran even though the read failed

    out = capsys.readouterr()
    combined = out.out + out.err
    assert "current_profile" in combined
    assert "WILL NOT" in out.out


# --- 4. --read-scroll-frac / --scroll-captures: validated, and honoured on the driver ------

def _load_must_not_be_called(_path):
    raise AssertionError("cfg_mod.load must not run once flag validation has already failed "
                          "-- a bad flag must never get far enough to touch config.yaml or "
                          "the device")


@pytest.mark.parametrize("bad_frac", ["0.05", "0.80", "nan", "inf", "-1"])
def test_main_rejects_read_scroll_frac_outside_hinge_pys_own_bounds(monkeypatch, tmp_path, bad_frac):
    # hinge._READ_SCROLL_FRAC_MIN/MAX are 0.10/0.75 -- 0.05 and 0.80 sit just outside either
    # edge, nan/inf/-1 are the "not even a sane fraction" cases math.isfinite exists to catch.
    monkeypatch.setattr(probe.cfg_mod, "load", _load_must_not_be_called)

    with pytest.raises(SystemExit) as exc:
        probe.main(["--out", str(tmp_path / "out"), "--read-scroll-frac", bad_frac])
    assert exc.value.code != 0


@pytest.mark.parametrize("bad_n", ["0", "-1", str(probe._MAX_SCROLL_CAPTURES + 1)])
def test_main_rejects_scroll_captures_outside_its_sanity_ceiling(monkeypatch, tmp_path, bad_n):
    monkeypatch.setattr(probe.cfg_mod, "load", _load_must_not_be_called)

    with pytest.raises(SystemExit) as exc:
        probe.main(["--out", str(tmp_path / "out"), "--scroll-captures", bad_n])
    assert exc.value.code != 0


def test_main_rejects_scroll_captures_non_integer(tmp_path):
    # argparse's own type=int parsing rejects this (exit code 2) before this tool's own
    # validation even runs -- still a fail-loud non-zero exit, which is what matters here.
    with pytest.raises(SystemExit) as exc:
        probe.main(["--out", str(tmp_path / "out"), "--scroll-captures", "12.5"])
    assert exc.value.code != 0


@pytest.mark.parametrize("frac", [probe.hinge._READ_SCROLL_FRAC_MIN, 0.16,
                                  probe.hinge._READ_SCROLL_FRAC_MAX])
def test_main_accepts_read_scroll_frac_at_and_inside_the_bounds(monkeypatch, tmp_path, frac):
    drv = _CaptureRaisesDriver()
    monkeypatch.setattr(probe.cfg_mod, "load", lambda path: _Cfg())
    monkeypatch.setattr(probe, "HingeDriver", lambda cfg: drv)
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    with pytest.raises(SystemExit):   # current_profile() still raises -- that's fine, expected
        probe.main(["--out", str(tmp_path / "out"), "--read-scroll-frac", str(frac)])

    assert drv.observed_read_scroll_frac == pytest.approx(frac)


def test_main_applies_both_overrides_to_the_drivers_own_attributes_before_capturing(
        monkeypatch, tmp_path, capsys):
    drv = _CaptureRaisesDriver()
    monkeypatch.setattr(probe.cfg_mod, "load", lambda path: _Cfg())
    monkeypatch.setattr(probe, "HingeDriver", lambda cfg: drv)
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    with pytest.raises(SystemExit):
        probe.main(["--out", str(tmp_path / "out"), "--read-scroll-frac", "0.16",
                    "--scroll-captures", "48"])

    # The fake driver's current_profile() recorded self.read_scroll_frac/self.scroll_captures
    # at call time -- i.e. AFTER main() applied the override and BEFORE the (fake) capture
    # path ran, proving the override reached the exact attribute _sample_read_scroll /
    # _capture_limit_for_profile read, not a copy main() kept to itself.
    assert drv.observed_read_scroll_frac == 0.16
    assert drv.observed_scroll_captures == 48

    out = capsys.readouterr().out
    assert "Effective read_scroll_frac: 0.16 (--read-scroll-frac override)" in out
    assert "Effective scroll_captures: 48 (--scroll-captures override)" in out


def test_main_leaves_driver_defaults_untouched_when_flags_omitted(monkeypatch, tmp_path, capsys):
    drv = _CaptureRaisesDriver()
    monkeypatch.setattr(probe.cfg_mod, "load", lambda path: _Cfg())
    monkeypatch.setattr(probe, "HingeDriver", lambda cfg: drv)
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    with pytest.raises(SystemExit):
        probe.main(["--out", str(tmp_path / "out")])

    # Neither flag was passed -- the driver's own config.yaml-derived defaults (set in
    # _CaptureRaisesDriver.__init__, mirroring HingeDriver's real construction) must survive
    # untouched.
    assert drv.observed_read_scroll_frac == 0.55
    assert drv.observed_scroll_captures == 12

    out = capsys.readouterr().out
    assert "Effective read_scroll_frac: 0.55 (from config.yaml)" in out
    assert "Effective scroll_captures: 12 (from config.yaml)" in out


# --- 5. install_ledger_capture: snapshot happens BEFORE _scroll_to_top's own reset ---------

class _LedgerDriver:
    """A minimal stand-in for the one thing install_ledger_capture touches:
    `_capture_scroll_ledger` and a `_scroll_to_top` method. The fake `_scroll_to_top` mimics
    the real one's own documented behavior (hinge.py ~2055) of resetting the ledger to `[]`
    once it runs, which is exactly the race this wrapper exists to win."""

    def __init__(self, ledger):
        self._capture_scroll_ledger = list(ledger)
        self.calls: list = []

    def _scroll_to_top(self, should_stop=None):
        self.calls.append(should_stop)
        self._capture_scroll_ledger = []   # the real method's own trailing reset
        return True


def test_install_ledger_capture_snapshots_before_the_wrapped_reset():
    drv = _LedgerDriver([(0.55, 0.5), (0.55, 0.5)])
    snapshots = probe.install_ledger_capture(drv)
    assert snapshots == []   # nothing captured yet -- only installed, not yet called

    result = drv._scroll_to_top()

    assert result is True                              # delegates through unchanged
    assert drv.calls == [None]                          # original still ran, same arguments
    assert snapshots == [[(0.55, 0.5), (0.55, 0.5)]]     # captured BEFORE the reset ran
    assert drv._capture_scroll_ledger == []              # the real reset still happened


def test_install_ledger_capture_passes_should_stop_through():
    drv = _LedgerDriver([(0.16, 0.5)])
    probe.install_ledger_capture(drv)
    def sentinel():
        return False

    drv._scroll_to_top(sentinel)

    assert drv.calls == [sentinel]


def test_install_ledger_capture_last_snapshot_is_the_post_capture_one():
    # Mirrors current_profile()'s real shape: an early _ensure_session_top() call with an
    # empty pre-session ledger, then the real per-profile ledger right before the call this
    # tool actually cares about.
    drv = _LedgerDriver([])
    snapshots = probe.install_ledger_capture(drv)

    drv._scroll_to_top()                                    # pre-capture: ledger already []
    drv._capture_scroll_ledger = [(0.16, 0.5)] * 40          # the real per-profile ledger
    drv._scroll_to_top()                                    # post-capture

    assert len(snapshots) == 2
    assert snapshots[0] == []
    assert snapshots[-1] == [(0.16, 0.5)] * 40


# --- 6. compute_heart_spacings_px / _step_spacing_stats: pure geometry, no cv2 needed -------

def test_compute_heart_spacings_px_only_counts_gaps_within_one_frame():
    hearts_per_frame = [
        [(500, 300), (500, 1400)],                 # one frame, two hearts -> one gap
        [(500, 900)],                               # single heart -> no gap
        [(500, 200), (500, 1300), (500, 2400)],      # two hearts -> two gaps
    ]
    assert probe.compute_heart_spacings_px(hearts_per_frame) == [1100, 1100, 1100]


def test_compute_heart_spacings_px_sorts_before_diffing():
    # detect_hearts's own contract is top-to-bottom already, but the function must not trust
    # that blindly -- an out-of-order input must still produce a positive gap.
    hearts_per_frame = [[(500, 1400), (500, 300)]]
    assert probe.compute_heart_spacings_px(hearts_per_frame) == [1100]


def test_compute_heart_spacings_px_empty_when_no_frame_ever_shows_two_hearts():
    hearts_per_frame = [[(500, 300)], [], [(500, 900)]]
    assert probe.compute_heart_spacings_px(hearts_per_frame) == []


def test_step_spacing_stats_computes_ratio_from_unfiltered_deltas():
    median_step, median_spacing, ratio = probe._step_spacing_stats(
        [1100.0, 1200.0, -1150.0], [1100.0, 1150.0])
    assert median_step == pytest.approx(1150.0)      # median(|1100|,|1200|,|-1150|)
    assert median_spacing == pytest.approx(1125.0)
    assert ratio == pytest.approx(1150.0 / 1125.0)


def test_step_spacing_stats_none_when_either_input_is_missing():
    assert probe._step_spacing_stats([], []) == (None, None, None)
    step, spacing, ratio = probe._step_spacing_stats([100.0], [])
    assert step == pytest.approx(100.0)
    assert spacing is None
    assert ratio is None


# --- 7. render_verdict: names ALIASING as root cause when step ~= spacing, not otherwise ----

def _report(*, frame_count=5, median_step, median_spacing, ratio,
            tracking_failures=(), orphaned_tracks=0, ambiguous_new=0, match_rate=1.0):
    chain = probe.ChainResult(distinct_confirmed=1, ambiguous_new=ambiguous_new,
                              orphaned_tracks=orphaned_tracks,
                              tracking_failures=list(tracking_failures))
    return probe.AnalysisReport(
        frame_count=frame_count, hearts_per_frame=[], pair_results=[], scroll_deltas_px=[],
        match_rate=match_rate, chain=chain, heart_spacings_px=[],
        median_scroll_step_px=median_step, median_heart_spacing_px=median_spacing,
        step_spacing_ratio=ratio)


def test_render_verdict_names_aliasing_as_root_cause_when_ratio_near_one():
    # The measured real-world regime this flag exists to move away from: step ~1150px against
    # spacing ~1100px, ratio ~1.05 -- squarely inside _ALIASING_RATIO_LOW.._ALIASING_RATIO_HIGH.
    report = _report(median_step=1150.0, median_spacing=1100.0, ratio=1150.0 / 1100.0,
                     tracking_failures=[(3, 4)], orphaned_tracks=1, match_rate=0.8)

    verdict = probe.render_verdict(report)

    assert "ALIASING" in verdict
    assert "No better delta estimator" in verdict
    assert "1150.0px" in verdict and "1100.0px" in verdict
    # the symptom list must still be present -- aliasing is the ROOT CAUSE framing, not a
    # replacement for what was actually observed.
    assert "tracking" in verdict.lower()


def test_render_verdict_does_not_call_it_aliasing_when_step_much_smaller_than_spacing():
    # The hypothesis this tool's flags exist to test: read_scroll_frac ~0.16 against the
    # shipped 0.55 baseline scales the step down by roughly the same factor, landing well
    # outside the aliasing band (ratio ~0.3 here, vs. the ~1.05 case above).
    report = _report(median_step=350.0, median_spacing=1100.0, ratio=350.0 / 1100.0,
                     tracking_failures=[(3, 4)], orphaned_tracks=1, match_rate=0.8)

    verdict = probe.render_verdict(report)

    assert "ALIASING" not in verdict
    assert "flagged as tracking" in verdict   # the ordinary symptom reporting still fires


def test_render_verdict_reliable_run_still_states_the_measured_ratio():
    report = _report(median_step=350.0, median_spacing=1100.0, ratio=350.0 / 1100.0)

    verdict = probe.render_verdict(report)

    assert verdict.startswith("RELIABLE")
    assert "ratio 0.32" in verdict


def test_render_verdict_reports_ratio_unmeasurable_without_calling_it_aliasing():
    report = _report(median_step=1150.0, median_spacing=None, ratio=None,
                     tracking_failures=[(3, 4)], match_rate=0.8)

    verdict = probe.render_verdict(report)

    assert "NOT MEASURABLE" in verdict
    assert "ALIASING" not in verdict


# --- 8. save_capture / print_report: the pre-reset ledger, reported honestly ----------------

class _FakeAdbForSave:
    def screen_size(self):
        return (1080, 2400)


class _FakeDriverForSave:
    def __init__(self):
        self.adb = _FakeAdbForSave()
        self.content_band = (0.125, 0.875)
        self.read_scroll_frac = 0.16
        self.scroll_captures = 48


def test_save_capture_records_the_given_pre_reset_ledger(tmp_path):
    drv = _FakeDriverForSave()
    ledger = [(0.16, 0.51), (0.17, 0.49)]

    manifest = probe.save_capture([b"frame1", b"frame2"], {"app": "hinge"}, drv,
                                  tmp_path / "out", driver_scroll_ledger=ledger)

    assert manifest["driver_scroll_ledger"] == [{"frac": 0.16, "x_frac": 0.51},
                                                 {"frac": 0.17, "x_frac": 0.49}]
    assert manifest["driver_scroll_ledger_note"] is None
    assert manifest["read_scroll_frac"] == 0.16
    assert manifest["scroll_captures"] == 48
    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert on_disk == manifest
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "out").stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600
                   for path in (tmp_path / "out").iterdir())


def test_save_capture_notes_honestly_when_the_ledger_could_not_be_captured(tmp_path):
    drv = _FakeDriverForSave()

    manifest = probe.save_capture([b"frame1"], {}, drv, tmp_path / "out",
                                  driver_scroll_ledger=None)

    assert manifest["driver_scroll_ledger"] is None
    assert manifest["driver_scroll_ledger_note"] is not None
    assert "NOT CAPTURED" in manifest["driver_scroll_ledger_note"]


def test_save_capture_distinguishes_a_genuinely_empty_ledger_from_not_captured(tmp_path):
    drv = _FakeDriverForSave()

    manifest = probe.save_capture([b"frame1"], {}, drv, tmp_path / "out",
                                  driver_scroll_ledger=[])

    assert manifest["driver_scroll_ledger"] == []          # recorded, and genuinely empty
    assert manifest["driver_scroll_ledger_note"] is None    # NOT the "not captured" case


def test_print_report_flags_a_missing_ledger_as_a_measurement_gap_not_zero(capsys):
    report = _report(median_step=None, median_spacing=None, ratio=None, frame_count=1)

    probe.print_report(report, driver_scroll_ledger=None)

    out = capsys.readouterr().out
    assert "NOT CAPTURED" in out
    assert "not evidence" in out


def test_print_report_prints_ledger_fracs_when_present(capsys):
    report = _report(median_step=1150.0, median_spacing=1100.0, ratio=1150.0 / 1100.0)

    probe.print_report(report, driver_scroll_ledger=[(0.55, 0.5), (0.55, 0.5)])

    out = capsys.readouterr().out
    assert "2 read-scroll(s) issued" in out
    assert "frac=0.550" in out
    assert "step/spacing: median scroll step 1150.0px" in out
