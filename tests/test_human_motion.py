"""Tests for the human touch-kinematics model — properties, not pixels.

Stochastic cases use a seeded random.Random; a couple use a FakeRng that pins
uniform/gauss to isolate the deterministic path (curvature) from the noise.
"""
import math
import random

from operation_love import human_motion as hm
from operation_love.human_motion import (
    fitts_duration_s,
    plan_swipe,
    plan_tap,
    think_time_s,
)


class FakeRng:
    """uniform -> max of range; gauss -> mean (no noise); lognormvariate -> median."""
    def uniform(self, a, b):
        return b

    def gauss(self, mu, sigma):
        return mu

    def lognormvariate(self, mu, sigma):
        return math.exp(mu)


def test_fitts_duration_monotonic_in_distance():
    short = fitts_duration_s(100, 180)
    long = fitts_duration_s(1500, 180)
    assert long > short
    assert short > 0


def test_velocity_profile_peaks_before_midstroke():
    w = hm._velocity_weights(60)
    peak = max(range(len(w)), key=lambda i: w[i])
    assert peak / len(w) < 0.5            # asymmetric: fastest in the first half
    assert all(v > 0 for v in w)


def test_swipe_endpoints_are_exact():
    s = plan_swipe(540, 1700, 540, 700, rng=random.Random(1))
    assert (s[0].x, s[0].y, s[0].tip) == (540, 1700, True)      # exact start
    assert (s[-2].x, s[-2].y, s[-2].tip) == (540, 700, True)    # exact end of drag
    assert s[-1].tip is False                                   # release appended
    assert s[-1].pressure == 0.0


def test_swipe_pressure_size_time_envelopes():
    s = plan_swipe(200, 1600, 900, 800, rng=random.Random(2))
    drag = s[:-1]
    assert all(hm._PRESSURE_FLOOR - 1e-9 <= p.pressure <= hm._PRESSURE_PEAK + 1e-9 for p in drag)
    assert all(0.0 <= p.size <= hm._SIZE_PEAK + 1e-9 for p in drag)
    # tip-down boundary samples ramp from/to the digitizer FLOOR, never 0 (real fingers don't
    # register a contact at zero pressure); only the tip=False release (s[-1]) carries 0.
    assert drag[0].pressure == drag[-1].pressure              # one sampled contact floor/stroke
    assert hm._PRESSURE_FLOOR <= drag[0].pressure <= 0.28
    assert max(p.pressure for p in drag) > 0.5                   # peaks mid-stroke
    ts = [p.t for p in s]
    assert all(b > a for a, b in zip(ts, ts[1:]))               # time strictly increasing


def test_no_tip_down_sample_has_zero_pressure():
    # Audit fix: a capacitive digitizer never reports a finger DOWN at exactly 0 pressure —
    # every in-contact (tip=True) sample sits at/above the floor; only the release is 0.
    for plan in (plan_swipe(540, 1700, 540, 700, rng=random.Random(11)),
                 plan_tap(540, 1200, rng=random.Random(12))):
        down = [p for p in plan if p.tip]
        release = [p for p in plan if not p.tip]
        assert down and all(p.pressure >= hm._PRESSURE_FLOOR - 1e-9 for p in down)
        assert release and all(p.pressure == 0.0 for p in release)


def test_swipe_sample_count_tracks_fitts_duration():
    x1, y1, x2, y2 = 100, 100, 100, 1300
    s = plan_swipe(x1, y1, x2, y2, rng=random.Random(3))
    # Fitts is the central tendency; each stroke samples bounded duration and report cadence.
    median_n = round(fitts_duration_s(math.hypot(x2 - x1, y2 - y1)) * hm.REPORT_HZ)
    assert median_n * 0.6 < len(s) - 2 < median_n * 1.5          # n+1 drag + release


def test_swipe_duration_scale_creates_a_quick_flick_without_changing_endpoints():
    ordinary = plan_swipe(540, 2100, 540, 300, rng=random.Random(19))
    flick = plan_swipe(540, 2100, 540, 300, duration_scale=160 / 450,
                       rng=random.Random(19))

    assert (flick[0].x, flick[0].y, flick[-2].x, flick[-2].y) == (540, 2100, 540, 300)
    assert flick[-2].t < ordinary[-2].t * 0.40
    assert len(flick) < len(ordinary)


def test_swipe_path_is_curved_not_straight():
    # FakeRng -> max perpendicular control offset, zero jitter: pure deterministic bow.
    # Measure MAX off-axis deviation (the bow peaks near the arc-length midpoint, which
    # the velocity reparameterization maps to an earlier sample index than len//2).
    s = plan_swipe(540, 1700, 540, 700, curve=0.2, rng=FakeRng())
    assert max(abs(p.x - 540) for p in s) > 50                  # bows well off the straight line


def test_swipe_zero_length_does_not_crash():
    s = plan_swipe(500, 500, 500, 500, rng=random.Random(4))
    assert s[0].x == 500 and s[0].y == 500
    assert s[-1].tip is False


def test_tap_lands_near_target_with_release():
    s = plan_tap(540, 1200, rng=random.Random(5))
    assert all(abs(p.x - 540) < 16 and abs(p.y - 1200) < 16 for p in s)  # bounded contact patch
    assert s[-1].tip is False and s[-1].pressure == 0.0
    assert (s[-1].x, s[-1].y) == (s[-2].x, s[-2].y)              # release stays at contact
    assert max(p.pressure for p in s) > 0.3                     # a real pressure pulse


def test_same_nominal_swipes_vary_kinematics_but_keep_safe_endpoints():
    # Remove curvature/tremor here so velocity-envelope variation cannot be
    # accidentally supplied by unrelated path noise.
    plans = [plan_swipe(540, 1700, 540, 700, curve=0.0, jitter_px=0.0,
                        rng=random.Random(seed)) for seed in range(80)]
    assert all((p[0].x, p[0].y, p[-2].x, p[-2].y) == (540, 1700, 540, 700)
               for p in plans)
    durations = {round(p[-2].t, 4) for p in plans}
    counts = {len(p) for p in plans}
    def velocity_shape(plan):
        speeds = [math.hypot(b.x - a.x, b.y - a.y) / (b.t - a.t)
                  for a, b in zip(plan[:-2], plan[1:-1])]
        peak = max(range(len(speeds)), key=speeds.__getitem__)
        half = [i for i, speed in enumerate(speeds) if speed >= max(speeds) * 0.5]
        return round(peak / len(speeds), 2), round((half[-1] - half[0]) / len(speeds), 2)
    shapes = {velocity_shape(p) for p in plans}
    def pressure_shape(plan):
        drag = plan[:-1]
        floor, peak = min(s.pressure for s in drag), max(s.pressure for s in drag)
        span = peak - floor or 1.0
        return tuple(round((drag[round(i * (len(drag) - 1) / 10)].pressure - floor) / span, 2)
                     for i in range(11))
    pressure_shapes = {pressure_shape(p) for p in plans}
    assert len(durations) > 20 and len(counts) > 8
    assert len(shapes) > 8 and len(pressure_shapes) > 20


def test_taps_have_varied_bounded_aim_and_pressure_envelopes():
    plans = [plan_tap(540, 1200, rng=random.Random(seed)) for seed in range(80)]
    releases = {(round(p[-1].x), round(p[-1].y)) for p in plans}
    peaks = {round(max(s.pressure for s in p), 3) for p in plans}
    def pressure_shape(plan):
        drag = plan[:-1]
        floor, peak = min(s.pressure for s in drag), max(s.pressure for s in drag)
        span = peak - floor or 1.0
        return tuple(round((drag[round(i * (len(drag) - 1) / 10)].pressure - floor) / span, 2)
                     for i in range(11))
    shapes = {pressure_shape(p) for p in plans}
    assert len(releases) > 20 and len(peaks) > 20 and len(shapes) > 12
    assert all(abs(p[-1].x - 540) < 8 and abs(p[-1].y - 1200) < 8 for p in plans)
    assert all((p[-1].x, p[-1].y) == (p[-2].x, p[-2].y) and p[-1].pressure == 0.0
               for p in plans)


def test_think_time_like_is_faster_than_pass():
    r = random.Random(6)
    likes = [think_time_s("like", r) for _ in range(300)]
    passes = [think_time_s("pass", r) for _ in range(300)]
    assert sum(likes) / len(likes) < sum(passes) / len(passes)
    assert min(likes) >= 1.2 and min(passes) >= 1.8            # never below the shift floor


def test_deterministic_for_same_seed():
    a = plan_swipe(100, 200, 800, 1500, rng=random.Random(42))
    b = plan_swipe(100, 200, 800, 1500, rng=random.Random(42))
    assert a == b


def test_ou_tremor_is_bounded_and_correlated():
    r = random.Random(7)
    n = 500
    offs = hm._ou_series(n, 1.0 / hm.REPORT_HZ, hm._JITTER_PX, [1.0] * n, r)
    xs = [dx for dx, _ in offs]
    assert max(abs(v) for v in xs) < 15                        # bounded (stationary std ~2.2px)
    num = sum(a * b for a, b in zip(xs, xs[1:]))
    den = sum(v * v for v in xs) or 1.0
    assert num / den > 0.5                                     # strong lag-1 autocorrelation (not white)
