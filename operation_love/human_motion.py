"""Human-like touch kinematics — pure, transport-agnostic gesture synthesis.

Generates the per-sample stream (t, x, y, pressure, size, tip) for taps and swipes
from motor-control models, calibrated to REAL Pixel 7a `getevent` data measured
2026-06-27 (raw pressure varies ~0.20-1.00 of 255, mean ~0.69; contact-major
~116-466; report rate ~183 Hz; ~25 samples/typical swipe). The UHID transport
consumes pressure+size; the degraded `motionevent` transport uses x,y only.

No device access / no I/O here — fully unit-testable. Determinism: every public
function accepts `rng=random.Random(seed)`.

Models (see ops research + memory):
- Fitts (Shannon) for movement duration.
- Cubic Bezier path with a perpendicular control-point perturbation (curved, not a
  straight line), reparameterized by ARC LENGTH so spatial spacing follows speed.
- Asymmetric (lognormal) velocity profile — peak before midstroke, longer decel —
  replacing a symmetric ease.
- Ornstein-Uhlenbeck correlated tremor with signal-dependent (speed-scaled)
  amplitude (Harris-Wolpert), bounded in pixels — replacing white-noise jitter.
- Beta-function pressure/size ramp over the stroke (rise, plateau, decay).
- Tap: lognormal dwell + damped micro-slip + a pressure pulse.
- Between-actions: per-decision shifted-lognormal "think time" (like faster than pass),
  reproducing the measured Hinge baseline (~3.2 s like / ~6.9 s pass).
"""
from __future__ import annotations

import math
import random
from collections import namedtuple

# A normalized touch sample. pressure/size in 0..1 (transport scales to raw range);
# tip True while the finger is down.
TouchSample = namedtuple("TouchSample", "t x y pressure size tip")

REPORT_HZ = 180.0                 # match the measured ~183 Hz digitizer rate

# Fitts (Shannon): MT = a + b*log2(D/W + 1), seconds. Mobile-touch typical.
_FITTS_A, _FITTS_B = 0.11, 0.17
_DEFAULT_WIDTH_PX = 180.0

# Lognormal tangential-velocity profile: peak at ~35% of MT, right-skewed.
_VEL_PEAK_FRAC, _VEL_SIGMA = 0.35, 0.18

# Ornstein-Uhlenbeck tremor: mean-reversion rate (1/s) and pixel amplitude. The
# stationary std is signal-dependent: jitter_px scaled by 0.4 + 0.6*speed_norm.
_OU_THETA = 22.0
_JITTER_PX = 2.2

# Beta-function pressure/size ramp: (u^a)*((1-u)^b), normalized so the peak == 1.
_P_ALPHA, _P_BETA = 1.25, 0.85
_PRESSURE_PEAK = 0.95             # normalized; mean over stroke ~0.6-0.7 (matches 0.69)
_PRESSURE_FLOOR = 0.22           # normalized ~56/255: a capacitive digitizer only reports a
                                  # contact ABOVE its detection threshold, so a tip-down frame
                                  # is never 0 (matches the measured ~0.20 raw min). Only the
                                  # explicit tip=False release carries 0 pressure.
_SIZE_PEAK = 0.55                # normalized; transport maps to raw contact-major

# Tap dwell (down->up): lognormal, median ~130 ms.
_TAP_DWELL_MEDIAN_S, _TAP_DWELL_SIGMA = 0.13, 0.25
_TAP_MICROSLIP_PX = 2.5          # finger-pad slide on impact

# Per-decision think time: (shift_s, mu, sigma) for shifted-lognormal.
_THINK = {
    "like": (1.2, 0.65, 0.35),   # mean ~3.2 s
    "pass": (1.8, 1.45, 0.42),   # mean ~6.9 s
}


def _rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random


def fitts_duration_s(distance_px: float, width_px: float = _DEFAULT_WIDTH_PX,
                     a: float = _FITTS_A, b: float = _FITTS_B) -> float:
    """Shannon-formulation movement time for a gesture of the given amplitude."""
    w = max(1.0, float(width_px))
    return a + b * math.log2(max(0.0, float(distance_px)) / w + 1.0)


def _cubic(p0, c1, c2, p3, u):
    mt = 1.0 - u
    x = mt**3 * p0[0] + 3 * mt * mt * u * c1[0] + 3 * mt * u * u * c2[0] + u**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt * mt * u * c1[1] + 3 * mt * u * u * c2[1] + u**3 * p3[1]
    return x, y


def _control_points(p0, p3, curve, rng):
    """Two control points at 1/3 and 2/3 along the chord, pushed perpendicular by a
    random fraction of chord length so the path bows like a real finger."""
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length            # unit perpendicular
    off = rng.uniform(-curve, curve) * length
    c1 = (p0[0] + dx / 3.0 + px * off, p0[1] + dy / 3.0 + py * off)
    c2 = (p0[0] + 2.0 * dx / 3.0 + px * off, p0[1] + 2.0 * dy / 3.0 + py * off)
    return c1, c2


def _arclen_table(p0, c1, c2, p3, segments=96):
    """Cumulative arc length at evenly-spaced u, for arc-length -> u inversion."""
    us = [i / segments for i in range(segments + 1)]
    pts = [_cubic(p0, c1, c2, p3, u) for u in us]
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return us, cum


def _u_at_arclen(frac, us, cum):
    """Invert the arc-length table: fraction of total length -> Bezier parameter u."""
    total = cum[-1] or 1.0
    target = max(0.0, min(1.0, frac)) * total
    # linear scan + interpolate (tables are small)
    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg = cum[i] - cum[i - 1] or 1.0
            t = (target - cum[i - 1]) / seg
            return us[i - 1] + (us[i] - us[i - 1]) * t
    return 1.0


def _velocity_weights(n, peak_frac=_VEL_PEAK_FRAC, sigma=_VEL_SIGMA):
    """Per-step relative speed from a lognormal profile peaking at peak_frac of the
    stroke. Returns n weights (one per inter-sample step), strictly positive."""
    mu = math.log(max(1e-6, peak_frac)) + sigma * sigma   # so the mode is at peak_frac
    w = []
    for k in range(n):
        u = (k + 0.5) / n                                  # mid-step fraction in (0,1)
        val = math.exp(-((math.log(u) - mu) ** 2) / (2 * sigma * sigma)) / u
        w.append(val)
    return w


def _ou_series(n, dt, jitter_px, speed_norm, rng):
    """Bivariate Ornstein-Uhlenbeck tremor: mean-reverting, temporally correlated,
    amplitude scaled by per-step normalized speed. Returns n+1 (dx,dy) offsets."""
    a = math.exp(-_OU_THETA * dt)
    ex = ey = 0.0
    out = [(0.0, 0.0)]
    for k in range(n):
        s = jitter_px * (0.4 + 0.6 * speed_norm[min(k, len(speed_norm) - 1)])
        inc = s * math.sqrt(max(0.0, 1.0 - a * a))
        ex = a * ex + rng.gauss(0.0, inc)
        ey = a * ey + rng.gauss(0.0, inc)
        out.append((ex, ey))
    return out


def _beta_ramp(u, peak, alpha=_P_ALPHA, beta=_P_BETA):
    """Pressure/size envelope over stroke fraction u in [0,1], scaled so peak==`peak`."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    mode = alpha / (alpha + beta)
    norm = (mode ** alpha) * ((1.0 - mode) ** beta)        # max of the unscaled beta
    return peak * (u ** alpha) * ((1.0 - u) ** beta) / (norm or 1.0)


def _down_pressure(env, peak=_PRESSURE_PEAK, floor=_PRESSURE_FLOOR):
    """In-contact pressure for a tip-down sample: ramp from the digitizer floor up to peak
    and back (never 0 while the finger is down), where env is the [0,1] beta envelope. The
    boundary samples (env==0) land on `floor`, not 0 — a finger-down report at exactly 0
    pressure is physically impossible on a capacitive sensor. The tip=False release is 0."""
    return floor + (peak - floor) * env


def plan_swipe(x1, y1, x2, y2, *, width_px=_DEFAULT_WIDTH_PX, curve=0.12,
               jitter_px=_JITTER_PX, hz=REPORT_HZ, rng=None):
    """Synthesize a human swipe: arc-length-reparameterized curved path, asymmetric
    velocity, correlated tremor, beta pressure/size. Returns a list of TouchSample
    (tip=True for the drag, a final tip=False release). Endpoints land on target."""
    r = _rng(rng)
    dist = math.hypot(x2 - x1, y2 - y1)
    dur = fitts_duration_s(dist, width_px)
    n = max(2, int(round(dur * hz)))                       # inter-sample steps
    dt = dur / n
    c1, c2 = _control_points((x1, y1), (x2, y2), curve, r)
    us, cum = _arclen_table((x1, y1), c1, c2, (x2, y2))
    w = _velocity_weights(n)
    tot = sum(w) or 1.0
    cumw, acc = [], 0.0
    for wi in w:                                           # cumulative arc-length fraction
        acc += wi
        cumw.append(acc / tot)
    speed_norm = [wi / max(w) for wi in w]
    noise = _ou_series(n, dt, jitter_px, speed_norm, r)
    samples = []
    for k in range(n + 1):
        frac = 0.0 if k == 0 else cumw[k - 1]
        u = _u_at_arclen(frac, us, cum)
        bx, by = _cubic((x1, y1), c1, c2, (x2, y2), u)
        nx, ny = noise[k]
        if k == 0:
            bx, by, nx, ny = x1, y1, 0.0, 0.0              # exact start, no jitter
        elif k == n:
            bx, by, nx, ny = x2, y2, 0.0, 0.0              # exact end, no jitter
        env = _beta_ramp(k / n, 1.0)
        samples.append(TouchSample(
            t=k * dt, x=bx + nx, y=by + ny,
            pressure=_down_pressure(env), size=_SIZE_PEAK * env, tip=True))
    last = samples[-1]
    samples.append(TouchSample(t=last.t + dt, x=last.x, y=last.y,
                               pressure=0.0, size=0.0, tip=False))   # release
    return samples


def plan_tap(x, y, *, hz=REPORT_HZ, jitter_px=_TAP_MICROSLIP_PX, rng=None):
    """Synthesize a human tap: a lognormal dwell with a damped micro-slip from the
    impact point and a beta pressure/size pulse. Final sample is the release."""
    r = _rng(rng)
    dwell = r.lognormvariate(math.log(_TAP_DWELL_MEDIAN_S), _TAP_DWELL_SIGMA)
    dwell = max(0.04, min(0.35, dwell))
    n = max(2, int(round(dwell * hz)))
    dt = dwell / n
    d0x, d0y = r.uniform(-1, 1) * jitter_px, r.uniform(-1, 1) * jitter_px
    lam, ftremor = 35.0, 10.0
    samples = []
    for k in range(n + 1):
        tt = k * dt
        decay = math.exp(-lam * tt) * math.cos(2 * math.pi * ftremor * tt)
        sx = x + d0x * decay + r.gauss(0.0, 0.15)
        sy = y + d0y * decay + r.gauss(0.0, 0.15)
        env = _beta_ramp(k / n, 1.0) if 0 < k < n else 0.0
        samples.append(TouchSample(t=tt, x=sx, y=sy,
                                   pressure=_down_pressure(env),
                                   size=_SIZE_PEAK * env, tip=True))
    samples.append(TouchSample(t=n * dt + dt, x=x, y=y, pressure=0.0, size=0.0, tip=False))
    return samples


def think_time_s(decision: str, rng=None) -> float:
    """Per-decision 'think time' between profiles (shifted-lognormal). 'like' is a
    faster reaction than 'pass', matching the measured Hinge dwell asymmetry."""
    r = _rng(rng)
    shift, mu, sigma = _THINK.get(decision, _THINK["pass"])
    return shift + math.exp(r.gauss(mu, sigma))
