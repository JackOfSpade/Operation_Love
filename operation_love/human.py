"""Human-like timing: log-normal delay multipliers.

Ported from the infinite-canvas project's humanDelay.js. A standard-normal sample
is converted to a log-normal multiplier (sigma=0.22) and applied to an anchor
delay: usually near the anchor, sometimes shorter/longer, with an organic long
tail and never negative — more human than a flat uniform range.

Works in SECONDS (for time.sleep / threading.Event.wait), unlike the JS source
which used milliseconds.
"""
from __future__ import annotations

import math
import random
from numbers import Real

_SIGMA = 0.22


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return resolved


def _parameters(seconds: float, sigma: float) -> tuple[float, float]:
    return _finite_nonnegative("seconds", seconds), _finite_nonnegative("sigma", sigma)


def _scaled_delay(seconds: float, exponent: float) -> float:
    try:
        result = seconds * math.exp(exponent)
    except OverflowError as exc:
        raise ValueError("generated delay is not finite") from exc
    if not math.isfinite(result):
        raise ValueError("generated delay is not finite")
    return result


def human_delay(seconds: float, sigma: float = _SIGMA) -> float:
    """Anchor delay with log-normal spread — may be shorter OR longer; never negative.

    For normal pauses / between-swipe delays.
    """
    seconds, sigma = _parameters(seconds, sigma)
    return _scaled_delay(seconds, random.gauss(0.0, sigma))


def human_cooldown(seconds: float, sigma: float = _SIGMA) -> float:
    """Like human_delay, but never below `seconds` — for minimum waits / backoffs."""
    seconds, sigma = _parameters(seconds, sigma)
    return _scaled_delay(seconds, abs(random.gauss(0.0, sigma)))
