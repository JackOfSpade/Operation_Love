"""Human-like timing: log-normal delay multipliers.

Ported from the infinite-canvas project's humanDelay.js. A Box-Muller normal
sample is converted to a log-normal multiplier (sigma=0.22) and applied to an
anchor delay: usually near the anchor, sometimes shorter/longer, with an organic
long tail and never negative — more human than a flat uniform range.

Works in SECONDS (for time.sleep / threading.Event.wait), unlike the JS source
which used milliseconds.
"""
from __future__ import annotations

import math
import random
import sys

_SIGMA = 0.22


def _normal() -> float:
    u1 = random.random() or sys.float_info.epsilon
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def human_delay(seconds: float, sigma: float = _SIGMA) -> float:
    """Anchor delay with log-normal spread — may be shorter OR longer; never negative.

    For normal pauses / between-swipe delays.
    """
    return seconds * math.exp(_normal() * sigma)


def human_cooldown(seconds: float, sigma: float = _SIGMA) -> float:
    """Like human_delay, but never below `seconds` — for minimum waits / backoffs."""
    return seconds * math.exp(abs(_normal()) * sigma)
