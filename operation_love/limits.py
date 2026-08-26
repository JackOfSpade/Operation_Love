"""Optional device-driven volume and like-ratio controls.

Per-app caps on swipes per run and per day, plus a per-run cap on LIKES apply
to both AUTO and Hub-reviewed Training.  ``target_like_ratio`` is AUTO-only:
it shapes model decisions and must never rewrite a human Training choice.  The
right-swipe is the action anti-bot systems weight most, so keeping likes well
under the total swipe count holds the right-swipe ratio in a human range.
Every field is optional; the shipped configuration leaves them unset. When configured,
they are checked before each autonomous action.

``target_like_ratio`` semantics (read this before enabling it): it is a SOFT,
score-aware shaper, not a hard cap — the hard cap on like volume is the separate
``max_likes_per_run``, which this never touches. Once the running like-rate reaches
the target, ``allow_like_ratio`` only vetoes a like that is MARGINAL against the
ranker's own like threshold (see its docstring for the worked numbers); a like the
ranker is clearly confident about is let through even though that means the ratio
briefly overshoots. An audit of an earlier, score-blind version of this shaper ran it
against real scores [0.80, 0.99, 0.81, 0.82] at target_like_ratio=0.5 and found it threw
away the model's *strongest* call (0.99) purely because of arrival order, while keeping
a weaker 0.80 — the most anti-"purely model-driven" outcome the shaper could produce.
This module intentionally cannot look ahead at future cards' scores to fix that in
general (a real run cannot un-swipe an earlier profile); the score-aware rule below
instead makes sure THIS call, on its own, never sacrifices a clearly strong like to
chase the ratio.
"""
from __future__ import annotations

import math
from numbers import Real


def _finite_real_as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


class RateLimiter:
    # A like scoring at least this many probability points above the ranker's own
    # like_threshold is "clearly strong" and always survives the ratio ceiling (see
    # allow_like_ratio). Chosen well above AutoSessionPolicy.max_threshold_lift's cap
    # (0.045, interaction.py) so the two score-aware shapers never double-count: any
    # like that only survived AutoSessionPolicy's contextual demotion because of ITS
    # lift is still, at most, base_threshold + 0.045 -- comfortably inside this 0.15
    # "marginal" band -- so it stays eligible for the ratio ceiling exactly as if
    # AutoSessionPolicy hadn't touched it, and only a like that is strong independent
    # of that lift is ever exempted here.
    _STRONG_LIKE_MARGIN = 0.15

    def __init__(self, max_per_run: int | None = None, max_per_day: int | None = None,
                 max_likes_per_run: int | None = None,
                 target_like_ratio: float | None = None):
        for name, value in (
                ("max_per_run", max_per_run),
                ("max_per_day", max_per_day),
                ("max_likes_per_run", max_likes_per_run)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        ratio = (
            _finite_real_as_float(target_like_ratio)
            if target_like_ratio is not None else None
        )
        if target_like_ratio is not None and (ratio is None or not 0.0 < ratio < 1.0):
            raise ValueError("target_like_ratio must be a finite number in (0, 1) or None")
        self.max_per_run = max_per_run
        self.max_per_day = max_per_day
        self.max_likes_per_run = max_likes_per_run
        self.target_like_ratio = ratio

    @staticmethod
    def _count(name: str, value: int) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    def allow(self, acted_this_run: int, acted_today: int) -> bool:
        self._count("acted_this_run", acted_this_run)
        self._count("acted_today", acted_today)
        if self.max_per_run is not None and acted_this_run >= self.max_per_run:
            return False
        if self.max_per_day is not None and acted_today >= self.max_per_day:
            return False
        return True

    def allow_like(self, liked_this_run: int) -> bool:
        """False once this run has hit its like budget. The worker then stops the
        run rather than mislabeling a wanted-like as a pass (which would corrupt
        the taste model)."""
        self._count("liked_this_run", liked_this_run)
        if self.max_likes_per_run is not None and liked_this_run >= self.max_likes_per_run:
            return False
        return True

    def allow_like_ratio(self, liked: int, acted: int, *, score: float | None = None,
                          like_threshold: float | None = None) -> bool:
        """False when the running like-rate is at/above target_like_ratio AND this
        specific like is only marginally over the ranker's own like threshold.

        The worker demotes a vetoed like to a pass instead of halting the run — so the
        ratio stays human-scale without cutting the session short. Below the ceiling
        this is always True regardless of score, exactly like the original shaper.

        ``score``/``like_threshold`` are keyword-only and optional: a caller with no
        per-decision score context gets the old score-blind ceiling (every like is
        vetoed once the ceiling is hit) rather than an exception. When both are
        supplied, a like scoring at least ``_STRONG_LIKE_MARGIN`` above
        ``like_threshold`` is judged "clearly strong" and is let through — worked
        example at like_threshold=0.5: a score of 0.51 is only 0.01 over the line
        (marginal, stays vetoable), while 0.90 is 0.40 over — comfortably past the
        0.15 margin — and always survives the ceiling. See the module docstring for
        why this can only ever look at the score in hand, never at other cards'.
        """
        self._count("liked", liked)
        self._count("acted", acted)
        if liked > acted:
            raise ValueError("liked cannot exceed acted")
        if self.target_like_ratio is None or acted == 0:
            return True
        if liked / acted < self.target_like_ratio:
            return True
        if score is None or like_threshold is None:
            return False
        score_value = _finite_real_as_float(score)
        threshold_value = _finite_real_as_float(like_threshold)
        if score_value is None or threshold_value is None:
            raise ValueError("score and like_threshold must be finite real numbers")
        return (score_value - threshold_value) >= self._STRONG_LIKE_MARGIN

    def describe(self) -> str:
        parts = []
        if self.max_per_run is not None:
            parts.append(f"{self.max_per_run}/run")
        if self.max_per_day is not None:
            parts.append(f"{self.max_per_day}/day")
        if self.max_likes_per_run is not None:
            parts.append(f"{self.max_likes_per_run} likes/run")
        if self.target_like_ratio is not None:
            parts.append(f"ratio≤{self.target_like_ratio:.0%}")
        return ", ".join(parts) or "unlimited"
