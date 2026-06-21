"""Rate limits — autonomous-mode safety caps to look human / avoid bans.

Applies to AUTO mode only (observe mode = your own manual swipes, never capped).
Per-app caps on swipes per run and per day, plus a per-run cap on LIKES — the
right-swipe is the action anti-bot systems weight most, so keeping likes well
under the total swipe count holds the right-swipe ratio in a human range.
Checked before each autonomous action.
"""
from __future__ import annotations


class RateLimiter:
    def __init__(self, max_per_run: int | None = None, max_per_day: int | None = None,
                 max_likes_per_run: int | None = None):
        self.max_per_run = max_per_run
        self.max_per_day = max_per_day
        self.max_likes_per_run = max_likes_per_run

    def allow(self, acted_this_run: int, acted_today: int) -> bool:
        if self.max_per_run is not None and acted_this_run >= self.max_per_run:
            return False
        if self.max_per_day is not None and acted_today >= self.max_per_day:
            return False
        return True

    def allow_like(self, liked_this_run: int) -> bool:
        """False once this run has hit its like budget. The worker then stops the
        run rather than mislabeling a wanted-like as a pass (which would corrupt
        the taste model)."""
        if self.max_likes_per_run is not None and liked_this_run >= self.max_likes_per_run:
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.max_per_run is not None:
            parts.append(f"{self.max_per_run}/run")
        if self.max_per_day is not None:
            parts.append(f"{self.max_per_day}/day")
        if self.max_likes_per_run is not None:
            parts.append(f"{self.max_likes_per_run} likes/run")
        return ", ".join(parts) or "unlimited"
