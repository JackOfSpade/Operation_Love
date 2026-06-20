"""Rate limits — autonomous-mode safety caps to look human / avoid bans.

Applies to AUTO mode only (observe mode = your own manual swipes, never capped).
Per-app caps on swipes per run and per day; checked before each autonomous action.
"""
from __future__ import annotations


class RateLimiter:
    def __init__(self, max_per_run: int | None = None, max_per_day: int | None = None):
        self.max_per_run = max_per_run
        self.max_per_day = max_per_day

    def allow(self, acted_this_run: int, acted_today: int) -> bool:
        if self.max_per_run is not None and acted_this_run >= self.max_per_run:
            return False
        if self.max_per_day is not None and acted_today >= self.max_per_day:
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.max_per_run is not None:
            parts.append(f"{self.max_per_run}/run")
        if self.max_per_day is not None:
            parts.append(f"{self.max_per_day}/day")
        return ", ".join(parts) or "unlimited"
