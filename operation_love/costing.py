"""Client-side spend tracking and per-run budget enforcement for opener calls.

Anthropic exposes NO public endpoint to read your remaining prepaid credit
balance, so we compute exact cost from the ``usage`` returned on every Messages
API response and enforce a per-run cap from config. We also detect the
out-of-credit/billing error so the bot degrades gracefully instead of crashing.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

MILLION = 1_000_000

# The Messages API echoes back the model id it actually served, which for some models is
# the dated full id (e.g. "claude-haiku-4-5-20251001") while config.yaml's pricing table
# is keyed by the bare alias ("claude-haiku-4-5"). Strip a trailing -YYYYMMDD so a dated
# id still resolves to its alias's pricing instead of crashing the worker on a KeyError.
_MODEL_DATE_SUFFIX = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1,000,000 tokens."""
    input: float
    output: float
    cache_read: float = 0.0   # ~0.1x input
    cache_write: float = 0.0  # 5-minute ephemeral write, ~1.25x input

    @classmethod
    def from_dict(cls, d: dict) -> "ModelPricing":
        return cls(
            input=float(d["input"]),
            output=float(d["output"]),
            cache_read=float(d.get("cache_read", 0.0)),
            cache_write=float(d.get("cache_write", 0.0)),
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage) -> "Usage":
        """Build from an Anthropic response.usage object (or any with these attrs)."""
        def g(name: str) -> int:
            return int(getattr(usage, name, 0) or 0)
        return cls(
            input_tokens=g("input_tokens"),
            output_tokens=g("output_tokens"),
            cache_read_input_tokens=g("cache_read_input_tokens"),
            cache_creation_input_tokens=g("cache_creation_input_tokens"),
        )


def cost_usd(usage: Usage, p: ModelPricing) -> float:
    return (
        usage.input_tokens / MILLION * p.input
        + usage.output_tokens / MILLION * p.output
        + usage.cache_read_input_tokens / MILLION * p.cache_read
        + usage.cache_creation_input_tokens / MILLION * p.cache_write
    )


class BudgetExceeded(Exception):
    """Raised/used to signal the per-run spend cap has been reached."""


class CostTracker:
    """Cumulative opener spend for one run against an optional cap.

    Thread-safe: one shared tracker enforces a single GLOBAL budget across all
    app workers (Bumble + Hinge run concurrently but draw on one cap).
    """

    def __init__(self, pricing: dict[str, ModelPricing], run_budget_usd: float | None):
        self.pricing = pricing
        self.run_budget_usd = run_budget_usd
        self.run_spend_usd: float = 0.0
        self.calls: int = 0
        self._lock = threading.Lock()

    def remaining(self) -> float | None:
        if self.run_budget_usd is None:
            return None
        with self._lock:
            return max(0.0, self.run_budget_usd - self.run_spend_usd)

    def budget_reached(self) -> bool:
        if self.run_budget_usd is None:
            return False
        with self._lock:
            return self.run_spend_usd >= self.run_budget_usd

    def record(self, model: str, usage: Usage) -> float:
        """Add a call's cost to the running total and return that cost."""
        p = self.pricing.get(model) or self.pricing.get(_MODEL_DATE_SUFFIX.sub("", model))
        if p is None:
            raise KeyError(f"No pricing configured for model {model!r}")
        c = cost_usd(usage, p)
        with self._lock:
            self.run_spend_usd += c
            self.calls += 1
        return c


def is_out_of_credit(exc: Exception) -> bool:
    """True if ``exc`` is Anthropic's out-of-credit / billing error.

    It surfaces as a 400 invalid_request_error whose message contains
    "credit balance is too low", and/or a 403 with error type "billing_error".
    """
    if getattr(exc, "type", None) == "billing_error":
        return True
    msg = (getattr(exc, "message", "") or str(exc) or "").lower()
    return "credit balance is too low" in msg
