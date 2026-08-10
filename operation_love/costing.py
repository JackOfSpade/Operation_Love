"""Provider-neutral client-side spend tracking and opener budget enforcement.

Providers return token usage but do not offer a portable remaining-credit API, so
we calculate configured prices locally and enforce a per-run cap.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

MILLION = 1_000_000


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
    def from_gemini(cls, usage) -> "Usage":
        """Normalize Gemini ``usage_metadata`` into the persistent cost schema.

        Gemini reports cached prompt tokens as a subset of ``prompt_token_count``.
        The normal input bucket therefore excludes cached tokens, while candidates and
        thoughts are both output/billed-generation tokens.  The latter is essential for
        thinking-capable Gemini models: omitting it would understate the run budget.
        """
        def g(name: str) -> int:
            return int(getattr(usage, name, 0) or 0)
        cached = g("cached_content_token_count")
        prompt = g("prompt_token_count")
        return cls(
            input_tokens=max(0, prompt - cached),
            output_tokens=g("candidates_token_count") + g("thoughts_token_count"),
            cache_read_input_tokens=cached,
        )


def cost_usd(usage: Usage, p: ModelPricing) -> float:
    return (
        usage.input_tokens / MILLION * p.input
        + usage.output_tokens / MILLION * p.output
        + usage.cache_read_input_tokens / MILLION * p.cache_read
        + usage.cache_creation_input_tokens / MILLION * p.cache_write
    )


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

    def budget_reached(self) -> bool:
        if self.run_budget_usd is None:
            return False
        with self._lock:
            return self.run_spend_usd >= self.run_budget_usd

    def record(self, model: str, usage: Usage) -> float:
        """Add a call's cost to the running total and return that cost.

        ``model`` is always the exact configured model id here, never a provider-echoed
        serving revision: GeminiOpener._parse deliberately prices against the model it was
        asked for (``requested_model``), not Gemini's optional ``modelVersion`` field, which
        can be an opaque revision string with no entry in budget.pricing. So a plain lookup
        is enough -- there is no dated/aliased id to normalize here.
        """
        p = self.pricing.get(model)
        if p is None:
            raise KeyError(f"No pricing configured for model {model!r}")
        c = cost_usd(usage, p)
        with self._lock:
            self.run_spend_usd += c
            self.calls += 1
        return c
