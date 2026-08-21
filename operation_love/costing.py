"""Provider-neutral client-side spend tracking and opener budget enforcement.

Providers return token usage but do not offer a portable remaining-credit API, so
we calculate configured prices locally and enforce a per-run cap.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

MILLION = 1_000_000


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe_value_repr(value: object) -> str:
    try:
        return repr(value)
    except ValueError:
        if isinstance(value, int):
            return f"<integer with {value.bit_length()} bits>"
        return f"<{type(value).__name__}>"


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1,000,000 tokens."""
    input: float
    output: float
    cache_read: float = 0.0   # ~0.1x input
    cache_write: float = 0.0  # 5-minute ephemeral write, ~1.25x input

    def __post_init__(self) -> None:
        for name in ("input", "output", "cache_read", "cache_write"):
            value = getattr(self, name)
            if not _is_finite_real(value) or value < 0:
                raise ValueError(
                    f"ModelPricing.{name} must be a finite nonnegative number "
                    f"(got {_safe_value_repr(value)})")

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ModelPricing":
        """Build pricing from its public mapping representation.

        Config loading performs the same checks with config-path context, but this
        factory is also used directly by callers and must not turn a malformed
        object into a raw subscription/attribute error or silently discard a typo.
        """
        if not isinstance(d, Mapping):
            raise ValueError("ModelPricing mapping must be a mapping")
        if any(not isinstance(key, str) or not key.strip() for key in d):
            raise ValueError("ModelPricing mapping keys must be non-empty strings")
        allowed = {"input", "output", "cache_read", "cache_write"}
        unknown = set(d) - allowed
        missing = {"input", "output"} - set(d)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            raise ValueError(
                "ModelPricing mapping must contain input/output and only supported "
                f"keys ({'; '.join(details)})")
        return cls(
            input=d["input"],
            output=d["output"],
            cache_read=d.get("cache_read", 0.0),
            cache_write=d.get("cache_write", 0.0),
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                     "cache_creation_input_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Usage.{name} must be a nonnegative integer "
                                 f"(got {value!r})")

    @classmethod
    def from_gemini(cls, usage) -> "Usage":
        """Normalize Gemini ``usage_metadata`` into the persistent cost schema.

        Gemini reports cached prompt tokens as a subset of ``prompt_token_count``.
        The normal input bucket therefore excludes cached tokens, while candidates and
        thoughts are both output/billed-generation tokens.  The latter is essential for
        thinking-capable Gemini models: omitting it would understate the run budget.
        """
        def g(name: str) -> int:
            value = getattr(usage, name, 0)
            if value is None:
                return 0
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Gemini usage {name} must be an integer (got {value!r})")
            if value < 0:
                raise ValueError(
                    f"Gemini usage {name} must be nonnegative (got {value!r})")
            return value
        cached = g("cached_content_token_count")
        prompt = g("prompt_token_count")
        if cached > prompt:
            raise ValueError(
                "Gemini usage cached_content_token_count cannot exceed "
                f"prompt_token_count (got {cached} > {prompt})")
        return cls(
            input_tokens=prompt - cached,
            output_tokens=g("candidates_token_count") + g("thoughts_token_count"),
            cache_read_input_tokens=cached,
        )


def cost_usd(usage: Usage, p: ModelPricing) -> float:
    if not isinstance(usage, Usage):
        raise ValueError("usage must be a Usage instance")
    if not isinstance(p, ModelPricing):
        raise ValueError("pricing must be a ModelPricing instance")

    def component(tokens: int, price: float) -> float:
        if tokens == 0 or price == 0:
            return 0.0
        try:
            amount = tokens / MILLION * price
        except OverflowError as exc:
            raise ValueError("Computed model cost must be finite") from exc
        if not math.isfinite(amount):
            raise ValueError("Computed model cost must be finite")
        return amount

    parts = (
        component(usage.input_tokens, p.input),
        component(usage.output_tokens, p.output),
        component(usage.cache_read_input_tokens, p.cache_read),
        component(usage.cache_creation_input_tokens, p.cache_write),
    )
    try:
        total = sum(parts)
    except OverflowError as exc:
        raise ValueError("Computed model cost must be finite") from exc
    if not math.isfinite(total):
        raise ValueError("Computed model cost must be finite")
    return total


class CostTracker:
    """Cumulative opener spend for one run against an optional cap.

    Thread-safe: one shared tracker enforces a single GLOBAL budget across all
    app workers (Bumble + Hinge run concurrently but draw on one cap).
    """

    def __init__(self, pricing: Mapping[str, ModelPricing], run_budget_usd: float | None):
        if not isinstance(pricing, Mapping):
            raise ValueError("pricing must be a mapping of model ids to ModelPricing")
        normalized_pricing: dict[str, ModelPricing] = {}
        for model, model_pricing in pricing.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("pricing model ids must be non-empty strings")
            if not isinstance(model_pricing, ModelPricing):
                raise ValueError(
                    f"pricing[{model!r}] must be a ModelPricing instance")
            normalized_pricing[model] = model_pricing
        if (run_budget_usd is not None
                and (not _is_finite_real(run_budget_usd) or run_budget_usd < 0)):
            raise ValueError("run_budget_usd must be None or a finite nonnegative number "
                             f"(got {_safe_value_repr(run_budget_usd)})")
        # Keep the tracker invariant stable if the caller later mutates its mapping.
        self.pricing = normalized_pricing
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
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(usage, Usage):
            raise ValueError("usage must be a Usage instance")
        p = self.pricing.get(model)
        if p is None:
            raise KeyError(f"No pricing configured for model {model!r}")
        c = cost_usd(usage, p)
        with self._lock:
            new_spend = self.run_spend_usd + c
            if not math.isfinite(new_spend):
                raise ValueError("Cumulative model cost must be finite")
            self.run_spend_usd = new_spend
            self.calls += 1
        return c
