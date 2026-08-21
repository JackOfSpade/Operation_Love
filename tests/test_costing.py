"""Unit tests for the budget / cost-control feature (runs without the SDK or a GPU)."""
import math

import pytest

from operation_love.costing import (
    CostTracker,
    ModelPricing,
    Usage,
    cost_usd,
)


def test_cost_usd_opus():
    p = ModelPricing(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(u, p) == 30.0  # $5 in + $25 out


def test_cost_includes_cache():
    p = ModelPricing(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
    u = Usage(cache_read_input_tokens=2_000_000, cache_creation_input_tokens=1_000_000)
    assert cost_usd(u, p) == 0.5 * 2 + 6.25 * 1  # 7.25


def test_tracker_budget_reached():
    pricing = {"m": ModelPricing(input=10.0, output=10.0)}
    t = CostTracker(pricing, run_budget_usd=0.02)
    assert not t.budget_reached()
    t.record("m", Usage(input_tokens=1_000_000))  # $10 -> way over $0.02
    assert t.budget_reached()


def test_tracker_no_budget():
    t = CostTracker({"m": ModelPricing(input=1.0, output=1.0)}, run_budget_usd=None)
    t.record("m", Usage(input_tokens=5_000_000))
    assert not t.budget_reached()


def test_record_returns_call_cost():
    pricing = {"m": ModelPricing(input=10.0, output=10.0)}
    t = CostTracker(pricing, run_budget_usd=None)
    c = t.record("m", Usage(input_tokens=1_000_000))  # $10 at 10.0/MTok
    assert c == 10.0
    assert t.run_spend_usd == c  # accumulation == returned delta


def test_record_raises_keyerror_for_a_model_with_no_pricing_entry():
    # record() prices against the EXACT configured model id -- GeminiOpener._parse always
    # passes the model it was asked for (requested_model), never a provider-echoed serving
    # revision (see GeminiOpener._parse's docstring / CostTracker.record's docstring), so
    # there is no dated/aliased id to normalize here, unlike the old Anthropic-shaped
    # dated-id (e.g. "-20251001" suffix) tolerance this replaced.
    pricing = {"gemini-3.6-flash": ModelPricing(input=10.0, output=10.0)}
    t = CostTracker(pricing, run_budget_usd=None)
    t.record("gemini-3.6-flash", Usage(input_tokens=1_000_000))  # exact match: fine
    try:
        t.record("gemini-3.6-flash-20251001", Usage(input_tokens=1_000_000))  # no stripping
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for a model id with no exact pricing entry")


def test_usage_from_gemini_separates_cached_prompt_and_thought_tokens():
    class Metadata:
        prompt_token_count = 1_000
        cached_content_token_count = 250
        candidates_token_count = 80
        thoughts_token_count = 120

    usage = Usage.from_gemini(Metadata())
    assert usage.input_tokens == 750
    assert usage.cache_read_input_tokens == 250
    assert usage.output_tokens == 200


@pytest.mark.parametrize("field,value", [
    ("input", -1),
    ("output", math.nan),
    ("cache_read", math.inf),
    ("cache_write", True),
    ("input", "5"),
])
def test_model_pricing_rejects_values_that_can_poison_or_reduce_spend(field, value):
    values = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        ModelPricing(**values)


def test_model_pricing_allows_exact_zero_free_tier_rates():
    assert ModelPricing(input=0, output=0, cache_read=0, cache_write=0).input == 0


def test_cost_boundaries_reject_enormous_integers_without_overflow_leaks():
    huge = 10 ** 10_000
    with pytest.raises(ValueError, match="finite"):
        ModelPricing(input=huge, output=0)
    with pytest.raises(ValueError, match="finite"):
        CostTracker({}, run_budget_usd=huge)
    with pytest.raises(ValueError, match="finite"):
        cost_usd(Usage(input_tokens=huge), ModelPricing(input=1, output=0))


def test_zero_price_avoids_overflow_for_a_mathematically_zero_component():
    assert cost_usd(
        Usage(input_tokens=10 ** 10_000), ModelPricing(input=0, output=0)) == 0


@pytest.mark.parametrize("field", [
    "prompt_token_count", "cached_content_token_count",
    "candidates_token_count", "thoughts_token_count",
])
def test_usage_from_gemini_rejects_negative_counters(field):
    metadata = type("Metadata", (), {field: -1})()
    with pytest.raises(ValueError, match=field):
        Usage.from_gemini(metadata)


def test_usage_from_gemini_rejects_cached_prompt_underflow():
    class Metadata:
        prompt_token_count = 10
        cached_content_token_count = 11

    with pytest.raises(ValueError, match="cannot exceed"):
        Usage.from_gemini(Metadata())


def test_usage_from_gemini_treats_missing_or_none_counters_as_zero():
    class Metadata:
        prompt_token_count = None

    assert Usage.from_gemini(Metadata()) == Usage()


@pytest.mark.parametrize("value", [True, 1.5, "5", [], math.inf])
def test_usage_from_gemini_rejects_malformed_nonnumeric_counters(value):
    class Metadata:
        prompt_token_count = value

    with pytest.raises(ValueError, match="prompt_token_count"):
        Usage.from_gemini(Metadata())


@pytest.mark.parametrize("field,value", [
    ("input_tokens", -1),
    ("output_tokens", True),
    ("cache_read_input_tokens", 1.5),
    ("cache_creation_input_tokens", "2"),
])
def test_usage_direct_construction_requires_exact_nonnegative_integers(field, value):
    values = {"input_tokens": 0, "output_tokens": 0,
              "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        Usage(**values)


@pytest.mark.parametrize("budget", [True, -1, math.nan, math.inf, "5"])
def test_cost_tracker_rejects_invalid_direct_budget_values(budget):
    with pytest.raises(ValueError, match="run_budget_usd"):
        CostTracker({"m": ModelPricing(input=0, output=0)}, budget)


@pytest.mark.parametrize("budget", [None, 0, 5.0])
def test_cost_tracker_accepts_none_zero_and_finite_nonnegative_budget(budget):
    tracker = CostTracker({"m": ModelPricing(input=0, output=0)}, budget)
    assert tracker.run_budget_usd == budget


@pytest.mark.parametrize("value", [None, [], 1, "pricing"])
def test_model_pricing_from_dict_rejects_nonmapping_inputs(value):
    with pytest.raises(ValueError, match="mapping"):
        ModelPricing.from_dict(value)


@pytest.mark.parametrize("values", [
    {"output": 1},
    {"input": 1},
    {"input": 1, "output": 1, "cache": 0},
])
def test_model_pricing_from_dict_rejects_missing_or_unknown_fields(values):
    with pytest.raises(ValueError, match="ModelPricing mapping"):
        ModelPricing.from_dict(values)


@pytest.mark.parametrize("values", [
    {1: 0, "input": 1, "output": 1},
    {"": 0, "input": 1, "output": 1},
])
def test_model_pricing_from_dict_rejects_nonstring_or_empty_keys(values):
    with pytest.raises(ValueError, match="keys"):
        ModelPricing.from_dict(values)


@pytest.mark.parametrize("pricing", [None, [], {"m": 1}, {"": ModelPricing(0, 0)},
                                     {True: ModelPricing(0, 0)}])
def test_cost_tracker_rejects_malformed_pricing_mapping(pricing):
    with pytest.raises(ValueError, match="pricing"):
        CostTracker(pricing, None)


def test_cost_tracker_copies_pricing_mapping_at_construction():
    pricing = {"m": ModelPricing(input=1, output=1)}
    tracker = CostTracker(pricing, None)
    pricing.clear()
    assert tracker.record("m", Usage(input_tokens=1_000_000)) == 1


@pytest.mark.parametrize("model,usage", [(None, Usage()), ("", Usage()), ("m", None)])
def test_cost_tracker_record_rejects_malformed_runtime_inputs(model, usage):
    tracker = CostTracker({"m": ModelPricing(input=0, output=0)}, None)
    with pytest.raises(ValueError):
        tracker.record(model, usage)


def test_cost_usd_rejects_non_domain_objects():
    with pytest.raises(ValueError, match="usage"):
        cost_usd(None, ModelPricing(input=0, output=0))
    with pytest.raises(ValueError, match="pricing"):
        cost_usd(Usage(), None)


def test_cost_tracker_does_not_publish_an_infinite_cumulative_total():
    tracker = CostTracker({"m": ModelPricing(input=1e308, output=0)}, None)
    tracker.run_spend_usd = 1e308
    with pytest.raises(ValueError, match="Cumulative"):
        tracker.record("m", Usage(input_tokens=1_000_000))
    assert tracker.run_spend_usd == 1e308
    assert tracker.calls == 0
