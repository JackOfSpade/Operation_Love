"""Unit tests for the budget / cost-control feature (runs without the SDK or a GPU)."""
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
