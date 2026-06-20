"""Unit tests for the budget / cost-control feature (runs without the SDK or a GPU)."""
from operation_love.costing import (
    CostTracker,
    ModelPricing,
    Usage,
    cost_usd,
    is_out_of_credit,
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
    assert t.remaining() == 0.0


def test_tracker_no_budget():
    t = CostTracker({"m": ModelPricing(input=1.0, output=1.0)}, run_budget_usd=None)
    t.record("m", Usage(input_tokens=5_000_000))
    assert t.remaining() is None
    assert not t.budget_reached()


def test_out_of_credit_detection():
    class Err400(Exception):
        type = "invalid_request_error"
        message = "Your credit balance is too low to access the Anthropic API"

    class Err403(Exception):
        type = "billing_error"
        message = "billing problem"

    assert is_out_of_credit(Err400())
    assert is_out_of_credit(Err403())
    assert not is_out_of_credit(ValueError("some unrelated error"))


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
