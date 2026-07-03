"""Smoke tests against the shipped config.yaml — catches typos and dead config keys
that unit tests with inline YAML strings would miss.

All three tests are offline (no GCP / no phone / no SDK). They intentionally do NOT
call validate() because that requires a BigQuery project_id + photo_bucket (which the
shipped yaml has for production but would fail in a clean CI env). They test the
structure and business-logic invariants that matter even before go-live.
"""
import pytest

from operation_love.config import load


@pytest.fixture(scope="module")
def cfg():
    return load("config.yaml")


def test_config_yaml_loads_without_error(cfg):
    """config.yaml parses and produces a valid Config object."""
    assert cfg.enabled_apps                        # at least one app
    assert cfg.mode in {"observe", "auto"}
    assert cfg.budget.run_budget_usd is not None   # a run budget is set


def test_opener_model_has_pricing_entry(cfg):
    """The model named in opener.model must have a pricing entry so spend tracking works."""
    model = cfg.opener.model
    assert model in cfg.budget.pricing, (
        f"opener.model={model!r} has no entry in budget.pricing. "
        f"Available: {sorted(cfg.budget.pricing.keys())}"
    )


def test_limits_are_positive(cfg):
    """Auto-mode caps must be positive integers so they actually cap something."""
    lim = cfg.limits
    if lim.get("max_per_run") is not None:
        assert lim["max_per_run"] > 0, "limits.max_per_run must be > 0"
    if lim.get("max_per_day") is not None:
        assert lim["max_per_day"] > 0, "limits.max_per_day must be > 0"
    if lim.get("max_likes_per_run") is not None:
        assert lim["max_likes_per_run"] > 0, "limits.max_likes_per_run must be > 0"
    if lim.get("target_like_ratio") is not None:
        ratio = lim["target_like_ratio"]
        assert 0 < ratio < 1, f"limits.target_like_ratio must be in (0, 1), got {ratio}"
