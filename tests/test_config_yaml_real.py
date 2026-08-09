"""Smoke tests against the shipped config.yaml — catches typos and dead config keys
that unit tests with inline YAML strings would miss.

All tests here are offline (no GCP / no phone / no SDK).

These used to skip validate() entirely, on the stated grounds that it "requires a
BigQuery project_id + photo_bucket ... would fail in a clean CI env". That premise was
wrong: validate() only checks those keys are present, non-empty strings — it opens no
connection and needs no credentials. The consequence was a real hole: every rule
validate() enforces (the registry's check_selection, halt_on_error-vs-auto coherence,
the pacing floor, budget.on_exhausted's enum, ranker.retrain_every) was exercised only
against inline YAML in other test files, never against the file we actually ship. A
config.yaml that could not start was therefore fully CI-green.
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


def test_shipped_config_actually_passes_validate():
    """The file we ship must satisfy every rule validate() enforces — not just parse.

    This is the test whose absence let the shipped config drift: validate() is what the
    real entry points call before a run, so a config.yaml that fails it cannot start the
    app at all, yet nothing here checked. It needs no network and no credentials.
    """
    from operation_love.config import validate
    validate(load("config.yaml"))          # must not raise


def test_shipped_config_selects_something_the_registry_can_actually_run():
    """enabled_apps must name a platform that is available right now.

    validate() deliberately allows an UNAVAILABLE platform (you must be able to configure
    Bumble's coordinates before Bumble is calibrated), so it alone cannot catch a shipped
    config that parses, validates, and then refuses to start. That gap is what this covers.
    """
    from operation_love import platforms
    cfg = load("config.yaml")
    assert platforms.check_runnable(cfg.enabled_apps) is None, (
        f"config.yaml ships enabled_apps={cfg.enabled_apps}, which cannot start: "
        f"{platforms.check_runnable(cfg.enabled_apps)}"
    )
