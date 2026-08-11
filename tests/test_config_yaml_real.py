"""Smoke tests against the shipped config.yaml — catches typos and dead config keys
that unit tests with inline YAML strings would miss.

All tests here are offline (no GCP / no phone / no SDK).

These used to skip validate() entirely, on the stated grounds that it "requires a
BigQuery project_id + photo_bucket ... would fail in a clean CI env". That premise was
wrong: validate() only checks those keys are present, non-empty strings — it opens no
connection and needs no credentials. The consequence was a real hole: every rule
validate() enforces (the registry's check_selection, halt_on_error-vs-auto coherence,
the pacing floor, opener.max_attempts' bounds, ranker.retrain_every) was exercised only
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


def test_shipped_opener_style_keeps_faithful_corey_framework_and_two_sentence_cap(cfg):
    style = " ".join(cfg.opener.style.lower().split())
    assert "90/10 framework" in style
    assert "genuinely curious" in style
    assert "do not force teasing into every opener" in style
    assert "one open, easy to answer question" in style
    assert "positive, fun conversation" in style
    assert "brief greeting is optional" in style
    assert "two sentences is the absolute maximum" in style
    assert "exactly one concrete detail" in style
    assert "never use an em dash or any hyphen" in style
    assert "low investment so she chases" not in style
    assert "tease her like a bratty little sister" not in style
    assert "no interview questions" not in style


def test_hinge_identity_top_name_band_matches_the_measured_ocr_band(cfg):
    """apps.hinge.identity_top_name_band must carry the exact band MEASURED on real Pixel 7a
    frames on 2026-08-10 (tesseract --psm 6 read the card-header name correctly on every
    scroll-top frame tested, both banner-present and banner-gone layouts). This is what fixes
    observe mode recording a pass that advanced Alina -> jessica as a scroll of Alina -- see
    android_spec.py's identity_top_name_band docstring for the full mechanism. A drifted or
    dropped value here would silently defeat the scroll-top name check on the one platform
    that actually runs (Hinge; see live-bringup-status)."""
    band = cfg.apps["hinge"].get("identity_top_name_band")
    assert band is not None, "apps.hinge.identity_top_name_band is missing from config.yaml"
    assert tuple(band) == (0.03, 0.130, 0.75, 0.250)


def test_limits_are_uncapped_by_default(cfg):
    """Auto-mode volume is deliberately uncapped by default (see config.yaml's `limits:`
    block): a fixed numeric ceiling is itself a bot signature (an identical hard step
    every run), so the shipped config must not set any of these by default. Timing remains
    paced; the profile queue, real stop conditions, or a manual stop end the run.
    """
    lim = cfg.limits
    assert lim.get("max_per_run") is None
    assert lim.get("max_per_day") is None
    assert lim.get("max_likes_per_run") is None
    assert lim.get("target_like_ratio") is None


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
