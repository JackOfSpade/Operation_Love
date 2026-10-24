"""Config surface of the recall-target threshold and the auto pass review section.

``ranker.like_recall_target`` / ``ranker.contextual_recall_budget`` / ``auto_review.*`` must be
validated at load time (a typo has to fail startup, not silently pick a default), the shipped
config.yaml must carry the owner's chosen values, and a config that omits them must keep the
legacy fixed-threshold behaviour.
"""

import math
import tempfile
from pathlib import Path

import pytest
import yaml

from operation_love import config as c

BASE = {
    "enabled_apps": ["hinge"],
    "mode": "training",
    "storage": {"backend": "sqlite"},
    "opener": {
        "enabled": True,
        "model": "gemini-3.6-flash",
        "thinking": {"gemini-3.6-flash": {}},
    },
    "budget": {
        "run_budget_usd": 5.0,
        "pricing": {"gemini-3.6-flash": {"input": 5, "output": 25}},
    },
}


def _load(d):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(d, f)
    try:
        return c.load(f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)


def _expect_error(d, needle):
    with pytest.raises(ValueError) as info:
        c.validate(_load(d))
    assert needle in str(info.value), str(info.value)


# --- defaults ------------------------------------------------------------------------------


def test_omitted_sections_keep_the_legacy_fixed_threshold():
    cfg = _load(BASE)
    c.validate(cfg)
    assert (
        cfg.ranker.like_recall_target is None
    )  # no recall target => fixed like_threshold
    assert cfg.ranker.contextual_recall_budget == 0.02
    assert cfg.auto_review == c.AutoReviewCfg(enabled=True, random_share=0.5)
    # Retention is on by default, never "keep every stranger's photos forever".
    assert (cfg.auto_review.max_archived_passes, cfg.auto_review.max_age_days) == (
        1500,
        60.0,
    )


def test_a_hand_built_config_gets_the_default_auto_review_section(tmp_path):
    cfg = c.Config(
        enabled_apps=["hinge"],
        mode="auto",
        apps={},
        limits={},
        data_dir=tmp_path,
        db_file=tmp_path / "s.db",
        ranker=c.RankerCfg(),
        quality_filter=c.QualityCfg(),
        opener=c.OpenerCfg(enabled=False),
        budget=c.BudgetCfg(pricing={}),
        pacing=c.PacingCfg(),
        storage=c.StorageCfg(backend="sqlite"),
    )
    assert cfg.auto_review.enabled is True and cfg.auto_review.random_share == 0.5
    # Each Config owns its section: mutating one must not leak into the next.
    cfg.auto_review.random_share = 0.9
    assert c.AutoReviewCfg().random_share == 0.5


def test_null_auto_review_and_null_recall_target_load_as_defaults():
    cfg = _load({**BASE, "auto_review": None, "ranker": {"like_recall_target": None}})
    c.validate(cfg)
    assert cfg.ranker.like_recall_target is None
    assert cfg.auto_review.enabled is True


# --- the shipped file ----------------------------------------------------------------------


def test_shipped_config_yaml_carries_the_owner_recall_target():
    cfg = c.load("config.yaml")
    c.validate(cfg)
    assert cfg.ranker.like_recall_target == 0.90
    assert cfg.ranker.contextual_recall_budget == 0.02
    assert cfg.auto_review.enabled is True
    assert cfg.auto_review.random_share == 0.5
    # The shipped file bounds the archive of strangers' screenshots (likes and passes) by count
    # AND age, sized for reviews that wait until the owner gets to them (~3.6 GB at 2.4 MB each).
    assert cfg.auto_review.max_archived_passes == 1500
    assert cfg.auto_review.max_age_days == 60
    # like_threshold stays valid: it is the pre-calibration placeholder.
    assert 0.0 < cfg.ranker.like_threshold < 1.0


# --- accepted values -----------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.001, 0.5, 0.95, 0.999])
def test_recall_target_in_open_unit_interval_is_accepted(target):
    cfg = _load({**BASE, "ranker": {"like_recall_target": target}})
    c.validate(cfg)
    assert cfg.ranker.like_recall_target == target


@pytest.mark.parametrize("budget", [0, 0.0, 0.02, 0.2])
def test_contextual_budget_bounds_are_inclusive(budget):
    c.validate(_load({**BASE, "ranker": {"contextual_recall_budget": budget}}))


@pytest.mark.parametrize("share", [0.001, 0.5, 0.999])
def test_random_share_in_open_unit_interval_is_accepted(share):
    cfg = _load({**BASE, "auto_review": {"random_share": share}})
    c.validate(cfg)
    assert cfg.auto_review.random_share == share


@pytest.mark.parametrize("count", [0, 1, 400, 100_000])
@pytest.mark.parametrize("days", [0, 0.5, 30, 3650])
def test_retention_bounds_accept_zero_as_unlimited(count, days):
    cfg = _load(
        {**BASE, "auto_review": {"max_archived_passes": count, "max_age_days": days}}
    )
    c.validate(cfg)
    assert (cfg.auto_review.max_archived_passes, cfg.auto_review.max_age_days) == (
        count,
        days,
    )


@pytest.mark.parametrize("bad", [-1, 1.5, True, "400", None, [400], math.nan, math.inf])
def test_retention_count_must_be_a_non_negative_integer(bad):
    _expect_error(
        {**BASE, "auto_review": {"max_archived_passes": bad}}, "max_archived_passes"
    )


@pytest.mark.parametrize("bad", [-1, -0.5, True, "30", None, [30], math.nan, math.inf])
def test_retention_age_must_be_a_non_negative_number(bad):
    _expect_error({**BASE, "auto_review": {"max_age_days": bad}}, "max_age_days")


def test_auto_review_can_be_disabled():
    cfg = _load({**BASE, "auto_review": {"enabled": False}})
    c.validate(cfg)
    assert cfg.auto_review.enabled is False


# --- rejected values -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", [0, 0.0, 1, 1.0, -0.1, 1.5, True, False, "0.95", [0.9], math.nan, math.inf]
)
def test_recall_target_outside_the_open_unit_interval_is_rejected(bad):
    _expect_error({**BASE, "ranker": {"like_recall_target": bad}}, "like_recall_target")


@pytest.mark.parametrize(
    "bad", [-0.01, 0.21, 1, True, "0.02", None, math.nan, math.inf]
)
def test_contextual_budget_outside_zero_to_point_two_is_rejected(bad):
    _expect_error(
        {**BASE, "ranker": {"contextual_recall_budget": bad}},
        "contextual_recall_budget",
    )


@pytest.mark.parametrize("bad", ["yes", "true", 1, 0, None, [True]])
def test_auto_review_enabled_must_be_a_real_bool(bad):
    _expect_error({**BASE, "auto_review": {"enabled": bad}}, "auto_review.enabled")


@pytest.mark.parametrize(
    "bad", [0, 0.0, 1, 1.0, -0.1, 1.5, True, "0.5", None, math.nan, math.inf]
)
def test_random_share_outside_the_open_unit_interval_is_rejected(bad):
    _expect_error({**BASE, "auto_review": {"random_share": bad}}, "random_share")


def test_a_disabled_auto_review_section_is_still_validated():
    # "enabled: false" must not become a way to smuggle a nonsense share past startup.
    _expect_error(
        {**BASE, "auto_review": {"enabled": False, "random_share": 2.0}},
        "random_share",
    )


# --- unknown keys --------------------------------------------------------------------------


def test_unknown_auto_review_key_is_rejected_by_name():
    with pytest.raises(ValueError, match="auto_review"):
        _load({**BASE, "auto_review": {"random_shar": 0.5}})


def test_unknown_ranker_key_next_to_the_new_ones_is_still_rejected():
    with pytest.raises(ValueError, match="ranker"):
        _load({**BASE, "ranker": {"like_recall_targt": 0.95}})


def test_auto_review_is_a_recognised_top_level_key():
    # Absent from _TOP_LEVEL_KEYS this would read "unknown key(s) under the top level".
    cfg = _load({**BASE, "auto_review": {"enabled": True}})
    assert cfg.auto_review.enabled is True
    assert "auto_review" in c._TOP_LEVEL_KEYS


def test_auto_review_section_must_be_a_mapping():
    with pytest.raises(ValueError, match="auto_review"):
        _load({**BASE, "auto_review": [True]})
