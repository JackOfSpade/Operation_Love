"""config.validate() — fail-fast checks (offline)."""
import copy
import hashlib
import json
import math
import re
import tempfile

import pytest
import yaml

from operation_love import config as c
from operation_love import targeting_policy as tp

BASE = {
    # hinge: the one platform the registry ships available/calibrated by default. "bumble"
    # is now an Android target that starts out uncalibrated (platforms.py) and would fail
    # the check_runnable() guard validate() now applies -- see test_unavailable_app_* below
    # for coverage of that rejection path.
    "enabled_apps": ["hinge"],
    "mode": "observe",
    "storage": {"backend": "sqlite"},
    # Model id deliberately matches OpenerCfg's own class default (see config.py) so that
    # a test overriding "opener" to None -- which falls back to those class defaults --
    # still resolves to a model with a budget.pricing entry below (see
    # test_null_opener_block_loads_cleanly_but_still_requires_gemini_thinking).
    "opener": {"enabled": True, "model": "gemini-3.6-flash",
               "thinking": {"gemini-3.6-flash": {}}},
    "budget": {"run_budget_usd": 5.0,
               "pricing": {"gemini-3.6-flash": {"input": 5, "output": 25}}},
}

_TARGETING_GEOMETRY = {
    "identity_band": [0.10, 0.048, 0.80, 0.094],
    "content_band": [0.125, 0.875],
}

_TARGETING_SCHEMA_V2 = {
    "schema_version": 3,
    "hinge_version_name": "9.134.0",
    "frame_size_px": [1080, 2400],
    "composer_layout_id": "hinge_inline_v1",
    "item_selection_policy_id": "hinge_photos_only_v2",
}


def _load(d):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    return c.load(f.name)


def _expect_error(d, needle):
    try:
        c.validate(_load(d))
    except ValueError as e:
        assert needle in str(e), f"expected '{needle}' in: {e}"
    else:
        raise AssertionError(f"expected ValueError containing '{needle}'")


def test_valid_config_passes():
    c.validate(_load(BASE))   # no raise


@pytest.mark.parametrize(("key", "value", "needle"), [
    ("limts", {}, "top level"),
    ("budegt", {}, "top level"),
    ("paths", [], "paths"),
    ("storage", "sqlite", "storage"),
    ("budget", [], "budget"),
    ("apps", [], "apps"),
])
def test_top_level_typos_and_non_mapping_sections_fail_cleanly(key, value, needle):
    d = {**BASE, key: value}
    if key in {"limts", "budegt"}:
        d.pop("limits" if key == "limts" else "budget", None)
    _expect_error(d, needle)


@pytest.mark.parametrize(("section", "bad_key"), [
    ("paths", "data_dr"),
    ("storage", "backed"),
])
def test_hand_built_sections_reject_unknown_keys(section, bad_key):
    d = copy.deepcopy(BASE)
    d[section] = {bad_key: "typo"}
    _expect_error(d, bad_key)


@pytest.mark.parametrize("key", ["data_dir", "db_file"])
@pytest.mark.parametrize("value", [None, [], {}, True, 12, ""])
def test_paths_require_nonempty_string_scalars(key, value):
    _expect_error({**BASE, "paths": {key: value}}, f"paths.{key}")


def test_bigquery_section_rejects_non_mapping_and_unknown_keys():
    for value in ([], "project"):
        d = {**BASE, "storage": {"backend": "bigquery", "bigquery": value}}
        _expect_error(d, "storage.bigquery")
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {
        "project_id": "p", "photo_bucket": "b", "dataset": "d", "location": "US",
        "flush_evry": 1,
    }}}
    _expect_error(d, "flush_evry")


@pytest.mark.parametrize("value", ["hinge", 1, {}, ["hinge", "hinge"], [""], [1]])
def test_enabled_apps_requires_unique_nonempty_string_list(value):
    _expect_error({**BASE, "enabled_apps": value}, "enabled_apps")


@pytest.mark.parametrize("value", [None, "", 1, []])
def test_legacy_singular_app_requires_nonempty_string(value):
    d = {k: v for k, v in BASE.items() if k != "enabled_apps"}
    d["app"] = value
    _expect_error(d, "legacy app")


def test_enabled_apps_and_legacy_app_cannot_both_be_present():
    _expect_error({**BASE, "app": "hinge"}, "not both")


@pytest.mark.parametrize(("path", "value", "needle"), [
    ("mode", ["auto"], "mode must be a string"),
    ("apps", {"hinge": []}, "apps.hinge"),
    ("apps", {"hinge": {"mode": ["auto"]}}, "apps.hinge.mode"),
])
def test_mode_and_app_blocks_fail_shape_checks_before_set_operations(path, value, needle):
    _expect_error({**BASE, path: value}, needle)


def test_unknown_app():
    d = {**BASE, "enabled_apps": ["tinder"]}
    _expect_error(d, "unknown app")


def test_unknown_disabled_app_block_is_rejected_as_a_likely_typo():
    _expect_error({**BASE, "apps": {"higne": {"dwell_s": 1.1}}}, "apps.higne")


def test_known_disabled_app_block_remains_supported():
    c.validate(_load({**BASE, "apps": {"bumble": {"debug_log": False}}}))


def test_bad_mode():
    d = {**BASE, "mode": "yolo"}
    _expect_error(d, "observe")


def test_bigquery_requires_project_id():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {}}}
    _expect_error(d, "project_id")


def test_bigquery_requires_photo_bucket():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {"project_id": "proj"}}}
    _expect_error(d, "photo_bucket")


def test_bigquery_photo_bucket_rejects_surrounding_whitespace():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {
        "project_id": "proj", "photo_bucket": " bucket ",
    }}}
    _expect_error(d, "photo_bucket")


def test_bigquery_dataset_and_location_defaults_remain_supported():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {
        "project_id": "valid-project", "photo_bucket": "valid-bucket",
    }}}
    c.validate(_load(d))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "10"])
def test_bigquery_flush_every_requires_an_exact_positive_integer(value):
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {
        "project_id": "valid-project", "photo_bucket": "valid-bucket",
        "flush_every": value,
    }}}
    _expect_error(d, "flush_every")


@pytest.mark.parametrize(("key", "value"), [
    ("project_id", "project.name"),
    ("project_id", "project`name"),
    ("dataset", "dataset.name"),
    ("dataset", "dataset name"),
    ("location", "US' OR 1=1"),
    ("location", "north america"),
])
def test_bigquery_sql_identifiers_reject_unsafe_characters(key, value):
    bq = {"project_id": "valid-project", "photo_bucket": "valid-bucket",
          "dataset": "valid_dataset", "location": "northamerica-northeast1"}
    bq[key] = value
    _expect_error(
        {**BASE, "storage": {"backend": "bigquery", "bigquery": bq}}, key)


def test_opener_model_needs_pricing():
    d = {**BASE, "opener": {"enabled": True, "model": "gemini-unknown-9"}}
    _expect_error(d, "budget.pricing")


def test_gemini_models_are_an_ordered_fallback_chain_and_each_needs_pricing():
    d = {**BASE,
         "opener": {"enabled": True, "provider": "gemini", "model": "legacy",
                    "models": ["gemini-primary", "gemini-fallback"],
                    "thinking": {"gemini-primary": {}, "gemini-fallback": {}}},
         "budget": {**BASE["budget"], "pricing": {
             "gemini-primary": {"input": 0, "output": 0},
             "gemini-fallback": {"input": 0, "output": 0},
         }}}
    cfg = _load(d)
    assert cfg.opener.effective_models == ["gemini-primary", "gemini-fallback"]
    c.validate(cfg)
    d["budget"]["pricing"].pop("gemini-fallback")
    _expect_error(d, "budget.pricing")


# --- opener.thinking: required per Gemini model, shape-validated -----------------------
# Thinking is ON BY DEFAULT for nearly every free-tier model in this project's cascade and
# is billed against opener.max_tokens (see config.py's _validate_gemini_thinking and
# GeminiOpener._parse's MAX_TOKENS diagnostic) -- an unset entry risks silently truncating
# every opener, so it's required rather than optional.

def _gemini_opener(models, thinking, **extra):
    return {"enabled": True, "provider": "gemini", "model": models[0],
            "models": models, "thinking": thinking, **extra}


def _gemini_budget(models):
    return {**BASE["budget"], "pricing": {m: {"input": 0, "output": 0} for m in models}}


def test_gemini_model_with_no_thinking_entry_fails_naming_the_model():
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary", "gemini-fallback"],
                                  {"gemini-primary": {}}),  # gemini-fallback missing
         "budget": _gemini_budget(["gemini-primary", "gemini-fallback"])}
    _expect_error(d, "gemini-fallback")


def test_gemini_explicit_empty_thinking_dict_passes():
    # {} is the sanctioned way to say "use this model's server-default thinking level".
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"], {"gemini-primary": {}}),
         "budget": _gemini_budget(["gemini-primary"])}
    c.validate(_load(d))   # no raise


def test_gemini_thinking_rejects_unconfigured_model_entries():
    d = {**BASE,
         "opener": _gemini_opener(
             ["gemini-primary"],
             {"gemini-primary": {}, "gemini-typo": {}}),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "unconfigured model")


def test_gemini_bad_thinking_level_value_fails_clearly():
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"],
                                  {"gemini-primary": {"thinkingLevel": "extreme"}}),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "thinkingLevel")


def test_gemini_unknown_thinking_key_fails_clearly():
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"],
                                  {"gemini-primary": {"thinkingLevel": "minimal",
                                                      "thinkingDepth": 3}}),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "unknown key")


def test_gemini_thinking_budget_must_be_an_int():
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"],
                                  {"gemini-primary": {"thinkingBudget": "zero"}}),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "thinkingBudget")


def test_gemini_thinking_budget_rejects_a_bool():
    """`bool` is a subclass of `int` in Python, so a bare isinstance(x, int) check waves
    `thinkingBudget: true` through -- and YAML's `true` is very easy to type where a 0 was
    meant. It would reach the API as JSON `true` and 400 every single opener call, which is
    precisely what this validator exists to prevent."""
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"],
                                  {"gemini-primary": {"thinkingBudget": True}}),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "thinkingBudget")


def test_gemini_thinking_budget_rejects_a_negative_int():
    """0 disables thinking and positive values cap it, so a negative budget is meaningless
    to the API and is only ever a mistake -- catch it here rather than as a live 400."""
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"],
                                  {"gemini-primary": {"thinkingBudget": -50}}),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, ">= 0")


def test_null_thinking_block_fails_cleanly_instead_of_crashing():
    """A bare `thinking:` key in YAML parses to None, and OpenerCfg is built generically via
    cls(**raw_section) so nothing coerces it first. This used to escape validate() as a raw
    TypeError ("argument of type 'NoneType' is not a container") instead of the actionable
    ValueError every other optional block in this file degrades to -- see the null-block
    tests further down for the convention this restores."""
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"], None),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "gemini-primary")     # reported as a missing entry, not a crash


def test_non_mapping_thinking_block_fails_cleanly():
    d = {**BASE,
         "opener": _gemini_opener(["gemini-primary"], ["gemini-primary"]),
         "budget": _gemini_budget(["gemini-primary"])}
    _expect_error(d, "must be a mapping")


def test_legacy_single_opener_model_remains_effective():
    # effective_models' fallback to the legacy singular `model:` key (when `models` is
    # empty) is deliberately retained -- see config.py's OpenerCfg docstring -- even though
    # gemini is now the only provider.
    cfg = _load(BASE)
    assert cfg.opener.provider == "gemini"
    assert cfg.opener.effective_models == ["gemini-3.6-flash"]


def test_unknown_opener_provider_is_rejected():
    d = {**BASE, "opener": {"enabled": True, "provider": "unknown", "model": "gemini-3.6-flash"}}
    _expect_error(d, "opener.provider")


def test_anthropic_opener_provider_is_rejected_as_removed():
    # The owner's explicit decision: the legacy Anthropic/Claude opener path was excised
    # from the codebase entirely (operation_love/opener/opener.py no longer defines
    # AnthropicOpener at all), not merely defaulted off. A config still naming it must fail
    # loudly at load time with a message that says so -- a stray/legacy `provider:
    # anthropic` must never be reachable as a silent fallback.
    d = {**BASE, "opener": {"enabled": True, "provider": "anthropic", "model": "claude-opus-4-8"}}
    _expect_error(d, "removed")


def test_opener_models_must_be_a_yaml_list():
    d = {**BASE, "opener": {"enabled": True, "model": "gemini-3.6-flash", "models": "gemini-3.6-flash"}}
    _expect_error(d, "opener.models must be a YAML list")


def test_bad_app_mode_override():
    d = {**BASE, "mode": "observe", "apps": {"hinge": {"mode": "yolo"}}}
    _expect_error(d, "observe")


def test_disabled_known_app_mode_is_still_validated():
    d = {**BASE, "apps": {"bumble": {"mode": "automatic"}}}
    _expect_error(d, "apps.bumble.mode")


@pytest.mark.parametrize(("field", "value"), [
    ("halt_on_error", "true"),
    ("limits", []),
    ("limits", {"max_per_run": True}),
])
def test_disabled_known_app_safety_fields_are_still_validated(field, value):
    d = {**BASE, "apps": {"bumble": {field: value}}}
    _expect_error(d, f"apps.bumble.{field}")


def test_disabled_known_app_cannot_stage_unverified_auto_halt_policy():
    d = {**BASE, "apps": {"bumble": {"mode": "auto", "halt_on_error": False}}}
    _expect_error(d, "halt_on_error=false")


def test_hinge_auto_app_mode_override_reports_structural_targeting_blocker_first():
    d = {**BASE, "mode": "observe", "apps": {"hinge": {"mode": "auto"}}}
    _expect_error(d, "positive still-photo discriminator unavailable")


def test_nonmanual_hinge_observe_source_requires_explicit_controller_metadata():
    d = {**BASE, "apps": {"hinge": {"observe_evidence_source": "automation"}}}
    _expect_error(d, "ai_reviewed_observe_controller")
    d["apps"]["hinge"]["ai_reviewed_observe_controller"] = {
        "schema_version": 1, "source": "automation",
        "acceptance": "I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK",
        "executor": {"model": "gpt", "id": "controller", "version": "v1", "process": "driver"},
    }
    c.validate(_load(d))


def test_nonmanual_hinge_observe_source_cannot_be_enabled_for_auto_mode():
    d = {**BASE, "mode": "auto", "apps": {"hinge": {
        "observe_evidence_source": "external_ai_review",
        "ai_reviewed_observe_controller": {
            "schema_version": 1, "source": "external_ai_review",
            "acceptance": "I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK",
            "executor": {"model": "gpt", "id": "controller", "version": "v1", "process": "driver"},
        },
    }}}
    _expect_error(d, "only in mode observe")


# --- budget.on_exhausted: removed 2026-08-10 (owner ruled out commentless likes) -----------
# A config still setting it -- 'stop' or 'swipe_without_opener' -- must fail loudly at load()
# time (before validate() even runs) rather than silently ignore a dead key or crash with a
# confusing TypeError from BudgetCfg's generic **raw_section construction.

def test_stale_on_exhausted_stop_fails_loudly_with_actionable_message():
    """Fires even for the value that used to be the default and "did nothing wrong" --
    otherwise someone who had `on_exhausted: stop` has no way to learn the key is dead and
    keeps believing it still controls something."""
    d = {**BASE, "budget": {**BASE["budget"], "on_exhausted": "stop"}}
    try:
        _load(d)
    except ValueError as e:
        msg = str(e)
        assert "on_exhausted" in msg
        assert "removed" in msg
        assert "commentless" in msg.lower() or "openerless" in msg.lower()
        # Must reassure a former `stop` user that nothing about their run's behavior changed.
        assert "nothing about your run's behavior changes" in msg
    else:
        raise AssertionError("expected ValueError for stale budget.on_exhausted: stop")


def test_stale_on_exhausted_swipe_without_opener_fails_loudly():
    d = {**BASE, "budget": {**BASE["budget"], "on_exhausted": "swipe_without_opener"}}
    try:
        _load(d)
    except ValueError as e:
        msg = str(e)
        assert "on_exhausted" in msg
        assert "removed" in msg
        assert "max_attempts" in msg   # points at the setting that replaced it
    else:
        raise AssertionError("expected ValueError for stale budget.on_exhausted: "
                             "swipe_without_opener")


def test_stale_on_exhausted_is_not_a_confusing_generic_typeerror():
    """BudgetCfg is built generically via cls(**raw_section) elsewhere in this file, and an
    unexpected key there normally surfaces as a TypeError wrapped into a generic ValueError.
    budget.on_exhausted must produce OUR dedicated, actionable message instead -- not that
    generic 'invalid budget section' wrapper."""
    d = {**BASE, "budget": {**BASE["budget"], "on_exhausted": "stop"}}
    try:
        _load(d)
    except ValueError as e:
        assert "invalid 'budget' section" not in str(e)
    else:
        raise AssertionError("expected ValueError for stale budget.on_exhausted")


@pytest.mark.parametrize("field", ["run_budget_usd", "day_budget_usd"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1, True, "5"])
def test_budget_caps_require_finite_nonnegative_numbers_or_null(field, value):
    budget = copy.deepcopy(BASE["budget"])
    budget[field] = value
    _expect_error({**BASE, "budget": budget}, f"budget.{field}")


@pytest.mark.parametrize("budget", [None, {}, {"pricing": {}}])
def test_omitted_run_budget_preserves_safe_five_dollar_default(budget):
    d = {**BASE, "opener": {"enabled": False}, "budget": budget}
    cfg = _load(d)
    assert cfg.budget.run_budget_usd == 5.00
    c.validate(cfg)


@pytest.mark.parametrize(("value", "expected"), [(None, None), (0, 0)])
def test_explicit_null_or_zero_run_budget_remains_distinct_from_omission(value, expected):
    d = {**BASE, "opener": {"enabled": False}, "budget": {"run_budget_usd": value}}
    cfg = _load(d)
    assert cfg.budget.run_budget_usd is expected or cfg.budget.run_budget_usd == expected
    c.validate(cfg)


@pytest.mark.parametrize("field", ["run_budget_usd", "day_budget_usd"])
@pytest.mark.parametrize("value", [None, 0, 5.25])
def test_budget_caps_allow_null_zero_and_finite_nonnegative_values(field, value):
    budget = copy.deepcopy(BASE["budget"])
    budget[field] = value
    c.validate(_load({**BASE, "budget": budget}))


def test_budget_pricing_requires_mapping_model_ids_and_exact_record_keys():
    for pricing, needle in [
        ([], "budget.pricing"),
        ({"": {"input": 0, "output": 0}}, "model ids"),
        ({"m": []}, "budget.pricing['m']"),
        ({"m": {"input": 0}}, "missing"),
        ({"m": {"input": 0, "output": 0, "ouptut": 0}}, "unknown"),
    ]:
        budget = {**BASE["budget"], "pricing": pricing}
        _expect_error({**BASE, "budget": budget}, needle)


@pytest.mark.parametrize("field,value", [
    ("input", -1),
    ("output", math.nan),
    ("cache_read", math.inf),
    ("cache_write", True),
    ("input", "0"),
])
def test_budget_pricing_rejects_values_that_can_poison_spend(field, value):
    record = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    record[field] = value
    budget = {**BASE["budget"], "pricing": {"gemini-3.6-flash": record}}
    _expect_error({**BASE, "budget": budget}, field)


def test_budget_pricing_keeps_valid_free_tier_zero_rates_legal():
    budget = {**BASE["budget"], "pricing": {
        "gemini-3.6-flash": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
    }}
    cfg = _load({**BASE, "budget": budget})
    c.validate(cfg)
    assert cfg.budget.pricing["gemini-3.6-flash"].input == 0


# --- opener.max_attempts: owner rule, "stop after 5 bad AI responses" ----------------------

def test_max_attempts_default_is_five():
    cfg = _load(BASE)
    assert cfg.opener.max_attempts == 5


def test_max_attempts_accepts_a_valid_int():
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 3}}
    cfg = _load(d)
    assert cfg.opener.max_attempts == 3
    c.validate(cfg)   # no raise


def test_max_attempts_rejects_zero():
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 0}}
    _expect_error(d, "max_attempts")


def test_max_attempts_rejects_negative():
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": -1}}
    _expect_error(d, "max_attempts")


def test_max_attempts_rejects_bool():
    # bool is a subclass of int in Python -- the same trap opener.thinking's thinkingBudget
    # guards against elsewhere in config.py. `max_attempts: true` must not silently pass as 1.
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": True}}
    _expect_error(d, "max_attempts")


def test_max_attempts_rejects_non_int():
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": "5"}}
    _expect_error(d, "max_attempts")


# --- opener.max_attempts: upper bound. An audit found max_attempts had NO ceiling -- a
# `max_attempts: 10000` config passed validate() cleanly, and since every attempt is a real,
# billed, quota-consuming API call that (on a rejected-content retry) ordinarily re-hits the
# SAME model rather than advancing the fallback cascade, that could burn a whole day's quota
# of one of this project's 20-requests/day models on a single stubborn profile, or hang a
# profile for hundreds of hours bounded only by request_timeout_s. See config.py's
# _MAX_ATTEMPTS_CEILING docstring for the full arithmetic behind the chosen ceiling of 15. ---

def test_max_attempts_ceiling_value_is_accepted():
    # Boundary: the ceiling itself (15) must still be a legal, usable value.
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 15}}
    cfg = _load(d)
    assert cfg.opener.max_attempts == 15
    c.validate(cfg)   # no raise


def test_max_attempts_one_above_ceiling_is_rejected():
    # Boundary: one past the ceiling (16) must fail -- proves the check is `>`, not `>=`.
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 16}}
    _expect_error(d, "max_attempts")


def test_max_attempts_above_ceiling_message_is_actionable():
    """The error must do three things per the spec this bound was added to satisfy: state
    the accepted range, explain why a ceiling exists at all (billed calls against a small
    daily quota), and say what to do instead of just cranking the number up."""
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 10000}}
    try:
        c.validate(_load(d))
    except ValueError as e:
        msg = str(e)
        assert "1 and 15" in msg                      # accepted range, stated explicitly
        assert "10000" in msg                          # echoes the offending value
        assert "billed" in msg.lower()                 # why a ceiling exists at all
        assert "quota" in msg.lower()
        # what to do instead of raising the ceiling further
        assert "debug log" in msg.lower() or "opener.style" in msg.lower()
    else:
        raise AssertionError("expected ValueError for max_attempts=10000")


def test_max_attempts_way_above_ceiling_rejected_same_as_just_above():
    # Regression test for the exact value an audit found `validate()` accepting.
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 10000}}
    _expect_error(d, "max_attempts")


# --- observe-mode opener retry policy -------------------------------------------------

def test_advisory_retry_defaults_are_bounded():
    cfg = _load(BASE)
    assert cfg.opener.advisory_max_attempts == 3
    assert cfg.opener.advisory_deadline_s == 60.0
    c.validate(cfg)


def test_advisory_retry_settings_accept_valid_values():
    d = {**BASE, "opener": {**BASE["opener"], "advisory_max_attempts": 2,
                             "advisory_deadline_s": 12.5}}
    cfg = _load(d)
    assert cfg.opener.advisory_max_attempts == 2
    assert cfg.opener.advisory_deadline_s == 12.5
    c.validate(cfg)


@pytest.mark.parametrize("value", [0, -1, True, "3", 2.5])
def test_advisory_max_attempts_rejects_invalid_values(value):
    d = {**BASE, "opener": {**BASE["opener"], "advisory_max_attempts": value}}
    _expect_error(d, "advisory_max_attempts")


def test_advisory_max_attempts_cannot_exceed_auto_budget():
    d = {**BASE, "opener": {**BASE["opener"], "max_attempts": 2,
                             "advisory_max_attempts": 3}}
    _expect_error(d, "advisory_max_attempts")


@pytest.mark.parametrize("value", [0, -1, True, "60", 301])
def test_advisory_deadline_rejects_invalid_values(value):
    d = {**BASE, "opener": {**BASE["opener"], "advisory_deadline_s": value}}
    _expect_error(d, "advisory_deadline_s")


@pytest.mark.parametrize(
    "field", ["max_attempts", "advisory_max_attempts", "advisory_deadline_s",
              "request_timeout_s"])
def test_bounded_opener_numbers_report_huge_integers_as_config_errors(field):
    cfg = _load(BASE)
    setattr(cfg.opener, field, 1 << 20_000)

    with pytest.raises(ValueError) as caught:
        c.validate(cfg)

    assert field in str(caught.value)


# --- opener.request_timeout_s: the only bound on how long a single opener API call can run.
# Previously unvalidated entirely (no type check, no floor, no ceiling). An unbounded value
# here would silently undo opener.max_attempts' own new ceiling, since one stalled call could
# still hang a profile indefinitely regardless of how few retries are allowed. See config.py's
# _MAX_REQUEST_TIMEOUT_S docstring for the arithmetic (2x the measured 90s worst case). -------

def test_request_timeout_s_default_passes():
    cfg = _load(BASE)   # BASE sets no request_timeout_s -> OpenerCfg's class default
    c.validate(cfg)     # no raise


def test_request_timeout_s_accepts_ceiling_value():
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": 180}}
    cfg = _load(d)
    assert cfg.opener.request_timeout_s == 180
    c.validate(cfg)   # no raise


def test_request_timeout_s_rejects_above_ceiling():
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": 181}}
    _expect_error(d, "request_timeout_s")


def test_request_timeout_s_rejects_zero():
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": 0}}
    _expect_error(d, "request_timeout_s")


def test_request_timeout_s_rejects_negative():
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": -1}}
    _expect_error(d, "request_timeout_s")


def test_request_timeout_s_rejects_bool():
    # Same bool-is-an-int-subclass trap guarded against elsewhere in config.py.
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": True}}
    _expect_error(d, "request_timeout_s")


def test_request_timeout_s_rejects_non_numeric():
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": "90"}}
    _expect_error(d, "request_timeout_s")


def test_request_timeout_s_accepts_a_float():
    # request_timeout_s is typed `float` on OpenerCfg -- a non-integer value must stay legal.
    d = {**BASE, "opener": {**BASE["opener"], "request_timeout_s": 45.5}}
    cfg = _load(d)
    assert cfg.opener.request_timeout_s == 45.5
    c.validate(cfg)   # no raise


# --- pacing.swipe_delay_s: live scale on worker._pace, must be bounded ---------------
def test_pacing_swipe_delay_rejects_negative_and_near_zero():
    # threading.Event.wait() treats a negative/near-zero timeout as "return immediately" --
    # worker._pace() multiplies this straight into the wait, so an unbounded value here is
    # machine-speed swiping on a live account. 0 is the explicit "pacing off" sentinel and
    # must stay legal (see test_pacing_swipe_delay_floor_and_off_and_default_pass).
    for bad in (-3.5, -0.01, 0.001, 0.999):
        d = {**BASE, "pacing": {"swipe_delay_s": bad}}
        _expect_error(d, "swipe_delay_s")


def test_pacing_swipe_delay_floor_and_off_and_default_pass():
    for ok in (0, 1.0, 3.5):
        d = {**BASE, "pacing": {"swipe_delay_s": ok}}
        c.validate(_load(d))   # no raise


def test_pacing_swipe_delay_rejects_values_that_can_overflow_event_wait():
    _expect_error(
        {**BASE, "pacing": {"swipe_delay_s": c._MAX_SWIPE_DELAY_S + 1}},
        "swipe_delay_s")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "0.5", -0.1, 1.1])
def test_ranker_like_threshold_requires_finite_probability(value):
    _expect_error({**BASE, "ranker": {"like_threshold": value}}, "like_threshold")


@pytest.mark.parametrize("field", ["min_labels_to_engage", "retrain_every", "min_per_class"])
@pytest.mark.parametrize("value", [True, 1.5, "4", 0, -1])
def test_ranker_counts_require_exact_positive_integers(field, value):
    _expect_error({**BASE, "ranker": {field: value}}, field)


@pytest.mark.parametrize(("field", "value"), [
    ("enabled", "true"),
    ("enabled", 1),
    ("metric", ""),
    ("metric", "   "),
    ("metric", 5),
    ("min_score", math.nan),
    ("min_score", math.inf),
    ("min_score", True),
    ("min_score", "0.3"),
    ("min_score", -0.1),
    ("min_score", 1.1),
])
def test_quality_filter_scalars_fail_cleanly(field, value):
    _expect_error({**BASE, "quality_filter": {field: value}}, f"quality_filter.{field}")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "3.5"])
def test_pacing_rejects_nonfinite_bool_and_non_numeric_values(value):
    _expect_error({**BASE, "pacing": {"swipe_delay_s": value}}, "swipe_delay_s")


def test_numeric_validation_rejects_enormous_integer_with_clean_value_error():
    _expect_error(
        {**BASE, "pacing": {"swipe_delay_s": 10 ** 1_000}},
        "swipe_delay_s")


@pytest.mark.parametrize(("field", "value"), [
    ("enabled", "false"),
    ("preflight", 1),
    ("provider", "   "),
    ("model", ""),
    ("model", " gemini-3.6-flash "),
    ("style", None),
    ("style", []),
    ("max_tokens", True),
    ("max_tokens", 1.5),
    ("max_tokens", "400"),
    ("max_tokens", 0),
])
def test_opener_scalar_shapes_fail_before_provider_calls(field, value):
    opener = {**BASE["opener"], field: value}
    _expect_error({**BASE, "opener": opener}, f"opener.{field}")


@pytest.mark.parametrize(
    "models", [["gemini-ok", "gemini-ok"], [""], ["   "], [" gemini-ok"], [1]])
def test_opener_model_fallback_ids_are_unique_nonempty_strings(models):
    opener = {**BASE["opener"], "models": models}
    _expect_error({**BASE, "opener": opener}, "opener.models")


@pytest.mark.parametrize("section", ["global", "per_app"])
@pytest.mark.parametrize("value", [[], "none"])
def test_limits_sections_must_be_mappings(section, value):
    if section == "global":
        d = {**BASE, "limits": value}
    else:
        d = {**BASE, "apps": {"hinge": {"limits": value}}}
    _expect_error(d, "limits")


@pytest.mark.parametrize("field", ["max_per_run", "max_per_day", "max_likes_per_run"])
@pytest.mark.parametrize("value", [True, 1.5, "5", 0, -1])
def test_limits_caps_require_exact_positive_integers(field, value):
    _expect_error({**BASE, "limits": {field: value}}, field)


@pytest.mark.parametrize("value", [True, "0.5", math.nan, math.inf, -math.inf, 0, 1, -0.1])
def test_target_like_ratio_requires_finite_open_interval_probability(value):
    _expect_error({**BASE, "limits": {"target_like_ratio": value}}, "target_like_ratio")


def test_valid_global_and_per_app_limits_still_pass():
    d = {**BASE, "limits": {"max_per_run": 8, "target_like_ratio": 0.5},
         "apps": {"hinge": {"limits": {"max_per_day": 20}}}}
    c.validate(_load(d))


# --- a bare `key:` (YAML null) must be treated as "key omitted", not crash -----------
def test_null_top_level_limits_does_not_crash_validate():
    d = {**BASE, "limits": None}
    c.validate(_load(d))   # no raise -- pre-fix this hit `set(None)` -> TypeError, not ValueError


def test_null_optional_blocks_are_treated_as_omitted():
    """Sweep: the same 'YAML null slips past a dict .get(..., {}) default' gap that broke
    `limits:` also affects every other optional block that gets spread (**) or further
    indexed after load() reads it -- fixed at the source in config.load(). `opener` is
    swept separately below: it loads to clean defaults the same as every key here, but
    since Gemini is the only opener provider now, its default model still has no
    `opener.thinking` entry, so validate() legitimately (and cleanly) rejects it -- see
    test_null_opener_block_loads_cleanly_but_still_requires_gemini_thinking."""
    for key in ("ranker", "quality_filter", "pacing", "paths", "apps"):
        d = {**BASE, key: None}
        c.validate(_load(d))   # no raise


def test_null_opener_block_loads_cleanly_but_still_requires_gemini_thinking():
    # `opener: null` must not crash with a raw TypeError (the same **None-spread bug the
    # sweep above guards for every other optional block) -- config.load() resolves it to
    # OpenerCfg's plain defaults without error. But those defaults carry no opener.thinking
    # entry, and there is no safe universal default for a field that can silently truncate
    # every opener (see config.py's _validate_gemini_thinking), so validate() must still
    # raise -- cleanly, naming the model -- rather than silently accept it.
    d = {**BASE, "opener": None}
    _expect_error(d, "opener.thinking")


def test_null_storage_bigquery_reports_clean_error_not_crash():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": None}}
    _expect_error(d, "project_id")   # clean ValueError, not AttributeError on None.get(...)


def test_empty_config_file_loads_with_defaults():
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.close()                          # zero-byte file -> yaml.safe_load returns None
    cfg = c.load(f.name)
    # hinge, not bumble: bumble is now an Android target that starts out uncalibrated
    # (platforms.py), so a from-scratch config defaulting to it would fail check_runnable().
    assert cfg.mode == "observe" and cfg.enabled_apps == ["hinge"]


def test_non_mapping_config_file_raises_clear_error():
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump([1, 2, 3], f)        # a YAML list, not a mapping
    f.close()
    try:
        c.load(f.name)
    except ValueError as e:
        assert "mapping" in str(e)
    else:
        raise AssertionError("expected ValueError for a non-mapping config file")


def test_unknown_key_in_section_raises_clear_error():
    d = {**BASE, "ranker": {"retrain_evry": 2}}   # typo'd key
    try:
        _load(d)
    except ValueError as e:
        assert "ranker" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown 'ranker' key")


# --- registry-driven validation: validate() defers to platforms.check_selection() ---------
#
# Deliberately STRUCTURAL only (unknown ids, two Android platforms contending for the one
# phone) -- NOT availability. Availability is a property of the world (calibrated? live
# target?) that changes without the config file changing, and writing Bumble's coordinates
# into config.yaml is exactly how Bumble gets calibrated -- a config merely NAMING an
# uncalibrated or web-dead platform must still load cleanly. The availability gate is
# start-time only: platforms.check_runnable(), asserted at supervisor.run() (see
# test_supervisor.py) and HubState.start() (see test_hub.py) instead.

def test_uncalibrated_android_app_loads_fine_at_config_time():
    # bumble is uncalibrated (unavailable) today, but that must not stop a config file that
    # merely enables it from loading -- see module docstring above.
    d = {**BASE, "enabled_apps": ["bumble"]}
    c.validate(_load(d))   # no raise


def test_two_android_platforms_together_rejected_at_config_time():
    # This one IS a config-time (structural) error regardless of either platform's
    # availability: Android shows one app in the foreground at a time, so two Android
    # platforms can never coexist in enabled_apps.
    d = {**BASE, "enabled_apps": ["hinge", "bumble"]}
    _expect_error(d, "cannot run together")


def test_still_unknown_app_uses_configs_own_message_not_registrys():
    # An app id the registry has never heard of must still fail on config.py's own
    # "unknown app(s)" check (with ITS message/format) before check_runnable ever runs --
    # check_runnable's "Unknown app" wording is capitalized differently and is only reached
    # for ids that ARE registered but not runnable.
    d = {**BASE, "enabled_apps": ["tinder"]}
    _expect_error(d, "unknown app(s)")


# --- halt_on_error: verification is not silently switchable in auto mode ------
# This key gates the driver's post-action checks ENTIRELY (_verify_progress /
# _verify_like_landed), not just what happens after one fails. With it off, a like whose
# "Send Like" tap missed returns normally, the worker records a decision for it, and
# nothing raises -- so the halt-on-unexpected path never engages either and the run keeps
# swiping while its record of what it did drifts from what actually happened. That
# corrupts the taste model, not merely the run. Until now the key had NO validation at
# all: no type check, no enum, no warning, so a stale or copy-pasted block could disable
# verification invisibly.

def test_halt_on_error_false_is_rejected_in_auto_mode():
    d = dict(BASE, mode="auto", apps={"hinge": {"halt_on_error": False}})
    _expect_error(d, "halt_on_error=false is not allowed with mode='auto'")


def test_halt_on_error_false_is_rejected_via_a_per_app_auto_override():
    # The global mode is observe, but this app overrides itself into auto -- the guard must
    # read the EFFECTIVE mode, not just the top-level one.
    d = dict(BASE, mode="observe", apps={"hinge": {"mode": "auto", "halt_on_error": False}})
    _expect_error(d, "halt_on_error=false is not allowed with mode='auto'")


def test_halt_on_error_false_is_allowed_in_observe_mode():
    # Observe is human-driven: the checks mostly guard against the bot's own missed taps,
    # and there is a person watching. Tolerable there, so don't over-restrict it.
    d = dict(BASE, mode="observe", apps={"hinge": {"halt_on_error": False}})
    c.validate(_load(d))   # no raise


def test_halt_on_error_must_be_a_boolean():
    # "false" (a string) is truthy in Python, so a quoted value would silently mean the
    # OPPOSITE of what it reads like in the YAML.
    d = dict(BASE, apps={"hinge": {"halt_on_error": "false"}})
    _expect_error(d, "must be true or false")


def test_hinge_auto_mode_is_structurally_blocked_when_halt_default_is_safe():
    d = dict(BASE, mode="auto", apps={"hinge": {}})
    _expect_error(d, "positive still-photo discriminator unavailable")


# --- enabled_apps: [] must not be silently rewritten to the default -------------------
# `raw.get("enabled_apps") or (...)` used to be the whole expression in load(): `[] or
# default` evaluates to `default`, because an empty list is falsy in Python. An operator who
# deliberately writes `enabled_apps: []` (or a config-generation bug that emits one) means
# "run nothing," and got Hinge started against the real phone instead -- silently, because
# validate()'s own `if not cfg.enabled_apps: raise ValueError(...)` guard never got a chance
# to fire: load() had already thrown the empty list away before validate() ever saw it.

def test_enabled_apps_explicit_empty_list_fails_loudly():
    d = {**BASE, "enabled_apps": []}
    cfg = _load(d)
    # Proves the bug is actually fixed at load() -- not merely that validate() has a guard
    # that was already unreachable before this fix.
    assert cfg.enabled_apps == []
    try:
        c.validate(cfg)
    except ValueError as e:
        assert "enabled_apps is empty" in str(e)
    else:
        raise AssertionError("expected ValueError for enabled_apps: []")


def test_enabled_apps_absent_key_still_defaults_to_hinge():
    d = {k: v for k, v in BASE.items() if k != "enabled_apps"}
    cfg = _load(d)
    assert cfg.enabled_apps == ["hinge"]
    c.validate(cfg)   # no raise


def test_enabled_apps_bare_null_is_treated_the_same_as_explicit_empty():
    # A bare `enabled_apps:` (YAML null) means "the key is present but nothing was written,"
    # not "the key was never mentioned" -- pinned to the same actionable failure as an
    # explicit [] rather than silently falling back to the default. This mirrors how a
    # present-but-null scalar behaves elsewhere in this file (e.g. `mode: null` does not
    # quietly become the "observe" default either -- it surfaces as a validation failure
    # downstream); it is only the {}-shaped OPTIONAL SECTIONS (budget, ranker, ...) that
    # deliberately fold null into "omitted; use defaults", because a null section changes
    # nothing about behavior, unlike a null enabled_apps.
    d = {**BASE, "enabled_apps": None}
    cfg = _load(d)
    assert cfg.enabled_apps == []
    try:
        c.validate(cfg)
    except ValueError as e:
        assert "enabled_apps is empty" in str(e)
    else:
        raise AssertionError("expected ValueError for enabled_apps: null")


def test_legacy_singular_app_key_still_works():
    d = {k: v for k, v in BASE.items() if k != "enabled_apps"}
    d["app"] = "hinge"
    cfg = _load(d)
    assert cfg.enabled_apps == ["hinge"]
    c.validate(cfg)   # no raise


def test_enabled_apps_present_and_non_empty_is_unaffected():
    # Regression: the ordinary, common case must behave exactly as before.
    d = {**BASE, "enabled_apps": ["hinge"]}
    cfg = _load(d)
    assert cfg.enabled_apps == ["hinge"]
    c.validate(cfg)   # no raise


# --- Android apps: coords entries and *_frac knobs must be real fractions in 0..1 -----
# Neither was validated anywhere before this: an out-of-range value -- a typo like 1.30 for
# 0.130, or a raw pixel written where a fraction was meant -- used to reach hinge.py's
# _assert_tap_allowed as the only backstop, and only after a driver session was already open
# on a real phone. This is the config-load-time half of a two-part fix; the sibling check,
# for a spec's own hardcoded defaults, is AndroidAppSpec.__post_init__ (see
# tests/test_android_spec.py). Bumble is used here (not Hinge) because it is the app whose
# coordinates are explicitly placeholder guesses awaiting a human typing real numbers in --
# exactly the population most likely to typo one.

def test_android_app_coords_entry_out_of_range_is_rejected():
    d = {**BASE, "enabled_apps": ["bumble"],
         "apps": {"bumble": {"coords": {"like_heart": [1.05, 0.5]}}}}
    _expect_error(d, "apps.bumble.coords.like_heart")


@pytest.mark.parametrize("value", [None, "", "com", "1com.hinge", "com.hinge-app",
                                    "com.hinge;echo injected", [], True])
def test_android_package_requires_a_safe_dotted_identifier(value):
    d = {**BASE, "apps": {"hinge": {"package": value}}}
    _expect_error(d, "apps.hinge.package")


def test_android_package_accepts_identifier_safe_segments():
    d = {**BASE, "apps": {"hinge": {"package": "co.hinge_app.v10"}}}
    c.validate(_load(d))


@pytest.mark.parametrize(("key", "value"), [
    ("scroll_captures", True), ("scroll_captures", 0),
    ("scroll_captures", 1.5), ("scroll_captures", "12"),
    ("scroll_captures", c._MAX_ANDROID_SCROLL_CAPTURES + 1),
    ("still_photo_dwell_candidates", True), ("still_photo_dwell_candidates", 0),
    ("still_photo_dwell_candidates", 1.5), ("still_photo_dwell_candidates", "3"),
    ("still_photo_dwell_candidates", c._MAX_STILL_PHOTO_DWELL_CANDIDATES + 1),
    ("dwell_s", math.nan), ("dwell_s", 0), ("dwell_s", -0.1),
    ("dwell_s", c._MAX_ANDROID_DWELL_S + 1), ("dwell_s", "1.1"),
    ("change_threshold", math.inf), ("change_threshold", 0),
    ("change_threshold", 256), ("change_threshold", "9"),
    ("debug_log", 1), ("observe_touch_watch", "false"),
    ("observe_name_ocr", None), ("touch_backend", "fallback"),
    ("touch_backend", []), ("adb_path", ""), ("debug_dir", None),
    ("serial", "   "), ("serial", 123),
])
def test_android_operational_scalars_fail_cleanly(key, value):
    d = {**BASE, "apps": {"hinge": {key: value}}}
    _expect_error(d, f"apps.hinge.{key}")


def test_android_operational_scalar_valid_boundaries_pass():
    d = {**BASE, "apps": {"hinge": {
        "scroll_captures": 1, "dwell_s": c._MIN_ANDROID_DWELL_S,
        "change_threshold": 0.1,
        "debug_log": False, "observe_touch_watch": False,
        "observe_name_ocr": True, "touch_backend": "uhid", "adb_path": "adb",
        "debug_dir": "./data/debug", "serial": None,
        "still_photo_dwell_candidates": c._MAX_STILL_PHOTO_DWELL_CANDIDATES,
    }}}
    c.validate(_load(d))


def test_touch_backend_accepts_uhid_persistent():
    # 2026-08-24: a fourth explicit touch_backend value (operation_love/drivers/uhid.py's
    # PersistentUhidTouch) alongside auto/uhid/adb -- an invalid value must still fail the
    # same clean way (test_android_operational_scalars_fail_cleanly's "fallback"/[] cases,
    # unchanged above), and this one specific new value must now be accepted.
    d = {**BASE, "apps": {"hinge": {"touch_backend": "uhid_persistent"}}}
    c.validate(_load(d))


def test_still_photo_dwell_candidates_huge_integer_reported_as_config_error():
    # Mirrors test_bounded_android_numbers_report_huge_integers_as_config_errors: a bignum must
    # fail the SAME clean way scroll_captures already does, not overflow or hang isinstance/int
    # comparison machinery.
    huge = 1 << 20_000
    cfg = _load(BASE)
    cfg.apps = {"hinge": {"still_photo_dwell_candidates": huge}}
    with pytest.raises(ValueError, match="still_photo_dwell_candidates"):
        c.validate(cfg)


def test_android_app_coords_entry_negative_is_rejected():
    d = {**BASE, "enabled_apps": ["bumble"],
         "apps": {"bumble": {"coords": {"pass_x": [0.5, -0.2]}}}}
    _expect_error(d, "apps.bumble.coords.pass_x")


def test_android_app_coords_entry_must_be_an_xy_pair():
    d = {**BASE, "enabled_apps": ["bumble"],
         "apps": {"bumble": {"coords": {"like_heart": [0.5, 0.5, 0.5]}}}}
    _expect_error(d, "apps.bumble.coords.like_heart")


@pytest.mark.parametrize("value", [[], False, 0, "", ()])
def test_supplied_falsy_android_coords_must_still_be_a_mapping(value):
    cfg = _load(BASE)
    cfg.apps = {"hinge": {"coords": value}}

    with pytest.raises(ValueError, match=r"apps\.hinge\.coords must be a mapping"):
        c.validate(cfg)


def test_bounded_android_numbers_report_huge_integers_as_config_errors():
    huge = 1 << 20_000
    cfg = _load(BASE)
    cfg.apps = {"hinge": {"scroll_captures": huge}}
    with pytest.raises(ValueError, match="scroll_captures"):
        c.validate(cfg)

    cfg.apps = {"hinge": {"coords": {"like_heart": [huge, 0.5]}}}
    with pytest.raises(ValueError, match="like_heart"):
        c.validate(cfg)


def test_android_app_frac_setting_out_of_range_is_rejected():
    # The other demonstrated exploit path: apps.bumble.read_scroll_frac=1.30 alone (no
    # coords entry at all) pushes an ordinary read-scroll's touch-down off-screen.
    d = {**BASE, "enabled_apps": ["bumble"], "apps": {"bumble": {"read_scroll_frac": 1.30}}}
    _expect_error(d, "apps.bumble.read_scroll_frac")


def test_android_app_frac_setting_rejects_a_bool():
    # bool is an int subclass in Python -- the same trap this file already guards against
    # for opener.max_attempts / opener.thinking[...].thinkingBudget.
    d = {**BASE, "enabled_apps": ["bumble"], "apps": {"bumble": {"read_scroll_frac": True}}}
    _expect_error(d, "apps.bumble.read_scroll_frac")


def test_android_app_frac_and_coords_within_range_pass():
    d = {**BASE, "enabled_apps": ["bumble"],
         "apps": {"bumble": {"read_scroll_frac": 0.6,
                             "coords": {"like_heart": [0.85, 0.9]}}}}
    c.validate(_load(d))   # no raise


def test_shipped_hinge_and_bumble_app_blocks_pass_fraction_validation():
    # Regression pin: the real config.yaml's apps.hinge/apps.bumble coords and
    # read_scroll_frac must stay valid under this check (also exercised end-to-end by
    # tests/test_config_yaml_real.py against the actual shipped file).
    d = {**BASE, "enabled_apps": ["hinge"],
         "apps": {
             "hinge": {"read_scroll_frac": 0.55,
                       "coords": {"like_heart": [0.868, 0.667], "pass_x": [0.116, 0.848]}},
             "bumble": {"coords": {"swipe_start": [0.50, 0.55],
                                   "swipe_like_end": [0.92, 0.52],
                                   "swipe_pass_end": [0.08, 0.52]}},
         }}
    c.validate(_load(d))   # no raise


# --- Per-item targeting calibration: evidence-backed, fail closed at the driver --------

def test_stale_targeting_calibration_cannot_license_unavailable_still_photo_policy():
    d = {**BASE, "apps": {"hinge": {"serial": "synthetic-pixel", "targeting_calibration": {
        "identity_match_max_dist": 2.0, "inline_item_max_dist": 4.0,
        "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
    }}}}
    _expect_error(d, "positive still-photo discriminator unavailable")


def test_hinge_observe_without_optional_targeting_calibration_remains_valid():
    d = {**BASE, "mode": "observe", "apps": {"hinge": {"serial": "synthetic-pixel"}}}

    c.validate(_load(d))


def test_targeting_calibration_must_be_complete_and_stay_below_known_false_accepts():
    incomplete = {**BASE, "apps": {"hinge": {"targeting_calibration": {
        **_TARGETING_SCHEMA_V2,
        "identity_match_max_dist": 2.0,
    }}}}
    _expect_error(incomplete, "targeting_calibration")
    unsafe = {**BASE, "apps": {"hinge": {"serial": "synthetic-pixel", "targeting_calibration": {
        "identity_match_max_dist": 2.565, "inline_item_max_dist": 4.0,
        "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
    }}}}
    _expect_error(unsafe, "strictly below 2.565")
    unsafe_sheet = {**BASE, "apps": {"hinge": {"serial": "synthetic-pixel", "targeting_calibration": {
        "identity_match_max_dist": 2.0, "inline_item_max_dist": 14.91,
        "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
    }}}}
    _expect_error(unsafe_sheet, "strictly below 14.91")


def test_targeting_calibration_rejects_the_legacy_six_key_mapping():
    """Schema-v1 evidence cannot license Hinge's inline composer geometry."""
    legacy = {
        "identity_match_max_dist": 2.0,
        "inline_item_max_dist": 4.0,
        "device": "synthetic-pixel",
        "calibrated_at": "2026-08-12",
        **_TARGETING_GEOMETRY,
    }
    d = {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel", "targeting_calibration": legacy,
    }}}
    _expect_error(d, "schema_version")


def test_targeting_calibration_rejects_nonfinite_bounds_and_empty_evidence():
    d = {**BASE, "apps": {"hinge": {"serial": "synthetic-pixel", "targeting_calibration": {
        "identity_match_max_dist": 2.0, "inline_item_max_dist": float("inf"),
        "device": "", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
    }}}}
    _expect_error(d, "inline_item_max_dist")


def test_targeting_calibration_requires_its_exact_adb_serial():
    missing = {**BASE, "apps": {"hinge": {"targeting_calibration": {
        "identity_match_max_dist": 2.0, "inline_item_max_dist": 4.0,
        "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
    }}}}
    _expect_error(missing, "apps.hinge.serial")
    mismatched = {**BASE, "apps": {"hinge": {"serial": "other-pixel", "targeting_calibration": {
        "identity_match_max_dist": 2.0, "inline_item_max_dist": 4.0,
        "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
    }}}}
    _expect_error(mismatched, "must exactly equal")


def test_targeting_calibration_must_bind_the_effective_identity_and_content_bands():
    mismatch = {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel", "identity_band": [0.11, 0.048, 0.80, 0.094],
        "targeting_calibration": {
            "identity_match_max_dist": 2.0, "inline_item_max_dist": 4.0,
            "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
            **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY,
        },
    }}}
    _expect_error(mismatch, "must exactly equal the effective")
    matching = {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel", "identity_band": [0.11, 0.048, 0.80, 0.094],
        "content_band": [0.13, 0.87],
        "targeting_calibration": {
            "identity_match_max_dist": 2.0, "inline_item_max_dist": 4.0,
            "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
            **_TARGETING_SCHEMA_V2,
            "identity_band": [0.11, 0.048, 0.80, 0.094], "content_band": [0.13, 0.87],
        },
    }}}
    _expect_error(matching, "positive still-photo discriminator unavailable")


# --- Artifact-derived still-photo numbering readiness ---------------------------------
# ops/STILL-PHOTO-DISCRIMINATOR.md section 5. There is no hand-editable ready boolean any
# more: apps.hinge.still_photo_bound_evidence is the ONLY thing that can turn numbering on,
# and it has to survive a sha256-bound on-disk artifact check first.

_BOUND_ARTIFACT = {
    "ground_truth_channel": "owner_tap_to_play_v1",
    "human_ground_truth": True,
    "video_cards": 60,
    "video_accepts": 0,
    "photo_cards": 60,
    "photo_false_refusals": 3,
    "max_video_exact_run_s": 1.5,
    "captured_at": "2026-08-21T00:00:00Z",
    "device": "synthetic-pixel",
    "hinge_version_name": "10.0.1",
}
_BOUND_MIRRORED_KEYS = ("ground_truth_channel", "video_cards", "video_accepts", "photo_cards",
                        "photo_false_refusals", "max_video_exact_run_s", "captured_at",
                        "device", "hinge_version_name")
_BOUND_REL_PATH = "ops/calibration/still_photo_bound.json"


@pytest.fixture(autouse=True)
def _still_photo_readiness_is_never_inherited():
    """Numbering readiness is process-global: never let one test license the next one."""
    tp._reset_installed_still_photo_bound_for_tests()
    yield
    tp._reset_installed_still_photo_bound_for_tests()


def _write_bound_artifact(root, **overrides):
    """Write a bound artifact under `root` and return its (repo-relative path, sha256, body)."""
    artifact = {**_BOUND_ARTIFACT, **overrides}
    out = root / _BOUND_REL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(artifact, sort_keys=True).encode("utf-8")
    out.write_bytes(raw)
    return _BOUND_REL_PATH, hashlib.sha256(raw).hexdigest(), artifact


def _bound_evidence(path, digest, artifact, **overrides):
    """The config mapping that binds that artifact, before any deliberate corruption."""
    mapping = {"artifact_path": path, "artifact_sha256": digest}
    mapping.update({key: artifact[key] for key in _BOUND_MIRRORED_KEYS})
    mapping.update(overrides)
    return mapping


def _calibration(**overrides):
    """A complete, valid v2 targeting calibration for the synthetic pixel."""
    return {
        "identity_match_max_dist": 2.0, "inline_item_max_dist": 4.0,
        "device": "synthetic-pixel", "calibrated_at": "2026-08-12",
        **_TARGETING_SCHEMA_V2, **_TARGETING_GEOMETRY, **overrides,
    }


def _bound_config(tmp_path, monkeypatch, *, evidence=None, artifact_overrides=None, **app):
    """A hinge config whose still-photo evidence points at a freshly written artifact."""
    monkeypatch.chdir(tmp_path)
    path, digest, artifact = _write_bound_artifact(tmp_path, **(artifact_overrides or {}))
    mapping = _bound_evidence(path, digest, artifact) if evidence is None else evidence(
        path, digest, artifact)
    return {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel", "still_photo_bound_evidence": mapping, **app,
    }}}


def test_no_bound_evidence_key_leaves_numbering_disabled_exactly_as_before():
    c.validate(_load(BASE))

    assert tp.installed_still_photo_bound() is None
    assert tp.hinge_targeting_unavailable_reason() == tp.HINGE_TARGETING_UNAVAILABLE_REASON


def test_verified_bound_evidence_installs_readiness_and_licenses_the_calibration(
        tmp_path, monkeypatch):
    d = _bound_config(tmp_path, monkeypatch, targeting_calibration=_calibration())

    c.validate(_load(d))   # no raise: the policy blocker is what used to reject this mapping

    summary = tp.installed_still_photo_bound()
    assert summary == tp.StillPhotoBoundSummary(
        ground_truth_channel="owner_tap_to_play_v1", human_ground_truth=True,
        video_cards=60, video_accepts=0, photo_cards=60, photo_false_refusals=3,
        max_video_exact_run_s=1.5, artifact_sha256=summary.artifact_sha256,
        device="synthetic-pixel", hinge_version_name="10.0.1")
    assert tp.hinge_targeting_unavailable_reason() is None


def test_a_later_validate_without_the_key_turns_readiness_back_off(tmp_path, monkeypatch):
    """Process-global readiness must never outlive the config that presented the evidence."""
    c.validate(_load(_bound_config(tmp_path, monkeypatch)))
    assert tp.installed_still_photo_bound() is not None

    c.validate(_load(BASE))

    assert tp.installed_still_photo_bound() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_a_failing_later_validate_also_turns_readiness_back_off(tmp_path, monkeypatch):
    c.validate(_load(_bound_config(tmp_path, monkeypatch)))

    _expect_error({**BASE, "enabled_apps": ["tinder"]}, "unknown app")

    assert tp.installed_still_photo_bound() is None


def test_bound_evidence_must_carry_its_exact_key_set(tmp_path, monkeypatch):
    incomplete = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: {
            k: v for k, v in _bound_evidence(path, digest, artifact).items()
            if k != "photo_cards"})
    _expect_error(incomplete, "missing ['photo_cards']")
    unknown = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, note="looks fine to me"))
    _expect_error(unknown, "unknown ['note']")


def test_bound_evidence_requires_the_artifact_to_exist_on_disk(tmp_path, monkeypatch):
    d = _bound_config(tmp_path, monkeypatch)
    (tmp_path / _BOUND_REL_PATH).unlink()

    _expect_error(d, "artifact is unreadable")


def test_bound_evidence_refuses_an_absolute_or_escaping_artifact_path(tmp_path, monkeypatch):
    absolute = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, artifact_path=str(tmp_path / path)))
    _expect_error(absolute, "must be a repo-relative local path")
    escaping = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, artifact_path="../" + path))
    _expect_error(escaping, "escapes the repository")


def test_bound_evidence_digest_must_match_the_artifact_bytes(tmp_path, monkeypatch):
    d = _bound_config(tmp_path, monkeypatch)
    (tmp_path / _BOUND_REL_PATH).write_text("{}")

    _expect_error(d, "artifact_sha256 does not match its artifact")


def test_bound_evidence_cannot_claim_numbers_the_artifact_does_not_carry(tmp_path, monkeypatch):
    """A pasted mapping is a claim; the digest-bound artifact is the evidence."""
    inflated = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, video_cards=600))
    _expect_error(inflated, "artifact disagrees on video_cards")
    retimed = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, max_video_exact_run_s=0.25))
    _expect_error(retimed, "artifact disagrees on max_video_exact_run_s")
    relabelled = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, hinge_version_name="9.134.0"))
    _expect_error(relabelled, "artifact disagrees on hinge_version_name")


@pytest.mark.parametrize("overrides", [
    {"human_ground_truth": False},
    {"human_ground_truth": "yes"},
])
def test_bound_evidence_requires_the_artifacts_own_human_ground_truth(
        tmp_path, monkeypatch, overrides):
    """The one claim nothing downstream can re-derive is not a config key at all."""
    d = _bound_config(tmp_path, monkeypatch, artifact_overrides=overrides)

    _expect_error(d, "must declare human_ground_truth=true")


def test_bound_evidence_requires_the_owner_tap_to_play_label_channel(tmp_path, monkeypatch):
    d = _bound_config(
        tmp_path, monkeypatch,
        artifact_overrides={"ground_truth_channel": "mute_matcher_v1"},
        evidence=lambda path, digest, artifact: _bound_evidence(path, digest, artifact))

    _expect_error(d, "owner_tap_to_play_v1")


def test_bound_evidence_device_must_equal_the_exact_adb_serial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, digest, artifact = _write_bound_artifact(tmp_path, device="other-pixel")
    mismatched = {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel",
        "still_photo_bound_evidence": _bound_evidence(path, digest, artifact),
    }}}
    _expect_error(mismatched, "must exactly equal apps.hinge.serial")

    d = _bound_config(tmp_path, monkeypatch)
    d["apps"]["hinge"].pop("serial")
    _expect_error(d, "requires a nonempty apps.hinge.serial")


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"video_cards": 59}, "video_cards must be an integer >= 60"),
    ({"video_accepts": 1}, "video_accepts must be exactly 0"),
    ({"photo_cards": 59, "photo_false_refusals": 0}, "photo_cards must be an integer >= 60"),
    ({"photo_false_refusals": 4}, "photo_false_refusals 4 exceeds"),
])
def test_bound_evidence_enforces_the_shippable_thresholds_end_to_end(
        tmp_path, monkeypatch, overrides, needle):
    """The install API's thresholds surface through config with their own reason attached."""
    d = _bound_config(
        tmp_path, monkeypatch, artifact_overrides=overrides,
        evidence=lambda path, digest, artifact: _bound_evidence(path, digest, artifact))

    _expect_error(d, "is not a shippable bound")
    _expect_error(d, needle)


@pytest.mark.parametrize(("key", "value"), [
    ("video_cards", True),
    ("video_accepts", 1.0),
    ("photo_cards", "60"),
    ("max_video_exact_run_s", float("inf")),
    ("captured_at", ""),
])
def test_bound_evidence_rejects_malformed_scalars_before_reading_the_artifact(
        tmp_path, monkeypatch, key, value):
    d = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, **{key: value}))

    _expect_error(d, f"still_photo_bound_evidence.{key}")


def test_bound_evidence_must_be_a_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = {**BASE, "apps": {"hinge": {"serial": "synthetic-pixel",
                                    "still_photo_bound_evidence": "trust me"}}}

    _expect_error(d, "still_photo_bound_evidence must be a mapping")


# --- The opted-in circular AI-labelled channel ----------------------------------------
# Owner decision 2026-08-21: a second channel whose video labels come from the mute-glyph
# matcher the bound is supposed to bound. It is admitted only with the acceptance phrase in
# BOTH the pasted mapping and the digest-bound artifact, and it may never claim human ground
# truth. The owner-labelled channel above is untouched and stays the preferred one.

_CIRCULAR_ARTIFACT = {
    "ground_truth_channel": tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL,
    "human_ground_truth": False,
    "accepted_circular_risk": tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE,
}


def _circular_config(tmp_path, monkeypatch, *, artifact_overrides=None, mapping=None, **app):
    """A hinge config on the circular channel; `mapping` overrides the pasted evidence keys."""
    if mapping is None:
        mapping = {"accepted_circular_risk": tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE}
    return _bound_config(
        tmp_path, monkeypatch,
        artifact_overrides={**_CIRCULAR_ARTIFACT, **(artifact_overrides or {})},
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact, **mapping),
        **app)


def test_an_accepted_circular_bound_round_trips_through_config_and_licenses_numbering(
        tmp_path, monkeypatch):
    d = _circular_config(tmp_path, monkeypatch)

    c.validate(_load(d))

    summary = tp.installed_still_photo_bound()
    assert summary is not None
    assert summary.ground_truth_channel == tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL
    # Recorded honestly on the installed summary: this bound has no human labels at all.
    assert summary.human_ground_truth is False
    assert summary.accepted_circular_risk == tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE
    assert tp.hinge_targeting_unavailable_reason() is None

    # The autouse fixture resets readiness, but assert the same drop config validation performs
    # so the circular channel cannot outlive the config that accepted it either.
    c.validate(_load(BASE))
    assert tp.installed_still_photo_bound() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_a_circular_mapping_without_the_acceptance_phrase_is_refused(tmp_path, monkeypatch):
    """Silence is not acceptance: the key set makes it optional, the channel makes it required."""
    d = _circular_config(tmp_path, monkeypatch, mapping={})

    _expect_error(d, "accepted_circular_risk must be exactly")
    _expect_error(d, tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE)
    _expect_error(d, tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL)


@pytest.mark.parametrize("phrase", [
    "I accept the circular risk",
    tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE.lower(),
    tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE + " ",
    True,
])
def test_a_circular_mapping_needs_the_acceptance_phrase_byte_for_byte(
        tmp_path, monkeypatch, phrase):
    d = _circular_config(tmp_path, monkeypatch, mapping={"accepted_circular_risk": phrase})

    _expect_error(d, "accepted_circular_risk must be exactly")


def test_the_owner_channel_refuses_an_accepted_circular_risk_key(tmp_path, monkeypatch):
    """An owner-labelled bound accepted nothing, so it may not look like it accepted something."""
    d = _bound_config(
        tmp_path, monkeypatch,
        evidence=lambda path, digest, artifact: _bound_evidence(
            path, digest, artifact,
            accepted_circular_risk=tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE))

    _expect_error(d, "accepted_circular_risk accepts the circular AI-labelled channel")
    _expect_error(d, "and is valid only there")
    _expect_error(d, "owner_tap_to_play_v1")


def test_a_circular_artifact_may_never_claim_human_ground_truth(tmp_path, monkeypatch):
    d = _circular_config(tmp_path, monkeypatch, artifact_overrides={"human_ground_truth": True})

    _expect_error(d, "must declare human_ground_truth=false")
    _expect_error(d, "is a lie")


@pytest.mark.parametrize("artifact_overrides", [
    {"accepted_circular_risk": "some other phrase"},
    {"accepted_circular_risk": None},
])
def test_the_artifact_must_carry_the_same_acceptance_as_the_mapping(
        tmp_path, monkeypatch, artifact_overrides):
    """A config edit alone can never opt the owner in: the measurement run has to say it too."""
    d = _circular_config(tmp_path, monkeypatch, artifact_overrides=artifact_overrides)

    _expect_error(d, "artifact disagrees on accepted_circular_risk")


def test_an_unaccepted_artifact_channel_names_both_admissible_channels(tmp_path, monkeypatch):
    """The mapping and artifact agree on a channel that is neither of the two we accept."""
    d = _bound_config(
        tmp_path, monkeypatch,
        artifact_overrides={"ground_truth_channel": "mute_matcher_v1"},
        evidence=lambda path, digest, artifact: _bound_evidence(path, digest, artifact))

    _expect_error(d, "owner_tap_to_play_v1")
    _expect_error(d, tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL)


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"video_cards": 59}, "video_cards must be an integer >= 60"),
    ({"video_accepts": 1}, "video_accepts must be exactly 0"),
    ({"photo_cards": 59, "photo_false_refusals": 0}, "photo_cards must be an integer >= 60"),
    ({"photo_false_refusals": 4}, "photo_false_refusals 4 exceeds"),
])
def test_the_circular_channel_buys_a_cheaper_campaign_never_a_looser_bound(
        tmp_path, monkeypatch, overrides, needle):
    """Accepting the circularity moves no number: same corpus sizes, same zero accepts."""
    d = _circular_config(tmp_path, monkeypatch, artifact_overrides=overrides)

    _expect_error(d, "is not a shippable bound")
    _expect_error(d, needle)


# --- Gate split: a bound licenses numbering, never Auto -------------------------------

def test_a_bound_alone_never_satisfies_the_hinge_auto_release_gate(tmp_path, monkeypatch):
    """Numbering readiness removes the policy blocker and nothing else (design doc, section 3)."""
    d = _bound_config(tmp_path, monkeypatch, mode="auto",
                      targeting_calibration=_calibration())

    _expect_error(d, "observe_release_evidence")

    d_no_bound = {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel", "mode": "auto",
        "targeting_calibration": _calibration(),
    }}}
    _expect_error(d_no_bound, "positive still-photo discriminator unavailable")


# --- The retired v1 selection policy --------------------------------------------------

def test_targeting_calibration_rejects_the_superseded_v1_policy_id(tmp_path, monkeypatch):
    """v1 measured a different selection contract, so its mapping can never be reinstalled."""
    d = _bound_config(
        tmp_path, monkeypatch,
        targeting_calibration=_calibration(item_selection_policy_id="hinge_photos_only_v1"))

    _expect_error(d, "hinge_photos_only_v1 is superseded by hinge_photos_only_v2")
    _expect_error(d, "recalibrate under the current policy")


def test_targeting_calibration_reports_an_unknown_policy_id_differently(tmp_path, monkeypatch):
    d = _bound_config(
        tmp_path, monkeypatch,
        targeting_calibration=_calibration(item_selection_policy_id="hinge_written_only_v9"))

    _expect_error(d, "item_selection_policy_id must be 'hinge_photos_only_v2'")


# --- install_verified_still_photo_bound(): the threshold matrix ------------------------
# Tested directly as well as through config because this function, not the config schema, is
# what every future consumer (the measurement tool included) has to clear.

def _summary(**overrides):
    fields = {
        "ground_truth_channel": "owner_tap_to_play_v1", "human_ground_truth": True,
        "video_cards": 60, "video_accepts": 0, "photo_cards": 60, "photo_false_refusals": 3,
        "max_video_exact_run_s": 1.5, "artifact_sha256": "a" * 64,
        "device": "synthetic-pixel", "hinge_version_name": "10.0.1",
    }
    fields.update(overrides)
    return tp.StillPhotoBoundSummary(**fields)


def test_a_valid_summary_installs_and_answers_the_policy_blocker():
    tp.install_verified_still_photo_bound(_summary())

    assert tp.installed_still_photo_bound() == _summary()
    assert tp.hinge_targeting_unavailable_reason() is None


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"ground_truth_channel": "mute_matcher_v1"}, "ground_truth_channel must be"),
    ({"human_ground_truth": False}, "human_ground_truth must be exactly True"),
    ({"human_ground_truth": 1}, "human_ground_truth must be exactly True"),
    ({"video_cards": 59}, "video_cards must be an integer >= 60"),
    ({"video_cards": 60.0}, "video_cards must be an integer >= 60"),
    ({"video_accepts": 1}, "video_accepts must be exactly 0"),
    ({"video_accepts": -1}, "video_accepts must be a non-negative integer"),
    ({"photo_cards": 59, "photo_false_refusals": 0}, "photo_cards must be an integer >= 60"),
    ({"photo_false_refusals": 4}, "photo_false_refusals 4 exceeds"),
    ({"photo_false_refusals": -1}, "photo_false_refusals must be a non-negative integer"),
    ({"max_video_exact_run_s": -0.1}, "max_video_exact_run_s must be a finite"),
    ({"max_video_exact_run_s": float("inf")}, "max_video_exact_run_s must be a finite"),
    ({"max_video_exact_run_s": float("nan")}, "max_video_exact_run_s must be a finite"),
    ({"artifact_sha256": "A" * 64}, "artifact_sha256 must be 64 lowercase hex"),
    ({"artifact_sha256": "a" * 63}, "artifact_sha256 must be 64 lowercase hex"),
    ({"device": ""}, "device must be the nonempty ADB serial"),
    ({"device": "   "}, "device must be the nonempty ADB serial"),
    ({"hinge_version_name": ""}, "hinge_version_name must be the nonempty"),
])
def test_install_refuses_every_unmet_threshold_with_its_own_reason(overrides, needle):
    with pytest.raises(ValueError, match=re.escape(needle)):
        tp.install_verified_still_photo_bound(_summary(**overrides))

    assert tp.installed_still_photo_bound() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_the_false_refusal_ceiling_is_a_fraction_of_the_photo_corpus():
    """The ceiling scales with the corpus: 4 refusals pass on 80 photos and fail on 60."""
    tp.install_verified_still_photo_bound(_summary(photo_cards=60, photo_false_refusals=3))
    tp._reset_installed_still_photo_bound_for_tests()
    tp.install_verified_still_photo_bound(_summary(photo_cards=80, photo_false_refusals=4))
    tp._reset_installed_still_photo_bound_for_tests()

    with pytest.raises(ValueError, match="exceeds 5%"):
        tp.install_verified_still_photo_bound(_summary(photo_cards=60, photo_false_refusals=4))
    with pytest.raises(ValueError, match="exceeds 5%"):
        tp.install_verified_still_photo_bound(_summary(photo_cards=80, photo_false_refusals=5))
    # A corpus that is too small never reaches the fraction check at all: the minimum-cards
    # threshold owns that refusal, and its message has to say so.
    with pytest.raises(ValueError, match="photo_cards must be an integer >= 60"):
        tp.install_verified_still_photo_bound(_summary(photo_cards=59, photo_false_refusals=0))


def test_reinstalling_the_identical_summary_is_a_no_op():
    tp.install_verified_still_photo_bound(_summary())
    tp.install_verified_still_photo_bound(_summary())

    assert tp.installed_still_photo_bound() == _summary()


def test_installing_a_different_summary_never_silently_swaps_the_licence():
    tp.install_verified_still_photo_bound(_summary())

    with pytest.raises(ValueError, match="different verified still-photo bound"):
        tp.install_verified_still_photo_bound(_summary(video_cards=120))

    assert tp.installed_still_photo_bound() == _summary()


def test_install_refuses_anything_that_is_not_a_summary():
    with pytest.raises(ValueError, match="must be a StillPhotoBoundSummary"):
        tp.install_verified_still_photo_bound({"video_cards": 60})


# --- install_verified_still_photo_bound(): the circular channel ------------------------

def _circular_summary(**overrides):
    """The same measured bound, declared on the opted-in circular AI-labelled channel."""
    fields = {
        "ground_truth_channel": tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL,
        "human_ground_truth": False,
        "accepted_circular_risk": tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE,
    }
    fields.update(overrides)
    return _summary(**fields)


def test_the_circular_channel_installs_with_the_phrase_and_an_honest_label():
    tp.install_verified_still_photo_bound(_circular_summary())

    assert tp.installed_still_photo_bound() == _circular_summary()
    assert tp.hinge_targeting_unavailable_reason() is None


def test_the_owner_channel_summary_still_defaults_to_having_accepted_nothing():
    """The new field is additive: every existing owner-labelled construction is unchanged."""
    assert _summary().accepted_circular_risk is None

    tp.install_verified_still_photo_bound(_summary())

    assert tp.installed_still_photo_bound().accepted_circular_risk is None


@pytest.mark.parametrize(("overrides", "needle"), [
    # The one refusal the whole channel exists to make: an AI-labelled bound that claims a human
    # labelled it is not a weaker bound, it is a false statement.
    ({"human_ground_truth": True}, "claiming human ground truth there is a lie"),
    ({"human_ground_truth": 0}, "human_ground_truth must be exactly False"),
    ({"accepted_circular_risk": None}, "accepted_circular_risk must be exactly"),
    ({"accepted_circular_risk": "I accept the circular risk"},
     "accepted_circular_risk must be exactly"),
    ({"accepted_circular_risk": tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE.lower()},
     "accepted_circular_risk must be exactly"),
    ({"accepted_circular_risk": tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE + " "},
     "accepted_circular_risk must be exactly"),
    # Identical thresholds on both channels: accepting the circularity buys a cheaper campaign,
    # never a smaller corpus or a nonzero accept count.
    ({"video_cards": 59}, "video_cards must be an integer >= 60"),
    ({"video_accepts": 1}, "video_accepts must be exactly 0"),
    ({"photo_cards": 59, "photo_false_refusals": 0}, "photo_cards must be an integer >= 60"),
    ({"photo_false_refusals": 4}, "photo_false_refusals 4 exceeds"),
])
def test_the_circular_channel_refuses_every_unmet_condition_with_its_own_reason(
        overrides, needle):
    with pytest.raises(ValueError, match=re.escape(needle)):
        tp.install_verified_still_photo_bound(_circular_summary(**overrides))

    assert tp.installed_still_photo_bound() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_the_owner_channel_never_carries_a_circular_acceptance():
    with pytest.raises(ValueError, match="accepted_circular_risk must be None"):
        tp.install_verified_still_photo_bound(
            _summary(accepted_circular_risk=tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE))

    assert tp.installed_still_photo_bound() is None


def test_an_unknown_channel_names_both_admissible_channels():
    with pytest.raises(ValueError, match="ground_truth_channel must be") as exc:
        tp.install_verified_still_photo_bound(_summary(ground_truth_channel="mute_matcher_v1"))

    assert tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL in str(exc.value)
    assert tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL in str(exc.value)


# --- the THIRD readiness channel: an accepted assumption, not a measurement ------------
# Owner decision 2026-08-21: measuring the held-out video false-accept rate costs ~420 profiles
# and ~420 real passes, which the owner judged not worth paying, and directed instead that a
# video moved to the centre of the screen is ASSUMED to be playing (and therefore visible to the
# deterministic motion test). These tests exist to pin that the licence stays legible as an
# assumption at every layer -- never installable as, comparable to, or reportable as a bound.

def _acceptance(**overrides):
    """The mapping an owner pastes to ship numbering on the unmeasured assumption."""
    mapping = {
        "acceptance": tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION,
        "accepted_at": "2026-08-21",
        "device": "synthetic-pixel",
        "hinge_version_name": "10.0.1",
        "rationale": ("owner judged a ~420-profile held-out campaign not worth its cost and "
                      "directed that a centred video is assumed to be playing"),
    }
    mapping.update(overrides)
    return mapping


def _acceptance_record(**overrides):
    return tp.StillPhotoAssumptionAcceptance(**_acceptance(**overrides))


def _assumption_config(*, acceptance=None, **app):
    """A hinge config licensed by the assumption instead of by a measured artifact."""
    return {**BASE, "apps": {"hinge": {
        "serial": "synthetic-pixel",
        "still_photo_assumption_acceptance": (
            _acceptance() if acceptance is None else acceptance),
        **app,
    }}}


def test_an_accepted_assumption_installs_readiness_and_names_itself_the_assumption_channel():
    tp.install_accepted_still_photo_assumption(_acceptance_record())

    licence = tp.installed_still_photo_licence()
    assert licence is not None
    assert licence.channel == tp.STILL_PHOTO_LICENCE_ASSUMPTION
    assert licence.assumed and not licence.measured
    assert licence.record == _acceptance_record()
    assert tp.hinge_targeting_unavailable_reason() is None
    # The one thing that must never be true: an assumption readable as a measured bound.
    assert tp.installed_still_photo_bound() is None


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"acceptance": "I accept the unmeasured centered autoplay assumption"},
     "acceptance must be exactly"),
    ({"acceptance": tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION.lower()},
     "acceptance must be exactly"),
    ({"acceptance": tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION + " "},
     "acceptance must be exactly"),
    ({"acceptance": tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE}, "acceptance must be exactly"),
    ({"accepted_at": ""}, "accepted_at must be nonempty text"),
    ({"accepted_at": "   "}, "accepted_at must be nonempty text"),
    ({"device": ""}, "device must be the nonempty ADB serial"),
    ({"hinge_version_name": ""}, "hinge_version_name must be the nonempty"),
    ({"rationale": ""}, "rationale must be the nonempty reason"),
    ({"rationale": "  "}, "rationale must be the nonempty reason"),
])
def test_installing_an_assumption_refuses_every_unmet_field_with_its_own_reason(
        overrides, needle):
    with pytest.raises(ValueError, match=re.escape(needle)):
        tp.install_accepted_still_photo_assumption(_acceptance_record(**overrides))

    assert tp.installed_still_photo_licence() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_installing_an_assumption_refuses_anything_that_is_not_an_acceptance():
    with pytest.raises(ValueError, match="must be a StillPhotoAssumptionAcceptance"):
        tp.install_accepted_still_photo_assumption(_acceptance())

    with pytest.raises(ValueError, match="must be a StillPhotoAssumptionAcceptance"):
        tp.install_accepted_still_photo_assumption(_summary())


def test_reinstalling_the_identical_assumption_is_a_no_op():
    tp.install_accepted_still_photo_assumption(_acceptance_record())
    tp.install_accepted_still_photo_assumption(_acceptance_record())

    assert tp.installed_still_photo_licence().record == _acceptance_record()


def test_installing_a_different_assumption_never_silently_swaps_the_licence():
    tp.install_accepted_still_photo_assumption(_acceptance_record())

    with pytest.raises(ValueError, match="different accepted still-photo assumption"):
        tp.install_accepted_still_photo_assumption(_acceptance_record(accepted_at="2026-09-01"))

    assert tp.installed_still_photo_licence().record == _acceptance_record()


def test_an_assumption_cannot_be_installed_beside_a_measured_bound():
    """One licence at a time: an assumption must never shadow a bound somebody measured."""
    tp.install_verified_still_photo_bound(_summary())

    with pytest.raises(ValueError, match="one licence at a time") as exc:
        tp.install_accepted_still_photo_assumption(_acceptance_record())

    assert tp.STILL_PHOTO_LICENCE_MEASURED in str(exc.value)
    assert tp.STILL_PHOTO_LICENCE_ASSUMPTION in str(exc.value)
    assert tp.installed_still_photo_bound() == _summary()


def test_a_measured_bound_cannot_be_installed_beside_an_assumption():
    """And the mirror: a bound must not be quietly swapped in under an accepted assumption."""
    tp.install_accepted_still_photo_assumption(_acceptance_record())

    with pytest.raises(ValueError, match="one licence at a time"):
        tp.install_verified_still_photo_bound(_summary())

    assert tp.installed_still_photo_licence().channel == tp.STILL_PHOTO_LICENCE_ASSUMPTION
    assert tp.installed_still_photo_bound() is None


def test_clearing_drops_an_assumption_exactly_like_it_drops_a_bound():
    tp.install_accepted_still_photo_assumption(_acceptance_record())

    tp.clear_installed_still_photo_bound()

    assert tp.installed_still_photo_licence() is None
    assert tp.hinge_targeting_unavailable_reason() == tp.HINGE_TARGETING_UNAVAILABLE_REASON
    # And the slot is free again, so the other channel can now take it.
    tp.install_verified_still_photo_bound(_summary())
    assert tp.installed_still_photo_bound() == _summary()


# --- provenance: the difference between the channels survives only in words ------------

def test_provenance_reports_nothing_while_numbering_is_unlicensed():
    assert tp.still_photo_licence_provenance() is None
    assert tp.still_photo_licence_operator_notice() is None


def test_provenance_of_a_measured_bound_quotes_the_corpus_and_claims_no_more():
    tp.install_verified_still_photo_bound(_summary())

    provenance = tp.still_photo_licence_provenance()

    assert provenance == "measured held-out bound (60 video cards, 0 accepts)"
    assert "UNMEASURED" not in provenance
    # A measured bound is the state the design doc assumes, so it produces no run-level notice:
    # a banner on every ordinary run is how operators learn to stop reading banners.
    assert tp.still_photo_licence_operator_notice() is None


def test_provenance_of_a_circular_bound_says_the_accept_count_is_zero_by_construction():
    """Quoting "0 accepts" from the circular channel without that clause overstates it."""
    tp.install_verified_still_photo_bound(_circular_summary())

    provenance = tp.still_photo_licence_provenance()

    assert provenance.startswith("measured held-out bound (60 video cards, 0 accepts)")
    assert "circular AI-labelled channel" in provenance
    assert "zero by construction" in provenance
    assert tp.still_photo_licence_operator_notice() is None


def test_provenance_of_an_assumption_leads_with_unmeasured_and_denies_a_false_accept_rate():
    tp.install_accepted_still_photo_assumption(_acceptance_record())

    provenance = tp.still_photo_licence_provenance()

    assert provenance == (
        "UNMEASURED: centered-autoplay assumption accepted by the owner; no video "
        "false-accept rate has been measured")
    assert provenance.startswith("UNMEASURED")
    assert "bound" not in provenance and "cards" not in provenance
    notice = tp.still_photo_licence_operator_notice()
    assert notice == (
        "targeted suggestions enabled under an UNMEASURED assumption (centered autoplay); "
        "no video false-accept rate has been measured")


# --- config: apps.hinge.still_photo_assumption_acceptance ------------------------------

def test_no_assumption_key_leaves_numbering_disabled_exactly_as_before():
    c.validate(_load(BASE))

    assert tp.installed_still_photo_licence() is None
    assert tp.hinge_targeting_unavailable_reason() == tp.HINGE_TARGETING_UNAVAILABLE_REASON


def test_configured_assumption_installs_readiness_and_licenses_the_calibration():
    c.validate(_load(_assumption_config(targeting_calibration=_calibration())))

    licence = tp.installed_still_photo_licence()
    assert licence.channel == tp.STILL_PHOTO_LICENCE_ASSUMPTION
    assert licence.record == _acceptance_record()
    assert tp.hinge_targeting_unavailable_reason() is None
    assert tp.still_photo_licence_provenance().startswith("UNMEASURED")


def test_a_later_validate_without_the_assumption_key_turns_readiness_back_off():
    """Process-global readiness must never outlive the config that accepted the assumption."""
    c.validate(_load(_assumption_config()))
    assert tp.installed_still_photo_licence() is not None

    c.validate(_load(BASE))

    assert tp.installed_still_photo_licence() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_assumption_acceptance_must_carry_its_exact_key_set():
    incomplete = _acceptance()
    incomplete.pop("rationale")
    _expect_error(_assumption_config(acceptance=incomplete), "must carry exactly")
    _expect_error(_assumption_config(acceptance=incomplete), "missing ['rationale']")

    extra = _acceptance(video_cards=60)
    _expect_error(_assumption_config(acceptance=extra), "unknown ['video_cards']")

    assert tp.installed_still_photo_licence() is None


@pytest.mark.parametrize("key", ["acceptance", "accepted_at", "device", "hinge_version_name",
                                 "rationale"])
def test_every_assumption_field_must_be_nonempty_text(key):
    _expect_error(_assumption_config(acceptance=_acceptance(**{key: ""})),
                  f"still_photo_assumption_acceptance.{key} must be nonempty text")
    _expect_error(_assumption_config(acceptance=_acceptance(**{key: 3})),
                  f"still_photo_assumption_acceptance.{key} must be nonempty text")

    assert tp.installed_still_photo_licence() is None


def test_assumption_acceptance_must_be_the_exact_phrase():
    _expect_error(
        _assumption_config(acceptance=_acceptance(
            acceptance="I accept the unmeasured centered autoplay assumption")),
        "acceptance must be exactly 'I_ACCEPT_UNMEASURED_CENTERED_AUTOPLAY_ASSUMPTION'")
    # The other channel's phrase is not this channel's phrase: accepting circular AI labels is a
    # different decision about a different risk, and one must never license the other.
    _expect_error(
        _assumption_config(acceptance=_acceptance(
            acceptance=tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE)),
        "acceptance must be exactly")

    assert tp.installed_still_photo_licence() is None


def test_assumption_acceptance_is_bound_to_the_exact_device_serial():
    _expect_error(_assumption_config(acceptance=_acceptance(device="some-other-pixel")),
                  "device must exactly equal apps.hinge.serial")
    without_serial = {**BASE, "apps": {"hinge": {
        "still_photo_assumption_acceptance": _acceptance()}}}
    _expect_error(without_serial, "requires a nonempty apps.hinge.serial")

    assert tp.installed_still_photo_licence() is None


def test_assumption_acceptance_must_be_a_mapping():
    _expect_error(_assumption_config(acceptance="I_ACCEPT_UNMEASURED_CENTERED_AUTOPLAY_ASSUMPTION"),
                  "still_photo_assumption_acceptance must be a mapping")


def test_configuring_both_readiness_channels_is_a_hard_error_naming_both_keys(
        tmp_path, monkeypatch):
    """A measured bound must not be shadowed by an assumption, nor an assumption by a bound."""
    both = _bound_config(tmp_path, monkeypatch,
                         still_photo_assumption_acceptance=_acceptance())

    _expect_error(both, "apps.hinge.still_photo_bound_evidence")
    _expect_error(both, "apps.hinge.still_photo_assumption_acceptance")
    _expect_error(both, "mutually exclusive")

    # And the rejected config leaves readiness OFF rather than installing whichever ran first.
    assert tp.installed_still_photo_licence() is None
    assert tp.hinge_targeting_unavailable_reason() is not None


def test_an_assumption_never_licenses_hinge_auto(tmp_path, monkeypatch):
    """The gate split of ops/STILL-PHOTO-DISCRIMINATOR.md section 3 holds for the new channel.

    An unmeasured assumption is the weakest licence in the system, so if anything at all could
    turn AUTO on without a production-OBSERVE release chain it would be this. AUTO must still
    refuse on observe_release_evidence, exactly as it does with a measured bound.
    """
    monkeypatch.chdir(tmp_path)
    auto = _assumption_config(mode="auto", targeting_calibration=_calibration())

    _expect_error(auto, "observe_release_evidence")

    # The refusal is the RELEASE gate, not the readiness gate: numbering itself was licensed.
    _expect_error(auto, "verified production-OBSERVE mapping")
    assert tp.installed_still_photo_licence().channel == tp.STILL_PHOTO_LICENCE_ASSUMPTION


def test_hinge_auto_without_a_calibration_still_demands_the_release_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    _expect_error(_assumption_config(mode="auto"),
                  "separately verified apps.hinge.observe_release_evidence")
