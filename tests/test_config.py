"""config.validate() — fail-fast checks (offline)."""
import tempfile

import yaml

from operation_love import config as c

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


def test_unknown_app():
    d = {**BASE, "enabled_apps": ["tinder"]}
    _expect_error(d, "unknown app")


def test_bad_mode():
    d = {**BASE, "mode": "yolo"}
    _expect_error(d, "observe")


def test_bigquery_requires_project_id():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {}}}
    _expect_error(d, "project_id")


def test_bigquery_requires_photo_bucket():
    d = {**BASE, "storage": {"backend": "bigquery", "bigquery": {"project_id": "proj"}}}
    _expect_error(d, "photo_bucket")


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


def test_valid_app_mode_override_passes():
    d = {**BASE, "mode": "observe", "apps": {"hinge": {"mode": "auto"}}}
    c.validate(_load(d))   # no raise


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

def test_bumble_web_is_a_known_app_id_and_loads_fine_despite_being_unrunnable():
    # bumble_web is a real registry id now (Bumble's web app is discontinued) -- config
    # validation only cares that it's a KNOWN id; check_runnable (start-time) is what
    # actually rejects running it.
    d = {**BASE, "enabled_apps": ["bumble_web"]}
    c.validate(_load(d))   # no raise


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


def test_auto_mode_is_fine_when_halt_on_error_is_left_at_its_default():
    d = dict(BASE, mode="auto", apps={"hinge": {}})
    c.validate(_load(d))   # no raise -- default is True


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


def test_android_app_coords_entry_negative_is_rejected():
    d = {**BASE, "enabled_apps": ["bumble"],
         "apps": {"bumble": {"coords": {"pass_x": [0.5, -0.2]}}}}
    _expect_error(d, "apps.bumble.coords.pass_x")


def test_android_app_coords_entry_must_be_an_xy_pair():
    d = {**BASE, "enabled_apps": ["bumble"],
         "apps": {"bumble": {"coords": {"like_heart": [0.5, 0.5, 0.5]}}}}
    _expect_error(d, "apps.bumble.coords.like_heart")


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


def test_web_app_config_is_not_subject_to_android_fraction_validation():
    # bumble_web is a web (Playwright) platform with no coords/*_frac concept -- CSS
    # `selectors` instead. This must not misfire on it even if a numeric-looking key there
    # happened to end in `_frac`.
    d = {**BASE, "enabled_apps": ["hinge"],
         "apps": {"bumble_web": {"lookalike_frac": 5.0}}}
    c.validate(_load(d))   # no raise -- bumble_web is not an Android app


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
