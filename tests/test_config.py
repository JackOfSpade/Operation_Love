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
    "opener": {"enabled": True, "model": "claude-opus-4-8"},
    "budget": {"run_budget_usd": 5.0,
               "pricing": {"claude-opus-4-8": {"input": 5, "output": 25}}},
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
    d = {**BASE, "opener": {"enabled": True, "model": "claude-unknown-9"}}
    _expect_error(d, "budget.pricing")


def test_bad_app_mode_override():
    d = {**BASE, "mode": "observe", "apps": {"hinge": {"mode": "yolo"}}}
    _expect_error(d, "observe")


def test_valid_app_mode_override_passes():
    d = {**BASE, "mode": "observe", "apps": {"hinge": {"mode": "auto"}}}
    c.validate(_load(d))   # no raise


def test_bad_on_exhausted():
    d = {**BASE, "budget": {**BASE["budget"], "on_exhausted": "stp"}}
    _expect_error(d, "on_exhausted")


def test_valid_on_exhausted_passes():
    for v in ("stop", "swipe_without_opener"):
        d = {**BASE, "budget": {**BASE["budget"], "on_exhausted": v}}
        c.validate(_load(d))   # no raise


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
    indexed after load() reads it -- fixed at the source in config.load()."""
    for key in ("ranker", "quality_filter", "opener", "pacing", "paths", "apps"):
        d = {**BASE, key: None}
        c.validate(_load(d))   # no raise


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
