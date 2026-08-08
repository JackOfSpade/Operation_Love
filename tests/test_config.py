"""config.validate() — fail-fast checks (offline)."""
import tempfile

import yaml

from operation_love import config as c

BASE = {
    "enabled_apps": ["bumble"],
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
    d = {**BASE, "mode": "observe", "apps": {"bumble": {"mode": "yolo"}}}
    _expect_error(d, "observe")


def test_valid_app_mode_override_passes():
    d = {**BASE, "mode": "observe", "apps": {"bumble": {"mode": "auto"}}}
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
