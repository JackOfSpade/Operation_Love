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


def test_opener_provider_must_be_supported():
    d = {**BASE, "opener": {"enabled": True, "model": "claude-opus-4-8", "provider": "openai"}}
    _expect_error(d, "opener.provider")


def test_empty_config_file_loads_with_defaults():
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.close()                          # zero-byte file -> yaml.safe_load returns None
    cfg = c.load(f.name)
    assert cfg.mode == "observe" and cfg.enabled_apps == ["bumble"]


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


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
