"""Review of the supervised-Training release path against the shipped config.yaml.

These pin what an operator meets on the real file, not on inline YAML: the AUTO refusal they read
and the retention protection that survives installing the new gate the way the verifier tells
them to.  Everything here is offline: config is loaded and mutated in memory, and HingeDriver is
only constructed (no ADB, no touch transport).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from operation_love.config import load, validate
from operation_love.drivers.hinge import HingeDriver

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    # Release validators resolve repo-relative artifact paths against the working directory.
    monkeypatch.chdir(REPO)


def _shipped_cfg():
    return load(REPO / "config.yaml")


def test_shipped_config_auto_refusal_names_the_training_release_tool():
    """The shipped file keeps the historical 10.0.1 observe gate, so AUTO reaches the old
    validator and its refusal never mentions the only tool that can now mint release evidence.
    Production OBSERVE is retired, so that refusal names no step an operator can take."""
    cfg = _shipped_cfg()
    assert "mode" not in cfg.apps["hinge"]
    cfg.mode = "auto"
    with pytest.raises(ValueError) as refused:
        validate(cfg)
    assert "tools.hinge_training_release" in str(refused.value), str(refused.value)


def test_installing_the_training_gate_keeps_every_signed_release_run_protected():
    """tools.hinge_training_release tells the operator to remove observe_release_evidence when
    installing its mapping (AUTO accepts exactly one gate).  That key is the only thing that
    protects d8547ff144b4 from debug retention: it is not listed in debug_protect_runs because
    it is derived.  ops/release/<id>/ holds signed artifacts, which is the stated reason
    312b80bf4b50 is listed, so every such run must stay protected after the swap."""
    cfg = _shipped_cfg()
    hinge_cfg = cfg.apps["hinge"]
    hinge_cfg.pop("observe_release_evidence", None)
    hinge_cfg.pop("ai_reviewed_observe_release_evidence", None)
    hinge_cfg["training_release_evidence"] = {
        "schema_version": 1,
        "production_run_reference": "data/hinge_debug/0123456789ab",
        "production_run_id": "0123456789ab",
    }
    protected = HingeDriver(cfg).debug_protect_runs
    signed_runs = sorted(
        p.name for p in (REPO / "ops" / "release").iterdir() if p.is_dir()
    )
    assert signed_runs, (
        "ops/release is tracked; it should list at least one release run"
    )
    assert "0123456789ab" in protected
    unprotected = [run for run in signed_runs if run not in protected]
    assert not unprotected, (
        f"runs with signed ops/release artifacts lose retention protection: {unprotected}"
    )


@pytest.mark.parametrize("value", ["pending", ["0123456789ab"]])
def test_a_malformed_training_release_mapping_cannot_crash_training_driver_startup(
    value,
):
    """In Training mode config does not validate training_release_evidence at all, but the
    driver's retention derivation calls ``.get`` on it.  A non-mapping value must be a clean
    config refusal or be ignored -- never an AttributeError while the Training run starts."""
    cfg = _shipped_cfg()
    cfg.apps["hinge"]["training_release_evidence"] = value
    try:
        validate(cfg)
    except ValueError:
        return
    HingeDriver(cfg)
