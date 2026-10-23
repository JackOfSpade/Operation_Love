"""The supervised-Training AUTO release gate's trust model, pinned against a hand-written artifact.

Config's ``training_release_evidence`` validator re-checks the calibration binding and re-hashes
the retained ``actions.jsonl``, but like the two OBSERVE gates it does not re-derive the Like/Pass
chains, the environment rows or the store counts: it trusts that ``tools.hinge_training_release``
wrote the artifact, which the tool does only after all three verify.  What config does bind is the
log's bytes, so any change to the retained run blocks AUTO.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operation_love import config as config_mod
from tests.test_hinge_training_release import TS, _calibration, _cfg, _environment
from tools import hinge_training_release as release


@pytest.fixture
def repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "hinge_targeting_unavailable_reason", lambda: None)
    return tmp_path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hand_written_release(root: Path, run_id: str, rows: list[dict]) -> dict:
    """Write a debug run and an artifact by hand, without running the verifier."""
    calibration = _calibration()
    run_dir = root / "data" / "hinge_debug" / run_id
    run_dir.mkdir(parents=True)
    actions = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (run_dir / "actions.jsonl").write_bytes(actions)
    run_ref = f"data/hinge_debug/{run_id}"
    verified_at = "2026-10-23T04:00:00+00:00"
    artifact = {
        "schema_version": 1,
        "kind": release.KIND,
        "completed": True,
        "human_ground_truth": True,
        "source": "manual",
        "calibration_calibrated_at": calibration["calibrated_at"],
        "calibration_sha256": release._canonical_sha256(calibration),
        "device": calibration["device"],
        "hinge_version_name": calibration["hinge_version_name"],
        "frame_size_px": calibration["frame_size_px"],
        "production_run_reference": run_ref,
        "production_run_id": run_id,
        "debug_actions_sha256": _sha(actions),
        "chain_evidence_sha256": "a" * 64,
        "store_persistence_evidence_sha256": "b" * 64,
        "not_exercised": list(release.NOT_EXERCISED),
        "verified_at": verified_at,
    }
    out = root / "ops" / "release" / run_id
    out.mkdir(parents=True)
    content = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
    (out / release.ARTIFACT_NAME).write_bytes(content)
    cfg = _cfg(calibration, mode="auto")
    cfg.apps["hinge"]["training_release_evidence"] = {
        "schema_version": 1,
        "calibration_calibrated_at": calibration["calibrated_at"],
        "calibration_sha256": release._canonical_sha256(calibration),
        "device": calibration["device"],
        "hinge_version_name": calibration["hinge_version_name"],
        "frame_size_px": calibration["frame_size_px"],
        "production_run_reference": run_ref,
        "production_run_id": run_id,
        "debug_actions_sha256": _sha(actions),
        "verification_file": f"ops/release/{run_id}/{release.ARTIFACT_NAME}",
        "verification_sha256": _sha(content),
        "verified_at": verified_at,
    }
    return cfg


def _binding(run_id: str) -> dict:
    return {
        "ts": TS,
        "action": "observe_release_run_binding",
        "run_id": run_id,
        "app": "hinge",
    }


def test_config_trusts_the_artifact_but_binds_it_to_the_retained_log_bytes(repo):
    """Config accepts a digest-consistent artifact without re-deriving its chains.

    The run below holds no Like or Pass chain, so the verifier refuses it and only a
    hand-written artifact can name it; config does not re-run the verifier.  What config does
    check is that the retained ``actions.jsonl`` still hashes to the mapping's digest, so a
    single changed byte refuses.
    """
    run_id = "run-nochain"
    calibration = _calibration()
    rows = [_binding(run_id), _environment(calibration)]
    run_dir = repo / "data" / "hinge_debug" / run_id
    cfg = _hand_written_release(repo, run_id, rows)
    with pytest.raises(release.ReleaseEvidenceRefused, match="LIKE chain"):
        release._verify_debug_run(run_dir, run_id, calibration)
    config_mod._validate_hinge_auto_release_gate(cfg)

    actions = run_dir / "actions.jsonl"
    tampered = bytearray(actions.read_bytes())
    tampered[-2] ^= 0x01
    actions.write_bytes(bytes(tampered))
    with pytest.raises(ValueError, match="does not match the retained run"):
        config_mod._validate_hinge_auto_release_gate(cfg)
