from __future__ import annotations

import hashlib
import io
import json
import sys
from argparse import Namespace

import pytest

from tools import hinge_calibrate as cal


def _gate(tmp_path):
    return cal._HybridReviewGate(
        tmp_path, device={"serial": "PIXEL-TEST"},
        config_provenance={"sha256": "config-digest"},
        reviewer_model="gpt-5.6-sol", reviewer_process="codex-test")


def test_hybrid_capture_requires_its_exact_acceptance(capsys):
    args = Namespace(profiles=1, unattended=False, hybrid_review=True, confirmation="no",
                     record_operational_checks=False, reviewer_model="gpt-5.6-sol",
                     reviewer_process="codex-test")

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    assert "requires the exact --confirmation" in capsys.readouterr().err


def _approval_for(monkeypatch, gate, frame=b"synthetic-frame"):
    # Predict the exact atomic JSON bytes the gate will publish for its first checkpoint.
    frame_sha = hashlib.sha256(frame).hexdigest()
    body = {
        "schema_version": cal._HYBRID_CHECKPOINT_SCHEMA_VERSION,
        "kind": cal._HYBRID_CHECKPOINT_KIND,
        "sequence": 1,
        "frame": {"file": "00001.png", "sha256": frame_sha},
        "claimed_state": "target_heart_visible",
        "action_plan": {"action": "automated_photo_heart", "photo_model_item": 1,
                        "point": [1, 2], "predicates": {"target_heart_visible": True}},
        "device": {"serial": "PIXEL-TEST"}, "config_sha256": "config-digest",
        "human_ground_truth": False,
    }
    checkpoint = {**body, "evidence_sha256": cal._canonical_json_digest(body)}
    raw = (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode()
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"APPROVE {hashlib.sha256(raw).hexdigest()}\n"))
    return frame


def test_hybrid_checkpoint_is_private_atomic_and_sha_bound(monkeypatch, tmp_path):
    gate = _gate(tmp_path)
    frame = _approval_for(monkeypatch, gate)

    decision = gate.checkpoint(
        frame, claimed_state="target_heart_visible",
        action_plan={"action": "automated_photo_heart", "photo_model_item": 1,
                     "point": [1, 2], "predicates": {"target_heart_visible": True}})

    assert decision["decision"] == "approved"
    assert decision["source"] == "external_ai_review"
    assert decision["human_ground_truth"] is False
    checkpoint_path = tmp_path / "hybrid_review" / "00001.checkpoint.json"
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == decision["checkpoint_sha256"]
    assert json.loads(checkpoint_path.read_text())["frame"]["sha256"] == hashlib.sha256(frame).hexdigest()
    assert not list((tmp_path / "hybrid_review").glob(".*.tmp"))


def test_hybrid_checkpoint_eof_aborts_before_another_action(monkeypatch, tmp_path):
    gate = _gate(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    with pytest.raises(cal._CaptureAbort, match="eof"):
        gate.checkpoint(b"synthetic-frame", claimed_state="target_heart_visible",
                        action_plan={"action": "automated_photo_heart", "photo_model_item": 1,
                                     "point": [1, 2], "predicates": {}})

    assert gate.decisions[-1]["decision"] == "eof"


def test_hybrid_retry_is_recorded_without_granting_transport(monkeypatch, tmp_path):
    gate = _gate(tmp_path)
    # A correct hash with RETRY is a control instruction, not approval.
    frame = b"synthetic-frame"
    frame_sha = hashlib.sha256(frame).hexdigest()
    body = {"schema_version": 1, "kind": cal._HYBRID_CHECKPOINT_KIND, "sequence": 1,
            "frame": {"file": "00001.png", "sha256": frame_sha}, "claimed_state": "state",
            "action_plan": {"action": "x"}, "device": {"serial": "PIXEL-TEST"},
            "config_sha256": "config-digest", "human_ground_truth": False}
    raw = (json.dumps({**body, "evidence_sha256": cal._canonical_json_digest(body)}, indent=2,
                      sort_keys=True) + "\n").encode()
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"RETRY {hashlib.sha256(raw).hexdigest()}\n"))

    assert gate.checkpoint(frame, claimed_state="state", action_plan={"action": "x"})["decision"] == "retry"
