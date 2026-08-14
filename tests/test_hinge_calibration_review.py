"""Hermetic contract tests for the stdlib-only unattended capture reviewer."""
from __future__ import annotations

import ast
import hashlib
import json
import struct
from pathlib import Path

import pytest

from tools import hinge_calibration_review as review
from tools import hinge_calibrate as calibrate


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _png(width=1080, height=2400) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def _capture(tmp_path: Path, name: str, *, mode=review._CAPTURE_MODE, provenance=True,
             card_scroll_count=1) -> Path:
    session = tmp_path / name
    session.mkdir()
    config = tmp_path / "config.yaml"
    if not config.exists():
        config.write_text("apps:\n  hinge:\n    serial: test\n")
    frames = []
    payloads = []
    frame_roles = [(1, "card_scroll", None)] * card_scroll_count
    frame_roles.extend(((1, "target_pre", 1), (1, "composer_open", 1),
                        (1, "profile_advance_clear", None), (1, "profile_advance_identity", None)))
    for ordinal, role, item in frame_roles:
        raw = _png()
        name_ = f"{len(frames) + 1:05d}.png"
        (session / name_).write_bytes(raw)
        payloads.append(raw)
        frames.append({"file": name_, "sha256": _sha(raw), "profile_ordinal": ordinal,
                       "profile_id": "auto-test", "role": role, "item_number": item})
    ledger = b'{"schema":1}'
    (session / "entry_anchor_ledger.json").write_bytes(ledger)
    config_hash = _sha(config.read_bytes())
    by_role_item = {(frame["role"], frame["item_number"]): frame
                    for frame in frames if frame["role"] != "card_scroll"}
    manifest = {
        "unattended_provenance_schema_version": 1 if provenance else None,
        "capture_mode": mode, "human_ground_truth": False, "interrupted": False,
        "split": "calibration", "skipped_attempts": [],
        "automated_target_strategy_id": review._AUTOMATED_TARGET_STRATEGY_ID,
        "automation_acceptance": {"confirmation": review._CONFIRMATION,
                                    "target_strategy_id": review._AUTOMATED_TARGET_STRATEGY_ID,
                                    "not_independent_ground_truth": True,
                                    "not_supervised_operational_evidence": True},
        "config_provenance": {"schema_version": 1, "path": str(config.resolve()),
                              "sha256": config_hash, "effective_hinge_sha256": "test"},
        "device": {"serial": "test", "model": "Pixel", "display_w": 1080,
                   "display_h": 2400, "density": 420, "hinge_package": "co.hinge.app",
                   "hinge_version_name": "1.0"},
        "frame_size_px": [1080, 2400], "frame_count": len(frames), "frames": frames,
        "entry_anchor_ledger": {"file": "entry_anchor_ledger.json", "sha256": _sha(ledger)},
        "profiles": [{"ordinal": 1, "action_evidence_mode": mode,
                      "target_strategy_id": review._AUTOMATED_TARGET_STRATEGY_ID,
                      "composer_items": [1],
                      "automated_actions": [
                          {"action": "automated_photo_heart", "photo_model_item": 1,
                           "pre_frame_sha256": by_role_item[("target_pre", 1)]["sha256"],
                           "post_frame_sha256": by_role_item[("composer_open", 1)]["sha256"],
                           "post_tap_composer_verified": True, "post_tap_item_relative_verified": True},
                          {"action": "automated_pass",
                           "post_frame_sha256": by_role_item[("profile_advance_clear", None)]["sha256"],
                           "send_like_tapped": False, "composer_clear_visible": True},
                      ]}],
    }
    (session / "manifest.json").write_text(json.dumps(manifest))
    return session


def _legacy_hybrid_capture(tmp_path: Path, name: str) -> Path:
    """A byte-bound pre-summary Pass trace, matching the one-time compatibility contract."""
    session = _capture(tmp_path, name)
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reviewer = {"source": "external_ai_review", "id": "terra-reviewer", "model": "gpt-5.6-terra",
                "version": "2026-08-14", "process": "independent-test"}
    manifest["capture_mode"] = review._HYBRID_CAPTURE_MODE
    manifest["automation_acceptance"].update({
        "confirmation": review._HYBRID_CONFIRMATION,
        "reviewer_source": "external_ai_review",
        "reviewer": reviewer,
    })
    profile = manifest["profiles"][0]
    profile["action_evidence_mode"] = review._HYBRID_CAPTURE_MODE
    pass_action = next(action for action in profile["automated_actions"]
                       if action["action"] == "automated_pass")
    pass_action.pop("send_like_tapped")
    pass_action.pop("composer_clear_visible")
    composer = next(frame for frame in manifest["frames"] if frame["role"] == "composer_open")
    advance = next(frame for frame in manifest["frames"] if frame["role"] == "profile_advance_clear")
    pass_action["pre_frame_sha256"] = composer["sha256"]
    pass_action["post_frame_sha256"] = advance["sha256"]
    pass_action["transport"] = ["HingeDriver._swipe(android_edge_back)",
                                  "HingeDriver._await_button(pass)",
                                  "HingeDriver._locate_button(pass)", "HingeDriver._tap"]
    pass_action["predicates"] = {
        "inline_composer_and_selected_photo_verified_before_action": True,
        "inline_composer_structurally_confirmed_before_action": True,
        "edge_back_transport_performed": True,
        "pass_vision_relocated_before_guarded_tap": True,
        "send_like_tapped": False,
        "deck_frame_changed": True,
        "composer_clear_visible": True,
        "new_profile_top_confirmed": True,
    }
    review_dir = session / "hybrid_review"
    review_dir.mkdir()
    frame_path = review_dir / "00001.png"
    frame_path.write_bytes((session / composer["file"]).read_bytes())
    plan = {
        "action": "automated_pass", "photo_model_item": None,
        "predicates": {
            "inline_composer_and_selected_photo_verified_before_action": True,
            "send_like_tapped": False,
            "forbidden_zone_guarded_transport": "HingeDriver._tap",
        },
    }
    checkpoint_body = {
        "schema_version": 1, "kind": "hinge_hybrid_calibration_checkpoint", "sequence": 1,
        "frame": {"file": frame_path.name, "sha256": composer["sha256"]},
        "claimed_state": "composer_open_before_pass", "action_plan": plan,
        "device": manifest["device"], "config_sha256": _sha(config.read_bytes()),
        "human_ground_truth": False,
    }
    checkpoint = {**checkpoint_body, "evidence_sha256": review._canonical_digest(checkpoint_body)}
    checkpoint_path = review_dir / "00001.checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    record = {
        "checkpoint_evidence_sha256": checkpoint["evidence_sha256"],
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path.read_bytes()),
        "frame_sha256": composer["sha256"], "frame_file": str(frame_path),
        "decision": "approved", "source": "external_ai_review", "reviewer": reviewer,
        "human_ground_truth": False, "action_plan": plan,
    }
    heart = next(action for action in profile["automated_actions"]
                 if action["action"] == "automated_photo_heart")
    heart_before = {
        "checkpoint_evidence_sha256": "heart-before", "frame_sha256": heart["pre_frame_sha256"],
        "decision": "approved", "source": "external_ai_review", "reviewer": reviewer,
        "human_ground_truth": False,
        "action_plan": {"action": "automated_photo_heart", "photo_model_item": 1},
    }
    heart_after = {
        "checkpoint_evidence_sha256": "heart-after", "frame_sha256": heart["post_frame_sha256"],
        "decision": "approved", "source": "external_ai_review", "reviewer": reviewer,
        "human_ground_truth": False,
        "action_plan": {"action": "review_heart_result", "photo_model_item": 1},
    }
    heart["review_checkpoints"] = {"before": heart_before, "after": heart_after}
    pass_action["review_checkpoints"] = {"before": record}
    manifest["hybrid_review"] = {"schema_version": 1, "protocol": "stdin_checkpoint_sha256_v1",
                                 "reviewer": reviewer, "human_ground_truth": False,
                                 "decisions": [heart_before, heart_after, record]}
    manifest_path.write_text(json.dumps(manifest))
    return session


def test_review_is_deterministic_self_hashed_and_binds_exact_capture_and_config(tmp_path):
    session = _capture(tmp_path, "calibration")
    config = tmp_path / "config.yaml"

    body = review.build_review([session], config)
    artifact = {**body, "evidence_sha256": review._canonical_digest(body)}

    assert artifact["human_ground_truth"] is False
    assert artifact["reviewer"]["deterministic_second_process"] is True
    assert artifact["captures"][0]["manifest_sha256"] == _sha((session / "manifest.json").read_bytes())
    assert artifact["evidence_sha256"] == review._canonical_digest(body)
    assert review.build_review([session], config) == body


def test_reviewer_is_structurally_independent_of_capture_and_vision_stacks():
    tree = ast.parse(Path(review.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith("operation_love") or name == "tools.hinge_calibrate"
                   for name in imported)


def test_review_refuses_tampered_frame_or_config(tmp_path):
    session = _capture(tmp_path, "calibration")
    config = tmp_path / "config.yaml"
    (session / "00003.png").write_bytes(_png(999, 2400))
    with pytest.raises(review.ReviewRefused, match="digest mismatch"):
        review.build_review([session], config)

    session = _capture(tmp_path, "heldout")
    config.write_text("apps:\n  hinge:\n    serial: changed\n")
    with pytest.raises(review.ReviewRefused, match="config bytes differ"):
        review.build_review([session], config)


def test_review_refuses_supervised_or_cross_mode_capture(tmp_path):
    config = tmp_path / "config.yaml"
    supervised = _capture(tmp_path, "supervised", mode="supervised_manual")
    unattended = _capture(tmp_path, "unattended")

    with pytest.raises(review.ReviewRefused, match="only explicit automated"):
        review.build_review([supervised], config)
    with pytest.raises(review.ReviewRefused, match="same capture session"):
        review.build_review([unattended, unattended], config)


def test_review_refuses_a_second_target_on_an_odd_ordinal(tmp_path):
    session = _capture(tmp_path, "calibration")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"][0]["composer_items"] = [1, 3]
    manifest["profiles"][0]["automated_actions"].insert(
        1, {"action": "automated_photo_heart", "photo_model_item": 3,
            "pre_frame_sha256": "wrong", "post_frame_sha256": "wrong",
            "post_tap_composer_verified": True, "post_tap_item_relative_verified": True})
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="alternating target trace"):
        review.build_review([session], config)


def test_review_allows_ordered_repeated_card_scroll_frames(tmp_path):
    session = _capture(tmp_path, "repeated-card-scroll", card_scroll_count=3)
    config = tmp_path / "config.yaml"

    assert review.build_review([session], config)["captures"][0]["frame_count"] == 7


def test_review_refuses_duplicate_singleton_or_out_of_order_frame_roles(tmp_path):
    session = _capture(tmp_path, "duplicate-singleton")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    duplicate = dict(manifest["frames"][1])
    duplicate["file"] = "00006.png"
    (session / duplicate["file"]).write_bytes((session / "00002.png").read_bytes())
    manifest["frames"].append(duplicate)
    manifest["frame_count"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="duplicate singleton frame role/item"):
        review.build_review([session], config)

    session = _capture(tmp_path, "out-of-order")
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["role"], manifest["frames"][1]["role"] = (
        manifest["frames"][1]["role"], manifest["frames"][0]["role"])
    manifest["frames"][0]["item_number"], manifest["frames"][1]["item_number"] = (
        manifest["frames"][1]["item_number"], manifest["frames"][0]["item_number"])
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="frame role sequence"):
        review.build_review([session], config)


def test_review_authenticates_only_the_exact_legacy_hybrid_pass_summary(tmp_path):
    session = _legacy_hybrid_capture(tmp_path, "legacy-pass")
    config = tmp_path / "config.yaml"

    assert review.build_review([session], config)["captures"][0]["capture_mode"] == review._HYBRID_CAPTURE_MODE

    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pass_action = next(action for action in manifest["profiles"][0]["automated_actions"]
                       if action["action"] == "automated_pass")
    pass_action["send_like_tapped"] = False
    pass_action["composer_clear_visible"] = True
    manifest_path.write_text(json.dumps(manifest))

    assert review.build_review([session], config)["captures"][0]["frame_count"] == 5

    pass_action["send_like_tapped"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="Pass-without-send"):
        review.build_review([session], config)


@pytest.mark.parametrize("tamper", ["missing_checkpoint", "checkpoint_predicate", "transport", "post_hash"])
def test_review_refuses_unbound_legacy_hybrid_pass_summary(tmp_path, tamper):
    session = _legacy_hybrid_capture(tmp_path, tamper)
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pass_action = next(action for action in manifest["profiles"][0]["automated_actions"]
                       if action["action"] == "automated_pass")
    if tamper == "missing_checkpoint":
        manifest["hybrid_review"]["decisions"] = []
    elif tamper == "checkpoint_predicate":
        checkpoint_path = Path(pass_action["review_checkpoints"]["before"]["checkpoint_file"])
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["action_plan"]["predicates"]["send_like_tapped"] = True
        checkpoint_path.write_text(json.dumps(checkpoint))
    elif tamper == "transport":
        pass_action["transport"][-2] = "HingeDriver._locate_button(send_like)"
    else:
        pass_action["post_frame_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused):
        review.build_review([session], config)


def test_review_accepts_only_hash_bound_hybrid_checkpoints(tmp_path):
    session = _capture(tmp_path, "hybrid")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reviewer = {"source": "external_ai_review", "id": "runner-42", "model": "reviewer",
                "version": "1", "process": "job-9"}
    manifest["capture_mode"] = review._HYBRID_CAPTURE_MODE
    manifest["automation_acceptance"].update({
        "confirmation": review._HYBRID_CONFIRMATION,
        "reviewer_protocol": "stdin_checkpoint_sha256_v1", "reviewer_source": "external_ai_review",
        "reviewer": reviewer,
    })
    decisions = []
    for action in manifest["profiles"][0]["automated_actions"]:
        checks = {}
        count = 2 if action["action"] == "automated_photo_heart" else 1
        for label in ("before", "after")[:count]:
            record = {"checkpoint_evidence_sha256": f"checkpoint-{len(decisions)}",
                          "frame_sha256": f"frame-{len(decisions)}", "decision": "approved",
                          "source": "external_ai_review", "reviewer": reviewer,
                          "human_ground_truth": False,
                          "action_plan": {"action": ("review_heart_result" if label == "after"
                                                     else action["action"]),
                                      "photo_model_item": action.get("photo_model_item")}}
            decisions.append(record)
            checks[label] = record
        action["review_checkpoints"] = checks
    manifest["profiles"][0]["action_evidence_mode"] = review._HYBRID_CAPTURE_MODE
    manifest["hybrid_review"] = {"schema_version": 1, "protocol": "stdin_checkpoint_sha256_v1",
                                 "reviewer": reviewer, "human_ground_truth": False, "decisions": decisions}
    manifest_path.write_text(json.dumps(manifest))

    assert review.build_review([session], config)["captures"][0]["capture_mode"] == review._HYBRID_CAPTURE_MODE
    session_data = calibrate._SessionData(session, "calibration", "test", (), (), [], manifest)
    assert calibrate._verified_hybrid_reviewed_evidence([session_data])["kind"] == "hybrid_ai_reviewed_automation"
    manifest["profiles"][0]["automated_actions"][0]["review_checkpoints"]["before"]["decision"] = "refused"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(review.ReviewRefused, match="exact action binding"):
        review.build_review([session], config)
