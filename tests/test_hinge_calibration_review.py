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
        "reviewer_protocol": "stdin_checkpoint_sha256_v1",
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
        "action": "automated_pass", "photo_model_item": None, "point": None,
        "point_source": "calibration-only verified-composer Pass transport",
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
        "human_ground_truth": False, "claimed_state": "composer_open_before_pass",
        "action_plan": plan,
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
    assert not any(
        (name.startswith("operation_love") and name != "operation_love.private_files")
        or name == "tools.hinge_calibrate"
        for name in imported
    )


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


# =====================================================================================
# photo-1-only: the second, explicitly-selected automated target strategy
# =====================================================================================
#
# WHY (owner decision 2026-08-22, mirrored from tools/hinge_calibrate.py): real decks rarely
# carry three numberable photos, so the alternating strategy's per-ordinal skip budget can never
# be spent successfully on every even ordinal. The owner may now explicitly select a second
# strategy that always targets photo model item 1 -- an accepted narrowing of the evidence. This
# reviewer must authenticate either strategy, and must refuse a manifest whose own references to
# the strategy disagree with each other.

def _retargeted_capture(tmp_path: Path, name: str, *, strategy_id: str) -> Path:
    """The default `_capture` fixture already targets ordinal 1 / item 1, which is valid under
    EITHER known strategy -- so relabeling every strategy-id reference in its manifest to
    `strategy_id` produces an otherwise-untouched, internally consistent capture for that
    strategy without duplicating the whole fixture."""
    session = _capture(tmp_path, name)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["automated_target_strategy_id"] = strategy_id
    manifest["automation_acceptance"]["target_strategy_id"] = strategy_id
    manifest["profiles"][0]["target_strategy_id"] = strategy_id
    manifest_path.write_text(json.dumps(manifest))
    return session


def test_review_accepts_a_manifest_consistently_using_photo_1_only(tmp_path):
    """A capture whose manifest, acceptance, and profile record all consistently name
    `photo_1_only_v1` must be ACCEPTED exactly like the alternating default -- narrowing the
    evidence is an owner-authorized choice, not a defect."""
    session = _retargeted_capture(
        tmp_path, "photo1only", strategy_id=review._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    config = tmp_path / "config.yaml"

    body = review.build_review([session], config)

    assert body["captures"][0]["target_strategy_id"] == review._PHOTO_1_ONLY_TARGET_STRATEGY_ID


def test_review_refuses_a_manifest_whose_profile_mixes_target_strategy_ids(tmp_path):
    """A manifest that consistently claims photo-1-only at the manifest/acceptance level but
    whose one profile record still claims the alternating id must be REFUSED -- membership in
    the accepted set alone is not enough; every reference within one session must agree with
    every other."""
    session = _retargeted_capture(
        tmp_path, "mixed", strategy_id=review._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"][0]["target_strategy_id"] = review._AUTOMATED_TARGET_STRATEGY_ID
    manifest_path.write_text(json.dumps(manifest))
    config = tmp_path / "config.yaml"

    with pytest.raises(review.ReviewRefused, match="exact alternating target trace"):
        review.build_review([session], config)


def test_review_refuses_when_manifest_and_acceptance_strategy_ids_disagree(tmp_path):
    """The manifest-level `automated_target_strategy_id` and the
    `automation_acceptance.target_strategy_id` must agree with EACH OTHER, not merely each
    independently belong to the accepted set."""
    session = _retargeted_capture(
        tmp_path, "top-level-mismatch", strategy_id=review._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["automation_acceptance"]["target_strategy_id"] = review._AUTOMATED_TARGET_STRATEGY_ID
    manifest_path.write_text(json.dumps(manifest))
    config = tmp_path / "config.yaml"

    with pytest.raises(review.ReviewRefused, match="inconsistent automated target strategy"):
        review.build_review([session], config)


def _retargeted_second_profile_capture(tmp_path: Path, name: str, *, strategy_id: str) -> Path:
    """A capture whose single profile is ordinal 2 targeting photo model item 3 under
    `strategy_id` -- the alternating strategy's even-ordinal target.  Relabels `_retargeted_
    capture`'s otherwise-untouched ordinal-1/item-1 fixture: frame bytes are item-number
    independent, so only the metadata this reviewer cross-checks needs to move to (2, 3)."""
    session = _retargeted_capture(tmp_path, name, strategy_id=strategy_id)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for frame in manifest["frames"]:
        frame["profile_ordinal"] = 2
        if frame["item_number"] == 1:
            frame["item_number"] = 3
    profile = manifest["profiles"][0]
    profile["ordinal"] = 2
    profile["composer_items"] = [3]
    for action in profile["automated_actions"]:
        if action["action"] == "automated_photo_heart":
            action["photo_model_item"] = 3
    manifest_path.write_text(json.dumps(manifest))
    return session


def test_review_accepts_photo_1_only_split_covering_only_depth_1_across_two_profiles(tmp_path):
    """`photo_1_only_v1` never targets item 3 by design (owner decision 2026-08-22): two
    profiles that both only ever touch item 1 collectively cover everything that strategy claims
    to exercise, so `build_review`'s split-level depth gate must accept rather than demand a
    depth the strategy never collects."""
    session_a = _retargeted_capture(
        tmp_path, "p1o-a", strategy_id=review._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    session_b = _retargeted_capture(
        tmp_path, "p1o-b", strategy_id=review._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    config = tmp_path / "config.yaml"

    body = review.build_review([session_a, session_b], config)

    assert len(body["captures"]) == 2


def test_review_refuses_alternating_strategy_split_with_two_profiles_missing_item_3(tmp_path):
    """The exact same shape of depth-1-only profiles that passes under `photo_1_only_v1` must
    still be REFUSED under the (default) alternating strategy, naming both the missing depth and
    the strategy that was asked for -- proving the gate did not simply go slack for every
    strategy."""
    session_a = _capture(tmp_path, "alt-missing-a")
    session_b = _capture(tmp_path, "alt-missing-b")
    config = tmp_path / "config.yaml"

    with pytest.raises(review.ReviewRefused, match="lacks composer evidence") as exc:
        review.build_review([session_a, session_b], config)
    assert "[3]" in str(exc.value)
    assert review._AUTOMATED_TARGET_STRATEGY_ID in str(exc.value)


def test_review_accepts_alternating_strategy_split_with_both_depths_covered(tmp_path):
    """Unchanged regression: the default alternating strategy still accepts a split whose two
    profiles collectively cover both photo-1 and photo-3, exactly as before this strategy-aware
    depth check existed."""
    session_a = _retargeted_capture(
        tmp_path, "alt-a", strategy_id=review._AUTOMATED_TARGET_STRATEGY_ID)
    session_b = _retargeted_second_profile_capture(
        tmp_path, "alt-b", strategy_id=review._AUTOMATED_TARGET_STRATEGY_ID)
    config = tmp_path / "config.yaml"

    body = review.build_review([session_a, session_b], config)

    assert len(body["captures"]) == 2


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


def _send_like_capture(tmp_path: Path, name: str) -> Path:
    """A hybrid Pass fixture changed into an equally byte-bound accepted real Send."""
    session = _legacy_hybrid_capture(tmp_path, name)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["automation_acceptance"]["send_like_accepted"] = True
    manifest["automation_acceptance"]["send_like_confirmation"] = review._SEND_LIKE_CONFIRMATION
    manifest["automation_acceptance"]["terminal_advance_action"] = "automated_send_priority_like"
    actions = manifest["profiles"][0]["automated_actions"]
    pass_index = next(i for i, a in enumerate(actions) if a["action"] == "automated_pass")
    pass_action = actions[pass_index]
    post_sha = pass_action["post_frame_sha256"]
    pre_sha = pass_action["pre_frame_sha256"]
    record = pass_action["review_checkpoints"]["before"]
    plan = {
        "action": "automated_send_priority_like", "photo_model_item": None,
        "point": [690, 1360],
        "point_source": "calibration-only verified-composer Send transport",
        "predicates": {
            "inline_composer_and_selected_photo_verified_before_action": True,
            "send_like_tapped": False,
            "forbidden_zone_guarded_transport": "HingeDriver._tap",
        },
    }
    checkpoint_path = Path(record["checkpoint_file"])
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["action_plan"] = plan
    body = dict(checkpoint)
    body.pop("evidence_sha256", None)
    checkpoint["evidence_sha256"] = review._canonical_digest(body)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    record.update({
        "action_plan": plan,
        "checkpoint_evidence_sha256": checkpoint["evidence_sha256"],
        "checkpoint_sha256": _sha(checkpoint_path.read_bytes()),
    })
    decision = next(
        decision for decision in manifest["hybrid_review"]["decisions"]
        if decision.get("checkpoint_file") == record["checkpoint_file"])
    decision.update(record)
    actions[pass_index] = {
        "action": "automated_send_priority_like",
        "transport": ["HingeDriver._tap(confirm_point)", "HingeDriver._handle_rose_upsell",
                      "HingeDriver._verify_like_landed"],
        "pre_frame_sha256": pre_sha, "post_frame_sha256": post_sha,
        "send_like_tapped": True,
        "confirm_point": [690, 1360],
        "predicates": {
            "inline_composer_and_selected_photo_verified_before_action": True,
            "send_like_tapped": True,
            "like_landed_verified": True,
        },
        "review_checkpoints": {"before": record},
    }
    manifest_path.write_text(json.dumps(manifest))
    return session


def test_review_authenticates_an_accepted_real_send_priority_like(tmp_path):
    session = _send_like_capture(tmp_path, "send-like")
    config = tmp_path / "config.yaml"

    record = review.build_review([session], config)["captures"][0]

    assert record["send_like_accepted"] is True
    assert record["terminal_advance_action"] == "automated_send_priority_like"


@pytest.mark.parametrize(
    "tamper",
    ["checkpoint_json", "checkpoint_png", "config", "action_plan", "frame", "landed_false"],
)
def test_review_refuses_unbound_or_false_send_evidence(tmp_path, tamper):
    session = _send_like_capture(tmp_path, f"send-tamper-{tamper}")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    action = manifest["profiles"][0]["automated_actions"][-1]
    before = action["review_checkpoints"]["before"]
    if tamper == "checkpoint_json":
        checkpoint_path = Path(before["checkpoint_file"])
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["sequence"] = 99
        checkpoint_path.write_text(json.dumps(checkpoint))
    elif tamper == "checkpoint_png":
        Path(before["frame_file"]).write_bytes(b"tampered checkpoint frame")
    elif tamper == "config":
        manifest["config_provenance"]["sha256"] = "0" * 64
    elif tamper == "action_plan":
        before["action_plan"]["point"] = [9, 9]
    elif tamper == "frame":
        action["pre_frame_sha256"] = "0" * 64
    else:
        action["predicates"]["like_landed_verified"] = False
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused):
        review.build_review([session], config)


@pytest.mark.parametrize("json_boolean_alias", [0, 1])
def test_review_rejects_integer_send_acceptance_aliases(tmp_path, json_boolean_alias):
    session = _send_like_capture(tmp_path, f"send-bool-{json_boolean_alias}")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["automation_acceptance"]["send_like_accepted"] = json_boolean_alias
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="exact boolean"):
        review.build_review([session], config)


def test_review_refuses_a_send_trace_without_accepted_send_like(tmp_path):
    """A send trace can never authenticate itself: without the manifest's explicit acceptance,
    the session is read as an ordinary capture that is simply missing its required Pass."""
    session = _send_like_capture(tmp_path, "unaccepted-send")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["automation_acceptance"]["send_like_accepted"] = False
    manifest["automation_acceptance"]["send_like_confirmation"] = None
    manifest["automation_acceptance"]["terminal_advance_action"] = "automated_pass"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="Pass-without-send"):
        review.build_review([session], config)


def test_review_refuses_a_send_confirmation_without_accepted_send_like(tmp_path):
    session = _capture(tmp_path, "confirmation-without-acceptance")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["automation_acceptance"]["send_like_confirmation"] = review._SEND_LIKE_CONFIRMATION
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="send-like confirmation present without accepted"):
        review.build_review([session], config)


def test_review_refuses_a_mixed_pass_and_send_ledger(tmp_path):
    session = _send_like_capture(tmp_path, "mixed-ledger")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"][0]["automated_actions"].append({
        "action": "automated_pass", "send_like_tapped": False, "composer_clear_visible": True,
        "post_frame_sha256": review._sha256(b"unused"),
    })
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(review.ReviewRefused, match="mixes Pass and Send"):
        review.build_review([session], config)


def test_review_accepts_only_hash_bound_hybrid_checkpoints(tmp_path):
    session = _legacy_hybrid_capture(tmp_path, "hybrid")
    config = tmp_path / "config.yaml"
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert review.build_review([session], config)["captures"][0]["capture_mode"] == review._HYBRID_CAPTURE_MODE
    raw = (session / manifest["frames"][-2]["file"]).read_bytes()
    identity = (session / manifest["frames"][-1]["file"]).read_bytes()
    profile_data = calibrate._ProfileData(
        1, "auto-test", [raw], [(1, raw, raw)], raw, identity)
    session_data = calibrate._SessionData(
        session, "calibration", "test", (), (), [profile_data], manifest)
    assert calibrate._verified_hybrid_reviewed_evidence([session_data])["kind"] == "hybrid_ai_reviewed_automation"
    manifest["profiles"][0]["automated_actions"][0]["review_checkpoints"]["before"]["decision"] = "refused"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(review.ReviewRefused, match="exact action binding"):
        review.build_review([session], config)
