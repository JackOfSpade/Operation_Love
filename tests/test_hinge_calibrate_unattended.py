from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import hinge_calibrate as cal


def _target_prefix(*, target_complete=True, predecessor_complete=True, identity_known=True,
                   target_item=3, target_ordinal=5):
    """Proof-shaped prefix with an unresolved lower tail that cannot affect the target."""
    blocks = (
        SimpleNamespace(heart_ordinal=None, complete=True, page_y0=20, page_y1=80),
        SimpleNamespace(heart_ordinal=1, complete=predecessor_complete, page_y0=81, page_y1=800),
        SimpleNamespace(heart_ordinal=target_ordinal, complete=target_complete,
                        page_y0=801, page_y1=1600),
        SimpleNamespace(heart_ordinal=None, complete=False, page_y0=1601, page_y1=2200),
    )
    index = SimpleNamespace(usable=True, at_scroll_top=True, blocks=blocks,
                            complete=False, identity=SimpleNamespace(known=identity_known))
    crop = SimpleNamespace(image=b"photo", signature=object(), frame_index=2,
                           heart_ordinal=target_ordinal)
    payload = SimpleNamespace(usable=True, item=lambda number: crop if number == target_item
                              else (_ for _ in ()).throw(cal.ItemCropError("not available")))
    return index, payload


def _automated_session(tmp_path: Path) -> cal._SessionData:
    manifest = {
        "capture_mode": "automated_circular_risk_accepted",
        "human_ground_truth": False,
        "automated_target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID,
        "automation_acceptance": {
            "confirmation": cal._UNATTENDED_CONFIRMATION,
            "target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID,
            "not_independent_ground_truth": True,
            "not_supervised_operational_evidence": True,
        },
        "profiles": [{
            "ordinal": 1,
            "target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID,
            "composer_items": [1],
            "automated_actions": [
                {"action": "automated_photo_heart", "photo_model_item": 1,
                 "post_tap_composer_verified": True, "post_tap_item_relative_verified": True},
                {"action": "automated_pass", "send_like_tapped": False,
                 "composer_clear_visible": True},
            ],
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return cal._SessionData(tmp_path, "calibration", "serial", (), (), [], manifest)


def _legacy_hybrid_session(tmp_path: Path) -> cal._SessionData:
    """Construct the exact pre-summary hybrid Pass shape with immutable checkpoint bytes."""
    reviewer = {"source": "external_ai_review", "id": "terra-reviewer", "model": "gpt-5.6-terra",
                "version": "2026-08-14", "process": "independent-test"}
    device = {"serial": "serial"}
    config_sha256 = hashlib.sha256(b"config").hexdigest()
    pre, composer, advance, identity = b"pre", b"composer", b"advance", b"identity"
    review_dir = tmp_path / "hybrid_review"
    review_dir.mkdir(parents=True)
    frame_path = review_dir / "00001.png"
    frame_path.write_bytes(composer)
    plan = {"action": "automated_pass", "photo_model_item": None,
            "predicates": {
                "inline_composer_and_selected_photo_verified_before_action": True,
                "send_like_tapped": False,
                "forbidden_zone_guarded_transport": "HingeDriver._tap",
            }}
    checkpoint_body = {
        "schema_version": cal._HYBRID_CHECKPOINT_SCHEMA_VERSION,
        "kind": cal._HYBRID_CHECKPOINT_KIND, "sequence": 1,
        "frame": {"file": frame_path.name, "sha256": cal._sha256(composer)},
        "claimed_state": "composer_open_before_pass", "action_plan": plan, "device": device,
        "config_sha256": config_sha256, "human_ground_truth": False,
    }
    checkpoint = {**checkpoint_body, "evidence_sha256": cal._canonical_json_digest(checkpoint_body)}
    checkpoint_path = review_dir / "00001.checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    pass_before = {
        "checkpoint_evidence_sha256": checkpoint["evidence_sha256"],
        "checkpoint_file": str(checkpoint_path), "checkpoint_sha256": cal._sha256(checkpoint_path.read_bytes()),
        "frame_file": str(frame_path), "frame_sha256": cal._sha256(composer),
        "decision": "approved", "source": "external_ai_review", "reviewer": reviewer,
        "human_ground_truth": False, "action_plan": plan,
    }
    heart_before = {"checkpoint_evidence_sha256": "heart-before", "frame_sha256": cal._sha256(pre),
                    "decision": "approved", "source": "external_ai_review", "reviewer": reviewer,
                    "human_ground_truth": False,
                    "action_plan": {"action": "automated_photo_heart", "photo_model_item": 1}}
    heart_after = {"checkpoint_evidence_sha256": "heart-after", "frame_sha256": cal._sha256(composer),
                   "decision": "approved", "source": "external_ai_review", "reviewer": reviewer,
                   "human_ground_truth": False,
                   "action_plan": {"action": "review_heart_result", "photo_model_item": 1}}
    pass_action = {
        "action": "automated_pass", "pre_frame_sha256": cal._sha256(composer),
        "post_frame_sha256": cal._sha256(advance),
        "transport": ["HingeDriver._swipe(android_edge_back)", "HingeDriver._await_button(pass)",
                      "HingeDriver._locate_button(pass)", "HingeDriver._tap"],
        "predicates": {
            "inline_composer_and_selected_photo_verified_before_action": True,
            "inline_composer_structurally_confirmed_before_action": True,
            "edge_back_transport_performed": True,
            "pass_vision_relocated_before_guarded_tap": True,
            "send_like_tapped": False, "deck_frame_changed": True,
            "composer_clear_visible": True, "new_profile_top_confirmed": True,
        }, "review_checkpoints": {"before": pass_before},
    }
    manifest = {
        "capture_mode": cal._HYBRID_CAPTURE_MODE, "human_ground_truth": False,
        "automated_target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID,
        "config_provenance": {"sha256": config_sha256},
        "automation_acceptance": {
            "confirmation": cal._HYBRID_REVIEW_CONFIRMATION,
            "target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID,
            "reviewer_protocol": "stdin_checkpoint_sha256_v1", "reviewer_source": "external_ai_review",
            "reviewer": reviewer,
        },
        "hybrid_review": {"schema_version": 1, "protocol": "stdin_checkpoint_sha256_v1",
                          "reviewer": reviewer, "human_ground_truth": False,
                          "decisions": [heart_before, heart_after, pass_before]},
        "profiles": [{
            "ordinal": 1, "action_evidence_mode": cal._HYBRID_CAPTURE_MODE,
            "target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID, "composer_items": [1],
            "automated_actions": [
                {"action": "automated_photo_heart", "photo_model_item": 1,
                 "post_tap_composer_verified": True, "post_tap_item_relative_verified": True,
                 "review_checkpoints": {"before": heart_before, "after": heart_after}},
                pass_action,
            ],
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    profile = cal._ProfileData(1, "profile", [b"card"], [(1, pre, composer)], advance, identity)
    return cal._SessionData(tmp_path, "calibration", "serial", (), (), [profile], manifest)


def test_automated_trace_is_distinct_from_supervised_operational_evidence(tmp_path):
    evidence = cal._verified_automated_circular_evidence([_automated_session(tmp_path)])

    assert evidence["kind"] == "automated_circular_risk_accepted"
    assert evidence["not_supervised_operational_evidence"] is True
    assert evidence["records"][0]["human_ground_truth"] is False


def test_automated_target_strategy_is_one_photo_per_ordinal():
    assert [cal._automated_composer_items_for_ordinal(ordinal) for ordinal in range(1, 5)] == [
        (1,), (3,), (1,), (3,)]
    assert [cal._automated_composer_items_for_ordinal(ordinal) for ordinal in range(1, 4)] == [
        (1,), (3,), (1,)]


def test_target_scoped_prefix_accepts_item3_with_absolute_ordinal_and_ignores_later_tail():
    index, payload = _target_prefix()

    assert cal._target_scoped_prefix_reason(index, payload, (3,)) is None
    assert payload.item(3).heart_ordinal == 5  # photo-model item 3 is the fifth real heart.
    assert index.complete is False


def test_target_scoped_prefix_does_not_mistake_confirmed_top_chrome_for_a_predecessor_item():
    index, payload = _target_prefix()
    chrome = SimpleNamespace(kind="leading_chrome", heart_ordinal=None, complete=False,
                             page_y0=0, page_y1=80)
    index.blocks = (chrome, *index.blocks[1:])

    assert cal._target_scoped_prefix_reason(index, payload, (3,)) is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_complete": False}, "target photo item 3 was not observed end-to-end"),
        ({"predecessor_complete": False}, "predecessor"),
        ({"identity_known": False}, "sticky-header identity"),
    ],
)
def test_target_scoped_prefix_refuses_partial_target_predecessor_or_unknown_identity(kwargs, match):
    index, payload = _target_prefix(**kwargs)

    assert match in cal._target_scoped_prefix_reason(index, payload, (3,))


def test_target_scoped_prefix_refuses_photo_not_in_the_exact_prefix_payload():
    index, payload = _target_prefix()

    assert "unavailable" in cal._target_scoped_prefix_reason(index, payload, (1,))


def test_target_scoped_post_tap_check_uses_the_exact_selected_photo(monkeypatch):
    index, payload = _target_prefix()
    calls = []
    surface = SimpleNamespace(layout_id=cal._COMPOSER_LAYOUT_ID)
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_args, **_kwargs: surface)
    monkeypatch.setattr(
        cal, "verify_sheet_item",
        lambda _frame, _payload, item, **_kwargs: calls.append(item)
        or SimpleNamespace(matched=True, reason="exact target"))

    assert cal._verified_automated_composer(
        b"composer", confirm_template=object(), payload=payload, item_number=3) is surface
    assert calls == [3]


def test_automated_validator_refuses_a_second_heart_after_composer_open(tmp_path):
    session = _automated_session(tmp_path)
    session.manifest["profiles"][0]["composer_items"] = [1, 3]
    session.manifest["profiles"][0]["automated_actions"].insert(
        1, {"action": "automated_photo_heart", "photo_model_item": 3,
            "post_tap_composer_verified": True, "post_tap_item_relative_verified": True})

    with pytest.raises(cal._MeasureRefused, match="exact automated"):
        cal._verified_automated_circular_evidence([session])


def test_hybrid_validator_accepts_only_byte_bound_legacy_pass_summary(tmp_path):
    session = _legacy_hybrid_session(tmp_path)

    assert cal._verified_hybrid_reviewed_evidence([session])["kind"] == cal._HYBRID_CAPTURE_MODE

    pass_action = session.manifest["profiles"][0]["automated_actions"][1]
    pass_action["send_like_tapped"] = False
    pass_action["composer_clear_visible"] = True
    assert cal._verified_hybrid_reviewed_evidence([session])["kind"] == cal._HYBRID_CAPTURE_MODE

    pass_action["send_like_tapped"] = True
    with pytest.raises(cal._MeasureRefused):
        cal._verified_hybrid_reviewed_evidence([session])


def test_hybrid_legacy_pass_resolves_absolute_checkpoint_paths_from_relative_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    session = _legacy_hybrid_session(Path("relative-session"))
    pass_before = session.manifest["profiles"][0]["automated_actions"][1]["review_checkpoints"]["before"]
    pass_before["checkpoint_file"] = str(Path(pass_before["checkpoint_file"]).resolve())
    pass_before["frame_file"] = str(Path(pass_before["frame_file"]).resolve())

    assert cal._verified_hybrid_reviewed_evidence([session])["kind"] == cal._HYBRID_CAPTURE_MODE


@pytest.mark.parametrize("tamper", ["checkpoint", "predicate", "transport", "post_hash"])
def test_hybrid_validator_refuses_unbound_legacy_pass_summary(tmp_path, tamper):
    session = _legacy_hybrid_session(tmp_path)
    pass_action = session.manifest["profiles"][0]["automated_actions"][1]
    if tamper == "checkpoint":
        session.manifest["hybrid_review"]["decisions"] = []
    elif tamper == "predicate":
        checkpoint_path = Path(pass_action["review_checkpoints"]["before"]["checkpoint_file"])
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["action_plan"]["predicates"]["send_like_tapped"] = True
        checkpoint_path.write_text(json.dumps(checkpoint))
    elif tamper == "transport":
        pass_action["transport"][-2] = "HingeDriver._locate_button(send_like)"
    else:
        pass_action["post_frame_sha256"] = "0" * 64

    with pytest.raises(cal._MeasureRefused):
        cal._verified_hybrid_reviewed_evidence([session])


def test_hybrid_legacy_pass_refuses_resolved_checkpoint_escape(tmp_path):
    session = _legacy_hybrid_session(tmp_path)
    pass_before = session.manifest["profiles"][0]["automated_actions"][1]["review_checkpoints"]["before"]
    checkpoint_path = Path(pass_before["checkpoint_file"])
    escaped_path = tmp_path / "escaped.checkpoint.json"
    escaped_path.write_bytes(checkpoint_path.read_bytes())
    pass_before["checkpoint_file"] = str(escaped_path)

    with pytest.raises(cal._MeasureRefused):
        cal._verified_hybrid_reviewed_evidence([session])


def test_hybrid_legacy_pass_refuses_checkpoint_symlink_escape(tmp_path):
    session = _legacy_hybrid_session(tmp_path)
    pass_before = session.manifest["profiles"][0]["automated_actions"][1]["review_checkpoints"]["before"]
    checkpoint_path = Path(pass_before["checkpoint_file"])
    escaped_path = tmp_path / "escaped.checkpoint.json"
    escaped_path.write_bytes(checkpoint_path.read_bytes())
    symlink_path = checkpoint_path.parent / "escaped-link.checkpoint.json"
    symlink_path.symlink_to(escaped_path)
    pass_before["checkpoint_file"] = str(symlink_path)

    with pytest.raises(cal._MeasureRefused):
        cal._verified_hybrid_reviewed_evidence([session])


def test_legacy_closed_set_hybrid_review_remains_accepted_by_measure_binding(tmp_path):
    """A live capture started before prefix scope exists must keep its stronger old contract."""
    session = _automated_session(tmp_path)
    session.manifest["capture_mode"] = cal._HYBRID_CAPTURE_MODE
    (session.dir / "manifest.json").write_text(json.dumps(session.manifest))
    config = tmp_path / "config.yaml"
    config.write_text("apps: {}\n")
    body = {
        "schema_version": cal._UNATTENDED_REVIEW_SCHEMA_VERSION,
        "kind": cal._UNATTENDED_REVIEW_KIND,
        "human_ground_truth": False,
        "not_independent_ground_truth": True,
        "reviewer": {"deterministic_second_process": True,
                     "imports_capture_or_vision_stack": False,
                     "implementation_sha256": "reviewer"},
        "config": {"sha256": cal._sha256(config.read_bytes())},
        "captures": [{
            "session": str(session.dir.resolve()),
            "manifest_sha256": cal._sha256((session.dir / "manifest.json").read_bytes()),
            "device": None,
            "frame_size_px": None,
            "config_sha256": cal._sha256(config.read_bytes()),
            "capture_evidence_scope": "closed_set_profile_v1",
            "human_ground_truth": False,
        }],
    }
    review = {**body, "evidence_sha256": cal._canonical_json_digest(body)}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))

    assert cal._unattended_review_reference_reason(
        str(path), [session], config_path=str(config)) is None


def test_unattended_capture_requires_exact_confirmation(capsys):
    args = Namespace(profiles=1, unattended=True, confirmation="no", record_operational_checks=False)

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    assert "requires the exact --confirmation" in capsys.readouterr().err
