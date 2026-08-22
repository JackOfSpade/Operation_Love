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
    plan = {"action": "automated_pass", "photo_model_item": None, "point": None,
            "point_source": "calibration-only verified-composer Pass transport",
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
        "human_ground_truth": False, "claimed_state": "composer_open_before_pass",
        "action_plan": plan,
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


@pytest.mark.parametrize(
    ("unattended", "hybrid_review", "confirmation"),
    [(False, False, ""), (True, False, cal._UNATTENDED_CONFIRMATION)],
    ids=["no-automated-transport", "unattended-has-no-reviewer"],
)
def test_send_like_requires_hybrid_review(capsys, unattended, hybrid_review, confirmation):
    """A real Send Priority Like is permanent, so it may only run where a reviewer approves each
    one. Neither a bare `--send-like` nor `--unattended --send-like` may reach the device.

    The unattended case matters twice over: `_verified_automated_circular_evidence` authenticates
    only the Pass-without-send terminal action, so an unattended send capture could never be
    measured either -- refusing at the CLI keeps capture and measurement consistent.
    """
    args = Namespace(profiles=1, unattended=unattended, hybrid_review=hybrid_review,
                     confirmation=confirmation, record_operational_checks=False,
                     send_like=True, send_like_confirmation=cal._SEND_LIKE_CONFIRMATION)

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    assert "requires --hybrid-review" in capsys.readouterr().err


@pytest.mark.parametrize(
    "send_like_confirmation", ["", "wrong phrase", cal._HYBRID_REVIEW_CONFIRMATION],
    ids=["missing", "wrong-phrase", "reuses-the-other-confirmation"],
)
def test_send_like_requires_its_own_exact_confirmation(capsys, send_like_confirmation):
    """`--send-like-confirmation` is a separate phrase from `--confirmation`; neither an empty
    value nor reusing the hybrid-review confirmation may substitute for it, so opting into real
    sends can never be a side effect of opting into reviewed automation."""
    args = Namespace(profiles=1, unattended=False, hybrid_review=True,
                     confirmation=cal._HYBRID_REVIEW_CONFIRMATION,
                     record_operational_checks=False, reviewer_model="m", reviewer_process="p",
                     send_like=True, send_like_confirmation=send_like_confirmation)

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    assert "requires the exact --send-like-confirmation" in capsys.readouterr().err


def _wire_inert_capture_driver(monkeypatch, tmp_path, config_path):
    """Enough of `_cmd_capture`'s outer shell (config, driver, serial, out-dir) to reach the
    profile loop, with `_rewind_automated_profile_to_confirmed_top` mocked to abort immediately
    on the first profile -- so these tests exercise only the manifest's `automation_acceptance`
    recording, never a real card scan/navigation/device sequence (already covered elsewhere)."""
    band = (0.10, 0.048, 0.80, 0.094)

    class _Adb:
        def shell(self, cmd):
            if "ro.product.model" in cmd:
                return "Pixel 7a"
            if "wm density" in cmd:
                return "420"
            return "versionName=1.0\n"

        def screen_size(self):
            return 1080, 2400

    class _Driver:
        def __init__(self, _cfg):
            self.identity_band = band
            self.content_band = (0.125, 0.875)
            self.serial = "PIXEL-TEST"
            self.package = "co.hinge.app"
            self.adb = _Adb()

        def _template(self, _name):
            return object()

        def open_session(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: SimpleNamespace(apps={}))
    # See test_hinge_calibrate.py's _wire_inert_skip_capture: the capture shell now validates
    # (installing the still-photo licence is validate's job); stubbed inert for these harnesses.
    monkeypatch.setattr(cal.cfg_mod, "validate", lambda _cfg_obj: None)
    monkeypatch.setattr(cal, "HingeDriver", _Driver)
    monkeypatch.setattr(cal, "_preflight_serial", lambda _cfg: ("PIXEL-TEST", "adb"))
    monkeypatch.setattr(cal, "_capture_out_dir", lambda *_a, **_kw: tmp_path)
    monkeypatch.setattr(
        cal, "_rewind_automated_profile_to_confirmed_top",
        lambda *_a, **_kw: (_ for _ in ()).throw(cal._CaptureAbort("stop-here")))


def test_send_like_manifest_records_acceptance_and_terminal_action(monkeypatch, tmp_path):
    """The manifest's `automation_acceptance` must say plainly whether real sends were accepted
    for this session, independent of how far the actual device capture got -- so evidence never
    silently reads as Pass-only when the owner opted into real sends.

    Sends require `--hybrid-review`, so this exercises that path rather than plain `--unattended`.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    _wire_inert_capture_driver(monkeypatch, tmp_path, config_path)
    monkeypatch.setattr(cal, "_config_provenance", lambda *_a, **_kw: {
        "schema_version": 1, "path": str(config_path), "sha256": "0" * 64,
        "effective_hinge_sha256": "1" * 64})

    args = Namespace(
        profiles=1, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=False, hybrid_review=True, confirmation=cal._HYBRID_REVIEW_CONFIRMATION,
        record_operational_checks=False, reviewer_model="claude", reviewer_id="claude",
        reviewer_version="v", reviewer_process="stdin", send_like=True,
        send_like_confirmation=cal._SEND_LIKE_CONFIRMATION)

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    acceptance = manifest["automation_acceptance"]
    assert acceptance["send_like_accepted"] is True
    assert acceptance["send_like_confirmation"] == cal._SEND_LIKE_CONFIRMATION
    assert acceptance["terminal_advance_action"] == "automated_send_priority_like"
    assert manifest["interrupted"] is True


def test_default_unattended_manifest_records_no_accepted_send(monkeypatch, tmp_path):
    """Regression guard: leaving `--send-like` off must record an explicit `false`, never an
    absent/None acceptance that a later reader could mistake for 'unspecified'."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    _wire_inert_capture_driver(monkeypatch, tmp_path, config_path)

    args = Namespace(
        profiles=1, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=True, hybrid_review=False, confirmation=cal._UNATTENDED_CONFIRMATION,
        record_operational_checks=False, send_like=False, send_like_confirmation="")

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    acceptance = manifest["automation_acceptance"]
    assert acceptance["send_like_accepted"] is False
    assert acceptance["send_like_confirmation"] is None
    assert acceptance["terminal_advance_action"] == "automated_pass"


def _hybrid_session_with_terminal(tmp_path: Path, *, send_like_accepted: bool) -> cal._SessionData:
    """A minimal, exact-shape hybrid session ending either with the default Pass or an accepted
    real Send -- the current (non-legacy) trace shape, unlike `_legacy_hybrid_session` above."""
    session = _legacy_hybrid_session(tmp_path)
    manifest = session.manifest
    acceptance = manifest["automation_acceptance"]
    terminal_action = ("automated_send_priority_like" if send_like_accepted
                       else "automated_pass")
    acceptance.update({
        "send_like_accepted": send_like_accepted,
        "send_like_confirmation": (cal._SEND_LIKE_CONFIRMATION
                                   if send_like_accepted else None),
        "terminal_advance_action": terminal_action,
    })
    actions = manifest["profiles"][0]["automated_actions"]
    pass_index = next(i for i, action in enumerate(actions)
                      if action["action"] == "automated_pass")
    pass_before = actions[pass_index]["review_checkpoints"]["before"]
    if send_like_accepted:
        plan = {
            "action": terminal_action, "photo_model_item": None, "point": [1, 2],
            "point_source": "calibration-only verified-composer Send transport",
            "predicates": {
                "inline_composer_and_selected_photo_verified_before_action": True,
                "send_like_tapped": False,
                "forbidden_zone_guarded_transport": "HingeDriver._tap",
            },
        }
        checkpoint_path = Path(pass_before["checkpoint_file"])
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["action_plan"] = plan
        body = dict(checkpoint)
        body.pop("evidence_sha256", None)
        checkpoint["evidence_sha256"] = cal._canonical_json_digest(body)
        checkpoint_path.write_text(json.dumps(checkpoint))
        pass_before.update({
            "action_plan": plan,
            "checkpoint_evidence_sha256": checkpoint["evidence_sha256"],
            "checkpoint_sha256": cal._sha256(checkpoint_path.read_bytes()),
        })
        actions[pass_index] = {
            "action": "automated_send_priority_like",
            "transport": ["HingeDriver._tap(confirm_point)", "HingeDriver._handle_rose_upsell",
                          "HingeDriver._verify_like_landed"],
            "pre_frame_sha256": cal._sha256(b"composer"),
            "post_frame_sha256": cal._sha256(b"advance"),
            "send_like_tapped": True, "confirm_point": [1, 2],
            "predicates": {"inline_composer_and_selected_photo_verified_before_action": True,
                          "send_like_tapped": True, "like_landed_verified": True},
            "review_checkpoints": {"before": pass_before},
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return session


def test_hybrid_measure_validates_an_accepted_real_send_priority_like(tmp_path):
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=True)

    evidence = cal._verified_hybrid_reviewed_evidence([session])

    assert evidence["kind"] == cal._HYBRID_CAPTURE_MODE


@pytest.mark.parametrize(
    "tamper",
    ["checkpoint_json", "checkpoint_png", "config", "action_plan", "frame", "landed_false"],
)
def test_hybrid_measure_refuses_unbound_or_false_send_evidence(tmp_path, tamper):
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=True)
    action = session.manifest["profiles"][0]["automated_actions"][-1]
    before = action["review_checkpoints"]["before"]
    if tamper == "checkpoint_json":
        checkpoint_path = Path(before["checkpoint_file"])
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["sequence"] = 99
        checkpoint_path.write_text(json.dumps(checkpoint))
    elif tamper == "checkpoint_png":
        Path(before["frame_file"]).write_bytes(b"tampered checkpoint frame")
    elif tamper == "config":
        session.manifest["config_provenance"]["sha256"] = "0" * 64
    elif tamper == "action_plan":
        before["action_plan"]["point"] = [9, 9]
    elif tamper == "frame":
        action["pre_frame_sha256"] = "0" * 64
    else:
        action["predicates"]["like_landed_verified"] = False

    with pytest.raises(cal._MeasureRefused):
        cal._verified_hybrid_reviewed_evidence([session])


@pytest.mark.parametrize("json_boolean_alias", [0, 1])
def test_hybrid_measure_rejects_integer_send_acceptance_aliases(tmp_path, json_boolean_alias):
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=True)
    session.manifest["automation_acceptance"]["send_like_accepted"] = json_boolean_alias

    with pytest.raises(cal._MeasureRefused, match="non-boolean"):
        cal._verified_hybrid_reviewed_evidence([session])


def test_hybrid_measure_still_validates_the_default_pass(tmp_path):
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=False)

    evidence = cal._verified_hybrid_reviewed_evidence([session])

    assert evidence["kind"] == cal._HYBRID_CAPTURE_MODE


def test_hybrid_measure_refuses_a_send_trace_without_accepted_send_like(tmp_path):
    """A manifest carrying a Send trace but no acceptance must not authenticate itself as one --
    it is read as an ordinary session that is simply missing its required Pass."""
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=True)
    session.manifest["automation_acceptance"]["send_like_accepted"] = False
    session.manifest["automation_acceptance"]["send_like_confirmation"] = None
    session.manifest["automation_acceptance"]["terminal_advance_action"] = "automated_pass"

    with pytest.raises(cal._MeasureRefused, match="lacks exact approved hybrid Pass checkpoint"):
        cal._verified_hybrid_reviewed_evidence([session])


def test_hybrid_measure_refuses_a_send_confirmation_without_accepted_send_like(tmp_path):
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=False)
    session.manifest["automation_acceptance"]["send_like_confirmation"] = cal._SEND_LIKE_CONFIRMATION

    with pytest.raises(cal._MeasureRefused, match="without accepting real sends"):
        cal._verified_hybrid_reviewed_evidence([session])


def test_hybrid_measure_refuses_a_mixed_pass_and_send_ledger(tmp_path):
    session = _hybrid_session_with_terminal(tmp_path, send_like_accepted=True)
    session.manifest["profiles"][0]["automated_actions"].append({
        "action": "automated_pass", "send_like_tapped": False, "composer_clear_visible": True,
        "post_frame_sha256": cal._sha256(b"advance"),
    })

    with pytest.raises(cal._MeasureRefused, match="mixes Pass and Send"):
        cal._verified_hybrid_reviewed_evidence([session])
