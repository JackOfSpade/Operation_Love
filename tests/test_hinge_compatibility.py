import hashlib
import json
from types import SimpleNamespace

import pytest

from operation_love.hinge_compatibility import (
    ACCEPTANCE,
    KIND,
    PUBLIC_RECEIPT_KIND,
    SUPERVISED_KIND,
    _SUPERVISED_FILES,
    canonical_sha256,
    validate_review,
)
from operation_love.drivers.hinge import AndroidDriver


def _save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fixture(root):
    calibration = {
        "device": "phone",
        "hinge_version_name": "10.7.0",
        "frame_size_px": [1080, 2400],
    }
    licence = {"device": "phone", "hinge_version_name": "10.7.0"}
    app = {
        "serial": "phone",
        "targeting_calibration": calibration,
        "still_photo_assumption_acceptance": licence,
    }
    captures = []
    review_captures = []
    for decision, terminal in (
        ("dislike", "automated_pass"),
        ("like", "automated_send_priority_like"),
    ):
        manifest = {
            "interrupted": False,
            "capture_mode": "hybrid_ai_reviewed_automation",
            "human_ground_truth": False,
            "device": {"serial": "phone", "hinge_version_name": "10.8.0"},
            "frame_size_px": [1080, 2400],
            "config_provenance": {"sha256": "a" * 64},
            "automation_acceptance": {
                "terminal_advance_action": terminal,
                "send_like_accepted": decision == "like",
            },
            "profiles": [
                {
                    "automated_actions": [
                        {"action": "automated_photo_heart"},
                        {"action": terminal},
                    ]
                }
            ],
        }
        ref = _save(root, f"{decision}/manifest.json", manifest)
        captures.append(
            {
                "decision": decision,
                "manifest_path": ref["path"],
                "manifest_sha256": ref["sha256"],
            }
        )
        review_captures.append(
            {
                "session": str((root / decision).resolve()),
                "manifest_sha256": ref["sha256"],
                "terminal_advance_action": terminal,
            }
        )
    review_body = {
        "kind": "hinge_unattended_calibration_independent_review",
        "human_ground_truth": False,
        "config": {"sha256": "a" * 64},
        "captures": review_captures,
    }
    review_ref = _save(
        root,
        "review.json",
        {**review_body, "evidence_sha256": canonical_sha256(review_body)},
    )
    body = {
        "schema_version": 1,
        "kind": KIND,
        "status": "training_only",
        "acceptance": ACCEPTANCE,
        "human_ground_truth": False,
        "video_false_accept_rate": "unmeasured",
        "source_build": "10.7.0",
        "target_build": "10.8.0",
        "device": "phone",
        "frame_size_px": [1080, 2400],
        "source_calibration_sha256": canonical_sha256(calibration),
        "source_licence_key": "still_photo_assumption_acceptance",
        "source_licence_sha256": canonical_sha256(licence),
        "captures": captures,
        "independent_review_path": review_ref["path"],
        "independent_review_sha256": review_ref["sha256"],
        "reviewed_at": "2026-10-14T12:00:00Z",
        "release_notes_url": "https://play.google.com/store/apps/details?id=co.hinge.app",
        "release_notes_summary": "Generic performance note; no layout detail.",
    }
    artifact_ref = _save(
        root, "compatibility.json", {**body, "evidence_sha256": canonical_sha256(body)}
    )
    app["version_compatibility_review"] = {
        "artifact_path": artifact_ref["path"],
        "artifact_sha256": artifact_ref["sha256"],
    }
    return app


def test_two_decision_review_licenses_only_the_reviewed_training_version(tmp_path):
    app = _fixture(tmp_path)
    assert validate_review(app, root=tmp_path, mode="training") == "10.8.0"
    with pytest.raises(ValueError, match="Training only"):
        validate_review(app, root=tmp_path, mode="auto")


def test_changed_capture_or_source_evidence_refuses_compatibility(tmp_path):
    app = _fixture(tmp_path)
    (tmp_path / "like/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_review(app, root=tmp_path, mode="training")

    app = _fixture(tmp_path)
    app["targeting_calibration"]["hinge_version_name"] = "10.6.0"
    with pytest.raises(ValueError, match="source evidence"):
        validate_review(app, root=tmp_path, mode="training")


def test_missing_like_review_refuses_compatibility(tmp_path):
    app = _fixture(tmp_path)
    review = json.loads((tmp_path / "review.json").read_text())
    review["captures"] = review["captures"][:1]
    review["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in review.items() if key != "evidence_sha256"}
    )
    ref = _save(tmp_path, "review.json", review)
    artifact = json.loads((tmp_path / "compatibility.json").read_text())
    artifact["independent_review_sha256"] = ref["sha256"]
    artifact["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in artifact.items() if key != "evidence_sha256"}
    )
    new_ref = _save(tmp_path, "compatibility.json", artifact)
    app["version_compatibility_review"]["artifact_sha256"] = new_ref["sha256"]
    with pytest.raises(ValueError, match="independent compatibility review"):
        validate_review(app, root=tmp_path, mode="training")


def test_driver_accepts_only_the_reviewed_live_build_with_the_old_frame_size():
    def driver(version, size):
        return SimpleNamespace(
            targeting_calibration=SimpleNamespace(
                hinge_version_name="10.7.0", frame_size_px=(1080, 2400)
            ),
            _targeting_calibration_unavailable=None,
            _targeting_compatible_version="10.8.0",
            _targeting_runtime_version_name=version,
            _targeting_runtime_frame_size=size,
            _adb=None,
        )

    reviewed = driver("10.8.0", (1080, 2400))
    assert AndroidDriver._refresh_targeting_calibration_binding(reviewed) == ""
    assert reviewed.targeting_calibration is not None

    for version, size in (("10.9.0", (1080, 2400)), ("10.8.0", (1080, 2340))):
        unreviewed = driver(version, size)
        assert (
            "does not exactly match"
            in AndroidDriver._refresh_targeting_calibration_binding(unreviewed)
        )
        assert unreviewed.targeting_calibration is None


def _supervised_fixture(root):
    calibration = {
        "device": "phone",
        "hinge_version_name": "10.7.0",
        "frame_size_px": [1080, 2400],
    }
    licence = {"device": "phone", "hinge_version_name": "10.7.0"}
    app = {
        "serial": "phone",
        "targeting_calibration": calibration,
        "still_photo_assumption_acceptance": licence,
    }
    files = {}
    for name in _SUPERVISED_FILES:
        if name.endswith(".png"):
            path = root / name
            path.write_bytes(name.encode())
            files[name] = {
                "path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    files["upsell_before.png"]["sha256"] = files["send_after.png"]["sha256"]
    (root / "upsell_before.png").write_bytes((root / "send_after.png").read_bytes())
    files["dislike.json"] = _save(
        root,
        "dislike.json",
        {
            "decision": "dislike",
            "before_sha256": files["dislike_before.png"]["sha256"],
            "after_sha256": files["dislike_after.png"]["sha256"],
        },
    )
    files["like.json"] = _save(
        root,
        "like.json",
        {
            "decision": "like",
            "before_sha256": files["send_before.png"]["sha256"],
            "after_sha256": files["send_after.png"]["sha256"],
            "send_point": [687, 1388],
        },
    )
    files["upsell.json"] = _save(
        root,
        "upsell.json",
        {
            "decision": "like",
            "upsell_before_sha256": files["upsell_before.png"]["sha256"],
            "upsell_after_sha256": files["upsell_after.png"]["sha256"],
            "dismissed": True,
        },
    )
    template = root / "operation_love/drivers/assets/hinge_like_button_10_8.png"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_bytes(b"template")
    body = {
        "schema_version": 2,
        "kind": SUPERVISED_KIND,
        "status": "training_only",
        "acceptance": ACCEPTANCE,
        "human_ground_truth": False,
        "video_false_accept_rate": "unmeasured",
        "source_build": "10.7.0",
        "target_build": "10.8.0",
        "device": "phone",
        "frame_size_px": [1080, 2400],
        "source_calibration_sha256": canonical_sha256(calibration),
        "source_licence_key": "still_photo_assumption_acceptance",
        "source_licence_sha256": canonical_sha256(licence),
        "files": files,
        "dislike_profile_name": "ProfileA",
        "like_profile_name": "ProfileB",
        "like_confirmation_text": "Nice, ProfileB will see your Like sooner.",
        "send_point": [687, 1388],
        "like_template_path": "operation_love/drivers/assets/hinge_like_button_10_8.png",
        "like_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "reviewed_at": "2026-10-14T16:34:00Z",
        "release_notes_url": "https://play.google.com/store/apps/details?id=co.hinge.app",
        "release_notes_summary": "Generic performance note; no layout detail.",
    }
    ref = _save(
        root, "supervised.json", {**body, "evidence_sha256": canonical_sha256(body)}
    )
    app["version_compatibility_review"] = {
        "artifact_path": ref["path"],
        "artifact_sha256": ref["sha256"],
    }
    return app


def test_supervised_two_pass_review_is_training_only_and_digest_bound(tmp_path):
    app = _supervised_fixture(tmp_path)
    assert validate_review(app, root=tmp_path, mode="training") == "10.8.0"
    with pytest.raises(ValueError, match="Training only"):
        validate_review(app, root=tmp_path, mode="auto")
    (tmp_path / "upsell_after.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_review(app, root=tmp_path, mode="training")


def test_redacted_receipt_validates_without_private_profile_frames(tmp_path):
    app = _supervised_fixture(tmp_path)
    private = json.loads((tmp_path / "supervised.json").read_text())
    body = {
        key: private[key]
        for key in (
            "status",
            "acceptance",
            "human_ground_truth",
            "video_false_accept_rate",
            "source_build",
            "target_build",
            "device",
            "frame_size_px",
            "source_calibration_sha256",
            "source_licence_key",
            "source_licence_sha256",
            "like_template_path",
            "like_template_sha256",
            "reviewed_at",
            "release_notes_url",
            "release_notes_summary",
        )
    }
    body.update(
        {
            "schema_version": 3,
            "kind": PUBLIC_RECEIPT_KIND,
            "decision_order": ["dislike", "like"],
            "like_sent_confirmed": True,
            "private_artifact_sha256": app["version_compatibility_review"][
                "artifact_sha256"
            ],
        }
    )
    ref = _save(
        tmp_path, "receipt.json", {**body, "evidence_sha256": canonical_sha256(body)}
    )
    app["version_compatibility_review"] = {
        "artifact_path": ref["path"],
        "artifact_sha256": ref["sha256"],
    }
    for name in _SUPERVISED_FILES:
        (tmp_path / name).unlink()
    (tmp_path / "supervised.json").unlink()
    assert validate_review(app, root=tmp_path, mode="training") == "10.8.0"


def test_10_8_like_template_is_selected_only_for_that_live_version():
    import numpy as np

    driver = SimpleNamespace(
        spec=SimpleNamespace(app="hinge", templates={"like": "hinge_like_button.png"}),
        _targeting_runtime_version_name="10.7.0",
    )
    old_template = AndroidDriver._template(driver, "like")
    driver._targeting_runtime_version_name = "10.8.0"
    new_template = AndroidDriver._template(driver, "like")
    assert old_template.shape == new_template.shape
    assert not np.array_equal(old_template, new_template)
    driver._targeting_runtime_version_name = "10.9.0"
    assert np.array_equal(AndroidDriver._template(driver, "like"), new_template)
