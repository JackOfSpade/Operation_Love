"""Validate a reviewed, two-decision Hinge update check for Training only.

The source calibration and still-photo licence keep their original build provenance.  A
compatibility record permits one separately reviewed live build; it never converts the two
profiles into a video false-accept measurement or an AUTO release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ACCEPTANCE = "I_ACCEPT_TWO_DECISION_UNMEASURED_UPDATE_COMPATIBILITY"
KIND = "hinge_two_decision_compatibility"
SUPERVISED_KIND = "hinge_two_decision_supervised_ui_compatibility"
PUBLIC_RECEIPT_KIND = "hinge_two_decision_redacted_compatibility_receipt"
_HEX = frozenset("0123456789abcdef")
_DECISIONS = {"dislike": "automated_pass", "like": "automated_send_priority_like"}
_SUPERVISED_FILES = frozenset({
    "dislike_before.png", "dislike_after.png", "dislike.json",
    "like_before.png", "like_composer.png", "send_before.png", "send_after.png",
    "upsell_before.png", "upsell_after.png", "like.json", "upsell.json",
})


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _read_bound_file(path: str, digest: str, *, root: Path, label: str) -> dict:
    content = _read_bound_bytes(path, digest, root=root, label=label)
    try:
        result = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def _read_bound_bytes(path: str, digest: str, *, root: Path, label: str) -> bytes:
    if not isinstance(path, str) or not path or not _digest(digest):
        raise ValueError(f"{label} needs a repo-relative path and SHA-256 digest")
    relative = Path(path)
    if relative.is_absolute():
        raise ValueError(f"{label} must be repo-relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
        content = resolved.read_bytes()
    except (ValueError, OSError) as exc:
        raise ValueError(f"{label} is outside the repo or unreadable") from exc
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return content


def _validate_supervised_review(artifact: dict, app_cfg: dict, *, root: Path) -> str:
    expected = {
        "schema_version", "kind", "status", "acceptance", "human_ground_truth",
        "video_false_accept_rate", "source_build", "target_build", "device",
        "frame_size_px", "source_calibration_sha256", "source_licence_key",
        "source_licence_sha256", "files", "dislike_profile_name", "like_profile_name",
        "like_confirmation_text", "send_point", "like_template_path", "like_template_sha256",
        "reviewed_at", "release_notes_url", "release_notes_summary", "evidence_sha256",
    }
    if set(artifact) != expected or artifact["schema_version"] != 2:
        raise ValueError("supervised compatibility artifact has an unsupported schema")
    body = {key: value for key, value in artifact.items() if key != "evidence_sha256"}
    if artifact["evidence_sha256"] != canonical_sha256(body):
        raise ValueError("supervised compatibility artifact self-hash mismatch")
    if (
        artifact["status"] != "training_only"
        or artifact["acceptance"] != ACCEPTANCE
        or artifact["human_ground_truth"] is not False
        or artifact["video_false_accept_rate"] != "unmeasured"
    ):
        raise ValueError("supervised compatibility review overstates its evidence")
    calibration = app_cfg.get("targeting_calibration")
    licence_key = artifact["source_licence_key"]
    if licence_key not in {"still_photo_bound_evidence", "still_photo_assumption_acceptance"}:
        raise ValueError("supervised compatibility review names an unsupported licence")
    licence = app_cfg.get(licence_key)
    if not isinstance(calibration, dict) or not isinstance(licence, dict):
        raise ValueError("supervised compatibility review lacks source evidence")
    source, target = artifact["source_build"], artifact["target_build"]
    if (
        canonical_sha256(calibration) != artifact["source_calibration_sha256"]
        or canonical_sha256(licence) != artifact["source_licence_sha256"]
        or not isinstance(target, str) or not target or target == source
        or source != calibration.get("hinge_version_name")
        or source != licence.get("hinge_version_name")
        or artifact["device"] != app_cfg.get("serial")
        or artifact["device"] != calibration.get("device")
        or artifact["frame_size_px"] != calibration.get("frame_size_px")
    ):
        raise ValueError("supervised compatibility review source/device/build mismatch")
    for key in ("reviewed_at", "release_notes_url", "release_notes_summary"):
        if not isinstance(artifact[key], str) or not artifact[key].strip():
            raise ValueError(f"supervised compatibility review lacks {key}")
    for key in ("dislike_profile_name", "like_profile_name", "like_confirmation_text"):
        if not isinstance(artifact[key], str) or not artifact[key].strip():
            raise ValueError(f"supervised compatibility review lacks {key}")
    if artifact["dislike_profile_name"] == artifact["like_profile_name"]:
        raise ValueError("supervised compatibility review reused one profile")
    if "your Like sooner" not in artifact["like_confirmation_text"]:
        raise ValueError("supervised compatibility review lacks a sent-Like confirmation")
    point = artifact["send_point"]
    if (
        not isinstance(point, list) or len(point) != 2
        or not all(isinstance(v, int) for v in point)
        or not (100 <= point[0] <= 980 and 1000 <= point[1] <= 2200)
    ):
        raise ValueError("supervised compatibility Send Like point is invalid")
    _read_bound_bytes(
        artifact["like_template_path"], artifact["like_template_sha256"],
        root=root, label="reviewed Like template",
    )
    files = artifact["files"]
    if not isinstance(files, dict) or set(files) != _SUPERVISED_FILES:
        raise ValueError("supervised compatibility review needs all decision frames and logs")
    for name, reference in files.items():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"supervised compatibility file {name} is malformed")
        if Path(reference["path"]).name != name:
            raise ValueError(f"supervised compatibility file {name} is misnamed")
        _read_bound_bytes(reference["path"], reference["sha256"], root=root, label=name)
    dislike = _read_bound_file(
        files["dislike.json"]["path"], files["dislike.json"]["sha256"],
        root=root, label="Dislike action record",
    )
    like = _read_bound_file(
        files["like.json"]["path"], files["like.json"]["sha256"],
        root=root, label="Like action record",
    )
    upsell = _read_bound_file(
        files["upsell.json"]["path"], files["upsell.json"]["sha256"],
        root=root, label="Like confirmation record",
    )
    if (
        dislike.get("decision") != "dislike"
        or dislike.get("before_sha256") != files["dislike_before.png"]["sha256"]
        or dislike.get("after_sha256") != files["dislike_after.png"]["sha256"]
        or like.get("decision") != "like"
        or like.get("before_sha256") != files["send_before.png"]["sha256"]
        or like.get("after_sha256") != files["send_after.png"]["sha256"]
        or like.get("send_point") != point
        or upsell.get("decision") != "like"
        or upsell.get("upsell_before_sha256") != files["upsell_before.png"]["sha256"]
        or upsell.get("upsell_after_sha256") != files["upsell_after.png"]["sha256"]
        or upsell.get("dismissed") is not True
        or files["send_after.png"]["sha256"] != files["upsell_before.png"]["sha256"]
    ):
        raise ValueError("supervised compatibility action record does not match its frames")
    return target


def _validate_redacted_receipt(artifact: dict, app_cfg: dict, *, root: Path) -> str:
    """Check a portable receipt produced after private frame review.

    The screenshot hashes and action logs stay in the local private artifact. A receipt makes
    a fresh checkout usable without publishing real profiles or treating the two decisions as
    a new statistical calibration.
    """
    expected = {
        "schema_version", "kind", "status", "acceptance", "human_ground_truth",
        "video_false_accept_rate", "source_build", "target_build", "device",
        "frame_size_px", "source_calibration_sha256", "source_licence_key",
        "source_licence_sha256", "decision_order", "like_sent_confirmed",
        "private_artifact_sha256", "like_template_path", "like_template_sha256",
        "reviewed_at", "release_notes_url", "release_notes_summary", "evidence_sha256",
    }
    if set(artifact) != expected or artifact["schema_version"] != 3:
        raise ValueError("redacted compatibility receipt has an unsupported schema")
    body = {key: value for key, value in artifact.items() if key != "evidence_sha256"}
    if artifact["evidence_sha256"] != canonical_sha256(body):
        raise ValueError("redacted compatibility receipt self-hash mismatch")
    if (
        artifact["status"] != "training_only"
        or artifact["acceptance"] != ACCEPTANCE
        or artifact["human_ground_truth"] is not False
        or artifact["video_false_accept_rate"] != "unmeasured"
        or artifact["decision_order"] != ["dislike", "like"]
        or artifact["like_sent_confirmed"] is not True
        or not _digest(artifact["private_artifact_sha256"])
    ):
        raise ValueError("redacted compatibility receipt overstates or lacks review evidence")
    calibration = app_cfg.get("targeting_calibration")
    licence_key = artifact["source_licence_key"]
    if licence_key not in {"still_photo_bound_evidence", "still_photo_assumption_acceptance"}:
        raise ValueError("redacted compatibility receipt names an unsupported licence")
    licence = app_cfg.get(licence_key)
    if not isinstance(calibration, dict) or not isinstance(licence, dict):
        raise ValueError("redacted compatibility receipt lacks source evidence")
    source, target = artifact["source_build"], artifact["target_build"]
    if (
        canonical_sha256(calibration) != artifact["source_calibration_sha256"]
        or canonical_sha256(licence) != artifact["source_licence_sha256"]
        or not isinstance(target, str) or not target or target == source
        or source != calibration.get("hinge_version_name")
        or source != licence.get("hinge_version_name")
        or artifact["device"] != app_cfg.get("serial")
        or artifact["device"] != calibration.get("device")
        or artifact["frame_size_px"] != calibration.get("frame_size_px")
    ):
        raise ValueError("redacted compatibility receipt source/device/build mismatch")
    for key in ("reviewed_at", "release_notes_url", "release_notes_summary"):
        if not isinstance(artifact[key], str) or not artifact[key].strip():
            raise ValueError(f"redacted compatibility receipt lacks {key}")
    if artifact["like_template_path"] != "operation_love/drivers/assets/hinge_like_button_10_8.png":
        raise ValueError("redacted compatibility receipt names an unexpected Like template")
    _read_bound_bytes(
        artifact["like_template_path"], artifact["like_template_sha256"],
        root=root, label="reviewed Like template",
    )
    return target


def validate_review(app_cfg: dict, *, root: Path, mode: str) -> str | None:
    """Return the reviewed live version, or None when no review is configured.

    Validation reads only local, digest-bound metadata.  The offline reviewer must replay and
    inspect all frames before writing its independent-review artifact.  Runtime validation
    checks those immutable references and the exact source evidence on every start.
    """
    raw = app_cfg.get("version_compatibility_review")
    if raw is None:
        return None
    if mode != "training":
        raise ValueError("Hinge version compatibility review licenses Training only, never AUTO")
    if not isinstance(raw, dict) or set(raw) != {"artifact_path", "artifact_sha256"}:
        raise ValueError(
            "apps.hinge.version_compatibility_review needs exactly artifact_path and "
            "artifact_sha256"
        )
    artifact = _read_bound_file(
        raw["artifact_path"], raw["artifact_sha256"], root=root, label="compatibility artifact"
    )
    if artifact.get("kind") == PUBLIC_RECEIPT_KIND:
        return _validate_redacted_receipt(artifact, app_cfg, root=root)
    if artifact.get("kind") == SUPERVISED_KIND:
        return _validate_supervised_review(artifact, app_cfg, root=root)
    expected_keys = {
        "schema_version", "kind", "status", "acceptance", "human_ground_truth",
        "video_false_accept_rate", "source_build", "target_build", "device",
        "frame_size_px", "source_calibration_sha256", "source_licence_key",
        "source_licence_sha256", "captures", "independent_review_path",
        "independent_review_sha256", "reviewed_at", "release_notes_url",
        "release_notes_summary", "evidence_sha256",
    }
    if set(artifact) != expected_keys or artifact.get("schema_version") != 1 or artifact.get("kind") != KIND:
        raise ValueError("compatibility artifact has an unsupported schema")
    body = {key: value for key, value in artifact.items() if key != "evidence_sha256"}
    if artifact["evidence_sha256"] != canonical_sha256(body):
        raise ValueError("compatibility artifact self-hash mismatch")
    if (
        artifact["status"] != "training_only"
        or artifact["acceptance"] != ACCEPTANCE
        or artifact["human_ground_truth"] is not False
        or artifact["video_false_accept_rate"] != "unmeasured"
    ):
        raise ValueError("compatibility artifact overstates its evidence or lacks acceptance")
    calibration = app_cfg.get("targeting_calibration")
    licence_key = artifact["source_licence_key"]
    if licence_key not in {"still_photo_bound_evidence", "still_photo_assumption_acceptance"}:
        raise ValueError("compatibility artifact names an unsupported still-photo licence")
    licence = app_cfg.get(licence_key)
    if not isinstance(calibration, dict) or not isinstance(licence, dict):
        raise ValueError("compatibility artifact source evidence is absent")
    if (
        canonical_sha256(calibration) != artifact["source_calibration_sha256"]
        or canonical_sha256(licence) != artifact["source_licence_sha256"]
    ):
        raise ValueError("compatibility artifact no longer matches source evidence")
    source, target = artifact["source_build"], artifact["target_build"]
    if (
        not isinstance(source, str) or not isinstance(target, str) or not target
        or source == target or source != calibration.get("hinge_version_name")
        or source != licence.get("hinge_version_name")
        or artifact["device"] != app_cfg.get("serial")
        or artifact["device"] != calibration.get("device")
        or artifact["frame_size_px"] != calibration.get("frame_size_px")
    ):
        raise ValueError("compatibility artifact device/build/frame binding mismatch")
    for key in ("reviewed_at", "release_notes_url", "release_notes_summary"):
        if not isinstance(artifact[key], str) or not artifact[key].strip():
            raise ValueError(f"compatibility artifact lacks {key}")
    captures = artifact["captures"]
    if not isinstance(captures, list) or len(captures) != 2:
        raise ValueError("compatibility artifact needs exactly Dislike and Like captures")
    manifest_by_path: dict[str, tuple[str, str]] = {}
    config_hashes: set[str] = set()
    for capture in captures:
        if not isinstance(capture, dict) or set(capture) != {"decision", "manifest_path", "manifest_sha256"}:
            raise ValueError("compatibility capture reference is malformed")
        decision = capture["decision"]
        if decision not in _DECISIONS:
            raise ValueError("compatibility capture has an unsupported decision")
        manifest = _read_bound_file(
            capture["manifest_path"], capture["manifest_sha256"], root=root,
            label=f"{decision} capture manifest",
        )
        device = manifest.get("device")
        acceptance = manifest.get("automation_acceptance")
        profiles = manifest.get("profiles")
        actions = profiles[0].get("automated_actions") if isinstance(profiles, list) and len(profiles) == 1 and isinstance(profiles[0], dict) else None
        terminal = _DECISIONS[decision]
        if (
            manifest.get("interrupted") is not False
            or manifest.get("capture_mode") != "hybrid_ai_reviewed_automation"
            or manifest.get("human_ground_truth") is not False
            or not isinstance(device, dict)
            or device.get("serial") != artifact["device"]
            or device.get("hinge_version_name") != target
            or manifest.get("frame_size_px") != artifact["frame_size_px"]
            or not isinstance(acceptance, dict)
            or acceptance.get("terminal_advance_action") != terminal
            or acceptance.get("send_like_accepted") is not (decision == "like")
            or not isinstance(actions, list)
            or len([a for a in actions if isinstance(a, dict) and a.get("action") == terminal]) != 1
            or len([a for a in actions if isinstance(a, dict) and a.get("action") == "automated_photo_heart"]) != 1
        ):
            raise ValueError(f"{decision} capture does not prove one reviewed photo and {terminal}")
        provenance = manifest.get("config_provenance")
        if not isinstance(provenance, dict) or not _digest(provenance.get("sha256")):
            raise ValueError(f"{decision} capture lacks config provenance")
        config_hashes.add(provenance["sha256"])
        manifest_by_path[str((root / capture["manifest_path"]).resolve().parent)] = (
            capture["manifest_sha256"], decision
        )
    if {capture["decision"] for capture in captures} != set(_DECISIONS) or len(config_hashes) != 1:
        raise ValueError("compatibility captures are not one Dislike and one Like under one config")
    review = _read_bound_file(
        artifact["independent_review_path"], artifact["independent_review_sha256"],
        root=root, label="independent compatibility review",
    )
    review_body = {key: value for key, value in review.items() if key != "evidence_sha256"}
    if (
        review.get("evidence_sha256") != canonical_sha256(review_body)
        or review.get("kind") != "hinge_unattended_calibration_independent_review"
        or review.get("human_ground_truth") is not False
        or not isinstance(review.get("config"), dict)
        or review["config"].get("sha256") not in config_hashes
        or not isinstance(review.get("captures"), list)
        or len(review["captures"]) != 2
    ):
        raise ValueError("independent compatibility review is missing or mismatched")
    seen = set()
    for record in review["captures"]:
        if not isinstance(record, dict):
            raise ValueError("independent compatibility review has a malformed capture")
        path = record.get("session")
        pair = manifest_by_path.get(path)
        if pair is None or path in seen or record.get("manifest_sha256") != pair[0] or record.get("terminal_advance_action") != _DECISIONS[pair[1]]:
            raise ValueError("independent compatibility review omits or changes a decision")
        seen.add(path)
    if seen != set(manifest_by_path):
        raise ValueError("independent compatibility review does not cover both captures")
    return target
