"""Offline two-decision review for a Hinge update, without re-freezing old bounds.

Input captures are made by ``hinge_calibrate capture --hybrid-review --target-items
photo-1-only --profiles 1`` on two distinct profiles.  The first ends in Pass, the second
in a real Send Priority Like.  This tool checks the independent frame review and applies
the original identity/item bounds unchanged to the new-build captures.  Its output licenses
Training only and explicitly keeps video false-accept risk unmeasured.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image

from operation_love import config
from operation_love.drivers.hinge import HingeDriver
from operation_love.hinge_compatibility import (
    ACCEPTANCE,
    KIND,
    PUBLIC_RECEIPT_KIND,
    SUPERVISED_KIND,
    _SUPERVISED_FILES,
    _DIRECT_SEND_FILES,
    canonical_sha256,
    validate_review,
)
from tools import hinge_calibrate as calibration
from tools.hinge_calibration_review import build_review


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root))


def produce_redacted_receipt(
    *,
    config_path: Path,
    private_artifact_path: Path,
    output_path: Path,
) -> dict:
    """Publish a portable, non-image receipt after validating the private review in full."""
    root = Path.cwd().resolve()
    cfg = config.load(config_path)
    config.validate(cfg)
    app = cfg.apps["hinge"]
    private_ref = {
        "artifact_path": _repo_relative(private_artifact_path, root),
        "artifact_sha256": _sha(private_artifact_path),
    }
    private = json.loads(private_artifact_path.read_text())
    if private.get("kind") != SUPERVISED_KIND:
        raise ValueError(
            "redacted receipt requires a supervised two-pass private artifact"
        )
    target = validate_review(
        {**app, "version_compatibility_review": private_ref}, root=root, mode="training"
    )
    body = {
        "schema_version": 3,
        "kind": PUBLIC_RECEIPT_KIND,
        "status": "training_only",
        "acceptance": ACCEPTANCE,
        "human_ground_truth": False,
        "video_false_accept_rate": "unmeasured",
        "source_build": private["source_build"],
        "target_build": target,
        "device": private["device"],
        "frame_size_px": private["frame_size_px"],
        "source_calibration_sha256": private["source_calibration_sha256"],
        "source_licence_key": private["source_licence_key"],
        "source_licence_sha256": private["source_licence_sha256"],
        "decision_order": ["dislike", "like"],
        "like_sent_confirmed": True,
        "private_artifact_sha256": private_ref["artifact_sha256"],
        "like_template_path": private["like_template_path"],
        "like_template_sha256": private["like_template_sha256"],
        "reviewed_at": private["reviewed_at"],
        "release_notes_url": private["release_notes_url"],
        "release_notes_summary": private["release_notes_summary"],
    }
    receipt = {**body, "evidence_sha256": canonical_sha256(body)}
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    mapping = {
        "artifact_path": _repo_relative(output_path, root),
        "artifact_sha256": _sha(output_path),
    }
    validate_review(
        {**app, "version_compatibility_review": mapping}, root=root, mode="training"
    )
    return mapping


def produce_supervised(
    *,
    config_path: Path,
    evidence_dir: Path,
    output_path: Path,
    dislike_profile: str,
    like_profile: str,
    release_notes_url: str,
    release_notes_summary: str,
    confirmation: str,
    direct_send: bool = False,
) -> dict:
    """Bind a visibly reviewed Dislike, photo heart, Send Like, and upsell outcome.

    The two decisions remain qualitative compatibility checks.  The old numerical bounds
    and video false-accept rate are neither remeasured nor relabeled as measured.
    """
    if confirmation != ACCEPTANCE:
        raise ValueError(f"exact --confirmation {ACCEPTANCE} is required")
    root = Path.cwd().resolve()
    cfg = config.load(config_path)
    config.validate(cfg)
    app = cfg.apps["hinge"]
    if cfg.mode != "training" or app.get("mode", cfg.mode) != "training":
        raise ValueError("supervised compatibility review licenses Training only")
    old = app.get("targeting_calibration")
    licence_key = (
        "still_photo_bound_evidence"
        if "still_photo_bound_evidence" in app
        else "still_photo_assumption_acceptance"
    )
    licence = app.get(licence_key)
    if not isinstance(old, dict) or not isinstance(licence, dict):
        raise ValueError("source calibration and still-photo licence are required")
    if (
        not dislike_profile.strip()
        or not like_profile.strip()
        or dislike_profile == like_profile
    ):
        raise ValueError("two distinct profile names are required")
    if not release_notes_url.strip() or not release_notes_summary.strip():
        raise ValueError("release notes source and findings are required")
    serial = app["serial"]
    package = subprocess.run(
        ["adb", "-s", serial, "shell", "dumpsys", "package", "co.hinge.app"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout
    version_match = re.search(r"\bversionName=([^\s]+)", package)
    if version_match is None:
        raise ValueError("live Hinge package version could not be read")
    target = version_match.group(1)
    if target == old["hinge_version_name"]:
        raise ValueError("live Hinge already matches the source calibration")
    files = {}
    for name in _DIRECT_SEND_FILES if direct_send else _SUPERVISED_FILES:
        path = evidence_dir / name
        if not path.is_file():
            raise ValueError(f"missing supervised evidence {name}")
        files[name] = {"path": _repo_relative(path, root), "sha256": _sha(path)}
    for name in (
        "dislike_before.png",
        "like_before.png",
        "send_settled.png" if direct_send else "upsell_after.png",
    ):
        with Image.open(evidence_dir / name) as image:
            if list(image.size) != old["frame_size_px"]:
                raise ValueError(f"{name} frame size differs from source calibration")

    def ocr(name: str) -> str:
        return subprocess.run(
            ["tesseract", str(evidence_dir / name), "stdout"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout

    dislike_ocr = ocr("dislike_before.png")
    like_ocr = ocr("like_before.png")
    confirmation_ocr = (
        "Continuous direct-send recording reviewed"
        if direct_send
        else " ".join(ocr("upsell_after.png").split())
    )
    if dislike_profile not in dislike_ocr or like_profile not in like_ocr:
        raise ValueError("profile names are not visible in the reviewed before frames")
    if not direct_send and (
        "your Like sooner" not in confirmation_ocr
        or like_profile not in confirmation_ocr
    ):
        raise ValueError("Hinge's sent-Like confirmation is not visible")
    dislike = json.loads((evidence_dir / "dislike.json").read_text())
    like = json.loads((evidence_dir / "like.json").read_text())
    upsell = (
        {} if direct_send else json.loads((evidence_dir / "upsell.json").read_text())
    )
    if (
        dislike.get("decision") != "dislike"
        or dislike.get("before_sha256") != files["dislike_before.png"]["sha256"]
        or dislike.get("after_sha256") != files["dislike_after.png"]["sha256"]
        or like.get("decision") != "like"
        or like.get("before_sha256") != files["send_before.png"]["sha256"]
        or like.get("after_sha256") != files["send_after.png"]["sha256"]
    ):
        raise ValueError("decision records do not bind the reviewed frames")
    if not direct_send and (
        upsell.get("decision") != "like"
        or upsell.get("upsell_before_sha256") != files["upsell_before.png"]["sha256"]
        or upsell.get("upsell_after_sha256") != files["upsell_after.png"]["sha256"]
        or upsell.get("dismissed") is not True
        or files["send_after.png"]["sha256"] != files["upsell_before.png"]["sha256"]
    ):
        raise ValueError("decision records do not bind the reviewed frames")
    template_path = Path("operation_love/drivers/assets/hinge_like_button_10_8.png")
    driver = HingeDriver(cfg)
    driver._targeting_runtime_version_name = target
    if direct_send:
        review = json.loads((evidence_dir / "transition_review.json").read_text())
        for name in ("send_after.png", "send_settled.png"):
            if not driver._observe_deck_ready((evidence_dir / name).read_bytes()):
                raise ValueError("direct-send successor is not a confirmed deck")
            if review["after_profile"] not in ocr(name):
                raise ValueError("direct-send successor name is not visible")
        # A recording review never replaces a visible source composer and send control.
        import io

        with Image.open(evidence_dir / "send_before.png") as frame:
            for box, expected_text in (
                ((0, 100, frame.width, 230), like_profile),
                ((350, 1300, 1000, 1460), "Send Priority Like"),
            ):
                output = io.BytesIO()
                frame.crop(box).save(output, format="PNG")
                text = subprocess.run(
                    ["tesseract", "stdin", "stdout", "--psm", "6"],
                    input=output.getvalue(),
                    capture_output=True,
                    check=True,
                    timeout=20,
                ).stdout.decode()
                if expected_text not in text:
                    raise ValueError("direct-send composer/name/control is not visible")
    if not driver._observe_deck_ready(
        (evidence_dir / "dislike_before.png").read_bytes()
    ):
        raise ValueError(
            "reviewed Dislike frame is not a confirmed deck under the fixed matcher"
        )
    if not driver._observe_deck_ready((evidence_dir / "like_before.png").read_bytes()):
        raise ValueError(
            "reviewed Like frame is not a confirmed deck under the fixed matcher"
        )
    body = {
        "schema_version": 4 if direct_send else 2,
        "kind": SUPERVISED_KIND,
        "status": "training_only",
        "acceptance": ACCEPTANCE,
        "human_ground_truth": False,
        "video_false_accept_rate": "unmeasured",
        "source_build": old["hinge_version_name"],
        "target_build": target,
        "device": serial,
        "frame_size_px": old["frame_size_px"],
        "source_calibration_sha256": canonical_sha256(old),
        "source_licence_key": licence_key,
        "source_licence_sha256": canonical_sha256(licence),
        "files": files,
        "dislike_profile_name": dislike_profile,
        "like_profile_name": like_profile,
        "like_confirmation_text": confirmation_ocr,
        "send_point": like["send_point"],
        "like_template_path": _repo_relative(template_path, root),
        "like_template_sha256": _sha(template_path),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "release_notes_url": release_notes_url.strip(),
        "release_notes_summary": release_notes_summary.strip(),
    }
    artifact = {**body, "evidence_sha256": canonical_sha256(body)}
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    output_path.chmod(0o600)
    mapping = {
        "artifact_path": _repo_relative(output_path, root),
        "artifact_sha256": _sha(output_path),
    }
    validate_review(
        {**app, "version_compatibility_review": mapping}, root=root, mode="training"
    )
    return mapping


def produce(
    *,
    config_path: Path,
    dislike_dir: Path,
    like_dir: Path,
    review_path: Path,
    output_path: Path,
    release_notes_url: str,
    release_notes_summary: str,
    confirmation: str,
) -> dict:
    if confirmation != ACCEPTANCE:
        raise ValueError(f"exact --confirmation {ACCEPTANCE} is required")
    root = Path.cwd().resolve()
    cfg = config.load(config_path)
    config.validate(cfg)
    if (
        cfg.mode != "training"
        or (cfg.apps.get("hinge", {}) or {}).get("mode", cfg.mode) != "training"
    ):
        raise ValueError("compatibility review can license Training only")
    app = cfg.apps["hinge"]
    old = app.get("targeting_calibration")
    licence_key = (
        "still_photo_bound_evidence"
        if "still_photo_bound_evidence" in app
        else "still_photo_assumption_acceptance"
    )
    licence = app.get(licence_key)
    if not isinstance(old, dict) or not isinstance(licence, dict):
        raise ValueError("source calibration and still-photo licence are required")
    if not release_notes_url.strip() or not release_notes_summary.strip():
        raise ValueError("release-notes source and findings are required")
    sessions = [calibration._load_session(path) for path in (dislike_dir, like_dir)]
    if any(len(session.profiles) != 1 for session in sessions):
        raise ValueError("exactly one profile is required in each capture")
    if sessions[0].profiles[0].profile_id == sessions[1].profiles[0].profile_id:
        raise ValueError("Dislike and Like must be on distinct real profiles")
    metadata = [session.manifest for session in sessions]
    versions = {manifest["device"]["hinge_version_name"] for manifest in metadata}
    if len(versions) != 1:
        raise ValueError("the two captures disagree on live Hinge version")
    (target,) = versions
    if target == old["hinge_version_name"]:
        raise ValueError("the live build already matches the source calibration")
    for session, terminal in zip(
        sessions, ("automated_pass", "automated_send_priority_like"), strict=True
    ):
        manifest = session.manifest
        device = manifest["device"]
        if (
            device["serial"] != old["device"]
            or manifest["frame_size_px"] != old["frame_size_px"]
            or manifest["automation_acceptance"].get("terminal_advance_action")
            != terminal
            or manifest.get("automated_target_strategy_id") != "photo_1_only_v1"
        ):
            raise ValueError(
                "capture device, frame, target strategy, or terminal action differs"
            )
    config_hash = _sha(config_path)
    independent = json.loads(review_path.read_text())
    expected = build_review([dislike_dir, like_dir], config_path)
    if independent != {**expected, "evidence_sha256": canonical_sha256(expected)}:
        raise ValueError(
            "independent review differs from freshly replayed capture evidence"
        )
    if expected["config"]["sha256"] != config_hash:
        raise ValueError("independent review was made under different config bytes")
    calibration._verified_hybrid_reviewed_evidence(sessions)
    driver = HingeDriver(cfg)  # offline: no ADB session or touch transport
    profiles = [session.profiles[0] for session in sessions]
    identity_band, content_band = (
        tuple(driver.identity_band),
        tuple(driver.content_band),
    )
    calibration._validate_profile_advance_clears(
        profiles,
        identity_band=identity_band,
        confirm_template=driver._template("confirm"),
    )
    identity_samples = calibration._profile_identity_samples(
        profiles, identity_band=identity_band
    )
    identity_ok, identity_reason, _ = calibration._apply_identity_bound(
        identity_samples, float(old["identity_match_max_dist"])
    )
    if not identity_ok:
        raise ValueError(f"old identity bound fails on new build: {identity_reason}")
    payloads = calibration._build_profile_payloads(
        profiles,
        content_band=content_band,
        identity_band=identity_band,
        like_template=driver._template("like"),
        like_threshold=calibration.hinge_mod._LIKE_MATCH_THRESHOLD,
    )
    inline_records = calibration._inline_distance_records(
        payloads, confirm_template=driver._template("confirm")
    )
    inline_ok, inline_reason, _ = calibration._apply_inline_bound(
        inline_records, float(old["inline_item_max_dist"])
    )
    if not inline_ok:
        raise ValueError(f"old inline-item bound fails on new build: {inline_reason}")
    captures = [
        {
            "decision": decision,
            "manifest_path": _repo_relative(session.dir / "manifest.json", root),
            "manifest_sha256": _sha(session.dir / "manifest.json"),
        }
        for decision, session in zip(("dislike", "like"), sessions, strict=True)
    ]
    body = {
        "schema_version": 1,
        "kind": KIND,
        "status": "training_only",
        "acceptance": ACCEPTANCE,
        "human_ground_truth": False,
        "video_false_accept_rate": "unmeasured",
        "source_build": old["hinge_version_name"],
        "target_build": target,
        "device": old["device"],
        "frame_size_px": old["frame_size_px"],
        "source_calibration_sha256": canonical_sha256(old),
        "source_licence_key": licence_key,
        "source_licence_sha256": canonical_sha256(licence),
        "captures": captures,
        "independent_review_path": _repo_relative(review_path, root),
        "independent_review_sha256": _sha(review_path),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "release_notes_url": release_notes_url.strip(),
        "release_notes_summary": release_notes_summary.strip(),
    }
    artifact = {**body, "evidence_sha256": canonical_sha256(body)}
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    output_path.chmod(0o600)
    config_mapping = {
        "artifact_path": _repo_relative(output_path, root),
        "artifact_sha256": _sha(output_path),
    }
    # Check the bytes just written through the same path production config validation uses.
    validate_review(
        {**app, "version_compatibility_review": config_mapping},
        root=root,
        mode="training",
    )
    return config_mapping


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dislike")
    parser.add_argument("--like")
    parser.add_argument("--independent-review")
    parser.add_argument("--supervised-dir")
    parser.add_argument(
        "--direct-send",
        action="store_true",
        help="require a separately reviewed continuous recording of a direct send",
    )
    parser.add_argument("--private-artifact")
    parser.add_argument("--dislike-profile")
    parser.add_argument("--like-profile")
    parser.add_argument("--out", required=True)
    parser.add_argument("--release-notes-url")
    parser.add_argument("--release-notes-summary")
    parser.add_argument("--confirmation")
    args = parser.parse_args(argv)
    if args.direct_send and (not args.supervised_dir or args.private_artifact):
        parser.error("--direct-send requires --supervised-dir")
    try:
        if args.private_artifact:
            mapping = produce_redacted_receipt(
                config_path=Path(args.config),
                private_artifact_path=Path(args.private_artifact),
                output_path=Path(args.out),
            )
        elif args.supervised_dir:
            if (
                not args.release_notes_url
                or not args.release_notes_summary
                or not args.confirmation
            ):
                raise ValueError(
                    "supervised review needs release notes and confirmation"
                )
            if not args.dislike_profile or not args.like_profile:
                raise ValueError(
                    "--supervised-dir needs --dislike-profile and --like-profile"
                )
            mapping = produce_supervised(
                config_path=Path(args.config),
                evidence_dir=Path(args.supervised_dir),
                output_path=Path(args.out),
                dislike_profile=args.dislike_profile,
                like_profile=args.like_profile,
                release_notes_url=args.release_notes_url,
                release_notes_summary=args.release_notes_summary,
                confirmation=args.confirmation,
                direct_send=args.direct_send,
            )
        else:
            if not args.dislike or not args.like or not args.independent_review:
                raise ValueError(
                    "capture review needs --dislike, --like, and --independent-review"
                )
            if (
                not args.release_notes_url
                or not args.release_notes_summary
                or not args.confirmation
            ):
                raise ValueError("capture review needs release notes and confirmation")
            mapping = produce(
                config_path=Path(args.config),
                dislike_dir=Path(args.dislike),
                like_dir=Path(args.like),
                review_path=Path(args.independent_review),
                output_path=Path(args.out),
                release_notes_url=args.release_notes_url,
                release_notes_summary=args.release_notes_summary,
                confirmation=args.confirmation,
            )
    except Exception as exc:  # noqa: BLE001 -- offline evidence failures are refusal, not fallback
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("apps:\n  hinge:\n    version_compatibility_review:")
    for key, value in mapping.items():
        print(f"      {key}: {value}")


if __name__ == "__main__":
    main()
