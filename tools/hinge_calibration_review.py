"""Independently authenticate automated Hinge calibration captures, offline.

This is intentionally a small, standard-library-only *second process*.  It neither imports the
capture/measurement harness nor any driver, vision, ADB, or touch code.  It can therefore check
that an automated capture's declared provenance and bytes are internally consistent without
using the same targeting stack that produced the trace.

It does not make automated evidence human ground truth and it never releases AUTO. Its self-hashed
artifact binds the exact capture manifests, every saved PNG digest, device/build/frame evidence,
and exact config.yaml bytes.  ``hinge_calibrate measure --accept-automated-circular-evidence``
requires this artifact for unattended candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from operation_love.private_files import atomic_write_private_text, ensure_private_dir


_TOOL_VERSION = "1"
_REVIEW_SCHEMA_VERSION = 1
_CAPTURE_PROVENANCE_SCHEMA_VERSION = 1
_CAPTURE_MODE = "automated_circular_risk_accepted"
_HYBRID_CAPTURE_MODE = "hybrid_ai_reviewed_automation"
_CONFIRMATION = "I_ACCEPT_UNATTENDED_CIRCULAR_CALIBRATION_RISK"
_HYBRID_CONFIRMATION = "I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK"
# Mirrors tools.hinge_calibrate._SEND_LIKE_CONFIRMATION. This module stays standard-library-only
# and deliberately imports nothing from the capture tool, so the phrase is duplicated rather than
# shared: an independent reviewer must not depend on the code whose output it is reviewing.
_SEND_LIKE_CONFIRMATION = "I_ACCEPT_REAL_PRIORITY_LIKE_SEND_RISK"
_AUTOMATED_TARGET_STRATEGY_ID = "alternate_photo_1_3_by_profile_ordinal_v1"
# Mirrors tools.hinge_calibrate._PHOTO_1_ONLY_TARGET_STRATEGY_ID, duplicated for the same reason
# _AUTOMATED_TARGET_STRATEGY_ID above is: this module never imports the capture tool it reviews.
# Owner-selected second strategy (2026-08-22): always targets photo model item 1, an explicit,
# accepted narrowing that does not exercise deep-item navigation.
_PHOTO_1_ONLY_TARGET_STRATEGY_ID = "photo_1_only_v1"
_ACCEPTED_TARGET_STRATEGY_IDS = frozenset(
    (_AUTOMATED_TARGET_STRATEGY_ID, _PHOTO_1_ONLY_TARGET_STRATEGY_ID))
_TARGET_SCOPED_PREFIX_PROOF_ID = "photo_only_confirmed_prefix_v1"
_MAX_PREACTION_PROFILE_SKIPS_PER_ORDINAL = 3
_MAX_PREACTION_PROFILE_SKIPS_PER_SESSION = 12
_SPLITS = frozenset(("calibration", "heldout"))
_DEVICE_KEYS = ("serial", "model", "display_w", "display_h", "density", "hinge_package",
                "hinge_version_name")
_CARD_SCROLL_ROLE = "card_scroll"
_SINGLETON_FRAME_ROLES = frozenset((
    "target_pre", "composer_open", "profile_advance_clear", "profile_advance_identity"))


class ReviewRefused(RuntimeError):
    """A capture or review is not exact enough to authenticate offline."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _png_size(raw: bytes, *, context: str) -> list[int]:
    """Read only the fixed PNG IHDR; no image/vision dependency belongs in this reviewer."""
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ReviewRefused(f"{context} is not a readable PNG with an IHDR")
    width, height = struct.unpack(">II", raw[16:24])
    if width < 1 or height < 1:
        raise ReviewRefused(f"{context} has invalid PNG dimensions")
    return [width, height]


def _require_exact_keys(value: object, keys: set[str], *, context: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ReviewRefused(f"{context} has wrong schema keys: {actual!r}")
    return value


def _expected_composer_items(
        ordinal: object, *, strategy_id: str = _AUTOMATED_TARGET_STRATEGY_ID) -> tuple[int, ...]:
    """The capture strategy's photo target(s) for a 1-based ordinal.

    Mirrors tools.hinge_calibrate._automated_composer_items_for_ordinal (duplicated, not
    imported -- see module docstring). `strategy_id` defaults to the alternating strategy so
    every existing caller is unaffected.
    """
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ReviewRefused(f"invalid automated profile ordinal {ordinal!r}")
    if strategy_id == _AUTOMATED_TARGET_STRATEGY_ID:
        return (1 if ordinal % 2 else 3,)
    if strategy_id == _PHOTO_1_ONLY_TARGET_STRATEGY_ID:
        return (1,)
    raise ReviewRefused(f"unknown automated target strategy id {strategy_id!r}")


def _automated_target_depths_for_strategy(strategy_id: str) -> frozenset[int]:
    """All distinct photo model items `strategy_id` can ever target.

    Mirrors tools.hinge_calibrate._automated_target_depths_for_strategy (duplicated, not
    imported -- see module docstring).  Derived from `_expected_composer_items` itself (ordinals 1
    and 2 span every branch of both known strategies) so this can never drift from that single
    source of truth.  Raises the same ReviewRefused for an unknown id.
    """
    depths: set[int] = set()
    for ordinal in (1, 2):
        depths.update(_expected_composer_items(ordinal, strategy_id=strategy_id))
    return frozenset(depths)


def _approved_hybrid_decision(record: object, *, acceptance: dict, decisions: list,
                              action: str, item: int | None) -> bool:
    """Keep reviewer-ledger binding independent from the capture implementation."""
    plan = record.get("action_plan") if isinstance(record, dict) else None
    return (isinstance(record, dict) and record.get("decision") == "approved"
            and record.get("source") == acceptance.get("reviewer_source")
            and record.get("human_ground_truth") is False
            and record.get("reviewer") == acceptance.get("reviewer")
            and isinstance(plan, dict) and plan.get("action") == action
            and plan.get("photo_model_item") == item
            and isinstance(record.get("checkpoint_evidence_sha256"), str)
            and isinstance(record.get("frame_sha256"), str)
            and record in decisions)


def _exact_hybrid_terminal_checkpoint(action: object, *, session: Path,
                                      config_sha256: str, acceptance: dict,
                                      decisions: list, expected_action: str,
                                      expected_point: list[int] | None) -> bool:
    """Open and authenticate one terminal checkpoint's JSON, PNG and exact plan."""
    if not isinstance(action, dict):
        return False
    checks = action.get("review_checkpoints")
    before = checks.get("before") if isinstance(checks, dict) else None
    if (not _approved_hybrid_decision(
            before, acceptance=acceptance, decisions=decisions,
            action=expected_action, item=None)
            or not isinstance(before, dict)):
        return False
    checkpoint_file, frame_file = before.get("checkpoint_file"), before.get("frame_file")
    if not isinstance(checkpoint_file, str) or not isinstance(frame_file, str):
        return False
    try:
        review_dir = (session / "hybrid_review").resolve(strict=True)
        checkpoint_path = Path(checkpoint_file).resolve(strict=True)
        checkpoint_frame_path = Path(frame_file).resolve(strict=True)
        if (checkpoint_path.parent != review_dir or checkpoint_frame_path.parent != review_dir
                or not checkpoint_path.is_file() or not checkpoint_frame_path.is_file()):
            return False
        checkpoint_raw = checkpoint_path.read_bytes()
        checkpoint = json.loads(checkpoint_raw)
        checkpoint_frame_sha256 = _sha256(checkpoint_frame_path.read_bytes())
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False
    if not isinstance(checkpoint, dict):
        return False
    checkpoint_body = dict(checkpoint)
    checkpoint_body.pop("evidence_sha256", None)
    plan = checkpoint.get("action_plan")
    predicates = plan.get("predicates") if isinstance(plan, dict) else None
    frame = checkpoint.get("frame")
    expected_source = ("calibration-only verified-composer Send transport"
                       if expected_action == "automated_send_priority_like"
                       else "calibration-only verified-composer Pass transport")
    return (
        before.get("checkpoint_sha256") == _sha256(checkpoint_raw)
        and checkpoint.get("evidence_sha256") == before.get("checkpoint_evidence_sha256")
        and checkpoint.get("evidence_sha256") == _canonical_digest(checkpoint_body)
        and checkpoint.get("schema_version") == 1
        and checkpoint.get("kind") == "hinge_hybrid_calibration_checkpoint"
        and checkpoint.get("config_sha256") == config_sha256
        and checkpoint.get("human_ground_truth") is False
        and checkpoint.get("claimed_state") == "composer_open_before_pass"
        and before.get("claimed_state") == checkpoint.get("claimed_state")
        and plan == before.get("action_plan")
        and isinstance(plan, dict)
        and plan.get("action") == expected_action
        and plan.get("photo_model_item") is None
        and plan.get("point") == expected_point
        and plan.get("point_source") == expected_source
        and isinstance(predicates, dict)
        and predicates.get(
            "inline_composer_and_selected_photo_verified_before_action") is True
        and predicates.get("send_like_tapped") is False
        and predicates.get("forbidden_zone_guarded_transport") == "HingeDriver._tap"
        and isinstance(frame, dict)
        and frame.get("file") == checkpoint_frame_path.name
        and frame.get("sha256") == before.get("frame_sha256")
        and frame.get("sha256") == checkpoint_frame_sha256
        and frame.get("sha256") == action.get("pre_frame_sha256")
    )


def _read_capture(session: Path, config_sha256: str) -> dict:
    manifest_path = session / "manifest.json"
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewRefused(f"{session}: cannot read manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReviewRefused(f"{session}: manifest is not a mapping")
    if manifest.get("interrupted") is not False:
        raise ReviewRefused(f"{session}: interrupted capture cannot authenticate evidence")
    if manifest.get("unattended_provenance_schema_version") != _CAPTURE_PROVENANCE_SCHEMA_VERSION:
        raise ReviewRefused(f"{session}: unsupported/missing unattended provenance schema")
    mode = manifest.get("capture_mode")
    if mode not in {_CAPTURE_MODE, _HYBRID_CAPTURE_MODE} or manifest.get("human_ground_truth") is not False:
        raise ReviewRefused(f"{session}: review accepts only explicit automated, non-human evidence modes")
    acceptance = manifest.get("automation_acceptance")
    expected_confirmation = _HYBRID_CONFIRMATION if mode == _HYBRID_CAPTURE_MODE else _CONFIRMATION
    if not isinstance(acceptance, dict) or acceptance.get("confirmation") != expected_confirmation:
        raise ReviewRefused(f"{session}: missing exact automated-risk acknowledgement")
    if (acceptance.get("not_independent_ground_truth") is not True
            or acceptance.get("not_supervised_operational_evidence") is not True):
        raise ReviewRefused(f"{session}: circular-risk provenance is incomplete")
    manifest_strategy_id = manifest.get("automated_target_strategy_id")
    acceptance_strategy_id = acceptance.get("target_strategy_id")
    if (manifest_strategy_id not in _ACCEPTED_TARGET_STRATEGY_IDS
            or acceptance_strategy_id not in _ACCEPTED_TARGET_STRATEGY_IDS
            or manifest_strategy_id != acceptance_strategy_id):
        # Membership alone is not enough: a manifest could legitimately use one accepted id while
        # its acceptance block claims another. Every reference to the strategy within one session
        # must agree with every other, never merely with the accepted set independently.
        raise ReviewRefused(
            f"{session}: unsupported, missing, or inconsistent automated target strategy "
            f"(manifest={manifest_strategy_id!r}, acceptance={acceptance_strategy_id!r})")
    strategy_id = manifest_strategy_id
    # A capture ends each profile either with the default Pass-without-send or -- only when the
    # owner explicitly accepted it at capture time -- with a REAL Send Priority Like. The
    # acceptance is read from the manifest rather than inferred from the traces, so a send trace
    # can never authenticate itself: an unaccepted session still requires the exact Pass trace.
    send_like_accepted = acceptance.get("send_like_accepted")
    if send_like_accepted is not None and type(send_like_accepted) is not bool:
        raise ReviewRefused(f"{session}: send_like_accepted is not an exact boolean")
    send_like_accepted = bool(send_like_accepted)
    if send_like_accepted and acceptance.get("send_like_confirmation") != _SEND_LIKE_CONFIRMATION:
        raise ReviewRefused(f"{session}: real-send evidence lacks the exact send-like acknowledgement")
    if not send_like_accepted and acceptance.get("send_like_confirmation") is not None:
        raise ReviewRefused(f"{session}: send-like confirmation present without accepted real sends")
    terminal_action = ("automated_send_priority_like" if send_like_accepted else "automated_pass")
    declared_terminal = acceptance.get("terminal_advance_action")
    if declared_terminal is not None and declared_terminal != terminal_action:
        raise ReviewRefused(f"{session}: declared terminal advance action contradicts its acceptance")
    target_scoped = manifest.get("capture_evidence_scope") == _TARGET_SCOPED_PREFIX_PROOF_ID
    provenance = _require_exact_keys(
        manifest.get("config_provenance"),
        {"schema_version", "path", "sha256", "effective_hinge_sha256"},
        context=f"{session}: config_provenance")
    if provenance["schema_version"] != _CAPTURE_PROVENANCE_SCHEMA_VERSION:
        raise ReviewRefused(f"{session}: unsupported config provenance schema")
    if not isinstance(provenance["sha256"], str) or provenance["sha256"] != config_sha256:
        raise ReviewRefused(f"{session}: exact config bytes differ from capture provenance")
    device = manifest.get("device")
    if not isinstance(device, dict) or any(device.get(key) in (None, "") for key in _DEVICE_KEYS):
        raise ReviewRefused(f"{session}: incomplete device/build evidence")
    frame_size = manifest.get("frame_size_px")
    if (not isinstance(frame_size, list) or len(frame_size) != 2 or frame_size !=
            [device["display_w"], device["display_h"]]):
        raise ReviewRefused(f"{session}: frame size does not bind device evidence")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ReviewRefused(f"{session}: no profiles")
    if manifest.get("split") not in _SPLITS:
        raise ReviewRefused(f"{session}: missing or unsupported split binding")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or manifest.get("frame_count") != len(frames):
        raise ReviewRefused(f"{session}: frame count/schema mismatch")
    # A profile can have an ordered series of card-scroll frames while locating its target.
    # Every action-bound role remains a singleton; the complete per-profile sequence is checked
    # below once the declared target item is available from ``profiles``.
    frame_by_role_item: dict[tuple[int, str, int | None], str] = {}
    frame_roles_by_ordinal: dict[int, list[tuple[str, int | None]]] = {}
    seen_names: set[str] = set()
    for sequence, rec in enumerate(frames, 1):
        if not isinstance(rec, dict):
            raise ReviewRefused(f"{session}: frame record {sequence} is not a mapping")
        name = rec.get("file")
        if not isinstance(name, str) or name != f"{sequence:05d}.png" or name in seen_names:
            raise ReviewRefused(f"{session}: non-contiguous or duplicate frame filename")
        seen_names.add(name)
        path = session / name
        try:
            bytes_ = path.read_bytes()
        except OSError as exc:
            raise ReviewRefused(f"{session}: missing recorded frame {name}") from exc
        if rec.get("sha256") != _sha256(bytes_):
            raise ReviewRefused(f"{session}: frame digest mismatch for {name}")
        if _png_size(bytes_, context=f"{session}/{name}") != frame_size:
            raise ReviewRefused(f"{session}: frame dimensions differ from captured framebuffer")
        ordinal, role, item = rec.get("profile_ordinal"), rec.get("role"), rec.get("item_number")
        if (not isinstance(ordinal, int) or ordinal < 1
                or role not in {_CARD_SCROLL_ROLE, *_SINGLETON_FRAME_ROLES}):
            raise ReviewRefused(f"{session}: malformed frame role binding")
        if role == _CARD_SCROLL_ROLE:
            if item is not None:
                raise ReviewRefused(f"{session}: card_scroll frame has an item number")
        else:
            if role in {"target_pre", "composer_open"}:
                if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                    raise ReviewRefused(f"{session}: {role} frame has an invalid item number")
            elif item is not None:
                raise ReviewRefused(f"{session}: {role} frame has an item number")
            key = (ordinal, role, item)
            if key in frame_by_role_item:
                raise ReviewRefused(f"{session}: duplicate singleton frame role/item binding {key!r}")
            frame_by_role_item[key] = rec["sha256"]
        frame_roles_by_ordinal.setdefault(ordinal, []).append((role, item))
    if {path.name for path in session.glob("*.png")} != seen_names:
        raise ReviewRefused(f"{session}: unlisted or missing PNG evidence")
    # Target-scoped evidence intentionally does not claim the full closed-set replay that an
    # entry-anchor ledger represents.  Existing closed-set automated evidence remains readable
    # so a reviewer can still authenticate prior completed captures.
    ledger = manifest.get("entry_anchor_ledger")
    if target_scoped:
        if ledger is not None:
            raise ReviewRefused(f"{session}: target-scoped capture must not carry a closed-set entry-anchor ledger")
    else:
        if not isinstance(ledger, dict) or set(ledger) != {"file", "sha256"}:
            raise ReviewRefused(f"{session}: missing hash-bound entry-anchor ledger")
        ledger_path = session / ledger["file"]
        if not isinstance(ledger["file"], str) or ledger_path.name != ledger["file"]:
            raise ReviewRefused(f"{session}: unsafe entry-anchor ledger path")
        try:
            if _sha256(ledger_path.read_bytes()) != ledger["sha256"]:
                raise ReviewRefused(f"{session}: entry-anchor ledger digest mismatch")
        except OSError as exc:
            raise ReviewRefused(f"{session}: entry-anchor ledger missing") from exc
    hybrid_decisions = None
    if mode == _HYBRID_CAPTURE_MODE:
        hybrid = manifest.get("hybrid_review")
        if (not isinstance(hybrid, dict) or hybrid.get("schema_version") != 1
                or hybrid.get("protocol") != "stdin_checkpoint_sha256_v1"
                or hybrid.get("human_ground_truth") is not False
                or not isinstance(hybrid.get("decisions"), list)):
            raise ReviewRefused(f"{session}: hybrid capture has no supported review decision ledger")
        reviewer = acceptance.get("reviewer")
        required_reviewer = {"source", "id", "model", "version", "process"}
        if (not isinstance(reviewer, dict) or not required_reviewer.issubset(reviewer)
                or any(not isinstance(reviewer[key], str) or not reviewer[key].strip()
                       for key in required_reviewer)):
            raise ReviewRefused(f"{session}: hybrid capture lacks reviewer id/model/version/process")
        if acceptance.get("reviewer_source") != "external_ai_review" or hybrid.get("reviewer") != reviewer:
            raise ReviewRefused(f"{session}: hybrid capture has an unsupported reviewer source/binding")
        hybrid_decisions = hybrid["decisions"]

    def require_hybrid_decision(record: object, *, action: str, item: int | None) -> None:
        if not _approved_hybrid_decision(
                record, acceptance=acceptance, decisions=hybrid_decisions,
                action=action, item=item):
            raise ReviewRefused(f"{session}: hybrid reviewer decision is not an exact action binding")

    def has_exact_send_summary(action: object, *, ordinal: int) -> bool:
        """Authenticate an owner-accepted REAL Send Priority Like terminal action.

        This is deliberately not a relaxation of the Pass checks below: it is a separate, equally
        exact shape that is only reachable when the manifest carries the explicit send-like
        acceptance. It still requires the composer/selected-photo proof taken before the tap, the
        production upsell-dismiss + landed-verification transport (never the paid Rose control),
        and the same advance-frame binding, so a real like is held to the same evidence bar as a
        Pass -- while remaining plainly labelled as a send rather than passing for Pass-only.
        """
        if not isinstance(action, dict) or not send_like_accepted:
            return False
        action_predicates = action.get("predicates")
        transport = action.get("transport")
        point = action.get("confirm_point")
        if (not isinstance(action_predicates, dict) or not isinstance(transport, list)
                or transport != ["HingeDriver._tap(confirm_point)",
                                 "HingeDriver._handle_rose_upsell",
                                 "HingeDriver._verify_like_landed"]
                or action.get("send_like_tapped") is not True
                or action_predicates.get("send_like_tapped") is not True
                or action_predicates.get(
                    "inline_composer_and_selected_photo_verified_before_action") is not True
                or action_predicates.get("like_landed_verified") is not True
                or action.get("post_frame_sha256") != frame_by_role_item.get(
                    (ordinal, "profile_advance_clear", None))):
            return False
        if (not isinstance(point, list) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in point)):
            return False
        return _exact_hybrid_terminal_checkpoint(
            action, session=session, config_sha256=config_sha256,
            acceptance=acceptance, decisions=hybrid_decisions,
            expected_action="automated_send_priority_like", expected_point=point)

    def has_exact_pass_summary(action: object, *, ordinal: int) -> bool:
        """Accept legacy omitted pass summaries only when the saved bytes prove both facts.

        Early hybrid captures stored these two facts under the action predicate ledger rather
        than duplicating them at the action top level.  This compatibility path is deliberately
        unavailable to non-hybrid evidence and requires the immutable checkpoint plus the saved
        frame/transport bindings that make the predicate applicable to this exact Pass.
        """
        if not isinstance(action, dict):
            return False
        missing = object()
        sent, clear = action.get("send_like_tapped", missing), action.get("composer_clear_visible", missing)
        if mode != _HYBRID_CAPTURE_MODE:
            return sent is False and clear is True
        if ((sent is not missing or clear is not missing)
                and not (sent is False and clear is True)):
            return False
        if not _exact_hybrid_terminal_checkpoint(
                action, session=session, config_sha256=config_sha256,
                acceptance=acceptance, decisions=hybrid_decisions,
                expected_action="automated_pass", expected_point=None):
            return False
        action_predicates = action.get("predicates")
        transport = action.get("transport")
        if (not isinstance(action_predicates, dict) or not isinstance(transport, list)
                or any(not isinstance(step, str) for step in transport)
                or len(transport) < 2 or transport[-2:] != ["HingeDriver._locate_button(pass)",
                                                             "HingeDriver._tap"]
                or action_predicates.get("inline_composer_and_selected_photo_verified_before_action") is not True
                or action_predicates.get("inline_composer_structurally_confirmed_before_action") is not True
                or action_predicates.get("edge_back_transport_performed") is not True
                or action_predicates.get("pass_vision_relocated_before_guarded_tap") is not True
                or action_predicates.get("send_like_tapped") is not False
                or action_predicates.get("deck_frame_changed") is not True
                or action_predicates.get("composer_clear_visible") is not True
                or action_predicates.get("new_profile_top_confirmed") is not True
                or action.get("post_frame_sha256") != frame_by_role_item.get(
                    (ordinal, "profile_advance_clear", None))):
            return False
        return True

    skipped_attempts = manifest.get("skipped_attempts", [])
    if not isinstance(skipped_attempts, list):
        raise ReviewRefused(f"{session}: skipped_attempts is not a list")
    if len(skipped_attempts) > _MAX_PREACTION_PROFILE_SKIPS_PER_SESSION:
        raise ReviewRefused(f"{session}: skipped_attempts exceeds the bounded session cap")
    by_ordinal: dict[int, int] = {}
    skip_codes = {
        "target_unavailable_or_incomplete_index", "target_verification_blocked",
        "pre_heart_navigation_refused", "reviewer_requested_restart_before_first_heart",
        "unsupported_entry_layout",
    }
    for sequence, skipped in enumerate(skipped_attempts, 1):
        if not isinstance(skipped, dict):
            raise ReviewRefused(f"{session}: skipped attempt {sequence} is not a mapping")
        ordinal = skipped.get("ordinal")
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ReviewRefused(f"{session}: skipped attempt {sequence} has invalid ordinal")
        by_ordinal[ordinal] = by_ordinal.get(ordinal, 0) + 1
        if by_ordinal[ordinal] > _MAX_PREACTION_PROFILE_SKIPS_PER_ORDINAL:
            raise ReviewRefused(f"{session}: skipped attempts exceed the per-ordinal cap")
        expected_skip_action = ("advance_unusable_profile_with_priority_like"
                                if send_like_accepted else "skip_profile_without_heart")
        expected_skip_transport = ("HingeDriver.like" if send_like_accepted
                                   else "HingeDriver.dislike")
        if (skipped.get("attempt_number_for_ordinal") != by_ordinal[ordinal]
                or skipped.get("session_skip_number") != sequence
                or skipped.get("reason_code") not in skip_codes
                or skipped.get("action") != expected_skip_action
                or skipped.get("transport") != expected_skip_transport):
            raise ReviewRefused(f"{session}: skipped attempt {sequence} has an unsupported trace")
        unsupported_entry = skipped.get("reason_code") == "unsupported_entry_layout"
        hash_keys = ["reason_sha256", "pre_frame_sha256", "post_frame_sha256"]
        hash_keys.append("post_deck_frame_sha256" if unsupported_entry
                         else "post_identity_frame_sha256")
        for key in hash_keys:
            value = skipped.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ReviewRefused(f"{session}: skipped attempt {sequence} has invalid {key}")
        predicates = skipped.get("predicates")
        public_action_proved = isinstance(predicates, dict) and (
            (predicates.get("send_like_requested_for_unusable_profile") is True
             and predicates.get("send_like_tapped") is True
             and predicates.get("public_action_guard_used") is True)
            if send_like_accepted else
            (predicates.get("no_photo_heart_or_send_like_on_current_profile") is True
             and predicates.get("public_dislike_guard_used") is True))
        if unsupported_entry:
            if (not isinstance(predicates, dict)
                    or not public_action_proved
                    or skipped.get("pre_scroll_top_state") != "cannot_tell"
                    or predicates.get("pre_action_deck_ready") is not True
                    or predicates.get("pre_action_composer_absent") is not True
                    or predicates.get("calibration_top_unconfirmed") is not True
                    or predicates.get("public_action_progress_verified") is not True
                    or predicates.get("deck_frame_changed") is not True
                    or predicates.get("new_deck_ready") is not True
                    or predicates.get("new_profile_composer_absent") is not True
                    or predicates.get("new_profile_identity_distinct") is not False
                    or predicates.get("identity_comparison_not_claimed") is not True
                    or skipped.get("post_scroll_top_state") not in {
                        "confirmed_top", "confirmed_not_top", "cannot_tell"}):
                raise ReviewRefused(
                    f"{session}: skipped attempt {sequence} lacks unsupported-layout proofs")
        elif (not isinstance(predicates, dict)
              or not public_action_proved
              or predicates.get("new_profile_top_confirmed") is not True
              or predicates.get("new_profile_composer_absent") is not True
              or predicates.get("new_profile_identity_distinct") is not True):
            raise ReviewRefused(f"{session}: skipped attempt {sequence} lacks public-Pass proofs")
        settle = skipped.get("post_pass_settle")
        if (not isinstance(settle, dict)
                or not isinstance(settle.get("modal_edge_back_used"), bool)
                or settle.get("ordinary_deck_ready") is not True):
            raise ReviewRefused(f"{session}: skipped attempt {sequence} lacks settled-deck proof")
        for key in ("initial_post_pass_frame_sha256", "settled_post_pass_frame_sha256"):
            value = settle.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ReviewRefused(
                    f"{session}: skipped attempt {sequence} has invalid settled-deck {key}")
        if not unsupported_entry:
            distance = skipped.get("new_profile_identity_distance")
            if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                    or distance <= 2.565):
                raise ReviewRefused(
                    f"{session}: skipped attempt {sequence} lacks a distinct new-profile identity")
        if mode == _HYBRID_CAPTURE_MODE:
            checks = skipped.get("review_checkpoints")
            if not isinstance(checks, dict):
                raise ReviewRefused(f"{session}: hybrid skipped attempt has no review checkpoint")
            require_hybrid_decision(
                checks.get("before"), action=expected_skip_action, item=None)

    profile_ordinals: set[int] = set()
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("ordinal"), int):
            raise ReviewRefused(f"{session}: malformed profile trace")
        ordinal = profile["ordinal"]
        if ordinal < 1 or ordinal in profile_ordinals:
            raise ReviewRefused(f"{session}: duplicate or invalid profile ordinal {ordinal!r}")
        profile_ordinals.add(ordinal)
        if ordinal not in frame_roles_by_ordinal:
            raise ReviewRefused(f"{session}: profile has no saved frame sequence")
        if profile.get("action_evidence_mode") != mode:
            raise ReviewRefused(f"{session}: profile is not marked automated circular evidence")
        expected_items = _expected_composer_items(ordinal, strategy_id=strategy_id)
        if (profile.get("target_strategy_id") != strategy_id
                or (target_scoped and profile.get("capture_evidence_scope") != _TARGET_SCOPED_PREFIX_PROOF_ID)
                or profile.get("composer_items") != list(expected_items)):
            raise ReviewRefused(f"{session}: profile does not carry its exact alternating target trace")
        actual_sequence = frame_roles_by_ordinal[ordinal]
        card_count = sum(role == _CARD_SCROLL_ROLE for role, _item in actual_sequence)
        expected_sequence = [(_CARD_SCROLL_ROLE, None)] * card_count
        for item in expected_items:
            expected_sequence.extend((("target_pre", item), ("composer_open", item)))
        expected_sequence.extend((("profile_advance_clear", None),
                                  ("profile_advance_identity", None)))
        if card_count < 1 or actual_sequence != expected_sequence:
            raise ReviewRefused(
                f"{session}: profile {ordinal} frame role sequence {actual_sequence!r} does not "
                f"match its ordered card/action evidence")
        actions = profile.get("automated_actions")
        if not isinstance(actions, list):
            raise ReviewRefused(f"{session}: profile has no automated trace")
        hearts = [a for a in actions if isinstance(a, dict) and a.get("action") == "automated_photo_heart"]
        if [a.get("photo_model_item") for a in hearts] != list(expected_items):
            raise ReviewRefused(f"{session}: automated hearts do not match the ordinal target")
        for heart in hearts:
            item = heart["photo_model_item"]
            ordinal = profile["ordinal"]
            if (heart.get("post_tap_composer_verified") is not True
                    or heart.get("post_tap_item_relative_verified") is not True
                    or heart.get("pre_frame_sha256") != frame_by_role_item.get((ordinal, "target_pre", item))
                    or heart.get("post_frame_sha256") != frame_by_role_item.get((ordinal, "composer_open", item))):
                raise ReviewRefused(f"{session}: heart trace does not bind its saved frames")
            if target_scoped:
                proof = heart.get("target_scoped_prefix_proof")
                target_ordinal = proof.get("target_heart_ordinal") if isinstance(proof, dict) else None
                if (not isinstance(proof, dict)
                        or proof.get("id") != _TARGET_SCOPED_PREFIX_PROOF_ID
                        or proof.get("target_photo_model_item") != item
                        or proof.get("predecessors_resolved") is not True
                        or proof.get("target_crop_complete") is not True
                        or proof.get("identity_known") is not True
                        or isinstance(target_ordinal, bool)
                        or not isinstance(target_ordinal, int) or target_ordinal < 1):
                    raise ReviewRefused(f"{session}: heart trace lacks a valid target-scoped prefix proof")
            if mode == _HYBRID_CAPTURE_MODE:
                checks = heart.get("review_checkpoints")
                if not isinstance(checks, dict):
                    raise ReviewRefused(f"{session}: hybrid heart has no before/after review checkpoints")
                require_hybrid_decision(checks.get("before"), action="automated_photo_heart", item=item)
                require_hybrid_decision(checks.get("after"), action="review_heart_result", item=item)
        passes = [a for a in actions if isinstance(a, dict) and a.get("action") == terminal_action]
        summary_ok = (has_exact_send_summary if send_like_accepted else has_exact_pass_summary)
        if len(passes) != 1 or not summary_ok(passes[0], ordinal=ordinal):
            raise ReviewRefused(
                f"{session}: missing exact automated "
                f"{'Send-Priority-Like' if send_like_accepted else 'Pass-without-send'} trace")
        # Whatever the accepted terminal action is, the session must not ALSO carry the other
        # one: a mixed ledger would let a single profile claim both that it never sent and that
        # its send was accepted.
        other_action = ("automated_pass" if send_like_accepted else "automated_send_priority_like")
        if any(isinstance(a, dict) and a.get("action") == other_action for a in actions):
            raise ReviewRefused(
                f"{session}: profile mixes Pass and Send terminal actions in one ledger")
        if passes[0].get("post_frame_sha256") != frame_by_role_item.get((profile["ordinal"], "profile_advance_clear", None)):
            raise ReviewRefused(f"{session}: terminal advance trace does not bind its clear frame")
        if mode == _HYBRID_CAPTURE_MODE:
            checks = passes[0].get("review_checkpoints")
            if not isinstance(checks, dict):
                raise ReviewRefused(f"{session}: hybrid terminal advance has no review checkpoint")
            require_hybrid_decision(checks.get("before"), action=terminal_action, item=None)
    if set(frame_roles_by_ordinal) != profile_ordinals:
        raise ReviewRefused(f"{session}: saved frames do not exactly bind completed profile ordinals")
    return {
        "session": str(session.resolve()),
        "manifest_sha256": _sha256(raw_manifest),
        "device": {key: device[key] for key in _DEVICE_KEYS},
        "frame_size_px": frame_size,
        "config_sha256": config_sha256,
        "profile_count": len(profiles),
        "frame_count": len(frames),
        "skipped_attempt_count": len(skipped_attempts),
        "human_ground_truth": False,
        "capture_mode": mode,
        # Surfaced so a downstream reader never has to infer whether real likes were delivered
        # while this evidence was gathered. False is the ordinary Pass-without-send capture.
        "send_like_accepted": send_like_accepted,
        "terminal_advance_action": terminal_action,
        "target_strategy_id": strategy_id,
        "capture_evidence_scope": (_TARGET_SCOPED_PREFIX_PROOF_ID if target_scoped else "closed_set_profile_v1"),
        "split": manifest.get("split"),
        "composer_items": [item for profile in profiles for item in profile["composer_items"]],
    }


def build_review(sessions: list[Path], config: Path) -> dict:
    """Build a stable review body; exposed for hermetic tests, not used by calibration code."""
    if not sessions:
        raise ReviewRefused("at least one capture session is required")
    try:
        config_bytes = config.read_bytes()
    except OSError as exc:
        raise ReviewRefused(f"cannot read config {config}: {exc}") from exc
    config_sha256 = _sha256(config_bytes)
    records = [_read_capture(session.resolve(), config_sha256) for session in sessions]
    if len({record["session"] for record in records}) != len(records):
        raise ReviewRefused("the same capture session was supplied more than once")
    bindings = {(tuple(record["device"].items()), tuple(record["frame_size_px"])) for record in records}
    if len(bindings) != 1:
        raise ReviewRefused("captures disagree on exact device/build/frame binding")
    for split in {record["split"] for record in records}:
        split_records = [record for record in records if record["split"] == split]
        profile_count = sum(record["profile_count"] for record in split_records)
        covered = {item for record in split_records for item in record["composer_items"]}
        # `_read_capture` already resolved and verified each record's own `target_strategy_id`;
        # reuse that value here rather than re-deriving it, and require every session in one
        # split to agree on exactly one strategy so "the depths it targets" stays a single answer.
        split_strategy_ids = {record["target_strategy_id"] for record in split_records}
        if len(split_strategy_ids) > 1:
            raise ReviewRefused(
                f"{split!r} split mixes automated target strategies {sorted(split_strategy_ids)}; "
                "one split must use exactly one target strategy across every session")
        split_strategy_id, = split_strategy_ids
        # The depth requirement exists so a calibration's evidence spans the depths its strategy
        # claims to exercise; deriving it from the strategy (rather than a fixed {1, 3}) keeps
        # that meaning under a narrower strategy instead of demanding evidence the strategy never
        # collects. `photo_1_only_v1` was chosen 2026-08-22 after item-3 targets refused three
        # times for three different legitimate reasons on real decks.
        required_depths = _automated_target_depths_for_strategy(split_strategy_id)
        if profile_count >= 2 and not required_depths.issubset(covered):
            missing = sorted(required_depths - covered)
            raise ReviewRefused(
                f"{split!r} split has {profile_count} profiles but lacks composer evidence for "
                f"required photo target depth(s) {missing} under target strategy "
                f"{split_strategy_id!r}")
    body = {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "kind": "hinge_unattended_calibration_independent_review",
        "reviewer": {
            "tool_version": _TOOL_VERSION,
            "implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "deterministic_second_process": True,
            "imports_capture_or_vision_stack": False,
        },
        "human_ground_truth": False,
        "not_independent_ground_truth": True,
        "config": {"path": str(config.resolve()), "sha256": config_sha256},
        "captures": records,
    }
    return body


def _cmd_review(args: argparse.Namespace) -> None:
    try:
        body = build_review([Path(raw) for raw in args.sessions], Path(args.config))
        destination = Path(args.out)
        if destination.exists():
            raise ReviewRefused(f"refusing to overwrite existing review artifact {destination}")
        if not destination.parent.exists():
            ensure_private_dir(destination.parent)
        artifact = {**body, "evidence_sha256": _canonical_digest(body)}
        atomic_write_private_text(
            destination, json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            parent=destination.parent,
        )
    except ReviewRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote deterministic self-hashed unattended review to {destination}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.hinge_calibration_review",
        description="Offline, standard-library-only independent review of unattended Hinge calibration captures.")
    parser.add_argument("sessions", nargs="+", help="completed unattended capture directories")
    parser.add_argument("--config", default="config.yaml", help="the exact config bytes used for capture")
    parser.add_argument("--out", required=True, help="new local JSON review artifact (never overwritten)")
    args = parser.parse_args(argv)
    _cmd_review(args)


if __name__ == "__main__":
    main()
