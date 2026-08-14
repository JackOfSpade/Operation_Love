"""Offline verifier for an explicitly accepted, AI-driven Hinge OBSERVE release.

This is deliberately separate from :mod:`tools.hinge_observe_release`: the latter remains the
supervised-manual gate.  This module consumes a completed production Worker debug run, a
non-manual action-provenance sidecar made by the automation controller, and a review emitted by
a distinct offline reviewer process.  It never constructs ADB, a driver, Worker, or provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from operation_love import config as cfg_mod
from operation_love.ranker import make_store

ACCEPTANCE = "I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK"
SOURCES = frozenset({"external_ai_review", "automation"})
PROVENANCE_KEYS = {
    "schema_version", "kind", "completed", "human_ground_truth", "source",
    "production_run_reference", "production_run_id", "debug_actions_sha256", "device",
    "hinge_version_name", "frame_size_px", "executor", "events",
}
REVIEW_KEYS = {
    "schema_version", "kind", "completed", "human_ground_truth", "source", "subject_sha256",
    "production_run_reference", "production_run_id", "debug_actions_sha256", "reviewer",
    "reviewed_event_hashes", "reviewed_at",
}
ARTIFACT_KEYS = {
    "schema_version", "kind", "completed", "human_ground_truth", "source", "acceptance",
    "calibration_calibrated_at", "calibration_sha256", "device", "hinge_version_name",
    "frame_size_px", "production_run_reference", "production_run_id", "debug_actions_sha256",
    "store_persistence_evidence_sha256", "provider_store_evidence_sha256",
    "automation_provenance_sha256", "independent_review_sha256", "verified_at",
}
REQUIRED_EVENTS = (
    ("pass", "observe_decision"),
    ("hub_pre_tap_published", "observe_release_hub_pre_tap_published"),
    ("like_anchor", "observe_like_anchor"),
    ("post_tap_item_verified", "observe_release_post_tap_item_verified"),
    # These records are emitted only by the Worker-owned reviewed-action path.  Do
    # not replace them with a passive ``observe_waiting(like_sending)`` transition:
    # Hinge can move directly to the next card after Send, and inventing a transient
    # state would make a release artifact less truthful, not more conservative.
    ("reviewed_open", "observe_reviewed_open"),
    ("reviewed_like_attempt", "observe_reviewed_like_attempt"),
    ("like", "observe_decision"),
    ("reviewed_like", "observe_reviewed_like"),
)


class AIReleaseEvidenceRefused(RuntimeError):
    """The non-manual evidence did not prove a complete, independently reviewed run."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def _read_json(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AIReleaseEvidenceRefused(f"could not read {label}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise AIReleaseEvidenceRefused(f"{label} must be a JSON object")
    return value, raw


def _inside_repo(path: Path, *, label: str) -> str:
    root = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as exc:
        raise AIReleaseEvidenceRefused(f"{label} must be inside the repository") from exc


def _identity(value, *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"model", "id", "version", "process"}:
        raise AIReleaseEvidenceRefused(f"{label} must carry exact model/id/version/process metadata")
    if any(not isinstance(value[k], str) or not value[k].strip() for k in value):
        raise AIReleaseEvidenceRefused(f"{label} metadata values must be nonempty text")
    return value


def _debug_rows(run_dir: Path, run_id: str) -> tuple[list[dict], bytes]:
    if run_dir.name != run_id:
        raise AIReleaseEvidenceRefused("debug-run directory name must exactly equal production run id")
    actions = run_dir / "actions.jsonl"
    try:
        raw = actions.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise AIReleaseEvidenceRefused(f"could not read production debug actions: {type(exc).__name__}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise AIReleaseEvidenceRefused("production debug actions are empty or malformed")
    binding = rows[0]
    if (set(binding) != {"ts", "action", "run_id", "app"}
            or binding.get("action") != "observe_release_run_binding"
            or binding.get("run_id") != run_id or binding.get("app") != "hinge"):
        raise AIReleaseEvidenceRefused("debug run lacks the required first production Worker binding")
    for row in rows:
        for key in ("before", "after", "anchor"):
            name = row.get(key)
            if name is not None and (not isinstance(name, str) or not (run_dir / name).is_file()):
                raise AIReleaseEvidenceRefused(f"debug action references missing frame {name!r}")
    return rows, raw


def _has_runtime_frame(row: dict) -> bool:
    return any(isinstance(row.get(key), str) for key in ("before", "after", "anchor"))


def _event_row_matches(name: str, row: dict) -> bool:
    """Return whether one row proves its named bridge-release event in isolation."""
    required = dict(REQUIRED_EVENTS)
    if row.get("action") != required[name]:
        return False
    if name == "pass":
        return row.get("decision") == "pass" and row.get("reviewed") is True
    if name == "like":
        return row.get("decision") == "like" and row.get("reviewed") is True
    if name in {"reviewed_open", "reviewed_like_attempt", "reviewed_like"}:
        return (row.get("verified") is True and type(row.get("model_item_index")) is int
                and row["model_item_index"] > 0)
    return True


def _verify_bridge_event_chain(rows: list[dict], indexes: dict[str, int]) -> None:
    """Bind the reviewed open/send/landing records to the one advertised item.

    The action bridge exposes no arbitrary tap or text endpoint.  These additional
    bindings ensure the release verifier does not accidentally accept a generic
    manual OBSERVE trace that merely has similarly named facts mixed into it.
    """
    open_row = rows[indexes["reviewed_open"]]
    attempt_row = rows[indexes["reviewed_like_attempt"]]
    like_row = rows[indexes["like"]]
    landed_row = rows[indexes["reviewed_like"]]
    item = open_row["model_item_index"]
    if any(row.get("model_item_index") != item for row in (attempt_row, like_row, landed_row)):
        raise AIReleaseEvidenceRefused("reviewed bridge events do not bind one exact target item")
    opener_chars = attempt_row.get("opener_chars")
    if type(opener_chars) is not int or opener_chars <= 0:
        raise AIReleaseEvidenceRefused("reviewed like attempt does not prove a nonempty sent opener")
    if any(row.get("opener_chars") != opener_chars for row in (like_row, landed_row)):
        raise AIReleaseEvidenceRefused("reviewed send and landing records do not bind one opener attempt")
    # ``observe_reviewed_like`` is appended only after _verify_like_landed returns;
    # retaining its post-action frame proves the Worker observed the accepted landing
    # rather than just tapping Send.  The attempt frame proves the physical Send path.
    if not _has_runtime_frame(attempt_row):
        raise AIReleaseEvidenceRefused("reviewed like attempt has no retained runtime frame")
    if not _has_runtime_frame(landed_row):
        raise AIReleaseEvidenceRefused("reviewed like landing has no retained runtime frame")


def _verify_provenance(provenance: dict, raw: bytes, *, run_dir: Path, run_id: str,
                       rows: list[dict], actions_raw: bytes) -> dict[str, str]:
    if set(provenance) != PROVENANCE_KEYS:
        raise AIReleaseEvidenceRefused("AI action provenance has an invalid exact schema")
    if provenance.get("schema_version") != 1 or provenance.get("kind") != "hinge_ai_observe_action_provenance":
        raise AIReleaseEvidenceRefused("AI action provenance has unsupported schema/kind")
    if provenance.get("completed") is not True or provenance.get("human_ground_truth") is not False:
        raise AIReleaseEvidenceRefused("AI action provenance must be completed with human_ground_truth=false")
    source = provenance.get("source")
    if source not in SOURCES:
        raise AIReleaseEvidenceRefused("AI action provenance source must be external_ai_review or automation")
    if provenance.get("production_run_reference") != _inside_repo(run_dir, label="debug-run"):
        raise AIReleaseEvidenceRefused("AI action provenance does not bind the exact debug run")
    if provenance.get("production_run_id") != run_id or provenance.get("debug_actions_sha256") != _sha(actions_raw):
        raise AIReleaseEvidenceRefused("AI action provenance does not bind the exact run/actions hash")
    _identity(provenance.get("executor"), label="AI action executor")
    events = provenance.get("events")
    if not isinstance(events, list) or len(events) != len(REQUIRED_EVENTS):
        raise AIReleaseEvidenceRefused("AI action provenance must contain each required event exactly once")
    required = dict(REQUIRED_EVENTS)
    hashes: dict[str, str] = {}
    previous = -1
    for event in events:
        if not isinstance(event, dict) or set(event) != {"name", "index", "row_sha256"}:
            raise AIReleaseEvidenceRefused("AI action provenance event has invalid exact schema")
        name, index = event.get("name"), event.get("index")
        if name not in required or name in hashes or type(index) is not int or not 0 <= index < len(rows):
            raise AIReleaseEvidenceRefused("AI action provenance has duplicate, missing, or invalid event index")
        if index <= previous:
            raise AIReleaseEvidenceRefused("AI action provenance events must preserve the real runtime order")
        row = rows[index]
        if not _event_row_matches(name, row):
            raise AIReleaseEvidenceRefused(f"AI action provenance event {name} binds the wrong debug action")
        # Action outcomes and the visual composer anchor require retained runtime frames; the
        # hub/verification facts intentionally contain no profile text or screenshots.
        if name in {"pass", "like", "like_anchor", "reviewed_open",
                    "reviewed_like_attempt", "reviewed_like"} and not _has_runtime_frame(row):
            raise AIReleaseEvidenceRefused(f"AI action provenance {name} has no retained runtime frame")
        row_sha = _canon(row)
        if event.get("row_sha256") != row_sha:
            raise AIReleaseEvidenceRefused(f"AI action provenance event {name} row hash does not match")
        hashes[name] = row_sha
        previous = index
    if set(hashes) != set(required):
        raise AIReleaseEvidenceRefused("AI action provenance is missing a required runtime event")
    indexes = {event["name"]: event["index"] for event in events}
    _verify_bridge_event_chain(rows, indexes)
    return hashes


def review(*, run_dir: Path, run_id: str, provenance_path: Path, out_dir: Path,
           reviewer: dict) -> dict:
    """Perform a deterministic second-process review of an automation-controller sidecar."""
    reviewer = _identity(reviewer, label="independent reviewer")
    provenance, provenance_raw = _read_json(provenance_path, "AI action provenance")
    rows, actions_raw = _debug_rows(run_dir, run_id)
    hashes = _verify_provenance(provenance, provenance_raw, run_dir=run_dir, run_id=run_id,
                                rows=rows, actions_raw=actions_raw)
    executor = _identity(provenance["executor"], label="AI action executor")
    if (reviewer["id"], reviewer["process"]) == (executor["id"], executor["process"]):
        raise AIReleaseEvidenceRefused("independent reviewer must not share executor identity and process")
    _inside_repo(out_dir, label="independent review output")
    artifact = {
        "schema_version": 1, "kind": "hinge_ai_observe_independent_review", "completed": True,
        "human_ground_truth": False, "source": provenance["source"], "subject_sha256": _sha(provenance_raw),
        "production_run_reference": _inside_repo(run_dir, label="debug-run"), "production_run_id": run_id,
        "debug_actions_sha256": _sha(actions_raw), "reviewer": reviewer,
        "reviewed_event_hashes": hashes, "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if set(artifact) != REVIEW_KEYS:
        raise AssertionError("AI review schema drift")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "hinge_ai_observe_independent_review.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def _verify_review(review_value: dict, review_raw: bytes, provenance_raw: bytes, *, run_dir: Path,
                   run_id: str, actions_raw: bytes, executor: dict, event_hashes: dict[str, str]) -> None:
    if set(review_value) != REVIEW_KEYS or review_value.get("schema_version") != 1:
        raise AIReleaseEvidenceRefused("independent review has an invalid exact schema")
    if (review_value.get("kind") != "hinge_ai_observe_independent_review" or review_value.get("completed") is not True
            or review_value.get("human_ground_truth") is not False):
        raise AIReleaseEvidenceRefused("independent review is incomplete or claims human ground truth")
    if (review_value.get("source") not in SOURCES
            or review_value.get("source") != json.loads(provenance_raw).get("source")
            or review_value.get("subject_sha256") != _sha(provenance_raw)):
        raise AIReleaseEvidenceRefused("independent review does not bind the action provenance")
    if (review_value.get("production_run_reference") != _inside_repo(run_dir, label="debug-run")
            or review_value.get("production_run_id") != run_id
            or review_value.get("debug_actions_sha256") != _sha(actions_raw)):
        raise AIReleaseEvidenceRefused("independent review does not bind the exact production run")
    reviewer = _identity(review_value.get("reviewer"), label="independent reviewer")
    if (reviewer["id"], reviewer["process"]) == (executor["id"], executor["process"]):
        raise AIReleaseEvidenceRefused("independent review reused the executor identity/process")
    if review_value.get("reviewed_event_hashes") != event_hashes:
        raise AIReleaseEvidenceRefused("independent review event hashes do not match verified runtime events")


def _persistence(store, run_id: str, source: str) -> tuple[str, str]:
    try:
        summary = store.ai_observe_release_persistence_summary(run_id, "hinge", source)
    except Exception as exc:  # noqa: BLE001
        raise AIReleaseEvidenceRefused(f"could not query non-manual store release summary: {type(exc).__name__}") from exc
    required = {"ai_pass_labels", "ai_like_labels", "ai_pass_decisions", "ai_like_decisions", "successful_hinge_openers"}
    if not isinstance(summary, dict) or set(summary) != required or any(type(summary[k]) is not int for k in required):
        raise AIReleaseEvidenceRefused("active store returned malformed non-manual release-persistence counts")
    missing = [k for k in ("ai_pass_labels", "ai_like_labels", "ai_pass_decisions", "ai_like_decisions") if summary[k] < 1]
    if missing:
        raise AIReleaseEvidenceRefused("active store lacks a complete non-manual pass+like cycle for this run_id")
    if summary["successful_hinge_openers"] < 1:
        raise AIReleaseEvidenceRefused("active store lacks a successful persisted Hinge opener for this run_id")
    return _canon({k: summary[k] for k in required if k != "successful_hinge_openers"}), _canon({"successful_hinge_openers": summary["successful_hinge_openers"]})


def record_provenance(*, cfg, run_dir: Path, run_id: str, out_dir: Path, source: str,
                      acceptance: str, executor: dict) -> dict:
    """Compile a controller provenance sidecar from immutable runtime facts, not prose.

    The caller supplies its own identity and declared source, but the action ordering, row
    digests, run binding, and device/build/framebuffer binding are all derived here from the
    actual Worker debug trace and current OBSERVE calibration.
    """
    if acceptance != ACCEPTANCE:
        raise AIReleaseEvidenceRefused("exact explicit AI-reviewed OBSERVE release acceptance is required")
    if source not in SOURCES:
        raise AIReleaseEvidenceRefused("AI provenance source must be external_ai_review or automation")
    executor = _identity(executor, label="AI action executor")
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    calibration = app_cfg.get("targeting_calibration")
    if not isinstance(calibration, dict) or app_cfg.get("mode", cfg.mode) != "observe":
        raise AIReleaseEvidenceRefused("provenance requires targeting calibration and config mode observe")
    controller = app_cfg.get("ai_reviewed_observe_controller")
    if (app_cfg.get("observe_evidence_source") != source or not isinstance(controller, dict)
            or controller.get("source") != source or controller.get("acceptance") != acceptance
            or controller.get("executor") != executor):
        raise AIReleaseEvidenceRefused(
            "provenance must exactly match the configured non-manual OBSERVE controller/source")
    rows, actions_raw = _debug_rows(run_dir, run_id)
    events = []
    previous = 0
    for name, action in REQUIRED_EVENTS:
        index = None
        for candidate in range(previous + 1, len(rows)):
            row = rows[candidate]
            if row.get("action") != action or not _event_row_matches(name, row):
                continue
            index = candidate
            break
        if index is None:
            raise AIReleaseEvidenceRefused(f"debug run has no ordered runtime fact for {name}")
        events.append({"name": name, "index": index, "row_sha256": _canon(rows[index])})
        previous = index
    artifact = {
        "schema_version": 1, "kind": "hinge_ai_observe_action_provenance", "completed": True,
        "human_ground_truth": False, "source": source,
        "production_run_reference": _inside_repo(run_dir, label="debug-run"), "production_run_id": run_id,
        "debug_actions_sha256": _sha(actions_raw), "device": calibration["device"],
        "hinge_version_name": calibration["hinge_version_name"], "frame_size_px": calibration["frame_size_px"],
        "executor": executor, "events": events,
    }
    if set(artifact) != PROVENANCE_KEYS:
        raise AssertionError("AI action provenance schema drift")
    # Fail before emitting a sidecar if a selected runtime event lacks the frame/shape proof
    # the later independent reviewer would require anyway.
    _verify_provenance(artifact, json.dumps(artifact, sort_keys=True).encode(), run_dir=run_dir,
                       run_id=run_id, rows=rows, actions_raw=actions_raw)
    _inside_repo(out_dir, label="AI action provenance output")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "hinge_ai_observe_action_provenance.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def verify(*, cfg, run_dir: Path, run_id: str, provenance_path: Path, review_path: Path,
           out_dir: Path, acceptance: str, store=None) -> tuple[dict, dict]:
    if acceptance != ACCEPTANCE:
        raise AIReleaseEvidenceRefused("exact explicit AI-reviewed OBSERVE release acceptance is required")
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    _inside_repo(out_dir, label="release output")
    calibration = app_cfg.get("targeting_calibration")
    if not isinstance(calibration, dict) or app_cfg.get("mode", cfg.mode) != "observe":
        raise AIReleaseEvidenceRefused("verification requires targeting calibration and config mode observe")
    provenance, provenance_raw = _read_json(provenance_path, "AI action provenance")
    review_value, review_raw = _read_json(review_path, "independent review")
    rows, actions_raw = _debug_rows(run_dir, run_id)
    event_hashes = _verify_provenance(provenance, provenance_raw, run_dir=run_dir, run_id=run_id,
                                      rows=rows, actions_raw=actions_raw)
    for key in ("device", "hinge_version_name", "frame_size_px"):
        if provenance.get(key) != calibration.get(key):
            raise AIReleaseEvidenceRefused(
                f"AI action provenance {key} does not bind the calibrated device/build/framebuffer")
    executor = _identity(provenance["executor"], label="AI action executor")
    _verify_review(review_value, review_raw, provenance_raw, run_dir=run_dir, run_id=run_id,
                   actions_raw=actions_raw, executor=executor, event_hashes=event_hashes)
    owned_store = store is None
    store = make_store(cfg, ensure=False) if store is None else store
    try:
        persistence_sha, provider_sha = _persistence(store, run_id, provenance["source"])
    finally:
        if owned_store:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
    run_ref = _inside_repo(run_dir, label="debug-run")
    verified_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "schema_version": 1, "kind": "hinge_ai_reviewed_production_observe_release", "completed": True,
        "human_ground_truth": False, "source": provenance["source"], "acceptance": acceptance,
        "calibration_calibrated_at": calibration["calibrated_at"], "calibration_sha256": _canon(calibration),
        "device": calibration["device"], "hinge_version_name": calibration["hinge_version_name"],
        "frame_size_px": calibration["frame_size_px"], "production_run_reference": run_ref,
        "production_run_id": run_id, "debug_actions_sha256": _sha(actions_raw),
        "store_persistence_evidence_sha256": persistence_sha, "provider_store_evidence_sha256": provider_sha,
        "automation_provenance_sha256": _sha(provenance_raw), "independent_review_sha256": _sha(review_raw),
        "verified_at": verified_at,
    }
    if set(artifact) != ARTIFACT_KEYS:
        raise AssertionError("AI release artifact schema drift")
    out_dir.mkdir(parents=True, exist_ok=False)
    output = out_dir / "hinge_ai_observe_release.json"
    output_raw = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
    output.write_bytes(output_raw)
    paste = {key: artifact[key] for key in (
        "schema_version", "acceptance", "calibration_calibrated_at", "calibration_sha256", "device",
        "hinge_version_name", "frame_size_px", "production_run_reference", "production_run_id", "verified_at")}
    paste["verification_file"] = _inside_repo(output, label="output")
    paste["verification_sha256"] = _sha(output_raw)
    return artifact, paste


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.hinge_observe_ai_release")
    sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug-run", required=True)
    common.add_argument("--run-id", required=True)
    review_cmd = sub.add_parser("review", parents=[common])
    review_cmd.add_argument("--provenance", required=True)
    review_cmd.add_argument("--out", required=True)
    review_cmd.add_argument("--reviewer-model", required=True)
    review_cmd.add_argument("--reviewer-id", required=True)
    review_cmd.add_argument("--reviewer-version", required=True)
    review_cmd.add_argument("--reviewer-process", required=True)
    provenance_cmd = sub.add_parser("provenance", parents=[common])
    provenance_cmd.add_argument("--config", default="config.yaml")
    provenance_cmd.add_argument("--out", required=True)
    provenance_cmd.add_argument("--source", required=True, choices=sorted(SOURCES))
    provenance_cmd.add_argument("--acceptance", required=True)
    provenance_cmd.add_argument("--executor-model", required=True)
    provenance_cmd.add_argument("--executor-id", required=True)
    provenance_cmd.add_argument("--executor-version", required=True)
    provenance_cmd.add_argument("--executor-process", required=True)
    verify_cmd = sub.add_parser("verify", parents=[common])
    verify_cmd.add_argument("--provenance", required=True)
    verify_cmd.add_argument("--config", default="config.yaml")
    verify_cmd.add_argument("--review", required=True)
    verify_cmd.add_argument("--out", required=True)
    verify_cmd.add_argument("--acceptance", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "provenance":
            record_provenance(cfg=cfg_mod.load(args.config), run_dir=Path(args.debug_run), run_id=args.run_id,
                              out_dir=Path(args.out), source=args.source, acceptance=args.acceptance,
                              executor={"model": args.executor_model, "id": args.executor_id,
                              "version": args.executor_version, "process": args.executor_process})
        elif args.command == "review":
            review(run_dir=Path(args.debug_run), run_id=args.run_id, provenance_path=Path(args.provenance),
                   out_dir=Path(args.out), reviewer={"model": args.reviewer_model, "id": args.reviewer_id,
                   "version": args.reviewer_version, "process": args.reviewer_process})
        else:
            _artifact, paste = verify(cfg=cfg_mod.load(args.config), run_dir=Path(args.debug_run), run_id=args.run_id,
                                      provenance_path=Path(args.provenance), review_path=Path(args.review),
                                      out_dir=Path(args.out), acceptance=args.acceptance)
            print(yaml.safe_dump({"apps": {"hinge": {"ai_reviewed_observe_release_evidence": paste}}}, sort_keys=False))
    except AIReleaseEvidenceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
