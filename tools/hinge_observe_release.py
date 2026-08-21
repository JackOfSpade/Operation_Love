"""Offline verifier for the production Hinge OBSERVE release gate.

This tool never constructs ADB, a driver, Worker, or opener client.  It verifies local evidence
from an already-completed, supervised production OBSERVE run and emits the exact
``apps.hinge.observe_release_evidence`` mapping that config validation requires before Hinge AUTO
may start. It takes no hand-authored status/API claim: control facts are derived from production
debug actions, while a persisted manual label/decision/provider record must already exist in the
active store. BigQuery is queried through the store's read-only aggregate API, so no personal row
data is exported.
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
from operation_love.private_files import atomic_write_private_bytes, ensure_private_dir
from operation_love.ranker import make_store

_ARTIFACT_KEYS = {
    "schema_version", "kind", "completed", "calibration_calibrated_at",
    "calibration_sha256", "device", "hinge_version_name", "frame_size_px",
    "production_run_reference", "production_run_id", "debug_actions_sha256", "store_persistence_evidence_sha256",
    "provider_store_evidence_sha256", "observe_control_evidence_sha256", "verified_at",
}


class ReleaseEvidenceRefused(RuntimeError):
    """The supplied local evidence cannot safely unlock Hinge AUTO."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8"))


def _verify_debug_run(run_dir: Path, run_id: str) -> str:
    actions = run_dir / "actions.jsonl"
    try:
        raw = actions.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceRefused(f"could not read production debug actions {actions}: {type(exc).__name__}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ReleaseEvidenceRefused("production debug actions are empty or malformed")
    if run_dir.name != run_id:
        raise ReleaseEvidenceRefused("debug-run directory name must exactly equal --run-id")
    binding = rows[0]
    if (set(binding) != {"ts", "action", "run_id", "app"}
            or binding.get("action") != "observe_release_run_binding"
            or binding.get("run_id") != run_id or binding.get("app") != "hinge"
            or not isinstance(binding.get("ts"), str) or not binding["ts"].strip()):
        raise ReleaseEvidenceRefused(
            "debug run lacks the required first production Worker run binding; it may be a "
            "legacy timestamp-named or unrelated actions.jsonl")
    decisions = {row.get("decision") for row in rows if row.get("action") == "observe_decision"}
    actions_seen = {row.get("action") for row in rows}
    waiting = {row.get("reason") for row in rows if row.get("action") == "observe_waiting"}
    if not {"pass", "like"} <= decisions:
        raise ReleaseEvidenceRefused("debug run lacks both observed manual pass and like decisions")
    # A refusal/paywall fact is retained when production naturally sees one, but no safe release
    # workflow may manufacture a paywall merely to produce evidence. It is therefore informative,
    # not an AUTO prerequisite.
    required_facts = {
        "observe_release_hub_pre_tap_published",
        "observe_release_post_tap_item_verified",
    }
    if ("observe_like_anchor" not in actions_seen or "like_sending" not in waiting
            or not required_facts <= actions_seen):
        raise ReleaseEvidenceRefused(
            "debug run lacks a required production fact (pre-tap hub publish, post-tap item "
            "verification, inline-composer anchor, or like_sending state)")
    for row in rows:
        for key in ("before", "after", "anchor"):
            name = row.get(key)
            if name is not None and (not isinstance(name, str) or not (run_dir / name).is_file()):
                raise ReleaseEvidenceRefused(f"debug action references missing frame {name!r}")
    return _sha256(raw)


def _verify_store_persistence(store, run_id: str) -> tuple[str, str]:
    """Read aggregate persistence proof through the active store, never raw profile data."""
    try:
        summary = store.observe_release_persistence_summary(run_id, "hinge")
    except Exception as exc:  # noqa: BLE001 -- store backend errors must fail the release closed
        raise ReleaseEvidenceRefused(
            f"could not query active store's read-only release summary: {type(exc).__name__}") from exc
    required = {
        "manual_pass_labels", "manual_like_labels",
        "manual_pass_decisions", "manual_like_decisions",
        "successful_hinge_openers",
    }
    if (not isinstance(summary, dict) or set(summary) != required
            or any(type(summary.get(key)) is not int for key in required)):
        raise ReleaseEvidenceRefused("active store returned malformed release-persistence counts")
    missing = [key for key in (
        "manual_pass_labels", "manual_like_labels",
        "manual_pass_decisions", "manual_like_decisions",
    ) if summary[key] < 1]
    if missing:
        raise ReleaseEvidenceRefused(
            "active store lacks a persisted complete manual Hinge pass+like cycle for this "
            f"run_id ({', '.join(missing)} missing)")
    if summary["successful_hinge_openers"] < 1:
        raise ReleaseEvidenceRefused(
            "active store lacks a successful persisted Hinge opener for this run_id; "
            "a rejection or spend row cannot substitute")
    persistence_summary = {key: summary[key] for key in required if key != "successful_hinge_openers"}
    provider_summary = {"successful_hinge_openers": summary["successful_hinge_openers"]}
    return _canonical_sha256(persistence_summary), _canonical_sha256(provider_summary)


def verify(*, cfg, run_dir: Path, run_id: str, out_dir: Path, store=None) -> tuple[dict, dict]:
    """Verify existing evidence only and return ``(artifact, paste_mapping)``."""
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    calibration = app_cfg.get("targeting_calibration")
    if not isinstance(calibration, dict):
        raise ReleaseEvidenceRefused("apps.hinge.targeting_calibration is required before production OBSERVE release verification")
    if app_cfg.get("mode", cfg.mode) != "observe":
        raise ReleaseEvidenceRefused("release verification requires config mode observe, never auto")
    if not run_id.strip():
        raise ReleaseEvidenceRefused("--run-id must be the production Worker run id")
    debug_sha = _verify_debug_run(run_dir, run_id)
    owned_store = store is None
    store = make_store(cfg, ensure=False) if store is None else store
    try:
        persistence_sha, provider_sha = _verify_store_persistence(store, run_id)
    finally:
        if owned_store:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- verification has already made the safe decision
                pass
    root = Path.cwd().resolve()
    try:
        run_ref = str(run_dir.resolve().relative_to(root))
    except ValueError as exc:
        raise ReleaseEvidenceRefused("--debug-run must be inside the repository") from exc
    verified_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "schema_version": 1,
        "kind": "hinge_production_observe_release",
        "completed": True,
        "calibration_calibrated_at": calibration["calibrated_at"],
        "calibration_sha256": _canonical_sha256(calibration),
        "device": calibration["device"],
        "hinge_version_name": calibration["hinge_version_name"],
        "frame_size_px": calibration["frame_size_px"],
        "production_run_reference": run_ref,
        "production_run_id": run_id,
        "debug_actions_sha256": debug_sha,
        "store_persistence_evidence_sha256": persistence_sha,
        "provider_store_evidence_sha256": provider_sha,
        "observe_control_evidence_sha256": debug_sha,
        "verified_at": verified_at,
    }
    if set(artifact) != _ARTIFACT_KEYS:
        raise AssertionError("release artifact schema drift")
    ensure_private_dir(out_dir, exist_ok=False)
    artifact_path = out_dir / "hinge_observe_release.json"
    artifact_bytes = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write_private_bytes(artifact_path, artifact_bytes, parent=out_dir)
    try:
        artifact_ref = str(artifact_path.resolve().relative_to(root))
    except ValueError as exc:  # defensive: output is restricted by CLI before creation
        raise ReleaseEvidenceRefused("release artifact must be inside the repository") from exc
    paste = {
        "schema_version": 1,
        "calibration_calibrated_at": artifact["calibration_calibrated_at"],
        "calibration_sha256": artifact["calibration_sha256"],
        "device": artifact["device"],
        "hinge_version_name": artifact["hinge_version_name"],
        "frame_size_px": artifact["frame_size_px"],
        "production_run_reference": run_ref,
        "production_run_id": run_id,
        "verification_file": artifact_ref,
        "verification_sha256": _sha256(artifact_bytes),
        "verified_at": verified_at,
    }
    return artifact, paste


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.hinge_observe_release")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug-run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = cfg_mod.load(args.config)
    out = Path(args.out)
    root = Path.cwd().resolve()
    if out.is_absolute():
        raise ReleaseEvidenceRefused("--out must be a repo-relative fresh directory")
    try:
        (root / out).resolve().relative_to(root)
    except ValueError as exc:
        raise ReleaseEvidenceRefused("--out must stay inside the repository") from exc
    try:
        _artifact, paste = verify(cfg=cfg, run_dir=Path(args.debug_run), run_id=args.run_id,
                                  out_dir=out)
    except ReleaseEvidenceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
    print(yaml.safe_dump({"apps": {"hinge": {"observe_release_evidence": paste}}}, sort_keys=False))


if __name__ == "__main__":
    main()
