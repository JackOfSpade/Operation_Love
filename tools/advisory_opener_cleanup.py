"""Dry-run first cleanup for advisory opener rows that predate decision gating."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from operation_love import config as cfg_mod
from operation_love.ranker import make_store
from operation_love.ranker.opener_retractions import make_plan, tombstone_rows, verify_plan
from operation_love.ranker.retractions import RetractionRefused

CONFIRM_PREFIX = "I_CONFIRM_APPEND_ONLY_ADVISORY_OPENER_CLEANUP:"
_LEGACY_712_RUN = "712bead758d5"
_LEGACY_712_ACTIONS = "data/hinge_debug/run_20260810_173631/actions.jsonl"
_LEGACY_712_SHA256 = "af1d71fab2e4e0c182c2333869fa467097e473bf6c8bf36d24c0028ecbefc2f3"
_LEGACY_712_OPENER = {
    "created_at": "2026-08-10T21:40:52.136804+00:00", "model": "gemini-3.6-flash",
    "opener_fingerprint": "04839c778f19c74cd1158afec4fc76a32ae41d34abbe8ffb4bb91c3161d2180b",
}


def _debug_evidence(actions: Path, *, run_id: str, app: str) -> tuple[str, dict]:
    """Bind cleanup to immutable debug evidence and prove it contains no observed Like.

    The 2026-08-10 legacy run predates per-run binding.  It is accepted only through the exact
    path/hash/run/app tuple below; it is not a generic compatibility escape hatch.
    """
    repo = Path.cwd().resolve()
    lexical = Path(os.path.abspath(actions))
    resolved = actions.resolve()
    try:
        relative = lexical.relative_to(repo)
        resolved.relative_to(repo)
    except ValueError as exc:
        raise RetractionRefused("debug actions must be a repository-local artifact") from exc
    if lexical != resolved:
        raise RetractionRefused("debug actions must be a non-symlinked repository-local artifact")
    raw = lexical.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise RetractionRefused("debug actions file is malformed") from exc
    digest = hashlib.sha256(raw).hexdigest()
    relative_text = relative.as_posix()
    legacy = run_id == _LEGACY_712_RUN and app == "hinge"
    if legacy:
        if relative_text != _LEGACY_712_ACTIONS or digest != _LEGACY_712_SHA256:
            raise RetractionRefused("legacy 712 evidence must be the exact approved path and SHA-256")
    else:
        binding = rows[0] if rows else None
        if (lexical.parent.name != run_id or not isinstance(binding, dict)
                or binding.get("action") != "observe_release_run_binding"
                or binding.get("run_id") != run_id or binding.get("app") != app):
            raise RetractionRefused("debug actions lacks required first run/app binding")
    outcomes = []
    for row in rows:
        if not isinstance(row, dict) or row.get("action") != "observe_decision":
            continue
        outcome = str(row.get("decision", "")).strip().lower()
        if outcome not in {"like", "pass", "dislike"}:
            raise RetractionRefused("debug observe decision is malformed")
        outcomes.append(outcome)
    if "like" in outcomes:
        raise RetractionRefused("debug log contains an observed Like; cleanup is refused")
    return (f"{relative_text};actions_sha256={digest}",
            {"debug_actions_sha256": digest, "debug_decision_outcomes": outcomes,
             "debug_action_rows": len(rows), "legacy_712_evidence": legacy})


def dry_run(*, store, run_id: str, app: str, reason: str, debug_actions: Path) -> dict:
    evidence_ref, metadata = _debug_evidence(debug_actions, run_id=run_id, app=app)
    rows = store.advisory_opener_run_rows(run_id, app)
    if run_id == _LEGACY_712_RUN:
        if app != "hinge" or rows.get("openers") != [_LEGACY_712_OPENER]:
            raise RetractionRefused("legacy 712 BigQuery opener snapshot is not the exact approved row")
    return make_plan(rows=rows, run_id=run_id, app=app,
                     reason=reason, evidence_ref=evidence_ref, evidence_metadata=metadata)


def apply(*, store, document: dict, confirmation: str) -> bool:
    plan = verify_plan(document)
    if confirmation != CONFIRM_PREFIX + plan["plan_sha256"]:
        raise RetractionRefused("exact confirmation must bind this plan hash")
    rows = store.advisory_opener_run_rows(plan["run_id"], plan["app"])
    expected = {item["opener_created_at"]: item for item in tombstone_rows(plan)}
    existing = {item.get("opener_created_at"): item for item in rows.get("retractions", [])}
    if len(existing) != len(rows.get("retractions", [])) or any(key not in expected for key in existing):
        raise RetractionRefused("run/app has an unknown or duplicate opener cleanup tombstone")
    exact_keys = ("correction_id", "opener_created_at", "model", "opener_fingerprint", "reason", "evidence_ref")
    for timestamp, item in existing.items():
        if {key: item.get(key) for key in exact_keys} != {key: expected[timestamp][key] for key in exact_keys}:
            raise RetractionRefused("run/app has a competing or mismatched opener cleanup tombstone")
    # A partial failure is safely resumable only if the pre-tombstone snapshot remains exactly
    # the reviewed plan.  Normalize the append-only rows away before re-hashing that snapshot.
    current = make_plan(rows={**rows, "retractions": []},
                        run_id=plan["run_id"], app=plan["app"], reason=plan["reason"],
                        evidence_ref=plan["evidence_ref"], evidence_metadata=plan["evidence_metadata"],
                        allow_existing=True)
    if current != plan:
        raise RetractionRefused("run rows changed since planning; create and review a new plan")
    changed = False
    for timestamp, row in expected.items():
        if timestamp in existing:
            continue
        changed = store.append_opener_retraction(row) or changed
    return changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.advisory_opener_cleanup")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("dry-run", help="write a self-hashed plan; no mutation")
    create.add_argument("--config", default="config.yaml")
    create.add_argument("--run-id", required=True)
    create.add_argument("--app", required=True)
    create.add_argument("--reason", required=True)
    create.add_argument("--debug-actions", required=True)
    create.add_argument("--out", required=True)
    commit = sub.add_parser("apply", help="append reviewed opener tombstones")
    commit.add_argument("--config", default="config.yaml")
    commit.add_argument("--plan", required=True)
    commit.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            output = Path(args.out)
            if output.exists():
                raise RetractionRefused("plan output already exists; refusing to overwrite it")
            store = make_store(cfg_mod.load(args.config), ensure=False)
            try:
                plan = dry_run(store=store, run_id=args.run_id, app=args.app, reason=args.reason,
                               debug_actions=Path(args.debug_actions))
            finally:
                store.close()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"plan": str(output), "plan_sha256": plan["plan_sha256"],
                              "confirmation": CONFIRM_PREFIX + plan["plan_sha256"]}))
        else:
            plan = json.loads(Path(args.plan).read_text())
            store = make_store(cfg_mod.load(args.config), ensure=True)
            try:
                changed = apply(store=store, document=plan, confirmation=args.confirmation)
            finally:
                store.close()
            print("appended" if changed else "already appended (idempotent no-op)")
    except (OSError, ValueError, json.JSONDecodeError, RetractionRefused) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
