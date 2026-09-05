"""Plan and append a narrowly bound, auditable correction without deleting training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from operation_love import config as cfg_mod
from operation_love.private_files import atomic_write_private_text, ensure_private_dir
from operation_love.ranker import make_store
from operation_love.ranker.retractions import (RetractionRefused, make_plan, profile_id_for_label_ordinal,
                                                is_exact_existing_retraction, retraction_row, verify_plan)

CONFIRM_PREFIX = "I_CONFIRM_APPEND_ONLY_LABEL_RETRACTION:"


def plan(*, store, run_id: str, app: str, source: str, profile_id: str, reason: str,
         evidence_ref: str, evidence_metadata: dict | None = None) -> dict:
    return make_plan(rows=store.retraction_run_rows(run_id, app, source), run_id=run_id, app=app,
                     source=source, profile_id=profile_id, reason=reason, evidence_ref=evidence_ref,
                     evidence_metadata=evidence_metadata)


def _debug_target(actions: Path, row_number: int, *, run_id: str, app: str) -> tuple[int, list[str], str, dict]:
    """Bind a correction target to a literal Worker log row, never an operator-supplied id."""
    repo = Path.cwd().resolve()
    # The correction plan's evidence must remain a stable, local release artifact.  In
    # particular, a symlink could make an apparently in-repo `data/.../actions.jsonl`
    # point at a mutable file elsewhere after the dry-run was reviewed.
    lexical = Path(os.path.abspath(actions))
    resolved = actions.resolve()
    try:
        relative = lexical.relative_to(repo)
        resolved.relative_to(repo)
    except ValueError as exc:
        raise RetractionRefused("debug actions must be a repository-local artifact") from exc
    if lexical != resolved:
        raise RetractionRefused("debug actions must not be a symlinked artifact")
    if actions.parent.name != run_id:
        raise RetractionRefused("debug actions parent directory must equal --run-id")
    raw = lexical.read_bytes()
    lines = [line for line in raw.splitlines() if line.strip()]
    if row_number < 1 or row_number > len(lines):
        raise RetractionRefused("debug row is outside actions.jsonl")
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise RetractionRefused("debug actions file is malformed") from exc
    binding = rows[0] if rows else None
    if (not isinstance(binding, dict) or set(binding) != {"ts", "action", "run_id", "app"}
            or binding.get("action") != "observe_release_run_binding"
            or binding.get("run_id") != run_id or binding.get("app") != app
            or not isinstance(binding.get("ts"), str) or not binding["ts"].strip()):
        raise RetractionRefused("debug actions lacks the required first Worker run/app binding")
    row = rows[row_number - 1]
    if not isinstance(row, dict) or row.get("action") != "observe_decision" or row.get("decision") != "pass":
        raise RetractionRefused("debug row is not the false OBSERVE pass decision")
    outcomes = []
    target_ordinal = None
    for number, value in enumerate(rows, start=1):
        # REFUSE, never skip. The sibling forensic tool only has to prove no Like exists and can
        # step over a row it cannot parse; this one binds a correction to a DECISION ORDINAL
        # counted across these rows, so a row it does not understand means the ordinal was
        # derived from evidence it did not fully read.
        if not isinstance(value, dict):
            raise RetractionRefused("debug actions file is malformed")
        if value.get("action") != "observe_decision":
            continue
        outcome = value.get("decision")
        if outcome not in {"pass", "like"}:
            raise RetractionRefused("debug observe decision has an unsupported outcome")
        if number == row_number:
            target_ordinal = len(outcomes)
        outcomes.append(outcome)
    ordinal = target_ordinal if target_ordinal is not None else -1
    if ordinal < 0:
        raise RetractionRefused("debug row has no decision ordinal")
    digest = hashlib.sha256(raw).hexdigest()
    row_digest = hashlib.sha256(lines[row_number - 1]).hexdigest()
    ref = f"{relative.as_posix()}#row={row_number};actions_sha256={digest};row_sha256={row_digest}"
    return ordinal, outcomes, ref, {"debug_actions_sha256": digest, "debug_row": row_number,
                                     "debug_row_sha256": row_digest,
                                     "debug_decision_outcomes": outcomes}


def _require_exact_debug_store_sequence(rows: dict, debug_outcomes: list[str]) -> None:
    """Prove that debug ordinal means the same thing as persisted ordinal.

    Decisions predate the correction schema's profile id, so ordinal binding is safe only
    when the full visible OBSERVE decision history is exactly the full persisted label and
    decision history for this run/source.  A dropped earlier label, extra decision, or
    inverted outcome otherwise shifts the target to an unrelated profile.
    """
    try:
        label_outcomes = ["like" if bool(value["liked"]) else "pass" for value in rows["labels"]]
        decision_outcomes = ["like" if value["decision"] == "like" else "pass"
                             if value["decision"] == "dislike" else None for value in rows["decisions"]]
    except (KeyError, TypeError) as exc:
        raise RetractionRefused("persisted sequence is malformed") from exc
    if None in decision_outcomes:
        raise RetractionRefused("persisted sequence has an unsupported decision outcome")
    if debug_outcomes != label_outcomes or debug_outcomes != decision_outcomes:
        raise RetractionRefused("debug and persisted decision sequences differ; refusing ordinal binding")


def apply(*, store, document: dict, confirmation: str) -> bool:
    plan_doc = verify_plan(document)
    if confirmation != CONFIRM_PREFIX + plan_doc["plan_sha256"]:
        raise RetractionRefused("exact confirmation must bind this plan hash")
    rows = store.retraction_run_rows(plan_doc["run_id"], plan_doc["app"], plan_doc["source"])
    if is_exact_existing_retraction(rows, plan_doc):
        return False
    current = make_plan(rows=rows, run_id=plan_doc["run_id"], app=plan_doc["app"],
                        source=plan_doc["source"], profile_id=plan_doc["profile_id"],
                        reason=plan_doc["reason"], evidence_ref=plan_doc["evidence_ref"],
                        evidence_metadata=plan_doc.get("evidence_metadata"))
    if current != plan_doc:
        raise RetractionRefused("run rows changed since planning; create and review a new plan")
    return store.append_label_retraction(retraction_row(plan_doc))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.label_retraction")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("dry-run", help="read exact rows and write a self-hashed plan; no mutation")
    create.add_argument("--config", default="config.yaml")
    create.add_argument("--run-id", required=True)
    create.add_argument("--app", required=True)
    create.add_argument("--source", required=True)
    create.add_argument("--reason", required=True)
    create.add_argument("--debug-actions", required=True)
    create.add_argument("--debug-row", required=True, type=int)
    create.add_argument("--out", required=True)
    append = sub.add_parser("apply", help="append exactly one reviewed plan")
    append.add_argument("--config", default="config.yaml")
    append.add_argument("--plan", required=True)
    append.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            output = Path(args.out)
            if output.exists():
                raise RetractionRefused("plan output already exists; refusing to overwrite it")
            store = make_store(cfg_mod.load(args.config), ensure=False)
            try:
                debug_actions = Path(args.debug_actions)
                ordinal, outcomes, evidence_ref, metadata = _debug_target(
                    debug_actions, args.debug_row, run_id=args.run_id, app=args.app)
                rows = store.retraction_run_rows(args.run_id, args.app, args.source)
                _require_exact_debug_store_sequence(rows, outcomes)
                profile_id = profile_id_for_label_ordinal(rows, ordinal)
                document = make_plan(rows=rows, run_id=args.run_id, app=args.app, source=args.source,
                                     profile_id=profile_id, reason=args.reason, evidence_ref=evidence_ref,
                                     evidence_metadata=metadata)
            finally:
                store.close()
            if not output.parent.exists():
                ensure_private_dir(output.parent)
            atomic_write_private_text(
                output, json.dumps(document, indent=2, sort_keys=True) + "\n",
                parent=output.parent,
            )
            print(json.dumps({"plan": str(output), "plan_sha256": document["plan_sha256"],
                              "confirmation": CONFIRM_PREFIX + document["plan_sha256"]}))
        else:
            document = json.loads(Path(args.plan).read_text())
            # Apply is the only path that may initialize the new append-only BQ table.
            store = make_store(cfg_mod.load(args.config), ensure=True)
            try:
                inserted = apply(store=store, document=document, confirmation=args.confirmation)
            finally:
                store.close()
            print("appended" if inserted else "already appended (idempotent no-op)")
    except (OSError, ValueError, json.JSONDecodeError, RetractionRefused) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
