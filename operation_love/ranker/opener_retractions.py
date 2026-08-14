"""Fail-closed, append-only cleanup plans for incorrectly persisted advisory openers."""
from __future__ import annotations

from datetime import datetime, timezone

from .retractions import RetractionRefused, canonical_sha


def make_plan(*, rows: dict, run_id: str, app: str, reason: str,
              evidence_ref: str, evidence_metadata: dict, allow_existing: bool = False) -> dict:
    """Bind every opener in a no-decision run, without putting opener text in the plan."""
    if not all(isinstance(value, str) and value.strip()
               for value in (run_id, app, reason, evidence_ref)):
        raise RetractionRefused("run-id/app/reason/evidence-ref must be nonempty text")
    if rows.get("run_id") != run_id or rows.get("app") != app:
        raise RetractionRefused("store snapshot is not bound to the requested run/app")
    counts = rows.get("preference_counts")
    if not isinstance(counts, dict):
        raise RetractionRefused("preference count snapshot is malformed")
    # Historical eager generation can coexist with real Pass records.  Those rows are useful
    # preference evidence and must survive.  The disqualifier is a *Like* (whether labelled or
    # merely a landed decision), not the presence of a card, photo, or pass.
    effective = rows.get("effective_counts")
    if (not isinstance(effective, dict)
            or int(effective.get("like_labels", -1)) != 0
            or int(effective.get("like_decisions", -1)) != 0):
        raise RetractionRefused("effective Like evidence exists or could not be proven absent")
    openers = list(rows.get("openers", []))
    if not openers:
        raise RetractionRefused("no persisted opener rows match this run/app")
    if rows.get("retractions") and not allow_existing:
        raise RetractionRefused("run/app already has advisory opener cleanup tombstones")
    required = {"created_at", "model", "opener_fingerprint"}
    if any(not isinstance(item, dict) or set(item) != required for item in openers):
        raise RetractionRefused("opener snapshot is malformed or exposes unsupported fields")
    timestamps = [item["created_at"] for item in openers]
    if any(not isinstance(value, str) or not value for value in timestamps) or len(set(timestamps)) != len(timestamps):
        raise RetractionRefused("opener rows do not have unique auditable created_at values")
    if any(not isinstance(item["model"], str) or not item["model"] or
           not isinstance(item["opener_fingerprint"], str) or len(item["opener_fingerprint"]) != 64
           for item in openers):
        raise RetractionRefused("opener snapshot has invalid model or fingerprint")
    body = {
        "schema_version": 1,
        "kind": "operation_love_advisory_opener_cleanup_plan",
        "run_id": run_id,
        "app": app,
        "reason": reason,
        "evidence_ref": evidence_ref,
        "evidence_metadata": evidence_metadata,
        "preference_counts": {key: int(counts.get(key, 0)) for key in
                              ("profiles", "profile_photos", "labels", "decisions")},
        "effective_counts": {key: int(effective[key]) for key in
                             ("like_labels", "like_decisions", "pass_labels", "pass_decisions")},
        "openers": sorted(openers, key=lambda item: item["created_at"]),
        "snapshot_sha256": canonical_sha(rows),
    }
    body["correction_id"] = canonical_sha({key: body[key] for key in (
        "run_id", "app", "reason", "evidence_ref", "snapshot_sha256")})
    body["plan_sha256"] = canonical_sha(body)
    return body


def verify_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("kind") != "operation_love_advisory_opener_cleanup_plan":
        raise RetractionRefused("invalid advisory opener cleanup plan")
    actual = canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("plan_sha256") != actual:
        raise RetractionRefused("cleanup plan self-hash does not verify")
    return plan


def tombstone_rows(plan: dict) -> list[dict]:
    verify_plan(plan)
    now = datetime.now(timezone.utc).isoformat()
    return [{
        "correction_id": plan["correction_id"], "run_id": plan["run_id"], "app": plan["app"],
        "opener_created_at": opener["created_at"], "model": opener["model"],
        "opener_fingerprint": opener["opener_fingerprint"], "reason": plan["reason"],
        "evidence_ref": plan["evidence_ref"], "created_at": now,
    } for opener in plan["openers"]]
