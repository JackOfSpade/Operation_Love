"""Fail-closed, append-only cleanup plans for incorrectly persisted advisory openers."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .retractions import RetractionRefused, canonical_sha


_PREFERENCE_COUNT_KEYS = frozenset({"profiles", "profile_photos", "labels", "decisions"})
_EFFECTIVE_COUNT_KEYS = frozenset(
    {"like_labels", "like_decisions", "pass_labels", "pass_decisions"})
_OPENER_KEYS = frozenset({"created_at", "model", "opener_fingerprint"})
_PLAN_KEYS = frozenset({
    "schema_version", "kind", "run_id", "app", "reason", "evidence_ref",
    "evidence_metadata", "preference_counts", "effective_counts", "openers",
    "snapshot_sha256", "correction_id", "plan_sha256",
})
_KIND = "operation_love_advisory_opener_cleanup_plan"


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _require_count_map(value: object, keys: frozenset[str], label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RetractionRefused(f"{label} snapshot must contain exactly {sorted(keys)}")
    if any(type(value[key]) is not int or value[key] < 0 for key in keys):
        raise RetractionRefused(f"{label} snapshot values must be nonnegative integers")
    return {key: value[key] for key in keys}


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        numeric = float(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.tzinfo is not None
    return math.isfinite(numeric)


def _require_openers(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise RetractionRefused("no persisted opener rows match this run/app")
    if any(not isinstance(item, dict) or set(item) != _OPENER_KEYS for item in value):
        raise RetractionRefused("opener snapshot is malformed or exposes unsupported fields")
    timestamps = [item["created_at"] for item in value]
    if any(not _valid_timestamp(timestamp) for timestamp in timestamps):
        raise RetractionRefused("opener rows do not have valid auditable created_at values")
    if len(set(timestamps)) != len(timestamps):
        raise RetractionRefused("opener rows do not have unique auditable created_at values")
    if any(not isinstance(item["model"], str) or not item["model"].strip()
           or not _is_sha256(item["opener_fingerprint"]) for item in value):
        raise RetractionRefused("opener snapshot has invalid model or fingerprint")
    return sorted((dict(item) for item in value), key=lambda item: item["created_at"])


def make_plan(*, rows: dict, run_id: str, app: str, reason: str,
              evidence_ref: str, evidence_metadata: dict, allow_existing: bool = False) -> dict:
    """Bind every opener in a no-decision run, without putting opener text in the plan."""
    if not all(isinstance(value, str) and value.strip()
               for value in (run_id, app, reason, evidence_ref)):
        raise RetractionRefused("run-id/app/reason/evidence-ref must be nonempty text")
    if not isinstance(rows, dict):
        raise RetractionRefused("store snapshot is malformed")
    if not isinstance(evidence_metadata, dict):
        raise RetractionRefused("evidence metadata must be a mapping")
    try:
        canonical_sha(evidence_metadata)
    except (TypeError, ValueError) as exc:
        raise RetractionRefused("evidence metadata is not canonical JSON data") from exc
    if rows.get("run_id") != run_id or rows.get("app") != app:
        raise RetractionRefused("store snapshot is not bound to the requested run/app")
    counts = _require_count_map(
        rows.get("preference_counts"), _PREFERENCE_COUNT_KEYS, "preference count")
    # Historical eager generation can coexist with real Pass records.  Those rows are useful
    # preference evidence and must survive.  The disqualifier is a *Like* (whether labelled or
    # merely a landed decision), not the presence of a card, photo, or pass.
    effective = _require_count_map(
        rows.get("effective_counts"), _EFFECTIVE_COUNT_KEYS, "effective count")
    if effective["like_labels"] != 0 or effective["like_decisions"] != 0:
        raise RetractionRefused("effective Like evidence exists or could not be proven absent")
    openers = _require_openers(rows.get("openers"))
    retractions = rows.get("retractions")
    if not isinstance(retractions, list):
        raise RetractionRefused("opener retraction snapshot is malformed")
    if retractions and not allow_existing:
        raise RetractionRefused("run/app already has advisory opener cleanup tombstones")
    body = {
        "schema_version": 1,
        "kind": _KIND,
        "run_id": run_id,
        "app": app,
        "reason": reason,
        "evidence_ref": evidence_ref,
        "evidence_metadata": evidence_metadata,
        "preference_counts": counts,
        "effective_counts": effective,
        "openers": openers,
    }
    try:
        body["snapshot_sha256"] = canonical_sha(rows)
    except (TypeError, ValueError) as exc:
        raise RetractionRefused("store snapshot is not canonical JSON data") from exc
    body["correction_id"] = canonical_sha({key: body[key] for key in (
        "run_id", "app", "reason", "evidence_ref", "snapshot_sha256")})
    body["plan_sha256"] = canonical_sha(body)
    return body


def verify_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or set(plan) != _PLAN_KEYS:
        raise RetractionRefused("invalid advisory opener cleanup plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise RetractionRefused("unsupported advisory opener cleanup plan schema")
    if plan["kind"] != _KIND:
        raise RetractionRefused("invalid advisory opener cleanup plan")
    if not all(isinstance(plan[key], str) and plan[key].strip()
               for key in ("run_id", "app", "reason", "evidence_ref")):
        raise RetractionRefused("cleanup plan run/app/reason/evidence-ref is malformed")
    if not isinstance(plan["evidence_metadata"], dict):
        raise RetractionRefused("cleanup plan evidence metadata is malformed")
    _require_count_map(plan["preference_counts"], _PREFERENCE_COUNT_KEYS, "preference count")
    effective = _require_count_map(
        plan["effective_counts"], _EFFECTIVE_COUNT_KEYS, "effective count")
    if effective["like_labels"] != 0 or effective["like_decisions"] != 0:
        raise RetractionRefused("cleanup plan does not prove effective Like evidence absent")
    openers = _require_openers(plan["openers"])
    if plan["openers"] != openers:
        raise RetractionRefused("cleanup plan opener rows are not in canonical order")
    for key in ("snapshot_sha256", "correction_id", "plan_sha256"):
        if not _is_sha256(plan[key]):
            raise RetractionRefused(f"cleanup plan {key} is not a SHA-256 digest")
    expected_correction = canonical_sha({key: plan[key] for key in (
        "run_id", "app", "reason", "evidence_ref", "snapshot_sha256")})
    if plan["correction_id"] != expected_correction:
        raise RetractionRefused("cleanup plan correction binding does not verify")
    try:
        actual = canonical_sha({
            key: value for key, value in plan.items() if key != "plan_sha256"})
    except (TypeError, ValueError) as exc:
        raise RetractionRefused("cleanup plan is not canonical JSON data") from exc
    if plan.get("plan_sha256") != actual:
        raise RetractionRefused("cleanup plan self-hash does not verify")
    return plan


def tombstone_rows(plan: dict) -> list[dict]:
    verified = verify_plan(plan)
    now = datetime.now(timezone.utc).isoformat()
    return [{
        "correction_id": verified["correction_id"], "run_id": verified["run_id"],
        "app": verified["app"],
        "opener_created_at": opener["created_at"], "model": opener["model"],
        "opener_fingerprint": opener["opener_fingerprint"], "reason": verified["reason"],
        "evidence_ref": verified["evidence_ref"], "created_at": now,
    } for opener in verified["openers"]]
