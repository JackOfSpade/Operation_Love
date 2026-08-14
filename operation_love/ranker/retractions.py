"""Pure, fail-closed planning for append-only label retractions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


class RetractionRefused(RuntimeError):
    """A requested correction cannot be bound to exactly one persisted pair."""


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True).encode()).hexdigest()


def _timestamp(value: str) -> float:
    # BigQuery's durable TIMESTAMP rows arrive as ISO-8601.  SQLite stores epoch
    # floats, however, and its correction planner deliberately emits their
    # round-trip decimal form (see SQLiteStore.retraction_run_rows): formatting a
    # float as an ISO timestamp discards sub-microsecond bits, which can make a
    # tombstone fail to match the original REAL value.  Accept both canonical
    # representations here; no loose / nearest-time matching is ever allowed.
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            raise ValueError("not timestamp text")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetractionRefused("run rows contain an invalid created_at value") from exc


def _strict(rows: list[dict], kind: str) -> list[dict]:
    ordered = sorted(rows, key=lambda row: _timestamp(row["created_at"]))
    values = [_timestamp(row["created_at"]) for row in ordered]
    if any(a >= b for a, b in zip(values, values[1:])):
        raise RetractionRefused(f"{kind} rows do not have strict unique created_at ordering")
    return ordered


def make_plan(*, rows: dict, run_id: str, app: str, source: str, profile_id: str,
              reason: str, evidence_ref: str, evidence_metadata: dict | None = None) -> dict:
    """Bind one profile label to its one legacy decision, or refuse safely.

    The old decisions schema has no profile id.  We therefore require a complete, strictly
    time-ordered one-to-one label/decision sequence and pair by ordinal.  This is intentionally
    stricter than a best-effort nearest-time guess: the plan is a correction, not recovery.
    """
    if not all(isinstance(x, str) and x.strip() for x in
               (run_id, app, source, profile_id, reason, evidence_ref)):
        raise RetractionRefused("run/app/source/profile-id/reason/evidence-ref must be nonempty text")
    if rows.get("run_id") != run_id or rows.get("app") != app or rows.get("source") != source:
        raise RetractionRefused("store snapshot is not bound to the requested run/app/source")
    labels = list(rows.get("labels", []))
    decisions = list(rows.get("decisions", []))
    profiles = list(rows.get("profiles", []))
    existing = list(rows.get("retractions", []))
    if any(r.get("profile_id") == profile_id for r in existing):
        raise RetractionRefused("target profile already has an append-only retraction")
    # BigQuery supplies an archive manifest; SQLite has no historical profile archive, so an
    # empty profiles list means only that backend's documented labels-only fallback.
    if profiles and sum(r.get("profile_id") == profile_id for r in profiles) != 1:
        raise RetractionRefused("target profile archive mapping is missing or ambiguous")
    target = [r for r in labels if r.get("profile_id") == profile_id]
    if len(target) != 1:
        raise RetractionRefused("target profile must map to exactly one label")
    labels = _strict(labels, "label")
    decisions = _strict(decisions, "decision")
    if len(labels) != len(decisions):
        raise RetractionRefused("label/decision sequence cardinality differs; cannot pair legacy decision")
    label = target[0]
    ordinal = labels.index(label)
    decision = decisions[ordinal]
    if _timestamp(decision["created_at"]) < _timestamp(label["created_at"]):
        raise RetractionRefused("paired decision predates its label")
    if bool(label["liked"]) != (decision.get("decision") == "like"):
        raise RetractionRefused("paired decision outcome does not match target label")
    decision_fingerprint = canonical_sha({
        "run_id": run_id, "app": app, "source": source,
        "created_at": decision["created_at"], "decision": decision.get("decision"),
        "score": decision.get("score"),
    })
    body = {
        "schema_version": 1, "kind": "operation_love_label_retraction_plan",
        "run_id": run_id, "app": app, "source": source, "profile_id": profile_id,
        "label_created_at": label["created_at"], "decision_created_at": decision["created_at"],
        "decision_fingerprint": decision_fingerprint, "label_ordinal": ordinal,
        "reason": reason, "evidence_ref": evidence_ref,
        "snapshot_sha256": canonical_sha(rows),
    }
    if evidence_metadata is not None:
        body["evidence_metadata"] = evidence_metadata
    body["correction_id"] = canonical_sha({key: body[key] for key in (
        "run_id", "app", "source", "profile_id", "label_created_at", "decision_created_at",
        "decision_fingerprint", "reason", "evidence_ref", "snapshot_sha256")})
    body["plan_sha256"] = canonical_sha(body)
    return body


def profile_id_for_label_ordinal(rows: dict, ordinal: int) -> str:
    labels = _strict(list(rows.get("labels", [])), "label")
    if ordinal < 0 or ordinal >= len(labels):
        raise RetractionRefused("debug decision ordinal has no matching persisted label")
    profile_id = labels[ordinal].get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise RetractionRefused("ordered target label has no auditable profile_id")
    return profile_id


def verify_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("kind") != "operation_love_label_retraction_plan":
        raise RetractionRefused("invalid correction plan")
    supplied = plan.get("plan_sha256")
    actual = canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"})
    if not isinstance(supplied, str) or supplied != actual:
        raise RetractionRefused("correction plan self-hash does not verify")
    return plan


def retraction_row(plan: dict, *, created_at: str | None = None) -> dict:
    verify_plan(plan)
    return {key: plan[key] for key in (
        "correction_id", "run_id", "app", "source", "profile_id", "label_created_at",
        "decision_created_at", "decision_fingerprint", "reason", "evidence_ref")} | {
        "created_at": created_at or datetime.now(timezone.utc).isoformat()}


def is_exact_existing_retraction(rows: dict, plan: dict) -> bool:
    """True only when a previous append has exactly the correction this plan authorizes."""
    expected = {key: plan[key] for key in ("correction_id", "profile_id", "label_created_at",
                                            "decision_created_at", "decision_fingerprint")}
    for existing in rows.get("retractions", []):
        comparable = {key: existing.get(key) for key in expected}
        if comparable == expected:
            return True
        if existing.get("profile_id") == plan.get("profile_id"):
            raise RetractionRefused("target profile has a different existing retraction")
    return False
