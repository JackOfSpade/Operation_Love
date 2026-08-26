"""Pure, fail-closed planning for append-only label retractions."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import pairwise


class RetractionRefused(RuntimeError):
    """A requested correction cannot be bound to exactly one persisted pair."""


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _timestamp(value: object) -> float:
    # BigQuery's durable TIMESTAMP rows arrive as ISO-8601.  SQLite stores epoch
    # floats, however, and its correction planner deliberately emits their
    # round-trip decimal form (see SQLiteStore.retraction_run_rows): formatting a
    # float as an ISO timestamp discards sub-microsecond bits, which can make a
    # tombstone fail to match the original REAL value.  Accept both canonical
    # representations here; no loose / nearest-time matching is ever allowed.
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a timestamp")
        if isinstance(value, (int, float)):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("timestamp is not finite")
            return result
        if not isinstance(value, str):
            raise ValueError("not timestamp text")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timestamp has no timezone")
            result = parsed.timestamp()
        except ValueError:
            result = float(value)
        if not math.isfinite(result):
            raise ValueError("timestamp is not finite")
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetractionRefused("run rows contain an invalid created_at value") from exc


def _strict(rows: list[dict], kind: str) -> list[dict]:
    try:
        ordered = sorted(rows, key=lambda row: _timestamp(row["created_at"]))
        values = [_timestamp(row["created_at"]) for row in ordered]
    except (KeyError, TypeError) as exc:
        raise RetractionRefused(f"{kind} rows are malformed") from exc
    if any(a >= b for a, b in pairwise(values)):
        raise RetractionRefused(f"{kind} rows do not have strict unique created_at ordering")
    return ordered


def _snapshot_rows(snapshot: object, key: str, kind: str) -> list[dict]:
    """Copy one untrusted snapshot array after validating its container and row shapes."""
    if not isinstance(snapshot, dict):
        raise RetractionRefused("store snapshot is malformed")
    value = snapshot.get(key, [])
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RetractionRefused(f"{kind} rows are malformed")
    return list(value)


def _decision_profile_id(row: dict) -> str | None:
    value = row.get("profile_id")
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise RetractionRefused("decision row has a malformed profile_id")
    return value


def _label_liked(row: dict) -> bool:
    try:
        value = row["liked"]
    except (KeyError, TypeError) as exc:
        raise RetractionRefused("label row has no liked outcome") from exc
    if not isinstance(value, bool):
        raise RetractionRefused("label row has a malformed liked outcome")
    return value


def _decision_liked(row: dict) -> bool:
    try:
        value = row["decision"]
    except (KeyError, TypeError) as exc:
        raise RetractionRefused("decision row has no outcome") from exc
    if value not in {"like", "dislike"}:
        raise RetractionRefused("decision row has an unsupported outcome")
    return value == "like"


def _legacy_pairs(labels: list[dict], decisions: list[dict]) -> list[tuple[dict, dict]]:
    """Pair rows that predate decision ``profile_id`` without guessing.

    The two historical persistence implementations used opposite orders.  A valid legacy
    snapshot must therefore be one strictly alternating timeline in one direction for the
    whole run: ``decision, label, ...`` (current worker ordering) or ``label, decision, ...``
    (the older ordering).  Adjacent rows must also agree on outcome.  This permits both known
    schemas while refusing dropped rows, overlapping actions, and mixed-order histories that
    could shift an ordinal onto the wrong profile.
    """
    if len(labels) != len(decisions):
        raise RetractionRefused(
            "label/decision sequence cardinality differs; cannot pair legacy decision")
    pairs = list(zip(labels, decisions, strict=True))
    if any(_label_liked(label) != _decision_liked(decision)
           for label, decision in pairs):
        raise RetractionRefused("label/decision outcomes differ; cannot pair legacy decision")
    label_times = [_timestamp(label["created_at"]) for label in labels]
    decision_times = [_timestamp(decision["created_at"]) for decision in decisions]
    last = len(pairs) - 1
    decision_then_label = all(
        decision_times[index] < label_times[index]
        and (index == last or label_times[index] < decision_times[index + 1])
        for index in range(len(pairs))
    )
    label_then_decision = all(
        label_times[index] < decision_times[index]
        and (index == last or decision_times[index] < label_times[index + 1])
        for index in range(len(pairs))
    )
    if decision_then_label == label_then_decision:
        raise RetractionRefused(
            "legacy rows do not form one unambiguous adjacent causal sequence")
    return pairs


def _paired_decision(*, labels: list[dict], decisions: list[dict],
                     label: dict, profile_id: str) -> dict:
    """Prefer exact modern lineage, falling back to the strict legacy timeline."""
    linked = [(row, _decision_profile_id(row)) for row in decisions]
    matches = [row for row, linked_id in linked if linked_id == profile_id]
    if len(matches) > 1:
        raise RetractionRefused("target profile maps to multiple linked decisions")
    if matches:
        return matches[0]
    if any(linked_id is not None for _, linked_id in linked):
        raise RetractionRefused("target profile has no linked decision")
    ordinal = labels.index(label)
    return _legacy_pairs(labels, decisions)[ordinal][1]


def make_plan(*, rows: dict, run_id: str, app: str, source: str, profile_id: str,
              reason: str, evidence_ref: str, evidence_metadata: dict | None = None) -> dict:
    """Bind one profile label to exactly one causal decision, or refuse safely.

    Current worker rows share a generated profile id and bind directly.  Older decision rows
    lack that lineage, so they use :func:`_legacy_pairs`' strict adjacent-timeline invariant.
    No nearest-time guess is allowed: the plan is a correction, not recovery.
    """
    if not all(isinstance(x, str) and x.strip() for x in
               (run_id, app, source, profile_id, reason, evidence_ref)):
        raise RetractionRefused("run/app/source/profile-id/reason/evidence-ref must be nonempty text")
    if not isinstance(rows, dict):
        raise RetractionRefused("store snapshot is malformed")
    if rows.get("run_id") != run_id or rows.get("app") != app or rows.get("source") != source:
        raise RetractionRefused("store snapshot is not bound to the requested run/app/source")
    # The plan is a durable JSON document.  Reject NaN/Infinity (and other
    # non-JSON objects) before binding individual rows so a malformed snapshot
    # cannot leak a raw json encoder exception out of this fail-closed API.
    try:
        snapshot_sha = canonical_sha(rows)
        if evidence_metadata is not None:
            canonical_sha(evidence_metadata)
    except (TypeError, ValueError) as exc:
        raise RetractionRefused("store snapshot or evidence metadata is not canonical JSON data") from exc
    labels = _snapshot_rows(rows, "labels", "label")
    decisions = _snapshot_rows(rows, "decisions", "decision")
    profiles = _snapshot_rows(rows, "profiles", "profile")
    existing = _snapshot_rows(rows, "retractions", "retraction")
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
    label = target[0]
    ordinal = labels.index(label)
    decision = _paired_decision(
        labels=labels, decisions=decisions, label=label, profile_id=profile_id)
    if _label_liked(label) != _decision_liked(decision):
        raise RetractionRefused("paired decision outcome does not match target label")
    decision_identity = {
        "run_id": run_id, "app": app, "source": source,
        "created_at": decision["created_at"], "decision": decision.get("decision"),
        "score": decision.get("score"),
    }
    linked_profile_id = _decision_profile_id(decision)
    if linked_profile_id is not None:
        decision_identity["profile_id"] = linked_profile_id
    decision_fingerprint = canonical_sha(decision_identity)
    body = {
        "schema_version": 1, "kind": "operation_love_label_retraction_plan",
        "run_id": run_id, "app": app, "source": source, "profile_id": profile_id,
        "label_created_at": label["created_at"], "decision_created_at": decision["created_at"],
        "decision_fingerprint": decision_fingerprint, "label_ordinal": ordinal,
        "reason": reason, "evidence_ref": evidence_ref,
        "snapshot_sha256": snapshot_sha,
    }
    if evidence_metadata is not None:
        body["evidence_metadata"] = evidence_metadata
    body["correction_id"] = canonical_sha({key: body[key] for key in (
        "run_id", "app", "source", "profile_id", "label_created_at", "decision_created_at",
        "decision_fingerprint", "reason", "evidence_ref", "snapshot_sha256")})
    body["plan_sha256"] = canonical_sha(body)
    return body


def profile_id_for_label_ordinal(rows: dict, ordinal: int) -> str:
    labels = _strict(_snapshot_rows(rows, "labels", "label"), "label")
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
    try:
        actual = canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"})
    except (TypeError, ValueError) as exc:
        raise RetractionRefused("correction plan is not canonical JSON data") from exc
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
    for existing in _snapshot_rows(rows, "retractions", "retraction"):
        comparable = {key: existing.get(key) for key in expected}
        if comparable == expected:
            return True
        if existing.get("profile_id") == plan.get("profile_id"):
            raise RetractionRefused("target profile has a different existing retraction")
    return False
