"""Deterministic pytest partitions and strict cross-runner result reconciliation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


SUCCESS_OUTCOMES = frozenset({"passed", "skipped", "xfailed", "xpassed"})


def canonical_inventory(nodeids: list[str]) -> list[str]:
    if not nodeids or any(not isinstance(n, str) or "::" not in n for n in nodeids):
        raise ValueError("Inventory must contain pytest node IDs")
    if len(set(nodeids)) != len(nodeids):
        raise ValueError("Duplicate node ID in inventory")
    return sorted(nodeids)


def inventory_digest(inventory: list[str]) -> str:
    encoded = json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def partition(inventory: list[str], count: int) -> list[list[str]]:
    if type(count) is not int or count < 1:
        raise ValueError("Shard count must be a positive integer")
    shards: list[list[str]] = [[] for _ in range(count)]
    for nodeid in canonical_inventory(inventory):
        index = int.from_bytes(hashlib.sha256(nodeid.encode()).digest(), "big") % count
        shards[index].append(nodeid)
    return shards


def outcome_from_reports(reports: list[dict]) -> str:
    phases = {r["when"]: r for r in reports}
    if len(phases) != len(reports) or set(phases) - {"setup", "call", "teardown"}:
        raise ValueError("Duplicate or unknown pytest report phase")
    if any(r["outcome"] == "failed" for r in reports):
        return "failed"
    if not {"setup", "teardown"} <= set(phases):
        return "incomplete"
    if any(r["outcome"] not in {"passed", "skipped"} for r in reports):
        return "incomplete"
    if phases["setup"]["outcome"] == "skipped" and "call" in phases:
        return "incomplete"
    if phases["setup"]["outcome"] == "passed" and "call" not in phases:
        return "incomplete"
    skipped = next((r for r in reports if r["outcome"] == "skipped"), None)
    if skipped:
        return "xfailed" if skipped.get("wasxfail") else "skipped"
    call = phases.get("call", {})
    if call.get("outcome") != "passed" or any(
        r["outcome"] != "passed" for r in reports
    ):
        return "incomplete"
    return "xpassed" if call.get("wasxfail") else "passed"


def validate_manifest(data: dict, count: int, source_sha: str | None = None) -> None:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be an object")
    index = data.get("shard_index")
    if (
        type(data.get("version")) is not int
        or data["version"] != 1
        or type(index) is not int
        or not 0 <= index < count
    ):
        raise ValueError("Invalid manifest version or shard index")
    if type(data.get("shard_count")) is not int or data["shard_count"] != count:
        raise ValueError("Wrong shard count")
    if source_sha is not None and data.get("source_sha") != source_sha:
        raise ValueError("Manifest belongs to a different source commit")
    inventory = canonical_inventory(data["inventory"])
    if data["inventory"] != inventory or data.get(
        "inventory_digest"
    ) != inventory_digest(inventory):
        raise ValueError("Inventory is unsorted or its digest is invalid")
    selected = partition(inventory, count)[index]
    if data.get("selected_nodeids") != selected:
        raise ValueError("Selected IDs do not match the deterministic partition")
    if data.get("completed") is not True or data.get("errors") != []:
        raise ValueError("Shard did not complete successfully")
    outcomes = data["outcomes"]
    if sorted(r["nodeid"] for r in outcomes) != selected:
        raise ValueError("Missing, duplicate or unexpected test outcomes")
    for result in outcomes:
        actual = outcome_from_reports(result["reports"])
        if actual not in SUCCESS_OUTCOMES or result.get("outcome") != actual:
            raise ValueError("Failed, incomplete or inconsistent test outcome")
    counts = dict(Counter(r["outcome"] for r in outcomes))
    if data.get("outcome_counts") != counts:
        raise ValueError("Outcome counts do not match the recorded tests")
    if data.get("inventory_count") != len(inventory) or data.get(
        "selected_count"
    ) != len(selected):
        raise ValueError("Inventory or selection count is inconsistent")
    if data.get("completed_count") != len(outcomes):
        raise ValueError("Completed count is inconsistent")
    if any(
        type(data.get(field)) is not int
        for field in ("inventory_count", "selected_count", "completed_count")
    ):
        raise ValueError("Result counts must be integers")


def reconcile(manifests: list[dict], count: int, source_sha: str | None = None) -> dict:
    if type(count) is not int or count < 1:
        raise ValueError("Shard count must be a positive integer")
    if len(manifests) != count:
        raise ValueError(
            f"Expected all {count} shard manifests, received {len(manifests)}"
        )
    seen: set[int] = set()
    inventory = None
    sha = None
    executed: list[str] = []
    for manifest in manifests:
        validate_manifest(manifest, count, source_sha)
        index = manifest["shard_index"]
        if index in seen:
            raise ValueError("Duplicate shard manifest")
        seen.add(index)
        if inventory is None:
            inventory = manifest["inventory"]
            sha = manifest.get("source_sha")
        elif inventory != manifest["inventory"] or sha != manifest.get("source_sha"):
            raise ValueError(
                "Shards did not collect the same inventory and source commit"
            )
        executed.extend(manifest["selected_nodeids"])
    if seen != set(range(count)) or sorted(executed) != inventory:
        raise ValueError("Shard selections are not a disjoint, complete inventory")
    return {
        "shards": count,
        "tests": len(executed),
        "inventory_digest": inventory_digest(inventory),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
